#include "ts_additive.h"

#include "ts_glm.h"
#include "ts_internal.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

// Every product and sum below is rounded on its own, so that the compilers either language is
// built with take the same steps.
#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

namespace timesift {

namespace {

using Matrix = std::vector<double>;  // column-major throughout

// The bounds on a smoothing parameter's logarithm. Past the upper one a smooth is its linear part
// to within the precision of the fit, and past the lower one it is unpenalised to the same.
constexpr double kLogLambdaMax = 25.0;
constexpr double kLogLambdaMin = -25.0;
// The longest Newton step and the longest steepest-descent step, in log smoothing parameters.
constexpr double kMaxNewtonStep = 5.0;
constexpr double kMaxDescentStep = 2.0;
constexpr int kMaxHalvings = 30;
// A Newton step whose predicted fall in the criterion is below this share of the criterion's scale
// cannot be told from the rounding of the criterion itself, and the search has settled.
constexpr double kResolution = 1e-14;
// A free column whose norm, orthogonalised against the free columns before it, falls below this
// share of its own is spanned by them.
constexpr double kAliasTolerance = 1e-7;
// The Lanczos iteration stops once every wanted Ritz pair's residual is below this share of the
// largest Ritz value in magnitude.
constexpr double kLanczosTolerance = 1e-14;

// ---------------------------------------------------------------------------------------------
// The knot subsample: R's generator after `set.seed(1)` and R's `sample()` without replacement.

// The Mersenne Twister (Matsumoto & Nishimura 1998, ACM Transactions on Modeling and Computer
// Simulation 8:3-30) seeded as R seeds it: the seed is scrambled by fifty steps of the linear
// congruence `69069 s + 1`, the next 625 steps fill R's seed vector, whose first word is replaced
// by the position 624 and whose other 624 are the state. A uniform is the tempered output times
// 2^-32, moved off zero and one by half of 2^-32.
class SeededTwister {
 public:
  explicit SeededTwister(std::uint32_t seed) {
    for (int j = 0; j < 50; ++j) seed = 69069u * seed + 1u;
    seed = 69069u * seed + 1u;  // the position word, which the position then replaces
    for (int j = 0; j < kN; ++j) {
      seed = 69069u * seed + 1u;
      state_[j] = seed;
    }
    position_ = kN;
  }

  double uniform() {
    const double v = static_cast<double>(next()) * 2.3283064365386963e-10;
    constexpr double half = 0.5 * 2.328306437080797e-10;
    if (v <= 0.0) return half;
    if (1.0 - v <= 0.0) return 1.0 - half;
    return v;
  }

  // An integer below `n`, by rejection: draw the bits of the next power of two at or above `n`,
  // sixteen at a time from successive uniforms, and draw again until the value is below `n`.
  std::uint64_t index_below(std::uint64_t n) {
    const int bits = static_cast<int>(std::ceil(std::log2(static_cast<double>(n))));
    for (;;) {
      std::uint64_t v = 0;
      for (int b = 0; b <= bits; b += 16) {
        v = 65536u * v + static_cast<std::uint64_t>(std::floor(uniform() * 65536.0));
      }
      v &= (std::uint64_t{1} << bits) - 1u;
      if (v < n) return v;
    }
  }

 private:
  static constexpr int kN = 624;
  static constexpr int kM = 397;

  std::uint32_t next() {
    if (position_ >= kN) {
      for (int i = 0; i < kN; ++i) {
        const std::uint32_t y = (state_[i] & 0x80000000u) | (state_[(i + 1) % kN] & 0x7fffffffu);
        state_[i] = state_[(i + kM) % kN] ^ (y >> 1) ^ ((y & 1u) ? 0x9908b0dfu : 0u);
      }
      position_ = 0;
    }
    std::uint32_t y = state_[position_++];
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= y >> 18;
    return y;
  }

  std::uint32_t state_[kN];
  int position_ = kN;
};

// `size` of the indices below `n`, drawn without replacement as R's `sample()` draws them: each
// draw takes an index below the count of those left, and the last of them moves into its place.
std::vector<std::size_t> sample_without_replacement(std::size_t n, std::size_t size,
                                                    SeededTwister& g) {
  std::vector<std::size_t> pool(n);
  for (std::size_t i = 0; i < n; ++i) pool[i] = i;
  std::vector<std::size_t> out(size);
  std::size_t left = n;
  for (std::size_t i = 0; i < size; ++i) {
    const std::size_t j = static_cast<std::size_t>(g.index_below(left));
    out[i] = pool[j];
    pool[j] = pool[--left];
  }
  return out;
}

// ---------------------------------------------------------------------------------------------
// Symmetric eigenproblems.

// Turns every column of `v` [m, cols] so that its entry of largest magnitude, the first where two
// tie, is positive.
void orient_columns(Matrix& v, std::size_t m, std::size_t cols) {
  for (std::size_t c = 0; c < cols; ++c) {
    double* col = v.data() + c * m;
    std::size_t at = 0;
    for (std::size_t i = 1; i < m; ++i) {
      if (std::abs(col[i]) > std::abs(col[at])) at = i;
    }
    if (col[at] < 0.0) {
      for (std::size_t i = 0; i < m; ++i) col[i] = -col[i];
    }
  }
}

// Orders the eigenpairs by `key` of their value, largest first, the lower index first where two
// tie, and keeps the first `keep`.
template <typename Key>
void order_pairs(std::vector<double>& values, Matrix& vectors, std::size_t m, std::size_t keep,
                 Key key) {
  const std::size_t count = values.size();
  std::vector<std::size_t> order(count);
  for (std::size_t i = 0; i < count; ++i) order[i] = i;
  std::stable_sort(order.begin(), order.end(),
                   [&](std::size_t a, std::size_t b) { return key(values[a]) > key(values[b]); });
  std::vector<double> v(keep);
  Matrix u(m * keep);
  for (std::size_t j = 0; j < keep; ++j) {
    v[j] = values[order[j]];
    std::copy(vectors.begin() + static_cast<std::ptrdiff_t>(order[j] * m),
              vectors.begin() + static_cast<std::ptrdiff_t>((order[j] + 1) * m),
              u.begin() + static_cast<std::ptrdiff_t>(j * m));
  }
  values = std::move(v);
  vectors = std::move(u);
}

// Every eigenpair of the symmetric `a` [m, m] by cyclic Jacobi rotations (Golub & Van Loan,
// Matrix Computations, 4th ed., sec. 8.5.2), with Rutishauser's rule that from the fifth sweep an
// off-diagonal entry negligible beside both diagonal entries is set to zero. The eigenvalues come
// back descending, the eigenvectors as the columns of `vectors` [m, m].
void symmetric_eigen(Matrix a, std::size_t m, std::vector<double>& values, Matrix& vectors) {
  vectors.assign(m * m, 0.0);
  for (std::size_t i = 0; i < m; ++i) vectors[i * m + i] = 1.0;
  for (int sweep = 0; sweep < 100; ++sweep) {
    double off = 0.0;
    for (std::size_t q = 1; q < m; ++q) {
      for (std::size_t p = 0; p < q; ++p) off += std::abs(a[p + q * m]);
    }
    if (off == 0.0) break;
    for (std::size_t q = 1; q < m; ++q) {
      for (std::size_t p = 0; p < q; ++p) {
        const double apq = a[p + q * m];
        const double app = a[p + p * m];
        const double aqq = a[q + q * m];
        const double g = 100.0 * std::abs(apq);
        if (sweep > 3 && std::abs(app) + g == std::abs(app) && std::abs(aqq) + g == std::abs(aqq)) {
          a[p + q * m] = 0.0;
          a[q + p * m] = 0.0;
          continue;
        }
        if (apq == 0.0) continue;
        const double tau = (aqq - app) / (2.0 * apq);
        const double t = std::abs(tau) > 1e150
                             ? 0.5 / tau
                             : std::copysign(1.0, tau) / (std::abs(tau) + std::sqrt(1.0 + tau * tau));
        const double c = 1.0 / std::sqrt(1.0 + t * t);
        const double s = t * c;
        for (std::size_t k = 0; k < m; ++k) {
          const double akp = a[k + p * m];
          const double akq = a[k + q * m];
          a[k + p * m] = c * akp - s * akq;
          a[k + q * m] = s * akp + c * akq;
        }
        for (std::size_t k = 0; k < m; ++k) {
          const double apk = a[p + k * m];
          const double aqk = a[q + k * m];
          a[p + k * m] = c * apk - s * aqk;
          a[q + k * m] = s * apk + c * aqk;
        }
        a[p + q * m] = 0.0;
        a[q + p * m] = 0.0;
        for (std::size_t k = 0; k < m; ++k) {
          const double vkp = vectors[k + p * m];
          const double vkq = vectors[k + q * m];
          vectors[k + p * m] = c * vkp - s * vkq;
          vectors[k + q * m] = s * vkp + c * vkq;
        }
      }
    }
  }
  values.resize(m);
  for (std::size_t i = 0; i < m; ++i) values[i] = a[i + i * m];
  order_pairs(values, vectors, m, m, [](double v) { return v; });
  orient_columns(vectors, m, m);
}

// Every eigenpair of the symmetric tridiagonal matrix of diagonal `d` [m] and subdiagonal `e`
// [m - 1], by the implicit QL iteration with Wilkinson's shift (Bowdler, Martin, Reinsch &
// Wilkinson 1968, Numerische Mathematik 11:293-306). The eigenvalues are left in `d` and the
// eigenvectors in the columns of `z` [m, m], neither ordered.
void tridiagonal_eigen(std::vector<double>& d, std::vector<double> e, std::size_t m, Matrix& z) {
  z.assign(m * m, 0.0);
  for (std::size_t i = 0; i < m; ++i) z[i * m + i] = 1.0;
  e.resize(m, 0.0);
  e[m - 1] = 0.0;
  for (std::size_t l = 0; l < m; ++l) {
    int iterations = 0;
    for (;;) {
      std::size_t s = l;
      for (; s + 1 < m; ++s) {
        const double dd = std::abs(d[s]) + std::abs(d[s + 1]);
        if (std::abs(e[s]) <= DBL_EPSILON * dd) break;
      }
      if (s == l) break;
      if (++iterations > 60) throw Error("the tridiagonal eigenproblem did not settle.");
      double g = (d[l + 1] - d[l]) / (2.0 * e[l]);
      double r = std::hypot(g, 1.0);
      g = d[s] - d[l] + e[l] / (g + std::copysign(r, g));
      double sn = 1.0, cs = 1.0, p = 0.0;
      bool deflated = false;
      for (std::size_t i = s; i-- > l;) {
        double f = sn * e[i];
        const double b = cs * e[i];
        r = std::hypot(f, g);
        e[i + 1] = r;
        if (r == 0.0) {
          d[i + 1] -= p;
          e[s] = 0.0;
          deflated = true;
          break;
        }
        sn = f / r;
        cs = g / r;
        g = d[i + 1] - p;
        r = (d[i] - g) * sn + 2.0 * cs * b;
        p = sn * r;
        d[i + 1] = g + p;
        g = cs * r - b;
        double* zi = z.data() + i * m;
        double* zj = z.data() + (i + 1) * m;
        for (std::size_t k = 0; k < m; ++k) {
          f = zj[k];
          zj[k] = sn * zi[k] + cs * f;
          zi[k] = cs * zi[k] - sn * f;
        }
      }
      if (deflated) continue;
      d[l] -= p;
      e[l] = g;
      e[s] = 0.0;
    }
  }
}

// A fixed sequence of numbers in [-0.5, 0.5), SplitMix32 from a counter, for the Lanczos
// iteration's starting vectors.
class StartSequence {
 public:
  double next() {
    counter_ += 0x9E3779B9u;
    std::uint32_t z = (counter_ ^ (counter_ >> 16)) * 0x85EBCA6Bu;
    z = (z ^ (z >> 13)) * 0xC2B2AE35u;
    z ^= z >> 16;
    return static_cast<double>(z) * 2.3283064365386963e-10 - 0.5;
  }

 private:
  std::uint32_t counter_ = 0;
};

// The inner product of `a` and `b`, over four running sums, so that the additions do not wait on
// one another.
double dot(const double* a, const double* b, std::size_t len) {
  double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
  std::size_t i = 0;
  for (; i + 4 <= len; i += 4) {
    s0 += a[i] * b[i];
    s1 += a[i + 1] * b[i + 1];
    s2 += a[i + 2] * b[i + 2];
    s3 += a[i + 3] * b[i + 3];
  }
  for (; i < len; ++i) s0 += a[i] * b[i];
  return (s0 + s1) + (s2 + s3);
}

// `v` [m] orthogonalised twice against the first `count` columns of `q` [m, ...], the
// classical Gram-Schmidt step repeated, which is enough to keep a Krylov basis orthogonal to
// working precision (Parlett 1998, The Symmetric Eigenvalue Problem, sec. 6.9).
void orthogonalise(double* v, const Matrix& q, std::size_t m, std::size_t count) {
  for (int pass = 0; pass < 2; ++pass) {
    for (std::size_t j = 0; j < count; ++j) {
      const double* qj = q.data() + j * m;
      const double t = dot(qj, v, m);
      for (std::size_t i = 0; i < m; ++i) v[i] -= t * qj[i];
    }
  }
}

// The `k` eigenpairs of greatest magnitude of the symmetric matrix `apply` multiplies by, of
// order `m`, by the Lanczos iteration with full reorthogonalisation (Golub & Van Loan sec. 10.1).
// Where the Krylov space closes before the pairs have settled, the iteration continues from a new
// starting vector orthogonal to it. The eigenvalues come back ordered by magnitude, largest first,
// and each eigenvector is turned as `orient_columns` turns it.
template <typename Apply>
void extreme_eigen(Apply apply, std::size_t m, std::size_t k, std::vector<double>& values,
                   Matrix& vectors) {
  StartSequence start;
  Matrix q(m);
  std::vector<double> alpha, beta, w(m);
  auto fresh = [&](std::size_t count, double* v) {
    for (int attempt = 0; attempt < 4; ++attempt) {
      for (std::size_t i = 0; i < m; ++i) v[i] = start.next();
      orthogonalise(v, q, m, count);
      const double norm = std::sqrt(dot(v, v, m));
      if (norm > 1e-8) {
        for (std::size_t i = 0; i < m; ++i) v[i] /= norm;
        return true;
      }
    }
    return false;
  };
  fresh(0, q.data());
  double scale = 0.0;
  for (std::size_t j = 0; j < m; ++j) {
    const double* qj = q.data() + j * m;
    apply(qj, w.data());
    const double a = dot(qj, w.data(), m);
    alpha.push_back(a);
    orthogonalise(w.data(), q, m, j + 1);
    const double b = std::sqrt(dot(w.data(), w.data(), m));
    scale = std::max(scale, std::abs(a) + b);
    const std::size_t steps = j + 1;
    const bool last = steps == m;
    if (last || (steps >= k && (steps - k) % 5 == 0)) {
      std::vector<double> theta = alpha;
      Matrix s;
      tridiagonal_eigen(theta, beta, steps, s);
      order_pairs(theta, s, steps, k, [](double v) { return std::abs(v); });
      bool settled = last;
      if (!settled) {
        settled = true;
        const double top = std::abs(theta[0]);
        for (std::size_t i = 0; i < k && settled; ++i) {
          settled = b * std::abs(s[(steps - 1) + i * steps]) <= kLanczosTolerance * top;
        }
      }
      if (settled) {
        values = theta;
        vectors.assign(m * k, 0.0);
        for (std::size_t i = 0; i < k; ++i) {
          double* out = vectors.data() + i * m;
          for (std::size_t t = 0; t < steps; ++t) {
            const double c = s[t + i * steps];
            const double* qt = q.data() + t * m;
            for (std::size_t r = 0; r < m; ++r) out[r] += c * qt[r];
          }
          const double norm = std::sqrt(dot(out, out, m));
          for (std::size_t r = 0; r < m; ++r) out[r] /= norm;
        }
        orient_columns(vectors, m, k);
        return;
      }
    }
    q.resize((j + 2) * m);
    double* next = q.data() + (j + 1) * m;
    if (b > 1e-13 * scale) {
      beta.push_back(b);
      for (std::size_t i = 0; i < m; ++i) next[i] = w[i] / b;
    } else {
      beta.push_back(0.0);
      if (!fresh(j + 1, next)) throw Error("the Lanczos iteration found no further direction.");
    }
  }
}

// ---------------------------------------------------------------------------------------------
// The terms.

double radial(double r) {
  const double a = std::abs(r);
  return a * a * a / 12.0;
}

struct Term {
  std::int32_t column = 0;
  std::size_t basis = 0;
  std::size_t size = 0;
  std::size_t penalised = 0;
  double shift = 0.0;
  std::vector<double> knots;
  Matrix radial;               // [knots, basis - 2]
  Matrix map;                  // [basis, size]
  std::vector<double> penalty; // [penalised]
};

// The raw basis of `term` at `x` into `out` [basis].
void raw_row(const Term& term, double x, double* out) {
  const std::size_t nr = term.basis - 2;
  const std::size_t nk = term.knots.size();
  std::fill(out, out + nr, 0.0);
  for (std::size_t i = 0; i < nk; ++i) {
    const double e = radial(x - term.knots[i]);
    if (e == 0.0) continue;
    for (std::size_t c = 0; c < nr; ++c) out[c] += e * term.radial[i + c * nk];
  }
  out[nr] = 1.0;
  out[nr + 1] = x - term.shift;
}

// `term`'s model columns at every row of `x` [n], into `out` [n, size] (leading dimension `ld`).
void term_columns(const Term& term, const double* x, std::size_t n, double* out, std::size_t ld) {
  std::vector<double> raw(term.basis);
  for (std::size_t i = 0; i < n; ++i) {
    raw_row(term, x[i], raw.data());
    for (std::size_t c = 0; c < term.size; ++c) {
      const double* mc = term.map.data() + c * term.basis;
      out[i + c * ld] = dot(raw.data(), mc, term.basis);
    }
  }
}

// The orthonormal complement, [m, m - cols], of the columns of `c` [m, cols] (of full rank), as
// the last `m - cols` columns of Q in a Householder QR of `c`.
Matrix complement(Matrix c, std::size_t m, std::size_t cols) {
  std::size_t rank = 0;
  std::vector<double> qraux;
  std::vector<std::size_t> jpvt;
  detail::householder_qr(c.data(), m, cols, 1e-12, rank, qraux, jpvt);
  if (rank < cols) throw Error("a smooth's constraints are dependent.");
  Matrix z(m * (m - cols));
  for (std::size_t j = cols; j < m; ++j) {
    double* col = z.data() + (j - cols) * m;
    col[j] = 1.0;
    detail::apply_q(c.data(), m, cols, qraux.data(), col);
  }
  return z;
}

// `a` [r, s] times `b` [s, t] into [r, t].
Matrix multiply(const Matrix& a, const Matrix& b, std::size_t r, std::size_t s, std::size_t t) {
  Matrix out(r * t, 0.0);
  for (std::size_t j = 0; j < t; ++j) {
    double* oj = out.data() + j * r;
    for (std::size_t l = 0; l < s; ++l) {
      const double blj = b[l + j * s];
      if (blj == 0.0) continue;
      const double* al = a.data() + l * r;
      for (std::size_t i = 0; i < r; ++i) oj[i] += al[i] * blj;
    }
  }
  return out;
}

// `a'` [s, r] times `b` [r, t] into [s, t].
Matrix cross(const Matrix& a, const Matrix& b, std::size_t r, std::size_t s, std::size_t t) {
  Matrix out(s * t);
  for (std::size_t j = 0; j < t; ++j) {
    for (std::size_t i = 0; i < s; ++i) out[i + j * s] = dot(a.data() + i * r, b.data() + j * r, r);
  }
  return out;
}

// One column's term: a thin plate regression spline over its distinct values, a linear term where
// it holds two, and none where it holds one.
bool build_term(const double* x, std::size_t n, std::int32_t column, const AdditiveSpec& spec,
                Term& term) {
  std::vector<double> u(x, x + n);
  std::sort(u.begin(), u.end());
  u.erase(std::unique(u.begin(), u.end()), u.end());
  const std::size_t distinct = u.size();
  if (distinct < 2) return false;
  term.column = column;
  double shift = 0.0;
  for (std::size_t i = 0; i < n; ++i) shift += x[i];
  term.shift = shift / static_cast<double>(n);

  if (distinct == 2) {
    double ss = 0.0;
    for (std::size_t i = 0; i < n; ++i) ss += (x[i] - term.shift) * (x[i] - term.shift);
    term.basis = 2;
    term.size = 1;
    term.penalised = 0;
    term.map = {0.0, 1.0 / std::sqrt(ss / static_cast<double>(n))};
    return true;
  }

  const std::size_t k = std::min<std::size_t>(static_cast<std::size_t>(spec.k), distinct);
  if (distinct > static_cast<std::size_t>(spec.max_knots)) {
    SeededTwister g(1u);
    const std::vector<std::size_t> pick =
        sample_without_replacement(distinct, static_cast<std::size_t>(spec.max_knots), g);
    term.knots.resize(pick.size());
    for (std::size_t i = 0; i < pick.size(); ++i) term.knots[i] = u[pick[i]];
    std::sort(term.knots.begin(), term.knots.end());
  } else {
    term.knots = u;
  }
  const std::vector<double>& kn = term.knots;
  const std::size_t nk = kn.size();

  // The radial functions among the knots, reduced to the `k` directions of greatest magnitude.
  std::vector<double> ev;
  Matrix uk;
  extreme_eigen(
      [&](const double* v, double* out) {
        for (std::size_t i = 0; i < nk; ++i) {
          double s = 0.0;
          for (std::size_t j = 0; j < nk; ++j) s += radial(kn[i] - kn[j]) * v[j];
          out[i] = s;
        }
      },
      nk, k, ev, uk);

  // The side condition T' delta = 0 on the radial coefficients (Wood 2003, sec. 2.1), with T the
  // constant and the shifted value at every knot, taken in the reduced directions.
  Matrix t(nk * 2);
  for (std::size_t i = 0; i < nk; ++i) {
    t[i] = 1.0;
    t[i + nk] = kn[i] - term.shift;
  }
  const Matrix ut = cross(uk, t, nk, k, 2);  // [k, 2]
  const Matrix z = complement(ut, k, 2);     // [k, k - 2]
  const std::size_t nr = k - 2;
  term.radial = multiply(uk, z, nk, k, nr);
  term.basis = k;

  // The penalty on the raw basis, Z' D Z on the radial columns.
  Matrix s(k * k, 0.0);
  for (std::size_t b = 0; b < nr; ++b) {
    for (std::size_t a = 0; a < nr; ++a) {
      double v = 0.0;
      for (std::size_t l = 0; l < k; ++l) v += z[l + a * k] * ev[l] * z[l + b * k];
      s[a + b * k] = v;
    }
  }

  // The raw basis at the units, every column scaled to a root mean square of one.
  term.map.assign(k * k, 0.0);
  for (std::size_t c = 0; c < k; ++c) term.map[c + c * k] = 1.0;
  term.size = k;
  Matrix raw(n * k);
  term_columns(term, x, n, raw.data(), n);
  std::vector<double> scale(k);
  for (std::size_t c = 0; c < k; ++c) {
    const double* col = raw.data() + c * n;
    scale[c] = std::sqrt(dot(col, col, n) / static_cast<double>(n));
    for (std::size_t i = 0; i < n; ++i) raw[i + c * n] /= scale[c];
  }
  for (std::size_t b = 0; b < k; ++b) {
    for (std::size_t a = 0; a < k; ++a) s[a + b * k] /= scale[a] * scale[b];
  }

  // The penalty brought to the scale of the design: its largest absolute column sum made the
  // square of the design's largest absolute row sum.
  double row_max = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    double sum = 0.0;
    for (std::size_t c = 0; c < k; ++c) sum += std::abs(raw[i + c * n]);
    row_max = std::max(row_max, sum);
  }
  double col_max = 0.0;
  for (std::size_t b = 0; b < k; ++b) {
    double sum = 0.0;
    for (std::size_t a = 0; a < k; ++a) sum += std::abs(s[a + b * k]);
    col_max = std::max(col_max, sum);
  }
  const double factor = row_max * row_max / col_max;
  for (double& v : s) v *= factor;

  // The smooth made to sum to zero over the units, then turned onto its penalty's eigenvectors.
  Matrix sums(k);
  for (std::size_t c = 0; c < k; ++c) {
    double v = 0.0;
    for (std::size_t i = 0; i < n; ++i) v += raw[i + c * n];
    sums[c] = v;
  }
  const Matrix zc = complement(sums, k, 1);  // [k, k - 1]
  const std::size_t size = k - 1;
  const Matrix sc = multiply(cross(zc, s, k, size, k), zc, size, k, size);
  std::vector<double> pv;
  Matrix pvec;
  symmetric_eigen(sc, size, pv, pvec);
  const Matrix turned = multiply(zc, pvec, k, size, size);
  term.map.assign(k * size, 0.0);
  for (std::size_t c = 0; c < size; ++c) {
    for (std::size_t a = 0; a < k; ++a) term.map[a + c * k] = turned[a + c * k] / scale[a];
  }
  term.size = size;
  term.penalised = size - 1;
  term.penalty.assign(pv.begin(), pv.begin() + static_cast<std::ptrdiff_t>(size - 1));
  for (double v : term.penalty) {
    if (!(v > 0.0)) throw Error("a smooth's penalty is not positive definite on its range.");
  }
  return true;
}

// ---------------------------------------------------------------------------------------------
// The model the responses share.

struct Design {
  std::size_t n = 0;
  std::size_t q = 0;                     // columns kept
  Matrix x;                              // [n, q]
  std::vector<double> d;                 // [q] the penalty's diagonal, zero on a free column
  std::vector<int> smooth;               // [q] the smooth a penalised column belongs to, or -1
  std::vector<std::vector<std::size_t>> columns;  // the penalised columns of each smooth
  std::vector<int> term;                 // [q] the term a column belongs to, -1 the intercept
  std::vector<std::size_t> coefficient;  // [q] its index among all coefficients
  std::size_t smooths = 0;
};

// ---------------------------------------------------------------------------------------------
// The families. A binomial and a Poisson response each run under their canonical link, so the
// working weight is the variance function times the case weight, and the scale is known: the
// criterion is the unbiased risk estimator and the penalised fit is iterated. A Gaussian response
// is one least-squares solve and its scale is estimated, which is the generalised cross-validation
// score.

// The mean the first working response is built from: the response pulled half a trial towards one
// half under the binomial family, and a tenth above zero under the Poisson one.
double starting_mean(Family family, double y, double w) {
  switch (family) {
    case Family::binomial: return (w * y + 0.5) / (w + 1.0);
    case Family::poisson: return y + 0.1;
    case Family::gaussian: break;
  }
  return y;
}

// The linear predictor at a mean, the canonical link's.
double canonical_link(Family family, double mu) {
  switch (family) {
    case Family::binomial: return std::log(mu / (1.0 - mu));
    case Family::poisson: return std::log(mu);
    case Family::gaussian: break;
  }
  return mu;
}

// The working weight of one case at its mean.
double working_weight_of(Family family, double w, double mu) {
  switch (family) {
    case Family::binomial: return w * mu * (1.0 - mu);
    case Family::poisson: return w * mu;
    case Family::gaussian: break;
  }
  return w;
}

// ---------------------------------------------------------------------------------------------
// The fit at given smoothing parameters.

class Response {
 public:
  // `workers` runs the derivatives' independent pieces, one per smooth and one per pair of
  // smooths, at once; each writes its own slot.
  Response(const Design& design, const double* y, const double* w, Family family, double gamma,
           const AdditiveSpec& spec, int workers)
      : g_(design), y_(y), w_(w), family_(family), gamma_(gamma), spec_(spec),
        workers_(workers) {}

  struct State {
    std::vector<double> beta, eta, mu;
    double deviance = 0.0;
    double penalised = 0.0;
    Matrix hinv;           // [q, q] the inverse of X'WX + S
    double tau = 0.0;      // the effective degrees of freedom
    double score = 0.0;
    bool settled = true;   // the inner fit met `epsilon`
  };

  // A response whose scale is known is fitted by iterating the penalised least squares, and its
  // smoothing parameters are chosen by the unbiased risk estimator.
  bool known_scale() const { return family_ != Family::gaussian; }

  double mean(double eta) const { return linkinv(family_, eta); }

  double deviance(const std::vector<double>& mu) const {
    double dev = 0.0;
    for (std::size_t i = 0; i < g_.n; ++i) {
      const double y = y_[i];
      if (family_ == Family::binomial) {
        double t = 0.0;
        if (y > 0.0) t += y * std::log(y / mu[i]);
        if (y < 1.0) t += (1.0 - y) * std::log((1.0 - y) / (1.0 - mu[i]));
        dev += 2.0 * w_[i] * t;
      } else if (family_ == Family::poisson) {
        dev += 2.0 * w_[i] * ((y > 0.0 ? y * std::log(y / mu[i]) : 0.0) - (y - mu[i]));
      } else {
        const double r = y - mu[i];
        dev += w_[i] * r * r;
      }
    }
    return dev;
  }

  double penalty_of(const std::vector<double>& beta, const std::vector<double>& lambda) const {
    double pen = 0.0;
    for (std::size_t c = 0; c < g_.q; ++c) {
      if (g_.smooth[c] >= 0) pen += lambda[static_cast<std::size_t>(g_.smooth[c])] * g_.d[c] *
                                    beta[c] * beta[c];
    }
    return pen;
  }

  void predictor(const std::vector<double>& beta, std::vector<double>& eta,
                 std::vector<double>& mu) const {
    eta.assign(g_.n, 0.0);
    for (std::size_t c = 0; c < g_.q; ++c) {
      const double b = beta[c];
      if (b == 0.0) continue;
      const double* xc = g_.x.data() + c * g_.n;
      for (std::size_t i = 0; i < g_.n; ++i) eta[i] += b * xc[i];
    }
    mu.resize(g_.n);
    for (std::size_t i = 0; i < g_.n; ++i) mu[i] = mean(eta[i]);
  }

  double working_weight(std::size_t i, double mu) const {
    return working_weight_of(family_, w_[i], mu);
  }

  // The working response of a case: the linear predictor plus the residual over the mean's
  // derivative, which under the canonical link is the variance function; the response itself under
  // the Gaussian family.
  double working_response(std::size_t i, double eta, double mu) const {
    switch (family_) {
      case Family::binomial: return eta + (y_[i] - mu) / (mu * (1.0 - mu));
      case Family::poisson: return eta + (y_[i] - mu) / mu;
      case Family::gaussian: break;
    }
    return y_[i];
  }

  // The penalised weighted least-squares step at the means `mu`: the coefficients, and in `r`
  // [q, q] the triangle of the QR of the weighted design stacked on the root of the penalty.
  std::vector<double> solve(const std::vector<double>& eta, const std::vector<double>& mu,
                            const std::vector<double>& lambda, Matrix& r) const {
    const std::size_t n = g_.n, q = g_.q, rows = n + q;
    Matrix a(rows * q, 0.0);
    std::vector<double> b(rows, 0.0);
    std::vector<double> root(n);
    for (std::size_t i = 0; i < n; ++i) {
      const double wi = working_weight(i, mu[i]);
      root[i] = std::sqrt(wi);
      b[i] = root[i] * working_response(i, eta[i], mu[i]);
    }
    for (std::size_t c = 0; c < q; ++c) {
      const double* xc = g_.x.data() + c * n;
      double* ac = a.data() + c * rows;
      for (std::size_t i = 0; i < n; ++i) ac[i] = root[i] * xc[i];
      if (g_.smooth[c] >= 0) {
        ac[n + c] = std::sqrt(lambda[static_cast<std::size_t>(g_.smooth[c])] * g_.d[c]);
      }
    }
    std::size_t rank = 0;
    std::vector<double> qraux;
    std::vector<std::size_t> jpvt;
    detail::householder_qr(a.data(), rows, q, 1e-13, rank, qraux, jpvt);
    if (rank < q) {
      throw Error("the additive model is not identifiable at the smoothing parameters reached.");
    }
    detail::apply_qt(a.data(), rows, q, qraux.data(), b.data());
    if (!detail::back_substitute(a.data(), rows, q, b.data())) {
      throw Error("the additive model's penalised least squares is singular.");
    }
    r.assign(q * q, 0.0);
    for (std::size_t c = 0; c < q; ++c) {
      for (std::size_t i = 0; i <= c; ++i) r[i + c * q] = a[i + c * rows];
    }
    return std::vector<double>(b.begin(), b.begin() + static_cast<std::ptrdiff_t>(q));
  }

  // Penalised iteratively reweighted least squares from `start`, or from the family's starting
  // means where it is empty. A step that raises the penalised deviance is halved back towards the
  // coefficients it left.
  State fit(const std::vector<double>& lambda, const std::vector<double>& start) const {
    const std::size_t n = g_.n;
    State st;
    Matrix r;
    std::vector<double> eta(n), mu(n);
    double previous = std::numeric_limits<double>::infinity();
    std::vector<double> beta;
    if (start.empty()) {
      for (std::size_t i = 0; i < n; ++i) {
        mu[i] = starting_mean(family_, y_[i], w_[i]);
        eta[i] = canonical_link(family_, mu[i]);
      }
    } else {
      beta = start;
      predictor(beta, eta, mu);
      previous = deviance(mu) + penalty_of(beta, lambda);
    }
    st.settled = false;
    const int steps = known_scale() ? spec_.max_irls : 1;
    for (int it = 0; it < steps; ++it) {
      std::vector<double> next = solve(eta, mu, lambda, r);
      std::vector<double> next_eta, next_mu;
      predictor(next, next_eta, next_mu);
      double pdev = deviance(next_mu) + penalty_of(next, lambda);
      if (!beta.empty()) {
        for (int h = 0; h < kMaxHalvings && !(pdev <= previous) ; ++h) {
          for (std::size_t c = 0; c < g_.q; ++c) next[c] = 0.5 * (next[c] + beta[c]);
          predictor(next, next_eta, next_mu);
          pdev = deviance(next_mu) + penalty_of(next, lambda);
        }
      }
      const bool small = std::abs(pdev - previous) <= spec_.epsilon * (std::abs(pdev) + 0.1);
      beta = std::move(next);
      eta = std::move(next_eta);
      mu = std::move(next_mu);
      previous = pdev;
      if (!known_scale() || small) {
        st.settled = true;
        break;
      }
    }
    if (known_scale()) solve(eta, mu, lambda, r);  // the triangle at the coefficients reached
    st.beta = std::move(beta);
    st.eta = std::move(eta);
    st.mu = std::move(mu);
    st.deviance = deviance(st.mu);
    st.penalised = previous;
    st.hinv = inverse_from_triangle(r, g_.q);
    // tr((X'WX + S)^-1 X'WX) = q - tr((X'WX + S)^-1 S), the penalty being diagonal.
    double tau = static_cast<double>(g_.q);
    for (std::size_t c = 0; c < g_.q; ++c) {
      if (g_.smooth[c] >= 0) {
        tau -= lambda[static_cast<std::size_t>(g_.smooth[c])] * g_.d[c] * st.hinv[c + c * g_.q];
      }
    }
    st.tau = tau;
    st.score = criterion(st.deviance, tau);
    return st;
  }

  // The unbiased risk estimator `D / n + 2 gamma tau / n - 1` under the binomial and the Poisson
  // family, whose scale is one, and the generalised cross-validation score
  // `n D / (n - gamma tau)^2` under the Gaussian one.
  double criterion(double dev, double tau) const {
    const double n = static_cast<double>(g_.n);
    if (known_scale()) return dev / n + 2.0 * gamma_ * tau / n - 1.0;
    const double delta = n - gamma_ * tau;
    return n * dev / (delta * delta);
  }

  // The criterion's gradient `g` [M] and Hessian `h` [M, M] in the log smoothing parameters at
  // the fit `st`, by implicit differentiation of the score equations X'(w (y - mu)) = S beta.
  void derivatives(const State& st, const std::vector<double>& lambda, std::vector<double>& g,
                   Matrix& h) const {
    const std::size_t n = g_.n, q = g_.q, m = g_.smooths;
    const Matrix& hi = st.hinv;
    // The working weight, and its first and second derivatives in the linear predictor.
    std::vector<double> wt(n), c1(n, 0.0), c2(n, 0.0), res(n);
    for (std::size_t i = 0; i < n; ++i) {
      const double mu = st.mu[i];
      wt[i] = working_weight(i, mu);
      if (family_ == Family::binomial) {
        const double v = mu * (1.0 - mu);
        c1[i] = w_[i] * v * (1.0 - 2.0 * mu);
        c2[i] = w_[i] * v * (1.0 - 6.0 * mu + 6.0 * mu * mu);
      } else if (family_ == Family::poisson) {
        c1[i] = w_[i] * mu;
        c2[i] = w_[i] * mu;
      }
      res[i] = w_[i] * (y_[i] - mu);
    }
    std::vector<double> pen(q, 0.0);
    for (std::size_t c = 0; c < q; ++c) {
      if (g_.smooth[c] >= 0) pen[c] = lambda[static_cast<std::size_t>(g_.smooth[c])] * g_.d[c];
    }
    // N = H^-1 X'WX = I - H^-1 S, and N H^-1.
    Matrix nm(q * q);
    for (std::size_t b = 0; b < q; ++b) {
      for (std::size_t a = 0; a < q; ++a) {
        nm[a + b * q] = (a == b ? 1.0 : 0.0) - hi[a + b * q] * pen[b];
      }
    }
    const Matrix nhi = multiply(nm, hi, q, q, q);
    // diag(X H^-1 X') and diag(X N H^-1 X').
    const Matrix xh = multiply(g_.x, hi, n, q, q);
    const Matrix xnh = multiply(g_.x, nhi, n, q, q);
    std::vector<double> da(n, 0.0), db(n, 0.0);
    for (std::size_t c = 0; c < q; ++c) {
      const double* xc = g_.x.data() + c * n;
      const double* hc = xh.data() + c * n;
      const double* nc = xnh.data() + c * n;
      for (std::size_t i = 0; i < n; ++i) {
        da[i] += hc[i] * xc[i];
        db[i] += nc[i] * xc[i];
      }
    }

    std::vector<std::vector<double>> bj(m), ej(m);
    std::vector<Matrix> lj(m), lnj(m), qj(m);
    std::vector<double> dj(m), tj(m);
    detail::run_tasks(m, workers_, [&](std::size_t j) {
      std::vector<double> sb(q, 0.0);
      for (std::size_t c : g_.columns[j]) sb[c] = g_.d[c] * st.beta[c];
      bj[j].assign(q, 0.0);
      for (std::size_t b = 0; b < q; ++b) {
        if (sb[b] == 0.0) continue;
        for (std::size_t a = 0; a < q; ++a) bj[j][a] -= lambda[j] * hi[a + b * q] * sb[b];
      }
      ej[j] = times(bj[j]);
      // L_j = H^-1 H_j, H_j = X' diag(c1 eta_j) X + lambda_j S_j, and Q_j its first part.
      double trg = 0.0;
      if (known_scale()) {
        std::vector<double> v(n);
        for (std::size_t i = 0; i < n; ++i) v[i] = c1[i] * ej[j][i];
        qj[j] = multiply(hi, weighted_cross(v), q, q, q);
        for (std::size_t i = 0; i < n; ++i) trg += v[i] * da[i];
      } else {
        qj[j].assign(q * q, 0.0);
      }
      lj[j] = qj[j];
      for (std::size_t c : g_.columns[j]) {
        const double s = lambda[j] * g_.d[c];
        for (std::size_t a = 0; a < q; ++a) lj[j][a + c * q] += hi[a + c * q] * s;
      }
      lnj[j] = multiply(lj[j], nm, q, q, q);
      double trln = 0.0;
      for (std::size_t a = 0; a < q; ++a) trln += lnj[j][a + a * q];
      dj[j] = -2.0 * dot(res.data(), ej[j].data(), n);
      tj[j] = -trln + trg;
    });

    Matrix dh(m * m), th(m * m);
    std::vector<std::pair<std::size_t, std::size_t>> pairs;
    for (std::size_t j = 0; j < m; ++j) {
      for (std::size_t k = j; k < m; ++k) pairs.emplace_back(j, k);
    }
    detail::run_tasks(pairs.size(), workers_, [&](std::size_t at) {
      const std::size_t j = pairs[at].first, k = pairs[at].second;
      // The second derivative of the coefficients.
      std::vector<double> rhs(q, 0.0);
      std::vector<double> v(n);
      for (std::size_t i = 0; i < n; ++i) v[i] = c1[i] * ej[j][i] * ej[k][i];
      if (known_scale()) {
        for (std::size_t c = 0; c < q; ++c) rhs[c] = dot(g_.x.data() + c * n, v.data(), n);
      }
      for (std::size_t c : g_.columns[k]) rhs[c] += lambda[k] * g_.d[c] * bj[j][c];
      for (std::size_t c : g_.columns[j]) rhs[c] += lambda[j] * g_.d[c] * bj[k][c];
      std::vector<double> bjk(q, 0.0);
      for (std::size_t b = 0; b < q; ++b) {
        if (rhs[b] == 0.0) continue;
        for (std::size_t a = 0; a < q; ++a) bjk[a] -= hi[a + b * q] * rhs[b];
      }
      if (j == k) {
        for (std::size_t a = 0; a < q; ++a) bjk[a] += bj[j][a];
      }
      const std::vector<double> ejk = times(bjk);
      double dd = 0.0, ua = 0.0, ub = 0.0;
      for (std::size_t i = 0; i < n; ++i) {
        dd += 2.0 * ej[j][i] * wt[i] * ej[k][i] - 2.0 * res[i] * ejk[i];
        const double u = c2[i] * ej[j][i] * ej[k][i] + c1[i] * ejk[i];
        ua += u * da[i];
        ub += u * db[i];
      }
      double t = trace_product(lj[k], lnj[j], q) + trace_product(lj[j], lnj[k], q) - ub + ua;
      if (known_scale()) t -= trace_product(lj[j], qj[k], q) + trace_product(lj[k], qj[j], q);
      if (j == k) {
        double s = 0.0;
        for (std::size_t c : g_.columns[j]) s += g_.d[c] * nhi[c + c * q];
        t -= lambda[j] * s;
      }
      dh[j + k * m] = dh[k + j * m] = dd;
      th[j + k * m] = th[k + j * m] = t;
    });

    const double nn = static_cast<double>(n);
    g.assign(m, 0.0);
    h.assign(m * m, 0.0);
    if (known_scale()) {
      for (std::size_t j = 0; j < m; ++j) g[j] = dj[j] / nn + 2.0 * gamma_ * tj[j] / nn;
      for (std::size_t a = 0; a < m * m; ++a) h[a] = dh[a] / nn + 2.0 * gamma_ * th[a] / nn;
    } else {
      const double delta = nn - gamma_ * st.tau;
      const double dev = st.deviance;
      const double d2 = delta * delta, d3 = d2 * delta, d4 = d3 * delta;
      for (std::size_t j = 0; j < m; ++j) {
        g[j] = nn * dj[j] / d2 + 2.0 * nn * gamma_ * dev * tj[j] / d3;
      }
      for (std::size_t k = 0; k < m; ++k) {
        for (std::size_t j = 0; j < m; ++j) {
          h[j + k * m] = nn * dh[j + k * m] / d2 +
                         2.0 * nn * gamma_ * (dj[j] * tj[k] + dj[k] * tj[j]) / d3 +
                         2.0 * nn * gamma_ * dev * th[j + k * m] / d3 +
                         6.0 * nn * gamma_ * gamma_ * dev * tj[j] * tj[k] / d4;
        }
      }
    }
  }

  // The effective degrees of freedom of every column, 1 - (H^-1 S)_cc.
  std::vector<double> column_edf(const State& st, const std::vector<double>& lambda) const {
    std::vector<double> edf(g_.q, 1.0);
    for (std::size_t c = 0; c < g_.q; ++c) {
      if (g_.smooth[c] >= 0) {
        edf[c] -= lambda[static_cast<std::size_t>(g_.smooth[c])] * g_.d[c] * st.hinv[c + c * g_.q];
      }
    }
    return edf;
  }

 private:
  std::vector<double> times(const std::vector<double>& b) const {
    std::vector<double> out(g_.n, 0.0);
    for (std::size_t c = 0; c < g_.q; ++c) {
      if (b[c] == 0.0) continue;
      const double* xc = g_.x.data() + c * g_.n;
      for (std::size_t i = 0; i < g_.n; ++i) out[i] += b[c] * xc[i];
    }
    return out;
  }

  // X' diag(v) X.
  Matrix weighted_cross(const std::vector<double>& v) const {
    const std::size_t n = g_.n, q = g_.q;
    Matrix vx(n * q);
    for (std::size_t b = 0; b < q; ++b) {
      const double* xb = g_.x.data() + b * n;
      double* ob = vx.data() + b * n;
      for (std::size_t i = 0; i < n; ++i) ob[i] = v[i] * xb[i];
    }
    Matrix out(q * q);
    for (std::size_t b = 0; b < q; ++b) {
      for (std::size_t a = 0; a <= b; ++a) {
        out[a + b * q] = out[b + a * q] = dot(g_.x.data() + a * n, vx.data() + b * n, n);
      }
    }
    return out;
  }

  static double trace_product(const Matrix& a, const Matrix& b, std::size_t q) {
    double t = 0.0;
    for (std::size_t j = 0; j < q; ++j) {
      for (std::size_t i = 0; i < q; ++i) t += a[i + j * q] * b[j + i * q];
    }
    return t;
  }

  // (R'R)^-1 from the upper triangle `r` [q, q].
  static Matrix inverse_from_triangle(const Matrix& r, std::size_t q) {
    Matrix ri(q * q, 0.0);
    for (std::size_t j = 0; j < q; ++j) {
      ri[j + j * q] = 1.0 / r[j + j * q];
      for (std::size_t i = j; i-- > 0;) {
        double s = 0.0;
        for (std::size_t l = i + 1; l <= j; ++l) s += r[i + l * q] * ri[l + j * q];
        ri[i + j * q] = -s / r[i + i * q];
      }
    }
    Matrix out(q * q);
    for (std::size_t b = 0; b < q; ++b) {
      for (std::size_t a = 0; a <= b; ++a) {
        double s = 0.0;
        for (std::size_t l = b; l < q; ++l) s += ri[a + l * q] * ri[b + l * q];
        out[a + b * q] = out[b + a * q] = s;
      }
    }
    return out;
  }

  const Design& g_;
  const double* y_;
  const double* w_;
  Family family_;
  double gamma_;
  const AdditiveSpec& spec_;
  int workers_;
};

struct ResponseFit {
  std::vector<double> beta, lambda, edf;
  double score = 0.0;
  int outer = 0;
  bool converged = false;
};

// The smoothing parameters by Newton's method on their logarithms, from a start that sets each
// penalty's mean diagonal to that of the weighted design's cross product over its columns. A
// parameter whose gradient is already below `tol` times the scale of the criterion, `|V| + D / n`,
// stays where it is for the step, and so does one at a bound with its gradient pointing out. The
// Hessian over the others is scaled by the root of its diagonal, so that a parameter on its way to
// either limit, whose curvature falls with its gradient, keeps a step of its own size; its
// eigenvalues are then replaced by their magnitudes, held above 1e-7 of the largest. A step is at
// most `kMaxNewtonStep` in any coordinate and is halved until the criterion falls, and where no
// halving of it does, the steepest descent step of at most `kMaxDescentStep` is tried the same way.
// The search has settled once every gradient is below the tolerance or the Newton step's predicted
// fall in the criterion is below `kResolution` of its scale, and stops unsettled where no step
// lowers the criterion. Where `spec.sp` is given there is no search: the fit is the one at those
// parameters, held as given with no bound.
ResponseFit fit_response(const Design& design, const double* y, const double* w,
                         const AdditiveSpec& spec, int workers) {
  const Response resp(design, y, w, spec.family, spec.gamma, spec, workers);
  const std::size_t m = design.smooths, n = design.n;
  std::vector<double> rho(m, 0.0);
  for (std::size_t j = 0; j < m && spec.sp.empty(); ++j) {
    double kd = 0.0, sd = 0.0;
    for (std::size_t c : design.columns[j]) {
      const double* xc = design.x.data() + c * n;
      for (std::size_t i = 0; i < n; ++i) {
        const double mu = starting_mean(spec.family, y[i], w[i]);
        kd += working_weight_of(spec.family, w[i], mu) * xc[i] * xc[i];
      }
      sd += design.d[c];
    }
    rho[j] = std::min(std::max(std::log(kd / sd), kLogLambdaMin), kLogLambdaMax);
  }
  auto lambdas = [](const std::vector<double>& r) {
    std::vector<double> out(r.size());
    for (std::size_t j = 0; j < r.size(); ++j) out[j] = std::exp(r[j]);
    return out;
  };

  ResponseFit out;
  std::vector<double> lambda = lambdas(rho);
  const bool fixed = !spec.sp.empty();
  if (fixed) lambda = spec.sp;
  Response::State st = resp.fit(lambda, {});
  int it = 0;
  bool converged = m == 0 || fixed;
  for (; it < spec.max_outer && !converged; ++it) {
    std::vector<double> g;
    Matrix h;
    resp.derivatives(st, lambda, g, h);
    const double scale = std::abs(st.score) + st.deviance / static_cast<double>(n);
    const double tolerance = spec.tol * scale;
    std::vector<std::size_t> free;
    double largest = 0.0;
    for (std::size_t j = 0; j < m; ++j) {
      const bool held = (rho[j] >= kLogLambdaMax && g[j] < 0.0) ||
                        (rho[j] <= kLogLambdaMin && g[j] > 0.0);
      if (!held && std::abs(g[j]) > tolerance) {
        free.push_back(j);
        largest = std::max(largest, std::abs(g[j]));
      }
    }
    if (free.empty()) {
      converged = true;
      break;
    }
    const std::size_t f = free.size();
    std::vector<double> root(f);
    for (std::size_t a = 0; a < f; ++a) {
      const double d = std::abs(h[free[a] + free[a] * m]);
      root[a] = d > 0.0 ? std::sqrt(d) : 1.0;
    }
    Matrix hf(f * f);
    for (std::size_t b = 0; b < f; ++b) {
      for (std::size_t a = 0; a < f; ++a) {
        hf[a + b * f] = h[free[a] + free[b] * m] / (root[a] * root[b]);
      }
    }
    std::vector<double> ev;
    Matrix vec;
    symmetric_eigen(hf, f, ev, vec);
    double top = 0.0;
    for (double v : ev) top = std::max(top, std::abs(v));
    std::vector<double> newton(m, 0.0);
    for (std::size_t e = 0; e < f; ++e) {
      const double lam = std::max(std::abs(ev[e]), 1e-7 * top);
      double proj = 0.0;
      for (std::size_t a = 0; a < f; ++a) proj += vec[a + e * f] * g[free[a]] / root[a];
      for (std::size_t a = 0; a < f; ++a) {
        newton[free[a]] -= vec[a + e * f] * proj / (lam * root[a]);
      }
    }
    double predicted = 0.0;
    for (std::size_t a : free) predicted -= g[a] * newton[a];
    if (predicted <= kResolution * scale) {
      converged = true;
      break;
    }
    std::vector<double> descent(m, 0.0);
    for (std::size_t a : free) descent[a] = -g[a];

    bool moved = false;
    for (int attempt = 0; attempt < 2 && !moved; ++attempt) {
      std::vector<double> step = attempt == 0 ? newton : descent;
      double longest = 0.0;
      for (double v : step) longest = std::max(longest, std::abs(v));
      const double cap = attempt == 0 ? kMaxNewtonStep : kMaxDescentStep;
      if (longest == 0.0) continue;
      if (longest > cap) {
        for (double& v : step) v *= cap / longest;
      }
      for (int half = 0; half <= kMaxHalvings; ++half) {
        std::vector<double> trial(m);
        for (std::size_t j = 0; j < m; ++j) {
          trial[j] = std::min(std::max(rho[j] + step[j], kLogLambdaMin), kLogLambdaMax);
        }
        const std::vector<double> tl = lambdas(trial);
        Response::State next = resp.fit(tl, st.beta);
        if (next.score < st.score) {
          rho = std::move(trial);
          lambda = tl;
          st = std::move(next);
          moved = true;
          break;
        }
        for (double& v : step) v *= 0.5;
      }
    }
    if (!moved) break;
  }
  out.beta = st.beta;
  out.lambda = lambda;
  out.edf = resp.column_edf(st, lambda);
  out.score = st.score;
  out.outer = it;
  out.converged = converged && st.settled;
  return out;
}

}  // namespace

Additive additive_fit(const double* x, std::size_t n, std::size_t p, const double* y,
                      const double* w, std::size_t r, const AdditiveSpec& spec) {
  detail::check_finite(x, n * p, "an additive model", "design");
  detail::check_finite(y, n * r, "an additive model", "response");
  detail::check_finite(w, n * r, "an additive model", "weights");
  if (spec.k < 3) throw Error("an additive model's basis holds at least three functions.");
  if (spec.max_knots < spec.k) {
    throw Error("an additive model's knots are at least as many as its basis dimension.");
  }
  for (std::size_t i = 0; i < n * r; ++i) {
    if (!(w[i] > 0.0)) throw Error("an additive model's case weights are positive.");
    if (spec.family == Family::binomial && (y[i] < 0.0 || y[i] > 1.0)) {
      throw Error("a binomial additive model's response lies in [0, 1].");
    }
    if (spec.family == Family::poisson && y[i] < 0.0) {
      throw Error("a Poisson additive model's response is a count of zero or more.");
    }
  }

  // The terms, one column at a time.
  std::vector<Term> built(p);
  std::vector<std::uint8_t> present(p, 0);
  detail::run_tasks(p, spec.threads, [&](std::size_t j) {
    present[j] = build_term(x + j * n, n, static_cast<std::int32_t>(j), spec, built[j]) ? 1 : 0;
  });
  std::vector<Term> terms;
  for (std::size_t j = 0; j < p; ++j) {
    if (present[j]) terms.push_back(std::move(built[j]));
  }

  std::size_t total = 1;
  for (const Term& t : terms) total += t.size;
  if (total > n) {
    throw Error("an additive model of " + std::to_string(total) + " coefficients has more than " +
                "the " + std::to_string(n) + " units it is fitted to; fewer columns or a smaller " +
                "`k` fit.");
  }

  // Every coefficient's column, then the free ones spanned by those before them set aside.
  Matrix full(n * total);
  std::fill(full.begin(), full.begin() + static_cast<std::ptrdiff_t>(n), 1.0);
  std::vector<int> term_of(total, -1);
  std::vector<std::size_t> start(terms.size());
  {
    std::size_t at = 1;
    for (std::size_t t = 0; t < terms.size(); ++t) {
      start[t] = at;
      term_columns(terms[t], x + static_cast<std::size_t>(terms[t].column) * n, n,
                   full.data() + at * n, n);
      for (std::size_t c = 0; c < terms[t].size; ++c) term_of[at + c] = static_cast<int>(t);
      at += terms[t].size;
    }
  }
  std::vector<std::size_t> free_cols{0};
  for (std::size_t t = 0; t < terms.size(); ++t) {
    for (std::size_t c = terms[t].penalised; c < terms[t].size; ++c) free_cols.push_back(start[t] + c);
  }
  std::vector<std::uint8_t> aliased(total, 0);
  {
    Matrix fx(n * free_cols.size());
    for (std::size_t c = 0; c < free_cols.size(); ++c) {
      std::copy(full.begin() + static_cast<std::ptrdiff_t>(free_cols[c] * n),
                full.begin() + static_cast<std::ptrdiff_t>((free_cols[c] + 1) * n),
                fx.begin() + static_cast<std::ptrdiff_t>(c * n));
    }
    std::size_t rank = 0;
    std::vector<double> qraux;
    std::vector<std::size_t> jpvt;
    detail::householder_qr(fx.data(), n, free_cols.size(), kAliasTolerance, rank, qraux, jpvt);
    for (std::size_t c = rank; c < free_cols.size(); ++c) aliased[free_cols[jpvt[c]]] = 1;
  }

  Design design;
  design.n = n;
  std::vector<int> smooth_of_term(terms.size(), -1);
  for (std::size_t t = 0; t < terms.size(); ++t) {
    if (terms[t].penalised > 0) smooth_of_term[t] = static_cast<int>(design.smooths++);
  }
  design.columns.resize(design.smooths);
  if (!spec.sp.empty()) {
    if (spec.sp.size() != design.smooths) {
      throw Error("an additive model has " + std::to_string(design.smooths) + " smoothing " +
                  "parameters and is given " + std::to_string(spec.sp.size()) + ".");
    }
    for (double v : spec.sp) {
      if (!(v > 0.0) || !std::isfinite(v)) {
        throw Error("an additive model's smoothing parameters are positive and finite.");
      }
    }
  }
  for (std::size_t c = 0; c < total; ++c) {
    if (aliased[c]) continue;
    const std::size_t col = design.q++;
    design.x.insert(design.x.end(), full.begin() + static_cast<std::ptrdiff_t>(c * n),
                    full.begin() + static_cast<std::ptrdiff_t>((c + 1) * n));
    design.coefficient.push_back(c);
    design.term.push_back(term_of[c]);
    double d = 0.0;
    int smooth = -1;
    if (term_of[c] >= 0) {
      const std::size_t t = static_cast<std::size_t>(term_of[c]);
      const std::size_t within = c - start[t];
      if (within < terms[t].penalised) {
        d = terms[t].penalty[within];
        smooth = smooth_of_term[t];
        design.columns[static_cast<std::size_t>(smooth)].push_back(col);
      }
    }
    design.d.push_back(d);
    design.smooth.push_back(smooth);
  }

  // One fit per response, over the shared design: the threads run the responses at once where
  // there are several, and a lone response's derivatives where there is one.
  const std::vector<ResponseFit> fits =
      detail::fit_responses(r, spec.threads, [&](std::size_t s, int inner) {
        return fit_response(design, y + s * n, w + s * n, spec, inner);
      });

  Additive out;
  out.family = spec.family;
  out.n_column = static_cast<std::int32_t>(p);
  out.n_coef = static_cast<std::int32_t>(total);
  out.knot_start.push_back(0);
  out.radial_start.push_back(0);
  out.map_start.push_back(0);
  for (const Term& t : terms) {
    out.term_column.push_back(t.column);
    out.term_basis.push_back(static_cast<std::int32_t>(t.basis));
    out.term_size.push_back(static_cast<std::int32_t>(t.size));
    out.term_penalised.push_back(static_cast<std::int32_t>(t.penalised));
    out.term_shift.push_back(t.shift);
    out.knots.insert(out.knots.end(), t.knots.begin(), t.knots.end());
    out.radial.insert(out.radial.end(), t.radial.begin(), t.radial.end());
    out.map.insert(out.map.end(), t.map.begin(), t.map.end());
    out.penalty.insert(out.penalty.end(), t.penalty.begin(),
                       t.penalty.begin() + static_cast<std::ptrdiff_t>(t.penalised));
    out.knot_start.push_back(static_cast<std::int32_t>(out.knots.size()));
    out.radial_start.push_back(static_cast<std::int32_t>(out.radial.size()));
    out.map_start.push_back(static_cast<std::int32_t>(out.map.size()));
  }
  for (std::size_t c = 0; c < total; ++c) {
    if (aliased[c]) out.aliased.push_back(static_cast<std::int32_t>(c));
  }
  out.n_response = static_cast<std::int32_t>(r);
  out.beta.assign(total * r, 0.0);
  out.sp.assign(design.smooths * r, 0.0);
  out.edf.assign(terms.size() * r, 0.0);
  for (std::size_t s = 0; s < r; ++s) {
    const ResponseFit& f = fits[s];
    for (std::size_t c = 0; c < design.q; ++c) {
      out.beta[design.coefficient[c] + s * total] = f.beta[c];
      if (design.term[c] >= 0) {
        out.edf[static_cast<std::size_t>(design.term[c]) + s * terms.size()] += f.edf[c];
      }
    }
    for (std::size_t j = 0; j < design.smooths; ++j) out.sp[j + s * design.smooths] = f.lambda[j];
    out.score.push_back(f.score);
    out.outer.push_back(f.outer);
    out.converged.push_back(f.converged ? 1 : 0);
  }
  return out;
}

void additive_predict(const Additive& fit, const double* x, std::size_t n, std::size_t p,
                      double* out) {
  if (p != static_cast<std::size_t>(fit.n_column)) {
    throw Error("the additive model was fitted on " + std::to_string(fit.n_column) +
                " columns and is asked to predict on " + std::to_string(p) + ".");
  }
  detail::check_finite(x, n * p, "an additive model", "design");
  const std::size_t r = static_cast<std::size_t>(fit.n_response);
  const std::size_t total = static_cast<std::size_t>(fit.n_coef);
  std::vector<double> eta(n * r, 0.0);
  for (std::size_t s = 0; s < r; ++s) {
    for (std::size_t i = 0; i < n; ++i) eta[i + s * n] = fit.beta[s * total];
  }
  std::size_t at = 1;
  Matrix cols;
  for (std::size_t t = 0; t < fit.term_column.size(); ++t) {
    Term term;
    term.column = fit.term_column[t];
    term.basis = static_cast<std::size_t>(fit.term_basis[t]);
    term.size = static_cast<std::size_t>(fit.term_size[t]);
    term.penalised = static_cast<std::size_t>(fit.term_penalised[t]);
    term.shift = fit.term_shift[t];
    term.knots.assign(fit.knots.begin() + fit.knot_start[t], fit.knots.begin() + fit.knot_start[t + 1]);
    term.radial.assign(fit.radial.begin() + fit.radial_start[t],
                       fit.radial.begin() + fit.radial_start[t + 1]);
    term.map.assign(fit.map.begin() + fit.map_start[t], fit.map.begin() + fit.map_start[t + 1]);
    cols.assign(n * term.size, 0.0);
    term_columns(term, x + static_cast<std::size_t>(term.column) * n, n, cols.data(), n);
    for (std::size_t s = 0; s < r; ++s) {
      for (std::size_t c = 0; c < term.size; ++c) {
        const double b = fit.beta[at + c + s * total];
        if (b == 0.0) continue;
        const double* col = cols.data() + c * n;
        for (std::size_t i = 0; i < n; ++i) eta[i + s * n] += b * col[i];
      }
    }
    at += term.size;
  }
  for (std::size_t i = 0; i < n * r; ++i) out[i] = linkinv(fit.family, eta[i]);
}

}  // namespace timesift
