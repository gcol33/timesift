#include "ts_glm.h"

#include "ts_normal.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <limits>
#include <utility>

// Every product and sum below is rounded on its own. Contracting a product into the following sum
// as one fused multiply-add rounds once where the fixtures' reference rounds twice, which moves the
// rank decisions and the coefficients in their last bits, so contraction is off for this file.
#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

namespace timesift {

namespace {

// The inner product of `a` and `b` over `len` entries, accumulated onto `seed` in index order.
double dot_onto(double seed, const double* a, const double* b, std::size_t len) {
  for (std::size_t i = 0; i < len; ++i) seed += a[i] * b[i];
  return seed;
}

// `b += t a` over `len` entries.
void add_multiple(double t, const double* a, double* b, std::size_t len) {
  for (std::size_t i = 0; i < len; ++i) b[i] += t * a[i];
}

double euclidean_norm(const double* v, std::size_t len) {
  return std::sqrt(dot_onto(0.0, v, v, len));
}

// A column of the matrix being decomposed, as the pivoting sees it.
struct PivotColumn {
  double norm;         // the norm of the part of the column below the rows reduced so far
  double floor;        // the norm under which the column counts as spanned by those before it
  std::size_t origin;  // the column's index in the matrix as given
};

// Overwrites `v` [m] with the Householder vector `u` that reflects it onto `-s e_1`, where `s` is
// its norm carrying the sign of `v[0]`, and returns `s`. The vector is scaled so that `u[0]` is
// `1 + |v[0]| / |v|`, which keeps it at least one and the reflection `I - u u' / u[0]`. A zero `v`
// has no reflection and is left as it is, with zero returned.
double form_reflector(double* v, std::size_t m) {
  double s = euclidean_norm(v, m);
  if (s == 0.0) return 0.0;
  if (v[0] != 0.0) s = std::copysign(s, v[0]);
  const double scale = 1.0 / s;
  for (std::size_t i = 0; i < m; ++i) v[i] *= scale;
  v[0] = 1.0 + v[0];
  return s;
}

// The norm of a column below the next row, downdated from its norm below the current one once the
// reflection has been applied (Golub & Van Loan, sec. 5.4.1): the entry `head` that leaves the
// active part takes its share of the square with it. Where that share is all but the whole of the
// square, the downdate has cancelled away its correct digits, and the norm is taken again from the
// entries `below` the head.
double downdate_norm(double norm, double head, const double* below, std::size_t len) {
  const double r = std::abs(head) / norm;
  const double left = std::max(1.0 - r * r, 0.0);
  if (left >= 1e-6) return norm * std::sqrt(left);
  return euclidean_norm(below, len);
}

// Applies the `j`th stored reflection to `y` [n]. The reflection vector is the part of column `j`
// below the diagonal, led by `lead` in the diagonal's place.
void apply_stored_reflection(const double* qr, std::size_t n, std::size_t j, double lead,
                             double* y) {
  const double* tail = qr + j * n + j + 1;
  const std::size_t len = n - j - 1;
  const double t = -dot_onto(lead * y[j], tail, y + j + 1, len) / lead;
  y[j] += t * lead;
  add_multiple(t, tail, y + j + 1, len);
}

// The inverse logit's derivative, the linear predictor held at the same 30 as the mean.
constexpr double kLogitBound = 30.0;

double logit_slope(double eta) {
  if (eta > kLogitBound || eta < -kLogitBound) return DBL_EPSILON;
  const double one_plus = 1.0 + std::exp(eta);
  return std::exp(eta) / (one_plus * one_plus);
}

// `y log(y / mu)`, taken as zero at `y = 0`.
double y_log_y_over(double y, double mu) { return y != 0.0 ? y * std::log(y / mu) : 0.0; }

// A family under its link: the mean as a function of the linear predictor, the mean's derivative,
// the link itself, the variance function, and the deviance (McCullagh & Nelder 1989, ch. 2 and 4).
class Model {
 public:
  Model(Family family, Link link)
      : binomial_(family == Family::binomial), probit_(link == Link::probit) {}

  double mean(double eta) const {
    if (!binomial_) return eta;
    return probit_ ? probit_linkinv(eta) : logit_linkinv(eta);
  }

  double slope(double eta) const {
    if (!binomial_) return 1.0;
    return probit_ ? std::max(detail::dnorm(eta), DBL_EPSILON) : logit_slope(eta);
  }

  double link(double mu) const {
    if (!binomial_) return mu;
    return probit_ ? detail::qnorm(mu) : std::log(mu / (1.0 - mu));
  }

  double variance(double mu) const { return binomial_ ? mu * (1.0 - mu) : 1.0; }

  // The starting means: the response itself for the gaussian family, and for the binomial one the
  // response pulled half a trial towards one half, so that no start sits at zero or one.
  void start(const double* y, const double* w, std::size_t n, double* eta, double* mu) const {
    for (std::size_t i = 0; i < n; ++i) {
      if (binomial_) {
        eta[i] = link((w[i] * y[i] + 0.5) / (w[i] + 1.0));
        mu[i] = mean(eta[i]);
      } else {
        eta[i] = y[i];
        mu[i] = y[i];
      }
    }
  }

  double deviance(const double* y, const double* mu, const double* w, std::size_t n) const {
    double d = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      if (binomial_) {
        d += 2.0 * w[i] * (y_log_y_over(y[i], mu[i]) + y_log_y_over(1.0 - y[i], 1.0 - mu[i]));
      } else {
        const double r = y[i] - mu[i];
        d += w[i] * r * r;
      }
    }
    return d;
  }

 private:
  bool binomial_;
  bool probit_;
};

}  // namespace

double logit_linkinv(double eta) {
  const double t = eta < -kLogitBound ? DBL_EPSILON
                                      : (eta > kLogitBound ? 1.0 / DBL_EPSILON : std::exp(eta));
  return t / (1.0 + t);
}

double probit_linkinv(double eta) {
  static const double bound = -detail::qnorm(DBL_EPSILON);
  return detail::pnorm(std::min(std::max(eta, -bound), bound));
}

namespace detail {

void dqrdc2(double* x, std::size_t n, std::size_t p, double tol, std::size_t& k,
            std::vector<double>& qraux, std::vector<std::size_t>& jpvt) {
  std::vector<PivotColumn> cols(p);
  for (std::size_t j = 0; j < p; ++j) {
    const double norm = euclidean_norm(x + j * n, n);
    cols[j] = {norm, (norm == 0.0 ? 1.0 : norm) * tol, j};
  }
  qraux.assign(p, 0.0);
  jpvt.resize(p);

  // Columns from `kept` on have been set aside as spanned by the ones before them. A column set
  // aside moves to the very end, and the columns after it close up behind, so the columns kept
  // stay in their given order.
  std::size_t kept = p;
  const std::size_t steps = std::min(n, p);
  for (std::size_t l = 0; l < steps; ++l) {
    while (l < kept && !(cols[l].norm >= cols[l].floor)) {
      std::rotate(x + l * n, x + (l + 1) * n, x + p * n);
      std::rotate(cols.begin() + static_cast<std::ptrdiff_t>(l),
                  cols.begin() + static_cast<std::ptrdiff_t>(l + 1), cols.end());
      --kept;
    }
    qraux[l] = cols[l].norm;
    jpvt[l] = cols[l].origin;
    if (l + 1 == n) continue;  // the last row has nothing below it to annihilate

    double* u = x + l * n + l;
    const std::size_t m = n - l;
    const double s = form_reflector(u, m);
    if (s == 0.0) continue;
    for (std::size_t j = l + 1; j < p; ++j) {
      double* xj = x + j * n + l;
      const double t = -dot_onto(0.0, u, xj, m) / u[0];
      add_multiple(t, u, xj, m);
      if (cols[j].norm != 0.0) cols[j].norm = downdate_norm(cols[j].norm, xj[0], xj + 1, m - 1);
    }
    qraux[l] = u[0];
    u[0] = -s;
  }
  for (std::size_t j = steps; j < p; ++j) {
    qraux[j] = cols[j].norm;
    jpvt[j] = cols[j].origin;
  }
  k = std::min(kept, n);
}

void qr_qty(const double* qr, std::size_t n, std::size_t k, const double* qraux, double* y) {
  const std::size_t reflections = std::min(k, n - 1);
  for (std::size_t j = 0; j < reflections; ++j) {
    if (qraux[j] != 0.0) apply_stored_reflection(qr, n, j, qraux[j], y);
  }
}

void qr_qy(const double* qr, std::size_t n, std::size_t k, const double* qraux, double* y) {
  const std::size_t reflections = std::min(k, n - 1);
  for (std::size_t j = reflections; j-- > 0;) {
    if (qraux[j] != 0.0) apply_stored_reflection(qr, n, j, qraux[j], y);
  }
}

bool qr_backsolve(const double* qr, std::size_t n, std::size_t k, double* b) {
  for (std::size_t j = k; j-- > 0;) {
    const double* col = qr + j * n;
    if (col[j] == 0.0) return false;
    b[j] /= col[j];
    add_multiple(-b[j], col, b, j);
  }
  return true;
}

std::vector<double> dqrls(double* x, std::size_t n, std::size_t p, double* b, double tol,
                          std::size_t& rank) {
  std::vector<double> qraux;
  std::vector<std::size_t> jpvt;
  dqrdc2(x, n, p, tol, rank, qraux, jpvt);
  std::vector<double> coef(p, 0.0);
  if (rank == 0) return coef;
  qr_qty(x, n, rank, qraux.data(), b);
  std::vector<double> solved(b, b + rank);
  if (!qr_backsolve(x, n, rank, solved.data())) {
    solved.assign(rank, std::numeric_limits<double>::quiet_NaN());
  }
  for (std::size_t j = 0; j < rank; ++j) coef[jpvt[j]] = solved[j];
  return coef;
}

}  // namespace detail

Glm glm_fit(const double* x, std::size_t n, std::size_t q, const double* y, const double* w,
            Family family, double epsilon, int max_iter, Link link) {
  const Model model(family, link);
  std::vector<double> eta(n), mu(n), slope(n);
  model.start(y, w, n, eta.data(), mu.data());
  double previous = model.deviance(y, mu.data(), w, n);

  Glm out;
  out.beta.assign(q, 0.0);
  const double tol = std::min(1e-7, epsilon / 1000.0);
  std::vector<std::size_t> rows;
  std::vector<double> root_w, a, z;
  for (int iter = 0; iter < max_iter; ++iter) {
    // The rows that carry weight: a positive prior weight and a mean still moving with the
    // linear predictor.
    rows.clear();
    for (std::size_t i = 0; i < n; ++i) {
      slope[i] = model.slope(eta[i]);
      if (w[i] > 0.0 && slope[i] != 0.0) rows.push_back(i);
    }
    if (rows.empty()) return out;

    // The weighted least-squares problem of this iteration: the working response
    // `eta + (y - mu) / mu'` and the design, each row scaled by the root of the working weight
    // `w mu'^2 / V(mu)`.
    const std::size_t m = rows.size();
    root_w.resize(m);
    z.resize(m);
    for (std::size_t r = 0; r < m; ++r) {
      const std::size_t i = rows[r];
      const double s = slope[i];
      root_w[r] = std::sqrt((w[i] * s * s) / model.variance(mu[i]));
      z[r] = (eta[i] + (y[i] - mu[i]) / s) * root_w[r];
    }
    a.resize(m * q);
    for (std::size_t c = 0; c < q; ++c) {
      const double* xc = x + c * n;
      double* ac = a.data() + c * m;
      for (std::size_t r = 0; r < m; ++r) ac[r] = xc[rows[r]] * root_w[r];
    }

    std::size_t rank = 0;
    std::vector<double> beta = detail::dqrls(a.data(), m, q, z.data(), tol, rank);
    if (!std::all_of(beta.begin(), beta.end(), [](double v) { return std::isfinite(v); })) {
      return out;
    }

    std::fill(eta.begin(), eta.end(), 0.0);
    for (std::size_t c = 0; c < q; ++c) add_multiple(beta[c], x + c * n, eta.data(), n);
    for (std::size_t i = 0; i < n; ++i) mu[i] = model.mean(eta[i]);
    const double dev = model.deviance(y, mu.data(), w, n);
    out.beta = std::move(beta);
    out.rank = static_cast<std::int32_t>(rank);
    out.deviance = dev;
    if (!std::isfinite(dev)) return out;
    if (std::abs(dev - previous) / (0.1 + std::abs(dev)) < epsilon) {
      out.converged = true;
      return out;
    }
    previous = dev;
  }
  return out;
}

}  // namespace timesift
