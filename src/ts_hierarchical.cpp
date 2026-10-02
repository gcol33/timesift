#include "ts_hierarchical.h"

#include "ts_internal.h"
#include "ts_sparse.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <queue>
#include <memory>
#include <string>
#include <utility>

namespace timesift {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kNegInf = -std::numeric_limits<double>::infinity();

// ---------------------------------------------------------------------------------------------
// Dense symmetric positive definite algebra, column-major.

// The lower Cholesky factor of the symmetric `a` [d, d], from its lower triangle, in place. False
// where a pivot is not positive.
bool cholesky(std::vector<double>& a, std::size_t d) {
  for (std::size_t j = 0; j < d; ++j) {
    double* cj = a.data() + j * d;
    for (std::size_t k = 0; k < j; ++k) {
      const double* ck = a.data() + k * d;
      const double f = ck[j];
      if (f == 0.0) continue;
      for (std::size_t i = j; i < d; ++i) cj[i] -= ck[i] * f;
    }
    if (!(cj[j] > 0.0) || !std::isfinite(cj[j])) return false;
    const double root = std::sqrt(cj[j]);
    cj[j] = root;
    for (std::size_t i = j + 1; i < d; ++i) cj[i] /= root;
  }
  return true;
}

// Solves `L L' x = b` in place from the factor `l`.
void cholesky_solve(const std::vector<double>& l, std::size_t d, double* b) {
  for (std::size_t j = 0; j < d; ++j) {
    const double* cj = l.data() + j * d;
    b[j] /= cj[j];
    for (std::size_t i = j + 1; i < d; ++i) b[i] -= cj[i] * b[j];
  }
  for (std::size_t j = d; j-- > 0;) {
    const double* cj = l.data() + j * d;
    double s = b[j];
    for (std::size_t i = j + 1; i < d; ++i) s -= cj[i] * b[i];
    b[j] = s / cj[j];
  }
}

double cholesky_log_det(const std::vector<double>& l, std::size_t d) {
  double s = 0.0;
  for (std::size_t j = 0; j < d; ++j) s += std::log(l[j + j * d]);
  return 2.0 * s;
}

// log(1 + e^x) without overflow on either side.
double softplus(double x) {
  if (x > 0.0) return x + std::log1p(std::exp(-x));
  return std::log1p(std::exp(x));
}

double logistic(double x) {
  if (x >= 0.0) return 1.0 / (1.0 + std::exp(-x));
  const double e = std::exp(x);
  return e / (1.0 + e);
}

// ---------------------------------------------------------------------------------------------
// Coordinates and the Hilbert-space basis.

// The coordinates centred on the column means and divided by one factor, the root of the mean of
// the columns' sample variances, so that distances keep their proportions.
struct Standardised {
  double centre[2] = {0.0, 0.0};
  double scale = 1.0;
  std::vector<double> xy;  // [n, 2] column-major
};

Standardised standardise(const double* coords, std::size_t n) {
  Standardised s;
  for (int c = 0; c < 2; ++c) {
    double mean = 0.0;
    for (std::size_t i = 0; i < n; ++i) mean += coords[i + c * n];
    s.centre[c] = mean / static_cast<double>(n);
  }
  double var = 0.0;
  if (n > 1) {
    for (int c = 0; c < 2; ++c) {
      double ss = 0.0;
      for (std::size_t i = 0; i < n; ++i) {
        const double d = coords[i + c * n] - s.centre[c];
        ss += d * d;
      }
      var += ss / static_cast<double>(n - 1);
    }
    var /= 2.0;
  }
  s.scale = var > 0.0 && std::isfinite(var) ? std::sqrt(var) : 1.0;
  s.xy.resize(2 * n);
  for (int c = 0; c < 2; ++c) {
    for (std::size_t i = 0; i < n; ++i) s.xy[i + c * n] = (coords[i + c * n] - s.centre[c]) / s.scale;
  }
  return s;
}

void standardise_like(const double* coords, std::size_t n, const double centre[2], double scale,
                      std::vector<double>& out) {
  out.resize(2 * n);
  for (int c = 0; c < 2; ++c) {
    for (std::size_t i = 0; i < n; ++i) out[i + c * n] = (coords[i + c * n] - centre[c]) / scale;
  }
}

// The box the Laplacian eigenfunctions live on: centred on the coordinates' extent, a factor
// `boundary` wider than it, and no narrower than 0.1 a side.
struct Box {
  double centre[2] = {0.0, 0.0};
  double half[2] = {0.1, 0.1};
};

Box make_box(const std::vector<double>& xy, std::size_t n, double boundary) {
  Box b;
  for (int c = 0; c < 2; ++c) {
    double lo = xy[c * n], hi = xy[c * n];
    for (std::size_t i = 1; i < n; ++i) {
      lo = std::min(lo, xy[i + c * n]);
      hi = std::max(hi, xy[i + c * n]);
    }
    b.centre[c] = (hi + lo) / 2.0;
    b.half[c] = std::max(boundary * (hi - lo) / 2.0, 0.1);
  }
  return b;
}

// The eigenfunctions `sin(pi j (x + L) / (2 L)) / sqrt(L)` of the Laplacian on `[-L, L]`, one per
// pair of frequencies `(j1, j2)`, `j2` fastest, at the rows of `xy` about the box's centre, and
// their eigenvalues `(pi j1 / 2 L1)^2 + (pi j2 / 2 L2)^2`.
void hsgp_basis(const Box& box, int m, const std::vector<double>& xy, std::size_t n,
                std::vector<double>& phi, std::vector<double>* eigenvalue) {
  const std::size_t total = static_cast<std::size_t>(m) * static_cast<std::size_t>(m);
  phi.assign(n * total, 0.0);
  if (eigenvalue != nullptr) eigenvalue->assign(total, 0.0);
  std::vector<double> along[2];
  for (int c = 0; c < 2; ++c) {
    along[c].resize(n * static_cast<std::size_t>(m));
    const double half = box.half[c];
    for (int j = 1; j <= m; ++j) {
      for (std::size_t i = 0; i < n; ++i) {
        const double x = xy[i + c * n] - box.centre[c];
        along[c][i + static_cast<std::size_t>(j - 1) * n] =
            std::sin(kPi * j * (x + half) / (2.0 * half)) / std::sqrt(half);
      }
    }
  }
  for (int j1 = 1; j1 <= m; ++j1) {
    for (int j2 = 1; j2 <= m; ++j2) {
      const std::size_t col = static_cast<std::size_t>(j1 - 1) * m + static_cast<std::size_t>(j2 - 1);
      const double* a = along[0].data() + static_cast<std::size_t>(j1 - 1) * n;
      const double* b = along[1].data() + static_cast<std::size_t>(j2 - 1) * n;
      double* out = phi.data() + col * n;
      for (std::size_t i = 0; i < n; ++i) out[i] = a[i] * b[i];
      if (eigenvalue != nullptr) {
        const double e1 = kPi * j1 / (2.0 * box.half[0]);
        const double e2 = kPi * j2 / (2.0 * box.half[1]);
        (*eigenvalue)[col] = e1 * e1 + e2 * e2;
      }
    }
  }
}

// The square root of the squared-exponential spectral density `sigma^2 2 pi l^2 exp(-l^2 w^2 / 2)`
// at each eigenvalue `w^2`.
void hsgp_scale(const std::vector<double>& eigenvalue, double sigma, double range,
                std::vector<double>& s) {
  s.resize(eigenvalue.size());
  const double l2 = range * range;
  for (std::size_t j = 0; j < eigenvalue.size(); ++j) {
    s[j] = std::sqrt(sigma * sigma * 2.0 * kPi * l2 * std::exp(-0.5 * l2 * eigenvalue[j]));
  }
}

// ---------------------------------------------------------------------------------------------
// The nearest-neighbour field's geometry.

double covariance(int cov, double d, double sigma2, double range) {
  if (d < 1e-10) return sigma2;
  switch (cov) {
    case 0:
      return sigma2 * std::exp(-d / range);
    case 1: {
      const double x = std::sqrt(3.0) * d / range;
      return sigma2 * (1.0 + x) * std::exp(-x);
    }
    case 2: {
      const double x = std::sqrt(5.0) * d / range;
      return sigma2 * (1.0 + x + x * x / 3.0) * std::exp(-x);
    }
    default: {
      const double r = d / range;
      return sigma2 * std::exp(-r * r);
    }
  }
}

// Added to the diagonal of each neighbourhood's covariance before it is factored, and the least
// conditional variance a location keeps.
constexpr double kNeighbourNugget = 1e-8;
constexpr double kVarianceFloor = 1e-10;
// The nugget of the neighbourhood a prediction is made from.
constexpr double kPredictNugget = 1e-6;

// The `k` locations nearest to `(qx, qy)` among the first `hi` of the locations `xy` [L, 2], which
// are sorted by their first coordinate, nearest first. The search starts at position `pos` and
// widens along the first coordinate, stopping once that coordinate alone is farther than the
// `k`-th nearest found.
std::vector<std::size_t> nearest_locations(const std::vector<double>& xy, std::size_t count,
                                           double qx, double qy, std::size_t k, std::size_t hi,
                                           std::size_t pos) {
  using Item = std::pair<double, std::size_t>;
  std::priority_queue<Item> heap;
  const double inf = std::numeric_limits<double>::infinity();
  std::ptrdiff_t down = static_cast<std::ptrdiff_t>(pos) - 1;
  std::size_t up = pos;
  while (down >= 0 || up < hi) {
    const double to_down = down >= 0 ? qx - xy[static_cast<std::size_t>(down)] : inf;
    const double to_up = up < hi ? xy[up] - qx : inf;
    const bool take_down = to_down <= to_up;
    const double dx = take_down ? to_down : to_up;
    if (heap.size() == k && dx * dx > heap.top().first) break;
    const std::size_t j = take_down ? static_cast<std::size_t>(down--) : up++;
    const double ex = xy[j] - qx, ey = xy[j + count] - qy;
    const Item item{ex * ex + ey * ey, j};
    if (heap.size() < k) {
      heap.push(item);
    } else if (item < heap.top()) {
      heap.pop();
      heap.push(item);
    }
  }
  std::vector<std::size_t> out(heap.size());
  for (std::size_t i = out.size(); i-- > 0;) {
    out[i] = heap.top().second;
    heap.pop();
  }
  return out;
}

// The targets' distinct locations in lexicographic order of the coordinates, and for each the
// nearest earlier ones it is conditioned on.
struct Vecchia {
  std::size_t count = 0;
  std::vector<double> xy;                            // [count, 2], location fastest
  std::vector<std::size_t> of_target;                // [n] each target's location
  std::vector<std::vector<std::size_t>> neighbour;   // [count] earlier locations, nearest first
};

Vecchia build_vecchia(const std::vector<double>& xy, std::size_t n, int neighbours) {
  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    if (xy[a] != xy[b]) return xy[a] < xy[b];
    return xy[a + n] < xy[b + n];
  });
  Vecchia v;
  v.of_target.assign(n, 0);
  std::vector<double> xs, ys;
  for (std::size_t r = 0; r < n; ++r) {
    const std::size_t i = order[r];
    if (r == 0 || xy[i] != xs.back() || xy[i + n] != ys.back()) {
      xs.push_back(xy[i]);
      ys.push_back(xy[i + n]);
    }
    v.of_target[i] = xs.size() - 1;
  }
  v.count = xs.size();
  v.xy.assign(2 * v.count, 0.0);
  std::copy(xs.begin(), xs.end(), v.xy.begin());
  std::copy(ys.begin(), ys.end(), v.xy.begin() + static_cast<std::ptrdiff_t>(v.count));
  v.neighbour.resize(v.count);
  for (std::size_t i = 1; i < v.count; ++i) {
    v.neighbour[i] = nearest_locations(v.xy, v.count, xs[i], ys[i],
                                       static_cast<std::size_t>(neighbours), i, i);
  }
  return v;
}

// The precision `Lambda = (I - A)' D^-1 (I - A)` of the field at the locations, `A` holding each
// location's regression on its neighbours and `D` its conditional variances, written into `values`
// by slot of `pattern`, and `log |Lambda| = -sum log D`.
bool vecchia_precision(const Vecchia& geo, const SparseSymmetric& pattern, double sigma,
                       double range, int cov, std::vector<double>& values, double& log_det) {
  const double sigma2 = sigma * sigma;
  values.assign(pattern.nonzeros(), 0.0);
  log_det = 0.0;
  std::vector<double> c, coef, l;
  std::vector<std::size_t> member;
  for (std::size_t i = 0; i < geo.count; ++i) {
    const std::vector<std::size_t>& nb = geo.neighbour[i];
    const std::size_t q = nb.size();
    double variance = sigma2;
    coef.assign(q, 0.0);
    if (q > 0) {
      c.assign(q, 0.0);
      l.assign(q * q, 0.0);
      for (std::size_t a = 0; a < q; ++a) {
        const double dx = geo.xy[nb[a]] - geo.xy[i];
        const double dy = geo.xy[nb[a] + geo.count] - geo.xy[i + geo.count];
        c[a] = covariance(cov, std::sqrt(dx * dx + dy * dy), sigma2, range);
        l[a + a * q] = sigma2 + kNeighbourNugget;
        for (std::size_t b = 0; b < a; ++b) {
          const double ex = geo.xy[nb[a]] - geo.xy[nb[b]];
          const double ey = geo.xy[nb[a] + geo.count] - geo.xy[nb[b] + geo.count];
          l[a + b * q] = covariance(cov, std::sqrt(ex * ex + ey * ey), sigma2, range);
        }
      }
      if (!cholesky(l, q)) return false;
      coef = c;
      cholesky_solve(l, q, coef.data());
      double explained = 0.0;
      for (std::size_t a = 0; a < q; ++a) explained += c[a] * coef[a];
      variance = std::max(sigma2 - explained, kVarianceFloor);
    }
    log_det -= std::log(variance);
    const double inv = 1.0 / variance;
    member.assign(1, i);
    member.insert(member.end(), nb.begin(), nb.end());
    for (std::size_t a = 0; a <= q; ++a) {
      const double ba = a == 0 ? 1.0 : -coef[a - 1];
      for (std::size_t b = 0; b <= a; ++b) {
        const double bb = b == 0 ? 1.0 : -coef[b - 1];
        const std::size_t row = std::max(member[a], member[b]);
        const std::size_t col = std::min(member[a], member[b]);
        values[pattern.slot(row, col)] += inv * ba * bb;
      }
    }
  }
  return std::isfinite(log_det);
}

// ---------------------------------------------------------------------------------------------
// The conditional fit at fixed hyperparameters.

struct Conditional {
  bool ok = false;
  double log_marginal = kNegInf;
  std::vector<double> latent;  // beta, the field's coefficients, the unit intercepts
};

struct Data {
  const double* x;
  std::size_t n;
  std::size_t p;
  const double* y;
  const double* w;
  const std::int32_t* unit;
  std::size_t g;
  double tau_beta;
};

// The latent vector's linear algebra at one set of hyperparameters: the linear predictor, the
// prior's quadratic form and determinant, and the Newton step from the information.
class System {
 public:
  virtual ~System() = default;
  virtual std::size_t size() const = 0;
  virtual void predictor(const std::vector<double>& latent, std::vector<double>& eta) const = 0;
  // Half the prior's quadratic form at `latent`.
  virtual double penalty(const std::vector<double>& latent) const = 0;
  // The log determinant of the prior's proper precision.
  virtual double log_prior_det() const = 0;
  // The step solving `H step = Z' residual - Q latent` at the case weights `weight`, leaving the
  // largest absolute gradient entry and `log |H|` to the accessors; false where `H` is singular.
  virtual bool newton(const std::vector<double>& latent, const std::vector<double>& weight,
                      const std::vector<double>& residual, std::vector<double>& step) = 0;
  virtual double gradient_max() const = 0;
  virtual double log_det() const = 0;
};

double log_lik(const Data& d, const std::vector<double>& eta) {
  double j = 0.0;
  for (std::size_t i = 0; i < d.n; ++i) j += d.w[i] * (d.y[i] * eta[i] - softplus(eta[i]));
  return j;
}

Conditional fit_latent(const Data& d, const HierSpec& spec, System& sys, const Conditional* warm) {
  const std::size_t n = d.n, dim = sys.size();
  Conditional out;
  out.latent.assign(dim, 0.0);
  if (warm != nullptr && warm->latent.size() == dim) out.latent = warm->latent;
  std::vector<double> eta(n), trial_eta(n), residual(n), weight(n), step(dim), trial(dim);
  sys.predictor(out.latent, eta);
  double current = log_lik(d, eta) - sys.penalty(out.latent);
  if (!std::isfinite(current)) return out;

  auto moments = [&]() {
    for (std::size_t i = 0; i < n; ++i) {
      const double pr = logistic(eta[i]);
      residual[i] = d.w[i] * (d.y[i] - pr);
      weight[i] = d.w[i] * pr * (1.0 - pr);
    }
  };
  bool settled = false;
  for (int it = 0; it < spec.max_newton; ++it) {
    moments();
    if (!sys.newton(out.latent, weight, residual, step)) return out;
    double t = 1.0;
    bool moved = false;
    for (int half = 0; half < 40; ++half) {
      for (std::size_t a = 0; a < dim; ++a) trial[a] = out.latent[a] + t * step[a];
      sys.predictor(trial, trial_eta);
      const double next = log_lik(d, trial_eta) - sys.penalty(trial);
      if (std::isfinite(next) && next >= current - 1e-12 * (1.0 + std::fabs(current))) {
        moved = true;
        const double gain = next - current;
        out.latent = trial;
        eta = trial_eta;
        current = next;
        double largest = 0.0, scale = 1.0;
        for (std::size_t a = 0; a < dim; ++a) {
          largest = std::max(largest, std::fabs(t * step[a]));
          scale = std::max(scale, std::fabs(out.latent[a]));
        }
        if (largest < spec.tol * scale || gain < 1e-15 * (1.0 + std::fabs(current))) settled = true;
        break;
      }
      t *= 0.5;
    }
    if (!moved) {
      // A step that no halving improves is the optimum to rounding where the gradient is already
      // at it.
      settled = sys.gradient_max() < 1e-5;
      break;
    }
    if (settled) break;
  }
  if (!settled) return out;

  // The information at the mode, for its determinant.
  moments();
  if (!sys.newton(out.latent, weight, residual, step)) return out;
  out.log_marginal = current + 0.5 * sys.log_prior_det() - 0.5 * sys.log_det();
  out.ok = std::isfinite(out.log_marginal);
  return out;
}

// Dense latent block: the fixed effects and the field's scaled basis coefficients, with the unit
// intercepts eliminated, the information being diagonal in them.
class DenseSystem : public System {
 public:
  DenseSystem(const Data& d, const std::vector<double>* phi, std::size_t m,
              const std::vector<double>& s, double sigma_u)
      : d_(d), d1_(d.p + m), tau_u_(1.0 / (sigma_u * sigma_u)) {
    const std::size_t n = d.n;
    z_.resize(n * d1_);
    std::copy(d.x, d.x + n * d.p, z_.begin());
    for (std::size_t j = 0; j < m; ++j) {
      const double* src = phi->data() + j * n;
      double* dst = z_.data() + (d.p + j) * n;
      for (std::size_t i = 0; i < n; ++i) dst[i] = src[i] * s[j];
    }
    prior_.assign(d1_, 1.0);
    for (std::size_t j = 0; j < d.p; ++j) prior_[j] = d.tau_beta;
    grad_.resize(d1_);
    hess_.resize(d1_ * d1_);
    b_col_.resize(n);
    grad_u_.resize(d.g);
    diag_u_.resize(d.g);
    couple_.resize(d.g * d1_);
  }

  std::size_t size() const override { return d1_ + d_.g; }

  void predictor(const std::vector<double>& latent, std::vector<double>& eta) const override {
    std::fill(eta.begin(), eta.end(), 0.0);
    for (std::size_t j = 0; j < d1_; ++j) {
      const double b = latent[j];
      if (b == 0.0) continue;
      const double* col = z_.data() + j * d_.n;
      for (std::size_t i = 0; i < d_.n; ++i) eta[i] += b * col[i];
    }
    if (d_.g > 0) {
      for (std::size_t i = 0; i < d_.n; ++i) eta[i] += latent[d1_ + static_cast<std::size_t>(d_.unit[i])];
    }
  }

  double penalty(const std::vector<double>& latent) const override {
    double q = 0.0;
    for (std::size_t a = 0; a < d1_; ++a) q += prior_[a] * latent[a] * latent[a];
    for (std::size_t g = 0; g < d_.g; ++g) q += tau_u_ * latent[d1_ + g] * latent[d1_ + g];
    return 0.5 * q;
  }

  double log_prior_det() const override {
    return static_cast<double>(d_.p) * std::log(d_.tau_beta) +
           static_cast<double>(d_.g) * std::log(tau_u_);
  }

  bool newton(const std::vector<double>& latent, const std::vector<double>& weight,
              const std::vector<double>& residual, std::vector<double>& step) override {
    const std::size_t n = d_.n, g = d_.g;
    gmax_ = 0.0;
    for (std::size_t a = 0; a < d1_; ++a) {
      const double* col = z_.data() + a * n;
      double s = 0.0;
      for (std::size_t i = 0; i < n; ++i) s += col[i] * residual[i];
      grad_[a] = s - prior_[a] * latent[a];
      gmax_ = std::max(gmax_, std::fabs(grad_[a]));
    }
    // The weighted Gram matrix of the dense design plus the prior precisions.
    std::fill(hess_.begin(), hess_.end(), 0.0);
    for (std::size_t a = 0; a < d1_; ++a) {
      const double* ca = z_.data() + a * n;
      for (std::size_t i = 0; i < n; ++i) b_col_[i] = weight[i] * ca[i];
      for (std::size_t b = 0; b <= a; ++b) {
        const double* cb = z_.data() + b * n;
        double s = 0.0;
        for (std::size_t i = 0; i < n; ++i) s += b_col_[i] * cb[i];
        hess_[a + b * d1_] = s;
      }
      hess_[a + a * d1_] += prior_[a];
    }
    log_det_ = 0.0;
    if (g > 0) {
      // The intercepts' information is diagonal, so the dense block's Schur complement is its Gram
      // matrix less one rank-one term per unit.
      std::fill(grad_u_.begin(), grad_u_.end(), 0.0);
      std::fill(diag_u_.begin(), diag_u_.end(), 0.0);
      std::fill(couple_.begin(), couple_.end(), 0.0);
      for (std::size_t i = 0; i < n; ++i) {
        const std::size_t u = static_cast<std::size_t>(d_.unit[i]);
        grad_u_[u] += residual[i];
        diag_u_[u] += weight[i];
        if (weight[i] == 0.0) continue;
        double* h = couple_.data() + u * d1_;
        for (std::size_t a = 0; a < d1_; ++a) h[a] += weight[i] * z_[i + a * n];
      }
      for (std::size_t u = 0; u < g; ++u) {
        grad_u_[u] -= tau_u_ * latent[d1_ + u];
        gmax_ = std::max(gmax_, std::fabs(grad_u_[u]));
        diag_u_[u] += tau_u_;
        log_det_ += std::log(diag_u_[u]);
        const double* h = couple_.data() + u * d1_;
        const double inv = 1.0 / diag_u_[u];
        for (std::size_t a = 0; a < d1_; ++a) {
          if (h[a] == 0.0) continue;
          const double f = h[a] * inv;
          for (std::size_t b = 0; b <= a; ++b) hess_[a + b * d1_] -= f * h[b];
          grad_[a] -= f * grad_u_[u];
        }
      }
    }
    chol_ = hess_;
    if (!cholesky(chol_, d1_)) return false;
    log_det_ += cholesky_log_det(chol_, d1_);
    step.assign(d1_ + g, 0.0);
    std::copy(grad_.begin(), grad_.end(), step.begin());
    cholesky_solve(chol_, d1_, step.data());
    for (std::size_t u = 0; u < g; ++u) {
      double s = grad_u_[u];
      const double* h = couple_.data() + u * d1_;
      for (std::size_t a = 0; a < d1_; ++a) s -= h[a] * step[a];
      step[d1_ + u] = s / diag_u_[u];
    }
    return true;
  }

  double gradient_max() const override { return gmax_; }
  double log_det() const override { return log_det_; }

 private:
  const Data& d_;
  std::size_t d1_;
  double tau_u_;
  std::vector<double> z_, prior_, grad_, hess_, chol_, b_col_, grad_u_, diag_u_, couple_;
  double gmax_ = 0.0;
  double log_det_ = 0.0;
};

// The sparsity of the nearest-neighbour model's information over the field's values and the unit
// intercepts, analysed once for every set of hyperparameters: the neighbourhoods' cliques, and the
// pairs of a target's location and unit.
struct VecchiaStructure {
  VecchiaStructure(const Data& d, const Vecchia& geo)
      : pattern(d.g > 0 ? geo.count + d.g : geo.count, positions(d, geo)), analysis(pattern) {
    const std::size_t L = geo.count;
    loc_slot.resize(d.n);
    for (std::size_t i = 0; i < d.n; ++i) loc_slot[i] = pattern.slot(geo.of_target[i], geo.of_target[i]);
    if (d.g > 0) {
      unit_slot.resize(d.n);
      cross_slot.resize(d.n);
      for (std::size_t i = 0; i < d.n; ++i) {
        const std::size_t u = L + static_cast<std::size_t>(d.unit[i]);
        unit_slot[i] = pattern.slot(u, u);
        cross_slot[i] = pattern.slot(u, geo.of_target[i]);
      }
    }
  }

  SparseSymmetric pattern;
  SparseCholesky analysis;
  std::vector<std::size_t> loc_slot, unit_slot, cross_slot;

 private:
  static std::vector<std::pair<std::size_t, std::size_t>> positions(const Data& d,
                                                                     const Vecchia& geo) {
    std::vector<std::pair<std::size_t, std::size_t>> out;
    for (std::size_t i = 0; i < geo.count; ++i)
      for (const std::size_t j : geo.neighbour[i]) {
        out.push_back({i, j});
        for (const std::size_t k : geo.neighbour[i])
          if (k < j) out.push_back({j, k});
      }
    if (d.g > 0)
      for (std::size_t i = 0; i < d.n; ++i)
        out.push_back({geo.count + static_cast<std::size_t>(d.unit[i]), geo.of_target[i]});
    return out;
  }
};

// Latent block: the fixed effects, the field's value at each distinct location, the unit
// intercepts. The field and the intercepts are sparse and factored together; the fixed effects, a
// few columns, are eliminated against them through a Schur complement.
class VecchiaSystem : public System {
 public:
  VecchiaSystem(const Data& d, const Vecchia& geo, const VecchiaStructure& st, double sigma_u,
                double sigma, double range, int cov)
      : d_(d), geo_(geo), st_(st), chol_(st.analysis), tau_u_(1.0 / (sigma_u * sigma_u)) {
    ok_ = vecchia_precision(geo, st.pattern, sigma, range, cov, precision_, log_det_precision_);
  }

  bool ok() const { return ok_; }

  std::size_t size() const override { return d_.p + geo_.count + d_.g; }

  void predictor(const std::vector<double>& latent, std::vector<double>& eta) const override {
    const std::size_t p = d_.p, n = d_.n, L = geo_.count;
    std::fill(eta.begin(), eta.end(), 0.0);
    for (std::size_t j = 0; j < p; ++j) {
      const double b = latent[j];
      if (b == 0.0) continue;
      for (std::size_t i = 0; i < n; ++i) eta[i] += b * d_.x[i + j * n];
    }
    for (std::size_t i = 0; i < n; ++i) {
      eta[i] += latent[p + geo_.of_target[i]];
      if (d_.g > 0) eta[i] += latent[p + L + static_cast<std::size_t>(d_.unit[i])];
    }
  }

  double penalty(const std::vector<double>& latent) const override {
    const std::size_t p = d_.p, L = geo_.count;
    double q = 0.0;
    for (std::size_t j = 0; j < p; ++j) q += d_.tau_beta * latent[j] * latent[j];
    std::vector<double> product(L + d_.g);
    st_.pattern.multiply(precision_, latent.data() + p, product.data());
    for (std::size_t s = 0; s < L; ++s) q += latent[p + s] * product[s];
    for (std::size_t u = 0; u < d_.g; ++u) q += tau_u_ * latent[p + L + u] * latent[p + L + u];
    return 0.5 * q;
  }

  double log_prior_det() const override {
    return static_cast<double>(d_.p) * std::log(d_.tau_beta) +
           static_cast<double>(d_.g) * std::log(tau_u_) + log_det_precision_;
  }

  bool newton(const std::vector<double>& latent, const std::vector<double>& weight,
              const std::vector<double>& residual, std::vector<double>& step) override {
    const std::size_t p = d_.p, n = d_.n, L = geo_.count, g = d_.g, ns = L + g;
    std::vector<double> values = precision_;
    std::vector<double> grad_s(ns, 0.0), grad_b(p, 0.0);
    st_.pattern.multiply(precision_, latent.data() + p, grad_s.data());
    for (std::size_t s = 0; s < ns; ++s) grad_s[s] = -grad_s[s];
    for (std::size_t u = 0; u < g; ++u) grad_s[L + u] -= tau_u_ * latent[p + L + u];
    std::vector<double> hbs(p * ns, 0.0), hbb(p * p, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      const double wt = weight[i];
      const std::size_t l = geo_.of_target[i];
      const std::size_t u = g > 0 ? L + static_cast<std::size_t>(d_.unit[i]) : 0;
      grad_s[l] += residual[i];
      if (g > 0) grad_s[u] += residual[i];
      for (std::size_t j = 0; j < p; ++j) grad_b[j] += d_.x[i + j * n] * residual[i];
      if (wt == 0.0) continue;
      values[st_.loc_slot[i]] += wt;
      if (g > 0) {
        values[st_.unit_slot[i]] += wt;
        values[st_.cross_slot[i]] += wt;
      }
      for (std::size_t j = 0; j < p; ++j) {
        const double xj = wt * d_.x[i + j * n];
        hbs[j + l * p] += xj;
        if (g > 0) hbs[j + u * p] += xj;
        for (std::size_t k = 0; k <= j; ++k) hbb[j + k * p] += xj * d_.x[i + k * n];
      }
    }
    gmax_ = 0.0;
    for (std::size_t j = 0; j < p; ++j) {
      grad_b[j] -= d_.tau_beta * latent[j];
      hbb[j + j * p] += d_.tau_beta;
      gmax_ = std::max(gmax_, std::fabs(grad_b[j]));
    }
    for (std::size_t s = 0; s < ns; ++s) gmax_ = std::max(gmax_, std::fabs(grad_s[s]));
    for (std::size_t u = 0; u < g; ++u) values[st_.pattern.slot(L + u, L + u)] += tau_u_;

    if (!chol_.factor(values)) return false;
    std::vector<double> solved_g = grad_s;
    chol_.solve(solved_g.data());
    std::vector<double> inverse_coupling(p * ns);  // Hss^-1 Hsb, column j at j * ns
    std::vector<double> column(ns);
    for (std::size_t j = 0; j < p; ++j) {
      for (std::size_t s = 0; s < ns; ++s) column[s] = hbs[j + s * p];
      chol_.solve(column.data());
      std::copy(column.begin(), column.end(), inverse_coupling.begin() + static_cast<std::ptrdiff_t>(j * ns));
    }
    std::vector<double> schur(p * p), rhs(p);
    for (std::size_t j = 0; j < p; ++j) {
      double r = grad_b[j];
      for (std::size_t s = 0; s < ns; ++s) r -= hbs[j + s * p] * solved_g[s];
      rhs[j] = r;
      for (std::size_t k = 0; k <= j; ++k) {
        double v = hbb[j + k * p];
        for (std::size_t s = 0; s < ns; ++s) v -= hbs[j + s * p] * inverse_coupling[k * ns + s];
        schur[j + k * p] = v;
      }
    }
    if (!cholesky(schur, p)) return false;
    log_det_ = chol_.log_det() + cholesky_log_det(schur, p);
    cholesky_solve(schur, p, rhs.data());
    step.assign(p + ns, 0.0);
    std::copy(rhs.begin(), rhs.end(), step.begin());
    for (std::size_t s = 0; s < ns; ++s) {
      double v = solved_g[s];
      for (std::size_t k = 0; k < p; ++k) v -= inverse_coupling[k * ns + s] * rhs[k];
      step[p + s] = v;
    }
    return true;
  }

  double gradient_max() const override { return gmax_; }
  double log_det() const override { return log_det_; }

 private:
  const Data& d_;
  const Vecchia& geo_;
  const VecchiaStructure& st_;
  SparseCholesky chol_;
  double tau_u_;
  std::vector<double> precision_;
  double log_det_precision_ = 0.0;
  bool ok_ = false;
  double gmax_ = 0.0;
  double log_det_ = 0.0;
};

// What a field contributes to the conditional fit: the basis and its eigenvalues under hsgp, the
// locations and their neighbourhoods under nngp.
struct FieldModel {
  const std::vector<double>* phi = nullptr;
  std::size_t m = 0;
  const std::vector<double>* eigenvalue = nullptr;
  const Vecchia* vecchia = nullptr;
  const VecchiaStructure* structure = nullptr;
  double extent = 1.0;
};

// ---------------------------------------------------------------------------------------------
// The hyperparameters.

// `log lambda - lambda sigma + log sigma` with `lambda = -log(alpha) / U`: the penalised-complexity
// prior on a standard deviation, `P(sigma > U) = alpha`, as a density in its logarithm.
double pc_log_density(double log_sigma, double u, double alpha) {
  const double lambda = -std::log(alpha) / u;
  return std::log(lambda) - lambda * std::exp(log_sigma) + log_sigma;
}

// The penalised-complexity prior on a range in two dimensions, `P(range < r0) = alpha`, as a density
// in its logarithm: an exponential prior on `range^-1` carried to the range, `log lambda -
// log range - lambda / range` with `lambda = -log(alpha) r0`.
double pc_range_log_density(double log_range, double r0, double alpha) {
  const double lambda = -std::log(alpha) * r0;
  return std::log(lambda) - log_range - lambda * std::exp(-log_range);
}

// The log posterior of the hyperparameters up to a constant, and the conditional fit behind it.
class Hyper {
 public:
  Hyper(const Data& data, const HierSpec& spec, const FieldModel& model)
      : data_(data), spec_(spec), model_(model) {}

  // The range the field starts from: the anchor of its prior.
  double range_anchor() const { return spec_.range_fraction * model_.extent; }

  std::size_t k() const { return (spec_.unit ? 1u : 0u) + (spec_.field != Field::none ? 2u : 0u); }

  double log_prior(const std::vector<double>& theta) const {
    double out = 0.0;
    std::size_t at = 0;
    if (spec_.unit) out += pc_log_density(theta[at++], spec_.sd_u, spec_.sd_alpha);
    if (spec_.field != Field::none) {
      out += pc_log_density(theta[at], spec_.sd_u, spec_.sd_alpha);
      out += pc_range_log_density(theta[at + 1], range_anchor(), spec_.range_alpha);
    }
    return out;
  }

  Conditional conditional(const std::vector<double>& theta, const Conditional* warm) const {
    std::size_t at = 0;
    double sigma_u = 1.0;
    if (spec_.unit) sigma_u = std::exp(theta[at++]);
    if (spec_.field == Field::nngp) {
      VecchiaSystem sys(data_, *model_.vecchia, *model_.structure, sigma_u, std::exp(theta[at]),
                        std::exp(theta[at + 1]), spec_.cov);
      if (!sys.ok()) return Conditional();
      return fit_latent(data_, spec_, sys, warm);
    }
    std::vector<double> s;
    if (spec_.field == Field::hsgp) {
      hsgp_scale(*model_.eigenvalue, std::exp(theta[at]), std::exp(theta[at + 1]), s);
    }
    DenseSystem sys(data_, model_.phi, model_.m, s, sigma_u);
    return fit_latent(data_, spec_, sys, warm);
  }

  // The log posterior at `theta`, -inf where the conditional fit failed.
  double value(const std::vector<double>& theta, const Conditional* warm, Conditional* kept) const {
    Conditional c = conditional(theta, warm);
    if (!c.ok) return kNegInf;
    const double v = c.log_marginal + log_prior(theta);
    if (kept != nullptr) *kept = std::move(c);
    return v;
  }

 private:
  const Data& data_;
  const HierSpec& spec_;
  const FieldModel& model_;
};

// The mode of `hyper`'s log posterior by BFGS on central differences, from `start`; the
// conditional fit at the mode is kept in `at_mode`. False where no starting point gave a finite
// value.
bool find_mode(const Hyper& hyper, std::vector<double> start, std::vector<double>& mode,
               Conditional& at_mode, double& log_post) {
  const std::size_t k = start.size();
  Conditional warm;
  double f = hyper.value(start, nullptr, &warm);
  if (!std::isfinite(f)) return false;
  std::vector<double> x = start;
  const double h = 1e-4;
  auto gradient = [&](const std::vector<double>& at, const Conditional& near, std::vector<double>& g) {
    g.assign(k, 0.0);
    for (std::size_t a = 0; a < k; ++a) {
      std::vector<double> up = at, down = at;
      up[a] += h;
      down[a] -= h;
      const double fu = hyper.value(up, &near, nullptr);
      const double fd = hyper.value(down, &near, nullptr);
      if (!std::isfinite(fu) || !std::isfinite(fd)) return false;
      g[a] = (fu - fd) / (2.0 * h);
    }
    return true;
  };
  std::vector<double> g, g_next;
  if (!gradient(x, warm, g)) return false;
  // The inverse Hessian of the negated log posterior, started at the identity.
  std::vector<double> inv(k * k, 0.0);
  for (std::size_t a = 0; a < k; ++a) inv[a + a * k] = 1.0;
  for (int it = 0; it < 200; ++it) {
    double gmax = 0.0;
    for (double v : g) gmax = std::max(gmax, std::fabs(v));
    if (gmax < 1e-7) break;
    // The ascent direction `inv g`, limited to a step of at most two on the log scale.
    std::vector<double> dir(k, 0.0);
    for (std::size_t a = 0; a < k; ++a) {
      for (std::size_t b = 0; b < k; ++b) dir[a] += inv[a + b * k] * g[b];
    }
    double longest = 0.0, slope = 0.0;
    for (std::size_t a = 0; a < k; ++a) {
      longest = std::max(longest, std::fabs(dir[a]));
      slope += dir[a] * g[a];
    }
    if (!(slope > 0.0)) {
      for (std::size_t a = 0; a < k; ++a) {
        dir[a] = g[a];
        for (std::size_t b = 0; b < k; ++b) inv[a + b * k] = a == b ? 1.0 : 0.0;
      }
      longest = gmax;
      slope = 0.0;
      for (std::size_t a = 0; a < k; ++a) slope += g[a] * g[a];
    }
    double t = longest > 2.0 ? 2.0 / longest : 1.0;
    std::vector<double> trial(k);
    Conditional trial_fit;
    double f_next = kNegInf;
    bool moved = false;
    for (int half = 0; half < 40; ++half) {
      for (std::size_t a = 0; a < k; ++a) trial[a] = x[a] + t * dir[a];
      f_next = hyper.value(trial, &warm, &trial_fit);
      if (std::isfinite(f_next) && f_next >= f + 1e-4 * t * slope) {
        moved = true;
        break;
      }
      t *= 0.5;
    }
    if (!moved) break;
    if (!gradient(trial, trial_fit, g_next)) break;
    // The BFGS update of the inverse Hessian from the step `s` and the change in gradient `y`.
    std::vector<double> sv(k), yv(k);
    double sy = 0.0;
    for (std::size_t a = 0; a < k; ++a) {
      sv[a] = trial[a] - x[a];
      yv[a] = g[a] - g_next[a];
      sy += sv[a] * yv[a];
    }
    const double gain = f_next - f;
    x = trial;
    f = f_next;
    warm = std::move(trial_fit);
    g = g_next;
    if (sy > 1e-12) {
      std::vector<double> iy(k, 0.0);
      double yiy = 0.0;
      for (std::size_t a = 0; a < k; ++a) {
        for (std::size_t b = 0; b < k; ++b) iy[a] += inv[a + b * k] * yv[b];
        yiy += yv[a] * iy[a];
      }
      for (std::size_t a = 0; a < k; ++a) {
        for (std::size_t b = 0; b < k; ++b) {
          inv[a + b * k] += (1.0 + yiy / sy) * sv[a] * sv[b] / sy - (iy[a] * sv[b] + sv[a] * iy[b]) / sy;
        }
      }
    }
    if (gain < 1e-13 * (1.0 + std::fabs(f))) {
      double gm = 0.0;
      for (double v : g) gm = std::max(gm, std::fabs(v));
      if (gm < 1e-5) break;
    }
  }
  mode = x;
  at_mode = std::move(warm);
  log_post = f;
  return true;
}

// The symmetric eigendecomposition of `a` [k, k] by Jacobi rotations; the eigenvectors are the
// columns of `vec`.
void jacobi_eigen(std::vector<double> a, std::size_t k, std::vector<double>& val,
                  std::vector<double>& vec) {
  vec.assign(k * k, 0.0);
  for (std::size_t i = 0; i < k; ++i) vec[i + i * k] = 1.0;
  for (int sweep = 0; sweep < 100; ++sweep) {
    double off = 0.0;
    for (std::size_t i = 0; i < k; ++i) {
      for (std::size_t j = 0; j < i; ++j) off += a[i + j * k] * a[i + j * k];
    }
    if (off < 1e-30) break;
    for (std::size_t p = 0; p + 1 < k; ++p) {
      for (std::size_t q = p + 1; q < k; ++q) {
        const double apq = a[q + p * k];
        if (std::fabs(apq) < 1e-300) continue;
        const double theta = (a[q + q * k] - a[p + p * k]) / (2.0 * apq);
        const double t = (theta >= 0.0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
        const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
        for (std::size_t r = 0; r < k; ++r) {
          const double arp = a[r + p * k], arq = a[r + q * k];
          a[r + p * k] = c * arp - s * arq;
          a[r + q * k] = s * arp + c * arq;
        }
        for (std::size_t r = 0; r < k; ++r) {
          const double apr = a[p + r * k], aqr = a[q + r * k];
          a[p + r * k] = c * apr - s * aqr;
          a[q + r * k] = s * apr + c * aqr;
        }
        for (std::size_t r = 0; r < k; ++r) {
          const double vrp = vec[r + p * k], vrq = vec[r + q * k];
          vec[r + p * k] = c * vrp - s * vrq;
          vec[r + q * k] = s * vrp + c * vrq;
        }
      }
    }
  }
  val.resize(k);
  for (std::size_t i = 0; i < k; ++i) val[i] = a[i + i * k];
}

// The curvature of the log posterior at `mode`, by central differences.
std::vector<double> curvature(const Hyper& hyper, const std::vector<double>& mode,
                              const Conditional& at_mode, double log_post) {
  const std::size_t k = mode.size();
  const double h = 5e-3;
  std::vector<double> hess(k * k, 0.0);
  auto at = [&](const std::vector<double>& th) {
    const double v = hyper.value(th, &at_mode, nullptr);
    return std::isfinite(v) ? v : kNegInf;
  };
  for (std::size_t a = 0; a < k; ++a) {
    std::vector<double> up = mode, down = mode;
    up[a] += h;
    down[a] -= h;
    hess[a + a * k] = -(at(up) - 2.0 * log_post + at(down)) / (h * h);
    for (std::size_t b = 0; b < a; ++b) {
      std::vector<double> pp = mode, pm = mode, mp = mode, mm = mode;
      pp[a] += h; pp[b] += h;
      pm[a] += h; pm[b] -= h;
      mp[a] -= h; mp[b] += h;
      mm[a] -= h; mm[b] -= h;
      const double v = -(at(pp) - at(pm) - at(mp) + at(mm)) / (4.0 * h * h);
      hess[a + b * k] = hess[b + a * k] = v;
    }
  }
  return hess;
}

}  // namespace

Hierarchical hierarchical_fit(const double* x, std::size_t n, std::size_t p, const double* y,
                              const double* w, const std::int32_t* unit, std::size_t n_unit,
                              const double* coords, const HierSpec& spec) {
  if (n == 0 || p == 0) throw Error("a hierarchical fit needs at least one target and one column.");
  detail::check_finite(x, n * p, "a hierarchical fit", "design");
  detail::check_finite(y, n, "a hierarchical fit", "response");
  detail::check_finite(w, n, "a hierarchical fit", "weights");
  for (std::size_t i = 0; i < n; ++i) {
    if (y[i] != 0.0 && y[i] != 1.0) throw Error("a hierarchical fit reads a response of zero and one.");
    if (w[i] < 0.0) throw Error("a hierarchical fit's case weights are zero or more.");
  }
  if (spec.unit) {
    if (unit == nullptr || n_unit == 0) throw Error("a hierarchical fit with unit intercepts names each target's unit.");
    for (std::size_t i = 0; i < n; ++i) {
      if (unit[i] < 0 || static_cast<std::size_t>(unit[i]) >= n_unit) {
        throw Error("a target's unit lies outside the units the fit declares.");
      }
    }
  } else {
    n_unit = 0;
  }
  if (spec.field != Field::none) {
    if (coords == nullptr) throw Error("a hierarchical fit with a field places each target by its coordinates.");
    detail::check_finite(coords, 2 * n, "a hierarchical fit", "coordinates");
  }
  if (spec.field == Field::nngp) {
    if (spec.neighbours < 1 || spec.neighbours > 200) throw Error("a nearest-neighbour field conditions on between 1 and 200 neighbours.");
    if (spec.cov < 0 || spec.cov > 3) throw Error("a nearest-neighbour field's covariance is one of exponential, matern32, matern52 and gaussian.");
  }

  Hierarchical out;
  out.field = spec.field;
  out.n_column = static_cast<std::int32_t>(p);
  out.n_unit = static_cast<std::int32_t>(n_unit);

  // The field's basis, on the standardised coordinates.
  std::vector<double> phi, eigenvalue;
  std::size_t n_field = 0;
  double extent = 1.0;
  if (spec.field != Field::none) {
    const Standardised st = standardise(coords, n);
    double diagonal = 0.0;
    for (int c = 0; c < 2; ++c) {
      double lo = st.xy[c * n], hi = st.xy[c * n];
      for (std::size_t i = 1; i < n; ++i) {
        lo = std::min(lo, st.xy[i + c * n]);
        hi = std::max(hi, st.xy[i + c * n]);
      }
      diagonal += (hi - lo) * (hi - lo);
    }
    extent = std::sqrt(diagonal);
    if (!(extent > 0.0)) throw Error("a hierarchical field needs targets at more than one place.");
  }
  if (spec.field == Field::hsgp) {
    if (spec.m < 3 || spec.m > 50) throw Error("a hierarchical field's basis has between 3 and 50 functions per axis.");
    if (!(spec.boundary >= 1.0)) throw Error("a hierarchical field's boundary factor is at least one.");
    const Standardised st = standardise(coords, n);
    const Box box = make_box(st.xy, n, spec.boundary);
    hsgp_basis(box, spec.m, st.xy, n, phi, &eigenvalue);
    n_field = eigenvalue.size();
    out.centre[0] = st.centre[0];
    out.centre[1] = st.centre[1];
    out.scale = st.scale;
    out.m = spec.m;
    for (int c = 0; c < 2; ++c) {
      out.box_centre[c] = box.centre[c];
      out.box_half[c] = box.half[c];
    }
  }

  const Data data{x, n, p, y, w, unit, n_unit, 1.0 / (spec.beta_sd * spec.beta_sd)};
  FieldModel model;
  model.phi = &phi;
  model.m = n_field;
  model.eigenvalue = &eigenvalue;
  model.extent = extent;
  Vecchia geo;
  std::unique_ptr<VecchiaStructure> structure;
  if (spec.field == Field::nngp) {
    const Standardised st = standardise(coords, n);
    geo = build_vecchia(st.xy, n, spec.neighbours);
    structure.reset(new VecchiaStructure(data, geo));
    model.vecchia = &geo;
    model.structure = structure.get();
    n_field = geo.count;
    out.centre[0] = st.centre[0];
    out.centre[1] = st.centre[1];
    out.scale = st.scale;
    out.location = geo.xy;
    out.n_location = static_cast<std::int32_t>(geo.count);
    out.neighbours = spec.neighbours;
    out.cov = spec.cov;
  }
  const Hyper hyper(data, spec, model);
  const std::size_t k = hyper.k();
  out.n_theta = static_cast<std::int32_t>(k);
  out.n_field = static_cast<std::int32_t>(n_field);

  std::vector<double> mode(k, 0.0);
  Conditional at_mode;
  double log_post = 0.0;
  const bool fixed = !spec.theta.empty();
  if (fixed) {
    if (spec.theta.size() != k) {
      throw Error("a hierarchical model has " + std::to_string(k) + " hyperparameters and is given " +
                  std::to_string(spec.theta.size()) + ".");
    }
    mode = spec.theta;
    at_mode = hyper.conditional(mode, nullptr);
    if (!at_mode.ok) throw Error("the hierarchical model's conditional fit did not settle at the given hyperparameters.");
    log_post = at_mode.log_marginal + hyper.log_prior(mode);
  } else if (k == 0) {
    at_mode = hyper.conditional(mode, nullptr);
    if (!at_mode.ok) throw Error("the hierarchical model's posterior mode did not settle.");
    log_post = at_mode.log_marginal;
  } else {
    std::vector<double> start(k, 0.0);
    if (spec.field != Field::none) start[k - 1] = std::log(hyper.range_anchor());
    if (!find_mode(hyper, start, mode, at_mode, log_post)) {
      throw Error("the hierarchical model's hyperparameters have no finite log posterior to start from.");
    }
  }
  out.theta_hat = mode;
  out.log_marginal = at_mode.log_marginal;

  // The nodes: the mode alone where the hyperparameters are not integrated over.
  std::vector<std::vector<double>> theta_nodes;
  if (spec.field == Field::none || fixed) {
    theta_nodes.push_back(mode);
  } else {
    std::vector<double> hess = curvature(hyper, mode, at_mode, log_post);
    for (double v : hess) {
      if (!std::isfinite(v)) {
        std::fill(hess.begin(), hess.end(), 0.0);
        for (std::size_t a = 0; a < k; ++a) hess[a + a * k] = 1.0;
        break;
      }
    }
    std::vector<double> val, vec;
    jacobi_eigen(hess, k, val, vec);
    // Each axis' spread is the inverse root of its curvature, held within [0.05, 2] on the log
    // scale so that a flat or a sharp direction still gets a grid that means something.
    std::vector<double> sd(k);
    for (std::size_t a = 0; a < k; ++a) {
      const double v = val[a] > 1e-12 ? 1.0 / std::sqrt(val[a]) : 2.0;
      sd[a] = std::min(std::max(v, 0.05), 2.0);
    }
    const int per = std::max(spec.nodes, 1);
    std::size_t total = 1;
    for (std::size_t a = 0; a < k; ++a) total *= static_cast<std::size_t>(per);
    for (std::size_t c = 0; c < total; ++c) {
      std::vector<double> z(k);
      std::size_t rest = c;
      for (std::size_t a = 0; a < k; ++a) {
        const int at = static_cast<int>(rest % static_cast<std::size_t>(per));
        rest /= static_cast<std::size_t>(per);
        z[a] = (at - (per - 1) / 2.0) * spec.step * sd[a];
      }
      std::vector<double> theta = mode;
      for (std::size_t b = 0; b < k; ++b) {
        for (std::size_t a = 0; a < k; ++a) theta[b] += vec[b + a * k] * z[a];
      }
      theta_nodes.push_back(std::move(theta));
    }
  }
  const std::size_t nodes = theta_nodes.size();
  std::vector<Conditional> fits(nodes);
  std::vector<double> post(nodes, kNegInf);
  detail::run_tasks(nodes, spec.threads, [&](std::size_t c) {
    Conditional f = hyper.conditional(theta_nodes[c], &at_mode);
    if (f.ok) post[c] = f.log_marginal + hyper.log_prior(theta_nodes[c]);
    fits[c] = std::move(f);
  });
  double top = kNegInf;
  for (double v : post) top = std::max(top, v);
  if (!std::isfinite(top)) throw Error("none of the hierarchical model's hyperparameter nodes settled.");
  std::vector<double> weight(nodes, 0.0);
  double total = 0.0;
  for (std::size_t c = 0; c < nodes; ++c) {
    weight[c] = std::isfinite(post[c]) ? std::exp(post[c] - top) : 0.0;
    total += weight[c];
  }
  for (double& v : weight) v /= total;

  out.n_node = static_cast<std::int32_t>(nodes);
  out.node_theta.assign(nodes * k, 0.0);
  out.node_weight = weight;
  out.node_log_post = post;
  out.node_field.assign(nodes * n_field, 0.0);
  out.beta.assign(p, 0.0);
  out.unit_effect.assign(n_unit, 0.0);
  out.converged = 1;
  for (std::size_t c = 0; c < nodes; ++c) {
    for (std::size_t a = 0; a < k; ++a) out.node_theta[c + a * nodes] = theta_nodes[c][a];
    if (!fits[c].ok) {
      if (weight[c] > 0.0) out.converged = 0;
      continue;
    }
    for (std::size_t j = 0; j < p; ++j) out.beta[j] += weight[c] * fits[c].latent[j];
    for (std::size_t g = 0; g < n_unit; ++g) {
      out.unit_effect[g] += weight[c] * fits[c].latent[p + n_field + g];
    }
    for (std::size_t j = 0; j < n_field; ++j) out.node_field[c + j * nodes] = fits[c].latent[p + j];
  }
  return out;
}

void hierarchical_predict(const Hierarchical& fit, const double* x, std::size_t n, std::size_t p,
                          const std::int32_t* unit, const double* coords, double* eta) {
  if (static_cast<std::int32_t>(p) != fit.n_column) {
    throw Error("the hierarchical model was fitted on " + std::to_string(fit.n_column) +
                " columns and is handed " + std::to_string(p) + ".");
  }
  detail::check_finite(x, n * p, "a hierarchical prediction", "design");
  std::fill(eta, eta + n, 0.0);
  for (std::size_t j = 0; j < p; ++j) {
    const double b = fit.beta[j];
    if (b == 0.0) continue;
    for (std::size_t i = 0; i < n; ++i) eta[i] += b * x[i + j * n];
  }
  if (fit.n_unit > 0) {
    if (unit == nullptr) throw Error("a prediction from a model with unit intercepts names each target's unit.");
    for (std::size_t i = 0; i < n; ++i) {
      if (unit[i] >= 0 && unit[i] < fit.n_unit) eta[i] += fit.unit_effect[static_cast<std::size_t>(unit[i])];
    }
  }
  if (fit.field == Field::nngp) {
    if (coords == nullptr) throw Error("a prediction from a model with a field places each target by its coordinates.");
    detail::check_finite(coords, 2 * n, "a hierarchical prediction", "coordinates");
    std::vector<double> xy;
    standardise_like(coords, n, fit.centre, fit.scale, xy);
    const std::size_t count = static_cast<std::size_t>(fit.n_location);
    const std::size_t nodes = static_cast<std::size_t>(fit.n_node);
    const std::size_t k = static_cast<std::size_t>(fit.n_theta);
    const std::size_t q = std::min(static_cast<std::size_t>(fit.neighbours), count);
    // Each target's neighbourhood among all the fitted locations does not depend on the node.
    std::vector<std::vector<std::size_t>> around(n);
    for (std::size_t i = 0; i < n; ++i) {
      const double qx = xy[i];
      std::size_t lo = 0, hi = count;
      while (lo < hi) {
        const std::size_t mid = (lo + hi) / 2;
        if (fit.location[mid] < qx) lo = mid + 1; else hi = mid;
      }
      around[i] = nearest_locations(fit.location, count, qx, xy[i + n], q, count, lo);
    }
    std::vector<double> c, l, coef;
    for (std::size_t node = 0; node < nodes; ++node) {
      const double wt = fit.node_weight[node];
      if (wt == 0.0) continue;
      const double sigma = std::exp(fit.node_theta[node + (k - 2) * nodes]);
      const double range = std::exp(fit.node_theta[node + (k - 1) * nodes]);
      const double sigma2 = sigma * sigma;
      for (std::size_t i = 0; i < n; ++i) {
        const std::vector<std::size_t>& nb = around[i];
        const std::size_t m = nb.size();
        c.assign(m, 0.0);
        l.assign(m * m, 0.0);
        for (std::size_t a = 0; a < m; ++a) {
          const double dx = fit.location[nb[a]] - xy[i];
          const double dy = fit.location[nb[a] + count] - xy[i + n];
          c[a] = covariance(fit.cov, std::sqrt(dx * dx + dy * dy), sigma2, range);
          l[a + a * m] = sigma2 + kPredictNugget;
          for (std::size_t b = 0; b < a; ++b) {
            const double ex = fit.location[nb[a]] - fit.location[nb[b]];
            const double ey = fit.location[nb[a] + count] - fit.location[nb[b] + count];
            l[a + b * m] = covariance(fit.cov, std::sqrt(ex * ex + ey * ey), sigma2, range);
          }
        }
        if (!cholesky(l, m)) throw Error("a neighbourhood's covariance is not positive definite.");
        coef = c;
        cholesky_solve(l, m, coef.data());
        double mean = 0.0;
        for (std::size_t a = 0; a < m; ++a) mean += coef[a] * fit.node_field[node + nb[a] * nodes];
        eta[i] += wt * mean;
      }
    }
  }
  if (fit.field == Field::hsgp) {
    if (coords == nullptr) throw Error("a prediction from a model with a field places each target by its coordinates.");
    detail::check_finite(coords, 2 * n, "a hierarchical prediction", "coordinates");
    std::vector<double> xy, phi, eigenvalue;
    standardise_like(coords, n, fit.centre, fit.scale, xy);
    Box box;
    for (int c = 0; c < 2; ++c) {
      box.centre[c] = fit.box_centre[c];
      box.half[c] = fit.box_half[c];
    }
    hsgp_basis(box, fit.m, xy, n, phi, &eigenvalue);
    const std::size_t nodes = static_cast<std::size_t>(fit.n_node);
    const std::size_t k = static_cast<std::size_t>(fit.n_theta);
    const std::size_t m = static_cast<std::size_t>(fit.n_field);
    std::vector<double> s;
    for (std::size_t c = 0; c < nodes; ++c) {
      const double wt = fit.node_weight[c];
      if (wt == 0.0) continue;
      const double sigma = std::exp(fit.node_theta[c + (k - 2) * nodes]);
      const double range = std::exp(fit.node_theta[c + (k - 1) * nodes]);
      hsgp_scale(eigenvalue, sigma, range, s);
      for (std::size_t j = 0; j < m; ++j) {
        const double coef = wt * s[j] * fit.node_field[c + j * nodes];
        if (coef == 0.0) continue;
        const double* col = phi.data() + j * n;
        for (std::size_t i = 0; i < n; ++i) eta[i] += coef * col[i];
      }
    }
  }
}

}  // namespace timesift
