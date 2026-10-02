#include "ts_penalised.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <functional>
#include <limits>
#include <system_error>
#include <thread>

// The elastic net by pathwise coordinate descent (Friedman, Hastie and Tibshirani 2010, "Regularization
// paths for generalized linear models via coordinate descent", Journal of Statistical Software
// 33(1)): the penalised objective is minimised one coefficient at a time by soft thresholding, at
// each penalty of a descending path, starting from the solution at the penalty above. A binomial or
// Poisson response is fitted by iteratively reweighted least squares, one penalised quadratic per
// reweighting. The sequential strong rule (Tibshirani et al. 2012, "Strong rules for discarding
// predictors in lasso-type problems", JRSSB 74:245-266) screens the columns offered at each
// penalty, and the optimality condition is checked afterwards over the columns it left out.
namespace timesift {
namespace {

// Constants of the numerical conventions the fixtures pin: the bound on a binomial linear
// predictor, the floor under the mixing when the largest penalty is derived (it gives a ridge a
// finite start), and the probability a held-out case is read at no closer than to zero or one. The
// bound on the linear predictor is read on a Poisson one as well, where it keeps the mean finite.
constexpr double kLinkBound = 250.0;
constexpr double kMixingFloor = 1e-3;
constexpr double kHeldOutProbFloor = 1e-5;

// ---------------------------------------------------------------------------------------------
// Vector arithmetic.
//
// The inner products are summed into four accumulators. A single accumulator chains every addition
// on the one before it, so the descent would run at the latency of a floating-point add rather than
// at its throughput; the package is compiled at the optimisation R sets, which neither vectorises
// this nor reassociates it. The order is fixed, so the sum is the same number on every platform.

double dot(const double* a, const double* b, std::size_t n) {
  double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
  std::size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    s0 += a[i] * b[i];
    s1 += a[i + 1] * b[i + 1];
    s2 += a[i + 2] * b[i + 2];
    s3 += a[i + 3] * b[i + 3];
  }
  double s = (s0 + s1) + (s2 + s3);
  for (; i < n; ++i) s += a[i] * b[i];
  return s;
}

double weighted_dot(const double* a, const double* b, const double* c, std::size_t n) {
  double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
  std::size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    s0 += a[i] * b[i] * c[i];
    s1 += a[i + 1] * b[i + 1] * c[i + 1];
    s2 += a[i + 2] * b[i + 2] * c[i + 2];
    s3 += a[i + 3] * b[i + 3] * c[i + 3];
  }
  double s = (s0 + s1) + (s2 + s3);
  for (; i < n; ++i) s += a[i] * b[i] * c[i];
  return s;
}

double sum_of(const double* a, std::size_t n) {
  double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
  std::size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    s0 += a[i];
    s1 += a[i + 1];
    s2 += a[i + 2];
    s3 += a[i + 3];
  }
  double s = (s0 + s1) + (s2 + s3);
  for (; i < n; ++i) s += a[i];
  return s;
}

// `out += b * a` and `out += b * a * c`, case by case. Two cases are written per turn of the loop,
// which lets the compiler issue them as one at the optimisation R builds a package at; each case is
// still its own expression in its own order, so the numbers are those of a case-at-a-time loop.
void axpy(double* out, const double* a, double b, std::size_t n) {
  std::size_t i = 0;
  for (; i + 2 <= n; i += 2) {
    out[i] += b * a[i];
    out[i + 1] += b * a[i + 1];
  }
  for (; i < n; ++i) out[i] += b * a[i];
}

void weighted_axpy(double* out, const double* a, const double* c, double b, std::size_t n) {
  std::size_t i = 0;
  for (; i + 2 <= n; i += 2) {
    out[i] += b * a[i] * c[i];
    out[i + 1] += b * a[i + 1] * c[i + 1];
  }
  for (; i < n; ++i) out[i] += b * a[i] * c[i];
}

// Twice the Poisson log likelihood's shortfall from the saturated model at one case:
// `2 (y log(y / mu) - (y - mu))`, the logarithm taken as zero at `y = 0`.
double poisson_unit_deviance(double y, double mu) {
  return 2.0 * ((y > 0.0 ? y * std::log(y / mu) : 0.0) - (y - mu));
}

// log(1 + e^x) without overflow on either side.
double softplus(double x) {
  if (x > 0.0) return x + std::log1p(std::exp(-x));
  return std::log1p(std::exp(x));
}

// ---------------------------------------------------------------------------------------------
// The standardised design.
//
// Every column is centred on its weighted mean and divided by its weighted standard deviation, so
// a penalty that is one number over every column costs each column the same whatever unit it was
// recorded in. A column holding one value has no spread and stays at zero throughout.
//
// A Gaussian fit's quadratic weight is the case weight, fixed along the whole path, so its square
// root is multiplied into every column and into the intercept's column: the quadratic then has
// unit weight and a coordinate update reads only the column and the residual. A binomial fit's
// weight moves with every reweighting, so its columns stay unweighted and its intercept column is
// a column of ones; so does a Poisson fit's.
struct Standardised {
  std::size_t n = 0;
  std::size_t p = 0;
  bool weighted_columns = false;
  std::vector<double> z;                // [n, p] column-major
  std::vector<double> centre, scale;    // per column, on the scale handed over
  std::vector<std::uint8_t> live;       // the column has spread
  std::vector<double> relative_penalty; // per column, summing to p
  std::vector<double> case_weight;      // summing to one
  std::vector<double> ones;             // the intercept's column

  const double* column(std::size_t j) const { return z.data() + j * n; }
};

std::vector<double> normalised_weights(const double* w, std::size_t n) {
  std::vector<double> out(n, 0.0);
  double sum = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double wi = w == nullptr ? 1.0 : w[i];
    if (!(wi >= 0.0) || !std::isfinite(wi)) {
      throw Error("a case weight of a penalised fit is negative or not a number.");
    }
    out[i] = wi;
    sum += wi;
  }
  if (!(sum > 0.0)) throw Error("a penalised fit was handed case weights that sum to zero.");
  for (double& wi : out) wi /= sum;
  return out;
}

// Penalty factors scaled to sum to the column count, so that factors all equal to one give the
// path no factors give.
std::vector<double> relative_penalties(const std::vector<double>& factor, std::size_t p) {
  std::vector<double> out(p, 1.0);
  if (factor.empty()) return out;
  if (factor.size() != p) throw Error("a penalised fit takes one penalty factor per column.");
  double sum = 0.0;
  for (std::size_t j = 0; j < p; ++j) {
    const double f = factor[j];
    if (!(f >= 0.0) || !std::isfinite(f)) {
      throw Error("a penalty factor is negative or not a number.");
    }
    out[j] = f;
    sum += f;
  }
  if (sum > 0.0) {
    const double to_p = static_cast<double>(p) / sum;
    for (double& f : out) f *= to_p;
  }
  return out;
}

Standardised standardise(const double* x, const double* w, std::size_t n, std::size_t p,
                         const PenaltySpec& spec, bool weighted_columns) {
  Standardised s;
  s.n = n;
  s.p = p;
  s.case_weight = normalised_weights(w, n);
  s.weighted_columns = weighted_columns;
  s.ones.assign(n, 1.0);
  if (weighted_columns) {
    for (std::size_t i = 0; i < n; ++i) s.ones[i] = std::sqrt(s.case_weight[i]);
  }
  const std::vector<double>& cw = s.case_weight;

  s.z.assign(n * p, 0.0);
  s.centre.assign(p, 0.0);
  s.scale.assign(p, 1.0);
  s.live.assign(p, 1);
  for (std::size_t j = 0; j < p; ++j) {
    const double* raw = x + j * n;
    double mean = 0.0;
    if (spec.intercept) {
      for (std::size_t i = 0; i < n; ++i) mean += cw[i] * raw[i];
    }
    double sd = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      const double dev = raw[i] - mean;
      sd += cw[i] * dev * dev;
    }
    sd = std::sqrt(sd);
    s.centre[j] = mean;
    if (!(sd > 0.0) || !std::isfinite(sd)) {
      s.live[j] = 0;
      continue;
    }
    const double unit = spec.standardize ? sd : 1.0;
    s.scale[j] = unit;
    double* col = s.z.data() + j * n;
    if (weighted_columns) {
      for (std::size_t i = 0; i < n; ++i) col[i] = (raw[i] - mean) / unit * s.ones[i];
    } else {
      for (std::size_t i = 0; i < n; ++i) col[i] = (raw[i] - mean) / unit;
    }
  }
  s.relative_penalty = relative_penalties(spec.penalty_factor, p);
  return s;
}

// ---------------------------------------------------------------------------------------------
// The solution carried from one penalty to the next.
//
// Coefficients are on the standardised scale. Two index sets grow monotonically along the path:
// the columns offered to the descent (the strong set, plus any the optimality check took back) and
// the columns that have ever left zero. The descent iterates over the first; the inner cycles and
// the extrapolation run over the second.
struct Solution {
  double intercept = 0.0;
  std::vector<double> coef;
  std::vector<std::uint8_t> is_nonzero_ever;
  std::vector<std::size_t> nonzero_ever;
  std::vector<std::uint8_t> is_offered;
  std::vector<std::size_t> offered;

  explicit Solution(std::size_t p) : coef(p, 0.0), is_nonzero_ever(p, 0), is_offered(p, 0) {}

  void offer(std::size_t j) {
    if (is_offered[j]) return;
    is_offered[j] = 1;
    offered.push_back(j);
  }

  void mark_moved(std::size_t j) {
    if (is_nonzero_ever[j]) return;
    is_nonzero_ever[j] = 1;
    nonzero_ever.push_back(j);
  }
};

// One penalised weighted least squares problem: the quadratic's case weights, the residual already
// multiplied by them, and each offered column's curvature. A binomial case whose fitted probability
// is pinned at zero or one carries a gradient and no curvature; a Poisson case carries the fitted
// mean as its curvature. On weighted columns the case
// weights live in the design and `case_curvature` is not read.
struct Quadratic {
  std::vector<double> case_curvature;
  std::vector<double> residual;
  std::vector<double> column_curvature;
};

// ---------------------------------------------------------------------------------------------
// Anderson acceleration of the cycle over the nonzero columns (Anderson 1965; for coordinate
// descent, Bertrand and Massias 2021, "Anderson acceleration of coordinate descent", AISTATS).
//
// A cyclic descent over correlated columns converges along a few slow directions, and its last few
// iterates reveal them: the affine combination of those iterates whose successive differences come
// closest to cancelling estimates the limit. The estimate is accepted only where it lowers the
// penalised quadratic below the cycle's own iterate, so the objective is still monotone and the
// descent still stops only on a cycle that moved nothing.
class Anderson {
 public:
  static constexpr std::size_t kDepth = 5;

  void restart(std::size_t width, const Solution& sol) {
    width_ = width;
    held_ = 0;
    history_.resize((kDepth + 1) * width_);
    push(sol);
  }

  void push(const Solution& sol) {
    double* at = history_.data() + held_ * width_;
    at[0] = sol.intercept;
    for (std::size_t k = 0; k < sol.nonzero_ever.size(); ++k) at[k + 1] = sol.coef[sol.nonzero_ever[k]];
    ++held_;
  }

  bool full() const { return held_ == kDepth + 1; }
  std::size_t width() const { return width_; }

  // Replaces the solution by the extrapolated point where that lowers the objective.
  void accelerate(const Standardised& d, double lambda, double alpha, bool fit_intercept,
                  Quadratic& q, Solution& sol);

 private:
  const double* iterate(std::size_t t) const { return history_.data() + t * width_; }
  bool mixing_weights(double* c, double& c_sum) const;

  std::size_t width_ = 0;     // intercept, then the nonzero columns
  std::size_t held_ = 0;
  std::vector<double> history_;  // [kDepth + 1, width_]
  std::vector<double> point_;
  std::vector<double> shift_;    // the move to the point, on the cases
};

// The weights, up to their sum, of the combination minimising the length of the combined
// difference under weights summing to one: the Gram matrix of successive differences solved
// against a vector of ones, by Cholesky.
bool Anderson::mixing_weights(double* c, double& c_sum) const {
  constexpr std::size_t K = kDepth;
  double gram[K][K];
  for (std::size_t a = 0; a < K; ++a) {
    const double* a_from = iterate(a);
    const double* a_to = iterate(a + 1);
    for (std::size_t b = 0; b <= a; ++b) {
      const double* b_from = iterate(b);
      const double* b_to = iterate(b + 1);
      double s = 0.0;
      for (std::size_t k = 0; k < width_; ++k) s += (a_to[k] - a_from[k]) * (b_to[k] - b_from[k]);
      gram[a][b] = s;
      gram[b][a] = s;
    }
  }
  double lower[K][K] = {};
  for (std::size_t a = 0; a < K; ++a) {
    for (std::size_t b = 0; b <= a; ++b) {
      double s = gram[a][b];
      for (std::size_t k = 0; k < b; ++k) s -= lower[a][k] * lower[b][k];
      if (a != b) {
        lower[a][b] = s / lower[b][b];
        continue;
      }
      if (!(s > 0.0) || !std::isfinite(s)) return false;
      lower[a][a] = std::sqrt(s);
    }
  }
  for (std::size_t a = 0; a < K; ++a) {
    double s = 1.0;
    for (std::size_t k = 0; k < a; ++k) s -= lower[a][k] * c[k];
    c[a] = s / lower[a][a];
  }
  for (std::size_t a = K; a-- > 0;) {
    double s = c[a];
    for (std::size_t k = a + 1; k < K; ++k) s -= lower[k][a] * c[k];
    c[a] = s / lower[a][a];
  }
  c_sum = 0.0;
  for (std::size_t a = 0; a < K; ++a) c_sum += c[a];
  return std::fabs(c_sum) > 0.0 && std::isfinite(c_sum);
}

void Anderson::accelerate(const Standardised& d, double lambda, double alpha, bool fit_intercept,
                          Quadratic& q, Solution& sol) {
  constexpr std::size_t K = kDepth;
  double c[K];
  double c_sum = 0.0;
  if (!mixing_weights(c, c_sum)) return;

  const std::size_t n = d.n;
  point_.resize(width_);
  for (std::size_t k = 0; k < width_; ++k) {
    double s = 0.0;
    for (std::size_t a = 0; a < K; ++a) s += c[a] * iterate(a + 1)[k];
    point_[k] = s / c_sum;
  }

  // The change in the penalised quadratic from the cycle's iterate to the point: the smooth part
  // changes by minus the residual's projection on the move plus half the move's curvature, and the
  // penalty is read off the coefficients.
  const double* now = iterate(K);
  shift_.assign(n, 0.0);
  const double intercept_move = fit_intercept ? point_[0] - now[0] : 0.0;
  if (intercept_move != 0.0) axpy(shift_.data(), d.ones.data(), intercept_move, n);
  double penalty_change = 0.0;
  for (std::size_t k = 0; k + 1 < width_; ++k) {
    const std::size_t j = sol.nonzero_ever[k];
    const double from = now[k + 1];
    const double to = point_[k + 1];
    if (to == from) continue;
    axpy(shift_.data(), d.column(j), to - from, n);
    const double pen = lambda * d.relative_penalty[j];
    penalty_change += pen * (alpha * (std::fabs(to) - std::fabs(from)) +
                             0.5 * (1.0 - alpha) * (to * to - from * from));
  }
  const double curvature = d.weighted_columns
                               ? dot(shift_.data(), shift_.data(), n)
                               : weighted_dot(q.case_curvature.data(), shift_.data(), shift_.data(), n);
  const double change = -dot(shift_.data(), q.residual.data(), n) + 0.5 * curvature + penalty_change;
  if (!(change < 0.0)) return;

  if (fit_intercept) sol.intercept = point_[0];
  for (std::size_t k = 0; k + 1 < width_; ++k) sol.coef[sol.nonzero_ever[k]] = point_[k + 1];
  if (d.weighted_columns) {
    axpy(q.residual.data(), shift_.data(), -1.0, n);
  } else {
    weighted_axpy(q.residual.data(), q.case_curvature.data(), shift_.data(), -1.0, n);
  }
}

// ---------------------------------------------------------------------------------------------
// Cyclic coordinate descent on one penalised quadratic.
//
// A cycle visits the columns of a set in index order and then the intercept, and reports its
// largest move, measured as the column's curvature times the squared change. The descent cycles
// over every offered column; where that moved something, it cycles over the nonzero columns alone
// until they settle and then sweeps the offered set again, to see whether a column outside them
// has started to move. The set is fixed inside the inner cycles, so the iterates the acceleration
// reads all live in one space.
class Descent {
 public:
  Descent(const Standardised& d, double lambda, double alpha, bool fit_intercept, double tolerance,
          Quadratic& q, Solution& sol, Anderson& acc)
      : d_(d), lambda_(lambda), alpha_(alpha), tolerance_(tolerance), q_(q), sol_(sol), acc_(acc),
        n_(d.n), residual_(q.residual.data()), case_curvature_(q.case_curvature.data()),
        column_curvature_(q.column_curvature.data()), penalty_(d.relative_penalty.data()),
        coef_(sol.coef.data()),
        intercept_weights_(d.weighted_columns ? d.ones.data() : q.case_curvature.data()),
        intercept_curvature_(dot(d.ones.data(), intercept_weights_, d.n)),
        fit_intercept_(fit_intercept && intercept_curvature_ > 0.0) {}

  // Whether the descent settled before `passes` ran out; every cycle spends one pass.
  bool run(int& passes) {
    for (;;) {
      const double moved = cycle(sol_.offered);
      if (--passes < 0) return false;
      if (moved < tolerance_) return true;
      if (!settle_nonzero(passes)) return false;
    }
  }

 private:
  bool settle_nonzero(int& passes) {
    acc_.restart(sol_.nonzero_ever.size() + 1, sol_);
    for (;;) {
      const double moved = cycle(sol_.nonzero_ever);
      if (--passes < 0) return false;
      if (moved < tolerance_) return true;
      acc_.push(sol_);
      if (acc_.full()) {
        acc_.accelerate(d_, lambda_, alpha_, fit_intercept_, q_, sol_);
        acc_.restart(acc_.width(), sol_);
      }
    }
  }

  double cycle(const std::vector<std::size_t>& set) {
    double largest = 0.0;
    for (std::size_t k = 0; k < set.size(); ++k) update_column(set[k], largest);
    update_intercept(largest);
    return largest;
  }

  // The exact minimiser along one coordinate: the soft-thresholded partial residual projection,
  // shrunk by the ridge part of the penalty.
  void update_column(std::size_t j, double& largest) {
    const double curv = column_curvature_[j];
    if (!(curv > 0.0)) return;
    const double* col = d_.column(j);
    const double old = coef_[j];
    const double projection = dot(residual_, col, n_) + curv * old;
    const double pen = lambda_ * penalty_[j];
    const double threshold = pen * alpha_;
    double updated = 0.0;
    if (std::fabs(projection) > threshold) {
      updated = std::copysign(std::fabs(projection) - threshold, projection) /
                (curv + pen * (1.0 - alpha_));
    }
    const double change = updated - old;
    if (change == 0.0) return;
    coef_[j] = updated;
    if (d_.weighted_columns) {
      axpy(residual_, col, -change, n_);
    } else {
      weighted_axpy(residual_, case_curvature_, col, -change, n_);
    }
    largest = std::max(largest, curv * change * change);
    sol_.mark_moved(j);
  }

  void update_intercept(double& largest) {
    if (!fit_intercept_) return;
    const double change = dot(d_.ones.data(), residual_, n_) / intercept_curvature_;
    if (change == 0.0) return;
    sol_.intercept += change;
    axpy(residual_, intercept_weights_, -change, n_);
    largest = std::max(largest, intercept_curvature_ * change * change);
  }

  const Standardised& d_;
  double lambda_, alpha_, tolerance_;
  Quadratic& q_;
  Solution& sol_;
  Anderson& acc_;
  // The descent resizes none of these, so their storage is read once.
  std::size_t n_;
  double* residual_;
  const double* case_curvature_;
  const double* column_curvature_;
  const double* penalty_;
  double* coef_;
  const double* intercept_weights_;
  double intercept_curvature_;
  bool fit_intercept_;
};

// ---------------------------------------------------------------------------------------------
// The path.

void check_response(const double* y, std::size_t n, Family family, double& low, double& high) {
  // Whether there is anything to fit is read off the response's own values rather than a
  // weighted mean of them: the weights sum to one only to the last bit, so a constant response
  // would show a spread of 1e-32 and a single-outcome binomial one a mean just under one.
  low = y[0];
  high = y[0];
  for (std::size_t i = 0; i < n; ++i) {
    if (!std::isfinite(y[i])) throw Error("a penalised fit was handed a response that is not a number.");
    low = std::min(low, y[i]);
    high = std::max(high, y[i]);
  }
  if (family == Family::poisson && low < 0.0) {
    throw Error("a Poisson penalised fit takes a response of counts, none of them negative.");
  }
  if (low == high) {
    throw Error(family == Family::binomial
                    ? "a binomial penalised fit was handed a response holding one outcome."
                    : "a penalised fit was handed a response holding one value, which has "
                      "nothing to penalise against.");
  }
}

class PathFit {
 public:
  // `low` and `high` are the response's range, already checked by `check_response`.
  PathFit(const double* x, const double* y, const double* w, std::size_t n, std::size_t p,
          Family family, const PenaltySpec& spec, double low, double high)
      : spec_(spec), family_(family), n_(n), p_(p),
        d_(standardise(x, w, n, p, spec, family == Family::gaussian)), sol_(p),
        eta_(n, 0.0), gradient_(p, 0.0), step_start_(p, 0.0) {
    q_.case_curvature.assign(n, 0.0);
    q_.residual.assign(n, 0.0);
    q_.column_curvature.assign(p, 0.0);
    if (family == Family::gaussian) {
      prepare_gaussian(y);
    } else if (family == Family::binomial) {
      prepare_binomial(y, low, high);
    } else {
      prepare_poisson(y);
    }
  }

  PenaltyPath run();

 private:
  bool gaussian() const { return family_ == Family::gaussian; }

  // A Gaussian response is centred and scaled the way a column is, which puts the reported
  // penalties on the response's own scale, and its null deviance is then one.
  void prepare_gaussian(const double* y) {
    const std::vector<double>& cw = d_.case_weight;
    if (spec_.intercept) {
      for (std::size_t i = 0; i < n_; ++i) y_centre_ += cw[i] * y[i];
    }
    double spread = 0.0;
    for (std::size_t i = 0; i < n_; ++i) {
      const double dev = y[i] - y_centre_;
      spread += cw[i] * dev * dev;
    }
    y_scale_ = std::sqrt(spread);
    if (!(y_scale_ > 0.0)) {
      throw Error("a penalised fit was handed a response holding one value, which has nothing to "
                  "penalise against.");
    }
    response_.resize(n_);
    target_.resize(n_);
    for (std::size_t i = 0; i < n_; ++i) {
      response_[i] = (y[i] - y_centre_) / y_scale_;
      target_[i] = d_.ones[i] * response_[i];
      q_.residual[i] = target_[i];
    }
    reported_null_deviance_ = spread;
    null_deviance_ = 1.0;
    for (std::size_t j = 0; j < p_; ++j) {
      if (d_.live[j]) q_.column_curvature[j] = dot(d_.column(j), d_.column(j), n_);
    }
  }

  // The null model of a binomial response is the weighted share of ones. The quadratic at it is
  // what the largest penalty is read off.
  void prepare_binomial(const double* y, double low, double high) {
    const std::vector<double>& cw = d_.case_weight;
    response_.resize(n_);
    double share = 0.0;
    for (std::size_t i = 0; i < n_; ++i) {
      if (y[i] != 0.0 && y[i] != 1.0) {
        throw Error("a binomial penalised fit takes a response holding zero and one.");
      }
      response_[i] = y[i];
      share += cw[i] * y[i];
    }
    if (low != 0.0 || high != 1.0 || !(share > 0.0) || !(share < 1.0)) {
      throw Error("a binomial penalised fit was handed a response holding one outcome.");
    }
    const double null_link = std::log(share / (1.0 - share));
    double loglik = 0.0;
    for (std::size_t i = 0; i < n_; ++i) {
      loglik += cw[i] * (response_[i] * null_link - softplus(null_link));
    }
    null_deviance_ = loglik * -2.0;
    reported_null_deviance_ = null_deviance_;
    if (spec_.intercept) sol_.intercept = std::log(share / (1.0 - share));
    for (std::size_t i = 0; i < n_; ++i) {
      q_.case_curvature[i] = cw[i] * share * (1.0 - share);
      q_.residual[i] = cw[i] * (response_[i] - share);
    }
  }

  // The null model of a Poisson response is the weighted mean count, with its log as the
  // intercept; without an intercept the null mean is one. The quadratic at it is what the largest
  // penalty is read off, as it is for the binomial response, with the mean as the curvature.
  void prepare_poisson(const double* y) {
    const std::vector<double>& cw = d_.case_weight;
    response_.assign(y, y + n_);
    double mean = 1.0;
    if (spec_.intercept) {
      mean = 0.0;
      for (std::size_t i = 0; i < n_; ++i) mean += cw[i] * y[i];
      if (!(mean > 0.0)) {
        throw Error("a Poisson penalised fit was handed a response holding no count above zero.");
      }
      sol_.intercept = std::log(mean);
    }
    double dev = 0.0;
    for (std::size_t i = 0; i < n_; ++i) dev += cw[i] * poisson_unit_deviance(y[i], mean);
    null_deviance_ = dev;
    reported_null_deviance_ = dev;
    for (std::size_t i = 0; i < n_; ++i) {
      q_.case_curvature[i] = cw[i] * mean;
      q_.residual[i] = cw[i] * (response_[i] - mean);
    }
  }

  double largest_penalty() {
    double top = 0.0;
    for (std::size_t j = 0; j < p_; ++j) {
      if (!d_.live[j]) continue;
      gradient_[j] = dot(q_.residual.data(), d_.column(j), n_);
      if (!(d_.relative_penalty[j] > 0.0)) continue;
      top = std::max(top, std::fabs(gradient_[j]) / d_.relative_penalty[j]);
    }
    return top / std::max(spec_.alpha, kMixingFloor);
  }

  // The penalties on the standardised response's scale: geometric from the largest down to its
  // ratio, or the supplied ones in descending order.
  std::vector<double> penalties(double top) const {
    std::vector<double> out;
    if (!spec_.lambda.empty()) {
      out = spec_.lambda;
      std::sort(out.begin(), out.end(), std::greater<double>());
      for (double& l : out) l /= y_scale_;
      return out;
    }
    if (spec_.n_lambda < 1) throw Error("a penalised path holds at least one penalty.");
    double ratio = spec_.lambda_min_ratio;
    if (!(ratio > 0.0)) ratio = n_ > p_ ? 1e-4 : 1e-2;
    if (!(ratio < 1.0)) throw Error("a penalty path's smallest ratio is below one.");
    out.resize(static_cast<std::size_t>(spec_.n_lambda));
    const double factor =
        spec_.n_lambda > 1 ? std::pow(ratio, 1.0 / static_cast<double>(spec_.n_lambda - 1)) : 1.0;
    double l = top;
    for (double& at : out) {
      at = l;
      l *= factor;
    }
    return out;
  }

  // The sequential strong rule: a column whose gradient at the penalty just fitted exceeds
  // alpha * (2 lambda - lambda_previous), times its penalty factor, is offered. The offered set
  // only grows, so the gradients this reads are exactly the ones the optimality check at the
  // previous penalty computed, and the path makes one sweep over the columns per penalty.
  void screen(double lambda, double previous) {
    const double bound = spec_.alpha * (2.0 * lambda - previous);
    const std::size_t before = sol_.offered.size();
    for (std::size_t j = 0; j < p_; ++j) {
      if (!d_.live[j] || sol_.is_offered[j]) continue;
      const double f = d_.relative_penalty[j];
      if (!(f > 0.0) || std::fabs(gradient_[j]) > f * bound) sol_.offer(j);
    }
    if (sol_.offered.size() != before) std::sort(sol_.offered.begin(), sol_.offered.end());
  }

  // The Karush-Kuhn-Tucker check over the columns the screen left out: a column whose gradient
  // exceeds its threshold is not at zero at the optimum and is offered. Whether any was.
  bool admit_violators(double lambda) {
    bool any = false;
    for (std::size_t j = 0; j < p_; ++j) {
      if (!d_.live[j] || sol_.is_offered[j]) continue;
      gradient_[j] = dot(q_.residual.data(), d_.column(j), n_);
      if (std::fabs(gradient_[j]) > d_.relative_penalty[j] * spec_.alpha * lambda) {
        sol_.offer(j);
        any = true;
      }
    }
    if (any) std::sort(sol_.offered.begin(), sol_.offered.end());
    return any;
  }

  // The linear predictor on the design's own columns, so on weighted columns it carries the root
  // of each case weight.
  void predict_link() {
    for (std::size_t i = 0; i < n_; ++i) eta_[i] = sol_.intercept * d_.ones[i];
    for (std::size_t j : sol_.nonzero_ever) {
      const double b = sol_.coef[j];
      if (b != 0.0) axpy(eta_.data(), d_.column(j), b, n_);
    }
  }

  // A residual updated in place along many warm starts drifts from the one its coefficients imply,
  // so it is rebuilt from them.
  void rebuild_gaussian_residual() {
    predict_link();
    for (std::size_t i = 0; i < n_; ++i) q_.residual[i] = target_[i] - eta_[i];
  }

  // The quadratic at the current coefficients. Binomial: the working weights p(1 - p), and the
  // residual y - p times the case weight; a probability within `prob_floor` of zero or one is
  // pinned there and carries no curvature. Poisson: the working weights are the mean, and the
  // residual is y minus it times the case weight.
  void reweight() {
    predict_link();
    const std::vector<double>& cw = d_.case_weight;
    if (family_ == Family::poisson) {
      for (std::size_t i = 0; i < n_; ++i) {
        const double mu = std::exp(std::min(std::max(eta_[i], -kLinkBound), kLinkBound));
        q_.case_curvature[i] = cw[i] * mu;
        q_.residual[i] = cw[i] * (response_[i] - mu);
      }
      return;
    }
    for (std::size_t i = 0; i < n_; ++i) {
      const double link = std::min(std::max(eta_[i], -kLinkBound), kLinkBound);
      double prob = 1.0 / (1.0 + std::exp(-link));
      double curv = prob * (1.0 - prob);
      if (prob < spec_.prob_floor) {
        prob = 0.0;
        curv = 0.0;
      } else if (prob > 1.0 - spec_.prob_floor) {
        prob = 1.0;
        curv = 0.0;
      }
      q_.case_curvature[i] = cw[i] * curv;
      q_.residual[i] = cw[i] * (response_[i] - prob);
    }
  }

  bool descend(double lambda) {
    Descent descent(d_, lambda, spec_.alpha, spec_.intercept, tolerance_, q_, sol_, acc_);
    return descent.run(passes_left_);
  }

  // Reweighted least squares at one penalty, under the binomial or the Poisson family. A step is settled when it moved nothing: the
  // coefficients it started from against those it reached, over the nonzero columns and the
  // intercept, each weighted by its curvature, on the scale the descent stops at. The step ends
  // on the reweighting the next would open with, so the residual is current for the check that
  // follows.
  bool fit_reweighted(double lambda) {
    for (int step = 1;; ++step) {
      const double intercept_from = sol_.intercept;
      for (std::size_t j : sol_.nonzero_ever) step_start_[j] = sol_.coef[j];
      for (std::size_t j : sol_.offered) {
        q_.column_curvature[j] = weighted_dot(q_.case_curvature.data(), d_.column(j), d_.column(j), n_);
      }
      if (!descend(lambda)) return false;
      reweight();
      const double intercept_move = sol_.intercept - intercept_from;
      double moved = sum_of(q_.case_curvature.data(), n_) * intercept_move * intercept_move;
      for (std::size_t j : sol_.nonzero_ever) {
        const double change = sol_.coef[j] - step_start_[j];
        moved = std::max(moved, q_.column_curvature[j] * change * change);
      }
      if (moved < tolerance_) return true;
      if (step >= spec_.max_irls) return false;
    }
  }

  // The fit at one penalty, repeated while the optimality check finds columns the screen missed.
  bool fit_at(double lambda) {
    for (;;) {
      if (gaussian() ? !descend(lambda) : !fit_reweighted(lambda)) return false;
      if (!admit_violators(lambda)) return true;
      if (gaussian()) rebuild_gaussian_residual();
    }
  }

  double deviance() const {
    if (gaussian()) return dot(q_.residual.data(), q_.residual.data(), n_);
    const std::vector<double>& cw = d_.case_weight;
    if (family_ == Family::poisson) {
      double dev = 0.0;
      for (std::size_t i = 0; i < n_; ++i) {
        const double mu = std::exp(std::min(std::max(eta_[i], -kLinkBound), kLinkBound));
        dev += cw[i] * poisson_unit_deviance(response_[i], mu);
      }
      return dev;
    }
    double loglik = 0.0;
    for (std::size_t i = 0; i < n_; ++i) loglik += cw[i] * (response_[i] * eta_[i] - softplus(eta_[i]));
    return loglik * -2.0;
  }

  // The coefficients back on the scale the columns were handed over in.
  void record(PenaltyPath& out, double lambda, std::int32_t nonzero, double explained) {
    double shift = y_centre_;
    const std::size_t at = out.beta.size();
    out.beta.resize(at + p_);
    double* beta = out.beta.data() + at;
    for (std::size_t j = 0; j < p_; ++j) {
      const double b = sol_.coef[j] * y_scale_ / d_.scale[j];
      beta[j] = b;
      shift -= b * d_.centre[j];
    }
    out.a0.push_back(sol_.intercept * y_scale_ + shift);
    out.lambda.push_back(lambda * y_scale_);
    out.df.push_back(nonzero);
    out.dev_ratio.push_back(explained);
  }

  const PenaltySpec& spec_;
  Family family_;
  std::size_t n_, p_;
  Standardised d_;
  Solution sol_;
  Quadratic q_;
  Anderson acc_;
  std::vector<double> response_;    // on the scale the descent reads
  std::vector<double> target_;      // Gaussian: the response times the root case weight
  std::vector<double> eta_;
  std::vector<double> gradient_;
  std::vector<double> step_start_;  // coefficients a reweighting step started from
  double y_centre_ = 0.0, y_scale_ = 1.0;
  double null_deviance_ = 0.0;           // on the scale the fit runs at
  double reported_null_deviance_ = 0.0;  // on the response's own scale
  double tolerance_ = 0.0;
  int passes_left_ = 0;
};

PenaltyPath PathFit::run() {
  const std::size_t max_active = spec_.max_active == 0 ? p_ : spec_.max_active;
  const double top = largest_penalty();
  const std::vector<double> path = penalties(top);
  const bool derived = spec_.lambda.empty();

  // The descent stops on a move small against the null deviance it is fitting; the Gaussian
  // response is scaled to a null deviance of one, so the two families read the threshold alike.
  tolerance_ = spec_.thresh * null_deviance_;
  passes_left_ = spec_.max_pass;
  if (!gaussian()) reweight();

  PenaltyPath out;
  out.n_column = p_;
  out.null_deviance = reported_null_deviance_;
  out.family = family_;
  double previous_explained = -std::numeric_limits<double>::infinity();
  std::vector<double> explained_by_point;
  double previous_lambda = top;

  for (std::size_t k = 0; k < path.size(); ++k) {
    const double lambda = path[k];
    screen(lambda, previous_lambda);

    // A penalty the fit did not settle at ends the path, and the points before it are returned.
    // With none before it there is nothing to return.
    if (!fit_at(lambda)) {
      if (out.lambda.empty()) {
        throw Error(gaussian()
                        ? "a penalised fit did not settle at the first penalty of its path inside "
                          "its pass budget."
                        : "a penalised fit under the binomial or Poisson family did not settle at "
                          "the first penalty of its path.");
      }
      out.stalled = static_cast<std::int32_t>(k) + 1;
      break;
    }
    // A binomial fit ends on a reweighting, which rebuilds its residual; a Gaussian one is rebuilt
    // here, once, for the deviance and for the next penalty to open on.
    if (gaussian()) rebuild_gaussian_residual();

    std::int32_t nonzero = 0;
    for (double b : sol_.coef) nonzero += b != 0.0 ? 1 : 0;
    if (static_cast<std::size_t>(nonzero) > max_active && k > 0) break;

    const double explained = null_deviance_ > 0.0 ? 1.0 - deviance() / null_deviance_ : 0.0;
    record(out, lambda, nonzero, explained);

    explained_by_point.push_back(explained);
    if (derived && static_cast<int>(k) + 1 >= spec_.min_lambda) {
      if (explained > spec_.dev_max) break;
      // A step that explains almost nothing more ends the path. The Gaussian family reads the
      // share gained relative to the deviance explained so far and the binomial one absolutely,
      // the convention the fixtures pin. The Poisson family reads the share gained over the last
      // `min_lambda - 1` steps, relative to the deviance explained now, against ten times `fdev`.
      if (spec_.fdev > 0.0) {
        if (family_ == Family::poisson) {
          const std::size_t span = static_cast<std::size_t>(spec_.min_lambda) - 1;
          const double before = explained_by_point[k - span];
          if ((explained - before) / explained < 10.0 * spec_.fdev) break;
        } else {
          const double enough = gaussian() ? spec_.fdev * std::fabs(explained) : spec_.fdev;
          if (explained - previous_explained < enough) break;
        }
      }
    }
    previous_explained = explained;
    previous_lambda = lambda;
  }
  if (out.lambda.empty()) throw Error("a penalised path fitted no penalty.");
  out.passes = spec_.max_pass - std::max(passes_left_, 0);
  return out;
}

// ---------------------------------------------------------------------------------------------
// Cross-validation.

// The rows `rows` of a column-major matrix with `n` rows and `p` columns.
std::vector<double> take_rows(const double* x, std::size_t n, std::size_t p,
                              const std::vector<std::size_t>& rows) {
  const std::size_t m = rows.size();
  std::vector<double> out(m * p);
  for (std::size_t j = 0; j < p; ++j) {
    for (std::size_t a = 0; a < m; ++a) out[a + j * m] = x[rows[a] + j * n];
  }
  return out;
}

struct HeldOutFold {
  std::vector<std::size_t> train, test;
  PenaltyPath fit;
  std::exception_ptr failure;
};

std::vector<HeldOutFold> split_folds(const std::int32_t* fold, std::size_t n, std::size_t k) {
  std::vector<HeldOutFold> folds(k);
  for (std::size_t i = 0; i < n; ++i) {
    const std::int32_t f = fold[i];
    if (f < 0 || static_cast<std::size_t>(f) >= k) {
      throw Error("a fold index of a cross-validated penalty is outside the folds it declares.");
    }
    for (std::size_t g = 0; g < k; ++g) {
      (static_cast<std::size_t>(f) == g ? folds[g].test : folds[g].train).push_back(i);
    }
  }
  for (const HeldOutFold& one : folds) {
    if (one.train.empty() || one.test.empty()) {
      throw Error("a fold of a cross-validated penalty holds every unit or none of them.");
    }
  }
  return folds;
}

// Runs `lead` on this thread and `task(0 .. count - 1)` shared between it and up to
// `workers - 1` further threads. A machine that refuses another thread leaves the work to the
// threads that did start.
void run_shared(int workers, std::size_t count, const std::function<void()>& lead,
                const std::function<void(std::size_t)>& task) {
  if (workers <= 1) {
    lead();
    for (std::size_t g = 0; g < count; ++g) task(g);
    return;
  }
  std::atomic<std::size_t> next{0};
  auto drain = [&]() {
    for (std::size_t g = next.fetch_add(1); g < count; g = next.fetch_add(1)) task(g);
  };
  std::vector<std::thread> pool;
  const std::size_t extra = std::min<std::size_t>(static_cast<std::size_t>(workers) - 1, count);
  pool.reserve(extra);
  for (std::size_t t = 0; t < extra; ++t) {
    try {
      pool.emplace_back(drain);
    } catch (const std::system_error&) {
      break;
    }
  }
  std::exception_ptr failure;
  try {
    lead();
  } catch (...) {
    failure = std::current_exception();
  }
  drain();
  for (std::thread& t : pool) t.join();
  if (failure) std::rethrow_exception(failure);
}

// The deviance of one held-out case: squared error, twice the binomial negative log likelihood
// with the probability kept off zero and one, or the Poisson deviance of the predicted mean.
double held_out_deviance(Family family, double y, double predicted) {
  if (family == Family::gaussian) {
    const double e = y - predicted;
    return e * e;
  }
  if (family == Family::poisson) return poisson_unit_deviance(y, predicted);
  const double q = std::min(std::max(predicted, kHeldOutProbFloor), 1.0 - kHeldOutProbFloor);
  return -2.0 * (y * std::log(q) + (1.0 - y) * std::log(1.0 - q));
}

}  // namespace

Family family_from_name(const std::string& name) {
  if (name == "gaussian") return Family::gaussian;
  if (name == "binomial") return Family::binomial;
  if (name == "poisson") return Family::poisson;
  throw Error("a penalised fit knows the gaussian, binomial and Poisson family, not '" + name +
              "'.");
}

const char* family_name(Family f) {
  switch (f) {
    case Family::gaussian: return "gaussian";
    case Family::binomial: return "binomial";
    case Family::poisson: return "poisson";
  }
  return "gaussian";
}

PenaltyPath penalised_path(const double* x, const double* y, const double* w, std::size_t n,
                           std::size_t p, Family family, const PenaltySpec& spec) {
  if (n == 0 || p == 0) throw Error("a penalised fit needs at least one unit and one column.");
  if (!(spec.alpha >= 0.0 && spec.alpha <= 1.0)) {
    throw Error("the elastic net's mixing is between zero and one.");
  }
  double low = 0.0, high = 0.0;
  check_response(y, n, family, low, high);
  PathFit fit(x, y, w, n, p, family, spec, low, high);
  return fit.run();
}

void penalised_coef(const PenaltyPath& path, double lambda, double* a0, double* beta) {
  const std::size_t p = path.n_column;
  const std::vector<double>& l = path.lambda;
  const std::size_t points = l.size();
  if (points == 0) throw Error("a penalised path holds no penalty to read a coefficient at.");
  if (points == 1) {
    *a0 = path.a0[0];
    std::copy(path.beta.begin(), path.beta.begin() + static_cast<std::ptrdiff_t>(p), beta);
    return;
  }
  // Linear in the penalty between the two points of the path around it, clamped to the path's
  // ends; `share` is the weight on the larger penalty.
  const double at = std::min(std::max(lambda, l.back()), l.front());
  std::size_t below = 1;
  while (below < points - 1 && l[below] > at) ++below;
  const std::size_t above = below - 1;
  const double gap = l[above] - l[below];
  const double share =
      std::fabs(gap) > std::numeric_limits<double>::epsilon() ? (at - l[below]) / gap : 1.0;
  *a0 = path.a0[above] * share + path.a0[below] * (1.0 - share);
  const double* hi = path.beta.data() + above * p;
  const double* lo = path.beta.data() + below * p;
  for (std::size_t j = 0; j < p; ++j) beta[j] = hi[j] * share + lo[j] * (1.0 - share);
}

void penalised_predict(const PenaltyPath& path, double lambda, const double* x, std::size_t n,
                       double* out) {
  const std::size_t p = path.n_column;
  double a0 = 0.0;
  std::vector<double> beta(p, 0.0);
  penalised_coef(path, lambda, &a0, beta.data());
  std::fill(out, out + n, a0);
  for (std::size_t j = 0; j < p; ++j) {
    if (beta[j] != 0.0) axpy(out, x + j * n, beta[j], n);
  }
  if (path.family == Family::gaussian) return;
  for (std::size_t i = 0; i < n; ++i) {
    const double link = std::min(std::max(out[i], -kLinkBound), kLinkBound);
    out[i] = path.family == Family::binomial ? 1.0 / (1.0 + std::exp(-link)) : std::exp(link);
  }
}

PenaltyCV penalised_cv(const double* x, const double* y, const double* w, std::size_t n,
                       std::size_t p, Family family, const PenaltySpec& spec,
                       const std::int32_t* fold, std::int32_t n_fold) {
  if (n_fold < 2) throw Error("a cross-validated penalty needs at least two folds.");
  const std::size_t k = static_cast<std::size_t>(n_fold);
  std::vector<HeldOutFold> folds = split_folds(fold, n, k);
  auto weight_of = [w](std::size_t i) { return w == nullptr ? 1.0 : w[i]; };

  // Each fold is fitted along a path of its own and read at the whole-unit path's penalties. The
  // largest penalty differs between folds, so aligning them on the penalty rather than on the
  // position along the path keeps the held-out deviance a function of the penalty. The fits are
  // independent, so a run on many threads returns the numbers a run on one returns.
  PenaltyCV out;
  auto fit_fold = [&](std::size_t g) {
    HeldOutFold& one = folds[g];
    try {
      const std::vector<double> train_x = take_rows(x, n, p, one.train);
      std::vector<double> train_y(one.train.size()), train_w(one.train.size());
      for (std::size_t a = 0; a < one.train.size(); ++a) {
        train_y[a] = y[one.train[a]];
        train_w[a] = weight_of(one.train[a]);
      }
      one.fit = penalised_path(train_x.data(), train_y.data(), train_w.data(), one.train.size(), p,
                               family, spec);
    } catch (...) {
      one.failure = std::current_exception();
    }
  };
  run_shared(std::max(1, spec.threads), k,
             [&]() { out.path = penalised_path(x, y, w, n, p, family, spec); }, fit_fold);
  for (const HeldOutFold& one : folds) {
    if (one.failure) std::rethrow_exception(one.failure);
  }
  out.fold_stalled.resize(k);
  for (std::size_t g = 0; g < k; ++g) out.fold_stalled[g] = folds[g].fit.stalled;

  // Each fold's weighted mean held-out deviance at every penalty, [fold, penalty] fold fastest.
  const std::size_t points = out.path.lambda.size();
  std::vector<double> fold_weight(k, 0.0);
  std::vector<double> fold_score(k * points, 0.0);
  std::vector<double> predicted;
  for (std::size_t g = 0; g < k; ++g) {
    const HeldOutFold& one = folds[g];
    const std::size_t m = one.test.size();
    const std::vector<double> test_x = take_rows(x, n, p, one.test);
    double weight = 0.0;
    for (std::size_t i : one.test) weight += weight_of(i);
    fold_weight[g] = weight;
    predicted.resize(m);
    for (std::size_t l = 0; l < points; ++l) {
      penalised_predict(one.fit, out.path.lambda[l], test_x.data(), m, predicted.data());
      double score = 0.0;
      for (std::size_t a = 0; a < m; ++a) {
        const std::size_t i = one.test[a];
        score += weight_of(i) * held_out_deviance(family, y[i], predicted[a]);
      }
      fold_score[g + l * k] = weight > 0.0 ? score / weight : 0.0;
    }
  }

  // Mean and standard error over the folds, each fold weighted by its held-out weight: a fold is
  // one reading of the penalty.
  double all = 0.0;
  for (double fw : fold_weight) all += fw;
  out.cv_mean.assign(points, 0.0);
  out.cv_sd.assign(points, 0.0);
  for (std::size_t l = 0; l < points; ++l) {
    const double* score = fold_score.data() + l * k;
    double mean = 0.0;
    for (std::size_t g = 0; g < k; ++g) mean += fold_weight[g] * score[g];
    mean /= all;
    double spread = 0.0;
    for (std::size_t g = 0; g < k; ++g) {
      const double e = score[g] - mean;
      spread += fold_weight[g] * e * e;
    }
    out.cv_mean[l] = mean;
    out.cv_sd[l] = std::sqrt(spread / all / static_cast<double>(k - 1));
  }

  // The least held-out deviance, and the largest penalty within one standard error of it.
  out.index_min = static_cast<std::size_t>(
      std::min_element(out.cv_mean.begin(), out.cv_mean.end()) - out.cv_mean.begin());
  const double within = out.cv_mean[out.index_min] + out.cv_sd[out.index_min];
  out.index_1se = out.index_min;
  for (std::size_t l = 0; l < points; ++l) {
    if (out.cv_mean[l] <= within) {
      out.index_1se = l;
      break;
    }
  }
  return out;
}

}  // namespace timesift
