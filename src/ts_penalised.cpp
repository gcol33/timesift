#include "ts_penalised.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>

namespace timesift {
namespace {

// glmnet's own control constants, which are part of what is being matched rather than settings of
// ours: the probability a fitted case is pinned at, the linear predictor's clamp, the floor under
// the mixing when the largest penalty is derived, and the probability the held-out deviance is
// read at.
constexpr double kProbFloor = 1e-9;
constexpr double kEtaClamp = 250.0;
constexpr double kAlphaFloor = 1e-3;
constexpr double kCVProbFloor = 1e-5;



// The inner products the descent lives on, summed into four accumulators. A single accumulator
// chains every addition on the one before it, and the descent then runs at the latency of a
// floating-point add rather than at the throughput of one; the package is compiled at the
// optimisation R sets, which neither vectorises this nor reassociates it. The order is fixed, so
// the sum is the same number on every platform.
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

double dot3(const double* a, const double* b, const double* c, std::size_t n) {
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

double total(const double* a, std::size_t n) {
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

double log1pexp(double x) {
  if (x > 0.0) return x + std::log1p(std::exp(-x));
  return std::log1p(std::exp(x));
}

// The design the descent runs over: every column centred on its weighted mean and divided by its
// weighted standard deviation, so a penalty that is one number over every column costs a column
// the same whatever it was recorded on. A column holding one value has no spread to divide by and
// is carried through at zero.
struct Design {
  std::size_t n = 0;
  std::size_t p = 0;
  std::vector<double> xt;            // [n, p] column-major
  std::vector<double> centre, scale;
  std::vector<std::uint8_t> usable;
  std::vector<double> vp;            // penalty factor, rescaled to sum to p
  std::vector<double> w;             // case weights, summing to one

  const double* column(std::size_t j) const { return xt.data() + j * n; }
};

Design build_design(const double* x, const double* w_in, std::size_t n, std::size_t p,
                    const PenaltySpec& spec) {
  Design d;
  d.n = n;
  d.p = p;
  d.w.assign(n, 0.0);
  double total = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double wi = w_in == nullptr ? 1.0 : w_in[i];
    if (!(wi >= 0.0) || !std::isfinite(wi)) {
      throw Error("a case weight of a penalised fit is negative or not a number.");
    }
    d.w[i] = wi;
    total += wi;
  }
  if (!(total > 0.0)) throw Error("a penalised fit was handed case weights that sum to zero.");
  for (std::size_t i = 0; i < n; ++i) d.w[i] /= total;

  d.xt.assign(n * p, 0.0);
  d.centre.assign(p, 0.0);
  d.scale.assign(p, 1.0);
  d.usable.assign(p, 1);
  for (std::size_t j = 0; j < p; ++j) {
    const double* col = x + j * n;
    double centre = 0.0;
    if (spec.intercept) {
      for (std::size_t i = 0; i < n; ++i) centre += d.w[i] * col[i];
    }
    double spread = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      const double z = col[i] - centre;
      spread += d.w[i] * z * z;
    }
    spread = std::sqrt(spread);
    if (!(spread > 0.0) || !std::isfinite(spread)) {
      d.usable[j] = 0;
      d.centre[j] = centre;
      d.scale[j] = 1.0;
      continue;
    }
    d.centre[j] = centre;
    d.scale[j] = spec.standardize ? spread : 1.0;
    double* out = d.xt.data() + j * n;
    for (std::size_t i = 0; i < n; ++i) out[i] = (col[i] - centre) / d.scale[j];
  }

  d.vp.assign(p, 1.0);
  if (!spec.penalty_factor.empty()) {
    if (spec.penalty_factor.size() != p) {
      throw Error("a penalised fit takes one penalty factor per column.");
    }
    double sum = 0.0;
    for (std::size_t j = 0; j < p; ++j) {
      const double f = spec.penalty_factor[j];
      if (!(f >= 0.0) || !std::isfinite(f)) {
        throw Error("a penalty factor is negative or not a number.");
      }
      d.vp[j] = f;
      sum += f;
    }
    // glmnet rescales the factors to sum to the column count, so the path a factor of one gives
    // is the path no factor gives.
    if (sum > 0.0) {
      for (std::size_t j = 0; j < p; ++j) d.vp[j] *= static_cast<double>(p) / sum;
    }
  }
  return d;
}

// The state a warm start carries from one penalty to the next: the coefficients on the
// standardised scale, which columns the descent may move at this penalty, and which have ever
// left zero.
struct Coefs {
  double a0 = 0.0;
  std::vector<double> b;
  std::vector<std::uint8_t> ever;
  std::vector<std::size_t> active;
  std::vector<std::uint8_t> offered;
  std::vector<std::size_t> candidates;

  void offer(std::size_t j) {
    if (offered[j]) return;
    offered[j] = 1;
    candidates.push_back(j);
  }
};

// One weighted least squares elastic net over the columns the caller has offered, by cyclic
// coordinate descent. `v` is the weight of the quadratic and `r` its residual already multiplied
// by that weight, so a case the family has pinned carries a gradient and no curvature, which is
// what glmnet does with a fitted probability at zero or one.
// Returns the largest move of its first sweep, which is what says whether the reweighting that
// set `v` and `r` had anything left to do: a reweighting whose first sweep moves nothing is a
// fixed point of the reweighted least squares, and that is the test the step is judged on.
double quadratic_solve(const Design& d, double lambda, double alpha, bool intercept,
                       double thresh, int& budget, const std::vector<double>& v,
                       const std::vector<double>& xv, std::vector<double>& r, Coefs& fit) {
  const std::size_t n = d.n;
  const double sv = total(v.data(), n);

  auto step = [&](std::size_t j, double& dlx) {
    if (!(xv[j] > 0.0)) return;
    const double* col = d.column(j);
    const double u = dot(r.data(), col, n) + xv[j] * fit.b[j];
    const double pen = lambda * d.vp[j];
    const double l1 = pen * alpha;
    double bj = 0.0;
    if (std::fabs(u) > l1) {
      bj = std::copysign(std::fabs(u) - l1, u) / (xv[j] + pen * (1.0 - alpha));
    }
    const double delta = bj - fit.b[j];
    if (delta == 0.0) return;
    fit.b[j] = bj;
    double* res = r.data();
    const double* weight = v.data();
    for (std::size_t i = 0; i < n; ++i) res[i] -= delta * weight[i] * col[i];
    dlx = std::max(dlx, xv[j] * delta * delta);
    if (!fit.ever[j]) {
      fit.ever[j] = 1;
      fit.active.push_back(j);
    }
  };

  auto shift = [&](double& dlx) {
    if (!intercept || !(sv > 0.0)) return;
    const double delta = total(r.data(), n) / sv;
    if (delta == 0.0) return;
    fit.a0 += delta;
    for (std::size_t i = 0; i < n; ++i) r[i] -= delta * v[i];
    dlx = std::max(dlx, sv * delta * delta);
  };

  double first = -1.0;
  for (;;) {
    double dlx = 0.0;
    for (std::size_t k = 0; k < fit.candidates.size(); ++k) step(fit.candidates[k], dlx);
    shift(dlx);
    if (first < 0.0) first = dlx;
    if (--budget < 0) throw Error("a penalised fit did not settle inside its pass budget.");
    if (dlx < thresh) break;
    // The columns that have left zero are then cycled on their own until they settle, and the
    // offered set is swept again only to see whether a column outside them has started to move.
    for (;;) {
      double inner = 0.0;
      for (std::size_t k = 0; k < fit.active.size(); ++k) {
        if (fit.offered[fit.active[k]]) step(fit.active[k], inner);
      }
      shift(inner);
      if (--budget < 0) throw Error("a penalised fit did not settle inside its pass budget.");
      if (inner < thresh) break;
    }
  }
  return first;
}

void linear_predictor(const Design& d, const Coefs& fit, std::vector<double>& eta) {
  const std::size_t n = d.n;
  eta.assign(n, fit.a0);
  for (std::size_t k = 0; k < fit.active.size(); ++k) {
    const std::size_t j = fit.active[k];
    const double bj = fit.b[j];
    if (bj == 0.0) continue;
    const double* col = d.column(j);
    double* out = eta.data();
    for (std::size_t i = 0; i < n; ++i) out[i] += bj * col[i];
  }
}

// The coefficients on the scale the columns were handed over in, from the standardised ones.
void unstandardise(const Design& d, const Coefs& fit, double y_centre, double y_scale, double& a0,
                   double* beta) {
  double shift = y_centre;
  for (std::size_t j = 0; j < d.p; ++j) {
    const double bj = fit.b[j] * y_scale / d.scale[j];
    beta[j] = bj;
    shift -= bj * d.centre[j];
  }
  a0 = fit.a0 * y_scale + shift;
}

}  // namespace

Family family_from_name(const std::string& name) {
  if (name == "gaussian") return Family::gaussian;
  if (name == "binomial") return Family::binomial;
  throw Error("a penalised fit knows the gaussian and the binomial family, not '" + name + "'.");
}

const char* family_name(Family f) {
  return f == Family::gaussian ? "gaussian" : "binomial";
}

PenaltyPath penalised_path(const double* x, const double* y, const double* w, std::size_t n,
                           std::size_t p, Family family, const PenaltySpec& spec) {
  if (n == 0 || p == 0) throw Error("a penalised fit needs at least one unit and one column.");
  if (!(spec.alpha >= 0.0 && spec.alpha <= 1.0)) {
    throw Error("the elastic net's mixing is between zero and one.");
  }
  // The response's own values decide whether there is anything to fit, not a weighted mean of
  // them: the weights sum to one only to the last bit, so a response holding one value has a
  // spread of 1e-32 rather than of zero and a binomial one holding a single outcome has a mean
  // just under one. Either would be fitted, on nothing.
  double low = y[0], high = y[0];
  for (std::size_t i = 0; i < n; ++i) {
    if (!std::isfinite(y[i])) {
      throw Error("a penalised fit was handed a response that is not a number.");
    }
    low = std::min(low, y[i]);
    high = std::max(high, y[i]);
  }
  if (low == high) {
    throw Error(family == Family::binomial
                    ? "a binomial penalised fit was handed a response holding one outcome."
                    : "a penalised fit was handed a response holding one value, which has "
                      "nothing to penalise against.");
  }

  const Design d = build_design(x, w, n, p, spec);
  const std::size_t max_active = spec.max_active == 0 ? p : spec.max_active;

  // The response on the scale the descent reads it. A Gaussian response is centred and scaled the
  // way a column is, which is what puts the reported penalties on the response's own scale; a
  // binomial one is the outcome itself.
  double y_centre = 0.0, y_scale = 1.0;
  std::vector<double> yt(n);
  double null_deviance = 0.0;
  double dev_null = 0.0;
  double pbar = 0.0;
  if (family == Family::gaussian) {
    if (spec.intercept) {
      for (std::size_t i = 0; i < n; ++i) y_centre += d.w[i] * y[i];
    }
    double spread = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      const double z = y[i] - y_centre;
      spread += d.w[i] * z * z;
    }
    y_scale = std::sqrt(spread);
    if (!(y_scale > 0.0)) {
      throw Error("a penalised fit was handed a response holding one value, which has nothing to "
                  "penalise against.");
    }
    for (std::size_t i = 0; i < n; ++i) yt[i] = (y[i] - y_centre) / y_scale;
    null_deviance = spread;
    dev_null = 1.0;
  } else {
    for (std::size_t i = 0; i < n; ++i) {
      if (y[i] != 0.0 && y[i] != 1.0) {
        throw Error("a binomial penalised fit takes a response holding zero and one.");
      }
      yt[i] = y[i];
      pbar += d.w[i] * y[i];
    }
    if (low != 0.0 || high != 1.0 || !(pbar > 0.0) || !(pbar < 1.0)) {
      throw Error("a binomial penalised fit was handed a response holding one outcome.");
    }
    const double eta0 = std::log(pbar / (1.0 - pbar));
    for (std::size_t i = 0; i < n; ++i) {
      dev_null += d.w[i] * (yt[i] * eta0 - log1pexp(eta0));
    }
    dev_null *= -2.0;
    null_deviance = dev_null;
  }

  std::vector<double> v(n), r(n), xv(p, 0.0), eta(n, 0.0), grad(p, 0.0);
  Coefs fit;
  fit.b.assign(p, 0.0);
  fit.ever.assign(p, 0);
  fit.offered.assign(p, 0);
  if (family == Family::gaussian) {
    for (std::size_t i = 0; i < n; ++i) {
      v[i] = d.w[i];
      r[i] = d.w[i] * yt[i];
    }
    // The curvature of a Gaussian fit does not move along the path, so it is read once.
    for (std::size_t j = 0; j < p; ++j) {
      if (!d.usable[j]) continue;
      xv[j] = dot3(v.data(), d.column(j), d.column(j), n);
    }
  } else {
    if (spec.intercept) fit.a0 = std::log(pbar / (1.0 - pbar));
    for (std::size_t i = 0; i < n; ++i) {
      v[i] = d.w[i] * pbar * (1.0 - pbar);
      r[i] = d.w[i] * (yt[i] - pbar);
    }
  }

  auto gradient = [&]() {
    for (std::size_t j = 0; j < p; ++j) {
      if (!d.usable[j]) continue;
      grad[j] = dot(r.data(), d.column(j), n);
    }
  };

  // The smallest penalty that leaves every coefficient at zero, from the gradient at the null
  // model. The floor under the mixing is glmnet's, and it is what gives a ridge a finite start.
  gradient();
  double lambda_max = 0.0;
  for (std::size_t j = 0; j < p; ++j) {
    if (!d.usable[j] || !(d.vp[j] > 0.0)) continue;
    lambda_max = std::max(lambda_max, std::fabs(grad[j]) / d.vp[j]);
  }
  lambda_max /= std::max(spec.alpha, kAlphaFloor);

  std::vector<double> path;
  const bool derived = spec.lambda.empty();
  if (derived) {
    if (spec.n_lambda < 1) throw Error("a penalised path holds at least one penalty.");
    double ratio = spec.lambda_min_ratio;
    if (!(ratio > 0.0)) ratio = n > p ? 1e-4 : 1e-2;
    if (!(ratio < 1.0)) throw Error("a penalty path's smallest ratio is below one.");
    path.resize(static_cast<std::size_t>(spec.n_lambda));
    const double step = spec.n_lambda > 1
                            ? std::pow(ratio, 1.0 / static_cast<double>(spec.n_lambda - 1))
                            : 1.0;
    double current = lambda_max;
    for (std::size_t k = 0; k < path.size(); ++k) {
      path[k] = current;
      current *= step;
    }
  } else {
    path = spec.lambda;
    std::sort(path.begin(), path.end(), std::greater<double>());
    for (std::size_t k = 0; k < path.size(); ++k) path[k] /= y_scale;
  }

  PenaltyPath out;
  out.n_column = p;
  out.null_deviance = null_deviance;
  out.family = family;
  std::vector<double> beta(p, 0.0);
  double previous_ratio = -std::numeric_limits<double>::infinity();
  double previous_lambda = lambda_max;
  int budget = spec.max_pass;
  // The descent stops on a move small against the deviance it is fitting, which is glmnet's
  // reading of the same number: a Gaussian response is scaled to a null deviance of one, so the
  // two families mean the same thing by it.
  const double tolerance = spec.thresh * dev_null;

  // The curvature of a binomial fit moves with every reweighting, and is read only over the
  // columns the descent may move: reading it over every column instead is where an unrestricted
  // descent spends its time.
  auto curvature = [&]() {
    for (std::size_t k = 0; k < fit.candidates.size(); ++k) {
      const std::size_t j = fit.candidates[k];
      xv[j] = dot3(v.data(), d.column(j), d.column(j), n);
    }
  };

  auto reweight = [&]() {
    linear_predictor(d, fit, eta);
    for (std::size_t i = 0; i < n; ++i) {
      const double e = std::min(std::max(eta[i], -kEtaClamp), kEtaClamp);
      double prob = 1.0 / (1.0 + std::exp(-e));
      double curve = prob * (1.0 - prob);
      if (prob < kProbFloor) {
        prob = 0.0;
        curve = 0.0;
      } else if (prob > 1.0 - kProbFloor) {
        prob = 1.0;
        curve = 0.0;
      }
      v[i] = d.w[i] * curve;
      r[i] = d.w[i] * (yt[i] - prob);
    }
  };

  for (std::size_t k = 0; k < path.size(); ++k) {
    const double lambda = path[k];

    // The residual is rebuilt from the coefficients at every penalty rather than carried forward,
    // because a residual updated in place along a hundred warm starts drifts from the one those
    // coefficients imply.
    if (family == Family::gaussian) {
      linear_predictor(d, fit, eta);
      for (std::size_t i = 0; i < n; ++i) r[i] = d.w[i] * (yt[i] - eta[i]);
    } else {
      reweight();
    }

    // Tibshirani's sequential strong rule: a column whose gradient at the penalty just fitted is
    // further than one step of the path from the threshold is offered to the descent, and the
    // rest are left out. It is a screen rather than a decision, and whatever it discards is
    // tested against the optimality condition below and taken back where it was wrong.
    gradient();
    const double bound = spec.alpha * (2.0 * lambda - previous_lambda);
    std::fill(fit.offered.begin(), fit.offered.end(), 0);
    fit.candidates.clear();
    for (std::size_t k2 = 0; k2 < fit.active.size(); ++k2) fit.offer(fit.active[k2]);
    for (std::size_t j = 0; j < p; ++j) {
      if (!d.usable[j] || fit.offered[j]) continue;
      if (!(d.vp[j] > 0.0) || std::fabs(grad[j]) > d.vp[j] * bound) fit.offer(j);
    }
    std::sort(fit.candidates.begin(), fit.candidates.end());

    for (;;) {
      if (family == Family::gaussian) {
        quadratic_solve(d, lambda, spec.alpha, spec.intercept, tolerance, budget, v, xv, r, fit);
      } else {
        for (int it = 0;; ++it) {
          reweight();
          curvature();
          if (quadratic_solve(d, lambda, spec.alpha, spec.intercept, tolerance, budget, v, xv, r,
                              fit) < tolerance) {
            break;
          }
          if (it + 1 >= spec.max_irls) {
            throw Error("a binomial penalised fit did not settle at one penalty.");
          }
        }
        reweight();
      }

      // What the screen left out, tested: a column outside the offered set whose gradient is over
      // the threshold is not at zero at the optimum, so it is taken back and the penalty refitted.
      bool recovered = false;
      for (std::size_t j = 0; j < p; ++j) {
        if (!d.usable[j] || fit.offered[j]) continue;
        if (std::fabs(dot(r.data(), d.column(j), n)) > d.vp[j] * spec.alpha * lambda) {
          fit.offer(j);
          recovered = true;
        }
      }
      if (!recovered) break;
      std::sort(fit.candidates.begin(), fit.candidates.end());
      if (family == Family::gaussian) {
        linear_predictor(d, fit, eta);
        for (std::size_t i = 0; i < n; ++i) r[i] = d.w[i] * (yt[i] - eta[i]);
      }
    }

    linear_predictor(d, fit, eta);
    std::int32_t nonzero = 0;
    for (std::size_t j = 0; j < p; ++j) {
      if (fit.b[j] != 0.0) ++nonzero;
    }
    if (static_cast<std::size_t>(nonzero) > max_active && k > 0) break;

    double dev = 0.0;
    if (family == Family::gaussian) {
      for (std::size_t i = 0; i < n; ++i) {
        const double e = yt[i] - eta[i];
        dev += d.w[i] * e * e;
      }
    } else {
      for (std::size_t i = 0; i < n; ++i) dev += d.w[i] * (yt[i] * eta[i] - log1pexp(eta[i]));
      dev *= -2.0;
    }
    const double ratio = dev_null > 0.0 ? 1.0 - dev / dev_null : 0.0;

    double a0 = 0.0;
    unstandardise(d, fit, y_centre, y_scale, a0, beta.data());
    out.a0.push_back(a0);
    out.lambda.push_back(lambda * y_scale);
    out.beta.insert(out.beta.end(), beta.begin(), beta.end());
    out.df.push_back(nonzero);
    out.dev_ratio.push_back(ratio);

    if (derived && static_cast<int>(k) + 1 >= spec.min_lambda) {
      if (ratio > spec.dev_max) break;
      // A step that explains almost nothing more ends the path. The Gaussian family reads that
      // share against the deviance explained so far and the binomial one reads it outright, which
      // is the difference glmnet's two solvers carry.
      const double gained = family == Family::gaussian ? spec.fdev * std::fabs(ratio) : spec.fdev;
      if (spec.fdev > 0.0 && ratio - previous_ratio < gained) break;
    }
    previous_ratio = ratio;
    previous_lambda = lambda;
  }
  if (out.lambda.empty()) throw Error("a penalised path fitted no penalty.");
  out.passes = spec.max_pass - budget;
  return out;
}

void penalised_coef(const PenaltyPath& path, double lambda, double* a0, double* beta) {
  const std::size_t p = path.n_column;
  const std::size_t k = path.lambda.size();
  if (k == 0) throw Error("a penalised path holds no penalty to read a coefficient at.");
  if (k == 1) {
    *a0 = path.a0[0];
    for (std::size_t j = 0; j < p; ++j) beta[j] = path.beta[j];
    return;
  }
  const double at = std::min(std::max(lambda, path.lambda.back()), path.lambda.front());
  // The two points of the path the penalty falls between, and how far it sits from the lower one,
  // which is how glmnet reads a penalty that is not a point of the path it fitted.
  std::size_t right = 1;
  while (right < k - 1 && path.lambda[right] > at) ++right;
  const std::size_t left = right - 1;
  const double span = path.lambda[left] - path.lambda[right];
  double frac = 1.0;
  if (std::fabs(span) > std::numeric_limits<double>::epsilon()) {
    frac = (at - path.lambda[right]) / span;
  }
  *a0 = path.a0[left] * frac + path.a0[right] * (1.0 - frac);
  for (std::size_t j = 0; j < p; ++j) {
    beta[j] = path.beta[left * p + j] * frac + path.beta[right * p + j] * (1.0 - frac);
  }
}

void penalised_predict(const PenaltyPath& path, double lambda, const double* x, std::size_t n,
                       double* out) {
  const std::size_t p = path.n_column;
  double a0 = 0.0;
  std::vector<double> beta(p, 0.0);
  penalised_coef(path, lambda, &a0, beta.data());
  for (std::size_t i = 0; i < n; ++i) out[i] = a0;
  for (std::size_t j = 0; j < p; ++j) {
    if (beta[j] == 0.0) continue;
    const double* col = x + j * n;
    for (std::size_t i = 0; i < n; ++i) out[i] += beta[j] * col[i];
  }
  if (path.family == Family::binomial) {
    for (std::size_t i = 0; i < n; ++i) {
      const double e = std::min(std::max(out[i], -kEtaClamp), kEtaClamp);
      out[i] = 1.0 / (1.0 + std::exp(-e));
    }
  }
}

PenaltyCV penalised_cv(const double* x, const double* y, const double* w, std::size_t n,
                       std::size_t p, Family family, const PenaltySpec& spec,
                       const std::int32_t* fold, std::int32_t n_fold) {
  if (n_fold < 2) throw Error("a cross-validated penalty needs at least two folds.");
  PenaltyCV out;
  out.path = penalised_path(x, y, w, n, p, family, spec);
  const std::size_t k = out.path.lambda.size();

  // Each fold is fitted the way the whole-unit path was, along a path of its own, and is then
  // read at the whole-unit path's penalties. A fold holds different units, so the largest penalty
  // that leaves every coefficient at zero is a different number there; aligning the folds on the
  // penalty rather than on the point of the path is what keeps the held-out deviance a function
  // of the penalty, and it is what glmnet aligns on.
  std::vector<double> fold_sum(static_cast<std::size_t>(n_fold), 0.0);
  std::vector<double> fold_mean(static_cast<std::size_t>(n_fold) * k, 0.0);
  std::vector<double> train_x, train_y, train_w, test_x, predicted;
  for (std::int32_t f = 0; f < n_fold; ++f) {
    std::vector<std::size_t> in, held_out;
    for (std::size_t i = 0; i < n; ++i) {
      if (fold[i] == f) held_out.push_back(i); else in.push_back(i);
    }
    if (in.empty() || held_out.empty()) {
      throw Error("a fold of a cross-validated penalty holds every unit or none of them.");
    }
    const std::size_t nt = in.size(), nh = held_out.size();
    train_x.assign(nt * p, 0.0);
    test_x.assign(nh * p, 0.0);
    for (std::size_t j = 0; j < p; ++j) {
      for (std::size_t a = 0; a < nt; ++a) train_x[a + j * nt] = x[in[a] + j * n];
      for (std::size_t a = 0; a < nh; ++a) test_x[a + j * nh] = x[held_out[a] + j * n];
    }
    train_y.resize(nt);
    train_w.resize(nt);
    for (std::size_t a = 0; a < nt; ++a) {
      train_y[a] = y[in[a]];
      train_w[a] = w == nullptr ? 1.0 : w[in[a]];
    }
    const PenaltyPath fit =
        penalised_path(train_x.data(), train_y.data(), train_w.data(), nt, p, family, spec);
    double weight = 0.0;
    for (std::size_t a = 0; a < nh; ++a) weight += w == nullptr ? 1.0 : w[held_out[a]];
    fold_sum[static_cast<std::size_t>(f)] = weight;
    predicted.resize(nh);
    for (std::size_t l = 0; l < k; ++l) {
      penalised_predict(fit, out.path.lambda[l], test_x.data(), nh, predicted.data());
      double held = 0.0;
      for (std::size_t a = 0; a < nh; ++a) {
        const std::size_t i = held_out[a];
        const double wi = w == nullptr ? 1.0 : w[i];
        double raw;
        if (family == Family::gaussian) {
          const double e = y[i] - predicted[a];
          raw = e * e;
        } else {
          const double q = std::min(std::max(predicted[a], kCVProbFloor), 1.0 - kCVProbFloor);
          raw = -2.0 * (y[i] * std::log(q) + (1.0 - y[i]) * std::log(1.0 - q));
        }
        held += wi * raw;
      }
      fold_mean[static_cast<std::size_t>(f) + l * static_cast<std::size_t>(n_fold)] =
          weight > 0.0 ? held / weight : 0.0;
    }
  }

  // The held-out deviance is summarised over the folds rather than over the units: a fold is one
  // reading of the penalty, and its spread over the folds is what the standard error is of.
  double total = 0.0;
  for (std::int32_t f = 0; f < n_fold; ++f) total += fold_sum[static_cast<std::size_t>(f)];
  out.cv_mean.assign(k, 0.0);
  out.cv_sd.assign(k, 0.0);
  for (std::size_t l = 0; l < k; ++l) {
    double mean = 0.0;
    for (std::int32_t f = 0; f < n_fold; ++f) {
      const std::size_t a = static_cast<std::size_t>(f);
      mean += fold_sum[a] * fold_mean[a + l * static_cast<std::size_t>(n_fold)];
    }
    mean /= total;
    double spread = 0.0;
    for (std::int32_t f = 0; f < n_fold; ++f) {
      const std::size_t a = static_cast<std::size_t>(f);
      const double e = fold_mean[a + l * static_cast<std::size_t>(n_fold)] - mean;
      spread += fold_sum[a] * e * e;
    }
    out.cv_mean[l] = mean;
    out.cv_sd[l] = std::sqrt(spread / total / static_cast<double>(n_fold - 1));
  }

  out.index_min = 0;
  for (std::size_t l = 1; l < k; ++l) {
    if (out.cv_mean[l] < out.cv_mean[out.index_min]) out.index_min = l;
  }
  const double within = out.cv_mean[out.index_min] + out.cv_sd[out.index_min];
  out.index_1se = out.index_min;
  for (std::size_t l = 0; l < k; ++l) {
    if (out.cv_mean[l] <= within) {
      out.index_1se = l;
      break;
    }
  }
  return out;
}

}  // namespace timesift
