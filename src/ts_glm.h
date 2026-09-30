#ifndef TIMESIFT_TS_GLM_H
#define TIMESIFT_TS_GLM_H

#include <cstddef>
#include <cstdint>
#include <vector>

#include "ts_penalised.h"

// R's least squares and generalised linear model, once, for every fit that reports one.
//
// A least-squares solve is LINPACK's `dqrdc2` Householder decomposition with its limited pivoting,
// which is what sets the rank R reports, followed by `dqrls`'s back substitution. A generalised
// linear model is `glm.fit`: iteratively reweighted least squares from the family's starting means,
// each step such a solve, and the deviance's relative change against `epsilon` as the stopping
// rule. The binomial family takes the logit link with R's clamp of the linear predictor at 30 either
// side, or the probit link with `make.link("probit")`'s clamp at the normal quantile of the machine
// epsilon.
namespace timesift {

// `canonical` is the identity under the gaussian family and the logit under the binomial one.
enum class Link { canonical, probit };

struct Glm {
  std::vector<double> beta;   // one per column of the design; an aliased column's is zero
  std::int32_t rank = 0;
  double deviance = 0.0;
  bool converged = false;
};

// `x` [n, q] column-major, the intercept's column included; `w` the prior weights. The probit link
// is defined under the binomial family only.
Glm glm_fit(const double* x, std::size_t n, std::size_t q, const double* y, const double* w,
            Family family, double epsilon, int max_iter, Link link = Link::canonical);

// The inverse logit as R's `family.c` takes it, the linear predictor held at 30 either side.
double logit_linkinv(double eta);

// The inverse probit as `make.link("probit")` takes it, the linear predictor held at
// `-qnorm(.Machine$double.eps)` either side.
double probit_linkinv(double eta);

namespace detail {

// Householder QR of `x` [n, p] in place. A column whose norm has fallen below `tol` times its
// original norm is moved to the end rather than pivoted by size; `k` is the rank and `jpvt` the
// original column at each position.
void dqrdc2(double* x, std::size_t n, std::size_t p, double tol, std::size_t& k,
            std::vector<double>& qraux, std::vector<std::size_t>& jpvt);

// LINPACK `dqrsl`'s pieces over a decomposition `dqrdc2` left in `qr` [n, k.. columns] with
// `qraux`, the first `k` Householder reflections applied: `y` [n] to Q'y, and to Qy.
void qr_qty(const double* qr, std::size_t n, std::size_t k, const double* qraux, double* y);
void qr_qy(const double* qr, std::size_t n, std::size_t k, const double* qraux, double* y);

// The upper triangle of the decomposition solved against its first `k` entries of `b`, in place.
// False, with `b` partly solved, where a diagonal entry is zero.
bool qr_backsolve(const double* qr, std::size_t n, std::size_t k, double* b);

// The least-squares coefficients of `b` on `x` [n, p], both overwritten, returned in the original
// column order with an aliased column's set to zero.
std::vector<double> dqrls(double* x, std::size_t n, std::size_t p, double* b, double tol,
                          std::size_t& rank);

}  // namespace detail
}  // namespace timesift

#endif  // TIMESIFT_TS_GLM_H
