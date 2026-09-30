#ifndef TIMESIFT_TS_GLM_H
#define TIMESIFT_TS_GLM_H

#include <cstddef>
#include <cstdint>
#include <vector>

#include "ts_penalised.h"

// Least squares and generalised linear models, once, for every fit that reports one.
//
// A least-squares solve is a Householder QR decomposition (Golub & Van Loan, Matrix Computations,
// ch. 5) with the limited column pivoting of LINPACK (Dongarra, Bunch, Moler & Stewart 1979) under a
// relative rank tolerance: a column is never reordered by size, only moved to the end once what is
// left of its norm falls below the tolerance times its original norm. That rule is what decides the
// rank, and so which columns a fit reports as aliased. The coefficients then come from Q'y and a
// back substitution through the triangle.
//
// A generalised linear model is fitted by iteratively reweighted least squares (McCullagh & Nelder
// 1989, sec. 2.5; Green 1984): from the family's starting means, each iteration is one such solve on
// the working response and weights, and the fit stops once the deviance's change relative to itself
// falls below `epsilon`. The binomial family takes the logit link with the linear predictor held at
// 30 either side, or the probit link held at the normal quantile of the machine epsilon, so neither
// mean reaches zero or one. These are the conventions under which the fixtures pin the fits against
// R's `glm()`.
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

// The inverse logit, the linear predictor held at 30 either side.
double logit_linkinv(double eta);

// The inverse probit, the linear predictor held at the normal quantile of the machine epsilon
// either side.
double probit_linkinv(double eta);

namespace detail {

// Householder QR of `x` [n, p] in place, with limited pivoting at relative tolerance `tol`. On
// return the upper triangle of `x` holds R, the entries below it and `qraux` the reflections (the
// leading entry of the `j`th reflection vector is `qraux[j]`, since R's diagonal occupies its
// place), `k` is the rank, and `jpvt[j]` the original index of the column now at position `j`; the
// columns from `k` on are the ones set aside as aliased.
void dqrdc2(double* x, std::size_t n, std::size_t p, double tol, std::size_t& k,
            std::vector<double>& qraux, std::vector<std::size_t>& jpvt);

// The first `k` reflections of a decomposition left in `qr` [n, ...] and `qraux` applied to `y`
// [n] in place: `qr_qty` forms Q'y, `qr_qy` forms Qy.
void qr_qty(const double* qr, std::size_t n, std::size_t k, const double* qraux, double* y);
void qr_qy(const double* qr, std::size_t n, std::size_t k, const double* qraux, double* y);

// The leading `k` by `k` upper triangle of the decomposition solved against the first `k` entries of
// `b`, in place. False, with `b` partly solved, where a diagonal entry is zero.
bool qr_backsolve(const double* qr, std::size_t n, std::size_t k, double* b);

// The least-squares coefficients of `b` on `x` [n, p], both overwritten, returned in the original
// column order with an aliased column's set to zero.
std::vector<double> dqrls(double* x, std::size_t n, std::size_t p, double* b, double tol,
                          std::size_t& rank);

}  // namespace detail
}  // namespace timesift

#endif  // TIMESIFT_TS_GLM_H
