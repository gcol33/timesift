#ifndef TIMESIFT_TS_MARS_H
#define TIMESIFT_TS_MARS_H

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "ts_penalised.h"

// Multivariate adaptive regression splines (Friedman 1991, Annals of Statistics 19:1-67), once,
// for both languages.
//
// The forward pass starts from the intercept, and each step multiplies a term already in the model
// by a pair of hinges on one column at one knot, `max(0, x - t)` and `max(0, t - x)`, choosing the
// parent, the column and the knot that most reduce the residual sum of squares of the
// least-squares fit to the scaled response. A column's knots are swept from its top down with
// Friedman's running updates of the hinge's covariances against an orthonormal basis of the terms,
// at most one knot in every `minspan` units and none among the `endspan` at either end, and a knot
// at the column's least value enters the column linearly. Fast MARS (Friedman 1993, Stanford
// technical report 110) ranks the parents by the reduction they last offered, aged by `fast_beta`
// for every step since, and searches the first `fast_k`. The pass stops at `nk` terms, when a step
// raises the R-squared by less than `thresh`, when the R-squared reaches `1 - thresh`, or when no
// term reduces the residuals.
//
// Case weights scale every term by the square root of its unit's weight, and a knot is kept where
// it lowers the weighted residual sum of squares by more than `1e-10`. The same running updates,
// against an orthonormal basis of every weighted column, give the residual sums a least-squares
// refit at each candidate knot would, at the cost of one pass per column; a hinge whose residual
// is below `1e-10` of its squared norm is taken as a combination of the terms. The terms, knots
// and coefficients are pinned against earth's in the fixtures, weighted and unweighted.
//
// The pruning pass eliminates terms backwards over an orthogonal reduction updated by plane
// rotations (Miller 1992, Algorithm AS 274, Applied Statistics 41:458-478), the intercept held
// in, and keeps the subset of least generalised cross-validation, `penalty` charged per knot. The
// kept terms are then refitted: by least squares under a squared-error head, and by the
// generalised linear model under the binomial one.
namespace timesift {

struct MarsSpec {
  Family family = Family::binomial;
  int degree = 1;
  // NaN: 2 at degree one, 3 above. -1 charges nothing for a term or a knot.
  double penalty = std::numeric_limits<double>::quiet_NaN();
  int nk = 0;                  // 0: min(200, max(20, 2 p)) + 1
  double thresh = 0.001;
  int minspan = 0;             // 0: Friedman's rule; negative: that many knots per column
  int endspan = 0;             // 0: Friedman's rule
  int fast_k = 20;             // 0: every term is a parent
  double fast_beta = 1.0;
  double adjust_endspan = 2.0; // the end span of an interaction is widened by this multiple
  bool prune = true;           // false keeps every term of the forward pass
  int nprune = 0;              // 0: no cap on the terms kept, the intercept included
  double epsilon = 1e-8;       // the binomial refit's convergence
  int max_iter = 25;
  int threads = 1;             // columns searched at once; the model is the same on any number
};

struct Mars {
  Family family = Family::binomial;
  std::int32_t n_column = 0;
  // Every term the forward pass kept, in the order it added them, the intercept first. Term `t`
  // is the product of its factors `factor_start[t]` to `factor_start[t + 1] - 1`, each a column,
  // a direction (1 for `max(0, x - cut)`, -1 for `max(0, cut - x)`, 2 for the column itself) and a
  // cut.
  std::vector<std::int32_t> factor_start;
  std::vector<std::int32_t> factor_column;
  std::vector<std::int32_t> factor_dir;
  std::vector<double> factor_cut;
  std::vector<std::int32_t> selected;  // the terms kept by the pruning pass, ascending
  std::vector<double> beta;            // one per kept term: the refit's coefficient
  // Why the forward pass stopped: 1 `nk` below three; 2 and 3 the generalised R-squared below
  // -1000 and below -10; 4 a step raising the R-squared by less than `thresh`; 5 the R-squared at
  // `1 - thresh`; 6 no term reducing the residuals; 7 `nk` reached.
  std::int32_t termcond = 0;
  double gcv = 0.0;                    // of the kept subset, on the weighted response
  bool converged = true;               // the binomial refit's
};

// `x` [n, p] column-major, `y` the response (zero and one under the binomial family), `w` the case
// weights, all positive.
Mars mars_fit(const double* x, const double* y, const double* w, std::size_t n, std::size_t p,
              const MarsSpec& spec);

// The kept terms at every row of `x` [n, p], into `out` [n, selected] column-major.
void mars_basis(const Mars& fit, const double* x, std::size_t n, std::size_t p, double* out);

// The fitted mean at every row of `x` [n, p].
void mars_predict(const Mars& fit, const double* x, std::size_t n, std::size_t p, double* out);

}  // namespace timesift

#endif  // TIMESIFT_TS_MARS_H
