#ifndef TIMESIFT_TS_FDA_H
#define TIMESIFT_TS_FDA_H

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

// Flexible discriminant analysis of two classes by optimal scoring (Hastie, Tibshirani & Buja
// 1994, JASA 89:1255-1270), over a basis of multivariate adaptive regression splines (Friedman
// 1991, Annals of Statistics 19:1-67), once, for both languages. The fixtures pin it against mda's
// `fda(method = mars)` and biomod2's `FDA`.
//
// Optimal scoring gives each class one score, from the classes' weighted shares, and regresses the
// scored response on the basis. Every term the forward pass adds is a pair of hinges, a single
// hinge, or a column entering linearly, found by Friedman's running updates against an orthonormal
// basis of the terms in, at every knot his spacing rules admit for the parent in hand. It stops
// when a step lowers the residuals by less than `thresh` of them, when the residuals reach `thresh`
// of the null, when the generalised cross-validation passes ten times the null model's, or at `nk`
// terms. The backward pass drops, one at a time, the term whose t statistic is least and keeps the
// subset of least generalised cross-validation. The case weights reach the scores and the
// recalibration, not the basis, which is fitted unweighted.
//
// The fitted score is the one canonical variate. A unit's posterior is that of two normal classes of
// unit variance around the class centroids on the variate, under the classes' unweighted shares as
// priors. The posterior is then recalibrated by a probit regression of the response on it, under
// the case weights, on the fitting units, unless `calibrate` is false.
namespace timesift {

struct FdaSpec {
  int degree = 1;
  double penalty = std::numeric_limits<double>::quiet_NaN();  // NaN: 2 at degree one, 3 above
  int nk = 0;             // 0: max(21, 2 p + 1); an even count is taken one lower
  double thresh = 0.001;
  bool prune = true;
  bool calibrate = true;
  double epsilon = 1e-8;  // the recalibration's convergence
  int max_iter = 25;
  int threads = 1;        // columns searched at once; the model is the same on any number
};

struct Fda {
  std::int32_t n_column = 0;
  // The kept terms in the order the forward pass added them, the intercept first. Term `t` is the
  // product of its factors `factor_start[t]` to `factor_start[t + 1] - 1`, each a column, a
  // direction (1 for `max(0, x - cut)`, -1 for `max(0, cut - x)`) and a cut, in column order.
  std::vector<std::int32_t> factor_start;
  std::vector<std::int32_t> factor_column;
  std::vector<std::int32_t> factor_dir;
  std::vector<double> factor_cut;
  // One per kept term, by position among the kept terms.
  std::vector<double> coef;
  std::int32_t forward_terms = 0;  // terms the forward pass added, the intercept included
  double gcv = 0.0;                // of the kept subset
  // False where the scored response has no variance on the basis; every unit is then predicted
  // `mean`, the share of the second class.
  bool discriminates = true;
  double mean = 0.0;
  double direction = 1.0;          // the variate's sign
  double scale = 1.0;              // the variate's divisor, sqrt(lambda (1 - lambda))
  double centroid[2] = {0.0, 0.0};
  double prior[2] = {0.0, 0.0};
  bool calibrated = false;
  double calibration[2] = {0.0, 0.0};  // intercept and slope of the probit on the posterior
  bool converged = true;               // the recalibration's
};

// `x` [n, p] column-major, `y` zero or one with both present, `w` the case weights, all positive.
Fda fda_fit(const double* x, const double* y, const double* w, std::size_t n, std::size_t p,
            const FdaSpec& spec);

// One fit per column of `y` and `w` [n, r]. `spec.threads` fit that many responses at once, or
// search a lone response's columns; a fit is the same either way.
std::vector<Fda> fda_fits(const double* x, std::size_t n, std::size_t p, const double* y,
                          const double* w, std::size_t r, const FdaSpec& spec);

// The second class's posterior at every row of `x` [n, p], recalibrated where the fit was.
void fda_predict(const Fda& fit, const double* x, std::size_t n, std::size_t p, double* out);

}  // namespace timesift

#endif  // TIMESIFT_TS_FDA_H
