#ifndef TIMESIFT_TS_ADDITIVE_H
#define TIMESIFT_TS_ADDITIVE_H

#include <cstddef>
#include <cstdint>
#include <vector>

#include "ts_penalised.h"

// A generalised additive model, one smooth per column, once, for both languages.
//
// Each column enters as a thin plate regression spline (Wood 2003, Journal of the Royal
// Statistical Society B 65:95-114) of order two in one dimension: the radial function
// `|r|^3 / 12` centred on each of the column's distinct values, reduced to the `k - 2` directions
// of greatest eigenvalue of the matrix those functions make among the values, together with the
// constant and the linear function, which the penalty leaves free. The penalty is the spline's
// integrated squared second derivative, which on the reduced directions is the diagonal of their
// eigenvalues. Above `max_knots` distinct values the radial functions are centred on a subsample
// of them, drawn as R's `sample()` draws after `set.seed(1)`, because that is the basis mgcv fits.
//
// Every column of a smooth is scaled to a root mean square of one over the units, the constant is
// removed by the constraint that the smooth sums to zero over the units, and the smooth is turned
// onto the eigenvectors of its penalty, so that each smooth is `k - 2` penalised columns and one
// free linear one. The model is the intercept and every column's smooth.
//
// The coefficients at given smoothing parameters are the penalised likelihood's maximum, by
// penalised iteratively reweighted least squares (Wood 2017, Generalized Additive Models, 2nd ed.,
// sec. 6.1) over a Householder QR of the weighted design stacked on the root of the penalty. The
// smoothing parameters minimise the unbiased risk estimator under the binomial family and the
// generalised cross-validation score under the Gaussian one (Wood 2008, Journal of the Royal
// Statistical Society B 70:495-518), by Newton's method on their logarithms with the exact
// gradient and Hessian, each obtained by differentiating the penalised likelihood's score
// equations implicitly. That is mgcv's `gam(method = "GCV.Cp")` with `s(x)` per column, which is
// what biomod2 fits as `GAM`.
namespace timesift {

struct AdditiveSpec {
  Family family = Family::binomial;
  int k = 10;                  // a column's basis dimension, the constant included
  double gamma = 1.0;          // the criterion's charge per effective degree of freedom
  int max_knots = 2000;        // above this many distinct values the knots are subsampled
  double epsilon = 1e-13;      // the penalised deviance's change, relative, that ends an inner fit
  int max_irls = 200;          // reweighted least-squares steps in one inner fit
  double tol = 1e-10;          // the gradient, relative to |criterion| + mean deviance, that ends
                               // the search
  int max_outer = 200;         // Newton steps of the smoothing parameters
  int threads = 1;             // columns and responses worked at once; the fit is the same on any
  // Smoothing parameters to fit at, one per penalised term in column order and shared by the
  // responses, in place of the search. The fit is then the penalised likelihood's maximum at
  // those parameters and `score` the criterion there.
  std::vector<double> sp;
};

// Every column's smooth, shared by the responses, and one set of coefficients per response.
//
// A term is one column's contribution. Its raw basis at a value `x` is the radial function at
// every knot times `radial` [n_knot, basis - 2], then `1` and `x - shift`; its `size` columns in
// the model are that raw row times `map` [basis, size], the first `penalised` of them carrying the
// penalty. A column of two distinct values enters linearly (`basis` 2, no knots) and a column of
// one is left out.
struct Additive {
  Family family = Family::binomial;
  std::int32_t n_column = 0;
  std::int32_t n_coef = 0;          // the intercept and every term's columns
  std::vector<std::int32_t> term_column;
  std::vector<std::int32_t> term_basis;
  std::vector<std::int32_t> term_size;
  std::vector<std::int32_t> term_penalised;
  std::vector<double> term_shift;
  std::vector<std::int32_t> knot_start;    // [terms + 1] into `knots`
  std::vector<double> knots;
  std::vector<std::int32_t> radial_start;  // [terms + 1] into `radial`
  std::vector<double> radial;
  std::vector<std::int32_t> map_start;     // [terms + 1] into `map`
  std::vector<double> map;
  // Each term's penalty on its `size` columns, the diagonal of its first `penalised` of them, term
  // after term.
  std::vector<double> penalty;
  // The free columns spanned by those before them, held at zero, in coefficient order.
  std::vector<std::int32_t> aliased;
  std::int32_t n_response = 0;
  std::vector<double> beta;         // [n_coef, n_response]
  std::vector<double> sp;           // [smooth terms, n_response]
  std::vector<double> edf;          // [terms, n_response]
  std::vector<double> score;        // [n_response]
  std::vector<std::int32_t> outer;  // [n_response] Newton steps taken
  std::vector<std::int32_t> converged;  // [n_response] 1 where the search met `tol`
};

// `x` [n, p] column-major; `y` and `w` [n, r] column-major, one response and its positive case
// weights per column, the response zero and one under the binomial family.
Additive additive_fit(const double* x, std::size_t n, std::size_t p, const double* y,
                      const double* w, std::size_t r, const AdditiveSpec& spec);

// The fitted mean of every response at every row of `x` [n, p], into `out` [n, n_response].
void additive_predict(const Additive& fit, const double* x, std::size_t n, std::size_t p,
                      double* out);

}  // namespace timesift

#endif  // TIMESIFT_TS_ADDITIVE_H
