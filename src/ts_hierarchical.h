#ifndef TIMESIFT_TS_HIERARCHICAL_H
#define TIMESIFT_TS_HIERARCHICAL_H

#include <cstddef>
#include <cstdint>
#include <vector>

#include "ts_core.h"

// A Bayesian logistic model with an intercept for each unit and a Gaussian-process field over the
// targets' coordinates, once, for both languages.
//
// The linear predictor of target `i` is `x_i' beta + u_g(i) + f(s_i)`: the fixed effects `beta`,
// an intercept `u_g` for the unit the target belongs to, and a field `f` at the target's place
// `s_i`. The response is Bernoulli through the logit link, each target's log likelihood scaled by
// its case weight. The priors are the ones tulpa fits under: `beta_j ~ N(0, 2.5^2)` for every
// fixed effect, the intercept included; `u_g ~ N(0, sd_u^2)` with a penalised-complexity prior on
// `sd_u` (Simpson et al. 2017, Statistical Science 32:1-28), `P(sd_u > 3) = 0.01`; and the same
// prior on the field's marginal standard deviation, with a penalised-complexity prior on its range
// (Fuglstad et al. 2019, JASA 114:445-452), `P(range < 0.2 extent) = 0.5`, on coordinates centred
// and divided by one factor, the root of the mean of the columns' variances, so distances keep
// their proportions, `extent` being the diagonal of their bounding box.
//
// The field is a Hilbert-space approximation (Solin and Sarkka 2020, Statistics and Computing
// 30:419-446; Riutort-Mayol et al. 2023, Statistics and Computing 33:17): `m` Laplacian
// eigenfunctions per axis on a box a factor `boundary` wider than the coordinates, weighted by the
// square root of the squared-exponential spectral density, or a nearest-neighbour Gaussian process
// (Datta et al. 2016, JASA 111:800-812): the Vecchia factorisation of the field over the distinct
// locations, each conditioned on its nearest `neighbours` among those before it in lexicographic
// order of the coordinates.
//
// Inference is Laplace's method (Tierney and Kadane 1986) over the latent vector, `beta`, the unit
// intercepts and the field's coefficients together, at fixed hyperparameters, and the
// hyperparameters are then integrated over (Rue, Martino and Chopin 2009, JRSS B 71:319-392): the
// mode of their log posterior is found, its curvature gives the axes of a grid of `nodes` points
// per hyperparameter `step` standard deviations apart, and each node's conditional fit is weighted
// by its marginal posterior. Without a field the only hyperparameter is the unit intercepts' sd,
// which is set at its mode, empirical Bayes, as tulpa's `mode = "eb"` does; with neither the fit is
// the posterior mode of `beta`, as `mode = "laplace"` is.
namespace timesift {

enum class Field { none, hsgp, nngp };

struct HierSpec {
  double beta_sd = 2.5;
  bool unit = false;               // an intercept for each unit
  Field field = Field::none;
  double sd_u = 3.0;               // P(a standard deviation > sd_u) = sd_alpha, the unit intercepts'
  double sd_alpha = 0.01;          // and the field's alike
  double range_fraction = 0.2;     // P(the field's range < range_fraction extent) = range_alpha
  double range_alpha = 0.5;
  int m = 6;                       // hsgp: eigenfunctions per axis
  double boundary = 1.5;           // hsgp: the box's half-width over the coordinates'
  int neighbours = 15;             // nngp: the neighbours each location is conditioned on
  int cov = 0;                     // nngp: 0 exponential, 1 Matern 3/2, 2 Matern 5/2, 3 Gaussian
  int nodes = 5;                   // grid points per hyperparameter
  double step = 1.25;              // their spacing, in standard deviations
  int max_newton = 100;            // Newton steps of one conditional fit
  double tol = 1e-10;              // a step whose largest move is below this, relatively, ends it
  int threads = 1;                 // the grid's conditional fits run at once; the fit is the same
  // Hyperparameters to fit at, in the order of `Hierarchical::theta_hat` and shared by nothing
  // else, in place of the search and the grid: the fit is then the conditional one at those
  // values, a single node of weight one.
  std::vector<double> theta;
};

// The fitted model as plain arrays. `theta` is the hyperparameters' logarithms in the order the
// model has them: the unit intercepts' sd, then the field's sd and range. Node `k` of the grid has
// hyperparameters `node_theta[k, ]`, a posterior weight `node_weight[k]`, and, with a field, the
// field's coefficients `node_field[k, ]`: the Laplacian eigenfunctions' under hsgp, the field's
// value at each distinct location under nngp.
struct Hierarchical {
  Field field = Field::none;
  std::int32_t n_column = 0;
  std::int32_t n_unit = 0;
  std::int32_t n_theta = 0;
  std::vector<double> beta;
  std::vector<double> unit_effect;
  std::vector<double> theta_hat;       // the mode of the hyperparameters' log posterior
  double log_marginal = 0.0;           // the Laplace log marginal likelihood at theta_hat
  std::int32_t n_node = 0;
  std::vector<double> node_theta;      // [node, theta], node fastest
  std::vector<double> node_weight;
  std::vector<double> node_log_post;   // each node's log marginal likelihood plus log hyperprior
  std::vector<double> node_field;      // [node, coefficient], node fastest
  std::int32_t n_field = 0;            // coefficients per node
  std::int32_t converged = 1;          // every conditional fit that carries weight settled
  // Coordinates: the centre and the factor they were divided by, and the box the basis is on.
  double centre[2] = {0.0, 0.0};
  double scale = 1.0;
  int m = 0;
  double box_centre[2] = {0.0, 0.0};
  double box_half[2] = {0.0, 0.0};
  // nngp: the distinct locations in the order the factorisation runs, and its settings.
  std::vector<double> location;        // [n_location, 2], location fastest
  std::int32_t n_location = 0;
  std::int32_t neighbours = 0;
  std::int32_t cov = 0;
};

// `x` [n, p] column-major, the intercept's column included; `y` zero or one; `w` the case weights;
// `unit` the 0-based unit each target belongs to, `n_unit` of them, null where `spec.unit` is off;
// `coords` [n, 2] column-major, null where there is no field. Throws Error for a response other
// than zero and one, a weight below zero, a non-finite value, and a unit or coordinates missing
// where the spec needs them.
Hierarchical hierarchical_fit(const double* x, std::size_t n, std::size_t p, const double* y,
                              const double* w, const std::int32_t* unit, std::size_t n_unit,
                              const double* coords, const HierSpec& spec);

// The linear predictor at `n` targets. `unit` holds each target's unit index, or -1 for a unit the
// fit did not see, whose intercept is zero; `coords` the targets' coordinates in the units they
// were fitted in.
void hierarchical_predict(const Hierarchical& fit, const double* x, std::size_t n, std::size_t p,
                          const std::int32_t* unit, const double* coords, double* eta);

}  // namespace timesift

#endif  // TIMESIFT_TS_HIERARCHICAL_H
