#ifndef TIMESIFT_TS_STEPWISE_H
#define TIMESIFT_TS_STEPWISE_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "ts_penalised.h"

// The stepwise generalised linear model, once, for both languages.
//
// Each fit is R's `glm.fit`: iteratively reweighted least squares from the family's starting
// means, each step a least-squares solve by LINPACK's `dqrdc2` Householder decomposition with its
// limited pivoting, which is what sets the rank R reports, and the deviance's relative change
// against `epsilon` as the stopping rule. The binomial family is the logit link with R's clamp of
// the linear predictor at 30 either side.
//
// The search is MASS's `stepAIC`. A term is a column's orthogonal polynomial, or one power of a
// column; a step compares the model as it stands, each term it holds dropped and each term it lacks
// added, and takes the lowest criterion, the model as it stands first where it ties, then the drops
// in the model's order, then the additions in column order. A term whose removal leaves the rank
// unchanged is dropped before anything else is compared, and an addition that does not raise the
// rank is not offered. A move whose fit does not settle within `max_iter` iterations is refused.
namespace timesift {

enum class StepDirection { forward, both, backward, none };
enum class StepTerms { column, power };

StepDirection step_direction_from_name(const std::string& name);
const char* step_direction_name(StepDirection d);
StepTerms step_terms_from_name(const std::string& name);
const char* step_terms_name(StepTerms t);

struct Glm {
  std::vector<double> beta;   // one per column of the design; an aliased column's is zero
  std::int32_t rank = 0;
  double deviance = 0.0;
  bool converged = false;
};

// `x` [n, q] column-major, the intercept's column included; `w` the prior weights.
Glm glm_fit(const double* x, std::size_t n, std::size_t q, const double* y, const double* w,
            Family family, double epsilon, int max_iter);

struct StepwiseSpec {
  Family family = Family::binomial;
  double max_terms = 3.0;      // terms a forward or two-way search holds at most; may be infinite
  int degree = 2;
  StepDirection direction = StepDirection::forward;
  StepTerms terms = StepTerms::column;
  double epsilon = 1e-8;
  int max_iter = 25;
  int threads = 1;             // candidate fits of one step at once
};

struct Stepwise {
  Family family = Family::binomial;
  std::int32_t n_column = 0;
  double constant = 0.0;                 // the prediction of a model holding no term
  std::vector<std::int32_t> term_column; // 0-based column each term reads
  std::vector<std::int32_t> term_power;  // 0 for an orthogonal polynomial, else the power
  std::vector<std::int32_t> term_degree; // columns the term enters as
  std::vector<double> alpha, norm2;      // the polynomial terms' recurrence, in term order:
                                         // `degree` alphas and `degree + 1` norms each
  std::vector<double> beta;              // the intercept, then each term's columns
  std::int32_t rank = 0;
  double deviance = 0.0;
  double aic = 0.0;
  bool converged = false;
  std::int32_t steps = 0;
};

// `x` [n, p] column-major, `y` the response (zero and one under the binomial family), `w` the
// prior weights, all positive.
Stepwise stepwise_fit(const double* x, const double* y, const double* w, std::size_t n,
                      std::size_t p, const StepwiseSpec& spec);

// The fitted mean at every row of `x` [n, p].
void stepwise_predict(const Stepwise& fit, const double* x, std::size_t n, std::size_t p,
                      double* out);

}  // namespace timesift

#endif  // TIMESIFT_TS_STEPWISE_H
