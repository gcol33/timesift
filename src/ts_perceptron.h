#ifndef TIMESIFT_TS_PERCEPTRON_H
#define TIMESIFT_TS_PERCEPTRON_H

#include <cstddef>
#include <cstdint>
#include <vector>

#include "ts_penalised.h"

// The feed-forward network with one hidden layer of logistic units (Ripley 1996, Pattern
// Recognition and Neural Networks, ch. 5; Venables and Ripley 2002, Modern Applied Statistics
// with S, sec. 8.10), once, for both languages.
//
// Each of `hidden` units takes a bias and every column, and passes the sum through the logistic
// function, held at exactly 0 below -15 and 1 above 15. The output takes a bias, every hidden unit
// and, with `skip`, every column again. Under the binomial family it is the logistic function of
// that sum, held as the hidden units are, and the fit minimises the weighted cross-entropy, a
// probability below 1e-80 read as 1e-80; under the gaussian family it is the sum itself and the
// fit minimises the weighted sum of squares; under the Poisson family it is the exponential of the
// sum and the fit minimises half the weighted deviance. `decay` times the sum of the squared
// weights, biases included, is added to each. The weights are ordered unit by unit, every hidden
// unit as its bias and then its columns, then the output as its bias, the hidden units and the
// skipped columns; they start uniform on `[-range, range]` and are fitted by the variable metric
// method of `src/ts_quasi_newton.cpp`. The objective, the order and the stopping rules are pinned
// against nnet's in the fixtures from the same starting weights.
//
// With `standardise`, each column is first centred on its mean and divided by its sample standard
// deviation over the fitting rows, one where the column is constant, and a prediction centres and
// scales by the fit's own. nnet reads the columns as given, and so does the default.
namespace timesift {

struct PerceptronSpec {
  Family family = Family::binomial;
  int hidden = 2;
  bool skip = false;
  double decay = 0.0;
  double range = 0.7;
  int max_iter = 100;
  double abs_tol = 1e-4;
  double rel_tol = 1e-8;
  bool standardise = false;
};

struct Perceptron {
  Family family = Family::binomial;
  std::int32_t n_column = 0;
  std::int32_t hidden = 0;
  bool skip = false;
  std::vector<double> centre;  // empty where the columns were read as given
  std::vector<double> scale;
  std::vector<double> weights;
  double value = 0.0;       // the objective at the weights
  std::int32_t iterations = 0;
  bool converged = true;    // false where `max_iter` was reached
};

// The number of weights a network of `hidden` units over `p` columns carries.
std::size_t perceptron_weight_count(std::size_t p, int hidden, bool skip);

// One network per response over the shared design `x` [n, p] column-major: response `s` is column
// `s` of `y` [n, r] under column `s` of the case weights `w` [n, r]. `start` holds the starting
// weights of every network where it is not empty; otherwise response `s` draws them from stream 0
// of `seeds[s]`. `threads` fit that many responses at once, and each network is the one its
// response gives fitted alone.
std::vector<Perceptron> perceptron_fit(const double* x, std::size_t n, std::size_t p,
                                       const double* y, const double* w, std::size_t r,
                                       const std::uint32_t* seeds, const PerceptronSpec& spec,
                                       int threads, const std::vector<double>& start = {});

// The fitted mean at every row of `x` [n, p].
void perceptron_predict(const Perceptron& fit, const double* x, std::size_t n, std::size_t p,
                        double* out);

}  // namespace timesift

#endif  // TIMESIFT_TS_PERCEPTRON_H
