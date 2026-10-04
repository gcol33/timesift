#include "ts_perceptron.h"

#include <cmath>
#include <stdexcept>

#include "ts_quasi_newton.h"
#include "ts_trees_internal.h"

namespace timesift {

namespace {

constexpr double kSaturation = 15.0;
constexpr double kProbabilityFloor = 1e-80;

double logistic(double z) {
  if (z < -kSaturation) return 0.0;
  if (z > kSaturation) return 1.0;
  return 1.0 / (1.0 + std::exp(-z));
}

// `t log(t / u)`, zero where `t` is.
double xlog_ratio(double t, double u) {
  return t > 0.0 ? t * std::log(t / (u < kProbabilityFloor ? kProbabilityFloor : u)) : 0.0;
}

class Network {
 public:
  Network(const double* x, const double* y, const double* w, std::size_t n, std::size_t p,
          const PerceptronSpec& spec)
      : x_(x), y_(y), w_(w), n_(n), p_(p), h_(static_cast<std::size_t>(spec.hidden)),
        skip_(spec.skip), family_(spec.family), decay_(spec.decay),
        out_(h_ * (p + 1)), act_(h_) {}

  // The output's sum at row `i`, the hidden units' values left in `act_`.
  double forward(const std::vector<double>& wt, std::size_t i) {
    for (std::size_t k = 0; k < h_; ++k) {
      const double* u = wt.data() + k * (p_ + 1);
      double s = u[0];
      for (std::size_t j = 0; j < p_; ++j) s += u[j + 1] * x_[i + j * n_];
      act_[k] = logistic(s);
    }
    const double* v = wt.data() + out_;
    double s = v[0];
    for (std::size_t k = 0; k < h_; ++k) s += v[k + 1] * act_[k];
    if (skip_) {
      for (std::size_t j = 0; j < p_; ++j) s += v[h_ + 1 + j] * x_[i + j * n_];
    }
    return s;
  }

  double mean(double z) const {
    switch (family_) {
      case Family::binomial: return logistic(z);
      case Family::poisson: return std::exp(z);
      default: return z;
    }
  }

  double error(double t, double m) const {
    switch (family_) {
      case Family::binomial: return xlog_ratio(t, m) + xlog_ratio(1.0 - t, 1.0 - m);
      case Family::poisson: return xlog_ratio(t, m) - (t - m);
      default: return (m - t) * (m - t);
    }
  }

  // The derivative of one row's error with respect to the output's sum.
  double delta(double t, double m) const {
    return family_ == Family::gaussian ? 2.0 * (m - t) : m - t;
  }

  double objective(const std::vector<double>& wt) {
    double total = 0.0;
    for (std::size_t i = 0; i < n_; ++i) {
      const double m = mean(forward(wt, i));
      total += w_[i] * error(y_[i], m);
    }
    double penalty = 0.0;
    for (double v : wt) penalty += decay_ * v * v;
    return total + penalty;
  }

  // Every slope starts at its decay term and gathers the rows in order; each row's error is
  // carried back unweighted, and its case weight enters where it is added into a slope. That is the
  // order nnet sums in, which the fixtures pin to the bit, since the minimiser's path carries the
  // last bit of every slope forward.
  void gradient(const std::vector<double>& wt, std::vector<double>& g) {
    g.resize(wt.size());
    for (std::size_t a = 0; a < wt.size(); ++a) g[a] = 2.0 * decay_ * wt[a];
    const double* v = wt.data() + out_;
    double* gv = g.data() + out_;
    for (std::size_t i = 0; i < n_; ++i) {
      const double m = mean(forward(wt, i));
      const double e = delta(y_[i], m);
      const double we = w_[i] * e;
      gv[0] += we;
      for (std::size_t k = 0; k < h_; ++k) gv[k + 1] += we * act_[k];
      if (skip_) {
        for (std::size_t j = 0; j < p_; ++j) gv[h_ + 1 + j] += we * x_[i + j * n_];
      }
      for (std::size_t k = 0; k < h_; ++k) {
        const double wek = w_[i] * (e * v[k + 1] * (act_[k] * (1.0 - act_[k])));
        double* gu = g.data() + k * (p_ + 1);
        gu[0] += wek;
        for (std::size_t j = 0; j < p_; ++j) gu[j + 1] += wek * x_[i + j * n_];
      }
    }
  }

 private:
  const double* x_;
  const double* y_;
  const double* w_;
  std::size_t n_, p_, h_;
  bool skip_;
  Family family_;
  double decay_;
  std::size_t out_;
  std::vector<double> act_;
};

}  // namespace

std::size_t perceptron_weight_count(std::size_t p, int hidden, bool skip) {
  const std::size_t h = static_cast<std::size_t>(hidden);
  return h * (p + 1) + h + 1 + (skip ? p : 0);
}

Perceptron perceptron_fit(const double* x, const double* y, const double* w, std::size_t n,
                          std::size_t p, const PerceptronSpec& spec,
                          const std::vector<double>& start) {
  if (spec.hidden < 1) throw std::invalid_argument("a network needs at least one hidden unit");
  if (spec.decay < 0.0) throw std::invalid_argument("the decay cannot be negative");
  const std::size_t nw = perceptron_weight_count(p, spec.hidden, spec.skip);
  std::vector<double> wt;
  if (!start.empty()) {
    if (start.size() != nw) {
      throw std::invalid_argument("the starting weights do not match the network's size");
    }
    wt = start;
  } else {
    detail::Stream stream(spec.seed, 0u);
    wt.resize(nw);
    for (double& v : wt) v = (2.0 * stream.uniform() - 1.0) * spec.range;
  }
  Network net(x, y, w, n, p, spec);
  const QuasiNewtonResult r = variable_metric(
      wt, [&](const std::vector<double>& at) { return net.objective(at); },
      [&](const std::vector<double>& at, std::vector<double>& g) { net.gradient(at, g); },
      spec.max_iter, spec.abs_tol, spec.rel_tol);
  Perceptron fit;
  fit.family = spec.family;
  fit.n_column = static_cast<std::int32_t>(p);
  fit.hidden = spec.hidden;
  fit.skip = spec.skip;
  fit.weights = std::move(wt);
  fit.value = r.value;
  fit.iterations = r.iterations;
  fit.converged = r.converged;
  return fit;
}

void perceptron_predict(const Perceptron& fit, const double* x, std::size_t n, std::size_t p,
                        double* out) {
  if (static_cast<std::int32_t>(p) != fit.n_column) {
    throw std::invalid_argument("the new data do not have the columns the network was fitted on");
  }
  if (fit.weights.size() != perceptron_weight_count(p, fit.hidden, fit.skip)) {
    throw std::invalid_argument("the network's weights do not match its size");
  }
  PerceptronSpec spec;
  spec.family = fit.family;
  spec.hidden = fit.hidden;
  spec.skip = fit.skip;
  Network net(x, nullptr, nullptr, n, p, spec);
  for (std::size_t i = 0; i < n; ++i) out[i] = net.mean(net.forward(fit.weights, i));
}

}  // namespace timesift
