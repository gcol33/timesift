#include "ts_quasi_newton.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace timesift {

namespace {

constexpr double kStepReduction = 0.2;
constexpr double kAcceptance = 1e-4;
constexpr double kRelativeTest = 10.0;

// The approximate inverse Hessian, symmetric, held as its lower triangle row by row.
class Packed {
 public:
  explicit Packed(std::size_t n) : n_(n), b_(n * (n + 1) / 2) {}

  void identity() {
    for (std::size_t i = 0; i < n_; ++i) {
      double* row = b_.data() + i * (i + 1) / 2;
      for (std::size_t j = 0; j < i; ++j) row[j] = 0.0;
      row[i] = 1.0;
    }
  }

  double at(std::size_t i, std::size_t j) const {
    return j <= i ? b_[i * (i + 1) / 2 + j] : b_[j * (j + 1) / 2 + i];
  }

  double& lower(std::size_t i, std::size_t j) { return b_[i * (i + 1) / 2 + j]; }

 private:
  std::size_t n_;
  std::vector<double> b_;
};

}  // namespace

QuasiNewtonResult variable_metric(std::vector<double>& b,
                                  const std::function<double(const std::vector<double>&)>& fn,
                                  const std::function<void(const std::vector<double>&,
                                                           std::vector<double>&)>& gr,
                                  int max_iter, double abs_tol, double rel_tol) {
  const std::size_t n = b.size();
  QuasiNewtonResult out;
  double f = fn(b);
  out.evaluations = 1;
  if (!std::isfinite(f)) {
    throw std::runtime_error("the objective at the starting point is not finite");
  }
  double fmin = f;
  if (max_iter <= 0 || n == 0) {
    out.value = f;
    out.converged = true;
    return out;
  }
  Packed B(n);
  std::vector<double> g(n), t(n), x(n), c(n);
  gr(b, g);
  int gradients = 1;
  int iter = 1;
  // `last_reset` is the gradient count at which B was last set to the identity: B is the identity
  // exactly when it equals `gradients`.
  int last_reset = gradients;
  std::size_t count = 0;
  do {
    if (last_reset == gradients) B.identity();
    for (std::size_t i = 0; i < n; ++i) {
      x[i] = b[i];
      c[i] = g[i];
    }
    double slope = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      double s = 0.0;
      for (std::size_t j = 0; j < n; ++j) s -= B.at(i, j) * g[j];
      t[i] = s;
      slope += s * g[i];
    }
    if (slope < 0.0) {
      double step = 1.0;
      bool accepted = false;
      do {
        count = 0;
        for (std::size_t i = 0; i < n; ++i) {
          b[i] = x[i] + step * t[i];
          if (kRelativeTest + x[i] == kRelativeTest + b[i]) ++count;
        }
        if (count < n) {
          f = fn(b);
          ++out.evaluations;
          accepted = std::isfinite(f) && f <= fmin + slope * step * kAcceptance;
          if (!accepted) step *= kStepReduction;
        }
      } while (!(count == n || accepted));
      const bool enough =
          f > abs_tol && std::fabs(f - fmin) > rel_tol * (std::fabs(fmin) + rel_tol);
      if (!enough) {
        count = n;
        fmin = f;
      }
      if (count < n) {
        fmin = f;
        gr(b, g);
        ++gradients;
        ++iter;
        double sc = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
          t[i] *= step;
          c[i] = g[i] - c[i];
          sc += t[i] * c[i];
        }
        if (sc > 0.0) {
          double cbc = 0.0;
          for (std::size_t i = 0; i < n; ++i) {
            double s = 0.0;
            for (std::size_t j = 0; j < n; ++j) s += B.at(i, j) * c[j];
            x[i] = s;
            cbc += s * c[i];
          }
          const double scale = 1.0 + cbc / sc;
          for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j <= i; ++j) {
              B.lower(i, j) += (scale * t[i] * t[j] - x[i] * t[j] - t[i] * x[j]) / sc;
            }
          }
        } else {
          last_reset = gradients;
        }
      } else if (last_reset < gradients) {
        count = 0;
        last_reset = gradients;
      }
    } else {
      count = 0;
      if (last_reset == gradients) {
        count = n;
      } else {
        last_reset = gradients;
      }
    }
    if (iter >= max_iter) break;
    if (gradients - last_reset > 2 * static_cast<int>(n)) last_reset = gradients;
  } while (count != n || last_reset != gradients);
  out.value = fmin;
  out.iterations = iter;
  out.converged = iter < max_iter;
  return out;
}

}  // namespace timesift
