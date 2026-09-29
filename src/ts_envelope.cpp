#include "ts_envelope.h"

#include <algorithm>
#include <cmath>
#include <string>

#include "ts_core.h"
#include "ts_internal.h"

namespace timesift {

double quantile_type7(const double* sorted, std::size_t n, double prob) {
  const double index = 1.0 + static_cast<double>(n - 1) * prob;
  const double lo = std::floor(index);
  const double hi = std::ceil(index);
  const double at_lo = sorted[static_cast<std::size_t>(lo) - 1];
  const double at_hi = sorted[static_cast<std::size_t>(hi) - 1];
  if (index > lo && at_hi != at_lo) {
    const double h = index - lo;
    return (1.0 - h) * at_lo + h * at_hi;
  }
  return at_lo;
}

Envelope envelope_fit(const double* x, const double* y, std::size_t n, std::size_t p,
                      double quantile) {
  if (!(quantile >= 0.0 && quantile <= 0.5)) {
    throw Error("the envelope's quantile lies in [0, 0.5], got " + std::to_string(quantile) + ".");
  }
  detail::check_finite(x, n * p, "the envelope", "design");
  detail::check_finite(y, n, "the envelope", "response");
  std::vector<std::size_t> present;
  for (std::size_t i = 0; i < n; ++i) {
    if (y[i] != 0.0 && y[i] != 1.0) {
      throw Error("the envelope reads a response of zero and one, and row " +
                  std::to_string(i + 1) + " holds " + std::to_string(y[i]) + ".");
    }
    if (y[i] == 1.0) present.push_back(i);
  }
  if (present.empty()) throw Error("the envelope is drawn around at least one presence.");

  Envelope fit;
  fit.n_column = static_cast<std::int32_t>(p);
  fit.n_presence = static_cast<std::int32_t>(present.size());
  fit.lo.resize(p);
  fit.hi.resize(p);
  // R forms the upper probability as `1 - quant` before it indexes, and so does this.
  const double upper = 1.0 - quantile;
  std::vector<double> v(present.size());
  for (std::size_t j = 0; j < p; ++j) {
    const double* col = x + j * n;
    for (std::size_t k = 0; k < present.size(); ++k) v[k] = col[present[k]];
    std::sort(v.begin(), v.end());
    fit.lo[j] = quantile_type7(v.data(), v.size(), quantile);
    fit.hi[j] = quantile_type7(v.data(), v.size(), upper);
  }
  return fit;
}

void envelope_predict(const Envelope& fit, const double* x, std::size_t n, std::size_t p,
                      double* out) {
  if (p != static_cast<std::size_t>(fit.n_column)) {
    throw Error("the envelope was drawn over " + std::to_string(fit.n_column) +
                " columns and is asked to predict over " + std::to_string(p) + ".");
  }
  detail::check_finite(x, n * p, "the envelope", "design");
  for (std::size_t i = 0; i < n; ++i) out[i] = 1.0;
  for (std::size_t j = 0; j < p; ++j) {
    const double* col = x + j * n;
    for (std::size_t i = 0; i < n; ++i) {
      if (col[i] < fit.lo[j] || col[i] > fit.hi[j]) out[i] = 0.0;
    }
  }
}

}  // namespace timesift
