#include "ts_envelope.h"

#include <algorithm>
#include <cmath>
#include <string>

#include "ts_core.h"
#include "ts_internal.h"

namespace timesift {
namespace {

// Hyndman & Fan's seventh sample quantile of the ascending `sorted` (`n > 0`) at probability
// `prob`: the order statistic at the 1-based position `1 + (n - 1) * prob`, and, where that
// position falls between two order statistics that differ, the straight line between them. Two
// equal neighbours give their common value exactly rather than a blend that could round away from
// it.
double sample_quantile(const std::vector<double>& sorted, double prob) {
  const double position = 1.0 + static_cast<double>(sorted.size() - 1) * prob;
  const double below = std::floor(position);
  const double above = std::ceil(position);
  const double a = sorted[static_cast<std::size_t>(below) - 1];
  const double b = sorted[static_cast<std::size_t>(above) - 1];
  if (!(position > below) || a == b) return a;
  const double t = position - below;
  return (1.0 - t) * a + t * b;
}

std::vector<std::size_t> presence_rows(const double* y, std::size_t n) {
  std::vector<std::size_t> rows;
  for (std::size_t i = 0; i < n; ++i) {
    if (y[i] != 0.0 && y[i] != 1.0) {
      throw Error("the envelope reads a response of zero and one, and row " +
                  std::to_string(i + 1) + " holds " + std::to_string(y[i]) + ".");
    }
    if (y[i] == 1.0) rows.push_back(i);
  }
  return rows;
}

}  // namespace

Envelope envelope_fit(const double* x, const double* y, std::size_t n, std::size_t p,
                      double quantile) {
  if (!(quantile >= 0.0 && quantile <= 0.5)) {
    throw Error("the envelope's quantile lies in [0, 0.5], got " + std::to_string(quantile) + ".");
  }
  detail::check_finite(x, n * p, "the envelope", "design");
  detail::check_finite(y, n, "the envelope", "response");
  const std::vector<std::size_t> present = presence_rows(y, n);
  if (present.empty()) throw Error("the envelope is drawn around at least one presence.");

  Envelope fit;
  fit.n_column = static_cast<std::int32_t>(p);
  fit.n_presence = static_cast<std::int32_t>(present.size());
  fit.lo.resize(p);
  fit.hi.resize(p);
  // The upper bound is read at the probability `1 - quantile`, formed before it is turned into a
  // position, so the two bounds are symmetric in probability.
  const double upper = 1.0 - quantile;
  std::vector<double> readings(present.size());
  for (std::size_t j = 0; j < p; ++j) {
    const double* col = x + j * n;
    std::transform(present.begin(), present.end(), readings.begin(),
                   [col](std::size_t i) { return col[i]; });
    std::sort(readings.begin(), readings.end());
    fit.lo[j] = sample_quantile(readings, quantile);
    fit.hi[j] = sample_quantile(readings, upper);
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
  std::fill(out, out + n, 1.0);
  for (std::size_t j = 0; j < p; ++j) {
    const double* col = x + j * n;
    const double lo = fit.lo[j], hi = fit.hi[j];
    for (std::size_t i = 0; i < n; ++i) {
      if (col[i] < lo || col[i] > hi) out[i] = 0.0;
    }
  }
}

}  // namespace timesift
