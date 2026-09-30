#ifndef TIMESIFT_TS_ENVELOPE_H
#define TIMESIFT_TS_ENVELOPE_H

#include <cstddef>
#include <cstdint>
#include <vector>

// The surface range envelope, once, for both languages.
//
// The climatic envelope of BIOCLIM (Busby 1991, in Margules & Austin, Nature Conservation: Cost
// Effective Biological Surveys and Data Analysis, CSIRO, 64-68): every column is bounded by two
// quantiles of its readings over the presences, `quantile` from below and `quantile` from above,
// and a unit is inside the envelope, predicted one, where every column lies within its bounds, the
// bounds included. The quantile is Hyndman & Fan's (1996, The American Statistician 50:361-365)
// seventh definition, the default of R's `quantile()`, which is what reproduces biomod2's SRE in
// the fixtures.
namespace timesift {

struct Envelope {
  std::int32_t n_column = 0;
  std::int32_t n_presence = 0;
  std::vector<double> lo, hi;   // each column's band over the presences
};

// `x` [n, p] column-major, `y` zero and one with at least one presence. `quantile` is in
// [0, 0.5].
Envelope envelope_fit(const double* x, const double* y, std::size_t n, std::size_t p,
                      double quantile);

// One where every column of a row of `x` [n, p] lies inside its band, zero elsewhere.
void envelope_predict(const Envelope& fit, const double* x, std::size_t n, std::size_t p,
                      double* out);

}  // namespace timesift

#endif  // TIMESIFT_TS_ENVELOPE_H
