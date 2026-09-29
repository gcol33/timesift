#ifndef TIMESIFT_TS_ENVELOPE_H
#define TIMESIFT_TS_ENVELOPE_H

#include <cstddef>
#include <cstdint>
#include <vector>

// The surface range envelope, once, for both languages.
//
// biomod2's SRE: over the units marked present, the `quantile` and `1 - quantile` quantiles of
// every column, and a unit inside the envelope where every column lies between its two, the ends
// included. The quantile is R's default (type 7), the one `quantile()` gives biomod2's `bm_SRE`.
namespace timesift {

// The type 7 quantile of `sorted` (ascending, `n > 0`) at `prob`: the value at the 1-based position
// `1 + (n - 1) * prob`, interpolated between its two neighbours where it falls between them and
// they differ.
double quantile_type7(const double* sorted, std::size_t n, double prob);

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
