#include "ts_normal.h"

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <limits>

// Every result here is pinned to the bit through the probit fits in the fixtures, and a fused
// multiply-add rounds a Horner step differently from a multiply followed by an add.
#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

namespace timesift {
namespace detail {

namespace {

constexpr double kInvSqrt2Pi = 0.398942280401432677939946059934;  // 1 / sqrt(2 pi)
constexpr double kTwoPi = 6.283185307179586476925286766559;
constexpr double kLn2 = 0.693147180559945309417232121458;

// ---------------------------------------------------------------------------------------------
// Rational functions.
//
// Cody's approximations are ratios N(t) / D(t) of polynomials of equal degree n with a monic
// denominator whose constant terms are held back to the end:
//
//   N(t) = ((num[0] t + num[1]) t + ... + num[n-1]) t + num[n]
//   D(t) = ((      t + den[0]) t + ... + den[n-2]) t + den[n-1]
//
// `CodyRatio` returns the two finished values, because each region of the distribution combines
// them with its own prefactor. Wichura's approximations are ordinary ratios of degree-7
// polynomials written from the highest coefficient down, evaluated by `horner`.

struct CodyRatio {
  double num;
  double den;
};

template <std::size_t N>
CodyRatio cody_ratio(double t, const double (&num)[N + 1], const double (&den)[N]) {
  double n = num[0] * t;
  double d = t;
  for (std::size_t k = 1; k < N; ++k) {
    n = (n + num[k]) * t;
    d = (d + den[k - 1]) * t;
  }
  return {n + num[N], d + den[N - 1]};
}

template <std::size_t N>
double horner(double r, const double (&c)[N]) {
  double acc = c[0];
  for (std::size_t k = 1; k < N; ++k) acc = acc * r + c[k];
  return acc;
}

// ---------------------------------------------------------------------------------------------
// Lower tail: W. J. Cody (1969), "Rational Chebyshev approximations for the error function",
// Mathematics of Computation 23:631-637, in three regions of |x|, each coefficient set listed
// from the highest power down.

// |x| <= 0.67448975, in t = x^2: Phi(x) = 1/2 + x N(t) / D(t).
constexpr double kCentreNum[5] = {0.065682337918207449113, 2.2352520354606839287,
                                  161.02823106855587881, 1067.6894854603709582,
                                  18154.981253343561249};
constexpr double kCentreDen[4] = {47.20258190468824187, 976.09855173777669322,
                                  10260.932208618978205, 45507.789335026729956};

// 0.67448975 < |x| <= sqrt(32), in t = |x|: Phi(-|x|) = exp(-x^2 / 2) N(t) / D(t).
constexpr double kMiddleNum[9] = {1.0765576773720192317e-8, 0.39894151208813466764,
                                  8.8831497943883759412,    93.506656132177855979,
                                  597.27027639480026226,    2494.5375852903726711,
                                  6848.1904505362823326,    11602.651437647350124,
                                  9842.7148383839780218};
constexpr double kMiddleDen[8] = {22.266688044328115691, 235.38790178262499861,
                                  1519.377599407554805,  6485.558298266760755,
                                  18615.571640885098091, 34900.952721145977266,
                                  38912.003286093271411, 19685.429676859990727};

// |x| > sqrt(32), in t = 1 / x^2: Phi(-|x|) = exp(-x^2 / 2) (1/sqrt(2 pi) - t N(t) / D(t)) / |x|.
constexpr double kOuterNum[6] = {0.02307344176494017303, 0.21589853405795699,
                                 0.1274011611602473639,  0.022235277870649807,
                                 0.001421619193227893466, 2.9112874951168792e-5};
constexpr double kOuterDen[5] = {1.28426009614491121, 0.468238212480865118,
                                 0.0659881378689285515, 0.00378239633202758244,
                                 7.29751555083966205e-5};

constexpr double kCentreEdge = 0.67448975;
constexpr double kMiddleEdge = 5.656854249492380195206754896838;  // sqrt(32)
// Outside these the lower tail rounds to 0 or 1 in double precision.
constexpr double kLowerLimit = -38.4674;
constexpr double kUpperLimit = 8.2924;

// exp(-y^2 / 2) * factor, with y split as a multiple of 1/16 plus a remainder so that the large
// part of the square is formed exactly (Cody's device against the loss of relative accuracy in
// the Gaussian factor of the tail).
double gaussian_factor(double y, double factor) {
  const double coarse = std::ldexp(std::trunc(std::ldexp(y, 4)), -4);
  const double rest = (y - coarse) * (y + coarse);
  return std::exp(-coarse * std::ldexp(coarse, -1)) * std::exp(-std::ldexp(rest, -1)) * factor;
}

// ---------------------------------------------------------------------------------------------
// Quantile: M. J. Wichura (1988), "Algorithm AS 241: The percentage points of the normal
// distribution", Applied Statistics 37:477-484 (PPND16), each set listed from the highest power
// down.

// |p - 1/2| <= 0.425, in r = 0.180625 - (p - 1/2)^2: x = (p - 1/2) A(r) / B(r).
constexpr double kBodyNum[8] = {2509.0809287301226727, 33430.575583588128105,
                                67265.770927008700853, 45921.953931549871457,
                                13731.693765509461125, 1971.5909503065514427,
                                133.14166789178437745, 3.387132872796366608};
constexpr double kBodyDen[8] = {5226.495278852854561,  28729.085735721942674,
                                39307.89580009271061,  21213.794301586595867,
                                5394.1960214247511077, 687.1870074920579083,
                                42.313330701600911252, 1.};

// s = sqrt(-log min(p, 1 - p)) <= 5, in r = s - 1.6.
constexpr double kNearNum[8] = {7.7454501427834140764e-4, .0227238449892691845833,
                                .24178072517745061177,    1.27045825245236838258,
                                3.64784832476320460504,   5.7694972214606914055,
                                4.6303378461565452959,    1.42343711074968357734};
constexpr double kNearDen[8] = {1.05075007164441684324e-9, 5.475938084995344946e-4,
                                .0151986665636164571966,   .14810397642748007459,
                                .68976733498510000455,     1.6763848301838038494,
                                2.05319162663775882187,    1.};

// 5 < s <= 27, in r = s - 5.
constexpr double kFarNum[8] = {2.01033439929228813265e-7, 2.71155556874348757815e-5,
                               .0012426609473880784386,   .026532189526576123093,
                               .29656057182850489123,     1.7848265399172913358,
                               5.4637849111641143699,     6.6579046435011037772};
constexpr double kFarDen[8] = {2.04426310338993978564e-15, 1.4215117583164458887e-7,
                               1.8463183175100546818e-5,   7.868691311456132591e-4,
                               .0148753612908506148525,    .13692988092273580531,
                               .59983220655588793769,      1.};

// The quantile of a tail probability p below exp(-729), past the range AS 241 covers.
//
// With x the quantile and u = x^2, the tail satisfies p = phi(x) m(x) / x, m the Mills-ratio series
// 1 - 1/u + 3/u^2 - ..., so
//
//   u = -2 log p - log(2 pi u) + 2 log m(x),
//
// solved by fixed-point iteration from u = -2 log p. The j-th iterate carries the series truncated
// at order j, in the continued-fraction form
//
//   2 log(1 - w_j / (2 + u)),   w_j = e_{j-1} - (e_{j-2} - ... / (8 + u)) / (6 + u)) / (4 + u),
//
// with e = 1, 1, 5, 9, and at first order the linear term -2 / (2 + u). The closer p is to zero the
// better the leading terms already are, so fewer iterates are taken as s = sqrt(-log p) grows.
constexpr double kMillsCoefficients[4] = {1., 1., 5., 9.};

double mills_correction(int order, double u) {
  if (order == 1) return -2. / (2. + u);
  double w = kMillsCoefficients[order - 1];
  for (int j = order - 1; j >= 1; --j) w = kMillsCoefficients[j - 1] - w / (2. * j + 2. + u);
  return 2 * std::log1p(-w / (2. + u));
}

int mills_order(double s) {
  if (s >= 36000.) return 0;
  if (s >= 840.) return 1;
  if (s >= 109.) return 2;
  if (s >= 55.) return 3;
  return 4;
}

double extreme_quantile(double log_p, double s) {
  const double two_log = -std::ldexp(log_p, 1);
  double u = two_log - std::log(kTwoPi * two_log);
  const int orders = mills_order(s);
  for (int j = 1; j <= orders; ++j) u = two_log - std::log(kTwoPi * u) + mills_correction(j, u);
  return std::sqrt(u);
}

// ---------------------------------------------------------------------------------------------
// Density.

// Past this |x| the density is below the smallest subnormal double.
double density_underflow() {
  static const double edge = std::sqrt(-2 * kLn2 * (DBL_MIN_EXP + 1 - DBL_MANT_DIG));
  return edge;
}

}  // namespace

double pnorm(double x) {
  if (std::isnan(x)) return x;
  const double y = std::fabs(x);

  if (y <= kCentreEdge) {
    // Below half an ulp of one the quadratic terms cannot reach the sum, and t = 0 drops them.
    const double t = y > DBL_EPSILON * 0.5 ? x * x : 0.0;
    const CodyRatio f = cody_ratio<4>(t, kCentreNum, kCentreDen);
    return 0.5 + x * f.num / f.den;
  }

  double factor;
  if (y <= kMiddleEdge) {
    const CodyRatio f = cody_ratio<8>(y, kMiddleNum, kMiddleDen);
    factor = f.num / f.den;
  } else if (kLowerLimit < x && x < kUpperLimit) {
    const double t = 1.0 / (x * x);
    const CodyRatio f = cody_ratio<5>(t, kOuterNum, kOuterDen);
    factor = (kInvSqrt2Pi - t * f.num / f.den) / y;
  } else {
    return x > 0.0 ? 1.0 : 0.0;
  }
  const double lower = gaussian_factor(y, factor);
  return x > 0.0 ? 1.0 - lower : lower;
}

double qnorm(double p) {
  if (std::isnan(p)) return p;
  if (p <= 0.0) return p == 0.0 ? -std::numeric_limits<double>::infinity()
                                : std::numeric_limits<double>::quiet_NaN();
  if (p >= 1.0) return p == 1.0 ? std::numeric_limits<double>::infinity()
                                : std::numeric_limits<double>::quiet_NaN();

  const double centred = p - 0.5;
  if (std::fabs(centred) <= .425) {
    const double r = .180625 - centred * centred;
    return centred * horner(r, kBodyNum) / horner(r, kBodyDen);
  }

  const double log_p = std::log(centred > 0 ? 1.0 - p : p);
  const double s = std::sqrt(-log_p);
  double magnitude;
  if (s <= 5.) {
    const double r = s - 1.6;
    magnitude = horner(r, kNearNum) / horner(r, kNearDen);
  } else if (s <= 27) {
    const double r = s - 5.;
    magnitude = horner(r, kFarNum) / horner(r, kFarDen);
  } else {
    magnitude = extreme_quantile(log_p, s);
  }
  return centred < 0.0 ? -magnitude : magnitude;
}

double dnorm(double x) {
  if (std::isnan(x)) return x;
  const double a = std::fabs(x);
  if (a < 5) return kInvSqrt2Pi * std::exp(-0.5 * a * a);
  if (a > density_underflow()) return 0.0;
  // a = hi + lo with hi on a grid of 2^-16, so hi^2 is exact and the exponent keeps its accuracy.
  const double hi = std::ldexp(std::nearbyint(std::ldexp(a, 16)), -16);
  const double lo = a - hi;
  return kInvSqrt2Pi * (std::exp(-0.5 * hi * hi) * std::exp((-0.5 * lo - hi) * lo));
}

}  // namespace detail
}  // namespace timesift
