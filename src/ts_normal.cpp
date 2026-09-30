#include "ts_normal.h"

#include <cfloat>
#include <cmath>
#include <limits>

// Operation for operation R's `nmath/pnorm.c`, `qnorm.c` and `dnorm.c`, lower tail and natural
// scale only. Contraction is off for the reason `ts_tree.cpp` gives.
#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

namespace timesift {
namespace detail {

namespace {

constexpr double kSqrt32 = 5.656854249492380195206754896838;
constexpr double k1Sqrt2Pi = 0.398942280401432677939946059934;
constexpr double kLn2 = 0.693147180559945309417232121458;

// exp(-x^2 / 2) * temp with x split at 1/16, so that the square is exact where it matters.
double tail(double x, double temp) {
  const double xsq = std::ldexp(std::trunc(std::ldexp(x, 4)), -4);
  const double del = (x - xsq) * (x + xsq);
  return std::exp(-xsq * std::ldexp(xsq, -1)) * std::exp(-std::ldexp(del, -1)) * temp;
}

}  // namespace

double pnorm(double x) {
  static const double a[5] = {2.2352520354606839287, 161.02823106855587881,
                              1067.6894854603709582, 18154.981253343561249,
                              0.065682337918207449113};
  static const double b[4] = {47.20258190468824187, 976.09855173777669322,
                              10260.932208618978205, 45507.789335026729956};
  static const double c[9] = {0.39894151208813466764, 8.8831497943883759412,
                              93.506656132177855979,  597.27027639480026226,
                              2494.5375852903726711,  6848.1904505362823326,
                              11602.651437647350124,  9842.7148383839780218,
                              1.0765576773720192317e-8};
  static const double d[8] = {22.266688044328115691, 235.38790178262499861,
                              1519.377599407554805,  6485.558298266760755,
                              18615.571640885098091, 34900.952721145977266,
                              38912.003286093271411, 19685.429676859990727};
  static const double p[6] = {0.21589853405795699,     0.1274011611602473639,
                              0.022235277870649807,    0.001421619193227893466,
                              2.9112874951168792e-5,   0.02307344176494017303};
  static const double q[5] = {1.28426009614491121, 0.468238212480865118,
                              0.0659881378689285515, 0.00378239633202758244,
                              7.29751555083966205e-5};
  if (std::isnan(x)) return x;
  const double eps = DBL_EPSILON * 0.5;
  const double y = std::fabs(x);
  double xnum, xden, temp;
  if (y <= 0.67448975) {
    if (y > eps) {
      const double xsq = x * x;
      xnum = a[4] * xsq;
      xden = xsq;
      for (int i = 0; i < 3; ++i) {
        xnum = (xnum + a[i]) * xsq;
        xden = (xden + b[i]) * xsq;
      }
    } else {
      xnum = xden = 0.0;
    }
    temp = x * (xnum + a[3]) / (xden + b[3]);
    return 0.5 + temp;
  }
  if (y <= kSqrt32) {
    xnum = c[8] * y;
    xden = y;
    for (int i = 0; i < 7; ++i) {
      xnum = (xnum + c[i]) * y;
      xden = (xden + d[i]) * y;
    }
    temp = (xnum + c[7]) / (xden + d[7]);
    const double cum = tail(y, temp);
    return x > 0.0 ? 1.0 - cum : cum;
  }
  if (-38.4674 < x && x < 8.2924) {
    const double xsq = 1.0 / (x * x);
    xnum = p[5] * xsq;
    xden = xsq;
    for (int i = 0; i < 4; ++i) {
      xnum = (xnum + p[i]) * xsq;
      xden = (xden + q[i]) * xsq;
    }
    temp = xsq * (xnum + p[4]) / (xden + q[4]);
    temp = (k1Sqrt2Pi - temp) / y;
    const double cum = tail(x, temp);
    return x > 0.0 ? 1.0 - cum : cum;
  }
  return x > 0.0 ? 1.0 : 0.0;
}

double qnorm(double p) {
  if (std::isnan(p)) return p;
  if (p <= 0.0) return p == 0.0 ? -std::numeric_limits<double>::infinity()
                                : std::numeric_limits<double>::quiet_NaN();
  if (p >= 1.0) return p == 1.0 ? std::numeric_limits<double>::infinity()
                                : std::numeric_limits<double>::quiet_NaN();
  const double q = p - 0.5;
  double r, val;
  if (std::fabs(q) <= .425) {
    r = .180625 - q * q;
    return q *
           (((((((r * 2509.0809287301226727 + 33430.575583588128105) * r +
                 67265.770927008700853) * r + 45921.953931549871457) * r +
               13731.693765509461125) * r + 1971.5909503065514427) * r +
             133.14166789178437745) * r + 3.387132872796366608) /
           (((((((r * 5226.495278852854561 + 28729.085735721942674) * r +
                 39307.89580009271061) * r + 21213.794301586595867) * r +
               5394.1960214247511077) * r + 687.1870074920579083) * r +
             42.313330701600911252) * r + 1.);
  }
  const double lp = std::log(q > 0 ? 1.0 - p : p);
  r = std::sqrt(-lp);
  if (r <= 5.) {
    r += -1.6;
    val = (((((((r * 7.7454501427834140764e-4 + .0227238449892691845833) * r +
                .24178072517745061177) * r + 1.27045825245236838258) * r +
              3.64784832476320460504) * r + 5.7694972214606914055) * r +
            4.6303378461565452959) * r + 1.42343711074968357734) /
          (((((((r * 1.05075007164441684324e-9 + 5.475938084995344946e-4) * r +
                .0151986665636164571966) * r + .14810397642748007459) * r +
              .68976733498510000455) * r + 1.6763848301838038494) * r +
            2.05319162663775882187) * r + 1.);
  } else if (r <= 27) {
    r += -5.;
    val = (((((((r * 2.01033439929228813265e-7 + 2.71155556874348757815e-5) * r +
                .0012426609473880784386) * r + .026532189526576123093) * r +
              .29656057182850489123) * r + 1.7848265399172913358) * r +
            5.4637849111641143699) * r + 6.6579046435011037772) /
          (((((((r * 2.04426310338993978564e-15 + 1.4215117583164458887e-7) * r +
                1.8463183175100546818e-5) * r + 7.868691311456132591e-4) * r +
              .0148753612908506148525) * r + .13692988092273580531) * r +
            .59983220655588793769) * r + 1.);
  } else {
    // min(p, 1 - p) below exp(-729): R's asymptotic expansion, reached only through a probability
    // the natural scale cannot hold apart from zero, kept so the quantile is total.
    const double s2 = -std::ldexp(lp, 1);
    const double m2pi = 6.283185307179586476925286766559;
    double x2 = s2 - std::log(m2pi * s2);
    if (r < 36000.) {
      x2 = s2 - std::log(m2pi * x2) - 2. / (2. + x2);
      if (r < 840.) {
        x2 = s2 - std::log(m2pi * x2) + 2 * std::log1p(-(1 - 1 / (4 + x2)) / (2. + x2));
        if (r < 109.) {
          x2 = s2 - std::log(m2pi * x2) +
               2 * std::log1p(-(1 - (1 - 5 / (6 + x2)) / (4. + x2)) / (2. + x2));
          if (r < 55.) {
            x2 = s2 - std::log(m2pi * x2) +
                 2 * std::log1p(-(1 - (1 - (5 - 9 / (8. + x2)) / (6. + x2)) / (4. + x2)) /
                                (2. + x2));
          }
        }
      }
    }
    val = std::sqrt(x2);
  }
  return q < 0.0 ? -val : val;
}

double dnorm(double x) {
  if (std::isnan(x)) return x;
  x = std::fabs(x);
  if (!std::isfinite(x) || x >= 2 * std::sqrt(DBL_MAX)) return 0.0;
  if (x < 5) return k1Sqrt2Pi * std::exp(-0.5 * x * x);
  if (x > std::sqrt(-2 * kLn2 * (DBL_MIN_EXP + 1 - DBL_MANT_DIG))) return 0.0;
  const double x1 = std::ldexp(std::nearbyint(std::ldexp(x, 16)), -16);
  const double x2 = x - x1;
  return k1Sqrt2Pi * (std::exp(-0.5 * x1 * x1) * std::exp((-0.5 * x2 - x1) * x2));
}

}  // namespace detail
}  // namespace timesift
