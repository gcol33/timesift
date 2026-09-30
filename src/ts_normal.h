#ifndef TIMESIFT_TS_NORMAL_H
#define TIMESIFT_TS_NORMAL_H

// The standard normal distribution, evaluated once for every fit that reads it: the lower tail by
// Cody's (1969) rational Chebyshev approximations, the quantile by Wichura's (1988) Algorithm
// AS 241, and the density with the argument split above five so the square in the exponent stays
// exact. A probit link is only as reproducible as these three, which is why they are fixed
// published approximations rather than the platform's `erf`.
namespace timesift {
namespace detail {

double pnorm(double x);  // P[X <= x]
double qnorm(double p);  // the x with P[X <= x] = p, for p in (0, 1)
double dnorm(double x);

}  // namespace detail
}  // namespace timesift

#endif  // TIMESIFT_TS_NORMAL_H
