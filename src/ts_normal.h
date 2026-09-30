#ifndef TIMESIFT_TS_NORMAL_H
#define TIMESIFT_TS_NORMAL_H

// The standard normal distribution as R's nmath evaluates it, once, for every fit that reads it:
// the lower tail by Cody's rational approximations (`pnorm_both`), the quantile by Wichura's AS 241
// (`qnorm5`), and the density with R's split of the argument above five (`dnorm4`). A probit link is
// only as reproducible as these three, which is why they are R's rather than the platform's `erf`.
namespace timesift {
namespace detail {

double pnorm(double x);  // P[X <= x]
double qnorm(double p);  // the x with P[X <= x] = p, for p in (0, 1)
double dnorm(double x);

}  // namespace detail
}  // namespace timesift

#endif  // TIMESIFT_TS_NORMAL_H
