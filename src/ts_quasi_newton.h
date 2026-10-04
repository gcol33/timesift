#ifndef TIMESIFT_TS_QUASI_NEWTON_H
#define TIMESIFT_TS_QUASI_NEWTON_H

#include <functional>
#include <vector>

// The variable metric minimiser of Nash (1990, Compact Numerical Methods for Computers, 2nd ed.,
// Algorithm 21): a BFGS update of an approximate inverse Hessian, started at the identity, with a
// backtracking line search along the quasi-Newton direction.
//
// Each iteration steps from the current point along `t = -B g`. The step starts at one and is cut
// by `kStepReduction` until the objective falls by at least `kAcceptance` times the step times the
// slope `g't`, or until no coordinate of the trial point differs from the current one at the scale
// `kRelativeTest` (that is, `kRelativeTest + x == kRelativeTest + x'` in every coordinate). An
// accepted step updates `B` from the step `s` and the change in gradient `c` by
// `B += ((1 + c'Bc / s'c) s s' - (Bc) s' - s (Bc)') / s'c` where `s'c` is positive, and resets
// `B` to the identity where it is not. A direction that is not downhill, or a search that cannot
// move, resets `B` too, and two such failures in a row from the identity end the search, as does
// an objective below `abs_tol`, a fall of no more than `rel_tol` times the objective, or
// `max_iter` gradients. `B` is also reset after `2 n` accepted steps without one.
namespace timesift {

struct QuasiNewtonResult {
  double value = 0.0;   // the objective at the point returned
  int iterations = 0;   // gradients taken, the one at the start included
  int evaluations = 0;  // objectives taken
  bool converged = false;  // false where `max_iter` was reached
};

// Minimises `fn` from `x`, which holds the minimiser on return. `fn` is the objective at a point,
// `gr` its gradient there; a non-finite objective is a point the line search does not accept.
// The objective at the start has to be finite.
QuasiNewtonResult variable_metric(std::vector<double>& x,
                                  const std::function<double(const std::vector<double>&)>& fn,
                                  const std::function<void(const std::vector<double>&,
                                                           std::vector<double>&)>& gr,
                                  int max_iter, double abs_tol, double rel_tol);

}  // namespace timesift

#endif  // TIMESIFT_TS_QUASI_NEWTON_H
