#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "ts_core.h"
#include "ts_tree.h"
#include "ts_trees_internal.h"

// Gradient boosting (Friedman 2001, Annals of Statistics 29:1189-1232): the score starts at the
// constant that minimises the loss and each tree is a least squares fit to the negative gradient,
// its leaves then set by one Newton step on the loss. Each tree sees a bag of the observations
// drawn without replacement, which is stochastic gradient boosting (Friedman 2002, Computational
// Statistics & Data Analysis 38:367-378). The second-order trees are those of Chen and Guestrin
// (2016, KDD '16:785-794): the split gain and the leaf read the sums of the gradient and the
// hessian under an L2 penalty on the leaf values.
//
// Every sum is accumulated in one fixed order, the order that reproduces gbm's and xgboost's fits
// in the fixtures, and a fused multiply-add would round differently from the separate multiply and
// add the order is written in, so contraction is off on every compiler that would otherwise do it.
#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

namespace timesift {
namespace {

// The least loss reduction a second-order split has to exceed, and the floor on the curvature of
// the logistic loss; both are the values that reproduce xgboost's trees in the fixtures.
constexpr double kMinSecondOrderGain = 1e-6;
constexpr double kMinCurvature = 1e-16;

// A Poisson fit's safeguards. The log link makes the loss's curvature the mean itself, which a
// second-order step can overshoot by orders of magnitude, so xgboost's count objective inflates the
// hessian by the exponential of a step bound and caps every leaf's step at it; the bound is its
// default of 0.7. A first-order leaf is gbm's: the step is the log of the ratio of events to
// expected events, set to -1 where a leaf holds no event and to 0 where it holds no exposure, and
// kept within 19 of the linear predictor's range in the leaf so no mean overflows.
constexpr double kPoissonStepBound = 0.7;
constexpr double kPoissonLinkBound = 19.0;

// What a set of rows brings to a split. For a first-order tree `first` is the weighted working
// response and `second` the weight; for a second-order tree they are the gradient and the hessian.
struct Moments {
  double first = 0.0;
  double second = 0.0;
  long rows = 0;
};

// A division of a node at `threshold` on `column`: rows below go to `below`, the rest to `above`.
struct Division {
  double gain = 0.0;
  int column = -1;
  double threshold = 0.0;
  Moments below;
  Moments above;
};

// A node of the tree being grown, numbered in the order it was created. `mean` is the least
// squares leaf of a first-order tree; `gain` is what the node's division reduced the loss by.
struct Draft {
  int column = -1;
  double threshold = 0.0;
  int below = -1;
  int above = -1;
  double mean = 0.0;
  double gain = 0.0;
};

// A leaf of the tree being grown: its rows' moments, the best division read for it so far, and
// the running state of the sweep along one sorted column.
struct Frontier {
  int draft;
  bool searching;
  Moments total;
  Division best;
  Moments passed;
  Moments remaining;
  double previous = -HUGE_VAL;

  Frontier(int at, const Moments& m) : draft(at), searching(true), total(m) {}

  void start_sweep() {
    passed = Moments();
    remaining = total;
    previous = -HUGE_VAL;
  }

  // Moves one row from the rows not yet swept to the rows swept.
  void pass(double first, double second, double x) {
    passed.first += first;
    passed.second += second;
    passed.rows++;
    remaining.first -= first;
    remaining.second -= second;
    remaining.rows--;
    previous = x;
  }
};

// The reduction in the weighted squared error of the working response when a node is cut into
// two sides holding `a` and `b`.
double squared_error_drop(const Moments& a, const Moments& b) {
  const double d = a.first / a.second - b.first / b.second;
  return a.second * b.second * d * d / (a.second + b.second);
}

// Selection sampling (Knuth, The Art of Computer Programming, vol. 2, Algorithm 3.4.2 S): each of
// `m` observations in turn is taken with the chance that fills what is left of a sample of `size`
// from what is left of the observations, so exactly `size` are taken in one pass.
std::size_t select_sample(detail::Stream& stream, std::size_t size, std::vector<char>& taken) {
  const std::size_t m = taken.size();
  std::size_t chosen = 0;
  for (std::size_t k = 0; k < m; ++k) {
    const bool take = stream.uniform() * static_cast<double>(m - k) <
                      static_cast<double>(size - chosen);
    taken[k] = take;
    chosen += take;
  }
  return chosen;
}

// The first `keep` trees of a table.
TreeTable first_trees(const TreeTable& all, std::size_t keep) {
  const auto nodes = static_cast<std::ptrdiff_t>(all.offset[keep]);
  const auto head = [nodes](const auto& v) {
    return std::decay_t<decltype(v)>(v.begin(), v.begin() + nodes);
  };
  TreeTable out;
  out.offset.assign(all.offset.begin(), all.offset.begin() + static_cast<std::ptrdiff_t>(keep + 1));
  out.column = head(all.column);
  out.threshold = head(all.threshold);
  out.less_left = head(all.less_left);
  out.left = head(all.left);
  out.right = head(all.right);
  out.value = head(all.value);
  return out;
}

// One boosted fit on a subset of the observations, optionally scored on a held-out subset after
// every tree.
class BoostRun {
 public:
  BoostRun(const double* x, const double* y, const double* w, std::size_t n, std::size_t p,
           Family family, const BoostSpec& spec)
      : x_(x), y_(y), w_(w), n_(n), p_(p), family_(family), spec_(spec) {}

  // Fits on the rows `train` and, where `held` holds any, records the held-out deviance after
  // each tree. `model` numbers the fit among those of one call; with a tree's index it picks the
  // tree's stream.
  void run(const std::vector<int>& train, const std::vector<int>& held, std::uint32_t model,
           double* init, TreeTable* trees, std::vector<double>* deviance) {
    rows_ = &train;
    const std::size_t m = train.size();
    *init = start(train);
    const std::size_t bag_size = static_cast<std::size_t>(spec_.subsample * static_cast<double>(m));
    if (bag_size < 1) throw Error("boosting's subsample of a fit holds no observation.");
    const std::size_t n_col = std::max<std::size_t>(
        1, static_cast<std::size_t>(spec_.colsample * static_cast<double>(p_)));
    sort_columns();

    std::vector<double> score(m, *init);
    std::vector<double> held_score(held.size(), *init);
    residual_.resize(m);
    first_.resize(m);
    second_.resize(m);
    in_bag_.resize(m);
    home_.resize(m);
    std::vector<int> pool(p_);
    std::vector<int> cols;
    std::vector<int> leaf(m);
    deviance->clear();

    const std::size_t n_tree = static_cast<std::size_t>(spec_.trees);
    for (std::size_t tree = 0; tree < n_tree; ++tree) {
      detail::Stream stream(spec_.seed, static_cast<std::uint32_t>(model * n_tree + tree));
      const std::size_t bagged = select_sample(stream, bag_size, in_bag_);
      for (std::size_t j = 0; j < p_; ++j) pool[j] = static_cast<int>(j);
      detail::draw_columns(stream, pool, n_col, cols);

      working_response(score);
      drafts_.clear();
      if (spec_.newton) {
        grow_level_wise(cols, bagged);
      } else {
        grow_best_first(cols, bagged);
      }
      const detail::Nodes table = lay_out(leaf);
      detail::append(*trees, table);

      for (std::size_t k = 0; k < m; ++k) score[k] += table.value[static_cast<std::size_t>(leaf[k])];
      if (!held.empty()) deviance->push_back(score_held(table, held, held_score));
    }
  }

 private:
  double at(std::size_t k, int column) const {
    return x_[static_cast<std::size_t>((*rows_)[k]) + static_cast<std::size_t>(column) * n_];
  }

  // Copies the fit's responses and weights and returns the starting score: the log-odds of the
  // weighted share of ones, the weighted mean, or the log of the weighted mean count.
  double start(const std::vector<int>& train) {
    const std::size_t m = train.size();
    y_fit_.resize(m);
    w_fit_.resize(m);
    double ones = 0.0;
    double total = 0.0;
    for (std::size_t k = 0; k < m; ++k) {
      y_fit_[k] = y_[train[k]];
      w_fit_[k] = w_[train[k]];
      ones += w_fit_[k] * y_fit_[k];
      total += w_fit_[k];
    }
    if (family_ == Family::binomial) {
      if (!(ones > 0) || !(total - ones > 0)) {
        throw Error("boosting on a binomial response needs weight on both classes in every fit, "
                    "and one fit has weight on one alone.");
      }
      return std::log(ones / (total - ones));
    }
    if (!(total > 0)) throw Error("boosting's case weights sum to more than zero in every fit.");
    if (family_ == Family::poisson) {
      if (!(ones > 0)) {
        throw Error("boosting on a count response needs a count above zero in every fit, and "
                    "one fit has none.");
      }
      return std::log(ones / total);
    }
    return ones / total;
  }

  // Each column's rows in ascending order, rows of equal value in row order, sorted once per fit
  // and swept by every tree.
  void sort_columns() {
    const std::size_t m = rows_->size();
    sorted_.resize(p_ * m);
    for (std::size_t v = 0; v < p_; ++v) {
      const auto first = sorted_.begin() + static_cast<std::ptrdiff_t>(v * m);
      const auto last = first + static_cast<std::ptrdiff_t>(m);
      for (std::size_t k = 0; k < m; ++k) first[static_cast<std::ptrdiff_t>(k)] = static_cast<int>(k);
      const double* col = x_ + v * n_;
      const std::vector<int>& rows = *rows_;
      std::stable_sort(first, last, [col, &rows](int a, int b) {
        return col[rows[static_cast<std::size_t>(a)]] < col[rows[static_cast<std::size_t>(b)]];
      });
    }
  }

  // What each row brings to the next tree. A first-order tree fits the negative gradient of the
  // loss, weighted; a second-order tree reads the gradient and the hessian, each weighted.
  void working_response(const std::vector<double>& score) {
    score_ = &score;
    for (std::size_t k = 0; k < score.size(); ++k) {
      const double y = y_fit_[k];
      const double w = w_fit_[k];
      if (spec_.newton) {
        if (family_ == Family::binomial) {
          const double prob = 1.0 / (1.0 + std::exp(-score[k]));
          first_[k] = (prob - y) * w;
          second_[k] = std::max(prob * (1.0 - prob), kMinCurvature) * w;
        } else if (family_ == Family::poisson) {
          const double mean = std::exp(score[k]);
          first_[k] = (mean - y) * w;
          second_[k] = mean * std::exp(kPoissonStepBound) * w;
        } else {
          first_[k] = (score[k] - y) * w;
          second_[k] = w;
        }
      } else {
        residual_[k] = mean_residual(y, score[k]);
        first_[k] = w * residual_[k];
        second_[k] = w;
      }
    }
  }

  // The negative gradient of the loss at a score: `y - p` for the binomial deviance, `y - exp(f)`
  // for the Poisson one, and `y - f` for squared error.
  double mean_residual(double y, double score) const {
    switch (family_) {
      case Family::binomial: return y - 1.0 / (1.0 + std::exp(-score));
      case Family::poisson: return y - std::exp(score);
      case Family::gaussian: break;
    }
    return y - score;
  }

  // The bag's moments, every row placed in the root's frontier slot.
  Moments root_moments(std::size_t bagged) {
    Moments root;
    for (std::size_t k = 0; k < home_.size(); ++k) {
      home_[k] = 0;
      if (!in_bag_[k]) continue;
      root.first += first_[k];
      root.second += second_[k];
    }
    root.rows = static_cast<long>(bagged);
    return root;
  }

  int new_draft(double mean) {
    Draft d;
    d.mean = mean;
    drafts_.push_back(d);
    return static_cast<int>(drafts_.size()) - 1;
  }

  // Cuts draft `parent` as `division` says and returns its two children's drafts.
  std::pair<int, int> divide(int parent, const Division& division, double below_mean,
                             double above_mean) {
    const int below = new_draft(below_mean);
    const int above = new_draft(above_mean);
    Draft& d = drafts_[static_cast<std::size_t>(parent)];
    d.column = division.column;
    d.threshold = division.threshold;
    d.below = below;
    d.above = above;
    d.gain = division.gain;
    return {below, above};
  }

  // A first-order tree grown best first: `depth` times, the leaf whose best division reduces the
  // squared error most is divided, the earliest in frontier order where several tie, and growth
  // stops early where no division reduces it. A division leaves at least `min_leaf` bagged rows on
  // each side. A leaf is searched once, when it is created, and keeps its best division until it
  // is divided; a divided leaf's frontier slot passes to its lower child and its upper child is
  // appended.
  void grow_best_first(const std::vector<int>& cols, std::size_t bagged) {
    const Moments root = root_moments(bagged);
    std::vector<Frontier> frontier;
    frontier.emplace_back(new_draft(root.first / root.second), root);

    for (int step = 0; step < spec_.depth; ++step) {
      sweep_squared_error(cols, frontier);

      std::size_t chosen = frontier.size();
      double most = 0.0;
      for (std::size_t i = 0; i < frontier.size(); ++i) {
        if (frontier[i].best.gain > most) {
          most = frontier[i].best.gain;
          chosen = i;
        }
      }
      if (chosen == frontier.size()) break;

      const Division division = frontier[chosen].best;
      const int parent = frontier[chosen].draft;
      const std::pair<int, int> children =
          divide(parent, division, division.below.first / division.below.second,
                 division.above.first / division.above.second);
      const int upper_slot = static_cast<int>(frontier.size());
      frontier.emplace_back(children.second, division.above);
      frontier[chosen] = Frontier(children.first, division.below);

      const int lower_slot = static_cast<int>(chosen);
      for (std::size_t k = 0; k < home_.size(); ++k) {
        if (home_[k] == lower_slot && !(at(k, division.column) < division.threshold)) {
          home_[k] = upper_slot;
        }
      }
    }
  }

  // Sweeps every drawn column in ascending order through the leaves still to be searched, keeping
  // each one's division of largest squared error drop. A cut lies midway between two distinct
  // successive values.
  void sweep_squared_error(const std::vector<int>& cols, std::vector<Frontier>& frontier) {
    const std::size_t m = home_.size();
    const double min_leaf = spec_.min_leaf;
    for (const int v : cols) {
      for (Frontier& f : frontier) {
        if (f.searching) f.start_sweep();
      }
      const int* order = sorted_.data() + static_cast<std::size_t>(v) * m;
      for (std::size_t o = 0; o < m; ++o) {
        const std::size_t k = static_cast<std::size_t>(order[o]);
        if (!in_bag_[k]) continue;
        Frontier& f = frontier[static_cast<std::size_t>(home_[k])];
        if (!f.searching) continue;
        const double xk = at(k, v);
        if (f.previous != xk && static_cast<double>(f.passed.rows) >= min_leaf &&
            static_cast<double>(f.remaining.rows) >= min_leaf) {
          const double gain = squared_error_drop(f.passed, f.remaining);
          if (gain > f.best.gain) {
            f.best = Division{gain, v, 0.5 * (f.previous + xk), f.passed, f.remaining};
          }
        }
        f.pass(first_[k], second_[k], xk);
      }
    }
    for (Frontier& f : frontier) f.searching = false;
  }

  // A second-order tree grown level by level to `depth`: every leaf of a level is searched over
  // the drawn columns and divided where its best gain exceeds the least gain; the rows of a leaf
  // left whole take no further part. The tree is then pruned where `gamma` asks for it.
  void grow_level_wise(const std::vector<int>& cols, std::size_t bagged) {
    std::vector<Frontier> frontier;
    frontier.emplace_back(new_draft(0.0), root_moments(bagged));

    for (int level = 0; level < spec_.depth; ++level) {
      sweep_second_order(cols, frontier);

      std::vector<Frontier> next;
      std::vector<int> moved_to(frontier.size(), -1);
      for (std::size_t i = 0; i < frontier.size(); ++i) {
        const Division& division = frontier[i].best;
        if (!(division.gain > kMinSecondOrderGain)) continue;
        const std::pair<int, int> children = divide(frontier[i].draft, division, 0.0, 0.0);
        moved_to[i] = static_cast<int>(next.size());
        next.emplace_back(children.first, division.below);
        next.emplace_back(children.second, division.above);
      }
      if (next.empty()) break;

      for (std::size_t k = 0; k < home_.size(); ++k) {
        const int from = home_[k];
        if (from < 0) continue;
        const int to = moved_to[static_cast<std::size_t>(from)];
        if (to < 0) {
          home_[k] = -1;
          continue;
        }
        const Draft& d =
            drafts_[static_cast<std::size_t>(frontier[static_cast<std::size_t>(from)].draft)];
        home_[k] = at(k, d.column) < d.threshold ? to : to + 1;
      }
      frontier = std::move(next);
    }
    if (spec_.gamma > 0) prune(0);
  }

  // What a node holding gradient `G` and hessian `H` is worth to a second-order tree:
  // `G^2 / (H + lambda)`, the loss it removes at its Newton step (Chen and Guestrin 2016, eq. 6,
  // without the halving). Where a step is capped, as a Poisson one is, it is what the capped step
  // removes: `-(2 G w + (H + lambda) w^2)` at the step `w`.
  double second_order_gain(const Moments& m, double lambda) const {
    if (family_ != Family::poisson) return m.first * m.first / (m.second + lambda);
    const double w = capped_step(m.first, m.second, lambda);
    return -(2.0 * m.first * w + (m.second + lambda) * w * w);
  }

  // The Newton step `-G / (H + lambda)` of a second-order leaf, held within the bound of a Poisson
  // fit.
  double capped_step(double gradient, double hessian, double lambda) const {
    const double d = hessian + lambda;
    const double step = d == 0 ? 0.0 : -gradient / d;
    if (family_ != Family::poisson) return step;
    return std::min(std::max(step, -kPoissonStepBound), kPoissonStepBound);
  }

  // The second-order gain of a cut (Chen and Guestrin 2016, eq. 7, without the halving and the
  // complexity term): `G_L^2 / (H_L + lambda) + G_R^2 / (H_R + lambda) - G^2 / (H + lambda)`, with
  // at least `min_leaf` of hessian on each side.
  void sweep_second_order(const std::vector<int>& cols, std::vector<Frontier>& frontier) {
    const std::size_t m = home_.size();
    const double lambda = spec_.lambda;
    const double min_leaf = spec_.min_leaf;
    for (const int v : cols) {
      for (Frontier& f : frontier) f.start_sweep();
      const int* order = sorted_.data() + static_cast<std::size_t>(v) * m;
      for (std::size_t o = 0; o < m; ++o) {
        const std::size_t k = static_cast<std::size_t>(order[o]);
        if (!in_bag_[k] || home_[k] < 0) continue;
        Frontier& f = frontier[static_cast<std::size_t>(home_[k])];
        const double xk = at(k, v);
        if (f.passed.rows > 0 && f.previous != xk) {
          const Moments& below = f.passed;
          Moments above;
          above.first = f.total.first - below.first;
          above.second = f.total.second - below.second;
          above.rows = f.total.rows - below.rows;
          if (below.second >= min_leaf && above.second >= min_leaf) {
            const double gain = second_order_gain(below, lambda) +
                                second_order_gain(above, lambda) -
                                second_order_gain(f.total, lambda);
            if (gain > f.best.gain) {
              f.best = Division{gain, v, 0.5 * (f.previous + xk), below, above};
            }
          }
        }
        f.passed.first += first_[k];
        f.passed.second += second_[k];
        f.passed.rows++;
        f.previous = xk;
      }
    }
  }

  // Makes a leaf of every division, from the leaves up, whose children are both leaves and whose
  // gain falls below `gamma`.
  void prune(int node) {
    if (drafts_[static_cast<std::size_t>(node)].column < 0) return;
    prune(drafts_[static_cast<std::size_t>(node)].below);
    prune(drafts_[static_cast<std::size_t>(node)].above);
    Draft& d = drafts_[static_cast<std::size_t>(node)];
    const bool over_leaves = drafts_[static_cast<std::size_t>(d.below)].column < 0 &&
                             drafts_[static_cast<std::size_t>(d.above)].column < 0;
    if (over_leaves && d.gain < spec_.gamma) {
      d.column = -1;
      d.below = -1;
      d.above = -1;
    }
  }

  // The grown tree as a node table in depth-first order, lower subtree first, with each leaf's
  // step scaled by the shrinkage, and the leaf every training row of the fit falls into.
  //
  // A leaf's step is one Newton step on the loss over the bagged rows in it (Friedman 2001,
  // section 4.5): `sum w r / sum w p (1 - p)` for the binomial deviance, `log(sum w y / sum w mu)`
  // for the Poisson one, the mean of the working response for squared error, and `-G / (H + lambda)`
  // for a second-order tree. Its sums are taken in row order.
  detail::Nodes lay_out(std::vector<int>& leaf) {
    detail::Nodes out;
    struct Pending {
      int draft;
      int parent;
      bool upper;
    };
    std::vector<Pending> stack{{0, -1, false}};
    while (!stack.empty()) {
      const Pending next = stack.back();
      stack.pop_back();
      const Draft& d = drafts_[static_cast<std::size_t>(next.draft)];
      const std::int32_t here = out.add_leaf(d.mean);
      if (next.parent >= 0) {
        (next.upper ? out.right : out.left)[static_cast<std::size_t>(next.parent)] = here;
      }
      if (d.column < 0) continue;
      out.column[static_cast<std::size_t>(here)] = d.column;
      out.threshold[static_cast<std::size_t>(here)] = d.threshold;
      out.less_left[static_cast<std::size_t>(here)] = 1;
      stack.push_back({d.above, here, true});
      stack.push_back({d.below, here, false});
    }

    const std::size_t size = out.column.size();
    std::vector<double> num(size, 0.0);
    std::vector<double> den(size, 0.0);
    std::vector<double> highest(size, -HUGE_VAL);
    std::vector<double> lowest(size, HUGE_VAL);
    for (std::size_t k = 0; k < leaf.size(); ++k) {
      std::size_t node = 0;
      while (out.column[node] >= 0) {
        node = static_cast<std::size_t>(at(k, out.column[node]) < out.threshold[node]
                                            ? out.left[node]
                                            : out.right[node]);
      }
      leaf[k] = static_cast<int>(node);
      if (!in_bag_[k]) continue;
      if (spec_.newton) {
        num[node] += first_[k];
        den[node] += second_[k];
      } else if (family_ == Family::binomial) {
        const double r = residual_[k];
        num[node] += first_[k];
        den[node] += w_fit_[k] * (y_fit_[k] - r) * (1 - y_fit_[k] + r);
      } else if (family_ == Family::poisson) {
        const double f = (*score_)[k];
        num[node] += w_fit_[k] * y_fit_[k];
        den[node] += w_fit_[k] * std::exp(f);
        highest[node] = std::max(highest[node], f);
        lowest[node] = std::min(lowest[node], f);
      }
    }
    for (std::size_t i = 0; i < size; ++i) {
      if (out.column[i] >= 0) {
        out.value[i] = 0.0;
        continue;
      }
      double step;
      if (spec_.newton) {
        step = capped_step(num[i], den[i], spec_.lambda);
      } else if (family_ == Family::binomial) {
        step = den[i] == 0 ? 0.0 : num[i] / den[i];
      } else if (family_ == Family::poisson) {
        if (num[i] == 0.0) {
          step = -1.0;
        } else {
          step = den[i] == 0.0 ? 0.0 : std::log(num[i] / den[i]);
        }
        if (highest[i] >= lowest[i]) {
          step = std::min(step, kPoissonLinkBound - highest[i]);
          step = std::max(step, -kPoissonLinkBound - lowest[i]);
        }
      } else {
        step = out.value[i];
      }
      out.value[i] = spec_.shrinkage * step;
    }
    return out;
  }

  // Adds the new tree to the held-out rows' scores and returns their weighted mean deviance: the
  // binomial deviance, the Poisson deviance without its saturated term, which no tree changes, or
  // the squared error.
  double score_held(const detail::Nodes& table, const std::vector<int>& held,
                    std::vector<double>& score) const {
    double loglik = 0.0;
    double weight = 0.0;
    for (std::size_t i = 0; i < held.size(); ++i) {
      const std::size_t r = static_cast<std::size_t>(held[i]);
      score[i] += table.value[detail::leaf_of(table.column.data(), table.threshold.data(),
                                              table.less_left.data(), table.left.data(),
                                              table.right.data(), x_, r, n_, p_)];
      const double f = score[i];
      if (family_ == Family::binomial) {
        loglik += w_[r] * (y_[r] * f - std::log(1.0 + std::exp(f)));
      } else if (family_ == Family::poisson) {
        loglik += w_[r] * (y_[r] * f - std::exp(f));
      } else {
        loglik += w_[r] * (y_[r] - f) * (y_[r] - f);
      }
      weight += w_[r];
    }
    return family_ == Family::gaussian ? loglik / weight : -2 * loglik / weight;
  }

  const double* x_;
  const double* y_;
  const double* w_;
  std::size_t n_;
  std::size_t p_;
  Family family_;
  BoostSpec spec_;

  const std::vector<int>* rows_ = nullptr;
  const std::vector<double>* score_ = nullptr;
  std::vector<double> y_fit_, w_fit_;
  std::vector<int> sorted_;
  std::vector<double> residual_, first_, second_;
  std::vector<char> in_bag_;
  std::vector<int> home_;
  std::vector<Draft> drafts_;
};

void check_spec(const BoostSpec& spec) {
  if (spec.trees < 1) throw Error("boosting fits at least one tree.");
  if (spec.depth < 1) throw Error("boosting's `depth` is at least one.");
  if (!(spec.shrinkage > 0) || !std::isfinite(spec.shrinkage)) {
    throw Error("boosting's `shrinkage` is a finite number above zero.");
  }
  if (!(spec.subsample > 0 && spec.subsample <= 1)) {
    throw Error("boosting's `subsample` lies above zero and at most one.");
  }
  if (!(spec.colsample > 0 && spec.colsample <= 1)) {
    throw Error("boosting's `colsample` lies above zero and at most one.");
  }
  if (!(spec.min_leaf >= 0) || !(spec.lambda >= 0) || !(spec.gamma >= 0)) {
    throw Error("boosting's `min_leaf`, `lambda` and `gamma` are zero or more.");
  }
}

}  // namespace

Boosted boost_fit(const double* x, const double* y, const double* w, std::size_t n,
                  std::size_t p, Family family, const BoostSpec& spec, const std::int32_t* fold,
                  std::int32_t n_fold) {
  if (n == 0 || p == 0) throw Error("boosting needs at least one observation and one column.");
  check_spec(spec);
  detail::check_finite(x, n * p, "boosting", "design");
  detail::check_finite(y, n, "boosting", "response");
  detail::check_finite(w, n, "boosting", "weights");
  for (std::size_t i = 0; i < n; ++i) {
    if (w[i] < 0) throw Error("boosting's case weights are zero or more.");
    if (family == Family::binomial && y[i] != 0.0 && y[i] != 1.0) {
      throw Error("boosting on a binomial response reads 0 and 1 alone.");
    }
    if (family == Family::poisson && y[i] < 0.0) {
      throw Error("boosting on a count response reads values of zero or more.");
    }
  }
  const bool cross = fold != nullptr && n_fold > 1;
  if (cross) {
    for (std::size_t i = 0; i < n; ++i) {
      if (fold[i] < 0 || fold[i] >= n_fold) {
        throw Error("a fold index lies between 0 and the number of folds less one.");
      }
    }
  }

  // Fit 0 is on every observation; fit g + 1 is on the complement of fold g and scored on fold g.
  const std::size_t fits = cross ? static_cast<std::size_t>(n_fold) + 1 : 1;
  std::vector<std::vector<int>> train(fits), held(fits);
  for (std::size_t i = 0; i < n; ++i) {
    const int row = static_cast<int>(i);
    train[0].push_back(row);
    if (!cross) continue;
    for (std::size_t g = 1; g < fits; ++g) {
      const bool out = fold[i] == static_cast<std::int32_t>(g - 1);
      (out ? held[g] : train[g]).push_back(row);
    }
  }
  std::vector<double> init(fits, 0.0);
  std::vector<TreeTable> tables(fits);
  std::vector<std::vector<double>> deviance(fits);
  detail::run_tasks(fits, spec.threads, [&](std::size_t f) {
    BoostRun fit(x, y, w, n, p, family, spec);
    fit.run(train[f], held[f], static_cast<std::uint32_t>(f), &init[f], &tables[f], &deviance[f]);
  });

  Boosted out;
  out.family = family;
  out.n_column = static_cast<std::int32_t>(p);
  out.init = init[0];
  std::size_t keep = static_cast<std::size_t>(spec.trees);
  if (cross) {
    // The cross-validated error after each tree: every fold's held-out deviance weighted by the
    // observations it holds, summed over the folds in order and divided by all the observations.
    // The number of trees kept is the first at which it is least.
    out.cv_error.assign(keep, 0.0);
    for (std::size_t t = 0; t < keep; ++t) {
      double total = 0.0;
      for (std::size_t g = 1; g < fits; ++g) {
        total += deviance[g][t] * static_cast<double>(held[g].size());
      }
      out.cv_error[t] = total / static_cast<double>(n);
    }
    keep = static_cast<std::size_t>(std::min_element(out.cv_error.begin(), out.cv_error.end()) -
                                    out.cv_error.begin()) + 1;
  }
  out.trees = first_trees(tables[0], keep);
  return out;
}

void boost_predict(const Boosted& model, const double* x, std::size_t n, std::size_t p,
                   double* out) {
  const std::size_t trees = detail::table_trees(model.trees);
  for (std::size_t i = 0; i < n; ++i) {
    double f = model.init;
    for (std::size_t t = 0; t < trees; ++t) f += detail::table_value(model.trees, t, x, i, n, p);
    switch (model.family) {
      case Family::binomial: out[i] = 1.0 / (1.0 + std::exp(-f)); break;
      case Family::poisson: out[i] = std::exp(f); break;
      case Family::gaussian: out[i] = f; break;
    }
  }
}

std::vector<Boosted> boost_fits(const double* x, std::size_t n, std::size_t p, const double* y,
                                const double* w, std::size_t r, Family family,
                                const BoostSpec& spec, const std::uint32_t* seeds,
                                const std::int32_t* fold, const std::int32_t* n_fold) {
  return detail::fit_responses(r, spec.threads, [&](std::size_t s, int inner) {
    BoostSpec each = spec;
    each.seed = seeds[s];
    each.threads = inner;
    return boost_fit(x, y + s * n, w + s * n, n, p, family, each,
                     detail::response_fold(fold, n, s),
                     detail::response_fold_count(fold, n_fold, s));
  });
}

}  // namespace timesift
