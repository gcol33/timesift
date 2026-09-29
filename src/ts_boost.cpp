#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "ts_core.h"
#include "ts_tree.h"
#include "ts_trees_internal.h"

// Every sum here is taken in the order gbm takes it, so a boosted fit is gbm's to the last place
// where the two are asked the same thing. Contraction is off for the reason `ts_tree.cpp` gives.
#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

namespace timesift {
namespace {

// xgboost's least loss change a split must reach.
constexpr double kRtEps = 1e-6;

// xgboost's floor on a logistic hessian.
constexpr double kMinHessian = 1e-16;

// A node while its tree is grown: its split, its children, its value for gbm's Gaussian leaf, and
// the loss change of its split for the second-order pruning.
struct GrowNode {
  int column = -1;
  double threshold = 0.0;
  int left = -1;
  int right = -1;
  double mean = 0.0;
  double loss = 0.0;
};

// A node a split is searched for: what it holds, the best split read so far, and the running sums
// of the column being scanned. gbm's `CNodeSearch` for the first-order trees; the second-order
// ones read the gradient and hessian sums through the same fields.
struct Search {
  int node = -1;         // the GrowNode, or -1 for gbm's missing-value branch, which holds nothing
  bool open = false;     // still to be searched
  double sum = 0.0;      // the weighted working response, or the gradient
  double weight = 0.0;   // the weight, or the hessian
  long count = 0;
  // The best split.
  double best = 0.0;
  int var = -1;
  double split = 0.0;
  double left_sum = 0.0, left_weight = 0.0, right_sum = 0.0, right_weight = 0.0;
  long left_count = 0, right_count = 0;
  // The column being scanned.
  double cur_left_sum = 0.0, cur_left_weight = 0.0, cur_right_sum = 0.0, cur_right_weight = 0.0;
  long cur_left_count = 0, cur_right_count = 0;
  double last = -HUGE_VAL;

  void reset_column() {
    cur_left_sum = 0.0;
    cur_left_weight = 0.0;
    cur_left_count = 0;
    cur_right_sum = sum;
    cur_right_weight = weight;
    cur_right_count = count;
    last = -HUGE_VAL;
  }

  void reset(int at, double s, double w, long c) {
    node = at;
    open = at >= 0;
    sum = s;
    weight = w;
    count = c;
    best = 0.0;
    var = -1;
  }
};

// gbm's improvement of a split: the reduction in the weighted squared error of the working
// response, with no observation missing.
double improvement(double lw, double rw, double ls, double rs) {
  const double d = ls / lw - rs / rw;
  return lw * rw * d * d / (lw + rw);
}

class Booster {
 public:
  Booster(const double* x, const double* y, const double* w, std::size_t n, std::size_t p,
          bool binomial, const BoostSpec& spec)
      : x_(x), y_(y), w_(w), n_(n), p_(p), binomial_(binomial), spec_(spec) {}

  // One fit on the rows `train`, scored after every tree on the rows `held` where there are any.
  // `model` numbers the fit among those of one call, and with the tree's index picks its stream.
  void run(const std::vector<int>& train, const std::vector<int>& held, std::uint32_t model,
           double* init, TreeTable* trees, std::vector<double>* deviance) {
    const std::size_t m = train.size();
    const std::size_t trees_n = static_cast<std::size_t>(spec_.trees);
    yl_.resize(m);
    wl_.resize(m);
    rows_ = &train;
    double s = 0.0;
    double t = 0.0;
    for (std::size_t k = 0; k < m; ++k) {
      yl_[k] = y_[train[k]];
      wl_[k] = w_[train[k]];
      s += wl_[k] * yl_[k];
      t += wl_[k];
    }
    if (binomial_) {
      if (!(s > 0) || !(t - s > 0)) {
        throw Error("boosting on a binomial response needs weight on both classes in every fit, "
                    "and one fit has weight on one alone.");
      }
      *init = std::log(s / (t - s));
    } else {
      if (!(t > 0)) throw Error("boosting's case weights sum to more than zero in every fit.");
      *init = s / t;
    }
    const std::size_t bag = static_cast<std::size_t>(spec_.subsample * static_cast<double>(m));
    if (bag < 1) throw Error("boosting's subsample of a fit holds no observation.");
    const std::size_t n_col = std::max<std::size_t>(
        1, static_cast<std::size_t>(spec_.colsample * static_cast<double>(p_)));

    // Each column's rows in ascending order, ties in row order, as R's `order()` leaves them.
    order_.assign(p_ * m, 0);
    std::vector<int> index(m);
    for (std::size_t v = 0; v < p_; ++v) {
      for (std::size_t k = 0; k < m; ++k) index[k] = static_cast<int>(k);
      const double* col = x_ + v * n_;
      std::stable_sort(index.begin(), index.end(), [&](int a, int b) {
        return col[train[static_cast<std::size_t>(a)]] < col[train[static_cast<std::size_t>(b)]];
      });
      std::copy(index.begin(), index.end(), order_.begin() + static_cast<std::ptrdiff_t>(v * m));
    }

    std::vector<double> f(m, *init);
    std::vector<double> fh(held.size(), *init);
    z_.resize(m);
    h_.resize(m);
    inbag_.resize(m);
    assign_.resize(m);
    std::vector<int> perm(p_);
    std::vector<int> cols;
    std::vector<int> leaf(m);
    deviance->clear();

    for (std::size_t tree = 0; tree < trees_n; ++tree) {
      detail::Stream stream(spec_.seed,
                            static_cast<std::uint32_t>(model * trees_n + tree));
      // gbm's draw of the bag: each observation in turn, kept with the chance of filling what is
      // left of the bag from what is left of the observations.
      std::size_t bagged = 0;
      for (std::size_t k = 0; k < m; ++k) {
        const bool in = stream.uniform() * static_cast<double>(m - k) <
                        static_cast<double>(bag - bagged);
        inbag_[k] = in;
        if (in) bagged++;
      }
      for (std::size_t j = 0; j < p_; ++j) perm[j] = static_cast<int>(j);
      detail::draw_columns(stream, perm, n_col, cols);

      for (std::size_t k = 0; k < m; ++k) {
        if (spec_.newton) {
          if (binomial_) {
            const double prob = 1.0 / (1.0 + std::exp(-f[k]));
            z_[k] = (prob - yl_[k]) * wl_[k];
            h_[k] = std::max(prob * (1.0 - prob), kMinHessian) * wl_[k];
          } else {
            z_[k] = (f[k] - yl_[k]) * wl_[k];
            h_[k] = wl_[k];
          }
        } else {
          z_[k] = binomial_ ? yl_[k] - 1.0 / (1.0 + std::exp(-f[k])) : yl_[k] - f[k];
        }
      }

      nodes_.clear();
      if (spec_.newton) {
        grow_newton(cols, bagged);
      } else {
        grow_gbm(cols, bagged);
      }
      detail::Nodes table = leaves(m, leaf);
      detail::append(*trees, table);

      for (std::size_t k = 0; k < m; ++k) f[k] += table.value[static_cast<std::size_t>(leaf[k])];
      if (!held.empty()) {
        double dl = 0.0;
        double dw = 0.0;
        for (std::size_t i = 0; i < held.size(); ++i) {
          const std::size_t r = static_cast<std::size_t>(held[i]);
          fh[i] += table.value[detail::leaf_of(table.column.data(), table.threshold.data(),
                                               table.less_left.data(), table.left.data(),
                                               table.right.data(), x_, r, n_, p_)];
          if (binomial_) {
            dl += w_[r] * (y_[r] * fh[i] - std::log(1.0 + std::exp(fh[i])));
          } else {
            dl += w_[r] * (y_[r] - fh[i]) * (y_[r] - fh[i]);
          }
          dw += w_[r];
        }
        deviance->push_back(binomial_ ? -2 * dl / dw : dl / dw);
      }
    }
  }

 private:
  double xv(std::size_t k, int v) const {
    return x_[static_cast<std::size_t>((*rows_)[k]) + static_cast<std::size_t>(v) * n_];
  }

  int add_node(double mean) {
    GrowNode node;
    node.mean = mean;
    nodes_.push_back(node);
    return static_cast<int>(nodes_.size() - 1);
  }

  // Sends every row of a split node at or above its threshold to the right child's slot; the rest
  // keep the node's slot, which the left child takes over.
  void route(int from, int to, const GrowNode& node) {
    for (std::size_t k = 0; k < assign_.size(); ++k) {
      if (assign_[k] != from) continue;
      if (!(xv(k, node.column) < node.threshold)) assign_[k] = to;
    }
  }

  // gbm's `CCARTTree::grow`: `depth` splits, each of the terminal node whose best split improves
  // most, the first such node where several tie. A node's best split is searched once, over every
  // column in order, and kept until it is split. Each split leaves three terminal nodes, the third
  // gbm's branch for missing values, which holds nothing here and keeps its place in the order the
  // nodes are compared in.
  void grow_gbm(const std::vector<int>& cols, std::size_t bagged) {
    const std::size_t m = assign_.size();
    double sum = 0.0;
    double weight = 0.0;
    for (std::size_t k = 0; k < m; ++k) {
      assign_[k] = 0;
      if (inbag_[k]) {
        sum += wl_[k] * z_[k];
        weight += wl_[k];
      }
    }
    std::vector<Search> searches(1);
    searches[0].reset(add_node(sum / weight), sum, weight, static_cast<long>(bagged));
    const double min_leaf = spec_.min_leaf;

    for (int step = 0; step < spec_.depth; ++step) {
      for (const int v : cols) {
        for (Search& s : searches) {
          if (s.open) s.reset_column();
        }
        const int* ord = order_.data() + static_cast<std::size_t>(v) * m;
        for (std::size_t o = 0; o < m; ++o) {
          const std::size_t k = static_cast<std::size_t>(ord[o]);
          if (!inbag_[k]) continue;
          Search& s = searches[static_cast<std::size_t>(assign_[k])];
          if (!s.open) continue;
          const double xk = xv(k, v);
          const double wz = wl_[k] * z_[k];
          const double split = 0.5 * (s.last + xk);
          if (s.last != xk && static_cast<double>(s.cur_left_count) >= min_leaf &&
              static_cast<double>(s.cur_right_count) >= min_leaf) {
            const double imp = improvement(s.cur_left_weight, s.cur_right_weight, s.cur_left_sum,
                                           s.cur_right_sum);
            if (imp > s.best) {
              s.best = imp;
              s.var = v;
              s.split = split;
              s.left_sum = s.cur_left_sum;
              s.left_weight = s.cur_left_weight;
              s.left_count = s.cur_left_count;
              s.right_sum = s.cur_right_sum;
              s.right_weight = s.cur_right_weight;
              s.right_count = s.cur_right_count;
            }
          }
          s.cur_left_sum += wz;
          s.cur_left_weight += wl_[k];
          s.cur_left_count++;
          s.cur_right_sum -= wz;
          s.cur_right_weight -= wl_[k];
          s.cur_right_count--;
          s.last = xk;
        }
      }
      for (Search& s : searches) s.open = false;

      std::size_t at = 0;
      double best = 0.0;
      for (std::size_t i = 0; i < searches.size(); ++i) {
        if (searches[i].best > best) {
          at = i;
          best = searches[i].best;
        }
      }
      if (best == 0.0) break;

      const Search chosen = searches[at];
      const int parent = chosen.node;
      const int left = add_node(chosen.left_sum / chosen.left_weight);
      const int right = add_node(chosen.right_sum / chosen.right_weight);
      nodes_[static_cast<std::size_t>(parent)].column = chosen.var;
      nodes_[static_cast<std::size_t>(parent)].threshold = chosen.split;
      nodes_[static_cast<std::size_t>(parent)].left = left;
      nodes_[static_cast<std::size_t>(parent)].right = right;
      const int right_slot = static_cast<int>(searches.size());
      searches.emplace_back();
      searches.emplace_back();
      searches[static_cast<std::size_t>(right_slot)].reset(right, chosen.right_sum,
                                                           chosen.right_weight,
                                                           chosen.right_count);
      searches[static_cast<std::size_t>(right_slot) + 1].reset(-1, 0.0, 0.0, 0);
      searches[at].reset(left, chosen.left_sum, chosen.left_weight, chosen.left_count);
      route(static_cast<int>(at), right_slot, nodes_[static_cast<std::size_t>(parent)]);
    }
  }

  // xgboost's exact greedy tree: every node of a level searched over every column in order, split
  // where its best split gains more than xgboost's least loss change, to `depth` levels; then
  // pruned from the leaves up wherever a split whose children are both leaves gains less than
  // `gamma`.
  void grow_newton(const std::vector<int>& cols, std::size_t bagged) {
    const std::size_t m = assign_.size();
    const double lambda = spec_.lambda;
    const double min_leaf = spec_.min_leaf;
    double g = 0.0;
    double h = 0.0;
    for (std::size_t k = 0; k < m; ++k) {
      assign_[k] = 0;
      if (inbag_[k]) {
        g += z_[k];
        h += h_[k];
      }
    }
    std::vector<Search> searches(1);
    searches[0].reset(add_node(0.0), g, h, static_cast<long>(bagged));

    for (int level = 0; level < spec_.depth; ++level) {
      for (const int v : cols) {
        for (Search& s : searches) s.reset_column();
        const int* ord = order_.data() + static_cast<std::size_t>(v) * m;
        for (std::size_t o = 0; o < m; ++o) {
          const std::size_t k = static_cast<std::size_t>(ord[o]);
          if (!inbag_[k] || assign_[k] < 0) continue;
          Search& s = searches[static_cast<std::size_t>(assign_[k])];
          const double xk = xv(k, v);
          if (s.cur_left_count > 0 && s.last != xk) {
            const double gl = s.cur_left_sum;
            const double hl = s.cur_left_weight;
            const double gr = s.sum - gl;
            const double hr = s.weight - hl;
            if (hl >= min_leaf && hr >= min_leaf) {
              const double loss = gl * gl / (hl + lambda) + gr * gr / (hr + lambda) -
                                  s.sum * s.sum / (s.weight + lambda);
              if (loss > s.best) {
                s.best = loss;
                s.var = v;
                s.split = 0.5 * (s.last + xk);
                s.left_sum = gl;
                s.left_weight = hl;
                s.right_sum = gr;
                s.right_weight = hr;
              }
            }
          }
          s.cur_left_sum += z_[k];
          s.cur_left_weight += h_[k];
          s.cur_left_count++;
          s.last = xk;
        }
      }

      // The level's nodes that gain enough are split, and their rows move to the next level's
      // searches; a row of a node that is not split takes no further part.
      std::vector<Search> next;
      std::vector<int> slot_of(searches.size(), -1);
      for (std::size_t i = 0; i < searches.size(); ++i) {
        Search& s = searches[i];
        if (!(s.best > kRtEps)) continue;
        const int parent = s.node;
        const int left = add_node(0.0);
        const int right = add_node(0.0);
        GrowNode& node = nodes_[static_cast<std::size_t>(parent)];
        node.column = s.var;
        node.threshold = s.split;
        node.left = left;
        node.right = right;
        node.loss = s.best;
        slot_of[i] = static_cast<int>(next.size());
        next.emplace_back();
        next.back().reset(left, s.left_sum, s.left_weight, 1);
        next.emplace_back();
        next.back().reset(right, s.right_sum, s.right_weight, 1);
      }
      if (next.empty()) break;
      for (std::size_t k = 0; k < m; ++k) {
        const int from = assign_[k];
        if (from < 0) continue;
        const int to = slot_of[static_cast<std::size_t>(from)];
        if (to < 0) {
          assign_[k] = -1;
          continue;
        }
        const GrowNode& node =
            nodes_[static_cast<std::size_t>(searches[static_cast<std::size_t>(from)].node)];
        assign_[k] = (xv(k, node.column) < node.threshold) ? to : to + 1;
      }
      searches = std::move(next);
    }
    if (spec_.gamma > 0) prune(0);
  }

  // Collapses, from the leaves up, a split whose children are both leaves and whose loss change
  // is below `gamma`.
  void prune(int at) {
    GrowNode& node = nodes_[static_cast<std::size_t>(at)];
    if (node.column < 0) return;
    prune(node.left);
    prune(node.right);
    GrowNode& self = nodes_[static_cast<std::size_t>(at)];
    const bool leaves = nodes_[static_cast<std::size_t>(self.left)].column < 0 &&
                        nodes_[static_cast<std::size_t>(self.right)].column < 0;
    if (leaves && self.loss < spec_.gamma) {
      self.column = -1;
      self.left = -1;
      self.right = -1;
    }
  }

  // The grown tree in depth-first order, each leaf's value its step scaled by the shrinkage, and
  // the leaf every training row falls into. A binomial leaf of gbm's takes one Newton step over
  // the bag's rows in it, and a second-order leaf `-G / (H + lambda)`, each sum taken in row
  // order; a Gaussian leaf of gbm's is the mean its split left it.
  detail::Nodes leaves(std::size_t m, std::vector<int>& leaf) {
    detail::Nodes out;
    emit(0, out);
    std::vector<double> num(out.column.size(), 0.0);
    std::vector<double> den(out.column.size(), 0.0);
    for (std::size_t k = 0; k < m; ++k) {
      std::size_t at = 0;
      while (out.column[at] >= 0) {
        at = static_cast<std::size_t>(xv(k, out.column[at]) < out.threshold[at] ? out.left[at]
                                                                                : out.right[at]);
      }
      leaf[k] = static_cast<int>(at);
      if (!inbag_[k]) continue;
      if (spec_.newton) {
        num[at] += z_[k];
        den[at] += h_[k];
      } else if (binomial_) {
        num[at] += wl_[k] * z_[k];
        den[at] += wl_[k] * (yl_[k] - z_[k]) * (1 - yl_[k] + z_[k]);
      }
    }
    for (std::size_t i = 0; i < out.column.size(); ++i) {
      if (out.column[i] >= 0) {
        out.value[i] = 0.0;
        continue;
      }
      double step;
      if (spec_.newton) {
        const double d = den[i] + spec_.lambda;
        step = d == 0 ? 0.0 : -num[i] / d;
      } else if (binomial_) {
        step = den[i] == 0 ? 0.0 : num[i] / den[i];
      } else {
        step = out.value[i];
      }
      out.value[i] = spec_.shrinkage * step;
    }
    return out;
  }

  void emit(int at, detail::Nodes& out) {
    const GrowNode& node = nodes_[static_cast<std::size_t>(at)];
    const auto here = static_cast<std::size_t>(out.add_leaf(node.mean));
    if (node.column < 0) return;
    out.column[here] = node.column;
    out.threshold[here] = node.threshold;
    out.less_left[here] = 1;
    out.left[here] = static_cast<std::int32_t>(out.column.size());
    emit(node.left, out);
    out.right[here] = static_cast<std::int32_t>(out.column.size());
    emit(node.right, out);
  }

  const double* x_;
  const double* y_;
  const double* w_;
  std::size_t n_;
  std::size_t p_;
  bool binomial_;
  BoostSpec spec_;
  const std::vector<int>* rows_ = nullptr;
  std::vector<double> yl_, wl_, z_, h_;
  std::vector<char> inbag_;
  std::vector<int> assign_;
  std::vector<int> order_;
  std::vector<GrowNode> nodes_;
};

}  // namespace

Boosted boost_fit(const double* x, const double* y, const double* w, std::size_t n,
                  std::size_t p, Family family, const BoostSpec& spec, const std::int32_t* fold,
                  std::int32_t n_fold) {
  if (n == 0 || p == 0) throw Error("boosting needs at least one observation and one column.");
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
  detail::check_finite(x, n * p, "boosting", "design");
  detail::check_finite(y, n, "boosting", "response");
  detail::check_finite(w, n, "boosting", "weights");
  const bool binomial = family == Family::binomial;
  for (std::size_t i = 0; i < n; ++i) {
    if (w[i] < 0) throw Error("boosting's case weights are zero or more.");
    if (binomial && y[i] != 0.0 && y[i] != 1.0) {
      throw Error("boosting on a binomial response reads 0 and 1 alone.");
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

  // The fit on every observation first, then one per fold on its complement, scored on the fold.
  const std::size_t fits = cross ? static_cast<std::size_t>(n_fold) + 1 : 1;
  std::vector<std::vector<int>> train(fits), held(fits);
  for (std::size_t i = 0; i < n; ++i) {
    train[0].push_back(static_cast<int>(i));
    if (!cross) continue;
    for (std::size_t g = 0; g + 1 < fits; ++g) {
      (fold[i] == static_cast<std::int32_t>(g) ? held[g + 1] : train[g + 1])
          .push_back(static_cast<int>(i));
    }
  }
  std::vector<double> init(fits, 0.0);
  std::vector<TreeTable> tables(fits);
  std::vector<std::vector<double>> deviance(fits);
  detail::run_tasks(fits, spec.threads, [&](std::size_t f) {
    Booster booster(x, y, w, n, p, binomial, spec);
    booster.run(train[f], held[f], static_cast<std::uint32_t>(f), &init[f], &tables[f],
                &deviance[f]);
  });

  Boosted out;
  out.family = family;
  out.n_column = static_cast<std::int32_t>(p);
  out.init = init[0];
  std::size_t keep = static_cast<std::size_t>(spec.trees);
  if (cross) {
    // gbm's `gbmCrossValErr`: each fold's held-out deviance times the observations it holds,
    // summed over the folds in order and divided by every observation; the first least is kept.
    out.cv_error.assign(keep, 0.0);
    for (std::size_t t = 0; t < keep; ++t) {
      double total = 0.0;
      for (std::size_t g = 1; g < fits; ++g) {
        total += deviance[g][t] * static_cast<double>(held[g].size());
      }
      out.cv_error[t] = total / static_cast<double>(n);
    }
    keep = static_cast<std::size_t>(
               std::min_element(out.cv_error.begin(), out.cv_error.end()) -
               out.cv_error.begin()) + 1;
  }
  const TreeTable& all = tables[0];
  const std::size_t end = static_cast<std::size_t>(all.offset[keep]);
  out.trees.offset.assign(all.offset.begin(), all.offset.begin() + static_cast<std::ptrdiff_t>(keep + 1));
  out.trees.column.assign(all.column.begin(), all.column.begin() + static_cast<std::ptrdiff_t>(end));
  out.trees.threshold.assign(all.threshold.begin(),
                             all.threshold.begin() + static_cast<std::ptrdiff_t>(end));
  out.trees.less_left.assign(all.less_left.begin(),
                             all.less_left.begin() + static_cast<std::ptrdiff_t>(end));
  out.trees.left.assign(all.left.begin(), all.left.begin() + static_cast<std::ptrdiff_t>(end));
  out.trees.right.assign(all.right.begin(), all.right.begin() + static_cast<std::ptrdiff_t>(end));
  out.trees.value.assign(all.value.begin(), all.value.begin() + static_cast<std::ptrdiff_t>(end));
  return out;
}

void boost_predict(const Boosted& model, const double* x, std::size_t n, std::size_t p,
                   double* out) {
  const std::size_t trees = detail::table_trees(model.trees);
  const bool binomial = model.family == Family::binomial;
  for (std::size_t i = 0; i < n; ++i) {
    double f = model.init;
    for (std::size_t t = 0; t < trees; ++t) f += detail::table_value(model.trees, t, x, i, n, p);
    out[i] = binomial ? 1.0 / (1.0 + std::exp(-f)) : f;
  }
}

}  // namespace timesift
