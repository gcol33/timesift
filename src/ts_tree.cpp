#include "ts_tree.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "ts_core.h"
#include "ts_trees_internal.h"

// A split, and every row of the complexity table, is decided by comparing sums, and a fused
// multiply-add rounds once where a product and a sum written apart round twice. A unit in the last
// place is enough to turn a tie between two splits, or between a split and the complexity
// threshold, the other way. Contraction is off so a tree is the same on every machine: clang
// contracts inside an expression by default, and GCC across expressions wherever the target has
// the instruction.
#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

namespace timesift {
namespace {

// ---------------------------------------------------------------------------------------------
// Ordering a column

// One value of a column and the row it belongs to.
struct Keyed {
  double key;
  int row;
};

// Straight insertion over `a[lo..hi]`, which leaves equal keys in the order they came.
void insert_in_order(Keyed* a, int lo, int hi) {
  for (int i = lo + 1; i <= hi; ++i) {
    const Keyed moving = a[i];
    int at = i;
    while (at > lo && a[at - 1].key > moving.key) {
      a[at] = a[at - 1];
      --at;
    }
    a[at] = moving;
  }
}

// The middle one of three keys by value.
double middle_of(double a, double b, double c) {
  return std::max(std::min(a, b), std::min(std::max(a, b), c));
}

// Sorts `a[0..n)` by key: a quicksort (Hoare 1962) partitioning about the middle of the first,
// middle and last keys, with every range of at most eleven finished by insertion. Its order among
// equal keys is not a stable one and is kept all the same, because the order a node's observations
// sit in is the order their weights are summed in: two sums of the same numbers in two orders can
// differ in the last place and turn a tie between two columns the other way. This is the order that
// reproduces rpart's splits in the fixtures. The ranges a partition leaves never overlap, so the
// order they are finished in does not matter.
void sort_keyed(Keyed* a, int n) {
  std::vector<std::pair<int, int>> ranges;
  if (n > 1) ranges.emplace_back(0, n - 1);
  while (!ranges.empty()) {
    const int lo = ranges.back().first;
    const int hi = ranges.back().second;
    ranges.pop_back();
    if (hi - lo < 11) {
      insert_in_order(a, lo, hi);
      continue;
    }
    const double pivot = middle_of(a[lo].key, a[(lo + hi) / 2].key, a[hi].key);
    int up = lo;
    int down = hi;
    while (up < down) {
      while (a[up].key < pivot) ++up;
      while (a[down].key > pivot) --down;
      if (up < down) {
        // Two keys equal to the pivot stay where they are.
        if (a[up].key > a[down].key) std::swap(a[up], a[down]);
        ++up;
        --down;
      }
    }
    while (a[up].key >= pivot && up > lo) --up;
    while (a[down].key <= pivot && down < hi) ++down;
    if (lo < up) ranges.emplace_back(lo, up);
    if (down < hi) ranges.emplace_back(down, hi);
  }
}

// ---------------------------------------------------------------------------------------------
// A node's prediction and the three impurities (Breiman et al. 1984, sections 4.3 and 8.3; for
// counts, Therneau and Atkinson's "An introduction to recursive partitioning using the RPART
// routines", on the Poisson deviance)

// What a node is cut on and what it predicts. Two classes are cut on the Gini index and a node
// predicts the share of ones; a continuous response is cut on the weighted sum of squares and a
// node predicts its mean; a count is cut on the Poisson deviance and a node predicts a rate. A
// leaf's rate is shrunk towards the rate of the units the tree is grown on, the posterior mean of a
// gamma prior of coefficient of variation `shrink` centred on it: with `a = 1 / shrink^2` and
// `b = a / rate`, a node holding events `S` over exposure `W` predicts `(S + a) / (W + b)`. No
// shrinkage, `a = b = 0`, predicts `S / W`. The tree grown on a cross-validation fold's units
// takes its prior from those units.
struct Criterion {
  enum class Kind { classify, squares, count };
  Kind kind = Kind::squares;
  double prior_shape = 0.0;
  double prior_rate = 0.0;

  bool classify() const { return kind == Kind::classify; }
  bool count() const { return kind == Kind::count; }
};

// What a node predicts and its risk on its own observations. Between two classes that is the weight
// of each, the class of the larger (class 0 on a tie) and the weight the other class puts on it;
// for a mean it is the weighted mean and the weighted sum of squares about it; for a count it is the
// shrunk rate and the Poisson deviance of the node's observations about it.
struct Fit {
  double risk = 0.0;
  double mean = 0.0;
  double weight0 = 0.0;
  double weight1 = 0.0;
  int label = 0;
};

// Twice the Poisson log likelihood's shortfall from the saturated model at one observation of
// weight `w`: `2 w (y log(y / rate) - (y - rate))`, the logarithm taken as zero at `y = 0`.
double count_deviance(double y, double w, double rate) {
  const double saturated = y > 0.0 ? y * std::log(y / rate) : 0.0;
  return 2.0 * w * (saturated - (y - rate));
}

Fit summarise(const double* y, const double* w, int n, const Criterion& criterion) {
  Fit fit;
  if (criterion.count()) {
    double events = 0.0;
    double exposure = 0.0;
    for (int i = 0; i < n; ++i) {
      events += y[i] * w[i];
      exposure += w[i];
    }
    fit.mean = (events + criterion.prior_shape) / (exposure + criterion.prior_rate);
    for (int i = 0; i < n; ++i) fit.risk += count_deviance(y[i], w[i], fit.mean);
    return fit;
  }
  if (criterion.classify()) {
    for (int i = 0; i < n; ++i) {
      if (y[i] == 0.0) {
        fit.weight0 += w[i];
      } else {
        fit.weight1 += w[i];
      }
    }
    fit.label = fit.weight0 < fit.weight1 ? 1 : 0;
    fit.risk = fit.label == 1 ? fit.weight0 : fit.weight1;
    return fit;
  }
  double weighted = 0.0;
  double total = 0.0;
  for (int i = 0; i < n; ++i) {
    weighted += y[i] * w[i];
    total += w[i];
  }
  fit.mean = weighted / total;
  for (int i = 0; i < n; ++i) {
    const double d = y[i] - fit.mean;
    fit.risk += d * d * w[i];
  }
  return fit;
}

// The value a node reports: the weighted share of ones, the mean, or the rate.
double reported(const Fit& fit, const Criterion& criterion) {
  if (!criterion.classify()) return fit.mean;
  const double total = fit.weight0 + fit.weight1;
  return total > 0 ? fit.weight1 / total : 0.0;
}

// The loss of the node's prediction on one observation: a misclassification, the squared error, or
// the Poisson deviance.
double loss_at(const Fit& fit, double y, const Criterion& criterion) {
  if (criterion.classify()) return static_cast<int>(y) == fit.label ? 0.0 : 1.0;
  if (criterion.count()) return count_deviance(y, 1.0, fit.mean);
  const double d = y - fit.mean;
  return d * d;
}

// A threshold on one column, and which side the values below it go to.
struct Cut {
  double gain = 0.0;
  double threshold = 0.0;
  bool below_left = true;
};

// `total` times the Gini index of a class holding `part` of it, `p (1 - p)`.
double gini_term(double part, double total) {
  const double p = part / total;
  return total * (p * (1.0 - p));
}

// The best cut of `n` observations sorted by `x`, each child keeping at least `edge`, under the Gini
// index of two classes. Observations move one at a time from the right child to the left, and a
// cut is tried only between two distinct values. The gain is the parent's weighted impurity less
// the children's; the side the values below the cut go to is the one of fewer ones.
Cut scan_gini(const double* x, const double* y, const double* w, int n, int edge) {
  double left_class[2] = {0.0, 0.0};
  double right_class[2] = {0.0, 0.0};
  double left_weight = 0.0;
  double right_weight = 0.0;
  int left_count = 0;
  int right_count = 0;
  for (int i = 0; i < n; ++i) {
    right_weight += w[i];
    right_class[y[i] == 0.0 ? 0 : 1] += w[i];
    ++right_count;
  }
  double parent = 0.0;
  for (int c = 0; c < 2; ++c) parent += gini_term(right_class[c], right_weight);

  double lowest = parent;
  int after = 0;
  bool below_left = true;
  for (int i = 0; right_count > edge; ++i) {
    const int c = y[i] == 0.0 ? 0 : 1;
    right_weight -= w[i];
    left_weight += w[i];
    --right_count;
    ++left_count;
    right_class[c] -= w[i];
    left_class[c] += w[i];
    if (left_count < edge || x[i + 1] == x[i]) continue;
    // The children's impurity and the mean class label on each side, summed class by class.
    double impurity = 0.0;
    double left_label = 0.0;
    double right_label = 0.0;
    for (int k = 0; k < 2; ++k) {
      const double pl = left_class[k] / left_weight;
      impurity += left_weight * (pl * (1.0 - pl));
      left_label += pl * k;
      const double pr = right_class[k] / right_weight;
      impurity += right_weight * (pr * (1.0 - pr));
      right_label += pr * k;
    }
    if (impurity < lowest) {
      lowest = impurity;
      after = i;
      below_left = left_label < right_label;
    }
  }
  Cut cut;
  cut.gain = parent - lowest;
  if (cut.gain > 0) {
    cut.below_left = below_left;
    cut.threshold = (x[after] + x[after + 1]) / 2;
  }
  return cut;
}

// The same scan under the weighted sum of squares. With the responses centred on the parent's
// mean, the reduction a cut makes is `S_l^2 / W_l + S_r^2 / W_r`, `S` a side's weighted sum and `W`
// its weight; the gain is that reduction as a share of the parent's risk `risk`, and the values
// below the cut go to the side of the smaller sum.
Cut scan_squares(const double* x, const double* y, const double* w, int n, int edge,
                 double risk) {
  double right_sum = 0.0;
  double right_weight = 0.0;
  for (int i = 0; i < n; ++i) {
    right_sum += y[i] * w[i];
    right_weight += w[i];
  }
  const double centre = right_sum / right_weight;
  right_sum = 0.0;
  double left_sum = 0.0;
  double left_weight = 0.0;
  int left_count = 0;
  int right_count = n;

  double largest = 0.0;
  int after = 0;
  bool below_left = true;
  for (int i = 0; right_count > edge; ++i) {
    left_weight += w[i];
    right_weight -= w[i];
    ++left_count;
    --right_count;
    const double moved = (y[i] - centre) * w[i];
    left_sum += moved;
    right_sum -= moved;
    if (x[i + 1] == x[i] || left_count < edge) continue;
    const double reduction =
        left_sum * left_sum / left_weight + right_sum * right_sum / right_weight;
    if (reduction > largest) {
      largest = reduction;
      after = i;
      below_left = left_sum < right_sum;
    }
  }
  Cut cut;
  cut.gain = largest / risk;
  if (largest > 0) {
    cut.below_left = below_left;
    cut.threshold = (x[after] + x[after + 1]) / 2;
  }
  return cut;
}

// `s log(s / t)` of a node holding events `s` over exposure `t`, which is zero for a node without
// events.
double event_term(double s, double t) { return s > 0.0 ? s * std::log(s / t) : 0.0; }

// The same scan under the Poisson deviance of the unshrunk rates. A node's deviance is
// `2 (sum y log y - S log(S / W))`, so the reduction a cut makes is
// `2 (S_l log(S_l / W_l) + S_r log(S_r / W_r) - S log(S / W))`; the gain is that reduction as a
// share of the parent's risk `risk`, and the values below the cut go to the side of the lower rate.
Cut scan_counts(const double* x, const double* y, const double* w, int n, int edge,
                double risk) {
  double right_events = 0.0;
  double right_weight = 0.0;
  for (int i = 0; i < n; ++i) {
    right_events += y[i] * w[i];
    right_weight += w[i];
  }
  const double whole = event_term(right_events, right_weight);
  double left_events = 0.0;
  double left_weight = 0.0;
  int left_count = 0;
  int right_count = n;

  double largest = 0.0;
  int after = 0;
  bool below_left = true;
  for (int i = 0; right_count > edge; ++i) {
    left_weight += w[i];
    right_weight -= w[i];
    ++left_count;
    --right_count;
    const double moved = y[i] * w[i];
    left_events += moved;
    right_events -= moved;
    if (x[i + 1] == x[i] || left_count < edge) continue;
    const double reduction = 2.0 * (event_term(left_events, left_weight) +
                                    event_term(right_events, right_weight) - whole);
    if (reduction > largest) {
      largest = reduction;
      after = i;
      below_left = left_events * right_weight < right_events * left_weight;
    }
  }
  Cut cut;
  cut.gain = risk > 0.0 ? largest / risk : 0.0;
  if (cut.gain > 0) {
    cut.below_left = below_left;
    cut.threshold = (x[after] + x[after + 1]) / 2;
  }
  return cut;
}

Cut best_cut(const double* x, const double* y, const double* w, int n, int edge,
             const Criterion& criterion, double risk) {
  switch (criterion.kind) {
    case Criterion::Kind::classify: return scan_gini(x, y, w, n, edge);
    case Criterion::Kind::count: return scan_counts(x, y, w, n, edge, risk);
    case Criterion::Kind::squares: break;
  }
  return scan_squares(x, y, w, n, edge, risk);
}

// Whether a gain is a real one rather than rounding: above 1e-10 of the largest gain seen so far,
// the gain itself counted first.
class GainFloor {
 public:
  bool admits(double gain) {
    if (gain > largest_) largest_ = gain;
    return gain > largest_ * 1e-10;
  }

 private:
  double largest_ = 0.0;
};

// The split a node is given.
struct Rule {
  int column = -1;
  double threshold = 0.0;
  bool below_left = true;

  bool sends_left(double v) const { return (v < threshold) == below_left; }
};

// Keeps the first rule of the largest admitted gain among the candidates offered in turn.
class RuleChoice {
 public:
  void offer(GainFloor& floor, int column, const Cut& cut) {
    if (!floor.admits(cut.gain)) return;
    if (rule_ && !(cut.gain > gain_)) return;
    rule_ = Rule{column, cut.threshold, cut.below_left};
    gain_ = cut.gain;
  }
  const std::optional<Rule>& rule() const { return rule_; }

 private:
  std::optional<Rule> rule_;
  double gain_ = 0.0;
};

// ---------------------------------------------------------------------------------------------
// The tree and its cost-complexity pruning (Breiman et al. 1984, chapter 3)

struct CartNode {
  Fit fit;
  // The complexity at which the node's subtree is pruned away: while growing, an upper bound on
  // it that stops a node whose split could never be kept.
  double cp = 0.0;
  double weight = 0.0;
  int count = 0;
  Rule rule;
  std::unique_ptr<CartNode> left;
  std::unique_ptr<CartNode> right;
};

// A grown subtree's splits and risk, after the weakest links below it are cut.
struct Subtree {
  int splits = 0;
  double risk = 0.0;
};

// One row of the complexity table, largest complexity first: the subtree's risk and splits there,
// and the held-out loss and its sum of squares, the latter turned into a standard error at the end.
struct CpRow {
  double cp = 0.0;
  double risk = 0.0;
  int splits = 0;
  double xrisk = 0.0;
  double xstd = 0.0;
};

// Grows the tree on every observation, and on each fold's complement where folds are given. The
// observations of a node are a stretch of `order_`, which holds for every column the rows sorted by
// that column; a split rearranges each column's stretch into the left child's rows then the right
// child's, each in the order it held.
class CartBuilder {
 public:
  CartBuilder(const double* x, const double* y, const double* w, std::size_t n, std::size_t p,
              const Criterion& criterion, const TreeSpec& spec)
      : x_(x), y_(y), w_(w), n_(static_cast<int>(n)), p_(static_cast<int>(p)),
        criterion_(criterion), min_split_(spec.min_split), min_leaf_(spec.min_leaf),
        deepest_((1LL << spec.max_depth) - 1), cp_share_(spec.cp), order_(n * p), goes_left_(n),
        spill_(n), column_x_(n), column_y_(n), column_w_(n), node_y_(n), node_w_(n) {
    std::vector<Keyed> keyed(n);
    for (int v = 0; v < p_; ++v) {
      const double* col = column(v);
      for (int r = 0; r < n_; ++r) keyed[r] = {col[r], r};
      sort_keyed(keyed.data(), n_);
      int* block = rows_by(v);
      for (int r = 0; r < n_; ++r) block[r] = keyed[r].row;
    }
  }

  // The tree on every observation with its complexity table, and the table's held-out losses
  // where there are two folds or more.
  std::unique_ptr<CartNode> build(const std::int32_t* fold, int n_fold,
                                  std::vector<CpRow>& table) {
    if (n_fold > 1) initial_order_ = order_;
    auto root = std::make_unique<CartNode>();
    double total = 0.0;
    for (int r = 0; r < n_; ++r) total += w_[r];
    root->fit = summarise(y_, w_, n_, criterion_);
    root->count = n_;
    root->weight = total;
    root->cp = root->fit.risk;
    alpha_ = cp_share_ * root->fit.risk;
    grow(*root, 1, 0, n_);

    CpRow top;
    top.cp = root->cp;
    top.risk = root->fit.risk;
    table.assign(1, top);
    if (root->left) {
      gather_complexities(*root, root->cp, table);
      tally(*root, root->cp, 0, table);
      if (n_fold > 1) cross_validate(fold, n_fold, table);
    }
    return root;
  }

  // The pruning threshold on the scale of the risk.
  double alpha() const { return alpha_; }

 private:
  const double* column(int v) const { return x_ + static_cast<std::size_t>(v) * n_; }
  int* rows_by(int v) { return order_.data() + static_cast<std::size_t>(v) * n_; }

  Subtree leaf(CartNode& node) const {
    node.cp = alpha_;
    return {0, node.fit.risk};
  }

  // Grows the node holding the stretch `[begin, end)`, numbered `id` (1 at the root, 2k and
  // 2k + 1 below k), and reports its subtree after the weakest links are cut. A split is kept
  // where the subtree's complexity `(R(t) - R(T_t)) / (|T_t| - 1)` exceeds `alpha_`.
  Subtree grow(CartNode& node, long long id, int begin, int end) {
    double bound = node.fit.risk;
    if (id > 1) {
      const int* rows = rows_by(0);
      double total = 0.0;
      for (int i = begin; i < end; ++i) {
        const int r = rows[i];
        node_y_[i - begin] = y_[r];
        node_w_[i - begin] = w_[r];
        total += w_[r];
      }
      node.fit = summarise(node_y_.data(), node_w_.data(), end - begin, criterion_);
      node.count = end - begin;
      node.weight = total;
      bound = std::min(node.fit.risk, node.cp);
    }
    if (node.count < min_split_ || bound <= alpha_ || id > deepest_) return leaf(node);

    const std::optional<Rule> rule = choose_rule(node.fit.risk, begin, end);
    if (!rule) return leaf(node);
    node.rule = *rule;
    const int middle = begin + send(node.rule, begin, end);
    const double risk = node.fit.risk;

    node.left = std::make_unique<CartNode>();
    node.left->cp = bound - alpha_;
    Subtree low = grow(*node.left, 2 * id, begin, middle);

    double right_bound = std::max((risk - low.risk) / (low.splits + 1), risk - node.left->fit.risk);
    right_bound = std::min(right_bound, node.cp);
    node.right = std::make_unique<CartNode>();
    node.right->cp = right_bound - alpha_;
    Subtree high = grow(*node.right, 2 * id + 1, middle, end);

    // A child whose own complexity lies below the link to it is cut back to a leaf, the weaker
    // child first, and the link measured again.
    const auto link = [&] {
      return (risk - (low.risk + high.risk)) / (low.splits + high.splits + 1);
    };
    const bool left_weaker = node.right->cp > node.left->cp;
    Subtree& weak = left_weaker ? low : high;
    Subtree& strong = left_weaker ? high : low;
    const CartNode& weak_node = left_weaker ? *node.left : *node.right;
    const CartNode& strong_node = left_weaker ? *node.right : *node.left;
    if (link() > weak_node.cp) {
      weak = {0, weak_node.fit.risk};
      if (link() > strong_node.cp) strong = {0, strong_node.fit.risk};
    }
    node.cp = link();

    if (node.cp <= alpha_) {
      node.left.reset();
      node.right.reset();
      return {0, risk};
    }
    return {low.splits + high.splits + 1, low.risk + high.risk};
  }

  // The best rule over every column in turn, among the node's observations of positive weight.
  std::optional<Rule> choose_rule(double risk, int begin, int end) {
    RuleChoice choice;
    for (int v = 0; v < p_; ++v) {
      const int* rows = rows_by(v);
      const double* col = column(v);
      int k = 0;
      for (int i = begin; i < end; ++i) {
        const int r = rows[i];
        if (!(w_[r] > 0)) continue;
        column_x_[k] = col[r];
        column_y_[k] = y_[r];
        column_w_[k] = w_[r];
        ++k;
      }
      if (k == 0 || column_x_[0] == column_x_[k - 1]) continue;
      choice.offer(floor_, v,
                   best_cut(column_x_.data(), column_y_.data(), column_w_.data(), k, min_leaf_,
                            criterion_, risk));
    }
    return choice.rule();
  }

  // Sends the stretch's rows to the two sides of `rule`, rearranging every column's stretch, and
  // returns how many went left.
  int send(const Rule& rule, int begin, int end) {
    const double* col = column(rule.column);
    const int* rows = rows_by(0);
    int n_left = 0;
    for (int i = begin; i < end; ++i) {
      const int r = rows[i];
      const bool left = rule.sends_left(col[r]);
      goes_left_[r] = left ? 1 : 0;
      n_left += left ? 1 : 0;
    }
    for (int v = 0; v < p_; ++v) {
      int* block = rows_by(v);
      int kept = begin;
      int spilled = 0;
      for (int i = begin; i < end; ++i) {
        const int r = block[i];
        if (goes_left_[r]) {
          block[kept++] = r;
        } else {
          spill_[spilled++] = r;
        }
      }
      std::copy(spill_.begin(), spill_.begin() + spilled, block + kept);
    }
    return n_left;
  }

  // The distinct complexities of the nested pruned subtrees, each node's complexity first capped at
  // its parent's so the sequence is nested, largest first.
  void gather_complexities(CartNode& node, double parent, std::vector<CpRow>& table) const {
    node.cp = std::min(node.cp, parent);
    const double own = std::max(node.cp, alpha_);
    if (node.left) {
      gather_complexities(*node.left, own, table);
      gather_complexities(*node.right, own, table);
    }
    if (!(own < parent)) return;
    auto at = table.begin();
    for (; at != table.end(); ++at) {
      if (own == at->cp) return;
      if (own > at->cp) break;
    }
    CpRow row;
    row.cp = own;
    table.insert(at, row);
  }

  // Adds each leaf's risk, and each split's count, to every row of the table whose pruned subtree
  // keeps it. Returns the first row, from the smallest complexity up, the node's parent reaches.
  static std::ptrdiff_t tally(const CartNode& node, double parent, int splits,
                              std::vector<CpRow>& table) {
    std::ptrdiff_t row;
    if (node.left) {
      tally(*node.left, node.cp, 0, table);
      row = tally(*node.right, node.cp, splits + 1, table);
    } else {
      row = static_cast<std::ptrdiff_t>(table.size()) - 1;
    }
    for (; row >= 0 && table[row].cp < parent; --row) {
      table[row].risk += node.fit.risk;
      table[row].splits += splits;
    }
    return row;
  }

  static void cap_complexity(CartNode& node, double parent) {
    if (node.cp > parent) node.cp = parent;
    if (!node.left) return;
    cap_complexity(*node.left, node.cp);
    cap_complexity(*node.right, node.cp);
  }

  // V-fold cross-validation of the complexity table (Breiman et al. 1984, section 3.4.2): a tree
  // grown on each fold's complement, with every complexity rescaled to the complement's weight,
  // and each held-out observation's loss read at the geometric midpoint of every row's interval.
  void cross_validate(const std::int32_t* fold, int n_fold, std::vector<CpRow>& table) {
    const double alpha_whole = alpha_;
    const Criterion criterion_whole = criterion_;
    const std::size_t rows = table.size();
    std::vector<double> probe(rows);
    probe[0] = 10 * table[0].cp;
    for (std::size_t i = 1; i < rows; ++i) probe[i] = std::sqrt(table[i - 1].cp * table[i].cp);
    double total = 0.0;
    for (int r = 0; r < n_; ++r) total += w_[r];
    double previous = total;
    std::vector<int> held;

    for (int g = 0; g < n_fold; ++g) {
      for (int v = 0; v < p_; ++v) {
        const int* from = initial_order_.data() + static_cast<std::size_t>(v) * n_;
        int* to = rows_by(v);
        int kept = 0;
        for (int i = 0; i < n_; ++i) {
          if (fold[from[i]] != g) to[kept++] = from[i];
        }
      }
      held.clear();
      int kept = 0;
      double weight = 0.0;
      for (int r = 0; r < n_; ++r) {
        if (fold[r] == g) {
          held.push_back(r);
          continue;
        }
        node_y_[kept] = y_[r];
        node_w_[kept] = w_[r];
        weight += w_[r];
        ++kept;
      }
      if (criterion_.count() && criterion_.prior_shape > 0.0) {
        double events = 0.0;
        for (int i = 0; i < kept; ++i) events += node_y_[i] * node_w_[i];
        criterion_.prior_rate = events > 0.0 ? criterion_.prior_shape * weight / events : 0.0;
      }
      const double shrink = weight / previous;
      for (double& c : probe) c *= shrink;
      alpha_ *= shrink;
      previous = weight;

      CartNode root;
      root.count = kept;
      root.fit = summarise(node_y_.data(), node_w_.data(), kept, criterion_);
      root.cp = root.fit.risk;
      grow(root, 1, 0, kept);
      cap_complexity(root, root.cp);

      for (const int r : held) {
        const CartNode* at = &root;
        for (std::size_t c = 0; c < rows; ++c) {
          while (probe[c] < at->cp && at->left) {
            const double v = x_[r + static_cast<std::size_t>(at->rule.column) * n_];
            at = at->rule.sends_left(v) ? at->left.get() : at->right.get();
          }
          const double loss = loss_at(at->fit, y_[r], criterion_);
          table[c].xrisk += loss * w_[r];
          table[c].xstd += loss * loss * w_[r];
        }
      }
    }
    for (CpRow& row : table) row.xstd = std::sqrt(row.xstd - row.xrisk * row.xrisk / total);
    alpha_ = alpha_whole;
    criterion_ = criterion_whole;
  }

  const double* x_;
  const double* y_;
  const double* w_;
  int n_;
  int p_;
  Criterion criterion_;
  int min_split_;
  int min_leaf_;
  long long deepest_;
  double cp_share_;
  double alpha_ = 0.0;
  GainFloor floor_;
  std::vector<int> order_;
  std::vector<int> initial_order_;
  std::vector<std::int8_t> goes_left_;
  std::vector<int> spill_;
  std::vector<double> column_x_;
  std::vector<double> column_y_;
  std::vector<double> column_w_;
  std::vector<double> node_y_;
  std::vector<double> node_w_;
};

// Writes the node and, where it keeps its split, its subtree into `out`, depth first and left
// before right.
void write_node(const CartNode& node, long long id, double scale, double alpha,
                const Criterion& criterion, Tree& out) {
  const std::size_t at = out.number.size();
  const bool split = node.left && node.cp > alpha;
  out.number.push_back(static_cast<std::int32_t>(id));
  out.n.push_back(node.count);
  out.weight.push_back(node.weight);
  out.risk.push_back(node.fit.risk);
  out.complexity.push_back(node.cp * scale);
  out.value.push_back(reported(node.fit, criterion));
  out.column.push_back(split ? node.rule.column : -1);
  out.threshold.push_back(split ? node.rule.threshold : 0.0);
  out.less_left.push_back(split && node.rule.below_left ? 1 : 0);
  out.left.push_back(-1);
  out.right.push_back(-1);
  if (!split) return;
  out.left[at] = static_cast<std::int32_t>(out.number.size());
  write_node(*node.left, 2 * id, scale, alpha, criterion, out);
  out.right[at] = static_cast<std::int32_t>(out.number.size());
  write_node(*node.right, 2 * id + 1, scale, alpha, criterion, out);
}

// Copies node `i` of `from` into `out`, and its subtree where its complexity lies above `cp`.
void copy_pruned(const Tree& from, std::size_t i, double cp, Tree& out) {
  const std::size_t at = out.number.size();
  const bool split = from.column[i] >= 0 && !(from.complexity[i] <= cp);
  out.number.push_back(from.number[i]);
  out.n.push_back(from.n[i]);
  out.weight.push_back(from.weight[i]);
  out.risk.push_back(from.risk[i]);
  out.complexity.push_back(from.complexity[i]);
  out.value.push_back(from.value[i]);
  out.column.push_back(split ? from.column[i] : -1);
  out.threshold.push_back(split ? from.threshold[i] : 0.0);
  out.less_left.push_back(split ? from.less_left[i] : 0);
  out.left.push_back(-1);
  out.right.push_back(-1);
  if (!split) return;
  out.left[at] = static_cast<std::int32_t>(out.number.size());
  copy_pruned(from, static_cast<std::size_t>(from.left[i]), cp, out);
  out.right[at] = static_cast<std::int32_t>(out.number.size());
  copy_pruned(from, static_cast<std::size_t>(from.right[i]), cp, out);
}

// ---------------------------------------------------------------------------------------------
// The random forest (Breiman 2001)

// Rows and their running weight. A draw is the first row whose running weight exceeds a uniform
// times the total, so a row of zero weight is never drawn.
struct WeightedPool {
  std::vector<int> rows;
  std::vector<double> running;

  void add(int r, double w) {
    rows.push_back(r);
    running.push_back((running.empty() ? 0.0 : running.back()) + w);
  }

  double total() const { return running.back(); }

  int draw(detail::Stream& stream) const {
    const double target = stream.uniform() * total();
    std::size_t at = static_cast<std::size_t>(
        std::upper_bound(running.begin(), running.end(), target) - running.begin());
    if (at >= running.size()) at = running.size() - 1;
    return rows[at];
  }
};

// One bootstrap draw per pool and the number of rows each gives.
struct Bootstrap {
  std::vector<WeightedPool> pools;
  std::vector<int> draws;
};

// Grows one tree of a forest. The drawn rows are held in ascending order, a row drawn twice held
// twice; a node is a stretch of them, and its children keep the order it held. Each column tried
// at a node is sorted afresh, ties kept in the node's order, and every drawn row weighs one.
class ForestTree {
 public:
  ForestTree(const double* x, const double* y, std::size_t n, std::size_t p,
             const Criterion& criterion, const ForestSpec& spec)
      : x_(x), y_(y), n_(n), p_(p), criterion_(criterion), min_leaf_(spec.min_leaf),
        mtry_(static_cast<std::size_t>(spec.mtry)) {}

  detail::Nodes grow(const Bootstrap& boot, std::uint32_t seed, std::uint32_t tree) {
    detail::Stream stream(seed, tree);
    std::vector<int> rows = draw_rows(boot, stream);
    const std::size_t m = rows.size();

    std::vector<int> columns(p_);
    for (std::size_t j = 0; j < p_; ++j) columns[j] = static_cast<int>(j);
    std::vector<int> tried;
    std::vector<int> spill(m);
    std::vector<double> ys(m);
    const std::vector<double> ones(m, 1.0);
    Sorted sorted(m);
    GainFloor floor;

    struct Pending {
      std::size_t begin;
      std::size_t end;
      std::int32_t parent;
      bool is_right;
    };
    std::vector<Pending> pending{{0, m, -1, false}};
    detail::Nodes out;
    while (!pending.empty()) {
      const Pending node = pending.back();
      pending.pop_back();
      const auto at = static_cast<std::int32_t>(out.column.size());
      if (node.parent >= 0) (node.is_right ? out.right : out.left)[node.parent] = at;

      const int k = static_cast<int>(node.end - node.begin);
      const int* held = rows.data() + node.begin;
      bool pure = true;
      for (int i = 0; i < k; ++i) {
        ys[i] = y_[held[i]];
        pure = pure && ys[i] == ys[0];
      }
      const Fit fit = summarise(ys.data(), ones.data(), k, criterion_);
      out.add_leaf(reported(fit, criterion_));
      if (k < 2 * min_leaf_ || pure) continue;

      // The node's columns, drawn from an arrangement of the column indices the tree keeps from
      // node to node.
      detail::draw_columns(stream, columns, mtry_, tried);
      RuleChoice choice;
      for (const int v : tried) {
        sorted.load(x_ + static_cast<std::size_t>(v) * n_, y_, held, k);
        if (sorted.x[0] == sorted.x[k - 1]) continue;
        choice.offer(floor, v,
                     best_cut(sorted.x.data(), sorted.y.data(), ones.data(), k, min_leaf_,
                              criterion_, fit.risk));
      }
      if (!choice.rule()) continue;
      const Rule rule = *choice.rule();

      const double* col = x_ + static_cast<std::size_t>(rule.column) * n_;
      std::size_t n_left = 0;
      std::size_t n_right = 0;
      for (std::size_t i = node.begin; i < node.end; ++i) {
        const int r = rows[i];
        if (rule.sends_left(col[r])) {
          rows[node.begin + n_left++] = r;
        } else {
          spill[n_right++] = r;
        }
      }
      std::copy(spill.begin(), spill.begin() + static_cast<std::ptrdiff_t>(n_right),
                rows.begin() + static_cast<std::ptrdiff_t>(node.begin + n_left));
      out.column[at] = rule.column;
      out.threshold[at] = rule.threshold;
      out.less_left[at] = rule.below_left ? 1 : 0;
      pending.push_back({node.begin + n_left, node.end, at, true});
      pending.push_back({node.begin, node.begin + n_left, at, false});
    }
    return out;
  }

 private:
  // One column's values at a node in ascending order with the responses beside them, ties in the
  // node's order.
  struct Sorted {
    explicit Sorted(std::size_t m) : keyed(m), x(m), y(m) {}

    void load(const double* col, const double* response, const int* held, int k) {
      for (int i = 0; i < k; ++i) keyed[i] = {col[held[i]], i};
      std::sort(keyed.begin(), keyed.begin() + k);
      for (int i = 0; i < k; ++i) {
        x[i] = keyed[i].first;
        y[i] = response[held[keyed[i].second]];
      }
    }

    std::vector<std::pair<double, int>> keyed;
    std::vector<double> x;
    std::vector<double> y;
  };

  std::vector<int> draw_rows(const Bootstrap& boot, detail::Stream& stream) const {
    std::vector<int> times(n_, 0);
    for (std::size_t u = 0; u < boot.pools.size(); ++u) {
      for (int d = 0; d < boot.draws[u]; ++d) {
        ++times[static_cast<std::size_t>(boot.pools[u].draw(stream))];
      }
    }
    std::vector<int> rows;
    for (std::size_t r = 0; r < n_; ++r) rows.insert(rows.end(), times[r], static_cast<int>(r));
    return rows;
  }

  const double* x_;
  const double* y_;
  std::size_t n_;
  std::size_t p_;
  Criterion criterion_;
  int min_leaf_;
  std::size_t mtry_;
};

}  // namespace

Tree tree_fit(const double* x, const double* y, const double* w, std::size_t n, std::size_t p,
              Family family, const TreeSpec& spec, const std::int32_t* fold,
              std::int32_t n_fold) {
  if (n == 0 || p == 0) throw Error("a tree needs at least one observation and one column.");
  if (spec.min_split < 1 || spec.min_leaf < 1) {
    throw Error("a tree's `min_split` and `min_leaf` are at least one observation.");
  }
  if (!(spec.cp >= 0.0)) throw Error("a tree's `cp` is zero or more.");
  if (spec.max_depth < 0 || spec.max_depth > 30) {
    throw Error("a tree's `max_depth` is between 0 and 30, as rpart's is.");
  }
  detail::check_finite(x, n * p, "a tree", "design");
  detail::check_finite(y, n, "a tree", "response");
  detail::check_finite(w, n, "a tree", "weights");
  double total = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    if (w[i] < 0) throw Error("a tree's case weights are zero or more.");
    total += w[i];
    if (family == Family::binomial && y[i] != 0.0 && y[i] != 1.0) {
      throw Error("a tree on a binomial response reads 0 and 1 alone.");
    }
    if (family == Family::poisson && y[i] < 0.0) {
      throw Error("a tree on a count response reads values of zero or more.");
    }
  }
  if (!(total > 0)) throw Error("a tree's case weights sum to more than zero.");
  if (!(spec.shrink >= 0.0)) throw Error("a tree's `shrink` is zero or more.");
  const bool cross = fold != nullptr && n_fold > 1;
  if (cross) {
    for (std::size_t i = 0; i < n; ++i) {
      if (fold[i] < 0 || fold[i] >= n_fold) {
        throw Error("a fold index lies between 0 and the number of folds less one.");
      }
    }
  }

  Criterion criterion;
  if (family == Family::binomial) {
    criterion.kind = Criterion::Kind::classify;
  } else if (family == Family::poisson) {
    criterion.kind = Criterion::Kind::count;
    double events = 0.0;
    for (std::size_t i = 0; i < n; ++i) events += y[i] * w[i];
    if (!(events > 0.0)) throw Error("a tree on a count response needs one count above zero.");
    if (spec.shrink > 0.0) {
      criterion.prior_shape = 1.0 / (spec.shrink * spec.shrink);
      criterion.prior_rate = criterion.prior_shape * total / events;
    }
  }
  CartBuilder builder(x, y, w, n, p, criterion, spec);
  std::vector<CpRow> table;
  const std::unique_ptr<CartNode> root = builder.build(fold, cross ? n_fold : 0, table);

  Tree out;
  out.family = family;
  out.root_risk = root->fit.risk;
  const double scale = root->fit.risk > 0 ? 1 / root->fit.risk : 1.0;
  write_node(*root, 1, scale, builder.alpha(), criterion, out);
  const bool validated = cross && root->left != nullptr;
  for (const CpRow& row : table) {
    out.cp.push_back(row.cp * scale);
    out.nsplit.push_back(row.splits);
    out.rel_error.push_back(row.risk * scale);
    if (validated) {
      out.xerror.push_back(row.xrisk * scale);
      out.xstd.push_back(row.xstd * scale);
    }
  }
  return out;
}

Tree tree_prune(const Tree& tree, double cp) {
  Tree out;
  out.family = tree.family;
  out.root_risk = tree.root_risk;
  out.cp = tree.cp;
  out.nsplit = tree.nsplit;
  out.rel_error = tree.rel_error;
  out.xerror = tree.xerror;
  out.xstd = tree.xstd;
  if (!tree.number.empty()) copy_pruned(tree, 0, cp, out);
  return out;
}

void tree_predict(const Tree& tree, const double* x, std::size_t n, std::size_t p, double* out) {
  if (tree.number.empty()) throw Error("the tree holds no node to predict from.");
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = tree.value[detail::leaf_of(tree.column.data(), tree.threshold.data(),
                                        tree.less_left.data(), tree.left.data(),
                                        tree.right.data(), x, i, n, p)];
  }
}

Forest forest_fit(const double* x, const double* y, const double* w, std::size_t n, std::size_t p,
                  Family family, const ForestSpec& spec) {
  if (n == 0 || p == 0) throw Error("a forest needs at least one observation and one column.");
  if (spec.trees < 1) throw Error("a forest has at least one tree.");
  if (spec.mtry < 1 || static_cast<std::size_t>(spec.mtry) > p) {
    throw Error("a forest's `mtry` lies between 1 and the " + std::to_string(p) +
                " columns it reads.");
  }
  if (spec.min_leaf < 1) throw Error("a forest's `min_leaf` is at least one observation.");
  detail::check_finite(x, n * p, "a forest", "design");
  detail::check_finite(y, n, "a forest", "response");
  detail::check_finite(w, n, "a forest", "weights");
  if (family == Family::poisson) {
    for (std::size_t i = 0; i < n; ++i) {
      if (y[i] < 0.0) throw Error("a forest on a count response reads values of zero or more.");
    }
  }
  // A forest cuts a continuous response on its variance and a count on it too, a leaf holding the
  // mean count of its draws.
  const bool classify = family == Family::binomial;
  Criterion criterion;
  if (classify) criterion.kind = Criterion::Kind::classify;
  if (spec.balance && !classify) {
    throw Error("a balanced forest draws from each class, and reads a binomial response.");
  }

  // Every observation, as many draws as there are observations; or, balanced, each class on its
  // own and from each the count of the smaller one, counting the observations that weigh anything.
  Bootstrap boot;
  if (spec.balance) {
    WeightedPool by_class[2];
    int weighing[2] = {0, 0};
    for (std::size_t i = 0; i < n; ++i) {
      if (w[i] < 0) throw Error("a forest's case weights are zero or more.");
      if (y[i] != 0.0 && y[i] != 1.0) throw Error("a forest on a binomial response reads 0 and 1 alone.");
      const int c = y[i] == 1.0 ? 1 : 0;
      by_class[c].add(static_cast<int>(i), w[i]);
      if (w[i] > 0) ++weighing[c];
    }
    if (weighing[0] == 0 || weighing[1] == 0) {
      throw Error("a balanced forest draws from each class, and one class weighs nothing.");
    }
    const int each = std::min(weighing[0], weighing[1]);
    boot.pools = {by_class[0], by_class[1]};
    boot.draws = {each, each};
  } else {
    WeightedPool all;
    for (std::size_t i = 0; i < n; ++i) {
      if (w[i] < 0) throw Error("a forest's case weights are zero or more.");
      if (classify && y[i] != 0.0 && y[i] != 1.0) {
        throw Error("a forest on a binomial response reads 0 and 1 alone.");
      }
      all.add(static_cast<int>(i), w[i]);
    }
    if (!(all.total() > 0)) throw Error("a forest's case weights sum to more than zero.");
    boot.pools = {all};
    boot.draws = {static_cast<int>(n)};
  }

  // Every tree is a function of the seed, its own index and the data alone, so trees run at once
  // where the caller asks for it and land in their own slot.
  const std::size_t trees = static_cast<std::size_t>(spec.trees);
  std::vector<detail::Nodes> grown(trees);
  detail::run_tasks(trees, spec.threads, [&](std::size_t t) {
    ForestTree builder(x, y, n, p, criterion, spec);
    grown[t] = builder.grow(boot, spec.seed, static_cast<std::uint32_t>(t));
  });

  Forest out;
  out.family = family;
  out.n_column = static_cast<std::int32_t>(p);
  for (detail::Nodes& nodes : grown) {
    detail::append(out.trees, nodes);
    nodes = detail::Nodes();
  }
  return out;
}

void forest_predict(const Forest& forest, const double* x, std::size_t n, std::size_t p,
                    double* out) {
  const std::size_t trees = detail::table_trees(forest.trees);
  if (trees == 0) throw Error("the forest holds no tree to predict from.");
  for (std::size_t i = 0; i < n; ++i) {
    double sum = 0.0;
    for (std::size_t t = 0; t < trees; ++t) sum += detail::table_value(forest.trees, t, x, i, n, p);
    out[i] = sum / static_cast<double>(trees);
  }
}

void forest_stream(std::uint32_t seed, std::uint32_t tree, std::size_t n, std::uint32_t* out) {
  detail::Stream stream(seed, tree);
  for (std::size_t i = 0; i < n; ++i) out[i] = stream.next();
}

std::vector<Tree> tree_fits(const double* x, std::size_t n, std::size_t p, const double* y,
                            const double* w, std::size_t r, Family family, const TreeSpec& spec,
                            const std::int32_t* fold, const std::int32_t* n_fold, int threads) {
  return detail::fit_responses(r, threads, [&](std::size_t s, int) {
    return tree_fit(x, y + s * n, w + s * n, n, p, family, spec,
                    detail::response_fold(fold, n, s),
                    detail::response_fold_count(fold, n_fold, s));
  });
}

std::vector<Forest> forest_fits(const double* x, std::size_t n, std::size_t p, const double* y,
                                const double* w, std::size_t r, Family family,
                                const ForestSpec& spec, const std::uint32_t* seeds) {
  return detail::fit_responses(r, spec.threads, [&](std::size_t s, int inner) {
    ForestSpec each = spec;
    each.seed = seeds[s];
    each.threads = inner;
    return forest_fit(x, y + s * n, w + s * n, n, p, family, each);
  });
}

}  // namespace timesift
