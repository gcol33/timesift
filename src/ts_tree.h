#ifndef TIMESIFT_TS_TREE_H
#define TIMESIFT_TS_TREE_H

#include <cstddef>
#include <cstdint>
#include <vector>

#include "ts_penalised.h"

// The classification and regression tree, once, for both languages.
//
// Recursive partitioning (Breiman, Friedman, Olshen and Stone 1984): a node is cut at the midpoint
// between two distinct values of one column, the cut lowering most the Gini index of the weighted
// class counts under a binomial family, the weighted sum of squares under a Gaussian one or the
// Poisson deviance of the unshrunk rates under a Poisson one. A node
// is split only where it holds `min_split` observations and each child keeps `min_leaf`, and a
// split is kept only where cost-complexity pruning finds it lowers the risk by at least `cp` of
// the root's. The complexity table carries the cross-validated error of every pruned subtree, with
// the folds dealt by the caller rather than drawn here. On the same data this is the tree rpart
// grows, which the fixtures pin.
//
// The design is column-major, `x[i + j * n]`. A binomial response is 0 or 1.
namespace timesift {

struct TreeSpec {
  int min_split = 20;    // observations a node needs before a split is tried
  int min_leaf = 7;      // observations each child of a split keeps
  double cp = 0.01;      // a split is kept where it lowers the risk by at least this share of
                         // the root's
  int max_depth = 30;    // the root is depth 0
  double shrink = 1.0;   // a Poisson leaf's rate is shrunk towards the response's own rate, by the
                         // gamma prior of this coefficient of variation; 0 for no shrinkage
};

// A fitted tree, its nodes depth first, left before right. A node's number is 1 at the root and
// 2k, 2k + 1 below node k.
struct Tree {
  Family family = Family::gaussian;
  std::vector<std::int32_t> number;
  std::vector<std::int32_t> column;      // the column a node splits on, -1 at a leaf
  std::vector<double> threshold;
  std::vector<std::int8_t> less_left;    // 1 where a value below the threshold goes left
  std::vector<std::int32_t> left;        // the index of the child, -1 at a leaf
  std::vector<std::int32_t> right;
  std::vector<std::int32_t> n;           // observations in the node
  std::vector<double> weight;            // their summed case weight
  std::vector<double> risk;              // weighted misclassification, weighted sum of squares, or
                                         // Poisson deviance
  std::vector<double> complexity;        // on the scale of the root's risk
  std::vector<double> value;             // the probability of a 1, the mean, or the rate
  double root_risk = 0.0;

  // The complexity table: one row per distinct complexity, largest first, each on the scale of
  // the root's risk. `xerror` and `xstd` are empty where no folds were given.
  std::vector<double> cp;
  std::vector<std::int32_t> nsplit;
  std::vector<double> rel_error;
  std::vector<double> xerror;
  std::vector<double> xstd;
};

// Throws Error for an empty design, a non-finite value, a negative weight, and a binomial
// response other than 0 or 1, a negative count and a count response without one count above zero.
// `fold` is one 0-based fold index per observation, or null for no
// cross-validation.
Tree tree_fit(const double* x, const double* y, const double* w, std::size_t n, std::size_t p,
              Family family, const TreeSpec& spec, const std::int32_t* fold,
              std::int32_t n_fold);

// One tree per column of `y` and `w` [n, r], response `s` under column `s` of `fold` [n, r] and
// `n_fold[s]` folds, or under none where `fold` is null. `threads` grow that many trees at once,
// each the tree it is alone.
std::vector<Tree> tree_fits(const double* x, std::size_t n, std::size_t p, const double* y,
                            const double* w, std::size_t r, Family family, const TreeSpec& spec,
                            const std::int32_t* fold, const std::int32_t* n_fold, int threads);

// The tree with every split of complexity at or below `cp` collapsed. The complexity table is
// kept whole.
Tree tree_prune(const Tree& tree, double cp);

// The value of the leaf each row of `x` falls into. `x` is column-major over `p` columns.
void tree_predict(const Tree& tree, const double* x, std::size_t n, std::size_t p, double* out);

// The random forest (Breiman 2001). Each tree is grown on a bootstrap draw of the observations, the draw weighted
// by the case weights, and each node is split on the best of `mtry` columns drawn for it; the split
// search is the tree's, run with every drawn observation weighing one. Nothing is pruned: a node is
// split while it holds at least twice `min_leaf` observations, its responses are not all equal, and
// a split leaving `min_leaf` on each side improves on it.
//
// Every draw comes from the generator below, and tree `t` of a forest seeded `seed` starts its own
// generator from `(seed, t)`, so a forest is the same forest on any number of threads and in
// either language.
struct ForestSpec {
  int trees = 500;
  int mtry = 1;          // columns drawn for each node, 1 to the column count
  int min_leaf = 1;      // observations each child of a split keeps
  bool balance = false;  // each tree draws the smaller class's count from each class
  std::uint32_t seed = 1;
  int threads = 1;       // trees grown at once; 1 is serial
};

// Many trees' nodes one after another, tree `t` at `offset[t]` to `offset[t + 1]`. Within a tree
// the nodes are depth first, left before right, and `left` and `right` count from the tree's first
// node.
struct TreeTable {
  std::vector<std::int32_t> offset;
  std::vector<std::int32_t> column;      // -1 at a leaf
  std::vector<double> threshold;
  std::vector<std::int8_t> less_left;
  std::vector<std::int32_t> left;
  std::vector<std::int32_t> right;
  std::vector<double> value;
};

// A forest's leaves hold the share of ones among their draws, or their mean.
struct Forest {
  Family family = Family::gaussian;
  std::int32_t n_column = 0;
  TreeTable trees;
};

// Throws Error for an empty design, a non-finite value, a negative weight, a binomial response
// other than 0 or 1, a negative count, `mtry` outside 1 to `p`, and `balance` on a Gaussian or a
// Poisson family or on a response one class of which weighs nothing.
Forest forest_fit(const double* x, const double* y, const double* w, std::size_t n, std::size_t p,
                  Family family, const ForestSpec& spec);

// One forest per column of `y` and `w` [n, r], response `s` seeded `seeds[s]`. `spec.threads` grow
// that many forests at once, or a lone response's trees; a forest is the same either way.
std::vector<Forest> forest_fits(const double* x, std::size_t n, std::size_t p, const double* y,
                                const double* w, std::size_t r, Family family,
                                const ForestSpec& spec, const std::uint32_t* seeds);

// The mean over the trees of the leaf each row of `x` falls into.
void forest_predict(const Forest& forest, const double* x, std::size_t n, std::size_t p,
                    double* out);

// The first `n` outputs of the generator tree `tree` of a forest seeded `seed` draws from, so a
// reimplementation of the generator can be checked against this one output for output.
void forest_stream(std::uint32_t seed, std::uint32_t tree, std::size_t n, std::uint32_t* out);

// Gradient boosted trees. The score starts at the log-odds of the weighted share of ones, the
// weighted mean, or the log of the weighted mean count, and each tree is fitted to the loss's gradient at the current score and added to
// it scaled by `shrinkage`. Each tree is grown on a subsample of the observations drawn without
// replacement, and reads a subsample of the columns.
//
// With `newton` off the trees are gbm's: a tree is `depth` splits grown best first, each split
// the one of largest reduction in the weighted squared error of the working response, with at
// least `min_leaf` observations on each side, and a leaf takes one Newton step on the loss (the
// mean of the working response under a Gaussian family). With `newton` on they are xgboost's
// exact greedy trees: grown level by level to `depth`, a split chosen by the second-order gain
// under the L2 penalty `lambda`, at least `min_leaf` of hessian on each side, pruned where a split
// gains less than `gamma`, and a leaf the Newton step `-G / (H + lambda)`.
//
// Where folds are given, the fit is repeated on each fold's complement with the fold held out, and
// the number of trees kept is the one of least held-out deviance summed over the folds, each
// fold's weighted by how many observations it holds, as gbm's `cv.folds` chooses it. A count
// response is boosted under the Poisson deviance with the log link.
struct BoostSpec {
  int trees = 100;
  int depth = 1;
  double shrinkage = 0.1;
  double min_leaf = 10;
  double subsample = 0.5;
  double colsample = 1.0;
  bool newton = false;
  double lambda = 0.0;
  double gamma = 0.0;
  std::uint32_t seed = 1;
  int threads = 1;      // the fit on every observation and the folds' fits run at once
};

// The trees' leaves hold their step already scaled by the shrinkage, so a score is `init` plus the
// sum of the leaves a row falls into.
struct Boosted {
  Family family = Family::gaussian;
  std::int32_t n_column = 0;
  double init = 0.0;
  TreeTable trees;               // the trees kept, the chosen number where folds were given
  std::vector<double> cv_error;  // the held-out deviance after each tree, empty without folds
};

// Throws Error for an empty design, a non-finite value, a negative weight, a binomial response
// other than 0 or 1 or holding one class alone, a negative count, and settings outside their range.
Boosted boost_fit(const double* x, const double* y, const double* w, std::size_t n,
                  std::size_t p, Family family, const BoostSpec& spec, const std::int32_t* fold,
                  std::int32_t n_fold);

// One boosted fit per column of `y` and `w` [n, r], response `s` seeded `seeds[s]` and under column
// `s` of `fold` [n, r] and `n_fold[s]` folds, or under none where `fold` is null. `spec.threads` fit
// that many responses at once, or a lone response's fits on every unit and on each fold's
// complement; a fit is the same either way.
std::vector<Boosted> boost_fits(const double* x, std::size_t n, std::size_t p, const double* y,
                                const double* w, std::size_t r, Family family,
                                const BoostSpec& spec, const std::uint32_t* seeds,
                                const std::int32_t* fold, const std::int32_t* n_fold);

// The score of each row, through the logistic function under a binomial family and the exponential
// under a Poisson one.
void boost_predict(const Boosted& model, const double* x, std::size_t n, std::size_t p,
                   double* out);

}  // namespace timesift

#endif  // TIMESIFT_TS_TREE_H
