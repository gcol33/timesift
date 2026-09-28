#ifndef TIMESIFT_TS_TREE_H
#define TIMESIFT_TS_TREE_H

#include <cstddef>
#include <cstdint>
#include <vector>

#include "ts_penalised.h"

// The classification and regression tree, once, for both languages.
//
// Recursive partitioning under rpart's rules, so a tree here is the tree rpart grows on the same
// data: the Gini index on weighted class counts for a binomial family and the weighted sum of
// squares for a Gaussian one, a split only between two distinct values of a column at their
// midpoint, `min_split` observations before a node is split and `min_leaf` in each child, and the
// cost-complexity bookkeeping that collapses a split whose share of the root's risk is below `cp`.
// The complexity table and its cross-validated error are rpart's too, with the folds dealt by the
// caller rather than drawn here.
//
// The design is column-major, `x[i + j * n]`. A binomial response is 0 or 1.
namespace timesift {

struct TreeSpec {
  int min_split = 20;    // observations a node needs before a split is tried
  int min_leaf = 7;      // observations each child of a split keeps
  double cp = 0.01;      // a split is kept where it lowers the risk by at least this share of
                         // the root's
  int max_depth = 30;    // the root is depth 0
};

// A fitted tree, its nodes in the order rpart's frame lists them: depth first, left before
// right. A node's number is rpart's, 1 at the root and 2k, 2k + 1 below node k.
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
  std::vector<double> risk;              // weighted misclassification, or weighted sum of squares
  std::vector<double> complexity;        // on the scale of the root's risk
  std::vector<double> value;             // the probability of a 1, or the mean
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
// response other than 0 or 1. `fold` is one 0-based fold index per observation, or null for no
// cross-validation.
Tree tree_fit(const double* x, const double* y, const double* w, std::size_t n, std::size_t p,
              Family family, const TreeSpec& spec, const std::int32_t* fold,
              std::int32_t n_fold);

// The tree with every split of complexity at or below `cp` collapsed, as rpart's `prune()`.
// The complexity table is kept whole.
Tree tree_prune(const Tree& tree, double cp);

// The value of the leaf each row of `x` falls into. `x` is column-major over `p` columns.
void tree_predict(const Tree& tree, const double* x, std::size_t n, std::size_t p, double* out);

}  // namespace timesift

#endif  // TIMESIFT_TS_TREE_H
