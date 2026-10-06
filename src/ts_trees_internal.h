#ifndef TIMESIFT_TS_TREES_INTERNAL_H
#define TIMESIFT_TS_TREES_INTERNAL_H

// What the forest and the boosted trees share and neither language sees: the generator every draw
// comes from, one tree's nodes while it is grown and the descent of a row to its leaf.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "ts_core.h"
#include "ts_internal.h"
#include "ts_tree.h"

namespace timesift {
namespace detail {

// The generator: xoshiro128** (Blackman and Vigna), its four words of state the outputs `4s + 1`
// to `4s + 4` of a SplitMix32 stream started at the seed, for stream `s`. Everything is arithmetic
// modulo 2^32, which a reimplementation in a language without unsigned integers can do exactly in
// doubles.
class Stream {
 public:
  Stream(std::uint32_t seed, std::uint32_t stream) {
    std::uint32_t counter = seed + 4u * stream * kGolden;
    for (std::uint32_t& word : s_) word = splitmix(counter);
  }

  std::uint32_t next() {
    const std::uint32_t result = rotl(s_[1] * 5u, 7) * 9u;
    const std::uint32_t t = s_[1] << 9;
    s_[2] ^= s_[0];
    s_[3] ^= s_[1];
    s_[1] ^= s_[2];
    s_[0] ^= s_[3];
    s_[2] ^= t;
    s_[3] = rotl(s_[3], 11);
    return result;
  }

  // A uniform on [0, 1): the output times 2^-32, exact.
  double uniform() { return static_cast<double>(next()) * 0x1p-32; }

  // An index on 0 to `k - 1`: the uniform times `k`, rounded down.
  std::size_t below(std::size_t k) {
    const std::size_t j = static_cast<std::size_t>(uniform() * static_cast<double>(k));
    return j < k ? j : k - 1;
  }

 private:
  static constexpr std::uint32_t kGolden = 0x9E3779B9u;

  static std::uint32_t rotl(std::uint32_t x, int k) { return (x << k) | (x >> (32 - k)); }

  static std::uint32_t splitmix(std::uint32_t& counter) {
    counter += kGolden;
    std::uint32_t z = counter;
    z = (z ^ (z >> 16)) * 0x85EBCA6Bu;
    z = (z ^ (z >> 13)) * 0xC2B2AE35u;
    return z ^ (z >> 16);
  }

  std::uint32_t s_[4];
};

// `k` of the indices 0 to `p - 1`, drawn by a partial Fisher-Yates shuffle of `perm` and returned
// in ascending order. `perm` holds an arrangement of the indices and keeps what the shuffle left.
inline void draw_columns(Stream& stream, std::vector<int>& perm, std::size_t k,
                         std::vector<int>& chosen) {
  const std::size_t p = perm.size();
  for (std::size_t c = 0; c < k; ++c) std::swap(perm[c], perm[c + stream.below(p - c)]);
  chosen.assign(perm.begin(), perm.begin() + static_cast<std::ptrdiff_t>(k));
  std::sort(chosen.begin(), chosen.end());
}

// One tree's nodes, depth first, its children counted from its own first node.
struct Nodes {
  std::vector<std::int32_t> column;
  std::vector<double> threshold;
  std::vector<std::int8_t> less_left;
  std::vector<std::int32_t> left;
  std::vector<std::int32_t> right;
  std::vector<double> value;

  std::int32_t add_leaf(double v) {
    column.push_back(-1);
    threshold.push_back(0.0);
    less_left.push_back(0);
    left.push_back(-1);
    right.push_back(-1);
    value.push_back(v);
    return static_cast<std::int32_t>(column.size() - 1);
  }
};

inline void append(TreeTable& table, const Nodes& nodes) {
  if (table.offset.empty()) table.offset.push_back(0);
  table.column.insert(table.column.end(), nodes.column.begin(), nodes.column.end());
  table.threshold.insert(table.threshold.end(), nodes.threshold.begin(), nodes.threshold.end());
  table.less_left.insert(table.less_left.end(), nodes.less_left.begin(), nodes.less_left.end());
  table.left.insert(table.left.end(), nodes.left.begin(), nodes.left.end());
  table.right.insert(table.right.end(), nodes.right.begin(), nodes.right.end());
  table.value.insert(table.value.end(), nodes.value.begin(), nodes.value.end());
  table.offset.push_back(static_cast<std::int32_t>(table.column.size()));
}

// The index of the leaf row `i` of the column-major `x` falls into, from the node table of one
// tree, its children counted from its first node.
inline std::size_t leaf_of(const std::int32_t* column, const double* threshold,
                           const std::int8_t* less_left, const std::int32_t* left,
                           const std::int32_t* right, const double* x, std::size_t i,
                           std::size_t n, std::size_t p) {
  std::size_t at = 0;
  while (column[at] >= 0) {
    const std::size_t c = static_cast<std::size_t>(column[at]);
    if (c >= p) throw Error("the design predicted on has fewer columns than the tree splits on.");
    const double v = x[i + c * n];
    if (!std::isfinite(v)) {
      throw Error("a tree predicts from finite values, and row " + std::to_string(i + 1) +
                  " holds one that is not in a column it splits on.");
    }
    const bool below = v < threshold[at];
    const bool go_left = below == (less_left[at] == 1);
    at = static_cast<std::size_t>(go_left ? left[at] : right[at]);
  }
  return at;
}

// The value of the leaf row `i` falls into in tree `t` of a table.
inline double table_value(const TreeTable& table, std::size_t t, const double* x, std::size_t i,
                          std::size_t n, std::size_t p) {
  const std::size_t o = static_cast<std::size_t>(table.offset[t]);
  return table.value[o + leaf_of(table.column.data() + o, table.threshold.data() + o,
                                 table.less_left.data() + o, table.left.data() + o,
                                 table.right.data() + o, x, i, n, p)];
}

inline std::size_t table_trees(const TreeTable& table) {
  return table.offset.empty() ? 0 : table.offset.size() - 1;
}

}  // namespace detail
}  // namespace timesift

#endif  // TIMESIFT_TS_TREES_INTERNAL_H
