#include "ts_sparse.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>

namespace timesift {

SparseSymmetric::SparseSymmetric(std::size_t n,
                                 std::vector<std::pair<std::size_t, std::size_t>> pattern)
    : n_(n) {
  std::vector<std::vector<std::size_t>> column(n);
  for (std::size_t j = 0; j < n; ++j) column[j].push_back(j);
  for (const auto& e : pattern) {
    if (e.first >= n || e.second >= n || e.first < e.second)
      throw std::invalid_argument("a sparse pattern holds positions of the lower triangle.");
    column[e.second].push_back(e.first);
  }
  start_.assign(n + 1, 0);
  for (std::size_t j = 0; j < n; ++j) {
    std::sort(column[j].begin(), column[j].end());
    column[j].erase(std::unique(column[j].begin(), column[j].end()), column[j].end());
    start_[j + 1] = start_[j] + column[j].size();
    row_.insert(row_.end(), column[j].begin(), column[j].end());
  }
}

std::size_t SparseSymmetric::slot(std::size_t i, std::size_t j) const {
  const auto first = row_.begin() + static_cast<std::ptrdiff_t>(start_[j]);
  const auto last = row_.begin() + static_cast<std::ptrdiff_t>(start_[j + 1]);
  const auto at = std::lower_bound(first, last, i);
  if (at == last || *at != i) throw std::invalid_argument("a position is outside the sparse pattern.");
  return static_cast<std::size_t>(at - row_.begin());
}

void SparseSymmetric::multiply(const std::vector<double>& values, const double* x,
                               double* out) const {
  std::fill(out, out + n_, 0.0);
  for (std::size_t j = 0; j < n_; ++j) {
    for (std::size_t s = start_[j]; s < start_[j + 1]; ++s) {
      const std::size_t i = row_[s];
      out[i] += values[s] * x[j];
      if (i != j) out[j] += values[s] * x[i];
    }
  }
}

namespace {

// Minimum degree: the node with the fewest neighbours is eliminated, its neighbours become a clique.
std::vector<std::size_t> minimum_degree(const SparseSymmetric& a) {
  const std::size_t n = a.size();
  std::vector<std::set<std::size_t>> adjacent(n);
  const auto& start = a.column_start();
  const auto& row = a.row_index();
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t s = start[j]; s < start[j + 1]; ++s) {
      const std::size_t i = row[s];
      if (i == j) continue;
      adjacent[i].insert(j);
      adjacent[j].insert(i);
    }
  }
  std::set<std::pair<std::size_t, std::size_t>> queue;
  for (std::size_t v = 0; v < n; ++v) queue.insert({adjacent[v].size(), v});
  std::vector<std::size_t> order;
  order.reserve(n);
  while (!queue.empty()) {
    const std::size_t v = queue.begin()->second;
    queue.erase(queue.begin());
    order.push_back(v);
    const std::vector<std::size_t> around(adjacent[v].begin(), adjacent[v].end());
    for (const std::size_t u : around) {
      queue.erase({adjacent[u].size(), u});
      adjacent[u].erase(v);
    }
    for (const std::size_t u : around)
      for (const std::size_t t : around)
        if (t != u) adjacent[u].insert(t);
    for (const std::size_t u : around) queue.insert({adjacent[u].size(), u});
    adjacent[v].clear();
  }
  return order;
}

}  // namespace

SparseCholesky::SparseCholesky(const SparseSymmetric& pattern)
    : pattern_(pattern), n_(pattern.size()) {
  perm_ = minimum_degree(pattern);
  inverse_.assign(n_, 0);
  for (std::size_t k = 0; k < n_; ++k) inverse_[perm_[k]] = k;

  // The permuted matrix's upper triangle by column.
  const auto& start = pattern.column_start();
  const auto& row = pattern.row_index();
  std::vector<std::size_t> count(n_ + 1, 0);
  for (std::size_t j = 0; j < n_; ++j)
    for (std::size_t s = start[j]; s < start[j + 1]; ++s)
      ++count[std::max(inverse_[row[s]], inverse_[j]) + 1];
  upper_start_.assign(n_ + 1, 0);
  for (std::size_t k = 0; k < n_; ++k) upper_start_[k + 1] = upper_start_[k] + count[k + 1];
  upper_row_.assign(pattern.nonzeros(), 0);
  upper_slot_.assign(pattern.nonzeros(), 0);
  std::vector<std::size_t> next(upper_start_.begin(), upper_start_.end() - 1);
  for (std::size_t j = 0; j < n_; ++j) {
    for (std::size_t s = start[j]; s < start[j + 1]; ++s) {
      const std::size_t a = inverse_[row[s]], b = inverse_[j];
      const std::size_t col = std::max(a, b);
      const std::size_t at = next[col]++;
      upper_row_[at] = std::min(a, b);
      upper_slot_[at] = s;
    }
  }

  // The elimination tree.
  parent_.assign(n_, -1);
  std::vector<std::ptrdiff_t> ancestor(n_, -1);
  for (std::size_t k = 0; k < n_; ++k) {
    for (std::size_t p = upper_start_[k]; p < upper_start_[k + 1]; ++p) {
      std::ptrdiff_t i = static_cast<std::ptrdiff_t>(upper_row_[p]);
      while (i != -1 && i < static_cast<std::ptrdiff_t>(k)) {
        const std::ptrdiff_t following = ancestor[static_cast<std::size_t>(i)];
        ancestor[static_cast<std::size_t>(i)] = static_cast<std::ptrdiff_t>(k);
        if (following == -1) parent_[static_cast<std::size_t>(i)] = static_cast<std::ptrdiff_t>(k);
        i = following;
      }
    }
  }

  // The factor's column counts, from the pattern of each row.
  mark_.assign(n_, -1);
  stack_.assign(n_, 0);
  std::vector<std::size_t> counts(n_, 1);
  for (std::size_t k = 0; k < n_; ++k) {
    mark_[k] = static_cast<std::ptrdiff_t>(k);
    for (std::size_t p = upper_start_[k]; p < upper_start_[k + 1]; ++p) {
      std::ptrdiff_t i = static_cast<std::ptrdiff_t>(upper_row_[p]);
      if (i >= static_cast<std::ptrdiff_t>(k)) continue;
      for (; mark_[static_cast<std::size_t>(i)] != static_cast<std::ptrdiff_t>(k);
           i = parent_[static_cast<std::size_t>(i)]) {
        ++counts[static_cast<std::size_t>(i)];
        mark_[static_cast<std::size_t>(i)] = static_cast<std::ptrdiff_t>(k);
      }
    }
  }
  lower_start_.assign(n_ + 1, 0);
  for (std::size_t k = 0; k < n_; ++k) lower_start_[k + 1] = lower_start_[k] + counts[k];
  lower_row_.assign(lower_start_[n_], 0);
  lower_value_.assign(lower_start_[n_], 0.0);
  fill_.assign(n_, 0);
  dense_.assign(n_, 0.0);
}

bool SparseCholesky::factor(const std::vector<double>& values) {
  std::fill(mark_.begin(), mark_.end(), -1);
  for (std::size_t k = 0; k < n_; ++k) fill_[k] = lower_start_[k];
  std::fill(dense_.begin(), dense_.end(), 0.0);
  for (std::size_t k = 0; k < n_; ++k) {
    // The row's pattern, in the order the entries can be solved for.
    std::size_t top = n_;
    mark_[k] = static_cast<std::ptrdiff_t>(k);
    double d = 0.0;
    for (std::size_t p = upper_start_[k]; p < upper_start_[k + 1]; ++p) {
      const std::size_t r = upper_row_[p];
      if (r == k) {
        d = values[upper_slot_[p]];
        continue;
      }
      dense_[r] = values[upper_slot_[p]];
      std::size_t length = 0;
      std::ptrdiff_t i = static_cast<std::ptrdiff_t>(r);
      for (; mark_[static_cast<std::size_t>(i)] != static_cast<std::ptrdiff_t>(k);
           i = parent_[static_cast<std::size_t>(i)]) {
        stack_[length++] = static_cast<std::size_t>(i);
        mark_[static_cast<std::size_t>(i)] = static_cast<std::ptrdiff_t>(k);
      }
      while (length > 0) stack_[--top] = stack_[--length];
    }
    for (; top < n_; ++top) {
      const std::size_t i = stack_[top];
      const double lki = dense_[i] / lower_value_[lower_start_[i]];
      dense_[i] = 0.0;
      for (std::size_t p = lower_start_[i] + 1; p < fill_[i]; ++p)
        dense_[lower_row_[p]] -= lower_value_[p] * lki;
      d -= lki * lki;
      const std::size_t at = fill_[i]++;
      lower_row_[at] = k;
      lower_value_[at] = lki;
    }
    if (!(d > 0.0) || !std::isfinite(d)) return false;
    const std::size_t at = fill_[k]++;
    lower_row_[at] = k;
    lower_value_[at] = std::sqrt(d);
  }
  return true;
}

void SparseCholesky::solve(double* b) const {
  std::vector<double> y(n_);
  for (std::size_t k = 0; k < n_; ++k) y[k] = b[perm_[k]];
  for (std::size_t j = 0; j < n_; ++j) {
    y[j] /= lower_value_[lower_start_[j]];
    for (std::size_t p = lower_start_[j] + 1; p < lower_start_[j + 1]; ++p)
      y[lower_row_[p]] -= lower_value_[p] * y[j];
  }
  for (std::size_t j = n_; j-- > 0;) {
    for (std::size_t p = lower_start_[j] + 1; p < lower_start_[j + 1]; ++p)
      y[j] -= lower_value_[p] * y[lower_row_[p]];
    y[j] /= lower_value_[lower_start_[j]];
  }
  for (std::size_t k = 0; k < n_; ++k) b[perm_[k]] = y[k];
}

double SparseCholesky::log_det() const {
  double s = 0.0;
  for (std::size_t k = 0; k < n_; ++k) s += std::log(lower_value_[lower_start_[k]]);
  return 2.0 * s;
}

}  // namespace timesift
