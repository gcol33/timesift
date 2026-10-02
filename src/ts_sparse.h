#ifndef TIMESIFT_TS_SPARSE_H
#define TIMESIFT_TS_SPARSE_H

#include <cstddef>
#include <utility>
#include <vector>

// The Cholesky factorisation of a sparse symmetric positive definite matrix, once, for the fits
// that carry a latent field over many locations.
//
// The matrix is declared by its pattern, the set of `(row, column)` positions its lower triangle
// can hold, and ordered once by minimum degree, a greedy elimination order that keeps the factor
// sparse (George and Liu 1981, Computer Solution of Large Sparse Positive Definite Systems). The
// factor is then computed by the up-looking algorithm along the elimination tree, row by row, each
// row's pattern read off the tree (Davis 2006, Direct Methods for Sparse Linear Systems, ch. 4).
// The pattern is analysed once and any number of matrices over it are factored, which is what a
// grid of conditional fits over one set of locations needs.
namespace timesift {

class SparseSymmetric {
 public:
  // `pattern` holds positions `(i, j)` of the lower triangle, `i >= j`, of an `n` by `n` matrix;
  // the diagonal is added where it is missing, and a position given twice is kept once.
  SparseSymmetric(std::size_t n, std::vector<std::pair<std::size_t, std::size_t>> pattern);

  std::size_t size() const { return n_; }
  std::size_t nonzeros() const { return row_.size(); }

  // The slot of the lower triangle's position `(i, j)`, `i >= j`, which has to be in the pattern.
  std::size_t slot(std::size_t i, std::size_t j) const;

  // Values by slot, zeroed; the caller adds each entry in.
  std::vector<double> zeros() const { return std::vector<double>(row_.size(), 0.0); }

  // The matrix `values` (by slot) times the vector `x`.
  void multiply(const std::vector<double>& values, const double* x, double* out) const;

  const std::vector<std::size_t>& column_start() const { return start_; }
  const std::vector<std::size_t>& row_index() const { return row_; }

 private:
  std::size_t n_;
  std::vector<std::size_t> start_;  // [n + 1] column starts of the lower triangle
  std::vector<std::size_t> row_;    // rows of each column, ascending, the diagonal first
};

class SparseCholesky {
 public:
  explicit SparseCholesky(const SparseSymmetric& pattern);

  // Factors the matrix `values` (by slot of the pattern). False where a pivot is not positive.
  bool factor(const std::vector<double>& values);

  // Solves `A x = b` in place from the last factor.
  void solve(double* b) const;

  // `log |A|` of the last factor.
  double log_det() const;

  std::size_t factor_nonzeros() const { return lower_row_.size(); }

 private:
  const SparseSymmetric& pattern_;
  std::size_t n_;
  std::vector<std::size_t> perm_;      // perm_[k] = the original index eliminated k-th
  std::vector<std::size_t> inverse_;   // inverse_[original] = its position
  // The permuted matrix's upper triangle by column, with the slot each entry reads.
  std::vector<std::size_t> upper_start_, upper_row_, upper_slot_;
  std::vector<std::ptrdiff_t> parent_;  // the elimination tree
  std::vector<std::size_t> lower_start_;  // columns of the factor L
  std::vector<std::size_t> lower_row_;
  std::vector<double> lower_value_;
  std::vector<std::size_t> fill_;       // scratch: next free position of each column
  std::vector<double> dense_;           // scratch: the row being solved
  std::vector<std::size_t> stack_;      // scratch: a row's pattern
  std::vector<std::ptrdiff_t> mark_;
};

}  // namespace timesift

#endif  // TIMESIFT_TS_SPARSE_H
