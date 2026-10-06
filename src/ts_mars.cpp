#include "ts_mars.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "ts_core.h"
#include "ts_glm.h"
#include "ts_internal.h"

// A knot that ties another to the last place is broken by the order the sums below are taken in,
// and that order is the one that reproduces earth's terms in the fixtures. Contraction is off for
// the reason `ts_tree.cpp` gives.
#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

namespace timesift {

namespace {

// The forward pass stops once the generalised R-squared of the model falls below this.
constexpr double kGrsqFloor = -10.0;
// A column whose residual on the columns before it is below this fraction of its own norm is
// taken as a combination of them.
constexpr double kRankTol = 1e-8;
// A candidate linear term whose squared residual on the basis is at most this is not tried.
constexpr double kMinCandidateSs = 0.01;
// A change in the residual sum of squares or in the R-squared below this in magnitude is none.
constexpr double kNegligible = 1e-10;
constexpr int kMaxDegree = 100;
// The squared residual of a hinge, relative to its squared norm, below which a weighted fit takes
// it as a combination of the terms.
constexpr double kHingeDependent = 1e-10;
const double kInf = std::numeric_limits<double>::infinity();

double sq(double v) { return v * v; }

double snap_zero(double v) { return (v > -kNegligible && v < kNegligible) ? 0.0 : v; }

double mean_of(const double* v, std::size_t n) {
  double m = 0.0;
  const double nd = static_cast<double>(n);
  for (std::size_t i = 0; i < n; ++i) m += v[i] / nd;
  return m;
}

double dot(const double* a, const double* b, std::size_t n) {
  double s = 0.0;
  for (std::size_t i = 0; i < n; ++i) s += a[i] * b[i];
  return s;
}

// Friedman's generalised cross-validation during the forward pass (1991, eq. 30), each knot
// charged `penalty` on top of the term it adds; -1 charges nothing.
double forward_gcv(int n_terms, std::size_t n, double rss, double penalty) {
  double cost = 0.0;
  if (penalty != -1.0) {
    const double n_knots = (static_cast<double>(n_terms) - 1.0) / 2.0;
    cost = (n_terms + penalty * n_knots) / static_cast<double>(n);
  }
  return cost >= 1.0 ? kInf : rss / (static_cast<double>(n) * sq(1.0 - cost));
}

// Removes from `v` its projection on each usable column of the orthonormal `basis` [n, *] below
// `count`, one column after another (modified Gram-Schmidt). Where `cache` is given, a projection
// already in it is read rather than taken, and one taken is kept there.
void project_out(double* v, const double* basis, std::size_t n, int count, const char* usable,
                 double* cache) {
  for (int t = 0; t < count; ++t) {
    if (!usable[t]) continue;
    const double* q = basis + static_cast<std::size_t>(t) * n;
    double beta;
    if (cache && cache[t] != kInf) {
      beta = cache[t];
    } else {
      beta = dot(q, v, n);
      if (cache) cache[t] = beta;
    }
    for (std::size_t i = 0; i < n; ++i) v[i] -= beta * q[i];
  }
}

bool independent_of_basis(double residual_norm, double norm) {
  return residual_norm >= kRankTol * (norm == 0.0 ? 1.0 : norm) && residual_norm > 0.0;
}

// ---- the knot sweep --------------------------------------------------------------------------
//
// The hinge `b(x) max(0, x - t)` on a parent `b` changes with the knot `t` only through sums over
// the units above it, so sweeping the knots of a column from its top down keeps the hinge's
// covariance with every basis column, with the response and with itself up to date in one pass
// over the sorted units (Friedman 1991, section 3.9, eqs. 49-52). The reduction a knot buys is
// then the part of its covariance with the response the basis does not already explain, over the
// part of its norm the basis does not.

// The basis a hinge is swept against: each column centred, one unit's values across the columns
// stored together, and the response's projection on each. `extra` is the column the pair's linear
// term would add, when it is tried with the knot.
struct SweepBasis {
  const double* rows = nullptr;  // column t at unit i: rows[t + i * stride]
  std::size_t stride = 0;
  int count = 0;
  const double* proj = nullptr;
  const double* extra = nullptr;  // [n], or null
  double extra_proj = 0.0;
};

// The hinge's parent and column. `centre` is the direction the hinge is centred on, null for the
// constant one of the unweighted fit, whose squared norm is `divisor = n`, and the weighted
// intercept's unit direction, `divisor = 1`, otherwise. `positive_norm` tries a knot only where
// the hinge's centred norm is above zero.
struct Hinge {
  const double* x = nullptr;
  const int* order = nullptr;  // the units by ascending `x`
  const double* parent = nullptr;
  const double* centre = nullptr;
  double divisor = 1.0;
  bool positive_norm = true;
};

// Which knots of a column are tried: none below the `end`-th unit from the bottom, then one in
// every `every` of the units the parent is positive on, counting down from `first`, which also
// spares the top of the column.
struct Spans {
  int end = 1;
  int every = 1;
  int first = 1;
};

// At every knot the spans admit, `on_knot(i, cov_y, norm, norm_raw)` is handed the knot's position
// in `order`, the hinge's covariance with the residual, its squared residual norm on the basis and
// its squared centred norm.
template <typename OnKnot>
void sweep_knots(const Hinge& h, const SweepBasis& b, const double* yc, std::size_t n,
                 const Spans& spans, std::vector<double>& unit_sum, std::vector<double>& cov,
                 OnKnot on_knot) {
  const int own = b.count + (b.extra ? 1 : 0);
  cov.assign(static_cast<std::size_t>(own) + 1, 0.0);
  unit_sum.assign(static_cast<std::size_t>(own) + 1, 0.0);
  double resp_run = 0.0, resp_cov = 0.0;
  double s_b = 0.0, s_bb = 0.0, s_bx = 0.0, s_bbx = 0.0, lever = 0.0;
  int countdown = spans.first;
  for (int i = static_cast<int>(n) - 2; i >= spans.end; --i) {
    const int enter = h.order[i + 1];
    const double knot = h.x[h.order[i]];
    const double x_in = h.x[enter];
    const double b_in = h.parent[enter];
    const double bb_in = sq(b_in);
    const double step = x_in - knot;

    const double* row = b.rows + static_cast<std::size_t>(enter) * b.stride;
    for (int t = 0; t < b.count; ++t) unit_sum[t] += b_in * row[t];
    if (b.extra) unit_sum[b.count] += b_in * b.extra[enter];
    for (int t = 0; t < own; ++t) cov[t] += step * unit_sum[t];

    const double cb = h.centre ? h.centre[enter] * b_in : b_in;
    s_b += cb;
    s_bb += bb_in;
    s_bx += cb * x_in;
    s_bbx += bb_in * x_in;
    const double lever_before = lever;
    lever = s_bx - s_b * knot;
    cov[own] += step * (2.0 * s_bbx - s_bb * (knot + x_in)) +
                (sq(lever_before) - sq(lever)) / h.divisor;

    resp_run += yc[enter] * b_in;
    resp_cov += step * resp_run;

    if (b_in > 0.0 && (!h.positive_norm || cov[own] > 0.0) && --countdown == 0) {
      countdown = spans.every;
      double explained_y = 0.0, explained_norm = 0.0;
      for (int t = 0; t < b.count; ++t) explained_y += b.proj[t] * cov[t];
      if (b.extra) explained_y += b.extra_proj * cov[b.count];
      for (int t = 0; t < own; ++t) explained_norm += cov[t] * cov[t];
      on_knot(i, resp_cov - explained_y, cov[own] - explained_norm, cov[own]);
    }
  }
}

// ---- the weighted basis ----------------------------------------------------------------------

// An orthonormal basis of every weighted term column so far, by modified Gram-Schmidt, with the
// weighted response's residual on it. A column within `kRankTol` of the span of those before it
// holds no direction, as a pivoted QR would drop it. The first column is the weighted intercept,
// and the rest are also kept centred on it, one unit's values across the columns together, for
// the knot sweep.
class WeightedBasis {
 public:
  void reset(const double* yw, std::size_t n, int n_max) {
    yw_ = yw;
    n_ = n;
    n_max_ = static_cast<std::size_t>(n_max);
    q_.assign(n * n_max_, 0.0);
    centred_.assign(n_max_ * n, 0.0);
    proj_.assign(n_max_, 0.0);
    usable_.assign(n_max_, 0);
  }

  void append(int t, const double* v) {
    const std::size_t ts = static_cast<std::size_t>(t);
    double* q = q_.data() + ts * n_;
    std::memcpy(q, v, n_ * sizeof(double));
    project_out(q, q_.data(), n_, t, usable_.data(), nullptr);
    const double norm_v = std::sqrt(dot(v, v, n_));
    const double norm_r = std::sqrt(dot(q, q, n_));
    usable_[ts] = independent_of_basis(norm_r, norm_v);
    if (!usable_[ts]) {
      std::fill(q, q + n_, 0.0);
      for (std::size_t i = 0; i < n_; ++i) centred_[ts + i * n_max_] = 0.0;
      proj_[ts] = 0.0;
      return;
    }
    for (std::size_t i = 0; i < n_; ++i) q[i] /= norm_r;
    if (t == 0) {
      intercept_.assign(q, q + n_);
      const double on = dot(yw_, q, n_);
      centred_y_.resize(n_);
      for (std::size_t i = 0; i < n_; ++i) centred_y_[i] = yw_[i] - on * q[i];
      residual_ = centred_y_;
    } else {
      const double on_intercept = dot(q, intercept_.data(), n_);
      for (std::size_t i = 0; i < n_; ++i) {
        centred_[ts + i * n_max_] = q[i] - on_intercept * intercept_[i];
      }
      const double on = dot(residual_.data(), q, n_);
      for (std::size_t i = 0; i < n_; ++i) residual_[i] -= on * q[i];
    }
    proj_[ts] = dot(centred_y_.data(), q, n_);
    rss_ = dot(residual_.data(), residual_.data(), n_);
  }

  // `v`'s residual on the first `count` columns into `out`, normalised, and true; false where `v`
  // is a combination of them.
  bool orthonormalise(const double* v, double* out, int count, double* cache) const {
    std::memcpy(out, v, n_ * sizeof(double));
    project_out(out, q_.data(), n_, count, usable_.data(), cache);
    const double norm_v = std::sqrt(dot(v, v, n_));
    const double norm_r = std::sqrt(dot(out, out, n_));
    if (!independent_of_basis(norm_r, norm_v)) return false;
    for (std::size_t i = 0; i < n_; ++i) out[i] /= norm_r;
    return true;
  }

  const double* centred_rows() const { return centred_.data(); }
  const double* proj() const { return proj_.data(); }
  const double* intercept() const { return intercept_.data(); }
  const double* centred_y() const { return centred_y_.data(); }
  const double* residual() const { return residual_.data(); }
  double rss() const { return rss_; }

 private:
  const double* yw_ = nullptr;
  std::size_t n_ = 0, n_max_ = 0;
  std::vector<double> q_, centred_, proj_;
  std::vector<char> usable_;
  std::vector<double> intercept_;  // the unit direction of the weighted intercept
  std::vector<double> centred_y_;  // the weighted response less its part on the intercept
  std::vector<double> residual_;   // and less its part on every column
  double rss_ = 0.0;
};

// ---- fast MARS's ranking of the parents ------------------------------------------------------

// Friedman (1993): every term that may take a factor is held with the largest reduction it offered
// when last searched and the step it was searched at. The terms are ranked by that reduction, and,
// with `beta` above zero, the rank is aged by `beta` for every step since, so a term searched long
// ago comes back to the front. Only the first `fast_k` of the ranking are searched at a step.
class ParentRanking {
 public:
  void reset(int n_max, double beta) {
    beta_ = beta;
    slots_.assign(static_cast<std::size_t>(n_max), Slot());
    for (int t = 0; t < n_max; ++t) slots_[static_cast<std::size_t>(t)].term = t;
    size_ = 0;
  }

  // Opens the next slot, at step `n_terms`, with the larger of `gain` and the one slot `from`
  // holds.
  void open(int from, int n_terms, double gain, bool rerank) {
    Slot& s = slots_[static_cast<std::size_t>(size_)];
    s.searched_at = n_terms;
    s.gain = std::max(slots_[static_cast<std::size_t>(from)].gain, gain);
    ++size_;
    if (rerank) rank(n_terms);
  }

  void searched(int term, int n_terms, double gain) {
    Slot& s = slots_[static_cast<std::size_t>(term)];
    s.searched_at = n_terms;
    s.gain = gain;
  }

  int size() const { return size_; }
  int at(int k) const { return ranked_[static_cast<std::size_t>(k)].term; }

 private:
  struct Slot {
    int term = 0;
    double gain = -1.0;
    int searched_at = -99;
    double age = -1.0;
  };

  static bool larger_gain(const Slot& a, const Slot& b) {
    if (a.gain > b.gain) return true;
    if (b.gain > a.gain) return false;
    return a.term < b.term;
  }

  static bool younger(const Slot& a, const Slot& b) {
    if (a.age < b.age) return true;
    if (b.age < a.age) return false;
    return larger_gain(a, b);
  }

  void rank(int n_terms) {
    ranked_.assign(slots_.begin(), slots_.begin() + size_);
    std::sort(ranked_.begin(), ranked_.end(), larger_gain);
    if (beta_ > 0.0) {
      for (int r = 0; r < size_; ++r) {
        Slot& s = ranked_[static_cast<std::size_t>(r)];
        s.age = r + beta_ * (n_terms - s.searched_at);
      }
      std::sort(ranked_.begin(), ranked_.end(), younger);
    }
  }

  double beta_ = 1.0;
  std::vector<Slot> slots_, ranked_;
  int size_ = 0;
};

// ---- the forward pass ------------------------------------------------------------------------

// A term: the product of one factor per column it uses, each `max(0, x - cut)` (1),
// `max(0, cut - x)` (-1) or `x` (2); 0 where the column is not used.
struct Term {
  int degree = 0;
  std::vector<int> dir;
  std::vector<double> cut;
};

// A step the pass may take: the parent, the column, the knot's position in the column's order,
// and the fall in the residual sum of squares. `linear` enters the column linearly; `pair_ok`
// lets the upper hinge in with the lower.
struct Step {
  int knot = -1;
  int column = -1;
  int parent = -1;
  double gain = 0.0;
  bool linear = false;
  bool pair_ok = false;
};

// What one column offers under one parent, in the order they are weighed.
struct Offers {
  int count = 0;
  Step step[2];
  void add(const Step& s) { step[count++] = s; }
};

struct OrthColumn {
  bool usable = true;
  double mean = 0.0;
};

class ForwardPass {
 public:
  ForwardPass(const double* x, const double* y, const double* yw, const double* w, std::size_t n,
              std::size_t p, const MarsSpec& spec, int n_max, double penalty)
      : x_(x), y_(y), yw_(yw), w_(w), n_(n), p_(static_cast<int>(p)), spec_(spec),
        n_max_(n_max), penalty_(penalty), weighted_(yw != nullptr) {}

  void run();

  std::vector<Term> terms;     // `n_max` slots
  std::vector<char> live;      // on the slots whose term is in the model
  std::vector<double> values;  // [n, n_max]: term t at unit i is values[i + t * n]
  int termcond = 0;

 private:
  const double* column(int j) const { return x_ + static_cast<std::size_t>(j) * n_; }
  const int* order(int j) const { return order_.data() + static_cast<std::size_t>(j) * n_; }
  double* values_of(int t) { return values.data() + static_cast<std::size_t>(t) * n_; }
  double* orth(int t) { return orth_.data() + static_cast<std::size_t>(t) * n_; }
  void centre_row(int t, const double* col, double mean);
  std::vector<double>& cache_for(std::vector<std::vector<double>>& caches, int j, int parent);

  int end_span(int degree) const;
  Spans spans_for(int degree, int parent) const;
  bool linear_is_new(int parent, int j, int n_terms) const;
  OrthColumn orthonormal_column(int t, const double* v, bool candidate, double* col,
                                double* cache);
  Offers offers_plain(int parent, int j, int n_terms, double max_legal, const Spans& spans);
  Offers offers_weighted(int parent, int j, int n_terms, double rss, const Spans& spans);
  Step choose_step(int n_terms, double rss, double max_legal);
  void append(int n_terms, const Step& step, bool& pair_ok);

  const double* x_;
  const double* y_;   // the scaled response
  const double* yw_;  // the scaled response times the root weights, or null
  const double* w_;   // the weights, all one when the fit is unweighted
  std::size_t n_;
  int p_;
  MarsSpec spec_;
  int n_max_;
  double penalty_;
  bool weighted_;

  std::vector<int> order_;  // [n, p]: each column's units by ascending value
  // The unweighted fit's orthonormal basis of the live terms, [n, n_max], and the same centred
  // with one unit's values across the terms together, [n_max, n].
  std::vector<double> orth_, centred_;
  std::vector<double> yc_;    // the scaled response less its mean
  std::vector<double> proj_;  // its projection on each column of `orth_`, per step
  // A candidate linear term's projections on the basis, per column and parent: the basis only
  // grows, so a projection taken at one step holds at every later one.
  std::vector<std::vector<double>> plain_cache_, weighted_cache_;
  WeightedBasis wb_;
  ParentRanking parents_;
};

void ForwardPass::centre_row(int t, const double* col, double mean) {
  const std::size_t nm = static_cast<std::size_t>(n_max_);
  double* row0 = centred_.data() + static_cast<std::size_t>(t);
  for (std::size_t i = 0; i < n_; ++i) row0[i * nm] = col[i] - mean;
}

std::vector<double>& ForwardPass::cache_for(std::vector<std::vector<double>>& caches, int j,
                                            int parent) {
  std::vector<double>& c =
      caches[static_cast<std::size_t>(j) * static_cast<std::size_t>(n_max_) +
             static_cast<std::size_t>(parent)];
  if (c.empty()) c.assign(static_cast<std::size_t>(n_max_), kInf);
  return c;
}

// Friedman's end span (1991, eq. 45) at a 5% chance of a run of that many one-signed errors:
// 3 + log2(20 p), widened for an interaction.
int ForwardPass::end_span(int degree) const {
  int span = 1;
  if (spec_.endspan > 0) {
    span = spec_.endspan;
  } else {
    static const double log_2 = 0.69315;
    static const double three_log2_20 = 7.32193;
    span = static_cast<int>(three_log2_20 + std::log(static_cast<double>(p_)) / log_2);
  }
  if (degree >= 2) span += static_cast<int>(spec_.adjust_endspan * span + 0.5);
  const int half = static_cast<int>(n_) / 2 - 1;
  if (span > half) span = half;
  return std::max(1, span);
}

// Friedman's minimum span (1991, eq. 43) over the units the parent is positive on, again at 5%:
// (-log(-log(0.95) / (N p)) / log 2) / 2.5, and the offset that centres the tried knots between
// the two end spans.
Spans ForwardPass::spans_for(int degree, int parent) const {
  Spans s;
  s.end = end_span(degree);
  if (spec_.minspan < 0) {
    s.every = static_cast<int>(std::ceil(static_cast<double>(n_) / (1.0 - spec_.minspan)));
    s.first = s.every;
    while (s.first < s.end) s.first += s.every;
    s.first = std::max(1, s.first - 1);
    return s;
  }
  int every = 0;
  if (spec_.minspan > 0) {
    every = spec_.minspan;
  } else {
    int positive = 0;
    const double* bp = values.data() + static_cast<std::size_t>(parent) * n_;
    for (std::size_t i = 0; i < n_; ++i) {
      if (bp[i] > 0.0) ++positive;
    }
    static const double log_ratio = 2.9702;
    static const double scale = 1.7329;
    every = static_cast<int>((log_ratio + std::log(static_cast<double>(p_ * positive))) / scale);
  }
  s.every = std::max(1, every);
  const int inner = std::max(0, static_cast<int>(n_) - 2 * s.end);
  int offset = inner / 2;
  if (inner > s.every) {
    const int whole = inner / s.every;
    offset = inner == whole * s.every ? s.every / 2 : (inner - whole * s.every) / 2;
  }
  s.first = std::max(1, s.end + offset);
  return s;
}

// Whether column `j` may enter linearly under `parent`: the model holds no term but the
// intercept, or a live term leaves `j` out, or one uses other columns than the parent does.
bool ForwardPass::linear_is_new(int parent, int j, int n_terms) const {
  const std::vector<int>& pdir = terms[static_cast<std::size_t>(parent)].dir;
  bool any = false;
  for (int t = 1; t < n_terms; ++t) {
    if (!live[static_cast<std::size_t>(t)]) continue;
    const Term& term = terms[static_cast<std::size_t>(t)];
    any = true;
    if (term.dir[static_cast<std::size_t>(j)] == 0) return true;
    for (int c = 0; c < p_; ++c) {
      const std::size_t cs = static_cast<std::size_t>(c);
      if (c != j && (term.dir[cs] != 0) != (pdir[cs] != 0)) return true;
    }
  }
  return !any;
}

// Column `t` of the orthonormal basis from the term values `v`, into `col`: the constant at
// `t = 0`, `v` centred at `t = 1`, and otherwise `v`'s residual on the live columns below `t`,
// normalised. A candidate's column whose squared norm is at most `kMinCandidateSs` is zeroed and
// unusable; a term's is kept down to zero. `mean` is the column's mean before normalising.
OrthColumn ForwardPass::orthonormal_column(int t, const double* v, bool candidate, double* col,
                                           double* cache) {
  OrthColumn out;
  if (t == 0) {
    const double len = 1.0 / std::sqrt(static_cast<double>(n_));
    for (std::size_t i = 0; i < n_; ++i) col[i] = len;
    out.mean = len;
    return out;
  }
  if (t == 1) {
    const double m = mean_of(v, n_);
    for (std::size_t i = 0; i < n_; ++i) col[i] = v[i] - m;
  } else {
    std::memcpy(col, v, n_ * sizeof(double));
    project_out(col, orth_.data(), n_, t, live.data(), cache);
  }
  const double ss = dot(col, col, n_);
  if (ss <= kMinCandidateSs) out.usable = false;
  if (ss > (candidate ? kMinCandidateSs : 0.0)) {
    out.mean = mean_of(col, n_);
    const double len = std::sqrt(ss);
    for (std::size_t i = 0; i < n_; ++i) col[i] /= len;
  } else {
    out.mean = 0.0;
    std::fill(col, col + n_, 0.0);
  }
  return out;
}

// Unweighted: the column entering linearly, when it is new, and the pair of it and the best
// hinge, each scored by the squared projection of the response on what it adds to the basis.
Offers ForwardPass::offers_plain(int parent, int j, int n_terms, double max_legal,
                                 const Spans& spans) {
  Offers out;
  const bool tried = linear_is_new(parent, j, n_terms);
  bool pair_ok = tried;
  const double* xj = column(j);
  const double* bp = values.data() + static_cast<std::size_t>(parent) * n_;
  std::vector<double> cand_centred;
  double cand_proj = 0.0, linear = 0.0;
  if (tried) {
    std::vector<double> v(n_), cand(n_, 0.0);
    for (std::size_t i = 0; i < n_; ++i) v[i] = xj[i] * bp[i];
    std::vector<double>& cache = cache_for(plain_cache_, j, parent);
    const OrthColumn c = orthonormal_column(n_terms, v.data(), true, cand.data(), cache.data());
    if (!c.usable) pair_ok = false;
    cand_centred.resize(n_);
    for (std::size_t i = 0; i < n_; ++i) cand_centred[i] = cand[i] - c.mean;
    for (std::size_t i = 0; i < n_; ++i) cand_proj += yc_[i] * cand[i];
    double on_y = 0.0;
    for (std::size_t i = 0; i < n_; ++i) on_y += y_[i] * cand[i];
    linear = sq(on_y);
    Step s;
    s.gain = linear;
    s.knot = 0;
    s.linear = true;
    out.add(s);
  }

  SweepBasis b;
  b.rows = centred_.data();
  b.stride = static_cast<std::size_t>(n_max_);
  b.count = n_terms;
  b.proj = proj_.data();
  if (pair_ok) {
    b.extra = cand_centred.data();
    b.extra_proj = cand_proj;
  }
  Hinge h;
  h.x = xj;
  h.order = order(j);
  h.parent = bp;
  h.divisor = static_cast<double>(n_);
  h.positive_norm = true;
  const double tol = (pair_ok ? n_terms + 1 : n_terms) < 15 ? 0.01 : 1e-5;

  Step pair;
  pair.gain = linear;
  pair.pair_ok = pair_ok;
  std::vector<double> unit_sum, cov;
  sweep_knots(h, b, yc_.data(), n_, spans, unit_sum, cov,
              [&](int i, double cov_y, double norm, double norm_raw) {
                const double gain = linear + (norm / norm_raw > tol ? sq(cov_y) / norm : 0.0);
                if (gain > pair.gain && gain < max_legal) {
                  pair.knot = i;
                  pair.gain = gain;
                }
              });
  out.add(pair);
  return out;
}

// Weighted: the step's fall in the weighted residual sum of squares against the tracked one,
// taken with the column's linear term when it is new and independent of the basis, and at the
// best hinge where one beats that by more than `kNegligible`.
Offers ForwardPass::offers_weighted(int parent, int j, int n_terms, double rss,
                                    const Spans& spans) {
  Offers out;
  const bool is_new = linear_is_new(parent, j, n_terms);
  const double* xj = column(j);
  const double* bp = values.data() + static_cast<std::size_t>(parent) * n_;
  std::vector<double> cand_centred;
  double cand_proj = 0.0;
  bool has_cand = false;
  double without_knot = rss;       // the residuals the knot is measured against
  double knot_base = wb_.rss();    // and those its reduction is taken from
  if (is_new) {
    std::vector<double> v(n_), res(n_);
    for (std::size_t i = 0; i < n_; ++i) v[i] = bp[i] * xj[i];
    std::vector<double>& cache = cache_for(weighted_cache_, j, parent);
    if (wb_.orthonormalise(v.data(), res.data(), n_terms, cache.data())) {
      has_cand = true;
      const double* q0 = wb_.intercept();
      const double on_intercept = dot(res.data(), q0, n_);
      cand_centred.resize(n_);
      for (std::size_t i = 0; i < n_; ++i) cand_centred[i] = res[i] - on_intercept * q0[i];
      cand_proj = dot(wb_.centred_y(), res.data(), n_);
      const double linear_rss = wb_.rss() - sq(dot(wb_.residual(), res.data(), n_));
      without_knot = linear_rss;
      knot_base = linear_rss;
    } else {
      without_knot = wb_.rss();
    }
  }

  SweepBasis b;
  b.rows = wb_.centred_rows();
  b.stride = static_cast<std::size_t>(n_max_);
  b.count = n_terms;
  b.proj = wb_.proj();
  std::vector<double> zero;
  if (is_new) {
    if (!has_cand) zero.assign(n_, 0.0);
    b.extra = has_cand ? cand_centred.data() : zero.data();
    b.extra_proj = cand_proj;
  }
  Hinge h;
  h.x = xj;
  h.order = order(j);
  h.parent = bp;
  h.centre = wb_.intercept();
  h.divisor = 1.0;
  h.positive_norm = false;

  double best_rss = without_knot;
  int best_knot = 0;
  std::vector<double> unit_sum, cov;
  sweep_knots(h, b, wb_.centred_y(), n_, spans, unit_sum, cov,
              [&](int i, double cov_y, double norm, double norm_raw) {
                double fall = 0.0;
                if (norm > kHingeDependent * norm_raw && norm > 0.0) fall = sq(cov_y) / norm;
                const double knot_rss = knot_base - fall;
                if (knot_rss < best_rss - kNegligible) {
                  best_knot = i;
                  best_rss = knot_rss;
                }
              });
  Step s;
  s.linear = without_knot <= best_rss;
  s.knot = best_knot;
  s.gain = rss - (s.linear ? without_knot : best_rss);
  s.pair_ok = is_new;
  out.add(s);
  return out;
}

Step ForwardPass::choose_step(int n_terms, double rss, double max_legal) {
  Step best;
  if (!weighted_) {
    proj_.assign(static_cast<std::size_t>(n_max_), 0.0);
    for (int t = 0; t < n_terms; ++t) {
      proj_[static_cast<std::size_t>(t)] = dot(yc_.data(), orth(t), n_);
    }
  }
  const int searched = std::min(parents_.size(), spec_.fast_k);
  std::vector<Offers> offers(static_cast<std::size_t>(p_));
  for (int k = 0; k < searched; ++k) {
    const int parent = parents_.at(k);
    const Term& pterm = terms[static_cast<std::size_t>(parent)];
    if (pterm.degree >= spec_.degree) continue;
    const Spans spans = spans_for(pterm.degree + 1, parent);
    std::fill(offers.begin(), offers.end(), Offers());
    detail::run_tasks(static_cast<std::size_t>(p_), spec_.threads, [&](std::size_t js) {
      const int j = static_cast<int>(js);
      if (pterm.dir[js] != 0) return;
      offers[js] = weighted_ ? offers_weighted(parent, j, n_terms, rss, spans)
                             : offers_plain(parent, j, n_terms, max_legal, spans);
    });
    double parent_best = -1.0;
    for (int j = 0; j < p_; ++j) {
      const Offers& o = offers[static_cast<std::size_t>(j)];
      for (int c = 0; c < o.count; ++c) {
        const Step& s = o.step[c];
        if (s.gain > parent_best) parent_best = s.gain;
        if (s.gain > best.gain) {
          best = s;
          best.column = j;
          best.parent = parent;
        }
      }
    }
    parents_.searched(parent, n_terms, parent_best);
  }
  return best;
}

// Adds the step's terms at `n_terms` and `n_terms + 1`: the lower hinge, or the column itself
// when the step is linear, and the upper hinge when `pair_ok` lets it in and it is not a
// combination of the terms.
void ForwardPass::append(int n_terms, const Step& step, bool& pair_ok) {
  const int lo_t = n_terms, hi_t = n_terms + 1;
  Term& lo = terms[static_cast<std::size_t>(lo_t)];
  Term& hi = terms[static_cast<std::size_t>(hi_t)];
  const Term& parent = terms[static_cast<std::size_t>(step.parent)];
  const int j = step.column;
  const std::size_t js = static_cast<std::size_t>(j);
  lo.dir = parent.dir;
  hi.dir = parent.dir;
  lo.cut = parent.cut;
  hi.cut = parent.cut;
  lo.degree = hi.degree = parent.degree + 1;
  lo.dir[js] = step.linear ? 2 : 1;
  hi.dir[js] = -1;
  const double* xj = column(j);
  const int* oj = order(j);
  const double cut = xj[oj[step.knot]];
  lo.cut[js] = hi.cut[js] = cut;

  double* lv = values_of(lo_t);
  double* hv = values_of(hi_t);
  std::fill(lv, lv + n_, 0.0);
  std::fill(hv, hv + n_, 0.0);
  const double* bp = values_of(step.parent);
  if (step.linear) {
    for (std::size_t i = 0; i < n_; ++i) lv[i] = bp[i] * xj[i];
  } else {
    for (int r = 0; r < static_cast<int>(n_); ++r) {
      const std::size_t u = static_cast<std::size_t>(oj[r]);
      if (r > step.knot) {
        lv[u] = bp[u] * (xj[u] - cut);
      } else {
        hv[u] = bp[u] * (cut - xj[u]);
      }
    }
  }

  live[static_cast<std::size_t>(lo_t)] = 1;
  OrthColumn c = orthonormal_column(lo_t, lv, false, orth(lo_t), nullptr);
  centre_row(lo_t, orth(lo_t), c.mean);
  char& hi_live = live[static_cast<std::size_t>(hi_t)];
  if (!step.linear && pair_ok) hi_live = 1;
  if (hi_live) {
    c = orthonormal_column(hi_t, hv, false, orth(hi_t), nullptr);
    centre_row(hi_t, orth(hi_t), c.mean);
    if (weighted_ && !c.usable) {
      pair_ok = false;
      hi_live = 0;
    }
  }
  if (!hi_live) {
    hi.degree = kMaxDegree + 1;
    std::fill(orth(hi_t), orth(hi_t) + n_, 0.0);
    centre_row(hi_t, orth(hi_t), 0.0);
  }
  if (weighted_) {
    wb_.append(lo_t, lv);
    wb_.append(hi_t, hv);
  }
}

void ForwardPass::run() {
  const std::size_t nm = static_cast<std::size_t>(n_max_);
  Term blank;
  blank.dir.assign(static_cast<std::size_t>(p_), 0);
  blank.cut.assign(static_cast<std::size_t>(p_), 0.0);
  terms.assign(nm, blank);
  live.assign(nm, 0);
  values.assign(n_ * nm, 0.0);
  orth_.assign(n_ * nm, 0.0);
  centred_.assign(nm * n_, 0.0);
  plain_cache_.assign(nm * static_cast<std::size_t>(p_), std::vector<double>());

  order_.resize(n_ * static_cast<std::size_t>(p_));
  for (int j = 0; j < p_; ++j) {
    int* o = order_.data() + static_cast<std::size_t>(j) * n_;
    std::iota(o, o + n_, 0);
    const double* xj = column(j);
    std::stable_sort(o, o + n_, [xj](int a, int b) { return xj[a] < xj[b]; });
  }

  double* intercept = values_of(0);
  for (std::size_t i = 0; i < n_; ++i) intercept[i] = std::sqrt(w_[i]);
  const OrthColumn c0 = orthonormal_column(0, intercept, false, orth(0), nullptr);
  centre_row(0, orth(0), c0.mean);
  live[0] = 1;

  const double y_mean = mean_of(y_, n_);
  yc_.resize(n_);
  for (std::size_t i = 0; i < n_; ++i) yc_[i] = y_[i] - y_mean;
  double sum_wy = 0.0, sum_w = 0.0;
  for (std::size_t i = 0; i < n_; ++i) {
    sum_wy += w_[i] * y_[i];
    sum_w += w_[i];
  }
  const double wmean = sum_wy / sum_w;
  double rss_null = 0.0;
  for (std::size_t i = 0; i < n_; ++i) rss_null += w_[i] * sq(y_[i] - wmean);
  if (rss_null < 1e-8 * static_cast<double>(n_)) rss_null = 1e-8 * static_cast<double>(n_);

  if (weighted_) {
    wb_.reset(yw_, n_, n_max_);
    weighted_cache_.assign(nm * static_cast<std::size_t>(p_), std::vector<double>());
    wb_.append(0, intercept);
  }

  double rss = rss_null, last_gain = rss_null, rsq = 0.0, rsq_gain = 0.0, gcv = 0.0;
  int n_used = 1;
  const double gcv_null = forward_gcv(n_used, n_, rss_null, penalty_);
  parents_.reset(n_max_, spec_.fast_beta);
  parents_.open(0, 1, rss_null, true);
  int n_terms = 1;
  Step step;
  if (n_max_ >= 3) {
    for (;;) {
      if (rss <= 0.0) {
        throw Error("the forward pass reached a residual sum of squares of zero; the response is "
                    "fitted exactly by the terms so far, so lower `thresh` no further than it "
                    "stops there.");
      }
      const double max_legal = std::min(1.01 * rss, 10.0 * last_gain);
      step = choose_step(n_terms, rss, max_legal);
      last_gain = step.gain;
      bool pair_ok = step.pair_ok;
      if (step.knot >= 0) append(n_terms, step, pair_ok);
      n_used += (step.knot > 0 && pair_ok) ? 2 : 1;
      rss = snap_zero(rss - last_gain);
      gcv = forward_gcv(n_used, n_, rss, penalty_);
      const double rsq_before = rsq;
      rsq = 1.0 - rss / rss_null;
      rsq_gain = snap_zero(rsq - rsq_before);
      const bool stop = step.knot < 0 || (spec_.thresh != 0.0 && rsq_gain < spec_.thresh) ||
                        (spec_.thresh != 0.0 && 1.0 - gcv / gcv_null < kGrsqFloor);
      if (stop) {
        live[static_cast<std::size_t>(n_terms)] = 0;
        live[static_cast<std::size_t>(n_terms) + 1] = 0;
        break;
      }
      if (!step.linear && pair_ok) {
        parents_.open(n_terms, n_terms, kInf, false);
        parents_.open(n_terms + 1, n_terms, kInf, true);
      } else {
        parents_.open(n_terms, n_terms, kInf, true);
      }
      n_terms += 2;
      if (rsq >= 1.0 - spec_.thresh) break;
      if (n_terms >= n_max_ - 1) break;
    }
  }
  const double grsq = 1.0 - gcv / gcv_null;
  if (n_max_ < 3) {
    termcond = 1;
  } else if (spec_.thresh != 0.0 && grsq < kGrsqFloor) {
    termcond = grsq < -1000.0 ? 2 : 3;
  } else if (spec_.thresh != 0.0 && rsq_gain < spec_.thresh) {
    termcond = 4;
  } else if (rsq >= 1.0 - spec_.thresh) {
    termcond = 5;
  } else if (step.knot < 0) {
    termcond = 6;
  } else {
    termcond = 7;
  }
}

// ---- the pruning pass ------------------------------------------------------------------------
//
// The least-squares fit on `k` columns held as an orthogonal reduction `X = Q D^{1/2} R`, with
// `R` unit upper triangular, `d` the diagonal of `D` and `theta` the rotated response, built one
// row at a time by square-root-free plane rotations (Gentleman 1973) and reordered by rotating
// adjacent columns past each other (Miller 1992, Algorithm AS 274). The residual sum of squares
// of the first `s` columns is read off as the residual plus `d_i theta_i^2` for every column past
// them.

class Reduction {
 public:
  explicit Reduction(int k)
      : k_(k), d_(static_cast<std::size_t>(k), 0.0),
        r_(static_cast<std::size_t>(k) * static_cast<std::size_t>(k > 0 ? k - 1 : 0) / 2, 0.0),
        theta_(static_cast<std::size_t>(k), 0.0), tol_(static_cast<std::size_t>(k), 0.0),
        rss_(static_cast<std::size_t>(k), 0.0), order_(static_cast<std::size_t>(k)) {
    std::iota(order_.begin(), order_.end(), 0);
  }

  // Rotates one row into the reduction of columns `first` onward: `row` holds its entries from
  // column `first` on and is overwritten, `weight` is its weight and `y` its response.
  void rotate_in(int first, double weight, double* row, double y) {
    double w = weight, yy = y;
    for (int i = first; i < k_; ++i) {
      if (w == 0.0) return;
      const double xi = row[i - first];
      if (xi == 0.0) continue;
      const double di = d_[static_cast<std::size_t>(i)];
      const double wxi = w * xi;
      const double dpi = di + wxi * xi;
      const double cbar = di / dpi;
      const double sbar = wxi / dpi;
      w = cbar * w;
      d_[static_cast<std::size_t>(i)] = dpi;
      if (i + 1 < k_) {
        double* ri = &r(i, i + 1);
        for (int j = i + 1; j < k_; ++j) {
          const double xj = row[j - first];
          row[j - first] = xj - xi * ri[j - i - 1];
          ri[j - i - 1] = cbar * ri[j - i - 1] + sbar * xj;
        }
      }
      const double y_before = yy;
      yy = y_before - xi * theta_[static_cast<std::size_t>(i)];
      theta_[static_cast<std::size_t>(i)] =
          cbar * theta_[static_cast<std::size_t>(i)] + sbar * y_before;
    }
    sse_ += w * yy * yy;
  }

  // Each column's tolerance: `eps` times its norm in the reduction's scale.
  void set_tolerances() {
    const double eps = 5e-10;
    const std::vector<double> root = roots();
    for (int j = 0; j < k_; ++j) {
      double s = root[static_cast<std::size_t>(j)];
      for (int i = 0; i < j; ++i) s += std::abs(r(i, j)) * root[static_cast<std::size_t>(i)];
      tol_[static_cast<std::size_t>(j)] = eps * s;
    }
  }

  // Zeroes the entries of `R` below their column's tolerance and returns the columns whose
  // diagonal is, each a combination of those before it; the rest of such a column's row is
  // rotated into the columns after it.
  std::vector<int> find_dependent() {
    std::vector<double> root = roots();
    std::vector<int> dependent;
    for (int j = 0; j < k_; ++j) {
      const std::size_t js = static_cast<std::size_t>(j);
      const double tj = tol_[js];
      for (int i = 0; i < j; ++i) {
        if (std::abs(r(i, j)) * root[static_cast<std::size_t>(i)] < tj) r(i, j) = 0.0;
      }
      if (root[js] <= tj) {
        dependent.push_back(j);
        if (j < k_ - 1) {
          rotate_in(j + 1, d_[js], &r(j, j + 1), theta_[js]);
        } else {
          sse_ += d_[js] * sq(theta_[js]);
        }
        d_[js] = 0.0;
        root[js] = 0.0;
        theta_[js] = 0.0;
      }
    }
    return dependent;
  }

  void set_rss() {
    double s = sse_;
    rss_[static_cast<std::size_t>(k_) - 1] = sse_;
    for (int i = k_ - 1; i >= 1; --i) {
      s += d_[static_cast<std::size_t>(i)] * sq(theta_[static_cast<std::size_t>(i)]);
      rss_[static_cast<std::size_t>(i) - 1] = s;
    }
  }

  // Moves the column at `from` to `to`, past every column between.
  void move_later(int from, int to) {
    for (int m = from; m < to; ++m) swap_adjacent(m);
  }

  // The column in `first` to `last` whose removal from the first `last + 1` adds least to the
  // residual sum of squares, and what it adds; a column whose diagonal is below its tolerance
  // adds nothing, and the last of those is taken.
  int cheapest_drop(int first, int last, double& added) const {
    int which = -1;
    added = 1e35;
    std::vector<double> rest(static_cast<std::size_t>(k_));
    for (int j = first; j <= last; ++j) {
      double dj = d_[static_cast<std::size_t>(j)];
      if (std::sqrt(dj) < tol_[static_cast<std::size_t>(j)]) {
        added = 0.0;
        which = j;
        continue;
      }
      double rhs = theta_[static_cast<std::size_t>(j)];
      for (int i = j + 1; i <= last; ++i) rest[static_cast<std::size_t>(i)] = r(j, i);
      for (int i = j + 1; i <= last; ++i) {
        const double xv = rest[static_cast<std::size_t>(i)];
        const double di = d_[static_cast<std::size_t>(i)];
        if (std::abs(xv) * std::sqrt(dj) < tol_[static_cast<std::size_t>(i)] || di == 0.0) {
          continue;
        }
        dj = dj * di / (di + dj * sq(xv));
        for (int c = i + 1; c <= last; ++c) {
          rest[static_cast<std::size_t>(c)] = rest[static_cast<std::size_t>(c)] - xv * r(i, c);
        }
        rhs = rhs - xv * theta_[static_cast<std::size_t>(i)];
      }
      const double ss = rhs * dj * rhs;
      if (ss < added) {
        which = j;
        added = ss;
      }
    }
    return which;
  }

  int size() const { return k_; }
  double rss_first(int s) const { return rss_[static_cast<std::size_t>(s) - 1]; }
  const std::vector<int>& order() const { return order_; }

 private:
  std::size_t row_start(int i) const {
    return static_cast<std::size_t>(i) * static_cast<std::size_t>(2 * k_ - i - 1) / 2;
  }
  double& r(int i, int j) { return r_[row_start(i) + static_cast<std::size_t>(j - i - 1)]; }
  double r(int i, int j) const { return r_[row_start(i) + static_cast<std::size_t>(j - i - 1)]; }

  std::vector<double> roots() const {
    std::vector<double> root(static_cast<std::size_t>(k_));
    for (std::size_t i = 0; i < root.size(); ++i) root[i] = std::sqrt(d_[i]);
    return root;
  }

  // Rotates columns `m` and `m + 1` past each other, keeping the reduction of the same design.
  void swap_adjacent(int m) {
    const std::size_t a = static_cast<std::size_t>(m), b = a + 1;
    const double d1 = d_[a];
    const double d2 = d_[b];
    if (!(d1 == 0.0 && d2 == 0.0)) {
      double xv = r(m, m + 1);
      if (std::abs(xv) * std::sqrt(d1) < tol_[b]) xv = 0.0;
      if (d1 == 0.0 || xv == 0.0) {
        d_[a] = d2;
        d_[b] = d1;
        r(m, m + 1) = 0.0;
        for (int c = m + 2; c < k_; ++c) std::swap(r(m, c), r(m + 1, c));
        std::swap(theta_[a], theta_[b]);
      } else if (d2 == 0.0) {
        d_[a] = d1 * sq(xv);
        r(m, m + 1) = 1.0 / xv;
        for (int c = m + 2; c < k_; ++c) r(m, c) = r(m, c) / xv;
        theta_[a] = theta_[a] / xv;
      } else {
        const double d1new = d2 + d1 * sq(xv);
        const double cbar = d2 / d1new;
        const double sbar = xv * d1 / d1new;
        const double d2new = d1 * cbar;
        d_[a] = d1new;
        d_[b] = d2new;
        r(m, m + 1) = sbar;
        for (int c = m + 2; c < k_; ++c) {
          const double upper = r(m, c);
          r(m, c) = cbar * r(m + 1, c) + sbar * upper;
          r(m + 1, c) = upper - xv * r(m + 1, c);
        }
        const double upper = theta_[a];
        theta_[a] = cbar * theta_[b] + sbar * upper;
        theta_[b] = upper - xv * theta_[b];
      }
    }
    for (int i = 0; i < m; ++i) std::swap(r(i, m), r(i, m + 1));
    std::swap(order_[a], order_[b]);
    std::swap(tol_[a], tol_[b]);
    rss_[a] = rss_[b] + d_[b] * sq(theta_[b]);
  }

  int k_;
  std::vector<double> d_, r_, theta_, tol_, rss_;
  std::vector<int> order_;  // the design column at each position
  double sse_ = 0.0;
};

// The best subset of every size the backward elimination passes through: its residual sum of
// squares and its columns, indexed by size.
struct Subsets {
  std::vector<double> rss;
  std::vector<std::vector<int>> members;

  // Records the first `size` columns of `red` where they beat the subset held, unless they fit
  // within 0.01% of it and are the same columns.
  void offer(const Reduction& red, int size) {
    const std::size_t s = static_cast<std::size_t>(size);
    const double ssq = red.rss_first(size);
    if (ssq >= rss[s]) return;
    const std::vector<int>& ord = red.order();
    if (ssq > 0.9999 * rss[s]) {
      bool same = true;
      for (int j = 0; j < size && same; ++j) {
        same = std::find(members[s].begin(), members[s].end(), ord[static_cast<std::size_t>(j)]) !=
               members[s].end();
      }
      if (same) return;
    }
    rss[s] = ssq;
    members[s].assign(ord.begin(), ord.begin() + size);
  }
};

// Backward elimination over the columns of `bxw` [n, m], the first held in: from the full model,
// the column whose removal costs least is moved behind the others, and every subset that leaves
// is offered as the best of its size. A column the reduction finds dependent on those before it
// is dropped and the reduction rebuilt. The members are indices into `bxw`'s columns.
Subsets backward_subsets(const std::vector<double>& bxw, std::size_t n, int m, const double* y) {
  std::vector<int> cols(static_cast<std::size_t>(m));
  std::iota(cols.begin(), cols.end(), 0);
  Reduction red(0);
  for (;;) {
    const int k = static_cast<int>(cols.size());
    red = Reduction(k);
    std::vector<double> row(static_cast<std::size_t>(k));
    for (std::size_t i = 0; i < n; ++i) {
      for (int c = 0; c < k; ++c) {
        const std::size_t col = static_cast<std::size_t>(cols[static_cast<std::size_t>(c)]);
        row[static_cast<std::size_t>(c)] = bxw[i + col * n];
      }
      red.rotate_in(0, 1.0, row.data(), y[i]);
    }
    red.set_tolerances();
    const std::vector<int> dependent = red.find_dependent();
    if (dependent.empty()) break;
    if (dependent.front() == 0) {
      throw Error("the intercept of the MARS basis is linearly dependent; the weights are not "
                  "all positive.");
    }
    std::vector<int> kept;
    for (int c = 0; c < k; ++c) {
      if (std::find(dependent.begin(), dependent.end(), c) == dependent.end()) {
        kept.push_back(cols[static_cast<std::size_t>(c)]);
      }
    }
    cols = kept;
  }
  red.set_rss();

  const int k = red.size();
  Subsets s;
  s.rss.assign(static_cast<std::size_t>(k) + 1, 0.0);
  s.members.assign(static_cast<std::size_t>(k) + 1, std::vector<int>());
  for (int size = 1; size <= k; ++size) {
    s.rss[static_cast<std::size_t>(size)] = red.rss_first(size);
    s.members[static_cast<std::size_t>(size)].assign(red.order().begin(),
                                                     red.order().begin() + size);
  }
  for (int last = k - 1; last >= 2; --last) {
    double added = 0.0;
    const int j = red.cheapest_drop(1, last, added);
    if (j >= 0 && j < last) {
      red.move_later(j, last);
      for (int size = j + 1; size <= last; ++size) s.offer(red, size);
    }
  }
  for (std::vector<int>& mem : s.members) {
    for (int& v : mem) v = cols[static_cast<std::size_t>(v)];
  }
  return s;
}

// Generalised cross-validation of the best subset of each size (Craven and Wahba 1979), a subset
// of `k` terms charged `k` plus `penalty` for each of its `(k - 1) / 2` knots; -1 charges nothing.
std::vector<double> subset_gcv(const Subsets& s, std::size_t n, double penalty) {
  const int top = static_cast<int>(s.rss.size()) - 1;
  std::vector<double> gcv(static_cast<std::size_t>(top) + 1, kInf);
  const double nd = static_cast<double>(n);
  for (int k = 1; k <= top; ++k) {
    const double nparams = penalty < 0.0 ? 0.0 : k + penalty * (k - 1) / 2.0;
    gcv[static_cast<std::size_t>(k)] =
        nparams >= nd ? kInf : s.rss[static_cast<std::size_t>(k)] / (nd * sq(1.0 - nparams / nd));
  }
  return gcv;
}

// The forward pass's live terms less any a rank-revealing QR of their columns finds dependent on
// those before it.
std::vector<int> independent_terms(const ForwardPass& fwd, std::size_t n, int n_max) {
  std::vector<int> kept;
  for (int t = 0; t < n_max; ++t) {
    if (fwd.live[static_cast<std::size_t>(t)]) kept.push_back(t);
  }
  std::vector<double> a(n * kept.size());
  for (std::size_t c = 0; c < kept.size(); ++c) {
    std::memcpy(a.data() + c * n, fwd.values.data() + static_cast<std::size_t>(kept[c]) * n,
                n * sizeof(double));
  }
  std::size_t rank = 0;
  std::vector<double> qraux;
  std::vector<std::size_t> jpvt;
  detail::householder_qr(a.data(), n, kept.size(), kRankTol, rank, qraux, jpvt);
  if (rank == kept.size()) return kept;
  std::vector<char> drop(kept.size(), 0);
  for (std::size_t c = rank; c < kept.size(); ++c) drop[jpvt[c]] = 1;
  std::vector<int> out;
  for (std::size_t c = 0; c < kept.size(); ++c) {
    if (!drop[c]) out.push_back(kept[c]);
  }
  return out;
}

}  // namespace

Mars mars_fit(const double* x, const double* y, const double* w, std::size_t n, std::size_t p,
              const MarsSpec& spec) {
  if (n < 2) throw Error("MARS is fitted on two units or more.");
  if (p < 1) throw Error("MARS is fitted on one column or more.");
  if (spec.degree < 1 || spec.degree > kMaxDegree) {
    throw Error("a MARS `degree` is between 1 and 100, got " + std::to_string(spec.degree) + ".");
  }
  if (spec.thresh < 0.0 || spec.thresh >= 1.0) {
    throw Error("a MARS `thresh` is in [0, 1).");
  }
  if (spec.fast_beta < 0.0) throw Error("a MARS `fast_beta` is zero or more.");
  if (spec.endspan < 0) throw Error("a MARS `endspan` is zero or more.");
  if (spec.adjust_endspan < 0.0 || spec.adjust_endspan > 10.0) {
    throw Error("a MARS `adjust_endspan` is between 0 and 10.");
  }
  detail::check_finite(x, n * p, "MARS", "design");
  detail::check_finite(y, n, "MARS", "response");
  detail::check_finite(w, n, "MARS", "weights");
  for (std::size_t i = 0; i < n; ++i) {
    if (w[i] < 0.0) throw Error("MARS takes weights of zero or more.");
  }
  const int pi = static_cast<int>(p);
  const int nk = spec.nk > 0 ? spec.nk : std::min(200, std::max(20, 2 * pi)) + 1;
  if (nk > 1000) throw Error("a MARS `nk` is at most 1000.");
  const double penalty = std::isnan(spec.penalty) ? (spec.degree > 1 ? 3.0 : 2.0) : spec.penalty;
  if (penalty < 0.0 && penalty != -1.0) {
    throw Error("a MARS `penalty` is zero or more, or -1 to charge nothing.");
  }
  MarsSpec s = spec;
  s.fast_k = spec.fast_k <= 0 ? 10001 : std::max(3, spec.fast_k);
  s.threads = std::max(1, spec.threads);

  // Weights all equal to the first are no weights. Otherwise one far below the mean is raised to
  // the mean over 1e8, so its root, which scales the unit's terms, stays positive.
  bool use_weights = false;
  for (std::size_t i = 0; i < n; ++i) {
    if (std::abs(w[i] - w[0]) > 1e-8) use_weights = true;
  }
  std::vector<double> wt(n, 1.0);
  if (use_weights) {
    const double mw = mean_of(w, n);
    if (!(mw >= 1e-8)) throw Error("MARS takes weights whose mean is above zero.");
    const double floor_w = mw / 1e8;
    for (std::size_t i = 0; i < n; ++i) wt[i] = w[i] < floor_w ? floor_w : w[i];
  }

  // The forward pass reads the response centred and scaled by its standard deviation, and the
  // weighted one the response times the root weights, scaled by the same deviation.
  double ym = 0.0;
  for (std::size_t i = 0; i < n; ++i) ym += y[i];
  ym /= static_cast<double>(n);
  double sd = 0.0;
  for (std::size_t i = 0; i < n; ++i) sd += sq(y[i] - ym);
  sd = std::sqrt(sd / static_cast<double>(n - 1));
  std::vector<double> ys(n), yws;
  for (std::size_t i = 0; i < n; ++i) ys[i] = sd > 0.0 ? (y[i] - ym) / sd : y[i];
  if (use_weights) {
    yws.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
      const double yw = std::sqrt(wt[i]) * y[i];
      yws[i] = sd > 0.0 ? yw / sd : yw;
    }
  }

  ForwardPass fwd(x, ys.data(), use_weights ? yws.data() : nullptr, wt.data(), n, p, s, nk,
                  penalty);
  fwd.run();

  const std::vector<int> terms = independent_terms(fwd, n, nk);
  const int m = static_cast<int>(terms.size());

  Mars fit;
  fit.family = spec.family;
  fit.n_column = static_cast<std::int32_t>(p);
  fit.termcond = fwd.termcond;
  fit.factor_start.push_back(0);
  for (int t : terms) {
    const Term& term = fwd.terms[static_cast<std::size_t>(t)];
    for (int j = 0; j < pi; ++j) {
      const std::size_t js = static_cast<std::size_t>(j);
      if (term.dir[js] == 0) continue;
      fit.factor_column.push_back(j);
      fit.factor_dir.push_back(term.dir[js]);
      fit.factor_cut.push_back(term.cut[js]);
    }
    fit.factor_start.push_back(static_cast<std::int32_t>(fit.factor_column.size()));
  }

  std::vector<double> bxw(n * static_cast<std::size_t>(m));
  for (int c = 0; c < m; ++c) {
    const std::size_t t = static_cast<std::size_t>(terms[static_cast<std::size_t>(c)]);
    std::memcpy(bxw.data() + static_cast<std::size_t>(c) * n, fwd.values.data() + t * n,
                n * sizeof(double));
  }
  std::vector<double> yp(n);
  for (std::size_t i = 0; i < n; ++i) yp[i] = use_weights ? std::sqrt(wt[i]) * y[i] : y[i];

  const Subsets subsets = backward_subsets(bxw, n, m, yp.data());
  const std::vector<double> gcv = subset_gcv(subsets, n, penalty);
  const int top = static_cast<int>(subsets.rss.size()) - 1;
  const int nprune = spec.nprune > 0 ? std::min(spec.nprune, top) : top;
  std::vector<int> chosen;
  if (!spec.prune) {
    chosen.resize(static_cast<std::size_t>(nprune));
    std::iota(chosen.begin(), chosen.end(), 0);
    fit.gcv = gcv[static_cast<std::size_t>(nprune)];
  } else {
    int k_best = 1;
    for (int k = 2; k <= nprune; ++k) {
      if (gcv[static_cast<std::size_t>(k)] < gcv[static_cast<std::size_t>(k_best)]) k_best = k;
    }
    chosen = subsets.members[static_cast<std::size_t>(k_best)];
    std::sort(chosen.begin(), chosen.end());
    fit.gcv = gcv[static_cast<std::size_t>(k_best)];
  }
  fit.selected.assign(chosen.begin(), chosen.end());

  // The kept terms without the root weights, refitted.
  const std::size_t k = chosen.size();
  std::vector<double> basis(n * k);
  for (std::size_t c = 0; c < k; ++c) {
    const double* src = bxw.data() + static_cast<std::size_t>(chosen[c]) * n;
    for (std::size_t i = 0; i < n; ++i) basis[i + c * n] = src[i] / std::sqrt(wt[i]);
  }
  if (spec.family != Family::gaussian) {
    const Glm g = glm_fit(basis.data(), n, k, y, w, spec.family, spec.epsilon, spec.max_iter);
    fit.beta = g.beta;
    fit.converged = g.converged;
  } else {
    std::vector<double> a(n * k), b(n);
    for (std::size_t i = 0; i < n; ++i) {
      const double r = use_weights ? std::sqrt(wt[i]) : 1.0;
      b[i] = y[i] * r;
      for (std::size_t c = 0; c < k; ++c) a[i + c * n] = basis[i + c * n] * r;
    }
    std::size_t rank = 0;
    fit.beta = detail::least_squares(a.data(), n, k, b.data(), 1e-7, rank);
    if (rank < k) {
      throw Error("the MARS terms kept are linearly dependent, so their least-squares refit is "
                  "not unique.");
    }
  }
  return fit;
}

void mars_basis(const Mars& fit, const double* x, std::size_t n, std::size_t p, double* out) {
  if (static_cast<std::int32_t>(p) != fit.n_column) {
    throw Error("the MARS model was fitted on " + std::to_string(fit.n_column) +
                " columns and is handed " + std::to_string(p) + ".");
  }
  for (std::size_t c = 0; c < fit.selected.size(); ++c) {
    const std::size_t t = static_cast<std::size_t>(fit.selected[c]);
    double* col = out + c * n;
    std::fill(col, col + n, 1.0);
    for (std::int32_t f = fit.factor_start[t]; f < fit.factor_start[t + 1]; ++f) {
      const double* xp = x + static_cast<std::size_t>(fit.factor_column[f]) * n;
      const double cut = fit.factor_cut[f];
      const std::int32_t dir = fit.factor_dir[f];
      for (std::size_t i = 0; i < n; ++i) {
        const double v = dir == 2 ? xp[i] : (dir == 1 ? xp[i] - cut : cut - xp[i]);
        col[i] *= dir == 2 ? v : (v > 0.0 ? v : 0.0);
      }
    }
  }
}

void mars_predict(const Mars& fit, const double* x, std::size_t n, std::size_t p, double* out) {
  const std::size_t k = fit.selected.size();
  std::vector<double> basis(n * k);
  mars_basis(fit, x, n, p, basis.data());
  for (std::size_t i = 0; i < n; ++i) {
    double eta = 0.0;
    for (std::size_t c = 0; c < k; ++c) eta += basis[i + c * n] * fit.beta[c];
    out[i] = linkinv(fit.family, eta);
  }
}

std::vector<Mars> mars_fits(const double* x, std::size_t n, std::size_t p, const double* y,
                            const double* w, std::size_t r, const MarsSpec& spec) {
  return detail::fit_responses(r, spec.threads, [&](std::size_t s, int inner) {
    MarsSpec each = spec;
    each.threads = inner;
    return mars_fit(x, y + s * n, w + s * n, n, p, each);
  });
}

}  // namespace timesift
