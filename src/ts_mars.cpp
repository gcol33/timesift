#include "ts_mars.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "ts_core.h"
#include "ts_glm.h"
#include "ts_internal.h"

// Every sum here is taken in the order earth and leaps take it, so a knot that ties another to
// the last place is broken the way earth breaks it. Contraction is off for the reason `ts_tree.cpp`
// gives.
#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

namespace timesift {

namespace {

// earth's constants, from `earth.c`.
constexpr double kMinGrsq = -10.0;
constexpr double kQrTol = 1e-8;
constexpr double kMinBxSos = 0.01;
constexpr double kAlmostZero = 1e-10;
constexpr int kMaxDegree = 100;
// The squared residual of a hinge, relative to its squared norm, below which a weighted fit takes
// it as a combination of the terms.
constexpr double kHingeDependent = 1e-10;
const double kInf = std::numeric_limits<double>::infinity();

double sq(double v) { return v * v; }

double maybe_zero(double v) { return (v > -kAlmostZero && v < kAlmostZero) ? 0.0 : v; }

double mean_of(const double* v, std::size_t n) {
  double m = 0.0;
  const double nd = static_cast<double>(n);
  for (std::size_t i = 0; i < n; ++i) m += v[i] / nd;
  return m;
}

double ss_about(const double* v, double mean, std::size_t n) {
  double s = 0.0;
  for (std::size_t i = 0; i < n; ++i) s += sq(v[i] - mean);
  return s;
}

double dot_of(const double* a, const double* b, std::size_t n) {
  double s = 0.0;
  for (std::size_t i = 0; i < n; ++i) s += a[i] * b[i];
  return s;
}

double gcv_of(int n_terms, std::size_t n, double rss, double penalty) {
  double cost = 0.0;
  if (penalty != -1.0) {
    const double n_knots = (static_cast<double>(n_terms) - 1.0) / 2.0;
    cost = (n_terms + penalty * n_knots) / static_cast<double>(n);
  }
  return cost >= 1.0 ? kInf : rss / (static_cast<double>(n) * sq(1.0 - cost));
}

// The fast MARS queue of parents, as `earth.c` keeps it: `q` indexed by the order terms entered it,
// `sorted` by rank, the highest reduction first and then the youngest by `fast_beta`.
struct QItem {
  int parent = 0;
  double delta = -1.0;
  int n_terms_for = -99;
  double aged = -1.0;
};

bool by_delta(const QItem& a, const QItem& b) {
  const double d = b.delta - a.delta;
  if (d < 0.0) return true;
  if (d > 0.0) return false;
  return a.parent < b.parent;
}

bool by_aged(const QItem& a, const QItem& b) {
  const double d = a.aged - b.aged;
  if (d < 0.0) return true;
  if (d > 0.0) return false;
  const double e = b.delta - a.delta;
  if (e < 0.0) return true;
  if (e > 0.0) return false;
  return a.parent < b.parent;
}

// What a hinge is orthogonalised against in the knot search: the centred basis, transposed so one
// unit's values across the terms lie together, the response's projection on each, and the
// candidate linear term when there is one.
struct ScanBasis {
  const double* ct = nullptr;  // [n_max, n]: term t of unit i at ct[t + i * n_max]
  int n_max = 0;
  int n_shared = 0;
  const double* cand_ct = nullptr;  // [n], or null
  const double* ycbo = nullptr;     // [n_shared]
  double cand_ycbo = 0.0;
};

// Friedman's knot search, `FindKnot` in `earth.c`: the column `pred` of the parent `parent`, from
// its second largest value down to `end_span`, updating the hinge's covariances with the basis and
// the response as each unit enters it. `s` centres the hinge: null for the unweighted fit, whose
// centring is on the constant and divides by `div = n`, and the weighted intercept's direction
// with `div = 1` otherwise. At every knot the span admits, `visit(i, temp1, temp2, cov_new)` is
// handed the hinge's covariance with the residual, its residual norm and its centred norm.
template <typename Visit>
void scan_knots(const double* x, const double* bx, const int* xorder, std::size_t n, int parent,
                int pred, const ScanBasis& b, const double* yc, const double* s, double div,
                int start_span, int min_span, int end_span, bool gate_on_cov,
                std::vector<double>& cov_sx, std::vector<double>& cov_col, Visit visit) {
  const int new_col = b.n_shared + (b.cand_ct ? 1 : 0);
  cov_col.assign(static_cast<std::size_t>(new_col) + 1, 0.0);
  cov_sx.assign(static_cast<std::size_t>(new_col) + 1, 0.0);
  double ycbo_new = 0.0, ybx = 0.0;
  double bx_sum = 0.0, bx_sq_sum = 0.0, bx_sqx_sum = 0.0, bxx_sum = 0.0, st = 0.0;
  int span = start_span;
  const double* xp = x + static_cast<std::size_t>(pred) * n;
  const int* op = xorder + static_cast<std::size_t>(pred) * n;
  const double* bp = bx + static_cast<std::size_t>(parent) * n;
  const std::size_t n_max = static_cast<std::size_t>(b.n_max);
  for (int i = static_cast<int>(n) - 2; i >= end_span; --i) {
    const int ix0 = op[i];
    const double x0 = xp[ix0];
    const int ix1 = op[i + 1];
    const double x1 = xp[ix1];
    const double bx1 = bp[ix1];
    const double bx_sq = sq(bx1);
    const double x_delta = x1 - x0;
    const double* row = b.ct + static_cast<std::size_t>(ix1) * n_max;
    for (int it = 0; it < b.n_shared; ++it) cov_sx[it] += bx1 * row[it];
    if (b.cand_ct) cov_sx[b.n_shared] += bx1 * b.cand_ct[ix1];
    for (int it = 0; it < new_col; ++it) cov_col[it] += x_delta * cov_sx[it];
    const double sb = s ? s[ix1] * bx1 : bx1;
    bx_sum += sb;
    bx_sq_sum += bx_sq;
    bxx_sum += sb * x1;
    bx_sqx_sum += bx_sq * x1;
    const double su = st;
    st = bxx_sum - bx_sum * x0;
    cov_col[new_col] += x_delta * (2.0 * bx_sqx_sum - bx_sq_sum * (x0 + x1)) +
                        (sq(su) - sq(st)) / div;
    ybx += yc[ix1] * bx1;
    ycbo_new += x_delta * ybx;
    if (bx1 > 0.0 && (!gate_on_cov || cov_col[new_col] > 0.0) && --span == 0) {
      span = min_span;
      double dot1 = 0.0, dot2 = 0.0;
      for (int it = 0; it < b.n_shared; ++it) dot1 += b.ycbo[it] * cov_col[it];
      if (b.cand_ct) dot1 += b.cand_ycbo * cov_col[b.n_shared];
      for (int it = 0; it < new_col; ++it) dot2 += cov_col[it] * cov_col[it];
      visit(i, ycbo_new - dot1, cov_col[new_col] - dot2, cov_col[new_col]);
    }
  }
}

struct Best {
  int case_ = -1;
  int pred = -1;
  int parent = -1;
  double delta = 0.0;
  bool is_new_form = false;
  bool lin_best = false;
};

// One column's candidates under one parent, found on its own so the columns of a parent can be
// searched at once and the best taken in column order afterwards, as earth's loop takes it.
struct PredResult {
  bool skip = true;
  bool new_form_in = false;  // the linear candidate was tried
  bool new_form = false;     // and the pair's upper hinge is usable
  double lin = 0.0;          // unweighted: the linear candidate's reduction
  int knot_case = -1;        // unweighted: the best knot's position, or -1
  double pair = 0.0;         // unweighted: the reduction of the linear term and that knot
  bool lin_best = false;     // weighted: no knot beat the linear candidate
  int case_ = 0;             // weighted: the knot's position, 0 for the linear candidate
  double delta = 0.0;        // weighted: the step's reduction on the tracked residuals
};

class Forward {
 public:
  Forward(const double* x, const double* y, const double* yw, const double* w, std::size_t n,
          std::size_t p, const MarsSpec& spec, int n_max, double penalty)
      : x_(x), y_(y), yw_(yw), w_(w), n_(n), p_(static_cast<int>(p)), spec_(spec),
        n_max_(n_max), penalty_(penalty), weighted_(yw != nullptr) {}

  void run();

  std::vector<char> full;
  std::vector<double> bx;
  std::vector<int> dirs;
  std::vector<double> cuts;
  int termcond = 0;

 private:
  double& B(std::size_t i, int t) { return bx[i + static_cast<std::size_t>(t) * n_]; }
  int& D(int t, int pred) { return dirs[static_cast<std::size_t>(t) +
                                        static_cast<std::size_t>(pred) * n_max_]; }
  double& C(int t, int pred) { return cuts[static_cast<std::size_t>(t) +
                                           static_cast<std::size_t>(pred) * n_max_]; }
  double X(std::size_t i, int pred) const { return x_[i + static_cast<std::size_t>(pred) * n_]; }
  int order(int i, int pred) const {
    return xorder_[static_cast<std::size_t>(i) + static_cast<std::size_t>(pred) * n_];
  }
  double* orth(int t) { return orth_.data() + static_cast<std::size_t>(t) * n_; }
  double& CT(int t, std::size_t i) { return ct_[static_cast<std::size_t>(t) + i * n_max_]; }

  int end_span_for(int degree) const;
  void span_params(int& min_span, int& end_span, int& start_span, int degree, int parent);
  bool new_form_flag(int pred, int term, int n_terms);
  void orthog_residuals(double* col, const double* v, int n_terms, const std::vector<char>& used,
                        std::vector<double>* cache);
  void init_orth_col(int t, const double* v, int n_terms, bool candidate, bool& good,
                     double* col, double& mean, std::vector<double>* cache);
  void add_to_q(int term, int n_terms, double delta, bool sort);
  void find_term(int n_terms, double rss, double max_legal, Best& best);
  PredResult search_unweighted(int parent, int pred, int n_terms, double max_legal,
                               int min_span, int end_span, int start_span);
  PredResult search_weighted(int parent, int pred, int n_terms, double rss, int min_span,
                             int end_span, int start_span);
  void add_term_pair(int n_terms, const Best& best, bool& is_new_form);
  void add_weighted_column(int t);

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

  std::vector<int> xorder_;
  std::vector<int> degree_;
  std::vector<int> uses_;
  std::vector<double> orth_, ct_, orth_mean_;
  std::vector<double> yc_;       // the scaled response less its mean
  std::vector<double> ycbo_;     // its projection on each column of `orth_`, per step
  std::vector<std::vector<double>> cache_;  // earth's beta cache, per column and parent

  // The weighted fit's own orthonormal basis of every column of `bx` so far, the one earth's QR
  // regresses on at each knot.
  std::vector<double> wq_, wct_, wycbo_, q0_, r0_, rw_;
  std::vector<char> wok_;
  std::vector<std::vector<double>> wcache_;
  double rss_exact_ = 0.0;

  std::vector<QItem> q_, sorted_;
  int nq_ = 0;
};

int Forward::end_span_for(int degree) const {
  int end_span = 1;
  if (spec_.endspan > 0) {
    end_span = spec_.endspan;
  } else {
    static const double log_2 = 0.69315;
    static const double temp1 = 7.32193;
    end_span = static_cast<int>(temp1 + std::log(static_cast<double>(p_)) / log_2);
  }
  if (degree >= 2) end_span += static_cast<int>(spec_.adjust_endspan * end_span + 0.5);
  const int half = static_cast<int>(n_) / 2 - 1;
  if (end_span > half) end_span = half;
  return std::max(1, end_span);
}

void Forward::span_params(int& min_span, int& end_span, int& start_span, int degree,
                          int parent) {
  end_span = end_span_for(degree);
  const int n = static_cast<int>(n_);
  if (spec_.minspan < 0) {
    min_span = static_cast<int>(std::ceil(static_cast<double>(n_) / (1.0 - spec_.minspan)));
    start_span = min_span;
    while (start_span < end_span) start_span += min_span;
    start_span = std::max(1, start_span - 1);
    return;
  }
  min_span = 0;
  if (spec_.minspan > 0) {
    min_span = spec_.minspan;
  } else {
    int used = 0;
    const double* bp = bx.data() + static_cast<std::size_t>(parent) * n_;
    for (std::size_t i = 0; i < n_; ++i) {
      if (bp[i] > 0.0) ++used;
    }
    static const double temp1 = 2.9702;
    static const double temp2 = 1.7329;
    min_span = static_cast<int>((temp1 + std::log(static_cast<double>(p_ * used))) / temp2);
  }
  min_span = std::max(1, min_span);
  const int avail = std::max(0, n - 2 * end_span);
  start_span = avail / 2;
  if (avail > min_span) {
    const int div = avail / min_span;
    start_span = avail == div * min_span ? min_span / 2 : (avail - div * min_span) / 2;
  }
  start_span = std::max(1, end_span + start_span);
}

bool Forward::new_form_flag(int pred, int term, int n_terms) {
  bool is_new = true;
  for (int i = 1; i < n_terms; ++i) {
    if (!full[i]) continue;
    is_new = false;
    if (D(i, pred) == 0) return true;
    for (int j = 0; j < p_; ++j) {
      if (j != pred && (D(i, j) != 0) != (D(term, j) != 0)) return true;
    }
  }
  return is_new;
}

// Residuals of `v` on the used columns of `orth_` below `n_terms`, by modified Gram-Schmidt, each
// projection read from `cache` where an earlier step computed it.
void Forward::orthog_residuals(double* col, const double* v, int n_terms,
                               const std::vector<char>& used, std::vector<double>* cache) {
  std::memcpy(col, v, n_ * sizeof(double));
  for (int t = 0; t < n_terms; ++t) {
    if (!used[t]) continue;
    const double* q = orth(t);
    double beta;
    if (cache && (*cache)[t] != kInf) {
      beta = (*cache)[t];
    } else {
      beta = dot_of(q, col, n_);
      if (cache) (*cache)[t] = beta;
    }
    const double neg = -beta;
    for (std::size_t i = 0; i < n_; ++i) col[i] += neg * q[i];
  }
}

// `InitBxOrthCol`: the column `t` of the basis from `v`, normalised, into `col` with its mean in
// `mean`. A candidate's column whose squared norm is at most `kMinBxSos` is zeroed; a term's is
// kept down to zero.
void Forward::init_orth_col(int t, const double* v, int n_terms, bool candidate, bool& good,
                            double* col, double& mean, std::vector<double>* cache) {
  good = true;
  if (t == 0) {
    const double len = 1.0 / std::sqrt(static_cast<double>(n_));
    for (std::size_t i = 0; i < n_; ++i) col[i] = len;
    mean = len;
    return;
  }
  if (t == 1) {
    const double m = mean_of(v, n_);
    for (std::size_t i = 0; i < n_; ++i) col[i] = v[i] - m;
  } else {
    orthog_residuals(col, v, n_terms, full, cache);
  }
  const double ss = ss_about(col, 0.0, n_);
  if (ss <= kMinBxSos) good = false;
  if (ss > (candidate ? kMinBxSos : 0.0)) {
    mean = mean_of(col, n_);
    const double len = std::sqrt(ss);
    for (std::size_t i = 0; i < n_; ++i) col[i] /= len;
  } else {
    mean = 0.0;
    std::fill(col, col + n_, 0.0);
  }
}

void Forward::add_to_q(int term, int n_terms, double delta, bool sort) {
  q_[nq_].n_terms_for = n_terms;
  q_[nq_].delta = std::max(q_[term].delta, delta);
  ++nq_;
  if (sort) {
    sorted_.assign(q_.begin(), q_.begin() + nq_);
    std::sort(sorted_.begin(), sorted_.end(), by_delta);
    if (spec_.fast_beta > 0.0) {
      for (int r = 0; r < nq_; ++r) {
        sorted_[r].aged = r + spec_.fast_beta * (n_terms - sorted_[r].n_terms_for);
      }
      std::sort(sorted_.begin(), sorted_.end(), by_aged);
    }
  }
}

PredResult Forward::search_unweighted(int parent, int pred, int n_terms, double max_legal,
                                      int min_span, int end_span, int start_span) {
  PredResult r;
  r.skip = false;
  bool is_new = new_form_flag(pred, parent, n_terms);
  r.new_form_in = is_new;
  std::vector<double> cand, cand_ct;
  double cand_ycbo = 0.0;
  double lin = 0.0;
  if (is_new) {
    std::vector<double> xbx(n_);
    const double* bp = bx.data() + static_cast<std::size_t>(parent) * n_;
    for (std::size_t i = 0; i < n_; ++i) xbx[i] = X(i, pred) * bp[i];
    std::vector<double>& cache = cache_[static_cast<std::size_t>(pred) * n_max_ + parent];
    if (cache.empty()) cache.assign(static_cast<std::size_t>(n_max_), kInf);
    cand.assign(n_, 0.0);
    double mean = 0.0;
    bool good = true;
    init_orth_col(n_terms, xbx.data(), n_terms, true, good, cand.data(), mean, &cache);
    if (!good) is_new = false;
    cand_ct.resize(n_);
    for (std::size_t i = 0; i < n_; ++i) cand_ct[i] = cand[i] - mean;
    for (std::size_t i = 0; i < n_; ++i) cand_ycbo += yc_[i] * cand[i];
    double ybo = 0.0;
    for (std::size_t i = 0; i < n_; ++i) ybo += y_[i] * cand[i];
    lin = sq(ybo);
  }
  r.lin = lin;
  r.new_form = is_new;
  r.pair = lin;
  ScanBasis b;
  b.ct = ct_.data();
  b.n_max = n_max_;
  b.n_shared = n_terms;
  b.ycbo = ycbo_.data();
  if (is_new) {
    b.cand_ct = cand_ct.data();
    b.cand_ycbo = cand_ycbo;
  }
  const int new_col = is_new ? n_terms + 1 : n_terms;
  const double tol = new_col < 15 ? 0.01 : 1e-5;
  std::vector<double> cov_sx, cov_col;
  scan_knots(x_, bx.data(), xorder_.data(), n_, parent, pred, b, yc_.data(), nullptr,
             static_cast<double>(n_), start_span, min_span, end_span, true, cov_sx, cov_col,
             [&](int i, double temp1, double temp2, double cov_new) {
               double delta = 0.0;
               if (temp2 / cov_new > tol) delta += sq(temp1) / temp2;
               delta = lin + delta;
               if (delta > r.pair && delta < max_legal) {
                 r.knot_case = i;
                 r.pair = delta;
               }
             });
  return r;
}

PredResult Forward::search_weighted(int parent, int pred, int n_terms, double rss, int min_span,
                                    int end_span, int start_span) {
  PredResult r;
  r.skip = false;
  const bool is_new = new_form_flag(pred, parent, n_terms);
  r.new_form = is_new;
  std::vector<double> cand_ct;
  double cand_ycbo = 0.0;
  bool has_cand = false;
  double before_knot = rss;      // RssBeforeKnot: the tracked residuals, or the linear fit's
  double base_knots = rss_exact_;
  if (is_new) {
    std::vector<double> v(n_), res(n_);
    const double* bp = bx.data() + static_cast<std::size_t>(parent) * n_;
    for (std::size_t i = 0; i < n_; ++i) v[i] = bp[i] * X(i, pred);
    std::vector<double>& cache = wcache_[static_cast<std::size_t>(pred) * n_max_ + parent];
    if (cache.empty()) cache.assign(static_cast<std::size_t>(n_max_), kInf);
    res = v;
    for (int t = 0; t < n_terms; ++t) {
      if (!wok_[t]) continue;
      const double* q = wq_.data() + static_cast<std::size_t>(t) * n_;
      double beta = cache[t];
      if (beta == kInf) {
        beta = dot_of(q, res.data(), n_);
        cache[t] = beta;
      }
      for (std::size_t i = 0; i < n_; ++i) res[i] -= beta * q[i];
    }
    const double norm_v = std::sqrt(dot_of(v.data(), v.data(), n_));
    const double norm_r = std::sqrt(dot_of(res.data(), res.data(), n_));
    if (norm_r >= kQrTol * (norm_v == 0.0 ? 1.0 : norm_v) && norm_r > 0.0) {
      has_cand = true;
      for (std::size_t i = 0; i < n_; ++i) res[i] /= norm_r;
      const double on_q0 = dot_of(res.data(), q0_.data(), n_);
      cand_ct.resize(n_);
      for (std::size_t i = 0; i < n_; ++i) cand_ct[i] = res[i] - on_q0 * q0_[i];
      cand_ycbo = dot_of(r0_.data(), res.data(), n_);
      const double lin_rss = rss_exact_ - sq(dot_of(rw_.data(), res.data(), n_));
      before_knot = lin_rss;
      base_knots = lin_rss;
    } else {
      before_knot = rss_exact_;
    }
  }
  double best_knot = before_knot;
  int best_case = 0;
  ScanBasis b;
  b.ct = wct_.data();
  b.n_max = n_max_;
  b.n_shared = n_terms;
  b.ycbo = wycbo_.data();
  std::vector<double> zero;
  if (is_new) {
    if (!has_cand) zero.assign(n_, 0.0);
    b.cand_ct = has_cand ? cand_ct.data() : zero.data();
    b.cand_ycbo = cand_ycbo;
  }
  std::vector<double> cov_sx, cov_col;
  scan_knots(x_, bx.data(), xorder_.data(), n_, parent, pred, b, r0_.data(), q0_.data(), 1.0,
             start_span, min_span, end_span, false, cov_sx, cov_col,
             [&](int i, double temp1, double temp2, double cov_new) {
               double delta = 0.0;
               if (temp2 > kHingeDependent * cov_new && temp2 > 0.0) delta = sq(temp1) / temp2;
               const double knot_rss = base_knots - delta;
               if (knot_rss < best_knot - kAlmostZero) {
                 best_case = i;
                 best_knot = knot_rss;
               }
             });
  r.lin_best = before_knot <= best_knot;
  r.case_ = best_case;
  r.delta = rss - (r.lin_best ? before_knot : best_knot);
  return r;
}

void Forward::find_term(int n_terms, double rss, double max_legal, Best& best) {
  best = Best();
  if (!weighted_) {
    ycbo_.assign(static_cast<std::size_t>(n_max_), 0.0);
    for (int t = 0; t < n_terms; ++t) {
      const double* q = orth(t);
      double s = 0.0;
      for (std::size_t i = 0; i < n_; ++i) s += yc_[i] * q[i];
      ycbo_[t] = s;
    }
  }
  const int limit = std::min(nq_, spec_.fast_k);
  for (int k = 0; k < limit; ++k) {
    const int parent = sorted_[k].parent;
    if (degree_[parent] >= spec_.degree) continue;
    int min_span = 0, end_span = 0, start_span = 0;
    span_params(min_span, end_span, start_span, degree_[parent] + 1, parent);
    std::vector<PredResult> found(static_cast<std::size_t>(p_));
    detail::run_tasks(static_cast<std::size_t>(p_), spec_.threads, [&](std::size_t j) {
      const int pred = static_cast<int>(j);
      if (D(parent, pred) != 0) return;
      found[j] = weighted_ ? search_weighted(parent, pred, n_terms, rss, min_span, end_span,
                                             start_span)
                           : search_unweighted(parent, pred, n_terms, max_legal, min_span,
                                               end_span, start_span);
    });
    double best_for_parent = -1.0;
    for (int pred = 0; pred < p_; ++pred) {
      const PredResult& r = found[static_cast<std::size_t>(pred)];
      if (r.skip) continue;
      if (weighted_) {
        if (r.delta > best_for_parent) best_for_parent = r.delta;
        if (r.delta > best.delta) {
          best.delta = r.delta;
          best.lin_best = r.lin_best;
          best.case_ = r.case_;
          best.pred = pred;
          best.parent = parent;
          best.is_new_form = r.new_form;
        }
        continue;
      }
      if (r.new_form_in) {
        if (r.lin > best_for_parent) best_for_parent = r.lin;
        if (r.lin > best.delta) {
          best.delta = r.lin;
          best.lin_best = true;
          best.case_ = 0;
          best.pred = pred;
          best.parent = parent;
        }
      }
      if (r.pair > best_for_parent) best_for_parent = r.pair;
      if (r.pair > best.delta) {
        best.delta = r.pair;
        best.lin_best = false;
        best.case_ = r.knot_case;
        best.pred = pred;
        best.parent = parent;
        best.is_new_form = r.new_form;
      }
    }
    q_[parent].n_terms_for = n_terms;
    q_[parent].delta = best_for_parent;
  }
}

void Forward::add_weighted_column(int t) {
  const double* v = bx.data() + static_cast<std::size_t>(t) * n_;
  double* q = wq_.data() + static_cast<std::size_t>(t) * n_;
  std::memcpy(q, v, n_ * sizeof(double));
  for (int k = 0; k < t; ++k) {
    if (!wok_[k]) continue;
    const double* qk = wq_.data() + static_cast<std::size_t>(k) * n_;
    const double beta = dot_of(qk, q, n_);
    for (std::size_t i = 0; i < n_; ++i) q[i] -= beta * qk[i];
  }
  const double norm_v = std::sqrt(dot_of(v, v, n_));
  const double norm_r = std::sqrt(dot_of(q, q, n_));
  wok_[t] = norm_r >= kQrTol * (norm_v == 0.0 ? 1.0 : norm_v) && norm_r > 0.0;
  if (!wok_[t]) {
    std::fill(q, q + n_, 0.0);
    for (std::size_t i = 0; i < n_; ++i) wct_[static_cast<std::size_t>(t) + i * n_max_] = 0.0;
    wycbo_[t] = 0.0;
    return;
  }
  for (std::size_t i = 0; i < n_; ++i) q[i] /= norm_r;
  const double on_q0 = t == 0 ? 1.0 : dot_of(q, q0_.data(), n_);
  for (std::size_t i = 0; i < n_; ++i) {
    wct_[static_cast<std::size_t>(t) + i * n_max_] = t == 0 ? 0.0 : q[i] - on_q0 * q0_[i];
  }
  if (t == 0) {
    q0_.assign(q, q + n_);
    const double on = dot_of(yw_, q, n_);
    r0_.resize(n_);
    for (std::size_t i = 0; i < n_; ++i) r0_[i] = yw_[i] - on * q[i];
    rw_ = r0_;
  } else {
    const double on = dot_of(rw_.data(), q, n_);
    for (std::size_t i = 0; i < n_; ++i) rw_[i] -= on * q[i];
  }
  wycbo_[t] = dot_of(r0_.data(), q, n_);
  rss_exact_ = dot_of(rw_.data(), rw_.data(), n_);
}

void Forward::add_term_pair(int n_terms, const Best& best, bool& is_new_form) {
  const int upper = n_terms + 1;
  for (int pred = 0; pred < p_; ++pred) {
    D(n_terms, pred) = D(upper, pred) = D(best.parent, pred);
    C(n_terms, pred) = C(upper, pred) = C(best.parent, pred);
  }
  degree_[n_terms] = degree_[upper] = degree_[best.parent] + 1;
  const int entry = best.lin_best ? 2 : 1;
  D(n_terms, best.pred) = entry;
  D(upper, best.pred) = -1;
  const double cut = X(static_cast<std::size_t>(order(best.case_, best.pred)), best.pred);
  C(n_terms, best.pred) = C(upper, best.pred) = cut;
  double* lo = bx.data() + static_cast<std::size_t>(n_terms) * n_;
  double* hi = bx.data() + static_cast<std::size_t>(upper) * n_;
  std::fill(lo, lo + n_, 0.0);
  std::fill(hi, hi + n_, 0.0);
  const double* bp = bx.data() + static_cast<std::size_t>(best.parent) * n_;
  if (entry == 2) {
    for (std::size_t i = 0; i < n_; ++i) lo[i] = bp[i] * X(i, best.pred);
  } else {
    for (int i = 0; i < static_cast<int>(n_); ++i) {
      const std::size_t io = static_cast<std::size_t>(order(i, best.pred));
      const double xi = X(io, best.pred);
      if (i > best.case_) {
        lo[io] = bp[io] * (xi - cut);
      } else {
        hi[io] = bp[io] * (cut - xi);
      }
    }
  }
  ++uses_[best.pred];
  full[n_terms] = 1;
  bool good = true;
  double mean = 0.0;
  init_orth_col(n_terms, lo, n_terms, false, good, orth(n_terms), mean, nullptr);
  orth_mean_[n_terms] = mean;
  for (std::size_t i = 0; i < n_; ++i) CT(n_terms, i) = orth(n_terms)[i] - mean;
  if (!best.lin_best && is_new_form) full[upper] = 1;
  if (full[upper]) {
    init_orth_col(upper, hi, upper, false, good, orth(upper), mean, nullptr);
    orth_mean_[upper] = mean;
    for (std::size_t i = 0; i < n_; ++i) CT(upper, i) = orth(upper)[i] - mean;
    if (weighted_ && !good) {
      is_new_form = false;
      full[upper] = 0;
    }
  }
  if (!full[upper]) {
    degree_[upper] = kMaxDegree + 1;
    std::fill(orth(upper), orth(upper) + n_, 0.0);
    orth_mean_[upper] = 0.0;
    for (std::size_t i = 0; i < n_; ++i) CT(upper, i) = 0.0;
  }
  if (weighted_) {
    add_weighted_column(n_terms);
    add_weighted_column(upper);
  }
}

void Forward::run() {
  const std::size_t nm = static_cast<std::size_t>(n_max_);
  full.assign(nm, 0);
  bx.assign(n_ * nm, 0.0);
  dirs.assign(nm * static_cast<std::size_t>(p_), 0);
  cuts.assign(nm * static_cast<std::size_t>(p_), 0.0);
  degree_.assign(nm, 0);
  uses_.assign(static_cast<std::size_t>(p_), 0);
  orth_.assign(n_ * nm, 0.0);
  ct_.assign(nm * n_, 0.0);
  orth_mean_.assign(nm, 0.0);
  cache_.assign(nm * static_cast<std::size_t>(p_), std::vector<double>());

  xorder_.resize(n_ * static_cast<std::size_t>(p_));
  for (int pred = 0; pred < p_; ++pred) {
    int* o = xorder_.data() + static_cast<std::size_t>(pred) * n_;
    std::iota(o, o + n_, 0);
    const double* xp = x_ + static_cast<std::size_t>(pred) * n_;
    std::stable_sort(o, o + n_, [xp](int a, int b) { return xp[a] < xp[b]; });
  }

  for (std::size_t i = 0; i < n_; ++i) B(i, 0) = std::sqrt(w_[i]);
  bool good = true;
  double mean = 0.0;
  init_orth_col(0, bx.data(), 0, false, good, orth(0), mean, nullptr);
  orth_mean_[0] = mean;
  for (std::size_t i = 0; i < n_; ++i) CT(0, i) = orth(0)[i] - mean;
  full[0] = 1;

  const double y_mean = mean_of(y_, n_);
  yc_.resize(n_);
  for (std::size_t i = 0; i < n_; ++i) yc_[i] = y_[i] - y_mean;
  double rss_null = 0.0, sum_y = 0.0, sum_w = 0.0;
  for (std::size_t i = 0; i < n_; ++i) {
    sum_y += w_[i] * y_[i];
    sum_w += w_[i];
  }
  const double wmean = sum_y / sum_w;
  for (std::size_t i = 0; i < n_; ++i) rss_null += w_[i] * sq(y_[i] - wmean);
  if (rss_null < 1e-8 * static_cast<double>(n_)) rss_null = 1e-8 * static_cast<double>(n_);

  if (weighted_) {
    wq_.assign(n_ * nm, 0.0);
    wct_.assign(nm * n_, 0.0);
    wycbo_.assign(nm, 0.0);
    wok_.assign(nm, 0);
    wcache_.assign(nm * static_cast<std::size_t>(p_), std::vector<double>());
    add_weighted_column(0);
  }

  double rss = rss_null, rss_delta = rss_null, rsq = 0.0, rsq_delta = 0.0;
  int n_used = 1;
  double gcv = 0.0;
  const double gcv_null = gcv_of(n_used, n_, rss_null, penalty_);
  q_.assign(nm, QItem());
  for (int t = 0; t < n_max_; ++t) q_[t].parent = t;
  nq_ = 0;
  add_to_q(0, 1, rss_null, true);
  int n_terms = 1;
  Best best;
  if (n_max_ >= 3) {
    for (;;) {
      if (rss <= 0.0) {
        throw Error("the forward pass reached a residual sum of squares of zero; the response is "
                    "fitted exactly by the terms so far, so lower `thresh` no further than it "
                    "stops there.");
      }
      const double max_legal = std::min(1.01 * rss, 10.0 * rss_delta);
      find_term(n_terms, rss, max_legal, best);
      rss_delta = best.delta;
      bool is_new_form = best.is_new_form;
      if (best.case_ >= 0) add_term_pair(n_terms, best, is_new_form);
      const bool is_pair = best.case_ > 0 && is_new_form;
      ++n_used;
      if (is_pair) ++n_used;
      rss = maybe_zero(rss - rss_delta);
      gcv = gcv_of(n_used, n_, rss, penalty_);
      const double old_rsq = rsq;
      rsq = 1.0 - rss / rss_null;
      rsq_delta = maybe_zero(rsq - old_rsq);
      if (best.case_ < 0) {
        full[n_terms] = full[n_terms + 1] = 0;
        break;
      }
      if (spec_.thresh != 0.0 && rsq_delta < spec_.thresh) {
        full[n_terms] = full[n_terms + 1] = 0;
        break;
      }
      const double grsq = 1.0 - gcv / gcv_null;
      if (spec_.thresh != 0.0 && grsq < kMinGrsq) {
        full[n_terms] = full[n_terms + 1] = 0;
        break;
      }
      if (!best.lin_best && is_new_form) {
        add_to_q(n_terms, n_terms, kInf, false);
        add_to_q(n_terms + 1, n_terms, kInf, true);
      } else {
        add_to_q(n_terms, n_terms, kInf, true);
      }
      n_terms += 2;
      if (rsq >= 1.0 - spec_.thresh) break;
      if (n_terms >= n_max_ - 1) break;
    }
  }
  const double grsq = 1.0 - gcv / gcv_null;
  if (n_max_ < 3) {
    termcond = 1;
  } else if (spec_.thresh != 0.0 && grsq < kMinGrsq) {
    termcond = grsq < -1000.0 ? 2 : 3;
  } else if (spec_.thresh != 0.0 && rsq_delta < spec_.thresh) {
    termcond = 4;
  } else if (rsq >= 1.0 - spec_.thresh) {
    termcond = 5;
  } else if (best.case_ < 0) {
    termcond = 6;
  } else {
    termcond = 7;
  }
}

// ---- the pruning pass: leaps' backward elimination (AS 274) ------------------------------------
//
// The arrays are one-based, as the Fortran they are read from, so every index below is the one
// the algorithm's own listing uses.

void includ(int np, double weight, double* xrow, double yelem, double* d, double* rbar,
            double* thetab, double& sserr) {
  double w = weight, y = yelem;
  int nextr = 1;
  for (int i = 1; i <= np; ++i) {
    if (w == 0.0) return;
    const double xi = xrow[i - 1];
    if (xi == 0.0) {
      nextr += np - i;
      continue;
    }
    const double di = d[i - 1];
    const double wxi = w * xi;
    const double dpi = di + wxi * xi;
    const double cbar = di / dpi;
    const double sbar = wxi / dpi;
    w = cbar * w;
    d[i - 1] = dpi;
    for (int k = i + 1; k <= np; ++k) {
      const double xk = xrow[k - 1];
      xrow[k - 1] = xk - xi * rbar[nextr - 1];
      rbar[nextr - 1] = cbar * rbar[nextr - 1] + sbar * xk;
      ++nextr;
    }
    const double xk = y;
    y = xk - xi * thetab[i - 1];
    thetab[i - 1] = cbar * thetab[i - 1] + sbar * xk;
  }
  sserr += w * y * y;
}

struct Reduction {
  int np = 0;
  std::vector<double> d, rbar, thetab, tol, rss;
  std::vector<int> vorder;
  double sserr = 0.0;
};

// `makeqr`, `tolset`, `sing` and `ssleaps` over the columns `cols` of `bx` [n, *]. Returns the
// positions `sing` finds linearly dependent, empty when there are none.
std::vector<int> reduce_columns(const std::vector<double>& bxw, const std::vector<int>& cols,
                                std::size_t n, const double* y, Reduction& red) {
  const int np = static_cast<int>(cols.size());
  red.np = np;
  const int nrbar = np * (np - 1) / 2;
  red.d.assign(static_cast<std::size_t>(np) + 1, 0.0);
  red.rbar.assign(static_cast<std::size_t>(nrbar) + 1, 0.0);
  red.thetab.assign(static_cast<std::size_t>(np) + 1, 0.0);
  red.sserr = 0.0;
  std::vector<double> xrow(static_cast<std::size_t>(np));
  for (std::size_t i = 0; i < n; ++i) {
    for (int c = 0; c < np; ++c) xrow[c] = bxw[i + static_cast<std::size_t>(cols[c]) * n];
    includ(np, 1.0, xrow.data(), y[i], red.d.data() + 1, red.rbar.data() + 1,
           red.thetab.data() + 1, red.sserr);
  }
  std::vector<double> work(static_cast<std::size_t>(np) + 1);
  red.tol.assign(static_cast<std::size_t>(np) + 1, 0.0);
  const double eps = 5e-10;
  for (int row = 1; row <= np; ++row) work[row] = std::sqrt(red.d[row]);
  for (int col = 1; col <= np; ++col) {
    int pos = col - 1;
    double sum = work[col];
    for (int row = 1; row <= col - 1; ++row) {
      sum += std::abs(red.rbar[pos]) * work[row];
      pos += np - row - 1;
    }
    red.tol[col] = eps * sum;
  }
  std::vector<int> lindep;
  for (int col = 1; col <= np; ++col) work[col] = std::sqrt(red.d[col]);
  for (int col = 1; col <= np; ++col) {
    const double temp = red.tol[col];
    int pos = col - 1;
    for (int row = 1; row <= col - 1; ++row) {
      if (std::abs(red.rbar[pos]) * work[row] < temp) red.rbar[pos] = 0.0;
      pos += np - row - 1;
    }
    if (work[col] <= temp) {
      lindep.push_back(col);
      if (col < np) {
        const int nc2 = np - col;
        const int pos2 = pos + np - col + 1;
        includ(nc2, red.d[col], red.rbar.data() + pos + 1, red.thetab[col],
               red.d.data() + col + 1, red.rbar.data() + pos2, red.thetab.data() + col + 1,
               red.sserr);
      } else {
        red.sserr += red.d[col] * sq(red.thetab[col]);
      }
      red.d[col] = 0.0;
      work[col] = 0.0;
      red.thetab[col] = 0.0;
    }
  }
  red.rss.assign(static_cast<std::size_t>(np) + 1, 0.0);
  double sum = red.sserr;
  red.rss[np] = red.sserr;
  for (int i = np; i >= 2; --i) {
    sum += red.d[i] * sq(red.thetab[i]);
    red.rss[i - 1] = sum;
  }
  red.vorder.resize(static_cast<std::size_t>(np) + 1);
  for (int i = 1; i <= np; ++i) red.vorder[i] = i;
  return lindep;
}

void vmove(Reduction& red, int from, int to) {
  const int np = red.np;
  if (from == to) return;
  int first, last, inc;
  if (from < to) {
    first = from;
    last = to - 1;
    inc = 1;
  } else {
    first = from - 1;
    last = to;
    inc = -1;
  }
  std::vector<double>& d = red.d;
  std::vector<double>& rbar = red.rbar;
  std::vector<double>& thetab = red.thetab;
  for (int m = first; inc > 0 ? m <= last : m >= last; m += inc) {
    int m1 = (m - 1) * (np + np - m) / 2 + 1;
    int m2 = m1 + np - m;
    const int mp1 = m + 1;
    const double d1 = d[m];
    const double d2 = mp1 <= np ? d[mp1] : 0.0;
    if (!(d1 == 0.0 && d2 == 0.0)) {
      double xv = rbar[m1];
      if (std::abs(xv) * std::sqrt(d1) < red.tol[mp1]) xv = 0.0;
      if (d1 == 0.0 || xv == 0.0) {
        d[m] = d2;
        d[mp1] = d1;
        rbar[m1] = 0.0;
        for (int col = m + 2; col <= np; ++col) {
          ++m1;
          const double t = rbar[m1];
          rbar[m1] = rbar[m2];
          rbar[m2] = t;
          ++m2;
        }
        const double t = thetab[m];
        thetab[m] = thetab[mp1];
        thetab[mp1] = t;
      } else if (d2 == 0.0) {
        d[m] = d1 * sq(xv);
        rbar[m1] = 1.0 / xv;
        for (int col = m + 2; col <= np; ++col) {
          ++m1;
          rbar[m1] = rbar[m1] / xv;
        }
        thetab[m] = thetab[m] / xv;
      } else {
        const double d1new = d2 + d1 * sq(xv);
        const double cbar = d2 / d1new;
        const double sbar = xv * d1 / d1new;
        const double d2new = d1 * cbar;
        d[m] = d1new;
        d[mp1] = d2new;
        rbar[m1] = sbar;
        for (int col = m + 2; col <= np; ++col) {
          ++m1;
          const double yv = rbar[m1];
          rbar[m1] = cbar * rbar[m2] + sbar * yv;
          rbar[m2] = yv - xv * rbar[m2];
          ++m2;
        }
        const double yv = thetab[m];
        thetab[m] = cbar * thetab[mp1] + sbar * yv;
        thetab[mp1] = yv - xv * thetab[mp1];
      }
    }
    if (m != 1) {
      int pos = m;
      for (int row = 1; row <= m - 1; ++row) {
        const double t = rbar[pos];
        rbar[pos] = rbar[pos - 1];
        rbar[pos - 1] = t;
        pos += np - row - 1;
      }
    }
    std::swap(red.vorder[m], red.vorder[mp1]);
    std::swap(red.tol[m], red.tol[mp1]);
    red.rss[m] = red.rss[mp1] + d[mp1] * sq(thetab[mp1]);
  }
}

// The residual sum of squares a drop of each position `first` to `last` would add; the least,
// and where, the last of equal ones where a diagonal is below its tolerance.
void drop1(const Reduction& red, int first, int last, double& smin, int& jmin) {
  const int np = red.np;
  const double large = 1e35;
  jmin = 0;
  smin = large;
  std::vector<double> wk(static_cast<std::size_t>(last) + 1);
  int pos1 = (first - 1) * (np + np - first) / 2 + 1;
  const int inc = np - last;
  for (int j = first; j <= last; ++j) {
    double d1 = red.d[j];
    if (std::sqrt(d1) < red.tol[j]) {
      smin = 0.0;
      jmin = j;
    } else {
      double rhs = red.thetab[j];
      if (j != last) {
        int pos = pos1;
        for (int i = j + 1; i <= last; ++i) {
          wk[i] = red.rbar[pos];
          ++pos;
        }
        pos += inc;
        for (int row = j + 1; row <= last; ++row) {
          const double xv = wk[row];
          const double d2 = red.d[row];
          if (std::abs(xv) * std::sqrt(d1) < red.tol[row] || d2 == 0.0) {
            pos += np - row;
            continue;
          }
          d1 = d1 * d2 / (d2 + d1 * sq(xv));
          for (int col = row + 1; col <= last; ++col) {
            wk[col] = wk[col] - xv * red.rbar[pos];
            ++pos;
          }
          rhs = rhs - xv * red.thetab[row];
          pos += inc;
        }
      }
      const double ss = rhs * d1 * rhs;
      if (ss < smin) {
        jmin = j;
        smin = ss;
      }
    }
    if (j < last) pos1 += np - j;
  }
}

struct Subsets {
  std::vector<double> ress;               // [size], one-based
  std::vector<std::vector<int>> lopt;     // [size]: the positions' variables
};

void report(Subsets& s, const Reduction& red, int pos, double ssq, int nvmax) {
  if (pos > nvmax) return;
  if (ssq >= s.ress[pos]) return;
  const double under1 = 0.9999;
  if (ssq > under1 * s.ress[pos]) {
    bool same = true;
    for (int j = 1; j <= pos && same; ++j) {
      const int v = red.vorder[j];
      same = std::find(s.lopt[pos].begin(), s.lopt[pos].end(), v) != s.lopt[pos].end();
    }
    if (same) return;
  }
  s.ress[pos] = ssq;
  s.lopt[pos].assign(red.vorder.begin() + 1, red.vorder.begin() + 1 + pos);
}

// Backward elimination with the intercept held in, as `leaps.setup(force.in = 1)` then
// `leaps.backward` run it over `bx` [n, m] weighted. Returns the residual sum of squares of the
// best subset of each size and its terms, as indices into `bx`'s columns.
Subsets backward_subsets(const std::vector<double>& bxw, std::size_t n, int m, const double* y) {
  std::vector<int> cols(static_cast<std::size_t>(m));
  std::iota(cols.begin(), cols.end(), 0);
  Reduction red;
  for (;;) {
    const std::vector<int> lindep = reduce_columns(bxw, cols, n, y, red);
    if (lindep.empty()) break;
    if (lindep.front() == 1) {
      throw Error("the intercept of the MARS basis is linearly dependent; the weights are not "
                  "all positive.");
    }
    std::vector<int> kept;
    for (int c = 1; c <= static_cast<int>(cols.size()); ++c) {
      if (std::find(lindep.begin(), lindep.end(), c) == lindep.end()) kept.push_back(cols[c - 1]);
    }
    cols = kept;
  }
  const int np = red.np;
  const int nvmax = np;
  Subsets s;
  s.ress.assign(static_cast<std::size_t>(nvmax) + 1, 0.0);
  s.lopt.assign(static_cast<std::size_t>(nvmax) + 1, std::vector<int>());
  for (int nvar = 1; nvar <= nvmax; ++nvar) {
    s.ress[nvar] = red.rss[nvar];
    s.lopt[nvar].assign(red.vorder.begin() + 1, red.vorder.begin() + 1 + nvar);
  }
  const int first = 2, last = np;
  for (int pos = last; pos >= first + 1; --pos) {
    double smin = 0.0;
    int jmin = 0;
    drop1(red, first, pos, smin, jmin);
    if (jmin > 0 && jmin < pos) {
      vmove(red, jmin, pos);
      for (int i = jmin; i <= pos - 1; ++i) report(s, red, i, red.rss[i], nvmax);
    }
  }
  for (int nvar = 1; nvar <= nvmax; ++nvar) {
    for (int& v : s.lopt[nvar]) v = cols[v - 1];
  }
  return s;
}

double linear_predictor(const double* basis, std::size_t n, std::size_t k, std::size_t i,
                        const std::vector<double>& beta) {
  double e = 0.0;
  for (std::size_t c = 0; c < k; ++c) e += basis[i + c * n] * beta[c];
  return e;
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

  // earth's weights: equal ones are no weights, and one far below the mean is raised to it over
  // 1e8 so its root stays positive.
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

  // `Scale.y`: the forward pass reads the response centred and scaled by its standard deviation.
  const double ym = [&] {
    double m = 0.0;
    for (std::size_t i = 0; i < n; ++i) m += y[i];
    return m / static_cast<double>(n);
  }();
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

  Forward fwd(x, ys.data(), use_weights ? yws.data() : nullptr, wt.data(), n, p, s, nk, penalty);
  fwd.run();

  // `RegressAndFix`: a term the QR finds dependent on those before it is dropped.
  std::vector<int> terms;
  for (int t = 0; t < nk; ++t) {
    if (fwd.full[t]) terms.push_back(t);
  }
  {
    std::vector<double> a(n * terms.size());
    for (std::size_t c = 0; c < terms.size(); ++c) {
      std::memcpy(a.data() + c * n, fwd.bx.data() + static_cast<std::size_t>(terms[c]) * n,
                  n * sizeof(double));
    }
    std::size_t rank = 0;
    std::vector<double> qraux;
    std::vector<std::size_t> jpvt;
    detail::dqrdc2(a.data(), n, terms.size(), kQrTol, rank, qraux, jpvt);
    if (rank < terms.size()) {
      std::vector<char> drop(terms.size(), 0);
      for (std::size_t c = rank; c < terms.size(); ++c) drop[jpvt[c]] = 1;
      std::vector<int> kept;
      for (std::size_t c = 0; c < terms.size(); ++c) {
        if (!drop[c]) kept.push_back(terms[c]);
      }
      terms = kept;
    }
  }
  const int m = static_cast<int>(terms.size());

  Mars fit;
  fit.family = spec.family;
  fit.n_column = static_cast<std::int32_t>(p);
  fit.termcond = fwd.termcond;
  fit.factor_start.push_back(0);
  for (int t : terms) {
    for (int pred = 0; pred < pi; ++pred) {
      const int dir = fwd.dirs[static_cast<std::size_t>(t) + static_cast<std::size_t>(pred) * nk];
      if (dir == 0) continue;
      fit.factor_column.push_back(pred);
      fit.factor_dir.push_back(dir);
      fit.factor_cut.push_back(
          fwd.cuts[static_cast<std::size_t>(t) + static_cast<std::size_t>(pred) * nk]);
    }
    fit.factor_start.push_back(static_cast<std::int32_t>(fit.factor_column.size()));
  }

  std::vector<double> bxw(n * static_cast<std::size_t>(m));
  for (int c = 0; c < m; ++c) {
    std::memcpy(bxw.data() + static_cast<std::size_t>(c) * n,
                fwd.bx.data() + static_cast<std::size_t>(terms[c]) * n, n * sizeof(double));
  }
  std::vector<double> yp(n);
  for (std::size_t i = 0; i < n; ++i) yp[i] = use_weights ? std::sqrt(wt[i]) * y[i] : y[i];

  const Subsets subsets = backward_subsets(bxw, n, m, yp.data());
  const int nvmax = static_cast<int>(subsets.ress.size()) - 1;
  std::vector<double> gcv(static_cast<std::size_t>(nvmax) + 1, kInf);
  for (int k = 1; k <= nvmax; ++k) {
    const double nparams = penalty < 0.0 ? 0.0 : k + penalty * (k - 1) / 2.0;
    gcv[k] = nparams >= static_cast<double>(n)
                 ? kInf
                 : subsets.ress[k] / (static_cast<double>(n) *
                                      sq(1.0 - nparams / static_cast<double>(n)));
  }
  const int nprune = spec.nprune > 0 ? std::min(spec.nprune, nvmax) : nvmax;
  std::vector<int> chosen;
  if (!spec.prune) {
    chosen.resize(static_cast<std::size_t>(nprune));
    std::iota(chosen.begin(), chosen.end(), 0);
    fit.gcv = gcv[nprune];
  } else {
    int k_best = 1;
    for (int k = 2; k <= nprune; ++k) {
      if (gcv[k] < gcv[k_best]) k_best = k;
    }
    chosen = subsets.lopt[k_best];
    std::sort(chosen.begin(), chosen.end());
    fit.gcv = gcv[k_best];
  }
  fit.selected.assign(chosen.begin(), chosen.end());

  // The kept terms unweighted, as earth divides them back, and refitted.
  const std::size_t k = chosen.size();
  std::vector<double> basis(n * k);
  for (std::size_t c = 0; c < k; ++c) {
    const double* src = bxw.data() + static_cast<std::size_t>(chosen[c]) * n;
    for (std::size_t i = 0; i < n; ++i) basis[i + c * n] = src[i] / std::sqrt(wt[i]);
  }
  if (spec.family == Family::binomial) {
    const Glm g = glm_fit(basis.data(), n, k, y, w, Family::binomial, spec.epsilon, spec.max_iter);
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
    fit.beta = detail::dqrls(a.data(), n, k, b.data(), 1e-7, rank);
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
    const double e = linear_predictor(basis.data(), n, k, i, fit.beta);
    out[i] = fit.family == Family::binomial ? logit_linkinv(e) : e;
  }
}

}  // namespace timesift
