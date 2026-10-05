#include "ts_fda.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <numeric>
#include <string>

#include "ts_glm.h"
#include "ts_internal.h"
#include "ts_penalised.h"

// The fit is pinned to the bit against reference output in the fixtures, and a fused multiply-add
// rounds once where the separate product and sum round twice, so contraction is off.
#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

namespace timesift {

namespace {

// Thresholds held at their single-precision values, which is what the reference fits compare
// against: the least norm a candidate column keeps after orthogonalisation, and the ceiling on a
// candidate's gain relative to the residual sum of squares, above which it is taken for rounding.
const double kMinNorm = static_cast<double>(0.01f);
const double kGainCeiling = static_cast<double>(1.01f);
// The forward pass stops once the generalised cross-validation passes this multiple of the null's.
constexpr double kGcvStop = 10.0;
// The relative tolerance at which a least-squares refit sets a basis column aside as aliased.
constexpr double kAliasTol = 1e-2;
// Friedman's (1991, eq. 43 and 45) significance level for the knot spacing rules.
constexpr double kSpanAlpha = 5e-2;

double sq(double v) { return v * v; }

// Generalised cross-validation (Friedman 1991, eq. 30 and 32) of a residual sum of squares `rss`
// over `n` units, for a basis of `dof` terms beyond the intercept each charged `penalty`.
double gcv(double rss, double dof, double penalty, double n) {
  const double cost = (1 + dof) + penalty * (.5 * dof);
  return (rss / n) / ((1.0 - cost / n) * (1.0 - cost / n));
}

// Friedman's knot spacing for a parent nonzero on `support` units among `p` columns: knots stand
// `gap` sorted readings apart (eq. 43) and at least `margin` in from either end (eq. 45).
struct Spacing {
  int gap;
  int margin;
};

Spacing knot_spacing(std::size_t p, int support) {
  double level = -(1.0 / static_cast<double>(static_cast<long long>(p) * support)) *
                 std::log(1.0 - kSpanAlpha);
  const int gap = static_cast<int>(-1.0 * (std::log(level) / std::log(2.0)) / 2.5);
  level = kSpanAlpha / static_cast<double>(p);
  const int margin = static_cast<int>(3.0 - std::log(level) / std::log(2.0));
  return {gap, margin};
}

// A term of the basis: a product of hinges, at most one per column. `dir[v]` is 1 for
// `max(0, x - cut)`, -1 for `max(0, cut - x)` and 0 where column `v` does not enter.
struct Term {
  std::vector<signed char> dir;
  std::vector<double> cut;
  int degree = 0;
};

// The best addition one column offers under one parent. `rank` counts the sorted readings at or
// below the knot, and is 1 for the column entering linearly, as a hinge at its least reading.
// `pair` is whether both hinges enter, which they do where the column is not already in the model
// on the parent's other columns.
struct Offer {
  double gain = 0.0;
  int rank = 0;
  bool pair = false;
};

// A least-squares fit of the response on a subset of the basis.
struct Refit {
  std::size_t rank = 0;
  std::vector<std::size_t> pivot;  // the subset position now at each position
  double rss = 0.0;
};

// Multivariate adaptive regression splines (Friedman 1991) of one response on the columns of `x`.
//
// The forward pass adds, at each step, the hinge pair (or single hinge, or linear column) that
// most lowers the residual sum of squares, found by sweeping each column's knots from the top down
// with running updates of the candidate's covariances against an orthonormal basis of the terms
// already in (Friedman 1991, section 3.9). Terms are held in slots added two at a time, the second
// left inactive where only one hinge enters. The backward pass (section 3.6) refits by QR and
// drops the term of least t statistic until none is left, keeping the subset of least generalised
// cross-validation.
class HingeBasis {
 public:
  HingeBasis(const double* x, std::size_t n, std::size_t p, const double* y, int max_degree,
             double penalty, int max_terms, double thresh, int threads)
      : x_(x), y_(y), n_(n), p_(p), cap_(static_cast<std::size_t>(max_terms)),
        max_degree_(max_degree), penalty_(penalty), thresh_(thresh), threads_(threads),
        dn_(static_cast<double>(n)), order_(n * p), terms_(cap_), active_(cap_, false),
        raw_(n * cap_, 0.0), orth_(n * cap_, 0.0), orth_rows_(n * cap_, 0.0),
        orth_mean_(cap_, 0.0), orth_cov_(cap_, 0.0), residual_(n, 0.0), coef_(cap_, 0.0),
        coef_var_(cap_, 0.0) {
    for (Term& t : terms_) {
      t.dir.assign(p, 0);
      t.cut.assign(p, 0.0);
    }
    for (std::size_t v = 0; v < p; ++v) {
      std::size_t* ord = order_.data() + v * n;
      std::iota(ord, ord + n, std::size_t{0});
      const double* col = x + v * n;
      std::stable_sort(ord, ord + n, [col](std::size_t a, std::size_t b) { return col[a] < col[b]; });
    }
  }

  void grow();
  void select(bool eliminate);

  std::size_t slots() const { return slots_; }
  int forward_terms() const {
    return static_cast<int>(std::count(active_.begin(), active_.begin() + slots_, true));
  }
  double best_gcv() const { return best_gcv_; }
  bool kept(std::size_t t) const { return keep_[t]; }
  const Term& term(std::size_t t) const { return terms_[t]; }
  // Coefficients by position among the kept terms; entries past the rank keep the values an
  // earlier refit left.
  double coef(std::size_t j) const { return coef_[j]; }
  double fitted(std::size_t i) const { return y_[i] - residual_[i]; }

 private:
  double* raw(std::size_t t) { return raw_.data() + t * n_; }
  const double* raw(std::size_t t) const { return raw_.data() + t * n_; }
  double* orth(std::size_t t) { return orth_.data() + t * n_; }
  const double* orth(std::size_t t) const { return orth_.data() + t * n_; }

  bool extends_existing(std::size_t parent, std::size_t v) const;
  void project_out(std::size_t count, const double* src, double* dst) const;
  void centre(const double* src, double* dst) const;
  void finish_column(std::size_t t);
  void record_column(std::size_t t);
  Offer offer(std::size_t parent, std::size_t v, const Spacing& sp, double rss,
              double prev_gain) const;
  Offer best_offer(double rss, double prev_gain, std::size_t& parent, std::size_t& column) const;
  void add_terms(std::size_t parent, std::size_t v, int rank, bool pair);
  Refit refit(const std::vector<bool>& in, bool variances);

  const double* x_;
  const double* y_;
  std::size_t n_, p_, cap_;
  int max_degree_;
  double penalty_, thresh_;
  int threads_;
  double dn_;
  std::vector<std::size_t> order_;  // [n, p]: each column's units in ascending order, stable
  std::vector<Term> terms_;
  std::vector<bool> active_, searched_, keep_;
  // [n, cap] column-major: each term's values, the same orthonormalised against the active terms
  // before it, and the latter again row-major for the knot sweep.
  std::vector<double> raw_, orth_, orth_rows_;
  std::vector<double> orth_mean_, orth_cov_;  // each orthonormal column's mean and centred cov
  std::vector<double> residual_, coef_, coef_var_;
  std::size_t slots_ = 0;
  double ybar_ = 0.0;
  double best_gcv_ = 0.0;
};

// Whether an active term already carries column `v` on exactly the parent's other columns, in
// which case the column does not enter linearly under the parent.
bool HingeBasis::extends_existing(std::size_t parent, std::size_t v) const {
  const Term& pt = terms_[parent];
  for (std::size_t t = 1; t < slots_; ++t) {
    if (!active_[t]) continue;
    const Term& tt = terms_[t];
    if (tt.dir[v] == 0) continue;
    bool same = true;
    for (std::size_t j = 0; j < p_ && same; ++j) {
      if (j != v && (tt.dir[j] != 0) != (pt.dir[j] != 0)) same = false;
    }
    if (same) return true;
  }
  return false;
}

// `src` less its projections, one after another, on the active orthonormal columns among the
// first `count` slots.
void HingeBasis::project_out(std::size_t count, const double* src, double* dst) const {
  std::copy(src, src + n_, dst);
  for (std::size_t t = 0; t < count; ++t) {
    if (!active_[t]) continue;
    const double* q = orth(t);
    double dot = 0.0, norm = 0.0;
    for (std::size_t i = 0; i < n_; ++i) {
      dot = dot + dst[i] * q[i];
      norm = norm + q[i] * q[i];
    }
    const double b = dot / norm;
    for (std::size_t i = 0; i < n_; ++i) dst[i] = dst[i] - b * q[i];
  }
}

// Against the intercept alone, orthogonalising is centring.
void HingeBasis::centre(const double* src, double* dst) const {
  double m = 0.0;
  for (std::size_t i = 0; i < n_; ++i) m = m + src[i] / dn_;
  for (std::size_t i = 0; i < n_; ++i) dst[i] = src[i] - m;
}

// Normalises the orthogonalised column in slot `t` and records what the knot sweep reads of it.
void HingeBasis::finish_column(std::size_t t) {
  double* q = orth(t);
  double m = 0.0, ss = 0.0;
  for (std::size_t i = 0; i < n_; ++i) m = m + q[i] / dn_;
  for (std::size_t i = 0; i < n_; ++i) ss = ss + q[i] * q[i];
  orth_mean_[t] = m;
  if (ss > 0.0) {
    const double s = std::sqrt(ss);
    for (std::size_t i = 0; i < n_; ++i) q[i] = q[i] / s;
  }
  record_column(t);
}

// The covariance of slot `t`'s orthonormal column with the centred response, and its row-major
// copy.
void HingeBasis::record_column(std::size_t t) {
  const double* q = orth(t);
  double c = 0.0;
  for (std::size_t i = 0; i < n_; ++i) c = c + (y_[i] - ybar_) * q[i];
  orth_cov_[t] = c;
  for (std::size_t i = 0; i < n_; ++i) orth_rows_[t + i * cap_] = q[i];
}

// The gain of the best term column `v` offers under `parent`: the linear entry, then every
// admissible knot, from the column's top reading down. For a knot at `xk` the hinge
// `h = b (x - xk)_+` gains `(c_hy - C_hQ C_Qy)^2 / (v_h - |C_hQ|^2)` over the model, with `C_hQ` its
// covariances with the model's orthonormal columns; moving the knot down one reading updates every
// sum in constant time per column (Friedman 1991, eq. 52).
Offer HingeBasis::offer(std::size_t parent, std::size_t v, const Spacing& sp, double rss,
                        double prev_gain) const {
  Offer best;
  if (terms_[parent].dir[v] != 0) return best;
  const int n = static_cast<int>(n_);
  const double* xv = x_ + v * n_;
  const double* hp = raw(parent);
  const std::size_t* ord = order_.data() + v * n_;

  bool linear = !extends_existing(parent, v);
  std::vector<double> lin;
  double lin_mean = 0.0, lin_cov = 0.0, base = 0.0;
  if (linear) {
    std::vector<double> prod(n_);
    for (std::size_t i = 0; i < n_; ++i) prod[i] = xv[i] * hp[i];
    lin.resize(n_);
    if (slots_ > 1) {
      project_out(slots_, prod.data(), lin.data());
    } else {
      centre(prod.data(), lin.data());
    }
    for (std::size_t i = 0; i < n_; ++i) lin_mean = lin_mean + lin[i] / dn_;
    double ss = 0.0;
    for (std::size_t i = 0; i < n_; ++i) ss = ss + lin[i] * lin[i];
    if (ss > kMinNorm) {
      const double s = std::sqrt(ss);
      for (std::size_t i = 0; i < n_; ++i) lin[i] = lin[i] / s;
      double yl = 0.0;
      for (std::size_t i = 0; i < n_; ++i) {
        lin_cov = lin_cov + (y_[i] - ybar_) * lin[i];
        yl = yl + y_[i] * lin[i];
      }
      base = yl * yl;
      if (base > best.gain) best = {base, 1, false};
    } else {
      linear = false;
    }
  }

  // Where the column enters linearly, a knot's hinge is taken against it too, as one more column
  // of the model.
  const std::size_t width = slots_ + (linear ? 1 : 0);
  std::vector<double> mean(width), cov(width);
  std::copy(orth_mean_.begin(), orth_mean_.begin() + slots_, mean.begin());
  std::copy(orth_cov_.begin(), orth_cov_.begin() + slots_, cov.begin());
  if (linear) {
    mean[slots_] = lin_mean;
    cov[slots_] = lin_cov;
  }
  // Running over the units above the knot: `above` the parent-weighted centred model columns,
  // `cross` the hinge's covariance with each, `y_above` and `y_cross` the same for the response,
  // `b`, `bx`, `b2`, `b2x` the parent's moments, `h` the hinge's sum and `var_h` its centred sum
  // of squares.
  std::vector<double> above(width, 0.0), cross(width, 0.0);
  double y_above = 0.0, y_cross = 0.0, var_h = 0.0;
  double b = 0.0, bx = 0.0, b2 = 0.0, b2x = 0.0, h = 0.0, h_prev = 0.0;
  for (int k = n - 1; k > 0; --k) {
    const std::size_t lo = ord[k - 1], hi = ord[k];
    const double bh = hp[hi];
    const double xlo = xv[lo], xhi = xv[hi];
    const double dx = xhi - xlo;
    const double* row = orth_rows_.data() + hi * cap_;
    for (std::size_t j = 0; j < slots_; ++j) {
      above[j] = above[j] + (row[j] - mean[j]) * bh;
      cross[j] = cross[j] + dx * above[j];
    }
    if (linear) {
      above[slots_] = above[slots_] + (lin[hi] - mean[slots_]) * bh;
      cross[slots_] = cross[slots_] + dx * above[slots_];
    }
    b2x = b2x + (bh * bh) * xhi;
    b2 = b2 + bh * bh;
    b = b + bh;
    bx = bx + bh * xhi;
    h_prev = h;
    h = bx - b * xlo;
    var_h = var_h + dx * (2 * b2x - b2 * (xlo + xhi)) + ((h_prev * h_prev) - (h * h)) / dn_;
    y_above = y_above + (y_[hi] - ybar_) * bh;
    y_cross = y_cross + dx * y_above;
    double num = y_cross, den = var_h;
    for (std::size_t j = 0; j < width; ++j) {
      num = num - cov[j] * cross[j];
      den = den - cross[j] * cross[j];
    }
    double gain = base;
    if (var_h > 0 && den / var_h > kMinNorm) gain = gain + (num * num) / den;
    if (gain > kGainCeiling * rss) gain = 0.0;
    if (gain > 2 * prev_gain) gain = 0.0;
    if (gain > best.gain && k % sp.gap == 0 && k >= sp.margin && k <= n - sp.margin && bh > 0 &&
        !(k > 1 && xlo == xv[ord[k - 2]])) {
      best = {gain, k, linear};
    }
  }
  return best;
}

// The best offer over every active parent below the degree limit and every column, the columns
// searched at once and merged in column order, so the choice is the same on any thread count.
Offer HingeBasis::best_offer(double rss, double prev_gain, std::size_t& parent,
                             std::size_t& column) const {
  Offer best;
  parent = 0;
  column = 0;
  std::vector<Offer> found(p_);
  for (std::size_t m = 0; m < slots_; ++m) {
    if (!active_[m] || terms_[m].degree >= max_degree_) continue;
    int support = 0;
    const double* hm = raw(m);
    for (std::size_t i = 0; i < n_; ++i) support += hm[i] > 0 ? 1 : 0;
    const Spacing sp = knot_spacing(p_, support);
    detail::run_tasks(p_, threads_,
                      [&](std::size_t v) { found[v] = offer(m, v, sp, rss, prev_gain); });
    for (std::size_t v = 0; v < p_; ++v) {
      if (found[v].gain > best.gain) {
        best = found[v];
        parent = m;
        column = v;
      }
    }
  }
  return best;
}

// Fills the next two slots with the hinges on column `v` at the reading of `rank` under
// `parent`, the second active only for a pair.
void HingeBasis::add_terms(std::size_t parent, std::size_t v, int rank, bool pair) {
  const std::size_t a = slots_, c = slots_ + 1;
  const double* xv = x_ + v * n_;
  const double knot = xv[order_[static_cast<std::size_t>(rank - 1) + v * n_]];
  for (std::size_t t : {a, c}) {
    terms_[t] = terms_[parent];
    terms_[t].degree = terms_[parent].degree + 1;
    terms_[t].cut[v] = knot;
  }
  terms_[a].dir[v] = 1;
  terms_[c].dir[v] = -1;
  active_[a] = true;
  active_[c] = pair;

  const double* hp = raw(parent);
  double* ha = raw(a);
  double* hc = raw(c);
  for (std::size_t i = 0; i < n_; ++i) {
    if (xv[i] - knot > 0) ha[i] = hp[i] * (xv[i] - knot);
    if (knot - xv[i] >= 0) hc[i] = hp[i] * (knot - xv[i]);
  }
  if (slots_ == 1) {
    centre(ha, orth(a));
  } else {
    project_out(slots_, ha, orth(a));
  }
  if (pair) {
    project_out(slots_ + 1, hc, orth(c));
  } else {
    std::fill(orth(c), orth(c) + n_, 0.0);
  }
  finish_column(a);
  finish_column(c);
  slots_ += 2;
}

void HingeBasis::grow() {
  active_[0] = true;
  std::fill(raw(0), raw(0) + n_, 1.0);
  const double unit = 1.0 / std::sqrt(dn_);
  std::fill(orth(0), orth(0) + n_, unit);
  for (std::size_t i = 0; i < n_; ++i) ybar_ = ybar_ + y_[i] / dn_;
  double rss_null = 0.0;
  for (std::size_t i = 0; i < n_; ++i) rss_null = rss_null + (y_[i] - ybar_) * (y_[i] - ybar_);
  orth_mean_[0] = 1 / std::sqrt(dn_);
  record_column(0);
  slots_ = 1;

  const double gcv_null = gcv(rss_null, 0.0, penalty_, dn_);
  double rss = rss_null, dof = 0.0, prev_gain = 10e9;
  while (slots_ < cap_ && rss / rss_null > thresh_) {
    searched_ = active_;
    std::size_t parent = 0, column = 0;
    const Offer best = best_offer(rss, prev_gain, parent, column);
    const bool pair = best.rank > 1 && best.pair;
    double trial_dof = dof + 1;
    if (pair) trial_dof = trial_dof + 1;
    const double trial_gcv = gcv(rss - best.gain, trial_dof, penalty_, dn_);
    if (!(best.gain / rss > thresh_ && trial_gcv / gcv_null < kGcvStop)) break;
    dof = trial_dof;
    rss = rss - best.gain;
    prev_gain = best.gain;
    add_terms(parent, column, best.rank, pair);
  }
}

// Least squares of the response on the active columns `in` marks, by pivoted QR. With
// `variances`, also the diagonal of (R'R)^-1, the coefficients' variances up to the residual
// variance, which is all the t statistics' ordering needs.
Refit HingeBasis::refit(const std::vector<bool>& in, bool variances) {
  std::vector<std::size_t> cols;
  for (std::size_t t = 0; t < slots_; ++t) {
    if (in[t]) cols.push_back(t);
  }
  const std::size_t m = cols.size();
  std::vector<double> qr(n_ * m);
  for (std::size_t c = 0; c < m; ++c) std::copy(raw(cols[c]), raw(cols[c]) + n_, qr.data() + c * n_);
  Refit out;
  std::vector<double> qraux;
  detail::householder_qr(qr.data(), n_, m, kAliasTol, out.rank, qraux, out.pivot);
  const std::size_t r = out.rank;

  std::vector<double> qty(y_, y_ + n_);
  detail::apply_qt(qr.data(), n_, r, qraux.data(), qty.data());
  std::vector<double> beta(qty.begin(), qty.begin() + r);
  detail::back_substitute(qr.data(), n_, r, beta.data());
  std::copy(beta.begin(), beta.end(), coef_.begin());
  std::vector<double> fit(n_, 0.0);
  std::copy(qty.begin(), qty.begin() + r, fit.begin());
  detail::apply_q(qr.data(), n_, r, qraux.data(), fit.data());
  for (std::size_t i = 0; i < n_; ++i) {
    residual_[i] = y_[i] - fit[i];
    out.rss = out.rss + residual_[i] * residual_[i];
  }
  if (!variances) return out;

  // The `j`th column of R^-1 solves the leading (j + 1) x (j + 1) triangle against e_j.
  std::vector<double> rinv(r * r, 0.0);
  for (std::size_t j = 0; j < r; ++j) {
    double* col = rinv.data() + j * r;
    col[j] = 1.0;
    detail::back_substitute(qr.data(), n_, j + 1, col);
  }
  for (std::size_t i = 0; i < r; ++i) {
    double s = 0.0;
    for (std::size_t k = i; k < r; ++k) s = s + rinv[i + k * r] * rinv[i + k * r];
    coef_var_[i] = s;
  }
  return out;
}

// Drops the terms the full refit finds aliased, then, with `eliminate`, removes one term at a time,
// the one whose squared coefficient over its variance is least, and keeps the subset of least
// generalised cross-validation.
void HingeBasis::select(bool eliminate) {
  keep_ = active_;
  double dof = -1;
  for (std::size_t t = 0; t < slots_; ++t) dof = dof + (active_[t] ? 1 : 0);
  Refit full = refit(keep_, false);
  // The aliased columns are read by their position among the kept terms, which is the term set
  // the reference fits keep.
  const std::size_t count = static_cast<std::size_t>(dof + 1);
  for (std::size_t c = full.rank; c < count; ++c) {
    keep_[full.pivot[c]] = false;
    active_[full.pivot[c]] = false;
    dof = dof - 1;
  }
  best_gcv_ = gcv(full.rss, dof, penalty_, dn_);
  if (!eliminate) return;

  // The first elimination reads the coefficients of the model the last forward search started
  // from.
  Refit fit = refit(searched_, true);
  std::vector<bool> trial = keep_;
  while (dof > 0) {
    std::size_t pos = 0, weakest = 0;
    double least = 10e99;
    for (std::size_t t = 1; t < slots_; ++t) {
      if (!trial[t]) continue;
      ++pos;
      const double stat = (coef_[pos] * coef_[pos]) / coef_var_[pos];
      if (stat < least) {
        weakest = t;
        least = stat;
      }
    }
    if (weakest == 0) break;
    const double rss = fit.rss + least;
    dof = dof - 1;
    const double g = gcv(rss, dof, penalty_, dn_);
    trial[weakest] = false;
    if (g < best_gcv_) {
      best_gcv_ = g;
      keep_ = trial;
    }
    fit.rss = rss;
    if (dof > 0) fit = refit(trial, true);
  }
  refit(keep_, true);
}

// Optimal scores for two classes (Hastie, Tibshirani & Buja 1994, section 3): the unit vector
// orthogonal to the roots of the classes' weighted shares, over those roots, so the scored
// response has weighted mean zero and weighted variance one. False where the shares leave no
// such vector.
bool two_class_scores(const double* y, const std::vector<double>& ww, std::size_t n,
                      double theta[2]) {
  const double dn = static_cast<double>(n);
  double share[2] = {0.0, 0.0};
  for (std::size_t i = 0; i < n; ++i) share[y[i] == 1.0 ? 1 : 0] += ww[i];
  share[0] = share[0] / dn;
  share[1] = share[1] / dn;
  const double total = share[0] + share[1];
  const double root[2] = {std::sqrt(share[0] / total), std::sqrt(share[1] / total)};
  // The roots beside the contrast (-1, 1), each scaled by its class's root; the second column of
  // the orthogonal factor is the score direction.
  double basis[4] = {1 * root[0], 1 * root[1], -1 * root[0], 1 * root[1]};
  std::size_t rank = 0;
  std::vector<double> qraux;
  std::vector<std::size_t> pivot;
  detail::householder_qr(basis, 2, 2, 1e-7, rank, qraux, pivot);
  if (rank < 2) return false;
  double e2[2] = {0.0, 1.0};
  detail::apply_q(basis, 2, rank, qraux.data(), e2);
  theta[0] = e2[0] / root[0];
  theta[1] = e2[1] / root[1];
  return true;
}

void check_input(const double* x, const double* y, const double* w, std::size_t n, std::size_t p,
                 const FdaSpec& spec, std::size_t count[2]) {
  if (n < 2) throw Error("a discriminant is fitted on two units or more.");
  if (p < 1) throw Error("a discriminant is fitted on one column or more.");
  if (spec.degree < 1) throw Error("a discriminant's `degree` is 1 or more.");
  if (!(spec.thresh >= 0.0 && spec.thresh < 1.0)) {
    throw Error("a discriminant's `thresh` is in [0, 1).");
  }
  detail::check_finite(x, n * p, "a discriminant", "design");
  detail::check_finite(w, n, "a discriminant", "weights");
  count[0] = count[1] = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (y[i] != 0.0 && y[i] != 1.0) {
      throw Error("a discriminant separates presences from absences, and the response holds " +
                  std::to_string(y[i]) + " at position " + std::to_string(i + 1) + ".");
    }
    if (!(w[i] > 0.0)) throw Error("a discriminant takes weights above zero.");
    ++count[y[i] == 1.0 ? 1 : 0];
  }
  if (count[0] == 0 || count[1] == 0) {
    throw Error("a discriminant is fitted on a response holding both classes.");
  }
}

// The kept terms as factors in column order, and their coefficients by position.
void store_terms(const HingeBasis& basis, std::size_t p, Fda& out) {
  std::size_t kept = 0;
  out.factor_start.push_back(0);
  for (std::size_t t = 0; t < basis.slots(); ++t) {
    if (!basis.kept(t)) continue;
    ++kept;
    const Term& term = basis.term(t);
    for (std::size_t v = 0; v < p; ++v) {
      if (term.dir[v] == 0) continue;
      out.factor_column.push_back(static_cast<std::int32_t>(v));
      out.factor_dir.push_back(term.dir[v] > 0 ? 1 : -1);
      out.factor_cut.push_back(term.cut[v]);
    }
    out.factor_start.push_back(static_cast<std::int32_t>(out.factor_column.size()));
  }
  out.coef.resize(kept);
  for (std::size_t j = 0; j < kept; ++j) out.coef[j] = basis.coef(j);
}

// The value of the fitted basis at row `i` of `x` [n, ...].
double basis_value(const Fda& fit, const double* x, std::size_t i, std::size_t n) {
  double f = 0.0;
  for (std::size_t t = 0; t < fit.coef.size(); ++t) {
    double b = 1.0;
    for (std::int32_t q = fit.factor_start[t]; q < fit.factor_start[t + 1]; ++q) {
      const double d = fit.factor_dir[q] * (x[i + fit.factor_column[q] * n] - fit.factor_cut[q]);
      b = b * d * (d > 0 ? 1.0 : 0.0);
    }
    f = f + fit.coef[t] * b;
  }
  return f;
}

// The second class's posterior at a basis value `f`: two normal classes of unit variance around
// the centroids on the canonical variate, under the priors.
double posterior(const Fda& fit, double f) {
  const double z = (f * fit.direction) / fit.scale;
  const double d0 = sq(z - fit.centroid[0]), d1 = sq(z - fit.centroid[1]);
  const double dmin = std::min(d0, d1);
  const double p0 = std::exp(-0.5 * (d0 - dmin)) * fit.prior[0];
  const double p1 = std::exp(-0.5 * (d1 - dmin)) * fit.prior[1];
  return p1 / (p0 + p1);
}

}  // namespace

Fda fda_fit(const double* x, const double* y, const double* w, std::size_t n, std::size_t p,
            const FdaSpec& spec) {
  std::size_t count[2];
  check_input(x, y, w, n, p, spec, count);
  const double penalty = std::isnan(spec.penalty) ? (spec.degree > 1 ? 3.0 : 2.0) : spec.penalty;
  int nk = spec.nk > 0 ? spec.nk : std::max(21, 2 * static_cast<int>(p) + 1);
  if (nk % 2 != 1) nk = nk - 1;
  if (nk < 3) throw Error("a discriminant's `nk` is 3 or more.");

  const double dn = static_cast<double>(n);
  Fda out;
  out.n_column = static_cast<std::int32_t>(p);
  out.prior[0] = static_cast<double>(count[0]) / dn;
  out.prior[1] = static_cast<double>(count[1]) / dn;
  out.mean = out.prior[1];

  // The weights rescaled to sum to the count of units.
  double sw = 0.0;
  for (std::size_t i = 0; i < n; ++i) sw = sw + w[i];
  std::vector<double> ww(n);
  for (std::size_t i = 0; i < n; ++i) ww[i] = (dn * w[i]) / sw;
  double theta[2];
  if (!two_class_scores(y, ww, n, theta)) {
    out.discriminates = false;
    return out;
  }
  std::vector<double> scored(n);
  for (std::size_t i = 0; i < n; ++i) scored[i] = theta[y[i] == 1.0 ? 1 : 0];

  // The basis is fitted to the scores unweighted, which is how the reference fits treat the case
  // weights.
  HingeBasis basis(x, n, p, scored.data(), spec.degree, penalty, nk, spec.thresh,
                   std::max(1, spec.threads));
  basis.grow();
  basis.select(spec.prune);
  out.forward_terms = basis.forward_terms();
  out.gcv = basis.best_gcv();
  store_terms(basis, p, out);

  // The canonical variate (Hastie, Tibshirani & Buja 1994, section 3): its eigenvalue is the
  // weighted mean product of the scores with their fit, and it is the fit rescaled so the classes'
  // centroids sit on it at unit within-class variance.
  double ssm = 0.0;
  for (std::size_t i = 0; i < n; ++i) ssm = ssm + basis.fitted(i) * (scored[i] * ww[i]);
  ssm = ssm / dn;
  double lambda = std::abs(ssm);
  out.direction = ssm < 0.0 ? -1.0 : 1.0;
  if (lambda > 1 - DBL_EPSILON) lambda = 1 - DBL_EPSILON;
  if (!(lambda > DBL_EPSILON)) {
    out.discriminates = false;
    return out;
  }
  const double alpha = std::sqrt(lambda);
  const double sqima = std::sqrt(1 - lambda);
  for (int j = 0; j < 2; ++j) out.centroid[j] = (theta[j] * out.direction) / (sqima / alpha);
  out.scale = sqima * alpha;

  // The recalibration: a probit regression of the response on the posterior, under the case
  // weights.
  if (spec.calibrate) {
    std::vector<double> design(2 * n, 1.0);
    fda_predict(out, x, n, p, design.data() + n);
    const Glm g = glm_fit(design.data(), n, 2, y, w, Family::binomial, spec.epsilon,
                          spec.max_iter, Link::probit);
    out.calibrated = true;
    out.calibration[0] = g.beta[0];
    out.calibration[1] = g.beta[1];
    out.converged = g.converged;
  }
  return out;
}

void fda_predict(const Fda& fit, const double* x, std::size_t n, std::size_t p, double* out) {
  if (static_cast<std::size_t>(fit.n_column) != p) {
    throw Error("the discriminant was fitted on " + std::to_string(fit.n_column) +
                " columns and is asked to predict on " + std::to_string(p) + ".");
  }
  if (!fit.discriminates) {
    std::fill(out, out + n, fit.mean);
    return;
  }
  for (std::size_t i = 0; i < n; ++i) {
    const double post = posterior(fit, basis_value(fit, x, i, n));
    out[i] = fit.calibrated
                 ? probit_linkinv(fit.calibration[0] * 1 + post * fit.calibration[1])
                 : post;
  }
}

std::vector<Fda> fda_fits(const double* x, std::size_t n, std::size_t p, const double* y,
                          const double* w, std::size_t r, const FdaSpec& spec) {
  return detail::fit_responses(r, spec.threads, [&](std::size_t s, int inner) {
    FdaSpec each = spec;
    each.threads = inner;
    return fda_fit(x, y + s * n, w + s * n, n, p, each);
  });
}

}  // namespace timesift
