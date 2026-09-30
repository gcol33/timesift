#include "ts_fda.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <numeric>
#include <string>

#include "ts_glm.h"
#include "ts_internal.h"
#include "ts_penalised.h"

// The basis is mda's Fortran `marss` and the scoring its R `fda()`, operation for operation, so
// the terms kept are mda's. Contraction is off for the reason `ts_tree.cpp` gives.
#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

namespace timesift {

namespace {

// `marss` writes these two as single-precision literals, which is the value it compares against.
const double kTolBx = static_cast<double>(0.01f);
const double kRssGuard = static_cast<double>(1.01f);
constexpr double kStopFac = 10.0;
constexpr double kQrTol = 1e-2;

// The best term a column offers under one parent: `k` is 1 for the column entering linearly and
// otherwise the rank of the knot among the column's sorted readings.
struct Candidate {
  double crit = 0.0;
  int k = 0;
  bool newform = false;
};

// mda's MARS on an `nclass`-column response. Indices below are `marss`'s, from one: a term `t`, a
// unit `i`, a column `v`, a sorted position `k`.
class MdaMars {
 public:
  MdaMars(const double* x, std::size_t n, std::size_t p, const double* y, std::size_t nclass,
          int degree, double penalty, int mmax, double thresh, int threads)
      : x_(x), n_(n), p_(p), y_(y), nc_(nclass), maxorder_(degree), penalty_(penalty),
        mmax_(mmax), thresh_(thresh), threads_(threads), dn_(static_cast<double>(n)),
        tagx_(n * p), bx_(n * mmax, 0.0), bo_(n * mmax, 0.0), orow_(n * mmax, 0.0),
        bom_(mmax + 1, 0.0), fullin_(mmax + 2, 0), bestin_(mmax + 2, 0), tempin_(mmax + 2, 0),
        termlen_(mmax + 1, 0), flag_((mmax + 1) * (p + 1), 0), cut_((mmax + 1) * (p + 1), 0.0),
        dir_((mmax + 1) * (p + 1), 0.0), covsy_((mmax + 1) * nclass, 0.0), ybar_(nclass, 0.0),
        res_(n * nclass, 0.0), beta_((mmax + 1) * nclass, 0.0), vard_(mmax + 1, 0.0) {
    std::vector<std::size_t> order(n);
    for (std::size_t v = 0; v < p; ++v) {
      std::iota(order.begin(), order.end(), std::size_t{0});
      const double* col = x + v * n;
      std::stable_sort(order.begin(), order.end(),
                       [col](std::size_t a, std::size_t b) { return col[a] < col[b]; });
      for (std::size_t k = 0; k < n; ++k) tagx_[k + v * n] = static_cast<int>(order[k]) + 1;
    }
  }

  void forward();
  void select(bool prune);

  int lenb() const { return lenb_; }
  int forward_terms() const {
    int c = 0;
    for (int t = 1; t <= nterms_; ++t) c += fullin_[t];
    return c;
  }
  double gcv() const { return bestgcv_; }
  bool selected(int t) const { return bestin_[t] == 1; }
  double dir(int t, int v) const { return dir_[t + v * (mmax_ + 1)]; }
  double cut(int t, int v) const { return cut_[t + v * (mmax_ + 1)]; }
  double beta(int j, int kc) const { return beta_[j + kc * (mmax_ + 1)]; }
  // R's `fitted.values`, the response less the residuals `marss` returns.
  double fitted(int i, int kc) const { return Y(i, kc) - res_[(i - 1) + kc * n_]; }

 private:
  double X(int i, int v) const { return x_[(i - 1) + (v - 1) * n_]; }
  double Y(int i, int kc) const { return y_[(i - 1) + kc * n_]; }
  int TAG(int k, int v) const { return tagx_[(k - 1) + (v - 1) * n_]; }
  double& BX(int i, int t) { return bx_[(i - 1) + (t - 1) * n_]; }
  double BX(int i, int t) const { return bx_[(i - 1) + (t - 1) * n_]; }
  double& BO(int i, int t) { return bo_[(i - 1) + (t - 1) * n_]; }
  int& FLAG(int t, int v) { return flag_[t + v * (mmax_ + 1)]; }
  int FLAG(int t, int v) const { return flag_[t + v * (mmax_ + 1)]; }
  double& CUT(int t, int v) { return cut_[t + v * (mmax_ + 1)]; }
  double& DIR(int t, int v) { return dir_[t + v * (mmax_ + 1)]; }
  double& COVSY(int t, int kc) { return covsy_[t + kc * (mmax_ + 1)]; }
  double COVSY(int t, int kc) const { return covsy_[t + kc * (mmax_ + 1)]; }
  double& BETA(int j, int kc) { return beta_[j + kc * (mmax_ + 1)]; }

  void sync_row(int t) {
    for (std::size_t i = 1; i <= n_; ++i) orow_[(t - 1) + (i - 1) * mmax_] = BO(i, t);
  }
  // `orthreg`: `src` less its projections on the orthogonal columns `1..cols` that `in` marks.
  void orthreg(int cols, const std::vector<int>& in, const double* src, double* res) const;
  Candidate scan(int m, int v, int minspan, int iendspan, double rss, double prevcrit) const;
  void addtrm(double rss, double prevcrit, Candidate& best, int& jmax, int& kmax) const;
  void qrreg(const std::vector<int>& in, bool cvar);

  const double* x_;
  std::size_t n_, p_;
  const double* y_;
  std::size_t nc_;
  int maxorder_;
  double penalty_;
  int mmax_;
  double thresh_;
  int threads_;
  double dn_;
  std::vector<int> tagx_;
  std::vector<double> bx_, bo_, orow_, bom_;
  std::vector<int> fullin_, bestin_, tempin_, termlen_, flag_;
  std::vector<double> cut_, dir_, covsy_, ybar_, res_, beta_, vard_;
  int nterms_ = 1, lenb_ = 1, qrank_ = 0;
  std::vector<int> qpivot_;
  double rss_ = 0.0, dofit_ = 0.0, bestgcv_ = 0.0;
};

void MdaMars::orthreg(int cols, const std::vector<int>& in, const double* src,
                      double* res) const {
  std::copy(src, src + n_, res);
  for (int j = 1; j <= cols; ++j) {
    if (in[j] != 1) continue;
    const double* xj = bo_.data() + (j - 1) * n_;
    double t1 = 0.0, t2 = 0.0;
    for (std::size_t i = 0; i < n_; ++i) {
      t1 = t1 + res[i] * xj[i];
      t2 = t2 + xj[i] * xj[i];
    }
    const double b = t1 / t2;
    for (std::size_t i = 0; i < n_; ++i) res[i] = res[i] - b * xj[i];
  }
}

Candidate MdaMars::scan(int m, int v, int minspan, int iendspan, double rss,
                        double prevcrit) const {
  Candidate c;
  if (FLAG(m, v) != 0) return c;
  const int n = static_cast<int>(n_);
  const int nterms = nterms_;

  // A column enters linearly only where no term in the model already carries it on the same
  // other columns as the parent.
  bool tnewform = true;
  int mm = 1;
  while (mm <= nterms && tnewform) {
    ++mm;
    if (tempin_[mm] != 1) continue;
    tnewform = false;
    if (FLAG(mm, v) != 1) {
      tnewform = true;
      continue;
    }
    for (int j = 1; j <= static_cast<int>(p_); ++j) {
      if (j != v && FLAG(mm, j) != FLAG(m, j)) {
        tnewform = true;
        break;
      }
    }
  }

  std::vector<double> lin(n_, 0.0), covsy_lin(nc_, 0.0);
  double linm = 0.0, critnew = 0.0;
  if (tnewform) {
    std::vector<double> scrat(n_);
    for (int i = 1; i <= n; ++i) scrat[i - 1] = X(i, v) * BX(i, m);
    if (nterms > 1) {
      orthreg(nterms, tempin_, scrat.data(), lin.data());
    } else {
      double tem = 0.0;
      for (int i = 0; i < n; ++i) tem = tem + scrat[i] / dn_;
      for (int i = 0; i < n; ++i) lin[i] = scrat[i] - tem;
    }
    for (int i = 0; i < n; ++i) linm = linm + lin[i] / dn_;
    double t1 = 0.0;
    for (int i = 0; i < n; ++i) t1 = t1 + lin[i] * lin[i];
    if (t1 > kTolBx) {
      const double s = std::sqrt(t1);
      for (int i = 0; i < n; ++i) lin[i] = lin[i] / s;
    } else {
      std::fill(lin.begin(), lin.end(), 0.0);
      tnewform = false;
    }
    for (std::size_t kc = 0; kc < nc_; ++kc) {
      double s = 0.0;
      for (int i = 1; i <= n; ++i) s = s + (Y(i, kc) - ybar_[kc]) * lin[i - 1];
      covsy_lin[kc] = s;
    }
    for (std::size_t kc = 0; kc < nc_; ++kc) {
      double t = 0.0;
      for (int i = 1; i <= n; ++i) t = t + Y(i, kc) * lin[i - 1];
      critnew = critnew + t * t;
    }
    if (critnew > c.crit) {
      c.crit = critnew;
      c.k = 1;
      c.newform = false;
    }
  }
  // `n2` terms the knot's hinge is taken against: the model's, and the column itself where it is
  // to enter linearly.
  const int n2 = tnewform ? nterms + 1 : nterms;
  if (!tnewform) critnew = 0.0;

  // The knot search below reads, for every sorted unit, one orthogonal row of the model's terms
  // and, where the column is to enter linearly, that column's entry after them: the row, the
  // column means and the terms' covariances with the response are laid out end to end once here,
  // in the order `addtrm` visits them.
  std::vector<double> om(n2), cs(static_cast<std::size_t>(n2) * nc_);
  for (int i = 0; i < nterms; ++i) om[i] = bom_[i + 1];
  if (n2 > nterms) om[nterms] = linm;
  for (std::size_t kc = 0; kc < nc_; ++kc) {
    for (int i = 0; i < nterms; ++i) cs[i + kc * n2] = COVSY(i + 1, kc);
    if (n2 > nterms) cs[nterms + kc * n2] = covsy_lin[kc];
  }
  std::vector<double> covcol(n2, 0.0), scr1(n2, 0.0), scr6(nc_, 0.0), covsy_new(nc_, 0.0);
  double cov21 = 0.0;
  double scr2 = 0.0, su = 0.0, st = 0.0, sumbx2 = 0.0, sumb = 0.0, sumbx = 0.0;
  int k0 = 0;
  for (int k = n - 1; k > 0; --k) {
    const int kk = TAG(k, v), kk1 = TAG(k + 1, v);
    const double b1 = BX(kk1, m);
    const double xk = X(kk, v), xk1 = X(kk1, v);
    const double dx = xk1 - xk;
    const double* r = orow_.data() + static_cast<std::size_t>(kk1 - 1) * mmax_;
    for (int i = 0; i < nterms; ++i) {
      scr1[i] = scr1[i] + (r[i] - om[i]) * b1;
      covcol[i] = covcol[i] + dx * scr1[i];
    }
    if (n2 > nterms) {
      scr1[nterms] = scr1[nterms] + (lin[kk1 - 1] - om[nterms]) * b1;
      covcol[nterms] = covcol[nterms] + dx * scr1[nterms];
    }
    scr2 = scr2 + (b1 * b1) * xk1;
    sumbx2 = sumbx2 + b1 * b1;
    sumb = sumb + b1;
    sumbx = sumbx + b1 * xk1;
    su = st;
    st = sumbx - sumb * xk;
    cov21 = cov21 + dx * (2 * scr2 - sumbx2 * (xk + xk1)) + ((su * su) - (st * st)) / dn_;
    double crittemp = critnew;
    for (std::size_t kc = 0; kc < nc_; ++kc) {
      scr6[kc] = scr6[kc] + (Y(kk1, kc) - ybar_[kc]) * b1;
      covsy_new[kc] = covsy_new[kc] + dx * scr6[kc];
      double t1 = covsy_new[kc];
      double t2 = cov21;
      const double* csk = cs.data() + kc * n2;
      for (int jk = 0; jk < n2; ++jk) {
        t1 = t1 - csk[jk] * covcol[jk];
        t2 = t2 - covcol[jk] * covcol[jk];
      }
      double critadd = 0.0;
      if (cov21 > 0 && t2 / cov21 > kTolBx) critadd = (t1 * t1) / t2;
      crittemp = crittemp + critadd;
      if (crittemp > kRssGuard * rss) crittemp = 0.0;
      if (crittemp > 2 * prevcrit) crittemp = 0.0;
    }
    if (k > 1) k0 = TAG(k - 1, v);
    if (crittemp > c.crit && k % minspan == 0 && k >= iendspan && k <= n - iendspan && b1 > 0 &&
        !(k > 1 && xk == X(k0, v))) {
      c.crit = crittemp;
      c.k = k;
      c.newform = tnewform;
    }
  }
  return c;
}

void MdaMars::addtrm(double rss, double prevcrit, Candidate& best, int& jmax, int& kmax) const {
  best = Candidate{};
  jmax = 0;
  kmax = 0;
  const int p = static_cast<int>(p_);
  std::vector<Candidate> found(p_);
  for (int m = 1; m <= nterms_; ++m) {
    int nm = 0;
    for (std::size_t i = 1; i <= n_; ++i) nm += BX(i, m) > 0 ? 1 : 0;
    double tem = -(1.0 / static_cast<double>(static_cast<long long>(p) * nm)) *
                 std::log(1.0 - 5e-2);
    const int minspan = static_cast<int>(-1.0 * (std::log(tem) / std::log(2.0)) / 2.5);
    tem = 5e-2 / p;
    const int iendspan = static_cast<int>(3.0 - std::log(tem) / std::log(2.0));
    if (termlen_[m] >= maxorder_) continue;
    detail::run_tasks(p_, threads_, [&](std::size_t j) {
      found[j] = scan(m, static_cast<int>(j) + 1, minspan, iendspan, rss, prevcrit);
    });
    for (int v = 1; v <= p; ++v) {
      if (found[v - 1].crit > best.crit) {
        best = found[v - 1];
        jmax = v;
        kmax = m;
      }
    }
  }
}

void MdaMars::forward() {
  const int n = static_cast<int>(n_);
  const int p = static_cast<int>(p_);
  double dofit = 0.0;
  double prevcrit = 10e9;
  fullin_[1] = 1;
  int nterms2 = 2;
  nterms_ = 1;
  for (int i = 1; i <= n; ++i) BO(i, 1) = 1.0 / std::sqrt(dn_);
  bom_[1] = 1 / std::sqrt(dn_);
  for (int i = 1; i <= n; ++i) BX(i, 1) = 1;
  sync_row(1);
  for (std::size_t kc = 0; kc < nc_; ++kc) {
    for (int i = 1; i <= n; ++i) ybar_[kc] = ybar_[kc] + Y(i, kc) / dn_;
  }
  double rssnull = 0.0;
  for (std::size_t kc = 0; kc < nc_; ++kc) {
    for (int i = 1; i <= n; ++i) rssnull = rssnull + (Y(i, kc) - ybar_[kc]) * (Y(i, kc) - ybar_[kc]);
  }
  double rss = rssnull;
  double cmm = (1 + dofit) + penalty_ * (.5 * dofit);
  const double gcvnull = (rssnull / dn_) / ((1.0 - cmm / dn_) * (1.0 - cmm / dn_));
  lenb_ = 1;
  int ii = 0;
  bool go = true;
  while (ii < mmax_ - 1 && rss / rssnull > thresh_ && go) {
    ii += 2;
    for (std::size_t kc = 0; kc < nc_; ++kc) {
      for (int j = 1; j <= nterms_; ++j) {
        double s = 0.0;
        for (int i = 1; i <= n; ++i) s = s + (Y(i, kc) - ybar_[kc]) * BO(i, j);
        COVSY(j, kc) = s;
      }
    }
    for (int t = 1; t <= mmax_; ++t) tempin_[t] = fullin_[t];
    Candidate best;
    int jmax = 0, kmax = 0;
    addtrm(rss, prevcrit, best, jmax, kmax);
    const bool pair = best.k > 1 && best.newform;
    double doftemp = dofit;
    doftemp = doftemp + 1;
    if (pair) doftemp = doftemp + 1;
    const double temprss = rss - best.crit;
    cmm = (1 + doftemp) + penalty_ * (.5 * doftemp);
    const double gcv = (temprss / dn_) / ((1.0 - cmm / dn_) * (1.0 - cmm / dn_));
    go = false;
    if (!(best.crit / rss > thresh_ && gcv / gcvnull < kStopFac)) continue;
    go = true;
    dofit = doftemp;
    rss = rss - best.crit;
    prevcrit = best.crit;
    for (int j = 1; j <= p; ++j) {
      FLAG(ii, j) = FLAG(kmax, j);
      FLAG(ii + 1, j) = FLAG(kmax, j);
      CUT(ii, j) = CUT(kmax, j);
      CUT(ii + 1, j) = CUT(kmax, j);
      DIR(ii, j) = DIR(kmax, j);
      DIR(ii + 1, j) = DIR(kmax, j);
    }
    termlen_[ii] = termlen_[kmax] + 1;
    termlen_[ii + 1] = termlen_[kmax] + 1;
    const double knot = X(TAG(best.k, jmax), jmax);
    fullin_[ii] = 1;
    if (pair) fullin_[ii + 1] = 1;
    FLAG(ii, jmax) = 1;
    FLAG(ii + 1, jmax) = 1;
    CUT(ii, jmax) = knot;
    CUT(ii + 1, jmax) = knot;
    DIR(ii, jmax) = 1;
    DIR(ii + 1, jmax) = -1;
    if (fullin_[ii + 1] == 0) termlen_[ii + 1] = maxorder_ + 1;
    for (int i = 1; i <= n; ++i) {
      if (X(i, jmax) - knot > 0) BX(i, ii) = BX(i, kmax) * (X(i, jmax) - knot);
      if (knot - X(i, jmax) >= 0) BX(i, ii + 1) = BX(i, kmax) * (knot - X(i, jmax));
    }
    if (nterms_ == 1) {
      double t = 0.0;
      for (int i = 1; i <= n; ++i) t = t + BX(i, 2) / dn_;
      for (int i = 1; i <= n; ++i) BO(i, 2) = BX(i, 2) - t;
    } else {
      orthreg(nterms_, fullin_, &BX(1, ii), &BO(1, nterms2));
    }
    if (fullin_[ii + 1] == 1) {
      orthreg(nterms_ + 1, fullin_, &BX(1, ii + 1), &BO(1, nterms2 + 1));
    } else {
      for (int i = 1; i <= n; ++i) BO(i, nterms2 + 1) = 0;
    }
    bom_[nterms2] = 0.0;
    bom_[nterms2 + 1] = 0.0;
    for (int i = 1; i <= n; ++i) {
      bom_[nterms2] = bom_[nterms2] + BO(i, nterms2) / dn_;
      bom_[nterms2 + 1] = bom_[nterms2 + 1] + BO(i, nterms2 + 1) / dn_;
    }
    double t1 = 0.0, t2 = 0.0;
    for (int i = 1; i <= n; ++i) {
      t1 = t1 + BO(i, nterms2) * BO(i, nterms2);
      t2 = t2 + BO(i, nterms2 + 1) * BO(i, nterms2 + 1);
    }
    if (t1 > 0.0) {
      const double s = std::sqrt(t1);
      for (int i = 1; i <= n; ++i) BO(i, nterms2) = BO(i, nterms2) / s;
    }
    if (t2 > 0.0) {
      const double s = std::sqrt(t2);
      for (int i = 1; i <= n; ++i) BO(i, nterms2 + 1) = BO(i, nterms2 + 1) / s;
    }
    sync_row(nterms2);
    sync_row(nterms2 + 1);
    lenb_ += 2;
    nterms_ += 2;
    nterms2 += 2;
  }
  rss_ = rss;
  dofit_ = dofit;
}

void MdaMars::qrreg(const std::vector<int>& in, bool cvar) {
  std::vector<int> cols;
  for (int j = 1; j <= lenb_; ++j) {
    if (in[j] == 1) cols.push_back(j);
  }
  const std::size_t nt = cols.size();
  std::vector<double> xsc(n_ * nt);
  for (std::size_t c = 0; c < nt; ++c) {
    std::copy(bx_.begin() + (cols[c] - 1) * n_, bx_.begin() + cols[c] * n_,
              xsc.begin() + c * n_);
  }
  std::size_t rank = 0;
  std::vector<double> qraux;
  std::vector<std::size_t> jpvt;
  detail::dqrdc2(xsc.data(), n_, nt, kQrTol, rank, qraux, jpvt);
  qrank_ = static_cast<int>(rank);
  qpivot_.assign(nt + 1, 0);
  for (std::size_t c = 0; c < nt; ++c) qpivot_[c + 1] = static_cast<int>(jpvt[c]) + 1;
  double rss = 0.0;
  std::vector<double> qty(n_), xb(n_);
  for (std::size_t kc = 0; kc < nc_; ++kc) {
    for (std::size_t i = 0; i < n_; ++i) qty[i] = y_[i + kc * n_];
    detail::qr_qty(xsc.data(), n_, rank, qraux.data(), qty.data());
    std::vector<double> b(qty.begin(), qty.begin() + rank);
    detail::qr_backsolve(xsc.data(), n_, rank, b.data());
    for (std::size_t j = 0; j < rank; ++j) BETA(static_cast<int>(j) + 1, kc) = b[j];
    std::fill(xb.begin(), xb.end(), 0.0);
    std::copy(qty.begin(), qty.begin() + rank, xb.begin());
    detail::qr_qy(xsc.data(), n_, rank, qraux.data(), xb.data());
    for (std::size_t i = 0; i < n_; ++i) {
      const double r = y_[i + kc * n_] - xb[i];
      res_[i + kc * n_] = r;
      rss = rss + r * r;
    }
  }
  rss_ = rss;
  if (!cvar) return;
  // `calcvar`: the diagonal of (R'R)^-1 over the first `qrank` pivoted columns, from R's inverse,
  // whose `j`th column is zero below its `j`th entry.
  std::vector<double> rinv(rank * rank, 0.0);
  for (std::size_t j = 0; j < rank; ++j) {
    double* col = rinv.data() + j * rank;
    col[j] = 1.0;
    detail::qr_backsolve(xsc.data(), n_, j + 1, col);
  }
  for (std::size_t i = 0; i < rank; ++i) {
    double s = 0.0;
    for (std::size_t k = i; k < rank; ++k) s = s + rinv[i + k * rank] * rinv[i + k * rank];
    vard_[i + 1] = s;
  }
}

void MdaMars::select(bool prune) {
  double dofit = -1;
  for (int t = 1; t <= nterms_; ++t) {
    bestin_[t] = fullin_[t];
    dofit = dofit + fullin_[t];
  }
  qrreg(bestin_, false);
  const int nt = static_cast<int>(dofit + 1);
  if (qrank_ < nt) {
    for (int i = qrank_ + 1; i <= nt; ++i) {
      bestin_[qpivot_[i]] = 0;
      fullin_[qpivot_[i]] = 0;
      dofit = dofit - 1;
    }
  }
  double cmm = (1 + dofit) + penalty_ * (.5 * dofit);
  bestgcv_ = (rss_ / dn_) / ((1.0 - cmm / dn_) * (1.0 - cmm / dn_));
  if (!prune) return;
  qrreg(tempin_, true);
  for (int t = 1; t <= mmax_; ++t) tempin_[t] = bestin_[t];
  while (dofit > 0) {
    int jo = 1;
    double rsstemp = 10e99;
    int minterm = 0;
    for (int t = 2; t <= lenb_; ++t) {
      if (tempin_[t] != 1) continue;
      ++jo;
      double temp7 = 0.0;
      for (std::size_t kc = 0; kc < nc_; ++kc) {
        temp7 = temp7 + (BETA(jo, kc) * BETA(jo, kc)) / vard_[jo];
      }
      if (temp7 < rsstemp) {
        minterm = t;
        rsstemp = temp7;
      }
    }
    if (minterm == 0) break;
    double rss = rss_ + rsstemp;
    dofit = dofit - 1;
    cmm = (1.0 + dofit) + penalty_ * (.5 * dofit);
    const double gcv = (rss / dn_) / ((1.0 - cmm / dn_) * (1.0 - cmm / dn_));
    tempin_[minterm] = 0;
    if (gcv < bestgcv_) {
      bestgcv_ = gcv;
      for (int t = 1; t <= mmax_; ++t) bestin_[t] = tempin_[t];
    }
    rss_ = rss;
    if (dofit > 0) qrreg(tempin_, true);
  }
  qrreg(bestin_, true);
}

double sq(double v) { return v * v; }

}  // namespace

Fda fda_fit(const double* x, const double* y, const double* w, std::size_t n, std::size_t p,
            const FdaSpec& spec) {
  if (n < 2) throw Error("a discriminant is fitted on two units or more.");
  if (p < 1) throw Error("a discriminant is fitted on one column or more.");
  if (spec.degree < 1) throw Error("a discriminant's `degree` is 1 or more.");
  if (!(spec.thresh >= 0.0 && spec.thresh < 1.0)) {
    throw Error("a discriminant's `thresh` is in [0, 1).");
  }
  detail::check_finite(x, n * p, "a discriminant", "design");
  detail::check_finite(w, n, "a discriminant", "weights");
  std::size_t count[2] = {0, 0};
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

  // `fda()`: the weights scaled to sum to the count of units, the classes' weighted shares, and
  // from them `contr.fda`'s scores, the second column of the Q of the shares' roots beside the
  // Helmert contrast, over the roots.
  double sw = 0.0;
  for (std::size_t i = 0; i < n; ++i) sw = sw + w[i];
  std::vector<double> ww(n);
  for (std::size_t i = 0; i < n; ++i) ww[i] = (dn * w[i]) / sw;
  double dp[2] = {0.0, 0.0};
  for (std::size_t i = 0; i < n; ++i) dp[y[i] == 1.0 ? 1 : 0] += ww[i];
  dp[0] = dp[0] / dn;
  dp[1] = dp[1] / dn;
  const double sp = dp[0] + dp[1];
  const double sqp[2] = {std::sqrt(dp[0] / sp), std::sqrt(dp[1] / sp)};
  double basis[4] = {1 * sqp[0], 1 * sqp[1], -1 * sqp[0], 1 * sqp[1]};
  std::size_t rank = 0;
  std::vector<double> qraux;
  std::vector<std::size_t> jpvt;
  detail::dqrdc2(basis, 2, 2, 1e-7, rank, qraux, jpvt);
  if (rank < 2) {
    out.discriminates = false;
    return out;
  }
  double e2[2] = {0.0, 1.0};
  detail::qr_qy(basis, 2, rank, qraux.data(), e2);
  const double theta[2] = {e2[0] / sqp[0], e2[1] / sqp[1]};
  std::vector<double> scored(n);
  for (std::size_t i = 0; i < n; ++i) scored[i] = theta[y[i] == 1.0 ? 1 : 0];

  MdaMars mars(x, n, p, scored.data(), 1, spec.degree, penalty, nk, spec.thresh,
               std::max(1, spec.threads));
  mars.forward();
  mars.select(spec.prune);
  out.forward_terms = mars.forward_terms();
  out.gcv = mars.gcv();

  // The kept terms, their factors in column order as `model.matrix.mars` multiplies them, and the
  // coefficients in the order `marss` leaves them.
  int kept = 0;
  out.factor_start.push_back(0);
  for (int t = 1; t <= mars.lenb(); ++t) {
    if (!mars.selected(t)) continue;
    ++kept;
    for (int v = 1; v <= static_cast<int>(p); ++v) {
      const double d = mars.dir(t, v);
      if (d == 0.0) continue;
      out.factor_column.push_back(v - 1);
      out.factor_dir.push_back(d > 0 ? 1 : -1);
      out.factor_cut.push_back(mars.cut(t, v));
    }
    out.factor_start.push_back(static_cast<std::int32_t>(out.factor_column.size()));
  }
  out.coef.resize(kept);
  for (int j = 0; j < kept; ++j) out.coef[j] = mars.beta(j + 1, 0);

  // The canonical variate: the weighted scores against the basis's fit to them, over the units.
  double ssm = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    ssm = ssm + mars.fitted(static_cast<int>(i) + 1, 0) * (scored[i] * ww[i]);
  }
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
  const std::size_t kept = fit.coef.size();
  for (std::size_t i = 0; i < n; ++i) {
    double f = 0.0;
    for (std::size_t t = 0; t < kept; ++t) {
      double b = 1.0;
      for (std::int32_t q = fit.factor_start[t]; q < fit.factor_start[t + 1]; ++q) {
        const double d = fit.factor_dir[q] * (x[i + fit.factor_column[q] * n] - fit.factor_cut[q]);
        b = b * d * (d > 0 ? 1.0 : 0.0);
      }
      f = f + fit.coef[t] * b;
    }
    const double z = (f * fit.direction) / fit.scale;
    const double d0 = sq(z - fit.centroid[0]), d1 = sq(z - fit.centroid[1]);
    const double dmin = std::min(d0, d1);
    const double p0 = std::exp(-0.5 * (d0 - dmin)) * fit.prior[0];
    const double p1 = std::exp(-0.5 * (d1 - dmin)) * fit.prior[1];
    const double post = p1 / (p0 + p1);
    out[i] = fit.calibrated
                 ? probit_linkinv(fit.calibration[0] * 1 + post * fit.calibration[1])
                 : post;
  }
}

}  // namespace timesift
