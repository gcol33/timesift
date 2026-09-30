#include "ts_glm.h"

#include "ts_normal.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <limits>

// The decomposition and the iteration are R's, operation for operation. Contraction is off for the
// reason `ts_tree.cpp` gives.
#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

namespace timesift {

namespace {

// R's logit link, from `family.c`: the linear predictor is held at 30 either side before the mean
// and its derivative are read, so neither reaches zero or one.
constexpr double kThresh = 30.0;
constexpr double kInvEps = 1.0 / DBL_EPSILON;

double logit_mu_eta(double eta) {
  const double opexp = 1.0 + std::exp(eta);
  return (eta > kThresh || eta < -kThresh) ? DBL_EPSILON : std::exp(eta) / (opexp * opexp);
}

double y_log_y(double y, double mu) { return y != 0.0 ? y * std::log(y / mu) : 0.0; }

double deviance(const double* y, const double* mu, const double* w, std::size_t n, Family f) {
  double d = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    if (f == Family::binomial) {
      d += 2.0 * w[i] * (y_log_y(y[i], mu[i]) + y_log_y(1.0 - y[i], 1.0 - mu[i]));
    } else {
      const double r = y[i] - mu[i];
      d += w[i] * r * r;
    }
  }
  return d;
}

double norm2_of(const double* v, std::size_t n) {
  double s = 0.0;
  for (std::size_t i = 0; i < n; ++i) s += v[i] * v[i];
  return std::sqrt(s);
}

}  // namespace

double logit_linkinv(double eta) {
  const double t = eta < -kThresh ? DBL_EPSILON : (eta > kThresh ? kInvEps : std::exp(eta));
  return t / (1.0 + t);
}

double probit_linkinv(double eta) {
  static const double thresh = -detail::qnorm(DBL_EPSILON);
  return detail::pnorm(std::min(std::max(eta, -thresh), thresh));
}

namespace {

// The link a fit iterates under: its inverse, its derivative, and the link itself, which only the
// starting means are read through.
struct LinkFns {
  bool binomial;
  Link link;
  double inv(double eta) const {
    if (!binomial) return eta;
    return link == Link::probit ? probit_linkinv(eta) : logit_linkinv(eta);
  }
  double mu_eta(double eta) const {
    if (!binomial) return 1.0;
    return link == Link::probit ? std::max(detail::dnorm(eta), DBL_EPSILON) : logit_mu_eta(eta);
  }
  double fun(double mu) const {
    if (!binomial) return mu;
    return link == Link::probit ? detail::qnorm(mu) : std::log(mu / (1.0 - mu));
  }
};

}  // namespace

namespace detail {

void dqrdc2(double* x, std::size_t n, std::size_t p, double tol, std::size_t& k,
            std::vector<double>& qraux, std::vector<std::size_t>& jpvt) {
  qraux.assign(p, 0.0);
  jpvt.resize(p);
  std::vector<double> work1(p), work2(p);
  for (std::size_t j = 0; j < p; ++j) {
    qraux[j] = norm2_of(x + j * n, n);
    work1[j] = qraux[j];
    work2[j] = qraux[j] == 0.0 ? 1.0 : qraux[j];
    jpvt[j] = j;
  }
  const std::size_t lup = std::min(n, p);
  std::size_t kk = p + 1;  // 1-based, as the Fortran
  for (std::size_t l = 0; l < lup; ++l) {
    while (!(l + 1 >= kk || qraux[l] >= work2[l] * tol)) {
      for (std::size_t i = 0; i < n; ++i) {
        const double t = x[i + l * n];
        for (std::size_t j = l + 1; j < p; ++j) x[i + (j - 1) * n] = x[i + j * n];
        x[i + (p - 1) * n] = t;
      }
      const std::size_t ji = jpvt[l];
      const double t = qraux[l], tt = work1[l], ttt = work2[l];
      for (std::size_t j = l + 1; j < p; ++j) {
        jpvt[j - 1] = jpvt[j];
        qraux[j - 1] = qraux[j];
        work1[j - 1] = work1[j];
        work2[j - 1] = work2[j];
      }
      jpvt[p - 1] = ji;
      qraux[p - 1] = t;
      work1[p - 1] = tt;
      work2[p - 1] = ttt;
      --kk;
    }
    if (l + 1 == n) continue;
    double* xl = x + l * n;
    double nrmxl = norm2_of(xl + l, n - l);
    if (nrmxl == 0.0) continue;
    if (xl[l] != 0.0) nrmxl = std::copysign(nrmxl, xl[l]);
    const double scale = 1.0 / nrmxl;
    for (std::size_t i = l; i < n; ++i) xl[i] *= scale;
    xl[l] = 1.0 + xl[l];
    for (std::size_t j = l + 1; j < p; ++j) {
      double* xj = x + j * n;
      double dot = 0.0;
      for (std::size_t i = l; i < n; ++i) dot += xl[i] * xj[i];
      const double t = -dot / xl[l];
      for (std::size_t i = l; i < n; ++i) xj[i] += t * xl[i];
      if (qraux[j] != 0.0) {
        const double r = std::abs(xj[l]) / qraux[j];
        const double tt = std::max(1.0 - r * r, 0.0);
        if (std::abs(tt) >= 1e-6) {
          qraux[j] *= std::sqrt(tt);
        } else {
          qraux[j] = norm2_of(xj + l + 1, n - l - 1);
          work1[j] = qraux[j];
        }
      }
    }
    qraux[l] = xl[l];
    xl[l] = -nrmxl;
  }
  k = std::min(kk - 1, n);
}

namespace {

// The `j`th Householder reflection of the decomposition applied to `y`. Its leading entry is
// `qraux[j]`, where the decomposition keeps R's diagonal.
void reflect(const double* qr, std::size_t n, std::size_t j, const double* qraux, double* y) {
  const double* xj = qr + j * n;
  double dot = qraux[j] * y[j];
  for (std::size_t i = j + 1; i < n; ++i) dot += xj[i] * y[i];
  const double t = -dot / qraux[j];
  y[j] += t * qraux[j];
  for (std::size_t i = j + 1; i < n; ++i) y[i] += t * xj[i];
}

}  // namespace

void qr_qty(const double* qr, std::size_t n, std::size_t k, const double* qraux, double* y) {
  const std::size_t ju = std::min(k, n - 1);
  for (std::size_t j = 0; j < ju; ++j) {
    if (qraux[j] != 0.0) reflect(qr, n, j, qraux, y);
  }
}

void qr_qy(const double* qr, std::size_t n, std::size_t k, const double* qraux, double* y) {
  const std::size_t ju = std::min(k, n - 1);
  for (std::size_t j = ju; j-- > 0;) {
    if (qraux[j] != 0.0) reflect(qr, n, j, qraux, y);
  }
}

bool qr_backsolve(const double* qr, std::size_t n, std::size_t k, double* b) {
  for (std::size_t j = k; j-- > 0;) {
    const double d = qr[j + j * n];
    if (d == 0.0) return false;
    b[j] /= d;
    for (std::size_t i = 0; i < j; ++i) b[i] -= b[j] * qr[i + j * n];
  }
  return true;
}

std::vector<double> dqrls(double* x, std::size_t n, std::size_t p, double* b, double tol,
                          std::size_t& rank) {
  std::vector<double> qraux;
  std::vector<std::size_t> jpvt;
  dqrdc2(x, n, p, tol, rank, qraux, jpvt);
  std::vector<double> coef(p, 0.0);
  if (rank > 0) {
    qr_qty(x, n, rank, qraux.data(), b);
    std::vector<double> s(b, b + rank);
    if (!qr_backsolve(x, n, rank, s.data())) {
      s.assign(rank, std::numeric_limits<double>::quiet_NaN());
    }
    for (std::size_t j = 0; j < rank; ++j) coef[jpvt[j]] = s[j];
  }
  return coef;
}

}  // namespace detail

Glm glm_fit(const double* x, std::size_t n, std::size_t q, const double* y, const double* w,
            Family family, double epsilon, int max_iter, Link link) {
  const bool binomial = family == Family::binomial;
  const LinkFns fns{binomial, link};
  std::vector<double> eta(n), mu(n), start(q, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    if (binomial) {
      const double m = (w[i] * y[i] + 0.5) / (w[i] + 1.0);
      eta[i] = fns.fun(m);
      mu[i] = fns.inv(eta[i]);
    } else {
      eta[i] = y[i];
      mu[i] = y[i];
    }
  }
  double devold = deviance(y, mu.data(), w, n, family);
  Glm out;
  out.beta.assign(q, 0.0);
  const double tol = std::min(1e-7, epsilon / 1000.0);
  std::vector<double> a, z;
  std::vector<std::size_t> good;
  for (int iter = 0; iter < max_iter; ++iter) {
    good.clear();
    for (std::size_t i = 0; i < n; ++i) {
      if (w[i] > 0.0 && fns.mu_eta(eta[i]) != 0.0) good.push_back(i);
    }
    if (good.empty()) return out;
    const std::size_t ng = good.size();
    a.assign(ng * q, 0.0);
    z.assign(ng, 0.0);
    for (std::size_t r = 0; r < ng; ++r) {
      const std::size_t i = good[r];
      const double me = fns.mu_eta(eta[i]);
      const double var = binomial ? mu[i] * (1.0 - mu[i]) : 1.0;
      const double ws = std::sqrt((w[i] * me * me) / var);
      z[r] = (eta[i] + (y[i] - mu[i]) / me) * ws;
      for (std::size_t c = 0; c < q; ++c) a[r + c * ng] = x[i + c * n] * ws;
    }
    std::size_t rank = 0;
    const std::vector<double> coef = detail::dqrls(a.data(), ng, q, z.data(), tol, rank);
    for (double c : coef) {
      if (!std::isfinite(c)) return out;
    }
    start = coef;
    for (std::size_t i = 0; i < n; ++i) {
      double e = 0.0;
      for (std::size_t c = 0; c < q; ++c) e += x[i + c * n] * start[c];
      eta[i] = e;
      mu[i] = fns.inv(e);
    }
    const double dev = deviance(y, mu.data(), w, n, family);
    out.beta = start;
    out.rank = static_cast<std::int32_t>(rank);
    out.deviance = dev;
    if (!std::isfinite(dev)) return out;
    if (std::abs(dev - devold) / (0.1 + std::abs(dev)) < epsilon) {
      out.converged = true;
      return out;
    }
    devold = dev;
  }
  return out;
}

}  // namespace timesift
