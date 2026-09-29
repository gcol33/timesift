#include "ts_stepwise.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <limits>
#include <set>

#include "ts_core.h"
#include "ts_internal.h"

namespace timesift {

StepDirection step_direction_from_name(const std::string& name) {
  if (name == "forward") return StepDirection::forward;
  if (name == "both") return StepDirection::both;
  if (name == "backward") return StepDirection::backward;
  if (name == "none") return StepDirection::none;
  throw Error("a stepwise search runs 'forward', 'both', 'backward' or 'none', not '" + name +
              "'.");
}

const char* step_direction_name(StepDirection d) {
  switch (d) {
    case StepDirection::forward: return "forward";
    case StepDirection::both: return "both";
    case StepDirection::backward: return "backward";
    case StepDirection::none: return "none";
  }
  return "forward";
}

StepTerms step_terms_from_name(const std::string& name) {
  if (name == "column") return StepTerms::column;
  if (name == "power") return StepTerms::power;
  throw Error("a stepwise term is a 'column' or a 'power', not '" + name + "'.");
}

const char* step_terms_name(StepTerms t) {
  return t == StepTerms::column ? "column" : "power";
}

namespace {

// R's logit link, from `family.c`: the linear predictor is held at 30 either side before the mean
// and its derivative are read, so neither reaches zero or one.
constexpr double kThresh = 30.0;
constexpr double kInvEps = 1.0 / DBL_EPSILON;
constexpr double kPi = 3.141592653589793238462643383279502884;

double logit_linkinv(double eta) {
  const double t = eta < -kThresh ? DBL_EPSILON : (eta > kThresh ? kInvEps : std::exp(eta));
  return t / (1.0 + t);
}

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

// LINPACK's dqrdc2, as R carries it: Householder QR of `x` [n, p] in place, a column whose norm has
// fallen below `tol` times its original norm moved to the end rather than pivoted by size. `k` is
// the rank, `jpvt` the original column at each position.
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

// LINPACK's dqrls: the least-squares coefficients of `b` on `x` [n, p], both overwritten, returned
// in the original column order with an aliased column's set to zero.
std::vector<double> dqrls(double* x, std::size_t n, std::size_t p, double* b, double tol,
                          std::size_t& rank) {
  std::vector<double> qraux;
  std::vector<std::size_t> jpvt;
  dqrdc2(x, n, p, tol, rank, qraux, jpvt);
  std::vector<double> coef(p, 0.0);
  if (rank > 0) {
    const std::size_t ju = std::min(rank, n - 1);
    for (std::size_t j = 0; j < ju; ++j) {
      if (qraux[j] == 0.0) continue;
      double* xj = x + j * n;
      const double diag = xj[j];
      xj[j] = qraux[j];
      double dot = 0.0;
      for (std::size_t i = j; i < n; ++i) dot += xj[i] * b[i];
      const double t = -dot / xj[j];
      for (std::size_t i = j; i < n; ++i) b[i] += t * xj[i];
      xj[j] = diag;
    }
    std::vector<double> s(b, b + rank);
    for (std::size_t jj = rank; jj-- > 0;) {
      const double d = x[jj + jj * n];
      if (d == 0.0) {
        s[jj] = std::numeric_limits<double>::quiet_NaN();
        continue;
      }
      s[jj] /= d;
      for (std::size_t i = 0; i < jj; ++i) s[i] -= s[jj] * x[i + jj * n];
    }
    for (std::size_t j = 0; j < rank; ++j) coef[jpvt[j]] = s[j];
  }
  return coef;
}

}  // namespace

Glm glm_fit(const double* x, std::size_t n, std::size_t q, const double* y, const double* w,
            Family family, double epsilon, int max_iter) {
  const bool binomial = family == Family::binomial;
  std::vector<double> eta(n), mu(n), start(q, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    if (binomial) {
      const double m = (w[i] * y[i] + 0.5) / (w[i] + 1.0);
      eta[i] = std::log(m / (1.0 - m));
      mu[i] = logit_linkinv(eta[i]);
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
      const double me = binomial ? logit_mu_eta(eta[i]) : 1.0;
      if (w[i] > 0.0 && me != 0.0) good.push_back(i);
    }
    if (good.empty()) return out;
    const std::size_t ng = good.size();
    a.assign(ng * q, 0.0);
    z.assign(ng, 0.0);
    for (std::size_t r = 0; r < ng; ++r) {
      const std::size_t i = good[r];
      const double me = binomial ? logit_mu_eta(eta[i]) : 1.0;
      const double var = binomial ? mu[i] * (1.0 - mu[i]) : 1.0;
      const double ws = std::sqrt((w[i] * me * me) / var);
      z[r] = (eta[i] + (y[i] - mu[i]) / me) * ws;
      for (std::size_t c = 0; c < q; ++c) a[r + c * ng] = x[i + c * n] * ws;
    }
    std::size_t rank = 0;
    const std::vector<double> coef = dqrls(a.data(), ng, q, z.data(), tol, rank);
    for (double c : coef) {
      if (!std::isfinite(c)) return out;
    }
    start = coef;
    for (std::size_t i = 0; i < n; ++i) {
      double e = 0.0;
      for (std::size_t c = 0; c < q; ++c) e += x[i + c * n] * start[c];
      eta[i] = e;
      mu[i] = binomial ? logit_linkinv(e) : e;
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

namespace {

// One term of the catalogue a search draws from, with its columns over the rows fitted.
struct Term {
  std::int32_t column = 0;
  std::int32_t power = 0;   // 0: the column's orthogonal polynomial
  std::int32_t degree = 1;
  std::vector<double> alpha, norm2;
  std::vector<double> values;  // [n, degree]
};

// The orthogonal polynomial's three-term recurrence, the basis R's `poly()` builds:
// `P_k = (v - alpha_k) P_{k-1} - (norm2_{k-1} / norm2_{k-2}) P_{k-2}` from `P_0 = 1`, with `alpha`
// and the squared norms `norm2` (the first is the row count) read off the rows fitted, and each
// column of the basis the polynomial over its norm.
class Recurrence {
 public:
  Recurrence(const double* v, std::size_t n) : v_(v), n_(n), prev_(n, 0.0), cur_(n, 1.0), next_(n) {}

  // `P_{k-1}`, the polynomial the next step starts from.
  const std::vector<double>& current() const { return cur_; }

  void step(double alpha, double ratio) {
    for (std::size_t i = 0; i < n_; ++i) next_[i] = (v_[i] - alpha) * cur_[i] - ratio * prev_[i];
    prev_.swap(cur_);
    cur_.swap(next_);
  }

  void write(double norm2, double* out) const {
    const double s = std::sqrt(norm2);
    for (std::size_t i = 0; i < n_; ++i) out[i] = cur_[i] / s;
  }

 private:
  const double* v_;
  std::size_t n_;
  std::vector<double> prev_, cur_, next_;
};

double recurrence_ratio(const std::vector<double>& norm2, int k) {
  return k == 1 ? 0.0
                : norm2[static_cast<std::size_t>(k - 1)] / norm2[static_cast<std::size_t>(k - 2)];
}

void poly_values(const double* v, std::size_t n, int degree, const std::vector<double>& alpha,
                 const std::vector<double>& norm2, double* out) {
  Recurrence r(v, n);
  for (int k = 1; k <= degree; ++k) {
    r.step(alpha[static_cast<std::size_t>(k - 1)], recurrence_ratio(norm2, k));
    r.write(norm2[static_cast<std::size_t>(k)], out + static_cast<std::size_t>(k - 1) * n);
  }
}

Term poly_term(const double* v, std::size_t n, std::int32_t column, int degree) {
  Term t;
  t.column = column;
  t.power = 0;
  t.degree = degree;
  t.values.resize(n * static_cast<std::size_t>(degree));
  t.norm2.push_back(static_cast<double>(n));
  Recurrence r(v, n);
  for (int k = 1; k <= degree; ++k) {
    const std::vector<double>& last = r.current();
    double num = 0.0;
    for (std::size_t i = 0; i < n; ++i) num += v[i] * last[i] * last[i];
    t.alpha.push_back(num / t.norm2.back());
    r.step(t.alpha.back(), recurrence_ratio(t.norm2, k));
    double s = 0.0;
    for (double c : r.current()) s += c * c;
    t.norm2.push_back(s);
    r.write(s, t.values.data() + static_cast<std::size_t>(k - 1) * n);
  }
  return t;
}

// A power as R's `^` takes it: the square as a product, the rest through `pow`.
double power_of(double v, int k) {
  if (k == 1) return v;
  if (k == 2) return v * v;
  return std::pow(v, static_cast<double>(k));
}

Term power_term(const double* v, std::size_t n, std::int32_t column, int power) {
  Term t;
  t.column = column;
  t.power = power;
  t.degree = 1;
  t.values.resize(n);
  for (std::size_t i = 0; i < n; ++i) t.values[i] = power_of(v[i], power);
  return t;
}

std::size_t distinct(const double* v, std::size_t n) {
  std::vector<double> s(v, v + n);
  std::sort(s.begin(), s.end());
  return static_cast<std::size_t>(std::unique(s.begin(), s.end()) - s.begin());
}

struct Candidate {
  Glm glm;
  double aic = std::numeric_limits<double>::infinity();
  bool ok = false;
};

class Search {
 public:
  Search(const double* y, const double* w, std::size_t n, const std::vector<Term>& catalogue,
         const StepwiseSpec& spec)
      : y_(y), w_(w), n_(n), catalogue_(catalogue), spec_(spec) {
    for (std::size_t i = 0; i < n; ++i) sum_log_w_ += std::log(w[i]);
  }

  Candidate fit(const std::vector<std::size_t>& terms) const {
    std::size_t q = 1;
    for (std::size_t t : terms) q += static_cast<std::size_t>(catalogue_[t].degree);
    std::vector<double> x(n_ * q);
    std::fill(x.begin(), x.begin() + static_cast<std::ptrdiff_t>(n_), 1.0);
    std::size_t c = 1;
    for (std::size_t t : terms) {
      const Term& term = catalogue_[t];
      std::copy(term.values.begin(), term.values.end(), x.begin() + static_cast<std::ptrdiff_t>(c * n_));
      c += static_cast<std::size_t>(term.degree);
    }
    Candidate out;
    out.glm = glm_fit(x.data(), n_, q, y_, w_, spec_.family, spec_.epsilon, spec_.max_iter);
    out.aic = criterion(out.glm);
    out.ok = out.glm.converged && std::isfinite(out.aic);
    return out;
  }

 private:
  // Akaike's criterion as R's families report it: for a 0/1 response the weighted deviance plus
  // twice the rank, and for the Gaussian family its log-likelihood under the prior weights.
  double criterion(const Glm& g) const {
    const double k = 2.0 * static_cast<double>(g.rank);
    if (spec_.family == Family::binomial) return g.deviance + k;
    const double nd = static_cast<double>(n_);
    return nd * (std::log(g.deviance / nd * 2.0 * kPi) + 1.0) + 2.0 - sum_log_w_ + k;
  }

  const double* y_;
  const double* w_;
  std::size_t n_;
  const std::vector<Term>& catalogue_;
  const StepwiseSpec& spec_;
  double sum_log_w_ = 0.0;
};

std::vector<std::size_t> without(const std::vector<std::size_t>& model, std::size_t at) {
  std::vector<std::size_t> out;
  out.reserve(model.size());
  for (std::size_t i = 0; i < model.size(); ++i) {
    if (i != at) out.push_back(model[i]);
  }
  return out;
}

}  // namespace

Stepwise stepwise_fit(const double* x, const double* y, const double* w, std::size_t n,
                      std::size_t p, const StepwiseSpec& spec) {
  if (spec.degree < 1) throw Error("a stepwise term enters at degree one or more.");
  if (!(spec.max_terms >= 0.0)) throw Error("a stepwise search holds zero terms or more.");
  if (n < 2) throw Error("a stepwise fit needs at least two rows.");
  detail::check_finite(x, n * p, "a stepwise fit", "design");
  detail::check_finite(y, n, "a stepwise fit", "response");
  detail::check_finite(w, n, "a stepwise fit", "weights");
  for (std::size_t i = 0; i < n; ++i) {
    if (!(w[i] > 0.0)) throw Error("a stepwise fit reads positive weights.");
    if (spec.family == Family::binomial && y[i] != 0.0 && y[i] != 1.0) {
      throw Error("a binomial stepwise fit reads a response of zero and one.");
    }
  }

  // A column holding one value has no polynomial to enter as, and is the intercept the model
  // already carries, so it is not a term.
  std::vector<Term> catalogue;
  for (std::size_t j = 0; j < p; ++j) {
    const double* v = x + j * n;
    const std::size_t u = distinct(v, n);
    if (u < 2) continue;
    const auto col = static_cast<std::int32_t>(j);
    if (spec.terms == StepTerms::column) {
      catalogue.push_back(poly_term(v, n, col, std::min<int>(spec.degree, static_cast<int>(u) - 1)));
    } else {
      for (int k = 1; k <= spec.degree; ++k) catalogue.push_back(power_term(v, n, col, k));
    }
  }

  const Search search(y, w, n, catalogue, spec);
  const bool backward = spec.direction == StepDirection::both ||
                        spec.direction == StepDirection::backward;
  const bool forward = spec.direction == StepDirection::both ||
                       spec.direction == StepDirection::forward;

  std::vector<std::size_t> model;
  if (spec.direction == StepDirection::backward || spec.direction == StepDirection::none) {
    for (std::size_t t = 0; t < catalogue.size(); ++t) model.push_back(t);
  }
  Candidate current = search.fit(model);
  std::int32_t steps = 0;

  while (spec.direction != StepDirection::none) {
    std::set<std::size_t> held(model.begin(), model.end());
    std::vector<std::size_t> adds;
    if (forward && static_cast<double>(model.size()) < spec.max_terms) {
      for (std::size_t t = 0; t < catalogue.size(); ++t) {
        if (!held.count(t)) adds.push_back(t);
      }
    }
    const std::size_t n_drop = backward ? model.size() : 0;
    std::vector<Candidate> moves(n_drop + adds.size());
    detail::run_tasks(moves.size(), spec.threads, [&](std::size_t i) {
      if (i < n_drop) {
        moves[i] = search.fit(without(model, i));
      } else {
        std::vector<std::size_t> grown = model;
        grown.push_back(adds[i - n_drop]);
        moves[i] = search.fit(grown);
      }
    });

    // A term whose removal leaves the rank where it was changes nothing but the count, and
    // stepAIC drops the last such term before comparing anything.
    std::size_t zero_df = n_drop;
    for (std::size_t i = 0; i < n_drop; ++i) {
      if (moves[i].ok && moves[i].glm.rank == current.glm.rank) zero_df = i;
    }
    std::size_t pick = moves.size();
    if (zero_df < n_drop) {
      pick = zero_df;
    } else {
      double best = current.aic;
      for (std::size_t i = 0; i < moves.size(); ++i) {
        if (!moves[i].ok || moves[i].glm.rank == current.glm.rank) continue;
        if (moves[i].aic < best) {
          best = moves[i].aic;
          pick = i;
        }
      }
    }
    if (pick == moves.size()) break;
    if (pick < n_drop) {
      model = without(model, pick);
    } else {
      model.push_back(adds[pick - n_drop]);
    }
    current = moves[pick];
    ++steps;
  }

  Stepwise out;
  out.family = spec.family;
  out.n_column = static_cast<std::int32_t>(p);
  double mean = 0.0;
  for (std::size_t i = 0; i < n; ++i) mean += y[i];
  out.constant = mean / static_cast<double>(n);
  for (std::size_t t : model) {
    const Term& term = catalogue[t];
    out.term_column.push_back(term.column);
    out.term_power.push_back(term.power);
    out.term_degree.push_back(term.degree);
    out.alpha.insert(out.alpha.end(), term.alpha.begin(), term.alpha.end());
    out.norm2.insert(out.norm2.end(), term.norm2.begin(), term.norm2.end());
  }
  out.beta = current.glm.beta;
  out.rank = current.glm.rank;
  out.deviance = current.glm.deviance;
  out.aic = current.aic;
  out.converged = current.glm.converged;
  out.steps = steps;
  return out;
}

void stepwise_predict(const Stepwise& fit, const double* x, std::size_t n, std::size_t p,
                      double* out) {
  if (p != static_cast<std::size_t>(fit.n_column)) {
    throw Error("the stepwise fit read " + std::to_string(fit.n_column) +
                " columns and is asked to predict over " + std::to_string(p) + ".");
  }
  detail::check_finite(x, n * p, "a stepwise fit", "design");
  if (fit.term_column.empty()) {
    for (std::size_t i = 0; i < n; ++i) out[i] = fit.constant;
    return;
  }
  std::vector<double> eta(n, fit.beta[0]);
  std::vector<double> basis;
  std::size_t b = 1, a = 0, m = 0;
  for (std::size_t t = 0; t < fit.term_column.size(); ++t) {
    const double* v = x + static_cast<std::size_t>(fit.term_column[t]) * n;
    const int degree = fit.term_degree[t];
    basis.assign(n * static_cast<std::size_t>(degree), 0.0);
    if (fit.term_power[t] == 0) {
      const std::vector<double> alpha(fit.alpha.begin() + static_cast<std::ptrdiff_t>(a),
                                      fit.alpha.begin() + static_cast<std::ptrdiff_t>(a + degree));
      const std::vector<double> norm2(fit.norm2.begin() + static_cast<std::ptrdiff_t>(m),
                                      fit.norm2.begin() + static_cast<std::ptrdiff_t>(m + degree + 1));
      poly_values(v, n, degree, alpha, norm2, basis.data());
      a += static_cast<std::size_t>(degree);
      m += static_cast<std::size_t>(degree) + 1;
    } else {
      for (std::size_t i = 0; i < n; ++i) basis[i] = power_of(v[i], fit.term_power[t]);
    }
    for (int k = 0; k < degree; ++k) {
      const double beta = fit.beta[b++];
      for (std::size_t i = 0; i < n; ++i) eta[i] += beta * basis[i + static_cast<std::size_t>(k) * n];
    }
  }
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = fit.family == Family::binomial ? logit_linkinv(eta[i]) : eta[i];
  }
}

}  // namespace timesift
