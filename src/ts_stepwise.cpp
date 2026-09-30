#include "ts_stepwise.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "ts_core.h"
#include "ts_glm.h"
#include "ts_internal.h"

namespace timesift {

namespace {

template <typename E>
struct NamedValue {
  const char* name;
  E value;
};

constexpr NamedValue<StepDirection> kDirections[] = {
    {"forward", StepDirection::forward},
    {"both", StepDirection::both},
    {"backward", StepDirection::backward},
    {"none", StepDirection::none},
};

constexpr NamedValue<StepTerms> kTermKinds[] = {
    {"column", StepTerms::column},
    {"power", StepTerms::power},
};

}  // namespace

StepDirection step_direction_from_name(const std::string& name) {
  for (const auto& d : kDirections) {
    if (name == d.name) return d.value;
  }
  throw Error("a stepwise search runs 'forward', 'both', 'backward' or 'none', not '" + name +
              "'.");
}

const char* step_direction_name(StepDirection d) {
  for (const auto& e : kDirections) {
    if (e.value == d) return e.name;
  }
  return "forward";
}

StepTerms step_terms_from_name(const std::string& name) {
  for (const auto& t : kTermKinds) {
    if (name == t.name) return t.value;
  }
  throw Error("a stepwise term is a 'column' or a 'power', not '" + name + "'.");
}

const char* step_terms_name(StepTerms t) {
  for (const auto& e : kTermKinds) {
    if (e.value == t) return e.name;
  }
  return "power";
}

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

// The polynomials orthogonal over the fitted values `v` of one column, by the three-term recurrence
// (Forsythe 1957; Kennedy & Gentle 1980, sec. 8.7):
//
//   P_0 = 1,  P_1 = (v - alpha_1) P_0,  P_k = (v - alpha_k) P_{k-1} - (N_{k-1} / N_{k-2}) P_{k-2},
//
// with `alpha_k = sum(v P_{k-1}^2) / N_{k-1}` and `N_k = sum(P_k^2)`, `N_0` the row count. A term
// enters as `P_k / sqrt(N_k)` for `k = 1..degree`, which is the basis R's `poly()` builds, and a
// prediction replays the recurrence at new values from the `alpha` and `N` of the fit.
class ThreeTermRecurrence {
 public:
  ThreeTermRecurrence(const double* v, std::size_t n)
      : v_(v), n_(n), older_(n, 0.0), last_(n, 1.0), next_(n) {}

  // `P_{k-1}`, the polynomial the next step starts from.
  const std::vector<double>& last() const { return last_; }

  // Steps to `P_k` given `alpha_k` and the ratio `N_{k-1} / N_{k-2}`, zero at the first step.
  void advance(double alpha, double ratio) {
    for (std::size_t i = 0; i < n_; ++i) next_[i] = (v_[i] - alpha) * last_[i] - ratio * older_[i];
    older_.swap(last_);
    last_.swap(next_);
  }

  // `P_k / sqrt(N_k)` into `out` [n].
  void write_normalised(double norm2, double* out) const {
    const double root = std::sqrt(norm2);
    for (std::size_t i = 0; i < n_; ++i) out[i] = last_[i] / root;
  }

 private:
  const double* v_;
  std::size_t n_;
  std::vector<double> older_, last_, next_;
};

// `N_{k-1} / N_{k-2}` from the squared norms `N_0..`, zero at `k = 1`.
double norm_ratio(const double* norm2, int k) {
  return k == 1 ? 0.0
                : norm2[static_cast<std::size_t>(k - 1)] / norm2[static_cast<std::size_t>(k - 2)];
}

// A term the search can hold: a column's orthogonal polynomial of some degree, or one power of a
// column, with the columns it enters the design as over the fitted rows.
struct Term {
  std::int32_t column = 0;
  std::int32_t power = 0;              // 0: the column's orthogonal polynomial
  std::int32_t degree = 1;             // design columns the term enters as
  std::vector<double> alpha, norm2;    // the recurrence of a polynomial term
  std::vector<double> design;          // [n, degree]
};

Term polynomial_term(const double* v, std::size_t n, std::int32_t column, int degree) {
  Term t;
  t.column = column;
  t.power = 0;
  t.degree = degree;
  t.design.resize(n * static_cast<std::size_t>(degree));
  t.norm2.push_back(static_cast<double>(n));
  ThreeTermRecurrence rec(v, n);
  for (int k = 1; k <= degree; ++k) {
    const std::vector<double>& p = rec.last();
    double moment = 0.0;
    for (std::size_t i = 0; i < n; ++i) moment += v[i] * p[i] * p[i];
    t.alpha.push_back(moment / t.norm2.back());
    rec.advance(t.alpha.back(), norm_ratio(t.norm2.data(), k));
    double sum_sq = 0.0;
    for (double c : rec.last()) sum_sq += c * c;
    t.norm2.push_back(sum_sq);
    rec.write_normalised(sum_sq, t.design.data() + static_cast<std::size_t>(k - 1) * n);
  }
  return t;
}

// The polynomial term of degree `degree` described by `alpha` [degree] and `norm2` [degree + 1],
// evaluated at `v` [n] into `out` [n, degree].
void evaluate_polynomial(const double* v, std::size_t n, int degree, const double* alpha,
                         const double* norm2, double* out) {
  ThreeTermRecurrence rec(v, n);
  for (int k = 1; k <= degree; ++k) {
    rec.advance(alpha[k - 1], norm_ratio(norm2, k));
    rec.write_normalised(norm2[k], out + static_cast<std::size_t>(k - 1) * n);
  }
}

// `v^k` as R's `^` evaluates it, which the fixtures pin: the square as a product, any other power
// through `pow`.
double raise(double v, int k) {
  if (k == 1) return v;
  if (k == 2) return v * v;
  return std::pow(v, static_cast<double>(k));
}

Term power_term(const double* v, std::size_t n, std::int32_t column, int power) {
  Term t;
  t.column = column;
  t.power = power;
  t.degree = 1;
  t.design.resize(n);
  for (std::size_t i = 0; i < n; ++i) t.design[i] = raise(v[i], power);
  return t;
}

std::size_t count_distinct(const double* v, std::size_t n) {
  std::vector<double> s(v, v + n);
  std::sort(s.begin(), s.end());
  return static_cast<std::size_t>(std::unique(s.begin(), s.end()) - s.begin());
}

// Every term the search may hold, in column order. A column holding one value is the intercept the
// model already carries and has no polynomial to enter as, so it gives no term; a polynomial's
// degree is capped at one less than the column's distinct values, the most they determine.
std::vector<Term> build_catalogue(const double* x, std::size_t n, std::size_t p,
                                  const StepwiseSpec& spec) {
  std::vector<Term> catalogue;
  for (std::size_t j = 0; j < p; ++j) {
    const double* v = x + j * n;
    const std::size_t distinct = count_distinct(v, n);
    if (distinct < 2) continue;
    const auto column = static_cast<std::int32_t>(j);
    if (spec.terms == StepTerms::column) {
      const int degree = std::min<int>(spec.degree, static_cast<int>(distinct) - 1);
      catalogue.push_back(polynomial_term(v, n, column, degree));
    } else {
      for (int k = 1; k <= spec.degree; ++k) catalogue.push_back(power_term(v, n, column, k));
    }
  }
  return catalogue;
}

// A model the search has fitted, scored by Akaike's criterion (Akaike 1974).
struct Scored {
  Glm glm;
  double aic = std::numeric_limits<double>::infinity();
  bool usable = false;   // the fit settled and its criterion is finite
};

// Fits and scores the models the search visits: an intercept and the design columns of the terms
// it names, in the order named.
class Scorer {
 public:
  Scorer(const double* y, const double* w, std::size_t n, const std::vector<Term>& catalogue,
         const StepwiseSpec& spec)
      : y_(y), w_(w), n_(n), catalogue_(catalogue), spec_(spec) {
    for (std::size_t i = 0; i < n; ++i) sum_log_w_ += std::log(w[i]);
  }

  Scored score(const std::vector<std::size_t>& terms) const {
    std::size_t q = 1;
    for (std::size_t t : terms) q += static_cast<std::size_t>(catalogue_[t].degree);
    std::vector<double> design(n_ * q);
    auto at = design.begin();
    at = std::fill_n(at, n_, 1.0);
    for (std::size_t t : terms) {
      at = std::copy(catalogue_[t].design.begin(), catalogue_[t].design.end(), at);
    }
    Scored out;
    out.glm = glm_fit(design.data(), n_, q, y_, w_, spec_.family, spec_.epsilon, spec_.max_iter);
    out.aic = criterion(out.glm);
    out.usable = out.glm.converged && std::isfinite(out.aic);
    return out;
  }

 private:
  // `-2 log L + 2 rank`. For a 0/1 response the saturated log-likelihood is zero, so the first
  // part is the deviance; for the gaussian family it is the likelihood at the maximum-likelihood
  // variance under the prior weights, whose estimate counts as one more parameter.
  double criterion(const Glm& g) const {
    const double penalty = 2.0 * static_cast<double>(g.rank);
    if (spec_.family == Family::binomial) return g.deviance + penalty;
    const double nd = static_cast<double>(n_);
    return nd * (std::log(g.deviance / nd * 2.0 * kPi) + 1.0) + 2.0 - sum_log_w_ + penalty;
  }

  const double* y_;
  const double* w_;
  std::size_t n_;
  const std::vector<Term>& catalogue_;
  const StepwiseSpec& spec_;
  double sum_log_w_ = 0.0;
};

// One step's alternatives to the model as it stands: dropping the term at a position of the model,
// or adding a catalogue term it lacks.
struct Move {
  bool drop;
  std::size_t index;
};

std::vector<std::size_t> apply_move(const std::vector<std::size_t>& model, const Move& move) {
  std::vector<std::size_t> out;
  out.reserve(model.size() + 1);
  for (std::size_t i = 0; i < model.size(); ++i) {
    if (!(move.drop && i == move.index)) out.push_back(model[i]);
  }
  if (!move.drop) out.push_back(move.index);
  return out;
}

// The move a step takes, as an index into `moves`, or `moves.size()` for none (Venables & Ripley,
// Modern Applied Statistics with S, 4th ed., sec. 6.8). A drop that leaves the rank where it was
// removes a term that contributed nothing, and the last such drop is taken before any criterion is
// compared. Otherwise the move with the lowest criterion is taken if it beats the model as it
// stands, the earliest move winning a tie; a move that leaves the rank unchanged is not a
// candidate, which is what keeps an aliased addition out.
std::size_t choose_move(const std::vector<Move>& moves, const std::vector<Scored>& scored,
                        const Scored& current) {
  const std::size_t none = moves.size();
  std::size_t free_drop = none;
  for (std::size_t i = 0; i < moves.size(); ++i) {
    if (moves[i].drop && scored[i].usable && scored[i].glm.rank == current.glm.rank) free_drop = i;
  }
  if (free_drop != none) return free_drop;
  std::size_t pick = none;
  double best = current.aic;
  for (std::size_t i = 0; i < moves.size(); ++i) {
    if (!scored[i].usable || scored[i].glm.rank == current.glm.rank) continue;
    if (scored[i].aic < best) {
      best = scored[i].aic;
      pick = i;
    }
  }
  return pick;
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

  const std::vector<Term> catalogue = build_catalogue(x, n, p, spec);
  const Scorer scorer(y, w, n, catalogue, spec);
  const bool can_drop = spec.direction == StepDirection::both ||
                        spec.direction == StepDirection::backward;
  const bool can_add = spec.direction == StepDirection::both ||
                       spec.direction == StepDirection::forward;

  // A forward search starts from the intercept alone, a backward one or none from every term.
  std::vector<std::size_t> model;
  if (spec.direction == StepDirection::backward || spec.direction == StepDirection::none) {
    for (std::size_t t = 0; t < catalogue.size(); ++t) model.push_back(t);
  }
  Scored current = scorer.score(model);
  std::int32_t steps = 0;

  std::vector<Move> moves;
  std::vector<char> held(catalogue.size());
  while (spec.direction != StepDirection::none) {
    moves.clear();
    if (can_drop) {
      for (std::size_t i = 0; i < model.size(); ++i) moves.push_back({true, i});
    }
    if (can_add && static_cast<double>(model.size()) < spec.max_terms) {
      std::fill(held.begin(), held.end(), 0);
      for (std::size_t t : model) held[t] = 1;
      for (std::size_t t = 0; t < catalogue.size(); ++t) {
        if (!held[t]) moves.push_back({false, t});
      }
    }
    std::vector<Scored> scored(moves.size());
    detail::run_tasks(moves.size(), spec.threads, [&](std::size_t i) {
      scored[i] = scorer.score(apply_move(model, moves[i]));
    });

    const std::size_t pick = choose_move(moves, scored, current);
    if (pick == moves.size()) break;
    model = apply_move(model, moves[pick]);
    current = std::move(scored[pick]);
    ++steps;
  }

  Stepwise out;
  out.family = spec.family;
  out.n_column = static_cast<std::int32_t>(p);
  double sum_y = 0.0;
  for (std::size_t i = 0; i < n; ++i) sum_y += y[i];
  out.constant = sum_y / static_cast<double>(n);
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
    std::fill(out, out + n, fit.constant);
    return;
  }
  // The linear predictor from the intercept, adding each term's design columns in the order the
  // fit holds them; the polynomial terms read their recurrences off `alpha` and `norm2` in turn.
  std::vector<double> eta(n, fit.beta[0]);
  std::vector<double> design;
  const double* beta = fit.beta.data() + 1;
  const double* alpha = fit.alpha.data();
  const double* norm2 = fit.norm2.data();
  for (std::size_t t = 0; t < fit.term_column.size(); ++t) {
    const double* v = x + static_cast<std::size_t>(fit.term_column[t]) * n;
    const int degree = fit.term_degree[t];
    design.resize(n * static_cast<std::size_t>(degree));
    if (fit.term_power[t] == 0) {
      evaluate_polynomial(v, n, degree, alpha, norm2, design.data());
      alpha += degree;
      norm2 += degree + 1;
    } else {
      for (std::size_t i = 0; i < n; ++i) design[i] = raise(v[i], fit.term_power[t]);
    }
    for (int k = 0; k < degree; ++k) {
      const double b = *beta++;
      const double* col = design.data() + static_cast<std::size_t>(k) * n;
      for (std::size_t i = 0; i < n; ++i) eta[i] += b * col[i];
    }
  }
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = fit.family == Family::binomial ? logit_linkinv(eta[i]) : eta[i];
  }
}

}  // namespace timesift
