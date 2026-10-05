#include "ts_maxnet.h"

#include "ts_internal.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iterator>

namespace timesift {
namespace {

// The presence-background likelihood: a background row weighs this much against a presence, the
// fitted probability of a row is held at least this far from zero and one, and the penalty path
// runs this many points over this many decades down to the Phillips-Dudik tolerance.
constexpr double kBackgroundWeight = 100.0;
constexpr double kProbabilityFloor = 1e-8;
constexpr int kPathPoints = 200;
constexpr double kPathDecades = 4.0;

// ---------------------------------------------------------------------------------------------
// Numerical helpers

// `count` points from `from` to `to`: the two ends exactly, and each interior point as `from` plus
// a whole number of equal steps, so no rounding accumulates along the grid.
std::vector<double> evenly_spaced(double from, double to, int count) {
  std::vector<double> grid(static_cast<std::size_t>(count), from);
  if (count == 1 || from == to) return grid;
  const double step = (to - from) / static_cast<double>(count - 1);
  for (int i = 1; i + 1 < count; ++i) grid[static_cast<std::size_t>(i)] = from + i * step;
  grid.back() = to;
  return grid;
}

// A piecewise linear curve through strictly increasing breakpoints, held flat beyond the ends.
struct Curve {
  const double* at;
  const double* value;
  int count;

  double operator()(double v) const {
    if (v < at[0]) return value[0];
    if (v > at[count - 1]) return value[count - 1];
    const int right = static_cast<int>(std::upper_bound(at, at + count, v) - at);
    const int left = right - 1;
    if (v == at[left]) return value[left];
    return value[left] +
           (value[right] - value[left]) * ((v - at[left]) / (at[right] - at[left]));
  }
};

// The sample standard deviation of `col` over `rows`, its mean corrected by a second pass over the
// residuals before the squares are summed.
double spread_over(const double* col, const std::vector<std::size_t>& rows) {
  const double count = static_cast<double>(rows.size());
  double total = 0.0;
  for (std::size_t i : rows) total += col[i];
  double centre = total / count;
  double drift = 0.0;
  for (std::size_t i : rows) drift += col[i] - centre;
  centre += drift / count;
  double squares = 0.0;
  for (std::size_t i : rows) {
    const double d = col[i] - centre;
    squares += d * d;
  }
  return std::sqrt(squares / (count - 1.0));
}

std::string two_digits(double v) {
  char buffer[64];
  std::snprintf(buffer, sizeof buffer, "%.2g", v);
  return buffer;
}

// ---------------------------------------------------------------------------------------------
// Features

MaxnetKind kind_of(const MaxnetFeatures& f, std::size_t k) {
  return static_cast<MaxnetKind>(f.kind[k]);
}

void append(MaxnetFeatures& f, MaxnetKind kind, std::int32_t a, std::int32_t b, double lo,
            double hi) {
  f.kind.push_back(static_cast<std::int8_t>(kind));
  f.a.push_back(a);
  f.b.push_back(b);
  f.lo.push_back(lo);
  f.hi.push_back(hi);
}

void append_copy(MaxnetFeatures& to, const MaxnetFeatures& from, std::size_t k) {
  append(to, kind_of(from, k), from.a[k], from.b[k], from.lo[k], from.hi[k]);
}

// A feature's value from the readings of its first column and, for a product, its second.
double evaluate(MaxnetKind kind, double lo, double hi, double first, double second) {
  switch (kind) {
    case MaxnetKind::linear:
      return first;
    case MaxnetKind::quadratic:
      return first * first;
    case MaxnetKind::hinge:
      return std::min(1.0, std::max(0.0, (first - lo) / (hi - lo)));
    case MaxnetKind::threshold:
      return first >= lo ? 1.0 : 0.0;
    case MaxnetKind::product:
      return first * second;
  }
  throw Error("a maxnet feature of a kind the core does not know.");
}

// Feature `k` at row `i` of the column-major block `x` with `n` rows.
double evaluate_at(const MaxnetFeatures& f, std::size_t k, const double* x, std::size_t n,
                   std::size_t i) {
  const double first = x[i + static_cast<std::size_t>(f.a[k]) * n];
  const double second = f.b[k] >= 0 ? x[i + static_cast<std::size_t>(f.b[k]) * n] : 0.0;
  return evaluate(kind_of(f, k), f.lo[k], f.hi[k], first, second);
}

// The range of every column, and the columns whose range is not a single point.
struct ColumnRanges {
  std::vector<double> low, high;
  std::vector<std::int32_t> varying;
};

ColumnRanges column_ranges(const double* x, std::size_t n, std::size_t p) {
  ColumnRanges r;
  r.low.resize(p);
  r.high.resize(p);
  for (std::size_t j = 0; j < p; ++j) {
    const double* col = x + j * n;
    double lo = col[0], hi = col[0];
    for (std::size_t i = 0; i < n; ++i) {
      if (!std::isfinite(col[i])) throw Error("maxnet takes finite values alone.");
      lo = std::min(lo, col[i]);
      hi = std::max(hi, col[i]);
    }
    r.low[j] = lo;
    r.high[j] = hi;
    if (hi > lo) r.varying.push_back(static_cast<std::int32_t>(j));
  }
  return r;
}

using ClassBuilder = void (*)(MaxnetFeatures&, const ColumnRanges&, int knots);

void build_linear(MaxnetFeatures& f, const ColumnRanges& r, int) {
  for (std::int32_t j : r.varying) append(f, MaxnetKind::linear, j, -1, 0.0, 0.0);
}

void build_quadratic(MaxnetFeatures& f, const ColumnRanges& r, int) {
  for (std::int32_t j : r.varying) append(f, MaxnetKind::quadratic, j, -1, 0.0, 0.0);
}

// A forward hinge rises from each knot but the last to the column's maximum, a reverse hinge from
// the column's minimum to each knot but the first.
void build_hinges(MaxnetFeatures& f, const ColumnRanges& r, int knots) {
  for (std::int32_t j : r.varying) {
    const std::vector<double> knot = evenly_spaced(r.low[j], r.high[j], knots);
    for (std::size_t i = 0; i + 1 < knot.size(); ++i) {
      append(f, MaxnetKind::hinge, j, -1, knot[i], r.high[j]);
    }
    for (std::size_t i = 1; i < knot.size(); ++i) {
      append(f, MaxnetKind::hinge, j, -1, r.low[j], knot[i]);
    }
  }
}

// Thresholds at the points of a `knots + 2` grid over the range that lie strictly inside it, less
// the first of those.
void build_thresholds(MaxnetFeatures& f, const ColumnRanges& r, int knots) {
  for (std::int32_t j : r.varying) {
    const std::vector<double> knot = evenly_spaced(r.low[j], r.high[j], knots + 2);
    for (std::size_t i = 2; i + 1 < knot.size(); ++i) {
      append(f, MaxnetKind::threshold, j, -1, knot[i], 0.0);
    }
  }
}

void build_products(MaxnetFeatures& f, const ColumnRanges& r, int) {
  for (std::size_t s = 0; s < r.varying.size(); ++s) {
    for (std::size_t t = s + 1; t < r.varying.size(); ++t) {
      append(f, MaxnetKind::product, r.varying[s], r.varying[t], 0.0, 0.0);
    }
  }
}

// The classes in the order their features are laid out in the design, whatever order the caller
// names them in.
struct FeatureClass {
  char letter;
  ClassBuilder build;
};

constexpr FeatureClass kFeatureClasses[] = {
    {'l', build_linear},     {'q', build_quadratic}, {'h', build_hinges},
    {'t', build_thresholds}, {'p', build_products},
};

bool names_class(const std::string& classes, char letter) {
  return classes.find(letter) != std::string::npos;
}

void check_classes(const std::string& classes) {
  if (classes.empty()) throw Error("maxnet's feature classes name at least one class.");
  for (char c : classes) {
    const bool known = std::any_of(std::begin(kFeatureClasses), std::end(kFeatureClasses),
                                   [c](const FeatureClass& fc) { return fc.letter == c; });
    if (!known) {
      throw Error(std::string("maxnet's feature classes are the letters l, q, p, h and t, not '") +
                  c + "'.");
    }
  }
}

MaxnetFeatures build_features(const double* x, std::size_t n, std::size_t p,
                              const std::string& classes, int knots) {
  check_classes(classes);
  if (knots < 2) throw Error("maxnet places at least two knots over a column's range.");
  const ColumnRanges ranges = column_ranges(x, n, p);
  MaxnetFeatures f;
  for (const FeatureClass& fc : kFeatureClasses) {
    if (names_class(classes, fc.letter)) fc.build(f, ranges, knots);
  }
  return f;
}

// The column-major design [n, features] over the rows of `x`.
std::vector<double> expand(const MaxnetFeatures& f, const double* x, std::size_t n) {
  std::vector<double> design(n * f.size());
  for (std::size_t k = 0; k < f.size(); ++k) {
    double* col = design.data() + k * n;
    for (std::size_t i = 0; i < n; ++i) col[i] = evaluate_at(f, k, x, n, i);
  }
  return design;
}

// The richer the model, the more flexible its fitted responses and the larger a presence count
// must be before they can be trusted: the classes a presence count supports by default.
std::string default_classes(std::size_t n_presence) {
  if (n_presence < 10) return "l";
  if (n_presence < 15) return "lq";
  if (n_presence < 80) return "lqh";
  return "lqph";
}

// ---------------------------------------------------------------------------------------------
// Regularisation (Phillips & Dudik 2008)

// Each class's multiplier as a function of the presence count. Linear, quadratic and product
// features share one curve, the one belonging to the richest of those classes the model holds;
// hinges and thresholds carry their own.
constexpr double kLinearAt[] = {0, 10, 30, 100}, kLinearBeta[] = {1, 1, 0.2, 0.05};
constexpr double kQuadraticAt[] = {0, 10, 17, 30, 100},
                 kQuadraticBeta[] = {1.3, 0.8, 0.5, 0.25, 0.05};
constexpr double kProductAt[] = {0, 10, 17, 30, 100},
                 kProductBeta[] = {2.6, 1.6, 0.9, 0.55, 0.05};
constexpr double kHingeAt[] = {0, 1}, kHingeBeta[] = {0.5, 0.5};
constexpr double kThresholdAt[] = {0, 100}, kThresholdBeta[] = {2, 1};

Curve shared_curve(const MaxnetFeatures& f) {
  bool quadratic = false, product = false;
  for (std::size_t k = 0; k < f.size(); ++k) {
    quadratic = quadratic || kind_of(f, k) == MaxnetKind::quadratic;
    product = product || kind_of(f, k) == MaxnetKind::product;
  }
  if (product) return {kProductAt, kProductBeta, 5};
  if (quadratic) return {kQuadraticAt, kQuadraticBeta, 5};
  return {kLinearAt, kLinearBeta, 4};
}

// Each feature's tolerance: its spread over the presences times the class multiplier over the
// square root of the presence count, held above a thousandth of its range and, for hinges and for
// thresholds every presence falls on one side of, above a floor of its own; all times `regmult`.
std::vector<double> tolerances(const MaxnetFeatures& f, const double* design, std::size_t n,
                               const double* presence, double regmult) {
  std::vector<std::size_t> present;
  for (std::size_t i = 0; i < n; ++i) {
    if (presence[i] == 1.0) present.push_back(i);
  }
  const double count = static_cast<double>(present.size());
  const double root = std::sqrt(count);
  const double shared = shared_curve(f)(count) / root;
  const double hinge = Curve{kHingeAt, kHingeBeta, 2}(count) / root;
  const double threshold = Curve{kThresholdAt, kThresholdBeta, 2}(count) / root;

  std::vector<double> reg(f.size());
  for (std::size_t k = 0; k < f.size(); ++k) {
    const double* col = design + k * n;
    const auto range = std::minmax_element(col, col + n);
    const double width = *range.second - *range.first;
    const double sd = spread_over(col, present);
    double multiplier = shared;
    double floor = 0.0;
    switch (kind_of(f, k)) {
      case MaxnetKind::hinge:
        multiplier = hinge;
        floor = std::max(sd, 1.0 / root) * 0.5 / root;
        break;
      case MaxnetKind::threshold: {
        multiplier = threshold;
        double on = 0.0;
        for (std::size_t i : present) on += col[i];
        if (on == 0.0 || on == count) floor = 1.0;
        break;
      }
      default:
        break;
    }
    reg[k] = std::max(std::max(0.001 * width, floor), sd * multiplier) * regmult;
  }
  return reg;
}

// ---------------------------------------------------------------------------------------------
// Rows fitted

// Whether some absence of `x` [n, p] reads exactly what unit `i` reads in every column.
bool absence_shares_readings(const double* x, const double* y, std::size_t n, std::size_t p,
                             std::size_t i) {
  for (std::size_t k = 0; k < n; ++k) {
    if (y[k] != 0.0) continue;
    std::size_t j = 0;
    while (j < p && x[i + j * n] == x[k + j * n]) ++j;
    if (j == p) return true;
  }
  return false;
}

// The units in order, then, in the presence-background model, each presence again as background
// where no absence already stands for it.
void choose_rows(const double* x, const double* y, std::size_t n, std::size_t p,
                 const MaxnetSpec& spec, MaxnetDesign& d) {
  d.rows.resize(n);
  for (std::size_t i = 0; i < n; ++i) d.rows[i] = i;
  d.y.assign(y, y + n);
  if (spec.formulation != MaxnetFormulation::background || !spec.add_samples) return;
  for (std::size_t i = 0; i < n; ++i) {
    if (y[i] != 1.0 || absence_shares_readings(x, y, n, p, i)) continue;
    d.rows.push_back(i);
    d.y.push_back(0.0);
  }
}

std::size_t count_presences(const double* y, std::size_t n) {
  std::size_t np = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (y[i] != 0.0 && y[i] != 1.0) throw Error("maxnet takes a response holding zero and one.");
    if (y[i] == 1.0) ++np;
  }
  return np;
}

// ---------------------------------------------------------------------------------------------
// The lasso

struct Selected {
  double lambda = 0.0;
  double intercept = 0.0;
  const double* beta = nullptr;   // one per feature, into the path that produced it
  std::int32_t stalled = 0;
  std::int32_t fold_stalled = 0;
};

PenaltySpec lasso_spec(const MaxnetSpec& spec, const std::vector<double>& reg) {
  PenaltySpec pen;
  pen.alpha = 1.0;
  pen.standardize = false;
  pen.intercept = true;
  pen.penalty_factor = reg;
  pen.thresh = spec.thresh;
  pen.max_pass = spec.max_pass;
  return pen;
}

// The presence-background path. With the case weights normalised to sum to one, the Phillips-Dudik
// tolerances enter the penalised likelihood as their mean times the presences' share of the total
// weight; the path descends to that value from four decades above it and is read at its end.
Selected select_background(const MaxnetDesign& d, PenaltySpec pen, PenaltyPath& path) {
  const std::size_t m = d.m;
  const std::size_t nf = d.features.size();
  std::vector<double> weight(m);
  double weight_total = 0.0, presence_total = 0.0, tolerance_total = 0.0;
  for (std::size_t r = 0; r < m; ++r) {
    weight[r] = d.y[r] + (1.0 - d.y[r]) * kBackgroundWeight;
    weight_total += weight[r];
    presence_total += d.y[r];
  }
  for (double v : d.reg) tolerance_total += v;
  const double base = tolerance_total / static_cast<double>(nf) * presence_total / weight_total;

  pen.prob_floor = kProbabilityFloor;
  const std::vector<double> decade = evenly_spaced(kPathDecades, 0.0, kPathPoints);
  pen.lambda.resize(decade.size());
  for (std::size_t k = 0; k < decade.size(); ++k) pen.lambda[k] = std::pow(10.0, decade[k]) * base;

  path = penalised_path(d.design.data(), d.y.data(), weight.data(), m, nf, Family::binomial, pen);
  const std::size_t end = path.lambda.size() - 1;
  Selected s;
  s.lambda = path.lambda[end];
  s.intercept = path.a0[end];
  s.beta = path.beta.data() + end * nf;
  s.stalled = path.stalled;
  return s;
}

// The presence-absence path, its penalty chosen by cross-validated deviance.
Selected select_absence(const MaxnetDesign& d, const MaxnetSpec& spec, PenaltySpec pen,
                        const double* w, const std::int32_t* fold, std::int32_t n_fold,
                        PenaltyCV& cv) {
  if (fold == nullptr || n_fold < 2) {
    throw Error("maxnet's absence formulation chooses its penalty on at least two folds.");
  }
  const std::size_t nf = d.features.size();
  pen.n_lambda = spec.n_lambda;
  pen.threads = spec.threads;
  cv = penalised_cv(d.design.data(), d.y.data(), w, d.m, nf, Family::binomial, pen, fold, n_fold);
  const std::size_t at = spec.one_se ? cv.index_1se : cv.index_min;
  Selected s;
  s.lambda = cv.path.lambda[at];
  s.intercept = cv.path.a0[at];
  s.beta = cv.path.beta.data() + at * nf;
  s.stalled = cv.path.stalled;
  for (std::int32_t f : cv.fold_stalled) s.fold_stalled += f > 0 ? 1 : 0;
  return s;
}

// The Gibbs distribution over the background rows: its log normaliser, taken about the largest
// link so that no exponential overflows, and its entropy.
void normalise_over_background(const MaxnetDesign& d, Maxnet& fit) {
  std::vector<double> link;
  for (std::size_t r = 0; r < d.m; ++r) {
    if (d.y[r] != 0.0) continue;
    double s = 0.0;
    for (std::size_t k = 0; k < fit.beta.size(); ++k) {
      s += fit.beta[k] * evaluate_at(fit.features, k, d.x.data(), d.m, r);
    }
    link.push_back(s);
  }
  const double top = *std::max_element(link.begin(), link.end());
  double mass = 0.0;
  for (double v : link) mass += std::exp(v - top);
  const double log_normaliser = top + std::log(mass);
  double entropy = 0.0;
  for (double v : link) {
    const double log_density = v - log_normaliser;
    entropy -= std::exp(log_density) * log_density;
  }
  fit.intercept = -log_normaliser;
  fit.entropy = entropy;
}

double respond(const Maxnet& fit, MaxnetOutput type, double link) {
  switch (type) {
    case MaxnetOutput::link:
      return link;
    case MaxnetOutput::exponential:
      return std::exp(link);
    case MaxnetOutput::cloglog:
      return 1.0 - std::exp(-std::exp(fit.entropy + link));
    case MaxnetOutput::logistic:
      return fit.formulation == MaxnetFormulation::background
                 ? 1.0 / (1.0 + std::exp(-fit.entropy - link))
                 : 1.0 / (1.0 + std::exp(-link));
  }
  return link;
}

}  // namespace

MaxnetFormulation maxnet_formulation_from_name(const std::string& name) {
  if (name == "background") return MaxnetFormulation::background;
  if (name == "absence") return MaxnetFormulation::absence;
  throw Error("maxnet's formulation is 'background' or 'absence', not '" + name + "'.");
}

const char* maxnet_formulation_name(MaxnetFormulation f) {
  return f == MaxnetFormulation::background ? "background" : "absence";
}

MaxnetOutput maxnet_output_from_name(const std::string& name) {
  if (name == "link") return MaxnetOutput::link;
  if (name == "exponential") return MaxnetOutput::exponential;
  if (name == "cloglog") return MaxnetOutput::cloglog;
  if (name == "logistic") return MaxnetOutput::logistic;
  throw Error("maxnet predicts 'link', 'exponential', 'cloglog' or 'logistic', not '" + name +
              "'.");
}

MaxnetDesign maxnet_design(const double* x, const double* y, std::size_t n, std::size_t p,
                           const MaxnetSpec& spec) {
  if (n == 0 || p == 0) throw Error("maxnet needs at least one unit and one column.");
  if (!(spec.regmult > 0.0) || !std::isfinite(spec.regmult)) {
    throw Error("maxnet's regularisation multiplier is a positive number.");
  }
  const std::size_t np = count_presences(y, n);
  if (np < 2) throw Error("maxnet fits at least two presences.");
  if (spec.formulation == MaxnetFormulation::absence && np == n) {
    throw Error("maxnet's absence formulation fits at least one absence.");
  }

  MaxnetDesign d;
  d.classes = spec.classes.empty() ? default_classes(np) : spec.classes;
  choose_rows(x, y, n, p, spec, d);
  d.m = d.rows.size();
  const std::size_t m = d.m;
  d.x.resize(m * p);
  for (std::size_t j = 0; j < p; ++j) {
    for (std::size_t r = 0; r < m; ++r) d.x[r + j * m] = x[d.rows[r] + j * n];
  }
  d.features = build_features(d.x.data(), m, p, d.classes, spec.knots);
  if (d.features.size() == 0) {
    throw Error("maxnet found no column holding more than one value to build on.");
  }
  const double gigabytes = static_cast<double>(m) * static_cast<double>(d.features.size()) *
                           sizeof(double) / 1e9;
  if (gigabytes > spec.max_design / spec.sharing) {
    throw Error("maxnet's '" + d.classes + "' classes over " + std::to_string(p) +
                " columns are " + std::to_string(d.features.size()) + " features over " +
                std::to_string(m) + " rows, a design of " + two_digits(gigabytes) +
                " GB, and the fit holds a centred copy beside it. The limit is " +
                two_digits(spec.max_design / spec.sharing) + " GB" +
                (spec.sharing > 1 ? ", `max_design = " + two_digits(spec.max_design) +
                                        "` shared by the " + std::to_string(spec.sharing) +
                                        " fits held at once"
                                  : std::string()) +
                ". Fit a coarser representation, fewer classes, fewer threads, or raise the "
                "limit.");
  }
  d.design = expand(d.features, d.x.data(), m);
  d.reg = tolerances(d.features, d.design.data(), m, d.y.data(), spec.regmult);
  return d;
}

namespace {

Maxnet fit_response(const double* x, const double* y, const double* w, std::size_t n,
                    std::size_t p, const MaxnetSpec& spec, const std::int32_t* fold,
                    std::int32_t n_fold) {
  const MaxnetDesign d = maxnet_design(x, y, n, p, spec);
  const std::size_t m = d.m;

  Maxnet fit;
  fit.formulation = spec.formulation;
  fit.classes = d.classes;
  fit.n_column = static_cast<std::int32_t>(p);
  for (std::size_t i = 0; i < n; ++i) fit.n_presence += y[i] == 1.0 ? 1 : 0;
  fit.n_feature = static_cast<std::int32_t>(d.features.size());
  fit.var_min.resize(p);
  fit.var_max.resize(p);
  for (std::size_t j = 0; j < p; ++j) {
    const double* col = d.x.data() + j * m;
    fit.var_min[j] = *std::min_element(col, col + m);
    fit.var_max[j] = *std::max_element(col, col + m);
  }

  const PenaltySpec pen = lasso_spec(spec, d.reg);
  PenaltyPath path;
  PenaltyCV cv;
  const Selected chosen = spec.formulation == MaxnetFormulation::background
                              ? select_background(d, pen, path)
                              : select_absence(d, spec, pen, w, fold, n_fold, cv);
  fit.lambda = chosen.lambda;
  fit.intercept = chosen.intercept;
  fit.lasso_intercept = chosen.intercept;
  fit.stalled = chosen.stalled;
  fit.fold_stalled = chosen.fold_stalled;

  for (std::size_t k = 0; k < d.features.size(); ++k) {
    if (chosen.beta[k] == 0.0) continue;
    append_copy(fit.features, d.features, k);
    const double* col = d.design.data() + k * m;
    const auto range = std::minmax_element(col, col + m);
    fit.feature_min.push_back(*range.first);
    fit.feature_max.push_back(*range.second);
    fit.beta.push_back(chosen.beta[k]);
  }

  if (spec.formulation == MaxnetFormulation::background) normalise_over_background(d, fit);
  return fit;
}

}  // namespace

std::vector<Maxnet> maxnet_fit(const double* x, std::size_t n, std::size_t p, const double* y,
                               const double* w, std::size_t r, const MaxnetSpec& spec,
                               const std::int32_t* fold, const std::int32_t* n_fold) {
  MaxnetSpec shared = spec;
  shared.sharing = static_cast<int>(std::max<std::size_t>(
      std::min<std::size_t>(static_cast<std::size_t>(std::max(spec.threads, 1)), r), 1));
  return detail::fit_responses(r, spec.threads, [&](std::size_t s, int inner) {
    MaxnetSpec each = shared;
    each.threads = inner;
    return fit_response(x, y + s * n, w == nullptr ? nullptr : w + s * n, n, p, each,
                        detail::response_fold(fold, n, s),
                        detail::response_fold_count(fold, n_fold, s));
  });
}

void maxnet_predict(const Maxnet& fit, const double* x, std::size_t n, std::size_t p, bool clamp,
                    MaxnetOutput type, double* out) {
  if (static_cast<std::int32_t>(p) != fit.n_column) {
    throw Error("a maxnet fit predicts on the columns it was fitted on.");
  }
  if (fit.formulation == MaxnetFormulation::absence &&
      (type == MaxnetOutput::exponential || type == MaxnetOutput::cloglog)) {
    throw Error("maxnet's absence formulation predicts its link and its probability, "
                "'logistic'; the exponential and cloglog outputs are the background "
                "formulation's.");
  }
  const MaxnetFeatures& f = fit.features;
  auto reading = [&](std::int32_t column, std::size_t i) {
    const double v = x[i + static_cast<std::size_t>(column) * n];
    if (!clamp) return v;
    return std::min(std::max(v, fit.var_min[column]), fit.var_max[column]);
  };
  for (std::size_t i = 0; i < n; ++i) {
    double link = fit.intercept;
    for (std::size_t k = 0; k < fit.beta.size(); ++k) {
      const double second = f.b[k] >= 0 ? reading(f.b[k], i) : 0.0;
      double v = evaluate(kind_of(f, k), f.lo[k], f.hi[k], reading(f.a[k], i), second);
      if (clamp) v = std::min(std::max(v, fit.feature_min[k]), fit.feature_max[k]);
      link += fit.beta[k] * v;
    }
    if (!std::isfinite(link)) throw Error("maxnet takes finite values alone.");
    out[i] = respond(fit, type, link);
  }
}

}  // namespace timesift
