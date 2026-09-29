#include "ts_maxnet.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace timesift {
namespace {

// maxnet's own constants: the weight of a background unit against a presence's, the probability
// glmnet pins a fitted case at under maxnet's `glmnet.control(pmin = 1e-8)`, and the length of its
// penalty path, which runs down four decades and is read at its last point.
constexpr double kBackgroundWeight = 100.0;
constexpr double kMaxnetProbFloor = 1e-8;
constexpr int kMaxnetPath = 200;

// R's `seq(from, to, length.out = m)`: the ends exactly, and the interior as `from` plus a whole
// number of steps.
std::vector<double> r_seq(double from, double to, int m) {
  std::vector<double> out(static_cast<std::size_t>(m));
  if (m == 1) {
    out[0] = from;
    return out;
  }
  if (from == to) {
    std::fill(out.begin(), out.end(), from);
    return out;
  }
  const double step = (to - from) / static_cast<double>(m - 1);
  out[0] = from;
  for (int i = 1; i < m - 1; ++i) out[static_cast<std::size_t>(i)] = from + i * step;
  out[static_cast<std::size_t>(m - 1)] = to;
  return out;
}

// R's `approx(x, y, v, rule = 2)`: linear between the two knots around `v`, the end value outside
// them, and a knot's own value where `v` is one.
double r_approx(const double* x, const double* y, int m, double v) {
  if (v < x[0]) return y[0];
  if (v > x[m - 1]) return y[m - 1];
  int i = 0, j = m - 1;
  while (i < j - 1) {
    const int ij = (i + j) / 2;
    if (v < x[ij]) j = ij; else i = ij;
  }
  if (v == x[j]) return y[j];
  if (v == x[i]) return y[i];
  return y[i] + (y[j] - y[i]) * ((v - x[i]) / (x[j] - x[i]));
}

// The standard deviation R's `sd()` reads, over the rows `rows` of one column: the mean refined by
// a second pass, then the squared deviations over one less than the count.
double r_sd(const double* col, const std::vector<std::size_t>& rows) {
  const double m = static_cast<double>(rows.size());
  double s = 0.0;
  for (std::size_t i : rows) s += col[i];
  double mean = s / m;
  double t = 0.0;
  for (std::size_t i : rows) t += col[i] - mean;
  mean += t / m;
  double ss = 0.0;
  for (std::size_t i : rows) {
    const double z = col[i] - mean;
    ss += z * z;
  }
  return std::sqrt(ss / (m - 1.0));
}

bool has_class(const std::string& classes, char c) {
  return classes.find(c) != std::string::npos;
}

void check_classes(const std::string& classes) {
  if (classes.empty()) throw Error("maxnet's feature classes name at least one class.");
  for (char c : classes) {
    if (std::string("lqpht").find(c) == std::string::npos) {
      throw Error(std::string("maxnet's feature classes are the letters l, q, p, h and t, not '") +
                  c + "'.");
    }
  }
}

void push(MaxnetFeatures& f, MaxnetKind kind, std::int32_t a, std::int32_t b, double lo,
          double hi) {
  f.kind.push_back(static_cast<std::int8_t>(kind));
  f.a.push_back(a);
  f.b.push_back(b);
  f.lo.push_back(lo);
  f.hi.push_back(hi);
}

std::string two_digits(double v) {
  char buffer[64];
  std::snprintf(buffer, sizeof buffer, "%.2g", v);
  return buffer;
}

double hinge_value(double x, double lo, double hi) {
  return std::min(1.0, std::max(0.0, (x - lo) / (hi - lo)));
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

std::string maxnet_default_classes(std::size_t n_presence) {
  if (n_presence < 10) return "l";
  if (n_presence < 15) return "lq";
  if (n_presence < 80) return "lqh";
  return "lqph";
}

MaxnetFeatures maxnet_features(const double* x, std::size_t n, std::size_t p,
                               const std::string& classes, int knots) {
  check_classes(classes);
  if (knots < 2) throw Error("maxnet places at least two knots over a column's range.");
  std::vector<std::int32_t> used;
  std::vector<double> low(p), high(p);
  for (std::size_t j = 0; j < p; ++j) {
    const double* col = x + j * n;
    double lo = col[0], hi = col[0];
    for (std::size_t i = 0; i < n; ++i) {
      if (!std::isfinite(col[i])) throw Error("maxnet takes finite values alone.");
      lo = std::min(lo, col[i]);
      hi = std::max(hi, col[i]);
    }
    low[j] = lo;
    high[j] = hi;
    if (hi > lo) used.push_back(static_cast<std::int32_t>(j));
  }

  MaxnetFeatures f;
  if (has_class(classes, 'l')) {
    for (std::int32_t j : used) push(f, MaxnetKind::linear, j, -1, 0.0, 0.0);
  }
  if (has_class(classes, 'q')) {
    for (std::int32_t j : used) push(f, MaxnetKind::quadratic, j, -1, 0.0, 0.0);
  }
  if (has_class(classes, 'h')) {
    for (std::int32_t j : used) {
      const std::vector<double> k = r_seq(low[j], high[j], knots);
      for (int i = 0; i + 1 < knots; ++i) {
        push(f, MaxnetKind::hinge, j, -1, k[static_cast<std::size_t>(i)], high[j]);
      }
      for (int i = 1; i < knots; ++i) {
        push(f, MaxnetKind::hinge, j, -1, low[j], k[static_cast<std::size_t>(i)]);
      }
    }
  }
  if (has_class(classes, 't')) {
    // maxnet cuts `knots + 2` points and keeps its third to its second last, which is
    // `seq(min, max, length = knots + 2)[2:knots + 1]` read as R reads it.
    for (std::int32_t j : used) {
      const std::vector<double> k = r_seq(low[j], high[j], knots + 2);
      for (int i = 2; i <= knots; ++i) {
        push(f, MaxnetKind::threshold, j, -1, k[static_cast<std::size_t>(i)], 0.0);
      }
    }
  }
  if (has_class(classes, 'p')) {
    for (std::size_t s = 0; s < used.size(); ++s) {
      for (std::size_t t = s + 1; t < used.size(); ++t) {
        push(f, MaxnetKind::product, used[s], used[t], 0.0, 0.0);
      }
    }
  }
  return f;
}

double maxnet_feature_value(const MaxnetFeatures& f, std::size_t k, const double* x,
                            std::size_t n, std::size_t i) {
  const double v = x[i + static_cast<std::size_t>(f.a[k]) * n];
  switch (static_cast<MaxnetKind>(f.kind[k])) {
    case MaxnetKind::linear:
      return v;
    case MaxnetKind::quadratic:
      return v * v;
    case MaxnetKind::hinge:
      return hinge_value(v, f.lo[k], f.hi[k]);
    case MaxnetKind::threshold:
      return v >= f.lo[k] ? 1.0 : 0.0;
    case MaxnetKind::product:
      return v * x[i + static_cast<std::size_t>(f.b[k]) * n];
  }
  throw Error("a maxnet feature of a kind the core does not know.");
}

void maxnet_expand(const MaxnetFeatures& f, const double* x, std::size_t n, double* out) {
  for (std::size_t k = 0; k < f.size(); ++k) {
    double* col = out + k * n;
    for (std::size_t i = 0; i < n; ++i) col[i] = maxnet_feature_value(f, k, x, n, i);
  }
}

std::vector<double> maxnet_regularization(const MaxnetFeatures& f, const double* design,
                                          std::size_t n, const double* presence, double regmult) {
  std::vector<std::size_t> present;
  for (std::size_t i = 0; i < n; ++i) {
    if (presence[i] == 1.0) present.push_back(i);
  }
  const std::size_t np = present.size();
  if (np < 2) throw Error("maxnet's regularisation reads at least two presences.");
  const double npd = static_cast<double>(np);

  // The linear, quadratic and product classes share one table, and which one follows the richest
  // class the model holds; hinges and thresholds carry their own.
  static const double l_x[] = {0, 10, 30, 100}, l_y[] = {1, 1, 0.2, 0.05};
  static const double q_x[] = {0, 10, 17, 30, 100}, q_y[] = {1.3, 0.8, 0.5, 0.25, 0.05};
  static const double p_x[] = {0, 10, 17, 30, 100}, p_y[] = {2.6, 1.6, 0.9, 0.55, 0.05};
  static const double h_x[] = {0, 1}, h_y[] = {0.5, 0.5};
  static const double t_x[] = {0, 100}, t_y[] = {2, 1};
  bool any_quadratic = false, any_product = false;
  for (std::size_t k = 0; k < f.size(); ++k) {
    any_quadratic = any_quadratic || f.kind[k] == static_cast<std::int8_t>(MaxnetKind::quadratic);
    any_product = any_product || f.kind[k] == static_cast<std::int8_t>(MaxnetKind::product);
  }
  const double* shared_x = l_x;
  const double* shared_y = l_y;
  int shared_m = 4;
  if (any_quadratic) {
    shared_x = q_x;
    shared_y = q_y;
    shared_m = 5;
  }
  if (any_product) {
    shared_x = p_x;
    shared_y = p_y;
    shared_m = 5;
  }
  const double shared = r_approx(shared_x, shared_y, shared_m, npd) / std::sqrt(npd);
  const double hinge = r_approx(h_x, h_y, 2, npd) / std::sqrt(npd);
  const double threshold = r_approx(t_x, t_y, 2, npd) / std::sqrt(npd);

  std::vector<double> reg(f.size());
  for (std::size_t k = 0; k < f.size(); ++k) {
    const double* col = design + k * n;
    const MaxnetKind kind = static_cast<MaxnetKind>(f.kind[k]);
    double lo = col[0], hi = col[0];
    for (std::size_t i = 0; i < n; ++i) {
      lo = std::min(lo, col[i]);
      hi = std::max(hi, col[i]);
    }
    const double sd = r_sd(col, present);
    double mindev = 0.0;
    double scale = shared;
    if (kind == MaxnetKind::hinge) {
      scale = hinge;
      mindev = std::max(sd, 1.0 / std::sqrt(npd)) * 0.5 / std::sqrt(npd);
    } else if (kind == MaxnetKind::threshold) {
      scale = threshold;
      double s = 0.0;
      for (std::size_t i : present) s += col[i];
      if (s == 0.0 || s == npd) mindev = 1.0;
    }
    reg[k] = std::max(std::max(0.001 * (hi - lo), mindev), sd * scale) * regmult;
  }
  return reg;
}

MaxnetDesign maxnet_design(const double* x, const double* y, std::size_t n, std::size_t p,
                           const MaxnetSpec& spec) {
  if (n == 0 || p == 0) throw Error("maxnet needs at least one unit and one column.");
  if (!(spec.regmult > 0.0) || !std::isfinite(spec.regmult)) {
    throw Error("maxnet's regularisation multiplier is a positive number.");
  }
  std::size_t np = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (y[i] != 0.0 && y[i] != 1.0) throw Error("maxnet takes a response holding zero and one.");
    if (y[i] == 1.0) ++np;
  }
  if (np < 2) throw Error("maxnet fits at least two presences.");
  if (spec.formulation == MaxnetFormulation::absence && np == n) {
    throw Error("maxnet's absence formulation fits at least one absence.");
  }

  MaxnetDesign d;
  d.classes = spec.classes.empty() ? maxnet_default_classes(np) : spec.classes;

  // The rows fitted. maxnet adds each presence to the background where no absence already carries
  // the same readings in every column, after every unit, in the order the presences come.
  d.rows.resize(n);
  for (std::size_t i = 0; i < n; ++i) d.rows[i] = i;
  d.y.assign(y, y + n);
  if (spec.formulation == MaxnetFormulation::background && spec.add_samples) {
    for (std::size_t i = 0; i < n; ++i) {
      if (y[i] != 1.0) continue;
      bool matched = false;
      for (std::size_t k = 0; k < n && !matched; ++k) {
        if (y[k] != 0.0) continue;
        bool same = true;
        for (std::size_t j = 0; j < p && same; ++j) same = x[i + j * n] == x[k + j * n];
        matched = same;
      }
      if (!matched) {
        d.rows.push_back(i);
        d.y.push_back(0.0);
      }
    }
  }
  d.m = d.rows.size();
  const std::size_t m = d.m;
  d.x.resize(m * p);
  for (std::size_t j = 0; j < p; ++j) {
    for (std::size_t r = 0; r < m; ++r) d.x[r + j * m] = x[d.rows[r] + j * n];
  }
  d.features = maxnet_features(d.x.data(), m, p, d.classes, spec.knots);
  if (d.features.size() == 0) {
    throw Error("maxnet found no column holding more than one value to build on.");
  }
  const double gigabytes = static_cast<double>(m) * static_cast<double>(d.features.size()) *
                           sizeof(double) / 1e9;
  if (gigabytes > spec.max_design) {
    throw Error("maxnet's '" + d.classes + "' classes over " + std::to_string(p) +
                " columns are " + std::to_string(d.features.size()) + " features over " +
                std::to_string(m) + " rows, a design of " + two_digits(gigabytes) +
                " GB, and the fit holds a centred copy beside it. The limit is " +
                two_digits(spec.max_design) +
                " GB. Fit a coarser representation, fewer classes, or raise the limit.");
  }
  d.design.resize(m * d.features.size());
  maxnet_expand(d.features, d.x.data(), m, d.design.data());
  d.reg = maxnet_regularization(d.features, d.design.data(), m, d.y.data(), spec.regmult);
  return d;
}

Maxnet maxnet_fit(const double* x, const double* y, const double* w, std::size_t n, std::size_t p,
                  const MaxnetSpec& spec, const std::int32_t* fold, std::int32_t n_fold) {
  const MaxnetDesign d = maxnet_design(x, y, n, p, spec);
  const std::size_t m = d.m;
  const std::size_t nf = d.features.size();
  const std::vector<double>& yr = d.y;
  const std::vector<double>& reg = d.reg;
  const std::vector<double>& design = d.design;
  const MaxnetFeatures& all = d.features;

  Maxnet out;
  out.formulation = spec.formulation;
  out.classes = d.classes;
  out.n_column = static_cast<std::int32_t>(p);
  for (std::size_t i = 0; i < n; ++i) out.n_presence += y[i] == 1.0 ? 1 : 0;
  out.n_feature = static_cast<std::int32_t>(nf);
  out.var_min.assign(p, 0.0);
  out.var_max.assign(p, 0.0);
  for (std::size_t j = 0; j < p; ++j) {
    const double* col = d.x.data() + j * m;
    out.var_min[j] = *std::min_element(col, col + m);
    out.var_max[j] = *std::max_element(col, col + m);
  }

  PenaltySpec pen;
  pen.alpha = 1.0;
  pen.standardize = false;
  pen.intercept = true;
  pen.penalty_factor = reg;
  pen.thresh = spec.thresh;
  pen.max_pass = spec.max_pass;

  std::vector<double> beta(nf, 0.0);
  if (spec.formulation == MaxnetFormulation::background) {
    // maxnet's path: four decades down to the mean factor times the presences' share of the total
    // weight, which is the regularisation MaxEnt's own objective carries once glmnet's weights are
    // normalised to sum to one.
    std::vector<double> wr(m);
    double wsum = 0.0, psum = 0.0, rsum = 0.0;
    for (std::size_t r = 0; r < m; ++r) {
      wr[r] = yr[r] + (1.0 - yr[r]) * kBackgroundWeight;
      wsum += wr[r];
      psum += yr[r];
    }
    for (double v : reg) rsum += v;
    const double scale = rsum / static_cast<double>(nf) * psum / wsum;
    pen.prob_floor = kMaxnetProbFloor;
    const std::vector<double> exponent = r_seq(4.0, 0.0, kMaxnetPath);
    pen.lambda.resize(exponent.size());
    for (std::size_t k = 0; k < exponent.size(); ++k) {
      pen.lambda[k] = std::pow(10.0, exponent[k]) * scale;
    }
    const PenaltyPath path = penalised_path(design.data(), yr.data(), wr.data(), m, nf,
                                            Family::binomial, pen);
    const std::size_t last = path.lambda.size() - 1;
    out.stalled = path.stalled;
    out.lambda = path.lambda[last];
    out.lasso_intercept = path.a0[last];
    for (std::size_t k = 0; k < nf; ++k) beta[k] = path.beta[last * nf + k];
  } else {
    if (fold == nullptr || n_fold < 2) {
      throw Error("maxnet's absence formulation chooses its penalty on at least two folds.");
    }
    pen.n_lambda = spec.n_lambda;
    pen.threads = spec.threads;
    const PenaltyCV cv = penalised_cv(design.data(), yr.data(), w, m, nf, Family::binomial, pen,
                                      fold, n_fold);
    const std::size_t at = spec.one_se ? cv.index_1se : cv.index_min;
    out.stalled = cv.path.stalled;
    for (std::int32_t s : cv.fold_stalled) out.fold_stalled += s > 0 ? 1 : 0;
    out.lambda = cv.path.lambda[at];
    out.intercept = cv.path.a0[at];
    out.lasso_intercept = out.intercept;
    for (std::size_t k = 0; k < nf; ++k) beta[k] = cv.path.beta[at * nf + k];
  }

  for (std::size_t k = 0; k < nf; ++k) {
    if (beta[k] == 0.0) continue;
    push(out.features, static_cast<MaxnetKind>(all.kind[k]), all.a[k], all.b[k], all.lo[k],
         all.hi[k]);
    const double* col = design.data() + k * m;
    out.feature_min.push_back(*std::min_element(col, col + m));
    out.feature_max.push_back(*std::max_element(col, col + m));
    out.beta.push_back(beta[k]);
  }

  if (spec.formulation == MaxnetFormulation::background) {
    // The intercept is not glmnet's. It is the normaliser that makes the exponential output sum to
    // one over the background, and the entropy of that distribution is what the cloglog and
    // logistic outputs are read against. Both are taken through the largest link, which is the same
    // number maxnet's `-log(sum(exp(link)))` is wherever that one does not overflow.
    std::vector<double> link;
    for (std::size_t r = 0; r < m; ++r) {
      if (yr[r] != 0.0) continue;
      double s = 0.0;
      for (std::size_t k = 0; k < out.beta.size(); ++k) {
        s += out.beta[k] * maxnet_feature_value(out.features, k, d.x.data(), m, r);
      }
      link.push_back(s);
    }
    const double top = *std::max_element(link.begin(), link.end());
    double total = 0.0;
    for (double v : link) total += std::exp(v - top);
    const double log_sum = top + std::log(total);
    double entropy = 0.0;
    for (double v : link) {
      const double lr = v - log_sum;
      entropy -= std::exp(lr) * lr;
    }
    out.intercept = -log_sum;
    out.entropy = entropy;
  }
  return out;
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
  std::vector<double> held;
  const double* xs = x;
  if (clamp) {
    held.assign(x, x + n * p);
    for (std::size_t j = 0; j < p; ++j) {
      double* col = held.data() + j * n;
      for (std::size_t i = 0; i < n; ++i) {
        col[i] = std::min(std::max(col[i], fit.var_min[j]), fit.var_max[j]);
      }
    }
    xs = held.data();
  }
  for (std::size_t i = 0; i < n; ++i) {
    double link = fit.intercept;
    for (std::size_t k = 0; k < fit.beta.size(); ++k) {
      double v = maxnet_feature_value(fit.features, k, xs, n, i);
      if (clamp) v = std::min(std::max(v, fit.feature_min[k]), fit.feature_max[k]);
      link += fit.beta[k] * v;
    }
    if (!std::isfinite(link)) throw Error("maxnet takes finite values alone.");
    switch (type) {
      case MaxnetOutput::link:
        out[i] = link;
        break;
      case MaxnetOutput::exponential:
        out[i] = std::exp(link);
        break;
      case MaxnetOutput::cloglog:
        out[i] = 1.0 - std::exp(-std::exp(fit.entropy + link));
        break;
      case MaxnetOutput::logistic:
        out[i] = fit.formulation == MaxnetFormulation::background
                     ? 1.0 / (1.0 + std::exp(-fit.entropy - link))
                     : 1.0 / (1.0 + std::exp(-link));
        break;
    }
  }
}

}  // namespace timesift
