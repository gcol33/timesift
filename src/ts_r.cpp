#include <cpp11.hpp>

#include <cmath>
#include <string>
#include <vector>

#include "ts_additive.h"
#include "ts_core.h"
#include "ts_envelope.h"
#include "ts_fda.h"
#include "ts_hierarchical.h"
#include "ts_mars.h"
#include "ts_perceptron.h"
#include "ts_maxnet.h"
#include "ts_penalised.h"
#include "ts_stepwise.h"
#include "ts_tree.h"

namespace {

std::vector<timesift::seconds> as_seconds(const cpp11::doubles& x) {
  std::vector<timesift::seconds> out(x.size());
  for (R_xlen_t i = 0; i < x.size(); ++i) {
    out[static_cast<std::size_t>(i)] =
        static_cast<timesift::seconds>(std::floor(x[i]));
  }
  return out;
}

// The storage a Request points into, built once for the reduction and for the coverage so the
// two read the same readings the same way.
struct Held {
  std::vector<std::int32_t> unit_index;
  std::vector<double> reading;
  std::vector<timesift::seconds> instant;
  std::vector<timesift::seconds> naive;
  std::vector<timesift::seconds> supplied;
  std::vector<std::string> names;
  std::vector<const char*> name_ptr;
  timesift::Request req;
};

Held hold(cpp11::integers unit, cpp11::sexp value, cpp11::doubles when, cpp11::doubles local,
          cpp11::sexp custom, cpp11::strings unit_names, const std::string& grain,
          int year_month, int year_day) {
  Held h;
  const std::size_t n = static_cast<std::size_t>(local.size());

  h.unit_index.resize(n);
  for (std::size_t i = 0; i < n; ++i) h.unit_index[i] = unit[static_cast<R_xlen_t>(i)] - 1;

  h.reading.assign(n, 0.0);
  if (value != R_NilValue) {
    cpp11::doubles v(value);
    for (std::size_t i = 0; i < n; ++i) h.reading[i] = v[static_cast<R_xlen_t>(i)];
  }

  h.instant = as_seconds(when);
  h.naive = as_seconds(local);
  if (custom != R_NilValue) h.supplied = as_seconds(cpp11::doubles(custom));

  h.names.reserve(static_cast<std::size_t>(unit_names.size()));
  for (R_xlen_t i = 0; i < unit_names.size(); ++i) {
    h.names.push_back(std::string(unit_names[i]));
  }
  h.name_ptr.reserve(h.names.size());
  for (const std::string& s : h.names) h.name_ptr.push_back(s.c_str());

  h.req.unit = h.unit_index.data();
  h.req.value = h.reading.data();
  h.req.when = h.instant.data();
  h.req.local = h.naive.data();
  h.req.custom = h.supplied.empty() ? nullptr : h.supplied.data();
  h.req.unit_name = h.name_ptr.empty() ? nullptr : h.name_ptr.data();
  h.req.n = n;
  h.req.n_unit = h.names.size();
  h.req.grain = timesift::grain_from_name(grain);
  h.req.year_start = timesift::YearStart{year_month, year_day};
  return h;
}

}  // namespace

[[cpp11::register]]
cpp11::list ts_reduce_(cpp11::integers unit, cpp11::doubles value, cpp11::doubles when,
                       cpp11::doubles local, cpp11::sexp custom, cpp11::strings unit_names,
                       std::string grain, int year_month, int year_day, cpp11::strings stats,
                       double sampling_step) {
  Held h = hold(unit, value, when, local, custom, unit_names, grain, year_month, year_day);
  timesift::Request& req = h.req;
  req.sampling_step = static_cast<timesift::seconds>(sampling_step);
  for (R_xlen_t i = 0; i < stats.size(); ++i) {
    req.stats.push_back(timesift::stat_from_name(std::string(stats[i])));
  }

  const timesift::Result result = timesift::reduce(req);
  const std::size_t n_bin = result.bin_start.size();

  cpp11::writable::doubles values(static_cast<R_xlen_t>(result.values.size()));
  for (std::size_t i = 0; i < result.values.size(); ++i) {
    values[static_cast<R_xlen_t>(i)] = result.values[i];
  }
  cpp11::writable::doubles bin_start(static_cast<R_xlen_t>(n_bin));
  cpp11::writable::doubles bin_end(static_cast<R_xlen_t>(n_bin));
  cpp11::writable::logicals bin_partial(static_cast<R_xlen_t>(n_bin));
  for (std::size_t k = 0; k < n_bin; ++k) {
    bin_start[static_cast<R_xlen_t>(k)] = static_cast<double>(result.bin_start[k]);
    bin_end[static_cast<R_xlen_t>(k)] = static_cast<double>(result.bin_end[k]);
    bin_partial[static_cast<R_xlen_t>(k)] =
        result.bin_partial[k] ? TRUE : FALSE;
  }
  cpp11::writable::integers bin_n(static_cast<R_xlen_t>(result.bin_n.size()));
  for (std::size_t i = 0; i < result.bin_n.size(); ++i) {
    bin_n[static_cast<R_xlen_t>(i)] = result.bin_n[i];
  }

  using namespace cpp11::literals;
  return cpp11::writable::list({
    "values"_nm = values,
    "bin_start"_nm = bin_start,
    "bin_end"_nm = bin_end,
    "bin_n"_nm = bin_n,
    "bin_partial"_nm = bin_partial
  });
}

[[cpp11::register]]
cpp11::list ts_coverage_(cpp11::integers unit, cpp11::doubles when, cpp11::doubles local,
                         cpp11::sexp custom, cpp11::strings unit_names, std::string grain,
                         int year_month, int year_day) {
  Held h = hold(unit, R_NilValue, when, local, custom, unit_names, grain, year_month, year_day);
  const timesift::Coverage result = timesift::coverage(h.req);

  cpp11::writable::doubles bin_start(static_cast<R_xlen_t>(result.bin_start.size()));
  for (std::size_t k = 0; k < result.bin_start.size(); ++k) {
    bin_start[static_cast<R_xlen_t>(k)] = static_cast<double>(result.bin_start[k]);
  }
  cpp11::writable::integers count(static_cast<R_xlen_t>(result.count.size()));
  for (std::size_t i = 0; i < result.count.size(); ++i) {
    count[static_cast<R_xlen_t>(i)] = result.count[i];
  }
  using namespace cpp11::literals;
  return cpp11::writable::list({"bin_start"_nm = bin_start, "count"_nm = count});
}

[[cpp11::register]]
cpp11::list ts_reduce_lookbacks_(cpp11::integers unit, cpp11::doubles value, cpp11::doubles when,
                               cpp11::doubles local, cpp11::strings unit_names,
                               cpp11::integers target_unit,
                               cpp11::doubles target_at, cpp11::strings target_names,
                               double span, double lag, int bins, cpp11::strings stats) {
  const std::size_t n = static_cast<std::size_t>(value.size());
  const std::size_t n_target = static_cast<std::size_t>(target_at.size());

  std::vector<std::int32_t> unit_index(n);
  for (std::size_t i = 0; i < n; ++i) unit_index[i] = unit[static_cast<R_xlen_t>(i)] - 1;

  std::vector<double> reading(n);
  for (std::size_t i = 0; i < n; ++i) reading[i] = value[static_cast<R_xlen_t>(i)];

  const std::vector<timesift::seconds> instant = as_seconds(when);
  const std::vector<timesift::seconds> naive = as_seconds(local);
  const std::vector<timesift::seconds> anchor = as_seconds(target_at);

  std::vector<std::int32_t> holder(n_target);
  for (std::size_t i = 0; i < n_target; ++i) {
    holder[i] = target_unit[static_cast<R_xlen_t>(i)] - 1;
  }

  std::vector<std::string> units, targets;
  for (R_xlen_t i = 0; i < unit_names.size(); ++i) units.push_back(std::string(unit_names[i]));
  for (R_xlen_t i = 0; i < target_names.size(); ++i) {
    targets.push_back(std::string(target_names[i]));
  }
  std::vector<const char*> unit_ptr, target_ptr;
  for (const std::string& s : units) unit_ptr.push_back(s.c_str());
  for (const std::string& s : targets) target_ptr.push_back(s.c_str());

  timesift::LookbackRequest req;
  req.unit = unit_index.data();
  req.value = reading.data();
  req.when = instant.data();
  req.local = naive.data();
  req.unit_name = unit_ptr.empty() ? nullptr : unit_ptr.data();
  req.n = n;
  req.n_unit = units.size();
  req.target_unit = holder.data();
  req.target_at = anchor.data();
  req.target_name = target_ptr.empty() ? nullptr : target_ptr.data();
  req.n_target = n_target;
  req.span = static_cast<timesift::seconds>(span);
  req.lag = static_cast<timesift::seconds>(lag);
  req.n_bin = bins;
  for (R_xlen_t i = 0; i < stats.size(); ++i) {
    req.stats.push_back(timesift::stat_from_name(std::string(stats[i])));
  }

  const timesift::LookbackResult result = timesift::reduce_lookbacks(req);

  cpp11::writable::doubles values(static_cast<R_xlen_t>(result.values.size()));
  for (std::size_t i = 0; i < result.values.size(); ++i) {
    values[static_cast<R_xlen_t>(i)] = result.values[i];
  }
  cpp11::writable::integers bin_n(static_cast<R_xlen_t>(result.bin_n.size()));
  for (std::size_t i = 0; i < result.bin_n.size(); ++i) {
    bin_n[static_cast<R_xlen_t>(i)] = result.bin_n[i];
  }

  using namespace cpp11::literals;
  return cpp11::writable::list({"values"_nm = values, "bin_n"_nm = bin_n});
}

// The seven grains and the civil arithmetic under them, reachable from the suites so the oracle
// can be checked against the core rather than only through a whole reduction.
[[cpp11::register]]
cpp11::doubles ts_bin_starts_(cpp11::doubles local, std::string grain, int year_month,
                              int year_day) {
  const std::vector<timesift::seconds> naive = as_seconds(local);
  std::vector<timesift::seconds> out(naive.size());
  timesift::bin_starts(naive.data(), naive.size(), timesift::grain_from_name(grain),
                        timesift::YearStart{year_month, year_day}, out.data());
  cpp11::writable::doubles result(static_cast<R_xlen_t>(out.size()));
  for (std::size_t i = 0; i < out.size(); ++i) {
    result[static_cast<R_xlen_t>(i)] = static_cast<double>(out[i]);
  }
  return result;
}

[[cpp11::register]]
cpp11::doubles ts_bin_nexts_(cpp11::doubles bins, std::string grain, int year_month,
                             int year_day) {
  const std::vector<timesift::seconds> start = as_seconds(bins);
  std::vector<timesift::seconds> out(start.size());
  timesift::bin_nexts(start.data(), start.size(), timesift::grain_from_name(grain),
                       timesift::YearStart{year_month, year_day}, out.data());
  cpp11::writable::doubles result(static_cast<R_xlen_t>(out.size()));
  for (std::size_t i = 0; i < out.size(); ++i) {
    result[static_cast<R_xlen_t>(i)] = static_cast<double>(out[i]);
  }
  return result;
}

// The position of each bin in a cycle, from the instants the representation carries. The fraction
// is reachable on its own because it is the part of the position the contract pins exactly.
[[cpp11::register]]
cpp11::doubles ts_cycle_fraction_(cpp11::doubles bin_start, cpp11::doubles bin_end,
                                  std::string cycle) {
  const std::vector<timesift::seconds> opens = as_seconds(bin_start);
  const std::vector<timesift::seconds> closes = as_seconds(bin_end);
  std::vector<double> frac(opens.size());
  timesift::cycle_fraction(cycle, opens.data(), closes.data(), opens.size(), frac.data());
  cpp11::writable::doubles out(static_cast<R_xlen_t>(frac.size()));
  for (std::size_t i = 0; i < frac.size(); ++i) out[static_cast<R_xlen_t>(i)] = frac[i];
  return out;
}

[[cpp11::register]]
cpp11::doubles ts_cycle_phase_(cpp11::doubles bin_start, cpp11::doubles bin_end,
                               std::string cycle) {
  const std::vector<timesift::seconds> opens = as_seconds(bin_start);
  const std::vector<timesift::seconds> closes = as_seconds(bin_end);
  const std::size_t n = opens.size();
  std::vector<double> out_sin(n), out_cos(n);
  timesift::cycle_phase(cycle, opens.data(), closes.data(), n, out_sin.data(), out_cos.data());
  cpp11::writable::doubles out(static_cast<R_xlen_t>(2 * n));
  for (std::size_t i = 0; i < n; ++i) {
    out[static_cast<R_xlen_t>(i)] = out_sin[i];
    out[static_cast<R_xlen_t>(n + i)] = out_cos[i];
  }
  return out;
}

// The penalised fit, from the same core the Python side calls. The design arrives as the
// column-major buffer an R matrix already is, so nothing is copied to reach the descent.
namespace {

timesift::PenaltySpec penalty_spec(double alpha, int n_lambda, double lambda_min_ratio,
                                   cpp11::sexp lambda, double thresh, bool standardize,
                                   bool intercept, double max_pass, int threads) {
  timesift::PenaltySpec spec;
  spec.alpha = alpha;
  spec.n_lambda = n_lambda;
  spec.lambda_min_ratio = lambda_min_ratio;
  spec.thresh = thresh;
  spec.max_pass = static_cast<int>(max_pass);
  spec.threads = threads;
  spec.standardize = standardize;
  spec.intercept = intercept;
  if (lambda != R_NilValue) {
    cpp11::doubles given(lambda);
    for (R_xlen_t i = 0; i < given.size(); ++i) spec.lambda.push_back(given[i]);
  }
  return spec;
}

cpp11::writable::doubles give(const std::vector<double>& from) {
  cpp11::writable::doubles out(static_cast<R_xlen_t>(from.size()));
  for (std::size_t i = 0; i < from.size(); ++i) out[static_cast<R_xlen_t>(i)] = from[i];
  return out;
}

cpp11::writable::integers give(const std::vector<std::int32_t>& from) {
  cpp11::writable::integers out(static_cast<R_xlen_t>(from.size()));
  for (std::size_t i = 0; i < from.size(); ++i) out[static_cast<R_xlen_t>(i)] = from[i];
  return out;
}

cpp11::writable::list give(const timesift::PenaltyPath& path) {
  using namespace cpp11::literals;
  return cpp11::writable::list({
    "lambda"_nm = give(path.lambda),
    "a0"_nm = give(path.a0),
    "beta"_nm = give(path.beta),
    "df"_nm = give(path.df),
    "dev_ratio"_nm = give(path.dev_ratio),
    "null_deviance"_nm = cpp11::as_sexp(path.null_deviance),
    "passes"_nm = cpp11::as_sexp(path.passes),
    "stalled"_nm = cpp11::as_sexp(path.stalled),
    "n_column"_nm = cpp11::as_sexp(static_cast<int>(path.n_column)),
    "family"_nm = cpp11::as_sexp(std::string(timesift::family_name(path.family)))
  });
}

timesift::PenaltyPath take(cpp11::doubles lambda, cpp11::doubles a0, cpp11::doubles beta,
                           const std::string& family) {
  timesift::PenaltyPath path;
  path.family = timesift::family_from_name(family);
  for (R_xlen_t i = 0; i < lambda.size(); ++i) path.lambda.push_back(lambda[i]);
  for (R_xlen_t i = 0; i < a0.size(); ++i) path.a0.push_back(a0[i]);
  for (R_xlen_t i = 0; i < beta.size(); ++i) path.beta.push_back(beta[i]);
  path.n_column = path.lambda.empty() ? 0 : path.beta.size() / path.lambda.size();
  return path;
}

// One 0-based fold index per observation, or none.
std::vector<std::int32_t> take_folds(cpp11::sexp fold) {
  std::vector<std::int32_t> which;
  if (fold != R_NilValue) {
    cpp11::integers given(fold);
    for (R_xlen_t i = 0; i < given.size(); ++i) which.push_back(given[i]);
  }
  return which;
}

// A fold map of one 0-based index per unit and response, [n, r], with one count of folds per
// response, or no map at all.
struct ResponseFolds {
  std::vector<std::int32_t> which;
  std::vector<std::int32_t> count;
  const std::int32_t* fold() const { return which.empty() ? nullptr : which.data(); }
  const std::int32_t* n_fold() const { return which.empty() ? nullptr : count.data(); }
};

ResponseFolds take_response_folds(cpp11::sexp fold, cpp11::integers n_fold, int n, int r,
                                  const char* who) {
  ResponseFolds out;
  out.which = take_folds(fold);
  out.count.assign(n_fold.begin(), n_fold.end());
  if (!out.which.empty() &&
      (out.which.size() != static_cast<std::size_t>(n) * static_cast<std::size_t>(r) ||
       out.count.size() != static_cast<std::size_t>(r))) {
    throw std::invalid_argument(std::string(who) + "'s folds are one per unit and response, and "
                                "one count per response");
  }
  return out;
}

// A seed is a number modulo 2^32, which an R integer cannot carry, so it crosses as a double.
std::uint32_t take_seed(double seed) {
  return static_cast<std::uint32_t>(static_cast<std::uint64_t>(seed));
}

std::vector<std::uint32_t> take_seeds(cpp11::doubles seeds, int r, const char* who) {
  if (seeds.size() != r) {
    throw std::invalid_argument(std::string(who) + " is fitted under one seed per response");
  }
  std::vector<std::uint32_t> out;
  for (double s : seeds) out.push_back(take_seed(s));
  return out;
}

}  // namespace

[[cpp11::register]]
cpp11::list ts_penalised_path_(cpp11::doubles x, cpp11::doubles y, cpp11::doubles w, int n, int p,
                               std::string family, double alpha, int n_lambda,
                               double lambda_min_ratio, cpp11::sexp lambda, double thresh,
                               bool standardize, bool intercept, double max_pass) {
  const timesift::PenaltySpec spec = penalty_spec(alpha, n_lambda, lambda_min_ratio, lambda,
                                                  thresh, standardize, intercept, max_pass, 1);
  const timesift::PenaltyPath path =
      timesift::penalised_path(REAL_RO(x.data()), REAL_RO(y.data()), REAL_RO(w.data()), static_cast<std::size_t>(n),
                               static_cast<std::size_t>(p), timesift::family_from_name(family),
                               spec);
  return give(path);
}

// `y` and `w` are [n, r], one cross-validated path per column under its own column of `fold`
// [n, r].
[[cpp11::register]]
cpp11::list ts_penalised_cv_(cpp11::doubles x, cpp11::doubles y, cpp11::doubles w, int n, int p,
                             int r, std::string family, double alpha, int n_lambda,
                             double lambda_min_ratio, cpp11::sexp lambda, double thresh,
                             bool standardize, bool intercept, cpp11::sexp fold,
                             cpp11::integers n_fold, double max_pass, int threads) {
  const timesift::PenaltySpec spec = penalty_spec(alpha, n_lambda, lambda_min_ratio, lambda,
                                                  thresh, standardize, intercept, max_pass,
                                                  threads);
  const ResponseFolds folds = take_response_folds(fold, n_fold, n, r, "a penalised fit");
  if (folds.fold() == nullptr) {
    throw std::invalid_argument("a cross-validated penalty is handed its folds");
  }
  const std::vector<timesift::PenaltyCV> fits = timesift::penalised_cvs(
      REAL_RO(x.data()), static_cast<std::size_t>(n), static_cast<std::size_t>(p),
      REAL_RO(y.data()), REAL_RO(w.data()), static_cast<std::size_t>(r),
      timesift::family_from_name(family), spec, folds.fold(), folds.n_fold());
  using namespace cpp11::literals;
  cpp11::writable::list all;
  for (const timesift::PenaltyCV& cv : fits) {
    cpp11::writable::list out = give(cv.path);
    out.push_back("cv_mean"_nm = give(cv.cv_mean));
    out.push_back("cv_sd"_nm = give(cv.cv_sd));
    out.push_back("index_min"_nm = cpp11::as_sexp(static_cast<int>(cv.index_min) + 1));
    out.push_back("index_1se"_nm = cpp11::as_sexp(static_cast<int>(cv.index_1se) + 1));
    out.push_back("fold_stalled"_nm = give(cv.fold_stalled));
    all.push_back(out);
  }
  return all;
}

[[cpp11::register]]
cpp11::doubles ts_penalised_predict_(cpp11::doubles lambda, cpp11::doubles a0,
                                     cpp11::doubles beta, std::string family, double at,
                                     cpp11::doubles newx, int n) {
  const timesift::PenaltyPath path = take(lambda, a0, beta, family);
  std::vector<double> out(static_cast<std::size_t>(n));
  timesift::penalised_predict(path, at, REAL_RO(newx.data()), static_cast<std::size_t>(n), out.data());
  return give(out);
}

[[cpp11::register]]
cpp11::doubles ts_penalised_coef_(cpp11::doubles lambda, cpp11::doubles a0, cpp11::doubles beta,
                                  std::string family, double at) {
  const timesift::PenaltyPath path = take(lambda, a0, beta, family);
  std::vector<double> coef(path.n_column + 1, 0.0);
  timesift::penalised_coef(path, at, coef.data(), coef.data() + 1);
  return give(coef);
}

// The tree, from the same core the Python side calls. A fitted tree crosses into R as a list of
// plain vectors, one per field of the node table and the complexity table, and comes back the
// same way to be pruned or to predict.
namespace {

cpp11::writable::integers give(const std::vector<std::int8_t>& from) {
  cpp11::writable::integers out(static_cast<R_xlen_t>(from.size()));
  for (std::size_t i = 0; i < from.size(); ++i) out[static_cast<R_xlen_t>(i)] = from[i];
  return out;
}

cpp11::writable::list give(const timesift::Tree& tree) {
  using namespace cpp11::literals;
  return cpp11::writable::list({
    "family"_nm = cpp11::as_sexp(std::string(timesift::family_name(tree.family))),
    "number"_nm = give(tree.number),
    "column"_nm = give(tree.column),
    "threshold"_nm = give(tree.threshold),
    "less_left"_nm = give(tree.less_left),
    "left"_nm = give(tree.left),
    "right"_nm = give(tree.right),
    "n"_nm = give(tree.n),
    "weight"_nm = give(tree.weight),
    "risk"_nm = give(tree.risk),
    "complexity"_nm = give(tree.complexity),
    "value"_nm = give(tree.value),
    "root_risk"_nm = cpp11::as_sexp(tree.root_risk),
    "cp"_nm = give(tree.cp),
    "nsplit"_nm = give(tree.nsplit),
    "rel_error"_nm = give(tree.rel_error),
    "xerror"_nm = give(tree.xerror),
    "xstd"_nm = give(tree.xstd)
  });
}

template <typename T>
std::vector<T> take_field(const cpp11::list& tree, const char* name) {
  cpp11::sexp field = tree[name];
  std::vector<T> out;
  if (TYPEOF(field) == INTSXP) {
    cpp11::integers v(field);
    for (R_xlen_t i = 0; i < v.size(); ++i) out.push_back(static_cast<T>(v[i]));
  } else {
    cpp11::doubles v(field);
    for (R_xlen_t i = 0; i < v.size(); ++i) out.push_back(static_cast<T>(v[i]));
  }
  return out;
}

timesift::Tree take_tree(const cpp11::list& tree) {
  timesift::Tree out;
  out.family = timesift::family_from_name(cpp11::as_cpp<std::string>(tree["family"]));
  out.number = take_field<std::int32_t>(tree, "number");
  out.column = take_field<std::int32_t>(tree, "column");
  out.threshold = take_field<double>(tree, "threshold");
  out.less_left = take_field<std::int8_t>(tree, "less_left");
  out.left = take_field<std::int32_t>(tree, "left");
  out.right = take_field<std::int32_t>(tree, "right");
  out.n = take_field<std::int32_t>(tree, "n");
  out.weight = take_field<double>(tree, "weight");
  out.risk = take_field<double>(tree, "risk");
  out.complexity = take_field<double>(tree, "complexity");
  out.value = take_field<double>(tree, "value");
  out.root_risk = cpp11::as_cpp<double>(tree["root_risk"]);
  out.cp = take_field<double>(tree, "cp");
  out.nsplit = take_field<std::int32_t>(tree, "nsplit");
  out.rel_error = take_field<double>(tree, "rel_error");
  out.xerror = take_field<double>(tree, "xerror");
  out.xstd = take_field<double>(tree, "xstd");
  return out;
}

}  // namespace

// `y` and `w` are [n, r], one tree per column, and `fold` [n, r] where given.
[[cpp11::register]]
cpp11::list ts_tree_fit_(cpp11::doubles x, cpp11::doubles y, cpp11::doubles w, int n, int p,
                         int r, std::string family, int min_split, int min_leaf, double cp,
                         int max_depth, double shrink, cpp11::sexp fold, cpp11::integers n_fold,
                         int threads) {
  timesift::TreeSpec spec;
  spec.min_split = min_split;
  spec.min_leaf = min_leaf;
  spec.cp = cp;
  spec.max_depth = max_depth;
  spec.shrink = shrink;
  const ResponseFolds folds = take_response_folds(fold, n_fold, n, r, "a tree");
  const std::vector<timesift::Tree> trees = timesift::tree_fits(
      REAL_RO(x.data()), static_cast<std::size_t>(n), static_cast<std::size_t>(p),
      REAL_RO(y.data()), REAL_RO(w.data()), static_cast<std::size_t>(r),
      timesift::family_from_name(family), spec, folds.fold(), folds.n_fold(), threads);
  cpp11::writable::list out;
  for (const timesift::Tree& tree : trees) out.push_back(give(tree));
  return out;
}

[[cpp11::register]]
cpp11::list ts_tree_prune_(cpp11::list tree, double cp) {
  return give(timesift::tree_prune(take_tree(tree), cp));
}

[[cpp11::register]]
cpp11::doubles ts_tree_predict_(cpp11::list tree, cpp11::doubles newx, int n, int p) {
  std::vector<double> out(static_cast<std::size_t>(n));
  timesift::tree_predict(take_tree(tree), REAL_RO(newx.data()), static_cast<std::size_t>(n),
                         static_cast<std::size_t>(p), out.data());
  return give(out);
}

// A forest and a boosted fit cross the same way: one vector per field of their node table, the
// trees one after another, and the offset of each tree's first node.
namespace {

void give_table(cpp11::writable::list& out, const timesift::TreeTable& table) {
  using namespace cpp11::literals;
  out.push_back("offset"_nm = give(table.offset));
  out.push_back("column"_nm = give(table.column));
  out.push_back("threshold"_nm = give(table.threshold));
  out.push_back("less_left"_nm = give(table.less_left));
  out.push_back("left"_nm = give(table.left));
  out.push_back("right"_nm = give(table.right));
  out.push_back("value"_nm = give(table.value));
}

timesift::TreeTable take_table(const cpp11::list& from) {
  timesift::TreeTable table;
  table.offset = take_field<std::int32_t>(from, "offset");
  table.column = take_field<std::int32_t>(from, "column");
  table.threshold = take_field<double>(from, "threshold");
  table.less_left = take_field<std::int8_t>(from, "less_left");
  table.left = take_field<std::int32_t>(from, "left");
  table.right = take_field<std::int32_t>(from, "right");
  table.value = take_field<double>(from, "value");
  return table;
}

}  // namespace

// `y` and `w` are [n, r], one forest per column, each under its own seed.
[[cpp11::register]]
cpp11::list ts_forest_fit_(cpp11::doubles x, cpp11::doubles y, cpp11::doubles w, int n, int p,
                           int r, std::string family, int trees, int mtry, int min_leaf,
                           bool balance, cpp11::doubles seeds, int threads) {
  using namespace cpp11::literals;
  timesift::ForestSpec spec;
  spec.trees = trees;
  spec.mtry = mtry;
  spec.min_leaf = min_leaf;
  spec.balance = balance;
  spec.threads = threads;
  const std::vector<std::uint32_t> seed = take_seeds(seeds, r, "a forest");
  const std::vector<timesift::Forest> forests = timesift::forest_fits(
      REAL_RO(x.data()), static_cast<std::size_t>(n), static_cast<std::size_t>(p),
      REAL_RO(y.data()), REAL_RO(w.data()), static_cast<std::size_t>(r),
      timesift::family_from_name(family), spec, seed.data());
  cpp11::writable::list all;
  for (const timesift::Forest& forest : forests) {
    cpp11::writable::list out({
      "family"_nm = cpp11::as_sexp(std::string(timesift::family_name(forest.family))),
      "n_column"_nm = cpp11::as_sexp(forest.n_column)
    });
    give_table(out, forest.trees);
    all.push_back(out);
  }
  return all;
}

[[cpp11::register]]
cpp11::doubles ts_forest_predict_(cpp11::list forest, cpp11::doubles newx, int n, int p) {
  timesift::Forest f;
  f.family = timesift::family_from_name(cpp11::as_cpp<std::string>(forest["family"]));
  f.n_column = cpp11::as_cpp<int>(forest["n_column"]);
  f.trees = take_table(forest);
  std::vector<double> out(static_cast<std::size_t>(n));
  timesift::forest_predict(f, REAL_RO(newx.data()), static_cast<std::size_t>(n),
                           static_cast<std::size_t>(p), out.data());
  return give(out);
}

// `y` and `w` are [n, r], one boosted fit per column, each under its own seed and, where given,
// its own column of `fold` [n, r].
[[cpp11::register]]
cpp11::list ts_boost_fit_(cpp11::doubles x, cpp11::doubles y, cpp11::doubles w, int n, int p,
                          int r, std::string family, int trees, int depth, double shrinkage,
                          double min_leaf, double subsample, double colsample, bool newton,
                          double lambda, double gamma, cpp11::doubles seeds, cpp11::sexp fold,
                          cpp11::integers n_fold, int threads) {
  using namespace cpp11::literals;
  timesift::BoostSpec spec;
  spec.trees = trees;
  spec.depth = depth;
  spec.shrinkage = shrinkage;
  spec.min_leaf = min_leaf;
  spec.subsample = subsample;
  spec.colsample = colsample;
  spec.newton = newton;
  spec.lambda = lambda;
  spec.gamma = gamma;
  spec.threads = threads;
  const std::vector<std::uint32_t> seed = take_seeds(seeds, r, "a boosted fit");
  const ResponseFolds folds = take_response_folds(fold, n_fold, n, r, "a boosted fit");
  const std::vector<timesift::Boosted> fits = timesift::boost_fits(
      REAL_RO(x.data()), static_cast<std::size_t>(n), static_cast<std::size_t>(p),
      REAL_RO(y.data()), REAL_RO(w.data()), static_cast<std::size_t>(r),
      timesift::family_from_name(family), spec, seed.data(), folds.fold(), folds.n_fold());
  cpp11::writable::list all;
  for (const timesift::Boosted& fit : fits) {
    cpp11::writable::list out({
      "family"_nm = cpp11::as_sexp(std::string(timesift::family_name(fit.family))),
      "n_column"_nm = cpp11::as_sexp(fit.n_column),
      "init"_nm = cpp11::as_sexp(fit.init),
      "cv_error"_nm = give(fit.cv_error)
    });
    give_table(out, fit.trees);
    all.push_back(out);
  }
  return all;
}

[[cpp11::register]]
cpp11::doubles ts_boost_predict_(cpp11::list fit, cpp11::doubles newx, int n, int p) {
  timesift::Boosted b;
  b.family = timesift::family_from_name(cpp11::as_cpp<std::string>(fit["family"]));
  b.n_column = cpp11::as_cpp<int>(fit["n_column"]);
  b.init = cpp11::as_cpp<double>(fit["init"]);
  b.trees = take_table(fit);
  std::vector<double> out(static_cast<std::size_t>(n));
  timesift::boost_predict(b, REAL_RO(newx.data()), static_cast<std::size_t>(n),
                          static_cast<std::size_t>(p), out.data());
  return give(out);
}

[[cpp11::register]]
cpp11::doubles ts_forest_stream_(double seed, double tree, int n) {
  std::vector<std::uint32_t> raw(static_cast<std::size_t>(n));
  timesift::forest_stream(take_seed(seed), take_seed(tree), raw.size(), raw.data());
  return give(std::vector<double>(raw.begin(), raw.end()));
}

// maxnet, from the same core the Python side calls. A fit crosses into R as a list of plain
// vectors, the features it gave a coefficient one field each, and comes back the same way to
// predict.
namespace {

timesift::MaxnetSpec maxnet_spec(const std::string& classes, int knots, double regmult,
                                 const std::string& formulation, bool add_samples, double thresh,
                                 double max_pass, int n_lambda, bool one_se, int threads,
                                 double max_design) {
  timesift::MaxnetSpec spec;
  spec.classes = classes;
  spec.knots = knots;
  spec.regmult = regmult;
  spec.formulation = timesift::maxnet_formulation_from_name(formulation);
  spec.add_samples = add_samples;
  spec.thresh = thresh;
  spec.max_pass = static_cast<int>(max_pass);
  spec.n_lambda = n_lambda;
  spec.one_se = one_se;
  spec.threads = threads;
  spec.max_design = max_design;
  return spec;
}

void give_features(cpp11::writable::list& out, const timesift::MaxnetFeatures& f) {
  using namespace cpp11::literals;
  out.push_back("kind"_nm = give(f.kind));
  out.push_back("a"_nm = give(f.a));
  out.push_back("b"_nm = give(f.b));
  out.push_back("lo"_nm = give(f.lo));
  out.push_back("hi"_nm = give(f.hi));
}

timesift::MaxnetFeatures take_features(const cpp11::list& fit) {
  timesift::MaxnetFeatures f;
  f.kind = take_field<std::int8_t>(fit, "kind");
  f.a = take_field<std::int32_t>(fit, "a");
  f.b = take_field<std::int32_t>(fit, "b");
  f.lo = take_field<double>(fit, "lo");
  f.hi = take_field<double>(fit, "hi");
  return f;
}

}  // namespace

[[cpp11::register]]
cpp11::list ts_maxnet_design_(cpp11::doubles x, cpp11::doubles y, int n, int p,
                              std::string classes, int knots, double regmult,
                              std::string formulation, bool add_samples, double max_design) {
  const timesift::MaxnetSpec spec = maxnet_spec(classes, knots, regmult, formulation, add_samples,
                                                1e-8, 1e6, 100, false, 1, max_design);
  const timesift::MaxnetDesign d =
      timesift::maxnet_design(REAL_RO(x.data()), REAL_RO(y.data()), static_cast<std::size_t>(n),
                              static_cast<std::size_t>(p), spec);
  using namespace cpp11::literals;
  std::vector<std::int32_t> rows(d.rows.begin(), d.rows.end());
  cpp11::writable::list out({
    "classes"_nm = cpp11::as_sexp(d.classes),
    "rows"_nm = give(rows),
    "y"_nm = give(d.y),
    "reg"_nm = give(d.reg)
  });
  give_features(out, d.features);
  return out;
}

[[cpp11::register]]
cpp11::list ts_maxnet_fit_(cpp11::doubles x, cpp11::doubles y, cpp11::doubles w, int n, int p,
                           int r, std::string classes, int knots, double regmult,
                           std::string formulation, bool add_samples, double thresh,
                           double max_pass, int n_lambda, bool one_se, cpp11::sexp fold,
                           cpp11::integers n_fold, int threads, double max_design) {
  const timesift::MaxnetSpec spec = maxnet_spec(classes, knots, regmult, formulation, add_samples,
                                                thresh, max_pass, n_lambda, one_se, threads,
                                                max_design);
  const ResponseFolds folds = take_response_folds(fold, n_fold, n, r, "maxnet");
  const std::vector<timesift::Maxnet> fits = timesift::maxnet_fit(
      REAL_RO(x.data()), static_cast<std::size_t>(n), static_cast<std::size_t>(p),
      REAL_RO(y.data()), REAL_RO(w.data()), static_cast<std::size_t>(r), spec, folds.fold(),
      folds.n_fold());
  using namespace cpp11::literals;
  cpp11::writable::list all;
  for (const timesift::Maxnet& fit : fits) {
    cpp11::writable::list out({
      "formulation"_nm = cpp11::as_sexp(std::string(timesift::maxnet_formulation_name(fit.formulation))),
      "classes"_nm = cpp11::as_sexp(fit.classes),
      "n_column"_nm = cpp11::as_sexp(fit.n_column),
      "n_presence"_nm = cpp11::as_sexp(fit.n_presence),
      "n_feature"_nm = cpp11::as_sexp(fit.n_feature),
      "var_min"_nm = give(fit.var_min),
      "var_max"_nm = give(fit.var_max),
      "feature_min"_nm = give(fit.feature_min),
      "feature_max"_nm = give(fit.feature_max),
      "beta"_nm = give(fit.beta),
      "intercept"_nm = cpp11::as_sexp(fit.intercept),
      "lasso_intercept"_nm = cpp11::as_sexp(fit.lasso_intercept),
      "entropy"_nm = cpp11::as_sexp(fit.entropy),
      "lambda"_nm = cpp11::as_sexp(fit.lambda),
      "stalled"_nm = cpp11::as_sexp(fit.stalled),
      "fold_stalled"_nm = cpp11::as_sexp(fit.fold_stalled)
    });
    give_features(out, fit.features);
    all.push_back(out);
  }
  return all;
}

[[cpp11::register]]
cpp11::doubles ts_maxnet_predict_(cpp11::list fit, cpp11::doubles newx, int n, int p, bool clamp,
                                  std::string type) {
  timesift::Maxnet m;
  m.formulation =
      timesift::maxnet_formulation_from_name(cpp11::as_cpp<std::string>(fit["formulation"]));
  m.n_column = cpp11::as_cpp<int>(fit["n_column"]);
  m.var_min = take_field<double>(fit, "var_min");
  m.var_max = take_field<double>(fit, "var_max");
  m.features = take_features(fit);
  m.feature_min = take_field<double>(fit, "feature_min");
  m.feature_max = take_field<double>(fit, "feature_max");
  m.beta = take_field<double>(fit, "beta");
  m.intercept = cpp11::as_cpp<double>(fit["intercept"]);
  m.entropy = cpp11::as_cpp<double>(fit["entropy"]);
  std::vector<double> out(static_cast<std::size_t>(n));
  timesift::maxnet_predict(m, REAL_RO(newx.data()), static_cast<std::size_t>(n),
                           static_cast<std::size_t>(p), clamp,
                           timesift::maxnet_output_from_name(type), out.data());
  return give(out);
}

// The envelope, from the same core the Python side calls: each column's band over the presences.
[[cpp11::register]]
cpp11::list ts_envelope_fit_(cpp11::doubles x, cpp11::doubles y, int n, int p, double quantile) {
  const timesift::Envelope fit =
      timesift::envelope_fit(REAL_RO(x.data()), REAL_RO(y.data()), static_cast<std::size_t>(n),
                             static_cast<std::size_t>(p), quantile);
  using namespace cpp11::literals;
  return cpp11::writable::list({
    "n_column"_nm = cpp11::as_sexp(fit.n_column),
    "n_presence"_nm = cpp11::as_sexp(fit.n_presence),
    "lo"_nm = give(fit.lo),
    "hi"_nm = give(fit.hi)
  });
}

[[cpp11::register]]
cpp11::doubles ts_envelope_predict_(cpp11::list fit, cpp11::doubles newx, int n, int p) {
  timesift::Envelope e;
  e.n_column = cpp11::as_cpp<int>(fit["n_column"]);
  e.n_presence = cpp11::as_cpp<int>(fit["n_presence"]);
  e.lo = take_field<double>(fit, "lo");
  e.hi = take_field<double>(fit, "hi");
  std::vector<double> out(static_cast<std::size_t>(n));
  timesift::envelope_predict(e, REAL_RO(newx.data()), static_cast<std::size_t>(n),
                             static_cast<std::size_t>(p), out.data());
  return give(out);
}

// The stepwise model, from the same core the Python side calls. A fit crosses into R as a list of
// plain vectors, its terms one field each, and comes back the same way to predict.
// `y` and `w` are [n, r], one search per column.
[[cpp11::register]]
cpp11::list ts_stepwise_fit_(cpp11::doubles x, cpp11::doubles y, cpp11::doubles w, int n, int p,
                             int r, std::string family, double max_terms, int degree,
                             std::string direction, std::string terms, int threads) {
  timesift::StepwiseSpec spec;
  spec.family = timesift::family_from_name(family);
  spec.max_terms = max_terms;
  spec.degree = degree;
  spec.direction = timesift::step_direction_from_name(direction);
  spec.terms = timesift::step_terms_from_name(terms);
  spec.threads = threads;
  const std::vector<timesift::Stepwise> fits = timesift::stepwise_fits(
      REAL_RO(x.data()), static_cast<std::size_t>(n), static_cast<std::size_t>(p),
      REAL_RO(y.data()), REAL_RO(w.data()), static_cast<std::size_t>(r), spec);
  using namespace cpp11::literals;
  cpp11::writable::list all;
  for (const timesift::Stepwise& fit : fits) {
    all.push_back(cpp11::writable::list({
      "family"_nm = cpp11::as_sexp(std::string(timesift::family_name(fit.family))),
      "n_column"_nm = cpp11::as_sexp(fit.n_column),
      "constant"_nm = cpp11::as_sexp(fit.constant),
      "term_column"_nm = give(fit.term_column),
      "term_power"_nm = give(fit.term_power),
      "term_degree"_nm = give(fit.term_degree),
      "alpha"_nm = give(fit.alpha),
      "norm2"_nm = give(fit.norm2),
      "beta"_nm = give(fit.beta),
      "rank"_nm = cpp11::as_sexp(fit.rank),
      "deviance"_nm = cpp11::as_sexp(fit.deviance),
      "aic"_nm = cpp11::as_sexp(fit.aic),
      "converged"_nm = cpp11::as_sexp(fit.converged),
      "steps"_nm = cpp11::as_sexp(fit.steps)
    }));
  }
  return all;
}

[[cpp11::register]]
cpp11::doubles ts_stepwise_predict_(cpp11::list fit, cpp11::doubles newx, int n, int p) {
  timesift::Stepwise s;
  s.family = timesift::family_from_name(cpp11::as_cpp<std::string>(fit["family"]));
  s.n_column = cpp11::as_cpp<int>(fit["n_column"]);
  s.constant = cpp11::as_cpp<double>(fit["constant"]);
  s.term_column = take_field<std::int32_t>(fit, "term_column");
  s.term_power = take_field<std::int32_t>(fit, "term_power");
  s.term_degree = take_field<std::int32_t>(fit, "term_degree");
  s.alpha = take_field<double>(fit, "alpha");
  s.norm2 = take_field<double>(fit, "norm2");
  s.beta = take_field<double>(fit, "beta");
  std::vector<double> out(static_cast<std::size_t>(n));
  timesift::stepwise_predict(s, REAL_RO(newx.data()), static_cast<std::size_t>(n),
                             static_cast<std::size_t>(p), out.data());
  return give(out);
}

// MARS, from the same core the Python side calls. A fit crosses into R as a list of plain vectors:
// every term of the forward pass as its factors, the terms kept, and their coefficients.
// `y` and `w` are [n, r], one fit per column.
[[cpp11::register]]
cpp11::list ts_mars_fit_(cpp11::doubles x, cpp11::doubles y, cpp11::doubles w, int n, int p,
                         int r, std::string family, int degree, double penalty, int nk,
                         double thresh, int minspan, int endspan, int fast_k, double fast_beta,
                         bool prune, int nprune, int threads) {
  timesift::MarsSpec spec;
  spec.family = timesift::family_from_name(family);
  spec.degree = degree;
  spec.penalty = std::isnan(penalty) ? std::numeric_limits<double>::quiet_NaN() : penalty;
  spec.nk = nk;
  spec.thresh = thresh;
  spec.minspan = minspan;
  spec.endspan = endspan;
  spec.fast_k = fast_k;
  spec.fast_beta = fast_beta;
  spec.prune = prune;
  spec.nprune = nprune;
  spec.threads = threads;
  const std::vector<timesift::Mars> fits = timesift::mars_fits(
      REAL_RO(x.data()), static_cast<std::size_t>(n), static_cast<std::size_t>(p),
      REAL_RO(y.data()), REAL_RO(w.data()), static_cast<std::size_t>(r), spec);
  using namespace cpp11::literals;
  cpp11::writable::list all;
  for (const timesift::Mars& fit : fits) {
    all.push_back(cpp11::writable::list({
      "family"_nm = cpp11::as_sexp(std::string(timesift::family_name(fit.family))),
      "n_column"_nm = cpp11::as_sexp(fit.n_column),
      "factor_start"_nm = give(fit.factor_start),
      "factor_column"_nm = give(fit.factor_column),
      "factor_dir"_nm = give(fit.factor_dir),
      "factor_cut"_nm = give(fit.factor_cut),
      "selected"_nm = give(fit.selected),
      "beta"_nm = give(fit.beta),
      "termcond"_nm = cpp11::as_sexp(fit.termcond),
      "gcv"_nm = cpp11::as_sexp(fit.gcv),
      "converged"_nm = cpp11::as_sexp(fit.converged)
    }));
  }
  return all;
}

[[cpp11::register]]
cpp11::doubles ts_mars_predict_(cpp11::list fit, cpp11::doubles newx, int n, int p) {
  timesift::Mars m;
  m.family = timesift::family_from_name(cpp11::as_cpp<std::string>(fit["family"]));
  m.n_column = cpp11::as_cpp<int>(fit["n_column"]);
  m.factor_start = take_field<std::int32_t>(fit, "factor_start");
  m.factor_column = take_field<std::int32_t>(fit, "factor_column");
  m.factor_dir = take_field<std::int32_t>(fit, "factor_dir");
  m.factor_cut = take_field<double>(fit, "factor_cut");
  m.selected = take_field<std::int32_t>(fit, "selected");
  m.beta = take_field<double>(fit, "beta");
  std::vector<double> out(static_cast<std::size_t>(n));
  timesift::mars_predict(m, REAL_RO(newx.data()), static_cast<std::size_t>(n),
                         static_cast<std::size_t>(p), out.data());
  return give(out);
}

// The one-hidden-layer network, from the same core the Python side calls. A fit crosses into R as
// a list of plain values: its size, its family, the centre and scale of its columns and its
// weights. `y` and `w` are [n, r], one network per column.
[[cpp11::register]]
cpp11::list ts_perceptron_fit_(cpp11::doubles x, cpp11::doubles y, cpp11::doubles w, int n, int p,
                               int r, cpp11::integers seeds, std::string family, int hidden,
                               bool skip, bool standardise, double decay, double range,
                               int max_iter, double abs_tol, double rel_tol, int threads,
                               cpp11::doubles start) {
  timesift::PerceptronSpec spec;
  spec.family = timesift::family_from_name(family);
  spec.hidden = hidden;
  spec.skip = skip;
  spec.standardise = standardise;
  spec.decay = decay;
  spec.range = range;
  spec.max_iter = max_iter;
  spec.abs_tol = abs_tol;
  spec.rel_tol = rel_tol;
  std::vector<std::uint32_t> seed(seeds.begin(), seeds.end());
  if (seed.size() != static_cast<std::size_t>(r)) {
    throw std::invalid_argument("a network is fitted under one seed per response");
  }
  const std::vector<double> init(start.begin(), start.end());
  const std::vector<timesift::Perceptron> fits = timesift::perceptron_fit(
      REAL_RO(x.data()), static_cast<std::size_t>(n), static_cast<std::size_t>(p),
      REAL_RO(y.data()), REAL_RO(w.data()), static_cast<std::size_t>(r), seed.data(), spec,
      threads, init);
  using namespace cpp11::literals;
  cpp11::writable::list out;
  for (const timesift::Perceptron& fit : fits) {
    out.push_back(cpp11::writable::list({
      "family"_nm = cpp11::as_sexp(std::string(timesift::family_name(fit.family))),
      "n_column"_nm = cpp11::as_sexp(fit.n_column),
      "hidden"_nm = cpp11::as_sexp(fit.hidden),
      "skip"_nm = cpp11::as_sexp(fit.skip),
      "centre"_nm = give(fit.centre),
      "scale"_nm = give(fit.scale),
      "weights"_nm = give(fit.weights),
      "value"_nm = cpp11::as_sexp(fit.value),
      "iterations"_nm = cpp11::as_sexp(fit.iterations),
      "converged"_nm = cpp11::as_sexp(fit.converged)
    }));
  }
  return out;
}

[[cpp11::register]]
cpp11::doubles ts_perceptron_predict_(cpp11::list fit, cpp11::doubles newx, int n, int p) {
  timesift::Perceptron m;
  m.family = timesift::family_from_name(cpp11::as_cpp<std::string>(fit["family"]));
  m.n_column = cpp11::as_cpp<int>(fit["n_column"]);
  m.hidden = cpp11::as_cpp<int>(fit["hidden"]);
  m.skip = cpp11::as_cpp<bool>(fit["skip"]);
  m.centre = take_field<double>(fit, "centre");
  m.scale = take_field<double>(fit, "scale");
  m.weights = take_field<double>(fit, "weights");
  std::vector<double> out(static_cast<std::size_t>(n));
  timesift::perceptron_predict(m, REAL_RO(newx.data()), static_cast<std::size_t>(n),
                               static_cast<std::size_t>(p), out.data());
  return give(out);
}

// Flexible discriminant analysis, from the same core the Python side calls. A fit crosses into R as
// a list of plain vectors: the kept terms as their factors, their coefficients, the variate and the
// recalibration. `y` and `w` are [n, r], one fit per column.
[[cpp11::register]]
cpp11::list ts_fda_fit_(cpp11::doubles x, cpp11::doubles y, cpp11::doubles w, int n, int p,
                        int r, int degree, double penalty, int nk, double thresh, bool prune,
                        bool calibrate, int threads) {
  timesift::FdaSpec spec;
  spec.degree = degree;
  spec.penalty = penalty;
  spec.nk = nk;
  spec.thresh = thresh;
  spec.prune = prune;
  spec.calibrate = calibrate;
  spec.threads = threads;
  const std::vector<timesift::Fda> fits = timesift::fda_fits(
      REAL_RO(x.data()), static_cast<std::size_t>(n), static_cast<std::size_t>(p),
      REAL_RO(y.data()), REAL_RO(w.data()), static_cast<std::size_t>(r), spec);
  using namespace cpp11::literals;
  cpp11::writable::list all;
  for (const timesift::Fda& fit : fits) {
    all.push_back(cpp11::writable::list({
      "n_column"_nm = cpp11::as_sexp(fit.n_column),
      "factor_start"_nm = give(fit.factor_start),
      "factor_column"_nm = give(fit.factor_column),
      "factor_dir"_nm = give(fit.factor_dir),
      "factor_cut"_nm = give(fit.factor_cut),
      "coef"_nm = give(fit.coef),
      "forward_terms"_nm = cpp11::as_sexp(fit.forward_terms),
      "gcv"_nm = cpp11::as_sexp(fit.gcv),
      "discriminates"_nm = cpp11::as_sexp(fit.discriminates),
      "mean"_nm = cpp11::as_sexp(fit.mean),
      "direction"_nm = cpp11::as_sexp(fit.direction),
      "scale"_nm = cpp11::as_sexp(fit.scale),
      "centroid"_nm = give(std::vector<double>(fit.centroid, fit.centroid + 2)),
      "prior"_nm = give(std::vector<double>(fit.prior, fit.prior + 2)),
      "calibrated"_nm = cpp11::as_sexp(fit.calibrated),
      "calibration"_nm = give(std::vector<double>(fit.calibration, fit.calibration + 2)),
      "converged"_nm = cpp11::as_sexp(fit.converged)
    }));
  }
  return all;
}

[[cpp11::register]]
cpp11::doubles ts_fda_predict_(cpp11::list fit, cpp11::doubles newx, int n, int p) {
  timesift::Fda f;
  f.n_column = cpp11::as_cpp<int>(fit["n_column"]);
  f.factor_start = take_field<std::int32_t>(fit, "factor_start");
  f.factor_column = take_field<std::int32_t>(fit, "factor_column");
  f.factor_dir = take_field<std::int32_t>(fit, "factor_dir");
  f.factor_cut = take_field<double>(fit, "factor_cut");
  f.coef = take_field<double>(fit, "coef");
  f.discriminates = cpp11::as_cpp<bool>(fit["discriminates"]);
  f.mean = cpp11::as_cpp<double>(fit["mean"]);
  f.direction = cpp11::as_cpp<double>(fit["direction"]);
  f.scale = cpp11::as_cpp<double>(fit["scale"]);
  const std::vector<double> centroid = take_field<double>(fit, "centroid");
  const std::vector<double> prior = take_field<double>(fit, "prior");
  const std::vector<double> calibration = take_field<double>(fit, "calibration");
  for (int j = 0; j < 2; ++j) {
    f.centroid[j] = centroid[j];
    f.prior[j] = prior[j];
    f.calibration[j] = calibration[j];
  }
  f.calibrated = cpp11::as_cpp<bool>(fit["calibrated"]);
  std::vector<double> out(static_cast<std::size_t>(n));
  timesift::fda_predict(f, REAL_RO(newx.data()), static_cast<std::size_t>(n),
                        static_cast<std::size_t>(p), out.data());
  return give(out);
}

// The additive model, from the same core the Python side calls. A fit crosses into R as a list of
// plain vectors: every column's term, shared by the responses, and each response's coefficients,
// smoothing parameters and effective degrees of freedom.
[[cpp11::register]]
cpp11::list ts_additive_fit_(cpp11::doubles x, cpp11::doubles y, cpp11::doubles w, int n, int p,
                             int r, std::string family, int k, double gamma, int max_knots,
                             int threads, cpp11::doubles sp) {
  timesift::AdditiveSpec spec;
  spec.family = timesift::family_from_name(family);
  spec.k = k;
  spec.gamma = gamma;
  spec.max_knots = max_knots;
  spec.threads = threads;
  spec.sp.assign(sp.begin(), sp.end());
  const timesift::Additive fit = timesift::additive_fit(
      REAL_RO(x.data()), static_cast<std::size_t>(n), static_cast<std::size_t>(p),
      REAL_RO(y.data()), REAL_RO(w.data()), static_cast<std::size_t>(r), spec);
  using namespace cpp11::literals;
  return cpp11::writable::list({
    "family"_nm = cpp11::as_sexp(std::string(timesift::family_name(fit.family))),
    "n_column"_nm = cpp11::as_sexp(fit.n_column),
    "n_coef"_nm = cpp11::as_sexp(fit.n_coef),
    "term_column"_nm = give(fit.term_column),
    "term_basis"_nm = give(fit.term_basis),
    "term_size"_nm = give(fit.term_size),
    "term_penalised"_nm = give(fit.term_penalised),
    "term_shift"_nm = give(fit.term_shift),
    "knot_start"_nm = give(fit.knot_start),
    "knots"_nm = give(fit.knots),
    "radial_start"_nm = give(fit.radial_start),
    "radial"_nm = give(fit.radial),
    "map_start"_nm = give(fit.map_start),
    "map"_nm = give(fit.map),
    "penalty"_nm = give(fit.penalty),
    "aliased"_nm = give(fit.aliased),
    "n_response"_nm = cpp11::as_sexp(fit.n_response),
    "beta"_nm = give(fit.beta),
    "sp"_nm = give(fit.sp),
    "edf"_nm = give(fit.edf),
    "score"_nm = give(fit.score),
    "outer"_nm = give(fit.outer),
    "converged"_nm = give(fit.converged)
  });
}

[[cpp11::register]]
cpp11::doubles ts_additive_predict_(cpp11::list fit, cpp11::doubles newx, int n, int p) {
  timesift::Additive a;
  a.family = timesift::family_from_name(cpp11::as_cpp<std::string>(fit["family"]));
  a.n_column = cpp11::as_cpp<int>(fit["n_column"]);
  a.n_coef = cpp11::as_cpp<int>(fit["n_coef"]);
  a.term_column = take_field<std::int32_t>(fit, "term_column");
  a.term_basis = take_field<std::int32_t>(fit, "term_basis");
  a.term_size = take_field<std::int32_t>(fit, "term_size");
  a.term_penalised = take_field<std::int32_t>(fit, "term_penalised");
  a.term_shift = take_field<double>(fit, "term_shift");
  a.knot_start = take_field<std::int32_t>(fit, "knot_start");
  a.knots = take_field<double>(fit, "knots");
  a.radial_start = take_field<std::int32_t>(fit, "radial_start");
  a.radial = take_field<double>(fit, "radial");
  a.map_start = take_field<std::int32_t>(fit, "map_start");
  a.map = take_field<double>(fit, "map");
  a.n_response = cpp11::as_cpp<int>(fit["n_response"]);
  a.beta = take_field<double>(fit, "beta");
  std::vector<double> out(static_cast<std::size_t>(n) * static_cast<std::size_t>(a.n_response));
  timesift::additive_predict(a, REAL_RO(newx.data()), static_cast<std::size_t>(n),
                             static_cast<std::size_t>(p), out.data());
  return give(out);
}

// The hierarchical model, from the same core the Python side calls. A fit crosses into R as a list
// of plain vectors and comes back the same way to predict.
namespace {

timesift::Field field_from_name(const std::string& name) {
  if (name == "none") return timesift::Field::none;
  if (name == "hsgp") return timesift::Field::hsgp;
  if (name == "nngp") return timesift::Field::nngp;
  cpp11::stop("a hierarchical field is none, hsgp or nngp, not '%s'.", name.c_str());
}

const char* field_name(timesift::Field f) {
  switch (f) {
    case timesift::Field::none: return "none";
    case timesift::Field::hsgp: return "hsgp";
    case timesift::Field::nngp: return "nngp";
  }
  return "none";
}

std::vector<std::int32_t> take_units(cpp11::sexp unit) {
  std::vector<std::int32_t> units;
  if (unit != R_NilValue) {
    cpp11::integers u(unit);
    for (R_xlen_t i = 0; i < u.size(); ++i) units.push_back(u[i]);
  }
  return units;
}

}  // namespace

[[cpp11::register]]
cpp11::list ts_hierarchical_fit_(cpp11::doubles x, cpp11::doubles y, cpp11::doubles w, int n,
                                 int p, cpp11::sexp unit, int n_unit, cpp11::sexp coords,
                                 std::string field, double beta_sd, double sd_u,
                                 double sd_alpha, double range_fraction, double range_alpha, int m,
                                 double boundary, int neighbours, int cov, int nodes, double step,
                                 int threads, cpp11::sexp theta) {
  timesift::HierSpec spec;
  spec.beta_sd = beta_sd;
  spec.unit = unit != R_NilValue;
  spec.field = field_from_name(field);
  spec.sd_u = sd_u;
  spec.sd_alpha = sd_alpha;
  spec.range_fraction = range_fraction;
  spec.range_alpha = range_alpha;
  spec.m = m;
  spec.boundary = boundary;
  spec.neighbours = neighbours;
  spec.cov = cov;
  spec.nodes = nodes;
  spec.step = step;
  spec.threads = threads;
  if (theta != R_NilValue) {
    cpp11::doubles t(theta);
    for (R_xlen_t i = 0; i < t.size(); ++i) spec.theta.push_back(t[i]);
  }
  const std::vector<std::int32_t> units = take_units(unit);
  const double* xy = coords == R_NilValue ? nullptr : REAL_RO(coords);
  const timesift::Hierarchical fit = timesift::hierarchical_fit(
      REAL_RO(x.data()), static_cast<std::size_t>(n), static_cast<std::size_t>(p),
      REAL_RO(y.data()), REAL_RO(w.data()), units.empty() ? nullptr : units.data(),
      static_cast<std::size_t>(n_unit), xy, spec);
  using namespace cpp11::literals;
  return cpp11::writable::list({
    "field"_nm = cpp11::as_sexp(std::string(field_name(fit.field))),
    "n_column"_nm = cpp11::as_sexp(fit.n_column),
    "n_unit"_nm = cpp11::as_sexp(fit.n_unit),
    "n_theta"_nm = cpp11::as_sexp(fit.n_theta),
    "beta"_nm = give(fit.beta),
    "unit_effect"_nm = give(fit.unit_effect),
    "theta_hat"_nm = give(fit.theta_hat),
    "log_marginal"_nm = cpp11::as_sexp(fit.log_marginal),
    "n_node"_nm = cpp11::as_sexp(fit.n_node),
    "node_theta"_nm = give(fit.node_theta),
    "node_weight"_nm = give(fit.node_weight),
    "node_log_post"_nm = give(fit.node_log_post),
    "node_field"_nm = give(fit.node_field),
    "n_field"_nm = cpp11::as_sexp(fit.n_field),
    "converged"_nm = cpp11::as_sexp(fit.converged),
    "centre"_nm = give(std::vector<double>(fit.centre, fit.centre + 2)),
    "scale"_nm = cpp11::as_sexp(fit.scale),
    "m"_nm = cpp11::as_sexp(fit.m),
    "box_centre"_nm = give(std::vector<double>(fit.box_centre, fit.box_centre + 2)),
    "box_half"_nm = give(std::vector<double>(fit.box_half, fit.box_half + 2)),
    "location"_nm = give(fit.location),
    "n_location"_nm = cpp11::as_sexp(fit.n_location),
    "neighbours"_nm = cpp11::as_sexp(fit.neighbours),
    "cov"_nm = cpp11::as_sexp(fit.cov)
  });
}

[[cpp11::register]]
cpp11::doubles ts_hierarchical_predict_(cpp11::list fit, cpp11::doubles newx, int n, int p,
                                        cpp11::sexp unit, cpp11::sexp coords) {
  timesift::Hierarchical h;
  h.field = field_from_name(cpp11::as_cpp<std::string>(fit["field"]));
  h.n_column = cpp11::as_cpp<int>(fit["n_column"]);
  h.n_unit = cpp11::as_cpp<int>(fit["n_unit"]);
  h.n_theta = cpp11::as_cpp<int>(fit["n_theta"]);
  h.beta = take_field<double>(fit, "beta");
  h.unit_effect = take_field<double>(fit, "unit_effect");
  h.n_node = cpp11::as_cpp<int>(fit["n_node"]);
  h.node_theta = take_field<double>(fit, "node_theta");
  h.node_weight = take_field<double>(fit, "node_weight");
  h.node_field = take_field<double>(fit, "node_field");
  h.n_field = cpp11::as_cpp<int>(fit["n_field"]);
  const std::vector<double> centre = take_field<double>(fit, "centre");
  const std::vector<double> box_centre = take_field<double>(fit, "box_centre");
  const std::vector<double> box_half = take_field<double>(fit, "box_half");
  for (int c = 0; c < 2; ++c) {
    h.centre[c] = centre[static_cast<std::size_t>(c)];
    h.box_centre[c] = box_centre[static_cast<std::size_t>(c)];
    h.box_half[c] = box_half[static_cast<std::size_t>(c)];
  }
  h.scale = cpp11::as_cpp<double>(fit["scale"]);
  h.m = cpp11::as_cpp<int>(fit["m"]);
  h.location = take_field<double>(fit, "location");
  h.n_location = cpp11::as_cpp<int>(fit["n_location"]);
  h.neighbours = cpp11::as_cpp<int>(fit["neighbours"]);
  h.cov = cpp11::as_cpp<int>(fit["cov"]);
  const std::vector<std::int32_t> units = take_units(unit);
  const double* xy = coords == R_NilValue ? nullptr : REAL_RO(coords);
  std::vector<double> out(static_cast<std::size_t>(n));
  timesift::hierarchical_predict(h, REAL_RO(newx.data()), static_cast<std::size_t>(n),
                                 static_cast<std::size_t>(p), units.empty() ? nullptr : units.data(),
                                 xy, out.data());
  return give(out);
}

