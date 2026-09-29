#include <cpp11.hpp>

#include <cmath>
#include <string>
#include <vector>

#include "ts_core.h"
#include "ts_envelope.h"
#include "ts_mars.h"
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

[[cpp11::register]]
cpp11::list ts_penalised_cv_(cpp11::doubles x, cpp11::doubles y, cpp11::doubles w, int n, int p,
                             std::string family, double alpha, int n_lambda,
                             double lambda_min_ratio, cpp11::sexp lambda, double thresh,
                             bool standardize, bool intercept, cpp11::integers fold, int n_fold,
                             double max_pass, int threads) {
  const timesift::PenaltySpec spec = penalty_spec(alpha, n_lambda, lambda_min_ratio, lambda,
                                                  thresh, standardize, intercept, max_pass,
                                                  threads);
  std::vector<std::int32_t> which(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) which[static_cast<std::size_t>(i)] = fold[i];
  const timesift::PenaltyCV cv =
      timesift::penalised_cv(REAL_RO(x.data()), REAL_RO(y.data()), REAL_RO(w.data()), static_cast<std::size_t>(n),
                             static_cast<std::size_t>(p), timesift::family_from_name(family),
                             spec, which.data(), n_fold);
  using namespace cpp11::literals;
  cpp11::writable::list out = give(cv.path);
  out.push_back("cv_mean"_nm = give(cv.cv_mean));
  out.push_back("cv_sd"_nm = give(cv.cv_sd));
  out.push_back("index_min"_nm = cpp11::as_sexp(static_cast<int>(cv.index_min) + 1));
  out.push_back("index_1se"_nm = cpp11::as_sexp(static_cast<int>(cv.index_1se) + 1));
  out.push_back("fold_stalled"_nm = give(cv.fold_stalled));
  return out;
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

// One 0-based fold index per observation, or none.
std::vector<std::int32_t> take_folds(cpp11::sexp fold) {
  std::vector<std::int32_t> which;
  if (fold != R_NilValue) {
    cpp11::integers given(fold);
    for (R_xlen_t i = 0; i < given.size(); ++i) which.push_back(given[i]);
  }
  return which;
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

[[cpp11::register]]
cpp11::list ts_tree_fit_(cpp11::doubles x, cpp11::doubles y, cpp11::doubles w, int n, int p,
                         std::string family, int min_split, int min_leaf, double cp,
                         int max_depth, cpp11::sexp fold, int n_fold) {
  timesift::TreeSpec spec;
  spec.min_split = min_split;
  spec.min_leaf = min_leaf;
  spec.cp = cp;
  spec.max_depth = max_depth;
  const std::vector<std::int32_t> which = take_folds(fold);
  const timesift::Tree tree = timesift::tree_fit(
      REAL_RO(x.data()), REAL_RO(y.data()), REAL_RO(w.data()), static_cast<std::size_t>(n),
      static_cast<std::size_t>(p), timesift::family_from_name(family), spec,
      which.empty() ? nullptr : which.data(), which.empty() ? 0 : n_fold);
  return give(tree);
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
// trees one after another, and the offset of each tree's first node. A seed is a number modulo
// 2^32, which an R integer cannot carry, so it crosses as a double.
namespace {

std::uint32_t take_seed(double seed) {
  return static_cast<std::uint32_t>(static_cast<std::uint64_t>(seed));
}

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

[[cpp11::register]]
cpp11::list ts_forest_fit_(cpp11::doubles x, cpp11::doubles y, cpp11::doubles w, int n, int p,
                           std::string family, int trees, int mtry, int min_leaf, bool balance,
                           double seed, int threads) {
  using namespace cpp11::literals;
  timesift::ForestSpec spec;
  spec.trees = trees;
  spec.mtry = mtry;
  spec.min_leaf = min_leaf;
  spec.balance = balance;
  spec.seed = take_seed(seed);
  spec.threads = threads;
  const timesift::Forest forest = timesift::forest_fit(
      REAL_RO(x.data()), REAL_RO(y.data()), REAL_RO(w.data()), static_cast<std::size_t>(n),
      static_cast<std::size_t>(p), timesift::family_from_name(family), spec);
  cpp11::writable::list out({
    "family"_nm = cpp11::as_sexp(std::string(timesift::family_name(forest.family))),
    "n_column"_nm = cpp11::as_sexp(forest.n_column)
  });
  give_table(out, forest.trees);
  return out;
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

[[cpp11::register]]
cpp11::list ts_boost_fit_(cpp11::doubles x, cpp11::doubles y, cpp11::doubles w, int n, int p,
                          std::string family, int trees, int depth, double shrinkage,
                          double min_leaf, double subsample, double colsample, bool newton,
                          double lambda, double gamma, double seed, cpp11::sexp fold, int n_fold,
                          int threads) {
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
  spec.seed = take_seed(seed);
  spec.threads = threads;
  const std::vector<std::int32_t> which = take_folds(fold);
  const timesift::Boosted fit = timesift::boost_fit(
      REAL_RO(x.data()), REAL_RO(y.data()), REAL_RO(w.data()), static_cast<std::size_t>(n),
      static_cast<std::size_t>(p), timesift::family_from_name(family), spec,
      which.empty() ? nullptr : which.data(), which.empty() ? 0 : n_fold);
  cpp11::writable::list out({
    "family"_nm = cpp11::as_sexp(std::string(timesift::family_name(fit.family))),
    "n_column"_nm = cpp11::as_sexp(fit.n_column),
    "init"_nm = cpp11::as_sexp(fit.init),
    "cv_error"_nm = give(fit.cv_error)
  });
  give_table(out, fit.trees);
  return out;
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
                           std::string classes, int knots, double regmult,
                           std::string formulation, bool add_samples, double thresh,
                           double max_pass, int n_lambda, bool one_se, cpp11::sexp fold,
                           int n_fold, int threads, double max_design) {
  const timesift::MaxnetSpec spec = maxnet_spec(classes, knots, regmult, formulation, add_samples,
                                                thresh, max_pass, n_lambda, one_se, threads,
                                                max_design);
  const std::vector<std::int32_t> which = take_folds(fold);
  const timesift::Maxnet fit = timesift::maxnet_fit(
      REAL_RO(x.data()), REAL_RO(y.data()), REAL_RO(w.data()), static_cast<std::size_t>(n),
      static_cast<std::size_t>(p), spec, which.empty() ? nullptr : which.data(),
      which.empty() ? 0 : n_fold);
  using namespace cpp11::literals;
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
  return out;
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
[[cpp11::register]]
cpp11::list ts_stepwise_fit_(cpp11::doubles x, cpp11::doubles y, cpp11::doubles w, int n, int p,
                             std::string family, double max_terms, int degree,
                             std::string direction, std::string terms, int threads) {
  timesift::StepwiseSpec spec;
  spec.family = timesift::family_from_name(family);
  spec.max_terms = max_terms;
  spec.degree = degree;
  spec.direction = timesift::step_direction_from_name(direction);
  spec.terms = timesift::step_terms_from_name(terms);
  spec.threads = threads;
  const timesift::Stepwise fit =
      timesift::stepwise_fit(REAL_RO(x.data()), REAL_RO(y.data()), REAL_RO(w.data()),
                             static_cast<std::size_t>(n), static_cast<std::size_t>(p), spec);
  using namespace cpp11::literals;
  return cpp11::writable::list({
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
  });
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
[[cpp11::register]]
cpp11::list ts_mars_fit_(cpp11::doubles x, cpp11::doubles y, cpp11::doubles w, int n, int p,
                         std::string family, int degree, double penalty, int nk, double thresh,
                         int minspan, int endspan, int fast_k, double fast_beta, bool prune,
                         int nprune, int threads) {
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
  const timesift::Mars fit =
      timesift::mars_fit(REAL_RO(x.data()), REAL_RO(y.data()), REAL_RO(w.data()),
                         static_cast<std::size_t>(n), static_cast<std::size_t>(p), spec);
  using namespace cpp11::literals;
  return cpp11::writable::list({
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
  });
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
