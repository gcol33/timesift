#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <exception>
#include <optional>
#include <string>
#include <vector>

#include "ts_additive.h"
#include "ts_core.h"
#include "ts_envelope.h"
#include "ts_fda.h"
#include "ts_hierarchical.h"
#include "ts_mars.h"
#include "ts_maxnet.h"
#include "ts_penalised.h"
#include "ts_stepwise.h"
#include "ts_tree.h"

namespace nb = nanobind;

namespace {

using ConstI32 = nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using ConstI64 = nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using ConstF64 = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using ConstMat = nb::ndarray<const double, nb::ndim<2>, nb::f_contig, nb::device::cpu>;

// Hands a vector to NumPy and lets the capsule free it when the array goes.
template <typename T>
nb::ndarray<nb::numpy, T> give(std::vector<T>&& from) {
  auto* held = new std::vector<T>(std::move(from));
  nb::capsule owner(held, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
  const std::size_t shape[1] = {held->size()};
  return nb::ndarray<nb::numpy, T>(held->data(), 1, shape, owner);
}

std::vector<timesift::Stat> parse_stats(const std::vector<std::string>& names) {
  std::vector<timesift::Stat> out;
  out.reserve(names.size());
  for (const std::string& name : names) out.push_back(timesift::stat_from_name(name));
  return out;
}

// The storage a Request points into, built once for the reduction and for the coverage so the
// two read the same readings the same way.
struct Held {
  std::vector<const char*> names;
  std::vector<double> zeros;
  timesift::Request req;
};

Held hold(ConstI32 unit, const double* value, ConstI64 when, ConstI64 local,
          std::optional<ConstI64> custom, const std::vector<std::string>& unit_names,
          const std::string& grain, int year_month, int year_day) {
  Held h;
  h.names.reserve(unit_names.size());
  for (const std::string& s : unit_names) h.names.push_back(s.c_str());
  if (value == nullptr) h.zeros.assign(local.size(), 0.0);

  h.req.unit = unit.data();
  h.req.value = value == nullptr ? h.zeros.data() : value;
  h.req.when = when.data();
  h.req.local = local.data();
  h.req.custom = custom.has_value() ? custom->data() : nullptr;
  h.req.unit_name = h.names.empty() ? nullptr : h.names.data();
  h.req.n = local.size();
  h.req.n_unit = unit_names.size();
  h.req.grain = timesift::grain_from_name(grain);
  h.req.year_start = timesift::YearStart{year_month, year_day};
  return h;
}

// The penalised fit's settings, from the keywords the Python side names them by.
timesift::PenaltySpec penalty_spec(double alpha, int n_lambda, double lambda_min_ratio,
                                   std::optional<std::vector<double>> lambda, double thresh,
                                   bool standardize, bool intercept, double max_pass,
                                   int threads) {
  timesift::PenaltySpec spec;
  spec.alpha = alpha;
  spec.n_lambda = n_lambda;
  spec.lambda_min_ratio = lambda_min_ratio;
  spec.thresh = thresh;
  spec.max_pass = static_cast<int>(max_pass);
  spec.threads = threads;
  spec.standardize = standardize;
  spec.intercept = intercept;
  if (lambda.has_value()) spec.lambda = *lambda;
  return spec;
}

nb::dict give(const timesift::PenaltyPath& path) {
  nb::dict out;
  out["lambda"] = give(std::vector<double>(path.lambda));
  out["a0"] = give(std::vector<double>(path.a0));
  out["beta"] = give(std::vector<double>(path.beta));
  out["df"] = give(std::vector<std::int32_t>(path.df));
  out["dev_ratio"] = give(std::vector<double>(path.dev_ratio));
  out["null_deviance"] = path.null_deviance;
  out["passes"] = path.passes;
  out["stalled"] = path.stalled;
  out["n_column"] = static_cast<std::int64_t>(path.n_column);
  out["family"] = std::string(timesift::family_name(path.family));
  return out;
}

timesift::PenaltyPath take(ConstF64 lambda, ConstF64 a0, ConstF64 beta,
                           const std::string& family) {
  timesift::PenaltyPath path;
  path.family = timesift::family_from_name(family);
  path.lambda.assign(lambda.data(), lambda.data() + lambda.size());
  path.a0.assign(a0.data(), a0.data() + a0.size());
  path.beta.assign(beta.data(), beta.data() + beta.size());
  path.n_column = path.lambda.empty() ? 0 : path.beta.size() / path.lambda.size();
  return path;
}

// maxnet's settings, and a fit as a dict of arrays, the features it gave a coefficient one field
// each.
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

void give_features(nb::dict& out, const timesift::MaxnetFeatures& f) {
  out["kind"] = give(std::vector<std::int8_t>(f.kind));
  out["a"] = give(std::vector<std::int32_t>(f.a));
  out["b"] = give(std::vector<std::int32_t>(f.b));
  out["lo"] = give(std::vector<double>(f.lo));
  out["hi"] = give(std::vector<double>(f.hi));
}

// A fitted tree crosses into Python as a dict of arrays, one per field of the node table and the
// complexity table, and comes back the same way to be pruned or to predict.
nb::dict give(const timesift::Tree& tree) {
  nb::dict out;
  out["family"] = std::string(timesift::family_name(tree.family));
  out["number"] = give(std::vector<std::int32_t>(tree.number));
  out["column"] = give(std::vector<std::int32_t>(tree.column));
  out["threshold"] = give(std::vector<double>(tree.threshold));
  out["less_left"] = give(std::vector<std::int8_t>(tree.less_left));
  out["left"] = give(std::vector<std::int32_t>(tree.left));
  out["right"] = give(std::vector<std::int32_t>(tree.right));
  out["n"] = give(std::vector<std::int32_t>(tree.n));
  out["weight"] = give(std::vector<double>(tree.weight));
  out["risk"] = give(std::vector<double>(tree.risk));
  out["complexity"] = give(std::vector<double>(tree.complexity));
  out["value"] = give(std::vector<double>(tree.value));
  out["root_risk"] = tree.root_risk;
  out["cp"] = give(std::vector<double>(tree.cp));
  out["nsplit"] = give(std::vector<std::int32_t>(tree.nsplit));
  out["rel_error"] = give(std::vector<double>(tree.rel_error));
  out["xerror"] = give(std::vector<double>(tree.xerror));
  out["xstd"] = give(std::vector<double>(tree.xstd));
  return out;
}

template <typename T>
std::vector<T> take_field(const nb::dict& tree, const char* name) {
  const auto a = nb::cast<nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>>(
      tree[name]);
  return std::vector<T>(a.data(), a.data() + a.size());
}

timesift::Tree take_tree(const nb::dict& tree) {
  timesift::Tree out;
  out.family = timesift::family_from_name(nb::cast<std::string>(tree["family"]));
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
  out.root_risk = nb::cast<double>(tree["root_risk"]);
  out.cp = take_field<double>(tree, "cp");
  out.nsplit = take_field<std::int32_t>(tree, "nsplit");
  out.rel_error = take_field<double>(tree, "rel_error");
  out.xerror = take_field<double>(tree, "xerror");
  out.xstd = take_field<double>(tree, "xstd");
  return out;
}

// A forest and a boosted fit carry their trees as one node table, the trees one after another.
void give_table(nb::dict& out, timesift::TreeTable&& table) {
  out["offset"] = give(std::move(table.offset));
  out["column"] = give(std::move(table.column));
  out["threshold"] = give(std::move(table.threshold));
  out["less_left"] = give(std::move(table.less_left));
  out["left"] = give(std::move(table.left));
  out["right"] = give(std::move(table.right));
  out["value"] = give(std::move(table.value));
}

timesift::TreeTable take_table(const nb::dict& from) {
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

NB_MODULE(_core, m) {
  m.doc() = "The binning, the reduction, the penalised fit, maxnet, the tree, the forest, the "
            "boosted trees, the envelope, the stepwise model, MARS and the additive model, shared "
            "with the R package "
            "as src/ts_core.cpp, src/ts_penalised.cpp, src/ts_maxnet.cpp, src/ts_tree.cpp, "
            "src/ts_boost.cpp, src/ts_envelope.cpp, src/ts_stepwise.cpp, src/ts_glm.cpp, "
            "src/ts_mars.cpp and src/ts_additive.cpp.";

  nb::register_exception_translator(
      [](const std::exception_ptr& p, void*) {
        try {
          std::rethrow_exception(p);
        } catch (const timesift::Error& e) {
          PyErr_SetString(PyExc_ValueError, e.what());
        }
      });

  m.def("reduce",
        [](ConstI32 unit, ConstF64 value, ConstI64 when, ConstI64 local,
           std::optional<ConstI64> custom, const std::vector<std::string>& unit_names,
           const std::string& grain, int year_month, int year_day,
           const std::vector<std::string>& stats, std::int64_t sampling_step) {
          Held h = hold(unit, value.data(), when, local, custom, unit_names, grain, year_month,
                        year_day);
          h.req.sampling_step = sampling_step;
          h.req.stats = parse_stats(stats);

          timesift::Result out = timesift::reduce(h.req);
          std::vector<std::uint8_t> partial = std::move(out.bin_partial);
          return nb::make_tuple(give(std::move(out.values)), give(std::move(out.bin_start)),
                                give(std::move(out.bin_end)), give(std::move(out.bin_n)),
                                give(std::move(partial)));
        },
        nb::arg("unit"), nb::arg("value"), nb::arg("when"), nb::arg("local"), nb::arg("custom"),
        nb::arg("unit_names"), nb::arg("grain"), nb::arg("year_month"), nb::arg("year_day"),
        nb::arg("stats"), nb::arg("sampling_step"));

  m.def("coverage",
        [](ConstI32 unit, ConstI64 when, ConstI64 local, std::optional<ConstI64> custom,
           const std::vector<std::string>& unit_names, const std::string& grain, int year_month,
           int year_day) {
          Held h = hold(unit, nullptr, when, local, custom, unit_names, grain, year_month,
                        year_day);
          timesift::Coverage out = timesift::coverage(h.req);
          return nb::make_tuple(give(std::move(out.bin_start)), give(std::move(out.count)));
        },
        nb::arg("unit"), nb::arg("when"), nb::arg("local"), nb::arg("custom"),
        nb::arg("unit_names"), nb::arg("grain"), nb::arg("year_month"), nb::arg("year_day"));

  m.def("reduce_lookbacks",
        [](ConstI32 unit, ConstF64 value, ConstI64 when, ConstI64 local,
           const std::vector<std::string>& unit_names, ConstI32 target_unit, ConstI64 target_at,
           const std::vector<std::string>& target_names, std::int64_t span, std::int64_t lag,
           std::int32_t bins, const std::vector<std::string>& stats) {
          std::vector<const char*> units, targets;
          units.reserve(unit_names.size());
          for (const std::string& s : unit_names) units.push_back(s.c_str());
          targets.reserve(target_names.size());
          for (const std::string& s : target_names) targets.push_back(s.c_str());

          timesift::LookbackRequest req;
          req.unit = unit.data();
          req.value = value.data();
          req.when = when.data();
          req.local = local.data();
          req.unit_name = units.empty() ? nullptr : units.data();
          req.n = value.size();
          req.n_unit = unit_names.size();
          req.target_unit = target_unit.data();
          req.target_at = target_at.data();
          req.target_name = targets.empty() ? nullptr : targets.data();
          req.n_target = target_at.size();
          req.span = span;
          req.lag = lag;
          req.n_bin = bins;
          req.stats = parse_stats(stats);

          timesift::LookbackResult out = timesift::reduce_lookbacks(req);
          return nb::make_tuple(give(std::move(out.values)), give(std::move(out.bin_n)));
        },
        nb::arg("unit"), nb::arg("value"), nb::arg("when"), nb::arg("local"),
        nb::arg("unit_names"), nb::arg("target_unit"), nb::arg("target_at"),
        nb::arg("target_names"), nb::arg("span"),
        nb::arg("lag"), nb::arg("bins"), nb::arg("stats"));

  m.def("bin_starts",
        [](ConstI64 local, const std::string& grain, int year_month, int year_day) {
          std::vector<timesift::seconds> out(local.size());
          timesift::bin_starts(local.data(), local.size(), timesift::grain_from_name(grain),
                                timesift::YearStart{year_month, year_day}, out.data());
          return give(std::move(out));
        },
        nb::arg("local"), nb::arg("grain"), nb::arg("year_month"), nb::arg("year_day"));

  m.def("cycle_fraction",
        [](ConstI64 bin_start, ConstI64 bin_end, const std::string& cycle) {
          std::vector<double> frac(bin_start.size());
          timesift::cycle_fraction(cycle, bin_start.data(), bin_end.data(), bin_start.size(),
                                   frac.data());
          return give(std::move(frac));
        },
        nb::arg("bin_start"), nb::arg("bin_end"), nb::arg("cycle"));

  m.def("cycle_phase",
        [](ConstI64 bin_start, ConstI64 bin_end, const std::string& cycle) {
          std::vector<double> out_sin(bin_start.size()), out_cos(bin_start.size());
          timesift::cycle_phase(cycle, bin_start.data(), bin_end.data(), bin_start.size(),
                                out_sin.data(), out_cos.data());
          return nb::make_tuple(give(std::move(out_sin)), give(std::move(out_cos)));
        },
        nb::arg("bin_start"), nb::arg("bin_end"), nb::arg("cycle"));

  m.def("bin_nexts",
        [](ConstI64 bins, const std::string& grain, int year_month, int year_day) {
          std::vector<timesift::seconds> out(bins.size());
          timesift::bin_nexts(bins.data(), bins.size(), timesift::grain_from_name(grain),
                               timesift::YearStart{year_month, year_day}, out.data());
          return give(std::move(out));
        },
        nb::arg("bins"), nb::arg("grain"), nb::arg("year_month"), nb::arg("year_day"));

  m.def("penalised_path",
        [](ConstMat x, ConstF64 y, ConstF64 w, const std::string& family, double alpha,
           int n_lambda, double lambda_min_ratio, std::optional<std::vector<double>> lambda,
           double thresh, bool standardize, bool intercept, double max_pass) {
          return give(timesift::penalised_path(
              x.data(), y.data(), w.data(), x.shape(0), x.shape(1),
              timesift::family_from_name(family),
              penalty_spec(alpha, n_lambda, lambda_min_ratio, std::move(lambda), thresh,
                           standardize, intercept, max_pass, 1)));
        },
        nb::arg("x"), nb::arg("y"), nb::arg("w"), nb::arg("family"), nb::arg("alpha") = 1.0,
        nb::arg("n_lambda") = 100, nb::arg("lambda_min_ratio") = 0.0,
        nb::arg("lambda") = nb::none(), nb::arg("thresh") = 1e-8,
        nb::arg("standardize") = true, nb::arg("intercept") = true,
        nb::arg("max_pass") = 1e6);

  m.def("penalised_cv",
        [](ConstMat x, ConstF64 y, ConstF64 w, ConstI32 fold, int n_fold,
           const std::string& family, double alpha, int n_lambda, double lambda_min_ratio,
           double thresh, bool standardize, bool intercept, double max_pass, int threads) {
          const timesift::PenaltyCV cv = timesift::penalised_cv(
              x.data(), y.data(), w.data(), x.shape(0), x.shape(1),
              timesift::family_from_name(family),
              penalty_spec(alpha, n_lambda, lambda_min_ratio, std::nullopt, thresh, standardize,
                           intercept, max_pass, threads),
              fold.data(), n_fold);
          nb::dict out = give(cv.path);
          out["cv_mean"] = give(std::vector<double>(cv.cv_mean));
          out["cv_sd"] = give(std::vector<double>(cv.cv_sd));
          out["index_min"] = static_cast<std::int64_t>(cv.index_min);
          out["index_1se"] = static_cast<std::int64_t>(cv.index_1se);
          out["fold_stalled"] = give(std::vector<std::int32_t>(cv.fold_stalled));
          return out;
        },
        nb::arg("x"), nb::arg("y"), nb::arg("w"), nb::arg("fold"), nb::arg("n_fold"),
        nb::arg("family"), nb::arg("alpha") = 1.0, nb::arg("n_lambda") = 100,
        nb::arg("lambda_min_ratio") = 0.0, nb::arg("thresh") = 1e-8,
        nb::arg("standardize") = true, nb::arg("intercept") = true,
        nb::arg("max_pass") = 1e6, nb::arg("threads") = 1);

  m.def("penalised_predict",
        [](ConstF64 lambda, ConstF64 a0, ConstF64 beta, const std::string& family, double at,
           ConstMat newx) {
          const timesift::PenaltyPath path = take(lambda, a0, beta, family);
          std::vector<double> out(newx.shape(0));
          timesift::penalised_predict(path, at, newx.data(), newx.shape(0), out.data());
          return give(std::move(out));
        },
        nb::arg("lambda"), nb::arg("a0"), nb::arg("beta"), nb::arg("family"), nb::arg("at"),
        nb::arg("newx"));

  m.def("penalised_coef",
        [](ConstF64 lambda, ConstF64 a0, ConstF64 beta, const std::string& family, double at) {
          const timesift::PenaltyPath path = take(lambda, a0, beta, family);
          std::vector<double> coef(path.n_column + 1, 0.0);
          timesift::penalised_coef(path, at, coef.data(), coef.data() + 1);
          return give(std::move(coef));
        },
        nb::arg("lambda"), nb::arg("a0"), nb::arg("beta"), nb::arg("family"), nb::arg("at"));

  m.def("tree_fit",
        [](ConstMat x, ConstF64 y, ConstF64 w, const std::string& family, int min_split,
           int min_leaf, double cp, int max_depth, std::optional<ConstI32> fold, int n_fold,
           double shrink) {
          timesift::TreeSpec spec;
          spec.min_split = min_split;
          spec.min_leaf = min_leaf;
          spec.cp = cp;
          spec.max_depth = max_depth;
          spec.shrink = shrink;
          return give(timesift::tree_fit(
              x.data(), y.data(), w.data(), x.shape(0), x.shape(1),
              timesift::family_from_name(family), spec,
              fold.has_value() ? fold->data() : nullptr, fold.has_value() ? n_fold : 0));
        },
        nb::arg("x"), nb::arg("y"), nb::arg("w"), nb::arg("family"), nb::arg("min_split"),
        nb::arg("min_leaf"), nb::arg("cp"), nb::arg("max_depth"), nb::arg("fold") = nb::none(),
        nb::arg("n_fold") = 0, nb::arg("shrink") = 1.0);

  m.def("tree_prune",
        [](const nb::dict& tree, double cp) {
          return give(timesift::tree_prune(take_tree(tree), cp));
        },
        nb::arg("tree"), nb::arg("cp"));

  m.def("tree_predict",
        [](const nb::dict& tree, ConstMat newx) {
          std::vector<double> out(newx.shape(0));
          timesift::tree_predict(take_tree(tree), newx.data(), newx.shape(0), newx.shape(1),
                                 out.data());
          return give(std::move(out));
        },
        nb::arg("tree"), nb::arg("newx"));

  m.def("forest_fit",
        [](ConstMat x, ConstF64 y, ConstF64 w, const std::string& family, int trees, int mtry,
           int min_leaf, bool balance, std::uint32_t seed, int threads) {
          timesift::ForestSpec spec;
          spec.trees = trees;
          spec.mtry = mtry;
          spec.min_leaf = min_leaf;
          spec.balance = balance;
          spec.seed = seed;
          spec.threads = threads;
          timesift::Forest forest;
          {
            nb::gil_scoped_release release;
            forest = timesift::forest_fit(x.data(), y.data(), w.data(), x.shape(0), x.shape(1),
                                          timesift::family_from_name(family), spec);
          }
          nb::dict out;
          out["family"] = std::string(timesift::family_name(forest.family));
          out["n_column"] = forest.n_column;
          give_table(out, std::move(forest.trees));
          return out;
        },
        nb::arg("x"), nb::arg("y"), nb::arg("w"), nb::arg("family"), nb::arg("trees"),
        nb::arg("mtry"), nb::arg("min_leaf"), nb::arg("balance"), nb::arg("seed"),
        nb::arg("threads") = 1);

  m.def("forest_predict",
        [](const nb::dict& forest, ConstMat newx) {
          timesift::Forest f;
          f.family = timesift::family_from_name(nb::cast<std::string>(forest["family"]));
          f.n_column = nb::cast<std::int32_t>(forest["n_column"]);
          f.trees = take_table(forest);
          std::vector<double> out(newx.shape(0));
          timesift::forest_predict(f, newx.data(), newx.shape(0), newx.shape(1), out.data());
          return give(std::move(out));
        },
        nb::arg("forest"), nb::arg("newx"));

  m.def("boost_fit",
        [](ConstMat x, ConstF64 y, ConstF64 w, const std::string& family, int trees, int depth,
           double shrinkage, double min_leaf, double subsample, double colsample, bool newton,
           double lambda, double gamma, std::uint32_t seed, std::optional<ConstI32> fold,
           int n_fold, int threads) {
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
          spec.seed = seed;
          spec.threads = threads;
          timesift::Boosted fit;
          {
            nb::gil_scoped_release release;
            fit = timesift::boost_fit(x.data(), y.data(), w.data(), x.shape(0), x.shape(1),
                                      timesift::family_from_name(family), spec,
                                      fold.has_value() ? fold->data() : nullptr,
                                      fold.has_value() ? n_fold : 0);
          }
          nb::dict out;
          out["family"] = std::string(timesift::family_name(fit.family));
          out["n_column"] = fit.n_column;
          out["init"] = fit.init;
          out["cv_error"] = give(std::move(fit.cv_error));
          give_table(out, std::move(fit.trees));
          return out;
        },
        nb::arg("x"), nb::arg("y"), nb::arg("w"), nb::arg("family"), nb::arg("trees"),
        nb::arg("depth"), nb::arg("shrinkage"), nb::arg("min_leaf"), nb::arg("subsample"),
        nb::arg("colsample"), nb::arg("newton"), nb::arg("lambda"), nb::arg("gamma"),
        nb::arg("seed"), nb::arg("fold") = nb::none(), nb::arg("n_fold") = 0,
        nb::arg("threads") = 1);

  m.def("boost_predict",
        [](const nb::dict& fit, ConstMat newx) {
          timesift::Boosted b;
          b.family = timesift::family_from_name(nb::cast<std::string>(fit["family"]));
          b.n_column = nb::cast<std::int32_t>(fit["n_column"]);
          b.init = nb::cast<double>(fit["init"]);
          b.trees = take_table(fit);
          std::vector<double> out(newx.shape(0));
          timesift::boost_predict(b, newx.data(), newx.shape(0), newx.shape(1), out.data());
          return give(std::move(out));
        },
        nb::arg("fit"), nb::arg("newx"));

  m.def("forest_stream",
        [](std::uint32_t seed, std::uint32_t tree, std::size_t n) {
          std::vector<std::uint32_t> out(n);
          timesift::forest_stream(seed, tree, n, out.data());
          return give(std::move(out));
        },
        nb::arg("seed"), nb::arg("tree"), nb::arg("n"));

  m.def("maxnet_design",
        [](ConstMat x, ConstF64 y, const std::string& classes, int knots, double regmult,
           const std::string& formulation, bool add_samples, double max_design) {
          const timesift::MaxnetDesign d = timesift::maxnet_design(
              x.data(), y.data(), x.shape(0), x.shape(1),
              maxnet_spec(classes, knots, regmult, formulation, add_samples, 1e-8, 1e6, 100,
                          false, 1, max_design));
          nb::dict out;
          out["classes"] = d.classes;
          out["rows"] = give(std::vector<std::int64_t>(d.rows.begin(), d.rows.end()));
          out["y"] = give(std::vector<double>(d.y));
          out["reg"] = give(std::vector<double>(d.reg));
          give_features(out, d.features);
          return out;
        },
        nb::arg("x"), nb::arg("y"), nb::arg("classes"), nb::arg("knots"), nb::arg("regmult"),
        nb::arg("formulation"), nb::arg("add_samples"), nb::arg("max_design"));

  m.def("maxnet_fit",
        [](ConstMat x, ConstF64 y, ConstF64 w, const std::string& classes, int knots,
           double regmult, const std::string& formulation, bool add_samples, double thresh,
           double max_pass, int n_lambda, bool one_se, std::optional<ConstI32> fold, int n_fold,
           int threads, double max_design) {
          const timesift::MaxnetSpec spec =
              maxnet_spec(classes, knots, regmult, formulation, add_samples, thresh, max_pass,
                          n_lambda, one_se, threads, max_design);
          timesift::Maxnet fit;
          {
            nb::gil_scoped_release release;
            fit = timesift::maxnet_fit(x.data(), y.data(), w.data(), x.shape(0), x.shape(1),
                                       spec, fold.has_value() ? fold->data() : nullptr,
                                       fold.has_value() ? n_fold : 0);
          }
          nb::dict out;
          out["formulation"] = std::string(timesift::maxnet_formulation_name(fit.formulation));
          out["classes"] = fit.classes;
          out["n_column"] = fit.n_column;
          out["n_presence"] = fit.n_presence;
          out["n_feature"] = fit.n_feature;
          out["var_min"] = give(std::move(fit.var_min));
          out["var_max"] = give(std::move(fit.var_max));
          out["feature_min"] = give(std::move(fit.feature_min));
          out["feature_max"] = give(std::move(fit.feature_max));
          out["beta"] = give(std::move(fit.beta));
          out["intercept"] = fit.intercept;
          out["lasso_intercept"] = fit.lasso_intercept;
          out["entropy"] = fit.entropy;
          out["lambda"] = fit.lambda;
          out["stalled"] = fit.stalled;
          out["fold_stalled"] = fit.fold_stalled;
          give_features(out, fit.features);
          return out;
        },
        nb::arg("x"), nb::arg("y"), nb::arg("w"), nb::arg("classes"), nb::arg("knots"),
        nb::arg("regmult"), nb::arg("formulation"), nb::arg("add_samples"), nb::arg("thresh"),
        nb::arg("max_pass"), nb::arg("n_lambda"), nb::arg("one_se"),
        nb::arg("fold") = nb::none(), nb::arg("n_fold") = 0, nb::arg("threads") = 1,
        nb::arg("max_design") = 2.0);

  m.def("maxnet_predict",
        [](const nb::dict& fit, ConstMat newx, bool clamp, const std::string& type) {
          timesift::Maxnet f;
          f.formulation = timesift::maxnet_formulation_from_name(
              nb::cast<std::string>(fit["formulation"]));
          f.n_column = nb::cast<std::int32_t>(fit["n_column"]);
          f.var_min = take_field<double>(fit, "var_min");
          f.var_max = take_field<double>(fit, "var_max");
          f.features.kind = take_field<std::int8_t>(fit, "kind");
          f.features.a = take_field<std::int32_t>(fit, "a");
          f.features.b = take_field<std::int32_t>(fit, "b");
          f.features.lo = take_field<double>(fit, "lo");
          f.features.hi = take_field<double>(fit, "hi");
          f.feature_min = take_field<double>(fit, "feature_min");
          f.feature_max = take_field<double>(fit, "feature_max");
          f.beta = take_field<double>(fit, "beta");
          f.intercept = nb::cast<double>(fit["intercept"]);
          f.entropy = nb::cast<double>(fit["entropy"]);
          std::vector<double> out(newx.shape(0));
          timesift::maxnet_predict(f, newx.data(), newx.shape(0), newx.shape(1), clamp,
                                   timesift::maxnet_output_from_name(type), out.data());
          return give(std::move(out));
        },
        nb::arg("fit"), nb::arg("newx"), nb::arg("clamp"), nb::arg("type"));

  m.def("envelope_fit",
        [](ConstMat x, ConstF64 y, double quantile) {
          timesift::Envelope fit =
              timesift::envelope_fit(x.data(), y.data(), x.shape(0), x.shape(1), quantile);
          nb::dict out;
          out["n_column"] = fit.n_column;
          out["n_presence"] = fit.n_presence;
          out["lo"] = give(std::move(fit.lo));
          out["hi"] = give(std::move(fit.hi));
          return out;
        },
        nb::arg("x"), nb::arg("y"), nb::arg("quantile"));

  m.def("envelope_predict",
        [](const nb::dict& fit, ConstMat newx) {
          timesift::Envelope e;
          e.n_column = nb::cast<std::int32_t>(fit["n_column"]);
          e.n_presence = nb::cast<std::int32_t>(fit["n_presence"]);
          e.lo = take_field<double>(fit, "lo");
          e.hi = take_field<double>(fit, "hi");
          std::vector<double> out(newx.shape(0));
          timesift::envelope_predict(e, newx.data(), newx.shape(0), newx.shape(1), out.data());
          return give(std::move(out));
        },
        nb::arg("fit"), nb::arg("newx"));

  m.def("stepwise_fit",
        [](ConstMat x, ConstF64 y, ConstF64 w, const std::string& family, double max_terms,
           int degree, const std::string& direction, const std::string& terms, int threads) {
          timesift::StepwiseSpec spec;
          spec.family = timesift::family_from_name(family);
          spec.max_terms = max_terms;
          spec.degree = degree;
          spec.direction = timesift::step_direction_from_name(direction);
          spec.terms = timesift::step_terms_from_name(terms);
          spec.threads = threads;
          timesift::Stepwise fit;
          {
            nb::gil_scoped_release release;
            fit = timesift::stepwise_fit(x.data(), y.data(), w.data(), x.shape(0), x.shape(1),
                                         spec);
          }
          nb::dict out;
          out["family"] = std::string(timesift::family_name(fit.family));
          out["n_column"] = fit.n_column;
          out["constant"] = fit.constant;
          out["term_column"] = give(std::move(fit.term_column));
          out["term_power"] = give(std::move(fit.term_power));
          out["term_degree"] = give(std::move(fit.term_degree));
          out["alpha"] = give(std::move(fit.alpha));
          out["norm2"] = give(std::move(fit.norm2));
          out["beta"] = give(std::move(fit.beta));
          out["rank"] = fit.rank;
          out["deviance"] = fit.deviance;
          out["aic"] = fit.aic;
          out["converged"] = fit.converged;
          out["steps"] = fit.steps;
          return out;
        },
        nb::arg("x"), nb::arg("y"), nb::arg("w"), nb::arg("family"), nb::arg("max_terms"),
        nb::arg("degree"), nb::arg("direction"), nb::arg("terms"), nb::arg("threads") = 1);

  m.def("stepwise_predict",
        [](const nb::dict& fit, ConstMat newx) {
          timesift::Stepwise s;
          s.family = timesift::family_from_name(nb::cast<std::string>(fit["family"]));
          s.n_column = nb::cast<std::int32_t>(fit["n_column"]);
          s.constant = nb::cast<double>(fit["constant"]);
          s.term_column = take_field<std::int32_t>(fit, "term_column");
          s.term_power = take_field<std::int32_t>(fit, "term_power");
          s.term_degree = take_field<std::int32_t>(fit, "term_degree");
          s.alpha = take_field<double>(fit, "alpha");
          s.norm2 = take_field<double>(fit, "norm2");
          s.beta = take_field<double>(fit, "beta");
          std::vector<double> out(newx.shape(0));
          timesift::stepwise_predict(s, newx.data(), newx.shape(0), newx.shape(1), out.data());
          return give(std::move(out));
        },
        nb::arg("fit"), nb::arg("newx"));

  m.def("mars_fit",
        [](ConstMat x, ConstF64 y, ConstF64 w, const std::string& family, int degree,
           double penalty, int nk, double thresh, int minspan, int endspan, int fast_k,
           double fast_beta, bool prune, int nprune, int threads) {
          timesift::MarsSpec spec;
          spec.family = timesift::family_from_name(family);
          spec.degree = degree;
          spec.penalty = penalty;
          spec.nk = nk;
          spec.thresh = thresh;
          spec.minspan = minspan;
          spec.endspan = endspan;
          spec.fast_k = fast_k;
          spec.fast_beta = fast_beta;
          spec.prune = prune;
          spec.nprune = nprune;
          spec.threads = threads;
          timesift::Mars fit;
          {
            nb::gil_scoped_release release;
            fit = timesift::mars_fit(x.data(), y.data(), w.data(), x.shape(0), x.shape(1), spec);
          }
          nb::dict out;
          out["family"] = std::string(timesift::family_name(fit.family));
          out["n_column"] = fit.n_column;
          out["factor_start"] = give(std::move(fit.factor_start));
          out["factor_column"] = give(std::move(fit.factor_column));
          out["factor_dir"] = give(std::move(fit.factor_dir));
          out["factor_cut"] = give(std::move(fit.factor_cut));
          out["selected"] = give(std::move(fit.selected));
          out["beta"] = give(std::move(fit.beta));
          out["termcond"] = fit.termcond;
          out["gcv"] = fit.gcv;
          out["converged"] = fit.converged;
          return out;
        },
        nb::arg("x"), nb::arg("y"), nb::arg("w"), nb::arg("family"), nb::arg("degree"),
        nb::arg("penalty"), nb::arg("nk"), nb::arg("thresh"), nb::arg("minspan"),
        nb::arg("endspan"), nb::arg("fast_k"), nb::arg("fast_beta"), nb::arg("prune"),
        nb::arg("nprune"), nb::arg("threads") = 1);

  m.def("mars_predict",
        [](const nb::dict& fit, ConstMat newx) {
          timesift::Mars s;
          s.family = timesift::family_from_name(nb::cast<std::string>(fit["family"]));
          s.n_column = nb::cast<std::int32_t>(fit["n_column"]);
          s.factor_start = take_field<std::int32_t>(fit, "factor_start");
          s.factor_column = take_field<std::int32_t>(fit, "factor_column");
          s.factor_dir = take_field<std::int32_t>(fit, "factor_dir");
          s.factor_cut = take_field<double>(fit, "factor_cut");
          s.selected = take_field<std::int32_t>(fit, "selected");
          s.beta = take_field<double>(fit, "beta");
          std::vector<double> out(newx.shape(0));
          timesift::mars_predict(s, newx.data(), newx.shape(0), newx.shape(1), out.data());
          return give(std::move(out));
        },
        nb::arg("fit"), nb::arg("newx"));

  m.def("additive_fit",
        [](ConstMat x, ConstMat y, ConstMat w, const std::string& family, int k, double gamma,
           int max_knots, int threads, std::optional<std::vector<double>> sp) {
          if (y.shape(0) != x.shape(0) || w.shape(0) != x.shape(0) || w.shape(1) != y.shape(1)) {
            throw timesift::Error("an additive model's response and weights have a row per unit "
                                  "and the same columns.");
          }
          timesift::AdditiveSpec spec;
          spec.family = timesift::family_from_name(family);
          spec.k = k;
          spec.gamma = gamma;
          spec.max_knots = max_knots;
          spec.threads = threads;
          if (sp) spec.sp = *sp;
          timesift::Additive fit;
          {
            nb::gil_scoped_release release;
            fit = timesift::additive_fit(x.data(), x.shape(0), x.shape(1), y.data(), w.data(),
                                         y.shape(1), spec);
          }
          nb::dict out;
          out["family"] = std::string(timesift::family_name(fit.family));
          out["n_column"] = fit.n_column;
          out["n_coef"] = fit.n_coef;
          out["term_column"] = give(std::move(fit.term_column));
          out["term_basis"] = give(std::move(fit.term_basis));
          out["term_size"] = give(std::move(fit.term_size));
          out["term_penalised"] = give(std::move(fit.term_penalised));
          out["term_shift"] = give(std::move(fit.term_shift));
          out["knot_start"] = give(std::move(fit.knot_start));
          out["knots"] = give(std::move(fit.knots));
          out["radial_start"] = give(std::move(fit.radial_start));
          out["radial"] = give(std::move(fit.radial));
          out["map_start"] = give(std::move(fit.map_start));
          out["map"] = give(std::move(fit.map));
          out["penalty"] = give(std::move(fit.penalty));
          out["aliased"] = give(std::move(fit.aliased));
          out["n_response"] = fit.n_response;
          out["beta"] = give(std::move(fit.beta));
          out["sp"] = give(std::move(fit.sp));
          out["edf"] = give(std::move(fit.edf));
          out["score"] = give(std::move(fit.score));
          out["outer"] = give(std::move(fit.outer));
          out["converged"] = give(std::move(fit.converged));
          return out;
        },
        nb::arg("x"), nb::arg("y"), nb::arg("w"), nb::arg("family"), nb::arg("k"),
        nb::arg("gamma"), nb::arg("max_knots"), nb::arg("threads") = 1,
        nb::arg("sp") = nb::none());

  m.def("additive_predict",
        [](const nb::dict& fit, ConstMat newx) {
          timesift::Additive a;
          a.family = timesift::family_from_name(nb::cast<std::string>(fit["family"]));
          a.n_column = nb::cast<std::int32_t>(fit["n_column"]);
          a.n_coef = nb::cast<std::int32_t>(fit["n_coef"]);
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
          a.n_response = nb::cast<std::int32_t>(fit["n_response"]);
          a.beta = take_field<double>(fit, "beta");
          std::vector<double> out(newx.shape(0) * static_cast<std::size_t>(a.n_response));
          timesift::additive_predict(a, newx.data(), newx.shape(0), newx.shape(1), out.data());
          return give(std::move(out));
        },
        nb::arg("fit"), nb::arg("newx"));

  m.def("fda_fit",
        [](ConstMat x, ConstF64 y, ConstF64 w, int degree, double penalty, int nk, double thresh,
           bool prune, bool calibrate, int threads) {
          timesift::FdaSpec spec;
          spec.degree = degree;
          spec.penalty = penalty;
          spec.nk = nk;
          spec.thresh = thresh;
          spec.prune = prune;
          spec.calibrate = calibrate;
          spec.threads = threads;
          timesift::Fda fit;
          {
            nb::gil_scoped_release release;
            fit = timesift::fda_fit(x.data(), y.data(), w.data(), x.shape(0), x.shape(1), spec);
          }
          nb::dict out;
          out["n_column"] = fit.n_column;
          out["factor_start"] = give(std::move(fit.factor_start));
          out["factor_column"] = give(std::move(fit.factor_column));
          out["factor_dir"] = give(std::move(fit.factor_dir));
          out["factor_cut"] = give(std::move(fit.factor_cut));
          out["coef"] = give(std::move(fit.coef));
          out["forward_terms"] = fit.forward_terms;
          out["gcv"] = fit.gcv;
          out["discriminates"] = fit.discriminates;
          out["mean"] = fit.mean;
          out["direction"] = fit.direction;
          out["scale"] = fit.scale;
          out["centroid"] = give(std::vector<double>(fit.centroid, fit.centroid + 2));
          out["prior"] = give(std::vector<double>(fit.prior, fit.prior + 2));
          out["calibrated"] = fit.calibrated;
          out["calibration"] = give(std::vector<double>(fit.calibration, fit.calibration + 2));
          out["converged"] = fit.converged;
          return out;
        },
        nb::arg("x"), nb::arg("y"), nb::arg("w"), nb::arg("degree"), nb::arg("penalty"),
        nb::arg("nk"), nb::arg("thresh"), nb::arg("prune"), nb::arg("calibrate"),
        nb::arg("threads") = 1);

  m.def("fda_predict",
        [](const nb::dict& fit, ConstMat newx) {
          timesift::Fda f;
          f.n_column = nb::cast<std::int32_t>(fit["n_column"]);
          f.factor_start = take_field<std::int32_t>(fit, "factor_start");
          f.factor_column = take_field<std::int32_t>(fit, "factor_column");
          f.factor_dir = take_field<std::int32_t>(fit, "factor_dir");
          f.factor_cut = take_field<double>(fit, "factor_cut");
          f.coef = take_field<double>(fit, "coef");
          f.discriminates = nb::cast<bool>(fit["discriminates"]);
          f.mean = nb::cast<double>(fit["mean"]);
          f.direction = nb::cast<double>(fit["direction"]);
          f.scale = nb::cast<double>(fit["scale"]);
          const std::vector<double> centroid = take_field<double>(fit, "centroid");
          const std::vector<double> prior = take_field<double>(fit, "prior");
          const std::vector<double> calibration = take_field<double>(fit, "calibration");
          for (int j = 0; j < 2; ++j) {
            f.centroid[j] = centroid[j];
            f.prior[j] = prior[j];
            f.calibration[j] = calibration[j];
          }
          f.calibrated = nb::cast<bool>(fit["calibrated"]);
          std::vector<double> out(newx.shape(0));
          timesift::fda_predict(f, newx.data(), newx.shape(0), newx.shape(1), out.data());
          return give(std::move(out));
        },
        nb::arg("fit"), nb::arg("newx"));

  m.def("hierarchical_fit",
        [](ConstMat x, ConstF64 y, ConstF64 w, std::optional<ConstI32> unit, std::size_t n_unit,
           std::optional<ConstMat> coords, const std::string& field, double beta_sd,
           double sd_u, double sd_alpha, double range_fraction, double range_alpha, int m_basis,
           double boundary, int neighbours, int cov, int nodes, double step, int threads,
           std::optional<std::vector<double>> theta) {
          timesift::HierSpec spec;
          spec.beta_sd = beta_sd;
          spec.unit = unit.has_value();
          if (field == "none") {
            spec.field = timesift::Field::none;
          } else if (field == "hsgp") {
            spec.field = timesift::Field::hsgp;
          } else if (field == "nngp") {
            spec.field = timesift::Field::nngp;
          } else {
            throw timesift::Error("a hierarchical field is none, hsgp or nngp, not '" + field + "'.");
          }
          spec.sd_u = sd_u;
          spec.sd_alpha = sd_alpha;
          spec.range_fraction = range_fraction;
          spec.range_alpha = range_alpha;
          spec.m = m_basis;
          spec.boundary = boundary;
          spec.neighbours = neighbours;
          spec.cov = cov;
          spec.nodes = nodes;
          spec.step = step;
          spec.threads = threads;
          if (theta.has_value()) spec.theta = *theta;
          const timesift::Hierarchical f = timesift::hierarchical_fit(
              x.data(), x.shape(0), x.shape(1), y.data(), w.data(),
              unit.has_value() ? unit->data() : nullptr, n_unit,
              coords.has_value() ? coords->data() : nullptr, spec);
          nb::dict out;
          out["field"] = field;
          out["n_column"] = f.n_column;
          out["n_unit"] = f.n_unit;
          out["n_theta"] = f.n_theta;
          out["beta"] = give(std::vector<double>(f.beta));
          out["unit_effect"] = give(std::vector<double>(f.unit_effect));
          out["theta_hat"] = give(std::vector<double>(f.theta_hat));
          out["log_marginal"] = f.log_marginal;
          out["n_node"] = f.n_node;
          out["node_theta"] = give(std::vector<double>(f.node_theta));
          out["node_weight"] = give(std::vector<double>(f.node_weight));
          out["node_log_post"] = give(std::vector<double>(f.node_log_post));
          out["node_field"] = give(std::vector<double>(f.node_field));
          out["n_field"] = f.n_field;
          out["converged"] = f.converged;
          out["centre"] = give(std::vector<double>(f.centre, f.centre + 2));
          out["scale"] = f.scale;
          out["m"] = f.m;
          out["box_centre"] = give(std::vector<double>(f.box_centre, f.box_centre + 2));
          out["box_half"] = give(std::vector<double>(f.box_half, f.box_half + 2));
          out["location"] = give(std::vector<double>(f.location));
          out["n_location"] = f.n_location;
          out["neighbours"] = f.neighbours;
          out["cov"] = f.cov;
          return out;
        },
        nb::arg("x"), nb::arg("y"), nb::arg("w"), nb::arg("unit") = nb::none(),
        nb::arg("n_unit") = 0, nb::arg("coords") = nb::none(), nb::arg("field") = "none",
        nb::arg("beta_sd") = 2.5, nb::arg("sd_u") = 3.0, nb::arg("sd_alpha") = 0.01,
        nb::arg("range_fraction") = 0.2, nb::arg("range_alpha") = 0.5, nb::arg("m") = 6,
        nb::arg("boundary") = 1.5, nb::arg("neighbours") = 15, nb::arg("cov") = 0,
        nb::arg("nodes") = 5, nb::arg("step") = 1.25, nb::arg("threads") = 1,
        nb::arg("theta") = nb::none());

  m.def("hierarchical_predict",
        [](const nb::dict& fit, ConstMat newx, std::optional<ConstI32> unit,
           std::optional<ConstMat> coords) {
          timesift::Hierarchical h;
          const std::string field = nb::cast<std::string>(fit["field"]);
          h.field = field == "none"   ? timesift::Field::none
                    : field == "hsgp" ? timesift::Field::hsgp
                                      : timesift::Field::nngp;
          h.n_column = nb::cast<std::int32_t>(fit["n_column"]);
          h.n_unit = nb::cast<std::int32_t>(fit["n_unit"]);
          h.n_theta = nb::cast<std::int32_t>(fit["n_theta"]);
          h.beta = take_field<double>(fit, "beta");
          h.unit_effect = take_field<double>(fit, "unit_effect");
          h.n_node = nb::cast<std::int32_t>(fit["n_node"]);
          h.node_theta = take_field<double>(fit, "node_theta");
          h.node_weight = take_field<double>(fit, "node_weight");
          h.node_field = take_field<double>(fit, "node_field");
          h.n_field = nb::cast<std::int32_t>(fit["n_field"]);
          const std::vector<double> centre = take_field<double>(fit, "centre");
          const std::vector<double> box_centre = take_field<double>(fit, "box_centre");
          const std::vector<double> box_half = take_field<double>(fit, "box_half");
          for (int c = 0; c < 2; ++c) {
            h.centre[c] = centre[static_cast<std::size_t>(c)];
            h.box_centre[c] = box_centre[static_cast<std::size_t>(c)];
            h.box_half[c] = box_half[static_cast<std::size_t>(c)];
          }
          h.scale = nb::cast<double>(fit["scale"]);
          h.m = nb::cast<std::int32_t>(fit["m"]);
          h.location = take_field<double>(fit, "location");
          h.n_location = nb::cast<std::int32_t>(fit["n_location"]);
          h.neighbours = nb::cast<std::int32_t>(fit["neighbours"]);
          h.cov = nb::cast<std::int32_t>(fit["cov"]);
          std::vector<double> out(newx.shape(0));
          timesift::hierarchical_predict(h, newx.data(), newx.shape(0), newx.shape(1),
                                         unit.has_value() ? unit->data() : nullptr,
                                         coords.has_value() ? coords->data() : nullptr, out.data());
          return give(std::move(out));
        },
        nb::arg("fit"), nb::arg("newx"), nb::arg("unit") = nb::none(),
        nb::arg("coords") = nb::none());
}
