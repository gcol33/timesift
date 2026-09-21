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

#include "ts_core.h"
#include "ts_penalised.h"

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

}  // namespace

NB_MODULE(_core, m) {
  m.doc() = "The binning, the reduction and the penalised fit, shared with the R package as "
            "src/ts_core.cpp and src/ts_penalised.cpp.";

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

  m.def("year_fraction",
        [](ConstI64 bin_start, ConstI64 bin_end) {
          std::vector<double> frac(bin_start.size());
          timesift::year_fraction(bin_start.data(), bin_end.data(), bin_start.size(),
                                   frac.data());
          return give(std::move(frac));
        },
        nb::arg("bin_start"), nb::arg("bin_end"));

  m.def("year_phase",
        [](ConstI64 bin_start, ConstI64 bin_end) {
          std::vector<double> year_sin(bin_start.size()), year_cos(bin_start.size());
          timesift::year_phase(bin_start.data(), bin_end.data(), bin_start.size(),
                                year_sin.data(), year_cos.data());
          return nb::make_tuple(give(std::move(year_sin)), give(std::move(year_cos)));
        },
        nb::arg("bin_start"), nb::arg("bin_end"));

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
}
