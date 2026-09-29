#ifndef TIMESIFT_TS_MAXNET_H
#define TIMESIFT_TS_MAXNET_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "ts_penalised.h"

// maxnet, once, for both languages.
//
// maxnet's feature classes over every column of a block, its regularisation of each feature, and a
// lasso over them on the penalised core. Two formulations share the features and the penalty
// factors. `background` is maxnet's own: every unit is background, each presence is added to the
// background again unless an absence carries the same readings, the background is weighted 100
// against a presence's 1, and the fit is read at the last of maxnet's 200 penalties, with the
// intercept replaced by the normaliser over the background. `absence` reads the absences as
// absences: a logistic lasso under the caller's case weights over the same features and factors,
// its penalty chosen by cross-validated deviance.
//
// A column holding one value over the units carries nothing and would divide a hinge by zero, so it
// takes no feature. Everything else is maxnet's own arithmetic, including its knots: forward and
// reverse hinges at the interior of `knots` equally spaced points of a column's range, and
// thresholds at the 49 interior points maxnet cuts from `knots + 2` points of it.
namespace timesift {

enum class MaxnetKind : std::int8_t {
  linear = 0,
  quadratic = 1,
  hinge = 2,       // (x - lo) / (hi - lo), clamped to [0, 1]
  threshold = 3,   // x >= lo
  product = 4      // x_a * x_b
};

enum class MaxnetFormulation { background, absence };

MaxnetFormulation maxnet_formulation_from_name(const std::string& name);
const char* maxnet_formulation_name(MaxnetFormulation f);

// The feature classes maxnet fits at a presence count: fewer than 10 linear, fewer than 15 linear
// and quadratic, fewer than 80 those and hinges, and products beside them from 80 on.
std::string maxnet_default_classes(std::size_t n_presence);

// Features over the columns of a block, in the order maxnet's model matrix holds them: every linear
// term, every quadratic, each column's hinges (forward, then reverse), each column's thresholds,
// and the products of each pair of columns.
struct MaxnetFeatures {
  std::vector<std::int8_t> kind;
  std::vector<std::int32_t> a, b;   // the columns read; `b` is -1 but for a product
  std::vector<double> lo, hi;       // a hinge's two ends, a threshold's knot in `lo`
  std::size_t size() const { return kind.size(); }
};

// The features of `classes` over the columns of `x` [n, p] that hold more than one value, with
// knots from each column's range over the rows given.
MaxnetFeatures maxnet_features(const double* x, std::size_t n, std::size_t p,
                               const std::string& classes, int knots);

// The value of one feature at one row of a column-major block.
double maxnet_feature_value(const MaxnetFeatures& f, std::size_t k, const double* x,
                            std::size_t n, std::size_t i);

// The features over every row, column-major [n, features].
void maxnet_expand(const MaxnetFeatures& f, const double* x, std::size_t n, double* out);

// maxnet's default regularisation of each feature (`maxnet.default.regularization`) times
// `regmult`, from the design and the rows marked present.
std::vector<double> maxnet_regularization(const MaxnetFeatures& f, const double* design,
                                          std::size_t n, const double* presence, double regmult);

struct MaxnetSpec {
  std::string classes;       // letters of "lqpht"; empty for maxnet's choice at the presence count
  int knots = 50;
  double regmult = 1.0;
  MaxnetFormulation formulation = MaxnetFormulation::background;
  bool add_samples = true;   // background: presences join the background, as maxnet's default
  double thresh = 1e-8;
  int max_pass = 1000000;
  int n_lambda = 100;        // absence: points of the derived path
  bool one_se = false;       // absence: read the path at the largest penalty within one standard
                             // error of the least held-out deviance rather than at the least
  int threads = 1;           // absence: fits of the cross-validation at once
  double max_design = 2.0;   // gigabytes the expanded design may take; the penalised core holds a
                             // centred copy beside it, so a fit needs about twice this
};

// What a fit is handed to the penalised core: the rows it fits, background rows added, with their
// response, the features and the design over those rows, and each feature's penalty factor.
struct MaxnetDesign {
  std::string classes;
  std::size_t m = 0;                // rows fitted
  std::vector<std::size_t> rows;    // the unit each row reads
  std::vector<double> y, x;         // x [m, p]
  MaxnetFeatures features;
  std::vector<double> design;       // [m, features]
  std::vector<double> reg;
};

MaxnetDesign maxnet_design(const double* x, const double* y, std::size_t n, std::size_t p,
                           const MaxnetSpec& spec);

// A fit keeps the features it gave a coefficient and nothing else.
struct Maxnet {
  MaxnetFormulation formulation = MaxnetFormulation::background;
  std::string classes;
  std::int32_t n_column = 0;
  std::int32_t n_presence = 0;
  std::int32_t n_feature = 0;       // features the lasso was offered
  std::vector<double> var_min, var_max;  // each column's range over the rows fitted
  MaxnetFeatures features;          // those with a coefficient
  std::vector<double> feature_min, feature_max, beta;
  double intercept = 0.0;           // background: maxnet's alpha; absence: the lasso's own
  double lasso_intercept = 0.0;     // the lasso's own either way, which the background discards
  double entropy = 0.0;             // background: the entropy of the fitted background
  double lambda = 0.0;              // the penalty read
  std::int32_t stalled = 0;         // the path's own, as `PenaltyPath::stalled`
  std::int32_t fold_stalled = 0;    // absence: folds whose path stalled
};

// `y` holds zero and one. `w` is read by the absence formulation alone, and `fold` (0-based, one
// per unit) is its cross-validation's; the background formulation takes neither.
Maxnet maxnet_fit(const double* x, const double* y, const double* w, std::size_t n, std::size_t p,
                  const MaxnetSpec& spec, const std::int32_t* fold, std::int32_t n_fold);

enum class MaxnetOutput { link, exponential, cloglog, logistic };

MaxnetOutput maxnet_output_from_name(const std::string& name);

// The prediction at every row of `x` [n, p]. With `clamp`, each column is first held inside the
// range it was fitted on and each feature inside its own, as maxnet's `predict(clamp = TRUE)`.
// The background formulation gives each output maxnet's; the absence formulation gives its link
// and its probability, which is `logistic`.
void maxnet_predict(const Maxnet& fit, const double* x, std::size_t n, std::size_t p, bool clamp,
                    MaxnetOutput type, double* out);

}  // namespace timesift

#endif  // TIMESIFT_TS_MAXNET_H
