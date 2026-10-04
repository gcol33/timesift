# The learners, response heads and metrics that ship are registered here, through the same public
# calls a user registers their own with. There is no second, privileged path into the registries.

# `self` is bound by torch inside a module's own methods rather than by this package, so the code
# checker has nothing to resolve it against.
utils::globalVariables("self")

.onLoad <- function(libname, pkgname) {
  register_metric("tss", tss)
  register_metric("roc_auc", roc_auc)
  register_metric("average_precision", average_precision)
  register_metric("kappa", function(y, p) kappa_score(y, p, "prevalence"))
  register_metric("kappa_youden", function(y, p) kappa_score(y, p, "youden"))
  register_metric("boyce", boyce_index)
  for (name in names(.table_metrics)) {
    register_metric(name, local({
      metric <- name
      function(y, p) table_metric(y, p, metric)
    }))
  }

  register_response("presence_absence", .presence_absence)
  heads <- .numeric_heads()
  for (name in names(heads)) {
    register_response(name, heads[[name]])
  }
  for (name in c("r_squared", "pearson")) {
    register_metric(name, local({
      metric <- name
      function(y, p) regression_metric(y, p, metric)
    }))
  }
  for (name in c("rmse", "mse", "mae", "max_error", "poisson_deviance")) {
    register_metric(paste0("neg_", name), local({
      metric <- name
      function(y, p) -regression_metric(y, p, metric)
    }))
  }
  for (name in names(.ordinal_metrics)) {
    register_metric(paste0("ordinal_", name), local({
      metric <- name
      function(y, p) ordinal_metric(y, p, metric)
    }))
  }

  grids <- .default_grids()
  for (name in names(grids)) {
    register_tuning(name, grids[[name]])
  }

  register_learner("elasticnet", elasticnet)
  register_learner("linear", linear)
  register_learner("forest", forest)
  register_learner("tree", tree)
  register_learner("boosting", boosting)
  register_learner("maxent", maxent)
  register_learner("envelope", envelope)
  register_learner("mars", mars)
  register_learner("discriminant", discriminant)
  register_learner("additive", additive)
  register_learner("perceptron", perceptron)
  register_learner("hierarchical", hierarchical)
  register_learner("mlp", mlp)
  register_learner("cnn", cnn)
  register_learner("rescnn", rescnn)
  invisible(NULL)
}
