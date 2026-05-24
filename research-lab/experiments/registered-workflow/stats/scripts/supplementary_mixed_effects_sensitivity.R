#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: supplementary_mixed_effects_sensitivity.R <mixed_model_input.csv> <tables_dir>")
}

input_path <- args[[1]]
tables_dir <- args[[2]]
dir.create(tables_dir, recursive = TRUE, showWarnings = FALSE)

if (!requireNamespace("lme4", quietly = TRUE)) {
  stop("Required R package not installed: lme4")
}

data <- read.csv(input_path, stringsAsFactors = FALSE)
required <- c("metric_value", "metric", "model", "split", "seed", "checkpoint_policy", "dataset", "run_id")
missing <- setdiff(required, names(data))
if (length(missing) > 0) {
  stop(paste("Missing required columns:", paste(missing, collapse = ", ")))
}

data$metric_value <- as.numeric(data$metric_value)
if (any(is.na(data$metric_value))) {
  stop("metric_value contains NA/non-numeric values")
}

data$model <- factor(data$model)
data$split <- factor(data$split)
data$seed <- factor(data$seed)
data$checkpoint_policy <- factor(data$checkpoint_policy)
data$dataset <- factor(data$dataset)
data$model_split <- interaction(data$model, data$split, drop = TRUE)

fit_model <- function(df, formula_text) {
  fit_type <- if (grepl("\\|", formula_text)) "lmer" else "lm"
  tryCatch({
    if (fit_type == "lmer") {
      model <- lme4::lmer(
        as.formula(formula_text),
        data = df,
        REML = TRUE,
        control = lme4::lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 100000))
      )
      singular <- lme4::isSingular(model, tol = 1e-4)
      list(ok = TRUE, model = model, fit_type = fit_type, singular = singular, message = "")
    } else {
      model <- lm(as.formula(formula_text), data = df)
      list(ok = TRUE, model = model, fit_type = fit_type, singular = NA, message = "lm_fallback")
    }
  }, error = function(e) {
    list(ok = FALSE, model = NULL, fit_type = fit_type, singular = NA, message = conditionMessage(e))
  })
}

bind_rows_fill <- function(rows) {
  if (length(rows) == 0) {
    return(data.frame())
  }
  all_names <- unique(unlist(lapply(rows, names)))
  aligned <- lapply(rows, function(df) {
    missing <- setdiff(all_names, names(df))
    for (name in missing) {
      df[[name]] <- NA
    }
    df[all_names]
  })
  do.call(rbind, aligned)
}

fixed_rows <- list()
variance_rows <- list()
status_rows <- list()

append_outputs <- function(result, df, dataset_i, policy_i, analysis_scope, model_spec, formula_text) {
  status_rows[[length(status_rows) + 1]] <<- data.frame(
    analysis_scope = analysis_scope,
    model_spec = model_spec,
    dataset = dataset_i,
    checkpoint_policy = policy_i,
    rows = nrow(df),
    models = length(unique(df$model)),
    splits = length(unique(df$split)),
    seeds = length(unique(df$seed)),
    formula = formula_text,
    fit_type = result$fit_type,
    ok = as.character(result$ok),
    singular = as.character(result$singular),
    message = result$message,
    role = "exploratory_sensitivity_not_primary_replacement",
    stringsAsFactors = FALSE
  )
  if (!isTRUE(result$ok)) {
    return(NULL)
  }

  fixed <- as.data.frame(coef(summary(result$model)))
  fixed$term <- rownames(fixed)
  rownames(fixed) <- NULL
  names(fixed) <- gsub(" ", "_", names(fixed))
  fixed$analysis_scope <- analysis_scope
  fixed$model_spec <- model_spec
  fixed$dataset <- dataset_i
  fixed$checkpoint_policy <- policy_i
  fixed_rows[[length(fixed_rows) + 1]] <<- fixed

  if (result$fit_type == "lmer") {
    vc <- as.data.frame(lme4::VarCorr(result$model))
    vc$analysis_scope <- analysis_scope
    vc$model_spec <- model_spec
    vc$dataset <- dataset_i
    vc$checkpoint_policy <- policy_i
    total_var <- sum(vc$vcov)
    vc$variance_proportion <- vc$vcov / total_var
    variance_rows[[length(variance_rows) + 1]] <<- vc
  }
  NULL
}

stratified_specs <- list(
  list(
    name = "sap_full_formula_attempt",
    formula = "metric_value ~ model + (1 | split) + (1 | seed) + (1 | model:split)"
  ),
  list(
    name = "primary_reported_fallback_model",
    formula = "metric_value ~ model + (1 | split) + (1 | model:split)"
  ),
  list(
    name = "simpler_split_random_intercept",
    formula = "metric_value ~ model + (1 | split)"
  ),
  list(
    name = "fixed_effects_only_lm_fallback",
    formula = "metric_value ~ model"
  )
)

strata <- unique(data[c("dataset", "checkpoint_policy")])
for (i in seq_len(nrow(strata))) {
  dataset_i <- as.character(strata$dataset[[i]])
  policy_i <- as.character(strata$checkpoint_policy[[i]])
  df <- droplevels(data[data$dataset == dataset_i & data$checkpoint_policy == policy_i, ])
  for (spec in stratified_specs) {
    result <- fit_model(df, spec$formula)
    append_outputs(result, df, dataset_i, policy_i, "dataset_checkpoint_policy_stratified", spec$name, spec$formula)
  }
}

joint_specs <- list(
  list(
    name = "joint_checkpoint_fixed_effect_model",
    formula = "metric_value ~ model * checkpoint_policy + (1 | split) + (1 | model_split)"
  ),
  list(
    name = "joint_checkpoint_fixed_effect_simpler_split",
    formula = "metric_value ~ model * checkpoint_policy + (1 | split)"
  ),
  list(
    name = "joint_checkpoint_fixed_effect_lm",
    formula = "metric_value ~ model * checkpoint_policy"
  )
)

for (dataset_i in unique(as.character(data$dataset))) {
  df <- droplevels(data[data$dataset == dataset_i, ])
  for (spec in joint_specs) {
    result <- fit_model(df, spec$formula)
    append_outputs(result, df, dataset_i, "A_and_B_joint", "dataset_joint_checkpoint_policy", spec$name, spec$formula)
  }
}

if (length(status_rows) > 0) {
  write.csv(bind_rows_fill(status_rows), file.path(tables_dir, "mixed_effects_sensitivity_model_status.csv"), row.names = FALSE)
}
if (length(fixed_rows) > 0) {
  write.csv(bind_rows_fill(fixed_rows), file.path(tables_dir, "mixed_effects_sensitivity_fixed_effects.csv"), row.names = FALSE)
}
if (length(variance_rows) > 0) {
  write.csv(bind_rows_fill(variance_rows), file.path(tables_dir, "mixed_effects_sensitivity_variance_components.csv"), row.names = FALSE)
}

summary <- data.frame(
  item = c(
    "analysis_role",
    "fold_random_effect_status",
    "primary_interpretation_boundary"
  ),
  value = c(
    "exploratory_sensitivity_not_primary_replacement",
    "not_applicable_no_stable_fold_level_in_registered_repeated_holdout_design",
    "Sensitivity models assess dependence on statistical model specification and do not alter the primary outcome construct."
  ),
  stringsAsFactors = FALSE
)
write.csv(summary, file.path(tables_dir, "mixed_effects_sensitivity_interpretation_record.csv"), row.names = FALSE)

cat("SUPPLEMENTARY_MIXED_EFFECTS_SENSITIVITY_OK\n")
