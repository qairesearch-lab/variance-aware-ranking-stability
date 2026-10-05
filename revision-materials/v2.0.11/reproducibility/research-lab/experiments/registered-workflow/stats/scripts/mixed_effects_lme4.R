#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: mixed_effects_lme4.R <mixed_model_input.csv> <output_dir>")
}

input_path <- args[[1]]
output_dir <- args[[2]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

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

sanitize <- function(value) {
  gsub("[^A-Za-z0-9_]+", "_", value)
}

write_empty_csv <- function(path, columns) {
  empty <- as.data.frame(setNames(replicate(length(columns), character(), simplify = FALSE), columns))
  write.csv(empty, path, row.names = FALSE)
}

fit_candidates <- function(df) {
  n_split <- length(unique(df$split))
  n_seed <- length(unique(df$seed))
  n_model_split <- length(unique(interaction(df$model, df$split, drop = TRUE)))
  candidates <- list()
  if (n_split > 1 && n_seed > 1 && n_model_split > 1) {
    candidates <- append(candidates, "metric_value ~ model + (1 | split) + (1 | seed) + (1 | model:split)")
  }
  if (n_split > 1 && n_model_split > 1) {
    candidates <- append(candidates, "metric_value ~ model + (1 | split) + (1 | model:split)")
  }
  if (n_split > 1 && n_seed > 1) {
    candidates <- append(candidates, "metric_value ~ model + (1 | split) + (1 | seed)")
  }
  if (n_split > 1) {
    candidates <- append(candidates, "metric_value ~ model + (1 | split)")
  }
  if (n_model_split > 1) {
    candidates <- append(candidates, "metric_value ~ model + (1 | model:split)")
  }
  append(candidates, "metric_value ~ model")
}

fit_one <- function(df) {
  formulas <- fit_candidates(df)
  records <- list()
  for (formula_text in formulas) {
    fit_type <- if (grepl("\\|", formula_text)) "lmer" else "lm"
    result <- tryCatch({
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
    records <- append(records, list(data.frame(
      formula = formula_text,
      fit_type = fit_type,
      ok = result$ok,
      singular = as.character(result$singular),
      message = result$message,
      stringsAsFactors = FALSE
    )))
    if (isTRUE(result$ok) && (!isTRUE(result$singular) || fit_type == "lm")) {
      return(list(model = result$model, formula = formula_text, fit_type = fit_type, singular = result$singular, attempts = do.call(rbind, records)))
    }
  }
  last <- tail(records, 1)[[1]]
  stop(paste("All model attempts failed; last formula:", last$formula, "message:", last$message))
}

data$model <- factor(data$model)
data$split <- factor(data$split)
data$seed <- factor(data$seed)
data$checkpoint_policy <- factor(data$checkpoint_policy)
data$dataset <- factor(data$dataset)

fixed_rows <- list()
variance_rows <- list()
status_rows <- list()
attempt_rows <- list()

strata <- unique(data[c("dataset", "checkpoint_policy")])
for (i in seq_len(nrow(strata))) {
  dataset_i <- as.character(strata$dataset[[i]])
  policy_i <- as.character(strata$checkpoint_policy[[i]])
  df <- data[data$dataset == dataset_i & data$checkpoint_policy == policy_i, ]
  df <- droplevels(df)
  stratum_id <- sanitize(paste(dataset_i, policy_i, sep = "_"))
  fit <- fit_one(df)
  model_path <- file.path(output_dir, paste0("mixed_effects_model_", stratum_id, ".rds"))
  saveRDS(fit$model, model_path)

  if (fit$fit_type == "lmer") {
    fixed <- as.data.frame(coef(summary(fit$model)))
    fixed$term <- rownames(fixed)
    rownames(fixed) <- NULL
    names(fixed) <- gsub(" ", "_", names(fixed))
    fixed$dataset <- dataset_i
    fixed$checkpoint_policy <- policy_i
    fixed_rows <- append(fixed_rows, list(fixed))

    vc <- as.data.frame(lme4::VarCorr(fit$model))
    vc$dataset <- dataset_i
    vc$checkpoint_policy <- policy_i
    total_var <- sum(vc$vcov)
    vc$variance_proportion <- vc$vcov / total_var
    variance_rows <- append(variance_rows, list(vc))
  } else {
    fixed <- as.data.frame(coef(summary(fit$model)))
    fixed$term <- rownames(fixed)
    rownames(fixed) <- NULL
    names(fixed) <- gsub(" ", "_", names(fixed))
    fixed$dataset <- dataset_i
    fixed$checkpoint_policy <- policy_i
    fixed_rows <- append(fixed_rows, list(fixed))
    variance_rows <- append(variance_rows, list(data.frame(
      grp = "Residual",
      var1 = NA,
      var2 = NA,
      vcov = sigma(fit$model)^2,
      sdcor = sigma(fit$model),
      dataset = dataset_i,
      checkpoint_policy = policy_i,
      variance_proportion = 1.0
    )))
  }

  attempts <- fit$attempts
  attempts$dataset <- dataset_i
  attempts$checkpoint_policy <- policy_i
  attempt_rows <- append(attempt_rows, list(attempts))

  status_rows <- append(status_rows, list(data.frame(
    dataset = dataset_i,
    checkpoint_policy = policy_i,
    rows = nrow(df),
    models = length(unique(df$model)),
    splits = length(unique(df$split)),
    seeds = length(unique(df$seed)),
    formula = fit$formula,
    fit_type = fit$fit_type,
    singular = as.character(fit$singular),
    model_object = model_path,
    status = "completed",
    interpretation_note = "Variance proportions are exploratory and should be interpreted cautiously with 10 split levels.",
    stringsAsFactors = FALSE
  )))
}

fixed_path <- file.path(output_dir, "mixed_effects_fixed_effects.csv")
variance_path <- file.path(output_dir, "mixed_effects_variance_components.csv")
status_path <- file.path(output_dir, "mixed_effects_model_status.csv")
attempts_path <- file.path(output_dir, "mixed_effects_model_attempts.csv")

if (length(fixed_rows) > 0) {
  write.csv(do.call(rbind, fixed_rows), fixed_path, row.names = FALSE)
} else {
  write_empty_csv(fixed_path, c("term", "dataset", "checkpoint_policy"))
}
if (length(variance_rows) > 0) {
  write.csv(do.call(rbind, variance_rows), variance_path, row.names = FALSE)
} else {
  write_empty_csv(variance_path, c("grp", "vcov", "dataset", "checkpoint_policy", "variance_proportion"))
}
write.csv(do.call(rbind, status_rows), status_path, row.names = FALSE)
write.csv(do.call(rbind, attempt_rows), attempts_path, row.names = FALSE)
capture.output(sessionInfo(), file = file.path(output_dir, "mixed_effects_session_info.txt"))

cat("MIXED_EFFECTS_LME4_OK\n")
