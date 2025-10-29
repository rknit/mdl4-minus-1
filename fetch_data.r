#!/usr/bin/env Rscript
# ==============================================================
# MDL4Microbiome-style Multi-Modality Exporter (Sparse-only)
# ==============================================================
# All modalities exported as sparse triplets (.csv.gz) to avoid OOM.
# Gene-family behavior controlled by GENE_FAMILY_MODE:
#   "skip"        → omit gene_families entirely
#   "use_pathway" → reuse pathway_abundance as functional modality
#   "full"        → include and aggregate gene_families (may use lots of RAM)
# ==============================================================

# ------------------------ CONFIG -------------------------------
MAX_SAMPLES        <- 2000
MIN_PREVALENCE     <- 0.05
TOP_N_FEATURES     <- 10000
VARIANCE_THRESHOLD <- 1e-6
SEED               <- 123
GENE_FAMILY_MODE   <- "use_pathway"  # "skip", "use_pathway", "full"

DISEASES <- list(
  list(study_id = "QinJ_2012", label = "T2D"),
  list(study_id = "YachidaS_2019", label = "CRC"),
  list(study_id = "NielsenHB_2014", label = "IBD"),
  list(study_id = "QinN_2014", label = "cirrhosis")
)
# ---------------------------------------------------------------

suppressPackageStartupMessages({
  library(curatedMetagenomicData)
  library(SummarizedExperiment)
  library(dplyr)
  library(Matrix)
  library(data.table)
})

export_dataset <- function(study_id, disease_label, normal_label = "control") {
  dir.create("data", showWarnings = FALSE)
  out_dir <- file.path("data", disease_label)
  dir.create(out_dir, showWarnings = FALSE)

  # --- Helper to ensure sparse ---
  to_sparse <- function(mat) {
    if (!inherits(mat, "dgCMatrix")) mat <- Matrix(mat, sparse = TRUE)
    return(mat)
  }

  # --- Fetch modalities ---
  get_se <- function(suffix) {
    lst <- curatedMetagenomicData(paste0(study_id, ".", suffix), dryrun = FALSE)
    if (length(lst) == 0) return(NULL)
    lst[[1]]
  }

  species_se      <- get_se("relative_abundance")
  pathways_se     <- get_se("pathway_abundance")
  genefamilies_se <- if (GENE_FAMILY_MODE == "full") get_se("gene_families") else NULL

  # --- Force all assays to sparse ---
  if (!is.null(species_se)) assay(species_se)      <- to_sparse(assay(species_se))
  if (!is.null(pathways_se)) assay(pathways_se)    <- to_sparse(assay(pathways_se))
  if (!is.null(genefamilies_se)) assay(genefamilies_se) <- to_sparse(assay(genefamilies_se))

  if (is.null(species_se) || is.null(pathways_se)) {
    message("⚠️ Missing required modalities for ", study_id, ", skipping.")
    return(invisible(NULL))
  }

  if (GENE_FAMILY_MODE == "skip") message("⚠️ Skipping gene_families for ", study_id)
  if (GENE_FAMILY_MODE == "use_pathway") message("ℹ️ Using pathway_abundance as functional modality")
  if (GENE_FAMILY_MODE == "full") message("🔬 Including raw gene_families (may be memory heavy)")

  # --- Common samples ---
  valid_list <- list(colnames(species_se), colnames(pathways_se))
  if (GENE_FAMILY_MODE == "full") valid_list <- c(valid_list, list(colnames(genefamilies_se)))
  common_samples <- Reduce(intersect, valid_list)

  metadata <- as.data.frame(colData(species_se))
  valid_samples <- rownames(metadata)[metadata$study_condition %in% c(disease_label, normal_label)]
  keep_samples  <- intersect(common_samples, valid_samples)
  if (length(keep_samples) == 0) stop("No samples found for ", disease_label)

  subset_modality <- function(se) se[, keep_samples]
  species_se  <- subset_modality(species_se)
  pathways_se <- subset_modality(pathways_se)
  if (GENE_FAMILY_MODE == "full") genefamilies_se <- subset_modality(genefamilies_se)
  metadata <- metadata[keep_samples, , drop = FALSE]

  # --- Balance classes ---
  disease_samp <- rownames(metadata)[metadata$study_condition == disease_label]
  control_samp <- rownames(metadata)[metadata$study_condition == normal_label]
  n_each <- min(floor(MAX_SAMPLES / 2), length(disease_samp), length(control_samp))
  set.seed(SEED)
  sel_samples <- c(sample(disease_samp, n_each), sample(control_samp, n_each))
  species_se  <- species_se[, sel_samples]
  pathways_se <- pathways_se[, sel_samples]
  if (GENE_FAMILY_MODE == "full") genefamilies_se <- genefamilies_se[, sel_samples]
  metadata <- metadata[sel_samples, , drop = FALSE]

  message(sprintf("✅ Exporting %d samples for %s (%d each)", nrow(metadata), disease_label, n_each))

  # --- Sparse modality writer ---
  write_sparse_modality <- function(mat, name, kegg_info = NULL) {
    # KEGG aggregation if applicable
    if (name == "modality_3_genefamilies" && !is.null(kegg_info)) {
      kegg_map <- kegg_info$kegg
      uniq_kegg <- unique(kegg_map)
      mat_new <- Matrix(0, nrow = length(uniq_kegg), ncol = ncol(mat), sparse = TRUE)
      rownames(mat_new) <- uniq_kegg
      names_k <- split(seq_along(kegg_map), kegg_map)
      for (k in names(names_k)) {
        if (is.na(k)) next
        idx <- names_k[[k]]
        mat_new[k, ] <- Matrix::colSums(mat[idx, , drop = FALSE])
      }
      mat <- mat_new
      message("    • Aggregated into KEGG pathways: ", nrow(mat))
    }

    # Normalize + log-transform
    col_sums <- Matrix::colSums(mat, na.rm = TRUE)
    col_sums[col_sums == 0] <- 1
    mat <- Matrix::t(Matrix::t(mat) / col_sums)
    mat@x <- log1p(mat@x * 1e6)

    # Prevalence + variance filtering
    prevalence <- Matrix::rowMeans(mat > 0)
    keep_idx <- which(prevalence >= MIN_PREVALENCE)
    mat <- mat[keep_idx, , drop = FALSE]

    row_means <- Matrix::rowMeans(mat)
    row_means_sq <- Matrix::rowMeans(mat^2)
    vars <- row_means_sq - row_means^2
    keep_idx <- which(vars > VARIANCE_THRESHOLD)
    mat <- mat[keep_idx, , drop = FALSE]
    vars <- vars[keep_idx]

    if (length(vars) > TOP_N_FEATURES) {
      top_idx <- order(vars, decreasing = TRUE)[1:TOP_N_FEATURES]
      mat <- mat[top_idx, , drop = FALSE]
    }

    message(sprintf("    • %s: %d features retained", name, nrow(mat)))

    # Export as sparse triplet
    triplet <- summary(mat)
    colnames(triplet) <- c("row", "col", "value")
    out_path <- file.path(out_dir, paste0(name, "_sparse.csv.gz"))
    fwrite(triplet, out_path, compress = "gzip")
    message("    • Saved sparse triplet (", nrow(triplet), " nonzeros)")
    return(out_path)
  }

  # --- Process modalities ---
  p1 <- write_sparse_modality(assay(species_se), "modality_1_species")
  p2 <- write_sparse_modality(assay(pathways_se), "modality_2_pathways")

  if (GENE_FAMILY_MODE == "skip") {
    p3 <- NULL
  } else if (GENE_FAMILY_MODE == "use_pathway") {
    p3 <- write_sparse_modality(assay(pathways_se), "modality_3_functional")
  } else if (GENE_FAMILY_MODE == "full") {
    kegg_info <- as.data.frame(rowData(genefamilies_se))
    p3 <- write_sparse_modality(assay(genefamilies_se), "modality_3_genefamilies", kegg_info)
  }

  rel_paths <- basename(c(p1, p2, p3))
  rel_paths <- rel_paths[!is.na(rel_paths)]
  writeLines(rel_paths, file.path(out_dir, "datasets.txt"))

  # --- Metadata ---
  meta_cols <- c("subject_id", "body_site", "age", "gender", "bmi")
  meta_avail <- meta_cols[meta_cols %in% colnames(metadata)]
  if (length(meta_avail) > 0) {
    meta_df <- metadata %>%
      select(all_of(meta_avail)) %>%
      mutate(
        gender = as.numeric(factor(gender)),
        body_site = as.numeric(factor(body_site))
      ) %>%
      select(-any_of("subject_id"))
    fwrite(meta_df, file.path(out_dir, "metadata.csv"))
  }

  writeLines(as.character(metadata$study_condition), file.path(out_dir, "ylab.txt"))
  message("--- ✅ Finished:", disease_label, "---\n")
}

# --- Run all ---
for (d in DISEASES) {
  export_dataset(d$study_id, d$label)
}

message("🎉 All datasets exported as sparse triplets under data/")
