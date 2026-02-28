 #pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "preprocessing.h"
#include "utils.h"

namespace mof::feature_engineering {

// ============================================================
// Descriptor selection (from numeric descriptor tables)
// ============================================================

enum class DescriptorSelectionMode {
    All,            // keep all descriptor columns
    ByIndex,        // select explicitly by column indices
    ByName,         // select explicitly by feature names
    ExcludeByIndex, // drop listed indices
    ExcludeByName   // drop listed names
};

struct DescriptorSelectionOptions {
    DescriptorSelectionMode mode = DescriptorSelectionMode::All;

    std::vector<std::size_t> include_indices{};
    std::vector<std::string> include_names{};

    std::vector<std::size_t> exclude_indices{};
    std::vector<std::string> exclude_names{};

    bool preserve_order = true;      // preserve original feature order when applicable
    bool require_all_names = true;   // throw if a requested name is missing
};

/// Resolve selected descriptor indices from a numeric dataset.
std::vector<std::size_t> select_descriptor_indices(
    const preprocessing::NumericDataset& dataset,
    const DescriptorSelectionOptions& options = {}
);

/// Return a new dataset with only selected descriptors/features.
/// Target values (if present) are preserved.
preprocessing::NumericDataset select_descriptors(
    const preprocessing::NumericDataset& dataset,
    const DescriptorSelectionOptions& options = {}
);

// ============================================================
// Derived feature creation (generic tabular feature engineering)
// ============================================================

enum class DerivedFeatureKind {
    PolynomialDegree2,   // x^2
    Log1p,               // log(1 + x), requires x > -1
    Sqrt,                // sqrt(x), requires x >= 0
    Inverse,             // 1/x, guarded by epsilon
    PairwiseProduct,     // x_i * x_j
    PairwiseRatio,       // x_i / x_j, guarded by epsilon
    PairwiseDifference,  // x_i - x_j
    PairwiseSum          // x_i + x_j
};

struct DerivedFeatureOptions {
    // Which transformations to enable
    bool add_square = false;
    bool add_log1p = false;
    bool add_sqrt = false;
    bool add_inverse = false;

    bool add_pairwise_product = false;
    bool add_pairwise_ratio = false;
    bool add_pairwise_difference = false;
    bool add_pairwise_sum = false;

    // Restrict transformations to these source feature columns; empty => all
    std::vector<std::size_t> source_feature_indices{};

    // Pairwise generation controls
    bool pairwise_upper_triangle_only = true; // use i < j only
    std::size_t max_pairwise_features = 0;    // 0 => no explicit cap

    // Numerical safety
    double eps = 1e-12;

    // Behavior on invalid math domain (e.g., log1p(x<=-1), sqrt(x<0))
    // If true, skip generating that derived feature column entirely.
    // If false, implementations may throw.
    bool skip_invalid_transform_columns = true;
};

struct DerivedFeatureSummary {
    std::size_t original_feature_count = 0;
    std::size_t added_feature_count = 0;
    std::size_t final_feature_count = 0;

    // Names of newly created features (optional to populate in implementation)
    std::vector<std::string> added_feature_names{};
};

/// Create derived features and return a new engineered dataset.
/// Existing features are preserved; new features are appended.
preprocessing::NumericDataset create_derived_features(
    const preprocessing::NumericDataset& dataset,
    const DerivedFeatureOptions& options,
    DerivedFeatureSummary* summary = nullptr
);

// ============================================================
// Feature scaling hooks (optional separate stage)
// These hooks are intentionally similar to preprocessing
// normalization so scaling can be done here if desired.
// ============================================================

enum class FeatureScalingMethod {
    None,
    MinMax,
    ZScore
};

struct FeatureScalingOptions {
    FeatureScalingMethod method = FeatureScalingMethod::None;

    // Columns in dataset.features to scale; empty => all
    std::vector<std::size_t> feature_column_indices{};

    // Min-max range (used when method == MinMax)
    double min_out = 0.0;
    double max_out = 1.0;

    // Numerical stability (used when method == ZScore)
    double eps = 1e-12;
};

struct FeatureScalingArtifacts {
    utils::ColumnStats feature_stats{};             // one entry per feature column
    std::vector<std::size_t> scaled_columns{};      // columns actually scaled
    FeatureScalingMethod method = FeatureScalingMethod::None;
};

/// Fit scaling statistics on the given dataset and scale in-place.
FeatureScalingArtifacts fit_and_apply_feature_scaling(
    preprocessing::NumericDataset& dataset,
    const FeatureScalingOptions& options = {}
);

/// Apply previously fitted feature scaling to another dataset (e.g., validation/test/inference).
void apply_feature_scaling(
    preprocessing::NumericDataset& dataset,
    const FeatureScalingArtifacts& artifacts,
    const FeatureScalingOptions& options = {}
);

// ============================================================
// Feature matrix export
// ============================================================

struct FeatureExportOptions {
    bool write_header = true;
    char delimiter = ',';

    // If true and target exists, append target column to exported CSV
    bool include_target = true;

    // Optional row identifier column (not generated by default)
    bool include_row_index = false;
    std::string row_index_column_name = "row_id";
};

/// Export engineered features (and optional target) to CSV.
/// Typical output path: data/features/<name>.csv
void export_feature_matrix(
    const preprocessing::NumericDataset& dataset,
    std::string_view output_csv_path,
    const FeatureExportOptions& options = {}
);

/// Export only X matrix (features) to CSV, without target.
void export_feature_matrix_only(
    const utils::Matrix& features,
    const std::vector<std::string>& feature_names,
    std::string_view output_csv_path,
    const FeatureExportOptions& options = {}
);

// ============================================================
// MOF-specific descriptor placeholders (future extension)
// ============================================================

struct MofDescriptorPlaceholders {
    // Future flags for MOF-specific structural descriptor extraction
    bool use_composition_descriptors = true;
    bool use_porosity_descriptors = true;
    bool use_geometric_descriptors = true;
    bool use_bond_type_descriptors = true;
    bool use_coordination_descriptors = false; // future
};

/// Placeholder for future CIF-driven descriptor extraction.
/// Expected future behavior:
///   - read MOF structure records (e.g., CIF metadata / parsed structures)
///   - generate a numeric descriptor dataset for downstream modeling
/// For now, implementations may throw "not implemented".
preprocessing::NumericDataset build_mof_descriptors_from_cif_placeholder(
    const std::vector<std::string>& cif_paths,
    const MofDescriptorPlaceholders& options = {}
);

/// Placeholder hook for adding MOF domain-specific derived descriptors
/// (e.g., coordination number summaries, pore-size ratios, chemistry interactions).
preprocessing::NumericDataset add_mof_specific_derived_features_placeholder(
    const preprocessing::NumericDataset& dataset,
    const MofDescriptorPlaceholders& options = {}
);

} // namespace mof::feature_engineering