#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "utils.h"

namespace mof::preprocessing {

// ------------------------------------------------------------
// Core data containers (MVP, tabular CSV-first)
// ------------------------------------------------------------
struct TabularData {
    std::vector<std::string> column_names;
    std::vector<std::vector<std::string>> rows; // raw string cells, row-major
};

struct NumericDataset {
    std::vector<std::string> feature_names;     // excludes target column
    utils::Matrix features;                     // rows x num_features
    std::vector<double> target;                 // optional target values for supervised learning
    std::string target_name;                    // empty if target not provided
};

struct TrainTestSplit {
    NumericDataset train;
    NumericDataset test;
};

// ------------------------------------------------------------
// CSV loading / saving
// ------------------------------------------------------------
struct CsvLoadOptions {
    bool has_header = true;
    char delimiter = ',';
    bool trim_fields = true;
    bool skip_empty_lines = true;
};

struct CsvSaveOptions {
    bool write_header = true;
    char delimiter = ',';
};

/// Load tabular CSV data into a raw string table.
/// Throws on file-not-found, malformed CSV row, or I/O failure.
TabularData load_csv_data(std::string_view csv_path, const CsvLoadOptions& options = {});

/// Save raw (cleaned) tabular data back to CSV.
void save_cleaned_data(
    const TabularData& table,
    std::string_view output_csv_path,
    const CsvSaveOptions& options = {}
);

/// Save numeric dataset to CSV (features + optional target column).
void save_cleaned_data(
    const NumericDataset& dataset,
    std::string_view output_csv_path,
    const CsvSaveOptions& options = {}
);

// ------------------------------------------------------------
// Missing values / row cleaning
// ------------------------------------------------------------
enum class MissingValueStrategy {
    DropRow,        // Remove rows containing missing cells in selected columns
    FillWithZero,   // Fill numeric-like missing values with 0
    FillWithMean,   // Fill numeric columns using column mean
    FillWithMedian, // Fill numeric columns using column median
    FillWithValue   // Fill with a user-provided constant string
};

struct MissingValueOptions {
    MissingValueStrategy strategy = MissingValueStrategy::DropRow;

    // Columns to process; empty => all columns
    std::vector<std::size_t> column_indices{};

    // Tokens treated as missing (case-sensitive exact match in MVP)
    std::vector<std::string> missing_tokens{"", "NaN", "nan", "NULL", "null"};

    // Used only when strategy == FillWithValue
    std::string fill_value = "0";
};

/// Handle missing values in-place on raw tabular data.
/// Returns number of cells modified (for fill strategies) or rows removed (for DropRow).
std::size_t handle_missing_values(TabularData& table, const MissingValueOptions& options = {});

struct RowValidationOptions {
    bool require_consistent_column_count = true;

    // If non-empty, validate only these columns as numeric.
    // If empty, no numeric validation is applied by this function.
    std::vector<std::size_t> numeric_columns{};

    // Remove rows where specified target column is missing/invalid (if provided).
    std::optional<std::size_t> target_column_index = std::nullopt;
};

/// Remove malformed/invalid rows in-place.
/// Returns number of rows removed.
std::size_t remove_invalid_rows(TabularData& table, const RowValidationOptions& options = {});

// ------------------------------------------------------------
// Conversion to numeric dataset (pre-modeling handoff)
// ------------------------------------------------------------
struct NumericConversionOptions {
    // If set, this column becomes target and is excluded from features.
    std::optional<std::size_t> target_column_index = std::nullopt;

    // If empty, infer all non-target columns as features.
    std::vector<std::size_t> feature_column_indices{};

    // Rows with non-parsable numeric values are dropped if true; otherwise throw.
    bool drop_rows_with_parse_errors = true;
};

/// Convert cleaned tabular data to numeric features/target.
/// Intended after missing-value handling and row validation.
NumericDataset to_numeric_dataset(
    const TabularData& table,
    const NumericConversionOptions& options = {}
);

// ------------------------------------------------------------
// Normalization
// ------------------------------------------------------------
struct NormalizationOptions {
    utils::NormalizationMethod method = utils::NormalizationMethod::ZScore;

    // Columns in `features` to normalize; empty => all feature columns
    std::vector<std::size_t> feature_column_indices{};

    // Min-max output range (used only for MinMax)
    double minmax_out_min = 0.0;
    double minmax_out_max = 1.0;

    // Numerical stability epsilon (used for ZScore)
    double eps = 1e-12;
};

struct NormalizationResult {
    utils::ColumnStats feature_stats;           // one entry per feature column in dataset.features
    std::vector<std::size_t> normalized_columns;
};

/// Normalize selected feature columns in-place.
/// Returns stats needed to reproduce the same normalization on test/inference data later.
NormalizationResult normalize_columns(
    NumericDataset& dataset,
    const NormalizationOptions& options = {}
);

/// Apply previously fitted normalization stats to another dataset (e.g., test set / inference set).
void apply_normalization(
    NumericDataset& dataset,
    const utils::ColumnStats& fitted_feature_stats,
    const NormalizationOptions& options = {}
);

// ------------------------------------------------------------
// Train / test splitting
// ------------------------------------------------------------
struct TrainTestSplitOptions {
    double test_ratio = 0.2;                    // (0,1)
    bool shuffle = true;
    std::optional<std::uint32_t> random_seed = 42;
};

/// Split numeric dataset into train/test partitions.
/// Throws if feature/target sizes mismatch or split ratio is invalid.
TrainTestSplit split_train_test(
    const NumericDataset& dataset,
    const TrainTestSplitOptions& options = {}
);

} // namespace mof::preprocessing