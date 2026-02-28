#include "preprocessing.h"

#include <algorithm>
#include <cerrno>
#include <string_view>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

namespace mof::preprocessing {
namespace {

constexpr double kDefaultEps = 1e-12;

// ------------------------------------------------------------
// Internal helpers
// ------------------------------------------------------------
bool is_missing_token(const std::string& value, const std::vector<std::string>& missing_tokens) {
    for (const auto& token : missing_tokens) {
        if (value == token) {
            return true;
        }
    }
    return false;
}

bool parse_double_strict(std::string_view sv, double& out) {
    const std::string s = utils::trim(sv);
    if (s.empty()) {
        return false;
    }

    char* end_ptr = nullptr;
    const char* begin = s.c_str();
    errno = 0;
    const double value = std::strtod(begin, &end_ptr);

    if (begin == end_ptr) {
        return false; // no conversion
    }
    if (errno == ERANGE) {
        return false; // overflow/underflow
    }
    if (*end_ptr != '\0') {
        return false; // trailing junk
    }
    if (!std::isfinite(value)) {
        return false;
    }

    out = value;
    return true;
}

std::string escape_csv_field(std::string_view field, char delimiter) {
    bool needs_quotes = false;
    for (char ch : field) {
        if (ch == delimiter || ch == '"' || ch == '\n' || ch == '\r') {
            needs_quotes = true;
            break;
        }
    }

    if (!needs_quotes) {
        return std::string(field);
    }

    std::string out;
    out.reserve(field.size() + 2);
    out.push_back('"');
    for (char ch : field) {
        if (ch == '"') {
            out.push_back('"'); // escaped quote
        }
        out.push_back(ch);
    }
    out.push_back('"');
    return out;
}

void write_csv_row(std::ostream& os, const std::vector<std::string>& row, char delimiter) {
    for (std::size_t i = 0; i < row.size(); ++i) {
        if (i > 0) {
            os << delimiter;
        }
        os << escape_csv_field(row[i], delimiter);
    }
    os << '\n';
}

std::size_t infer_column_count(const TabularData& table) {
    if (!table.column_names.empty()) {
        return table.column_names.size();
    }
    if (!table.rows.empty()) {
        return table.rows.front().size();
    }
    return 0;
}

std::vector<std::size_t> make_selected_columns(
    std::size_t total_columns,
    const std::vector<std::size_t>& requested,
    const char* func_name
) {
    if (total_columns == 0) {
        return {};
    }

    std::vector<std::size_t> selected;
    selected.reserve(requested.empty() ? total_columns : requested.size());

    std::vector<bool> seen(total_columns, false);

    if (requested.empty()) {
        for (std::size_t c = 0; c < total_columns; ++c) {
            selected.push_back(c);
        }
        return selected;
    }

    for (std::size_t c : requested) {
        if (c >= total_columns) {
            throw std::out_of_range(
                std::string(func_name) + ": column index out of range: " + std::to_string(c)
            );
        }
        if (!seen[c]) {
            seen[c] = true;
            selected.push_back(c);
        }
    }

    return selected;
}

double compute_mean(const std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    const double sum = std::accumulate(values.begin(), values.end(), 0.0);
    return sum / static_cast<double>(values.size());
}

double compute_median(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }

    const std::size_t n = values.size();
    const std::size_t mid = n / 2;

    std::nth_element(values.begin(), values.begin() + mid, values.end());
    double median = values[mid];

    if ((n % 2) == 0) {
        const auto max_left_it = std::max_element(values.begin(), values.begin() + mid);
        median = (*max_left_it + values[mid]) / 2.0;
    }

    return median;
}

std::string double_to_string(double value) {
    std::ostringstream oss;
    oss << std::setprecision(17) << value;
    return oss.str();
}

bool row_has_required_indices(const std::vector<std::string>& row, const std::vector<std::size_t>& cols) {
    if (cols.empty()) {
        return true;
    }
    const std::size_t row_size = row.size();
    for (std::size_t c : cols) {
        if (c >= row_size) {
            return false;
        }
    }
    return true;
}

std::vector<std::size_t> selected_feature_columns(
    std::size_t feature_count,
    const std::vector<std::size_t>& requested,
    const char* func_name
) {
    return make_selected_columns(feature_count, requested, func_name);
}

void validate_numeric_dataset_shapes(const NumericDataset& dataset, const char* func_name) {
    const std::size_t rows = dataset.features.size();

    // Features matrix should be rectangular.
    if (!dataset.features.empty()) {
        const std::size_t cols = dataset.features.front().size();
        for (std::size_t r = 1; r < dataset.features.size(); ++r) {
            if (dataset.features[r].size() != cols) {
                throw std::invalid_argument(
                    std::string(func_name) + ": features matrix is not rectangular at row " +
                    std::to_string(r)
                );
            }
        }

        if (!dataset.feature_names.empty() && dataset.feature_names.size() != cols) {
            throw std::invalid_argument(
                std::string(func_name) + ": feature_names count mismatch with feature columns"
            );
        }
    }

    if (!dataset.target.empty() && dataset.target.size() != rows) {
        throw std::invalid_argument(
            std::string(func_name) + ": target size mismatch with number of feature rows"
        );
    }
}

std::vector<std::string> generate_default_feature_names(std::size_t count) {
    std::vector<std::string> names;
    names.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        names.emplace_back("feature_" + std::to_string(i));
    }
    return names;
}

std::vector<std::size_t> make_normalization_columns(
    const NumericDataset& dataset,
    const NormalizationOptions& options,
    const char* func_name
) {
    std::size_t feature_count = 0;
    if (!dataset.features.empty()) {
        feature_count = dataset.features.front().size();
    } else if (!dataset.feature_names.empty()) {
        feature_count = dataset.feature_names.size();
    }

    return selected_feature_columns(feature_count, options.feature_column_indices, func_name);
}

void normalize_selected_columns_in_place(
    utils::Matrix& features,
    const utils::ColumnStats& stats,
    const std::vector<std::size_t>& cols,
    const NormalizationOptions& options
) {
    if (features.empty() || cols.empty()) {
        return;
    }

    for (auto& row : features) {
        for (std::size_t c : cols) {
            const auto& s = stats[c];

            if (options.method == utils::NormalizationMethod::MinMax) {
                const double in_range = s.max - s.min;
                if (std::abs(in_range) <= kDefaultEps) {
                    row[c] = options.minmax_out_min;
                } else {
                    const double out_range = options.minmax_out_max - options.minmax_out_min;
                    row[c] =
                        ((row[c] - s.min) / in_range) * out_range + options.minmax_out_min;
                }
            } else { // ZScore
                const double denom = std::max(s.stddev, options.eps);
                row[c] = (row[c] - s.mean) / denom;
            }
        }
    }
}

} // namespace

// ------------------------------------------------------------
// CSV loading / saving
// ------------------------------------------------------------
TabularData load_csv_data(std::string_view csv_path, const CsvLoadOptions& options) {
    if (csv_path.empty()) {
        throw std::invalid_argument("load_csv_data: csv_path is empty");
    }

    if (!utils::file_exists(csv_path)) {
        throw std::runtime_error("load_csv_data: file not found: " + std::string(csv_path));
    }

    std::ifstream ifs{std::string(csv_path)};
    if (!ifs) {
        throw std::runtime_error("load_csv_data: failed to open file: " + std::string(csv_path));
    }

    TabularData table;
    std::string line;
    bool header_consumed = false;

    std::size_t line_no = 0;
    while (std::getline(ifs, line)) {
        ++line_no;

        if (!line.empty() && line.back() == '\r') {
            line.pop_back(); // handle CRLF
        }

        if (options.skip_empty_lines) {
            const std::string trimmed_line = utils::trim(line);
            if (trimmed_line.empty()) {
                continue;
            }
        }

        std::vector<std::string> row;
        try {
            row = utils::parse_csv_row(line, options.delimiter, options.trim_fields);
        } catch (const std::exception& e) {
            throw std::runtime_error(
                "load_csv_data: CSV parse error at line " + std::to_string(line_no) + ": " + e.what()
            );
        }

        if (options.has_header && !header_consumed) {
            table.column_names = std::move(row);
            header_consumed = true;
            continue;
        }

        table.rows.emplace_back(std::move(row));
    }

    return table;
}

void save_cleaned_data(
    const TabularData& table,
    std::string_view output_csv_path,
    const CsvSaveOptions& options
) {
    if (output_csv_path.empty()) {
        throw std::invalid_argument("save_cleaned_data(TabularData): output path is empty");
    }

    const std::filesystem::path out_path{std::string(output_csv_path)};
    if (out_path.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(out_path.parent_path(), ec);
        if (ec) {
            throw std::runtime_error(
                "save_cleaned_data(TabularData): failed to create output directory: " + ec.message()
            );
        }
    }

    std::ofstream ofs(out_path);
    if (!ofs) {
        throw std::runtime_error(
            "save_cleaned_data(TabularData): failed to open output file: " + out_path.string()
        );
    }

    if (options.write_header && !table.column_names.empty()) {
        write_csv_row(ofs, table.column_names, options.delimiter);
    }

    for (const auto& row : table.rows) {
        write_csv_row(ofs, row, options.delimiter);
    }

    if (!ofs.good()) {
        throw std::runtime_error(
            "save_cleaned_data(TabularData): failed while writing file: " + out_path.string()
        );
    }
}

void save_cleaned_data(
    const NumericDataset& dataset,
    std::string_view output_csv_path,
    const CsvSaveOptions& options
) {
    validate_numeric_dataset_shapes(dataset, "save_cleaned_data(NumericDataset)");

    if (output_csv_path.empty()) {
        throw std::invalid_argument("save_cleaned_data(NumericDataset): output path is empty");
    }

    const std::filesystem::path out_path{std::string(output_csv_path)};
    if (out_path.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(out_path.parent_path(), ec);
        if (ec) {
            throw std::runtime_error(
                "save_cleaned_data(NumericDataset): failed to create output directory: " + ec.message()
            );
        }
    }

    std::ofstream ofs(out_path);
    if (!ofs) {
        throw std::runtime_error(
            "save_cleaned_data(NumericDataset): failed to open output file: " + out_path.string()
        );
    }

    const std::size_t rows = dataset.features.size();
    const std::size_t cols = rows > 0 ? dataset.features.front().size()
                                      : dataset.feature_names.size();
    const bool has_target = !dataset.target.empty();

    // Header
    if (options.write_header) {
        std::vector<std::string> header =
            !dataset.feature_names.empty() ? dataset.feature_names : generate_default_feature_names(cols);

        if (has_target) {
            header.push_back(dataset.target_name.empty() ? "target" : dataset.target_name);
        }

        write_csv_row(ofs, header, options.delimiter);
    }

    ofs << std::setprecision(17);
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < dataset.features[r].size(); ++c) {
            if (c > 0) {
                ofs << options.delimiter;
            }
            ofs << dataset.features[r][c];
        }

        if (has_target) {
            if (!dataset.features[r].empty()) {
                ofs << options.delimiter;
            }
            ofs << dataset.target[r];
        }

        ofs << '\n';
    }

    if (!ofs.good()) {
        throw std::runtime_error(
            "save_cleaned_data(NumericDataset): failed while writing file: " + out_path.string()
        );
    }
}

// ------------------------------------------------------------
// Missing values / row cleaning
// ------------------------------------------------------------
std::size_t handle_missing_values(TabularData& table, const MissingValueOptions& options) {
    const std::size_t total_columns = infer_column_count(table);
    if (total_columns == 0 || table.rows.empty()) {
        return 0;
    }

    const auto selected_cols =
        make_selected_columns(total_columns, options.column_indices, "handle_missing_values");

    if (selected_cols.empty()) {
        return 0;
    }

    auto is_missing_cell = [&](const std::string& cell) {
        return is_missing_token(cell, options.missing_tokens);
    };

    if (options.strategy == MissingValueStrategy::DropRow) {
        const std::size_t before = table.rows.size();

        auto new_end = std::remove_if(table.rows.begin(), table.rows.end(),
            [&](const std::vector<std::string>& row) {
                if (!row_has_required_indices(row, selected_cols)) {
                    return true; // malformed row treated as invalid for selected columns
                }
                for (std::size_t c : selected_cols) {
                    if (is_missing_cell(row[c])) {
                        return true;
                    }
                }
                return false;
            });

        table.rows.erase(new_end, table.rows.end());
        return before - table.rows.size();
    }

    // Fill strategies require rows to contain selected columns.
    for (std::size_t r = 0; r < table.rows.size(); ++r) {
        if (!row_has_required_indices(table.rows[r], selected_cols)) {
            throw std::invalid_argument(
                "handle_missing_values: row " + std::to_string(r) +
                " has fewer columns than expected for fill strategy"
            );
        }
    }

    if (options.strategy == MissingValueStrategy::FillWithValue ||
        options.strategy == MissingValueStrategy::FillWithZero) {
        const std::string replacement =
            (options.strategy == MissingValueStrategy::FillWithZero) ? std::string("0")
                                                                     : options.fill_value;

        std::size_t modified = 0;
        for (auto& row : table.rows) {
            for (std::size_t c : selected_cols) {
                if (is_missing_cell(row[c])) {
                    row[c] = replacement;
                    ++modified;
                }
            }
        }
        return modified;
    }

    // Mean / Median imputation (numeric columns)
    // Compute one replacement per selected column.
    std::vector<std::optional<double>> replacements(total_columns, std::nullopt);

    for (std::size_t c : selected_cols) {
        std::vector<double> observed;
        observed.reserve(table.rows.size());

        for (const auto& row : table.rows) {
            const std::string& cell = row[c];
            if (is_missing_cell(cell)) {
                continue;
            }
            double v = 0.0;
            if (parse_double_strict(cell, v)) {
                observed.push_back(v);
            }
        }

        double fill = 0.0;
        if (!observed.empty()) {
            fill = (options.strategy == MissingValueStrategy::FillWithMean)
                       ? compute_mean(observed)
                       : compute_median(std::move(observed));
        }
        replacements[c] = fill;
    }

    std::size_t modified = 0;
    for (auto& row : table.rows) {
        for (std::size_t c : selected_cols) {
            if (is_missing_cell(row[c])) {
                row[c] = double_to_string(*replacements[c]);
                ++modified;
            }
        }
    }

    return modified;
}

std::size_t remove_invalid_rows(TabularData& table, const RowValidationOptions& options) {
    if (table.rows.empty()) {
        return 0;
    }

    const std::size_t expected_columns = infer_column_count(table);

    // Validate requested indices against known/expected column count when available.
    if (expected_columns > 0) {
        for (std::size_t c : options.numeric_columns) {
            if (c >= expected_columns) {
                throw std::out_of_range(
                    "remove_invalid_rows: numeric column index out of range: " + std::to_string(c)
                );
            }
        }
        if (options.target_column_index.has_value() &&
            *options.target_column_index >= expected_columns) {
            throw std::out_of_range(
                "remove_invalid_rows: target column index out of range: " +
                std::to_string(*options.target_column_index)
            );
        }
    }

    const std::size_t before = table.rows.size();

    auto new_end = std::remove_if(table.rows.begin(), table.rows.end(),
        [&](const std::vector<std::string>& row) -> bool {
            if (options.require_consistent_column_count && expected_columns > 0 &&
                row.size() != expected_columns) {
                return true;
            }

            // Numeric column validation
            for (std::size_t c : options.numeric_columns) {
                if (c >= row.size()) {
                    return true;
                }
                double v = 0.0;
                if (!parse_double_strict(row[c], v)) {
                    return true;
                }
            }

            // Target validation (numeric + non-missing)
            if (options.target_column_index.has_value()) {
                const std::size_t tc = *options.target_column_index;
                if (tc >= row.size()) {
                    return true;
                }

                const std::string trimmed = utils::trim(row[tc]);
                if (trimmed.empty()) {
                    return true;
                }

                double target_value = 0.0;
                if (!parse_double_strict(trimmed, target_value)) {
                    return true;
                }
            }

            return false;
        });

    table.rows.erase(new_end, table.rows.end());
    return before - table.rows.size();
}

// ------------------------------------------------------------
// Conversion to numeric dataset
// ------------------------------------------------------------
NumericDataset to_numeric_dataset(
    const TabularData& table,
    const NumericConversionOptions& options
) {
    NumericDataset out{};

    if (table.rows.empty()) {
        // Build empty dataset with inferred names if possible.
        const std::size_t total_cols = infer_column_count(table);

        std::vector<std::size_t> feature_cols;
        if (!options.feature_column_indices.empty()) {
            feature_cols = make_selected_columns(total_cols, options.feature_column_indices, "to_numeric_dataset");
        } else {
            for (std::size_t c = 0; c < total_cols; ++c) {
                if (options.target_column_index.has_value() && c == *options.target_column_index) {
                    continue;
                }
                feature_cols.push_back(c);
            }
        }

        out.feature_names.reserve(feature_cols.size());
        for (std::size_t c : feature_cols) {
            if (!table.column_names.empty()) {
                out.feature_names.push_back(table.column_names[c]);
            } else {
                out.feature_names.push_back("feature_" + std::to_string(c));
            }
        }

        if (options.target_column_index.has_value()) {
            out.target_name = !table.column_names.empty()
                                  ? table.column_names[*options.target_column_index]
                                  : "target";
        }
        return out;
    }

    const std::size_t total_cols = infer_column_count(table);
    if (total_cols == 0) {
        throw std::invalid_argument("to_numeric_dataset: input table has no columns");
    }

    if (options.target_column_index.has_value() && *options.target_column_index >= total_cols) {
        throw std::out_of_range("to_numeric_dataset: target_column_index out of range");
    }

    std::vector<std::size_t> feature_cols;
    if (!options.feature_column_indices.empty()) {
        feature_cols = make_selected_columns(total_cols, options.feature_column_indices, "to_numeric_dataset");
    } else {
        feature_cols.reserve(total_cols - (options.target_column_index.has_value() ? 1u : 0u));
        for (std::size_t c = 0; c < total_cols; ++c) {
            if (options.target_column_index.has_value() && c == *options.target_column_index) {
                continue;
            }
            feature_cols.push_back(c);
        }
    }

    if (feature_cols.empty()) {
        throw std::invalid_argument("to_numeric_dataset: no feature columns selected");
    }

    if (options.target_column_index.has_value()) {
        for (std::size_t c : feature_cols) {
            if (c == *options.target_column_index) {
                throw std::invalid_argument(
                    "to_numeric_dataset: target column cannot also be a feature column"
                );
            }
        }
    }

    // Names
    out.feature_names.reserve(feature_cols.size());
    for (std::size_t c : feature_cols) {
        if (!table.column_names.empty()) {
            out.feature_names.push_back(table.column_names[c]);
        } else {
            out.feature_names.push_back("feature_" + std::to_string(c));
        }
    }
    if (options.target_column_index.has_value()) {
        out.target_name = !table.column_names.empty()
                              ? table.column_names[*options.target_column_index]
                              : "target";
    }

    out.features.reserve(table.rows.size());
    if (options.target_column_index.has_value()) {
        out.target.reserve(table.rows.size());
    }

    for (std::size_t r = 0; r < table.rows.size(); ++r) {
        const auto& row = table.rows[r];

        // Row width check
        if (row.size() != total_cols) {
            if (options.drop_rows_with_parse_errors) {
                continue;
            }
            throw std::invalid_argument(
                "to_numeric_dataset: inconsistent column count at row " + std::to_string(r)
            );
        }

        std::vector<double> feature_row;
        feature_row.reserve(feature_cols.size());

        bool row_ok = true;

        for (std::size_t c : feature_cols) {
            double value = 0.0;
            if (!parse_double_strict(row[c], value)) {
                row_ok = false;
                break;
            }
            feature_row.push_back(value);
        }

        double target_value = 0.0;
        if (row_ok && options.target_column_index.has_value()) {
            if (!parse_double_strict(row[*options.target_column_index], target_value)) {
                row_ok = false;
            }
        }

        if (!row_ok) {
            if (options.drop_rows_with_parse_errors) {
                continue;
            }
            throw std::invalid_argument(
                "to_numeric_dataset: non-numeric value encountered at row " + std::to_string(r)
            );
        }

        out.features.emplace_back(std::move(feature_row));
        if (options.target_column_index.has_value()) {
            out.target.push_back(target_value);
        }
    }

    return out;
}

// ------------------------------------------------------------
// Normalization
// ------------------------------------------------------------
NormalizationResult normalize_columns(
    NumericDataset& dataset,
    const NormalizationOptions& options
) {
    validate_numeric_dataset_shapes(dataset, "normalize_columns");

    if (dataset.features.empty()) {
        throw std::invalid_argument("normalize_columns: dataset.features is empty");
    }

    if (options.method == utils::NormalizationMethod::MinMax &&
        !(options.minmax_out_max > options.minmax_out_min)) {
        throw std::invalid_argument("normalize_columns: invalid min-max output range");
    }
    if (options.method == utils::NormalizationMethod::ZScore && options.eps <= 0.0) {
        throw std::invalid_argument("normalize_columns: eps must be > 0 for z-score");
    }

    // Fit stats using the shared utility, then (for z-score) enforce a POPULATION
    // standard deviation in the fitted stats that are actually used for normalization.
    //
    // Why this matters:
    // - Different implementations/tests may use sample vs population stddev.
    // - The preprocessing tests expect the normalized column stddev to be ~1 under
    //   population-stddev verification.
    //
    // By explicitly computing population stddev here, normalize_columns() becomes
    // stable regardless of how utils::fit_column_stats computes stddev internally.
    auto stats = utils::fit_column_stats(dataset.features);
    const auto cols = make_normalization_columns(dataset, options, "normalize_columns");

    if (options.method == utils::NormalizationMethod::ZScore && !cols.empty()) {
        const std::size_t n = dataset.features.size();

        if (n == 0) {
            throw std::invalid_argument("normalize_columns: dataset.features is empty");
        }

        for (std::size_t c : cols) {
            if (c >= stats.size()) {
                throw std::out_of_range(
                    "normalize_columns: fitted stats column index out of range: " + std::to_string(c)
                );
            }

            const double mean = stats[c].mean;
            double sq_sum = 0.0;

            for (const auto& row : dataset.features) {
                if (c >= row.size()) {
                    throw std::invalid_argument(
                        "normalize_columns: features matrix is not rectangular during z-score fit"
                    );
                }
                const double d = row[c] - mean;
                sq_sum += d * d;
            }

            // Population variance/stddev (divide by n, not n-1)
            const double var_pop = sq_sum / static_cast<double>(n);
            stats[c].stddev = std::sqrt(std::max(0.0, var_pop));
        }
    }

    normalize_selected_columns_in_place(dataset.features, stats, cols, options);

    return NormalizationResult{stats, cols};
}

void apply_normalization(
    NumericDataset& dataset,
    const utils::ColumnStats& fitted_feature_stats,
    const NormalizationOptions& options
) {
    validate_numeric_dataset_shapes(dataset, "apply_normalization");

    if (dataset.features.empty()) {
        return; // no-op for empty inference batch
    }

    if (fitted_feature_stats.empty()) {
        throw std::invalid_argument("apply_normalization: fitted_feature_stats is empty");
    }

    const std::size_t cols_in_data = dataset.features.front().size();
    if (fitted_feature_stats.size() != cols_in_data) {
        throw std::invalid_argument(
            "apply_normalization: stats size mismatch with feature column count"
        );
    }

    if (options.method == utils::NormalizationMethod::MinMax &&
        !(options.minmax_out_max > options.minmax_out_min)) {
        throw std::invalid_argument("apply_normalization: invalid min-max output range");
    }
    if (options.method == utils::NormalizationMethod::ZScore && options.eps <= 0.0) {
        throw std::invalid_argument("apply_normalization: eps must be > 0 for z-score");
    }

    const auto cols = make_normalization_columns(dataset, options, "apply_normalization");
    normalize_selected_columns_in_place(dataset.features, fitted_feature_stats, cols, options);
}

// ------------------------------------------------------------
// Train / test split
// ------------------------------------------------------------
TrainTestSplit split_train_test(
    const NumericDataset& dataset,
    const TrainTestSplitOptions& options
) {
    validate_numeric_dataset_shapes(dataset, "split_train_test");

    if (!(options.test_ratio > 0.0 && options.test_ratio < 1.0)) {
        throw std::invalid_argument("split_train_test: test_ratio must be in (0, 1)");
    }

    const std::size_t n = dataset.features.size();
    if (n == 0) {
        throw std::invalid_argument("split_train_test: dataset is empty");
    }

    std::vector<std::size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    if (options.shuffle) {
        auto rng = utils::make_rng(options.random_seed);
        std::shuffle(indices.begin(), indices.end(), rng);
    }

    std::size_t test_count = 0;
    if (n > 1) {
        test_count = static_cast<std::size_t>(std::llround(static_cast<double>(n) * options.test_ratio));
        test_count = std::max<std::size_t>(1, test_count);
        test_count = std::min<std::size_t>(n - 1, test_count);
    }
    const std::size_t train_count = n - test_count;

    TrainTestSplit split{};
    split.train.feature_names = dataset.feature_names;
    split.test.feature_names = dataset.feature_names;
    split.train.target_name = dataset.target_name;
    split.test.target_name = dataset.target_name;

    split.train.features.reserve(train_count);
    split.test.features.reserve(test_count);

    const bool has_target = !dataset.target.empty();
    if (has_target) {
        split.train.target.reserve(train_count);
        split.test.target.reserve(test_count);
    }

    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t src_idx = indices[i];
        const bool goes_to_test = (i < test_count);

        if (goes_to_test) {
            split.test.features.push_back(dataset.features[src_idx]);
            if (has_target) {
                split.test.target.push_back(dataset.target[src_idx]);
            }
        } else {
            split.train.features.push_back(dataset.features[src_idx]);
            if (has_target) {
                split.train.target.push_back(dataset.target[src_idx]);
            }
        }
    }

    return split;
}

} // namespace mof::preprocessing