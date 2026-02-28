 #include "feature_engineering.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mof::feature_engineering {
namespace {

constexpr double kNearConstantEps = 1e-12;

// ------------------------------------------------------------
// Internal validation helpers
// ------------------------------------------------------------
void validate_numeric_dataset(const preprocessing::NumericDataset& dataset, const char* func_name) {
    const std::size_t rows = dataset.features.size();
    std::size_t cols = 0;

    if (!dataset.features.empty()) {
        cols = dataset.features.front().size();
        for (std::size_t r = 1; r < rows; ++r) {
            if (dataset.features[r].size() != cols) {
                throw std::invalid_argument(
                    std::string(func_name) + ": features matrix is not rectangular at row " +
                    std::to_string(r)
                );
            }
        }
    } else if (!dataset.feature_names.empty()) {
        cols = dataset.feature_names.size();
    }

    if (!dataset.feature_names.empty() && dataset.feature_names.size() != cols) {
        throw std::invalid_argument(
            std::string(func_name) + ": feature_names count mismatch with feature column count"
        );
    }

    if (!dataset.target.empty() && dataset.target.size() != rows) {
        throw std::invalid_argument(
            std::string(func_name) + ": target size mismatch with feature row count"
        );
    }
}

std::vector<std::string> default_feature_names(std::size_t count) {
    std::vector<std::string> names;
    names.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        names.emplace_back("feature_" + std::to_string(i));
    }
    return names;
}

const std::vector<std::string>& get_feature_names_or_throw(
    const preprocessing::NumericDataset& dataset,
    std::vector<std::string>& tmp_names
) {
    if (!dataset.feature_names.empty()) {
        return dataset.feature_names;
    }

    std::size_t cols = 0;
    if (!dataset.features.empty()) {
        cols = dataset.features.front().size();
    }
    tmp_names = default_feature_names(cols);
    return tmp_names;
}

std::vector<std::size_t> make_all_indices(std::size_t n) {
    std::vector<std::size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    return idx;
}

std::vector<std::size_t> validate_unique_indices(
    std::size_t total_cols,
    const std::vector<std::size_t>& indices,
    const char* func_name
) {
    std::vector<bool> seen(total_cols, false);
    std::vector<std::size_t> out;
    out.reserve(indices.size());

    for (std::size_t c : indices) {
        if (c >= total_cols) {
            throw std::out_of_range(
                std::string(func_name) + ": column index out of range: " + std::to_string(c)
            );
        }
        if (!seen[c]) {
            seen[c] = true;
            out.push_back(c);
        }
    }
    return out;
}

std::unordered_map<std::string, std::size_t> build_name_to_index_map(const std::vector<std::string>& names) {
    std::unordered_map<std::string, std::size_t> map;
    map.reserve(names.size());
    for (std::size_t i = 0; i < names.size(); ++i) {
        // keep first occurrence if duplicates exist
        map.emplace(names[i], i);
    }
    return map;
}

std::vector<std::size_t> resolve_names_to_indices(
    const std::vector<std::string>& feature_names,
    const std::vector<std::string>& names,
    bool require_all_names,
    const char* func_name
) {
    const auto map = build_name_to_index_map(feature_names);
    std::vector<std::size_t> out;
    out.reserve(names.size());

    std::vector<bool> seen(feature_names.size(), false);

    for (const auto& name : names) {
        auto it = map.find(name);
        if (it == map.end()) {
            if (require_all_names) {
                throw std::invalid_argument(
                    std::string(func_name) + ": feature name not found: " + name
                );
            }
            continue;
        }
        if (!seen[it->second]) {
            seen[it->second] = true;
            out.push_back(it->second);
        }
    }
    return out;
}

std::vector<std::size_t> apply_exclusions(
    std::size_t total_cols,
    const std::vector<std::size_t>& base,
    const std::vector<std::size_t>& excluded
) {
    if (excluded.empty()) {
        return base;
    }

    std::vector<bool> drop(total_cols, false);
    for (std::size_t c : excluded) {
        if (c < total_cols) {
            drop[c] = true;
        }
    }

    std::vector<std::size_t> out;
    out.reserve(base.size());
    for (std::size_t c : base) {
        if (!drop[c]) {
            out.push_back(c);
        }
    }
    return out;
}

void append_feature_column(
    preprocessing::NumericDataset& dataset,
    const std::string& name,
    const std::vector<double>& column
) {
    const std::size_t rows = dataset.features.size();
    if (rows != column.size()) {
        throw std::invalid_argument("append_feature_column: column size mismatch");
    }

    if (dataset.features.empty()) {
        // No rows to append into; just record name if truly empty dataset.
        dataset.feature_names.push_back(name);
        return;
    }

    for (std::size_t r = 0; r < rows; ++r) {
        dataset.features[r].push_back(column[r]);
    }
    dataset.feature_names.push_back(name);
}

bool should_skip_invalid_unary_value(
    DerivedFeatureKind kind,
    double x,
    double eps
) {
    switch (kind) {
        case DerivedFeatureKind::Log1p: return !(x > -1.0);
        case DerivedFeatureKind::Sqrt:  return x < 0.0;
        case DerivedFeatureKind::Inverse: return std::abs(x) <= eps;
        default: return false;
    }
}

double transform_unary(DerivedFeatureKind kind, double x, double eps) {
    switch (kind) {
        case DerivedFeatureKind::PolynomialDegree2: return x * x;
        case DerivedFeatureKind::Log1p:             return std::log1p(x);
        case DerivedFeatureKind::Sqrt:              return std::sqrt(x);
        case DerivedFeatureKind::Inverse:           return 1.0 / x;
        default:
            throw std::invalid_argument("transform_unary: unsupported unary feature kind");
    }
}

bool should_skip_invalid_pair_value(
    DerivedFeatureKind kind,
    double xi,
    double xj,
    double eps
) {
    if (kind == DerivedFeatureKind::PairwiseRatio) {
        return std::abs(xj) <= eps;
    }
    (void)xi;
    return false;
}

double transform_pair(DerivedFeatureKind kind, double xi, double xj) {
    switch (kind) {
        case DerivedFeatureKind::PairwiseProduct:    return xi * xj;
        case DerivedFeatureKind::PairwiseRatio:      return xi / xj;
        case DerivedFeatureKind::PairwiseDifference: return xi - xj;
        case DerivedFeatureKind::PairwiseSum:        return xi + xj;
        default:
            throw std::invalid_argument("transform_pair: unsupported pairwise feature kind");
    }
}

std::string unary_suffix(DerivedFeatureKind kind) {
    switch (kind) {
        case DerivedFeatureKind::PolynomialDegree2: return "sq";
        case DerivedFeatureKind::Log1p:             return "log1p";
        case DerivedFeatureKind::Sqrt:              return "sqrt";
        case DerivedFeatureKind::Inverse:           return "inv";
        default: return "u";
    }
}

std::string pairwise_suffix(DerivedFeatureKind kind) {
    switch (kind) {
        case DerivedFeatureKind::PairwiseProduct:    return "mul";
        case DerivedFeatureKind::PairwiseRatio:      return "div";
        case DerivedFeatureKind::PairwiseDifference: return "sub";
        case DerivedFeatureKind::PairwiseSum:        return "add";
        default: return "p";
    }
}

// ------------------------------------------------------------
// Internal: constant / near-constant feature removal
// (header has no public declaration yet, so keep internal)
// ------------------------------------------------------------
struct ConstantFilterResult {
    preprocessing::NumericDataset dataset;
    std::size_t removed_count = 0;
    std::vector<std::string> removed_names;
};

ConstantFilterResult remove_constant_or_near_constant_features_internal(
    const preprocessing::NumericDataset& input,
    double eps = kNearConstantEps
) {
    validate_numeric_dataset(input, "remove_constant_or_near_constant_features_internal");

    ConstantFilterResult result;
    result.dataset = input; // copy metadata + data

    if (input.features.empty()) {
        return result;
    }

    const std::size_t rows = input.features.size();
    const std::size_t cols = input.features.front().size();

    if (cols == 0) {
        return result;
    }

    std::vector<std::string> temp_names;
    const auto& names = get_feature_names_or_throw(input, temp_names);

    std::vector<bool> keep(cols, true);

    for (std::size_t c = 0; c < cols; ++c) {
        double min_v = std::numeric_limits<double>::infinity();
        double max_v = -std::numeric_limits<double>::infinity();
        double sum = 0.0;
        double sum_sq = 0.0;

        for (std::size_t r = 0; r < rows; ++r) {
            const double v = input.features[r][c];
            min_v = std::min(min_v, v);
            max_v = std::max(max_v, v);
            sum += v;
            sum_sq += v * v;
        }

        const double mean = sum / static_cast<double>(rows);
        double var = (sum_sq / static_cast<double>(rows)) - (mean * mean);
        if (var < 0.0 && var > -1e-15) {
            var = 0.0;
        }
        const double stddev = (var > 0.0) ? std::sqrt(var) : 0.0;
        const double range = max_v - min_v;

        if (std::abs(range) <= eps || std::abs(stddev) <= eps) {
            keep[c] = false;
            result.removed_names.push_back(c < names.size() ? names[c] : ("feature_" + std::to_string(c)));
        }
    }

    std::size_t kept_cols = 0;
    for (bool k : keep) {
        if (k) ++kept_cols;
    }

    result.removed_count = cols - kept_cols;
    if (result.removed_count == 0) {
        return result;
    }

    result.dataset.feature_names.clear();
    result.dataset.feature_names.reserve(kept_cols);
    for (std::size_t c = 0; c < cols; ++c) {
        if (keep[c]) {
            result.dataset.feature_names.push_back(c < names.size() ? names[c] : ("feature_" + std::to_string(c)));
        }
    }

    for (auto& row : result.dataset.features) {
        std::vector<double> new_row;
        new_row.reserve(kept_cols);
        for (std::size_t c = 0; c < cols; ++c) {
            if (keep[c]) {
                new_row.push_back(row[c]);
            }
        }
        row = std::move(new_row);
    }

    return result;
}

// ------------------------------------------------------------
// Internal: basic feature statistics (header has no public API yet)
// ------------------------------------------------------------
struct FeatureBasicStats {
    std::string name;
    double min = 0.0;
    double max = 0.0;
    double mean = 0.0;
    double stddev = 0.0;
};

std::vector<FeatureBasicStats> compute_basic_feature_statistics_internal(
    const preprocessing::NumericDataset& dataset
) {
    validate_numeric_dataset(dataset, "compute_basic_feature_statistics_internal");

    std::vector<FeatureBasicStats> out;
    if (dataset.features.empty()) {
        return out;
    }

    std::vector<std::string> tmp_names;
    const auto& names = get_feature_names_or_throw(dataset, tmp_names);
    const auto stats = utils::fit_column_stats(dataset.features);

    out.reserve(stats.size());
    for (std::size_t c = 0; c < stats.size(); ++c) {
        FeatureBasicStats s;
        s.name = (c < names.size()) ? names[c] : ("feature_" + std::to_string(c));
        s.min = stats[c].min;
        s.max = stats[c].max;
        s.mean = stats[c].mean;
        s.stddev = stats[c].stddev;
        out.push_back(std::move(s));
    }

    return out;
}

void log_basic_feature_statistics_preview(const preprocessing::NumericDataset& dataset) {
    const auto stats = compute_basic_feature_statistics_internal(dataset);
    if (stats.empty()) {
        utils::log_info("Feature stats: dataset is empty.");
        return;
    }

    std::ostringstream oss;
    oss << "Feature stats: " << stats.size() << " columns, preview -> ";
    const std::size_t preview = std::min<std::size_t>(stats.size(), 5);
    for (std::size_t i = 0; i < preview; ++i) {
        if (i > 0) {
            oss << " | ";
        }
        oss << stats[i].name
            << " [min=" << stats[i].min
            << ", max=" << stats[i].max
            << ", mean=" << stats[i].mean
            << ", std=" << stats[i].stddev << "]";
    }
    if (stats.size() > preview) {
        oss << " ...";
    }
    utils::log_info(oss.str());
}

// ------------------------------------------------------------
// CSV export helpers
// ------------------------------------------------------------
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
            out.push_back('"');
        }
        out.push_back(ch);
    }
    out.push_back('"');
    return out;
}

void write_string_row(std::ostream& os, const std::vector<std::string>& row, char delimiter) {
    for (std::size_t i = 0; i < row.size(); ++i) {
        if (i > 0) {
            os << delimiter;
        }
        os << escape_csv_field(row[i], delimiter);
    }
    os << '\n';
}

void ensure_parent_dir_exists(const std::filesystem::path& path, const char* func_name) {
    if (!path.has_parent_path()) {
        return;
    }
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    if (ec) {
        throw std::runtime_error(std::string(func_name) + ": failed to create output directory: " + ec.message());
    }
}

void export_numeric_dataset_csv_internal(
    const preprocessing::NumericDataset& dataset,
    std::string_view output_csv_path,
    const FeatureExportOptions& options
) {
    validate_numeric_dataset(dataset, "export_numeric_dataset_csv_internal");

    if (output_csv_path.empty()) {
        throw std::invalid_argument("export_numeric_dataset_csv_internal: output path is empty");
    }

    const std::filesystem::path out_path{std::string(output_csv_path)};
    ensure_parent_dir_exists(out_path, "export_numeric_dataset_csv_internal");

    std::ofstream ofs(out_path);
    if (!ofs) {
        throw std::runtime_error("export_numeric_dataset_csv_internal: failed to open " + out_path.string());
    }

    const std::size_t rows = dataset.features.size();
    const std::size_t cols = (!dataset.features.empty()) ? dataset.features.front().size()
                                                         : dataset.feature_names.size();

    std::vector<std::string> feature_names = dataset.feature_names;
    if (feature_names.empty()) {
        feature_names = default_feature_names(cols);
    }

    const bool write_target = options.include_target && !dataset.target.empty();

    if (options.write_header) {
        std::vector<std::string> header;
        header.reserve(cols + (options.include_row_index ? 1u : 0u) + (write_target ? 1u : 0u));

        if (options.include_row_index) {
            header.push_back(options.row_index_column_name);
        }
        for (const auto& n : feature_names) {
            header.push_back(n);
        }
        if (write_target) {
            header.push_back(dataset.target_name.empty() ? "target" : dataset.target_name);
        }

        write_string_row(ofs, header, options.delimiter);
    }

    ofs << std::setprecision(17);
    for (std::size_t r = 0; r < rows; ++r) {
        bool first = true;

        auto emit_delim = [&]() {
            if (!first) {
                ofs << options.delimiter;
            }
            first = false;
        };

        if (options.include_row_index) {
            emit_delim();
            ofs << r;
        }

        for (std::size_t c = 0; c < cols; ++c) {
            emit_delim();
            ofs << dataset.features[r][c];
        }

        if (write_target) {
            emit_delim();
            ofs << dataset.target[r];
        }

        ofs << '\n';
    }

    if (!ofs.good()) {
        throw std::runtime_error("export_numeric_dataset_csv_internal: write failed for " + out_path.string());
    }

    std::ostringstream msg;
    msg << "Exported feature matrix to " << out_path.string()
        << " (rows=" << rows << ", features=" << cols
        << (write_target ? ", target=yes" : ", target=no") << ")";
    utils::log_info(msg.str());

    log_basic_feature_statistics_preview(dataset);
}

std::vector<std::size_t> normalize_requested_columns_for_feature_space(
    const preprocessing::NumericDataset& dataset,
    const std::vector<std::size_t>& requested,
    const char* func_name
) {
    std::size_t total_cols = 0;
    if (!dataset.features.empty()) {
        total_cols = dataset.features.front().size();
    } else {
        total_cols = dataset.feature_names.size();
    }

    if (requested.empty()) {
        return make_all_indices(total_cols);
    }
    return validate_unique_indices(total_cols, requested, func_name);
}

} // namespace

// ============================================================
// Descriptor selection
// ============================================================
std::vector<std::size_t> select_descriptor_indices(
    const preprocessing::NumericDataset& dataset,
    const DescriptorSelectionOptions& options
) {
    validate_numeric_dataset(dataset, "select_descriptor_indices");

    std::size_t total_cols = 0;
    if (!dataset.features.empty()) {
        total_cols = dataset.features.front().size();
    } else {
        total_cols = dataset.feature_names.size();
    }

    std::vector<std::string> tmp_names;
    const auto& feature_names = get_feature_names_or_throw(dataset, tmp_names);

    std::vector<std::size_t> selected;

    switch (options.mode) {
        case DescriptorSelectionMode::All: {
            selected = make_all_indices(total_cols);
            break;
        }

        case DescriptorSelectionMode::ByIndex: {
            selected = validate_unique_indices(total_cols, options.include_indices, "select_descriptor_indices");
            break;
        }

        case DescriptorSelectionMode::ByName: {
            selected = resolve_names_to_indices(
                feature_names, options.include_names, options.require_all_names, "select_descriptor_indices");
            break;
        }

        case DescriptorSelectionMode::ExcludeByIndex: {
            selected = make_all_indices(total_cols);
            const auto excluded = validate_unique_indices(
                total_cols, options.exclude_indices, "select_descriptor_indices");
            selected = apply_exclusions(total_cols, selected, excluded);
            break;
        }

        case DescriptorSelectionMode::ExcludeByName: {
            selected = make_all_indices(total_cols);
            const auto excluded = resolve_names_to_indices(
                feature_names, options.exclude_names, options.require_all_names, "select_descriptor_indices");
            selected = apply_exclusions(total_cols, selected, excluded);
            break;
        }

        default:
            throw std::invalid_argument("select_descriptor_indices: unsupported selection mode");
    }

    // Optional extra exclusions even for inclusion modes
    if (!options.exclude_indices.empty()) {
        const auto excluded = validate_unique_indices(total_cols, options.exclude_indices, "select_descriptor_indices");
        selected = apply_exclusions(total_cols, selected, excluded);
    }
    if (!options.exclude_names.empty()) {
        const auto excluded = resolve_names_to_indices(
            feature_names, options.exclude_names, options.require_all_names, "select_descriptor_indices");
        selected = apply_exclusions(total_cols, selected, excluded);
    }

    // If preserve_order is false for include modes, sort by index of request order isn't necessary
    // because include_indices/include_names order is already preserved. For exclusion modes, original
    // order is always preserved (preferred for model compatibility).
    (void)options.preserve_order;

    return selected;
}

preprocessing::NumericDataset select_descriptors(
    const preprocessing::NumericDataset& dataset,
    const DescriptorSelectionOptions& options
) {
    validate_numeric_dataset(dataset, "select_descriptors");

    preprocessing::NumericDataset out;
    out.target = dataset.target;
    out.target_name = dataset.target_name;

    const auto indices = select_descriptor_indices(dataset, options);

    std::vector<std::string> tmp_names;
    const auto& feature_names = get_feature_names_or_throw(dataset, tmp_names);

    out.feature_names.reserve(indices.size());
    for (std::size_t c : indices) {
        out.feature_names.push_back(feature_names[c]);
    }

    out.features.reserve(dataset.features.size());
    for (const auto& row : dataset.features) {
        std::vector<double> new_row;
        new_row.reserve(indices.size());
        for (std::size_t c : indices) {
            new_row.push_back(row[c]);
        }
        out.features.push_back(std::move(new_row));
    }

    return out;
}

// ============================================================
// Derived feature creation
// ============================================================
preprocessing::NumericDataset create_derived_features(
    const preprocessing::NumericDataset& dataset,
    const DerivedFeatureOptions& options,
    DerivedFeatureSummary* summary
) {
    validate_numeric_dataset(dataset, "create_derived_features");

    preprocessing::NumericDataset out = dataset;

    const std::size_t rows = out.features.size();
    std::size_t original_cols = 0;
    if (!out.features.empty()) {
        original_cols = out.features.front().size();
    } else {
        original_cols = out.feature_names.size();
    }

    if (out.feature_names.empty()) {
        out.feature_names = default_feature_names(original_cols);
    }

    if (rows == 0 || original_cols == 0) {
        if (summary) {
            summary->original_feature_count = original_cols;
            summary->added_feature_count = 0;
            summary->final_feature_count = original_cols;
            summary->added_feature_names.clear();
        }
        return out;
    }

    if (options.eps <= 0.0) {
        throw std::invalid_argument("create_derived_features: eps must be > 0");
    }

    const auto source_cols = normalize_requested_columns_for_feature_space(
        out, options.source_feature_indices, "create_derived_features");

    std::vector<std::string> added_names;
    added_names.reserve(64);

    auto try_add_unary = [&](DerivedFeatureKind kind) {
        for (std::size_t c : source_cols) {
            std::vector<double> col;
            col.reserve(rows);

            bool invalid_column = false;
            for (std::size_t r = 0; r < rows; ++r) {
                const double x = out.features[r][c];
                if (should_skip_invalid_unary_value(kind, x, options.eps)) {
                    invalid_column = true;
                    break;
                }
                col.push_back(transform_unary(kind, x, options.eps));
            }

            if (invalid_column) {
                if (options.skip_invalid_transform_columns) {
                    continue;
                }
                throw std::domain_error("create_derived_features: invalid domain for unary transform");
            }

            const std::string name = out.feature_names[c] + "__" + unary_suffix(kind);
            append_feature_column(out, name, col);
            added_names.push_back(name);
        }
    };

    if (options.add_square)  try_add_unary(DerivedFeatureKind::PolynomialDegree2);
    if (options.add_log1p)   try_add_unary(DerivedFeatureKind::Log1p);
    if (options.add_sqrt)    try_add_unary(DerivedFeatureKind::Sqrt);
    if (options.add_inverse) try_add_unary(DerivedFeatureKind::Inverse);

    std::size_t pairwise_generated = 0;
    const std::size_t pairwise_cap =
        (options.max_pairwise_features == 0) ? std::numeric_limits<std::size_t>::max()
                                             : options.max_pairwise_features;

    auto try_add_pairwise_kind = [&](DerivedFeatureKind kind) {
        for (std::size_t ai = 0; ai < source_cols.size(); ++ai) {
            const std::size_t i = source_cols[ai];

            if (options.pairwise_upper_triangle_only) {
                for (std::size_t aj = ai + 1; aj < source_cols.size(); ++aj) {
                    if (pairwise_generated >= pairwise_cap) {
                        return;
                    }

                    const std::size_t j = source_cols[aj];
                    std::vector<double> col;
                    col.reserve(rows);

                    bool invalid_column = false;
                    for (std::size_t r = 0; r < rows; ++r) {
                        const double xi = out.features[r][i];
                        const double xj = out.features[r][j];

                        if (should_skip_invalid_pair_value(kind, xi, xj, options.eps)) {
                            invalid_column = true;
                            break;
                        }
                        col.push_back(transform_pair(kind, xi, xj));
                    }

                    if (invalid_column) {
                        if (options.skip_invalid_transform_columns) {
                            continue;
                        }
                        throw std::domain_error("create_derived_features: invalid domain for pairwise transform");
                    }

                    const std::string name = out.feature_names[i] + "__" +
                                             pairwise_suffix(kind) + "__" +
                                             out.feature_names[j];
                    append_feature_column(out, name, col);
                    added_names.push_back(name);
                    ++pairwise_generated;
                }
            } else {
                for (std::size_t aj = 0; aj < source_cols.size(); ++aj) {
                    if (aj == ai) {
                        continue;
                    }
                    if (pairwise_generated >= pairwise_cap) {
                        return;
                    }

                    const std::size_t j = source_cols[aj];
                    std::vector<double> col;
                    col.reserve(rows);

                    bool invalid_column = false;
                    for (std::size_t r = 0; r < rows; ++r) {
                        const double xi = out.features[r][i];
                        const double xj = out.features[r][j];

                        if (should_skip_invalid_pair_value(kind, xi, xj, options.eps)) {
                            invalid_column = true;
                            break;
                        }
                        col.push_back(transform_pair(kind, xi, xj));
                    }

                    if (invalid_column) {
                        if (options.skip_invalid_transform_columns) {
                            continue;
                        }
                        throw std::domain_error("create_derived_features: invalid domain for pairwise transform");
                    }

                    const std::string name = out.feature_names[i] + "__" +
                                             pairwise_suffix(kind) + "__" +
                                             out.feature_names[j];
                    append_feature_column(out, name, col);
                    added_names.push_back(name);
                    ++pairwise_generated;
                }
            }
        }
    };

    if (options.add_pairwise_product)    try_add_pairwise_kind(DerivedFeatureKind::PairwiseProduct);
    if (options.add_pairwise_ratio)      try_add_pairwise_kind(DerivedFeatureKind::PairwiseRatio);
    if (options.add_pairwise_difference) try_add_pairwise_kind(DerivedFeatureKind::PairwiseDifference);
    if (options.add_pairwise_sum)        try_add_pairwise_kind(DerivedFeatureKind::PairwiseSum);

    // Remove constant / near-constant features (internal helper)
    const auto filtered = remove_constant_or_near_constant_features_internal(out, std::max(options.eps, kNearConstantEps));
    if (filtered.removed_count > 0) {
        std::ostringstream oss;
        oss << "Feature engineering removed " << filtered.removed_count
            << " constant/near-constant feature(s).";
        utils::log_info(oss.str());
        out = filtered.dataset;
    }

    if (summary) {
        std::size_t final_cols = 0;
        if (!out.features.empty()) {
            final_cols = out.features.front().size();
        } else {
            final_cols = out.feature_names.size();
        }

        summary->original_feature_count = original_cols;
        summary->final_feature_count = final_cols;
        summary->added_feature_count = (final_cols >= original_cols) ? (final_cols - original_cols) : 0;
        summary->added_feature_names = std::move(added_names);
        // Note: some added names may have been removed by constant filtering; keep as creation log.
    }

    return out;
}

// ============================================================
// Feature scaling hooks
// ============================================================
FeatureScalingArtifacts fit_and_apply_feature_scaling(
    preprocessing::NumericDataset& dataset,
    const FeatureScalingOptions& options
) {
    validate_numeric_dataset(dataset, "fit_and_apply_feature_scaling");

    FeatureScalingArtifacts artifacts{};
    artifacts.method = options.method;

    if (options.method == FeatureScalingMethod::None) {
        return artifacts;
    }

    preprocessing::NormalizationOptions p{};
    p.feature_column_indices = options.feature_column_indices;
    p.minmax_out_min = options.min_out;
    p.minmax_out_max = options.max_out;
    p.eps = options.eps;

    switch (options.method) {
        case FeatureScalingMethod::MinMax:
            p.method = utils::NormalizationMethod::MinMax;
            break;
        case FeatureScalingMethod::ZScore:
            p.method = utils::NormalizationMethod::ZScore;
            break;
        case FeatureScalingMethod::None:
        default:
            return artifacts;
    }

    auto result = preprocessing::normalize_columns(dataset, p);
    artifacts.feature_stats = std::move(result.feature_stats);
    artifacts.scaled_columns = std::move(result.normalized_columns);
    return artifacts;
}

void apply_feature_scaling(
    preprocessing::NumericDataset& dataset,
    const FeatureScalingArtifacts& artifacts,
    const FeatureScalingOptions& options
) {
    validate_numeric_dataset(dataset, "apply_feature_scaling");

    FeatureScalingMethod effective_method = options.method;
    if (effective_method == FeatureScalingMethod::None) {
        effective_method = artifacts.method;
    }

    if (effective_method == FeatureScalingMethod::None) {
        return; // no-op
    }

    if (artifacts.feature_stats.empty()) {
        throw std::invalid_argument("apply_feature_scaling: missing fitted feature stats");
    }

    if (artifacts.method != FeatureScalingMethod::None &&
        effective_method != artifacts.method) {
        throw std::invalid_argument("apply_feature_scaling: method mismatch with fitted artifacts");
    }

    preprocessing::NormalizationOptions p{};
    p.feature_column_indices = options.feature_column_indices.empty()
                                   ? artifacts.scaled_columns
                                   : options.feature_column_indices;
    p.minmax_out_min = options.min_out;
    p.minmax_out_max = options.max_out;
    p.eps = options.eps;

    switch (effective_method) {
        case FeatureScalingMethod::MinMax:
            p.method = utils::NormalizationMethod::MinMax;
            break;
        case FeatureScalingMethod::ZScore:
            p.method = utils::NormalizationMethod::ZScore;
            break;
        case FeatureScalingMethod::None:
        default:
            return;
    }

    preprocessing::apply_normalization(dataset, artifacts.feature_stats, p);
}

// ============================================================
// Feature matrix export
// ============================================================
void export_feature_matrix(
    const preprocessing::NumericDataset& dataset,
    std::string_view output_csv_path,
    const FeatureExportOptions& options
) {
    // Typical usage is data/features/<name>.csv, but any path is allowed.
    export_numeric_dataset_csv_internal(dataset, output_csv_path, options);
}

void export_feature_matrix_only(
    const utils::Matrix& features,
    const std::vector<std::string>& feature_names,
    std::string_view output_csv_path,
    const FeatureExportOptions& options
) {
    preprocessing::NumericDataset ds;
    ds.features = features;
    ds.feature_names = feature_names;
    ds.target.clear();
    ds.target_name.clear();

    FeatureExportOptions opt = options;
    opt.include_target = false; // force feature-only export
    export_numeric_dataset_csv_internal(ds, output_csv_path, opt);
}

// ============================================================
// MOF-specific placeholders (future extension)
// ============================================================
preprocessing::NumericDataset build_mof_descriptors_from_cif_placeholder(
    const std::vector<std::string>& cif_paths,
    const MofDescriptorPlaceholders& options
) {
    (void)cif_paths;
    (void)options;
    throw std::logic_error(
        "build_mof_descriptors_from_cif_placeholder: not implemented yet "
        "(planned for MOF-specific CIF descriptor extraction)"
    );
}

preprocessing::NumericDataset add_mof_specific_derived_features_placeholder(
    const preprocessing::NumericDataset& dataset,
    const MofDescriptorPlaceholders& options
) {
    (void)dataset;
    (void)options;
    throw std::logic_error(
        "add_mof_specific_derived_features_placeholder: not implemented yet "
        "(planned for MOF-specific domain feature engineering)"
    );
}

} // namespace mof::feature_engineering