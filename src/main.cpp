 #include <memory> // workaround if include/modeling.h does not include <memory>

#include "feature_engineering.h"
#include "modeling.h"
#include "preprocessing.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

using mof::preprocessing::NumericDataset;

// ============================================================
// CLI
// ============================================================

struct CliOptions {
    std::string input_path;
    std::string target_column;            // column name or zero-based index (as string)
    std::string model_type = "linear";    // linear | rf | svm | nn
    std::string output_dir = "output";

    double test_ratio = 0.2;              // (0,1)
    std::uint32_t seed = 42;

    bool has_header = true;
    char delimiter = ',';

    // Pipeline knobs (simple MVP defaults)
    bool add_square_features = true;
    bool add_pairwise_product = false;
    std::size_t max_pairwise_features = 64;
    bool save_model = true;
};

void print_usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " --input <csv_path> --target <column_name_or_index> [options]\n\n"
        << "Required:\n"
        << "  --input <path>           Path to input CSV file\n"
        << "  --target <name|index>    Target column (header name or zero-based index)\n\n"
        << "Optional:\n"
        << "  --model <type>           linear | rf | svm | nn   (default: linear)\n"
        << "  --outdir <dir>           Output directory root    (default: output)\n"
        << "  --test-ratio <float>     Test split ratio (0,1)   (default: 0.2)\n"
        << "  --seed <int>             Random seed              (default: 42)\n"
        << "  --delimiter <char>       CSV delimiter            (default: ,)\n"
        << "  --no-header              Treat CSV as headerless (target must be index)\n"
        << "  --no-square-features     Disable x^2 derived features\n"
        << "  --pairwise-product       Enable pairwise product derived features\n"
        << "  --max-pairwise <n>       Cap pairwise derived features (default: 64)\n"
        << "  --no-save-model          Skip saving trained model\n"
        << "  --help                   Show this help message\n";
}

bool parse_double_strict(const std::string& s, double& out) {
    try {
        size_t pos = 0;
        out = std::stod(s, &pos);
        return pos == s.size() && std::isfinite(out);
    } catch (...) {
        return false;
    }
}

bool parse_uint32_strict(const std::string& s, std::uint32_t& out) {
    try {
        size_t pos = 0;
        unsigned long v = std::stoul(s, &pos);
        if (pos != s.size() || v > std::numeric_limits<std::uint32_t>::max()) {
            return false;
        }
        out = static_cast<std::uint32_t>(v);
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_size_t_strict(const std::string& s, std::size_t& out) {
    try {
        size_t pos = 0;
        unsigned long long v = std::stoull(s, &pos);
        if (pos != s.size()) return false;
        out = static_cast<std::size_t>(v);
        return true;
    } catch (...) {
        return false;
    }
}

CliOptions parse_args(int argc, char** argv) {
    CliOptions opt{};

    auto require_value = [&](int& i, std::string_view flag) -> std::string {
        if (i + 1 >= argc) {
            throw std::invalid_argument("Missing value for argument: " + std::string(flag));
        }
        ++i;
        return argv[i];
    };

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--input") {
            opt.input_path = require_value(i, arg);
        } else if (arg == "--target") {
            opt.target_column = require_value(i, arg);
        } else if (arg == "--model") {
            opt.model_type = require_value(i, arg);
        } else if (arg == "--outdir" || arg == "--output-dir") {
            opt.output_dir = require_value(i, arg);
        } else if (arg == "--test-ratio") {
            double v = 0.0;
            const auto s = require_value(i, arg);
            if (!parse_double_strict(s, v)) {
                throw std::invalid_argument("Invalid --test-ratio: " + s);
            }
            opt.test_ratio = v;
        } else if (arg == "--seed") {
            std::uint32_t v = 0;
            const auto s = require_value(i, arg);
            if (!parse_uint32_strict(s, v)) {
                throw std::invalid_argument("Invalid --seed: " + s);
            }
            opt.seed = v;
        } else if (arg == "--delimiter") {
            const auto s = require_value(i, arg);
            if (s.empty()) {
                throw std::invalid_argument("Invalid --delimiter: empty");
            }
            opt.delimiter = s[0];
        } else if (arg == "--no-header") {
            opt.has_header = false;
        } else if (arg == "--no-square-features") {
            opt.add_square_features = false;
        } else if (arg == "--pairwise-product") {
            opt.add_pairwise_product = true;
        } else if (arg == "--max-pairwise") {
            std::size_t n = 0;
            const auto s = require_value(i, arg);
            if (!parse_size_t_strict(s, n)) {
                throw std::invalid_argument("Invalid --max-pairwise: " + s);
            }
            opt.max_pairwise_features = n;
        } else if (arg == "--no-save-model") {
            opt.save_model = false;
        } else {
            throw std::invalid_argument("Unknown argument: " + arg);
        }
    }

    if (opt.input_path.empty()) {
        throw std::invalid_argument("--input is required");
    }
    if (opt.target_column.empty()) {
        throw std::invalid_argument("--target is required");
    }
    if (!(opt.test_ratio > 0.0 && opt.test_ratio < 1.0)) {
        throw std::invalid_argument("--test-ratio must be in (0,1)");
    }

    return opt;
}

// ============================================================
// CSV loading / basic preprocessing (local adapter for MVP)
// ============================================================

std::string trim_copy(const std::string& s) {
    std::size_t b = 0;
    while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) ++b;

    std::size_t e = s.size();
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;

    return s.substr(b, e - b);
}

bool is_missing_token(const std::string& token) {
    const std::string t = trim_copy(token);
    if (t.empty()) return true;

    std::string lower;
    lower.reserve(t.size());
    for (char ch : t) {
        lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    }
    return (lower == "nan" || lower == "na" || lower == "null" || lower == "none" || lower == "?");
}

// Basic CSV row parser with quote support.
std::vector<std::string> parse_csv_row(const std::string& line, char delimiter) {
    std::vector<std::string> row;
    std::string cell;
    bool in_quotes = false;

    for (std::size_t i = 0; i < line.size(); ++i) {
        const char ch = line[i];

        if (in_quotes) {
            if (ch == '"') {
                if (i + 1 < line.size() && line[i + 1] == '"') {
                    cell.push_back('"'); // escaped quote
                    ++i;
                } else {
                    in_quotes = false;
                }
            } else {
                cell.push_back(ch);
            }
        } else {
            if (ch == '"') {
                in_quotes = true;
            } else if (ch == delimiter) {
                row.push_back(trim_copy(cell));
                cell.clear();
            } else if (ch == '\r') {
                // ignore
            } else {
                cell.push_back(ch);
            }
        }
    }

    row.push_back(trim_copy(cell));
    return row;
}

struct CsvTable {
    std::vector<std::string> header;
    std::vector<std::vector<std::string>> rows;
};

CsvTable load_csv_table(const std::string& path, bool has_header, char delimiter) {
    std::ifstream ifs(path);
    if (!ifs) {
        throw std::runtime_error("Failed to open input CSV: " + path);
    }

    CsvTable table;
    std::string line;
    std::size_t line_no = 0;
    std::size_t expected_cols = 0;
    bool header_consumed = false;

    while (std::getline(ifs, line)) {
        ++line_no;
        if (line.empty()) {
            continue;
        }

        auto row = parse_csv_row(line, delimiter);

        if (!header_consumed && has_header) {
            table.header = std::move(row);
            expected_cols = table.header.size();
            header_consumed = true;
            continue;
        }

        if (!header_consumed && !has_header) {
            expected_cols = row.size();
            table.header.reserve(expected_cols);
            for (std::size_t c = 0; c < expected_cols; ++c) {
                table.header.push_back("col_" + std::to_string(c));
            }
            header_consumed = true;
        }

        if (row.size() < expected_cols) {
            row.resize(expected_cols);
        } else if (row.size() > expected_cols) {
            // Keep strict width (ignore extra trailing cells to preserve rectangular shape).
            row.resize(expected_cols);
        }

        table.rows.push_back(std::move(row));
    }

    if (table.header.empty()) {
        throw std::runtime_error("CSV appears empty or has no readable rows");
    }
    if (table.rows.empty()) {
        throw std::runtime_error("CSV has header but no data rows");
    }

    return table;
}

std::size_t resolve_target_column_index(const CsvTable& table, const std::string& target_spec) {
    // Try numeric index first
    std::size_t idx = 0;
    if (parse_size_t_strict(target_spec, idx)) {
        if (idx >= table.header.size()) {
            throw std::out_of_range("Target column index out of range: " + target_spec);
        }
        return idx;
    }

    // Else treat as column name
    for (std::size_t c = 0; c < table.header.size(); ++c) {
        if (table.header[c] == target_spec) {
            return c;
        }
    }

    std::ostringstream oss;
    oss << "Target column not found: '" << target_spec << "'. Available columns: ";
    for (std::size_t c = 0; c < table.header.size(); ++c) {
        if (c) oss << ", ";
        oss << table.header[c];
    }
    throw std::invalid_argument(oss.str());
}

struct PreprocessSummary {
    std::size_t total_rows = 0;
    std::size_t kept_rows = 0;
    std::size_t dropped_rows_invalid_target = 0;
    std::size_t dropped_rows_no_usable_features = 0;
    std::size_t numeric_feature_columns = 0;
    std::size_t dropped_non_numeric_feature_columns = 0;
    std::size_t imputed_feature_values = 0;
};

NumericDataset build_numeric_dataset_from_csv(
    const CsvTable& table,
    std::size_t target_col_idx,
    PreprocessSummary* summary_out = nullptr
) {
    if (target_col_idx >= table.header.size()) {
        throw std::out_of_range("build_numeric_dataset_from_csv: target_col_idx out of range");
    }

    const std::size_t total_cols = table.header.size();
    const std::size_t total_rows = table.rows.size();

    // Identify numeric-capable feature columns (excluding target). Missing is allowed.
    std::vector<std::size_t> candidate_feature_cols;
    std::vector<bool> keep_feature_col(total_cols, false);

    for (std::size_t c = 0; c < total_cols; ++c) {
        if (c == target_col_idx) continue;

        bool has_any_non_missing = false;
        bool all_non_missing_parseable = true;

        for (const auto& row : table.rows) {
            const std::string token = (c < row.size()) ? row[c] : "";
            if (is_missing_token(token)) continue;

            has_any_non_missing = true;
            double v = 0.0;
            if (!parse_double_strict(token, v)) {
                all_non_missing_parseable = false;
                break;
            }
        }

        if (all_non_missing_parseable && has_any_non_missing) {
            keep_feature_col[c] = true;
            candidate_feature_cols.push_back(c);
        }
    }

    if (candidate_feature_cols.empty()) {
        throw std::runtime_error("No usable numeric feature columns found (excluding target)");
    }

    // First pass: parse target + features, mark missing feature values for later mean imputation.
    struct ParsedRow {
        std::vector<double> features;
        std::vector<bool> missing_mask;
        double target = 0.0;
    };

    std::vector<ParsedRow> parsed_rows;
    parsed_rows.reserve(total_rows);

    std::vector<double> feature_sums(candidate_feature_cols.size(), 0.0);
    std::vector<std::size_t> feature_counts(candidate_feature_cols.size(), 0);

    PreprocessSummary summary{};
    summary.total_rows = total_rows;
    summary.numeric_feature_columns = candidate_feature_cols.size();
    summary.dropped_non_numeric_feature_columns =
        (total_cols - 1) - candidate_feature_cols.size();

    for (const auto& row : table.rows) {
        const std::string target_token = (target_col_idx < row.size()) ? row[target_col_idx] : "";
        double y = 0.0;
        if (is_missing_token(target_token) || !parse_double_strict(target_token, y)) {
            ++summary.dropped_rows_invalid_target;
            continue;
        }

        ParsedRow pr;
        pr.features.resize(candidate_feature_cols.size(), 0.0);
        pr.missing_mask.resize(candidate_feature_cols.size(), false);
        pr.target = y;

        std::size_t usable_non_missing_features = 0;

        for (std::size_t j = 0; j < candidate_feature_cols.size(); ++j) {
            const std::size_t c = candidate_feature_cols[j];
            const std::string token = (c < row.size()) ? row[c] : "";

            if (is_missing_token(token)) {
                pr.missing_mask[j] = true;
                continue;
            }

            double x = 0.0;
            // Should parse due to column screening, but keep defensive fallback.
            if (!parse_double_strict(token, x)) {
                pr.missing_mask[j] = true;
                continue;
            }

            pr.features[j] = x;
            feature_sums[j] += x;
            feature_counts[j] += 1;
            ++usable_non_missing_features;
        }

        if (usable_non_missing_features == 0) {
            ++summary.dropped_rows_no_usable_features;
            continue;
        }

        parsed_rows.push_back(std::move(pr));
    }

    if (parsed_rows.empty()) {
        throw std::runtime_error("No valid rows remain after preprocessing");
    }

    // Compute per-feature means for imputation. If a feature has no observed values, drop it.
    std::vector<bool> keep_parsed_feature(candidate_feature_cols.size(), true);
    std::vector<double> feature_means(candidate_feature_cols.size(), 0.0);
    std::size_t final_feature_count = 0;

    for (std::size_t j = 0; j < candidate_feature_cols.size(); ++j) {
        if (feature_counts[j] == 0) {
            keep_parsed_feature[j] = false;
            continue;
        }
        feature_means[j] = feature_sums[j] / static_cast<double>(feature_counts[j]);
        ++final_feature_count;
    }

    if (final_feature_count == 0) {
        throw std::runtime_error("All numeric feature columns are empty/missing after screening");
    }

    NumericDataset ds;
    ds.target_name = table.header[target_col_idx];
    ds.target.reserve(parsed_rows.size());
    ds.feature_names.reserve(final_feature_count);
    ds.features.reserve(parsed_rows.size());

    for (std::size_t j = 0; j < candidate_feature_cols.size(); ++j) {
        if (keep_parsed_feature[j]) {
            ds.feature_names.push_back(table.header[candidate_feature_cols[j]]);
        }
    }

    for (auto& pr : parsed_rows) {
        std::vector<double> out_row;
        out_row.reserve(final_feature_count);

        for (std::size_t j = 0; j < candidate_feature_cols.size(); ++j) {
            if (!keep_parsed_feature[j]) continue;

            double v = pr.features[j];
            if (pr.missing_mask[j]) {
                v = feature_means[j];
                ++summary.imputed_feature_values;
            }

            if (!std::isfinite(v)) {
                throw std::runtime_error("Non-finite feature value after imputation");
            }

            out_row.push_back(v);
        }

        ds.features.push_back(std::move(out_row));
        ds.target.push_back(pr.target);
    }

    summary.kept_rows = ds.features.size();
    summary.numeric_feature_columns = ds.feature_names.size();

    if (summary_out) {
        *summary_out = summary;
    }

    return ds;
}

// ============================================================
// Dataset split helpers
// ============================================================

struct TrainTestSplit {
    NumericDataset train;
    NumericDataset test;
    std::vector<std::size_t> train_indices; // original row indices
    std::vector<std::size_t> test_indices;  // original row indices
};

NumericDataset subset_rows(const NumericDataset& ds, const std::vector<std::size_t>& indices) {
    NumericDataset out;
    out.feature_names = ds.feature_names;
    out.target_name = ds.target_name;
    out.features.reserve(indices.size());
    out.target.reserve(indices.size());

    for (std::size_t idx : indices) {
        if (idx >= ds.features.size()) {
            throw std::out_of_range("subset_rows: row index out of range");
        }
        out.features.push_back(ds.features[idx]);
        if (!ds.target.empty()) {
            out.target.push_back(ds.target[idx]);
        }
    }

    return out;
}

TrainTestSplit split_train_test_dataset(
    const NumericDataset& ds,
    double test_ratio,
    std::uint32_t seed
) {
    if (ds.features.empty()) {
        throw std::invalid_argument("split_train_test_dataset: empty dataset");
    }
    if (!(test_ratio > 0.0 && test_ratio < 1.0)) {
        throw std::invalid_argument("split_train_test_dataset: test_ratio must be in (0,1)");
    }

    const std::size_t n = ds.features.size();
    std::vector<std::size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    std::mt19937 rng(seed);
    std::shuffle(indices.begin(), indices.end(), rng);

    std::size_t test_count = static_cast<std::size_t>(std::llround(test_ratio * static_cast<double>(n)));
    test_count = std::max<std::size_t>(1, std::min<std::size_t>(n - 1, test_count));
    const std::size_t train_count = n - test_count;

    TrainTestSplit split;
    split.train_indices.assign(indices.begin(), indices.begin() + static_cast<std::ptrdiff_t>(train_count));
    split.test_indices.assign(indices.begin() + static_cast<std::ptrdiff_t>(train_count), indices.end());

    split.train = subset_rows(ds, split.train_indices);
    split.test = subset_rows(ds, split.test_indices);
    return split;
}

// ============================================================
// Model selection
// ============================================================

mof::modeling::ModelType parse_model_type(std::string model) {
    std::string lower;
    lower.reserve(model.size());
    for (char ch : model) {
        lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    }

    if (lower == "linear" || lower == "linreg" || lower == "lr") {
        return mof::modeling::ModelType::LinearRegression;
    }
    if (lower == "rf" || lower == "randomforest" || lower == "random_forest") {
        return mof::modeling::ModelType::RandomForest;
    }
    if (lower == "svm" || lower == "svr") {
        return mof::modeling::ModelType::SVM;
    }
    if (lower == "nn" || lower == "mlp" || lower == "neuralnet" || lower == "neural_net") {
        return mof::modeling::ModelType::NeuralNet;
    }

    throw std::invalid_argument("Unsupported --model type: " + model);
}

// ============================================================
// Output writers
// ============================================================

void ensure_dir(const fs::path& dir) {
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) {
        throw std::runtime_error("Failed to create directory '" + dir.string() + "': " + ec.message());
    }
}

void write_predictions_csv(
    const fs::path& path,
    const std::vector<std::size_t>& original_indices,
    const std::vector<double>& y_true,
    const std::vector<double>& y_pred
) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("write_predictions_csv: y_true/y_pred size mismatch");
    }
    if (!original_indices.empty() && original_indices.size() != y_true.size()) {
        throw std::invalid_argument("write_predictions_csv: original_indices size mismatch");
    }

    ensure_dir(path.parent_path());

    std::ofstream ofs(path);
    if (!ofs) {
        throw std::runtime_error("Failed to open predictions output: " + path.string());
    }

    ofs << "row_index,y_true,y_pred,error,abs_error\n";
    ofs << std::setprecision(17);

    for (std::size_t i = 0; i < y_true.size(); ++i) {
        const double err = y_pred[i] - y_true[i];
        const double abs_err = std::abs(err);
        const std::size_t row_index = original_indices.empty() ? i : original_indices[i];

        ofs << row_index << ','
            << y_true[i] << ','
            << y_pred[i] << ','
            << err << ','
            << abs_err << '\n';
    }
}

void write_metrics_report(
    const fs::path& path,
    const CliOptions& cli,
    const PreprocessSummary& prep,
    const NumericDataset& cleaned,
    const NumericDataset& engineered_all,
    const TrainTestSplit& split,
    const mof::modeling::TrainSummary& train_summary,
    const mof::evaluation::RegressionMetrics& test_metrics
) {
    ensure_dir(path.parent_path());

    std::ofstream ofs(path);
    if (!ofs) {
        throw std::runtime_error("Failed to open metrics report output: " + path.string());
    }

    ofs << "MOF Property Prediction Tool - Run Report\n";
    ofs << "========================================\n\n";

    ofs << "[CLI]\n";
    ofs << "input_path=" << cli.input_path << "\n";
    ofs << "target_column=" << cli.target_column << "\n";
    ofs << "model_type=" << cli.model_type << "\n";
    ofs << "output_dir=" << cli.output_dir << "\n";
    ofs << "test_ratio=" << cli.test_ratio << "\n";
    ofs << "seed=" << cli.seed << "\n\n";

    ofs << "[Preprocessing]\n";
    ofs << "total_rows=" << prep.total_rows << "\n";
    ofs << "kept_rows=" << prep.kept_rows << "\n";
    ofs << "dropped_rows_invalid_target=" << prep.dropped_rows_invalid_target << "\n";
    ofs << "dropped_rows_no_usable_features=" << prep.dropped_rows_no_usable_features << "\n";
    ofs << "numeric_feature_columns=" << prep.numeric_feature_columns << "\n";
    ofs << "dropped_non_numeric_feature_columns=" << prep.dropped_non_numeric_feature_columns << "\n";
    ofs << "imputed_feature_values=" << prep.imputed_feature_values << "\n\n";

    ofs << "[Dataset]\n";
    ofs << "target_name=" << cleaned.target_name << "\n";
    ofs << "cleaned_rows=" << cleaned.features.size() << "\n";
    ofs << "cleaned_feature_count=" << cleaned.feature_names.size() << "\n";
    ofs << "engineered_rows=" << engineered_all.features.size() << "\n";
    ofs << "engineered_feature_count=" << engineered_all.feature_names.size() << "\n";
    ofs << "train_rows=" << split.train.features.size() << "\n";
    ofs << "test_rows=" << split.test.features.size() << "\n\n";

    ofs << "[Model]\n";
    ofs << "model_name=" << train_summary.model_name << "\n";
    ofs << "fitted=" << (train_summary.fitted ? 1 : 0) << "\n";
    ofs << "train_feature_count=" << train_summary.feature_count << "\n";
    if (train_summary.training_metrics.has_value()) {
        ofs << "train_metrics="
            << mof::evaluation::format_metrics(*train_summary.training_metrics, 8) << "\n";
    }
    ofs << "\n";

    ofs << "[Test Metrics]\n";
    ofs << mof::evaluation::format_metrics(test_metrics, 8) << "\n";
}

// ============================================================
// Main pipeline runner
// ============================================================

int run_pipeline(const CliOptions& cli) {
    // Output layout
    const fs::path out_root(cli.output_dir);
    const fs::path processed_dir = out_root / "data" / "processed";
    const fs::path features_dir  = out_root / "data" / "features";
    const fs::path reports_dir   = out_root / "reports";
    const fs::path models_dir    = out_root / "models";

    ensure_dir(processed_dir);
    ensure_dir(features_dir);
    ensure_dir(reports_dir);
    ensure_dir(models_dir);

    std::cout << "[1/8] Loading raw CSV...\n";
    CsvTable table = load_csv_table(cli.input_path, cli.has_header, cli.delimiter);
    const std::size_t target_idx = resolve_target_column_index(table, cli.target_column);

    std::cout << "[2/8] Preprocessing (filter + impute numeric features)...\n";
    PreprocessSummary prep_summary{};
    NumericDataset cleaned = build_numeric_dataset_from_csv(table, target_idx, &prep_summary);

    // Save cleaned dataset using feature export helper (works for any NumericDataset)
    {
        mof::feature_engineering::FeatureExportOptions export_opt{};
        export_opt.include_target = true;
        mof::feature_engineering::export_feature_matrix(
            cleaned, (processed_dir / "cleaned.csv").string(), export_opt);
    }

    std::cout << "    Kept rows: " << cleaned.features.size()
              << " | Features: " << cleaned.feature_names.size()
              << " | Target: " << cleaned.target_name << "\n";

    std::cout << "[3/8] Feature engineering (descriptor selection + derived features)...\n";
    mof::feature_engineering::DescriptorSelectionOptions sel_opt{};
    sel_opt.mode = mof::feature_engineering::DescriptorSelectionMode::All;
    NumericDataset selected = mof::feature_engineering::select_descriptors(cleaned, sel_opt);

    mof::feature_engineering::DerivedFeatureOptions dopt{};
    dopt.add_square = cli.add_square_features;
    dopt.add_pairwise_product = cli.add_pairwise_product;
    dopt.max_pairwise_features = cli.max_pairwise_features;
    dopt.pairwise_upper_triangle_only = true;
    dopt.eps = 1e-12;

    mof::feature_engineering::DerivedFeatureSummary dsummary{};
    NumericDataset engineered = mof::feature_engineering::create_derived_features(selected, dopt, &dsummary);

    {
        mof::feature_engineering::FeatureExportOptions export_opt{};
        export_opt.include_target = true;
        mof::feature_engineering::export_feature_matrix(
            engineered, (features_dir / "engineered_all_unscaled.csv").string(), export_opt);
    }

    std::cout << "    Original features: " << dsummary.original_feature_count
              << " | Added: " << dsummary.added_feature_count
              << " | Final: " << dsummary.final_feature_count << "\n";

    std::cout << "[4/8] Splitting data...\n";
    TrainTestSplit split = split_train_test_dataset(engineered, cli.test_ratio, cli.seed);
    std::cout << "    Train rows: " << split.train.features.size()
              << " | Test rows: " << split.test.features.size() << "\n";

    std::cout << "[5/8] Scaling features (fit on train, apply to test)...\n";
    {
        mof::feature_engineering::FeatureScalingOptions sopt{};
        sopt.method = mof::feature_engineering::FeatureScalingMethod::ZScore;
        sopt.eps = 1e-12;

        auto scaler = mof::feature_engineering::fit_and_apply_feature_scaling(split.train, sopt);
        mof::feature_engineering::apply_feature_scaling(split.test, scaler, sopt);

        mof::feature_engineering::FeatureExportOptions export_opt{};
        export_opt.include_target = true;
        mof::feature_engineering::export_feature_matrix(
            split.train, (features_dir / "train_scaled.csv").string(), export_opt);
        mof::feature_engineering::export_feature_matrix(
            split.test, (features_dir / "test_scaled.csv").string(), export_opt);
    }

    std::cout << "[6/8] Training model...\n";
    const auto model_type = parse_model_type(cli.model_type);
    auto model = mof::modeling::create_model(model_type);
    if (!model) {
        throw std::runtime_error("Failed to create requested model");
    }

    mof::modeling::FitOptions fit_opt{};
    fit_opt.enable_training_metrics = true;
    fit_opt.random_seed = cli.seed;
    auto train_summary = model->fit(split.train.features, split.train.target, fit_opt);

    std::cout << "    Model: " << train_summary.model_name << "\n";
    if (train_summary.training_metrics.has_value()) {
        std::cout << "    Train metrics: "
                  << mof::evaluation::format_metrics(*train_summary.training_metrics, 6) << "\n";
    }

    std::cout << "[7/8] Predicting + evaluating...\n";
    auto y_pred = model->predict(split.test.features);
    auto test_metrics = mof::evaluation::evaluate_regression(split.test.target, y_pred);

    std::cout << "    Test metrics: "
              << mof::evaluation::format_metrics(test_metrics, 6) << "\n";

    std::cout << "[8/8] Saving results...\n";
    write_predictions_csv(reports_dir / "predictions.csv", split.test_indices, split.test.target, y_pred);
    write_metrics_report(
        reports_dir / "metrics.txt",
        cli,
        prep_summary,
        cleaned,
        engineered,
        split,
        train_summary,
        test_metrics
    );

    if (cli.save_model) {
        try {
            model->save_model((models_dir / "model.kv").string());
        } catch (const std::exception& e) {
            std::cerr << "    Warning: model save skipped/failed: " << e.what() << "\n";
        }
    }

    std::cout << "Done.\n";
    std::cout << "Outputs saved under: " << fs::absolute(out_root).string() << "\n";
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const CliOptions cli = parse_args(argc, argv);
        return run_pipeline(cli);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n\n";
        print_usage(argv[0]);
        return 1;
    } catch (...) {
        std::cerr << "Error: unknown exception\n";
        return 1;
    }
}