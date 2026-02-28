#include "preprocessing.h"
#include "modeling.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

void die(const std::string& msg) {
    std::cerr << "ERROR: " << msg << "\n";
    std::exit(1);
}

std::size_t infer_total_columns(const mof::preprocessing::TabularData& t) {
    if (!t.column_names.empty()) return t.column_names.size();
    if (!t.rows.empty()) return t.rows.front().size();
    return 0;
}

void write_predictions_csv(const std::string& out_path,
                           const mof::preprocessing::NumericDataset& test_ds,
                           const std::vector<double>& y_pred) {
    if (out_path.empty()) return;
    if (test_ds.features.size() != y_pred.size()) {
        die("write_predictions_csv: size mismatch");
    }

    fs::path p(out_path);
    if (p.has_parent_path()) {
        std::error_code ec;
        fs::create_directories(p.parent_path(), ec);
        if (ec) die("Failed to create output dir: " + ec.message());
    }

    std::ofstream ofs(out_path);
    if (!ofs) die("Failed to open output file: " + out_path);

    ofs << std::setprecision(17);

    // Header: feature columns + y_true + y_pred + abs_error
    const std::size_t cols = test_ds.features.empty() ? 0 : test_ds.features.front().size();
    std::vector<std::string> fnames = test_ds.feature_names;
    if (fnames.size() != cols) {
        fnames.clear();
        fnames.reserve(cols);
        for (std::size_t c = 0; c < cols; ++c) {
            fnames.push_back("feature_" + std::to_string(c));
        }
    }

    for (std::size_t c = 0; c < cols; ++c) {
        if (c) ofs << ',';
        ofs << fnames[c];
    }
    if (cols) ofs << ',';
    ofs << "y_true,y_pred,abs_error\n";

    const bool has_target = !test_ds.target.empty();
    for (std::size_t i = 0; i < test_ds.features.size(); ++i) {
        for (std::size_t c = 0; c < cols; ++c) {
            if (c) ofs << ',';
            ofs << test_ds.features[i][c];
        }
        if (cols) ofs << ',';

        const double y_true = has_target ? test_ds.target[i] : std::numeric_limits<double>::quiet_NaN();
        const double err = has_target ? std::fabs(y_true - y_pred[i]) : std::numeric_limits<double>::quiet_NaN();

        ofs << y_true << ',' << y_pred[i] << ',' << err << '\n';
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage:\n"
                  << "  manual_pipeline_test <input.csv> [predictions_out.csv]\n\n"
                  << "Notes:\n"
                  << "  - Assumes last column is the regression target.\n"
                  << "  - Normalizes features (ZScore), splits train/test, trains LinearRegression.\n";
        return 2;
    }

    const std::string input_path = argv[1];
    const std::string pred_out_path = (argc >= 3) ? argv[2] : std::string();

    using namespace mof;

    // 1) Load CSV
    preprocessing::CsvLoadOptions load_opt{};
    load_opt.has_header = true;
    auto table = preprocessing::load_csv_data(input_path, load_opt);

    const std::size_t total_cols = infer_total_columns(table);
    if (total_cols < 2) {
        die("CSV must have at least 2 columns (>=1 feature + 1 target).");
    }
    const std::size_t target_col = total_cols - 1;

    std::cout << "Loaded rows: " << table.rows.size()
              << ", columns: " << total_cols << "\n";

    // 2) Fill missing feature values (mean) only on feature columns
    preprocessing::MissingValueOptions miss{};
    miss.strategy = preprocessing::MissingValueStrategy::FillWithMean;
    miss.missing_tokens = {"", "NA", "NaN", "nan", "null", "NULL"};

    miss.column_indices.clear();
    miss.column_indices.reserve(target_col);
    for (std::size_t c = 0; c < target_col; ++c) miss.column_indices.push_back(c);

    const auto filled = preprocessing::handle_missing_values(table, miss);
    std::cout << "Filled missing feature values: " << filled << "\n";

    // 3) Remove invalid rows (must parse numeric for all features + target)
    preprocessing::RowValidationOptions rv{};
    rv.require_consistent_column_count = true;
    rv.numeric_columns.clear();
    rv.numeric_columns.reserve(total_cols);
    for (std::size_t c = 0; c < total_cols; ++c) rv.numeric_columns.push_back(c);
    rv.target_column_index = target_col;

    const auto removed = preprocessing::remove_invalid_rows(table, rv);
    std::cout << "Removed invalid rows: " << removed << "\n";
    std::cout << "Remaining rows: " << table.rows.size() << "\n";

    // 4) Convert to numeric dataset
    preprocessing::NumericConversionOptions conv{};
    conv.target_column_index = target_col;
    conv.drop_rows_with_parse_errors = true;

    auto dataset = preprocessing::to_numeric_dataset(table, conv);
    std::cout << "Numeric dataset -> rows: " << dataset.features.size()
              << ", features: " << (dataset.features.empty() ? 0 : dataset.features.front().size())
              << "\n";

    if (dataset.features.empty() || dataset.target.empty()) {
        die("Dataset ended up empty after cleaning/conversion.");
    }

    // 5) Normalize features (ZScore)
    preprocessing::NormalizationOptions norm{};
    norm.method = utils::NormalizationMethod::ZScore;
    norm.eps = 1e-12;

    auto norm_res = preprocessing::normalize_columns(dataset, norm);

    // 6) Split train/test
    preprocessing::TrainTestSplitOptions split_opt{};
    split_opt.test_ratio = 0.25;
    split_opt.shuffle = true;
    split_opt.random_seed = 42;

    auto split = preprocessing::split_train_test(dataset, split_opt);

    // Apply same normalization to test split (using fitted stats)
    preprocessing::apply_normalization(split.test, norm_res.feature_stats, norm);

    std::cout << "Train rows: " << split.train.features.size()
              << ", Test rows: " << split.test.features.size() << "\n";

    // 7) Train model
    modeling::LinearRegressionModel model;
    modeling::FitOptions fit_opt{};
    fit_opt.enable_training_metrics = true;

    auto summary = model.fit(split.train.features, split.train.target, fit_opt);

    std::cout << "Model trained: " << summary.model_name
              << " | fitted=" << (summary.fitted ? "true" : "false") << "\n";

    if (summary.training_metrics.has_value()) {
        std::cout << "Train metrics -> RMSE: " << summary.training_metrics->rmse
                  << ", MAE: " << summary.training_metrics->mae
                  << ", R2: " << summary.training_metrics->r2 << "\n";
    }

    // 8) Predict + evaluate on test
    auto y_pred = model.predict(split.test.features);

    std::cout << "Predictions (" << y_pred.size() << "): ";
    for (std::size_t i = 0; i < y_pred.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << std::fixed << std::setprecision(4) << y_pred[i];
    }
    std::cout << "\n";

    auto test_metrics = model.evaluate(split.test.features, split.test.target);
    std::cout << "Test metrics  -> RMSE: " << test_metrics.rmse
              << ", MAE: " << test_metrics.mae
              << ", R2: " << test_metrics.r2 << "\n";

    // 9) Optional: write predictions CSV for debugging
    if (!pred_out_path.empty()) {
        write_predictions_csv(pred_out_path, split.test, y_pred);
        std::cout << "Wrote predictions to: " << pred_out_path << "\n";
    }

    std::cout << "Manual pipeline test completed successfully.\n";
    return 0;
}
