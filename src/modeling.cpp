 #include <memory> // included before modeling.h because modeling.h declares std::unique_ptr

#include "modeling.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mof::modeling {
namespace {

constexpr double kPivotEps = 1e-12;
constexpr double kRidgeRetryStart = 1e-12;

// ============================================================
// Internal helpers: validation / math
// ============================================================

void validate_finite_scalar(double v, std::string_view name, std::string_view caller) {
    if (!std::isfinite(v)) {
        throw std::invalid_argument(std::string(caller) + ": non-finite value in " + std::string(name));
    }
}

std::size_t infer_feature_count(const utils::Matrix& X, std::string_view caller) {
    if (X.empty()) {
        throw std::invalid_argument(std::string(caller) + ": X must be non-empty");
    }
    if (X.front().empty()) {
        throw std::invalid_argument(std::string(caller) + ": X must have at least one feature column");
    }
    return X.front().size();
}

double mean_of(const std::vector<double>& y) {
    if (y.empty()) {
        throw std::invalid_argument("mean_of: empty vector");
    }
    const double s = std::accumulate(y.begin(), y.end(), 0.0);
    return s / static_cast<double>(y.size());
}

void ensure_predict_feature_count(const utils::Matrix& X, std::size_t expected_cols, std::string_view caller) {
    validate_feature_matrix(X, caller);
    const std::size_t cols = X.front().size();
    if (cols != expected_cols) {
        throw std::invalid_argument(
            std::string(caller) + ": feature count mismatch (expected " +
            std::to_string(expected_cols) + ", got " + std::to_string(cols) + ")"
        );
    }
}

double dot_row_with_weights(const std::vector<double>& row, const std::vector<double>& w) {
    double s = 0.0;
    for (std::size_t j = 0; j < row.size(); ++j) {
        s += row[j] * w[j];
    }
    return s;
}

// Solve A x = b using Gaussian elimination with partial pivoting.
// A is copied intentionally (small-to-medium baseline use, avoids mutating caller).
std::vector<double> solve_linear_system_gaussian(
    std::vector<std::vector<double>> A,
    std::vector<double> b,
    double pivot_eps = kPivotEps
) {
    const std::size_t n = A.size();
    if (n == 0 || b.size() != n) {
        throw std::invalid_argument("solve_linear_system_gaussian: invalid dimensions");
    }
    for (const auto& row : A) {
        if (row.size() != n) {
            throw std::invalid_argument("solve_linear_system_gaussian: matrix must be square");
        }
    }
    if (pivot_eps <= 0.0) {
        throw std::invalid_argument("solve_linear_system_gaussian: pivot_eps must be > 0");
    }

    // Forward elimination
    for (std::size_t col = 0; col < n; ++col) {
        // Pivot search
        std::size_t pivot_row = col;
        double max_abs = std::abs(A[col][col]);
        for (std::size_t r = col + 1; r < n; ++r) {
            const double v = std::abs(A[r][col]);
            if (v > max_abs) {
                max_abs = v;
                pivot_row = r;
            }
        }

        if (max_abs <= pivot_eps) {
            throw std::runtime_error("solve_linear_system_gaussian: singular or ill-conditioned system");
        }

        if (pivot_row != col) {
            std::swap(A[pivot_row], A[col]);
            std::swap(b[pivot_row], b[col]);
        }

        // Eliminate below
        const double pivot = A[col][col];
        for (std::size_t r = col + 1; r < n; ++r) {
            const double factor = A[r][col] / pivot;
            if (factor == 0.0) {
                continue;
            }

            A[r][col] = 0.0; // exact by construction
            for (std::size_t c = col + 1; c < n; ++c) {
                A[r][c] -= factor * A[col][c];
            }
            b[r] -= factor * b[col];
        }
    }

    // Back substitution
    std::vector<double> x(n, 0.0);
    for (std::size_t i = n; i-- > 0;) {
        double rhs = b[i];
        for (std::size_t c = i + 1; c < n; ++c) {
            rhs -= A[i][c] * x[c];
        }

        const double diag = A[i][i];
        if (std::abs(diag) <= pivot_eps) {
            throw std::runtime_error("solve_linear_system_gaussian: zero diagonal during back substitution");
        }

        x[i] = rhs / diag;
        if (!std::isfinite(x[i])) {
            throw std::runtime_error("solve_linear_system_gaussian: non-finite solution value");
        }
    }

    return x;
}

struct LinearFitResult {
    std::vector<double> coefficients;
    double intercept = 0.0;
};

LinearFitResult fit_linear_regression_normal_equation(
    const utils::Matrix& X,
    const std::vector<double>& y,
    bool fit_intercept,
    double l2_regularization
) {
    validate_training_data(X, y, "LinearRegressionModel::fit");
    if (l2_regularization < 0.0) {
        throw std::invalid_argument("LinearRegressionModel::fit: l2_regularization must be >= 0");
    }

    const std::size_t n = X.size();
    const std::size_t p = X.front().size();
    const std::size_t d = p + (fit_intercept ? 1 : 0);

    std::vector<std::vector<double>> A(d, std::vector<double>(d, 0.0));
    std::vector<double> b(d, 0.0);

    auto x_at = [&](const std::vector<double>& row, std::size_t j) -> double {
        if (fit_intercept) {
            if (j == 0) return 1.0;
            return row[j - 1];
        }
        return row[j];
    };

    // Build X^T X and X^T y
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t r = 0; r < d; ++r) {
            const double xr = x_at(X[i], r);
            b[r] += xr * y[i];

            for (std::size_t c = 0; c < d; ++c) {
                A[r][c] += xr * x_at(X[i], c);
            }
        }
    }

    // Ridge-like regularization (placeholder support from params).
    // Common practice: do not regularize intercept.
    if (l2_regularization > 0.0) {
        for (std::size_t j = 0; j < d; ++j) {
            if (fit_intercept && j == 0) {
                continue;
            }
            A[j][j] += l2_regularization;
        }
    }

    std::vector<double> w;
    try {
        w = solve_linear_system_gaussian(A, b);
    } catch (const std::exception&) {
        // Retry with tiny ridge jitter to improve numerical stability for singular-ish systems.
        const double jitter = std::max(kRidgeRetryStart, l2_regularization);
        for (std::size_t j = 0; j < d; ++j) {
            if (fit_intercept && j == 0) continue;
            A[j][j] += jitter;
        }
        w = solve_linear_system_gaussian(A, b);
    }

    LinearFitResult out{};
    if (fit_intercept) {
        out.intercept = w[0];
        out.coefficients.assign(w.begin() + 1, w.end());
    } else {
        out.intercept = 0.0;
        out.coefficients = std::move(w);
    }
    return out;
}

std::vector<double> predict_linear(
    const utils::Matrix& X,
    const std::vector<double>& coefficients,
    double intercept,
    bool clip_non_finite_outputs,
    std::string_view caller
) {
    ensure_predict_feature_count(X, coefficients.size(), caller);

    std::vector<double> y_pred;
    y_pred.reserve(X.size());

    for (const auto& row : X) {
        double pred = intercept + dot_row_with_weights(row, coefficients);
        if (!std::isfinite(pred)) {
            if (clip_non_finite_outputs) {
                pred = 0.0;
            } else {
                throw std::runtime_error(std::string(caller) + ": non-finite prediction generated");
            }
        }
        y_pred.push_back(pred);
    }

    return y_pred;
}

TrainSummary make_train_summary(
    ModelType type,
    std::string model_name,
    std::size_t rows,
    std::size_t feature_count,
    bool fitted
) {
    TrainSummary s{};
    s.model_type = type;
    s.model_name = std::move(model_name);
    s.train_rows = rows;
    s.feature_count = feature_count;
    s.fitted = fitted;
    return s;
}

// ============================================================
// Internal helpers: lightweight text persistence
// ============================================================

std::string trim_copy(std::string s) {
    const auto not_space = [](unsigned char ch) {
        return !std::isspace(ch);
    };

    auto it1 = std::find_if(s.begin(), s.end(), not_space);
    auto it2 = std::find_if(s.rbegin(), s.rend(), not_space).base();

    if (it1 >= it2) {
        return {};
    }
    return std::string(it1, it2);
}

std::pair<std::string, std::string> split_key_value(const std::string& line) {
    const auto pos = line.find('=');
    if (pos == std::string::npos) {
        return {trim_copy(line), ""};
    }
    return {trim_copy(line.substr(0, pos)), trim_copy(line.substr(pos + 1))};
}

std::vector<std::string> split_csv_tokens(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    for (char ch : s) {
        if (ch == ',') {
            out.push_back(trim_copy(cur));
            cur.clear();
        } else {
            cur.push_back(ch);
        }
    }
    out.push_back(trim_copy(cur));
    return out;
}

std::vector<double> parse_csv_doubles(const std::string& s) {
    if (trim_copy(s).empty()) {
        return {};
    }
    const auto toks = split_csv_tokens(s);
    std::vector<double> out;
    out.reserve(toks.size());
    for (const auto& t : toks) {
        out.push_back(std::stod(t));
    }
    return out;
}

std::string join_csv_doubles(const std::vector<double>& values) {
    std::ostringstream oss;
    oss.precision(17);
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) oss << ',';
        oss << values[i];
    }
    return oss.str();
}

void ensure_writable_output_path(std::string_view path, const ModelIOOptions& options, std::string_view caller) {
    if (path.empty()) {
        throw std::invalid_argument(std::string(caller) + ": path is empty");
    }

    const std::filesystem::path p{std::string(path)};
    if (std::filesystem::exists(p) && !options.overwrite) {
        throw std::runtime_error(std::string(caller) + ": file already exists and overwrite=false");
    }

    if (p.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(p.parent_path(), ec);
        if (ec) {
            throw std::runtime_error(std::string(caller) + ": failed to create directories: " + ec.message());
        }
    }
}

std::unordered_map<std::string, std::string> read_kv_file(std::string_view path, std::string_view caller) {
    if (path.empty()) {
        throw std::invalid_argument(std::string(caller) + ": path is empty");
    }

    std::ifstream ifs{std::string(path)};
    if (!ifs) {
        throw std::runtime_error(std::string(caller) + ": failed to open file for reading");
    }

    std::unordered_map<std::string, std::string> kv;
    std::string line;
    while (std::getline(ifs, line)) {
        line = trim_copy(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }
        const auto [k, v] = split_key_value(line);
        if (!k.empty()) {
            kv[k] = v;
        }
    }

    return kv;
}

bool parse_bool01(const std::string& s) {
    if (s == "1" || s == "true" || s == "TRUE") return true;
    if (s == "0" || s == "false" || s == "FALSE") return false;
    throw std::invalid_argument("parse_bool01: invalid boolean value: " + s);
}

// ============================================================
// Placeholder mean-regressor helpers (safe: no direct private-member access)
// ============================================================

TrainSummary fit_placeholder_mean_regressor_impl(
    bool& fitted,
    std::size_t& feature_count,
    double& fallback_mean_target,
    ModelType model_type,
    const std::string& model_name,
    const utils::Matrix& X,
    const std::vector<double>& y,
    std::string_view caller
) {
    validate_training_data(X, y, caller);

    feature_count = X.front().size();
    fallback_mean_target = mean_of(y);
    fitted = true;

    return make_train_summary(model_type, model_name, X.size(), feature_count, true);
}

std::vector<double> predict_placeholder_mean_regressor_impl(
    bool fitted,
    std::size_t feature_count,
    double fallback_mean_target,
    const utils::Matrix& X,
    std::string_view caller
) {
    if (!fitted) {
        throw std::runtime_error(std::string(caller) + ": model is not fitted");
    }
    ensure_predict_feature_count(X, feature_count, caller);
    return std::vector<double>(X.size(), fallback_mean_target);
}

evaluation::RegressionMetrics evaluate_placeholder_mean_regressor_impl(
    bool fitted,
    std::size_t feature_count,
    double fallback_mean_target,
    const utils::Matrix& X,
    const std::vector<double>& y_true,
    const evaluation::MetricOptions& metric_options,
    const PredictOptions& /*predict_options*/,
    std::string_view caller
) {
    const auto y_pred = predict_placeholder_mean_regressor_impl(
        fitted, feature_count, fallback_mean_target, X, caller
    );
    return evaluation::evaluate_regression(y_true, y_pred, metric_options);
}

void save_placeholder_model_kv_impl(
    bool fitted,
    std::size_t feature_count,
    double fallback_mean_target,
    std::string_view model_name,
    std::string_view path,
    const ModelIOOptions& options,
    std::string_view caller
) {
    if (!fitted) {
        throw std::runtime_error(std::string(caller) + ": model is not fitted");
    }

    ensure_writable_output_path(path, options, caller);

    std::ofstream ofs{std::string(path)};
    if (!ofs) {
        throw std::runtime_error(std::string(caller) + ": failed to open file for writing");
    }

    ofs.precision(17);
    ofs << "format=MOFModelKV\n";
    ofs << "version=1\n";
    ofs << "model_name=" << model_name << "\n";
    ofs << "feature_count=" << feature_count << "\n";
    ofs << "fitted=1\n";
    ofs << "fallback_mean_target=" << fallback_mean_target << "\n";
}

void load_placeholder_model_kv_impl(
    bool& fitted,
    std::size_t& feature_count,
    double& fallback_mean_target,
    std::string_view expected_name,
    std::string_view path,
    std::string_view caller
) {
    const auto kv = read_kv_file(path, caller);

    auto it_name = kv.find("model_name");
    if (it_name == kv.end() || it_name->second != expected_name) {
        throw std::runtime_error(std::string(caller) + ": model_name mismatch in file");
    }

    auto it_fc = kv.find("feature_count");
    auto it_fit = kv.find("fitted");
    auto it_mean = kv.find("fallback_mean_target");
    if (it_fc == kv.end() || it_fit == kv.end() || it_mean == kv.end()) {
        throw std::runtime_error(std::string(caller) + ": missing required fields");
    }

    feature_count = static_cast<std::size_t>(std::stoull(it_fc->second));
    fitted = parse_bool01(it_fit->second);
    fallback_mean_target = std::stod(it_mean->second);

    if (!fitted) {
        throw std::runtime_error(std::string(caller) + ": loaded placeholder model is marked unfitted");
    }
}

} // namespace

// ============================================================
// Public validation helpers
// ============================================================

void validate_feature_matrix(const utils::Matrix& X, std::string_view caller) {
    const std::size_t cols = infer_feature_count(X, caller);

    for (std::size_t i = 0; i < X.size(); ++i) {
        if (X[i].size() != cols) {
            throw std::invalid_argument(
                std::string(caller) + ": X is not rectangular at row " + std::to_string(i)
            );
        }
        for (std::size_t j = 0; j < cols; ++j) {
            validate_finite_scalar(X[i][j], "X", caller);
        }
    }
}

void validate_training_data(
    const utils::Matrix& X,
    const std::vector<double>& y,
    std::string_view caller
) {
    validate_feature_matrix(X, caller);

    if (y.empty()) {
        throw std::invalid_argument(std::string(caller) + ": y must be non-empty");
    }
    if (y.size() != X.size()) {
        throw std::invalid_argument(
            std::string(caller) + ": row mismatch (X rows=" + std::to_string(X.size()) +
            ", y size=" + std::to_string(y.size()) + ")"
        );
    }

    for (double v : y) {
        validate_finite_scalar(v, "y", caller);
    }
}

// ============================================================
// LinearRegressionModel (real baseline implementation)
// ============================================================

LinearRegressionModel::LinearRegressionModel(LinearRegressionParams params)
    : params_(std::move(params)) {}

ModelType LinearRegressionModel::type() const noexcept {
    return ModelType::LinearRegression;
}

std::string LinearRegressionModel::name() const {
    return "LinearRegressionModel";
}

bool LinearRegressionModel::is_fitted() const noexcept {
    return fitted_;
}

TrainSummary LinearRegressionModel::fit(
    const utils::Matrix& X,
    const std::vector<double>& y,
    const FitOptions& options
) {
    validate_training_data(X, y, "LinearRegressionModel::fit");

    const bool effective_fit_intercept = params_.fit_intercept && options.fit_intercept;

    auto fit_result = fit_linear_regression_normal_equation(
        X, y, effective_fit_intercept, params_.l2_regularization);

    coefficients_ = std::move(fit_result.coefficients);
    intercept_ = fit_result.intercept;
    feature_count_ = X.front().size();
    fitted_ = true;

    TrainSummary summary = make_train_summary(type(), name(), X.size(), feature_count_, true);

    if (options.enable_training_metrics) {
        summary.training_metrics = evaluate(X, y);
    }

    return summary;
}

std::vector<double> LinearRegressionModel::predict(
    const utils::Matrix& X,
    const PredictOptions& options
) const {
    if (!fitted_) {
        throw std::runtime_error("LinearRegressionModel::predict: model is not fitted");
    }

    return predict_linear(
        X, coefficients_, intercept_, options.clip_non_finite_outputs, "LinearRegressionModel::predict");
}

evaluation::RegressionMetrics LinearRegressionModel::evaluate(
    const utils::Matrix& X,
    const std::vector<double>& y_true,
    const evaluation::MetricOptions& metric_options,
    const PredictOptions& predict_options
) const {
    const auto y_pred = predict(X, predict_options);
    return evaluation::evaluate_regression(y_true, y_pred, metric_options);
}

void LinearRegressionModel::save_model(
    std::string_view path,
    const ModelIOOptions& options
) const {
    if (!fitted_) {
        throw std::runtime_error("LinearRegressionModel::save_model: model is not fitted");
    }

    ensure_writable_output_path(path, options, "LinearRegressionModel::save_model");

    std::ofstream ofs{std::string(path)};
    if (!ofs) {
        throw std::runtime_error("LinearRegressionModel::save_model: failed to open file for writing");
    }

    ofs.precision(17);
    ofs << "format=MOFModelKV\n";
    ofs << "version=1\n";
    ofs << "model_name=" << name() << "\n";
    ofs << "feature_count=" << feature_count_ << "\n";
    ofs << "fitted=1\n";
    ofs << "fit_intercept=" << (params_.fit_intercept ? 1 : 0) << "\n";
    ofs << "l2_regularization=" << params_.l2_regularization << "\n";
    ofs << "intercept=" << intercept_ << "\n";
    ofs << "coefficients=" << join_csv_doubles(coefficients_) << "\n";
}

void LinearRegressionModel::load_model(std::string_view path) {
    const auto kv = read_kv_file(path, "LinearRegressionModel::load_model");

    auto require = [&](std::string_view key) -> const std::string& {
        auto it = kv.find(std::string(key));
        if (it == kv.end()) {
            throw std::runtime_error("LinearRegressionModel::load_model: missing field: " + std::string(key));
        }
        return it->second;
    };

    if (require("model_name") != name()) {
        throw std::runtime_error("LinearRegressionModel::load_model: model_name mismatch");
    }

    feature_count_ = static_cast<std::size_t>(std::stoull(require("feature_count")));
    fitted_ = parse_bool01(require("fitted"));
    params_.fit_intercept = parse_bool01(require("fit_intercept"));
    params_.l2_regularization = std::stod(require("l2_regularization"));
    intercept_ = std::stod(require("intercept"));
    coefficients_ = parse_csv_doubles(require("coefficients"));

    if (!fitted_) {
        throw std::runtime_error("LinearRegressionModel::load_model: loaded model marked unfitted");
    }
    if (coefficients_.size() != feature_count_) {
        throw std::runtime_error("LinearRegressionModel::load_model: coefficients size mismatch");
    }
}

// ============================================================
// RandomForestModel (placeholder mean regressor implementation)
// ============================================================

RandomForestModel::RandomForestModel(RandomForestParams params)
    : params_(std::move(params)) {}

ModelType RandomForestModel::type() const noexcept {
    return ModelType::RandomForest;
}

std::string RandomForestModel::name() const {
    return "RandomForestModel";
}

bool RandomForestModel::is_fitted() const noexcept {
    return fitted_;
}

TrainSummary RandomForestModel::fit(
    const utils::Matrix& X,
    const std::vector<double>& y,
    const FitOptions& options
) {
    auto s = fit_placeholder_mean_regressor_impl(
        fitted_, feature_count_, fallback_mean_target_,
        type(), name(), X, y, "RandomForestModel::fit"
    );

    if (options.enable_training_metrics) {
        s.training_metrics = evaluate(X, y);
    }
    return s;
}

std::vector<double> RandomForestModel::predict(
    const utils::Matrix& X,
    const PredictOptions& options
) const {
    (void)options;
    return predict_placeholder_mean_regressor_impl(
        fitted_, feature_count_, fallback_mean_target_, X, "RandomForestModel::predict"
    );
}

evaluation::RegressionMetrics RandomForestModel::evaluate(
    const utils::Matrix& X,
    const std::vector<double>& y_true,
    const evaluation::MetricOptions& metric_options,
    const PredictOptions& predict_options
) const {
    return evaluate_placeholder_mean_regressor_impl(
        fitted_, feature_count_, fallback_mean_target_,
        X, y_true, metric_options, predict_options, "RandomForestModel::evaluate"
    );
}

void RandomForestModel::save_model(
    std::string_view path,
    const ModelIOOptions& options
) const {
    save_placeholder_model_kv_impl(
        fitted_, feature_count_, fallback_mean_target_,
        name(), path, options, "RandomForestModel::save_model"
    );
}

void RandomForestModel::load_model(std::string_view path) {
    load_placeholder_model_kv_impl(
        fitted_, feature_count_, fallback_mean_target_,
        name(), path, "RandomForestModel::load_model"
    );
}

// ============================================================
// SVMModel (placeholder mean regressor implementation)
// ============================================================

SVMModel::SVMModel(SVMParams params)
    : params_(std::move(params)) {}

ModelType SVMModel::type() const noexcept {
    return ModelType::SVM;
}

std::string SVMModel::name() const {
    return "SVMModel";
}

bool SVMModel::is_fitted() const noexcept {
    return fitted_;
}

TrainSummary SVMModel::fit(
    const utils::Matrix& X,
    const std::vector<double>& y,
    const FitOptions& options
) {
    auto s = fit_placeholder_mean_regressor_impl(
        fitted_, feature_count_, fallback_mean_target_,
        type(), name(), X, y, "SVMModel::fit"
    );

    if (options.enable_training_metrics) {
        s.training_metrics = evaluate(X, y);
    }
    return s;
}

std::vector<double> SVMModel::predict(
    const utils::Matrix& X,
    const PredictOptions& options
) const {
    (void)options;
    return predict_placeholder_mean_regressor_impl(
        fitted_, feature_count_, fallback_mean_target_, X, "SVMModel::predict"
    );
}

evaluation::RegressionMetrics SVMModel::evaluate(
    const utils::Matrix& X,
    const std::vector<double>& y_true,
    const evaluation::MetricOptions& metric_options,
    const PredictOptions& predict_options
) const {
    return evaluate_placeholder_mean_regressor_impl(
        fitted_, feature_count_, fallback_mean_target_,
        X, y_true, metric_options, predict_options, "SVMModel::evaluate"
    );
}

void SVMModel::save_model(
    std::string_view path,
    const ModelIOOptions& options
) const {
    save_placeholder_model_kv_impl(
        fitted_, feature_count_, fallback_mean_target_,
        name(), path, options, "SVMModel::save_model"
    );
}

void SVMModel::load_model(std::string_view path) {
    load_placeholder_model_kv_impl(
        fitted_, feature_count_, fallback_mean_target_,
        name(), path, "SVMModel::load_model"
    );
}

// ============================================================
// NeuralNetModel (placeholder mean regressor implementation)
// ============================================================

NeuralNetModel::NeuralNetModel(NeuralNetParams params)
    : params_(std::move(params)) {}

ModelType NeuralNetModel::type() const noexcept {
    return ModelType::NeuralNet;
}

std::string NeuralNetModel::name() const {
    return "NeuralNetModel";
}

bool NeuralNetModel::is_fitted() const noexcept {
    return fitted_;
}

TrainSummary NeuralNetModel::fit(
    const utils::Matrix& X,
    const std::vector<double>& y,
    const FitOptions& options
) {
    auto s = fit_placeholder_mean_regressor_impl(
        fitted_, feature_count_, fallback_mean_target_,
        type(), name(), X, y, "NeuralNetModel::fit"
    );

    if (options.enable_training_metrics) {
        s.training_metrics = evaluate(X, y);
    }
    return s;
}

std::vector<double> NeuralNetModel::predict(
    const utils::Matrix& X,
    const PredictOptions& options
) const {
    (void)options;
    return predict_placeholder_mean_regressor_impl(
        fitted_, feature_count_, fallback_mean_target_, X, "NeuralNetModel::predict"
    );
}

evaluation::RegressionMetrics NeuralNetModel::evaluate(
    const utils::Matrix& X,
    const std::vector<double>& y_true,
    const evaluation::MetricOptions& metric_options,
    const PredictOptions& predict_options
) const {
    return evaluate_placeholder_mean_regressor_impl(
        fitted_, feature_count_, fallback_mean_target_,
        X, y_true, metric_options, predict_options, "NeuralNetModel::evaluate"
    );
}

void NeuralNetModel::save_model(
    std::string_view path,
    const ModelIOOptions& options
) const {
    save_placeholder_model_kv_impl(
        fitted_, feature_count_, fallback_mean_target_,
        name(), path, options, "NeuralNetModel::save_model"
    );
}

void NeuralNetModel::load_model(std::string_view path) {
    load_placeholder_model_kv_impl(
        fitted_, feature_count_, fallback_mean_target_,
        name(), path, "NeuralNetModel::load_model"
    );
}

// ============================================================
// Factory
// ============================================================

std::unique_ptr<IRegressionModel> create_model(ModelType type) {
    switch (type) {
        case ModelType::LinearRegression:
            return std::make_unique<LinearRegressionModel>();
        case ModelType::RandomForest:
            return std::make_unique<RandomForestModel>();
        case ModelType::SVM:
            return std::make_unique<SVMModel>();
        case ModelType::NeuralNet:
            return std::make_unique<NeuralNetModel>();
        default:
            return nullptr;
    }
}

} // namespace mof::modeling