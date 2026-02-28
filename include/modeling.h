 #pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "evaluation.h"
#include "utils.h"

namespace mof::modeling {

// ============================================================
// Shared types and options
// ============================================================

enum class ModelType {
    LinearRegression,
    RandomForest,
    SVM,
    NeuralNet
};

struct FitOptions {
    // Common training controls (model-specific implementations may ignore some)
    bool fit_intercept = true;
    bool shuffle = true;
    std::optional<std::uint32_t> random_seed = 42;

    // Iterative solvers / future models (SVM / NN)
    std::size_t max_iterations = 1000;
    double learning_rate = 1e-3;
    double tolerance = 1e-8;

    // Validation / diagnostics hooks (optional; actual use later)
    bool enable_training_metrics = false;
};

struct PredictOptions {
    // Placeholder for future options (batching, clipping, etc.)
    bool clip_non_finite_outputs = false;
};

struct TrainSummary {
    ModelType model_type = ModelType::LinearRegression;
    std::string model_name;
    std::size_t train_rows = 0;
    std::size_t feature_count = 0;
    bool fitted = false;

    // Optional training-set metrics (if computed by implementation)
    std::optional<evaluation::RegressionMetrics> training_metrics;
};

struct ModelIOOptions {
    bool overwrite = true;
};

// ============================================================
// Hyperparameter structs (MVP + future-ready placeholders)
// ============================================================

struct LinearRegressionParams {
    bool fit_intercept = true;

    // Optional regularization placeholders for future extension
    double l2_regularization = 0.0;  // Ridge-like (0 => disabled)
};

struct RandomForestParams {
    std::size_t n_trees = 100;
    std::size_t max_depth = 10;                // 0 => unlimited (implementation-defined)
    std::size_t min_samples_split = 2;
    std::size_t min_samples_leaf = 1;
    std::size_t max_features = 0;              // 0 => auto
    bool bootstrap = true;
    std::optional<std::uint32_t> random_seed = 42;
};

struct SVMParams {
    enum class Kernel {
        Linear,
        RBF,
        Polynomial
    };

    Kernel kernel = Kernel::RBF;
    double c = 1.0;                // regularization strength
    double epsilon = 0.1;          // epsilon-insensitive regression loss
    double gamma = 0.1;            // for RBF / polynomial
    int degree = 3;                // polynomial degree
    double coef0 = 0.0;            // polynomial/rbf variants if needed later
    std::size_t max_iterations = 1000;
    double tolerance = 1e-6;
};

struct NeuralNetParams {
    // Placeholder for later implementation
    std::vector<std::size_t> hidden_layers{64, 32};
    std::string activation = "relu";
    double learning_rate = 1e-3;
    std::size_t epochs = 50;
    std::size_t batch_size = 32;
    double weight_decay = 0.0;
    std::optional<std::uint32_t> random_seed = 42;
};

// ============================================================
// Common model interface
// ============================================================

class IRegressionModel {
public:
    virtual ~IRegressionModel() = default;

    // Metadata
    virtual ModelType type() const noexcept = 0;
    virtual std::string name() const = 0;
    virtual bool is_fitted() const noexcept = 0;

    // Core API
    virtual TrainSummary fit(
        const utils::Matrix& X,
        const std::vector<double>& y,
        const FitOptions& options = {}
    ) = 0;

    virtual std::vector<double> predict(
        const utils::Matrix& X,
        const PredictOptions& options = {}
    ) const = 0;

    // Convenience evaluation helper (implemented in src/modeling.cpp)
    virtual evaluation::RegressionMetrics evaluate(
        const utils::Matrix& X,
        const std::vector<double>& y_true,
        const evaluation::MetricOptions& metric_options = {},
        const PredictOptions& predict_options = {}
    ) const = 0;

    // Optional persistence hooks (can throw/not-implement for now)
    virtual void save_model(
        std::string_view path,
        const ModelIOOptions& options = {}
    ) const = 0;

    virtual void load_model(std::string_view path) = 0;
};

// ============================================================
// Linear Regression
// ============================================================

class LinearRegressionModel final : public IRegressionModel {
public:
    explicit LinearRegressionModel(LinearRegressionParams params = {});
    ~LinearRegressionModel() override = default;

    // Metadata
    ModelType type() const noexcept override;
    std::string name() const override;
    bool is_fitted() const noexcept override;

    // Core API
    TrainSummary fit(
        const utils::Matrix& X,
        const std::vector<double>& y,
        const FitOptions& options = {}
    ) override;

    std::vector<double> predict(
        const utils::Matrix& X,
        const PredictOptions& options = {}
    ) const override;

    evaluation::RegressionMetrics evaluate(
        const utils::Matrix& X,
        const std::vector<double>& y_true,
        const evaluation::MetricOptions& metric_options = {},
        const PredictOptions& predict_options = {}
    ) const override;

    // Persistence (optional)
    void save_model(
        std::string_view path,
        const ModelIOOptions& options = {}
    ) const override;

    void load_model(std::string_view path) override;

    // Accessors
    const LinearRegressionParams& params() const noexcept { return params_; }
    const std::vector<double>& coefficients() const noexcept { return coefficients_; }
    double intercept() const noexcept { return intercept_; }

private:
    LinearRegressionParams params_{};
    bool fitted_ = false;

    // Model state
    std::vector<double> coefficients_{};
    double intercept_ = 0.0;

    // Training metadata
    std::size_t feature_count_ = 0;
};

// ============================================================
// Random Forest Regressor (interface + placeholder state)
// ============================================================

class RandomForestModel final : public IRegressionModel {
public:
    explicit RandomForestModel(RandomForestParams params = {});
    ~RandomForestModel() override = default;

    ModelType type() const noexcept override;
    std::string name() const override;
    bool is_fitted() const noexcept override;

    TrainSummary fit(
        const utils::Matrix& X,
        const std::vector<double>& y,
        const FitOptions& options = {}
    ) override;

    std::vector<double> predict(
        const utils::Matrix& X,
        const PredictOptions& options = {}
    ) const override;

    evaluation::RegressionMetrics evaluate(
        const utils::Matrix& X,
        const std::vector<double>& y_true,
        const evaluation::MetricOptions& metric_options = {},
        const PredictOptions& predict_options = {}
    ) const override;

    void save_model(
        std::string_view path,
        const ModelIOOptions& options = {}
    ) const override;

    void load_model(std::string_view path) override;

    const RandomForestParams& params() const noexcept { return params_; }

private:
    RandomForestParams params_{};
    bool fitted_ = false;
    std::size_t feature_count_ = 0;

    // Placeholder state until implementation (trees, splits, etc.)
    double fallback_mean_target_ = 0.0;
};

// ============================================================
// SVM Regressor (interface + placeholder state)
// ============================================================

class SVMModel final : public IRegressionModel {
public:
    explicit SVMModel(SVMParams params = {});
    ~SVMModel() override = default;

    ModelType type() const noexcept override;
    std::string name() const override;
    bool is_fitted() const noexcept override;

    TrainSummary fit(
        const utils::Matrix& X,
        const std::vector<double>& y,
        const FitOptions& options = {}
    ) override;

    std::vector<double> predict(
        const utils::Matrix& X,
        const PredictOptions& options = {}
    ) const override;

    evaluation::RegressionMetrics evaluate(
        const utils::Matrix& X,
        const std::vector<double>& y_true,
        const evaluation::MetricOptions& metric_options = {},
        const PredictOptions& predict_options = {}
    ) const override;

    void save_model(
        std::string_view path,
        const ModelIOOptions& options = {}
    ) const override;

    void load_model(std::string_view path) override;

    const SVMParams& params() const noexcept { return params_; }

private:
    SVMParams params_{};
    bool fitted_ = false;
    std::size_t feature_count_ = 0;

    // Placeholder state until implementation
    double fallback_mean_target_ = 0.0;
};

// ============================================================
// Neural Net Regressor (later; interface now)
// ============================================================

class NeuralNetModel final : public IRegressionModel {
public:
    explicit NeuralNetModel(NeuralNetParams params = {});
    ~NeuralNetModel() override = default;

    ModelType type() const noexcept override;
    std::string name() const override;
    bool is_fitted() const noexcept override;

    TrainSummary fit(
        const utils::Matrix& X,
        const std::vector<double>& y,
        const FitOptions& options = {}
    ) override;

    std::vector<double> predict(
        const utils::Matrix& X,
        const PredictOptions& options = {}
    ) const override;

    evaluation::RegressionMetrics evaluate(
        const utils::Matrix& X,
        const std::vector<double>& y_true,
        const evaluation::MetricOptions& metric_options = {},
        const PredictOptions& predict_options = {}
    ) const override;

    void save_model(
        std::string_view path,
        const ModelIOOptions& options = {}
    ) const override;

    void load_model(std::string_view path) override;

    const NeuralNetParams& params() const noexcept { return params_; }

private:
    NeuralNetParams params_{};
    bool fitted_ = false;
    std::size_t feature_count_ = 0;

    // Placeholder state until later implementation
    double fallback_mean_target_ = 0.0;
};

// ============================================================
// Factory / utility helpers
// ============================================================

/// Lightweight validation utility for model inputs (throws on mismatch/empty/non-rectangular X).
void validate_training_data(
    const utils::Matrix& X,
    const std::vector<double>& y,
    std::string_view caller = "modeling"
);

/// Validation for prediction-only input (throws on empty/non-rectangular X).
void validate_feature_matrix(
    const utils::Matrix& X,
    std::string_view caller = "modeling"
);

/// Create a model instance by type (useful for CLI/config-driven workflows).
/// Implementations may return nullptr only on unsupported type (not recommended).
std::unique_ptr<IRegressionModel> create_model(ModelType type);

} // namespace mof::modeling