#include <memory> // workaround if include/modeling.h does not yet include <memory>

#include "modeling.h"

#include <cmath>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

constexpr double kEps = 1e-8;

struct TestStats {
    int passed = 0;
    int failed = 0;
};

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error(msg);
}

void expect_true(bool cond, const std::string& msg) {
    if (!cond) {
        fail(msg);
    }
}

// --- FIX: declare this helper at namespace scope, BEFORE expect_eq ---
template <typename T>
std::string debug_value_to_string(const T& value) {
    using Decayed = std::decay_t<T>;

    if constexpr (std::is_same_v<Decayed, std::string>) {
        return value;
    } else if constexpr (std::is_same_v<Decayed, const char*>) {
        return value ? std::string(value) : std::string("<null>");
    } else if constexpr (std::is_same_v<Decayed, char*>) {
        return value ? std::string(value) : std::string("<null>");
    } else if constexpr (std::is_enum_v<Decayed>) {
        using U = std::underlying_type_t<Decayed>;
        return std::to_string(static_cast<U>(value));
    } else {
        std::ostringstream oss;
        oss << value;
        return oss.str();
    }
}

template <typename T>
void expect_eq(const T& actual, const T& expected, const std::string& msg) {
    if (!(actual == expected)) {
        std::ostringstream oss;
        oss << msg
            << " | expected: " << debug_value_to_string(expected)
            << ", actual: " << debug_value_to_string(actual);
        fail(oss.str());
    }
}

void expect_near(double actual, double expected, double eps, const std::string& msg) {
    if (std::fabs(actual - expected) > eps) {
        std::ostringstream oss;
        oss << msg
            << " | expected: " << expected
            << ", actual: " << actual
            << ", diff: " << std::fabs(actual - expected)
            << ", eps: " << eps;
        fail(oss.str());
    }
}

void expect_all_finite(const std::vector<double>& v, const std::string& msg_prefix) {
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (!std::isfinite(v[i])) {
            std::ostringstream oss;
            oss << msg_prefix << " | non-finite value at index " << i << ": " << v[i];
            fail(oss.str());
        }
    }
}

mof::utils::Matrix make_matrix(std::initializer_list<std::initializer_list<double>> rows) {
    mof::utils::Matrix X;
    X.reserve(rows.size());
    for (const auto& row : rows) {
        X.emplace_back(row);
    }
    return X;
}

void test_model_trains_on_tiny_synthetic_data() {
    using namespace mof::modeling;

    // Exact linear relation:
    // y = 2*x1 - 3*x2 + 5
    const auto X = make_matrix({
        {1.0, 2.0},  // y =  1
        {2.0, 1.0},  // y =  6
        {3.0, 0.0},  // y = 11
        {0.0, 3.0},  // y = -4
        {4.0, 5.0},  // y = -2
        {5.0, 4.0}   // y =  3
    });

    const std::vector<double> y = {1.0, 6.0, 11.0, -4.0, -2.0, 3.0};

    LinearRegressionModel model;
    FitOptions fit_opt;
    fit_opt.enable_training_metrics = true;

    const auto summary = model.fit(X, y, fit_opt);

    expect_true(model.is_fitted(), "Model should be fitted after fit()");
    expect_true(summary.fitted, "TrainSummary.fitted should be true");

    expect_eq(summary.model_type, ModelType::LinearRegression, "TrainSummary model_type mismatch");
    expect_eq(summary.model_name, std::string("LinearRegressionModel"), "TrainSummary model_name mismatch");
    expect_eq(summary.train_rows, static_cast<std::size_t>(6), "TrainSummary train_rows mismatch");
    expect_eq(summary.feature_count, static_cast<std::size_t>(2), "TrainSummary feature_count mismatch");

    expect_true(summary.training_metrics.has_value(), "Training metrics should be present when enabled");
    expect_near(summary.training_metrics->rmse, 0.0, 1e-7, "Training RMSE should be near zero on exact linear data");
    expect_near(summary.training_metrics->mae, 0.0, 1e-7, "Training MAE should be near zero on exact linear data");
    expect_near(summary.training_metrics->r2, 1.0, 1e-7, "Training R2 should be near one on exact linear data");

    // Optional coefficient sanity (implementation-specific but should be stable here)
    const auto& coef = model.coefficients();
    expect_eq(coef.size(), static_cast<std::size_t>(2), "Coefficient count mismatch");
    expect_near(coef[0], 2.0, 1e-7, "Coefficient[0] mismatch");
    expect_near(coef[1], -3.0, 1e-7, "Coefficient[1] mismatch");
    expect_near(model.intercept(), 5.0, 1e-7, "Intercept mismatch");
}

void test_prediction_shape_is_correct() {
    using namespace mof::modeling;

    // Train on simple 2-feature exact line
    const auto X_train = make_matrix({
        {0.0, 0.0},
        {1.0, 0.0},
        {0.0, 1.0},
        {2.0, 1.0},
        {1.0, 2.0}
    });

    // y = x1 + 2*x2 + 1
    const std::vector<double> y_train = {
        1.0,  // 0 + 0 + 1
        2.0,  // 1 + 0 + 1
        3.0,  // 0 + 2 + 1
        5.0,  // 2 + 2 + 1
        6.0   // 1 + 4 + 1
    };

    LinearRegressionModel model;
    (void)model.fit(X_train, y_train);

    const auto X_test = make_matrix({
        {3.0, 0.0},
        {0.0, 3.0},
        {2.0, 2.0},
        {10.0, -1.0}
    });

    const auto pred = model.predict(X_test);

    expect_eq(pred.size(), static_cast<std::size_t>(4), "Prediction vector size should match X_test rows");
    expect_all_finite(pred, "Predictions must be finite");

    // Spot-check values from y = x1 + 2*x2 + 1
    expect_near(pred[0], 4.0, 1e-7, "Prediction[0] mismatch");
    expect_near(pred[1], 7.0, 1e-7, "Prediction[1] mismatch");
    expect_near(pred[2], 7.0, 1e-7, "Prediction[2] mismatch");
    expect_near(pred[3], 9.0, 1e-7, "Prediction[3] mismatch");
}

void test_basic_overfit_on_tiny_data_sanity_check() {
    using namespace mof::modeling;

    // Tiny exact dataset should be overfit almost perfectly by linear regression.
    // y = 4*x + 1
    const auto X = make_matrix({
        {0.0},
        {1.0},
        {2.0},
        {3.0},
        {4.0}
    });
    const std::vector<double> y = {1.0, 5.0, 9.0, 13.0, 17.0};

    LinearRegressionModel model;
    (void)model.fit(X, y);

    const auto y_pred = model.predict(X);
    expect_eq(y_pred.size(), y.size(), "Prediction size mismatch on training data");
    expect_all_finite(y_pred, "Training predictions must be finite");

    const auto metrics = model.evaluate(X, y);

    // Sanity check: should fit tiny exact linear data nearly perfectly
    expect_near(metrics.rmse, 0.0, 1e-8, "Overfit sanity RMSE should be ~0");
    expect_near(metrics.mae, 0.0, 1e-8, "Overfit sanity MAE should be ~0");
    expect_near(metrics.r2, 1.0, 1e-8, "Overfit sanity R2 should be ~1");

    // Direct prediction equality (within tolerance)
    for (std::size_t i = 0; i < y.size(); ++i) {
        expect_near(y_pred[i], y[i], 1e-8, "Overfit sanity prediction mismatch");
    }
}

void run_test(const std::string& name, const std::function<void()>& fn, TestStats& stats) {
    try {
        fn();
        ++stats.passed;
        std::cout << "[PASS] " << name << '\n';
    } catch (const std::exception& e) {
        ++stats.failed;
        std::cerr << "[FAIL] " << name << " -> " << e.what() << '\n';
    } catch (...) {
        ++stats.failed;
        std::cerr << "[FAIL] " << name << " -> unknown exception\n";
    }
}

} // namespace

int main() {
    TestStats stats{};

    run_test("model trains on tiny synthetic data", test_model_trains_on_tiny_synthetic_data, stats);
    run_test("prediction shape is correct", test_prediction_shape_is_correct, stats);
    run_test("basic overfit-on-tiny-data sanity check", test_basic_overfit_on_tiny_data_sanity_check, stats);

    std::cout << "\nTest summary: " << stats.passed << " passed, "
              << stats.failed << " failed.\n";

    return (stats.failed == 0) ? 0 : 1;
}