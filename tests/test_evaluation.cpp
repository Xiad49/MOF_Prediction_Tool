 #include "evaluation.h"

#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kEps = 1e-9;

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

template <typename T>
void expect_eq(const T& actual, const T& expected, const std::string& msg) {
    if (!(actual == expected)) {
        std::ostringstream oss;
        oss << msg << " | expected: " << expected << ", actual: " << actual;
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

template <typename ExceptionT, typename Fn>
void expect_throw(Fn&& fn, const std::string& msg) {
    try {
        fn();
    } catch (const ExceptionT&) {
        return;
    } catch (const std::exception& e) {
        std::ostringstream oss;
        oss << msg << " | threw unexpected exception type: " << e.what();
        fail(oss.str());
    } catch (...) {
        fail(msg + " | threw non-std::exception");
    }
    fail(msg + " | expected exception was not thrown");
}

void test_known_metric_values_manual_verification() {
    using namespace mof::evaluation;

    // Manually verifiable example:
    // y_true = [1,2,3,4]
    // y_pred = [1,2,2,5]
    // errors = [0,0,1,-1] in absolute terms [0,0,1,1]
    //
    // MAE  = (0+0+1+1)/4 = 0.5
    // RMSE = sqrt((0^2+0^2+1^2+1^2)/4) = sqrt(0.5)
    //
    // mean(y_true)=2.5
    // SS_tot = (1-2.5)^2 + (2-2.5)^2 + (3-2.5)^2 + (4-2.5)^2 = 5
    // SS_res = 2
    // R² = 1 - 2/5 = 0.6
    //
    // MAPE = average([0/1, 0/2, 1/3, 1/4]) * 100
    //      = ((0 + 0 + 0.3333333333 + 0.25) / 4) * 100
    //      = 14.5833333333 %

    const std::vector<double> y_true{1.0, 2.0, 3.0, 4.0};
    const std::vector<double> y_pred{1.0, 2.0, 2.0, 5.0};

    expect_near(rmse(y_true, y_pred), std::sqrt(0.5), 1e-12, "RMSE mismatch");
    expect_near(mae(y_true, y_pred), 0.5, 1e-12, "MAE mismatch");
    expect_near(r2_score(y_true, y_pred), 0.6, 1e-12, "R2 mismatch");

    const auto m = mape(y_true, y_pred);
    expect_true(m.valid, "MAPE should be valid");
    expect_eq(m.used_count, static_cast<std::size_t>(4), "MAPE used_count mismatch");
    expect_eq(m.skipped_count, static_cast<std::size_t>(0), "MAPE skipped_count mismatch");
    expect_near(m.value, 14.583333333333334, 1e-12, "MAPE mismatch");

    const auto all = evaluate_regression(y_true, y_pred);
    expect_eq(all.sample_count, static_cast<std::size_t>(4), "evaluate_regression sample_count mismatch");
    expect_near(all.rmse, std::sqrt(0.5), 1e-12, "evaluate_regression RMSE mismatch");
    expect_near(all.mae, 0.5, 1e-12, "evaluate_regression MAE mismatch");
    expect_near(all.r2, 0.6, 1e-12, "evaluate_regression R2 mismatch");
    expect_true(all.has_valid_r2, "R2 should be marked valid");
    expect_true(all.has_valid_mape, "MAPE should be marked valid");
    expect_near(all.mape.value, 14.583333333333334, 1e-12, "evaluate_regression MAPE mismatch");
}

void test_mape_near_zero_handling() {
    using namespace mof::evaluation;

    const std::vector<double> y_true{0.0, 1.0, 2.0};
    const std::vector<double> y_pred{1.0, 1.0, 1.0};

    // Default behavior: skip near-zero targets for MAPE
    {
        MetricOptions opt;
        opt.eps = 1e-12;
        opt.ignore_near_zero_targets_for_mape = true;

        const auto m = mape(y_true, y_pred, opt);

        // Used samples: indices 1 and 2
        // abs pct errors: |1-1|/1 *100 = 0%
        //                 |2-1|/2 *100 = 50%
        // average = 25%
        expect_true(m.valid, "MAPE should be valid when at least one sample is usable");
        expect_eq(m.used_count, static_cast<std::size_t>(2), "MAPE used_count mismatch with skip");
        expect_eq(m.skipped_count, static_cast<std::size_t>(1), "MAPE skipped_count mismatch with skip");
        expect_near(m.value, 25.0, 1e-12, "MAPE value mismatch with skip");
    }

    // Strict behavior: throw on near-zero target
    {
        MetricOptions opt;
        opt.eps = 1e-12;
        opt.ignore_near_zero_targets_for_mape = false;

        expect_throw<std::domain_error>(
            [&]() { (void)mape(y_true, y_pred, opt); },
            "MAPE should throw when near-zero target exists and skipping is disabled"
        );
    }

    // All near-zero targets -> invalid MAPE
    {
        const std::vector<double> yt{0.0, 0.0};
        const std::vector<double> yp{1.0, 2.0};

        MetricOptions opt;
        opt.ignore_near_zero_targets_for_mape = true;

        const auto m = mape(yt, yp, opt);
        expect_true(!m.valid, "MAPE should be invalid when no samples are usable");
        expect_eq(m.used_count, static_cast<std::size_t>(0), "MAPE used_count should be 0");
        expect_eq(m.skipped_count, static_cast<std::size_t>(2), "MAPE skipped_count should be 2");
        expect_near(m.value, 0.0, 1e-12, "MAPE invalid-case value should be 0");
    }
}

void test_r2_constant_target_behavior() {
    using namespace mof::evaluation;

    const std::vector<double> y_true{5.0, 5.0, 5.0};
    const std::vector<double> y_pred{4.0, 5.0, 6.0};

    // Default fallback behavior (header defaults to fallback=0.0)
    {
        MetricOptions opt;
        opt.allow_undefined_r2_fallback = true;
        opt.fallback_r2_when_undefined = 0.0;

        const double r2 = r2_score(y_true, y_pred, opt);
        expect_near(r2, 0.0, 1e-12, "R2 fallback value mismatch");

        const auto metrics = evaluate_regression(y_true, y_pred, opt);
        expect_true(!metrics.has_valid_r2, "R2 should be marked invalid when fallback is used");
        expect_near(metrics.r2, 0.0, 1e-12, "evaluate_regression R2 fallback mismatch");
    }

    // Custom fallback value
    {
        MetricOptions opt;
        opt.allow_undefined_r2_fallback = true;
        opt.fallback_r2_when_undefined = -1.0;

        const double r2 = r2_score(y_true, y_pred, opt);
        expect_near(r2, -1.0, 1e-12, "Custom R2 fallback mismatch");
    }

    // Strict behavior: throw when undefined
    {
        MetricOptions opt;
        opt.allow_undefined_r2_fallback = false;

        expect_throw<std::domain_error>(
            [&]() { (void)r2_score(y_true, y_pred, opt); },
            "R2 should throw for constant y_true when fallback is disabled"
        );
    }
}

void test_validation_errors_and_non_finite_inputs() {
    using namespace mof::evaluation;

    // is_valid_prediction_pair basics
    expect_true(!is_valid_prediction_pair({}, {}), "Empty vectors should be invalid");
    expect_true(!is_valid_prediction_pair({1.0}, {1.0, 2.0}), "Mismatched size should be invalid");
    expect_true(is_valid_prediction_pair({1.0, 2.0}, {1.0, 2.0}), "Same-size non-empty vectors should be valid");

    // Empty inputs
    expect_throw<std::invalid_argument>(
        [&]() { validate_prediction_pair({}, {}, "unit_test"); },
        "validate_prediction_pair should throw on empty inputs"
    );

    // Size mismatch
    expect_throw<std::invalid_argument>(
        [&]() { (void)rmse(std::vector<double>{1.0}, std::vector<double>{1.0, 2.0}); },
        "RMSE should throw on size mismatch"
    );

    // Non-finite values (NaN / Inf)
    {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        const double inf = std::numeric_limits<double>::infinity();

        expect_throw<std::invalid_argument>(
            [&]() { (void)mae(std::vector<double>{1.0, nan}, std::vector<double>{1.0, 2.0}); },
            "MAE should throw on NaN in y_true"
        );

        expect_throw<std::invalid_argument>(
            [&]() { (void)r2_score(std::vector<double>{1.0, 2.0}, std::vector<double>{1.0, inf}); },
            "R2 should throw on Inf in y_pred"
        );
    }

    // Invalid eps in options
    {
        MetricOptions opt;
        opt.eps = 0.0;

        expect_throw<std::invalid_argument>(
            [&]() { (void)mape(std::vector<double>{1.0}, std::vector<double>{1.0}, opt); },
            "MAPE should throw when eps <= 0"
        );

        expect_throw<std::invalid_argument>(
            [&]() { (void)r2_score(std::vector<double>{1.0}, std::vector<double>{1.0}, opt); },
            "R2 should throw when eps <= 0"
        );
    }
}

void test_format_metrics_output() {
    using namespace mof::evaluation;

    const std::vector<double> y_true{1.0, 2.0, 3.0};
    const std::vector<double> y_pred{1.0, 2.0, 4.0};

    const auto metrics = evaluate_regression(y_true, y_pred);
    const std::string text = format_metrics(metrics, 6);

    expect_true(text.find("n=3") != std::string::npos, "format_metrics should include sample count");
    expect_true(text.find("RMSE=") != std::string::npos, "format_metrics should include RMSE");
    expect_true(text.find("MAE=") != std::string::npos, "format_metrics should include MAE");
    expect_true(text.find("R2=") != std::string::npos, "format_metrics should include R2");
    expect_true(text.find("MAPE=") != std::string::npos, "format_metrics should include MAPE");
    expect_true(text.find("%") != std::string::npos, "format_metrics should include percent sign for MAPE");
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

    run_test("known metric values (manual verification)", test_known_metric_values_manual_verification, stats);
    run_test("MAPE near-zero handling", test_mape_near_zero_handling, stats);
    run_test("R2 constant-target behavior", test_r2_constant_target_behavior, stats);
    run_test("validation errors and non-finite inputs", test_validation_errors_and_non_finite_inputs, stats);
    run_test("format_metrics output", test_format_metrics_output, stats);

    std::cout << "\nTest summary: " << stats.passed << " passed, "
              << stats.failed << " failed.\n";

    return (stats.failed == 0) ? 0 : 1;
}