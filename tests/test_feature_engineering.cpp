 #include <type_traits>   // add this if missing
 #include "feature_engineering.h"

#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>
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

        auto to_debug_string = [](const auto& v) -> std::string {
            using V = std::decay_t<decltype(v)>;
            std::ostringstream local;
            if constexpr (std::is_enum_v<V>) {
                using U = std::underlying_type_t<V>;
                local << static_cast<U>(v);
            } else {
                local << v;
            }
            return local.str();
        };

        oss << msg
            << " | expected: " << to_debug_string(expected)
            << ", actual: " << to_debug_string(actual);
        throw std::runtime_error(oss.str());
    }
}

void expect_near(double actual, double expected, double eps, const std::string& msg) {
    if (std::fabs(actual - expected) > eps) {
        std::ostringstream oss;
        oss << msg << " | expected: " << expected
            << ", actual: " << actual
            << ", abs diff: " << std::fabs(actual - expected)
            << ", eps: " << eps;
        fail(oss.str());
    }
}

void expect_all_finite(const std::vector<std::vector<double>>& mat, const std::string& msg_prefix) {
    for (std::size_t r = 0; r < mat.size(); ++r) {
        for (std::size_t c = 0; c < mat[r].size(); ++c) {
            const double v = mat[r][c];
            if (!std::isfinite(v) || std::isnan(v)) {
                std::ostringstream oss;
                oss << msg_prefix << " | non-finite value at (" << r << "," << c << "): " << v;
                fail(oss.str());
            }
        }
    }
}

mof::preprocessing::NumericDataset make_dataset(
    std::vector<std::string> feature_names,
    std::vector<std::vector<double>> features,
    std::vector<double> target = {}
) {
    mof::preprocessing::NumericDataset ds;
    ds.feature_names = std::move(feature_names);
    ds.features = std::move(features);
    ds.target = std::move(target);
    ds.target_name = ds.target.empty() ? "" : "target";
    return ds;
}

void test_feature_count_matches_expectation() {
    using namespace mof::feature_engineering;

    auto ds = make_dataset(
        {"f1", "f2", "f3"},
        {
            {1.0, 2.0, 3.0},
            {2.0, 4.0, 5.0},
            {3.0, 6.0, 7.0}
        },
        {10.0, 20.0, 30.0}
    );

    DerivedFeatureOptions opt;
    opt.add_square = true;              // +3
    opt.add_pairwise_product = true;    // +3 (f1*f2, f1*f3, f2*f3)
    opt.pairwise_upper_triangle_only = true;
    opt.max_pairwise_features = 0;      // unlimited
    opt.eps = 1e-12;

    DerivedFeatureSummary summary;
    auto out = create_derived_features(ds, opt, &summary);

    // Expected: original 3 + 3 square + 3 pairwise product = 9
    // (chosen values avoid constant/near-constant removal)
    expect_eq(summary.original_feature_count, static_cast<std::size_t>(3), "Original feature count mismatch");
    expect_eq(summary.final_feature_count, static_cast<std::size_t>(9), "Final feature count mismatch");
    expect_eq(summary.added_feature_count, static_cast<std::size_t>(6), "Added feature count mismatch");

    expect_eq(out.feature_names.size(), static_cast<std::size_t>(9), "Output feature name count mismatch");
    expect_eq(out.features.size(), static_cast<std::size_t>(3), "Output row count mismatch");
    expect_eq(out.features[0].size(), static_cast<std::size_t>(9), "Output feature width mismatch");

    // Target should be preserved
    expect_eq(out.target.size(), static_cast<std::size_t>(3), "Target size should be preserved");
    expect_eq(out.target_name, std::string("target"), "Target name should be preserved");
}

void test_derived_feature_values_correct() {
    using namespace mof::feature_engineering;

    auto ds = make_dataset(
        {"x1", "x2"},
        {
            {2.0, 3.0},
            {4.0, 5.0}
        }
    );

    DerivedFeatureOptions opt;
    opt.add_square = true;           // adds x1__sq, x2__sq
    opt.add_pairwise_product = true; // adds x1__mul__x2
    opt.eps = 1e-12;

    DerivedFeatureSummary summary;
    auto out = create_derived_features(ds, opt, &summary);

    expect_eq(summary.original_feature_count, static_cast<std::size_t>(2), "Original feature count mismatch");
    expect_eq(summary.final_feature_count, static_cast<std::size_t>(5), "Final feature count mismatch");
    expect_eq(out.features[0].size(), static_cast<std::size_t>(5), "Feature width mismatch");

    // Column order from implementation:
    // original: x1, x2
    // unary squares: x1__sq, x2__sq
    // pairwise product (upper triangle): x1__mul__x2
    expect_eq(out.feature_names[0], std::string("x1"), "Feature name[0] mismatch");
    expect_eq(out.feature_names[1], std::string("x2"), "Feature name[1] mismatch");
    expect_eq(out.feature_names[2], std::string("x1__sq"), "Feature name[2] mismatch");
    expect_eq(out.feature_names[3], std::string("x2__sq"), "Feature name[3] mismatch");
    expect_eq(out.feature_names[4], std::string("x1__mul__x2"), "Feature name[4] mismatch");

    // Row 0: [2,3,4,9,6]
    expect_near(out.features[0][0], 2.0, kEps, "row0 x1");
    expect_near(out.features[0][1], 3.0, kEps, "row0 x2");
    expect_near(out.features[0][2], 4.0, kEps, "row0 x1^2");
    expect_near(out.features[0][3], 9.0, kEps, "row0 x2^2");
    expect_near(out.features[0][4], 6.0, kEps, "row0 x1*x2");

    // Row 1: [4,5,16,25,20]
    expect_near(out.features[1][0], 4.0, kEps, "row1 x1");
    expect_near(out.features[1][1], 5.0, kEps, "row1 x2");
    expect_near(out.features[1][2], 16.0, kEps, "row1 x1^2");
    expect_near(out.features[1][3], 25.0, kEps, "row1 x2^2");
    expect_near(out.features[1][4], 20.0, kEps, "row1 x1*x2");
}

void test_constant_feature_removal_works() {
    using namespace mof::feature_engineering;

    auto ds = make_dataset(
        {"varying", "constant"},
        {
            {1.0, 5.0},
            {2.0, 5.0},
            {3.0, 5.0}
        }
    );

    // No derived features requested; create_derived_features still runs internal
    // constant/near-constant filtering.
    DerivedFeatureOptions opt;
    opt.eps = 1e-12;

    DerivedFeatureSummary summary;
    auto out = create_derived_features(ds, opt, &summary);

    expect_eq(summary.original_feature_count, static_cast<std::size_t>(2), "Original feature count mismatch");
    expect_eq(summary.final_feature_count, static_cast<std::size_t>(1), "Constant feature should be removed");

    expect_eq(out.feature_names.size(), static_cast<std::size_t>(1), "Feature names count after constant removal");
    expect_eq(out.feature_names[0], std::string("varying"), "Wrong feature kept after constant removal");

    expect_eq(out.features.size(), static_cast<std::size_t>(3), "Row count should be preserved");
    for (std::size_t r = 0; r < out.features.size(); ++r) {
        expect_eq(out.features[r].size(), static_cast<std::size_t>(1), "Output width should be 1 after removal");
        expect_near(out.features[r][0], static_cast<double>(r + 1), kEps, "Remaining feature value mismatch");
    }
}

void test_no_nan_in_output_features() {
    using namespace mof::feature_engineering;

    // Include a constant feature to stress z-score scaling path (should not produce NaN).
    auto ds = make_dataset(
        {"const_f", "var_f"},
        {
            {5.0, 1.0},
            {5.0, 2.0},
            {5.0, 3.0},
            {5.0, 4.0}
        }
    );

    // Apply feature scaling directly (uses preprocessing normalization hooks internally).
    FeatureScalingOptions sopt;
    sopt.method = FeatureScalingMethod::ZScore;
    sopt.eps = 1e-12;

    auto artifacts = fit_and_apply_feature_scaling(ds, sopt);

    expect_eq(artifacts.method, FeatureScalingMethod::ZScore, "Scaling method artifact mismatch");
    expect_eq(artifacts.feature_stats.size(), static_cast<std::size_t>(2), "Feature stats size mismatch");
    expect_eq(artifacts.scaled_columns.size(), static_cast<std::size_t>(2), "Scaled columns size mismatch");

    expect_all_finite(ds.features, "Scaled feature matrix should contain no NaN/Inf");

    // Constant column should become zeros after safe z-score scaling
    for (std::size_t r = 0; r < ds.features.size(); ++r) {
        expect_near(ds.features[r][0], 0.0, 1e-9, "Constant z-score feature should be zero");
    }

    // Also test no NaN after derived features + scaling pipeline
    auto ds2 = make_dataset(
        {"a", "b"},
        {
            {1.0, 2.0},
            {2.0, 4.0},
            {3.0, 8.0}
        }
    );

    DerivedFeatureOptions dopt;
    dopt.add_square = true;
    dopt.add_pairwise_difference = true;
    dopt.add_pairwise_sum = true;
    dopt.eps = 1e-12;

    auto engineered = create_derived_features(ds2, dopt, nullptr);

    FeatureScalingOptions sopt2;
    sopt2.method = FeatureScalingMethod::MinMax;
    sopt2.min_out = 0.0;
    sopt2.max_out = 1.0;

    (void)fit_and_apply_feature_scaling(engineered, sopt2);
    expect_all_finite(engineered.features, "Engineered+scaled matrix should contain no NaN/Inf");
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

    run_test("feature count matches expectation", test_feature_count_matches_expectation, stats);
    run_test("derived feature values correct", test_derived_feature_values_correct, stats);
    run_test("constant feature removal works", test_constant_feature_removal_works, stats);
    run_test("no NaN in output features", test_no_nan_in_output_features, stats);

    std::cout << "\nTest summary: " << stats.passed << " passed, "
              << stats.failed << " failed.\n";

    return (stats.failed == 0) ? 0 : 1;
}