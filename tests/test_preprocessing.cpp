#include "preprocessing.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

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
        oss << msg << " | expected: " << expected << ", actual: " << actual
            << ", abs diff: " << std::fabs(actual - expected)
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

fs::path make_test_temp_dir() {
    const auto now_ticks = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    const auto unique = std::to_string(static_cast<long long>(now_ticks));

    fs::path dir = fs::temp_directory_path() / "mof_preprocessing_tests" / unique;
    fs::create_directories(dir);
    return dir;
}

void write_text_file(const fs::path& path, const std::string& content) {
    fs::create_directories(path.parent_path());
    std::ofstream ofs(path);
    if (!ofs) {
        throw std::runtime_error("Failed to create file: " + path.string());
    }
    ofs << content;
    if (!ofs.good()) {
        throw std::runtime_error("Failed to write file: " + path.string());
    }
}

double as_double(const std::string& s) {
    return std::stod(s);
}

void test_missing_value_handling_mean_and_median() {
    using namespace mof::preprocessing;

    // Header + rows with missing values in numeric column "f1"
    TabularData table;
    table.column_names = {"id", "f1", "target"};
    table.rows = {
        {"a", "1",   "10"},
        {"b", "",    "20"},  // missing in f1
        {"c", "100", "30"},
        {"d", "2",   "40"}
    };

    // Mean imputation on column 1 => mean(1,100,2) = 34.333333...
    {
        TabularData t = table;
        MissingValueOptions opt;
        opt.strategy = MissingValueStrategy::FillWithMean;
        opt.column_indices = {1};

        const std::size_t modified = handle_missing_values(t, opt);
        expect_eq(modified, static_cast<std::size_t>(1), "Mean imputation should modify 1 cell");

        const double filled = as_double(t.rows[1][1]);
        expect_near(filled, (1.0 + 100.0 + 2.0) / 3.0, 1e-12, "Mean imputation value mismatch");
    }

    // Median imputation on column 1 => median(1,2,100) = 2
    {
        TabularData t = table;
        MissingValueOptions opt;
        opt.strategy = MissingValueStrategy::FillWithMedian;
        opt.column_indices = {1};

        const std::size_t modified = handle_missing_values(t, opt);
        expect_eq(modified, static_cast<std::size_t>(1), "Median imputation should modify 1 cell");

        const double filled = as_double(t.rows[1][1]);
        expect_near(filled, 2.0, 1e-12, "Median imputation value mismatch");
    }

    // Missing detection + DropRow
    {
        TabularData t = table;
        MissingValueOptions opt;
        opt.strategy = MissingValueStrategy::DropRow;
        opt.column_indices = {1};

        const std::size_t removed = handle_missing_values(t, opt);
        expect_eq(removed, static_cast<std::size_t>(1), "DropRow should remove 1 row");
        expect_eq(t.rows.size(), static_cast<std::size_t>(3), "Remaining rows after DropRow mismatch");
    }
}

void test_normalization_correctness() {
    using namespace mof::preprocessing;

    NumericDataset ds;
    ds.feature_names = {"f1", "f2"};
    ds.features = {
        {1.0, 10.0},
        {2.0, 20.0},
        {3.0, 30.0}
    };
    ds.target_name = "y";
    ds.target = {100.0, 200.0, 300.0};

    // Min-max normalization should map columns exactly to [0, 0.5, 1]
    {
        NumericDataset t = ds;
        NormalizationOptions opt;
        opt.method = mof::utils::NormalizationMethod::MinMax;
        opt.minmax_out_min = 0.0;
        opt.minmax_out_max = 1.0;

        const auto result = normalize_columns(t, opt);
        expect_eq(result.feature_stats.size(), static_cast<std::size_t>(2), "Feature stats count mismatch");
        expect_eq(result.normalized_columns.size(), static_cast<std::size_t>(2), "Normalized columns count mismatch");

        expect_near(t.features[0][0], 0.0, kEps, "MinMax f1 row0");
        expect_near(t.features[1][0], 0.5, kEps, "MinMax f1 row1");
        expect_near(t.features[2][0], 1.0, kEps, "MinMax f1 row2");

        expect_near(t.features[0][1], 0.0, kEps, "MinMax f2 row0");
        expect_near(t.features[1][1], 0.5, kEps, "MinMax f2 row1");
        expect_near(t.features[2][1], 1.0, kEps, "MinMax f2 row2");
    }

    // Z-score normalization should produce mean ~0 and stddev ~1 for each column
    {
        NumericDataset t = ds;
        NormalizationOptions opt;
        opt.method = mof::utils::NormalizationMethod::ZScore;
        opt.eps = 1e-12;

        normalize_columns(t, opt);

        for (std::size_t c = 0; c < 2; ++c) {
            double mean = 0.0;
            for (const auto& row : t.features) {
                mean += row[c];
            }
            mean /= static_cast<double>(t.features.size());

            double var = 0.0;
            for (const auto& row : t.features) {
                const double d = row[c] - mean;
                var += d * d;
            }
            var /= static_cast<double>(t.features.size()); // population variance
            const double stddev = std::sqrt(var);

            expect_near(mean, 0.0, 1e-12, "ZScore mean should be ~0");
            expect_near(stddev, 1.0, 1e-12, "ZScore stddev should be ~1");
        }
    }
}

void test_invalid_column_handling() {
    using namespace mof::preprocessing;

    TabularData table;
    table.column_names = {"a", "b"};
    table.rows = {
        {"1", "2"},
        {"3", "4"}
    };

    // Out-of-range column index in missing-value handler
    {
        MissingValueOptions opt;
        opt.strategy = MissingValueStrategy::FillWithMean;
        opt.column_indices = {99};

        expect_throw<std::out_of_range>(
            [&]() { (void)handle_missing_values(table, opt); },
            "handle_missing_values should throw on invalid column index"
        );
    }

    // Out-of-range column index in row validation
    {
        RowValidationOptions opt;
        opt.numeric_columns = {0, 5};

        expect_throw<std::out_of_range>(
            [&]() { (void)remove_invalid_rows(table, opt); },
            "remove_invalid_rows should throw on invalid numeric column index"
        );
    }

    // Out-of-range target column in numeric conversion
    {
        NumericConversionOptions opt;
        opt.target_column_index = 10;

        expect_throw<std::out_of_range>(
            [&]() { (void)to_numeric_dataset(table, opt); },
            "to_numeric_dataset should throw on invalid target column index"
        );
    }
}

void test_empty_file_behavior() {
    using namespace mof::preprocessing;

    const fs::path temp_dir = make_test_temp_dir();
    const fs::path empty_csv = temp_dir / "empty.csv";

    // Create an empty file
    write_text_file(empty_csv, "");

    // Empty file should load successfully and produce empty table
    const auto table = load_csv_data(empty_csv.string());

    expect_true(table.column_names.empty(), "Empty CSV should have no header");
    expect_true(table.rows.empty(), "Empty CSV should have no rows");

    // Missing value handling on empty table should be a no-op
    MissingValueOptions mv;
    mv.strategy = MissingValueStrategy::FillWithMean;
    const std::size_t modified = handle_missing_values(const_cast<TabularData&>(table), mv); // local copy not needed

    expect_eq(modified, static_cast<std::size_t>(0), "Missing handling on empty table should return 0");

    // Converting empty table should produce an empty dataset (no throw)
    NumericConversionOptions conv;
    const auto ds = to_numeric_dataset(table, conv);
    expect_true(ds.features.empty(), "Empty table should convert to empty dataset");
    expect_true(ds.target.empty(), "Empty table should produce empty target");

    // Cleanup best-effort
    std::error_code ec;
    fs::remove_all(temp_dir, ec);
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

    run_test("missing value handling (mean/median/drop-row)", test_missing_value_handling_mean_and_median, stats);
    run_test("normalization correctness (min-max + z-score)", test_normalization_correctness, stats);
    run_test("invalid column handling", test_invalid_column_handling, stats);
    run_test("empty file behavior", test_empty_file_behavior, stats);

    std::cout << "\nTest summary: " << stats.passed << " passed, "
              << stats.failed << " failed.\n";

    return (stats.failed == 0) ? 0 : 1;
}