#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace mof::evaluation {

// ============================================================
// Regression metric configuration
// ============================================================

struct MetricOptions {
    // Numerical stability epsilon used by metrics that may divide by small values
    // (e.g., MAPE and some R^2 edge cases).
    double eps = 1e-12;

    // MAPE handling:
    // - if true: skip samples where |y_true| <= eps
    // - if false: implementations may throw on near-zero y_true
    bool ignore_near_zero_targets_for_mape = true;

    // If all y_true are near-constant, R^2 may be undefined.
    // - if true: return fallback_r2_when_undefined
    // - if false: implementations may throw
    bool allow_undefined_r2_fallback = true;
    double fallback_r2_when_undefined = 0.0;
};

// ============================================================
// Metric result structs
// ============================================================

struct MapeResult {
    double value = 0.0;                 // percentage (e.g., 12.3 means 12.3%)
    std::size_t used_count = 0;         // number of samples used in MAPE denominator
    std::size_t skipped_count = 0;      // skipped near-zero targets
    bool valid = false;                 // false if no samples were usable
};

struct RegressionMetrics {
    std::size_t sample_count = 0;

    double rmse = 0.0;
    double mae = 0.0;
    double r2 = 0.0;

    // Optional metric (may be invalid depending on target values / options)
    MapeResult mape{};

    // Convenience flags for consumers (logging/reporting/model selection)
    bool has_valid_r2 = true;
    bool has_valid_mape = false;
};

// ============================================================
// Validation helpers
// ============================================================

/// Returns true if y_true and y_pred are both non-empty and have the same size.
bool is_valid_prediction_pair(
    const std::vector<double>& y_true,
    const std::vector<double>& y_pred
);

/// Throws if vectors are empty or sizes mismatch.
/// Intended for metric implementations and callers that want fail-fast behavior.
void validate_prediction_pair(
    const std::vector<double>& y_true,
    const std::vector<double>& y_pred,
    std::string_view caller = "evaluation"
);

// ============================================================
// Core regression metrics
// ============================================================

/// Root Mean Squared Error (RMSE)
double rmse(
    const std::vector<double>& y_true,
    const std::vector<double>& y_pred
);

/// Mean Absolute Error (MAE)
double mae(
    const std::vector<double>& y_true,
    const std::vector<double>& y_pred
);

/// Coefficient of Determination (R²)
/// May be undefined for near-constant y_true depending on options.
double r2_score(
    const std::vector<double>& y_true,
    const std::vector<double>& y_pred,
    const MetricOptions& options = {}
);

/// Mean Absolute Percentage Error (MAPE), in percentage units.
/// Example: returns 12.5 for 12.5% error.
MapeResult mape(
    const std::vector<double>& y_true,
    const std::vector<double>& y_pred,
    const MetricOptions& options = {}
);

// ============================================================
// Aggregate evaluation helper
// ============================================================

/// Compute a standard regression metric bundle in one call (RMSE, MAE, R², MAPE).
RegressionMetrics evaluate_regression(
    const std::vector<double>& y_true,
    const std::vector<double>& y_pred,
    const MetricOptions& options = {}
);

// ============================================================
// Optional reporting / formatting helpers (lightweight)
// ============================================================

/// Format metrics into a compact human-readable line for logs/reports.
/// Implementations may omit invalid metrics or mark them explicitly.
std::string format_metrics(
    const RegressionMetrics& metrics,
    int precision = 6
);

} // namespace mof::evaluation