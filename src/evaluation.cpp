 #include "evaluation.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>

namespace mof::evaluation {
namespace {

struct R2ComputationResult {
    double value = 0.0;
    bool valid = true;   // false when y_true variance is ~0 and fallback is used
};

void validate_finite_values(
    const std::vector<double>& values,
    std::string_view name,
    std::string_view caller
) {
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (!std::isfinite(values[i])) {
            throw std::invalid_argument(
                std::string(caller) + ": non-finite value in " + std::string(name) +
                " at index " + std::to_string(i)
            );
        }
    }
}

R2ComputationResult compute_r2_impl(
    const std::vector<double>& y_true,
    const std::vector<double>& y_pred,
    const MetricOptions& options
) {
    validate_prediction_pair(y_true, y_pred, "r2_score");

    if (options.eps <= 0.0) {
        throw std::invalid_argument("r2_score: options.eps must be > 0");
    }

    const std::size_t n = y_true.size();

    const double mean_true =
        std::accumulate(y_true.begin(), y_true.end(), 0.0) / static_cast<double>(n);

    double ss_res = 0.0;
    double ss_tot = 0.0;

    for (std::size_t i = 0; i < n; ++i) {
        const double err = y_true[i] - y_pred[i];
        ss_res += err * err;

        const double d = y_true[i] - mean_true;
        ss_tot += d * d;
    }

    // R² is undefined when y_true has zero (or near-zero) variance.
    if (ss_tot <= options.eps) {
        if (!options.allow_undefined_r2_fallback) {
            throw std::domain_error("r2_score: undefined because y_true variance is near zero");
        }
        return R2ComputationResult{options.fallback_r2_when_undefined, false};
    }

    return R2ComputationResult{1.0 - (ss_res / ss_tot), true};
}

} // namespace

// ============================================================
// Validation helpers
// ============================================================
bool is_valid_prediction_pair(
    const std::vector<double>& y_true,
    const std::vector<double>& y_pred
) {
    return !y_true.empty() && (y_true.size() == y_pred.size());
}

void validate_prediction_pair(
    const std::vector<double>& y_true,
    const std::vector<double>& y_pred,
    std::string_view caller
) {
    if (y_true.empty() || y_pred.empty()) {
        throw std::invalid_argument(std::string(caller) + ": y_true and y_pred must be non-empty");
    }

    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument(
            std::string(caller) + ": size mismatch (y_true=" + std::to_string(y_true.size()) +
            ", y_pred=" + std::to_string(y_pred.size()) + ")"
        );
    }

    validate_finite_values(y_true, "y_true", caller);
    validate_finite_values(y_pred, "y_pred", caller);
}

// ============================================================
// Core regression metrics
// ============================================================
double rmse(
    const std::vector<double>& y_true,
    const std::vector<double>& y_pred
) {
    validate_prediction_pair(y_true, y_pred, "rmse");

    double sse = 0.0;
    for (std::size_t i = 0; i < y_true.size(); ++i) {
        const double e = y_true[i] - y_pred[i];
        sse += e * e;
    }

    const double mse = sse / static_cast<double>(y_true.size());
    return std::sqrt(mse);
}

double mae(
    const std::vector<double>& y_true,
    const std::vector<double>& y_pred
) {
    validate_prediction_pair(y_true, y_pred, "mae");

    double sae = 0.0;
    for (std::size_t i = 0; i < y_true.size(); ++i) {
        sae += std::abs(y_true[i] - y_pred[i]);
    }

    return sae / static_cast<double>(y_true.size());
}

double r2_score(
    const std::vector<double>& y_true,
    const std::vector<double>& y_pred,
    const MetricOptions& options
) {
    return compute_r2_impl(y_true, y_pred, options).value;
}

MapeResult mape(
    const std::vector<double>& y_true,
    const std::vector<double>& y_pred,
    const MetricOptions& options
) {
    validate_prediction_pair(y_true, y_pred, "mape");

    if (options.eps <= 0.0) {
        throw std::invalid_argument("mape: options.eps must be > 0");
    }

    MapeResult result{};
    double sum_abs_pct = 0.0;

    for (std::size_t i = 0; i < y_true.size(); ++i) {
        const double denom = std::abs(y_true[i]);

        if (denom <= options.eps) {
            if (options.ignore_near_zero_targets_for_mape) {
                ++result.skipped_count;
                continue;
            }
            throw std::domain_error(
                "mape: near-zero target at index " + std::to_string(i) +
                " (|y_true| <= eps)"
            );
        }

        const double abs_pct = std::abs((y_true[i] - y_pred[i]) / denom) * 100.0;
        sum_abs_pct += abs_pct;
        ++result.used_count;
    }

    if (result.used_count == 0) {
        result.value = 0.0;
        result.valid = false;
        return result;
    }

    result.value = sum_abs_pct / static_cast<double>(result.used_count);
    result.valid = true;
    return result;
}

// ============================================================
// Aggregate evaluation helper
// ============================================================
RegressionMetrics evaluate_regression(
    const std::vector<double>& y_true,
    const std::vector<double>& y_pred,
    const MetricOptions& options
) {
    validate_prediction_pair(y_true, y_pred, "evaluate_regression");

    RegressionMetrics out{};
    out.sample_count = y_true.size();

    out.rmse = rmse(y_true, y_pred);
    out.mae = mae(y_true, y_pred);

    const auto r2res = compute_r2_impl(y_true, y_pred, options);
    out.r2 = r2res.value;
    out.has_valid_r2 = r2res.valid;

    out.mape = mape(y_true, y_pred, options);
    out.has_valid_mape = out.mape.valid;

    return out;
}

// ============================================================
// Optional formatting helper
// ============================================================
std::string format_metrics(
    const RegressionMetrics& metrics,
    int precision
) {
    if (precision < 0) {
        precision = 0;
    }

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision);

    oss << "n=" << metrics.sample_count
        << ", RMSE=" << metrics.rmse
        << ", MAE=" << metrics.mae
        << ", R2=";

    if (metrics.has_valid_r2) {
        oss << metrics.r2;
    } else {
        oss << metrics.r2 << " (fallback)";
    }

    oss << ", MAPE=";
    if (metrics.has_valid_mape) {
        oss << metrics.mape.value << "%";
        if (metrics.mape.skipped_count > 0) {
            oss << " (used=" << metrics.mape.used_count
                << ", skipped=" << metrics.mape.skipped_count << ")";
        }
    } else {
        oss << "N/A";
        if (metrics.mape.skipped_count > 0 || metrics.mape.used_count > 0) {
            oss << " (used=" << metrics.mape.used_count
                << ", skipped=" << metrics.mape.skipped_count << ")";
        }
    }

    return oss.str();
}

} // namespace mof::evaluation