#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace mof::utils {

// -----------------------------
// File utilities
// -----------------------------
bool file_exists(std::string_view path);
bool directory_exists(std::string_view path);

// -----------------------------
// String utilities
// -----------------------------
std::string trim(std::string_view input);
std::vector<std::string> split(std::string_view input, char delimiter, bool keep_empty = true);

// CSV row parser (supports quoted fields and escaped quotes "")
// Example: a,"b,c","x""y"  -> ["a", "b,c", "x\"y"]
std::vector<std::string> parse_csv_row(
    std::string_view line,
    char delimiter = ',',
    bool trim_fields = false
);

// -----------------------------
// Normalization utilities
// -----------------------------
enum class NormalizationMethod {
    MinMax,
    ZScore
};

struct NormalizationStats {
    double min = 0.0;
    double max = 0.0;
    double mean = 0.0;
    double stddev = 1.0; // kept >= epsilon for safe division
};

using Matrix = std::vector<std::vector<double>>;
using ColumnStats = std::vector<NormalizationStats>;

// 1D helpers
NormalizationStats fit_normalization_stats(const std::vector<double>& values);
std::vector<double> normalize_min_max(
    const std::vector<double>& values,
    double out_min = 0.0,
    double out_max = 1.0
);
std::vector<double> normalize_z_score(
    const std::vector<double>& values,
    double eps = 1e-12
);

// Column-wise helpers for matrix-like tabular data
ColumnStats fit_column_stats(const Matrix& data);
Matrix normalize_columns_min_max(
    const Matrix& data,
    const ColumnStats& stats,
    double out_min = 0.0,
    double out_max = 1.0
);
Matrix normalize_columns_z_score(
    const Matrix& data,
    const ColumnStats& stats,
    double eps = 1e-12
);

// -----------------------------
// Random utilities
// -----------------------------
std::uint32_t make_seed(std::optional<std::uint32_t> fixed_seed = std::nullopt);
std::mt19937 make_rng(std::optional<std::uint32_t> fixed_seed = std::nullopt);

// -----------------------------
// Logging / printing helpers
// -----------------------------
enum class LogLevel {
    Debug,
    Info,
    Warning,
    Error
};

void log(LogLevel level, std::string_view message, bool newline = true);
void log_debug(std::string_view message);
void log_info(std::string_view message);
void log_warning(std::string_view message);
void log_error(std::string_view message);

std::string format_vector_preview(
    const std::vector<double>& values,
    std::size_t max_items = 8,
    int precision = 4
);
void print_vector_preview(
    const std::vector<double>& values,
    std::string_view label = "values",
    std::size_t max_items = 8,
    int precision = 4
);

// -----------------------------
// Simple timer utilities
// -----------------------------
class Timer {
public:
    Timer();

    void reset();

    [[nodiscard]] double elapsed_seconds() const;
    [[nodiscard]] long long elapsed_milliseconds() const;

private:
    using Clock = std::chrono::steady_clock;
    Clock::time_point start_;
};

class ScopedTimer {
public:
    explicit ScopedTimer(std::string label, LogLevel level = LogLevel::Info);
    ~ScopedTimer() noexcept;

    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    std::string label_;
    LogLevel level_;
    Timer timer_;
};

} // namespace mof::utils