#include "utils.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <ios>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <type_traits>

namespace mof::utils {
namespace {

constexpr double kDefaultEps = 1e-12;

[[nodiscard]] bool is_space_char(unsigned char c) {
    return std::isspace(c) != 0;
}

[[nodiscard]] std::string level_to_string(LogLevel level) {
    switch (level) {
        case LogLevel::Debug:   return "DEBUG";
        case LogLevel::Info:    return "INFO";
        case LogLevel::Warning: return "WARN";
        case LogLevel::Error:   return "ERROR";
    }
    return "INFO";
}

[[nodiscard]] std::ostream& stream_for_level(LogLevel level) {
    return (level == LogLevel::Error) ? std::cerr : std::cout;
}

[[nodiscard]] std::string current_timestamp() {
    using clock = std::chrono::system_clock;
    const auto now = clock::now();
    const std::time_t now_c = clock::to_time_t(now);

    std::tm tm_buf{};
#if defined(_WIN32)
    localtime_s(&tm_buf, &now_c);
#else
    localtime_r(&now_c, &tm_buf);
#endif

    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

void validate_non_empty_vector(const std::vector<double>& values, std::string_view func_name) {
    if (values.empty()) {
        throw std::invalid_argument(std::string(func_name) + ": input vector is empty");
    }
}

void validate_rectangular_matrix(const Matrix& data, std::string_view func_name) {
    if (data.empty()) {
        throw std::invalid_argument(std::string(func_name) + ": matrix is empty");
    }
    const std::size_t cols = data.front().size();
    if (cols == 0) {
        throw std::invalid_argument(std::string(func_name) + ": matrix has zero columns");
    }
    for (std::size_t r = 0; r < data.size(); ++r) {
        if (data[r].size() != cols) {
            throw std::invalid_argument(
                std::string(func_name) + ": matrix is not rectangular at row " + std::to_string(r)
            );
        }
    }
}

void validate_stats_shape(
    const Matrix& data,
    const ColumnStats& stats,
    std::string_view func_name
) {
    validate_rectangular_matrix(data, func_name);
    if (stats.size() != data.front().size()) {
        throw std::invalid_argument(
            std::string(func_name) + ": column stats size mismatch (expected " +
            std::to_string(data.front().size()) + ", got " + std::to_string(stats.size()) + ")"
        );
    }
}

[[nodiscard]] double safe_stddev(double variance, double eps) {
    if (variance < 0.0 && variance > -1e-15) {
        variance = 0.0; // guard tiny negative due to floating-point error
    }
    if (variance < 0.0) {
        throw std::runtime_error("safe_stddev: negative variance encountered");
    }
    return std::max(std::sqrt(variance), eps);
}

[[nodiscard]] std::vector<double> allocate_like(const std::vector<double>& values) {
    std::vector<double> out;
    out.resize(values.size());
    return out;
}

} // namespace

// -----------------------------
// File utilities
// -----------------------------
bool file_exists(std::string_view path) {
    std::error_code ec;
    const std::filesystem::path p{std::string(path)};
    return std::filesystem::exists(p, ec) && std::filesystem::is_regular_file(p, ec) && !ec;
}

bool directory_exists(std::string_view path) {
    std::error_code ec;
    const std::filesystem::path p{std::string(path)};
    return std::filesystem::exists(p, ec) && std::filesystem::is_directory(p, ec) && !ec;
}

// -----------------------------
// String utilities
// -----------------------------
std::string trim(std::string_view input) {
    std::size_t first = 0;
    while (first < input.size() && is_space_char(static_cast<unsigned char>(input[first]))) {
        ++first;
    }

    std::size_t last = input.size();
    while (last > first && is_space_char(static_cast<unsigned char>(input[last - 1]))) {
        --last;
    }

    return std::string(input.substr(first, last - first));
}

std::vector<std::string> split(std::string_view input, char delimiter, bool keep_empty) {
    std::vector<std::string> parts;
    if (input.empty()) {
        return parts;
    }

    std::size_t start = 0;
    while (start <= input.size()) {
        const std::size_t pos = input.find(delimiter, start);
        const std::size_t end = (pos == std::string_view::npos) ? input.size() : pos;
        const auto token = input.substr(start, end - start);

        if (keep_empty || !token.empty()) {
            parts.emplace_back(token);
        }

        if (pos == std::string_view::npos) {
            break;
        }
        start = pos + 1;
    }

    return parts;
}

std::vector<std::string> parse_csv_row(std::string_view line, char delimiter, bool trim_fields) {
    std::vector<std::string> fields;
    if (line.empty()) {
        return fields;
    }

    std::string current;
    current.reserve(line.size());

    bool in_quotes = false;

    for (std::size_t i = 0; i < line.size(); ++i) {
        const char ch = line[i];

        if (in_quotes) {
            if (ch == '"') {
                // Escaped quote inside quoted field: ""
                if (i + 1 < line.size() && line[i + 1] == '"') {
                    current.push_back('"');
                    ++i; // skip second quote
                } else {
                    in_quotes = false; // closing quote
                }
            } else {
                current.push_back(ch);
            }
            continue;
        }

        if (ch == delimiter) {
            fields.emplace_back(trim_fields ? trim(current) : current);
            current.clear();
        } else if (ch == '"') {
            // Start quoted field. If quote appears mid-token, we still treat it as quote start.
            in_quotes = true;
        } else {
            current.push_back(ch);
        }
    }

    if (in_quotes) {
        throw std::runtime_error("parse_csv_row: unmatched quote in CSV line");
    }

    fields.emplace_back(trim_fields ? trim(current) : current);
    return fields;
}

// -----------------------------
// Normalization utilities (1D)
// -----------------------------
NormalizationStats fit_normalization_stats(const std::vector<double>& values) {
    validate_non_empty_vector(values, "fit_normalization_stats");

    NormalizationStats stats{};

    auto [min_it, max_it] = std::minmax_element(values.begin(), values.end());
    stats.min = *min_it;
    stats.max = *max_it;

    const double sum = std::accumulate(values.begin(), values.end(), 0.0);
    stats.mean = sum / static_cast<double>(values.size());

    double sq_sum = 0.0;
    for (double v : values) {
        const double d = v - stats.mean;
        sq_sum += d * d;
    }

    const double variance = sq_sum / static_cast<double>(values.size()); // population stddev
    stats.stddev = safe_stddev(variance, kDefaultEps);

    return stats;
}

std::vector<double> normalize_min_max(
    const std::vector<double>& values,
    double out_min,
    double out_max
) {
    validate_non_empty_vector(values, "normalize_min_max");

    if (!(out_max > out_min)) {
        throw std::invalid_argument("normalize_min_max: out_max must be greater than out_min");
    }

    const auto stats = fit_normalization_stats(values);
    std::vector<double> out = allocate_like(values);

    const double in_range = stats.max - stats.min;
    const double out_range = out_max - out_min;

    if (std::abs(in_range) <= kDefaultEps) {
        // Constant input: map all values to out_min (stable, deterministic)
        std::fill(out.begin(), out.end(), out_min);
        return out;
    }

    for (std::size_t i = 0; i < values.size(); ++i) {
        const double x = values[i];
        out[i] = ((x - stats.min) / in_range) * out_range + out_min;
    }

    return out;
}

std::vector<double> normalize_z_score(const std::vector<double>& values, double eps) {
    validate_non_empty_vector(values, "normalize_z_score");

    if (eps <= 0.0) {
        throw std::invalid_argument("normalize_z_score: eps must be > 0");
    }

    auto stats = fit_normalization_stats(values);
    stats.stddev = std::max(stats.stddev, eps);

    std::vector<double> out = allocate_like(values);

    for (std::size_t i = 0; i < values.size(); ++i) {
        out[i] = (values[i] - stats.mean) / stats.stddev;
    }

    return out;
}

// -----------------------------
// Normalization utilities (column-wise)
// -----------------------------
ColumnStats fit_column_stats(const Matrix& data) {
    validate_rectangular_matrix(data, "fit_column_stats");

    const std::size_t rows = data.size();
    const std::size_t cols = data.front().size();

    ColumnStats stats(cols);

    // Initialize min/max
    for (std::size_t c = 0; c < cols; ++c) {
        stats[c].min = std::numeric_limits<double>::infinity();
        stats[c].max = -std::numeric_limits<double>::infinity();
    }

    // First pass: min/max/sum
    for (std::size_t r = 0; r < rows; ++r) {
        const auto& row = data[r];
        for (std::size_t c = 0; c < cols; ++c) {
            const double v = row[c];
            stats[c].min = std::min(stats[c].min, v);
            stats[c].max = std::max(stats[c].max, v);
            stats[c].mean += v;
        }
    }

    for (std::size_t c = 0; c < cols; ++c) {
        stats[c].mean /= static_cast<double>(rows);
    }

    // Second pass: variance
    for (std::size_t r = 0; r < rows; ++r) {
        const auto& row = data[r];
        for (std::size_t c = 0; c < cols; ++c) {
            const double d = row[c] - stats[c].mean;
            stats[c].stddev += d * d;
        }
    }

    for (std::size_t c = 0; c < cols; ++c) {
        const double variance = stats[c].stddev / static_cast<double>(rows); // population variance
        stats[c].stddev = safe_stddev(variance, kDefaultEps);
    }

    return stats;
}

Matrix normalize_columns_min_max(
    const Matrix& data,
    const ColumnStats& stats,
    double out_min,
    double out_max
) {
    validate_stats_shape(data, stats, "normalize_columns_min_max");

    if (!(out_max > out_min)) {
        throw std::invalid_argument("normalize_columns_min_max: out_max must be greater than out_min");
    }

    const std::size_t rows = data.size();
    const std::size_t cols = data.front().size();

    Matrix out(rows, std::vector<double>(cols));
    const double out_range = out_max - out_min;

    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            const double in_range = stats[c].max - stats[c].min;
            if (std::abs(in_range) <= kDefaultEps) {
                out[r][c] = out_min;
            } else {
                out[r][c] = ((data[r][c] - stats[c].min) / in_range) * out_range + out_min;
            }
        }
    }

    return out;
}

Matrix normalize_columns_z_score(
    const Matrix& data,
    const ColumnStats& stats,
    double eps
) {
    validate_stats_shape(data, stats, "normalize_columns_z_score");

    if (eps <= 0.0) {
        throw std::invalid_argument("normalize_columns_z_score: eps must be > 0");
    }

    const std::size_t rows = data.size();
    const std::size_t cols = data.front().size();

    Matrix out(rows, std::vector<double>(cols));

    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            const double denom = std::max(stats[c].stddev, eps);
            out[r][c] = (data[r][c] - stats[c].mean) / denom;
        }
    }

    return out;
}

// -----------------------------
// Random utilities
// -----------------------------
std::uint32_t make_seed(std::optional<std::uint32_t> fixed_seed) {
    if (fixed_seed.has_value()) {
        return *fixed_seed;
    }

    std::random_device rd;
    const auto now_ns = static_cast<std::uint64_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count()
    );
    const auto tid_hash = static_cast<std::uint64_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));

    // Mix entropy sources into 32 bits
    std::uint64_t mixed = now_ns;
    mixed ^= (static_cast<std::uint64_t>(rd()) << 1);
    mixed ^= (tid_hash << 7) ^ (tid_hash >> 3);
    mixed ^= 0x9E3779B97F4A7C15ULL;

    mixed ^= (mixed >> 30);
    mixed *= 0xBF58476D1CE4E5B9ULL;
    mixed ^= (mixed >> 27);
    mixed *= 0x94D049BB133111EBULL;
    mixed ^= (mixed >> 31);

    return static_cast<std::uint32_t>(mixed & 0xFFFFFFFFu);
}

std::mt19937 make_rng(std::optional<std::uint32_t> fixed_seed) {
    return std::mt19937(make_seed(fixed_seed));
}

// -----------------------------
// Logging / printing helpers
// -----------------------------
void log(LogLevel level, std::string_view message, bool newline) {
    static std::mutex log_mutex;

    std::lock_guard<std::mutex> lock(log_mutex);
    std::ostream& os = stream_for_level(level);

    os << "[" << current_timestamp() << "] "
       << "[" << level_to_string(level) << "] "
       << message;

    if (newline) {
        os << '\n';
    }
    os.flush();
}

void log_debug(std::string_view message)   { log(LogLevel::Debug, message); }
void log_info(std::string_view message)    { log(LogLevel::Info, message); }
void log_warning(std::string_view message) { log(LogLevel::Warning, message); }
void log_error(std::string_view message)   { log(LogLevel::Error, message); }

std::string format_vector_preview(
    const std::vector<double>& values,
    std::size_t max_items,
    int precision
) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(std::max(0, precision));

    oss << "[";
    const std::size_t n = values.size();
    const std::size_t shown = std::min(n, max_items);

    for (std::size_t i = 0; i < shown; ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << values[i];
    }

    if (n > shown) {
        oss << ", ... (" << n << " total)";
    }
    oss << "]";

    return oss.str();
}

void print_vector_preview(
    const std::vector<double>& values,
    std::string_view label,
    std::size_t max_items,
    int precision
) {
    std::ostringstream oss;
    oss << label << " = " << format_vector_preview(values, max_items, precision);
    log_info(oss.str());
}

// -----------------------------
// Timer utilities
// -----------------------------
Timer::Timer() : start_(Clock::now()) {}

void Timer::reset() {
    start_ = Clock::now();
}

double Timer::elapsed_seconds() const {
    const auto elapsed = Clock::now() - start_;
    return std::chrono::duration<double>(elapsed).count();
}

long long Timer::elapsed_milliseconds() const {
    const auto elapsed = Clock::now() - start_;
    return std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
}

ScopedTimer::ScopedTimer(std::string label, LogLevel level)
    : label_(std::move(label)), level_(level), timer_() {}

ScopedTimer::~ScopedTimer() noexcept {
    try {
        std::ostringstream oss;
        oss << label_ << " took " << timer_.elapsed_milliseconds() << " ms";
        log(level_, oss.str());
    } catch (...) {
        // Never throw from destructor.
    }
}

} // namespace mof::utils