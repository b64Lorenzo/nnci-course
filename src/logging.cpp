#include "nnci/logging.hpp"

#include <atomic>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace nnci {
namespace {

std::atomic<unsigned int> run_sequence{0};

std::string timestamp(const char* format) {
    const auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm utc{};
#if defined(_WIN32)
    if (gmtime_s(&utc, &now) != 0) {
        throw std::runtime_error("Could not create UTC timestamp");
    }
#else
    if (gmtime_r(&now, &utc) == nullptr) {
        throw std::runtime_error("Could not create UTC timestamp");
    }
#endif

    std::ostringstream output;
    output << std::put_time(&utc, format);
    return output.str();
}

const char* level_name(LogLevel level) {
    switch (level) {
        case LogLevel::Debug: return "DEBUG";
        case LogLevel::Info: return "INFO";
        case LogLevel::Warning: return "WARNING";
        case LogLevel::Error: return "ERROR";
        case LogLevel::Off: return "OFF";
    }
    throw std::invalid_argument("Unknown log level");
}

int level_rank(LogLevel level) {
    switch (level) {
        case LogLevel::Debug: return 0;
        case LogLevel::Info: return 1;
        case LogLevel::Warning: return 2;
        case LogLevel::Error: return 3;
        case LogLevel::Off: return 4;
    }
    throw std::invalid_argument("Unknown log level");
}

std::string one_line(std::string_view text) {
    std::string result;
    result.reserve(text.size());

    for (const unsigned char character : text) {
        switch (character) {
            case '\\': result += "\\\\"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '|': result += "\\|"; break;
            default:
                result += character < 32 || character == 127
                    ? '?'
                    : static_cast<char>(character);
        }
    }

    return result;
}

}  // namespace

Logger::Logger(LogLevel minimum_level, const std::filesystem::path& log_root)
    : minimum_level_(minimum_level) {
    (void)level_name(minimum_level_);
    if (log_root.empty()) {
        throw std::invalid_argument("Log directory must not be empty");
    }

    std::filesystem::create_directories(log_root);
    for (int attempt = 0; attempt < 100; ++attempt) {
        const auto run_id = timestamp("%Y%m%dT%H%M%SZ") + "-" +
                            std::to_string(run_sequence.fetch_add(1));
        const auto candidate = log_root / run_id;
        if (std::filesystem::create_directory(candidate)) {
            run_directory_ = candidate;
            break;
        }
    }

    if (run_directory_.empty()) {
        throw std::runtime_error("Could not create a unique run directory");
    }

    log_file_ = run_directory_ / "run.log";
    file_.open(log_file_, std::ios::out | std::ios::binary);
    if (!file_) {
        throw std::runtime_error("Could not open run log for writing");
    }
}

void Logger::write(LogLevel level, std::string_view component, std::string_view message) {
    if (minimum_level_ == LogLevel::Off || level == LogLevel::Off ||
        level_rank(level) < level_rank(minimum_level_)) {
        return;
    }

    const auto line = timestamp("%Y-%m-%dT%H:%M:%SZ") + "|" + level_name(level) + "|" +
                      one_line(component) + "|" + one_line(message);
    file_ << line << '\n';
    file_.flush();
    std::cerr << line << '\n';

    if (!file_) {
        throw std::runtime_error("Could not write to run log");
    }
}

const std::filesystem::path& Logger::run_directory() const noexcept {
    return run_directory_;
}

const std::filesystem::path& Logger::log_file() const noexcept {
    return log_file_;
}

}  // namespace nnci
