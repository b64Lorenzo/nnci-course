#pragma once

#include <filesystem>
#include <fstream>
#include <string_view>

namespace nnci {

enum class LogLevel { Debug, Info, Warning, Error, Off };

class Logger {
public:
    explicit Logger(LogLevel minimum_level = LogLevel::Info,
                    const std::filesystem::path& log_root = "logs");
    ~Logger() = default;

    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    void write(LogLevel level, std::string_view component, std::string_view message);

    [[nodiscard]] const std::filesystem::path& run_directory() const noexcept;
    [[nodiscard]] const std::filesystem::path& log_file() const noexcept;

private:
    LogLevel minimum_level_;
    std::filesystem::path run_directory_;
    std::filesystem::path log_file_;
    std::ofstream file_;
};

}  // namespace nnci
