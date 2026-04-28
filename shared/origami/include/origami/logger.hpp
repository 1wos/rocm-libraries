// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace origami {

enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR
};

class Logger {
public:
    static Logger& instance();
    void log(LogLevel level, const std::string& message, const char* file, int line);
    bool is_enabled() const { return enabled_; }
    void flush();

    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    Logger(Logger&&) = delete;
    Logger& operator=(Logger&&) = delete;

private:
    Logger();
    ~Logger();

    std::ofstream log_file_;
    std::mutex mutex_;
    bool enabled_;

    const char* level_to_string(LogLevel level) const;
};

class CsvLogger {
public:
    static CsvLogger& instance();
    bool is_enabled() const { return enabled_; }

    void process_debug_message(const std::string& message);

    CsvLogger(const CsvLogger&) = delete;
    CsvLogger& operator=(const CsvLogger&) = delete;
    CsvLogger(CsvLogger&&) = delete;
    CsvLogger& operator=(CsvLogger&&) = delete;

private:
    CsvLogger();
    ~CsvLogger();

    void begin_row();
    void record(const std::string& key, const std::string& value);
    void end_row();
    void flush_to_file();
    static std::string escape_csv(const std::string& field);

    std::string csv_path_;
    std::mutex mutex_;
    bool enabled_;

    std::vector<std::string> columns_;
    std::unordered_map<std::string, size_t> column_index_;
    std::vector<std::vector<std::string>> rows_;
};

class LogStream {
public:
    LogStream(LogLevel level, const char* file, int line)
        : level_(level), file_(file), line_(line) {}

    ~LogStream() {
        auto msg = stream_.str();
        Logger::instance().log(level_, msg, file_, line_);
        if (level_ == LogLevel::DEBUG && CsvLogger::instance().is_enabled()) {
            CsvLogger::instance().process_debug_message(msg);
        }
    }

    template<typename T>
    LogStream& operator<<(const T& value) {
        stream_ << value;
        return *this;
    }

private:
    std::ostringstream stream_;
    LogLevel level_;
    const char* file_;
    int line_;
};

} // namespace origami

#define ORIGAMI_LOG_DEBUG(msg) \
    if (origami::Logger::instance().is_enabled() || origami::CsvLogger::instance().is_enabled()) \
        origami::LogStream(origami::LogLevel::DEBUG, __FILE__, __LINE__) << msg

#define ORIGAMI_LOG_INFO(msg) \
    if (origami::Logger::instance().is_enabled()) \
        origami::LogStream(origami::LogLevel::INFO, __FILE__, __LINE__) << msg

#define ORIGAMI_LOG_WARNING(msg) \
    if (origami::Logger::instance().is_enabled()) \
        origami::LogStream(origami::LogLevel::WARNING, __FILE__, __LINE__) << msg

#define ORIGAMI_LOG_ERROR(msg) \
    if (origami::Logger::instance().is_enabled()) \
        origami::LogStream(origami::LogLevel::ERROR, __FILE__, __LINE__) << msg

#define OLOG_DEBUG ORIGAMI_LOG_DEBUG
#define OLOG_INFO ORIGAMI_LOG_INFO
#define OLOG_WARNING ORIGAMI_LOG_WARNING
#define OLOG_ERROR ORIGAMI_LOG_ERROR
