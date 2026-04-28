// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "origami/logger.hpp"

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>

namespace origami {

// ---------------------------------------------------------------------------
// Logger
// ---------------------------------------------------------------------------

Logger::Logger() : enabled_(false) {
    const char* log_file_path = std::getenv("ORIGAMI_LOG_FILE");
    
    if (log_file_path != nullptr && log_file_path[0] != '\0') {
        log_file_.open(log_file_path, std::ios::out | std::ios::app);
        
        if (log_file_.is_open()) {
            enabled_ = true;
            log(LogLevel::INFO, 
                "Origami logger initialized, writing to: " + std::string(log_file_path),
                __FILE__, __LINE__);
        } else {
            std::cerr << "Warning: Failed to open log file: " << log_file_path << std::endl;
        }
    }
}

Logger::~Logger() {
    if (enabled_ && log_file_.is_open()) {
        log(LogLevel::INFO, "Logger shutting down", __FILE__, __LINE__);
        log_file_.close();
    }
}

Logger& Logger::instance() {
    static Logger logger;
    return logger;
}

void Logger::log(LogLevel level, const std::string& message, const char* file, int line) {
    if (!enabled_) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    
    if (log_file_.is_open()) {
        const char* filename = file;
        for (const char* p = file; *p; p++) {
            if (*p == '/' || *p == '\\') {
                filename = p + 1;
            }
        }

        log_file_ << "[" << level_to_string(level) << "] "
                  << filename << ":" << line << " - "
                  << message << std::endl;
    }
}

void Logger::flush() {
    if (enabled_ && log_file_.is_open()) {
        std::lock_guard<std::mutex> lock(mutex_);
        log_file_.flush();
    }
}

const char* Logger::level_to_string(LogLevel level) const {
    switch (level) {
        case LogLevel::DEBUG:   return "DEBUG";
        case LogLevel::INFO:    return "INFO ";
        case LogLevel::WARNING: return "WARN ";
        case LogLevel::ERROR:   return "ERROR";
        default:                return "UNKN ";
    }
}

// ---------------------------------------------------------------------------
// CsvLogger -- per-thread row state
// ---------------------------------------------------------------------------

namespace {
struct CsvThreadState {
    bool row_in_progress = false;
    std::vector<std::pair<std::string, std::string>> current_row;
};
}  // namespace

static thread_local CsvThreadState tl_csv_state;

CsvLogger::CsvLogger() : enabled_(false) {
    const char* csv_path = std::getenv("ORIGAMI_CSV_FILE");
    if (csv_path != nullptr && csv_path[0] != '\0') {
        csv_path_ = csv_path;
        enabled_  = true;
    }
}

CsvLogger::~CsvLogger() {
    if (enabled_ && !rows_.empty()) {
        flush_to_file();
    }
}

CsvLogger& CsvLogger::instance() {
    static CsvLogger logger;
    return logger;
}

void CsvLogger::process_debug_message(const std::string& message) {
    if (!enabled_) return;

    static constexpr const char* BEGIN_MARKER = "======== Origami Debug Info ========";
    static constexpr const char* END_MARKER   = "=================================";

    if (message.find(BEGIN_MARKER) != std::string::npos) {
        begin_row();
        return;
    }
    if (message.find(END_MARKER) != std::string::npos) {
        end_row();
        return;
    }

    if (!tl_csv_state.row_in_progress) return;

    auto pos = message.find(": ");
    if (pos != std::string::npos) {
        record(message.substr(0, pos), message.substr(pos + 2));
    }
}

void CsvLogger::begin_row() {
    tl_csv_state.current_row.clear();
    tl_csv_state.row_in_progress = true;
}

void CsvLogger::record(const std::string& key, const std::string& value) {
    if (tl_csv_state.row_in_progress) {
        tl_csv_state.current_row.emplace_back(key, value);
    }
}

void CsvLogger::end_row() {
    if (!tl_csv_state.row_in_progress) return;
    tl_csv_state.row_in_progress = false;

    std::lock_guard<std::mutex> lock(mutex_);

    for (const auto& [key, _] : tl_csv_state.current_row) {
        if (column_index_.find(key) == column_index_.end()) {
            column_index_[key] = columns_.size();
            columns_.push_back(key);
        }
    }

    std::vector<std::string> row(columns_.size());
    for (const auto& [key, value] : tl_csv_state.current_row) {
        row[column_index_[key]] = value;
    }
    rows_.push_back(std::move(row));

    tl_csv_state.current_row.clear();
}

std::string CsvLogger::escape_csv(const std::string& field) {
    if (field.find_first_of(",\"\n\r") == std::string::npos) {
        return field;
    }
    std::string escaped = "\"";
    for (char c : field) {
        if (c == '"') escaped += "\"\"";
        else escaped += c;
    }
    escaped += "\"";
    return escaped;
}

void CsvLogger::flush_to_file() {
    std::ofstream file(csv_path_, std::ios::out | std::ios::trunc);
    if (!file.is_open()) {
        std::cerr << "Warning: Failed to open CSV file: " << csv_path_ << std::endl;
        return;
    }

    for (size_t i = 0; i < columns_.size(); ++i) {
        if (i > 0) file << ",";
        file << escape_csv(columns_[i]);
    }
    file << "\n";

    for (const auto& row : rows_) {
        for (size_t i = 0; i < columns_.size(); ++i) {
            if (i > 0) file << ",";
            if (i < row.size()) {
                file << escape_csv(row[i]);
            }
        }
        file << "\n";
    }
}

} // namespace origami
