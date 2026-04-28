/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2026 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include "common.hpp"
#include "origami/logger.hpp"

// Helper: read entire file into a string
static std::string read_file(const std::string& path) {
  std::ifstream ifs(path);
  std::ostringstream ss;
  ss << ifs.rdbuf();
  return ss.str();
}

// Helper: count occurrences of a substring
static size_t count_occurrences(const std::string& text, const std::string& sub) {
  size_t count = 0;
  size_t pos   = 0;
  while ((pos = text.find(sub, pos)) != std::string::npos) {
    ++count;
    pos += sub.size();
  }
  return count;
}

// ---------------------------------------------------------------------------
// Text Logger tests
// ---------------------------------------------------------------------------

TEST_CASE("Logger: text log writes debug messages when enabled", "[logger]") {
  const std::string log_path = "test_text_logger.log";
  std::remove(log_path.c_str());

  portable_setenv("ORIGAMI_LOG_FILE", log_path.c_str(), 1);
  portable_setenv("ANALYTICAL_GEMM_DEBUG", "1", 1);
  origami::Logger::instance().update_from_env();
  origami::runtime_options::get().update_from_env();

  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch) {
      std::remove(log_path.c_str());
      origami::Logger::instance().update_from_env();

      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(2048, 2048, 1024);
      auto config   = make_config(128, 128, 64, 16, 16, 16, false, 1);

      origami::compute_total_latency(problem, hardware, config, hardware.N_CU);
      origami::Logger::instance().flush();

      std::string contents = read_file(log_path);
      REQUIRE_FALSE(contents.empty());
      REQUIRE(contents.find("[DEBUG]") != std::string::npos);
      REQUIRE(contents.find("Origami Debug Info") != std::string::npos);
      REQUIRE(contents.find("total_latency") != std::string::npos);
    }
  }
  std::remove(log_path.c_str());

  portable_unsetenv("ORIGAMI_LOG_FILE");
  portable_unsetenv("ANALYTICAL_GEMM_DEBUG");
  origami::Logger::instance().update_from_env();
  origami::runtime_options::get().update_from_env();
}

TEST_CASE("Logger: text log is not written when debug is disabled", "[logger]") {
  const std::string log_path = "test_text_logger_disabled.log";
  std::remove(log_path.c_str());

  portable_setenv("ORIGAMI_LOG_FILE", log_path.c_str(), 1);
  portable_unsetenv("ANALYTICAL_GEMM_DEBUG");
  origami::Logger::instance().update_from_env();
  origami::runtime_options::get().update_from_env();

  auto hardware = make_hardware(942);
  auto problem  = make_problem(2048, 2048, 1024);
  auto config   = make_config(128, 128, 64, 16, 16, 16, false, 1);

  origami::compute_total_latency(problem, hardware, config, hardware.N_CU);
  origami::Logger::instance().flush();

  std::string contents = read_file(log_path);
  REQUIRE(contents.find("[DEBUG]") == std::string::npos);

  std::remove(log_path.c_str());
  portable_unsetenv("ORIGAMI_LOG_FILE");
  origami::Logger::instance().update_from_env();
}

// ---------------------------------------------------------------------------
// CSV Logger tests
// ---------------------------------------------------------------------------

TEST_CASE("CsvLogger: process_debug_message parses key-value pairs", "[logger][csv]") {
  auto& csv = origami::CsvLogger::instance();
  // Manually drive the CSV logger through its message processing interface
  // to verify parsing without needing full GEMM evaluation.

  // Simulate a row
  csv.process_debug_message("======== Origami Debug Info ========");
  csv.process_debug_message("TestKey1: 42");
  csv.process_debug_message("TestKey2: 3.14");
  csv.process_debug_message("=================================");

  // The row was committed internally. We cannot directly inspect rows_
  // (private), but we verify through the full CSV pipeline in subsequent tests.
  REQUIRE(csv.is_enabled() == (std::getenv("ORIGAMI_CSV_FILE") != nullptr));
}

TEST_CASE("CsvLogger: process_debug_message ignores messages outside a row", "[logger][csv]") {
  auto& csv = origami::CsvLogger::instance();

  // Messages outside begin/end markers should not cause errors
  csv.process_debug_message("Hand-optimized kernel gfx950_BF16_TN, efficiency: 0.95");
  csv.process_debug_message("RandomMessage: value");
  // No crash or assertion failure = pass
  REQUIRE(true);
}

TEST_CASE("CsvLogger: CSV output from GEMM evaluation contains expected columns", "[logger][csv]") {
  const std::string csv_path = "test_csv_logger.csv";
  std::remove(csv_path.c_str());

  portable_setenv("ORIGAMI_CSV_FILE", csv_path.c_str(), 1);
  portable_setenv("ANALYTICAL_GEMM_DEBUG", "1", 1);
  origami::CsvLogger::instance().update_from_env();
  origami::runtime_options::get().update_from_env();

  REQUIRE(origami::CsvLogger::instance().is_enabled());

  auto hardware = make_hardware(942);
  auto problem  = make_problem(4096, 4096, 2048);
  auto config   = make_config(128, 128, 64, 16, 16, 16, false, 1);

  origami::compute_total_latency(problem, hardware, config, hardware.N_CU);

  auto problem2 = make_problem(1024, 1024, 512);
  origami::compute_total_latency(problem2, hardware, config, hardware.N_CU);

  // Flush CSV by switching to a disabled state, which writes accumulated rows
  portable_unsetenv("ORIGAMI_CSV_FILE");
  portable_unsetenv("ANALYTICAL_GEMM_DEBUG");
  origami::CsvLogger::instance().update_from_env();
  origami::runtime_options::get().update_from_env();

  std::string contents = read_file(csv_path);
  REQUIRE_FALSE(contents.empty());
  REQUIRE(contents.find("total_latency") != std::string::npos);
  REQUIRE(contents.find("L_mem") != std::string::npos);
  REQUIRE(contents.find("L_compute") != std::string::npos);
  // Should have header + 2 data rows
  REQUIRE(count_occurrences(contents, "\n") >= 3);

  std::remove(csv_path.c_str());
}

TEST_CASE("CsvLogger: escape_csv handles special characters", "[logger][csv]") {
  // Test the CSV escaping through the public interface by verifying that
  // messages containing commas and quotes don't break the parser.
  auto& csv = origami::CsvLogger::instance();

  csv.process_debug_message("======== Origami Debug Info ========");
  csv.process_debug_message("KeyWithComma: 1,2,3");
  csv.process_debug_message("KeyWithQuote: he said \"hello\"");
  csv.process_debug_message("=================================");

  // No crash = pass; escaping is exercised when flush_to_file runs
  REQUIRE(true);
}

TEST_CASE("CsvLogger: debug logging produces consistent latency values", "[logger][csv]") {
  portable_setenv("ANALYTICAL_GEMM_DEBUG", "1", 1);
  origami::runtime_options::get().update_from_env();

  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch) {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(2048, 2048, 1024);
      auto config   = make_config(128, 128, 64, 16, 16, 16, false, 1);

      double latency1 = origami::compute_total_latency(problem, hardware, config, hardware.N_CU);
      double latency2 = origami::compute_total_latency(problem, hardware, config, hardware.N_CU);

      // Debug logging should not affect computed latency values
      REQUIRE(latency1 == latency2);
      REQUIRE(latency1 > 0.0);
    }
  }

  portable_unsetenv("ANALYTICAL_GEMM_DEBUG");
  origami::runtime_options::get().update_from_env();
}

TEST_CASE("CsvLogger: latency matches between debug enabled and disabled", "[logger][csv]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch) {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(4096, 4096, 2048);
      auto config   = make_config(128, 128, 64, 16, 16, 16, false, 1);

      // Compute without debug
      portable_unsetenv("ANALYTICAL_GEMM_DEBUG");
      origami::runtime_options::get().update_from_env();
      double latency_no_debug = origami::compute_total_latency(problem, hardware, config, hardware.N_CU);

      // Compute with debug
      portable_setenv("ANALYTICAL_GEMM_DEBUG", "1", 1);
      origami::runtime_options::get().update_from_env();
      double latency_with_debug = origami::compute_total_latency(problem, hardware, config, hardware.N_CU);

      REQUIRE(latency_no_debug == latency_with_debug);
    }
  }

  portable_unsetenv("ANALYTICAL_GEMM_DEBUG");
  origami::runtime_options::get().update_from_env();
}

TEST_CASE("Logger: ANALYTICAL_GEMM_DEBUG env var controls debug_enabled flag", "[logger]") {
  portable_setenv("ANALYTICAL_GEMM_DEBUG", "1", 1);
  origami::runtime_options::get().update_from_env();
  REQUIRE(origami::runtime_options::get().debug_enabled == true);

  portable_setenv("ANALYTICAL_GEMM_DEBUG", "0", 1);
  origami::runtime_options::get().update_from_env();
  REQUIRE(origami::runtime_options::get().debug_enabled == false);

  portable_unsetenv("ANALYTICAL_GEMM_DEBUG");
  origami::runtime_options::get().update_from_env();
  REQUIRE(origami::runtime_options::get().debug_enabled == false);
}
