// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// origami::profiler — opt-in selection-time profiler for the rank_configs path.
//
// Productized core (origami.cpp / ml_recommender.cpp) ships zero timing or
// printing overhead. Callers that want sel-time numbers register a callback
// via origami::profiler::set_callback(); the ORIGAMI_PROFILE_SCOPE macro at
// instrumented points then invokes it with elapsed wall time + meta.
//
// In Release builds without ORIGAMI_ENABLE_PROFILING defined, the macro
// expands to ((void)0) and the chrono header is not pulled in by callers.
// The bench harness (tests/bench_ml_recommender_seltime.cpp) defines
// ORIGAMI_ENABLE_PROFILING for itself; the productized library never does.
//
// Recommended instrumentation point: Tensile/PredictionLibrary.hpp's
// findTopSolutions, since that captures the user-visible "selection event"
// wall time (config-list copy + rank_configs + predicate filter).

#pragma once

#include <cstddef>

namespace origami {
namespace profiler {

struct meta_t {
  std::size_t m;
  std::size_t n;
  std::size_t k;
  std::size_t batch;
  std::size_t n_configs;
  int         cluster_id;   // -1 if not applicable / not yet routed
};

// Callback signature: tag is a static string literal naming the scope,
// us is wall time in microseconds, meta carries problem context.
using callback_t = void (*)(const char* tag, double us, const meta_t& meta);

void set_callback(callback_t cb);
callback_t get_callback();

}  // namespace profiler
}  // namespace origami

#ifdef ORIGAMI_ENABLE_PROFILING

#include <chrono>

namespace origami {
namespace profiler {

class ScopedTimer {
 public:
  ScopedTimer(const char* tag, const meta_t& meta)
      : tag_(tag), meta_(meta), t0_(std::chrono::steady_clock::now()) {}
  ~ScopedTimer() {
    callback_t cb = get_callback();
    if (!cb) return;
    auto t1 = std::chrono::steady_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0_).count();
    cb(tag_, us, meta_);
  }
  ScopedTimer(const ScopedTimer&) = delete;
  ScopedTimer& operator=(const ScopedTimer&) = delete;

 private:
  const char* tag_;
  meta_t meta_;
  std::chrono::steady_clock::time_point t0_;
};

}  // namespace profiler
}  // namespace origami

#define ORIGAMI_PROFILE_CONCAT_(a, b) a##b
#define ORIGAMI_PROFILE_CONCAT(a, b)  ORIGAMI_PROFILE_CONCAT_(a, b)
#define ORIGAMI_PROFILE_SCOPE(tag, ...) \
  ::origami::profiler::ScopedTimer ORIGAMI_PROFILE_CONCAT(_origami_prof_, __LINE__){tag, __VA_ARGS__}

#else

#define ORIGAMI_PROFILE_SCOPE(tag, ...) ((void)0)

#endif  // ORIGAMI_ENABLE_PROFILING
