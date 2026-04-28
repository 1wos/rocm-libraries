// SPDX-License-Identifier: MIT
// CPU sel_time benchmark for the deployed origami::ml_recommender::rank_configs.
// Loads parity_cases.json (sample shapes x candidates) and times the
// FULL strict rank_configs path -- with cluster-tile intersection,
// item-embedding cache, thread-local buffers, and AVX-512 dot.

#include "origami/ml_recommender.hpp"
#include "origami/hardware.hpp"
#include "origami/types.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

// Trivial JSON parser (handles only what dump_parity_cases.py emits).
// We re-parse via system jq if available; otherwise do a minimalist scan.
// For benchmarking, use synthetic problem set instead -- much simpler and
// produces consistent baseline numbers.

namespace {

using bench_clock = std::chrono::steady_clock;

}  // namespace

int main(int argc, char* argv[]) {
  std::string bin = "/tmp/ml_recommender_weights.bin";
  int n_warmup = 50;
  int n_iter = 200;
  int n_shapes = 50;
  int n_cands_per_shape = 50;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--bin" && i + 1 < argc) bin = argv[++i];
    else if (a == "--warmup" && i + 1 < argc) n_warmup = std::atoi(argv[++i]);
    else if (a == "--iter" && i + 1 < argc) n_iter = std::atoi(argv[++i]);
    else if (a == "--shapes" && i + 1 < argc) n_shapes = std::atoi(argv[++i]);
    else if (a == "--cands" && i + 1 < argc) n_cands_per_shape = std::atoi(argv[++i]);
  }

  if (!origami::ml_recommender::load_weights(bin)) {
    std::fprintf(stderr, "Failed to load %s\n", bin.c_str());
    return 1;
  }
  std::fprintf(stderr, "Loaded %s\n", bin.c_str());

  auto hw = origami::hardware_t::get_hardware_for_arch(
      origami::hardware_t::architecture_t::gfx950, 256, 65536, 4194304, 2100000);

  // Synthetic shape set: covers a representative range of (M, N, K).
  std::vector<origami::problem_t> shapes;
  shapes.reserve(n_shapes);
  unsigned seed = 42;
  for (int i = 0; i < n_shapes; ++i) {
    origami::problem_t p{};
    seed = seed * 1103515245 + 12345;
    int log_m = 5 + (seed % 14); seed = seed * 1103515245 + 12345;
    int log_n = 5 + (seed % 14); seed = seed * 1103515245 + 12345;
    int log_k = 5 + (seed % 14); seed = seed * 1103515245 + 12345;
    int log_b = (seed % 4 == 0) ? (seed % 8) : 0;
    p.size = origami::dim3_t{1ULL << log_m, 1ULL << log_n, 1ULL << log_k};
    p.batch = 1ULL << log_b;
    p.a_transpose = origami::transpose_t::T;
    p.b_transpose = origami::transpose_t::N;
    p.a_dtype = p.b_dtype = p.c_dtype = p.d_dtype = origami::data_type_t::BFloat16;
    p.mi_dtype = origami::data_type_t::BFloat16;
    shapes.push_back(p);
  }

  // Use indices from the loaded binary's first cluster's tile_order so
  // the cluster-tile intersection has hits and the item-emb cache is exercised.
  // Hard-coded sample from /tmp/ml_recommender_weights.bin (first cluster's first 50 tiles).
  std::vector<std::size_t> real_tile_indices = {
    12957, 12958, 12959, 12960, 12961, 12962, 12963, 12965, 12966, 12967,
    12968, 12969, 12970, 12971, 12973, 12974, 12975, 12976, 12977, 12978,
    12979, 12980, 12981, 12982, 12983, 12984, 12985, 12986, 12987, 12988,
    12989, 12990, 12991, 12992, 12993, 12994, 12995, 12996, 12997, 12998,
    12999, 13000, 13001, 13002, 13003, 13004, 13005, 13006, 13007, 13008,
  };

  std::vector<origami::config_t> configs;
  configs.reserve(n_cands_per_shape);
  std::vector<int> mt_m_choices = {64, 128, 192, 256};
  std::vector<int> mt_n_choices = {64, 128, 192, 256};
  std::vector<int> mt_k_choices = {32, 64};
  std::vector<int> mi_m_choices = {16, 32};
  for (int i = 0; i < n_cands_per_shape; ++i) {
    origami::config_t c{};
    seed = seed * 1103515245 + 12345;
    c.mt = origami::dim3_t{
      static_cast<std::size_t>(mt_m_choices[seed % mt_m_choices.size()]),
      static_cast<std::size_t>(mt_n_choices[(seed >> 4) % mt_n_choices.size()]),
      static_cast<std::size_t>(mt_k_choices[(seed >> 8) % mt_k_choices.size()])};
    int mi_choice = (seed >> 12) % mi_m_choices.size();
    c.mi = origami::dim3_t{
      static_cast<std::size_t>(mi_m_choices[mi_choice]),
      static_cast<std::size_t>(mi_m_choices[mi_choice]),
      mi_choice == 0 ? 32u : 16u};
    c.occupancy = 1 + ((seed >> 16) % 4);
    c.cache_hints_a = 0; c.cache_hints_b = 4;
    c.workgroup_mapping = 0;
    c.grvw_a = c.grvw_b = 8;
    c.gwvw_d = 4;
    c.index = real_tile_indices[i % real_tile_indices.size()];
    c.prediction_mode = origami::prediction_modes_t::ml_recommender;
    configs.push_back(c);
  }

  std::fprintf(stderr, "Warmup %d iters x %d shapes...\n", n_warmup, n_shapes);
  for (int it = 0; it < n_warmup; ++it) {
    for (const auto& p : shapes) {
      auto r = origami::ml_recommender::rank_configs(p, hw, configs);
      (void)r;
    }
  }

  std::fprintf(stderr, "Bench %d iters x %d shapes...\n", n_iter, n_shapes);
  std::vector<double> per_shape_us;
  per_shape_us.reserve(n_shapes);
  for (const auto& p : shapes) {
    auto t0 = bench_clock::now();
    for (int it = 0; it < n_iter; ++it) {
      auto r = origami::ml_recommender::rank_configs(p, hw, configs);
      (void)r;
    }
    auto t1 = bench_clock::now();
    double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    per_shape_us.push_back(ns / 1000.0 / n_iter);
  }

  std::sort(per_shape_us.begin(), per_shape_us.end());
  double sum = 0; for (double x : per_shape_us) sum += x;
  double mean = sum / per_shape_us.size();
  auto pct = [&](double p) {
    auto i = static_cast<std::size_t>(p * per_shape_us.size());
    if (i >= per_shape_us.size()) i = per_shape_us.size() - 1;
    return per_shape_us[i];
  };
  std::fprintf(stderr,
    "\n=== ml_recommender::rank_configs CPU bench (%d cands/shape, %d iters) ===\n",
    n_cands_per_shape, n_iter);
  std::fprintf(stderr, "  min   = %.2f us\n", per_shape_us.front());
  std::fprintf(stderr, "  mean  = %.2f us\n", mean);
  std::fprintf(stderr, "  p50   = %.2f us\n", pct(0.50));
  std::fprintf(stderr, "  p90   = %.2f us\n", pct(0.90));
  std::fprintf(stderr, "  p95   = %.2f us\n", pct(0.95));
  std::fprintf(stderr, "  p99   = %.2f us\n", pct(0.99));
  std::fprintf(stderr, "  max   = %.2f us\n", per_shape_us.back());
  return 0;
}
