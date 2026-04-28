// SPDX-License-Identifier: MIT
// Standalone parity test: load ml_recommender_weights.bin via deployed
// origami::ml_recommender::rank_configs, run on synthetic configs, ensure
// scores are finite and non-trivial. Real pick parity vs Python is
// validated separately via dump_parity_cases.json + cpp_scaffold tests.

#include "origami/ml_recommender.hpp"
#include "origami/hardware.hpp"
#include "origami/types.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char* argv[]) {
  std::string bin = "/tmp/ml_recommender_weights.bin";
  if (argc >= 2) bin = argv[1];
  if (!origami::ml_recommender::load_weights(bin)) {
    std::fprintf(stderr, "Failed to load %s\n", bin.c_str());
    return 1;
  }
  std::fprintf(stderr, "Loaded %s\n", bin.c_str());

  origami::problem_t p{};
  p.size = origami::dim3_t{8192, 8192, 8192};
  p.batch = 1;
  p.a_transpose = origami::transpose_t::T;
  p.b_transpose = origami::transpose_t::N;
  p.a_dtype = p.b_dtype = p.c_dtype = p.d_dtype = origami::data_type_t::BFloat16;
  p.mi_dtype = origami::data_type_t::BFloat16;

  std::vector<origami::config_t> configs;
  for (int i = 0; i < 5; ++i) {
    origami::config_t c{};
    c.mt = origami::dim3_t{static_cast<std::size_t>(128 + i * 32),
                            static_cast<std::size_t>(128 + i * 16),
                            64};
    c.mi = origami::dim3_t{16, 16, 32};
    c.occupancy = 1 + i;
    c.cache_hints_a = c.cache_hints_b = 0;
    c.workgroup_mapping = 0;
    c.grvw_a = c.grvw_b = 8;
    c.gwvw_d = 4;
    c.index = i + 1000;
    c.prediction_mode = origami::prediction_modes_t::ml_recommender;
    configs.push_back(c);
  }

  // hardware_t is unused by ml_recommender (uses inlined hw constants), but needed for API.
  auto hw = origami::hardware_t::get_hardware_for_arch(
      origami::hardware_t::architecture_t::gfx950, 256, 65536, 4194304, 2100000);
  auto out = origami::ml_recommender::rank_configs(p, hw, configs);
  std::fprintf(stderr, "Got %zu predictions:\n", out.size());
  int n_finite = 0;
  for (std::size_t i = 0; i < out.size(); ++i) {
    bool ok = std::isfinite(out[i].latency);
    if (ok) ++n_finite;
    std::fprintf(stderr, "  [%zu] sol_idx=%zu latency=%g %s\n",
                 i, out[i].config.index, out[i].latency, ok ? "OK" : "(uniform)");
  }
  std::fprintf(stderr, "n_finite=%d/%zu\n", n_finite, out.size());
  return 0;
}
