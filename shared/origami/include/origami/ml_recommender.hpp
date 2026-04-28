// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "origami/types.hpp"
#include "origami/hardware.hpp"

#include <string>
#include <vector>

namespace origami {
namespace ml_recommender {

// Predict tile / rank configs API mirrors origami::ml_recommender. The
// strict pipeline differs in:
//   * 131-feature catalog (origami::config_t excl. tensile_params_t,
//     origami::problem_t, origami::hardware_t, math).
//   * Per-cluster two-tower model (TwoTower) with adaptive capacity.
//   * Weight binary 'MLREC' (ml_recommender_weights.bin), produced by
//     export_strict_cpp.py.

// Load weights from path (or env ML_RECOMMENDER_WEIGHTS).
// Lazily called by rank_configs() on first invocation.
bool load_weights(const std::string& bin_path);

// True if weights are loaded (init_weights succeeded).
bool weights_loaded();

// Same signature as ml_recommender::rank_configs. Falls back to
// uniform-score order if weights missing or candidate-pool intersection
// with cluster.tile_order is empty.
std::vector<prediction_result_t> rank_configs(const problem_t& problem,
                                              const hardware_t& hardware,
                                              const std::vector<config_t>& configs);

// A4 hybrid routing (2026-04-20): compile-time per-cluster decision ML vs
// Origami. cluster_uses_ml(cid) returns true if cluster cid should use ML
// (the default; can be overridden via ML_HYBRID_ROUTING file). False
// means caller should use Origami's analytical estimator instead. The
// routing file is plain text: one "<cid> ml" or "<cid> origami" per line.
bool cluster_uses_ml(int cluster_id);

// Route a problem to a cluster (exposes the K-means routing for callers
// implementing hybrid dispatch).
int route_cluster_for_problem(const problem_t& problem);

}  // namespace ml_recommender
}  // namespace origami
