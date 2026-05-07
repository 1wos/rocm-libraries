#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
FMHA tile engine benchmark runner.

Uses the dispatcher's setup_multiple_fmha_dispatchers() for pipelined JIT
compilation, then runs GPU benchmarks and reports results.

Usage:
    python fmha_benchmark.py configs/fwd.json
    python fmha_benchmark.py configs/receipt0_fwd.json --workers 256 --build-dir /tmp/fmha_build
    python fmha_benchmark.py configs/fwd.json --problems "2,8,1024,128" --verify
"""

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

_DISPATCHER_ROOT = Path(__file__).resolve().parents[3] / "dispatcher"
sys.path.insert(0, str(_DISPATCHER_ROOT / "python"))
sys.path.insert(0, str(_DISPATCHER_ROOT / "codegen"))

from fmha_utils import (  # noqa: E402
    FmhaProblem,
    cpu_attention_fwd,
    detect_gpu_arch,
    setup_multiple_fmha_dispatchers,
)

from fmha.instance_gen import expand_sweep, apply_filter, enumerate_all_tiles_brute_force  # noqa: E402


def parse_problems(spec: str) -> List[FmhaProblem]:
    """Parse problem specs: 'batch,nhead,seqlen,hdim;...'"""
    problems = []
    for part in spec.split(";"):
        vals = [int(x) for x in part.split(",")]
        if len(vals) == 4:
            b, h, s, d = vals
            problems.append(
                FmhaProblem(
                    batch=b,
                    nhead_q=h,
                    nhead_k=h,
                    seqlen_q=s,
                    seqlen_k=s,
                    hdim_q=d,
                    hdim_v=d,
                )
            )
        elif len(vals) == 6:
            b, hq, hk, sq, sk, d = vals
            problems.append(
                FmhaProblem(
                    batch=b,
                    nhead_q=hq,
                    nhead_k=hk,
                    seqlen_q=sq,
                    seqlen_k=sk,
                    hdim_q=d,
                    hdim_v=d,
                )
            )
    return problems


def main():
    parser = argparse.ArgumentParser(description="FMHA Tile Engine Benchmark")
    parser.add_argument("configs", nargs="+", help="Sweep config JSON(s)")
    parser.add_argument("--arch", default=detect_gpu_arch())
    parser.add_argument("--workers", type=int, default=8, help="Parallel JIT workers")
    parser.add_argument(
        "--problems",
        default="2,8,1024,128",
        help="Problem sizes: batch,nhead,seqlen,hdim",
    )
    parser.add_argument("--receipt", type=int, default=0)
    parser.add_argument(
        "--verify", action="store_true", help="Verify against CPU reference"
    )
    parser.add_argument(
        "--best", action="store_true", help="Show best kernel per problem"
    )
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--json", type=str, default=None)
    parser.add_argument(
        "--build-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "build"),
        help="JIT build output directory",
    )
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument(
        "--filter",
        dest="filter_expr",
        default="",
        help='Python expr per config, e.g. "c.hdim_q == 128"',
    )
    parser.add_argument(
        "--filter-file", default="", help="Path to .py with filter_config(c) -> bool"
    )
    parser.add_argument(
        "--tiles",
        choices=["rules", "exhaustive"],
        default="rules",
        help="Tile enumeration mode: 'rules' (default) uses constraint-based generation; "
        "'exhaustive' brute-forces ALL compilable tiles (like the oracle)",
    )
    args = parser.parse_args()

    problems = parse_problems(args.problems)
    build_dir = Path(args.build_dir).resolve()

    if args.clean and build_dir.exists():
        print(f"  Cleaning {build_dir} ...")
        shutil.rmtree(build_dir)

    build_dir.mkdir(parents=True, exist_ok=True)

    # Phase 0: Expand configs
    all_configs = []
    if args.tiles == "exhaustive":
        # Brute-force: enumerate ALL compilable tiles for problems' hdim
        # Requires at least one config JSON for feature flags (pipeline, dtype, mode, etc.)
        cfg_path = args.configs[0]
        with open(cfg_path) as f:
            spec = json.load(f)
        trait = spec.get("trait_config", {})
        dtype = trait.get("data_type", {}).get("values", ["fp16"])[0]
        pipeline = trait.get("pipeline", {}).get("values", ["qr"])[0]
        mode = trait.get("mode", {}).get("values", ["batch"])[0]
        mask = trait.get("mask", {}).get("values", ["no"])[0]
        bias = trait.get("bias", {}).get("values", ["no"])[0]
        lse = trait.get("lse", {}).get("values", [False])[0]
        dropout = trait.get("dropout", {}).get("values", [False])[0]

        # Derive hdim from problems
        hdim_q = problems[0].hdim_q
        hdim_v = problems[0].hdim_v

        tiles_and_cfgs = enumerate_all_tiles_brute_force(
            args.arch, dtype, hdim_q, hdim_v, pipeline
        )
        for _tile, cfg in tiles_and_cfgs:
            # Override features from the JSON config
            cfg.mode = mode
            cfg.mask = mask
            cfg.bias = bias
            cfg.has_lse = lse
            cfg.has_dropout = dropout
            all_configs.append(cfg)

        print(
            f"  Exhaustive: {len(all_configs)} tile combos"
            f" ({dtype}/{pipeline}/h{hdim_q})"
        )
    else:
        for cfg_path in args.configs:
            configs = expand_sweep(cfg_path, args.arch, args.receipt)
            all_configs.extend(configs)
            print(f"  {cfg_path}: {len(configs)} kernel configs")

    if args.filter_expr or args.filter_file:
        before = len(all_configs)
        all_configs = apply_filter(all_configs, args.filter_expr, args.filter_file)
        print(f"  Filter: {before} -> {len(all_configs)} configs")

    print(f"\n{'=' * 70}")
    print("FMHA Tile Engine Benchmark")
    print(f"{'=' * 70}")
    print(f"  Arch:     {args.arch}")
    print(f"  Kernels:  {len(all_configs)}")
    print(f"  Problems: {len(problems)}")
    print(f"  Workers:  {args.workers}")
    print(f"  Build:    {build_dir}")

    # Phase 1: Pipelined JIT via the dispatcher
    print(
        f"\n--- Phase 1: JIT compile ({len(all_configs)} kernels,"
        f" {args.workers} workers) ---"
    )
    jit_t0 = time.perf_counter()

    setups = setup_multiple_fmha_dispatchers(
        all_configs,
        output_dir=build_dir,
        verbose=True,
        max_workers=args.workers,
    )

    jit_time = time.perf_counter() - jit_t0
    built = sum(1 for s in setups if s.success)
    failed = len(all_configs) - built
    print(f"\n  Built {built}/{len(all_configs)} in {jit_time:.0f}s ({failed} failed)")

    # Load runners for successful builds (setup_multiple doesn't create them)
    from fmha_utils import FmhaRunner  # noqa: E402

    for s in setups:
        if s.success and s.runner is None and s.library_path:
            try:
                s.runner = FmhaRunner.from_library(s.library_path, args.arch)
            except Exception:
                s.runner = None

    if args.compile_only:
        print(f"\n{'=' * 70}")
        print(f"  Compile-only mode. {built}/{len(all_configs)} kernels compiled.")
        if failed > 0:
            print(f"\n  Failed kernels:")
            for cfg, s in zip(all_configs, setups):
                if not s.success:
                    err = (s.error or "unknown")[:80]
                    print(f"    {cfg.name}: {err}")
        if args.tiles == "exhaustive":
            # Oracle-style analysis: find tiles missed by rules vs compilable
            from fmha.instance_gen import validate_tile, FmhaTileConfig  # noqa: E402
            missed = []
            for cfg, s in zip(all_configs, setups):
                if s.success:
                    tile = FmhaTileConfig(
                        bm0=cfg.tile_m0, bn0=cfg.tile_n0, bk0=cfg.tile_k0,
                        bn1=cfg.tile_n1, bk1=cfg.tile_k1, bk0max=cfg.tile_k0max,
                        rm0=cfg.wave_m0, rn0=1, rk0=1,
                        rm1=cfg.wave_m1, rn1=1, rk1=1,
                        wm0=cfg.warp_m0, wn0=cfg.warp_n0, wk0=cfg.warp_k0,
                        wm1=cfg.warp_m1, wn1=cfg.warp_n1, wk1=cfg.warp_k1,
                    )
                    if not validate_tile(tile, args.arch, cfg.data_type, cfg.hdim_q, cfg.hdim_v, cfg.pipeline):
                        missed.append(cfg)
            if missed:
                print(f"\n  MISSED by rules ({len(missed)} tiles compile but rules reject):")
                seen = set()
                for cfg in missed:
                    key = (cfg.tile_m0, cfg.tile_n0, cfg.tile_k0)
                    if key not in seen:
                        seen.add(key)
                        print(f"    ({cfg.tile_m0:>3}, {cfg.tile_n0:>3}, {cfg.tile_k0:>3})")
            else:
                print(f"\n  Rules are COMPLETE — all compilable tiles are generated by rules.")
        print(f"{'=' * 70}")
        return

    # Phase 2: Benchmark
    print(f"\n--- Phase 2: Benchmark ({built} kernels x {len(problems)} problems) ---")

    dtype_map = {
        "fp16": np.float16,
        "bf16": np.float16,
        "fp32": np.float32,
        "fp8bf16": np.float16,
        "fp8fp32": np.float16,
        "bf8": np.float16,
    }
    np.random.seed(42)
    all_results = []
    bench_t0 = time.perf_counter()

    for prob_idx, prob in enumerate(problems):
        first_dtype = all_configs[0].data_type if all_configs else "fp16"
        np_dtype = dtype_map.get(first_dtype, np.float16)
        Q = (np.random.randn(*prob.q_shape()) * 0.1).astype(np_dtype)
        K = (np.random.randn(*prob.k_shape()) * 0.1).astype(np_dtype)
        V = (np.random.randn(*prob.v_shape()) * 0.1).astype(np_dtype)

        ref = None
        if args.verify:
            ref = cpu_attention_fwd(
                Q.astype(np.float32),
                K.astype(np.float32),
                V.astype(np.float32),
                prob.scale,
            )

        prob_str = f"B={prob.batch} H={prob.nhead_q} S={prob.seqlen_q} D={prob.hdim_q}"
        print(f"\n  Problem [{prob_idx}]: {prob_str}")
        print(
            f"  {'Kernel':<50} {'Time(ms)':>10} {'TFLOPS':>10}"
            f" {'MaxErr':>10} {'Status':>6}"
        )
        print(f"  {'-' * 90}")

        for config, setup in zip(all_configs, setups):
            if not setup.success or setup.runner is None:
                continue

            result = setup.runner.run(Q, K, V, prob)
            if not result.success:
                continue

            max_err = 0.0
            status = "OK"
            if ref is not None and result.output is not None:
                max_err = float(np.abs(result.output.astype(np.float32) - ref).max())
                status = "PASS" if max_err < 0.01 else "FAIL"

            print(
                f"  {config.name:<50} {result.time_ms:>10.3f}"
                f" {result.tflops:>10.2f} {max_err:>10.2e} {status:>6}"
            )

            all_results.append(
                {
                    "kernel": config.name,
                    "dtype": config.data_type,
                    "hdim": config.hdim_q,
                    "pipeline": config.pipeline,
                    "problem": {
                        "batch": prob.batch,
                        "nhead_q": prob.nhead_q,
                        "seqlen_q": prob.seqlen_q,
                        "hdim_q": prob.hdim_q,
                    },
                    "latency_ms": result.time_ms,
                    "tflops": result.tflops,
                    "max_err": max_err,
                }
            )

    bench_time = time.perf_counter() - bench_t0

    # Cleanup
    for setup in setups:
        if setup.success and setup.runner:
            try:
                setup.runner.cleanup()
            except Exception:
                pass

    # Report
    print(f"\n{'=' * 70}")
    print(f"  JIT:       {jit_time:.0f}s ({built} kernels)")
    print(f"  Benchmark: {bench_time:.1f}s")
    print(f"  Results:   {len(all_results)} measurements")

    if args.best and all_results:
        from collections import defaultdict

        by_problem = defaultdict(list)
        for r in all_results:
            key = json.dumps(r["problem"], sort_keys=True)
            by_problem[key].append(r)

        print("\n  Best kernel per problem:")
        for key, results in by_problem.items():
            best = max(results, key=lambda x: x["tflops"])
            prob = json.loads(key)
            print(
                f"    B={prob['batch']} H={prob['nhead_q']}"
                f" S={prob['seqlen_q']} D={prob['hdim_q']}"
                f" -> {best['kernel']} ({best['tflops']:.2f} TFLOPS)"
            )

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "kernel",
                    "dtype",
                    "hdim",
                    "pipeline",
                    "batch",
                    "nhead_q",
                    "seqlen_q",
                    "hdim_q",
                    "latency_ms",
                    "tflops",
                    "max_err",
                ],
            )
            writer.writeheader()
            for r in all_results:
                row = {**r, **r["problem"]}
                del row["problem"]
                writer.writerow(row)
        print(f"\n  CSV: {args.csv}")

    if args.json:
        report = {
            "metadata": {
                "arch": args.arch,
                "jit_time_s": jit_time,
                "bench_time_s": bench_time,
                "num_kernels": len(all_configs),
                "num_built": built,
                "num_problems": len(problems),
            },
            "results": all_results,
        }
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  JSON: {args.json}")

    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
