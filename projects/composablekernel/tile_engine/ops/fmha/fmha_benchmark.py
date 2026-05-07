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
    FmhaRunner,
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
    parser.add_argument("configs", nargs="*", help="Sweep config JSON(s) (optional for exhaustive)")
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
        "--log", type=str, default=None,
        help="Path to detailed log file (compilation status, failures, timings)",
    )
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
    parser.add_argument(
        "--num-splits",
        default="1,2,4,8",
        help="Comma-separated num_splits values to sweep for splitkv (default: 1,2,4,8)",
    )
    args = parser.parse_args()

    problems = parse_problems(args.problems)
    num_splits_list = [int(x) for x in args.num_splits.split(",")]
    build_dir = Path(args.build_dir).resolve()

    if args.clean and build_dir.exists():
        print(f"  Cleaning {build_dir} ...")
        shutil.rmtree(build_dir)

    build_dir.mkdir(parents=True, exist_ok=True)

    # Phase 0: Expand configs
    all_configs = []
    if args.tiles == "exhaustive":
        # TRUE exhaustive: ALL tiles × ALL features × ALL block_per_cu values
        # JSON config is optional — if provided, it scopes the sweep; if not, sweep everything
        trait = {}
        if args.configs:
            with open(args.configs[0]) as f:
                spec = json.load(f)
            trait = spec.get("trait_config", {})

        from fmha.instance_gen import (  # noqa: E402
            WARP_CLASSES, VALID_BM0, VALID_BN0, VALID_BK0,
            K0_MAX_SUBMAX_MAP, derive_bk1, derive_bk1_fp8,
        )
        from fmha.specs import ARCH_DTYPES, SUPPORTED_HDIMS  # noqa: E402
        from fmha_utils import FmhaKernelConfig  # noqa: E402

        # Defaults = sweep EVERYTHING unless constrained by JSON
        dtypes = trait.get("data_type", {}).get("values", None)
        if dtypes is None:
            dtypes = ARCH_DTYPES.get(args.arch, ["fp16", "bf16"])
        pipelines = trait.get("pipeline", {}).get("values",
            ["qr", "qr_async", "qr_async_trload", "qr_async_trload_v3"])
        modes = trait.get("mode", {}).get("values", ["batch", "group"])
        masks = trait.get("mask", {}).get("values", ["no", "top_left", "bottom_right"])
        biases = trait.get("bias", {}).get("values", ["no", "bias", "alibi"])
        lse_vals = trait.get("lse", {}).get("values", [False, True])
        dropout_vals = trait.get("dropout", {}).get("values", [False, True])
        logits_vals = trait.get("logits", {}).get("values", [False, True])
        sink_vals = trait.get("sink", {}).get("values", [False])
        bpc_vals = trait.get("block_per_cu", {}).get("values", [-1, 1, 2])

        # Get hdims from problems
        hdim_q = problems[0].hdim_q
        hdim_v = problems[0].hdim_v

        seen_names = set()
        for dtype in dtypes:
            if (hdim_q, hdim_v) not in SUPPORTED_HDIMS.get(dtype, []):
                continue
            warp_classes = WARP_CLASSES.get(dtype, [(32, 32, 16)])
            is_fp8 = "fp8" in dtype or dtype in ("bf8", "mxfp8", "mxfp4")
            bk0max = K0_MAX_SUBMAX_MAP.get(hdim_q, hdim_q)

            for pipeline in pipelines:
                for bm0 in VALID_BM0:
                    for bn0 in VALID_BN0:
                        for bk0 in VALID_BK0:
                            if bk0 > hdim_q:
                                continue
                            for wm0, wn0, wk0 in warp_classes:
                                if bm0 % wm0 != 0 or bn0 % wn0 != 0 or bk0 % wk0 != 0:
                                    continue
                                rm0 = bm0 // wm0
                                bk1 = derive_bk1_fp8(bm0, bn0, bk0, hdim_q, hdim_v) if is_fp8 else derive_bk1(bm0, bn0, bk0, hdim_q, hdim_v)

                                for mode in modes:
                                    for mask in masks:
                                        for bias in biases:
                                            for lse in lse_vals:
                                                for dropout in dropout_vals:
                                                    for logits in logits_vals:
                                                        for sink in sink_vals:
                                                            for bpc in bpc_vals:
                                                                cfg = FmhaKernelConfig(
                                                                    family="fwd",
                                                                    data_type=dtype,
                                                                    mode=mode,
                                                                    hdim_q=hdim_q,
                                                                    hdim_v=hdim_v,
                                                                    pipeline=pipeline,
                                                                    gfx_arch=args.arch,
                                                                    tile_m0=bm0,
                                                                    tile_n0=bn0,
                                                                    tile_k0=bk0,
                                                                    tile_n1=hdim_v,
                                                                    tile_k1=bk1,
                                                                    tile_k0max=bk0max,
                                                                    wave_m0=rm0, wave_n0=1, wave_k0=1,
                                                                    wave_m1=rm0, wave_n1=1, wave_k1=1,
                                                                    warp_m0=wm0, warp_n0=wn0, warp_k0=wk0,
                                                                    warp_m1=wm0, warp_n1=wn0, warp_k1=wk0,
                                                                    mask=mask,
                                                                    bias=bias,
                                                                    lse=lse,
                                                                    dropout=dropout,
                                                                    logits=logits,
                                                                    sink=sink,
                                                                    block_per_cu=bpc,
                                                                )
                                                                if cfg.name not in seen_names:
                                                                    seen_names.add(cfg.name)
                                                                    all_configs.append(cfg)

        print(
            f"  Exhaustive: {len(all_configs)} total combos"
            f" (tiles × {len(dtypes)} dtypes × {len(pipelines)} pipes"
            f" × {len(modes)} modes × {len(masks)} masks × {len(biases)} biases"
            f" × {len(bpc_vals)} bpc)"
        )
    else:
        if not args.configs:
            parser.error("Config JSON(s) required for rules mode. Use --tiles exhaustive to run without.")
        for cfg_path in args.configs:
            configs = expand_sweep(cfg_path, args.arch, args.receipt)
            all_configs.extend(configs)
            print(f"  {cfg_path}: {len(configs)} kernel configs")

    if args.filter_expr or args.filter_file:
        before = len(all_configs)
        all_configs = apply_filter(all_configs, args.filter_expr, args.filter_file)
        print(f"  Filter: {before} -> {len(all_configs)} configs")

    # Remove standalone combine configs -- they are auto-paired during JIT
    all_configs = [c for c in all_configs if c.family != "fwd_splitkv_combine"]

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

    def _progress(stage, done, total):
        elapsed = time.perf_counter() - jit_t0
        pct = done * 100 // total
        print(f"\r  [{stage}] {done}/{total} ({pct}%) - {elapsed:.0f}s", end="", flush=True)
        if done == total:
            print()

    setups = setup_multiple_fmha_dispatchers(
        all_configs,
        output_dir=build_dir,
        verbose=True,
        max_workers=args.workers,
        progress_callback=_progress,
    )

    jit_time = time.perf_counter() - jit_t0
    built = sum(1 for s in setups if s.success)
    failed = len(all_configs) - built
    print(f"\n  Built {built}/{len(all_configs)} in {jit_time:.0f}s ({failed} failed)")

    # Load runners for successfully compiled kernels
    for setup in setups:
        if setup.success and setup.library_path and setup.runner is None:
            try:
                setup.runner = FmhaRunner.from_library(setup.library_path, args.arch)
            except Exception as e:
                print(f"  Warning: Failed to load runner: {e}")
                setup.success = False

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
        "bf16": np.float32,
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
        first_mask = all_configs[0].mask if all_configs else "no"
        np_dtype = dtype_map.get(first_dtype, np.float16)
        Q = (np.random.randn(*prob.q_shape()) * 0.1).astype(np_dtype)
        K = (np.random.randn(*prob.k_shape()) * 0.1).astype(np_dtype)
        V = (np.random.randn(*prob.v_shape()) * 0.1).astype(np_dtype)

        _MASK_INT = {"no": 0, "top_left": 1, "bottom_right": 2, "generic": 3}
        first_mask_int = _MASK_INT.get(first_mask, 0)

        ref = None
        if args.verify:
            ref = cpu_attention_fwd(
                Q.astype(np.float32),
                K.astype(np.float32),
                V.astype(np.float32),
                prob.scale,
                mask_type=first_mask_int,
            )

        h_str = f"H={prob.nhead_q}" if prob.nhead_q == prob.nhead_k else f"Hq={prob.nhead_q} Hk={prob.nhead_k}"
        s_str = f"S={prob.seqlen_q}" if prob.seqlen_q == prob.seqlen_k else f"Sq={prob.seqlen_q} Sk={prob.seqlen_k}"
        prob_str = f"B={prob.batch} {h_str} {s_str} D={prob.hdim_q}"
        print(f"\n  Problem [{prob_idx}]: {prob_str}")
        print(
            f"  {'Kernel':<105} {'Time(ms)':>10} {'TFLOPS':>10}"
            f" {'MaxErr':>10} {'Status':>6}"
        )
        print(f"  {'-' * 145}")

        _BIAS_INT = {"no": 0, "bias": 1, "alibi": 2}

        for config, setup in zip(all_configs, setups):
            if not setup.success or setup.runner is None:
                continue

            # Skip kernels whose hdim doesn't match the problem
            if config.hdim_q != prob.hdim_q or config.hdim_v != prob.hdim_v:
                continue

            mask_int = _MASK_INT.get(config.mask, 0)
            # Causal masks need window_right=0 (no future tokens visible)
            is_causal = config.mask in ("top_left", "bottom_right")
            is_group = config.mode == "group"

            # Map instance-builder family to runner api_family
            _FAMILY_TO_API = {
                "fwd_splitkv": "splitkv",
                "fwd_pagedkv": "pagedkv",
                "fwd_appendkv": "appendkv",
            }
            api_family = _FAMILY_TO_API.get(config.family, config.family)

            # Sweep num_splits for splitkv; non-splitkv runs once
            splits_to_try = num_splits_list if api_family == "splitkv" else [0]

            for ns in splits_to_try:
                run_kwargs = dict(
                    mask_type=mask_int,
                    bias_type=_BIAS_INT.get(config.bias, 0),
                    has_lse=int(config.lse),
                    has_dropout=int(config.dropout),
                    has_logits=int(config.logits),
                    has_sink=int(config.sink),
                    data_type=config.data_type,
                    is_group_mode=int(is_group),
                    is_v_rowmajor=int(config.vlayout == "r"),
                    api_family=api_family,
                    window_left=-1,
                    window_right=0 if is_causal else -1,
                )
                if api_family == "splitkv":
                    run_kwargs["num_splits"] = ns

                result = setup.runner.run(Q, K, V, prob, **run_kwargs)
                if not result.success:
                    continue

                # Adjust TFLOPS for causal mask (~half the ops)
                tflops = result.tflops
                if is_causal and result.time_ms > 0:
                    sq, sk = prob.seqlen_q, prob.seqlen_k
                    causal_ratio = (min(sq, sk) + 1) / (2.0 * sk)
                    tflops = prob.num_ops * causal_ratio / (result.time_ms * 1e-3) / 1e12

                max_err = 0.0
                status = "OK"
                if ref is not None and result.output is not None:
                    max_err = float(np.abs(result.output.astype(np.float32) - ref).max())
                    status = "PASS" if max_err < 0.01 else "FAIL"

                splits_tag = f"  [ns={ns}]" if api_family == "splitkv" else ""
                display_name = f"{config.name}{splits_tag}"
                print(
                    f"  {display_name:<105} {result.time_ms:>10.3f}"
                    f" {tflops:>10.2f} {max_err:>10.2e} {status:>6}"
                )

                all_results.append(
                    {
                        "kernel": config.name,
                        "dtype": config.data_type,
                        "hdim": config.hdim_q,
                        "pipeline": config.pipeline,
                        "num_splits": ns if api_family == "splitkv" else None,
                        "problem": {
                            "batch": prob.batch,
                            "nhead_q": prob.nhead_q,
                            "nhead_k": prob.nhead_k,
                            "seqlen_q": prob.seqlen_q,
                            "seqlen_k": prob.seqlen_k,
                            "hdim_q": prob.hdim_q,
                        },
                        "latency_ms": result.time_ms,
                        "tflops": tflops,
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

    if all_results:
        from collections import defaultdict

        by_problem = defaultdict(list)
        for r in all_results:
            key = json.dumps(r["problem"], sort_keys=True)
            by_problem[key].append(r)

        print("\n  Best kernel per problem:")
        for key, results in by_problem.items():
            best = max(results, key=lambda x: x["tflops"])
            prob = json.loads(key)
            ns_tag = f"  [ns={best['num_splits']}]" if best.get("num_splits") else ""
            h_str = f"H={prob['nhead_q']}" if prob['nhead_q'] == prob['nhead_k'] else f"Hq={prob['nhead_q']} Hk={prob['nhead_k']}"
            s_str = f"S={prob['seqlen_q']}" if prob['seqlen_q'] == prob['seqlen_k'] else f"Sq={prob['seqlen_q']} Sk={prob['seqlen_k']}"
            print(
                f"    B={prob['batch']} {h_str}"
                f" {s_str} D={prob['hdim_q']}"
                f" -> {best['kernel']}{ns_tag}"
                f" ({best['tflops']:.2f} TFLOPS, {best['latency_ms']:.3f} ms)"
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

    if args.log:
        from datetime import datetime

        with open(args.log, "w") as lf:
            lf.write(f"FMHA Benchmark Log - {datetime.now().isoformat()}\n")
            lf.write(f"{'=' * 80}\n\n")
            lf.write(f"Command: {' '.join(sys.argv)}\n")
            lf.write(f"Arch: {args.arch}\n")
            lf.write(f"Tiles mode: {args.tiles}\n")
            lf.write(f"Workers: {args.workers}\n")
            lf.write(f"Build dir: {build_dir}\n")
            lf.write(f"Total configs: {len(all_configs)}\n")
            lf.write(f"Built: {built}\n")
            lf.write(f"Failed: {failed}\n")
            lf.write(f"JIT time: {jit_time:.1f}s\n")
            lf.write(f"Bench time: {bench_time:.1f}s\n")
            lf.write(f"Problems: {[str(p) for p in problems]}\n\n")

            # All configs attempted
            lf.write(f"{'=' * 80}\n")
            lf.write(f"ALL CONFIGS ({len(all_configs)})\n")
            lf.write(f"{'=' * 80}\n\n")
            for i, (cfg, setup) in enumerate(zip(all_configs, setups)):
                status = "OK" if setup.success else "FAILED"
                lf.write(f"[{i:4d}] {status:6s} {cfg.name}\n")
                lf.write(f"         tile=({cfg.tile_m0},{cfg.tile_n0},{cfg.tile_k0},{cfg.tile_n1},{cfg.tile_k1},{cfg.tile_k0max})"
                         f"  warp=({cfg.warp_m0},{cfg.warp_n0},{cfg.warp_k0})"
                         f"  bpc={cfg.block_per_cu}\n")
                if not setup.success and setup.error:
                    lf.write(f"         error: {setup.error}\n")
                lf.write("\n")

            # Failed configs summary
            lf.write(f"\n{'=' * 80}\n")
            lf.write(f"FAILED CONFIGS ({failed})\n")
            lf.write(f"{'=' * 80}\n\n")
            for cfg, setup in zip(all_configs, setups):
                if not setup.success:
                    lf.write(f"  {cfg.name}\n")
                    if setup.error:
                        lf.write(f"    {setup.error}\n")

            # Benchmark results
            if all_results:
                lf.write(f"\n{'=' * 80}\n")
                lf.write(f"BENCHMARK RESULTS ({len(all_results)} measurements)\n")
                lf.write(f"{'=' * 80}\n\n")
                sorted_results = sorted(all_results, key=lambda x: -x["tflops"])
                for r in sorted_results:
                    p = r["problem"]
                    lf.write(f"  {r['tflops']:8.2f} TFLOPS  {r['latency_ms']:8.3f} ms"
                             f"  B={p['batch']} H={p['nhead_q']} S={p['seqlen_q']} D={p['hdim_q']}"
                             f"  {r['kernel']}\n")

        print(f"  Log: {args.log}")

    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
