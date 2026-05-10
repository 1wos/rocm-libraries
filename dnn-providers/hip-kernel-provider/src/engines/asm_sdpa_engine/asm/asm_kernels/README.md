# ASM Kernel Binaries — Provenance

This directory holds vendored GPU code-object (`.co`) binaries from the AITER
project for both forward and backward FMHA kernels. The forward and backward
snapshots were taken from **different** AITER commits (see table below).

| Kernel group | AITER commit | Platforms | Per-arch SOURCE.md |
|---|---|---|---|
| `fmha_v3_fwd` (gfx942) | `17d4a33b6f9535e820353ebc6217769efc3766d6` | gfx942 (MI300X/MI300A) | — |
| `fmha_v3_bwd` (gfx942, gfx950) | `9522048dc10de20ba9dcda1c0a3f640867e7a586` | gfx942, gfx950 | `gfx942/fmha_v3_bwd/SOURCE.md`, `gfx950/fmha_v3_bwd/SOURCE.md` |

The forward POC snapshot (`17d4a33b`) pre-dates the backward snapshot
(`9522048`). There is no single unified AITER commit that contains both the
forward and backward kernels used here. When performing an AITER refresh,
both groups must be updated together (or the commit mismatch documented
explicitly) so reviewers can verify provenance from a single source of truth.

## Local overrides

`gfx942/fmha_v3_bwd/bwd_hd128_odo_bf16.co` is a **local override** — it is
not the binary from the `9522048` AITER snapshot. See
`gfx942/fmha_v3_bwd/SOURCE.md` for SHA256, builder identity, and rationale.
No other overrides exist in this tree.

## Directory layout notes

- **Forward kernels** (`fmha_v3_fwd/`): AITER has separate `MI300/` and `MI308/` subdirectories
  because MI308 (MI300A APU) has different fp8 support. We only include MI300 for the POC.
- **Backward kernels** (`fmha_v3_bwd/`): AITER stores these flat (no MI300/MI308 split) because
  backward kernels only support bf16/fp16 (no fp8), so the same binary works on both MI300X and
  MI300A. SHA256 manifests for each arch are recorded in the per-arch `SOURCE.md` files.
