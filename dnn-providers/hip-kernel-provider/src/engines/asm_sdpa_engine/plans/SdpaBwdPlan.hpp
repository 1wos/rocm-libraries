// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

#include "HipKernelHandle.hpp"
#include "HipKernelSettings.hpp"
#include "SdpaBwdParams.hpp"
#include "SdpaKernelUtils.hpp"

#include <optional>

namespace asm_sdpa_engine
{

/**
 * @brief SDPA backward kernel plan.
 *
 * Orchestrates 2 or 3 ASM kernels for the backward pass:
 *   1. ODO       — D reduction: D[b,h,i] = sum_j(O * dO)
 *   2. DQDKDV    — Main gradients: dQ (FP32 via A32, or direct BF16 via A16), dK, dV
 *   3. DQ_CONVERT — Post-processing: FP32 dQ → BF16  (A32 path only)
 *
 * When the resolved DQDKDV row uses A16 (atomic32==0), `postKernel` is
 * std::nullopt and the third launch + dq_acc workspace are both skipped.
 */
class SdpaBwdPlan : public hipdnn_plugin_sdk::IPlan<HipKernelHandle>
{
public:
    SdpaBwdPlan(HipModuleGuard odoKernel,
                HipModuleGuard dqdkdvKernel,
                std::optional<HipModuleGuard> postKernel,
                SdpaBwdParams params);

    ~SdpaBwdPlan() override = default;

    SdpaBwdPlan(const SdpaBwdPlan&) = delete;
    SdpaBwdPlan& operator=(const SdpaBwdPlan&) = delete;
    SdpaBwdPlan(SdpaBwdPlan&&) noexcept = default;
    SdpaBwdPlan& operator=(SdpaBwdPlan&&) noexcept = default;

    size_t getWorkspaceSize(const HipKernelHandle& handle) const override;

    void execute(const HipKernelHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override;

private:
    HipModuleGuard _odoKernel;
    HipModuleGuard _dqdkdvKernel;
    std::optional<HipModuleGuard> _postKernel; // nullopt when A16 (dq_convert not needed)
    SdpaBwdParams _params;
};

} // namespace asm_sdpa_engine
