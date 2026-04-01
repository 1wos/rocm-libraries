// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace ck_tile::core::arch::mma {

/**
 * @enum WmmaCtrlFlags
 * @brief Common wmma control flags for gfx11 and gfx12
 */
enum struct WmmaCtrlFlags : bool
{
    // Only has an effect on gfx11 when the accumulator is 16-bit
    // Determines which half of the 32-bit accum register to use
    // Low = bits [15:0]
    // High = bits[31:16]
    LOW  = false,
    HIGH = true,
};

/**
 * @class DefaultWmmaFlags
 * @brief Generates default WMMA control flags based on data types.
 * @tparam ADataType Data type of matrix A
 * @tparam BDataType Data type of matrix B
 * @tparam CDataType Data type of the accumulator
 */
template <typename ADataType, typename BDataType, typename CDataType>
struct DefaultWmmaCtrlFlags
{
    constexpr static bool Clamp = false;

    // Generate default flags for accumulator destination bits.
    // Only used if accumulation size is 16-bit in gfx11
    constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
};

} // namespace ck_tile::core::arch::mma

// Include the architecture-specific WMMA implementations and traits
#include "wmma_gfx11.hpp"
#include "wmma_gfx12.hpp"
#include "wmma_selector.hpp"
#include "wmma_traits.hpp"
#include "wmma_transforms.hpp"
