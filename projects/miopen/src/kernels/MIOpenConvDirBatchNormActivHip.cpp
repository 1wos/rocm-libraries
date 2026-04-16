/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
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

#ifndef MIOPEN_HIP_RUNTIME_COMPILE
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "float_types.h"

#define UNUSED __attribute__((__unused__))

#ifndef MLO_FILTER_STRIDE0
#define MLO_FILTER_STRIDE0 1
#endif
#ifndef MLO_FILTER_STRIDE1
#define MLO_FILTER_STRIDE1 1
#endif

#define MLO_FILTER_SZ (MLO_FILTER_SIZE1 * MLO_FILTER_SIZE0)

#define MLO_GRP_SZ0 (MLO_GRP_TILE0 * MLO_GRP_TILE1)
#define MLO_GRP_SZ1 1
#define MLO_GRP_SZ2 1
#define MLO_GRP_SZ (MLO_GRP_SZ0 * MLO_GRP_SZ1 * MLO_GRP_SZ2)
#define MLO_N_PROC_WAVES ((MLO_GRP_SZ + MLO_N_READ_PROCS - 1) / MLO_N_READ_PROCS)
#define MLO_OUT_TILE_SZ (MLO_OUT_TILE1 * MLO_OUT_TILE0)
#define MLO_ALU_TILE_SZ (MLO_ALU_VTILE1 * MLO_ALU_VTILE0)

#if MLO_IN_TILE0 < MLO_OUT_WIDTH || MLO_IN_TILE1 < MLO_OUT_HEIGHT
#define MLO_LARGE_MAP 1
#else
#define MLO_LARGE_MAP 0
#endif

#if(MLO_IN_WIDTH == MLO_OUT_WIDTH &&                                \
    (MLO_IN_WIDTH / MLO_IN_TILE0) * MLO_IN_TILE0 == MLO_IN_WIDTH && \
    MLO_IN_HEIGHT == MLO_OUT_HEIGHT &&                              \
    (MLO_IN_HEIGHT / MLO_IN_TILE1) * MLO_IN_TILE1 == MLO_IN_HEIGHT)
#define MLO_OUT_ALIGNED 1
#else
#define MLO_OUT_ALIGNED 0
#endif

#define MLO_N_ALUTILES_TOTAL ((MLO_GRP_TILE0 * MLO_GRP_TILE1) / (MLO_ALU_TILE_SZ))
#define MLO_N_ALUTILES_PERSTACK (MLO_N_ALUTILES_TOTAL / MLO_N_STACKS)
#define MLO_ALUTILES_STACK_SZ (MLO_N_ALUTILES_PERSTACK * MLO_ALU_TILE_SZ)
#define MLO_N_IN_TILES_TOTAL (MLO_N_IN_TILES_PERSTACK * MLO_N_STACKS)

#define MLO_N_OUT_TILE_BLOCKS0 ((MLO_OUT_WIDTH + MLO_IN_TILE0 - 1) / MLO_IN_TILE0)
#define MLO_N_OUT_TILE_BLOCKS1 ((MLO_OUT_HEIGHT + MLO_IN_TILE1 - 1) / MLO_IN_TILE1)
#define MLO_N_IN_PACKS ((MLO_N_INPUTS + MLO_N_IN_TILES_PERSTACK - 1) / MLO_N_IN_TILES_PERSTACK)

#define MLO_N_IN_READ (MLO_N_IN_PACKS * MLO_N_IN_TILES_PERSTACK)
#if MLO_N_IN_READ == MLO_N_INPUTS
#define MLO_INPUTS_ALIGNED 1
#else
#define MLO_INPUTS_ALIGNED 0
#endif

#define MLO_N_OUT_PACKS (MLO_N_OUTPUTS / MLO_N_OUT_TILES_PERSTACK)
#if MLO_N_OUT_PACKS * MLO_N_OUT_TILES_PERSTACK == MLO_N_OUTPUTS && \
    MLO_N_OUT_TILES_PERSTACK != MLO_N_OUTPUTS
#define MLO_OUTPUTS_ALIGNED 1
#else
#define MLO_OUTPUTS_ALIGNED 0
#endif

#define MLO_N_BATCH_PACKS (MLO_BATCH_SZ / MLO_N_STACKS)
#if MLO_N_BATCH_PACKS * MLO_N_STACKS == MLO_BATCH_SZ && MLO_N_STACKS != MLO_BATCH_SZ
#define MLO_BATCH_ALIGNED 1
#else
#define MLO_BATCH_ALIGNED 0
#endif

#if MLO_DIR_FORWARD == 1
#define MLO_IN_LCL_WIDTH \
    ((MLO_IN_TILE0 - 1) * MLO_FILTER_STRIDE0 + MLO_FILTER_SIZE0)
#define MLO_IN_LCL_HEIGHT ((MLO_IN_TILE1 - 1) * MLO_FILTER_STRIDE1 + MLO_FILTER_SIZE1)
#else
#define MLO_IN_LCL_WIDTH                                              \
    ((MLO_IN_TILE0 + MLO_FILTER_SIZE0 - 1 + MLO_FILTER_STRIDE0 - 1) / \
     MLO_FILTER_STRIDE0)
#define MLO_IN_LCL_HEIGHT \
    ((MLO_IN_TILE1 + MLO_FILTER_SIZE1 - 1 + MLO_FILTER_STRIDE1 - 1) / MLO_FILTER_STRIDE1)
#endif
#define MLO_IN_LCL_TILE_SZ (MLO_IN_LCL_WIDTH * MLO_IN_LCL_HEIGHT)
#define MLO_IN_LCL_PERSTACK_SZ (MLO_IN_LCL_TILE_SZ * MLO_N_IN_TILES_PERSTACK)
#define MLO_IN_LCL_SZ (MLO_IN_LCL_PERSTACK_SZ * MLO_N_STACKS)

#define MLO_WEIGHTS_SZ (MLO_N_OUT_TILES_PERSTACK * MLO_N_IN_TILES_PERSTACK * MLO_FILTER_SZ)

#define MLO_PVT_ACCUM_DATA_SZ (MLO_N_OUT_TILES * MLO_OUT_TILE_SZ)
#if MLO_DIR_FORWARD == 1
#define MLO_PVT_IN_WIDTH ((MLO_OUT_TILE0 - 1) * MLO_FILTER_STRIDE0 + MLO_FILTER_SIZE0)
#define MLO_PVT_IN_HEIGHT ((MLO_OUT_TILE1 - 1) * MLO_FILTER_STRIDE1 + 1)
#else
#define MLO_PVT_IN_WIDTH \
    ((MLO_OUT_TILE0 + MLO_FILTER_SIZE0 - 1 + MLO_FILTER_STRIDE0 - 1) / MLO_FILTER_STRIDE0)
#define MLO_PVT_IN_HEIGHT ((MLO_OUT_TILE1 + MLO_FILTER_STRIDE1 - 1) / MLO_FILTER_STRIDE1)
#endif

#define MLO_LCL_WEIGHTS 1

#define MLO_PADDING_SHIFT1 (MLO_FILTER_SIZE1 - MLO_FILTER_PAD1 - 1)
#define MLO_PADDING_SHIFT0 (MLO_FILTER_SIZE0 - MLO_FILTER_PAD0 - 1)

#define MLO_PADDING_FIX1 (MLO_FILTER_SIZE1 % MLO_OUT_TILE1)
#define MLO_PADDING_FIX0 (MLO_FILTER_SIZE0 % MLO_OUT_TILE0)

#if defined(__AMDGCN__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored \
    "-Wunknown-warning-option"
#pragma clang diagnostic ignored "-Wreserved-identifier"
#pragma clang diagnostic pop
#define uniform(x) __builtin_amdgcn_readfirstlane(x)
#else
#define uniform(x) (x)
#endif

// Math helper functions (HIP equivalents of math_ops.h)
inline __device__ unsigned int iDiv_legacy(unsigned int v, unsigned int d)
{
    unsigned int r = static_cast<unsigned int>(
        static_cast<float>(v) * (1.0f / static_cast<float>(d)) + 0.00001f);
    return r;
}

inline __device__ unsigned int iMod(unsigned int v, unsigned int u, unsigned int d)
{
    return v - __mul24(u, d);
}

// Data helper functions (HIP equivalents of data_ops.h)
inline __device__ void
calculateXYPos(unsigned int linPos, unsigned int width, unsigned int* x, unsigned int* y)
{
    (*y) = static_cast<unsigned int>(
        static_cast<float>(linPos) * (1.0f / static_cast<float>(width)) + 0.00001f);
    (*x) = linPos - __mul24((*y), width);
}

inline __device__ unsigned int
calculateOffset(unsigned int stride, unsigned int x, unsigned int y)
{
    return y * stride + x;
}

inline __device__ void readDataElem(unsigned int linPos,
                                    FLOAT* lcl_data,
                                    unsigned int lcl_base,
                                    UNUSED unsigned int lcl_height,
                                    unsigned int lcl_width,
                                    unsigned int lcl_stride,
                                    unsigned int lcl_y,
                                    unsigned int lcl_x,
                                    const FLOAT* gbl_data,
                                    unsigned int gbl_base,
                                    unsigned int gbl_height,
                                    unsigned int gbl_width,
                                    unsigned int gbl_stride,
                                    unsigned int gbl_y,
                                    unsigned int gbl_x,
                                    bool vis,
                                    UNUSED bool debug)
{
    unsigned int x, y;
    calculateXYPos(linPos, lcl_width, &x, &y);
    unsigned int g_x      = x + gbl_x;
    unsigned int g_y      = y + gbl_y;
    unsigned int gbl_off0 = calculateOffset(gbl_stride, g_x, g_y);
    unsigned int gbl_off  = gbl_off0 + gbl_base;

#if MLO_LARGE_MAP == 1
    unsigned int lcl_off = lcl_base + linPos;
    (void)lcl_stride;
    (void)lcl_x;
    (void)lcl_y;
#else
    unsigned int l_x     = x + lcl_x;
    unsigned int l_y     = y + lcl_y;
    unsigned int lcl_off = lcl_base + __mul24(l_y, lcl_stride) + l_x;
#endif

#if MLO_LARGE_MAP == 1
    vis &= (g_x < gbl_width && g_y < gbl_height);
#else
    (void)gbl_width;
    (void)gbl_height;
#endif
    gbl_off        = (vis) ? gbl_off : 0;
    FLOAT gbl_val  = gbl_data[gbl_off];
    gbl_val        = (vis) ? gbl_val : static_cast<FLOAT>(0);

    lcl_data[lcl_off] = gbl_val;
}

inline __device__ void readData(unsigned int lcl_id,
                                unsigned int size,
                                unsigned int lcl_p_stride,
                                FLOAT* lcl_data,
                                unsigned int lcl_base,
                                unsigned int lcl_height,
                                unsigned int lcl_width,
                                unsigned int lcl_stride,
                                unsigned int lcl_y,
                                unsigned int lcl_x,
                                const FLOAT* gbl_data,
                                unsigned int gbl_base,
                                unsigned int gbl_height,
                                unsigned int gbl_width,
                                unsigned int gbl_stride,
                                unsigned int gbl_y,
                                unsigned int gbl_x,
                                bool vis,
                                bool debug)
{
    for(unsigned int i = lcl_id; i < size; i += lcl_p_stride)
    {
        readDataElem(i,
                     lcl_data,
                     lcl_base,
                     lcl_height,
                     lcl_width,
                     lcl_stride,
                     lcl_y,
                     lcl_x,
                     gbl_data,
                     gbl_base,
                     gbl_height,
                     gbl_width,
                     gbl_stride,
                     gbl_y,
                     gbl_x,
                     vis,
                     debug);
    }
}

// Convolution helper function
static __device__ void Conv(unsigned int o_map_base,
                            unsigned int in_stg_off,
                            FLOAT* __restrict pvt_in_stage,
                            FLOAT* __restrict lcl_indata,
                            FLOAT* __restrict pvt_wei_stage,
                            FLOAT* __restrict lcl_wei,
                            float* __restrict pvt_accum)
{
    // convolution

    // over all inputs in stack
    unsigned int in_stg_off1 = in_stg_off;
    for(unsigned int i_c = 0; i_c < MLO_N_IN_TILES_PERSTACK;
        ++i_c, in_stg_off1 += MLO_IN_LCL_TILE_SZ)
    {
        // preload input
        unsigned int wei_stg_base_off =
            __mul24(o_map_base,
                    static_cast<unsigned int>(MLO_N_IN_TILES_PERSTACK * MLO_FILTER_SZ)) +
            __mul24(i_c, static_cast<unsigned int>(MLO_FILTER_SZ));
        unsigned int in_stg_off2 = in_stg_off1;
        for(unsigned int j = 0; j < MLO_PVT_IN_HEIGHT - 1; ++j,
#if MLO_DIR_FORWARD == 1
                     in_stg_off2 += MLO_IN_LCL_WIDTH
#else
                     in_stg_off2 += (((j - MLO_PADDING_SHIFT1 + MLO_PADDING_FIX1) %
                                      MLO_FILTER_STRIDE1)
                                         ? 0
                                         : MLO_IN_LCL_WIDTH)
#endif
        )
        {
            for(unsigned int i = 0; i < MLO_PVT_IN_WIDTH; ++i)
            {
                pvt_in_stage[j * MLO_PVT_IN_WIDTH + i] = lcl_indata[in_stg_off2 + i];
            }
        }

// over filter rows
#ifdef __AMDGCN__
#if MLO_FILTER_SIZE1 < 6
#pragma unroll
#elif MLO_FILTER_SIZE1 < 9
#pragma unroll 2
#endif
#endif
#if MLO_DIR_FORWARD == 1
        for(unsigned int k = 0; k < MLO_FILTER_SIZE1; ++k, in_stg_off2 += MLO_IN_LCL_WIDTH)
#else
        for(unsigned int k = 0; k < MLO_FILTER_SIZE1; ++k,
                     in_stg_off2 +=
                     (((k - MLO_PADDING_SHIFT1 + MLO_PADDING_FIX1) % MLO_FILTER_STRIDE1)
                          ? 0
                          : MLO_IN_LCL_WIDTH))
#endif
        {
            unsigned int k_act = 0;
#if MLO_DIR_FORWARD == 1
            k_act = k;
#else
            // load filter in reverse order
            k_act = MLO_FILTER_SIZE1 - 1 - k;
#endif
            // load next input row
            for(unsigned int i_pvt = 0; i_pvt < MLO_PVT_IN_WIDTH; ++i_pvt)
            {
                pvt_in_stage[(MLO_PVT_IN_HEIGHT - 1) * MLO_PVT_IN_WIDTH + i_pvt] =
                    lcl_indata[in_stg_off2 + i_pvt];
            }

            // over all outputs
            for(unsigned int o_c = 0; o_c < MLO_N_OUT_TILES; ++o_c)
            {
                unsigned int wei_stg_off = wei_stg_base_off +
                                           o_c * MLO_N_IN_TILES_PERSTACK * MLO_FILTER_SZ +
                                           k_act * MLO_FILTER_SIZE0;
                for(unsigned int i = 0; i < MLO_FILTER_SIZE0; ++i)
                {
                    pvt_wei_stage[i] = lcl_wei[wei_stg_off + i];
                }

                // actual conv
                for(unsigned int j = 0; j < MLO_OUT_TILE1; ++j)
                {
#if MLO_DIR_FORWARD == 0
                    if(((j + k + 1 - MLO_PADDING_SHIFT1 +
                         (MLO_FILTER_SIZE1 % MLO_FILTER_STRIDE1)) %
                        MLO_FILTER_STRIDE1) == 0)
#endif
                        for(unsigned int i = 0; i < MLO_OUT_TILE0; ++i)
                        {
#if MLO_DIR_FORWARD == 1
                            float sum = 0.0f;
#endif
                            for(unsigned int l = 0; l < MLO_FILTER_SIZE0; ++l)
                            {
                                unsigned int l_act = 0;
#if MLO_DIR_FORWARD == 1
                                l_act = l;
#else
                                // in reverse horizontal and vertical orders
                                l_act = MLO_FILTER_SIZE0 - 1 - l;
#endif

#if MLO_DIR_FORWARD == 1
                                sum +=
                                    static_cast<float>(
                                        pvt_in_stage[j * MLO_PVT_IN_WIDTH * MLO_FILTER_STRIDE1 +
                                                     i * MLO_FILTER_STRIDE0 + l] *
                                        pvt_wei_stage[l_act]);
#else
                                if(((i + l + 1 - MLO_PADDING_SHIFT0 +
                                     (MLO_FILTER_SIZE0 % MLO_FILTER_STRIDE0)) %
                                    MLO_FILTER_STRIDE0) == 0)
                                {
                                    pvt_accum[(o_c * MLO_OUT_TILE1 + j) * MLO_OUT_TILE0 + i] +=
                                        pvt_in_stage[(j / MLO_FILTER_STRIDE1) * MLO_PVT_IN_WIDTH +
                                                     (i + l) / MLO_FILTER_STRIDE0] *
                                        pvt_wei_stage[l_act];
                                }
#endif
                            }
#if MLO_DIR_FORWARD == 1
                            pvt_accum[(o_c * MLO_OUT_TILE1 + j) * MLO_OUT_TILE0 + i] += sum;
#endif
                        }
                }

            } // for(unsigned int o_c = 0; o_c < MLO_N_OUT_TILES; ++o_c)

            // move data up
            for(unsigned int j = 0; j < MLO_PVT_IN_HEIGHT - 1; ++j)
            {
                for(unsigned int i = 0; i < MLO_PVT_IN_WIDTH; ++i)
                {
                    pvt_in_stage[j * MLO_PVT_IN_WIDTH + i] =
                        pvt_in_stage[(j + 1) * MLO_PVT_IN_WIDTH + i];
                }
            }

        } // for(unsigned int k = 0; k < MLO_FILER_SIZE1; ...)

    } // for(unsigned int i_c = 0; i_c < MLO_N_IN_TILES_PERSTACK; ...)
}

#ifndef MLO_CONV_BIAS
#define MLO_CONV_BIAS 0
#endif

#include "activation_functions.hpp"

// if the BN / Bias ops are not used define appropriate symbols
#if !defined SPATIAL_BN && !defined PERACT_BN
#define NO_BN
#endif

extern "C" __global__ void __launch_bounds__(MLO_GRP_SZ0)
    MIOpenConvUniBatchNormActiv(
#ifdef MIOPEN_YES_ACTIV
        const FLOAT alpha,
        const FLOAT beta,
        const FLOAT gamma,
#endif
#ifndef NO_BN
        double epsilon,
#endif
        const FLOAT* __restrict in,
        FLOAT* __restrict out,
        const FLOAT* __restrict weights
#if MLO_CONV_BIAS
        ,
        const FLOAT* __restrict conv_bias
#endif
#ifndef NO_BN
        ,
        const FLOAT* __restrict bn_bias,
        const FLOAT* __restrict scale,
        const FLOAT* __restrict estimatedMean,
        const FLOAT* __restrict estimatedVariance
#endif
        )
{
    __shared__ FLOAT lcl_indata[MLO_IN_LCL_SZ];
    __shared__ FLOAT lcl_wei[MLO_WEIGHTS_SZ];
    float pvt_accum[MLO_PVT_ACCUM_DATA_SZ];
    FLOAT pvt_in_stage[MLO_PVT_IN_HEIGHT * MLO_PVT_IN_WIDTH];
    FLOAT pvt_wei_stage[MLO_FILTER_SIZE0];

    unsigned int grp_id0 = blockIdx.x;
#if MLO_N_OUT_TILE_BLOCKS0 & (MLO_N_OUT_TILE_BLOCKS0 - 1)
    unsigned int y_tile_blk = iDiv_legacy(grp_id0, MLO_N_OUT_TILE_BLOCKS0);
    unsigned int x_tile_blk = iMod(grp_id0, y_tile_blk, MLO_N_OUT_TILE_BLOCKS0);
#else
    unsigned int y_tile_blk = grp_id0 / MLO_N_OUT_TILE_BLOCKS0;
    unsigned int x_tile_blk = grp_id0 & (MLO_N_OUT_TILE_BLOCKS0 - 1);
#endif
    unsigned int o_pack = blockIdx.y; // block of outputs
    unsigned int b_pack = blockIdx.z; // batch block

    unsigned int lcl_id = threadIdx.x;
#if MLO_ALUTILES_STACK_SZ >= MLO_GRP_SZ
    unsigned int stack        = 0;
    unsigned int alu_stack_id = lcl_id;
#elif MLO_ALUTILES_STACK_SZ & (MLO_ALUTILES_STACK_SZ - 1)
    unsigned int stack        = iDiv_legacy(lcl_id, MLO_ALUTILES_STACK_SZ);
    unsigned int alu_stack_id = iMod(lcl_id, stack, MLO_ALUTILES_STACK_SZ);
#else
    unsigned int stack        = lcl_id / MLO_ALUTILES_STACK_SZ;
    unsigned int alu_stack_id = lcl_id & (MLO_ALUTILES_STACK_SZ - 1);
#if MLO_ALUTILES_STACK_SZ >= 64
    stack = uniform(stack);
#endif
#endif
// ALU plane inside stack
#if MLO_ALU_TILE_SZ & (MLO_ALU_TILE_SZ - 1)
    unsigned int alu_out_plane_id = iDiv_legacy(alu_stack_id, MLO_ALU_TILE_SZ);
    unsigned int alu_out_id =
        iMod(alu_stack_id, alu_out_plane_id, MLO_ALU_TILE_SZ);
#else
    unsigned int alu_out_plane_id = alu_stack_id / MLO_ALU_TILE_SZ;
    unsigned int alu_out_id = alu_stack_id & (MLO_ALU_TILE_SZ - 1);
#endif
// pos inside ALU tile
#if MLO_ALU_VTILE0 & (MLO_ALU_VTILE0 - 1)
    unsigned int alu_tl1 = iDiv_legacy(alu_out_id, MLO_ALU_VTILE0);
    unsigned int alu_tl0 = iMod(alu_out_id, alu_tl1, MLO_ALU_VTILE0);
#else
    unsigned int alu_tl1 = alu_out_id / MLO_ALU_VTILE0;
    unsigned int alu_tl0 = alu_out_id & (MLO_ALU_VTILE0 - 1);
#endif

    unsigned int o_map_plane =
        o_pack * MLO_N_OUT_TILES_PERSTACK; // first output maps index per full ALU plane stack
    unsigned int o_map_base = alu_out_plane_id * MLO_N_OUT_TILES; // local output map offset
    unsigned int o_map      = o_map_plane + o_map_base;           // output map index per ALU plane
    unsigned int b_index    = b_pack * MLO_N_STACKS;

#if MLO_LARGE_MAP != 1
#if MLO_N_READ_PROCS >= MLO_GRP_SZ
    unsigned int wave_id     = 0;
    unsigned int wave_lcl_id = lcl_id;
#elif MLO_N_READ_PROCS & (MLO_N_READ_PROCS - 1)
    unsigned int wave_id     = iDiv_legacy(lcl_id, MLO_N_READ_PROCS);
    unsigned int wave_lcl_id = iMod(lcl_id, wave_id, MLO_N_READ_PROCS);
#else
    unsigned int wave_id     = lcl_id / MLO_N_READ_PROCS;
    unsigned int wave_lcl_id = lcl_id & (MLO_N_READ_PROCS - 1);
#if MLO_N_READ_PROCS >= 64
    wave_id = uniform(wave_id);
#endif
#endif
#endif

#if MLO_DIR_FORWARD == 1
    unsigned int x_grp = x_tile_blk * MLO_IN_TILE0 * MLO_FILTER_STRIDE0;
    unsigned int y_grp = y_tile_blk * MLO_IN_TILE1 * MLO_FILTER_STRIDE1;
#if MLO_LARGE_MAP == 1
    unsigned int x_in_grp = x_grp - MLO_FILTER_PAD0;
    unsigned int y_in_grp = y_grp - MLO_FILTER_PAD1;
#endif
    unsigned int x_in_lcl = alu_tl0 * MLO_OUT_TILE0 * MLO_FILTER_STRIDE0;
    unsigned int y_in_lcl = alu_tl1 * MLO_OUT_TILE1 * MLO_FILTER_STRIDE1;
#else
    unsigned int x_grp = x_tile_blk * (MLO_IN_TILE0 / MLO_FILTER_STRIDE0);
    unsigned int y_grp = y_tile_blk * (MLO_IN_TILE1 / MLO_FILTER_STRIDE1);
#if MLO_LARGE_MAP == 1
    unsigned int x_in_grp = x_grp - (MLO_FILTER_PAD0 / MLO_FILTER_STRIDE0);
    unsigned int y_in_grp = y_grp - (MLO_FILTER_PAD1 / MLO_FILTER_STRIDE1);
#endif
    unsigned int x_in_lcl = alu_tl0 * (MLO_OUT_TILE0 / MLO_FILTER_STRIDE0);
    unsigned int y_in_lcl = alu_tl1 * (MLO_OUT_TILE1 / MLO_FILTER_STRIDE1);
#endif

    // base offset to read data from local input data
    unsigned int in_stg_off =
        stack * MLO_IN_LCL_PERSTACK_SZ + (y_in_lcl) * MLO_IN_LCL_WIDTH + x_in_lcl;

    unsigned int in_off = b_index * MLO_IN_BATCH_STRIDE;

#if MLO_DIR_FORWARD == 1
    unsigned int wei_off =
        __mul24(o_map_plane, static_cast<unsigned int>(MLO_N_INPUTS * MLO_FILTER_SZ));
#else
    unsigned int wei_off =
        __mul24(o_map_plane, static_cast<unsigned int>(MLO_FILTER_SZ));
#endif

#if MLO_LARGE_MAP == 0
    for(unsigned int i = lcl_id; i < MLO_IN_LCL_SZ; i += MLO_GRP_SZ)
    {
        lcl_indata[i] = 0;
    }
#endif

    for(unsigned int i = 0; i < MLO_PVT_ACCUM_DATA_SZ; ++i)
    {
        pvt_accum[i] = 0;
    }

    for(unsigned int ic = 0; ic < MLO_N_INPUTS; ic += MLO_N_IN_TILES_PERSTACK,
                     in_off += MLO_IN_CHANNEL_STRIDE * MLO_N_IN_TILES_PERSTACK,
                     wei_off += MLO_N_IN_TILES_PERSTACK * MLO_FILTER_SZ
#if MLO_DIR_FORWARD == 0
                                                    * MLO_N_OUTPUTS
#endif
    )
    {
        __syncthreads();

        // small map has been read in full continuously into the LDS buffer within padded rect,
        // padding has been done on initialization.
        // large map calculates padding on the fly and fills it with 0.

#if 1 // all inputs

#if MLO_LARGE_MAP == 1
        unsigned int in_lcl_off1 = 0;
        unsigned int in_off1     = in_off;
        for(unsigned int i_b = 0; i_b < MLO_N_STACKS;
            ++i_b, in_off1 += MLO_IN_BATCH_STRIDE, in_lcl_off1 += MLO_IN_LCL_PERSTACK_SZ)
        {
            bool vis = true;
#if MLO_BATCH_ALIGNED == 0
            vis &= (b_index + i_b < MLO_BATCH_SZ);
#endif

            // over all inputs in stack
            unsigned int in_off2     = in_off1;
            unsigned int in_lcl_off2 = in_lcl_off1;
            for(unsigned int i_c = 0; i_c < MLO_N_IN_TILES_PERSTACK;
                ++i_c, in_off2 += MLO_IN_CHANNEL_STRIDE, in_lcl_off2 += MLO_IN_LCL_TILE_SZ)
            {
#if MLO_INPUTS_ALIGNED == 0
                vis &= (ic + i_c < MLO_N_INPUTS);
#endif

                unsigned int elem_id      = lcl_id;
                unsigned int lcl_p_stride = MLO_GRP_SZ0;
                unsigned int lcl_base     = 0;
                unsigned int lcl_y        = 0;
                unsigned int lcl_x        = 0;
                unsigned int gbl_base     = in_off2;

                readData(elem_id,
                         (MLO_IN_LCL_HEIGHT * MLO_IN_LCL_WIDTH),
                         lcl_p_stride,
                         &lcl_indata[in_lcl_off2],
                         lcl_base,
                         MLO_IN_LCL_HEIGHT,
                         MLO_IN_LCL_WIDTH,
                         MLO_IN_LCL_WIDTH,
                         lcl_y,
                         lcl_x,
                         &in[0],
                         gbl_base,
                         MLO_IN_HEIGHT,
                         MLO_IN_WIDTH,
                         MLO_IN_STRIDE,
                         y_in_grp,
                         x_in_grp,
                         vis,
                         false);
            }
        }
#else
        for(unsigned int i = wave_id; i < MLO_N_IN_TILES_TOTAL; i += MLO_N_PROC_WAVES)
        {
#if MLO_N_IN_TILES_PERSTACK & (MLO_N_IN_TILES_PERSTACK - 1)
            unsigned int i_b = iDiv_legacy(i, MLO_N_IN_TILES_PERSTACK);
            unsigned int i_c = iMod(i, i_b, MLO_N_IN_TILES_PERSTACK);
#else
            unsigned int i_b = i / MLO_N_IN_TILES_PERSTACK;
            unsigned int i_c = i & (MLO_N_IN_TILES_PERSTACK - 1);
#endif

            bool vis = true;

#if MLO_BATCH_ALIGNED == 0
            vis &= (b_index + i_b < MLO_BATCH_SZ);
#endif

#if MLO_INPUTS_ALIGNED == 0
            vis &= (ic + i_c < MLO_N_INPUTS);
#endif
            unsigned int in_off2 =
                in_off + i_b * MLO_IN_BATCH_STRIDE + i_c * MLO_IN_CHANNEL_STRIDE;
            unsigned int in_lcl_off2 =
                i_b * MLO_IN_LCL_PERSTACK_SZ + i_c * MLO_IN_LCL_TILE_SZ;

            unsigned int elem_id      = wave_lcl_id;
            unsigned int lcl_p_stride = MLO_N_READ_PROCS;
            unsigned int lcl_base     = 0;
#if MLO_DIR_FORWARD == 1
            unsigned int lcl_y = MLO_FILTER_PAD1;
            unsigned int lcl_x = MLO_FILTER_PAD0;
#else
            unsigned int lcl_y = (MLO_FILTER_PAD1 / MLO_FILTER_STRIDE0);
            unsigned int lcl_x = (MLO_FILTER_PAD0 / MLO_FILTER_STRIDE1);
#endif
            unsigned int gbl_base = in_off2;

            readData(elem_id,
                     (MLO_IN_HEIGHT * MLO_IN_WIDTH),
                     lcl_p_stride,
                     &lcl_indata[in_lcl_off2],
                     lcl_base,
                     MLO_IN_HEIGHT,
                     MLO_IN_WIDTH,
                     MLO_IN_LCL_WIDTH,
                     lcl_y,
                     lcl_x,
                     &in[0],
                     gbl_base,
                     MLO_IN_HEIGHT,
                     MLO_IN_WIDTH,
                     MLO_IN_STRIDE,
                     y_grp,
                     x_grp,
                     vis,
                     false);
        }
#endif

        // read inputs and weights
        // put weights into LDS

#if 1 // only weights

        for(unsigned int i = lcl_id; i < MLO_WEIGHTS_SZ; i += MLO_GRP_SZ)
        {
#if MLO_DIR_FORWARD == 1
// here is [tops][bottoms]
#if(MLO_N_IN_TILES_PERSTACK * MLO_FILTER_SZ) & ((MLO_N_IN_TILES_PERSTACK * MLO_FILTER_SZ) - 1)
            unsigned int lcl_o = iDiv_legacy(i, (MLO_N_IN_TILES_PERSTACK * MLO_FILTER_SZ));
            unsigned int gbl_i = iMod(i, lcl_o, (MLO_N_IN_TILES_PERSTACK * MLO_FILTER_SZ));
#else
            unsigned int lcl_o = i / (MLO_N_IN_TILES_PERSTACK * MLO_FILTER_SZ);
            unsigned int gbl_i = i & ((MLO_N_IN_TILES_PERSTACK * MLO_FILTER_SZ) - 1);
#endif
            unsigned int gbl_we_off = wei_off + lcl_o * MLO_N_INPUTS * MLO_FILTER_SZ + gbl_i;
            bool within_range =
                gbl_we_off < static_cast<unsigned int>(MLO_N_OUTPUTS * MLO_N_INPUTS * MLO_FILTER_SZ);

            gbl_we_off     = (within_range) ? gbl_we_off : 0;
            FLOAT wei      = weights[gbl_we_off];
            wei            = (within_range) ? wei : static_cast<FLOAT>(0);
            lcl_wei[i]     = wei;
#else
// outputs are bottoms(inputs))
// inputs are tops(outputs)
#if(MLO_N_OUT_TILES_PERSTACK * MLO_FILTER_SZ) & \
    ((MLO_N_OUT_TILES_PERSTACK * MLO_FILTER_SZ) - 1)
            unsigned int lcl_o = iDiv_legacy(i, (MLO_N_OUT_TILES_PERSTACK * MLO_FILTER_SZ));
            unsigned int gbl_i = iMod(i, lcl_o, (MLO_N_OUT_TILES_PERSTACK * MLO_FILTER_SZ));
#else
            unsigned int lcl_o = i / (MLO_N_OUT_TILES_PERSTACK * MLO_FILTER_SZ);
            unsigned int gbl_i = i & ((MLO_N_OUT_TILES_PERSTACK * MLO_FILTER_SZ) - 1);
#endif
#if MLO_FILTER_SZ & (MLO_FILTER_SZ - 1)
            unsigned int lcl_c = iDiv_legacy(gbl_i, MLO_FILTER_SZ);
            unsigned int lcl_i = iMod(gbl_i, lcl_c, MLO_FILTER_SZ);
#else
            unsigned int lcl_c = gbl_i / MLO_FILTER_SZ;
            unsigned int lcl_i = gbl_i & (MLO_FILTER_SZ - 1);
#endif

            unsigned int lcl_we_off =
                __mul24(__mul24(lcl_c, static_cast<unsigned int>(MLO_N_IN_TILES_PERSTACK)) + lcl_o,
                        static_cast<unsigned int>(MLO_FILTER_SZ)) +
                lcl_i;
            unsigned int gbl_we_off =
                __mul24(__mul24(lcl_o, static_cast<unsigned int>(MLO_N_OUTPUTS)) + lcl_c,
                        static_cast<unsigned int>(MLO_FILTER_SZ)) +
                wei_off + lcl_i;
            bool within_range =
                gbl_we_off < static_cast<unsigned int>(MLO_N_OUTPUTS * MLO_N_INPUTS * MLO_FILTER_SZ);
            gbl_we_off          = (within_range) ? gbl_we_off : 0;
            FLOAT wei           = weights[gbl_we_off];
            wei                 = (within_range) ? wei : static_cast<FLOAT>(0);
            lcl_wei[lcl_we_off] = wei;

#endif
        }

#endif

        // over all batch stacks

#endif // all input

        __syncthreads();

        // convolution
        Conv(o_map_base, in_stg_off, pvt_in_stage, lcl_indata, pvt_wei_stage, lcl_wei, pvt_accum);
    }
// write results out
#if MLO_DIR_FORWARD == 1
#if MLO_FILTER_STRIDE0 == 1
    unsigned int x_out_grp = x_grp;
#else
    unsigned int x_out_grp = x_tile_blk * MLO_IN_TILE0;
#endif
#if MLO_FILTER_STRIDE1 == 1
    unsigned int y_out_grp = y_grp;
#else
    unsigned int y_out_grp = y_tile_blk * MLO_IN_TILE1;
#endif
#else
    unsigned int x_out_grp = x_grp * MLO_FILTER_STRIDE0;
    unsigned int y_out_grp = y_grp * MLO_FILTER_STRIDE1;
#endif
    unsigned int x_out_lcl = alu_tl0 * MLO_OUT_TILE0;
    unsigned int y_out_lcl = alu_tl1 * MLO_OUT_TILE1;

    unsigned int out_off = (b_index + stack) * MLO_OUT_BATCH_STRIDE +
                           o_map * MLO_OUT_CHANNEL_STRIDE +
                           (y_out_grp + y_out_lcl) * MLO_OUT_STRIDE + x_out_grp + x_out_lcl;

    FLOAT conv_res   = static_cast<FLOAT>(0);
    FLOAT bn_res     = static_cast<FLOAT>(0);
    FLOAT output_res = static_cast<FLOAT>(0);
#ifdef MIOPEN_YES_ACTIV
    FLOAT actv_res;
#endif

// over all local stacks
#if MLO_BATCH_ALIGNED == 0
    if(b_index + stack < MLO_BATCH_SZ)
#endif
    {

        // over all local outputs
        unsigned int out_off1 = out_off;
        for(unsigned int o = 0; o < MLO_N_OUT_TILES; ++o, out_off1 += MLO_OUT_CHANNEL_STRIDE)
        {
#ifndef NO_BN
#ifdef SPATIAL_BN
            unsigned int c_i    = o_map + o;
            FLOAT pmean         = estimatedMean[c_i];
            FLOAT pvar          = estimatedVariance[c_i];
            FLOAT pscale        = scale[c_i];
            FLOAT pbias         = bn_bias[c_i];
            FLOAT pinvVariance  = rsqrt(fabs(pvar + epsilon));
#endif
#endif

#if MLO_OUTPUTS_ALIGNED == 0
            if(o_map + o < MLO_N_OUTPUTS)
#endif
            {
                // over output tile
                unsigned int out_off2 = out_off1;
#if MLO_OUT_TILE0 == 1
                for(unsigned int j = 0;
                    j < MLO_OUT_TILE1 && y_out_grp + y_out_lcl + j < MLO_OUT_HEIGHT;
                    ++j, out_off2 += MLO_OUT_STRIDE)
                {
                    for(unsigned int i = 0;
                        i < MLO_OUT_TILE0 && x_out_grp + x_out_lcl + i < MLO_OUT_WIDTH &&
                        out_off2 + i < MLO_OUT_BATCH_STRIDE * MLO_BATCH_SZ;
                        ++i)
                    {
                        if(1)
                        {
#else
                for(unsigned int j = 0; j < MLO_OUT_TILE1; ++j, out_off2 += MLO_OUT_STRIDE)
                {
                    if(y_out_grp + y_out_lcl + j < MLO_OUT_HEIGHT)
                        for(unsigned int i = 0; i < MLO_OUT_TILE0; ++i)
                        {
                            if(x_out_grp + x_out_lcl + i < MLO_OUT_WIDTH &&
                               out_off2 + i < MLO_OUT_BATCH_STRIDE * MLO_BATCH_SZ)
                            {
#endif

#ifndef NO_BN
#ifdef PERACT_BN
                            unsigned int chw_i = (out_off2 + i)
#if MLO_BATCH_SZ > 1
                                                 % (MLO_OUT_BATCH_STRIDE)
#endif
                                ;
                            FLOAT pmean         = estimatedMean[chw_i];
                            FLOAT pvar          = estimatedVariance[chw_i];
                            FLOAT pscale        = scale[chw_i];
                            FLOAT pbias         = bn_bias[chw_i];
                            FLOAT pinvVariance  = rsqrt(fabs(pvar + epsilon));
#endif
#endif
                            conv_res = static_cast<FLOAT>(
                                pvt_accum[o * MLO_OUT_TILE_SZ + j * MLO_OUT_TILE0 + i])
#if MLO_CONV_BIAS
                                + conv_bias[o_map + o]
#endif
                                ;

#ifdef NO_BN
                            bn_res = conv_res;
#else
                            bn_res = pscale * (conv_res - pmean) * pinvVariance + pbias;
#endif

#if(MIOPEN_NRN_OP_ID > 0)
#ifdef MIOPEN_YES_ACTIV
                            {
                                FLOAT actv_arr[1];
                                const FLOAT bn_arr[1] = {bn_res};
                                ActivationFunction(actv_arr, bn_arr, gamma, beta, alpha);
                                actv_res = actv_arr[0];
                            }
                            output_res = actv_res;
#endif

#else
                            output_res = bn_res;
#endif
                            out[out_off2 + i] = output_res;
                        }
                    }
                }
            }
        }
    }
}
