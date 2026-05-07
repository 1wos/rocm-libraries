// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Regression test for PyTorch issue 177834 / MIOpen RNN workspace int-overflow.
//
// Bidirectional 4-layer LSTM with batch=1000, seq=600, hidden=128 produces a workspace
// tensor whose flat fp16-element count (~5.5e9) exceeds INT_MAX. Pre-fix, the four
// packed-RNN paths in src/ocl/rnnocl.cpp constructed the workspace TensorDescriptor from
// std::vector<int>; the assignment `sp_size[2] = workSpaceSize / GetTypeSize(...)`
// silently truncated to a negative int, and ConvertLengthsOrThrow rejected the result
// with "Lengths must be > 0", returning miopenStatusBadParm. Post-fix the descriptor
// vectors are std::vector<size_t> and the call succeeds.

#include <gtest/gtest.h>
#include <miopen/miopen.h>

#include <hip/hip_runtime.h>

#include <climits>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "get_handle.hpp"
#include "gtest_desc_guard.hpp"
#include "../workspace.hpp"

namespace {

struct LargeRnnConfig
{
    int batch_per_step              = 1000;
    int seq_len                     = 600;
    int input_size                  = 60;
    int hidden_size                 = 128;
    int num_layers                  = 4;
    miopenRNNDirectionMode_t dir    = miopenRNNbidirection;
    miopenRNNMode_t mode            = miopenLSTM;
    miopenRNNInputMode_t input_mode = miopenRNNlinear;
    miopenRNNBiasMode_t bias_mode   = miopenRNNwithBias;
    miopenRNNAlgo_t algo            = miopenRNNdefault;
    miopenDataType_t dtype          = miopenHalf; // fp16: 2 bytes/element
};

// Build per-step xDescs for the configured LSTM. Each xDesc is a 2D
// {batch_per_step, input_size} tensor with row-major strides.
void MakeXDescs(const LargeRnnConfig& cfg, std::vector<miopenTensorDescriptor_t>& xDescs)
{
    xDescs.assign(cfg.seq_len, nullptr);
    for(int i = 0; i < cfg.seq_len; ++i)
    {
        ASSERT_EQ(miopenCreateTensorDescriptor(&xDescs[i]), miopenStatusSuccess);
        std::array<int, 2> lens = {cfg.batch_per_step, cfg.input_size};
        ASSERT_EQ(miopenSetTensorDescriptor(xDescs[i], cfg.dtype, 2, lens.data(), nullptr),
                  miopenStatusSuccess);
    }
}

void DestroyDescs(std::vector<miopenTensorDescriptor_t>& descs)
{
    for(auto* d : descs)
    {
        if(d != nullptr)
        {
            miopenDestroyTensorDescriptor(d);
        }
    }
    descs.clear();
}

} // namespace

struct GPU_LSTM_LargeWorkspace_FP16 : public ::testing::TestWithParam<int>
{
};

// Workspace-size getter must succeed and report > INT_MAX bytes.
// Always runs (no GPU allocation). Verifies the test config still triggers the
// would-be-overflow condition: if a future MIOpen change shrinks the workspace
// below INT_MAX bytes for this config the assertion will fire and someone needs
// to choose a larger LargeRnnConfig.
TEST_P(GPU_LSTM_LargeWorkspace_FP16, GetWorkspaceSizeOverflowsInt)
{
    auto&& handle = get_handle();
    LargeRnnConfig cfg;

    RNNDescGuard rnn;
    ASSERT_EQ(rnn.getStatus(), miopenStatusSuccess);
    ASSERT_EQ(miopenSetRNNDescriptor(rnn,
                                     cfg.hidden_size,
                                     cfg.num_layers,
                                     cfg.input_mode,
                                     cfg.dir,
                                     cfg.mode,
                                     cfg.bias_mode,
                                     cfg.algo,
                                     cfg.dtype),
              miopenStatusSuccess);

    std::vector<miopenTensorDescriptor_t> xDescs;
    MakeXDescs(cfg, xDescs);

    size_t ws_size = 0;
    auto status    = miopenGetRNNWorkspaceSize(&handle, rnn, cfg.seq_len, xDescs.data(), &ws_size);

    EXPECT_EQ(status, miopenStatusSuccess);
    EXPECT_GT(ws_size, static_cast<size_t>(INT_MAX))
        << "Workspace size " << ws_size << " no longer overflows int. "
        << "Adjust LargeRnnConfig to keep this regression meaningful.";

    DestroyDescs(xDescs);
}

// Full ForwardInference path. Pre-fix this returned miopenStatusBadParm
// with "Lengths must be > 0"; post-fix it must return miopenStatusSuccess.
TEST_P(GPU_LSTM_LargeWorkspace_FP16, ForwardInferenceSucceedsWhenWorkspaceExceedsInt)
{
#ifdef _WIN32
    GTEST_SKIP() << "Skipped on Windows: > 8 GB workspace exceeds typical Windows runner VRAM";
#endif

    auto&& handle = get_handle();
    LargeRnnConfig cfg;

    RNNDescGuard rnn;
    ASSERT_EQ(rnn.getStatus(), miopenStatusSuccess);
    ASSERT_EQ(miopenSetRNNDescriptor(rnn,
                                     cfg.hidden_size,
                                     cfg.num_layers,
                                     cfg.input_mode,
                                     cfg.dir,
                                     cfg.mode,
                                     cfg.bias_mode,
                                     cfg.algo,
                                     cfg.dtype),
              miopenStatusSuccess);

    std::vector<miopenTensorDescriptor_t> xDescs;
    std::vector<miopenTensorDescriptor_t> yDescs;
    MakeXDescs(cfg, xDescs);

    // y per-step desc: {batch_per_step, hidden_size * directions}
    const int directions = (cfg.dir == miopenRNNbidirection) ? 2 : 1;
    const int y_vec      = cfg.hidden_size * directions;
    yDescs.assign(cfg.seq_len, nullptr);
    for(int i = 0; i < cfg.seq_len; ++i)
    {
        ASSERT_EQ(miopenCreateTensorDescriptor(&yDescs[i]), miopenStatusSuccess);
        std::array<int, 2> lens = {cfg.batch_per_step, y_vec};
        ASSERT_EQ(miopenSetTensorDescriptor(yDescs[i], cfg.dtype, 2, lens.data(), nullptr),
                  miopenStatusSuccess);
    }

    // Hidden-state desc: 3D {num_layers * directions, batch_per_step, hidden_size}
    TensorDescGuard hxDesc;
    ASSERT_EQ(hxDesc.getStatus(), miopenStatusSuccess);
    {
        std::array<int, 3> lens = {
            cfg.num_layers * directions, cfg.batch_per_step, cfg.hidden_size};
        ASSERT_EQ(miopenSetTensorDescriptor(hxDesc, cfg.dtype, 3, lens.data(), nullptr),
                  miopenStatusSuccess);
    }

    // Workspace size
    size_t ws_size = 0;
    ASSERT_EQ(miopenGetRNNWorkspaceSize(&handle, rnn, cfg.seq_len, xDescs.data(), &ws_size),
              miopenStatusSuccess);
    ASSERT_GT(ws_size, static_cast<size_t>(INT_MAX));

    // Weight buffer size + descriptor
    size_t w_size = 0;
    ASSERT_EQ(miopenGetRNNParamsSize(&handle, rnn, xDescs[0], &w_size, cfg.dtype),
              miopenStatusSuccess);
    TensorDescGuard wDesc;
    ASSERT_EQ(wDesc.getStatus(), miopenStatusSuccess);
    ASSERT_EQ(miopenGetRNNParamsDescriptor(&handle, rnn, xDescs[0], wDesc, cfg.dtype),
              miopenStatusSuccess);

    // Use the official MIOpen sizing APIs (matches MIOpenDriver pattern) so that
    // buffer sizes account for all internal alignment / padding requirements.
    std::size_t x_bytes = 0;
    std::size_t y_bytes = 0;
    std::size_t h_bytes = 0;
    ASSERT_EQ(miopenGetRNNInputTensorSize(&handle, rnn, cfg.seq_len, xDescs.data(), &x_bytes),
              miopenStatusSuccess);
    ASSERT_EQ(miopenGetRNNInputTensorSize(&handle, rnn, cfg.seq_len, yDescs.data(), &y_bytes),
              miopenStatusSuccess);
    ASSERT_EQ(miopenGetRNNHiddenTensorSize(&handle, rnn, cfg.seq_len, xDescs.data(), &h_bytes),
              miopenStatusSuccess);

    Workspace x_buf{x_bytes};
    Workspace y_buf{y_bytes};
    Workspace w_buf{w_size};
    Workspace hx_buf{h_bytes};
    Workspace cx_buf{h_bytes};
    Workspace hy_buf{h_bytes};
    Workspace cy_buf{h_bytes};
    Workspace ws_buf{ws_size};

    // Zero-initialise weights and inputs for determinism. We don't care
    // about numerical output, only the descriptor-construction status code.
    ASSERT_EQ(hipMemset(w_buf.ptr(), 0, w_size), hipSuccess);
    ASSERT_EQ(hipMemset(hx_buf.ptr(), 0, h_bytes), hipSuccess);
    ASSERT_EQ(hipMemset(cx_buf.ptr(), 0, h_bytes), hipSuccess);
    ASSERT_EQ(hipMemset(x_buf.ptr(), 0, x_bytes), hipSuccess);

    auto status = miopenRNNForwardInference(&handle,
                                            rnn,
                                            cfg.seq_len,
                                            xDescs.data(),
                                            x_buf.ptr(),
                                            hxDesc,
                                            hx_buf.ptr(),
                                            hxDesc,
                                            cx_buf.ptr(),
                                            wDesc,
                                            w_buf.ptr(),
                                            yDescs.data(),
                                            y_buf.ptr(),
                                            hxDesc,
                                            hy_buf.ptr(),
                                            hxDesc,
                                            cy_buf.ptr(),
                                            ws_buf.ptr(),
                                            ws_size);

    EXPECT_EQ(status, miopenStatusSuccess)
        << "miopenRNNForwardInference returned " << status
        << "; pre-fix this returned miopenStatusBadParm with \"Lengths must be > 0\".";

    DestroyDescs(xDescs);
    DestroyDescs(yDescs);
}

INSTANTIATE_TEST_SUITE_P(Standard, GPU_LSTM_LargeWorkspace_FP16, testing::Values(0));
