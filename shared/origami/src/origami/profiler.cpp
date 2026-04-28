// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "origami/profiler.hpp"

#include <atomic>

namespace origami {
namespace profiler {

namespace {
std::atomic<callback_t> g_cb{nullptr};
}

void set_callback(callback_t cb) { g_cb.store(cb, std::memory_order_release); }
callback_t get_callback()        { return g_cb.load(std::memory_order_acquire); }

}  // namespace profiler
}  // namespace origami
