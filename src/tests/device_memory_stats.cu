// SPDX-FileCopyrightText: 2011 - 2026 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include "unit_test.h"
#include <global_thread_handle.h>
#include <resources.h>

namespace amgx
{

DECLARE_UNITTEST_BEGIN(DeviceMemoryStats);

void run()
{
    Resources resources;
    const memory::DeviceMemoryStats before = memory::getDeviceMemoryStats();
    const size_t allocation_bytes = 64 * 1024;
    void *ptr = NULL;

    const cudaError_t allocation_status = memory::cudaMallocAsync(&ptr, allocation_bytes);
    UNITTEST_ASSERT_TRUE_DESC(
        "AMGX tracked-memory test allocation failed", allocation_status == cudaSuccess);

    const memory::DeviceMemoryStats allocated = memory::getDeviceMemoryStats();
    UNITTEST_ASSERT_TRUE_DESC(
        "AMGX live-byte counter did not increase after allocation",
        allocated.live_bytes >= before.live_bytes + allocation_bytes);
    UNITTEST_ASSERT_TRUE_DESC(
        "AMGX used bytes exceed AMGX-held bytes",
        allocated.live_bytes <= allocated.reserved_bytes);
    UNITTEST_ASSERT_TRUE_DESC(
        "AMGX peak-live counter is below current live bytes",
        allocated.peak_live_bytes >= allocated.live_bytes);
    UNITTEST_ASSERT_TRUE_DESC(
        "AMGX peak-reserved counter is below current reserved bytes",
        allocated.peak_reserved_bytes >= allocated.reserved_bytes);

    const cudaError_t free_status = memory::cudaFreeAsync(ptr);
    UNITTEST_ASSERT_TRUE_DESC(
        "AMGX tracked-memory test free failed", free_status == cudaSuccess);
    memory::cudaFreeWait();

    const memory::DeviceMemoryStats released = memory::getDeviceMemoryStats();
    UNITTEST_ASSERT_EQUAL_DESC(
        "AMGX live-byte counter did not return to its baseline after free",
        before.live_bytes, released.live_bytes);
    UNITTEST_ASSERT_TRUE_DESC(
        "AMGX peak-live counter did not retain the allocation high-water mark",
        released.peak_live_bytes >= allocated.live_bytes);
}

DECLARE_UNITTEST_END(DeviceMemoryStats);

DeviceMemoryStats<TemplateMode<AMGX_mode_dDDI>::Type> DeviceMemoryStats_instance_mode_dDDI;

} // namespace amgx
