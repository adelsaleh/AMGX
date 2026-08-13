// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <memory_info.h>
#include <global_thread_handle.h>

namespace amgx
{

namespace
{
float bytes_to_gib(size_t bytes)
{
    return static_cast<float>(bytes / 1024.0 / 1024.0 / 1024.0);
}
}

float MemoryInfo::getTotalMemory()
{
    size_t free;
    size_t total;
    cudaMemGetInfo(&free, &total);
    return bytes_to_gib(total);
}

size_t MemoryInfo::getFreeMemory()
{
    size_t free;
    size_t total;
    cudaMemGetInfo(&free, &total);
    return static_cast<size_t>(bytes_to_gib(free));
}

float MemoryInfo::getMemoryUsage()
{
    return bytes_to_gib(memory::getDeviceMemoryStats().live_bytes);
}

float MemoryInfo::getReservedMemoryUsage()
{
    return bytes_to_gib(memory::getDeviceMemoryStats().reserved_bytes);
}

float MemoryInfo::getMaxMemoryUsage()
{
    return bytes_to_gib(memory::getDeviceMemoryStats().peak_live_bytes);
}

float MemoryInfo::getMaxReservedMemoryUsage()
{
    return bytes_to_gib(memory::getDeviceMemoryStats().peak_reserved_bytes);
}

void MemoryInfo::updateMaxMemoryUsage()
{
    memory::getDeviceMemoryStats();
}

}
