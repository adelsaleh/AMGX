// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace amgx
{
class MemoryInfo
{
    public:
        static float getTotalMemory();
        static size_t getFreeMemory();
        static float getMemoryUsage();
        static float getReservedMemoryUsage();
        static float getMaxMemoryUsage();
        static float getMaxReservedMemoryUsage();
        static void updateMaxMemoryUsage();
};
}
