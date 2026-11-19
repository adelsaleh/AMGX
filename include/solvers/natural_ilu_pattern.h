// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>
#include <vector>

namespace amgx { namespace multicolor_ilu_solver {

// Host symbolic ILU(k), in the supplied block-row order. Values are never read.
// Original entries (including explicit zeros) have level 0; eliminating j<i
// creates level(i,k)=min(level(i,k),level(i,j)+level(j,k)+1), for k>j.
// This is an elimination-level pattern, not a power of the adjacency graph.
struct NaturalILUPattern
{
    std::vector<int> rows, columns, levels, source_to_factor;
};

inline void validate_natural_ilu_input(int n, const int *rows, const int *columns)
{
    if (n <= 0 || rows == nullptr || columns == nullptr || rows[0] != 0)
        throw std::invalid_argument("Natural ILU requires nonempty zero-based CSR/BSR");
    for (int i = 0; i < n; ++i)
    {
        if (rows[i] < 0 || rows[i+1] < rows[i])
            throw std::invalid_argument("Natural ILU row pointers are not monotone");
        int previous = -1;
        bool diagonal = false;
        for (int p = rows[i]; p < rows[i+1]; ++p)
        {
            const int j = columns[p];
            if (j <= previous || j < 0 || j >= n)
                throw std::invalid_argument("Natural ILU requires sorted, unique, in-range block columns");
            previous = j;
            diagonal = diagonal || j == i;
        }
        if (!diagonal)
            throw std::invalid_argument("Natural ILU requires a stored diagonal block in every row");
    }
}

inline NaturalILUPattern natural_ilu_pattern(
    int n, const int *rows, const int *columns, int fill_level,
    std::size_t max_blocks = static_cast<std::size_t>(std::numeric_limits<int>::max()-1),
    const std::function<void(int, std::size_t)> &progress = {})
{
    if (fill_level < 0 || fill_level == std::numeric_limits<int>::max())
        throw std::invalid_argument("Natural ILU fill level must be nonnegative and below INT_MAX");
    validate_natural_ilu_input(n, rows, columns);
    max_blocks = std::min(max_blocks, static_cast<std::size_t>(std::numeric_limits<int>::max()-1));
    if (static_cast<std::size_t>(rows[n]) > max_blocks)
        throw std::length_error("Natural ILU input already exceeds the factor storage bound");
    NaturalILUPattern out;
    out.rows.resize(n+1, 0);
    out.source_to_factor.resize(rows[n]);
    out.columns.reserve(rows[n]);
    out.levels.reserve(rows[n]);
    const int absent = fill_level + 1;
    std::vector<int> level(n, absent), touched, upper_start(n);
    for (int i = 0; i < n; ++i)
    {
        touched.clear();
        std::priority_queue<int, std::vector<int>, std::greater<int>> pivots;
        for (int p = rows[i]; p < rows[i+1]; ++p)
        {
            const int j = columns[p];
            level[j] = 0;
            touched.push_back(j);
            if (j < i) pivots.push(j);
        }
        while (!pivots.empty())
        {
            const int j = pivots.top();
            pivots.pop();
            if (level[j] >= fill_level) continue;
            for (int p = upper_start[j]; p < out.rows[j+1]; ++p)
            {
                const int k = out.columns[p];
                const std::int64_t candidate = static_cast<std::int64_t>(level[j]) + out.levels[p] + 1;
                if (candidate > fill_level || candidate >= level[k]) continue;
                if (level[k] == absent)
                {
                    touched.push_back(k);
                    if (k < i) pivots.push(k);
                }
                level[k] = static_cast<int>(candidate);
            }
        }
        if (touched.size() > max_blocks-out.columns.size())
            throw std::length_error("Natural ILU fill exceeds the factor storage bound");
        std::sort(touched.begin(), touched.end());
        int source = rows[i];
        for (int j : touched)
        {
            const int position = static_cast<int>(out.columns.size());
            out.columns.push_back(j);
            out.levels.push_back(level[j]);
            if (source < rows[i+1] && columns[source] == j)
                out.source_to_factor[source++] = position;
            if (j == i) upper_start[i] = position+1;
            level[j] = absent;
        }
        if (source != rows[i+1])
            throw std::logic_error("Natural ILU pattern lost an original entry");
        out.rows[i+1] = static_cast<int>(out.columns.size());
        if (progress && (i+1 == n || (i+1)%16384 == 0))
            progress(i+1, out.columns.size());
    }
    return out;
}

}} // namespace amgx::multicolor_ilu_solver
