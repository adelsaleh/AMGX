// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <classical/block_graph.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

#include <amgx_types/util.h>
#include <csr_multiply.h>
#include <global_thread_handle.h>
#include <util.h>
#include <vector_thrust_allocator.h>

namespace amgx
{
namespace classical
{
namespace
{

template <typename IndexType>
__global__ void fill_sequence_kernel(IndexType *sequence, IndexType count)
{
    for (IndexType i = threadIdx.x + blockIdx.x * blockDim.x;
         i < count; i += blockDim.x * gridDim.x)
    {
        sequence[i] = i;
    }
}

template <typename IndexType, typename ValueType>
__global__ void build_frobenius_graph_kernel(
    const IndexType *row_offsets, const IndexType *col_indices,
    const ValueType *block_values, IndexType num_rows, int block_size,
    ValueType *graph_values, int *missing_diagonal)
{
    typedef typename types::PODTypes<ValueType>::type PodType;

    for (IndexType row = threadIdx.x + blockIdx.x * blockDim.x;
         row < num_rows; row += blockDim.x * gridDim.x)
    {
        IndexType diagonal = IndexType(-1);
        PodType off_diagonal_sum = PodType(0);

        for (IndexType entry = row_offsets[row]; entry < row_offsets[row + 1]; ++entry)
        {
            if (col_indices[entry] == row)
            {
                diagonal = entry;
                continue;
            }

            PodType squared_norm = PodType(0);
            const size_t value_begin = static_cast<size_t>(entry) * block_size;

            for (int k = 0; k < block_size; ++k)
            {
                const PodType magnitude = types::util<ValueType>::abs(
                                              block_values[value_begin + k]);
                squared_norm += magnitude * magnitude;
            }

            const PodType norm = sqrt(squared_norm);
            graph_values[entry] = ValueType(-norm);
            off_diagonal_sum += norm;
        }

        if (diagonal >= 0)
        {
            graph_values[diagonal] = ValueType(
                                         off_diagonal_sum > PodType(0)
                                         ? off_diagonal_sum : PodType(1));
        }
        else
        {
            atomicExch(missing_diagonal, 1);
        }
    }
}

template <typename IndexType, typename ValueType>
__global__ void compute_diagonal_frobenius_norms_kernel(
    const IndexType *row_offsets, const IndexType *col_indices,
    const ValueType *block_values, IndexType num_rows, int block_size,
    typename types::PODTypes<ValueType>::type *diagonal_norms,
    int *invalid_diagonal)
{
    typedef typename types::PODTypes<ValueType>::type PodType;

    for (IndexType row = threadIdx.x + blockIdx.x * blockDim.x;
         row < num_rows; row += blockDim.x * gridDim.x)
    {
        IndexType diagonal = IndexType(-1);

        for (IndexType entry = row_offsets[row]; entry < row_offsets[row + 1]; ++entry)
        {
            if (col_indices[entry] == row)
            {
                diagonal = entry;
                break;
            }
        }

        if (diagonal < 0)
        {
            atomicCAS(invalid_diagonal, 0, 1);
            continue;
        }

        PodType squared_norm = PodType(0);
        const size_t value_begin = static_cast<size_t>(diagonal) * block_size;

        for (int k = 0; k < block_size; ++k)
        {
            const PodType magnitude = types::util<ValueType>::abs(
                                          block_values[value_begin + k]);
            squared_norm += magnitude * magnitude;
        }

        const PodType norm = sqrt(squared_norm);
        diagonal_norms[row] = norm;

        if (!(norm > PodType(0)))
        {
            atomicCAS(invalid_diagonal, 0, 2);
        }
    }
}

template <typename IndexType, typename ValueType>
__global__ void build_diagonal_normalized_frobenius_graph_kernel(
    const IndexType *row_offsets, const IndexType *col_indices,
    const ValueType *block_values, IndexType num_rows, int block_size,
    const typename types::PODTypes<ValueType>::type *diagonal_norms,
    ValueType *graph_values)
{
    typedef typename types::PODTypes<ValueType>::type PodType;

    for (IndexType row = threadIdx.x + blockIdx.x * blockDim.x;
         row < num_rows; row += blockDim.x * gridDim.x)
    {
        IndexType diagonal = IndexType(-1);
        PodType off_diagonal_sum = PodType(0);

        for (IndexType entry = row_offsets[row]; entry < row_offsets[row + 1]; ++entry)
        {
            const IndexType column = col_indices[entry];

            if (column == row)
            {
                diagonal = entry;
                continue;
            }

            PodType squared_norm = PodType(0);
            const size_t value_begin = static_cast<size_t>(entry) * block_size;

            for (int k = 0; k < block_size; ++k)
            {
                const PodType magnitude = types::util<ValueType>::abs(
                                              block_values[value_begin + k]);
                squared_norm += magnitude * magnitude;
            }

            const PodType norm = sqrt(squared_norm)
                                 / sqrt(diagonal_norms[row]
                                        * diagonal_norms[column]);
            graph_values[entry] = ValueType(-norm);
            off_diagonal_sum += norm;
        }

        // The preceding kernel has already validated the internal diagonal.
        graph_values[diagonal] = ValueType(
                                     off_diagonal_sum > PodType(0)
                                     ? off_diagonal_sum : PodType(1));
    }
}

template <typename IndexType, typename ValueType>
__global__ void lift_scalar_transfer_kernel(const ValueType *scalar_values,
        IndexType num_nonzero_blocks, int block_dim, ValueType *block_values)
{
    const int block_size = block_dim * block_dim;
    const size_t value_count = static_cast<size_t>(num_nonzero_blocks) * block_size;

    for (size_t value = threadIdx.x + static_cast<size_t>(blockIdx.x) * blockDim.x;
         value < value_count;
         value += static_cast<size_t>(blockDim.x) * gridDim.x)
    {
        const IndexType block = static_cast<IndexType>(value / block_size);
        const int local = static_cast<int>(value % block_size);
        const int local_row = local / block_dim;
        const int local_col = local % block_dim;
        block_values[value] = local_row == local_col
                              ? scalar_values[block]
                              : types::util<ValueType>::get_zero();
    }
}

template <typename IndexType>
__global__ void validate_transfer_structure_kernel(
    const IndexType *row_offsets, const IndexType *column_indices,
    IndexType num_block_rows, IndexType num_nonzero_blocks,
    IndexType num_block_cols, int *invalid)
{
    for (IndexType row = threadIdx.x + blockIdx.x * blockDim.x;
         row < num_block_rows; row += blockDim.x * gridDim.x)
    {
        const IndexType begin = row_offsets[row];
        const IndexType end = row_offsets[row + 1];

        if (begin < 0 || begin > end || end > num_nonzero_blocks)
        {
            if (atomicCAS(invalid, 0, 1) == 0)
            {
                invalid[1] = static_cast<int>(row);
                invalid[2] = static_cast<int>(begin);
                invalid[3] = static_cast<int>(end);
            }
            continue;
        }

        for (IndexType entry = begin; entry < end; ++entry)
        {
            const IndexType column = column_indices[entry];

            if (column < 0 || column >= num_block_cols)
            {
                if (atomicCAS(invalid, 0, 2) == 0)
                {
                    invalid[1] = static_cast<int>(row);
                    invalid[2] = static_cast<int>(entry);
                    invalid[3] = static_cast<int>(column);
                }
            }
        }
    }

    if (threadIdx.x == 0 && blockIdx.x == 0
            && row_offsets[num_block_rows] != num_nonzero_blocks)
    {
        if (atomicCAS(invalid, 0, 3) == 0)
        {
            invalid[1] = static_cast<int>(num_block_rows);
            invalid[2] = static_cast<int>(row_offsets[num_block_rows]);
            invalid[3] = static_cast<int>(num_nonzero_blocks);
        }
    }
}

template <typename IndexType, typename MatrixValue, typename VectorValue>
__global__ void identity_transfer_spmv_kernel(
    const IndexType *row_offsets, const IndexType *column_indices,
    const MatrixValue *values, IndexType num_block_rows, int block_dim,
    const VectorValue *input, VectorValue *output)
{
    const size_t scalar_rows = static_cast<size_t>(num_block_rows) * block_dim;
    const int block_size = block_dim * block_dim;

    for (size_t scalar_row = threadIdx.x
                             + static_cast<size_t>(blockIdx.x) * blockDim.x;
         scalar_row < scalar_rows;
         scalar_row += static_cast<size_t>(blockDim.x) * gridDim.x)
    {
        const IndexType block_row = static_cast<IndexType>(scalar_row / block_dim);
        const int component = static_cast<int>(scalar_row % block_dim);
        VectorValue sum = types::util<VectorValue>::get_zero();

        for (IndexType entry = row_offsets[block_row];
             entry < row_offsets[block_row + 1]; ++entry)
        {
            const IndexType block_col = column_indices[entry];
            const MatrixValue weight =
                values[static_cast<size_t>(entry) * block_size
                       + component * block_dim + component];
            sum += weight
                   * input[static_cast<size_t>(block_col) * block_dim + component];
        }

        output[scalar_row] = sum;
    }
}

template <typename IndexType, typename ValueType>
__global__ void extract_scalar_transfer_kernel(const ValueType *block_values,
        IndexType num_nonzero_blocks, int block_size, ValueType *scalar_values)
{
    for (IndexType block = threadIdx.x + blockIdx.x * blockDim.x;
         block < num_nonzero_blocks; block += blockDim.x * gridDim.x)
    {
        // Entry (0,0) equals the scalar weight in either dense-block layout.
        scalar_values[block] = block_values[static_cast<size_t>(block) * block_size];
    }
}


template <typename ValueType>
__device__ __forceinline__ int dense_block_offset(
    int row, int col, int block_dim, bool row_major)
{
    return row_major ? row * block_dim + col : col * block_dim + row;
}

template <typename IndexType>
__device__ __forceinline__ IndexType find_block_entry(
    const IndexType *row_offsets, const IndexType *col_indices,
    IndexType row, IndexType col)
{
    for (IndexType entry = row_offsets[row]; entry < row_offsets[row + 1]; ++entry)
    {
        if (col_indices[entry] == col)
        {
            return entry;
        }
    }

    return IndexType(-1);
}

template <typename IndexType, typename ValueType>
__global__ void invert_diagonal_blocks_kernel(
    const IndexType *row_offsets, const IndexType *diagonal,
    const ValueType *block_values, IndexType num_rows, int block_dim,
    bool row_major, typename types::PODTypes<ValueType>::type pivot_tolerance,
    ValueType *work, ValueType *inverse, int *invalid)
{
    typedef typename types::PODTypes<ValueType>::type PodType;
    const int block_size = block_dim * block_dim;

    for (IndexType row = threadIdx.x + blockIdx.x * blockDim.x;
         row < num_rows; row += blockDim.x * gridDim.x)
    {
        const IndexType diagonal_entry = diagonal[row];

        if (diagonal_entry < row_offsets[row]
                || diagonal_entry >= row_offsets[row + 1])
        {
            if (atomicCAS(invalid, 0, 1) == 0)
            {
                invalid[1] = static_cast<int>(row);
            }

            continue;
        }

        const size_t work_begin = static_cast<size_t>(row) * block_size;
        const size_t value_begin =
            static_cast<size_t>(diagonal_entry) * block_size;
        PodType block_norm = PodType(0);

        for (int local_row = 0; local_row < block_dim; ++local_row)
        {
            for (int local_col = 0; local_col < block_dim; ++local_col)
            {
                const int logical = local_row * block_dim + local_col;
                const ValueType value =
                    block_values[value_begin + dense_block_offset<ValueType>(
                        local_row, local_col, block_dim, row_major)];
                work[work_begin + logical] = value;
                inverse[work_begin + logical] =
                    local_row == local_col
                    ? types::util<ValueType>::get_one()
                    : types::util<ValueType>::get_zero();
                const PodType magnitude =
                    types::util<ValueType>::abs(value);

                if (magnitude > block_norm)
                {
                    block_norm = magnitude;
                }
            }
        }

        const PodType threshold =
            pivot_tolerance
            * (block_norm > PodType(1) ? block_norm : PodType(1));

        for (int pivot_col = 0; pivot_col < block_dim; ++pivot_col)
        {
            int pivot_row = pivot_col;
            PodType pivot_abs = types::util<ValueType>::abs(
                work[work_begin + pivot_col * block_dim + pivot_col]);

            for (int candidate = pivot_col + 1; candidate < block_dim; ++candidate)
            {
                const PodType candidate_abs = types::util<ValueType>::abs(
                    work[work_begin + candidate * block_dim + pivot_col]);

                if (candidate_abs > pivot_abs)
                {
                    pivot_abs = candidate_abs;
                    pivot_row = candidate;
                }
            }

            if (pivot_abs <= threshold)
            {
                if (atomicCAS(invalid, 0, 2) == 0)
                {
                    invalid[1] = static_cast<int>(row);
                }

                break;
            }

            if (pivot_row != pivot_col)
            {
                for (int col = 0; col < block_dim; ++col)
                {
                    ValueType temporary =
                        work[work_begin + pivot_col * block_dim + col];
                    work[work_begin + pivot_col * block_dim + col] =
                        work[work_begin + pivot_row * block_dim + col];
                    work[work_begin + pivot_row * block_dim + col] = temporary;
                    temporary =
                        inverse[work_begin + pivot_col * block_dim + col];
                    inverse[work_begin + pivot_col * block_dim + col] =
                        inverse[work_begin + pivot_row * block_dim + col];
                    inverse[work_begin + pivot_row * block_dim + col] = temporary;
                }
            }

            const ValueType pivot =
                work[work_begin + pivot_col * block_dim + pivot_col];

            for (int col = 0; col < block_dim; ++col)
            {
                work[work_begin + pivot_col * block_dim + col] =
                    work[work_begin + pivot_col * block_dim + col] / pivot;
                inverse[work_begin + pivot_col * block_dim + col] =
                    inverse[work_begin + pivot_col * block_dim + col] / pivot;
            }

            for (int eliminate_row = 0;
                 eliminate_row < block_dim; ++eliminate_row)
            {
                if (eliminate_row == pivot_col)
                {
                    continue;
                }

                const ValueType factor =
                    work[work_begin + eliminate_row * block_dim + pivot_col];

                for (int col = 0; col < block_dim; ++col)
                {
                    work[work_begin + eliminate_row * block_dim + col] -=
                        factor
                        * work[work_begin + pivot_col * block_dim + col];
                    inverse[work_begin + eliminate_row * block_dim + col] -=
                        factor
                        * inverse[work_begin + pivot_col * block_dim + col];
                }
            }
        }
    }
}

template <typename IndexType, typename ValueType>
__global__ void build_symmetric_inverse_diagonal_graph_kernel(
    const IndexType *row_offsets, const IndexType *col_indices,
    const ValueType *block_values, const ValueType *inverse_diagonal,
    IndexType num_rows, int block_dim, bool row_major,
    ValueType *graph_values, int *missing_reverse)
{
    typedef typename types::PODTypes<ValueType>::type PodType;
    const int block_size = block_dim * block_dim;

    for (IndexType row = threadIdx.x + blockIdx.x * blockDim.x;
         row < num_rows; row += blockDim.x * gridDim.x)
    {
        IndexType diagonal = IndexType(-1);
        PodType off_diagonal_sum = PodType(0);
        const size_t inverse_row_begin = static_cast<size_t>(row) * block_size;

        for (IndexType entry = row_offsets[row]; entry < row_offsets[row + 1]; ++entry)
        {
            const IndexType column = col_indices[entry];

            if (column == row)
            {
                diagonal = entry;
                continue;
            }

            const IndexType reverse = find_block_entry(
                row_offsets, col_indices, column, row);

            if (reverse < 0)
            {
                atomicExch(missing_reverse, 1);
                continue;
            }

            const size_t inverse_column_begin =
                static_cast<size_t>(column) * block_size;
            const size_t value_begin = static_cast<size_t>(entry) * block_size;
            const size_t reverse_begin = static_cast<size_t>(reverse) * block_size;
            PodType left_squared = PodType(0);
            PodType right_squared = PodType(0);

            for (int local_row = 0; local_row < block_dim; ++local_row)
            {
                for (int local_col = 0; local_col < block_dim; ++local_col)
                {
                    ValueType left = types::util<ValueType>::get_zero();
                    ValueType right = types::util<ValueType>::get_zero();

                    for (int inner = 0; inner < block_dim; ++inner)
                    {
                        left += inverse_diagonal[
                                    inverse_row_begin
                                    + local_row * block_dim + inner]
                                * block_values[
                                    value_begin
                                    + dense_block_offset<ValueType>(
                                        inner, local_col, block_dim, row_major)];
                        right += inverse_diagonal[
                                     inverse_column_begin
                                     + local_row * block_dim + inner]
                                 * block_values[
                                     reverse_begin
                                     + dense_block_offset<ValueType>(
                                         inner, local_col, block_dim, row_major)];
                    }

                    const PodType left_magnitude =
                        types::util<ValueType>::abs(left);
                    const PodType right_magnitude =
                        types::util<ValueType>::abs(right);
                    left_squared += left_magnitude * left_magnitude;
                    right_squared += right_magnitude * right_magnitude;
                }
            }

            const PodType weight =
                sqrt(sqrt(left_squared) * sqrt(right_squared));
            graph_values[entry] = ValueType(-weight);
            off_diagonal_sum += weight;
        }

        graph_values[diagonal] = ValueType(
                                     off_diagonal_sum > PodType(0)
                                     ? off_diagonal_sum : PodType(1));
    }
}

template <typename IndexType, typename ValueType>
__global__ void smooth_dense_transfer_values_kernel(
    const IndexType *A_row_offsets, const IndexType *A_col_indices,
    const ValueType *A_values, IndexType num_rows, int block_dim,
    bool row_major, const IndexType *P_row_offsets,
    const IndexType *P_col_indices, const ValueType *P_values,
    const ValueType *inverse_diagonal, const int *cf_map,
    bool constant_vector,
    typename types::PODTypes<ValueType>::type smoothing_weight,
    ValueType *smoothed_P_values)
{
    const int block_size = block_dim * block_dim;

    for (IndexType row = blockIdx.x; row < num_rows; row += gridDim.x)
    {
        const IndexType P_begin = P_row_offsets[row];
        const IndexType P_end = P_row_offsets[row + 1];
        const size_t row_value_count =
            static_cast<size_t>(P_end - P_begin) * block_size;

        for (size_t row_value = threadIdx.x; row_value < row_value_count;
             row_value += blockDim.x)
        {
            const IndexType P_entry =
                P_begin + static_cast<IndexType>(row_value / block_size);
            const int logical = static_cast<int>(row_value % block_size);
            const int local_row = logical / block_dim;
            const int local_col = logical % block_dim;
            const IndexType coarse_col = P_col_indices[P_entry];
            if (constant_vector && cf_map[row] >= 0)
            {
                // Exact coarse-point injection keeps P full column rank.
                // Preserving only P*1=1 would not enforce this by itself.
                smoothed_P_values[
                    static_cast<size_t>(P_entry) * block_size
                    + dense_block_offset<ValueType>(
                        local_row, local_col, block_dim, row_major)] =
                    coarse_col == cf_map[row] && local_row == local_col
                    ? types::util<ValueType>::get_one()
                    : types::util<ValueType>::get_zero();
                continue;
            }
            ValueType diagonal_correction =
                types::util<ValueType>::get_zero();

            for (int inverse_col = 0; inverse_col < block_dim; ++inverse_col)
            {
                ValueType AP_value = types::util<ValueType>::get_zero();

                for (IndexType A_entry = A_row_offsets[row];
                     A_entry < A_row_offsets[row + 1]; ++A_entry)
                {
                    const IndexType neighbor = A_col_indices[A_entry];
                    const IndexType neighbor_P_entry = find_block_entry(
                        P_row_offsets, P_col_indices, neighbor, coarse_col);

                    if (neighbor_P_entry < 0)
                    {
                        continue;
                    }

                    for (int inner = 0; inner < block_dim; ++inner)
                    {
                        AP_value +=
                            A_values[static_cast<size_t>(A_entry) * block_size
                                     + dense_block_offset<ValueType>(
                                         inverse_col, inner, block_dim,
                                         row_major)]
                            * P_values[
                                static_cast<size_t>(neighbor_P_entry)
                                    * block_size
                                + dense_block_offset<ValueType>(
                                    inner, local_col, block_dim, row_major)];
                    }
                }

                diagonal_correction +=
                    inverse_diagonal[
                        static_cast<size_t>(row) * block_size
                        + local_row * block_dim + inverse_col]
                    * AP_value;
            }

            const size_t value_offset =
                static_cast<size_t>(P_entry) * block_size
                + dense_block_offset<ValueType>(
                    local_row, local_col, block_dim, row_major);
            smoothed_P_values[value_offset] =
                P_values[value_offset]
                - diagonal_correction * smoothing_weight;
        }
    }
}

template <typename IndexType>
__global__ void build_coarse_row_lookup_kernel(
    const int *cf_map, IndexType num_rows, IndexType num_coarse_rows,
    IndexType *coarse_rows, int *invalid)
{
    for (IndexType row = threadIdx.x + blockIdx.x * blockDim.x;
         row < num_rows; row += blockDim.x * gridDim.x)
    {
        const int coarse = cf_map[row];

        if (coarse >= 0)
        {
            if (static_cast<IndexType>(coarse) >= num_coarse_rows)
            {
                atomicExch(invalid, 1);
            }
            else
            {
                coarse_rows[coarse] = row;
            }
        }
    }
}

template <typename IndexType, typename ValueType>
__global__ void build_extended_i_raw_values_kernel(
    const IndexType *A_row_offsets, const IndexType *A_col_indices,
    const ValueType *A_values, IndexType num_rows, int block_dim,
    bool row_major, const int *cf_map, const bool *s_con,
    const IndexType *P_row_offsets, const IndexType *P_col_indices,
    IndexType num_coarse_rows, const IndexType *coarse_rows,
    const ValueType *inverse_diagonal, ValueType *raw_values,
    int *invalid)
{
    const int block_size = block_dim * block_dim;

    for (IndexType row = blockIdx.x; row < num_rows; row += gridDim.x)
    {
        const IndexType P_begin = P_row_offsets[row];
        const IndexType P_end = P_row_offsets[row + 1];
        const size_t row_value_count =
            static_cast<size_t>(P_end - P_begin) * block_size;

        for (size_t row_value = threadIdx.x; row_value < row_value_count;
             row_value += blockDim.x)
        {
            const IndexType P_entry =
                P_begin + static_cast<IndexType>(row_value / block_size);
            const int logical = static_cast<int>(row_value % block_size);
            const int local_row = logical / block_dim;
            const int local_col = logical % block_dim;
            const size_t output_offset =
                static_cast<size_t>(P_entry) * block_size
                + dense_block_offset<ValueType>(
                    local_row, local_col, block_dim, row_major);

            if (cf_map[row] >= 0)
            {
                raw_values[output_offset] =
                    local_row == local_col
                    ? types::util<ValueType>::get_one()
                    : types::util<ValueType>::get_zero();
                continue;
            }

            const IndexType coarse_col = P_col_indices[P_entry];

            if (coarse_col < 0 || coarse_col >= num_coarse_rows)
            {
                raw_values[output_offset] =
                    types::util<ValueType>::get_zero();
                atomicExch(invalid, 1);
                continue;
            }

            const IndexType coarse_row = coarse_rows[coarse_col];

            if (coarse_row < 0 || coarse_row >= num_rows)
            {
                raw_values[output_offset] =
                    types::util<ValueType>::get_zero();
                atomicExch(invalid, 1);
                continue;
            }

            ValueType raw = types::util<ValueType>::get_zero();
            const IndexType direct = find_block_entry(
                A_row_offsets, A_col_indices, row, coarse_row);

            if (direct >= 0)
            {
                raw = A_values[
                    static_cast<size_t>(direct) * block_size
                    + dense_block_offset<ValueType>(
                        local_row, local_col, block_dim, row_major)];
            }

            for (IndexType A_entry = A_row_offsets[row];
                 A_entry < A_row_offsets[row + 1]; ++A_entry)
            {
                const IndexType fine_neighbor = A_col_indices[A_entry];

                if (fine_neighbor == row || !s_con[A_entry]
                        || cf_map[fine_neighbor] >= 0)
                {
                    continue;
                }

                const IndexType neighbor_to_coarse = find_block_entry(
                    A_row_offsets, A_col_indices,
                    fine_neighbor, coarse_row);

                if (neighbor_to_coarse < 0)
                {
                    continue;
                }

                ValueType path = types::util<ValueType>::get_zero();

                for (int left_col = 0; left_col < block_dim; ++left_col)
                {
                    const ValueType left = A_values[
                        static_cast<size_t>(A_entry) * block_size
                        + dense_block_offset<ValueType>(
                            local_row, left_col, block_dim, row_major)];

                    for (int right_row = 0;
                         right_row < block_dim; ++right_row)
                    {
                        path +=
                            left
                            * inverse_diagonal[
                                static_cast<size_t>(fine_neighbor) * block_size
                                + left_col * block_dim + right_row]
                            * A_values[
                                static_cast<size_t>(neighbor_to_coarse)
                                    * block_size
                                + dense_block_offset<ValueType>(
                                    right_row, local_col, block_dim,
                                    row_major)];
                    }
                }

                raw -= path;
            }

            raw_values[output_offset] = raw;
        }
    }
}

template <typename IndexType, typename ValueType>
__global__ void left_scale_extended_i_values_kernel(
    const IndexType *P_row_offsets, const int *cf_map,
    const ValueType *raw_values, const ValueType *inverse_diagonal,
    IndexType num_rows, int block_dim, bool row_major,
    ValueType *scaled_values)
{
    const int block_size = block_dim * block_dim;

    for (IndexType row = blockIdx.x; row < num_rows; row += gridDim.x)
    {
        const IndexType P_begin = P_row_offsets[row];
        const IndexType P_end = P_row_offsets[row + 1];
        const size_t row_value_count =
            static_cast<size_t>(P_end - P_begin) * block_size;

        for (size_t row_value = threadIdx.x; row_value < row_value_count;
             row_value += blockDim.x)
        {
            const IndexType P_entry =
                P_begin + static_cast<IndexType>(row_value / block_size);
            const int logical = static_cast<int>(row_value % block_size);
            const int local_row = logical / block_dim;
            const int local_col = logical % block_dim;
            const size_t output_offset =
                static_cast<size_t>(P_entry) * block_size
                + dense_block_offset<ValueType>(
                    local_row, local_col, block_dim, row_major);

            if (cf_map[row] >= 0)
            {
                scaled_values[output_offset] = raw_values[output_offset];
                continue;
            }

            ValueType scaled = types::util<ValueType>::get_zero();

            for (int inner = 0; inner < block_dim; ++inner)
            {
                scaled -=
                    inverse_diagonal[
                        static_cast<size_t>(row) * block_size
                        + local_row * block_dim + inner]
                    * raw_values[
                        static_cast<size_t>(P_entry) * block_size
                        + dense_block_offset<ValueType>(
                            inner, local_col, block_dim, row_major)];
            }

            scaled_values[output_offset] = scaled;
        }
    }
}

template <typename ValueType>
__global__ void copy_dense_transfer_values_kernel(
    const ValueType *source, size_t value_count, ValueType *target)
{
    for (size_t value = threadIdx.x
                        + static_cast<size_t>(blockIdx.x) * blockDim.x;
         value < value_count;
         value += static_cast<size_t>(blockDim.x) * gridDim.x)
    {
        target[value] = source[value];
    }
}

template <typename IndexType, typename ValueType>
__global__ void compute_dense_row_correction_kernel(
    const IndexType *P_row_offsets, const ValueType *block_P_values,
    IndexType num_rows, int block_dim, bool row_major, bool constant_vector,
    ValueType *row_correction)
{
    const int block_size = block_dim * block_dim;
    const size_t correction_count = static_cast<size_t>(num_rows) * block_size;

    for (size_t correction = threadIdx.x
                             + static_cast<size_t>(blockIdx.x) * blockDim.x;
         correction < correction_count;
         correction += static_cast<size_t>(blockDim.x) * gridDim.x)
    {
        const IndexType row =
            static_cast<IndexType>(correction / block_size);
        const int logical = static_cast<int>(correction % block_size);
        const int local_row = logical / block_dim;
        const int local_col = logical % block_dim;
        ValueType row_sum = types::util<ValueType>::get_zero();

        for (IndexType entry = P_row_offsets[row];
             entry < P_row_offsets[row + 1]; ++entry)
        {
            const int begin_col = constant_vector ? 0 : local_col;
            const int end_col = constant_vector ? block_dim : local_col + 1;
            for (int col = begin_col; col < end_col; ++col)
            {
                row_sum +=
                    block_P_values[static_cast<size_t>(entry) * block_size
                                   + dense_block_offset<ValueType>(
                                       local_row, col, block_dim, row_major)];
            }
        }

        const ValueType target =
            constant_vector || local_row == local_col
            ? types::util<ValueType>::get_one()
            : types::util<ValueType>::get_zero();
        // Scalar Poisson in nodal coordinates needs only sum_c P_ic * 1 = 1.
        // Distribute its defect equally over block columns. The application
        // kernel distributes it over coarse neighbors with normalized weights.
        typedef typename types::PODTypes<ValueType>::type PodType;
        row_correction[correction] = constant_vector
            ? (target - row_sum) / static_cast<PodType>(block_dim)
            : target - row_sum;
    }
}

template <typename IndexType, typename ValueType>
__global__ void apply_dense_row_correction_kernel(
    const IndexType *P_row_offsets, const ValueType *scalar_P_values,
    IndexType num_rows, int block_dim, bool row_major,
    typename types::PODTypes<ValueType>::type denominator_tolerance,
    const ValueType *row_correction, ValueType *block_P_values)
{
    typedef typename types::PODTypes<ValueType>::type PodType;
    const int block_size = block_dim * block_dim;

    for (IndexType row = blockIdx.x; row < num_rows; row += gridDim.x)
    {
        const IndexType P_begin = P_row_offsets[row];
        const IndexType P_end = P_row_offsets[row + 1];
        const IndexType row_entries = P_end - P_begin;

        if (row_entries == 0)
        {
            continue;
        }

        ValueType weight_sum = types::util<ValueType>::get_zero();

        for (IndexType entry = P_begin; entry < P_end; ++entry)
        {
            weight_sum += scalar_P_values[entry];
        }

        const bool use_scalar_weights =
            types::util<ValueType>::abs(weight_sum) > denominator_tolerance;
        const size_t row_value_count =
            static_cast<size_t>(row_entries) * block_size;

        for (size_t row_value = threadIdx.x; row_value < row_value_count;
             row_value += blockDim.x)
        {
            const IndexType P_entry =
                P_begin + static_cast<IndexType>(row_value / block_size);
            const int logical = static_cast<int>(row_value % block_size);
            const int local_row = logical / block_dim;
            const int local_col = logical % block_dim;
            const ValueType alpha =
                use_scalar_weights
                ? scalar_P_values[P_entry] / weight_sum
                : types::util<ValueType>::get_one()
                  / static_cast<PodType>(row_entries);
            block_P_values[static_cast<size_t>(P_entry) * block_size
                           + dense_block_offset<ValueType>(
                               local_row, local_col, block_dim, row_major)] +=
                alpha
                * row_correction[
                    static_cast<size_t>(row) * block_size + logical];
        }
    }
}

template <typename IndexType, typename ValueType>
__global__ void invert_dense_transfer_row_sums_kernel(
    const IndexType *P_row_offsets, const ValueType *P_values,
    IndexType num_rows, int block_dim, bool row_major,
    typename types::PODTypes<ValueType>::type pivot_tolerance,
    ValueType *work, ValueType *inverse, int *invalid)
{
    typedef typename types::PODTypes<ValueType>::type PodType;
    const int block_size = block_dim * block_dim;

    for (IndexType row = threadIdx.x + blockIdx.x * blockDim.x;
         row < num_rows; row += blockDim.x * gridDim.x)
    {
        const size_t work_begin = static_cast<size_t>(row) * block_size;
        PodType block_norm = PodType(0);

        for (int local_row = 0; local_row < block_dim; ++local_row)
        {
            for (int local_col = 0; local_col < block_dim; ++local_col)
            {
                const int logical = local_row * block_dim + local_col;
                ValueType value = types::util<ValueType>::get_zero();

                for (IndexType entry = P_row_offsets[row];
                     entry < P_row_offsets[row + 1]; ++entry)
                {
                    value += P_values[
                        static_cast<size_t>(entry) * block_size
                        + dense_block_offset<ValueType>(
                            local_row, local_col, block_dim, row_major)];
                }

                work[work_begin + logical] = value;
                inverse[work_begin + logical] =
                    local_row == local_col
                    ? types::util<ValueType>::get_one()
                    : types::util<ValueType>::get_zero();
                const PodType magnitude =
                    types::util<ValueType>::abs(value);
                block_norm = magnitude > block_norm ? magnitude : block_norm;
            }
        }

        const PodType threshold =
            pivot_tolerance
            * (block_norm > PodType(1) ? block_norm : PodType(1));

        for (int pivot_col = 0; pivot_col < block_dim; ++pivot_col)
        {
            int pivot_row = pivot_col;
            PodType pivot_abs = types::util<ValueType>::abs(
                work[work_begin + pivot_col * block_dim + pivot_col]);

            for (int candidate = pivot_col + 1;
                 candidate < block_dim; ++candidate)
            {
                const PodType candidate_abs = types::util<ValueType>::abs(
                    work[work_begin + candidate * block_dim + pivot_col]);

                if (candidate_abs > pivot_abs)
                {
                    pivot_abs = candidate_abs;
                    pivot_row = candidate;
                }
            }

            if (pivot_abs <= threshold)
            {
                if (atomicCAS(invalid, 0, 1) == 0)
                {
                    invalid[1] = static_cast<int>(row);
                }

                break;
            }

            if (pivot_row != pivot_col)
            {
                for (int col = 0; col < block_dim; ++col)
                {
                    ValueType temporary =
                        work[work_begin + pivot_col * block_dim + col];
                    work[work_begin + pivot_col * block_dim + col] =
                        work[work_begin + pivot_row * block_dim + col];
                    work[work_begin + pivot_row * block_dim + col] = temporary;
                    temporary =
                        inverse[work_begin + pivot_col * block_dim + col];
                    inverse[work_begin + pivot_col * block_dim + col] =
                        inverse[work_begin + pivot_row * block_dim + col];
                    inverse[work_begin + pivot_row * block_dim + col] = temporary;
                }
            }

            const ValueType pivot =
                work[work_begin + pivot_col * block_dim + pivot_col];

            for (int col = 0; col < block_dim; ++col)
            {
                work[work_begin + pivot_col * block_dim + col] =
                    work[work_begin + pivot_col * block_dim + col] / pivot;
                inverse[work_begin + pivot_col * block_dim + col] =
                    inverse[work_begin + pivot_col * block_dim + col] / pivot;
            }

            for (int eliminate_row = 0;
                 eliminate_row < block_dim; ++eliminate_row)
            {
                if (eliminate_row == pivot_col)
                {
                    continue;
                }

                const ValueType factor =
                    work[work_begin + eliminate_row * block_dim + pivot_col];

                for (int col = 0; col < block_dim; ++col)
                {
                    work[work_begin + eliminate_row * block_dim + col] -=
                        factor
                        * work[work_begin + pivot_col * block_dim + col];
                    inverse[work_begin + eliminate_row * block_dim + col] -=
                        factor
                        * inverse[work_begin + pivot_col * block_dim + col];
                }
            }
        }
    }
}

template <typename IndexType, typename ValueType>
__global__ void right_normalize_dense_transfer_kernel(
    const IndexType *P_row_offsets, const ValueType *P_values,
    const ValueType *row_sum_inverse, IndexType num_rows, int block_dim,
    bool row_major, ValueType *normalized_P_values)
{
    const int block_size = block_dim * block_dim;

    for (IndexType row = blockIdx.x; row < num_rows; row += gridDim.x)
    {
        const IndexType P_begin = P_row_offsets[row];
        const IndexType P_end = P_row_offsets[row + 1];
        const size_t row_value_count =
            static_cast<size_t>(P_end - P_begin) * block_size;

        for (size_t row_value = threadIdx.x; row_value < row_value_count;
             row_value += blockDim.x)
        {
            const IndexType P_entry =
                P_begin + static_cast<IndexType>(row_value / block_size);
            const int logical = static_cast<int>(row_value % block_size);
            const int local_row = logical / block_dim;
            const int local_col = logical % block_dim;
            ValueType normalized = types::util<ValueType>::get_zero();

            for (int inner = 0; inner < block_dim; ++inner)
            {
                normalized +=
                    P_values[static_cast<size_t>(P_entry) * block_size
                             + dense_block_offset<ValueType>(
                                 local_row, inner, block_dim, row_major)]
                    * row_sum_inverse[
                        static_cast<size_t>(row) * block_size
                        + inner * block_dim + local_col];
            }

            normalized_P_values[
                static_cast<size_t>(P_entry) * block_size
                + dense_block_offset<ValueType>(
                    local_row, local_col, block_dim, row_major)] = normalized;
        }
    }
}

template <typename IndexType, typename ValueType>
__global__ void validate_dense_row_constraint_kernel(
    const IndexType *P_row_offsets, const ValueType *block_P_values,
    IndexType num_rows, int block_dim, bool row_major, bool constant_vector,
    typename types::PODTypes<ValueType>::type tolerance, int *invalid)
{
    const int block_size = block_dim * block_dim;
    const size_t constraint_count = static_cast<size_t>(num_rows) * block_size;

    for (size_t constraint = threadIdx.x
                             + static_cast<size_t>(blockIdx.x) * blockDim.x;
         constraint < constraint_count;
         constraint += static_cast<size_t>(blockDim.x) * gridDim.x)
    {
        const IndexType row =
            static_cast<IndexType>(constraint / block_size);
        const int logical = static_cast<int>(constraint % block_size);
        const int local_row = logical / block_dim;
        const int local_col = logical % block_dim;
        ValueType row_sum = types::util<ValueType>::get_zero();

        for (IndexType entry = P_row_offsets[row];
             entry < P_row_offsets[row + 1]; ++entry)
        {
            const int begin_col = constant_vector ? 0 : local_col;
            const int end_col = constant_vector ? block_dim : local_col + 1;
            for (int col = begin_col; col < end_col; ++col)
            {
                row_sum +=
                    block_P_values[static_cast<size_t>(entry) * block_size
                                   + dense_block_offset<ValueType>(
                                       local_row, col, block_dim, row_major)];
            }
        }

        const ValueType target =
            constant_vector || local_row == local_col
            ? types::util<ValueType>::get_one()
            : types::util<ValueType>::get_zero();

        if (!(types::util<ValueType>::abs(row_sum - target) <= tolerance))
        {
            if (atomicCAS(invalid, 0, 1) == 0)
            {
                invalid[1] = static_cast<int>(row);
            }
        }
    }
}

template <typename IndexType, typename ValueType>
__global__ void transpose_dense_transfer_kernel(
    const IndexType *R_row_offsets, const IndexType *R_col_indices,
    IndexType num_coarse_rows, const IndexType *P_row_offsets,
    const IndexType *P_col_indices, const ValueType *P_values,
    int block_dim, bool row_major, ValueType *R_values, int *invalid)
{
    const int block_size = block_dim * block_dim;

    for (IndexType coarse_row = blockIdx.x;
         coarse_row < num_coarse_rows; coarse_row += gridDim.x)
    {
        const IndexType R_begin = R_row_offsets[coarse_row];
        const IndexType R_end = R_row_offsets[coarse_row + 1];
        const size_t row_value_count =
            static_cast<size_t>(R_end - R_begin) * block_size;

        for (size_t row_value = threadIdx.x; row_value < row_value_count;
             row_value += blockDim.x)
        {
            const IndexType R_entry =
                R_begin + static_cast<IndexType>(row_value / block_size);
            const int logical = static_cast<int>(row_value % block_size);
            const int local_row = logical / block_dim;
            const int local_col = logical % block_dim;
            const IndexType fine_row = R_col_indices[R_entry];
            const IndexType P_entry = find_block_entry(
                P_row_offsets, P_col_indices, fine_row, coarse_row);

            if (P_entry < 0)
            {
                if (atomicCAS(invalid, 0, 1) == 0)
                {
                    invalid[1] = static_cast<int>(coarse_row);
                }

                continue;
            }

            R_values[static_cast<size_t>(R_entry) * block_size
                     + dense_block_offset<ValueType>(
                         local_row, local_col, block_dim, row_major)] =
                types::util<ValueType>::conjugate(
                    P_values[static_cast<size_t>(P_entry) * block_size
                             + dense_block_offset<ValueType>(
                                 local_col, local_row, block_dim, row_major)]);
        }
    }
}

template <typename IndexType, typename ValueType>
__global__ void dense_bsr_galerkin_kernel(
    const IndexType *A_row_offsets, const IndexType *A_col_indices,
    const ValueType *A_values, const IndexType *R_row_offsets,
    const IndexType *R_col_indices, const ValueType *R_values,
    const IndexType *P_row_offsets, const IndexType *P_col_indices,
    const ValueType *P_values, const IndexType *Ac_row_offsets,
    const IndexType *Ac_col_indices, IndexType num_coarse_rows,
    int block_dim, bool row_major, ValueType *Ac_values)
{
    extern __shared__ unsigned char shared_bytes[];
    ValueType *left_product = reinterpret_cast<ValueType *>(shared_bytes);
    const int block_size = block_dim * block_dim;

    for (IndexType coarse_row = blockIdx.x;
         coarse_row < num_coarse_rows; coarse_row += gridDim.x)
    {
        for (IndexType coarse_entry = Ac_row_offsets[coarse_row];
             coarse_entry < Ac_row_offsets[coarse_row + 1]; ++coarse_entry)
        {
            const IndexType coarse_col = Ac_col_indices[coarse_entry];

            for (int logical = threadIdx.x; logical < block_size;
                 logical += blockDim.x)
            {
                const int local_row = logical / block_dim;
                const int local_col = logical % block_dim;
                Ac_values[static_cast<size_t>(coarse_entry) * block_size
                          + dense_block_offset<ValueType>(
                              local_row, local_col, block_dim, row_major)] =
                    types::util<ValueType>::get_zero();
            }

            __syncthreads();

            for (IndexType R_entry = R_row_offsets[coarse_row];
                 R_entry < R_row_offsets[coarse_row + 1]; ++R_entry)
            {
                const IndexType fine_row = R_col_indices[R_entry];

                for (IndexType A_entry = A_row_offsets[fine_row];
                     A_entry < A_row_offsets[fine_row + 1]; ++A_entry)
                {
                    const IndexType fine_col = A_col_indices[A_entry];
                    const IndexType P_entry = find_block_entry(
                        P_row_offsets, P_col_indices, fine_col, coarse_col);

                    if (P_entry < 0)
                    {
                        continue;
                    }

                    for (int logical = threadIdx.x; logical < block_size;
                         logical += blockDim.x)
                    {
                        const int local_row = logical / block_dim;
                        const int local_col = logical % block_dim;
                        ValueType sum = types::util<ValueType>::get_zero();

                        for (int inner = 0; inner < block_dim; ++inner)
                        {
                            sum +=
                                R_values[
                                    static_cast<size_t>(R_entry) * block_size
                                    + dense_block_offset<ValueType>(
                                        local_row, inner, block_dim, row_major)]
                                * A_values[
                                    static_cast<size_t>(A_entry) * block_size
                                    + dense_block_offset<ValueType>(
                                        inner, local_col, block_dim, row_major)];
                        }

                        left_product[logical] = sum;
                    }

                    __syncthreads();

                    for (int logical = threadIdx.x; logical < block_size;
                         logical += blockDim.x)
                    {
                        const int local_row = logical / block_dim;
                        const int local_col = logical % block_dim;
                        ValueType contribution =
                            types::util<ValueType>::get_zero();

                        for (int inner = 0; inner < block_dim; ++inner)
                        {
                            contribution +=
                                left_product[local_row * block_dim + inner]
                                * P_values[
                                    static_cast<size_t>(P_entry) * block_size
                                    + dense_block_offset<ValueType>(
                                        inner, local_col, block_dim, row_major)];
                        }

                        Ac_values[
                            static_cast<size_t>(coarse_entry) * block_size
                            + dense_block_offset<ValueType>(
                                local_row, local_col, block_dim, row_major)] +=
                            contribution;
                    }

                    __syncthreads();
                }
            }
        }
    }
}

template <typename IndexType, typename ValueType>
__global__ void weighted_bsr_galerkin_kernel(
    const IndexType *A_row_offsets, const IndexType *A_col_indices,
    const ValueType *A_values, const IndexType *R_row_offsets,
    const IndexType *R_col_indices, const ValueType *R_values,
    const IndexType *P_row_offsets, const IndexType *P_col_indices,
    const ValueType *P_values, const IndexType *Ac_row_offsets,
    const IndexType *Ac_col_indices, IndexType num_coarse_rows,
    int block_size, ValueType *Ac_values)
{
    for (IndexType coarse_row = blockIdx.x; coarse_row < num_coarse_rows;
         coarse_row += gridDim.x)
    {
        const IndexType coarse_begin = Ac_row_offsets[coarse_row];
        const IndexType coarse_end = Ac_row_offsets[coarse_row + 1];
        const size_t row_value_count =
            static_cast<size_t>(coarse_end - coarse_begin) * block_size;

        for (size_t local_value = threadIdx.x; local_value < row_value_count;
             local_value += blockDim.x)
        {
            const IndexType coarse_entry = coarse_begin
                + static_cast<IndexType>(local_value / block_size);
            const int block_entry = static_cast<int>(local_value % block_size);
            const IndexType coarse_col = Ac_col_indices[coarse_entry];
            ValueType sum = types::util<ValueType>::get_zero();

            for (IndexType r_entry = R_row_offsets[coarse_row];
                 r_entry < R_row_offsets[coarse_row + 1]; ++r_entry)
            {
                const IndexType fine_row = R_col_indices[r_entry];
                const ValueType left_weight = R_values[r_entry];

                for (IndexType a_entry = A_row_offsets[fine_row];
                     a_entry < A_row_offsets[fine_row + 1]; ++a_entry)
                {
                    const IndexType fine_col = A_col_indices[a_entry];

                    for (IndexType p_entry = P_row_offsets[fine_col];
                         p_entry < P_row_offsets[fine_col + 1]; ++p_entry)
                    {
                        if (P_col_indices[p_entry] == coarse_col)
                        {
                            sum += left_weight
                                   * A_values[static_cast<size_t>(a_entry) * block_size
                                              + block_entry]
                                   * P_values[p_entry];
                        }
                    }
                }
            }

            Ac_values[static_cast<size_t>(coarse_entry) * block_size
                      + block_entry] = sum;
        }
    }
}

template <class TConfig>
void initialize_matrix_from_structure(const Matrix<TConfig> &structure,
                                      int block_dim, BlockFormat block_format,
                                      Matrix<TConfig> &target)
{
    typedef typename Matrix<TConfig>::index_type IndexType;
    target.set_initialized(0);
    target.delProps(target.getProps());
    target.set_num_rows(0);
    target.set_num_cols(0);
    target.set_num_nz(0);
    target.addProps(CSR);
    target.set_num_rows(structure.get_num_rows());
    target.set_num_cols(structure.get_num_cols());
    target.set_num_nz(structure.get_num_nz());
    target.set_block_dimx(block_dim);
    target.set_block_dimy(block_dim);
    target.setBlockFormat(block_format);
    target.setResources(structure.getResources());
    target.row_offsets = structure.row_offsets;
    target.col_indices = structure.col_indices;
    target.values.resize(static_cast<size_t>(structure.get_num_nz())
                         * block_dim * block_dim);
    target.diag.resize(structure.get_num_rows());
    target.m_diag_end_offsets.resize(structure.get_num_rows());
    target.m_seq_offsets.resize(structure.get_num_rows() + 1);

    const IndexType sequence_size = structure.get_num_rows() + 1;

    if (sequence_size > 0)
    {
        const int threads = 256;
        const int blocks = std::min(AMGX_GRID_MAX_SIZE,
                                    static_cast<int>((sequence_size + threads - 1) / threads));
        fill_sequence_kernel<<<
            blocks, threads, 0,
            amgx::thrust::global_thread_handle::get_stream()>>>(
                target.m_seq_offsets.raw(), sequence_size);
        cudaCheckError();
    }
}

} // namespace

template <AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I>
void Block_Graph_Ops<TemplateConfig<AMGX_host, V, M, I> >::build_graph(
    const Matrix<TConfig> &, Matrix<TConfig> &, int)
{
    FatalError("Classical block-graph BSR hierarchy is device-only", AMGX_ERR_NOT_IMPLEMENTED);
}

template <AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I>
void Block_Graph_Ops<TemplateConfig<AMGX_host, V, M, I> >::lift_scalar_transfer(
    const Matrix<TConfig> &, int, BlockFormat, Matrix<TConfig> &)
{
    FatalError("Classical block-graph BSR hierarchy is device-only", AMGX_ERR_NOT_IMPLEMENTED);
}

template <AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I>
void Block_Graph_Ops<TemplateConfig<AMGX_host, V, M, I> >::extract_scalar_transfer(
    const Matrix<TConfig> &, Matrix<TConfig> &)
{
    FatalError("Classical block-graph BSR hierarchy is device-only", AMGX_ERR_NOT_IMPLEMENTED);
}

template <AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I>
void Block_Graph_Ops<TemplateConfig<AMGX_host, V, M, I> >::multiply_identity_transfer(
    Matrix<TConfig> &, Vector<TConfig> &, Vector<TConfig> &)
{
    FatalError("Classical block-graph BSR hierarchy is device-only", AMGX_ERR_NOT_IMPLEMENTED);
}

template <AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I>
void Block_Graph_Ops<TemplateConfig<AMGX_host, V, M, I> >::smooth_dense_transfer(
    const Matrix<TConfig> &, const Matrix<TConfig> &, double, int, bool,
    double, double, Matrix<TConfig> &,
    const Vector<typename TConfig::template setVecPrec<AMGX_vecInt>::Type> *, bool)
{
    FatalError("Classical dense block-graph BSR hierarchy is device-only",
               AMGX_ERR_NOT_IMPLEMENTED);
}

template <AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I>
void Block_Graph_Ops<TemplateConfig<AMGX_host, V, M, I> >::extended_i_dense_transfer(
    const Matrix<TConfig> &, const Matrix<TConfig> &,
    const Vector<typename TConfig::template setVecPrec<AMGX_vecInt>::Type> &,
    const Vector<typename TConfig::template setVecPrec<AMGX_vecBool>::Type> &,
    double, double, Matrix<TConfig> &)
{
    FatalError("Classical block Extended+i interpolation is device-only",
               AMGX_ERR_NOT_IMPLEMENTED);
}

template <AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I>
void Block_Graph_Ops<TemplateConfig<AMGX_host, V, M, I> >::transpose_dense_transfer(
    const Matrix<TConfig> &, const Matrix<TConfig> &, Matrix<TConfig> &)
{
    FatalError("Classical dense block-graph BSR hierarchy is device-only",
               AMGX_ERR_NOT_IMPLEMENTED);
}

template <AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I>
void Block_Graph_Ops<TemplateConfig<AMGX_host, V, M, I> >::weighted_galerkin(
    const Matrix<TConfig> &, const Matrix<TConfig> &, const Matrix<TConfig> &,
    const Matrix<TConfig> &, Matrix<TConfig> &, void *)
{
    FatalError("Classical block-graph BSR hierarchy is device-only", AMGX_ERR_NOT_IMPLEMENTED);
}

template <AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I>
void Block_Graph_Ops<TemplateConfig<AMGX_host, V, M, I> >::dense_galerkin(
    const Matrix<TConfig> &, const Matrix<TConfig> &, const Matrix<TConfig> &,
    const Matrix<TConfig> &, const Matrix<TConfig> &, const Matrix<TConfig> &,
    Matrix<TConfig> &, void *)
{
    FatalError("Classical dense block-graph BSR hierarchy is device-only",
               AMGX_ERR_NOT_IMPLEMENTED);
}

template <AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I>
void Block_Graph_Ops<TemplateConfig<AMGX_device, V, M, I> >::build_graph(
    const Matrix<TConfig> &A, Matrix<TConfig> &graph,
    int strength_metric)
{
    typedef typename Matrix<TConfig>::index_type IndexType;
    typedef typename Matrix<TConfig>::value_type ValueType;
    typedef typename types::PODTypes<ValueType>::type PodType;
    BlockFormat scalar_format = ROW_MAJOR;
    initialize_matrix_from_structure(A, 1, scalar_format, graph);

    if (A.get_num_rows() > 0)
    {
        const int threads = 128;
        const int blocks = std::min(AMGX_GRID_MAX_SIZE,
                                    static_cast<int>((A.get_num_rows() + threads - 1) / threads));

        if (strength_metric == 1)
        {
            device_vector_alloc<PodType> diagonal_norms(A.get_num_rows());
            device_vector_alloc<int> invalid_diagonal(1, 0);
            compute_diagonal_frobenius_norms_kernel<<<
                blocks, threads, 0,
                amgx::thrust::global_thread_handle::get_stream()>>>(
                A.row_offsets.raw(), A.col_indices.raw(), A.values.raw(),
                A.get_num_rows(), A.get_block_size(),
                amgx::thrust::raw_pointer_cast(diagonal_norms.data()),
                amgx::thrust::raw_pointer_cast(invalid_diagonal.data()));
            cudaCheckError();

            const int diagonal_error = invalid_diagonal[0];

            if (diagonal_error == 1)
            {
                FatalError("Block-graph BSR hierarchy requires an internal diagonal in every row",
                           AMGX_ERR_BAD_PARAMETERS);
            }
            else if (diagonal_error == 2)
            {
                FatalError("Block-graph BSR hierarchy requires nonzero diagonal blocks",
                           AMGX_ERR_BAD_PARAMETERS);
            }

            build_diagonal_normalized_frobenius_graph_kernel<<<
                blocks, threads, 0,
                amgx::thrust::global_thread_handle::get_stream()>>>(
                A.row_offsets.raw(), A.col_indices.raw(), A.values.raw(),
                A.get_num_rows(), A.get_block_size(),
                amgx::thrust::raw_pointer_cast(diagonal_norms.data()),
                graph.values.raw());
            cudaCheckError();
        }
        else if (strength_metric == 2)
        {
            if (A.diag.size() < static_cast<size_t>(A.get_num_rows()))
            {
                FatalError("Inverse-scaled block strength requires computed internal diagonals",
                           AMGX_ERR_BAD_PARAMETERS);
            }

            const int block_size = A.get_block_size();
            const size_t inverse_count =
                static_cast<size_t>(A.get_num_rows()) * block_size;
            device_vector_alloc<ValueType> diagonal_work(inverse_count);
            device_vector_alloc<ValueType> inverse_diagonal(inverse_count);
            device_vector_alloc<int> invalid_inverse(2, 0);
            device_vector_alloc<int> missing_reverse(1, 0);
            const PodType pivot_tolerance =
                PodType(100) * std::numeric_limits<PodType>::epsilon();
            invert_diagonal_blocks_kernel<<<
                blocks, threads, 0,
                amgx::thrust::global_thread_handle::get_stream()>>>(
                A.row_offsets.raw(), A.diag.raw(), A.values.raw(),
                A.get_num_rows(), A.get_block_dimx(),
                A.getBlockFormat() == ROW_MAJOR, pivot_tolerance,
                amgx::thrust::raw_pointer_cast(diagonal_work.data()),
                amgx::thrust::raw_pointer_cast(inverse_diagonal.data()),
                amgx::thrust::raw_pointer_cast(invalid_inverse.data()));
            cudaCheckError();

            const int inverse_error = invalid_inverse[0];

            if (inverse_error == 1)
            {
                FatalError("Inverse-scaled block strength found an invalid diagonal index",
                           AMGX_ERR_BAD_PARAMETERS);
            }
            else if (inverse_error == 2)
            {
                FatalError("Inverse-scaled block strength found a singular diagonal block",
                           AMGX_ERR_BAD_PARAMETERS);
            }

            build_symmetric_inverse_diagonal_graph_kernel<<<
                blocks, threads, 0,
                amgx::thrust::global_thread_handle::get_stream()>>>(
                A.row_offsets.raw(), A.col_indices.raw(), A.values.raw(),
                amgx::thrust::raw_pointer_cast(inverse_diagonal.data()),
                A.get_num_rows(), A.get_block_dimx(),
                A.getBlockFormat() == ROW_MAJOR, graph.values.raw(),
                amgx::thrust::raw_pointer_cast(missing_reverse.data()));
            cudaCheckError();

            if (missing_reverse[0] != 0)
            {
                FatalError("Inverse-scaled block strength requires symmetric block sparsity",
                           AMGX_ERR_BAD_PARAMETERS);
            }
        }
        else
        {
            device_vector_alloc<int> missing_diagonal(1, 0);
            build_frobenius_graph_kernel<<<
                blocks, threads, 0,
                amgx::thrust::global_thread_handle::get_stream()>>>(
                A.row_offsets.raw(), A.col_indices.raw(), A.values.raw(),
                A.get_num_rows(), A.get_block_size(), graph.values.raw(),
                amgx::thrust::raw_pointer_cast(missing_diagonal.data()));
            cudaCheckError();

            if (missing_diagonal[0] != 0)
            {
                FatalError("Block-graph BSR hierarchy requires an internal diagonal in every row",
                           AMGX_ERR_BAD_PARAMETERS);
            }
        }
    }

    graph.computeDiagonal();
    graph.set_initialized(1);
}

template <AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I>
void Block_Graph_Ops<TemplateConfig<AMGX_device, V, M, I> >::lift_scalar_transfer(
    const Matrix<TConfig> &scalar_transfer, int block_dim,
    BlockFormat block_format, Matrix<TConfig> &block_transfer)
{
    initialize_matrix_from_structure(scalar_transfer, block_dim, block_format,
                                     block_transfer);
    block_transfer.setParameter("block_graph_expected_num_rows",
                                block_transfer.get_num_rows());
    block_transfer.setParameter("block_graph_expected_num_cols",
                                block_transfer.get_num_cols());
    block_transfer.setParameter("block_graph_expected_num_nz",
                                block_transfer.get_num_nz());
    const size_t value_count = static_cast<size_t>(scalar_transfer.get_num_nz())
                               * block_dim * block_dim;

    if (value_count > 0)
    {
        const int threads = 256;
        const int blocks = std::min<size_t>(AMGX_GRID_MAX_SIZE,
                                            (value_count + threads - 1) / threads);
        lift_scalar_transfer_kernel<<<
            blocks, threads, 0,
            amgx::thrust::global_thread_handle::get_stream()>>>(
            scalar_transfer.values.raw(), scalar_transfer.get_num_nz(), block_dim,
            block_transfer.values.raw());
        cudaCheckError();

    }

    block_transfer.computeDiagonal();
    block_transfer.set_initialized(1);

    device_vector_alloc<int> invalid_structure(4, 0);
    const int validation_threads = 256;

    if (block_transfer.get_num_rows() > 0)
    {
        const int validation_blocks = std::min(
            AMGX_GRID_MAX_SIZE,
            static_cast<int>((block_transfer.get_num_rows() + validation_threads - 1)
                             / validation_threads));
        validate_transfer_structure_kernel<<<
            validation_blocks, validation_threads, 0,
            amgx::thrust::global_thread_handle::get_stream()>>>(
            block_transfer.row_offsets.raw(), block_transfer.col_indices.raw(),
            block_transfer.get_num_rows(), block_transfer.get_num_nz(),
            block_transfer.get_num_cols(),
            amgx::thrust::raw_pointer_cast(invalid_structure.data()));
        cudaCheckError();
    }

    const int invalid_code = invalid_structure[0];
    if (invalid_code == 1)
    {
        FatalError("Block-graph transfer has invalid row offsets",
                   AMGX_ERR_BAD_PARAMETERS);
    }
    else if (invalid_code == 2)
    {
        FatalError("Block-graph transfer has an out-of-range column",
                   AMGX_ERR_BAD_PARAMETERS);
    }
    else if (invalid_code == 3)
    {
        FatalError("Block-graph transfer terminal offset disagrees with nnz",
                   AMGX_ERR_BAD_PARAMETERS);
    }
}

template <AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I>
void Block_Graph_Ops<TemplateConfig<AMGX_device, V, M, I> >::extract_scalar_transfer(
    const Matrix<TConfig> &block_transfer, Matrix<TConfig> &scalar_transfer)
{
    BlockFormat scalar_format = ROW_MAJOR;
    initialize_matrix_from_structure(block_transfer, 1, scalar_format,
                                     scalar_transfer);

    if (block_transfer.get_num_nz() > 0)
    {
        const int threads = 256;
        const int blocks = std::min(AMGX_GRID_MAX_SIZE,
                                    static_cast<int>((block_transfer.get_num_nz()
                                                      + threads - 1) / threads));
        extract_scalar_transfer_kernel<<<
            blocks, threads, 0,
            amgx::thrust::global_thread_handle::get_stream()>>>(
            block_transfer.values.raw(), block_transfer.get_num_nz(),
            block_transfer.get_block_size(), scalar_transfer.values.raw());
        cudaCheckError();
    }

    scalar_transfer.computeDiagonal();
    scalar_transfer.set_initialized(1);
}

template <AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I>
void Block_Graph_Ops<TemplateConfig<AMGX_device, V, M, I> >::multiply_identity_transfer(
    Matrix<TConfig> &transfer, Vector<TConfig> &input, Vector<TConfig> &output)
{
    typedef typename Matrix<TConfig>::index_type IndexType;
    typedef typename Matrix<TConfig>::value_type MatrixValue;
    typedef typename TConfig::VecPrec VectorValue;

    if (!transfer.is_initialized()
            || transfer.get_block_dimx() != transfer.get_block_dimy()
            || transfer.get_block_dimx() <= 1)
    {
        FatalError("Invalid BSR identity-block transfer matrix",
                   AMGX_ERR_BAD_PARAMETERS);
    }

    const size_t required_input = static_cast<size_t>(transfer.get_num_cols())
                                  * transfer.get_block_dimx();
    const size_t required_output = static_cast<size_t>(transfer.get_num_rows())
                                   * transfer.get_block_dimy();

    if (input.size() < required_input || output.size() < required_output)
    {
        FatalError("Pure-BSR identity transfer vector size mismatch",
                   AMGX_ERR_INTERNAL);
    }

    if (required_output > 0)
    {
        const int threads = 256;
        device_vector_alloc<int> invalid_structure(4, 0);
        const int validation_blocks = std::min(
            AMGX_GRID_MAX_SIZE,
            static_cast<int>((transfer.get_num_rows() + threads - 1) / threads));
        validate_transfer_structure_kernel<<<
            validation_blocks, threads, 0,
            amgx::thrust::global_thread_handle::get_stream()>>>(
            transfer.row_offsets.raw(), transfer.col_indices.raw(),
            transfer.get_num_rows(), transfer.get_num_nz(),
            transfer.get_num_cols(),
            amgx::thrust::raw_pointer_cast(invalid_structure.data()));
        cudaCheckError();

        const int invalid_code = invalid_structure[0];

        if (invalid_code != 0)
        {
            const int expected_rows = transfer.template getParameter<int>(
                                          "block_graph_expected_num_rows");
            const int expected_cols = transfer.template getParameter<int>(
                                          "block_graph_expected_num_cols");
            const int expected_nnz = transfer.template getParameter<int>(
                                         "block_graph_expected_num_nz");
            amgx_printf(
                "Pure-BSR transfer diagnostic: current=(rows=%d, cols=%d, nnz=%d), "
                "expected=(rows=%d, cols=%d, nnz=%d), code=%d, "
                "detail=(%d, %d, %d)\n",
                transfer.get_num_rows(), transfer.get_num_cols(),
                transfer.get_num_nz(), expected_rows, expected_cols,
                expected_nnz, invalid_code, invalid_structure[1],
                invalid_structure[2], invalid_structure[3]);
        }

        if (invalid_code == 1)
        {
            FatalError("Pure-BSR transfer row offsets were corrupted after setup",
                       AMGX_ERR_INTERNAL);
        }
        else if (invalid_code == 2)
        {
            FatalError("Pure-BSR transfer columns were corrupted after setup",
                       AMGX_ERR_INTERNAL);
        }
        else if (invalid_code == 3)
        {
            FatalError("Pure-BSR transfer terminal offset changed after setup",
                       AMGX_ERR_INTERNAL);
        }

        const int blocks = std::min<size_t>(
            AMGX_GRID_MAX_SIZE, (required_output + threads - 1) / threads);
        identity_transfer_spmv_kernel<IndexType, MatrixValue, VectorValue>
            <<<blocks, threads, 0,
               amgx::thrust::global_thread_handle::get_stream()>>>(
                transfer.row_offsets.raw(), transfer.col_indices.raw(),
                transfer.values.raw(), transfer.get_num_rows(),
                transfer.get_block_dimx(), input.raw(), output.raw());
        cudaCheckError();
    }

    output.dirtybit = 1;
    output.set_block_dimy(transfer.get_block_dimx());
}

template <AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I>
void Block_Graph_Ops<TemplateConfig<AMGX_device, V, M, I> >::smooth_dense_transfer(
    const Matrix<TConfig> &A, const Matrix<TConfig> &scalar_transfer,
    double smoothing_weight, int smoothing_steps, bool right_normalize,
    double pivot_tolerance, double constraint_tolerance,
    Matrix<TConfig> &block_transfer,
    const Vector<typename TConfig::template setVecPrec<AMGX_vecInt>::Type> *cf_map,
    bool constant_vector)
{
    typedef typename Matrix<TConfig>::index_type IndexType;
    typedef typename Matrix<TConfig>::value_type ValueType;
    typedef typename types::PODTypes<ValueType>::type PodType;

    if (A.get_block_dimx() != A.get_block_dimy()
            || A.get_block_dimx() <= 1
            || A.get_num_rows() != A.get_num_cols()
            || scalar_transfer.get_block_size() != 1
            || scalar_transfer.get_num_rows() != A.get_num_rows())
    {
        FatalError("Invalid matrices for dense block-graph interpolation",
                   AMGX_ERR_BAD_PARAMETERS);
    }

    if (constant_vector && (right_normalize || cf_map == NULL
            || cf_map->size() < static_cast<size_t>(A.get_num_rows())))
    {
        FatalError("Constant-vector interpolation requires an additive correction and coarse-point map",
                   AMGX_ERR_BAD_PARAMETERS);
    }

    if (A.diag.size() < static_cast<size_t>(A.get_num_rows()))
    {
        FatalError("Dense block interpolation requires computed internal diagonals",
                   AMGX_ERR_BAD_PARAMETERS);
    }

    if (!(smoothing_weight > 0.0) || !(smoothing_weight <= 2.0)
            || smoothing_steps < 1 || smoothing_steps > 8
            || !(pivot_tolerance > 0.0)
            || !(constraint_tolerance > 0.0))
    {
        FatalError("Invalid dense block interpolation parameters",
                   AMGX_ERR_BAD_PARAMETERS);
    }

    const int block_dim = A.get_block_dimx();
    const int block_size = block_dim * block_dim;
    const bool row_major = A.getBlockFormat() == ROW_MAJOR;
    BlockFormat block_format = A.getBlockFormat();
    initialize_matrix_from_structure(
        scalar_transfer, block_dim, block_format, block_transfer);
    block_transfer.setParameter("block_graph_expected_num_rows",
                                block_transfer.get_num_rows());
    block_transfer.setParameter("block_graph_expected_num_cols",
                                block_transfer.get_num_cols());
    block_transfer.setParameter("block_graph_expected_num_nz",
                                block_transfer.get_num_nz());

    const PodType precision_floor =
        PodType(100) * std::numeric_limits<PodType>::epsilon();
    const PodType effective_pivot_tolerance =
        static_cast<PodType>(pivot_tolerance) > precision_floor
        ? static_cast<PodType>(pivot_tolerance) : precision_floor;
    const PodType effective_constraint_tolerance =
        static_cast<PodType>(constraint_tolerance) > precision_floor
        ? static_cast<PodType>(constraint_tolerance) : precision_floor;
    const size_t work_count =
        static_cast<size_t>(A.get_num_rows()) * block_size;
    const size_t transfer_value_count =
        static_cast<size_t>(block_transfer.get_num_nz()) * block_size;
    device_vector_alloc<ValueType> diagonal_work(work_count);
    device_vector_alloc<ValueType> inverse_diagonal(work_count);
    device_vector_alloc<ValueType> smoothed_values(transfer_value_count);
    device_vector_alloc<ValueType> row_sum_inverse(
        right_normalize ? work_count : 0);
    device_vector_alloc<int> invalid_inverse(2, 0);
    device_vector_alloc<int> invalid_normalization(2, 0);
    const int threads = 128;
    const int row_blocks = std::min(
        AMGX_GRID_MAX_SIZE,
        static_cast<int>((A.get_num_rows() + threads - 1) / threads));

    if (A.get_num_rows() > 0)
    {
        invert_diagonal_blocks_kernel<<<
            row_blocks, threads, 0,
            amgx::thrust::global_thread_handle::get_stream()>>>(
                A.row_offsets.raw(), A.diag.raw(), A.values.raw(),
                A.get_num_rows(), block_dim, row_major,
                effective_pivot_tolerance,
                amgx::thrust::raw_pointer_cast(diagonal_work.data()),
                amgx::thrust::raw_pointer_cast(inverse_diagonal.data()),
                amgx::thrust::raw_pointer_cast(invalid_inverse.data()));
        cudaCheckError();
    }

    const int inverse_error = invalid_inverse[0];

    if (inverse_error == 1)
    {
        FatalError("Dense block interpolation found an invalid diagonal index",
                   AMGX_ERR_BAD_PARAMETERS);
    }
    else if (inverse_error == 2)
    {
        FatalError("Dense block interpolation found a singular diagonal block",
                   AMGX_ERR_BAD_PARAMETERS);
    }

    if (A.get_num_rows() > 0)
    {
        const int transfer_blocks =
            std::min(AMGX_GRID_MAX_SIZE,
                     static_cast<int>(A.get_num_rows()));
        const int value_blocks = std::min<size_t>(
            AMGX_GRID_MAX_SIZE,
            (transfer_value_count + threads - 1) / threads);
        const int correction_blocks = std::min<size_t>(
            AMGX_GRID_MAX_SIZE, (work_count + threads - 1) / threads);
        ValueType *smoothed =
            amgx::thrust::raw_pointer_cast(smoothed_values.data());

        lift_scalar_transfer_kernel<<<
            value_blocks, threads, 0,
            amgx::thrust::global_thread_handle::get_stream()>>>(
                scalar_transfer.values.raw(), scalar_transfer.get_num_nz(),
                block_dim, block_transfer.values.raw());
        cudaCheckError();

        for (int step = 0; step < smoothing_steps; ++step)
        {
            smooth_dense_transfer_values_kernel<<<
                transfer_blocks, threads, 0,
                amgx::thrust::global_thread_handle::get_stream()>>>(
                    A.row_offsets.raw(), A.col_indices.raw(), A.values.raw(),
                    A.get_num_rows(), block_dim, row_major,
                    scalar_transfer.row_offsets.raw(),
                    scalar_transfer.col_indices.raw(),
                    block_transfer.values.raw(),
                    amgx::thrust::raw_pointer_cast(inverse_diagonal.data()),
                    constant_vector ? cf_map->raw() : NULL, constant_vector,
                    static_cast<PodType>(smoothing_weight), smoothed);
            cudaCheckError();

            if (right_normalize)
            {
                invert_dense_transfer_row_sums_kernel<<<
                    row_blocks, threads, 0,
                    amgx::thrust::global_thread_handle::get_stream()>>>(
                        scalar_transfer.row_offsets.raw(), smoothed,
                        A.get_num_rows(), block_dim, row_major,
                        effective_pivot_tolerance,
                        amgx::thrust::raw_pointer_cast(diagonal_work.data()),
                        amgx::thrust::raw_pointer_cast(row_sum_inverse.data()),
                        amgx::thrust::raw_pointer_cast(
                            invalid_normalization.data()));
                cudaCheckError();

                if (invalid_normalization[0] != 0)
                {
                    FatalError(
                        "Dense block interpolation found a singular block row sum",
                        AMGX_ERR_BAD_PARAMETERS);
                }

                right_normalize_dense_transfer_kernel<<<
                    transfer_blocks, threads, 0,
                    amgx::thrust::global_thread_handle::get_stream()>>>(
                        scalar_transfer.row_offsets.raw(), smoothed,
                        amgx::thrust::raw_pointer_cast(row_sum_inverse.data()),
                        A.get_num_rows(), block_dim, row_major,
                        block_transfer.values.raw());
                cudaCheckError();
            }
            else
            {
                compute_dense_row_correction_kernel<<<
                    correction_blocks, threads, 0,
                    amgx::thrust::global_thread_handle::get_stream()>>>(
                        scalar_transfer.row_offsets.raw(), smoothed,
                        A.get_num_rows(), block_dim, row_major, constant_vector,
                        amgx::thrust::raw_pointer_cast(diagonal_work.data()));
                cudaCheckError();

                const PodType denominator_tolerance = precision_floor;
                apply_dense_row_correction_kernel<<<
                    transfer_blocks, threads, 0,
                    amgx::thrust::global_thread_handle::get_stream()>>>(
                        scalar_transfer.row_offsets.raw(),
                        scalar_transfer.values.raw(), A.get_num_rows(),
                        block_dim, row_major, denominator_tolerance,
                        amgx::thrust::raw_pointer_cast(diagonal_work.data()),
                        smoothed);
                cudaCheckError();

                copy_dense_transfer_values_kernel<<<
                    value_blocks, threads, 0,
                    amgx::thrust::global_thread_handle::get_stream()>>>(
                        smoothed, transfer_value_count,
                        block_transfer.values.raw());
                cudaCheckError();
            }
        }

        device_vector_alloc<int> invalid_constraint(2, 0);
        validate_dense_row_constraint_kernel<<<
            correction_blocks, threads, 0,
            amgx::thrust::global_thread_handle::get_stream()>>>(
                scalar_transfer.row_offsets.raw(),
                block_transfer.values.raw(), A.get_num_rows(), block_dim,
                row_major, constant_vector, effective_constraint_tolerance,
                amgx::thrust::raw_pointer_cast(invalid_constraint.data()));
        cudaCheckError();

        if (invalid_constraint[0] != 0)
        {
            FatalError(
                "Dense block interpolation failed the block constant-mode constraint",
                AMGX_ERR_INTERNAL);
        }
    }

    block_transfer.computeDiagonal();
    block_transfer.set_initialized(1);
}

template <AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I>
void Block_Graph_Ops<TemplateConfig<AMGX_device, V, M, I> >::extended_i_dense_transfer(
    const Matrix<TConfig> &A, const Matrix<TConfig> &scalar_transfer,
    const Vector<typename TConfig::template setVecPrec<AMGX_vecInt>::Type> &cf_map,
    const Vector<typename TConfig::template setVecPrec<AMGX_vecBool>::Type> &s_con,
    double pivot_tolerance, double constraint_tolerance,
    Matrix<TConfig> &block_transfer)
{
    typedef typename Matrix<TConfig>::index_type IndexType;
    typedef typename Matrix<TConfig>::value_type ValueType;
    typedef typename types::PODTypes<ValueType>::type PodType;

    if (A.get_block_dimx() != A.get_block_dimy()
            || A.get_block_dimx() <= 1
            || A.get_num_rows() != A.get_num_cols()
            || scalar_transfer.get_block_size() != 1
            || scalar_transfer.get_num_rows() != A.get_num_rows()
            || cf_map.size() < static_cast<size_t>(A.get_num_rows())
            || s_con.size() < static_cast<size_t>(A.get_num_nz()))
    {
        FatalError("Invalid matrices for block Extended+i interpolation",
                   AMGX_ERR_BAD_PARAMETERS);
    }

    if (A.diag.size() < static_cast<size_t>(A.get_num_rows())
            || !(pivot_tolerance > 0.0)
            || !(constraint_tolerance > 0.0))
    {
        FatalError("Invalid block Extended+i interpolation parameters",
                   AMGX_ERR_BAD_PARAMETERS);
    }

    const int block_dim = A.get_block_dimx();
    const int block_size = block_dim * block_dim;
    const bool row_major = A.getBlockFormat() == ROW_MAJOR;
    const IndexType num_coarse_rows = scalar_transfer.get_num_cols();
    BlockFormat block_format = A.getBlockFormat();
    initialize_matrix_from_structure(
        scalar_transfer, block_dim, block_format, block_transfer);
    block_transfer.setParameter("block_graph_expected_num_rows",
                                block_transfer.get_num_rows());
    block_transfer.setParameter("block_graph_expected_num_cols",
                                block_transfer.get_num_cols());
    block_transfer.setParameter("block_graph_expected_num_nz",
                                block_transfer.get_num_nz());

    const PodType precision_floor =
        PodType(100) * std::numeric_limits<PodType>::epsilon();
    const PodType effective_pivot_tolerance =
        static_cast<PodType>(pivot_tolerance) > precision_floor
        ? static_cast<PodType>(pivot_tolerance) : precision_floor;
    const PodType effective_constraint_tolerance =
        static_cast<PodType>(constraint_tolerance) > precision_floor
        ? static_cast<PodType>(constraint_tolerance) : precision_floor;
    const size_t work_count =
        static_cast<size_t>(A.get_num_rows()) * block_size;
    const size_t transfer_value_count =
        static_cast<size_t>(block_transfer.get_num_nz()) * block_size;
    device_vector_alloc<ValueType> diagonal_work(work_count);
    device_vector_alloc<ValueType> inverse_diagonal(work_count);
    device_vector_alloc<ValueType> scaled_values(transfer_value_count);
    device_vector_alloc<IndexType> coarse_rows(num_coarse_rows, IndexType(-1));
    device_vector_alloc<int> invalid_inverse(2, 0);
    device_vector_alloc<int> invalid_lookup(1, 0);
    const int threads = 128;
    const int row_blocks = std::min(
        AMGX_GRID_MAX_SIZE,
        static_cast<int>((A.get_num_rows() + threads - 1) / threads));
    const int transfer_blocks = std::min(
        AMGX_GRID_MAX_SIZE, static_cast<int>(A.get_num_rows()));
    const int value_blocks = std::min<size_t>(
        AMGX_GRID_MAX_SIZE,
        (transfer_value_count + threads - 1) / threads);
    const int correction_blocks = std::min<size_t>(
        AMGX_GRID_MAX_SIZE, (work_count + threads - 1) / threads);

    if (A.get_num_rows() > 0)
    {
        invert_diagonal_blocks_kernel<<<
            row_blocks, threads, 0,
            amgx::thrust::global_thread_handle::get_stream()>>>(
                A.row_offsets.raw(), A.diag.raw(), A.values.raw(),
                A.get_num_rows(), block_dim, row_major,
                effective_pivot_tolerance,
                amgx::thrust::raw_pointer_cast(diagonal_work.data()),
                amgx::thrust::raw_pointer_cast(inverse_diagonal.data()),
                amgx::thrust::raw_pointer_cast(invalid_inverse.data()));
        cudaCheckError();
    }

    const int inverse_error = invalid_inverse[0];

    if (inverse_error == 1)
    {
        FatalError("Block Extended+i interpolation found an invalid diagonal index",
                   AMGX_ERR_BAD_PARAMETERS);
    }
    else if (inverse_error == 2)
    {
        FatalError("Block Extended+i interpolation found a singular diagonal block",
                   AMGX_ERR_BAD_PARAMETERS);
    }

    if (A.get_num_rows() > 0)
    {
        build_coarse_row_lookup_kernel<<<
            row_blocks, threads, 0,
            amgx::thrust::global_thread_handle::get_stream()>>>(
                cf_map.raw(), A.get_num_rows(), num_coarse_rows,
                amgx::thrust::raw_pointer_cast(coarse_rows.data()),
                amgx::thrust::raw_pointer_cast(invalid_lookup.data()));
        cudaCheckError();
        build_extended_i_raw_values_kernel<<<
            transfer_blocks, threads, 0,
            amgx::thrust::global_thread_handle::get_stream()>>>(
                A.row_offsets.raw(), A.col_indices.raw(), A.values.raw(),
                A.get_num_rows(), block_dim, row_major, cf_map.raw(),
                s_con.raw(), scalar_transfer.row_offsets.raw(),
                scalar_transfer.col_indices.raw(), num_coarse_rows,
                amgx::thrust::raw_pointer_cast(coarse_rows.data()),
                amgx::thrust::raw_pointer_cast(inverse_diagonal.data()),
                block_transfer.values.raw(),
                amgx::thrust::raw_pointer_cast(invalid_lookup.data()));
        cudaCheckError();

        if (invalid_lookup[0] != 0)
        {
            FatalError("Block Extended+i interpolation found an invalid coarse map",
                       AMGX_ERR_INTERNAL);
        }

        left_scale_extended_i_values_kernel<<<
            transfer_blocks, threads, 0,
            amgx::thrust::global_thread_handle::get_stream()>>>(
                scalar_transfer.row_offsets.raw(), cf_map.raw(),
                block_transfer.values.raw(),
                amgx::thrust::raw_pointer_cast(inverse_diagonal.data()),
                A.get_num_rows(), block_dim, row_major,
                amgx::thrust::raw_pointer_cast(scaled_values.data()));
        cudaCheckError();
        compute_dense_row_correction_kernel<<<
            correction_blocks, threads, 0,
            amgx::thrust::global_thread_handle::get_stream()>>>(
                scalar_transfer.row_offsets.raw(),
                amgx::thrust::raw_pointer_cast(scaled_values.data()),
                A.get_num_rows(), block_dim, row_major, false,
                amgx::thrust::raw_pointer_cast(diagonal_work.data()));
        cudaCheckError();
        apply_dense_row_correction_kernel<<<
            transfer_blocks, threads, 0,
            amgx::thrust::global_thread_handle::get_stream()>>>(
                scalar_transfer.row_offsets.raw(),
                scalar_transfer.values.raw(), A.get_num_rows(),
                block_dim, row_major, precision_floor,
                amgx::thrust::raw_pointer_cast(diagonal_work.data()),
                amgx::thrust::raw_pointer_cast(scaled_values.data()));
        cudaCheckError();
        copy_dense_transfer_values_kernel<<<
            value_blocks, threads, 0,
            amgx::thrust::global_thread_handle::get_stream()>>>(
                amgx::thrust::raw_pointer_cast(scaled_values.data()),
                transfer_value_count, block_transfer.values.raw());
        cudaCheckError();

        device_vector_alloc<int> invalid_constraint(2, 0);
        validate_dense_row_constraint_kernel<<<
            correction_blocks, threads, 0,
            amgx::thrust::global_thread_handle::get_stream()>>>(
                scalar_transfer.row_offsets.raw(),
                block_transfer.values.raw(), A.get_num_rows(), block_dim,
                row_major, false, effective_constraint_tolerance,
                amgx::thrust::raw_pointer_cast(invalid_constraint.data()));
        cudaCheckError();

        if (invalid_constraint[0] != 0)
        {
            FatalError(
                "Block Extended+i interpolation failed the block constant-mode constraint",
                AMGX_ERR_INTERNAL);
        }
    }

    block_transfer.computeDiagonal();
    block_transfer.set_initialized(1);
}

template <AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I>
void Block_Graph_Ops<TemplateConfig<AMGX_device, V, M, I> >::transpose_dense_transfer(
    const Matrix<TConfig> &scalar_transpose,
    const Matrix<TConfig> &block_transfer,
    Matrix<TConfig> &block_transpose)
{
    if (scalar_transpose.get_block_size() != 1
            || block_transfer.get_block_dimx()
               != block_transfer.get_block_dimy()
            || block_transfer.get_block_dimx() <= 1
            || scalar_transpose.get_num_rows()
               != block_transfer.get_num_cols()
            || scalar_transpose.get_num_cols()
               != block_transfer.get_num_rows())
    {
        FatalError("Invalid matrices for dense block transfer transpose",
                   AMGX_ERR_BAD_PARAMETERS);
    }

    const int block_dim = block_transfer.get_block_dimx();
    const bool row_major = block_transfer.getBlockFormat() == ROW_MAJOR;
    BlockFormat block_format = block_transfer.getBlockFormat();
    initialize_matrix_from_structure(
        scalar_transpose, block_dim, block_format, block_transpose);
    block_transpose.setParameter("block_graph_expected_num_rows",
                                block_transpose.get_num_rows());
    block_transpose.setParameter("block_graph_expected_num_cols",
                                block_transpose.get_num_cols());
    block_transpose.setParameter("block_graph_expected_num_nz",
                                block_transpose.get_num_nz());
    device_vector_alloc<int> invalid(2, 0);

    if (block_transpose.get_num_rows() > 0)
    {
        const int threads = 128;
        const int blocks = std::min(
            AMGX_GRID_MAX_SIZE,
            static_cast<int>(block_transpose.get_num_rows()));
        transpose_dense_transfer_kernel<<<
            blocks, threads, 0,
            amgx::thrust::global_thread_handle::get_stream()>>>(
                block_transpose.row_offsets.raw(),
                block_transpose.col_indices.raw(),
                block_transpose.get_num_rows(),
                block_transfer.row_offsets.raw(),
                block_transfer.col_indices.raw(),
                block_transfer.values.raw(), block_dim, row_major,
                block_transpose.values.raw(),
                amgx::thrust::raw_pointer_cast(invalid.data()));
        cudaCheckError();
    }

    if (invalid[0] != 0)
    {
        FatalError("Dense block transfer transpose graphs disagree",
                   AMGX_ERR_INTERNAL);
    }

    block_transpose.computeDiagonal();
    block_transpose.set_initialized(1);
}

template <AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I>
void Block_Graph_Ops<TemplateConfig<AMGX_device, V, M, I> >::weighted_galerkin(
    const Matrix<TConfig> &A, const Matrix<TConfig> &graph_A,
    const Matrix<TConfig> &graph_R, const Matrix<TConfig> &graph_P,
    Matrix<TConfig> &Ac, void *workspace)
{
    Matrix<TConfig> coarse_graph;
    coarse_graph.addProps(CSR);
    CSR_Multiply<TConfig>::csr_galerkin_product(
        graph_R, graph_A, graph_P, coarse_graph,
        NULL, NULL, NULL, NULL, NULL, NULL, workspace);

    BlockFormat block_format = A.getBlockFormat();
    initialize_matrix_from_structure(coarse_graph, A.get_block_dimx(),
                                     block_format, Ac);
    Ac.setParameter("bsr_spmv_backend",
                    A.template getParameter<int>("bsr_spmv_backend"));
    Ac.setParameter("use_subgroup_5x5_spmv",
                    A.template getParameter<int>("use_subgroup_5x5_spmv"));

    if (coarse_graph.get_num_rows() > 0)
    {
        const int threads = 128;
        const int blocks = std::min(AMGX_GRID_MAX_SIZE,
                                    static_cast<int>(coarse_graph.get_num_rows()));
        weighted_bsr_galerkin_kernel<<<
            blocks, threads, 0,
            amgx::thrust::global_thread_handle::get_stream()>>>(
            A.row_offsets.raw(), A.col_indices.raw(), A.values.raw(),
            graph_R.row_offsets.raw(), graph_R.col_indices.raw(),
            graph_R.values.raw(), graph_P.row_offsets.raw(),
            graph_P.col_indices.raw(), graph_P.values.raw(),
            coarse_graph.row_offsets.raw(), coarse_graph.col_indices.raw(),
            coarse_graph.get_num_rows(), A.get_block_size(), Ac.values.raw());
        cudaCheckError();
    }

    Ac.computeDiagonal();
    Ac.set_initialized(1);
}

template <AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I>
void Block_Graph_Ops<TemplateConfig<AMGX_device, V, M, I> >::dense_galerkin(
    const Matrix<TConfig> &A, const Matrix<TConfig> &graph_A,
    const Matrix<TConfig> &graph_R, const Matrix<TConfig> &graph_P,
    const Matrix<TConfig> &R, const Matrix<TConfig> &P,
    Matrix<TConfig> &Ac, void *workspace)
{
    typedef typename Matrix<TConfig>::value_type ValueType;

    if (A.get_block_dimx() != A.get_block_dimy()
            || R.get_block_dimx() != A.get_block_dimx()
            || R.get_block_dimy() != A.get_block_dimy()
            || P.get_block_dimx() != A.get_block_dimx()
            || P.get_block_dimy() != A.get_block_dimy()
            || R.getBlockFormat() != A.getBlockFormat()
            || P.getBlockFormat() != A.getBlockFormat())
    {
        FatalError("Incompatible dense block matrices in Galerkin product",
                   AMGX_ERR_BAD_PARAMETERS);
    }

    Matrix<TConfig> coarse_graph;
    coarse_graph.addProps(CSR);
    CSR_Multiply<TConfig>::csr_galerkin_product(
        graph_R, graph_A, graph_P, coarse_graph,
        NULL, NULL, NULL, NULL, NULL, NULL, workspace);

    BlockFormat block_format = A.getBlockFormat();
    initialize_matrix_from_structure(
        coarse_graph, A.get_block_dimx(), block_format, Ac);
    Ac.setParameter("bsr_spmv_backend",
                    A.template getParameter<int>("bsr_spmv_backend"));
    Ac.setParameter("use_subgroup_5x5_spmv",
                    A.template getParameter<int>("use_subgroup_5x5_spmv"));

    if (coarse_graph.get_num_rows() > 0)
    {
        const int threads = 128;
        const int blocks = std::min(
            AMGX_GRID_MAX_SIZE,
            static_cast<int>(coarse_graph.get_num_rows()));
        const size_t shared_bytes =
            static_cast<size_t>(A.get_block_size()) * sizeof(ValueType);
        dense_bsr_galerkin_kernel<<<
            blocks, threads, shared_bytes,
            amgx::thrust::global_thread_handle::get_stream()>>>(
                A.row_offsets.raw(), A.col_indices.raw(), A.values.raw(),
                R.row_offsets.raw(), R.col_indices.raw(), R.values.raw(),
                P.row_offsets.raw(), P.col_indices.raw(), P.values.raw(),
                coarse_graph.row_offsets.raw(),
                coarse_graph.col_indices.raw(),
                coarse_graph.get_num_rows(), A.get_block_dimx(),
                A.getBlockFormat() == ROW_MAJOR, Ac.values.raw());
        cudaCheckError();
    }

    Ac.computeDiagonal();
    Ac.set_initialized(1);
}

#define AMGX_CASE_LINE(CASE) template class Block_Graph_Ops<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace classical
} // namespace amgx
