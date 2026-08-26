// SPDX-FileCopyrightText: 2011 - 2026 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include "unit_test.h"
#include "amg_solver.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <vector>

namespace amgx
{

DECLARE_UNITTEST_BEGIN(BlockILUBackend);

void build_block_tridiagonal(Matrix_h &A, std::vector<double> &dense,
                             int block_rows, int block_dim)
{
    const int scalar_rows = block_rows * block_dim;
    const int block_size = block_dim * block_dim;
    const int block_nnz = 3 * block_rows - 2;
    dense.assign(scalar_rows * scalar_rows, 0.0);

    A.set_initialized(0);
    A.addProps(CSR);
    A.resize(block_rows, block_rows, block_nnz, block_dim, block_dim);

    int block_offset = 0;
    A.row_offsets[0] = 0;

    for (int block_row = 0; block_row < block_rows; ++block_row)
    {
        const int first_block_col = std::max(0, block_row - 1);
        const int last_block_col = std::min(block_rows - 1, block_row + 1);

        for (int block_col = first_block_col; block_col <= last_block_col; ++block_col)
        {
            A.col_indices[block_offset] = block_col;

            for (int local_row = 0; local_row < block_dim; ++local_row)
            {
                for (int local_col = 0; local_col < block_dim; ++local_col)
                {
                    double value;

                    if (block_row == block_col)
                    {
                        value = local_row == local_col
                                    ? 4.0 + 0.25 * block_row + 0.1 * local_row
                                    : 0.015 * (local_row + local_col + 1);
                    }
                    else
                    {
                        value = local_row == local_col
                                    ? -0.30 - 0.02 * block_row
                                    : -0.005 * (local_row + local_col + 1);
                    }

                    const int block_entry = local_row * block_dim + local_col;
                    A.values[block_offset * block_size + block_entry] = ValueTypeA(value);

                    const int scalar_row = block_row * block_dim + local_row;
                    const int scalar_col = block_col * block_dim + local_col;
                    dense[scalar_row * scalar_rows + scalar_col] = value;
                }
            }

            ++block_offset;
        }

        A.row_offsets[block_row + 1] = block_offset;
    }

    A.computeDiagonal();
    A.set_initialized(1);
}

void dense_solve(std::vector<double> matrix, std::vector<double> &rhs)
{
    const int n = static_cast<int>(rhs.size());

    for (int pivot_col = 0; pivot_col < n; ++pivot_col)
    {
        int pivot_row = pivot_col;

        for (int row = pivot_col + 1; row < n; ++row)
        {
            if (std::abs(matrix[row * n + pivot_col]) >
                std::abs(matrix[pivot_row * n + pivot_col]))
            {
                pivot_row = row;
            }
        }

        UNITTEST_ASSERT_TRUE_DESC("Host reference encountered a singular pivot",
                                  std::abs(matrix[pivot_row * n + pivot_col]) > 1.0e-14);

        if (pivot_row != pivot_col)
        {
            for (int col = pivot_col; col < n; ++col)
            {
                std::swap(matrix[pivot_col * n + col], matrix[pivot_row * n + col]);
            }

            std::swap(rhs[pivot_col], rhs[pivot_row]);
        }

        const double diagonal = matrix[pivot_col * n + pivot_col];

        for (int row = pivot_col + 1; row < n; ++row)
        {
            const double multiplier = matrix[row * n + pivot_col] / diagonal;
            matrix[row * n + pivot_col] = 0.0;

            for (int col = pivot_col + 1; col < n; ++col)
            {
                matrix[row * n + col] -= multiplier * matrix[pivot_col * n + col];
            }

            rhs[row] -= multiplier * rhs[pivot_col];
        }
    }

    for (int row = n - 1; row >= 0; --row)
    {
        double value = rhs[row];

        for (int col = row + 1; col < n; ++col)
        {
            value -= matrix[row * n + col] * rhs[col];
        }

        rhs[row] = value / matrix[row * n + row];
    }
}

void run_case(int block_dim, const char *backend, bool reorder_by_color)
{
    const int block_rows = 4;
    const int scalar_rows = block_rows * block_dim;
    Resources resources;
    Matrix_h A_host;
    std::vector<double> dense;
    build_block_tridiagonal(A_host, dense, block_rows, block_dim);

    Vector_h b_host(scalar_rows), expected_host(scalar_rows);
    std::vector<double> rhs(scalar_rows);

    for (int row = 0; row < scalar_rows; ++row)
    {
        rhs[row] = 1.0 + 0.125 * row;
        b_host[row] = ValueTypeB(rhs[row]);
    }

    dense_solve(dense, rhs);

    for (int row = 0; row < scalar_rows; ++row)
    {
        expected_host[row] = ValueTypeB(rhs[row]);
    }

    b_host.set_block_dimx(1);
    b_host.set_block_dimy(block_dim);
    expected_host.set_block_dimx(1);
    expected_host.set_block_dimy(block_dim);

    MatrixA A_device = A_host;
    typename Matrix_h::IVector *row_colors = new typename Matrix_h::IVector(block_rows);

    for (int row = 0; row < block_rows; ++row)
    {
        (*row_colors)[row] = row;
    }

    A_device.template setParameter<int>("coloring_size", block_rows);
    A_device.template setParameter<int>("colors_num", block_rows);
    A_device.template setParameterPtr<typename Matrix_h::IVector>("coloring", row_colors);

    VVector b_device = b_host;
    VVector x_device(scalar_rows, ValueTypeB(0.0));
    x_device.set_block_dimx(1);
    x_device.set_block_dimy(block_dim);

    std::stringstream parameters;
    parameters << "config_version=2, solver(main)=MULTICOLOR_ILU,"
               << "main:ilu_sparsity_level=0, main:max_iters=1,"
               << "main:monitor_residual=0, main:relaxation_factor=1.0,"
               << "main:coloring_level=1, main:matrix_coloring_scheme=MIN_MAX,"
               << "main:max_uncolored_percentage=0.0,"
               << "main:block_ilu_backend=" << backend << ","
               << "main:reorder_cols_by_color=" << (reorder_by_color ? 1 : 0) << ","
               << "main:insert_diag_while_reordering=" << (reorder_by_color ? 1 : 0)
               << ", determinism_flag=1";
    const std::string parameter_string = parameters.str();
    std::string mutable_parameters(parameter_string.size() + 1, 0);
    std::copy(parameter_string.begin(), parameter_string.end(), mutable_parameters.begin());

    AMG_Configuration config;
    UNITTEST_ASSERT_TRUE(config.parseParameterString(mutable_parameters.c_str()) == AMGX_OK);
    AMG_Solver<TConfig> solver(&resources, config);
    solver.setup(A_device);

    AMGX_STATUS status = AMGX_ST_NOT_CONVERGED;
    solver.solve(b_device, x_device, status);

    Vector_h result_host = x_device;
    std::ostringstream description;
    description << backend << " block ILU(0), block size " << block_dim
                << ", differs from the host reference";
    UNITTEST_ASSERT_EQUAL_TOL_DESC(
        description.str().c_str(), result_host, expected_host, 1.0e-10);
}

void run()
{
    // Preserve the existing specialized 4x4 backend as a regression check.
    run_case(4, "amgx", true);

    // Qualify the natural-order cuSPARSE BSR ILU(0) oracle at every HDG
    // polynomial block size p+1 requested for the first checkpoint.
    for (int block_dim = 2; block_dim <= 7; ++block_dim)
    {
        run_case(block_dim, "cusparse_legacy", false);
    }
}

DECLARE_UNITTEST_END(BlockILUBackend);

BlockILUBackend<TemplateMode<AMGX_mode_dDDI>::Type>
BlockILUBackend_instance_mode_dDDI;

} // namespace amgx
