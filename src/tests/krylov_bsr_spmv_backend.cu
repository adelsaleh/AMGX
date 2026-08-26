// SPDX-FileCopyrightText: 2011 - 2026 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include "unit_test.h"
#include "amg_solver.h"
#include <algorithm>
#include <blas.h>
#include <multiply.h>
#include <sstream>

namespace amgx
{

DECLARE_UNITTEST_BEGIN(KrylovBsrSpmvBackend);

void build_block_identity(Matrix_h &A, int block_dim)
{
    const int block_rows = 2;
    const int block_size = block_dim * block_dim;
    A.set_initialized(0);
    A.addProps(CSR);
    A.resize(block_rows, block_rows, block_rows, block_dim, block_dim);

    for (int row = 0; row <= block_rows; ++row)
    {
        A.row_offsets[row] = row;
    }

    for (int row = 0; row < block_rows; ++row)
    {
        A.col_indices[row] = row;

        for (int entry = 0; entry < block_size; ++entry)
        {
            A.values[row * block_size + entry] = ValueTypeA(0.0);
        }

        for (int component = 0; component < block_dim; ++component)
        {
            A.values[row * block_size + component * block_dim + component] = ValueTypeA(1.0);
        }
    }

    A.computeDiagonal();
    A.set_initialized(1);
}

void check_solver(const std::string &solver_name)
{
    const int block_dim = 3;
    const int scalar_size = 2 * block_dim;
    Resources resources;
    Matrix_h A_host;
    build_block_identity(A_host, block_dim);
    MatrixA A_device = A_host;

    std::stringstream parameters;
    parameters << "config_version=2, solver(main)=" << solver_name << ","
               << "main:bsr_spmv_backend=cusparse_generic,"
               << "main:max_iters=1, main:monitor_residual=0,";

    if (solver_name == "PBICGSTAB")
    {
        // Deliberately let the nested solver select legacy. The outer Krylov
        // scope must be applied after preconditioner setup.
        parameters << "main:preconditioner(preconditioner)=BLOCK_JACOBI,"
                   << "preconditioner:bsr_spmv_backend=legacy,";
    }

    parameters << "determinism_flag=1";
    const std::string parameter_string = parameters.str();
    std::string mutable_parameters(parameter_string.size() + 1, 0);
    std::copy(parameter_string.begin(), parameter_string.end(), mutable_parameters.begin());

    AMG_Configuration config;
    UNITTEST_ASSERT_TRUE(config.parseParameterString(mutable_parameters.c_str()) == AMGX_OK);
    AMG_Solver<TConfig> solver(&resources, config);
    solver.setup(A_device);

#if CUDART_VERSION >= 13000
    const int expected_backend = 1;
#else
    const int expected_backend = 0;
#endif

    UNITTEST_ASSERT_EQUAL_DESC(
        "Krylov solver did not propagate its BSR SpMV backend to the fine matrix",
        A_device.template getParameter<int>("bsr_spmv_backend"), expected_backend);
    UNITTEST_ASSERT_EQUAL_DESC(
        "cusparse_generic must disable the custom 5x5 subgroup dispatcher",
        A_device.template getParameter<int>("use_subgroup_5x5_spmv"), 0);

    VVector x(scalar_size, ValueTypeB(1.0));
    VVector y(scalar_size, ValueTypeB(0.0));
    x.set_block_dimx(1);
    x.set_block_dimy(block_dim);
    y.set_block_dimx(1);
    y.set_block_dimy(block_dim);
    multiply(A_device, x, y);
    cudaCheckError();
    UNITTEST_ASSERT_EQUAL_TOL_DESC(
        "3x3 identity BSR multiply failed after Krylov backend propagation",
        x, y, sizeof(ValueTypeB) == 4 ? 1.0e-6 : 1.0e-12);
}

void run()
{
    check_solver("BICGSTAB");
    check_solver("PBICGSTAB");
}

DECLARE_UNITTEST_END(KrylovBsrSpmvBackend);

KrylovBsrSpmvBackend<TemplateMode<AMGX_mode_dDDI>::Type>
KrylovBsrSpmvBackend_instance_mode_dDDI;

} // namespace amgx
