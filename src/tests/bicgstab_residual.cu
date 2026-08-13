// SPDX-FileCopyrightText: 2011 - 2026 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include "unit_test.h"
#include "amg_solver.h"
#include <algorithm>
#include <blas.h>
#include <multiply.h>
#include <norm.h>
#include <sstream>

namespace amgx
{

DECLARE_UNITTEST_BEGIN(BiCGStabResidual);

void run()
{
    Resources resources;
    Matrix_h A;
    A.set_initialized(0);
    A.addProps(CSR);
    A.resize(3, 3, 7);
    A.row_offsets[0] = 0;
    A.row_offsets[1] = 2;
    A.row_offsets[2] = 5;
    A.row_offsets[3] = 7;
    A.col_indices[0] = 0;
    A.col_indices[1] = 1;
    A.col_indices[2] = 0;
    A.col_indices[3] = 1;
    A.col_indices[4] = 2;
    A.col_indices[5] = 1;
    A.col_indices[6] = 2;
    A.values[0] = ValueTypeA(4.0);
    A.values[1] = ValueTypeA(1.0);
    A.values[2] = ValueTypeA(2.0);
    A.values[3] = ValueTypeA(3.0);
    A.values[4] = ValueTypeA(1.0);
    A.values[5] = ValueTypeA(1.0);
    A.values[6] = ValueTypeA(2.0);
    A.set_block_dimx(1);
    A.set_block_dimy(1);
    A.computeDiagonal();
    A.set_initialized(1);

    Vector_h b(3), x_initial(3, ValueTypeB(0.0));
    b[0] = ValueTypeB(1.0);
    b[1] = ValueTypeB(2.0);
    b[2] = ValueTypeB(3.0);
    b.set_block_dimx(1);
    b.set_block_dimy(1);
    x_initial.set_block_dimx(1);
    x_initial.set_block_dimy(1);

    MatrixA A_device = A;
    VVector b_device = b;
    VVector x_device = x_initial;

    std::stringstream parameters;
    parameters << "config_version=2, solver(main)=BICGSTAB,"
               << "main:preconditioner=NOSOLVER, main:max_iters=1,"
               << "main:norm=L2, main:tolerance=1e-30,"
               << "main:convergence=ABSOLUTE, main:monitor_residual=1,"
               << "main:store_res_history=1, determinism_flag=1";
    const std::string parameter_string = parameters.str();
    std::string mutable_parameters(parameter_string.size() + 1, 0);
    std::copy(parameter_string.begin(), parameter_string.end(), mutable_parameters.begin());

    AMG_Configuration config;
    UNITTEST_ASSERT_TRUE(config.parseParameterString(mutable_parameters.c_str()) == AMGX_OK);
    AMG_Solver<TConfig> solver(&resources, config);
    solver.setup(A_device);

    AMGX_STATUS status = AMGX_ST_NOT_CONVERGED;
    solver.solve(b_device, x_device, status);

    VVector explicit_residual(3, ValueTypeB(0.0));
    explicit_residual.set_block_dimx(1);
    explicit_residual.set_block_dimy(1);
    multiply(A_device, x_device, explicit_residual);
    axpby(b_device, explicit_residual, explicit_residual, ValueTypeB(1.0), ValueTypeB(-1.0));

    Vector_h explicit_norm(1);
    get_norm(A_device, explicit_residual, 1, L2, explicit_norm);
    const auto &reported_norm = solver.get_residual(1);
    const double tolerance = sizeof(ValueTypeB) == 4 ? 1.0e-5 : 1.0e-12;

    UNITTEST_ASSERT_TRUE_DESC("BiCGSTAB did not execute exactly one iteration", solver.get_num_iters() == 1);
    UNITTEST_ASSERT_EQUAL_TOL_DESC(
        "BiCGSTAB stored residual does not match ||b-Ax|| after iteration 1",
        reported_norm[0], explicit_norm[0], tolerance);
}

DECLARE_UNITTEST_END(BiCGStabResidual);

BiCGStabResidual<TemplateMode<AMGX_mode_dDDI>::Type> BiCGStabResidual_instance_mode_dDDI;
BiCGStabResidual<TemplateMode<AMGX_mode_dFFI>::Type> BiCGStabResidual_instance_mode_dFFI;

} // namespace amgx
