// SPDX-FileCopyrightText: 2011 - 2026 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include "unit_test.h"
#include "amg_solver.h"
#include "test_utils.h"
#include <blas.h>
#include <cusp/gallery/poisson.h>
#include <multiply.h>
#include <norm.h>
#include <sstream>

namespace amgx
{

DECLARE_UNITTEST_BEGIN(GMRESReliableResidual);

void run()
{
    Resources resources;
    Matrix_h A;
    A.set_initialized(0);
    A.addProps(CSR);
    MatrixCusp<TConfig_h, cusp::csr_format> wrapped_A(&A);
    cusp::gallery::poisson5pt(wrapped_A, 12, 12);
    A.computeDiagonal();
    A.set_initialized(1);

    const int n_rows = A.get_num_rows();
    Vector_h b(n_rows, ValueTypeB(1.0));
    Vector_h x_initial(n_rows, ValueTypeB(0.0));
    b.set_block_dimx(1);
    b.set_block_dimy(1);
    x_initial.set_block_dimx(1);
    x_initial.set_block_dimy(1);

    MatrixA A_device = A;
    VVector b_device = b;
    VVector x_device = x_initial;
    VVector residual(n_rows, ValueTypeB(0.0));
    residual.set_block_dimx(1);
    residual.set_block_dimy(1);

    const double requested_tolerance = sizeof(ValueTypeB) == 4 ? 1.0e-5 : 1.0e-10;
    std::stringstream parameters;
    parameters << "config_version=2, solver(main)=GMRES, main:preconditioner(jacobi)=BLOCK_JACOBI,"
               << "jacobi:max_iters=1, main:max_iters=200, main:norm=L2,"
               << "main:tolerance=" << requested_tolerance << ", main:gmres_n_restart=20,"
               << "main:gmres_reliable_residual=1, main:gmres_reorthogonalization=DGKS,"
               << "main:convergence=RELATIVE_INI_CORE, main:monitor_residual=1";
    const std::string parameter_string = parameters.str();
    std::string mutable_parameters(parameter_string.size() + 1, '\0');
    std::copy(parameter_string.begin(), parameter_string.end(), mutable_parameters.begin());

    AMG_Configuration config;
    UNITTEST_ASSERT_TRUE(config.parseParameterString(mutable_parameters.c_str()) == AMGX_OK);
    AMG_Solver<TConfig> solver(&resources, config);
    solver.setup(A_device);

    AMGX_STATUS status = AMGX_ST_NOT_CONVERGED;
    solver.solve(b_device, x_device, status);
    UNITTEST_ASSERT_TRUE_DESC("reliable GMRES did not report convergence", status == AMGX_ST_CONVERGED);

    multiply(A_device, x_device, residual);
    axpby(b_device, residual, residual, ValueTypeB(1.0), ValueTypeB(-1.0));
    Vector_h initial_norm(1), final_norm(1);
    get_norm(A_device, b_device, 1, L2, initial_norm);
    get_norm(A_device, residual, 1, L2, final_norm);
    UNITTEST_ASSERT_TRUE_DESC(
        "reliable GMRES reported convergence above the explicit residual tolerance",
        final_norm[0] / initial_norm[0] <= requested_tolerance);

    thrust_wrapper::fill<TConfig::memSpace>(b_device.begin(), b_device.end(), ValueTypeB(0.0));
    thrust_wrapper::fill<TConfig::memSpace>(x_device.begin(), x_device.end(), ValueTypeB(0.0));
    status = AMGX_ST_NOT_CONVERGED;
    solver.solve(b_device, x_device, status);
    UNITTEST_ASSERT_TRUE_DESC("zero-residual system did not converge immediately", status == AMGX_ST_CONVERGED);
}

DECLARE_UNITTEST_END(GMRESReliableResidual);

GMRESReliableResidual<TemplateMode<AMGX_mode_dDDI>::Type> GMRESReliableResidual_instance_mode_dDDI;
GMRESReliableResidual<TemplateMode<AMGX_mode_dFFI>::Type> GMRESReliableResidual_instance_mode_dFFI;

} // namespace amgx
