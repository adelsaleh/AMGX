// SPDX-License-Identifier: BSD-3-Clause
#include "unit_test.h"
#include "amgx_c.h"

namespace amgx
{
DECLARE_UNITTEST_BEGIN(BorrowedCSR);
void run()
{
    const AMGX_Mode mode = static_cast<AMGX_Mode>(TConfig::mode);
    AMGX_config_handle cfg;
    UNITTEST_ASSERT_EQUAL(AMGX_config_create(&cfg,
        "config_version=2,exception_handling=0,solver(main)=FGMRES,"
        "main:preconditioner=NOSOLVER,main:max_iters=20,main:gmres_n_restart=8,"
        "main:monitor_residual=1,main:convergence=ABSOLUTE,main:tolerance=1e-5"), AMGX_RC_OK);
    AMGX_resources_handle resources;
    UNITTEST_ASSERT_EQUAL(AMGX_resources_create_simple(&resources, cfg), AMGX_RC_OK);
    AMGX_matrix_handle matrix;
    AMGX_solver_handle solver;
    AMGX_vector_handle b, x;
    UNITTEST_ASSERT_EQUAL(AMGX_matrix_create(&matrix, resources, mode), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_solver_create(&solver, resources, mode, cfg), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_vector_create(&b, resources, mode), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_vector_create(&x, resources, mode), AMGX_RC_OK);
    // Release fixture allocations before destroying their resource memory pool.
    {
        IVector rows(3), cols(4);
        MVector data(4);
        int host_rows[] = {0, 2, 4}, host_cols[] = {0, 1, 0, 1};
        ValueTypeA host_data[] = {4, 1, 1, 3};
        cudaMemcpy(rows.raw(), host_rows, sizeof(host_rows), cudaMemcpyHostToDevice);
        cudaMemcpy(cols.raw(), host_cols, sizeof(host_cols), cudaMemcpyHostToDevice);
        cudaMemcpy(data.raw(), host_data, sizeof(host_data), cudaMemcpyHostToDevice);
        UNITTEST_ASSERT_EQUAL(AMGX_matrix_attach_csr(matrix, 2, 4, rows.raw(), cols.raw(), data.raw(),
            rows.bytes(), cols.bytes(), data.bytes(), 1, 1, 1), AMGX_RC_OK);
        void *rp, *cp, *vp;
        UNITTEST_ASSERT_EQUAL(AMGX_matrix_get_attached_data(matrix, &rp, &cp, &vp), AMGX_RC_OK);
        UNITTEST_ASSERT_TRUE(rp == rows.raw() && cp == cols.raw() && vp == data.raw());
        UNITTEST_ASSERT_EQUAL(AMGX_matrix_replace_coefficients(matrix, 2, 4, data.raw(), nullptr), AMGX_RC_BAD_PARAMETERS);
        UNITTEST_ASSERT_EQUAL(AMGX_matrix_upload_all(matrix, 2, 4, 1, 1,
            rows.raw(), cols.raw(), data.raw(), nullptr), AMGX_RC_BAD_PARAMETERS);
        ValueTypeB rhs[] = {6, 7}, output[2];
        UNITTEST_ASSERT_EQUAL(AMGX_vector_upload(b, 2, 1, rhs), AMGX_RC_OK);
        UNITTEST_ASSERT_EQUAL(AMGX_vector_set_zero(x, 2, 1), AMGX_RC_OK);
        UNITTEST_ASSERT_EQUAL(AMGX_solver_setup(solver, matrix), AMGX_RC_OK);
        UNITTEST_ASSERT_EQUAL(AMGX_matrix_detach(matrix), AMGX_RC_BAD_PARAMETERS);
        UNITTEST_ASSERT_EQUAL(AMGX_matrix_destroy(matrix), AMGX_RC_BAD_PARAMETERS);
        const bool mixed = !std::is_same<ValueTypeA, ValueTypeB>::value;
        UNITTEST_ASSERT_EQUAL(AMGX_solver_solve(solver, b, x), mixed ? AMGX_RC_NOT_IMPLEMENTED : AMGX_RC_OK);
        if (!mixed)
        {
            UNITTEST_ASSERT_EQUAL(AMGX_vector_download(x, output), AMGX_RC_OK);
            UNITTEST_ASSERT_EQUAL_TOL(output[0], ValueTypeB(1), 1e-5);
            UNITTEST_ASSERT_EQUAL_TOL(output[1], ValueTypeB(2), 1e-5);
        }
        UNITTEST_ASSERT_EQUAL(AMGX_matrix_get_attached_data(matrix, &rp, &cp, &vp), AMGX_RC_OK);
        UNITTEST_ASSERT_TRUE(rp == rows.raw() && cp == cols.raw() && vp == data.raw());
        UNITTEST_ASSERT_EQUAL(AMGX_solver_destroy(solver), AMGX_RC_OK);
        UNITTEST_ASSERT_EQUAL(AMGX_matrix_detach(matrix), AMGX_RC_OK);
        UNITTEST_ASSERT_EQUAL(AMGX_matrix_attach_csr(matrix, 2, 4, rows.raw(), cols.raw(), data.raw(),
            rows.bytes(), cols.bytes(), data.bytes(), 1, 1, 1), AMGX_RC_OK);
        UNITTEST_ASSERT_EQUAL(AMGX_matrix_destroy(matrix), AMGX_RC_OK);
        ValueTypeA after[4];
        UNITTEST_ASSERT_TRUE(cudaMemcpy(after, data.raw(), sizeof(after), cudaMemcpyDeviceToHost) == cudaSuccess);
        for (int i = 0; i < 4; ++i) UNITTEST_ASSERT_EQUAL(after[i], host_data[i]);
    }
    UNITTEST_ASSERT_EQUAL(AMGX_vector_destroy(x), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_vector_destroy(b), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_resources_destroy(resources), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_config_destroy(cfg), AMGX_RC_OK);
}
DECLARE_UNITTEST_END(BorrowedCSR);
BorrowedCSR<TemplateMode<AMGX_mode_dDDI>::Type> borrowed_csr_dDDI;
BorrowedCSR<TemplateMode<AMGX_mode_dFFI>::Type> borrowed_csr_dFFI;
BorrowedCSR<TemplateMode<AMGX_mode_dDFI>::Type> borrowed_csr_dDFI;
} // namespace amgx
