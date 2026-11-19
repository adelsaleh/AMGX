// SPDX-License-Identifier: BSD-3-Clause
// Borrowed square-block BSR: the borrowed_csr test on a 2x2-block matrix (two block rows,
// four blocks), with the BSR-specific guards: byte capacity per block, rectangular blocks
// rejected, pointer identity through setup and solve, values untouched by detach/destroy.
#include "unit_test.h"
#include "amgx_c.h"

namespace amgx
{
DECLARE_UNITTEST_BEGIN(BorrowedBSR);
void run()
{
    const AMGX_Mode mode = static_cast<AMGX_Mode>(TConfig::mode);
    AMGX_config_handle cfg;
    UNITTEST_ASSERT_EQUAL(AMGX_config_create(&cfg,
        "config_version=2,exception_handling=0,solver(main)=PBICGSTAB,"
        "main:preconditioner(pre)=BLOCK_JACOBI,pre:max_iters=1,main:max_iters=40,"
        "main:monitor_residual=1,main:convergence=ABSOLUTE,main:tolerance=1e-8"), AMGX_RC_OK);
    AMGX_resources_handle resources;
    UNITTEST_ASSERT_EQUAL(AMGX_resources_create_simple(&resources, cfg), AMGX_RC_OK);
    AMGX_matrix_handle matrix;
    AMGX_solver_handle solver;
    AMGX_vector_handle b, x;
    UNITTEST_ASSERT_EQUAL(AMGX_matrix_create(&matrix, resources, mode), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_solver_create(&solver, resources, mode, cfg), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_vector_create(&b, resources, mode), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_vector_create(&x, resources, mode), AMGX_RC_OK);
    {
        // [[A00 A01], [A10 A11]] with 2x2 blocks: A00 = A11 = 4I + [[0,1],[1,0]], A01 = A10 = -I.
        IVector rows(3), cols(4);
        MVector data(16);
        int host_rows[] = {0, 2, 4}, host_cols[] = {0, 1, 0, 1};
        ValueTypeA host_data[] = {4, 1, 1, 4, -1, 0, 0, -1, -1, 0, 0, -1, 4, 1, 1, 4};
        cudaMemcpy(rows.raw(), host_rows, sizeof(host_rows), cudaMemcpyHostToDevice);
        cudaMemcpy(cols.raw(), host_cols, sizeof(host_cols), cudaMemcpyHostToDevice);
        cudaMemcpy(data.raw(), host_data, sizeof(host_data), cudaMemcpyHostToDevice);
        // Rectangular blocks and a values buffer too small for the blocks are refused.
        // Reject dimensions/counts that overflow the internal block/scalar indexing.
        UNITTEST_ASSERT_EQUAL(AMGX_matrix_attach_bsr(matrix, 2, 4, 65536, 65536, rows.raw(), cols.raw(), data.raw(),
            rows.bytes(), cols.bytes(), data.bytes(), 1, 1, 1), AMGX_RC_BAD_PARAMETERS);
        UNITTEST_ASSERT_EQUAL(AMGX_matrix_attach_bsr(matrix, 2, 536870911, 2, 2, rows.raw(), cols.raw(), data.raw(),
            rows.bytes(), cols.bytes(), data.bytes(), 1, 1, 1), AMGX_RC_BAD_PARAMETERS);
        UNITTEST_ASSERT_EQUAL(AMGX_matrix_attach_bsr(matrix, 2, 4, 2, 1, rows.raw(), cols.raw(), data.raw(),
            rows.bytes(), cols.bytes(), data.bytes(), 1, 1, 1), AMGX_RC_BAD_PARAMETERS);
        UNITTEST_ASSERT_EQUAL(AMGX_matrix_attach_bsr(matrix, 2, 4, 2, 2, rows.raw(), cols.raw(), data.raw(),
            rows.bytes(), cols.bytes(), 4 * sizeof(ValueTypeA), 1, 1, 1), AMGX_RC_BAD_PARAMETERS);
        UNITTEST_ASSERT_EQUAL(AMGX_matrix_attach_bsr(matrix, 2, 4, 2, 2, rows.raw(), cols.raw(), data.raw(),
            rows.bytes(), cols.bytes(), data.bytes(), 1, 1, 1), AMGX_RC_OK);
        void *rp, *cp, *vp;
        UNITTEST_ASSERT_EQUAL(AMGX_matrix_get_attached_data(matrix, &rp, &cp, &vp), AMGX_RC_OK);
        UNITTEST_ASSERT_TRUE(rp == rows.raw() && cp == cols.raw() && vp == data.raw());
        int n, bx, by;
        UNITTEST_ASSERT_EQUAL(AMGX_matrix_get_size(matrix, &n, &bx, &by), AMGX_RC_OK);
        UNITTEST_ASSERT_TRUE(n == 2 && bx == 2 && by == 2);
        UNITTEST_ASSERT_EQUAL(AMGX_matrix_upload_all(matrix, 2, 4, 2, 2,
            rows.raw(), cols.raw(), data.raw(), nullptr), AMGX_RC_BAD_PARAMETERS);
        // b = A * [1, 2, 3, 4]: block row 0 = A00 [1,2] + A01 [3,4] = [6,9] - [3,4] = [3,5];
        // block row 1 = A10 [1,2] + A11 [3,4] = [-1,-2] + [16,19] = [15,17].
        ValueTypeB rhs[] = {3, 5, 15, 17}, output[4];
        UNITTEST_ASSERT_EQUAL(AMGX_vector_upload(b, 2, 2, rhs), AMGX_RC_OK);
        UNITTEST_ASSERT_EQUAL(AMGX_vector_set_zero(x, 2, 2), AMGX_RC_OK);
        UNITTEST_ASSERT_EQUAL(AMGX_solver_setup(solver, matrix), AMGX_RC_OK);
        UNITTEST_ASSERT_EQUAL(AMGX_matrix_detach(matrix), AMGX_RC_BAD_PARAMETERS);
        const bool mixed = !std::is_same<ValueTypeA, ValueTypeB>::value;
        UNITTEST_ASSERT_EQUAL(AMGX_solver_solve(solver, b, x), mixed ? AMGX_RC_NOT_IMPLEMENTED : AMGX_RC_OK);
        if (!mixed)
        {
            UNITTEST_ASSERT_EQUAL(AMGX_vector_download(x, output), AMGX_RC_OK);
            for (int i = 0; i < 4; ++i) UNITTEST_ASSERT_EQUAL_TOL(output[i], ValueTypeB(i + 1), 1e-5);
        }
        UNITTEST_ASSERT_EQUAL(AMGX_matrix_get_attached_data(matrix, &rp, &cp, &vp), AMGX_RC_OK);
        UNITTEST_ASSERT_TRUE(rp == rows.raw() && cp == cols.raw() && vp == data.raw());
        UNITTEST_ASSERT_EQUAL(AMGX_solver_destroy(solver), AMGX_RC_OK);
        UNITTEST_ASSERT_EQUAL(AMGX_matrix_detach(matrix), AMGX_RC_OK);
        UNITTEST_ASSERT_EQUAL(AMGX_matrix_destroy(matrix), AMGX_RC_OK);
        ValueTypeA after[16];
        UNITTEST_ASSERT_TRUE(cudaMemcpy(after, data.raw(), sizeof(after), cudaMemcpyDeviceToHost) == cudaSuccess);
        for (int i = 0; i < 16; ++i) UNITTEST_ASSERT_EQUAL(after[i], host_data[i]);
    }
    UNITTEST_ASSERT_EQUAL(AMGX_vector_destroy(x), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_vector_destroy(b), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_resources_destroy(resources), AMGX_RC_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_config_destroy(cfg), AMGX_RC_OK);
}
DECLARE_UNITTEST_END(BorrowedBSR);
BorrowedBSR<TemplateMode<AMGX_mode_dDDI>::Type> borrowed_bsr_dDDI;
BorrowedBSR<TemplateMode<AMGX_mode_dFFI>::Type> borrowed_bsr_dFFI;
BorrowedBSR<TemplateMode<AMGX_mode_dDFI>::Type> borrowed_bsr_dDFI;
} // namespace amgx
