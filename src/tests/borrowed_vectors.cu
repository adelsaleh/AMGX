// SPDX-License-Identifier: BSD-3-Clause
#include "unit_test.h"
#include "amg_solver.h"
#include <thrust/fill.h>

namespace amgx
{
DECLARE_UNITTEST_BEGIN(BorrowedVectors);

void run()
{
    Resources resources;
    // DistributedManager compares integer Vector objects even in a non-MPI
    // build. Preserve the comparisons formerly inherited from Thrust.
    {
        IVector map_a(3, 1), map_b(3, 1), shorter(2, 1), empty_a, empty_b;
        const IVector &const_a = map_a;
        const IVector &const_b = map_b;
        UNITTEST_ASSERT_TRUE(const_a == const_b);
        UNITTEST_ASSERT_TRUE(!(const_a != const_b));
        UNITTEST_ASSERT_TRUE(map_a != shorter);
        UNITTEST_ASSERT_TRUE(empty_a == empty_b);
        map_b[1] = 2;
        UNITTEST_ASSERT_TRUE(const_a != const_b);
        UNITTEST_ASSERT_TRUE(!(const_a == const_b));
        // Coloring exchanges owning allocations without moving their elements.
        auto *map_ptr = map_a.raw();
        auto *shorter_ptr = shorter.raw();
        map_a.swap(shorter);
        UNITTEST_ASSERT_TRUE(map_a.raw() == shorter_ptr && map_a.size() == 2);
        UNITTEST_ASSERT_TRUE(shorter.raw() == map_ptr && shorter.size() == 3);
        shorter.swap(map_a);
        UNITTEST_ASSERT_TRUE(map_a.raw() == map_ptr && map_a.size() == 3);
        UNITTEST_ASSERT_TRUE(shorter.raw() == shorter_ptr && shorter.size() == 2);
    }
    struct Allocation
    {
        ValueTypeB *ptr = nullptr;
        ~Allocation() { if (ptr) cudaFree(ptr); }
    } rhs, solution;
    UNITTEST_ASSERT_TRUE(cudaMalloc(reinterpret_cast<void **>(&rhs.ptr), 3 * sizeof(ValueTypeB)) == cudaSuccess);
    UNITTEST_ASSERT_TRUE(cudaMalloc(reinterpret_cast<void **>(&solution.ptr), 3 * sizeof(ValueTypeB)) == cudaSuccess);
    ValueTypeB input[] = {ValueTypeB(2), ValueTypeB(8), ValueTypeB(18)};
    UNITTEST_ASSERT_TRUE(cudaMemcpy(rhs.ptr, input, sizeof(input), cudaMemcpyHostToDevice) == cudaSuccess);
    {
        VVector b, x;
        b.attach_storage(rhs.ptr, 3);
        x.attach_storage(solution.ptr, 3);
        UNITTEST_ASSERT_TRUE(b.raw() == rhs.ptr && x.raw() == solution.ptr);
        UNITTEST_ASSERT_TRUE(amgx::thrust::raw_pointer_cast(&*x.begin()) == solution.ptr);
        UNITTEST_ASSERT_TRUE(amgx::thrust::raw_pointer_cast(x.data()) == solution.ptr);
        amgx::thrust::fill(x.begin(), x.end(), ValueTypeB(0));
        UNITTEST_ASSERT_TRUE(x.size() == 3 && x.capacity() == 3);
        x.resize(3); // must be a no-op, not an allocation or initialization
        bool rejected = false;
        try { x.resize(4); }
        catch (const amgx_exception &) { rejected = true; }
        UNITTEST_ASSERT_TRUE_DESC("Borrowed vector resize was not rejected", rejected);
        UNITTEST_ASSERT_TRUE(x.raw() == solution.ptr && x.size() == 3);

        Vector_h host_copy(b);
        VVector owned_copy(b);
        Vector_h host_copy2(owned_copy);
        UNITTEST_ASSERT_TRUE(owned_copy.raw() != rhs.ptr && !owned_copy.is_borrowed());
        UNITTEST_ASSERT_TRUE(b == owned_copy && owned_copy == b);
        UNITTEST_ASSERT_TRUE(!(b != owned_copy));
        owned_copy[0] = ValueTypeB(-1);
        UNITTEST_ASSERT_TRUE(b != owned_copy && owned_copy != b);
        UNITTEST_ASSERT_TRUE(b.raw() == rhs.ptr);
        auto *owned_ptr = owned_copy.raw();
        rejected = false;
        try { b.swap(owned_copy); }
        catch (const amgx_exception &) { rejected = true; }
        UNITTEST_ASSERT_TRUE_DESC("Swapping borrowed storage was not rejected", rejected);
        rejected = false;
        try { owned_copy.swap(b); }
        catch (const amgx_exception &) { rejected = true; }
        UNITTEST_ASSERT_TRUE_DESC("Swapping with borrowed storage was not rejected", rejected);
        UNITTEST_ASSERT_TRUE(b.raw() == rhs.ptr && b.is_borrowed() && b.size() == 3);
        UNITTEST_ASSERT_TRUE(owned_copy.raw() == owned_ptr && !owned_copy.is_borrowed());
        for (int i = 0; i < 3; ++i)
        {
            UNITTEST_ASSERT_EQUAL(host_copy[i], input[i]);
            UNITTEST_ASSERT_EQUAL(host_copy2[i], input[i]);
        }

        Matrix_h host_A;
        host_A.set_initialized(0);
        host_A.addProps(CSR);
        host_A.resize(3, 3, 3);
        for (int i = 0; i < 3; ++i)
        {
            host_A.row_offsets[i] = i;
            host_A.col_indices[i] = i;
            host_A.values[i] = ValueTypeA(2 * (i + 1));
        }
        host_A.row_offsets[3] = 3;
        host_A.computeDiagonal();
        host_A.set_initialized(1);
        MatrixA A = host_A;
        AMG_Configuration cfg;
        UNITTEST_ASSERT_TRUE(cfg.parseParameterString(
            "config_version=2, solver(main)=FGMRES, main:preconditioner=NOSOLVER,"
            "main:max_iters=20, main:gmres_n_restart=8, main:monitor_residual=1,"
            "main:convergence=ABSOLUTE, main:tolerance=1e-5") == AMGX_OK);
        AMG_Solver<TConfig> solver(&resources, cfg);
        UNITTEST_ASSERT_TRUE(solver.setup(A) == AMGX_OK);
        AMGX_STATUS status;
        // CUDA-13 AMGX rejects mixed FP32-matrix/FP64-vector SpMV for
        // owning and borrowed vectors alike. Check that existing error path.
        constexpr bool mixed_precision = !std::is_same<ValueTypeA, ValueTypeB>::value;
        const AMGX_ERROR solve_result = solver.solve(b, x, status, true);
        UNITTEST_ASSERT_TRUE(solve_result == (mixed_precision ? AMGX_ERR_NOT_IMPLEMENTED : AMGX_OK));
        UNITTEST_ASSERT_TRUE(cudaStreamSynchronize(cudaStreamLegacy) == cudaSuccess);
        if (!mixed_precision) UNITTEST_ASSERT_TRUE(status == AMGX_ST_CONVERGED);
        ValueTypeB output[3], after_rhs[3];
        UNITTEST_ASSERT_TRUE(cudaMemcpy(output, solution.ptr, sizeof(output), cudaMemcpyDeviceToHost) == cudaSuccess);
        UNITTEST_ASSERT_TRUE(cudaMemcpy(after_rhs, rhs.ptr, sizeof(after_rhs), cudaMemcpyDeviceToHost) == cudaSuccess);
        for (int i = 0; i < 3; ++i)
        {
            if (!mixed_precision) UNITTEST_ASSERT_EQUAL_TOL(output[i], ValueTypeB(i + 1), 1e-5);
            UNITTEST_ASSERT_EQUAL(after_rhs[i], input[i]);
        }
        UNITTEST_ASSERT_TRUE(b.raw() == rhs.ptr && x.raw() == solution.ptr);
        x.detach_storage();
        UNITTEST_ASSERT_TRUE(x.empty() && !x.is_borrowed());
        // b remains attached through destruction; it must not free rhs.ptr.
    }
    ValueTypeB after_destruction[3];
    UNITTEST_ASSERT_TRUE(cudaMemcpy(after_destruction, rhs.ptr, sizeof(after_destruction), cudaMemcpyDeviceToHost) == cudaSuccess);
    for (int i = 0; i < 3; ++i) UNITTEST_ASSERT_EQUAL(after_destruction[i], input[i]);
}

DECLARE_UNITTEST_END(BorrowedVectors);
BorrowedVectors<TemplateMode<AMGX_mode_dDDI>::Type> borrowed_vectors_dDDI;
BorrowedVectors<TemplateMode<AMGX_mode_dDFI>::Type> borrowed_vectors_dDFI;
BorrowedVectors<TemplateMode<AMGX_mode_dFFI>::Type> borrowed_vectors_dFFI;
} // namespace amgx
