// SPDX-License-Identifier: BSD-3-Clause
#pragma once

// Included inside namespace amgx, after borrowed_vector_api.h.
enum class BorrowedMatrixOperation { attach, detach, synchronize, query };

// Inspect on the GPU; only the validation status crosses to the host.
// Diagonal-based AMGX solvers assume every row has an explicit diagonal.
__global__ void validate_borrowed_csr(int n, int nnz, const int *rows,
                                    const int *cols, int *invalid)
{
    for (size_t row = blockIdx.x * blockDim.x + threadIdx.x; row < static_cast<size_t>(n);
         row += blockDim.x * gridDim.x)
    {
        const int first = rows[row], last = rows[row + 1];
        if (first < 0 || last < first || last > nnz ||
            (row == 0 && first != 0) || (row == n - 1 && last != nnz))
        {
            atomicExch(invalid, 1);
            continue;
        }
        bool diagonal = false;
        int previous = -1;
        for (int j = first; j < last; ++j)
        {
            const int col = cols[j];
            if (col <= previous || col >= n) atomicExch(invalid, 1);
            diagonal = diagonal || col == row;
            previous = col;
        }
        if (!diagonal) atomicExch(invalid, 1);
    }
}

inline bool borrowed_csr_pointer(void *ptr, size_t bytes, size_t alignment, int device)
{
    if (!ptr || reinterpret_cast<uintptr_t>(ptr) % alignment ||
        bytes > SIZE_MAX - reinterpret_cast<uintptr_t>(ptr)) return false;
    cudaPointerAttributes attr{};
    if (cudaPointerGetAttributes(&attr, ptr) != cudaSuccess)
    {
        cudaGetLastError();
        return false;
    }
    return attr.type == cudaMemoryTypeDevice && attr.device == device;
}

template<AMGX_Mode CASE>
AMGX_RC borrowed_matrix_operation(AMGX_matrix_handle handle,
        BorrowedMatrixOperation op, int n, int nnz, int *rows, int *cols, void *data,
        size_t rows_bytes, size_t cols_bytes, size_t data_bytes,
        uintptr_t rows_stream, uintptr_t cols_stream, uintptr_t data_stream,
        void **out_rows, void **out_cols, void **out_data)
{
    using Config = typename TemplateMode<CASE>::Type;
    using Mat = Matrix<Config>;
    using Value = typename Mat::value_type;
    CWrapHandle<AMGX_matrix_handle, Mat> wrap(handle);
    Mat &A = *wrap.wrapped();
    const int device = A.getResources()->getDevice(0);
    int previous;
    if (cudaGetDevice(&previous) != cudaSuccess || cudaSetDevice(device) != cudaSuccess)
        return AMGX_RC_CUDA_FAILURE;
    struct Restore { int device; ~Restore() { cudaSetDevice(device); } } restore{previous};
    if (op == BorrowedMatrixOperation::attach)
    {
        if (n <= 0 || n == INT_MAX || nnz < n || A.is_initialized() ||
            A.get_num_rows() != 0 || A.values.is_borrowed() || A.manager != nullptr ||
            wrap.wrapped().use_count() != 2 ||
            rows_bytes < (static_cast<size_t>(n) + 1) * sizeof(int) ||
            cols_bytes < static_cast<size_t>(nnz) * sizeof(int) ||
            data_bytes < static_cast<size_t>(nnz) * sizeof(Value) ||
            !borrowed_csr_pointer(rows, rows_bytes, alignof(int), device) ||
            !borrowed_csr_pointer(cols, cols_bytes, alignof(int), device) ||
            !borrowed_csr_pointer(data, data_bytes, alignof(Value), device))
            return AMGX_RC_BAD_PARAMETERS;
        const uintptr_t ptrs[] = {reinterpret_cast<uintptr_t>(rows),
            reinterpret_cast<uintptr_t>(cols), reinterpret_cast<uintptr_t>(data)};
        const size_t sizes[] = {(static_cast<size_t>(n) + 1) * sizeof(int),
            static_cast<size_t>(nnz) * sizeof(int), static_cast<size_t>(nnz) * sizeof(Value)};
        for (int i = 0; i < 3; ++i)
            for (int j = i + 1; j < 3; ++j)
                if (ptrs[i] < ptrs[j] ? ptrs[j] - ptrs[i] < sizes[i] : ptrs[i] - ptrs[j] < sizes[j])
                    return AMGX_RC_BAD_PARAMETERS;
    }
    else if (!A.values.is_borrowed()) return AMGX_RC_BAD_PARAMETERS;

    if (op == BorrowedMatrixOperation::query)
    {
        if (!out_rows || !out_cols || !out_data) return AMGX_RC_BAD_PARAMETERS;
        *out_rows = A.row_offsets.raw(); *out_cols = A.col_indices.raw(); *out_data = A.values.raw();
        return AMGX_RC_OK;
    }
    if (op == BorrowedMatrixOperation::detach)
    {
        // One owning C handle plus this temporary wrapper. Solvers retain a
        // shared_ptr even after setup failure, so they must be destroyed first.
        if (wrap.wrapped().use_count() != 2) return AMGX_RC_BAD_PARAMETERS;
        if (cudaStreamSynchronize(cudaStreamLegacy) != cudaSuccess) return AMGX_RC_CUDA_FAILURE;
        A.set_initialized(0);
        A.row_offsets.detach_storage(); A.col_indices.detach_storage(); A.values.detach_storage();
        A.set_num_rows(0); A.set_num_cols(0); A.set_num_nz(0);
        A.set_is_matrix_setup(false);
        return AMGX_RC_OK;
    }
    for (uintptr_t stream : {rows_stream, cols_stream, data_stream})
    {
        const AMGX_RC result = synchronize_borrowed_producer(stream, device);
        if (result != AMGX_RC_OK) return result;
    }
    if (op == BorrowedMatrixOperation::synchronize) return AMGX_RC_OK;

    int *invalid = nullptr;
    if (cudaMalloc(reinterpret_cast<void **>(&invalid), sizeof(int)) != cudaSuccess)
        return AMGX_RC_CUDA_FAILURE;
    struct Free { int *p; ~Free() { cudaFree(p); } } free_invalid{invalid};
    int host_invalid = 0;
    if (cudaMemsetAsync(invalid, 0, sizeof(int), cudaStreamLegacy) != cudaSuccess)
        return AMGX_RC_CUDA_FAILURE;
    validate_borrowed_csr<<<std::min(4096, (n - 1) / 256 + 1), 256, 0, cudaStreamLegacy>>>(n, nnz, rows, cols, invalid);
    if (cudaGetLastError() != cudaSuccess ||
        cudaMemcpyAsync(&host_invalid, invalid, sizeof(int), cudaMemcpyDeviceToHost, cudaStreamLegacy) != cudaSuccess ||
        cudaStreamSynchronize(cudaStreamLegacy) != cudaSuccess) return AMGX_RC_CUDA_FAILURE;
    if (host_invalid) return AMGX_RC_BAD_PARAMETERS;

    // The normal resize adds a sentinel value and allocates all three buffers.
    // Initialize only metadata here: CuPy's data allocation has exactly nnz entries.
    // addProps(CSR) would convert COO row indices if dimensions were already
    // nonzero. Set the property while the matrix is still empty, then discard
    // its one-element empty row-offset allocation before borrowing the arrays.
    A.addProps(CSR); A.delProps(COO); A.delProps(DIAG); A.delProps(COLORING);
    A.values.clear(); A.col_indices.clear(); A.row_offsets.clear();
    A.set_num_rows(n); A.set_num_cols(n); A.set_num_nz(nnz);
    A.set_block_dimx(1); A.set_block_dimy(1);
    A.setColsReorderedByColor(false);
    A.set_is_matrix_setup(false);
    try
    {
        A.row_offsets.attach_storage(rows, static_cast<size_t>(n) + 1);
        A.col_indices.attach_storage(cols, nnz);
        A.values.attach_storage(static_cast<Value *>(data), nnz);
        A.m_seq_offsets.resize(static_cast<size_t>(n) + 1);
        thrust_wrapper::sequence<AMGX_device>(A.m_seq_offsets.begin(), A.m_seq_offsets.end());
        A.computeDiagonal();
        A.set_initialized(1);
        if (cudaStreamSynchronize(cudaStreamLegacy) != cudaSuccess)
            FatalError("CSR attachment initialization failed", AMGX_ERR_CUDA_FAILURE);
    }
    catch (...)
    {
        cudaStreamSynchronize(cudaStreamLegacy);
        A.set_initialized(0);
        A.row_offsets.detach_storage(); A.col_indices.detach_storage(); A.values.detach_storage();
        A.set_num_rows(0); A.set_num_cols(0); A.set_num_nz(0);
        throw;
    }
    return AMGX_RC_OK;
}

inline AMGX_RC dispatch_borrowed_matrix(AMGX_matrix_handle handle,
        BorrowedMatrixOperation op, int n = 0, int nnz = 0, int *rows = nullptr,
        int *cols = nullptr, void *data = nullptr, size_t rows_bytes = 0,
        size_t cols_bytes = 0, size_t data_bytes = 0, uintptr_t rows_stream = 0,
        uintptr_t cols_stream = 0, uintptr_t data_stream = 0,
        void **out_rows = nullptr, void **out_cols = nullptr, void **out_data = nullptr)
{
    Resources *resources = nullptr;
    AMGX_CHECK_API_ERROR_NORSRC(getAMGXerror(getResourcesFromMatrixHandle(handle, &resources)))
    AMGX_ERROR rc = AMGX_OK;
    AMGX_RC result = AMGX_RC_BAD_MODE;
    AMGX_TRIES()
    {
        switch (get_mode_from<AMGX_matrix_handle>(handle))
        {
#define AMGX_CASE_LINE(CASE) case CASE: result = borrowed_matrix_operation<CASE>(handle, op, n, nnz, rows, cols, data, rows_bytes, cols_bytes, data_bytes, rows_stream, cols_stream, data_stream, out_rows, out_cols, out_data); break;
            AMGX_FORALL_BUILDS_DEVICE(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE
            default: return AMGX_RC_BAD_MODE;
        }
    }
    AMGX_CATCHES(rc)
    AMGX_CHECK_API_ERROR(rc, resources)
    return result;
}
