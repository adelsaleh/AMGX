// SPDX-License-Identifier: BSD-3-Clause
#pragma once

// Included by amgx_c.cu after its handle helpers, inside namespace amgx.
enum class BorrowedVectorOperation { attach, detach, synchronize, query };

inline AMGX_RC synchronize_borrowed_producer(uintptr_t stream, int device)
{
    if (!stream) return AMGX_RC_OK;
#if CUDART_VERSION >= 13000
    int stream_device;
    if (cudaStreamGetDevice(reinterpret_cast<cudaStream_t>(stream), &stream_device) != cudaSuccess)
        return AMGX_RC_CUDA_FAILURE;
    if (stream_device != device) return AMGX_RC_BAD_PARAMETERS;
#endif
    return cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)) == cudaSuccess
        ? AMGX_RC_OK : AMGX_RC_CUDA_FAILURE;
}

template<AMGX_Mode CASE>
AMGX_RC borrowed_vector_operation(AMGX_vector_handle handle,
        BorrowedVectorOperation operation, int n, void *data, size_t capacity,
        uintptr_t producer_stream, void **out_data, size_t *out_size, int *out_device)
{
    using Config = typename TemplateMode<CASE>::Type;
    using Vec = Vector<Config>;
    using Value = typename Vec::value_type;
    Vec &v = *get_mode_object_from<CASE, Vector, AMGX_vector_handle>(handle);
    const int device = v.getResources()->getDevice(0);
    int previous_device;
    if (cudaGetDevice(&previous_device) != cudaSuccess || cudaSetDevice(device) != cudaSuccess)
        return AMGX_RC_CUDA_FAILURE;
    struct RestoreDevice
    {
        int device;
        ~RestoreDevice() { cudaSetDevice(device); }
    } restore{previous_device};

    if (operation == BorrowedVectorOperation::attach)
    {
        if (n < 0 || v.is_borrowed() || !v.empty() || v.getManager() != nullptr ||
            capacity < static_cast<size_t>(n) * sizeof(Value) ||
            (n != 0 && data == nullptr) || reinterpret_cast<uintptr_t>(data) % alignof(Value))
            return AMGX_RC_BAD_PARAMETERS;
        if (data != nullptr)
        {
            cudaPointerAttributes attributes{};
            const cudaError_t status = cudaPointerGetAttributes(&attributes, data);
            if (status != cudaSuccess)
            {
                cudaGetLastError(); // do not leak an invalid-pointer error to the next call
                return AMGX_RC_BAD_PARAMETERS;
            }
            if (attributes.type != cudaMemoryTypeDevice || attributes.device != device)
                return AMGX_RC_BAD_PARAMETERS;
        }
        const AMGX_RC producer_status = synchronize_borrowed_producer(producer_stream, device);
        if (producer_status != AMGX_RC_OK) return producer_status;
        v.attach_storage(static_cast<Value *>(data), static_cast<size_t>(n));
        v.set_block_dimx(1);
        v.set_block_dimy(1);
        v.unset_transformed();
        return AMGX_RC_OK;
    }
    if (!v.is_borrowed()) return AMGX_RC_BAD_PARAMETERS;
    if (operation == BorrowedVectorOperation::query)
    {
        if (!out_data || !out_size || !out_device) return AMGX_RC_BAD_PARAMETERS;
        *out_data = amgx::thrust::raw_pointer_cast(v.data());
        *out_size = v.bytes();
        *out_device = device;
        return AMGX_RC_OK;
    }
    if (operation == BorrowedVectorOperation::synchronize)
    {
        return synchronize_borrowed_producer(producer_stream, device);
    }
    if (cudaStreamSynchronize(cudaStreamLegacy) != cudaSuccess) return AMGX_RC_CUDA_FAILURE;
    v.detach_storage();
    return AMGX_RC_OK;
}

inline AMGX_RC dispatch_borrowed_vector(AMGX_vector_handle handle,
        BorrowedVectorOperation operation, int n = 0, void *data = nullptr,
        size_t capacity = 0, uintptr_t producer_stream = 0,
        void **out_data = nullptr, size_t *out_size = nullptr, int *out_device = nullptr)
{
    Resources *resources = nullptr;
    AMGX_CHECK_API_ERROR_NORSRC(getAMGXerror(getResourcesFromVectorHandle(handle, &resources)))
    AMGX_ERROR rc = AMGX_OK;
    AMGX_RC result = AMGX_RC_BAD_MODE;
    AMGX_TRIES()
    {
        switch (get_mode_from<AMGX_vector_handle>(handle))
        {
#define AMGX_CASE_LINE(CASE) case CASE: result = borrowed_vector_operation<CASE>( \
            handle, operation, n, data, capacity, producer_stream, out_data, out_size, out_device); break;
            AMGX_FORALL_BUILDS_DEVICE(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE
            default: return AMGX_RC_BAD_MODE;
        }
    }
    AMGX_CATCHES(rc)
    AMGX_CHECK_API_ERROR(rc, resources)
    return result;
}
