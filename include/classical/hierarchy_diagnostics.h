// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <matrix.h>
#include <error.h>
#include <cutil.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <string>
#include <type_traits>

namespace amgx { namespace classical {

// Diagnostic format v1: native-endian, unmodified arrays plus a JSON sidecar.
// No public C ABI change. The parent directory must already exist. One prefix
// belongs to one setup; fail instead of silently mixing two hierarchies.
template<class Vector>
void write_hierarchy_array(const std::string &path, const Vector &v, size_t count)
{
    std::ofstream out(path.c_str(), std::ios::binary | std::ios::trunc);
    if (count > v.size())
        FatalError("Hierarchy export array is shorter than its logical size", AMGX_ERR_INTERNAL);
    if (count) out.write(reinterpret_cast<const char *>(v.raw()), count * sizeof(*v.raw()));
    out.close();
    if (!out) FatalError(("Cannot write hierarchy array: " + path).c_str(), AMGX_ERR_IO);
}

template<class TConfig>
void export_hierarchy_matrix(const std::string &prefix, int level,
                             int source_level, const char *role, const Matrix<TConfig> &matrix)
{
    typedef typename TConfig::template setMemSpace<AMGX_host>::Type Host;
    typedef typename TConfig::MatPrec Value;
    if (!matrix.is_matrix_singleGPU() || !matrix.hasProps(CSR))
        FatalError("Hierarchy export requires a single-GPU CSR/BSR matrix", AMGX_ERR_NOT_IMPLEMENTED);
    if (!std::is_same<Value, double>::value && !std::is_same<Value, float>::value)
        FatalError("Hierarchy export currently supports real FP32/FP64 matrices", AMGX_ERR_NOT_IMPLEMENTED);
    const std::string base = prefix + ".L" + std::to_string(level) + "." + role;
    if (std::ifstream((base + ".json").c_str()).good())
        FatalError(("Hierarchy export already exists: " + base).c_str(), AMGX_ERR_IO);
    Matrix<Host> h = matrix; // completion and device-to-host copy, only when enabled
    const auto rows = h.get_num_rows(), cols = h.get_num_cols(), nnz = h.get_num_nz();
    write_hierarchy_array(base + ".indptr.bin", h.row_offsets, rows + 1);
    write_hierarchy_array(base + ".indices.bin", h.col_indices, nnz);
    write_hierarchy_array(base + ".values.bin", h.values, h.values.size());
    write_hierarchy_array(base + ".diag.bin", h.diag, h.diag.size());
    const uint16_t endian_probe = 1;
    const char *endian = *reinterpret_cast<const unsigned char *>(&endian_probe) ? "<" : ">";
    std::ofstream out((base + ".json").c_str(), std::ios::trunc);
    out << "{\n  \"version\": 1, \"level\": " << level
        << ", \"source_level\": " << source_level << ", \"role\": \"" << role
        << "\",\n  \"block_rows\": " << rows << ", \"block_cols\": " << cols
        << ", \"nnzb\": " << nnz << ", \"block_dimy\": " << h.get_block_dimy()
        << ", \"block_dimx\": " << h.get_block_dimx()
        << ",\n  \"block_order\": \"" << (matrix.getBlockFormat() == ROW_MAJOR ? "C" : "F")
        << "\", \"external_diagonal\": " << (h.hasProps(DIAG) ? "true" : "false")
        << ",\n  \"index_dtype\": \"" << endian << "i" << sizeof(typename Matrix<Host>::index_type)
        << "\", \"value_dtype\": \"" << endian << "f" << sizeof(Value)
        << "\", \"diag_count\": " << h.diag.size()
        << ", \"values_count\": " << h.values.size() << "\n}\n";
    out.close();
    if (!out) FatalError(("Cannot write hierarchy metadata: " + base).c_str(), AMGX_ERR_IO);
}

// Deliberately synchronous diagnostic timings; never use this replay's solve
// time as the normal baseline. Counts cover full multiply() actions; fused
// smoother work and direct factorization are held constant in the projection.
class HierarchyProductProbe
{
    cudaEvent_t begin_ = nullptr, end_ = nullptr;
    std::string path_, key_;
public:
    template<class TConfig>
    explicit HierarchyProductProbe(const Matrix<TConfig> &matrix)
    {
        const char *path = std::getenv("AMGX_CLASSICAL_PROFILE_PATH");
        if (!path || !*path || TConfig::memSpace != AMGX_device
                || !matrix.hasParameter("classical_hierarchy_operator")) return;
        if (!matrix.is_matrix_singleGPU())
            FatalError("Hierarchy profiling requires single GPU", AMGX_ERR_NOT_IMPLEMENTED);
        path_ = path;
        key_ = matrix.template getParameter<std::string>("classical_hierarchy_operator");
        cudaEventCreate(&begin_);
        cudaEventCreate(&end_);
        cudaDeviceSynchronize();
        cudaEventRecord(begin_, 0);
        cudaCheckError();
    }
    ~HierarchyProductProbe()
    {
        if (!begin_) return;
        cudaEventRecord(end_, 0);
        cudaEventSynchronize(end_);
        float ms = 0;
        cudaEventElapsedTime(&ms, begin_, end_);
        cudaEventDestroy(begin_);
        cudaEventDestroy(end_);
        cudaCheckError();
        std::ofstream out(path_.c_str(), std::ios::app);
        out << key_ << ',' << std::setprecision(9) << ms << '\n';
        out.close();
        if (!out) FatalError("Cannot write hierarchy product profile", AMGX_ERR_IO);
    }
};

} } // namespace amgx::classical
