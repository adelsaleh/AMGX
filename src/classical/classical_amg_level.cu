// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#define COARSE_CLA_CONSO 0

#include <classical/classical_amg_level.h>
#include <classical/block_graph.h>
#include <amg_level.h>

#include <basic_types.h>
#include <cutil.h>
#include <multiply.h>
#include <transpose.h>
#include <truncate.h>
#include <blas.h>
#include <util.h>
#include <thrust/logical.h>
#include <thrust/remove.h>
#include <thrust/adjacent_difference.h>
#include <thrust_wrapper.h>

#include <thrust/extrema.h> // for minmax_element

#include <algorithm>
#include <limits>
#include <string>
#include <assert.h>
#include <matrix_io.h>

#include <csr_multiply.h>

#include <thrust/logical.h>
#include <thrust/count.h>
#include <thrust/sort.h>

#include <distributed/glue.h>
namespace amgx
{

namespace classical
{

struct is_zero
{
    __host__ __device__
    bool operator()(const double &v)
    {
        return fabs(v) < 1e-10;
    }
};

inline int block_spmv_backend_id(const std::string &backend)
{
    return backend == "cusparse_generic" ? 1
           : backend == "custom_5x5" ? 2 : 0;
}

template <class TConfig>
void set_block_spmv_backend(Matrix<TConfig> &matrix, int backend)
{
    matrix.setParameter("bsr_spmv_backend", backend);
    matrix.setParameter("use_subgroup_5x5_spmv", int(backend == 2));
}

#define AMGX_CAL_BLOCK_SIZE 256

template <typename IndexType>
__global__ void expand_bsr_row_offsets_kernel(const IndexType *block_row_offsets,
        IndexType num_block_rows, int block_dim, IndexType *scalar_row_offsets,
        IndexType *scalar_sequence)
{
    const IndexType num_scalar_rows = num_block_rows * block_dim;

    for (IndexType scalar_row = threadIdx.x + blockIdx.x * blockDim.x;
         scalar_row <= num_scalar_rows;
         scalar_row += blockDim.x * gridDim.x)
    {
        scalar_sequence[scalar_row] = scalar_row;

        if (scalar_row == num_scalar_rows)
        {
            scalar_row_offsets[scalar_row] = block_row_offsets[num_block_rows] * block_dim * block_dim;
        }
        else
        {
            const IndexType block_row = scalar_row / block_dim;
            const IndexType local_row = scalar_row % block_dim;
            const IndexType row_blocks = block_row_offsets[block_row + 1] - block_row_offsets[block_row];
            scalar_row_offsets[scalar_row] = block_row_offsets[block_row] * block_dim * block_dim
                                             + local_row * row_blocks * block_dim;
        }
    }
}

template <typename IndexType, typename ValueType>
__global__ void expand_bsr_values_kernel(const IndexType *block_row_offsets,
        const IndexType *block_col_indices, const ValueType *block_values,
        IndexType num_block_rows, int block_dim, bool row_major,
        const IndexType *scalar_row_offsets,
        IndexType *scalar_col_indices, ValueType *scalar_values)
{
    const IndexType num_scalar_rows = num_block_rows * block_dim;

    for (IndexType scalar_row = threadIdx.x + blockIdx.x * blockDim.x;
         scalar_row < num_scalar_rows;
         scalar_row += blockDim.x * gridDim.x)
    {
        const IndexType block_row = scalar_row / block_dim;
        const IndexType local_row = scalar_row % block_dim;
        const IndexType block_begin = block_row_offsets[block_row];
        const IndexType block_end = block_row_offsets[block_row + 1];
        IndexType output = scalar_row_offsets[scalar_row];

        for (IndexType block = block_begin; block < block_end; ++block)
        {
            const IndexType scalar_col_begin = block_col_indices[block] * block_dim;
            const IndexType value_begin = block * block_dim * block_dim;

            for (int local_col = 0; local_col < block_dim; ++local_col, ++output)
            {
                scalar_col_indices[output] = scalar_col_begin + local_col;
                const IndexType local_value = row_major
                                              ? local_row * block_dim + local_col
                                              : local_col * block_dim + local_row;
                scalar_values[output] = block_values[value_begin + local_value];
            }
        }
    }
}

template <class TConfig>
void expand_bsr_to_scalar_matrix(const Matrix<TConfig> &A,
                                 Matrix<TConfig> &scalar_A)
{
    typedef typename Matrix<TConfig>::index_type IndexType;
    const IndexType block_dim = A.get_block_dimx();
    const int64_t scalar_rows_64 =
        static_cast<int64_t>(A.get_num_rows()) * block_dim;
    const int64_t scalar_cols_64 =
        static_cast<int64_t>(A.get_num_cols()) * block_dim;
    const int64_t scalar_nnz_64 =
        static_cast<int64_t>(A.get_num_nz()) * block_dim * block_dim;

    if (scalar_rows_64 > std::numeric_limits<IndexType>::max()
            || scalar_cols_64 > std::numeric_limits<IndexType>::max()
            || scalar_nnz_64 > std::numeric_limits<IndexType>::max())
    {
        FatalError("Classical BSR scalar expansion exceeds the configured index precision",
                   AMGX_ERR_BAD_PARAMETERS);
    }

    const IndexType scalar_rows = static_cast<IndexType>(scalar_rows_64);
    const IndexType scalar_cols = static_cast<IndexType>(scalar_cols_64);
    const IndexType scalar_nnz = static_cast<IndexType>(scalar_nnz_64);
    scalar_A.set_initialized(0);
    scalar_A.addProps(CSR);
    scalar_A.setResources(A.getResources());
    // Allocate explicitly instead of Matrix::resize(): large device-side
    // thrust::sequence calls in resize are not reliable with CUDA 13 on
    // pre-Ampere GPUs. The kernels initialize both CSR metadata arrays.
    scalar_A.set_num_rows(scalar_rows);
    scalar_A.set_num_cols(scalar_cols);
    scalar_A.set_num_nz(scalar_nnz);
    scalar_A.set_block_dimx(1);
    scalar_A.set_block_dimy(1);
    scalar_A.row_offsets.resize(scalar_rows + 1);
    scalar_A.col_indices.resize(scalar_nnz);
    scalar_A.values.resize(scalar_nnz + 1);
    scalar_A.diag.resize(scalar_rows);
    scalar_A.m_diag_end_offsets.resize(scalar_rows);
    scalar_A.m_seq_offsets.resize(scalar_rows + 1);

    if (scalar_rows > 0)
    {
        const int threads = 256;
        const int blocks = std::min(
            AMGX_GRID_MAX_SIZE,
            static_cast<int>((scalar_rows_64 + threads - 1) / threads));
        expand_bsr_row_offsets_kernel<<<blocks, threads>>>(
            A.row_offsets.raw(), A.get_num_rows(), block_dim,
            scalar_A.row_offsets.raw(), scalar_A.m_seq_offsets.raw());
        cudaCheckError();
        expand_bsr_values_kernel<<<blocks, threads>>>(
            A.row_offsets.raw(), A.col_indices.raw(), A.values.raw(),
            A.get_num_rows(), block_dim, A.getBlockFormat() == ROW_MAJOR,
            scalar_A.row_offsets.raw(), scalar_A.col_indices.raw(),
            scalar_A.values.raw());
        cudaCheckError();
    }

    scalar_A.computeDiagonal();
    scalar_A.set_initialized(1);
}

__global__ void collapse_scalar_cf_to_block_any_kernel(
    const int *scalar_cf_map, int num_block_rows, int block_dim,
    int *block_cf_map)
{
    for (int block_row = threadIdx.x + blockIdx.x * blockDim.x;
         block_row < num_block_rows;
         block_row += blockDim.x * gridDim.x)
    {
        bool any_coarse = false;
        const int scalar_begin = block_row * block_dim;

        for (int component = 0; component < block_dim; ++component)
        {
            any_coarse = any_coarse
                         || scalar_cf_map[scalar_begin + component] == COARSE;
        }

        block_cf_map[block_row] = any_coarse ? COARSE : FINE;
    }
}

/* There might be a situation where not all local_to_global_map columns are present in the matrix (because some rows were removed
   and the columns in these rows are therefore no longer present. This kernel creates the flags array that marks existing columns. */
template<typename ind_t>
__global__ __launch_bounds__( AMGX_CAL_BLOCK_SIZE )
void flag_existing_local_to_global_columns(ind_t n, ind_t *row_offsets, ind_t *col_indices, ind_t *flags)
{
    ind_t i, j, s, e, col;

    //go through the matrix
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x)
    {
        s = row_offsets[i];
        e = row_offsets[i + 1];

        for (j = s; j < e; j++)
        {
            col = col_indices[j];

            //flag columns outside of the square part (which correspond to local_to_global_map)
            if (col >= n)
            {
                flags[col - n] = 1;
            }
        }
    }
}

/* Renumber the indices based on the prefix-scan/sum of the flags array */
template<typename ind_t>
__global__ __launch_bounds__( AMGX_CAL_BLOCK_SIZE )
void compress_existing_local_columns(ind_t n, ind_t *row_offsets, ind_t *col_indices, ind_t *flags)
{
    ind_t i, j, s, e, col;

    //go through the matrix
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x)
    {
        s = row_offsets[i];
        e = row_offsets[i + 1];

        for (j = s; j < e; j++)
        {
            col = col_indices[j];

            //flag columns outside of the square part (which correspond to local_to_global_map)
            if (col >= n)
            {
                col_indices[j] = n + flags[col - n];
            }
        }
    }
}

/* compress the local to global columns indices based on the prefix-scan/sum of the flags array */
template<typename ind_t, typename ind64_t>
__global__ __launch_bounds__( AMGX_CAL_BLOCK_SIZE )
void compress_existing_local_to_global_columns(ind_t n, ind64_t *l2g_in, ind64_t *l2g_out, ind_t *flags)
{
    ind_t i;

    //go through the arrays (and copy the updated indices when needed)
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x)
    {
        if (flags[i] != flags[i + 1])
        {
            l2g_out[flags[i]] = l2g_in[i];
        }
    }
}


template <class T_Config>
Selector<T_Config> *chooseAggressiveSelector(AMG_Config *m_cfg, std::string std_scope)
{
    AMG_Config cfg;
    std::string cfg_string("");
    cfg_string += "default:";
    // if necessary, allocate aggressive selector + interpolator
    bool use_pmis = false, use_hmis = false;
    // default argument - use the same selector as normal coarsening
    std::string agg_selector = m_cfg->AMG_Config::template getParameter<std::string>("aggressive_selector", std_scope);

    if (agg_selector == "DEFAULT")
    {
        std::string std_selector = m_cfg->AMG_Config::template getParameter<std::string>("selector", std_scope);

        if      (std_selector == "PMIS") { cfg_string += "selector=AGGRESSIVE_PMIS"; use_pmis = true; }
        else if (std_selector == "HMIS") { cfg_string += "selector=AGGRESSIVE_HMIS"; use_hmis = true; }
        else
        {
            FatalError("Must use either PMIS or HMIS algorithms with aggressive coarsening", AMGX_ERR_NOT_IMPLEMENTED);
        }
    }
    // otherwise use specified selector
    else if (agg_selector == "PMIS") { cfg_string += "selector=AGGRESSIVE_PMIS"; use_pmis = true; }
    else if (agg_selector == "HMIS") { cfg_string += "selector=AGGRESSIVE_HMIS"; use_hmis = true; }
    else
    {
        FatalError("Invalid aggressive coarsener selected", AMGX_ERR_NOT_IMPLEMENTED);
    }

    // check a selector has been selected
    if (!use_pmis && !use_hmis)
    {
        FatalError("No aggressive selector chosen", AMGX_ERR_NOT_IMPLEMENTED);
    }

    cfg.parseParameterString(cfg_string.c_str());
    // now allocate the selector and interpolator
    return classical::SelectorFactory<T_Config>::allocate(cfg, "default" /*std_scope*/);
}

template <class T_Config>
Interpolator<T_Config> *chooseAggressiveInterpolator(AMG_Config *m_cfg, std::string std_scope)
{
    // temporary config and pointer to main config
    AMG_Config cfg(*m_cfg);
    std::string cfg_string("default:");
    // Set the interpolator
    cfg_string += "interpolator=";
    cfg_string += m_cfg->AMG_Config::template getParameter<std::string>("aggressive_interpolator", std_scope);
    cfg.parseParameterString(cfg_string.c_str());
    // now allocate the selector and interpolator
    return InterpolatorFactory<T_Config>::allocate(cfg, "default");
}

template <class T_Config>
Classical_AMG_Level_Base<T_Config>::Classical_AMG_Level_Base(AMG_Class *amg) : AMG_Level<T_Config>(amg)
{
    m_coarsening_A = NULL;
    m_block_graph_P = NULL;
    m_block_graph_R = NULL;
    const std::string bsr_mode = amg->m_cfg->AMG_Config::template getParameter<std::string>(
                                     "classical_bsr_hierarchy", amg->m_cfg_scope);

    if (bsr_mode != "scalar_expand"
            && bsr_mode != "block_graph_identity"
            && bsr_mode != "block_graph_dense")
    {
        FatalError(
            "classical_bsr_hierarchy must be scalar_expand, "
            "block_graph_identity, or block_graph_dense",
            AMGX_ERR_BAD_PARAMETERS);
    }

    if (bsr_mode == "block_graph_identity"
            || bsr_mode == "block_graph_dense")
    {
        const std::string coarse_selector =
            amg->m_cfg->AMG_Config::template getParameter<std::string>(
                "block_graph_coarse_selector", amg->m_cfg_scope);

        if (coarse_selector != "block_graph"
                && coarse_selector != "scalar_guided_any")
        {
            FatalError("Invalid block_graph_coarse_selector",
                       AMGX_ERR_BAD_PARAMETERS);
        }

        const std::string strength_metric =
            amg->m_cfg->AMG_Config::template getParameter<std::string>(
                "block_graph_strength_metric", amg->m_cfg_scope);

        if (strength_metric != "frobenius"
                && strength_metric != "diagonal_normalized_frobenius"
                && strength_metric != "symmetric_inverse_diagonal_frobenius")
        {
            FatalError("Invalid block_graph_strength_metric",
                       AMGX_ERR_BAD_PARAMETERS);
        }
    }

    if (bsr_mode == "block_graph_dense")
    {
        const std::string interpolation_mode =
            amg->m_cfg->AMG_Config::template getParameter<std::string>(
                "block_graph_dense_interpolation_mode", amg->m_cfg_scope);
        const double smoothing_weight =
            amg->m_cfg->AMG_Config::template getParameter<double>(
                "block_graph_dense_smoothing_weight", amg->m_cfg_scope);
        const int smoothing_steps =
            amg->m_cfg->AMG_Config::template getParameter<int>(
                "block_graph_dense_smoothing_steps", amg->m_cfg_scope);
        const std::string constraint_mode =
            amg->m_cfg->AMG_Config::template getParameter<std::string>(
                "block_graph_dense_constraint_mode", amg->m_cfg_scope);
        const double pivot_tolerance =
            amg->m_cfg->AMG_Config::template getParameter<double>(
                "block_graph_dense_pivot_tolerance", amg->m_cfg_scope);
        const double constraint_tolerance =
            amg->m_cfg->AMG_Config::template getParameter<double>(
                "block_graph_dense_constraint_tolerance", amg->m_cfg_scope);
        const int structure_reuse_levels =
            amg->m_cfg->AMG_Config::template getParameter<int>(
                "structure_reuse_levels", amg->m_cfg_scope);
        const int aggressive_levels =
            amg->m_cfg->AMG_Config::template getParameter<int>(
                "aggressive_levels", amg->m_cfg_scope);

        if (structure_reuse_levels != 0)
        {
            FatalError(
                "block_graph_dense requires structure_reuse_levels=0",
                AMGX_ERR_NOT_IMPLEMENTED);
        }

        if (aggressive_levels != 0)
        {
            FatalError(
                "block_graph_dense currently requires aggressive_levels=0; "
                "aggressive D2 can leave fine block rows without interpolation support",
                AMGX_ERR_NOT_IMPLEMENTED);
        }

        if ((interpolation_mode != "jacobi"
                    && interpolation_mode != "extended_i")
                || !(smoothing_weight > 0.0) || !(smoothing_weight <= 2.0)
                || smoothing_steps < 1 || smoothing_steps > 8
                || (constraint_mode != "additive"
                    && constraint_mode != "right_normalize")
                || !(pivot_tolerance > 0.0)
                || !(constraint_tolerance > 0.0))
        {
            FatalError("Invalid block_graph_dense interpolation parameters",
                       AMGX_ERR_BAD_PARAMETERS);
        }
    }

    strength = StrengthFactory<T_Config>::allocate(*(amg->m_cfg), amg->m_cfg_scope);
    selector = classical::SelectorFactory<T_Config>::allocate(*(amg->m_cfg), amg->m_cfg_scope);
    interpolator = InterpolatorFactory<T_Config>::allocate(*(amg->m_cfg), amg->m_cfg_scope);
    trunc_factor = amg->m_cfg->AMG_Config::template getParameter<double>("interp_truncation_factor", amg->m_cfg_scope);
    max_elmts = amg->m_cfg->AMG_Config::template getParameter<int>("interp_max_elements", amg->m_cfg_scope);
    max_row_sum = amg->m_cfg->AMG_Config::template getParameter<double>("max_row_sum", amg->m_cfg_scope);
    num_aggressive_levels = amg->m_cfg->AMG_Config::template getParameter<int>("aggressive_levels", amg->m_cfg_scope);
}

template <class T_Config>
Classical_AMG_Level_Base<T_Config>::~Classical_AMG_Level_Base()
{
    releaseCoarseningMatrix();
    releaseBlockGraphTransfers();
    delete strength;
    delete selector;
    delete interpolator;
}
template <class T_Config>
bool Classical_AMG_Level_Base<T_Config>::usesBlockGraphHierarchy() const
{
    return (usesIdentityBlockGraphHierarchy()
            || usesDenseBlockGraphHierarchy())
           && this->A->get_block_size() > 1;
}

template <class T_Config>
bool Classical_AMG_Level_Base<T_Config>::usesIdentityBlockGraphHierarchy() const
{
    const std::string mode =
        this->amg->m_cfg->AMG_Config::template getParameter<std::string>(
            "classical_bsr_hierarchy", this->amg->m_cfg_scope);
    return mode == "block_graph_identity";
}

template <class T_Config>
bool Classical_AMG_Level_Base<T_Config>::usesDenseBlockGraphHierarchy() const
{
    const std::string mode =
        this->amg->m_cfg->AMG_Config::template getParameter<std::string>(
            "classical_bsr_hierarchy", this->amg->m_cfg_scope);
    return mode == "block_graph_dense";
}

template <class T_Config>
Matrix<T_Config> &Classical_AMG_Level_Base<T_Config>::getCoarseningMatrix()
{
    return m_coarsening_A == NULL ? this->getA() : *m_coarsening_A;
}

template <class T_Config>
void Classical_AMG_Level_Base<T_Config>::releaseCoarseningMatrix()
{
    delete m_coarsening_A;
    m_coarsening_A = NULL;
}

template <class T_Config>
void Classical_AMG_Level_Base<T_Config>::releaseBlockGraphTransfers()
{
    delete m_block_graph_P;
    delete m_block_graph_R;
    m_block_graph_P = NULL;
    m_block_graph_R = NULL;
}

template <class T_Config>
void Classical_AMG_Level_Base<T_Config>::transfer_level(AMG_Level<TConfig1> *ref_lvl)
{
    Classical_AMG_Level_Base<TConfig1> *ref_cla_lvl = dynamic_cast<Classical_AMG_Level_Base<TConfig1>*>(ref_lvl);
    this->P.copy(ref_cla_lvl->P);
    this->R.copy(ref_cla_lvl->R);
    this->m_s_con.copy(ref_cla_lvl->m_s_con);
    this->m_scratch.copy(ref_cla_lvl->m_scratch);
    this->m_cf_map.copy(ref_cla_lvl->m_cf_map);
}

/****************************************
 * Computes the A, P, and R operators
 ***************************************/
template <class T_Config>
void Classical_AMG_Level_Base<T_Config>::createCoarseVertices()
{
    prepareCoarseningMatrix();
    if (AMG_Level<T_Config>::getLevelIndex() < this->num_aggressive_levels)
    {
        if (selector) { delete selector; }

        selector = chooseAggressiveSelector<T_Config>(AMG_Level<T_Config>::amg->m_cfg, AMG_Level<T_Config>::amg->m_cfg_scope);
    }

    Matrix<T_Config> &RAP = this->getNextLevel( typename Matrix<T_Config>::memory_space( ) )->getA( );
    Matrix<T_Config> &A = getCoarseningMatrix();
    int size_all, size_full, nnz_full;

    if (!A.is_matrix_singleGPU())
    {
        int offset;
        // Need to get number of 2-ring rows
        A.getOffsetAndSizeForView(ALL, &offset, &size_all);
        A.getOffsetAndSizeForView(FULL, &offset, &size_full);
        A.getNnzForView(FULL, &nnz_full);
    }
    else
    {
        size_all = A.get_num_rows();
        size_full = A.get_num_rows();
        nnz_full = A.get_num_nz();
    }

    this->m_cf_map.resize(size_all);
    this->m_s_con.resize(nnz_full);
    this->m_scratch.resize(size_full);
    thrust_wrapper::fill<T_Config::memSpace>(this->m_cf_map.begin(), this->m_cf_map.end(), 0);
    cudaCheckError();
    thrust_wrapper::fill<T_Config::memSpace>(this->m_s_con.begin(), this->m_s_con.end(), false);
    cudaCheckError();
    thrust_wrapper::fill<T_Config::memSpace>(this->m_scratch.begin(), this->m_scratch.end(), 0);
    cudaCheckError();
    markCoarseFinePoints();
    // The hierarchy driver can reject this coarse level after selection. Do
    // not retain a temporary scalar expansion or block graph in that case.
    releaseCoarseningMatrix();
}

template <class T_Config>
void Classical_AMG_Level_Base<T_Config>::createCoarseMatrices()
{
    // Recreate the temporary scalar operator for interpolation and RAP. This
    // also covers hierarchy-structure reuse, where createCoarseVertices is
    // intentionally skipped by the driver.
    prepareCoarseningMatrix();
    const bool block_graph_hierarchy = usesBlockGraphHierarchy();

    if (block_graph_hierarchy)
    {
        releaseBlockGraphTransfers();
        const bool aggressive = AMG_Level<T_Config>::getLevelIndex() < this->num_aggressive_levels;
        const std::string interpolation = this->amg->m_cfg->AMG_Config::template getParameter<std::string>(
                                              aggressive ? "aggressive_interpolator" : "interpolator",
                                              this->amg->m_cfg_scope);

        if (interpolation != "D2")
        {
            FatalError(
                "Block-graph classical BSR hierarchies require D2 "
                "interpolation on every level",
                AMGX_ERR_NOT_IMPLEMENTED);
        }
    }

    // allocate aggressive interpolator if needed
    if (AMG_Level<T_Config>::getLevelIndex() < this->num_aggressive_levels)
    {
        if (interpolator) { delete interpolator; }

        interpolator = chooseAggressiveInterpolator<T_Config>(AMG_Level<T_Config>::amg->m_cfg, AMG_Level<T_Config>::amg->m_cfg_scope);
    }

    Matrix<T_Config> &RAP = this->getNextLevel( typename Matrix<T_Config>::memory_space( ) )->getA( );
    Matrix<T_Config> &A = getCoarseningMatrix();
    /* WARNING: exit if D1 interpolator is selected in distributed setting */
    std::string s("");
    s += AMG_Level<T_Config>::amg->m_cfg->AMG_Config::template getParameter<std::string>("interpolator", AMG_Level<T_Config>::amg->m_cfg_scope);

    if (A.is_matrix_distributed() && (s.compare("D1") == 0))
    {
        FatalError("D1 interpolation is not supported in distributed settings", AMGX_ERR_NOT_IMPLEMENTED);
    }

    /* WARNING: do not recompute prolongation (P) and restriction (R) when you
                are reusing the level structure (structure_reuse_levels > 0) */
    if (this->isReuseLevel() == false)
    {
        computeProlongationOperator();
    }

    // Compute Restriction operator and coarse matrix Ac
    if (!this->A->is_matrix_distributed() || this->A->manager->get_num_partitions() == 1)
    {
        /* WARNING: see above warning. */
        if (this->isReuseLevel() == false)
        {
            computeRestrictionOperator();
        }
        else if (block_graph_hierarchy)
        {
            if (usesDenseBlockGraphHierarchy())
            {
                FatalError(
                    "block_graph_dense does not yet support hierarchy "
                    "structure reuse",
                    AMGX_ERR_NOT_IMPLEMENTED);
            }

            m_block_graph_P = new Matrix<TConfig>();
            m_block_graph_R = new Matrix<TConfig>();
            Block_Graph_Ops<TConfig>::extract_scalar_transfer(
                P, *m_block_graph_P);
            Block_Graph_Ops<TConfig>::extract_scalar_transfer(
                R, *m_block_graph_R);
        }

        computeAOperator();
    }
    else
    {
        /* WARNING: notice that in this case the computeRestructionOperator() is called
                    inside computeAOperator_distributed() routine. */
        computeAOperator_distributed();
    }

    // The weighted block Galerkin kernel consumes the scalar P/R shadows
    // asynchronously.  Complete this level's setup before constructing later
    // levels that may reuse the device pool.
    if (block_graph_hierarchy)
    {
        cudaDeviceSynchronize();
        cudaCheckError();
    }

    // Keep the small scalar transfer shadows alive with a pure-BSR level.
    // Retained V-cycle operators are still the lifted BSR P/R matrices; the
    // shadows are only setup data, but releasing them here invalidates storage
    // still referenced by the lifted transfer structure in the current AMGX
    // device-vector ownership path.  The level destructor releases them.
    if (!block_graph_hierarchy)
    {
        releaseBlockGraphTransfers();
    }

// we also need to renumber columns of P and rows or R correspondingly since we changed RAP halo columns
// for R we just keep track of renumbering in and exchange proper vectors in restriction
// for P we actually need to renumber columns for prolongation:
    if (A.is_matrix_distributed() && this->A->manager->get_num_partitions() > 1)
    {
        RAP.set_initialized(0);
        // Renumber the owned nodes as interior and boundary (renumber rows and columns)
        // We are passing reuse flag to not create neighbours list from scratch, but rather update based on new halos
        RAP.manager->renumberMatrixOneRing(this->isReuseLevel());
        // Renumber the column indices of P and shuffle rows of P
        RAP.manager->renumber_P_R(this->P, this->R, A);
        // Create the B2L_maps for RAP
        RAP.manager->createOneRingHaloRows();
        RAP.manager->getComms()->set_neighbors(RAP.manager->num_neighbors());
        RAP.setView(OWNED);
        RAP.set_initialized(1);
        // update # of columns in P - this is necessary for correct CSR multiply
        P.set_initialized(0);
        int new_num_cols = thrust_wrapper::reduce<TConfig::memSpace>(P.col_indices.begin(), P.col_indices.end(), int(0), amgx::thrust::maximum<int>()) + 1;
        cudaCheckError();
        P.set_num_cols(new_num_cols);
        P.set_initialized(1);
    }

    RAP.copyAuxData(block_graph_hierarchy ? &this->getA() : &A);

    if (!A.is_matrix_singleGPU() && RAP.manager == NULL)
    {
        RAP.manager = new DistributedManager<TConfig>();
    }

    if (this->getA().is_matrix_singleGPU())
    {
        this->m_next_level_size = this->getNextLevel(typename Matrix<TConfig>::memory_space() )->getA().get_num_rows() * this->getNextLevel(typename Matrix<TConfig>::memory_space() )->getA().get_block_dimy();
    }
    else
    {
        // m_next_level_size is the size that will be used to allocate xc, bc vectors
        int size, offset;
        this->getNextLevel(typename Matrix<TConfig>::memory_space())->getA().getOffsetAndSizeForView(FULL, &offset, &size);
        this->m_next_level_size = size * this->getNextLevel(typename Matrix<TConfig>::memory_space() )->getA().get_block_dimy();
    }
    releaseCoarseningMatrix();
}

template <class T_Config>
void Classical_AMG_Level_Base<T_Config>::markCoarseFinePoints()
{
    if (usesBlockGraphHierarchy())
    {
        const std::string coarse_selector =
            this->amg->m_cfg->AMG_Config::template getParameter<std::string>(
                "block_graph_coarse_selector", this->amg->m_cfg_scope);

        if (coarse_selector == "scalar_guided_any")
        {
            markScalarGuidedCoarseFinePoints();
            return;
        }
    }

    Matrix<T_Config> &A = getCoarseningMatrix();
    //allocate necessary memory
    typedef Vector<typename TConfig::template setVecPrec<AMGX_vecInt>::Type> IVector;
    typedef Vector<typename TConfig::template setVecPrec<AMGX_vecBool>::Type> BVector;
    typedef Vector<typename TConfig::template setVecPrec<AMGX_vecFloat>::Type> FVector;
    FVector weights;

    if (!A.is_matrix_singleGPU())
    {
        int size, offset;
        A.getOffsetAndSizeForView(FULL, &offset, &size);
        // size should now contain the number of 1-ring rows
        weights.resize(size);
    }
    else
    {
        weights.resize(A.get_num_rows());
    }

    thrust_wrapper::fill<TConfig::memSpace>(weights.begin(), weights.end(), 0.0);
    cudaCheckError();

    // extend A to include 1st ring nodes
    // compute strong connections and weights
    if (!A.is_matrix_singleGPU())
    {
        ViewType oldView = A.currentView();
        A.setView(FULL);
        strength->computeStrongConnectionsAndWeights(A, this->m_s_con, weights, this->max_row_sum);
        A.setView(oldView);
    }
    else
    {
        strength->computeStrongConnectionsAndWeights(A, this->m_s_con, weights, this->max_row_sum);
    }

    // Exchange the one-ring of the weights
    if (!A.is_matrix_singleGPU())
    {
        A.manager->exchange_halo(weights, weights.tag);
    }

    //mark coarse and fine points
    selector->markCoarseFinePoints(A, weights, this->m_s_con, this->m_cf_map, this->m_scratch);
    // we do resize cf_map to zero later, so we are saving separate copy
    this->m_cf_map.dirtybit = 1;

    // Do a two ring exchange of cf_map
    if (!A.is_matrix_singleGPU())
    {
        A.manager->exchange_halo_2ring(this->m_cf_map, m_cf_map.tag);
    }

    // Modify cf_map array such that coarse points are assigned a local index, while fine points entries are not touched
    selector->renumberAndCountCoarsePoints(this->m_cf_map, this->m_num_coarse_vertices, A.get_num_rows());
}


template <class T_Config>
void Classical_AMG_Level_Base<T_Config>::computeProlongationOperator()
{
    this->Profile.tic("computeP");
    Matrix<T_Config> &A = getCoarseningMatrix();
    const bool block_graph_hierarchy = usesBlockGraphHierarchy();
    Matrix<TConfig> *generated_P = &P;

    if (block_graph_hierarchy)
    {
        m_block_graph_P = new Matrix<TConfig>();
        generated_P = m_block_graph_P;
    }

    interpolator->generateInterpolationMatrix(A, this->m_cf_map, this->m_s_con,
                                               this->m_scratch, *generated_P);

    // Truncate the scalar D2 support before either identity lifting or dense
    // block smoothing. Both pure-BSR modes retain this fixed block graph.
    if (this->max_elmts > 0 && generated_P->get_num_rows() > 0)
    {
        Truncate<TConfig>::truncateByMaxElements(*generated_P, this->max_elmts);
    }

    if (block_graph_hierarchy)
    {
        Matrix<TConfig> &block_A = this->getA();
        const std::string backend =
            this->amg->m_cfg->AMG_Config::template getParameter<std::string>(
                "bsr_spmv_backend", this->amg->m_cfg_scope);
        const int backend_id = block_spmv_backend_id(backend);
        BlockFormat block_format = block_A.getBlockFormat();

        if (usesDenseBlockGraphHierarchy())
        {
            const std::string interpolation_mode =
                this->amg->m_cfg->AMG_Config::template getParameter<std::string>(
                    "block_graph_dense_interpolation_mode",
                    this->amg->m_cfg_scope);
            const double smoothing_weight =
                this->amg->m_cfg->AMG_Config::template getParameter<double>(
                    "block_graph_dense_smoothing_weight",
                    this->amg->m_cfg_scope);
            const int smoothing_steps =
                this->amg->m_cfg->AMG_Config::template getParameter<int>(
                    "block_graph_dense_smoothing_steps",
                    this->amg->m_cfg_scope);
            const std::string constraint_mode =
                this->amg->m_cfg->AMG_Config::template getParameter<std::string>(
                    "block_graph_dense_constraint_mode",
                    this->amg->m_cfg_scope);
            const double pivot_tolerance =
                this->amg->m_cfg->AMG_Config::template getParameter<double>(
                    "block_graph_dense_pivot_tolerance",
                    this->amg->m_cfg_scope);
            const double constraint_tolerance =
                this->amg->m_cfg->AMG_Config::template getParameter<double>(
                    "block_graph_dense_constraint_tolerance",
                    this->amg->m_cfg_scope);
            if (interpolation_mode == "extended_i")
            {
                Block_Graph_Ops<TConfig>::extended_i_dense_transfer(
                    block_A, *generated_P, this->m_cf_map, this->m_s_con,
                    pivot_tolerance, constraint_tolerance, P);
            }
            else
            {
                Block_Graph_Ops<TConfig>::smooth_dense_transfer(
                    block_A, *generated_P, smoothing_weight, smoothing_steps,
                    constraint_mode == "right_normalize", pivot_tolerance,
                    constraint_tolerance, P);
            }
        }
        else
        {
            Block_Graph_Ops<TConfig>::lift_scalar_transfer(
                *generated_P, block_A.get_block_dimx(), block_format, P);
        }

        set_block_spmv_backend(block_A, backend_id);
        set_block_spmv_backend(P, backend_id);
    }

    this->m_cf_map.clear();
    this->m_cf_map.shrink_to_fit();
    this->m_scratch.clear();
    this->m_scratch.shrink_to_fit();
    this->m_s_con.clear();
    this->m_s_con.shrink_to_fit();

    if (!P.isLatencyHidingEnabled(*this->amg->m_cfg))
    {
        // This will cause bsrmv_with_mask to not do latency hiding
        P.setInteriorView(OWNED);
        P.setExteriorView(OWNED);
    }
}

/**********************************************
 * computes R=P^T
 **********************************************/
template <class T_Config>
void Classical_AMG_Level_Base<T_Config>::computeRestrictionOperator()
{
    this->Profile.tic("computeR");
    const bool block_graph_hierarchy = usesBlockGraphHierarchy();
    Matrix<TConfig> *source_P = &P;
    Matrix<TConfig> *generated_R = &R;

    if (block_graph_hierarchy)
    {
        if (m_block_graph_P == NULL)
        {
            FatalError("Missing scalar block-graph prolongation", AMGX_ERR_INTERNAL);
        }

        m_block_graph_R = new Matrix<TConfig>();
        source_P = m_block_graph_P;
        generated_R = m_block_graph_R;
    }

    generated_R->set_initialized(0);
    source_P->setView(OWNED);
    transpose(*source_P, *generated_R, source_P->get_num_rows());
    generated_R->set_initialized(1);

    if (block_graph_hierarchy)
    {
        Matrix<TConfig> &block_A = this->getA();
        const std::string backend =
            this->amg->m_cfg->AMG_Config::template getParameter<std::string>(
                "bsr_spmv_backend", this->amg->m_cfg_scope);
        const int backend_id = block_spmv_backend_id(backend);
        BlockFormat block_format = block_A.getBlockFormat();

        if (usesDenseBlockGraphHierarchy())
        {
            Block_Graph_Ops<TConfig>::transpose_dense_transfer(
                *generated_R, P, R);
        }
        else
        {
            Block_Graph_Ops<TConfig>::lift_scalar_transfer(
                *generated_R, block_A.get_block_dimx(), block_format, R);
        }

        set_block_spmv_backend(block_A, backend_id);
        set_block_spmv_backend(R, backend_id);
    }

    if (!R.isLatencyHidingEnabled(*this->amg->m_cfg))
    {
        // This will cause bsrmv_with_mask_restriction to not do latency hiding
        R.setInteriorView(OWNED);
        R.setExteriorView(OWNED);
    }

    if(P.is_matrix_distributed())
    {
        // Setup the number of non-zeros in R using stub DistributedManager
        R.manager = new DistributedManager<T_Config>();
        int nrows_owned = P.manager->halo_offsets[0];
        int nrows_full = P.manager->halo_offsets[P.manager->neighbors.size()];
        int nz_full = R.row_offsets[nrows_full];
        int nz_owned = R.row_offsets[nrows_owned];
        R.manager->setViewSizes(nrows_owned, nz_owned, nrows_owned, nz_owned,
                                nrows_full, nz_full, R.get_num_rows(),
                                R.get_num_nz());
    }

    R.set_initialized(1);
    this->Profile.toc("computeR");
}

/**********************************************
 * computes the Galerkin product: A_c=R*A*P
 **********************************************/

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Classical_AMG_Level<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::prepareCoarseningMatrix()
{
    Matrix<TConfig_h> &A = this->getA();

    if (A.get_block_size() == 1)
    {
        return;
    }

    if (this->usesBlockGraphHierarchy())
    {
        FatalError("Classical block-graph BSR hierarchy is device-only",
                   AMGX_ERR_NOT_IMPLEMENTED);
    }

    if (!A.is_matrix_singleGPU())
    {
        FatalError("Classical BSR setup is currently single-GPU only", AMGX_ERR_NOT_IMPLEMENTED);
    }

    this->releaseCoarseningMatrix();
    this->m_coarsening_A = new Matrix<TConfig_h>();
    const unsigned int props = CSR | (A.hasProps(DIAG) ? DIAG : 0);
    this->m_coarsening_A->convert(A, props, 1, 1);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Classical_AMG_Level<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::prepareCoarseningMatrix()
{
    Matrix<TConfig_d> &A = this->getA();

    if (A.get_block_size() == 1)
    {
        return;
    }

    if (!A.is_matrix_singleGPU())
    {
        FatalError("Classical BSR setup is currently single-GPU only", AMGX_ERR_NOT_IMPLEMENTED);
    }

    if (A.get_block_dimx() != A.get_block_dimy())
    {
        FatalError("Classical BSR setup requires square blocks", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }

    if (A.hasProps(DIAG))
    {
        FatalError("Classical BSR setup currently requires an internal diagonal",
                   AMGX_ERR_NOT_IMPLEMENTED);
    }

    if (this->usesBlockGraphHierarchy())
    {
        this->releaseCoarseningMatrix();
        this->m_coarsening_A = new Matrix<TConfig_d>();
        const std::string strength_metric =
            this->amg->m_cfg->AMG_Config::template getParameter<std::string>(
                "block_graph_strength_metric", this->amg->m_cfg_scope);
        const int strength_metric_id =
            strength_metric == "diagonal_normalized_frobenius" ? 1
            : strength_metric == "symmetric_inverse_diagonal_frobenius" ? 2
            : 0;
        Block_Graph_Ops<TConfig_d>::build_graph(
            A, *this->m_coarsening_A, strength_metric_id);
        return;
    }

    this->releaseCoarseningMatrix();
    this->m_coarsening_A = new Matrix<TConfig_d>();
    expand_bsr_to_scalar_matrix(A, *this->m_coarsening_A);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Classical_AMG_Level<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::computeAOperator_1x1()
{
    Matrix<TConfig_h> &A = this->getCoarseningMatrix();
    this->Profile.tic("computeA");
    Matrix<TConfig_h> RA;
    RA.addProps(CSR);
    RA.set_block_dimx(A.get_block_dimx());
    RA.set_block_dimy(A.get_block_dimy());
    Matrix<TConfig_h> &RAP = this->getNextLevel( typename Matrix<TConfig_h>::memory_space( ) )->getA( );
    RAP.addProps(CSR);
    RAP.set_block_dimx(A.get_block_dimx());
    RAP.set_block_dimy(A.get_block_dimy());
    multiplyMM(this->R, A, RA);
    multiplyMM(RA, this->P, RAP);
    RAP.sortByRowAndColumn();
    RAP.set_initialized(1);
    this->Profile.toc("computeA");
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Classical_AMG_Level<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::computeAOperator_block_graph()
{
    FatalError("Classical block-graph BSR hierarchy is device-only",
               AMGX_ERR_NOT_IMPLEMENTED);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Classical_AMG_Level<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::computeAOperator_1x1_distributed()
{
    FatalError("Distributed classical AMG not implemented for host\n", AMGX_ERR_NOT_IMPLEMENTED);
}


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Classical_AMG_Level<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::markScalarGuidedCoarseFinePoints()
{
    FatalError("Scalar-guided block coarsening is device-only",
               AMGX_ERR_NOT_IMPLEMENTED);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Classical_AMG_Level<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::markScalarGuidedCoarseFinePoints()
{
    typedef Vector<typename TConfig_d::template setVecPrec<AMGX_vecBool>::Type> BoolVector;
    typedef Vector<typename TConfig_d::template setVecPrec<AMGX_vecFloat>::Type> FloatVector;
    Matrix<TConfig_d> &block_A = this->getA();
    Matrix<TConfig_d> &graph_A = this->getCoarseningMatrix();
    Matrix<TConfig_d> scalar_A;
    expand_bsr_to_scalar_matrix(block_A, scalar_A);

    IVector scalar_cf_map(scalar_A.get_num_rows(), 0);
    BoolVector scalar_s_con(scalar_A.get_num_nz(), false);
    IVector scalar_scratch(scalar_A.get_num_rows(), 0);
    FloatVector scalar_weights(scalar_A.get_num_rows(), 0.0);
    this->strength->computeStrongConnectionsAndWeights(
        scalar_A, scalar_s_con, scalar_weights, this->max_row_sum);
    this->selector->markCoarseFinePoints(
        scalar_A, scalar_weights, scalar_s_con,
        scalar_cf_map, scalar_scratch);

    if (block_A.get_num_rows() > 0)
    {
        const int threads = 256;
        const int blocks = std::min(
            AMGX_GRID_MAX_SIZE,
            static_cast<int>((block_A.get_num_rows() + threads - 1) / threads));
        collapse_scalar_cf_to_block_any_kernel<<<blocks, threads>>>(
            scalar_cf_map.raw(), block_A.get_num_rows(),
            block_A.get_block_dimx(), this->m_cf_map.raw());
        cudaCheckError();
    }

    // D2 still operates on the scalar block graph, so rebuild its strong-edge
    // mask independently of the coefficient-exact scalar guidance graph.
    FloatVector block_weights(graph_A.get_num_rows(), 0.0);
    thrust_wrapper::fill<TConfig_d::memSpace>(
        this->m_s_con.begin(), this->m_s_con.end(), false);
    cudaCheckError();
    this->strength->computeStrongConnectionsAndWeights(
        graph_A, this->m_s_con, block_weights, this->max_row_sum);
    this->m_cf_map.dirtybit = 1;
    this->selector->renumberAndCountCoarsePoints(
        this->m_cf_map, this->m_num_coarse_vertices,
        graph_A.get_num_rows());
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Classical_AMG_Level<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::computeAOperator_1x1()
{
    Matrix<TConfig_d> &A = this->getCoarseningMatrix();
    this->Profile.tic("computeA");
    Matrix<TConfig_d> &RAP = this->getNextLevel( device_memory( ) )->getA( );
    RAP.addProps(CSR);
    RAP.set_block_dimx(A.get_block_dimx());
    RAP.set_block_dimy(A.get_block_dimy());
    this->R.set_initialized( 0 );
    this->R.addProps( CSR );
    this->R.set_initialized( 1 );
    this->P.set_initialized( 0 );
    this->P.addProps( CSR );
    this->P.set_initialized( 1 );
    void *wk = AMG_Level<TConfig_d>::amg->getCsrWorkspace();

    if ( wk == NULL )
    {
        wk = CSR_Multiply<TConfig_d>::csr_workspace_create( *(AMG_Level<TConfig_d>::amg->m_cfg), AMG_Level<TConfig_d>::amg->m_cfg_scope );
        AMG_Level<TConfig_d>::amg->setCsrWorkspace( wk );
    }

    int spmm_verbose = this->amg->m_cfg->AMG_Config::template getParameter<int>("spmm_verbose", this->amg->m_cfg_scope);

    if ( spmm_verbose )
    {
        typedef typename Matrix<TConfig_d>::IVector::const_iterator Iterator;
        typedef amgx::thrust::pair<Iterator, Iterator> Result;
        std::ostringstream buffer;
        buffer << "SPMM: Level " << this->getLevelIndex() << std::endl;

        if ( this->getLevelIndex() == 0 )
        {
            device_vector_alloc<int> num_nz( A.row_offsets.size() );
            amgx::thrust::adjacent_difference( A.row_offsets.begin(), A.row_offsets.end(), num_nz.begin() );
            cudaCheckError();
            Result result = amgx::thrust::minmax_element( num_nz.begin() + 1, num_nz.end() );
            cudaCheckError();
            int min_size = *result.first;
            int max_size = *result.second;
            int sum = thrust_wrapper::reduce<AMGX_device>( num_nz.begin() + 1, num_nz.end() );
            cudaCheckError();
            double avg_size = double(sum) / A.get_num_rows();
            buffer << "SPMM: A: " << std::endl;
            buffer << "SPMM: Matrix avg row size: " << avg_size << std::endl;
            buffer << "SPMM: Matrix min row size: " << min_size << std::endl;
            buffer << "SPMM: Matrix max row size: " << max_size << std::endl;
        }

        device_vector_alloc<int> num_nz( this->P.row_offsets.size() );
        amgx::thrust::adjacent_difference( this->P.row_offsets.begin(), this->P.row_offsets.end(), num_nz.begin() );
        cudaCheckError();
        Result result = amgx::thrust::minmax_element( num_nz.begin() + 1, num_nz.end() );
        cudaCheckError();
        int min_size = *result.first;
        int max_size = *result.second;
        int sum = thrust_wrapper::reduce<AMGX_device>( num_nz.begin() + 1, num_nz.end() );
        cudaCheckError();
        double avg_size = double(sum) / this->P.get_num_rows();
        buffer << "SPMM: P: " << std::endl;
        buffer << "SPMM: Matrix avg row size: " << avg_size << std::endl;
        buffer << "SPMM: Matrix min row size: " << min_size << std::endl;
        buffer << "SPMM: Matrix max row size: " << max_size << std::endl;
        num_nz.resize( this->R.row_offsets.size() );
        amgx::thrust::adjacent_difference( this->R.row_offsets.begin(), this->R.row_offsets.end(), num_nz.begin() );
        cudaCheckError();
        result = amgx::thrust::minmax_element( num_nz.begin() + 1, num_nz.end() );
        cudaCheckError();
        min_size = *result.first;
        max_size = *result.second;
        sum = thrust_wrapper::reduce<AMGX_device>( num_nz.begin() + 1, num_nz.end() );
        cudaCheckError();
        avg_size = double(sum) / this->R.get_num_rows();
        buffer << "SPMM: R: " << std::endl;
        buffer << "SPMM: Matrix avg row size: " << avg_size << std::endl;
        buffer << "SPMM: Matrix min row size: " << min_size << std::endl;
        buffer << "SPMM: Matrix max row size: " << max_size << std::endl;
        amgx_output( buffer.str().c_str(), static_cast<int>( buffer.str().length() ) );
    }

    RAP.set_initialized( 0 );
    CSR_Multiply<TConfig_d>::csr_galerkin_product( this->R, A, this->P, RAP, NULL, NULL, NULL, NULL, NULL, NULL, wk );
    RAP.set_initialized( 1 );
    int spmm_no_sort = this->amg->m_cfg->AMG_Config::template getParameter<int>("spmm_no_sort", this->amg->m_cfg_scope);
    this->Profile.toc("computeA");
}
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Classical_AMG_Level<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::computeAOperator_block_graph()
{
    if (this->m_block_graph_P == NULL || this->m_block_graph_R == NULL)
    {
        FatalError("Missing scalar block-graph transfer weights", AMGX_ERR_INTERNAL);
    }

    Matrix<TConfig_d> &A = this->getA();
    Matrix<TConfig_d> &graph_A = this->getCoarseningMatrix();
    Matrix<TConfig_d> &RAP = this->getNextLevel(device_memory())->getA();
    this->Profile.tic("computeA_block_graph");
    void *wk = AMG_Level<TConfig_d>::amg->getCsrWorkspace();

    if (wk == NULL)
    {
        wk = CSR_Multiply<TConfig_d>::csr_workspace_create(
                 *(AMG_Level<TConfig_d>::amg->m_cfg),
                 AMG_Level<TConfig_d>::amg->m_cfg_scope);
        AMG_Level<TConfig_d>::amg->setCsrWorkspace(wk);
    }

    if (this->usesDenseBlockGraphHierarchy())
    {
        Block_Graph_Ops<TConfig_d>::dense_galerkin(
            A, graph_A, *this->m_block_graph_R, *this->m_block_graph_P,
            this->R, this->P, RAP, wk);
    }
    else
    {
        Block_Graph_Ops<TConfig_d>::weighted_galerkin(
            A, graph_A, *this->m_block_graph_R, *this->m_block_graph_P,
            RAP, wk);
    }

    this->Profile.toc("computeA_block_graph");
}

/**********************************************
 * computes the restriction: rr=R*r
 **********************************************/
template <class T_Config>
void Classical_AMG_Level_Base<T_Config>::restrictResidual(VVector &r, VVector &rr)
{
// we need to resize residual vector to make sure it can store halo rows to be sent
    if (!P.is_matrix_singleGPU())
    {
        typedef typename TConfig::MemSpace MemorySpace;
        Matrix<TConfig> &Ac = this->getNextLevel( MemorySpace( ) )->getA();
#if COARSE_CLA_CONSO
        int desired_size ;

        if (this->getNextLevel(MemorySpace())->isConsolidationLevel())
        {
            desired_size = std::max(P.manager->halo_offsets[P.manager->neighbors.size()], Ac.manager->halo_offsets_before_glue[Ac.manager->neighbors_before_glue.size()] * rr.get_block_size());
        }
        else
        {
            desired_size = std::max(P.manager->halo_offsets[P.manager->neighbors.size()], Ac.manager->halo_offsets[Ac.manager->neighbors.size()] * rr.get_block_size());
        }

#else
        int desired_size = std::max(P.manager->halo_offsets[P.manager->neighbors.size()], Ac.manager->halo_offsets[Ac.manager->neighbors.size()] * rr.get_block_size());
#endif
        rr.resize(desired_size);
    }

#if 1
    // The hybrid hierarchy has scalar R over a contiguous fine BSR vector.
    // The block-graph hierarchy already has a b x b restriction and needs no
    // metadata reinterpretation.
    const short fine_r_block_dimy = r.get_block_dimy();
    const bool scalar_transfer_on_block_vector =
        R.get_block_size() == 1 && fine_r_block_dimy > 1;

    if (scalar_transfer_on_block_vector)
    {
        r.set_block_dimy(R.get_block_dimx());
    }

    this->Profile.tic("restrictRes");

    if (usesBlockGraphHierarchy() && P.is_matrix_singleGPU())
    {
        const size_t required_input = static_cast<size_t>(R.get_num_cols())
                                      * R.get_block_dimx();
        const size_t required_output = static_cast<size_t>(R.get_num_rows())
                                       * R.get_block_dimy();

        if (r.size() < required_input || rr.size() < required_output)
        {
            FatalError("Pure-BSR restriction vector size mismatch",
                       AMGX_ERR_INTERNAL);
        }
    }

    // Disable speculative send of rr
    if (P.is_matrix_singleGPU())
    {
        if (usesIdentityBlockGraphHierarchy())
        {
            Block_Graph_Ops<TConfig>::multiply_identity_transfer(R, r, rr);
        }
        else
        {
            multiply(R, r, rr);
        }
    }
    else
    {
        multiply_with_mask_restriction( R, r, rr, P);
    }

#endif
    if (scalar_transfer_on_block_vector)
    {
        r.set_block_dimy(fine_r_block_dimy);
    }

    // exchange halo residuals & add residual contribution from neighbors
    rr.dirtybit = 1;

    if (!P.is_matrix_singleGPU())
    {
        int desired_size = P.manager->halo_offsets[P.manager->neighbors.size()] * rr.get_block_size();

        if (rr.size() < desired_size)
        {
            rr.resize(desired_size);
        }
    }

    this->Profile.toc("restrictRes");
}

struct is_minus_one
{
    __host__ __device__
    bool operator()(const int &x)
    {
        return x == -1;
    }
};


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Classical_AMG_Level<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::computeAOperator_1x1_distributed()
{
    Matrix<TConfig_d> &A = this->getA();
    Matrix<TConfig_d> &P = this->P;
    Matrix<TConfig_d> &RAP = this->getNextLevel( device_memory( ) )->getA( );
    RAP.addProps(CSR);
    RAP.set_block_dimx(this->getA().get_block_dimx());
    RAP.set_block_dimy(this->getA().get_block_dimy());
    IndexType num_parts = A.manager->get_num_partitions();
    IndexType num_neighbors = A.manager->num_neighbors();
    IndexType my_rank = A.manager->global_id();
    // OWNED includes interior and boundary
    A.setView(OWNED);
    int num_owned_coarse_pts = P.manager->halo_offsets[0];
    int num_owned_fine_pts = A.manager->halo_offsets[0];

    // Initialize RAP.manager
    if (RAP.manager == NULL)
    {
        RAP.manager = new DistributedManager<TConfig_d>();
    }

    RAP.manager->A = &RAP;
    RAP.manager->setComms(A.manager->getComms());
    RAP.manager->set_global_id(my_rank);
    RAP.manager->set_num_partitions(num_parts);
    RAP.manager->part_offsets_h = P.manager->part_offsets_h;
    RAP.manager->part_offsets = P.manager->part_offsets;
    RAP.manager->set_base_index(RAP.manager->part_offsets_h[my_rank]);
    RAP.manager->set_index_range(num_owned_coarse_pts);
    RAP.manager->num_rows_global = RAP.manager->part_offsets_h[num_parts];
    // --------------------------------------------------------------------
    // Using the B2L_maps of matrix A, identify the rows of P that need to be sent to neighbors,
    // so that they can compute A*P
    // Once rows of P are identified, convert the column indices to global indices, and send them to neighbors
    //  ---------------------------------------------------------------------------
    // Copy some information about the manager of P, since we don't want to modify those
    IVector_h P_neighbors = P.manager->neighbors;
    I64Vector_h P_halo_ranges_h = P.manager->halo_ranges_h;
    I64Vector_d P_halo_ranges = P.manager->halo_ranges;
    RAP.manager->local_to_global_map = P.manager->local_to_global_map;
    IVector_h P_halo_offsets = P.manager->halo_offsets;
    // Create a temporary distributed arranger
    DistributedArranger<TConfig_d> *prep = new DistributedArranger<TConfig_d>;
    prep->exchange_halo_rows_P(A, this->P, RAP.manager->local_to_global_map, P_neighbors, P_halo_ranges_h, P_halo_ranges, P_halo_offsets, RAP.manager->part_offsets_h, RAP.manager->part_offsets, num_owned_coarse_pts, RAP.manager->part_offsets_h[my_rank]);
    cudaCheckError();
    // At this point, we can compute RAP_full which contains some rows that will need to be sent to neighbors
    // i.e. RAP_full = [ RAP_int ]
    //                 [ RAP_ext ]
    // RAP is [ RAP_int ] + [RAP_ext_received_from_neighbors]
    // We can reuse the serial galerkin product since R, A and P use local indices
    // TODO: latency hiding (i.e. compute RAP_ext, exchange_matrix_halo, then do RAP_int)
    /* WARNING: do not recompute prolongation (P) and restriction (R) when you
                are reusing the level structure (structure_reuse_levels > 0) */
    /* We force for matrix P to have only owned rows to be seen for the correct galerkin product computation*/
    this->P.set_initialized(0);
    this->P.set_num_rows(num_owned_fine_pts);
    this->P.addProps( CSR );
    this->P.set_initialized(1);

    if (this->isReuseLevel() == false)
    {
        this->R.set_initialized( 0 );
        this->R.addProps( CSR );
        // Take the tranpose of P to get R
        // Single-GPU transpose, no mpi exchange
        this->computeRestrictionOperator();
        this->R.set_initialized( 1 );
    }

    this->Profile.tic("computeA");
    Matrix<TConfig_d> RAP_full;
    // Initialize the workspace needed for galerkin product
    void *wk = AMG_Level<TConfig_d>::amg->getCsrWorkspace();

    if ( wk == NULL )
    {
        wk = CSR_Multiply<TConfig_d>::csr_workspace_create( *(AMG_Level<TConfig_d>::amg->m_cfg), AMG_Level<TConfig_d>::amg->m_cfg_scope );
        AMG_Level<TConfig_d>::amg->setCsrWorkspace( wk );
    }

    // Single-GPU RAP, no mpi exchange
    RAP_full.set_initialized( 0 );
    /* WARNING: Since A is reordered (into interior and boundary nodes), while R and P are not reordered,
                you must unreorder A when performing R*A*P product in ordre to obtain the correct result. */
    CSR_Multiply<TConfig_d>::csr_galerkin_product( this->R, this->getA(), this->P, RAP_full,
            /* permutation for rows of R, A and P */       NULL, NULL /*&(this->getA().manager->renumbering)*/,        NULL,
            /* permutation for cols of R, A and P */       NULL, NULL /*&(this->getA().manager->inverse_renumbering)*/, NULL,
            wk );
    RAP_full.set_initialized( 1 );
    this->Profile.toc("computeA");
    // ----------------------------------------------------------------------------------------------
    // Now, send rows of RAP_full requireq by neighbors, received rows from neighbors and create RAP
    // ----------------------------------------------------------------------------------------------
    prep->exchange_RAP_ext(RAP, RAP_full, A, this->P, P_halo_offsets, RAP.manager->local_to_global_map, P_neighbors, P_halo_ranges_h, P_halo_ranges, RAP.manager->part_offsets_h, RAP.manager->part_offsets, num_owned_coarse_pts, RAP.manager->part_offsets_h[my_rank], wk);
    // Delete temporary distributed arranger
    delete prep;
    /* WARNING: The RAP matrix generated at this point contains extra rows (that correspond to rows of R,
       that was obtained by locally transposing P). This rows are ignored by setting the # of matrix rows
       to be smaller, so that they correspond to number of owned coarse nodes. This should be fine, but
       it leaves holes in the matrix as there might be columns that belong to the extra rows that now do not
       belong to the smaller matrix with number of owned coarse nodes rows. The same is trued about the
       local_to_global_map. These two data structures match at this point. However, in the next calls
       local_to_global (exclusively) will be used to geberate B2L_maps (wihtout going through column indices)
       which creates extra elements in the B2L that simply do not exist in the new matrices. I strongly suspect
       this is the reason fore the bug. The below fix simply compresses the matrix so that there are no holes
       in it, or in the local_2_global_map. */
    //mark local_to_global_columns that exist in the owned coarse nodes rows.
    IndexType nrow = RAP.get_num_rows();
    IndexType ncol = RAP.get_num_cols();
    IndexType nl2g = ncol - nrow;

    if (nl2g > 0)
    {
        IVector   l2g_p(nl2g + 1, 0); //+1 is needed for prefix_sum/exclusive_scan
        I64Vector l2g_t(nl2g, 0);
        IndexType nblocks = (nrow + AMGX_CAL_BLOCK_SIZE - 1) / AMGX_CAL_BLOCK_SIZE;

        if (nblocks > 0)
            flag_existing_local_to_global_columns<int> <<< nblocks, AMGX_CAL_BLOCK_SIZE>>>
            (nrow, RAP.row_offsets.raw(), RAP.col_indices.raw(), l2g_p.raw());

        cudaCheckError();
        /*
        //slow version of the above kernel
        for(int ii=0; ii<nrow; ii++){
            int s = RAP.row_offsets[ii];
            int e = RAP.row_offsets[ii+1];
            for (int jj=s; jj<e; jj++) {
                int col = RAP.col_indices[jj];
                if (col>=nrow){
                    int kk = col-RAP.get_num_rows();
                    l2g_p[kk] = 1;
                }
            }
        }
        cudaCheckError();
        */
        //create a pointer map for their location using prefix sum
        thrust_wrapper::exclusive_scan<AMGX_device>(l2g_p.begin(), l2g_p.end(), l2g_p.begin());
        int new_nl2g = l2g_p[nl2g];

        //compress the columns using the pointer map
        if (nblocks > 0)
            compress_existing_local_columns<int> <<< nblocks, AMGX_CAL_BLOCK_SIZE>>>
            (nrow, RAP.row_offsets.raw(), RAP.col_indices.raw(), l2g_p.raw());

        cudaCheckError();
        /*
        //slow version of the above kernel
        for(int ii=0; ii<nrow; ii++){
            int s = RAP.row_offsets[ii];
            int e = RAP.row_offsets[ii+1];
            for (int jj=s; jj<e; jj++) {
                int col = RAP.col_indices[jj];
                if (col>=nrow){
                    int kk = col-RAP.get_num_rows();
                    RAP.col_indices[jj] = nrow+l2g_p[kk];
                }
            }
        }
        cudaCheckError();
        */
        //adjust matrix size (number of columns) accordingly
        RAP.set_initialized(0);
        RAP.set_num_cols(nrow + new_nl2g);
        RAP.set_initialized(1);
        //compress local_to_global_map using the pointer map
        nblocks = (nl2g + AMGX_CAL_BLOCK_SIZE - 1) / AMGX_CAL_BLOCK_SIZE;

        if (nblocks > 0)
            compress_existing_local_to_global_columns<int, int64_t> <<< nblocks, AMGX_CAL_BLOCK_SIZE>>>
            (nl2g, RAP.manager->local_to_global_map.raw(), l2g_t.raw(), l2g_p.raw());

        cudaCheckError();
        amgx::thrust::copy(l2g_t.begin(), l2g_t.begin() + new_nl2g, RAP.manager->local_to_global_map.begin());
        cudaCheckError();
        /*
        //slow version of the above kernel (through Thrust)
        for(int ii=0; ii<(l2g_p.size()-1); ii++){
            if (l2g_p[ii] != l2g_p[ii+1]){
                RAP.manager->local_to_global_map[l2g_p[ii]] = RAP.manager->local_to_global_map[ii];
            }
        }
        cudaCheckError();
        */
        //adjust local_to_global_map size accordingly
        RAP.manager->local_to_global_map.resize(new_nl2g);
    }
}

/**********************************************
 * prolongates the error: x+=P*e
 **********************************************/
template <class T_Config>
void Classical_AMG_Level_Base<T_Config>::prolongateAndApplyCorrection(VVector &e, VVector &bc, VVector &x, VVector &tmp)
{
    this->Profile.tic("proCorr");
    const short fine_x_block_dimy = x.get_block_dimy();
    const short fine_tmp_block_dimy = tmp.get_block_dimy();
    // Use P.manager to exchange halo of e before doing P
    // (since P has columns belonging to one of P.neighbors)
    e.dirtybit = 1;

    if (!P.is_matrix_singleGPU())
    {
        // get coarse matrix
        typedef typename TConfig::MemSpace MemorySpace;
        Matrix<TConfig> &Ac = this->getNextLevel( MemorySpace( ) )->getA();
#if COARSE_CLA_CONSO
        int e_size;

        if (this->getNextLevel(MemorySpace())->isConsolidationLevel())
        {
            e_size = std::max(P.manager->halo_offsets[P.manager->neighbors.size()], Ac.manager->halo_offsets_before_glue[Ac.manager->neighbors_before_glue.size()]) * e.get_block_size();
        }
        else
        {
            e_size = std::max(P.manager->halo_offsets[P.manager->neighbors.size()], Ac.manager->halo_offsets[Ac.manager->neighbors.size()]) * e.get_block_size();
        }

        if (e.size() < e_size) { e.resize(e_size); }

#else
        int e_size = std::max(P.manager->halo_offsets[P.manager->neighbors.size()], Ac.manager->halo_offsets[Ac.manager->neighbors.size()]) * e.get_block_size();
        e.resize(e_size);
#endif
    }

    if (P.is_matrix_singleGPU())
    {
        if (e.size() > 0)
        {
            if (usesBlockGraphHierarchy())
            {
                const size_t required_input = static_cast<size_t>(P.get_num_cols())
                                              * P.get_block_dimx();
                const size_t required_output = static_cast<size_t>(P.get_num_rows())
                                               * P.get_block_dimy();

                if (e.size() < required_input || tmp.size() < required_output)
                {
                    FatalError("Pure-BSR prolongation vector size mismatch",
                               AMGX_ERR_INTERNAL);
                }
            }

            if (usesIdentityBlockGraphHierarchy())
            {
                Block_Graph_Ops<TConfig>::multiply_identity_transfer(P, e, tmp);
            }
            else
            {
                multiply(P, e, tmp);
            }
        }
    }
    else
    {
        multiply_with_mask( P, e, tmp);
    }

    // get owned num rows for fine matrix
    int owned_size;

    if (this->A->is_matrix_distributed())
    {
        int owned_offset;
        P.manager->getOffsetAndSizeForView(OWNED, &owned_offset, &owned_size);
    }
    else
    {
        // axpby() measures size in vector blocks and multiplies it by
        // x.get_block_size().  In the pure-BSR hierarchy one vector block is
        // one matrix block row, so passing the scalar size would multiply by
        // the block dimension twice and overrun x/tmp.
        owned_size = usesBlockGraphHierarchy()
                     ? P.get_num_rows()
                     : x.size();
    }

    // Apply the correction. Only the hybrid scalar-transfer path needs a
    // temporary scalar metadata view; pure-BSR transfers already preserve b.
    const bool scalar_transfer_on_block_vector =
        P.get_block_size() == 1 && fine_x_block_dimy > 1;

    if (scalar_transfer_on_block_vector)
    {
        x.set_block_dimy(P.get_block_dimy());
        tmp.set_block_dimy(P.get_block_dimy());
    }

    if (usesBlockGraphHierarchy())
    {
        const size_t x_entries = static_cast<size_t>(owned_size)
                                 * x.get_block_size();
        const size_t tmp_entries = static_cast<size_t>(owned_size)
                                   * tmp.get_block_size();

        if (x_entries > x.size() || tmp_entries > tmp.size())
        {
            FatalError("Pure-BSR correction vector extent mismatch",
                       AMGX_ERR_INTERNAL);
        }
    }

    axpby(x, tmp, x, ValueType(1), ValueType(1), 0, owned_size);

    if (scalar_transfer_on_block_vector)
    {
        x.set_block_dimy(fine_x_block_dimy);
        tmp.set_block_dimy(fine_tmp_block_dimy);
    }
    this->Profile.toc("proCorr");
    x.dirtybit = 1;
}

template <class T_Config>
void Classical_AMG_Level_Base<T_Config>::computeAOperator()
{
    if (usesBlockGraphHierarchy())
    {
        computeAOperator_block_graph();
        return;
    }

    Matrix<TConfig> &A = getCoarseningMatrix();

    if (A.get_block_size() != 1)
    {
        FatalError("Classical AMG coarsening matrix must be scalar", AMGX_ERR_INTERNAL);
    }

    computeAOperator_1x1();
}

template <class T_Config>
void Classical_AMG_Level_Base<T_Config>::computeAOperator_distributed()
{
    if (usesBlockGraphHierarchy())
    {
        FatalError("Distributed block-graph BSR hierarchy is not implemented",
                   AMGX_ERR_NOT_IMPLEMENTED);
    }

    if (this->A->get_block_size() == 1)
    {
        computeAOperator_1x1_distributed();
    }
    else
    {
        FatalError("Classical AMG not implemented for block_size != 1", AMGX_ERR_NOT_IMPLEMENTED);
    }
}


template <class T_Config>
void Classical_AMG_Level_Base<T_Config>::consolidateVector(VVector &x)
{
#ifdef AMGX_WITH_MPI
#if COARSE_CLA_CONSO
    typedef typename TConfig::MemSpace MemorySpace;
    Matrix<TConfig> &A = this->getA();
    Matrix<TConfig> &Ac = this->getNextLevel( MemorySpace( ) )->getA();
    MPI_Comm comm, temp_com;
    comm = Ac.manager->getComms()->get_mpi_comm();
    temp_com = compute_glue_matrices_communicator(Ac);
    glue_vector(Ac, comm, x, temp_com);
#endif
#endif
}

template <class T_Config>
void Classical_AMG_Level_Base<T_Config>::unconsolidateVector(VVector &x)
{
#ifdef AMGX_WITH_MPI
#if COARSE_CLA_CONSO
    typedef typename TConfig::MemSpace MemorySpace;
    Matrix<TConfig> &A = this->getA();
    Matrix<TConfig> &Ac = this->getNextLevel( MemorySpace( ) )->getA();
    MPI_Comm comm, temp_com;
    comm = Ac.manager->getComms()->get_mpi_comm();
    temp_com = compute_glue_matrices_communicator(Ac);
    unglue_vector(Ac, comm, x, temp_com, x);
#endif
#endif
}

/****************************************
 * Explict instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class Classical_AMG_Level_Base<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class Classical_AMG_Level<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace classical

} // namespace amgx
