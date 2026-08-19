// SPDX-FileCopyrightText: 2013 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <solvers/jacobi_l1_solver.h>
#include <solvers/block_common_solver.h>
#include <thrust/transform.h>
#include <basic_types.h>
#include <blas.h>
#include <string.h>
#include <cutil.h>
#include <sm_utils.inl>
#include <util.h>
#include <multiply.h>
#include <miscmath.h>
#include <amgx_cusparse.h>

#include <algorithm>

namespace amgx
{

// -----------
// Kernels
// -----------

template <typename ValueTypeA, typename ValueTypeB>
__global__ void jacobi_l1_postsmooth(
    int n, ValueTypeB omega, ValueTypeB* x, ValueTypeA* d_in, ValueTypeB* b_in, ValueTypeB* y_in)
{
    int gid = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i = gid; i < n; i += blockDim.x * gridDim.x)
    {
        ValueTypeA d = d_in[i];
        ValueTypeB b = b_in[i];
        ValueTypeB y = y_in[i];
        d = ValueTypeA( 1 ) / (isNotCloseToZero(d) ? d : epsilon(d));
        b -= y;
        b *= omega;
        x[i] = b * d + x[i];
    }
}

template <typename ValueTypeA, typename ValueTypeB>
__global__ void jacobi_l1_postsmooth_zero(
    int n, ValueTypeB omega, ValueTypeB* x, ValueTypeA* d_in, ValueTypeB* b_in)
{
    int gid = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i = gid; i < n; i += blockDim.x * gridDim.x)
    {
        ValueTypeA d = d_in[i];
        ValueTypeB b = b_in[i];
        x[i] = omega * b / (isNotCloseToZero(d) ? d : epsilon(d));
    }
}

template<typename IndexType, typename ValueTypeA, typename ValueTypeB>
__global__ void compute_d_kernel(const IndexType num_block_rows,
                                 const IndexType *Ap,
                                 const IndexType *Aj,
                                 const ValueTypeA *Ax,
                                 const int block_dim,
                                 const bool row_major,
                                 ValueTypeA *d)
{
    const IndexType num_scalar_rows = num_block_rows * block_dim;
    IndexType tidx = blockDim.x * blockIdx.x + threadIdx.x;

    for (IndexType scalar_row = tidx; scalar_row < num_scalar_rows;
         scalar_row += blockDim.x * gridDim.x)
    {
        const IndexType block_row = scalar_row / block_dim;
        const IndexType local_row = scalar_row % block_dim;
        ValueTypeB d_ = 0;
        bool is_npd = false;

        for (IndexType block = Ap[block_row]; block < Ap[block_row + 1]; ++block)
        {
            const IndexType block_col = Aj[block];
            const IndexType value_begin = block * block_dim * block_dim;

            for (int local_col = 0; local_col < block_dim; ++local_col)
            {
                const IndexType local_value = row_major
                                              ? local_row * block_dim + local_col
                                              : local_col * block_dim + local_row;
                ValueTypeB Aij = Ax[value_begin + local_value];

                if (block_col == block_row && local_col == local_row && Aij < 0.)
                {
                    is_npd = true;
                }

                d_ += fabs(Aij);
            }
        }

        d[scalar_row] = is_npd ? -d_ : d_;
    }
}

template<typename IndexType, typename ValueTypeA, typename ValueTypeB, int CtaSize>
__global__ void compute_d_4x4_kernel(const IndexType num_rows,
                                     const IndexType *A_row_offsets,
                                     const IndexType *A_col_indices,
                                     const IndexType *A_dia_indices,
                                     const ValueTypeA *A_nonzero_values,
                                     ValueTypeA *d)
{
    const int nWarps = CtaSize / 32; // Number of warps per block
    const int warpId = utils::warp_id();
    const int laneId = utils::lane_id();
    // Lane ID in the 2 16-wide segments.
    const int lane_id_div_16 = laneId / 16;
    const int lane_id_mod_16 = laneId % 16;
    // Coordinates inside a 4x4 block of the matrix.
    const int idx_i = lane_id_mod_16 / 4;
    const int idx_j = lane_id_mod_16 % 4;
    __shared__ volatile ValueTypeA s_diag[nWarps][32];
    int aRowId = blockIdx.x * nWarps + warpId;

    while (aRowId < num_rows )
    {
        int aColBeg = A_row_offsets[aRowId];
        int aColEnd = A_row_offsets[aRowId + 1];
        s_diag[warpId][laneId] = 0.;

        //TODO: Should prefetch the column indices
        for (int i = aColBeg; i < aColEnd; i += 2)
        {
            const int my_i = i + lane_id_div_16;
            int aColId = my_i < aColEnd ? A_col_indices[my_i] : int (-1);
            s_diag[warpId][laneId] += (aRowId != aColId && aColId != -1) ? fabs(A_nonzero_values[16 * my_i + lane_id_mod_16]) : 0.;
        }

        // Threads 0 to 15 compute inverse
        if (lane_id_div_16 == 0)
        {
            s_diag[warpId][laneId] += s_diag[warpId][laneId + 16];
            s_diag[warpId][laneId] += fabs( A_nonzero_values[A_dia_indices[aRowId] * 16 + laneId]);
            const int offset = 16 * aRowId;
            compute_block_inverse_row_major4x4_formula2<int, ValueTypeA, 4, true>
            (s_diag[warpId], 16 * lane_id_div_16, offset, idx_i, idx_j, d);
        }

        aRowId += nWarps * gridDim.x;
    }
}


template<typename IndexType, typename ValueTypeA, typename ValueTypeB>
__global__ void jacobi_smooth_kernel(const IndexType num_rows,
                                     const IndexType *Ap,
                                     const IndexType *Aj,
                                     const ValueTypeA *Ax,
                                     const ValueTypeA *d,
                                     const ValueTypeB *b,
                                     const ValueTypeB *x,
                                     ValueTypeB *xout,
                                     const IndexType row_offset)

{
    IndexType tidx = blockDim.x * blockIdx.x + threadIdx.x;

    for (int ridx = row_offset + tidx; ridx < num_rows; ridx += blockDim.x * gridDim.x)
    {
        IndexType row_start = Ap[ridx];
        IndexType row_end   = Ap[ridx + 1];
        ValueTypeB Axi = 0.0;

        for (int j = row_start; j < row_end; j++)
        {
            Axi += Ax[j] * x[Aj[j]];
        }

        xout[ridx] = x[ridx] + (b[ridx] - Axi) / ( isNotCloseToZero( d[ridx]) ? d[ridx] : epsilon(d[ridx]) );
    }
}


template<typename IndexType, typename ValueTypeA, typename ValueTypeB, int blockrows_per_cta, int blockrows_per_warp, int bsize>
__global__ void jacobi_smooth_4x4_kernel(const IndexType num_rows,
        const IndexType *row_offsets,
        const IndexType *column_indices,
        const ValueTypeA *nonzero_values,
        const ValueTypeA *d,
        const ValueTypeB *b,
        const ValueTypeB *x,
        ValueTypeB *xout,
        const IndexType row_offset)

{
    int warp_id = threadIdx.x / 32;
    int warp_thread_id = threadIdx.x & 31;

    // padding row blocks to fit in a single warp
    if ( warp_thread_id >= blockrows_per_warp * bsize ) { return; }

    // new thread id with padding
    int tid = warp_id * blockrows_per_warp * bsize + warp_thread_id;
    // Here we use one thread per row (not block row)
    int cta_blockrow_id = (tid) / bsize;
    int blockrow_id = blockIdx.x * blockrows_per_cta + cta_blockrow_id;
    const int vec_entry_index = tid - cta_blockrow_id * bsize;
    volatile __shared__ ValueTypeB s_xtemp[ bsize * blockrows_per_cta];
    int offset, s_offset, i;
    ValueTypeB bmAx, temp[bsize];

    while (blockrow_id < num_rows &&  cta_blockrow_id < blockrows_per_cta)
    {
        i = blockrow_id;
        // Load RHS and x
        // groups of 4 threads load 32-byte, through texture, no wasted bandwidth
        offset = i * bsize + vec_entry_index;
        bmAx = b[offset];
        // Contribution from each nonzero column
        int jmax = row_offsets[i + 1];

        for (int jind = row_offsets[i]; jind < jmax; jind++)
        {
            IndexType jcol = column_indices[jind];
            offset = jcol * bsize + vec_entry_index;
            s_xtemp[tid] = x[offset] * (jcol != i);
            //for some reason we do better in perf if we write to volatile shared memory and then read this back...
            //s_xtemp[threadIdx.x] += delta[offset]*(row_colors[jcol] < current_color && current_color!= 0);
            // Load nonzero_values
            offset = jind * bsize * bsize + vec_entry_index * bsize;
            loadAsVector<bsize>(nonzero_values + offset, temp);
            // Do matrix multiply
            s_offset = cta_blockrow_id * bsize;
#pragma unroll

            for (int m = 0; m < bsize; m++)
            {
                bmAx -= temp[m] * s_xtemp[s_offset++];
            }
        }

        // Each thread stores bmAx in xtemp
        s_xtemp[tid] = bmAx;
        bmAx = 0.;
        // Load Dinv
        offset = i * bsize * bsize + vec_entry_index * bsize;
        loadAsVector<bsize>(d + offset, temp);
        // Do matrix-vector multiply
        s_offset = cta_blockrow_id * bsize;
#pragma unroll

        for (int m = 0; m < bsize; m++)
        {
            bmAx += temp[m] * s_xtemp[s_offset++];
        }

        xout[i * bsize + vec_entry_index] = bmAx;
        blockrow_id += blockrows_per_cta * gridDim.x;
    }
}



template<typename IndexType, typename ValueTypeA, typename ValueTypeB>
__global__ void jacobi_smooth_with_0_initial_guess_kernel(const IndexType num_rows,
        const ValueTypeA *d,
        const ValueTypeB *b,
        ValueTypeB *x,
        ValueTypeB weight,
        const IndexType row_offset)

{
    IndexType tidx = blockDim.x * blockIdx.x + threadIdx.x;

    for (int ridx = row_offset + tidx; ridx < num_rows; ridx += blockDim.x * gridDim.x)
    {
        x[ridx] = weight * b[ridx] /  ( isNotCloseToZero( d[ridx]) ? d[ridx] : epsilon(d[ridx]) );
    }
}

// -----------------
// Methods
// -----------------

// Constructor
template<class T_Config>
JacobiL1Solver_Base<T_Config>::JacobiL1Solver_Base( AMG_Config &cfg, const std::string &cfg_scope) : Solver<T_Config>( cfg, cfg_scope), m_d(0)
{
    weight = cfg.AMG_Config::template getParameter<double>("relaxation_factor", cfg_scope);
    scalar_rows_for_blocks = cfg.AMG_Config::template getParameter<int>("jacobi_l1_scalar_rows_for_blocks", cfg_scope) != 0;

    if (weight == 0)
    {
        weight = 1.;
        amgx_printf("Warning, setting weight to 1 instead of estimating largest_eigen_value in Block Jacobi smoother\n");;
    }
}

// Destructor
template<class T_Config>
JacobiL1Solver_Base<T_Config>::~JacobiL1Solver_Base()
{
}

// Solver setup
template<class T_Config>
void
JacobiL1Solver_Base<T_Config>::solver_setup(bool reuse_matrix_structure)
{
    m_explicit_A = dynamic_cast<Matrix<T_Config>*>(Base::m_A);

    if (!m_explicit_A)
    {
        FatalError("JacobiL1Solver only works with explicit matrices", AMGX_ERR_INTERNAL);
    }

    compute_d( *this->m_explicit_A );
    int N = this->m_explicit_A->get_num_cols() * this->m_explicit_A->get_block_dimy();
    this->y_tmp.resize(N);
}

template<class T_Config>
void JacobiL1Solver_Base<T_Config>::compute_d( Matrix<T_Config> &A)
{
    const bool scalar_matrix = A.get_block_dimx() == 1 && A.get_block_dimy() == 1;
    const bool scalar_rows = scalar_rows_for_blocks
                             && A.get_block_dimx() == A.get_block_dimy();
    const int diagonal_entries_per_block_row = scalar_matrix || scalar_rows
                                               ? A.get_block_dimy()
                                               : A.get_block_size();
    this->m_d.resize(A.get_num_rows() * diagonal_entries_per_block_row);
    ViewType oldView = A.currentView();
    A.setView(this->m_explicit_A->getViewExterior());

    if (scalar_matrix || scalar_rows)
    {
        compute_d_1x1(A);
    }
    else if (A.get_block_dimx() == 4 && A.get_block_dimy() == 4)
    {
        compute_d_4x4(A);
    }
    else
    {
        FatalError("Unsupported block size for JacobiL1Solver; set jacobi_l1_scalar_rows_for_blocks=1 for square BSR matrices", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }

    A.setView(oldView);
}


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void JacobiL1Solver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::compute_d_1x1(const Matrix_h &A)
{
    if (A.hasProps(DIAG))
    {
        FatalError("Scalar-row JacobiL1 does not support a separately stored block diagonal", AMGX_ERR_NOT_IMPLEMENTED);
    }

    const int block_dim = A.get_block_dimy();
    const bool row_major = A.getBlockFormat() == ROW_MAJOR;

    for (int block_row = 0; block_row < A.get_num_rows(); ++block_row)
    {
        for (int local_row = 0; local_row < block_dim; ++local_row)
        {
            ValueTypeB d_ = 0;
            bool is_npd = false;

            for (int block = A.row_offsets[block_row]; block < A.row_offsets[block_row + 1]; ++block)
            {
                for (int local_col = 0; local_col < block_dim; ++local_col)
                {
                    const int local_value = row_major
                                            ? local_row * block_dim + local_col
                                            : local_col * block_dim + local_row;
                    ValueTypeB Aij = A.values[block * block_dim * block_dim + local_value];

                    if (A.col_indices[block] == block_row && local_col == local_row && Aij < 0.)
                    {
                        is_npd = true;
                    }

                    d_ += fabs(Aij);
                }
            }

            this->m_d[block_row * block_dim + local_row] = is_npd ? -d_ : d_;
        }
    }
}


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void JacobiL1Solver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::compute_d_4x4(const Matrix_h &A)
{
    FatalError("4x4 block matrices not supported on host", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
}


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void JacobiL1Solver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::compute_d_1x1(const Matrix_d &A)
{
    if (A.hasProps(DIAG))
    {
        FatalError("Scalar-row JacobiL1 does not support a separately stored block diagonal", AMGX_ERR_NOT_IMPLEMENTED);
    }

    typedef typename Matrix_d::index_type IndexType;
    typedef typename Matrix_d::value_type ValueTypeA;
    const int threads_per_block = 128;
    const int scalar_rows = A.get_num_rows() * A.get_block_dimy();
    const int num_blocks = std::min(AMGX_GRID_MAX_SIZE,
                                    (scalar_rows + threads_per_block - 1) / threads_per_block);

    if (scalar_rows > 0)
    {
        compute_d_kernel<IndexType, ValueTypeA, ValueTypeB>
        <<<num_blocks, threads_per_block>>>(A.get_num_rows(),
                                            A.row_offsets.raw(),
                                            A.col_indices.raw(),
                                            A.values.raw(),
                                            A.get_block_dimy(),
                                            A.getBlockFormat() == ROW_MAJOR,
                                            this->m_d.raw());
    }

    cudaCheckError();
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void JacobiL1Solver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::compute_d_4x4(const Matrix_d &A)
{
    // now leaving !DIAG only
    typedef typename Matrix_d::index_type IndexType;
    typedef typename Matrix_d::value_type ValueTypeA;
    const int cta_size = 128;
    const int nWarps = cta_size / 32;
    int grid_size = std::min( 1024, ( A.get_num_rows() + nWarps - 1 ) / nWarps );
    compute_d_4x4_kernel<IndexType, ValueTypeA, ValueTypeB, cta_size > <<< grid_size, cta_size >>>
    (A.get_num_rows(),
     A.row_offsets.raw(),
     A.col_indices.raw(),
     A.diag.raw(),
     A.values.raw(),
     this->m_d.raw());
    cudaCheckError();
}





//
template<class T_Config>
void
JacobiL1Solver_Base<T_Config>::solve_init( VVector &b, VVector &x, bool xIsZero )
{
}


// Solve one iteration
template<class T_Config>
AMGX_STATUS
JacobiL1Solver_Base<T_Config>::solve_iteration( VVector &b, VVector &x, bool xIsZero )
{
    if (xIsZero) { x.dirtybit = 0; }

    ViewType oldView = this->m_explicit_A->currentView();
    this->m_explicit_A->setViewExterior();

    const bool scalar_matrix = this->m_explicit_A->get_block_dimx() == 1
                               && this->m_explicit_A->get_block_dimy() == 1;
    const bool scalar_rows = scalar_rows_for_blocks
                             && this->m_explicit_A->get_block_dimx() == this->m_explicit_A->get_block_dimy();

    if (scalar_matrix || scalar_rows)
    {
        if (xIsZero)
        {
            smooth_with_0_initial_guess_1x1(*this->m_explicit_A, b, x, this->m_explicit_A->getViewExterior());
        }
        else
        {
            smooth_1x1(*this->m_explicit_A, b, x, this->m_explicit_A->getViewExterior(), false);
        }
    }
    else if (this->m_explicit_A->get_block_dimx() == 4
             && this->m_explicit_A->get_block_dimy() == 4)
    {
        smooth_4x4(*this->m_explicit_A, b, x, this->m_explicit_A->getViewExterior());
    }
    else
    {
        FatalError("Unsupported block size for JacobiL1Solver; set jacobi_l1_scalar_rows_for_blocks=1 for square BSR matrices", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }

    x.dirtybit = 1;
    this->m_explicit_A->setView(oldView);
    return this->converged( b, x );
}

template<class T_Config>
void
JacobiL1Solver_Base<T_Config>::solve_finalize( VVector &b, VVector &x )
{
}



template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void JacobiL1Solver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::smooth_1x1(Matrix_h &A, VVector &b, VVector &x, ViewType separation_flags, bool latency_hiding)
{
    VVector newx((int)x.size());
    const int block_dim = A.get_block_dimy();
    const bool row_major = A.getBlockFormat() == ROW_MAJOR;

    for (int block_row = 0; block_row < A.get_num_rows(); ++block_row)
    {
        for (int local_row = 0; local_row < block_dim; ++local_row)
        {
            const int scalar_row = block_row * block_dim + local_row;
            ValueTypeB Axi = 0.0;

            for (int block = A.row_offsets[block_row]; block < A.row_offsets[block_row + 1]; ++block)
            {
                const int scalar_col_begin = A.col_indices[block] * block_dim;

                for (int local_col = 0; local_col < block_dim; ++local_col)
                {
                    const int local_value = row_major
                                            ? local_row * block_dim + local_col
                                            : local_col * block_dim + local_row;
                    Axi += A.values[block * block_dim * block_dim + local_value]
                           * x[scalar_col_begin + local_col];
                }
            }

            ValueTypeA d = this->m_d[scalar_row];
            newx[scalar_row] = x[scalar_row]
                               + (b[scalar_row] - Axi)
                               / (isNotCloseToZero(d) ? d : epsilon(d));
        }
    }

    x.swap(newx);
}


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void JacobiL1Solver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::smooth_4x4(Matrix_h &A, VVector &b, VVector &x, ViewType separation_flags)
{
    FatalError("4x4 block size not supported on host", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void JacobiL1Solver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::smooth_with_0_initial_guess_1x1(Matrix_h &A, VVector &b, VVector &x, ViewType separation_flags)
{
    const int scalar_rows = A.get_num_rows() * A.get_block_dimy();

    for (int i = 0; i < scalar_rows; ++i)
    {
        ValueTypeA d = this->m_d[i];
        x[i] = b[i] / (isNotCloseToZero(d) ? d : epsilon(d));
    }
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void JacobiL1Solver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::smooth_1x1(Matrix_d &A, VVector &b, VVector &x, ViewType separation_flags, bool latency_hiding)
{
    this->y_tmp.set_block_dimx(b.get_block_dimx());
    this->y_tmp.set_block_dimy(b.get_block_dimy());

    multiply(A, x, this->y_tmp, A.getViewExterior());

    int offset, num_rows;
    A.getOffsetAndSizeForView(A.getViewExterior(), &offset, &num_rows);

    const int scalar_offset = offset * A.get_block_dimy();
    const int n = (num_rows - offset) * A.get_block_dimy();
    const int threads_per_block = 128;
    const int num_blocks = (n + threads_per_block - 1) / threads_per_block;

    if (n > 0)
    {
        jacobi_l1_postsmooth<<<num_blocks, threads_per_block>>>(n, this->weight,
                x.raw() + scalar_offset, this->m_d.raw() + scalar_offset,
                b.raw() + scalar_offset, this->y_tmp.raw() + scalar_offset);
    }

    cudaCheckError();
}


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void JacobiL1Solver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::smooth_4x4(Matrix_d &A, VVector &b, VVector &x, ViewType separation_flags)
{
    typedef typename Matrix_d::index_type IndexType;
    typedef typename Matrix_d::value_type ValueTypeA;
    VVector xout(x.size());
    ValueTypeB *x_ptr = x.raw();
    ValueTypeB *xout_ptr = xout.raw();
    IndexType num_rows = A.get_num_rows();
    IndexType offset = 0;

    //It is only safe to swap these vectors if there is no halo exchange in process
    if (separation_flags != this->m_explicit_A->getViewExterior())
    {
        amgx::thrust::copy(x.begin(), x.end(), xout.begin()); //TODO: only interior+bndry part
        cudaCheckError();
        x_ptr = xout.raw();
        xout_ptr = x.raw();

        if (separation_flags & INTERIOR)
        {
        }
        else
        {
            A.getOffsetAndSizeForView(separation_flags, &offset, &num_rows);
        }
    }

    const int threads_per_block = 128;
    const int blockrows_per_warp = 32 / 4;
    const int blockrows_per_cta = (threads_per_block / 32) * blockrows_per_warp;
    const int num_blocks = std::min( AMGX_GRID_MAX_SIZE, (int) (A.get_num_rows() - 1) / blockrows_per_cta + 1);
    jacobi_smooth_4x4_kernel<IndexType, ValueTypeA, ValueTypeB, blockrows_per_cta, blockrows_per_warp, 4> <<< num_blocks, threads_per_block >>>
    ((int)A.get_num_rows(),
     A.row_offsets.raw(),
     A.col_indices.raw(),
     A.values.raw(),
     this->m_d.raw(),
     b.raw(),
     x_ptr,
     xout_ptr, offset);
    cudaCheckError();

    if ((separation_flags != this->m_explicit_A->getViewExterior())/* && (separation_flags & INTERIOR)*/) {}
    else { x.swap(xout); }
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void JacobiL1Solver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::smooth_with_0_initial_guess_1x1(Matrix_d &A, VVector &b, VVector &x, ViewType separation_flags)
{
    ViewType oldView = A.currentView();
    // Process all rows
    A.setViewExterior();

    int offset, num_rows;
    A.getOffsetAndSizeForView(A.getViewExterior(), &offset, &num_rows);

    const int scalar_offset = offset * A.get_block_dimy();
    const int n = (num_rows - offset) * A.get_block_dimy();
    const int threads_per_block = 128;
    const int num_blocks = (n + threads_per_block - 1) / threads_per_block;

    if (n > 0)
    {
        jacobi_l1_postsmooth_zero<<<num_blocks, threads_per_block>>>(n, this->weight,
                x.raw() + scalar_offset, this->m_d.raw() + scalar_offset,
                b.raw() + scalar_offset);
    }

    cudaCheckError();

    A.setView(oldView);
    cudaCheckError();
}


/****************************************
 * Explict instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class JacobiL1Solver_Base<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class JacobiL1Solver<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace amgx
