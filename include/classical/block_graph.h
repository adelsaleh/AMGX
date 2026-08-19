// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <matrix.h>
#include <vector.h>

namespace amgx
{
namespace classical
{

// Setup helpers for the single-GPU classical block-graph hierarchy.  The
// scalar matrices passed to these helpers are temporary graph/weight objects;
// every retained hierarchy operator remains BSR.
template <class T_Config>
class Block_Graph_Ops;

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec,
          AMGX_IndPrecision t_indPrec>
class Block_Graph_Ops<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
{
    typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig;

public:
    static void build_graph(const Matrix<TConfig> &A, Matrix<TConfig> &graph,
                            int strength_metric);
    static void lift_scalar_transfer(const Matrix<TConfig> &scalar_transfer,
                                     int block_dim, BlockFormat block_format,
                                     Matrix<TConfig> &block_transfer);
    static void extract_scalar_transfer(const Matrix<TConfig> &block_transfer,
                                        Matrix<TConfig> &scalar_transfer);
    static void multiply_identity_transfer(Matrix<TConfig> &transfer,
                                           Vector<TConfig> &input,
                                           Vector<TConfig> &output);
    static void smooth_dense_transfer(const Matrix<TConfig> &A,
                                      const Matrix<TConfig> &scalar_transfer,
                                      double smoothing_weight,
                                      int smoothing_steps,
                                      bool right_normalize,
                                      double pivot_tolerance,
                                      double constraint_tolerance,
                                      Matrix<TConfig> &block_transfer);
    static void extended_i_dense_transfer(
        const Matrix<TConfig> &A,
        const Matrix<TConfig> &scalar_transfer,
        const Vector<typename TConfig::template setVecPrec<AMGX_vecInt>::Type> &cf_map,
        const Vector<typename TConfig::template setVecPrec<AMGX_vecBool>::Type> &s_con,
        double pivot_tolerance, double constraint_tolerance,
        Matrix<TConfig> &block_transfer);
    static void transpose_dense_transfer(const Matrix<TConfig> &scalar_transpose,
                                         const Matrix<TConfig> &block_transfer,
                                         Matrix<TConfig> &block_transpose);
    static void weighted_galerkin(const Matrix<TConfig> &A,
                                  const Matrix<TConfig> &graph_A,
                                  const Matrix<TConfig> &graph_R,
                                  const Matrix<TConfig> &graph_P,
                                  Matrix<TConfig> &Ac, void *workspace);
    static void dense_galerkin(const Matrix<TConfig> &A,
                               const Matrix<TConfig> &graph_A,
                               const Matrix<TConfig> &graph_R,
                               const Matrix<TConfig> &graph_P,
                               const Matrix<TConfig> &R,
                               const Matrix<TConfig> &P,
                               Matrix<TConfig> &Ac, void *workspace);
};

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec,
          AMGX_IndPrecision t_indPrec>
class Block_Graph_Ops<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
{
    typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig;

public:
    static void build_graph(const Matrix<TConfig> &A, Matrix<TConfig> &graph,
                            int strength_metric);
    static void lift_scalar_transfer(const Matrix<TConfig> &scalar_transfer,
                                     int block_dim, BlockFormat block_format,
                                     Matrix<TConfig> &block_transfer);
    static void extract_scalar_transfer(const Matrix<TConfig> &block_transfer,
                                        Matrix<TConfig> &scalar_transfer);
    static void multiply_identity_transfer(Matrix<TConfig> &transfer,
                                           Vector<TConfig> &input,
                                           Vector<TConfig> &output);
    static void smooth_dense_transfer(const Matrix<TConfig> &A,
                                      const Matrix<TConfig> &scalar_transfer,
                                      double smoothing_weight,
                                      int smoothing_steps,
                                      bool right_normalize,
                                      double pivot_tolerance,
                                      double constraint_tolerance,
                                      Matrix<TConfig> &block_transfer);
    static void extended_i_dense_transfer(
        const Matrix<TConfig> &A,
        const Matrix<TConfig> &scalar_transfer,
        const Vector<typename TConfig::template setVecPrec<AMGX_vecInt>::Type> &cf_map,
        const Vector<typename TConfig::template setVecPrec<AMGX_vecBool>::Type> &s_con,
        double pivot_tolerance, double constraint_tolerance,
        Matrix<TConfig> &block_transfer);
    static void transpose_dense_transfer(const Matrix<TConfig> &scalar_transpose,
                                         const Matrix<TConfig> &block_transfer,
                                         Matrix<TConfig> &block_transpose);
    static void weighted_galerkin(const Matrix<TConfig> &A,
                                  const Matrix<TConfig> &graph_A,
                                  const Matrix<TConfig> &graph_R,
                                  const Matrix<TConfig> &graph_P,
                                  Matrix<TConfig> &Ac, void *workspace);
    static void dense_galerkin(const Matrix<TConfig> &A,
                               const Matrix<TConfig> &graph_A,
                               const Matrix<TConfig> &graph_R,
                               const Matrix<TConfig> &graph_P,
                               const Matrix<TConfig> &R,
                               const Matrix<TConfig> &P,
                               Matrix<TConfig> &Ac, void *workspace);
};

} // namespace classical
} // namespace amgx
