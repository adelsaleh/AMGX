# AMGX communication and logging roadmap

## P0 — block sparse GPU performance

**Next roadmap item:** promote the modern cuSPARSE generic-BSR reference path to a matrix-owned cache with persistent descriptors, preprocessing state, and workspace.

- [x] Add an opt-in `cusparse_generic` BSR SpMV reference backend for complete device views, with legacy fallback for partial/distributed views, unsupported precision combinations, and CUDA Toolkits older than 13.0.
  - CUDA/cuSPARSE 12.8 accepts `cusparseCreateBsr`, but its `cusparseSpMV_bufferSize` rejects BSR at runtime; the local smoke test therefore exercises and verifies the legacy fallback. Generic BSR SpMV arrived in CUDA Toolkit 13.0 Update 1. Gate it on `CUDART_VERSION`, because Toolkit 13.0 ships cuSPARSE library major 12 and `CUSPARSE_VER_MAJOR` is not a CUDA Toolkit version test.
- [x] Add opt-in subgroup kernels for the hot 5x5 BSR paths, retaining legacy cuSPARSE/two-`bsrmv` behavior when `block_jacobi_use_fused_small_blocks=0`.
  - The standalone SpMV and fused Block Jacobi specializations assign an aligned eight-thread subgroup to each block row. Five lanes compute the five output/residual components concurrently; Jacobi reuses its residual through shared memory for the inverse-diagonal update.
  - Store dispatch state on each matrix through copied `AuxData`, so hierarchy levels and concurrent solver objects do not depend on process-global state.
  - Keep 2x2 and 3x3 on the one-thread fused specialization until profiling justifies a subgroup variant; 3x3 and 4x4 standalone SpMV already have AMGX custom kernels.
- [ ] Add an opt-in modern cuSPARSE BSR SpMV path for generic block sizes, with descriptors and external workspace owned by the matrix/resource lifecycle.
  - Cache `cusparseSpMatDescr_t`, dense-vector descriptors, preprocessing state, and workspace using matrix structure/value revisions; invalidate them on pointer, shape, block-format, precision, device, or stream-incompatible changes.
  - Do not use a process-global pointer-keyed cache: it is unsafe under matrix destruction, allocator address reuse, resetup, and concurrent solver objects.
  - Retain the legacy `cusparse*bsrmv` path for A/B tests and unsupported CUDA/type combinations.
- [ ] Profile standalone BSR SpMV and complete AMG solves separately for block sizes 2 through 5. Record kernel time, launch count, achieved bandwidth, setup/preprocess cost, smoother time, coarse-level format changes, iterations, and end-to-end solve time.
- [ ] Decide specialization policy from measurements: modern cuSPARSE for generic blocks, custom subgroup kernels for repeatedly hot awkward sizes, or a size/architecture-dependent dispatcher.

## P0 — classical AMG on BSR fine operators

- [x] Fix the CUDA 13 classical-CSR setup failure caused by a large device-side `thrust::sequence` dispatch in CSR Galerkin products. Use AMGX's small compatibility kernel for CSR row sequences instead; the 152,909-triangle, p=2 HDG disk case now completes in 19 iterations with an independently checked relative residual of `8.822e-09`.
- [x] Build and validate the source-complete classical-BSR compatibility path on the same CSR/BSR matrix pair.
  - Keep the uploaded fine operator in BSR for the fine smoother and fine-level SpMV.
  - During hierarchy construction only, expand each dense BSR block coefficient-exactly into a temporary scalar CSR matrix; run the existing scalar strength, PMIS/aggressive selection, interpolation, and Galerkin code unchanged. Release the selection expansion immediately (including when the driver rejects the candidate coarse level), recreate it for an accepted/reused level, and release it again after the first coarse operator is built. Coarse levels remain scalar.
  - The first implementation supports a single GPU, square blocks, row- or column-major blocks, and an internal diagonal. It deliberately rejects distributed BSR matrices and separately stored diagonal blocks.
  - This is a hybrid fine-BSR/scalar-hierarchy method intended to reproduce the classical CSR hierarchy and convergence. It is not a claim that AMGX now implements block-valued strength, interpolation, or Galerkin coarsening.
  - Small p=2 validation (175 triangles) completes with the CSR classical config and gives an `8.841e-17` relative primal-coefficient difference between CSR and BSR.
  - Add the opt-in `jacobi_l1_scalar_rows_for_blocks=1` path to compute the exact scalar-CSR row L1 diagonal while retaining BSR SpMV. Tag coarse V-cycle work vectors with the next-level scalar block dimensions at the fine-BSR/scalar-CSR transition.
  - Heavy validation on the unstructured radius-5 trigonometric-Poisson disk (152,909 triangles, p=2, 686,724 trace unknowns) completes with independently checked relative residuals of `8.822e-09` for CSR and `4.666e-09` for BSR. CSR uses 19 iterations, 1.474 s setup, and 0.150 s solve; BSR uses 20 iterations, 0.060 s setup, and 0.116 s solve. The BSR path is 1.30x faster in solve and reduces compressed-pattern storage from 41.9 MiB to 5.2 MiB.
  - The solve is BSR on the fine level, including its SpMV and Jacobi-L1 smoother, but restriction, prolongation, and coarse operators remain scalar CSR. A completely CSR-free V-cycle requires the separate block-aware classical-hierarchy project below.
- [ ] Compare classical CSR, legacy-SpMV BSR, and generic-cuSPARSE-SpMV BSR on the unstructured radius-5 trigonometric Poisson disk for p=1 through p=6. Separate scalar-expansion setup cost, total hierarchy setup, fine SpMV/smoother time, iterations, solve time, peak memory, and independently checked residual.
- [ ] Validate and optimize the opt-in pure-BSR classical hierarchy.
  - [x] Document and implement the first `block_graph_identity` algorithm: Frobenius-norm scalar block graph, existing PMIS/D2 on that graph, retained transfers `P_ic = w_ic I_b`, and a weighted BSR Galerkin numeric kernel. The coefficient-exact `scalar_expand` hybrid remains the default.
  - [x] Generalize DenseLU's BSR-to-dense conversion beyond 32 coefficients per block. Warp lanes now traverse block entries in strides of 32, so 6x6 and 7x7 coarse blocks are copied completely instead of producing a singular truncated dense matrix.
  - [x] Run bounded post-rebuild solver validation on the radius-5, 2,079-triangle trigonometric-Poisson disk. The apparent pre-fix p=6 PCGF breakdown was later traced to the multilevel correction overrun rather than a CG-incompatible hierarchy.
  - [x] Fix the multilevel prolongation overrun: `axpby` consumes a vector-block count, so pure BSR must pass `P.num_rows`, not `P.num_rows*b`. Add an extent invariant and use AMGX's configured stream for every block-graph kernel. The former three-level p=6 reproducer passes CUDA Initcheck and Memcheck with zero errors.
  - [x] Repeat the unstructured radius-5 trigonometric-Poisson sweep at 99,896, 124,831, and 150,209 triangles for p=1..6. All 72 CSR/FGMRES-BSR runs converge without memory faults, but identity-lifted BSR needs 375--959 FGMRES iterations versus 19--21 CSR PCGF iterations and is not production-competitive.
  - [x] Validate pure-BSR PCGF on the same 18 large mesh/degree cases. All converge with independently checked relative residual at most `9.869e-9`; PCGF is 1.34--2.04x faster than pure-BSR FGMRES but remains 11.8--33.9x slower than coefficient-exact scalar classical AMG. The former SPD-compatibility concern was an overrun artifact; coarse-space quality remains the limiting issue.
  - [ ] Reconfigure/rebuild the native test launcher and run the new 6x6/7x7 DenseLU regression plus small coefficient-level `RAP` and constant-mode tests.
  - [ ] Validate and optimize dense `b x b` interpolation on the same coarse block graph.
    - [x] Implement the separate opt-in `block_graph_dense` mode: fixed-support block-Jacobi smoothing of `w_ic I_b`, pivoted diagonal-block solves, the explicit constraint `sum_c P_ic = I_b`, exact `R=P^*`, and dense BSR `P^* A P`. Keep `block_graph_identity` numerically unchanged and reject unsupported hierarchy reuse explicitly.
    - [x] Rebuild and run the first multilevel numerical validation. On the 4,658-triangle radius-5 disk at p=6 and relative tolerance `1e-10`, dense pure BSR used 27 iterations versus scalar CSR's 25, with a `1.476e-10` primal-coefficient relative difference. Its solve was still slower (0.098 s versus 0.065 s), but compressed-pattern storage was 2.38% of CSR's. At the default tolerance, it reduced the weak identity hierarchy from 117 to 22 iterations on the same case.
    - [x] Reject `aggressive_levels > 0` for the first dense implementation. Aggressive D2 can produce empty interpolation rows, which cannot satisfy `sum_c P_ic = I_b`; support completion is a separate algorithmic extension.
    - [x] Execute the 152,909-triangle radius-5 trigonometric-Poisson comparison for p=1..6 with two runs each of scalar CSR, hybrid fine-BSR/scalar hierarchy, and dense pure BSR. All 36 solves converge. Dense pure BSR uses 15--53 iterations versus hybrid's 20--23, but becomes 3.05--4.45x slower than hybrid for p=3..6; it retains the 27.28%--2.38% BSR/CSR pattern ratio.
    - [x] Isolate the first large-p interpolation limitation: on the 152,909-triangle p=6 case, increasing D2 support from 4 to unlimited changes 53 iterations to 54, while weights from 0.25 through 1.2 remain in the 52--54 range. Neither scalar lever explains the hybrid gap.
    - [x] Screen the Frobenius/AHAT block strength threshold independently. At p=6, `0.47` improves 53 iterations/1.227 s solve to 41/1.038 s, while `0.48` crosses a hierarchy transition and degrades to 68 iterations. At p=2, `0.47` changes 20 to 24 iterations but reduces setup/solve from 1.222/0.217 s to 0.295/0.181 s. Retain both `0.25` and `0.47` in the post-rebuild factorial screen.
    - [x] Add configurable true dense-block interpolation smoothing steps and opt-in right block-row normalization `P_ic <- P_ic (sum_d P_id)^-1`. Keep one-step additive correction as the reproducible baseline; both variants retain BSR `P`, `R=P^*`, and `RAP`.
    - [x] Rebuild and screen one through four smoothing steps with additive and right-normalized constraints at p=2 and p=6. Extra additive sweeps are ineffective: p=2 moves only 20 to 19 iterations and p=6 only 53 to 52. Right normalization is rejected: one sweep raises p=6 to 146 iterations, while two or more sweeps create singular coarse diagonal blocks or fail the constant-mode check. Do not run a p=1..6 promotion screen for this rejected candidate.
    - [x] Evaluate the opt-in diagonal-normalized Frobenius coarse-face metric `||A_ij||_F / sqrt(||A_ii||_F ||A_jj||_F)`. Its p=6 optimum is 41 iterations at threshold `0.44`, exactly matching raw Frobenius's 41 at `0.47`; p=2 uses 24 iterations. It shifts the hierarchy transition but does not improve the best coarse space.
    - [x] Evaluate the full symmetric inverse-diagonal block-action metric `sqrt(||A_ii^-1 A_ij||_F ||A_jj^-1 A_ji||_F)`. Its best p=6 result is 42 iterations at thresholds `0.42`--`0.44`, versus 41 for the cheaper raw/normalized metrics, and setup rises to about 2.6 s. At p=2 it gives 20 iterations at `0.25` and 26 at `0.42`. Scalar edge metrics are no longer the leading hypothesis.
    - [x] Reject coefficient-exact scalar-guided any-mode face promotion. It improves the 4,658-triangle p=2 smoke case from 14 to 12 iterations, but regresses the 152,909-triangle p=2 solve from 20 to 28 iterations (0.218 s to 0.811 s). The p=6 temporary scalar expansion also exceeds the remaining memory on the 24 GiB device. Whole-face promotion from any coarse scalar mode is too aggressive and not memory-safe at production scale.
    - [x] Reject block Extended+i interpolation on the existing PMIS/D2 face support. It is correct and memory-safe but changes production p=2 from 20 to 21 iterations and p=6 from 53 to 55 at threshold 0.25; at the p=6 threshold-0.47 optimum it gives 43 versus Jacobi's 41. Retain the implementation as a diagnostic, not the production default.
    - [ ] Add native coefficient-level constant-mode, transpose, and Galerkin regression tests and run CUDA Initcheck/Memcheck.
    - [ ] Profile the reference dense Galerkin setup kernel and replace it with a tiled or library-backed implementation only if setup cost is material.

This roadmap focuses on trustworthy solver telemetry, actionable diagnostics, and stable communication contracts for C/C++, Python, MPI, CuPy, and other consumers. A log line, stored residual, convergence decision, and C API query must describe the same solver state. GPU interoperability work is coordinated against the local `../pyamgx` and `../cupy` repositories rather than inferred from raw pointers alone.

## P0 — HDG face-block p/h multigrid support

- [ ] Support the reduced scalar-AMG role in the HDG [face-block hp-multigrid
  plan](../hdg/docs/development/plans/face_block_hp_multigrid.md). Accept the
  p=0 face Galerkin operator as a scalar CSR matrix, build one reusable classical
  hierarchy, and expose a fixed one-V-cycle application suitable for a symmetric
  outer preconditioner without rebuilding or reallocating per right-hand side.
- [ ] Add/verify a symmetric classical-AMG application contract for this role:
  `presweeps=postsweeps=1`, transpose restriction, fixed smoother parameters,
  fixed coarse work, `error_scaling=0`, and an SPD-compatible terminal solve.
  Publish enough machine-readable setup/apply state for HDG to test adjointness
  and distinguish hierarchy setup from repeated coarse applications.
- [ ] Keep the coefficient-exact hybrid fine-BSR/scalar-hierarchy solver as the
  performance baseline. The p/h candidate is promoted only by complete
  end-to-end timings and independently checked FP64 residuals, never by block
  storage or standalone SpMV results alone.
- [ ] Coordinate generic-BSR tuning for block sizes 8, 9, and 10 so the
  cuSPARSE-first path covers HDG Poisson p=7,8,9; retain custom kernels only
  where profiling beats cached generic cuSPARSE on the target architecture.

## P0 — residual correctness

- [ ] Fix BiCGSTAB iteration residual reporting in `src/solvers/bicgstab_solver.cu`.
  - After `r = s - omega * t`, call `compute_norm_and_converged(*m_r, ...)`; the current normal path passes `m_s`, so `m_nrm`, `print_solve_stats`, and `store_res_history` describe the intermediate residual rather than the completed iteration.
  - Keep the `s`-based early-exit check, but after updating `x`, recompute or otherwise verify `b - A*x` before publishing the completed iteration.
  - Confirm that the matrix view is restored on every return path, including convergence, breakdown, and last-iteration exits.
- [ ] Add a deterministic BiCGSTAB regression test using a nonsymmetric matrix and `max_iters=1`.
  - Reproduction observed through pyamgx on 2026-08-11: initial/printed/history residual `3.7416573867739413`, independently evaluated final `||b-Ax||_2 = 0.7197162587140464`.
  - Assert the per-iteration printed value, stored value returned by `AMGX_solver_get_iteration_residual`, convergence value, and independently computed norm agree within a precision-dependent tolerance.
  - Cover unpreconditioned and preconditioned solves, zero and nonzero initial guesses, the `s` early-exit branch, a normal `omega` step, the final allowed iteration, block systems, and single-/double-precision modes.
  - Add breakdown cases for zero or near-zero `dot(r_tilde, v)`, `dot(t, t)`, `rho`, and `omega`; return a defined status and diagnostic instead of publishing NaN/Inf as an ordinary residual.
- [ ] Define and document the residual contract shared by every iterative solver.
  - Specify whether each reported value is a recursive/estimated residual or an explicitly recomputed true residual.
  - Give iteration zero and iteration `k` unambiguous meanings; keep table row numbers, `m_num_iters`, history indices, and the C API aligned.
  - If recurrence drift is possible, add a configurable reliable residual recomputation policy and label estimated versus true residuals in structured output.
- [ ] Audit CG, PCG/PBICGSTAB, GMRES/FGMRES, IDR, and nested AMG paths against that contract. Reuse a solver-independent test harness instead of testing only final convergence.

## P0 — convergence diagnostics

- [ ] Fix [AMGX issue #350](https://github.com/NVIDIA/AMGX/issues/350): an `ABSOLUTE` solve can stop with the message “Relative residual has reached machine precision.”
  - Make machine-precision/stagnation checks respect the configured convergence mode and norm scaling.
  - Log the actual stopping reason, criterion, threshold, current value, initial/reference value, and resulting `AMGX_SOLVE_STATUS`.
  - Test `ABSOLUTE`, `RELATIVE_INI(_CORE)`, and `RELATIVE_MAX(_CORE)` independently, including tiny initial residuals and exact-zero right-hand sides.
- [ ] Separate “converged,” “maximum iterations,” “stagnated,” “numerical breakdown,” and “diverged” internally and in the public status/diagnostic path; do not collapse a numerical stop into a successful convergence message.
- [ ] Future project: provide a solver-independent path that attempts to satisfy the user-requested true relative residual target `||A*x-b|| / ||b||`, regardless of the iterative solver's native convergence convention.
  - Treat an explicitly recomputed residual of the solved system as the authoritative final accuracy measure; do not relabel `||r_k|| / ||r_0||`, a recursive Krylov estimate, or a preconditioned/scaled residual as this quantity.
  - Define behavior for zero or numerically tiny `||b||`, nonzero initial guesses, scaled systems, block norms, distributed matrices, finite-precision stagnation, and targets below attainable precision.
  - Investigate reliable residual replacement, iterative refinement, and bounded retry/fallback policies so AMGX can continue toward the requested target when its native stopping test finishes too early, while still terminating with an explicit reason when the target is unattainable.

## P1 — logging and communication API

- [ ] Centralize output behind one logging interface; remove solver-path `printf`/`std::cout` bypasses so `AMGX_register_print_callback` receives all user-visible diagnostics.
- [ ] Add a backward-compatible structured callback API alongside the existing text callback. Each event should carry severity, component/solver and scope, event kind, iteration, residual kind/norm, rank/device, and message; retain the text formatter for existing applications.
- [ ] Define callback behavior explicitly: invocation thread, ordering, ownership and byte length, reentrancy restrictions, lifetime, unregister/reset behavior, and what happens when the consumer is slow or fails.
- [ ] Make MPI output deterministic and useful.
  - Default summary output to rank 0, with opt-in all-rank diagnostics.
  - Attach rank/device to structured records and avoid interleaved partial lines.
  - Ensure consolidation does not change residual indices, final status, or history length.
- [ ] Decouple telemetry controls that are currently entangled (`monitor_residual`, `store_res_history`, `print_solve_stats`, and convergence monitoring). Validate incompatible configurations with an actionable error naming the scope and accepted alternatives.
- [ ] Provide machine-readable solve results through the C API: status/reason, iteration count, initial/final residuals, residual kind, timing fields, and optional history. Python and other wrappers should not have to parse formatted tables.
- [ ] Add `verbosity_level=0` silence tests and callback-capture tests for initialization, configuration errors, setup, solve, nested solvers, and finalization.

## P1 — repeated-system and preconditioner reuse

- [ ] Define an explicit fixed-operator reuse contract for the C API.
  - Document the supported lifecycle: create resources/config/solver, upload and set up a matrix once, solve any number of right-hand sides, then destroy.
  - Specify which matrix/configuration changes preserve the hierarchy, which require `AMGX_solver_resetup`, and which require a complete rebuild. Cover values-only replacement, sparsity changes, block dimensions, scaling, precision, device/resource changes, and nested preconditioner parameters.
  - Make stale hierarchy use detectable: track matrix/configuration revisions or return an error when coefficients or structure changed without the required resetup instead of relying entirely on caller discipline.
- [ ] Expose setup/reuse state and per-call statistics through a machine-readable C API.
  - Report setup generation, matrix generation, `rebuilt`/`structure_reused`/`hierarchy_reused`, setup time charged to this call, retained setup time, solve time, allocations, and iteration/status data.
  - Keep the corrected text timing output consistent with these fields: a retained setup may be shown for context but must never be charged repeatedly to `Total Time`.
- [ ] Audit and optimize the warm repeated-RHS path.
  - Retain Krylov vectors, preconditioner workspaces, communication buffers, descriptors, and analysis objects whenever their compatibility key is unchanged.
  - Identify and remove avoidable per-solve allocations, descriptor reconstruction, coloring/analysis, device synchronization, and host-side setup. Preserve a bounded-memory mode for applications that prefer lower residency.
  - Add profiler-backed benchmarks separating cold setup, first solve, and steady-state solves for PCGF/PCG, BiCGStab/PBiCGStab, GMRES/FGMRES, and representative AMG/ILU/DILU preconditioners.
- [ ] Add efficient multiple-right-hand-side support.
  - Provide a batched or block solve API with explicit layout, stride, initial-guess, status, residual, and completion semantics so clients do not need one C/Python transition and upload/download cycle per vector.
  - Reuse the hierarchy and compatible workspace across the batch; evaluate block Krylov algorithms separately from a convenience batch of independent solves.
- [ ] Add reuse correctness and performance tests.
  - Run hundreds of right-hand sides after one setup and assert no setup generation change, no repeated hierarchy construction, stable memory after warmup, correct per-call timing, and independently checked residuals.
  - Cover coefficient-only resetup, structure changes, failure recovery, zero/nonzero guesses, multiple devices, MPI, and concurrent independent solver objects before documenting thread-safety guarantees.

## P1 — external GPU array and stream interoperability

- [ ] Specify the memory contract for every C API accepting `void *` data.
  - Document accepted host, pinned-host, CUDA device, and managed pointers; direction, byte count, alignment, supported value/index types, and whether data is copied or borrowed.
  - State explicitly that current `AMGX_matrix_upload_all`, `AMGX_vector_upload`, and download paths use `cudaMemcpyDefault` into or out of AMGX-owned storage. A CuPy device pointer therefore enables a direct device-to-device copy, not zero-copy sharing.
  - Validate pointers with CUDA pointer attributes where possible and return an actionable error for wrong-device pointers, inaccessible peer memory, unsupported memory kinds, null nonempty buffers, and overlapping output.
- [ ] Design backward-compatible stream-aware transfer APIs instead of overloading the synchronization behavior of existing calls.
  - Add async upload/download variants that accept an external CUDA stream or a resource-level stream association, plus an explicit completion event/query contract.
  - Preserve the current synchronous APIs and define exactly what is complete when each function returns.
  - Avoid `cudaDeviceSynchronize()` and implicit legacy-default-stream ordering in interoperability paths; use event dependencies between producer, AMGX, and consumer streams.
  - Represent stream/event handles in the C ABI without exposing C++ types, and define ownership, device affinity, lifetime, error propagation, and thread safety.
- [ ] Determine whether solve/setup can safely run on a caller-provided stream.
  - Audit kernels, Thrust/CUBLAS/CUSPARSE calls, memory pools, reductions, nested solvers, and distributed communication for hard-coded/default streams and device-wide synchronization.
  - If full stream injection is not yet possible, expose honest synchronization boundaries and a completion event; do not advertise asynchronous solves prematurely.
  - Add single-stream correctness first, then non-default/per-thread-default streams, concurrent independent solvers, and CUDA graph capture only after all dependencies are explicit.
- [ ] Add device and resource introspection APIs needed by wrappers: resource device IDs, current execution stream or completion event, pointer-memory diagnostics, and supported precision/index capabilities.
- [ ] Define local CSR interoperability rules for GPU producers such as `cupyx.scipy.sparse.csr_matrix` and `csr_array`.
  - Document 32-bit versus 64-bit index support, sorted/canonical requirements, duplicate-entry behavior, block layout, empty matrices, and diagonal handling.
  - Never silently narrow 64-bit indices; return a range-checked error or provide a documented conversion API.
  - Do not mutate producer buffers while canonicalizing or sorting.
- [ ] Add C API tests using device allocations populated on a non-default stream.
  - Record a producer event, transfer into AMGX, solve, transfer out on a consumer stream, and verify the result without a host/device-wide synchronize.
  - Test legacy stream (`1`), per-thread default stream (`2`), ordinary stream handles, multiple devices, peer-access on/off, managed memory, zero-size buffers, and lifetime until completion.
  - Profile the path to prove device inputs do not bounce through host memory and to quantify synchronization/copy costs.
- [ ] Keep the AMGX layer independent of a CuPy build dependency. CuPy-specific adaptation belongs in pyamgx; AMGX supplies a correct CUDA C ABI that any CUDA Array Interface or CUDA Stream Protocol consumer can use.

## P1 — error messages and supportability

- [ ] Standardize errors as: operation, solver/scope, relevant parameter/value, rank/device, underlying AMGX/CUDA/MPI status, and a concrete corrective hint when known.
- [ ] Preserve the first/root failure while adding context as it crosses solver, C API, and wrapper boundaries; avoid duplicated banners and diagnostics without new information.
- [ ] Add build/runtime version data (AMGX commit/version, CUDA runtime/driver, MPI, compile options, GPU) through a query API so bug reports can attach it without scraping startup text.
- [ ] Review open issue reports quarterly and link each accepted bug to a minimal reproducer and regression test. Keep build compatibility such as [issue #351](https://github.com/NVIDIA/AMGX/issues/351) in the general maintenance backlog while this roadmap tracks its diagnostic quality.

## P2 — documentation and release communication

- [ ] Document residual definitions, convergence modes, status meanings, iteration numbering, logging controls, callback guarantees, MPI rank behavior, and performance costs in one canonical diagnostics guide.
- [ ] Add small C and MPI examples that consume structured events and residual history without parsing console output.
- [ ] Maintain a “known issues and behavioral changes” section in `CHANGELOG`, including corrected historical residual output and any iteration-history compatibility impact.
- [ ] Coordinate an AMGX/pyamgx compatibility matrix and cross-repository release notes whenever the C API, status values, logging events, CUDA requirements, or residual semantics change.
- [ ] Maintain a three-repository interoperability note identifying the AMGX C API/ABI, pyamgx binding revision, tested CuPy releases/main commit, CUDA runtime/driver, supported CUDA Array Interface version, and stream semantics.

## Definition of done

- Every solver’s published per-iteration residual agrees with its documented residual definition and a reference calculation.
- Printed output, callbacks, stored history, C API queries, convergence decisions, and final summaries agree on iteration, residual, and stop reason.
- Diagnostics remain deterministic and attributable under multiple GPUs/ranks, and callback-disabled runs pay no material logging overhead.
- Device-array transfers have explicit copy and completion semantics, work correctly across non-default streams and devices, and never rely on accidental global/default-stream synchronization.
- Each fixed external or locally discovered bug has a permanent automated regression and a changelog entry.
