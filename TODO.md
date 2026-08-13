# AMGX communication and logging roadmap

This roadmap focuses on trustworthy solver telemetry, actionable diagnostics, and stable communication contracts for C/C++, Python, MPI, CuPy, and other consumers. A log line, stored residual, convergence decision, and C API query must describe the same solver state. GPU interoperability work is coordinated against the local `../pyamgx` and `../cupy` repositories rather than inferred from raw pointers alone.

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
