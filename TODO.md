# AMGX communication and logging roadmap

## P0/P1 — findings from the matK native-configuration search (2026-10-30)

Evidence below comes from AMGX 2.5.0 built on 2026-10-26 with CUDA 13.1,
FP64 `dDDI`, and H200 NVL. The real nonsymmetric matK system has 1,600,400
unknowns and 633,564,800 stored entries. Full-system convergence is **not**
established. Synthetic smoke passes establish compatibility only. Preserve the
distinction between a source defect, an observed runtime failure, a numerical
limitation, and a configuration mistake when resolving these items.

The complete source audits, candidate coverage, and reproduction details are in
native AMG findings (`docs/matk_native_amg_findings.txt`, local, untracked) and
native Krylov findings (`docs/matk_native_krylov_findings.txt`, local, untracked). Their frozen
configuration dictionaries are in
AMG proposals (`docs/matk_native_amg_proposals.json`, local, untracked) and
Krylov proposals (`docs/matk_native_krylov_proposals.json`, local, untracked). Per-worker benchmark
reports remain authoritative for the parameters and results of an actual run.
Follow-up configurations with deeper hierarchies and native GPU coarse LU are
in coarse LU proposals (`docs/matk_native_coarse_lu_proposals.json`, local, untracked).

- [ ] **Honor zero-start state for reused Krylov-child and AMG coarse vectors.**
  - `src/solvers/fgmres_solver.cu:484` calls the preconditioner with
    `solve(V(m), Z(m), true)` without clearing `Z(m)`. The subspace allocator
    retains those vectors across restarts and repeated solves. The base
    `Solver::solve` zero-guess branch (`src/solvers/solver.cu:869-872`) copies
    the RHS into residual storage but does not clear the solution vector.
    Native unpreconditioned BiCGSTAB updates that solution additively
    (`src/solvers/bicgstab_solver.cu:128,164`). A reused nonzero correction can
    therefore be included despite the zero-guess residual initialization.
  - This is a source-confirmed state-contract mismatch; no standalone GPU
    reproducer of the nested combination has been run. Test FGMRES with a
    native BiCGSTAB child across at least two restart cycles and repeated
    solves, compare fresh versus deliberately nonzero reused correction storage,
    and independently check each child equation and the final original residual.
    Do not generalize this finding to child solvers that explicitly overwrite
    their output. The scalar follow-up records the audit and prior trial coverage.
  - **Additional GMRES/FGMRES mechanism:** both ignore `xIsZero` in their
    `solve_init`, and their first Arnoldi residual explicitly uses the retained
    solution (`gmres_solver.cu:240-258,321-323`,
    `fgmres_solver.cu:350-358,422-426`). With a dirty child output they therefore
    warm-start despite the caller's zero-start flag; the base initial norm
    still assumes residual `b`. This differs from BiCGSTAB's additive update
    from a residual initialized independently of stale `x`. A converged
    GMRES/FGMRES child can still solve its equation correctly; finite inner
    state dependence does not alone prove invalid flexible-GMRES algebra or
    explain a full matK failure.
  - Parent GMRES also passes its retained `m_Z_vector` at lines 271/350, and
    its final right-preconditioner call at line 452 uses an already nonzero
    Arnoldi vector `m_V_vectors[0]` as output. This latter path is exposed on
    the first cycle, before any restart. The fill at line 435 clears the RHS
    accumulation buffer, not that output. FGMRES retains same-sized `Z`
    buffers across restarts/repeated solves (`fgmres_solver.cu:96-145,179,284`).
  - AMG ordinary init pre-smoothing already physically clears its output in
    `src/cycles/fixed_cycle.cu:80-86`. Terminal coarse dispatch bypasses it:
    `include/amg_level.h:234-236` calls `launchCoarseSolver`, which passes
    `true` with persistent `xc` at `src/amg_level.cu:138-140` without a clear.
    Audit repeated coarse GMRES/FGMRES solves with changing RHS; do not
    generalize the defect to protected pre-smoothing, intentional nonzero
    post-smoothing, or children that overwrite their output.
  - Clear before initial convergence, not only in derived `solve_init`:
    `Solver::solve:1001-1012` can skip initialization after treating `b=0` as
    converged, leaving stale nonzero `x` unchanged. Include that edge, false
    flags preserving intentional warm starts, parent GMRES's final child call,
    multiple FGMRES restarts and repeated AMG terminal solves in qualification.
    This remains source-confirmed; a standalone native GPU reproduction is
    pending. The observed coarse growth near `1e69` in a matK hierarchy run of
    2026-10-30 is not attribution. See the independent state audit (`docs/matk_nested_zero_start_audit.txt`, local, untracked).
  - A reviewed configuration-only isolation uses `coarse_solver=NOSOLVER`
    plus GMRES/L1 smoothing through the already-clearing `FixedCycle`, with
    construction `max_iters=restart=200`, fine pre/post 30 and coarsest 500.
    Construct enough Krylov storage before mutable sweep limits are applied;
    `set_max_iters` does not resize it. FixedCycle forces tolerance zero, so
    this changes work versus the old coarse tolerance 0.01 and is not a
    numerical-equivalence claim. The unqualified frozen proposal is in
    native zero-start proposals (`docs/matk_native_zero_start_proposals.json`, local, untracked).
    The downstream benchmark's `validate_krylov` currently overwrites explicit
    pre/post counts with the smoother construction limit; the reviewed
    200-allocation/30-effective probe requires a narrow preservation override
    and assertions on its generated config. Keep that specification metadata
    outside the native AMGX config and retain the old default for prior trials.

The independent matrix/RHS import audit (`docs/matk_import_audit.txt`, local, untracked) found no
defect in the actual matK import, including full coefficient/index/RHS checks.
Its source review also notes the existing binding validation gap for strided
uploads, already tracked in [pyamgx's buffer-descriptor TODO](../pyamgx/TODO.md).
The current matK arrays are verified contiguous; that gap does not explain these
runs. Exporter intent and an independent device round-trip remain outside the
CPU audit's evidence.

- [ ] **Wire or reject the registered LOCALLY_DOWNWIND coloring scheme.**
  - `src/matrix_coloring/locally_downwind.cu:377` implements ordinary
    `colorMatrix` by printing `coloring denied` and returning without colors.
    The normal matrix setup then calls `createColorArrays`, which expects one
    color per row. The specialized `colorMatrixUsingAggregates` exists, but
    source search finds no setup call site invoking the matrix-level aggregate
    hook. Registration alone does not make this usable for native ILU.
  - That specialized implementation currently defines `CPU_VERSION`, copies
    coefficients/structure to the host, and has no active distance-two check;
    the GPU one-ring-only guard is under the inactive `#else`. Qualify both
    dispatch and coloring validity before exposing it as a supported option.
    This is a source finding; no GPU failure is asserted. See
    reordering audit (`docs/matk_reordering_findings.txt`, local, untracked).

- [ ] **Clear IDR's host initialization buffer on every solve.**
  - `src/solvers/idr_solver.cu:134-158` clears the device work arrays but not
    `h_chk`. Device `setup_arrays` at521-549 writes only the diagonal of that
    host buffer before copying its first s*s entries into M, then overwrites
    the buffer with random shadow-space coefficients for s>1. The second solve
    therefore initializes M with stale nonzero off-diagonals instead of I.
    IDRMSYNC explicitly clears its host buffer before the same construction.
  - Source-confirmed repeat-initialization defect; GPU reproduction pending.
    Test repeated solves with multiple RHS/initial guesses, including IDR as
    a repeatedly applied preconditioner. No current matK result is attributed
    to this defect. See `docs/matk_scalar_krylov_followup.txt`.

- [ ] **Give IDR/IDRMSYNC and the base solver one consistent iteration counter.**
  - IDR (`idr_solver.cu:257`) and IDRMSYNC (`idrmsync_solver.cu:332`) increment
    the base `m_curr_iter` inside their s-loops; `Solver::solve` increments it
    again. `is_last_iter` checks equality only, so some limits can be skipped.
    For s=4,max_iters=6, a nonconverging control-flow trace reaches history
    index10 although only7 entries were allocated. In release builds the guard
    at `solver.cu:1116-1117` is an inactive assert before an unchecked write.
  - This is a source/control-flow finding, not a GPU reproduction. Separate
    internal updates from outer iteration accounting, make limits robust and
    guard history bounds. Qualify awkward limits and early-convergence paths;
    aligned iteration limits are only a temporary test restriction.

- [ ] **Confirmed launch-size issue: qualify or reject 20x20 GS/Block Jacobi before launching.**
  - `MulticolorGaussSeidelSolver::computeDinv_bxb` in
    `src/solvers/multicolor_gauss_seidel_solver.cu` and
    `BlockJacobiSolver::computeDinv_Big` in `src/solvers/block_jacobi_solver.cu`
    request `sizeof(double) * 20 * 20 * (512 / 16) = 102400` bytes of dynamic
    shared memory. These call sites set cache preference but do not opt into the
    required larger dynamic-shared-memory limit with `cudaFuncSetAttribute`.
  - A matK probe run of 2026-10-30: all four 20x20 GS hierarchy probes fail with CUDA `invalid argument`
    at GS line 759; the Block Jacobi probe fails at line 1261. Small smoke inputs
    are sufficient; this is not merely nonconvergence on matK.
  - Query device/kernel limits, reduce the launch footprint or opt in where
    supported, and reject unsupported sizes with an actionable diagnostic.
    Qualify the numerical kernels independently after fixing the launch.

- [ ] **Confirmed index-width hazard: widen DenseLU BSR-to-dense addressing.**
  - `csr_to_dense_kernel` in `src/solvers/dense_lu_solver.cu` indexes
    `A_dense[dense_col * lda + dense_row]` using signed `int` operands, including
    the separately stored diagonal path. A square dense matrix above 46,340
    scalar rows can overflow that multiplication although its allocation uses
    `size_t` and can fit on H200. Cast before multiplication and audit all related
    dense indexing, products, and workspace sizes.
  - **Runtime failure, root cause unproven:** in a full-matrix run of 2026-10-30,
    `fgmres_native_extended_i20k_w_b10` and `fgmres_native_dense8_20k_w_b10`
    fail with CUDA illegal memory access during DenseLU setup, first surfaced in
    `DenseLUSolver::cudense_getrf` at line 548. Do not assert index overflow caused
    these particular failures until the actual coarse dimensions and failing
    kernel are captured. Reproduce under Compute Sanitizer with synchronization
    after conversion and factorization separately.

- [ ] **Audit DenseLU coarse-size thresholds and allocation guards.**
  - In `src/amg.cu`, `AMG::setup` converts `dense_lu_num_rows` from scalar DOFs to
    block rows, while the setup loop compares `dense_lu_max_rows` directly with
    block-row counts. Make the public units explicit and consistent.
  - The setup loop rejects a proposed next level below `min_rows` and retains
    its larger parent. Consequently the actual dense coarse matrix can exceed
    the trigger substantially; the early-break path also needs a cap audit.
  - Validate the *actual scalar dimensions and byte requirement* immediately
    before dense allocation/factorization, report them in setup errors, and test
    a large coarsening jump, a block-size transition, and the configured cap.

- [x] **Fix CLASSICAL interpolation truncation's shared-memory overrun.**
  - **Qualified and merged, 2026-10-31:** the truncation change of that date
    bounds the fallback launch to 128/64/32 threads for caps 1–8/9–16/17–32.
    The H200 qualification passed 191 checks plus all three `TruncateMaxElements`
    native real modes (828 fixture comparisons). The gate includes 24 clean
    memchecks, 15 clean syncchecks, all caps 1–32, eight reproduced baseline
    faults and 24 bitwise safe-cap comparisons. See the
    [qualification record](docs/truncation_qualification.md) for exact identities
    and coverage. The following paragraphs preserve the historical defect.
  - The original `truncate_kernel` in `src/truncate.cu:388–455` allocates
    `s_ind[1024]` and `s_val[1024]`, then uses
    `threadIdx.x * max_elmts + j` for rows longer than `max_elmts`.
    `truncateByMaxElements` launched 128 threads (`:810`, `:847`), but accepted
    limits up to 32 (`:795–797`). Limit 32 needs 4096 elements per shared array:
    thread 32 already starts at index 1024; limit 16 also overruns from thread
    64 onward. A positive limit at most 8 fits this specific 128-thread launch.
  - Short rows use the direct-copy branch and hide the defect in small smoke
    matrices. The historical baseline library contained the compiled
    `amgx::truncate_kernel<int,double>` symbol; neither source nor build flags
    enabled `USE_EXPERIMENTAL_NLARGEST`. This was a source-confirmed reachable
    out-of-bounds write, independent of the separate matrix value-reordering bug.
  - Four full reordered runs of 2026-10-30, two mixed pairwise/classical (cap 32)
    and two CLASSICAL MULTIPASS size probes (corrected cap 32), all passed
    512-block smoke, then failed setup with an initial Thrust
    `parallel_for ... cudaErrorIllegalAddress`. Later
    `global_thread_handle.cu:1247`/cleanup errors do not identify the first
    kernel. The affected CLASSICAL truncation path is a strong candidate cause,
    not yet a sanitizer attribution of these particular failures. Per-candidate
    logs are in the local records of the Hybridge campaign `matk`.
  - The applied correction changes only launch geometry; selection, stable
    ties, CSR output and row rescaling retain their existing operations.
    The H200 direct-operator qualification does not establish full-matK
    convergence or attribute the earlier hierarchy failures to this kernel.
    Host/complex support, nonpositive-cap and empty-matrix semantics, the
    experimental preprocessing branch and Quadro/Blackwell qualification
    remain outside this fix's evidence. Historical configuration-only
    isolations used an explicitly named `interp8` variant or
    `interp_max_elements=-1` to skip truncation, with fill/memory checks for
    the latter. Preserve failed cap-32 specs as evidence; do not claim their
    advertised support gate establishes memory safety. See
    reordered preconditioner audit (`docs/matk_pardiso_next_preconditioners.txt`, local, untracked).
    The original launch-geometry proposal and qualification plan remain in
    fix design (`docs/matk_truncation_launch_fix_design.txt`, local, untracked),
    patch (`docs/matk_truncation_launch_fix.patch`, local, untracked) and
    independent review (`docs/matk_truncation_launch_fix_review.txt`, local, untracked).

- [ ] **Reject exhausted LOW_DEG hash-sparsity retries before consuming output.**
  - `compute_sparsity_dispatch` in
    `src/aggregation/coarseAgenerators/low_deg_coarse_A_generator.cu:858–920`
    retries at most 10 times, sets `done` from the device status, then returns
    without checking that the final attempt succeeded. In count mode the
    kernel may return early on a failed hash/status (`:80–87`), leaving row
    counts incomplete; `computeAOperator` subsequently scans those counts and
    allocates/fills the coarse matrix (`:1202–1225`). Propagate exhaustion
    explicitly rather than treating partial output as a valid CSR pattern.
  - This is a separate source-confirmed missing failure guard. The three
    warning pairs in the two mixed pairwise/classical runs originate at `:865–866` when a dispatch
    first expands its hash storage, but do not prove that any dispatch reached
    10 failed attempts or identify a level. Do not attribute the observed
    illegal access to this guard without a failing-status/attempt trace.

- [ ] **Wire or reject inactive AMG tuning parameters.**
  - `max_coarse_iters` is registered in `src/core.cu:452` and declared in
    `include/amg.h:174`, but no setup or solve path reads it. The iterative
    coarse solver uses its own scoped `max_iters`; `launchCoarseSolver` in
    `src/amg_level.cu:133` does not override that limit. Changing the advertised
    AMG parameter therefore has no effect in this source tree.
  - CLASSICAL reads `interp_truncation_factor` into `trunc_factor` at
    `src/classical/classical_amg_level.cu:452` but never applies the member.
    Interpolation construction at lines 827–834 only invokes
    `truncateByMaxElements` for `interp_max_elements`; the separately implemented
    `truncateByFactor` has no CLASSICAL call site. Do not claim that changing
    the factor changes interpolation strength until it is wired and tested.
  - These are source-confirmed configuration no-ops, not a diagnosis of a
    particular matK failure. Add config-effect tests or an unsupported-option
    diagnostic. Evidence and practical alternatives are in
    next reordered preconditioners (`docs/matk_pardiso_next_preconditioners.txt`, local, untracked).

- [ ] **Fix source-confirmed partial normalization in native BSR GMRES.**
  - Synthetic multilevel smoke runs of 2026-10-30 exposed divergent block-GMRES
    results where scalar GMRES and block FGMRES passed. Isolate a coefficient-
    identical CSR/BSR pair and independently check `b-A*x`, block dimensions,
    vector reduction lengths, restart, and zero/nonzero guesses.
  - Both `solve_iteration` and `solve_one_iteration` in
    `src/solvers/gmres_solver.cu` normalize the first basis vector using raw
    `Cublas::scal(size, ..., vector.raw()+offset, 1)` (around lines 259/338).
    `Matrix::getOffsetAndSizeForView` returns block-row counts/offsets, whereas
    the normal `scal` wrapper in `src/blas.cu` multiplies by the vector block
    size. The raw calls therefore normalize only part of a BSR vector and use
    the wrong nonzero block offset. Use block-aware operations or explicitly
    convert both the count and offset to scalar components. Scalar CSR is
    unaffected. Add a regression covering both iteration paths.
  - Extend the existing residual-correctness roadmap to FGMRES loss of
    orthogonality, restart drift and happy breakdown. The current
    `src/solvers/fgmres_solver.cu` performs one MGS pass and normalizes by the next
    Hessenberg entry without the reliable-residual/reorthogonalization policy
    present in `gmres_solver.cu`. Shared configuration flags currently do not
    enable those features in FGMRES; implement them or diagnose unsupported use.

- [ ] **Fix source-confirmed BINORMALIZATION exit and NBINORMALIZATION factor inconsistency.**
  - Native `BINORMALIZATION`/`NBINORMALIZATION` probes (2026-10-30) were run with
    host scaling disabled. Some scalar ILU probes exited with code zero before
    producing a terminal solve result, with memory-pool/leak diagnostics; an
    `NBINORMALIZATION` smoke reported native success but original relative
    residual approximately 4745.
  - `src/scalers/binormalization.cu` unconditionally calls `exit(0)` at the end
    of device scaling (around line 394), explaining the successful process exit
    without a solver result. Implement the path or return a typed unsupported
    error; library code must not terminate the process successfully.
  - `src/scalers/nbinormalization.cu` divides the matrix by normalization `C`,
    but multiplies the stored left/right **squared** scale factors by
    `sqrt(1/C)` (around lines 494-498). Matrix/vector application takes square
    roots of those factors, so each effective factor gains only `C^(-1/4)`.
    The stored factors should each gain `1/C` to match the matrix's total `1/C`.
    As written, the first unscale restores `A/sqrt(C)` and the reconstructed
    solution can satisfy `A*x=sqrt(C)*b` despite native success. Preserve an
    independently checked matrix round-trip and solution-scaling regression.
  - Capture the exact stored config and test matrix, inspect RHS/solution
    scaling and unscaling, repeated solves, zero initial guesses, teardown and
    error propagation. Never accept a zero process exit or native success as
    proof of the original-system residual target.

- [ ] **Investigate legacy ILU(0) setup on native coarse RAP matrices.**
  - 20x20 classical hierarchy smoke probes (2026-10-30) fail cuSPARSE structural
    analysis at block row 3 despite standalone fine-matrix legacy ILU working.
    Native aggregation coarse ILU probes also failed (2026-10-30).
  - New full scalar-CSR evidence after the PARDISO permutation: two ordering-ladder
    runs of 2026-10-30 both fail `fgmres_classical_legacy_ilu0_sweep4_csr` during setup with
    earliest error `CUSPARSE_STATUS_INTERNAL_ERROR` at
    `multicolor_ilu_solver.cu:217`, in `LegacyCusparseBsrIlu::setup_and_factor`.
    The first run's `fgmres_classical_legacy_ilu0_krylov30_csr` fails the same way.
    Later illegal-instruction errors occur during cleanup and must not replace
    that first error in the diagnosis. No hierarchy dimensions were printed,
    so the failing level/matrix is not yet localized. Exact per-candidate logs
    are in the local records of the Hybridge campaign `matk`.
  - Dump a minimal failing coarse matrix, verify sorted/canonical column indices,
    internal diagonal presence, dimensions, color permutations, and the values
    passed to cuSPARSE. These are hypotheses to test, not diagnosed causes.
    A genuine missing diagonal must be distinguished from invalid storage or a
    valid matrix rejected because the backend's input contract was not met.

- [ ] **Improve numerical-breakdown and interpolation diagnostics on nonsymmetric systems.**
  - Three full-matrix runs of 2026-10-30 produced NaNs on the first raw-DILU AMG application.
    Detect and return a defined breakdown/divergence status with solver scope,
    level, row/pivot and residual context; do not rely on an external log parser
    to terminate the worker. Divergence alone is not evidence of a kernel bug.
  - A full 20x20 Extended+i run (2026-10-30) fails the block constant-mode constraint;
    Jacobi interpolation reports a singular diagonal block. A 10x10
    right-normalized transfer run fails the constant-mode check. These extend the
    existing rejected-interpolation findings below to a nonsymmetric matrix.
    Report failing level/block row, smallest pivot, and constraint error.
    Rejecting a singular local block is valid; investigate whether nonfinite
    coarse coefficients or an incorrectly imposed constraint caused rejection.

- [ ] **Document effective Krylov smoothing lengths and expose them in diagnostics.**
  - This is a configuration/usability finding, **not a demonstrated numerical
    kernel bug**: `FixedCycle::cycle` overwrites a smoother's `max_iters` with
    `presweeps`/`postsweeps` (or `finest_sweeps`) and sets its tolerance to zero.
    Thus `smoother={FGMRES,max_iters:30}` with one pre/post sweep runs one
    Krylov iteration, not one 30-step smoothing application.
  - The downstream benchmark generator was corrected at 15:45 on 2026-10-30;
    earlier one-to-three-step runs must not be interpreted as failed 30/50-step
    trials. The 10x10 run's native log confirms effective 30-iteration calls.
    Add explicit effective limits to config output and a nested-Krylov regression.

- [ ] **Make ILU support and row-workspace limits discoverable before factorization.**
  - Native factorization supports 4x4 blocks and ILU(0/1); full matK ILU(1)
    exceeds the fixed per-row shared-memory workspace despite small smoke passes.
    This is a support limit, not proof of incorrect factorization.
  - Include the failing row, actual degree/fill, allowed limit, supported block
    sizes and backend in errors; validate before partially constructing factors.
    The opt-in `cusparse_legacy` backend supports ILU(0), not higher fill levels.
  - **Isolated extension in progress:** work on raising native ILU capacity
    started on 2026-10-30. `multicolor_ilu_solver.cu` now measures the
    filled LU row degree before factorization and selects the original 128 or
    a 1024-column kernel. The existing library is unchanged; only this object
    is replaced in an isolated build. Boundary, precision/layout,
    and sanitizer tests are in `../hybridge/campaigns/matk/check_ilu_capacity.py`.
    This does not yet establish full-matK convergence or qualify every API path.

- [ ] **Propagate the requested BSR SpMV backend from GMRES and FGMRES.**
  - `Solver::configure_bsr_spmv_backend` is called by BICGSTAB/PBICGSTAB, but
    GMRES/FGMRES do not call it during setup. With a native ILU preconditioner
    that also does not set the matrix auxiliary flag, a configured
    `bsr_spmv_backend=cusparse_generic` can leave the actual backend at legacy.
    Some other preconditioners/hierarchy paths set the flag independently.
  - Make backend selection consistent and report the actual selected backend.
    Test every Krylov family, preconditioner nesting, setup/resetup and default
    behavior. Configuration JSON alone is not evidence that cuSPARSE was used.

- [ ] **Remove FP32 truncation from the legacy FP64 column-major BSR path.**
  - `util<double>::to_uptype` in `include/amgx_types/util.h` takes a `const float&`
    rather than a double. Legacy column-major 4x4 SpMV calls this conversion in
    `src/multiply.cu` (around lines 399, 714 and 749), rounding FP64 inputs to
    FP32 before returning double.
  - In the capacity regression of 2026-10-30, small column-major FP64 cases show physical
    residuals around 2.7e-8 despite native residual estimates near 1e-15. The
    128-row case fails identically with the original and isolated libraries;
    this issue predates the ILU capacity extension.
  - **Separate high-degree failure resolved in an isolated build:** the FP64
    1024-column star test originally had physical residual around 0.08--0.11
    despite native success and clean memcheck. Verified generic dispatch and
    baseline same-matrix diagnostics ruled out the capacity extension as a
    necessary cause. The second isolated build fixes both layouts to about 2.6e-16 by correcting
    the row-length reduction below. This is separate from the legacy FP32
    truncation, which the generic path avoids without fixing that conversion.

- [x] **Fix the row-length reduction used by in-place column reordering.**
  - The original `src/matrix.cu:22-30` computed a CSR row length as
    `*(&x + 1) - x` inside a unary Thrust reduction functor. Its argument need
    not reference the original CSR offsets: CUDA 13.1 CUB passes local loaded
    values or temporaries. The working-tree correction uses explicit
    `row_offsets[row + 1] - row_offsets[row]` with a counting iterator.
  - The result sizes and strides each CTA's scratch region in
    `reorderElementsDeviceCSR` (around line 1209). A wrong maximum can make
    scratch regions overlap inside a valid allocation; a clean memcheck does
    not exclude that failure. Compare the computed maximum with a known host
    maximum and test skewed row lengths in both precisions/layouts.
  - A diagnostic run of 2026-10-30 reproduces a
    wrong operator with the ORIGINAL library after a rejected high-degree
    ILU setup, then solves using NOSOLVER on the same matrix. The exact known
    initial solution produces native residual norm 48.4006869, identical to
    the isolated ILU build, while DILU gives 7.78e-14. This proves the capacity
    extension is unnecessary for that defect.
  - **Tested repair:** the second isolated build preserves the original ILU
    capacity change and adds only this matrix-source correction. Its
    qualification run of 2026-10-30 passes all 20 qualification cases (16 independent numerical checks and
    four expected 1025-column rejections), including FP64 star1024 ROW/COL
    physical residuals 2.63e-16/2.62e-16. Both corresponding memchecks pass
    with zero errors. Baseline and first isolated libraries remain unchanged.
    This controlled comparison attributes the tested high-degree failure to
    the row-length preprocessing defect; the exact scratch-overlap event was
    not instrumented. It does not qualify every API path or full-matK convergence.
    Details and separate loop-barrier reasoning:
    native Krylov findings (`docs/matk_native_krylov_findings.txt`, local, untracked).

- [ ] **Guard diagonal-symmetric scaling inputs and release its temporary flag.**
  - `src/scalers/diagonal_symmetric.cu` checks for negative diagonals but does
    not reject zero/missing diagonals before reciprocal square roots. Audit
    finite positive diagonals and external-diagonal/block input support.
  - The device setup allocates `d_positive` and does not free that temporary
    flag. These are source observations; no matK runtime failure was assigned
    to this scaler, which has not been promoted for the indefinite matrix.

- [ ] **Confirmed source dispatch mismatch: GPU `strength=ALL` uses AHAT tests.**
  - `Strength_All::strongly_connected` in `include/classical/strength/all.h`
    returns true, but both GPU kernels reached from
    `Strength_Base::computeStrongConnectionsAndWeights_1x1` hard-code
    `stronglyConnectedAHat` (`src/classical/strength/strength_base.cu`, around
    lines 320 and 432). Switching `use_opt_kernels` does not select an ALL
    predicate; host/device strength semantics need to agree.
  - Add a mixed-sign scalar matrix regression for ALL versus AHAT and confirm
    the generated strength masks differ as specified. Do not count ALL as an
    independent sign-agnostic GPU configuration until this dispatch is fixed.

- [ ] **Audit scalar lengths/strides in BSR IDR and IDRMSYNC.**
  - Both implementations allocate subspace storage with scalar length
    `num_cols * block_dimy` in setup, but reset `N` to `A.get_num_rows()` in
    iteration (`src/solvers/idr_solver.cu`, around lines 56/179;
    `src/solvers/idrmsync_solver.cu`, around lines 56/213/266).
    Specifically, IDR's `gemv_extnd` calls around lines 197/216 pass block-row
    `m`, `lda` and `offsetA=k*N` to a wrapper that forwards them directly to
    CUBLAS (`src/blas.cu`, around lines 404-424), inconsistent with scalar-sized
    subspace storage. Ordinary block-aware axpy/dot wrappers must not be blamed
    for this separate raw-matrix-stride defect. Audit IDRMSYNC's raw calls too.
    A coefficient-identical CSR/BSR regression is still required; this is not
    a matK runtime diagnosis.
  - **Additional scalar-path breakdown issue:** IDR's `dotc_div` returns zero
    both for a zero denominator and a legitimate zero numerator. Its caller
    treats every zero `alpha` as a zero-pivot failure (around lines 230-232).
    Preserve numerator/denominator status separately so an already orthogonal
    basis vector is not rejected. The omega path also calls `exit(1)` from the
    library (around line 323), and its zero-angle safeguard needs finite/zero
    denominator checks. These are source findings, not a full-matK IDR result.

- [ ] **Bound and diagnose the native randomized Kaczmarz path.**
  - `src/solvers/kaczmarz_solver.cu` hard-codes randomized mode (around line 498),
    silently disables requested multicolor mode during setup (around line 532),
    and sizes the inverse-CDF table using approximately
    `sum(row_norm_squared) / min(row_norm_squared)` (around lines 539-540).
    Audit zero rows, extreme row scaling, integer conversion and allocation
    bounds before attempting large ill-conditioned systems.
  - The active naive/randomized path launches one warp in one block (around
    lines 765-773), so it is not a scalable parallel fallback in this build.
    Expose the selected algorithm and supported execution mode explicitly.

- [ ] **Remove process-global AFFINITY setup state and propagate setup errors.**
  - `src/classical/strength/affinity.cu` stores `level_counter` and
    `is_random_initialized` as globals (around lines 41-42), changing the number
    of test vectors across calls/solver instances rather than keeping that
    history with the hierarchy. Audit independent resources, repeated setups,
    resetup and concurrency. `max_row_sum=-1` avoids adaptive vector-count growth
    in this implementation; it does not repair the ownership model.
  - The same setup path ignores the return value from
    `solver->setup_no_throw(m_aff, false)` (around line 442) and proceeds to batch
    smoothing. Propagate failure before accessing partially initialized state.
  - **Additional full-matrix runtime failure, not localized:** in a run of
    2026-10-30, `fgmres_affinity_multipass_gmres30_l1_csr` reports CUDA illegal memory access
    during setup, surfaced in `global_thread_handle.cu:1247`. The same candidate
    passed the multilevel smoke in an earlier run. Capture the failing kernel and
    intermediate dimensions/fill under Compute Sanitizer; do not attribute this
    to the global-state issue without evidence. In the same full run,
    scalar-guided BSR20 coarsening exhausted GPU memory; record that separately
    as a configuration resource limit unless allocation/size checking is faulty.

- [x] **Build valid color offsets for unused interior BFS colors.**
  - `run_createColorArrays_on_cpu` in
    `src/matrix_coloring/serial_greedy_bfs.cu:507-542` zero-initializes every
    offset and writes entries only for colors actually present. Unused
    interior colors therefore keep zero boundaries instead of the proper
    repeated prefix offset. The base `createColorArrays` preserves this
    prepopulated array rather than recomputing lower bounds.
  - This is reachable in the normal BFS algorithm: its 128-color windows
    choose offsets 1..127, so a 128-vertex clique gets colors 1..127,129.
    A CPU-only reproduction gives color 127 offsets `[126,0]`, a negative
    row count -126 and native ILU factor grid -30. Correct boundaries are
    `[126,127]` for color 127 and `[127,127]` for empty color 128.
  - A full BFS ILU1 setup run (2026-10-30) fails with CUDA `invalid argument` at the
    factor launch check after reporting filled degree 320/capacity 1024.
    Negative offsets are a specific plausible cause, but the full run's
    actual color/offset dump is not yet available; do not call that runtime
    attribution confirmed. All zero grids are already skipped for every color.
  - Construct boundaries with counts/prefix sums or lower bounds, validate
    monotonic offsets and the terminal row count, and test a clique of 128
    or more rows plus arbitrary color gaps. Keep this separate from the IDO
    incomplete-coloring defect. Source proof and CPU reproduction (`docs/matk_bfs_color_offset_findings.txt`, local, untracked).
  - **Tested isolated repair (2026-10-30):** the third isolated build
    uses a monotone cursor to fill every
    lower-bound boundary, including unused colors. All four failures
    reproduced with the previous build now pass: FP64 original residuals below 9e-16 in both layouts,
    FP32 below 3.3e-7. The baseline and two earlier libraries remain unchanged.

- [ ] **Replace the native ILU4 absolute determinant cutoff with a scale-aware policy.**
  - The active formula-2 block inverse in `include/solvers/block_common_solver.h`
    uses absolute determinant thresholds (1e-12 for FP64, 1e-7 for FP32) and
    silently substitutes diagonal reciprocals when the determinant is small.
    Uniformly rescaling an invertible, well-conditioned block can therefore
    discard its coupling, even though its condition number is unchanged.
  - A GPU one-application test of 2026-10-30 (`../hybridge/campaigns/matk/check_ilu_inverse.py`)
    verifies this independently of an outer solver. A nonsymmetric block with
    condition number about 2.36 passes at scale 1; at scale 1e-4 native FP64
    ROW/COL produce original relative residual 0.3628976. FP32 also fails at
    scale 1e-2. Legacy row-major ILU passes every tested scale/precision.
  - Preserve block coupling for safely invertible scaled blocks, use a relative
    pivot/singularity criterion, and expose fallback counts/reasons. Add a
    one-application oracle over scales and dense nonsymmetric blocks. Actual
    fallback frequency in matK is not yet measured; do not assign its poor
    convergence to this mechanism without evidence. Ordinary uniform right
    scaling is being tested as a configuration workaround, not a kernel repair.

- [x] **Synchronize native ILU shared-memory handoffs on tested real single-GPU paths.**
  - **2026-10-31 qualification:** the synchronization change of that date
    combines the reviewed setup and application corrections plus optional
    fallback counters. H200 qualification runs passed 28 capacity cases, 40 native oracle cases,
    40 synccheck and 32 zero-warning/error racecheck invocations, including
    diagnostics off/on, both layouts/precisions and grid-stride reuse.
    Numerical policy is unchanged. The
    [qualification record](docs/ilu_synchronization_qualification.md) bounds the
    tested paths; the deferred branches have a separate open item below.
    The following paragraphs preserve the earlier failure sequence.
  - Source and compiled PTX show cross-lane shared reads without matching warp
    barriers in A-to-LU mapping, active 4x4 factorization, and the formula-2
    inverse helper. The ILU1 symbolic pattern kernels have similar broadcasts.
    `volatile` accesses do not supply inter-thread synchronization.
  - A reviewed snapshot-only patch (`docs/matk_ilu_sync_fix.patch`, local, untracked) is compiled
    in a separate isolated build. It adds publication/reuse barriers
    and an ILU-local tile-synchronized inverse while retaining the same algebra
    and determinant fallback. Independent review (`docs/matk_ilu_sync_fix_review.txt`, local, untracked)
    found no source-level blocker, but that setup-only build did not complete
    the qualification gate. The later complete correction is qualified above.
  - The racecheck run of that one-application test reports 498 warnings (100 hazards displayed, zero
    errors) on the filled-129 FP64 case. Its final numerical check still passes
    at 9.48e-16; this is runtime evidence of hazards, not evidence that this
    particular answer or the full matK factors were corrupted. Memcheck-only
    success does not establish race freedom. These paths predate the capacity
    extension; the extension reuses them.
  - A follow-up racecheck run (2026-10-30) completed under the longer timeout with numerical
    residual 9.48e-16, but exited 99 with **919 racecheck warnings**. All 100
    displayed groups concern the unchanged native triangular application
    kernels: `LU_forward_4x4_kernel_warp` reuses shared values after reads
    499/503 without a completion barrier before the next store 477;
    `LU_backward_4x4_kernel_warp` has the same issue at 749/771/775 and before
    compressed-RHS stores 799/806. The nonzero-guess forward path also needs
    publication/reuse ordering around 405/431/435. Source and SASS agree;
    this is not established as a sanitizer false positive or as corruption
    of this passing answer. Use explicit 16-lane row masks, since neighboring
    rows can have different lengths and an odd final tile can be inactive.
    See application-stage findings (`docs/matk_ilu_application_sync_findings.txt`, local, untracked).
    A separate application patch artifact (`docs/matk_ilu_application_sync_fix.patch`, local, untracked)
    and mask/coverage design (`docs/matk_ilu_application_sync_fix_design.txt`, local, untracked)
    were preserved for later qualification. Both setup and application patches
    are applied in the synchronization change of 2026-10-31; the later passing evidence is listed above.
  - Add barriers at producer/consumer and workspace-reuse boundaries, using
    full-warp masks only at warp-uniform points and 16-lane masks inside inverse
    tiles. Audit other callers before changing a shared helper's contract.
    Require racecheck and synccheck as well as meaningful one-application
    numerical oracles. Details: kernel audit (`docs/matk_ilu_kernel_audit.txt`, local, untracked).

- [ ] **Complete deferred native ILU symbolic/failure-path qualification.**
  - The normal solver uses the 32-thread symbolic overload. Subwarp widths
    2/4/8/16 and deliberately induced hash-failure returns were source-reviewed
    and compiled, but not exercised by the GPU runs above. New qualification was
    stopped after those runs on 2026-10-31; a direct CSR
    fixture remains inactive deferred work. This is distinct from the separate
    retry-exhaustion guard and complex-mode issues below.

- [ ] **Widen or explicitly bound native ILU4 value-index arithmetic.**
  - The active kernel forms `16 * int_block_offset + lane` in signed 32-bit
    arithmetic before widening to an address. The compiled FP64 1024-capacity
    kernel preserves this behavior. Block offset 134217728 is the first unsafe
    one; a factor with more than 134217728 stored blocks can overflow indexing.
    Larger fill enabled by the extension can encounter this limit while still
    fitting on an H200. This defect predates the capacity change.
  - Report actual filled LU nnz/value counts and either widen every affected
    index calculation or reject unsafe sizes before mapping/factorization.
    matK's maximum filled row alone does not establish its total nnz; no claim
    is made that the current matK trials reach this threshold.
  - The new preflight's `(num_rows+127)/128` also overflows for rows near
    INT_MAX, as can its signed grid-stride increment. Use overflow-safe ceil
    division and a widened/guarded loop. This extreme case is far beyond matK;
    it is documented in the kernel audit, not a proposed cause of these runs.

- [ ] **Make native ILU4 relaxation independent of the zero-initial-guess hint.**
  - Source audit of `LU_backward_4x4_kernel_warp` finds that the xIsZero path
    reads already weighted upper-neighbor x values, while the other path uses
    unweighted Delta values. With relaxation_factor != 1, identical zero
    vectors can therefore produce different single applications depending on
    the hint. Test a fully coupled two-block nonsymmetric system against
    `weight * A^-1 b`, both layouts and zero-guess flags. Runtime reproduction
    is pending. Current matK native ILU candidates use weight 1 and are not
    implicated by this specific inconsistency.

- [ ] **Check exhaustion of native ILU1 symbolic hash retries before using counts.**
  - `CSR_Multiply_Impl::sparsity_ilu1` retries symbolic counting ten times but
    does not test a final success flag before consuming counts and allocating
    the filled pattern. Add a defined failure return if every attempt fails.
    This is a source-observed defensive gap, not a reproduced matK failure.

- [ ] **Make nested solver output attributable and honor requested verbosity.**
  - Root-level object JSON automatically receives `default_sub_solver` and
    distinct child scopes (`amg_config.cu:576-585`); an omitted explicit JSON
    scope does not itself trigger the default-scope compatibility reset.
    Named scopes are useful for attribution, not a demonstrated parser fix.
  - A downstream run of 2026-10-30 froze outer `print_solve_stats=0` from a verbosity-2
    config and subsequently normalized only truthy flags. CLI verbosity 3
    therefore printed inner PBICGSTAB rows but suppressed outer FGMRES rows.
    The benchmark generator now explicitly reapplies outer print controls.
  - Include solver/scope in nested headings and test quiet-to-verbose config
    reuse. Do not reinterpret the actual outer row 276 of a nested-GMRES run
    of 2026-10-30 as an inner row: its norm is 1.045697e6, ratio 0.9994; the final complete outer row
    1084 is about 0.9991. These remain estimates without a final physical
    sample. Parser and saved-log audit (`docs/matk_nested_wide_gmres_findings.txt`, local, untracked).

- [ ] **Fix incomplete native IDO coloring and validate all rows before ILU.**
  - `run_cpu_serial_IDO_ordering_greedy` in
    `src/matrix_coloring/serial_greedy_bfs.cu` sets `use_ido_for_bfs=false` and
    `bfs=true` (around lines 233-234). The shared neighbor checker then marks
    discovered uncolored neighbors `-1` (around lines 66-71), but the IDO queue
    update only reinserts neighbors still equal to zero (around lines 359-370).
    This can exhaust the queue and exit with rows left uncolored.
  - Select the proper IDO traversal semantics and check coloring completeness
    and neighbor conflicts before factorization. An IDO smoke run of 2026-10-30 failed
    while BFS and MIN_MAX variants on the same system passed. Preserve that
    case as a regression; exclude IDO from promoted configurations until fixed.

- [ ] **Guard zero coarse-correction scaling and document CLASSICAL semantics.**
  - `src/aggregation/aggregation_amg_level.cu` (around lines 777-817) computes
    `alpha=nominator/denominator`, then clamps a small magnitude using
    `(alpha/abs(alpha))*0.3`. Zero numerator with nonzero denominator produces
    `alpha=0`, then `0/0` and NaN. Handle zero/nonfinite alpha explicitly and test
    an error correction orthogonal to the residual, not just zero denominator.
  - The minimum-magnitude clamp of 0.3 can also enlarge a valid small
    residual-minimizing step on poorly conditioned nonsymmetric systems;
    document this policy and evaluate a safeguarded alternative.
  - CLASSICAL's prolongation path in `src/classical/classical_amg_level.cu`
    (around lines 1581-1703) directly adds `P*e` and does not implement the same
    `error_scaling` option. Do not describe `error_scaling=2` as stabilizing a
    CLASSICAL configuration; implement consistent semantics or expose that
    the option is unsupported/ignored for that algorithm.
  - **Distributed semantics require verification:** option 2 uses full vector
    extents and no global reduction, whereas option 3 explicitly reduces owned
    entries. Establish whether option 2 intends partition-local or global
    minimization and test halo handling and partition invariance. This source
    observation is not a reproduced MPI failure.

- [ ] **Audit CR selector ownership, empty fine neighborhoods, and norm assumptions.**
  - `CR_iteration` in `src/classical/selectors/cr.cu` overwrites `m_smoother`
    with a new allocation (around line 478), while the destructor deletes only
    the last pointer. Repeated selection/setup can leak the previous smoother;
    make ownership explicit and test repeated calls on one selector.
  - The locally enabled `EXPERIMENTAL_CR` path divides by `fine_count` without
    checking for zero (around line 179). Handle a fine row with no fine
    off-diagonal neighbors before forming its threshold.
  - The convergence metric takes `sqrt(sum(diag(A)*u*u))` without guarding a
    negative value. Document/enforce the positive-energy contract or implement
    a suitable nonsymmetric metric. These are source findings, not reproduced
    full-matK CR failures; CR was not promoted for this indefinite system.

- [ ] **Remove or implement the apparently unused `GS_L1_variant` parameter.**
  - It is registered in `src/core.cu`, but the source audit found no solver
    implementation consuming it. Confirm its intended contract and reject or
    document unsupported use. Setting it must not be advertised as selecting an
    L1 multicolor GS smoother; the explicit `JACOBI_L1` implementation is separate.
  - `fine_smoother` and `coarse_smoother` are likewise registered in `core.cu`
    with no consumers found in the source audit. They cannot currently be used
    to give the finest AMG level ILU while selecting a different coarse-level
    smoother; expose the support contract rather than silently accepting them.

Reproductions and actual configuration dictionaries are in the local records
of the Hybridge campaign `matk`. The driver is
`../hybridge/campaigns/matk/benchmark_amgx_strong.py`; use explicit candidate names
and the stored scaling/CLI parameters because configurations evolved during the
search. The isolated native ILU capacity extension described above followed
this search. Keep that build separate from the installed baseline;
other source findings remain recorded for later implementation work.

## P1 — coarse solver configuration (2026-11-02)

- [ ] **`dense_lu_max_rows` is not applied when coarsening ends by discarding a next level.**
  `src/amg.cu` checks the cap only at the top of the setup loop (`setup`, `setup_v2`), which covers
  `max_levels` and levels at or below `dense_lu_num_rows`. A next level smaller than
  `dense_lu_num_rows` is deleted further down, the current level becomes coarsest, and the loop
  exits without the check. Separately, the check uses `>` where `src/core.cu` documents `>=`.
  - Evidence: `../hybridge/campaigns/fork_integration/check_dense_lu_cap.py`, run on 2026-11-02 with
    the library of SHA256 prefix `b6a01d2d`; its records are in the local records of the Hybridge
    campaign `fork_integration`. On a
    6,400-row Laplacian with one level and cap 4096, the discard exit solved directly (1 iteration,
    345 MiB) while the `max_levels` exit used the smoother (102 iterations, 1 MiB). A cap of exactly
    6,400 still solved directly. On a 90,000-row Laplacian with an 8,729-row coarsest level, cap 4096
    gave a residual history identical to no cap (25 iterations); forcing the other exit gave 394.
  - Consequence: 23 Hybridge configurations set this cap (21 at 4096, 2 at 256) and it has no effect
    on this exit. 9 of 16 profiled article hierarchies have a coarsest level above 4096 rows, up to
    25,899. Applying the cap there replaces dense LU with coarsest sweeps, so a fix changes the
    numerics of those configurations and needs requalification. Nothing has been changed.

## Completed milestone — borrowed CuPyX CSR

- [x] Build and qualify `AMGX_matrix_attach_csr` and PyAMGX `Matrix.attach_CSR()`.
  All three native CSR regressions, 136 Python tests, 43 PTDS attachment tests,
  memory checking, and FP32/FP64 transfer profiles pass. Qualification covers
  the documented single-GPU scalar CSR contract; dDFI checks its existing
  unsupported mixed-precision solve error. See [borrowed CSR](docs/borrowed_csr.md).

## Completed downstream milestone — borrowed device vectors

- [x] Qualify the new borrowed-vector storage and C API with PyAMGX
  `Vector.attach()`. The build and dDDI/dFFI native, Python, and
  transfer-profile checks pass. The dDFI solve regression exposes the existing
  unsupported mixed-precision SpMV path; mixed solves are documented as outside
  this milestone. All three native regressions pass, including its expected error. See
  [borrowed vectors](docs/borrowed_vectors.md) for ownership, ordering, scope,
  and the exact build command. BSR integration is paused for this milestone.

## P0 — block sparse GPU performance

**Next roadmap item:** promote the modern cuSPARSE generic-BSR reference path to a matrix-owned cache with persistent descriptors, preprocessing state, and workspace.

- [x] Add an opt-in `cusparse_generic` BSR SpMV reference backend for complete device views, with legacy fallback for partial/distributed views, unsupported precision combinations, and CUDA Toolkits older than 13.0.
  - CUDA/cuSPARSE 12.8 accepts `cusparseCreateBsr`, but its `cusparseSpMV_bufferSize` rejects BSR at runtime; the local smoke test therefore exercises and verifies the legacy fallback. Generic BSR SpMV arrived in CUDA Toolkit 13.0 Update 1. Gate it on `CUDART_VERSION`, because Toolkit 13.0 ships cuSPARSE library major 12 and `CUSPARSE_VER_MAJOR` is not a CUDA Toolkit version test.
  - BICGSTAB and PBICGSTAB now propagate the backend selected in their outer solver scope directly to the fine matrix. PBICGSTAB reapplies the outer choice after nested preconditioner setup, so its `A*Mp` and `A*Ms` operations have deterministic backend ownership.
  - An explicit `cusparse_generic` request now takes precedence over AMGX's historical 3x3 and 4x4 standalone kernels. The generic implementation retains complete-view, precision, square-block, and CUDA-version guards and falls back to legacy cuSPARSE BSR where required.
- [x] Add opt-in subgroup kernels for the hot 5x5 BSR paths, retaining legacy cuSPARSE/two-`bsrmv` behavior when `block_jacobi_use_fused_small_blocks=0`.
  - The standalone SpMV and fused Block Jacobi specializations assign an aligned eight-thread subgroup to each block row. Five lanes compute the five output/residual components concurrently; Jacobi reuses its residual through shared memory for the inverse-diagonal update.
  - Store dispatch state on each matrix through copied `AuxData`, so hierarchy levels and concurrent solver objects do not depend on process-global state.
  - Keep 2x2 and 3x3 on the one-thread fused specialization until profiling justifies a subgroup variant; 3x3 and 4x4 standalone SpMV already have AMGX custom kernels.
- [ ] Add an opt-in modern cuSPARSE BSR SpMV path for generic block sizes, with descriptors and external workspace owned by the matrix/resource lifecycle.
  - Cache `cusparseSpMatDescr_t`, dense-vector descriptors, preprocessing state, and workspace using matrix structure/value revisions; invalidate them on pointer, shape, block-format, precision, device, or stream-incompatible changes.
  - Do not use a process-global pointer-keyed cache: it is unsafe under matrix destruction, allocator address reuse, resetup, and concurrent solver objects.
  - Retain the legacy `cusparse*bsrmv` path for A/B tests and unsupported CUDA/type combinations.
- [ ] Profile standalone BSR SpMV and complete AMG solves separately for block sizes 2 through 5. Record kernel time, launch count, achieved bandwidth, setup/preprocess cost, smoother time, coarse-level format changes, iterations, and end-to-end solve time.
  - [x] Re-ran the HDG advection p=1..6 CSR/BSR sweep after rebuilding the 2026-08-24 BICGSTAB/PBICGSTAB propagation patch. Warmed alternating trials covered nx=64, 128, and 256. At nx=256, AMGX-solve speedups were 1.16x/0.97x/1.33x/1.00x/1.22x/1.31x for p=1..6. Nsight p=2 recorded generic `cusparseSpMV` and a 3x3 `bsrmv_tiny_core`, not the old AMGX custom kernel; the isolated BSR GPU SpMV work was about 1.42x faster than CSR, but block-vector reductions and per-call descriptor/workspace handling erased that gain for the complete p=2 solve. The external application benchmark report is not bundled here.
  - [x] Screen direct PBICGSTAB+`MULTICOLOR_DILU` on the unstructured HDG advection matrix. Unscaled parallel-greedy DILU at weight 0.7 improves fine-mesh p=1..3 wall time over scalar-row L1, but loses at p=4. Trial 6x6/7x7 large-kernel dispatches fail to converge on p=5/6 and are not retained; do not infer support from a random finite-output smoke test.
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

This roadmap focuses on trustworthy solver telemetry, actionable diagnostics, and stable communication contracts for C/C++, Python, MPI, CuPy, and other consumers. A log line, stored residual, convergence decision, and C API query must describe the same solver state. GPU interoperability work is coordinated against the companion `../pyamgx` repository and CuPy rather than inferred from raw pointers alone.

## P0 — HDG face-block p/h multigrid support

- [ ] Support the reduced scalar-AMG role in the HDG [face-block hp-multigrid
  plan](../hybridge/docs/development/plans/face_block_hp_multigrid.md). Accept the
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
- [ ] Future project: provide a solver-independent path that attempts to satisfy the requested true relative residual target `||A*x-b|| / ||b||`, regardless of the iterative solver's native convergence convention.
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

## P1 — complex-double warp shuffle argument forwarding

- [ ] Fix the `cuDoubleComplex` overloads of `utils::shfl` and `utils::shfl_xor` in `include/sm_utils.inl`: their real-double calls pass `(mask, bound)` where the declared order is `(bound, mask)`. This supplies an invalid shuffle width and the wrong participant mask, including for the existing default arguments. Preserve the public signature and test distinct lane values with full-warp and lower/upper 16-lane masks under synccheck. This is a source-confirmed pre-existing defect; no complex GPU reproducer has run. Real FP32/FP64 matK runs are unaffected. Evidence and scope: application patch review (`docs/matk_ilu_application_sync_fix_review.txt`, local, untracked).
# matK consecutive BSR upload finding (2026-10-31)

- [x] Validate sorted block-column indices before the `cusparse_legacy` ILU
  setup, or give an explicit diagnostic for unsorted input. SciPy CSR-to-BSR
  conversion can produce unsorted block columns even for sorted scalar CSR.
  In new PARDISO consecutive-block smokes of 2026-10-31 (b8 and b16), the input
  had unsorted block rows and setup reported an upper-triangular-analysis zero
  pivot (rows7 and2 respectively). This does not establish a numerical pivot
  failure for valid input. The benchmark now sorts BSR block columns before
  upload, preserving block-row/elimination order and all coefficients. Sorted
  rerun results are in a local results file (`docs/matk_pardiso_block_ilu_results.txt`, local, untracked). Backend
  detection is implemented in the isolated natural-order ILU(k) build:
  the final build qualification of 2026-10-31 verifies explicit BAD_PARAMETERS returns for unsorted ILU0/ILU2
  input, duplicate columns and missing diagonals. The production baseline
  remains unchanged. See `docs/matk_natural_iluk.txt`.

# matK natural-order ILU(k) extension review (2026-10-31)

- [ ] Architecture autotuning of supplied-order ILU(k): **deferred
  on 2026-10-31**. Keep `block_ilu_backend=cusparse_legacy` explicitly opt-in and
  experimental through this merge. Future equivalent-kernel tuning must retain
  the selected fill, ordering, block size, precision, relaxation and solver
  tolerances. No tuning implementation or additional exploration is part of
  the current reconciliation.

- [x] Preserve existing AMGX error codes through the new symbolic-setup catch
  block. `amgx_exception` derives from `std::exception`; the initial generic
  catch would have relabelled device allocation/CUDA failures as bad parameters
  and replaced the original call site. An explicit rethrow now precedes generic
  catches. This is a source-confirmed error-reporting defect, not a numerical
  factorization defect; full-system trials did not trigger it. Build history
  and numerical qualification are in `docs/matk_natural_iluk.txt`.

- [ ] Investigate FGMRES success with a much larger explicitly recomputed
  residual after higher-fill ILU. Unscaled matK runs with ILU10 and ILU20 (2026-10-31)
  return success at a native relative tolerance1e-10, but original-system FP64
  relative residuals are2.52873e-8 and7.46321e-5 respectively. In
  `src/solvers/fgmres_solver.cu`, convergence uses `abs(m_s[m+1])` before the
  final solution update; true `b-Ax` is recomputed at restart, with no final
  verification before success. This is measured residual drift/acceptance
  evidence, not proof of a defect in the new ILU factors. Consider reliable
  final residual verification and orthogonality controls. The benchmark
  correctly rejects both results. Reports and follow-up stopping/restart
  trials: `docs/matk_natural_iluk.txt`.

- [ ] Investigate reliable PGMRES + supplied-order b8 ILU(10) failure on the
  320-unknown nonsymmetric matK smoke. Runs of 2026-10-31 (none/diagonal scaling,
  restart1000, ALWAYS reorthogonalization, reliable residual verification)
  completed100 iterations with original relative residuals4.38910e8/5.99654e8.
  Comparable PFGMRES smokes and the independent ILU application oracle pass.
  No full matK solve was launched for these variants. This is a potential
  GMRES/block/preconditioner integration defect or numerical failure, not an
  established root cause. Preserve the smoke logs in the local records of the Hybridge campaign
  `matk` for isolation.

- [ ] Consider audited 64-bit scalar-value addressing for higher-fill block
  ILU. Input-order matK ILU(10) (2026-10-31) correctly trips the factor bound
  `(INT_MAX / 64) - 1 = 33,554,430` blocks before GPU allocation; the same
  supplied-order backend with PARDISO uses29,152,500 blocks and converges.
  This is an implementation capacity limitation, not evidence of hardware OOM
  or a numerical defect. Do not remove the bound without auditing AMGX vector
  sizes, scatter/index arithmetic, allocation products and backend interfaces.
  Evidence: `docs/matk_natural_iluk.txt`.

## Integration review (2026-10-31)

- [x] Reject borrowed BSR dimensions/counts before overflowing the internal signed-short block size or signed-int scalar offsets. Found during the integration review; guarded in `include/borrowed_matrix_api.h` and pyamgx's descriptor, with C API and protocol regression cases.

- [ ] Implement or formally exclude mixed-precision smoother SpMV (`dDFI`) on CUDA 10.1+. Integration runs of 2026-10-31 reproduced the existing `amgx_cusparse.cu` rejection in both old and combined libraries. FP32/FP64 smoother suites pass; this is inherited unsupported functionality, not a new regression.
