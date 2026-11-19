# Downstream AMGX change registry

This checkout is not an unmodified NVIDIA AMGX tree. It includes downstream solver experiments and APIs used by companion
Python bindings and application integrations. Keep this
file current whenever source behavior, the C API, configuration semantics, CUDA
requirements, or build/runtime coupling changes.

The upstream comparison point for the current branch is `upstream/main` at
`91a8413` (`Fix CUDA 13 tuple compatibility`). The downstream branch is
`hdg-cuda13-integration`.

## Applied changes

### Combined H200 integration (2026-10-31)

The working-tree changes below are integrated together in a fresh CUDA 13.1
sm_90 build, rather than only in the earlier isolated libraries. See
[the regression record](docs/h200_integration_validation.md) for exact coverage
and inherited limitations. Patch files under `docs/matk_*.patch` retain their
historical proposals; the entries below identify which are now applied and
qualified. A patch file's presence alone does not establish its runtime status.

- Borrowed square-block BSR extends the existing CSR ownership/stream contract
  through `AMGX_matrix_attach_bsr`, with canonical structure and explicit block
  diagonals. Bounds checks reject overflowing block sizes/scalar counts before
  buffer access; native invalid-input cases cover the integration-review fix.
- Native DILU now includes block sizes 6 and 9. The native smoother checks cover
  FP32/FP64 sizes 1–10, and independent embedding/convergence tests cover the
  newly enabled paths.
- Supplied-order ILU(k), native multicolor capacity dispatch, CSR row-length
  reduction and empty BFS color-boundary corrections are built together.
  Existing numerical and synchronization limitations remain tracked in TODO.md.

### Experimental supplied-order cuSPARSE ILU(k) (2026-10-31)

- This remains an explicitly selected experimental backend after the merge.
  Automatic architecture tuning was deferred on 2026-10-31;
  existing arithmetic, fill/order settings and the opt-in selector are unchanged.

- `block_ilu_backend=cusparse_legacy` now accepts nonnegative
  `ilu_sparsity_level`: a host symbolic elimination pass builds the exact ILU(k)
  block pattern in the supplied order, then existing GPU cuSPARSE routines
  factor and apply it. The original operator remains separate from private
  factor storage. There is no pivoting, graph coloring, or new global ordering.
- Existing native multicolor ILU(1) remains unchanged. The extension is in
  `include/solvers/natural_ilu_pattern.h` and the legacy branch of
  `src/solvers/multicolor_ilu_solver.cu`. Canonical sorted input, internal
  diagonals, single-GPU operation and signed32 allocation bounds are enforced.
- An isolated build that recompiled only `multicolor_ilu_solver.cu` preserves the
  baseline and excludes previous experimental multicolor capacity changes.
  No C API or class-layout change and no pyamgx rebuild are required.
- Qualification:600 CPU symbolic comparisons;20 baseline ILU0 and80 extended
  GPU cases across five block sizes, k0..3, FP64/FP32, permutations, nonzero
  guesses, damping and coefficient resetup; exact original-operator integrity
  checks; CUDA memcheck zero errors; five C API input-rejection checks. Final
  error-path review preserves existing AMGX exception codes. Six full matK
  initial trials at k1/2/3 with/without diagonal scaling did not converge.
  Follow-up k5/10/20 reference tests passed; PARDISO+b8 ILU10/PFGMRES with
  restart1000 and native tolerance1e-14 converged from zero to an original-system
  relative residual1.65e-11 (2026-10-31). Reliable PGMRES alternatives failed small
  checks and are tracked separately in TODO.md. See [design and results](docs/matk_natural_iluk.txt) and downstream
  `../hybridge/campaigns/matk/commands_natural_iluk.txt`.

### Native ILU synchronization and optional inverse diagnostics (2026-10-31)

- The synchronization change of 2026-10-31 orders shared publication/reuse in A-to-LU mapping,
  factorization, inverse tiles, symbolic ILU1 broadcasts and triangular
  application. Application masks follow independent 16-lane row ownership.
  Discarded odd-pair source-list reads are avoided; symbolic failure exits are
  warp-uniform. No common header, C API or numerical policy changes.
- Optional `print_coloring_info` counters report inverted blocks, determinant
  fallbacks, exact-zero determinants and nonfinite determinants. Diagnostics
  default off; no counter allocation/atomic/copy/summary occurs when disabled.
  All 16 enabled/disabled numerical comparisons were bitwise identical.
- H200 qualification runs of 2026-10-31 passed four smoke stages, 28 capacity cases,
  40 native application/diagnostic cases, 40 synccheck invocations and 32
  racecheck invocations with zero warnings/errors. Both real precisions and
  layouts, external diagonals, 8193 paired inverses and 16385 rows per color
  were exercised. [Exact identities and limits](docs/ilu_synchronization_qualification.md)
  distinguish these corrections from experimental supplied-order ILU(k).
- Symbolic subwarp widths 2/4/8/16 and deliberately induced hash-failure exits
  remain source-reviewed/compiled but not GPU-qualified. Further qualification
  was stopped after these runs. Complex/distributed execution,
  nonunit damping, determinant-scaling policy and extreme index limits retain
  their separate evidence limits and TODO entries.

### Bounded CLASSICAL interpolation truncation (2026-10-31)

- The truncation change of 2026-10-31 bounds the fallback
  `truncateByMaxElements` launch to its 1024-entry shared arrays: 128 threads
  for caps 1–8, 64 for 9–16, and 32 for 17–32. Coefficient selection, stable
  ties, CSR output and row rescaling retain their existing operations.
- The H200 qualification of 2026-10-31 passed all 191 checks plus the three registered native real
  modes (828 fixture comparisons). This includes an independent CPU oracle
  for every cap 1–32 in dDDI/dDFI/dFFI, 24 bitwise safe-cap comparisons, eight
  reproduced baseline faults, 24 zero-error memchecks and 15 zero-error
  syncchecks. [Exact evidence and limits](docs/truncation_qualification.md)
  cover partial blocks, grid strides, signed values, ties and cancellation.
- This qualifies the H200 direct operator. Full-matK convergence and attribution
  of earlier AMG hierarchy failures remain unestablished. Host/complex and
  cap-above-32 rejection contracts are unchanged; no new nonpositive-cap or
  empty-matrix semantics, experimental preprocessing behavior, or
  Quadro/Blackwell qualification is claimed.

### Native ILU row-capacity extension (historical qualification sequence)

- Native 4x4 ILU now preflights the filled LU row lengths and selects a 128- or
  1024-column numeric workspace before factorization. The arithmetic and color
  order are unchanged; there is no retry on partially factored values. Larger
  rows are rejected with their required capacity. Legacy cuSPARSE ILU is unchanged.
- An isolated build recompiles only `multicolor_ilu_solver.cu` and links the
  other baseline objects unchanged. Source snapshots, the patch, compiler output
  and library hashes are retained with that build; the baseline library hash is checked before and after.
- Compiled FP64 1024-column kernels use 18,432 bytes of shared memory and no
  register spills. GPU regression covers both layouts/precisions, 128/129/1024
  boundaries, rejection at 1025, and a fill-generated 129-column ILU(1) row.
  See `../hybridge/campaigns/matk/check_ilu_capacity.py` and
  the source/runtime audit (`docs/matk_native_krylov_findings.txt`, local, untracked).
- At this initial stage, independent residual checks exposed separate
  backend-selection and column-major precision issues tracked in `TODO.md`;
  a smoke pass is not a claim of full-matK convergence.
- The first isolated build exposed a pre-existing matrix-reordering defect at
  high row degree. `matrix.cu` now reduces row indices and explicitly reads
  adjacent CSR offsets; it no longer takes the address of a reduction argument
  and assumes the next offset follows it. A second isolated build
  replaces these two objects while
  preserving both older libraries. Its SHA256 is
  `7ece26037d7a3008d09bfcfd6d2708359e3db974917b6014912c24ecc40a18c8`.
- A qualification run of 2026-10-30 passed all 20 isolated-build cases (16 numerical
  checks and four expected capacity rejections) and all 20 baseline outcomes.
  The previously failing FP64 degree1024 systems now have independent residuals
  about 2.6e-16 in both layouts, versus about 0.096 before the correction.
  Both layout variants pass compute-sanitizer memcheck with zero errors.
  These results qualify the tested capacity paths; full-matK convergence remains
  unestablished. `matk_native_ilu_capacity_proposals.json` records the next
  three native configurations, pinned to this qualified library hash.
- A third isolated build also fixes empty interior BFS color offsets in
  `serial_greedy_bfs.cu`. A run with the previous build reproduces the old CUDA
  launch failure in four precision/layout cases; a run with the third build
  passes all four after the boundary correction (2026-10-30). Its library SHA256
  is `1c5f728bf6ba0dd6f9b9f2f7d2349607991504ed8390ed56c179e5a3e860a7fb`.
  The source, build script, compiler output and hashes are preserved separately.
- Independent audits found inherited determinant-scaling,
  synchronization and indexing issues, now tracked in `TODO.md`. The
  one-application inverse test and racecheck of 2026-10-30 demonstrate why
  end-to-end known-solution and memcheck passes are insufficient to qualify the
  factors themselves. The numeric cutoff/synchronization kernels are not yet
  changed in these isolated builds; no claim of universal qualification is made.
- A fourth, snapshot-only isolated build
  applies the synchronization patch (`docs/matk_ilu_sync_fix.patch`, local, untracked) to ILU
  mapping/factorization/local inversion and ILU1 symbolic broadcasts. It leaves
  production sources and all previous libraries unchanged. The independent
  review (`docs/matk_ilu_sync_fix_review.txt`, local, untracked) found no source-level blocker and
  checked 1,600 CPU inverse/layout cases. The existing determinant cutoff and
  index-width limits remain unchanged. Library SHA256:
  `0c82515f5ae6ab2fc2248bb9c6606ea261e7f5dd64daa183a42168cae69d563c`.
  This setup-only build did not complete the GPU qualification gate (see
  `TODO.md`); no full-matK result is claimed for this new library.

### Borrowed device vectors and scalar CSR

- Commit: `f73ec35bd08171ba2f25c282af6b3d102025751a`.
- Adds `AMGX_vector_attach` and `AMGX_matrix_attach_csr`, plus synchronization,
  detach, and attached-pointer queries. Caller-owned GPU buffers remain shared;
  the existing upload APIs still copy. Internal metadata and solver workspaces
  remain AMGX-owned.
- Setup/solve involving borrowed storage complete synchronously, with producer
  stream ordering and guards against incompatible shapes, overlapping buffers,
  CSR mutation, or releasing a matrix retained by a solver. Coefficient changes
  require producer synchronization and a new setup; use
  `structure_reuse_levels=0` to rebuild AMG.
- Qualification is limited to the documented single-GPU scalar FP32/FP64
  contract: canonical int32 CSR with explicit diagonals. Mixed dDFI solves
  retain the existing unsupported SpMV error; borrowed BSR and distributed
  use are excluded. See [CSR](docs/borrowed_csr.md) and
  [vector](docs/borrowed_vectors.md) contracts.
- After the build, all three native BorrowedCSR and three
  BorrowedVectors regressions passed. Companion PyAMGX records Python/PTDS
  tests, memory checks, and FP32/FP64 transfer profiles. Its `quality-of-life`
  branch exposes these C APIs and the reusable-solver wrapper; the binding
  must be built against matching headers and library.

### Solver diagnostics and memory reporting

- Commit: `7e6515188dfd06468f95c2626ab30de0c8f8cacb`
- Adds explicit monitored-residual descriptions and stopping-criterion output,
  reliable GMRES residual verification, corrected completed-iteration BiCGStab
  residuals, and compact solve/memory telemetry.
- Adds C API allocator counters for live, reserved, and peak memory and preserves
  allocation-failure classification across CUDA/Thrust paths.
- The initialization banner reports the compile-time CUDA toolkit separately
  from the loaded CUDA runtime and driver, so a newer runtime cannot disguise
  a binary built with an older toolkit and compile-time feature guards.
- Native regressions: `device_memory_stats.cu`,
  `gmres_reliable_residual.cu`, and `bicgstab_residual.cu`.

### Compact interval-controlled solve statistics

- Commit: `6e72880873cf7a4cd3b9adcd65790746b71fa567`
- Adds `print_solve_stats_interval` and a compact aggregate block-L2 display for
  native iteration tables. The aggregate is presentation-only; componentwise
  stopping tests and stored residual history are unchanged.
- Separates the compile-time CUDA Toolkit version from the loaded runtime and
  driver versions in the initialization banner.
- HDG verbosity level three requests every outer iteration, while other callers
  may retain a coarser printed cadence.

### Block-aware classical AMG and BSR operators

- Commit: `6699fa4276c0ec0ea0aee6513855e1cc7a866d68`
- Adds coefficient-exact scalar-expanded and pure block-graph classical AMG
  hierarchies for dense BSR matrices, block interpolation/Galerkin variants,
  larger-block DenseLU fixes, and block-aware smoother paths.
- Adds the opt-in `bsr_spmv_backend` choices `legacy`, `cusparse_generic`, and
  `custom_5x5`. Generic BSR uses CUDA 13's `cusparseCreateBsr` and
  `cusparseSpMV(..., CUSPARSE_SPMV_BSR_ALG1, ...)` for supported complete device
  views and retains legacy cuSPARSE fallback.
- Detailed algorithmic status, validation cases, and performance findings remain
  in `TODO.md` under the block-sparse and classical-AMG sections.
- Adds `device_mem_pool_enabled` as an optional allocator-pool control. It
  does not imply allocation-free setup/solve or zero runtime reservations.

### Direct BICGSTAB/PBICGSTAB generic-BSR selection

- Commit: `d8832c4ad6962d248e0f5c9594e897ce6ee2ce6f`
- Applied and qualified on 2026-08-24.
- `Solver::configure_bsr_spmv_backend` maps the outer solver-scope string onto
  the explicit matrix's integer dispatch flags. CUDA toolkits older than 13.0
  warn and select legacy BSR.
- `BICGSTAB` applies the choice during setup. `PBICGSTAB` applies it after nested
  preconditioner setup, making the outer scope authoritative for the fine
  operator products `A*Mp` and `A*Ms`.
- `multiply_block_size` gives `cusparse_generic` priority over the historical
  AMGX 3x3 and 4x4 kernels. Unsupported partial/distributed views or precision
  combinations still fall back inside `Cusparse::bsrmv_internal`.
- Plain `BICGSTAB` remains mathematically unpreconditioned. A nested
  `preconditioner` object in a BICGSTAB JSON is not instantiated by that solver;
  use `PBICGSTAB` when an active preconditioner is intended.
- Native regression: `src/tests/krylov_bsr_spmv_backend.cu` checks outer-scope
  propagation for both solvers on a 3x3 BSR identity and exercises the multiply.
- Validation status: the CUDA 13.0.1 shared library was rebuilt and loaded by
  PyAMGX. Warmed HDG p=1..6 CSR/BSR sweeps completed at nx=64, 128, and 256;
  the fine nx=256 case made BSR 1.16x, 1.33x, 1.22x, and 1.31x faster in AMGX
  solve time at p=1, 3, 5, and 6. Nsight confirmed CUDA-13 generic BSR dispatch
  at p=2 with `cusparseSpMV`/`bsrmv_tiny_core` and no legacy or custom-3x3
  SpMV. The external application benchmark report is not bundled here.
- Native-regression status: passed after rebuilding `amgx_tests_launcher`.
  `KrylovBsrSpmvBackend`, `DeviceMemoryStats`, `GMRESReliableResidual`, and
  `BiCGStabResidual` all passed in `dDDI` mode (4 tests, 0 failures). The later
  direct-DILU qualification also adds and passes `SmootherBlocksizes`.

### Direct block-DILU qualification

- Commit: `9572acd95d2d872c0a0e6502c23310a4f255d2b8`
- Qualified on 2026-08-24. `MULTICOLOR_DILU` is AMGX's
  direct block DILU(0)-style preconditioner; it is distinct from the separate
  `MULTICOLOR_ILU` solver. The existing GPU DILU dispatch supports square block
  dimensions 1, 2, 3, 4, 5, 8, and 10.
- HDG advection face blocks have dimension p+1. Warmed legacy-`test2` BSR runs
  found direct unscaled DILU with parallel-greedy level-1 coloring and weight
  0.7 competitive at p=1..3, but slower than L1 at p=4 on the fine mesh.
- Experimental 6x6 and 7x7 large-kernel dispatches compiled and passed the
  random smoother smoke, but were not retained: p=5/6 advection became
  non-finite without external scaling and did not converge in 1500 iterations
  with scaling, for both parallel-greedy and min-max coloring.
- `src/tests/smoother_blocksizes.cu` is now part of the focused launcher, checks
  finite output, and exercises supported block dimensions 1 through 5.

### Generic block-ILU qualification

- Commit: `e023bc09bd32ffe752834ade6537c9a925bdf567`
- Started and retained as an experimental oracle on 2026-08-25. The historical AMGX 4x4
  color-reordered `MULTICOLOR_ILU` path and all defaults remain unchanged.
- Added opt-in `block_ilu_backend=cusparse_legacy`: natural-order, row-major BSR
  ILU(0) through deprecated CUDA `bsrilu02` and `bsrsv2`. It supports equal real
  precision and block sizes 2 through 7; it rejects ILU(1), mixed precision,
  external diagonals, color-reordered columns, column-major blocks, and non-int32
  indices. Setup reuses descriptors/analysis/workspace across applications and
  reports structural or numerical zero pivots.
- CUDA 13.0.1 headers declare `*bufferSizeExt`, but the linked
  `libcusparse.so.12.6.3.3` does not export those symbols; the implementation
  uses the exported int-sized `*bufferSize` variants.
- Extended `src/tests/block_ilu_backend.cu`: the original 4x4 regression remains,
  and deterministic exact ILU(0) applications for b=2,...,7 match a pivoted
  host dense reference at `1e-10`. The focused five-test dDDI set passes.
- HDG legacy-test2 BSR benchmarks converged through p=6. On 248 triangles the
  oracle needs 7 iterations, but is faster than block Jacobi only at p=1. On
  3,704 triangles it needs 16--17 iterations versus 89--105 for scaled plain
  BICGSTAB, yet its CUDA-13 `bsrsv2` applications make AMGX solve time 2.9--5.2x
  slower. The backend remains a correctness oracle, not a promoted preset.
- Design constraints, source hashes, exact commands, timings, residuals, raw
  artifact paths, and the next `bsrsv2` policy / scalar-CSR `cusparseSpSV`
  checkpoints are maintained in `doc/experimental_block_ilu_backend.md`.

### Experimental block-AMG smoothing and residual monitoring changes

- Commit: `7096fa209038c8c7a07f1a761af976e6629d59a3` (2026-09-23).
- Clears reusable coarse-correction buffers before zero-start fixed-cycle
  pre-smoothing to avoid retaining a previous correction. Adds opt-in
  `block_jacobi_zero_start_fastpath`; corrects diagonal-block SpMV nonzero
  counts and generic-BSR dispatch, and revises multicolor DILU block handling.
  These changes do not establish better performance across block sizes.
- Extends `chebyshev_lambda_estimate_mode` with user bounds (3) and an SPD
  weighted-power estimate (4), and adds optional
  `chebyshev_reuse_initial_preconditioner` for fixed preconditioners. The SPD
  estimate is not a general spectral estimator for nonsymmetric systems.
- Adds `block_graph_dense_constraint_mode=constant_vector` for nodal scalar
  Poisson interpolation experiments and `classical_hierarchy_export_prefix`
  for lossless exports of interpolation, restriction, and coarse matrices.
  These are experimental algorithm/inspection options, not recommended presets.
- Revises the optional `rel_div_tolerance` guard to track growth against the
  best monitored residual with a roundoff floor. Adds `divergence_patience`
  and `divergence_grace_iters`, checks suspected divergence against an
  independent `b-A*x` residual when the current solution is available, and
  handles nonfinite residuals in GMRES/FGMRES paths. The growth guard is disabled
  by default; it is not a guarantee of convergence or breakdown recovery.
- Validation at commit time was source inspection and whitespace checks only;
  no native build or GPU execution was performed for that commit. Subsequent
  builds and scalar CSR/AMG attachment benchmarks exercise selected paths,
  including a million-DOF shifted Laplacian. They do not qualify all of these
  block-smoother, Chebyshev, interpolation, export, or divergence options.
  Broader numerical and performance validation remains outstanding.

### Multicolor DILU block sizes 6 and 9

- Integrated on `hdg-cuda13-integration` after combined H200 regression checks (2026-10-31).
- Adds `case 6` and `case 9` to the four block-size switches of
  `src/solvers/multicolor_dilu_solver.cu`: setup (`DILU_setup_NxN_kernel_large`,
  2 and 3 warp iterations per block), forward and backward
  (`DILU_*_NxN_dispatch_large`), and the last-colour `DILU_backward_NxN_kernel_skip`.
  The `_large` kernels are generic in N; blocks 6 and 9, like 7, do not divide the warp,
  which the 7096fa2 shuffle guards already handle. The sizes follow HDG BSR
  blocks p+1 for p=5 and p=8, which previously aborted with "block size was not
  enabled".
- `src/tests/smoother_blocksizes.cu` now covers block sizes 1-10 (finite result,
  DIAG on/off agreement). The combined build passes dDDI/dFFI; dDFI
  retains the prior cuSPARSE mixed-precision rejection.
- First built in an isolated build that recompiled only this one file with
  the flags of the primary build and relinked, without writing to the primary
  build tree. sm_90 ptxas:
  38-40 registers at FP64, no spills for 6 and 9 (the existing block-10 setup
  kernel spills).
- Original validation entry point: H200 stage `slv-03-dilu69`
  (`../hybridge/campaigns/gpu_lane/stages/amgx_dilu_check.sh`). It checks
  embedding parity (block b against b+1 with a decoupled identity component),
  FGMRES+DILU convergence, and bitwise identity with the production library on
  1-5, 7, 8, 10. It then reruns the Hybridge campaign cases that had aborted on these block sizes.

## Configuration contract

Place `bsr_spmv_backend` in the outer BICGSTAB/PBICGSTAB solver object when it
controls the Krylov fine-operator product:

```json
{
  "solver": {
    "solver": "PBICGSTAB",
    "bsr_spmv_backend": "cusparse_generic",
    "preconditioner": {
      "solver": "AMG"
    }
  }
}
```

Nested preconditioners and AMG hierarchy components may also use their own
backend setting for matrices they own. For PBICGSTAB the outer setting is
reapplied after preconditioner setup and therefore controls the shared fine
matrix used by the outer Krylov products.

For scalar CSR input, the setting does not convert storage to BSR and does not
change the scalar SpMV route. BSR block dimensions are supplied by the caller via
`AMGX_matrix_upload_all`.

## CUDA and binary coupling

- Generic BSR SpMV requires CUDA 13.0 Update 1 or newer in practice. The referenced qualification used CUDA 13.0.1; select your toolkit
  directory when configuring the build.
- Rebuild AMGX after every source change and rebuild PyAMGX whenever the AMGX
  library location, ABI, or linked CUDA toolkit changes.
- At runtime, confirm `libamgxsh.so`, cuBLAS, cuSPARSE, and cuSOLVER all resolve
  from the intended build/toolkit using `ldd`.
- Do not treat a successful BSR solve as proof of generic dispatch. Profile with
  Nsight Systems `--trace=cuda,nvtx,cusparse,cusparse-verbose` and require
  `cusparseCreateBsr`, `cusparseSpMV_bufferSize`, and `cusparseSpMV` ranges.

## Maintenance rule

When adding another downstream patch, record its commit (or mark it pending),
the numerical/configuration contract, affected public interfaces, CUDA/version
constraints, regression coverage, validation state, and cross-repository work
required in `hybridge` or `pyamgx`.
