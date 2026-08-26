# Downstream AMGX change registry

This checkout is not an unmodified NVIDIA AMGX tree. It is the AMGX component
used by the sibling `hdg` and `pyamgx` repositories in this workspace. Keep this
file current whenever source behavior, the C API, configuration semantics, CUDA
requirements, or build/runtime coupling changes.

The upstream comparison point for the current branch is `upstream/main` at
`91a8413` (`Fix CUDA 13 tuple compatibility`). The downstream branch is
`hdg-cuda13-integration`.

## Applied changes

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
  SpMV. See the sibling HDG report
  `docs/backends/advection_bsr_benchmark_20260824.md`.
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

- Generic BSR SpMV requires CUDA 13.0 Update 1 or newer in practice. The active
  workspace toolkit is `/tmp/cuda-13.0.1`.
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
required in `hdg` or `pyamgx`.
