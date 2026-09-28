# Experimental generic block ILU backend

This document records the design, validation gates, build commands, and
benchmark results for extending `MULTICOLOR_ILU` beyond its historical 4x4
CUDA kernels. The implementation is deliberately opt-in until every supported
block dimension passes the correctness and application-level performance gates
below.

## Scope and invariants

For HDG advection of polynomial order `p`, the trace matrix has dense face
blocks of dimension `b = p + 1`. The initial qualification range is therefore
`b = 2, ..., 7`, corresponding to `p = 1, ..., 6`.

The work is split into three independent stages:

1. reuse AMGX's existing block-graph ILU(0) or color-aware ILU(1) sparsity;
2. factor the supplied fixed block pattern numerically;
3. apply the resulting lower and upper block factors.

The following behavior must not change unless a configuration explicitly
selects an experimental backend:

- the existing 4x4 factorization and triangular-application kernels;
- the default `MULTICOLOR_ILU` configuration;
- the matrix, solver, or C APIs;
- HDG assembly and PyAMGX upload ownership;
- coloring, block row/column ordering, tolerances, and relaxation weights.

An ILU(1) implementation must use the existing AMGX symbolic pattern. It must
not reinterpret ILU(1) as a second application of ILU(0), nor introduce fill
outside that pattern.

## Backend sequence

Paths below use generic placeholders. Set them to your own AMGX/CUDA
locations. Historical application harnesses and raw logs are not bundled;
`/path/to/application/benchmark.py` denotes an external harness, not a script
provided by this repository. `results/` denotes a user-selected output directory.

### Checkpoint 0: establish the unchanged 4x4 baseline

Add a deterministic native test for a row-major 4x4 BSR matrix. Compare one
`MULTICOLOR_ILU` application with a host dense solve on a block-tridiagonal
system whose ILU(0) factorization is exact. Record native-test and small HDG
timings before introducing a backend selector.

### Checkpoint 1: legacy cuSPARSE correctness oracle

Add an opt-in setup/application backend using `bsrilu02` and `bsrsv2` for
ILU(0). These CUDA 13 routines support arbitrary block dimensions but are
deprecated, so this path is a correctness and performance oracle rather than a
permanent default.

The first oracle deliberately uses the matrix's natural block-row ordering and
requires numerically sorted block columns. It does not reproduce the historical
4x4 backend's color-ordered factorization. AMGX's existing
`reorderColumnsByColor()` only reorders entries inside original rows, so a
future like-for-like color-order comparison would require a true symmetric
block permutation of rows, columns, vectors, and the solution.

Qualify `b = 2, ..., 7` before enabling ILU(1). For ILU(1), pass AMGX's
zero-initialized expanded pattern to the same fixed-pattern numeric routine and
verify the factors and application against a host reference. Natural-order
ILU(1) also requires bypassing the existing color-column reorder while retaining
an exact mapping from the original values into the expanded pattern.

### Checkpoint 2: maintained generic AMGX kernels

Implement one templated factorization body and one templated forward/backward
application body, dispatched for block dimensions 2 through 7. Compile-time
dispatch is intentional: it provides one maintainable algorithm while allowing
CUDA to unroll the small dense-block operations.

The legacy cuSPARSE backend remains available only as a comparison oracle. A
modern scalar-CSR `cusparseSpSV` application and fixed-sweep generic BSR SpMV
application may be compared later, but neither is required to qualify the
first exact generic AMGX backend.

## Correctness gates

Each checkpoint must pass, in order:

1. deterministic finite-output native smoke test;
2. comparison with a host reference application for every enabled block size;
3. ILU(0) and then ILU(1) fixed-pattern comparisons;
4. row-major double-precision device mode, followed by the other compiled
   floating-point modes;
5. missing, singular, and near-singular diagonal-block diagnostics;
6. PBICGSTAB convergence and independently recomputed HDG physical residual.

Column-major matrices remain rejected until they receive their own explicit
qualification. Numerical pivot regularization must be documented and must not
silently change the existing 4x4 behavior.

## Benchmark protocol

Report setup, warmed application, solver, and end-to-end time separately.
PBICGSTAB applies its preconditioner up to twice per outer iteration, so solve
time and iteration count must always accompany isolated application timings.

For native microbenchmarks:

- use deterministic matrices and right-hand sides;
- perform at least one warm-up;
- report the median of repeated measured applications;
- synchronize only at timing boundaries;
- test small matrices where launch overhead dominates and larger matrices where
  block arithmetic and memory traffic dominate.

For HDG advection:

- case: legacy unstructured-square `test2`;
- assembly: raw CUDA fused local assembly with cooperative LU;
- matrix: row-major BSR;
- basis: `dub_orth` with legacy-Lagrange traces;
- orders: `p = 1, ..., 6` as support becomes available;
- mesh sizes: begin with `0.20`, then `0.10`, `0.05`, `0.02`, and `0.01`;
- warm each configuration once and use at least three measured trials for
  promoted performance claims;
- record AMGX iterations, setup/solve time, wall time, solver relative residual,
  and original unscaled physical relative residual.

## Recorded results

Results are appended here only after the exact source revision, linked CUDA
libraries, command, warm-up count, measured trial count, and pass/fail status
have been recorded. Raw bulky output belongs in a user-selected results directory; this document keeps
the durable summary and artifact paths.

### Checkpoint 0

Status: passed for the current 4x4 ILU(0) implementation on 2026-08-25. No
generic block-ILU solver source changes have been made.

Environment:

- AMGX branch: `quality-of-life`;
- base revision: `6699fa4276c0ec0ea0aee6513855e1cc7a866d68` plus the documented
  BICGSTAB/PBICGSTAB generic-BSR working-tree changes;
- build tree: `/path/to/AMGX/build`;
- GPU: Quadro RTX 6000, compute capability 7.5;
- AMGX linked against CUDA 13.0.1 libraries under
  `/path/to/cuda/targets/x86_64-linux/lib`;
- CuPy-reported runtime: 13.2; CUDA driver: 13.0.

Focused build:

```bash
cmake --build /path/to/AMGX/build \
  --target amgx_tests_launcher -j 16
```

The incremental build completed in 21.1 seconds. The deterministic
`BlockILUBackend` test constructs four 4x4 block rows with dense diagonal and
off-diagonal blocks, injects the row colors `0, 1, 2, 3`, applies one current
`MULTICOLOR_ILU(0)` iteration, and compares it with a pivoted host dense solve.
Explicit custom colors are used because AMGX intentionally rejects its current
round-robin coloring when `determinism_flag=1`.

Focused native test:

```bash
env \
  LD_LIBRARY_PATH=/path/to/AMGX/build:/path/to/cuda/targets/x86_64-linux/lib \
  /path/to/AMGX/build/src/amgx_tests_launcher \
  --mode dDDI BlockILUBackend --verbose
```

Result: passed at tolerance `1e-10`; launcher wall time was 0.77 seconds. The
five-test focused regression containing `BlockILUBackend`,
`KrylovBsrSpmvBackend`, `SmootherBlocksizes`, `BiCGStabResidual`, and
`DeviceMemoryStats` also passed with zero failures.

The HDG baseline used the retained matched benchmark harness
`/path/to/application/benchmark.py` and this preconditioner:

```text
PBICGSTAB
  outer BSR SpMV: cusparse_generic
  preconditioner: MULTICOLOR_ILU
  ILU sparsity level: 0
  coloring: PARALLEL_GREEDY, level 1, exact
  column reorder and diagonal insertion: enabled
  relaxation factor: 0.7
  external system scaling: disabled
```

Representative convergent command:

```bash
env \
  LD_LIBRARY_PATH=/path/to/AMGX/build:/path/to/cuda/targets/x86_64-linux/lib \
  python /path/to/application/benchmark.py \
  --mesh-size 0.05 --orders 3 \
  --variants pbicgstab_multicolor_ilu0_07 \
  --formats bsr --scales off --warmups 1 --repeats 3 \
  --output results/block-ilu-checkpoint0-ms005-p3.jsonl
```

Tiny-mesh screening produced the following results. A failed warm-up has no
measured timing median and is retained as a numerical failure rather than
silently omitted.

| mesh size | triangles | scaling | result | median iterations | median AMGX setup | median AMGX solve | median wall | physical relative residual |
| ---: | ---: | :---: | :--- | ---: | ---: | ---: | ---: | ---: |
| 0.20 | 248 | off | diverged in warm-up; 1500 iterations | - | - | - | - | `9.554e20` |
| 0.20 | 248 | on | diverged in warm-up; 1500 iterations | - | - | - | - | `9.251e55` |
| 0.10 | 944 | off | diverged in warm-up; 1500 iterations | - | - | - | - | `9.816e45` |
| 0.05 | 3,704 | off | all three trials converged | 31 | 1.691 ms | 9.954 ms | 85.750 ms | median `3.728e-11` |

The three converged `mesh_size=0.05` trials required 31, 32, and 31 iterations.
Their physical relative residuals were `3.886e-11`, `2.545e-11`, and
`3.728e-11`. The first process-level warm-up took substantially longer because
it included initialization/JIT/cache effects and is excluded from the median.

Raw result artifacts:

- `results/block-ilu-checkpoint0-ms020-p3.jsonl`;
- `results/block-ilu-checkpoint0-ms020-p3-scaled.jsonl`;
- `results/block-ilu-checkpoint0-ms010-p3.jsonl`;
- `results/block-ilu-checkpoint0-ms005-p3.jsonl`.

Interpretation: the native arithmetic baseline is valid, but the current
unscaled p=3 ILU(0) preset has a small-mesh stability boundary between 944 and
3,704 triangles for this deterministic mesh/configuration sequence. Future
backends must reproduce or improve this behavior; performance comparisons must
use a convergent case and report failures separately.



### Checkpoint 1

Status: the natural-order legacy-cuSPARSE BSR ILU(0) oracle passed correctness
for `b=2,...,7` on 2026-08-25. It is not promoted as a preset because its
triangular applications lose decisively to scaled plain BICGSTAB once the mesh
has 3,704 triangles.

#### Source and configuration

The base revision is
`6699fa4276c0ec0ea0aee6513855e1cc7a866d68` on branch `quality-of-life`, plus
the documented BSR Krylov working-tree changes. The checkpoint adds
`block_ilu_backend`, whose default is `amgx`. Only
`block_ilu_backend=cusparse_legacy` selects the new path. No C API, PyAMGX,
HDG assembly, current 4x4 kernel, or default preset changed.

The final implementation-file hashes are:

```text
a15d5579e7367cee292cd2efe244fb2ea1f07116dab119e9e580fb1c9158a422  include/solvers/multicolor_ilu_solver.h
467aa56138a3fb6a0b47920e775a2c9e2f2c683a2153f9520783a3c03c268169  src/solvers/multicolor_ilu_solver.cu
587b3dd91a87acfc7216c1d8d79f528dd94877997de04c72bcb740758932a904  src/core.cu
8dc946a8e8448e8ea102fcbe287e004dae04a1ef0a7b4a222063e55490d7c5e2  src/tests/block_ilu_backend.cu
```

The opt-in backend has this contract:

- fixed-pattern block ILU(0), no pivoting or numerical boost;
- natural block-row ordering, zero-based int32 BSR indices, sorted block
  columns, square row-major dense blocks, and an internal diagonal;
- equal real matrix/vector precision (`dDDI` or `dFFI`); mixed precision is
  rejected before an application;
- CUDA `bsrilu02` factorization and two `bsrsv2` triangular solves;
- one workspace allocation and reusable analysis objects per solver instance;
- explicit structural/numerical pivot checks during setup;
- no AMGX coloring or color-column reorder in the opt-in path;
- ILU(1), column-major blocks, external diagonals, already color-reordered
  columns, and non-int32 indices are rejected.

`relaxation_factor` scales the complete preconditioner application. For
PBICGSTAB this uniform nonzero scaling cancels algebraically, so weights 0.7,
0.9, and 1.0 produced identical iteration counts and statistically
indistinguishable solve times. This is expected and is not a tuning opportunity.

CUDA compatibility exposed one header/library mismatch. CUDA 13.0.1 headers
still declare `cusparse[SD]bsrilu02_bufferSizeExt` and
`cusparse[SD]bsrsv2_bufferSizeExt`, but
`libcusparse.so.12.6.3.3` does not export those symbols. It does export the
int-sized `*bufferSize` functions, which the oracle now uses. The factorization,
analysis, solve, and pivot symbols are exported. All of these APIs are
deprecated and announced for removal in the next major CUDA release.

#### Build and native correctness

Build command:

```bash
cmake --build /path/to/AMGX/build \
  --target amgx_tests_launcher -j 16
```

The build passed. Its deprecation warnings are expected and retained in
`results/amgx-block-ilu-build-final.log`.

Focused test command:

```bash
cd /path/to/AMGX/build/src
./amgx_tests_launcher --mode dDDI \
  BlockILUBackend KrylovBsrSpmvBackend SmootherBlocksizes \
  BiCGStabResidual DeviceMemoryStats
```

All five tests passed. `BlockILUBackend` first preserves the historical AMGX
4x4 color-reordered regression, then applies the legacy backend to deterministic
block-tridiagonal matrices for every `b=2,...,7`. Because that pattern has no
missing fill, block ILU(0) is exact; every application agreed with the pivoted
host dense solve to `1e-10`. The final launcher output is
`results/amgx-block-ilu-tests-final.log`.

#### HDG benchmark commands

The final timed rows used the legacy unstructured-square `test2`, row-major
BSR, external scaling enabled, one warm-up, and three measured trials. Commands
were run from the `hdg` repository with:

```bash
LD_LIBRARY_PATH=/path/to/AMGX/build:/path/to/AMGX/install/lib:\
/path/to/cuda/targets/x86_64-linux/lib \
python /path/to/application/benchmark.py \
  --mesh-size MESH --orders 1,2,3,4,5,6 \
  --variants pbicgstab_cusparse_legacy_ilu0_09 \
  --formats bsr --scales on --warmups 1 --repeats 3 \
  --output OUTPUT.jsonl
```

The comparison baselines use the same warmed protocol, BSR upload, generic
cuSPARSE outer Krylov SpMV, and convergence target. `BICG` below is scaled plain
BICGSTAB; `BJ` is PBICGSTAB with one block-Jacobi application at weight 1.0;
`ILU0` is PBICGSTAB with this oracle. Times are medians in milliseconds.

Tiny mesh (`mesh_size=0.20`, 248 triangles):

| p | b | BICG it / solve | BJ it / solve | ILU0 it | ILU0 setup | ILU0 solve | ILU0 wall | physical relative residual |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 32 / 5.955 | 23 / 5.416 | 7 | 1.682 | 4.911 | 79.983 | `2.453e-11` |
| 2 | 3 | 31 / 6.313 | 24 / 6.176 | 7 | 1.753 | 6.161 | 81.497 | `3.053e-11` |
| 3 | 4 | 30 / 5.628 | 26 / 5.262 | 7 | 1.874 | 6.437 | 82.676 | `8.107e-12` |
| 4 | 5 | 28 / 5.735 | 24 / 6.104 | 7 | 2.205 | 8.003 | 85.006 | `1.203e-12` |
| 5 | 6 | 28 / 6.206 | 32 / 8.683 | 7 | 2.256 | 8.991 | 85.504 | `3.285e-12` |
| 6 | 7 | 29 / 6.445 | 32 / 8.657 | 7 | 2.382 | 9.216 | 88.344 | `6.944e-12` |

On the tiny mesh ILU0 wins solve time only at p=1 (9% versus block Jacobi),
ties block Jacobi at p=2, and is 15--43% slower than the fastest baseline at
p=3,...,6. Its setup is 1.7--2.4 ms versus approximately 0.27--0.35 ms for the
baselines. Process/Python/PyAMGX overhead dominates the roughly 80 ms wall time.

Modest mesh (`mesh_size=0.05`, 3,704 triangles):

| p | b | BICG it / solve | BJ it / solve | ILU0 it | ILU0 setup | ILU0 solve | ILU0 wall | physical relative residual |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 105 / 18.205 | 88 / 19.124 | 17 | 6.922 | 58.897 | 139.162 | `3.519e-11` |
| 2 | 3 | 98 / 19.561 | 85 / 22.384 | 16 | 5.934 | 57.588 | 137.589 | `2.837e-11` |
| 3 | 4 | 92 / 16.732 | 85 / 15.911 | 17 | 6.567 | 71.112 | 152.635 | `2.805e-12` |
| 4 | 5 | 89 / 18.658 | 87 / 26.381 | 16 | 8.298 | 83.690 | 167.439 | `1.495e-11` |
| 5 | 6 | 91 / 20.798 | 102 / 32.965 | 17 | 9.086 | 101.316 | 189.419 | `8.013e-12` |
| 6 | 7 | 89 / 21.407 | 103 / 36.299 | 17 | 9.988 | 112.009 | 206.118 | `8.894e-12` |

All cases converged and passed the independently recomputed physical residual.
Despite reducing the iteration count by about 5--6x, ILU0 takes 2.9--5.2x the
AMGX solve time of scaled plain BICGSTAB. PBICGSTAB applies the preconditioner
twice per outer iteration; the legacy level-scheduled `bsrsv2` solves therefore
cost roughly an order of magnitude more per iteration than the saved BSR SpMVs.
The factor quality is good, but the application backend is not competitive.

#### Fine-mesh rerun and BSR handoff timing (2026-08-25)

A corrected interpretation of "very small mesh size" used a fine Gmsh size
parameter, not a small problem. The matched run used `mesh_size=0.0065`, which
produced 219,472 triangles, 329,824 edges, and 328,592 interior edges. Every
stable row is the median of five measured solves after two warm-ups on the
Quadro RTX 6000. Fused raw-CUDA assembly, row-major BSR, external left scaling,
and the same `1e-10` AMGX tolerance were retained.

The HDG PyAMGX boundary was instrumented without changing AMGX source. The
existing aggregate `amgx_setup_elapsed_seconds` remains unchanged, while
`amgx_matrix_upload_elapsed_seconds` measures the synchronized
`AMGX_matrix_upload_all` call and `amgx_solver_setup_elapsed_seconds` measures
the synchronized `AMGX_solver_setup` call. Their sum reconciled with the
aggregate timer in the pre-run smoke test. The 219k run used the first timing
revision, which also included the preceding block-shape checks; the final
source starts immediately before `Matrix.upload`, so the tabulated upload
values are conservative upper bounds by a negligible Python-side cost.

| p | b | plain BICGSTAB it / solve (s) | block Jacobi it / solve (s) | legacy BSR ILU0 it / solve (s) |
|---:|---:|---:|---:|---:|
| 1 | 2 | unstable | 697 / 0.855 | 109 / 3.476 |
| 2 | 3 | 736 / 1.031 | 733 / 2.393 | 106 / 4.417 |
| 3 | 4 | unstable | 747 / 1.311 | 113 / 5.473 |
| 4 | 5 | unstable | 738 / 4.800 | 115 / 6.932 |
| 5 | 6 | 736 / 2.359 | 848 / 5.942 | 116 / 8.107 |
| 6 | 7 | unstable | 836 / 6.823 | 116 / 8.996 |

| p | GPU BSR assembly (ms) | BSR upload to AMGX (ms) | block-Jacobi setup excluding upload (ms) | ILU0 setup excluding upload (ms) |
|---:|---:|---:|---:|---:|
| 1 | 12.41--13.30 | 0.513--0.519 | 0.301 | 76.319 |
| 2 | 23.77--24.12 | 0.861--0.863 | 0.699 | 88.920 |
| 3 | 45.03--45.07 | 1.343--1.349 | 1.656 | 100.704 |
| 4 | 85.96--86.66 | 2.137--2.150 | 2.681 | 118.453 |
| 5 | 182.75--182.92 | 2.924--2.938 | 21.359 | 131.659 |
| 6 | 347.83--350.56 | 3.875--3.898 | 4.562 | 145.072 |

The assembly and upload ranges span the stable solver variants and show that
passing the already device-resident BSR matrix to AMGX is inexpensive: upload is
about 0.5--3.9 ms, compared with 12--351 ms for assembly. ILU0 cuts the outer
iteration count by roughly 6--8x but is slower than block Jacobi at every p; the
repeated legacy block-triangular applications, not upload or ILU factorization,
remain the limiting cost.

Plain BICGSTAB is fastest where it is stable, but it is not robust on this mesh.
The main run failed at p=1 and p=3 during warm-up, after one measured p=4 trial,
and at p=6 during warm-up. An identical retry produced only two measured p=1
solves before a non-finite failure and failed again for p=3,4,6. These rows are
therefore marked unstable instead of reporting partial medians. The full run is
`results/block-ilu-fine-219k-20260825.jsonl`; its `.stdout` sibling contains
warm-ups and failures. The controlled retry is
`results/block-ilu-fine-219k-bicgstab-retry-20260825.jsonl`.

Additional screening retained in `/tmp`:

- `codex-block-ilu-checkpoint1-smoke.jsonl`: p=1..6, scaled and unscaled,
  one trial; every case converged in 7 iterations on 248 triangles;
- `codex-block-ilu-checkpoint1-warmed.jsonl`: 60 tiny-mesh summaries covering
  both scaling modes, BICGSTAB, block Jacobi, and ILU weights 0.7/0.9/1.0;
- `codex-block-ilu-checkpoint1-final-tiny.jsonl` and
  `codex-block-ilu-checkpoint1-final-medium.jsonl`: final post-coloring-cleanup
  timed ILU rows used in the tables;
- matching `.stdout` files contain warm-up, summary, and error records. No final
  benchmark emitted an error record.

#### Decision and conservative next checkpoints

This backend succeeds as the arbitrary-block ILU(0) correctness oracle and
proves that natural-order block ILU can give excellent iteration counts through
p=6. It fails the performance gate and must remain opt-in.

Proceed in this order:

1. Add a native application-only microbenchmark and compare
   `CUSPARSE_SOLVE_POLICY_USE_LEVEL` with `NO_LEVEL` for `bsrsv2`. This is the
   smallest possible change and determines whether the current policy is the
   immediate regression.
2. If legacy BSR triangular solves remain slow, keep `bsrilu02` only as the
   factorization oracle and materialize its block factors once into scalar CSR
   L/U matrices. Split each dense diagonal block into scalar unit-lower and
   nonunit-upper parts, then benchmark modern reusable `cusparseSpSV`
   descriptors/analysis/buffers. This is generic for every p and confines the
   experiment to the adapter.
3. Only after the application backend passes should ILU(1) be enabled. Reuse
   AMGX's existing ILU(1) symbolic block pattern in natural order, preserve the
   exact A-to-LU value mapping, and compare the fixed-pattern factor/application
   with a host reference for `b=2,...,7` before any HDG timing.
4. If scalar `SpSV` is competitive, replace deprecated numeric factorization
   with one maintained templated block factor kernel dispatched for b=2,...,7.
   If it is not, implement one generic level-scheduled block triangular kernel,
   not separate p-specific implementations. Precompute diagonal-block solves
   or inverses once and retain natural ordering until a true symmetric color
   permutation is justified by measured performance.
5. Keep the historical 4x4 path and all defaults untouched until both ILU(0)
   and ILU(1) pass correctness, pivot-diagnostic, small/medium HDG, and profiling
   gates.
