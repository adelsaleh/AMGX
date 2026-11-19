# Native ILU synchronization qualification

The 2026-10-31 candidate combines the reviewed setup synchronization,
application synchronization and fallback instrumentation proposals in
`matk_ilu_sync_fix.patch`, `matk_ilu_application_sync_fix.patch` and
`matk_ilu_fallback_instrumentation.patch`. The earlier patch files and their
reviews remain historical artifacts.

The source changes are confined to `src/solvers/multicolor_ilu_solver.cu` and
`src/csr_multiply_detail.cu`. Both translation units must be rebuilt against
the integrated H200 baseline of 2026-10-31, which already includes the row-capacity,
matrix-row-length and BFS-color-offset corrections. No primary runtime is
replaced during qualification.

The setup kernels order shared table publication and reuse. The inverse uses
one explicit 16-lane tile per diagonal block. The forward/backward application
kernels use explicit 16-lane row masks and order every shared-memory phase;
neighboring rows can have different lengths or an inactive final tile.
An unmatched upper half of a final column pair uses a valid dummy shuffle
source without reading an unpublished compacted-list slot. Both ILU1 symbolic
overloads make their failure exit warp-uniform before subsequent collectives.

The inverse formula, determinant cutoff, diagonal fallback, relaxation factor,
fill policy and ordering are preserved. The known loss of dense coupling for
small uniformly scaled blocks remains an explicit numerical limitation.
`print_coloring_info=1` additionally reports diagonal-block, fallback,
zero-determinant and nonfinite-determinant counts. With the default disabled,
there is no counter allocation, atomic update, host copy or diagnostic line;
the nullable kernel argument and checks remain. Diagnostics can add setup cost.

## Qualification results

The qualification passed on 2026-10-31. The candidate source, with the
synchronization changes described above, was built on a CPU node against the
qualified integrated objects. The resulting
library SHA256 is
`218138a36fde113d53f1770c4884ba71260480035f24d08713ceea2e361626d0`.
Execution used NVIDIA H200 NVL, driver 590.48.01 and CUDA 13.1.115.

| Stage | Completed evidence |
|---|---|
| Smoke | Four smoke checks: coupled application, external-diagonal ILU1, paired inverse and diagnostic classification |
| Numerics and synccheck | 28 capacity cases, 40 native oracle/diagnostic cases, 40 zero-error synccheck runs |
| Racecheck | 24 racecheck runs, all with zero errors and warnings |
| Application-reuse racecheck | Eight application-reuse racechecks, all with zero errors and warnings |

The smoke and build stages used the Hybridge launcher of 2026-10-31; the broad
qualification used a later revision of the same date that adds
diagnostics-disabled sanitizer gates. Source and library identities were held
fixed throughout each run. Exact source/object/dependency hashes, commands,
logs and results are kept in the local records of the Hybridge campaign `matk`.
The campaign's tracked summary is Hybridge's `campaigns/matk/ILU_QUALIFICATION.md`.

Both real precisions and both block layouts passed. The largest per-block
application-oracle errors were `4.55e-14` in FP64 and `1.66e-5` in FP32, within
the declared `3e-10`/`8e-5` gates. All 16 diagnostics-on/off configuration
comparisons were bitwise identical. Repeated setup reset the counters.

Completed gates:

- Repeat the previously failing filled-129 case with completed zero-warning
  racecheck and zero-error synccheck summaries; allow at least 600 seconds per
  instrumented trial rather than the old 120-second limit.
- Check 127/128/129 and 1023/1024 row capacities, with explicit 1025 rejection,
  and filled rows crossing 128, in real FP32/FP64 and both block layouts.
- Check a single native application against independent block-factor and
  triangular-solve oracles, including zero and non-solution initial guesses,
  unequal neighboring row lengths, partial groups and odd occupied colors.
- Exercise external and internal diagonal mapping, the 8193-row paired inverse
  schedule and the 16385-row application workspace-reuse schedule.
- Compare diagnostics enabled/disabled and repeated setup counters. Preserve
  the known scaled inverse outcomes; classify deliberately zero/nonfinite
  determinants without treating those inputs as successful numerical solves.

The historical setup-only library failed application racecheck in a run of
2026-10-30 despite a passing final residual. A passing solve therefore does not substitute
for the sanitizer gate.

## Limits and deferred work

The normal solver path exercises the 32-thread symbolic ILU1 overload. The
2/4/8/16-thread overloads and deliberately induced hash-failure returns were
source-reviewed and compiled but not GPU-qualified. New qualification was
stopped after the runs above, so those direct-CSR fixtures remain
deferred. This is separate from the pre-existing symbolic retry-exhaustion
guard gap. Complex modes, multiple GPUs, extreme signed-index arithmetic and
nonunit-relaxation equivalence also retain their independent TODO entries.

The absolute determinant cutoff and diagonal fallback are deliberately
unchanged. Agreement with that retained policy on scaled blocks is not an
exact-inverse claim. The zero/NaN/Inf cases validate counter classification,
not valid numerical solves. No full-matK performance or robustness improvement
is claimed from these bounded tests.

Supplied-order ILU(k) (`block_ilu_backend=cusparse_legacy`) is a distinct,
explicitly selected experimental path. Its architecture autotuning was
deferred on 2026-10-31, and the merge followed these runs.
No numerical settings, backend defaults or autotuning policy were changed here.
