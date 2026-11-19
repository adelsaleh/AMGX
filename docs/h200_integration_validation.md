# H200 integration validation — 2026-10-31

The combined working tree was rebuilt in a fresh CUDA 13.1 Release/sm_90,
non-MPI CMake tree. PyAMGX was rebuilt into an isolated target directory.
Production libraries, installed environments and historical trial builds were
preserved. The final validation run **completed successfully in 67 seconds**
after an initial clean build. All source hashes recorded at the final run match
the final checkout; the machine-readable record (`docs/h200_integration_validation.json`, local, untracked)
contains library/source identities and local log locations.

| Check | Result |
|---|---|
| Native ownership, diagnostics, Krylov/block ILU, FP32/FP64 smoothers | 18 passed |
| Complete PyAMGX suite | 176 passed |
| Hybridge borrowed CSR/BSR, fixed cycles and packed kernels | 34 passed |
| Hybridge representative GPU smoke lane | 272 passed |
| Supplied-order ILU(k), k0–3, b1/2/4/8/16, FP32/FP64 | 80 passed |
| Fill-chain ILU(k), k5/10/20, b8, FP32/FP64 | 12 passed |
| Native multicolor ILU capacity boundaries | 20 passed, including expected rejections |
| DILU embedding and solve checks, block sizes 1–10 | 10 passed |
| DILU/FGMRES arrays for previously supported sizes vs previous library | 16/16 bitwise identical |
| Hybridge host-fast | 839 passed, 9 skipped |
| matK host checks | 27 passed, 11 subtests passed |
| Independent CPU symbolic ILU(k) checks | 600 patterns and 5 rejection cases passed |

The native `dDFI` smoother test fails with the existing mixed-precision cuSPARSE
rejection in both the previous and combined libraries. It is outside the
supported FP32/FP64 gate; its baseline failure log is retained. Native borrowed
BSR tests explicitly check the existing unsupported mixed-precision solve result.
Existing native multicolor ILU numerical/synchronization limitations remain in
AMGX's TODO; the archived proposal patches are not silently applied.

The integration review added missing BSR integer bounds checks, retained optional
symbol resolution for libraries without BSR attachment, and updated three old
pytest lifecycle hooks. Hybridge's one-shot path also needed `borrow_csr`
propagation and capture of its ownership report before cleanup. These fixes have
independent failure-path and real GPU solve coverage.

Reproduce from the sibling Hybridge checkout with
`sbatch campaigns/fork_integration/qualify.sbatch` after checking the shared GPU
allocation limit. The script records provenance and each stage's output in a per-run output
directory of the Hybridge checkout. Qualification is bounded; it does not assert
all AMGX configurations, distributed operation, Windows or other architectures.
