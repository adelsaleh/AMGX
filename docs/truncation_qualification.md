# Native truncation launch qualification

The fallback `truncateByMaxElements` kernel reserves 1024 shared entries for
indices and another 1024 for matrix values. Its original 128-thread launch
overruns those arrays for retained row lengths above eight. The launch now uses
128 threads for caps 1–8, 64 for 9–16, and 32 for 17–32. Selection, stable tie
handling, CSR output and row rescaling retain their existing operations.

This applies the reviewed `matk_truncation_launch_fix.patch` to the committed
H200 integration baseline of 2026-10-31 (borrowed BSR, DILU block sizes 6 and 9
and supplied-order ILU(k)). The original
patch, design and review remain historical evidence. Complex and host rejection,
the cap-above-32 rejection, and the experimental preprocessing branch are unchanged.

`TruncateMaxElements` is the maintained native regression. Its shared independent
CPU oracle, `src/tests/truncate_max_elements_oracle.h`, performs a stable full-row
sort and existing row rescaling, checking exact selected columns, finite values,
CSR lengths, precision-appropriate value tolerances and output hashes. Cases
cover every cap 1–32, short/long/empty rows, signed values, same- and opposite-sign
ties, cancellation, and partial thread blocks in dDDI, dDFI and dFFI modes.

The downstream `hybridge/campaigns/matk/check_truncation.cu` uses that same oracle
and verifies the loaded library and imported operator provenance. Its
`qualify_truncation.py` runner compares explicit fixed/baseline builds, reproduces
the baseline shared-memory fault, exercises all caps/real precisions and grid
strides, checks unchanged safe-cap hashes and runs CUDA memory/synchronization
checks. Build only the replacement `src/truncate.cu` through the downstream
`campaigns/fork_integration/build_variant.py` against the qualified integration
objects, or use a fresh full AMGX CMake build. Keep the primary runtime unchanged
until qualification is complete.

H200 qualification passed on 2026-10-31, using the AMGX candidate of that date
(the bounded launch plus the independent oracle) and a frozen Hybridge runner of
the same date. The
machine-readable result (`docs/truncation_qualification.json`, local, untracked) records exact source,
library and log identities. All **191 checks** passed, followed by all three
registered `TruncateMaxElements` native modes (828 fixture comparisons).

| Acceptance check | Result |
|---|---|
| Every cap 1–32, dDDI/dDFI/dFFI, independent CPU oracle | 96 processes passed |
| Safe caps 1–8 versus the unchanged integration baseline | 24 bitwise comparisons passed |
| Small/partial blocks, including rows 1/31/32/33/63/64/65/127/128/129 | 15 processes passed |
| Baseline shared-memory fault, caps 9/16/17/32, FP64 and FP32 | All 8 deliberately unsafe cases reproduced |
| Patched Compute Sanitizer memcheck, including grid strides | 24 processes, zero errors |
| Patched Compute Sanitizer synccheck | 15 processes, zero errors |
| Host/complex and cap-above-32 rejection contracts | 9 processes passed |

Grid-stride checks exercised 524289, 262145 and 131073 rows for caps 8, 16 and
32 respectively, in all three real modes. Signed values, opposite-sign ties,
zero retained sums, empty individual rows and short rows passed the same
independent selection/rescaling oracle. These are correctness checks; their
wall times are not controlled performance measurements.

The patched library SHA256 is
`1fe67ac5cd97d9d38fe12ded991b71a92f4762643ff8fcd793eedc553cda71cb`;
the baseline is the previously qualified integration library
`2c723ce35c42e1e392a042f48060ee464f0773fda449659514df64c3c44ca1ae`.
The checker validates its loaded library and imported truncation symbol in each
process. Explicit `RUNPATH` linking permits the requested baseline/candidate
selection; the host loader check caught and corrected an initial linker
`RPATH` choice before any GPU submission.

Compact logs, completion records, audited build recipes and execution outputs
are kept in the local records of the Hybridge campaigns `fork_integration` and
`matk`. The candidate was built in a separate isolated build tree. The
production runtime was unchanged during qualification.

This qualifies the launch fix for the H200 direct operator. It does not claim
full matK convergence, repair of unrelated AMG setup failures, new host/complex
support, new nonpositive-cap or empty-matrix semantics, or qualification on
Quadro/Blackwell. The experimental preprocessing branch is unchanged.
