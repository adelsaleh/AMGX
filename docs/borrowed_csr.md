# Borrowed device CSR

Status: qualified for the contract below. All three native CSR regressions,
136 Python tests, 43 PTDS attachment tests, the small FP64 memory check, and
FP32/FP64 transfer profiles pass. Results are recorded in the companion PyAMGX repository's
`docs/borrowed_csr_validation.md`.

`AMGX_matrix_attach_csr` aliases the caller's three device allocations. The normal
`AMGX_matrix_upload_all` remains a copying API. No replacement CSR arrays or
extra value sentinel are allocated on attachment. AMGX allocates its own diagonal
indices and other metadata, and validates the structure on the GPU. Validation
returns one integer status to the host. Solver workspaces and coarse AMG levels
remain AMGX-owned; zero copy describes sharing the supplied fine CSR arrays.

The initial contract is single-GPU square scalar canonical CSR, int32 indices,
and FP32 or FP64 values in dFFI/dDDI. dDFI attachment is available but retains the
existing unsupported mixed-precision SpMV limitation. There must be at least one
row and an explicit diagonal entry in every row (its value may be zero). Columns
must be strictly increasing within each row. Invalid inputs are rejected, never
silently sorted, padded, converted, or copied. Matrices requiring missing-diagonal
handling, distributed storage, or BSR are not qualified by this API.

The matrix handle must be empty and not retained by a solver. Buffers must be
nonoverlapping, aligned allocations on the resource GPU with the declared byte
capacities. The caller must provide valid allocations for the full declared
extents; pointer attributes do not prove allocation extent. Ownership remains
with the caller. Do not modify the structure or replace/free any allocation while
attached. Values may change in place between operations; synchronize the producer
and rerun setup before the next solve so preconditioner data is refreshed.
Use `structure_reuse_levels=0` when changing values; AMGX's full-hierarchy reuse
setting (`-1`) intentionally skips setup and cannot refresh a preconditioner.

Each buffer has a producer stream using CUDA Array Interface encoding: 1 for the
legacy stream, 2 for PTDS, a native handle otherwise; native 0 asserts completed
producer work. `AMGX_matrix_synchronize` waits for subsequent producer writes.
Setup/solve with a borrowed matrix complete AMGX's legacy execution stream before
returning. Calls must not overlap access to the same buffers.

`AMGX_matrix_get_attached_data` returns native row/column/value pointers.
`AMGX_matrix_detach` waits and empties the view, without copying or freeing the
caller buffers. Destroy every solver retaining the matrix before detach/destroy;
the native API checks shared ownership, including failed setup. Upload,
coefficient replacement, sorting, distributed conversion, and resizing are
rejected while attached. Setup requiring scaling, column reordering, or diagonal
insertion is rejected. These guards do not imply that all solver configurations
have been qualified.

## Build and qualification

After configuring AMGX, build with your own build directory:

```sh
cmake --build /path/to/AMGX/build \
  --target amgxsh amgx_tests_launcher --parallel 16
```

Then run `amgx_tests_launcher BorrowedCSR` with the matching CUDA libraries, followed
by the existing BorrowedVectors suite. Python tests and the profiling fixture are
described in the companion PyAMGX repository's `docs/borrowed_csr.md`.

Qualification evidence includes: successful native build, native and Python tests,
legacy/PTDS/nonblocking producer and independent consumer ordering, pointer
identity before/after setup and solve, independent residuals, unchanged CSR
contents, owner retention and safe error cleanup, representative AMG setup, and
Nsight confirmation that attachment/solve never stage the fine CSR arrays.
