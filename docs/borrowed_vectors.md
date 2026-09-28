# Borrowed device vectors

Implementation status (2026-09-28): the user-run build succeeds. Native borrowed
vector tests pass in dDDI and dFFI, and PyAMGX GPU tests and transfer profiles
validate these modes. The dDFI solve reaches the existing mixed-precision
SpMV rejection, which returns NOT_IMPLEMENTED. Mixed-mode solve qualification
is excluded from this milestone by the accepted scope; attachment support does
not imply mixed-mode solver support. The revised native mixed-mode regression
checks the expected error and passes with the rebuilt launcher. All three
native regressions pass; the agreed borrowed-vector milestone is complete.
See the companion PyAMGX repository's `docs/borrowed_vectors_validation.md` for evidence.

`AMGX_vector_attach` borrows an existing device allocation. It does not upload,
allocate a second vector, copy data, or transfer allocation ownership. The
application keeps the allocation alive until successful detach or destruction.
`AMGX_vector_get_attached_data` exposes the actual attached pointer and logical
byte count for identity checks. `AMGX_vector_detach` leaves an empty vector and
does not copy the result elsewhere. An attached solution is already in the
application's allocation.

The initial contract supports scalar real device modes dDDI, dDFI, and dFFI,
contiguous writable memory on the resource device, and a single-GPU unscaled
scalar matrix. Host/managed memory, distributed vector managers, overlapping
RHS/solution, incompatible sizes, and attempts to resize borrowed storage are
rejected. Attachment requires an empty, unattached vector. The native caller
provides an accurate capacity in bytes; as with other pointer-based C APIs,
AMGX cannot infer every producer's logical allocation bounds from a pointer.

The new internal storage adapter keeps its owning Thrust container private.
Pointers, iterators, references, size, and capacity all describe the borrowed
allocation when attached. Explicit copies into solver workspaces remain normal
algorithm operations; implicit replacement or resizing of attached storage is
forbidden. Copy construction creates an independent owning vector. Borrowing
does not promise allocation-free hierarchy construction or Krylov iterations.

## Ordering and mutation

Attach waits for the supplied producer stream. Use the CUDA Array Interface
stream encoding: 1 is the legacy stream, 2 is the per-thread default stream,
and larger integers are handles. Native value 0 explicitly asserts the caller
has already completed producer work. CUDA 13 validates stream device affinity.
No stream or allocation ownership is transferred.

After any subsequent external write, call `AMGX_vector_synchronize` with its
producer stream before AMGX accesses the vector. The PyAMGX wrapper handles
this before solves and zeroing. The caller must ensure its exported stream
orders all prior uses of the allocation, including work on other streams.
There must be no concurrent external access during an AMGX operation.

The tested CUDA-13 scalar FGMRES/NOSOLVER solve path executes on its
legacy stream. C API solves involving borrowed vectors explicitly synchronize
that stream before returning, including native solve failures. Detach and
destroy also wait for that stream. This is a synchronous zero-copy interface;
it does not inject an external execution stream into AMGX or advertise general
asynchronous execution. Existing native debug/error paths may still perform
their own device synchronization.

The regression gates are the `BorrowedVectors` native test and PyAMGX's
`test_vector_attach.py`. They cover owning/borrowed copy construction, pointer
identity, iterator writes, forbidden resizing, lifetime, repeated RHS changes,
three real modes, independent residuals, and legacy/PTDS/nonblocking streams.
Profiling is an additional gate before a performance or no-boundary-transfer
claim is published. Tests passing is not a promise that every AMGX solver or
preconditioner configuration has been qualified.

No matrix borrowing or BSR integration is included in this milestone. Existing
matrix upload APIs retain their copy semantics.

## Build

After configuring AMGX with CUDA, use your build directory:

```bash
cmake --build /path/to/AMGX/build \
  --target amgxsh amgx_tests_launcher --parallel 16
```

The vector storage header affects AMGX internals, so a full dependent rebuild
is required. No CMake cache replacement or AMGX installation is needed when
the wrapper links to this build directory and uses this source's headers.
Afterward, follow the PyAMGX `docs/borrowed_vectors.md` build/test commands.
