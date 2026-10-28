# Path to the first stable release

Review: 2026-09-24. The current version remains `0.1.0a1` and is not being published
as part of this task. “Stable” must describe a verified contract and matrix;
it does not require winning every benchmark or achieving full NumPy compatibility.

## Work estimate

After the current three iterations, the estimate is **2–4 hardening iterations
for a release candidate with a limited scope**, provided no structural defects
emerge. This is a work estimate, not a committed date.
A beta with external users must precede the declaration of stability;
its duration depends on obtaining representative tests and reports.

An initial `0.1.0` can stabilize the documented API and certify the
local configuration while explicitly keeping other hardware experimental.
Claiming stability across vendors still requires evidence from physical
AMD/Intel hardware; PoCL helps check OpenCL portability but does not replace
those GPUs. Access to that equipment is an external dependency, not
something another NVIDIA optimization can resolve.

## Exit criteria

| Area | Available evidence | Remaining before the candidate |
| --- | --- | --- |
| API | NumPy subset, explicit backend, compilation and documented limits | Review names, errors, array/aliasing contracts, RNG and compatibility/deprecation policy; freeze the release scope |
| Correctness | Local CPU/GPU suites, irregular/extreme inputs, lifetime and concurrency | Repeat against the candidate artifact; classify known numerical failures and fix those that violate the declared contract |
| Accuracy | float64/Decimal80 references and visible alerts | Define guarantees per operation and limits for accumulation, cancellation, subnormals and FMA; do not treat “finite result” as numerical approval |
| Installation | Independent extras, minimal CPU wheel and local builds | Smoke tests from artifacts in clean environments for each extra, missing dependencies and minimum Python version; check missing driver/compiler messages |
| CI | CPU workflow on Windows/Linux/macOS, benchmark infrastructure and OpenCL PoCL defined | Actually run it remotely; verify installed artifacts in the matrix, not just their build; do not confuse syntax analysis with Python 3.10 execution |
| Performance | Results with copies, synchronization, errors and controls | Establish a short regression suite and record the candidate baseline; do not use microbenchmark improvements as the sole stability criterion |
| Traceability | Sources and hashes of local experiments | Create versioned history and a reproducible release: this checkout still has no commits and the project files are untracked |
| Distribution and support | MIT and technical documentation | Confirm publication name/account, changelog, contribution instructions, issue template and support matrix; publication is a separate action |

Backends may have different scopes if these are declared. Types that are not yet
implemented must not be advertised as supported. They do not automatically
block a limited release
if they fail explicitly; an advertised operation that returns incorrect results
does. Cancellation alerts require numerical judgment and
documentation, not simply raising tolerances until the suite passes.

## Proposed sequence

1. **Freeze the contract and close correctness gaps:** review API/capabilities, classify
   alerts and resolve inconsistencies. Keep sparse operations, autodiff, solvers as
   an API and full NumPy compatibility outside the scope.
2. **Harden distribution and CI:** installation by extras, clean artifacts,
   actual CPU/PoCL matrix, resources and initialization errors. Version the code
   and establish a reproducible candidate.
3. **External beta and candidate:** publish the exact verified matrix, collect
   real cases and failures, run the regression suite and resolve blocking issues.
   One extra round is reserved for findings from that validation.

Performance will continue improving after a stable release. The following
extensions should not delay it on their own: more specialized kernels,
general autotuning, new vendors awaiting measurement and complete parity
between backends. It should be delayed if the advertised semantics fail or
the candidate installation and tests cannot be reproduced.
