# Changelog

## 0.3.3

### Changed
- Replace fixed CUDA buffer/dataset caps with a live VRAM budget and kernel-occupancy-based scheduling. Leave at least 256 MiB or 25% of currently free VRAM unused, whichever is larger.
- Refresh limits between batches, pack by actual shape bytes, and validate direct backend callers before allocation. Free-memory queries are snapshots; competing GPU allocations can still cause a clear error rather than CPU fallback.
- Reduce epoch chunks from eight towards one for larger datasets, without changing synchronous batch learning, host RNG, result order, or CPU behavior.

### Tests
- GPU-independent reserve, low-memory, occupancy/grid, exact/one-over and overflow boundaries.
- Native/CLI adaptive batch bounds, mixed-shape tails, large-dataset launch counts, deterministic results and seeded comparisons with v0.3.2.

## 0.3.2

### Changed
- Replace CUDA's online updates with synchronous batch Hebbian learning: freeze epoch potentials, select `Eμ <= c`, reduce the selected vectors in parallel, and update weights once by their total divided by N.
- Recompute signed potentials in parallel and converge only when their minimum is strictly positive; empty selections and cancelling sums run to the epoch limit if strict separation is not reached.
- Use cooperative 256-thread dataset blocks, parallel feature/pattern reductions and at most eight epochs per launch. Existing mixed N/P/trial scheduling and CPU online behavior remain unchanged.
- Include potential and selection-mask buffers in the 64 MiB resident-device budget. CUDA results may now differ from CPU results by design.

### Tests
- Independent scalar batch oracle, hand-calculated updates, strict zero/negative-potential boundaries, cancellation, mixed N/P options, long loops, deterministic reruns, subnormals and numerical errors.
- Keep CPU-only rejection, CPU-parallel parity, CUDA CLI modes, per-N epoch defaults/overrides, mixed-case batches and tail coverage.

## 0.3.1

### Added
- Opt-in `--parallelize` for bounded CPU concurrency across independent `--capacity` and `--NCapacity` work; CPU execution remains sequential by default.
- CUDA batching across independent P/N/trial datasets, bounded to 1024 datasets and 64 MiB of device buffers, while preserving result order and CPU parity.

### Fixed
- Prevent floating-point roundoff at empirical rates 0 and 1 from producing negative plot error-bar lengths.

### Tests
- Cover CPU-parallel output parity, mixed-shape CUDA batches and chunk limits, and extreme-rate capacity plotting.

## 0.3.0

### Added
- Opt-in `--cuda` execution for single runs, capacity/N sweeps, strength studies, and validation; CPU remains the default.
- Real GPU parallelism across independent datasets with deterministic host RNG, ordered results, and sequential FP64 updates within each dataset.
- Bounded device batches and resumable, work-bounded kernels; explicit errors for unsupported builds/devices/drivers without CPU fallback.
- CUDA-capable Windows distribution with a static runtime, NVIDIA license, and architectures 75/80/86/89/90 plus PTX. No toolkit is needed at runtime; GPU execution requires a compatible NVIDIA driver.

### Fixed
- Reject non-finite thresholds with `InvalidOptions (11)` on both backends.

### Tests
- Exact native CPU/GPU comparisons for weights, epochs, updates, strengths, convergence, numeric errors, partial blocks, batch tails, and long-loop continuation.
- Byte-identical CLI CSVs in every mode, including automatic and explicit per-N epoch budgets.
- CPU-only builds and CPU-default execution with devices hidden; SDK checksum, size, and archive-path validation.
- Native execution verified on RTX 4070 Laptop GPU (sm_89); other packaged architectures are compiled, not hardware-verified. Small jobs can be slower on GPU.

## 0.2.1

### Changed
- Default the CLI epoch budget to `N * nD`: feature dimensions times independent datasets/trials. Recompute it for every dimension in `--NCapacity`; keep explicit positive `--max-epochs` values fixed.
- Reject automatic epoch-budget overflow instead of wrapping the multiplication.

### Tests
- Cover option-order independence, overrides, invalid zero, overflow, and the effective epoch limit in each swept dimension's CSV.
- Keep generated CTest plots and logs in the build tree rather than the source tree.
