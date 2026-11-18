# Changelog

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
