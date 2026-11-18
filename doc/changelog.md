# Changelog

## 0.2.1

### Changed
- Default the CLI epoch budget to `N * nD`: feature dimensions times independent datasets/trials. Recompute it for every dimension in `--NCapacity`; keep explicit positive `--max-epochs` values fixed.
- Reject automatic epoch-budget overflow instead of wrapping the multiplication.

### Tests
- Cover option-order independence, overrides, invalid zero, overflow, and the effective epoch limit in each swept dimension's CSV.
- Keep generated CTest plots and logs in the build tree rather than the source tree.
