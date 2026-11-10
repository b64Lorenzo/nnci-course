# Development and testing

## Requirements

- CMake 3.20 or newer.
- A C++20-capable compiler. Windows CI and release builds use Visual Studio/MSVC.
- Python 3 and Matplotlib for tests that exercise automatic plotting or for manual plot generation. Install the declared dependency with `py -3 -m pip install -r requirements-plot.txt` on Windows.

## Configure, build, and test

Windows multi-configuration example:

```powershell
cmake -S . -B build -A x64
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

For a single-configuration generator, configure with `-DCMAKE_BUILD_TYPE=Release`, then run `cmake --build build` and `ctest --test-dir build --output-on-failure`.

The test suite covers perceptron behavior, version output and Windows executable metadata, capacity CSV/plot generation, combined N-sweep plotting, strength plotting from an unrelated working directory, capacity plot input validation, and explicit P-range handling in an N sweep. Always inspect CTest's summary and require all registered tests to pass.

## Project layout

```text
include/nnci/       Public logging and perceptron interfaces
src/main.cpp        CLI parser, command dispatch, CSV writing, plot invocation
src/assignment1_study.cpp
                    Random dataset generation and repeated studies
src/rosenblatt_perceptron.cpp
                    Rosenblatt training loop
scripts/            Capacity and strength plotting scripts
tests/              C++ and Python regression tests
 doc/               User, algorithm, development, and troubleshooting guides
 assignments/       Assignment I handout and assignment-specific notes
```

CMake copies the plotting scripts beside the built executable. At runtime the CLI finds scripts relative to `nnci.exe`; results and logs remain relative to the caller's working directory. Avoid moving an executable without its `scripts/` directory.

## Release packaging

The version in `VERSION` is the source of truth for CMake version metadata and release tags. The GitHub Actions workflow runs when `VERSION` changes on `main` (or by manual dispatch). The workflow validates the semantic version and tag, configures and builds a Windows x64 Release executable, runs CTest and smoke tests, then creates `dist/nnci-windows-x64.zip`.

The archive is expected to contain `nnci.exe`, both plot scripts under `scripts/`, `requirements-plot.txt`, root `README.md`, a Windows `README.txt` quick-start, `doc/`, `LICENSE`, and `THIRD_PARTY_NOTICES.md`. The workflow extracts and checks the package, then runs a strength plotting smoke test from a separate working directory before publishing the version tag and ZIP asset.

To prepare a release, update `VERSION`, keep the root README and documentation accurate for the CLI, verify the package list in `.github/workflows/release.yml`, run the local Release build/tests, and review the diff. Do not reuse a published version tag. Treat the published ZIP as the deliverable; a loose executable is not the complete plotting-enabled distribution.

## Licensing and generated files

The MIT license applies to original project code and general project documentation. Course handouts retain their upstream terms; consult `THIRD_PARTY_NOTICES.md` before redistribution. Generated build output, logs, and experiment results are excluded from Git by `.gitignore`.
