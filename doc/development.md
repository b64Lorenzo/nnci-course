# Development

Requires CMake 3.20+ and a C++20 compiler. Python 3 with Matplotlib is needed for plotting and plot tests.

```powershell
cmake -S . -B build -A x64
cmake --build build --config Release
py -3 -m pip install -r requirements-plot.txt
ctest --test-dir build -C Release --output-on-failure
```

For single-configuration generators, set `-DCMAKE_BUILD_TYPE=Release` and omit `--config Release`.

## Layout

- `include/nnci/` — public library headers.
- `src/` — perceptron and study implementation.
- `src/app/` — private CLI, option parsing, output, and plotting code.
- `scripts/` — Python plotting scripts.
- `tests/` — C++ and Python tests.

`VERSION` is the release version. GitHub Actions builds, tests, and packages the Windows release.
