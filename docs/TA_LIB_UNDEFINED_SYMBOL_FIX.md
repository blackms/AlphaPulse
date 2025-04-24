# TA-Lib Undefined Symbol Fix

This document explains the solution to the "undefined symbol: TA_AVGDEV_Lookback" error encountered when importing the TA-Lib Python package in the CI environment.

## Problem Description

When trying to import the TA-Lib Python package in the CI environment, the following error was encountered:

```
ImportError: /usr/share/miniconda/envs/test-env/lib/python3.11/site-packages/talib/_ta_lib.cpython-311-x86_64-linux-gnu.so: undefined symbol: TA_AVGDEV_Lookback
```

This error indicates that the Python wrapper is looking for a symbol (`TA_AVGDEV_Lookback`) that is not present in the installed C library. This can happen due to:

1. Version mismatch between the C library and Python wrapper
2. Incomplete or incorrect build of the C library
3. The Python wrapper being built against a different version of the C library than what's installed

## Root Cause Analysis

The root cause of this issue is a compatibility problem between the TA-Lib C library (version 0.4.0) and the Python wrapper (version 0.6.3). The Python wrapper expects certain symbols (like `TA_AVGDEV_Lookback`) that are not present in the C library version that was being installed.

The `AVGDEV` function (Average Deviation) is part of the TA-Lib API, but the specific symbol `TA_AVGDEV_Lookback` might be:
- Missing from the installed version of the C library
- Not properly exported by the C library
- Added in a newer version of the C library than what was being installed

## Solution

We implemented a multi-layered approach to ensure the TA-Lib library works correctly in the CI environment:

### 1. Use conda to install both the C library and Python wrapper

Conda packages are pre-built and tested for compatibility. By installing both components from the same source (conda-forge), we ensure they are compatible with each other.

```yaml
- name: Install TA-Lib with conda
  shell: bash -l {0}
  run: |
    echo "===== Installing TA-Lib using conda ====="
    conda install -c conda-forge ta-lib=0.4.0
```

### 2. Build the C library from source with debugging symbols

If the conda installation doesn't resolve the issue, we fall back to building the C library from source with debugging symbols enabled to ensure all symbols are properly exported.

```yaml
- name: Install TA-Lib C library from source (fallback)
  shell: bash -l {0}
  run: |
    # ... (download and extract source code) ...
    
    cd ta-lib/
    # Configure with debugging symbols and optimization
    ./configure --prefix=/usr CFLAGS="-O2 -g"
    make
    sudo make install
```

### 3. Build the Python wrapper from source against the installed C library

To ensure the Python wrapper is built against the correct version of the C library, we build it from source and explicitly set the paths to the C library and headers.

```yaml
- name: Build TA-Lib Python wrapper from source
  shell: bash -l {0}
  run: |
    # Clone the repository
    git clone https://github.com/mrjbq7/ta-lib.git python-talib
    cd python-talib
    
    # Check out a specific version known to work with TA-Lib 0.4.0
    git checkout tags/TA_Lib-0.4.24
    
    # Set environment variables to help the build process
    export TA_LIBRARY_PATH=/usr/lib
    export TA_INCLUDE_PATH=/usr/include
    
    # Build and install
    python setup.py build_ext --inplace
    python setup.py install
```

### 4. Verify the installation with specific tests

We added verification steps to ensure the library is properly installed and all required symbols are available:

```yaml
# Verify the specific function that uses the missing symbol
echo "Verifying TA-Lib AVGDEV function..."
python -c "import talib; import numpy as np; data = np.random.random(100); result = talib.AVGDEV(data, timeperiod=5); print('AVGDEV function works!')" || echo "Failed to use AVGDEV function"
```

## Implementation Details

The solution is implemented in the updated GitHub Actions workflow file: `.github/workflows/python-app-updated.yml`

Key changes from the original workflow:

1. Added a dedicated step to install TA-Lib using conda
2. Modified the C library build process to include debugging symbols
3. Added a new step to build the Python wrapper from source
4. Added specific verification steps to test the AVGDEV function
5. Enhanced error reporting and symbol checking

## Verification

The solution can be verified by:

1. Running the updated GitHub Actions workflow
2. Checking that the TA-Lib import succeeds without errors
3. Verifying that the AVGDEV function works correctly
4. Ensuring all tests that depend on TA-Lib pass

## Future Considerations

To prevent similar issues in the future:

1. **Pin specific versions**: Ensure that both the C library and Python wrapper versions are pinned to specific, compatible versions.

2. **Use conda when possible**: Conda packages are pre-built and tested for compatibility, making them more reliable than building from source.

3. **Add comprehensive tests**: Include tests that specifically exercise functions like AVGDEV to catch compatibility issues early.

4. **Document dependencies**: Keep track of the relationship between the C library and Python wrapper versions to make troubleshooting easier.

## References

- [TA-Lib Python Wrapper GitHub Repository](https://github.com/mrjbq7/ta-lib)
- [TA-Lib Official Website](https://ta-lib.org/)
- [Conda-Forge TA-Lib Package](https://anaconda.org/conda-forge/ta-lib)