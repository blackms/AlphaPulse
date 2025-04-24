# TA-Lib Installation Guide for CI Environment

This document explains how to install TA-Lib in the CI environment using Poetry, addressing the "package not found" error previously encountered with conda.

## Problem Description

When trying to install TA-Lib using conda in the CI environment, the following error was encountered:

```
PackagesNotFoundError: The following packages are not available from current channels:

  - ta-lib=0.4.0

Current channels:

  - https://conda.anaconda.org/conda-forge
  - defaults
  - https://repo.anaconda.com/pkgs/main
  - https://repo.anaconda.com/pkgs/r
```

## Solution

We implemented a solution that uses Poetry instead of conda for managing Python dependencies, while still building the TA-Lib C library from source:

### 1. Configure Poetry in pyproject.toml

We updated the `pyproject.toml` file to use Poetry as the build system and added TA-Lib as a dependency:

```toml
[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"

[tool.poetry]
name = "alpha-pulse"
version = "0.1.0"
description = "AlphaPulse Trading System"
authors = ["AlphaPulse Team"]
readme = "README.md"
packages = [{include = "alpha_pulse", from = "src"}]

[tool.poetry.dependencies]
python = ">=3.11,<3.12"
ta-lib = "^0.6.3"
# ... other dependencies ...
```

### 2. Install TA-Lib C Library from Source

The TA-Lib Python package requires the C library to be installed first. We build it from source:

```yaml
- name: Install TA-Lib C library from source
  run: |
    echo "===== Installing TA-Lib C library from source ====="
    # Install build dependencies
    sudo apt-get update
    sudo apt-get install -y build-essential wget pkg-config
    
    # Download and install TA-Lib C library
    wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
    tar -xzf ta-lib-0.4.0-src.tar.gz
    
    cd ta-lib/
    ./configure --prefix=/usr CFLAGS="-O2 -g"
    make
    sudo make install
    cd ..
    
    # Update linker cache
    sudo ldconfig
    
    # Set environment variables to help the linker find the library
    echo "LD_LIBRARY_PATH=/usr/lib:/usr/local/lib:$LD_LIBRARY_PATH" >> $GITHUB_ENV
    echo "LIBRARY_PATH=/usr/lib:/usr/local/lib:$LIBRARY_PATH" >> $GITHUB_ENV
    echo "CPATH=/usr/include:$CPATH" >> $GITHUB_ENV
```

### 3. Install Python Dependencies with Poetry

After the C library is installed, we use Poetry to install all Python dependencies, including the TA-Lib Python wrapper:

```yaml
- name: Install dependencies with Poetry
  run: |
    echo "===== Installing Python dependencies with Poetry ====="
    poetry install --no-interaction
    
    # Verify the TA-Lib installation
    poetry run python -c "import talib; print(f'TA-Lib version: {talib.__version__}')"
```

### 4. Verify the Installation

We added verification steps to ensure the library is properly installed and all required functions work:

```yaml
# Verify the specific function that uses the missing symbol
poetry run python -c "import talib; import numpy as np; data = np.random.random(100); result = talib.AVGDEV(data, timeperiod=5); print('AVGDEV function works!')"
```

## Implementation Details

The solution is implemented in the GitHub Actions workflow file: `.github/workflows/python-app.yml`

Key changes from the original workflow:

1. Replaced conda with Poetry for dependency management
2. Updated the pyproject.toml file to use Poetry as the build system
3. Kept the C library installation from source
4. Added verification steps to ensure TA-Lib works correctly

## Verification

The solution can be verified by:

1. Running the updated GitHub Actions workflow
2. Checking that the TA-Lib import succeeds without errors
3. Verifying that the AVGDEV function works correctly
4. Ensuring all tests that depend on TA-Lib pass

## Future Considerations

To prevent similar issues in the future:

1. **Pin specific versions**: Ensure that both the C library and Python wrapper versions are pinned to specific, compatible versions.

2. **Use Poetry for dependency management**: Poetry provides more reliable dependency resolution than conda for Python packages.

3. **Add comprehensive tests**: Include tests that specifically exercise functions like AVGDEV to catch compatibility issues early.

4. **Document dependencies**: Keep track of the relationship between the C library and Python wrapper versions to make troubleshooting easier.

## References

- [TA-Lib Python Wrapper GitHub Repository](https://github.com/mrjbq7/ta-lib)
- [TA-Lib Official Website](https://ta-lib.org/)
- [Poetry Documentation](https://python-poetry.org/docs/)