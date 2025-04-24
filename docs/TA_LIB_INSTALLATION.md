# TA-Lib Installation Guide

This document provides guidance on installing the TA-Lib library, which is required for technical analysis functionality in AlphaPulse.

## Overview

TA-Lib (Technical Analysis Library) is a widely used open-source library that provides technical analysis functions. It consists of two main components:

1. **C/C++ library**: The core implementation written in C
2. **Python wrapper**: A Python package that provides bindings to the C library

The Python package (`ta-lib`) requires the C library to be installed on the system before it can be built and used.

## Installation Methods

### Method 1: Using Package Managers (Recommended for Development)

#### On Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install -y build-essential
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
```

#### On macOS (using Homebrew):
```bash
brew install ta-lib
```

#### On Windows:
Download the pre-built binary from [TA-Lib's SourceForge page](http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-msvc.zip), extract it, and add the library location to your PATH.

### Method 2: Using Conda (Alternative)

```bash
conda install -c conda-forge ta-lib
```

Note: The conda method installs both the C library and the Python package, but may not work in all environments.

## Installation in CI/CD Environments

For CI/CD environments like GitHub Actions, we use a comprehensive approach to ensure TA-Lib is properly installed and accessible:

1. Install the C library from source with extensive verification
2. Update the linker cache and set environment variables
3. Create symbolic links if needed
4. Use conda as a fallback
5. Explicitly attempt to install the Python package with verbose output

This robust approach is implemented in our GitHub Actions workflow (`.github/workflows/python-app.yml`):

```yaml
- name: Install TA-Lib C library
  shell: bash -l {0}
  run: |
    echo "===== Installing TA-Lib C library ====="
    # Install build dependencies
    echo "Installing build dependencies..."
    sudo apt-get update
    sudo apt-get install -y build-essential wget pkg-config
    
    # Download and install TA-Lib C library
    echo "Downloading TA-Lib source..."
    wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
    tar -xzf ta-lib-0.4.0-src.tar.gz
    
    echo "Building and installing TA-Lib..."
    cd ta-lib/
    ./configure --prefix=/usr
    make
    sudo make install
    cd ..
    
    # Update linker cache
    echo "Updating linker cache..."
    sudo ldconfig
    
    # Verify installation
    echo "Verifying installation..."
    find /usr -name "libta_lib*" || echo "No libta_lib files found in /usr"
    ls -la /usr/lib/libta_lib* || echo "No libta_lib files found in /usr/lib"
    
    # Set environment variables to help the linker find the library
    echo "Setting environment variables..."
    echo "LD_LIBRARY_PATH=/usr/lib:/usr/local/lib:$LD_LIBRARY_PATH" >> $GITHUB_ENV
    echo "LIBRARY_PATH=/usr/lib:/usr/local/lib:$LIBRARY_PATH" >> $GITHUB_ENV
    echo "CPATH=/usr/include:$CPATH" >> $GITHUB_ENV
    
    # Create symbolic links if needed
    echo "Creating symbolic links for compatibility..."
    if [ -f "/usr/lib/libta_lib.so" ] && [ ! -f "/usr/lib/libta-lib.so" ]; then
      sudo ln -s /usr/lib/libta_lib.so /usr/lib/libta-lib.so
      echo "Created symbolic link from libta_lib.so to libta-lib.so"
    fi
    
    # Verify the library can be found by the linker
    echo "Checking if library can be found by the linker..."
    ldconfig -p | grep ta_lib || echo "TA-Lib not found in linker cache"
    
    echo "TA-Lib C library installation completed"

- name: Install dependencies
  shell: bash -l {0}
  run: |
    echo "===== Installing Python dependencies ====="
    
    # Try both approaches for TA-Lib installation
    echo "Installing TA-Lib from conda-forge as a fallback..."
    conda install -c conda-forge ta-lib
    
    # ... other dependencies ...
    
    # Explicitly try to install ta-lib with debug output
    echo "Attempting to install ta-lib Python package directly..."
    pip install --verbose ta-lib
    
    # Verify ta-lib can be imported
    echo "Verifying ta-lib import..."
    python -c "import talib; print('TA-Lib successfully imported')" || echo "Failed to import talib"
```

### Key Improvements in the CI Installation Process

1. **Verbose Logging**: Added detailed logging at each step to identify exactly where any failures occur.

2. **Linker Cache Update**: Added `sudo ldconfig` to update the dynamic linker run-time bindings after installation.

3. **Environment Variables**: Set `LD_LIBRARY_PATH`, `LIBRARY_PATH`, and `CPATH` to help the compiler and linker find the library and headers.

4. **Symbolic Links**: Created a symbolic link from `libta_lib.so` to `libta-lib.so` if needed, as some builds might look for the library with a different name.

5. **Verification Steps**: Added multiple verification steps to confirm the library is installed and accessible:
   - Using `find` to locate all instances of the library
   - Checking the linker cache with `ldconfig -p`
   - Attempting to import the Python package

6. **Multiple Installation Methods**: Tried multiple approaches to install the Python package:
   - Via conda-forge
   - Via pip with verbose output
   - As part of the project dependencies

## Installation in Docker

In our Docker environment, we install the necessary build tools but rely on pip to install the Python package:

```dockerfile
# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*
```

## Troubleshooting

### Common Issues

1. **Missing C Library**:
   ```
   error: command '/usr/bin/gcc' failed with exit code 1
   ```
   Solution: Install the C library as described above.

2. **Linker Cannot Find Library**:
   ```
   /usr/share/miniconda/envs/test-env/compiler_compat/ld: cannot find -lta-lib: No such file or directory
   ```
   Solution:
   - Update the linker cache with `sudo ldconfig`
   - Set environment variables: `LD_LIBRARY_PATH=/usr/lib:/usr/local/lib:$LD_LIBRARY_PATH`
   - Create a symbolic link if needed: `sudo ln -s /usr/lib/libta_lib.so /usr/lib/libta-lib.so`

3. **Outdated Build Tools**:
   ```
   error: command 'x86_64-linux-gnu-gcc' failed with exit code 1
   ```
   Solution: Ensure you have the latest build-essential package installed.

4. **Path Issues on Windows**:
   ```
   fatal error C1083: Cannot open include file: 'ta-lib.h': No such file or directory
   ```
   Solution: Ensure the TA-Lib headers are in your include path.

### Debugging TA-Lib Installation

If you're having issues with TA-Lib installation, these commands can help diagnose the problem:

1. **Find all instances of the library**:
   ```bash
   find / -name "libta_lib*" 2>/dev/null
   ```

2. **Check if the library is in the linker cache**:
   ```bash
   ldconfig -p | grep ta_lib
   ```

3. **Verify the library can be loaded**:
   ```bash
   ldd /path/to/libta_lib.so
   ```

4. **Check environment variables**:
   ```bash
   echo $LD_LIBRARY_PATH
   echo $LIBRARY_PATH
   echo $CPATH
   ```

5. **Verify Python package installation**:
   ```bash
   pip show ta-lib
   python -c "import talib; print(talib.__file__)"
   ```

## References

- [TA-Lib Official Website](https://ta-lib.org/)
- [TA-Lib Python Wrapper on GitHub](https://github.com/mrjbq7/ta-lib)
- [TA-Lib on SourceForge](https://sourceforge.net/projects/ta-lib/)