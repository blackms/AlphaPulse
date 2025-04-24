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

For CI/CD environments like GitHub Actions, we use a combination of approaches:

1. Install the C library from source
2. Use conda as a fallback

This is implemented in our GitHub Actions workflow (`.github/workflows/python-app.yml`):

```yaml
- name: Install TA-Lib C library
  shell: bash -l {0}
  run: |
    # Install build dependencies
    sudo apt-get update
    sudo apt-get install -y build-essential wget
    
    # Download and install TA-Lib C library
    wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
    tar -xzf ta-lib-0.4.0-src.tar.gz
    cd ta-lib/
    ./configure --prefix=/usr
    make
    sudo make install
    cd ..
    
    # Verify installation
    ls -la /usr/lib/libta_lib*
    echo "TA-Lib C library installed successfully"

- name: Install dependencies
  shell: bash -l {0}
  run: |
    # Try both approaches for TA-Lib installation
    # 1. Install TA-Lib from conda-forge as a fallback
    conda install -c conda-forge ta-lib
    
    # Continue with other dependencies...
```

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

2. **Outdated Build Tools**:
   ```
   error: command 'x86_64-linux-gnu-gcc' failed with exit code 1
   ```
   Solution: Ensure you have the latest build-essential package installed.

3. **Path Issues on Windows**:
   ```
   fatal error C1083: Cannot open include file: 'ta-lib.h': No such file or directory
   ```
   Solution: Ensure the TA-Lib headers are in your include path.

## References

- [TA-Lib Official Website](https://ta-lib.org/)
- [TA-Lib Python Wrapper on GitHub](https://github.com/mrjbq7/ta-lib)
- [TA-Lib on SourceForge](https://sourceforge.net/projects/ta-lib/)