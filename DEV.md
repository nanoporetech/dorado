# Developer Quickstart

## Dependencies

### Linux dependencies

This project requires `cmake 3.25` or higher. This can be installed via binary download from [cmake.org](https://cmake.org/download/) or using `python3-venv`: 

```bash
apt update
apt install python3-venv -y --no-install-recommends
python3 -m venv venv
. venv/bin/activate
pip install "cmake>=3.25"
```

The following packages are necessary to build Dorado in a barebones environment (e.g. the official `ubuntu:noble` Docker image).

```bash
apt update
apt install -y --no-install-recommends \
        build-essential \
        git \
        libz-dev \
        autoconf \
        automake \
        ca-certificates
```
Dorado requires a minimum of gcc-11 to build. This is included in the `build-essential` package on Noble, but may need to be installed separately on other versions.

Dorado requires the CUDA toolkit on Linux platforms. The minimum version is 11.8, but for optimal performance 12.8 should be used. 
If the system you are running on does not have CUDA 12.8 installed, and you do not have sudo privileges, you can install locally from a run file as follows:

```bash
apt install wget libxml2 -y --no-install-recommends
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run
sh cuda_12.8.0_570.86.10_linux.run --silent --toolkit --toolkitpath=${HOME}/dorado_deps/cuda12.8
```

In this case, cmake should be invoked with `CUDAToolkit_ROOT` in order to tell the build process where to find CUDA:

```bash
cmake -DCUDAToolkit_ROOT=${HOME}/dorado_deps/cuda12.8 -S . -B cmake-build
```

Note that a [suitable NVIDIA driver](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#id3) will be required in order to run Dorado.

All other dependencies will be fetched automatically by the cmake build process.  

If libtorch is already downloaded on the host system and you do not wish the build process to re-download it, you can specify `DORADO_LIBTORCH_DIR` to cmake, in order to specify where the build process should locate it.  For example:

```bash
cmake -DDORADO_LIBTORCH_DIR=/usr/local/libtorch -S . -B cmake-build
```

### OSX dependencies

On OSX, the following packages need to be available:

```bash
brew install autoconf openssl
```

### Clone and build

```bash
git clone https://github.com/nanoporetech/dorado.git dorado --recurse-submodules
cd dorado
cmake -S . -B cmake-build
cmake --build cmake-build --config Release -j
ctest --test-dir cmake-build
```

The `-j` flag will use all available threads to build Dorado and usage is around 1-2 GB per thread. If you are constrained
by the amount of available memory on your system, you can lower the number of threads i.e. `-j 4`.

After building, you can run Dorado from the build directory `./cmake-build/bin/dorado` or install it somewhere else on your
system i.e. `/opt` *(note: you will need the relevant permissions for the target installation directory)*.

```bash
cmake --install cmake-build --prefix /opt
```

### Pre-commit

The project uses pre-commit to ensure code is consistently formatted; you can set this up using pip:

```bash
$ pip install pre-commit
$ pre-commit install
```
