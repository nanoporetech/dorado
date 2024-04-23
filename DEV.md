## Setup notes

Dorado requires CUDA 11.8 on linux platforms. If the system you are running on does not have CUDA 11.8 installed, and you do not have sudo privileges, you can install locally from a run file as follows:

```
$ wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
$ sh cuda_11.8.0_520.61.05_linux.run --silent --toolkit --toolkitpath=${PWD}/cuda11.8
```

In this case, cmake should be invoked with `CUDAToolkit_ROOT` in order to tell the build process where to find CUDA:

```
$ cmake -DCUDAToolkit_ROOT=~/dorado_deps/cuda11.8 -S . -B cmake-build
```

Note that a [suitable NVIDIA driver](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#id3) will be required in order to run dorado.

All other dependencies will be fetched automatically by the cmake build process.

If libtorch is already downloaded on the host system and you do not wish the build process to re-download it, you can specify `DORADO_LIBTORCH_DIR` to cmake, in order to specify where the build process should locate it.  For example:

```
$ cmake -DDORADO_LIBTORCH_DIR=/usr/local/libtorch -S . -B cmake-build
```

### OSX

On OSX, version 2.69 of autoconf is required:

```bash
$ brew uninstall autoconf # not necessary if autoconf is not already installed
$ brew install autoconf@2.69
$ brew link autoconf@2.69
```

The following other packages need to be available as well
```bash
brew install openssl zstd
```
