# slow5-dorado

This is our fork of [Dorado](https://github.com/nanoporetech/dorado) that supports (S/BLOW5)[https://www.nature.com/articles/s41587-021-01147-4].
Dorado is a high-performance, easy-to-use, open source basecaller for Oxford Nanopore reads.

## Features

* One executable with sensible defaults, automatic hardware detection and configuration.
* Runs on Apple silicon (M1/2 family) and Nvidia GPUs including multi-GPU with linear scaling.
* Modified basecalling (Remora models).
* Duplex basecalling.
* [S/BLOW5](https://www.nature.com/articles/s41587-021-01147-4) support for highest basecalling performance.
* Based on libtorch, the C++ API for pytorch.
* Multiple custom optimisations in CUDA and Metal for maximising inference performance.

If you encounter any problems building or running sow5-dorado please [report an issue](https://github.com/hiruna72/slow5-dorado/issues).

## Installation

Binaries are provided for Linux x64 under [Relases](https://github.com/hiruna72/slow5-dorado/releases/). 

```
wget <link> -O slow5-dorado.tar.gz
tar xf slow5-dorado.tar.gz
cd slow5-dorado/bin
./slow5-dorado --version
```

## Running

To run slow5-dorado, download a model and point it to S/BLOW5 files.

```

$ slow5-dorado download --model dna_r10.4.1_e8.2_260bps_hac@v4.0.0
$ slow5-dorado basecaller dna_r10.4.1_e8.2_260bps_hac@v4.0.0 BLOW5s/ > calls.sam # blow5 directory
./slow5-dorado basecaller dna_r10.4.1_e8.2_260bps_hac@v4.0.0 merged.blow5 >calls.samq # a single BLOW5 file
```

To call modifications simply add `--modified-bases`.

```
$ slow5-dorado basecaller dna_r10.4.1_e8.2_260bps_hac@v4.0.0 BLOW5s/ --modified-bases 5mCG_5hmCG > calls.sam
```

For unaligned BAM output, dorado output can be piped to BAM using samtoools:

```
$ slow5-dorado basecaller dna_r10.4.1_e8.2_260bps_hac@v4.0.0 BLOW5s/ | samtools view -Sh > calls.bam
```

Stereo Duplex Calling:

```
$ slow5-dorado duplex dna_r10.4.1_e8.2_260bps_sup@v4.0.0 BLOW5s/ --pairs pairs.txt > duplex.sam
```

## Platforms

Dorado has been tested on the following systems:

| Platform | GPU/CPU                      |
| -------- | ---------------------------- |
| Windows  | (G)V100, A100                |
| Apple    | M1, M1 Pro, M1 Max, M1 Ultra |
| Linux    | (G)V100, A100                |

Systems not listed above but which have Nvidia GPUs with >=8GB VRAM and architecture from Volta onwards have not been widely tested but are expected to work. If you encounter problems with running on your system please [report an issue](https://github.com/nanoporetech/dorado/issues)

## Roadmap

Dorado is still in alpha stage and not feature-complete, the following features form the core of our roadmap:

1. DNA Barcode multiplexing
2. Alignment *(output aligned BAMs)*.
3. Python API

## Performance tips

1. slow5-dorado requires S/BLOW5 file input. Please [convert your Fast5 files](https://github.com/hasindu2008/slow5tools) before basecalling.
2. Dorado will automatically detect your GPUs' free memory and select an appropriate batch size.
3. Dorado will automatically run in multi-GPU (`'cuda:all'`) mode. If you have a hetrogenous collection of GPUs select the faster GPUs using the `--device` flag (e.g `--device "cuda:0,2`). Not doing this will have a detrimental impact on performance.

## Available basecalling models

To download all available dorado models run:

```
$ dorado download --model all
```

The following models are currently available:

* dna_r10.4.1_e8.2_260bps_fast@v4.0.0
* dna_r10.4.1_e8.2_260bps_hac@v4.0.0
* dna_r10.4.1_e8.2_260bps_sup@v4.0.0
* dna_r10.4.1_e8.2_400bps_fast@v4.0.0
* dna_r10.4.1_e8.2_400bps_hac@v4.0.0
* dna_r10.4.1_e8.2_400bps_sup@v4.0.0
* dna_r10.4.1_e8.2_260bps_fast@v3.5.2
* dna_r10.4.1_e8.2_260bps_hac@v3.5.2
* dna_r10.4.1_e8.2_260bps_sup@v3.5.2
* dna_r10.4.1_e8.2_400bps_fast@v3.5.2
* dna_r10.4.1_e8.2_400bps_hac@v3.5.2
* dna_r10.4.1_e8.2_400bps_sup@v3.5.2
* dna_r9.4.1_e8_fast@v3.4
* dna_r9.4.1_e8_hac@v3.3
* dna_r9.4.1_e8_sup@v3.3

## Developer quickstart

### Linux dependencies
The following packages are necessary to build dorado in a barebones environment (e.g. the official ubuntu:jammy docker image)
```
apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        git \
        ca-certificates \
        build-essential \
        nvidia-cuda-toolkit \
        libhdf5-dev \
        libssl-dev \
        libzstd-dev \
        cmake \
        autoconf \
        automake
```

### Clone and build
The commands below will build dorado and install it in `/opt`.
 `NUM_THREADS` controls the number of threads that cmake uses to compile dorado. It can be set to a value higher than "1", but using too many threads can use all available RAM and cause compilation to fail. Peak memory usage seems to be 1-2GB per thread.

```
export NUM_THREADS=1
git clone https://github.com/nanoporetech/dorado.git /dorado
cd /dorado
cmake -S . -B cmake-build -DCMAKE_CUDA_COMPILER=nvcc
cmake --build cmake-build --config Release --parallel $NUM_THREADS
ctest --test-dir cmake-build
cmake --install cmake-build --prefix /opt
rm -rf /dorado
cd /
```

### Pre commit

The project uses pre-commit to ensure code is consistently formatted, you can set this up using pip:

```bash
$ pip install pre-commit
$ pre-commit install
```

### Licence and Copyright
(c) 2022 Oxford Nanopore Technologies Ltd.

Dorado is distributed under the terms of the Oxford Nanopore
Technologies, Ltd.  Public License, v. 1.0.  If a copy of the License
was not distributed with this file, You can obtain one at
http://nanoporetech.com
