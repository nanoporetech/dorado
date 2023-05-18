# Dorado

Dorado is a high-performance, easy-to-use, open source basecaller for Oxford Nanopore reads.

## Features

* One executable with sensible defaults, automatic hardware detection and configuration.
* Runs on Apple silicon (M1/2 family) and Nvidia GPUs including multi-GPU with linear scaling.
* Modified basecalling.
* Duplex basecalling.
* Support for aligned read output in SAM/BAM.
* [POD5](https://github.com/nanoporetech/pod5-file-format) support for highest basecalling performance.
* Based on libtorch, the C++ API for pytorch.
* Multiple custom optimisations in CUDA and Metal for maximising inference performance.

If you encounter any problems building or running Dorado please [report an issue](https://github.com/nanoporetech/dorado/issues).

## Installation

 - [dorado-0.2.4-linux-x64](https://cdn.oxfordnanoportal.com/software/analysis/dorado-0.3.0-linux-x64.tar.gz)
 - [dorado-0.2.4-linux-arm64](https://cdn.oxfordnanoportal.com/software/analysis/dorado-0.3.0-linux-arm64.tar.gz)
 - [dorado-0.2.4-osx-arm64](https://cdn.oxfordnanoportal.com/software/analysis/dorado-0.3.0-osx-arm64.tar.gz)
 - [dorado-0.2.4-win64](https://cdn.oxfordnanoportal.com/software/analysis/dorado-0.3.0-win64.zip)

## Platforms

Dorado is heavily-optimised for Nvidia A100 and H100 GPUs and will deliver maximal performance on systems with these GPUs.

Dorado has been tested extensively and supported on the following systems:

| Platform | GPU/CPU                      |
| -------- |------------------------------|
| Windows  | (G)V100, A100, H100          |
| Apple    | M1, M1 Pro, M1 Max, M1 Ultra |
| Linux    | (G)V100, A100, H100          |

Systems not listed above but which have Nvidia GPUs with >=8GB VRAM and architecture from Volta onwards have not been widely tested but are expected to work. If you encounter problems with running on your system please [report an issue](https://github.com/nanoporetech/dorado/issues)

## Roadmap

Dorado is still in alpha stage and not feature-complete, the following features form the core of our roadmap:

1. DNA Barcode multiplexing
2. Alignment *(output aligned BAMs)*.
3. Python API

## Performance tips

1. For optimal performance Dorado requires POD5 file input. Please [convert your Fast5 files](https://github.com/nanoporetech/pod5-file-format) before basecalling.
2. Dorado will automatically detect your GPUs' free memory and select an appropriate batch size.
3. Dorado will automatically run in multi-GPU (`'cuda:all'`) mode. If you have a hetrogenous collection of GPUs select the faster GPUs using the `--device` flag (e.g `--device "cuda:0,2`). Not doing this will have a detrimental impact on performance.

## Running

To run Dorado basecalling, download a model and point it to POD5 files _(Fast5 files are supported but will not be as performant)_.

```
$ dorado download --model dna_r10.4.1_e8.2_400bps_hac@v4.1.0
$ dorado basecaller dna_r10.4.1_e8.2_400bps_hac@v4.1.0 pod5s/ > calls.bam
```

To call modifications simply add `--modified-bases`.

```
$ dorado basecaller dna_r10.4.1_e8.2_400bps_hac@v4.1.0 pod5s/ --modified-bases 5mCG_5hmCG > calls.bam
```

To run Duplex basecalling run the command:

```
$ dorado duplex dna_r10.4.1_e8.2_400bps_sup@v4.1.0 pod5s/ > duplex.bam
```

Duplex pair detection and read splitting previously used a separate tool, but this is now integrated into Dorado.


## Available basecalling models

To download all available dorado models run:

```
$ dorado download --model all
```

Simplex models for our latesat released condition are V4.1.0 (4kHz)

* dna_r10.4.1_e8.2_260bps_fast@v4.1.0
* dna_r10.4.1_e8.2_260bps_hac@v4.1.0
* dna_r10.4.1_e8.2_260bps_sup@v4.1.0
* dna_r10.4.1_e8.2_400bps_fast@v4.1.0
* dna_r10.4.1_e8.2_400bps_hac@v4.1.0
* dna_r10.4.1_e8.2_400bps_sup@v4.1.0

The following models are also available:

**Simplex models:**

* dna_r9.4.1_e8_fast@v3.4
* dna_r9.4.1_e8_hac@v3.3
* dna_r9.4.1_e8_sup@v3.3
* dna_r9.4.1_e8_sup@v3.6
* dna_r10.4.1_e8.2_260bps_fast@v3.5.2
* dna_r10.4.1_e8.2_260bps_hac@v3.5.2
* dna_r10.4.1_e8.2_260bps_sup@v3.5.2
* dna_r10.4.1_e8.2_400bps_fast@v3.5.2
* dna_r10.4.1_e8.2_400bps_hac@v3.5.2
* dna_r10.4.1_e8.2_400bps_sup@v3.5.2
* dna_r10.4.1_e8.2_260bps_fast@v4.0.0
* dna_r10.4.1_e8.2_260bps_hac@v4.0.0
* dna_r10.4.1_e8.2_260bps_sup@v4.0.0
* dna_r10.4.1_e8.2_400bps_fast@v4.0.0
* dna_r10.4.1_e8.2_400bps_hac@v4.0.0
* dna_r10.4.1_e8.2_400bps_sup@v4.0.0
* dna_r10.4.1_e8.2_400bps_fast@v4.2.0 (5kHz)
* dna_r10.4.1_e8.2_400bps_hac@v4.2.0 (5kHz)
* dna_r10.4.1_e8.2_400bps_sup@v4.2.0 (5kHz)


**RNA models:**

* rna003_120bps_sup@v3

**Modified base models***

* dna_r9.4.1_e8_fast@v3.4_5mCG@v0
* dna_r9.4.1_e8_hac@v3.3_5mCG@v0
* dna_r9.4.1_e8_sup@v3.3_5mCG@v0
* dna_r10.4.1_e8.2_260bps_fast@v3.5.2_5mCG@v2
* dna_r10.4.1_e8.2_260bps_hac@v3.5.2_5mCG@v2
* dna_r10.4.1_e8.2_260bps_sup@v3.5.2_5mCG@v2
* dna_r10.4.1_e8.2_400bps_fast@v3.5.2_5mCG@v2
* dna_r10.4.1_e8.2_400bps_hac@v3.5.2_5mCG@v2
* dna_r10.4.1_e8.2_400bps_sup@v3.5.2_5mCG@v2
* dna_r10.4.1_e8.2_260bps_fast@v4.0.0_5mCG_5hmCG@v2
* dna_r10.4.1_e8.2_260bps_hac@v4.0.0_5mCG_5hmCG@v2
* dna_r10.4.1_e8.2_260bps_sup@v4.0.0_5mCG_5hmCG@v2
* dna_r10.4.1_e8.2_400bps_fast@v4.0.0_5mCG_5hmCG@v2
* dna_r10.4.1_e8.2_400bps_hac@v4.0.0_5mCG_5hmCG@v2
* dna_r10.4.1_e8.2_400bps_sup@v4.0.0_5mCG_5hmCG@v2
* dna_r10.4.1_e8.2_260bps_fast@v4.1.0_5mCG_5hmCG@v2
* dna_r10.4.1_e8.2_260bps_hac@v4.1.0_5mCG_5hmCG@v2
* dna_r10.4.1_e8.2_260bps_sup@v4.1.0_5mCG_5hmCG@v2
* dna_r10.4.1_e8.2_400bps_fast@v4.1.0_5mCG_5hmCG@v2
* dna_r10.4.1_e8.2_400bps_hac@v4.1.0_5mCG_5hmCG@v2
* dna_r10.4.1_e8.2_400bps_sup@v4.1.0_5mCG_5hmCG@v2
* dna_r10.4.1_e8.2_400bps_fast@v4.2.0_5mCG_5hmCG@v2 (5kHz)
* dna_r10.4.1_e8.2_400bps_hac@v4.2.0_5mCG_5hmCG@v2 (5kHz)
* dna_r10.4.1_e8.2_400bps_sup@v4.2.0_5mCG_5hmCG@v2 (5kHz)
* dna_r10.4.1_e8.2_400bps_sup@v4.2.0_5mC@v2 (5kHz)
* dna_r10.4.1_e8.2_400bps_sup@v4.2.0_6mA@v2 (5kHz)

## Developer quickstart

### Linux dependencies

The following packages are necessary to build dorado in a barebones environment (e.g. the official ubuntu:jammy docker image)

```
$ apt-get update && apt-get install -y --no-install-recommends \
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

```
$ git clone https://github.com/nanoporetech/dorado.git dorado
$ cd dorado
$ cmake -S . -B cmake-build
$ cmake --build cmake-build --config Release -j
$ ctest --test-dir cmake-build
```

The `-j` flag will use all available threads to build dorado and usage is around 1-2GB per thread. If you are constrained
by the amount of available memory on your system you can lower the number of threads i.e.` -j 4`.

After building you can run dorado from the build directory `./cmake-build/bin/dorado` or install it somewhere else on your
system i.e. `/opt` *(note: you will need the relevant permissions for the target installation directory)*.

```
$ cmake --install cmake-build --prefix /opt
```

### Pre commit

The project uses pre-commit to ensure code is consistently formatted, you can set this up using pip:

```bash
$ pip install pre-commit
$ pre-commit install
```

### Licence and Copyright
(c) 2022 Oxford Nanopore Technologies PLC.

Dorado is distributed under the terms of the Oxford Nanopore
Technologies PLC.  Public License, v. 1.0.  If a copy of the License
was not distributed with this file, You can obtain one at
http://nanoporetech.com
