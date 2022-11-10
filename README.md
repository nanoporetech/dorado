# Dorado

Dorado is a high-performance, easy-to-use, open source basecaller for Oxford Nanopore reads.

## Features

* One executable with sensible defaults, automatic hardware detection and configuration.
* Runs on Apple silicon (M1-family) and Nvidia GPUs including multi-GPU with linear scaling.
* Modified basecalling (Remora models).
* [POD5](https://github.com/nanoporetech/pod5-file-format) support for highest basecalling performance.
* Based on libtorch, the C++ API for pytorch.
* Multiple custom optimisations in CUDA and Metal for maximising inference performance.

This is an alpha of Dorado . This software is being released for evaluation. If you encounter any problems building or running Dorado please [report an issue](https://github.com/nanoporetech/dorado/issues).

## Installation

 - [dorado-0.0.2-linux-x64](https://cdn.oxfordnanoportal.com/software/analysis/dorado-0.0.2-linux-x64.tar.gz)
 - [dorado-0.0.2-osx-arm64](https://cdn.oxfordnanoportal.com/software/analysis/dorado-0.0.2-osx-arm64.tar.gz)
 - [dorado-0.0.2-win64](https://cdn.oxfordnanoportal.com/software/analysis/dorado-0.0.2-win64.zip)

## Running

To run Dorado, download a model and point it to POD5 files. Fast5 files are supported but will not be as performant.

```
$ dorado download --model dna_r10.4.1_e8.2_260bps_hac@v3.5.2
$ dorado basecaller dna_r10.4.1_e8.2_260bps_hac@v3.5.2 pod5s/ > calls.sam
```

For unaligned BAM output, dorado output can be piped to BAM using samtoools:

```
$ dorado basecaller dna_r10.4.1_e8.2_260bps_hac@v3.5.2 pod5s/ | samtools view -Sh > calls.bam
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
2. Duplex basecalling
3. Alignmnet (output aligned BAMs)
4. Python API

## Performance tips

1. For optimal performance Dorado requires POD5 file input. Please [convert your Fast5 files](https://github.com/nanoporetech/pod5-file-format) before basecalling.
1. Dorado will automatically detect your GPUs' free memory and select an appropriate batch size. If you know what you're doing, you can use the     `--batch` parameter to tune batch size.
2. Dorado will automatically run in multi-GPU (`'cuda:all'`) mode. If you have a hetrogenous collection of GPUs select the faster GPUs using the `--device` flag (e.g `--device "cuda:0,2`). Not doing this will have a detrimental impact on performance.

## Available basecalling models

To download all available dorado models run:

```
$ dorado download --model all
```

The following models are currently available:

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

### Get Linux dependencies

```
apt-get update && apt-get install -y --no-install-recommends libhdf5-dev libssl-dev libzstd-dev
```

### Clone and build

```
$ git clone git@github.com:nanoporetech/dorado.git
$ cd dorado
$ cmake -S . -B cmake-build -DCMAKE_CUDA_COMPILER=<NVCC_DIR>/nvcc
$ cmake --build cmake-build --config Release -j
$ ctest --test-dir cmake-build
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
