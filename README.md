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

 - [dorado-0.3.0-linux-x64](https://cdn.oxfordnanoportal.com/software/analysis/dorado-0.3.0-linux-x64.tar.gz)
 - [dorado-0.3.0-linux-arm64](https://cdn.oxfordnanoportal.com/software/analysis/dorado-0.3.0-linux-arm64.tar.gz)
 - [dorado-0.3.0-osx-arm64](https://cdn.oxfordnanoportal.com/software/analysis/dorado-0.3.0-osx-arm64.tar.gz)
 - [dorado-0.3.0-win64](https://cdn.oxfordnanoportal.com/software/analysis/dorado-0.3.0-win64.zip)

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

Dorado is Oxford Nanopore's recommended basecaller for offline basecalling. We are working on a number of features which we expect to release soon:

1. DNA Barcode multiplexing
2. Adapter trimming
3. Python API
4. Statically linked binary

## Performance tips

1. For optimal performance Dorado requires POD5 file input. Please [convert your Fast5 files](https://github.com/nanoporetech/pod5-file-format) before basecalling.
2. Dorado will automatically detect your GPUs' free memory and select an appropriate batch size.
3. Dorado will automatically run in multi-GPU `cuda:all` mode. If you have a hetrogenous collection of GPUs select the faster GPUs using the `--device` flag (e.g `--device cuda:0,2`). Not doing this will have a detrimental impact on performance.

## Running

### Simplex basecalling

To run Dorado basecalling, download a model and point it to POD5 files _(Fast5 files are supported but will not be as performant)_.

```
$ dorado download --model dna_r10.4.1_e8.2_400bps_hac@v4.1.0
$ dorado basecaller dna_r10.4.1_e8.2_400bps_hac@v4.1.0 pod5s/ > calls.bam
```

### Modified basecalling

To call modifications simply add `--modified-bases` to the basecaller command

```
$ dorado basecaller dna_r10.4.1_e8.2_400bps_hac@v4.1.0 pod5s/ --modified-bases 5mCG_5hmCG > calls.bam
```

### Duplex
To run Duplex basecalling run the command:

```
$ dorado duplex dna_r10.4.1_e8.2_400bps_sup@v4.1.0 pod5s/ > duplex.bam
```

This command will output both simplex and duplex reads. Duplex reads will have the `dx` tag set to `1` in the output BAM, simplex reads will have the `dx` tag set to `0`.

Dorado duplex previously required a separate tool to perform duplex pair detection and read splitting, but this is now integrated into Dorado.

### Alignment

Dorado supports aligning existing basecalls or producing aligned output directly.

To align existing basecalls run:

```
$ dorado aligner <index> <reads> 
```
where `index` is a reference to align to in (fastq/fasta/mmi) format and `reads` is a file in any HTS format.

to basecall with alignment with duplex or simplex run with the `--reference` option:

```
$ dorado basecaller <model> <reads> --reference <index>
```

Alignment uses [minimap2](https://github.com/lh3/minimap2) and by default uses the `map-ont` preset. This can be overridden with the `-k` and `-w` options to set kmer and window size respectively.

## Available basecalling models

To download all available dorado models run:

```
$ dorado download --model all
```

**Simplex models:**

v4.1.0 models are recommended for our latest released condition (4kHz).

* dna_r10.4.1_e8.2_260bps_fast@v4.1.0
* dna_r10.4.1_e8.2_260bps_hac@v4.1.0
* dna_r10.4.1_e8.2_260bps_sup@v4.1.0
* dna_r10.4.1_e8.2_400bps_fast@v4.1.0
* dna_r10.4.1_e8.2_400bps_hac@v4.1.0
* dna_r10.4.1_e8.2_400bps_sup@v4.1.0

The following simplex models are also available:

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

**Modified base models**

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

## Decoding Dorado Model Names

The names of Dorado models are systematically structured, each segment corresponding to a different aspect of the model, which include both chemistry and run settings. To help understand this, let's dissect a sample model name:

`dna_r10.4.1_e8.2_400bps_fast@v4.2.0`

- **Analyte Type (`dna`/`rna`)**: This denotes the type of analyte being sequenced. For DNA sequencing, it's represented as `dna`. If you're using the direct RNA kit, this will be `rna`.

- **Pore Type (`r10.4.1`)**: This section corresponds to the type of flow cell used. For instance, FLO-MIN114/FLO-FLG114 is indicated by `r10.4.1`, while FLO-MIN106/FLO-FLG001 is signified by `r9.4.1`.

- **Chemistry Type (`e.8.2`)**: This represents the chemistry type, which corresponds to the kit used for sequencing. For example, kit 14 is denoted by `e.8.2`.

- **Translocation Speed (`400bps`)**: This parameter, selected at the run setup in MinKNOW, refers to the speed of translocation. Prior to starting your run, a prompt will ask if you prefer to run at 260bps or 400bps. The former yields more accurate results but provides less data. As of MinKNOW version 23.04, the 260bps option has been deprecated.

- **Model Type (`fast`)**: This represents the size of the model, where larger models yield more accurate basecalls but take more time. The three types of models are `fast`, `hac`, and `sup`. The `fast` model is the quickest, `sup` is the most accurate, and `hac` provides a balance between speed and accuracy. For most users, the `hac` model is recommended.

- **Model Version Number (`v4.2.0`)**: This denotes the version of the model. Model updates are regularly released, and higher version numbers typically signify greater accuracy.

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

## Licence and Copyright

(c) 2022 Oxford Nanopore Technologies PLC.

Dorado is distributed under the terms of the Oxford Nanopore
Technologies PLC.  Public License, v. 1.0.  If a copy of the License
was not distributed with this file, You can obtain one at
http://nanoporetech.com
