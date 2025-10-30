# Dorado

Dorado is a high-performance, easy-to-use, open source analysis engine for Oxford Nanopore reads.

Detailed information about Dorado and its features is available in the [Dorado Documentation](https://software-docs.nanoporetech.com/dorado/latest/).

## Features

* One executable with sensible defaults, automatic hardware detection and configuration.
* Runs on Apple silicon (M series) and Nvidia GPUs including multi-GPU with linear scaling (see [Platforms](#platforms)).
* [Modified basecalling](https://software-docs.nanoporetech.com/dorado/latest/basecaller/mods/).
* [Duplex basecalling](https://software-docs.nanoporetech.com/dorado/latest/basecaller/duplex/) (watch the following video for an introduction to [Duplex](https://youtu.be/8DVMG7FEBys)).
* Simplex [barcode classification](https://software-docs.nanoporetech.com/dorado/latest/barcoding/barcoding/).
* Support for aligned read output in SAM/BAM.
* Initial support for [poly(A) tail estimation](https://software-docs.nanoporetech.com/dorado/latest/basecaller/polya_estimation/).
* Support for [single-read error correction](https://software-docs.nanoporetech.com/dorado/latest/assembly/correct/).
* [POD5](https://github.com/nanoporetech/pod5-file-format) support for highest basecalling performance ([documentation](https://software-docs.nanoporetech.com/pod5/latest/)).
* Based on libtorch, the C++ API for pytorch.
* Multiple custom optimisations in CUDA and Metal for maximising inference performance.

If you encounter any problems building or running Dorado, please [report an issue](https://github.com/nanoporetech/dorado/issues).

## Installation

First, download the relevant installer for your platform:

* [dorado-1.2.0-linux-x64](https://cdn.oxfordnanoportal.com/software/analysis/dorado-1.2.0-linux-x64.tar.gz)
* [dorado-1.2.0-linux-arm64](https://cdn.oxfordnanoportal.com/software/analysis/dorado-1.2.0-linux-arm64.tar.gz)
* [dorado-1.2.0-osx-arm64](https://cdn.oxfordnanoportal.com/software/analysis/dorado-1.2.0-osx-arm64.zip)
* [dorado-1.2.0-win64](https://cdn.oxfordnanoportal.com/software/analysis/dorado-1.2.0-win64.zip)

Once the relevant `.tar.gz` or `.zip` archive is downloaded, extract the archive to your desired location.

You can then call Dorado using the full path, for example:

```bash
/path/to/dorado-x.y.z-linux-x64/bin/dorado basecaller hac pod5s/ > calls.bam
```

Or you can add the bin path to your `$PATH` environment variable, and run with the `dorado` command instead, for example:

```bash
dorado basecaller hac pod5s/ > calls.bam
```

Please visit the [dorado documentation](https://software-docs.nanoporetech.com/dorado/latest/) for more information on getting started.

See [DEV.md](DEV.md) for details about building Dorado for development.

## Platforms

Dorado is heavily-optimised for Nvidia A100 and H100 GPUs and will deliver maximal performance on systems with these GPUs.

Dorado has been tested extensively and supported on the following systems:

| Platform | GPU/CPU | Minimum Software Requirements |
| --- |---------|--------------|
| Linux x86_64  | (G)V100, A100, H100 | CUDA Driver ≥525.105 |
| Linux arm64 | Jetson Orin | Linux for Tegra ≥36.4.3 (JetPack ≥6.2) |
| Windows x86_64 | (G)V100, A100, H100 | CUDA Driver ≥529.19 |
| Apple | Apple Silicon (M series) | macOS ≥14 |

Linux x64 or Windows systems not listed above but which have Nvidia GPUs with ≥8 GB VRAM and architecture from Pascal onwards (except P100/GP100) have not been widely tested but are expected to work. When basecalling with Apple devices, we recommend systems with ≥16 GB of unified memory.

If you encounter problems with running on your system, please [report an issue](https://github.com/nanoporetech/dorado/issues).

AWS Benchmarks on Nvidia GPUs for Dorado 0.3.0 are available [here](https://aws.amazon.com/blogs/hpc/benchmarking-the-oxford-nanopore-technologies-basecallers-on-aws/). Please note: Dorado's basecalling speed is continuously improving, so these benchmarks may not reflect performance with the latest release.

## Performance tips

1. Dorado will automatically detect your GPU's free memory and select an appropriate batch size.
2. Dorado will automatically run in multi-GPU `cuda:all` mode. If you have a heterogeneous collection of GPUs, select the faster GPUs using the `--device` flag (e.g., `--device cuda:0,2`). Not doing this will have a detrimental impact on performance.
3. On Windows systems with Nvidia GPUs, open Nvidia Control Panel, navigate into “Manage 3D settings” and then set “CUDA - Sysmem Fallback Policy” to “Prefer No Sysmem Fallback”.  This will provide a significant performance improvement.

## Running

The following are helpful commands for getting started with Dorado.
To see all options and their defaults, run `dorado -h` and `dorado <subcommand> -h`.

### Simplex basecalling

To run Dorado basecalling, using the automatically downloaded `hac` model on a directory of POD5 files or a single POD5 file.

```bash
dorado basecaller hac pod5s/ > calls.bam
```

To basecall a single file, simply replace the directory `pod5s/` with a path to your data file.

Click here for more details on [simplex basecalling](https://software-docs.nanoporetech.com/dorado/latest/basecaller/simplex/) including how to use the
`--resume-from` feature.

### DNA adapter and primer trimming

Dorado can detect and remove any adapter and/or primer sequences from the beginning and end of DNA reads. Note that if you intend to demultiplex the reads at some later time, trimming primers will likely result in some portions of the flanking regions of the barcodes being removed, which could prevent demultiplexing from working properly. For details see the dorado documentation on [read trimming](https://software-docs.nanoporetech.com/dorado/latest/basecaller/read_trimming/).

### Modified basecalling

Beyond the traditional A, T, C, and G basecalling, Dorado can also detect modified bases such as 5-methylcytosine (5mC), 5-hydroxymethylcytosine (5hmC), and N<sup>6</sup>-methyladenosine (6mA). These modified bases play crucial roles in epigenetic regulation.

For full details please read the documentation on [modified basecalling](https://software-docs.nanoporetech.com/dorado/latest/basecaller/mods/#introduction).

To call modifications, extend the [models argument](https://software-docs.nanoporetech.com/dorado/latest/models/selection/) with a comma-separated list of modifications:

```
dorado basecaller hac,5mCG_5hmCG,6mA pod5s/ > calls.bam
```

In the example above, basecalling is performed with the detection of both 5mC/5hmC in CG contexts and 6mA in all contexts. See here for details on [modified basecalling context](https://software-docs.nanoporetech.com/dorado/latest/basecaller/mods/#modification-context).

Refer to the [models list](https://software-docs.nanoporetech.com/dorado/latest/models/list/) table's _Compatible Modifications_ column to see available modifications.

Modified basecalling is also supported with [Duplex basecalling](https://software-docs.nanoporetech.com/dorado/latest/basecaller/duplex/#hemi-methylation-duplex-basecalling), where it produces hemi-methylation calls.

### Duplex

To run Duplex basecalling, run the command:

```
dorado duplex sup pod5s/ > duplex.bam
```

For more details please head to the the [dorado duplex basecalling documentation](https://software-docs.nanoporetech.com/dorado/latest/basecaller/duplex/).

### Alignment

Dorado supports aligning existing basecalls or producing aligned output directly, internally using [minimap2](https://github.com/lh3/minimap2).

To align existing basecalls, run:

```bash
dorado aligner <index> <reads>  > aligned.bam
```

where `index` is a reference to align to in (FASTQ/FASTA/.mmi) format and `reads` is a folder or file in any HTS format.

To basecall with alignment with duplex or simplex, run with the `--reference` option:

```bash
dorado basecaller <model> <reads> --reference <index> > calls.bam
```

For more details please check out the [dorado aligner documentation](https://software-docs.nanoporetech.com/dorado/latest/basecaller/alignment/).

### Sequencing Summary

The `dorado summary` command outputs a tab-separated file with read level sequencing information from the BAM file generated during basecalling. To create a summary, run:

```bash
dorado summary <bam> > summary.tsv
```

### Barcode Classification

Dorado supports barcode classification for existing basecalls as well as producing classified basecalls directly. Further details can be found at the [dorado barcoding documentation](https://software-docs.nanoporetech.com/dorado/latest/barcoding/barcoding/).

### Poly(A) tail estimation

Dorado has initial support for estimating poly(A) tail lengths for cDNA (PCS and PCB kits) and RNA, and can be configured for use with custom primer sequences, interrupted tails, and plasmids. Note that Oxford Nanopore cDNA reads are sequenced in two different orientations and Dorado poly(A) tail length estimation handles both (A and T homopolymers). This feature can be enabled by passing `--estimate-poly-a` to the `basecaller` command. For more details check out the [dorado poly(A) estimation documentation](https://software-docs.nanoporetech.com/dorado/latest/basecaller/polya_estimation/).

### Read Error Correction

Dorado supports single-read error correction with the integration of the [HERRO](https://github.com/lbcb-sci/herro) algorithm. HERRO uses all-vs-all alignment followed by haplotype-aware correction using a deep learning model to achieve higher single-read accuracies. The corrected reads are primarily useful for generating _de novo_ assemblies of diploid organisms.

To correct reads, run:

```bash
dorado correct reads.fastq > corrected_reads.fasta
```

Checkout the [doroado correct documentation](https://software-docs.nanoporetech.com/dorado/latest/assembly/correct/) for all the details.

### Polishing

Dorado `polish` is a high accuracy assembly polishing tool which outperforms similar tools for most ONT-based assemblies.

It takes as input a draft assembly produced by a tool such as [Hifiasm](https://github.com/chhylp123/hifiasm) or [Flye](https://github.com/mikolmogorov/Flye) and aligned reads and outputs an updated version of the assembly.

Additionally, Dorado `polish` can output a VCF file containing records for all variants discovered during polishing, or a gVCF file containing records for all locations in the input draft sequences.

Note that Dorado `polish` is a **haploid** polishing tool and does _not_ implement any sort of phasing internally. It will take input alignment data _as is_ and run it through the polishing model to produce the consensus sequences. For more information, please take a look at [this section](https://software-docs.nanoporetech.com/dorado/latest/assembly/polish/#polishing-diploidpolyploid-assemblies) of Dorado Docs.

For more information on how to get started head to the [dorado polish documentation](https://software-docs.nanoporetech.com/dorado/latest/assembly/polish/).

### Variant Calling - Alpha preview release

Dorado `variant` is an early-stage diploid small variant caller, released for experimental use and evaluation purposes.
This version is intended for feedback and should not yet be considered production-ready. For more information check out the [dorado variant documentation](https://software-docs.nanoporetech.com/dorado/latest/assembly/variant/).

## Available basecalling models

Click here for a list of [all available Dorado models](https://software-docs.nanoporetech.com/dorado/latest/models/list/).

Dorado can often download [models automatically](https://software-docs.nanoporetech.com/dorado/latest/models/selection/#automatic-model-download) based on the [model argument](https://software-docs.nanoporetech.com/dorado/latest/models/selection/) used.

To download all models instead of using the automatic download, run:

```bash
dorado download --model all
```

Click here for more information on [dorado downloader](https://software-docs.nanoporetech.com/dorado/latest/models/downloader/) and [dorado model selection](https://software-docs.nanoporetech.com/dorado/latest/models/selection/).

## Troubleshooting Guide

Click here for the [dorado troubleshooting documentation](https://software-docs.nanoporetech.com/dorado/latest/troubleshooting/troubleshooting/).

## Licence and Copyright

(c) 2025 Oxford Nanopore Technologies PLC.

Dorado is distributed under the terms of the Oxford Nanopore Technologies PLC.  Public License, v. 1.0.  If a copy of the License was not distributed with this file, You can obtain one at https://nanoporetech.com
