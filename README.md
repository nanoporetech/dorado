# Dorado

Dorado is a high-performance, easy-to-use, open source basecaller for Oxford Nanopore reads.

## Features

* One executable with sensible defaults, automatic hardware detection and configuration.
* Runs on Apple silicon (M1/2 family) and Nvidia GPUs including multi-GPU with linear scaling (see [Platforms](#platforms)).
* [Modified basecalling](#modified-basecalling).
* [Duplex basecalling](#duplex) (watch the following video for an introduction to [Duplex](https://youtu.be/8DVMG7FEBys)).
* Simplex [barcode classification](#barcode-classification).
* Support for aligned read output in SAM/BAM.
* Initial support for [poly(A) tail estimation](#polya-tail-estimation).
* Support for [single-read error correction](#read-error-correction).
* [POD5](https://github.com/nanoporetech/pod5-file-format) support for highest basecalling performance.
* Based on libtorch, the C++ API for pytorch.
* Multiple custom optimisations in CUDA and Metal for maximising inference performance.

If you encounter any problems building or running Dorado, please [report an issue](https://github.com/nanoporetech/dorado/issues).

## Installation

First, download the relevant installer for your platform:

 - [dorado-0.7.1-linux-x64](https://cdn.oxfordnanoportal.com/software/analysis/dorado-0.7.1-linux-x64.tar.gz)
 - [dorado-0.7.1-linux-arm64](https://cdn.oxfordnanoportal.com/software/analysis/dorado-0.7.1-linux-arm64.tar.gz)
 - [dorado-0.7.1-osx-arm64](https://cdn.oxfordnanoportal.com/software/analysis/dorado-0.7.1-osx-arm64.zip)
 - [dorado-0.7.1-win64](https://cdn.oxfordnanoportal.com/software/analysis/dorado-0.7.1-win64.zip)

Once the relevant `.tar.gz` or `.zip` archive is downloaded, extract the archive to your desired location.

You can then call Dorado using the full path, for example:
```
$ /path/to/dorado-x.y.z-linux-x64/bin/dorado basecaller hac pod5s/ > calls.bam
```

Or you can add the bin path to your `$PATH` environment variable, and run with the `dorado` command instead, for example:
```
$ dorado basecaller hac pod5s/ > calls.bam
```

See [DEV.md](DEV.md) for details about building Dorado for development.

## Platforms

Dorado is heavily-optimised for Nvidia A100 and H100 GPUs and will deliver maximal performance on systems with these GPUs.

Dorado has been tested extensively and supported on the following systems:

| Platform | GPU/CPU | Minimum Software Requirements |
| --- |---------|--------------|
| Linux x86_64  | (G)V100, A100 | CUDA Driver ≥450.80.02 |
| | H100 | CUDA Driver ≥520 |
| Linux arm64 | Jetson Orin | Linux for Tegra ≥34.1.1 |
| Windows x86_64 | (G)V100, A100 | CUDA Driver ≥452.39 |
| | H100 | CUDA Driver ≥520 |
| Apple | Apple Silicon (M1/M2) | |

Linux or Windows systems not listed above but which have Nvidia GPUs with ≥8 GB VRAM and architecture from Pascal onwards (except P100/GP100) have not been widely tested but are expected to work. When basecalling with Apple devices, we recommend systems with ≥16 GB of unified memory.

If you encounter problems with running on your system, please [report an issue](https://github.com/nanoporetech/dorado/issues).

AWS Benchmarks on Nvidia GPUs for Dorado 0.3.0 are available [here](https://aws.amazon.com/blogs/hpc/benchmarking-the-oxford-nanopore-technologies-basecallers-on-aws/). Please note: Dorado's basecalling speed is continuously improving, so these benchmarks may not reflect performance with the latest release.

## Performance tips

1. For optimal performance, Dorado requires POD5 file input. Please [convert your .fast5 files](https://github.com/nanoporetech/pod5-file-format) before basecalling.
2. Dorado will automatically detect your GPU's free memory and select an appropriate batch size.
3. Dorado will automatically run in multi-GPU `cuda:all` mode. If you have a hetrogenous collection of GPUs, select the faster GPUs using the `--device` flag (e.g `--device cuda:0,2`). Not doing this will have a detrimental impact on performance.

## Running

The following are helpful commands for getting started with Dorado.
To see all options and their defaults, run `dorado -h` and `dorado <subcommand> -h`.

### Model selection foreword

Dorado can automatically select a basecalling model using a selection of model speed (`fast`, `hac`, `sup`) and the pod5 data. This feature is **not** supported for fast5 data. If the model does not exist locally, dorado will automatically downloaded the model and delete it when finished. To re-use downloaded models, manually download models using `dorado download`.

Dorado continues to support model paths.

For details read [Automatic model selection complex](#automatic-model-selection-complex).

### Simplex basecalling

To run Dorado basecalling, using the automatically downloaded `hac` model on a directory of POD5 files or a single POD5 file _(.fast5 files are supported, but will not be as performant)_.

```
$ dorado basecaller hac pod5s/ > calls.bam
```

To basecall a single file, simply replace the directory `pod5s/` with a path to your data file.

If basecalling is interrupted, it is possible to resume basecalling from a BAM file. To do so, use the `--resume-from` flag to specify the path to the incomplete BAM file. For example:

```
$ dorado basecaller hac pod5s/ --resume-from incomplete.bam > calls.bam
```

`calls.bam` will contain all of the reads from `incomplete.bam` plus the new basecalls *(`incomplete.bam` can be discarded after basecalling is complete)*.

**Note: it is important to choose a different filename for the BAM file you are writing to when using `--resume-from`**. If you use the same filename, the interrupted BAM file will lose the existing basecalls and basecalling will restart from the beginning.

### DNA adapter and primer trimming

Dorado can detect and remove any adapter and/or primer sequences from the beginning and end of DNA reads. Note that if you intend to demultiplex the reads at some later time, trimming adapters and primers may result in some portions of the flanking regions of the barcodes being removed, which could interfere with correct demultiplexing.

#### In-line with basecalling

By default, `dorado basecaller` will attempt to detect any adapter or primer sequences at the beginning and ending of reads, and remove them from the output sequence.

This functionality can be altered by using either the `--trim` or `--no-trim` options with `dorado basecaller`. The `--no-trim` option will prevent the trimming of detected barcode sequences as well as the detection and trimming of adapter and primer sequences.

The `--trim` option takes as its argument one of the following values:

* `all` This is the the same as the default behavior. Any detected adapters or primers will be trimmed, and if barcoding is enabled then any detected barcodes will be trimmed.
* `primers` This will result in any detected adapters or primers being trimmed, but if barcoding is enabled the barcode sequences will not be trimmed.
* `adapters` This will result in any detected adapters being trimmed, but primers will not be trimmed, and if barcoding is enabled then barcodes will not be trimmed either.
* `none` This is the same as using the --no-trim option. Nothing will be trimmed.

If adapter/primer trimming is done in-line with basecalling in combination with demultiplexing, then the software will automatically ensure that the trimming of adapters and primers does not interfere with the demultiplexing process. However, if you intend to do demultiplexing later as a separate step, then it is recommended that you disable adapter/primer trimming when basecalling with the `--no-trim` option, to ensure that any barcode sequences remain completely intact in the reads.

#### Trimming existing datasets

Existing basecalled datasets can be scanned for adapter and/or primer sequences at either end, and trim any such found sequences. To do this, run:

```
$ dorado trim <reads> > trimmed.bam
```

`<reads>` can either be an HTS format file (e.g. FASTQ, BAM, etc.) or a stream of an HTS format (e.g. the output of Dorado basecalling).

The `--no-trim-primers` option can be used to prevent the trimming of primer sequences. In this case only adapter sequences will be trimmed.

If it is also your intention to demultiplex the data, then it is recommended that you demultiplex before trimming any adapters and primers, as trimming adapters and primers first may interfere with correct barcode classification.

The output of `dorado trim` will always be unaligned records, regardless of whether the input is aligned/sorted or not.

#### Custom primer trimming

The software automatically searches for primer sequences used in Oxford Nanopore kits. However, you can specify an alternative set of primer sequences to search for when trimming either in-line with basecalling, or in combination with the `--trim` option. In both cases this is accomplished using the `--primer-sequences` command line option, followed by the full path and filename of a FASTA file containing the primer sequences you want to search for. The record names of the sequences do not matter. Note that if you use this option the normal primer sequences built-in to the dorado software will not be searched for.

### RNA adapter trimming

Adapters for RNA002 and RNA004 kits are automatically trimmed during basecalling. However, unlike in DNA, the RNA adapter cannot be trimmed post-basecalling.

### Modified basecalling

Beyond the traditional A, T, C, and G basecalling, Dorado can also detect modified bases such as 5-methylcytosine (5mC), 5-hydroxymethylcytosine (5hmC), and N<sup>6</sup>-methyladenosine (6mA). These modified bases play crucial roles in epigenetic regulation.

To call modifications, extend the models argument with a comma-separated list of modifications:

```
$ dorado basecaller hac,5mCG_5hmCG pod5s/ > calls.bam
```

Refer to the [DNA models](#dna-models) table's _Compatible Modifications_ column to see available modifications that can be called with the `--modified-bases` option.

Modified basecalling is also supported with [Duplex basecalling](#duplex), where it produces hemi-methylation calls.

### Duplex

To run Duplex basecalling, run the command:

```
$ dorado duplex sup pod5s/ > duplex.bam
```

When using the `duplex` command, two types of DNA sequence results will be produced: 'simplex' and 'duplex'. Any specific position in the DNA which is in a duplex read is also seen in two simplex strands (the template and complement).  So, each DNA position which is duplex sequenced will be covered by a minimum of three separate readings in the output.

The `dx` tag in the BAM record for each read can be used to distinguish between simplex and duplex reads:
* `dx:i:1` for duplex reads.
* `dx:i:0` for simplex reads which don't have duplex offsprings.
* `dx:i:-1` for simplex reads which have duplex offsprings.

Dorado will report the duplex rate as the number of nucleotides in the duplex basecalls multiplied by two and divided by the total number of nucleotides in the simplex basecalls. This value is a close approximation for the proportion of nucleotides which participated in a duplex basecall.

Duplex basecalling can be performed with modified base detection, producing hemi-methylation calls for duplex reads:

```
$ dorado duplex hac,5mCG_5hmCG pod5s/ > duplex.bam
```
More information on how hemi-methylation calls are represented can be found in [page 7 of the SAM specification document (version aa7440d)](https://samtools.github.io/hts-specs/SAMtags.pdf) and [Modkit documentation](https://nanoporetech.github.io/modkit/intro_pileup_hemi.html).


### Alignment

Dorado supports aligning existing basecalls or producing aligned output directly.

To align existing basecalls, run:

```
$ dorado aligner <index> <reads>  > aligned.bam
```
where `index` is a reference to align to in (FASTQ/FASTA/.mmi) format and `reads` is a folder or file in any HTS format.

When reading from an input folder, `dorado aligner` also supports emitting aligned files to an output folder, which will preserve the file structure of the inputs:

```
$ dorado aligner <index> <input_read_folder> --output-dir <output_read_folder>
```

An alignment summary containing alignment statistics for each read can be generated with the `--emit-summary` option. The file will be saved in the `--output-dir` folder.

To basecall with alignment with duplex or simplex, run with the `--reference` option:

```
$ dorado basecaller <model> <reads> --reference <index> > calls.bam
```

Alignment uses [minimap2](https://github.com/lh3/minimap2) and by default uses the `lr:hq` preset. This can be overridden by passing a minimap option string, `--mm2-opts`, using the '-x <preset>' option and/or individual options such as `-k` and `-w` to set kmer and window size respectively. For a complete list of supported minimap2 options use '--mm2-opts --help'. For example:
```
$ dorado aligner <index> <input_read_folder> --output-dir <output_read_folder> --mm2-opt "-x splice --junc-bed <annotations_file>"
$ dorado aligner <index> <input_read_folder> --output-dir <output_read_folder> --mm2-opt --help
$ dorado basecaller <model> <reads> --reference <index> --mm2-opt "-k 15 -w 10" > calls.bam
```


### Sequencing Summary

The `dorado summary` command outputs a tab-separated file with read level sequencing information from the BAM file generated during basecalling. To create a summary, run:

```
$ dorado summary <bam> > summary.tsv
```

Note that summary generation is only available for reads basecalled from POD5 files. Reads basecalled from .fast5 files are not compatible with the summary command.

### Barcode Classification

Dorado supports barcode classification for existing basecalls as well as producing classified basecalls directly.

#### In-line with basecalling

In this mode, reads are classified into their barcode groups during basecalling as part of the same command. To enable this, run:
```
$ dorado basecaller <model> <reads> --kit-name <barcode-kit-name> > calls.bam
```

This will result in a single output stream with classified reads. The classification will be reflected in the read group name as well as in the `BC` tag of the output record.

By default, Dorado is set up to trim the barcode from the reads. To disable trimming, add `--no-trim` to the cmdline.

The default heuristic for double-ended barcodes is to look for them on either end of the read. This results in a higher classification rate but can also result in a higher false positive count. To address this, `dorado basecaller` also provides a `--barcode-both-ends` option to force double-ended barcodes to be detected on both ends before classification. This will reduce false positives dramatically, but also lower overall classification rates.

The output from `dorado basecaller` can be demultiplexed into per-barcode BAMs using `dorado demux`. e.g.

```
$ dorado demux --output-dir <output-dir> --no-classify <input-bam>
```
This will output a BAM file per barcode in the `output-dir`.

The barcode information is reflected in the BAM `RG` header too. Therefore demultiplexing is also possible through `samtools split`. e.g.
```
$ samtools split -u <output-dir>/unclassified.bam -f "<output-dir>/<prefix>_%!.bam" <input-bam>
```
However, `samtools split` uses the full `RG` string as the filename suffix, which can result in very long file names. We recommend using `dorado demux` to split barcoded BAMs.

#### Classifying existing datasets

Existing basecalled datasets can be classified as well as demultiplexed into per-barcode BAMs using the standalone `demux` command in `dorado`. To use this, run

```
$ dorado demux --kit-name <kit-name> --output-dir <output-folder-for-demuxed-bams> <reads>
```

`<reads>` can either be a folder or a single file in an HTS format file (e.g. FASTQ, BAM, etc.) or a stream of an HTS format (e.g. the output of dorado basecalling).

This results in multiple BAM files being generated in the output folder, one per barcode (formatted as `KITNAME_BARCODEXX.bam`) and one for all unclassified reads. As with the in-line mode, `--no-trim` and `--barcode-both-ends` are also available as additional options.

If the input file is aligned/sorted and `--no-trim` is chosen, each of the output barcode-specific BAM files will also be sorted and indexed. However, if trimming is enabled (which is the default), the alignment information is removed and the output BAMs are unaligned. This is done because the alignment tags and positions are invalidated once a sequence is altered.

Here is an example output folder

```
$ dorado demux --kit-name SQK-RPB004 --output-dir /tmp/demux reads.fastq

$ ls -1 /tmp/demux
SQK-RPB004_barcode01.bam
SQK-RPB004_barcode02.bam
SQK-RPB004_barcode03.bam
...
unclassified.bam
```

A summary file listing each read and its classified barcode can be generated with the `--emit-summary` option in `dorado demux`. The file will be saved in the `--output-dir` folder.

#### Demultiplexing mapped reads

If the input data files contain mapping data, this information can be preserved in the output files. To do this, you must use the `--no-trim` option. Trimming the barcodes will invalidate any mapping information that may be contained in the input files, and therefore the application will exclude any mapping information if `--no-trim` is not specified.

It is also possible to get `dorado demux` to sort and index any output bam files that contain mapped reads. To enable this, use the `--sort-bam` option. If you use this option then you must also use the `--no-trim` option, as trimming will prevent any mapping information from being included in the output files. Index files (.bai extension) will only be created for BAM files that contain mapped reads and were sorted. Note that for large datasets sorting the output files may take a few minutes.

#### Using a sample sheet

Dorado is able to use a sample sheet to restrict the barcode classifications to only those present, and to apply aliases to the detected classifications. This is enabled by passing the path to a sample sheet to the `--sample-sheet` argument when using the `basecaller` or `demux` commands. See [here](documentation/SampleSheets.md) for more information.

#### Custom barcodes

In addition to supporting the standard barcode kits from Oxford Nanopore, Dorado also supports specifying custom barcode kit arrangements and sequences. This is done by passing a barcode arrangement file via the `--barcode-arrangement` argument (either to `dorado demux` or `dorado basecaller`). Custom barcode sequences can optionally be specified via the `--barcode-sequences` option. See [here](documentation/CustomBarcodes.md) for more details.

### Poly(A) tail estimation

Dorado has initial support for estimating poly(A) tail lengths for cDNA (PCS and PCB kits) and RNA. Note that Oxford Nanopore cDNA reads are sequenced in two different orientations and Dorado poly(A) tail length estimation handles both (A and T homopolymers). This feature can be enabled by passing `--estimate-poly-a` to the `basecaller` command. It is disabled by default. The estimated tail length is stored in the `pt:i` tag of the output record. Reads for which the tail length could not be estimated will not have the `pt:i` tag. Custom primer sequences and estimation of interrupted tails can be configured through the `--poly-a-config` option. See [here](documentation/PolyTailConfig.md) for more details.

### Read Error Correction

Dorado supports single-read error correction with the integration of the [HERRO](https://github.com/lbcb-sci/herro) algorithm. HERRO uses all-vs-all alignment followed by haplotype-aware correction using a deep learning model to achieve higher single-read accuracies. The corrected reads are primarily useful for generating *de novo* assemblies of diploid organisms.

To correct reads, run:
```
$ dorado correct reads.fastq(.gz) > corrected_reads.fasta
```

Dorado correct only supports FASTX(.gz) as the input and generates a FASTA file as output. The input can be uncompressed or compressed with `bgz`. An index file is generated for the input FASTX file in the same folder unless one is already present. Please ensure that the folder with the input file is writeable by the `dorado` process and has sufficient disk space (no more than 10GB should be necessary for a whole genome dataset).

The error correction tool is both compute and memory intensive. As a result, it is best run on a system with multiple high performance CPU cores ( > 64 cores), large system memory ( > 256GB) and a modern GPU with a large VRAM ( > 32GB).

All required model weights are downloaded automatically by Dorado. However, the weights can also be pre-downloaded and passed via command line in case of offline execution. To do so, run:
```
$ dorado download --model herro-v1
$ dorado correct -m herro-v1 reads.fastq(.gz) > corrected_reads.fasta
```

## Available basecalling models

To download all available Dorado models, run:

```
$ dorado download --model all
```

### Decoding Dorado model names

The names of Dorado models are systematically structured, each segment corresponding to a different aspect of the model, which include both chemistry and run settings. Below is a sample model name explained:

`dna_r10.4.1_e8.2_400bps_hac@v4.3.0`

- **Analyte Type (`dna`)**: This denotes the type of analyte being sequenced. For DNA sequencing, it is represented as `dna`. If you are using a Direct RNA Sequencing Kit, this will be `rna002` or `rna004`, depending on the kit.

- **Pore Type (`r10.4.1`)**: This section corresponds to the type of flow cell used. For instance, FLO-MIN114/FLO-FLG114 is indicated by `r10.4.1`, while FLO-MIN106D/FLO-FLG001 is signified by `r9.4.1`.

- **Chemistry Type (`e8.2`)**: This represents the chemistry type, which corresponds to the kit used for sequencing. For example, Kit 14 chemistry is denoted by `e8.2` and Kit 10 or Kit 9 are denoted by `e8`.

- **Translocation Speed (`400bps`)**: This parameter, selected at the run setup in MinKNOW, refers to the speed of translocation. Prior to starting your run, a prompt will ask if you prefer to run at 260 bps or 400 bps. The former yields more accurate results but provides less data. As of MinKNOW version 23.04, the 260 bps option has been deprecated.

- **Model Type (`hac`)**: This represents the size of the model, where larger models yield more accurate basecalls but take more time. The three types of models are `fast`, `hac`, and `sup`. The `fast` model is the quickest, `sup` is the most accurate, and `hac` provides a balance between speed and accuracy. For most users, the `hac` model is recommended.

- **Model Version Number (`v4.3.0`)**: This denotes the version of the model. Model updates are regularly released, and higher version numbers typically signify greater accuracy.


### **DNA models:**

Below is a table of the available basecalling models and the modified basecalling models that can be used with them. The bolded models are for the latest released condition with 5 kHz data.

The versioning of modification models is bound to the basecalling model. This means that the modification model version is reset for each new simplex model release. For example, `6mA@v1` compatible with `v4.3.0` basecalling models is more recent than `6mA@v2` compatible with `v4.2.0` basecalling models.

| Basecalling Models | Compatible<br />Modifications | Modifications<br />Model<br />Version | Data<br />Sampling<br />Frequency |
| :-------- | :------- | :--- | :--- |
| **dna_r10.4.1_e8.2_400bps_fast@v5.0.0** | | | 5 kHz |
| **dna_r10.4.1_e8.2_400bps_hac@v5.0.0** | 4mC_5mC<br />5mCG_5hmCG<br />5mC_5hmC<br />6mA<br /> | v1<br />v1<br />v1<br />v1 | 5 kHz |
| **dna_r10.4.1_e8.2_400bps_sup@v5.0.0** | 4mC_5mC<br />5mCG_5hmCG<br />5mC_5hmC<br />6mA<br /> | v1<br />v1<br />v1<br />v1 | 5 kHz |
| dna_r10.4.1_e8.2_400bps_fast@v4.3.0 | | | 5 kHz |
| dna_r10.4.1_e8.2_400bps_hac@v4.3.0 | 5mCG_5hmCG<br />5mC_5hmC<br />6mA<br /> | v1<br />v1<br />v2 | 5 kHz |
| dna_r10.4.1_e8.2_400bps_sup@v4.3.0 | 5mCG_5hmCG<br />5mC_5hmC<br />6mA<br /> | v1<br />v1<br />v2 | 5 kHz |
| dna_r10.4.1_e8.2_400bps_fast@v4.2.0 | 5mCG_5hmCG | v2 | 5 kHz |
| dna_r10.4.1_e8.2_400bps_hac@v4.2.0 | 5mCG_5hmCG | v2 | 5 kHz |
| dna_r10.4.1_e8.2_400bps_sup@v4.2.0 | 5mCG_5hmCG<br />5mC_5hmC<br />5mC<br />6mA<br />| v3.1<br />v1<br />v2<br />v3| 5 kHz |
| dna_r10.4.1_e8.2_400bps_fast@v4.1.0 | 5mCG_5hmCG | v2 | 4 kHz |
| dna_r10.4.1_e8.2_400bps_hac@v4.1.0 | 5mCG_5hmCG | v2 | 4 kHz |
| dna_r10.4.1_e8.2_400bps_sup@v4.1.0 | 5mCG_5hmCG | v2 | 4 kHz |
| dna_r10.4.1_e8.2_260bps_fast@v4.1.0 | 5mCG_5hmCG | v2 | 4 kHz |
| dna_r10.4.1_e8.2_260bps_hac@v4.1.0 | 5mCG_5hmCG | v2 | 4 kHz |
| dna_r10.4.1_e8.2_260bps_sup@v4.1.0 | 5mCG_5hmCG | v2 | 4 kHz |
| dna_r10.4.1_e8.2_400bps_fast@v4.0.0 | 5mCG_5hmCG | v2 | 4 kHz |
| dna_r10.4.1_e8.2_400bps_hac@v4.0.0 | 5mCG_5hmCG | v2 | 4 kHz |
| dna_r10.4.1_e8.2_400bps_sup@v4.0.0 | 5mCG_5hmCG | v2 | 4 kHz |
| dna_r10.4.1_e8.2_260bps_fast@v4.0.0 | 5mCG_5hmCG | v2 | 4 kHz |
| dna_r10.4.1_e8.2_260bps_hac@v4.0.0 | 5mCG_5hmCG | v2 | 4 kHz |
| dna_r10.4.1_e8.2_260bps_sup@v4.0.0 | 5mCG_5hmCG | v2 | 4 kHz |
| dna_r10.4.1_e8.2_260bps_fast@v3.5.2 | 5mCG | v2 | 4 kHz |
| dna_r10.4.1_e8.2_260bps_hac@v3.5.2 | 5mCG | v2 | 4 kHz |
| dna_r10.4.1_e8.2_260bps_sup@v3.5.2 | 5mCG | v2 | 4 kHz |
| dna_r10.4.1_e8.2_400bps_fast@v3.5.2 | 5mCG | v2 | 4 kHz |
| dna_r10.4.1_e8.2_400bps_hac@v3.5.2 | 5mCG | v2 | 4 kHz |
| dna_r10.4.1_e8.2_400bps_sup@v3.5.2 | 5mCG | v2 | 4 kHz |
| dna_r9.4.1_e8_sup@v3.6 |  |  | 4 kHz |
| dna_r9.4.1_e8_fast@v3.4 | 5mCG_5hmCG<br />5mCG | v0<br />v0.1 | 4 kHz |
| dna_r9.4.1_e8_hac@v3.3 | 5mCG_5hmCG<br />5mCG | v0<br />v0.1 |4 kHz |
| dna_r9.4.1_e8_sup@v3.3 | 5mCG_5hmCG<br />5mCG | v0<br />v0.1 |4 kHz |

### **RNA models:**

**Note:** The BAM format does not support `U` bases. Therefore, when Dorado is performing RNA basecalling, the resulting output files will include `T` instead of `U`. This is consistent across output file types. The same applies to parsing inputs. Any input HTS file (e.g. FASTQ generated by `guppy`/`basecall_server`) with `U` bases is not handled by `dorado`.

| Basecalling Models | Compatible<br />Modifications | Modifications<br />Model<br />Version | Data<br />Sampling<br />Frequency |
| :-------- | :------- | :--- | :--- |
| **rna004_130bps_fast@v5.0.0** | N/A | N/A | 4 kHz |
| **rna004_130bps_hac@v5.0.0** | m6A<br />pseU | v1<br />v1<br />v1 | 4 kHz |
| **rna004_130bps_sup@v5.0.0** | m6A<br />pseU | v1<br />v1<br />v1 | 4 kHz |
| rna004_130bps_fast@v3.0.1 | N/A | N/A | 4 kHz |
| rna004_130bps_hac@v3.0.1 | N/A | N/A | 4 kHz |
| rna004_130bps_sup@v3.0.1 | m6A_DRACH | v1 | 4 kHz |
| rna002_70bps_fast@v3 | N/A | N/A | 3 kHz |
| rna002_70bps_hac@v3 | N/A | N/A | 3 kHz |


## Automatic model selection complex

The `model` argument in dorado can specify either a model path or a model **_complex_**. A model complex must start with the **simplex model speed**, and follows this syntax:

```
(fast|hac|sup)[@(version|latest)][,modification[@(version|latest)]][,...]
```

Automatically selected modification models will always match the base simplex model version and will be the latest compatible version unless a specific version is set by the user. Automatic modification model selection will not allow the mixing of modification models which are bound to different simplex model versions.  

Here are a few examples of model complexes:

| Model Complex | Description |
| :------------ | :---------- |
| fast  | Latest compatible **fast** model |
| hac  | Latest compatible **hac** model |
| sup  | Latest compatible **sup** model |
| hac@latest | Latest compatible **hac** simplex basecalling model |
| hac@v4.2.0  | Simplex basecalling **hac** model with version `v4.2.0` |
| hac@v3.5 | Simplex basecalling **hac** model with version `v3.5.0` |
| hac,5mCG_5hmCG  | Latest compatible **hac** simplex model and latest **5mCG_5hmCG** modifications model for the chosen basecall model |
| hac,5mCG_5hmCG@v2  | Latest compatible **hac** simplex model and **5mCG_5hmCG** modifications model with version `v2.0.0` |
| sup,5mCG_5hmCG,6mA  | Latest compatible **sup** model and latest compatible **5mCG_5hmCG** and **6mA** modifications models |


## Troubleshooting Guide

### Library Path Errors

Dorado comes equipped with the necessary libraries (such as CUDA) for its execution. However, on some operating systems, the system libraries might be chosen over Dorado's. This discrepancy can result in various errors, for instance,  `CuBLAS error 8`.

To resolve this issue, you need to set the `LD_LIBRARY_PATH` to point to Dorado's libraries. Use a command like the following on Linux (change path as appropriate):

```
$ export LD_LIBRARY_PATH=<PATH_TO_DORADO>/dorado-x.y.z-linux-x64/lib:$LD_LIBRARY_PATH
```

On macOS, the equivalent export would be (change path as appropriate):

```
$ export DYLD_LIBRARY_PATH=<PATH_TO_DORADO>/dorado-x.y.z-osx-arm64/lib:$DYLD_LIBRARY_PATH
```

### Improving the Speed of Duplex Basecalling

Duplex basecalling is an IO-intensive process and can perform poorly if using networked storage or HDD. This can generally be improved by splitting up POD5 files appropriately.

Firstly install the POD5 python tools:

The POD5 documentation can be found [here](https://pod5-file-format.readthedocs.io/en/latest/docs/tools.html).


```
$ pip install pod5
```

Then run `pod5 view` to generate a table containing information to split on specifically, the "channel" information.

```
$ pod5 view /path/to/your/dataset/ --include "read_id, channel" --output summary.tsv
```

This will create "summary.tsv" file which should look like:

```
read_id channel
0000173c-bf67-44e7-9a9c-1ad0bc728e74    109
002fde30-9e23-4125-9eae-d112c18a81a7    463
...
```

Now run `pod5 subset` to copy records from your source data into outputs per-channel. This might take some time depending on the size of your dataset
```
$ pod5 subset /path/to/your/dataset/ --summary summary.tsv --columns channel --output split_by_channel
```

The command above will create the output directory `split_by_channel` and write into it one pod5 file per unique channel.  Duplex basecalling these split reads will now be much faster.

### Running Duplex Basecalling in a Distributed Fashion

If running duplex basecalling in a distributed fashion (e.g. on a SLURM or Kubernetes cluster) it is important to split POD5 files as described above. The reason is that duplex basecalling requires aggregation of reads from across a whole sequencing run, which will be distributed over multiple POD5 files.
The splitting strategy described above ensures that all reads which need to be aggregated are in the same POD5 file. Once the split is performed one can execute multiple jobs against smaller subsets of POD5 (e.g one job per 100 channels). This will allow basecalling to be distributed across nodes on a cluster. 
This will generate multiple BAMs which can be merged. This apporach also offers some resilience as if any job fails it can be restarted without having to re-run basecalling against the entire dataset.

### GPU Out of Memory Errors

Dorado operates on a broad range of GPUs but it is primarily developed for Nvidia A100/H100 and Apple Silicon. Dorado attempts to find the optimal batch size for basecalling. Nevertheless, on some low-RAM GPUs, users may face out of memory crashes.

A potential solution to this issue could be setting a manual batch size using the following command:

`dorado basecaller --batchsize 64 ...`

**Note:** Reducing memory consumption by modifying the `chunksize` parameter is not recommended as it influences the basecalling results.

### Low GPU Utilization

Low GPU utilization can lead to reduced basecalling speed. This problem can be identified using tools such as `nvidia-smi` and `nvtop`. Low GPU utilization often stems from I/O bottlenecks in basecalling. Here are a few steps you can take to improve the situation:

1. Opt for POD5 instead of .fast5: POD5 has superior I/O performance and will enhance the basecall speed in I/O constrained environments.
2. Transfer data to the local disk before basecalling: Slow basecalling often occurs because network disks cannot supply Dorado with adequate speed. To mitigate this, make sure your data is as close to your host machine as possible.
3. Choose SSD over HDD: Particularly for duplex basecalling, using a local SSD can offer significant speed advantages. This is due to the duplex basecalling algorithm's reliance on heavy random access of data.


## Licence and Copyright

(c) 2024 Oxford Nanopore Technologies PLC.

Dorado is distributed under the terms of the Oxford Nanopore
Technologies PLC.  Public License, v. 1.0.  If a copy of the License
was not distributed with this file, You can obtain one at
http://nanoporetech.com
