# Dorado

Dorado is a high-performance, easy-to-use, open source analysis engine for Oxford Nanopore reads.

Detailed information about Dorado and its features is available in the [Dorado Documentation](https://dorado-docs.readthedocs.io/en/latest/).

## Features

* One executable with sensible defaults, automatic hardware detection and configuration.
* Runs on Apple silicon (M series) and Nvidia GPUs including multi-GPU with linear scaling (see [Platforms](#platforms)).
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

 - [dorado-1.1.1-linux-x64](https://cdn.oxfordnanoportal.com/software/analysis/dorado-1.1.1-linux-x64.tar.gz)
 - [dorado-1.1.1-linux-arm64](https://cdn.oxfordnanoportal.com/software/analysis/dorado-1.1.1-linux-arm64.tar.gz)
 - [dorado-1.1.1-osx-arm64](https://cdn.oxfordnanoportal.com/software/analysis/dorado-1.1.1-osx-arm64.zip)
 - [dorado-1.1.1-win64](https://cdn.oxfordnanoportal.com/software/analysis/dorado-1.1.1-win64.zip)

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
| Linux x86_64  | (G)V100, A100, H100 | CUDA Driver ≥525.105 |
| Linux arm64 | Jetson Orin | Linux for Tegra ≥36.4.3 (JetPack ≥6.2) |
| Windows x86_64 | (G)V100, A100, H100 | CUDA Driver ≥529.19 |
| Apple | Apple Silicon (M series) | macOS ≥13 |

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

### Model selection foreword

Dorado can automatically select a basecalling model using a selection of model speed (`fast`, `hac`, `sup`) and the pod5 data. If the model does not exist locally, dorado will automatically download the model and use it.

Dorado continues to support model paths.

For details read [Automatic model selection complex](#automatic-model-selection-complex).

### Simplex basecalling

To run Dorado basecalling, using the automatically downloaded `hac` model on a directory of POD5 files or a single POD5 file.

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

Dorado can detect and remove any adapter and/or primer sequences from the beginning and end of DNA reads. Note that if you intend to demultiplex the reads at some later time, trimming primers will likely result in some portions of the flanking regions of the barcodes being removed, which could prevent demultiplexing from working properly.

#### In-line with basecalling

By default, `dorado basecaller` will attempt to detect any adapter or primer sequences at the beginning and ending of reads, and remove them from the output sequence. Additionally, dorado will attempt to use any detected primers to determine whether the DNA went through the pore in the 5'-to-3' direction, or the 3'-to-5' direction. If this can be inferred, then the `TS:A` tag will be included in the BAM output for the read, with a value of `+` or `-` respectively. If it cannot be inferred, then this tag will not be included in the output.

In the specific cases of the SQK-PCS114 and SQK-PCB114 sequencing kits, if a UMI tag is present, it will also be detected and trimmed. Additionally, the UMI tag, if found, will be included in the BAM output for the read using the `RX:Z` tag.

This functionality can be altered by using either the `--trim` or `--no-trim` options with `dorado basecaller`. The `--no-trim` option will prevent the trimming of detected barcode sequences as well as the detection and trimming of adapter and primer sequences. Note that if primer trimming is not enabled, then no attempt will be made to detect primers, or to classify the orientation of the strand based on them, or to detect UMI tags.

The `--trim` option takes as its argument one of the following values:

* `all` This is the same as the default behaviour. Any detected adapters or primers will be trimmed, and if barcoding is enabled then any detected barcodes will be trimmed.
* `adapters` This will result in any detected adapters being trimmed, but primers will not be trimmed, and if barcoding is enabled then barcodes will not be trimmed either.
* `none` This is the same as using the --no-trim option. Nothing will be trimmed.

Dorado determines which adapter and primer sequences to search for and trim based on the sequencing-kit specified in the input file. If the sequencing-kit is not specified in the file, or is not a recognized and supported kit, then no adapter or primer trimming will be done. Note that by default the dorado software only supports adapter and primer trimming for kit14 sequencing kits.

If adapter/primer trimming is done in-line with basecalling in combination with demultiplexing, then the software will automatically ensure that the trimming of adapters and primers does not interfere with the demultiplexing process. However, if you intend to do demultiplexing later as a separate step, then it is recommended that you disable adapter/primer trimming when basecalling with the `--no-trim` option, to ensure that any barcode sequences remain completely intact in the reads.

#### Trimming existing datasets

Existing basecalled datasets can be scanned for adapter and/or primer sequences at either end, and trim any such found sequences. To do this, run:

```
$ dorado trim <reads> --sequencing-kit <kit_name> > trimmed.bam
```

`<reads>` can either be an HTS format file (e.g. FASTQ, BAM, etc.) or a stream of an HTS format (e.g. the output of Dorado basecalling).

`<kit_name>` is required to tell dorado what sequencing kit was used for the experiment, since this information is not encoded in the input files.

The `--no-trim-primers` option can be used to prevent the trimming of primer sequences. In this case only adapter sequences will be trimmed.

If it is also your intention to demultiplex the data, then it is recommended that you demultiplex before trimming any adapters and primers, as trimming adapters and primers first may interfere with correct barcode classification. It is also recommended in this case that you turn off trimming when you demultiplex, otherwise the trimming of the barcodes may prevent the proper detection of primers, resulting in partial primers remaining after trimming.

The output of `dorado trim` will always be unaligned records, regardless of whether the input is aligned/sorted or not.

#### Custom primer trimming

The software automatically searches for primer sequences used in Oxford Nanopore kits. However, you can specify an alternative set of primer sequences to search for when trimming either in-line with basecalling, or in combination with the `--trim` option. In both cases this is accomplished using the `--primer-sequences` command line option. The argument can be either the full path and filename of a FASTA file containing the primer sequences you want to search for, or a string code specifying a supported 3rd-party primer set.

If a FASTA file is specified, then the file must have either the `.fa` or `.fasta` extension and must conform to the specification defined in `CustomPrimers.md`.

The `--help` option will list the supported 3rd-party primer sets. Currently the only 3rd-party primer set supported is the set of primers used for 10X Genomics sequencing. Support for detecting and trimming these primers can be enabled by using:

`--primer-sequences 10X_Genomics`

In this case, in addition to detecting and trimming the primers, Dorado will extract the section of the read corresponding to the cell-barcodes and UMI tags and place it in the RX:Z BAM tag.

### RNA adapter trimming

Adapters for RNA004 kits are automatically trimmed during basecalling. However, unlike in DNA, the RNA adapter cannot be trimmed post-basecalling.

### Modified basecalling

Beyond the traditional A, T, C, and G basecalling, Dorado can also detect modified bases such as 5-methylcytosine (5mC), 5-hydroxymethylcytosine (5hmC), and N<sup>6</sup>-methyladenosine (6mA). These modified bases play crucial roles in epigenetic regulation.

To call modifications, extend the models argument with a comma-separated list of modifications:

```
$ dorado basecaller hac,5mCG_5hmCG,6mA pod5s/ > calls.bam
```

In the example above, basecalling is performed with the detection of both 5mC/5hmC in CG contexts and 6mA in all contexts.

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

```bash
$ dorado aligner <index> <reads>  > aligned.bam
```
where `index` is a reference to align to in (FASTQ/FASTA/.mmi) format and `reads` is a folder or file in any HTS format.

When reading from an input folder, `dorado aligner` also supports emitting aligned files to an output folder, which will preserve the file structure of the inputs:

```bash
$ dorado aligner <index> <input_read_folder> --output-dir <output_read_folder>
```

An alignment summary containing alignment statistics for each read can be generated with the `--emit-summary` option. The file will be saved in the `--output-dir` folder.

To basecall with alignment with duplex or simplex, run with the `--reference` option:

```bash
$ dorado basecaller <model> <reads> --reference <index> > calls.bam
```

Alignment uses [minimap2](https://github.com/lh3/minimap2) and by default uses the `lr:hq` preset. This can be overridden by passing a minimap option string, `--mm2-opts`, using the '-x <preset>' option and/or individual options such as `-k` and `-w` to set kmer and window size respectively. For a complete list of supported minimap2 options use '--mm2-opts --help'. For example:
```bash
$ dorado aligner <index> <input_read_folder> --output-dir <output_read_folder> --mm2-opt "-x splice --junc-bed <annotations_file>"
$ dorado aligner <index> <input_read_folder> --output-dir <output_read_folder> --mm2-opt --help
$ dorado basecaller <model> <reads> --reference <index> --mm2-opt "-k 15 -w 10" > calls.bam
```

Note that dorado does support split indexes, however the entire index must be able to fit in memory. Aligning to a split index may result in some spurious secondary and/or supplementary alignments, and the mapping score may not be as reliable as for a non-split index. So it is recommended that, if possible, you generate your `mmi` index files using the `-I` option with a large enough value to generate a non-split index. Or, if you are directly using a large fasta reference, pass a large enough value of the `-I` minimap2 option using `--mm2-opts` to insure that the index is not split.

### Sequencing Summary

The `dorado summary` command outputs a tab-separated file with read level sequencing information from the BAM file generated during basecalling. To create a summary, run:

```
$ dorado summary <bam> > summary.tsv
```

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

In addition to supporting the standard barcode kits from Oxford Nanopore, Dorado also supports specifying custom barcode kit arrangements and sequences. This is done by passing a barcode arrangement file via the `--barcode-arrangement` argument (either to `dorado demux` or `dorado basecaller`) and specifying the `--kit-name` listed in that file. Custom barcode sequences can optionally be specified via the `--barcode-sequences` option. See [here](documentation/CustomBarcodes.md) for more details.

### Poly(A) tail estimation

Dorado has initial support for estimating poly(A) tail lengths for cDNA (PCS and PCB kits) and RNA, and can be configured for use with custom primer sequences, interrupted tails, and plasmids. Note that Oxford Nanopore cDNA reads are sequenced in two different orientations and Dorado poly(A) tail length estimation handles both (A and T homopolymers). This feature can be enabled by passing `--estimate-poly-a` to the `basecaller` command. It is disabled by default. The estimated tail length is stored in the `pt:i` tag of the output record. Reads for which the tail length could not be estimated will have a value of -1 for the `pt:i` tag if the primer anchor for the tail was not found, or a value of 0 if the primer anchor was found, but the length could not be estimated. Custom primer sequences, estimation of interrupted tails, and plasmid support can be configured through the `--poly-a-config` option. See [here](documentation/PolyTailConfig.md) for more details.

### Read Error Correction

Dorado supports single-read error correction with the integration of the [HERRO](https://github.com/lbcb-sci/herro) algorithm. HERRO uses all-vs-all alignment followed by haplotype-aware correction using a deep learning model to achieve higher single-read accuracies. The corrected reads are primarily useful for generating *de novo* assemblies of diploid organisms.

To correct reads, run:
```
$ dorado correct reads.fastq > corrected_reads.fasta
```

Dorado correct only supports FASTQ(.gz) as the input and generates a FASTA file as output. The input can be uncompressed or compressed with `bgz`. An index file is generated for the input FASTQ file in the same folder unless one is already present. Please ensure that the folder with the input file is writeable by the `dorado` process and has sufficient disk space.

The error correction tool is both compute and memory intensive. As a result, it is best run on a system with multiple high performance CPU cores ( >= 64 cores), large system memory ( >= 256GB) and a modern GPU with a large VRAM ( >= 32GB).

All required model weights are downloaded automatically by Dorado. However, the weights can also be pre-downloaded and passed via command line in case of offline execution. To do so, run:
```
$ dorado download --model herro-v1
$ dorado correct -m herro-v1 reads.fastq > corrected_reads.fasta
```
Dorado Correct now also provides a feature to run mapping (CPU-only stage) and inference (GPU-intensive stage) individually. This enables separation of the CPU and GPU heavy stages into individual steps which can even be run on different nodes with appropriate compute characteristics. Example:
```
$ dorado correct reads.fastq --to-paf > overlaps.paf
$ dorado correct reads.fastq --from-paf overlaps.paf > corrected_reads.fasta
```
Gzipped PAF is currently not supported for the `--from-paf` option.

Additionally, if a run was stopped or has failed, Dorado Correct provides a "resume" functionality. The resume feature takes a list of previously corrected reads (e.g. a `.fai` index from the previous run) and skips the previously processed reads:
```
$ samtools faidx corrected_reads.1.fasta    # Output from the previously interrupted run.
$ dorado correct reads.fastq --resume-from corrected_reads.1.fasta.fai > corrected_reads.2.fasta
```
The input file format for the `--resume-from` feature can be any plain text file where the first whitespace-delimited column (or a full row) consists of sequence names to skip, one per row.

#### HPC support
Dorado `correct` now also provides a feature to enable simpler distributed computation.
It is now possible to run a single block of the input target reads file, specified by the block ID. This enables granularization of the correction process, making it possible to easily utilise distributed HPC architectures.

For example, this is now possible:
```
# Determine the number of input target blocks.
num_blocks=$(dorado correct in.fastq --compute-num-blocks)

# For every block, run correction of those target reads.
for ((i=0; i<${num_blocks}; i++)); do
    dorado correct in.fastq --run-block-id ${i} > out.block_${i}.fasta
done

# Optionally, concatenate the corrected reads.
cat out.block_*.fasta > out.all.fasta
```

On an HPC system, individual blocks can simply be submitted to the cluster management system. For example:
```
# Determine the number of input target blocks.
num_blocks=$(dorado correct in.fastq --compute-num-blocks)

# For every block, run correction of those target reads.
for ((i=0; i<${num_blocks}; i++)); do
    qsub ... dorado correct in.fastq --run-block-id ${i} > out.block_${i}.fasta
done
```

In case that the available HPC nodes do not have GPUs available, the CPU power of those nodes can still be leveraged for overlap computation - it is possible to combine a blocked run with the `--to-paf` option. Inference stage can then be run afterwards on another node with GPU devices from the generated PAF and the `--from-paf` option.


#### Troubleshooting
1. In case the process is consuming too much memory for your system, try running it with a smaller index size. For example:
    ```
    $ dorado correct reads.fastq --index-size 4G > corrected_reads.fasta
    ```
2. The auto-computed inference batch size may still be too high for your system. If you are experiencing warnings/errors regarding available GPU memory, try reducing the batch size / selecting it manually. For example:
    ```
    $ dorado correct reads.fastq --batch-size <number> > corrected_reads.fasta
    ```
3. In case your output FASTA file contains a very low amount of corrected reads compared to the input, please check the following:
    - The input dataset has average read length `>=10kbp`. Dorado Correct is designed for long reads, and it will not work on short libraries.
    - Input coverage is reasonable, preferrably `>=30x`.
    - Check the average base qualities of the input dataset. Dorado Correct expects accurate inputs for both mapping and inference.

### Polishing

Dorado `polish` is a high accuracy assembly polishing tool which outperforms similar tools for most ONT-based assemblies.

It takes as input a draft assembly produced by a tool such as [Hifiasm](https://github.com/chhylp123/hifiasm) or [Flye](https://github.com/mikolmogorov/Flye) and aligned reads and outputs an updated version of the assembly.

Additionally, Dorado `polish` can output a VCF file containing records for all variants discovered during polishing, or a gVCF file containing records for all locations in the input draft sequences.

Note that Dorado `polish` is a **haploid** polishing tool and does _not_ implement any sort of phasing internally. It will take input alignment data _as is_ and run it through the polishing model to produce the consensus sequences. For more information, please take a look at [this section](https://dorado-docs.readthedocs.io/en/latest/assembly/polish/#polishing-diploidpolyploid-assemblies) of Dorado Docs.

#### Quick Start

##### Consensus
```bash
# Align unmapped reads to a reference using dorado aligner, sort and index
dorado aligner <draft.fasta> <unmapped_reads.bam> | samtools sort --threads <num_threads> > aligned_reads.bam
samtools index aligned_reads.bam

# Call consensus
dorado polish <aligned_reads.bam> <draft.fasta> > polished_assembly.fasta
```

In the above example, `<aligned_reads>` is a BAM of reads aligned to a draft by Dorado `aligner` and `<draft>` is a FASTA or FASTQ file containing the draft assembly. The draft can be uncompressed or compressed with `bgzip`.

##### Consensus from a FASTQ input instead of BAM

In case a FASTQ file was produced during basecalling instead of a BAM file, you will need to provide a flag `--add-fastq-rg` to Dorado `aligner` to have it generate the proper BAM header required for Dorado `polish`.

Note that this may take some time to run because it requires an extra pass over the input data prior to alignment.

This feature supports only FASTQ files with HTS-style tags in the header and will not work for the old MinKnow style FASTQ files.

Here is a full example:
```bash
# Align reads to a reference using dorado aligner, sort and index
dorado aligner --add-fastq-rg <draft.fasta> <reads.fastq> | samtools sort --threads <num_threads> > aligned_reads.bam
samtools index aligned_reads.bam

# Call consensus
dorado polish <aligned_reads.bam> <draft.fasta> > polished_assembly.fasta
```

##### Consensus on bacterial genomes
```bash
dorado polish <aligned_reads> <draft> --bacteria > polished_assembly.fasta
```

This will automatically resolve a suitable bacterial polishing model, if one exits for the input data type.

##### Variant calling
```bash
dorado polish <aligned_reads> <draft> --vcf > polished_assembly.vcf
dorado polish <aligned_reads> <draft> --gvcf > polished_assembly.all.vcf
```

Specifying `--vcf` or `--gvcf` flags will output a VCF file to stdout instead of the consensus sequences.

##### Output to a folder
```bash
dorado polish <aligned_reads> <draft> -o <output_dir>
```

Specifying `-o` will write multiple files to a given output directory (and create the directory if it doesn't exist):
- Consensus file: `<output_dir>/consensus.fasta` by default, or `<output_dir>/consensus.fastq` if `--qualities` is specified.
- VCF file: `<output_dir>/variants.vcf` which contains only variant calls by default, or records for all positions if `--gvcf` is specified.

#### Resources

Dorado `polish` will automatically select the compute resources to perform polishing. It can use one or more GPU devices, or the CPU, to call consensus.

To specify resources manually use:
- `-x / --device` - to specify specific GPU resources (if available).
- `--threads` -  to set the maximum number of threads to be used for everything but the inference.
- `--infer-threads` -  to set the number of CPU threads for inference (when "--device cpu" is used).
- `--batchsize` - batch size for inference, important to control memory usage on the GPUs. Automatically computed by default (`--batchsize 0`).

Example:

```bash
dorado polish reads_to_draft.bam draft.fasta --device cuda:0 --threads 24 > consensus.fasta
```

#### Models

Dorado `polish` auto-resolves the polishing model based on the input BAM file. The BAM file needs to contain the `@RG` headers with the basecaller model name specified, otherwise the model will not be resolved. If the input BAM records contain move tables, an appropriate move-aware polishing model will be selected.

Once the model is resolved, Dorado `polish` will either download it or look it up in the models-directory if specified.

For example:
```bash
dorado polish reads_to_draft.bam draft.fasta > consensus.fasta
```
will find the compatible model based on the input BAM file and download it to a temporary folder.

When `--models-directory` is specified, the resolved polishing model will first be looked up in the models-directory, and only downloaded if the model does not exist. The specified models-directory must exist. Example:
```bash
mkdir -p models
dorado polish --models-directory models reads_to_draft.bam draft.fasta > consensus.fasta
```

More information about the `--models-directory` can be found in [this section](#model-search-directory-and-temporary-model-downloads)

If there are multiple read groups in the input dataset which were generated using different basecaller models, Dorado `polish` will report an error and stop execution.

##### Move Table Aware Models

Significantly more accurate assemblies can be produced by giving the polishing model access to additional information about the underlying signal for each read. For more information, see this section from the [NCM 2024](https://youtu.be/IB6DmU40NIU?t=377) secondary analysis update.

Dorado `polish` includes models which can use the move table to get temporal information about each read. These models will be selected automatically if the corresponding `mv` tag is in the input BAM. To do this, pass the `--emit-moves` tag to Dorado `basecaller` when basecalling. To check if a BAM contains the move table for reads, use samtools:
```bash
samtools view --keep-tag "mv" -c <reads_to_draft_bam>
```

The output should be equal to the total number of reads in the bam (`samtools view -c <reads_to_draft_bam>`).

If move tables are not available in the BAM, then the non-move table-aware model will be automatically selected.

#### FAQ

##### How is Dorado `polish` different from Medaka?

[Medaka](https://github.com/nanoporetech/medaka) and Dorado `polish` are both assembly polishing tools. They accept the same input formats and produce the same output formats, and in principle they could run the same polishing model to produce equivalent results. However, Dorado `polish` is optimised for higher performance, and can support more accurate models with more computationally intensive architectures. For use cases in low-resource settings (small genomes such as bacteria with CPUs only available) Medaka remains the recommended tool. For large genomes or in other instances where speed is important, we suggest trying Dorado `polish`.

##### Should I use `correct` or `polish`?

Dorado `polish` is a post-assembly tool and it is intended to improve the accuracy of pre-existing
assemblies. Dorado `correct` conversely is a pre-assembly tool and is intended to improve the
contiguity of an assembly by improving the fidelity of reads used to create it.

##### "How do I go from raw POD5 data to a polished T2T assembly?"
Here is a high-level example workflow:
```bash
# Generate basecalled data with dorado basecaller
dorado basecaller <model> pod5s/ --emit-moves > calls.bam
samtools fastq calls.bam > calls.fastq

# Apply dorado correct to a set of reads that can be used as input in an assembly program.
dorado correct calls.fastq > corrected.fasta

# Assemble the genome using those corrected reads
<some_assembler> --input corrected.fasta > draft_assembly.fasta

# Align original calls to the draft assembly
dorado aligner draft_assembly.fasta calls.bam > aligned_calls.bam

# Run dorado polish using the raw reads aligned to the draft assembly
dorado polish aligned_calls.bam draft_assembly.fasta > polished_assembly.fasta
```

##### Polishing diploid/polyploid assemblies

Dorado `polish` is a **haploid** polishing tool and does _not_ implement any sort of phasing internally. It will take input alignment data _as is_ and run it through the polishing model to produce the consensus sequences.

In order to polish diploid/polyploid assemblies, it is up to the user to properly separate haplotypes before giving the data to Dorado `polish`.

We are currently working on a set of best practices. In the meantime, an unofficially suggested approach to polish diploid genomes would be to align the reads using the `lr:hqae` [Minimap2 setting](https://github.com/lh3/minimap2/releases/tag/v2.28) as this was specifically designed for alignment back to a diploid genome. This setting is available through Dorado `aligner` using the following option:
```bash
dorado aligner --mm2-opts "-x lr:hqae" <ref> <reads>
```

##### Troubleshooting

###### Memory consumption / Torch out-of-memory (OOM) issues

The inference batch size is computed to fit the largest possible batches into the available GPU memory (default `--batchsize 0`).

There are two cases when an OOM issue can happen:

1. The auto batch size feature is underestimating the memory consumption. If an Out-Of-Memory (OOM) warning/error is raised with the default auto batch size, try setting the batch size manually to a fixed value instead. For example:
    ```bash
    dorado polish reads_to_draft.bam draft.fasta --batchsize <number> > consensus.fasta
    ```
    A good rule of thumb would be `--batchsize 16` for a large GPU, or try using a smaller value if this is still too high.

    Additionally, the number of inference workers can be reduced to lower the memory usage (the default is `2` workers per device):
    ```bash
    dorado polish reads_to_draft.bam draft.fasta --infer-threads 1 > consensus.fasta
    ```

    Alternatively, consider running inference on the CPU, although this can take longer:
    ```bash
    dorado polish reads_to_draft.bam draft.fasta --device "cpu" > consensus.fasta
    ```

    Note that using multiple CPU inference threads can cause much higher memory usage.

2. GPU memory fragmentation during the run. This can happen when there were many small allocations followed by a large memory allocation which then cannot be fitted into a single contiguous block of memory. Such errors will have a specific Torch error message which looks like this:
    > Exception caught: CUDA out of memory. Tried to allocate 15.12 GiB. GPU 1 has a total capacity of 31.73 GiB of which 14.77 GiB is free. Including non-PyTorch memory, this process has 16.95 GiB memory in use. Of the allocated memory 2.10 GiB is allocated by PyTorch, and 14.46 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

    The key portion here is: `2.10 GiB is allocated by PyTorch, and 14.46 GiB is reserved by PyTorch but unallocated.`, which means that almost all non-free memory is actually unused.

    In this case, follow the suggestion from the error message, and it should resolve the issue.

    Example:
    ```
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True dorado polish reads_to_draft.bam draft.fasta > consensus.fasta
    ```

###### "[error] Could not open index for BAM file: 'aln.bam'!"

Example message:
```bash
$ dorado polish aln.bam assembly.fasta > polished.fasta
[2024-12-23 07:18:23.978] [info] Running: "polish" "aln.bam" "assembly.fasta"
[E::idx_find_and_load] Could not retrieve index file for 'aln.bam'
[2024-12-23 07:18:23.987] [error] Could not open index for BAM file: 'aln.bam'!
```

This message means that there the input BAM file does not have an accompanying index file `.bai`. This may also mean that the input BAM file is not sorted, which is a prerequisite for producing the `.bai` index using `samtools`.

Dorado `polish` requires input alignments to be produced using Dorado `aligner`. When Dorado `aligner` outputs alignments to `stdout`, they are not sorted automatically. Instead, `samtools` needs to be used to sort and index the BAM file. For example:
```bash
dorado aligner <draft.fasta> <reads.bam> | samtools sort --threads <num_threads> > aln.bam
samtools index aln.bam
```
Note that the sorting step is added after the pipe symbol.

The output from dorado aligner is already sorted when the output is to a folder, specified using the `--output-dir` option.
```bash
dorado aligner --output-dir <out_dir> <draft.fasta> <reads.bam>
```

###### "[error] Input BAM file has no basecaller models listed in the header."

Dorado `polish` requires that the aligned BAM has one or more `@RG` lines in the header. Each `@RG` line needs to contain a basecaller model used for generating the reads in this group. This information is required to determine the compatibility of the selected polishing model, as well as for auto-resolving the model from data.

When using Dorado `aligner` please provide the input basecalled reads in the BAM format. The basecalled reads BAM file (e.g. `calls.bam`) contains the `@RG` header lines, and this will be propagated into the aligned BAM file. Example:
```bash
dorado aligner draft.fasta calls.bam | samtools sort --threads <num_threads> > aligned_reads.bam
samtools index aligned_reads.bam
```
Alternatively, Dorado `aligner` will automatically sort and index the alignments when an output directory is specified instead of `stdout`.
```bash
dorado aligner --output-dir out draft.fasta calls.bam
```

However, if input basecalled reads are given in the **FASTQ** format, the aligned BAM file will _not_ contain `@RG` lines by default.
In this case, a flag `--add-fastq-rg` can be passed to Dorado `aligner`. Dorado `aligner` will then perform an additional pass over the input FASTQ data and collect all the read group / basecaller information and add it to the header.

Note that this feature will only work for the HTS-style FASTQ headers, such as:
```bash
@74960cfd-0b82-43ed-ae04-05162e3c0a5a qs:f:27.7534 du:f:75.1604 ns:i:375802 ts:i:1858 mx:i:1 ch:i:295 st:Z:2024-08-29T22:06:03.400+00:00 rn:i:585 fn:Z:FBA17175_7da7e070_f8e851a5_5.pod5 sm:f:414.101 sd:f:107.157 sv:Z:pa dx:i:0 RG:Z:f8e851a5d56475e9ecaa43496da18fad316883d8_dna_r10.4.1_e8.2_400bps_sup@v5.0.0
```

Example usage:
```bash
dorado aligner --add-fastq-rg --output-dir out draft.fasta calls.bam
```

Dorado `polish` currently supports data generated using only the simplex basecallers.

###### "[error] Input BAM file was not aligned using Dorado."

Dorado `polish` accepts only BAMs aligned with Dorado `aligner`. Aligners other than Dorado `aligner` are not supported.

Example usage:
```bash
dorado aligner <draft.fasta> <reads.bam> | samtools sort --threads <num_threads> > aln.bam
samtools index aln.bam
```

###### "[error] The input BAM contains more than one read group. Please specify --RG to select which read group to process."

It is possible that the input BAM file contains more than 1 read group. In this case, Dorado `polish` requires that a single read group is selected for processing using the `--RG <id>` command line argument. The `<id>` should exactly match the `ID:` field in one of the `@RG` lines in the input BAM/SAM file.

Specifying the `--RG` option will filter out any read which does not belong to that read group and will apply the appropriate polishing model for that read group based on the basecaller model specified in the corresponding `@RG` line in the input BAM file.

Specifying a read group which corresponds to duplex data will not work because Dorado `polish` currently does not have duplex polishing models available.

In case of a duplex BAM - note that by default the simplex parents of the duplex reads will also be present in the output BAM file from Dorado. Consider filtering these out first if this could bias your results.

###### "[error] Duplex basecalling models are not supported."

Dorado `polish` currently supports data generated using only the simplex basecallers.

###### I created a merged BAM file composed of multiple different data types. Why can't I polish it? Using `--ignore-read-groups` does not help either.

In case you created a merged BAM file, one of the following scenarios is possible:

1. **There are zero read groups in the merged BAM file.** Something went wrong in the process of data preparation. There needs to be at least one read group in the BAM file which links the data to a basecaller model.
2. **The merged BAM file has only one read group.** This is the best option, and merging was performed in a way that all colliding `@RG` headers were merged too. Since there is only one read group, there is also one basecaller model for the entire merged BAM dataset.
3. **The merged BAM file has more than one read group, but only a single basecaller model.** This can occur when data originally belonged to the same read group but the colliding read groups were not merged in the process (check the `-c` option of `samtools merge`). For example, `samtools merge` will add a unique hash to the end of each read group, because the prefix of the read groups is the same (e.g. `bc8993f4557dd53bf0cbda5fd68453fea5e94485_dna_r10.4.1_e8.2_400bps_hac@v5.0.0-1C79A650` and `bc8993f4557dd53bf0cbda5fd68453fea5e94485_dna_r10.4.1_e8.2_400bps_hac@v5.0.0-6E00935B`). Alternatively, data from multiple sequencing runs were combined, but the same basecaller model was used in all cases.
    - Using `--ignore-read-groups` will run the process using all data in this case, since it was generated using a single basecaller model.
    - Alternatively, using `--RG <read_group_id>` will select only reads which belong to this specific read group, and ignore all other reads.
    - Auto model detection is possible from the BAM file in this case, since only one basecaller model was used to produce the data.
4. **The merged BAM file has more than one read group and _more than one basecaller model_.** One or more read groups were generated using one particular basecaller model, while some other read groups were generated using another particular basecaller model. (For example, combining old and new data.) Sometimes, users may attempt to combine simplex and duplex reads into the same BAM file.
    - Dorado `polish`/`variant` can use only one selected model for inference. All currently available models were trained on individual data types (data generated by a single basecaller version) and not on a mixture of data (with the exception of the bacterial methylation polishing model). Running any model on a mixture of data may produce inferior results. This is why Dorado `polish` and Dorado `variant` enforce that only a single basecaller model is present in the input.
    - In this case, not even `--ignore-read-groups` will work because there was more than one basecaller model used to produce the data in this BAM file.
    - Using `--RG <read_group_id>` will select only reads which belong to one specific read group, and ignore all other reads.
    - Using the auto model selection cannot resolve a model from a BAM file if the input BAM file contains multiple models.
    - Auto model selection in this case is only possible if `--RG` is used.
    - Duplex basecaller models are not supported by Dorado `polish` or Dorado `variant`.

### Variant Calling - Alpha preview release

Dorado `variant` is an early-stage diploid small variant caller, released for experimental use and evaluation purposes.
This version is intended for feedback and should not yet be considered production-ready.

> **Should I use `variant` or `polish`?**
Dorado variant is a short variant caller for diploid samples aligned to a haploid species reference (e.g. GRCh38) whereas `polish` is intended for workflows involving reads aligned to a haplotype-resolved (or haploid) draft assembly.

Although Dorado `polish` can also generate a VCF file of variants, there are some substantial distinctions between the two tools.

| `dorado polish`             | `dorado variant`             |
| -------------------- | -------------------- |
| - Polishing of draft assemblies<br>- Input is a haplotype-resolved draft assembly<br>- Output is a polished sequence<br>- Optionally, a VCF/gVCF of diffs is output<br>- Uses specialised polishing models| - Diploid variant calling<br>- Input is a reference genome<br>- Output is a VCF/gVCF of called diploid variants<br>- Uses specialised variant calling models|

#### Quick Start

```bash
# Align the reads using dorado aligner, sort and index
dorado aligner <ref.fasta> <reads.bam> | samtools sort --threads <num_threads> > aligned_reads.bam
samtools index aligned_reads.bam

# Call variants
dorado variant <aligned_reads.bam> <ref.fasta> > variants.vcf
```

For this preview release, current models require signal-level information encoded in the move tables in the input BAM file. This requires the `--emit-moves` flag to be set during basecalling.

In case the input basecalled reads are in a FASTQ format with the HTS-style ONT tags, please use the `--add-fastq-rg` option with Dorado `aligner` to ensure proper header formatting:
```bash
# Align the reads using dorado aligner, sort and index
dorado aligner --add-fastq-rg <ref.fasta> <reads.fastq> | samtools sort --threads <num_threads> > aligned_reads.bam
samtools index aligned_reads.bam
```

##### Output to a folder

```bash
dorado variant <aligned_reads> <reference> -o <output_dir>
```

Specifying `-o` will write the output to one or more files stored in the given output directory (and create the directory if it doesn't exist). Concretely:

- VCF file: `<output_dir>/variants.vcf` which contains only variant calls by default, or records for all positions if `--gvcf` is specified.

#### Resources

Dorado `variant` will automatically select the compute resources to perform variant calling. It can use one or more GPU devices. Variant calling can be performed on CPU-only, but we highly recommend to run on GPU for desired performance. High-memory GPUs are recommended to run this tool.

To specify resources manually use:

- `-x / --device` - to specify specific GPU resources (if available).
- `--threads` - to set the maximum number of threads to be used for everything but the inference.
- `--infer-threads` - number of inference workers to use (per device). For CPU-only runs, this specifies the number of CPU inference threads.
- `--batchsize` - batch size for inference, important to control memory usage on the GPUs. Automatically computed by default (`--batchsize 0`).

Example:

```bash
dorado variant aligned_reads.bam reference.fasta --device cuda:0 --threads 24 > variants.vcf
```

#### Models

By default, `variant` queries the BAM and selects the best model for the basecalled reads, if supported.

Alternatively, a model can be selected through the command line in the following way:

```bash
dorado variant --model <value> ...
```

| Value    | Description |
| -------- | ------- |
| auto  | Determine the best compatible model based on input data. |
| \<basecaller_model\> | Simplex basecaller model name (e.g. `dna_r10.4.1_e8.2_400bps_hac@v5.0.0`) |
| \<variant_model\> | Variant calling model name (e.g. `dna_r10.4.1_e8.2_400bps_hac@v5.0.0_variant_mv@v1.0`) |
| \<path\> | Local path on disk where the model can be loaded from. |

When the `auto` or the `<basecaller_model>` syntax is used the most recent version of a compatible model will be selected for variant calling.

Current variant calling models require the presence of move tables in the input BAM file. Move tables need to be exported during basecalling.

If a non-compatible model is selected for the input data, or if there are multiple read groups in the input dataset which were generated using different basecaller models, Dorado `variant` will report an error and stop execution.

##### Supported basecaller models

- `dna_r10.4.1_e8.2_400bps_hac@v5.0.0`

More models will be supported in the near future. This is an alpha release.

#### Common questions and Troubleshooting

##### I created a merged BAM file composed of multiple different data types. Why can't I call variants on this dataset? Using `--ignore-read-groups` does not help either.

Please see the following section in Dorado `polish`:
[I created a merged BAM file composed of multiple different data types](https://dorado-docs.readthedocs.io/en/latest/assembly/polish/#i-created-a-merged-bam-file-composed-of-multiple-different-data-types-why-cant-i-polish-it-using-ignore-read-groups-does-not-help-either)


##### Memory consumption / Torch out-of-memory (OOM) issues

The inference batch size is computed to fit the largest possible batches into the available GPU memory (default `--batchsize 0`).

There are two cases when an OOM issue can happen:

1. The auto batch size feature is underestimating the memory consumption. If an Out-Of-Memory (OOM) warning/error is raised with the default auto batch size, try setting the batch size manually to a fixed value instead. For example:
    ```bash
    dorado variant aligned_reads.bam reference.fasta --batchsize <number> > variants.vcf
    ```
    A good rule of thumb would be `--batchsize 10` for a large GPU, or try using a smaller value if this is still too high.

    Additionally, the number of inference workers can be reduced to lower the memory usage (the default is `2` workers per device):
    ```bash
    dorado variant aligned_reads.bam reference.fasta --infer-threads 1 > variants.vcf
    ```

2. GPU memory fragmentation during the run. This can happen when there were many small allocations followed by a large memory allocation which then cannot be fitted into a single contiguous block of memory. Such errors will have a specific Torch error message which looks like this:
    > Exception caught: CUDA out of memory. Tried to allocate 15.12 GiB. GPU 1 has a total capacity of 31.73 GiB of which 14.77 GiB is free. Including non-PyTorch memory, this process has 16.95 GiB memory in use. Of the allocated memory 2.10 GiB is allocated by PyTorch, and 14.46 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

    The key portion here is: `2.10 GiB is allocated by PyTorch, and 14.46 GiB is reserved by PyTorch but unallocated.`, which means that almost all non-free memory is actually unused.

    In this case, follow the suggestion from the error message, and it should resolve the issue.

    Example:
    ```
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True dorado variant aligned_reads.bam reference.fasta > variants.vcf
    ```

##### "[error] Input BAM file was not aligned using Dorado."

Dorado `variant` accepts only BAMs aligned with Dorado `aligner`. Aligners other than Dorado `aligner` are not supported.

Example usage:
```bash
dorado aligner <draft.fasta> <reads.bam> | samtools sort --threads <num_threads> > aln.bam
samtools index aln.bam
```

###### "[error] Input BAM file has no basecaller models listed in the header."

Please refer to this [section](#error-input-bam-file-has-no-basecaller-models-listed-in-the-header).

##### "[error] Duplex basecalling models are not supported."

Dorado `variant` currently supports data generated using only the simplex basecallers.

##### Does Dorado Variant phase variants?

At this early stage, Dorado `variant` does not yet produce phased VCF variants. This is work in progress.


## Available basecalling models

To download all available Dorado models, run:

```
$ dorado download --model all
```

### Decoding Dorado model names

The names of Dorado models are systematically structured, each segment corresponding to a different aspect of the model, which include both chemistry and run settings. Below is a sample model name explained:

`dna_r10.4.1_e8.2_400bps_hac@v5.2.0`

- **Analyte Type (`dna`)**: This denotes the type of analyte being sequenced. For DNA sequencing, it is represented as `dna`. If you are using a Direct RNA Sequencing Kit, this will be `rna004`.

- **Pore Type (`r10.4.1`)**: This section corresponds to the type of flow cell used. For instance, FLO-MIN114/FLO-FLG114 is indicated by `r10.4.1`, while FLO-MIN106D/FLO-FLG001 is signified by `r9.4.1`.

- **Chemistry Type (`e8.2`)**: This represents the chemistry type, which corresponds to the kit used for sequencing. For example, Kit 14 chemistry is denoted by `e8.2` and Kit 10 or Kit 9 are denoted by `e8`.

- **Translocation Speed (`400bps`)**: This parameter, selected at the run setup in MinKNOW, refers to the speed of translocation. Prior to starting your run, a prompt will ask if you prefer to run at 260 bps or 400 bps. The former yields more accurate results but provides less data. As of MinKNOW version 23.04, the 260 bps option has been deprecated.

- **Model Type (`hac`)**: This represents the size of the model, where larger models yield more accurate basecalls but take more time. The three types of models are `fast`, `hac`, and `sup`. The `fast` model is the quickest, `sup` is the most accurate, and `hac` provides a balance between speed and accuracy. For most users, the `hac` model is recommended.

- **Model Version Number (`v5.2.0`)**: This denotes the version of the model. Model updates are regularly released, and higher version numbers typically signify greater accuracy.


### **DNA models:**

Below is a table of the available basecalling models and the modified basecalling models that can be used with them. The bolded models are for the latest released condition with 5 kHz data.

The versioning of modification models is bound to the basecalling model. This means that the modification model version is reset for each new simplex model release. For example, `6mA@v1` compatible with `v4.3.0` basecalling models is more recent than `6mA@v2` compatible with `v4.2.0` basecalling models.

| Basecalling Models | Compatible<br />Modifications | Modifications<br />Model<br />Version | Data<br />Sampling<br />Frequency |
| :-------- | :------- | :--- | :--- |
| **dna_r10.4.1_e8.2_400bps_fast@v5.2.0** | | | 5 kHz |
| **dna_r10.4.1_e8.2_400bps_hac@v5.2.0** | 4mC_5mC<br />5mCG_5hmCG<br />5mC_5hmC<br />6mA<br /> | v1<br />v2<br />v2<br />v1 | 5 kHz |
| **dna_r10.4.1_e8.2_400bps_sup@v5.2.0** | 4mC_5mC<br />5mCG_5hmCG<br />5mC_5hmC<br />6mA<br /> | v1<br />v2<br />v2<br />v1 | 5 kHz |
| dna_r10.4.1_e8.2_400bps_fast@v5.0.0 | | | 5 kHz |
| dna_r10.4.1_e8.2_400bps_hac@v5.0.0 | 4mC_5mC<br />5mCG_5hmCG<br />5mC_5hmC<br />6mA<br /> | v3<br />v3<br />v3<br />v3 | 5 kHz |
| dna_r10.4.1_e8.2_400bps_sup@v5.0.0 | 4mC_5mC<br />5mCG_5hmCG<br />5mC_5hmC<br />6mA<br /> | v3<br />v3<br />v3<br />v3 | 5 kHz |
| dna_r10.4.1_e8.2_400bps_fast@v4.3.0 | | | 5 kHz |
| dna_r10.4.1_e8.2_400bps_hac@v4.3.0 | 5mCG_5hmCG<br />5mC_5hmC<br />6mA<br /> | v1<br />v1<br />v2 | 5 kHz |
| dna_r10.4.1_e8.2_400bps_sup@v4.3.0 | 5mCG_5hmCG<br />5mC_5hmC<br />6mA<br /> | v1<br />v1<br />v2 | 5 kHz |
| dna_r10.4.1_e8.2_400bps_fast@v4.2.0 | 5mCG_5hmCG | v2 | 5 kHz |
| dna_r10.4.1_e8.2_400bps_hac@v4.2.0 | 5mCG_5hmCG | v2 | 5 kHz |
| dna_r10.4.1_e8.2_400bps_sup@v4.2.0 | 5mCG_5hmCG<br />5mC_5hmC<br />5mC<br />6mA<br />| v3.1<br />v1<br />v2<br />v3| 5 kHz |

### **RNA models:**

**Note:** The BAM format does not support `U` bases. Therefore, when Dorado is performing RNA basecalling, the resulting output files will include `T` instead of `U`. This is consistent across output file types.

| Basecalling Models | Compatible<br />Modifications | Modifications<br />Model<br />Version | Data<br />Sampling<br />Frequency |
| :-------- | :------- | :--- | :--- |
| **rna004_130bps_fast@v5.2.0** | | | 4 kHz |
| **rna004_130bps_hac@v5.2.0** | m5C<br />m6A_DRACH<br />inosine_m6A<br />pseU | v1<br />v1<br />v1<br />v1 | 4 kHz |
| **rna004_130bps_sup@v5.2.0** | m5C_2OmeC<br />m6A_DRACH<br />inosine_m6A_2OmeA<br />pseU_2OmeU<br />2OmeG | v1<br />v1<br />v1<br />v1<br />v1 | 4 kHz |
| rna004_130bps_fast@v5.1.0 | | | 4 kHz |
| rna004_130bps_hac@v5.1.0 | m5C<br />m6A_DRACH<br />inosine_m6A<br />pseU | v1<br />v1<br />v1<br />v1 | 4 kHz |
| rna004_130bps_sup@v5.1.0 | m5C<br />m6A_DRACH<br />inosine_m6A<br />pseU | v1<br />v1<br />v1<br />v1 | 4 kHz |
| rna004_130bps_fast@v5.0.0 | | | 4 kHz |
| rna004_130bps_hac@v5.0.0 | m6A<br />m6A_DRACH<br />pseU | v1<br />v1<br />v1 | 4 kHz |
| rna004_130bps_sup@v5.0.0 | m6A<br />m6A_DRACH<br />pseU | v1<br />v1<br />v1 | 4 kHz |
| rna004_130bps_fast@v3.0.1 | | | 4 kHz |
| rna004_130bps_hac@v3.0.1 | | | 4 kHz |
| rna004_130bps_sup@v3.0.1 | m6A_DRACH | v1 | 4 kHz |

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

### Model search directory and temporary model downloads

Once the automatic model selection process has found the appropriate model given the input data, it will search for existing model directories to avoid downloading models unnecessarily. The behaviour of this search can be controlled as follows:

1. Setting the `--models-directory` CLI argument - The `--models-directory` argument can be used to specify a directory where models will be searched.
2. Setting the `DORADO_MODELS_DIRECTORY` environment variable - This is the same as setting `--models-directory` but has lower priority than the CLI equivalent.
3. If neither `--models-directory` or `DORADO_MODELS_DIRECORY` are set then the current working directory is searched.

If `--models-directory` or `DORADO_MODELS_DIRECTORY` is set automatically downloaded models will persist, otherwise models will be downloaded into a local temporary directory
and deleted after dorado has finished.

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
This will generate multiple BAMs which can be merged. This approach also offers some resilience as if any job fails it can be restarted without having to re-run basecalling against the entire dataset.

### GPU Out of Memory Errors

Dorado operates on a broad range of GPUs but it is primarily developed for Nvidia A100/H100 and Apple Silicon. Dorado attempts to find the optimal batch size for basecalling. Nevertheless, on some low-RAM GPUs, users may face out of memory crashes.

A potential solution to this issue could be setting a manual batch size using the following command:

```
dorado basecaller --batchsize 64 ...
```

**Note:** Reducing memory consumption by modifying the `chunksize` parameter is not recommended as it influences the basecalling results.

### Low GPU Utilization

Low GPU utilization can lead to reduced basecalling speed. This problem can be identified using tools such as `nvidia-smi` and `nvtop`. Low GPU utilization often stems from I/O bottlenecks in basecalling. Here are a few steps you can take to improve the situation:

1. Transfer data to the local disk before basecalling: Slow basecalling often occurs because network disks cannot supply Dorado with adequate speed. To mitigate this, make sure your data is as close to your host machine as possible.
2. Choose SSD over HDD: Particularly for duplex basecalling, using a local SSD can offer significant speed advantages. This is due to the duplex basecalling algorithm's reliance on heavy random access of data.

### Windows PowerShell encoding

When running in PowerShell on Windows, care must be taken, as the default encoding for application output is typically UTF-16LE.  This will cause file corruption if standard output is redirected to a file.  It is recommended to use the `--output-dir` argument to emit BAM files if PowerShell must be used.  For example, the following command will create corrupt output which cannot be read by samtools:

```
PS > dorado basecaller <args> > out.bam
```

Instead, use:

```
PS > dorado basecaller <args> --output-dir .
```

For text-based output formats (SAM or FASTQ), it is possible to override the encoding on output using the `out-file` command.  This command will produce a well formed ascii SAM file:

```
PS > dorado basecaller <args> --emit-sam | out-file -encoding Ascii out.sam
```

Note that `out-file` with `Ascii` encoding will not produce well-formed BAM files.

Read more about Powershell output encoding [here](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_character_encoding?view=powershell-7.4).

## Licence and Copyright

(c) 2024 Oxford Nanopore Technologies PLC.

Dorado is distributed under the terms of the Oxford Nanopore Technologies PLC.  Public License, v. 1.0.  If a copy of the License was not distributed with this file, You can obtain one at http://nanoporetech.com
