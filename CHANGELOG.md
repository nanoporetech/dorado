# Changelog

All notable changes to Dorado will be documented in this file.

# [0.8.3] (11 Nov 2024)

This release of Dorado includes fixes and improvements to the Dorado 0.8.2 release, including a fix to SUP basecalling on Apple Silicon.

* 03e5eecd22f48ea13ce3fc8f65269fab96fa1c27 - Fix incorrect basecalls being emitted from SUP models on Apple Silicon
* 987a237e192f28dc474baaea8d2aca81b2f5d57b - Improve error messaging when unable to run basecalling kernels on Apple GPU


# [0.8.2] (21 Oct 2024)

This release of Dorado includes fixes and improvements to the Dorado 0.8.1 release.

* 9a81cbc269b742cad64b514a50aeb30a7bf25811 - Fix errors when running `dorado basecaller` on multi-GPU systems
* bac446937f5bc378ebf14169d7287b6e4c5b5efb - Fix error when passing invalid option to basecaller `--modified-bases` argument
* 7f40154aed13324865123386b278f3782eafa107 - Fix bug when running on GPUs with less than the recommended VRAM
* a993a32f6904991135ce14387740b0216b2136c9 - Prevent loss of small numbers of reads from `dorado correct`
* bbbeb9a72a79e77101f67dbab2a1c5f1cc4dd1d3 - Fix error when `dorado demux` is provided an empty input directory
* 3a8094a134dc448803c00aa93f6160c96241a9ab - Clarify "Unable to find chunk benchmarks" warning to indicate that it is not an error
* 543ccc1a4b78c8d04960a769d03d1f61ca75d0e3 - Improve documentation on running Dorado in PowerShell on Windows


# [0.8.1] (3 Oct 2024)

This release of Dorado includes fixes and improvements to the Dorado 0.8.0 release, including corrected configuration for DNA v5 SUP-compatible 5mC_5hmC and 5mCG_5hmCG models, improved cDNA poly(A) tail estimation for data from MinION flow cells, reduced basecaller startup time on supported GPUs, and more.

* f74d8917ba3472e07ae4d00bb2e3a745048c8c84 - Corrected bug causing dna_r10.4.1_e8.2_400bps_sup@v5.0.0_5mC_5hmC@v2 to call CpG contexts only and dna_r10.4.1_e8.2_400bps_sup@v5.0.0_5mCG_5hmCG@v2 to call all contexts
* eb4649442bbe1ef84eb72fd62e30586c9a45c10d - Improve cDNA poly(A) tail estimation for MinION flow cells
* 762e88689b31099081a7fcd39b959ddc4c7eb2e1 - Cache batch sizes to significantly reduce basecaller startup time on supported GPUs
* 22269a8eed928ed04fbf7731ff99fe4454f87493 - Prevent "Trim interval <range> is invalid for sequence" error when performing trimming
* f156ae6237fa4602d759dc6ee05b1b367e77c09c - Prevent write permission error for model download folder when file write is not required
* fcb9d53ab168f42cdd02cfa740158a0ce0b5cc84 - Include run name in output files from `dorado demux` even if input files are FASTQ
* a4c96490a4409d961ce59b642478ff453d217776 - BED file handling: only split columns on tabs, not spaces; load files with spaces in region names
* e62cbc851a087d778fbde24ebe0ae85b96d28a9e - Allow comment lines in the middle of the BED file
* f15c0b38286dccda86e3244e535e69e30edb60cd - Fix compilation in AppleClang 16


# [0.8.0] (16 Sept 2024)

This release of Dorado adds v5.1 RNA models with new `inosine_m6A` and `m5C` RNA modified base models, updates existing modified base models, improves the speed of v5 SUP basecalling models on A100/H100 GPUs, and enhances the flexibility and stability of `dorado correct`. It also introduces per-barcode configuration for poly(A) estimation with interrupted tails, adds new `--output-dir` and `--bed-file` arguments to Dorado basecalling commands, and includes a variety of other improvements for stability and usability.

* a69c0a2987e60f3889cc56cd820e8a7713887f33 - Add v5.1.0 RNA basecalling models, including new `inosine_m6A` and `m5C` modified base models, and updated existing DNA and RNA modified base models
* 8e3a8707be5248d7bcc47d3e89b80c0bdc9c2f36 - Improve speed of v5 SUP basecalling models on A100 and H100 GPUs
* 6ee90189197d11bfe50e919067582da6eccf513e - Reduce false positive calls from v5 DNA modifed base models
* 69cb26032d8393a781a9a3d32aa2ceb13ec65491 - Fix bug causing intermittent crashing with v5 SUP models
* e9dec497a38fa2a1935f64d30e35db246da58a08 - Add `--resume-from` functionality to `dorado correct`
* cb6eee1c3d63da2f1f11fb8fcc63418908154f81 - Decouple alignment and inference stages in `dorado correct`
* df861db10d77b4056702857ca11d2e50b63946af - Prevent segfaults in `dorado correct`
* f35c8cc3ebf900cfd6e19cf6ebaebc94fbb8619b - Fix bug when downloading models for `dorado correct`
* 66467011c1f22f7037e1055bb435bef090790dd1 - Add per-barcode poly(A) configuration for interrupted tails
* 0b79407afdfc8f4fa66d5393a722e29358a6302d - Improve poly(A) length estimation for RNA and DNA
* df614abee24523abc2858b02d18205b9ebca53fe - Add `--output-dir` argument to `dorado basecaller` and `dorado duplex`
* f9beb393cd8237a142dba43f3b04a77cb688d1c0 - Add `--bed-file` argument to `dorado basecaller` and `dorado duplex`
* 1fc6f1eb5a535262ef601a8fe4674edc87a137c9 - Add `--models-directory` option to `basecaller`, `duplex`, and `download` to download and reuse models
* 966c2ca38369a21855cdd491b025979a9628b5b5 - Update POD5 version to v0.3.15
* 6ec77c8b6cfc3a53433ae27f7a5383f77097eefa - Fix errors when performing duplex calling with modified bases
* 4a28d589d5e244f62543ebb4d744e8c2843bde93 - Always trim DNA adapter signal before processing RNA reads
* a90fbf9729a1791be5e7da0f3aacc9d5c20135a8 - Fix loading of FASTQ files containing RNA with U bases
* 9e5db84725635ceaa282691e8e430dd56851ffa2 - Fix duplicated alignment tags in re-aligned files
* 3cc4de3c941601fad906c80b6c770fef2814ad9c - Prevent "Too many open files" error when using `--sort-bam` with `dorado demux`
* b53191858fd33e7a0b4832df6e9e38cf5af22add - Prevent `dorado basecaller` crash when signal-space trimming removes all raw data
* adc60bae22648fef6521608a625c0e5bc842ac2f - Package `libcupti.so` into ARM Linux builds
* 667d16001845c8f173ae44ecdef0befaabf2af10 - Remove kit name requirement in custom barcode configuration
* e9281fa6d9ff36a7fb51efed0caf2c776b7d0c33 - Emit an error message if header from input HTS file cannot be read
* 7f42b8fd869da5210f9a60a83a6656173023dcb0 - Warn and exit instead of crashing if a model path does not exist
* 7d7424615830f46f7246a20125b73573c78ef7c9 - Improve index file error handling
* c77733a9d5ec054a426147cdcd9f6e8a03399aff - Add a mechanism to cache auto batch size calculations
* a674dadec1b3feb1ed8e8a6421d5d3fc17b0b5bd - Update `--help` documentation for `basecaller`, `duplex`, and `correct`
* 022901e29864fdeb9a99c4961c679882fe4a6b34 - Fix JSON output when using `--list-structured` with `dorado download`


# [0.7.3] (1 Aug 2024)

This release of Dorado updates `dorado correct` to fix handling of high copy repeats and avoid shutdown hanging. It also includes `dorado demux` improvements to reduce false matches in midstrand barcode detection and ensure correct file naming, along with other fixes.

* 5dc78ab677fbf4c67467ecfc4fb89438dc68c8d4 - Remove limit on number of overlaps considered during all-vs-all alignment in `dorado correct`
* 2741de70278dd47850c548266d8b3bff5b387aaf - Prevent hang during shutdown of `dorado correct` and prevent out of memory errors
* 37d316c4db26df6121f5c09410eb8bd966ab6b75 - Remove unused `--read-ids` and `--threads` parameters from `dorado correct`
* ddb13dea5144769e0a5542321aa03867f06024c1 - Increase the threshold for midstrand barcode detection to reduce false matches
* 845a3ad37f2a7d2b7e62fcd50ff1971f47c139c3 - Fix misnaming by `dorado demux` of barcode file for barcodes ending in a letter (e.g., `12a`)
* 56d3e8e2ea55b4873e0ef4f35c144d210b20755e - Fix seq/qual orientation when demultiplexing aligned BAMs
* 5ddfc2fa6d639fa52c735184a2c92e9ce4306a3c - Fix bug causing CUDA illegal memory access with v5 RNA SUP and mods


# [0.7.2] (18 June 2024)

This release of Dorado resolves basecalling failures when running v5 SUP models on CPU-only devices or v5 RNA HAC on Apple silicon. It also fixes bugs in `dorado demux` and `dorado correct`, and corrects `sm` and `sd` tags to match the Dorado SAM specification.

* 383527291e0553fa08d647af81a6c8a4bf4045a5 - Fix bug causing v5 SUP models to fail when running on CPU-only devices
* c36f4443d982a2ca47dc84fb8b840929811ea087 - Fix bug causing RNA v5 HAC basecalling to fail on Apple silicon
* 36218004593b0ebba37ff870eb3d92c8238fb0c6 - Fix bug causing segfault in `dorado demux`
* 3b51c1b3c694453d7da04ea91030d7e98b4e9681 - Fix sub-par alignments in `dorado correct`
* d0df79c49f29d2f7cdff3423071380be3b6c6918 - Correct shift and scale (`sm` and `sd`) SAM tags to match SAM specification

# [0.7.1] (3 June 2024)

This release of Dorado fixes out of memory errors when using the v5 SUP model with methylation calling, resolves several bugs in `dorado correct` and adds correct handling of the `BC:Z` tag when running `demux` multiple times.

* a9c6f59bff450d2a822ca5b64b18162bfd9b9a09 - Fetch available memory correctly for autobatch calculation with modbase models
* eb24124ba8f1a72c6730c1ce4b178f020c9d3565 - Move developer quickstart and extend installation instructions
* 45b8acc730ddbe6b438cf17153a64c084d1af2fb - Package missing CUDA Toolkit dependencies with `dorado`
* 33578e7c6d3af063389411ae6d7436ed6d0f94a1 - Update BC tag instead of adding a new one
* 580ad61ccd0f193c88202c852cbea38790c50700 - Prevent creation of CUDA stream when device is CPU
* 82078c5e0bc9545010f2405e92f93d8cff0c35db - Fix segfault with htslib pointer freeing in Windows

# [0.7.0] (21 May 2024)

This release of Dorado introduces new and more accurate v5 models for improved basecalling. It also adds a new subcommand, `dorado correct`, for single-read error correction to help Nanopore based *de novo* assemblies of haploid or diploid genomes. In addition, this release contains a slew of bug fixes, stability enhancements and updates to barcode classification.

## New feature highlights

1. DNA, RNA and duplex basecalling models with improved single read accuracy.
2. Support for `4mC_5mC` methylation calling in DNA and all-context `m6A` and `pseU` in RNA.
3. `dorado correct` subcommand for single-read error correction of haploid and diploid genomes (for assembly pipelines).
4. Poly(A) tail estimation for plasmids and transcripts with interrupted tails.
5. Support for `--junc-bed` minimap2 splice option.
6. Faster BAM indexing and sorting code.

## Changes to default behavior

1. Data type of mean Q-score tag (`qs`) updated to `float`.
2. Adapter trimming is enabled when poly(A) estimation is requested.

## All key changes

* 7a09ca3d1d1e469570a7df1e5819c39e9dd2325e - Add v5 basecalling models for DNA, RNA and duplex
* 159b73c7fea64d374b562af32abeaa382af54354 - Add new models for calling DNA and RNA base modifications (4mC_5mC, m6A, pseU)
* be8ac08652d5fe0b73c1126048b7fd96f29f3419 - Add `dorado correct` support for read error correction
* 67dc5bab58d74ee636e492619a6802db38059534 - Poly(A) estimation for plasmids and interrupted tails
* 381f6c3038fb69523ea591b1942d3293d7e9b9aa - Enable adapter trimming when poly(A) estimation is requested
* d6b0f68b3617f34a321db676b780e1a1183b6060 - Change data type of mean Q-score (`qs` tag) to float
* f938c415ddc9f458fe718af72c82001448d9c3c7 - List supported models in structured format
* 70ff95d84b316adb4701f7f43a19151e73b58b5b - Enable `dorado summary` to run on trimmed BAM files
* 6373792b686538758a16aacb063434c2b3260077 - Detect presence of midstrand barcodes to reduce false positive classifications
* 68d40da45da886384508173219a9fb677fc50cef - Add support for `--junc-bed` minimap2 splice option
* c443f75314708b7aed0aafa38fffdb8b2e76e9f2 - Output BAM instead of SAM from `dorado trim` command
* a3dce7ebe298ce3e17f3d61ad180b099700afb6a - Support `dorado demux` from input folders with mix of PG and SQ headers
* 08e2c7bb2538c2ba89203a68bbf153e6a6054535 - Speed up sorting and merging of BAM files
* b8de2d900d9aeb1c349931a216db7e05aa2ff2c4 - Set maximum memory sizes in minimap2
* b8de2d900d9aeb1c349931a216db7e05aa2ff2c4 - Calculate scaling for RNA on non-adapter signal only
* c88e9f753219f3c462c3678ddfad6b4561830f33 - Update CMake Minimum Version to 3.23

# [0.6.2] (9 May 2024)

This release of Dorado disables trimming of the rapid adapter during basecalling which was causing some RBK datasets to have a high unclassified rate during demux.

* a64492b69eb59c1d60d602fee1670085338450c4 - Fix bug with loading reverse aligned records in dorado demux and trim
* 6cc278f4d7759a7aaaa9a9b336d843b127b0d7ed - Disable rapid adapter trimming to prevent signal overtrimming in some RBK datasets

# [0.6.1] (23 April 2024)

This release of Dorado fixes bugs in `dorado aligner` related to using presets incorrectly and in `dorado demux` which were causing demultiplexed outputs to be malformed.

* 3e060db5a35ab09fecbeef9754cc545ba400edf1 - Skip stripping of SQ header lines in dorado demux --no-classify
* a2abf83852e895b1016c690769b59c06587684fe - Fix incorrect overriding of minimap2 options when minimap2 preset is specified
* 1cc207a166b1cafcbd012f5c70b5c817c788c7f3 - Fix bug causing unclassified records from `dorado demux` to be unreadable by samtools
* 298277150ad2522ca6c1928c4981782ce2893a5a - Fix issue with allocating memory on unused GPU during basecalling
* fa79f4a77fca737704d8a9e08d0495b9988f88ef - Fix reverse strand alignments when re-mapping a SAM/BAM file with `dorado aligner`
* 3b2c8252d1a40bb0f941ca2ceca0849be15d15fa - Propagate `sv` tag to split reads
* 11675a565da9af52de89a3f6614d15e57d10765d - Fix bug where errors were being swallowed in HtsFile class
* 73046e19fd443dfb48f3fbb82c0b37c5c7cfb8d5 - Fix typo in Warnings.cmake

# [0.6.0] (1 April 2024)

This release of Dorado improves performance for short read basecalling and RBK barcode classification rates, introduces sorted and indexed BAM generation in Dorado aligner and demux, and updates the minimap2 version and default mapping preset. It also adds GPU information to the output BAM or FASTQ and includes several other improvements and bug fixes.

## New feature highlights

1. `--emit-summary` option to generate summary files from `dorado demux` and `dorado aligner`.
2. Support for loading inputs from/saving outputs to a folder for`dorado demux` and `dorado aligner`
3. `--bed-file` option in `dorado aligner` to capture alignments hits in specific intervals of the reference. Hits per read stored in the `bh:i` tag.
4. `--sort-bam` option in `dorado demux` to output sorted reads when input is sorted and barcodes are not trimmed.

## Changes to default behavior

1. Default mapping preset for `dorado aligner` updated to `lr:hq`.
2. `dorado trim` and `dorado demux` now output unaligned records by default (i.e. all alignment information such as tags and headers removed).

## Backwards incompatible changes

1. New scoring parameters for barcode classification to support an updated classification algorithm. Older scoring config files will no longer be compatible.

## All key changes

* dc22d7f21215082796e972760db68cbaa3faf242 - Update method for barcode classification
* e65eaf4d4dee3d85d5eee007013c9a8815f4aaa2 - Improve basecalling speed on short reads
* f0b829d5eb5e1e56b96d6d8a7749445fa3d0f4d2 - Emit sorted, indexed BAM files from `dorado demux` and `dorado aligner`
* 913f062d6efac4e661bea0142c644331a01a5c29 - Add DS:gpu information to output FASTQ and SAM/BAM files
* c4598901d764f073a18232f02039a618d1d94d3c - Added support for `demux` and `aligner` reading from a folder and a `--recursive` option
* d994a4d5232226ddac68eac60a14de5db087cf5c - Add `--emit-summary` option to `dorado demux` and `dorado aligner`
* 246b9b995360a384e1b122a7ef477f43acc29843 - Add `--bed-file` argument to `dorado aligner`
* f6b65540d1edfc84f06fda3a751ccae8d96dcfba - Add `--sort-bam` option to `dorado demux`
* 9b49ae5e2be1df596c0aaed821c63dfb0add5a9c - Update to minimap2-2.27 and use `lr:hq` as default mapping preset
* a0f9462761df979540826d4725a2414a68db9503 - Add `RG` and `st` tags to FASTQ for consistency with BAM
* ae47155115e5f22475f424e8da3553312315cf53 - Calculate mean Q-score for RNA on bases after the poly(A)
* 3cf15fa02edd65d036cd830970719cb15d8c7e99 - Trimming rapid adapter from raw signal
* b40d0015618d3ef355445d35ea7a1ebeb2029101 - Improve read splitting for RBK
* 9d3af872b2aa4ddda4044c8a52113b765abd0701 - Trim low-quality data from reads with end reason mux_change or unblock_mux_change
* ec106d65eed2592683a8d30f936318721369b427 - Improve performance of calling modified bases on NVIDIA GPUs
* 77c55999499f72ff56fdfa792745dd0b1b0d00bd - Improve Apple silicon auto batch sizing
* b4fdb2453b8719f36eb2f99ce5814a110f7b300f - Fix bug with `MM/ML` tags not updating correctly with `dorado trim`
* bacd354210f74373946a0f4d248162bb22e88063 - Remove invalidated tags if running `dorado demux` or `dorado trim` on aligned BAM
* b6077db36f18947c27cfba1d2d3b95bca3a014a7 - Fix bug with modbase model auto detection on `@v0`
* ba0d70862f1b861b996244d83d7f547da9845d59 - Ensure `ts` set to zero if `--no-trim` or `--estimate-poly-a` enabled
* 12c5a3e23dff454f28a6c27ce24afbba62414e72 - Fix duplicate SQ lines in header of aligned BAM
* 9dc052dce52014aa7c2eaaf4c57eab290974194d - Ensure read group header lines include custom barcodes
* e8fb085897feeda7dd101b201f856d64e33b125c - Skip barcode trimming when running poly(A) estimation
* bbe6ad6964944158236a369f4e723c7e5f21330f - Handle issues related to user locale
* bdc05e37dd6eea975615d37f246c1529fa8144c7 - Fix bug using simplex-only model complex and `--modified-bases{-models}` arguments
* b31e5c88b71913ae9f6900fdd2938ae58e84ead6 - Fix resume loading for split reads
* 2919fe0b1ded1bce64ee66534e13d7fbf06ccb96 - Fix bug with custom barcode arrangements
* 98763daba089be12708479ade687c8146f2ca245 - Fix bug when aligner writing to stdout
* 74b4b536f621ec31d795f4559c9fdd205fd71fc7 - Fix regression with calling modified bases on macOS
* 392900332620a21da1b6e86f264c0459b163511c - Perform an allocation-less matmul when using torch
* 6f283a5f6876df767b09ee4c4ec4d6e268731b9d - Prevent CUDA OOM due to small allocations
* 0fa2c2f7f964942287548ecc3afc439a9c530b5c - Fix Cuda OOM during batch size calculation
* 7506d44333d980e2d381a4731608fe3137354975 - Add support for additional barcodes
* 13ba5af85aecfd7c651b249a127cf1b3243c1aec - Add deprecation warning for FAST5
* b5dc9f846bbc6f487a1f971824690c3a0f5fe3e4 - Update to Koi v0.4.5
* c9c5ad0ea395bd528c99680daa73208df4b09165 - Update to POD5 v0.2.4
* 901f700e00c76158a11aa4fa34425c107fa15a18 - Improve error reporting when the device string is invalid for CUDA devices
* e3442ec2633238ca6bfd073c9e817b6b6c14728c - Log errors reported by Metal and enable warnings
* e61cfe4ac28be6e06ea42721eeee2e7693527638 - Output Dorado commandline arguments in logs
* de59f33492f75ecb6777b372c9f0a0382a65a768 - Move default download path for third-party libraries into the build folder
* d7defcc424f566ff3a8de193262da0531cdd4fe5 - Log a warning message if running on Apple Silicon with less than 16GB RAM
* 8dfd180d526bf426624d2129bf1a495d1e1d43dc - Consolidate pipeline node input thread handling
* 401882331baf86d7a3e4b8e5a018389a8363cfbb - Update DEV.md to install the correct package

# [0.5.3] (06 Feb 2024)

This release of Dorado fixes a bug causing low Poly(A) estimation rates in RNA.

* 59a083c9d9ee359049fc30ea4ca9ec4aacb84fe7 - Fix RNA Poly(A) tail estimated in the absence of adapter trimming.
* f0f98838156fe45c51a3e39795ee45248e045100 - Clarify `ns` tag in Dorado SAM spec.

# [0.5.2] (18 Jan 2024)

This release of Dorado fixes a bug causing malformed CIGAR strings, prevents crashing when calling modifications with duplex, and improves adapter and primer trimming support.

* 062e5e32a23f7f88705c3e5ef989f3f8c524b340 - Fix malformed CIGAR string for non-primary alignment
* 0a057bb14c6342092eaf5787c5de90f41c08a93b - Fix duplex modifications crash
* d453db274d68f5f315deb1f1238814ec24953588 - Add missing support for RAD adapter detection and trimming
* 8c2d004d71c9c21fb7bfbe283ba44bc100a67793 - Correctly trim modbase tags for reverse strand alignments
* 76f24b29312af4a6bb22e02e79f439bd1ccfd725 - Update custom barcode documentation
* 9959654ba7377d3807d5d96aabdc9b40d74c5d0d - Only require standardisation parameters if standardisation is active

# [0.5.1] (21 Dec 2023)

This release of Dorado fixes bugs with adapter trimming and custom barcodes, introduces a more accurate 6mA model, and adds several quality of life improvements.

* 9a4639261576bf957392b406f7e36664b1ddf9eb - Replace use of constant with a parameter from custom barcode file.
* 1893d69a4f472df00936db3fe7c31a72de31c23a - Decouple basecall library from models library.
* e42761c65b96e4feecdbdaff343928e30e258ee8 - Allow RNA adapter trimming to be skipped.
* a510d531af46dc07c445d59914e809b4a18de0ad - Prevent simultaneous usage of multiple modbase models affecting the same canonical base.
* 371a252ac5d62512597d288bb3396cc93a2b18fc - Fix incorrect sample count in the ns tag with sequence trimming.
* 9f532ffe298c72f39984fc68e36e9fa24d811382 - Remove modbase tags for non-primary alignments except when soft clipping is enabled.
* 52431e685dae82fd94ecf12c52f5d3cd8f78fd1a - Update 6mA model.
* 7109c1c639f19b8bea5ea4e7faacb6cf32e58fb6 - Remove superfluous clamp from Metal model implementation.
* 5fa4de73a38e5ff858e9ed7662d8b72844588c6d - Refactor decoder interfaces.
* a3dfc94a1bf37644d95ed2acabd7bb27e5229245 - Improve README for adapter trimming.
* 3bfb1f052bdf5b09998a511cb6369c2227d8bc3a - Fix bug with out-of-order primer trimming positions.
* b1302aee6accd8e9a56037311c176649f8b166bd - Allow alignment to be skipped for disconnected clients.
* 55d09f9f985923494b8ef407f35e788cfd6039d4 - Update HDF5 pre-built library location.
* 2048ad5f371d50ec03268d4a69c566b73a661bf0 - Decrease httplib connection timeout.
* aae47b1313511c330765c418cff55e2cc7f51323 - Refactor codebase to unify interfaces and reduce dependencies.
* 6ed81c5edcf0d6e0d20da4d250891e0a7e77b92a - Run separate modbase models in different CUDA streams.
* decb9e7ca8688e365f5f340802534dedaf7bf026 - Update build settings to simplify integration into basecall server.
* e8b07e20925dcfb79538d7cdcc40a9e064560bbc - Report warning and skip FAST5 files when datasets contain FAST5 and POD5 files.
* 6c984a0b04617866251920d11092618e24cc5f43 - Enable Xcode builds.
* 6d317935ffb4974cab4bc2a1617eeddaf47976a3 - Split Metal LSTM kernel into multiple command buffers.
* 364d15d9c4fbc6d88a7cc8f35657da5d31c51aaa - Fix bug with passing custom barcode file into basecaller command.
* 951e3c3f4cd645580b8f09ee7d24f94f99043910 - Allow read to override adapter/primer trimming defaults.
* d6e2a801608e1b3ae0baaee346c1690f63f5032f - Clean up model auto download directories.
* c5523515d66b57a3aa862cf5b1296e6082942e07 - Improve error handling during model auto download.
* 936d40883b1c4fcd6d8dccbc90d62a5d4d444b81 - Report incorrect results warning for CPU basecalling on TX2.

# [0.5.0] (5 Dec 2023)

This release of Dorado introduces new, more accurate, and faster v4.3 basecalling models. It also enables hemi-methylation basecalling of duplex reads. Dorado now supports DNA primer and adapter trimming, custom barcode arrangements and sequences, and can automatically select the correct model for your data. Furthermore, this release introduces speed and memory enhancements for basecalling on Apple silicon and various stability improvements.

* 14159695955dd0d08322f26b545069fbfecb5003 - Add v4.3 basecalling models
* b7d4b380f17d4a15ed43d8d383cc770d121fca17 - Support for modified bases with duplex basecalling (hemi-methylation)
* 30e639cf66c1c24d0f61f1e7b91c6ce5db2cf7bf - Primer and adapter trimming
* fb85a70609eedfe895587275d06429515a1ce61e - Enable automatic model selection
* 16e5b6ad577f5485eb3a78c755313fc8314b2b1c - Support for custom barcode arrangements and sequences
* 46bbfddda06a7088f7031ef79eecf03b0f04660c - Add barcode column to summary file
* e9f060c1afff8d72fd51da4201d3062d8c8a2064 - Improve the precision of read splitting
* 4102ffc3454c609479665a337e1ad7c2f33b9d22 - Increase speed of v4.3 model execution
* 0a0711012ad906f94aa6e26c3a6b540e5ccbcc0e - Prevent progress bar from `--resume-from` logging excessive dots
* 20b5637dbbf944efcc3878c5271a8bd84d2b6eab - Ensure that aligner outputs SAM when not piped to a file
* 942a35a69832883904a1116b9b21d5c1641d0e2b - Add `MN` tag to ouput BAM to help downstream tools interpret modified base tags
* f0ac935035423d3b913940bf1b9b7fd50d832993 - Added modbase model name to BAM files in RG header section.
* a7fa37132b0f442ce87a154e7f2db21dfaa66933 - Improve performance of HAC and SUP on Apple silicon
* 152d5fdc782d14b1e9853d9242051d1f7064b63c - Improvements to auto batch sizing on Apple silicon
* b0767a6f31cd7f084491b2b3313d33d048bcc5a0 - Fix bug causing segfault with `summary` command on Windows
* 1c2c6a9e9bcf980702afec9b9f6a17cd27c3ae07 - Make AVX `reverse_complement` implementation preserve nucleotide case
* 4a4dd1cffe9db32e4c58e79ca6dc5dc79125f0c9 - Use updated Koi functions for small LSTM layers, final convolutional layer in LSTM models, and final linear layer


# [0.4.3] (14 Nov 2023)

This release of Dorado introduces a new RNA m6A modified base model and initial support for poly(A)/poly(T) tail length estimation. It also introduces duplex performance enhancements and bug fixes to improve the stability of Dorado.

* 803e3a7ce2590b1c95b4754117185983ac2ad560 - Add RNA m6A DRACH-context model
* 0f282cde507a36bf91863270bd0323564235c15b - Add poly(A)/poly(T) tail length estimation support for RNA and cDNA
* 54e14ca01e7391c8857989da7db086a4591375a1 - Add RNA read splitting
* 2dc1f039cac7f3e6cd082b77a5b020fed5488e2f - Enable RNA adapter trimming
* 80114c08c45bc902843a2e18b5949ebf5cfefdf2 - Correctly update CIGAR and POS entries when trimming barcodes
* 4b2025c57fd3b87b2ce6cd52be07adfd9ae5acf9 - Add documentation for sample sheet support
* 641cb08b457d727c3da682185c6fe491df49dab2 - Reduce host memory footprint for duplex basecalling
* 7c1c0f04d93113d4dd2c632bdcd242304b54d270 - Reduce working reads size, in particular for duplex.
* 831f0a91f0100c2586720f6026450fdbae1a8d21 - Fix pairing check for split reads in duplex basecalling
* b63056743be6e5442f2f5af65a36c592bbf96184 - Account for split reads during progress tracking
* 383fe0226bfa7956705376ac5e4a32096ff80c45 - Update to Koi v0.4.1
* 873c6b11e0113735b21305afce5057138558388d - Fix warnings about `ONLY_C_LOCAL` mismatches in PCH builds
* 52cbabff83de3c9fb6f1a0db9194828b92418855 - Encapsulate `date` dependency
* 8fb8a4df567ba22df6a298f4e30277a0d47ceaa4 - Disable Cutlass LSTM codepath for 128-wide LSTM layers because this kernel does not work
* 6a9dad907af8dd2b4e556d49a329a8a0fbc5c32c - Enable warnings as errors at build time
* 5aaef312027836ffbd6e2b944e6cd3ba4a259267 - Address auto batchsize issues on unified memory Linux systems
* 92b5a6792fca4d2bb2b76727ec486efe8bdfae97 - Reduce compilation times
* 062e3fd53f58380070efff660303b71c03cd02c0 - Minor speed improvements to CPU beam search


# [0.4.2] (30 Oct 2023)

This release of Dorado fixes a bug with the CpG-context 5mC/5hmC model calling all contexts and adds beta support for using a barcode alias from a sample sheet.

* 90a4d013938672af6dac399afbbafbd8bf7814ad - Fix motif for `5mCG_5hmCG` compatible with `dna_r10.4.1_e8.2_400bps_sup@v4.2.0`
* 616b951e3ede257ae6359b6005f60e44a53dd2cf - Beta support for sample sheet aliasing

# [0.4.1] (12 Oct 2023)

This release of Dorado fixes a bug with duplex tags and adds functionality to dorado demux.

* 7fefd5d949d32decc3b347778afa80c90b9c6417 - Fix missing `dx:i:-1` tag for simplex reads with duplex offsprings
* d532ef1517413b536714d47c01a998305890d922 - Enable dorado debug build in Visual Studio 22
* eeef7575b6dabab8d5647cebed0f92ef793786ef - Add `dorado demux` option to demux pre-classified barcoded data

# [0.4.0] (9 Oct 2023)

This release of Dorado introduces barcode demultiplexing, barcode trimming, simplex read splitting, and updated models for calling 6mA and 5mC/5hmC modified bases. Furthermore, it contains improvements to duplex pairing for increased yield, basecalling speed enhancements and reductions to memory consumption, bug fixes, and additional tests for enhanced stability.

* e836fa46a452e0cff69f1ea0b2017878ff5b85b2 - New all-context and updated CpG-context 5mC/5hmC models
* e4aca7696392fe5836da22d1df62c399c2bfd664 - Handle custom batch sizes that exceed maximum safe batch size
* ad463ead384119b756d9f8a940919b1ab9cfe4fc - Updated all-context 6mA model
* 21d25a3d00331027ba24227e95d9b5f353a882d2 - SSL host verification enabled and `dorado download` HTTPS proxy support improved
* 8ae95e732310400016f34b8f9499f1fdb4a0dd41 - Enable simplex read splitting
* 1210607cc0f647772053ed82dcae94b416a0b821 - Fix mean-qscore calculation with trimming
* d27666eb02add2e9a991442d8ba4f2bc7f849df5 - Beam search optimisation
* 46e68c1138ebc25e53818e185e967f5a156c1896 - Report the error we get back from httplib if a download fails
* 89db1e197c68d6cf7209f1366372cb24dfe87ba2 - Support ChEBI codes for modified bases
* 92097cade8f53effbc1610c2fcac679c19203cc6 - Add options to `dorado aligner`
* 4cb986265ffcc0576752e20b5c20acb01ac635e7 - Fix Linux ccache usage
* 692ecd3ed9dc16a94a86bf4ac563c0e618745bb3 - Fixed issues with internal representation of signal during duplex calling
* 80efe8cc7116883ee5413efb223bc5189f893322 - Fix `std::logic_error` (issue #205)
* d92547a49fb95210656fcef7e9486f228ab4cfeb - Make CUDA kernel profiling to stderr available via `--devopts`
* 69bf701f2668c52d325375706821a00b72ea9b1c - Refactoring to support further duplex developments
* 510e2e9e0b2e24d9b5c8bd1c13057c4b05d145a4 - Verify models when downloaded using `dorado download`
* 12476ee058271b353ef1e2cfd4a159397ce9c824 - Improve memory usage diagnostics in Metal
* 3172413657b9be0bf0b6c280cb8c03c7627f9784 - Fix non-determinism in selecting matrix multiplication sub-routine during basecalling
* f14b418e8497fcb033d68f3e6561b5b9a91a07fb - Provide NVIDIA driver version in server API
* 207871ee7f98d1e0cbefeb790ffdd02c1f6c929c - Use `DORADO_GPU_BUILD` rather than `!defined(__x86_64__)`
* d16ccbea8d99c1d820d508ee3ce9175374293049 - Get tests that use CUDA working when ASAN is enabled
* 41bfb99503372db84d218d5ccc0b8f2462d1bba9 - Update duplex pairing heuristics to improve duplex yield
* 87c2c6ed947c7e36bd6bffada2dc2f1856f847a6 - Change `ReadPair` to take full ownership of its data, and drop `ReadPtr`
* 88aa9f3f1ede6803b531bf6f9a719a2cc4f6f74c - Only use pre-compiled headers if ccache isn't available
* c04c145afde62f0f7c1c739b69e5b6f98758dda5 - Bump Koi version to 0.3.9
* 6b8064df59bce3de35b951480b56584484f6c317 - Separate out data shared by both Duplex and Simplex reads into `ReadCommon` class
* 34e9b550b1dcbd40bd8df1ba726773881627ff90 - Flip RNA signal for modified basecalling
* 636ac83e642bda153d6f44d0de979bdd37ccee98 - `CRFModel`: Update to use `KoiActivation` and corresponding Koi host functions
* cd50a01a0b555a75f208bc827e6aaf34d4f00163 - Add parent ID tag (`pi:Z:parentid`) to split reads so that original read can be inferred
* 2956bf6afef66e86b7abc0122ae19a2e37c96cc8 - Add support for barcode trimming
* 92dee851d39ce48fb12b221ac0bfa7575ae7e182 - Fix segfault in modified basecalling
* fe22d2158e4c5fb95b2e2875f3e10d3482035ed1 - Add support for read groups with barcoding
* 2f69da6f659442e0b3caebbb94dfc9e84432e6d3 - Allow basecalling from single files rather than a directory
* c22e46f5407c0863180b0e273551fc9fb5ebc145 - Support ambiguous motifs for modified basecalling
* 882da60f3d2cd9a9633966fe8e5e8241310ffd87 - Refactor to create separate utils lib target
* b792bba1b07986d8ec4dcd601ded55d7021eb4d0 - Enforce const-west style via clang-format
* 611a4eaf2ae52041216c4c369d9f61d481275df1 - Skip code signing on x64 to prevent crashes
* a6a1902efa462d25637be056c352502a2c262366 - Fix various deficiencies of the iOS htslib build
* 5ff8034a0f9f6555223e2d12e53db8061ce718ef - Add barcode check for both ends
* e42b8c8da47c97d7a9fd0844f9d87531e4f35afa - Remove mux from pairing cache key to reduce memory consumption by up to 4x
* 93d052bce610797bf5fae169471635c32bd5bb8d - Setup signing of executables on macOS
* a45b97ce2ded09baac3c06618e5d5b8f8ce9ad8e - Bump RNA004 models to V3.0.1 with corrected scaling
* 7e152fd556401d2af3bc14423c048fadd02f744e - Reinstate `ReadFilterNode` tests
* c3e412ea3a134f18ef4c5eb07e5b6ae46940260c - Add duplex commandline test
* 3ba422b1678a61510900ef647eb501d1a18e736f - Don't perform adapter trimming on RNA signals
* 066b815c4dd0e75a0629d5432199ab6a830637f5 - Add barcoding support to Dorado
* 09cc44f12ff32e6606b192229e6a2e8fb14c892d - Add unit test coverage report generation
* b2e54b8bb6348dc418b45722b96fe001a36a0497 - Fix typo for 5fC modification
* 829dba39bf85250e8275456c3e87a2d6eec72f8e - Enforce that values passed to a sink are mutable rvalues
* 8aa7722fea91abb780071dda7414d0a5bc20cf0a - Remove dependence of `dorado_lib` on `dorado_models_lib`
* cd6d2bfbf547a555c5a7e649db700807ef6d17ea - Extend the lifetime of the `NNTasks` in the metal backend
* abed8ee88e9648981a0996b7f9c0507eca018fb0 - Bug fixes for iOS build
* 64b3ae33377972474a3ec2c4752c95d274f1c908 - Clarify `--min-qscore` option in help
* 225a153e9defb873925e4a0e2e414a1fa1a5aeed - Further restrict Metal kernel run times to improve stability of basecalling on Apple silicon
* 3ae95e40d61b8f383bd778a779aa4a54fe7abdcc - Refactor modified basecalling code
* 8ec58f0153b39f9e60d0606da0a4cc24c6d83e59 - Option for `--guard-gpus` no longer used in duplex
* 1f3cade20f6074880997833783ae1be645974557 - Add `CUDAGuard` before cache clear to reduce CUDA memory consumption
* dae5e30531b3687145e15cfda7655cca267b0aa5 - Move `ModBaseCallerNode` to using an `unordered_set` for working reads, like `BasecallerNode`
* b43adfab9571ab14ac8e41e9d5102fe5fa98f541 - Improvements to pipeline API

# [0.3.4] (14 Aug 2023)

This release of Dorado contains a few bug fixes and a hotfix for CUDA out of memory issues encountered during duplex runs with v0.3.3.

 * c5c0ea003d2b2019f8d400ef203f45ef28aed3d3 - Introduce pipeline API functions to simplify setup for simplex/duplex basecalling.
 * 9614ebaccbed591a82be2b3683a406023f4fab32 - Fix potential hang in modbase calling node.
 * 67f84a69b6ae3ed12729c9723d7c827daaeaa3a7 - Set the `--max-reads` default to unlimited for `dorado aligner`
 * f6a0422b34cdd89637bddb939364a38ea507d198 - Fix CUDA OOM in duplex by removing tensor caching in decode and updating memory fraction for stereo model.
 * 107ebba28c9f77424c0ba271b663338dcd49a902 - Account for filtered duplex reads when tagging simplex parents and calculating duplex yield.

# [0.3.3] (8 Aug 2023)

This release of Dorado introduces improvements to Duplex pair identification for improved duplex yields, faster basecalling on A100 and H100, improvements to modified base calling speed on Apple silicon, and major enhancements to the portability of Dorado binaries.

 * 7307146b55096ee339b1da953b5acf05d112e124 - Major reduction to required GPU memory, especially for A100/H100. Allows greater batch size and consequently improved basecalling speed
 * 8073364fa8e4390d21da56216a2ef0d78b57f206 - Improvements to Duplex pairing algorithm for increased Duplex yield. Situations where complement read is truncated are now handled.
 * 39ffb35a90a2389b9f45ba229e908136e770991f - Report the duplex rate percentage
 * 65b8b8a81a202d0166ca2d0ea4724941aa3102a4 - Major speed upgrade to modified base calling on Apple silicon devices
 * 481438a3415e009174e5f0406d90ad9fd0868339 - Improve performance of basecalling of Fast model on M silicon by 6% by inlining of function used during decode
 * fe8dbf2af44900bcb60e602653cf01dc119b702b - Improve basecalling stability on Apple silicon by limiting run time of LSTM kernel to avoid CB submission errors
 * 0abea5fb58545a679fcc1638ababe2a866789921 - Upgrade dna_r9.4.1_e8 5mCG models to version 0.1 for improved accuracy
 * 752e094b5c4769ff4375249a23ab51d51cf9486b - Upgrade to OpenSSL3
 * 447b5592758ddf1d26d6ae72a43e836259b53836 - Switch to target-based includes CMake builds - Prevents dependent projects from copying all nclude paths and link libs
 * b0d10a985e3f4e8d43778367a1710c67d1cde544 - Fix edge case where it is possible to spawn no basecaller worker threads
 * 1650b83106c3a06e038b98badc3c529e12046630 - Fix issue with inability to find `CUPTI` library during compilation on some systems
 * e4ba3e55fe3e8cd83d59a2f89a415a63be672b94 - Add missing `SA:Z` tag to alignments generated by Dorado
 * c5a4cfcabb93084c6d5a4d0e069daefc5cc261b1 - Fix various linker errors
 * bf72fdd7f2f21491f85930f55f7bf0f9231eac4e - Dorado will error out gracefully if no POD5s are provided
 * bcdeb8f0093bc5c01ef42befecda7f524b894aee - Improvements to portability of Dorado binaries via use of static linking on macOS and Linux
 * e14a7e61a6a3e60c0516342df7516b0ba5b92e62 - Improvements to error handling on Apple Silicon devices (Metal command buffer error handling)
 * ca1d191ffba5cc1397e4b704e3dab1014fa07175 - Improvements to read ids for non-split reads
 * 5d9238a809656d6fab59713c7ab8346790efd2cc - Revamp `AsyncQueue` interface
 * 6a3ccb6af5a56527571bf4938447bf85ab5c38e5 - Removed RNA003 model which is obsolesced by RNA004
 * 1a94facbad52db0908eae3a7fe3242d525cbd6da - Add summary stats and progress bar for basecalling from Fast5
 * 072ed964276fc8e26f0a705c89055cece900edbf - Add pipeline restart capability and pairing cache retention option to termination
 * 01acbd5628948bf934985777bcade6bf51103200 - Stop progress bar cutting out near the end of the run
 * 211968cd38fee6fa7476a4033a2b59ca37b19199 - Add suppression for false-positive vptr issues in older Xcodes
 * 1e14d2af7bff07c62b0286f17e1d61a7b99390fb - Reduce unnecessary startup cost by setting an upper limit on the maximum batchsize to 10,240
 * 09c5b28f51707c72df1436967d5da28f850beef1 - Speed up the fixed cost of auto batchsize detection on mGPU systems by running in parallel

# [0.3.2] (13 Jul 2023)

This release of Dorado introduces basecalling models for the RNA004 chemistry, better identification of duplex read pairs and improved read Q score estimation. It also incorporates various important bug fixes which improve the stability and usability of Dorado.

 * 4ce0cb038eebbaa3de6d561caa11f0ee80d2e82a - Add RNA004 models
 * 24b6c4e2854c32aaca37b039c7bdf5a773db8995 - Retry basecalling on CUDA OOM after clearing allocator cache
 * 3897ba505bdda80ff1f0fe9f517cb463bc5347bf - Add troubleshooting guide to README.md
 * 9d55b446bce88e22de3584ad6c106cdfa5f7abb4 - Fix bug with resume file header parsing
 * f9289e2f5c93641bfc2d2e753e5d612c87f7b34e - Improvements to duplex read splitting algorithm for improved accuracy
 * 2869dbc52326d148f9562686d8d89a6dc373072a - Solve memory leak during modified base calling
 * 4d8ca172128cb4854e39599077d7c477eeb02504 - Fix race condition which was introducing nondeterministic basecalls
 * ddb6f711bf22c3b1f14690a85e6980e4e39a6c9f - Fix aligner regressions from pipeline change
 * a57987f2fef3cdb17f1a92e76dbf3110e9adf5bd - Add R941 v3.3 5mCG 5hmCG models
 * ba40e53e697bfcfd33beeaa44583a8cc56ca8976 - Refactor of basecalling pipeline management strategy
 * 26be1a017baf7131a82271ebb624a484135d29ba - Query enabled Apple silicon cores, not all cores
 * 185058ef226d508e2ead05659149514530e90adb - Replace empty value with "Unknown" in read group tags to satisfy SAM specification
 * d953f339e1ad32533bfb3d53c7427bbbabea4d1d - Add time ordered reads pair cache strategy to PairingNode to support greater variety of ways to run duplex basecalling
 * d2700dd44a0a5fbdbf6cbbc71d1f1a791ba25aa9 - Fix to enable Duplex basecalling on CPU. This will be slow but functionally correct.
 * f5ccd0d261ac384e05107290e214b3a422eacb26 - Add channel/mux/start_time/start_time_ms to the duplex read
 * 0ee5a9b5a24022fdb5330b512e14213c4cbe0e24 - Fixes to host OOM issues
 * 2fe609bcbec5158b3ba1f1a1c200212cf1a66a0c - Exclude some non-informative bases at read start from mean qscore calculation
 * 3090328a5b12689b649537a4dc83a25c00730c59 - Fix no output when stdout == stderr and both are the tty
 * 7e70de7197ffc09586fe6f097f9b25c30dc17802 - Add support for compute 6.1 (GTX 1080 Ti) plus handling CUDA failures
 * 995d0fb60c2ce3209e45c3645d59012326ce515d - Runner creation refactor
 * 9d21036b309e4dde1125ccddae382a5f1729f725 - Modbase smoke test BLAS fix
 * 30658c4d7b0823bd8924257aac5a09d3f1f96b22 - Add `ScopedAutoReleasePool` to prevent autorelease leaks on MacOS
 * cf502e3c67809b263e9d7645608e67643f90769e - Update sample rate check to allow some tolerance

# [0.3.1] (26 Jun 2023)

This is a minor release of Dorado. It introduces various bug fixes, as well as performance and usability enhancements. Of particular note, this release introduces the ability to resume simplex basecalling if interrupted, adds RNA002 models, improves the speed of modified base calling and duplex calling, and solves an issue whereby Dorado users were experiencing segmentation faults on version of Linux with older glibc.

 * 08218cda24dae7af12115bc749e038d89b41ec5a - Added `--resume-from` for simplex basecalling.
 * 790a002be6e69de390ca9d4259e126a02c0beafa - Mitigate simplex scaling performance regression on mGPU systems.
 * e0c1beb96daf88a932146bee878b8c7195d5e265 - Turn off all logging if stdout and stderr point to the same file to avoid curruption
 * ae1e5e39190860d5eadac1191c4baa7114ca259d - Support for http proxies in `dorado download` via `dorado_proxy` and `dorado_proxy_port` envvars.
 * a62465cd7fd555cc64a5592b6b05cffeff3a03f0 - Added warning if user tries to use duplex with fast model
 * 99f7483c22aa292682b9cd3a48f3795b99451a57 - Improved error reporting on OSX
 * 4f61c18498ffa846ff11b3b8ba3ef1ecdfb4aae0 - Removed source of error related to race-condition
 * b1405ab50c831281c36647fcd07acee40380b228 - Added new RNA002 models with V3 architecture
 * e75e327f2a52a4d77afbc9b2384b662725ea42b0 - Aligner throws error when reference path does not exist
 * a9779164e2b29793617a03128017b75433154d1e - Refactor of progress bar and account for filtered reads
 * f1627051edba1bff67a8b8fba510bc456b1e6621 - Fix the build when compiling with VS2017
 * 84ecf32158fb4a587b7e0263021eea19c5533d1f - Update Koi to v0.3.3
 * d6463b351264dacecf100aeb89c8b493966649fd - Refactor aligner/hts nodes
 * c46a2a3558f42831e85efce56765b37be12040b9 - Fix segfault on Ubuntu 16.04
 * 2a96f891e35fb567e696c3fe91cdba708c005d8a - Improve performance of mod base calling by changing encoding format
 * 612ba2043739ee0418d3addb643bfd145f04d58e - Node performance stats monitoring
 * dc9ea3b11c1df8521285fd075b562ec7e8d83a5e - CRFModel: perform clamp on torch tensors in-place. Use ``torch::InferenceMode``
 * bf183144f2b9caceea64a690f6946b9d1683c5a2 - Solve various bugs in duplex pairing algorithm
 * 164ca6ea3e2ca7491c8cf82162ecb19d540642ab - Add support for running sanitizers as part of CI
 * 379fc21b16834eeee88a335a1da8bd34ac211be0 - Add RG tag to duplex pipeline
 * 68b5b81aaacea8f9f48402147a6421ddf74f0052 - Add missing dependencies for Mac
 * 628722d0e8f85f334cbe4b8662423e4954f1fd21 - Add error checking to pod5 api calls
 * c48ab081a681025c34a462bee2816444ff44bb92 - Progress bar fixes for Windows
 * ce68b552c98cf5e2ee11331e4a6a7d7157507baf - Update metal-cpp so we can make use of ``NS::SharedPtr<>``
 * 53aec516b9a017880e47ba81deeb9084b092b311 - Update Readme to include roadmap
 * 5ea670504c052426ef9cc8e2b51f3b213be1b58a - Adding smoke tests

# [0.3.0] (18 May 2023)

This is a major release of Dorado which introduces: Duplex pairing and splitting for directly going from POD5 to duplex reads, major performance improvements to simplex and duplex basecalling on A100 GPUs via int8 model quantization and the output of aligned BAM from Dorado and support for producing summary tsv files from BAM.

 * ddb7c1e20b8df5935764cf1d014d3f2202eb29a4 - Improvements to modified basecalling performance
 * f879af586dd122a5e9f071fa28970e1531ba1530 - Add support for CPU basecalling of modified bases
 * 282a66c4730b381694c170385db3fd124a4c2048 - Add duplex pair alignment accuracy check to reduce risk of incorrect matches
 * 3bb0ffc3cc44ca2319b02b6e91b7a0a0fc0a8ca3 - Add `dx:i` tag to Dorado output to indicate whether a read is duplex `dx:i:1` or simplex `dx:i:0`
 * 78d6bc403ff840b917ffe89c0bdfc24f779df224 - Improvements to Duplex calling performance
 * 20972d4638f8c2d6a3dfbbb3350590cb5c006e7e - Added ability to filter reads by read length, default of 5
 * b8ceee484a112f9982d581f15fc143f1a2e1404e - Include simplex output in SAM when running duplex basecalling
 * 724bafd32cea0603771c6bb07bd24e7201669a31 - Add `dorado summary` command which produces a summary.tsv file from a SAM/BAM file
 * da13d36cd4aa25aa5d2c76c9a8f1da8a6491a783 - Add splitting of live splitting of concatemer reads into duplex pairs
 * 8992e6731cf416ef1c29a5eb09b6ecb11d5993ff - Fix for segfault in older glibc version
 * d1377a0f838b2823ee906795ae4e19474b5d8655 - Add `dna_r9.4.1_e8_sup@v3.6` simplex model
 * 88b547ce61adedcc73bbd19428707c7bc81bde31 - Add 5kHz duplex Stereo model and duplex 5kHz support
 * 26609569c401c19bc2b31135f7c99f09522ee731 - Various CPU performance improvements
 * f4ea66453971fedb7eaf9f8c82e539c0fefa014b - All context modbase tags
 * 338911db23fdeb96345e4aab34184c35283d18c6 - Add v4.2 6mA 5mC modbase models
 * 1684168809f9126611d752624fb47339e21f7b31 - Improved support for short read duplex basecalling
 * 6fe6adb381fde06778f8f56ac8d8f76adae3de99 - Add verbose logging option for duplex basecalling.
 * a035d7f12f60753a7f027d3e881bfeae1843dd9f - Check model sample rate agrees with raw data, add option to skip check
 * 26c11122478d8593d2d5f34fe57d8b0744aec750 - Add ability to perform automatic duplex pairing in dorado.
 * a824a7d344051beb7ee0e14cf1cb829143365f24 - Output uncompressed BAM when dorado output is a pipe.
 * f27d672f32a7d567cdf116314b0edcb64086403e - Add CPU to list of devices in help
 * 3329bb5cb1e850536d38c570770fe89544aa9f8b - Fix Fast5 basecalling
 * 4d91533610ca908dd4daf61a81e3a3fe634ace89 - Improvements to reduce possibility of out of memory issues on CUDA devices via a GPU device mutex
 * 14de2e8e05bc737948a7ec2f82fe14b8a8d7b7b7 - Improvements to progress bar reporting
 * 2095fea10169121d52051a50d48e37648b31defd - Add alignment and BAM generation to duplex and simplex
 * fef15ae0a74155460f175e55d2adf0421201cb48 - Improvements to stereo duplex encoding.
 * c6dc18796c7f77d7276972798af1606d7b76ceff - Add `--reference` option to basecalling to allow basecalls to be aligned to a reference.
 * 51ca9e7c1968827e07c8720b872acdab66adb9d9 - Add v4.2.0 5kHz simplex models
 * 1a215e793146a1c7625e55c95d0e57c4bbfd7b12 - Reduce CPU load from mean_q_score_from_qstring
 * d3f7320469822ae69b76afbbd58156ef82f44ea8 - Handle empty read-ids file
 * 25e2cd103561df1feb55c9bea6809d966e480e94 - Upgrade to Pod5 v0.1.20
 * 98eb30d3a23c0e40121d030cb73be38a045dbfa2 - Add Cutlass LSTM kernels for significant performance improvement on A100 GPUs
 * 6aea63a9bd747de727ac909f8ecb7c7901d1d5a6 - Reduce CPU load due to trimming
 * 7459371e837a0be3ab4e502d5efcb071e37ce543 - Increase per-device ScalerNode thread count
 * 758d0d9e17196327fbdcce53a0b20c2e163cec3d - Minor improvements to RemoraEncoder::encode_kmer
 * b2af21b40b511cd77b90b171c78a7433026310ff - Add read filter node to filter reads by Q score, length etc.
 * fb604256639fd6513ce501fa8eb432ac4badd522 - Reduce stereo duplex CPU load
 * 0bca7d84ee23cb8827499e6c7578a9974df42a2c - Reduce torch indexing overhead in modified basecalling
 * 4632f05a194a081e076bb36031480a325b710da7 - Expose `k`, `w` comandmline options to dorado aligner
 * d560661ad52d51dca36d8d946b9d80aa0b848039 - Improved read trimming
 * 3cd1c80faf45b8df44180b6bd9e8bf3294924df2 - Improve performance of reverse_complement calculation
 * 92ef398874e9f4d09c7a57e5d979d4e704a12a74 - Fix segfault in modified basecalling

# [0.2.4] (12 Apr 2023)

 * 92ef398874e9f4d09c7a57e5d979d4e704a12a74 - Fix out of bounds access when modbase calling

# [0.2.3] (06 Apr 2023)

 * 6bf227b8139132854927884b59246922f92f0bdc - Upgrade to Pod5 v0.1.16

# [0.2.2] (04 Apr 2023)

 * e1159c4aa8562f539f86d0d2d30899cb69ec9e54 - Add V4.1 modified base models
 * dd389f1239ab5372eadc9305064053a7ac8941ac - Add ability to load POD5/FAST5 files recursively, remove the `runners` flag
 * 5a55416a9e414bb325d10c07e68904eb3816eeaa - Fix bug whereby the last read was ignored for read lists not terminated by a newline character
 * 1c7988a3c47c34e4b87264e988489feb1a1af752 - Performance improvements to Stereo Basecalling with better batch size selection
 * 27c8a2efbcbabc689eda0817e6a572cb9f2887c6 - Upgrade to Pod5 v0.1.13 which allows loading large files on vmem-limited systems
 * 3a9bb7a4bed6490f7c7a913200c9534a8238329f - Incorporate improvements in Koi which give more predictive QV Scores
 * 6896f096f1cfba67d3a831df6ff911bcff38a9ed - Fixes to dorado duplex basespace
 * 1079b75303ddf642a6569141c7cdcbfd808e6826 - Upgrade to Torch 2.0 and Cuda 11.8
 * bcfd64d2552aae5fc498c6722bed8fc3c2b87961 - If a Stereo model is not available for duplex basecalling, download it automatically
 * 8b9064359140b882530a3b750b9b235019d56fad - Improve Dorado startup time when using POD5 files by getting ReadGroup information from metadata tables
 * e86e9707a7a60dd030e2338e77c2562939cd6fdc - Use `jemalloc` to override allocator on Linux
 * 4d6a898240ede28726079ba0418f62667e8d2b49 - Enable Stereo Duplex on M1
 * db097be3849ea957092aaf34fe8791afc93c8c59 - Make metal stereo conv1 kernel available
 * 17d97d3c5faccd6f0cdce17d2ae52bc923b25fee - Make ``MetalModel`` capable of handling > 1 input channel (Required for Stereo Duplex)
 * 6d5f07f6784a6e77ed86339d57abb78b20cf3830 - Fix memory leak in getting read groups
 * 3e3b21a4c5e571ee420e8b37cbafd42aa6b7fd5a - Remove deprecated use of FindCUDA and show real location of found toolkit
 * ff80a9fae368389271d01f84419f776b93c84461 - Improvements to Mk1C performance
 * 35dcb6558924b91160b16e6eb136696fd4eadebd - Fix meaning of TLEN in SAM output
 * 7f13113ecc215a0eec59b62457b05fa9399bf551 - Add support for arbitrary messages in Dorado pipelines
 * a93ae2cbeb8ea325b009b3932e798ff007284ee6 - Addition of a progress bar for basecalling
 * a93ae2cbeb8ea325b009b3932e798ff007284ee6 - Solve bug which was occuring when a read had no mod base context hits
 * 22a3140528f4b01ffb7e81c9bc668ec82aed20ab - Refactoring of Dorado pipeline system to Add AsyncQueue and use it in ReadSink/Nodes
 * 824459e4f4b8a7fa4c160c1af76d2a5ef760c66f - Add `"cuda:auto"` as alternative to `"cuda:all"` when selecting a compute accelerator device on CLI
 * d0c9387fc6083eddd5fb4c1659d0e73f99c32b10 - Store reads raw data internally in fp16 format
 * 6cd81705f879e350f1bb15760c02204945679385 - Switch post-ScalerNode ``raw_data`` to float16

# [0.2.1] (15 Feb 2023)

 * 121dddf9a3a288ca2c01dd2b732017a8f02c19a2 - Fix malformed SAM header

# [0.2.0] (15 Feb 2023)

 * 9b8ea1133b94e4071d7214000c8d7e10ed379540 - v4.1.0 with models higher accuracy basecalling in low-complexity genomic regions
 * 1311cbe06fbfa1a09785d4f3171548f99e0f739e - Increased basecalling performance on macOS
 * 74a04fc51765750e712289ecc7b7f9f162a96a54 - The stereo duplex model has been updated with calibrated qscores parameters
 * 1170ef6d0e0c04729530851b33d79e03ec65df4a - SAM output now include read groups
 * 63fb334a687876565c05e47707500142456a5ee4 - linux-arm64 builds
 * 117be24d9db35417ed8c06414eaac1a0f9349013 - Added `--read-id` filter and only basecalling paired reads in duplex for higher basecalling speed
 * a8ec89510b3f7208e43d41757bc3b025eb28e0b3 - Added `--min-qscore` filter
 * c80bae6020e7d60c17da54d3a47cdc0f876fd199 - Set builds to default to Release
 * e802181f401d7a2d283076f54e8d7da283e16f78 - Modbase CPU fixes
 * f0b96548651a9365712e0ca93483914a222a6bb4 - Better auto-batchsize detection on macos
 * 1a422db5b4008138927f12e1ad01410ef35a7139 - Switch to transparent model URLs
 * c2e694e66e573f7d4f49e41ce5a48a0c643918a5 - 260bps v4.0.0 5mCG_5hmCG@v2 models
 * 6b9249f4cc64ecb43134239fba2fe5682c5deb72 - Initial CUDA 12.0 support

# [0.1.1] (12 Dec 2022)

 * 38b953f3863d44067593adb49d9496dd704fef69 - Improved stereo duplex calling performance
 * 0de73feeeab9cd21f4b5c72685f7008fe143f846 - Fixed reverse calls in stereo
 * 1f62d8d71dc3393e78b4bb572576ba057d2f155b - Fixed modification model matching

# [0.1.0] (05 Dec 2022)

 * 0ff3e59b3e6bf1231e7e4b5d4521de0f6c371278 - Introducing stereo duplex calling introduced
 * fe42f660d962782213b351fb057929fcd88cb67b - New v4.0.0 simplex models for kit 14
 * 9e9fae377b2ec95f6fa80bb8234fe38cf05f50ea - v4.0.0 400bps simplex 5mCG_5hmGC modification models
 * 59aef4968cdacb671ecf1d707a35951c08829e5b - Simplified modification calling interface `--modified-bases 5mCG_5hmGC`
 * c8d4f39eb37986548b7703cac4e3d66af4f8ff0c - Initial RNA calling support
 * 9747efde97887d06721aa357023faa8b3836681f - Remove move table overhang
 * eb4854a280e1dcdd7af19c7194c8ae8a40c2f459 - Improved simplex calling performance by selecting the optimal batchsize at startup

# [0.0.4] (21 Nov 2022)

 * ae353913ff86af943f5c7569e0692e0d6662c3b1 - Basespace duplex
 * 0e1541b206a6cd3a17ca845da3a780f635ff55b4 - Simplex v4 model support
 * 6855fc1a4a4a76609f9232d7a692a7bb231bc5bb - Basecalling performance improvements

# [0.0.3] (12 Nov 2022)

Upgrade to Pod5 v0.0.41

 * 47be749be07b3f0de05206308103da58dd8cca4b - Support new pod5 version (0.0.41)

# [0.0.2] (10 Nov 2022)

Release of version 0.0.2 is a minor release which introduces several performance and usability improvements to Dorado. In particular, we are happy to announce the inclusion of multi-GPU base modification calling and 5mC models.

## Major changes

 * 163615381a3ba7b3c247d001df348535a7a112fd - Major upgrade to modified base calling speed.
 * fea918116182fe8f6809cf332900209a13a57bf5 61a7bc0181f0bf3e2e46727ccdf0717b0ee015e2 - Improve convolutional layer performance for increase in basecalling speed.
 * fb7cb242779c0cf3c6da73cbc7b57d892e97a5f8 - Reduction in memory requirement for M1 basecalling, allows running on larger number of M1 devices.
 * f5371d16cfd7eef71995edbfeeb86df7f4201440 - Add signal move tables to SAM output.
 * 621cc4d7624d8c99569a6cad5bd1bb7842e45be9 - Fix CPU basecalling
 * 9c4bf7172024a5d8fc7719890ff62f7d701f6248 - Add `dorado` namespace
 * acbca36edb1459d493df7aa445069e9f2e725f9f - Improvements to Dorado portability - Dorado archive now bundles more dependencies.

## Minor changes

 * 16cb5cb6f15ad6e04cb806ee03c5bebe2a7d45ae - Fix downloading models in Windows
 * 3202db8ff153dbad7d62d7efbddcbdeef70c3e9d - Basecaller node timeout, enables computation of non-full batches after timeout elapses.
 * 1a40836f2d8e1054232f05563298d6905955782e - Add duration tag to SAM.
 * 0b3a3367d7e44eb2470913a16806f2bd6d146860 - Add verbose flag to Dorado.
 * d75303d6e4d3bd97e4363e00210e1fef96db6d49 - Solve segfault-causing race condition
 * ac0a4d5a7946ad9b1036ab90ab54f27c9913bb0e - Adjust overlap and chunk size to be a multiple of the model stride
 * 626137a2fae13b8ac641fc7c032bb0ae150cbfad - Improve the way we look up GPU core count on Apple
 * b2864dbf198b0a3974828387cc89d43c06b3fa97 - Added [spdlog](https://github.com/gabime/spdlog) logging framework
 * e9c78665ed8cf4d76de5a0fb57cd19fdd03c5a44 - Reduced Startup cost on multi-GPU systems
 * fe71021aa24ffdb6ac110d5ce4f9a47933c03a53 - Migrate archive distributions from Box to CDN
 * 4b67720fd1037dc002264badf11f743d372dba3a - Resolved issue with SSL verification for model downloads.

# [0.0.1] (05 Oct 2022)

We are excited to announce the first binary release of Dorado. This Dorado release introduces important new features such as support for modified base calling, and significant improvements to basecalling performance and usability, taking it to the state of the art for speed and accuracy.

## Major changes

 * d3ddd1f078adc5b52ebfbb7d6aa5ee71acb0b7fb, 37e28f7b3d70dda469f3c498dcbe1ea5df722936, - Support for mod base calling, performance enhancements.
 * dd79bf5fb4b005052eb46969cfadd8ef2af8378e, 2a7fc176a5c0075a6fbf95dd3f7a41d52e420963, 465cb4a29e8cfd45b74064f13eb5c152fa2fa1c6, 56482fbd364a8d2cacb608b13b3a7f1792a604e3 -  Support for basecalling on M1 family Apple Silicon.
 * bd6014edc8de374645ade284dd103eccbfa481db - Support for basecalling on systems with multiple Nvidia GPUs.
 * 41fdb1189a4677c6932a4c4467d69c73407dfaaa - Support for POD5 file format.
 * 075065447d1a273f3101037c1578647bc2ad8b1e - Addition of new “Quantile” - based read scaling algorithm for higher accuracy.
 * 8acf2baa35932a9c42a419b6b620f92e25a87bba - Upgrade to torch 1.12.1
 * 9955d0de71d36e279b44e46545f5dfb6c742f224 - Added fast int8-quantization optimisation for LSTM networks with layer size of 96 or 128
 * f2e993d3961a52072cf43b0f327dbc21029c3aad - New cuBLAS-based implementation for LSTMs leading to state-of-the-art performance.
 * 6ec50dc5cecc65f0ff940420c0de152ba561f85c - Major rearchitecture of CUDA model runners for  higher basecalling speed and lower GPU memory utilisation.
 * a0a197f4950d390221b6ffc82ccd8ce012c3c765 - Accuracy improvements to handling of short reads (<1Kb) with an upgraded padding strategy.
 * d01bf04f7dd84b14753790fae83ba20b7776f498 - Ability to download basecalling models from Dorado.
 * 7c7e59c6d65464f0eee40bf3d9f6885aae27839b - Support for SAM output

## Minor Changes

 * 0e89d633d66f36256ad437ca0a3b64ff9eb0b1a1 - Automatic selection of batch size if user does not specify.
 * 6afceea0195c07b02a40e43e0a395c3d82d44add - Dorado version added to SAM output, including commit hash
 * 339b2fc5d7eee5be7f8289d51422a70ad06f6d58 - Scaling information recorded in SAM output
 * afbfab92f8207b9a67aae0aa87478c4b95e647b8 - Timestamps added to SAM output
 * 9ec2d970a0e5dee739daaafdfd08c20844140cb3 - Support for multi-threaded read scaling,  preventing CPU bottlenecks and improving basecall speed.
 * 7cbdbe04e76edf7d704e28263d64dddd6ab7d375 - Support for multi-threaded POD5 reading for higher data ingestion rate and improved performance.
 * 5a33e83512343e9fd36470fa84fa36c97211672b - Automatic querying of M1 device type for detection of optimal basecalling parameters.
 * 42703d0c02638633b44f68c0fc53534a9566b634 - Basecalling progress  (Number of reads basecalled) printed out to terminal on Linux.
