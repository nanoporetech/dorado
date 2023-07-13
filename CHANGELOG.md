# Changelog

All notable changes to Dorado will be documented in this file.


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
