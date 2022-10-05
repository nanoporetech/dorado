# Changelog

All notable changes to Dorado will be documented in this file.

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
