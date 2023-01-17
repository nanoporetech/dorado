# Changelog

All notable changes to Dorado will be documented in this file.

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
