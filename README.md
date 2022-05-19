# Dorado

This is a *preview version* of Dorado, a Libtorch Basecaller for Oxford Nanopore Reads. This software is alpha preview version being released for early evaluation. It is subject to change. If you encounter any problems building or running Dorado please [report an issue](https://github.com/nanoporetech/dorado/).

## Downloading Dorado
TBC

## Running

```
$ dorado download --model dna_r9.4.1_e8_hac@v3.3
$ dorado basecaller dna_r9.4.1_e8_hac@v3.3 fast5_pass/ > calls.sam
```

## Developer quickstart

```
$ git clone git@git.oxfordnanolabs.local:machine-learning/dorado.git
$ cd dorado
$ cmake -S . -B cmake-build
$ cmake --build cmake-build --config Release -- -j
$ ctest --test-dir cmake-build
```

## Platforms

Dorado has been tested on the following systems:

| Platform | GPU/CPU |
| ------ | ------ |
| Windows | x86 |
| Apple  | M1, M1 Max |
| Linux | A100 40GB PCIe|

Other Platforms may work, if you encounter problems with running on your system please [report an issue](https://github.com/nanoporetech/dorado/issues)

## Known limitations

* Multi-GPU support is limited and likely to work.
* GPU memory utilisation on Nvidia devices is high (compared to [Bonito](https://github.com/nanoporetech/bonito)). This issue is currently being investigated and resolved.
* Support for M1 GPUs is should be considered experimental.
