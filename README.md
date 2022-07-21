# Dorado

This is a *preview version* of Dorado, a Libtorch Basecaller for Oxford Nanopore Reads. This software is in alpha preview stage and being released for early evaluation. It is subject to change. If you encounter any problems building or running Dorado please [report an issue](https://github.com/nanoporetech/dorado/).

## Downloading Dorado
We will be publishing pre-built releases in the next few days.

## Running

```
$ dorado download --model dna_r9.4.1_e8_hac@v3.3
$ dorado basecaller dna_r9.4.1_e8_hac@v3.3 fast5_pass/ > calls.sam
```

## Platforms

Dorado has been tested on the following systems:

| Platform | GPU/CPU              |
| -------- | -------------------- |
| Windows  | x86                  |
| Apple    | M1, M1 Max, M1 Ultra |
| Linux    | A100 40GB PCIe       |

Other Platforms may work, if you encounter problems with running on your system please [report an issue](https://github.com/nanoporetech/dorado/issues)

## Known limitations

* Multi-GPU support is limited and likely not to work.
* GPU memory utilisation on Nvidia devices is high (compared to [Bonito](https://github.com/nanoporetech/bonito)). This issue is currently being investigated and resolved.
* Support for M1 GPUs is should be considered experimental.

## Developer quickstart

### Get Linux dependencies

```
apt-get update && apt-get install -y --no-install-recommends libhdf5-dev libssl-dev libzstd-dev
```

### Clone and build

```
$ git clone git@github.com:nanoporetech/dorado.git
$ cd dorado
$ cmake -S . -B cmake-build
$ cmake --build cmake-build --config Release -j
$ ctest --test-dir cmake-build
```

### Pre commit

The project uses pre-commit to ensure code is consistently formatted, you can set this up using pip:

```bash
$ pip install pre-commit
$ pre-commit install
```
