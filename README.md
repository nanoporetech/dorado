# Dorado

This is a *preview version* of Dorado, a Libtorch Basecaller for Oxford Nanopore Reads. This software is in alpha preview stage and being released for early evaluation. It is subject to change. If you encounter any problems building or running Dorado please [report an issue](https://github.com/nanoporetech/dorado/issues).

## Downloading Dorado

We will be publishing pre-built releases in the next few days.

## Running

```
$ dorado download --model dna_r10.4.1_e8.2_260bps_hac@v3.5.2
$ dorado basecaller dna_r10.4.1_e8.2_260bps_hac@v3.5.2 pod5/ > calls.sam
```

## Platforms

Dorado has been tested on the following systems:

| Platform | GPU/CPU              |
| -------- | -------------------- |
| Windows  | x86                  |
| Apple    | M1, M1 Max, M1 Ultra |
| Linux    | (G)V100, A100        |

Other Platforms may work, if you encounter problems with running on your system please [report an issue](https://github.com/nanoporetech/dorado/issues)

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
