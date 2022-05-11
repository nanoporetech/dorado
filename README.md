# Dorado

## Quickstart

```
$ git clone git@git.oxfordnanolabs.local:machine-learning/dorado.git
$ cd dorado
$ cmake -S . -B cmake-build
$ cmake --build cmake-build --config Release -- -j
$ ctest --test-dir cmake-build
```

## Running

```
./dorado basecaller dna_r9.4.1_e8_hac@v3.3 fast5_pass/ > calls.fastq
```
