Test `--compute-num-blocks` with default index size (8GB). Since the input is small, there should be only 1 block.
  $ rm -rf out; mkdir -p out
  > in_reads=${TEST_DATA_DIR}/read_correction/reads.fq
  > ${DORADO_BIN} correct ${in_reads} --device cpu -t 4 --compute-num-blocks -v > out/out.num_blocks 2> out/out.fasta.stderr
  > grep -i "\[error\]" out/out.fasta.stderr | sed -E 's/^.*error] //g'
  > grep -i "\[warning\]" out/out.fasta.stderr | sed -E 's/^.*warning] //g'
  > cat out/out.num_blocks
  1

Empty input test.
  $ rm -rf out; mkdir -p out
  > in_reads=out/in.fastq
  > touch ${in_reads}
  > touch ${in_reads}.fai
  > ${DORADO_BIN} correct ${in_reads} --device cpu -t 4 --compute-num-blocks -v --index-size 100000 > out/out.num_blocks 2> out/out.fasta.stderr
  > grep -i "\[error\]" out/out.fasta.stderr | sed -E 's/^.*error] //g'
  > grep -i "\[warning\]" out/out.fasta.stderr | sed -E 's/^.*warning] //g'
  Caught exception: Failed to load index file out/in.fastq.

Test `--compute-num-blocks` with index-size of 100,000 bp. There should be 4 blocks containing this many reads: (1) 1, (2) 2, (3) 2, (4) 1.
  $ rm -rf out; mkdir -p out
  > in_reads=${TEST_DATA_DIR}/read_correction/reads.fq
  > ${DORADO_BIN} correct ${in_reads} --device cpu -t 4 --compute-num-blocks -v --index-size 100000 > out/out.num_blocks 2> out/out.fasta.stderr
  > grep -i "\[error\]" out/out.fasta.stderr | sed -E 's/^.*error] //g'
  > grep -i "\[warning\]" out/out.fasta.stderr | sed -E 's/^.*warning] //g'
  > cat out/out.num_blocks
  4

Test `--compute-num-blocks` with index-size of 150,000 bp. There should be 3 blocks containing this many reads: (1) 2, (2) 3, (3)1.
  $ rm -rf out; mkdir -p out
  > in_reads=${TEST_DATA_DIR}/read_correction/reads.fq
  > ${DORADO_BIN} correct ${in_reads} --device cpu -t 4 --compute-num-blocks -v --index-size 150000 > out/out.num_blocks 2> out/out.fasta.stderr
  > grep -i "\[error\]" out/out.fasta.stderr | sed -E 's/^.*error] //g'
  > grep -i "\[warning\]" out/out.fasta.stderr | sed -E 's/^.*warning] //g'
  > cat out/out.num_blocks
  3