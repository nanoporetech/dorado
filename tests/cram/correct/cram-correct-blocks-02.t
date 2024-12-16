Test `--run-block-id` when there is only 1 large block.
Input is a valid value of 0.
  $ rm -rf out; mkdir -p out
  > in_reads=${TEST_DATA_DIR}/read_correction/reads.fq
  > model_var=${MODEL_DIR:+--model-path ${MODEL_DIR}}
  > ${DORADO_BIN} correct ${in_reads} --device cpu -t 4 ${model_var} -v --run-block-id 0 > out/out.fasta 2> out/out.fasta.stderr
  > grep -i "\[error\]" out/out.fasta.stderr
  > grep -i "\[warning\]" out/out.fasta.stderr
  > samtools faidx out/out.fasta
  > cat out/out.fasta.fai | cut -f1,1 | sort
  > grep "Align with index" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  3855985e-bb9b-4df4-9825-cc08f373342b
  3c88104d-7964-43dc-ac47-c22b12cdc994
  73d5eb75-700e-42a0-9a5d-f9952bd7d829
  b93514e5-c61b-48d8-b730-f6c97d169ff7
  df814002-1961-4262-aaf5-e8f2760aa77a
  e3066d3e-2bdf-4803-89b9-0f077ac7ff7f
  1

Test `--run-block-id` when there is only 1 large block.
Block id is negative.
  $ rm -rf out; mkdir -p out
  > in_reads=${TEST_DATA_DIR}/read_correction/reads.fq
  > model_var=${MODEL_DIR:+--model-path ${MODEL_DIR}}
  > ${DORADO_BIN} correct ${in_reads} --device cpu -t 4 ${model_var} -v --run-block-id -1 > out/out.fasta 2> out/out.fasta.stderr
  > grep -i "\[error\]" out/out.fasta.stderr | sed -E 's/^.*error] //g'
  > grep -i "\[warning\]" out/out.fasta.stderr | sed -E 's/^.*warning] //g'
  The --run-block-id option cannot be negative.

Test `--run-block-id` when there are 4 blocks and we only run one of them. Index-size is 100,000 bp, so there should be 4 blocks containing this many reads: (1) 1, (2) 2, (3) 2, (4) 1.
Processing only block ID 0.
All input reads are:
# e3066d3e-2bdf-4803-89b9-0f077ac7ff7f	109743	38	109743	109744	109784
# 73d5eb75-700e-42a0-9a5d-f9952bd7d829	88713	219566	88713	88714	308282
# 3c88104d-7964-43dc-ac47-c22b12cdc994	59049	397034	59049	59050	456086
# df814002-1961-4262-aaf5-e8f2760aa77a	62692	515174	62692	62693	577869
# b93514e5-c61b-48d8-b730-f6c97d169ff7	49685	640600	49685	49686	690288
# 3855985e-bb9b-4df4-9825-cc08f373342b	80252	740012	80252	80253	820267
  $ rm -rf out; mkdir -p out
  > in_reads=${TEST_DATA_DIR}/read_correction/reads.fq
  > model_var=${MODEL_DIR:+--model-path ${MODEL_DIR}}
  > ${DORADO_BIN} correct ${in_reads} --device cpu -t 4 ${model_var} -v --index-size 100000 --run-block-id 0 > out/out.fasta 2> out/out.fasta.stderr
  > grep -i "\[error\]" out/out.fasta.stderr
  > grep -i "\[warning\]" out/out.fasta.stderr
  > samtools faidx out/out.fasta
  > cat out/out.fasta.fai | cut -f1,1 | sort
  > grep "Align with index" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  e3066d3e-2bdf-4803-89b9-0f077ac7ff7f
  1

Same test as the previous one but with `--run-block-id 1`.
Index-size is 100,000 bp, so there should be 4 blocks containing this many reads: (1) 1, (2) 2, (3) 2, (4) 1. Processing only block ID 1.
All input reads are:
# e3066d3e-2bdf-4803-89b9-0f077ac7ff7f	109743	38	109743	109744	109784
# 73d5eb75-700e-42a0-9a5d-f9952bd7d829	88713	219566	88713	88714	308282
# 3c88104d-7964-43dc-ac47-c22b12cdc994	59049	397034	59049	59050	456086
# df814002-1961-4262-aaf5-e8f2760aa77a	62692	515174	62692	62693	577869
# b93514e5-c61b-48d8-b730-f6c97d169ff7	49685	640600	49685	49686	690288
# 3855985e-bb9b-4df4-9825-cc08f373342b	80252	740012	80252	80253	820267
  $ rm -rf out; mkdir -p out
  > in_reads=${TEST_DATA_DIR}/read_correction/reads.fq
  > model_var=${MODEL_DIR:+--model-path ${MODEL_DIR}}
  > ${DORADO_BIN} correct ${in_reads} --device cpu -t 4 ${model_var} -v --index-size 100000 --run-block-id 1 > out/out.fasta 2> out/out.fasta.stderr
  > grep -i "\[error\]" out/out.fasta.stderr
  > grep -i "\[warning\]" out/out.fasta.stderr
  > samtools faidx out/out.fasta
  > cat out/out.fasta.fai | cut -f1,1 | sort
  > grep "Align with index" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  3c88104d-7964-43dc-ac47-c22b12cdc994
  73d5eb75-700e-42a0-9a5d-f9952bd7d829
  1

Same test as the previous one but with `--run-block-id 2`.
Index-size is 100,000 bp, so there should be 4 blocks containing this many reads: (1) 1, (2) 2, (3) 2, (4) 1. Processing only block ID 1.
All input reads are:
# e3066d3e-2bdf-4803-89b9-0f077ac7ff7f	109743	38	109743	109744	109784
# 73d5eb75-700e-42a0-9a5d-f9952bd7d829	88713	219566	88713	88714	308282
# 3c88104d-7964-43dc-ac47-c22b12cdc994	59049	397034	59049	59050	456086
# df814002-1961-4262-aaf5-e8f2760aa77a	62692	515174	62692	62693	577869
# b93514e5-c61b-48d8-b730-f6c97d169ff7	49685	640600	49685	49686	690288
# 3855985e-bb9b-4df4-9825-cc08f373342b	80252	740012	80252	80253	820267
  $ rm -rf out; mkdir -p out
  > in_reads=${TEST_DATA_DIR}/read_correction/reads.fq
  > model_var=${MODEL_DIR:+--model-path ${MODEL_DIR}}
  > ${DORADO_BIN} correct ${in_reads} --device cpu -t 4 ${model_var} -v --index-size 100000 --run-block-id 2 > out/out.fasta 2> out/out.fasta.stderr
  > grep -i "\[error\]" out/out.fasta.stderr
  > grep -i "\[warning\]" out/out.fasta.stderr
  > samtools faidx out/out.fasta
  > cat out/out.fasta.fai | cut -f1,1 | sort
  > grep "Align with index" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  b93514e5-c61b-48d8-b730-f6c97d169ff7
  df814002-1961-4262-aaf5-e8f2760aa77a
  1

Same test as the previous one but with `--run-block-id 3`.
Index-size is 100,000 bp, so there should be 4 blocks containing this many reads: (1) 1, (2) 2, (3) 2, (4) 1. Processing only block ID 1.
All input reads are:
# e3066d3e-2bdf-4803-89b9-0f077ac7ff7f	109743	38	109743	109744	109784
# 73d5eb75-700e-42a0-9a5d-f9952bd7d829	88713	219566	88713	88714	308282
# 3c88104d-7964-43dc-ac47-c22b12cdc994	59049	397034	59049	59050	456086
# df814002-1961-4262-aaf5-e8f2760aa77a	62692	515174	62692	62693	577869
# b93514e5-c61b-48d8-b730-f6c97d169ff7	49685	640600	49685	49686	690288
# 3855985e-bb9b-4df4-9825-cc08f373342b	80252	740012	80252	80253	820267
  $ rm -rf out; mkdir -p out
  > in_reads=${TEST_DATA_DIR}/read_correction/reads.fq
  > model_var=${MODEL_DIR:+--model-path ${MODEL_DIR}}
  > ${DORADO_BIN} correct ${in_reads} --device cpu -t 4 ${model_var} -v --index-size 100000 --run-block-id 3 > out/out.fasta 2> out/out.fasta.stderr
  > grep -i "\[error\]" out/out.fasta.stderr
  > grep -i "\[warning\]" out/out.fasta.stderr
  > samtools faidx out/out.fasta
  > cat out/out.fasta.fai | cut -f1,1 | sort
  > grep "Align with index" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  3855985e-bb9b-4df4-9825-cc08f373342b
  1

Block ID `15` is well above the number of blocks.
Empty output should be generated
  $ rm -rf out; mkdir -p out
  > in_reads=${TEST_DATA_DIR}/read_correction/reads.fq
  > model_var=${MODEL_DIR:+--model-path ${MODEL_DIR}}
  > ${DORADO_BIN} correct ${in_reads} --device cpu -t 4 ${model_var} -v --index-size 100000 --run-block-id 15 > out/out.fasta 2> out/out.fasta.stderr
  > grep -i "\[error\]" out/out.fasta.stderr
  > grep -i "\[warning\]" out/out.fasta.stderr
  > wc -l out/out.fasta | awk '{ print $1 }'
  0