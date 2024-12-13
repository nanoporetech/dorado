Test resuming in combination with decoupled mapping/inference.
Both the mapping and the inference are run with the skip set.
Using the `--index-size 1` (1bp in size) will make each read to be in its own chunk.
  $ rm -rf out; mkdir -p out
  > in_reads=${TEST_DATA_DIR}/read_correction/reads.fq
  > model_var=${MODEL_DIR:+--model-path ${MODEL_DIR}}
  > echo "df814002-1961-4262-aaf5-e8f2760aa77a" > out/skip_set.txt
  > ${DORADO_BIN} correct ${in_reads} -t 4 ${model_var} --resume-from out/skip_set.txt -v --index-size 1 --to-paf > out/out.paf 2> out/out.paf.stderr
  > ${DORADO_BIN} correct ${in_reads} -t 4 ${model_var} --resume-from out/skip_set.txt -v --index-size 1 --from-paf out/out.paf > out/out.fasta 2> out/out.fasta.stderr
  > grep -i "\[error\]" out/out.paf.stderr out/out.fasta.stderr
  > grep -i "\[warning\]" out/out.paf.stderr out/out.fasta.stderr
  > samtools faidx out/out.fasta
  > cut -f6,6 out/out.paf | sort | uniq
  > echo "-"
  > cat out/out.fasta.fai | cut -f1,1 | sort
  3855985e-bb9b-4df4-9825-cc08f373342b
  b93514e5-c61b-48d8-b730-f6c97d169ff7
  -
  3855985e-bb9b-4df4-9825-cc08f373342b
  b93514e5-c61b-48d8-b730-f6c97d169ff7

Apply the skip-set in the mapping stage.
Expected: the last 3 target reads of the input reads.fq file.
Explanation: this test uses a larger index-size of 100000bp. There will be 4 chunks containing this many reads: (1) 1, (2) 2, (3) 2, (4) 1.
The skip_set read is located in chunk (3) (first read in its chunk). That means that only overlaps for chunks (3) and (4) will be generated,
containing in total 3 target reads. Read "df814002-1961-4262-aaf5-e8f2760aa77a" is in the skip set, so it will be skipped during writing to PAF.
  $ rm -rf out; mkdir -p out
  > in_reads=${TEST_DATA_DIR}/read_correction/reads.fq
  > model_var=${MODEL_DIR:+--model-path ${MODEL_DIR}}
  > echo "df814002-1961-4262-aaf5-e8f2760aa77a" > out/skip_set.txt
  > ${DORADO_BIN} correct ${in_reads} -t 4 ${model_var} --resume-from out/skip_set.txt -v --index-size 100000 --to-paf > out/out.paf 2> out/out.paf.stderr
  > grep -i "\[error\]" out/out.paf.stderr
  > grep -i "\[warning\]" out/out.paf.stderr
  > cut -f6,6 out/out.paf | sort | uniq
  3855985e-bb9b-4df4-9825-cc08f373342b
  b93514e5-c61b-48d8-b730-f6c97d169ff7

Apply the skip-set in the inference stage.
Mapping is done without a skip-set (i.e. all-vs-all overlaps are generated), while the `--resume-from` is applied only in the inference stage.
Only the single read in the skip-set should not be present in the output, because it is blacklisted in the skip-set.
  $ rm -rf out; mkdir -p out
  > in_reads=${TEST_DATA_DIR}/read_correction/reads.fq
  > model_var=${MODEL_DIR:+--model-path ${MODEL_DIR}}
  > echo "df814002-1961-4262-aaf5-e8f2760aa77a" > out/skip_set.txt
  > ${DORADO_BIN} correct ${in_reads} -t 4 ${model_var} -v --index-size 100000 --to-paf > out/out.paf 2> out/out.paf.stderr
  > ${DORADO_BIN} correct ${in_reads} -t 4 ${model_var} --resume-from out/skip_set.txt -v --index-size 100000 --from-paf out/out.paf > out/out.fasta 2> out/out.fasta.stderr
  > grep -i "\[error\]" out/out.paf.stderr out/out.fasta.stderr
  > grep -i "\[warning\]" out/out.paf.stderr out/out.fasta.stderr
  > samtools faidx out/out.fasta
  > cat out/out.fasta.fai | cut -f1,1 | sort
  3855985e-bb9b-4df4-9825-cc08f373342b
  3c88104d-7964-43dc-ac47-c22b12cdc994
  73d5eb75-700e-42a0-9a5d-f9952bd7d829
  b93514e5-c61b-48d8-b730-f6c97d169ff7
  e3066d3e-2bdf-4803-89b9-0f077ac7ff7f
