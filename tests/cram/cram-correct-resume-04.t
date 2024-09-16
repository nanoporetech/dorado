
Test a larger index-size of 100000bp. There will be several chunks containing this many reads: (1) 1, (2) 2, (3) 2, (4) 1.
The skip_set read is located in chunk (3) (first read in its chunk).
  $ rm -rf out; mkdir -p out
  > in_reads=${TEST_DATA_DIR}/read_correction/reads.fq
  > model_var=${MODEL_DIR:+--model-path ${MODEL_DIR}}
  > echo "df814002-1961-4262-aaf5-e8f2760aa77a" > out/skip_set.txt
  > ${DORADO_BIN} correct ${in_reads} -t 4 ${model_var} --resume-from out/skip_set.txt -v --index-size 100000 > out/out.fasta 2> out/out.fasta.stderr
  > grep -i "\[error\]" out/out.fasta.stderr
  > grep -i "\[warning\]" out/out.fasta.stderr
  > samtools faidx out/out.fasta
  > cat out/out.fasta.fai | cut -f1,1 | sort
  3855985e-bb9b-4df4-9825-cc08f373342b
  b93514e5-c61b-48d8-b730-f6c97d169ff7
