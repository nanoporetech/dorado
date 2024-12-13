Test skip_set with the first column empty. This should throw.
  $ rm -rf out; mkdir -p out
  > in_reads=${TEST_DATA_DIR}/read_correction/reads.fq
  > model_var=${MODEL_DIR:+--model-path ${MODEL_DIR}}
  > printf "\tdf814002-1961-4262-aaf5-e8f2760aa77a" > out/skip_set.txt
  > ${DORADO_BIN} correct ${in_reads} -t 4 ${model_var} --resume-from out/skip_set.txt -v > out/out.fasta 2> out/out.fasta.stderr
  > grep "Caught exception: Found empty string" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  1
