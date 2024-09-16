Test a skip set composed of 4 reads, and a proper default index-size of 8GB as is in production.
  $ rm -rf out; mkdir -p out
  > in_reads=${TEST_DATA_DIR}/read_correction/reads.fq
  > model_var=${MODEL_DIR:+--model-path ${MODEL_DIR}}
  > cat ${in_reads}.fai | cut -f 1,1 | head -n 4 > out/skip_set.txt
  > ${DORADO_BIN} correct ${in_reads} -t 4 ${model_var} --resume-from out/skip_set.txt > out/out.fasta 2> out/out.fasta.stderr
  > grep -i "\[error\]" out/out.fasta.stderr
  > grep -i "\[warning\]" out/out.fasta.stderr
  > samtools faidx out/out.fasta
  > cat out/out.fasta.fai | cut -f1,1 | sort
  3855985e-bb9b-4df4-9825-cc08f373342b
  b93514e5-c61b-48d8-b730-f6c97d169ff7
