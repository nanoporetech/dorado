Test a proper default index-size of 8GB as is in production.
The entire dataset will be loaded in only one chunk.
Since all reads are in the same chunk as the skip_set read, everything will be overlapped.
Actual filtering will happen at the inference stage, when inference is skipped for reads stored in the skip_set.
That is why the output has 5 reads.
  $ rm -rf out; mkdir -p out
  > in_reads=${TEST_DATA_DIR}/read_correction/reads.fq
  > model_var=${MODEL_DIR:+--model-path ${MODEL_DIR}}
  > echo "df814002-1961-4262-aaf5-e8f2760aa77a" > out/skip_set.txt
  > ${DORADO_BIN} correct ${in_reads} -t 4 ${model_var} --resume-from out/skip_set.txt -v > out/out.fasta 2> out/out.fasta.stderr
  > grep -i "\[error\]" out/out.fasta.stderr
  > grep -i "\[warning\]" out/out.fasta.stderr
  > samtools faidx out/out.fasta
  > cat out/out.fasta.fai | cut -f1,1 | sort
  3855985e-bb9b-4df4-9825-cc08f373342b
  3c88104d-7964-43dc-ac47-c22b12cdc994
  73d5eb75-700e-42a0-9a5d-f9952bd7d829
  b93514e5-c61b-48d8-b730-f6c97d169ff7
  e3066d3e-2bdf-4803-89b9-0f077ac7ff7f
