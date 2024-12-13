Test the resume functionality. The input consists of 6 reads (below). The skip set has only 1 read (marked below).
Everything before this read is presumed to have been processed before.
Everything after this read still needs to be generated in the output.
- "e3066d3e-2bdf-4803-89b9-0f077ac7ff7f"    <- Previously processed
- "73d5eb75-700e-42a0-9a5d-f9952bd7d829"    <- Previously processed
- "3c88104d-7964-43dc-ac47-c22b12cdc994"    <- Previously processed
- "df814002-1961-4262-aaf5-e8f2760aa77a"    <- `skip_set.txt`
- "b93514e5-c61b-48d8-b730-f6c97d169ff7"    <- Needs to be generated.
- "3855985e-bb9b-4df4-9825-cc08f373342b"    <- Needs to be generated.
Using the `--index-size 1` (1bp in size) will make each read to be in its own chunk.
  $ rm -rf out; mkdir -p out
  > in_reads=${TEST_DATA_DIR}/read_correction/reads.fq
  > model_var=${MODEL_DIR:+--model-path ${MODEL_DIR}}
  > echo "df814002-1961-4262-aaf5-e8f2760aa77a" > out/skip_set.txt
  > ${DORADO_BIN} correct ${in_reads} -t 4 ${model_var} --resume-from out/skip_set.txt -v --index-size 1 > out/out.fasta 2> out/out.fasta.stderr
  > grep -i "\[error\]" out/out.fasta.stderr
  > grep -i "\[warning\]" out/out.fasta.stderr
  > samtools faidx out/out.fasta
  > cat out/out.fasta.fai | cut -f1,1 | sort
  3855985e-bb9b-4df4-9825-cc08f373342b
  b93514e5-c61b-48d8-b730-f6c97d169ff7
