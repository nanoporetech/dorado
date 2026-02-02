Input BAM and Draft are empty.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > in_bam=out/in.aln.bam
  > in_ref=out/in.ref.fasta
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > touch ${in_bam}
  > touch ${in_ref}
  > ${DORADO_BIN} variant --device cpu ${in_bam} ${in_ref} -t 4 ${model_var} > out/out.vcf 2> out/stderr
  > echo "Exit code: $?"
  > grep "\[error\]" out/stderr | sed -E 's/^.*error] //g'
  Exit code: 1
  Input file 'out/in.aln.bam' does not exist or is empty.

Input BAM is not empty, but Draft is empty.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > in_bam=${in_dir}/in.aln.bam
  > in_ref=out/in.draft.fasta
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > touch ${in_ref}
  > ${DORADO_BIN} variant --device cpu ${in_bam} ${in_ref} -t 4 ${model_var} > out/out.vcf 2> out/stderr
  > echo "Exit code: $?"
  > grep "\[error\]" out/stderr | sed -E 's/^.*error] //g'
  Exit code: 1
  Input file 'out/in.draft.fasta' does not exist or is empty.
