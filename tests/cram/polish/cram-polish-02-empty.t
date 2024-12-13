Input BAM and Draft are empty.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=out/calls_to_draft.bam
  > in_draft=out/in.draft.fasta
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > touch ${in_bam}
  > touch ${in_draft}
  > ${DORADO_BIN} polish --device cpu ${in_bam} ${in_draft} -t 4 ${model_var} > out/out.fasta 2> out/stderr
  > echo "Exit code: $?"
  > grep "\[error\]" out/stderr | sed -E 's/^.*error] //g'
  Exit code: 1
  Input file out/calls_to_draft.bam does not exist or is empty.

Input BAM is not empty, but Draft is empty.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=${in_dir}/calls_to_draft.bam
  > in_draft=out/in.draft.fasta
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > touch ${in_draft}
  > ${DORADO_BIN} polish --device cpu ${in_bam} ${in_draft} -t 4 ${model_var} > out/out.fasta 2> out/stderr
  > echo "Exit code: $?"
  > grep "\[error\]" out/stderr | sed -E 's/^.*error] //g'
  Exit code: 1
  Input file out/in.draft.fasta does not exist or is empty.
