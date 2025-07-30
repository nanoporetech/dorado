Run a tiny test case using the default parameters.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=${in_dir}/calls_to_draft.bam
  > in_draft=${in_dir}/draft.fasta.gz
  > expected=${in_dir}/medaka.consensus.fastq.gz
  > model_var="--models-directory ${MODEL_ROOT_DIR}"
  > ${DORADO_BIN} polish --device cpu ${in_bam} ${in_draft} -t 4 -vv ${model_var} > out/out.fasta 2> out/stderr
  > echo "Exit code: $?"
  > gunzip -d --stdout ${expected} | head -n 2 | sed 's/@/>/g' > out/expected.fasta
  > grep "Copying contig verbatim from input" out/stderr | wc -l | awk '{ print $1 }'
  > diff out/expected.fasta out/out.fasta
  Exit code: 0
  0
