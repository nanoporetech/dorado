Run a tiny test case of 10kbp. The entire sequence fits into one window.
FASTA output.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=${in_dir}/calls_to_draft.bam
  > in_draft=${in_dir}/draft.fasta.gz
  > expected=${in_dir}/medaka.consensus.fastq.gz
  > model_var=${MODEL_DIR:+--model-path ${MODEL_DIR}}
  > ${DORADO_BIN} polish --window-len 10000 --window-overlap 1000 --bam-chunk 1000000 ${in_bam} ${in_draft} out/out.fasta -t 4 ${model_var} 2> out/stderr
  > gunzip -d --stdout ${expected} | head -n 2 | sed 's/@/>/g' > out/expected.fasta
  > diff out/expected.fasta out/out.fasta

# FASTQ output, otherwise the same.
#   $ rm -rf out; mkdir -p out
#   > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
#   > in_bam=${in_dir}/calls_to_draft.bam
#   > in_draft=${in_dir}/draft.fasta.gz
#   > expected=${in_dir}/medaka.consensus.fastq.gz
#   > model_var=${MODEL_DIR:+--model-path ${MODEL_DIR}}
#   > ${DORADO_BIN} polish ${in_bam} ${in_draft} out/out.fastq -t 4 ${model_var} 2> out/stderr
#   > gunzip -d --stdout ${expected} > out/expected.fastq
#   > diff out/expected.fastq out/out.fastq
