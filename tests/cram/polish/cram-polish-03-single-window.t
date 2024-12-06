Run a tiny test case of 10kbp. The entire sequence fits into one window.
FASTA output.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=${in_dir}/calls_to_draft.bam
  > in_draft=${in_dir}/draft.fasta.gz
  > expected=${in_dir}/medaka.consensus.fastq.gz
  > model_var=${MODEL_DIR:+--model-path ${MODEL_DIR}}
  > ${DORADO_BIN} polish --device cpu --window-len 10000 --window-overlap 1000 --bam-chunk 1000000 ${in_bam} ${in_draft} -t 4 ${model_var} > out/out.fasta 2> out/stderr
  > gunzip -d --stdout ${expected} | head -n 2 | sed 's/@/>/g' > out/expected.fasta
  > grep "Copying contig verbatim from input" out/stderr | wc -l | awk '{ print $1 }'
  > diff out/expected.fasta out/out.fasta
  0

FASTQ output, otherwise the same.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=${in_dir}/calls_to_draft.bam
  > in_draft=${in_dir}/draft.fasta.gz
  > expected=${in_dir}/medaka.consensus.fastq.gz
  > model_var=${MODEL_DIR:+--model-path ${MODEL_DIR}}
  > ${DORADO_BIN} polish --device cpu --window-len 10000 --window-overlap 1000 --bam-chunk 1000000 ${in_bam} ${in_draft} -t 4 ${model_var} --qualities > out/out.fastq 2> out/stderr
  > gunzip -d --stdout ${expected} > out/expected.fastq
  > diff out/expected.fastq out/out.fastq
