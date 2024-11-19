Write output to a path defined by the `-o` option.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=${in_dir}/calls_to_draft.bam
  > in_draft=${in_dir}/draft.fasta.gz
  > expected=${in_dir}/medaka.consensus.fastq.gz
  > model_var=${MODEL_DIR:+--model-path ${MODEL_DIR}}
  > ${DORADO_BIN} polish --window-len 10000 --window-overlap 1000 --bam-chunk 1000000 ${in_bam} ${in_draft} -t 4 ${model_var} -o out/out.fasta 2> out/stderr
  > gunzip -d --stdout ${expected} | head -n 2 | sed 's/@/>/g' > out/expected.fasta
  > grep "Copying contig verbatim from input" out/stderr | wc -l | awk '{ print $1 }'
  > diff out/expected.fasta out/out.fasta
  > wc -l out/out.fasta | awk '{ print $1 }'
  0
  2

Write output to a path defined by the `--out-path` option.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=${in_dir}/calls_to_draft.bam
  > in_draft=${in_dir}/draft.fasta.gz
  > expected=${in_dir}/medaka.consensus.fastq.gz
  > model_var=${MODEL_DIR:+--model-path ${MODEL_DIR}}
  > ${DORADO_BIN} polish --window-len 10000 --window-overlap 1000 --bam-chunk 1000000 ${in_bam} ${in_draft} -t 4 ${model_var} --out-path out/out.fasta 2> out/stderr
  > gunzip -d --stdout ${expected} | head -n 2 | sed 's/@/>/g' > out/expected.fasta
  > grep "Copying contig verbatim from input" out/stderr | wc -l | awk '{ print $1 }'
  > diff out/expected.fasta out/out.fasta
  > wc -l out/out.fasta | awk '{ print $1 }'
  0
  2

Write QVs. The QVs still do not match exactly the ones in Medaka.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=${in_dir}/calls_to_draft.bam
  > in_draft=${in_dir}/draft.fasta.gz
  > expected=${in_dir}/medaka.consensus.fastq.gz
  > model_var=${MODEL_DIR:+--model-path ${MODEL_DIR}}
  > ${DORADO_BIN} polish --window-len 10000 --window-overlap 1000 --bam-chunk 1000000 ${in_bam} ${in_draft} -t 4 ${model_var} -o out/out.fastq -q 2> out/stderr
  > gunzip -d --stdout ${expected} | head -n 2 | sed 's/@/>/g' > out/expected.fasta
  > head -n 2 out/out.fastq | sed 's/@/>/g' > out/out.fasta
  > wc -l out/out.fastq | awk '{ print $1 }'
  > diff out/expected.fasta out/out.fasta
  > grep "Copying contig verbatim from input" out/stderr | wc -l | awk '{ print $1 }'
  4
  0