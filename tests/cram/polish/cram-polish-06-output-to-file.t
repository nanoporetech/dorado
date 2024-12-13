Write output to a path defined by the `-o` option.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=${in_dir}/calls_to_draft.bam
  > in_draft=${in_dir}/draft.fasta.gz
  > expected=${in_dir}/medaka.consensus.fastq.gz
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > ${DORADO_BIN} polish --device cpu --window-len 10000 --window-overlap 1000 --bam-chunk 1000000 ${in_bam} ${in_draft} -t 4 ${model_var} -o out/out.fasta 2> out/stderr
  > echo "Exit code: $?"
  > gunzip -d --stdout ${expected} | head -n 2 | sed 's/@/>/g' > out/expected.fasta
  > grep "Copying contig verbatim from input" out/stderr | wc -l | awk '{ print $1 }'
  > diff out/expected.fasta out/out.fasta
  > wc -l out/out.fasta | awk '{ print $1 }'
  Exit code: 0
  0
  2

Write output to a path defined by the `--out-path` option.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=${in_dir}/calls_to_draft.bam
  > in_draft=${in_dir}/draft.fasta.gz
  > expected=${in_dir}/medaka.consensus.fastq.gz
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > ${DORADO_BIN} polish --device cpu --window-len 10000 --window-overlap 1000 --bam-chunk 1000000 ${in_bam} ${in_draft} -t 4 ${model_var} --out-path out/out.fasta 2> out/stderr
  > echo "Exit code: $?"
  > gunzip -d --stdout ${expected} | head -n 2 | sed 's/@/>/g' > out/expected.fasta
  > grep "Copying contig verbatim from input" out/stderr | wc -l | awk '{ print $1 }'
  > diff out/expected.fasta out/out.fasta
  > wc -l out/out.fasta | awk '{ print $1 }'
  Exit code: 0
  0
  2

Write output to a path defined by the `--out-path` option, but the output folder is not reachable.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=${in_dir}/calls_to_draft.bam
  > in_draft=${in_dir}/draft.fasta.gz
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > ${DORADO_BIN} polish --device cpu --window-len 10000 --window-overlap 1000 --bam-chunk 1000000 ${in_bam} ${in_draft} -t 4 ${model_var} --out-path this_folder_does_not_exist/out.fasta 2> out/stderr
  > echo "Exit code: $?"
  > grep "Failed to open file" out/stderr | wc -l | awk '{ print $1 }'
  Exit code: 1
  1

Output file is the same as the input draft file. This should fail.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=${in_dir}/calls_to_draft.bam
  > in_draft=${in_dir}/draft.fasta.gz
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > ${DORADO_BIN} polish --device cpu --window-len 10000 --window-overlap 1000 --bam-chunk 1000000 ${in_bam} ${in_draft} -t 4 ${model_var} --out-path ${in_draft} 2> out/stderr
  > echo "Exit code: $?"
  > grep "Output path matches one of the input paths" out/stderr | wc -l | awk '{ print $1 }'
  Exit code: 1
  1

Output file is the same as the input BAM file. This should fail.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=${in_dir}/calls_to_draft.bam
  > in_draft=${in_dir}/draft.fasta.gz
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > ${DORADO_BIN} polish --device cpu --window-len 10000 --window-overlap 1000 --bam-chunk 1000000 ${in_bam} ${in_draft} -t 4 ${model_var} --out-path ${in_bam} 2> out/stderr
  > echo "Exit code: $?"
  > grep "Output path matches one of the input paths" out/stderr | wc -l | awk '{ print $1 }'
  Exit code: 1
  1

FASTQ output.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=${in_dir}/calls_to_draft.bam
  > in_draft=${in_dir}/draft.fasta.gz
  > expected=${in_dir}/medaka.consensus.fastq.gz
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > ${DORADO_BIN} polish --device cpu --window-len 10000 --window-overlap 1000 --bam-chunk 1000000 ${in_bam} ${in_draft} -t 4 ${model_var} --qualities > out/out.fastq 2> out/stderr
  > echo "Exit code: $?"
  > gunzip -d --stdout ${expected} > out/expected.fastq
  > wc -l out/out.fastq | awk '{ print $1 }'
  > diff out/expected.fastq out/out.fastq
  > grep "Copying contig verbatim from input" out/stderr | wc -l | awk '{ print $1 }'
  Exit code: 0
  4
  0
