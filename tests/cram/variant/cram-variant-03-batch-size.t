Data construction.
Create a very small synthetic input BAM file of only one record.
Reason: reducing the inference time for successful tests.
  $ rm -rf data; mkdir -p data
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > in_bam=${in_dir}/in.aln.bam
  > samtools view -H ${in_bam} > data/in.micro.sam
  > samtools view ${in_bam} | head -n 2 >> data/in.micro.sam
  > samtools view -Sb data/in.micro.sam | samtools sort > data/in.micro.bam
  > samtools index data/in.micro.bam

Negative batch size should fail.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > in_bam="data/in.micro.bam"
  > in_ref=${in_dir}/in.ref.fasta.gz
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > ${DORADO_BIN} variant -vv --batchsize -1 --device cpu ${in_bam} ${in_ref} -t 4 ${model_var} --ignore-read-groups --window-len 100 --window-overlap 10 --regions "chr20:1-100" > out/out.vcf 2> out/stderr
  > echo "Exit code: $?"
  > grep "\[error\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "Using auto-estimated batch-size" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "Using fixed batch-size" out/stderr | sed -E 's/.*\[/\[/g'
  Exit code: 1
  [error] Batch size should be > 0. Given: -1.
  [warning] This is an alpha preview of Dorado Variant. Results should be considered experimental.

Zero batch size should fail.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > in_bam="data/in.micro.bam"
  > in_ref=${in_dir}/in.ref.fasta.gz
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > ${DORADO_BIN} variant -vv --batchsize 0 --device cpu ${in_bam} ${in_ref} -t 4 ${model_var} --ignore-read-groups --window-len 100 --window-overlap 10 --regions "chr20:1-100" > out/out.vcf 2> out/stderr
  > echo "Exit code: $?"
  > grep "\[error\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "Using auto-estimated batch-size" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "Using fixed batch-size" out/stderr | sed -E 's/.*\[/\[/g'
  Exit code: 1
  [error] Batch size should be > 0. Given: 0.
  [warning] This is an alpha preview of Dorado Variant. Results should be considered experimental.

Fixed batch size.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > in_bam="data/in.micro.bam"
  > in_ref=${in_dir}/in.ref.fasta.gz
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > ${DORADO_BIN} variant -vv --batchsize 3 --device cpu ${in_bam} ${in_ref} -t 4 ${model_var} --ignore-read-groups --window-len 100 --window-overlap 10 --regions "chr20:1-100" > out/out.vcf 2> out/stderr
  > echo "Exit code: $?"
  > grep "\[error\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "Using auto-estimated batch-size" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "Using fixed batch-size" out/stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  [warning] This is an alpha preview of Dorado Variant. Results should be considered experimental.
  [trace] Using fixed batch-size of 3.

# Auto-batch size.
#   $ rm -rf out; mkdir -p out
#   > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
#   > in_bam="data/in.micro.bam"
#   > in_ref=${in_dir}/in.ref.fasta.gz
#   > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
#   > ${DORADO_BIN} variant -vv --batchsize 0 --device cpu ${in_bam} ${in_ref} -t 4 ${model_var} --ignore-read-groups --window-len 100 --window-overlap 10 --regions "chr20:1-100" > out/out.vcf 2> out/stderr
#   > echo "Exit code: $?"
#   > grep "\[error\]" out/stderr | sed -E 's/.*\[/\[/g'
#   > grep "\[warning\]" out/stderr | sed -E 's/.*\[/\[/g'
#   > grep "Using auto-estimated batch-size" out/stderr | sed -E 's/.*\[/\[/g'
#   > grep "Using fixed batch-size" out/stderr | sed -E 's/.*\[/\[/g'
#   Exit code: 0
#   [warning] This is an alpha preview of Dorado Variant. Results should be considered experimental.
#   [trace] Using auto-estimated batch-size.
