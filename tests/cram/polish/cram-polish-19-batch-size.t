Data construction.
Create a very small synthetic input BAM file of only one record.
Reason: reducing the inference time for successful tests.
  $ rm -rf data; mkdir -p data
  > in_dir="${TEST_DATA_DIR}/polish/test-01-supertiny"
  > in_bam="${in_dir}/calls_to_draft.bam"
  > samtools view -H ${in_bam} > data/in.micro.sam
  > samtools view ${in_bam} | head -n 2 >> data/in.micro.sam
  > samtools view -Sb data/in.micro.sam | samtools sort > data/in.micro.bam
  > samtools index data/in.micro.bam

Negative batch size should fail.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam="data/in.micro.bam"
  > in_draft=${in_dir}/draft.fasta.gz
  > model_var="--models-directory ${MODEL_ROOT_DIR}"
  > ${DORADO_BIN} polish -vv --batchsize -1 --device cpu ${in_bam} ${in_draft} -t 4 ${model_var} > out/out.fasta 2> out/stderr
  > echo "Exit code: $?"
  > grep "\[error\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "Using fixed batch size" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "Using auto computed batch size." out/stderr | sed -E 's/.*\[/\[/g'
  > grep "Estimating batch memory for fixed batch size" out/stderr | sed -E 's/.*\[/\[/g'
  Exit code: 1
  [error] Batch size should be >= 0. Given: -1.

Fixed positive batch size should run.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam="data/in.micro.bam"
  > in_draft=${in_dir}/draft.fasta.gz
  > model_var="--models-directory ${MODEL_ROOT_DIR}"
  > ${DORADO_BIN} polish -vv --batchsize 1 --device cpu ${in_bam} ${in_draft} -t 4 ${model_var} --window-len 100 --window-overlap 10 --regions "contig_1:1-99" > out/out.fasta 2> out/stderr
  > echo "Exit code: $?"
  > grep "\[error\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "Using fixed batch size" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "Using auto computed batch size." out/stderr | sed -E 's/.*\[/\[/g'
  > grep "Estimating batch memory for fixed batch size" out/stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  [info] Using fixed batch size: 1
  [producer] Estimating batch memory for fixed batch size:

Auto batch size.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam="data/in.micro.bam"
  > in_draft=${in_dir}/draft.fasta.gz
  > model_var="--models-directory ${MODEL_ROOT_DIR}"
  > ${DORADO_BIN} polish -vv --batchsize 0 --device cpu ${model_var} --window-len 100 --window-overlap 10 --regions "contig_1:1-99" ${in_bam} ${in_draft} -t 4 > out/out.fasta 2> out/stderr
  > echo "Exit code: $?"
  > grep "\[error\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "Using fixed batch size" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "Using auto computed batch size." out/stderr | sed -E 's/.*\[/\[/g' | sed -E 's/memory:.*/memory:/g'
  > grep "Estimating batch memory for fixed batch size" out/stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  [info] Using auto computed batch size. Usable per-worker memory:

Batch size zero is auto batch size.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam="data/in.micro.bam"
  > in_draft=${in_dir}/draft.fasta.gz
  > model_var="--models-directory ${MODEL_ROOT_DIR}"
  > ${DORADO_BIN} polish -vv --batchsize 0 --device cpu ${model_var} --window-len 100 --window-overlap 10 --regions "contig_1:1-99" ${in_bam} ${in_draft} -t 4 > out/out.fasta 2> out/stderr
  > echo "Exit code: $?"
  > grep "\[error\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "Using fixed batch size" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "Using auto computed batch size." out/stderr | sed -E 's/.*\[/\[/g' | sed -E 's/memory:.*/memory:/g'
  > grep "Estimating batch memory for fixed batch size" out/stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  [info] Using auto computed batch size. Usable per-worker memory:
