Data construction.
Create a very small synthetic input BAM file of only one record.
Reason: reducing the inference time for successful tests.
  $ rm -rf data; mkdir -p data
  > in_dir="${TEST_DATA_DIR}/polish/test-01-supertiny"
  > in_bam="${in_dir}/calls_to_draft.bam"
  > samtools view -H ${in_bam} > data/in.micro.sam
  > samtools view ${in_bam} | head -n 1 >> data/in.micro.sam
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
  > grep "Using auto-estimated batch-size" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "Using fixed batch-size" out/stderr | sed -E 's/.*\[/\[/g'
  Exit code: 1
  [error] Batch size should be >= 0. Given: -1.

Auto-batch size, GRU model.
  $ rm -rf out; mkdir -p out
  > # Mock the basecaller model to force GRU selection.
  > samtools view -h data/in.micro.bam | sed -E 's/dna_r10.4.1_e8.2_400bps_hac@v5.0.0/dna_r10.4.1_e8.2_400bps_hac@v4.3.0/g' | samtools view -Sb > out/in.bam
  > samtools index out/in.bam
  > # Run the test.
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam="out/in.bam"
  > in_draft=${in_dir}/draft.fasta.gz
  > model_var="--models-directory ${MODEL_ROOT_DIR}"
  > ${DORADO_BIN} polish -vv --batchsize 0 ${model_var} --device cpu --window-len 100 --window-overlap 10 --regions "contig_1:1-99" ${in_bam} ${in_draft} -t 4 > out/out.fasta 2> out/stderr
  > echo "Exit code: $?"
  > grep "\[error\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "Using auto-estimated batch-size" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "Using fixed batch-size" out/stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  [trace] Using auto-estimated batch-size.

Fixed batch size, GRU model.
  $ rm -rf out; mkdir -p out
  > # Mock the basecaller model to force GRU selection.
  > samtools view -h data/in.micro.bam | sed -E 's/dna_r10.4.1_e8.2_400bps_hac@v5.0.0/dna_r10.4.1_e8.2_400bps_hac@v4.3.0/g' | samtools view -Sb > out/in.bam
  > samtools index out/in.bam
  > # Run the test.
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam="out/in.bam"
  > in_draft=${in_dir}/draft.fasta.gz
  > model_var="--models-directory ${MODEL_ROOT_DIR}"
  > ${DORADO_BIN} polish -vv --batchsize 3 ${model_var} --skip-model-compatibility-check --device cpu --window-len 100 --window-overlap 10 --regions "contig_1:1-99" ${in_bam} ${in_draft} -t 4 > out/out.fasta 2> out/stderr
  > echo "Exit code: $?"
  > grep "\[error\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "Using auto-estimated batch-size" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "Using fixed batch-size" out/stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  [trace] Using fixed batch-size of 3.

Auto-batch size, LSTM model.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam="data/in.micro.bam"
  > in_draft=${in_dir}/draft.fasta.gz
  > model_var="--models-directory ${MODEL_ROOT_DIR}"
  > ${DORADO_BIN} polish -vv --batchsize 0 ${model_var} --skip-model-compatibility-check --device cpu --window-len 100 --window-overlap 10 --regions "contig_1:1-99" ${in_bam} ${in_draft} -t 4 > out/out.fasta 2> out/stderr
  > echo "Exit code: $?"
  > grep "\[error\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "Using auto-estimated batch-size" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "Using fixed batch-size" out/stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  [trace] Using auto-estimated batch-size.

Fixed batch size, LSTM model.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam="data/in.micro.bam"
  > in_draft=${in_dir}/draft.fasta.gz
  > model_var="--models-directory ${MODEL_ROOT_DIR}"
  > ${DORADO_BIN} polish -vv --batchsize 3 ${model_var} --skip-model-compatibility-check --device cpu --window-len 100 --window-overlap 10 --regions "contig_1:1-99" ${in_bam} ${in_draft} -t 4 > out/out.fasta 2> out/stderr
  > echo "Exit code: $?"
  > grep "\[error\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "Using auto-estimated batch-size" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "Using fixed batch-size" out/stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  [trace] Using fixed batch-size of 3.
