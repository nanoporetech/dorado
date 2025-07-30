Data construction.
Create a very small synthetic input BAM file of only one record.
Reason: reducing the inference time for successful tests.
  $ rm -rf data; mkdir -p data
  > in_dir="${TEST_DATA_DIR}/polish/test-01-supertiny"
  > in_bam="${in_dir}/calls_to_draft.bam"
  > ### Create a BAM file with zero read groups.
  > samtools view -H ${in_bam} > data/in.small.sam
  > samtools view ${in_bam} | head -n 1 >> data/in.small.sam
  > samtools view -Sb data/in.small.sam | samtools sort > data/in.micro.bam
  > samtools index data/in.micro.bam

Write output to a directory defined by the `-o` option.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam="data/in.micro.bam"
  > in_draft=${in_dir}/draft.fasta.gz
  > model_var="--models-directory ${MODEL_ROOT_DIR}"
  > ${DORADO_BIN} polish --device cpu ${in_bam} ${in_draft} -t 4 ${model_var} --regions "contig_1:1-100" -o out 2> out/consensus.fasta.stderr
  > echo "Exit code: $?"
  > samtools faidx out/consensus.fasta
  > awk '{ print $1,$2 }' out/consensus.fasta.fai
  > grep "Copying contig verbatim from input" out/consensus.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/consensus.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/consensus.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  contig_1 9995
  0

Write output to a directory defined by the `--output-dir` option.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam="data/in.micro.bam"
  > in_draft=${in_dir}/draft.fasta.gz
  > expected=${in_dir}/medaka.consensus.fastq.gz
  > model_var="--models-directory ${MODEL_ROOT_DIR}"
  > ${DORADO_BIN} polish --device cpu ${in_bam} ${in_draft} -t 4 ${model_var} --regions "contig_1:1-100" --output-dir out 2> out/consensus.fasta.stderr
  > echo "Exit code: $?"
  > samtools faidx out/consensus.fasta
  > awk '{ print $1,$2 }' out/consensus.fasta.fai
  > grep "Copying contig verbatim from input" out/consensus.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/consensus.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/consensus.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  contig_1 9995
  0

Output directory matches an existing file.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam="data/in.micro.bam"
  > in_draft=${in_dir}/draft.fasta.gz
  > model_var="--models-directory ${MODEL_ROOT_DIR}"
  > touch out/file.txt
  > ${DORADO_BIN} polish --device cpu ${in_bam} ${in_draft} -t 4 ${model_var} -o out/file.txt 2> out/consensus.fasta.stderr
  > echo "Exit code: $?"
  > grep "\[error\]" out/consensus.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/consensus.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 1
  [error] Path specified as output directory exists, but it is not a directory: 'out/file.txt'.

Output file is the same as the input draft file. This should fail.
Although this results in the same error as above, this test safeguards that the output is not identical to an input.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam="data/in.micro.bam"
  > in_draft=${in_dir}/draft.fasta.gz
  > model_var="--models-directory ${MODEL_ROOT_DIR}"
  > cp ${in_draft}* out/
  > ${DORADO_BIN} polish --device cpu ${in_bam} out/draft.fasta.gz -t 4 ${model_var} -o out/draft.fasta.gz 2> out/consensus.fasta.stderr
  > echo "Exit code: $?"
  > grep "\[error\]" out/consensus.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/consensus.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 1
  [error] Path specified as output directory exists, but it is not a directory: 'out/draft.fasta.gz'.

Output file is the same as the input BAM file. This should fail.
Although this results in the same error as above, this test safeguards that the output is not identical to an input.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam="data/in.micro.bam"
  > in_draft=${in_dir}/draft.fasta.gz
  > model_var="--models-directory ${MODEL_ROOT_DIR}"
  > cp ${in_bam}* out/
  > ${DORADO_BIN} polish --device cpu out/in.micro.bam ${in_draft} -t 4 ${model_var} -o out/in.micro.bam 2> out/consensus.fasta.stderr
  > echo "Exit code: $?"
  > grep "\[error\]" out/consensus.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/consensus.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 1
  [error] Path specified as output directory exists, but it is not a directory: 'out/in.micro.bam'.

FASTQ output to stdout.
Checking only the number of lines in the output and the number of bases.
IMPORTANT: not comparing the qualities because they may vary with Torch versions and architectures.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam="data/in.micro.bam"
  > in_draft=${in_dir}/draft.fasta.gz
  > model_var="--models-directory ${MODEL_ROOT_DIR}"
  > ${DORADO_BIN} polish --device cpu ${in_bam} ${in_draft} -t 4 ${model_var} --regions "contig_1:1-100" --qualities > out/out.fastq 2> out/out.fastq.stderr
  > echo "Exit code: $?"
  > wc -l out/out.fastq | awk '{ print $1 }'
  > samtools faidx out/out.fastq
  > awk '{ print $1,$2 }' out/out.fastq.fai
  > grep "Copying contig verbatim from input" out/out.fastq.stderr
  > grep "\[error\]" out/out.fastq.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fastq.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  4
  contig_1 9995

FASTQ output to folder.
Checking only the number of lines in the output and the bases.
IMPORTANT: not comparing the qualities because they may vary with Torch versions and architectures.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam="data/in.micro.bam"
  > in_draft=${in_dir}/draft.fasta.gz
  > expected=${in_dir}/medaka.consensus.fastq.gz
  > model_var="--models-directory ${MODEL_ROOT_DIR}"
  > ${DORADO_BIN} polish --device cpu ${in_bam} ${in_draft} -t 4 ${model_var} --regions "contig_1:1-100" --qualities --output-dir out 2> out/consensus.fastq.stderr
  > echo "Exit code: $?"
  > wc -l out/consensus.fastq | awk '{ print $1 }'
  > samtools faidx out/consensus.fastq
  > awk '{ print $1,$2 }' out/consensus.fastq.fai
  > grep "Copying contig verbatim from input" out/consensus.fastq.stderr
  > grep "\[error\]" out/consensus.fastq.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/consensus.fastq.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  4
  contig_1 9995
