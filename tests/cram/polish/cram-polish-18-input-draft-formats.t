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

Run a tiny test case using the default parameters.
Input is .fasta.gz.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam="data/in.micro.bam"
  > in_draft=${in_dir}/draft.fasta.gz
  > expected=${in_dir}/medaka.consensus.fastq.gz
  > model_var="--models-directory ${MODEL_ROOT_DIR}"
  > ${DORADO_BIN} polish --device cpu ${in_bam} ${in_draft} -t 4 --regions "contig_1:1-100" ${model_var} > out/out.fasta 2> out/out.fasta.stderr
  > echo "Exit code: $?"
  > samtools faidx out/out.fasta
  > awk '{ print $1,$2 }' out/out.fasta.fai
  > grep "Copying contig verbatim from input" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  contig_1 9995
  0

Same tiny test but use a non-bgzipped .fasta reference.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > gunzip -c ${in_dir}/draft.fasta.gz > out/draft.fasta
  > samtools faidx out/draft.fasta
  > in_bam="data/in.micro.bam"
  > in_draft=out/draft.fasta
  > expected=${in_dir}/medaka.consensus.fastq.gz
  > model_var="--models-directory ${MODEL_ROOT_DIR}"
  > ${DORADO_BIN} polish --device cpu ${in_bam} ${in_draft} -t 4 --regions "contig_1:1-100" ${model_var} > out/out.fasta 2> out/out.fasta.stderr
  > echo "Exit code: $?"
  > samtools faidx out/out.fasta
  > awk '{ print $1,$2 }' out/out.fasta.fai
  > grep "Copying contig verbatim from input" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  contig_1 9995
  0

Same tiny test but use a non-bgzipped .fa reference.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > gunzip -c ${in_dir}/draft.fasta.gz > out/draft.fa
  > samtools faidx out/draft.fa
  > in_bam="data/in.micro.bam"
  > in_draft=out/draft.fa
  > expected=${in_dir}/medaka.consensus.fastq.gz
  > model_var="--models-directory ${MODEL_ROOT_DIR}"
  > ${DORADO_BIN} polish --device cpu ${in_bam} ${in_draft} -t 4 --regions "contig_1:1-100" ${model_var} > out/out.fasta 2> out/out.fasta.stderr
  > echo "Exit code: $?"
  > samtools faidx out/out.fasta
  > awk '{ print $1,$2 }' out/out.fasta.fai
  > grep "Copying contig verbatim from input" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  contig_1 9995
  0

Same tiny test but use a .fa.gz reference.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > cp ${in_dir}/draft.fasta.gz out/draft.fa.gz
  > samtools faidx out/draft.fa.gz
  > in_bam="data/in.micro.bam"
  > in_draft=out/draft.fa.gz
  > expected=${in_dir}/medaka.consensus.fastq.gz
  > model_var="--models-directory ${MODEL_ROOT_DIR}"
  > ${DORADO_BIN} polish --device cpu ${in_bam} ${in_draft} -t 4 --regions "contig_1:1-100" ${model_var} > out/out.fasta 2> out/out.fasta.stderr
  > echo "Exit code: $?"
  > samtools faidx out/out.fasta
  > awk '{ print $1,$2 }' out/out.fasta.fai
  > grep "Copying contig verbatim from input" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  contig_1 9995
  0

Same tiny test but use a .fna reference.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > gunzip -c ${in_dir}/draft.fasta.gz > out/draft.fna
  > samtools faidx out/draft.fna
  > in_bam="data/in.micro.bam"
  > in_draft=out/draft.fna
  > expected=${in_dir}/medaka.consensus.fastq.gz
  > model_var="--models-directory ${MODEL_ROOT_DIR}"
  > ${DORADO_BIN} polish --device cpu ${in_bam} ${in_draft} -t 4 --regions "contig_1:1-100" ${model_var} > out/out.fasta 2> out/out.fasta.stderr
  > echo "Exit code: $?"
  > samtools faidx out/out.fasta
  > awk '{ print $1,$2 }' out/out.fasta.fai
  > grep "Copying contig verbatim from input" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  contig_1 9995
  0

Same tiny test but use a .fna.gz reference.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > cp ${in_dir}/draft.fasta.gz out/draft.fna.gz
  > samtools faidx out/draft.fna.gz
  > in_bam="data/in.micro.bam"
  > in_draft=out/draft.fna.gz
  > expected=${in_dir}/medaka.consensus.fastq.gz
  > model_var="--models-directory ${MODEL_ROOT_DIR}"
  > ${DORADO_BIN} polish --device cpu ${in_bam} ${in_draft} -t 4 --regions "contig_1:1-100" ${model_var} > out/out.fasta 2> out/out.fasta.stderr
  > echo "Exit code: $?"
  > samtools faidx out/out.fasta
  > awk '{ print $1,$2 }' out/out.fasta.fai
  > grep "Copying contig verbatim from input" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  contig_1 9995
  0

Input draft is a .fastq file (added dummy qualities to existing test data).
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > ### Create the mocked FASTQ input file.
  > seq_len=$(cat ${in_dir}/draft.fasta.gz.fai | awk '{ print $2 }')
  > samtools faidx "${in_dir}/draft.fasta.gz" "contig_1" --length 100000 | sed 's/>/@/' > out/draft.fastq
  > python3 -c "print('+\n' + 'I'*${seq_len})" >> out/draft.fastq
  > samtools fqidx out/draft.fastq
  > ### Run the test.
  > in_bam="data/in.micro.bam"
  > in_draft=out/draft.fastq
  > expected=${in_dir}/medaka.consensus.fastq.gz
  > model_var="--models-directory ${MODEL_ROOT_DIR}"
  > ${DORADO_BIN} polish --device cpu ${in_bam} ${in_draft} -t 4 --regions "contig_1:1-100" ${model_var} > out/out.fasta 2> out/out.fasta.stderr
  > echo "Exit code: $?"
  > samtools faidx out/out.fasta
  > awk '{ print $1,$2 }' out/out.fasta.fai
  > grep "Copying contig verbatim from input" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  contig_1 9995
  0

Input draft is a .fq file (added dummy qualities to existing test data).
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > ### Create the mocked FASTQ input file.
  > seq_len=$(cat ${in_dir}/draft.fasta.gz.fai | awk '{ print $2 }')
  > samtools faidx "${in_dir}/draft.fasta.gz" "contig_1" --length 100000 | sed 's/>/@/' > out/draft.fq
  > python3 -c "print('+\n' + 'I'*${seq_len})" >> out/draft.fq
  > samtools fqidx out/draft.fq
  > ### Run the test.
  > in_bam="data/in.micro.bam"
  > in_draft=out/draft.fq
  > expected=${in_dir}/medaka.consensus.fastq.gz
  > model_var="--models-directory ${MODEL_ROOT_DIR}"
  > ${DORADO_BIN} polish --device cpu ${in_bam} ${in_draft} -t 4 --regions "contig_1:1-100" ${model_var} > out/out.fasta 2> out/out.fasta.stderr
  > echo "Exit code: $?"
  > samtools faidx out/out.fasta
  > awk '{ print $1,$2 }' out/out.fasta.fai
  > grep "Copying contig verbatim from input" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  contig_1 9995
  0

Input draft is a .fastq.gz file (added dummy qualities to existing test data).
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > ### Create the mocked FASTQ input file.
  > seq_len=$(cat ${in_dir}/draft.fasta.gz.fai | awk '{ print $2 }')
  > samtools faidx "${in_dir}/draft.fasta.gz" "contig_1" --length 100000 | sed 's/>/@/' > out/draft.fastq
  > python3 -c "print('+\n' + 'I'*${seq_len})" >> out/draft.fastq
  > bgzip out/draft.fastq
  > samtools fqidx out/draft.fastq.gz
  > ### Run the test.
  > in_bam="data/in.micro.bam"
  > in_draft=out/draft.fastq.gz
  > expected=${in_dir}/medaka.consensus.fastq.gz
  > model_var="--models-directory ${MODEL_ROOT_DIR}"
  > ${DORADO_BIN} polish --device cpu ${in_bam} ${in_draft} -t 4 --regions "contig_1:1-100" ${model_var} > out/out.fasta 2> out/out.fasta.stderr
  > echo "Exit code: $?"
  > samtools faidx out/out.fasta
  > awk '{ print $1,$2 }' out/out.fasta.fai
  > grep "Copying contig verbatim from input" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  contig_1 9995
  0

Input draft is a .fq.gz file (added dummy qualities to existing test data).
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > ### Create the mocked FASTQ input file.
  > seq_len=$(cat ${in_dir}/draft.fasta.gz.fai | awk '{ print $2 }')
  > samtools faidx "${in_dir}/draft.fasta.gz" "contig_1" --length 100000 | sed 's/>/@/' > out/draft.fq
  > python3 -c "print('+\n' + 'I'*${seq_len})" >> out/draft.fq
  > bgzip out/draft.fq
  > samtools fqidx out/draft.fq.gz
  > ### Run the test.
  > in_bam="data/in.micro.bam"
  > in_draft=out/draft.fq.gz
  > expected=${in_dir}/medaka.consensus.fastq.gz
  > model_var="--models-directory ${MODEL_ROOT_DIR}"
  > ${DORADO_BIN} polish --device cpu ${in_bam} ${in_draft} -t 4 --regions "contig_1:1-100" ${model_var} > out/out.fasta 2> out/out.fasta.stderr
  > echo "Exit code: $?"
  > samtools faidx out/out.fasta
  > awk '{ print $1,$2 }' out/out.fasta.fai
  > grep "Copying contig verbatim from input" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  contig_1 9995
  0
