Run a tiny test case using the default parameters.
Input is .fasta.gz.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=${in_dir}/calls_to_draft.bam
  > in_draft=${in_dir}/draft.fasta.gz
  > expected=${in_dir}/medaka.consensus.fastq.gz
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > ${DORADO_BIN} polish --device cpu ${in_bam} ${in_draft} -t 4 ${model_var} > out/out.fasta 2> out/stderr
  > echo "Exit code: $?"
  > gunzip -d --stdout ${expected} | head -n 2 | sed 's/@/>/g' > out/expected.fasta
  > grep "Copying contig verbatim from input" out/stderr | wc -l | awk '{ print $1 }'
  > diff out/expected.fasta out/out.fasta
  Exit code: 0
  0

Same tiny test but use a non-bgzipped .fasta reference.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > gunzip -c ${in_dir}/draft.fasta.gz > out/draft.fasta
  > samtools faidx out/draft.fasta
  > in_bam=${in_dir}/calls_to_draft.bam
  > in_draft=out/draft.fasta
  > expected=${in_dir}/medaka.consensus.fastq.gz
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > ${DORADO_BIN} polish --device cpu ${in_bam} ${in_draft} -t 4 ${model_var} > out/out.fasta 2> out/stderr
  > echo "Exit code: $?"
  > gunzip -d --stdout ${expected} | head -n 2 | sed 's/@/>/g' > out/expected.fasta
  > grep "Copying contig verbatim from input" out/stderr | wc -l | awk '{ print $1 }'
  > diff out/expected.fasta out/out.fasta
  Exit code: 0
  0

Same tiny test but use a non-bgzipped .fa reference.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > gunzip -c ${in_dir}/draft.fasta.gz > out/draft.fa
  > samtools faidx out/draft.fa
  > in_bam=${in_dir}/calls_to_draft.bam
  > in_draft=out/draft.fa
  > expected=${in_dir}/medaka.consensus.fastq.gz
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > ${DORADO_BIN} polish --device cpu ${in_bam} ${in_draft} -t 4 ${model_var} > out/out.fasta 2> out/stderr
  > echo "Exit code: $?"
  > gunzip -d --stdout ${expected} | head -n 2 | sed 's/@/>/g' > out/expected.fasta
  > grep "Copying contig verbatim from input" out/stderr | wc -l | awk '{ print $1 }'
  > diff out/expected.fasta out/out.fasta
  Exit code: 0
  0

Same tiny test but use a .fa.gz reference.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > cp ${in_dir}/draft.fasta.gz out/draft.fa.gz
  > samtools faidx out/draft.fa.gz
  > in_bam=${in_dir}/calls_to_draft.bam
  > in_draft=out/draft.fa.gz
  > expected=${in_dir}/medaka.consensus.fastq.gz
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > ${DORADO_BIN} polish --device cpu ${in_bam} ${in_draft} -t 4 ${model_var} > out/out.fasta 2> out/stderr
  > echo "Exit code: $?"
  > gunzip -d --stdout ${expected} | head -n 2 | sed 's/@/>/g' > out/expected.fasta
  > grep "Copying contig verbatim from input" out/stderr | wc -l | awk '{ print $1 }'
  > diff out/expected.fasta out/out.fasta
  Exit code: 0
  0

Same tiny test but use a .fna reference.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > gunzip -c ${in_dir}/draft.fasta.gz > out/draft.fna
  > samtools faidx out/draft.fna
  > in_bam=${in_dir}/calls_to_draft.bam
  > in_draft=out/draft.fna
  > expected=${in_dir}/medaka.consensus.fastq.gz
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > ${DORADO_BIN} polish --device cpu ${in_bam} ${in_draft} -t 4 ${model_var} > out/out.fasta 2> out/stderr
  > echo "Exit code: $?"
  > gunzip -d --stdout ${expected} | head -n 2 | sed 's/@/>/g' > out/expected.fasta
  > grep "Copying contig verbatim from input" out/stderr | wc -l | awk '{ print $1 }'
  > diff out/expected.fasta out/out.fasta
  Exit code: 0
  0

Same tiny test but use a .fna.gz reference.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > cp ${in_dir}/draft.fasta.gz out/draft.fna.gz
  > samtools faidx out/draft.fna.gz
  > in_bam=${in_dir}/calls_to_draft.bam
  > in_draft=out/draft.fna.gz
  > expected=${in_dir}/medaka.consensus.fastq.gz
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > ${DORADO_BIN} polish --device cpu ${in_bam} ${in_draft} -t 4 ${model_var} > out/out.fasta 2> out/stderr
  > echo "Exit code: $?"
  > gunzip -d --stdout ${expected} | head -n 2 | sed 's/@/>/g' > out/expected.fasta
  > grep "Copying contig verbatim from input" out/stderr | wc -l | awk '{ print $1 }'
  > diff out/expected.fasta out/out.fasta
  Exit code: 0
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
  > in_bam=${in_dir}/calls_to_draft.bam
  > in_draft=out/draft.fastq
  > expected=${in_dir}/medaka.consensus.fastq.gz
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > ${DORADO_BIN} polish --device cpu ${in_bam} ${in_draft} -t 4 ${model_var} > out/out.fasta 2> out/stderr
  > echo "Exit code: $?"
  > gunzip -d --stdout ${expected} | head -n 2 | sed 's/@/>/g' > out/expected.fasta
  > grep "Copying contig verbatim from input" out/stderr | wc -l | awk '{ print $1 }'
  > diff out/expected.fasta out/out.fasta
  Exit code: 0
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
  > in_bam=${in_dir}/calls_to_draft.bam
  > in_draft=out/draft.fq
  > expected=${in_dir}/medaka.consensus.fastq.gz
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > ${DORADO_BIN} polish --device cpu ${in_bam} ${in_draft} -t 4 ${model_var} > out/out.fasta 2> out/stderr
  > echo "Exit code: $?"
  > gunzip -d --stdout ${expected} | head -n 2 | sed 's/@/>/g' > out/expected.fasta
  > grep "Copying contig verbatim from input" out/stderr | wc -l | awk '{ print $1 }'
  > diff out/expected.fasta out/out.fasta
  Exit code: 0
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
  > in_bam=${in_dir}/calls_to_draft.bam
  > in_draft=out/draft.fastq.gz
  > expected=${in_dir}/medaka.consensus.fastq.gz
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > ${DORADO_BIN} polish --device cpu ${in_bam} ${in_draft} -t 4 ${model_var} > out/out.fasta 2> out/stderr
  > echo "Exit code: $?"
  > gunzip -d --stdout ${expected} | head -n 2 | sed 's/@/>/g' > out/expected.fasta
  > grep "Copying contig verbatim from input" out/stderr | wc -l | awk '{ print $1 }'
  > diff out/expected.fasta out/out.fasta
  Exit code: 0
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
  > in_bam=${in_dir}/calls_to_draft.bam
  > in_draft=out/draft.fq.gz
  > expected=${in_dir}/medaka.consensus.fastq.gz
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > ${DORADO_BIN} polish --device cpu ${in_bam} ${in_draft} -t 4 ${model_var} > out/out.fasta 2> out/stderr
  > echo "Exit code: $?"
  > gunzip -d --stdout ${expected} | head -n 2 | sed 's/@/>/g' > out/expected.fasta
  > grep "Copying contig verbatim from input" out/stderr | wc -l | awk '{ print $1 }'
  > diff out/expected.fasta out/out.fasta
  Exit code: 0
  0
