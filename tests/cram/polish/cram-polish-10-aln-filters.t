Filter: "--min-mapq".
Test filtering alignments by min mapq. Since all alignments in the test BAM have mapq == 60, we can only try to set it higher than that
and see if everything is filtered out.
If the filter works, then there will be zero alignments usable for pileup features, and the input draft will be copied verbatim to output.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=${in_dir}/calls_to_draft.bam
  > in_draft=${in_dir}/draft.fasta.gz
  > model_var=${MODEL_DIR:+--model-path ${MODEL_DIR}}
  > ${DORADO_BIN} polish --device cpu ${in_bam} ${in_draft} ${model_var} --qualities --min-mapq 100 -vv > out/out.fastq 2> out/stderr
  > echo "Exit code: $?"
  > ${DORADO_BIN} aligner ${in_draft} out/out.fastq 1> out/out.sam 2> out/out.sam.stderr
  > samtools view out/out.sam | sed -E 's/^.*(NM:i:[0-9]+).*/\1/g'
  > samtools faidx out/out.fastq
  > cut -f 2,2 out/out.fastq.fai
  > grep "Copying contig verbatim from input" out/stderr | wc -l | awk '{ print $1 }'
  Exit code: 0
  NM:i:0
  10000
  1

Filter: "--tag-name" and "--tag-value".
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=${in_dir}/calls_to_draft.bam
  > in_draft=${in_dir}/draft.fasta.gz
  > model_var=${MODEL_DIR:+--model-path ${MODEL_DIR}}
  > ${DORADO_BIN} polish --device cpu ${in_bam} ${in_draft} ${model_var} --qualities --tag-name "mx" --tag-value 2 -vv > out/out.fastq 2> out/stderr
  > echo "Exit code: $?"
  > ${DORADO_BIN} aligner ${in_draft} out/out.fastq 1> out/out.sam 2> out/out.sam.stderr
  > samtools view out/out.sam | sed -E 's/^.*(NM:i:[0-9]+).*/\1/g'
  > samtools faidx out/out.fastq
  > cut -f 2,2 out/out.fastq.fai
  > grep "Copying contig verbatim from input" out/stderr | wc -l | awk '{ print $1 }'
  Exit code: 0
  NM:i:28
  9988
  0

Filter: "--tag-name" and "--tag-value".
Bad tag name - it only has one character instead of two.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=${in_dir}/calls_to_draft.bam
  > in_draft=${in_dir}/draft.fasta.gz
  > model_var=${MODEL_DIR:+--model-path ${MODEL_DIR}}
  > ${DORADO_BIN} polish --device cpu ${in_bam} ${in_draft} ${model_var} --qualities --tag-name "m" --tag-value 2 -vv > out/out.fastq 2> out/stderr
  > echo "Exit code: $?"
  > grep "The tag_name is specified, but it needs to contain exactly two characters." out/stderr | wc -l | awk '{ print $1 }'
  Exit code: 1
  1

Filter: no "--tag-keep-missing" and "--tag-value" with non-existent tag.
Without specifying "--tag-keep-missing" and with "--tag-value" pointing to a tag which does not exist, this should filter out all alignments and output the
draft contig verbatim.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=${in_dir}/calls_to_draft.bam
  > in_draft=${in_dir}/draft.fasta.gz
  > model_var=${MODEL_DIR:+--model-path ${MODEL_DIR}}
  > ${DORADO_BIN} polish --device cpu ${in_bam} ${in_draft} ${model_var} --qualities --tag-name "xy" --tag-value 2 -vv > out/out.fastq 2> out/stderr
  > echo "Exit code: $?"
  > grep "Copying contig verbatim from input" out/stderr | wc -l | awk '{ print $1 }'
  Exit code: 0
  1

Filter: with "--tag-keep-missing" and "--tag-value" with non-existent tag.
Test filtering alignments by min mapq. Since all alignments in the test BAM have mapq == 60, we can only try to set it higher than that
and see if everything is filtered out.
If the filter works, then there will be zero alignments usable for pileup features, and the input draft will be copied verbatim to output.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=${in_dir}/calls_to_draft.bam
  > in_draft=${in_dir}/draft.fasta.gz
  > expected=${in_dir}/ref.fasta.gz
  > model_var=${MODEL_DIR:+--model-path ${MODEL_DIR}}
  > ${DORADO_BIN} polish --device cpu ${in_bam} ${in_draft} ${model_var} --qualities --tag-keep-missing -vv > out/out.fastq 2> out/stderr
  > echo "Exit code: $?"
  > ${DORADO_BIN} aligner ${expected} out/out.fastq 1> out/out.sam 2> out/out.sam.stderr
  > samtools view out/out.sam | sed -E 's/^.*(NM:i:[0-9]+).*/\1/g'
  > samtools faidx out/out.fastq
  > cut -f 2,2 out/out.fastq.fai
  > grep "Copying contig verbatim from input" out/stderr | wc -l | awk '{ print $1 }'
  Exit code: 0
  NM:i:2
  9998
  0

Filter: "--RG" with a non-existent read group specified, should ignore all alignments.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=${in_dir}/calls_to_draft.bam
  > in_draft=${in_dir}/draft.fasta.gz
  > model_var=${MODEL_DIR:+--model-path ${MODEL_DIR}}
  > ${DORADO_BIN} polish --device cpu ${in_bam} ${in_draft} ${model_var} --qualities --RG "some-read-group" -vv > out/out.fastq 2> out/stderr
  > echo "Exit code: $?"
  > grep "Copying contig verbatim from input" out/stderr | wc -l | awk '{ print $1 }'
  Exit code: 0
  1

# Filter: "--RG" with a valid read group specified.
#   $ rm -rf out; mkdir -p out
#   > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
#   > in_bam=${in_dir}/calls_to_draft.bam
#   > in_draft=${in_dir}/draft.fasta.gz
#   > model_var=${MODEL_DIR:+--model-path ${MODEL_DIR}}
#   > ${DORADO_BIN} polish --device cpu ${in_bam} ${in_draft} ${model_var} --qualities --RG "bc8993f4557dd53bf0cbda5fd68453fea5e94485_dna_r10.4.1_e8.2_400bps_hac@v5.0.0-3AD2AF3E" -vv > out/out.fastq 2> out/stderr
#   > echo "Exit code: $?"
#   > ${DORADO_BIN} aligner ${expected} out/out.fastq 1> out/out.sam 2> out/out.sam.stderr
#   > samtools view out/out.sam | sed -E 's/^.*(NM:i:[0-9]+).*/\1/g'
#   > samtools faidx out/out.fastq
#   > cut -f 2,2 out/out.fastq.fai
#   > grep "Copying contig verbatim from input" out/stderr | wc -l | awk '{ print $1 }'
#   Exit code: 0
#   NM:i:2
#   9998
#   0
