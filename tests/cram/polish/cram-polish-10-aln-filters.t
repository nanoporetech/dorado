Filter: "--min-mapq".
Test filtering alignments by min mapq. Since all alignments in the test BAM have mapq == 60, we can only try to set it higher than that
and see if everything is filtered out.
If the filter works, then there will be zero alignments usable for pileup features, and the input draft will be copied verbatim to output.
Polish only region for speed.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=${in_dir}/calls_to_draft.bam
  > in_draft=${in_dir}/draft.fasta.gz
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > ${DORADO_BIN} polish --device cpu ${in_bam} ${in_draft} ${model_var} --regions "contig_1:1-100" --qualities --min-mapq 100 -vv > out/out.fastq 2> out/stderr
  > echo "Exit code: $?"
  > grep "\[error\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "Copying contig verbatim from input" out/stderr | sed -E 's/.*\[/\[/g'
  > samtools fqidx out/out.fastq
  > awk '{ print $1,$2 }' out/out.fastq.fai
  Exit code: 0
  [debug] Sequence 'contig_1' of length 10000 has zero inferred samples. Copying contig verbatim from input.
  contig_1 10000

Filter: "--tag-name" and "--tag-value".
Polish only region for speed.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=${in_dir}/calls_to_draft.bam
  > in_draft=${in_dir}/draft.fasta.gz
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > ${DORADO_BIN} polish --device cpu ${in_bam} ${in_draft} ${model_var} --regions "contig_1:5400-5900" --qualities --tag-name "mx" --tag-value 2 -vv > out/out.fastq 2> out/stderr
  > echo "Exit code: $?"
  > grep "\[error\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "Copying contig verbatim from input" out/stderr | sed -E 's/.*\[/\[/g'
  > samtools fqidx out/out.fastq
  > awk '{ print $1,$2 }' out/out.fastq.fai
  Exit code: 0
  contig_1 9998

Filter: "--tag-name" and "--tag-value".
Bad tag name - it only has one character instead of two.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=${in_dir}/calls_to_draft.bam
  > in_draft=${in_dir}/draft.fasta.gz
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > ${DORADO_BIN} polish --device cpu ${in_bam} ${in_draft} ${model_var} --regions "contig_1:1-100" --qualities --tag-name "m" --tag-value 2 -vv > out/out.fastq 2> out/stderr
  > echo "Exit code: $?"
  > grep "\[error\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "Copying contig verbatim from input" out/stderr | sed -E 's/.*\[/\[/g'
  Exit code: 1
  [error] The tag_name is specified, but it needs to contain exactly two characters. Given: 'm'.

Filter: no "--tag-keep-missing" and "--tag-value" with non-existent tag.
Without specifying "--tag-keep-missing" and with "--tag-value" pointing to a tag which does not exist, this should filter out all alignments and output the
draft contig verbatim.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=${in_dir}/calls_to_draft.bam
  > in_draft=${in_dir}/draft.fasta.gz
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > # Create a small dataset.
  > samtools view -H ${in_bam} > out/in.sam
  > samtools view ${in_bam} | head -n 1 >> out/in.sam
  > samtools view -Sb out/in.sam > out/in.bam
  > samtools index out/in.bam
  > ${DORADO_BIN} polish --device cpu --threads 4 out/in.bam ${in_draft} ${model_var} --regions "contig_1:1-100" --qualities --tag-name "xy" --tag-value 2 -vv > out/out.fastq 2> out/stderr
  > echo "Exit code: $?"
  > grep "\[error\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "Copying contig verbatim from input" out/stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  [debug] Sequence 'contig_1' of length 10000 has zero inferred samples. Copying contig verbatim from input.

Filter: with "--tag-keep-missing" and "--tag-value" with non-existent tag.
For speed, polish only the first 100bp.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=${in_dir}/calls_to_draft.bam
  > in_draft=${in_dir}/draft.fasta.gz
  > expected=${in_dir}/ref.fasta.gz
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > ${DORADO_BIN} polish --device cpu ${in_bam} ${in_draft} ${model_var} --regions "contig_1:5400-5900" --qualities --tag-keep-missing -vv > out/out.fastq 2> out/stderr
  > echo "Exit code: $?"
  > grep "\[error\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "Copying contig verbatim from input" out/stderr | sed -E 's/.*\[/\[/g'
  > samtools fqidx out/out.fastq
  > awk '{ print $1,$2 }' out/out.fastq.fai
  Exit code: 0
  contig_1 9998

Filter `--min-depth` set to 1. No bases are under this limit, everything should be identical as if this were not used.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=${in_dir}/calls_to_draft.bam
  > in_draft=${in_dir}/draft.fasta.gz
  > expected=${in_dir}/ref.fasta.gz
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > ${DORADO_BIN} polish --min-depth 1 --device cpu ${in_bam} ${in_draft} ${model_var} --regions "contig_1:5400-5900" -vv -o out --vcf 2> out/stderr
  > echo "Exit code: $?"
  > grep "\[error\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "Copying contig verbatim from input" out/stderr | sed -E 's/.*\[/\[/g'
  > samtools fqidx out/consensus.fasta
  > awk '{ print $1,$2 }' out/consensus.fasta.fai
  Exit code: 0
  contig_1 9998

Filter `--min-depth` set to 11. Many bases should be filtered out.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=${in_dir}/calls_to_draft.bam
  > in_draft=${in_dir}/draft.fasta.gz
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > ${DORADO_BIN} polish --min-depth 11 --device cpu ${in_bam} ${in_draft} ${model_var} --regions "contig_1:5400-5900" -vv -o out --vcf 2> out/stderr
  > echo "Exit code: $?"
  > grep "\[error\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "Copying contig verbatim from input" out/stderr | sed -E 's/.*\[/\[/g'
  > samtools fqidx out/consensus.fasta
  > awk '{ print $1,$2 }' out/consensus.fasta.fai
  Exit code: 0
  [debug] Sequence 'contig_1' of length 10000 has zero inferred samples. Copying contig verbatim from input.
  contig_1 10000

Filter `--min-depth` set to 11 with `--no-fill-gaps`. This should just output nothing.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=${in_dir}/calls_to_draft.bam
  > in_draft=${in_dir}/draft.fasta.gz
  > expected=${in_dir}/expected.medaka.min_depth_11.no_fillgaps.fasta.gz
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > ${DORADO_BIN} polish --min-depth 11 --no-fill-gaps --device cpu ${in_bam} ${in_draft} ${model_var} --regions "contig_1:5400-5900" -vv -o out --vcf 2> out/stderr
  > echo "Exit code: $?"
  > grep "\[error\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "Copying contig verbatim from input" out/stderr | sed -E 's/.*\[/\[/g'
  > wc -l out/consensus.fasta | awk '{ print $1 }'
  Exit code: 0
  0
