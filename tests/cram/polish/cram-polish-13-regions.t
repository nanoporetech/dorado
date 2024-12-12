Region string is empty. This should fail.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > # Run test.
  > ${DORADO_BIN} polish --device cpu ${in_dir}/calls_to_draft.bam ${in_dir}/draft.fasta.gz ${model_var} --no-fill-gaps --regions "" -vv > out/out.fasta 2> out/out.fasta.stderr
  > # Eval.
  > echo "Exit code: $?"
  > grep "Option --regions is specified, but an empty set of regions is given" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  Exit code: 1
  1

Region: full length of the input chromosome.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > # Run test.
  > ${DORADO_BIN} polish --device cpu ${in_dir}/calls_to_draft.bam ${in_dir}/draft.fasta.gz ${model_var} --qualities --regions "contig_1:1-10000" -vv > out/out.fastq 2> out/out.fastq.stderr
  > # Eval.
  > echo "Exit code: $?"
  > ${DORADO_BIN} aligner ${in_dir}/ref.fasta.gz out/out.fastq 1> out/out.sam 2> out/out.sam.stderr
  > samtools view out/out.sam | sed -E 's/^.*(NM:i:[0-9]+).*/\1/g'
  > samtools faidx out/out.fastq
  > cut -f 2,2 out/out.fastq.fai
  > grep "Copying contig verbatim from input" out/out.fastq.stderr | wc -l | awk '{ print $1 }'
  Exit code: 0
  NM:i:2
  9998
  0

Region: only contig name is provided, polish the entire sequence.
Evaluate using the "--no-gap-fill" to get only the polished region.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > # Run test.
  > ${DORADO_BIN} polish --device cpu ${in_dir}/calls_to_draft.bam ${in_dir}/draft.fasta.gz ${model_var} --no-fill-gaps --regions "contig_1" -vv > out/out.fasta 2> out/out.fasta.stderr
  > # Eval.
  > echo "Exit code: $?"
  > grep ">" out/out.fasta | tr -d ">"
  > samtools faidx out/out.fasta
  > awk '{ print $1,$2 }' out/out.fasta.fai
  Exit code: 0
  contig_1_0 0-10000
  contig_1_0 9998

Unknown contig name is specified.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > # Run test.
  > ${DORADO_BIN} polish --device cpu ${in_dir}/calls_to_draft.bam ${in_dir}/draft.fasta.gz ${model_var} --no-fill-gaps --regions "nonexistent_contig" -vv > out/out.fasta 2> out/out.fasta.stderr
  > # Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[error\] //g'
  Exit code: 1
  Caught exception: Region validation failed: sequence name for region 'nonexistent_contig:0--1' does not exist in the input sequence file.

Failing region: input region ends after contig length.
Evaluate using the "--no-gap-fill" to get only the polished region.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > # Run test.
  > ${DORADO_BIN} polish --device cpu ${in_dir}/calls_to_draft.bam ${in_dir}/draft.fasta.gz ${model_var} --no-fill-gaps --regions "contig_1:1-999999" -vv > out/out.fasta 2> out/out.fasta.stderr
  > # Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[error\] //g'
  Exit code: 1
  Caught exception: Region validation failed: coordinates for region 'contig_1:1-999999' are not valid. Sequence length: 10000

Region: only start coordinate is specified. Clamp to [start, contig_len] interval.
Evaluate using the "--no-gap-fill" to get only the polished region.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > # Run test.
  > ${DORADO_BIN} polish --device cpu ${in_dir}/calls_to_draft.bam ${in_dir}/draft.fasta.gz ${model_var} --no-fill-gaps --regions "contig_1:500" -vv > out/out.fasta 2> out/out.fasta.stderr
  > # Eval.
  > echo "Exit code: $?"
  > grep ">" out/out.fasta | tr -d ">"
  > samtools faidx out/out.fasta
  > awk '{ print $1,$2 }' out/out.fasta.fai
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[error\] //g'
  Exit code: 0
  contig_1_0 499-10000
  contig_1_0 9499

Region: only end coordinate is specified. Clamp to [0, end] interval.
Evaluate using the "--no-gap-fill" to get only the polished region.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > # Run test.
  > ${DORADO_BIN} polish --device cpu ${in_dir}/calls_to_draft.bam ${in_dir}/draft.fasta.gz ${model_var} --no-fill-gaps --regions "contig_1:-500" -vv > out/out.fasta 2> out/out.fasta.stderr
  > # Eval.
  > echo "Exit code: $?"
  > grep ">" out/out.fasta | tr -d ">"
  > samtools faidx out/out.fasta
  > awk '{ print $1,$2 }' out/out.fasta.fai
  Exit code: 0
  contig_1_0 0-500
  contig_1_0 500

Region completely out of coordinate range of the target.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > # Run test.
  > ${DORADO_BIN} polish --device cpu ${in_dir}/calls_to_draft.bam ${in_dir}/draft.fasta.gz ${model_var} --no-fill-gaps --regions "contig_1:50000-55000" -vv > out/out.fasta 2> out/out.fasta.stderr
  > # Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[error\] //g'
  Exit code: 1
  Caught exception: Region validation failed: coordinates for region 'contig_1:50000-55000' are not valid. Sequence length: 10000

Multiple regions.
Evaluate using the "--no-gap-fill" to get only the polished region.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > # Run test.
  > ${DORADO_BIN} polish --device cpu ${in_dir}/calls_to_draft.bam ${in_dir}/draft.fasta.gz ${model_var} --no-fill-gaps --regions "contig_1:1-500,contig_1:7000-7200,contig_1:8123-8124" -vv > out/out.fasta 2> out/out.fasta.stderr
  > # Eval.
  > echo "Exit code: $?"
  > grep ">" out/out.fasta | tr -d ">"
  > samtools faidx out/out.fasta
  > awk '{ print $1,$2 }' out/out.fasta.fai
  Exit code: 0
  contig_1_0 0-500
  contig_1_1 6999-7200
  contig_1_2 8122-8124
  contig_1_0 500
  contig_1_1 201
  contig_1_2 2

Checks that the interval coordinates internally are represented well
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > # Run test.
  > ${DORADO_BIN} polish --device cpu ${in_dir}/calls_to_draft.bam ${in_dir}/draft.fasta.gz ${model_var} --no-fill-gaps --regions "contig_1:1-500,contig_1:501-7200" -vv > out/out.fasta 2> out/out.fasta.stderr
  > # Eval.
  > echo "Exit code: $?"
  > grep ">" out/out.fasta | tr -d ">"
  > samtools faidx out/out.fasta
  > awk '{ print $1,$2 }' out/out.fasta.fai
  > cp -r out /Users/ivan.sovic/work/gitlab/dorado/temp/
  Exit code: 0
  contig_1_0 0-7200
  contig_1_0 7198

Overlapping regions.
Fails because region overlap is not allowed.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > # Run test.
  > ${DORADO_BIN} polish --device cpu ${in_dir}/calls_to_draft.bam ${in_dir}/draft.fasta.gz ${model_var} --no-fill-gaps --regions "contig_1:1-500,contig_1:501-7200,contig_1:7001-7100,contig_1:8123-8124" -vv > out/out.fasta 2> out/out.fasta.stderr
  > # Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[error\] //g'
  Exit code: 1
  Caught exception: Region validation failed: region 'contig_1:501-7200' overlaps other regions. Regions have to be unique.

Zero length region.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > # Run test.
  > ${DORADO_BIN} polish --device cpu ${in_dir}/calls_to_draft.bam ${in_dir}/draft.fasta.gz ${model_var} --no-fill-gaps --regions "contig_1:1-500,contig_1:502-501,contig_1:7001-7100,contig_1:8123-8124" -vv > out/out.fasta 2> out/out.fasta.stderr
  > # Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[error\] //g'
  Exit code: 1
  Caught exception: Region validation failed: coordinates for region 'contig_1:502-501' are not valid. Sequence length: 10000

Wrong region coordinates.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > # Run test.
  > ${DORADO_BIN} polish --device cpu ${in_dir}/calls_to_draft.bam ${in_dir}/draft.fasta.gz ${model_var} --no-fill-gaps --regions "contig_1:700-500" -vv > out/out.fasta 2> out/out.fasta.stderr
  > # Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[error\] //g'
  Exit code: 1
  Caught exception: Region validation failed: coordinates for region 'contig_1:700-500' are not valid. Sequence length: 10000

Loading from a BED file with multiple regions.
Evaluate using the "--no-gap-fill" to get only the polished region.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > printf 'contig_1\t0\t500\n' > out/in.bed
  > printf 'contig_1\t6999\t7200\n' >> out/in.bed
  > printf 'contig_1\t8122\t8124\n' >> out/in.bed
  > # Run test.
  > ${DORADO_BIN} polish --device cpu ${in_dir}/calls_to_draft.bam ${in_dir}/draft.fasta.gz ${model_var} --no-fill-gaps --regions out/in.bed -vv > out/out.fasta 2> out/out.fasta.stderr
  > # Eval.
  > echo "Exit code: $?"
  > grep ">" out/out.fasta | tr -d ">"
  > samtools faidx out/out.fasta
  > awk '{ print $1,$2 }' out/out.fasta.fai
  Exit code: 0
  contig_1_0 0-500
  contig_1_1 6999-7200
  contig_1_2 8122-8124
  contig_1_0 500
  contig_1_1 201
  contig_1_2 2

Loading from an empty BED file.
Evaluate using the "--no-gap-fill" to get only the polished region.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > touch out/in.bed
  > # Run test.
  > ${DORADO_BIN} polish --device cpu ${in_dir}/calls_to_draft.bam ${in_dir}/draft.fasta.gz ${model_var} --no-fill-gaps --regions out/in.bed -vv > out/out.fasta 2> out/out.fasta.stderr
  > # Eval.
  > echo "Exit code: $?"
  > grep "Option --regions is specified, but an empty set of regions is given" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  Exit code: 1
  1

Loading from a BED file with adjacent regions but with no overlap.
Checks that the interval coordinates internally are represented well
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > printf 'contig_1\t0\t500\n' > out/in.bed
  > printf 'contig_1\t500\t7200\n' >> out/in.bed
  > # Run test.
  > ${DORADO_BIN} polish --device cpu ${in_dir}/calls_to_draft.bam ${in_dir}/draft.fasta.gz ${model_var} --no-fill-gaps --regions out/in.bed -vv > out/out.fasta 2> out/out.fasta.stderr
  > # Eval.
  > echo "Exit code: $?"
  > grep ">" out/out.fasta | tr -d ">"
  > samtools faidx out/out.fasta
  > awk '{ print $1,$2 }' out/out.fasta.fai
  > cp -r out /Users/ivan.sovic/work/gitlab/dorado/temp/
  Exit code: 0
  contig_1_0 0-7200
  contig_1_0 7198

Loading from a BED file with overlapping regions.
Fails because region overlap is not allowed.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > printf 'contig_1\t0\t500\n' > out/in.bed
  > printf 'contig_1\t500\t7200\n' >> out/in.bed
  > printf 'contig_1\t7000\t7100\n' >> out/in.bed
  > printf 'contig_1\t8122\t8124\n' >> out/in.bed
  > # Run test.
  > ${DORADO_BIN} polish --device cpu ${in_dir}/calls_to_draft.bam ${in_dir}/draft.fasta.gz ${model_var} --no-fill-gaps --regions out/in.bed -vv > out/out.fasta 2> out/out.fasta.stderr
  > # Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[error\] //g'
  Exit code: 1
  Caught exception: Region validation failed: region 'contig_1:501-7200' overlaps other regions. Regions have to be unique.

# Regions for testing:
# + Region longer than target
# + Region with negative coords (not possible)
# + Region completely out of coordinate range of the target.
# + Multiple regions
# + BED file with regions.
# + Partially defined region. (E.g. missing end coordinate, or chromosome, etc.)
# + Region for a target which is not present in the input.
# + Empty region string provided to `--region ""`.
# - Overlapping regions - merge?
# - Duplicate regions
# - Are these regions set as BAM regions or as chromosomal regions?
