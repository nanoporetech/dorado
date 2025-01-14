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

Region: only the contig name is provided, polish the entire sequence.
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

Region: only the start coordinate is specified. Clamp to [start, contig_len] interval.
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

Region: only the end coordinate is specified. Clamp to [0, end] interval.
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

Single valid region is specified.
Evaluate using the "--no-gap-fill" to get only the polished region.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > # Run test.
  > ${DORADO_BIN} polish --device cpu ${in_dir}/calls_to_draft.bam ${in_dir}/draft.fasta.gz ${model_var} --no-fill-gaps --regions "contig_1:7000-7200" -vv > out/out.fasta 2> out/out.fasta.stderr
  > # Eval.
  > echo "Exit code: $?"
  > grep ">" out/out.fasta | tr -d ">"
  > samtools faidx out/out.fasta
  > awk '{ print $1,$2 }' out/out.fasta.fai
  Exit code: 0
  contig_1_0 6999-7200
  contig_1_0 201

Multiple valid regions on the same contig.
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

Multiple contigs, single valid region is specified.
Draft batch size is set to 1, so each contig is processed separately.
Evaluate using the "--no-gap-fill" to get only the polished region.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > # Create a synthetic draft with 2
  > gunzip -c ${in_dir}/draft.fasta.gz | sed -E 's/contig_1/contig_2/g' > out/in.draft.fasta
  > gunzip -c ${in_dir}/draft.fasta.gz >> out/in.draft.fasta
  > samtools faidx out/in.draft.fasta
  > # Run test.
  > ${DORADO_BIN} polish  --regions "contig_1:7000-7200" --draft-batchsize 1 --device cpu ${in_dir}/calls_to_draft.bam out/in.draft.fasta ${model_var} --no-fill-gaps -vv > out/out.fasta 2> out/out.fasta.stderr
  > # Eval.
  > echo "Exit code: $?"
  > grep ">" out/out.fasta | tr -d ">"
  > samtools faidx out/out.fasta
  > awk '{ print $1,$2 }' out/out.fasta.fai
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[error\] //g'
  Exit code: 0
  contig_1_0 6999-7200
  contig_1_0 201

Checks that the interval coordinates internally are represented well. Two neighboring regions, adjacent (0bp apart).
If there was an off-by-one edge case (e.g. non-inclusive end coordinate), this would fail.
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
Checks that the interval coordinates internally are represented well.
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
