
Preparing test data here. This block does not actually run tests. The data produced here will be used in the tests below.
  $ rm -rf data; mkdir -p data
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=${in_dir}/calls_to_draft.bam
  > in_draft=${in_dir}/draft.fasta.gz
  > expected=${in_dir}/ref.fasta.gz
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > # Produce a draft consisting of 3 sequences.
  > gunzip -c ${in_draft} > data/original.draft.fasta
  > cat data/original.draft.fasta > data/in.draft.fasta
  > cat data/original.draft.fasta | sed -E 's/contig_1/contig_2/g' >> data/in.draft.fasta
  > cat data/original.draft.fasta | sed -E 's/contig_1/contig_3/g' >> data/in.draft.fasta
  > # Produce a BAM for the new contigs.
  > samtools view -H ${in_bam} | awk -v OFS='\t' '{ if ($1 == "@SQ") { a = $0; gsub(/contig_1/, "contig_2", a); b = $0; gsub(/contig_1/, "contig_3", b); print $0; print a; print b; } else { print }  }' > data/in.aln.sam
  > samtools view ${in_bam} | head -n 5 | tail -n 1 >> data/in.aln.sam
  > samtools view ${in_bam} | head -n 5 | tail -n 1 | sed -E 's/contig_1/contig_2/g' >> data/in.aln.sam
  > samtools view ${in_bam} | head -n 5 | tail -n 1 | sed -E 's/contig_1/contig_3/g' >> data/in.aln.sam
  > samtools view -Sb data/in.aln.sam | samtools sort > data/in.aln.bam
  > samtools index data/in.aln.bam

Polish with a very LARGE draft batch size - all sequences should be in a single batch.
There are 3 input sequences, so 3 NM lines and length lines.
Use a slightly larger --window-len to fit each input contig data into single samples (for speed reasons only).
(Even though the contig is of length 10000, the tensor might be longer due to insertions.)
  $ rm -rf out; mkdir -p out
  > ${DORADO_BIN} polish --device cpu --draft-batchsize 200M --window-len 11000 data/in.aln.bam data/in.draft.fasta -t 4 --infer-threads 1 ${model_var} -vv > out/out.fasta 2> out/out.fasta.stderr
  > echo "Exit code: $?"
  > samtools faidx out/out.fasta
  > awk '{ print $1,$2 }' out/out.fasta.fai
  > grep "Copying contig verbatim from input" out/out.fasta.stderr
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "Starting to produce consensus for regions: 0-3/3" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  Exit code: 0
  contig_1 10015
  contig_2 10015
  contig_3 10015
  1

Polish with a very SMALL draft batch size - each sequence should be in an individual batch.
There are 3 input sequences, so 3 NM lines and length lines.
Use a slightly larger --window-len to fit each input contig data into single samples (for speed reasons only).
(Even though the contig is of length 10000, the tensor might be longer due to insertions.)
  $ rm -rf out; mkdir -p out
  > ${DORADO_BIN} polish --device cpu --draft-batchsize 1 --window-len 11000 data/in.aln.bam data/in.draft.fasta -t 4 --infer-threads 1 ${model_var} -vv > out/out.fasta 2> out/out.fasta.stderr
  > echo "Exit code: $?"
  > samtools faidx out/out.fasta
  > awk '{ print $1,$2 }' out/out.fasta.fai
  > grep "Copying contig verbatim from input" out/out.fasta.stderr
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "Copying contig verbatim from input" out/out.fasta.stderr
  > grep "Starting to produce consensus for regions: 0-1/3" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "Starting to produce consensus for regions: 1-2/3" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "Starting to produce consensus for regions: 2-3/3" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  Exit code: 0
  contig_1 10015
  contig_2 10015
  contig_3 10015
  1
  1
  1

Polish with a draft batch that fits 2 sequences. There will be 2 batches - first one with 2 sequences and second one with 1 sequence.
There are 3 input sequences, so 3 NM lines and length lines.
  $ rm -rf out; mkdir -p out
  > ${DORADO_BIN} polish --device cpu --draft-batchsize 20k --window-len 11000 data/in.aln.bam data/in.draft.fasta -t 4 --infer-threads 1 ${model_var} -vv > out/out.fasta 2> out/out.fasta.stderr
  > echo "Exit code: $?"
  > samtools faidx out/out.fasta
  > awk '{ print $1,$2 }' out/out.fasta.fai
  > grep "Copying contig verbatim from input" out/out.fasta.stderr
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "Copying contig verbatim from input" out/out.fasta.stderr
  > grep "Starting to produce consensus for regions: 0-2/3" out/out.fasta.stderr  | wc -l | awk '{ print $1 }'
  > grep "Starting to produce consensus for regions: 2-3/3" out/out.fasta.stderr  | wc -l | awk '{ print $1 }'
  Exit code: 0
  contig_1 10015
  contig_2 10015
  contig_3 10015
  1
  1

Edge case test - draft batch size < 0.
  $ rm -rf out; mkdir -p out
  > ${DORADO_BIN} polish --device cpu --draft-batchsize -1 data/in.aln.bam data/in.draft.fasta -t 1 --infer-threads 1 ${model_var} -vv > out/out.fasta 2> out/out.fasta.stderr
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 1
  [error] Draft batch size should be > 0. Given: 0.

Edge case test - draft batch size == 0.
  $ rm -rf out; mkdir -p out
  > ${DORADO_BIN} polish --device cpu --draft-batchsize 0 data/in.aln.bam data/in.draft.fasta -t 1 --infer-threads 1 ${model_var} -vv > out/out.fasta 2> out/out.fasta.stderr
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 1
  [error] Draft batch size should be > 0. Given: 0.
