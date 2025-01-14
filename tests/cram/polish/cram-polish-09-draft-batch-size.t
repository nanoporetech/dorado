
Preparing test data here. This block does not actually run tests. The data produced here will be used in the tests below.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=${in_dir}/calls_to_draft.bam
  > in_draft=${in_dir}/draft.fasta.gz
  > expected=${in_dir}/ref.fasta.gz
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > # Produce the draft consisting of 3 sequences.
  > gunzip -c ${in_draft} > out/original.draft.fasta
  > cat out/original.draft.fasta > out/in.draft.fasta
  > cat out/original.draft.fasta | sed -E 's/contig_1/contig_2/g' >> out/in.draft.fasta
  > cat out/original.draft.fasta | sed -E 's/contig_1/contig_3/g' >> out/in.draft.fasta
  > # Produce a BAM for the new contigs.
  > samtools view -H ${in_bam} | awk -v OFS='\t' '{ if ($1 == "@SQ") { a = $0; gsub(/contig_1/, "contig_2", a); b = $0; gsub(/contig_1/, "contig_3", b); print $0; print a; print b; } else { print }  }' > out/in.aln.sam
  > samtools view ${in_bam} >> out/in.aln.sam
  > samtools view ${in_bam} | sed -E 's/contig_1/contig_2/g' >> out/in.aln.sam
  > samtools view ${in_bam} | sed -E 's/contig_1/contig_3/g' >> out/in.aln.sam
  > samtools view -Sb out/in.aln.sam | samtools sort > out/in.aln.bam
  > samtools index out/in.aln.bam

Polish with a very LARGE draft batch size - all sequences should be in a single batch.
There are 3 input sequences, so 3 NM lines and length lines.
  $ ${DORADO_BIN} polish --device cpu --draft-batchsize 200M out/in.aln.bam out/in.draft.fasta -t 1 --infer-threads 1 ${model_var} -vv > out/out.fasta 2> out/stderr
  > echo "Exit code: $?"
  > ${DORADO_BIN} aligner ${expected} out/out.fasta 1> out/out.sam 2> out/out.sam.stderr
  > samtools view out/out.sam | sed -E 's/^.*(NM:i:[0-9]+).*/\1/g'
  > samtools faidx out/out.fasta
  > cut -f 2,2 out/out.fasta.fai
  > grep "Copying contig verbatim from input" out/stderr | wc -l | awk '{ print $1 }'
  > grep "Starting to produce consensus for regions: 0-3/3" out/stderr | wc -l | awk '{ print $1 }'
  Exit code: 0
  NM:i:2
  NM:i:2
  NM:i:2
  9998
  9998
  9998
  0
  1

Polish with a very SMALL draft batch size - each sequence should be in an individual batch.
There are 3 input sequences, so 3 NM lines and length lines.
  $ ${DORADO_BIN} polish --device cpu --draft-batchsize 1 out/in.aln.bam out/in.draft.fasta -t 1 --infer-threads 1 ${model_var} -vv > out/out.fasta 2> out/stderr
  > echo "Exit code: $?"
  > ${DORADO_BIN} aligner ${expected} out/out.fasta 1> out/out.sam 2> out/out.sam.stderr
  > samtools view out/out.sam | sed -E 's/^.*(NM:i:[0-9]+).*/\1/g'
  > samtools faidx out/out.fasta
  > cut -f 2,2 out/out.fasta.fai
  > grep "Copying contig verbatim from input" out/stderr | wc -l | awk '{ print $1 }'
  > grep "Starting to produce consensus for regions: 0-1/3" out/stderr | wc -l | awk '{ print $1 }'
  > grep "Starting to produce consensus for regions: 1-2/3" out/stderr | wc -l | awk '{ print $1 }'
  > grep "Starting to produce consensus for regions: 2-3/3" out/stderr | wc -l | awk '{ print $1 }'
  Exit code: 0
  NM:i:2
  NM:i:2
  NM:i:2
  9998
  9998
  9998
  0
  1
  1
  1

Polish with a draft batch that fits 2 sequences. There will be 2 batches - first one with 2 sequences and second one with 1 sequence.
There are 3 input sequences, so 3 NM lines and length lines.
  $ ${DORADO_BIN} polish --device cpu --draft-batchsize 20k out/in.aln.bam out/in.draft.fasta -t 1 --infer-threads 1 ${model_var} -vv > out/out.fasta 2> out/stderr
  > echo "Exit code: $?"
  > ${DORADO_BIN} aligner ${expected} out/out.fasta 1> out/out.sam 2> out/out.sam.stderr
  > samtools view out/out.sam | sed -E 's/^.*(NM:i:[0-9]+).*/\1/g'
  > samtools faidx out/out.fasta
  > cut -f 2,2 out/out.fasta.fai
  > grep "Copying contig verbatim from input" out/stderr | wc -l | awk '{ print $1 }'
  > grep "Starting to produce consensus for regions: 0-2/3" out/stderr | wc -l | awk '{ print $1 }'
  > grep "Starting to produce consensus for regions: 2-3/3" out/stderr | wc -l | awk '{ print $1 }'
  Exit code: 0
  NM:i:2
  NM:i:2
  NM:i:2
  9998
  9998
  9998
  0
  1
  1

Edge case test - draft batch size < 0.
  $ ${DORADO_BIN} polish --device cpu --draft-batchsize -1 out/in.aln.bam out/in.draft.fasta -t 1 --infer-threads 1 ${model_var} -vv > out/out.fasta 2> out/stderr
  > echo "Exit code: $?"
  > grep "Draft batch size should be > 0" out/stderr | wc -l | awk '{ print $1 }'
  Exit code: 1
  1

Edge case test - draft batch size == 0.
  $ ${DORADO_BIN} polish --device cpu --draft-batchsize 0 out/in.aln.bam out/in.draft.fasta -t 1 --infer-threads 1 ${model_var} -vv > out/out.fasta 2> out/stderr
  > echo "Exit code: $?"
  > grep "Draft batch size should be > 0" out/stderr | wc -l | awk '{ print $1 }'
  Exit code: 1
  1
