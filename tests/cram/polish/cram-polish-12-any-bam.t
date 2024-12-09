Test input BAM which was not aligned using Dorado (no "dorado aligner" in the @PG lines).
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=${in_dir}/calls_to_draft.bam
  > in_draft=${in_dir}/draft.fasta.gz
  > expected=${in_dir}/ref.fasta.gz
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > # Create an input BAM with no @PG lines in the header (so no way to identify Dorado).
  > samtools view -H ${in_bam} | grep -v "@PG" > out/in.sam
  > samtools view ${in_bam} >> out/in.sam
  > samtools view -Sb out/in.sam | samtools sort > out/in.bam
  > samtools index out/in.bam
  > # Run.
  > ${DORADO_BIN} polish --device cpu --window-len 1000 --window-overlap 100 --bam-chunk 2000 out/in.bam ${in_draft} -t 4 --infer-threads 2 ${model_var} -vv > out/out.fasta 2> out/stderr
  > echo "Exit code: $?"
  > grep "Input BAM file was not aligned using Dorado" out/stderr | wc -l | awk '{ print $1 }'
  Exit code: 1
  1

Allow any BAM as input.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=${in_dir}/calls_to_draft.bam
  > in_draft=${in_dir}/draft.fasta.gz
  > expected=${in_dir}/ref.fasta.gz
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > # Create an input BAM with no @PG lines in the header (so no way to identify Dorado).
  > samtools view -H ${in_bam} | grep -v "@PG" > out/in.sam
  > samtools view ${in_bam} >> out/in.sam
  > samtools view -Sb out/in.sam | samtools sort > out/in.bam
  > samtools index out/in.bam
  > # Run.
  > ${DORADO_BIN} polish --any-bam --device cpu --window-len 1000 --window-overlap 100 --bam-chunk 2000 out/in.bam ${in_draft} -t 4 --infer-threads 2 ${model_var} -vv > out/out.fasta 2> out/stderr
  > echo "Exit code: $?"
  > # Test.
  > ${DORADO_BIN} aligner ${expected} out/out.fasta 1> out/out.sam 2> out/out.sam.stderr
  > samtools view out/out.sam | sed -E 's/^.*(NM:i:[0-9]+).*/\1/g'
  > samtools faidx out/out.fasta
  > cut -f 2,2 out/out.fasta.fai
  > grep "Copying contig verbatim from input" out/stderr | wc -l | awk '{ print $1 }'
  Exit code: 0
  NM:i:2
  9998
  0
