Process the input in multiple overlapping windows. Compare with reference via alignment, but also make sure that the output length matches.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=${in_dir}/calls_to_draft.bam
  > in_draft=${in_dir}/draft.fasta.gz
  > expected=${in_dir}/ref.fasta.gz
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > ${DORADO_BIN} polish --device cpu --window-len 1000 --window-overlap 100 --bam-chunk 1000000 ${in_bam} ${in_draft} -t 4 ${model_var} > out/out.fasta 2> out/stderr
  > echo "Exit code: $?"
  > ${DORADO_BIN} aligner ${expected} out/out.fasta 1> out/out.sam 2> out/out.sam.stderr
  > samtools view out/out.sam | sed -E 's/^.*(NM:i:[0-9]+).*/\1/g'
  > samtools faidx out/out.fasta
  > cut -f 2,2 out/out.fasta.fai
  > grep "Copying contig verbatim from input" out/stderr | wc -l | awk '{ print $1 }'
  Exit code: 0
  NM:i:2
  9998
  0
