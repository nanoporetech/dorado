Process the input in multiple overlapping windows. Compare with reference via alignment, but also make sure that the output length matches.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=${in_dir}/calls_to_draft.bam
  > in_draft=${in_dir}/draft.fasta.gz
  > expected=${in_dir}/ref.fasta.gz
  > model_var=${MODEL_DIR:+--model-path ${MODEL_DIR}}
  > ${DORADO_BIN} polish --window-len 1000 --window-overlap 100 --bam-chunk 1000000 ${in_bam} ${in_draft} out/out.fasta -t 4 ${model_var} 2> out/stderr
  > ${DORADO_BIN} aligner ${expected} out/out.fasta 1> out/out.sam 2> out/out.sam.stderr
  > samtools view out/out.sam | sed -E 's/^.*(NM:i:[0-9]+).*/\1/g'
  > samtools faidx out/out.fasta
  > cut -f 2,2 out/out.fasta.fai
  NM:i:0
  10012
