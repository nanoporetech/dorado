BAM file contains a broken record - SEQ field is shorter than the move tag ("mv:") because a BAM file was realigned, and hard clippings got lost.
This should throw a warning, and the output should be identical to the input draft.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=${in_dir}/calls_to_draft.broken_single.bam
  > in_draft=${in_dir}/draft.fasta.gz
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > ${DORADO_BIN} polish --device cpu --window-len 1000 --window-overlap 100 --bam-chunk 2000 ${in_bam} ${in_draft} -t 4 --infer-threads 2 ${model_var} -vv > out/out.fasta 2> out/stderr
  > echo "Exit code: $?"
  > ${DORADO_BIN} aligner ${in_draft} out/out.fasta 1> out/out.sam 2> out/out.sam.stderr
  > samtools view out/out.sam | sed -E 's/^.*(NM:i:[0-9]+).*/\1/g'
  > samtools faidx out/out.fasta
  > cut -f 2,2 out/out.fasta.fai
  > grep "Bad BAM alignment" out/stderr | wc -l | awk '{ print $1 }'
  Exit code: 0
  NM:i:0
  10000
  1
