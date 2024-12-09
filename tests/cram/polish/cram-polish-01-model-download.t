Unknown model.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=${in_dir}/calls_to_draft.broken_single.bam
  > in_draft=${in_dir}/draft.fasta.gz
  > model="unknown"
  > ${DORADO_BIN} polish --device cpu --window-len 1000 --window-overlap 100 --bam-chunk 2000 ${in_bam} ${in_draft} -t 4 --infer-threads 1 --model-path "${model}" -vv > out/out.fasta 2> out/stderr
  > echo "Exit code: $?"
  > grep "Selected model doesn't exist" out/stderr | wc -l | awk '{ print $1 }'
  Exit code: 1
  1

Download a model by name.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=${in_dir}/calls_to_draft.bam
  > in_draft=${in_dir}/draft.fasta.gz
  > expected=${in_dir}/ref.fasta.gz
  > model="read_level_lstm384_unidirectional_20241204"
  > ${DORADO_BIN} polish --device cpu --window-len 1000 --window-overlap 100 --bam-chunk 1000000 ${in_bam} ${in_draft} -t 4 -vv --model-path "${model}" > out/out.fasta 2> out/stderr
  > echo "Exit code: $?"
  > ${DORADO_BIN} aligner ${expected} out/out.fasta 1> out/out.sam 2> out/out.sam.stderr
  > samtools view out/out.sam | sed -E 's/^.*(NM:i:[0-9]+).*/\1/g'
  > samtools faidx out/out.fasta
  > cut -f 2,2 out/out.fasta.fai
  > grep "Copying contig verbatim from input" out/stderr | wc -l | awk '{ print $1 }'
  > grep "Downloading model" out/stderr | wc -l | awk '{ print $1 }'
  Exit code: 0
  NM:i:2
  9998
  0
  1

Load a model from a path.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=${in_dir}/calls_to_draft.bam
  > in_draft=${in_dir}/draft.fasta.gz
  > expected=${in_dir}/ref.fasta.gz
  > model_var=${MODEL_DIR:+--model-path ${MODEL_DIR}}
  > ${DORADO_BIN} polish --device cpu --window-len 1000 --window-overlap 100 --bam-chunk 1000000 ${in_bam} ${in_draft} -t 4 ${model_var} -vv > out/out.fasta 2> out/stderr
  > echo "Exit code: $?"
  > ${DORADO_BIN} aligner ${expected} out/out.fasta 1> out/out.sam 2> out/out.sam.stderr
  > samtools view out/out.sam | sed -E 's/^.*(NM:i:[0-9]+).*/\1/g'
  > samtools faidx out/out.fasta
  > cut -f 2,2 out/out.fasta.fai
  > grep "Copying contig verbatim from input" out/stderr | wc -l | awk '{ print $1 }'
  > grep "Using a model specified by path" out/stderr | wc -l | awk '{ print $1 }'
  Exit code: 0
  NM:i:2
  9998
  0
  1
