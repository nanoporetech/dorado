Auto-resolving the model from BAM (default).
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > ${DORADO_BIN} polish --device cpu ${in_dir}/calls_to_draft.bam ${in_dir}/draft.fasta.gz -t 4 --infer-threads 1 -vv > out/out.fasta 2> out/stderr
  > echo "Exit code: $?"
  > ${DORADO_BIN} aligner ${in_dir}/ref.fasta.gz out/out.fasta 1> out/out.sam 2> out/out.sam.stderr
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

Download a model by name.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > model="dna_r10.4.1_e8.2_400bps_hac@v5.0.0"
  > ${DORADO_BIN} polish --model "${model}" --device cpu ${in_dir}/calls_to_draft.bam ${in_dir}/draft.fasta.gz -t 4 --infer-threads 1 -vv > out/out.fasta 2> out/stderr
  > echo "Exit code: $?"
  > ${DORADO_BIN} aligner ${in_dir}/ref.fasta.gz out/out.fasta 1> out/out.sam 2> out/out.sam.stderr
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

Load a model from path.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > model="${MODEL_DIR}"
  > ${DORADO_BIN} polish --model "${model}" --device cpu ${in_dir}/calls_to_draft.bam ${in_dir}/draft.fasta.gz -t 4 --infer-threads 1 -vv > out/out.fasta 2> out/stderr
  > echo "Exit code: $?"
  > ${DORADO_BIN} aligner ${in_dir}/ref.fasta.gz out/out.fasta 1> out/out.sam 2> out/out.sam.stderr
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

Selected model does not match the model in the BAM.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > model="unknown"
  > ${DORADO_BIN} polish --model "${model}" --device cpu ${in_dir}/calls_to_draft.bam ${in_dir}/draft.fasta.gz -t 4 --infer-threads 1 -vv > out/out.fasta 2> out/stderr
  > echo "Exit code: $?"
  > grep "Specified basecaller model 'unknown' not compatible with the input BAM!" out/stderr | wc -l | awk '{ print $1 }'
  Exit code: 1
  1

Empty model string provided.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > model="unknown"
  > ${DORADO_BIN} polish --model "${model}" --device cpu ${in_dir}/calls_to_draft.bam ${in_dir}/draft.fasta.gz -t 4 --infer-threads 1 -vv > out/out.fasta 2> out/stderr
  > echo "Exit code: $?"
  > grep "Could not resolve model from string" out/stderr | wc -l | awk '{ print $1 }'
  Exit code: 1
  1

# Negative cases:
#   - BAM has a model which is not available for download in auto mode.
#   + Selected model does not match the model in BAM.
#   + Empty model string provided: --model ""
#   - BAM has no models listed (no RG tags).
#   - Implement and test --any-model
