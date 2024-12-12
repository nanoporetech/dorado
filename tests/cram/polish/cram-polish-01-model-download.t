Negative test: Selected model does not match the model in the BAM.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > model="unknown"
  > ${DORADO_BIN} polish --model "${model}" --device cpu ${in_dir}/calls_to_draft.bam ${in_dir}/draft.fasta.gz -t 4 --infer-threads 1 -vv > out/out.fasta 2> out/out.fasta.stderr
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[error\] //g'
  Exit code: 1
  Caught exception: Specified basecaller model 'unknown' not compatible with the input BAM!

Negative test: Empty model string provided.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > ${DORADO_BIN} polish --model "" --device cpu ${in_dir}/calls_to_draft.bam ${in_dir}/draft.fasta.gz -t 4 --infer-threads 1 -vv > out/out.fasta 2> out/out.fasta.stderr
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[error\] //g'
  Exit code: 1
  Caught exception: Could not resolve model from string: ''.

Negative test: BAM has a model which is not available for download in auto mode.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > model="auto"
  > samtools view -h ${in_dir}/calls_to_draft.bam | sed -E 's/dna_r10.4.1_e8.2_400bps_hac@v5.0.0/dna_r10.4.1_e8.2_400bps_hac@v1.0.0/g' | samtools view -Sb > out/in.bam
  > samtools index out/in.bam
  > ${DORADO_BIN} polish --model "${model}" --device cpu out/in.bam ${in_dir}/draft.fasta.gz -t 4 --infer-threads 1 -vv > out/out.fasta 2> out/out.fasta.stderr
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[error\] //g'
  Exit code: 1
  Selected model doesn't exist: dna_r10.4.1_e8.2_400bps_hac_v1.0.0_polish_rl_mv
  Could not download model: dna_r10.4.1_e8.2_400bps_hac_v1.0.0_polish_rl_mv

Negative test: BAM has no models listed (no RG tags).
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > model="auto"
  > samtools view -h ${in_dir}/calls_to_draft.bam | grep -v "@RG" | samtools view -Sb > out/in.bam
  > samtools index out/in.bam
  > ${DORADO_BIN} polish --model "${model}" --device cpu out/in.bam ${in_dir}/draft.fasta.gz -t 4 --infer-threads 1 -vv > out/out.fasta 2> out/out.fasta.stderr
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[error\] //g'
  Exit code: 1
  Caught exception: Input BAM file has no basecaller models listed in the header.

Auto-resolving the model from BAM (default).
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > ${DORADO_BIN} polish --device cpu ${in_dir}/calls_to_draft.bam ${in_dir}/draft.fasta.gz -t 4 --infer-threads 1 -vv > out/out.fasta 2> out/out.fasta.stderr
  > echo "Exit code: $?"
  > ${DORADO_BIN} aligner ${in_dir}/ref.fasta.gz out/out.fasta 1> out/out.sam 2> out/out.sam.stderr
  > samtools view out/out.sam | sed -E 's/^.*(NM:i:[0-9]+).*/\1/g'
  > samtools faidx out/out.fasta
  > cut -f 2,2 out/out.fasta.fai
  > grep "Copying contig verbatim from input" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "Downloading model" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[error\] //g'
  Exit code: 0
  NM:i:2
  9998
  0
  1

Download a model by name.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > model="dna_r10.4.1_e8.2_400bps_hac@v5.0.0"
  > ${DORADO_BIN} polish --model "${model}" --device cpu ${in_dir}/calls_to_draft.bam ${in_dir}/draft.fasta.gz -t 4 --infer-threads 1 -vv > out/out.fasta 2> out/out.fasta.stderr
  > echo "Exit code: $?"
  > ${DORADO_BIN} aligner ${in_dir}/ref.fasta.gz out/out.fasta 1> out/out.sam 2> out/out.sam.stderr
  > samtools view out/out.sam | sed -E 's/^.*(NM:i:[0-9]+).*/\1/g'
  > samtools faidx out/out.fasta
  > cut -f 2,2 out/out.fasta.fai
  > grep "Copying contig verbatim from input" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "Downloading model" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[error\] //g'
  Exit code: 0
  NM:i:2
  9998
  0
  1

Load a model from path.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > model="${MODEL_DIR}"
  > ${DORADO_BIN} polish --model "${model}" --device cpu ${in_dir}/calls_to_draft.bam ${in_dir}/draft.fasta.gz -t 4 --infer-threads 1 -vv > out/out.fasta 2> out/out.fasta.stderr
  > echo "Exit code: $?"
  > ${DORADO_BIN} aligner ${in_dir}/ref.fasta.gz out/out.fasta 1> out/out.sam 2> out/out.sam.stderr
  > samtools view out/out.sam | sed -E 's/^.*(NM:i:[0-9]+).*/\1/g'
  > samtools faidx out/out.fasta
  > cut -f 2,2 out/out.fasta.fai
  > grep "Copying contig verbatim from input" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "Model specified by path" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[error\] //g'
  Exit code: 0
  NM:i:2
  9998
  0
  1

Download an explicit polishing model by name with `--any-model`.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > model="dna_r10.4.1_e8.2_400bps_hac_v5.0.0_polish_rl_mv"
  > ${DORADO_BIN} polish --any-model --model "${model}" --device cpu ${in_dir}/calls_to_draft.bam ${in_dir}/draft.fasta.gz -t 4 --infer-threads 1 -vv > out/out.fasta 2> out/out.fasta.stderr
  > echo "Exit code: $?"
  > ${DORADO_BIN} aligner ${in_dir}/ref.fasta.gz out/out.fasta 1> out/out.sam 2> out/out.sam.stderr
  > samtools view out/out.sam | sed -E 's/^.*(NM:i:[0-9]+).*/\1/g'
  > samtools faidx out/out.fasta
  > cut -f 2,2 out/out.fasta.fai
  > grep "Copying contig verbatim from input" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "Downloading model" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[error\] //g'
  Exit code: 0
  NM:i:2
  9998
  0
  1

Fail to download model by basecaller model name with `--any-model`.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > model="dna_r10.4.1_e8.2_400bps_hac@v5.0.0"
  > ${DORADO_BIN} polish --any-model --model "${model}" --device cpu ${in_dir}/calls_to_draft.bam ${in_dir}/draft.fasta.gz -t 4 --infer-threads 1 -vv > out/out.fasta 2> out/out.fasta.stderr
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[error\] //g'
  Exit code: 1
  Caught exception: Could not find specified polishing model: 'dna_r10.4.1_e8.2_400bps_hac@v5.0.0'.
