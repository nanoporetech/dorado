HAC v4.3.0 with dwells. This is a non-read-level legacy model and there is no difference if it has or does not have dwells.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > ### Create synthetic data with mocked model name.
  > samtools view -h ${in_dir}/calls_to_draft.bam | sed 's/dna_r10.4.1_e8.2_400bps_hac@v5.0.0/dna_r10.4.1_e8.2_400bps_hac@v4.3.0/g' | samtools view -Sb > out/in.bam
  > samtools index out/in.bam
  > ### Run the unit under test.
  > ${DORADO_BIN} polish --device cpu out/in.bam ${in_dir}/draft.fasta.gz -t 4 -v > out/out.fasta 2> out/out.fasta.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "Resolved model" out/out.fasta.stderr | sed -E 's/.*\[debug\] //g'
  > grep "Downloading model" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[error\] //g'
  Exit code: 0
  Resolved model from input data: dna_r10.4.1_e8.2_400bps_hac@v4.3.0_polish
  1

Bacterial HAC v4.3.0 with dwells. Resolves to a compatible bacterial model.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > ### Create synthetic data with mocked model name.
  > samtools view -h ${in_dir}/calls_to_draft.bam | sed 's/dna_r10.4.1_e8.2_400bps_hac@v5.0.0/dna_r10.4.1_e8.2_400bps_hac@v4.3.0/g' | samtools view -Sb > out/in.bam
  > samtools index out/in.bam
  > ### Run the unit under test.
  > ${DORADO_BIN} polish --bacteria --device cpu out/in.bam ${in_dir}/draft.fasta.gz -t 4 -v > out/out.fasta 2> out/out.fasta.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "Resolved model" out/out.fasta.stderr | sed -E 's/.*\[debug\] //g'
  > grep "Downloading model" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[error\] //g'
  Exit code: 0
  Resolved model from input data: dna_r10.4.1_e8.2_400bps_polish_bacterial_methylation_v5.0.0
  1

SUP v4.3.0 with dwells. This is a non-read-level legacy model and there is no difference if it has or does not have dwells.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > ### Create synthetic data with mocked model name.
  > samtools view -h ${in_dir}/calls_to_draft.bam | sed 's/dna_r10.4.1_e8.2_400bps_hac@v5.0.0/dna_r10.4.1_e8.2_400bps_sup@v4.3.0/g' | samtools view -Sb > out/in.bam
  > samtools index out/in.bam
  > ### Run the unit under test.
  > ${DORADO_BIN} polish --device cpu out/in.bam ${in_dir}/draft.fasta.gz -t 4 -v > out/out.fasta 2> out/out.fasta.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "Resolved model" out/out.fasta.stderr | sed -E 's/.*\[debug\] //g'
  > grep "Downloading model" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[error\] //g'
  Exit code: 0
  Resolved model from input data: dna_r10.4.1_e8.2_400bps_sup@v4.3.0_polish
  1

Bacterial SUP v4.3.0 with dwells. Resolves to a compatible bacterial model.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > ### Create synthetic data with mocked model name.
  > samtools view -h ${in_dir}/calls_to_draft.bam | sed 's/dna_r10.4.1_e8.2_400bps_hac@v5.0.0/dna_r10.4.1_e8.2_400bps_sup@v4.3.0/g' | samtools view -Sb > out/in.bam
  > samtools index out/in.bam
  > ### Run the unit under test.
  > ${DORADO_BIN} polish --bacteria --device cpu out/in.bam ${in_dir}/draft.fasta.gz -t 4 -v > out/out.fasta 2> out/out.fasta.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "Resolved model" out/out.fasta.stderr | sed -E 's/.*\[debug\] //g'
  > grep "Downloading model" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[error\] //g'
  Exit code: 0
  Resolved model from input data: dna_r10.4.1_e8.2_400bps_polish_bacterial_methylation_v5.0.0
  1

HAC v4.3.0 with no dwells. This is a non-read-level legacy model and there is no difference if it has or does not have dwells.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > ### Create synthetic data with mocked model name.
  > samtools view -H ${in_dir}/calls_to_draft.bam | sed 's/dna_r10.4.1_e8.2_400bps_hac@v5.0.0/dna_r10.4.1_e8.2_400bps_hac@v4.3.0/g' > out/in.sam
  > samtools view ${in_dir}/calls_to_draft.bam | cut -f1-11 >> out/in.sam
  > samtools view -Sb out/in.sam > out/in.bam
  > samtools index out/in.bam
  > ### Run the unit under test.
  > ${DORADO_BIN} polish --device cpu out/in.bam ${in_dir}/draft.fasta.gz -t 4 -v > out/out.fasta 2> out/out.fasta.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "Resolved model" out/out.fasta.stderr | sed -E 's/.*\[debug\] //g'
  > grep "Downloading model" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[error\] //g'
  Exit code: 0
  Resolved model from input data: dna_r10.4.1_e8.2_400bps_hac@v4.3.0_polish
  1

Bacterial HAC v4.3.0 with no dwells. Resolves to a compatible bacterial model.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > ### Create synthetic data with mocked model name.
  > samtools view -H ${in_dir}/calls_to_draft.bam | sed 's/dna_r10.4.1_e8.2_400bps_hac@v5.0.0/dna_r10.4.1_e8.2_400bps_hac@v4.3.0/g' > out/in.sam
  > samtools view ${in_dir}/calls_to_draft.bam | cut -f1-11 >> out/in.sam
  > samtools view -Sb out/in.sam > out/in.bam
  > samtools index out/in.bam
  > ### Run the unit under test.
  > ${DORADO_BIN} polish --bacteria --device cpu out/in.bam ${in_dir}/draft.fasta.gz -t 4 -v > out/out.fasta 2> out/out.fasta.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "Resolved model" out/out.fasta.stderr | sed -E 's/.*\[debug\] //g'
  > grep "Downloading model" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[error\] //g'
  Exit code: 0
  Resolved model from input data: dna_r10.4.1_e8.2_400bps_polish_bacterial_methylation_v5.0.0
  1

SUP v4.3.0 with no dwells. This is a non-read-level legacy model and there is no difference if it has or does not have dwells.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > ### Create synthetic data with mocked model name.
  > samtools view -H ${in_dir}/calls_to_draft.bam | sed 's/dna_r10.4.1_e8.2_400bps_hac@v5.0.0/dna_r10.4.1_e8.2_400bps_sup@v4.3.0/g' > out/in.sam
  > samtools view ${in_dir}/calls_to_draft.bam | cut -f1-11 >> out/in.sam
  > samtools view -Sb out/in.sam > out/in.bam
  > samtools index out/in.bam
  > ### Run the unit under test.
  > ${DORADO_BIN} polish --device cpu out/in.bam ${in_dir}/draft.fasta.gz -t 4 -v > out/out.fasta 2> out/out.fasta.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "Resolved model" out/out.fasta.stderr | sed -E 's/.*\[debug\] //g'
  > grep "Downloading model" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[error\] //g'
  Exit code: 0
  Resolved model from input data: dna_r10.4.1_e8.2_400bps_sup@v4.3.0_polish
  1

Bacterial SUP v4.3.0 with no dwells. Resolves to a compatible bacterial model.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > ### Create synthetic data with mocked model name.
  > samtools view -H ${in_dir}/calls_to_draft.bam | sed 's/dna_r10.4.1_e8.2_400bps_hac@v5.0.0/dna_r10.4.1_e8.2_400bps_sup@v4.3.0/g' > out/in.sam
  > samtools view ${in_dir}/calls_to_draft.bam | cut -f1-11 >> out/in.sam
  > samtools view -Sb out/in.sam > out/in.bam
  > samtools index out/in.bam
  > ### Run the unit under test.
  > ${DORADO_BIN} polish --bacteria --device cpu out/in.bam ${in_dir}/draft.fasta.gz -t 4 -v > out/out.fasta 2> out/out.fasta.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "Resolved model" out/out.fasta.stderr | sed -E 's/.*\[debug\] //g'
  > grep "Downloading model" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[error\] //g'
  Exit code: 0
  Resolved model from input data: dna_r10.4.1_e8.2_400bps_polish_bacterial_methylation_v5.0.0
  1
