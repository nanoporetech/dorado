HAC v5.2.0 with dwells.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > ### Create synthetic data with mocked model name.
  > samtools view -h ${in_dir}/calls_to_draft.bam | sed 's/dna_r10.4.1_e8.2_400bps_hac@v5.0.0/dna_r10.4.1_e8.2_400bps_hac@v5.2.0/g' | samtools view -Sb > out/in.bam
  > samtools index out/in.bam
  > ### Run the unit under test.
  > ${DORADO_BIN} polish --device cpu out/in.bam ${in_dir}/draft.fasta.gz -t 4 -v > out/out.fasta 2> out/out.fasta.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "Resolved model" out/out.fasta.stderr | sed -E 's/.*\[debug\] //g'
  > grep "Downloading model" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[error\] //g'
  Exit code: 0
  Resolved model from input data: dna_r10.4.1_e8.2_400bps_hac@v5.2.0_polish_rl_mv
  1

Bacterial HAC v5.2.0 with dwells.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > ### Create synthetic data with mocked model name.
  > samtools view -h ${in_dir}/calls_to_draft.bam | sed 's/dna_r10.4.1_e8.2_400bps_hac@v5.0.0/dna_r10.4.1_e8.2_400bps_hac@v5.2.0/g' | samtools view -Sb > out/in.bam
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

SUP v5.2.0 with dwells.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > ### Create synthetic data with mocked model name.
  > samtools view -h ${in_dir}/calls_to_draft.bam | sed 's/dna_r10.4.1_e8.2_400bps_hac@v5.0.0/dna_r10.4.1_e8.2_400bps_sup@v5.2.0/g' | samtools view -Sb > out/in.bam
  > samtools index out/in.bam
  > ### Run the unit under test.
  > ${DORADO_BIN} polish --device cpu out/in.bam ${in_dir}/draft.fasta.gz -t 4 -v > out/out.fasta 2> out/out.fasta.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "Resolved model" out/out.fasta.stderr | sed -E 's/.*\[debug\] //g'
  > grep "Downloading model" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[error\] //g'
  Exit code: 0
  Resolved model from input data: dna_r10.4.1_e8.2_400bps_sup@v5.2.0_polish_rl_mv
  1

Bacterial SUP v5.2.0 with dwells.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > ### Create synthetic data with mocked model name.
  > samtools view -h ${in_dir}/calls_to_draft.bam | sed 's/dna_r10.4.1_e8.2_400bps_hac@v5.0.0/dna_r10.4.1_e8.2_400bps_sup@v5.2.0/g' | samtools view -Sb > out/in.bam
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

HAC v5.2.0 with no dwells.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > ### Create synthetic data with mocked model name.
  > samtools view -H ${in_dir}/calls_to_draft.bam | sed 's/dna_r10.4.1_e8.2_400bps_hac@v5.0.0/dna_r10.4.1_e8.2_400bps_hac@v5.2.0/g' > out/in.sam
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
  Resolved model from input data: dna_r10.4.1_e8.2_400bps_hac@v5.2.0_polish_rl
  1

Bacterial HAC v5.2.0 with no dwells.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > ### Create synthetic data with mocked model name.
  > samtools view -H ${in_dir}/calls_to_draft.bam | sed 's/dna_r10.4.1_e8.2_400bps_hac@v5.0.0/dna_r10.4.1_e8.2_400bps_hac@v5.2.0/g' > out/in.sam
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

SUP v5.2.0 with no dwells.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > ### Create synthetic data with mocked model name.
  > samtools view -H ${in_dir}/calls_to_draft.bam | sed 's/dna_r10.4.1_e8.2_400bps_hac@v5.0.0/dna_r10.4.1_e8.2_400bps_sup@v5.2.0/g' > out/in.sam
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
  Resolved model from input data: dna_r10.4.1_e8.2_400bps_sup@v5.2.0_polish_rl
  1

Bacterial SUP v5.2.0 with no dwells.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > ### Create synthetic data with mocked model name.
  > samtools view -H ${in_dir}/calls_to_draft.bam | sed 's/dna_r10.4.1_e8.2_400bps_hac@v5.0.0/dna_r10.4.1_e8.2_400bps_sup@v5.2.0/g' > out/in.sam
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
