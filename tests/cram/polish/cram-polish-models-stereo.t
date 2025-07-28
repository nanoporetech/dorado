Data construction.
Create a very small synthetic input BAM file of only one record.
Reason: reducing the inference time for successful tests.
  $ rm -rf data; mkdir -p data
  > in_dir="${TEST_DATA_DIR}/polish/test-01-supertiny"
  > in_bam="${in_dir}/calls_to_draft.bam"
  > ### Create a BAM file with zero read groups.
  > samtools view -H ${in_bam} > data/in.micro.sam
  > samtools view ${in_bam} | head -n 1 >> data/in.micro.sam
  > samtools view -Sb data/in.micro.sam | samtools sort > data/in.micro.bam
  > samtools index data/in.micro.bam

Stereo models should fail.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > ### Create synthetic data with mocked model name.
  > samtools view -h data/in.micro.bam | sed 's/dna_r10.4.1_e8.2_400bps_hac@v5.0.0/dna_r10.4.1_e8.2_5khz_stereo@v1.3/g' | samtools view -Sb > out/in.bam
  > samtools index out/in.bam
  > ### Run the unit under test.
  > ${DORADO_BIN} polish --device cpu out/in.bam ${in_dir}/draft.fasta.gz -t 4 -v > out/out.fasta 2> out/out.fasta.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*(\[warning\].*)/\1/g'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*(\[error\].*)/\1/g'
  Exit code: 1
  [error] Inputs from duplex basecalling are not supported. Detected model: 'dna_r10.4.1_e8.2_5khz_stereo@v1.3' in the input BAM.

Using `--skip-model-compatibility-check` will have no effect when `--models-directory` is used.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > ### Create synthetic data with mocked model name.
  > samtools view -h data/in.micro.bam | sed 's/dna_r10.4.1_e8.2_400bps_hac@v5.0.0/dna_r10.4.1_e8.2_5khz_stereo@v1.3/g' | samtools view -Sb > out/in.bam
  > samtools index out/in.bam
  > ### Run the unit under test.
  > ${DORADO_BIN} polish --skip-model-compatibility-check --device cpu out/in.bam ${in_dir}/draft.fasta.gz -t 4 -v > out/out.fasta 2> out/out.fasta.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*(\[warning\].*)/\1/g'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*(\[error\].*)/\1/g'
  Exit code: 1
  [error] Inputs from duplex basecalling are not supported. Detected model: 'dna_r10.4.1_e8.2_5khz_stereo@v1.3' in the input BAM.
