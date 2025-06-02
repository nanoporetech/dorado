Stereo models should fail.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > ### Create synthetic data with mocked model name.
  > samtools view -h ${in_dir}/calls_to_draft.bam | sed 's/dna_r10.4.1_e8.2_400bps_hac@v5.0.0/dna_r10.4.1_e8.2_5khz_stereo@v1.3/g' | samtools view -Sb > out/in.bam
  > samtools index out/in.bam
  > ### Run the unit under test.
  > ${DORADO_BIN} polish --device cpu out/in.bam ${in_dir}/draft.fasta.gz -t 4 -v > out/out.fasta 2> out/out.fasta.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*(\[warning\].*)/\1/g'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*(\[error\].*)/\1/g'
  Exit code: 1
  [error] Duplex basecalling models are not supported. Model: 'dna_r10.4.1_e8.2_5khz_stereo@v1.3'.

Using `--skip-model-compatibility-check` with a stereo basecaller model should emit a warning first.
This still fails because the model cannot be resolved automatically.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > ### Create synthetic data with mocked model name.
  > samtools view -h ${in_dir}/calls_to_draft.bam | sed 's/dna_r10.4.1_e8.2_400bps_hac@v5.0.0/dna_r10.4.1_e8.2_5khz_stereo@v1.3/g' | samtools view -Sb > out/in.bam
  > samtools index out/in.bam
  > ### Run the unit under test.
  > ${DORADO_BIN} polish --skip-model-compatibility-check --device cpu out/in.bam ${in_dir}/draft.fasta.gz -t 4 -v > out/out.fasta 2> out/out.fasta.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*(\[warning\].*)/\1/g'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*(\[error\].*)/\1/g'
  Exit code: 1
  [warning] Duplex basecalling models are not supported. Model: 'dna_r10.4.1_e8.2_5khz_stereo@v1.3'. This may produce inferior results.
  [error] There are no polishing models for the basecaller model: 'dna_r10.4.1_e8.2_5khz_stereo@v1.3'.

Using `--skip-model-compatibility-check` with a stereo basecaller model should emit a warning first.
This succeeds because the model was explicitly specified.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > ### Create synthetic data with mocked model name.
  > samtools view -h ${in_dir}/calls_to_draft.bam | sed 's/dna_r10.4.1_e8.2_400bps_hac@v5.0.0/dna_r10.4.1_e8.2_5khz_stereo@v1.3/g' | samtools view -Sb > out/in.bam
  > samtools index out/in.bam
  > ### Run the unit under test.
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > ${DORADO_BIN} polish --skip-model-compatibility-check --device cpu out/in.bam ${in_dir}/draft.fasta.gz -t 4 ${model_var} -v > out/out.fasta 2> out/out.fasta.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*(\[warning\].*)/\1/g'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*(\[error\].*)/\1/g'
  Exit code: 0
  [warning] Duplex basecalling models are not supported. Model: 'dna_r10.4.1_e8.2_5khz_stereo@v1.3'. This may produce inferior results.
  [warning] Polishing model is not compatible with the input BAM. This may produce inferior results.

Two read groups are present in the input BAM, and two basecaller models. One of the basecaller models is stereo.
Since `--skip-model-compatibility-check` is used and the model explicitly specified, only warnings should be emitted.
  $ rm -rf out; mkdir -p out
  > in_dir="${TEST_DATA_DIR}/polish/test-01-supertiny"
  > in_bam="${in_dir}/calls_to_draft.bam"
  > ### Create a BAM file with two read groups and two basecallers.
  > samtools view -h ${in_bam} | sed -E 's/bc8993f4557dd53bf0cbda5fd68453fea5e94485_dna_r10.4.1_e8.2_400bps_hac@v5.0.0-3AD2AF3E/f9f2e0901209274f132de3554913a1dc0a4439ae_dna_r10.4.1_e8.2_400bps_hac@v5.0.0-2DEAC5EC/g' > out/tmp.sam
  > cat out/tmp.sam | sed -E 's/dna_r10.4.1_e8.2_400bps_hac@v5.0.0/dna_r10.4.1_e8.2_5khz_stereo@v1.3/g' > out/tmp2.sam
  > samtools view -Sb out/tmp2.sam | samtools sort > out/tmp2.bam
  > samtools merge out/in.two_read_groups.two_basecallers.bam ${in_bam} out/tmp2.bam
  > samtools index out/in.two_read_groups.two_basecallers.bam
  > ### Run the test
  > in_bam="out/in.two_read_groups.two_basecallers.bam"
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > ${DORADO_BIN} polish --skip-model-compatibility-check --ignore-read-groups --device cpu ${in_bam} ${in_dir}/draft.fasta.gz ${model_var} -vv > out/out.fasta 2> out/out.fasta.stderr
  > echo "Exit code: $?"
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*(\[warning\].*)/\1/g'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*(\[error\].*)/\1/g'
  Exit code: 0
  [warning] Duplex basecalling models are not supported. Model: 'dna_r10.4.1_e8.2_5khz_stereo@v1.3'. This may produce inferior results.
  [warning] Polishing model is not compatible with the input BAM. This may produce inferior results.

Two read groups are present in the input BAM, and two basecaller models. One of the basecaller models is stereo.
Reusing the same data from the test above, but run without `--skip-model-compatibility-check`.
This should error out.
  $ ### Run the test
  > in_bam="out/in.two_read_groups.two_basecallers.bam"
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > ${DORADO_BIN} polish --ignore-read-groups --device cpu ${in_bam} ${in_dir}/draft.fasta.gz ${model_var} -vv > out/out.fasta 2> out/out.fasta.stderr
  > echo "Exit code: $?"
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*(\[warning\].*)/\1/g'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*(\[error\].*)/\1/g'
  Exit code: 1
  [error] Duplex basecalling models are not supported. Model: 'dna_r10.4.1_e8.2_5khz_stereo@v1.3'.
