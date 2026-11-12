Stereo models should fail.
  $ rm -rf out; mkdir -p out
  > #
  > MODEL_BAM="dna_r10.4.1_e8.2_5khz_stereo@v1.3"
  > SMALLVAR_PARAMS="--models-directory ${MODEL_ROOT_DIR}"
  > REMOVE_RG=0
  > REMOVE_DWELLS=1
  > #
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > #
  > source ${TEST_DIR}/cram/variant/helpers.sh
  > prepare_variant_mock_data "${in_dir}/in.aln.bam" "${MODEL_BAM}" ${REMOVE_RG} ${REMOVE_DWELLS} "out/in.bam"
  > #
  > ### Run the unit under test.
  > ${DORADO_BIN} smallvar --device cpu out/in.bam ${in_dir}/in.ref.fasta.gz -t 4 -v ${SMALLVAR_PARAMS} > out/out.vcf 2> out/out.stderr
  > #
  > ### Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 1
  [error] Inputs from duplex basecalling are not supported. Detected model: 'dna_r10.4.1_e8.2_5khz_stereo@v1.3' in the input BAM.

Using `--model-override` with a stereo basecaller model should emit warnings.
This succeeds because the model was explicitly specified.
  $ rm -rf out; mkdir -p out
  > #
  > MODEL_BAM="dna_r10.4.1_e8.2_5khz_stereo@v1.3"
  > MODEL_SMALLVAR="dna_r10.4.1_e8.2_400bps_hac@v5.2.0"
  > SMALLVAR_PARAMS="--model-override ${MODEL_ROOT_DIR}/${MODEL_SMALLVAR}_smallvar@v1.0"
  > REMOVE_RG=0
  > REMOVE_DWELLS=1
  > #
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > #
  > source ${TEST_DIR}/cram/variant/helpers.sh
  > prepare_variant_mock_data "${in_dir}/in.aln.bam" "${MODEL_BAM}" ${REMOVE_RG} ${REMOVE_DWELLS} "out/in.bam"
  > #
  > ### Run the unit under test.
  > ${DORADO_BIN} smallvar --device cpu out/in.bam ${in_dir}/in.ref.fasta.gz -t 4 -v ${SMALLVAR_PARAMS} > out/out.vcf 2> out/out.stderr
  > #
  > ### Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  [warning] Inputs from duplex basecalling are not supported. Detected model: 'dna_r10.4.1_e8.2_5khz_stereo@v1.3' in the input BAM. This may produce inferior results.
  [warning] Skipping basecaller compatibility checks for user-specified model override. The accuracy of the results is not guaranteed.
  [warning] Variant calling model is not compatible with the input BAM. This may produce inferior results.

Two read groups are present in the input BAM, and two basecaller models. One of the basecaller models is stereo.
Since `--model-override` is used and the model explicitly specified, only warnings should be emitted.
  $ rm -rf out; mkdir -p out
  > #
  > MODEL_BAM_RG_01="dna_r10.4.1_e8.2_400bps_hac@v5.2.0"
  > MODEL_BAM_RG_02="dna_r10.4.1_e8.2_5khz_stereo@v1.3"
  > MODEL_SMALLVAR="dna_r10.4.1_e8.2_400bps_hac@v5.2.0"
  > SMALLVAR_PARAMS="--model-override ${MODEL_ROOT_DIR}/${MODEL_SMALLVAR}_smallvar@v1.0 --regions chr20:1-100 --ignore-read-groups"
  > #
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > #
  > source ${TEST_DIR}/cram/variant/helpers.sh
  > prepare_variant_mock_data_two_rg "${in_dir}/in.aln.bam" "${MODEL_BAM_RG_01}" "${MODEL_BAM_RG_02}" "out/in.bam"
  > #
  > ### Run the unit under test.
  > ${DORADO_BIN} smallvar --device cpu out/in.bam ${in_dir}/in.ref.fasta.gz -t 4 -v ${SMALLVAR_PARAMS} > out/out.vcf 2> out/out.stderr
  > #
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.stderr | sed -E 's/.*\[/\[/g' | sed -E "s/user-specified model: '[^']*'/user-specified model/g"
  Exit code: 0
  [warning] Inputs from duplex basecalling are not supported. Detected model: 'dna_r10.4.1_e8.2_5khz_stereo@v1.3' in the input BAM. This may produce inferior results.
  [warning] Skipping basecaller compatibility checks for user-specified model override. The accuracy of the results is not guaranteed.
  [warning] Variant calling model is not compatible with the input BAM. This may produce inferior results.

Two read groups are present in the input BAM, and two basecaller models. One of the basecaller models is stereo.
Reusing the same data from the test above, but run without `--model-override`.
This should error out.
  $ rm -rf out; mkdir -p out
  > #
  > MODEL_BAM_RG_01="dna_r10.4.1_e8.2_400bps_hac@v5.2.0"
  > MODEL_BAM_RG_02="dna_r10.4.1_e8.2_5khz_stereo@v1.3"
  > MODEL_SMALLVAR="dna_r10.4.1_e8.2_400bps_hac@v5.2.0"
  > SMALLVAR_PARAMS="--regions chr20:1-100 --ignore-read-groups"
  > #
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > #
  > source ${TEST_DIR}/cram/variant/helpers.sh
  > prepare_variant_mock_data_two_rg "${in_dir}/in.aln.bam" "${MODEL_BAM_RG_01}" "${MODEL_BAM_RG_02}" "out/in.bam"
  > #
  > ### Run the unit under test.
  > ${DORADO_BIN} smallvar --device cpu out/in.bam ${in_dir}/in.ref.fasta.gz -t 4 -v ${SMALLVAR_PARAMS} > out/out.vcf 2> out/out.stderr
  > #
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 1
  [error] Input BAM file has a mix of different basecaller models. Only one basecaller model can be processed. List of all basecaller models found in the BAM file: dna_r10.4.1_e8.2_400bps_hac@v5.2.0, dna_r10.4.1_e8.2_5khz_stereo@v1.3
