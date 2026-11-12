Resolve a model from a user-specified smallvar model name.
Resolving from models-directory. Downloads will be tested separately.
  $ rm -rf out; mkdir -p out
  > #
  > TARGET_MODEL="dna_r10.4.1_e8.2_400bps_hac@v5.2.0"
  > SMALLVAR_PARAMS="--models-directory ${MODEL_ROOT_DIR} --model-override ${TARGET_MODEL}_smallvar@v1.0"
  > REMOVE_RG=0
  > REMOVE_DWELLS=1
  > #
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > #
  > source ${TEST_DIR}/cram/variant/helpers.sh
  > prepare_variant_mock_data "${in_dir}/in.aln.bam" "${TARGET_MODEL}" ${REMOVE_RG} ${REMOVE_DWELLS} "out/in.bam"
  > #
  > ### Run the unit under test.
  > ${DORADO_BIN} smallvar --device cpu out/in.bam ${in_dir}/in.ref.fasta.gz -t 4 --regions "chr20:1-100" -v ${SMALLVAR_PARAMS} > out/out.vcf 2> out/out.stderr
  > #
  > ### Eval.
  > echo "Exit code: $?"
  > grep "Resolved model" out/out.stderr | sed -E 's/.*\[debug\] //g' | sed -E "s/'.*${TARGET_MODEL}.*'*/TARGET_MODEL/"
  > grep "bam_info.has_dwells" out/out.stderr | sed -E 's/.*\[.*\] //g'
  > grep "\- downloading" out/out.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  Resolved model from user-specified smallvar model name: TARGET_MODEL
  bam_info.has_dwells = false
  0
  [warning] Skipping basecaller compatibility checks for user-specified model override. The accuracy of the results is not guaranteed.

Resolve a model from a user-specified basecaller model name.
Resolving from models-directory. Downloads will be tested separately.
  $ rm -rf out; mkdir -p out
  > #
  > TARGET_MODEL="dna_r10.4.1_e8.2_400bps_hac@v5.2.0"
  > SMALLVAR_PARAMS="--models-directory ${MODEL_ROOT_DIR} --model-override ${TARGET_MODEL}"
  > REMOVE_RG=0
  > REMOVE_DWELLS=1
  > #
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > #
  > source ${TEST_DIR}/cram/variant/helpers.sh
  > prepare_variant_mock_data "${in_dir}/in.aln.bam" "${TARGET_MODEL}" ${REMOVE_RG} ${REMOVE_DWELLS} "out/in.bam"
  > #
  > ### Run the unit under test.
  > ${DORADO_BIN} smallvar --device cpu out/in.bam ${in_dir}/in.ref.fasta.gz -t 4 --regions "chr20:1-100" -v ${SMALLVAR_PARAMS} > out/out.vcf 2> out/out.stderr
  > #
  > ### Eval.
  > echo "Exit code: $?"
  > grep "Resolved model" out/out.stderr | sed -E 's/.*\[debug\] //g' | sed -E "s/'.*${TARGET_MODEL}.*'*/TARGET_MODEL/"
  > grep "bam_info.has_dwells" out/out.stderr | sed -E 's/.*\[.*\] //g'
  > grep "\- downloading" out/out.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  Resolved model from user-specified basecaller model name: TARGET_MODEL
  bam_info.has_dwells = false
  0
  [warning] Skipping basecaller compatibility checks for user-specified model override. The accuracy of the results is not guaranteed.

Resolve a model from a user-specified folder.
  $ rm -rf out; mkdir -p out
  > #
  > TARGET_MODEL="dna_r10.4.1_e8.2_400bps_hac@v5.2.0"
  > SMALLVAR_PARAMS="--model-override ${MODEL_ROOT_DIR}/${TARGET_MODEL}_smallvar@v1.0"
  > REMOVE_RG=0
  > REMOVE_DWELLS=1
  > #
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > #
  > source ${TEST_DIR}/cram/variant/helpers.sh
  > prepare_variant_mock_data "${in_dir}/in.aln.bam" "${TARGET_MODEL}" ${REMOVE_RG} ${REMOVE_DWELLS} "out/in.bam"
  > #
  > ### Run the unit under test.
  > ${DORADO_BIN} smallvar --device cpu out/in.bam ${in_dir}/in.ref.fasta.gz -t 4 --regions "chr20:1-100" -v ${SMALLVAR_PARAMS} > out/out.vcf 2> out/out.stderr
  > #
  > ### Eval.
  > echo "Exit code: $?"
  > grep "Resolved model" out/out.stderr | sed -E 's/.*\[debug\] //g' | sed -E "s/'.*${TARGET_MODEL}.*'*/TARGET_MODEL/"
  > grep "bam_info.has_dwells" out/out.stderr | sed -E 's/.*\[.*\] //g'
  > grep "\- downloading" out/out.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  Resolved model from user-specified path: TARGET_MODEL
  bam_info.has_dwells = false
  0
  [warning] Skipping basecaller compatibility checks for user-specified model override. The accuracy of the results is not guaranteed.

Passing test with warnings: user-selected model does not match the basecaller model specified in the BAM.
Allowed because of `--model-override`.
  $ rm -rf out; mkdir -p out
  > #
  > MODEL_BAM="dna_r10.4.1_e8.2_400bps_hac@v1.0.0"
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
  > ${DORADO_BIN} smallvar --device cpu out/in.bam ${in_dir}/in.ref.fasta.gz -t 4 --regions "chr20:1-100" -v ${SMALLVAR_PARAMS} > out/out.vcf 2> out/out.stderr
  > #
  > ### Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  [warning] Skipping basecaller compatibility checks for user-specified model override. The accuracy of the results is not guaranteed.
  [warning] Variant calling model is not compatible with the input BAM. This may produce inferior results.

A Polishing model is provided instead of a Variant Calling model.
Incompatible label scheme should fail even with `--model-override`.
Using `--model-override`.
  $ rm -rf out; mkdir -p out
  > #
  > MODEL_BAM="dna_r10.4.1_e8.2_400bps_hac@v5.2.0"
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
  > model=${POLISH_MODEL}
  > #
  > ### Run the unit under test.
  > in_bam="out/in.bam"
  > in_ref=${in_dir}/in.ref.fasta.gz
  > ${DORADO_BIN} smallvar --model-override ${model} --device cpu ${in_bam} ${in_ref} -t 4 --infer-threads 1 -vv > out/out.vcf 2> out/out.stderr
  > #
  > ### Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.stderr | sed -E 's/.*\[/\[/g' | sed -E 's/model: .*/model/g'
  Exit code: 1
  [error] Incompatible model label scheme! Expected DiploidLabelScheme but got HaploidLabelScheme.
  [warning] Skipping basecaller compatibility checks for user-specified model override. The accuracy of the results is not guaranteed.

Cannot resolve the model, it does not match a Basecaller model, a SmallVar model, or a path.
  $ rm -rf out; mkdir -p out
  > #
  > SMALLVAR_PARAMS="--model-override unknown_model"
  > REMOVE_DWELLS=1
  > #
  > ### Run the unit under test.
  > ${DORADO_BIN} smallvar --device cpu "${in_dir}/in.aln.bam" ${in_dir}/in.ref.fasta.gz -t 4 --regions "chr20:1-100" -v ${SMALLVAR_PARAMS} > out/out.vcf 2> out/out.stderr
  > #
  > ### Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 1
  [error] Could not resolve model from string: 'unknown_model'.
  [warning] Skipping basecaller compatibility checks for user-specified model override. The accuracy of the results is not guaranteed.

Keyword 'auto' is not accepted as a model override alias.
  $ rm -rf out; mkdir -p out
  > #
  > SMALLVAR_PARAMS="--model-override auto"
  > REMOVE_DWELLS=1
  > #
  > ### Run the unit under test.
  > ${DORADO_BIN} smallvar --device cpu "${in_dir}/in.aln.bam" ${in_dir}/in.ref.fasta.gz -t 4 --regions "chr20:1-100" -v ${SMALLVAR_PARAMS} > out/out.vcf 2> out/out.stderr
  > #
  > ### Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 1
  [error] Could not resolve model from string: 'auto'.
  [warning] Skipping basecaller compatibility checks for user-specified model override. The accuracy of the results is not guaranteed.

Fail to download a non-existent auto-resolved model
  $ rm -rf out; mkdir -p out
  > #
  > ### Prepare mocked data.
  > source ${TEST_DIR}/cram/variant/helpers.sh
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > prepare_variant_mock_data "${in_dir}/in.aln.bam" "dna_r10.4.1_e8.2_400bps_sup@v5.0.0" 0 0 "out/in.bam"
  > in_bam=out/in.bam
  > in_ref=${in_dir}/in.ref.fasta.gz
  > #
  > ### Run the unit under test.
  > ${DORADO_BIN} smallvar --device cpu ${in_bam} ${in_ref} -t 4 -v > out/out.vcf 2> out/out.stderr
  > #
  > ### Eval.
  > echo "Exit code: $?"
  > grep "Resolved model from input data: 'dna_r10.4.1_e8.2_400bps_sup@v5.0.0_smallvar@v1.0'" out/out.stderr | wc -l | awk '{ print $1 }'
  > grep -- " - downloading" out/out.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 1
  0
  0
  [error] Could not find any smallvar model compatible with the basecaller model 'dna_r10.4.1_e8.2_400bps_sup@v5.0.0'.

No @RG lines in the input header.
Fails with auto-detect mode because the model cannot be resolved.
  $ rm -rf out; mkdir -p out
  > #
  > SMALLVAR_PARAMS="--models-directory ${MODEL_ROOT_DIR}"
  > REMOVE_RG=1
  > REMOVE_DWELLS=1
  > #
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > #
  > source ${TEST_DIR}/cram/variant/helpers.sh
  > prepare_variant_mock_data "${in_dir}/in.aln.bam" "" ${REMOVE_RG} ${REMOVE_DWELLS} "out/in.bam"
  > #
  > ### Run the unit under test.
  > ${DORADO_BIN} smallvar --device cpu out/in.bam ${in_dir}/in.ref.fasta.gz -t 4 --regions "chr20:1-100" -v ${SMALLVAR_PARAMS} > out/out.vcf 2> out/out.stderr
  > #
  > ### Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 1
  [error] Input BAM file has no basecaller models listed in the header.

No @RG lines in the input header.
Allowed because of `--model-override`.
  $ rm -rf out; mkdir -p out
  > #
  > MODEL_SMALLVAR="dna_r10.4.1_e8.2_400bps_hac@v5.2.0"
  > SMALLVAR_PARAMS="--model-override ${MODEL_ROOT_DIR}/${MODEL_SMALLVAR}_smallvar@v1.0"
  > REMOVE_RG=1
  > REMOVE_DWELLS=1
  > #
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > #
  > source ${TEST_DIR}/cram/variant/helpers.sh
  > prepare_variant_mock_data "${in_dir}/in.aln.bam" "" ${REMOVE_RG} ${REMOVE_DWELLS} "out/in.bam"
  > #
  > ### Run the unit under test.
  > ${DORADO_BIN} smallvar --device cpu out/in.bam ${in_dir}/in.ref.fasta.gz -t 4 --regions "chr20:1-100" -v ${SMALLVAR_PARAMS} > out/out.vcf 2> out/out.stderr
  > #
  > ### Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  [warning] Skipping basecaller compatibility checks for user-specified model override. The accuracy of the results is not guaranteed.
