Auto-resolve HAC v6.0.0. BAM does not have move tables (no `mv:`).
Resolving from models-directory. Downloads will be tested separately.
  $ rm -rf out; mkdir -p out
  > #
  > TARGET_MODEL="dna_r10.4.1_e8.2_400bps_hac@v6.1.0"
  > SMALLVAR_PARAMS="--models-directory ${MODEL_ROOT_DIR}"
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
  > grep "Resolved model" out/out.stderr | sed -E 's/.*\[debug\] //g' | sed -E "s/${TARGET_MODEL}[^']*/TARGET_MODEL/"
  > grep "bam_info.has_dwells" out/out.stderr | sed -E 's/.*\[.*\] //g'
  > grep "\- downloading" out/out.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  Resolved model from input data: 'TARGET_MODEL'
  bam_info.has_dwells = false
  0

Auto-resolve HAC v6.0.0. BAM has move tables (`mv:`).
Resolving from models-directory. Downloads will be tested separately.
  $ rm -rf out; mkdir -p out
  > #
  > TARGET_MODEL="dna_r10.4.1_e8.2_400bps_hac@v6.1.0"
  > SMALLVAR_PARAMS="--models-directory ${MODEL_ROOT_DIR}"
  > REMOVE_RG=0
  > REMOVE_DWELLS=0
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
  > grep "Resolved model" out/out.stderr | sed -E 's/.*\[debug\] //g' | sed -E "s/${TARGET_MODEL}[^']*/TARGET_MODEL/"
  > grep "bam_info.has_dwells" out/out.stderr | sed -E 's/.*\[.*\] //g'
  > grep "\- downloading" out/out.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  Resolved model from input data: 'TARGET_MODEL'
  bam_info.has_dwells = true
  0

Download the model.
  $ rm -rf out; mkdir -p out
  > #
  > TARGET_MODEL="dna_r10.4.1_e8.2_400bps_hac@v6.1.0"
  > SMALLVAR_PARAMS=""
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
  > grep "Resolved model" out/out.stderr | sed -E 's/.*\[debug\] //g' | sed -E "s/${TARGET_MODEL}[^']*/TARGET_MODEL/"
  > grep "bam_info.has_dwells" out/out.stderr | sed -E 's/.*\[.*\] //g'
  > grep "\- downloading" out/out.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  Resolved model from input data: 'TARGET_MODEL'
  bam_info.has_dwells = false
  1
