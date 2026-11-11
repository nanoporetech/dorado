Test the latest available model. Assumption is that the last one on the list is the latest one.
This is intended to capture ill defined model config files.
If this fails, the `supported_basecallers` in the model `config.toml` likely hasn't been updated
and new integration tests for the new model have not been added.
  $ rm -rf out; mkdir -p out
  > #
  > TARGET_MODEL=$(${DORADO_BIN} download --list-structured 2>/dev/null | grep "smallvar" | sed -E 's/^[^"]*"([^"]*)".*/\1/' | tail -n 1 | sed 's/_smallvar.*//g' )
  > SMALLVAR_PARAMS="--models-directory ${MODEL_ROOT_DIR}"
  > REMOVE_RG=0
  > REMOVE_DWELLS=1
  > #
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > #
  > source ${TEST_DIR}/cram/variant/helpers.sh
  > prepare_mock_data "${in_dir}/in.aln.bam" "${TARGET_MODEL}" ${REMOVE_RG} ${REMOVE_DWELLS} "out/in.bam"
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
