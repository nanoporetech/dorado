Latest available HAC model, NO DWELLS.
Assumption is that the last one on the list is the latest one.
This is intended to capture ill defined model config files.
If this fails, the `supported_basecallers` in the model `config.toml` likely hasn't been updated
and new integration tests for the new model have not been added.
  $ rm -rf out; mkdir -p out
  > #
  > SOURCE_MODEL="dna_r10.4.1_e8.2_400bps_hac@v5.0.0"
  > TARGET_MODEL=$(${DORADO_BIN} download --list-structured 2>/dev/null | grep "polish" | sed -E 's/^[^"]*"([^"]*)".*/\1/' | grep -v "polish_models" | sed 's/_polish.*//g' | sort -u | grep "hac" | tail -n 1)
  > POLISH_PARAMS="--models-directory ${MODEL_ROOT_DIR} --device cpu -t 4 --regions contig_1:1-100 -v"
  > REMOVE_RG=0
  > REMOVE_DWELLS=1
  > #
  > in_dir="${TEST_DATA_DIR}/polish/test-01-supertiny"
  > #
  > source ${TEST_DIR}/cram/variant/helpers.sh
  > prepare_mock_data "${in_dir}/calls_to_draft.bam" "${SOURCE_MODEL}" "${TARGET_MODEL}" ${REMOVE_RG} ${REMOVE_DWELLS} "out/in.bam"
  > #
  > ### Run the unit under test.
  > ${DORADO_BIN} polish out/in.bam ${in_dir}/draft.fasta.gz ${POLISH_PARAMS} > out/out.vcf 2> out/out.stderr
  > #
  > ### Eval.
  > echo "Exit code: $?"
  > grep "Resolved model" out/out.stderr | sed -E 's/.*\[debug\] //g' | sed -E "s|${TARGET_MODEL}([^@]*).*$|TARGET_MODEL\1|"
  > grep "bam_info.has_dwells" out/out.stderr | sed -E 's/.*\[.*\] //g'
  > grep "\- downloading" out/out.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  Resolved model from input data: TARGET_MODEL_polish_rl
  bam_info.has_dwells = false
  0

Latest available HAC model, WITH DWELLS.
(Same description as above.)
  $ rm -rf out; mkdir -p out
  > #
  > SOURCE_MODEL="dna_r10.4.1_e8.2_400bps_hac@v5.0.0"
  > TARGET_MODEL=$(${DORADO_BIN} download --list-structured 2>/dev/null | grep "polish" | sed -E 's/^[^"]*"([^"]*)".*/\1/' | grep -v "polish_models" | sed 's/_polish.*//g' | sort -u | grep "hac" | tail -n 1)
  > POLISH_PARAMS="--models-directory ${MODEL_ROOT_DIR} --device cpu -t 4 --regions contig_1:1-100 -v"
  > REMOVE_RG=0
  > REMOVE_DWELLS=0
  > #
  > in_dir="${TEST_DATA_DIR}/polish/test-01-supertiny"
  > #
  > source ${TEST_DIR}/cram/variant/helpers.sh
  > prepare_mock_data "${in_dir}/calls_to_draft.bam" "${SOURCE_MODEL}" "${TARGET_MODEL}" ${REMOVE_RG} ${REMOVE_DWELLS} "out/in.bam"
  > #
  > ### Run the unit under test.
  > ${DORADO_BIN} polish out/in.bam ${in_dir}/draft.fasta.gz ${POLISH_PARAMS} > out/out.vcf 2> out/out.stderr
  > #
  > ### Eval.
  > echo "Exit code: $?"
  > grep "Resolved model" out/out.stderr | sed -E 's/.*\[debug\] //g' | sed -E "s|${TARGET_MODEL}([^@]*).*$|TARGET_MODEL\1|"
  > grep "bam_info.has_dwells" out/out.stderr | sed -E 's/.*\[.*\] //g'
  > grep "\- downloading" out/out.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  Resolved model from input data: TARGET_MODEL_polish_rl_mv
  bam_info.has_dwells = true
  0

Latest available bacterial model.
(Same description as above.)
  $ rm -rf out; mkdir -p out
  > #
  > SOURCE_MODEL="dna_r10.4.1_e8.2_400bps_hac@v5.0.0"
  > TARGET_MODEL=$(${DORADO_BIN} download --list-structured 2>/dev/null | grep "polish" | sed -E 's/^[^"]*"([^"]*)".*/\1/' | grep -v "polish_models" | sed 's/_polish.*//g' | sort -u | grep "hac" | tail -n 1)
  > POLISH_PARAMS="--bacteria --models-directory ${MODEL_ROOT_DIR} --device cpu -t 4 --regions contig_1:1-100 -v"
  > REMOVE_RG=0
  > REMOVE_DWELLS=1
  > #
  > in_dir="${TEST_DATA_DIR}/polish/test-01-supertiny"
  > #
  > source ${TEST_DIR}/cram/variant/helpers.sh
  > prepare_mock_data "${in_dir}/calls_to_draft.bam" "${SOURCE_MODEL}" "${TARGET_MODEL}" ${REMOVE_RG} ${REMOVE_DWELLS} "out/in.bam"
  > #
  > ### Run the unit under test.
  > ${DORADO_BIN} polish out/in.bam ${in_dir}/draft.fasta.gz ${POLISH_PARAMS} > out/out.vcf 2> out/out.stderr
  > #
  > ### Eval.
  > echo "Exit code: $?"
  > grep "Resolved model" out/out.stderr | sed -E 's/.*\[debug\] //g' | sed -E "s|([^@]*).*$|\1<VERSION>|"
  > grep "bam_info.has_dwells" out/out.stderr | sed -E 's/.*\[.*\] //g'
  > grep "\- downloading" out/out.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  Resolved model from input data: dna_r10.4.1_e8.2_400bps_polish_bacterial_methylation<VERSION>
  bam_info.has_dwells = false
  0