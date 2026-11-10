Data construction.
Create a very small synthetic input BAM file of only one record.
Reason: reducing the inference time for successful tests.
  $ rm -rf data; mkdir -p data
  > in_dir="${TEST_DATA_DIR}/polish/test-01-supertiny"
  > in_bam="${in_dir}/calls_to_draft.bam"
  > samtools view -H ${in_bam} > data/in.micro.sam
  > samtools view ${in_bam} | head -n 1 >> data/in.micro.sam
  > samtools view -Sb data/in.micro.sam | samtools sort > data/in.micro.bam
  > samtools index data/in.micro.bam

###################################################
### Test auto-resolve from the input BAM file.  ###
###################################################
Auto-resolve the move-table `dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl_mv` model from the BAM file and download it to a temporary folder.
No need to test all available models exhaustively, there are separate Cram files for that.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > ${DORADO_BIN} polish --device cpu data/in.micro.bam ${in_dir}/draft.fasta.gz -t 4 --regions "contig_1:1-100" -v > out/out.fasta 2> out/out.fasta.stderr
  > echo "Exit code: $?"
  > grep "Resolved model from input data: dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl_mv" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep -- " - downloading dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl_mv with " out/out.fasta.stderr | wc -l | awk '{ print ($1 > 0 ? 1 : 0) }'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  1
  1

Auto-resolve the NON-move-table `dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl` model from the BAM file and download it to a temporary folder.
The input BAM is stripped of the `mv:B:c` tag.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=data/in.micro.bam
  > # Prepare data. Strip off all tags.
  > samtools view -h ${in_bam} | sed -E 's/\tmv:B:c[0-9,:]*//g' | samtools view -Sb > out/in.bam
  > samtools index out/in.bam
  > # Run test.
  > ${DORADO_BIN} polish --device cpu out/in.bam ${in_dir}/draft.fasta.gz --regions "contig_1:1-100" -vv > out/out.fasta 2> out/out.fasta.stderr
  > # Eval.
  > echo "Exit code: $?"
  > grep "Resolved model from input data: dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep -- " - downloading dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl with " out/out.fasta.stderr | wc -l | awk '{ print ($1 > 0 ? 1 : 0) }'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  1
  1

Auto-resolve the `dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl_mv` model from the BAM file and use the pre-cached folder.
There should be no "downloading" log line and the process should succeed.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > model_var="--models-directory ${MODEL_ROOT_DIR}"
  > ${DORADO_BIN} polish --device cpu data/in.micro.bam ${in_dir}/draft.fasta.gz -t 4 --regions "contig_1:1-100" -v ${model_var} > out/out.fasta 2> out/out.fasta.stderr
  > echo "Exit code: $?"
  > grep "Resolved model from input data: dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl_mv" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep -- " - downloading dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl_mv with " out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  1
  0

Auto resolve to a custom folder.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > ${DORADO_BIN} polish --models-directory out --device cpu data/in.micro.bam ${in_dir}/draft.fasta.gz -t 4 --regions "contig_1:1-100" -v > out/out.fasta 2> out/out.fasta.stderr
  > echo "Exit code: $?"
  > grep "Resolved model from input data: dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl_mv" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep -- " - downloading dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl_mv with " out/out.fasta.stderr | wc -l | awk '{ print ($1 > 0 ? 1 : 0) }'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  1
  1

Reuse the pre-downloaded model (freshly downloaded one test above). There should be no "downloading" log line and the process should succeed.
  $ in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > ${DORADO_BIN} polish --models-directory out --device cpu data/in.micro.bam ${in_dir}/draft.fasta.gz -t 4 --regions "contig_1:1-100" -v > out/out.fasta 2> out/out.fasta.stderr
  > echo "Exit code: $?"
  > grep "Resolved model from input data: dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl_mv" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep -- " - downloading dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl_mv with " out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  1
  0

Manually specify a model from the models directory.
  $ in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > ${DORADO_BIN} polish --models-directory ${MODEL_ROOT_DIR} --model-override "dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl_mv" --device cpu data/in.micro.bam ${in_dir}/draft.fasta.gz -t 4 --regions "contig_1:1-100" -v > out/out.fasta 2> out/out.fasta.stderr
  > echo "Exit code: $?"
  > grep -- " - downloading dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl_mv with " out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  0
  [warning] Skipping basecaller compatibility checks for user-specified model override. The accuracy of the results is not guaranteed.

###################################################
### Test resolving from the Basecaller model    ###
### name, Polishing model name or a path.       ###
###################################################
Resolve the model from a Basecaller model name `dna_r10.4.1_e8.2_400bps_hac@v5.0.0`.
This is not supported and should fail.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > model="${MODEL_ROOT_DIR}/dna_r10.4.1_e8.2_400bps_hac@v5.0.0"
  > ### Run the unit under test.
  > ${DORADO_BIN} polish -v --device cpu data/in.micro.bam ${in_dir}/draft.fasta.gz -t 4 --regions "contig_1:1-100" --infer-threads 1 --model-override "${model}" > out/out.fasta 2> out/out.fasta.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g' | sed -E 's/from string: .*/from string: <>/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 1
  [error] Could not resolve model from string: <>
  [warning] Skipping basecaller compatibility checks for user-specified model override. The accuracy of the results is not guaranteed.

Resolve the model from a local Polishing model path.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > model="${MODEL_ROOT_DIR}/dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl_mv"
  > ### Run the unit under test.
  > ${DORADO_BIN} polish -v --device cpu data/in.micro.bam ${in_dir}/draft.fasta.gz -t 4 --regions "contig_1:1-100" --infer-threads 1 --model-override "${model}" > out/out.fasta 2> out/out.fasta.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "Resolved model from user-specified path: " out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep -- " - downloading" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  1
  0
  [warning] Skipping basecaller compatibility checks for user-specified model override. The accuracy of the results is not guaranteed.

Resolve the model from a local path.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > model="${POLISH_MODEL:-${MODEL_ROOT_DIR}/dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl_mv}"
  > ### Run the unit under test.
  > ${DORADO_BIN} polish -v --device cpu data/in.micro.bam ${in_dir}/draft.fasta.gz -t 4 --regions "contig_1:1-100" --infer-threads 1 --model-override "${model}" > out/out.fasta 2> out/out.fasta.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "Resolved model from user-specified path: " out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep -- " - downloading" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  1
  0
  [warning] Skipping basecaller compatibility checks for user-specified model override. The accuracy of the results is not guaranteed.

Resolve the bacterial model from a Basecaller model name `dna_r10.4.1_e8.2_400bps_hac@v5.0.0`.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > model="${MODEL_ROOT_DIR}/dna_r10.4.1_e8.2_400bps_hac@v5.0.0"
  > ### Run the unit under test.
  > ${DORADO_BIN} polish --bacteria -v --device cpu data/in.micro.bam ${in_dir}/draft.fasta.gz -t 4 --regions "contig_1:1-100" --infer-threads 1 --model-override "${model}" > out/out.fasta 2> out/out.fasta.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g' | sed -E 's/from string: .*/from string: <>/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 1
  [error] Could not resolve model from string: <>
  [warning] Skipping basecaller compatibility checks for user-specified model override. The accuracy of the results is not guaranteed.

##############################################
### Compatibility checks.                  ###
##############################################
All basecaller models resolve to the latest bacterial model.
Passing test: bacterial model resolves for any basecaller.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > model_var="--models-directory ${MODEL_ROOT_DIR}"
  > ### Create the mocked data for a non-supported basecaller version (1.0.0).
  > samtools view -h data/in.micro.bam | sed -E 's/dna_r10.4.1_e8.2_400bps_hac@v5.0.0/dna_r10.4.1_e8.2_400bps_hac@v1.0.0/g' | samtools view -Sb > out/in.bam
  > samtools index out/in.bam
  > ### Run the unit under test.
  > ${DORADO_BIN} polish --bacteria ${model_var} --device cpu out/in.bam ${in_dir}/draft.fasta.gz -t 4 --regions "contig_1:1-100" --infer-threads 1 -vv > out/out.fasta 2> out/out.fasta.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0

Passing test with warnings: data has dwells, but the model does not support them.
The `--model-override` option allows this, but emits a warning.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > model="${MODEL_ROOT_DIR}/dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl"
  > ### Run the unit under test.
  > ${DORADO_BIN} polish --device cpu data/in.micro.bam ${in_dir}/draft.fasta.gz -t 4 --regions "contig_1:1-100" --infer-threads 1 -vv --model-override "${model}" > out/out.fasta 2> out/out.fasta.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  [warning] Skipping basecaller compatibility checks for user-specified model override. The accuracy of the results is not guaranteed.
  [warning] Input data has move tables, but a model without move table support has been chosen. This may produce inferior results.

Passing test with warnings: no dwells in data, but the model uses them for polishing.
The `--model-override` option allows this, but emits a warning.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > model="${MODEL_ROOT_DIR}/dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl_mv"
  > ### Create the synthetic data with no dwells.
  > samtools view -H data/in.micro.bam > out/in.no_dwells.sam
  > samtools view data/in.micro.bam | cut -f1-11 >> out/in.no_dwells.sam
  > samtools view -Sb out/in.no_dwells.sam > out/in.no_dwells.bam
  > samtools index out/in.no_dwells.bam
  > ### Run the unit under test.
  > ${DORADO_BIN} polish --device cpu out/in.no_dwells.bam ${in_dir}/draft.fasta.gz -t 4 --regions "contig_1:1-100" --infer-threads 1 -vv --model-override "${model}" > out/out.fasta 2> out/out.fasta.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  [warning] Skipping basecaller compatibility checks for user-specified model override. The accuracy of the results is not guaranteed.
  [warning] Input data does not contain move tables, but a model which requires move tables has been chosen. This may produce inferior results.

Passing test with warnings: Basecaller model specified in the BAM does not match the Basecaller model specified in the Polishing model.
The `--model-override` option allows this, but emits a warning.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > model_var="--model-override ${POLISH_MODEL}"
  > ### Create the mocked data for a non-supported basecaller version (1.0.0).
  > samtools view -h data/in.micro.bam | sed -E 's/dna_r10.4.1_e8.2_400bps_hac@v5.0.0/dna_r10.4.1_e8.2_400bps_hac@v1.0.0/g' | samtools view -Sb > out/in.bam
  > samtools index out/in.bam
  > ### Run the unit under test.
  > ${DORADO_BIN} polish --device cpu out/in.bam ${in_dir}/draft.fasta.gz -t 4 --regions "contig_1:1-100" --infer-threads 1 -vv ${model_var} > out/out.fasta 2> out/out.fasta.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  [warning] Skipping basecaller compatibility checks for user-specified model override. The accuracy of the results is not guaranteed.
  [warning] Polishing model is not compatible with the input BAM. This may produce inferior results.

Negative test: Cannot resolve the model, it does not match a Polishing model nor a path.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > model="unknown"
  > ### Run the unit under test.
  > ${DORADO_BIN} polish --device cpu data/in.micro.bam ${in_dir}/draft.fasta.gz -t 4 --regions "contig_1:1-100" --infer-threads 1 -vv --model-override "${model}" > out/out.fasta 2> out/out.fasta.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g' | sed -E 's/model: .*/model/g'
  Exit code: 1
  [error] Could not resolve model from string: 'unknown'.
  [warning] Skipping basecaller compatibility checks for user-specified model override. The accuracy of the results is not guaranteed.

A Variant Calling model is provided instead of a Polishing model.
Incompatible label scheme should fail even with `--model-override`.
Using `--model-override`.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > model="${SMALLVAR_MODEL}"
  > ### Run the unit under test.
  > ${DORADO_BIN} polish --device cpu data/in.micro.bam ${in_dir}/draft.fasta.gz -t 4 --regions "contig_1:1-100" --infer-threads 1 -vv --model-override "${model}" > out/out.fasta 2> out/out.fasta.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 1
  [error] Incompatible model label scheme! Expected HaploidLabelScheme but got DiploidLabelScheme.
  [warning] Skipping basecaller compatibility checks for user-specified model override. The accuracy of the results is not guaranteed.

#############################################
## Resolve move tables                    ###
#############################################
Quality field in the BAM is set to `*` instead of actual qualities (valid according to the SAM format specification).
All other custom tags are left inside.
Use a model without move tables.
Results will be worse than using QVs, but this input should successfully run nonetheless.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=data/in.micro.bam
  > model="${MODEL_ROOT_DIR}/dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl"
  > # Prepare data. Strip off all tags.
  > samtools view -H ${in_bam} > out/in.sam
  > samtools view ${in_bam} | awk -v OFS='\t' '{ $11="*"; print }' >> out/in.sam
  > samtools view -Sb out/in.sam | samtools sort > out/in.bam
  > samtools index out/in.bam
  > # Run test.
  > ${DORADO_BIN} polish --device cpu out/in.bam ${in_dir}/draft.fasta.gz --regions "contig_1:1-100" -vv --model-override "${model}" > out/out.fasta 2> out/out.fasta.stderr
  > # Eval.
  > echo "Exit code: $?"
  > samtools faidx out/out.fasta
  > awk '{ print $1,$2 }' out/out.fasta.fai
  > grep "Copying contig verbatim from input" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  contig_1 10001
  0
  [warning] Skipping basecaller compatibility checks for user-specified model override. The accuracy of the results is not guaranteed.
  [warning] Input data has move tables, but a model without move table support has been chosen. This may produce inferior results.

The input BAM is stripped of all tags (no move tables) but the specified model was trained using the move tables.
The `--model-override` option allows this, but emits a warning.
Results are worse than using the correct model, as expected.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=data/in.micro.bam
  > model="${MODEL_ROOT_DIR}/dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl_mv"
  > # Prepare data. Strip off all tags.
  > samtools view -H ${in_bam} > out/in.sam
  > samtools view ${in_bam} | cut -f1-11 >> out/in.sam
  > samtools view -Sb out/in.sam | samtools sort > out/in.bam
  > samtools index out/in.bam
  > # Run test.
  > ${DORADO_BIN} polish --device cpu out/in.bam ${in_dir}/draft.fasta.gz --regions "contig_1:1-100" -vv  --model-override "${model}" > out/out.fasta 2> out/out.fasta.stderr
  > # Eval.
  > echo "Exit code: $?"
  > samtools faidx out/out.fasta
  > awk '{ print $1,$2 }' out/out.fasta.fai
  > grep "Copying contig verbatim from input" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  contig_1 10000
  0
  [warning] Skipping basecaller compatibility checks for user-specified model override. The accuracy of the results is not guaranteed.
  [warning] Input data does not contain move tables, but a model which requires move tables has been chosen. This may produce inferior results.

The input BAM _has_ move tables, but the specified model was trained without the move tables.
The `--model-override` option allows this, but emits a warning.
Results are worse than using the correct model, as expected.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=data/in.micro.bam
  > model="${MODEL_ROOT_DIR}/dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl"
  > # Run test.
  > ${DORADO_BIN} polish --device cpu ${in_bam} ${in_dir}/draft.fasta.gz --regions "contig_1:1-100" -vv > out/out.fasta --model-override "${model}" 2> out/out.fasta.stderr
  > # Eval.
  > echo "Exit code: $?"
  > samtools faidx out/out.fasta
  > awk '{ print $1,$2 }' out/out.fasta.fai
  > grep "Copying contig verbatim from input" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  contig_1 9995
  0
  [warning] Skipping basecaller compatibility checks for user-specified model override. The accuracy of the results is not guaranteed.
  [warning] Input data has move tables, but a model without move table support has been chosen. This may produce inferior results.
