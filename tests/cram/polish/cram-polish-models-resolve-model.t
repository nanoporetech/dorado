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
  > grep "\- downloading" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
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
  > grep "\- downloading" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
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
  > grep "\- downloading" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
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
  > grep "\- downloading" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
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
  > grep "\- downloading" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  1
  0

###################################################
### Test resolving from the Basecaller model    ###
### name, Polishing model name or a path.       ###
###################################################
Resolve the model from a Basecaller model name `dna_r10.4.1_e8.2_400bps_hac@v5.0.0`.
This is not supported and should fail.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > model="dna_r10.4.1_e8.2_400bps_hac@v5.0.0"
  > ### Run the unit under test.
  > ${DORADO_BIN} polish -v --device cpu data/in.micro.bam ${in_dir}/draft.fasta.gz -t 4 --regions "contig_1:1-100" --infer-threads 1 --model-override "${model}" > out/out.fasta 2> out/out.fasta.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 1
  [error] Could not resolve model from string: 'dna_r10.4.1_e8.2_400bps_hac@v5.0.0'.

Resolve the model from an exact Polishing model name: `dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl_mv`.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > model="dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl_mv"
  > ### Run the unit under test.
  > ${DORADO_BIN} polish -v --device cpu data/in.micro.bam ${in_dir}/draft.fasta.gz -t 4 --regions "contig_1:1-100" --infer-threads 1 --model-override "${model}" > out/out.fasta 2> out/out.fasta.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "Resolved model from user-specified polishing model name: dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl_mv" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "\- downloading" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  1
  1

Resolve the model from a local path.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > ### Run the unit under test.
  > ${DORADO_BIN} polish -v --device cpu data/in.micro.bam ${in_dir}/draft.fasta.gz -t 4 --regions "contig_1:1-100" --infer-threads 1 --model-override "${model}" > out/out.fasta 2> out/out.fasta.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "Resolved model from user-specified polishing model name: " out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "\- downloading" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  1
  1

Resolve the bacterial model from a Basecaller model name `dna_r10.4.1_e8.2_400bps_hac@v5.0.0`.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > model="dna_r10.4.1_e8.2_400bps_hac@v5.0.0"
  > ### Run the unit under test.
  > ${DORADO_BIN} polish --bacteria -v --device cpu data/in.micro.bam ${in_dir}/draft.fasta.gz -t 4 --regions "contig_1:1-100" --infer-threads 1 --model-override "${model}" > out/out.fasta 2> out/out.fasta.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 1
  [error] Could not resolve model from string: 'dna_r10.4.1_e8.2_400bps_hac@v5.0.0'.

##############################################
### Compatibility checks.                  ###
##############################################
Negative test: data has dwells, but the model does not support them.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > model="dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl"
  > ### Run the unit under test.
  > ${DORADO_BIN} polish --device cpu data/in.micro.bam ${in_dir}/draft.fasta.gz -t 4 --regions "contig_1:1-100" --infer-threads 1 -vv --model-override "${model}" > out/out.fasta 2> out/out.fasta.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "Resolved model from user-specified polishing model name: dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 1
  1
  [error] Input data has move tables, but a model without move table support has been chosen.

Negative test: no dwells in data, but the model uses them for polishing.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > model="dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl_mv"
  > ### Create the synthetic data with no dwells.
  > samtools view -H data/in.micro.bam > out/in.no_dwells.sam
  > samtools view data/in.micro.bam | cut -f1-11 >> out/in.no_dwells.sam
  > samtools view -Sb out/in.no_dwells.sam > out/in.no_dwells.bam
  > samtools index out/in.no_dwells.bam
  > ### Run the unit under test.
  > ${DORADO_BIN} polish --device cpu out/in.no_dwells.bam ${in_dir}/draft.fasta.gz -t 4 --regions "contig_1:1-100" --infer-threads 1 -vv --model-override "${model}" > out/out.fasta 2> out/out.fasta.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "Resolved model from user-specified polishing model name: dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl_mv" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 1
  1
  [error] Input data does not contain move tables, but a model which requires move tables has been chosen.

Negative test: Basecaller model specified in the BAM does not match the Basecaller model specified in the Polishing model.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > model_var="--model-override ${MODEL_DIR}"
  > ### Create the mocked data for a non-supported basecaller version (1.0.0).
  > samtools view -h data/in.micro.bam | sed -E 's/dna_r10.4.1_e8.2_400bps_hac@v5.0.0/dna_r10.4.1_e8.2_400bps_hac@v1.0.0/g' | samtools view -Sb > out/in.bam
  > samtools index out/in.bam
  > ### Run the unit under test.
  > ${DORADO_BIN} polish --device cpu out/in.bam ${in_dir}/draft.fasta.gz -t 4 --regions "contig_1:1-100" --infer-threads 1 -vv ${model_var} > out/out.fasta 2> out/out.fasta.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 1
  [error] Polishing model is not compatible with the input BAM!

Negative test: no bacterial model is compatible with the model listed in the input BAM file.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > ### Create the mocked data for a non-supported basecaller version (1.0.0).
  > samtools view -h data/in.micro.bam | sed -E 's/dna_r10.4.1_e8.2_400bps_hac@v5.0.0/dna_r10.4.1_e8.2_400bps_hac@v1.0.0/g' | samtools view -Sb > out/in.bam
  > samtools index out/in.bam
  > ### Run the unit under test.
  > ${DORADO_BIN} polish --bacteria --device cpu out/in.bam ${in_dir}/draft.fasta.gz -t 4 --regions "contig_1:1-100" --infer-threads 1 -vv > out/out.fasta 2> out/out.fasta.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 1
  [error] There are no bacterial models for the basecaller model: 'dna_r10.4.1_e8.2_400bps_hac@v1.0.0'.

Passing test with warnings: data has dwells, but the model does not support them.
Using `--skip-model-compatibility-check`.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > model="dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl"
  > ### Run the unit under test.
  > ${DORADO_BIN} polish --skip-model-compatibility-check --device cpu data/in.micro.bam ${in_dir}/draft.fasta.gz -t 4 --regions "contig_1:1-100" --infer-threads 1 -vv --model-override "${model}" > out/out.fasta 2> out/out.fasta.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "Resolved model from user-specified polishing model name: dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  1
  [warning] Input data has move tables, but a model without move table support has been chosen. This may produce inferior results.

Passing test with warnings: no dwells in data, but the model uses them for polishing.
Using `--skip-model-compatibility-check`.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > model="dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl_mv"
  > ### Create the synthetic data with no dwells.
  > samtools view -H data/in.micro.bam > out/in.no_dwells.sam
  > samtools view data/in.micro.bam | cut -f1-11 >> out/in.no_dwells.sam
  > samtools view -Sb out/in.no_dwells.sam > out/in.no_dwells.bam
  > samtools index out/in.no_dwells.bam
  > ### Run the unit under test.
  > ${DORADO_BIN} polish --skip-model-compatibility-check --device cpu out/in.no_dwells.bam ${in_dir}/draft.fasta.gz -t 4 --regions "contig_1:1-100" --infer-threads 1 -vv --model-override "${model}" > out/out.fasta 2> out/out.fasta.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "Resolved model from user-specified polishing model name: dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl_mv" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  1
  [warning] Input data does not contain move tables, but a model which requires move tables has been chosen. This may produce inferior results.

Passing test with warnings: Basecaller model specified in the BAM does not match the Basecaller model specified in the Polishing model.
Using `--skip-model-compatibility-check`.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > model_var="--model-override ${MODEL_DIR}"
  > ### Create the mocked data for a non-supported basecaller version (1.0.0).
  > samtools view -h data/in.micro.bam | sed -E 's/dna_r10.4.1_e8.2_400bps_hac@v5.0.0/dna_r10.4.1_e8.2_400bps_hac@v1.0.0/g' | samtools view -Sb > out/in.bam
  > samtools index out/in.bam
  > ### Run the unit under test.
  > ${DORADO_BIN} polish --skip-model-compatibility-check --device cpu out/in.bam ${in_dir}/draft.fasta.gz -t 4 --regions "contig_1:1-100" --infer-threads 1 -vv ${model_var} > out/out.fasta 2> out/out.fasta.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  [warning] Polishing model is not compatible with the input BAM. This may produce inferior results.

Passing test: bacterial model is not compatible with the model listed in the input BAM file.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > ### Create the mocked data for a non-supported basecaller version (1.0.0).
  > samtools view -h data/in.micro.bam | sed -E 's/dna_r10.4.1_e8.2_400bps_hac@v5.0.0/dna_r10.4.1_e8.2_400bps_hac@v1.0.0/g' | samtools view -Sb > out/in.bam
  > samtools index out/in.bam
  > ### Run the unit under test.
  > ${DORADO_BIN} polish --bacteria --skip-model-compatibility-check --device cpu out/in.bam ${in_dir}/draft.fasta.gz -t 4 --regions "contig_1:1-100" --infer-threads 1 -vv > out/out.fasta 2> out/out.fasta.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 1
  [error] There are no bacterial models for the basecaller model: 'dna_r10.4.1_e8.2_400bps_hac@v1.0.0'.

Negative test: Cannot resolve the model, it does not match a Polishing model nor a path.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > model="unknown"
  > ### Run the unit under test.
  > ${DORADO_BIN} polish --device cpu data/in.micro.bam ${in_dir}/draft.fasta.gz -t 4 --regions "contig_1:1-100" --infer-threads 1 -vv --model-override "${model}" > out/out.fasta 2> out/out.fasta.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 1
  [error] Could not resolve model from string: 'unknown'.

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
  > # Prepare data. Strip off all tags.
  > samtools view -H ${in_bam} > out/in.sam
  > samtools view ${in_bam} | awk -v OFS='\t' '{ $11="*"; print }' >> out/in.sam
  > samtools view -Sb out/in.sam | samtools sort > out/in.bam
  > samtools index out/in.bam
  > # Run test.
  > ${DORADO_BIN} polish --skip-model-compatibility-check --device cpu out/in.bam ${in_dir}/draft.fasta.gz --regions "contig_1:1-100" -vv --model-override "dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl" > out/out.fasta 2> out/out.fasta.stderr
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
  [warning] Input data has move tables, but a model without move table support has been chosen. This may produce inferior results.

The input BAM is stripped of all tags (no move tables) but the specified model was trained using the move tables.
This should fail by default.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=data/in.micro.bam
  > # Prepare data. Strip off all tags.
  > samtools view -H ${in_bam} > out/in.sam
  > samtools view ${in_bam} | cut -f1-11 >> out/in.sam
  > samtools view -Sb out/in.sam | samtools sort > out/in.bam
  > samtools index out/in.bam
  > # Run test.
  > ${DORADO_BIN} polish --device cpu out/in.bam ${in_dir}/draft.fasta.gz --regions "contig_1:1-100" -vv --model-override "dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl_mv" > out/out.fasta 2> out/out.fasta.stderr
  > # Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 1
  [error] Input data does not contain move tables, but a model which requires move tables has been chosen.

The input BAM _has_ move tables, but the specified model was trained without the move tables.
This should fail by default.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=data/in.micro.bam
  > # Run test.
  > ${DORADO_BIN} polish --device cpu ${in_bam} ${in_dir}/draft.fasta.gz --regions "contig_1:1-100" -vv --model-override "dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl" > out/out.fasta 2> out/out.fasta.stderr
  > # Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 1
  [error] Input data has move tables, but a model without move table support has been chosen.

The input BAM is stripped of all tags (no move tables) but the specified model was trained using the move tables.
Permit this using `--skip-model-compatibility-check` but check for an emitted warning.
Results are worse than using the correct model, as expected.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=data/in.micro.bam
  > # Prepare data. Strip off all tags.
  > samtools view -H ${in_bam} > out/in.sam
  > samtools view ${in_bam} | cut -f1-11 >> out/in.sam
  > samtools view -Sb out/in.sam | samtools sort > out/in.bam
  > samtools index out/in.bam
  > # Run test.
  > ${DORADO_BIN} polish --device cpu out/in.bam ${in_dir}/draft.fasta.gz --skip-model-compatibility-check --regions "contig_1:1-100" -vv  --model-override "dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl_mv" > out/out.fasta 2> out/out.fasta.stderr
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
  [warning] Input data does not contain move tables, but a model which requires move tables has been chosen. This may produce inferior results.

The input BAM _has_ move tables, but the specified model was trained without the move tables.
Permit this using `--skip-model-compatibility-check` but check for an emitted warning.
Results are worse than using the correct model, as expected.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=data/in.micro.bam
  > # Run test.
  > ${DORADO_BIN} polish --device cpu ${in_bam} ${in_dir}/draft.fasta.gz  --skip-model-compatibility-check --regions "contig_1:1-100" -vv > out/out.fasta --model-override "dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl" 2> out/out.fasta.stderr
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
  [warning] Input data has move tables, but a model without move table support has been chosen. This may produce inferior results.
