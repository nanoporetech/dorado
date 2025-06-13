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

###################################################
### Test auto-resolve from the input BAM file.  ###
###################################################
HAC, with dwells. Auto-resolve the `dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl_mv` model from the BAM file and download it.
No need to test all available models exhaustively, there are separate Cram files for that.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > ${DORADO_BIN} polish --device cpu data/in.micro.bam ${in_dir}/draft.fasta.gz -t 4 -v > out/out.fasta 2> out/out.fasta.stderr
  > echo "Exit code: $?"
  > grep "Resolved model from input data: dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl_mv" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "Downloading model" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  1
  1

###################################################
### Test auto-resolve from the Basecaller model ###
### name, Polishing model name or a path.       ###
###################################################
Resolve the model from a Basecaller model name `dna_r10.4.1_e8.2_400bps_hac@v5.0.0` + dwell info from the BAM.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > model="dna_r10.4.1_e8.2_400bps_hac@v5.0.0"
  > ### Run the unit under test.
  > ${DORADO_BIN} polish --model "${model}" --device cpu data/in.micro.bam ${in_dir}/draft.fasta.gz -t 4 --infer-threads 1 -v > out/out.fasta 2> out/out.fasta.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "Resolved model from user-specified basecaller model name: dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl_mv" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "Downloading model" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  1
  1

Resolve the model from an exact Polishing model name: `dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl_mv`.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > model="dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl_mv"
  > ### Run the unit under test.
  > ${DORADO_BIN} polish --model "${model}" --device cpu data/in.micro.bam ${in_dir}/draft.fasta.gz -t 4 --infer-threads 1 -v > out/out.fasta 2> out/out.fasta.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "Resolved model from user-specified polishing model name: dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl_mv" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "Downloading model" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
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
  > ${DORADO_BIN} polish --model "${model}" --device cpu data/in.micro.bam ${in_dir}/draft.fasta.gz -t 4 --infer-threads 1 -v > out/out.fasta 2> out/out.fasta.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "Resolved model from user-specified polishing model name: " out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "Downloading model" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
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
  > ${DORADO_BIN} polish --bacteria --model "${model}" --device cpu data/in.micro.bam ${in_dir}/draft.fasta.gz -t 4 --infer-threads 1 -v > out/out.fasta 2> out/out.fasta.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "Resolved model from user-specified basecaller model name: dna_r10.4.1_e8.2_400bps_polish_bacterial_methylation_v5.0.0" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "Downloading model" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  1
  1

##############################################
### Compatibility checks.                  ###
##############################################
Negative test: data has dwells, but the model does not support them.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > model="dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl"
  > ### Run the unit under test.
  > ${DORADO_BIN} polish --model "${model}" --device cpu data/in.micro.bam ${in_dir}/draft.fasta.gz -t 4 --infer-threads 1 -vv > out/out.fasta 2> out/out.fasta.stderr
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
  > ${DORADO_BIN} polish --model "${model}" --device cpu out/in.no_dwells.bam ${in_dir}/draft.fasta.gz -t 4 --infer-threads 1 -vv > out/out.fasta 2> out/out.fasta.stderr
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
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > ### Create the mocked data for a non-supported basecaller version (1.0.0).
  > samtools view -h data/in.micro.bam | sed -E 's/dna_r10.4.1_e8.2_400bps_hac@v5.0.0/dna_r10.4.1_e8.2_400bps_hac@v1.0.0/g' | samtools view -Sb > out/in.bam
  > samtools index out/in.bam
  > ### Run the unit under test.
  > ${DORADO_BIN} polish ${model_var} --device cpu out/in.bam ${in_dir}/draft.fasta.gz -t 4 --infer-threads 1 -vv > out/out.fasta 2> out/out.fasta.stderr
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
  > ${DORADO_BIN} polish --bacteria --device cpu out/in.bam ${in_dir}/draft.fasta.gz -t 4 --infer-threads 1 -vv > out/out.fasta 2> out/out.fasta.stderr
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
  > ${DORADO_BIN} polish --model "${model}" --skip-model-compatibility-check --device cpu data/in.micro.bam ${in_dir}/draft.fasta.gz -t 4 --infer-threads 1 -vv > out/out.fasta 2> out/out.fasta.stderr
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
  > ${DORADO_BIN} polish --model "${model}" --skip-model-compatibility-check --device cpu out/in.no_dwells.bam ${in_dir}/draft.fasta.gz -t 4 --infer-threads 1 -vv > out/out.fasta 2> out/out.fasta.stderr
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
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > ### Create the mocked data for a non-supported basecaller version (1.0.0).
  > samtools view -h data/in.micro.bam | sed -E 's/dna_r10.4.1_e8.2_400bps_hac@v5.0.0/dna_r10.4.1_e8.2_400bps_hac@v1.0.0/g' | samtools view -Sb > out/in.bam
  > samtools index out/in.bam
  > ### Run the unit under test.
  > ${DORADO_BIN} polish ${model_var} --skip-model-compatibility-check --device cpu out/in.bam ${in_dir}/draft.fasta.gz -t 4 --infer-threads 1 -vv > out/out.fasta 2> out/out.fasta.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  [warning] Polishing model is not compatible with the input BAM. This may produce inferior results.

Passing test: bacterial model is not compatible with the model listed in the input BAM file, but still runs.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > ### Create the mocked data for a non-supported basecaller version (1.0.0).
  > samtools view -h data/in.micro.bam | sed -E 's/dna_r10.4.1_e8.2_400bps_hac@v5.0.0/dna_r10.4.1_e8.2_400bps_hac@v1.0.0/g' | samtools view -Sb > out/in.bam
  > samtools index out/in.bam
  > ### Run the unit under test.
  > ${DORADO_BIN} polish --bacteria --skip-model-compatibility-check --model "dna_r10.4.1_e8.2_400bps_hac@v5.0.0" --device cpu out/in.bam ${in_dir}/draft.fasta.gz -t 4 --infer-threads 1 -vv > out/out.fasta 2> out/out.fasta.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  [warning] Polishing model is not compatible with the input BAM. This may produce inferior results.

##############################################
### Negative tests.                        ###
##############################################
Negative test: Cannot resolve the model, it does not match a Basecaller model, a Polishing model, a path nor is it 'auto'.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > model="unknown"
  > ### Run the unit under test.
  > ${DORADO_BIN} polish --model "${model}" --device cpu data/in.micro.bam ${in_dir}/draft.fasta.gz -t 4 --infer-threads 1 -vv > out/out.fasta 2> out/out.fasta.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 1
  [error] Could not resolve model from string: 'unknown'.

Negative test: Empty model string provided.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > ### Run the unit under test.
  > ${DORADO_BIN} polish --model "" --device cpu data/in.micro.bam ${in_dir}/draft.fasta.gz -t 4 --infer-threads 1 -vv > out/out.fasta 2> out/out.fasta.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 1
  [error] Could not resolve model from string: ''.

Negative test: BAM has a model which is not available for download in auto mode.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > model="auto"
  > ### Create the mocked data for a non-supported basecaller version (1.0.0).
  > samtools view -h data/in.micro.bam | sed -E 's/dna_r10.4.1_e8.2_400bps_hac@v5.0.0/dna_r10.4.1_e8.2_400bps_hac@v1.0.0/g' | samtools view -Sb > out/in.bam
  > samtools index out/in.bam
  > ### Run the unit under test.
  > ${DORADO_BIN} polish --model "${model}" --device cpu out/in.bam ${in_dir}/draft.fasta.gz -t 4 --infer-threads 1 -vv > out/out.fasta 2> out/out.fasta.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 1
  [error] There are no polishing models for the basecaller model: 'dna_r10.4.1_e8.2_400bps_hac@v1.0.0'.

Negative test: using 'auto' but the BAM has no models listed (no RG tags).
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > model="auto"
  > ### Create the mocked data, remove the @RG lines.
  > samtools view -h data/in.micro.bam | grep -v "@RG" | samtools view -Sb > out/in.bam
  > samtools index out/in.bam
  > ### Run the unit under test.
  > ${DORADO_BIN} polish --model "${model}" --device cpu out/in.bam ${in_dir}/draft.fasta.gz -t 4 --infer-threads 1 -vv > out/out.fasta 2> out/out.fasta.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 1
  [error] Input BAM file has no basecaller models listed in the header.

Negative test: Using `auto` and `--skip-model-compatibility-check`, but BAM has no models listed (no RG tags), so this should fail even without compatibility check.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > model="auto"
  > ### Create the mocked data, remove the @RG lines.
  > samtools view -h data/in.micro.bam | grep -v "@RG" | samtools view -Sb > out/in.bam
  > samtools index out/in.bam
  > ### Run the unit under test.
  > ${DORADO_BIN} polish --model "${model}" --skip-model-compatibility-check --device cpu out/in.bam ${in_dir}/draft.fasta.gz -t 4 --infer-threads 1 -vv > out/out.fasta 2> out/out.fasta.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 1
  [error] Input BAM file has no basecaller models listed in the header. Cannot use 'auto' to resolve the model.
