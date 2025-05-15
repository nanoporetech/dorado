###################################################
### Test auto-resolve for all available models  ###
### from the input BAM file.                    ###
###################################################
HAC. Auto-resolve the `dna_r10.4.1_e8.2_400bps_hac@v5.0.0_variant_mv@v1.0` model from the BAM file and download it.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > in_bam=${in_dir}/in.aln.bam
  > in_ref=${in_dir}/in.ref.fasta.gz
  > ${DORADO_BIN} variant --device cpu ${in_bam} ${in_ref} -t 4 -v > out/out.vcf 2> out/out.stderr
  > echo "Exit code: $?"
  > grep "Resolved model from input data: dna_r10.4.1_e8.2_400bps_hac@v5.0.0_variant_mv@v1.0" out/out.stderr | wc -l | awk '{ print $1 }'
  > grep "Downloading model" out/out.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.stderr | sed -E 's/.*\[error\] //g'
  Exit code: 0
  1
  1

SUP. Auto-resolve the `dna_r10.4.1_e8.2_400bps_sup@v5.0.0_variant_mv@v1.0` model from the BAM file and download it.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > ### Create the synthetic data with no dwells.
  > samtools view -h ${in_dir}/in.aln.bam | sed 's/dna_r10.4.1_e8.2_400bps_hac@v5.0.0/dna_r10.4.1_e8.2_400bps_sup@v5.0.0/g' | samtools view -Sb > out/in.modified.bam
  > samtools index out/in.modified.bam
  > ### Run the unit under test.
  > in_bam=out/in.modified.bam
  > in_ref=${in_dir}/in.ref.fasta.gz
  > ${DORADO_BIN} variant --device cpu ${in_bam} ${in_ref} -t 4 -v > out/out.vcf 2> out/out.stderr
  > echo "Exit code: $?"
  > grep "Resolved model from input data: dna_r10.4.1_e8.2_400bps_sup@v5.0.0_variant_mv@v1.0" out/out.stderr | wc -l | awk '{ print $1 }'
  > grep "Downloading model" out/out.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.stderr | sed -E 's/.*\[error\] //g'
  Exit code: 0
  1
  1

##############################################
### Test auto-resolve from the Basecaller  ###
### or Polishing model name and the dwell  ###
### info in the input BAM.                 ###
##############################################
Resolve the model from a Basecaller model name `dna_r10.4.1_e8.2_400bps_hac@v5.0.0`.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > in_bam=${in_dir}/in.aln.bam
  > in_ref=${in_dir}/in.ref.fasta.gz
  > model="dna_r10.4.1_e8.2_400bps_hac@v5.0.0"
  > ${DORADO_BIN} variant --model ${model} --device cpu ${in_bam} ${in_ref} -t 4 -v > out/out.vcf 2> out/out.stderr
  > echo "Exit code: $?"
  > grep "Resolved model from user-specified basecaller model name: dna_r10.4.1_e8.2_400bps_hac@v5.0.0_variant_mv@v1.0" out/out.stderr | wc -l | awk '{ print $1 }'
  > grep "Downloading model" out/out.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.stderr | sed -E 's/.*\[error\] //g'
  Exit code: 0
  1
  1

Resolve the model from an exact Variant Calling model name.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > in_bam=${in_dir}/in.aln.bam
  > in_ref=${in_dir}/in.ref.fasta.gz
  > model="dna_r10.4.1_e8.2_400bps_hac@v5.0.0_variant_mv@v1.0"
  > ${DORADO_BIN} variant --model ${model} --device cpu ${in_bam} ${in_ref} -t 4 -v > out/out.vcf 2> out/out.stderr
  > echo "Exit code: $?"
  > grep "Resolved model from user-specified variant calling model name: dna_r10.4.1_e8.2_400bps_hac@v5.0.0_variant_mv@v1.0" out/out.stderr | wc -l | awk '{ print $1 }'
  > grep "Downloading model" out/out.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.stderr | sed -E 's/.*\[error\] //g'
  Exit code: 0
  1
  1

Resolve the model from a local path.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > in_bam=${in_dir}/in.aln.bam
  > in_ref=${in_dir}/in.ref.fasta.gz
  > model=${MODEL_DIR}
  > ${DORADO_BIN} variant --model ${model} --device cpu ${in_bam} ${in_ref} -t 4 -v > out/out.vcf 2> out/out.stderr
  > echo "Exit code: $?"
  > grep "Resolved model from user-specified path: " out/out.stderr | wc -l | awk '{ print $1 }'
  > grep "Downloading model" out/out.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.stderr | sed -E 's/.*\[error\] //g'
  Exit code: 0
  1
  0

##############################################
### Compatibility checks.                  ###
##############################################
Negative test: no dwells in data, but the model uses them for polishing.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > model="dna_r10.4.1_e8.2_400bps_hac@v5.0.0_variant_mv@v1.0"
  > ### Create the synthetic data with no dwells.
  > samtools view -H ${in_dir}/in.aln.bam > out/in.no_dwells.sam
  > samtools view ${in_dir}/in.aln.bam | cut -f1-11 >> out/in.no_dwells.sam
  > samtools view -Sb out/in.no_dwells.sam > out/in.no_dwells.bam
  > samtools index out/in.no_dwells.bam
  > ### Run the unit under test.
  > in_bam=out/in.no_dwells.bam
  > in_ref=${in_dir}/in.ref.fasta.gz
  > ${DORADO_BIN} variant --model "${model}" --device cpu ${in_bam} ${in_ref} -t 4 --infer-threads 1 -vv > out/out.vcf 2> out/out.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "Resolved model from user-specified variant calling model name: dna_r10.4.1_e8.2_400bps_hac@v5.0.0_variant_mv@v1.0" out/out.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 1
  1
  [error] Caught exception: Input data does not contain move tables, but a model which requires move tables has been chosen.

Negative test: Basecaller model specified in the BAM does not match the Basecaller model specified in the Variant Calling model.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > model=${MODEL_DIR}
  > ### Create the mocked data for a non-supported basecaller version (1.0.0).
  > samtools view -h ${in_dir}/in.aln.bam | sed -E 's/dna_r10.4.1_e8.2_400bps_hac@v5.0.0/dna_r10.4.1_e8.2_400bps_hac@v1.0.0/g' | samtools view -Sb > out/in.bam
  > samtools index out/in.bam
  > ### Run the unit under test.
  > in_bam=out/in.bam
  > in_ref=${in_dir}/in.ref.fasta.gz
  > ${DORADO_BIN} variant --model ${model} --device cpu ${in_bam} ${in_ref} -t 4 --infer-threads 1 -vv > out/out.vcf 2> out/out.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 1
  [error] Caught exception: Variant calling model is not compatible with the input BAM!

Passing test with warnings: no dwells in data, but the model uses them for polishing.
Using `--skip-model-compatibility-check`.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > model="dna_r10.4.1_e8.2_400bps_hac@v5.0.0_variant_mv@v1.0"
  > ### Create the synthetic data with no dwells.
  > samtools view -H ${in_dir}/in.aln.bam > out/in.no_dwells.sam
  > samtools view ${in_dir}/in.aln.bam | cut -f1-11 >> out/in.no_dwells.sam
  > samtools view -Sb out/in.no_dwells.sam > out/in.no_dwells.bam
  > samtools index out/in.no_dwells.bam
  > ### Run the unit under test.
  > in_bam=out/in.no_dwells.bam
  > in_ref=${in_dir}/in.ref.fasta.gz
  > ${DORADO_BIN} variant --model "${model}" --skip-model-compatibility-check --device cpu ${in_bam} ${in_ref} -t 4 --infer-threads 1 -vv > out/out.vcf 2> out/out.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "Resolved model from user-specified variant calling model name: dna_r10.4.1_e8.2_400bps_hac@v5.0.0_variant_mv@v1.0" out/out.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  1
  [warning] Input data does not contain move tables, but a model which requires move tables has been chosen. This may produce inferior results.

Passing test with warnings: Basecaller model specified in the BAM does not match the Basecaller model specified in the Variant Calling model.
Using `--skip-model-compatibility-check`.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > model=${MODEL_DIR}
  > ### Create the mocked data for a non-supported basecaller version (1.0.0).
  > samtools view -h ${in_dir}/in.aln.bam | sed -E 's/dna_r10.4.1_e8.2_400bps_hac@v5.0.0/dna_r10.4.1_e8.2_400bps_hac@v1.0.0/g' | samtools view -Sb > out/in.bam
  > samtools index out/in.bam
  > ### Run the unit under test.
  > in_bam=out/in.bam
  > in_ref=${in_dir}/in.ref.fasta.gz
  > ${DORADO_BIN} variant --model ${model} --skip-model-compatibility-check --device cpu ${in_bam} ${in_ref} -t 4 --infer-threads 1 -vv > out/out.vcf 2> out/out.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  [warning] Variant calling model is not compatible with the input BAM. This may produce inferior results.

Negative test: A Polishing model is provided instead of a Variant Calling model. This should fail.
  $ rm -rf out; mkdir -p out
  > ${DORADO_BIN} download --model "dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl_mv" --models-directory out
  > model=out/dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl_mv
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > ### Run the unit under test.
  > in_bam=${in_dir}/in.aln.bam
  > in_ref=${in_dir}/in.ref.fasta.gz
  > ${DORADO_BIN} variant --model ${model} --device cpu ${in_bam} ${in_ref} -t 4 --infer-threads 1 -vv > out/out.vcf 2> out/out.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 1
  [error] Caught exception: Incompatible model label scheme! Expected DiploidLabelScheme but got HaploidLabelScheme.

Passing test with warnings: A Polishing model is provided instead of a Variant Calling model
Using `--skip-model-compatibility-check`.
  $ rm -rf out; mkdir -p out
  > ${DORADO_BIN} download --model "dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl_mv" --models-directory out
  > model=out/dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl_mv
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > ### Run the unit under test.
  > in_bam=${in_dir}/in.aln.bam
  > in_ref=${in_dir}/in.ref.fasta.gz
  > ${DORADO_BIN} variant --model ${model} --skip-model-compatibility-check --device cpu ${in_bam} ${in_ref} -t 4 --infer-threads 1 -vv > out/out.vcf 2> out/out.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  [warning] Incompatible model label scheme! Expected DiploidLabelScheme but got HaploidLabelScheme. This may produce unexpected results.

##############################################
### Negative tests.                        ###
##############################################
Negative test: Cannot resolve the model, it does not match a Basecaller model, a Variant Calling model, a path nor is it 'auto'.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > model="unknown"
  > ### Run the unit under test.
  > ${DORADO_BIN} variant --model "${model}" --device cpu ${in_dir}/in.aln.bam ${in_dir}/in.ref.fasta.gz -t 4 --infer-threads 1 -vv > out/out.vcf 2> out/out.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.stderr | sed -E 's/.*\[error\] //g'
  Exit code: 1
  Caught exception: Could not resolve model from string: 'unknown'.

Negative test: Empty model string provided.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > ### Run the unit under test.
  > ${DORADO_BIN} variant --model "" --device cpu ${in_dir}/in.aln.bam ${in_dir}/in.ref.fasta.gz -t 4 --infer-threads 1 -vv > out/out.vcf 2> out/out.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.stderr | sed -E 's/.*\[error\] //g'
  Exit code: 1
  Caught exception: Could not resolve model from string: ''.

Negative test: BAM has a model which is not available for download in auto mode.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > model="auto"
  > ### Create the mocked data for a non-supported basecaller version (1.0.0).
  > samtools view -h ${in_dir}/in.aln.bam | sed -E 's/dna_r10.4.1_e8.2_400bps_hac@v5.0.0/dna_r10.4.1_e8.2_400bps_hac@v1.0.0/g' | samtools view -Sb > out/in.bam
  > samtools index out/in.bam
  > ### Run the unit under test.
  > ${DORADO_BIN} variant --model "${model}" --device cpu out/in.bam ${in_dir}/in.ref.fasta.gz -t 4 --infer-threads 1 -vv > out/out.vcf 2> out/out.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.stderr | sed -E 's/.*\[error\] //g'
  Exit code: 1
  Caught exception: Could not find any variant calling model compatible with the basecaller model 'dna_r10.4.1_e8.2_400bps_hac@v1.0.0'.

Negative test: using 'auto' but the BAM has no models listed (no RG tags).
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > model="auto"
  > ### Create the mocked data, remove the @RG lines.
  > samtools view -h ${in_dir}/in.aln.bam | grep -v "@RG" | samtools view -Sb > out/in.bam
  > samtools index out/in.bam
  > ### Run the unit under test.
  > ${DORADO_BIN} variant --model "${model}" --device cpu out/in.bam ${in_dir}/in.ref.fasta.gz -t 4 --infer-threads 1 -vv > out/out.vcf 2> out/out.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.stderr | sed -E 's/.*\[error\] //g'
  Exit code: 1
  Caught exception: Input BAM file has no basecaller models listed in the header.

Negative test: Using `auto` and `--skip-model-compatibility-check`, but BAM has no models listed (no RG tags), so this should fail even without compatibility check.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > model="auto"
  > ### Create the mocked data, remove the @RG lines.
  > samtools view -h ${in_dir}/in.aln.bam | grep -v "@RG" | samtools view -Sb > out/in.bam
  > samtools index out/in.bam
  > ### Run the unit under test.
  > ${DORADO_BIN} variant --model "${model}" --skip-model-compatibility-check --device cpu out/in.bam ${in_dir}/in.ref.fasta.gz -t 4 --infer-threads 1 -vv > out/out.vcf 2> out/out.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.stderr | sed -E 's/.*\[error\] //g'
  Exit code: 1
  Caught exception: Input BAM file has no basecaller models listed in the header. Cannot use 'auto' to resolve the model.
