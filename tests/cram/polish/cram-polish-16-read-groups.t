Create synthetic test data with 2 read groups. It will be used by tests below. This block does not run any tests.
  $ rm -rf data; mkdir -p data
  > in_dir="${TEST_DATA_DIR}/polish/test-01-supertiny"
  > in_bam="${in_dir}/calls_to_draft.bam"
  > ### Create a BAM file with zero read groups.
  > samtools view -H ${in_bam} | grep -v "@RG" > data/in.01.zero_read_groups.sam
  > samtools view ${in_bam} | head -n 1 >> data/in.01.zero_read_groups.sam
  > samtools view -Sb data/in.01.zero_read_groups.sam | samtools sort > data/in.01.zero_read_groups.bam
  > samtools index data/in.01.zero_read_groups.bam
  > ###
  > ### Create a BAM file with two read groups and one basecaller.
  > # First, make tmp0.bam which is the same as ${in_bam} but contains only one record.
  > samtools view -H ${in_bam} > data/tmp0.sam
  > samtools view ${in_bam} | head -n 1 >> data/tmp0.sam
  > samtools view -Sb data/tmp0.sam > data/tmp0.bam
  > # Next, create tmp.bam which contains the same header+record but with a changed read group.
  > samtools view -h data/tmp0.bam | sed -E 's/bc8993f4557dd53bf0cbda5fd68453fea5e94485_dna_r10.4.1_e8.2_400bps_hac@v5.0.0-3AD2AF3E/f9f2e0901209274f132de3554913a1dc0a4439ae_dna_r10.4.1_e8.2_400bps_hac@v5.0.0-2DEAC5EC/g' > data/tmp.sam
  > samtools view -Sb data/tmp.sam | samtools sort > data/tmp.bam
  > # Merge the two.
  > samtools merge data/in.02.two_read_groups.one_basecaller.bam data/tmp0.bam data/tmp.bam
  > samtools index data/in.02.two_read_groups.one_basecaller.bam
  > ###
  > ### Create a BAM file with two read groups and two basecallers.
  > # Create a tmp2.bam from the existing tmp.sam, this one containing a different basecaller ID.
  > cat data/tmp.sam | sed -E 's/dna_r10.4.1_e8.2_400bps_hac@v5.0.0/dna_r10.0.0_e8.2_400bps_hac@v4.3.0/g' > data/tmp2.sam
  > samtools view -Sb data/tmp2.sam | samtools sort > data/tmp2.bam
  > samtools merge data/in.03.two_read_groups.two_basecallers.bam data/tmp0.bam data/tmp2.bam
  > samtools index data/in.03.two_read_groups.two_basecallers.bam

Zero read groups in the input and auto model selection.
Fails because the basecaller model cannot be determined.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam="data/in.01.zero_read_groups.bam"
  > # Run test.
  > ${DORADO_BIN} polish --device cpu ${in_bam} ${in_dir}/draft.fasta.gz -vv > out/out.fasta 2> out/out.fasta.stderr
  > # Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 1
  [error] Input BAM file has no basecaller models listed in the header.

Zero read groups in the input and auto model selection.
Fails even with `--skip-model-compatibility-check`, because a model cannot be auto resolved with this feature.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam="data/in.01.zero_read_groups.bam"
  > # Run test.
  > ${DORADO_BIN} polish --skip-model-compatibility-check --device cpu ${in_bam} ${in_dir}/draft.fasta.gz -vv > out/out.fasta 2> out/out.fasta.stderr
  > # Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 1
  [error] Input BAM file has no basecaller models listed in the header. Cannot use 'auto' to resolve the model.

Zero read groups in the input, and a model is specified by path.
Fails because it still cannot match the model to the RG basecaller name in the BAM file (since there are no @RG header lines).
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam="data/in.01.zero_read_groups.bam"
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > # Run test.
  > ${DORADO_BIN} polish --device cpu ${in_bam} ${in_dir}/draft.fasta.gz ${model_var} -vv > out/out.fasta 2> out/out.fasta.stderr
  > # Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 1
  [error] Input BAM file has no basecaller models listed in the header.

Zero read groups in the input, and a model is specified by path.
Allowed with`--skip-model-compatibility-check`.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam="data/in.01.zero_read_groups.bam"
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > # Run test.
  > ${DORADO_BIN} polish --skip-model-compatibility-check --device cpu ${in_bam} ${in_dir}/draft.fasta.gz ${model_var} -v > out/out.fasta 2> out/out.fasta.stderr
  > # Eval.
  > echo "Exit code: $?"
  > samtools faidx out/out.fasta
  > awk '{ print $1,$2 }' out/out.fasta.fai
  > grep "Copying contig verbatim from input" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  contig_1 9996
  0

Zero read groups in the input and `--ignore-read-groups`.
This does not help because this option only ignores RG checking after model selection. Model still cannot be selected.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam="data/in.01.zero_read_groups.bam"
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > # Run test.
  > ${DORADO_BIN} polish --ignore-read-groups --device cpu ${in_bam} ${in_dir}/draft.fasta.gz ${model_var} -vv > out/out.fasta 2> out/out.fasta.stderr
  > # Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 1
  [error] Input BAM file has no basecaller models listed in the header.

Zero read groups in the input and `--RG` option provided.
Fails because a read group was selected, but there are no read groups in the BAM file.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam="data/in.01.zero_read_groups.bam"
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > # Run test.
  > ${DORADO_BIN} polish --RG "bc8993f4557dd53bf0cbda5fd68453fea5e94485_dna_r10.4.1_e8.2_400bps_hac@v5.0.0-3AD2AF3E" --device cpu ${in_bam} ${in_dir}/draft.fasta.gz ${model_var} -vv > out/out.fasta 2> out/out.fasta.stderr
  > # Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 1
  [error] No @RG headers found in the input BAM, but user-specified RG was given. RG: 'bc8993f4557dd53bf0cbda5fd68453fea5e94485_dna_r10.4.1_e8.2_400bps_hac@v5.0.0-3AD2AF3E'

Zero read groups in the input, and both `--RG` and `--ignore-read-groups` options provided.
Fails because a read group was selected, but there are no read groups in the BAM file.
The option `--ignore-read-groups` cannot help if a specific read group was requested and it does not exist.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam="data/in.01.zero_read_groups.bam"
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > # Run test.
  > ${DORADO_BIN} polish --ignore-read-groups --RG "bc8993f4557dd53bf0cbda5fd68453fea5e94485_dna_r10.4.1_e8.2_400bps_hac@v5.0.0-3AD2AF3E" --device cpu ${in_bam} ${in_dir}/draft.fasta.gz ${model_var} -vv > out/out.fasta 2> out/out.fasta.stderr
  > # Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 1
  [error] No @RG headers found in the input BAM, but user-specified RG was given. RG: 'bc8993f4557dd53bf0cbda5fd68453fea5e94485_dna_r10.4.1_e8.2_400bps_hac@v5.0.0-3AD2AF3E'

Two read groups are present in the input BAM, but only one basecaller model.
Fails because only one RG is allowed.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam="data/in.02.two_read_groups.one_basecaller.bam"
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > # Run test.
  > ${DORADO_BIN} polish --device cpu ${in_bam} ${in_dir}/draft.fasta.gz ${model_var} -vv > out/out.fasta 2> out/out.fasta.stderr
  > # Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 1
  [error] The input BAM contains more than one read group. Please specify --RG to select which read group to process.

Two read groups are present in the input BAM, but only one basecaller model.
Allow multiple read groups with `--ignore-read-groups`.
This will succeed only because both read groups have the same basecaller model.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam="data/in.02.two_read_groups.one_basecaller.bam"
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > # Run test.
  > ${DORADO_BIN} polish --ignore-read-groups --device cpu ${in_bam} ${in_dir}/draft.fasta.gz ${model_var} -vv > out/out.fasta 2> out/out.fasta.stderr
  > # Eval.
  > echo "Exit code: $?"
  > samtools faidx out/out.fasta
  > awk '{ print $1,$2 }' out/out.fasta.fai
  > grep "Copying contig verbatim from input" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  contig_1 9996
  0

Two read groups are present in the input BAM, but only one basecaller model.
Select one read group with the `--RG` tag.
This will succeed only because both read groups have the same basecaller model.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam="data/in.02.two_read_groups.one_basecaller.bam"
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > # Run test.
  > ${DORADO_BIN} polish --RG "bc8993f4557dd53bf0cbda5fd68453fea5e94485_dna_r10.4.1_e8.2_400bps_hac@v5.0.0-3AD2AF3E" --device cpu ${in_bam} ${in_dir}/draft.fasta.gz ${model_var} -vv > out/out.fasta 2> out/out.fasta.stderr
  > # Eval.
  > echo "Exit code: $?"
  > samtools faidx out/out.fasta
  > awk '{ print $1,$2 }' out/out.fasta.fai
  > grep "Copying contig verbatim from input" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  contig_1 9996
  0

Two read groups are present in the input BAM, but only one basecaller model.
Select one read group with the `--RG` tag, but this read group does not exist.
Fails because the requested read group does not exist.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam="data/in.02.two_read_groups.one_basecaller.bam"
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > # Run test.
  > ${DORADO_BIN} polish --RG "nonexistent-read-group" --device cpu ${in_bam} ${in_dir}/draft.fasta.gz ${model_var} -vv > out/out.fasta 2> out/out.fasta.stderr
  > # Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 1
  [error] No @RG headers found in the input BAM, but user-specified RG was given. RG: 'nonexistent-read-group'

Two read groups are present in the input BAM, and two basecaller models (one for each read group).
There is a cascade of exceptions that would be triggered in this case.
Fails because only one RG is allowed.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam="data/in.03.two_read_groups.two_basecallers.bam"
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > # Run test.
  > ${DORADO_BIN} polish --device cpu ${in_bam} ${in_dir}/draft.fasta.gz ${model_var} -vv > out/out.fasta 2> out/out.fasta.stderr
  > # Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 1
  [error] The input BAM contains more than one read group. Please specify --RG to select which read group to process.

Two read groups are present in the input BAM, and two basecaller models (one for each read group).
Ignore the RG check using `--ignore-read-groups`.
Fails because multiple basecaller models are present.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam="data/in.03.two_read_groups.two_basecallers.bam"
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > # Run test.
  > ${DORADO_BIN} polish --ignore-read-groups --device cpu ${in_bam} ${in_dir}/draft.fasta.gz ${model_var} -vv > out/out.fasta 2> out/out.fasta.stderr
  > # Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 1
  [error] Input BAM file has a mix of different basecaller models. Only one basecaller model can be processed.

Two read groups are present in the input BAM, and two basecaller models (one for each read group).
Ignore the RG check using `--ignore-read-groups`.
Passes because `--skip-model-compatibility-check` is used.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam="data/in.03.two_read_groups.two_basecallers.bam"
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > # Run test.
  > ${DORADO_BIN} polish --skip-model-compatibility-check --ignore-read-groups --device cpu ${in_bam} ${in_dir}/draft.fasta.gz ${model_var} -vv > out/out.fasta 2> out/out.fasta.stderr
  > # Eval.
  > echo "Exit code: $?"
  > samtools faidx out/out.fasta
  > awk '{ print $1,$2 }' out/out.fasta.fai
  > grep "Copying contig verbatim from input" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  contig_1 9996
  0
  [warning] Polishing model is not compatible with the input BAM. This may produce inferior results.

Two read groups are present in the input BAM, and two basecaller models (one for each read group).
Select one read group with `--RG`, which should automatically resolve the model for that group.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam="data/in.03.two_read_groups.two_basecallers.bam"
  > # Run test.
  > ${DORADO_BIN} polish --RG "bc8993f4557dd53bf0cbda5fd68453fea5e94485_dna_r10.4.1_e8.2_400bps_hac@v5.0.0-3AD2AF3E" --device cpu ${in_bam} ${in_dir}/draft.fasta.gz -vv > out/out.fasta 2> out/out.fasta.stderr
  > # Eval.
  > echo "Exit code: $?"
  > samtools faidx out/out.fasta
  > awk '{ print $1,$2 }' out/out.fasta.fai
  > grep "Copying contig verbatim from input" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  contig_1 9996
  0

Two read groups are present in the input BAM, and two basecaller models (one for each read group).
Select one read group with `--RG`, which shoud pass because the Basecaller model for that group matches the Polishing model info.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam="data/in.03.two_read_groups.two_basecallers.bam"
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > # Run test.
  > ${DORADO_BIN} polish --RG "bc8993f4557dd53bf0cbda5fd68453fea5e94485_dna_r10.4.1_e8.2_400bps_hac@v5.0.0-3AD2AF3E" --device cpu ${in_bam} ${in_dir}/draft.fasta.gz ${model_var} -vv > out/out.fasta 2> out/out.fasta.stderr
  > # Eval.
  > echo "Exit code: $?"
  > samtools faidx out/out.fasta
  > awk '{ print $1,$2 }' out/out.fasta.fai
  > grep "Copying contig verbatim from input" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  contig_1 9996
  0

Two read groups are present in the input BAM, and two basecaller models (one for each read group).
Select the second read group with `--RG`.
Fails because the basecaller model for the second group is not supported (per test design).
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam="data/in.03.two_read_groups.two_basecallers.bam"
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > # Run test.
  > ${DORADO_BIN} polish --RG "f9f2e0901209274f132de3554913a1dc0a4439ae_dna_r10.0.0_e8.2_400bps_hac@v4.3.0-2DEAC5EC" --device cpu ${in_bam} ${in_dir}/draft.fasta.gz ${model_var} -vv > out/out.fasta 2> out/out.fasta.stderr
  > # Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 1
  [error] Polishing model is not compatible with the input BAM!

Two read groups are present in the input BAM, and two basecaller models (one for each read group).
Select the second read group with `--RG`.
Passes because `--skip-model-compatibility-check` is used.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam="data/in.03.two_read_groups.two_basecallers.bam"
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > # Run test.
  > ${DORADO_BIN} polish --skip-model-compatibility-check --RG "f9f2e0901209274f132de3554913a1dc0a4439ae_dna_r10.0.0_e8.2_400bps_hac@v4.3.0-2DEAC5EC" --device cpu ${in_bam} ${in_dir}/draft.fasta.gz ${model_var} -vv > out/out.fasta 2> out/out.fasta.stderr
  > # Eval.
  > echo "Exit code: $?"
  > samtools faidx out/out.fasta
  > awk '{ print $1,$2 }' out/out.fasta.fai
  > grep "Copying contig verbatim from input" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  contig_1 9996
  0
  [warning] Polishing model is not compatible with the input BAM. This may produce inferior results.