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

The input BAM is stripped of all tags.
Auto-resolve the model The one without move tables should be used.
The output should be as good as it gets with other tests.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=data/in.micro.bam
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > # Prepare data. Strip off all tags.
  > samtools view -H ${in_bam} > out/in.sam
  > samtools view ${in_bam} | cut -f1-11 >> out/in.sam
  > samtools view -Sb out/in.sam | samtools sort > out/in.bam
  > samtools index out/in.bam
  > # Run test.
  > ${DORADO_BIN} polish --device cpu out/in.bam ${in_dir}/draft.fasta.gz -vv > out/out.fasta 2> out/out.fasta.stderr
  > # Eval.
  > echo "Exit code: $?"
  > samtools faidx out/out.fasta
  > awk '{ print $1,$2 }' out/out.fasta.fai
  > grep "Copying contig verbatim from input" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  contig_1 9993
  0

Quality field in the BAM is set to `*` instead of actual qualities (valid according to the SAM format specification).
All other custom tags are left inside.
Auto-resolve the model (the one with move tables will be chosen).
Results will be worse than using QVs, but this input should successfully run nonetheless.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=data/in.micro.bam
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > # Prepare data. Strip off all tags.
  > samtools view -H ${in_bam} > out/in.sam
  > samtools view ${in_bam} | awk -v OFS='\t' '{ $11="*"; print }' >> out/in.sam
  > samtools view -Sb out/in.sam | samtools sort > out/in.bam
  > samtools index out/in.bam
  > # Run test.
  > ${DORADO_BIN} polish --device cpu out/in.bam ${in_dir}/draft.fasta.gz -vv > out/out.fasta 2> out/out.fasta.stderr
  > # Eval.
  > echo "Exit code: $?"
  > samtools faidx out/out.fasta
  > awk '{ print $1,$2 }' out/out.fasta.fai
  > grep "Copying contig verbatim from input" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  contig_1 9990
  0

Quality field in the BAM is set to `*` instead of actual qualities (valid according to the SAM format specification).
All other custom tags are left inside.
Use the model without move tables.
Results will be worse than using QVs, but this input should successfully run nonetheless.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=data/in.micro.bam
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > # Prepare data. Strip off all tags.
  > samtools view -H ${in_bam} > out/in.sam
  > samtools view ${in_bam} | awk -v OFS='\t' '{ $11="*"; print }' >> out/in.sam
  > samtools view -Sb out/in.sam | samtools sort > out/in.bam
  > samtools index out/in.bam
  > # Run test.
  > ${DORADO_BIN} polish --model "dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl" --skip-model-compatibility-check --device cpu out/in.bam ${in_dir}/draft.fasta.gz -vv > out/out.fasta 2> out/out.fasta.stderr
  > # Eval.
  > echo "Exit code: $?"
  > samtools faidx out/out.fasta
  > awk '{ print $1,$2 }' out/out.fasta.fai
  > grep "Copying contig verbatim from input" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  contig_1 10003
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
  > ${DORADO_BIN} polish --device cpu out/in.bam ${in_dir}/draft.fasta.gz --model "dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl_mv" -vv > out/out.fasta 2> out/out.fasta.stderr
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
  > ${DORADO_BIN} polish --device cpu ${in_bam} ${in_dir}/draft.fasta.gz --model "dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl" -vv > out/out.fasta 2> out/out.fasta.stderr
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
  > ${DORADO_BIN} polish --device cpu out/in.bam ${in_dir}/draft.fasta.gz --model "dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl_mv" --skip-model-compatibility-check -vv > out/out.fasta 2> out/out.fasta.stderr
  > # Eval.
  > echo "Exit code: $?"
  > samtools faidx out/out.fasta
  > awk '{ print $1,$2 }' out/out.fasta.fai
  > grep "Copying contig verbatim from input" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  contig_1 10002
  0
  [warning] Input data does not contain move tables, but a model which requires move tables has been chosen. This may produce inferior results.

The input BAM _has_ move tables, but the specified model was trained without the move tables.
Permit this using `--skip-model-compatibility-check` but check for an emitted warning.
Results are worse than using the correct model, as expected.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=data/in.micro.bam
  > # Run test.
  > ${DORADO_BIN} polish --device cpu ${in_bam} ${in_dir}/draft.fasta.gz --model "dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl" --skip-model-compatibility-check -vv > out/out.fasta 2> out/out.fasta.stderr
  > # Eval.
  > echo "Exit code: $?"
  > samtools faidx out/out.fasta
  > awk '{ print $1,$2 }' out/out.fasta.fai
  > grep "Copying contig verbatim from input" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  contig_1 9993
  0
  [warning] Input data has move tables, but a model without move table support has been chosen. This may produce inferior results.
