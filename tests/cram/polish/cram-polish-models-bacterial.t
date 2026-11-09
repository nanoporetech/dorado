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

Manually select the bacterial model v1.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > ### Create synthetic data with mocked model name.
  > model_var="--models-directory ${MODEL_ROOT_DIR} --model-override dna_r10.4.1_e8.2_400bps_polish_bacterial_methylation@v1.0"
  > ### Run the unit under test.
  > ${DORADO_BIN} polish --bacteria --device cpu data/in.micro.bam ${in_dir}/draft.fasta.gz ${model_var} -t 4 --regions "contig_1:1-100" -v > out/out.fasta 2> out/out.fasta.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "Resolved model" out/out.fasta.stderr | sed -E 's/.*\[debug\] //g'
  > grep "\- downloading" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[error\] //g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  Resolved model from user-specified polishing model name: dna_r10.4.1_e8.2_400bps_polish_bacterial_methylation@v1.0
  0
  [warning] Skipping basecaller compatibility checks for user-specified model override. The accuracy of the results is not guaranteed.

Manually select the bacterial model v2.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > ### Create synthetic data with mocked model name.
  > model_var="--models-directory ${MODEL_ROOT_DIR} --model-override dna_r10.4.1_e8.2_400bps_polish_bacterial_methylation@v2.0"
  > ### Run the unit under test.
  > ${DORADO_BIN} polish --bacteria --device cpu data/in.micro.bam ${in_dir}/draft.fasta.gz ${model_var} -t 4 --regions "contig_1:1-100" -v > out/out.fasta 2> out/out.fasta.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "Resolved model" out/out.fasta.stderr | sed -E 's/.*\[debug\] //g'
  > grep "\- downloading" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[error\] //g'
  > grep "\[warning\]" out/out.fasta.stderr | sed -E 's/.*\[/\[/g'
  Exit code: 0
  Resolved model from user-specified polishing model name: dna_r10.4.1_e8.2_400bps_polish_bacterial_methylation@v2.0
  0
  [warning] Skipping basecaller compatibility checks for user-specified model override. The accuracy of the results is not guaranteed.
