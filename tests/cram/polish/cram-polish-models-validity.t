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

Config is not valid - there is a mix of legacy and read-level models in the `supported_models` list.
This should fail.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > ### Create a mock config.toml with multiple types of incompatible basecaller models listed.
  > mkdir -p out/models
  > cp -r ${MODEL_DIR} out/models/
  > cat ${MODEL_DIR}/config.toml | sed -E 's/supported_basecallers.*//g' | sed -E 's/\[model\].*/supported_basecallers = \[ "dna_r10.4.1_e8.2_400bps_hac@v5.0.0", "dna_r10.4.1_e8.2_400bps_hac@v4.2.0"\]\n\n\[model\]/g' > out/models/${MODEL_NAME}/config.toml
  > model_var="--models-directory out/models"
  > ### Run the unit under test.
  > ${DORADO_BIN} polish --device cpu data/in.micro.bam ${in_dir}/draft.fasta.gz ${model_var} -t 4 --regions "contig_1:1-100" -v > out/out.fasta 2> out/out.fasta.stderr
  > ### Eval.
  > echo "Exit code: $?"
  > grep "\[error\]" out/out.fasta.stderr | sed -E 's/.*\[error\] //g'
  Exit code: 1
  The model_config.supported_basecallers contains a mixture of legacy and new models. Is the model config ill defined?
