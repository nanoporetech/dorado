Test:
1. Computing the number of blocks.
2. Running a for loop to correct (with inference) each of those blocks individually.
  $ rm -rf out; mkdir -p out
  > in_reads=${TEST_DATA_DIR}/read_correction/reads.fq
  > model_var=${MODEL_DIR:+--model-path ${MODEL_DIR}}
  > num_blocks=$(${DORADO_BIN} correct ${in_reads} --device cpu -t 4 --compute-num-blocks -v --index-size 100000 2> out/out.compute_num_blocks.stderr)
  > echo ${num_blocks}
  > # Note: this would be run using a for loop, but some interpreters complain on the style of the for loop.
  > i=0; ${DORADO_BIN} correct ${in_reads} --device cpu -t 4 ${model_var} -v --index-size 100000 --run-block-id ${i} > out/out.block_${i}.fasta 2> out/out.block_${i}.fasta.stderr
  > i=1; ${DORADO_BIN} correct ${in_reads} --device cpu -t 4 ${model_var} -v --index-size 100000 --run-block-id ${i} > out/out.block_${i}.fasta 2> out/out.block_${i}.fasta.stderr
  > i=2; ${DORADO_BIN} correct ${in_reads} --device cpu -t 4 ${model_var} -v --index-size 100000 --run-block-id ${i} > out/out.block_${i}.fasta 2> out/out.block_${i}.fasta.stderr
  > i=3; ${DORADO_BIN} correct ${in_reads} --device cpu  -t 4 ${model_var} -v --index-size 100000 --run-block-id ${i} > out/out.block_${i}.fasta 2> out/out.block_${i}.fasta.stderr
  > ls -l out/*.fasta | wc -l | awk '{ print $1 }'
  > cat out/out.block_*.fasta > out/out.all.fasta
  > samtools faidx out/out.all.fasta
  > cat out/out.all.fasta.fai | cut -f1,1 | sort
  4
  4
  3855985e-bb9b-4df4-9825-cc08f373342b
  3c88104d-7964-43dc-ac47-c22b12cdc994
  73d5eb75-700e-42a0-9a5d-f9952bd7d829
  b93514e5-c61b-48d8-b730-f6c97d169ff7
  df814002-1961-4262-aaf5-e8f2760aa77a
  e3066d3e-2bdf-4803-89b9-0f077ac7ff7f

Test:
1. Computing the number of blocks.
2. Running a `for` loop to:
2a. Align those blocks individually without calling inference and writing those to individual PAF files.
2b. Call inference from individual PAF files.
3. Merge all FASTA corrected reads.
  $ rm -rf out; mkdir -p out
  > in_reads=${TEST_DATA_DIR}/read_correction/reads.fq
  > model_var=${MODEL_DIR:+--model-path ${MODEL_DIR}}
  > num_blocks=$(${DORADO_BIN} correct ${in_reads} --device cpu -t 4 --compute-num-blocks -v --index-size 100000 2> out/out.compute_num_blocks.stderr)
  > echo ${num_blocks}
  > # Note: this would be run using a for loop, but some interpreters complain on the style of the for loop.
  > i=0; ${DORADO_BIN} correct ${in_reads} --device cpu -t 4 ${model_var} -v --index-size 100000 --run-block-id ${i} --to-paf > out/out.block_${i}.paf 2> out/out.block_${i}.paf.stderr
  > i=0; ${DORADO_BIN} correct ${in_reads} --device cpu -t 4 ${model_var} -v --index-size 100000 --run-block-id ${i} --from-paf out/out.block_${i}.paf > out/out.block_${i}.fasta 2> out/out.block_${i}.fasta.stderr
  > i=1; ${DORADO_BIN} correct ${in_reads} --device cpu -t 4 ${model_var} -v --index-size 100000 --run-block-id ${i} --to-paf > out/out.block_${i}.paf 2> out/out.block_${i}.paf.stderr
  > i=1; ${DORADO_BIN} correct ${in_reads} --device cpu -t 4 ${model_var} -v --index-size 100000 --run-block-id ${i} --from-paf out/out.block_${i}.paf > out/out.block_${i}.fasta 2> out/out.block_${i}.fasta.stderr
  > i=2; ${DORADO_BIN} correct ${in_reads} --device cpu -t 4 ${model_var} -v --index-size 100000 --run-block-id ${i} --to-paf > out/out.block_${i}.paf 2> out/out.block_${i}.paf.stderr
  > i=2; ${DORADO_BIN} correct ${in_reads} --device cpu -t 4 ${model_var} -v --index-size 100000 --run-block-id ${i} --from-paf out/out.block_${i}.paf > out/out.block_${i}.fasta 2> out/out.block_${i}.fasta.stderr
  > i=3; ${DORADO_BIN} correct ${in_reads} --device cpu -t 4 ${model_var} -v --index-size 100000 --run-block-id ${i} --to-paf > out/out.block_${i}.paf 2> out/out.block_${i}.paf.stderr
  > i=3; ${DORADO_BIN} correct ${in_reads} --device cpu -t 4 ${model_var} -v --index-size 100000 --run-block-id ${i} --from-paf out/out.block_${i}.paf > out/out.block_${i}.fasta 2> out/out.block_${i}.fasta.stderr
  > ls -l out/*.fasta | wc -l | awk '{ print $1 }'
  > cat out/out.block_*.fasta > out/out.all.fasta
  > samtools faidx out/out.all.fasta
  > cat out/out.all.fasta.fai | cut -f1,1 | sort
  4
  4
  3855985e-bb9b-4df4-9825-cc08f373342b
  3c88104d-7964-43dc-ac47-c22b12cdc994
  73d5eb75-700e-42a0-9a5d-f9952bd7d829
  b93514e5-c61b-48d8-b730-f6c97d169ff7
  df814002-1961-4262-aaf5-e8f2760aa77a
  e3066d3e-2bdf-4803-89b9-0f077ac7ff7f

Test:
1. Computing the number of blocks.
2. Running a `for` loop to align those blocks individually without calling inference and writing those to individual PAF files.
3. Merging the PAF alignments.
4. Calling the inference.
  $ rm -rf out; mkdir -p out
  > in_reads=${TEST_DATA_DIR}/read_correction/reads.fq
  > model_var=${MODEL_DIR:+--model-path ${MODEL_DIR}}
  > num_blocks=$(${DORADO_BIN} correct ${in_reads} --device cpu -t 4 --compute-num-blocks -v --index-size 100000 2> out/out.compute_num_blocks.stderr)
  > echo ${num_blocks}
  > # Note: this would be run using a for loop, but some interpreters complain on the style of the for loop.
  > i=0; ${DORADO_BIN} correct ${in_reads} --device cpu -t 4 ${model_var} -v --index-size 100000 --run-block-id ${i} --to-paf > out/out.block_${i}.paf 2> out/out.block_${i}.paf.stderr
  > i=1; ${DORADO_BIN} correct ${in_reads} --device cpu -t 4 ${model_var} -v --index-size 100000 --run-block-id ${i} --to-paf > out/out.block_${i}.paf 2> out/out.block_${i}.paf.stderr
  > i=2; ${DORADO_BIN} correct ${in_reads} --device cpu -t 4 ${model_var} -v --index-size 100000 --run-block-id ${i} --to-paf > out/out.block_${i}.paf 2> out/out.block_${i}.paf.stderr
  > i=3; ${DORADO_BIN} correct ${in_reads} --device cpu -t 4 ${model_var} -v --index-size 100000 --run-block-id ${i} --to-paf > out/out.block_${i}.paf 2> out/out.block_${i}.paf.stderr
  > cat out/out.block_*.paf > out/out.all.paf
  > ${DORADO_BIN} correct ${in_reads} --device cpu -t 4 ${model_var} -v --index-size 100000 --run-block-id ${i} --from-paf out/out.all.paf > out/out.all.fasta 2> out/out.all.fasta.stderr
  > ls -l out/out.block_*.paf | wc -l | awk '{ print $1 }'
  > samtools faidx out/out.all.fasta
  > cat out/out.all.fasta.fai | cut -f1,1 | sort
  4
  4
  3855985e-bb9b-4df4-9825-cc08f373342b
  3c88104d-7964-43dc-ac47-c22b12cdc994
  73d5eb75-700e-42a0-9a5d-f9952bd7d829
  b93514e5-c61b-48d8-b730-f6c97d169ff7
  df814002-1961-4262-aaf5-e8f2760aa77a
  e3066d3e-2bdf-4803-89b9-0f077ac7ff7f
