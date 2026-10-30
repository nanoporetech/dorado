Provide candidate variant sites to seed inference windows (the --candidates feature).
  $ rm -rf out; mkdir -p out
  > in_dir_1=${TEST_DATA_DIR}/variant/test-02-supertiny
  > in_dir_2=${TEST_DATA_DIR}/variant/test-03-kadayashi-varcall
  > in_bam=${in_dir_1}/in.aln.bam
  > in_ref=${in_dir_1}/in.ref.fasta.gz
  > in_candidates=${in_dir_2}/in.varcall.unsr.list
  > in_expected=${in_dir_2}/expected.varcall.dorado.no_regions.vcf
  > in_expected_proc_regions=${in_dir_2}/expected.processed_regions.no_regions.bed
  > model_var=${MODEL_ROOT_DIR:+--models-directory ${MODEL_ROOT_DIR}}
  > ${DORADO_BIN} smallvar -vv --device cpu ${in_bam} ${in_ref} -t 4 ${model_var} --candidate-filtering true --candidate-centered-regions --candidates ${in_candidates} --window-len 300 --window-overlap 100 --variant-flanking-bases 100 --ignore-read-groups -o out 2> out/stderr
  > echo "Exit code: $?"
  > grep "\[error\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/stderr | sed -E 's/.*\[/\[/g'
  > ls -1 out
  > ### Remove the qual field because Torch results can vary slightly cross-platform.
  > cat ${in_expected} | grep -v "#" | cut -f 1-5,7-8 > out/expected.no_header.no_qual.vcf
  > gunzip -c out/variants.vcf.gz | grep -v "#" | cut -f 1-5,7-8 > out/result.no_header.no_qual.vcf
  > diff out/expected.no_header.no_qual.vcf out/result.no_header.no_qual.vcf
  > diff ${in_expected_proc_regions} out/processed_regions.bed
  Exit code: 0
  processed_regions.bed
  stderr
  variants.vcf.gz
  variants.vcf.gz.tbi

Additionally provide a regions BED file to limit where the candidate sites are considered for inference.
  $ rm -rf out; mkdir -p out
  > in_dir_1=${TEST_DATA_DIR}/variant/test-02-supertiny
  > in_dir_2=${TEST_DATA_DIR}/variant/test-03-kadayashi-varcall
  > in_bam=${in_dir_1}/in.aln.bam
  > in_ref=${in_dir_1}/in.ref.fasta.gz
  > in_candidate_bed=${in_dir_2}/in.varcall.bed
  > in_candidates=${in_dir_2}/in.varcall.unsr.list
  > in_expected=${in_dir_2}/expected.varcall.dorado.with_regions.vcf
  > in_expected_proc_regions=${in_dir_2}/expected.processed_regions.with_regions.bed
  > model_var=${MODEL_ROOT_DIR:+--models-directory ${MODEL_ROOT_DIR}}
  > ${DORADO_BIN} smallvar -vv --device cpu ${in_bam} ${in_ref} -t 4 ${model_var} --regions ${in_candidate_bed} --candidate-filtering true --candidate-centered-regions --candidates ${in_candidates} --window-len 300 --window-overlap 100 --variant-flanking-bases 100 --ignore-read-groups -o out 2> out/stderr
  > echo "Exit code: $?"
  > grep "\[error\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/stderr | sed -E 's/.*\[/\[/g'
  > ls -1 out
  > ### Remove the qual field because Torch results can vary slightly cross-platform.
  > cat ${in_expected} | grep -v "#" | cut -f 1-5,7-8 > out/expected.no_header.no_qual.vcf
  > gunzip -c out/variants.vcf.gz | grep -v "#" | cut -f 1-5,7-8 > out/result.no_header.no_qual.vcf
  > diff out/expected.no_header.no_qual.vcf out/result.no_header.no_qual.vcf
  > diff ${in_expected_proc_regions} out/processed_regions.bed
  Exit code: 0
  processed_regions.bed
  stderr
  variants.vcf.gz
  variants.vcf.gz.tbi

Tiled region selection is the default windowing approach for candidate-filtered inference.
  $ rm -rf out; mkdir -p out
  > in_dir_1=${TEST_DATA_DIR}/variant/test-02-supertiny
  > in_dir_2=${TEST_DATA_DIR}/variant/test-03-kadayashi-varcall
  > in_bam=${in_dir_1}/in.aln.bam
  > in_ref=${in_dir_1}/in.ref.fasta.gz
  > in_candidates=${in_dir_2}/in.varcall.unsr.list
  > in_expected=${in_dir_2}/expected.varcall.dorado.tiled.no_regions.vcf
  > in_expected_proc_regions=${in_dir_2}/expected.processed_regions.tiled.no_regions.bed
  > model_var=${MODEL_ROOT_DIR:+--models-directory ${MODEL_ROOT_DIR}}
  > ${DORADO_BIN} smallvar -vv --device cpu ${in_bam} ${in_ref} -t 4 ${model_var} --candidate-filtering true --candidates ${in_candidates} --window-len 300 --window-overlap 100 --ignore-read-groups -o out 2> out/stderr
  > echo "Exit code: $?"
  > grep "\[error\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/stderr | sed -E 's/.*\[/\[/g'
  > ls -1 out
  > ### Remove the qual field because Torch results can vary slightly cross-platform.
  > cat ${in_expected} | grep -v "#" | cut -f 1-5,7-8 > out/expected.no_header.no_qual.vcf
  > gunzip -c out/variants.vcf.gz | grep -v "#" | cut -f 1-5,7-8 > out/result.no_header.no_qual.vcf
  > diff out/expected.no_header.no_qual.vcf out/result.no_header.no_qual.vcf
  > diff ${in_expected_proc_regions} out/processed_regions.bed
  Exit code: 0
  processed_regions.bed
  stderr
  variants.vcf.gz
  variants.vcf.gz.tbi

Candidate filtering with an empty candidates file should succeed and produce no calls.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > in_bam=${in_dir}/in.aln.bam
  > in_ref=${in_dir}/in.ref.fasta.gz
  > model_var=${MODEL_ROOT_DIR:+--models-directory ${MODEL_ROOT_DIR}}
  > touch out/empty_candidates.list
  > ${DORADO_BIN} smallvar -vv --device cpu ${in_bam} ${in_ref} -t 4 ${model_var} --candidate-filtering true --candidate-centered-regions --candidates out/empty_candidates.list --window-len 300 --window-overlap 100 --variant-flanking-bases 100 --ignore-read-groups -o out 2> out/stderr
  > echo "Exit code: $?"
  > grep "\[error\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/stderr | sed -E 's/.*\[/\[/g'
  > ls -1 out
  > gunzip -c out/variants.vcf.gz | grep -v "^#" > out/result.no_header.vcf || true
  > test ! -s out/result.no_header.vcf
  > test ! -s out/processed_regions.bed
  Exit code: 0
  empty_candidates.list
  processed_regions.bed
  stderr
  variants.vcf.gz
  variants.vcf.gz.tbi
