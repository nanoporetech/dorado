Provide candidate variant sites to seed inference windows (the --candidates feature).
  $ rm -rf out; mkdir -p out
  > in_dir_1=${TEST_DATA_DIR}/variant/test-02-supertiny
  > in_dir_2=${TEST_DATA_DIR}/variant/test-03-kadayashi-varcall
  > in_bam=${in_dir_1}/in.aln.bam
  > in_ref=${in_dir_1}/in.ref.fasta.gz
  > in_expected=${in_dir_2}/expected.varcall.merged_kadayashi_and_inference.vcf
  > in_expected_proc_regions=${in_dir_2}/expected.processed_regions.full.bed
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > ${DORADO_BIN} variant -vv --device cpu ${in_bam} ${in_ref} -t 4 ${model_var} --candidate-filtering --window-len 300 --window-overlap 100 --variant-flanking-bases 100 --kada-max-clipping 100000 --kada-min-strand-cov 1 --ignore-read-groups -o out 2> out/stderr
  > echo "Exit code: $?"
  > grep "\[error\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/stderr | sed -E 's/.*\[/\[/g'
  > ### Remove the qual field because Torch results can vary slightly cross-platform.
  > cat ${in_expected} | grep -v "#" | cut -f 1-5,7-8 > out/expected.no_header.no_qual.vcf
  > cat out/variants.vcf | grep -v "#" | cut -f 1-5,7-8 > out/result.no_header.no_qual.vcf
  > sort ${in_expected_proc_regions} > out/expected.processed_regions.sorted.bed
  > sort out/processed_regions.bed > out/processed_regions.sorted.bed
  > diff out/expected.no_header.no_qual.vcf out/result.no_header.no_qual.vcf
  > diff out/expected.processed_regions.sorted.bed out/processed_regions.sorted.bed
  Exit code: 0
  [warning] This is an alpha preview of Dorado Variant. Results should be considered experimental.
