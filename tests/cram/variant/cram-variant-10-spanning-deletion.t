Internally call simple variants and run inference only on computed candidate regions.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/variant/test-09-spanning-deletion-empty-alt
  > in_bam=${in_dir}/in.aln.bam
  > in_ref=${in_dir}/in.ref.fasta.gz
  > in_expected=${in_dir}/expected.dorado.vcf
  > model_var=${MODEL_ROOT_DIR:+--models-directory ${MODEL_ROOT_DIR}}
  > ${DORADO_BIN} smallvar -vv --device cpu ${in_bam} ${in_ref} -t 4 ${model_var} --ignore-read-groups --any-bam -o out 2> out/stderr
  > echo "Exit code: $?"
  > grep "\[error\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/stderr | sed -E 's/.*\[/\[/g'
  > ### Remove the qual field because Torch results can vary slightly cross-platform.
  > cat ${in_expected} | grep -v "#" | cut -f 1-5,7-8 > out/expected.no_header.no_qual.vcf
  > cat out/variants.vcf | grep -v "#" | cut -f 1-5,7-8 > out/result.no_header.no_qual.vcf
  > diff out/expected.no_header.no_qual.vcf out/result.no_header.no_qual.vcf
  Exit code: 0
