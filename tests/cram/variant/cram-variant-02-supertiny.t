Run a supertiny test case.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > in_bam=${in_dir}/in.aln.bam
  > in_ref=${in_dir}/in.ref.fasta.gz
  > in_expected=${in_dir}/expected.dorado.vcf
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > ${DORADO_BIN} variant --device cpu ${in_bam} ${in_ref} -t 4 ${model_var} --ignore-read-groups > out/out.vcf 2> out/stderr
  > echo "Exit code: $?"
  > ### Remove the qual field because Torch results can vary slightly cross-platform.
  > cat ${in_expected} | grep -v "#" | cut -f 1-5,7-8 > out/expected.no_header.no_qual.vcf
  > cat out/out.vcf | grep -v "#" | cut -f 1-5,7-8 > out/result.no_header.no_qual.vcf
  > diff out/expected.no_header.no_qual.vcf out/result.no_header.no_qual.vcf
  Exit code: 0
