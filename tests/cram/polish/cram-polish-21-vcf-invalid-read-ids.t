Regression test exposing failure due to invalidated read ids when merging chunks
with increasing depth.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-02-vcf-invalid-read-ids
  > in_bam=${in_dir}/aln.bam
  > in_draft=${in_dir}/ref.fasta
  > expected=${in_dir}/expected.variants.vcf.gz
  > model_var="--models-directory ${MODEL_ROOT_DIR}"
  > ${DORADO_BIN} polish --vcf --device cpu ${in_bam} ${in_draft} -t 4 ${model_var} --bam-subchunk 1000 --any-bam > out/variants.vcf 2> out/stderr
  > echo "Exit code: $?"
  > grep "\[error\]" out/stderr | sed -E 's/.*\[error\] //g'
  > grep "\[warning\]" out/stderr | sed -E 's/.*\[warning\] //g'
  > gunzip -c ${expected} | grep -v "#" > out/expected.no_header.vcf
  > grep -v "#" out/variants.vcf > out/result.no_header.vcf
  > diff out/expected.no_header.vcf out/result.no_header.vcf
  Exit code: 0

