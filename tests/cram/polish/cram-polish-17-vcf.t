VCF output to stdout (variants only).
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=${in_dir}/calls_to_draft.bam
  > in_draft=${in_dir}/draft.fasta.gz
  > expected=${in_dir}/medaka.variants.vcf
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > ${DORADO_BIN} polish --vcf --device cpu ${in_bam} ${in_draft} -t 4 ${model_var} > out/variants.vcf 2> out/stderr
  > echo "Exit code: $?"
  > grep -v "#" ${expected} > out/expected.no_header.vcf
  > grep -v "#" out/variants.vcf > out/result.no_header.vcf
  > diff out/expected.no_header.vcf out/result.no_header.vcf
  Exit code: 0

gVCF output to stdout (variants + non-variant positions).
IMPORTANT: not comparing the variant/genotype qualities because they may vary with Torch versions and architectures.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=${in_dir}/calls_to_draft.bam
  > in_draft=${in_dir}/draft.fasta.gz
  > expected=${in_dir}/medaka.variants.gvcf.gz
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > ${DORADO_BIN} polish --gvcf --device cpu ${in_bam} ${in_draft} -t 4 ${model_var} > out/variants.vcf 2> out/stderr
  > echo "Exit code: $?"
  > gunzip -d --stdout ${expected} | grep -v "#" | sed 's/70\.0/70/g' | sort > out/expected.no_header.vcf
  > grep -v "#" out/variants.vcf | sort > out/result.no_header.vcf
  > ### Remove the qual field because Torch results can vary slightly cross-platform.
  > cut -f 1-5,7-8 out/expected.no_header.vcf > out/expected.no_header.no_qual.vcf
  > cut -f 1-5,7-8 out/result.no_header.vcf > out/result.no_header.no_qual.vcf
  > diff out/expected.no_header.no_qual.vcf out/result.no_header.no_qual.vcf
  Exit code: 0

No VCF output to a directory. By default, when specifying an output directory there should be no .vcf files generated.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=${in_dir}/calls_to_draft.bam
  > in_draft=${in_dir}/draft.fasta.gz
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > ${DORADO_BIN} polish --device cpu ${in_bam} ${in_draft} -t 4 ${model_var} -o out 2> out/stderr
  > echo "Exit code: $?"
  > ls -1 out
  Exit code: 0
  consensus.fasta
  stderr

VCF output to a directory (variants only).
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=${in_dir}/calls_to_draft.bam
  > in_draft=${in_dir}/draft.fasta.gz
  > expected=${in_dir}/medaka.variants.vcf
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > ${DORADO_BIN} polish --vcf --device cpu ${in_bam} ${in_draft} -t 4 ${model_var} -o out 2> out/stderr
  > echo "Exit code: $?"
  > ls -1 out
  > grep -v "#" ${expected} > out/expected.no_header.vcf
  > grep -v "#" out/variants.vcf > out/result.no_header.vcf
  > diff out/expected.no_header.vcf out/result.no_header.vcf
  Exit code: 0
  consensus.fasta
  stderr
  variants.vcf

gVCF output to a directory (variants + non-variant positions).
IMPORTANT: not comparing the variant/genotype qualities because they may vary with Torch versions and architectures.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=${in_dir}/calls_to_draft.bam
  > in_draft=${in_dir}/draft.fasta.gz
  > expected=${in_dir}/medaka.variants.gvcf.gz
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > ${DORADO_BIN} polish --gvcf --device cpu ${in_bam} ${in_draft} -t 4 ${model_var} -o out 2> out/stderr
  > echo "Exit code: $?"
  > ls -1 out
  > gunzip -d --stdout ${expected} | grep -v "#" | sed 's/70\.0/70/g' | sort > out/expected.no_header.vcf
  > grep -v "#" out/variants.vcf | sort > out/result.no_header.vcf
  > ### Remove the qual field because Torch results can vary slightly cross-platform.
  > cut -f 1-5,7-8 out/expected.no_header.vcf > out/expected.no_header.no_qual.vcf
  > cut -f 1-5,7-8 out/result.no_header.vcf > out/result.no_header.no_qual.vcf
  > diff out/expected.no_header.no_qual.vcf out/result.no_header.no_qual.vcf
  Exit code: 0
  consensus.fasta
  stderr
  variants.vcf

Both --vcf and --gvcf are specified, this should fail.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=${in_dir}/calls_to_draft.bam
  > in_draft=${in_dir}/draft.fasta.gz
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > ${DORADO_BIN} polish --vcf --gvcf --device cpu ${in_bam} ${in_draft} -t 4 ${model_var} -o out 2> out/stderr
  > echo "Exit code: $?"
  > grep "\[error\]" out/stderr | sed -E 's/.*\[error\] //g'
  Exit code: 1
  Caught exception: Both --vcf and --gvcf are specified. Only one of these options can be used.
