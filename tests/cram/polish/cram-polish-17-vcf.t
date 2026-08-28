VCF output to stdout (variants only).
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=${in_dir}/calls_to_draft.bam
  > in_draft=${in_dir}/draft.fasta.gz
  > expected=${in_dir}/medaka.variants.vcf
  > model_var="--models-directory ${MODEL_ROOT_DIR}"
  > ${DORADO_BIN} polish --vcf --device cpu ${in_bam} ${in_draft} -t 4 ${model_var} > out/variants.vcf 2> out/stderr
  > echo "Exit code: $?"
  > grep -v "#" ${expected} > out/expected.no_header.vcf
  > grep -v "#" out/variants.vcf > out/result.no_header.vcf
  > diff out/expected.no_header.vcf out/result.no_header.vcf
  Exit code: 0

gVCF output to stdout (variants + non-variant positions).
IMPORTANT: not comparing exact reference block boundaries because they depend on qualities which may vary with Torch versions and architectures.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=${in_dir}/calls_to_draft.bam
  > in_draft=${in_dir}/draft.fasta.gz
  > expected=${in_dir}/medaka.variants.vcf
  > model_var="--models-directory ${MODEL_ROOT_DIR}"
  > ${DORADO_BIN} polish --gvcf --device cpu ${in_bam} ${in_draft} -t 4 ${model_var} > out/variants.vcf 2> out/stderr
  > echo "Exit code: $?"
  > grep -E "^##INFO=<ID=END|^##ALT=<ID=\*|^##FORMAT=<ID=LEN" out/variants.vcf
  > grep -v "#" ${expected} | awk 'BEGIN { FS=OFS="\t" } $5 != "." && $5 !~ /(^|,)<\*>($|,)/ { $5 = $5 ",<*>" } { print }' | cut -f 1-5,7-8 > out/expected.variants.no_qual.vcf
  > grep -v "#" out/variants.vcf | awk '$5 != "." && $5 != "<*>"' | cut -f 1-5,7-8 > out/result.variants.no_qual.vcf
  > diff out/expected.variants.no_qual.vcf out/result.variants.no_qual.vcf
  > awk '$5 == "<*>" && $8 ~ /^END=[0-9]+$/ { found=1 } END { print found ? "has_reference_blocks" : "missing_reference_blocks" }' out/variants.vcf
  > awk 'BEGIN { compact="compact_reference_blocks" } !/^#/ { ++num_records } END { if (num_records >= 2000) compact="too_many_reference_records"; print compact }' out/variants.vcf
  Exit code: 0
  ##INFO=<ID=END,Number=1,Type=Integer,Description="End position of the reference block">
  ##FORMAT=<ID=LEN,Number=1,Type=Integer,Description="Length of <*> reference block">
  has_reference_blocks
  compact_reference_blocks

No VCF output to a directory. By default, when specifying an output directory there should be no .vcf files generated.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=${in_dir}/calls_to_draft.bam
  > in_draft=${in_dir}/draft.fasta.gz
  > model_var="--models-directory ${MODEL_ROOT_DIR}"
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
  > model_var="--models-directory ${MODEL_ROOT_DIR}"
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
IMPORTANT: not comparing exact reference block boundaries because they depend on qualities which may vary with Torch versions and architectures.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=${in_dir}/calls_to_draft.bam
  > in_draft=${in_dir}/draft.fasta.gz
  > expected=${in_dir}/medaka.variants.vcf
  > model_var="--models-directory ${MODEL_ROOT_DIR}"
  > ${DORADO_BIN} polish --gvcf --device cpu ${in_bam} ${in_draft} -t 4 ${model_var} -o out 2> out/stderr
  > echo "Exit code: $?"
  > ls -1 out
  > ### Check the command line in the header
  > expected_cmdline="##cmdline=polish --gvcf --device cpu ${in_bam} ${in_draft} -t 4 ${model_var} -o out"
  > grep -Fxq "${expected_cmdline}" out/variants.vcf || exit 1
  > ### Check other lines.
  > grep -E "^##INFO=<ID=END|^##ALT=<ID=\*|^##FORMAT=<ID=LEN" out/variants.vcf
  > grep -v "#" ${expected} | awk 'BEGIN { FS=OFS="\t" } $5 != "." && $5 !~ /(^|,)<\*>($|,)/ { $5 = $5 ",<*>" } { print }' | cut -f 1-5,7-8 > out/expected.variants.no_qual.vcf
  > grep -v "#" out/variants.vcf | awk '$5 != "." && $5 != "<*>"' | cut -f 1-5,7-8 > out/result.variants.no_qual.vcf
  > diff out/expected.variants.no_qual.vcf out/result.variants.no_qual.vcf
  > awk '$5 == "<*>" && $8 ~ /^END=[0-9]+$/ { found=1 } END { print found ? "has_reference_blocks" : "missing_reference_blocks" }' out/variants.vcf
  > awk 'BEGIN { compact="compact_reference_blocks" } !/^#/ { ++num_records } END { if (num_records >= 2000) compact="too_many_reference_records"; print compact }' out/variants.vcf
  Exit code: 0
  consensus.fasta
  stderr
  variants.vcf
  ##INFO=<ID=END,Number=1,Type=Integer,Description="End position of the reference block">
  ##FORMAT=<ID=LEN,Number=1,Type=Integer,Description="Length of <*> reference block">
  has_reference_blocks
  compact_reference_blocks

Both --vcf and --gvcf are specified, this should fail.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=${in_dir}/calls_to_draft.bam
  > in_draft=${in_dir}/draft.fasta.gz
  > model_var="--models-directory ${MODEL_ROOT_DIR}"
  > ${DORADO_BIN} polish --vcf --gvcf --device cpu ${in_bam} ${in_draft} -t 4 ${model_var} -o out 2> out/stderr
  > echo "Exit code: $?"
  > grep "\[error\]" out/stderr | sed -E 's/.*\[error\] //g'
  Exit code: 1
  Both --vcf and --gvcf are specified. Only one of these options can be used.

VCF output to stdout, lower-case reference.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > in_bam=${in_dir}/calls_to_draft.bam
  > in_draft=${in_dir}/draft.lowercase.fasta.gz
  > expected=${in_dir}/medaka.variants.vcf
  > model_var="--models-directory ${MODEL_ROOT_DIR}"
  > ${DORADO_BIN} polish --vcf --device cpu ${in_bam} ${in_draft} -t 4 ${model_var} > out/variants.vcf 2> out/stderr
  > echo "Exit code: $?"
  > grep -v "#" ${expected} > out/expected.no_header.vcf
  > grep -v "#" out/variants.vcf > out/result.no_header.vcf
  > diff out/expected.no_header.vcf out/result.no_header.vcf
  Exit code: 0
