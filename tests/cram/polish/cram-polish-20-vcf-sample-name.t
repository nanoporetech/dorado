Default sample name detection from BAM.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > in_bam=${in_dir}/in.aln.bam
  > in_ref=${in_dir}/in.ref.fasta.gz
  > model_var=${MODEL_ROOT_DIR:+--models-directory ${MODEL_ROOT_DIR}}
  > ${DORADO_BIN} polish --vcf --device cpu ${in_bam} ${in_ref} -t 4 ${model_var} --ignore-read-groups > out/out.vcf 2> out/stderr
  > echo "Exit code: $?"
  > grep "\[error\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "\[E::" out/stderr || true
  > ### Check the column names
  > expected_header="#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	K36A_400bps_1"
  > grep -Fxq "${expected_header}" out/out.vcf || exit 1
  Exit code: 0

Test overriding sample name via CLI.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > in_bam=${in_dir}/in.aln.bam
  > in_ref=${in_dir}/in.ref.fasta.gz
  > model_var=${MODEL_ROOT_DIR:+--models-directory ${MODEL_ROOT_DIR}}
  > ${DORADO_BIN} polish --vcf --device cpu ${in_bam} ${in_ref} -t 4 ${model_var} --ignore-read-groups --sample-name override > out/out.vcf 2> out/stderr
  > echo "Exit code: $?"
  > grep "\[error\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "\[E::" out/stderr || true
  > ### Check the column names
  > expected_header="#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	override"
  > grep -Fxq "${expected_header}" out/out.vcf || exit 1
  Exit code: 0

Test selection via RG cli
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > in_bam=${in_dir}/in.aln.extrarg.bam
  > in_ref=${in_dir}/in.ref.fasta.gz
  > model_var=${MODEL_ROOT_DIR:+--models-directory ${MODEL_ROOT_DIR}}
  > ${DORADO_BIN} polish --vcf --device cpu ${in_bam} ${in_ref} -t 4 ${model_var} --RG "bc8993f4557dd53bf0cbda5fd68453fea5e94485_dna_r10.4.1_e8.2_400bps_hac@v5.2.0" > out/out.vcf 2> out/stderr
  > echo "Exit code: $?"
  > grep "\[error\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "\[E::" out/stderr || true
  > ### Check the column names
  > expected_header="#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	K36A_400bps_1"
  > grep -Fxq "${expected_header}" out/out.vcf || exit 1
  Exit code: 0

Test fallback when no other source of sample name.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > in_bam=${in_dir}/in.aln.norg.bam
  > in_ref=${in_dir}/in.ref.fasta.gz
  > model_var="--model-override ${POLISH_MODEL}"
  > ${DORADO_BIN} polish --vcf --device cpu ${in_bam} ${in_ref} -t 4 ${model_var} --any-bam > out/out.vcf 2> out/stderr
  > echo "Exit code: $?"
  > grep "\[error\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/stderr | sed -E 's/.*\[/\[/g' | sed -E 's/model: .*/model/g'
  > grep "\[E::" out/stderr || true
  > ### Check the column names
  > expected_header="#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	SAMPLE"
  > grep -Fxq "${expected_header}" out/out.vcf || exit 1
  Exit code: 0
  [warning] Skipping basecaller compatibility checks for user-specified model

Multiple RGs with single sample name passes
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > in_bam=${in_dir}/in.aln.doublerg.bam
  > in_ref=${in_dir}/in.ref.fasta.gz
  > model_var=${MODEL_ROOT_DIR:+--models-directory ${MODEL_ROOT_DIR}}
  > ${DORADO_BIN} polish --vcf --device cpu ${in_bam} ${in_ref} -t 4 ${model_var} --ignore-read-groups > out/out.vcf 2> out/stderr
  > echo "Exit code: $?"
  > grep "\[error\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "\[E::" out/stderr || true
  > ### Check the column names
  > expected_header="#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	K36A_400bps_1"
  > grep -Fxq "${expected_header}" out/out.vcf || exit 1
  Exit code: 0

Multiple LB values causes early exit
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > in_bam=${in_dir}/in.aln.extrarg.bam
  > in_ref=${in_dir}/in.ref.fasta.gz
  > model_var=${MODEL_ROOT_DIR:+--models-directory ${MODEL_ROOT_DIR}}
  > ${DORADO_BIN} polish --vcf --device cpu ${in_bam} ${in_ref} -t 4 ${model_var} --ignore-read-groups > out/out.vcf 2> out/stderr
  > echo "Exit code: $?"
  > grep "\[error\]" out/stderr | sed -E 's/.*\]\ \[/\[/'
  Exit code: 1
  [error] BAM read groups contain multiple sample names: [extra_sample,K36A_400bps_1]
  [error] Please specify a sample name with the `--sample-name` option.

Multiple LB values proceed if not running variant calling
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > in_bam=${in_dir}/in.aln.extrarg.bam
  > in_ref=${in_dir}/in.ref.fasta.gz
  > model_var=${MODEL_ROOT_DIR:+--models-directory ${MODEL_ROOT_DIR}}
  > ${DORADO_BIN} polish --device cpu ${in_bam} ${in_ref} -t 4 ${model_var} --ignore-read-groups > out/out.vcf 2> out/stderr
  > echo "Exit code: $?"
  Exit code: 0