Run a supertiny test case.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > result_dir=${TEST_DATA_DIR}/variant/test-06-supertiny-hemizygous
  > in_bam=${in_dir}/in.aln.bam
  > in_ref=${in_dir}/in.ref.fasta.gz
  > in_expected=${result_dir}/expected.dorado.vcf
  > model_var=${MODEL_ROOT_DIR:+--models-directory ${MODEL_ROOT_DIR}}
  > ${DORADO_BIN} smallvar --device cpu ${in_bam} ${in_ref} -t 4 ${model_var} --ignore-read-groups --hemizygous-regions chr20:2500-5002 > out/out.vcf 2> out/stderr
  > echo "Exit code: $?"
  > grep "\[error\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/stderr | sed -E 's/.*\[/\[/g'
  > ### Remove the qual field because Torch results can vary slightly cross-platform.
  > cat ${in_expected} | grep -v "#" | cut -f 1-5,7-8 > out/expected.no_header.no_qual.vcf
  > cat out/out.vcf | grep -v "#" | cut -f 1-5,7-8 > out/result.no_header.no_qual.vcf
  > diff out/expected.no_header.no_qual.vcf out/result.no_header.no_qual.vcf
  > grep "\[E::" out/stderr || true
  Exit code: 0

Run a supertiny test case with gvcf output, reference blocks should be split on hemizygous region breaks.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > in_bam=${in_dir}/in.aln.bam
  > in_ref=${in_dir}/in.ref.fasta.gz
  > model_var=${MODEL_ROOT_DIR:+--models-directory ${MODEL_ROOT_DIR}}
  > ${DORADO_BIN} smallvar --device cpu ${in_bam} ${in_ref} -t 4 ${model_var} --ignore-read-groups --hemizygous-regions chr20:2500-5002 --gvcf > out/out.vcf 2> out/stderr
  > echo "Exit code: $?"
  > grep "\[error\]" out/stderr | sed -E 's/.*\[/\[/g'
  > grep "\[warning\]" out/stderr | sed -E 's/.*\[/\[/g'
  > ### check the partial records at the split position
  > awk '$2 == 2447' out/out.vcf | sed -E 's/\t/ /g'
  > awk '$2 == 2500' out/out.vcf | sed -E 's/\t/ /g'
  Exit code: 0
  chr20 2447 . T <*> 60 . END=2499 GT:GQ:LEN:MIN_DP:SRC 0/0:60:53:18:R
  chr20 2500 . C <*> 60 . END=2966 GT:GQ:LEN:MIN_DP:SRC 0:60:467:18:R

