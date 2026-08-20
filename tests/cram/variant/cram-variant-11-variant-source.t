Verify the per-sample variant source (SRC) FORMAT tag.
The SRC tag records how each call was produced and can take the following values:
I = inference, K = Kadayashi (simple variant caller), R = gVCF reference block, . = unknown.
Other cram tests drop the SAMPLE column (they compare `cut -f 1-5,7-8`) because the GQ
score can fluctuate cross-platform, so the SRC value itself is not checked there. This test
looks up SRC by name in the FORMAT column and reads the matching subfield of SAMPLE, so it is
independent of both the GQ value and the ordering of FORMAT fields. The missing value "." is only
a defensive default emitted by the writer when a variant carries no source; it is not reachable
through the CLI and is covered by the VCFWriter unit test.

Default run: inference (I) and Kadayashi (K) sources.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/variant/test-02-supertiny
  > in_bam=${in_dir}/in.aln.bam
  > in_ref=${in_dir}/in.ref.fasta.gz
  > model_var=${MODEL_ROOT_DIR:+--models-directory ${MODEL_ROOT_DIR}}
  > ${DORADO_BIN} smallvar --device cpu ${in_bam} ${in_ref} -t 4 ${model_var} --ignore-read-groups > out/out.vcf 2> out/stderr
  > echo "Exit code: $?"
  > grep "\[error\]" out/stderr | sed -E 's/.*\[/\[/g'
  > ### Derive the SRC value by looking it up by name in FORMAT ($9) and reading the matching
  > ### SAMPLE ($10) subfield, so the check does not assume any particular FORMAT field ordering.
  > src_of='src=""; nf=split($9,f,":"); split($10,s,":"); for(i=1;i<=nf;i++) if(f[i]=="SRC") src=s[i];'
  > ### SRC is declared as a FORMAT field (not INFO).
  > grep -E "^##(INFO|FORMAT)=<ID=SRC" out/out.vcf
  > ### SRC must never appear in the INFO column; it lives in FORMAT/SAMPLE.
  > grep -v "#" out/out.vcf | awk '$8 ~ /SRC/ { print "SRC_IN_INFO:"$0 } END { print "info_column_has_no_src" }'
  > ### All variant source values present.
  > grep -v "#" out/out.vcf | awk "{ ${src_of} print src }" | sort -u | paste -sd' ' -
  > ### Explicit per-variant source.
  > grep -v "#" out/out.vcf | awk "{ ${src_of} print \$1, \$2, src }"
  Exit code: 0
  ##FORMAT=<ID=SRC,Number=1,Type=String,Description="Source of the variant">
  info_column_has_no_src
  I K
  chr20 94 K
  chr20 306 K
  chr20 1472 I
  chr20 1959 I
  chr20 2102 I
  chr20 2126 I
  chr20 2438 K
  chr20 2446 K
  chr20 2967 K
  chr20 3176 K
  chr20 3735 K
  chr20 4696 I
  chr20 5000 I
  chr20 5387 K
  chr20 6948 I
  chr20 7430 I

gVCF run: reference-block (R) and inference (I) sources.
  $ rm -rf out; mkdir -p out
  > in_dir_1=${TEST_DATA_DIR}/variant/test-02-supertiny
  > in_dir_2=${TEST_DATA_DIR}/variant/test-03-kadayashi-varcall
  > in_bam=${in_dir_1}/in.aln.bam
  > in_ref=${in_dir_1}/in.ref.fasta.gz
  > in_candidates=${in_dir_2}/in.varcall.unsr.list
  > model_var="--model-override ${MODEL_ROOT_DIR}/${MODEL_v600}"
  > ${DORADO_BIN} smallvar --device cpu ${in_bam} ${in_ref} -t 4 ${model_var} --candidate-filtering true --candidates ${in_candidates} --window-len 300 --window-overlap 100 --variant-flanking-bases 100 --regions "chr20:1000-2145" --ignore-read-groups --gvcf -o out 2> out/stderr
  > echo "Exit code: $?"
  > grep "\[error\]" out/stderr | sed -E 's/.*\[/\[/g'
  > ### Order-independent SRC lookup (see the first block).
  > src_of='src=""; nf=split($9,f,":"); split($10,s,":"); for(i=1;i<=nf;i++) if(f[i]=="SRC") src=s[i];'
  > ### All variant source values present (I from inference, R from the outside-inference reference block).
  > grep -v "#" out/variants.vcf | awk "{ ${src_of} print src }" | sort -u | paste -sd' ' -
  > ### Reference blocks (INFO carries END) are I inside inference regions and R outside; never K or U.
  > grep -v "#" out/variants.vcf | awk "\$8 ~ /^END=/ { ${src_of} if (src != \"R\" && src != \"I\") print \"BAD:\"\$0 } END { print \"reference_blocks_only_R_or_I\" }"
  > ### The leading reference block starts at the region start (pos 1000), which is outside the
  > ### inference region (chr20:1345-...), so it is sourced R.
  > grep -v "#" out/variants.vcf | awk "\$2 == 1000 { ${src_of} print \$2, src }"
  > ### The first reference block inside the inference region is sourced I.
  > grep -v "#" out/variants.vcf | awk "\$8 ~ /^END=/ { ${src_of} if (src == \"I\") { print \$2, src; exit } }"
  > ### The non-reference (variant) records are inference-sourced.
  > grep -v "#" out/variants.vcf | awk "\$8 == \".\" { ${src_of} print \$2, src }"
  Exit code: 0
  I R
  reference_blocks_only_R_or_I
  1000 R
  1346 I
  1472 I
  1959 I
  2102 I
  2126 I
