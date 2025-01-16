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
Note: Htslib prints round values such as `70.0` as `70`. Medaka outputs `70.0` insted. We sed the 70.0/70 here.
Note2: There is one diff out of 10000bp - the QV differs in the second decimal position for one loci. We will
allow it here for now.
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
  > diff out/expected.no_header.vcf out/result.no_header.vcf > out/diff.txt
  > cat out/diff.txt | grep "[<>]" | sed "s/[<>] //g"
  Exit code: 0
  contig_1\t8950\t.\tG\t.\t43.382\t.\t.\tGT:GQ\t0:43 (esc)
  contig_1\t8950\t.\tG\t.\t43.394\t.\t.\tGT:GQ\t0:43 (esc)
