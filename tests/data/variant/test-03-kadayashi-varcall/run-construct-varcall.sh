#! /bin/bash
set -vex
CWD=$(pwd)

out_dir=temp
mkdir -p ${out_dir}
pushd ${out_dir}

region="chr20"
threads="4"
in_ref="${CWD}/../test-02-supertiny/in.ref.fasta.gz"
in_bam="${CWD}/../test-02-supertiny/in.aln.bam"
${KADAYASHI_BIN} --version 2>&1 | tee kadayashi.version.txt
${KADAYASHI_BIN} varcall -r ${region} -o varcall -t ${threads} -p 5000 ${in_ref} ${in_bam} 2>&1 | tee log.kadayashi.tee

cat varcall.unsr.list | awk '{ if (($2 >= 1500) && ($2 < 4000)) { print } }' > in.varcall.unsr.list
cat varcall.unsr.list | awk '{ if (($2 >= 6000) && ($2 < 7000)) { print } }' >> in.varcall.unsr.list
printf "chr20\t5500\t7500\n" > in.varcall.bed

cp in.* ../

popd
