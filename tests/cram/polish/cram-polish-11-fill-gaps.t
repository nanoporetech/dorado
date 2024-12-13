Create a synthetic test case used by tests below. This block does not run any tests.
  $ rm -rf data; mkdir -p data
  > in_dir="${TEST_DATA_DIR}/polish/test-01-supertiny"
  > in_draft="${in_dir}/draft.fasta.gz"
  > in_bam="${in_dir}/calls_to_draft.bam"
  > # Create a draft with a large gap in contig_1, and add a synthetic contig_2 which has zero alignments.
  > samtools faidx ${in_draft} "contig_1:1-5000" > data/draft.fasta.tmp
  > python3 -c "print(''.join(['C']*20000))" >> data/draft.fasta.tmp
  > samtools faidx ${in_draft} "contig_1:5001-10000" | tail -n+2 >> data/draft.fasta.tmp
  > echo ">contig_1" > data/draft.fasta
  > tail -n+2 data/draft.fasta.tmp |  tr -d '\n' >> data/draft.fasta
  > echo "" >> data/draft.fasta
  > echo ">contig_2" >> data/draft.fasta
  > echo "AAAAAAAAAACCCCCCCCCCTTTTTTTTTTGGGGGGGGGG" >> data/draft.fasta
  > # Align and filter all reads to the new draft.
  > samtools view -hb -F 0x904 ${in_bam} > data/tmp.calls_to_draft.filtered.bam
  > ${DORADO_BIN} aligner data/draft.fasta data/tmp.calls_to_draft.filtered.bam 1> data/tmp.bam 2> data/tmp.bam.stderr
  > samtools view -hb -F 0x904 data/tmp.bam | samtools sort > data/calls_to_draft.bam
  > samtools index data/calls_to_draft.bam

Fill gaps with draft sequence (default behaviour).
Expected: no diffs, exit code 0 and one sequence copied verbatim (contig_2 with no coverage).
  $ rm -rf out; mkdir -p out
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > in_draft="data/draft.fasta"
  > in_bam="data/calls_to_draft.bam"
  > expected="${in_dir}/expected.synth.medaka.w_fill_gaps.no_fill_char.fasta.gz"
  > ${DORADO_BIN} polish --device cpu ${in_bam} ${in_draft} ${model_var} -vv > out/out.fasta 2> out/stderr
  > echo "Exit code: $?"
  > grep "Copying contig verbatim from input" out/stderr | wc -l | awk '{ print $1 }'
  > gunzip -c ${expected} > out/expected.fasta
  > diff out/expected.fasta out/out.fasta
  Exit code: 0
  1

No filling of gaps with draft sequence.
Expected: only parts of contig_1 with coverage should be output. contig_2 will not be output because it has zero coverage.
  $ rm -rf out; mkdir -p out
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > in_draft="data/draft.fasta"
  > in_bam="data/calls_to_draft.bam"
  > expected="${in_dir}/expected.synth.medaka.no_fill_gaps.no_fill_char.fasta.gz"
  > ${DORADO_BIN} polish --device cpu ${in_bam} ${in_draft} ${model_var} --no-fill-gaps -vv > out/out.fasta 2> out/stderr
  > echo "Exit code: $?"
  > grep "Copying contig verbatim from input" out/stderr | wc -l | awk '{ print $1 }'
  > gunzip -c ${expected} > out/expected.fasta
  > diff out/expected.fasta out/out.fasta
  Exit code: 0
  0

Fill gaps with custom user-defined character.
The zero coverage gap in contig_1 should be filled with 'Z' characters instead of bases from the draft.
Also, the `contig_2` sequence should be taken verbatim from input draft, and not be filled with 'Z' characters.
  $ rm -rf out; mkdir -p out
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > in_draft="data/draft.fasta"
  > in_bam="data/calls_to_draft.bam"
  > expected="${in_dir}/expected.synth.medaka.w_fill_gaps.fill_char_Z.fasta"
  > ${DORADO_BIN} polish --device cpu ${in_bam} ${in_draft} ${model_var} --fill-char "Z" -vv > out/out.fasta 2> out/stderr
  > exit_code=$?
  > echo "Exit code: ${exit_code}"
  > grep "Copying contig verbatim from input" out/stderr | wc -l | awk '{ print $1 }'
  > gunzip -c ${expected} > out/expected.fasta
  > diff out/expected.fasta out/out.fasta
  > tail -n 1 out/out.fasta
  Exit code: 0
  1
  AAAAAAAAAACCCCCCCCCCTTTTTTTTTTGGGGGGGGGG

No filling of gaps with draft sequence, while `--fill-char` is used.
The `--fill-char` should have no effect.
Expected: only parts of contig_1 with coverage should be output. contig_2 will not be output because it has zero coverage.
  $ rm -rf out; mkdir -p out
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > in_draft="data/draft.fasta"
  > in_bam="data/calls_to_draft.bam"
  > expected="${in_dir}/expected.synth.medaka.no_fill_gaps.no_fill_char.fasta.gz"
  > ${DORADO_BIN} polish --device cpu ${in_bam} ${in_draft} ${model_var} --no-fill-gaps --fill-char "Z" -vv > out/out.fasta 2> out/stderr
  > echo "Exit code: $?"
  > grep "Copying contig verbatim from input" out/stderr | wc -l | awk '{ print $1 }'
  > gunzip -c ${expected} > out/expected.fasta
  > diff out/expected.fasta out/out.fasta
  Exit code: 0
  0
