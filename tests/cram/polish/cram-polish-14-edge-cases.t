Giant insertion - one of the alignments has a large insertion which causes some samples to be fully immersed in the insertion and should be completely trimmed.
Edge case is when there are two overlapping BAM regions, and a sample window is within the overlapping overhang of the regions.
The issue is caused when region trimming is applied, and one of the coordinates (trim.start or trim.end) is < 0 because the sample should be fully trimmed.
# |=============================================|       -> Draft
# |//BAM region 1////////////|                          -> Overlapping BAM
#                 |///////////////BAM region 2//|          regions
#                 |<-------->|                          -> Overlapping overhang
#                      ^                                -> There is a 10000I insertion in the draft in this region.
#                      10000I
#                    ->|<-                              -> Window size is much shorter than the insertion, and the sample is fully
#                                                           covered by the insertion (i.e. all positions_major have the same coordinate).
#                                                           In this case, the sample has window_len positions in the features tensor, but all have the same coordinate,
#                                                           and this coordinate is outside of the BAM_region_2 start/end interval, so it should be fully trimmed.
This test case adds one manully-constructed (synthetic) alignment to the input BAM.
This alignment is based on the query "8c9df1f1-513e-4756-8259-3d541bb92b02" but has a 10k 'A' insertion after query position 3107.
Corresponding qualities are set to `!` (also 10k of them). The `mv:` tag was updated to have an additional 10k `1` moves for those positions.
Given the right BAM region size and overap, this triggers the edge case.
Such samples should be filtered out from the output.
  $ rm -rf out; mkdir -p out
  > in_dir=${TEST_DATA_DIR}/polish/test-01-supertiny
  > model_var=${MODEL_DIR:+--model ${MODEL_DIR}}
  > # Create the edge-case input BAM.
  > samtools merge -c out/in.bam ${in_dir}/calls_to_draft.bam ${in_dir}/calls_to_draft.single_read.long_insertion.bam
  > samtools index out/in.bam
  > # Run.
  > ${DORADO_BIN} polish --any-bam --device cpu --bam-chunk 110 --bam-subchunk 20 --window-len 50 --window-overlap 20 out/in.bam ${in_dir}/draft.fasta.gz -t 1 --infer-threads 1 ${model_var} -vv > out/out.fasta 2> out/out.fasta.stderr
  > echo "Exit code: $?"
  > # Test.
  > ${DORADO_BIN} aligner ${in_dir}/ref.fasta.gz out/out.fasta 1> out/out.bam 2> out/out.bam.stderr
  > samtools view out/out.bam | sed -E 's/^.*(NM:i:[0-9]+).*/\1/g'
  > samtools faidx out/out.fasta
  > cut -f 2,2 out/out.fasta.fai
  > grep "Copying contig verbatim from input" out/out.fasta.stderr | wc -l | awk '{ print $1 }'
  Exit code: 0
  NM:i:4
  10000
  0
