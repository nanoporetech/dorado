## Description of the test.

This test is composed of a pileup of 6 reads.
These reads will be corrected by Dorado Correct.
Verification of the per-base accuracy is performed by reference-based alignment (using HG002 v1.0.1 form the T2T consortium).

```
$ minimap2 --eqx -c -x map-ont --secondary=no reference/hg002-v1.0.1/hg002v1.0.1.fasta expected.fasta > expected.fasta.paf
$ sort -k1,1 -k2,2n expected.fasta.paf.bed | bedtools merge -d 1000 > expected.region.bed
$ cat expected.region.bed
chr6_PATERNAL	115004963	115077006

# Take slightly larger regions because flanks may have been unaligned.
$ samtools faidx ~/data/reference/hg002-v1.0.1/hg002v1.0.1.fasta "chr6_PATERNAL:115004800-115077200" > ref.fasta
```
