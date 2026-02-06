# CRAM Single Read Test Data

This directory contains test data for asserting that Dorado creates valid M5 hashes with FASTA reference containing IUPAC ambiguity codes.

Files:

- `chr1_MAT.pod5` - a single native chr1-MATERNAL na24385 read from a prom r10.4.1 400bps 5kHz flowcell 327be0dc.
- `chr1_MAT_iupac.fasta` - reference to span this single read but it has also been "doped" with IUPAC ambiguity codes.
- `chr1_MAT_iupac.fasta.fai` - reference index.
- `chr1_MAT_iupac.dict` - The samtools dict output for the reference which contains the expeted SQ M5 value.
