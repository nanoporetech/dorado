### Custom Adapter and Primer Sequences

Dorado will normally automatically detect and trim any adapter or primer sequences it finds. The specific sequences it searches for depend on the specified sequencing kit. This applies to both the basecaller subcommand, where the kit name is expected to be embedded in the read in the input pod5 file, and the trim subcommand, where the kit must be specified as a command-line option to dorado.

In some cases, it may be necessary to find and remove adapter and/or primer sequences that would not normally be associated with the sequencing kit that was used, or you may be working with older data for which the sequencing kit and/or primers being used are no longer directly supported by dorado (for example, anything prior to kit14). In such cases, you can specify a custom adapter/primer file, using the command-line option `--primer-sequences`.

If this option is used, then the sequences encoded in the specified file will be used instead of the built-in sequences that dorado normally searches for.

#### Custom adapter/primer file format

The custom adapter/primer file is really just a fasta file, with the desired sequences specified within. However, some additional metadata is needed to allow dorado to properly interpret how the sequences should be used.

* The record name for each sequence must be of the form `[id]_front` or `[id]_rear`.
* The `id` part of the record name may occur, at most, twice in the file: Once with `_front` and once with `_rear`.
* Immediately following the record name must be the following two, tab-delimited HTS-style tags:
  - `et:Z:[sequence_type]`, where `[sequence_type]` can be either `adapter` or `primer`.
  - `sk:Z:[sequencing_kits]`, where `[sequencing_kits]` can be `any` or a list of comma-seperated sequencing kit names (e.g., `[kit1],[kit2],[kit3]`).

The `_front` and `_rear` part of the record name tells dorado how to search for the sequence. In the case of adapters, dorado will look for the `front` sequence near the beginning of the read, and for the `rear` sequence near the end of the read. For primers things work a bit differently. Dorado will look for the `front` sequence near the beginning of the read, and the reverse-complement of the `rear` sequence near the end of the read. It will also look for the `rear` sequence near the beginning of the read, and the reverse-complement of the `front` sequence near the end of the read. This allows dorado to infer whether the forward or reverse strand has been sequenced. If dorado is able to infer this from the primers, then the BAM tag `TS` will be used to record this, using `+` for forward reads and `-` for reverse reads. If the sense cannot be inferred, the `TS` tag is not written.

The `et:Z:` tag is required to designate whether the sequence is an adapter or a primer sequence, so that dorado knows how it should be used.

The `sk:Z:` tag is required to indicate which sequencing kit the adapter or primer sequence may be used with. The sequence will only be searched for if the sequencing-kit information in the read matches one of the kit names in the custom file. If the `sk:Z:` tag has the value `any`, then the sequence will be searched for in all reads, regardless of the kit that was used. Note that the kit names are case-insensitive.

#### Example custom adapter/primer file.

The following could be used to detect the PCR_PSK_rev1 and PCR_PSK_rev2 primers, along with the LSK109 adapters, for older data.

```
>LSK109_front   et:Z:adapter    sk:Z:any
AATGTACTTCGTTCAGTTACGTATTGCT

>LSK109_rear    et:Z:adapter    sk:any
AGCAATACGTAACTGAACGAAGT

>PCR_PSK_front  et:Z:primer sk:any
ACTTGCCTGTCGCTCTATCTTCGGCGTCTGCTTGGGTGTTTAACC

>PCR_PSK_rear   et:Z:primer sk:any
AGGTTAAACACCCAAGCAGACGCCGCAATATCAGCACCAACAGAAA
```

In this case, the above adapters and primers would be searched for in all reads, regardless of the sequencing-kit information encoded in the read file, or in the case of dorado trim, regardless of the sequencing-kit specified on the command-line. If you wanted to restrict the software so that the primers would only be searched for in reads with `SQK-PSK004` specified as the kit name, and the adapters would only be searched for if the kit name was specified as either `SQK-PSK004` or `SQK-LSK109`, then the following could be used.

```
>LSK109_front   et:Z:adapter    sk:Z:SQK-PSK004,SQK-LSK109
AATGTACTTCGTTCAGTTACGTATTGCT

>LSK109_rear    et:Z:adapter    sk:Z:SQK-PSK004,SQK-LSK109
AGCAATACGTAACTGAACGAAGT

>PCR_PSK_front  et:Z:primer sk:Z:SQK-PSK004
ACTTGCCTGTCGCTCTATCTTCGGCGTCTGCTTGGGTGTTTAACC

>PCR_PSK_rear   et:Z:primer sk:Z:SQK-PSK004
AGGTTAAACACCCAAGCAGACGCCGCAATATCAGCACCAACAGAAA
```

Note that for cDNA type experiments, the 3' primer will actually appear at the beginning of the forward strand, and the 5' primer at the end of the forward strand. For this reason, it is important that for this type of experiment your custom primer file must list what would normally be considered the "back" primer as the `front` sequence, and what would normally be considered the "front" primer as the `rear` sequence.
