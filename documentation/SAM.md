# SAM specification

#### Header

```
@HD     VN:1.6  SO:unknown
@PG     ID:basecaller   PN:dorado       VN:0.2.4+3fc2b0f        CL:dorado basecaller dna_r10.4.1_e8.2_400bps_hac@v4.1.0 pod5/        DS:gpu:Quadro GV100
```

#### Read Group Header

|    |    |                                                                                            |
| -- | -- | ------------------------------------------------------------------------------------------ |
| RG | ID | `<runid>_<basecalling_model>_<barcode_arrangement>`                                        |
|    | PU | `<flow_cell_id>`                                                                           |
|    | PM | `<device_id>`                                                                              |
|    | DT | `<exp_start_time>`                                                                         |
|    | PL | `ONT`                                                                                      |
|    | DS | `basecall_model=<basecall_model_name> modbase_models=<modbase_model_names> runid=<run_id>` |
|    | LB | `<sample_id>`                                                                              |
|    | SM | `<sample_id>`                                                                              |

#### Read Tags

|        |                                                            |
| ------ | -----------------------------------------------------------|
| RG:Z:  | `<runid>_<basecalling_model>_<barcode_arrangement>`        |
| qs:f:  | mean basecall qscore                                       |
| ts:i:  | the number of samples trimmed from the start of the signal |
| ns:i:  | the basecalled sequence corresponds to the interval `signal[ts : ns]` <br /> the move table maps to the same interval. <br /> note that `ns` reflects trimming (if any) from the rear <br /> of the signal. |
| mx:i:  | read mux                                                   |
| ch:i:  | read channel                                               |
| rn:i:  | read number                                                |
| st:Z:  | read start time (in UTC)                                   |
| du:f:  | duration of the read (in seconds)                          |
| fn:Z:  | file name                                                  |
| sm:f:  | scaling midpoint/mean/median (pA to ~0-mean/1-sd)          |
| sd:f:  | scaling dispersion  (pA to ~0-mean/1-sd)                   |
| sv:Z:  | scaling version                                            |
| mv:B:c | sequence to signal move table _(optional)_                 |
| dx:i:  | bool to signify duplex read _(only in duplex mode)_        |
| pi:Z:  | parent read id for a split read                            |
| sp:i:  | start coordinate of split read in parent read signal       |
| pt:i:  | estimated poly(A/T) tail length in cDNA and dRNA reads     |
| bh:i:  | number of detected bedfile hits _(only if alignment was performed with a specified bed-file)_ |
| MN:i:  | Length of sequence at the time MM and ML were produced     |

#### Modified Base Tags

When modified base output is requested (via the `--modified-bases` CLI argument), the modified base calls will be output directly in the output files via SAM tags.
The `MM` and `ML` tags are specified in the [SAM format specification documentation](https://samtools.github.io/hts-specs/SAMtags.pdf).
Briefly, these tags represent the relative positions and probability that particular canonical bases have the specified modified bases.

These tags in the SAM/BAM/CRAM formats can be parsed by the [`modkit`](https://github.com/nanoporetech/modkit) software for downstream analysis.
For aligned outputs, visualization of these tags is available in popular genome browsers, including IGV and JBrowse.

#### Minimap2 Alignment Tags

When `dorado` is run with alignment enabled, additional tags from minimap2 are added to each SAM record. Details of those tags
are available on the [minimap2 manpage](https://lh3.github.io/minimap2/minimap2.html#10).

### Split Read Tags

When a single input read contains multiple concatenated reads, `dorado basecaller` will split the original input read into separate subreads. This operation is performed by default for both DNA and RNA. Each subread has a new read id that is assigned by `dorado`. The following tags can be used to associate a subread to its parent:

* `pi:Z` contains the parent read id the subread was generated from.
* `sp:i` maps the start of the subread's signal data to the corresponding location in the parent read's signal data.
* `ns:i` is the number of samples corresponding to the subread after splitting.
* `ts:i` is the number samples trimmed from the start of subread's signal after splitting.
