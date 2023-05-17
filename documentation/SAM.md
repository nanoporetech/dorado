# SAM specification

#### Header

```
@HD     VN:1.6  SO:unknown
@PG     ID:basecaller   PN:dorado       VN:0.2.4+3fc2b0f        CL:dorado basecaller dna_r10.4.1_e8.2_400bps_hac@v4.1.0 pod5/
```

#### Read Group Header

|    |    |                                                       |
| -- | -- | ----------------------------------------------------- |
| RG | ID | `<runid>_<basecalling_model>`  	                  |
|    | PU | `<flow_cell_id>`                                      |
|    | PM | `<device_id>`                                         |
|    | DT | `<exp_start_time>`                                    |
|    | PL | `ONT`                                                 |
|    | DS | `basecall_model=<basecall_model_name> runid=<run_id>` |
|    | LB | `<sample_id>`                                         |
|    | SM | `<sample_id>`                                         |

#### Read Tags

|        |                                                            |
| ------ | -----------------------------------------------------------|
| RG:Z:  | `<runid>_<basecalling_model>`                              |
| qs:i:  | mean basecall qscore rounded to the nearest integer        |
| ns:i:  | the number of samples in the signal prior to trimming      |
| ts:i:  | the number of samples trimmed from the start of the signal |
| mx:i:	 | read mux                                                   |
| ch:i:  | read channel                                               |
| rn:i:	 | read number                                                |
| st:Z:	 | read start time (in UTC)                                   |
| du:f:	 | duration of the read (in seconds)                          |
| fn:Z:	 | file name                                                  |
| sm:f:	 | scaling midpoint/mean/median (pA to ~0-mean/1-sd)          |
| sd:f:	 | scaling dispersion  (pA to ~0-mean/1-sd)                   |
| sv:Z:	 | scaling version                                            |
| mv:B:c | sequence to signal move table _(optional)_                 |
| dx:i:  | bool to signify duplex read _(only in duplex mode)_        |

#### Modified Base Tags

When modified base output is requested (via the `--modified-bases` CLI argument), the modified base calls will be output directly in the output files via SAM tags.
The `MM` and `ML` tags are specified in the [SAM format specification documentation](https://samtools.github.io/hts-specs/SAMtags.pdf).
Briefly, these tags represent the relative positions and probability that particular canonical bases have the specified modified bases.

These tags in the SAM/BAM/CRAM formats can be parsed by the [`modkit`](https://github.com/nanoporetech/modkit) software for downstream analysis.
For aligned outputs, visualization of these tags is available in popular genome browsers, including IGV and JBrowse.
