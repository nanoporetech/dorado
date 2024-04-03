# Custom Barcode Arrangements

Dorado supports barcode demultiplexing using custom barcode arrangements. These include customizations of existing kits (e.g. using only a subset of the barcodes from a kit) or entirely new kits containing new barcode sequences and layouts.

The format to define a custom arrangement is inspired by the arrangement specification in Guppy, with some adjustments to account for the algorithmic changes in Dorado.

## Barcode Reference Diagram

A double-ended barcode with different flanks and barcode sequences for front and rear barcodes is described here.

```
5' ---- ADAPTER/PRIMER ---- LEADING_FLANK_1 ---- BARCODE_1 ---- TRAILING_FLANK_1 --- READ --- RC(TRAILING_FLANK_2) --- RC(BARCODE_2) --- RC(LEADING_FLANK_2) --- 3'
```

* For single-ended barcodes, there is no barcode sequence at the rear of the read.
* For double-ended barcodes which are symmetric, the flank and barcode sequences for front and rear windows are same.

## Specification Format

The custom arrangements are defined using a `toml` file, and custom barcode sequences are passed through a `FASTA` file.

### Arrangement File

The following are all the options that can be defined in the arrangement file.

```
[arrangement]
name = "custom_barcode"
kit = "BC"

mask1_front = "ATCG"
mask1_rear = "ATCG"
mask2_front = "TTAA"
mask2_rear = "GGCC"

# Barcode sequences
barcode1_pattern = "BC%02i"
barcode2_pattern = "BC%02i"
first_index = 1
last_index = 96

## Scoring options
[scoring]
max_barcode_penalty = 11
barcode_end_proximity = 75
min_barcode_penalty_dist = 3
min_separation_only_dist = 6
flank_left_pad = 5
flank_right_pad = 10
front_barcode_window = 175
rear_barcode_window = 175
```

#### Arrangement Options

The table below describes the arrangement options in more detail.

| Option | Description |
| -- | -- |
| name | (Required) Name of the barcode arrangement. This name will be used to report the barcode classification. |
| kit | (Optional) Which class of barcodes this arrangement belongs to (if any). |
| mask_1_front | (Required) The leading flank for the front barcode (applies to single and double ended barcodes). Can be an empty string. |
| mask_1_rear | (Required) The trailing flank for the front barcode (applies to single and double ended barcodes). Can be an empty string. |
| mask_2_front | (Optional) The leading flank for the rear barcode (applies to double ended barcodes only). Can be an empty string. |
| mask_2_rear | (Optional) The trailing flank for the rear barcode (applies to double ended barcodes only). Can be an empty string. |
| barcode1_pattern | (Required) An expression capturing the sequences to use for the front barcode. Pattern must match sequences from pre-built list in Dorado or in the custom sequences file. |
| barcode2_pattern | (Optional) An expression capturing the sequences to use for the rear barcode. Pattern must match sequences from pre-built list in Dorado or in the custom sequences file. |
| first_index | (Required) Start index for range of barcode sequences to use in the arrangement. Used in combination with the `last_index`. |
| last_index | (Required) End index for range of barcode sequences to use in the arrangement. Used in combination with the `first_index`. |

The pre-built barcode sequence in Dorado can be found in this [file](../dorado/utils/barcode_kits.cpp) under the `barcodes` map.

#### Scoring Options

Dorado maintains a default set of parameters for scoring each barcode to determine the best classification. These parameters have been tuned based on barcoding kits from Oxford Nanopore. However, the default parameters may not be optimal for new arrangements and kits.

The classification heuristic applied by Dorado is the following -
1. Dorado uses the flanking sequences defined in `maskX_front/rear` to find a window in the read where the barcode is situated. 
2. For double ended barcodes, the __best__ window (either from the front or rear of the read) is chosen based on the alignment of the flanking mask sequences.
3. After choosing the best window for an arrangement, each barcode candidate within the arrangement is aligned to the subsequence within the window. The alignment may optionally consider additional bases from the preceding/succeeding flank (as specifed in the `flank_left_pad` and `flank_right_pad` parameters). The edit distance of this alignment is assigned as a penalty to each barcode.

Once barcodes are sorted by barcode penalty, the top candidate is checked against the following rules -
1. Is the barcode penalty below `max_barcode_penalty` and the distance between top 2 barcode penalties greater than `min_barcode_penalty_dist`?.
2. Is the barcode penalty above `max_barcode_penalty` but the distance between top 2 barcodes penalties greater then `min_separation_only_dist`?
3. Is the flank score below the `min_flank_score`?

If a candidate meets (1) or (2) AND (3), and the location of the start/end of the barcode construct is within `barcode_end_proximity` bases of the ends of the read, then it is considered a hit.

| Scoring option | Description |
| -- | -- |
| max_barcode_penalty | The maximum edit distance allowed for a classified barcode. Considered in conjunction with the `min_barcode_penalty_dist` parameter. |
| min_barcode_penalty_dist | The minimum penalty difference between top-2 barcodes required for classification. Used in conjunction with `max_barcode_cost`. |
| min_separation_only_dist | The minimum penalty difference between the top-2 barcodes required for classification when the `max_barcode_cost` is not met. |
| barcode_end_proximity | Proximity of the end of the barcode construct to the ends of the read required for classification. |
| flank_left_pad | Number of bases to use from preceding flank during barcode alignment. |
| flank_right_pad | Number of bases to use from succeeding flank during barcode alignment. |
| front_barcode_window | Number of bases at the front of the read within which to look for barcodes. |
| rear_barcode_window | Number of bases at the rear of the read within which to look for barcodes. |
| min_flank_score | Minimum score for the flank alignment. Score here is 1.f - (edit distance) / flank_length |

For `flank_left_pad` and `flank_right_pad`, something in the range of 5-10 bases is typically good. Note that errors from this padding region are also part of the barcode alignment penalty. Therefore a bigger padding region may require a higher `max_barcode_cost` for classification.

### Custom Sequences File 

In addition to specifying a custom barcode arrangement, new barcode sequences can also be specified in a FASTQ format. There are only 2 requirements -
* The sequence names to follow the `prefix%\d+i` format (e.g. `BC%02i` for barcodes needing 2 digit indexing, or `NB%04i` for barcodes with 4 digit indexing, etc.).
* All barcode sequence lengths must match.

This is an example sequences file.

```
>BC01
TTTT
>BC02
AAAA
>BC03
GGGG
>BC04
CCCC
```
