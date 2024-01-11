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

The custom arrangements are defined using a `toml` file, and custom barcode sequences are passed through a `FASTQ` file.

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
min_soft_barcode_threshold = 0.2
min_hard_barcode_threshold = 0.2
min_soft_flank_threshold = 0.3
min_hard_flank_threshold = 0.3
min_barcode_score_dist = 0.1
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
1. Dorado calculates a separate score for the barcode flanks and the barcode sequence itself. The score is roughly `1.0f - edit_distance / length(target_seq)`, where the `target_seq` is either the reference flank or reference barcode sequence.
2. For double ended barcodes, the __best__ window (either front or rear) is chosen based on the flank scores.
3. After choosing the best window for an arrangement, each barcode candidate within the arrangement is sorted by the barcode score.

Once barcodes are sorted by barcode score, the top candidate is checked against the following rules -
1. If the flank score is above `min_soft_flank_threshold` and the barcode score is above `min_hard_barcode_threshold`, then the barcode is kept as a candidate.
2. If the barcode score is above `min_soft_barcode_threshold` and the flank score is above `min_hard_flank_threshold`, then the barcode is kept as a candidate.
3. If the arrangement is double ended, if both the top and bottom barcode sequence scores are above `min_hard_barcode_threshold`, then the barcode is kept as a candidate.

If a candidate still remains and there are at least 2 scores candidates in the sorted list, the difference between the best and second best candidate is computed. Only
if that score is greater than `min_barcode_score_dist`, the best candidate is considered a hit. Otherwise the read is considered as `unclassified`.

| Scoring option | Description |
| -- | -- |
| min_soft_barcode_threshold | If barcode score meets this threshold and flank score meets its hard threshold, consider a hit. Soft score is higher than hard score. |
| min_hard_barcode_threshold | Minimum score threshold a barcode must meet. |
| min_soft_flank_threshold | If flank score meets this threshold and barcode score meets its hard threshold, consider a hit. Soft score is higher than hard score. |
| min_hard_barcode_threshold | Minimum score threshold a flank must meet. |
| min_barcode_score_dist | Minimum distance between barcode scores of best and second best hits. |

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
