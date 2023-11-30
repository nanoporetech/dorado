# Custom Barcode Arrangements

Dorado supports barcode demultiplexing using custom barcode arrangements. These include customizations of existing kits (e.g. using only a subset of the barcodes from a kit) or entirely new kits containing new barcode sequences and layouts.

The format to define a custom arrangement is inspired by the arrangement specification in Guppy, with some adjustments to account for the algorithmic changes in Dorado.

## Specification Format

The custom arrangements are defined using a `toml` file, and custom barcode sequences are passed through a `FASTQ` file.

### Arrangement File

The following are all the options that can be defined in the arrangement file.

```
[arrangement]
name = "custom_barcode"
kit = "BC"

mask_1_front = "ATCG"
mask_1_rear = "ATCG"
mask_2_front = "TTAA"
mask_2_rear = "GGCC"

# Barcode sequences
barcode1_pattern = "BC%02i"
barcode2_pattern = "BC%02i"
first_index = 1
last_index = 96

## Scoring options
min_soft_barcode_threshold = 0.2
min_hard_barcode_threshold = 0.2
min_soft_flank_threshold = 0.3
min_hard_barcode_threshold = 0.3
min_barcode_score_dist = 0.1
```

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
