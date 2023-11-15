# Sample sheet specification

`dorado` can make use of a MinKNOW-compatible sample sheet containing data used to identify a particular classification of read. To apply a sample sheet, provide the path to the appropriate CSV file using the `--sample-sheet` argument:

```
$ dorado basecaller dna_r10.4.1_e8.2_400bps_hac@v4.2.0 reads/ \
    --kit-name SQK-16S114-24 \
    --sample-sheet <path_to_sample_sheet_csv> \ 
    > calls.bam
```

A sample sheet can also be applied to the `demux` command in the same way:
```
$ dorado demux calls.bam \
    --output-dir classified_reads
    --kit-name SQK-16S114-24 \
    --sample-sheet <path_to_sample_sheet_csv> 
```
Note that `dorado` currently uses the sample sheet only for barcode filtering and aliasing, so a `--kit-name` argument is required.

In the case of `demux`, the sample sheet must contain a 1-to-1 mapping of `barcode` identifiers to `flow_cell_id`/`position_id` - i.e. all entries in the `barcode` column must be unique.

#### Column headers

A sample sheet may only contain the column names below:
|           |                          |                                   |  
| --------- | ------------------------ | --------------------------------- |
| Standard  | `experiment_id`          | Required*                         |
|           | `kit`                    | Required                          |
|           | `flow_cell_id`           | Optional if `position_id` is set  |
|           | `position_id`            | Optional if `flow_cell_id` is set |
|           | `protocol_run_id`        | Optional                          |
|           | `sample_id`              | Optional*                         |
|           | `flow_cell_product_code` | Optional                          |
| Barcoding | `alias`                  | Optional*                         |
|           | `type`                   | Optional                          |
|           | `barcode`                | Optional                          |  

\* These fields must be a maximum of 40 characters, which must be either alphanumeric (`A-Z`, `a-z`, `0-9`), `_` or `-`.

At a minimum a sample sheet must contain `kit`, `experiment_id` and one of `position_id` or `flow_cell_id`. All rows in a sample sheet must contain the same `experiment_id`.

For a full description of the format of the sample sheet, see [the MinKNOW Sample Sheet documentation](https://community.nanoporetech.com/docs/prepare/library_prep_protocols/experiment-companion-minknow/v/mke_1013_v1_revcy_11apr2016/sample-sheet-upload).

Note that `dorado` does not currently support dual barcodes.

#### Barcode filtering

If a sample sheet is present and barcoding is requested, `dorado` will only attempt to find matches to the barcode identifiers listed in the `barcode` column (if present).

#### Barcode aliasing

If a sample sheet contains an `alias` column, this will be used to replace the `barcode` identifer for reads matching the `flow_cell_id`/`position_id` and `experiment_id`. This will be reflected in the read group ID `@RG ID` in the file header, and in the `BC` and `RG` tags of the classified reads. Values in the `alias` column must not be valid barcode identifiers (e.g. `barcode##` or `unclassified`).

Note that if both `flow_cell_id` and `position_id` are present, both must match the read data for an alias to be applied.
