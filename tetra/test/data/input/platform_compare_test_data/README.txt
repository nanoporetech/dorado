The folders `platform1` and `platform2` differ as follows:
1) The `sequencing_summary.txt` files in the `basecalling_tests` folders differ by a small amount.
2) The `sequencing_summary.txt` files in the `barcoding_tests` folders are completely different,
   but should pass due to the `special_case` tolerance being set to `None`.
3) The `passes_filtering` columns in the `sequencing_summary.txt` files in the `alignment_tests`
   folders differ, which should be allowed by the tests with the `exclude_columns` option.
