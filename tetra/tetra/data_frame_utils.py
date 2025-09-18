import pathlib

from pandas import concat, read_csv
from pandas.errors import EmptyDataError

from tetra.error_display import pretty_print_diffs

TOLERANCE = 0.0008  # 0.08%


def load_sorted_data_frame(path: pathlib.Path):
    df = None
    try:
        df = read_csv(path, sep="\t", index_col="read_id", keep_default_na=False)
    except (EmptyDataError, ValueError):
        pass

    # Fall back to indexing on the first column
    if df is None:
        df = read_csv(path, sep="\t")
        df.set_index(df.columns[0], inplace=True)

    df.sort_index(inplace=True)
    for col in df.select_dtypes(bool):
        df[col] = df[col].astype(int)

    return df


def merge_csv_files(
    input_files: list[pathlib.Path], output_file: pathlib.Path, sep: str, index_col: str
):
    merged_df = None
    for file in input_files:
        # We check that the file exist, as this function needs to support the possibility that
        # some may not exist.
        if file.is_file():
            file_df = read_csv(file, sep=sep, index_col=index_col)
            if merged_df is None:
                merged_df = file_df
            else:
                merged_df = concat([merged_df, file_df])

    merged_df.sort_index(inplace=True)
    # We output the merged file in case we need to rebase/compare
    merged_df.to_csv(output_file, sep=sep)


def check_pandas_equal(actual, expected, tolerance_factor=1.0):
    # We do a pre-check for equality and pretty-print any diffs if required
    try:
        if expected.equals(actual):
            return ""
    except ValueError:
        # Different length series will be handled by pretty diff
        pass
    pretty_diffs = pretty_print_diffs(
        expected, actual, tolerance=(TOLERANCE * tolerance_factor)
    )
    return pretty_diffs
