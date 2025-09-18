import numpy
import pandas


def pretty_print_diffs(exp, act, tolerance=0):
    # First we check which rows/columns fail to match
    pretty_diffs = []
    r_match, r_exp, r_act = row_match(exp, act)
    if r_exp:
        pretty_diffs.append(
            "SizeError({} Rows only in expected: {})".format(
                len(r_exp), truncated_print(r_exp)
            )
        )
    if r_act:
        pretty_diffs.append(
            "SizeError({} Rows only in actual  : {})".format(
                len(r_act), truncated_print(r_act)
            )
        )
    c_match, c_exp, c_act = col_match(exp, act)
    if c_exp:
        pretty_diffs.append(
            "SizeError({} Columns only in expected: {})".format(
                len(c_exp), truncated_print(c_exp)
            )
        )
    if c_act:
        pretty_diffs.append(
            "SizeError({} Columns only in actual  : {})".format(
                len(c_act), truncated_print(c_act)
            )
        )

    # If any rows/columns are different we need to trim the dfs before comparing values
    if r_act or r_exp or c_act or c_exp:
        act = act.loc[list(r_match), list(c_match)]
        exp = exp.loc[list(r_match), list(c_match)]
    diffs = find_value_differences(exp, act)

    if not diffs.empty:
        row_count = len(exp.index.values)
        for col_name in diffs.columns.values:
            try:
                col_tolerance = tolerance
                if (
                    "barcode_" in col_name
                    or "alignment_" in col_name
                    or "lamp_" in col_name
                    or "adapter_" in col_name
                    or "primer_" in col_name
                ):
                    col_tolerance = (
                        col_tolerance * 12.5
                    )  # equates to 1% with TOLERANCE at 0.08%
                numeric_diff = abs(
                    numpy.mean(exp[col_name]) - numpy.mean(act[col_name])
                ) / numpy.mean(exp[col_name])
                if numeric_diff >= col_tolerance:
                    diff_count = numpy.sum(
                        numpy.abs(exp[col_name] - act[col_name])
                        > numpy.abs(exp[col_name] * col_tolerance)
                    )
                    pretty_diffs.append(
                        '\t{}/{} rows\t mean("{}"); : expected {}, actual {} diff {}% tolerance {}%'.format(
                            diff_count,
                            row_count,
                            col_name,
                            numpy.mean(exp[col_name]),
                            numpy.mean(act[col_name]),
                            numeric_diff * 100,
                            col_tolerance * 100,
                        )
                    )
            except TypeError:
                diff_count = numpy.sum(exp[col_name] != act[col_name])
                pretty_diffs.append(
                    '\t{}/{} rows\t "{}": Numeric diff failed on {} rows'.format(
                        diff_count, row_count, col_name, diff_count
                    )
                )
    return "\n".join(pretty_diffs)


def row_match(exp, act):
    act_rows = set(act.index.values)
    exp_rows = set(exp.index.values)
    return act_rows.intersection(exp_rows), exp_rows - act_rows, act_rows - exp_rows


def col_match(exp, act):
    act_cols = set(act.columns.values)
    exp_cols = set(exp.columns.values)
    return act_cols.intersection(exp_cols), exp_cols - act_cols, act_cols - exp_cols


def find_value_differences(exp, act):
    # Cut down the array to only where we have differences
    diffs = act != exp
    act_diffs = (
        act.where(diffs)
        .dropna(axis="columns", how="all")
        .dropna(axis="rows", how="all")
    )
    exp_diffs = (
        exp.where(diffs)
        .dropna(axis="columns", how="all")
        .dropna(axis="rows", how="all")
    )

    # Find the numeric differences between numeric values
    act_numeric = act_diffs.select_dtypes(numpy.number)
    exp_numeric = exp_diffs.select_dtypes(numpy.number)
    num_diffs = act_numeric - exp_numeric

    # Find the differences in columns with text strings and format nicely
    str_act = act_diffs.select_dtypes(exclude=numpy.number)
    str_exp = exp_diffs.select_dtypes(exclude=numpy.number)

    try:
        str_exp = str_exp.where(str_exp != str_act, "-")
        str_exp = str_exp.where(str_exp == str_act, str_exp + " -> " + str_act)
    except ValueError as e:
        print(
            "Expected columns: {}\nActual columns: {}\n Error:\n{}".format(
                str_exp.columns.values, str_act.columns.values, e
            )
        )
    # Rejoin the numeric and string differences and reinstate column order
    visual_diffs = pandas.concat([num_diffs, str_exp], axis=1)[
        act_diffs.columns.tolist()
    ]
    return visual_diffs


def truncated_print(input_data):
    input_data = list(str(x) for x in input_data)
    if len(input_data) > 3:
        return ", ".join(input_data[:2]) + ", ..., " + input_data[-1]
    else:
        return ", ".join(input_data)
