import pathlib
from collections import defaultdict

from Bio import SeqIO

from tetra.utilities import get_platform

USE_PYSAM = True
try:
    import pysam
except Exception:
    USE_PYSAM = False


def compare_fastq_files(file_a: pathlib.Path, file_b: pathlib.Path) -> tuple[bool, str]:
    record_dict_a = _load_fastq_file(file_a)
    record_dict_b = _load_fastq_file(file_b)
    count_a = len(record_dict_a)
    count_b = len(record_dict_b)
    mismatches: list[str] = []
    if count_a == count_b:
        total_count = count_a
        match_count = 0
        for key_a, record_a in record_dict_a.items():
            if key_a in record_dict_b:
                if _compare_fastq_records(record_a, record_dict_b[key_a], mismatches):
                    match_count += 1
            else:
                mismatches.append(f"Record {key_a} missing in file {file_b}.")
        if match_count == total_count:
            result = True
            message = "Fastq records in files match"
        else:
            result = False
            message = f"Fastq record mismatch: {total_count - match_count} of {total_count} records do not match.\nFirst mismatch: {mismatches[0]}"
    else:
        result = False
        message = f"Fastq record mismatch: {count_a} records found, {count_b} records expected"
    return result, message


def compare_alignment_files(
    file_a: pathlib.Path, file_b: pathlib.Path, print_all_errors: bool = False
) -> tuple[bool, str]:
    # The pysam module is not available on all platforms. This function should throw
    # if it is called on an unsupported platform.
    if not USE_PYSAM:
        raise Exception(
            "Can't compare BAM/SAM files on platforms where Pysam is unavailable."
        )
    with (
        pysam.AlignmentFile(file_a, check_sq=False) as handle_a,
        pysam.AlignmentFile(file_b, check_sq=False) as handle_b,
    ):
        errors = []
        header_a = handle_a.header.to_dict()
        header_b = handle_b.header.to_dict()
        error = _compare_sam_headers(header_a, header_b, print_all_errors)
        if error is not None:
            errors.append(error)

        records_a = _get_sorted_records(handle_a)
        records_b = _get_sorted_records(handle_b)
        error = _compare_all_sam_records(records_a, records_b, print_all_errors)
        if error is not None:
            errors.append(error)
    if errors:
        result = False
        message = "\n".join(errors)
    else:
        result = True
        message = "SAM files match."
    return result, message


# Internal support functions for comparing fastq files.


def _load_fastq_file(filename: pathlib.Path) -> dict:
    with filename.open() as file:
        return SeqIO.to_dict(SeqIO.parse(file, "fastq"))


def _split_token(token: str) -> tuple[str, str | None]:
    # We normally expect HTS-style tags, which will be like `XX:X:[value]`.
    # For such tokens, we return (tag_name, tag_value).
    if len(token) > 5 and token[2] == ":" and token[4] == ":":
        return (token[0:5], token[5:])
    # For other tokens, return the token as the first part, and None as the
    # second part.
    return (token, None)


def _compare_fastq_descr(descr1: str, descr2: str) -> bool:
    # We expect our fastq files to have only tab-delimited HTS-style tags after the
    # record-id. We don't care if those tags are in the same order, and we may want
    # to allow some keys to have differing values on some platforms.
    skip_field = []
    if get_platform() == "osx_arm":
        skip_field.append("DS:Z:")  # GPU name can vary between CI runners.
    tokens1 = descr1.split("\t")
    tokens2 = descr2.split("\t")
    split_tokens1 = [_split_token(token) for token in tokens1]
    split_tokens2 = [_split_token(token) for token in tokens2]
    tags1 = {key: value for key, value in split_tokens1}
    tags2 = {key: value for key, value in split_tokens2}
    if len(tags1) != len(tags2):
        return False
    for key, value in tags1.items():
        if key not in tags2:
            return False
        if key not in skip_field and value != tags2[key]:
            return False
    return True


def _compare_fastq_records(
    record_a: dict, record_b: dict, mismatches: list[str]
) -> bool:
    phred_a = None
    phred_b = None
    if "phred_quality" in record_a.letter_annotations:
        phred_a = record_a.letter_annotations["phred_quality"]
    if "phred_quality" in record_b.letter_annotations:
        phred_b = record_b.letter_annotations["phred_quality"]
    records_match = True
    if record_a.name != record_b.name:
        records_match = False
        mismatches.append(f"Record name '{record_a.name}' != '{record_b.name}'.")
    if record_a.seq != record_b.seq:
        records_match = False
        mismatches.append(f"Sequences for record '{record_a.name}' do not match.")
    if phred_a != phred_b:
        records_match = False
        mismatches.append(f"Quality scores for record '{record_a.name}' do not match.")
    if not _compare_fastq_descr(record_a.description, record_b.description):
        records_match = False
        mismatches.append(
            f"Record headers do not match:\nRecord 1: {record_a.description}\nRecord 2: {record_b.description}"
        )
    return records_match


# Internal support functions for comparing SAM/BAM files.


def _compare_sam_headers(
    header_a: dict, header_b: dict, print_all_errors: bool
) -> str | None:
    header_errors = []
    # The first element of tuple is the header line type.
    # The second element lists any fields that should not have their values compared.
    # The third element indicates whether header lines of that type are required.
    line_info = [
        ("HD", ["VN"], True),
        ("PG", ["VN", "CL", "DS"], True),
        ("RG", ["DT"], True),
        ("SQ", [], False),
    ]
    for line_type, skip_fields, is_required in line_info:
        if line_type in header_a and line_type in header_b:
            header_errors.extend(
                _compare_line_type(
                    header_a, header_b, line_type, skip_fields=skip_fields
                )
            )
        else:
            if is_required:
                header_errors.append(
                    f"SAM header mismatch: {line_type} line(s) missing"
                )
            else:
                if line_type in header_a:
                    header_errors.append(
                        f"SAM header mismatch: Unexpected {line_type} line(s) found"
                    )
                if line_type in header_b:
                    header_errors.append(
                        f"SAM header mismatch: Expected {line_type} line(s), but none found"
                    )
    if header_errors:
        if print_all_errors:
            for error in header_errors:
                print(error)
        return f"SAM headers do not match:\nFirst mismatch: {header_errors[0]}"
    else:
        return None


def _compare_line_type(
    header_a: dict, header_b: dict, line_type: str, skip_fields: list[str]
) -> list[str]:
    lines_a = header_a[line_type]
    lines_b = header_b[line_type]
    if line_type == "HD":
        # The HD field contains the single line as a dict, rather than a list of lines (each as a dict).
        lines_a = [lines_a]
        lines_b = [lines_b]
    errors = []
    if len(lines_a) == len(lines_b):
        for line_a, line_b in zip(lines_a, lines_b):
            ok = _compare_header_line(line_a, line_b, skip_fields=skip_fields)
            if not ok:
                errors.append(
                    f"{line_type} line mismatch:\n\tFound: {line_a}\n\tExpected: {line_b}"
                )
    else:
        errors.append(
            f"SAM header mismatch: Found {len(lines_a)} {line_type} lines, expected {len(lines_b)}"
        )
    return errors


def _compare_header_line(line_a: dict, line_b: dict, skip_fields: list[str]) -> bool:
    keys1 = set(line_a.keys())
    keys2 = set(line_b.keys())
    if keys1 != keys2:
        return False
    for key in keys1:
        if key in skip_fields:
            continue
        if line_a[key] != line_b[key]:
            return False
    return True


def _get_sorted_records(sam_data) -> list:
    data = list(sam_data.fetch(until_eof=True))
    return sorted(data, key=lambda x: x.query_name)


def _compare_fields(field1, field2, comp: defaultdict) -> bool:
    if comp is None:
        return field1 == field2
    return comp(field1, field2)


def _compare_all_sam_records(
    records_a: list, records_b: list, print_all_errors: bool
) -> str | None:
    record_errors = []
    comp_map = defaultdict(lambda: None)
    if len(records_a) == len(records_b):
        for record_a, record_b in zip(records_a, records_b):
            error_message = _compare_sam_records(record_a, record_b, comp_map)
            if error_message is not None:
                record_errors.append(error_message)
    else:
        return f"SAM record mismatch: {len(records_a)} records found, {len(records_b)} records expected"
    if record_errors:
        if print_all_errors:
            for error in record_errors:
                print(error)
        return f"SAM record mismatch: {len(record_errors)} of {len(records_a)} do not match.\nFirst mismatch: {record_errors[0]}"
    else:
        return None


def _compare_sam_records(record1, record2, comp_map: defaultdict) -> str | None:
    if not _compare_fields(record1.query_name, record2.query_name, comp_map["qname"]):
        return f"QNAME fields do not match (found {record1.qname}, expected {record2.qname})."
    if not _compare_fields(record1.flag, record2.flag, comp_map["flag"]):
        return (
            f"FLAG fields do not match (found {record1.flag}, expected {record2.flag})."
        )
    if not _compare_fields(
        record1.reference_id, record2.reference_id, comp_map["rname"]
    ):
        return f"RNAME fields do not match (found {record1.reference_id}, expected {record2.reference_id})."
    if not _compare_fields(
        record1.reference_start, record2.reference_start, comp_map["pos"]
    ):
        return f"POS fields do not match (found {record1.reference_start}, expected {record2.reference_start})."
    if not _compare_fields(
        record1.mapping_quality, record2.mapping_quality, comp_map["mapq"]
    ):
        return "MAPQ fields do not match (found {}, expected {}).".format(
            record1.mapping_quality, record2.mapping_quality
        )
    if not _compare_fields(record1.cigarstring, record2.cigarstring, comp_map["cigar"]):
        return "CIGAR fields do not match."
    if not _compare_fields(
        record1.next_reference_id, record2.next_reference_id, comp_map["rnext"]
    ):
        return f"RNEXT fields do not match (found {record1.next_reference_id}, expected {record2.next_reference_id})."
    if not _compare_fields(
        record1.next_reference_start, record2.next_reference_start, comp_map["pnext"]
    ):
        return f"PNEXT fields do not match (found {record1.next_reference_start}, expected {record2.next_reference_start})."
    if not _compare_fields(
        record1.template_length, record2.template_length, comp_map["tlen"]
    ):
        return f"TLEN fields do not match (found {record1.template_length}, expected {record2.template_length})."
    if not _compare_fields(
        record1.query_sequence, record2.query_sequence, comp_map["seq"]
    ):
        return "SEQ fields do not match."
    if not _compare_fields(
        record1.query_qualities, record2.query_qualities, comp_map["qual"]
    ):
        return "QUAL fields do not match"

    tags1 = record1.get_tags(with_value_type=True)
    tags2 = record2.get_tags(with_value_type=True)

    if len(tags1) != len(tags2):
        return (
            f"Tags do not match ({len(tags1)} tags found, {len(tags2)} tags expected)."
        )

    tag_map1 = {name: (value, typename) for name, value, typename in tags1}
    tag_map2 = {name: (value, typename) for name, value, typename in tags2}

    for key, (value, typename) in tag_map1.items():
        if key not in tag_map2:
            return f"Unexpected tag {key} found."
        value2, typename2 = tag_map2[key]
        if typename != typename2:
            return f"Tag types for tag {key} do not match (found {typename}, expected {typename2})."
        if not _compare_fields(value, value2, comp_map[key]):
            return f"Tag values for tag {key} do not match (found {value}, expected {value2})."
    return None
