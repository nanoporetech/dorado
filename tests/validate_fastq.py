import argparse
import pathlib
import sys

from ont_output_specification_validator import (
    errors,
    settings,
    validate_file,
    validators,
)

from ont_output_specification_validator.specification import (
    load_local_specification_bundle,
)

from ont_output_specification_validator.validators.fastq import HtsFastqValidator

parser = argparse.ArgumentParser()
parser.add_argument("fastq_file", help="Input fastq file to validate")
parser.add_argument(
    "spec_folder",
    help="Folder containing the specification to validate fastq_file against",
)
args = parser.parse_args()

exp_settings = settings.ExperimentSettings(
    gpu_calling_enabled=False,
    basecalling_enabled=False,
    duplex_enabled=False,
    alignment_enabled=False,
    bed_file_enabled=False,
    modified_bases_enabled=False,
    barcode_count=False,
    estimate_polya_tail_enabled=False,
    has_input_sample_sheet=True,
    offline_analysis_enabled=False,
)

spec_path = pathlib.Path(args.spec_folder)
specification_bundle = load_local_specification_bundle(spec_path)
validator = HtsFastqValidator(
    specification_bundle.get_file_content("fastq/header-spec-hts.yaml")
)
validator_errors = errors.ErrorContainer()

validate_file(pathlib.Path(args.fastq_file), exp_settings, validator_errors, validator)

if validator_errors.has_errors():
    formatted_errors = validator_errors.format_errors()
    print(f"Errors from validation:\n{formatted_errors}")
    sys.exit(1)
else:
    print(f"FastQ file {args.fastq_file} validated")
