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

from ont_output_specification_validator.validators.bam import (
    BamValidator,
    CramValidator,
)

parser = argparse.ArgumentParser()
parser.add_argument("bam_file", help="Input bam file to validate", type=pathlib.Path)
parser.add_argument(
    "spec_folder",
    help="Folder containing the specification to validate bam_file against",
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
    moves=False,  # Note that moves are not emitted by default by dorado standalone
)

spec_path = pathlib.Path(args.spec_folder)
specification_bundle = load_local_specification_bundle(spec_path)
specification = specification_bundle.get_file_content("bam/spec.yaml")

path = pathlib.Path(args.bam_file)
if path.suffix == ".bam":
    validator = BamValidator(specification)
elif path.suffix == ".cram":
    validator = CramValidator(specification)
else:
    print(f"Unknown file extension '{path.suffix}' for '{path}'")
    sys.exit(1)

validator_errors = errors.ErrorContainer()

validate_file(path, exp_settings, validator_errors, validator)

if validator_errors.has_errors():
    formatted_errors = validator_errors.format_errors()
    print(f"Errors from validating {path}:\n{formatted_errors}")
    sys.exit(1)
else:
    print(f"{validator.name.upper()} file {path} validated")
