import copy
import pathlib

from ont_output_specification_validator import validate_file
from ont_output_specification_validator.errors import ErrorContainer
from ont_output_specification_validator.settings import ExperimentSettings
from ont_output_specification_validator.specification import (
    SpecificationBundle,
    load_local_specification_bundle,
)
from ont_output_specification_validator.validators import (
    BamValidator,
    FastqValidator,
    SequencingSummaryValidator,
    Validator,
)

from tetra import ValidationOptions

DEFAULT_SETTINGS = {
    "gpu_calling_enabled": False,
    "alignment_enabled": False,
    "modified_bases_enabled": False,
    "bed_file_enabled": False,
    "basecalling_enabled": True,
    "barcode_count": 0,
    "duplex_enabled": False,
    "has_input_sample_sheet": False,
    "estimate_polya_tail_enabled": False,
    "offline_analysis_enabled": True,
    "moves": False,
}


class OutputValidator(object):
    def __init__(self, spec_file: str):
        spec_path = pathlib.Path(spec_file)
        self.specification_bundle = load_local_specification_bundle(spec_path)
        self.settings = copy.deepcopy(DEFAULT_SETTINGS)

    def validate_file(self, file: pathlib.Path) -> tuple[bool, str | None]:
        """
        Validate a single file.
        file (pathlib.Path): The file to be validated.
        return (tuple): True or False, indicating whether the validation was successful or not,
            followed by a string containing all detected validation errors (or `None` if there
            were no errors.
        """
        type_map = {".fastq": "fastq", ".bam": "bam", ".txt": "sequencing_summary"}
        extension = file.suffix
        file_type = type_map[extension]
        file_validator = validator_from_name(file_type, self.specification_bundle)
        if file_validator is None:
            return (
                False,
                f"Files of type {extension} are not a supported for validation.",
            )
        validator_errors = ErrorContainer()
        validator_settings = ExperimentSettings(**self.settings)

        validate_file(file, validator_settings, validator_errors, file_validator)

        if validator_errors.has_errors():
            formatted_errors = validator_errors.format_errors()
            return (False, f"Errors for {file_type} valdiation:\n{formatted_errors}")

        return (True, None)

    def validate_files(self, folder: str, file_type: str) -> tuple[bool, str | None]:
        """
        Validate all files in the specified folder of the specified type.

        folder (str): Path containing files to validate.
        file_type (str): The type of file to validate.
        :return (tuple): True or False, indicating whether the validation was successful or not,
            followed by a string containing all detected validation errors (or `None` if there
            were no errors.

        The input path `file_path` is searched recursively for files with extension matching
        the `file_type`. Supported `file_type` values are "fastq", "bam", and "sequencing_summary".
        Note that this means that if you specify `file_type="sequencing_summary"`, it will
        check all "*.txt" files as though they were `sequencing_summary` files.
        """
        ext_map = {"fastq": ".fastq", "bam": ".bam", "sequencing_summary": ".txt"}
        if file_type not in ext_map:
            return (
                False,
                f"Validation error: {file_type} is not a supported file type for validation.",
            )
        extension = ext_map[file_type]
        file_validator = validator_from_name(file_type, self.specification_bundle)
        validator_errors = ErrorContainer()
        file_path = pathlib.Path(folder)
        files = list(file_path.glob(f"**/*{extension}"))
        validator_settings = ExperimentSettings(**self.settings)

        for query in files:
            validate_file(query, validator_settings, validator_errors, file_validator)

        if validator_errors.has_errors():
            formatted_errors = validator_errors.format_errors()
            return (False, f"Errors for {file_type} valdiation:\n{formatted_errors}")

        return (True, None)


def setup_validator(
    settings: dict | None,
) -> tuple[OutputValidator | None, ValidationOptions | None]:
    """
    If `settings` is not `None`, then it must be a `dict` which overrides some, or all,
    of the settings defined in `DEFAULT_SETTINGS` above.
    """
    if settings is None:
        return None, None
    options = ValidationOptions()
    if options.spec_file is None:
        return None, None

    validator = OutputValidator(options.spec_file)
    if options.is_gpu_build:
        validator.settings.update({"gpu_calling_enabled": True})
    if settings is not None:
        validator.settings.update(settings)
    return validator, options


def validate_output_folder(folder, settings: dict | None):
    """
    Validate all files in the specified folder. The folder is searched recursively.
    If `settings` is not `None`, then it must be a `dict` which overrides some, or all,
    of the settings defined in `DEFAULT_SETTINGS` above.
    """
    validator, options = setup_validator(settings)
    if validator is None:
        return

    fastq_ok, bam_ok, summary_ok = (True, True, True)
    all_errors = []
    if options.do_fastq:
        fastq_ok, errors = validator.validate_files(folder, "fastq")
        if errors is not None:
            all_errors.append(errors)
    if options.do_bam:
        bam_ok, errors = validator.validate_files(folder, "bam")
        if errors is not None:
            all_errors.append(errors)
    if options.do_summary:
        summary_ok, errors = validator.validate_files(folder, "sequencing_summary")
        if errors is not None:
            all_errors.append(errors)
    if not fastq_ok or not bam_ok or not summary_ok:
        error_string = "\n".join(all_errors)
        raise AssertionError(error_string)


def validator_from_name(name: str, spec: SpecificationBundle) -> Validator | None:
    """
    Create a validator from a name, eg. "bam".

    :param name: The name of the validator to create. Valid options are "bam".
    :param spec: The specification to use for validation.
    :return: A validator instance.
    """

    if name == "bam":
        return BamValidator(spec.get_file_content("bam/spec.yaml"))
    elif name == "fastq":
        return FastqValidator(
            spec.get_file_content("fastq/header-spec-hts.yaml"), legacy_header=False
        )
    elif name == "sequencing_summary":
        return SequencingSummaryValidator(
            spec.get_file_content("sequencing_summary/spec.yaml")
        )
    return None
