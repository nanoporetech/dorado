import os


# These options can be overridden by setting environment variables with
# the same names before importing tetra.
class ValidationOptions(object):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ValidationOptions, cls).__new__(cls)
            cls._instance.do_fastq = cls.check_variable("VALIDATE_FASTQ_FILES", True)
            cls._instance.do_bam = cls.check_variable("VALIDATE_BAM_FILES", True)
            cls._instance.do_summary = cls.check_variable(
                "VALIDATE_SUMMARY_FILES", True
            )
            cls._instance.is_gpu_build = cls.check_variable("BUILD_CUDA", False)
            cls._instance.spec_file = os.environ.get("SPECIFICATION_FILE")
            if cls._instance.spec_file is None:
                print(
                    "Warning: The SPECIFICATION_FILE environment variable has not been set. "
                    "Skipping output-file specification validation."
                )
        return cls._instance

    @classmethod
    def check_variable(cls, name: str, default: bool) -> bool:
        return default or name in os.environ
