import argparse
import csv
import io
import os
import pathlib
import re
import dateutil.parser as dateutilparser
from datetime import datetime, timezone

from tetra.database import DBWrapper
from tetra.utilities import get_env_or_raise


# Split <preset>_<config> into parts.
def extract_preset_config(name: str, for_latency: bool):
    # Presets taken from their relevant scripts.
    valid_latency_presets = [
        "custom",
        "gpu_mode",
        "gridion",
        "macos",
        "orin",
        "promethion",
    ]
    valid_speed_presets = [
        "custom",
        "gridion",
        "macos",
        "orin",
        "p24",
        "p48",
    ]
    valid_presets = valid_latency_presets if for_latency else valid_speed_presets
    for preset in valid_presets:
        if name.startswith(preset):
            config = name[len(preset) + 1 :]
            return preset, config
    raise RuntimeError(f"Unknown file format: {name}")


def upload_job_results(
    db_wrapper: DBWrapper,
    test_dir: pathlib.Path,
    platform: str,
    runner: str,
    job_id: str,
    commit: str,
    date: datetime,
):
    # Ignore this if it's just the logs.
    if test_dir.name.endswith("_logs"):
        return
    print(f"Processing {test_dir.resolve()}")

    # The name tells us if this is a latency or speed measurement.
    test_name = test_dir.name
    if "latency" in test_name:
        # There should now only be 1 file, for adaptive sampling latencies.
        files = [file for file in test_dir.iterdir() if file.suffix == ".txt"]
        if len(files) != 1:
            raise RuntimeError(f"Unexpected number of files: {len(files)}: {files}")

        # The name should be {prefix}_as.
        original_name = files[0].stem
        underscore_pos = original_name.rfind("_")
        if underscore_pos == -1 or original_name[underscore_pos:] != "_as":
            raise RuntimeError(f"Expected file '{original_name}' to end with '_as'.")
        prefix = original_name[:underscore_pos]

        # Extract the preset and config from the prefix.
        preset, config = extract_preset_config(prefix, for_latency=True)

        # Contents should be a single row CSV for gpu_mode, 5 rows for gridion, and
        # 24 rows for promethion. For now we just look at the first row.
        reader = csv.DictReader(io.StringIO(files[0].read_text()), delimiter="\t")
        rows = [row for row in reader]
        assert len(rows) in [1, 5, 24]
        row_as = rows[0]

        # Add the measurement.
        db_wrapper.add_latency_entry(
            name=test_name,
            config=config,
            preset=preset,
            platform=platform,
            runner=runner,
            job_id=job_id,
            commit=commit,
            date=date,
            as_latency_quartile1=float(row_as["quartile1"]),
            as_latency_median=float(row_as["median"]),
            as_latency_quartile3=float(row_as["quartile3"]),
            live_latency_speed=0.0,
        )

    else:
        for file in test_dir.iterdir():
            # Only upload the .txt files.
            if file.suffix != ".txt":
                continue
            print(f"Uploading entry for {file.resolve()}")

            # Extract the preset and config for this file.
            preset, config = extract_preset_config(file.stem, for_latency=False)

            # Contents should be a single entry CSV, but no need to enforce that.
            reader = csv.DictReader(io.StringIO(file.read_text()), delimiter="\t")
            for row in reader:
                db_wrapper.add_speed_entry(
                    name=test_name,
                    config=config,
                    preset=preset,
                    platform=platform,
                    runner=runner,
                    job_id=job_id,
                    commit=commit,
                    date=date,
                    speed=int(row["speed"]),
                )


def upload_output(project: str, output_dir: pathlib.Path):
    print("Starting upload")
    db_wrapper = DBWrapper(project)

    # This is only expected to be run from CI.
    job_id = get_env_or_raise("CI_JOB_ID")
    date = dateutilparser.parse(get_env_or_raise("CI_JOB_STARTED_AT"))
    runner = get_env_or_raise("CI_RUNNER_DESCRIPTION")
    commit = get_env_or_raise("CI_COMMIT_SHA")

    # We can infer the platform from the output dir.
    platforms = [f for f in output_dir.iterdir()]
    if len(platforms) != 1:
        raise Exception(f"Unexpected number of platforms: {platforms}")
    platform = platforms[0].name

    # Find all the relevant tests to upload.
    for test_dir in (output_dir / platform).iterdir():
        upload_job_results(
            db_wrapper=db_wrapper,
            test_dir=test_dir,
            platform=platform,
            runner=runner,
            job_id=job_id,
            commit=commit,
            date=date,
        )


def upload_batch(project_url: str, skip_ssl: bool, previous_days: int):
    import tempfile
    import zipfile

    import gitlab  # pip install python-gitlab
    from gitlab.v4.objects import ProjectJob, ProjectPipelineJob

    delims = [m.start() for m in re.finditer("/", project_url)]
    # We expect the URL to contain at least 4 "/" characters.
    assert len(delims) > 3
    split_point = delims[2] + 1
    gitlab_url = project_url[:split_point]
    project_name = project_url[split_point:]

    def setup_gitlab() -> gitlab.Gitlab:
        # Read in the token.
        private_token = os.getenv("GITLAB_TOKEN")
        if private_token is None:
            raise RuntimeError(
                "GITLAB_TOKEN environment variable not set.\n"
                "Visit https://docs.gitlab.com/ee/user/profile/personal_access_tokens.html to create one.\n"
                "Note that only the `read_api` scope is required."
            )

        # Connect to gitlab.
        gitlab_connection = gitlab.Gitlab(
            url=gitlab_url,
            ssl_verify=not skip_ssl,
            private_token=private_token,
        )
        gitlab_connection.auth()
        return gitlab_connection

    print("Starting batch transfer")

    # Connect to gitlab and iterate through all the scheduled pipeline's jobs.
    # This is significantly faster than filtering all the jobs.
    gitlab_connection = setup_gitlab()
    current_project = gitlab_connection.projects.get(project_name)

    # Make a connection to the DB.
    project = "core_cpp" if project_name == "ont_core_cpp" else project_name
    db_wrapper = DBWrapper(project)
    now = datetime.now(timezone.utc)

    def upload_job_artifacts(job: ProjectPipelineJob) -> bool:
        name: str = job.name
        if ":speed:" not in name or job.status != "success":
            return True

        # Fetch the job.
        job: ProjectJob = current_project.jobs.get(job.id)

        # Break out once the jobs are old enough.
        if (now - dateutilparser.parse(job.created_at)).days > previous_days:
            return False

        # Extract common metadata.
        date = dateutilparser.parse(job.created_at)
        runner = job.attributes["runner"]["description"]
        commit = job.attributes["commit"]["id"]
        job_id: int = job.id
        print(f"Uploading job {job_id}")

        # Check if this job has already been processed.
        if db_wrapper.has_job_entry(job_id=str(job_id)):
            print(f"Job {job_id} already uploaded - skipping")
            return True

        # Grab artifacts.
        try:
            artifacts = job.artifacts()
        except gitlab.exceptions.GitlabGetError:
            print(f"No artifacts for job {job_id}")
            return True
        zip = zipfile.ZipFile(io.BytesIO(artifacts))

        # Unzip the artifacts.
        with tempfile.TemporaryDirectory() as temp_dir:
            zip.extractall(path=temp_dir)

            # Descend into it.
            output_dir = pathlib.Path(temp_dir) / "regression_test" / "output"
            for platform_dir in output_dir.iterdir():
                platform = platform_dir.stem
                for test_dir in platform_dir.iterdir():
                    # Upload it to the DB.
                    upload_job_results(
                        db_wrapper=db_wrapper,
                        test_dir=test_dir,
                        platform=platform,
                        runner=runner,
                        job_id=job_id,
                        commit=commit,
                        date=date,
                    )

        return True

    finished = False
    for pipeline in current_project.pipelines.list(iterator=True, source="schedule"):
        for job in pipeline.jobs.list(iterator=True, status="success"):
            finished = not upload_job_artifacts(job)
            if finished:
                break
        if finished:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload performance regression test results to database."
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="The directory to upload."
    )
    parser.add_argument(
        "--dorado", action="store_true", help="Upload for dorado project."
    )
    parser.add_argument(
        "--core_cpp", action="store_true", help="Upload for core_cpp project."
    )

    # Batch transfer from CI to the DB.
    batch_group = parser.add_argument_group("Batch upload from CI to DB.")
    batch_group.add_argument(
        "--batch",
        action="store_true",
        help="Batch transfer from old CI jobs.",
    )
    batch_group.add_argument(
        "--prev_days",
        type=int,
        help="How many days worth of jobs to upload.",
        default=10,
    )
    batch_group.add_argument(
        "--project_url",
        type=str,
        help="Gitlab project URL to use.",
    )
    batch_group.add_argument(
        "--skip_ssl", action="store_true", help="Workaround for running locally."
    )

    args = parser.parse_args()

    if args.dorado and args.core_cpp:
        raise Exception("You can only specify one of `--dorado` and `--core_cpp`.")

    project = None
    if args.dorado:
        project = "dorado"
    if args.core_cpp:
        project = "ont_core"

    if not args.batch and project is None:
        raise Exception("You must specify either `--dorado` or `--core_cpp`.")

    # Silence warning spam.
    if args.skip_ssl:
        import urllib3

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    if not args.batch and args.output_dir is None:
        raise Exception("You must either specify `--output_dir` or `--batch`.")

    if args.batch:
        upload_batch(args.project_url, args.skip_ssl, args.prev_days)
    else:
        upload_output(project, pathlib.Path(args.output_dir))
