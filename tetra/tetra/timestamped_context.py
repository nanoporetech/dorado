import os
import subprocess
import sys
from datetime import datetime

psutil_available = True
try:
    import psutil
except ImportError:
    print("CPU monitoring unavailable")
    psutil_available = False

VERBOSE = False


def get_process_priority(pid: int) -> str:
    if psutil_available:
        return str(psutil.Process(pid).nice())
    else:
        return "unknown"


class TimestampedContext(object):
    def __init__(self, context_name: str):
        self.timestamped_context_name = context_name

    def __enter__(self):
        self._print_section_start()
        self.init_cpu_percent_start_time_per_process()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.report_cpu_usage()
        if exc_type == subprocess.TimeoutExpired:
            self.report_running_processes()
        self._print_section_end()

    def report_cpu_usage(self):
        if psutil_available:
            self.timestamp_print(
                f"Utilization per CPU: {psutil.cpu_percent(interval=None, percpu=True)}"
            )
            self.timestamp_print(f"Load avg 1, 5, 15mins: {psutil.getloadavg()}")
            self.timestamp_print(psutil.virtual_memory())

    def init_cpu_percent_start_time_per_process(self):
        if psutil_available:
            for proc in psutil.process_iter(["pid", "name", "username"]):
                try:
                    proc.cpu_percent(
                        interval=None
                    )  # subsequent call to cpu_percent will use this as the start time
                except Exception as ex:
                    # Simply Ignore this process as we are just gathering extra
                    # diagnostic info in case of a regression test failure, we don't
                    # want to flood the output with error messages. E.g. We do expect
                    # AccessDenied on OSX for processes of other users.
                    self._print_verbose_msg(
                        f"Warning: Exception thrown by psutil.cpu_percent\n{ex}"
                    )

    def _print_section_start(self):
        # If we're in CI then put this test in a section by itself.
        if "CI_JOB_ID" in os.environ:
            timestamp = int(datetime.now().timestamp())
            section_id = abs(hash(self.timestamped_context_name))
            header = self.timestamped_context_name
            self._print_msg(
                f"\033[0Ksection_start:{timestamp}:section_{section_id}[collapsed=true]\r\033[0K{header}"
            )
        self.timestamp_print("Entering: " + self.timestamped_context_name)

    def _print_section_end(self):
        self.timestamp_print("Exiting: " + self.timestamped_context_name)
        # If we're in CI then put this test in a section by itself.
        if "CI_JOB_ID" in os.environ:
            timestamp = int(datetime.now().timestamp())
            section_id = abs(hash(self.timestamped_context_name))
            self._print_msg(
                f"\033[0Ksection_end:{timestamp}:section_{section_id}\r\033[0K"
            )

    def timestamp_print(self, msg: str):
        self._print_msg(f"[{datetime.now()}] {msg}")

    def _print_msg(self, msg):
        # gitlab doesn't synchronise the streams so flush both before writing our message so
        # that it appears after everything that's already been written on either of them.
        self._flush_streams()
        print(msg)
        # Flush again so that any messages on either stream appear after this one.
        self._flush_streams()

    def _print_verbose_msg(self, msg: str):
        if VERBOSE:
            self._print_msg(msg)

    def _flush_streams(self):
        for file in [sys.stdout, sys.stderr]:
            file.flush()
