import os
import pathlib
import platform
import subprocess
from datetime import datetime

ROOT_DIR = pathlib.Path(__file__).parent.parent


def get_env_or_raise(name: str) -> str:
    val = os.getenv(name)
    if val is None:
        raise Exception(f"Couldn't find environment variable {name}")
    return val


def get_platform() -> str:
    platform_id = platform.system().lower()
    if platform_id == "linux":
        import distro

        if distro.name().lower() == "centos":
            return "manylinux"
        elif (
            platform.uname().machine == "aarch64"
            and "tegra" in platform.uname().release
            and distro.version() == "22.04"
        ):
            return "orin"
        return "linux"
    elif platform_id == "darwin":
        processor = platform.processor()
        if processor == "arm":
            return "osx_arm"
        else:
            raise RuntimeError(f"Unknown macOS processor: {processor}")
    elif platform_id == "windows":
        return "windows"
    else:
        raise RuntimeError(f"Unknown platform: {platform_id}")


def run_with_timeout(cmd_args: list[str], timeout: float) -> int:
    try:
        print("Command line: ", " ".join(cmd_args))
        start_time = datetime.now()
        subprocess.check_call(cmd_args, timeout=timeout, text=True)
        return (datetime.now() - start_time).seconds
    except subprocess.CalledProcessError as e:
        print(
            f"Error running {cmd_args}: returncode={e.returncode}\nstdout:\n{e.stdout}\nstderr:\n{e.stderr}"
        )
        raise
