import argparse
import json
import matplotlib.dates as mpd
import matplotlib.pyplot as plt
import os
import pathlib
import typing
from datetime import datetime, timedelta

from tetra.database import DBWrapper
from tetra.utilities import get_env_or_raise


def download_results(project: str, graph_history: int):
    # Map from type -> all measurements.
    results: typing.Dict[str, typing.List[typing.Dict]] = {}
    print("Starting download")

    # Grab a range of data.
    oldest = datetime.now() - timedelta(days=graph_history)
    db_wrapper = DBWrapper(project)

    results["speed"] = []
    for entry in db_wrapper.get_speed_entries(oldest):
        results["speed"].append(entry)
    results["speed"].sort(key=lambda x: x["date"])

    results["latency"] = []
    for entry in db_wrapper.get_latency_entries(oldest):
        results["latency"].append(entry)
    results["latency"].sort(key=lambda x: x["date"])

    return results


def is_outlier(test_type: str, entry: typing.Dict) -> typing.Optional[str]:
    if test_type == "latency":
        latency = entry["as_latency_median"]
        if latency > 5:
            job_id = entry["job_id"]
            current_job_url = os.get_env("CI_JOB_URL")
            if current_job_url is None:
                return "Running locally, so not checking for outliers."
            last_slash = current_job_url.rfind("/")
            job_url = f"{current_job_url[:last_slash]}/{job_id}"
            return f"Outlier with latency {latency}s in job {job_url}"
    return None


# Returns any outliers found while generating the graphs.
def generate_graphs(
    project: str,
    graph_dir: pathlib.Path,
    individual_graphs: bool,
    graph_history: int,
) -> typing.List[str]:
    # Create the folder we'll dump to.
    graph_dir.mkdir(exist_ok=True)

    results = download_results(project, graph_history)
    print("Generating graphs")

    # Process the data.
    outlier_msgs: typing.List[str] = []
    all_data_sets = {}
    grouped_data_sets: typing.Dict[str, typing.Set[typing.Tuple[str, str]]] = {}
    for test_type, entries in results.items():
        is_latency = test_type == "latency"
        for entry in entries:
            # Don't display outliers in graphs, but make sure to report them.
            outlier_msg = is_outlier(test_type, entry)
            if outlier_msg is not None:
                outlier_msgs.append(outlier_msg)
                continue

            test_name = entry["name"]
            config = entry["config"]
            preset = entry["preset"]
            date = entry["date"]
            platform = entry["platform"]
            score = entry["as_latency_median" if is_latency else "speed"]

            # Add it.
            full_name = f"{test_type}-{platform}-{test_name}-{config}-{preset}"
            if full_name not in all_data_sets:
                all_data_sets[full_name] = {
                    "x": [],
                    "y": [],
                    "runner": [],
                    "job_id": [],
                }
            datasets = all_data_sets[full_name]
            datasets["x"].append(date)
            datasets["y"].append(score)
            datasets["runner"].append(entry["runner"])
            datasets["job_id"].append(entry["job_id"])

            display_name = f"{platform}/{preset}/{config}/{test_name}"

            def add_to_group(group_name: str):
                if group_name not in grouped_data_sets:
                    grouped_data_sets[group_name] = set()
                grouped_data_sets[group_name].add((full_name, display_name))

            # Add this result to the relevant groups.
            if is_latency:
                # All latency measurements go in 1 graph.
                add_to_group("all_latency")
            else:
                # Split the others up based on their model type, preset, and device.
                if "A100" in test_name:
                    device_suffix = "_a100"
                elif "V100" in test_name:
                    device_suffix = "_v100"
                elif "A6000" in test_name:
                    device_suffix = "_a6000"
                else:
                    device_suffix = ""
                if "fast@" in config:
                    add_to_group(f"{preset}_fast{device_suffix}")
                if "hac@" in config:
                    add_to_group(f"{preset}_hac{device_suffix}")
                if "sup@" in config:
                    add_to_group(f"{preset}_sup{device_suffix}")
            # TODO: other groups?

    # Generate graphs for each individually.
    if individual_graphs:
        for name, data in all_data_sets.items():
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(data["x"], data["y"], marker="x")
            ax.set_title("\n".join(name.split("-")), fontsize=8)
            ax.set_xlabel("Date")
            ax.set_ylabel("Metric")
            ax.set_ylim(bottom=0)
            fig.gca().xaxis.set_major_formatter(mpd.DateFormatter("%d-%m-%Y"))
            fig.gca().xaxis.set_major_locator(mpd.DayLocator(interval=7))
            fig.autofmt_xdate()
            fig.savefig(graph_dir / (name + ".png"))
            plt.close(fig)

    # Generate graphs for the groups.
    for group_name, names in grouped_data_sets.items():
        # Make the names list order deterministic.
        names = list(names)
        names.sort()
        # Create the figure we'll draw on.
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.set_title(group_name)
        # Each series gets its own properties to make them distinct.
        ax.set_prop_cycle(
            plt.cycler(linestyle=["-", ":", "-."]) * plt.cycler(color="bgrmyk")
        )
        for full_name, display_name in names:
            data = all_data_sets[full_name]
            ax.plot(
                data["x"],
                data["y"],
                label=display_name,
                linewidth=1,
                marker="x",
                markersize=3,
            )
        ax.set_xlabel("Date")
        ax.set_ylabel("Metric")
        ax.set_ylim(bottom=0)
        fig.legend(loc="center right", fontsize=6)
        fig.tight_layout()
        fig.subplots_adjust(right=0.6)
        fig.gca().xaxis.set_major_formatter(mpd.DateFormatter("%d-%m-%Y"))
        fig.gca().xaxis.set_major_locator(mpd.DayLocator(interval=7))
        fig.autofmt_xdate()
        fig.savefig(graph_dir / (group_name + ".png"))
        plt.close(fig)

    # Serialise the data to JSON so that we can render it in other ways.
    # default=str handles datetime not being serialisable.
    series_json = json.dumps(all_data_sets, default=str)
    (graph_dir / "series.json").write_text(series_json)

    return outlier_msgs


def send_email(graph_dir: pathlib.Path, outlier_msgs: typing.List[str]):
    import smtplib
    from email.generator import Generator
    from email.message import EmailMessage
    from email.utils import make_msgid

    # Fill out basic message details.
    EMAIL_LIST = get_env_or_raise("PERF_GRAPH_EMAIL_LIST").split(",")
    EMAIL_SENDER = get_env_or_raise("OFAN_TESTER_EMAIL")
    EMAIL_PASSWORD = get_env_or_raise("OFAN_TESTER_PW")

    msg = EmailMessage()
    msg["To"] = ", ".join([name + "@nanoporetech.com" for name in EMAIL_LIST])
    msg["From"] = EMAIL_SENDER
    msg["Subject"] = "Performance benchmark graphs"
    content = "<html><body><h1>Graphs!</h1><br/>Also attached is a JSON of the raw data for all plots.<br/>"

    # Report any outliers first.
    if len(outlier_msgs) > 0:
        content += "<br/><h3>Outliers (not graphed):</h3>"
    for outlier_msg in outlier_msgs:
        content += f"<li><b>{outlier_msg}</b></li>"

    # Add each image to the message.
    images: typing.List[typing.Tuple[bytes, str]] = []
    for image in sorted(graph_dir.iterdir()):
        if not image.name.endswith(".png"):
            continue

        # Make an ID for the attachment so that we can embed it.
        cid = make_msgid(domain="perf")
        content += f'<br/><h3>{image.name}</h3><br/><img src="cid:{cid[1:-1]}"/>'

        # Load the image that we'll add to the email.
        images.append((image.read_bytes(), cid))

    # The content of the message must come before the images.
    content += "</body></html>"
    msg.set_content(content, "html")
    for image, cid in images:
        msg.add_related(image, maintype="image", subtype="png", cid=cid)

    # Add the json too so that users can optionally process it.
    msg.add_attachment(
        (graph_dir / "series.json").read_bytes(),
        maintype="application",
        subtype="json",
        filename="series.json",
    )

    # Save it so you can preview it locally.
    with (graph_dir / "graphs.eml").open("w") as f:
        Generator(f).flatten(msg)

    # Send the email.
    with smtplib.SMTP("smtp.office365.com", port=587) as smtp:
        smtp.starttls()
        smtp.login(
            EMAIL_SENDER,
            EMAIL_PASSWORD,
        )
        smtp.send_message(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a graph of regression test results."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="The directory to save the graph."
    )
    parser.add_argument(
        "--dorado", action="store_true", help="Upload for dorado project."
    )
    parser.add_argument(
        "--core_cpp", action="store_true", help="Upload for core_cpp project."
    )
    parser.add_argument(
        "--send_email", action="store_true", help="Send out an email of the graphs."
    )
    parser.add_argument(
        "--individual_graphs",
        action="store_true",
        help="Generate individual graphs in addition to the groups.",
    )
    parser.add_argument(
        "--graph_history",
        type=int,
        help="How many days worth of data to graph.",
        default=90,
    )

    args = parser.parse_args()

    if args.dorado and args.core_cpp:
        raise Exception("You can only specify one of `--dorado` and `--core_cpp`.")

    project = None
    if args.dorado:
        project = "dorado"
    if args.core_cpp:
        project = "ont_core"

    if project is None:
        raise Exception("You must specify either `--dorado` or `--core_cpp`.")

    outlier_msgs = generate_graphs(
        project,
        pathlib.Path(args.output_dir),
        args.individual_graphs,
        args.graph_history,
    )
    for outlier_msg in outlier_msgs:
        print(outlier_msg)

    if args.send_email:
        send_email(pathlib.Path(args.output_dir), outlier_msgs)
