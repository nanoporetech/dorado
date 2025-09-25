import pathlib


def reformat_txt(file: pathlib.Path) -> None:
    # Read it all
    rows = []
    with open(file) as input:
        while line := input.readline():
            rows.append(line)
    # Sort it, ignoring the header line
    rows[1:] = sorted(rows[1:])
    # Save it
    with open(file, "w+") as output:
        output.writelines(rows)


def reformat_fastq(file: pathlib.Path) -> None:
    # Group lines into 4 fields
    groups = []
    counter = 0
    with open(file) as input:
        fields = []
        while line := input.readline():
            fields.append(line)
            counter += 1
            if counter == 4:
                counter = 0
                groups.append(fields)
                fields = []
    # There shouldn't be any stray lines
    if counter > 0:
        raise Exception(f"Stray lines found in '{file}'.")
    # Sort by first field
    groups.sort(key=lambda fields: fields[0])
    # Save it
    with open(file, "w+") as output:
        for fields in groups:
            output.writelines(fields)


def reformat_sam(file: pathlib.Path) -> None:
    # Read off header and alignment sections
    headers = []
    alignments = []
    current = headers
    with open(file) as input:
        while line := input.readline():
            if not line.startswith("@"):
                current = alignments
            current.append(line)
    # Sort the alignments
    alignments.sort()
    # Save it
    with open(file, "w+") as output:
        output.writelines(headers)
        output.writelines(alignments)


def reformat_files(output_folder: pathlib.Path) -> None:
    for file in output_folder.glob("**/*.txt"):
        reformat_txt(file)
    for file in output_folder.glob("**/*.fastq"):
        reformat_fastq(file)
    for file in output_folder.glob("**/*.sam"):
        reformat_sam(file)
