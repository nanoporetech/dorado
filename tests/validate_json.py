import json
import sys


def validate_json_stream(stream):
    try:
        json.load(stream)
        return True
    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e.msg}", file=sys.stderr)
        print(f"Line {e.lineno}, column {e.colno}", file=sys.stderr)
        return False


def main() -> int:
    if len(sys.argv) != 2:
        print(
            'Usage: "python validate_json.py <file.json>" OR "python validate_json.py -" to read from stdin ',
            file=sys.stderr,
        )
        return 2

    input = sys.argv[1]

    if input == "-":
        valid_json = validate_json_stream(sys.stdin)
        return 0 if valid_json else 1

    try:
        with open(input, "r", encoding="utf-8") as f:
            valid_json = validate_json_stream(f)
            if not valid_json:
                print(f"In file {input}", file=sys.stderr)
        return 0 if valid_json else 1
    except FileNotFoundError:
        print(f"File not found: {input}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
