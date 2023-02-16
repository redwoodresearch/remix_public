"""
Replace smart quotes, long dashes, and other special characters that we don't want in the source.
"""
# TBD lowpri: make this a pre-commit hook on the solution files (before generating instructions)

import re
import argparse
import sys

special_re = re.compile(r"([^\x00-\x7f])")

REPLACEMENTS = {
    "—": "-",
    "’": "'",
    "“": '"',
    "”": '"',
    "…": "...",
}


def fix(text: str) -> tuple[str, set[str]]:
    unrecognized: set[str] = set()
    n_fixed = 0

    def replace_fn(m: re.Match):
        global n_fixed
        c = m.group(0)
        fixed = REPLACEMENTS.get(c)
        if fixed is None:
            unrecognized.add(c)
            return c
        n_fixed += 1
        return fixed

    fixed_text = special_re.sub(replace_fn, text)
    if unrecognized:
        print("Failed to fix: ", unrecognized)
    if n_fixed > 0:
        print(f"{n_fixed} fixes made.")
    return fixed_text, unrecognized


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_filename")
    args = parser.parse_args()

    with open(args.input_filename, "r", encoding="utf-8") as f:
        text = f.read()

    fixed_text, unrecognized = fix(text)

    with open(args.input_filename, "w", encoding="utf-8") as f:
        f.write(fixed_text)

    sys.exit(1 if unrecognized else 0)
