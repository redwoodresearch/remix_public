import argparse
import asyncio
import glob
import logging
import os
import stat

from build_instructions import build, guess_outname

logging.basicConfig(level=logging.INFO)


def needs_build(inpath, outpath):
    if not os.path.exists(outpath):
        return True
    mod_in = os.stat(inpath).st_mtime
    mod_out = os.stat(outpath).st_mtime
    return mod_in > mod_out


def build_instructions(inpath, outpath):
    if os.path.exists(outpath):
        os.chmod(outpath, stat.S_IWRITE | stat.S_IWGRP | stat.S_IWOTH)

    with open(inpath, "r") as infile:
        with open(outpath, "w") as outfile:
            try:
                build(infile, outfile)
            except SyntaxError as e:
                print("Syntax error - will not rebuild until next save.")
                print(e)
            except UnicodeDecodeError as e:
                print("Unicode error - will not rebuild until next save.")
                print(e)
    os.chmod(outpath, 0o777 ^ (stat.S_IWRITE | stat.S_IWGRP | stat.S_IWOTH))


def get_in_out_paths():
    fnames = glob.glob("*_solution.py")
    outpaths = [guess_outname(fname) for fname in fnames]
    return list(zip(fnames, outpaths))


def build_all(force=False) -> list:
    built_files = []
    for inpath, outpath in get_in_out_paths():
        if needs_build(inpath, outpath) or force:
            built_files.append(outpath)
            build_instructions(inpath, outpath)
    return built_files


async def watch_for_changes(interval_secs=2):
    """Watch and rebuild on input changes."""
    while True:
        build_all()
        await asyncio.sleep(interval_secs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", "-o", action="store_true")
    parser.add_argument("--force", "-f", action="store_true")
    args = parser.parse_args()
    if args.once:
        build_all(force=args.force)
    else:
        asyncio.run(watch_for_changes())
