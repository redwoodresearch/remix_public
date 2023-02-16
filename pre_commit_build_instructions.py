import subprocess

from build_all import build_all

if __name__ == "__main__":
    built_files = build_all()
    for filename in built_files:
        subprocess.check_call(("git", "add", filename))
