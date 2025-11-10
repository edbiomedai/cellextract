#!/usr/bin/env python
"""
For each plate in input_dir, create a job script that will rsync the plate to
the stage_dir. The job script will be saved to job_dir. The stage_dir, log_dir,
and job_dir will be created if they don't exist.

Usage:
    stage_data.py <input_dir> <job_dir> <stage_dir>

Arguments:
    <input_dir>   Input directory
    <job_dir>     Directory to save job scripts
    <stage_dir>   Directory to stage data to

Options:
    -h --help     Show this screen
"""
import argparse
import os
import subprocess

JOBSTUB = """rsync -rt -s --perms --chmod=u+rwx --ignore-existing --exclude "*thumb*" "{indir}" "{stagedir}"""


def stage_data(input_dir, stage_dir):
    "Stage data from input_dir to stage_dir"
    if not os.path.exists(stage_dir):
        os.makedirs(stage_dir)
    script = f'rsync -rt -s --perms --chmod=u+rwx --ignore-existing --exclude "*thumb*" "{input_dir}" "{stage_dir}'
    subprocess.run(script, shell=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="Input directory")
    parser.add_argument("stage_dir", type=str, help="Staging directory")
    args = parser.parse_args()
    stage_data(args.input_dir, args.stage_dir)


if __name__ == "__main__":
    main()
