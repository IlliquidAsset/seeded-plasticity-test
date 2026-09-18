#!/usr/bin/env python3
"""
Launch the pure SNN experiment on Lightning.ai free tier.
Uses the Lightning SDK.
"""
import sys
import os
import time
import json

# Use the Hermes venv which has lightning_sdk installed
LIGHTNING_PYTHON = os.path.expanduser("~/.hermes/hermes-agent/venv/bin/python3")


def main():
    # We'll run via lightning job run CLI since it's simpler
    # The job clones the GitHub repo and runs the experiment
    cmd = (
        f'{LIGHTNING_PYTHON} -m lightning job run '
        f'--name "pure-snn-learning-{int(time.time())}" '
        f'--machine CPU '
        f'--teamspace "illiquidasset-org/big-girl-production" '
        f'--command "pip install torch && python lightning_run.py" '
        f'--image "python:3.11" '
        f'--json'
    )
    print(f"Running: {cmd}")
    os.system(cmd)


if __name__ == "__main__":
    main()
