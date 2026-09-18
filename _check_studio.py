#!/usr/bin/env python3
"""Check Lightning studio and launch experiment."""
import sys
import os
import time
import json
import subprocess

LIGHTNING_PYTHON = os.path.expanduser("~/.hermes/hermes-agent/venv/bin/python3")

code = """
from lightning_sdk import Studio, Machine
import json

studio = Studio(name="big-girl-motion")
print(f"Studio: {studio.name}, status: {studio.status}")

# Try to use the studio to run a job
# We can use job.run() from a studio context
print(f"Studio teamspace: {studio.teamspace}")
print(f"Studio machine: {studio.machine}")
print("Available machines:")
for m in Machine:
    print(f"  {m.value}")
"""

result = subprocess.run(
    [LIGHTNING_PYTHON, "-c", code],
    capture_output=True, text=True, timeout=30
)
print(result.stdout)
if result.stderr:
    # Filter out warnings
    for line in result.stderr.split('\n'):
        if 'UserWarning' not in line and 'warnings.warn' not in line:
            print(f"STDERR: {line}")
