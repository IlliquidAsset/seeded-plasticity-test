#!/usr/bin/env python3
"""
Launch the pure SNN experiment on Lightning.ai via the Python SDK.
"""
import sys
import os
import time
import json
import subprocess

LIGHTNING_PYTHON = os.path.expanduser("~/.hermes/hermes-agent/venv/bin/python3")

script = """
import sys
import time
import json

from lightning_sdk import Studio, Machine

# Use the existing studio
studio = Studio(name="big-girl-motion")
print(f"Studio: {studio.name}, status: {studio.status}")

# Start if stopped
if studio.status == "stopped" or studio.status == "Stopped":
    print("Starting studio...")
    studio.start(machine=Machine.CPU)  # Free tier CPU
    print("Studio started")

# Wait for it to be ready
for i in range(30):
    studio.refresh()
    if studio.status == "running" or studio.status == "Running":
        print(f"Studio is running after {i*10}s")
        break
    time.sleep(10)

# Now run the experiment
# Use studio.run() to execute a command
print("\\nCloning repo and running experiment...")
result = studio.run(
    "git clone https://github.com/IlliquidAsset/seeded-plasticity-test.git /tmp/snn && "
    "cd /tmp/snn && "
    "python lightning_run.py"
)
print(f"Result: {result}")

# Get the results file
if studio.file_exists("/tmp/snn/results.json"):
    results = studio.read("/tmp/snn/results.json")
    print(f"\\nResults: {results}")
else:
    print("\\nNo results file found")
    # List files
    print("Files in /tmp/snn:")
    print(studio.run("ls -la /tmp/snn/"))
"""

result = subprocess.run(
    [LIGHTNING_PYTHON, "-c", script],
    capture_output=True, text=True, timeout=600
)
print(result.stdout)
if result.stderr:
    for line in result.stderr.split('\\n'):
        if 'UserWarning' not in line and 'warnings.warn' not in line:
            print(f"STDERR: {line}")
