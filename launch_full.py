#!/usr/bin/env python3
"""Launch experiment on Lightning via SDK - start studio and run."""
import sys
import os
import time
import json
import subprocess

LIGHTNING_PYTHON = os.path.expanduser("~/.hermes/hermes-agent/venv/bin/python3")

launch_code = """
import sys
import time
import os
import json

from lightning_sdk import Studio, Machine

studio = Studio(name="big-girl-motion")
print(f"Status: {studio.status}")

# Start the studio (free tier CPU)
if studio.status.lower() == "stopped":
    print("Starting studio on CPU (free tier)...")
    try:
        studio.start(machine="CPU")
        print("Start initiated")
    except Exception as e:
        print(f"Start failed: {e}")
        print("Trying CPU_4...")
        try:
            studio.start(machine="CPU_4")
            print("Start with CPU_4 initiated")
        except Exception as e2:
            print(f"CPU_4 also failed: {e2}")
            sys.exit(1)

# Wait for running
for i in range(60):
    studio.refresh()
    print(f"  Attempt {i+1}: status = {studio.status}")
    if studio.status.lower() == "running":
        print("Studio is running!")
        break
    time.sleep(5)
else:
    print("Studio did not start within 5 minutes")
    sys.exit(1)

# Upload project files
print("\\nUploading project files...")
project_dir = "/Users/kendrick/projects/pure-snn-learning"
studio.run(f"mkdir -p /tmp/snn")
studio.upload_folder(project_dir, "/tmp/snn")
print("Upload complete")

# Run experiment
print("\\nRunning experiment...")
# The studio has torch pre-installed
result = studio.run("cd /tmp/snn && python lightning_run.py 2>&1")
print(f"\\n--- Experiment Output ---\\n{result}\\n--- End Output ---")

# Download results
print("\\nDownloading results...")
try:
    results_content = studio.run("cat /tmp/snn/results.json")
    print(f"Results JSON:\\n{results_content}")
except:
    print("No results.json found, checking files...")
    ls_result = studio.run("ls -la /tmp/snn/")
    print(f"Files: {ls_result}")

# Stop studio to save free tier quota
print("\\nStopping studio...")
studio.stop()
print("Done")
"""

result = subprocess.run(
    [LIGHTNING_PYTHON, "-c", launch_code],
    capture_output=True, text=True, timeout=600
)
print(result.stdout)
if result.stderr:
    for line in result.stderr.split('\\n'):
        if 'UserWarning' not in line and 'warnings.warn' not in line:
            print(f"STDERR: {line}")
