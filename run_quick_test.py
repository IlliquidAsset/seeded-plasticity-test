#!/usr/bin/env python3
"""Transfer and run quick test on Lightning VM."""
import sys
import os
import base64

# Add the Hermes venv to path for lightning_sdk
sys.path.insert(0, os.path.expanduser("~/.hermes/hermes-agent/venv/lib/python3.11/site-packages"))

from lightning_sdk import Studio

# Read the test script
script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vm_quick_test.py')
with open(script_path, 'rb') as f:
    encoded = base64.b64encode(f.read()).decode()

# Connect to studio
s = Studio(name='aib-arena', teamspace='illiquidasset-org/big-girl-production')
print(f"Studio status: {s.status}")

# Transfer script
print("Transferring script...")
result = s.run(f'echo {encoded} | base64 -d > /tmp/snn/vm_quick_test.py && chmod +x /tmp/snn/vm_quick_test.py')
print(f"Transfer result: {result}")

# Run the test
print("\nRunning quick test...")
result = s.run('cd /tmp/snn && python3 vm_quick_test.py 2>&1')
print(f"\n=== Quick Test Output ===\n{result}\n=== End Output ===")
