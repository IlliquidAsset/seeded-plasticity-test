#!/usr/bin/env python3
"""Transfer and run debug test on Lightning VM."""
import sys
import os
import base64

sys.path.insert(0, os.path.expanduser("~/.hermes/hermes-agent/venv/lib/python3.11/site-packages"))
from lightning_sdk import Studio

# Read the debug script
script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vm_debug_test.py')
with open(script_path, 'rb') as f:
    encoded = base64.b64encode(f.read()).decode()

s = Studio(name='aib-arena', teamspace='illiquidasset-org/big-girl-production')
print(f"Studio: {s.status}")

# Transfer and run
s.run(f'echo {encoded} | base64 -d > /tmp/snn/vm_debug_test.py')
result = s.run('cd /tmp/snn && python3 vm_debug_test.py 2>&1')
print(result)
