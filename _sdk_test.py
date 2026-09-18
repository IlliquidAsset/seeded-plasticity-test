#!/usr/bin/env python3
"""Quick SDK test - check studio capabilities."""
from lightning_sdk import Studio

s = Studio(name="big-girl-motion")
print("Methods:", [m for m in dir(s) if not m.startswith('_')])
print("Status:", s.status)
print("Teamspace:", s.teamspace)
