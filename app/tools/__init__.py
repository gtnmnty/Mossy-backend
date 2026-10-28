# Feature 5: Tools registry
# TOOLS is the list of schemas sent to Claude so it knows which tools exist.
# REGISTRY maps each tool name to the Python function that runs it.
# To add a tool: make a module with SCHEMA and run(), then add it to _MODULES.

from app.tools import time_tool, marine

_MODULES = [time_tool, marine]
TOOLS = [m.SCHEMA for m in _MODULES]
REGISTRY = {m.SCHEMA["name"]: m.run for m in _MODULES}