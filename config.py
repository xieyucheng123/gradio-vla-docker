import os

OPENVLA_API = os.environ.get("OPENVLA_API", "http://localhost:8002")
ROBOTWIN_API = os.environ.get("ROBOTWIN_API", "http://localhost:8003")
DEFAULT_MAX_STEPS = 100
DEFAULT_TASK = "handover_block"
DEFAULT_INSTRUCTION = "hand over the red block to the blue target"
API_TIMEOUT = 30
LOOP_TIMEOUT = 300
