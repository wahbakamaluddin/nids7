from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent.resolve()
FLOW_TIMEOUT = 120
CLUMP_TIMEOUT = 1
ACTIVE_TIMEOUT = 5
BULK_BOUND = 4
PACKETS_PER_GC = 1000
