from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

# Finds the main project folder automatically.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Location of the NASA CMAPSS dataset.
DATA_DIR = PROJECT_ROOT / "data" / "CMAPSSData"

# Files we will use for the FD001 experiment.
TRAIN_FILE = DATA_DIR / "train_FD001.txt"
TEST_FILE = DATA_DIR / "test_FD001.txt"
RUL_FILE = DATA_DIR / "RUL_FD001.txt"


# ============================================================
# DATASET COLUMNS
# ============================================================

# CMAPSS files don't contain column names.
# We define them ourselves.
COLUMNS = (
    ["unit", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


# ============================================================
# MODEL SETTINGS
# ============================================================

# If RUL <= 30 cycles, we'll consider the engine
# to be inside our critical maintenance window.
FAILURE_THRESHOLD = 30

# Number of previous cycles used to calculate
# temporal/rolling features.
WINDOW_SIZE = 15


# ============================================================
# BUSINESS COST ASSUMPTIONS
# ============================================================

AOG_COST = 150_000

PREVENTIVE_MAINTENANCE_COST = 25_000

INVENTORY_CARRYING_COST_PER_CYCLE = 200