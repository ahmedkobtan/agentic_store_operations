from pathlib import Path
from walmart_ahmedkobtan_agentic_store_operations.src.utils.yaml_utils import read_yaml

# Determine project root
REPO_NAME = "agentic_store_operations"
current_path = Path(__file__).resolve()
for p in current_path.parents:
    if p.name ==  "walmart_ahmedkobtan_" + REPO_NAME:
        ROOT_DIR = p
        break
    if p.name == REPO_NAME:
        ROOT_DIR = p / Path("walmart_ahmedkobtan_" + REPO_NAME)
        break

# Determine necessary paths
DATA_DIR = ROOT_DIR / Path("data")
ROSTER_FILE = DATA_DIR / Path("roster/employees.csv")
RAW_PATH_1 = DATA_DIR / Path("raw/online_retail_II_2009.xlsx")
RAW_PATH_2 = DATA_DIR / Path("raw/online_retail_II_2010.xlsx")
PROCESSED_OUT_DIR  = DATA_DIR / Path("processed")
PROCESSED_OUT_PATH = PROCESSED_OUT_DIR / "hourly_demand.parquet"
ARTIFACT_OUT_DIR = DATA_DIR / Path("artifacts")
DATA_CONFIG_PATH = DATA_DIR / Path("configs")
MODELS_PATH = ROOT_DIR / Path("models")
PROPHET_MODEL_PATH = MODELS_PATH / Path("prophet.pkl")


# forecasting and employee roster constants
WEEKDAYS = ["mon","tue","wed","thu","fri","sat","sun"]

CONSTRAINTS_YAML_PATH = DATA_CONFIG_PATH / "constraints.yaml"
FORECASTING_GUARDRAILS_YAML_PATH = DATA_CONFIG_PATH / "forecasting_guardrails.yaml"
CONSTRAINTS = read_yaml(CONSTRAINTS_YAML_PATH)
FORECASTING_GUARDRAILS = read_yaml(FORECASTING_GUARDRAILS_YAML_PATH)


# hour constants
OPEN_HOUR = int(CONSTRAINTS.get("store", {}).get("open_hour", 6))
CLOSE_HOUR = int(CONSTRAINTS.get("store", {}).get("close_hour", 21))

# forecasting guardrail constants
MIN_VOL = float(FORECASTING_GUARDRAILS.get("guard_rails", {}).get("min_vol", 20.0))  # drop windows whose total actuals < MIN_VOL when reporting per-window stats
PROMOTE_TO_P90_THRESHOLD = float(FORECASTING_GUARDRAILS.get("quantiles", {}).get("promote_to_p90_if_rel_uncertainty_ge", 0.5))  # relative uncertainty threshold to promote p50 to p90
BASE_Q = float(FORECASTING_GUARDRAILS.get("quantiles", {}).get("base_quantile", 0.50))
ALWAYS_P90_CFG = FORECASTING_GUARDRAILS.get("always_p90", [])

# role target constants
TX_PER_CASHIER = float(FORECASTING_GUARDRAILS.get("cashier_mapping", {}).get("tx_per_cashier_per_hour", 20.0))
CASHIER_TRIGGER = float(FORECASTING_GUARDRAILS.get("cashier_mapping", {}).get("cashier_trigger_tx_per_hour", 5.0))
LEAD_ALL_OPEN = bool(FORECASTING_GUARDRAILS.get("lead_policy", {}).get("lead_required_all_open_hours", True))

# Feedback bias constants
ALPHA = 0.4  # EWMA weight
MAX_BIAS = 2.0

# Schema and model/solver versioning
SCHEMA_VERSION = "1.0"
MODEL_VERSION = "forecast_how_v1"
SOLVER_VERSION = "cp_sat_v2"  # includes shortfall/overstaff formulation
