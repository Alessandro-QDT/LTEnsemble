"""
LTEnsemble Configuration
========================
Set your project settings here. All pipeline scripts will use these values.
"""

# =============================================================================
# PROJECT CONFIGURATION - EDIT THESE VALUES
# =============================================================================

# Your project ID (from QML)
PROJECT_ID = 21

# Project name (used for folder naming - no spaces, use underscores)
PROJECT_NAME = "USA_Beef_Tallow"

# =============================================================================
# DERIVED PATHS (don't edit these)
# =============================================================================

# Data directory path (note: folder has underscore prefix)
DATA_DIR = f"data/_{PROJECT_ID}_{PROJECT_NAME}"

# Main data file
PARQUET_PATH = f"{DATA_DIR}/all_children_data.parquet"

# Output files
GRID_SEARCH_RESULTS = f"{DATA_DIR}/grid_search_results.csv"
OPTIMAL_PARAMS = f"{DATA_DIR}/optimal_ewma_params.csv"
BASELINE_VS_EWMA = f"{DATA_DIR}/baseline_vs_ewma.csv"
EQUITY_CURVES = f"{DATA_DIR}/equity_curves.json"
CHART_DATA = f"{DATA_DIR}/chart_data.json"
SNAKE_DATA = f"{DATA_DIR}/snake_data.json"
DA_COMPARISON = f"{DATA_DIR}/da_comparison.json"
LIVE_FORECAST = f"{DATA_DIR}/live_ewma_forecast.json"
QA_RESULTS = f"{DATA_DIR}/quantitative_accuracy.csv"
MULTI_OBJ_RESULTS = f"{DATA_DIR}/multi_objective_results.csv"

# =============================================================================
# TRAIN/TEST SPLIT
# =============================================================================
TRAIN_END = "2024-12-31"
TEST_START = "2025-01-01"

# =============================================================================
# API CONFIGURATION (for fetch_data.py)
# =============================================================================
API_BASE_URL = "https://superforecast.cloud-effem.com"
API_ENDPOINT = "/api/v1/get_qml_models"
