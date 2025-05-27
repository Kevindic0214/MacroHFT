import sys
import pathlib
import pdb
import numpy as np

# Get the path to the MacroHFT-main directory (parent of env directory)
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Add project root to path
sys.path.append(str(PROJECT_ROOT))
sys.path.insert(0, ".")

# Construct paths to the feature list files
single_features_path = PROJECT_ROOT / 'data' / 'feature_list' / 'single_features.npy'
trend_features_path = PROJECT_ROOT / 'data' / 'feature_list' / 'trend_features.npy'

# Load the feature lists
try:
    tech_indicator_list = np.load(single_features_path, allow_pickle=True).tolist()
    print("Single features:", tech_indicator_list)
    
    tech_indicator_list_trend = np.load(trend_features_path, allow_pickle=True).tolist()
    print("Trend features:", tech_indicator_list_trend)
except FileNotFoundError as e:
    print(f"Error: {e}")
    print(f"Attempted to load from: {single_features_path}")
    print(f"Project root is: {PROJECT_ROOT}")
    print("Please ensure the data files exist in the expected location.")