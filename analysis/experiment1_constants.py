"""Common constants for experiment 1 scripts.

These constants define shared directory names and model filenames so that
related scripts can build paths consistently without hardcoding values.
"""

# Directory names
RESULTS_DIR = "results"
OPTIMAL_DIR = "optimal"
EXPLANATIONS_DIR = "explanations"

# Model filenames
LOGISTIC_MODEL_FILENAME = "logistic_model.joblib"
FCNN_MODEL_FILENAME = "fcnn_model.pth"
PNET_MODEL_FILENAME = "trained_model.pth"
PNET_CONFIG_FILENAME = "best_params.json"
PNET_NORES_MODEL_FILENAME = "trained_model_nores.pth"
PNET_NORES_CONFIG_FILENAME = "best_params_nores.json"

# Hyperparameters
FCNN_HIDDEN_DIM = 128
DEFAULT_BATCH_SIZE = 32

