"""
Global constants and configuration for the Latent Space Firewall.
"""

# Model configuration
MODEL_NAME = "gpt2"  # Default model for experiments

# Layer configuration
TARGET_LAYERS = [6, 7, 8]  # Layers to monitor for activations

# Threshold configuration
ACTIVATION_THRESHOLD = 0.5  # Threshold for flagging suspicious activations
CONFIDENCE_THRESHOLD = 0.8  # Minimum confidence for intervention

# Data paths
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"

# Intervention settings
ENABLE_INTERVENTION = False  # Set to True to enable active blocking
