"""
Global constants and configuration for the Latent Space Firewall.
"""
import os

# Model configuration
MODEL_NAME = "gpt2-small"
TARGET_LAYER_IDX = 6  # The layer we harvested from

# Threshold configuration (Calculated via Split Conformal Prediction)
# guarantees 95% Recall on the calibration set.
ACTIVATION_THRESHOLD = 0.0355 
CONFIDENCE_THRESHOLD = 0.90 

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "firewall_classifier.pkl")

# Intervention settings
ENABLE_INTERVENTION = True
