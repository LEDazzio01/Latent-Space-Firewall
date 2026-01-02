"""
Global constants and configuration for the Latent Space Firewall.
"""
import os

# Model configuration
MODEL_NAME = "gpt2-small"
TARGET_LAYER_IDX = 6  # The layer we harvested from

# Threshold configuration
# FALLBACK ONLY: Used if meta.json is missing from the model artifact.
# The authoritative threshold is saved alongside the model during training.
ACTIVATION_THRESHOLD_FALLBACK = 0.0355
CONFIDENCE_THRESHOLD = 0.90 

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "firewall_classifier.pkl")
META_PATH = os.path.join(BASE_DIR, "models", "meta.json")

# Intervention settings
ENABLE_INTERVENTION = True
