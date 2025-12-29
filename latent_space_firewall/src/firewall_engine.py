"""
The core intervention logic for the Latent Space Firewall.
Intersects the residual stream to detect adversarial intent vectors.
"""

import numpy as np
import pickle
import torch
from typing import Dict, Optional
from transformer_lens import HookedTransformer

from latent_space_firewall.src import config

class FirewallEngine:
    def __init__(self):
        """
        Initialize the firewall engine and load the trained probe.
        """
        self.threshold = config.ACTIVATION_THRESHOLD
        self.layer_idx = config.TARGET_LAYER_IDX
        self.classifier = self._load_classifier()
        
    def _load_classifier(self):
        """
        Loads the Logistic Regression probe from the models directory.
        """
        try:
            with open(config.MODEL_PATH, 'rb') as f:
                clf = pickle.load(f)
            print(f"✅ Firewall Probe Loaded from {config.MODEL_PATH}")
            return clf
        except FileNotFoundError:
            print(f"⚠️ Warning: Classifier not found at {config.MODEL_PATH}. Run Phase 2 notebook.")
            return None

    def scan(self, prompt: str, model: HookedTransformer) -> Dict:
        """
        Performs a 'dry run' forward pass to extract activations and detect harm.
        
        Args:
            prompt: The user input text.
            model: The loaded HookedTransformer instance (passed from app.py).
            
        Returns:
            Dictionary containing the harm score and blocking decision.
        """
        # 1. Extract the Latent Vector (Mechanistic Interpretability)
        # We hook into the residual stream at the final token of the prompt.
        hook_name = f"blocks.{self.layer_idx}.hook_resid_post"
        
        # We use run_with_cache to surgically extract just the vector we need
        with torch.no_grad():
            _, cache = model.run_with_cache(
                prompt, 
                names_filter=lambda name: name == hook_name
            )
            
        # Shape: [batch, pos, d_model] -> Extract last token of first sequence
        last_token_vector = cache[hook_name][0, -1, :].cpu().numpy()
        
        # 2. Project onto the Probe (The "Mind Reading" Step)
        if self.classifier:
            # Get the probability of class 1 (Harmful)
            # Reshape to (1, -1) because sklearn expects a batch
            harm_prob = self.classifier.predict_proba(last_token_vector.reshape(1, -1))[0, 1]
        else:
            # Fallback if model missing (fail-safe)
            harm_prob = 0.0

        # 3. Apply the Conformal Guarantee
        is_blocked = harm_prob > self.threshold
        
        return {
            "harm_score": float(harm_prob),
            "threshold": self.threshold,
            "is_blocked": bool(is_blocked),
            "activation_vector": last_token_vector  # Optional: for visualization
        }