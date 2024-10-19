"""
This module initializes the bettertransformers package, which provides a manager and transformation utilities to
convert transformers models into their optimized versions using the BetterTransformer architecture.

Modules imported:
- BetterTransformerManager: Manages the conversion of HuggingFace models to BetterTransformer models.
- BetterTransformer: Provides the transformation logic for models.
"""

# Importing BetterTransformerManager class to manage conversion processes of HuggingFace models.
from .models import BetterTransformerManager

# Importing BetterTransformer class to handle the actual transformation of the models.
from .transformation import BetterTransformer
