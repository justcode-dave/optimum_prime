"""
Constants for Promise Optimizer

This module defines key constants used across the Promise Optimizer project. These constants
are primarily related to model configuration file names, subfolder names for diffusion models, 
and the default file name for ONNX weights. They serve as centralized references to ensure 
consistency and avoid hardcoding values throughout the project.

Attributes:
    CONFIG_NAME (str): The default name for model configuration files.
    DIFFUSION_MODEL_UNET_SUBFOLDER (str): Subfolder name for the UNet component of diffusion models.
    DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER (str): Subfolder name for the text encoder of diffusion models.
    DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER (str): Subfolder name for the VAE decoder of diffusion models.
    DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER (str): Subfolder name for the VAE encoder of diffusion models.
    DIFFUSION_MODEL_TEXT_ENCODER_2_SUBFOLDER (str): Subfolder name for the second text encoder in diffusion models.
    ONNX_WEIGHTS_NAME (str): The default file name for ONNX model weights.
"""

# Name of the configuration file used for models
CONFIG_NAME = "config.json"

# Subfolder names for diffusion model components
DIFFUSION_MODEL_UNET_SUBFOLDER = "unet"  # Subfolder for UNet architecture in diffusion models
DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER = "text_encoder"  # Subfolder for text encoder in diffusion models
DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER = "vae_decoder"  # Subfolder for VAE decoder in diffusion models
DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER = "vae_encoder"  # Subfolder for VAE encoder in diffusion models
DIFFUSION_MODEL_TEXT_ENCODER_2_SUBFOLDER = "text_encoder_2"  # Subfolder for secondary text encoder in some models

# Default file name for ONNX model weights
ONNX_WEIGHTS_NAME = "model.onnx"
