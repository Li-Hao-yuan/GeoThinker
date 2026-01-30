"""Factory for creating geometry encoders."""

from typing import Optional
from .base import BaseGeometryEncoder, GeometryEncoderConfig
from .vggt_encoder import VGGTEncoder
from .pi3_encoder import Pi3Encoder


def create_geometry_encoder(
    encoder_type: str,
    model_path: Optional[str] = None,
    reference_frame: str = "first",
    freeze_encoder: bool = True,
    **encoder_kwargs
) -> BaseGeometryEncoder:
    """
    Factory function to create geometry encoders.
    
    Args:
        encoder_type: Type of encoder ("vggt", "pi3", etc.)
        model_path: Path to pretrained model
        reference_frame: Reference frame setting
        freeze_encoder: Whether to freeze encoder parameters
        **encoder_kwargs: Additional encoder-specific arguments
        
    Returns:
        Geometry encoder instance
    """
    config = GeometryEncoderConfig(
        encoder_type=encoder_type,
        model_path=model_path,
        reference_frame=reference_frame,
        freeze_encoder=freeze_encoder,
        encoder_kwargs=encoder_kwargs
    )
    
    if encoder_type == "vggt":
        return VGGTEncoder(config)
    elif encoder_type == "pi3":
        return Pi3Encoder(config)
    else:
        raise ValueError(f"Unknown geometry encoder type: {encoder_type}")


def get_available_encoders():
    """Get list of available encoder types."""
    return ["vggt", "pi3"]
