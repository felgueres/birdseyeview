"""
Satellite imagery ingestion and processing for Earth observation.

Supports:
- Sentinel-2 (10m RGB/NIR via AWS Open Data)
"""

from bird.satellite.multispectral_viewer import MultispectralViewer, BandConfig

__all__ = ['MultispectralViewer', 'BandConfig']
