"""
Minimal standalone copy of DEIMv2 backbone + encoder modules,
for use inside AMOT without the full DEIMv2 engine package.

Removes:
  - @register() decorators (requires tensorboard via engine.core)
  - All DEIMv2-specific training/data/optim imports

Keeps:
  - DINOv3STAs (backbone)
  - HybridEncoder (neck)
"""
from .dinov3_adapter import DINOv3STAs, SpatialPriorModulev2
from .hybrid_encoder import HybridEncoder

__all__ = ['DINOv3STAs', 'SpatialPriorModulev2', 'HybridEncoder']
