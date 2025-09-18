# Genome handlers for different coordinate representations

from .base import GenomeHandler
from .cartesian_euler_genome_handler import CartesianEulerDroneGenomeHandler
from .spherical_angular_genome_handler import SphericalAngularDroneGenomeHandler

# Optional MLP handler (requires PyTorch)
try:
    from .mlp_genome_handler import MLPDroneGenomeHandler
    MLP_AVAILABLE = True
except ImportError:
    MLP_AVAILABLE = False
    MLPDroneGenomeHandler = None

__all__ = [
    'GenomeHandler',
    'CartesianEulerDroneGenomeHandler', 
    'SphericalAngularDroneGenomeHandler'
]

if MLP_AVAILABLE:
    __all__.append('MLPDroneGenomeHandler')