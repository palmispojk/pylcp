"""
pylcp - Python Laser Cooling Physics

A JAX-accelerated simulation package for laser cooling physics, providing tools
for computing optical Bloch equations, rate equations, and heuristic models of
atom-light interactions in the presence of laser beams and magnetic fields.
"""
import numpy as np

from . import hamiltonians
from .atom import atom
from .fields import (
                     clippedGaussianBeam,
                     constantMagneticField,
                     conventional3DMOTBeams,
                     gaussianBeam,
                     infinitePlaneWaveBeam,
                     iPMagneticField,
                     laserBeam,
                     laserBeams,
                     magField,
                     quadrupoleMagneticField,
)
from .hamiltonian import hamiltonian
from .heuristiceq import heuristiceq
from .obe import obe
from .rateeq import rateeq
