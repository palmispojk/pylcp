"""
pylcp - Python Laser Cooling Physics

A JAX-accelerated simulation package for laser cooling physics, providing tools
for computing optical Bloch equations, rate equations, and heuristic models of
atom-light interactions in the presence of laser beams and magnetic fields.
"""
import numpy as np

from . import hamiltonians
from .atom import atom
from .heuristiceq import heuristiceq
from .rateeq import rateeq
from .obe import obe
from .hamiltonian import hamiltonian
from .fields import (magField, constantMagneticField, quadrupoleMagneticField, iPMagneticField,
                     laserBeam, laserBeams, infinitePlaneWaveBeam, gaussianBeam,
                     clippedGaussianBeam, conventional3DMOTBeams)
