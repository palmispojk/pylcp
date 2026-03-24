"""Sr88 green MOT constants shared by simulation and plotting scripts."""
import numpy as np
import scipy.constants as const

frq_real = 603976506.6e6 * 2 * np.pi
gamma_real = 61.4e6
kmag_real = frq_real / const.c
muB_real = const.physical_constants["Bohr magneton"][0]
mass_real = const.value('atomic mass constant') * 88
alpha_real = 0.4  # T/m

# Natural units
gamma = 1
kmag = 1
muB = 1
mass = mass_real * gamma_real / const.hbar / kmag_real**2
alpha = alpha_real * muB_real / (gamma_real * kmag_real * const.hbar)
det = -2.1 * gamma
s = 2

# Initial condition sampling scales
rscale = np.array([2, 2, 2]) / alpha
roffset = np.array([0.0, 0.0, 0.0])
vscale = np.array([0.1, 0.1, 0.1])
voffset = np.array([0.0, 0.0, 0.0])