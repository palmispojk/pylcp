"""Sr88 low-power blue MOT constants (1S0 -> 1P1, 461 nm).

Stage 2 of Kristensen sequence (PhD thesis NBI 2023, Sec. 3.1.2 & 4.2):
after the high-power loading MOT, the MOT-AOM RF drive is halved (600 -> 300 mV)
for 50 ms. Cloud cools 3-5 mK -> 1-3 mK and diffuses, ~doubling transfer
efficiency to the red MOT. Same transition / detuning / gradient as the
high-power blue MOT — only saturation differs.
Natural units: gamma = k = muB = 1.
"""
import numpy as np
import scipy.constants as const

# Physical constants
wavelength = 460.862e-9
frq_real   = const.c / wavelength * 2 * np.pi
gamma_real = 2 * np.pi * 30.24e6
kmag_real  = 2 * np.pi / wavelength
muB_real   = const.physical_constants["Bohr magneton"][0]
mass_real  = const.value('atomic mass constant') * 88

# Natural units
gamma = 1
kmag  = 1
muB   = 1
mass  = mass_real * gamma_real / const.hbar / kmag_real**2

# MOT parameters
det   = -1.3228 * gamma   # -40 MHz, unchanged from blue MOT
s     = 0.087             # halved from blue MOT s=0.173 (Tang-calibrated)
# Quadrupole gradient still 35 G/cm here; ramp to red MOT happens at the
# blue->red handoff (Kristensen Sec. 3.5.1).
alpha = 0.35              # T/m

alpha_nat = alpha * muB_real / (gamma_real * kmag_real * const.hbar)

# Fallback initial conditions (used only if no upstream pickle):
# 4 mK isotropic cloud at trap centre (Kristensen Table 4.2). sigma_v = sqrt(kB*T/m).
_v_scale = kmag_real / gamma_real
_r_scale = kmag_real
_T_fallback_K = 4e-3
_sigma_v_si   = np.sqrt(const.Boltzmann * _T_fallback_K / mass_real)
_sigma_r_si   = 1e-3       # 1 mm cloud radius

vscale  = np.full(3, _sigma_v_si * _v_scale)
voffset = np.zeros(3)
rscale  = np.full(3, _sigma_r_si * _r_scale)
roffset = np.zeros(3)

a_grav = np.array([0.0, 0.0, -const.g * kmag_real / gamma_real**2])

# Simulation control
# Stage duration: 50 ms -> 50e-3 * gamma_real ~ 9.5e6 nat
tmax      = 1e7
MAX_ATOMS = 65536
