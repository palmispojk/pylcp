"""Sr88 single-frequency (SF) red MOT constants (1S0 -> 3P1, 689 nm).

Stage 4 of Kristensen sequence (PhD thesis NBI 2023, Sec. 3.2.3 & 4.2):
after the BB stage, FM is turned off, beams run at a single detuning, and power
is linearly ramped 10x down over 50 ms. End T ~2-3 uK (well below the 800 kHz
cavity linewidth); cloud sags under gravity into an elliptical shell.
Natural units: gamma = k = muB = 1.
"""
import numpy as np
import scipy.constants as const

# Physical constants
wavelength = 689.449e-9
frq_real   = const.c / wavelength * 2 * np.pi
gamma_real = 2 * np.pi * 7.48e3
kmag_real  = 2 * np.pi / wavelength
muB_real   = const.physical_constants["Bohr magneton"][0]
mass_real  = const.value('atomic mass constant') * 88

# Natural units
gamma = 1
kmag  = 1
muB   = 1
mass  = mass_real * gamma_real / const.hbar / kmag_real**2

# MOT parameters
# Single-frequency detuning: Kristensen Fig. 4.11 uses 0, -500 kHz, -1 MHz;
# -500 kHz produces the compact, mildly sagged cloud chosen here.
det_real_Hz = -500e3
det     = 2 * np.pi * det_real_Hz / gamma_real   # ~ -67 gamma
# Linear 10x ramp from BB-stage end value down over the 50 ms stage; the ramp
# is driven by a callable s(R, t) in sf_red_mot_sim.py.
s_start = 1270.0
s_end   = 127.0
# Gradient unchanged from BB stage (Kristensen Sec. 4.4.2, Fig. 4.12)
alpha   = 0.0374          # T/m  (3.74 G/cm)

alpha_nat = alpha * muB_real / (gamma_real * kmag_real * const.hbar)

# Fallback initial conditions (used only if no upstream pickle):
# 15 uK at end of BB stage, compact 0.5 mm red MOT cloud
_v_scale = kmag_real / gamma_real
_r_scale = kmag_real
_T_fallback_K = 15e-6
_sigma_v_si   = np.sqrt(const.Boltzmann * _T_fallback_K / mass_real)
_sigma_r_si   = 0.5e-3

vscale  = np.full(3, _sigma_v_si * _v_scale)
voffset = np.zeros(3)
rscale  = np.full(3, _sigma_r_si * _r_scale)
roffset = np.zeros(3)

# Simulation control
# Stage duration: 50 ms -> 50e-3 * gamma_real ~ 2.35e3 nat
tmax      = 3e3
MAX_ATOMS = 65536
