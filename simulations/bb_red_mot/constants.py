"""Sr88 broadband (BB) red MOT constants (1S0 -> 3P1, 689 nm).

Stage 3 of Kristensen sequence (PhD thesis NBI 2023, Sec. 3.2.3 & 4.2):
689 nm beams switched on with frequency swept 3 MHz wide at 50 kHz repetition
to broaden the narrow-line capture range. Gradient simultaneously ramped from
~35 G/cm (blue MOT) down to ~3.7 G/cm. Stage duration ~70 ms; end T ~10-20 uK.

The 7.48 kHz natural linewidth is much smaller than the 3 MHz sweep, so the FM
spans ~400 gamma. The sweep is implemented as a triangle-wave phase modulation
in bb_red_mot_sim.py (analytic phase integral); the carrier sits at det.
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

# FM sweep parameters (Kristensen Fig. 3.10: comb sits between -4 MHz and
# atomic resonance, so carrier ~ half-sweep below resonance)
sweep_width_Hz = 3.0e6
sweep_rate_Hz  = 50e3
det_real_Hz    = -1.5e6
det   = 2 * np.pi * det_real_Hz / gamma_real    # ~ -200 gamma

# MOT parameters
# 3 mW per beam through 1 cm-diameter beams: I = 3 mW / (pi*(0.5 cm)^2) ~ 3.82 mW/cm^2;
# I_sat(3P1) = 3 uW/cm^2 -> s ~ 1270 per beam. The effective single-frequency
# saturation seen by any given velocity class is much lower (sweep spreads power).
s     = 1270.0
# Quadrupole gradient inferred from gravitational sag of the SF MOT
# (Kristensen Sec. 4.4.2)
alpha = 0.0374            # T/m  (3.74 G/cm)

alpha_nat = alpha * muB_real / (gamma_real * kmag_real * const.hbar)

# Fallback initial conditions (used only if no upstream pickle):
# 2 mK cloud, 1.5 mm sigma — more diffuse than the blue MOT (end of low-power stage)
_v_scale = kmag_real / gamma_real
_r_scale = kmag_real
_T_fallback_K = 2e-3
_sigma_v_si   = np.sqrt(const.Boltzmann * _T_fallback_K / mass_real)
_sigma_r_si   = 1.5e-3

vscale  = np.full(3, _sigma_v_si * _v_scale)
voffset = np.zeros(3)
rscale  = np.full(3, _sigma_r_si * _r_scale)
roffset = np.zeros(3)

# Simulation control
# Stage duration: 70 ms -> 70e-3 * gamma_real ~ 3.3e3 nat
tmax      = 4e3
MAX_ATOMS = 65536
