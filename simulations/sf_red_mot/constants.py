"""
Sr88 single-frequency (SF) red MOT constants (1S0 -> 3P1, 689 nm).

Final cooling stage in the Kristensen sequence (PhD thesis, NBI 2023,
Sec. 3.2.3 & 4.2): after the 70 ms broadband stage, the frequency
modulation is turned off, the beams run at a single detuning, and the
power is linearly ramped an order of magnitude down over 50 ms. End
temperature 2-3 uK (gamma_Dopp/2pi ~ 47 kHz, well below the 800 kHz
cavity linewidth), cloud sags under gravity into an elliptical shell.

Natural units: gamma = k = muB = 1.
"""
import numpy as np
import scipy.constants as const

# ============================================================================
#  PHYSICAL CONSTANTS  (Sr88 1S0 -> 3P1, 689 nm)
# ============================================================================
wavelength = 689.449e-9                         # m
frq_real   = const.c / wavelength * 2 * np.pi   # angular frequency (rad/s)
gamma_real = 2 * np.pi * 7.48e3                 # natural linewidth (rad/s)
kmag_real  = 2 * np.pi / wavelength             # wavevector (1/m)
muB_real   = const.physical_constants["Bohr magneton"][0]  # J/T
mass_real  = const.value('atomic mass constant') * 88      # kg

# ============================================================================
#  NATURAL UNITS  (gamma = k = muB = 1)
# ============================================================================
gamma = 1
kmag  = 1
muB   = 1
mass  = mass_real * gamma_real / const.hbar / kmag_real**2

# ============================================================================
#  MOT PARAMETERS
# ============================================================================
# Detuning: single frequency, typically a few hundred kHz red of resonance.
# Kristensen Fig. 4.11 uses 0, -500 kHz, and -1 MHz; -500 kHz is the common
# operating point that produces a compact, mildly sagged cloud.
det_real_Hz = -500e3
det   = 2 * np.pi * det_real_Hz / gamma_real    # ~ -67 gamma

# Saturation: the BB stage starts at ~3 mW/beam (s ~ 1270), and the SF stage
# ramps this down by ~10x over 50 ms. End-of-ramp power ~0.3 mW/beam ->
# s ~ 127. Use the geometric mean of the ramp as a representative value for a
# constant-parameter run; override per-segment if you simulate the ramp.
s     = 380.0             # geometric mean of 1270 and 127

# Quadrupole gradient: unchanged from the BB stage. 3.74 G/cm from the
# gravitational sag calibration (Kristensen Sec. 4.4.2, Fig. 4.12).
alpha = 0.0374            # T/m  (= 3.74 G/cm)

# Convert gradient to natural units
alpha_nat = alpha * muB_real / (gamma_real * kmag_real * const.hbar)

# ============================================================================
#  INITIAL CONDITIONS
#
#  Loaded from bb_red_mot/bb_red_mot_final_state.npz.
#  Fallback scales below: 10-20 uK cloud at the trap centre.
# ============================================================================
_v_scale = kmag_real / gamma_real
_r_scale = kmag_real

_T_fallback_K = 15e-6      # end of the BB stage
_sigma_v_si   = np.sqrt(const.Boltzmann * _T_fallback_K / mass_real)
_sigma_r_si   = 0.5e-3     # compact red MOT cloud

vscale  = np.full(3, _sigma_v_si * _v_scale)
voffset = np.zeros(3)
rscale  = np.full(3, _sigma_r_si * _r_scale)
roffset = np.zeros(3)

# ============================================================================
#  SIMULATION CONTROL
# ============================================================================
# Stage duration: 50 ms  ->  50e-3 * gamma_real ~ 2.35e3 natural time units
tmax            = 3e3
MAX_ATOMS       = 65536
