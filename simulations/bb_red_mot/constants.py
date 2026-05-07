"""
Sr88 broadband (BB) red MOT constants (1S0 -> 3P1, 689 nm).

Third cooling stage in the Kristensen sequence (PhD thesis, NBI 2023,
Sec. 3.2.3 & 4.2): right after the low-power blue MOT is turned off, the
689 nm beams are switched on with their frequency swept 3 MHz wide at a
50 kHz repetition rate to broaden the narrow-line capture range. The
gradient is simultaneously ramped from ~35 G/cm (blue MOT) down to a few
G/cm. Stage duration 70 ms; end temperature 10-20 uK, up to 1.2e8 atoms.

The 3P1 linewidth is 7.48 kHz (tau = 21.3 us), so the 3 MHz sweep spans
~400 gamma — pylcp.conventional3DMOTBeams takes a single detuning, so the
BB sweep is approximated here by a constant effective detuning at the
centre of the swept range. Run a second simulation with the SF MOT
parameters for the single-frequency stage.

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
# Detuning: Kristensen sweeps a 3 MHz comb (50 kHz rep) around atomic
# resonance. Use the centre of the swept range as a constant-detuning
# approximation; set the sweep width for reference.
sweep_width_Hz = 3.0e6
sweep_rate_Hz  = 50e3
# Centre detuning ~ half the sweep below resonance (Kristensen fig. 3.10
# shows the comb sitting between -4 MHz and atomic resonance).
det_real_Hz = -1.5e6
det   = 2 * np.pi * det_real_Hz / gamma_real    # ~ -200 gamma

# Saturation: 3 mW per beam through 1 cm-diameter MOT beams.
# I_sat(3P1) = 3 uW/cm^2 ; I = 3 mW / (pi*(0.5 cm)^2) = 3.82 mW/cm^2.
# s0 = I/I_sat ~ 1270 per beam — typical for a narrow-line red MOT that
# uses FM to spread power over a wide detuning range. The effective
# single-frequency saturation seen by any given velocity class is lower.
s     = 1270.0

# Quadrupole gradient: 3-5 G/cm during the red MOT (Kristensen inferred
# 3.74 G/cm from SF-MOT gravitational sag, Sec. 4.4.2).
alpha = 0.0374            # T/m  (= 3.74 G/cm)

# Convert gradient to natural units
alpha_nat = alpha * muB_real / (gamma_real * kmag_real * const.hbar)

# ============================================================================
#  INITIAL CONDITIONS
#
#  Loaded from low_power_blue_mot/low_power_blue_mot_final_state.npz.
#  Fallback scales below: 1-3 mK isotropic cloud at the trap center.
# ============================================================================
_v_scale = kmag_real / gamma_real
_r_scale = kmag_real

_T_fallback_K = 2e-3       # end of the low-power blue MOT stage
_sigma_v_si   = np.sqrt(const.Boltzmann * _T_fallback_K / mass_real)
_sigma_r_si   = 1.5e-3     # cloud is more diffuse than the loading MOT

vscale  = np.full(3, _sigma_v_si * _v_scale)
voffset = np.zeros(3)
rscale  = np.full(3, _sigma_r_si * _r_scale)
roffset = np.zeros(3)

# ============================================================================
#  SIMULATION CONTROL
# ============================================================================
# Stage duration: 70 ms  ->  70e-3 * gamma_real ~ 3.3e3 natural time units
tmax            = 4e3
MAX_ATOMS       = 65536
