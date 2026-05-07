"""
Sr88 low-power blue MOT constants (1S0 -> 1P1, 461 nm).

Second stage of the Kristensen cooling sequence (PhD thesis, NBI 2023,
"Narrow linewidth superradiant lasing with cold 88Sr", Sec. 3.1.2 & 4.2):
after 950 ms of high-power blue MOT loading, the RF drive on the MOT AOM
is halved (600 mV -> 300 mV control voltage) for 50 ms. The cloud
becomes more diffuse but cools from 3-5 mK to 1-3 mK, doubling the
transfer efficiency into the red MOT.

Same transition, detuning, and gradient as the high-power blue MOT — the
only knob is saturation.

Natural units: gamma = k = muB = 1.
"""
import numpy as np
import scipy.constants as const

# ============================================================================
#  PHYSICAL CONSTANTS  (Sr88 1S0 -> 1P1, 461 nm)
# ============================================================================
wavelength = 460.862e-9                         # m
frq_real   = const.c / wavelength * 2 * np.pi   # angular frequency (rad/s)
gamma_real = 2 * np.pi * 30.24e6                # natural linewidth (rad/s)
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
# Detuning unchanged from the high-power stage: -40 MHz (Kristensen Sec. 3.1.2)
det   = -1.3228 * gamma

# Saturation: Kristensen halves the MOT-AOM drive. Taking the Tang-calibrated
# high-power value (s = 0.173 at 7 mW / 10 mm) and halving gives:
s     = 0.087

# Quadrupole gradient: still 35 G/cm during this stage — the ramp to the red
# MOT gradient (3-5 G/cm) happens on the blue->red MOT handoff (Sec. 3.5.1).
alpha = 0.35              # T/m

# Convert gradient to natural units
alpha_nat = alpha * muB_real / (gamma_real * kmag_real * const.hbar)

# ============================================================================
#  INITIAL CONDITIONS
#
#  Loaded from the upstream stage (blue_mot/blue_mot_final_state.npz).
#  The fallback scales below are only used for standalone runs where no
#  upstream file is available.
# ============================================================================
_v_scale = kmag_real / gamma_real
_r_scale = kmag_real

# Fallback: a 3-5 mK isotropic cloud at the trap center (Kristensen Table 4.2).
# Thermal velocity sigma_v = sqrt(kB*T/m).  T = 4 mK -> sigma_v ~ 0.66 m/s.
_T_fallback_K = 4e-3
_sigma_v_si   = np.sqrt(const.Boltzmann * _T_fallback_K / mass_real)
_sigma_r_si   = 1e-3       # 1 mm cloud radius

vscale  = np.full(3, _sigma_v_si * _v_scale)
voffset = np.zeros(3)
rscale  = np.full(3, _sigma_r_si * _r_scale)
roffset = np.zeros(3)

# ============================================================================
#  SIMULATION CONTROL
# ============================================================================
# Stage duration: 50 ms  ->  50e-3 * gamma_real ~ 9.5e6 natural time units
tmax            = 1e7
MAX_ATOMS       = 65536
