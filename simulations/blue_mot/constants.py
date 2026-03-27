"""
Sr88 blue MOT constants (1S0 -> 1P1, 461 nm).

Default values are from M. Tang's thesis "Continuous superradiance" (2024),
Sec. 3.2.1, validated against experimental blue MOT data (T ~ 1 mK,
R ~ 1 mm, ~10^8 atoms). Override with your setup's values where noted.

Natural units: gamma = k = muB = 1.
"""
import numpy as np
import scipy.constants as const

# ============================================================================
#  PHYSICAL CONSTANTS  (Sr88 1S0 -> 1P1, 461 nm)
# ============================================================================
wavelength = 460.862e-9                         # m
frq_real   = const.c / wavelength * 2 * np.pi   # angular frequency (rad/s)
gamma_real = 2 * np.pi * 30.5e6                 # natural linewidth (rad/s)
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
# Detuning: 40 MHz chosen for larger capture range (Tang p.43)
# UvA setup uses ~45 MHz — change to -1.48 * gamma to match
det   = -1.31 * gamma    # 40 MHz / 30.5 MHz

# Saturation: 7 mW horizontal beams, 10 mm waist -> I/Isat = 0.173 (Tang p.43)
# UvA setup: ~10-15 mW, ~20 mm beams -> s ~ 0.09
s     = 0.173

# Quadrupole gradient: 370 mT/m vertical (Tang p.43)
# UvA setup: ~50 G/cm = 0.5 T/m (to be confirmed)
alpha = 0.37              # T/m

# Convert gradient to natural units
alpha_nat = alpha * muB_real / (gamma_real * kmag_real * const.hbar)

# ============================================================================
#  ZEEMAN SLOWER BEAM — initial conditions
#
#  Atoms exit the slower along +z into the MOT region.
#  Slower decelerates from hundreds of m/s to ~10 m/s (Tang p.39).
# ============================================================================
_v_scale = kmag_real / gamma_real   # natural velocity unit -> m/s
_r_scale = kmag_real                # natural position unit -> 1/m

# Longitudinal (z): slower output velocity (Tang p.39)
v_longitudinal_mean_si  = 15.0     # m/s
v_longitudinal_sigma_si = 5.0      # m/s

# Transverse (x, y): set by beam divergence, not measured in thesis
v_transverse_sigma_si = 2.0        # m/s (estimate)
v_transverse_mean_si  = 0.0        # m/s

# Position: atomic beam size at MOT, not given in thesis
# 2 mm sigma is a typical Zeeman slower output estimate
r_transverse_sigma_si   = 2.0e-3   # m
r_longitudinal_sigma_si = 3.0e-3   # m
z_offset_si             = -5.0e-3  # m (beam enters from -z)

# Convert to natural units
v_longitudinal_mean  = v_longitudinal_mean_si  * _v_scale
v_longitudinal_sigma = v_longitudinal_sigma_si * _v_scale
v_transverse_mean    = v_transverse_mean_si    * _v_scale
v_transverse_sigma   = v_transverse_sigma_si   * _v_scale
r_transverse_sigma   = r_transverse_sigma_si   * _r_scale
r_longitudinal_sigma = r_longitudinal_sigma_si * _r_scale
z_offset             = z_offset_si             * _r_scale

# Packed arrays for the simulation (x, y, z)
vscale  = np.array([v_transverse_sigma, v_transverse_sigma, v_longitudinal_sigma])
voffset = np.array([v_transverse_mean,  v_transverse_mean,  v_longitudinal_mean])
rscale  = np.array([r_transverse_sigma, r_transverse_sigma, r_longitudinal_sigma])
roffset = np.array([0.0, 0.0, z_offset])

# ============================================================================
#  SIMULATION CONTROL
# ============================================================================
# 1 / gamma_real ~ 5.2 ns, so 1e6 natural units ~ 5 ms
# Tang Fig 3.8 (p.43) shows equilibrium within 15 ms; start with 5 ms
tmax            = 1e6
MAX_STEPS       = 2_000_000
SAVE_EVERY      = 400        # 5000 output points
INNER_MAX_STEPS = 64
MAX_ATOMS       = 8192
