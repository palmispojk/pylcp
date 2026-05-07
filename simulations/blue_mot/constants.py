"""Sr88 blue MOT constants (1S0 -> 1P1, 461 nm).

Defaults from M. Tang thesis "Continuous superradiance" (2024) Sec. 3.2.1,
validated against experimental blue MOT data (T ~ 1 mK, R ~ 1 mm, ~1e8 atoms).
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

# MOT parameters (Tang p.43)
det   = -1.3228 * gamma   # -40 MHz, chosen for larger capture range
s     = 0.173             # xy beams: 7 mW per beam, 10 mm waist
s_z   = 0.121             # z beams: 4.9 mW per beam, 10 mm waist
alpha = 0.37              # T/m  (370 mT/m vertical)

alpha_nat = alpha * muB_real / (gamma_real * kmag_real * const.hbar)

# Zeeman slower beam initial conditions (Tang p.39).
# Beam axis lies in the xy plane at 45 deg from +x toward +y; orthogonal to z.
beam_axis_xy_angle = np.pi / 4

_v_scale = kmag_real / gamma_real
_r_scale = kmag_real

# Longitudinal: Gaussian around 20 m/s, clipped at 0 in sim
v_longitudinal_mean_si  = 20.0
v_longitudinal_sigma_si = 5.0
# Transverse: estimated from beam divergence (not measured in thesis)
v_transverse_sigma_si   = 2.0
v_transverse_mean_si    = 0.0
# Position: 2 mm sigma is a typical Zeeman slower output
r_transverse_sigma_si   = 2.0e-3
r_longitudinal_sigma_si = 3.0e-3
longitudinal_offset_si  = -5.0e-3

# Beam-frame scales/offsets (axis 0 = beam direction, 1,2 = transverse).
vscale_beam  = np.array([v_longitudinal_sigma_si, v_transverse_sigma_si, v_transverse_sigma_si]) * _v_scale
voffset_beam = np.array([v_longitudinal_mean_si,  v_transverse_mean_si,  v_transverse_mean_si])  * _v_scale
rscale_beam  = np.array([r_longitudinal_sigma_si, r_transverse_sigma_si, r_transverse_sigma_si]) * _r_scale
roffset_beam = np.array([longitudinal_offset_si * _r_scale, 0.0, 0.0])

# Rotation about z that maps beam frame -> lab frame.
# Beam-frame axis 0 -> (cos a, sin a, 0); axis 1 -> (-sin a, cos a, 0); axis 2 -> z.
_c, _s = np.cos(beam_axis_xy_angle), np.sin(beam_axis_xy_angle)
R_beam = np.array([[_c, -_s, 0.0],
                   [_s,  _c, 0.0],
                   [0.0, 0.0, 1.0]])
beam_dir = R_beam @ np.array([1.0, 0.0, 0.0])

a_grav = np.array([0.0, 0.0, -const.g * kmag_real / gamma_real**2])

# Simulation control
# 1/gamma_real ~ 5.2 ns -> 1e6 nat ~ 5 ms; Tang Fig 3.8 shows equilibrium within 15 ms
tmax      = 1e6
MAX_ATOMS = 65536
