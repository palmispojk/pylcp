"""
Shared analysis utilities for MOT simulations.

Extracts physical quantities (temperature, cloud size, capture fraction,
phase-space density, etc.) from simulation result pickles.

All functions accept a list of result dicts as produced by the simulation
scripts (keys: 't', 'r', 'v', 'success', 't_random', 'n_random') and a
unit-conversion dict.

Usage from any simulation folder::

    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from analysis import load_results, make_units, cloud_summary
    import constants

    results = load_results('mot_simulation_data.pkl')
    units = make_units(constants.kmag_real, constants.gamma_real, constants.mass_real)
    print(cloud_summary(results, units))
"""
import pickle
import numpy as np
import scipy.constants as const


def load_results(path):
    """Load simulation results from a pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def make_units(kmag_real, gamma_real, mass_real):
    """Build a unit-conversion dict from physical constants.

    Args:
        kmag_real: Wavevector magnitude (1/m).
        gamma_real: Natural linewidth (rad/s).
        mass_real: Atom mass (kg).

    Returns:
        dict with conversion factors from natural units to SI.
    """
    v_nat_to_si = gamma_real / kmag_real          # (m/s) per natural velocity unit
    r_nat_to_si = 1.0 / kmag_real                 # (m) per natural position unit
    t_nat_to_si = 1.0 / gamma_real                # (s) per natural time unit

    return {
        'kmag_real': kmag_real,
        'gamma_real': gamma_real,
        'mass_real': mass_real,
        'v_to_si': v_nat_to_si,          # multiply natural v to get m/s
        'r_to_si': r_nat_to_si,          # multiply natural r to get m
        'r_to_mm': r_nat_to_si * 1e3,    # multiply natural r to get mm
        'r_to_um': r_nat_to_si * 1e6,    # multiply natural r to get um
        't_to_si': t_nat_to_si,          # multiply natural t to get s
        't_to_us': t_nat_to_si * 1e6,    # multiply natural t to get us
    }


# ---------------------------------------------------------------------------
#  Position and velocity extraction
# ---------------------------------------------------------------------------

def final_positions(results, units=None):
    """Return final positions of all atoms as (N, 3) array.

    Args:
        results: list of result dicts.
        units: if provided, positions are converted to metres.

    Returns:
        np.ndarray of shape (N, 3).
    """
    r = np.array([res['r'][:, -1] for res in results])
    if units is not None:
        r *= units['r_to_si']
    return r


def final_velocities(results, units=None):
    """Return final velocities of all atoms as (N, 3) array.

    Args:
        results: list of result dicts.
        units: if provided, velocities are converted to m/s.

    Returns:
        np.ndarray of shape (N, 3).
    """
    v = np.array([res['v'][:, -1] for res in results])
    if units is not None:
        v *= units['v_to_si']
    return v


def initial_positions(results, units=None):
    """Return initial positions of all atoms as (N, 3) array."""
    r = np.array([res['r'][:, 0] for res in results])
    if units is not None:
        r *= units['r_to_si']
    return r


def initial_velocities(results, units=None):
    """Return initial velocities of all atoms as (N, 3) array."""
    v = np.array([res['v'][:, 0] for res in results])
    if units is not None:
        v *= units['v_to_si']
    return v


# ---------------------------------------------------------------------------
#  Capture classification
# ---------------------------------------------------------------------------

def classify_captured(results, r_thresh=5000, v_thresh=0.5):
    """Return a boolean mask of captured atoms.

    An atom is "captured" if its final distance from the origin is below
    ``r_thresh`` AND its final speed is below ``v_thresh`` (both in
    natural units).

    Returns:
        np.ndarray of bool, shape (N,).
    """
    final_r = np.array([res['r'][:, -1] for res in results])
    final_v = np.array([res['v'][:, -1] for res in results])
    dist = np.sqrt(np.sum(final_r**2, axis=1))
    speed = np.sqrt(np.sum(final_v**2, axis=1))
    return (dist < r_thresh) & (speed < v_thresh)


def capture_fraction(results, **kwargs):
    """Return the fraction of atoms captured."""
    mask = classify_captured(results, **kwargs)
    return mask.sum() / len(results)


# ---------------------------------------------------------------------------
#  Temperature
# ---------------------------------------------------------------------------

def temperature(results, units, mask=None):
    """Compute temperature from velocity spread of (optionally masked) atoms.

    Uses the equipartition theorem:  T = m * <v_i^2> / k_B  per axis,
    where <v_i^2> is the variance (mean subtracted) along axis i.

    Args:
        results: list of result dicts.
        units: conversion dict from ``make_units``.
        mask: boolean array to select a subset (e.g. captured atoms only).

    Returns:
        dict with keys 'T_x', 'T_y', 'T_z', 'T_mean' (all in Kelvin),
        and 'v_rms_x', 'v_rms_y', 'v_rms_z' (in m/s).
    """
    v = final_velocities(results, units)
    if mask is not None:
        v = v[mask]

    if len(v) == 0:
        return {k: np.nan for k in
                ['T_x', 'T_y', 'T_z', 'T_mean',
                 'v_rms_x', 'v_rms_y', 'v_rms_z']}

    m = units['mass_real']
    out = {}
    temps = []
    for i, label in enumerate(['x', 'y', 'z']):
        var = np.var(v[:, i])  # variance in (m/s)^2
        T = m * var / const.k
        out[f'T_{label}'] = T
        out[f'v_rms_{label}'] = np.sqrt(var)
        temps.append(T)
    out['T_mean'] = np.mean(temps)
    return out


def doppler_temperature(gamma_real):
    """Theoretical Doppler temperature T_D = hbar * gamma / (2 * k_B)."""
    return const.hbar * gamma_real / (2 * const.k)


# ---------------------------------------------------------------------------
#  Cloud size
# ---------------------------------------------------------------------------

def cloud_size(results, units, mask=None):
    """Compute 1-sigma cloud size along each axis.

    Args:
        results: list of result dicts.
        units: conversion dict from ``make_units``.
        mask: boolean array to select a subset.

    Returns:
        dict with 'sigma_x', 'sigma_y', 'sigma_z' (in metres),
        'sigma_x_mm', etc. (in mm), and 'center_x', 'center_y',
        'center_z' (in metres).
    """
    r = final_positions(results, units)
    if mask is not None:
        r = r[mask]

    if len(r) == 0:
        return {k: np.nan for k in
                ['sigma_x', 'sigma_y', 'sigma_z',
                 'sigma_x_mm', 'sigma_y_mm', 'sigma_z_mm',
                 'center_x', 'center_y', 'center_z']}

    out = {}
    for i, label in enumerate(['x', 'y', 'z']):
        out[f'center_{label}'] = np.mean(r[:, i])
        sigma = np.std(r[:, i])
        out[f'sigma_{label}'] = sigma
        out[f'sigma_{label}_mm'] = sigma * 1e3
    return out


# ---------------------------------------------------------------------------
#  Scattering rate
# ---------------------------------------------------------------------------

def scattering_rate(results, units, mask=None):
    """Compute mean scattering rate from scatter event counts.

    Args:
        results: list of result dicts (must contain 't_random', 'n_random').
        units: conversion dict from ``make_units``.
        mask: boolean array to select a subset.

    Returns:
        dict with 'mean_rate' (scatters/s), 'total_scatters',
        'mean_scatters_per_atom'.
    """
    subset = results
    if mask is not None:
        subset = [r for r, m in zip(results, mask) if m]

    if len(subset) == 0:
        return {'mean_rate': np.nan, 'total_scatters': 0,
                'mean_scatters_per_atom': np.nan}

    total = 0
    total_time = 0.0
    for res in subset:
        n = np.sum(res['n_random'])
        t_span = res['t'][-1] - res['t'][0]
        total += n
        total_time += t_span

    mean_per_atom = total / len(subset)
    mean_time = total_time / len(subset) * units['t_to_si']
    rate = mean_per_atom / mean_time if mean_time > 0 else np.nan

    return {
        'mean_rate': rate,
        'total_scatters': int(total),
        'mean_scatters_per_atom': mean_per_atom,
    }


# ---------------------------------------------------------------------------
#  Summary
# ---------------------------------------------------------------------------

def cloud_summary(results, units, r_thresh=5000, v_thresh=0.5):
    """Print a full summary of the simulation results.

    Returns the summary as a formatted string.
    """
    N = len(results)
    mask = classify_captured(results, r_thresh=r_thresh, v_thresh=v_thresh)
    n_cap = mask.sum()
    frac = n_cap / N

    temp = temperature(results, units, mask=mask)
    T_D = doppler_temperature(units['gamma_real'])
    size = cloud_size(results, units, mask=mask)
    scat = scattering_rate(results, units, mask=mask)

    lines = [
        f"=== MOT Simulation Summary ===",
        f"  Total atoms:       {N}",
        f"  Captured:          {n_cap} ({100*frac:.1f}%)",
        f"  Lost:              {N - n_cap} ({100*(1-frac):.1f}%)",
        f"",
        f"--- Temperature (captured atoms) ---",
        f"  T_x = {temp['T_x']*1e6:.1f} uK    (v_rms = {temp['v_rms_x']*100:.2f} cm/s)",
        f"  T_y = {temp['T_y']*1e6:.1f} uK    (v_rms = {temp['v_rms_y']*100:.2f} cm/s)",
        f"  T_z = {temp['T_z']*1e6:.1f} uK    (v_rms = {temp['v_rms_z']*100:.2f} cm/s)",
        f"  T_mean = {temp['T_mean']*1e6:.1f} uK",
        f"  T_Doppler = {T_D*1e6:.1f} uK   (ratio T/T_D = {temp['T_mean']/T_D:.2f})",
        f"",
        f"--- Cloud size (captured atoms, 1-sigma) ---",
        f"  sigma_x = {size['sigma_x_mm']:.3f} mm",
        f"  sigma_y = {size['sigma_y_mm']:.3f} mm",
        f"  sigma_z = {size['sigma_z_mm']:.3f} mm",
        f"  center  = ({size['center_x']*1e3:.3f}, {size['center_y']*1e3:.3f}, {size['center_z']*1e3:.3f}) mm",
        f"",
        f"--- Scattering ---",
        f"  Mean rate:          {scat['mean_rate']:.2e} /s",
        f"  Mean per atom:      {scat['mean_scatters_per_atom']:.0f} scatters",
    ]
    return '\n'.join(lines)
