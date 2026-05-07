"""
Shared analysis utilities for MOT simulations.

Extracts physical quantities (temperature, cloud size, capture fraction,
phase-space density, etc.) from simulation result pickles.

CLI usage::

    python analysis.py <pkl_path> <constants.py>

Library usage::

    from analysis import analyze
    results, units, summary = analyze('data.pkl', kmag_real, gamma_real, mass_real)
"""
import os
import pickle
import numpy as np
import scipy.constants as const
from scipy.stats import norm


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

def classify_captured(results, r_thresh=10000, v_thresh=0.5):
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
#  Phase-space density
# ---------------------------------------------------------------------------

def phase_space_density(results, units, mask=None):
    """Compute peak phase-space density of the cloud.

    PSD = n_peak * lambda_dB^3, where n_peak is the peak density
    (assuming Gaussian cloud) and lambda_dB is the thermal de Broglie
    wavelength.

    Returns:
        dict with 'psd', 'lambda_dB' (m), 'n_peak' (1/m^3).
    """
    temp_info = temperature(results, units, mask=mask)
    size_info = cloud_size(results, units, mask=mask)
    n_cap = mask.sum() if mask is not None else len(results)

    T = temp_info['T_mean']
    m = units['mass_real']

    if T <= 0 or n_cap == 0:
        return {'psd': np.nan, 'lambda_dB': np.nan, 'n_peak': np.nan}

    # Thermal de Broglie wavelength
    lambda_dB = np.sqrt(2 * np.pi * const.hbar**2 / (m * const.k * T))

    # Peak density of a 3D Gaussian: N / ((2*pi)^(3/2) * sigma_x * sigma_y * sigma_z)
    sx = size_info['sigma_x']
    sy = size_info['sigma_y']
    sz = size_info['sigma_z']
    n_peak = n_cap / ((2 * np.pi)**1.5 * sx * sy * sz)

    psd = n_peak * lambda_dB**3

    return {'psd': psd, 'lambda_dB': lambda_dB, 'n_peak': n_peak}


# ---------------------------------------------------------------------------
#  Capture velocity
# ---------------------------------------------------------------------------

def capture_velocity(results, units, mask=None):
    """Estimate the capture velocity of the MOT.

    Finds the maximum initial speed among atoms that ended up captured.

    Args:
        results: list of result dicts.
        units: conversion dict from ``make_units``.
        mask: boolean array from ``classify_captured``.  If None, all atoms
            are classified using the default thresholds.

    Returns:
        dict with 'v_capture_si' (m/s), 'v_capture_nat' (natural units),
        'v_capture_95' (95th percentile of captured atoms' initial speeds, m/s).
    """
    if mask is None:
        mask = classify_captured(results)
    v0 = np.array([res['v'][:, 0] for res in results])
    speed0 = np.sqrt(np.sum(v0**2, axis=1))

    cap_speeds = speed0[mask]
    if len(cap_speeds) == 0:
        return {'v_capture_si': np.nan, 'v_capture_nat': np.nan,
                'v_capture_95': np.nan}

    v_max = np.max(cap_speeds)
    v_95 = np.percentile(cap_speeds, 95)

    return {
        'v_capture_si': v_max * units['v_to_si'],
        'v_capture_nat': v_max,
        'v_capture_95': v_95 * units['v_to_si'],
    }


# ---------------------------------------------------------------------------
#  Equilibration time
# ---------------------------------------------------------------------------

def equilibration_time(results, units, mask=None, frac=0.90):
    """Estimate the time for the cloud to reach steady state.

    Computes the RMS speed of (optionally masked) atoms at each saved
    time step.  The equilibration time is defined as the first time the
    RMS speed drops below ``frac`` of the way from initial to final value.

    Returns:
        dict with 't_eq_nat' (natural units), 't_eq_si' (seconds),
        't_eq_ms' (milliseconds).
    """
    subset = results
    if mask is not None:
        subset = [r for r, m in zip(results, mask) if m]

    if len(subset) == 0:
        return {'t_eq_nat': np.nan, 't_eq_si': np.nan, 't_eq_ms': np.nan}

    # Use the time grid from the first atom (all share the same grid)
    t = subset[0]['t']
    n_steps = len(t)

    # Compute mean RMS speed at each time step across atoms
    rms_speed = np.zeros(n_steps)
    for res in subset:
        v = res['v']  # shape (3, n_steps)
        rms_speed += np.sqrt(np.sum(v**2, axis=0))
    rms_speed /= len(subset)

    v_init = rms_speed[0]
    v_final = rms_speed[-1]
    threshold = v_final + (1.0 - frac) * (v_init - v_final)

    # Find first crossing
    below = np.where(rms_speed <= threshold)[0]
    if len(below) == 0:
        t_eq = t[-1]
    else:
        t_eq = t[below[0]]

    return {
        't_eq_nat': t_eq,
        't_eq_si': t_eq * units['t_to_si'],
        't_eq_ms': t_eq * units['t_to_si'] * 1e3,
    }


# ---------------------------------------------------------------------------
#  Atom number vs time (loading curve)
# ---------------------------------------------------------------------------

def loading_curve(results, units, r_thresh=None, v_thresh=None):
    """Compute the number of atoms within the capture region at each time step.

    Uses the same default thresholds as ``classify_captured`` when not
    specified.

    Returns:
        dict with 't' (natural units), 't_ms' (milliseconds),
        'n_captured' (atom count at each time step).
    """
    defaults = classify_captured.__defaults__  # (r_thresh, v_thresh)
    if r_thresh is None:
        r_thresh = defaults[0]
    if v_thresh is None:
        v_thresh = defaults[1]

    t = results[0]['t']
    n_steps = len(t)
    n_captured = np.zeros(n_steps, dtype=int)

    for res in results:
        r = res['r']  # (3, n_steps)
        v = res['v']  # (3, n_steps)
        dist = np.sqrt(np.sum(r**2, axis=0))
        speed = np.sqrt(np.sum(v**2, axis=0))
        captured = (dist < r_thresh) & (speed < v_thresh)
        n_captured += captured.astype(int)

    return {
        't': t,
        't_ms': t * units['t_to_si'] * 1e3,
        'n_captured': n_captured,
    }


# ---------------------------------------------------------------------------
#  Distribution fitting
# ---------------------------------------------------------------------------

def fit_distributions(results, units, mask=None, n_bins=60):
    """Fit normal distributions to final positions and velocities per axis.

    Args:
        results: list of result dicts.
        units: conversion dict from ``make_units``.
        mask: boolean array to select a subset (e.g. captured atoms only).
        n_bins: number of histogram bins.

    Returns:
        dict with keys 'position' and 'velocity', each containing per-axis
        dicts with keys 'data', 'mean', 'std', 'bin_edges', 'bin_centers',
        'counts', 'fit_pdf', 'unit_label'.
    """
    r = final_positions(results, units)
    v = final_velocities(results, units)
    if mask is not None:
        r = r[mask]
        v = v[mask]

    out = {'position': {}, 'velocity': {}}
    labels = ['x', 'y', 'z']

    for i, label in enumerate(labels):
        for kind, arr, unit_label, scale in [
            ('position', r[:, i] * 1e3, 'mm', 1.0),
            ('velocity', v[:, i] * 1e2, 'cm/s', 1.0),
        ]:
            data = arr * scale
            mu, std = norm.fit(data)
            counts, bin_edges = np.histogram(data, bins=n_bins, density=True)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            fit_x = np.linspace(bin_edges[0], bin_edges[-1], 300)
            fit_pdf = norm.pdf(fit_x, mu, std)

            out[kind][label] = {
                'data': data,
                'mean': mu,
                'std': std,
                'bin_edges': bin_edges,
                'bin_centers': bin_centers,
                'counts': counts,
                'fit_x': fit_x,
                'fit_pdf': fit_pdf,
                'unit_label': unit_label,
                'n_atoms': len(data),
            }

    return out


# ---------------------------------------------------------------------------
#  Summary
# ---------------------------------------------------------------------------

def cloud_summary(results, units):
    """Summary of cloud size and temperature (captured atoms).

    Returns the summary as a formatted string.
    """
    N = len(results)
    mask = classify_captured(results)
    n_cap = int(mask.sum())

    temp = temperature(results, units, mask=mask)
    T_D = doppler_temperature(units['gamma_real'])
    size = cloud_size(results, units, mask=mask)

    lines = [
        f"=== MOT Simulation Summary ===",
        f"  Total atoms:  {N}   Captured: {n_cap} ({100*n_cap/N:.1f}%)",
        f"",
        f"--- Temperature (captured atoms) ---",
        f"  T_x = {temp['T_x']*1e6:.1f} uK    (v_rms = {temp['v_rms_x']*100:.2f} cm/s)",
        f"  T_y = {temp['T_y']*1e6:.1f} uK    (v_rms = {temp['v_rms_y']*100:.2f} cm/s)",
        f"  T_z = {temp['T_z']*1e6:.1f} uK    (v_rms = {temp['v_rms_z']*100:.2f} cm/s)",
        f"  T_mean   = {temp['T_mean']*1e6:.1f} uK",
        f"  T_Doppler = {T_D*1e6:.1f} uK   (ratio T/T_D = {temp['T_mean']/T_D:.2f})",
        f"",
        f"--- Cloud size (captured atoms, 1-sigma) ---",
        f"  sigma_x = {size['sigma_x_mm']:.3f} mm",
        f"  sigma_y = {size['sigma_y_mm']:.3f} mm",
        f"  sigma_z = {size['sigma_z_mm']:.3f} mm",
        f"  center  = ({size['center_x']*1e3:.3f}, "
        f"{size['center_y']*1e3:.3f}, {size['center_z']*1e3:.3f}) mm",
    ]
    return '\n'.join(lines)


def analyze(pkl_path, kmag_real, gamma_real, mass_real, log=True):
    """Load results, print summary, and optionally save to a text file.

    Args:
        pkl_path: Path to the simulation pickle file.
        kmag_real, gamma_real, mass_real: Physical constants for unit conversion.
        log: If True, write summary to ``<pkl_stem>_analysis.txt``.

    Returns:
        tuple of (results, units, summary_string).
    """
    results = load_results(pkl_path)
    units = make_units(kmag_real, gamma_real, mass_real)

    header = f"Loaded {pkl_path} ({len(results)} atoms)\n"
    summary = header + cloud_summary(results, units)
    print(summary)

    if log:
        log_path = os.path.splitext(pkl_path)[0] + '_analysis.txt'
        with open(log_path, 'w') as f:
            f.write(summary + '\n')
        print(f"\nSaved to {log_path}")

    return results, units, summary


if __name__ == '__main__':
    import sys
    import importlib.util

    if len(sys.argv) < 3:
        print("Usage: python analysis.py <pkl_path> <constants.py>")
        sys.exit(1)

    pkl_path = sys.argv[1]
    constants_path = sys.argv[2]

    if not os.path.exists(pkl_path):
        print(f"Error: {pkl_path} not found")
        sys.exit(1)
    if not os.path.exists(constants_path):
        print(f"Error: {constants_path} not found")
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("constants", constants_path)
    constants = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(constants)

    analyze(pkl_path, constants.kmag_real, constants.gamma_real, constants.mass_real)
