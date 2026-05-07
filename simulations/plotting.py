"""
Shared plotting utilities for MOT simulations.

All functions accept a list of result dicts as produced by the simulation
scripts (keys: 't', 'r', 'v', 'success', 't_random', 'n_random').

Usage from any simulation folder::

    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from plotting import load_results, plot_final_positions, plot_trajectories, animate_3d
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt


def load_results(path):
    """Load simulation results from a pickle file.

    Returns a list of dicts with keys 't', 'r', 'v', etc.
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def plot_final_positions(results, kmag_real, title='MOT Cloud',
                         filename='mot_cloud_2d_xy.png', axes='xy'):
    """2D scatter of final atom positions.

    Args:
        results: list of result dicts (must contain 'r').
        kmag_real: real wavevector magnitude (1/m), used for unit conversion.
        title: plot title.
        filename: output file path.  None to skip saving.
        axes: which two axes to plot — 'xy', 'xz', or 'yz'.
    """
    ax_map = {'x': 0, 'y': 1, 'z': 2}
    i0, i1 = ax_map[axes[0]], ax_map[axes[1]]

    unit_to_mm = 1e3 / kmag_real
    pos0 = np.array([res['r'][i0, -1] * unit_to_mm for res in results])
    pos1 = np.array([res['r'][i1, -1] * unit_to_mm for res in results])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(pos0, pos1, color='royalblue', s=12, alpha=0.4)
    ax.scatter([0], [0], color='red', marker='+', s=200, label='Trap Center')
    ax.set_xlabel(f'{axes[0].upper()} position (mm)')
    ax.set_ylabel(f'{axes[1].upper()} position (mm)')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
    if filename:
        fig.savefig(filename, dpi=300, bbox_inches='tight')
    return fig, ax


def plot_trajectories(results, alpha_nat, time_scale=1e3,
                      title=None, filename='mot_trajectories.png'):
    """3x2 grid of velocity (left) and position (right) vs time.

    Args:
        results: list of result dicts (must contain 't', 'r', 'v').
        alpha_nat: magnetic field gradient in natural units, used to
            scale position as alpha*r (gives displacement in units of
            the Zeeman shift).  Pass 1.0 to plot raw natural-unit position.
        time_scale: divide time axis by this value.  Default 1e3 gives
            units of 10^3 / gamma.
        title: optional suptitle.
        filename: output file path.  None to skip saving.
    """
    fig, ax = plt.subplots(3, 2, figsize=(6.25, 5.5))

    for res in results:
        t = res['t'] / time_scale
        for ii in range(3):
            ax[ii, 0].plot(t, res['v'][ii], linewidth=0.25, color='blue', alpha=0.3)
            ax[ii, 1].plot(t, res['r'][ii] * alpha_nat, linewidth=0.25, color='red', alpha=0.3)

    for ax_i in ax[-1, :]:
        ax_i.set_xlabel(r'$10^3 \Gamma t$')

    for jj in range(2):
        for ax_i in ax[jj, :]:
            ax_i.set_xticklabels([])

    for ax_i, lbl in zip(ax[:, 0], ['x', 'y', 'z']):
        ax_i.set_ylabel(r'$v_' + lbl + r'/(\Gamma/k)$')

    for ax_i, lbl in zip(ax[:, 1], ['x', 'y', 'z']):
        ax_i.set_ylabel(r'$\alpha ' + lbl + '$')

    if title:
        fig.suptitle(title)
    fig.subplots_adjust(left=0.1, bottom=0.08, wspace=0.3)
    if filename:
        fig.savefig(filename, dpi=300, bbox_inches='tight')
    return fig, ax


def plot_distributions(dist_fits, title='MOT Atom Distributions',
                       filename='mot_distributions.png'):
    """Plot histograms with Gaussian fits for position and velocity per axis.

    Args:
        dist_fits: dict returned by ``analysis.fit_distributions``.
        title: overall figure title.
        filename: output file path.  None to skip saving.

    Returns:
        (fig, axes) tuple.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    labels = ['x', 'y', 'z']

    for col, label in enumerate(labels):
        for row, kind in enumerate(['velocity', 'position']):
            ax = axes[row, col]
            info = dist_fits[kind][label]

            ax.hist(info['data'], bins=info['bin_edges'], density=True,
                    alpha=0.55, color='steelblue', edgecolor='white',
                    linewidth=0.5, label='Data')
            ax.plot(info['fit_x'], info['fit_pdf'], 'r-', linewidth=2,
                    label='Gaussian fit')

            mu = info['mean']
            std = info['std']
            unit = info['unit_label']
            ax.set_xlabel(f'{label} ({unit})')
            ax.set_ylabel('Probability density')

            kind_label = 'Position' if kind == 'position' else 'Velocity'
            ax.set_title(f'{kind_label} {label}')

            textstr = (f'$\\mu = {mu:.3f}$ {unit}\n'
                       f'$\\sigma = {std:.3f}$ {unit}\n'
                       f'$N = {info["n_atoms"]}$')
            ax.text(0.97, 0.95, textstr, transform=ax.transAxes,
                    fontsize=9, verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            ax.legend(fontsize=8, loc='upper left')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    if filename:
        fig.savefig(filename, dpi=300, bbox_inches='tight')
    return fig, axes


def plot_temperature_vs_time(T_data, title='MOT Temperature vs Time',
                             filename='mot_temperature.png', yscale='log',
                             target_T=None, target_label='Doppler limit'):
    """Plot per-axis and mean temperature over time.

    Args:
        T_data: dict returned by ``analysis.temperature_vs_time``.
        title: figure title.
        filename: output file path. None to skip saving.
        yscale: 'log' or 'linear'.
        target_T: optional reference temperature in Kelvin (e.g. the Doppler
            limit). Drawn as a dashed horizontal line.
        target_label: legend label for the reference line.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    t_ms = T_data['t_ms']

    for axis, color in zip(['x', 'y', 'z'],
                           ['tab:blue', 'tab:orange', 'tab:green']):
        ax.plot(t_ms, T_data[f'T_{axis}'] * 1e6, linewidth=1.0,
                color=color, alpha=0.55, label=f'$T_{axis}$')
    ax.plot(t_ms, T_data['T_mean'] * 1e6, linewidth=2.0,
            color='black', label=r'$\overline{T}$')

    if target_T is not None:
        ax.axhline(target_T * 1e6, linestyle='--', color='red', alpha=0.6,
                   label=f'{target_label} ({target_T*1e6:.1f} μK)')

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Temperature (μK)')
    ax.set_yscale(yscale)
    ax.set_title(title)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='best')

    if filename:
        fig.savefig(filename, dpi=300, bbox_inches='tight')
    return fig, ax


def animate_3d(results, kmag_real, filename='mot_capture.gif',
               num_frames=100, fps=20, lim_mm=2.0, title_prefix='MOT Capture'):
    """Animated 3D GIF of atom positions over time.

    Args:
        results: list of result dicts (must contain 'r').
        kmag_real: real wavevector magnitude (1/m).
        filename: output GIF path.
        num_frames: number of animation frames.
        fps: frames per second.
        lim_mm: axis limits in mm (symmetric).
        title_prefix: prefix for the frame title.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter

    unit_to_mm = 1e3 / kmag_real
    n_steps = results[0]['r'].shape[1]
    t_indices = np.linspace(0, n_steps - 1, num_frames, dtype=int)

    # Pre-extract positions for speed
    all_r = np.array([res['r'] for res in results])  # (N, 3, n_steps)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    scat = ax.scatter([], [], [], s=10, color='royalblue', alpha=0.7)

    ax.set_xlim(-lim_mm, lim_mm)
    ax.set_ylim(-lim_mm, lim_mm)
    ax.set_zlim(-lim_mm, lim_mm)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')

    def update(frame):
        ti = t_indices[frame]
        x = all_r[:, 0, ti] * unit_to_mm
        y = all_r[:, 1, ti] * unit_to_mm
        z = all_r[:, 2, ti] * unit_to_mm
        scat._offsets3d = (x, y, z)
        ax.set_title(f'{title_prefix}: frame {frame}/{num_frames}')
        return scat,

    ani = FuncAnimation(fig, update, frames=num_frames, interval=1000 // fps, blit=False)
    writer = PillowWriter(fps=fps)
    ani.save(filename, writer=writer)
    return ani
