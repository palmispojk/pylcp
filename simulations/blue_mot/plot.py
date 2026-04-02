"""Static plots of blue MOT simulation results."""
import matplotlib
matplotlib.use('Agg')
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from plotting import load_results, plot_final_positions, plot_trajectories, plot_distributions
from analysis import make_units, classify_captured, fit_distributions
import constants

results = load_results('blue_mot_simulation_data.pkl')
print(f"Loaded {len(results)} atoms")

plot_final_positions(
    results, constants.kmag_real,
    title='Sr88 Blue MOT Cloud',
    filename='blue_mot_cloud_2d_xy.png',
)

# Also show x-z plane to see Zeeman slower beam capture
plot_final_positions(
    results, constants.kmag_real,
    title='Sr88 Blue MOT Cloud (beam axis)',
    filename='blue_mot_cloud_2d_xz.png',
    axes='xz',
)

plot_trajectories(
    results, constants.alpha_nat,
    filename='blue_mot_3x2_trajectories.png',
)

# Distribution fits for captured atoms
units = make_units(constants.kmag_real, constants.gamma_real, constants.mass_real)
mask = classify_captured(results)
dist_fits = fit_distributions(results, units, mask=mask)
plot_distributions(
    dist_fits,
    title='Sr88 Blue MOT — Captured Atom Distributions',
    filename='blue_mot_distributions.png',
)
