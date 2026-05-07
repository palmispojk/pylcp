"""Static plots of broadband (BB) red MOT simulation results."""
import matplotlib
matplotlib.use('Agg')
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from plotting import (load_results, plot_final_positions, plot_trajectories,
                      plot_distributions, plot_temperature_vs_time)
from analysis import (make_units, classify_captured, fit_distributions,
                      temperature_vs_time, doppler_temperature)
import constants

results = load_results('bb_red_mot_simulation_data.pkl')
print(f"Loaded {len(results)} atoms")

plot_final_positions(
    results, constants.kmag_real,
    title='Sr88 BB Red MOT Cloud',
    filename='bb_red_mot_cloud_2d_xy.png',
)
plot_final_positions(
    results, constants.kmag_real,
    title='Sr88 BB Red MOT Cloud (beam axis)',
    filename='bb_red_mot_cloud_2d_xz.png',
    axes='xz',
)

plot_trajectories(
    results, constants.alpha_nat,
    filename='bb_red_mot_3x2_trajectories.png',
)

units = make_units(constants.kmag_real, constants.gamma_real, constants.mass_real)
mask = classify_captured(results)
dist_fits = fit_distributions(results, units, mask=mask)
plot_distributions(
    dist_fits,
    title='Sr88 BB Red MOT — Captured Atom Distributions',
    filename='bb_red_mot_distributions.png',
)

T_data = temperature_vs_time(results, units, mask=mask)
plot_temperature_vs_time(
    T_data,
    title='Sr88 BB Red MOT — Temperature vs Time',
    filename='bb_red_mot_temperature.png',
    target_T=doppler_temperature(constants.gamma_real),
)
