"""Static plots of blue MOT simulation results."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from plotting import load_results, plot_final_positions, plot_trajectories
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
