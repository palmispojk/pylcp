"""Static plots of green MOT simulation results."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from plotting import load_results, plot_final_positions, plot_trajectories
import constants

results = load_results('mot_simulation_data.pkl')
print(f"Loaded {len(results)} atoms")

plot_final_positions(
    results, constants.kmag_real,
    title='Sr88 Green MOT Cloud',
    filename='mot_cloud_2d_xy.png',
)

plot_trajectories(
    results, constants.alpha,
    filename='mot_3x2_trajectories.png',
)
