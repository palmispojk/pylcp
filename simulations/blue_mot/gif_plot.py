"""Animated 3D GIF of blue MOT capture."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from plotting import load_results, animate_3d
import constants

results = load_results('blue_mot_simulation_data.pkl')
print(f"Loaded {len(results)} atoms")

animate_3d(
    results, constants.kmag_real,
    filename='blue_mot_capture_animation.gif',
    title_prefix='Sr88 Blue MOT',
)
