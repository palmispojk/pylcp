"""Animated 3D GIF of MOT capture."""
import pickle
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import constants

all_results = []
for file_path in glob.glob("mot_simulation_data.pkl"):
    with open(file_path, 'rb') as f:
        all_results.extend(pickle.load(f))

print(f"Successfully loaded data for {len(all_results)} atoms!")

unit_to_mm = (1 / constants.kmag_real) * 1000

# Use the number of time steps from the first trajectory
n_steps = all_results[0][1].shape[1]
num_frames = 100
t_indices = np.linspace(0, n_steps - 1, num_frames, dtype=int)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
scat = ax.scatter([], [], [], s=10, color='royalblue', alpha=0.7)

ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-2, 2)
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')


def update(frame):
    ti = t_indices[frame]
    x, y, z = [], [], []
    for _, r, _ in all_results:
        x.append(r[0, ti] * unit_to_mm)
        y.append(r[1, ti] * unit_to_mm)
        z.append(r[2, ti] * unit_to_mm)
    scat._offsets3d = (x, y, z)
    ax.set_title(f'MOT Capture: Time Index {ti}')
    return scat,


ani = FuncAnimation(fig, update, frames=num_frames, interval=50, blit=False)
writer = PillowWriter(fps=20)
ani.save("mot_capture_animation.gif", writer=writer)