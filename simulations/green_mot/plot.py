"""Static plots of MOT simulation results."""
import pickle
import glob
import numpy as np
import matplotlib.pyplot as plt
import constants

all_results = []
for file_path in glob.glob("mot_simulation_data.pkl"):
    with open(file_path, 'rb') as f:
        all_results.extend(pickle.load(f))

print(f"Successfully loaded data for {len(all_results)} atoms!")

unit_to_mm = (1 / constants.kmag_real) * 1000

# --- 2D scatter of final positions ---
final_x, final_y = [], []
for t, r, v in all_results:
    final_x.append(r[0, -1] * unit_to_mm)
    final_y.append(r[1, -1] * unit_to_mm)

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(final_x, final_y, color='royalblue', s=12, alpha=0.4)
ax.scatter([0], [0], color='red', marker='+', s=200, label='Trap Center')
ax.set_xlabel('X position (mm)')
ax.set_ylabel('Y position (mm)')
ax.set_title('Strontium-88 MOT Cloud (Real Scale)')
ax.set_aspect('equal')
ax.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.savefig('mot_cloud_2d_xy.png', dpi=300, bbox_inches='tight')

# --- 3x2 trajectory grid ---
fig, ax = plt.subplots(3, 2, figsize=(6.25, 5.5))

for t, r, v in all_results:
    for ii in range(3):
        ax[ii, 0].plot(t / 1e3, v[ii], linewidth=0.25, color='blue', alpha=0.3)
        ax[ii, 1].plot(t / 1e3, r[ii] * constants.alpha, linewidth=0.25, color='red', alpha=0.3)

for ax_i in ax[-1, :]:
    ax_i.set_xlabel(r'$10^3 \Gamma t$')

for jj in range(2):
    for ax_i in ax[jj, :]:
        ax_i.set_xticklabels([])

for ax_i, lbl in zip(ax[:, 0], ['x', 'y', 'z']):
    ax_i.set_ylabel(r'$v_' + lbl + r'/(\Gamma/k)$')

for ax_i, lbl in zip(ax[:, 1], ['x', 'y', 'z']):
    ax_i.set_ylabel(r'$\alpha ' + lbl + '$')

fig.subplots_adjust(left=0.1, bottom=0.08, wspace=0.3)
plt.savefig('mot_3x2_trajectories.png', dpi=300, bbox_inches='tight')