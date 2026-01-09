import json
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import json
from data.pc_sequence import PC_Sequence
from losses.objectives import chamfer_distance
from matplotlib import pyplot as plt
import numpy as np

# Load dataset config
with open("/home/adumouli/Bureau/garment-diffusion/configs/config.json") as f:
    config = json.load(f)

# files = [
#             "/home/adumouli/Data/experiments/fit_sue-cos-walk/results",
#             "/home/adumouli/Data/experiments/fit_sue-cos-run/results"
#          ]
files = [
            "/home/adumouli/Data/experiments/fit_sue-cos-walk/opti_material",
            "/home/adumouli/Data/experiments/fit_sue-cos-run/opti_material"
         ]

agregated_chamfer_s2t = []
for i, f in enumerate(files):
    print(os.path.join(os.path.dirname(f), "capture"))
    capture_seq = PC_Sequence(os.path.join(os.path.dirname(f), "capture"))
    cloth_seq = PC_Sequence(os.path.join(f, "cloth_meshes"))
    for pc, mesh in zip(capture_seq.meshes, cloth_seq.meshes):
        target = pc.verts_packed()[None, ...]
        v = mesh.verts_packed()[None, ...]
        _, _, _, _, chamx, chamy = chamfer_distance(v, target)
        agregated_chamfer_s2t.append(chamx)

agregated_chamfer_s2t = torch.stack(agregated_chamfer_s2t).flatten().cpu()*100
max_v = agregated_chamfer_s2t.max().item()
mean_v = agregated_chamfer_s2t.mean().item()
print(agregated_chamfer_s2t.shape, mean_v, max_v)


# Compute histogram
counts, bins = np.histogram(agregated_chamfer_s2t.cpu(), bins=1000)
cdf = np.cumsum(counts) / np.sum(counts)
x_values = np.vstack((bins, np.roll(bins, -1))).T.flatten()[:-2]
y_values = (np.vstack((cdf, cdf)).T.flatten()*100)
threshold_90 = np.interp(0.9, cdf, bins[:-1])  # interpolate to find x at y=0.9

# Plot
fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
ax.plot(
    x_values, y_values,
    color='black', linewidth=2.2,
    solid_capstyle='round', zorder=3
)

ax.fill_between(x_values, 0, y_values, color='black', alpha=0.12, zorder=2)
ax.axvline(x=threshold_90, color='gray', linestyle='--', linewidth=1.0, alpha=0.6, zorder=4)
ax.text(threshold_90, 1, f'{threshold_90:.2f} cm', color='gray', fontsize=9,
        ha='right', va='bottom', rotation=90, alpha=0.8)
ax.set_xlim([0.0, max_v])
ax.set_ylim([0, 100])
ax.set_xlabel('Surface to target distance (cm)', fontsize=10, fontweight='medium')
ax.set_ylabel('Vertices below threshold %', fontsize=10, fontweight='medium')
ax.tick_params(left=False, bottom=False, labelleft=True, labelbottom=True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)
ax.xaxis.grid(False)

plt.tight_layout()
plt.savefig("../plots/cumulative_plot.png")
plt.show()
