import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from MDAnalysis.analysis import align

print("Loading trajectory...")

# Load protein-only topology and trajectory
u = mda.Universe("reference.pdb", "MD_fit.xtc")

# Select C-alpha atoms
protein = u.select_atoms("name CA")

n_res = len(protein)
resids = protein.resids  # Real residue numbers

print(f"Number of C-alpha atoms: {n_res}")

# Align trajectory using C-alpha atoms
print("Aligning trajectory...")
align.AlignTraj(u, u, select="name CA", in_memory=True).run()

# Extract coordinates
print("Extracting coordinates...")
coords = np.array([protein.positions.copy() for ts in u.trajectory])

# Remove mean structure
mean_coords = coords.mean(axis=0)
fluctuations = coords - mean_coords

print("Calculating DCCM...")

corr_matrix = np.zeros((n_res, n_res))

for i in range(n_res):
    for j in range(n_res):
        numerator = np.mean(
            np.sum(fluctuations[:, i, :] *
                   fluctuations[:, j, :], axis=1)
        )

        denom_i = np.mean(np.sum(fluctuations[:, i, :]**2, axis=1))
        denom_j = np.mean(np.sum(fluctuations[:, j, :]**2, axis=1))

        if denom_i > 0 and denom_j > 0:
            corr_matrix[i, j] = numerator / np.sqrt(denom_i * denom_j)
        else:
            corr_matrix[i, j] = 0.0

# Remove possible NaN values
corr_matrix = np.nan_to_num(corr_matrix)

print("Plotting DCCM...")

plt.figure(figsize=(10, 8))

sns.heatmap(
    corr_matrix,
    cmap="RdBu_r",
    center=0,
    vmin=-1,
    vmax=1,
    cbar_kws={"label": "Correlation Coefficient"}
)

plt.title("Dynamic Cross-Correlation Matrix (Cα atoms)")
plt.xlabel("Residue Number")
plt.ylabel("Residue Number")

# Reduce tick density for readability
tick_positions = np.linspace(0, n_res - 1, 10, dtype=int)
tick_labels = resids[tick_positions]

plt.xticks(tick_positions, tick_labels, rotation=45)
plt.yticks(tick_positions, tick_labels)

plt.tight_layout()
plt.savefig("DCCM_high_quality.png", dpi=600)
plt.show()

print("DCCM calculation completed successfully.")
