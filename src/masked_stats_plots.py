import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# gals, groups = load_and_format_sharks_gals(
#     "/Users/sp624AA/Downloads/group_finding_mocks/galaxies_shark.parquet",
#     region="wide"
# )
gals = pd.read_parquet(
    "/Users/sp624AA/Downloads/group_finding_mocks/galaxies_filtered_wide_mock_fixed_masking.parquet"
)

# -------------------------------------
# Config
# -------------------------------------
group_col = "id_fof"
N_col = "n_group_fof"
masked_col = "masked"
mass_col = "stellar_mass"

# -------------------------------------
# Clean data
# -------------------------------------
g = gals.copy()

# remove galaxies without valid group
g = g[g[group_col] != -1].copy()

# ensure masked is boolean
g[masked_col] = g[masked_col].astype(bool)

# -------------------------------------
# Group statistics
# -------------------------------------
group_stats = (
    g.groupby(group_col)
    .agg(
        N=(N_col, "first"),
        n_masked=(masked_col, "sum"),
        n_total=(masked_col, "size"),
        stellar_mass_total=(mass_col, "sum"),
        stellar_mass_masked=(
            mass_col,
            lambda x: x[g.loc[x.index, masked_col]].sum()
        ),
    )
)

group_stats["masked_fraction"] = group_stats["n_masked"] / group_stats["n_total"]
group_stats["missing_mass_fraction"] = (
    group_stats["stellar_mass_masked"] / group_stats["stellar_mass_total"]
)

# -------------------------------------
# Fraction of groups with masked members
# (including fully masked groups)
# -------------------------------------
has_mask = group_stats["n_masked"] > 0
frac_masked_groups = has_mask.mean()

print(f"Fraction of groups with ≥1 masked galaxy: {frac_masked_groups:.3f}")

# -------------------------------------
# Panel 2 statistics:
# 1) ratio of sums
# 2) average per-group masked fraction
# 3) fraction of groups with >=1 masked member
# -------------------------------------
masked_sum_vs_N = group_stats.groupby("N")["n_masked"].sum()
members_sum_vs_N = group_stats.groupby("N")["n_total"].sum()
ratio_of_sums_vs_N = masked_sum_vs_N / members_sum_vs_N

mean_frac_vs_N = group_stats.groupby("N")["masked_fraction"].mean()

pct_groups_with_mask_vs_N = (
    group_stats.assign(has_mask=(group_stats["n_masked"] > 0).astype(float))
    .groupby("N")["has_mask"]
    .mean()
)

pct_groups_fully_masked_vs_N = (
    group_stats.assign(all_masked=(group_stats["n_masked"] == group_stats["n_total"]).astype(float))
    .groupby("N")["all_masked"]
    .mean()
)

# -------------------------------------
# Subset for groups with >=1 masked member
# and not fully masked
# (plots 3 and 4 unchanged)
# -------------------------------------
subset = group_stats[group_stats["n_masked"] > 0].copy()
subset = subset[subset["n_masked"] < subset["n_total"]].copy()

# -------------------------------------
# Richness distributions for top panel
# -------------------------------------
n_groups_vs_N = group_stats.groupby("N").size()
n_masked_groups_vs_N = group_stats[group_stats["n_masked"] > 0].groupby("N").size()

N_vals = np.arange(1, int(group_stats["N"].max()) + 1)
n_groups_plot = n_groups_vs_N.reindex(N_vals, fill_value=0)
n_masked_plot = n_masked_groups_vs_N.reindex(N_vals, fill_value=0)

# -------------------------------------
# Arrays for panels 3 and 4
# -------------------------------------
x_members = subset["N"].to_numpy().astype(int)
y_members_pct = subset["masked_fraction"].to_numpy() * 100.0

x_mass = subset["N"].to_numpy().astype(int)
y_mass_pct = subset["missing_mass_fraction"].to_numpy() * 100.0

# -------------------------------------
# Figure
# -------------------------------------
fig, (ax0, ax1, ax2, ax3) = plt.subplots(
    nrows=4,
    sharex=True,
    figsize=(10, 12),
    gridspec_kw={"height_ratios": [1, 1, 1.6, 1.6], "hspace": 0.18}
)

# -------------------------------------
# Panel 1: number of groups vs richness r"$N_{\rm groups}(N_{\rm masked}>0) \,/\, N_{\rm groups}$"
# -------------------------------------
ax0.plot(
    N_vals,
    n_groups_plot.values,
    marker="o",
    ms=3,
    lw=1,
    label=r"$N_{\rm groups}$"
)
ax0.plot(
    N_vals,
    n_masked_plot.values,
    marker="o",
    ms=3,
    lw=1,
    label=r"$N_{\rm groups}(N_{\rm masked}>0)$"
)

ax0.set_ylabel("Number of groups")
ax0.set_yscale("log")
ax0.set_title("Masked Galaxy Groups in WAVESwide Mock", fontsize=14)
ax0.grid(alpha=0.3)
ax0.legend(frameon=False, fontsize=9)

# -------------------------------------
# Panel 2: % masked vs richness
# -------------------------------------
ax1.plot(
    ratio_of_sums_vs_N.index,
    ratio_of_sums_vs_N.values * 100.0,
    marker="o",
    ms=3,
    lw=1,
    label=r"$\sum N_{\rm masked} \,/\, \sum N_{\rm members}$"
)
ax1.plot(
    mean_frac_vs_N.index,
    mean_frac_vs_N.values * 100.0,
    marker="o",
    ms=3,
    lw=1,
    label=r"$\left\langle N_{\rm masked} \,/\, N_{\rm members} \right\rangle$"
)
ax1.plot(
    pct_groups_with_mask_vs_N.index,
    pct_groups_with_mask_vs_N.values * 100.0,
    marker="o",
    ms=3,
    lw=1,
    label=r"$N_{\rm groups}(N_{\rm masked}>0) \,/\, N_{\rm groups}$"
)

ax1.plot(
    pct_groups_fully_masked_vs_N.index,
    pct_groups_fully_masked_vs_N.values * 100.0,
    marker="o",
    ms=3,
    lw=1,
    label=r"$N_{\rm groups}(N_{\rm masked}=N_{\rm members}) \,/\, N_{\rm groups}$"
)

ax1.set_ylabel("% masked")
ax1.grid(alpha=0.3)
ax1.legend(frameon=False, fontsize=9)
ax1.set_ylim(0, 75)

# -------------------------------------
# Heatmap config
# -------------------------------------
xmin, xmax = 2, 50
Ns_use = np.arange(xmin, xmax + 1)

# -------------------------------------
# Precompute member-heatmap counts
# -------------------------------------
member_bins = {}
for N in Ns_use:
    possible = 100.0 * np.arange(1, N) / N
    if len(possible) == 0:
        continue

    vals = y_members_pct[x_members == N]
    if len(vals) == 0:
        counts = np.zeros(len(possible), dtype=int)
    else:
        k_vals = np.rint(vals * N / 100.0).astype(int)
        k_vals = np.clip(k_vals, 1, N - 1)
        counts = np.bincount(k_vals, minlength=N)[1:N]

    yedges = np.linspace(0, 100, N)

    member_bins[N] = {
        "counts": counts,
        "yedges": yedges,
    }

# -------------------------------------
# Precompute mass-heatmap counts
# -------------------------------------
xbins_mass = np.arange(xmin - 0.5, xmax + 1.5, 1.0)
ybins_mass = np.linspace(0, 100, 15)

H_mass, _, _ = np.histogram2d(
    x_mass,
    y_mass_pct,
    bins=[xbins_mass, ybins_mass]
)

# -------------------------------------
# Shared colour normalisation across both heatmaps
# -------------------------------------
all_positive_counts = []

for N in member_bins:
    c = member_bins[N]["counts"]
    all_positive_counts.extend(c[c > 0])

all_positive_counts.extend(H_mass[H_mass > 0].ravel())

if len(all_positive_counts) > 0:
    shared_norm = LogNorm(vmin=1, vmax=np.max(all_positive_counts))
else:
    shared_norm = LogNorm(vmin=1, vmax=1)

# -------------------------------------
# Panel 3: adaptive-binned counts heatmap
#          for % members masked
# -------------------------------------
mesh_members = None

for N in Ns_use:
    if N not in member_bins:
        continue

    counts = member_bins[N]["counts"]
    yedges = member_bins[N]["yedges"]

    Xe, Ye = np.meshgrid([N - 0.5, N + 0.5], yedges)
    C = counts[:, None]

    mesh_members = ax2.pcolormesh(
        Xe,
        Ye,
        C,
        cmap="viridis",
        norm=shared_norm,
        shading="flat"
    )

ax2.set_ylabel("% members missing\n(incomplete groups)", labelpad=10)
ax2.set_ylim(0, 100)
ax2.grid(alpha=0.3)

# -------------------------------------
# Panel 4: counts heatmap for
#          % stellar mass missing
# -------------------------------------
h3 = ax3.hist2d(
    x_mass,
    y_mass_pct,
    bins=[xbins_mass, ybins_mass],
    cmap="viridis",
    density=False,
    norm=shared_norm
)

ax3.set_ylabel("% missing $M_{\\mathrm{stellar}}$\n(incomplete groups)", labelpad=10)
ax3.set_xlabel("Group richness $N$")
ax3.set_ylim(0, 100)
ax3.grid(alpha=0.3)

# -------------------------------------
# One shared colourbar for both heatmaps
# -------------------------------------
cbar = fig.colorbar(
    h3[3],
    ax=[ax2, ax3],
    orientation="horizontal",
    pad=0.12,
    fraction=0.08
)
cbar.set_label("Counts per bin")

# -------------------------------------
# Shared x limits
# -------------------------------------
ax3.set_xlim(xmin, xmax)

plt.savefig("../plots/masked_galaxy_stats.png", dpi=300, bbox_inches="tight")
plt.show()