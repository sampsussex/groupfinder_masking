import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy import cosmology
from utils import load_and_format_sharks_gals
from nessie import FlatCosmology, RedshiftCatalog
from nessie.helper_funcs import create_density_function
from matplotlib.colors import LogNorm

# ============================================================
# Setup
# ============================================================
astropycosmo = cosmology.FlatLambdaCDM(H0=70, Om0=0.3)
REGION = 'wide'
AREA_ALL, AREA_MASKED = 0.02748648583, 0.02565146777
area_ratio = AREA_ALL / AREA_MASKED
area_ratio = area_ratio # Just for testing.

IDS = 'id_fof'  # 'id_fof' or 'Nessie'

N_BOOTSTRAP = 1000
RNG = np.random.default_rng(seed=42)

# gals, groups = load_and_format_sharks_gals(
#     "/Users/sp624AA/Downloads/group_finding_mocks/galaxies_shark.parquet",
#     region=REGION
# )
gals = pd.read_parquet("/Users/sp624AA/Downloads/group_finding_mocks/galaxies_filtered_wide_mock_fixed_masking.parquet")
gals_masked = gals[gals['masked'] == False]

cosmo = FlatCosmology(h=0.7, omega_matter=0.3)
ra, dec, z = gals['ra'], gals['dec'], gals['redshift_observed']
ra_m, dec_m, z_m = gals_masked['ra'], gals_masked['dec'], gals_masked['redshift_observed']

density_all    = create_density_function(z,   total_counts=len(z),   survey_fractional_area=AREA_ALL,    cosmology=cosmo)
density_masked = create_density_function(z_m, total_counts=len(z_m), survey_fractional_area=AREA_MASKED, cosmology=cosmo)

# ============================================================
# Run group finder
# ============================================================
def run_catalog(ra, dec, z, density, mag, cosmo, gals_df, id_type=IDS):
    cat = RedshiftCatalog(ra, dec, z, density, cosmo)
    cat.set_completeness()

    if id_type == 'Nessie':
        cat.run_fof(b0=0.05, r0=32)
    elif id_type == 'id_fof':
        cat.group_ids = gals_df[id_type].values
    else:
        raise ValueError(f"Unknown id_type: {id_type}")

    return pd.DataFrame(cat.calculate_group_table(mag, velocity_errors=np.zeros_like(mag)))


group_cat_all    = run_catalog(ra,   dec,   z,   density_all,    gals['mag_abs_Z_VISTA'],        cosmo, gals)
group_cat_masked = run_catalog(ra_m, dec_m, z_m, density_masked, gals_masked['mag_abs_Z_VISTA'], cosmo, gals_masked)

A, G, = 10, 4.302e-9

for cat in (group_cat_all, group_cat_masked):
    cat['dynamical_mass'] = A * cat['velocity_dispersion_gap']**2 * cat['r50'] * 2 / G

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# ------------------------------------------------------------
# Config: column names
# ------------------------------------------------------------
group_id_col = "group_id"
mult_col = "multiplicity"
vd_col = "velocity_dispersion_gap"
r50_col = "r50"
mass_col = "dynamical_mass"

# ------------------------------------------------------------
# 1) Keep only the columns we need
# ------------------------------------------------------------
cols_needed = [group_id_col, mult_col, vd_col, r50_col, mass_col]

g_all = group_cat_all[cols_needed].copy()
g_masked = group_cat_masked[cols_needed].copy()

# ------------------------------------------------------------
# 2) Remove invalid / isolated groups
#    id_fof == -1 means isolated galaxies, not true matched groups
# ------------------------------------------------------------
g_all = g_all[g_all[group_id_col] != -1].copy()
g_masked = g_masked[g_masked[group_id_col] != -1].copy()

# Optional safety in case there is more than one row per group id
g_all = g_all.drop_duplicates(subset=group_id_col)
g_masked = g_masked.drop_duplicates(subset=group_id_col)

# ------------------------------------------------------------
# 3) Inner join to keep only groups common to both catalogs
# ------------------------------------------------------------
merged = g_all.merge(
    g_masked,
    on=group_id_col,
    how="inner",
    suffixes=("_all", "_masked")
)

# ------------------------------------------------------------
# 4) Compute changes
#    Using masked relative to unmasked
# ------------------------------------------------------------
merged["delta_vd"] = merged[f"{vd_col}_masked"] - merged[f"{vd_col}_all"]
merged["delta_r50"] = merged[f"{r50_col}_masked"] - merged[f"{r50_col}_all"]

merged["log_mass_ratio"] = np.log10(
    merged[f"{mass_col}_masked"] / merged[f"{mass_col}_all"]
)

merged["delta_multiplicity"] = (
    merged[f"{mult_col}_masked"] - merged[f"{mult_col}_all"]
)

merged["pct_change_multiplicity"] = (
    100.0 * merged["delta_multiplicity"] / merged[f"{mult_col}_all"]
)

# ------------------------------------------------------------
# 5) Filter only groups where multiplicity changed
# ------------------------------------------------------------
plot_df = merged[merged["delta_multiplicity"] != 0].copy()

# Keep only finite values for all plotted quantities
plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna(
    subset=[
        "pct_change_multiplicity",
        "delta_vd",
        "delta_r50",
        "log_mass_ratio",
        f"{mult_col}_all",
    ]
)

print(f"Number of matched groups with multiplicity changes: {len(plot_df)}")

# ------------------------------------------------------------
# 6) Plot
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, constrained_layout=True)

cvals = plot_df[f"{mult_col}_all"].values

norm = LogNorm(
    vmin=max(1, np.nanmin(cvals)),   # log scale cannot include 0
    vmax=np.nanmax(cvals)
)

scatter_kwargs = dict(
    c=cvals,
    cmap="viridis",
    norm=norm,
    s=10,
    alpha=0.3,
    edgecolors="none",
)

# Panel 1: delta velocity dispersion vs percentage membership change
sc = axes[0].scatter(
    plot_df["pct_change_multiplicity"],
    plot_df["delta_vd"],
    **scatter_kwargs
)
axes[0].axhline(0, color="k", ls="--", lw=1)
axes[0].axvline(0, color="k", ls="--", lw=1)
axes[0].set_xlabel(r"% change in multiplicity")
axes[0].set_ylabel(r"$\Delta$ velocity_dispersion_gap")
axes[0].set_title(r"$\Delta \sigma_{\rm gap}$ vs %$\Delta N$")
axes[0].grid(alpha=0.3)

# Panel 2: delta r50 vs percentage membership change
axes[1].scatter(
    plot_df["pct_change_multiplicity"],
    plot_df["delta_r50"],
    **scatter_kwargs
)
axes[1].axhline(0, color="k", ls="--", lw=1)
axes[1].axvline(0, color="k", ls="--", lw=1)
axes[1].set_xlabel(r"% change in multiplicity")
axes[1].set_ylabel(r"$\Delta r_{50}$")
axes[1].set_title(r"$\Delta r_{50}$ vs %$\Delta N$")
axes[1].grid(alpha=0.3)

# Panel 3: log mass ratio vs percentage membership change
axes[2].scatter(
    plot_df["pct_change_multiplicity"],
    plot_df["log_mass_ratio"],
    **scatter_kwargs
)
axes[2].axhline(0, color="k", ls="--", lw=1)
axes[2].axvline(0, color="k", ls="--", lw=1)
axes[2].set_xlabel(r"% change in multiplicity")
axes[2].set_ylabel(r"$\log_{10}(M_{{\rm dyn, masked}} / M_{{\rm dyn, all}})$")
axes[2].set_title(r"$\log_{10}(M_{\rm dyn, masked}/M_{\rm dyn, all})$ vs %$\Delta N$")
axes[2].grid(alpha=0.3)

# Shared colourbar
cbar = fig.colorbar(sc, ax=axes, pad=0.02)
cbar.set_label("Unmasked group multiplicity")

plt.savefig(f"../plots/percent_delta_vd_r50_mass_vs_delta_mult.png", dpi=300, bbox_inches="tight")
plt.show()