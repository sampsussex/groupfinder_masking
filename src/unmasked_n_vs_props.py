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
from matplotlib.colors import LogNorm

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
# ------------------------------------------------------------
g_all = g_all[g_all[group_id_col] != -1].copy()
g_masked = g_masked[g_masked[group_id_col] != -1].copy()

# Safety in case one group appears more than once
g_all = g_all.drop_duplicates(subset=group_id_col)
g_masked = g_masked.drop_duplicates(subset=group_id_col)

# ------------------------------------------------------------
# 3) Match common groups
# ------------------------------------------------------------
merged = g_all.merge(
    g_masked,
    on=group_id_col,
    how="inner",
    suffixes=("_all", "_masked")
)

# ------------------------------------------------------------
# 4) Compute property changes as log-ratios
# ------------------------------------------------------------
merged["delta_multiplicity"] = (
    merged[f"{mult_col}_masked"] - merged[f"{mult_col}_all"]
)

merged["log_vd_ratio"] = np.log10(
    merged[f"{vd_col}_masked"] / merged[f"{vd_col}_all"]
)

merged["log_r50_ratio"] = np.log10(
    merged[f"{r50_col}_masked"] / merged[f"{r50_col}_all"]
)

merged["log_mass_ratio"] = np.log10(
    merged[f"{mass_col}_masked"] / merged[f"{mass_col}_all"]
)

# ------------------------------------------------------------
# 5) Filter:
#    - only groups with changed multiplicity
#    - only unmasked multiplicity <= 25
# ------------------------------------------------------------
plot_df = merged[
    (merged["delta_multiplicity"] != 0) &
    (merged[f"{mult_col}_all"] <= 25)
].copy()

plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna(
    subset=[
        f"{mult_col}_all",
        "log_vd_ratio",
        "log_r50_ratio",
        "log_mass_ratio",
    ]
)

print(f"Number of matched groups plotted: {len(plot_df)}")

# ------------------------------------------------------------
# 6) Prepare x bins: integer-centred multiplicity bins
# ------------------------------------------------------------
x_bins = np.arange(1.5, 26.5, 1)  # bins centred on N=2,...,25

# ------------------------------------------------------------
# 7) Prepare y bins
# ------------------------------------------------------------
def make_linear_bins(arr, nbins=50, pad_frac=0.03):
    arr = np.asarray(arr)
    lo = np.nanmin(arr)
    hi = np.nanmax(arr)

    if lo == hi:
        delta = 1.0 if lo == 0 else 0.05 * abs(lo)
        lo -= delta
        hi += delta
    else:
        pad = pad_frac * (hi - lo)
        lo -= pad
        hi += pad

    return np.linspace(lo, hi, nbins + 1)

vd_bins = make_linear_bins(plot_df["log_vd_ratio"], nbins=50)
r50_bins = make_linear_bins(plot_df["log_r50_ratio"], nbins=50)
mass_bins = make_linear_bins(plot_df["log_mass_ratio"], nbins=50)

# ------------------------------------------------------------
# 8) Plot
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

# Panel 1: log velocity-dispersion ratio vs unmasked N
h1 = axes[0].hist2d(
    plot_df[f"{mult_col}_all"],
    plot_df["log_vd_ratio"],
    bins=[x_bins, vd_bins],
    # norm=LogNorm(),
    cmap="viridis",
)
axes[0].axhline(0, color="k", ls="--", lw=1)
axes[0].set_xlabel("Unmasked group multiplicity")
axes[0].set_ylabel(r"$\log_{10}(\sigma_{\rm gap,masked}/\sigma_{\rm gap,all})$")
axes[0].set_title(r"$\log_{10}(\sigma_{\rm gap,masked}/\sigma_{\rm gap,all})$ vs unmasked group $N$")
axes[0].set_xlim(1.5, 25.5)
axes[0].grid(alpha=0.25)

cbar1 = fig.colorbar(h1[3], ax=axes[0], pad=0.02)
cbar1.set_label("Groups per bin")

# Panel 2: log r50 ratio vs unmasked N
h2 = axes[1].hist2d(
    plot_df[f"{mult_col}_all"],
    plot_df["log_r50_ratio"],
    bins=[x_bins, r50_bins],
#    norm=LogNorm(),
    cmap="viridis",
)
axes[1].axhline(0, color="k", ls="--", lw=1)
axes[1].set_xlabel("Unmasked group multiplicity")
axes[1].set_ylabel(r"$\log_{10}(r_{50,\rm masked}/r_{50,\rm all})$")
axes[1].set_title(r"$\log_{10}(r_{50,\rm masked}/r_{50,\rm all})$ vs unmasked group $N$")
axes[1].set_xlim(1.5, 25.5)
axes[1].grid(alpha=0.25)

cbar2 = fig.colorbar(h2[3], ax=axes[1], pad=0.02)
cbar2.set_label("Groups per bin")

# Panel 3: log mass ratio vs unmasked N
h3 = axes[2].hist2d(
    plot_df[f"{mult_col}_all"],
    plot_df["log_mass_ratio"],
    bins=[x_bins, mass_bins],
#    norm=LogNorm(),
    cmap="viridis",
)
axes[2].axhline(0, color="k", ls="--", lw=1)
axes[2].set_xlabel("Unmasked group multiplicity")
axes[2].set_ylabel(r"$\log_{10}(M_{\rm dyn,masked}/M_{\rm dyn,all})$")
axes[2].set_title(r"$\log_{10}(M_{\rm dyn,masked}/M_{\rm dyn,all})$ vs unmasked group $N$")
axes[2].set_xlim(1.5, 25.5)
axes[2].grid(alpha=0.25)

cbar3 = fig.colorbar(h3[3], ax=axes[2], pad=0.02)
cbar3.set_label("Groups per bin")

plt.savefig("../plots/unmasked_n_vs_props.png", dpi=300, bbox_inches="tight")
plt.show()