import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import load_and_format_sharks_gals
from matplotlib.colors import LogNorm

from  nessie  import  FlatCosmology, RedshiftCatalog
from  nessie.helper_funcs  import  create_density_function


REGION = 'wide'
gals, groups = load_and_format_sharks_gals(
    "/Users/sp624AA/Downloads/group_finding_mocks/galaxies_shark.parquet",
    region=REGION
)

gals_masked = gals[gals['masked'] == False]

ra, dec, redshifts = gals['ra'], gals['dec'], gals['redshift_observed']
ra_masked, dec_masked, redshifts_masked = gals_masked['ra'], gals_masked['dec'], gals_masked['redshift_observed']
cosmo = FlatCosmology(h = 0.7, omega_matter = 0.3)


if REGION == 'wide':
    running_density_all = create_density_function(redshifts, total_counts = len(redshifts), survey_fractional_area = 0.02748648583, cosmology = cosmo)
    runnning_density_masked = create_density_function(redshifts_masked, total_counts = len(redshifts_masked), survey_fractional_area = 0.02565146777, cosmology = cosmo)

area_ratio = 0.02748648583 / 0.02565146777
# Running group catalog
red_cat_all = RedshiftCatalog(ra, dec, redshifts, running_density_all, cosmo)
red_cat_masked = RedshiftCatalog(ra_masked, dec_masked, redshifts_masked, runnning_density_masked, cosmo)

red_cat_all.set_completeness()
red_cat_masked.set_completeness()

red_cat_all.run_fof(b0 = 0.05, r0 = 32)
red_cat_masked.run_fof(b0 = 0.05, r0 = 32)

#red_cat_all.group_ids = gals['id_fof'].values
#red_cat_masked.group_ids = gals_masked['id_fof'].values

group_cat_all = red_cat_all.calculate_group_table(gals['mag_abs_Z_VISTA'], velocity_errors = np.zeros_like(gals['mag_abs_Z_VISTA']))
group_cat_masked = red_cat_masked.calculate_group_table(gals_masked['mag_abs_Z_VISTA'], velocity_errors = np.zeros_like(gals_masked['mag_abs_Z_VISTA']))


# ============================================================
# Inputs
# ============================================================
r50_all = np.asarray(group_cat_all["r50"], dtype=float)
r50_masked = np.asarray(group_cat_masked["r50"], dtype=float)

vd_all = np.asarray(group_cat_all["velocity_dispersion_gap"], dtype=float)
vd_masked = np.asarray(group_cat_masked["velocity_dispersion_gap"], dtype=float)

# ============================================================
# Cleaning
# ============================================================
def clean_finite(x):
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]

r50_all = clean_finite(r50_all)
r50_masked = clean_finite(r50_masked)
vd_all = clean_finite(vd_all)
vd_masked = clean_finite(vd_masked)

# Optional: if you want to exclude non-physical values, uncomment these
# r50_all = r50_all[r50_all > 0]
# r50_masked = r50_masked[r50_masked > 0]
# vd_all = vd_all[vd_all > 0]
# vd_masked = vd_masked[vd_masked > 0]

# ============================================================
# Bin helpers
# ============================================================
def make_linear_bins(x1, x2, nbins=30):
    xmin = min(np.min(x1), np.min(x2))
    xmax = max(np.max(x1), np.max(x2))
    if xmin == xmax:
        xmin -= 0.5
        xmax += 0.5
    return np.linspace(xmin, xmax, nbins + 1)

r50_bins = make_linear_bins(r50_all, r50_masked, nbins=30)
vd_bins = make_linear_bins(vd_all, vd_masked, nbins=30)

# ============================================================
# Histogram helpers
# ============================================================
def fractional_residual(h_masked, h_all):
    resid = np.full_like(h_all, np.nan, dtype=float)
    m = h_all > 0
    resid[m] = (h_masked[m] - h_all[m]) / h_all[m]
    return resid

# ============================================================
# 1) LINEAR-X VERSION: raw counts, log y-axis
# ============================================================
r50_h_all, r50_edges = np.histogram(r50_all, bins=r50_bins)
r50_h_masked, _ = np.histogram(r50_masked, bins=r50_bins)

vd_h_all, vd_edges = np.histogram(vd_all, bins=vd_bins)
vd_h_masked, _ = np.histogram(vd_masked, bins=vd_bins)

r50_centres = 0.5 * (r50_edges[:-1] + r50_edges[1:])
vd_centres = 0.5 * (vd_edges[:-1] + vd_edges[1:])

r50_resid = fractional_residual(r50_h_masked, r50_h_all)
vd_resid = fractional_residual(vd_h_masked, vd_h_all)

fig1, axs = plt.subplots(
    4, 1,
    figsize=(10, 11),
    sharex=False,
    gridspec_kw={"height_ratios": [3, 1, 3, 1]},
    constrained_layout=True,
)

ax1, ax2, ax3, ax4 = axs

# R50 counts
ax1.hist(r50_all, bins=r50_bins, histtype="step", linewidth=2, label="All")
ax1.hist(r50_masked, bins=r50_bins, histtype="step", linewidth=2, label="Masked")
ax1.set_yscale("log")
ax1.set_ylabel("Count")
ax1.set_title("Nessie Groups R50 distribution")
ax1.legend()
ax1.grid(alpha=0.3)

# R50 residual
ax2.axhline(0, color="k", linestyle="--", linewidth=1)
ax2.plot(r50_centres, r50_resid, drawstyle="steps-mid", linewidth=1.8)
ax2.set_ylabel("Frac.\nresid")
ax2.set_xlabel("R50 [Mpc]")
ax2.grid(alpha=0.3)

# Velocity dispersion counts
ax3.hist(vd_all, bins=vd_bins, histtype="step", linewidth=2, label="All")
ax3.hist(vd_masked, bins=vd_bins, histtype="step", linewidth=2, label="Masked")
ax3.set_yscale("log")
ax3.set_ylabel("Count")
ax3.set_title("Nessie Groups Velocity dispersion distribution")
ax3.legend()
ax3.grid(alpha=0.3)

# Velocity dispersion residual
ax4.axhline(0, color="k", linestyle="--", linewidth=1)
ax4.plot(vd_centres, vd_resid, drawstyle="steps-mid", linewidth=1.8)
ax4.set_ylabel("Frac.\nresid")
ax4.set_xlabel("$\sigma_{GAP}$ [km/s^2]")
ax4.grid(alpha=0.3)

fig1.savefig("../plots/nessie_group_property_histograms_log_counts.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# 2) LINEAR-X VERSION: raw counts corrected by area ratio
# ============================================================
r50_masked_weights = np.full_like(r50_masked, area_ratio, dtype=float)
vd_masked_weights = np.full_like(vd_masked, area_ratio, dtype=float)

r50_h_all_w, r50_edges_w = np.histogram(r50_all, bins=r50_bins)
r50_h_masked_w, _ = np.histogram(
    r50_masked,
    bins=r50_bins,
    weights=r50_masked_weights,
)

vd_h_all_w, vd_edges_w = np.histogram(vd_all, bins=vd_bins)
vd_h_masked_w, _ = np.histogram(
    vd_masked,
    bins=vd_bins,
    weights=vd_masked_weights,
)

r50_centres_w = 0.5 * (r50_edges_w[:-1] + r50_edges_w[1:])
vd_centres_w = 0.5 * (vd_edges_w[:-1] + vd_edges_w[1:])

r50_resid_w = fractional_residual(r50_h_masked_w, r50_h_all_w)
vd_resid_w = fractional_residual(vd_h_masked_w, vd_h_all_w)

fig2, axs = plt.subplots(
    4, 1,
    figsize=(10, 11),
    sharex=False,
    gridspec_kw={"height_ratios": [3, 1, 3, 1]},
    constrained_layout=True,
)

ax1, ax2, ax3, ax4 = axs

# R50 weighted counts
ax1.hist(
    r50_all,
    bins=r50_bins,
    histtype="step",
    linewidth=2,
    label="All",
)
ax1.hist(
    r50_masked,
    bins=r50_bins,
    weights=r50_masked_weights,
    histtype="step",
    linewidth=2,
    label=f"Masked × {area_ratio:.3f}",
)
ax1.set_yscale("log")
ax1.set_ylabel("Weighted count")
ax1.set_title("Nessie Groups R50 distribution, area corrected")
ax1.legend()
ax1.grid(alpha=0.3)

# R50 residual
ax2.axhline(0, color="k", linestyle="--", linewidth=1)
ax2.plot(r50_centres_w, r50_resid_w, drawstyle="steps-mid", linewidth=1.8)
ax2.set_ylabel("Frac.\nresid")
ax2.set_xlabel("R50 [Mpc]")
ax2.grid(alpha=0.3)

# Velocity dispersion weighted counts
ax3.hist(
    vd_all,
    bins=vd_bins,
    histtype="step",
    linewidth=2,
    label="All",
)
ax3.hist(
    vd_masked,
    bins=vd_bins,
    weights=vd_masked_weights,
    histtype="step",
    linewidth=2,
    label=f"Masked × {area_ratio:.3f}",
)
ax3.set_yscale("log")
ax3.set_ylabel("Weighted count")
ax3.set_title("Nessie groups Velocity dispersion distribution, area corrected")
ax3.legend()
ax3.grid(alpha=0.3)

# Velocity dispersion residual
ax4.axhline(0, color="k", linestyle="--", linewidth=1)
ax4.plot(vd_centres_w, vd_resid_w, drawstyle="steps-mid", linewidth=1.8)
ax4.set_ylabel("Frac.\nresid")
ax4.set_xlabel("$\sigma_{GAP}$ [km/s^2]")
ax4.grid(alpha=0.3)

fig2.savefig(
    "../plots/nessie_group_property_histograms_log_counts_area_corrected.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()