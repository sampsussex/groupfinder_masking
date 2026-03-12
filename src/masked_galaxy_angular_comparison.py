#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print('loading groups')
from utils import load_and_format_sharks_gals
from nessie import FlatCosmology, RedshiftCatalog
from nessie.helper_funcs import create_density_function

#gals, groups = load_and_format_sharks_gals(
#    "/Users/sp624AA/Downloads/group_finding_mocks/galaxies_shark.parquet",
#    region='wide'
#)
gals = pd.read_parquet("/Users/sp624AA/Downloads/group_finding_mocks/galaxies_filtered_wide_mock_fixed_masking.parquet")


AREA_ALL = 0.02748648583
cosmo = FlatCosmology(h=0.7, omega_matter=0.3)
ra, dec, z = gals['ra'], gals['dec'], gals['redshift_observed']
mag = gals['mag_abs_Z_VISTA']
density = create_density_function(z,   total_counts=len(z),   survey_fractional_area=AREA_ALL,    cosmology=cosmo)
cat = RedshiftCatalog(ra, dec, z, density, cosmo)
cat.group_ids = gals['id_fof'].values

groups = pd.DataFrame(cat.calculate_group_table(mag, velocity_errors=np.zeros_like(mag)))

from astropy.cosmology import FlatLambdaCDM
astropycosmo = FlatLambdaCDM(H0=70, Om0=0.3)
# find comoving distance for each group
groups['distance_groups'] = astropycosmo.comoving_distance(groups['iter_redshift']).to('Mpc').value
groups['id_fof'] = groups['group_id'].astype("Int64")
print(groups.columns)
print('groups and stars loaded')

gals['id_fof'] = gals['id_fof'].astype("Int64")
# ============================================================
# User settings
# ============================================================

# ---- groups columns ----
GROUP_ID_COL = "id_fof"
GROUP_MULT_COL = "multiplicity"
GROUP_R100_COL = "r100"                  # Mpc
GROUP_DIST_COL = "distance_groups"       # same distance units as r100
GROUP_RA_COL = "iter_ra"            # degrees
GROUP_DEC_COL = "iter_dec"          # degrees

# ---- galaxy columns ----
GAL_GROUP_ID_COL = "id_fof"
GAL_RA_COL = "ra"                        # degrees
GAL_DEC_COL = "dec"                      # degrees
GAL_MASKED_COL = "masked"                # boolean

# ---- output ----
SAVE_PATH = "../plots/stacked_masked_galaxy_density_by_group_multiplicity.png"
DPI = 300

# ---- heatmap settings ----
NBINS = 13
XRANGE = (-1., 1.)
YRANGE = (-1., 1.)
NORMALIZE_TO_DENSITY = False
FIGSIZE = (16, 8)
CMAP = "magma"

# Requested bins had overlap at 6-7 and 7-10
# so using a non-overlapping interpretation for 8 panels
BIN_DEFS = [
    ("N = 3-5", 3, 5),
#    ("N = 4", 4, 4),
#    ("N = 5", 5, 5),
    ("N = 6–10", 6, 10),
    ("N = 11-20", 11, 20),
    ("N = 20-50", 20, 50),
#    ("N = 16–25", 16, 25),
#    ("N = 26–50", 26, 50),
]


# ============================================================
# Utilities
# ============================================================

def wrapped_delta_ra_deg(ra_deg, ra0_deg):
    """
    Smallest signed RA difference in degrees.
    """
    return (ra_deg - ra0_deg + 180.0) % 360.0 - 180.0


def prepare_group_table(groups):
    cols = [
        GROUP_ID_COL,
        GROUP_MULT_COL,
        GROUP_R100_COL,
        GROUP_DIST_COL,
        GROUP_RA_COL,
        GROUP_DEC_COL,
    ]
    g = groups[cols].copy()
    g = g.replace([np.inf, -np.inf], np.nan).dropna()

    g = g[
        (g[GROUP_ID_COL] != -1) &
        (g[GROUP_R100_COL] > 0) &
        (g[GROUP_DIST_COL] > 0)
    ].copy()

    # one row per group
    g = g.drop_duplicates(subset=[GROUP_ID_COL]).copy()
    return g


def prepare_galaxy_table(gals):
    cols = [
        GAL_GROUP_ID_COL,
        GAL_RA_COL,
        GAL_DEC_COL,
        GAL_MASKED_COL,
    ]
    g = gals[cols].copy()
    g = g.replace([np.inf, -np.inf], np.nan).dropna()

    g = g[g[GAL_GROUP_ID_COL] != -1].copy()
    g = g[g[GAL_MASKED_COL].astype(bool)].copy()
    return g


def collect_scaled_positions(groups_sub, gals_merged):
    """
    Compute stacked galaxy positions in units of r100.

    Uses angular scaling directly:
        x_scaled = (ΔRA cos(dec0)) / theta_r100
        y_scaled = (ΔDec)          / theta_r100

    where theta_r100 = r100 / distance_groups   [radians]
    """
    if len(groups_sub) == 0:
        return np.empty(0), np.empty(0)

    valid_ids = set(groups_sub[GROUP_ID_COL].to_numpy())
    gm = gals_merged[gals_merged[GAL_GROUP_ID_COL].isin(valid_ids)].copy()

    if len(gm) == 0:
        return np.empty(0), np.empty(0)

    ra = gm[GAL_RA_COL].to_numpy()
    dec = gm[GAL_DEC_COL].to_numpy()
    ra0 = gm[GROUP_RA_COL].to_numpy()
    dec0 = gm[GROUP_DEC_COL].to_numpy()
    r100 = gm[GROUP_R100_COL].to_numpy()
    dist = gm[GROUP_DIST_COL].to_numpy()

    # angular offsets in degrees
    dra_deg = wrapped_delta_ra_deg(ra, ra0)
    ddec_deg = dec - dec0

    # RA needs cos(dec0)
    x_ang_deg = dra_deg * np.cos(np.deg2rad(dec0))
    y_ang_deg = ddec_deg

    # angular size of r100 on the sky
    # theta_r100 in radians, then convert to degrees
    theta_r100_deg = np.rad2deg(r100 / dist)

    finite = (
        np.isfinite(x_ang_deg) &
        np.isfinite(y_ang_deg) &
        np.isfinite(theta_r100_deg) &
        (theta_r100_deg > 0)
    )

    x_scaled = x_ang_deg[finite] / theta_r100_deg[finite]
    y_scaled = y_ang_deg[finite] / theta_r100_deg[finite]

    return x_scaled, y_scaled


def compute_histogram(x, y, xrange, yrange, nbins=50, density=True):
    H, xedges, yedges = np.histogram2d(
        x,
        y,
        bins=[nbins, nbins],
        range=[xrange, yrange],
        density=density,
    )
    return H.T, xedges, yedges


# ============================================================
# Main
# ============================================================

def plot_masked_galaxy_density(groups, gals, save_path=SAVE_PATH):
    groups_use = prepare_group_table(groups)
    gals_use = prepare_galaxy_table(gals)

    if len(groups_use) == 0:
        raise ValueError("No valid rows left in groups after cleaning.")
    if len(gals_use) == 0:
        raise ValueError("No valid masked grouped galaxies left in gals after cleaning.")

    # merge group properties onto galaxies
    gals_merged = gals_use.merge(
        groups_use[
            [
                GROUP_ID_COL,
                GROUP_MULT_COL,
                GROUP_R100_COL,
                GROUP_DIST_COL,
                GROUP_RA_COL,
                GROUP_DEC_COL,
            ]
        ],
        left_on=GAL_GROUP_ID_COL,
        right_on=GROUP_ID_COL,
        how="inner",
    )

    if len(gals_merged) == 0:
        raise ValueError("No masked galaxies matched to groups by id_fof.")

    from matplotlib.colors import LogNorm

    panel_data = []
    vmax = 0.0
    positive_vals = []

    for label, lo, hi in BIN_DEFS:
        groups_sub = groups_use[
            (groups_use[GROUP_MULT_COL] >= lo) &
            (groups_use[GROUP_MULT_COL] <= hi)
        ].copy()

        x_scaled, y_scaled = collect_scaled_positions(groups_sub, gals_merged)

        H, _, _ = compute_histogram(
            x_scaled,
            y_scaled,
            xrange=XRANGE,
            yrange=YRANGE,
            nbins=NBINS,
            density=NORMALIZE_TO_DENSITY,
        )

        pos = H[H > 0]
        if pos.size:
            positive_vals.append(pos.min())
            vmax = max(vmax, pos.max())

        panel_data.append({
            "label": label,
            "groups_sub": groups_sub,
            "x": x_scaled,
            "y": y_scaled,
            "H": H,
        })

    global_vmin = min(positive_vals) if positive_vals else None

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    im = None
    for ax, pdata in zip(axes, panel_data):
        H = pdata["H"]
        n_groups = len(pdata["groups_sub"])
        n_gals = len(pdata["x"])

        im = ax.imshow(
            H,
            origin="lower",
            extent=[XRANGE[0], XRANGE[1], YRANGE[0], YRANGE[1]],
            aspect="equal",
            cmap=CMAP,
            interpolation="nearest",
            norm=LogNorm(vmin=global_vmin, vmax=vmax) if global_vmin is not None and vmax > 0 else None,
        )

        circ = plt.Circle((0, 0), 1.0, fill=False, color="cyan", lw=1.2)
        ax.add_patch(circ)

        ax.axhline(0, color="white", lw=0.5, alpha=0.3)
        ax.axvline(0, color="white", lw=0.5, alpha=0.3)

        ax.set_xlim(*XRANGE)
        ax.set_ylim(*YRANGE)
        ax.set_xlabel(r"$\Delta x / r_{100}$")
        ax.set_ylabel(r"$\Delta y / r_{100}$")
        ax.set_title(f"{pdata['label']}\n{n_groups} groups, {n_gals} masked galaxies", fontsize=11)

    cbar = fig.colorbar(im, ax=axes, shrink=0.95, pad=0.02)
    cbar.set_label("Density" if NORMALIZE_TO_DENSITY else "Count per bin")

    fig.suptitle(
        "Stacked masked-galaxy density relative to group centres, scaled by $r_{100}$",
        fontsize=16,
    )

    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure to: {save_path}")

    # useful debugging summary
    for pdata in panel_data:
        if len(pdata["x"]) > 0:
            print(
                f"{pdata['label']:>8}: "
                f"{len(pdata['groups_sub']):6d} groups, "
                f"{len(pdata['x']):8d} masked galaxies, "
                f"x range [{np.nanmin(pdata['x']): .3f}, {np.nanmax(pdata['x']): .3f}], "
                f"y range [{np.nanmin(pdata['y']): .3f}, {np.nanmax(pdata['y']): .3f}]"
            )
        else:
            print(f"{pdata['label']:>8}: {len(pdata['groups_sub']):6d} groups, 0 masked galaxies")


# ============================================================
# Example call
# ============================================================

plot_masked_galaxy_density(groups, gals)