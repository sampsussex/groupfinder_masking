#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection


# ============================================================
# Load in stars/ghosts
# ============================================================
print('loading stars')
star_path = '/Users/sp624AA/Code/waves_polygon_mask/23-06-25_masked_objects_list/Masking/gaiastarmaskwaves.csv'
stars = pd.read_csv(star_path)

def mask_radius_waves(g):
    """
    Calculate r[deg] based on the given formula.
    """
    g = np.asarray(g)
    r = np.zeros_like(g, dtype=float)
    mask1 = (g > 3.5) & (g < 16)
    mask2 = g <= 3.5
    r[mask1] = (10 ** (1.3 - 0.13 * g[mask1]))
    r[mask2] = 7
    return r / 60


# Get masking Radii
stars['radius'] = mask_radius_waves(stars['phot_g_mean_mag'])
stars = stars[(stars['phot_g_mean_mag'] < 16)][['ra', 'dec', 'radius']]

ghosts_path = '/Users/sp624AA/Code/waves_polygon_mask/23-06-25_masked_objects_list/Masking/GhostLocations_v0.csv'
ghosts = pd.read_csv(ghosts_path)
ghosts['radius'] = ghosts['radius'] / 60
ghosts = ghosts[['ra', 'dec', 'radius']]

stars = pd.concat([stars, ghosts], ignore_index=True)

# ============================================================
# Load in groups
# ============================================================
print('loading groups')
from utils import load_and_format_sharks_gals
from nessie import FlatCosmology, RedshiftCatalog
from nessie.helper_funcs import create_density_function

# gals, groups = load_and_format_sharks_gals(
#     "/Users/sp624AA/Downloads/group_finding_mocks/galaxies_shark.parquet",
#    region='wide'
# )
gals = pd.read_parquet("/Users/sp624AA/Downloads/group_finding_mocks/galaxies_filtered_wide_mock_fixed_masking.parquet")


AREA_ALL = 0.02748648583
cosmo = FlatCosmology(h=0.7, omega_matter=0.3)
ra, dec, z = gals['ra'], gals['dec'], gals['redshift_observed']
mag = gals['mag_abs_Z_VISTA']
density = create_density_function(
    z,
    total_counts=len(z),
    survey_fractional_area=AREA_ALL,
    cosmology=cosmo
)
cat = RedshiftCatalog(ra, dec, z, density, cosmo)
cat.group_ids = gals['id_fof'].values

groups = pd.DataFrame(cat.calculate_group_table(mag, velocity_errors=np.zeros_like(mag)))

from astropy.cosmology import FlatLambdaCDM
astropycosmo = FlatLambdaCDM(H0=70, Om0=0.3)
groups['distance_groups'] = astropycosmo.comoving_distance(groups['iter_redshift']).to('Mpc').value

print(groups.columns)
print('groups and stars loaded')

# Required columns in groups
GROUP_MULT_COL = "multiplicity"
GROUP_R100_COL = "r100"
GROUP_DIST_COL = "distance_groups"
GROUP_RA_COL = "iter_ra"
GROUP_DEC_COL = "iter_dec"

# Required columns in stars
STAR_RA_COL = "ra"
STAR_DEC_COL = "dec"
STAR_RADIUS_COL = "radius"

# Star radius unit: "deg", "arcmin", or "arcsec"
STAR_RADIUS_UNIT = "deg"

# Output
SAVE_PATH = "stacked_star_masks_by_group_multiplicity.png"
DPI = 300

# Plot appearance
FIGSIZE = (8, 8)
ALPHA = 0.075
EDGECOLOR = "none"
FACECOLOR = "black"
GROUP_BOUNDARY_COLOR = "tab:red"
GROUP_BOUNDARY_LW = 1.2

# Subsampling
MAX_GROUPS_PER_BIN = 50
RANDOM_SEED = 99

# Optional: clip stars whose centres are too far from the group centre
APPLY_INTERSECTION_FILTER = True

# Bins for the 2x4 grid
BIN_DEFS = [
    ("N = 3", 3, 3),
    ("N = 4", 4, 4),
    ("N = 5", 5, 5),
    ("N = 7", 7, 7),
    ("N = 9", 9, 9),
    ("N = 11", 11, 11),
    ("N = 15", 15, 15),
    ("N = 20-23", 20, 23),
    ("N = 30-50", 30, 50),

]


# ============================================================
# Utilities
# ============================================================

def radius_to_radians(radius, unit="deg"):
    """Convert angular radius array to radians."""
    radius = np.asarray(radius, dtype=float)
    if unit == "deg":
        return np.deg2rad(radius)
    elif unit == "arcmin":
        return np.deg2rad(radius / 60.0)
    elif unit == "arcsec":
        return np.deg2rad(radius / 3600.0)
    else:
        raise ValueError("STAR_RADIUS_UNIT must be 'deg', 'arcmin', or 'arcsec'")


def spherical_to_unitvec(ra_deg, dec_deg):
    """Convert RA/Dec in degrees to 3D unit vectors."""
    ra = np.deg2rad(np.asarray(ra_deg, dtype=float))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=float))

    cos_dec = np.cos(dec)
    x = cos_dec * np.cos(ra)
    y = cos_dec * np.sin(ra)
    z = np.sin(dec)
    return np.column_stack((x, y, z))


def wrapped_delta_ra_deg(ra_deg, ra0_deg):
    """Smallest signed RA difference in degrees."""
    return (ra_deg - ra0_deg + 180.0) % 360.0 - 180.0


def build_star_kdtree(stars, ra_col, dec_col):
    """Build a KD-tree in 3D unit-vector space for spherical cone searches."""
    star_xyz = spherical_to_unitvec(stars[ra_col].to_numpy(), stars[dec_col].to_numpy())
    tree = cKDTree(star_xyz)
    return tree, star_xyz


def groups_in_bin(groups, mult_col, lo, hi):
    mask = (groups[mult_col] >= lo) & (groups[mult_col] <= hi)
    return groups.loc[mask].copy()


def sample_groups(groups_sub, max_groups, rng):
    """
    Randomly subsample up to max_groups rows from groups_sub.
    Returns sampled dataframe and total number originally in the bin.
    """
    n_total = len(groups_sub)
    if n_total <= max_groups:
        return groups_sub.copy(), n_total

    choice = rng.choice(n_total, size=max_groups, replace=False)
    sampled = groups_sub.iloc[choice].copy()
    return sampled, n_total


def make_stacked_mask_panel(
    ax,
    groups_sub,
    stars,
    tree,
    max_star_radius_rad,
    label,
    n_total_in_bin,
    group_ra_col,
    group_dec_col,
    group_r100_col,
    group_dist_col,
    star_ra_col,
    star_dec_col,
    star_radius_rad,
    alpha=0.02,
    facecolor="black",
    edgecolor="none",
    draw_group_boundary=True,
):
    """
    Draw one panel using angular coordinates on the sky.

    r100 is assumed to be in comoving Mpc.
    distance_groups is assumed to be comoving distance in Mpc.
    """

    patches = []
    n_groups_used = 0
    n_star_circles = 0

    star_ra = stars[star_ra_col].to_numpy()
    star_dec = stars[star_dec_col].to_numpy()

    for row in groups_sub.itertuples(index=False):
        ra0 = getattr(row, group_ra_col)
        dec0 = getattr(row, group_dec_col)
        r100 = getattr(row, group_r100_col)      # comoving Mpc
        dist = getattr(row, group_dist_col)      # comoving Mpc

        if not np.isfinite(ra0) or not np.isfinite(dec0) or not np.isfinite(r100) or not np.isfinite(dist):
            continue
        if r100 <= 0 or dist <= 0:
            continue

        # angular group radius on sky
        theta100 = r100 / dist

        # search to plot-box corner plus largest mask
        theta_group = np.sqrt(2.0) * theta100
        theta_query = theta_group + max_star_radius_rad
        chord_query = 2.0 * np.sin(0.5 * theta_query)

        group_xyz = spherical_to_unitvec(np.array([ra0]), np.array([dec0]))[0]
        idx = tree.query_ball_point(group_xyz, r=chord_query)

        if len(idx) == 0:
            n_groups_used += 1
            continue

        idx = np.asarray(idx, dtype=int)

        dra_deg = wrapped_delta_ra_deg(star_ra[idx], ra0)
        ddec_deg = star_dec[idx] - dec0
        cos_dec0 = np.cos(np.deg2rad(dec0))

        # tangent-plane angular offsets in radians
        x_rad = np.deg2rad(dra_deg) * cos_dec0
        y_rad = np.deg2rad(ddec_deg)

        # star mask radius already in radians
        r_rad = star_radius_rad[idx]

        # normalise by group angular radius
        x_plot = x_rad / theta100
        y_plot = y_rad / theta100
        r_plot = r_rad / theta100

        if APPLY_INTERSECTION_FILTER:
            keep = (
                (x_plot + r_plot >= -1.0) &
                (x_plot - r_plot <=  1.0) &
                (y_plot + r_plot >= -1.0) &
                (y_plot - r_plot <=  1.0)
            )
            x_plot = x_plot[keep]
            y_plot = y_plot[keep]
            r_plot = r_plot[keep]

        for x, y, r in zip(x_plot, y_plot, r_plot):
            if np.isfinite(x) and np.isfinite(y) and np.isfinite(r) and (r > 0):
                patches.append(Circle((x, y), r))
                n_star_circles += 1

        n_groups_used += 1

    if patches:
        pc = PatchCollection(
            patches,
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha=alpha,
            linewidths=0.0,
            rasterized=True,
        )
        ax.add_collection(pc)

    if draw_group_boundary:
        ax.add_patch(
            Circle((0, 0), 1.0, fill=False, lw=GROUP_BOUNDARY_LW, color=GROUP_BOUNDARY_COLOR)
        )

    # minimal per-panel legend text only
    legend_text = f"{label}\nMasks: {n_star_circles:,}"
    ax.text(
        0.03, 0.97, legend_text,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.7", alpha=0.9)
    )

    ax.set_aspect("equal")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.axhline(0, color="0.85", lw=0.6, zorder=0)
    ax.axvline(0, color="0.85", lw=0.6, zorder=0)

    return {
        "n_groups_used": n_groups_used,
        "n_patches": n_star_circles,
        "n_total_in_bin": n_total_in_bin,
        "n_plotted": len(groups_sub),
    }


# ============================================================
# Main plotting function
# ============================================================

def plot_stacked_star_masks(groups, stars, save_path=SAVE_PATH, max_groups_per_bin=1000, random_seed=42):
    """
    Make a 2x4 grid of stacked transparent star-mask circle plots,
    binned by group multiplicity, using at most max_groups_per_bin
    random groups in each bin.
    """

    rng = np.random.default_rng(random_seed)

    groups_use = groups[
        [GROUP_MULT_COL, GROUP_R100_COL, GROUP_DIST_COL, GROUP_RA_COL, GROUP_DEC_COL]
    ].copy()

    stars_use = stars[
        [STAR_RA_COL, STAR_DEC_COL, STAR_RADIUS_COL]
    ].copy()

    groups_use = groups_use.replace([np.inf, -np.inf], np.nan).dropna()
    stars_use = stars_use.replace([np.inf, -np.inf], np.nan).dropna()

    groups_use = groups_use[
        (groups_use[GROUP_R100_COL] > 0) &
        (groups_use[GROUP_DIST_COL] > 0)
    ].copy()

    stars_use = stars_use[stars_use[STAR_RADIUS_COL] > 0].copy()

    if len(groups_use) == 0:
        raise ValueError("No valid rows left in groups after cleaning.")
    if len(stars_use) == 0:
        raise ValueError("No valid rows left in stars after cleaning.")

    star_radius_rad = radius_to_radians(
        stars_use[STAR_RADIUS_COL].to_numpy(),
        unit=STAR_RADIUS_UNIT
    )
    max_star_radius_rad = np.nanmax(star_radius_rad)

    tree, _ = build_star_kdtree(stars_use, STAR_RA_COL, STAR_DEC_COL)

    fig, axes = plt.subplots(
        3, 3,  # 2, 4
        figsize=FIGSIZE,
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0.0, "hspace": 0.0}
    )
    axes = axes.ravel()

    summaries = []

    for i, (ax, (label, lo, hi)) in enumerate(zip(axes, BIN_DEFS)):
        groups_bin = groups_in_bin(groups_use, GROUP_MULT_COL, lo, hi)

        if len(groups_bin) > max_groups_per_bin:
            groups_sub = groups_bin.sample(n=max_groups_per_bin, random_state=random_seed).copy()
        else:
            groups_sub = groups_bin.copy()

        summary = make_stacked_mask_panel(
            ax=ax,
            groups_sub=groups_sub,
            stars=stars_use,
            tree=tree,
            max_star_radius_rad=max_star_radius_rad,
            label=label,
            n_total_in_bin=len(groups_bin),
            group_ra_col=GROUP_RA_COL,
            group_dec_col=GROUP_DEC_COL,
            group_r100_col=GROUP_R100_COL,
            group_dist_col=GROUP_DIST_COL,
            star_ra_col=STAR_RA_COL,
            star_dec_col=STAR_DEC_COL,
            star_radius_rad=star_radius_rad,
            alpha=ALPHA,
            facecolor=FACECOLOR,
            edgecolor=EDGECOLOR,
            draw_group_boundary=True,
        )
        summaries.append((label, summary))

        ax.set_xticks([-1, -0.5, 0, 0.5])
        ax.set_yticks([-1, -0.5, 0, 0.5])

        # only show outer tick labels
        row, col = divmod(i, 3)

        # remove ticks inside the grid
        if row != 2:   # not bottom row
            ax.tick_params(bottom=False, labelbottom=False)

        if col != 0:   # not left column
            ax.tick_params(left=False, labelleft=False)

        # remove ticks on touching edges
        if row != 0:
            ax.tick_params(top=False)

        if col != 2:
            ax.tick_params(right=False)

        # if the tick is has a value of 1, set it to ""

    # axis labels only on outer edge
    # axes[0].set_ylabel(r"$\Delta \mathrm{Dec} / \theta_{100}$")
    axes[3].set_ylabel(r"$\Delta \mathrm{Dec} / \theta_{100}$")
    # axes[6].set_ylabel(r"$\Delta \mathrm{Dec} / \theta_{100}$")
    #axes[4].set_xlabel(r"$\Delta \mathrm{RA}\cos(\mathrm{Dec}_0) / \theta_{100}$")
    # axes[6].set_xlabel(r"$\Delta \mathrm{RA}\cos(\mathrm{Dec}_0) / \theta_{100}$")
    axes[7].set_xlabel(r"$\Delta \mathrm{RA}\cos(\mathrm{Dec}_0) / \theta_{100}$")
    # axes[8].set_xlabel(r"$\Delta \mathrm{RA}\cos(\mathrm{Dec}_0) / \theta_{100}$")

    fig.suptitle(
        f"Scaled stacked group star/ghostmask footprints\n {max_groups_per_bin:,} samples",
        fontsize=16
    )
    #fig.subplots_adjust(left=0.07, right=0.98, bottom=0.07, top=0.93)
    fig.savefig(save_path, dpi=DPI)#, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure to: {save_path}")
    for label, s in summaries:
        avg = s['n_patches'] / s['n_groups_used'] if s['n_groups_used'] > 0 else np.nan
        print(
            f"{label:>8}: plotted {s['n_plotted']:6d} / total {s['n_total_in_bin']:6d}, "
            f"{s['n_patches']:12,d} circles, average {avg:.2f} circles/group"
        )


plot_stacked_star_masks(
    groups,
    stars,
    save_path="../plots/stacked_star_masks_by_group_multiplicity.png",
    max_groups_per_bin=MAX_GROUPS_PER_BIN,
    random_seed=42,
)