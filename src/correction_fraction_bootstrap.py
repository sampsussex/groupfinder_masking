import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import load_and_format_sharks_gals

# gals, groups = load_and_format_sharks_gals(
#     "/Users/sp624AA/Downloads/group_finding_mocks/galaxies_shark.parquet",
#     region='wide-S'
#)
gals = pd.read_parquet("/Users/sp624AA/Downloads/group_finding_mocks/galaxies_filtered_wide_mock_fixed_masking.parquet")


# ============================================================
# USER SETTINGS
# ============================================================
GROUP_COL = "id_fof"
MASKED_COL = "masked"


print(len(gals), len(gals[gals['masked'] == 0]), len(gals)/len(gals[gals['masked'] == 0]))
NMIN = 2
NMAX = 25
NBOOT = 1000
RANDOM_SEED = 42

# If masked == True means "this galaxy is masked out", leave as False below.
# If masked == True means "this galaxy is kept / unmasked", set to True.
UNMASKED_FLAG_VALUE = False


# ============================================================
# HELPERS
# ============================================================
def get_group_multiplicities(gals, group_col="id_fof", masked_col="masked",
                             unmasked_flag_value=False):
    """
    Return multiplicities for:
      - all groups
      - the catalogue after masking (i.e. keeping only unmasked galaxies)

    Assumes id_fof == -1 means no group.
    """
    g = gals.copy()

    # keep only grouped galaxies
    g = g[g[group_col] != -1].copy()

    # multiplicity in the full catalogue
    mult_all = g.groupby(group_col).size().to_numpy()

    # multiplicity after masking
    g_unmasked = g[g[masked_col] == unmasked_flag_value].copy()
    mult_masked = g_unmasked.groupby(group_col).size().to_numpy()

    return mult_all, mult_masked


def hist_integer_counts(multiplicities, nmin=2, nmax=50):
    """
    Histogram integer multiplicities from nmin..nmax inclusive.
    """
    bins = np.arange(nmin - 0.5, nmax + 1.5, 1.0)
    counts, _ = np.histogram(multiplicities, bins=bins)
    centres = np.arange(nmin, nmax + 1)
    return counts, centres, bins


def bootstrap_hist_and_residual(mult_all, mult_masked, nmin=2, nmax=50,
                                nboot=1000, seed=42):
    """
    Bootstrap over groups, returning 1-sigma uncertainties on:
      - full counts
      - masked counts
      - residual = N_all / N_masked - 1
    """
    rng = np.random.default_rng(seed)

    counts_all_boot = np.empty((nboot, nmax - nmin + 1), dtype=float)
    counts_masked_boot = np.empty((nboot, nmax - nmin + 1), dtype=float)
    residual_boot = np.empty((nboot, nmax - nmin + 1), dtype=float)

    for i in range(nboot):
        sample_all = rng.choice(mult_all, size=len(mult_all), replace=True)
        sample_masked = rng.choice(mult_masked, size=len(mult_masked), replace=True)

        c_all, _, _ = hist_integer_counts(sample_all, nmin=nmin, nmax=nmax)
        c_masked, _, _ = hist_integer_counts(sample_masked, nmin=nmin, nmax=nmax)

        counts_all_boot[i] = c_all
        counts_masked_boot[i] = c_masked

        with np.errstate(divide="ignore", invalid="ignore"):
            resid = c_all / c_masked.astype(float) - 1.0
            resid[~np.isfinite(resid)] = np.nan
        residual_boot[i] = resid

    err_all = np.nanstd(counts_all_boot, axis=0, ddof=1)
    err_masked = np.nanstd(counts_masked_boot, axis=0, ddof=1)
    err_resid = np.nanstd(residual_boot, axis=0, ddof=1)

    return err_all, err_masked, err_resid


# ============================================================
# MAIN
# ============================================================
mult_all, mult_masked = get_group_multiplicities(
    gals,
    group_col=GROUP_COL,
    masked_col=MASKED_COL,
    unmasked_flag_value=UNMASKED_FLAG_VALUE,
)

counts_all, xvals, bins = hist_integer_counts(mult_all, nmin=NMIN, nmax=NMAX)
counts_masked, _, _ = hist_integer_counts(mult_masked, nmin=NMIN, nmax=NMAX)

with np.errstate(divide="ignore", invalid="ignore"):
    residual = counts_all / counts_masked.astype(float) - 1.0
    residual[~np.isfinite(residual)] = np.nan

err_all, err_masked, err_resid = bootstrap_hist_and_residual(
    mult_all,
    mult_masked,
    nmin=NMIN,
    nmax=NMAX,
    nboot=NBOOT,
    seed=RANDOM_SEED,
)

# ============================================================
# PLOTTING
# ============================================================

#plt.rcParams.update({
#    "font.size": 18,
#    "axes.labelsize": 20,
#    "axes.titlesize": 24,
#    "xtick.labelsize": 16,
#    "ytick.labelsize": 16,
#    "legend.fontsize": 16,
#})

fig, (ax_resid, ax_hist) = plt.subplots(
    2, 1,
    figsize=(8, 4),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1]},
    constrained_layout=True,
)

# ------------------------------------------------------------
# Residual panel (main)
# ------------------------------------------------------------
valid = np.isfinite(residual)

ax_resid.axhline(0, color="k", linestyle="--", linewidth=1.5)

ax_resid.axhline(
    0.072,
    color="crimson",
    linestyle="--",
    linewidth=2,
)

ax_resid.text(
    NMAX-0.5,
    0.072,
    "Area correction factor",
    color="crimson",
    ha="right",
    va="bottom",
)

ax_resid.errorbar(
    xvals[valid],
    residual[valid],
    yerr=err_resid[valid],
    fmt="o",
    markersize=6,
    linewidth=2,
    capsize=3,
)

ax_resid.set_ylabel(r"$N_{\rm all}/N_{\rm masked} - 1$")
ax_resid.set_ylim(-0.20, 0.20)
ax_resid.grid(alpha=0.3)

ax_resid.set_title(
    "Effect of Stellar Masking on Group Multiplicity Distribution"
)

# ------------------------------------------------------------
# Histogram panel
# ------------------------------------------------------------
bins = np.arange(NMIN - 0.5, NMAX + 1.5, 1)

ax_hist.hist(
    mult_all,
    bins=bins,
    histtype="step",
    alpha=0.5,
    color='blue',
    label="All groups",
)

ax_hist.hist(
    mult_masked,
    bins=bins,
    histtype="step",
    alpha=0.5,
    color = 'red',
    label="Masked catalogue",
)

# bootstrap errorbars on histogram
ax_hist.errorbar(
    xvals,
    counts_all,
    yerr=err_all,
    fmt="none",
    color="blue",
    capsize=3,
)

ax_hist.errorbar(
    xvals,
    counts_masked,
    yerr=err_masked,
    fmt="none",
    color="red",
    capsize=3,
)

ax_hist.set_yscale("log")

ax_hist.set_xlabel("Number of Group Members")
ax_hist.set_ylabel("Number of Groups")

ax_hist.legend()
#ax_hist.grid(alpha=0.3, which="both")

plt.savefig("../plots/group_multiplicity_bootstrap.png", dpi=300)#, bbox_inches="tight")
plt.show()