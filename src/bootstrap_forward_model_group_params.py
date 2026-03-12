import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy import cosmology
from utils import load_and_format_sharks_gals
from nessie import FlatCosmology, RedshiftCatalog
from nessie.helper_funcs import create_density_function

# ============================================================
# Setup
# ============================================================
astropycosmo = cosmology.FlatLambdaCDM(H0=70, Om0=0.3)
REGION = 'wide'
AREA_ALL, AREA_MASKED = 0.02748648583, 0.02565146777
area_ratio = AREA_ALL / AREA_MASKED
area_ratio = area_ratio # Just for testing.

IDS = 'Nessie'  # 'id_fof' or 'Nessie'

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

for cat in (group_cat_all, group_cat_masked):
    cat['multiplicity'] = cat['multiplicity'].astype(int)

# ============================================================
# Bootstrap histogram
#
# Resampling is done on the *group* catalogue with replacement, so the
# survey footprint (and therefore the density function / area normalisation)
# is never altered — only the effective weight of each group changes.
# Errors are the 16th–84th percentile interval of the bootstrap distribution,
# equivalent to ±1σ for a Gaussian.
# ============================================================

def bootstrap_histogram(data_vals, bins, weight=1.0, n_boot=N_BOOTSTRAP, rng=RNG):
    """
    Return the full-sample weighted histogram and 16th/84th percentile
    bootstrap errors.

    Parameters
    ----------
    data_vals : (N,) float array  – already-filtered (finite) values
    bins      : bin edges
    weight    : scalar weight per entry (e.g. area-correction factor)
    n_boot    : number of bootstrap iterations
    rng       : numpy Generator

    Returns
    -------
    h_full  : (nbins,) array  – weighted counts using all data
    h_lo    : (nbins,) array  – 16th percentile of bootstrap distribution
    h_hi    : (nbins,) array  – 84th percentile of bootstrap distribution
    """
    data_vals = np.asarray(data_vals, dtype=float)
    n         = len(data_vals)
    w_scalar  = float(weight)

    h_full, _ = np.histogram(data_vals, bins=bins,
                              weights=np.full(n, w_scalar))

    h_boot = np.empty((n_boot, len(bins) - 1), dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)          # resample with replacement
        h_boot[i], _ = np.histogram(data_vals[idx], bins=bins,
                                     weights=np.full(n, w_scalar))

    h_lo = np.percentile(h_boot, 16, axis=0)
    h_hi = np.percentile(h_boot, 84, axis=0)

    return h_full, h_lo, h_hi


# ============================================================
# Plotting helpers
# ============================================================

def frac_resid(h_ref, h_comp):
    out = np.full_like(h_ref, np.nan, dtype=float)
    m = h_ref > 0
    out[m] = (h_comp[m] - h_ref[m]) / h_ref[m]
    return out


def frac_resid_percentiles(h_ref, h_ref_lo, h_ref_hi,
                            h_comp, h_comp_lo, h_comp_hi):
    """
    Propagate the asymmetric bootstrap percentile bands into fractional
    residual space: r = (comp - ref) / ref.

    Upper error on r comes from high comp / low ref, and vice-versa.
    """
    nbins   = len(h_ref)
    r       = frac_resid(h_ref, h_comp)
    r_lo    = np.full(nbins, np.nan)
    r_hi    = np.full(nbins, np.nan)

    m = h_ref > 0
    # pessimistic propagation: use the bound that maximises the error
    r_lo[m] = (h_comp_lo[m] - h_ref_hi[m]) / h_ref[m]   # most negative
    r_hi[m] = (h_comp_hi[m] - h_ref_lo[m]) / h_ref[m]   # most positive

    return r, r_lo, r_hi


def hist_resid_plot(
    data_all, data_masked,
    bins, weight, xlabel, title, outfile
):
    centres = 0.5 * (bins[:-1] + bins[1:])
    widths  = np.diff(bins)

    h_all,    all_lo,    all_hi    = bootstrap_histogram(data_all,    bins, weight=1.0)
    h_masked, masked_lo, masked_hi = bootstrap_histogram(data_masked, bins, weight=weight)

    r, r_lo, r_hi = frac_resid_percentiles(
        h_all, all_lo, all_hi,
        h_masked, masked_lo, masked_hi
    )

    label_masked = "Masked" if weight == 1.0 else f"Masked x {weight:.3f}"

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
        constrained_layout=True
    )

    # --- top panel: histograms + bootstrap percentile bands ---
    ax1.hist(data_all,    bins=bins, histtype="step", lw=2, color="C0", label="All")
    ax1.hist(data_masked, bins=bins, histtype="step", lw=2, color="C1", label=label_masked,
             weights=np.full(len(data_masked), weight))

    ax1.bar(centres, all_hi    - all_lo,    width=widths, bottom=all_lo,
            alpha=0.20, color="C0", label="Bootstrap 16–84% (All)")
    ax1.bar(centres, masked_hi - masked_lo, width=widths, bottom=masked_lo,
            alpha=0.20, color="C1", label="Bootstrap 16–84% (Masked)")

    ax1.set(yscale="log", ylabel="Count", title=title)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # --- bottom panel: fractional residual + propagated bootstrap band ---
    ax2.axhline(0, color="k", ls="--", lw=1)
    ax2.plot(centres, r, drawstyle="steps-mid", lw=1.8, color="C2")
    ax2.fill_between(
        centres, r_lo, r_hi,
        step="mid", alpha=0.35, color="C2", label="Bootstrap 16–84%"
    )
    ax2.set(ylabel="Frac.\nresid", xlabel=xlabel)
    ax2.set_ylim(-0.2, 0.2)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.show()


def make_shared_bins(a, b, nbins=20, log=False):
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    return (np.logspace(np.log10(lo), np.log10(hi), nbins + 1)
            if log else np.linspace(lo, hi, nbins + 1))


# ============================================================
# R50 & velocity dispersion plots (multiplicity >= 5)
# ============================================================

for col, xlabel, label in [
    ("r50",                     "R50 [Mpc]",                      "R50"),
    ("velocity_dispersion_gap", r"$\sigma_\mathrm{GAP}$ [km/s]",  "Velocity dispersion"),
]:
    sub_all    = group_cat_all   [group_cat_all   ['multiplicity'] >= 5]
    sub_masked = group_cat_masked[group_cat_masked['multiplicity'] >= 5]

    data_all    = sub_all[col].astype(float).values
    data_masked = sub_masked[col].astype(float).values

    data_all    = data_all   [np.isfinite(data_all)]
    data_masked = data_masked[np.isfinite(data_masked)]

    bins = make_shared_bins(data_all, data_masked)

    for weight, suffix, title_sfx in [
        (1.0,        "log_counts",               ""),
        (area_ratio, "log_counts_area_corrected", ", area corrected"),
    ]:
        hist_resid_plot(
            data_all, data_masked,
            bins, weight, xlabel,
            title=f"{IDS} Groups {label} distribution{title_sfx}",
            outfile=f"../plots/{IDS}_group_{col}_histograms_{suffix}.png",
        )

# ============================================================
# Dynamical mass plots (multiplicity >= 5)
# ============================================================
A, G, = 10, 4.302e-9

for cat in (group_cat_all, group_cat_masked):
    cat['dynamical_mass'] = A * cat['velocity_dispersion_gap']**2 * cat['r50'] * 2 / G

cats5 = {
    'all':    group_cat_all   [group_cat_all   ['multiplicity'] >= 5],
    'masked': group_cat_masked[group_cat_masked['multiplicity'] >= 5],
}


def get_logmass(cat):
    v  = np.asarray(cat['dynamical_mass'], dtype=float)
    ok = (v > 0) & np.isfinite(v)
    return np.log10(v[ok])


logm_all    = get_logmass(cats5['all'])
logm_masked = get_logmass(cats5['masked'])

bins   = make_shared_bins(logm_all, logm_masked, nbins=20)
xlabel = r"$\log_{10}(M_{\rm dyn}\,[h^{-1} M_\odot])$"

for weight, suffix, title_sfx in [
    (1.0,       "uncorrected",    ""),
    (area_ratio, "area_corrected", f" (masked x {area_ratio:.3f})"),
]:
    hist_resid_plot(
        logm_all, logm_masked,
        bins, weight, xlabel,
        title=f"{IDS} Group dynamical mass distribution{title_sfx}",
        outfile=f"../plots/{IDS}_dynam_mass_no_lum_corr_{suffix}.png",
    )