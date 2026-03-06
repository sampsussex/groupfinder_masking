import numpy as np
import pandas as pd



def load_and_format_sharks_gals(
    file_path,
    region="deep",
    cols=None,
    bcg_on="Z_VISTA",
):
    """
    Load and format a SHARKS galaxy catalog.

    Parameters
    ----------
    file_path : str
        Path to the parquet file.
    region : {"deep", "wide", None}, optional
        Survey selection to apply.
    cols : list of str, optional
        Columns to read from parquet. If None, uses a sensible default set.
    bcg_on : {"Z_VISTA", "Stellar Mass"}, optional
        Criterion for central/BCG assignment within each FoF group.

    Returns
    -------
    gals : pandas.DataFrame
        Galaxy-level table with derived properties and broadcast group/BCG properties.
    groups : pandas.DataFrame
        Group-level table with one row per FoF group.
    """

    if cols is None:
        cols = [
            "ra",
            "dec",
            "id_galaxy_sky",
            "id_group_sky",
            "type",
            "zcos",
            "zobs",
            "mstars_bulge",
            "mstars_disk",
            "mgas_disk",
            "mgas_bulge",
            "mvir_hosthalo",
            "mvir_subhalo",
            "id_fof",
            "sfr_disk",
            "sfr_burst",
            "total_ab_dust_u_VST",
            "total_ab_dust_g_VST",
            "total_ab_dust_r_VST",
            "total_ab_dust_i_VST",
            "total_ab_dust_Z_VISTA",
            "total_ab_dust_Y_VISTA",
            "total_ab_dust_J_VISTA",
            "total_ab_dust_H_VISTA",
            "total_ab_dust_K_VISTA",
            "total_ap_dust_Z_VISTA",
        ]

    valid_regions = ["deep", "wide", None]
    if region not in valid_regions:
        raise ValueError(f"Invalid region={region!r}. Must be one of {valid_regions}.")

    if bcg_on not in ["Z_VISTA", "Stellar Mass"]:
        raise ValueError("Invalid value for bcg_on. Must be 'Z_VISTA' or 'Stellar Mass'.")

    required_cols = {
        "ra",
        "dec",
        "zobs",
        "id_fof",
        "id_group_sky",
        "mvir_hosthalo",
        "mstars_disk",
        "mstars_bulge",
        "sfr_disk",
        "sfr_burst",
        "total_ab_dust_Z_VISTA",
        "total_ap_dust_Z_VISTA",
    }
    missing = required_cols.difference(cols)
    if missing:
        raise ValueError(
            f"The following required columns are missing from `cols`: {sorted(missing)}"
        )

    gals = pd.read_parquet(file_path, columns=cols).copy()

    # ------------------------------------------------------------------
    # 1) Basic derived galaxy properties
    # ------------------------------------------------------------------
    h = 0.67

    gals["stellar_mass"] = (gals["mstars_disk"] + gals["mstars_bulge"]) / h
    gals["sfr_total"] = (gals["sfr_disk"] + gals["sfr_burst"]) / h

    # Safe logs
    gals["log_stellar_mass"] = np.log10(gals["stellar_mass"].where(gals["stellar_mass"] > 0))
    gals["log_sfr_total"] = np.log10(gals["sfr_total"].where(gals["sfr_total"] > 0) * 1e-9)

    # sSFR = SFR / Mstar
    gals["log_sSFR"] = np.log10(
        ((gals["sfr_disk"] + gals["sfr_burst"]) * 1e-9 / (gals["mstars_disk"] + gals["mstars_bulge"]))
        .where((gals["mstars_disk"] + gals["mstars_bulge"]) > 0)
    )

    # Basic cleaning
    mask = (
        gals["log_stellar_mass"].notna()
        & (gals["log_stellar_mass"] > 8.0)
        & gals["total_ab_dust_Z_VISTA"].notna()
        & (gals["total_ab_dust_Z_VISTA"] > -99)
    )
    gals = gals.loc[mask].reset_index(drop=True)

    group_col = "id_fof"
    host_id_col = "id_group_sky"
    mass_col = "mvir_hosthalo"
    mag_col = "total_ab_dust_Z_VISTA"
    stellar_mass_col = "stellar_mass"

    # ------------------------------------------------------------------
    # 2) BCG assignment
    #    - id_fof == -1 => each galaxy is its own central
    #    - otherwise choose brightest or most massive within FoF
    # ------------------------------------------------------------------
    gals["is_bcg"] = False

    ungrouped = gals[group_col].eq(-1)
    gals.loc[ungrouped, "is_bcg"] = True

    valid_grouped = (
        gals[group_col].notna()
        & gals[group_col].ne(-1)
        & gals[mag_col].notna()
        & gals[stellar_mass_col].notna()
    )

    if bcg_on == "Z_VISTA":
        # Brightest = minimum magnitude
        bcg_idx = gals.loc[valid_grouped].groupby(group_col)[mag_col].idxmin()
    else:  # bcg_on == "Stellar Mass"
        bcg_idx = gals.loc[valid_grouped].groupby(group_col)[stellar_mass_col].idxmax()

    gals.loc[bcg_idx, "is_bcg"] = True

    # ------------------------------------------------------------------
    # 3) Broadcast BCG properties to all members
    # ------------------------------------------------------------------
    bcg_broadcast_cols = ["ra", "dec", "zobs", mag_col, stellar_mass_col]

    for col in bcg_broadcast_cols:
        gals[f"{col}_bcg"] = gals[col]

    bcg_rows = gals.loc[
        gals["is_bcg"] & gals[group_col].ne(-1),
        [group_col] + bcg_broadcast_cols
    ].drop_duplicates(subset=[group_col])

    for col in bcg_broadcast_cols:
        mapper = bcg_rows.set_index(group_col)[col]
        gals.loc[gals[group_col].ne(-1), f"{col}_bcg"] = (
            gals.loc[gals[group_col].ne(-1), group_col].map(mapper)
        )

    gals["log_stellar_mass_bcg"] = np.log10(
        gals["stellar_mass_bcg"].where(gals["stellar_mass_bcg"] > 0)
    )

    # ------------------------------------------------------------------
    # 4) FoF halo mass = sum of UNIQUE subhalo host masses within each FoF
    #    using one mass per unique id_group_sky inside each id_fof
    # ------------------------------------------------------------------
    gals["fof_halo_mass"] = np.nan

    grouped = gals[group_col].notna() & gals[group_col].ne(-1)

    if host_id_col not in gals.columns:
        raise ValueError(
            f"Host ID column '{host_id_col}' not found. Cannot compute FoF halo mass."
        )

    fof_mass = (
        gals.loc[grouped, [group_col, host_id_col, mass_col]]
        .dropna()
        .drop_duplicates(subset=[group_col, host_id_col])
        .groupby(group_col, sort=False)[mass_col]
        .sum()
    )

    gals.loc[grouped, "fof_halo_mass"] = gals.loc[grouped, group_col].map(fof_mass)
    gals.loc[ungrouped, "fof_halo_mass"] = gals.loc[ungrouped, mass_col]
    gals["log_fof_halo_mass"] = np.log10(gals["fof_halo_mass"].where(gals["fof_halo_mass"] > 0))

    # ------------------------------------------------------------------
    # 5) Colour flag
    # ------------------------------------------------------------------
    gals["is_red"] = gals["log_sSFR"] < -11.0

    # ------------------------------------------------------------------
    # 6) Luminosities and group-integrated properties
    # ------------------------------------------------------------------
    # Assumed solar absolute magnitude in VISTA Z
    M_sun_Z = 4.51
    gals["L"] = 10.0 ** (-0.4 * (gals[mag_col] - M_sun_Z))

    # Only sum real groups; leave ungrouped as self-values
    gals["group_L"] = gals["L"]
    real_group_L = gals.loc[grouped].groupby(group_col)["L"].sum()
    gals.loc[grouped, "group_L"] = gals.loc[grouped, group_col].map(real_group_L)
    gals["log_group_L"] = np.log10(gals["group_L"].where(gals["group_L"] > 0))

    gals["n_group_fof"] = 1
    real_group_n = gals.loc[grouped].groupby(group_col).size()
    gals.loc[grouped, "n_group_fof"] = gals.loc[grouped, group_col].map(real_group_n).astype(int)

    gals["group_stellar_mass"] = gals["stellar_mass"]
    real_group_sm = gals.loc[grouped].groupby(group_col)["stellar_mass"].sum()
    gals.loc[grouped, "group_stellar_mass"] = gals.loc[grouped, group_col].map(real_group_sm)
    gals["log_group_stellar_mass"] = np.log10(
        gals["group_stellar_mass"].where(gals["group_stellar_mass"] > 0)
    )

    # ------------------------------------------------------------------
    # 7) Region cuts
    # ------------------------------------------------------------------
    if region == "deep":
        mask_region = (
            (gals["ra"] > 339.0)
            & (gals["ra"] < 350.0)
            & (gals["dec"] > -35.0)
            & (gals["dec"] < -30.0)
            & (gals["zobs"] < 0.8)
            & (gals["total_ap_dust_Z_VISTA"] < 21.25)
        )
        gals = gals.loc[mask_region].reset_index(drop=True)

    elif region == "wide":
        # Allow RA wrap-around cleanly
        ra_360 = gals["ra"] % 360.0

        mask_wide_N = (
            (ra_360 > 157.25)
            & (ra_360 < 225.0)
            & (gals["dec"] > -3.95)
            & (gals["dec"] < 3.95)
            & (gals["zobs"] < 0.2)
            & (gals["total_ap_dust_Z_VISTA"] < 21.25)
        )

        mask_wide_S = (
            ((ra_360 > 330.0) | (ra_360 < 51.6))
            & (gals["dec"] > -35.6)
            & (gals["dec"] < -27.0)
            & (gals["zobs"] < 0.2)
            & (gals["total_ap_dust_Z_VISTA"] < 21.25)
        )

        gals = gals.loc[mask_wide_N | mask_wide_S].reset_index(drop=True)

    # ------------------------------------------------------------------
    # 8) Build group table AFTER region selection
    # ------------------------------------------------------------------
    group_cols = [
        "ra_bcg",
        "dec_bcg",
        "zobs_bcg",
        f"{mag_col}_bcg",
        "stellar_mass_bcg",
        "log_stellar_mass_bcg",
        "fof_halo_mass",
        "log_fof_halo_mass",
        "group_L",
        "log_group_L",
        "n_group_fof",
        "group_stellar_mass",
        "log_group_stellar_mass",
    ]

    groups = (
        gals.groupby(group_col, sort=False)[group_cols]
        .first()
        .reset_index()
    )

    # ------------------------------------------------------------------
    # 9) Satellite counts and BCG colour at group level
    # ------------------------------------------------------------------
    groups["n_sat"] = (groups["n_group_fof"] - 1).clip(lower=0).astype(int)

    red_sat_counts = (
        gals.loc[gals[group_col].ne(-1) & (~gals["is_bcg"]) & gals["is_red"]]
        .groupby(group_col)
        .size()
    )

    # Reindex by id_fof, not by groups.index
    groups["n_sat_red"] = (
        groups[group_col].map(red_sat_counts).fillna(0).astype(int)
    )

    groups["n_sat_blue"] = (groups["n_sat"] - groups["n_sat_red"]).clip(lower=0).astype(int)

    bcg_is_red = (
        gals.loc[gals["is_bcg"]]
        .groupby(group_col)["is_red"]
        .first()
    )

    groups["is_red_bcg"] = groups[group_col].map(bcg_is_red).fillna(False).astype(bool)
    groups["is_blue_bcg"] = ~groups["is_red_bcg"]

    return gals, groups



