# So, i want to look these questions; 
# For nessie groups, how many are perfectly recovered? I could also use S score here as well.
# For those that arent, I need to have a look at the pathways each group take
# in each N bin. I think a sankey diagram could be a fun way of looking
# at this. 
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

    return pd.DataFrame(cat.calculate_group_table(mag, velocity_errors=np.zeros_like(mag))), cat


group_cat_all, nessie_all   = run_catalog(ra,   dec,   z,   density_all,    gals['mag_abs_Z_VISTA'],        cosmo, gals)
group_cat_masked, nessie_masked = run_catalog(ra_m, dec_m, z_m, density_masked, gals_masked['mag_abs_Z_VISTA'], cosmo, gals_masked)


nessie_masked.mock_group_ids = nessie_all.group_ids[(gals['masked'] == False)]

#score = nessie_masked.compare_to_mock(min_group_size=5)
#print("S score, masked vs unmasked catalogs, n_group size_comp = ", score)
#score2 = nessie_masked.compare_to_mock(min_group_size=3)
#print("S score, masked vs unmasked catalogs", score2)
ns_for_scores = []
scores = []
for i in range(3, 50):
    score = nessie_masked.compare_to_mock(min_group_size=i)
    print(f"S score, masked vs unmasked catalogs, n_group size_comp >= {i}: ", score)
    scores.append(score)
    ns_for_scores.append(i)

other_ns = [20, 30, 50]

for i in other_ns:
    score = nessie_masked.compare_to_mock(min_group_size=i)
    print(f"S score, masked vs unmasked catalogs, n_group size_comp >= {i}: ", score)
#    scores.append(score)
#    ns_for_scores.append(i)

plt.plot(ns_for_scores, scores, marker='o')
plt.xlabel("Minimum group size for S score comparison")
plt.ylabel("S score")
plt.title("Nessie groups S-score for masked vs unmasked catalogs")
plt.grid()
plt.ylim(0.90, 1.0)
plt.savefig("../plots/nessie_masked_vs_unmasked_S_score.png", dpi=300)
plt.show()
