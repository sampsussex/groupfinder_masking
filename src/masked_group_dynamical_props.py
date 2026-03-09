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


ra, dec, redshifts = gals['ra'], gals['dec'], gals['redshift_observed']
cosmo = FlatCosmology(h = 0.7, omega_matter = 0.3)

if REGION == 'wide':
    running_density = create_density_function(redshifts, total_counts = len(redshifts), survey_fractional_area = 0.02748648583, cosmology = cosmo)

# Running group catalog
red_cat = RedshiftCatalog(ra, dec, redshifts, running_density, cosmo)
red_cat.set_completeness()

red_cat.group_ids = gals['id_fof'].values

group_catalog_dict = red_cat.calculate_group_table(gals['mag_abs_Z_VISTA'], velocity_errors = np.zeros_like(gals['mag_abs_Z_VISTA']))

print("Group catalog dictionary keys:", group_catalog_dict.keys())
print(group_catalog_dict)