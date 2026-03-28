import pandas as pd
import os

import tov.plots as plots

import matplotlib.pyplot as plt

MIN_MASS = 1.97 #%PAR, in solar masses, according to the most massive neutron star observed until now (PSR J0740+6620). Any EOS that doesn't reach this mass won't be graphed (cannot replicate observations).

#Depends on the data structure
DATA_DIR = os.path.join('./data/output_rml/R_M_L_2026-03-24_19-00-41.csv')

data = pd.read_csv(DATA_DIR, sep = ",")

print(data.head())

print(data['M'])

eos_groups = data.groupby('id')

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
plots.plot_mass_radius(ax[0], eos_groups, min_mass=MIN_MASS)
plots.plot_mass_lambda(ax[1], eos_groups, min_mass=MIN_MASS)

#ax[1].tick_params(labelleft=False)  # hide y tick labels
ax[1].set_ylabel("")               # remove duplicate label
plt.show()
fig.savefig('output/tov_calculations/M(R)_and_Lambda(M).png')
