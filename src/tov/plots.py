import matplotlib.pyplot as plt

MIN_MASS = 1.97 #%PAR, in solar masses, according to the most massive neutron star observed until now (PSR J0740+6620). Any EOS that doesn't reach this mass won't be graphed (cannot replicate observations).

def plot_mass_radius(plt, data_groups, min_mass=MIN_MASS):
    plt.set_title('Mass vs. Radius')
    plt.set_xlabel('Radius (km)')
    plt.set_ylabel('Solar Masses (M☉)')

    for entry_id, group in data_groups:
        max_mass = group['M'].max()
        if max_mass >= min_mass:
            plt.plot(group['R'], group['M'], label=f'EoS {entry_id}')
    #plt.savefig('M(R).png')
    #plt.show()

def plot_mass_lambda(plt, data_groups, min_mass=MIN_MASS):
    for entry_id, group in data_groups:
        max_mass = group['M'].max()
        if max_mass >= min_mass:
            plt.plot(group['Lambda'], group['M'], label=f'EoS {entry_id}')

    plt.set_title('Mass vs. Lambda')
    plt.set_ylabel('Solar Masses (M☉)')
    plt.set_xlabel('Tidal Deformability (Lambda)')
    plt.set_xscale('log')

    #plt.savefig('Lambda(M).png')
    #plt.show()