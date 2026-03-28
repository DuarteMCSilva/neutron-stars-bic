import pandas as pd
import os
import matplotlib.pyplot as plt

#Depends on the data structure
DATA_DIR = os.path.join('./data/generated_eos/2026-03-24_18-34-02.csv')

data = pd.read_csv(DATA_DIR, sep = " ")

print(data.head())

data = data.dropna()

def plot_vs(plt, data_groups: pd.DataFrameGroupBy):
    plt.set_title('Density vs. id')
    plt.set_xlabel('rho (fm^-3)')
    plt.set_ylabel('step id')
    plt.set_yscale('log')

    sum = 0
    for entry_id, group in data_groups:
        if entry_id < 5:
            plt.plot(group.index - sum, group['rho'], label=f'EoS {entry_id}')
        sum += len(group)
    #plt.savefig('M(R).png')
    #plt.show()

fig, ax = plt.subplots(1, 1, figsize=(6, 6))

grouped = data.groupby('id')
plot_vs(ax, grouped)
plt.show()
