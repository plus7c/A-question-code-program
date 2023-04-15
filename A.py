import pandas as pd
import numpy as np
import dimod
from dimod import SimulatedAnnealingSampler

# Read the CSV file and reshape it in to a 3D array
df = pd.read_csv('附件1：data_100.csv')
a = df.values.reshape((10, 100, 2))

# Define the problem
n_cards = 100
n_yuzhi = 10
Q = {(i, j): 80000 * a[i//n_cards, i%n_cards, 0] * a[j//n_cards, j%n_cards, 0] - 1080000 * a[i//n_cards, i%n_cards, 0] * a[i//n_cards, i%n_cards, 1] * a[j//n_cards, j%n_cards, 0] * a[j//n_cards, j%n_cards, 1] for i in range(n_cards*n_yuzhi) for j in range(n_cards*n_yuzhi) if i != j}
bqm = dimod.BinaryQuadraticModel.from_qubo(Q)

# Solve the problem
sampler = SimulatedAnnealingSampler()
sampleset = sampler.sample(bqm)

# Extract the solution
x = np.zeros((n_yuzhi, n_cards))
for i, j in sampleset.first.sample.items():
    x[i//n_cards, i%n_cards] = j

# Calculate the final income
final_income = 80000 * np.sum(x * a[:,:,0]) - 1080000 * np.sum(x * a[:,:,0] * a[:,:,1])
print('Maximum final income:', final_income)


