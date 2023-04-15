import numpy as np
import dwavebinarycsp
from dwave.system import LeapHybridSampler
# Load the pass rates and bad debt rates from the CSV file
data = np.loadtxt('附件1：data_100.csv', delimiter=',')
pass_rates = data[:, ::2]
bad_debt_rates = data[:, 1::2]

# Define the loan funds, interest income rate, and number of scorecards and thresholds
loan_funds = 1000000
interest_income_rate = 0.08
n_scorecards = 100
n_thresholds = 10

# Define the binary variables for each scorecard and threshold combination
variables = np.zeros((n_scorecards, n_thresholds), dtype=int)
for i in range(n_scorecards):
    for j in range(n_thresholds):
        variables[i, j] = i * n_thresholds + j

# Define the objective function as the negative of the final income
def objective(variables):
    selected = np.where(variables == 1)[0]
    pass_rates_selected = pass_rates[selected // n_thresholds, selected % n_thresholds]
    bad_debt_rates_selected = bad_debt_rates[selected // n_thresholds, selected % n_thresholds]
    total_pass_rate = np.prod(pass_rates_selected)
    total_bad_debt_rate = np.mean(bad_debt_rates_selected)
    loan_interest_income = loan_funds * interest_income_rate * total_pass_rate * (1 - total_bad_debt_rate)
    bad_debt_loss = loan_funds * total_pass_rate * total_bad_debt_rate
    return -(loan_interest_income - bad_debt_loss)

# Define the constraints to ensure that only one threshold is selected for each scorecard
def threshold_constraints(variables):
    constraints = np.zeros((n_scorecards, n_thresholds, n_scorecards, n_thresholds), dtype=int)
    for i in range(n_scorecards):
        for j in range(n_thresholds):
            for k in range(n_thresholds):
                if j != k:
                    constraints[i, j, i, k] = 1
    return constraints.flatten(), np.full(n_scorecards * n_thresholds, 1)

# Define the constraints to ensure that the total number of selected scorecards is equal to 1
def scorecard_constraints(variables):
    constraints = np.zeros((n_scorecards, n_thresholds, n_scorecards, n_thresholds), dtype=int)
    for i in range(n_scorecards):
        for j in range(n_thresholds):
            for k in range(n_scorecards):
                if i != k:
                    constraints[i, j, k, j] = 1
    return constraints.flatten(), np.full(n_scorecards * n_thresholds, 1)

# Convert the constraints to penalties and add them to the objective function
def penalties(variables):
    threshold_constraints_matrix, threshold_constraints_weights = threshold_constraints(variables)
    scorecard_constraints_matrix, scorecard_constraints_weights = scorecard_constraints(variables)
    penalty = 1000
    return penalty * (np.dot(threshold_constraints_matrix, threshold_constraints_weights) + np.dot(scorecard_constraints_matrix, scorecard_constraints_weights))

# Convert the objective function and constraints to QUBO form
def qubo(variables):
    q = {}
    for i in range(n_scorecards * n_thresholds):
        q[(i, i)] = objective(variables) + penalties(variables)
        for j in range(i + 1, n_scorecards * n_thresholds):
            q[(i, j)] = 2 * Q[(i, j)]
    return q


# Define the QUBO problem
qubo = qubo(variables)

# Convert the QUBO problem to the format expected by the D-Wave API
bqm = dwavebinarycsp.stitch(qubo)

# Solve the QUBO problem using the D-Wave quantum annealer
sampler = LeapHybridSampler()
sampleset = sampler.sample(bqm, num_reads=100)

# Print the top solution
top_solution = sampleset.first.sample
print(top_solution)