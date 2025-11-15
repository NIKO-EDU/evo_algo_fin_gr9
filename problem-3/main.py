# main.py

from knapsack import KnapsackProblem
from knapsack import Solution
from knapsack import repairsolution
from baseline import solvegreedy
from basolver import BeesAlgorithm


print("Starting program")
print()

problemfile = 'problems/knapPI_11_50_1000.csv'
myproblem = KnapsackProblem(problemfile)

print("Loaded problem with n=" + str(myproblem.n) + " items and W=" + str(myproblem.W) + " capacity")
print()

# Run greedy baseline solver
greedysolution = solvegreedy(myproblem)

print("--- Greedy Baseline Solution ---")
print("Total Value: " + str(greedysolution.totalvalue))
print("Total Weight: " + str(greedysolution.totalweight))
print("Is Feasible: " + str(greedysolution.isfeasible))
print()

# Run Bees Algorithm solver
print("--- Running Bees Algorithm Solver ---")

# Define algorithm settings
settingsns = 50
settingsnre = 5
settingsnrb = 15
settingsnbe = 10
settingsnbb = 5
settingsmaxiter = 1000

# Create the Bees Algorithm solver
solver = BeesAlgorithm(myproblem, settingsns, settingsnre, settingsnrb, settingsnbe, settingsnbb, settingsmaxiter)

# Run the solver
bestsolution, history = solver.run()

# Print the final results
print()
print("--- Bees Algorithm Final Result ---")
print("Best Value Found: " + str(bestsolution.totalvalue))
print("Best Weight Used: " + str(bestsolution.totalweight))
print("Improvement vs Greedy: " + str(bestsolution.totalvalue - greedysolution.totalvalue))

# The history list will be used for plots later
