# main.py
# This is the main script to run the knapsack problem solver


# Import the KnapsackProblem class from knapsack.py
from knapsack import KnapsackProblem

# Import the Solution class from knapsack.py
from knapsack import Solution

# Import the repairsolution function from knapsack.py
from knapsack import repairsolution

# Import the solvegreedy function from baseline.py
from baseline import solvegreedy


# Print a starting message
print("Starting program")
print()

# Define the path to the problem file
problemfile = 'problems/knapPI_11_50_1000.csv'

# Create an instance of the knapsack problem
myproblem = KnapsackProblem(problemfile)

# Print the loaded problem details
print("Loaded problem with n=" + str(myproblem.n) + " items and W=" + str(myproblem.W) + " capacity")
print()

# Run the greedy baseline solver
greedysolution = solvegreedy(myproblem)

# Print the greedy baseline results
print("--- Greedy Baseline Solution ---")
print("Total Value: " + str(greedysolution.totalvalue))
print("Total Weight: " + str(greedysolution.totalweight))
print("Is Feasible: " + str(greedysolution.isfeasible))
print()

# Test the repair function
print("Testing repair function on an invalid solution...")

# Create an invalid solution with all items selected
testvector = []
# Loop to add all ones
for i in range(myproblem.n):
    # Add a one to the list
    testvector.append(1)

# Call the repair function
repairedvector = repairsolution(testvector, myproblem)

# Create a Solution object from the repaired vector
repairedsol = Solution(repairedvector, myproblem)

# Print the repaired solution results
print("--- Repaired Solution ---")
print("Repaired Value: " + str(repairedsol.totalvalue))
print("Repaired Weight: " + str(repairedsol.totalweight))
print("Is Feasible: " + str(repairedsol.isfeasible))
