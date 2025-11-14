# knapsack.py
# This file contains the basic classes and functions for the 0/1 Knapsack Problem


class KnapsackProblem:
    """
    This class represents a knapsack problem instance.
    It loads data from a file and stores all problem information.
    """
    
    def __init__(self, filepath):
        # Initialize the number of items to zero
        self.n = 0
        
        # Initialize the capacity to zero
        self.W = 0
        
        # Initialize an empty list for item values
        self.values = []
        
        # Initialize an empty list for item weights
        self.weights = []
        
        # Initialize an empty list for value-to-weight ratios
        self.ratios = []
        
        # Load the data from the file
        self.loaddata(filepath)
    
    def loaddata(self, filepath):
        """
        This function reads a Pisinger format file and loads the problem data.
        """
        # Open the file for reading
        file = open(filepath, 'r')
        
        # Read all lines from the file
        lines = file.readlines()
        
        # Close the file
        file.close()
        
        # Loop through each line in the file
        for line in lines:
            # Remove whitespace from the beginning and end of the line
            line = line.strip()
            
            # Check if the line starts with 'n ' (number of items)
            if line.startswith('n '):
                # Split the line by spaces
                parts = line.split()
                # The second part is the number of items
                self.n = int(parts[1])
            
            # Check if the line starts with 'c ' (capacity)
            elif line.startswith('c '):
                # Split the line by spaces
                parts = line.split()
                # The second part is the capacity
                self.W = int(parts[1])
            
            # Check if the line contains a comma (data line)
            elif ',' in line:
                # Split the line by commas
                parts = line.split(',')
                # The second part is the value
                value = int(parts[1])
                # The third part is the weight
                weight = int(parts[2])
                # Add the value to the values list
                self.values.append(value)
                # Add the weight to the weights list
                self.weights.append(weight)
        
        # Now calculate the ratios for all items
        # Loop through each item by index
        for i in range(self.n):
            # Get the value of the current item
            value = self.values[i]
            # Get the weight of the current item
            weight = self.weights[i]
            # Calculate the ratio (value divided by weight)
            ratio = value / weight
            # Add the ratio to the ratios list
            self.ratios.append(ratio)


class Solution:
    """
    This class represents a solution to the knapsack problem.
    It stores the solution vector and calculates its metrics.
    """
    
    def __init__(self, vector, problem):
        # Store the solution vector (e.g., [0, 1, 1, 0])
        self.vector = vector
        
        # Store the problem object
        self.problem = problem
        
        # Initialize total value to zero
        self.totalvalue = 0
        
        # Initialize total weight to zero
        self.totalweight = 0
        
        # Initialize feasibility flag to False
        self.isfeasible = False
        
        # Initialize fitness to zero
        self.fitness = 0
        
        # Calculate the metrics for this solution
        self.calculatemetrics()
    
    def calculatemetrics(self):
        """
        This function calculates the total value, weight, and fitness of the solution.
        """
        # Reset total value to zero
        self.totalvalue = 0
        
        # Reset total weight to zero
        self.totalweight = 0
        
        # Loop through each item by index
        for i in range(self.problem.n):
            # Check if item i is in the knapsack
            if self.vector[i] == 1:
                # Add the value of item i to the total value
                self.totalvalue += self.problem.values[i]
                # Add the weight of item i to the total weight
                self.totalweight += self.problem.weights[i]
        
        # Check if the solution is feasible
        if self.totalweight <= self.problem.W:
            # The solution is feasible
            self.isfeasible = True
            # Set fitness equal to total value
            self.fitness = self.totalvalue
        else:
            # The solution is not feasible
            self.isfeasible = False
            # Set fitness to zero
            self.fitness = 0


def repairsolution(solutionvector, problem):
    """
    This function repairs an infeasible solution by removing items.
    It removes items with the lowest ratios first until the solution is feasible.
    """
    # Calculate the current total weight of the solution
    currentweight = 0
    # Loop through each item
    for i in range(problem.n):
        # Check if item i is in the knapsack
        if solutionvector[i] == 1:
            # Add the weight of item i to current weight
            currentweight += problem.weights[i]
    
    # Check if the solution is already feasible
    if currentweight <= problem.W:
        # The solution is already feasible, return it as is
        return solutionvector
    
    # The solution is infeasible, we need to repair it
    # Create a list to store items currently in the bag
    itemsinbag = []
    
    # Loop through each item by index
    for i in range(problem.n):
        # Check if item i is in the knapsack
        if solutionvector[i] == 1:
            # Get the ratio of item i
            ratio = problem.ratios[i]
            # Add a tuple of (index, ratio) to the list
            itemsinbag.append((i, ratio))
    
    # Sort items by ratio from lowest to highest
    # Items with lower ratios will be removed first
    itemsinbag.sort(key=lambda x: x[1])
    
    # Loop through the sorted items
    for item in itemsinbag:
        # Get the item index
        itemindex = item[0]
        
        # Remove the item from the bag
        solutionvector[itemindex] = 0
        
        # Subtract the weight of the removed item
        currentweight -= problem.weights[itemindex]
        
        # Check if the solution is now feasible
        if currentweight <= problem.W:
            # The solution is now feasible, stop removing items
            break
    
    # Return the repaired solution vector
    return solutionvector
