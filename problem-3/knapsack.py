# knapsack.py

class KnapsackProblem:
    """
    Represents a knapsack problem instance.
    Loads data from Pisinger format files and stores problem information.
    """
    
    def __init__(self, filepath):
        self.n = 0
        self.W = 0
        self.values = []
        self.weights = []
        self.ratios = []
        self.loaddata(filepath)
    
    def loaddata(self, filepath):
        """Reads a Pisinger format file and loads the problem data."""
        file = open(filepath, 'r')
        lines = file.readlines()
        file.close()
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('n '):
                parts = line.split()
                self.n = int(parts[1])
            
            elif line.startswith('c '):
                parts = line.split()
                self.W = int(parts[1])
            
            elif ',' in line:
                parts = line.split(',')
                value = int(parts[1])
                weight = int(parts[2])
                self.values.append(value)
                self.weights.append(weight)
        
        # Calculate value-to-weight ratios
        for i in range(self.n):
            value = self.values[i]
            weight = self.weights[i]
            # Avoid division by zero (shouldn't happen in valid knapsack problems)
            ratio = value / weight if weight > 0 else 0.0
            self.ratios.append(ratio)


class Solution:
    """
    Represents a solution to the knapsack problem.
    Stores the solution vector and calculates its metrics.
    """
    
    def __init__(self, vector, problem):
        self.vector = vector
        self.problem = problem
        self.totalvalue = 0
        self.totalweight = 0
        self.isfeasible = False
        self.fitness = 0
        self.calculatemetrics()
    
    def calculatemetrics(self):
        """Calculates the total value, weight, and fitness of the solution."""
        self.totalvalue = 0
        self.totalweight = 0
        
        for i in range(self.problem.n):
            if self.vector[i] == 1:
                self.totalvalue += self.problem.values[i]
                self.totalweight += self.problem.weights[i]
        
        if self.totalweight <= self.problem.W:
            self.isfeasible = True
            self.fitness = self.totalvalue
        else:
            self.isfeasible = False
            self.fitness = 0


def repairsolution(solutionvector, problem):
    """
    Repairs an infeasible solution by removing items with lowest ratios first.
    """
    currentweight = 0
    for i in range(problem.n):
        if solutionvector[i] == 1:
            currentweight += problem.weights[i]
    
    if currentweight <= problem.W:
        return solutionvector
    
    # Build list of items in bag with their ratios
    itemsinbag = []
    for i in range(problem.n):
        if solutionvector[i] == 1:
            ratio = problem.ratios[i]
            itemsinbag.append((i, ratio))
    
    # Sort by ratio (lowest first) and remove until feasible
    itemsinbag.sort(key=lambda x: x[1])
    
    for item in itemsinbag:
        itemindex = item[0]
        solutionvector[itemindex] = 0
        currentweight -= problem.weights[itemindex]
        
        if currentweight <= problem.W:
            break
    
    return solutionvector
