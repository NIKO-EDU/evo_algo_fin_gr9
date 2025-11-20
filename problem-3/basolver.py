# basolver.py

import random
from knapsack import KnapsackProblem
from knapsack import Solution
from knapsack import repairsolution


class BeesAlgorithm:
    """
    Implements the Bees Algorithm for solving the 0/1 Knapsack Problem.
    """
    
    def __init__(self, problem, ns, nre, nrb, nbe, nbb, maxiter):
        # Store the problem instance
        self.problem = problem
        # Store the total number of bees in the population
        self.ns = ns
        # Store the number of elite bees
        self.nre = nre
        # Store the number of best bees
        self.nrb = nrb
        # Store the number of neighbors for elite bees
        self.nbe = nbe
        # Store the number of neighbors for best bees
        self.nbb = nbb
        # Store the maximum number of iterations
        self.maxiter = maxiter
        # Initialize the population as an empty list
        self.population = []
        # Initialize the global best solution as None
        self.globalbest = None
    
    def createrandomsolution(self):
        """Creates one random solution vector."""
        # Create an empty vector
        vector = []
        # Loop through each item
        for i in range(self.problem.n):
            # Add a zero to the vector
            vector.append(0)
        
        # Randomly set some items to 1
        for i in range(self.problem.n):
            # Flip a coin to decide if item is included
            if random.random() < 0.5:
                vector[i] = 1
        
        # Return the random vector
        return vector
    
    def generateneighbor(self, solutionvector):
        """Generates a neighbor solution by flipping one random bit."""
        # Make a copy of the solution vector
        newvector = list(solutionvector)
        # Pick a random index
        index = random.randint(0, self.problem.n - 1)
        # Flip the bit at that index
        if newvector[index] == 0:
            newvector[index] = 1
        else:
            newvector[index] = 0
        # Return the neighbor vector
        return newvector
    
    def localsearch(self, originalsolution, numneighbors):
        """
        Performs local search around a solution.
        Explores numneighbors neighbors and returns the best one found.
        """
        # Start with the original solution as the best
        bestneighbor = originalsolution
        
        # Loop to explore neighbors
        for i in range(numneighbors):
            # Generate a neighbor vector
            neighborvector = self.generateneighbor(originalsolution.vector)
            # Repair the neighbor if it is infeasible
            repairedvector = repairsolution(neighborvector, self.problem)
            # Create a Solution object for the neighbor
            neighborsolution = Solution(repairedvector, self.problem)
            # Check if this neighbor is better than the current best
            if neighborsolution.fitness > bestneighbor.fitness:
                bestneighbor = neighborsolution
        
        # Return the best neighbor found
        return bestneighbor
    
    def run(self):
        """
        Runs the Bees Algorithm and returns the best solution found.
        """
        # Initialize list to track convergence history
        convergencehistory = []
        
        # Print starting message
        print("Starting Bees Algorithm")
        
        # STEP 1: Initialization - Create initial population
        for i in range(self.ns):
            # Create a random solution vector
            randomvector = self.createrandomsolution()
            # Repair the solution if needed
            repairedvector = repairsolution(randomvector, self.problem)
            # Create a Solution object
            bee = Solution(repairedvector, self.problem)
            # Add the bee to the population
            self.population.append(bee)
        
        # Find the best bee in the initial population
        self.globalbest = max(self.population, key=lambda x: x.fitness)
        # Print the initial best fitness
        print("Initial best fitness: " + str(self.globalbest.fitness))
        
        # STEP 2: Main iteration loop
        for iter in range(self.maxiter):
            # Sort the population by fitness (highest first)
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            
            # Create a new empty population
            newpopulation = []
            
            # ELITE BEES: Perform intensive local search
            for i in range(self.nre):
                # Get the elite bee
                originalbee = self.population[i]
                # Perform local search with more neighbors
                bestnewbee = self.localsearch(originalbee, self.nbe)
                # Add the result to new population
                newpopulation.append(bestnewbee)
            
            # BEST BEES: Perform less intensive local search
            # Only process if nrb > nre (should always be true, but check for safety)
            if self.nrb > self.nre:
                for i in range(self.nre, self.nrb):
                    # Get the best bee
                    originalbee = self.population[i]
                    # Perform local search with fewer neighbors
                    bestnewbee = self.localsearch(originalbee, self.nbb)
                    # Add the result to new population
                    newpopulation.append(bestnewbee)
            
            # SCOUT BEES: Explore new random solutions
            numscouts = self.ns - self.nrb
            for i in range(numscouts):
                # Create a random solution vector
                randomvector = self.createrandomsolution()
                # Repair the solution if needed
                repairedvector = repairsolution(randomvector, self.problem)
                # Create a Solution object
                scoutbee = Solution(repairedvector, self.problem)
                # Add the scout to new population
                newpopulation.append(scoutbee)
            
            # Update the population with the new one
            self.population = newpopulation
            
            # Find the best bee in the current population
            currentbest = max(self.population, key=lambda x: x.fitness)
            # Check if it is better than the global best
            if currentbest.fitness > self.globalbest.fitness:
                self.globalbest = currentbest
            
            # Record the current best fitness
            convergencehistory.append(self.globalbest.fitness)
            
            # Print status every 100 iterations
            if (iter + 1) % 100 == 0:
                print("Iteration " + str(iter + 1) + ": Best fitness = " + str(self.globalbest.fitness))
        
        # Print completion message
        print("Algorithm finished")
        
        # Return the best solution and the convergence history
        return self.globalbest, convergencehistory
