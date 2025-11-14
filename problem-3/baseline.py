# baseline.py
# This file contains the greedy baseline solver for the knapsack problem


def solvegreedy(problem):
    """
    This function implements a greedy baseline solver.
    It adds items to the knapsack in order of highest value-to-weight ratio first.
    """
    # Create a list to store all items with their indices and ratios
    allitems = []
    
    # Loop through each item by index
    for i in range(problem.n):
        # Get the ratio of item i
        ratio = problem.ratios[i]
        # Add a tuple of (index, ratio) to the list
        allitems.append((i, ratio))
    
    # Sort the items by ratio from highest to lowest
    # We use reverse=True to sort in descending order
    allitems.sort(key=lambda x: x[1], reverse=True)
    
    # Create an empty solution vector (all zeros)
    finalvector = []
    # Loop to create n zeros
    for i in range(problem.n):
        # Add a zero to the list
        finalvector.append(0)
    
    # Initialize the current weight to zero
    currentweight = 0
    
    # Loop through the sorted list of items
    for item in allitems:
        # Get the item index
        itemindex = item[0]
        
        # Get the weight of this item
        itemweight = problem.weights[itemindex]
        
        # Calculate what the weight would be if we add this item
        newweight = currentweight + itemweight
        
        # Check if adding this item would exceed the capacity
        if newweight <= problem.W:
            # The item fits, so add it to the knapsack
            finalvector[itemindex] = 1
            # Update the current weight
            currentweight = newweight
    
    # Import the Solution class
    from knapsack import Solution
    
    # Create a Solution object from the final vector
    finalsolution = Solution(finalvector, problem)
    
    # Return the solution
    return finalsolution
