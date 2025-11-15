# baseline.py

def solvegreedy(problem):
    """
    Greedy baseline solver.
    Adds items to the knapsack in order of highest value-to-weight ratio first.
    """
    allitems = []
    for i in range(problem.n):
        ratio = problem.ratios[i]
        allitems.append((i, ratio))
    
    # Sort by ratio (highest first)
    allitems.sort(key=lambda x: x[1], reverse=True)
    
    finalvector = []
    for i in range(problem.n):
        finalvector.append(0)
    
    currentweight = 0
    
    for item in allitems:
        itemindex = item[0]
        itemweight = problem.weights[itemindex]
        newweight = currentweight + itemweight
        
        if newweight <= problem.W:
            finalvector[itemindex] = 1
            currentweight = newweight
    
    from knapsack import Solution
    finalsolution = Solution(finalvector, problem)
    return finalsolution
