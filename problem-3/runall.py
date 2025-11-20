# runall.py
# Script to run all problem files in the problems/ folder

import os
from knapsack import KnapsackProblem
from baseline import solvegreedy
from basolver import BeesAlgorithm
from plotting import plotconvergence
from plotting import plotcomparison
from plotting import plotdetailed


def runproblem(filepath, problemname):
    """Runs the complete solver pipeline for one problem file."""
    
    print("=" * 70)
    print("Running problem: " + problemname)
    print("=" * 70)
    print()
    
    # Load the problem
    myproblem = KnapsackProblem(filepath)
    
    print("Loaded problem with n=" + str(myproblem.n) + " items and W=" + str(myproblem.W) + " capacity")
    print()
    
    # Run greedy baseline solver
    print("Running Greedy Baseline...")
    greedysolution = solvegreedy(myproblem)
    
    print("--- Greedy Baseline Solution ---")
    print("Total Value: " + str(greedysolution.totalvalue))
    print("Total Weight: " + str(greedysolution.totalweight))
    print("Is Feasible: " + str(greedysolution.isfeasible))
    print()
    
    # Run Bees Algorithm solver
    print("--- Running Bees Algorithm Solver ---")
    
    # Define algorithm settings based on problem size
    if myproblem.n <= 50:
        settingsns = 50
        settingsnre = 5
        settingsnrb = 15
        settingsnbe = 10
        settingsnbb = 5
        settingsmaxiter = 1000
    elif myproblem.n <= 100:
        settingsns = 60
        settingsnre = 6
        settingsnrb = 18
        settingsnbe = 8
        settingsnbb = 4
        settingsmaxiter = 500
    elif myproblem.n <= 500:
        settingsns = 80
        settingsnre = 8
        settingsnrb = 24
        settingsnbe = 6
        settingsnbb = 3
        settingsmaxiter = 300
    else:
        settingsns = 100
        settingsnre = 10
        settingsnrb = 30
        settingsnbe = 5
        settingsnbb = 2
        settingsmaxiter = 200
    
    # Create the Bees Algorithm solver
    solver = BeesAlgorithm(myproblem, settingsns, settingsnre, settingsnrb, settingsnbe, settingsnbb, settingsmaxiter)
    
    # Run the solver
    bestsolution, history = solver.run()
    
    # Print the final results
    print()
    print("--- Bees Algorithm Final Result ---")
    print("Best Value Found: " + str(bestsolution.totalvalue))
    print("Best Weight Used: " + str(bestsolution.totalweight))
    improvement = bestsolution.totalvalue - greedysolution.totalvalue
    print("Improvement vs Greedy: " + str(improvement))
    
    if improvement > 0:
        percentageimprovement = (improvement / greedysolution.totalvalue) * 100
        print("Improvement percentage: " + str(round(percentageimprovement, 2)) + "%")
    
    print()
    print("Generating plots...")
    
    # Create plot filenames based on problem name
    plotbasename = problemname.replace('.csv', '').replace('knapPI_11_', '')
    
    # Create convergence plot
    plotconvergence(history, problemname, 'plots/convergence_' + plotbasename + '.png')
    
    # Create comparison plot
    plotcomparison(greedysolution.totalvalue, bestsolution.totalvalue, problemname, 'plots/comparison_' + plotbasename + '.png')
    
    # Create detailed combined plot
    plotdetailed(history, greedysolution.totalvalue, bestsolution.totalvalue, problemname, 'plots')
    
    print()
    print()
    
    return {
        'problem': problemname,
        'items': myproblem.n,
        'capacity': myproblem.W,
        'greedy_value': greedysolution.totalvalue,
        'bees_value': bestsolution.totalvalue,
        'improvement': improvement,
        'feasible': bestsolution.isfeasible
    }


def main():
    """Main function that runs all problems."""
    
    print()
    print("=" * 70)
    print("RUNNING ALL KNAPSACK PROBLEMS")
    print("=" * 70)
    print()
    
    # Get all CSV files in problems folder
    problemsfolder = 'problems'
    allfiles = os.listdir(problemsfolder)
    
    # Filter to only CSV files and sort them
    csvfiles = []
    for file in allfiles:
        if file.endswith('.csv'):
            csvfiles.append(file)
    
    csvfiles.sort()
    
    print("Found " + str(len(csvfiles)) + " problem files:")
    for i in range(len(csvfiles)):
        print(str(i + 1) + ". " + csvfiles[i])
    print()
    
    # Run all problems and store results
    results = []
    for csvfile in csvfiles:
        filepath = problemsfolder + '/' + csvfile
        result = runproblem(filepath, csvfile)
        results.append(result)
    
    # Print summary table
    print()
    print("=" * 70)
    print("SUMMARY OF ALL RESULTS")
    print("=" * 70)
    print()
    
    print("Problem Name" + " " * 20 + "| Items | Greedy | Bees  | Improvement | Feasible")
    print("-" * 90)
    
    for result in results:
        problemname = result['problem']
        items = result['items']
        greedyvalue = result['greedy_value']
        beesvalue = result['bees_value']
        improvement = result['improvement']
        feasible = result['feasible']
        
        # Truncate problem name if too long
        if len(problemname) > 30:
            displayname = problemname[:27] + "..."
        else:
            displayname = problemname
        
        print(displayname.ljust(30) + " | " + str(items).rjust(5) + " | " + 
              str(greedyvalue).rjust(6) + " | " + str(beesvalue).rjust(5) + " | " + 
              str(improvement).rjust(11) + " | " + str(feasible))
    
    print()
    print("Summary completed! All plots saved to plots/ folder.")
    print()


if __name__ == "__main__":
    main()
