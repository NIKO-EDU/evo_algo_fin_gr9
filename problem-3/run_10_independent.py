# run_10_independent.py
# Script to run all problems 10 times with different random seeds

import random
from knapsack import KnapsackProblem
from baseline import solvegreedy
from basolver import BeesAlgorithm


def run_single_experiment(problem, seed):
    """Run Bees Algorithm once with given seed."""
    random.seed(seed)
    
    # Determine parameters based on problem size
    if problem.n <= 50:
        ns = 50
        nre = 5
        nrb = 15
        nbe = 10
        nbb = 5
        maxiter = 1000
    elif problem.n <= 100:
        ns = 60
        nre = 6
        nrb = 18
        nbe = 8
        nbb = 4
        maxiter = 500
    elif problem.n <= 500:
        ns = 80
        nre = 8
        nrb = 24
        nbe = 6
        nbb = 3
        maxiter = 300
    else:
        ns = 100
        nre = 10
        nrb = 30
        nbe = 5
        nbb = 2
        maxiter = 200
    
    # Create and run solver
    solver = BeesAlgorithm(problem, ns, nre, nrb, nbe, nbb, maxiter)
    bestsolution, history = solver.run()
    
    return bestsolution.totalvalue, bestsolution.totalweight


def run_10_times(problemfile):
    """Run experiment 10 times with different seeds."""
    print("=" * 70)
    print("Processing:", problemfile)
    print("=" * 70)
    
    # Load problem
    problem = KnapsackProblem('problems/' + problemfile)
    
    print("Problem: n=" + str(problem.n) + ", W=" + str(problem.W))
    
    # Get greedy baseline
    greedy_solution = solvegreedy(problem)
    greedy_value = greedy_solution.totalvalue
    greedy_weight = greedy_solution.totalweight
    
    print("Greedy baseline: value=" + str(greedy_value) + ", weight=" + str(greedy_weight))
    print()
    print("Running 10 independent trials...")
    
    # Run 10 times with different seeds
    results = []
    for seed in range(10):
        print("  Run " + str(seed + 1) + "/10 (seed=" + str(seed) + ")...", end=" ")
        value, weight = run_single_experiment(problem, seed)
        results.append(value)
        print("value=" + str(value))
    
    # Calculate statistics
    best_value = max(results)
    worst_value = min(results)
    mean_value = sum(results) / len(results)
    
    # Calculate standard deviation
    variance = sum((x - mean_value) ** 2 for x in results) / len(results)
    std_dev = variance ** 0.5
    
    print()
    print("Statistics:")
    print("  Best:  " + str(best_value))
    print("  Worst: " + str(worst_value))
    print("  Mean:  " + str(round(mean_value, 2)))
    print("  Std:   " + str(round(std_dev, 2)))
    print()
    
    return {
        'file': problemfile,
        'n': problem.n,
        'W': problem.W,
        'greedy_value': greedy_value,
        'greedy_weight': greedy_weight,
        'best': best_value,
        'worst': worst_value,
        'mean': mean_value,
        'std': std_dev,
        'all_runs': results
    }


def main():
    """Run all problems 10 times each and generate statistics."""
    
    print()
    print("=" * 70)
    print("RUNNING 10 INDEPENDENT TRIALS FOR ALL PROBLEMS")
    print("=" * 70)
    print()
    print("This will take approximately 30-60 minutes...")
    print()
    
    # List of problem files
    problem_files = [
        'knapPI_11_20_1000.csv',
        'knapPI_11_50_1000.csv',
        'knapPI_11_100_1000.csv',
        'knapPI_11_200_1000.csv',
        'knapPI_11_500_1000.csv',
        'knapPI_11_1000_1000.csv',
        'knapPI_11_2000_1000.csv'
    ]
    
    # Run all experiments
    all_results = []
    for pfile in problem_files:
        result = run_10_times(pfile)
        all_results.append(result)
        print()
    
    # Print summary table
    print()
    print("=" * 90)
    print("SUMMARY TABLE: 10-RUN STATISTICS")
    print("=" * 90)
    print()
    
    print("Instance          | n    | W       | Greedy | Best   | Worst  | Mean   | Std    |")
    print("------------------|------|---------|--------|--------|--------|--------|--------|")
    
    for r in all_results:
        instance_name = r['file'].replace('knapPI_11_', '').replace('_1000.csv', '')
        
        print(
            instance_name.ljust(17) + " | " +
            str(r['n']).rjust(4) + " | " +
            str(r['W']).rjust(7) + " | " +
            str(r['greedy_value']).rjust(6) + " | " +
            str(r['best']).rjust(6) + " | " +
            str(r['worst']).rjust(6) + " | " +
            str(int(r['mean'])).rjust(6) + " | " +
            str(round(r['std'], 1)).rjust(6) + " |"
        )
    
    print()
    print()
    
    # Print LaTeX table format
    print("=" * 90)
    print("LATEX TABLE FORMAT (copy to report)")
    print("=" * 90)
    print()
    
    print("% --- TABLE: 10-RUN STATISTICS ---")
    print("\\begin{table}[H]")
    print("    \\centering")
    print("    \\caption{Statistical Analysis over 10 Independent Runs with Different Random Seeds}")
    print("    \\label{tab:stats}")
    print("    \\begin{tabular}{lrrrr}")
    print("        \\toprule")
    print("        \\textbf{Instance} & \\textbf{Best} & \\textbf{Worst} & \\textbf{Mean} & \\textbf{Std. Dev} \\\\")
    print("        \\midrule")
    
    for r in all_results:
        instance_name = r['file'].replace('knapPI_11_', 'knapPI\\_').replace('_1000.csv', '')
        best_str = "{:,}".format(r['best']).replace(",", ",")
        worst_str = "{:,}".format(r['worst']).replace(",", ",")
        mean_str = "{:,}".format(int(r['mean'])).replace(",", ",")
        std_str = str(round(r['std'], 1))
        
        print(
            "        " + instance_name.ljust(20) + " & " +
            best_str.rjust(7) + " & " +
            worst_str.rjust(7) + " & " +
            mean_str.rjust(7) + " & " +
            std_str.rjust(6) + " \\\\"
        )
    
    print("        \\bottomrule")
    print("    \\end{tabular}")
    print("\\end{table}")
    print()
    
    # Save results to file
    output_file = open('10_runs_results.txt', 'w')
    output_file.write("10 INDEPENDENT RUNS RESULTS\n")
    output_file.write("=" * 90 + "\n\n")
    
    for r in all_results:
        output_file.write("Problem: " + r['file'] + "\n")
        output_file.write("  n=" + str(r['n']) + ", W=" + str(r['W']) + "\n")
        output_file.write("  Greedy: " + str(r['greedy_value']) + "\n")
        output_file.write("  Best:   " + str(r['best']) + "\n")
        output_file.write("  Worst:  " + str(r['worst']) + "\n")
        output_file.write("  Mean:   " + str(round(r['mean'], 2)) + "\n")
        output_file.write("  Std:    " + str(round(r['std'], 2)) + "\n")
        output_file.write("  All 10 runs: " + str(r['all_runs']) + "\n")
        output_file.write("\n")
    
    output_file.close()
    
    print("Results saved to: 10_runs_results.txt")
    print()
    print("=" * 90)
    print("ALL EXPERIMENTS COMPLETED!")
    print("=" * 90)


if __name__ == "__main__":
    main()
