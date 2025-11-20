# plotting.py
# Script for plotting the Bees Algorithm convergence and comparing with baseline

import matplotlib.pyplot as plt


def plotconvergence(history, problemname, filename):
    """
    Plots the convergence history of the Bees Algorithm.
    """
    # Create a new figure
    fig = plt.figure()
    
    # Create the plot
    plt.plot(history, linewidth=2, color='blue')
    
    # Add title and labels
    plt.title('Bees Algorithm Convergence: ' + problemname)
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness Value')
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    plt.savefig(filename)
    print("Plot saved to: " + filename)
    
    # Close the figure
    plt.close(fig)


def plotcomparison(greedyvalue, beesvalue, problemname, filename):
    """
    Plots a comparison between Greedy and Bees Algorithm results.
    """
    # Create a new figure
    fig = plt.figure()
    
    # Define the algorithm names
    algorithms = ['Greedy', 'Bees Algorithm']
    
    # Define the values
    values = [greedyvalue, beesvalue]
    
    # Create a bar plot
    bars = plt.bar(algorithms, values, color=['orange', 'blue'], width=0.6)
    
    # Add title and labels
    plt.title('Algorithm Comparison: ' + problemname)
    plt.ylabel('Total Value Found')
    
    # Add value labels on bars
    for i in range(len(bars)):
        height = bars[i].get_height()
        plt.text(bars[i].get_x() + bars[i].get_width() / 2.0, height,
                 str(int(values[i])), ha='center', va='bottom')
    
    # Save the figure
    plt.savefig(filename)
    print("Plot saved to: " + filename)
    
    # Close the figure
    plt.close(fig)


def plotdetailed(history, greedyvalue, beesvalue, problemname, outputdir):
    """
    Creates detailed plots with multiple subplots.
    """
    # Create a figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # SUBPLOT 1: Convergence curve
    axes[0].plot(history, linewidth=2, color='blue')
    axes[0].set_title('Convergence History')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Best Fitness Value')
    axes[0].grid(True, alpha=0.3)
    
    # SUBPLOT 2: Comparison bar chart
    algorithms = ['Greedy', 'Bees Algorithm']
    values = [greedyvalue, beesvalue]
    bars = axes[1].bar(algorithms, values, color=['orange', 'blue'], width=0.6)
    axes[1].set_title('Algorithm Comparison')
    axes[1].set_ylabel('Total Value Found')
    
    # Add value labels on bars
    for i in range(len(bars)):
        height = bars[i].get_height()
        axes[1].text(bars[i].get_x() + bars[i].get_width() / 2.0, height,
                     str(int(values[i])), ha='center', va='bottom')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    filename = outputdir + '/comparison_' + problemname + '.png'
    plt.savefig(filename)
    print("Detailed plot saved to: " + filename)
    
    # Close the figure
    plt.close(fig)
