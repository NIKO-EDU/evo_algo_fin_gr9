"""
Ant Colony Optimization (ACO) for One-Dimensional Bin Packing Problem

This module implements ACO to solve the bin packing problem:
- Given items with sizes w_1, ..., w_n and box capacity C
- Assign each item to a box to minimize the number of boxes used
- No box can exceed capacity C
"""

import numpy as np
from typing import List, Tuple, Dict
import random


class ACOBinPacking:
    """
    Ant Colony Optimization solver for the bin packing problem.
    
    The algorithm uses:
    - Pheromone trails (tau): tracks how good it is to assign item i to box b
    - Heuristic information (eta): prefers tight fits (small remaining capacity)
    - Ant construction: each ant builds a solution probabilistically
    - Pheromone update: best solution reinforces, others evaporate
    """
    
    def __init__(
        self,
        item_sizes: List[int],
        box_capacity: int,
        n_ants: int = 10,
        n_iterations: int = 100,
        alpha: float = 1.0,  # Pheromone importance
        beta: float = 2.0,   # Heuristic importance
        rho: float = 0.1,    # Evaporation rate
        Q: float = 1.0,      # Pheromone deposit amount
        tau_init: float = 0.1,  # Initial pheromone level
        seed: int = None     # Random seed for reproducibility
    ):
        """
        Initialize ACO solver.
        
        Parameters:
        -----------
        item_sizes : List[int]
            Sizes of items to pack
        box_capacity : int
            Maximum capacity of each box
        n_ants : int
            Number of ants (solutions) per iteration
        n_iterations : int
            Number of iterations to run
        alpha : float
            Weight for pheromone trail (higher = more exploitation)
        beta : float
            Weight for heuristic information (higher = more exploration)
        rho : float
            Evaporation rate (0 < rho < 1)
        Q : float
            Pheromone deposit quantity
        tau_init : float
            Initial pheromone level
        """
        self.item_sizes = np.array(item_sizes)
        self.n_items = len(item_sizes)
        self.box_capacity = box_capacity
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        
        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Sort items in decreasing order (common heuristic for bin packing)
        self.sorted_indices = np.argsort(item_sizes)[::-1]
        self.sorted_sizes = self.item_sizes[self.sorted_indices]
        
        # Pheromone matrix: tau[i][b] = pheromone for assigning item i to box b
        # We'll use a dynamic approach where boxes are created as needed
        # For efficiency, we'll use a dictionary or limit max boxes
        self.max_boxes = self.n_items  # Worst case: one item per box
        self.tau = np.ones((self.n_items, self.max_boxes)) * tau_init
        
        # Track best solution found
        self.best_solution = None
        self.best_num_boxes = float('inf')
        self.best_unused_capacity = float('inf')
        
        # Track convergence
        self.convergence_history = []
        
    def _heuristic(self, item_size: int, box_load: int) -> float:
        """
        Heuristic function: prefers tight fits.
        
        This is a key component of ACO - it guides ants to make good decisions.
        Returns higher value when remaining capacity is small (tight fit).
        This encourages placing items in boxes with little space left, leading
        to more efficient packing.
        
        Example: If box has 50 capacity and 40 load, placing a size-10 item
        gives remaining=0 (perfect fit) -> returns 1.0 (best).
        
        Parameters:
        -----------
        item_size : int
            Size of the item to place
        box_load : int
            Current load of the box
            
        Returns:
        --------
        float : Heuristic value (higher = better, range 0.0 to 1.0)
        """
        remaining = self.box_capacity - box_load
        if remaining < item_size:
            return 0.0  # Cannot fit
        
        # Prefer tight fits: if remaining capacity is small after placing item
        # This encourages efficient packing
        if remaining == item_size:
            return 1.0  # Perfect fit
        else:
            # Normalize: prefer smaller remaining capacity
            # Use inverse of remaining capacity (with smoothing)
            return 1.0 / (remaining - item_size + 1.0)
    
    def _probability(self, item_idx: int, box_idx: int, box_loads: List[int]) -> float:
        """
        Calculate probability of assigning item to box.
        
        This is the core of ACO decision-making. Uses the formula:
        P ∝ (tau^alpha) * (eta^beta)
        
        Where:
        - tau = pheromone trail (what worked well before)
        - eta = heuristic value (tight fit preference)
        - alpha = weight for pheromone (exploitation)
        - beta = weight for heuristic (exploration)
        
        Higher alpha = trust past experience more
        Higher beta = trust heuristic more
        
        Parameters:
        -----------
        item_idx : int
            Index of item (in sorted order)
        box_idx : int
            Index of box
        box_loads : List[int]
            Current loads of all boxes
            
        Returns:
        --------
        float : Probability value (0.0 if infeasible, >0.0 otherwise)
        """
        item_size = self.sorted_sizes[item_idx]
        box_load = box_loads[box_idx]
        
        # Check feasibility
        if box_load + item_size > self.box_capacity:
            return 0.0
        
        # Get pheromone and heuristic
        tau_value = self.tau[item_idx, box_idx]
        eta_value = self._heuristic(item_size, box_load)
        
        # Calculate probability component
        prob = (tau_value ** self.alpha) * (eta_value ** self.beta)
        return prob
    
    def _construct_solution(self) -> Tuple[Dict[int, int], int, int]:
        """
        Construct a solution using one ant.
        
        This is where each ant builds a complete packing solution:
        1. Process items in decreasing order (largest first)
        2. For each item, calculate probabilities for all feasible boxes
        3. Use roulette wheel selection to choose a box probabilistically
        4. Assign item to chosen box (or create new box if needed)
        
        The probabilistic selection balances exploration (trying new things)
        and exploitation (using what worked before).
        
        Returns:
        --------
        Tuple[Dict[int, int], int, int]
            (assignment, num_boxes, unused_capacity)
            - assignment: dict mapping item_idx -> box_idx
            - num_boxes: total number of boxes used
            - unused_capacity: sum of unused space across all boxes
        """
        assignment = {}  # item_idx -> box_idx
        box_loads = []  # Current load of each box
        
        # Process items in decreasing order
        for item_idx in range(self.n_items):
            item_size = self.sorted_sizes[item_idx]
            
            # Get feasible boxes and their probabilities
            feasible_boxes = []
            probabilities = []
            
            # Check existing boxes
            for box_idx in range(len(box_loads)):
                if box_loads[box_idx] + item_size <= self.box_capacity:
                    prob = self._probability(item_idx, box_idx, box_loads)
                    feasible_boxes.append(box_idx)
                    probabilities.append(prob)
            
            # Always allow creating a new box
            new_box_idx = len(box_loads)
            prob_new = self._probability(item_idx, new_box_idx, box_loads + [0])
            feasible_boxes.append(new_box_idx)
            probabilities.append(prob_new)
            
            # Select box using roulette wheel selection
            if not feasible_boxes:
                # Should not happen, but create new box as fallback
                box_idx = len(box_loads)
                box_loads.append(item_size)
                assignment[item_idx] = box_idx
            else:
                # Normalize probabilities
                prob_sum = sum(probabilities)
                if prob_sum > 0:
                    probabilities = [p / prob_sum for p in probabilities]
                    box_idx = np.random.choice(feasible_boxes, p=probabilities)
                else:
                    # Fallback: choose randomly among feasible
                    box_idx = random.choice(feasible_boxes)
                
                # Assign item to box
                if box_idx == len(box_loads):
                    # Create new box
                    box_loads.append(item_size)
                else:
                    # Add to existing box
                    box_loads[box_idx] += item_size
                
                assignment[item_idx] = box_idx
        
        # Calculate metrics
        num_boxes = len(box_loads)
        unused_capacity = sum(self.box_capacity - load for load in box_loads)
        
        return assignment, num_boxes, unused_capacity
    
    def _update_pheromone(self, best_assignment: Dict[int, int], best_num_boxes: int):
        """
        Update pheromone trails after each iteration.
        
        This is how ACO learns from experience:
        1. EVAPORATION: All pheromone decreases by factor (1 - rho)
           - Prevents getting stuck in local optima
           - Higher rho = faster forgetting of old solutions
        2. REINFORCEMENT: Best solution adds pheromone to its edges
           - tau[i][b] += Q / num_boxes
           - Better solutions (fewer boxes) deposit more pheromone per edge
           - Guides future ants toward good solutions
        
        Parameters:
        -----------
        best_assignment : Dict[int, int]
            Best assignment found in this iteration (item_idx -> box_idx)
        best_num_boxes : int
            Number of boxes in best solution (used to calculate deposit amount)
        """
        # Evaporation
        self.tau = (1 - self.rho) * self.tau
        
        # Reinforcement: add pheromone to edges in best solution
        if best_num_boxes < float('inf'):
            deposit = self.Q / best_num_boxes  # More boxes = less deposit per edge
            
            for item_idx, box_idx in best_assignment.items():
                self.tau[item_idx, box_idx] += deposit
    
    def solve(self) -> Dict:
        """
        Run the complete ACO algorithm to solve the bin packing problem.
        
        Main ACO loop:
        1. For each iteration:
           a. All ants construct solutions independently
           b. Find best solution among all ants
           c. Update global best if improved
           d. Update pheromone trails (evaporate + reinforce)
           e. Track convergence
        2. Return best solution found
        
        The algorithm iteratively improves by learning which item-to-box
        assignments work well, gradually converging toward good solutions.
        
        Returns:
        --------
        Dict with solution details:
            - 'assignment': dict mapping original item indices to box indices
            - 'num_boxes': number of boxes used (minimized)
            - 'unused_capacity': total unused capacity across all boxes
            - 'box_loads': list of loads for each box
            - 'convergence': list of best num_boxes per iteration (for plotting)
        """
        print(f"Starting ACO with {self.n_ants} ants for {self.n_iterations} iterations...")
        
        for iteration in range(self.n_iterations):
            # All ants construct solutions
            solutions = []
            for ant in range(self.n_ants):
                assignment, num_boxes, unused_capacity = self._construct_solution()
                solutions.append((assignment, num_boxes, unused_capacity))
            
            # Find best solution in this iteration
            iteration_best = min(solutions, key=lambda x: (x[1], x[2]))  # Minimize boxes, then unused capacity
            best_assignment, best_num_boxes, best_unused = iteration_best
            
            # Update global best
            if (best_num_boxes < self.best_num_boxes or 
                (best_num_boxes == self.best_num_boxes and best_unused < self.best_unused_capacity)):
                self.best_solution = best_assignment
                self.best_num_boxes = best_num_boxes
                self.best_unused_capacity = best_unused
            
            # Update pheromone using global best
            if self.best_solution is not None:
                self._update_pheromone(self.best_solution, self.best_num_boxes)
            
            # Track convergence
            self.convergence_history.append(self.best_num_boxes)
            
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.n_iterations}: Best = {self.best_num_boxes} boxes")
        
        # Convert solution back to original item indices
        final_assignment = {}
        box_loads = [0] * self.best_num_boxes
        
        for sorted_idx, box_idx in self.best_solution.items():
            original_idx = self.sorted_indices[sorted_idx]
            final_assignment[original_idx] = box_idx
            box_loads[box_idx] += self.item_sizes[original_idx]
        
        return {
            'assignment': final_assignment,
            'num_boxes': self.best_num_boxes,
            'unused_capacity': self.best_unused_capacity,
            'box_loads': box_loads,
            'convergence': self.convergence_history
        }

