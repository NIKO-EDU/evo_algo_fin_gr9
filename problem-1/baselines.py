"""
Baselines for the 1D Bin Packing problem.

Currently includes a First-Fit Decreasing (FFD) heuristic implementation.
"""

from typing import Dict, List


def first_fit_decreasing(item_sizes: List[int], box_capacity: int) -> Dict:
    """
    First-Fit Decreasing (FFD) heuristic for 1D bin packing.

    Steps:
    1. Sort items in non-increasing order.
    2. For each item, place it in the first box that can accommodate it.
    3. If no box can accommodate it, open a new box.
    """
    # Keep track of original indices so we can reconstruct assignments
    sorted_items = sorted(enumerate(item_sizes), key=lambda x: x[1], reverse=True)

    box_loads: List[int] = []
    assignment: Dict[int, int] = {}

    for original_idx, size in sorted_items:
        placed = False
        for box_idx, load in enumerate(box_loads):
            if load + size <= box_capacity:
                box_loads[box_idx] += size
                assignment[original_idx] = box_idx
                placed = True
                break

        if not placed:
            box_loads.append(size)
            assignment[original_idx] = len(box_loads) - 1

    num_boxes = len(box_loads)
    unused_capacity = sum(box_capacity - load for load in box_loads)

    return {
        "assignment": assignment,
        "box_loads": box_loads,
        "num_boxes": num_boxes,
        "unused_capacity": unused_capacity,
    }


