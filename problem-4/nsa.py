# nsa.py

import numpy as np

def match_r_contiguous(
    vec1: np.ndarray, 
    vec2: np.ndarray, 
    r: int
) -> bool:
    # Checks if two binary vectors match based on the r-contiguous bits rule.
    # A match occurs if the two vectors share at least 'r' identical bits in a consecutive sequence.
    # Find where the two vectors are identical element-wise.
    # This creates a boolean array, e.g., [True, False, True, True, True]
    comparison = (vec1 == vec2)
    
    consecutive_matches = 0
    for match in comparison:
        if match:
            consecutive_matches += 1
            if consecutive_matches >= r:
                return True # Found a long enough match, no need to check further
        else:
            # The sequence of matches is broken, reset the counter
            consecutive_matches = 0
            
    return False

def generate_detectors(
    encoded_self_set: list[np.ndarray], 
    num_detectors: int, 
    hash_size: int, 
    r_value: int
) -> tuple[list[np.ndarray], float]:
    # Generates a set of detectors using the Negative Selection Algorithm.
    # This is the "censoring" or training phase. Detectors are generated randomly
    # and are only kept if they DO NOT match any item in the 'self' set.

    # return a list of valid detector binary vectors.
    detectors: list[np.ndarray] = []
    
    print(f"Generating {num_detectors} detectors...")
    
    # manual progress printing setup
    candidates_generated = 0
    print_increment = max(1, int(num_detectors * 0.05))
    next_milestone = print_increment

    while len(detectors) < num_detectors:
        # generate a random candidate detector
        candidate = np.random.randint(0, 2, size=hash_size, dtype=np.uint8)
        candidates_generated += 1
        
        # assume it's a valid detector until proven otherwise
        is_valid = True
        
        # compare the candidate against every 'self' vector
        for self_vector in encoded_self_set:
            if match_r_contiguous(candidate, self_vector, r_value):
                # it matched a self_vector, so it's NOT valid.
                is_valid = False
                break # No need to check against other self_vectors
        
        # if, after all checks, it's still valid, keep it.
        if is_valid:
            detectors.append(candidate)
            # check if we should print a progress update
            if len(detectors) >= next_milestone:
                percent_done = (len(detectors) / num_detectors) * 100
                print(f"  ... {percent_done:.0f}% complete ({len(detectors)}/{num_detectors} detectors found).")
                next_milestone += print_increment
    
    acceptance_rate = 0.0

    if candidates_generated > 0:
        acceptance_rate = (num_detectors / candidates_generated) * 100
        print("Detector generation complete.")
        print(f" -> Total candidates generated: {candidates_generated}")
        print(f" -> Acceptance Rate: {acceptance_rate:.2f}%")
    else:
        print("No detectors were generated. This is unexpected.")

    return detectors, acceptance_rate