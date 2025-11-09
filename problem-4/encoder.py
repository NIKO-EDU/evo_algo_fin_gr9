# encoder.py 

import numpy as np

def _generate_ngrams(
    text: str, 
    ngram_size: int
) -> list[str]:
    # Helper function to generate a list of overlapping n-grams.
    # Example (n=3): "hello" -> ["hel", "ell", "llo"]
    return [
        text[i:i+ngram_size] 
        for i in range(len(text) - ngram_size + 1)
    ]

def encode_message(
    message: str, 
    hash_size: int, 
    ngram_size: int
) -> np.ndarray:
    # Convert a single message string into a binary numpy array using n-gram hashing.

    # 1. Start with a vector of all zeros.
    binary_vector = np.zeros(hash_size, dtype=np.uint8)
    
    # 2. Pre-process the text: convert to lowercase.
    processed_message = message.lower()
    
    # 3. Generate n-grams from the processed message.
    ngrams = _generate_ngrams(processed_message, ngram_size)
    
    # 4. Hash each n-gram to an index and set that position to 1.
    for ngram in ngrams:
        hash_index = hash(ngram) % hash_size
        binary_vector[hash_index] = 1
            
    return binary_vector