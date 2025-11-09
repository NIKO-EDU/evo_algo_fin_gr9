# data_loader.py

from typing import Tuple, List
import os

def load_sms_data(filepath: str) -> Tuple[List[str], List[str]]:
    if not os.path.exists(filepath):
        print(f"Error: File not found at '{filepath}'")
        return [], []

    ham_messages: List[str] = []
    spam_messages: List[str] = []

    print(f"Loading and parsing data from '{filepath}'...")
    
    with open(filepath, 'r', encoding='latin-1') as f:
        for line in f:
            parts = line.strip().split('\t', 1)
            
            if len(parts) == 2:
                label, message = parts
                if label == 'ham':
                    ham_messages.append(message)
                elif label == 'spam':
                    spam_messages.append(message)

    print(f"Loading complete.")
    print(f" -> Found {len(ham_messages)} ham messages.")
    print(f" -> Found {len(spam_messages)} spam messages.")
    
    return ham_messages, spam_messages

if __name__ == '__main__':
    dataset_path = os.path.join('data', 'dataset', 'SMSSpamCollection')

    ham, spam = load_sms_data(dataset_path)

    if ham:
        print("\n--- Example Ham Message ---")
        print(ham[0])
    
    if spam:
        print("\n--- Example Spam Message ---")
        print(spam[0])