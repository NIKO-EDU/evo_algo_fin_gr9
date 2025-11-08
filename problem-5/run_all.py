# this is a utility script to run all experiments and save the results

import subprocess
import sys

# Get the python executable's path to be robust (works in virtual environments)
PYTHON_EXEC = sys.executable

# --- List of all experiments to run ---
EXPERIMENTS = [
    "4x4_BASIC",
    "8x8_HARD",
    "4x4_MYOPIC",
    "4x4_CAUTIOUS"
]

def run_command(command_str: str):
    """Prints and runs a single command string in the terminal."""
    print(f"\n> Executing: {command_str}")
    # Using shell=True to simply run the command string as is.
    # For this controlled use case, it's safe and simple.
    subprocess.run(command_str, shell=True, check=True)

# --- THE SCRIPT'S MAIN LOGIC ---

print("="*50)
print("--- STARTING FULL TRAINING & ANALYSIS PIPELINE ---")
print("="*50)

# --- 1. RUN ALL TRAINING ---
print("\n--- Phase 1: Training All Agents ---")
for name in EXPERIMENTS:
    command = f"{PYTHON_EXEC} main.py {name}"
    run_command(command)

# --- 2. RUN ALL ANALYSIS ---
print("\n--- Phase 2: Analyzing All Results ---")
for name in EXPERIMENTS:
    command = f"{PYTHON_EXEC} analysis.py {name}"
    run_command(command)

print("\n" + "="*50)
print("--- PIPELINE COMPLETE ---")
print("="*50)