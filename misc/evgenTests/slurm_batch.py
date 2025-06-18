"""Short script to launch slurm jobs for a fixed process with all SIMD modes."""

import subprocess
import os
import sys
import shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import time
import csv
import random

def sbatch_card_check(slurm_card: str) -> str:
    """Check if the slurm card is valid."""
    if not slurm_card or len(slurm_card) > 255:
        raise ValueError("Invalid slurm card name. It must not be empty and should not exceed 255 characters.")
    card_path = Path.cwd() / slurm_card
    if not card_path.exists():
        raise FileNotFoundError(f"Slurm card {slurm_card} does not exist.")
    if not card_path.is_file():
        raise ValueError(f"Slurm card {slurm_card} is not a file.")
    run_command = f"sbatch {card_path}"
    return slurm_card

def launch_slurm_job(slurm_card: str) -> None:
    """Launch a slurm job using the provided slurm card."""
    try:
        slurm_card = sbatch_card_check(slurm_card)
        print(f"Launching slurm job with card: {slurm_card}")
        subprocess.run(["sbatch", slurm_card], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error launching slurm job: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
        
def main():
    if len(sys.argv) < 2:
        print("Usage: python slurm_batch.py <process_path>")
        sys.exit(1)

    process_path = sys.argv[1]
    backends = ["fortran", "none", "sse4", "avx2", "y512", "z512", "cuda"]

    run_cards = [f"{process_path}_{backend}.sh" for backend in backends]
    
    for card in run_cards:
        launch_slurm_job(card)
    print("All slurm jobs launched successfully.")

if __name__ == "__main__":
    main()