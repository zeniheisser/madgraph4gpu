"""Short script to write slurm cards for the evgen tests.
This script generates slurm job submission scripts for running gridpack generation
and gridpack timing test, based on the specific processes and the SIMD compilation mode."""

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

def write_backend(simd_mode: str) -> Path:
    """Writes the simd_mode.back file in the grid_input directory."""
    grid_input_dir = Path.cwd() / "grid_input"
    if not grid_input_dir.exists():
        grid_input_dir.mkdir(parents=True)
    # if simd_mode not in ["fortran", "none", "sse4", "avx2", "y512", "z512", "cuda"]:
    #     raise ValueError(f"Invalid SIMD mode: {simd_mode}. Expected 'fortran', 'none', 'sse4', 'avx2', 'y512', 'z512', or 'cuda'.")
    if simd_mode not in ["fortran", "cppnone", "cppsse4", "cppavx2", "cpp512y", "cpp512z", "cuda"]:
        raise ValueError(f"Invalid SIMD mode: {simd_mode}. Expected 'fortran', 'cppnone', 'cppsse4', 'cppavx2', 'cpp512y', 'cpp512z', or 'cuda'.")
    # Create the grid_input directory if it doesn't exist
    file_name = f"{simd_mode}.back"
    if not file_name or len(file_name) > 255:
        raise ValueError("Invalid backend file name. It must not be empty and should not exceed 255 characters.")
    backend_file = grid_input_dir / file_name
    if backend_file.exists():
        print(f"Warning: {backend_file} already exists. Overwriting it.")
        backend_file.unlink()
    with open(backend_file, 'w') as f:
        if simd_mode == "cuda":
            f.write("gpu, cuda\n")
        else:
            f.write("simd, " + simd_mode + "\n")
    return f"grid_input/{file_name}"

def write_process(process: str) -> Path:
    """Writes the process.proc file in the grid_input directory."""
    grid_input_dir = Path.cwd() / "grid_input"
    if not grid_input_dir.exists():
        grid_input_dir.mkdir(parents=True)
    if not process:
        raise ValueError("Process name cannot be empty.")
    file_name = process_to_name(process) + ".proc"
    if not file_name or len(file_name) > 255:
        raise ValueError("Invalid process name. It must not be empty and should not exceed 255 characters.")
    process_file = grid_input_dir / file_name
    if process_file.exists():
        print(f"Warning: {process_file} already exists. Overwriting it.")
        process_file.unlink()
    with open(process_file, 'w') as f:
        f.write(f"{process}\n")
    return f"grid_input/{file_name}"

def process_to_name(process: str) -> str:
    """Converts a process string to a valid filename."""
    if not process:
        raise ValueError("Process name is empty.")
    return process.replace(" ", "").replace(">", "_2_").replace("+", "p").replace("-", "m").replace("~", "x").replace("\n", "").replace("\r", "")

def main():
    """Main function to write slurm cards for the evgen tests."""
    # Check if the script is run with the correct number of arguments
    if len(sys.argv) != 2:
        print("Usage: python write_slurm_cards.py <run_directory>")
        sys.exit(1)

    run_directory = Path(sys.argv[1])
    if not run_directory.exists() or not run_directory.is_dir():
        print(f"Error: {run_directory} does not exist or is not a directory.")
        sys.exit(1)

    n_cpus_gridgen = 6
    n_cpus_gridrun = 1

    # processes = ["u u~ > e+ e-", "u u~ > e+ e- g", "u u~ > e+ e- g g", "u u~ > e+ e- g g g",]
    processes = ["g g > t t~", "g g > t t~ g", "g g > t t~ g g", "g g > t t~ g g g",]
    # processes = ["p p > l+ l-", "p p > l+ l- j", "p p > l+ l- j j", "p p > l+ l- j j j",]
    # processes = ["p p > t t~", "p p > t t~ j", "p p > t t~ j j", "p p > t t~ j j j",]

    partition = "Def"
    #partition = "pelican"

    procs = [process_to_name(proc) for proc in processes]
    # Write the process files in the grid_input directory
    proc_cards = [write_process(proc) for proc in processes]
    print(f"Process files written: {proc_cards}")
    # Write the backend files in the grid_input directory
    simd_modes = ["fortran", "none", "sse4", "avx2", "y512", "z512"]
    simd_modes_explicit = ["fortran", "cppnone", "cppsse4", "cppavx2", "cpp512y", "cpp512z"]
    backend_cards = [write_backend(simd_mode) for simd_mode in simd_modes_explicit]
    print(f"Backend files written: {backend_cards}")

    # Write the slurm scripts to run gen_gridpack.py for each process and SIMD mode
    for process, proc_card in zip(procs,proc_cards):
        for simd_mode, backend_card in zip(simd_modes, backend_cards):
            slurm_card_path = run_directory / f"grid_{process}_{simd_mode}.sh"
            loc_proc = process[process.rfind("_") + 1:]  # Get the last part of the process name
            with open(slurm_card_path, 'w') as f:
                f.write("#!/bin/bash\n")
                f.write(f"#SBATCH --job-name={loc_proc}_{simd_mode}\n")
                f.write(f"#SBATCH --output=logs/log_grid_{loc_proc}_{simd_mode}.txt\n")
                f.write(f"#SBATCH --error=logs/error_grid_{loc_proc}_{simd_mode}.txt\n")
                f.write("#SBATCH --time=24:00:00\n")
                f.write("#SBATCH --ntasks=1\n")
                f.write(f"#SBATCH --cpus-per-task={n_cpus_gridgen}\n")
                f.write("#SBATCH --mem-per-cpu=1024\n")
                f.write("\n")
                f.write("module load Python\n")
                f.write(f"python gen_gridpack.py {backend_card} {proc_card} > logs/gen_gridpack_{loc_proc}_{simd_mode}.log\n")
            print(f"Slurm card written to {slurm_card_path}")
    
    seed = random.randint(1, 2**16)
    
    # Write the slurm scripts to run run_gridpack.py for each process and SIMD mode
    for process in procs:
        for simd_mode, simd_mode_explicit in zip(simd_modes, simd_modes_explicit):
            curr_proc = process + f"_simd_{simd_mode_explicit}"
            slurm_card_path = run_directory / f"run_{process}_{simd_mode}.sh"
            loc_proc = process[process.rfind("_") + 1:]  # Get the last part of the process name
            if "g" in loc_proc:
                # If there are final state gluons, replace name with number of gluons
                ng = loc_proc.count("g")
                loc_proc = loc_proc.replace("g", "")
                loc_proc += str(ng) + "g"
            if "j" in loc_proc:
                # If there are final state jets, replace name with number of jets
                nj = loc_proc.count("j")
                loc_proc = loc_proc.replace("j", "")
                loc_proc += str(nj) + "j"
            with open(slurm_card_path, 'w') as f:
                f.write("#!/bin/bash\n")
                f.write(f"#SBATCH --job-name={loc_proc}_{simd_mode}\n")
                f.write(f"#SBATCH --partition={partition}\n")
                if partition == "pelican":
                    f.write("#SBATCH --constraint=\"IceLake\"\n")
                #f.write("#SBATCH --partition=Def\n")
                f.write(f"#SBATCH --output=logs/log_run_{loc_proc}_{simd_mode}.txt\n")
                f.write(f"#SBATCH --error=logs/error_run_{loc_proc}_{simd_mode}.txt\n")
                f.write("#SBATCH --time=24:00:00\n")
                f.write("#SBATCH --ntasks=1\n")
                f.write(f"#SBATCH --cpus-per-task={n_cpus_gridrun}\n")
                # if n_cpus_gridrun > 1:
                    # f.write("#SBATCH --hint=compute_bound\n")
                f.write("#SBATCH --mem-per-cpu=1024\n")
                f.write("\n")
                f.write("module load Python\n")
                if n_cpus_gridrun > 1:
                    f.write(f"python run_gridpack.py {curr_proc} -s {seed} -p > logs/run_gridpack_{loc_proc}_{simd_mode}.log\n")
                else:
                    f.write(f"python run_gridpack.py {curr_proc} -s {seed} > logs/run_gridpack_{loc_proc}_{simd_mode}.log\n")
            print(f"Slurm card written to {slurm_card_path}")
    print("All slurm cards have been written successfully.")
if __name__ == "__main__":
    main()
    # Run the main function
    # when the script is executed directly
    # main()
    # else:
    #     print("This script is not meant to be imported as a module.")
#     sys.exit(1)