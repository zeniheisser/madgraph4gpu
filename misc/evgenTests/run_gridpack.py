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

def run_gridpack(run_directory: Path, n_events: int, seed: int, num_processes: int = None, max_events: int = None) -> None:
    """Runs run.sh script in the specified directory with given number of events and seed."""
    run_sh = run_directory / "run.sh"
    if not run_sh.exists() or not run_sh.is_file():
        print(f"Error: {run_sh} does not exist or is not a file.")
        sys.exit(1)
    run_args = ["./run.sh"]
    if num_processes is not None:
        run_args.append("-p")
        run_args.append(str(num_processes))
    if max_events is not None:
        run_args.append("-m")
        run_args.append(str(max_events))
    run_args.append(str(n_events))
    run_args.append(str(seed))
    try:
        subprocess.run(run_args, cwd=run_directory, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"Error running gridpack for {n_events} events: {e}")
        sys.exit(1)

def time_gridpack(run_directory: Path, n_events: int, seed: int, num_processes: int = None, max_events: int = None) -> float:
    """Times the gridpack generation process."""
    start_time = time.perf_counter()
    run_gridpack(run_directory, n_events, seed, num_processes, max_events)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    # Convert nanoseconds to seconds
    # elapsed_time /= 1e9
    return elapsed_time

def time_gridpack_ntimes(run_directory: Path, n_events: int, seed: int, n_times: int, num_processes: int = None, max_events: int = None) -> float:
    """Times the gridpack generation process multiple times and returns"""
    """mean, standard deviation, minimum, and maximum time."""
    times = []
    for _ in range(n_times):
        times.append(time_gridpack(run_directory, n_events, seed, num_processes, max_events))
    return np.mean(times), np.std(times), np.min(times), np.max(times)

def run_test_over_events(run_directory: Path, n_events: list, seed: int, n_times: int = 10, num_processes: int = None, max_events: int = None) -> None:
    test_dir = Path.cwd() / "results"
    if not test_dir.exists():
        test_dir.mkdir(parents=True)
        print(f"Created directory {test_dir}.")
    elif not test_dir.is_dir():
        print(f"Error: {test_dir} is not a directory.")
        sys.exit(1)
    outfile = test_dir / f"{run_directory.name}_events_{seed}s.csv"
    if outfile.exists():
        print(f"Warning: {outfile} already exists. It will be overwritten.")
    with open(outfile, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["n_events", "mean", "std", "min", "max"])
    for events in n_events:
        mean_time, std_time, min_time, max_time = time_gridpack_ntimes(run_directory, events, seed, n_times, num_processes, max_events)
        print(f"Events: {events}, Mean Time: {mean_time}, Std Time: {std_time}, Min Time: {min_time}, Max Time: {max_time}")
        with open(outfile, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([events, mean_time, std_time, min_time, max_time])

def run_test_over_processes(run_directory: Path, n_events: int, seed: int, num_processes: list, n_times: int = 10, max_events: int = None) -> None:
    test_dir = Path.cwd() / "results"
    if not test_dir.exists():
        test_dir.mkdir(parents=True)
        print(f"Created directory {test_dir}.")
    elif not test_dir.is_dir():
        print(f"Error: {test_dir} is not a directory.")
        sys.exit(1)
    outfile = test_dir / f"{run_directory.name}_processes_{seed}s.csv"
    if outfile.exists():
        print(f"Warning: {outfile} already exists. It will be overwritten.")
    with open(outfile, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["n_procs", "mean", "std", "min", "max"])
    for processes in num_processes:
        mean_time, std_time, min_time, max_time = time_gridpack_ntimes(run_directory, n_events, seed, n_times, processes, max_events)
        print(f"Processes: {processes}, Mean Time: {mean_time}, Std Time: {std_time}, Min Time: {min_time}, Max Time: {max_time}")
        with open(outfile, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([processes, mean_time, std_time, min_time, max_time])
    print(f"Results saved to {outfile}.")

def parse_args():
    """Parse command line arguments."""
    import argparse
    parser = argparse.ArgumentParser(description="Run gridpack timing tests.")
    parser.add_argument("run_directory", type=str, help="Path to the run directory containing run.sh.")
    parser.add_argument("-s","--seed", type=int, default=None, help="Seed for random number generation (default: random seed).")
    parser.add_argument("-n", "--n_times", type=int, default=10, help="Number of times to run the gridpack generation (default: 10).")
    parser.add_argument("-p", "--processes", action="store_true", help="Test with different number of processes.")
    return parser.parse_args()

def main():
    """Main function to run the gridpack generation."""
    args = parse_args()
    run_directory = args.run_directory
    seed = args.seed
    n_iterations = args.n_times

    start_dir = Path.cwd()
    # Get the run directory from command line arguments
    run_directory = args.run_directory
    run_directory = start_dir / run_directory
    # Check if the run directory is a valid path
    if not run_directory.exists():
        print(f"Error: {run_directory} does not exist.")
        sys.exit(1)
    if not run_directory.is_dir():
        print(f"Error: {run_directory} is not a directory.")
        sys.exit(1)
    # Check if the run directory exists
    if not run_directory.exists() or not run_directory.is_dir():
        print(f"Error: {run_directory} does not exist or is not a directory.")
        sys.exit(1)
    # Check if the run directory contains the necessary files
    run_sh = run_directory / "run.sh"
    if not run_sh.exists() or not run_sh.is_file():
        print(f"Error: {run_sh} does not exist or is not a file.")
        sys.exit(1)
    
    # seed = 123456789
    if seed is None:
        print("No seed provided. Generating a random seed.")
        seed = random.randint(1, 2**16 )  # Random seed for each run
    # n_iterations = 10
    n_events = [100*2**i for i in range(0, 11)]
    n_procs = [1, 2, 4, 6, 8, 12, 16, 24, 32]
    
    if not args.p:
        run_test_over_events(run_directory, n_events, seed, n_iterations)
    else:
        run_test_over_processes(run_directory, 10000, seed, n_procs, n_iterations)

    print("All gridpack runs completed successfully.")


if __name__ == "__main__":
    main()
    sys.exit(0)