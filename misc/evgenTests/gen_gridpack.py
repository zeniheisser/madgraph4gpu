import subprocess
import os
import sys
import shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def check_dir( dir_path: Path) -> bool:
    """Checks if the given path is a directory."""
    if not dir_path.exists():
        print(f"Error: Directory {dir_path} does not exist.")
        return False
    if not dir_path.is_dir():
        print(f"Error: {dir_path} is not a directory.")
        return False
    return True

def check_file( file_path: Path) -> bool:
    """Checks if the given path is a file."""
    if not file_path.exists():
        print(f"Error: File {file_path} does not exist.")
        return False
    if not file_path.is_file():
        print(f"Error: {file_path} is not a file.")
        return False
    return True

def parse_hardware_and_backend(run_type: str) -> tuple:
    """Parses the hardware and backend from a run type string."""
    if not run_type:
        raise ValueError("Run type is empty.")
    parts = run_type.split(",")
    if len(parts) < 2:
        raise ValueError(f"Invalid run type format: {run_type}")
    hardware = parts[0].strip()
    if hardware not in ["simd", "gpu", "fortran"]:
        raise ValueError(f"Invalid hardware type: {hardware}. Expected 'simd', 'gpu', or 'fortran'.")
    backend = parts[1].strip()
    if backend not in ["cppnone", "cppsse4", "cppavx2", "cpp512y", "cpp512z", "cppauto", "fortran"]:
        raise ValueError(f"Invalid backend type: {backend}. Expected 'cppnone', 'cppsse4', 'cppavx2', 'cpp512y', 'cpp512z', 'cppauto', or 'fortran'.")
    return hardware, backend

def parse_run_types(run_types: list) -> list:
    """Parses a list of run types into a list of tuples (hardware, backend)."""
    parsed_run_types = []
    for run_type in run_types:
        if isinstance(run_type, str):
            try:
                hardware, backend = parse_hardware_and_backend(run_type)
                parsed_run_types.append((hardware, backend))
            except ValueError as e:
                print(f"Error parsing run type '{run_type}': {e}")
        elif isinstance(run_type, tuple) and len(run_type) == 2:
            parsed_run_types.append(run_type)
        else:
            print(f"Invalid run type format: {run_type}. Expected string or tuple of (hardware, backend).")
    return parsed_run_types

def process_to_name(process: str) -> str:
    """Converts a process string to a valid filename."""
    if not process:
        raise ValueError("Process name is empty.")
    return process.replace(" ", "").replace(">", "_2_").replace("+", "p").replace("-", "m").replace("~", "x").replace("\n", "").replace("\r", "")

def write_run_card( card_dir: Path, process: str, hardware: str = "simd", backend: str = "cppauto") -> Path:
    """Writes the run card content to a file."""
    if not card_dir.exists():
        os.makedirs(card_dir)
        print(f"Created directory {card_dir}.")
    if not card_dir.is_dir():
        print(f"Error: {card_dir} is not a directory.")
        return
    if process == "":
        print("Error: Process name is empty.")
        return
    card_name = process_to_name(process)
    if not card_name:
        print("Error: Invalid process name.")
        return
    out_type = "madevent"
    if hardware not in ["simd", "gpu", "fortran"]:
        print(f"Warning: Hardware type '{hardware}' is not recognized. Defaulting to 'simd'.")
        hardware = "simd"
    if hardware == "simd" or hardware == "gpu":
        out_type += "_%s" % hardware
    if backend not in ["cppnone", "cppsse4", "cppavx2", "cpp512y", "cpp512z", "cppauto", "fortran"]:
        print(f"Warning: Backend type '{backend}' is not recognized. Defaulting to 'cppauto'.")
        backend = "cppauto"
    card_name += f"_{hardware}_{backend}"
    run_card_path = card_dir / f"{card_name}.run"
    if run_card_path.exists():
        print(f"Warning: {run_card_path} already exists. Overwriting.")
    run_card_content = ""
    if hardware == "fortran":
        run_card_content = f"""\
generate {process}
output {out_type} {card_name}
launch
0
set gridpack True
0
exit"""
    else:
        run_card_content = f"""\
generate {process}
output {out_type} {card_name}
launch
0
set gridpack True
set cudacpp_backend {backend}
0
exit"""
    with run_card_path.open("w") as f:
        f.write(run_card_content)
    print(f"Run card written to {run_card_path}")
    return run_card_path

def generate_and_run_madevent(run_card_path: Path):
    """Runs bin/mg5_aMC with the input run_card."""
    cmd = ["bin/mg5_aMC"]

    if not run_card_path.exists():
        print(f"Error: run card file '{run_card_path}' does not exist.")
        sys.exit(1)
    with run_card_path.open("r") as infile:
        print(f"Running MadEvent with input from {run_card_path}...")
        subprocess.run(cmd, stdin=infile, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Finished running MadEvent with input from {run_card_path}.")

def untar_gridpack(proc_dir: Path) -> Path:
    """Untars the gridpack in the given directory."""
    if not check_dir(proc_dir):
        return
    tar_file = proc_dir / "run_01_gridpack.tar.gz"
    if not check_file(tar_file):
        print(f"Error: {tar_file} does not exist or is not a file.")
        return
    subprocess.run(["tar", "-xf", str(tar_file)], cwd=proc_dir, check=True)
    run_sh = proc_dir / "run.sh"
    if not check_file(run_sh):
        print(f"Error: {run_sh} does not exist or is not a file.")
        return
    print(f"Untarred gridpack in {proc_dir}.")
    return run_sh

# def generate_gridpack(process: str, hardware: str = "simd", backend: str = "cppauto") -> Path:
def generate_gridpack(process: str, run_type: tuple = ("simd", "cppauto")) -> Path:
    """Generates a gridpack for the given process."""
    curr_path = Path.cwd()
    run_cards_dir = curr_path / "run_cards"
    if not run_cards_dir.exists():
        os.makedirs(run_cards_dir)
        print(f"Created directory {run_cards_dir}.")
    if not run_cards_dir.is_dir():
        print(f"Error: {run_cards_dir} is not a directory.")
        return
    run_card_path = write_run_card(run_cards_dir, process, run_type[0], run_type[1])
    if not run_card_path:
        return
    proc_name = process_to_name(process)
    proc_name += f"_{run_type[0]}_{run_type[1]}"
    proc_dir = curr_path / proc_name
    if proc_dir.exists():
        print(f"Warning: Directory {proc_dir} already exists. Removing it.")
        shutil.rmtree(proc_dir)
    generate_and_run_madevent(run_card_path)
    untar_gridpack(proc_dir)
    return proc_dir

def generate_gridpacks(process: str, run_types: list = [("simd", "cppauto")]) -> list:
    """Generates gridpacks for the given process with multiple run types."""
    proc_dirs = []
    for run_type in run_types:
        proc_dir = generate_gridpack(process, run_type)
        if proc_dir:
            proc_dirs.append(proc_dir)
    return proc_dirs

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <runs_filename> <process_filename>")
        sys.exit(1)

    input_runs_filename = sys.argv[1]
    if not check_file(Path(input_runs_filename)):
        print(f"Error: {input_runs_filename} does not exist or is not a file.")
        sys.exit(1)

    input_process_filename = sys.argv[2]
    if not check_file(Path(input_process_filename)):
        print(f"Error: {input_process_filename} does not exist or is not a file.")
        sys.exit(1)
        
    #read input file, split by newlines
    with open(input_process_filename, "r") as f:
        processes = f.readlines()
    if not processes:
        print("Error: No processes found in the input file.")
        sys.exit(1)

    for process in processes:
        process = process.strip()
        if not process:
            print("Error: Empty process found in the input file.")
            sys.exit(1)

    # read run types from input file
    with open(input_runs_filename, "r") as f:
        runs = f.readlines()
    if not runs:
        print("Error: No run types found in the input file.")
        sys.exit(1)
    
    runs = [run.strip() for run in runs if run.strip()]  # Remove empty lines
    if not runs:
        print("Error: No valid run types found in the input file.")
        sys.exit(1)
    # parse run types
    try:
        run_types = parse_run_types(runs)
    except ValueError as e:
        print(f"Error parsing run types: {e}")
        sys.exit(1)

    for proc in processes:
        proc = proc.strip()
        if not proc:
            print("Error: Empty process found in the input file.")
            continue
        print(f"Generating gridpack for process: {proc}")
        runs = generate_gridpacks(proc, run_types)
    print(f"Generated gridpacks for processes: {processes}")
    if not runs:
        print("Error: No gridpacks were generated.")
        sys.exit(1)
    for run in runs:
        print(f"Gridpack generated in directory: {run}")
    if len(runs) == 0:
        print("Error: No gridpacks were generated.")
        sys.exit(1)
    if len(runs) > 0:
        print(f"Generated {len(runs)} gridpacks successfully.")
    
if __name__ == "__main__":
    main()
    print("Gridpack generation script completed.")
    sys.exit(0)