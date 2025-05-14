import subprocess
import os
import sys
import shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def run_generate_events(process_dir: Path = Path.cwd(), run_card_path: Path = None):
    """Runs the generate_events binary, optionally piping in a run card."""
    cmd = ["bin/generate_events"]

    if run_card_path:
        if not run_card_path.exists():
            print(f"Error: run card file '{run_card_path}' does not exist.")
            sys.exit(1)

        with run_card_path.open("r") as infile:
            subprocess.run(cmd, stdin=infile, check=True)
            print(f"Finished running generate_events with input from {run_card_path}.")
    else:
        subprocess.run(cmd, check=True)
        print("Finished running generate_events without run card.")


def collect_amps(subprocesses_path: Path = Path.cwd(), output_filename: str = "data.tmp"):
    """Combines 'amps.dat' files from G* subdirectories in P* subdirectories."""
    for p_dir in subprocesses_path.iterdir():
        if p_dir.is_dir() and p_dir.name.startswith("P"):
            print(f"Processing directory: {p_dir}")
            for g_dir in p_dir.iterdir():
                if g_dir.is_dir() and g_dir.name.startswith("G"):
                    amps_file = g_dir / "amps.dat"
                    if amps_file.exists():
                        print(f"Found: {amps_file}")
                        with amps_file.open("r", encoding="utf-8") as src, open(output_filename, "a", encoding="utf-8") as dst:
                            data = src.read().replace("\n", "").replace("\r", "")
                            dst.write(data)

                    else:
                        print(f"Warning: {amps_file} not found.")

def replace_make_opts(process_dir: Path = Path.cwd()):
    """Replaces Source/make_opts with make_opts_clean from the current directory."""
    source_path = process_dir / "Source/make_opts"
    replacement_path = Path("make_opts_clean")

    if not replacement_path.exists():
        try:
            replacement_path = process_dir / "make_opts_clean"
        except FileNotFoundError:
            print("Error: make_opts_clean not found in the current directory.")
            return
    if not source_path.exists():
        print(f"Error: {source_path} does not exist.")
        return

    if not source_path.parent.exists():
        print("Error: Source directory does not exist.")
        return

    try:
        shutil.copyfile(replacement_path, source_path)
        print(f"Replaced {source_path} with {replacement_path}.")
    except Exception as e:
        print(f"Failed to replace make_opts: {e}")


def clean_source_directory(process_dir: Path = Path.cwd()):
    """Runs 'make cleanall' in the Source directory."""
    source_dir = process_dir / "Source"

    if not source_dir.exists():
        print("Error: Source directory does not exist.")
        sys.exit(1)

    try:
        subprocess.run(["make", "cleanall"], cwd=source_dir, check=True)
        print("Successfully ran 'make cleanall' in Source directory.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to run 'make cleanall': {e}")
        sys.exit(1)

def compile_subprocesses(process_dir: Path = Path.cwd(), fp_flag: str = "m"):
    """Runs 'make FPFLAG=<flag>' in each P* subdirectory of SubProcesses."""
    if fp_flag not in {"m", "d", "f"}:
        print(f"Error: Invalid FPTYPE value '{fp_flag}'. Must be 'm', 'd', or 'f'.")
        sys.exit(1)

    base_path = process_dir / "SubProcesses"

    for p_dir in base_path.iterdir():
        if p_dir.is_dir() and p_dir.name.startswith("P"):
            print(f"Compiling in {p_dir} with FPTYPE={fp_flag}...")
            try:
                subprocess.run(["make", f"FPTYPE={fp_flag}"], cwd=p_dir, check=True)
                print(f"Compilation succeeded in {p_dir}")
            except subprocess.CalledProcessError as e:
                print(f"Error during compilation in {p_dir}: {e}")

def run_subprocess_tests(process_dir: Path = Path.cwd(), suffix: str = "cpp"):
    """Runs madevet_cpp in G* subdirectories and collects amps.dat content."""
    base_path = process_dir / "SubProcesses"
    if not base_path.exists():
        print("Error: SubProcesses directory does not exist.")
        sys.exit(1)
    if not base_path.is_dir():
        print("Error: SubProcesses is not a directory.")
        sys.exit(1)
    executable = "madevent"
    if suffix != "":
        if suffix not in {"cpp", "cuda", "fortran"}:
            print(f"Error: Invalid suffix '{suffix}'. Must be 'cpp', 'cuda', or 'fortran'.")
            sys.exit(1)
        executable += f"_{suffix}"
    
    for p_dir in base_path.iterdir():
        if p_dir.is_dir() and p_dir.name.startswith("P"):
            for g_dir in p_dir.iterdir():
                if g_dir.is_dir() and g_dir.name.startswith("G"):
                    input_file = g_dir / "input_app.txt"
                    madevent_binary = Path("..") / executable  # located in P* directory
                    if not input_file.exists():
                        print(f"Warning: {input_file} not found.")
                        continue
                    if not (p_dir / executable).exists():
                        print(f"Warning: {madevent_binary} not found.")
                        continue

                    print(f"Running madevent_cpp in {g_dir}...")
                    try:
                        with input_file.open("r") as infile:
                            subprocess.run([str(madevent_binary)], stdin=infile, cwd=g_dir, check=True)
                    except subprocess.CalledProcessError as e:
                        print(f"Execution failed in {g_dir}: {e}")
                        continue

def clear_amps_files(process_dir: Path = Path.cwd()):
    """Removes all amps.dat files in G* directories under each P* directory in SubProcesses."""
    base_path = process_dir / "SubProcesses"
    if not base_path.exists():
        print("Error: SubProcesses directory does not exist.")
        sys.exit(1)
    if not base_path.is_dir():
        print("Error: SubProcesses is not a directory.")
        sys.exit(1)

    for p_dir in base_path.iterdir():
        if p_dir.is_dir() and p_dir.name.startswith("P"):
            for g_dir in p_dir.iterdir():
                if g_dir.is_dir() and g_dir.name.startswith("G"):
                    amps_file = g_dir / "amps.dat"
                    if amps_file.exists():
                        try:
                            amps_file.unlink()
                            print(f"Removed: {amps_file}")
                        except Exception as e:
                            print(f"Failed to remove {amps_file}: {e}")

def run_and_collect(process_dir: Path = Path.cwd(), executable_suffix: str = "cpp", output_filename: str = "data.tmp"):
    """Runs the subprocesses and collects amps.dat content."""
    subprocesses_path = process_dir / "SubProcesses"
    if not subprocesses_path.exists():
        print("Error: SubProcesses directory does not exist.")
        sys.exit(1)
    if not subprocesses_path.is_dir():
        print("Error: SubProcesses is not a directory.")
        sys.exit(1)
        
    try:
        run_subprocess_tests(process_dir, executable_suffix)
    except subprocess.CalledProcessError as e:
        print(f"Error running madevent: {e}")
        sys.exit(1)
    # Remove output file if it exists to avoid appending to an old file
    if os.path.exists(output_filename):
        os.remove(output_filename)
    collect_amps(subprocesses_path, output_filename)
    clear_amps_files(process_dir)
    print(f"Finished combining files into {output_filename}")

def compile_and_run(process_dir: Path = Path.cwd(), fp_flag: str = "m", executable_suffix: str = "cpp", output_filename: str = "data.tmp"):
    """Compiles and runs the subprocesses."""
    start_dir = process_dir
    if not start_dir.exists():
        print(f"Error: {start_dir} does not exist.")
        sys.exit(1)
    if not start_dir.is_dir():
        print(f"Error: {start_dir} is not a directory.")
        sys.exit(1)
    clear_amps_files(start_dir)
    subprocesses_path = start_dir / "SubProcesses"
    if not subprocesses_path.exists():
        print("Error: SubProcesses directory does not exist.")
        sys.exit(1)
    clean_source_directory(start_dir)
    replace_make_opts(start_dir)
    compile_subprocesses(start_dir, fp_flag)
    try:
        run_subprocess_tests(start_dir, executable_suffix)
    except subprocess.CalledProcessError as e:
        print(f"Error running madevent: {e}")
        sys.exit(1)
    # Remove output file if it exists to avoid appending to an old file
    if os.path.exists(output_filename):
        os.remove(output_filename)
    collect_amps(subprocesses_path, output_filename)
    print(f"Finished combining files into {output_filename}")
    clear_amps_files(start_dir)
    print("Finished cleaning up amps files and Source directory.")

def load_amps_results(filenames: dict) -> dict:
    """
    Loads amps.dat aggregate results into numpy arrays.
    
    :param filenames: Dictionary of {label: filepath}, e.g., {"m": "results_m.txt"}
    :return: Dictionary of {label: numpy_array}
    """
    results = {}
    for label, file_path in filenames.items():
        try:
            data = np.fromfile(file_path, sep=" ")
            results[label] = data
            print(f"Loaded {label} ({file_path}) with {data.size} entries.")
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")
    return results

def plot_relative_diff_histogram(relative_diffs, label=None):
    """
    Plots a histogram of relative differences, binned by order of magnitude.
    """
    # Avoid division by zero: treat exact zero as 1e-15
    safe_diffs = np.where(relative_diffs == 0, 1e-15, relative_diffs)
    
    # Convert to log10 scale
    log_diffs = np.log10(safe_diffs)

    # Define 15 bins from 10^-15 to 10^-1
    bins = np.arange(-15, 0)  # 15 bins: -15 to -1 (inclusive of left)
    
    plt.figure(figsize=(8, 5))
    plt.hist(log_diffs, bins=bins, edgecolor="black", alpha=0.75)
    
    # Customize ticks and labels to show powers of 10
    tick_labels = [f"$10^{{{int(b)}}}$" for b in bins]
    plt.xticks(bins, tick_labels, rotation=45)
    
    plt.xlabel("Relative difference (order of magnitude)")
    plt.ylabel("Count")
    plt.title(f"Histogram of Relative Differences {f'({label})' if label else ''}")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

def compare_relative_diff_histograms(relative_diff_dict):
    """
    Plots overlaid histograms of relative differences from multiple FPFLAG configurations.
    
    :param relative_diff_dict: Dict of {label: relative_diff_array}
    """
    bins = np.arange(-15, 0)  # log10 bins from 1e-15 to 1e-1

    plt.figure(figsize=(9, 5))

    for label, diffs in relative_diff_dict.items():
        # Replace exact 0s with 1e-15 to keep them in -15 bin
        safe_diffs = np.where(diffs == 0, 1e-15, diffs)
        log_diffs = np.log10(safe_diffs)

        plt.hist(log_diffs, bins=bins, alpha=0.6, label=label, edgecolor='black', histtype='stepfilled')

    # Format x-axis ticks as powers of ten
    tick_labels = [f"$10^{{{int(b)}}}$" for b in bins]
    plt.xticks(bins, tick_labels, rotation=45)

    plt.xlabel("Relative difference (order of magnitude)")
    plt.ylabel("Count")
    plt.title("Comparison of Relative Differences by FPFLAG")
    plt.legend(title="FPFLAG")
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

def stacked_relative_diff_histogram(relative_diff_dict):
    """
    Plots a stacked histogram of relative differences across FPFLAG configurations.

    :param relative_diff_dict: Dict of {label: relative_diff_array}
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Define 14 bins using 15 edges (from 10^-15 to 10^-1)
    bin_edges = np.arange(-15, 1)  # 15 edges => 14 bins from -15 to 0
    bin_centers = bin_edges[:-1] + 0.5  # Midpoints for plotting

    # Bin counts for each label
    counts = {}
    for label, diffs in relative_diff_dict.items():
        safe_diffs = np.where(diffs == 0, 1e-15, diffs)
        log_diffs = np.log10(safe_diffs)
        hist, _ = np.histogram(log_diffs, bins=bin_edges)
        counts[label] = hist

    # Stack bar data
    labels = list(relative_diff_dict.keys())
    stacked_data = np.vstack([counts[label] for label in labels])

    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(bin_centers, stacked_data[0], width=1, label=labels[0])
    bottom = stacked_data[0]

    for i in range(1, len(labels)):
        plt.bar(bin_centers, stacked_data[i], width=1, bottom=bottom, label=labels[i])
        bottom += stacked_data[i]

    # Format x-axis ticks as powers of ten
    xtick_labels = [f"$10^{{{int(b)}}}$" for b in bin_edges[:-1]]
    plt.xticks(bin_centers, xtick_labels, rotation=45)

    plt.xlabel("Relative difference (order of magnitude)")
    plt.ylabel("Count")
    plt.title("Stacked Histogram of Relative Differences by FPFLAG")
    plt.legend(title="FPFLAG")
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

def grouped_relative_diff_histogram(relative_diff_dict, process: str = None):
    """
    Plots a grouped bar histogram (side-by-side) of relative differences by FPFLAG.

    :param relative_diff_dict: Dict of {label: relative_diff_array}
    """

    bin_edges = np.arange(-15, 1)           # 15 edges => 14 bins
    bin_centers = bin_edges[:-1] + 0.5      # Midpoints for 14 bins
    n_bins = len(bin_centers)
    labels = list(relative_diff_dict.keys())
    n_labels = len(labels)

    # Collect histograms
    histograms = []
    for label in labels:
        diffs = relative_diff_dict[label]
        safe_diffs = np.where(diffs == 0, 1e-15, diffs)
        log_diffs = np.log10(safe_diffs)
        hist, _ = np.histogram(log_diffs, bins=bin_edges)
        hist = hist.astype(float)  # Ensure float type for division
        hist /= np.sum(hist)  # Normalize to get probabilities
        histograms.append(hist)

    # Plot bars side-by-side
    bar_width = 0.8 / n_labels  # Total width = 0.8 split among bars
    offsets = np.linspace(-0.4, 0.4, n_labels, endpoint=False)

    plot_name = process_to_name(process) if process else "relative_diff"
    plot_dir = Path.cwd() / "plots"
    if not plot_dir.exists():
        os.makedirs(plot_dir)
        print(f"Created directory {plot_dir}.")
    if not plot_dir.is_dir():
        print(f"Error: {plot_dir} is not a directory.")
        return
    plot_path = plot_dir / (plot_name + "_lin.pdf")
    if plot_path.exists():
        print(f"Warning: {plot_path} already exists. Overwriting.")
    plot_title = "Relative difference to double precision evaluation"
    if process:
        plot_title += f" for process: {process}"
    plt.figure(figsize=(10, 6))

    for i, (label, hist) in enumerate(zip(labels, histograms)):
        x_positions = bin_centers + offsets[i]
        plt.bar(x_positions, hist, width=bar_width, label=label)

    # Format x-axis
    xtick_labels = [f"$10^{{{int(b)}}}$" for b in bin_edges[:-1]]
    plt.xticks(bin_centers, xtick_labels, rotation=45)

    plt.xlabel("Relative difference (order of magnitude)")
    plt.ylabel("Fraction of events")
    plt.title(plot_title)
    plt.legend(title="FPTYPE")
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(plot_path, format="pdf")
    print(f"Saved lin-scaled plot to {plot_path}")
    plot_path = plot_dir / (plot_name + "_log.pdf")
    if plot_path.exists():
        print(f"Warning: {plot_path} already exists. Overwriting.")
    plt.yscale("log")
    plt.savefig(plot_path, format="pdf")
    print(f"Saved log-scaled plot to {plot_path}")

def process_to_name(process: str) -> str:
    """Converts a process string to a valid filename."""
    if not process:
        raise ValueError("Process name is empty.")
    return process.replace(" ", "_").replace(">", "2").replace("+", "p").replace("-", "m").replace("~", "x").replace("\n", "").replace("\r", "")

def write_run_card( card_dir: Path, process: str ):
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
    run_card_path = card_dir / f"{card_name}.run"
    if run_card_path.exists():
        print(f"Warning: {run_card_path} already exists. Overwriting.")
    run_card_content = f"""\
generate {process}
output madevent_simd {card_name}.fp
launch
0
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
        subprocess.run(cmd, stdin=infile, check=True)
        print(f"Finished running MadEvent with input from {run_card_path}.")

def plot_difference_heatmaps(ref, abs_diff, rel_diff, bins=100, label="", process: str = None):
    """
    Plots two heatmaps:
    1. Relative difference vs absolute difference
    2. Relative difference vs reference amplitude

    Safely applies log scale by mapping zeros to smallest positive non-zero value.
    Keeps arrays aligned and uses absolute values as needed.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Ensure non-negative input
    abs_diff = np.abs(abs_diff)
    rel_diff = np.abs(rel_diff)
    ref = np.abs(ref)
    proc_name = process_to_name(process) if process else "relative_diff"
    plot_dir = Path.cwd() / "plots"
    if not plot_dir.exists():
        os.makedirs(plot_dir)
        print(f"Created directory {plot_dir}.")
    if not plot_dir.is_dir():
        print(f"Error: {plot_dir} is not a directory.")
        return
    plot_path = plot_dir / (proc_name + "_heatmap.pdf")
    if plot_path.exists():
        print(f"Warning: {plot_path} already exists. Overwriting.")
    # Avoid log(0) by replacing zeros with min non-zero value in each array
    def safe_log(data, label):
        positive_mask = data > 0
        if not np.any(positive_mask):
            raise ValueError(f"All values in {label} are zero or negative.")
        min_nonzero = data[positive_mask].min()
        safe_data = np.where(data == 0, min_nonzero, data)
        return np.log10(safe_data)

    log_abs_diff = safe_log(abs_diff, "abs_diff")
    log_rel_diff = safe_log(np.clip(rel_diff, 1e-15, None), "rel_diff")  # clip large relative errors, keep min=1e-15
    log_ref = safe_log(ref, "ref")

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Relative vs Absolute Difference
    h1 = axs[0].hist2d(
        log_abs_diff, log_rel_diff,
        bins=bins, cmap="viridis", cmin=1
    )
    axs[0].set_xlabel("log10(Absolute Difference)")
    axs[0].set_ylabel("log10(Relative Difference)")
    axs[0].set_title(f"Rel. vs Abs. Difference {f'({label})' if label else ''}")
    fig.colorbar(h1[3], ax=axs[0])

    # 2. Relative Difference vs Reference Amplitude
    h2 = axs[1].hist2d(
        log_ref, log_rel_diff,
        bins=bins, cmap="plasma", cmin=1
    )
    axs[1].set_xlabel("log10(Reference Amplitude)")
    axs[1].set_ylabel("log10(Relative Difference)")
    axs[1].set_title(f"Rel. Diff vs Ref. Amplitude {f'({label})' if label else ''}")
    fig.colorbar(h2[3], ax=axs[1])

    plt.tight_layout()
    plt.savefig(plot_path, format="pdf")
    print(f"Saved heatmap plot to {plot_path}")

def plot_comp_difference_heatmaps(ref, rel_diff_1, rel_diff_2, bins=100, label1="", label2="", process: str = None):
    """
    Plots two heatmaps:
    1. Relative difference vs absolute difference
    2. Relative difference vs reference amplitude

    Safely applies log scale by mapping zeros to smallest positive non-zero value.
    Keeps arrays aligned and uses absolute values as needed.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Ensure non-negative input
    #abs_diff = np.abs(abs_diff)
    rel_diff_1 = np.abs(rel_diff_1)
    rel_diff_2 = np.abs(rel_diff_2)
    ref = np.abs(ref)
    proc_name = process_to_name(process) if process else "relative_diff"
    plot_dir = Path.cwd() / "plots"
    if not plot_dir.exists():
        os.makedirs(plot_dir)
        print(f"Created directory {plot_dir}.")
    if not plot_dir.is_dir():
        print(f"Error: {plot_dir} is not a directory.")
        return
    plot_path = plot_dir / (proc_name + "_comp_heatmap.pdf")
    if plot_path.exists():
        print(f"Warning: {plot_path} already exists. Overwriting.")
    # Avoid log(0) by replacing zeros with min non-zero value in each array
    def safe_log(data, label):
        positive_mask = data > 0
        if not np.any(positive_mask):
            raise ValueError(f"All values in {label} are zero or negative.")
        min_nonzero = data[positive_mask].min()
        safe_data = np.where(data == 0, min_nonzero, data)
        return np.log10(safe_data)

    #log_abs_diff = safe_log(abs_diff, "abs_diff")
    log_rel_diff_1 = safe_log(np.clip(rel_diff_1, 1e-15, None), "rel_diff_1")  # clip large relative errors, keep min=1e-15
    log_rel_diff_2 = safe_log(np.clip(rel_diff_2, 1e-15, None), "rel_diff_2")
    log_ref = safe_log(ref, "ref")

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Relative vs Absolute Difference
    h1 = axs[0].hist2d(
        log_ref, log_rel_diff_1,
        bins=bins, cmap="plasma", cmin=1, density=True, vmin=0, vmax=1
    )
    axs[0].set_xlabel("log10(Scattering amplitude)")
    axs[0].set_ylabel("log10(Relative difference)")
    axs[0].set_title(f"{f'{label1}' if label1 else ''}")
    fig.colorbar(h1[3], ax=axs[0])

    # 2. Relative Difference vs Reference Amplitude
    h2 = axs[1].hist2d(
        log_ref, log_rel_diff_2,
        bins=bins, cmap="plasma", cmin=1, density=True, vmin=0, vmax=1
    )
    axs[1].set_xlabel("log10(Scattering amplitude)")
    axs[1].set_ylabel("log10(Relative difference)")
    axs[1].set_title(f"{f'{label2}' if label2 else ''}")
    fig.colorbar(h2[3], ax=axs[1])
    
    plt.suptitle(f"Relative differences across phase space for {label1} and {label2} {f'({process})' if process else ''}")
    plt.tight_layout()
    plt.savefig(plot_path, format="pdf")
    print(f"Saved heatmap plot to {plot_path}")

def signed_difference_histogram(ref_amps, new_amps, label: str = "new", process: str = None):
    """
    Plots a grouped bar histogram showing the sign of differences (new - ref) 
    binned by the order of magnitude of the reference amplitude.

    :param ref_amps: Reference amplitude values (1D array)
    :param new_amps: New amplitude values to compare (1D array)
    :param label: Label for the new amplitude set (e.g., "m" or "f")
    :param process: Optional process name to include in the plot title and filename
    """

    # Take absolute of ref, and signed diff between new and ref
    ref = np.abs(ref_amps)
    diff = new_amps - ref_amps
    sign = np.sign(diff)

    # Handle zeros in reference (map to smallest nonzero)
    ref_safe = np.where(ref == 0, np.min(ref[ref > 0]), ref)
    log_ref = np.log10(ref_safe)

    # Bin setup based on reference amplitudes
    min_log = np.floor(log_ref.min())
    max_log = np.ceil(log_ref.max())
    bin_edges = np.arange(min_log, max_log + 1)
    bin_centers = bin_edges[:-1] + 0.5
    n_bins = len(bin_centers)

    # Count signs in each bin
    neg_counts = np.zeros(n_bins)
    pos_counts = np.zeros(n_bins)

    bin_indices = np.digitize(log_ref, bin_edges) - 1  # adjust for 0-based indexing

    for i in range(len(sign)):
        bin_idx = bin_indices[i]
        if 0 <= bin_idx < n_bins:
            if sign[i] < 0:
                neg_counts[bin_idx] += 1
            elif sign[i] > 0:
                pos_counts[bin_idx] += 1
    
    for i in range(n_bins):
        #Normalise each bin to the total number of events in that bin
        total = neg_counts[i] + pos_counts[i]
        if total > 0:
            neg_counts[i] /= total
            pos_counts[i] /= total

    # Plot setup
    bar_width = 0.4
    offsets = [-bar_width / 2, bar_width / 2]

    plot_name = process_to_name(process) + f"_{label}" if process else f"{label}_sign_diff"
    plot_dir = Path.cwd() / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_title = f"Sign of amplitude difference per amplitude"
    if process:
        plot_title += f" for process: {process}"

    xtick_labels = [f"$10^{{{int(b)}}}$" for b in bin_edges[:-1]]

    # --- Linear scale plot ---
    plt.figure(figsize=(10, 6))
    plt.bar(bin_centers + offsets[0], neg_counts, width=bar_width, label="Negative diff")
    plt.bar(bin_centers + offsets[1], pos_counts, width=bar_width, label="Positive diff")

    plt.xticks(bin_centers, xtick_labels, rotation=45)
    plt.xlabel("Amplitude evaluated with FP64")
    plt.ylabel("Fraction of events per bin")
    plt.title(plot_title)
    plt.legend()
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    plot_path = plot_dir / f"{plot_name}_lin.pdf"
    plt.savefig(plot_path, format="pdf")
    print(f"Saved linear-scale plot to {plot_path}")

    # --- Log scale plot ---
    plt.yscale("log")
    plot_path = plot_dir / f"{plot_name}_log.pdf"
    plt.savefig(plot_path, format="pdf")
    print(f"Saved log-scale plot to {plot_path}")
    plt.close()


def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_filename>")
        sys.exit(1)

#    output = sys.argv[1]
    input_filename = sys.argv[1]
    if not os.path.exists(input_filename):
        print(f"Error: {input_filename} does not exist.")
        sys.exit(1)
    if not os.path.isfile(input_filename):
        print(f"Error: {input_filename} is not a file.")
        sys.exit(1)
        
    #read input file, split by newlines
    with open(input_filename, "r") as f:
        processes = f.readlines()
    if not processes:
        print("Error: No processes found in the input file.")
        sys.exit(1)

    for process in processes:
        process = process.strip()
        if not process:
            print("Error: Empty process found in the input file.")
            sys.exit(1)
    
    # clear_amps_files()
    curr_path = Path.cwd()
    run_cards = curr_path / "run_cards"
    if not run_cards.exists(): 
        os.makedirs(run_cards)
        print(f"Created directory {run_cards}.")
    if not run_cards.is_dir():
        print(f"Error: {run_cards} is not a directory.")
        return
    # process = "g g > t t~"
    #processes = ["e+ e- > a a", "e+ e- > a a a", "e+ e- > a a a a"]
    for process in processes:
        proc_name = process_to_name(process)
        run_card_path = write_run_card(run_cards, process)
        proc_dir = curr_path / (proc_name + ".fp")
        if proc_dir.exists():
            print(f"Warning: {proc_dir} already exists. Assuming it is correct process.")
        else:
            generate_and_run_madevent(run_card_path)
        if not proc_dir.exists():
            print(f"Error: {proc_dir} does not exist.")
            sys.exit(1)
        fp_types = ["d", "f", "m"]
        output = {fp: f"{proc_name}.fp/results_{fp}.txt" for fp in fp_types}
        for fp in fp_types:
            compile_and_run(process_dir=proc_dir,fp_flag=fp, output_filename=output[fp])
        # fp_types = ["d", "f", "m"]
        # output = {fp: f"results_{fp}.txt" for fp in fp_types}
        # for fp in fp_types:
        #     compile_and_run(fp_flag = fp, output_filename=output[fp])
        # # run_and_collect(executable_suffix = "fortran", output_filename="results_fortran.txt")
        # Load results into numpy arrays
        all_results = load_amps_results(output)
        # fortran_results = load_amps_results({"fortran": "results_fortran.txt"})
        # fortran_max = np.max(fortran_results["fortran"])
        # print(f"Max fortran result: {fortran_max}")
        # fortran_min = np.min(fortran_results["fortran"])
        # print(f"Min fortran result: {fortran_min}")
        double = all_results["d"]
        results = {"m": all_results["m"], "f": all_results["f"]}
        reldiffs = {}
        # Write the averages of the results to the terminal
        for label, data in results.items():
            if data.size != double.size:
                print(f"Warning: Size mismatch for {label} and doubles.")
                continue
            if data.size > 0:
                signed_difference_histogram(double, data, label=label, process=process)
                abs_diff = np.abs(data - double)
                avg_abs_diff = np.mean(abs_diff)
                print(f"Average absolute difference for {label}: {avg_abs_diff}")
                max_abs_diff = np.max(abs_diff)
                print(f"Max absolute difference for {label}: {max_abs_diff}")
                min_abs_diff = np.min(abs_diff)
                print(f"Min absolute difference for {label}: {min_abs_diff}")
                rel_diff = abs_diff / np.abs(double)
                reldiffs[label] = rel_diff
                avg_rel_diff = np.mean(rel_diff)
                print(f"Average relative difference for {label}: {avg_rel_diff}")
                max_rel_diff = np.max(rel_diff)
                print(f"Max relative difference for {label}: {max_rel_diff}")
                min_rel_diff = np.min(rel_diff)
                print(f"Min relative difference for {label}: {min_rel_diff}")
                print("\n")
                plot_relative_diff_histogram(double, abs_diff, rel_diff, label=label, process=process)
        #         # avg = np.mean(data)
        #         # print(f"Average of {label}: {avg}")
        #     else:
        #         print(f"No data for {label}")
        # # Plot histograms of relative differences
        # # compare_relative_diff_histograms(reldiffs)
        grouped_relative_diff_histogram(reldiffs, process=process)
        plot_comp_difference_heatmaps(double, reldiffs["m"], reldiffs["f"], label1="mixed", label2="FP32", process=process)

if __name__ == "__main__":
    main()

