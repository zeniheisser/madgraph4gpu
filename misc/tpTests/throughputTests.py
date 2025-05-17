import subprocess
import os
from itertools import product
from statistics import mean, stdev
import json
import sys
import argparse
from pathlib import Path
import csv
from datetime import datetime


def resolve_target_directory(base_dir: Path) -> Path:
    """Find the single P1* subdirectory under base_dir/SubProcesses/"""
    subproc_dir = base_dir / "SubProcesses"
    if not subproc_dir.exists() or not subproc_dir.is_dir():
        raise FileNotFoundError(f"'SubProcesses' not found in {base_dir}")
    
    p1_dirs = [d for d in subproc_dir.iterdir() if d.is_dir() and d.name.startswith("P1")]
    
    if len(p1_dirs) != 1:
        raise ValueError(f"Expected exactly one 'P1*' directory in SubProcesses, found {len(p1_dirs)}.")
    
    return p1_dirs[0]


def run_make(settings: dict, work_dir: Path):
    """Run make cleanall and make with settings in the given directory."""
    print(f"Compiling with settings: {settings}")
    log_file = work_dir / "compilation.log"
    try:
        with open(log_file, "w") as log:
            subprocess.run(["make", "cleanall"], cwd=work_dir, check=True, stdout=log, stderr=log)
            make_command = ["make"] + [f"{k}={v}" for k, v in settings.items()]
            subprocess.run(make_command, cwd=work_dir, check=True, stdout=log, stderr=log)
        os.remove(log_file)
    except subprocess.CalledProcessError:
        print(f"Compilation failed. See '{log_file}' for details.")



def run_executable(args, env_vars, num_runs, work_dir: Path):
    """Run the executable in the given directory."""
    results = []
    env = os.environ.copy()
    env.update(env_vars)

    exe_path = work_dir / "check_cpp.exe"

    for _ in range(num_runs):
        result = subprocess.run(
            [str(exe_path)] + list(map(str, args)),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            cwd=work_dir
        )
        output = result.stdout.strip()
        try:
            value = float(output)
            results.append(value)
        except ValueError:
            print(f"Non-numeric output encountered: {output}")
    return results

def main():
    parser = argparse.ArgumentParser(description="Automated test runner")
    parser.add_argument("directory", help="Base directory containing SubProcesses")
    args = parser.parse_args()

    base_dir = Path(args.directory).resolve()
    work_dir = resolve_target_directory(base_dir)

    num_runs = 5
    runtime_args = (2, 32, 2)

    # Define compilation and environment options
    compilation_options = [
        {"USEOPENMP": "1", "BACKEND": "cppavx2"},
        {"USEOPENMP": "1", "BACKEND": "cpp512y"},
        {"USEOPENMP": "1", "BACKEND": "cpp512z"},
    ]
    environment_options = [
        {"OMP_NUM_THREADS": "1"},
        {"OMP_NUM_THREADS": "2"},
    ]

    results = {}

    for comp_opt in compilation_options:
        comp_key = "_".join(f"{k}={v}" for k, v in comp_opt.items())
        run_make(comp_opt, work_dir)
        results[comp_key] = {}

        for env_opt in environment_options:
            env_key = "_".join(f"{k}={v}" for k, v in env_opt.items())
            print(f"Running for {comp_key} and {env_key}...")
            run_results = run_executable(runtime_args, env_opt, num_runs, work_dir)
            results[comp_key][env_key] = {
                "values": run_results,
                "mean": mean(run_results) if run_results else None
            }

        # Prepare output filename with timestamp and directory tag
    tag = base_dir.name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_name = f"results_{tag}_{timestamp}"
    
    script_dir = Path(__file__).parent.resolve()
    json_path = script_dir / f"{base_output_name}.json"
    csv_path = script_dir / f"{base_output_name}.csv"

    # Save JSON
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"JSON results saved to {json_path}")

    # Save CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Compilation", "Environment", "RunIndex", "Value"])

        for comp_key, env_data in results.items():
            for env_key, data in env_data.items():
                for i, val in enumerate(data["values"]):
                    writer.writerow([comp_key, env_key, i, val])

    print(f"CSV results saved to {csv_path}")

        # Export summary statistics
    summary_csv_path = script_dir / f"{base_output_name}_summary.csv"
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Compilation", "Environment", "Mean", "Min", "Max", "StdDev"])

        for comp_key, env_data in results.items():
            for env_key, data in env_data.items():
                values = data["values"]
                if values:
                    stats_row = [
                        comp_key,
                        env_key,
                        round(mean(values), 6),
                        round(min(values), 6),
                        round(max(values), 6),
                        round(stdev(values), 6) if len(values) > 1 else 0.0
                    ]
                    writer.writerow(stats_row)
                else:
                    writer.writerow([comp_key, env_key, "N/A", "N/A", "N/A", "N/A"])

    print(f"Summary statistics saved to {summary_csv_path}")
    print("All tests completed.")


if __name__ == "__main__":
    main()
