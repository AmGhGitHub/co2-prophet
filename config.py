"""
CO2 Prophet Configuration File
Centralized management of all file and folder paths used across the project.
"""

from pathlib import Path

# Base directory paths
PROJECT_ROOT = Path(__file__).parent
BASE_DIR = PROJECT_ROOT

# Input files and directories
BASE_FILE = BASE_DIR / "base"
INPUT_CSV_FILE = BASE_DIR / "sen-runs" / "sen_fbv.csv"

# Output directories for Prophet simulations
PROPHET_DATAFILE_DIR = Path("C:/vDos/Prophet/sen-datafiles")
PROPHET_OUTPUT_DIR = Path("C:/vDos/Prophet/sen-output")
PROPHET_CSV_OUTPUT_DIR = Path("C:/vDos/Prophet/sen-output-csv")
PROPHET_CSV_VARS_DIR = Path("C:/vDos/Prophet/sen-runs")  # Copy of input CSV file

# Visualization outputs
RESULTS_DIR = BASE_DIR / "results"
PLOT_OUTPUT_FILE = RESULTS_DIR / "oil_vs_injected.png"

# Configuration parameters for file processing
OUTPUT_PREFIX = "sen"

# Input file configuration
INPUT_GENERATOR_CONFIG = {
    "base_file": str(BASE_FILE),
    "csv_file": str(INPUT_CSV_FILE),
    "output_prefix": OUTPUT_PREFIX,
    "output_dir": str(PROPHET_DATAFILE_DIR),
    "vdos_csv_dir": str(PROPHET_CSV_VARS_DIR),
}

# Output converter configuration
OUTPUT_CONVERTER_CONFIG = {
    "input_dir": str(PROPHET_OUTPUT_DIR),
    "output_dir": str(PROPHET_CSV_OUTPUT_DIR),
}

# Plotter configuration
PLOTTER_CONFIG = {
    "csv_dir": str(PROPHET_CSV_OUTPUT_DIR),
    "output_plot": str(PLOT_OUTPUT_FILE),
}

# Results analyzer configuration
RESULTS_ANALYZER_CONFIG = {
    "csv_dir": str(PROPHET_CSV_OUTPUT_DIR),
    "output_file": str(RESULTS_DIR / "summary_metrics.csv"),
}

# Parameter generation configuration (sensitivity analysis)
#
# Sensitivity Level Guidelines (for 6 parameters):
#   - 'minimum':    ~7 runs   (n_params + 1) - Very sparse, quick testing only
#   - 'low':        ~12 runs  (2 * n_params) - Basic coverage, preliminary analysis
#   - 'medium':     ~60 runs  (10 * n_params) - Recommended for most cases
#   - 'high':       ~300 runs (50 * n_params) - Detailed sensitivity analysis
#   - 'very_high':  ~600 runs (100 * n_params) - Comprehensive analysis
#
PARAMETER_GENERATOR_CONFIG = {
    "output_file": str(INPUT_CSV_FILE),
    "backup_dir": str(PROPHET_CSV_VARS_DIR),
    "n_runs": None,  # Auto-calculate based on parameters (or set a specific number)
    "sensitivity_level": "medium",  # 'minimum', 'low', 'medium', 'high', 'very_high'
    "seed": 42,  # Random seed (set to None for different data each run)
    "use_lhs": True,  # Use Latin Hypercube Sampling for better parameter space coverage
    "dpcoef_range": (0.3, 0.99),
    "poros_range": (0.08, 0.13),
    "mmp_range": (1200, 2200),
    "soinit_range": (0.4, 0.6),
    "xkvh_range": (0.01, 0.1),
    "distributions": {
        "DPCOEF": "uniform",
        "POROS": "normal",
        "MMP": "uniform",
        "SOINIT": "uniform",
        "XKVH": "uniform",
    },
}

# Task configuration - Enable/disable specific tasks
TASKS = {
    "generate_parameters": False,  # Generate random parameters for sensitivity analysis
    "extract_key_metrics": True,  # Extract oil produced at 1 and 2 HCPV
    "plot_results": True,  # Plot Oil produced vs Injected total
    "ml_analysis": False,  # Machine learning analysis
}

# Available tasks description
AVAILABLE_TASKS = {
    "generate_parameters": "Generate parameters & input files (combined)",
    "extract_key_metrics": "Extract key metrics (Oil at 1 & 2 HCPV) from results",
    "plot_results": "Plot Oil produced vs Injected total",
    "ml_analysis": "Machine Learning analysis (correlations & predictive models)",
}


def get_all_paths():
    """Return a dictionary of all configured paths for inspection."""
    return {
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "BASE_FILE": str(BASE_FILE),
        "INPUT_CSV_FILE": str(INPUT_CSV_FILE),
        "PROPHET_DATAFILE_DIR": str(PROPHET_DATAFILE_DIR),
        "PROPHET_OUTPUT_DIR": str(PROPHET_OUTPUT_DIR),
        "PROPHET_CSV_OUTPUT_DIR": str(PROPHET_CSV_OUTPUT_DIR),
        "PROPHET_CSV_VARS_DIR": str(PROPHET_CSV_VARS_DIR),
        "PLOT_OUTPUT_FILE": str(PLOT_OUTPUT_FILE),
    }


def get_tasks_to_run():
    """Return a dictionary of enabled tasks."""
    return {task: enabled for task, enabled in TASKS.items() if enabled}


def print_task_info():
    """Print available tasks and their current status."""
    print("\n" + "=" * 60)
    print("AVAILABLE TASKS")
    print("=" * 60)
    for task_name, description in AVAILABLE_TASKS.items():
        status = "[OK] ENABLED" if TASKS[task_name] else "[X] DISABLED"
        print(f"{status:12} | {task_name:25} | {description}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Display all paths when module is run directly
    import json

    print("All configured paths:")
    print(json.dumps(get_all_paths(), indent=2))
    print_task_info()
