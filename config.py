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
PROPHET_DATA_OUTPUT_DIR = Path("C:/vDos/Prophet/sen-datafiles")
PROPHET_RESULTS_DIR = Path("C:/vDos/Prophet/sen-output")
PROPHET_CSV_OUTPUT_DIR = Path("C:/vDos/Prophet/sen-output-csv")
PROPHET_CSV_VARS_DIR = Path("C:/vDos/Prophet/sen-runs")  # Copy of input CSV file

# Visualization outputs
PLOT_OUTPUT_FILE = PROPHET_CSV_OUTPUT_DIR / "oil_vs_injected.png"

# Configuration parameters for file processing
OUTPUT_PREFIX = "sen"

# Input file configuration
INPUT_GENERATOR_CONFIG = {
    "base_file": str(BASE_FILE),
    "csv_file": str(INPUT_CSV_FILE),
    "output_prefix": OUTPUT_PREFIX,
    "output_dir": str(PROPHET_DATA_OUTPUT_DIR),
    "vdos_csv_dir": str(PROPHET_CSV_VARS_DIR),
}

# Output converter configuration
OUTPUT_CONVERTER_CONFIG = {
    "input_dir": str(PROPHET_RESULTS_DIR),
    "output_dir": str(PROPHET_CSV_OUTPUT_DIR),
}

# Plotter configuration
PLOTTER_CONFIG = {
    "csv_dir": str(PROPHET_CSV_OUTPUT_DIR),
    "output_plot": str(PLOT_OUTPUT_FILE),
}

# Parameter generation configuration (sensitivity analysis)
PARAMETER_GENERATOR_CONFIG = {
    "dpcoef_range": (0.7, 0.95),
    "permav_range": (100, 1000),
    "thick_range": (14, 16),
    "poros_range": (0.08, 0.13),
    "nlayers_range": (3, 9),
    "distributions": {
        "DPCOEF": "uniform",
        "PERMAV": "lognormal",
        "THICK": "uniform",
        "POROS": "normal",
        "NLAYERS": "uniform",
    },
}

# Task configuration - Enable/disable specific tasks
TASKS = {
    "generate_input_files": True,  # Generate input files from CSV
    "convert_output_to_csv": True,  # Convert OUTPUT files to CSV
    "plot_results": True,  # Plot Oil produced vs Injected total
}

# Available tasks description
AVAILABLE_TASKS = {
    "generate_input_files": "Generate input files from CSV parameters",
    "convert_output_to_csv": "Convert OUTPUT files to CSV format",
    "plot_results": "Plot Oil produced vs Injected total",
}


def get_all_paths():
    """Return a dictionary of all configured paths for inspection."""
    return {
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "BASE_FILE": str(BASE_FILE),
        "INPUT_CSV_FILE": str(INPUT_CSV_FILE),
        "PROPHET_DATA_OUTPUT_DIR": str(PROPHET_DATA_OUTPUT_DIR),
        "PROPHET_RESULTS_DIR": str(PROPHET_RESULTS_DIR),
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
        status = "✓ ENABLED" if TASKS[task_name] else "✗ DISABLED"
        print(f"{status:12} | {task_name:25} | {description}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Display all paths when module is run directly
    import json

    print("All configured paths:")
    print(json.dumps(get_all_paths(), indent=2))
    print_task_info()
