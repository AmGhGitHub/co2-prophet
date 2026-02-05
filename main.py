"""
CO2 Prophet Main Script
Coordinates input generation, output conversion, and visualization.
Supports selective task execution via command-line arguments.
"""

import argparse
import os
import sys

from config import (
    AVAILABLE_TASKS,
    INPUT_GENERATOR_CONFIG,
    OUTPUT_CONVERTER_CONFIG,
    PLOTTER_CONFIG,
    TASKS,
    get_tasks_to_run,
    print_task_info,
)
from input_generator import process_csv_and_generate_input_files
from output_converter import convert_output_to_csv
from plotter import plot_oil_vs_injected


def parse_arguments():
    """Parse command-line arguments for task selection."""
    parser = argparse.ArgumentParser(
        description="CO2 Prophet - Run selected tasks for simulation and analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Run all enabled tasks (default)
  python main.py --tasks input plot        # Run only input generation and plotting
  python main.py --skip output csv         # Skip output conversion and plotting
  python main.py --info                    # Show task configuration info
  python main.py --all                     # Force run all tasks (override config)
  python main.py --interactive             # Prompt user to select tasks
        """,
    )

    parser.add_argument(
        "--tasks",
        nargs="+",
        help="Specify which tasks to run (space-separated). Options: generate_input_files, convert_output_to_csv, plot_results",
        metavar="TASK",
    )

    parser.add_argument(
        "--skip",
        nargs="+",
        help="Specify which tasks to skip (space-separated). Options: generate_input_files, convert_output_to_csv, plot_results",
        metavar="TASK",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tasks regardless of config settings",
    )

    parser.add_argument(
        "--info",
        action="store_true",
        help="Display task configuration and exit",
    )

    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Prompt user to select tasks interactively",
    )

    return parser.parse_args()


def get_task_aliases():
    """Return short aliases for task names."""
    return {
        "input": "generate_input_files",
        "output": "convert_output_to_csv",
        "csv": "convert_output_to_csv",
        "plot": "plot_results",
        "generate": "generate_input_files",
        "convert": "convert_output_to_csv",
    }


def resolve_task_names(task_list):
    """Convert task aliases to full task names."""
    aliases = get_task_aliases()
    resolved = []
    for task in task_list:
        resolved.append(aliases.get(task.lower(), task.lower()))
    return resolved


def validate_tasks(task_list):
    """Validate that all specified tasks are valid."""
    valid_tasks = set(AVAILABLE_TASKS.keys())
    for task in task_list:
        if task not in valid_tasks:
            print(f"Error: Unknown task '{task}'")
            print(f"Valid tasks: {', '.join(sorted(valid_tasks))}")
            sys.exit(1)


def determine_tasks_to_run(args):
    """Determine which tasks to run based on arguments."""
    tasks_config = TASKS.copy()

    if args.all:
        # Force all tasks to run
        return {task: True for task in AVAILABLE_TASKS.keys()}

    if args.tasks:
        # User specified exact tasks to run
        tasks_to_run = resolve_task_names(args.tasks)
        validate_tasks(tasks_to_run)
        return {task: task in tasks_to_run for task in AVAILABLE_TASKS.keys()}

    if args.skip:
        # User specified tasks to skip
        tasks_to_skip = resolve_task_names(args.skip)
        validate_tasks(tasks_to_skip)
        return {task: task not in tasks_to_skip for task in AVAILABLE_TASKS.keys()}

    # Use config file settings
    return tasks_config


def prompt_user_for_tasks():
    """Interactively prompt user to select which tasks to run."""
    print("\n" + "=" * 70)
    print("CO2 PROPHET - TASK SELECTION")
    print("=" * 70 + "\n")

    tasks_list = list(AVAILABLE_TASKS.items())
    tasks_config = {}

    print("Please select which tasks you want to execute:\n")

    for idx, (task_name, description) in enumerate(tasks_list, 1):
        default_status = TASKS.get(task_name, True)
        default_str = "[default: YES]" if default_status else "[default: NO]"

        while True:
            response = (
                input(f"{idx}. {description}\n   Execute? (y/n) {default_str}: ")
                .strip()
                .lower()
            )

            if response == "":
                # Use default
                tasks_config[task_name] = default_status
                print()
                break
            elif response in ["y", "yes"]:
                tasks_config[task_name] = True
                print()
                break
            elif response in ["n", "no"]:
                tasks_config[task_name] = False
                print()
                break
            else:
                print("   Please enter 'y', 'n', or press Enter for default.\n")

    # Show summary
    print("=" * 70)
    enabled_tasks = [task for task, enabled in tasks_config.items() if enabled]
    print(f"Summary: {len(enabled_tasks)} task(s) selected")
    print("=" * 70 + "\n")

    return tasks_config


def run_tasks(tasks_config):
    """Execute the selected tasks."""
    enabled_tasks = [task for task, enabled in tasks_config.items() if enabled]

    if not enabled_tasks:
        print("No tasks enabled. Use --all or --tasks to specify tasks to run.")
        return

    print("\n" + "=" * 60)
    print(f"Running {len(enabled_tasks)} task(s)")
    print("=" * 60 + "\n")

    if tasks_config.get("generate_input_files"):
        print("[1/3] Generating input files from CSV...")
        try:
            # Check if base file exists
            base_file = INPUT_GENERATOR_CONFIG["base_file"]
            if not os.path.exists(base_file):
                raise FileNotFoundError(f"Base file not found: {base_file}")

            # Check if CSV file exists
            csv_file = INPUT_GENERATOR_CONFIG["csv_file"]
            if not os.path.exists(csv_file):
                raise FileNotFoundError(f"CSV file not found: {csv_file}")

            process_csv_and_generate_input_files(
                base_file,
                csv_file,
                INPUT_GENERATOR_CONFIG["output_prefix"],
                INPUT_GENERATOR_CONFIG["output_dir"],
            )
            print("✓ Input files generated successfully\n")
        except FileNotFoundError as e:
            print(f"✗ File not found: {e}\n")
        except Exception as e:
            print(f"✗ Error generating input files: {e}\n")

    if tasks_config.get("convert_output_to_csv"):
        print("[2/3] Converting OUTPUT files to CSV...")
        try:
            input_dir = OUTPUT_CONVERTER_CONFIG["input_dir"]

            # Check if directory exists
            if not os.path.exists(input_dir):
                print(f"⚠ Input directory not found: {input_dir}")
                print(f"  Skipping output conversion and plotting.\n")
                # Skip plotting task too since we don't have CSV files
                tasks_config["plot_results"] = False
            else:
                # Check if directory has any OUTPUT_* files
                output_files = [
                    f for f in os.listdir(input_dir) if f.startswith("OUTPUT_")
                ]
                if not output_files:
                    print(f"⚠ No OUTPUT files found in: {input_dir}")
                    print(f"  Skipping output conversion and plotting.\n")
                    # Skip plotting task too since we won't have CSV files
                    tasks_config["plot_results"] = False
                else:
                    output_dir = OUTPUT_CONVERTER_CONFIG["output_dir"]
                    convert_output_to_csv(input_dir, output_dir)
                    print("✓ Output files converted successfully\n")
        except Exception as e:
            print(f"✗ Error converting output files: {e}\n")
            # Skip plotting if conversion failed
            tasks_config["plot_results"] = False

    if tasks_config.get("plot_results"):
        print("[3/3] Plotting results...")
        try:
            csv_dir = PLOTTER_CONFIG["csv_dir"]

            # Check if directory exists
            if not os.path.exists(csv_dir):
                print(f"⚠ CSV directory not found: {csv_dir}")
                print(f"  Skipping plotting.\n")
            else:
                # Check if directory has any CSV files
                csv_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]
                if not csv_files:
                    print(f"⚠ No CSV files found in: {csv_dir}")
                    print(f"  Skipping plotting.\n")
                else:
                    plot_oil_vs_injected(
                        csv_dir,
                        PLOTTER_CONFIG["output_plot"],
                    )
                    print("✓ Results plotted successfully\n")
        except Exception as e:
            print(f"✗ Error plotting results: {e}\n")

    print("=" * 60)
    print("Pipeline completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    args = parse_arguments()

    if args.info:
        print_task_info()
        sys.exit(0)

    if args.interactive:
        tasks_to_run = prompt_user_for_tasks()
    else:
        tasks_to_run = determine_tasks_to_run(args)

    run_tasks(tasks_to_run)
