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
    PARAMETER_GENERATOR_CONFIG,
    PLOTTER_CONFIG,
    RESULTS_ANALYZER_CONFIG,
    TASKS,
    get_tasks_to_run,
    print_task_info,
)
from input_generator import process_csv_and_generate_input_files
from output_converter import convert_output_to_csv
from param_generator import generate_sensitivity_csv
from plotter import plot_oil_vs_injected
from results_analyzer import (
    extract_key_metrics,
    merge_parameters_with_results,
    print_summary_statistics,
)


# ANSI color codes for terminal output
class Colors:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def print_error(message):
    """Print error message in red."""
    print(f"{Colors.RED}✗ {message}{Colors.RESET}")


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

    print("Please select which tasks you want to execute:\n")

    for idx, (task_name, description) in enumerate(tasks_list, 1):
        print(f"{idx}. {description}")

    print(f"{len(tasks_list) + 1}. Run tasks 2&3 (Extract metrics + Plot)")
    print(f"{len(tasks_list) + 2}. Run ALL tasks")
    print("0. Exit without running anything\n")

    while True:
        try:
            choice = input(
                "Enter your choice (e.g., 1 or 1,2,3 for multiple): "
            ).strip()

            if choice == "0":
                print("\nExiting...")
                sys.exit(0)

            # Parse multiple selections
            selected = [int(x.strip()) for x in choice.split(",")]

            # Validate selections
            if any(s < 0 or s > len(tasks_list) + 2 for s in selected):
                print(
                    f"Invalid choice. Please enter numbers between 0 and {len(tasks_list) + 2}.\n"
                )
                continue

            # Build tasks config
            tasks_config = {}

            if (len(tasks_list) + 2) in selected:
                # Run all tasks (option 6)
                for task_name, _ in tasks_list:
                    tasks_config[task_name] = True
            elif (len(tasks_list) + 1) in selected:
                # Run tasks 2&3 (option 4: extract metrics + plot)
                tasks_config["generate_parameters"] = False
                tasks_config["extract_key_metrics"] = True
                tasks_config["plot_results"] = True
            else:
                # Run selected tasks only
                for idx, (task_name, _) in enumerate(tasks_list, 1):
                    tasks_config[task_name] = idx in selected

            # Show summary
            print("\n" + "=" * 70)
            enabled_tasks = [
                AVAILABLE_TASKS[task]
                for task, enabled in tasks_config.items()
                if enabled
            ]
            if enabled_tasks:
                print(f"Selected {len(enabled_tasks)} task(s):")
                for i, task_desc in enumerate(enabled_tasks, 1):
                    print(f"  {i}. {task_desc}")
            else:
                print("No tasks selected.")
            print("=" * 70 + "\n")

            return tasks_config

        except ValueError:
            print(
                "Invalid input. Please enter numbers separated by commas (e.g., 1 or 1,2,3).\n"
            )
        except KeyboardInterrupt:
            print("\n\nExiting...")
            sys.exit(0)


def prompt_sensitivity_level():
    """Prompt user to select sensitivity level for parameter generation."""
    print("\n" + "=" * 70)
    print("SELECT SENSITIVITY LEVEL")
    print("=" * 70 + "\n")

    sensitivity_options = {
        1: ("minimum", "~7 runs", "Very sparse, quick testing only"),
        2: ("low", "~12 runs", "Basic coverage, preliminary analysis"),
        3: ("medium", "~60 runs", "Recommended for most cases"),
        4: ("high", "~300 runs", "Detailed sensitivity analysis"),
        5: ("very_high", "~600 runs", "Comprehensive analysis"),
    }

    print("Choose sensitivity level for parameter generation (5 parameters):\n")
    for key, (level, runs, desc) in sensitivity_options.items():
        marker = " [Recommended]" if level == "medium" else ""
        print(
            f"{key}. {level.replace('_', ' ').title():12} - {runs:10} - {desc}{marker}"
        )

    print("\n0. Use config file setting\n")

    while True:
        try:
            choice = input("Enter your choice (0-5): ").strip()

            if choice == "0":
                return None  # Use config file setting

            choice_num = int(choice)
            if choice_num in sensitivity_options:
                selected_level = sensitivity_options[choice_num][0]
                print(
                    f"\n✓ Selected: {selected_level.replace('_', ' ').title()} ({sensitivity_options[choice_num][1]})"
                )
                return selected_level
            else:
                print(f"Invalid choice. Please enter a number between 0 and 5.\n")
        except ValueError:
            print("Invalid input. Please enter a number.\n")
        except KeyboardInterrupt:
            print("\n\nUsing config file setting...")
            return None


def run_tasks(tasks_config):
    """Execute the selected tasks."""
    enabled_tasks = [task for task, enabled in tasks_config.items() if enabled]

    if not enabled_tasks:
        print("No tasks enabled. Use --all or --tasks to specify tasks to run.")
        return

    print("\n" + "=" * 60)
    print(f"Running {len(enabled_tasks)} task(s)")
    print("=" * 60 + "\n")

    if tasks_config.get("generate_parameters"):
        print("[0/5] Generating random parameters for sensitivity analysis...")
        try:
            config = PARAMETER_GENERATOR_CONFIG

            # Empty the PROPHET_DATAFILE_DIR folder before generating
            datafile_dir = INPUT_GENERATOR_CONFIG["output_dir"]
            if os.path.exists(datafile_dir):
                print(f"Cleaning datafile directory: {datafile_dir}")
                import shutil

                for item in os.listdir(datafile_dir):
                    item_path = os.path.join(datafile_dir, item)
                    try:
                        if os.path.isfile(item_path) or os.path.islink(item_path):
                            os.unlink(item_path)
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                    except Exception as e:
                        print_error(f"Failed to delete {item_path}: {e}")
                print(f"✓ Cleaned {datafile_dir}\n")
            else:
                os.makedirs(datafile_dir, exist_ok=True)
                print(f"✓ Created directory: {datafile_dir}\n")

            # Prompt for sensitivity level if running interactively
            sensitivity_level = prompt_sensitivity_level()
            if sensitivity_level is None:
                # Use config file setting
                sensitivity_level = config.get("sensitivity_level", "medium")
                print(
                    f"Using config file setting: {sensitivity_level.replace('_', ' ').title()}"
                )

            # Prepare custom ranges
            custom_ranges = {
                "dpcoef_range": config["dpcoef_range"],
                "poros_range": config["poros_range"],
                "mmp_range": config["mmp_range"],
                "soinit_range": config["soinit_range"],
                "xkvh_range": config["xkvh_range"],
            }

            generate_sensitivity_csv(
                output_file=config["output_file"],
                n_runs=config["n_runs"],
                seed=config["seed"],
                custom_ranges=custom_ranges,
                custom_distributions=config["distributions"],
                backup_dir=config["backup_dir"],
                use_lhs=config.get("use_lhs", True),
                sensitivity_level=sensitivity_level,
            )
            actual_runs = config["n_runs"] if config["n_runs"] else "auto-calculated"
            print(f"✓ Generated {actual_runs} parameter sets successfully\n")

            # Automatically generate input files after parameter generation
            print("[1/5] Generating input files from CSV (auto)...")
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
                    INPUT_GENERATOR_CONFIG["vdos_csv_dir"],
                )
                print("✓ Input files generated successfully\n")
            except FileNotFoundError as e:
                print_error(f"File not found: {e}\n")
            except Exception as e:
                print_error(f"Error generating input files: {e}\n")

        except Exception as e:
            print_error(f"Error generating parameters: {e}\n")

    elif tasks_config.get("generate_input_files"):
        print("[1/5] Generating input files from CSV...")
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
                INPUT_GENERATOR_CONFIG["vdos_csv_dir"],
            )
            print("✓ Input files generated successfully\n")
        except FileNotFoundError as e:
            print_error(f"File not found: {e}\n")
        except Exception as e:
            print_error(f"Error generating input files: {e}\n")

    if tasks_config.get("convert_output_to_csv"):
        print("[2/5] Converting OUTPUT files to CSV...")
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
            print_error(f"Error converting output files: {e}\n")
            # Skip plotting if conversion failed
            tasks_config["plot_results"] = False

    if tasks_config.get("extract_key_metrics"):
        print("[3/5] Extracting key metrics from results...")
        try:
            csv_dir = RESULTS_ANALYZER_CONFIG["csv_dir"]
            output_file = RESULTS_ANALYZER_CONFIG["output_file"]

            # Check if directory exists
            if not os.path.exists(csv_dir):
                print(f"⚠ CSV directory not found: {csv_dir}")
                print(f"  Skipping metrics extraction.\n")
            else:
                # Check if directory has any CSV files
                csv_files = [
                    f
                    for f in os.listdir(csv_dir)
                    if f.startswith("OUTPUT_") and f.endswith(".csv")
                ]
                if not csv_files:
                    print(f"⚠ No OUTPUT CSV files found in: {csv_dir}")
                    print(f"  Skipping metrics extraction.\n")
                else:
                    summary_df = extract_key_metrics(csv_dir, output_file)
                    print_summary_statistics(summary_df)
                    print("✓ Key metrics extracted successfully\n")

                    # Automatically merge with parameters CSV
                    print("Merging parameters with results...")
                    try:
                        params_csv = PARAMETER_GENERATOR_CONFIG["output_file"]
                        results_dir = os.path.dirname(output_file)
                        merged_output = os.path.join(
                            results_dir, "sen_fbv_with_results.csv"
                        )

                        if os.path.exists(params_csv):
                            merge_parameters_with_results(
                                params_csv=params_csv,
                                metrics_csv=output_file,
                                output_file=merged_output,
                                verbose=True,
                            )
                        else:
                            print(f"⚠ Parameters CSV not found: {params_csv}")
                            print("  Skipping merge step.\n")
                    except Exception as e:
                        print_error(f"Error merging files: {e}\n")
        except Exception as e:
            print_error(f"Error extracting metrics: {e}\n")

    if tasks_config.get("plot_results"):
        print("[4/5] Plotting results...")
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
            print_error(f"Error plotting results: {e}\n")

    if tasks_config.get("ml_analysis"):
        print("[5/5] Running machine learning analysis...")
        try:
            from ml_analyzer import analyze_correlations, build_ml_models

            # Get paths
            results_dir = os.path.dirname(RESULTS_ANALYZER_CONFIG["output_file"])
            merged_csv = os.path.join(results_dir, "sen_fbv_with_results.csv")

            # Check if merged CSV exists
            if not os.path.exists(merged_csv):
                print(f"⚠ Merged CSV not found: {merged_csv}")
                print(
                    f"  Run 'Extract key metrics' task first to generate this file.\n"
                )
            else:
                # Run correlation analysis
                print("\nStep 1: Analyzing correlations...")
                correlations = analyze_correlations(
                    merged_csv, results_dir, verbose=True
                )

                # Build ML models
                print("\nStep 2: Building predictive models...")
                models = build_ml_models(merged_csv, results_dir, verbose=True)

                print("✓ Machine learning analysis completed successfully\n")
        except ImportError as e:
            print_error(f"Missing required libraries for ML analysis: {e}")
            print("  Install with: pip install scikit-learn seaborn\n")
        except Exception as e:
            print_error(f"Error in ML analysis: {e}\n")

    print("=" * 60)
    print("Pipeline completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    args = parse_arguments()

    if args.info:
        print_task_info()
        sys.exit(0)

    # Always prompt user interactively unless specific arguments are provided
    if args.all or args.tasks or args.skip:
        tasks_to_run = determine_tasks_to_run(args)
    else:
        tasks_to_run = prompt_user_for_tasks()

    run_tasks(tasks_to_run)
