"""
CO2 Prophet Main Script
Generates parameters and input files for sensitivity analysis.
"""

import argparse
import os
import sys

from config import INPUT_GENERATOR_CONFIG, PARAMETER_GENERATOR_CONFIG
from input_generator import process_csv_and_generate_input_files
from param_generator import generate_sensitivity_csv


# ANSI color codes for terminal output
class Colors:
    RED = "\033[91m"
    RESET = "\033[0m"


def print_error(message):
    """Print error message in red."""
    print(f"{Colors.RED}✗ {message}{Colors.RESET}")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="CO2 Prophet - Generate parameters and input files for sensitivity analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Run with interactive sensitivity selection
  python main.py --help                    # Show this help message
        """,
    )

    return parser.parse_args()


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


def run_parameter_generation():
    """Generate parameters and input files."""
    print("\n" + "=" * 60)
    print("CO2 Prophet - Parameter Generation")
    print("=" * 60 + "\n")

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

        # Prompt for sensitivity level
        sensitivity_level = prompt_sensitivity_level()
        if sensitivity_level is None:
            # Use config file setting
            sensitivity_level = config.get("sensitivity_level", "medium")
            print(
                f"\nUsing config file setting: {sensitivity_level.replace('_', ' ').title()}\n"
            )

        # Prepare custom ranges
        custom_ranges = {
            "dpcoef_range": config["dpcoef_range"],
            "poros_range": config["poros_range"],
            "mmp_range": config["mmp_range"],
            "soinit_range": config["soinit_range"],
            "xkvh_range": config["xkvh_range"],
        }

        print("Generating sensitivity parameters...")
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
        print("Generating input files from CSV...")
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
        sys.exit(1)

    print("=" * 60)
    print("Parameter generation completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    args = parse_arguments()
    run_parameter_generation()
