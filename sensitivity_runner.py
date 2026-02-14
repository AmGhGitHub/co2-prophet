"""
Sensitivity Runner Module
Generates parameters and input files for CO2 Prophet sensitivity analysis.

This module can be:
1. Called as a function from other scripts
2. Run directly with CLI arguments for automation (e.g., Power Automate)

Usage:
    # As a module
    from sensitivity_runner import run_sensitivity_analysis
    run_sensitivity_analysis(sensitivity_level="medium")

    # From command line
    python sensitivity_runner.py -s medium
    python sensitivity_runner.py --sensitivity high
    python sensitivity_runner.py -s config  # Use config file setting
"""

import argparse
import os
import shutil
import sys

from config import INPUT_GENERATOR_CONFIG, PARAMETER_GENERATOR_CONFIG
from input_generator import process_csv_and_generate_input_files
from param_generator import generate_sensitivity_csv

# Valid sensitivity levels
SENSITIVITY_LEVELS = ["minimum", "low", "medium", "high", "very_high"]


# ANSI color codes for terminal output
class Colors:
    RED = "\033[91m"
    RESET = "\033[0m"


def _print_error(message):
    """Print error message in red."""
    print(f"{Colors.RED}[X] {message}{Colors.RESET}")


def run_sensitivity_analysis(sensitivity_level: str = None, verbose: bool = True):
    """
    Generate parameters and input files for sensitivity analysis.

    Args:
        sensitivity_level: Sensitivity level (minimum/low/medium/high/very_high).
                          If None, uses config file setting.
                          Case-insensitive.
        verbose: If True, print progress messages. Default is True.

    Returns:
        bool: True if successful, False otherwise.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("CO2 Prophet - Parameter Generation")
        print("=" * 60 + "\n")

    try:
        config = PARAMETER_GENERATOR_CONFIG

        # Normalize sensitivity level to lowercase
        if sensitivity_level is not None:
            sensitivity_level = sensitivity_level.lower()
            # Validate against known levels
            if sensitivity_level not in SENSITIVITY_LEVELS:
                _print_error(
                    f"Invalid sensitivity level: '{sensitivity_level}'. "
                    f"Valid options: {', '.join(SENSITIVITY_LEVELS)}\n"
                )
                return False

        # Empty the PROPHET_DATAFILE_DIR folder before generating
        datafile_dir = INPUT_GENERATOR_CONFIG["output_dir"]
        if os.path.exists(datafile_dir):
            if verbose:
                print(f"Cleaning datafile directory: {datafile_dir}")

            for item in os.listdir(datafile_dir):
                item_path = os.path.join(datafile_dir, item)
                try:
                    if os.path.isfile(item_path) or os.path.islink(item_path):
                        os.unlink(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                except Exception as e:
                    _print_error(f"Failed to delete {item_path}: {e}")

            if verbose:
                print(f"[OK] Cleaned {datafile_dir}\n")
        else:
            os.makedirs(datafile_dir, exist_ok=True)
            if verbose:
                print(f"[OK] Created directory: {datafile_dir}\n")

        # Use provided sensitivity level or fall back to config
        if sensitivity_level is None:
            sensitivity_level = config.get("sensitivity_level", "medium")

        if verbose:
            print(f"Sensitivity level: {sensitivity_level.replace('_', ' ').title()}\n")

        # Prepare custom ranges
        custom_ranges = {
            "dpcoef_range": config["dpcoef_range"],
            "poros_range": config["poros_range"],
            "mmp_range": config["mmp_range"],
            "soinit_range": config["soinit_range"],
            "xkvh_range": config["xkvh_range"],
        }

        if verbose:
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

        if verbose:
            actual_runs = config["n_runs"] if config["n_runs"] else "auto-calculated"
            print(f"[OK] Generated {actual_runs} parameter sets successfully\n")

        # Automatically generate input files after parameter generation
        if verbose:
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

            if verbose:
                print("[OK] Input files generated successfully\n")

        except FileNotFoundError as e:
            _print_error(f"File not found: {e}\n")
            return False
        except Exception as e:
            _print_error(f"Error generating input files: {e}\n")
            return False

    except Exception as e:
        _print_error(f"Error generating parameters: {e}\n")
        return False

    if verbose:
        print("=" * 60)
        print("Parameter generation completed!")
        print("=" * 60 + "\n")

    return True


def _parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="CO2 Prophet - Generate parameters and input files for sensitivity analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sensitivity_runner.py -s medium              # Use 'medium' sensitivity (~60 runs)
  python sensitivity_runner.py --sensitivity high     # Use 'high' sensitivity (~300 runs)
  python sensitivity_runner.py -s config              # Use config file setting

Sensitivity levels:
  minimum   - ~7 runs    - Very sparse, quick testing only
  low       - ~12 runs   - Basic coverage, preliminary analysis
  medium    - ~60 runs   - Recommended for most cases
  high      - ~300 runs  - Detailed sensitivity analysis
  very_high - ~600 runs  - Comprehensive analysis
  config    - Use config file setting
        """,
    )

    parser.add_argument(
        "-s",
        "--sensitivity",
        type=str,
        choices=SENSITIVITY_LEVELS + ["config"],
        required=True,
        help="Sensitivity level (minimum/low/medium/high/very_high/config)",
    )

    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress progress messages",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_arguments()

    # Determine sensitivity level
    level = None if args.sensitivity == "config" else args.sensitivity

    # Run analysis
    success = run_sensitivity_analysis(
        sensitivity_level=level,
        verbose=not args.quiet,
    )

    sys.exit(0 if success else 1)
