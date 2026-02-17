"""
CO2 Prophet Main Script
Generates parameters and input files for sensitivity analysis.

For CLI automation, use sensitivity_runner.py directly:
    python sensitivity_runner.py -s medium
"""

import sys

from sensitivity_runner import SENSITIVITY_LEVELS, run_sensitivity_analysis


def prompt_sensitivity_level():
    """Prompt user to select sensitivity level for parameter generation."""
    print("\n" + "=" * 70)
    print("SELECT SENSITIVITY LEVEL")
    print("=" * 70 + "\n")

    sensitivity_options = {
        1: ("minimum", "~14 runs", "Very sparse, quick testing only"),
        2: ("low", "~26 runs", "Basic coverage, preliminary analysis"),
        3: ("medium", "~130 runs", "Recommended for most cases"),
        4: ("high", "~650 runs", "Detailed sensitivity analysis"),
        5: ("very_high", "~1300 runs", "Comprehensive analysis"),
    }

    print("Choose sensitivity level for parameter generation (13 parameters):\n")
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
                    f"\n[OK] Selected: {selected_level.replace('_', ' ').title()} ({sensitivity_options[choice_num][1]})"
                )
                return selected_level
            else:
                print("Invalid choice. Please enter a number between 0 and 5.\n")
        except ValueError:
            print("Invalid input. Please enter a number.\n")
        except KeyboardInterrupt:
            print("\n\nUsing config file setting...")
            return None


if __name__ == "__main__":
    # Interactive mode - prompt for sensitivity level
    selected_level = prompt_sensitivity_level()
    success = run_sensitivity_analysis(sensitivity_level=selected_level)
    sys.exit(0 if success else 1)
