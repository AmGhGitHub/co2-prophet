"""
CO2 Prophet Output Converter Module
Handles conversion of OUTPUT files to CSV format and saves labelout files.
"""

import csv
import os
import shutil

from config import PROPHET_CSV_OUTPUT_DIR, PROPHET_LABELOUT_DIR, PROPHET_OUTPUT_DIR

# Let's try git


def convert_output_to_csv(
    input_dir: str,
    output_dir: str,
    labelout_dir: str = None,
    labelout_source_dir: str = None,
) -> None:
    """
    Convert CO2 Prophet OUTPUT files to CSV format and save labelout files.

    Args:
        input_dir: Directory containing OUTPUT_* files
        output_dir: Directory to save CSV files
        labelout_dir: Directory to save labelout files (optional)
        labelout_source_dir: Directory where labelout files are located (default: parent of input_dir)
    """
    # Create output directory if it doesn't exist
    # os.makedirs(output_dir, exist_ok=True)

    # Create labelout directory if specified
    if labelout_dir:
        os.makedirs(labelout_dir, exist_ok=True)

    # Determine labelout source directory (default to parent of input_dir if not specified)
    if labelout_source_dir is None and labelout_dir is not None:
        labelout_source_dir = os.path.dirname(
            input_dir
        )  # Parent directory of sen-output

    # Process all OUTPUT_* files
    for filename in os.listdir(input_dir):
        if filename.startswith("OUTPUT_"):
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, f"{filename}.csv")

            # Read the data file
            with open(input_file, "r") as f:
                lines = f.readlines()

            # Write to CSV with headers
            with open(output_file, "w", newline="") as f:
                writer = csv.writer(f)
                # Write headers
                writer.writerow(
                    ["Injected total", "Oil produced", "CO2 produced", "Water produced"]
                )

                # Write data rows
                for line in lines:
                    line = line.strip()
                    if line:  # Skip empty lines
                        # Split by whitespace and convert to list
                        values = line.split()
                        if len(values) == 4:  # Ensure we have 4 columns
                            writer.writerow(values)

            print(f"Converted {filename} -> {output_file}")

            # Copy corresponding labelout file if it exists and labelout_dir is specified
            if labelout_dir and labelout_source_dir:
                # Extract run number from OUTPUT_X to find labelout file
                run_number = filename.replace("OUTPUT_", "")

                # Try different possible labelout filename patterns
                possible_patterns = [
                    f"labelout_{run_number}",  # labelout_1
                    f"LABELOUT_{run_number}",  # LABELOUT_1
                    f"labelout{run_number}",  # labelout1
                    f"LABELOUT{run_number}",  # LABELOUT1
                    "labelout",  # single labelout file (overwritten each run)
                    "LABELOUT",  # single LABELOUT file (overwritten each run)
                ]

                labelout_copied = False
                for labelout_filename in possible_patterns:
                    labelout_source = os.path.join(
                        labelout_source_dir, labelout_filename
                    )

                    if os.path.exists(labelout_source):
                        # If it's a single labelout file, rename it with run number
                        if labelout_filename in ["labelout", "LABELOUT"]:
                            dest_filename = f"{labelout_filename}_{run_number}"
                        else:
                            dest_filename = labelout_filename

                        labelout_dest = os.path.join(labelout_dir, dest_filename)
                        shutil.copy2(labelout_source, labelout_dest)
                        print(f"  Saved {labelout_filename} -> {labelout_dest}")
                        labelout_copied = True
                        break

                if not labelout_copied:
                    print(
                        f"  Warning: No labelout file found for {filename} in {labelout_source_dir}"
                    )


if __name__ == "__main__":
    convert_output_to_csv(
        PROPHET_OUTPUT_DIR, PROPHET_CSV_OUTPUT_DIR, PROPHET_LABELOUT_DIR
    )
