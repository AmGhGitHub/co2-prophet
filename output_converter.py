"""
CO2 Prophet Output Converter Module
Handles conversion of OUTPUT files to CSV format.
"""

import csv
import os

# Let's try git


def convert_output_to_csv(input_dir: str, output_dir: str) -> None:
    """
    Convert CO2 Prophet OUTPUT files to CSV format.

    Args:
        input_dir: Directory containing OUTPUT_* files
        output_dir: Directory to save CSV files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

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
