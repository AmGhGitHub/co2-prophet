"""
CO2 Prophet Input File Generator Module
Handles generation of input files from CSV parameters.
"""

import csv
import os


def format_value(value: float) -> str:
    """Format a numeric value, removing unnecessary decimal points."""
    if value == int(value):
        return str(int(value))
    return str(value)


def replace_value_preserve_spacing(
    old_part: str, new_value: float, has_newline: bool = False
) -> str:
    """
    Replace a value while preserving spacing alignment.

    Args:
        old_part: Original string part (may include leading spaces)
        new_value: New numeric value to insert
        has_newline: Whether the part ends with newline
    """
    # Handle newline at end
    trailing = ""
    if has_newline and old_part.endswith("\n"):
        old_part = old_part.rstrip("\n")
        trailing = "\n"

    stripped = old_part.lstrip()
    leading_spaces = len(old_part) - len(stripped)

    old_value_len = len(stripped)
    new_value_str = format_value(new_value)
    new_value_len = len(new_value_str)

    # Adjust leading spaces to maintain column alignment
    adjusted_spaces = leading_spaces + (old_value_len - new_value_len)
    adjusted_spaces = max(1, adjusted_spaces)  # At least one space

    return " " * adjusted_spaces + new_value_str + trailing


def generate_input_file(base_file: str, output_file: str, params: dict) -> None:
    """
    Generate a CO2 Prophet input file with modified parameters.

    Args:
        base_file: Path to the base input file
        output_file: Path to the output file
        params: Dictionary with keys: DPCOEF, PERMAV, THICK, POROS, NLAYERS
    """
    with open(base_file, "r") as f:
        lines = f.readlines()

    # Find the line containing the header and modify the next line
    for i, line in enumerate(lines):
        if "DPCOEF" in line and "PERMAV" in line and "THICK" in line:
            # Next line contains the values
            value_line = lines[i + 1]
            parts = value_line.split(",")

            # parts[0] = DPCOEF, parts[1] = PERMAV, parts[2] = THICK
            # parts[3] = POROS, parts[4] = NLAYERS (with newline)

            # First value (DPCOEF) - no leading space in original
            parts[0] = format_value(params["DPCOEF"])

            # Other values - preserve spacing
            parts[1] = replace_value_preserve_spacing(parts[1], params["PERMAV"])
            parts[2] = replace_value_preserve_spacing(parts[2], params["THICK"])
            parts[3] = replace_value_preserve_spacing(parts[3], params["POROS"])
            parts[4] = replace_value_preserve_spacing(
                parts[4], params["NLAYERS"], has_newline=True
            )

            lines[i + 1] = ",".join(parts)
            break

    with open(output_file, "w") as f:
        f.writelines(lines)

    print(
        f"Created '{output_file}' with: DPCOEF={params['DPCOEF']}, PERMAV={params['PERMAV']}, "
        f"THICK={params['THICK']}, POROS={params['POROS']}, NLAYERS={params['NLAYERS']}"
    )


def process_csv_and_generate_input_files(
    base_file: str,
    csv_file: str,
    output_prefix: str = "sen",
    output_file_dir: str = "./sen-datafile/",
) -> None:
    """
    Read parameters from CSV and generate input files.

    Args:
        base_file: Path to the base input file
        csv_file: Path to CSV file with parameters
        output_prefix: Prefix for output files (default: "sen")
        output_file_dir: Directory for output files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_file_dir, exist_ok=True)

    with open(csv_file, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_number = int(row["RUN"])
            params = {
                "DPCOEF": float(row["DPCOEF"]),
                "PERMAV": float(row["PERMAV"]),
                "THICK": float(row["THICK"]),
                "POROS": float(row["POROS"]),
                "NLAYERS": int(row["NLAYERS"]),
            }
            output_file = f"{output_file_dir}/{output_prefix}{run_number}.SAV".upper()
            generate_input_file(base_file, output_file, params)
