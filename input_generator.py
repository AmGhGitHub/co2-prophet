"""
CO2 Prophet Input File Generator Module
Handles generation of input files from CSV parameters.
"""

import csv
import os
import shutil


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
        params: Dictionary with keys: DPCOEF, PERMAV, POROS, MMP, SOINIT, SWINIT, XKVH,
                SOLRAT, SORW, SORG, SORM, SGR, SSR, SWC, SWIR, KWRO, KRSMAX, W
    """
    with open(base_file, "r") as f:
        lines = f.readlines()

    # Update MMP value (line with 'TRES     P     MMP')
    for i, line in enumerate(lines):
        if "TRES" in line and "P" in line and "MMP" in line:
            value_line = lines[i + 1]
            parts = value_line.split(",")
            # parts[0] = TRES, parts[1] = P, parts[2] = MMP
            parts[2] = replace_value_preserve_spacing(
                parts[2], params["MMP"], has_newline=True
            )
            lines[i + 1] = ",".join(parts)
            break

    # Update DPCOEF, PERMAV, POROS (line with 'DPCOEF   PERMAV   THICK   POROS   NLAYERS')
    # Note: We're keeping THICK and NLAYERS from base file, only updating DPCOEF, PERMAV, POROS
    for i, line in enumerate(lines):
        if "DPCOEF" in line and "PERMAV" in line:
            value_line = lines[i + 1]
            parts = value_line.split(",")
            # parts[0] = DPCOEF, parts[1] = PERMAV, parts[2] = THICK, parts[3] = POROS, parts[4] = NLAYERS
            parts[0] = format_value(params["DPCOEF"])
            parts[1] = replace_value_preserve_spacing(parts[1], params["PERMAV"])
            # Skip parts[2] (THICK) - keep original value
            parts[3] = replace_value_preserve_spacing(parts[3], params["POROS"])
            # Skip parts[4] (NLAYERS) - keep original value
            lines[i + 1] = ",".join(parts)
            break

    # Update SOINIT and SWINIT (line with 'SOINIT     SGINIT     SWINIT')
    for i, line in enumerate(lines):
        if "SOINIT" in line and "SGINIT" in line and "SWINIT" in line:
            value_line = lines[i + 1]
            parts = value_line.split(",")
            # parts[0] = SOINIT, parts[1] = SGINIT, parts[2] = SWINIT
            parts[0] = format_value(params["SOINIT"])
            # Keep SGINIT as is (parts[1])
            parts[2] = replace_value_preserve_spacing(
                parts[2], params["SWINIT"], has_newline=True
            )
            lines[i + 1] = ",".join(parts)
            break

    # Update XKVH (line with 'AREA     XKVH')
    for i, line in enumerate(lines):
        if "AREA" in line and "XKVH" in line:
            value_line = lines[i + 1]
            parts = value_line.split(",")
            # parts[0] = AREA, parts[1] = XKVH
            # Keep AREA as is (parts[0])
            parts[1] = replace_value_preserve_spacing(
                parts[1], params["XKVH"], has_newline=True
            )
            lines[i + 1] = ",".join(parts)
            break

    # Update SORW, SORG, SORM (line with 'SORW     SORG     SORM')
    for i, line in enumerate(lines):
        if "SORW" in line and "SORG" in line and "SORM" in line:
            value_line = lines[i + 1]
            parts = value_line.split(",")
            # parts[0] = SORW, parts[1] = SORG, parts[2] = SORM
            parts[0] = format_value(params["SORW"])
            parts[1] = replace_value_preserve_spacing(parts[1], params["SORG"])
            parts[2] = replace_value_preserve_spacing(
                parts[2], params["SORM"], has_newline=True
            )
            lines[i + 1] = ",".join(parts)
            break

    # Update SGR, SSR (line with 'SGR     SSR')
    for i, line in enumerate(lines):
        if "SGR" in line and "SSR" in line:
            value_line = lines[i + 1]
            parts = value_line.split(",")
            # parts[0] = SGR, parts[1] = SSR
            parts[0] = format_value(params["SGR"])
            parts[1] = replace_value_preserve_spacing(
                parts[1], params["SSR"], has_newline=True
            )
            lines[i + 1] = ",".join(parts)
            break

    # Update SWC, SWIR (line with 'SWC     SWIR')
    for i, line in enumerate(lines):
        if "SWC" in line and "SWIR" in line:
            value_line = lines[i + 1]
            parts = value_line.split(",")
            # parts[0] = SWC, parts[1] = SWIR
            parts[0] = format_value(params["SWC"])
            parts[1] = replace_value_preserve_spacing(
                parts[1], params["SWIR"], has_newline=True
            )
            lines[i + 1] = ",".join(parts)
            break

    # Update KWRO, KRSMAX (line with 'KROCW     KWRO     KRSMAX     KRGCW')
    for i, line in enumerate(lines):
        if "KROCW" in line and "KWRO" in line and "KRSMAX" in line:
            value_line = lines[i + 1]
            parts = value_line.split(",")
            # parts[0] = KROCW, parts[1] = KWRO, parts[2] = KRSMAX, parts[3] = KRGCW
            # Keep KROCW (parts[0]) and KRGCW (parts[3]) as is
            parts[1] = replace_value_preserve_spacing(parts[1], params["KWRO"])
            parts[2] = replace_value_preserve_spacing(parts[2], params["KRSMAX"])
            lines[i + 1] = ",".join(parts)
            break

    # Update W (line with 'KRMSEL     W')
    for i, line in enumerate(lines):
        if "KRMSEL" in line and "W" in line:
            value_line = lines[i + 1]
            parts = value_line.split(",")
            # parts[0] = KRMSEL, parts[1] = W
            # Keep KRMSEL as is (parts[0])
            parts[1] = replace_value_preserve_spacing(
                parts[1], params["W"], has_newline=True
            )
            lines[i + 1] = ",".join(parts)
            break

    # Update SOLRAT (line with 'HCPVI    WTRRAT    SOLRAT     TMORVL')
    for i, line in enumerate(lines):
        if "HCPVI" in line and "SOLRAT" in line:
            value_line = lines[i + 1]
            parts = value_line.split(",")
            # parts[0] = HCPVI, parts[1] = WTRRAT, parts[2] = SOLRAT, parts[3] = TMORVL
            # Keep HCPVI, WTRRAT, TMORVL as is, only update SOLRAT
            parts[2] = replace_value_preserve_spacing(parts[2], params["SOLRAT"])
            lines[i + 1] = ",".join(parts)
            break

    with open(output_file, "w") as f:
        f.writelines(lines)

    print(
        f"Created '{output_file}' with: DPCOEF={params['DPCOEF']:.2f}, PERMAV={params['PERMAV']:.1f}, "
        f"POROS={params['POROS']:.3f}, MMP={params['MMP']:.0f}, SOINIT={params['SOINIT']:.3f}, "
        f"SWINIT={params['SWINIT']:.3f}, XKVH={params['XKVH']:.2f}, SOLRAT={params['SOLRAT']:.1f}, SORM={params['SORM']:.3f}"
    )


def process_csv_and_generate_input_files(
    base_file: str,
    csv_file: str,
    output_prefix: str = "sen",
    output_file_dir: str = "./sen-datafile/",
    vdos_csv_dir: str = None,
) -> None:
    """
    Read parameters from CSV and generate input files.

    Args:
        base_file: Path to the base input file
        csv_file: Path to CSV file with parameters
        output_prefix: Prefix for output files (default: "sen")
        output_file_dir: Directory for output files
        vdos_csv_dir: Directory to copy CSV file to (optional)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_file_dir, exist_ok=True)

    with open(csv_file, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_number = int(row["RUN"])
            params = {
                "DPCOEF": float(row["DPCOEF"]),
                "PERMAV": 500.0,  # Fixed default value (PERMAV is no longer a sensitivity parameter)
                "POROS": float(row["POROS"]),
                "MMP": float(row["MMP"]),
                "SOINIT": float(row["SOINIT"]),
                "SWINIT": float(row["SWINIT"]),
                "XKVH": float(row["XKVH"]),
                "SOLRAT": float(row["SOLRAT"]),
                # Relative permeability parameters (now included from CSV)
                "SORW": float(row["SORW"]),
                "SORG": float(row["SORG"]),
                "SORM": float(row["SORM"]),
                "SGR": float(row["SGR"]),
                "SSR": float(row["SSR"]),
                "SWC": float(row["SWC"]),
                "SWIR": float(row["SWIR"]),
                "KWRO": float(row["KWRO"]),
                "KRSMAX": float(row["KRSMAX"]),
                "W": float(row["W"]),
            }
            output_file = f"{output_file_dir}/{output_prefix}{run_number}.SAV".upper()
            generate_input_file(base_file, output_file, params)

    # Copy CSV file to vDos Prophet directory if specified
    if vdos_csv_dir:
        try:
            os.makedirs(vdos_csv_dir, exist_ok=True)
            csv_filename = os.path.basename(csv_file)
            vdos_csv_path = os.path.join(vdos_csv_dir, csv_filename)
            shutil.copy2(csv_file, vdos_csv_path)
            print(f"\n[OK] Copied '{csv_filename}' to '{vdos_csv_dir}'")
        except Exception as e:
            print(f"\n[WARNING] Could not copy CSV to vDos directory: {e}")
