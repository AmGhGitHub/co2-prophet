"""
CO2 Prophet Labelout Parser Module
Extracts injection and production table data from labelout files.
"""

import re
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


def parse_labelout_file(filepath: str) -> Dict[str, pd.DataFrame]:
    """
    Parse a Prophet labelout file and extract all table data.

    Args:
        filepath: Path to the labelout file

    Returns:
        Dictionary containing:
        - 'reservoir_params': Single-row DataFrame with reservoir parameters
        - 'injection_cumulative': DataFrame with cumulative injection data
        - 'production_cumulative': DataFrame with cumulative production (HCPV)
        - 'production_cumulative_rates': DataFrame with cumulative production (surface rates)
        - 'production_incremental': DataFrame with incremental production (HCPV)
        - 'production_incremental_rates': DataFrame with incremental production (surface rates)
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    result = {}

    # Extract reservoir parameters
    result["reservoir_params"] = _extract_reservoir_params(lines)

    # Extract injection data
    result["injection_cumulative"] = _extract_injection_table(lines)

    # Extract production data
    production_data = _extract_production_tables(lines)
    result.update(production_data)

    return result


def _extract_reservoir_params(lines: list) -> pd.DataFrame:
    """Extract single-row reservoir parameters."""
    params = {}

    # Find reservoir data section
    for i, line in enumerate(lines):
        if "PRESSURE" in line and "MMP" in line:
            # Next line has values
            values = lines[i + 2].split()
            params["TEMP_F"] = float(values[0])
            params["PRESSURE_PSIA"] = float(values[1])
            params["MMP_PSIA"] = float(values[2])
            params["POROSITY"] = float(values[3])
            params["THICKNESS_FT"] = float(values[4])
            params["AREA_ACRES"] = float(values[5])
            break

    # Find initial saturations
    for i, line in enumerate(lines):
        if "SOINIT" in line and "SWINIT" in line and "SGINIT" in line:
            values = lines[i + 1].split()
            params["SOINIT"] = float(values[0])
            params["SWINIT"] = float(values[1])
            params["SGINIT"] = float(values[2])
            params["HCPV_MMRB"] = float(values[3])
            params["DYKSTRA_PARSONS"] = float(values[4])
            params["LAYERS"] = int(values[5])
            break

    # Find relative permeability parameters
    for i, line in enumerate(lines):
        if "SORW" in line and "SORG" in line and "SORM" in line and "WTR FLD" in line:
            values = lines[i + 1].split(",")
            params["SORW"] = float(values[0])
            params["SORG"] = float(values[1])
            params["SORM"] = float(values[2])
            break

    return pd.DataFrame([params])


def _extract_injection_table(lines: list) -> pd.DataFrame:
    """Extract cumulative injection data table."""
    data = []
    in_table = False

    for i, line in enumerate(lines):
        # Find injection table header
        if (
            "TIME" in line
            and "HCPV INPUT" in line
            and "WATER" in line
            and "SOLVENT" in line
        ):
            # Skip units line
            in_table = True
            continue

        if in_table:
            # Check if we've reached the end of table
            if line.strip() == "" or "***" in line:
                break

            # Parse data line
            values = line.split()
            if len(values) >= 5:
                try:
                    data.append(
                        {
                            "TIME_YRS": float(values[0]),
                            "TOTAL_HCPV": float(values[1]),
                            "WATER_HCPV": float(values[2]),
                            "SOLVENT_HCPV": float(values[3]),
                            "WATER_MSTB": float(values[4]),
                            "SOLVENT_MMSCF": float(values[5]),
                        }
                    )
                except (ValueError, IndexError):
                    continue

    return pd.DataFrame(data)


def _extract_production_tables(lines: list) -> Dict[str, pd.DataFrame]:
    """Extract both cumulative and incremental production tables."""
    result = {}

    # Extract cumulative HCPV production
    result["production_cumulative"] = _extract_production_cumulative_hcpv(lines)

    # Extract cumulative surface rates
    result["production_cumulative_rates"] = _extract_production_cumulative_rates(lines)

    # Extract incremental HCPV production
    result["production_incremental"] = _extract_production_incremental_hcpv(lines)

    # Extract incremental surface rates
    result["production_incremental_rates"] = _extract_production_incremental_rates(
        lines
    )

    return result


def _extract_production_cumulative_hcpv(lines: list) -> pd.DataFrame:
    """Extract cumulative production data (HCPV OUTPUT section)."""
    data = []
    in_table = False
    skip_lines = 0

    for i, line in enumerate(lines):
        # Find cumulative production table header
        if "HCPV OUTPUT" in line and "RECOVERY" in line:
            # Check if this is cumulative by looking back a few lines
            is_cumulative = False
            for j in range(max(0, i - 10), i):
                if "CUMULATIVE" in lines[j]:
                    is_cumulative = True
                    break

            # Also check it's not incremental
            is_incremental = False
            for j in range(max(0, i - 10), i):
                if "INCREMENTAL" in lines[j]:
                    is_incremental = True
                    break

            if is_cumulative and not is_incremental:
                skip_lines = 2
                in_table = True
                continue

        if in_table and skip_lines > 0:
            skip_lines -= 1
            continue

        if in_table and skip_lines == 0:
            # Check if we've reached the end of table (blank line or new section)
            if line.strip() == "" or "***" in line:
                break

            # Parse data line
            values = line.split()
            if len(values) >= 7:
                try:
                    data.append(
                        {
                            "TIME_YRS": float(values[0]),
                            "TOTAL_HCPV": float(values[1]),
                            "OIL_HCPV": float(values[2]),
                            "WATER_HCPV": float(values[3]),
                            "SOLVENT_HCPV": float(values[4]),
                            "OIL_RECOVERY_PCT_OOIP": float(values[5]),
                            "WATER_RECOVERY_PCT": float(values[6]),
                            "SOLVENT_RECOVERY_PCT": float(values[7]),
                        }
                    )
                except (ValueError, IndexError):
                    continue

    return pd.DataFrame(data)


def _extract_production_cumulative_rates(lines: list) -> pd.DataFrame:
    """Extract cumulative production surface rates (ER OIL section after HCPV)."""
    data = []
    in_table = False
    skip_lines = 0

    for i, line in enumerate(lines):
        # Find the surface rates table (appears after HCPV table)
        if "ER OIL" in line and "OIL" in line and "WATER" in line:
            # Check if this is cumulative by looking back and checking NOT incremental
            is_incremental = False
            for j in range(max(0, i - 5), i):
                if "INCREMENTAL" in lines[j]:
                    is_incremental = True
                    break

            is_cumulative = False
            for j in range(max(0, i - 10), i):
                if "CUMULATIVE" in lines[j]:
                    is_cumulative = True
                    break

            if is_cumulative and not is_incremental:
                skip_lines = 1  # Skip units line
                in_table = True
                continue

        if in_table and skip_lines > 0:
            skip_lines -= 1
            continue

        if in_table and skip_lines == 0:
            # Check if we've reached the end of table
            if line.strip() == "" or "***" in line:
                break

            # Parse data line
            values = line.split()
            if len(values) >= 7:
                try:
                    data.append(
                        {
                            "TIME_YRS": float(values[0]),
                            "OIL_RECOVERY_PCT_OOIP": float(values[1]),
                            "OIL_MSTB": float(values[2]),
                            "WATER_MSTB": float(values[3]),
                            "HC_GAS_MMSCF": float(values[4]),
                            "SOLVENT_MMSCF": float(values[5]),
                            "GOR_MSCF_PER_STB": float(values[6]),
                            "WOR_STB_PER_STB": float(values[7]),
                        }
                    )
                except (ValueError, IndexError):
                    continue

    return pd.DataFrame(data)


def _extract_production_incremental_hcpv(lines: list) -> pd.DataFrame:
    """Extract incremental production data (HCPV OUTPUT section)."""
    data = []
    in_table = False
    skip_lines = 0

    for i, line in enumerate(lines):
        # Find incremental production table header
        if "HCPV OUTPUT" in line and "OIL" in line:
            # Check if this is incremental
            is_incremental = False
            for j in range(max(0, i - 10), i):
                if "INCREMENTAL" in lines[j]:
                    is_incremental = True
                    break

            if is_incremental:
                skip_lines = 2  # Skip column headers and units
                in_table = True
                continue

        if in_table and skip_lines > 0:
            skip_lines -= 1
            continue

        if in_table and skip_lines == 0:
            # Check if we've reached the end of table
            if line.strip() == "" or "ER OIL" in line:
                break

            # Parse data line
            values = line.split()
            if len(values) >= 5:
                try:
                    data.append(
                        {
                            "TIME_YRS": float(values[0]),
                            "TOTAL_HCPV": float(values[1]),
                            "OIL_HCPV": float(values[2]),
                            "WATER_HCPV": float(values[3]),
                            "SOLVENT_HCPV": float(values[4]),
                            "OIL_RECOVERY_PCT_OOIP": float(values[5]),
                        }
                    )
                except (ValueError, IndexError):
                    continue

    return pd.DataFrame(data)


def _extract_production_incremental_rates(lines: list) -> pd.DataFrame:
    """Extract incremental production surface rates."""
    data = []
    in_table = False
    skip_lines = 0

    for i, line in enumerate(lines):
        # Find the incremental surface rates table
        if "ER OIL" in line and "OIL" in line and "WATER" in line:
            # Check if this is incremental
            is_incremental = False
            for j in range(max(0, i - 5), i):
                if "INCREMENTAL" in lines[j]:
                    is_incremental = True
                    break

            if is_incremental:
                skip_lines = 1  # Skip units line
                in_table = True
                continue

        if in_table and skip_lines > 0:
            skip_lines -= 1
            continue

        if in_table and skip_lines == 0:
            # Check if we've reached the end of table or file
            if line.strip() == "" or i >= len(lines) - 1:
                break

            # Parse data line
            values = line.split()
            if len(values) >= 7:
                try:
                    data.append(
                        {
                            "TIME_YRS": float(values[0]),
                            "OIL_RECOVERY_PCT_OOIP": float(values[1]),
                            "OIL_MSTB": float(values[2]),
                            "WATER_MSTB": float(values[3]),
                            "HC_GAS_MMSCF": float(values[4]),
                            "SOLVENT_MMSCF": float(values[5]),
                            "GOR_MSCF_PER_STB": float(values[6]),
                            "WOR_STB_PER_STB": float(values[7]),
                        }
                    )
                except (ValueError, IndexError):
                    continue

    return pd.DataFrame(data)


def create_summary_dataframe(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Create a summary DataFrame combining key data from injection and production tables.

    Merges:
    1. injection_cumulative: TIME_YRS, TOTAL_HCPV
    2. production_cumulative_rates: TIME_YRS, OIL_RECOVERY_PCT_OOIP (inner join)
    3. production_incremental_rates: All columns (left join to preserve all time points)

    Args:
        tables: Dictionary of DataFrames returned by parse_labelout_file()

    Returns:
        Summary DataFrame with merged injection and production data
    """
    # Extract injection data (TIME_YRS, TOTAL_HCPV)
    injection_df = tables["injection_cumulative"][["TIME_YRS", "TOTAL_HCPV"]].copy()

    # Extract cumulative production rates (TIME_YRS, OIL_RECOVERY_PCT_OOIP)
    cum_prod_df = tables["production_cumulative_rates"][
        ["TIME_YRS", "OIL_RECOVERY_PCT_OOIP"]
    ].copy()

    # Merge injection and cumulative production on TIME_YRS (inner join)
    summary_df = pd.merge(injection_df, cum_prod_df, on="TIME_YRS", how="inner")

    # Merge with incremental production rates on TIME_YRS (LEFT join to keep all rows)
    # Rename columns to avoid conflicts (both have OIL_RECOVERY_PCT_OOIP)
    incr_prod_df = tables["production_incremental_rates"].copy()

    # Rename incremental columns to distinguish them
    incr_prod_df = incr_prod_df.rename(
        columns={
            "OIL_MSTB": "INCR_OIL_MSTB",
            "WATER_MSTB": "INCR_WATER_MSTB",
            "HC_GAS_MMSCF": "INCR_HC_GAS_MMSCF",
            "SOLVENT_MMSCF": "INCR_SOLVENT_MMSCF",
        }
    )

    # Drop columns not needed in summary
    incr_prod_df = incr_prod_df.drop(
        columns=["OIL_RECOVERY_PCT_OOIP", "GOR_MSCF_PER_STB", "WOR_STB_PER_STB"]
    )

    # Final merge with LEFT join (keeps all rows from summary_df)
    summary_df = pd.merge(summary_df, incr_prod_df, on="TIME_YRS", how="left")

    # Convert MSTB to bbl/day: MSTB * 1000 / 30.5
    summary_df["INCR_OIL_MSTB"] = (summary_df["INCR_OIL_MSTB"] * 1000 / 30.5).round(3)
    summary_df["INCR_WATER_MSTB"] = (summary_df["INCR_WATER_MSTB"] * 1000 / 30.5).round(
        3
    )

    # Convert MMSCF to mmscf/day: MMSCF / 30.5
    summary_df["INCR_HC_GAS_MMSCF"] = (summary_df["INCR_HC_GAS_MMSCF"] / 30.5).round(3)
    summary_df["INCR_SOLVENT_MMSCF"] = (summary_df["INCR_SOLVENT_MMSCF"] / 30.5).round(
        3
    )

    # Rename columns to match desired format
    summary_df = summary_df.rename(
        columns={
            "TIME_YRS": "Time, year",
            "TOTAL_HCPV": "Inj. CO2, hcpv",
            "OIL_RECOVERY_PCT_OOIP": "RF,%COIP",
            "INCR_OIL_MSTB": "Prd Oil Rate, bbl/day",
            "INCR_WATER_MSTB": "Prd Water Rate, bbl/day",
            "INCR_HC_GAS_MMSCF": "Prd HC Gas Rate, mmscf/day",
            "INCR_SOLVENT_MMSCF": "Prd CO2 Rate, mmscf/day",
        }
    )

    # Add "Time, month" column with sequential values 1, 2, 3, ...
    summary_df.insert(1, "Time, month", range(1, len(summary_df) + 1))

    return summary_df


def save_all_tables_to_csv(
    labelout_file: str, output_dir: str = None, include_summary: bool = True
) -> None:
    """
    Parse labelout file and save all tables as CSV files.

    Args:
        labelout_file: Path to labelout file
        output_dir: Directory to save CSV files (default: same as labelout file)
        include_summary: If True, also create and save summary DataFrame (default: True)
    """
    # Parse the file
    tables = parse_labelout_file(labelout_file)

    # Determine output directory
    if output_dir is None:
        output_dir = Path(labelout_file).parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Get base filename
    base_name = Path(labelout_file).stem

    # Save each table
    for table_name, df in tables.items():
        if not df.empty:
            output_file = output_dir / f"{base_name}_{table_name}.csv"
            df.to_csv(output_file, index=False)
            print(f"Saved {table_name}: {output_file} ({len(df)} rows)")

    # Create and save summary DataFrame
    if include_summary:
        summary_df = create_summary_dataframe(tables)
        summary_file = output_dir / f"{base_name}_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        print(
            f"Saved summary: {summary_file} ({len(summary_df)} rows, {len(summary_df.columns)} columns)"
        )


if __name__ == "__main__":
    # Example usage
    labelout_file = r"d:\temp\co2-prophet\results\archive\labelout_1"

    print("Parsing labelout file...")
    tables = parse_labelout_file(labelout_file)

    print("\nExtracted tables:")
    for name, df in tables.items():
        print(f"\n{name}: {len(df)} rows, {len(df.columns)} columns")
        if len(df) > 0:
            print(f"Columns: {', '.join(df.columns)}")
            print(f"First row:\n{df.head(1)}")

    # Create summary DataFrame
    print("\n" + "=" * 60)
    print("Creating summary DataFrame...")
    summary_df = create_summary_dataframe(tables)
    print(
        f"\nlabelout_1_summary: {len(summary_df)} rows, {len(summary_df.columns)} columns"
    )
    print(f"Columns: {', '.join(summary_df.columns)}")
    print(f"\nFirst 5 rows:")
    print(summary_df.head())
    print(f"\nLast 5 rows:")
    print(summary_df.tail())

    # Save all tables to CSV (including summary)
    print("\n" + "=" * 60)
    print("Saving tables to CSV...")
    save_all_tables_to_csv(labelout_file)
    print("\nDone!")
