"""
CO2 Prophet Metrics Extractor Module
Extracts oil recovery at key injection points (1 HCPV and 2 HCPV) from simulation results.
"""

import os

import pandas as pd

OIL_RECOVERY_FACTORS = [
    "oil_recovery_at_1hcpv",
    "oil_recovery_at_2hcpv",
]


def get_oil_recovery_factors(
    csv_dir: str, params_csv: str = None, output_file: str = None, verbose: bool = True
) -> pd.DataFrame:
    """
    Extract RF, %OOIP at key injection points (≈1 and ≈2 HCPV) from labelout CSV files.

    Args:
        csv_dir: Directory containing labelout CSV files (from labelout_parser.py)
        params_csv: Path to parameters CSV file (not used, kept for compatibility)
        output_file: Optional path to save the summary dataframe as CSV
        verbose: If True, print status messages (default: True)

    Returns:
        DataFrame with columns: RUN, oil_recovery_at_1hcpv, oil_recovery_at_2hcpv
    """
    results = []

    # Get all labelout CSV files and sort by run number
    csv_files = sorted(
        [
            f
            for f in os.listdir(csv_dir)
            if (f.startswith("labelout_") or f.startswith("LABELOUT_"))
            and f.endswith(".csv")
        ],
        key=lambda x: int(x.split("_")[1].split(".")[0]),
    )

    if verbose:
        print(f"Found {len(csv_files)} labelout CSV files in {csv_dir}")

    for csv_file in csv_files:
        file_path = os.path.join(csv_dir, csv_file)
        # Extract run number from filename (e.g., labelout_1.csv -> 1 or LABELOUT_1.csv -> 1)
        run_number = int(csv_file.lower().replace("labelout_", "").replace(".csv", ""))

        # Read the CSV file
        df = pd.read_csv(file_path)

        # Find RF, %OOIP at Inj. CO2, hcpv ≈ 1
        oil_at_1 = _find_rf_ooip_at_injection(df, target_injection=1.0)

        # Find RF, %OOIP at Inj. CO2, hcpv ≈ 2
        oil_at_2 = _find_rf_ooip_at_injection(df, target_injection=2.0)

        results.append(
            {
                "RUN": run_number,
                OIL_RECOVERY_FACTORS[0]: oil_at_1,
                OIL_RECOVERY_FACTORS[1]: oil_at_2,
            }
        )

    # Create dataframe
    oil_recovery_df = pd.DataFrame(results)

    # Round to 2 decimal places
    oil_recovery_df[OIL_RECOVERY_FACTORS[0]] = oil_recovery_df[
        OIL_RECOVERY_FACTORS[0]
    ].round(2)
    oil_recovery_df[OIL_RECOVERY_FACTORS[1]] = oil_recovery_df[
        OIL_RECOVERY_FACTORS[1]
    ].round(2)

    # Save to file if specified
    if output_file:
        oil_recovery_df.to_csv(output_file, index=False)
        if verbose:
            print(f"Saved summary metrics to {output_file}")

    return oil_recovery_df


def _find_rf_ooip_at_injection(df: pd.DataFrame, target_injection: float) -> float:
    """
    Find RF, %OOIP value at a target injection point using interpolation.

    Args:
        df: DataFrame with 'Inj. CO2, hcpv' and 'RF, %OOIP' columns
        target_injection: Target injection value in HCPV (e.g., 1.0 or 2.0)

    Returns:
        Interpolated RF, %OOIP value or None if target is out of range
    """
    # Check if required columns exist
    if "Inj. CO2, hcpv" not in df.columns or "RF, %OOIP" not in df.columns:
        return None

    # Drop rows with NaN values in required columns
    df_clean = df[["Inj. CO2, hcpv", "RF, %OOIP"]].dropna()

    if df_clean.empty:
        return None

    # Check if target is within range
    if target_injection < df_clean["Inj. CO2, hcpv"].min():
        return None  # Target is before simulation start

    if target_injection > df_clean["Inj. CO2, hcpv"].max():
        return None  # Target is beyond simulation end

    # Find the two closest points
    idx_below = df_clean[df_clean["Inj. CO2, hcpv"] <= target_injection].index[-1]
    idx_above = df_clean[df_clean["Inj. CO2, hcpv"] >= target_injection].index[0]

    # If exact match
    if idx_below == idx_above:
        return df_clean.loc[idx_below, "RF, %OOIP"]

    # Linear interpolation
    x1 = df_clean.loc[idx_below, "Inj. CO2, hcpv"]
    y1 = df_clean.loc[idx_below, "RF, %OOIP"]
    x2 = df_clean.loc[idx_above, "Inj. CO2, hcpv"]
    y2 = df_clean.loc[idx_above, "RF, %OOIP"]

    # Interpolate
    rf_ooip = y1 + (target_injection - x1) * (y2 - y1) / (x2 - x1)

    return round(rf_ooip, 2)


def merge_results_with_input_parameters(
    metrics_df: pd.DataFrame,
    params_csv: str,
    output_file: str = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Merge extracted metrics with input parameters based on RUN number.
    Uses LEFT join to keep all parameter runs (null values for incomplete runs).

    Args:
        metrics_df: DataFrame with extracted metrics (RUN, Oil_at_1HCPV, Oil_at_2HCPV)
        params_csv: Path to parameters CSV file (e.g., sen_fbv.csv)
        output_file: Optional path to save merged dataframe as CSV
        verbose: If True, print status messages (default: True)

    Returns:
        DataFrame with merged parameters and results (all runs, with nulls for incomplete)
    """
    # Read parameters CSV
    params_df = pd.read_csv(params_csv)

    # LEFT join - keep all parameter runs, show null for incomplete
    merged_df = pd.merge(params_df, metrics_df, on="RUN", how="left")

    # Count complete and incomplete runs
    complete_runs = merged_df[OIL_RECOVERY_FACTORS[0]].notna().sum()
    incomplete_runs = merged_df[OIL_RECOVERY_FACTORS[0]].isna().sum()

    if verbose:
        print(f"\nMerged parameters with results:")
        print(f"  Total parameter runs: {len(params_df)}")
        print(f"  Runs with results: {complete_runs}")
        if incomplete_runs > 0:
            print(f"  Incomplete runs (null results): {incomplete_runs}")

    # Save to file if specified
    if output_file:
        merged_df.to_csv(output_file, index=False)
        if verbose:
            print(f"Saved merged data to {output_file}\n")

    return merged_df


if __name__ == "__main__":
    # Hard-coded paths
    csv_dir = r"C:\vDos\Prophet\sen-labelout-csv"
    results_dir = r"d:\temp\co2-prophet\results\csv-results"
    params_csv = r"d:\temp\co2-prophet\sen-runs\sen_fbv.csv"

    output_file = os.path.join(results_dir, "oil_recovery_factors.csv")
    merged_output = os.path.join(results_dir, "sen_fbv_with_results.csv")

    # Extract metrics
    print("Extracting key metrics from simulation results...")
    recovery_factors_df = get_oil_recovery_factors(csv_dir, params_csv, output_file)

    # Merge with parameters (LEFT join - keeps all runs, shows null for incomplete)
    print("Merging with input parameters...")
    merged_results_with_params = merge_results_with_input_parameters(
        recovery_factors_df, params_csv, merged_output
    )

    print(
        f"\nComplete! Merged data saved with all {len(merged_results_with_params)} parameter runs."
    )
