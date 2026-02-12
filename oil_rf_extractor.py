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
    csv_dir: str, output_file: str = None, verbose: bool = True
) -> pd.DataFrame:
    """
    Extract oil produced at key injection points (≈1 and ≈2 HCPV) from all runs.

    Args:
        csv_dir: Directory containing CSV files with simulation results
        output_file: Optional path to save the summary dataframe as CSV
        verbose: If True, print status messages (default: True)

    Returns:
        DataFrame with columns: RUN, Oil_at_1HCPV, Oil_at_2HCPV
    """
    results = []

    # Get all CSV files and sort by run number
    csv_files = sorted(
        [
            f
            for f in os.listdir(csv_dir)
            if f.startswith("OUTPUT_") and f.endswith(".csv")
        ],
        key=lambda x: int(x.split("_")[1].split(".")[0]),
    )

    for csv_file in csv_files:
        file_path = os.path.join(csv_dir, csv_file)
        # Extract run number from filename (e.g., OUTPUT_1.csv -> 1)
        run_number = int(csv_file.replace("OUTPUT_", "").replace(".csv", ""))

        # Read the CSV file
        df = pd.read_csv(file_path)

        # Find oil produced at Injected total ≈ 1
        oil_at_1 = _find_value_at_injection(df, target_injection=1.0)

        # Find oil produced at Injected total ≈ 2
        oil_at_2 = _find_value_at_injection(df, target_injection=2.0)

        # Convert to percentage (%OOIP)
        if oil_at_1 is not None:
            oil_at_1 = oil_at_1 * 100
        if oil_at_2 is not None:
            oil_at_2 = oil_at_2 * 100

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


def _find_value_at_injection(df: pd.DataFrame, target_injection: float) -> float:
    """
    Find oil produced value at a target injection point using interpolation.

    Args:
        df: DataFrame with 'Injected total' and 'Oil produced' columns
        target_injection: Target injection value (e.g., 1.0 or 2.0)

    Returns:
        Interpolated oil produced value or None if target is out of range
    """
    # Check if target is within range
    if target_injection < df["Injected total"].min():
        return None  # Target is before simulation start

    if target_injection > df["Injected total"].max():
        return None  # Target is beyond simulation end

    # Find the two closest points
    idx_below = df[df["Injected total"] <= target_injection].index[-1]
    idx_above = df[df["Injected total"] >= target_injection].index[0]

    # If exact match
    if idx_below == idx_above:
        return df.loc[idx_below, "Oil produced"]

    # Linear interpolation
    x1 = df.loc[idx_below, "Injected total"]
    y1 = df.loc[idx_below, "Oil produced"]
    x2 = df.loc[idx_above, "Injected total"]
    y2 = df.loc[idx_above, "Oil produced"]

    # Interpolate
    oil_value = y1 + (target_injection - x1) * (y2 - y1) / (x2 - x1)

    return oil_value


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
    csv_dir = r"C:\vDos\Prophet\sen-output-csv"
    results_dir = r"d:\temp\co2-prophet\results\csv-results"
    params_csv = r"d:\temp\co2-prophet\sen-runs\sen_fbv.csv"

    output_file = os.path.join(results_dir, "oil_recovery_factors.csv")
    merged_output = os.path.join(results_dir, "sen_fbv_with_results.csv")

    # Extract metrics
    print("Extracting key metrics from simulation results...")
    recovery_factors_df = get_oil_recovery_factors(csv_dir, output_file)

    # Merge with parameters (LEFT join - keeps all runs, shows null for incomplete)
    print("Merging with input parameters...")
    merged_results_with_params = merge_results_with_input_parameters(
        recovery_factors_df, params_csv, merged_output
    )

    print(
        f"\nComplete! Merged data saved with all {len(merged_results_with_params)} parameter runs."
    )
