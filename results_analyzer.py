"""
CO2 Prophet Results Analyzer Module
Extracts key metrics from simulation results.
"""

import os

import pandas as pd


def extract_key_metrics(csv_dir: str, output_file: str = None, verbose: bool = True) -> pd.DataFrame:
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
                "Oil_at_1HCPV": oil_at_1,
                "Oil_at_2HCPV": oil_at_2,
            }
        )

    # Create dataframe
    summary_df = pd.DataFrame(results)

    # Round to 2 decimal places
    summary_df["Oil_at_1HCPV"] = summary_df["Oil_at_1HCPV"].round(2)
    summary_df["Oil_at_2HCPV"] = summary_df["Oil_at_2HCPV"].round(2)

    # Save to file if specified
    if output_file:
        summary_df.to_csv(output_file, index=False)
        if verbose:
            print(f"✓ Saved summary metrics to {output_file}")

    return summary_df


def _find_value_at_injection(df: pd.DataFrame, target_injection: float) -> float:
    """
    Find oil produced value at a target injection point using interpolation.

    Args:
        df: DataFrame with 'Injected total' and 'Oil produced' columns
        target_injection: Target injection value (e.g., 1.0 or 2.0)

    Returns:
        Interpolated oil produced value
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


def print_summary_statistics(summary_df: pd.DataFrame):
    """
    Print summary statistics of the key metrics.

    Args:
        summary_df: DataFrame with extracted metrics
    """
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60 + "\n")

    print(f"Total runs analyzed: {len(summary_df)}\n")

    if "Oil_at_1HCPV" in summary_df.columns:
        print("Oil produced at ~1 HCPV (%OOIP):")
        print(f"  Mean:   {summary_df['Oil_at_1HCPV'].mean():.3f}")
        print(f"  Std:    {summary_df['Oil_at_1HCPV'].std():.3f}")
        print(f"  Min:    {summary_df['Oil_at_1HCPV'].min():.3f}")
        print(f"  Max:    {summary_df['Oil_at_1HCPV'].max():.3f}\n")

    if "Oil_at_2HCPV" in summary_df.columns:
        print("Oil produced at ~2 HCPV (%OOIP):")
        print(f"  Mean:   {summary_df['Oil_at_2HCPV'].mean():.3f}")
        print(f"  Std:    {summary_df['Oil_at_2HCPV'].std():.3f}")
        print(f"  Min:    {summary_df['Oil_at_2HCPV'].min():.3f}")
        print(f"  Max:    {summary_df['Oil_at_2HCPV'].max():.3f}\n")

    print("=" * 60 + "\n")
