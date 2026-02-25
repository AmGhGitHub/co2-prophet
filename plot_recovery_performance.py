"""
CO2 Prophet Results Plotter Module
Handles visualization of simulation results using matplotlib.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd


def plot_oil_vs_injected(
    csv_dir: str, params_csv: str = None, output_plot: str = None, verbose: bool = True
) -> None:
    """
    Plot Oil produced vs Injected total for all cases using matplotlib.
    Scales results by (SOINIT / 0.85) to normalize to %OOIP basis.

    Args:
        csv_dir: Directory containing CSV files
        params_csv: Path to parameters CSV file (to read SOINIT for scaling)
        output_plot: Optional path to save the plot (if None, displays plot)
        verbose: If True, print status messages (default: True)
    """
    # Read SOINIT values from parameters CSV if provided
    soinit_dict = {}
    if params_csv:
        try:
            params_df = pd.read_csv(params_csv)
            soinit_dict = dict(zip(params_df["RUN"], params_df["SOINIT"]))
            if verbose:
                print(f"Loaded SOINIT values from {params_csv} for scaling")
        except Exception as e:
            if verbose:
                print(f"Warning: Could not read SOINIT from {params_csv}: {e}")
                print("Will use default scaling factor of 1.0")

    # Get all CSV files and sort them (only OUTPUT_*.csv files)
    csv_files = sorted(
        [
            f
            for f in os.listdir(csv_dir)
            if f.startswith("OUTPUT_") and f.endswith(".csv")
        ],
        key=lambda x: int(x.split("_")[1].split(".")[0]),  # Sort by run number
    )

    # Create figure
    plt.figure(figsize=(16, 10))

    # Plot each case
    for csv_file in csv_files:
        file_path = os.path.join(csv_dir, csv_file)
        # Extract run number from filename (e.g., OUTPUT_1.csv -> 1)
        run_number = int(csv_file.replace("OUTPUT_", "").replace(".csv", ""))

        # Read the CSV file
        df = pd.read_csv(file_path)

        # Get SOINIT for this run (default to 0.85 if not found)
        soinit = soinit_dict.get(run_number, 0.85)
        scaling_factor = soinit / 0.85

        # Plot Oil produced vs Injected total with SOINIT scaling
        plt.plot(
            df["Injected total"],
            df["Oil produced"] * 100 * scaling_factor,
            linewidth=1.5,
            alpha=0.8,
        )

    plt.xlabel("Inj. CO2, HCPV", fontsize=18, fontweight="bold")
    plt.ylabel("Incremental Oil R.F, %OOIP", fontsize=18, fontweight="bold")
    plt.title(
        "Oil Produced vs Injected Total - All Cases", fontsize=20, fontweight="bold"
    )

    # Increase tick label font sizes
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Set axis limits to start at origin (0,0) with no gaps
    plt.xlim(left=0)
    plt.ylim(bottom=0)

    plt.grid(True, alpha=0.3, linewidth=0.5)
    plt.tight_layout()

    if output_plot:
        # Ensure extension is .png
        if not output_plot.endswith(".png"):
            output_plot = output_plot.replace(".html", ".png")
        plt.savefig(output_plot, dpi=300, bbox_inches="tight")
        if verbose:
            print(f"Plot saved to {output_plot}")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    # Hard-coded paths
    csv_dir = r"C:\vDos\Prophet\sen-output-csv"
    results_dir = r"d:\temp\co2-prophet\results\img-results"
    params_csv = r"d:\temp\co2-prophet\sen-runs\sen_fbv.csv"

    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)

    # Output plot path
    output_plot = os.path.join(results_dir, "oil_vs_injected.png")

    # Generate plot
    print("Generating Oil vs Injected plot...")
    plot_oil_vs_injected(csv_dir, params_csv, output_plot)

    print(f"\nComplete! Plot saved to {output_plot}")
