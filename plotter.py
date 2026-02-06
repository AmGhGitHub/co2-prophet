"""
CO2 Prophet Results Plotter Module
Handles visualization of simulation results.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd


def plot_oil_vs_injected(csv_dir: str, output_plot: str = None) -> None:
    """
    Plot Oil produced vs Injected total for all cases.

    Args:
        csv_dir: Directory containing CSV files
        output_plot: Optional path to save the plot (if None, displays plot)
    """
    plt.figure(figsize=(10, 6))

    # Get all CSV files and sort them
    csv_files = sorted(
        [f for f in os.listdir(csv_dir) if f.endswith(".csv")],
        key=lambda x: int(x.split("_")[1].split(".")[0]),  # Sort by run number
    )

    # Plot each case
    for csv_file in csv_files:
        file_path = os.path.join(csv_dir, csv_file)
        # Extract run number from filename (e.g., OUTPUT_1.csv -> 1)
        run_number = csv_file.replace("OUTPUT_", "").replace(".csv", "")

        # Read the CSV file
        df = pd.read_csv(file_path)

        # Plot Oil produced vs Injected total
        plt.plot(
            df["Injected total"],
            df["Oil produced"] * 100,
            label=f"Run {run_number}",
            linewidth=1.5,
        )

    plt.xlabel("Inj. CO2, HCPV", fontsize=12)
    plt.ylabel("Incremental Oil R.F, %OOIP", fontsize=12)
    plt.title(
        "Oil Produced vs Injected Total - All Cases", fontsize=14, fontweight="bold"
    )

    # Set axis limits to start at origin (0,0) with no gaps
    plt.xlim(left=0)
    plt.ylim(bottom=0)

    # Place legend below plot with multiple columns
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=10,  # Number of columns in legend
        fontsize=8,
        frameon=True,
        fancybox=True,
        shadow=True,
    )

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_plot:
        plt.savefig(output_plot, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {output_plot}")
    else:
        plt.show()
