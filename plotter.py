"""
CO2 Prophet Results Plotter Module
Handles visualization of simulation results using Plotly.
"""

import os
import pandas as pd
import plotly.graph_objects as go


def plot_oil_vs_injected(csv_dir: str, output_plot: str = None, verbose: bool = True) -> None:
    """
    Plot Oil produced vs Injected total for all cases using Plotly.

    Args:
        csv_dir: Directory containing CSV files
        output_plot: Optional path to save the plot as HTML (if None, displays plot)
        verbose: If True, print status messages (default: True)
    """
    # Get all CSV files and sort them (only OUTPUT_*.csv files)
    csv_files = sorted(
        [
            f
            for f in os.listdir(csv_dir)
            if f.startswith("OUTPUT_") and f.endswith(".csv")
        ],
        key=lambda x: int(x.split("_")[1].split(".")[0]),  # Sort by run number
    )

    # Create Plotly figure
    fig = go.Figure()

    # Plot each case
    for csv_file in csv_files:
        file_path = os.path.join(csv_dir, csv_file)
        # Extract run number from filename (e.g., OUTPUT_1.csv -> 1)
        run_number = csv_file.replace("OUTPUT_", "").replace(".csv", "")

        # Read the CSV file
        df = pd.read_csv(file_path)

        # Add trace for this run
        fig.add_trace(go.Scatter(
            x=df["Injected total"],
            y=df["Oil produced"] * 100,
            mode='lines',
            name=f"Run {run_number}",
            line=dict(width=1.5),
            hovertemplate='<b>Run %{fullData.name}</b><br>' +
                         'Injected: %{x:.3f} HCPV<br>' +
                         'Oil Recovery: %{y:.2f}%<br>' +
                         '<extra></extra>'
        ))

    # Update layout
    fig.update_layout(
        title={
            'text': "Oil Produced vs Injected Total - All Cases",
            'font': {'size': 18, 'family': 'Arial, sans-serif'},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis=dict(
            title="Inj. CO2, HCPV",
            titlefont=dict(size=14),
            range=[0, None],  # Start from 0
            gridcolor='lightgray',
            gridwidth=0.5,
        ),
        yaxis=dict(
            title="Incremental Oil R.F, %OOIP",
            titlefont=dict(size=14),
            range=[0, None],  # Start from 0
            gridcolor='lightgray',
            gridwidth=0.5,
        ),
        hovermode='closest',
        showlegend=False,  # Hide legend
        plot_bgcolor='white',
        autosize=True,  # Auto-resize to container
        margin=dict(l=60, r=30, t=80, b=60)  # Optimized margins
    )

    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    if output_plot:
        # Save as HTML for interactive plot
        if not output_plot.endswith('.html'):
            output_plot = output_plot.replace('.png', '.html')
        fig.write_html(output_plot)
        if verbose:
            print(f"Interactive plot saved to {output_plot}")
    else:
        fig.show()
