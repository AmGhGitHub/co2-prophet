"""
Monte Carlo Simulation Module for CO2 Prophet
Generates probabilistic forecasts using the trained ML regression model.
"""

import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def triangular_sample(n_samples, min_val, max_val, mode):
    """Generate samples from triangular distribution."""
    return np.random.triangular(min_val, mode, max_val, n_samples)


def normal_sample(n_samples, mean, std, min_val=None, max_val=None):
    """Generate samples from normal distribution with optional truncation."""
    samples = np.random.normal(mean, std, n_samples)
    if min_val is not None and max_val is not None:
        samples = np.clip(samples, min_val, max_val)
    return samples


def uniform_sample(n_samples, min_val, max_val):
    """Generate samples from uniform distribution."""
    return np.random.uniform(min_val, max_val, n_samples)


def generate_monte_carlo_samples(n_samples=20000, seed=None):
    """
    Generate Monte Carlo samples using specified distributions.

    Based on FBV Defaults:
    - DP Coeff: Triangular(0.7, 0.95, mode=0.8)
    - Porosity: Triangular(0.09, 0.11, mode=0.1)
    - MMP: Triangular(1300, 1900, mode=1600) kPa
    - Sorw: Triangular(0.33, 0.46, mode=0.4)
    - Kv/Kh: Normal(mean=0.1, std=0.01, truncated to [0.01, 0.1])
    - CO2 Rate: Uniform(2, 10)
    """
    if seed is not None:
        np.random.seed(seed)

    dpcoef = triangular_sample(n_samples, 0.7, 0.95, 0.8)
    poros = triangular_sample(n_samples, 0.09, 0.11, 0.1)
    mmp = triangular_sample(n_samples, 1300, 1900, 1600)  # Already in kPa
    sorw = triangular_sample(n_samples, 0.33, 0.46, 0.4)
    xkvh = normal_sample(n_samples, 0.1, 0.01, 0.01, 0.1)
    solrat = uniform_sample(n_samples, 2.0, 10.0)

    df = pd.DataFrame(
        {
            "DPCOEF": dpcoef,
            "POROS": poros,
            "MMP": mmp,
            "SOINIT": sorw,  # Using SORW as SOINIT for model compatibility
            "XKVH": xkvh,
            "SOLRAT": solrat,
        }
    )

    return df


def load_ml_model(equations_file):
    """Load ML model parameters from regression_equations.txt."""
    with open(equations_file, "r", encoding="utf-8") as f:
        content = f.read()

    model_data = {"scaling_params": {}, "models": {}}

    # Extract scaling parameters
    scaling_section = re.search(
        r"Scaling parameters - Min and Max for each feature:\s*\n(.*?)\n\n",
        content,
        re.DOTALL,
    )
    if scaling_section:
        for line in scaling_section.group(1).strip().split("\n"):
            match = re.match(
                r"\s*(\w+)\s+:\s+min\s*=\s*([\d.]+),\s*max\s*=\s*([\d.]+)", line
            )
            if match:
                param_name = match.group(1)
                min_val = float(match.group(2))
                max_val = float(match.group(3))
                model_data["scaling_params"][param_name] = {
                    "min": min_val,
                    "max": max_val,
                }

    # Extract coefficients for each target
    target_pattern = r"Target: ([\w_]+)\s*\n-+\s*\n\s*\nR² Score.*?: ([\d.]+)\s*\n\s*\nPolynomial Equation.*?:\s*\n\s+[\w_]+ = ([\d.-]+)\s*\n\s*\n\s+All terms \(sorted by importance\):\s*\n(.*?)\n\n"

    for match in re.finditer(target_pattern, content, re.DOTALL):
        target_name = match.group(1)
        r2_score = float(match.group(2))
        intercept = float(match.group(3))
        terms_text = match.group(4)

        terms = []
        for term_line in terms_text.strip().split("\n"):
            term_match = re.match(
                r"\s*([+-])\s*([\d.]+)\s*\*\s*(.+)", term_line.strip()
            )
            if term_match:
                sign = term_match.group(1)
                coeff_abs = float(term_match.group(2))
                term_type = term_match.group(3).strip()

                coeff = coeff_abs if sign == "+" else -coeff_abs
                terms.append({"type": term_type, "coeff": coeff})

        model_data["models"][target_name] = {
            "r2_score": r2_score,
            "intercept": intercept,
            "terms": terms,
        }

    return model_data


def predict_with_model(df, model_data, param_names):
    """Predict oil recovery using polynomial regression model."""
    scaling_params = model_data["scaling_params"]

    # Scale features
    min_vals = np.array([scaling_params[p]["min"] for p in param_names])
    max_vals = np.array([scaling_params[p]["max"] for p in param_names])

    X = df[param_names].values
    X_scaled = (X - min_vals) / (max_vals - min_vals)

    # Create dictionary for easy access to scaled values
    scaled_dict = {param_names[i]: X_scaled[:, i] for i in range(len(param_names))}

    predictions = {}

    for target_name, model in model_data["models"].items():
        # Start with intercept
        y_pred = np.full(len(df), model["intercept"])

        # Add each term
        for term in model["terms"]:
            term_type = term["type"]
            coeff = term["coeff"]

            if "^2" in term_type:
                # Squared term: "DPCOEF^2"
                param = term_type.replace("^2", "")
                if param in scaled_dict:
                    y_pred += coeff * (scaled_dict[param] ** 2)
            elif " " in term_type:
                # Interaction term: "DPCOEF SOINIT"
                params = term_type.split(" ")
                if (
                    len(params) == 2
                    and params[0] in scaled_dict
                    and params[1] in scaled_dict
                ):
                    y_pred += coeff * scaled_dict[params[0]] * scaled_dict[params[1]]
            else:
                # Linear term: "DPCOEF"
                if term_type in scaled_dict:
                    y_pred += coeff * scaled_dict[term_type]

        predictions[target_name] = y_pred

    return predictions


def plot_monte_carlo_results(predictions_1hcpv, predictions_2hcpv, output_dir):
    """Create histogram plots showing P10, P50, P90 percentiles."""
    os.makedirs(output_dir, exist_ok=True)

    # Calculate percentiles
    p10_1 = np.percentile(predictions_1hcpv, 10)
    p50_1 = np.percentile(predictions_1hcpv, 50)
    p90_1 = np.percentile(predictions_1hcpv, 90)

    p10_2 = np.percentile(predictions_2hcpv, 10)
    p50_2 = np.percentile(predictions_2hcpv, 50)
    p90_2 = np.percentile(predictions_2hcpv, 90)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1.0 HCPV
    ax1.hist(
        predictions_1hcpv, bins=50, color="#6B7280", edgecolor="white", linewidth=0.5
    )
    ax1.axvline(
        p10_1, color="#EF4444", linestyle="--", linewidth=2, label=f"P10: {p10_1:.2f}"
    )
    ax1.axvline(
        p50_1, color="#10B981", linestyle="--", linewidth=2, label=f"P50: {p50_1:.2f}"
    )
    ax1.axvline(
        p90_1, color="#3B82F6", linestyle="--", linewidth=2, label=f"P90: {p90_1:.2f}"
    )

    ax1.set_title("Distribution @ 1.0 HCPV", fontsize=18, fontweight="bold", pad=20)
    ax1.set_xlabel("RF, %OOIP", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Frequency", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=12, frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.tick_params(labelsize=11)

    # Plot 2.0 HCPV
    ax2.hist(
        predictions_2hcpv, bins=50, color="#6B7280", edgecolor="white", linewidth=0.5
    )
    ax2.axvline(
        p10_2, color="#EF4444", linestyle="--", linewidth=2, label=f"P10: {p10_2:.2f}"
    )
    ax2.axvline(
        p50_2, color="#10B981", linestyle="--", linewidth=2, label=f"P50: {p50_2:.2f}"
    )
    ax2.axvline(
        p90_2, color="#3B82F6", linestyle="--", linewidth=2, label=f"P90: {p90_2:.2f}"
    )

    ax2.set_title("Distribution @ 2.0 HCPV", fontsize=18, fontweight="bold", pad=20)
    ax2.set_xlabel("RF, %OOIP", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Frequency", fontsize=14, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=12, frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.tick_params(labelsize=11)

    plt.tight_layout()

    output_file = os.path.join(output_dir, "monte_carlo_distributions.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved Monte Carlo distribution plot: {output_file}")
    plt.close()

    # Print statistics
    print("\n" + "=" * 70)
    print("MONTE CARLO SIMULATION RESULTS")
    print("=" * 70)
    print(f"\nOil Recovery @ 1.0 HCPV:")
    print(f"  P10: {p10_1:.2f} %OOIP")
    print(f"  P50: {p50_1:.2f} %OOIP")
    print(f"  P90: {p90_1:.2f} %OOIP")
    print(f"\nOil Recovery @ 2.0 HCPV:")
    print(f"  P10: {p10_2:.2f} %OOIP")
    print(f"  P50: {p50_2:.2f} %OOIP")
    print(f"  P90: {p90_2:.2f} %OOIP")
    print("=" * 70 + "\n")


def run_monte_carlo(
    n_samples=20000,
    equations_file="results/ml-results/regression_equations.txt",
    output_dir="results/monte-carlo",
    seed=None,
):
    """Run Monte Carlo simulation and generate distribution plots."""
    print(f"\nRunning Monte Carlo simulation with {n_samples:,} samples...")

    # Generate samples
    print("Generating parameter samples...")
    df_samples = generate_monte_carlo_samples(n_samples, seed)

    # Load ML model
    print("Loading ML regression model...")
    model_data = load_ml_model(equations_file)

    # Parameter order must match model training
    param_names = ["DPCOEF", "POROS", "MMP", "SOINIT", "XKVH", "SOLRAT"]

    # Make predictions
    print("Running predictions...")
    predictions = predict_with_model(df_samples, model_data, param_names)

    # Extract predictions for each target
    pred_1hcpv = predictions.get("oil_recovery_at_1hcpv", None)
    pred_2hcpv = predictions.get("oil_recovery_at_2hcpv", None)

    if pred_1hcpv is None or pred_2hcpv is None:
        print(
            "ERROR: Could not find predictions for oil_recovery_at_1hcpv and oil_recovery_at_2hcpv"
        )
        return

    # Save samples and predictions to CSV with 3 decimal precision
    df_results = df_samples.copy()
    df_results["RF_1HCPV"] = pred_1hcpv
    df_results["RF_2HCPV"] = pred_2hcpv

    # Round all numeric columns to 3 decimals
    df_results = df_results.round(3)

    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, "monte_carlo_results.csv")
    df_results.to_csv(results_file, index=False)
    print(f"Saved Monte Carlo results: {results_file}")

    # Plot results
    print("Generating distribution plots...")
    plot_monte_carlo_results(pred_1hcpv, pred_2hcpv, output_dir)

    print("Monte Carlo simulation complete!")


if __name__ == "__main__":
    # Run Monte Carlo simulation with 20,000 samples
    script_dir = Path(__file__).parent.resolve()
    equations_file = script_dir / "results" / "ml-results" / "regression_equations.txt"
    output_dir = script_dir / "results" / "monte-carlo"

    run_monte_carlo(
        n_samples=20000,
        equations_file=str(equations_file),
        output_dir=str(output_dir),
        seed=42,  # For reproducibility
    )
