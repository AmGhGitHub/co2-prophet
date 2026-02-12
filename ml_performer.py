"""
CO2 Prophet Machine Learning Analyzer Module
Analyzes correlations and builds predictive models for oil recovery.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

OIL_RECOVERY_FACTORS = [
    "oil_recovery_at_1hcpv",
    "oil_recovery_at_2hcpv",
]


def analyze_correlations(csv_file: str, output_dir: str = None, verbose: bool = True):
    """
    Analyze correlations between parameters and oil recovery results.

    Args:
        csv_file: Path to merged CSV file with parameters and results
        output_dir: Directory to save correlation plots and reports
        verbose: If True, print analysis results

    Returns:
        Dictionary with correlation analysis results
    """
    # Read data
    df = pd.read_csv(csv_file)

    # Remove rows with missing results
    df_complete = df.dropna(subset=["oil_recovery_at_1hcpv", "oil_recovery_at_2hcpv"])

    if verbose:
        print("\n" + "=" * 70)
        print("CORRELATION ANALYSIS")
        print("=" * 70)
        print(f"\nTotal runs: {len(df)}")
        print(f"Complete runs: {len(df_complete)}")
        print(f"Incomplete runs: {len(df) - len(df_complete)}\n")

    # Parameters to analyze (excluding SWINIT since SWINIT = 1.0 - SOINIT)
    params = ["DPCOEF", "POROS", "MMP", "SOINIT", "XKVH"]
    targets = ["oil_recovery_at_1hcpv", "oil_recovery_at_2hcpv"]

    # Calculate correlations
    correlations = {}
    for target in targets:
        corr = df_complete[params + [target]].corr()[target].drop(target)
        correlations[target] = corr.to_dict()

        if verbose:
            print(f"Correlations with {target}:")
            for param in params:
                corr_val = corr[param]
                strength = _get_correlation_strength(abs(corr_val))
                print(f"  {param:10} : {corr_val:7.4f}  ({strength})")
            print()

    # Create correlation heatmap if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        _plot_correlation_heatmap(df_complete, params, targets, output_dir)
        _plot_parameter_importance(correlations, output_dir)

    return correlations


def build_ml_models(csv_file: str, output_dir: str = None, verbose: bool = True):
    """
    Build machine learning models to predict oil recovery.

    Args:
        csv_file: Path to merged CSV file with parameters and results
        output_dir: Directory to save model results and plots
        verbose: If True, print model performance

    Returns:
        Dictionary with trained models and performance metrics
    """
    # Read data
    df = pd.read_csv(csv_file)
    df_complete = df.dropna(subset=["oil_recovery_at_1hcpv", "oil_recovery_at_2hcpv"])

    # Prepare features and targets (excluding SWINIT since SWINIT = 1.0 - SOINIT)
    params = ["DPCOEF", "POROS", "MMP", "SOINIT", "XKVH"]
    X = df_complete[params].values
    y_1hcpv = df_complete["oil_recovery_at_1hcpv"].values
    y_2hcpv = df_complete["oil_recovery_at_2hcpv"].values

    # Split data
    test_size = 0.2
    X_train, X_test, y1_train, y1_test = train_test_split(
        X, y_1hcpv, test_size=test_size, random_state=42
    )
    _, _, y2_train, y2_test = train_test_split(
        X, y_2hcpv, test_size=test_size, random_state=42
    )

    # Scale features using MinMaxScaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create polynomial features for Linear Regression (degree=2 with interactions)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)
    poly_feature_names = poly.get_feature_names_out(params)

    if verbose:
        print("\n" + "=" * 70)
        print("MACHINE LEARNING MODEL TRAINING")
        print("=" * 70)
        print(f"\nTraining set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Features: {', '.join(params)}")
        print(f"Polynomial features (degree=2): {len(poly_feature_names)} terms\n")

    # Train models
    results = {}
    results["poly_features"] = poly
    results["poly_feature_names"] = poly_feature_names

    for target_name, y_train, y_test in [
        (OIL_RECOVERY_FACTORS[0], y1_train, y1_test),
        (OIL_RECOVERY_FACTORS[1], y2_train, y2_test),
    ]:
        if verbose:
            print(f"\n{'=' * 70}")
            print(f"Target: {target_name}")
            print("=" * 70)

        results[target_name] = {}

        # Create polynomial regression model (degree=2)
        model = LinearRegression()
        model_name = "Polynomial Regression (degree=2)"

        # Use polynomial features
        X_train_model = X_train_poly
        X_test_model = X_test_poly

        # Train model
        model.fit(X_train_model, y_train)

        # Predictions
        y_pred_train = model.predict(X_train_model)
        y_pred_test = model.predict(X_test_model)

        # Metrics
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        mae_test = mean_absolute_error(y_test, y_pred_test)

        # Cross-validation (use min of 5 or number of training samples)
        n_splits = min(5, len(X_train_model))
        if n_splits < 2:
            # Not enough samples for cross-validation
            cv_scores_mean = r2_train
            cv_scores_std = 0.0
            if verbose:
                print(
                    f"  Warning: Too few samples ({len(X_train_model)}) for cross-validation, using R² (train) instead"
                )
        else:
            cv_scores = cross_val_score(
                model, X_train_model, y_train, cv=n_splits, scoring="r2", n_jobs=-1
            )
            cv_scores_mean = cv_scores.mean()
            cv_scores_std = cv_scores.std()

        results[target_name][model_name] = {
            "model": model,
            "r2_train": r2_train,
            "r2_test": r2_test,
            "rmse_test": rmse_test,
            "mae_test": mae_test,
            "cv_mean": cv_scores_mean,
            "cv_std": cv_scores_std,
            "predictions_test": y_pred_test,
            "actual_test": y_test,
        }

        if verbose:
            print(f"\n{model_name}:")
            print(f"  R² (train):     {r2_train:.4f}")
            print(f"  R² (test):      {r2_test:.4f}")
            print(f"  RMSE (test):    {rmse_test:.4f}")
            print(f"  MAE (test):     {mae_test:.4f}")
            print(f"  CV R² (mean):   {cv_scores_mean:.4f} ± {cv_scores_std:.4f}")

            print(f"\n  Polynomial Regression Equation (degree=2):")
            print(f"  {target_name} = {model.intercept_:.4f}")
            # Show top 10 most important terms by coefficient magnitude
            coef_importance = [
                (poly_feature_names[i], model.coef_[i]) for i in range(len(model.coef_))
            ]
            coef_importance.sort(key=lambda x: abs(x[1]), reverse=True)
            print(f"\n  Top 10 terms by importance:")
            for i, (feature, coef) in enumerate(coef_importance[:10]):
                sign = "+" if coef >= 0 else ""
                print(f"    {i+1}. {sign} {coef:.6f} * {feature}")
            print(
                f"\n  Total terms: {len(poly_feature_names)} (including all squared terms and interactions)"
            )
            print(f"  Note: Features scaled to [0, 1] range using MinMaxScaler")

    # Save results and plots
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        _save_model_plots(results, params, output_dir)
        _save_regression_equations(results, params, scaler, output_dir)

    # Save scaler for future predictions
    results["scaler"] = scaler
    results["feature_names"] = params

    if verbose:
        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70 + "\n")

    return results


def _get_correlation_strength(corr_value):
    """Classify correlation strength."""
    if corr_value >= 0.7:
        return "Very Strong"
    elif corr_value >= 0.5:
        return "Strong"
    elif corr_value >= 0.3:
        return "Moderate"
    elif corr_value >= 0.1:
        return "Weak"
    else:
        return "Very Weak"


def _plot_correlation_heatmap(df, params, targets, output_dir):
    """Create correlation heatmap."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Calculate correlation matrix
    corr_matrix = df[params + targets].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=1,
    )
    plt.title(
        "Correlation Heatmap: Parameters and Oil Recovery",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=300)
    plt.close()
    print(f"Saved correlation heatmap to {output_dir}/correlation_heatmap.png")


def _plot_parameter_importance(correlations, output_dir):
    """Plot parameter importance based on correlations."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Oil at 1 HCPV
    params_1 = list(correlations[OIL_RECOVERY_FACTORS[0]].keys())
    values_1 = list(correlations[OIL_RECOVERY_FACTORS[0]].values())
    colors_1 = ["green" if v > 0 else "red" for v in values_1]

    ax1.barh(params_1, values_1, color=colors_1, alpha=0.7)
    ax1.set_xlabel("Correlation Coefficient", fontsize=11)
    ax1.set_title("Correlation with Oil at 1 HCPV", fontsize=12, fontweight="bold")
    ax1.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
    ax1.grid(axis="x", alpha=0.3)

    # Oil at 2 HCPV
    params_2 = list(correlations[OIL_RECOVERY_FACTORS[1]].keys())
    values_2 = list(correlations[OIL_RECOVERY_FACTORS[1]].values())
    colors_2 = ["green" if v > 0 else "red" for v in values_2]

    ax2.barh(params_2, values_2, color=colors_2, alpha=0.7)
    ax2.set_xlabel("Correlation Coefficient", fontsize=11)
    ax2.set_title("Correlation with Oil at 2 HCPV", fontsize=12, fontweight="bold")
    ax2.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
    ax2.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "parameter_correlations.png"), dpi=300)
    plt.close()
    print(f"Saved parameter correlations to {output_dir}/parameter_correlations.png")


def _save_model_plots(results, params, output_dir):
    """Save prediction vs actual plot for polynomial regression."""
    for target in ["oil_recovery_at_1hcpv", "oil_recovery_at_2hcpv"]:
        # Get model data
        model_data = results[target]["Polynomial Regression (degree=2)"]
        y_test = model_data["actual_test"]
        y_pred = model_data["predictions_test"]
        r2 = model_data["r2_test"]

        # Create single plot
        plt.figure(figsize=(8, 6))

        # Scatter plot
        plt.scatter(y_test, y_pred, alpha=0.6, s=50, color="steelblue")

        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            lw=2,
            label="Perfect Prediction",
        )

        plt.xlabel("Actual (%OOIP)", fontsize=11)
        plt.ylabel("Predicted (%OOIP)", fontsize=11)
        plt.title(
            f"Polynomial Regression (degree=2)\n{target}\nR² = {r2:.4f}",
            fontsize=12,
            fontweight="bold",
        )
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        filename = f"model_predictions_{target}.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        plt.close()
        print(f"Saved model predictions to {output_dir}/{filename}")


def _save_regression_equations(results, params, scaler, output_dir):
    """Save polynomial regression equations to text file."""
    output_file = os.path.join(output_dir, "regression_equations.txt")

    poly_feature_names = results.get("poly_feature_names", [])

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("POLYNOMIAL REGRESSION EQUATIONS (DEGREE=2 WITH INTERACTIONS)\n")
        f.write("=" * 80 + "\n\n")
        f.write(
            "Note: These equations use MinMaxScaler (features scaled to [0, 1] range).\n"
        )
        f.write(f"Scaling parameters - Min and Max for each feature:\n\n")

        # Write scaling parameters
        for i, param in enumerate(params):
            f.write(
                f"  {param:10} : min = {scaler.data_min_[i]:10.4f}, max = {scaler.data_max_[i]:10.4f}\n"
            )

        f.write("\n" + "=" * 80 + "\n\n")

        # Write equations for each target
        for target in ["oil_recovery_at_1hcpv", "oil_recovery_at_2hcpv"]:
            poly_model_key = None
            for key in results[target].keys():
                if "Polynomial" in key:
                    poly_model_key = key
                    break

            if poly_model_key:
                model = results[target][poly_model_key]["model"]
                r2 = results[target][poly_model_key]["r2_test"]

                f.write(f"Target: {target}\n")
                f.write("-" * 80 + "\n\n")
                f.write(f"R² Score (test): {r2:.4f}\n\n")
                f.write("Polynomial Equation (degree=2 with interactions):\n")
                f.write(f"  {target} = {model.intercept_:.6f}\n")

                # Sort coefficients by magnitude to show most important terms first
                coef_importance = [
                    (poly_feature_names[i], model.coef_[i])
                    for i in range(len(model.coef_))
                ]
                coef_importance.sort(key=lambda x: abs(x[1]), reverse=True)

                f.write("\n  All terms (sorted by importance):\n")
                for i, (feature, coef) in enumerate(coef_importance):
                    sign = "+" if coef >= 0 else ""
                    f.write(f"    {sign} {coef:.8f} * {feature}\n")

                f.write(f"\n\n  Total terms: {len(poly_feature_names)}\n")
                f.write("  Including:\n")
                f.write("  - Linear terms (5): DPCOEF, POROS, MMP, SOINIT, XKVH\n")
                f.write(
                    "  - Squared terms (5): DPCOEF², POROS², MMP², SOINIT², XKVH²\n"
                )
                f.write("  - Interaction terms (10): All pairwise products\n")

                f.write("\n\nTo use this equation:\n")
                f.write("1. Scale input parameters using MinMaxScaler:\n")
                f.write(f"   parameter_scaled = (parameter - min) / (max - min)\n")
                f.write("   Result is in range [0, 1]\n")
                f.write(
                    "2. Create polynomial features (squared terms and interactions)\n"
                )
                f.write("3. Apply the equation above with all terms\n")
                f.write("4. Result is the predicted oil recovery in %OOIP\n")
                f.write("\n" + "=" * 80 + "\n\n")

        f.write("\nExample calculation:\n")
        f.write("-" * 80 + "\n")
        f.write(
            "If you have: DPCOEF=0.85, POROS=0.10, MMP=1350, SOINIT=0.50, XKVH=0.05\n\n"
        )
        f.write(
            "1. Scale each parameter to [0, 1] using: (value - min) / (max - min)\n"
        )
        f.write("2. Create polynomial features:\n")
        f.write("   - Squared: DPCOEF_scaled², POROS_scaled², etc.\n")
        f.write(
            "   - Interactions: DPCOEF_scaled * POROS_scaled, DPCOEF_scaled * MMP_scaled, etc.\n"
        )
        f.write("3. Plug all terms into the equation\n")
        f.write("4. The result is the predicted oil recovery\n")
        f.write(
            "\nNote: MinMaxScaler is easier to work with since all scaled values are in [0, 1] range.\n\n"
        )

    print(f"Saved regression equations to {output_dir}/regression_equations.txt")


if __name__ == "__main__":
    # Hard-coded paths
    csv_file = r"d:\temp\co2-prophet\results\csv-results\sen_fbv_with_results.csv"
    results_dir = r"d:\temp\co2-prophet\results\ml-results"

    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)

    print("Starting ML analysis...")
    print(f"Input file: {csv_file}")
    print(f"Output directory: {results_dir}\n")

    # Analyze correlations
    print("Step 1: Analyzing correlations...")
    correlations = analyze_correlations(csv_file, results_dir)

    # Build ML models
    print("\nStep 2: Building ML models...")
    models = build_ml_models(csv_file, results_dir)

    print(f"\nComplete! All results saved to {results_dir}")
