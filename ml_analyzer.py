"""
CO2 Prophet Machine Learning Analyzer Module
Analyzes correlations and builds predictive models for oil recovery.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler


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
    df_complete = df.dropna(subset=["Oil_at_1HCPV", "Oil_at_2HCPV"])

    if verbose:
        print("\n" + "=" * 70)
        print("CORRELATION ANALYSIS")
        print("=" * 70)
        print(f"\nTotal runs: {len(df)}")
        print(f"Complete runs: {len(df_complete)}")
        print(f"Incomplete runs: {len(df) - len(df_complete)}\n")

    # Parameters to analyze (excluding SWINIT since SWINIT = 1.0 - SOINIT)
    params = ["DPCOEF", "POROS", "MMP", "SOINIT", "XKVH"]
    targets = ["Oil_at_1HCPV", "Oil_at_2HCPV"]

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
    df_complete = df.dropna(subset=["Oil_at_1HCPV", "Oil_at_2HCPV"])

    # Prepare features and targets (excluding SWINIT since SWINIT = 1.0 - SOINIT)
    params = ["DPCOEF", "POROS", "MMP", "SOINIT", "XKVH"]
    X = df_complete[params].values
    y_1hcpv = df_complete["Oil_at_1HCPV"].values
    y_2hcpv = df_complete["Oil_at_2HCPV"].values

    # Split data
    test_size = 0.2
    X_train, X_test, y1_train, y1_test = train_test_split(
        X, y_1hcpv, test_size=test_size, random_state=42
    )
    _, _, y2_train, y2_test = train_test_split(
        X, y_2hcpv, test_size=test_size, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if verbose:
        print("\n" + "=" * 70)
        print("MACHINE LEARNING MODEL TRAINING")
        print("=" * 70)
        print(f"\nTraining set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Features: {', '.join(params)}\n")

    # Train models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=42, max_depth=10
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, random_state=42, max_depth=5
        ),
    }

    results = {}

    for target_name, y_train, y_test in [
        ("Oil_at_1HCPV", y1_train, y1_test),
        ("Oil_at_2HCPV", y2_train, y2_test),
    ]:
        if verbose:
            print(f"\n{'=' * 70}")
            print(f"Target: {target_name}")
            print("=" * 70)

        results[target_name] = {}

        for model_name, model in models.items():
            # Train model
            model.fit(X_train_scaled, y_train)

            # Predictions
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)

            # Metrics
            r2_train = r2_score(y_train, y_pred_train)
            r2_test = r2_score(y_test, y_pred_test)
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
            mae_test = mean_absolute_error(y_test, y_pred_test)

            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train_scaled, y_train, cv=5, scoring="r2", n_jobs=-1
            )

            results[target_name][model_name] = {
                "model": model,
                "r2_train": r2_train,
                "r2_test": r2_test,
                "rmse_test": rmse_test,
                "mae_test": mae_test,
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "predictions_test": y_pred_test,
                "actual_test": y_test,
            }

            if verbose:
                print(f"\n{model_name}:")
                print(f"  R² (train):     {r2_train:.4f}")
                print(f"  R² (test):      {r2_test:.4f}")
                print(f"  RMSE (test):    {rmse_test:.4f}")
                print(f"  MAE (test):     {mae_test:.4f}")
                print(
                    f"  CV R² (mean):   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
                )

                # Display Linear Regression equation
                if model_name == "Linear Regression":
                    print(f"\n  Linear Regression Equation:")
                    print(f"  {target_name} = {model.intercept_:.4f}")
                    for i, param in enumerate(params):
                        coef = model.coef_[i]
                        sign = "+" if coef >= 0 else ""
                        print(f"              {sign} {coef:.4f} * {param}_scaled")
                    print(f"\n  Note: Features are standardized (scaled)")

    # Save results and plots
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        _save_model_plots(results, params, output_dir)
        _save_feature_importance(results, params, output_dir)
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
    print(f"✓ Saved correlation heatmap to {output_dir}/correlation_heatmap.png")


def _plot_parameter_importance(correlations, output_dir):
    """Plot parameter importance based on correlations."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Oil at 1 HCPV
    params_1 = list(correlations["Oil_at_1HCPV"].keys())
    values_1 = list(correlations["Oil_at_1HCPV"].values())
    colors_1 = ["green" if v > 0 else "red" for v in values_1]

    ax1.barh(params_1, values_1, color=colors_1, alpha=0.7)
    ax1.set_xlabel("Correlation Coefficient", fontsize=11)
    ax1.set_title("Correlation with Oil at 1 HCPV", fontsize=12, fontweight="bold")
    ax1.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
    ax1.grid(axis="x", alpha=0.3)

    # Oil at 2 HCPV
    params_2 = list(correlations["Oil_at_2HCPV"].keys())
    values_2 = list(correlations["Oil_at_2HCPV"].values())
    colors_2 = ["green" if v > 0 else "red" for v in values_2]

    ax2.barh(params_2, values_2, color=colors_2, alpha=0.7)
    ax2.set_xlabel("Correlation Coefficient", fontsize=11)
    ax2.set_title("Correlation with Oil at 2 HCPV", fontsize=12, fontweight="bold")
    ax2.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
    ax2.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "parameter_correlations.png"), dpi=300)
    plt.close()
    print(f"✓ Saved parameter correlations to {output_dir}/parameter_correlations.png")


def _save_model_plots(results, params, output_dir):
    """Save prediction vs actual plots for all models."""
    for target in ["Oil_at_1HCPV", "Oil_at_2HCPV"]:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for idx, (model_name, metrics) in enumerate(results[target].items()):
            ax = axes[idx]

            y_test = metrics["actual_test"]
            y_pred = metrics["predictions_test"]
            r2 = metrics["r2_test"]

            # Scatter plot
            ax.scatter(y_test, y_pred, alpha=0.6, s=50)

            # Perfect prediction line
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            ax.plot(
                [min_val, max_val],
                [min_val, max_val],
                "r--",
                lw=2,
                label="Perfect Prediction",
            )

            ax.set_xlabel("Actual", fontsize=10)
            ax.set_ylabel("Predicted", fontsize=10)
            ax.set_title(f"{model_name}\nR² = {r2:.4f}", fontsize=11, fontweight="bold")
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)

        plt.suptitle(f"Model Performance: {target}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        filename = f"model_predictions_{target}.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        plt.close()
        print(f"✓ Saved model predictions to {output_dir}/{filename}")


def _save_feature_importance(results, params, output_dir):
    """Save feature importance plots for tree-based models."""
    for target in ["Oil_at_1HCPV", "Oil_at_2HCPV"]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        tree_models = ["Random Forest", "Gradient Boosting"]

        for idx, model_name in enumerate(tree_models):
            if model_name in results[target]:
                model = results[target][model_name]["model"]
                importances = model.feature_importances_

                ax = axes[idx]
                ax.barh(params, importances, color="steelblue", alpha=0.7)
                ax.set_xlabel("Feature Importance", fontsize=10)
                ax.set_title(f"{model_name}", fontsize=11, fontweight="bold")
                ax.grid(axis="x", alpha=0.3)

        plt.suptitle(f"Feature Importance: {target}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        filename = f"feature_importance_{target}.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        plt.close()
        print(f"✓ Saved feature importance to {output_dir}/{filename}")


def _save_regression_equations(results, params, scaler, output_dir):
    """Save linear regression equations to text file."""
    output_file = os.path.join(output_dir, "regression_equations.txt")

    with open(output_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("LINEAR REGRESSION EQUATIONS\n")
        f.write("=" * 80 + "\n\n")
        f.write("Note: These equations use STANDARDIZED (scaled) features.\n")
        f.write(f"Scaling parameters - Mean and Std Dev for each feature:\n\n")

        # Write scaling parameters
        for i, param in enumerate(params):
            f.write(
                f"  {param:10} : mean = {scaler.mean_[i]:8.4f}, std = {scaler.scale_[i]:8.4f}\n"
            )

        f.write("\n" + "=" * 80 + "\n\n")

        # Write equations for each target
        for target in ["Oil_at_1HCPV", "Oil_at_2HCPV"]:
            if "Linear Regression" in results[target]:
                model = results[target]["Linear Regression"]["model"]
                r2 = results[target]["Linear Regression"]["r2_test"]

                f.write(f"Target: {target}\n")
                f.write("-" * 80 + "\n\n")
                f.write(f"R² Score (test): {r2:.4f}\n\n")
                f.write("Equation (scaled features):\n")
                f.write(f"  {target} = {model.intercept_:.6f}\n")

                for i, param in enumerate(params):
                    coef = model.coef_[i]
                    sign = "+" if coef >= 0 else ""
                    f.write(f"             {sign} {coef:.6f} * {param}_scaled\n")

                f.write("\n\nTo use this equation:\n")
                f.write("1. Standardize input parameters:\n")
                f.write(f"   {param}_scaled = ({param} - mean) / std\n")
                f.write("2. Apply the equation above with scaled values\n")
                f.write("3. Result is the predicted oil recovery in %OOIP\n")
                f.write("\n" + "=" * 80 + "\n\n")

        f.write("\nExample calculation:\n")
        f.write("-" * 80 + "\n")
        f.write(
            "If you have: DPCOEF=0.85, POROS=0.10, MMP=1350, SOINIT=0.50, SWINIT=0.50, XKVH=0.05\n\n"
        )
        f.write("1. Standardize each parameter using the mean and std above\n")
        f.write("2. Plug scaled values into the equation\n")
        f.write("3. The result is the predicted oil recovery\n\n")

    print(f"✓ Saved regression equations to {output_dir}/regression_equations.txt")


if __name__ == "__main__":
    # Example usage
    csv_file = "results/sen_fbv_with_results.csv"
    output_dir = "results"

    # Analyze correlations
    correlations = analyze_correlations(csv_file, output_dir)

    # Build ML models
    models = build_ml_models(csv_file, output_dir)
