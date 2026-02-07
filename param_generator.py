"""
CO2 Prophet Parameter Generator Module
Generates sensitivity analysis parameters from statistical distributions.
"""

import csv
import os
import shutil

import numpy as np
from scipy.stats import qmc  # For Latin Hypercube Sampling


class ParameterGenerator:
    """Generate random parameters for CO2 Prophet sensitivity analysis."""

    def __init__(self, seed=None):
        """
        Initialize parameter generator.

        Args:
            seed: Random seed for reproducibility (optional)
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def generate_parameters(
        self,
        n_runs: int,
        dpcoef_range=(0.7, 0.95),
        poros_range=(0.08, 0.13),
        mmp_range=(1200, 1500),
        soinit_range=(0.4, 0.6),
        xkvh_range=(0.01, 0.1),
        distributions=None,
        use_lhs=True,
    ) -> list:
        """
        Generate random parameters for sensitivity analysis using Latin Hypercube Sampling.

        Args:
            n_runs: Number of simulation runs to generate
            dpcoef_range: (min, max) for depletion coefficient
            poros_range: (min, max) for porosity (fraction)
            mmp_range: (min, max) for minimum miscibility pressure (psi)
            soinit_range: (min, max) for initial oil saturation (fraction)
            xkvh_range: (min, max) for vertical to horizontal permeability ratio
            distributions: Dictionary specifying distribution types for each parameter
                          Options: 'uniform', 'normal', 'triangular'
                          Example: {'DPCOEF': 'uniform', 'POROS': 'normal'}
            use_lhs: If True, use Latin Hypercube Sampling for better space coverage

        Returns:
            List of dictionaries containing parameter sets
        """
        if distributions is None:
            distributions = {
                "DPCOEF": "uniform",
                "POROS": "normal",
                "MMP": "uniform",
                "SOINIT": "uniform",
                "XKVH": "uniform",
            }

        # Parameter names and ranges
        param_names = ["DPCOEF", "POROS", "MMP", "SOINIT", "XKVH"]
        param_ranges = [
            dpcoef_range,
            poros_range,
            mmp_range,
            soinit_range,
            xkvh_range,
        ]

        params_list = []

        if use_lhs:
            # Use Latin Hypercube Sampling for better parameter space coverage
            n_params = len(param_names)
            sampler = qmc.LatinHypercube(
                d=n_params, seed=self.seed if hasattr(self, "seed") else None
            )

            # Generate LHS samples in [0, 1] range
            lhs_samples = sampler.random(n=n_runs)

            # Transform samples to parameter ranges
            for run_idx in range(n_runs):
                params = {"RUN": run_idx + 1}

                for param_idx, param_name in enumerate(param_names):
                    uniform_sample = lhs_samples[run_idx, param_idx]
                    param_range = param_ranges[param_idx]
                    dist_type = distributions.get(param_name, "uniform")

                    # Transform uniform [0,1] sample to parameter value
                    value = self._transform_lhs_sample(
                        uniform_sample, param_name, param_range, dist_type
                    )
                    params[param_name] = value

                # Calculate SWINIT as complement of SOINIT
                params["SWINIT"] = 1.0 - params["SOINIT"]
                params_list.append(params)

        else:
            # Original random sampling (for comparison)
            for run in range(1, n_runs + 1):
                params = {"RUN": run}

                # Generate DPCOEF
                params["DPCOEF"] = self._generate_value(
                    "DPCOEF", dpcoef_range, distributions.get("DPCOEF", "uniform")
                )

                # Generate POROS
                params["POROS"] = self._generate_value(
                    "POROS", poros_range, distributions.get("POROS", "normal")
                )

                # Generate MMP
                params["MMP"] = self._generate_value(
                    "MMP", mmp_range, distributions.get("MMP", "uniform")
                )

                # Generate SOINIT
                params["SOINIT"] = self._generate_value(
                    "SOINIT", soinit_range, distributions.get("SOINIT", "uniform")
                )

                # Generate SWINIT (complementary to SOINIT)
                params["SWINIT"] = 1.0 - params["SOINIT"]

                # Generate XKVH
                params["XKVH"] = self._generate_value(
                    "XKVH", xkvh_range, distributions.get("XKVH", "uniform")
                )

                params_list.append(params)

        return params_list

    def _generate_value(
        self, param_name: str, value_range: tuple, distribution: str
    ) -> float:
        """
        Generate a single parameter value from specified distribution.

        Args:
            param_name: Name of the parameter
            value_range: (min, max) tuple
            distribution: Type of distribution

        Returns:
            Generated value
        """
        min_val, max_val = value_range

        if distribution == "uniform":
            return np.random.uniform(min_val, max_val)

        elif distribution == "normal":
            # Use mean and std based on range
            mean = (min_val + max_val) / 2
            std = (max_val - min_val) / 6  # 99.7% of values within range
            value = np.random.normal(mean, std)
            # Clip to ensure within range
            return np.clip(value, min_val, max_val)

        elif distribution == "lognormal":
            # Generate lognormal distribution within range
            log_min = np.log(max(min_val, 0.001))  # Avoid log(0)
            log_max = np.log(max_val)
            mean_log = (log_min + log_max) / 2
            std_log = (log_max - log_min) / 6
            value = np.random.lognormal(mean_log, std_log)
            return np.clip(value, min_val, max_val)

        elif distribution == "triangular":
            # Allow value_range to be either (min, max) or (min, mode, max)
            if len(value_range) == 3:
                min_val, mode, max_val = value_range
            else:
                mode = (min_val + max_val) / 2
            return np.random.triangular(min_val, mode, max_val)

        else:
            # Default to uniform
            return np.random.uniform(min_val, max_val)

    def _transform_lhs_sample(
        self,
        uniform_sample: float,
        param_name: str,
        value_range: tuple,
        distribution: str,
    ) -> float:
        """
        Transform a uniform [0,1] LHS sample to a parameter value using the specified distribution.

        Args:
            uniform_sample: LHS sample in [0, 1] range
            param_name: Name of the parameter
            value_range: (min, max) tuple for the parameter
            distribution: Type of distribution to apply

        Returns:
            Transformed parameter value
        """
        from scipy import stats

        min_val, max_val = value_range

        if distribution == "uniform":
            # Simple linear transformation
            return min_val + uniform_sample * (max_val - min_val)

        elif distribution == "normal":
            # Transform using normal distribution inverse CDF (percent point function)
            mean = (min_val + max_val) / 2
            std = (max_val - min_val) / 6  # 99.7% of values within range
            value = stats.norm.ppf(uniform_sample, loc=mean, scale=std)
            return np.clip(value, min_val, max_val)

        elif distribution == "lognormal":
            # Transform using lognormal distribution
            log_min = np.log(max(min_val, 0.001))
            log_max = np.log(max_val)
            mean_log = (log_min + log_max) / 2
            std_log = (log_max - log_min) / 6
            value = stats.lognorm.ppf(uniform_sample, s=std_log, scale=np.exp(mean_log))
            return np.clip(value, min_val, max_val)

        elif distribution == "triangular":
            # Transform using triangular distribution
            # Accept value_range as (min, max) or (min, mode, max)
            if len(value_range) == 3:
                min_val, mode, max_val = value_range
            else:
                mode = (min_val + max_val) / 2

            c = (mode - min_val) / (max_val - min_val)  # Mode parameter
            value = stats.triang.ppf(
                uniform_sample, c=c, loc=min_val, scale=max_val - min_val
            )
            return value

        else:
            # Default to uniform
            return min_val + uniform_sample * (max_val - min_val)

    def save_to_csv(self, params_list: list, output_file: str) -> None:
        """
        Save generated parameters to CSV file.

        Args:
            params_list: List of parameter dictionaries
            output_file: Path to output CSV file
        """
        fieldnames = [
            "RUN",
            "DPCOEF",
            "POROS",
            "MMP",
            "SOINIT",
            "SWINIT",
            "XKVH",
        ]

        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(params_list)

        print(f"Generated {len(params_list)} parameter sets -> {output_file}")

    def format_parameter(self, param_name: str, value: float) -> str:
        """
        Format parameter value for output.

        Args:
            param_name: Name of parameter
            value: Parameter value

        Returns:
            Formatted string
        """
        if param_name in ["DPCOEF", "POROS", "SOINIT", "SWINIT"]:
            return f"{value:.3f}"
        elif param_name == "XKVH":
            return f"{value:.2f}"
        elif param_name == "MMP":
            return f"{value:.0f}"
        else:
            return f"{value:.1f}"


def calculate_recommended_runs(n_params: int, sensitivity_level: str = "medium") -> int:
    """
    Calculate recommended number of runs for Latin Hypercube Sampling.

    Common guidelines:
    - Minimum: n_params + 1 (very sparse)
    - Low: 2 * n_params (basic coverage)
    - Medium: 10 * n_params (recommended for most cases)
    - High: 50 * n_params (detailed sensitivity)
    - Very High: 100 * n_params (comprehensive analysis)

    Args:
        n_params: Number of parameters to sample
        sensitivity_level: One of 'minimum', 'low', 'medium', 'high', 'very_high'

    Returns:
        Recommended number of runs
    """
    multipliers = {
        "minimum": 1,
        "low": 2,
        "medium": 10,
        "high": 50,
        "very_high": 100,
    }

    multiplier = multipliers.get(sensitivity_level.lower(), 10)
    recommended = max(n_params + 1, multiplier * n_params)

    return recommended


def generate_sensitivity_csv(
    output_file: str,
    n_runs: int = None,
    seed: int = None,
    custom_ranges: dict = None,
    custom_distributions: dict = None,
    backup_dir: str = None,
    use_lhs: bool = True,
    sensitivity_level: str = "medium",
) -> None:
    """
    Convenience function to generate sensitivity analysis CSV.

    Args:
        output_file: Path to output CSV file
        n_runs: Number of runs to generate (if None, auto-calculated based on parameters)
        seed: Random seed for reproducibility
        custom_ranges: Dictionary of custom ranges for parameters
        custom_distributions: Dictionary of distribution types for parameters
        backup_dir: Optional directory to save a backup copy of the CSV
        use_lhs: Use Latin Hypercube Sampling (default: True)
        sensitivity_level: 'minimum', 'low', 'medium', 'high', 'very_high' (used if n_runs is None)
    """
    generator = ParameterGenerator(seed=seed)

    # Default ranges
    ranges = {
        "dpcoef_range": (0.7, 1.0),
        "poros_range": (0.15, 0.35),
        "mmp_range": (1200, 1500),
        "soinit_range": (0.4, 0.6),
        "xkvh_range": (0.01, 0.1),
    }

    # Update with custom ranges if provided
    if custom_ranges:
        ranges.update(custom_ranges)

    # Auto-calculate number of runs if not specified
    if n_runs is None:
        n_params = 5  # DPCOEF, POROS, MMP, SOINIT, XKVH
        n_runs = calculate_recommended_runs(n_params, sensitivity_level)
        print(
            f"Auto-calculated {n_runs} runs for {n_params} parameters (sensitivity level: {sensitivity_level})"
        )

    # Generate parameters
    params_list = generator.generate_parameters(
        n_runs=n_runs, distributions=custom_distributions, use_lhs=use_lhs, **ranges
    )

    # Format values for better readability
    for params in params_list:
        params["DPCOEF"] = round(params["DPCOEF"], 2)
        params["POROS"] = round(params["POROS"], 3)
        params["MMP"] = round(params["MMP"], 0)
        params["SOINIT"] = round(params["SOINIT"], 3)
        params["SWINIT"] = round(params["SWINIT"], 3)
        params["XKVH"] = round(params["XKVH"], 2)

    # Save to CSV
    generator.save_to_csv(params_list, output_file)

    # Save backup copy if directory specified
    if backup_dir:
        os.makedirs(backup_dir, exist_ok=True)
        backup_file = os.path.join(backup_dir, os.path.basename(output_file))
        shutil.copy2(output_file, backup_file)
        print(f"Backup copy saved -> {backup_file}")


if __name__ == "__main__":
    # Example usage
    OUTPUT_CSV = "./sen-vars/sen_fbv.csv"
    BACKUP_DIR = "C:/vDos/Prophet/sen-input"

    # Generate 100 runs with default settings
    generate_sensitivity_csv(
        output_file=OUTPUT_CSV,
        n_runs=100,
        seed=42,  # For reproducibility
        backup_dir=BACKUP_DIR,
    )

    # Or with custom distributions
    # custom_dist = {
    #     'DPCOEF': 'uniform',
    #     'PERMAV': 'lognormal',
    #     'THICK': 'triangular',
    #     'POROS': 'normal',
    #     'NLAYERS': 'uniform'
    # }
    #
    # generate_sensitivity_csv(
    #     output_file=OUTPUT_CSV,
    #     n_runs=50,
    #     seed=42,
    #     custom_distributions=custom_dist
    # )
