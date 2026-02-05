"""
CO2 Prophet Parameter Generator Module
Generates sensitivity analysis parameters from statistical distributions.
"""

import csv
import os
import shutil

import numpy as np


class ParameterGenerator:
    """Generate random parameters for CO2 Prophet sensitivity analysis."""

    def __init__(self, seed=None):
        """
        Initialize parameter generator.

        Args:
            seed: Random seed for reproducibility (optional)
        """
        if seed is not None:
            np.random.seed(seed)

    def generate_parameters(
        self,
        n_runs: int,
        dpcoef_range=(0.7, 0.95),
        permav_range=(100, 1000),
        thick_range=(14, 16),
        poros_range=(0.08, 0.13),
        nlayers_range=(3, 9),
        distributions=None,
    ) -> list:
        """
        Generate random parameters for sensitivity analysis.

        Args:
            n_runs: Number of simulation runs to generate
            dpcoef_range: (min, max) for depletion coefficient
            permav_range: (min, max) for average permeability (mD)
            thick_range: (min, max) for thickness (ft)
            poros_range: (min, max) for porosity (fraction)
            nlayers_range: (min, max) for number of layers (integer)
            distributions: Dictionary specifying distribution types for each parameter
                          Options: 'uniform', 'normal', 'lognormal', 'triangular'
                          Example: {'DPCOEF': 'uniform', 'PERMAV': 'lognormal'}

        Returns:
            List of dictionaries containing parameter sets
        """
        if distributions is None:
            distributions = {
                "DPCOEF": "uniform",
                "PERMAV": "lognormal",
                "THICK": "uniform",
                "POROS": "normal",
                "NLAYERS": "uniform",
            }

        params_list = []

        for run in range(1, n_runs + 1):
            params = {"RUN": run}

            # Generate DPCOEF
            params["DPCOEF"] = self._generate_value(
                "DPCOEF", dpcoef_range, distributions.get("DPCOEF", "uniform")
            )

            # Generate PERMAV
            params["PERMAV"] = self._generate_value(
                "PERMAV", permav_range, distributions.get("PERMAV", "lognormal")
            )

            # Generate THICK
            params["THICK"] = self._generate_value(
                "THICK", thick_range, distributions.get("THICK", "uniform")
            )

            # Generate POROS
            params["POROS"] = self._generate_value(
                "POROS", poros_range, distributions.get("POROS", "normal")
            )

            # Generate NLAYERS (integer)
            params["NLAYERS"] = int(
                self._generate_value(
                    "NLAYERS", nlayers_range, distributions.get("NLAYERS", "uniform")
                )
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
            # Triangular with mode at center
            mode = (min_val + max_val) / 2
            return np.random.triangular(min_val, mode, max_val)

        else:
            # Default to uniform
            return np.random.uniform(min_val, max_val)

    def save_to_csv(self, params_list: list, output_file: str) -> None:
        """
        Save generated parameters to CSV file.

        Args:
            params_list: List of parameter dictionaries
            output_file: Path to output CSV file
        """
        fieldnames = ["RUN", "DPCOEF", "PERMAV", "THICK", "POROS", "NLAYERS"]

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
            Formatted stringCO2 Indata

        """
        if param_name == "NLAYERS":
            return str(int(value))
        elif param_name in ["DPCOEF", "POROS"]:
            return f"{value:.3f}"
        elif param_name == "THICK":
            return f"{value:.1f}"
        else:  # PERMAV
            return f"{value:.1f}"


def generate_sensitivity_csv(
    output_file: str,
    n_runs: int = 10,
    seed: int = None,
    custom_ranges: dict = None,
    custom_distributions: dict = None,
    backup_dir: str = None,
) -> None:
    """
    Convenience function to generate sensitivity analysis CSV.

    Args:
        output_file: Path to output CSV file
        n_runs: Number of runs to generate
        seed: Random seed for reproducibility
        custom_ranges: Dictionary of custom ranges for parameters
        custom_distributions: Dictionary of distribution types for parameters
        backup_dir: Optional directory to save a backup copy of the CSV
    """
    generator = ParameterGenerator(seed=seed)

    # Default ranges
    ranges = {
        "dpcoef_range": (0.7, 1.0),
        "permav_range": (10, 200),
        "thick_range": (1, 20),
        "poros_range": (0.15, 0.35),
        "nlayers_range": (3, 10),
    }

    # Update with custom ranges if provided
    if custom_ranges:
        ranges.update(custom_ranges)

    # Generate parameters
    params_list = generator.generate_parameters(
        n_runs=n_runs, distributions=custom_distributions, **ranges
    )

    # Format values for better readability
    for params in params_list:
        params["DPCOEF"] = round(params["DPCOEF"], 2)
        params["PERMAV"] = round(params["PERMAV"], 1)
        params["THICK"] = round(params["THICK"], 1)
        params["POROS"] = round(params["POROS"], 3)

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
