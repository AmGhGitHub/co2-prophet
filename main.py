"""
CO2 Prophet Main Script
Coordinates input generation, output conversion, and visualization.
"""

from input_generator import process_csv_and_generate_input_files
from output_converter import convert_output_to_csv
from plotter import plot_oil_vs_injected

if __name__ == "__main__":
    # Configuration
    BASE_FILE = "base"
    CSV_FILE = "./sen-vars/sen_fbv.csv"
    OUTPUT_FILE_DIR = "C:/vDos/Prophet/sen-datafiles"
    OUTPUT_PREFIX = "sen"
    process_csv_and_generate_input_files(
        BASE_FILE, CSV_FILE, OUTPUT_PREFIX, OUTPUT_FILE_DIR
    )

    # Generate input files from CSV
    # process_csv(BASE_FILE, CSV_FILE, OUTPUT_PREFIX, OUTPUT_FILE_DIR)

    # Convert OUTPUT files to CSV
    INPUT_RESULTS_DIR = "C:/vDos/Prophet/sen-output"
    OUTPUT_CSV_DIR = "C:/vDos/Prophet/sen-output-csv"
    convert_output_to_csv(INPUT_RESULTS_DIR, OUTPUT_CSV_DIR)

    # Plot Oil produced vs Injected total
    PLOT_OUTPUT = "C:/vDos/Prophet/sen-output-csv/oil_vs_injected.png"
    plot_oil_vs_injected(OUTPUT_CSV_DIR, PLOT_OUTPUT)
