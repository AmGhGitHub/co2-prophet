import io
import os
import contextlib

import streamlit as st
import pandas as pd

from config import (
    INPUT_GENERATOR_CONFIG,
    PARAMETER_GENERATOR_CONFIG,
    OUTPUT_CONVERTER_CONFIG,
    PLOTTER_CONFIG,
    RESULTS_ANALYZER_CONFIG,
    TASKS,
    AVAILABLE_TASKS,
    get_all_paths,
)
from param_generator import generate_sensitivity_csv
from input_generator import process_csv_and_generate_input_files
from output_converter import convert_output_to_csv
from results_analyzer import extract_key_metrics
from plotter import plot_oil_vs_injected


def run_and_capture(fn, *args, **kwargs):
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            fn(*args, **kwargs)
    except Exception as e:
        buf.write(f"ERROR: {e}\n")
    return buf.getvalue()


def main():
    st.set_page_config(layout="wide")
    st.title("CO2 Prophet — Streamlit UI")

    tab1, tab2, tab3 = st.tabs(["Configure", "Parameters", "Run & Results"])

    # Configure tab: allow editing of paths used by other tasks
    with tab1:
        st.header("Configure paths")
        paths = get_all_paths()
        cfg = {}
        cfg["base_file"] = st.text_input(
            "Base input file (base)", value=INPUT_GENERATOR_CONFIG.get("base_file", paths.get("BASE_FILE"))
        )
        cfg["input_csv"] = st.text_input(
            "Input CSV file",
            value=INPUT_GENERATOR_CONFIG.get("csv_file", paths.get("INPUT_CSV_FILE")),
        )
        cfg["prophet_data_out"] = st.text_input(
            "Prophet data output dir",
            value=INPUT_GENERATOR_CONFIG.get("output_dir", paths.get("PROPHET_DATA_OUTPUT_DIR")),
        )
        cfg["prophet_results_dir"] = st.text_input(
            "Prophet results dir (OUTPUT files)",
            value=OUTPUT_CONVERTER_CONFIG.get("input_dir", paths.get("PROPHET_RESULTS_DIR")),
        )
        cfg["csv_output_dir"] = st.text_input(
            "CSV output dir",
            value=OUTPUT_CONVERTER_CONFIG.get("output_dir", paths.get("PROPHET_CSV_OUTPUT_DIR")),
        )
        cfg["csv_vars_dir"] = st.text_input(
            "CSV vars dir (copy of input CSV)",
            value=INPUT_GENERATOR_CONFIG.get("vdos_csv_dir", paths.get("PROPHET_CSV_VARS_DIR")),
        )
        cfg["plot_output"] = st.text_input(
            "Plot output file",
            value=PLOTTER_CONFIG.get("output_plot", paths.get("PLOT_OUTPUT_FILE")),
        )

        if st.button("Apply paths"):
            st.session_state["paths_cfg"] = cfg
            st.success("Paths applied to session")

    # Parameters tab: configure distributions and ranges per parameter
    with tab2:
        st.header("Parameter distributions & ranges")
        default_dists = PARAMETER_GENERATOR_CONFIG.get("distributions", {})
        # Map lognormal to uniform since we only support uniform, normal, triangular
        default_dists = {k: ("uniform" if v == "lognormal" else v) for k, v in default_dists.items()}
        param_ranges_defaults = {
            "DPCOEF": PARAMETER_GENERATOR_CONFIG.get("dpcoef_range", (0.7, 0.95)),
            "POROS": PARAMETER_GENERATOR_CONFIG.get("poros_range", (0.08, 0.13)),
            "MMP": PARAMETER_GENERATOR_CONFIG.get("mmp_range", (1200, 1500)),
            "SOINIT": PARAMETER_GENERATOR_CONFIG.get("soinit_range", (0.4, 0.6)),
            "XKVH": PARAMETER_GENERATOR_CONFIG.get("xkvh_range", (0.01, 0.1)),
        }

        distributions = {}
        custom_ranges = {}

        dist_options = ["uniform", "normal", "triangular"]

        # Show each parameter on one line: name | distribution | value fields
        for pname in ["DPCOEF", "POROS", "MMP", "SOINIT", "XKVH"]:
            col_name, col_dist, col_a, col_b, col_c = st.columns([0.8, 1.8, 1.8, 1.8, 1.8])
            
            # Parameter name (fixed, right-aligned)
            col_name.markdown(f"**{pname}**", help=None)
            
            # Distribution selector
            distributions[pname] = col_dist.selectbox(
                "", dist_options, index=dist_options.index(default_dists.get(pname, "uniform")), key=f"dist_{pname}", label_visibility="collapsed"
            )

            # Default values
            dmin, dmax = param_ranges_defaults[pname]

            # Uniform: min, max
            if distributions[pname] == "uniform":
                a = col_a.number_input("", value=float(dmin), key=f"{pname}_a", label_visibility="collapsed")
                b = col_b.number_input("", value=float(dmax), key=f"{pname}_b", label_visibility="collapsed")
                col_c.empty()
                custom_ranges[f"{pname.lower()}_range"] = (a, b)

            # Triangular: min, mode, max
            elif distributions[pname] == "triangular":
                default_mode = (dmin + dmax) / 2
                a = col_a.number_input("", value=float(dmin), key=f"{pname}_a", label_visibility="collapsed")
                mode = col_b.number_input("", value=float(default_mode), key=f"{pname}_mode", label_visibility="collapsed")
                c = col_c.number_input("", value=float(dmax), key=f"{pname}_c", label_visibility="collapsed")
                custom_ranges[f"{pname.lower()}_range"] = (a, mode, c)

            # Normal: mean, std (we convert to min/max = mean ± 3*std for generator)
            elif distributions[pname] == "normal":
                default_mean = (dmin + dmax) / 2
                default_std = (dmax - dmin) / 6
                mean = col_a.number_input("", value=float(default_mean), key=f"{pname}_mean", label_visibility="collapsed")
                std = col_b.number_input("", value=float(default_std), key=f"{pname}_std", label_visibility="collapsed")
                col_c.empty()
                custom_ranges[f"{pname.lower()}_range"] = (mean - 3 * std, mean + 3 * std)

        st.markdown("---")
        st.checkbox("Use Latin Hypercube Sampling (LHS)", value=PARAMETER_GENERATOR_CONFIG.get("use_lhs", True), key="use_lhs")
        st.number_input("Seed", value=PARAMETER_GENERATOR_CONFIG.get("seed", 42), key="seed")
        st.number_input("n_runs (0 = auto)", value=0, min_value=0, step=1, key="n_runs")
        st.selectbox(
            "Sensitivity level",
            ["minimum", "low", "medium", "high", "very_high"],
            index=["minimum", "low", "medium", "high", "very_high"].index(
                PARAMETER_GENERATOR_CONFIG.get("sensitivity_level", "medium")
            ),
            key="sensitivity_level",
        )

        if st.button("Generate sensitivity table & datafiles"):
            # Store distributions and ranges for later use
            st.session_state["param_dists"] = distributions
            st.session_state["param_ranges"] = custom_ranges
            
            # Get paths from session state or use defaults
            output_csv = st.session_state.get("paths_cfg", {}).get("input_csv") or PARAMETER_GENERATOR_CONFIG["output_file"]
            base_file = st.session_state.get("paths_cfg", {}).get("base_file") or INPUT_GENERATOR_CONFIG["base_file"]
            output_dir = st.session_state.get("paths_cfg", {}).get("prophet_data_out") or INPUT_GENERATOR_CONFIG["output_dir"]
            csv_vars_dir = st.session_state.get("paths_cfg", {}).get("csv_vars_dir") or INPUT_GENERATOR_CONFIG["vdos_csv_dir"]
            
            try:
                # Generate the sensitivity parameter CSV
                generate_sensitivity_csv(
                    output_file=output_csv,
                    n_runs=None if st.session_state.get("n_runs", 0) == 0 else int(st.session_state.get("n_runs")),
                    seed=int(st.session_state.get("seed", PARAMETER_GENERATOR_CONFIG.get("seed", 42))),
                    custom_ranges=custom_ranges,
                    custom_distributions=distributions,
                    backup_dir=csv_vars_dir,
                    use_lhs=st.session_state.get("use_lhs", PARAMETER_GENERATOR_CONFIG.get("use_lhs", True)),
                    sensitivity_level=st.session_state.get("sensitivity_level", PARAMETER_GENERATOR_CONFIG.get("sensitivity_level", "medium")),
                )
                
                # Read CSV into DataFrame
                df = pd.read_csv(output_csv)
                st.session_state["generated_params_df"] = df
                st.success(f"Generated {len(df)} parameter sets")
                st.dataframe(df, use_container_width=True)
                
                # Clear output directory before writing new datafiles
                st.info("Clearing output directory...")
                import shutil
                import glob
                if os.path.exists(output_dir):
                    # Delete all files in the directory
                    for file in glob.glob(os.path.join(output_dir, "*")):
                        try:
                            if os.path.isfile(file):
                                os.remove(file)
                        except Exception as e:
                            st.warning(f"Could not delete {file}: {e}")
                
                # Generate input datafiles from the parameters
                st.info("Generating input datafiles...")
                process_csv_and_generate_input_files(
                    base_file=base_file,
                    csv_file=output_csv,
                    output_prefix=INPUT_GENERATOR_CONFIG["output_prefix"],
                    output_file_dir=output_dir,
                    vdos_csv_dir=csv_vars_dir,
                )
                st.success(f"Generated {len(df)} input datafiles in {output_dir}")
                
            except Exception as e:
                st.error(f"Error: {e}")

    # Run & Results tab: select tasks and run
    with tab3:
        st.header("Run tasks & view results")
        # Load session configs or defaults
        paths_cfg = st.session_state.get("paths_cfg", {})
        param_dists = st.session_state.get("param_dists", PARAMETER_GENERATOR_CONFIG.get("distributions"))
        param_ranges = st.session_state.get("param_ranges", {
            "dpcoef_range": PARAMETER_GENERATOR_CONFIG.get("dpcoef_range"),
            "poros_range": PARAMETER_GENERATOR_CONFIG.get("poros_range"),
            "mmp_range": PARAMETER_GENERATOR_CONFIG.get("mmp_range"),
            "soinit_range": PARAMETER_GENERATOR_CONFIG.get("soinit_range"),
            "xkvh_range": PARAMETER_GENERATOR_CONFIG.get("xkvh_range"),
        })

        selected = {}
        for task_key, task_desc in AVAILABLE_TASKS.items():
            default = TASKS.get(task_key, False)
            selected[task_key] = st.checkbox(task_desc, value=default, key=f"task_{task_key}")

        run_btn = st.button("Run selected tasks")
        log_area = st.empty()

        if run_btn:
            logs = []

            # Helper to resolve path with session override
            def path_or(default_key, session_key):
                return paths_cfg.get(session_key) if session_key in paths_cfg else globals()[default_key]

            # Generate parameters
            if selected.get("generate_parameters"):
                out = run_and_capture(
                    generate_sensitivity_csv,
                    paths_cfg.get("input_csv") if paths_cfg.get("input_csv") else PARAMETER_GENERATOR_CONFIG["output_file"],
                    None if st.session_state.get("n_runs", 0) == 0 else int(st.session_state.get("n_runs")),
                    int(st.session_state.get("seed", PARAMETER_GENERATOR_CONFIG.get("seed", 42))),
                    param_ranges,
                    param_dists,
                    paths_cfg.get("csv_vars_dir", PARAMETER_GENERATOR_CONFIG.get("backup_dir")),
                    st.session_state.get("use_lhs", PARAMETER_GENERATOR_CONFIG.get("use_lhs", True)),
                    st.session_state.get("sensitivity_level", PARAMETER_GENERATOR_CONFIG.get("sensitivity_level", "medium")),
                )
                logs.append("--- Generate parameters ---\n" + out)

            # Generate input files
            if selected.get("generate_input_files"):
                out = run_and_capture(
                    process_csv_and_generate_input_files,
                    paths_cfg.get("base_file", INPUT_GENERATOR_CONFIG["base_file"]),
                    paths_cfg.get("input_csv", INPUT_GENERATOR_CONFIG["csv_file"]),
                    INPUT_GENERATOR_CONFIG["output_prefix"],
                    paths_cfg.get("prophet_data_out", INPUT_GENERATOR_CONFIG["output_dir"]),
                    paths_cfg.get("csv_vars_dir", INPUT_GENERATOR_CONFIG.get("vdos_csv_dir")),
                )
                logs.append("--- Generate input files ---\n" + out)

            # Convert outputs to CSV
            if selected.get("convert_output_to_csv"):
                out = run_and_capture(
                    convert_output_to_csv,
                    paths_cfg.get("prophet_results_dir", OUTPUT_CONVERTER_CONFIG["input_dir"]),
                    paths_cfg.get("csv_output_dir", OUTPUT_CONVERTER_CONFIG["output_dir"]),
                )
                logs.append("--- Convert outputs ---\n" + out)

            # Extract key metrics
            if selected.get("extract_key_metrics"):
                out = run_and_capture(
                    extract_key_metrics,
                    paths_cfg.get("csv_output_dir", RESULTS_ANALYZER_CONFIG["csv_dir"]),
                    paths_cfg.get("csv_output_dir", RESULTS_ANALYZER_CONFIG.get("output_file")),
                )
                logs.append("--- Extract key metrics ---\n" + out)
                # Show table if available
                try:
                    df = extract_key_metrics(paths_cfg.get("csv_output_dir", RESULTS_ANALYZER_CONFIG["csv_dir"]), None)
                    st.subheader("Summary metrics")
                    st.dataframe(df)
                except Exception:
                    pass

            # Plot results
            if selected.get("plot_results"):
                plot_out = paths_cfg.get("plot_output", PLOTTER_CONFIG.get("output_plot"))
                # Ensure output directory exists
                try:
                    os.makedirs(os.path.dirname(plot_out), exist_ok=True)
                except Exception:
                    pass
                out = run_and_capture(
                    plot_oil_vs_injected,
                    paths_cfg.get("csv_output_dir", PLOTTER_CONFIG["csv_dir"]),
                    plot_out,
                )
                logs.append("--- Plot results ---\n" + out)
                # Show image if created
                if plot_out and os.path.exists(plot_out):
                    st.image(plot_out, caption="Oil vs Injected total", use_column_width=True)

            log_text = "\n".join(logs) if logs else "No tasks selected."
            log_area.text_area("Execution log", value=log_text, height=400)


if __name__ == "__main__":
    main()
