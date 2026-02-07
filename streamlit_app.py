import io
import os
import contextlib

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

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

    tab1, tab2, tab3 = st.tabs(["Configure", "Parameters", "Results"])

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
                f"Distribution for {pname}", dist_options, index=dist_options.index(default_dists.get(pname, "uniform")), key=f"dist_{pname}", label_visibility="collapsed"
            )

            # Default values
            dmin, dmax = param_ranges_defaults[pname]

            # Uniform: min, max
            if distributions[pname] == "uniform":
                a = col_a.number_input(f"{pname} Min", value=float(dmin), key=f"{pname}_a", label_visibility="collapsed")
                b = col_b.number_input(f"{pname} Max", value=float(dmax), key=f"{pname}_b", label_visibility="collapsed")
                col_c.empty()
                custom_ranges[f"{pname.lower()}_range"] = (a, b)

            # Triangular: min, mode, max
            elif distributions[pname] == "triangular":
                default_mode = (dmin + dmax) / 2
                a = col_a.number_input(f"{pname} Min", value=float(dmin), key=f"{pname}_a", label_visibility="collapsed")
                mode = col_b.number_input(f"{pname} Mode", value=float(default_mode), key=f"{pname}_mode", label_visibility="collapsed")
                c = col_c.number_input(f"{pname} Max", value=float(dmax), key=f"{pname}_c", label_visibility="collapsed")
                custom_ranges[f"{pname.lower()}_range"] = (a, mode, c)

            # Normal: mean, std (we convert to min/max = mean ± 3*std for generator)
            elif distributions[pname] == "normal":
                default_mean = (dmin + dmax) / 2
                default_std = (dmax - dmin) / 6
                mean = col_a.number_input(f"{pname} Mean", value=float(default_mean), key=f"{pname}_mean", label_visibility="collapsed")
                std = col_b.number_input(f"{pname} Std Dev", value=float(default_std), key=f"{pname}_std", label_visibility="collapsed")
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

    # Results tab: Process simulation outputs
    with tab3:
        st.header("Process Simulation Results")
        
        # Load session configs or defaults
        paths_cfg = st.session_state.get("paths_cfg", {})
        
        # Get directories
        prophet_results_dir = paths_cfg.get("prophet_results_dir", OUTPUT_CONVERTER_CONFIG["input_dir"])
        csv_output_dir = paths_cfg.get("csv_output_dir", OUTPUT_CONVERTER_CONFIG["output_dir"])
        
        st.subheader("📊 Convert OUTPUT Files to CSV")
        st.info(f"**Input:** {prophet_results_dir}")
        st.info(f"**Output:** {csv_output_dir}")
        
        # Check if OUTPUT files exist
        if os.path.exists(prophet_results_dir):
            output_files = [f for f in os.listdir(prophet_results_dir) if f.startswith("OUTPUT_")]
            st.metric("OUTPUT Files Found", len(output_files))
        else:
            st.warning(f"Directory not found: {prophet_results_dir}")
            output_files = []
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Convert OUTPUT to CSV", use_container_width=True, disabled=len(output_files) == 0):
                with st.spinner("Converting OUTPUT files to CSV..."):
                    try:
                        # Create output directory if it doesn't exist
                        os.makedirs(csv_output_dir, exist_ok=True)
                        
                        # Convert files
                        convert_output_to_csv(prophet_results_dir, csv_output_dir)
                        st.success(f"✅ Converted {len(output_files)} files successfully!")
                        
                        # Show converted files
                        csv_files = [f for f in os.listdir(csv_output_dir) if f.startswith("OUTPUT_") and f.endswith(".csv")]
                        st.info(f"Created {len(csv_files)} CSV files in {csv_output_dir}")
                        
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
        
        with col2:
            if st.button("📈 Extract Key Metrics", use_container_width=True):
                with st.spinner("Extracting metrics at 1 & 2 HCPV..."):
                    try:
                        # Check if CSV files exist
                        if not os.path.exists(csv_output_dir):
                            st.error(f"CSV directory not found: {csv_output_dir}")
                        else:
                            csv_files = [f for f in os.listdir(csv_output_dir) if f.startswith("OUTPUT_") and f.endswith(".csv")]
                            if not csv_files:
                                st.warning("No OUTPUT CSV files found. Please convert OUTPUT files first.")
                            else:
                                # Extract metrics
                                output_file = os.path.join(csv_output_dir, "summary_metrics.csv")
                                summary_df = extract_key_metrics(csv_output_dir, output_file)
                                
                                st.success(f"✅ Extracted metrics from {len(csv_files)} files")
                                
                                # Display summary
                                st.subheader("Summary Metrics")
                                st.dataframe(summary_df, use_container_width=True)
                                
                                # Display statistics
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.metric("Oil at ~1 HCPV (Mean)", f"{summary_df['Oil_at_1HCPV'].mean():.2f}%")
                                    st.metric("Oil at ~1 HCPV (Std)", f"{summary_df['Oil_at_1HCPV'].std():.2f}%")
                                with col_b:
                                    st.metric("Oil at ~2 HCPV (Mean)", f"{summary_df['Oil_at_2HCPV'].mean():.2f}%")
                                    st.metric("Oil at ~2 HCPV (Std)", f"{summary_df['Oil_at_2HCPV'].std():.2f}%")
                                
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
        
        st.markdown("---")
        
        # Display summary metrics if they exist
        st.subheader("📊 Oil Recovery at Key Injection Points")
        
        summary_file = os.path.join(csv_output_dir, "summary_metrics.csv")
        if os.path.exists(summary_file):
            try:
                summary_df = pd.read_csv(summary_file)
                
                # Display metrics in columns
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric("Total Runs", len(summary_df), help="Number of simulation runs")
                
                with col_b:
                    st.metric(
                        "Oil at ~1 HCPV",
                        f"{summary_df['Oil_at_1HCPV'].mean():.2f}%",
                        delta=f"± {summary_df['Oil_at_1HCPV'].std():.2f}%",
                        help="Mean ± Std Dev"
                    )
                
                with col_c:
                    st.metric(
                        "Oil at ~2 HCPV",
                        f"{summary_df['Oil_at_2HCPV'].mean():.2f}%",
                        delta=f"± {summary_df['Oil_at_2HCPV'].std():.2f}%",
                        help="Mean ± Std Dev"
                    )
                
                # Show detailed statistics
                with st.expander("📈 View Detailed Statistics"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Oil Recovery at ~1 HCPV (%OOIP)**")
                        stats_1 = {
                            "Mean": f"{summary_df['Oil_at_1HCPV'].mean():.2f}%",
                            "Std Dev": f"{summary_df['Oil_at_1HCPV'].std():.2f}%",
                            "Min": f"{summary_df['Oil_at_1HCPV'].min():.2f}%",
                            "Max": f"{summary_df['Oil_at_1HCPV'].max():.2f}%",
                            "Median": f"{summary_df['Oil_at_1HCPV'].median():.2f}%",
                        }
                        for key, value in stats_1.items():
                            st.text(f"{key:10}: {value}")
                    
                    with col2:
                        st.markdown("**Oil Recovery at ~2 HCPV (%OOIP)**")
                        stats_2 = {
                            "Mean": f"{summary_df['Oil_at_2HCPV'].mean():.2f}%",
                            "Std Dev": f"{summary_df['Oil_at_2HCPV'].std():.2f}%",
                            "Min": f"{summary_df['Oil_at_2HCPV'].min():.2f}%",
                            "Max": f"{summary_df['Oil_at_2HCPV'].max():.2f}%",
                            "Median": f"{summary_df['Oil_at_2HCPV'].median():.2f}%",
                        }
                        for key, value in stats_2.items():
                            st.text(f"{key:10}: {value}")
                    
                    # Show full table
                    st.markdown("**Full Summary Table**")
                    st.dataframe(summary_df, use_container_width=True, height=300)
                
                # Show Plotly histograms
                st.markdown("**Distribution Histograms**")
                col_hist1, col_hist2 = st.columns(2)
                
                with col_hist1:
                    fig1 = go.Figure()
                    fig1.add_trace(go.Histogram(
                        x=summary_df['Oil_at_1HCPV'],
                        nbinsx=20,
                        name='Frequency',
                        marker_color='#1f77b4',
                        opacity=0.7
                    ))
                    fig1.add_vline(
                        x=summary_df['Oil_at_1HCPV'].mean(),
                        line_dash="dash",
                        line_color="red",
                        line_width=2,
                        annotation_text=f"Mean: {summary_df['Oil_at_1HCPV'].mean():.2f}%",
                        annotation_position="top right"
                    )
                    fig1.update_layout(
                        title="Distribution at ~1 HCPV",
                        xaxis_title="Oil Recovery at ~1 HCPV (%OOIP)",
                        yaxis_title="Frequency",
                        showlegend=False,
                        height=400
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col_hist2:
                    fig2 = go.Figure()
                    fig2.add_trace(go.Histogram(
                        x=summary_df['Oil_at_2HCPV'],
                        nbinsx=20,
                        name='Frequency',
                        marker_color='#ff7f0e',
                        opacity=0.7
                    ))
                    fig2.add_vline(
                        x=summary_df['Oil_at_2HCPV'].mean(),
                        line_dash="dash",
                        line_color="red",
                        line_width=2,
                        annotation_text=f"Mean: {summary_df['Oil_at_2HCPV'].mean():.2f}%",
                        annotation_position="top right"
                    )
                    fig2.update_layout(
                        title="Distribution at ~2 HCPV",
                        xaxis_title="Oil Recovery at ~2 HCPV (%OOIP)",
                        yaxis_title="Frequency",
                        showlegend=False,
                        height=400
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                
            except Exception as e:
                st.warning(f"Could not load summary metrics: {e}")
        else:
            st.info("No summary metrics available. Click 'Extract Key Metrics' button above to generate them.")
        
        st.markdown("---")
        
        # Plot results
        st.subheader("📉 Visualization")
        
        if st.button("🎨 Generate Plot", use_container_width=True):
            with st.spinner("Generating plot..."):
                try:
                    plot_out = paths_cfg.get("plot_output", PLOTTER_CONFIG.get("output_plot"))
                    
                    # Change extension to .html for Plotly
                    if plot_out and not plot_out.endswith('.html'):
                        plot_out = plot_out.replace('.png', '.html')
                    
                    # Ensure output directory exists
                    os.makedirs(os.path.dirname(plot_out), exist_ok=True)
                    
                    # Generate plot
                    plot_oil_vs_injected(csv_output_dir, plot_out)
                    
                    st.success("✅ Interactive plot generated successfully!")
                    
                    # Display plot - read HTML and use components
                    if os.path.exists(plot_out):
                        with open(plot_out, 'r', encoding='utf-8') as f:
                            html_content = f.read()
                        st.components.v1.html(html_content, height=850, scrolling=True)
                    
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        
        st.markdown("---")
        
        # View existing results
        st.subheader("📁 View Results")
        
        if os.path.exists(csv_output_dir):
            csv_files = [f for f in os.listdir(csv_output_dir) if f.endswith(".csv")]
            if csv_files:
                selected_file = st.selectbox("Select CSV file to view:", sorted(csv_files))
                if selected_file:
                    try:
                        df = pd.read_csv(os.path.join(csv_output_dir, selected_file))
                        st.dataframe(df, use_container_width=True, height=400)
                    except Exception as e:
                        st.error(f"Error reading file: {e}")
            else:
                st.info("No CSV files found in output directory")


if __name__ == "__main__":
    main()
