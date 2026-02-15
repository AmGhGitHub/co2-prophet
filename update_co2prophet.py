"""
Update regression coefficients in web app files from regression_equations.txt.

This script parses the regression_equations.txt file and updates:
1. FastAPI router (oil_recovery_factor_router.py) - Python coefficients
2. Next.js page (page.tsx) - TypeScript/JavaScript coefficients

Usage:
    python update_coefficients.py
"""

import argparse
import os
import re
import subprocess
from typing import Dict

# Configuration
DEFAULT_COMMIT_MESSAGE = "Update ML regression coefficients and equations"


def parse_regression_file(filepath: str) -> Dict:
    """
    Parse regression_equations.txt and extract scaling parameters and coefficients.

    Returns:
        Dictionary with:
        - scaling_params: dict of min/max for each parameter
        - oil_recovery_at_1hcpv: dict with intercept and terms
        - oil_recovery_at_2hcpv: dict with intercept and terms
        - tornado_data: dict with correlation data for tornado charts
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    result = {"scaling_params": {}, "targets": {}, "tornado_data": {}}

    # Extract scaling parameters
    scaling_section = re.search(
        r"Scaling parameters - Min and Max for each feature:\s*\n(.*?)\n\n",
        content,
        re.DOTALL,
    )
    if scaling_section:
        for line in scaling_section.group(1).strip().split("\n"):
            # Match: "  DPCOEF     : min =     0.3100, max =     0.9200"
            # Note: DPCOEF may have 0 or more leading spaces
            match = re.match(
                r"\s*(\w+)\s+:\s+min\s*=\s*([\d.]+),\s*max\s*=\s*([\d.]+)", line
            )
            if match:
                param_name = match.group(1)
                min_val = float(match.group(2))
                max_val = float(match.group(3))
                result["scaling_params"][param_name] = {"min": min_val, "max": max_val}

    # Extract coefficients for each target
    target_pattern = r"Target: ([\w_]+)\s*\n-+\s*\n\s*\nR² Score \(test\): ([\d.]+)\s*\n\s*\nPolynomial Equation.*?:\s*\n\s+[\w_]+ = ([\d.-]+)\s*\n\s*\n\s+All terms \(sorted by importance\):\s*\n(.*?)\n\n"

    for match in re.finditer(target_pattern, content, re.DOTALL):
        target_name = match.group(1)
        r2_score = float(match.group(2))
        intercept = float(match.group(3))
        terms_text = match.group(4)

        terms = []
        for term_line in terms_text.strip().split("\n"):
            # Match: "    + 8.79125455 * SOINIT" or "     -9.78520695 * DPCOEF SOINIT"
            term_match = re.match(
                r"\s*([+-])\s*([\d.]+)\s*\*\s*(.+)", term_line.strip()
            )
            if term_match:
                sign = term_match.group(1)
                coeff_abs = float(term_match.group(2))
                term_type = term_match.group(3).strip()

                coeff = coeff_abs if sign == "+" else -coeff_abs
                terms.append({"type": term_type, "coeff": coeff})

        result["targets"][target_name] = {
            "r2_score": r2_score,
            "intercept": intercept,
            "terms": terms,
        }

    # Extract tornado chart data (correlations with targets)
    # Look for correlation data in the file
    correlation_pattern = (
        r"Correlations with ([\w_]+):\s*\n((?:\s+\w+\s+:\s+[-\d.]+\s+\([^)]+\)\s*\n)+)"
    )

    for match in re.finditer(correlation_pattern, content):
        target_name = match.group(1)
        corr_text = match.group(2)

        correlations = {}
        for line in corr_text.strip().split("\n"):
            # Match: "  DPCOEF     : -0.5511  (Strong)" - note multiple spaces between param and colon
            corr_match = re.match(r"(\w+)\s+:\s+([-\d.]+)\s+\([^)]+\)", line.strip())
            if corr_match:
                param_name = corr_match.group(1)
                corr_value = float(corr_match.group(2))
                correlations[param_name] = corr_value

        if correlations:
            result["tornado_data"][target_name] = correlations

    return result


def format_term_for_python(term_type: str) -> str:
    """Convert term type to Python dict format."""
    # Replace spaces with * for interactions
    # Replace ^ with ^
    return term_type.replace(" ", "*")


def format_term_for_typescript(term_type: str) -> str:
    """Convert term type to TypeScript format."""
    return term_type.replace(" ", "*")


def format_term_for_jsx_display(term_type: str, coeff: float) -> str:
    """
    Format a term for JSX display with proper subscripts and symbols.

    Args:
        term_type: The term type (e.g., "DPCOEF*SOINIT", "MMP^2", "POROS")
        coeff: The coefficient value

    Returns:
        Formatted JSX string like: "− 9.785×V<sub>DP</sub>×So<sub>i</sub>"
    """
    # Determine sign
    sign = "− " if coeff < 0 else "+ "
    abs_coeff = abs(coeff)

    # Format coefficient (3 decimal places)
    coeff_str = f"{abs_coeff:.3f}"

    # Replace parameter names with formatted versions
    formatted = term_type.replace(" ", "*")  # Normalize spaces to *

    # Handle squared terms
    if "^2" in formatted:
        param = formatted.replace("^2", "")
        formatted_param = format_param_for_jsx(param)
        return f"{sign}{coeff_str}×{formatted_param}²"

    # Handle interaction terms
    if "*" in formatted:
        parts = formatted.split("*")
        formatted_parts = [format_param_for_jsx(p) for p in parts]
        return f"{sign}{coeff_str}×" + "×".join(formatted_parts)

    # Handle linear terms
    formatted_param = format_param_for_jsx(formatted)
    return f"{sign}{coeff_str}×{formatted_param}"


def format_param_for_jsx(param: str) -> str:
    """
    Format a parameter name for JSX display with proper subscripts.

    Args:
        param: Parameter name (DPCOEF, POROS, MMP, SOINIT, XKVH)

    Returns:
        Formatted JSX string with subscripts
    """
    param_map = {
        "DPCOEF": "V<sub>DP</sub>",
        "POROS": "φ",
        "MMP": "MMP",
        "SOINIT": "So<sub>i</sub>",
        "XKVH": "K<sub>v</sub>/K<sub>h</sub>",
    }
    return param_map.get(param, param)


def generate_typescript_tornado_data(tornado_data: Dict) -> str:
    """
    Generate TypeScript tornado chart data.

    Args:
        tornado_data: Dictionary with correlation data for each target

    Returns:
        TypeScript code for tornado chart data
    """
    lines = []
    lines.append("// Tornado chart data (correlations)")
    lines.append("const TORNADO_DATA = {")

    for target_name, correlations in tornado_data.items():
        if "1hcpv" in target_name:
            key = "oil_1hcpv"
        else:
            key = "oil_2hcpv"

        # Sort by absolute correlation value (descending)
        sorted_params = sorted(
            correlations.items(), key=lambda x: abs(x[1]), reverse=True
        )

        lines.append(f"  {key}: [")
        for param, corr_value in sorted_params:
            # Map parameter names to display labels (using Unicode subscripts where needed)
            param_label_map = {
                "DPCOEF": "Vᴅᴘ",  # V with DP in small caps (closest to subscript)
                "POROS": "φ",
                "MMP": "MMP",
                "SOINIT": "Sₒᵢ",  # S with oi subscript
                "XKVH": "Kᵥ/Kₕ",  # K with v and h subscripts
            }
            label = param_label_map.get(param, param)
            color = "#2ab300" if corr_value > 0 else "#042ec7"

            lines.append(
                f'    {{ param: "{label}", correlation: {corr_value:.4f}, color: "{color}" }},'
            )
        lines.append("  ],")

    lines.append("};")
    lines.append("")

    return "\n".join(lines)


def generate_jsx_equation(target_name: str, target_data: Dict) -> str:
    """
    Generate JSX equation display for TypeScript page with dynamic tornado chart.

    Args:
        target_name: Target name (oil_recovery_at_1hcpv or oil_recovery_at_2hcpv)
        target_data: Dictionary with intercept, terms, and r2_score

    Returns:
        Multi-line JSX string for equation display
    """
    # Determine color and label
    if "1hcpv" in target_name:
        color = "blue-600"
        hcpv = "1.0"
        tornado_key = "oil_1hcpv"
    else:
        color = "teal-600"
        hcpv = "2.0"
        tornado_key = "oil_2hcpv"

    # Use actual R² score from data
    r2 = target_data["r2_score"]

    lines = []
    lines.append(f'                <div className="space-y-2">')
    lines.append(
        f'                  <h3 className="font-semibold text-{color}">Oil Recovery @ {hcpv} HCPV</h3>'
    )
    lines.append(f'                  <div className="flex gap-4 items-start">')
    lines.append(
        f'                    <div className="flex-shrink-0" style={{{{ width: "550px" }}}}>'
    )

    # Dynamic tornado chart instead of static image
    lines.append(
        f'                      <ResponsiveContainer width="100%" height={{300}}>'
    )
    lines.append(
        f'                        <BarChart data={{TORNADO_DATA.{tornado_key}}} layout="vertical" margin={{{{ top: 5, right: 30, left: 80, bottom: 5 }}}}>'
    )
    lines.append(f'                          <CartesianGrid strokeDasharray="3 3" />')
    lines.append(
        f'                          <XAxis type="number" domain={{[-1, 1]}} ticks={{[-1, -0.5, 0, 0.5, 1]}} />'
    )
    lines.append(
        f"                          <YAxis type=\"category\" dataKey=\"param\" width={{80}} style={{{{ fontSize: '14px', fontWeight: 'bold' }}}} />"
    )
    lines.append(
        f"                          <Tooltip formatter={{(value: number) => value.toFixed(3)}} />"
    )
    lines.append(
        f'                          <Bar dataKey="correlation" fill="#8884d8">'
    )
    lines.append(
        f"                            {{TORNADO_DATA.{tornado_key}.map((entry, index) => ("
    )
    lines.append(
        f"                              <Cell key={{`cell-${{index}}`}} fill={{entry.color}} />"
    )
    lines.append(f"                            ))}}")
    lines.append(f"                          </Bar>")
    lines.append(f"                        </BarChart>")
    lines.append(f"                      </ResponsiveContainer>")
    lines.append(f"                    </div>")
    lines.append(f'                    <div className="flex-1">')
    lines.append(
        f"                      <div className=\"bg-slate-50 p-4 rounded-md text-xs\" style={{{{fontFamily: 'Georgia, serif'}}}}>"
    )
    lines.append(f'                        <div className="leading-relaxed">')

    # First line: intercept and first few terms
    intercept = target_data["intercept"]
    line_content = f'                          <span className="font-semibold">Recovery</span> = {intercept:.3f} '

    # Group terms into lines (aim for ~3-4 terms per line for readability)
    terms = target_data["terms"]
    current_line = line_content
    term_count = 0

    for i, term in enumerate(terms):
        formatted_term = format_term_for_jsx_display(term["type"], term["coeff"])

        # Check if we should start a new line (every 3-4 terms)
        if term_count > 0 and term_count % 3 == 0:
            lines.append(current_line + "<br/>")
            current_line = "                          " + formatted_term
            term_count = 1
        else:
            if term_count == 0 and i == 0:
                # First term on first line
                current_line += formatted_term
            else:
                current_line += " " + formatted_term
            term_count += 1

    # Add the last line
    if current_line.strip():
        lines.append(current_line)

    lines.append(f"                        </div>")
    lines.append(f"                      </div>")
    lines.append(
        f'                      <p className="text-xs text-slate-500 italic mt-2">R² = {r2:.4f} | 20 terms (5 linear + 5 squared + 10 interactions)</p>'
    )
    lines.append(f"                    </div>")
    lines.append(f"                  </div>")
    lines.append(f"                </div>")

    return "\n".join(lines)


def generate_python_coefficients(data: Dict) -> str:
    """Generate Python code for coefficients."""
    lines = []

    # Scaling parameters
    lines.append("# Scaling parameters from regression_equations.txt")
    lines.append("SCALING_PARAMS = {")
    for param, values in data["scaling_params"].items():
        lines.append(
            f'    "{param}": {{"min": {values["min"]}, "max": {values["max"]}}},'
        )
    lines.append("}")
    lines.append("")

    # Coefficients for each target
    for target_name, target_data in data["targets"].items():
        if target_name == "oil_recovery_at_1hcpv":
            var_name = "COEFFS_1HCPV"
            comment = "Oil_at_1HCPV"
        else:
            var_name = "COEFFS_2HCPV"
            comment = "Oil_at_2HCPV"

        lines.append(f"# Regression coefficients for {comment}")
        lines.append(f"{var_name} = {{")
        lines.append(f'    "intercept": {target_data["intercept"]:.6f},')
        lines.append('    "terms": [')

        for term in target_data["terms"]:
            term_type = format_term_for_python(term["type"])
            coeff = term["coeff"]
            lines.append(f'        {{"type": "{term_type}", "coeff": {coeff:.8f}}},')

        lines.append("    ],")
        lines.append("}")
        lines.append("")

    return "\n".join(lines)


def generate_typescript_coefficients(data: Dict) -> str:
    """Generate TypeScript code for coefficients and tornado data."""
    lines = []

    # Scaling parameters
    lines.append("// Scaling parameters from regression_equations.txt")
    lines.append("const SCALING_PARAMS = {")
    for param, values in data["scaling_params"].items():
        label = {
            "DPCOEF": "DP Coeff",
            "POROS": "Porosity",
            "MMP": "MMP (kPa)",
            "SOINIT": "So,init",
            "XKVH": "Kv/Kh",
        }.get(param, param)
        lines.append(
            f'  {param}: {{ min: {values["min"]}, max: {values["max"]}, label: "{label}" }},'
        )
    lines.append("};")
    lines.append("")

    # Coefficients for each target
    for target_name, target_data in data["targets"].items():
        if target_name == "oil_recovery_at_1hcpv":
            var_name = "COEFFS_1HCPV"
            comment = "Oil_at_1HCPV"
        else:
            var_name = "COEFFS_2HCPV"
            comment = "Oil_at_2HCPV"

        lines.append(f"// Regression coefficients for {comment}")
        lines.append(f"const {var_name} = {{")
        lines.append(f"  intercept: {target_data['intercept']:.6f},")
        lines.append("  terms: [")

        for term in target_data["terms"]:
            term_type = format_term_for_typescript(term["type"])
            coeff = term["coeff"]
            lines.append(f'    {{ type: "{term_type}", coeff: {coeff:.8f} }},')

        lines.append("  ],")
        lines.append("};")
        lines.append("")

    # Add tornado data if available
    if data.get("tornado_data"):
        tornado_ts = generate_typescript_tornado_data(data["tornado_data"])
        lines.append(tornado_ts)

    return "\n".join(lines)


def update_python_file(filepath: str, new_content: str) -> None:
    """Update Python file with new coefficients."""
    if not os.path.exists(filepath):
        print(f"Warning: Python file not found: {filepath}")
        return

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Split content into sections
    sections = new_content.split("\n\n")
    if len(sections) < 3:
        print(f"Warning: Not enough coefficient sections generated for {filepath}")
        return

    # Helper function to match balanced braces for Python dicts
    def replace_dict_declaration(
        pattern_start: str, replacement: str, text: str
    ) -> str:
        """Replace a dict declaration with balanced brace matching."""
        import re

        # Find the start of the declaration
        match = re.search(pattern_start, text)
        if not match:
            return text

        start_pos = match.start()
        # Find the opening brace
        brace_pos = text.find("{", start_pos)
        if brace_pos == -1:
            return text

        # Count braces to find the closing one
        brace_count = 0
        pos = brace_pos
        while pos < len(text):
            if text[pos] == "{":
                brace_count += 1
            elif text[pos] == "}":
                brace_count -= 1
                if brace_count == 0:
                    # Found the matching closing brace
                    # Replace from start to closing brace (inclusive)
                    return text[:start_pos] + replacement + text[pos + 1 :]
            pos += 1

        return text

    # Replace scaling parameters (simple, no nested braces)
    content = re.sub(
        r"# Scaling parameters from regression_equations\.txt\s*\nSCALING_PARAMS = \{[^{]*\}",
        sections[0],
        content,
    )

    # Replace COEFFS_1HCPV with balanced brace matching
    content = replace_dict_declaration(
        r"# Regression coefficients for Oil_at_1HCPV\s*\nCOEFFS_1HCPV =",
        sections[1].rstrip(),
        content,
    )

    # Replace COEFFS_2HCPV with balanced brace matching
    content = replace_dict_declaration(
        r"# Regression coefficients for Oil_at_2HCPV\s*\nCOEFFS_2HCPV =",
        sections[2].rstrip(),
        content,
    )

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Updated: {filepath}")


def update_typescript_file(filepath: str, new_content: str, data: Dict) -> None:
    """Update TypeScript file with new coefficients and equation displays."""
    if not os.path.exists(filepath):
        print(f"Warning: TypeScript file not found: {filepath}")
        return

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Split content into sections
    sections = new_content.split("\n\n")
    if len(sections) < 3:
        print(f"Warning: Not enough coefficient sections generated for {filepath}")
        return

    # Helper function to match balanced braces
    def replace_const_declaration(
        pattern_start: str, replacement: str, text: str
    ) -> str:
        """Replace a const declaration with balanced brace matching."""
        import re

        # Find the start of the declaration
        match = re.search(pattern_start, text)
        if not match:
            return text

        start_pos = match.start()
        # Find the opening brace
        brace_pos = text.find("{", start_pos)
        if brace_pos == -1:
            return text

        # Count braces to find the closing one
        brace_count = 0
        pos = brace_pos
        while pos < len(text):
            if text[pos] == "{":
                brace_count += 1
            elif text[pos] == "}":
                brace_count -= 1
                if brace_count == 0:
                    # Found the matching closing brace
                    # Now find the semicolon
                    semi_pos = text.find(";", pos)
                    if (
                        semi_pos != -1 and semi_pos - pos < 3
                    ):  # Semicolon should be right after }
                        # Replace from start to semicolon
                        return text[:start_pos] + replacement + text[semi_pos + 1 :]
                    break
            pos += 1

        return text

    # Replace scaling parameters (simple, no nested braces)
    content = re.sub(
        r"// Scaling parameters from regression_equations\.txt\s*\nconst SCALING_PARAMS = \{[^{]*\};",
        sections[0],
        content,
    )

    # Replace COEFFS_1HCPV with balanced brace matching
    content = replace_const_declaration(
        r"// Regression coefficients for Oil_at_1HCPV\s*\nconst COEFFS_1HCPV =",
        sections[1].rstrip(),
        content,
    )

    # Replace COEFFS_2HCPV with balanced brace matching
    content = replace_const_declaration(
        r"// Regression coefficients for Oil_at_2HCPV\s*\nconst COEFFS_2HCPV =",
        sections[2].rstrip(),
        content,
    )

    # Add or replace TORNADO_DATA if available (it will be in sections[3])
    if len(sections) > 3 and data.get("tornado_data"):
        # Check if TORNADO_DATA already exists
        if "const TORNADO_DATA" in content:
            content = replace_const_declaration(
                r"// Tornado chart data \(correlations\)\s*\nconst TORNADO_DATA =",
                sections[3].rstrip(),
                content,
            )
        else:
            # Insert after COEFFS_2HCPV using balanced brace matching
            # Find position after COEFFS_2HCPV declaration
            match = re.search(r"const COEFFS_2HCPV = ", content)
            if match:
                start_pos = match.start()
                brace_pos = content.find("{", start_pos)

                # Count braces to find the closing one
                brace_count = 0
                pos = brace_pos
                while pos < len(content):
                    if content[pos] == "{":
                        brace_count += 1
                    elif content[pos] == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            # Found matching closing brace, find semicolon
                            semi_pos = content.find(";", pos)
                            if semi_pos != -1:
                                # Insert after the semicolon and newline
                                insert_pos = semi_pos + 1
                                # Skip any whitespace/newlines
                                while (
                                    insert_pos < len(content)
                                    and content[insert_pos] in "\n \t"
                                ):
                                    insert_pos += 1

                                # Insert TORNADO_DATA
                                content = (
                                    content[:insert_pos]
                                    + "\n"
                                    + sections[3]
                                    + "\n\n"
                                    + content[insert_pos:]
                                )
                            break
                    pos += 1

        # Ensure Cell is imported from recharts
        if "Cell" not in content:
            # Add Cell to the recharts imports - handle multiline import
            content = re.sub(
                r'(Line,)\s*\n(\} from "recharts";)', r"\1\n  Cell,\n\2", content
            )

    # Generate and replace equation JSX displays
    # Find and replace equation for 1.0 HCPV
    equation_1hcpv = generate_jsx_equation(
        "oil_recovery_at_1hcpv", data["targets"]["oil_recovery_at_1hcpv"]
    )

    # Pattern to match the entire equation div for 1.0 HCPV (updated for new structure with tornado chart)
    pattern_1hcpv = r'<div className="space-y-2">\s*<h3 className="font-semibold text-blue-600">Oil Recovery @ 1\.0 HCPV</h3>.*?</div>\s*</div>\s*</div>'

    content = re.sub(pattern_1hcpv, equation_1hcpv, content, flags=re.DOTALL, count=1)

    # Find and replace equation for 2.0 HCPV
    equation_2hcpv = generate_jsx_equation(
        "oil_recovery_at_2hcpv", data["targets"]["oil_recovery_at_2hcpv"]
    )

    # Pattern to match the entire equation div for 2.0 HCPV
    pattern_2hcpv = r'<div className="space-y-2">\s*<h3 className="font-semibold text-teal-600">Oil Recovery @ 2\.0 HCPV</h3>.*?</div>\s*</div>\s*</div>'

    content = re.sub(pattern_2hcpv, equation_2hcpv, content, flags=re.DOTALL, count=1)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Updated: {filepath}")
    print(f"  - Updated coefficient constants")
    print(f"  - Updated equation displays")


def git_commit_and_push(repo_dir: str, commit_message: str) -> bool:
    """
    Commit and push changes to git repository.

    Args:
        repo_dir: Path to the git repository
        commit_message: Commit message

    Returns:
        True if successful, False otherwise
    """
    try:
        # Change to repo directory
        original_dir = os.getcwd()
        os.chdir(repo_dir)

        # Check if there are changes to commit
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )

        if not result.stdout.strip():
            print("No changes to commit.")
            os.chdir(original_dir)
            return True

        # Git add, commit, push
        commands = [
            ["git", "add", "."],
            ["git", "commit", "-m", commit_message],
            ["git", "push", "origin", "main"],
        ]

        for cmd in commands:
            subprocess.run(cmd, check=True)

        print("Git push successful!")
        os.chdir(original_dir)
        return True

    except subprocess.CalledProcessError as e:
        print(f"Git operation failed: {e}")
        os.chdir(original_dir)
        return False
    except Exception as e:
        print(f"Error during git operations: {e}")
        return False


def main(push_to_git: bool = False, commit_message: str = None):
    """
    Main function to update coefficients.

    Args:
        push_to_git: If True, commit and push changes to git. Default is False.
        commit_message: Custom git commit message. Uses DEFAULT_COMMIT_MESSAGE if None.
    """
    # File paths
    regression_file = r"d:\temp\co2-prophet\results\ml-results\regression_equations.txt"
    python_file = (
        r"d:\CodingProjects\NextCO2Abacus\api\routers\oil_recovery_factor_router.py"
    )
    typescript_file = (
        r"d:\CodingProjects\NextCO2Abacus\app\oil-recovery-factor-estimator\page.tsx"
    )
    repo_dir = r"d:\CodingProjects\NextCO2Abacus"

    # Check if regression file exists
    if not os.path.exists(regression_file):
        print(f"Error: Regression file not found: {regression_file}")
        return

    print("Parsing regression_equations.txt...")
    data = parse_regression_file(regression_file)

    print(f"\nFound {len(data['scaling_params'])} scaling parameters")
    print(f"Found {len(data['targets'])} target equations")

    # Validate we have enough data
    if len(data["targets"]) != 2:
        print(
            f"Error: Expected 2 target equations, found {len(data['targets'])}. Check regression_equations.txt format."
        )
        return

    # Generate new content
    print("\nGenerating Python coefficients...")
    python_content = generate_python_coefficients(data)

    print("Generating TypeScript coefficients...")
    typescript_content = generate_typescript_coefficients(data)

    # Update files
    print("\nUpdating files...")
    update_python_file(python_file, python_content)
    update_typescript_file(typescript_file, typescript_content, data)

    print("\nDone! Coefficients updated successfully.")
    print("\nSummary:")
    print(f"  - Scaling parameters: {list(data['scaling_params'].keys())}")
    for target_name, target_data in data["targets"].items():
        print(
            f"  - {target_name}: R² = {target_data['r2_score']:.4f}, {len(target_data['terms'])} terms"
        )

    # Git commit and push (only if enabled)
    if push_to_git:
        print("\n" + "=" * 60)
        print("Committing and pushing changes to git...")
        msg = commit_message if commit_message else DEFAULT_COMMIT_MESSAGE
        git_commit_and_push(repo_dir, msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Update ML regression coefficients in web app files"
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Commit and push changes to git after updating files",
    )
    parser.add_argument(
        "--message",
        "-m",
        type=str,
        nargs="*",
        default=None,
        help=f"Custom git commit message (default: '{DEFAULT_COMMIT_MESSAGE}')",
    )

    args = parser.parse_args()

    # Join message parts if provided
    commit_msg = None
    if args.message:
        commit_msg = " ".join(args.message)

    main(push_to_git=args.push, commit_message=commit_msg)
