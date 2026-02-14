with open(
    r"d:\CodingProjects\NextCO2Abacus\app\oil-recovery-factor-estimator\page.tsx",
    "r",
    encoding="utf-8",
) as f:
    lines = f.readlines()

# Find line with TORNADO_DATA
for i, line in enumerate(lines):
    if "TORNADO_DATA" in line:
        print(f"Line {i+1}: {line.rstrip()}")
        # Print surrounding lines
        if i > 0:
            print(f"Line {i}: {lines[i-1].rstrip()}")
        if i < len(lines) - 5:
            for j in range(1, 6):
                print(f"Line {i+1+j}: {lines[i+j].rstrip()}")
        break
