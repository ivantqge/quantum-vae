#!/usr/bin/env python3
# export_trig_luts.py
#
# Generate LUTs for dynamic input angle encoding:
# - eta/phi domain: [-pi, pi]
# - pt domain: [0, pi]
#
# These tables store cos(theta/2), sin(theta/2) for each quantized bin.

import argparse
import math
from pathlib import Path


def emit_array(f, ctype, name, values, per_line=4):
    f.write(f"static const {ctype} {name}[{len(values)}] = {{\n")
    for i, v in enumerate(values):
        if i % per_line == 0:
            f.write("    ")
        f.write(f"{v:.18e}")
        if i != len(values) - 1:
            f.write(", ")
        if (i % per_line == per_line - 1) or (i == len(values) - 1):
            f.write("\n")
    f.write("};\n\n")


def make_lut(nbits, theta_min, theta_max):
    n = 1 << nbits
    c_vals = []
    s_vals = []
    theta_vals = []
    for i in range(n):
        # Center-of-bin representative (better than left edge)
        t = theta_min + (i + 0.5) * (theta_max - theta_min) / n
        theta_vals.append(t)
        c_vals.append(math.cos(0.5 * t))
        s_vals.append(math.sin(0.5 * t))
    return theta_vals, c_vals, s_vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output header path (e.g. trig_luts.h)")
    ap.add_argument("--nbits", type=int, default=10, help="LUT index bits (10=>1024 entries)")
    args = ap.parse_args()

    out_path = Path(args.out)
    nbits = args.nbits

    eta_phi_theta, eta_phi_c, eta_phi_s = make_lut(nbits, -math.pi, math.pi)
    pt_theta, pt_c, pt_s = make_lut(nbits, 0.0, math.pi)

    with open(out_path, "w") as f:
        f.write("// Auto-generated trig LUTs for angle encoding\n\n")
        f.write("#pragma once\n\n")
        f.write(f"static const int TRIG_LUT_NBITS = {nbits};\n")
        f.write(f"static const int TRIG_LUT_SIZE = {1 << nbits};\n\n")

        emit_array(f, "double", "eta_phi_lut_cos", eta_phi_c)
        emit_array(f, "double", "eta_phi_lut_sin", eta_phi_s)
        emit_array(f, "double", "pt_lut_cos", pt_c)
        emit_array(f, "double", "pt_lut_sin", pt_s)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()