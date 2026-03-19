#!/usr/bin/env python3
"""
Export fixed-point headers for Extended VAE Catapult HLS.
Generates:
  - catapult_extended_vae_trig_luts.h: Fixed-point trig LUTs for physics-aware normalization
  - catapult_extended_vae_weights.h: Fixed-point trainable weights with depth constants
"""

import argparse
import math
from pathlib import Path
import torch


def emit_ac_fixed_array(f, typename, name, values, per_line=4):
    """Emit a fixed-point array for Catapult HLS."""
    f.write(f"const {typename} {name}[{len(values)}] = {{\n")
    for i, v in enumerate(values):
        if i % per_line == 0:
            f.write("    ")
        f.write(f"{v:.10f}")
        if i != len(values) - 1:
            f.write(", ")
        if (i % per_line == per_line - 1) or (i == len(values) - 1):
            f.write("\n")
    f.write("};\n\n")


def generate_trig_luts(out_path, nbits=10):
    """Generate fixed-point trig LUTs for Catapult HLS.
    
    Extended VAE uses physics-aware normalization:
      - pt: [0, π]
      - eta/phi: [-π, π]
    """
    n = 1 << nbits
    
    # eta/phi range: [-π, π]
    eta_phi_c = []
    eta_phi_s = []
    for i in range(n):
        t = -math.pi + (i + 0.5) * (2 * math.pi) / n
        eta_phi_c.append(math.cos(0.5 * t))
        eta_phi_s.append(math.sin(0.5 * t))
    
    # pt range: [0, π]
    pt_c = []
    pt_s = []
    for i in range(n):
        t = (i + 0.5) * math.pi / n
        pt_c.append(math.cos(0.5 * t))
        pt_s.append(math.sin(0.5 * t))
    
    with open(out_path, "w") as f:
        f.write("// Auto-generated fixed-point trig LUTs for Extended VAE Catapult HLS\n")
        f.write("// Physics-aware normalization: pt in [0,π], eta/phi in [-π,π]\n")
        f.write("#pragma once\n\n")
        f.write('#include <ac_fixed.h>\n\n')
        f.write("typedef ac_fixed<16, 2, true, AC_RND, AC_SAT> angle_t;\n\n")
        f.write(f"#define TRIG_LUT_BITS {nbits}\n")
        f.write(f"#define TRIG_LUT_SIZE {n}\n\n")
        
        emit_ac_fixed_array(f, "angle_t", "eta_phi_lut_cos", eta_phi_c)
        emit_ac_fixed_array(f, "angle_t", "eta_phi_lut_sin", eta_phi_s)
        emit_ac_fixed_array(f, "angle_t", "pt_lut_cos", pt_c)
        emit_ac_fixed_array(f, "angle_t", "pt_lut_sin", pt_s)
    
    print(f"Wrote {out_path}")


def generate_weights(ckpt_path, out_path):
    """Generate fixed-point weights header for Catapult HLS."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    # Extract state dict
    if "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt
    
    def get_weights(keys):
        for k in keys:
            if k in sd:
                return sd[k].detach().cpu().double().flatten().tolist()
        raise KeyError(f"Could not find any of {keys}")
    
    # Extended VAE has RX, RY, RZ weights per block
    met_w = get_weights(["encoder.met_weights", "met_weights"])
    ele_w = get_weights(["encoder.ele_weights", "ele_weights"])
    mu_w = get_weights(["encoder.mu_weights", "mu_weights"])
    jet_w = get_weights(["encoder.jet_weights", "jet_weights"])
    
    # Infer depths from weight sizes
    met_depth = len(met_w) // 3
    ele_depth = len(ele_w) // 8
    mu_depth = len(mu_w) // 8
    jet_depth = len(jet_w) // 13
    
    def cs_arrays(angles):
        c = [math.cos(0.5 * a) for a in angles]
        s = [math.sin(0.5 * a) for a in angles]
        return c, s
    
    met_c, met_s = cs_arrays(met_w)
    ele_c, ele_s = cs_arrays(ele_w)
    mu_c, mu_s = cs_arrays(mu_w)
    jet_c, jet_s = cs_arrays(jet_w)
    
    with open(out_path, "w") as f:
        f.write("// Auto-generated fixed-point weights for Extended VAE Catapult HLS\n")
        f.write("#pragma once\n\n")
        f.write('#include <ac_fixed.h>\n\n')
        f.write("typedef ac_fixed<16, 2, true, AC_RND, AC_SAT> angle_t;\n\n")
        
        # Depth constants
        f.write(f"#define MET_DEPTH {met_depth}\n")
        f.write(f"#define ELE_DEPTH {ele_depth}\n")
        f.write(f"#define MU_DEPTH {mu_depth}\n")
        f.write(f"#define JET_DEPTH {jet_depth}\n\n")
        
        emit_ac_fixed_array(f, "angle_t", "met_weights_c", met_c)
        emit_ac_fixed_array(f, "angle_t", "met_weights_s", met_s)
        emit_ac_fixed_array(f, "angle_t", "ele_weights_c", ele_c)
        emit_ac_fixed_array(f, "angle_t", "ele_weights_s", ele_s)
        emit_ac_fixed_array(f, "angle_t", "mu_weights_c", mu_c)
        emit_ac_fixed_array(f, "angle_t", "mu_weights_s", mu_s)
        emit_ac_fixed_array(f, "angle_t", "jet_weights_c", jet_c)
        emit_ac_fixed_array(f, "angle_t", "jet_weights_s", jet_s)
    
    print(f"Wrote {out_path}")
    print(f"  Depths: MET={met_depth}, ELE={ele_depth}, MU={mu_depth}, JET={jet_depth}")


def emit_ac_fixed_2d_array(f, typename, name, values_2d):
    """Emit a 2D fixed-point array."""
    rows = len(values_2d)
    cols = len(values_2d[0]) if rows > 0 else 0
    f.write(f"const {typename} {name}[{rows}][{cols}] = {{\n")
    for i, row in enumerate(values_2d):
        f.write("    {")
        for j, v in enumerate(row):
            f.write(f"{v:.10f}")
            if j != len(row) - 1:
                f.write(", ")
        f.write("}")
        if i != rows - 1:
            f.write(",")
        f.write("\n")
    f.write("};\n\n")


def generate_classical_weights(ckpt_path, out_path):
    """Generate classical layer weights for full anomaly detection pipeline."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    if "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt
    
    def get_param(keys):
        for k in keys:
            if k in sd:
                return sd[k].detach().cpu().double()
        raise KeyError(f"Could not find any of {keys}")
    
    dense_hidden_w = get_param(["encoder.dense_hidden.weight", "dense_hidden.weight"])
    dense_hidden_b = get_param(["encoder.dense_hidden.bias", "dense_hidden.bias"])
    z_mean_w = get_param(["encoder.z_mean.weight", "z_mean.weight"])
    z_mean_b = get_param(["encoder.z_mean.bias", "z_mean.bias"])
    
    hidden_dim = dense_hidden_w.shape[0]
    latent_dim = z_mean_w.shape[0]
    
    with open(out_path, "w") as f:
        f.write("// Auto-generated classical layer weights for Extended VAE Catapult HLS\n")
        f.write("// Pipeline: Quantum(32) -> Dense(hidden_dim) -> z_mean(latent_dim) -> CKL score\n")
        f.write("#pragma once\n\n")
        f.write('#include <ac_fixed.h>\n\n')
        f.write("typedef ac_fixed<18, 2, true, AC_RND, AC_SAT> output_t;\n")
        f.write("typedef ac_fixed<24, 4, true, AC_RND, AC_SAT> acc_t;\n\n")
        f.write(f"#define HIDDEN_DIM {hidden_dim}\n")
        f.write(f"#define LATENT_DIM {latent_dim}\n\n")
        
        emit_ac_fixed_2d_array(f, "output_t", "dense_hidden_weight", dense_hidden_w.tolist())
        emit_ac_fixed_array(f, "output_t", "dense_hidden_bias", dense_hidden_b.tolist())
        emit_ac_fixed_2d_array(f, "output_t", "z_mean_weight", z_mean_w.tolist())
        emit_ac_fixed_array(f, "output_t", "z_mean_bias", z_mean_b.tolist())
    
    print(f"Wrote {out_path}")
    print(f"  hidden_dim={hidden_dim}, latent_dim={latent_dim}")


def main():
    parser = argparse.ArgumentParser(description="Export fixed-point headers for Extended VAE Catapult HLS")
    parser.add_argument("--ckpt", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--out-dir", default="catapult_extended_vae_headers", help="Output directory")
    parser.add_argument("--nbits", type=int, default=10, help="LUT index bits")
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    generate_trig_luts(out_dir / "catapult_extended_vae_trig_luts.h", args.nbits)
    generate_weights(args.ckpt, out_dir / "catapult_extended_vae_weights.h")
    generate_classical_weights(args.ckpt, out_dir / "catapult_extended_vae_classical.h")
    
    print(f"\nDone! Headers written to {out_dir}/")
    print("\nTo use with Catapult HLS:")
    print(f"  1. Copy {out_dir}/*.h and extended_vae_inference_catapult.cpp to your Catapult project")
    print("  2. Top function: extended_vae_anomaly_top (full anomaly detection)")
    print("  3. Run C simulation, then synthesis")


if __name__ == "__main__":
    main()
