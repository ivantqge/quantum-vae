// extended_vae_inference_catapult.cpp
//
// Fixed-point implementation for Siemens Catapult HLS.
// Extended Quantum VAE encoder: 56 features -> 32 outputs.
//
// Architecture:
// - MET: 1 qubit -> PauliZ, PauliX, PauliY -> 3 outputs
// - Electrons: 4 qubits -> PauliZ (4) + PauliX (4) -> 8 outputs
// - Muons: 4 qubits -> PauliZ (4) + PauliX (4) -> 8 outputs
// - Jets: 10 qubits -> PauliZ (10) + PauliX (3) -> 13 outputs
// Total: 32 outputs
//
// Input normalization (physics-aware, angles):
//   - pt:  log(pt + 1) scaled to [0, π]
//   - eta: scaled from [-3, 3] to [-π, π]
//   - phi: clamped to [-π, π]
// NOTE: Input should be PRE-NORMALIZED (use LazyH5Array with norm=True)
//
// Bit widths:
// - Angles/trig values: 16 bits (2 integer, 14 fractional)
// - Statevector amplitudes: 18 bits (2 integer, 16 fractional)
// - Accumulator for measurements: 24 bits (4 integer, 20 fractional)
// - Outputs: 18 bits (2 integer, 16 fractional)

#include <ac_fixed.h>
#include <ac_channel.h>

// ========================== Type definitions ==========================

typedef ac_fixed<16, 2, true, AC_RND, AC_SAT> angle_t;
typedef ac_fixed<18, 2, true, AC_RND, AC_SAT> amp_t;
typedef ac_fixed<24, 4, true, AC_RND, AC_SAT> acc_t;
typedef ac_fixed<18, 2, true, AC_RND, AC_SAT> output_t;
typedef ac_fixed<16, 4, true, AC_RND, AC_SAT> input_t;  // Pre-normalized angles

struct cx_t {
    amp_t re;
    amp_t im;
};

// ========================== Constants ==========================
#define PI_VAL 3.14159265358979323846

// ========================== LUT configuration ==========================
#define TRIG_LUT_BITS 10
#define TRIG_LUT_SIZE (1 << TRIG_LUT_BITS)

// Include generated headers
#include "catapult_extended_vae_weights.h"
#include "catapult_extended_vae_trig_luts.h"

// ========================== LUT indexing ==========================
// Extended VAE uses physics-aware normalization (angles in radians)

inline ac_int<TRIG_LUT_BITS, false> lut_index_eta_phi(input_t theta) {
    const input_t lo = -PI_VAL;
    const input_t hi = PI_VAL;
    input_t t = theta;
    if (t < lo) t = lo;
    if (t > hi) t = hi;
    input_t u = (t - lo) / (hi - lo);
    return (ac_int<TRIG_LUT_BITS+4, false>)(u * (TRIG_LUT_SIZE - 1));
}

inline ac_int<TRIG_LUT_BITS, false> lut_index_pt(input_t theta) {
    const input_t lo = 0.0;
    const input_t hi = PI_VAL;
    input_t t = theta;
    if (t < lo) t = lo;
    if (t > hi) t = hi;
    input_t u = (t - lo) / (hi - lo);
    return (ac_int<TRIG_LUT_BITS+4, false>)(u * (TRIG_LUT_SIZE - 1));
}

inline void get_cs_eta_phi(input_t theta, angle_t& c, angle_t& s) {
    ac_int<TRIG_LUT_BITS, false> idx = lut_index_eta_phi(theta);
    c = eta_phi_lut_cos[idx];
    s = eta_phi_lut_sin[idx];
}

inline void get_cs_pt(input_t theta, angle_t& c, angle_t& s) {
    ac_int<TRIG_LUT_BITS, false> idx = lut_index_pt(theta);
    c = pt_lut_cos[idx];
    s = pt_lut_sin[idx];
}

// ========================== Gate operations ==========================

template<int NQ>
void apply_rx(cx_t sv[1 << NQ], int wire, angle_t c, angle_t s) {
    const int stride = 1 << wire;
    const int DIM = 1 << NQ;
    
    #pragma hls_unroll yes
    for (int base = 0; base < DIM; base += (stride << 1)) {
        #pragma hls_unroll yes
        for (int j = 0; j < stride; ++j) {
            int i0 = base + j;
            int i1 = i0 + stride;
            
            cx_t a = sv[i0];
            cx_t b = sv[i1];
            
            cx_t ap, bp;
            ap.re = c * a.re + s * b.im;
            ap.im = c * a.im - s * b.re;
            bp.re = s * a.im + c * b.re;
            bp.im = -s * a.re + c * b.im;
            
            sv[i0] = ap;
            sv[i1] = bp;
        }
    }
}

template<int NQ>
void apply_ry(cx_t sv[1 << NQ], int wire, angle_t c, angle_t s) {
    const int stride = 1 << wire;
    const int DIM = 1 << NQ;
    
    #pragma hls_unroll yes
    for (int base = 0; base < DIM; base += (stride << 1)) {
        #pragma hls_unroll yes
        for (int j = 0; j < stride; ++j) {
            int i0 = base + j;
            int i1 = i0 + stride;
            
            cx_t a = sv[i0];
            cx_t b = sv[i1];
            
            cx_t ap, bp;
            ap.re = c * a.re - s * b.re;
            ap.im = c * a.im - s * b.im;
            bp.re = s * a.re + c * b.re;
            bp.im = s * a.im + c * b.im;
            
            sv[i0] = ap;
            sv[i1] = bp;
        }
    }
}

template<int NQ>
void apply_rz(cx_t sv[1 << NQ], int wire, angle_t c, angle_t s) {
    const int DIM = 1 << NQ;
    const int bit = 1 << wire;
    
    #pragma hls_unroll yes
    for (int i = 0; i < DIM; ++i) {
        cx_t a = sv[i];
        cx_t ap;
        
        if ((i & bit) == 0) {
            ap.re = c * a.re + s * a.im;
            ap.im = -s * a.re + c * a.im;
        } else {
            ap.re = c * a.re - s * a.im;
            ap.im = s * a.re + c * a.im;
        }
        sv[i] = ap;
    }
}

template<int NQ>
void apply_cnot(cx_t sv[1 << NQ], int control, int target) {
    const int DIM = 1 << NQ;
    const int cb = 1 << control;
    const int tb = 1 << target;
    
    #pragma hls_unroll yes
    for (int i = 0; i < DIM; ++i) {
        if ((i & cb) && ((i & tb) == 0)) {
            int j = i | tb;
            cx_t tmp = sv[i];
            sv[i] = sv[j];
            sv[j] = tmp;
        }
    }
}

// ========================== Measurements ==========================

template<int NQ>
acc_t measure_expZ(const cx_t sv[1 << NQ], int wire) {
    const int DIM = 1 << NQ;
    const int bit = 1 << wire;
    acc_t acc = 0;
    
    #pragma hls_unroll yes
    for (int i = 0; i < DIM; ++i) {
        acc_t p = sv[i].re * sv[i].re + sv[i].im * sv[i].im;
        if (i & bit) {
            acc -= p;
        } else {
            acc += p;
        }
    }
    return acc;
}

template<int NQ>
acc_t measure_expX(const cx_t sv[1 << NQ], int wire) {
    const int DIM = 1 << NQ;
    const int stride = 1 << wire;
    acc_t acc = 0;
    
    #pragma hls_unroll yes
    for (int base = 0; base < DIM; base += (stride << 1)) {
        #pragma hls_unroll yes
        for (int j = 0; j < stride; ++j) {
            int i0 = base + j;
            int i1 = i0 + stride;
            acc += acc_t(2.0) * (sv[i0].re * sv[i1].re + sv[i0].im * sv[i1].im);
        }
    }
    return acc;
}

template<int NQ>
acc_t measure_expY(const cx_t sv[1 << NQ], int wire) {
    const int DIM = 1 << NQ;
    const int stride = 1 << wire;
    acc_t acc = 0;
    
    #pragma hls_unroll yes
    for (int base = 0; base < DIM; base += (stride << 1)) {
        #pragma hls_unroll yes
        for (int j = 0; j < stride; ++j) {
            int i0 = base + j;
            int i1 = i0 + stride;
            acc += acc_t(2.0) * (sv[i0].re * sv[i1].im - sv[i0].im * sv[i1].re);
        }
    }
    return acc;
}

// ========================== Block circuits ==========================

// MET block: 1 qubit -> 3 outputs (Z, X, Y)
void run_met_block(input_t pt, input_t phi, output_t out[3]) {
    cx_t sv[2];
    
    sv[0].re = 1; sv[0].im = 0;
    sv[1].re = 0; sv[1].im = 0;
    
    angle_t c, s;
    get_cs_pt(pt, c, s);
    apply_ry<1>(sv, 0, c, s);
    
    get_cs_eta_phi(phi, c, s);
    apply_rz<1>(sv, 0, c, s);
    
    for (int d = 0; d < MET_DEPTH; ++d) {
        int base = d * 3;
        apply_rx<1>(sv, 0, met_weights_c[base], met_weights_s[base]);
        apply_ry<1>(sv, 0, met_weights_c[base + 1], met_weights_s[base + 1]);
        apply_rz<1>(sv, 0, met_weights_c[base + 2], met_weights_s[base + 2]);
    }
    
    out[0] = output_t(measure_expZ<1>(sv, 0));
    out[1] = output_t(measure_expX<1>(sv, 0));
    out[2] = output_t(measure_expY<1>(sv, 0));
}

// Electron block: 4 qubits -> 8 outputs (4 Z + 4 X)
void run_ele_block(input_t pt[4], input_t eta[4], input_t phi[4], output_t out[8]) {
    cx_t sv[16];
    
    #pragma hls_unroll yes
    for (int i = 0; i < 16; ++i) {
        sv[i].re = (i == 0) ? amp_t(1) : amp_t(0);
        sv[i].im = 0;
    }
    
    for (int k = 0; k < 4; ++k) {
        angle_t c, s;
        get_cs_eta_phi(eta[k], c, s);
        apply_rx<4>(sv, k, c, s);
        get_cs_pt(pt[k], c, s);
        apply_ry<4>(sv, k, c, s);
        get_cs_eta_phi(phi[k], c, s);
        apply_rz<4>(sv, k, c, s);
    }
    
    for (int k = 0; k < 3; ++k) {
        apply_cnot<4>(sv, k, k + 1);
    }
    
    for (int d = 0; d < ELE_DEPTH; ++d) {
        int base = d * 8;
        for (int k = 0; k < 4; ++k) {
            apply_rx<4>(sv, k, ele_weights_c[base + k], ele_weights_s[base + k]);
            apply_ry<4>(sv, k, ele_weights_c[base + 4 + k], ele_weights_s[base + 4 + k]);
        }
        for (int k = 0; k < 3; ++k) {
            apply_cnot<4>(sv, k, k + 1);
        }
    }
    
    for (int k = 0; k < 4; ++k) {
        out[k] = output_t(measure_expZ<4>(sv, k));
    }
    for (int k = 0; k < 4; ++k) {
        out[4 + k] = output_t(measure_expX<4>(sv, k));
    }
}

// Muon block: 4 qubits -> 8 outputs (4 Z + 4 X)
void run_mu_block(input_t pt[4], input_t eta[4], input_t phi[4], output_t out[8]) {
    cx_t sv[16];
    
    #pragma hls_unroll yes
    for (int i = 0; i < 16; ++i) {
        sv[i].re = (i == 0) ? amp_t(1) : amp_t(0);
        sv[i].im = 0;
    }
    
    for (int k = 0; k < 4; ++k) {
        angle_t c, s;
        get_cs_eta_phi(eta[k], c, s);
        apply_rx<4>(sv, k, c, s);
        get_cs_pt(pt[k], c, s);
        apply_ry<4>(sv, k, c, s);
        get_cs_eta_phi(phi[k], c, s);
        apply_rz<4>(sv, k, c, s);
    }
    
    for (int k = 0; k < 3; ++k) {
        apply_cnot<4>(sv, k, k + 1);
    }
    
    for (int d = 0; d < MU_DEPTH; ++d) {
        int base = d * 8;
        for (int k = 0; k < 4; ++k) {
            apply_rx<4>(sv, k, mu_weights_c[base + k], mu_weights_s[base + k]);
            apply_ry<4>(sv, k, mu_weights_c[base + 4 + k], mu_weights_s[base + 4 + k]);
        }
        for (int k = 0; k < 3; ++k) {
            apply_cnot<4>(sv, k, k + 1);
        }
    }
    
    for (int k = 0; k < 4; ++k) {
        out[k] = output_t(measure_expZ<4>(sv, k));
    }
    for (int k = 0; k < 4; ++k) {
        out[4 + k] = output_t(measure_expX<4>(sv, k));
    }
}

// Jet block: 10 qubits -> 13 outputs (10 Z + 3 X)
void run_jet_block(input_t pt[10], input_t eta[10], input_t phi[10], output_t out[13]) {
    cx_t sv[1024];
    
    for (int i = 0; i < 1024; ++i) {
        #pragma hls_pipeline_init_interval 1
        sv[i].re = (i == 0) ? amp_t(1) : amp_t(0);
        sv[i].im = 0;
    }
    
    for (int k = 0; k < 10; ++k) {
        angle_t c, s;
        get_cs_eta_phi(eta[k], c, s);
        apply_rx<10>(sv, k, c, s);
        get_cs_pt(pt[k], c, s);
        apply_ry<10>(sv, k, c, s);
        get_cs_eta_phi(phi[k], c, s);
        apply_rz<10>(sv, k, c, s);
    }
    
    for (int k = 0; k < 9; ++k) {
        apply_cnot<10>(sv, k, k + 1);
    }
    
    for (int d = 0; d < JET_DEPTH; ++d) {
        int base = d * 13;
        for (int k = 0; k < 10; ++k) {
            apply_rx<10>(sv, k, jet_weights_c[base + k], jet_weights_s[base + k]);
        }
        for (int k = 0; k < 3; ++k) {
            apply_ry<10>(sv, k, jet_weights_c[base + 10 + k], jet_weights_s[base + 10 + k]);
        }
        for (int k = 0; k < 9; ++k) {
            apply_cnot<10>(sv, k, k + 1);
        }
    }
    
    for (int k = 0; k < 10; ++k) {
        out[k] = output_t(measure_expZ<10>(sv, k));
    }
    for (int k = 0; k < 3; ++k) {
        out[10 + k] = output_t(measure_expX<10>(sv, k));
    }
}

// ========================== Top-level function ==========================

#pragma hls_design top
void extended_vae_encoder_top(
    input_t features[56],
    output_t encoder_output[32]
) {
    // Extract features for each block
    input_t met_pt = features[0];
    input_t met_phi = features[1];
    
    input_t ele_pt[4], ele_eta[4], ele_phi[4];
    for (int i = 0; i < 4; ++i) {
        ele_pt[i] = features[2 + i];
        ele_eta[i] = features[6 + i];
        ele_phi[i] = features[10 + i];
    }
    
    input_t mu_pt[4], mu_eta[4], mu_phi[4];
    for (int i = 0; i < 4; ++i) {
        mu_pt[i] = features[14 + i];
        mu_eta[i] = features[18 + i];
        mu_phi[i] = features[22 + i];
    }
    
    input_t jet_pt[10], jet_eta[10], jet_phi[10];
    for (int i = 0; i < 10; ++i) {
        jet_pt[i] = features[26 + i];
        jet_eta[i] = features[36 + i];
        jet_phi[i] = features[46 + i];
    }
    
    // Run blocks
    output_t met_out[3];
    output_t ele_out[8];
    output_t mu_out[8];
    output_t jet_out[13];
    
    run_met_block(met_pt, met_phi, met_out);
    run_ele_block(ele_pt, ele_eta, ele_phi, ele_out);
    run_mu_block(mu_pt, mu_eta, mu_phi, mu_out);
    run_jet_block(jet_pt, jet_eta, jet_phi, jet_out);
    
    // Combine outputs
    for (int i = 0; i < 3; ++i) {
        encoder_output[i] = met_out[i];
    }
    for (int i = 0; i < 8; ++i) {
        encoder_output[3 + i] = ele_out[i];
    }
    for (int i = 0; i < 8; ++i) {
        encoder_output[11 + i] = mu_out[i];
    }
    for (int i = 0; i < 13; ++i) {
        encoder_output[19 + i] = jet_out[i];
    }
}
