// qae_inference_vitis.cpp
//
// Fixed-point implementation for Xilinx Vitis HLS.
// Uses ap_fixed types from Vitis HLS library.
//
// Input normalization (linear to [0,1]):
//   - pt:  pt / 1200
//   - eta: (eta + 5) / 10
//   - phi: (phi + pi) / (2*pi)
//
// Bit widths:
// - Angles/trig values: 16 bits (2 integer, 14 fractional)
// - Statevector amplitudes: 18 bits (2 integer, 16 fractional)
// - Accumulator for measurements: 24 bits (4 integer, 20 fractional)
// - Scores: 20 bits (4 integer, 16 fractional)

#include <ap_fixed.h>
#include <hls_math.h>

// ========================== Type definitions ==========================

// Fixed-point types for Vitis HLS
// ap_fixed<W, I, Q, O> where W=total bits, I=integer bits, Q=quantization, O=overflow
typedef ap_fixed<16, 2, AP_RND, AP_SAT> angle_t;      // [-2, 2) with 14 frac bits
typedef ap_fixed<18, 2, AP_RND, AP_SAT> amp_t;        // Statevector amplitudes
typedef ap_fixed<24, 4, AP_RND, AP_SAT> acc_t;        // Accumulators
typedef ap_fixed<20, 4, AP_RND, AP_SAT> score_t;      // Block/anomaly scores
typedef ap_fixed<16, 12, AP_RND, AP_SAT> input_t;     // Input features (raw physics values)

// Complex amplitude
struct cx_t {
    amp_t re;
    amp_t im;
};

// ========================== Constants ==========================
#define PI_VAL 3.14159265358979323846

// ========================== Input normalization ==========================
// QAE was trained with linear normalization to [0,1]

inline input_t normalize_pt(input_t pt) {
    #pragma HLS INLINE
    return pt / input_t(1200.0);
}

inline input_t normalize_eta(input_t eta) {
    #pragma HLS INLINE
    return (eta + input_t(5.0)) / input_t(10.0);
}

inline input_t normalize_phi(input_t phi) {
    #pragma HLS INLINE
    const input_t PI = PI_VAL;
    return (phi + PI) / (input_t(2.0) * PI);
}

// ========================== Trig LUT ==========================
// LUT maps normalized value [0,1] to cos(theta/2) and sin(theta/2)

#define TRIG_LUT_BITS 10
#define TRIG_LUT_SIZE (1 << TRIG_LUT_BITS)

// Include generated LUT headers
#include "vitis_trig_luts.h"
#include "vitis_weights.h"
#include "vitis_mahalanobis.h"

inline void get_cs(input_t norm_val, angle_t& c, angle_t& s) {
    #pragma HLS INLINE
    // Clamp to [0, 1]
    input_t clamped = norm_val;
    if (clamped < input_t(0.0)) clamped = input_t(0.0);
    if (clamped > input_t(1.0)) clamped = input_t(1.0);
    
    // Map to LUT index
    ap_uint<TRIG_LUT_BITS> idx = (ap_uint<TRIG_LUT_BITS+4>)(clamped * (TRIG_LUT_SIZE - 1));
    
    c = trig_lut_cos[idx];
    s = trig_lut_sin[idx];
}

// ========================== Gate operations ==========================

// RX gate: exp(-i * theta/2 * X)
template<int NQ>
void apply_rx(cx_t sv[1 << NQ], int wire, angle_t c, angle_t s) {
    #pragma HLS INLINE off
    const int stride = 1 << wire;
    const int DIM = 1 << NQ;
    
    for (int base = 0; base < DIM; base += (stride << 1)) {
        #pragma HLS PIPELINE II=1
        for (int j = 0; j < stride; ++j) {
            #pragma HLS UNROLL factor=4
            int i0 = base + j;
            int i1 = i0 + stride;
            
            cx_t a = sv[i0];
            cx_t b = sv[i1];
            
            cx_t ap, bp;
            // RX: [[c, -is], [-is, c]]
            ap.re = c * a.re + s * b.im;
            ap.im = c * a.im - s * b.re;
            bp.re = s * a.im + c * b.re;
            bp.im = -s * a.re + c * b.im;
            
            sv[i0] = ap;
            sv[i1] = bp;
        }
    }
}

// RY gate: exp(-i * theta/2 * Y)
template<int NQ>
void apply_ry(cx_t sv[1 << NQ], int wire, angle_t c, angle_t s) {
    #pragma HLS INLINE off
    const int stride = 1 << wire;
    const int DIM = 1 << NQ;
    
    for (int base = 0; base < DIM; base += (stride << 1)) {
        #pragma HLS PIPELINE II=1
        for (int j = 0; j < stride; ++j) {
            #pragma HLS UNROLL factor=4
            int i0 = base + j;
            int i1 = i0 + stride;
            
            cx_t a = sv[i0];
            cx_t b = sv[i1];
            
            cx_t ap, bp;
            // RY: [[c, -s], [s, c]]
            ap.re = c * a.re - s * b.re;
            ap.im = c * a.im - s * b.im;
            bp.re = s * a.re + c * b.re;
            bp.im = s * a.im + c * b.im;
            
            sv[i0] = ap;
            sv[i1] = bp;
        }
    }
}

// RZ gate: exp(-i * theta/2 * Z)
template<int NQ>
void apply_rz(cx_t sv[1 << NQ], int wire, angle_t c, angle_t s) {
    #pragma HLS INLINE off
    const int DIM = 1 << NQ;
    const int bit = 1 << wire;
    
    for (int i = 0; i < DIM; ++i) {
        #pragma HLS PIPELINE II=1
        cx_t a = sv[i];
        cx_t ap;
        
        if ((i & bit) == 0) {
            // e^{-i*theta/2}: multiply by (c - is)
            ap.re = c * a.re + s * a.im;
            ap.im = -s * a.re + c * a.im;
        } else {
            // e^{+i*theta/2}: multiply by (c + is)
            ap.re = c * a.re - s * a.im;
            ap.im = s * a.re + c * a.im;
        }
        sv[i] = ap;
    }
}

// CNOT gate
template<int NQ>
void apply_cnot(cx_t sv[1 << NQ], int control, int target) {
    #pragma HLS INLINE off
    const int DIM = 1 << NQ;
    const int cb = 1 << control;
    const int tb = 1 << target;
    
    for (int i = 0; i < DIM; ++i) {
        #pragma HLS PIPELINE II=1
        if ((i & cb) && ((i & tb) == 0)) {
            int j = i | tb;
            cx_t tmp = sv[i];
            sv[i] = sv[j];
            sv[j] = tmp;
        }
    }
}

// ========================== Measurement ==========================

template<int NQ>
acc_t measure_expZ(const cx_t sv[1 << NQ], int wire) {
    #pragma HLS INLINE off
    const int DIM = 1 << NQ;
    const int bit = 1 << wire;
    acc_t acc = 0;
    
    for (int i = 0; i < DIM; ++i) {
        #pragma HLS PIPELINE II=1
        acc_t p = sv[i].re * sv[i].re + sv[i].im * sv[i].im;
        if (i & bit) {
            acc -= p;
        } else {
            acc += p;
        }
    }
    return acc;
}

// ========================== Block circuits ==========================

// MET block: 1 qubit
// Input: raw (unnormalized) pt and phi
score_t run_met_block(input_t pt_raw, input_t phi_raw) {
    #pragma HLS INLINE off
    cx_t sv[2];
    #pragma HLS ARRAY_PARTITION variable=sv complete
    
    sv[0].re = 1; sv[0].im = 0;
    sv[1].re = 0; sv[1].im = 0;
    
    // Normalize inputs
    input_t pt_norm = normalize_pt(pt_raw);
    input_t phi_norm = normalize_phi(phi_raw);
    
    // Apply gates
    angle_t c, s;
    get_cs(pt_norm, c, s);
    apply_ry<1>(sv, 0, c, s);
    
    get_cs(phi_norm, c, s);
    apply_rz<1>(sv, 0, c, s);
    
    // Trainable layer
    apply_ry<1>(sv, 0, met_weights_c[0], met_weights_s[0]);
    
    // Measure
    acc_t z = measure_expZ<1>(sv, 0);
    score_t excitation = score_t(0.5) * (score_t(1.0) - score_t(z));
    
    return excitation;
}

// Electron block: 4 qubits
// Input: raw (unnormalized) pt, eta, phi arrays
score_t run_ele_block(input_t pt_raw[4], input_t eta_raw[4], input_t phi_raw[4]) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=pt_raw complete
    #pragma HLS ARRAY_PARTITION variable=eta_raw complete
    #pragma HLS ARRAY_PARTITION variable=phi_raw complete
    
    cx_t sv[16];
    #pragma HLS ARRAY_PARTITION variable=sv complete
    
    // Initialize |0...0>
    for (int i = 0; i < 16; ++i) {
        #pragma HLS UNROLL
        sv[i].re = (i == 0) ? amp_t(1) : amp_t(0);
        sv[i].im = 0;
    }
    
    // Feature encoding
    for (int k = 0; k < 4; ++k) {
        #pragma HLS UNROLL
        input_t pt_norm = normalize_pt(pt_raw[k]);
        input_t eta_norm = normalize_eta(eta_raw[k]);
        input_t phi_norm = normalize_phi(phi_raw[k]);
        
        angle_t c, s;
        get_cs(eta_norm, c, s);
        apply_rx<4>(sv, k, c, s);
        
        get_cs(pt_norm, c, s);
        apply_ry<4>(sv, k, c, s);
        
        get_cs(phi_norm, c, s);
        apply_rz<4>(sv, k, c, s);
    }
    
    // Ansatz: trash -> latent CNOTs
    for (int ti = 0; ti < 2; ++ti) {
        for (int li = 0; li < 2; ++li) {
            apply_cnot<4>(sv, 2 + ti, li);
        }
    }
    
    // Trainable RY
    for (int k = 0; k < 4; ++k) {
        #pragma HLS UNROLL
        apply_ry<4>(sv, k, ele_weights_c[k], ele_weights_s[k]);
    }
    
    // latent -> trash CNOTs
    for (int li = 0; li < 2; ++li) {
        for (int ti = 0; ti < 2; ++ti) {
            apply_cnot<4>(sv, li, 2 + ti);
        }
    }
    
    // Measure trash qubits
    acc_t z2 = measure_expZ<4>(sv, 2);
    acc_t z3 = measure_expZ<4>(sv, 3);
    
    score_t exc2 = score_t(0.5) * (score_t(1.0) - score_t(z2));
    score_t exc3 = score_t(0.5) * (score_t(1.0) - score_t(z3));
    
    return (exc2 + exc3) * score_t(0.5);
}

// Muon block: 4 qubits (same structure as electron)
score_t run_mu_block(input_t pt_raw[4], input_t eta_raw[4], input_t phi_raw[4]) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=pt_raw complete
    #pragma HLS ARRAY_PARTITION variable=eta_raw complete
    #pragma HLS ARRAY_PARTITION variable=phi_raw complete
    
    cx_t sv[16];
    #pragma HLS ARRAY_PARTITION variable=sv complete
    
    for (int i = 0; i < 16; ++i) {
        #pragma HLS UNROLL
        sv[i].re = (i == 0) ? amp_t(1) : amp_t(0);
        sv[i].im = 0;
    }
    
    for (int k = 0; k < 4; ++k) {
        #pragma HLS UNROLL
        input_t pt_norm = normalize_pt(pt_raw[k]);
        input_t eta_norm = normalize_eta(eta_raw[k]);
        input_t phi_norm = normalize_phi(phi_raw[k]);
        
        angle_t c, s;
        get_cs(eta_norm, c, s);
        apply_rx<4>(sv, k, c, s);
        
        get_cs(pt_norm, c, s);
        apply_ry<4>(sv, k, c, s);
        
        get_cs(phi_norm, c, s);
        apply_rz<4>(sv, k, c, s);
    }
    
    for (int ti = 0; ti < 2; ++ti) {
        for (int li = 0; li < 2; ++li) {
            apply_cnot<4>(sv, 2 + ti, li);
        }
    }
    
    for (int k = 0; k < 4; ++k) {
        #pragma HLS UNROLL
        apply_ry<4>(sv, k, mu_weights_c[k], mu_weights_s[k]);
    }
    
    for (int li = 0; li < 2; ++li) {
        for (int ti = 0; ti < 2; ++ti) {
            apply_cnot<4>(sv, li, 2 + ti);
        }
    }
    
    acc_t z2 = measure_expZ<4>(sv, 2);
    acc_t z3 = measure_expZ<4>(sv, 3);
    
    score_t exc2 = score_t(0.5) * (score_t(1.0) - score_t(z2));
    score_t exc3 = score_t(0.5) * (score_t(1.0) - score_t(z3));
    
    return (exc2 + exc3) * score_t(0.5);
}

// Jet block: 10 qubits, depth=4
score_t run_jet_block(input_t pt_raw[10], input_t eta_raw[10], input_t phi_raw[10]) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=pt_raw complete
    #pragma HLS ARRAY_PARTITION variable=eta_raw complete
    #pragma HLS ARRAY_PARTITION variable=phi_raw complete
    
    cx_t sv[1024];
    #pragma HLS ARRAY_PARTITION variable=sv cyclic factor=16
    
    // Initialize
    for (int i = 0; i < 1024; ++i) {
        #pragma HLS PIPELINE II=1
        sv[i].re = (i == 0) ? amp_t(1) : amp_t(0);
        sv[i].im = 0;
    }
    
    // Feature encoding
    for (int k = 0; k < 10; ++k) {
        input_t pt_norm = normalize_pt(pt_raw[k]);
        input_t eta_norm = normalize_eta(eta_raw[k]);
        input_t phi_norm = normalize_phi(phi_raw[k]);
        
        angle_t c, s;
        get_cs(eta_norm, c, s);
        apply_rx<10>(sv, k, c, s);
        
        get_cs(pt_norm, c, s);
        apply_ry<10>(sv, k, c, s);
        
        get_cs(phi_norm, c, s);
        apply_rz<10>(sv, k, c, s);
    }
    
    // Ansatz: depth=4
    for (int d = 0; d < 4; ++d) {
        // trash -> latent CNOTs
        for (int ti = 0; ti < 4; ++ti) {
            for (int li = 0; li < 6; ++li) {
                apply_cnot<10>(sv, 6 + ti, li);
            }
        }
        
        // Trainable RY
        int base = d * 10;
        for (int k = 0; k < 10; ++k) {
            apply_ry<10>(sv, k, jet_weights_c[base + k], jet_weights_s[base + k]);
        }
        
        // latent -> trash CNOTs
        for (int li = 0; li < 6; ++li) {
            for (int ti = 0; ti < 4; ++ti) {
                apply_cnot<10>(sv, li, 6 + ti);
            }
        }
    }
    
    // Measure trash qubits
    acc_t z6 = measure_expZ<10>(sv, 6);
    acc_t z7 = measure_expZ<10>(sv, 7);
    acc_t z8 = measure_expZ<10>(sv, 8);
    acc_t z9 = measure_expZ<10>(sv, 9);
    
    score_t exc6 = score_t(0.5) * (score_t(1.0) - score_t(z6));
    score_t exc7 = score_t(0.5) * (score_t(1.0) - score_t(z7));
    score_t exc8 = score_t(0.5) * (score_t(1.0) - score_t(z8));
    score_t exc9 = score_t(0.5) * (score_t(1.0) - score_t(z9));
    
    return (exc6 + exc7 + exc8 + exc9) * score_t(0.25);
}

// ========================== Mahalanobis scoring ==========================

score_t mahalanobis_score(score_t s[4]) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=s complete
    
    score_t d[4];
    #pragma HLS ARRAY_PARTITION variable=d complete
    
    for (int i = 0; i < 4; ++i) {
        #pragma HLS UNROLL
        d[i] = s[i] - MAHA_MU[i];
    }
    
    score_t q = 0;
    for (int i = 0; i < 4; ++i) {
        #pragma HLS UNROLL
        score_t row = 0;
        for (int j = 0; j < 4; ++j) {
            #pragma HLS UNROLL
            row += MAHA_PREC[i][j] * d[j];
        }
        q += d[i] * row;
    }
    
    return score_t(0.5) * q;
}

// ========================== Top-level function ==========================

// Top function for HLS synthesis
// Input: 56 raw (unnormalized) features
// Output: Mahalanobis anomaly score
void qae_inference_top(
    input_t features[56],
    score_t* anomaly_score
) {
    #pragma HLS INTERFACE mode=s_axilite port=return
    #pragma HLS INTERFACE mode=s_axilite port=features
    #pragma HLS INTERFACE mode=s_axilite port=anomaly_score
    
    #pragma HLS ARRAY_PARTITION variable=features complete
    
    // Extract features for each block
    // MET: pt=0, phi=1
    input_t met_pt = features[0];
    input_t met_phi = features[1];
    
    // Electrons: pt[2:6], eta[6:10], phi[10:14]
    input_t ele_pt[4], ele_eta[4], ele_phi[4];
    #pragma HLS ARRAY_PARTITION variable=ele_pt complete
    #pragma HLS ARRAY_PARTITION variable=ele_eta complete
    #pragma HLS ARRAY_PARTITION variable=ele_phi complete
    for (int i = 0; i < 4; ++i) {
        #pragma HLS UNROLL
        ele_pt[i] = features[2 + i];
        ele_eta[i] = features[6 + i];
        ele_phi[i] = features[10 + i];
    }
    
    // Muons: pt[14:18], eta[18:22], phi[22:26]
    input_t mu_pt[4], mu_eta[4], mu_phi[4];
    #pragma HLS ARRAY_PARTITION variable=mu_pt complete
    #pragma HLS ARRAY_PARTITION variable=mu_eta complete
    #pragma HLS ARRAY_PARTITION variable=mu_phi complete
    for (int i = 0; i < 4; ++i) {
        #pragma HLS UNROLL
        mu_pt[i] = features[14 + i];
        mu_eta[i] = features[18 + i];
        mu_phi[i] = features[22 + i];
    }
    
    // Jets: pt[26:36], eta[36:46], phi[46:56]
    input_t jet_pt[10], jet_eta[10], jet_phi[10];
    #pragma HLS ARRAY_PARTITION variable=jet_pt complete
    #pragma HLS ARRAY_PARTITION variable=jet_eta complete
    #pragma HLS ARRAY_PARTITION variable=jet_phi complete
    for (int i = 0; i < 10; ++i) {
        #pragma HLS UNROLL
        jet_pt[i] = features[26 + i];
        jet_eta[i] = features[36 + i];
        jet_phi[i] = features[46 + i];
    }
    
    // Run blocks (can be parallelized with DATAFLOW)
    #pragma HLS DATAFLOW
    
    score_t s_met = run_met_block(met_pt, met_phi);
    score_t s_ele = run_ele_block(ele_pt, ele_eta, ele_phi);
    score_t s_mu = run_mu_block(mu_pt, mu_eta, mu_phi);
    score_t s_jet = run_jet_block(jet_pt, jet_eta, jet_phi);
    
    // Compute Mahalanobis score
    score_t block_scores[4] = {s_met, s_ele, s_mu, s_jet};
    #pragma HLS ARRAY_PARTITION variable=block_scores complete
    
    *anomaly_score = mahalanobis_score(block_scores);
}
