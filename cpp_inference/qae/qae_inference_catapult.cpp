// qae_inference_catapult.cpp
//
// Fixed-point implementation for Siemens Catapult HLS.
// Uses ac_fixed types from ac_datatypes library.
//
// Bit widths:
// - Angles/trig values: 16 bits (2 integer, 14 fractional)
// - Statevector amplitudes: 18 bits (2 integer, 16 fractional)
// - Accumulator for measurements: 24 bits (4 integer, 20 fractional)
// - Scores: 20 bits (4 integer, 16 fractional)

#include <ac_fixed.h>
#include <ac_channel.h>

// ========================== Type definitions ==========================

// Fixed-point types for Catapult
// ac_fixed<W, I, signed, quantization, overflow>
typedef ac_fixed<16, 2, true, AC_RND, AC_SAT> angle_t;      // [-2, 2) with 14 frac bits
typedef ac_fixed<18, 2, true, AC_RND, AC_SAT> amp_t;        // Statevector amplitudes
typedef ac_fixed<24, 4, true, AC_RND, AC_SAT> acc_t;        // Accumulators
typedef ac_fixed<20, 4, true, AC_RND, AC_SAT> score_t;      // Block/anomaly scores
typedef ac_fixed<16, 8, true, AC_RND, AC_SAT> input_t;      // Input features

// Complex amplitude
struct cx_t {
    amp_t re;
    amp_t im;
};

// ========================== LUT configuration ==========================
#define TRIG_LUT_BITS 10
#define TRIG_LUT_SIZE (1 << TRIG_LUT_BITS)

// Include generated headers
#include "catapult_trig_luts.h"
#include "catapult_weights.h"
#include "catapult_mahalanobis.h"

// ========================== LUT indexing ==========================

inline ac_int<TRIG_LUT_BITS, false> lut_index_eta_phi(input_t theta) {
    const input_t lo = -3.14159265;
    const input_t hi = 3.14159265;
    input_t t = theta;
    if (t < lo) t = lo;
    if (t > hi) t = hi;
    input_t u = (t - lo) / (hi - lo);
    ac_int<TRIG_LUT_BITS, false> idx = (ac_int<TRIG_LUT_BITS+4, false>)(u * (TRIG_LUT_SIZE - 1));
    return idx;
}

inline ac_int<TRIG_LUT_BITS, false> lut_index_pt(input_t theta) {
    const input_t lo = 0.0;
    const input_t hi = 3.14159265;
    input_t t = theta;
    if (t < lo) t = lo;
    if (t > hi) t = hi;
    input_t u = (t - lo) / (hi - lo);
    ac_int<TRIG_LUT_BITS, false> idx = (ac_int<TRIG_LUT_BITS+4, false>)(u * (TRIG_LUT_SIZE - 1));
    return idx;
}

// ========================== Gate operations ==========================

// RX gate
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

// RY gate
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

// RZ gate
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

// CNOT gate
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

// ========================== Measurement ==========================

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

// ========================== Block circuits ==========================

// MET block: 1 qubit
score_t run_met_block(input_t pt, input_t phi) {
    cx_t sv[2];
    
    sv[0].re = 1; sv[0].im = 0;
    sv[1].re = 0; sv[1].im = 0;
    
    ac_int<TRIG_LUT_BITS, false> pt_idx = lut_index_pt(pt);
    apply_ry<1>(sv, 0, pt_lut_cos[pt_idx], pt_lut_sin[pt_idx]);
    
    ac_int<TRIG_LUT_BITS, false> phi_idx = lut_index_eta_phi(phi);
    apply_rz<1>(sv, 0, eta_phi_lut_cos[phi_idx], eta_phi_lut_sin[phi_idx]);
    
    apply_ry<1>(sv, 0, met_weights_c[0], met_weights_s[0]);
    
    acc_t z = measure_expZ<1>(sv, 0);
    score_t excitation = (score_t)0.5 * ((score_t)1.0 - (score_t)z);
    
    return excitation;
}

// Electron block: 4 qubits
score_t run_ele_block(input_t pt[4], input_t eta[4], input_t phi[4]) {
    cx_t sv[16];
    
    #pragma hls_unroll yes
    for (int i = 0; i < 16; ++i) {
        sv[i].re = (i == 0) ? (amp_t)1 : (amp_t)0;
        sv[i].im = 0;
    }
    
    for (int k = 0; k < 4; ++k) {
        ac_int<TRIG_LUT_BITS, false> eta_idx = lut_index_eta_phi(eta[k]);
        ac_int<TRIG_LUT_BITS, false> pt_idx = lut_index_pt(pt[k]);
        ac_int<TRIG_LUT_BITS, false> phi_idx = lut_index_eta_phi(phi[k]);
        
        apply_rx<4>(sv, k, eta_phi_lut_cos[eta_idx], eta_phi_lut_sin[eta_idx]);
        apply_ry<4>(sv, k, pt_lut_cos[pt_idx], pt_lut_sin[pt_idx]);
        apply_rz<4>(sv, k, eta_phi_lut_cos[phi_idx], eta_phi_lut_sin[phi_idx]);
    }
    
    // depth=1 ansatz
    // trash -> latent CNOTs
    for (int ti = 0; ti < 2; ++ti) {
        for (int li = 0; li < 2; ++li) {
            apply_cnot<4>(sv, 2 + ti, li);
        }
    }
    
    // Trainable RY
    for (int k = 0; k < 4; ++k) {
        apply_ry<4>(sv, k, ele_weights_c[k], ele_weights_s[k]);
    }
    
    // latent -> trash CNOTs
    for (int li = 0; li < 2; ++li) {
        for (int ti = 0; ti < 2; ++ti) {
            apply_cnot<4>(sv, li, 2 + ti);
        }
    }
    
    acc_t z2 = measure_expZ<4>(sv, 2);
    acc_t z3 = measure_expZ<4>(sv, 3);
    
    score_t exc2 = (score_t)0.5 * ((score_t)1.0 - (score_t)z2);
    score_t exc3 = (score_t)0.5 * ((score_t)1.0 - (score_t)z3);
    
    return (exc2 + exc3) * (score_t)0.5;
}

// Muon block: 4 qubits (same structure as electron)
score_t run_mu_block(input_t pt[4], input_t eta[4], input_t phi[4]) {
    cx_t sv[16];
    
    #pragma hls_unroll yes
    for (int i = 0; i < 16; ++i) {
        sv[i].re = (i == 0) ? (amp_t)1 : (amp_t)0;
        sv[i].im = 0;
    }
    
    for (int k = 0; k < 4; ++k) {
        ac_int<TRIG_LUT_BITS, false> eta_idx = lut_index_eta_phi(eta[k]);
        ac_int<TRIG_LUT_BITS, false> pt_idx = lut_index_pt(pt[k]);
        ac_int<TRIG_LUT_BITS, false> phi_idx = lut_index_eta_phi(phi[k]);
        
        apply_rx<4>(sv, k, eta_phi_lut_cos[eta_idx], eta_phi_lut_sin[eta_idx]);
        apply_ry<4>(sv, k, pt_lut_cos[pt_idx], pt_lut_sin[pt_idx]);
        apply_rz<4>(sv, k, eta_phi_lut_cos[phi_idx], eta_phi_lut_sin[phi_idx]);
    }
    
    for (int ti = 0; ti < 2; ++ti) {
        for (int li = 0; li < 2; ++li) {
            apply_cnot<4>(sv, 2 + ti, li);
        }
    }
    
    for (int k = 0; k < 4; ++k) {
        apply_ry<4>(sv, k, mu_weights_c[k], mu_weights_s[k]);
    }
    
    for (int li = 0; li < 2; ++li) {
        for (int ti = 0; ti < 2; ++ti) {
            apply_cnot<4>(sv, li, 2 + ti);
        }
    }
    
    acc_t z2 = measure_expZ<4>(sv, 2);
    acc_t z3 = measure_expZ<4>(sv, 3);
    
    score_t exc2 = (score_t)0.5 * ((score_t)1.0 - (score_t)z2);
    score_t exc3 = (score_t)0.5 * ((score_t)1.0 - (score_t)z3);
    
    return (exc2 + exc3) * (score_t)0.5;
}

// Jet block: 10 qubits, depth=4
score_t run_jet_block(input_t pt[10], input_t eta[10], input_t phi[10]) {
    cx_t sv[1024];
    
    for (int i = 0; i < 1024; ++i) {
        #pragma hls_pipeline_init_interval 1
        sv[i].re = (i == 0) ? (amp_t)1 : (amp_t)0;
        sv[i].im = 0;
    }
    
    // Feature encoding
    for (int k = 0; k < 10; ++k) {
        ac_int<TRIG_LUT_BITS, false> eta_idx = lut_index_eta_phi(eta[k]);
        ac_int<TRIG_LUT_BITS, false> pt_idx = lut_index_pt(pt[k]);
        ac_int<TRIG_LUT_BITS, false> phi_idx = lut_index_eta_phi(phi[k]);
        
        apply_rx<10>(sv, k, eta_phi_lut_cos[eta_idx], eta_phi_lut_sin[eta_idx]);
        apply_ry<10>(sv, k, pt_lut_cos[pt_idx], pt_lut_sin[pt_idx]);
        apply_rz<10>(sv, k, eta_phi_lut_cos[phi_idx], eta_phi_lut_sin[phi_idx]);
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
    
    acc_t z6 = measure_expZ<10>(sv, 6);
    acc_t z7 = measure_expZ<10>(sv, 7);
    acc_t z8 = measure_expZ<10>(sv, 8);
    acc_t z9 = measure_expZ<10>(sv, 9);
    
    score_t exc6 = (score_t)0.5 * ((score_t)1.0 - (score_t)z6);
    score_t exc7 = (score_t)0.5 * ((score_t)1.0 - (score_t)z7);
    score_t exc8 = (score_t)0.5 * ((score_t)1.0 - (score_t)z8);
    score_t exc9 = (score_t)0.5 * ((score_t)1.0 - (score_t)z9);
    
    return (exc6 + exc7 + exc8 + exc9) * (score_t)0.25;
}

// ========================== Mahalanobis scoring ==========================

score_t mahalanobis_score(score_t s[4]) {
    score_t d[4];
    
    #pragma hls_unroll yes
    for (int i = 0; i < 4; ++i) {
        d[i] = s[i] - MAHA_MU[i];
    }
    
    score_t q = 0;
    #pragma hls_unroll yes
    for (int i = 0; i < 4; ++i) {
        score_t row = 0;
        #pragma hls_unroll yes
        for (int j = 0; j < 4; ++j) {
            row += MAHA_PREC[i][j] * d[j];
        }
        q += d[i] * row;
    }
    
    return q * (score_t)0.5;
}

// ========================== Top-level function ==========================

#pragma hls_design top
void qae_inference_top(
    input_t features[56],
    score_t &anomaly_score
) {
    // Extract features for each block
    input_t met_pt = features[0];
    input_t met_phi = features[1];
    
    input_t ele_pt[4], ele_eta[4], ele_phi[4];
    input_t mu_pt[4], mu_eta[4], mu_phi[4];
    input_t jet_pt[10], jet_eta[10], jet_phi[10];
    
    #pragma hls_unroll yes
    for (int k = 0; k < 4; ++k) {
        ele_pt[k] = features[2 + k];
        ele_eta[k] = features[6 + k];
        ele_phi[k] = features[10 + k];
        
        mu_pt[k] = features[14 + k];
        mu_eta[k] = features[18 + k];
        mu_phi[k] = features[22 + k];
    }
    
    #pragma hls_unroll yes
    for (int k = 0; k < 10; ++k) {
        jet_pt[k] = features[26 + k];
        jet_eta[k] = features[36 + k];
        jet_phi[k] = features[46 + k];
    }
    
    // Run blocks
    score_t met_score = run_met_block(met_pt, met_phi);
    score_t ele_score = run_ele_block(ele_pt, ele_eta, ele_phi);
    score_t mu_score = run_mu_block(mu_pt, mu_eta, mu_phi);
    score_t jet_score = run_jet_block(jet_pt, jet_eta, jet_phi);
    
    // Compute Mahalanobis score
    score_t block_scores[4] = {met_score, ele_score, mu_score, jet_score};
    anomaly_score = mahalanobis_score(block_scores);
}
