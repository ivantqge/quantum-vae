// extended_vae_inference_ref.cpp
//
// C++ reference implementation for extended_quantum_vae.py encoder inference.
// This implements the quantum encoder portion that maps 56 features -> 32 outputs.
//
// Architecture:
// - MET: 1 qubit -> PauliZ, PauliX, PauliY -> 3 outputs
// - Electrons: 4 qubits -> PauliZ (4) + PauliX (4) -> 8 outputs
// - Muons: 4 qubits -> PauliZ (4) + PauliX (4) -> 8 outputs
// - Jets: 10 qubits -> PauliZ (10) + PauliX (3) -> 13 outputs
// Total: 32 outputs
//
// Encoding: RX(eta), RY(pt), RZ(phi) per qubit
// Entangling: nearest-neighbor CNOT chain
// Trainable: RX, RY rotations per layer
//
// Requires generated headers:
//   - extended_vae_weights.h
//   - trig_luts.h

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iomanip>
#include <chrono>

#include "extended_vae_weights.h"
#include "trig_luts.h"

// ========================== Complex + Statevector ==========================

struct Cx {
    double re;
    double im;
};

inline Cx cx_add(const Cx& a, const Cx& b) { return {a.re + b.re, a.im + b.im}; }
inline Cx cx_sub(const Cx& a, const Cx& b) { return {a.re - b.re, a.im - b.im}; }
inline Cx cx_mul_real(const Cx& a, double x) { return {a.re * x, a.im * x}; }
inline Cx cx_mul_minus_i(const Cx& a, double s) { return {s * a.im, -s * a.re}; }
inline Cx cx_mul_phase_pos(const Cx& a, double c, double s) {
    return {a.re * c - a.im * s, a.re * s + a.im * c};
}
inline Cx cx_mul_phase_neg(const Cx& a, double c, double s) {
    return {a.re * c + a.im * s, -a.re * s + a.im * c};
}

template <int NQ>
struct StateVec {
    static constexpr int DIM = 1 << NQ;
    std::array<Cx, DIM> a{};

    void init_zero() {
        for (int i = 0; i < DIM; ++i) a[i] = {0.0, 0.0};
        a[0] = {1.0, 0.0};
    }
};

// ========================== LUT Helpers ==========================

inline int clamp_idx(int idx, int n) {
    if (idx < 0) return 0;
    if (idx >= n) return n - 1;
    return idx;
}

inline int lut_index_eta_phi(double theta) {
    const double lo = -M_PI, hi = M_PI;
    double t = std::max(lo, std::min(hi, theta));
    double u = (t - lo) / (hi - lo);
    int idx = static_cast<int>(std::llround(u * (TRIG_LUT_SIZE - 1)));
    return clamp_idx(idx, TRIG_LUT_SIZE);
}

inline int lut_index_pt(double theta) {
    const double lo = 0.0, hi = M_PI;
    double t = std::max(lo, std::min(hi, theta));
    double u = (t - lo) / (hi - lo);
    int idx = static_cast<int>(std::llround(u * (TRIG_LUT_SIZE - 1)));
    return clamp_idx(idx, TRIG_LUT_SIZE);
}

inline void get_cs_eta_phi(double theta, double& c, double& s) {
    int idx = lut_index_eta_phi(theta);
    c = eta_phi_lut_cos[idx];
    s = eta_phi_lut_sin[idx];
}

inline void get_cs_pt(double theta, double& c, double& s) {
    int idx = lut_index_pt(theta);
    c = pt_lut_cos[idx];
    s = pt_lut_sin[idx];
}

// ========================== Gate Kernels ==========================

template <int NQ>
void apply_rx(StateVec<NQ>& sv, int wire, double c, double s) {
    const int stride = 1 << wire;
    constexpr int DIM = StateVec<NQ>::DIM;
    for (int base = 0; base < DIM; base += (stride << 1)) {
        for (int j = 0; j < stride; ++j) {
            int i0 = base + j, i1 = i0 + stride;
            Cx a = sv.a[i0], b = sv.a[i1];
            sv.a[i0] = cx_add(cx_mul_real(a, c), cx_mul_minus_i(b, s));
            sv.a[i1] = cx_add(cx_mul_minus_i(a, s), cx_mul_real(b, c));
        }
    }
}

template <int NQ>
void apply_ry(StateVec<NQ>& sv, int wire, double c, double s) {
    const int stride = 1 << wire;
    constexpr int DIM = StateVec<NQ>::DIM;
    for (int base = 0; base < DIM; base += (stride << 1)) {
        for (int j = 0; j < stride; ++j) {
            int i0 = base + j, i1 = i0 + stride;
            Cx a = sv.a[i0], b = sv.a[i1];
            sv.a[i0] = cx_sub(cx_mul_real(a, c), cx_mul_real(b, s));
            sv.a[i1] = cx_add(cx_mul_real(a, s), cx_mul_real(b, c));
        }
    }
}

template <int NQ>
void apply_rz(StateVec<NQ>& sv, int wire, double c, double s) {
    constexpr int DIM = StateVec<NQ>::DIM;
    const int bit = 1 << wire;
    for (int i = 0; i < DIM; ++i) {
        if ((i & bit) == 0) {
            sv.a[i] = cx_mul_phase_neg(sv.a[i], c, s);
        } else {
            sv.a[i] = cx_mul_phase_pos(sv.a[i], c, s);
        }
    }
}

template <int NQ>
void apply_cnot(StateVec<NQ>& sv, int control, int target) {
    constexpr int DIM = StateVec<NQ>::DIM;
    const int cb = 1 << control, tb = 1 << target;
    for (int i = 0; i < DIM; ++i) {
        if ((i & cb) && ((i & tb) == 0)) {
            int j = i | tb;
            std::swap(sv.a[i], sv.a[j]);
        }
    }
}

// ========================== Measurements ==========================

template <int NQ>
double measure_expZ(const StateVec<NQ>& sv, int wire) {
    constexpr int DIM = StateVec<NQ>::DIM;
    const int bit = 1 << wire;
    double acc = 0.0;
    for (int i = 0; i < DIM; ++i) {
        double p = sv.a[i].re * sv.a[i].re + sv.a[i].im * sv.a[i].im;
        acc += ((i & bit) ? -p : +p);
    }
    return acc;
}

template <int NQ>
double measure_expX(const StateVec<NQ>& sv, int wire) {
    constexpr int DIM = StateVec<NQ>::DIM;
    const int stride = 1 << wire;
    double acc = 0.0;
    for (int base = 0; base < DIM; base += (stride << 1)) {
        for (int j = 0; j < stride; ++j) {
            int i0 = base + j, i1 = i0 + stride;
            acc += 2.0 * (sv.a[i0].re * sv.a[i1].re + sv.a[i0].im * sv.a[i1].im);
        }
    }
    return acc;
}

template <int NQ>
double measure_expY(const StateVec<NQ>& sv, int wire) {
    constexpr int DIM = StateVec<NQ>::DIM;
    const int stride = 1 << wire;
    double acc = 0.0;
    for (int base = 0; base < DIM; base += (stride << 1)) {
        for (int j = 0; j < stride; ++j) {
            int i0 = base + j, i1 = i0 + stride;
            // <Y> = 2 * Im(a* b) = 2 * (a.re * b.im - a.im * b.re)
            acc += 2.0 * (sv.a[i0].re * sv.a[i1].im - sv.a[i0].im * sv.a[i1].re);
        }
    }
    return acc;
}

// ========================== Block Circuits ==========================

// MET block: 1 qubit -> 3 outputs (Z, X, Y)
std::array<double, 3> run_met_block(double pt, double phi) {
    StateVec<1> sv;
    sv.init_zero();

    double c, s;
    get_cs_pt(pt, c, s);
    apply_ry(sv, 0, c, s);
    
    get_cs_eta_phi(phi, c, s);
    apply_rz(sv, 0, c, s);

    // Trainable layers: RX, RY, RZ per depth
    for (int d = 0; d < MET_DEPTH; ++d) {
        int base = d * 3;
        apply_rx(sv, 0, met_weights_c[base], met_weights_s[base]);
        apply_ry(sv, 0, met_weights_c[base + 1], met_weights_s[base + 1]);
        apply_rz(sv, 0, met_weights_c[base + 2], met_weights_s[base + 2]);
    }

    return {measure_expZ(sv, 0), measure_expX(sv, 0), measure_expY(sv, 0)};
}

// Electron block: 4 qubits -> 8 outputs (4 Z + 4 X)
std::array<double, 8> run_ele_block(const double pt[4], const double eta[4], const double phi[4]) {
    StateVec<4> sv;
    sv.init_zero();

    // Feature encoding
    for (int k = 0; k < 4; ++k) {
        double c, s;
        get_cs_eta_phi(eta[k], c, s); apply_rx(sv, k, c, s);
        get_cs_pt(pt[k], c, s); apply_ry(sv, k, c, s);
        get_cs_eta_phi(phi[k], c, s); apply_rz(sv, k, c, s);
    }

    // Initial CNOT chain
    for (int k = 0; k < 3; ++k) {
        apply_cnot(sv, k, k + 1);
    }

    // Trainable layers
    for (int d = 0; d < ELE_DEPTH; ++d) {
        int base = d * 8;
        for (int k = 0; k < 4; ++k) {
            apply_rx(sv, k, ele_weights_c[base + k], ele_weights_s[base + k]);
            apply_ry(sv, k, ele_weights_c[base + 4 + k], ele_weights_s[base + 4 + k]);
        }
        for (int k = 0; k < 3; ++k) {
            apply_cnot(sv, k, k + 1);
        }
    }

    std::array<double, 8> out;
    for (int k = 0; k < 4; ++k) out[k] = measure_expZ(sv, k);
    for (int k = 0; k < 4; ++k) out[4 + k] = measure_expX(sv, k);
    return out;
}

// Muon block: 4 qubits -> 8 outputs (4 Z + 4 X)
std::array<double, 8> run_mu_block(const double pt[4], const double eta[4], const double phi[4]) {
    StateVec<4> sv;
    sv.init_zero();

    for (int k = 0; k < 4; ++k) {
        double c, s;
        get_cs_eta_phi(eta[k], c, s); apply_rx(sv, k, c, s);
        get_cs_pt(pt[k], c, s); apply_ry(sv, k, c, s);
        get_cs_eta_phi(phi[k], c, s); apply_rz(sv, k, c, s);
    }

    for (int k = 0; k < 3; ++k) {
        apply_cnot(sv, k, k + 1);
    }

    for (int d = 0; d < MU_DEPTH; ++d) {
        int base = d * 8;
        for (int k = 0; k < 4; ++k) {
            apply_rx(sv, k, mu_weights_c[base + k], mu_weights_s[base + k]);
            apply_ry(sv, k, mu_weights_c[base + 4 + k], mu_weights_s[base + 4 + k]);
        }
        for (int k = 0; k < 3; ++k) {
            apply_cnot(sv, k, k + 1);
        }
    }

    std::array<double, 8> out;
    for (int k = 0; k < 4; ++k) out[k] = measure_expZ(sv, k);
    for (int k = 0; k < 4; ++k) out[4 + k] = measure_expX(sv, k);
    return out;
}

// Jet block: 10 qubits -> 13 outputs (10 Z + 3 X)
std::array<double, 13> run_jet_block(const double pt[10], const double eta[10], const double phi[10]) {
    StateVec<10> sv;
    sv.init_zero();

    for (int k = 0; k < 10; ++k) {
        double c, s;
        get_cs_eta_phi(eta[k], c, s); apply_rx(sv, k, c, s);
        get_cs_pt(pt[k], c, s); apply_ry(sv, k, c, s);
        get_cs_eta_phi(phi[k], c, s); apply_rz(sv, k, c, s);
    }

    for (int k = 0; k < 9; ++k) {
        apply_cnot(sv, k, k + 1);
    }

    for (int d = 0; d < JET_DEPTH; ++d) {
        int base = d * 13;
        for (int k = 0; k < 10; ++k) {
            apply_rx(sv, k, jet_weights_c[base + k], jet_weights_s[base + k]);
        }
        for (int k = 0; k < 3; ++k) {
            apply_ry(sv, k, jet_weights_c[base + 10 + k], jet_weights_s[base + 10 + k]);
        }
        for (int k = 0; k < 9; ++k) {
            apply_cnot(sv, k, k + 1);
        }
    }

    std::array<double, 13> out;
    for (int k = 0; k < 10; ++k) out[k] = measure_expZ(sv, k);
    for (int k = 0; k < 3; ++k) out[10 + k] = measure_expX(sv, k);
    return out;
}

// ========================== Full Encoder ==========================

// Input: 56 features
// Output: 32 quantum encoder outputs
std::array<double, 32> extended_vae_encoder(const double x[56]) {
    std::array<double, 32> out;

    // MET: pt=x[0], phi=x[1] -> 3 outputs
    auto met_out = run_met_block(x[0], x[1]);
    out[0] = met_out[0];
    out[1] = met_out[1];
    out[2] = met_out[2];

    // Electrons: pt[2:6], eta[6:10], phi[10:14] -> 8 outputs
    double ele_pt[4] = {x[2], x[3], x[4], x[5]};
    double ele_eta[4] = {x[6], x[7], x[8], x[9]};
    double ele_phi[4] = {x[10], x[11], x[12], x[13]};
    auto ele_out = run_ele_block(ele_pt, ele_eta, ele_phi);
    for (int i = 0; i < 8; ++i) out[3 + i] = ele_out[i];

    // Muons: pt[14:18], eta[18:22], phi[22:26] -> 8 outputs
    double mu_pt[4] = {x[14], x[15], x[16], x[17]};
    double mu_eta[4] = {x[18], x[19], x[20], x[21]};
    double mu_phi[4] = {x[22], x[23], x[24], x[25]};
    auto mu_out = run_mu_block(mu_pt, mu_eta, mu_phi);
    for (int i = 0; i < 8; ++i) out[11 + i] = mu_out[i];

    // Jets: pt[26:36], eta[36:46], phi[46:56] -> 13 outputs
    double jet_pt[10], jet_eta[10], jet_phi[10];
    for (int i = 0; i < 10; ++i) {
        jet_pt[i] = x[26 + i];
        jet_eta[i] = x[36 + i];
        jet_phi[i] = x[46 + i];
    }
    auto jet_out = run_jet_block(jet_pt, jet_eta, jet_phi);
    for (int i = 0; i < 13; ++i) out[19 + i] = jet_out[i];

    return out;
}

// ========================== CSV I/O ==========================

std::vector<std::array<double, 56>> read_csv(const std::string& filename) {
    std::vector<std::array<double, 56>> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << std::endl;
        return data;
    }
    std::string line;
    while (std::getline(file, line)) {
        std::array<double, 56> row;
        std::stringstream ss(line);
        std::string val;
        int col = 0;
        while (std::getline(ss, val, ',') && col < 56) {
            row[col++] = std::stod(val);
        }
        if (col == 56) data.push_back(row);
    }
    return data;
}

void write_outputs_csv(const std::string& filename,
                       const std::vector<std::array<double, 32>>& outputs) {
    std::ofstream file(filename);
    file << std::fixed << std::setprecision(12);
    for (const auto& row : outputs) {
        for (int i = 0; i < 32; ++i) {
            file << row[i];
            if (i < 31) file << ",";
        }
        file << "\n";
    }
    std::cout << "Wrote " << outputs.size() << " encoder outputs to " << filename << std::endl;
}

// ========================== Main ==========================

int main(int argc, char* argv[]) {
    std::string input_file = "test_samples.csv";
    std::string output_file = "cpp_encoder_outputs.csv";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-i" && i + 1 < argc) input_file = argv[++i];
        else if (arg == "-o" && i + 1 < argc) output_file = argv[++i];
        else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [-i input.csv] [-o output.csv]\n";
            return 0;
        }
    }

    std::cout << "Loading data from " << input_file << "..." << std::endl;
    auto data = read_csv(input_file);
    if (data.empty()) {
        std::cerr << "No data loaded." << std::endl;
        return 1;
    }
    std::cout << "Loaded " << data.size() << " samples" << std::endl;

    std::vector<std::array<double, 32>> all_outputs;
    all_outputs.reserve(data.size());

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < data.size(); ++i) {
        auto out = extended_vae_encoder(data[i].data());
        all_outputs.push_back(out);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Inference completed in " << duration.count() << " ms" << std::endl;
    std::cout << "Throughput: " << (data.size() * 1000.0 / duration.count()) << " samples/sec" << std::endl;

    write_outputs_csv(output_file, all_outputs);

    return 0;
}
