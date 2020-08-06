#include "chemistry_file.H"

#ifndef AMREX_USE_CUDA
namespace thermo
{
    double fwd_A[211], fwd_beta[211], fwd_Ea[211];
    double low_A[211], low_beta[211], low_Ea[211];
    double rev_A[211], rev_beta[211], rev_Ea[211];
    double troe_a[211],troe_Ts[211], troe_Tss[211], troe_Tsss[211];
    double sri_a[211], sri_b[211], sri_c[211], sri_d[211], sri_e[211];
    double activation_units[211], prefactor_units[211], phase_units[211];
    int is_PD[211], troe_len[211], sri_len[211], nTB[211], *TBid[211];
    double *TB[211];
    std::vector<std::vector<double>> kiv(211); 
    std::vector<std::vector<double>> nuv(211); 
    std::vector<std::vector<double>> kiv_qss(211); 
    std::vector<std::vector<double>> nuv_qss(211); 
};

using namespace thermo;
#endif

/* Inverse molecular weights */
static AMREX_GPU_DEVICE_MANAGED double imw[25] = {
    1.0 / 28.013400,  /*N2 */
    1.0 / 15.999400,  /*O */
    1.0 / 2.015940,  /*H2 */
    1.0 / 1.007970,  /*H */
    1.0 / 17.007370,  /*OH */
    1.0 / 18.015340,  /*H2O */
    1.0 / 31.998800,  /*O2 */
    1.0 / 33.006770,  /*HO2 */
    1.0 / 34.014740,  /*H2O2 */
    1.0 / 44.009950,  /*CO2 */
    1.0 / 28.010550,  /*CO */
    1.0 / 30.026490,  /*CH2O */
    1.0 / 15.035060,  /*CH3 */
    1.0 / 16.043030,  /*CH4 */
    1.0 / 32.042430,  /*CH3OH */
    1.0 / 30.070120,  /*C2H6 */
    1.0 / 42.037640,  /*CH2CO */
    1.0 / 46.025890,  /*HOCHO */
    1.0 / 26.038240,  /*C2H2 */
    1.0 / 42.081270,  /*C3H6 */
    1.0 / 28.054180,  /*C2H4 */
    1.0 / 40.065330,  /*C3H4XA */
    1.0 / 54.092420,  /*C4H6 */
    1.0 / 56.108360,  /*C4H8X1 */
    1.0 / 100.205570};  /*NXC7H16 */

/* Molecular weights */
static AMREX_GPU_DEVICE_MANAGED double molecular_weights[25] = {
    28.013400,  /*N2 */
    15.999400,  /*O */
    2.015940,  /*H2 */
    1.007970,  /*H */
    17.007370,  /*OH */
    18.015340,  /*H2O */
    31.998800,  /*O2 */
    33.006770,  /*HO2 */
    34.014740,  /*H2O2 */
    44.009950,  /*CO2 */
    28.010550,  /*CO */
    30.026490,  /*CH2O */
    15.035060,  /*CH3 */
    16.043030,  /*CH4 */
    32.042430,  /*CH3OH */
    30.070120,  /*C2H6 */
    42.037640,  /*CH2CO */
    46.025890,  /*HOCHO */
    26.038240,  /*C2H2 */
    42.081270,  /*C3H6 */
    28.054180,  /*C2H4 */
    40.065330,  /*C3H4XA */
    54.092420,  /*C4H6 */
    56.108360,  /*C4H8X1 */
    100.205570};  /*NXC7H16 */

AMREX_GPU_HOST_DEVICE
void get_imw(double imw_new[]){
    for(int i = 0; i<25; ++i) imw_new[i] = imw[i];
}

/* TODO: check necessity because redundant with CKWT */
AMREX_GPU_HOST_DEVICE
void get_mw(double mw_new[]){
    for(int i = 0; i<25; ++i) mw_new[i] = molecular_weights[i];
}


#ifndef AMREX_USE_CUDA
/* Initializes parameter database */
void CKINIT()
{

    // (0):  O + H2 => H + OH
    kiv[18] = {1,2,3,4};
    nuv[18] = {-1,-1,1,1};
    kiv_qss[18] = {};
    nuv_qss[18] = {};
    // (0):  O + H2 => H + OH
    fwd_A[18]     = 50800;
    fwd_beta[18]  = 2.6699999999999999;
    fwd_Ea[18]    = 6292.0699999999997;
    prefactor_units[18]  = 1.0000000000000002e-06;
    activation_units[18] = 0.50321666580471969;
    phase_units[18]      = pow(10,-12.000000);
    is_PD[18] = 0;
    nTB[18] = 0;

    // (1):  H + OH => O + H2
    kiv[19] = {3,4,1,2};
    nuv[19] = {-1,-1,1,1};
    kiv_qss[19] = {};
    nuv_qss[19] = {};
    // (1):  H + OH => O + H2
    fwd_A[19]     = 22310;
    fwd_beta[19]  = 2.6699999999999999;
    fwd_Ea[19]    = 4196.9399999999996;
    prefactor_units[19]  = 1.0000000000000002e-06;
    activation_units[19] = 0.50321666580471969;
    phase_units[19]      = pow(10,-12.000000);
    is_PD[19] = 0;
    nTB[19] = 0;

    // (2):  OH + H2 => H + H2O
    kiv[20] = {4,2,3,5};
    nuv[20] = {-1,-1,1,1};
    kiv_qss[20] = {};
    nuv_qss[20] = {};
    // (2):  OH + H2 => H + H2O
    fwd_A[20]     = 216000000;
    fwd_beta[20]  = 1.51;
    fwd_Ea[20]    = 3429.9699999999998;
    prefactor_units[20]  = 1.0000000000000002e-06;
    activation_units[20] = 0.50321666580471969;
    phase_units[20]      = pow(10,-12.000000);
    is_PD[20] = 0;
    nTB[20] = 0;

    // (3):  H + H2O => OH + H2
    kiv[21] = {3,5,4,2};
    nuv[21] = {-1,-1,1,1};
    kiv_qss[21] = {};
    nuv_qss[21] = {};
    // (3):  H + H2O => OH + H2
    fwd_A[21]     = 935200000;
    fwd_beta[21]  = 1.51;
    fwd_Ea[21]    = 18580.07;
    prefactor_units[21]  = 1.0000000000000002e-06;
    activation_units[21] = 0.50321666580471969;
    phase_units[21]      = pow(10,-12.000000);
    is_PD[21] = 0;
    nTB[21] = 0;

    // (4):  H + OH + M => H2O + M
    kiv[11] = {3,4,5};
    nuv[11] = {-1,-1,1};
    kiv_qss[11] = {};
    nuv_qss[11] = {};
    // (4):  H + OH + M => H2O + M
    fwd_A[11]     = 2.2499999999999999e+22;
    fwd_beta[11]  = -2;
    fwd_Ea[11]    = 0;
    prefactor_units[11]  = 1.0000000000000002e-12;
    activation_units[11] = 0.50321666580471969;
    phase_units[11]      = pow(10,-12.000000);
    is_PD[11] = 0;
    nTB[11] = 4;
    TB[11] = (double *) malloc(4 * sizeof(double));
    TBid[11] = (int *) malloc(4 * sizeof(int));
    TBid[11][0] = 10; TB[11][0] = 1.8999999999999999; // CO
    TBid[11][1] = 2; TB[11][1] = 2.5; // H2
    TBid[11][2] = 5; TB[11][2] = 12; // H2O
    TBid[11][3] = 9; TB[11][3] = 3.7999999999999998; // CO2

    // (5):  O + H2O => 2.000000 OH
    kiv[22] = {1,5,4};
    nuv[22] = {-1,-1,2.0};
    kiv_qss[22] = {};
    nuv_qss[22] = {};
    // (5):  O + H2O => 2.000000 OH
    fwd_A[22]     = 2970000;
    fwd_beta[22]  = 2.02;
    fwd_Ea[22]    = 13400.1;
    prefactor_units[22]  = 1.0000000000000002e-06;
    activation_units[22] = 0.50321666580471969;
    phase_units[22]      = pow(10,-12.000000);
    is_PD[22] = 0;
    nTB[22] = 0;

    // (6):  2.000000 OH => O + H2O
    kiv[23] = {4,1,5};
    nuv[23] = {-2.0,1,1};
    kiv_qss[23] = {};
    nuv_qss[23] = {};
    // (6):  2.000000 OH => O + H2O
    fwd_A[23]     = 301300;
    fwd_beta[23]  = 2.02;
    fwd_Ea[23]    = -3849.9000000000001;
    prefactor_units[23]  = 1.0000000000000002e-06;
    activation_units[23] = 0.50321666580471969;
    phase_units[23]      = pow(10,-12.000000);
    is_PD[23] = 0;
    nTB[23] = 0;

    // (7):  H + O2 (+M) => HO2 (+M)
    kiv[0] = {3,6,7};
    nuv[0] = {-1,-1,1};
    kiv_qss[0] = {};
    nuv_qss[0] = {};
    // (7):  H + O2 (+M) => HO2 (+M)
    fwd_A[0]     = 1475000000000;
    fwd_beta[0]  = 0.59999999999999998;
    fwd_Ea[0]    = 0;
    low_A[0]     = 35000000000000000;
    low_beta[0]  = -0.40999999999999998;
    low_Ea[0]    = -1115.9200000000001;
    troe_a[0]    = 0.5;
    troe_Tsss[0] = 0;
    troe_Ts[0]   = 1e+30;
    troe_Tss[0]  = 1e+100;
    troe_len[0]  = 4;
    prefactor_units[0]  = 1.0000000000000002e-06;
    activation_units[0] = 0.50321666580471969;
    phase_units[0]      = pow(10,-12.000000);
    is_PD[0] = 1;
    nTB[0] = 4;
    TB[0] = (double *) malloc(4 * sizeof(double));
    TBid[0] = (int *) malloc(4 * sizeof(int));
    TBid[0][0] = 10; TB[0][0] = 1.8999999999999999; // CO
    TBid[0][1] = 2; TB[0][1] = 2.5; // H2
    TBid[0][2] = 5; TB[0][2] = 12; // H2O
    TBid[0][3] = 9; TB[0][3] = 3.7999999999999998; // CO2

    // (8):  H + O2 => O + OH
    kiv[24] = {3,6,1,4};
    nuv[24] = {-1,-1,1,1};
    kiv_qss[24] = {};
    nuv_qss[24] = {};
    // (8):  H + O2 => O + OH
    fwd_A[24]     = 197000000000000;
    fwd_beta[24]  = 0;
    fwd_Ea[24]    = 16539.91;
    prefactor_units[24]  = 1.0000000000000002e-06;
    activation_units[24] = 0.50321666580471969;
    phase_units[24]      = pow(10,-12.000000);
    is_PD[24] = 0;
    nTB[24] = 0;

    // (9):  O + OH => H + O2
    kiv[25] = {1,4,3,6};
    nuv[25] = {-1,-1,1,1};
    kiv_qss[25] = {};
    nuv_qss[25] = {};
    // (9):  O + OH => H + O2
    fwd_A[25]     = 15550000000000;
    fwd_beta[25]  = 0;
    fwd_Ea[25]    = 424.94999999999999;
    prefactor_units[25]  = 1.0000000000000002e-06;
    activation_units[25] = 0.50321666580471969;
    phase_units[25]      = pow(10,-12.000000);
    is_PD[25] = 0;
    nTB[25] = 0;

    // (10):  HO2 + OH => H2O + O2
    kiv[26] = {7,4,5,6};
    nuv[26] = {-1,-1,1,1};
    kiv_qss[26] = {};
    nuv_qss[26] = {};
    // (10):  HO2 + OH => H2O + O2
    fwd_A[26]     = 28900000000000;
    fwd_beta[26]  = 0;
    fwd_Ea[26]    = -500;
    prefactor_units[26]  = 1.0000000000000002e-06;
    activation_units[26] = 0.50321666580471969;
    phase_units[26]      = pow(10,-12.000000);
    is_PD[26] = 0;
    nTB[26] = 0;

    // (11):  HO2 + O => OH + O2
    kiv[27] = {7,1,4,6};
    nuv[27] = {-1,-1,1,1};
    kiv_qss[27] = {};
    nuv_qss[27] = {};
    // (11):  HO2 + O => OH + O2
    fwd_A[27]     = 32500000000000;
    fwd_beta[27]  = 0;
    fwd_Ea[27]    = 0;
    prefactor_units[27]  = 1.0000000000000002e-06;
    activation_units[27] = 0.50321666580471969;
    phase_units[27]      = pow(10,-12.000000);
    is_PD[27] = 0;
    nTB[27] = 0;

    // (12):  2.000000 HO2 => H2O2 + O2
    kiv[28] = {7,8,6};
    nuv[28] = {-2.0,1,1};
    kiv_qss[28] = {};
    nuv_qss[28] = {};
    // (12):  2.000000 HO2 => H2O2 + O2
    fwd_A[28]     = 420000000000000;
    fwd_beta[28]  = 0;
    fwd_Ea[28]    = 11979.92;
    prefactor_units[28]  = 1.0000000000000002e-06;
    activation_units[28] = 0.50321666580471969;
    phase_units[28]      = pow(10,-12.000000);
    is_PD[28] = 0;
    nTB[28] = 0;

    // (13):  HO2 + H => H2 + O2
    kiv[29] = {7,3,2,6};
    nuv[29] = {-1,-1,1,1};
    kiv_qss[29] = {};
    nuv_qss[29] = {};
    // (13):  HO2 + H => H2 + O2
    fwd_A[29]     = 16600000000000;
    fwd_beta[29]  = 0;
    fwd_Ea[29]    = 820.02999999999997;
    prefactor_units[29]  = 1.0000000000000002e-06;
    activation_units[29] = 0.50321666580471969;
    phase_units[29]      = pow(10,-12.000000);
    is_PD[29] = 0;
    nTB[29] = 0;

    // (14):  2.000000 HO2 => H2O2 + O2
    kiv[30] = {7,8,6};
    nuv[30] = {-2.0,1,1};
    kiv_qss[30] = {};
    nuv_qss[30] = {};
    // (14):  2.000000 HO2 => H2O2 + O2
    fwd_A[30]     = 130000000000;
    fwd_beta[30]  = 0;
    fwd_Ea[30]    = -1629.0599999999999;
    prefactor_units[30]  = 1.0000000000000002e-06;
    activation_units[30] = 0.50321666580471969;
    phase_units[30]      = pow(10,-12.000000);
    is_PD[30] = 0;
    nTB[30] = 0;

    // (15):  HO2 + H => 2.000000 OH
    kiv[31] = {7,3,4};
    nuv[31] = {-1,-1,2.0};
    kiv_qss[31] = {};
    nuv_qss[31] = {};
    // (15):  HO2 + H => 2.000000 OH
    fwd_A[31]     = 70800000000000;
    fwd_beta[31]  = 0;
    fwd_Ea[31]    = 299.94999999999999;
    prefactor_units[31]  = 1.0000000000000002e-06;
    activation_units[31] = 0.50321666580471969;
    phase_units[31]      = pow(10,-12.000000);
    is_PD[31] = 0;
    nTB[31] = 0;

    // (16):  H2O2 + OH => H2O + HO2
    kiv[32] = {8,4,5,7};
    nuv[32] = {-1,-1,1,1};
    kiv_qss[32] = {};
    nuv_qss[32] = {};
    // (16):  H2O2 + OH => H2O + HO2
    fwd_A[32]     = 580000000000000;
    fwd_beta[32]  = 0;
    fwd_Ea[32]    = 9559.9899999999998;
    prefactor_units[32]  = 1.0000000000000002e-06;
    activation_units[32] = 0.50321666580471969;
    phase_units[32]      = pow(10,-12.000000);
    is_PD[32] = 0;
    nTB[32] = 0;

    // (17):  H2O2 (+M) => 2.000000 OH (+M)
    kiv[1] = {8,4};
    nuv[1] = {-1,2.0};
    kiv_qss[1] = {};
    nuv_qss[1] = {};
    // (17):  H2O2 (+M) => 2.000000 OH (+M)
    fwd_A[1]     = 1.324e+20;
    fwd_beta[1]  = -1.538;
    fwd_Ea[1]    = 52449.330000000002;
    low_A[1]     = 3.258e+36;
    low_beta[1]  = -5.798;
    low_Ea[1]    = 54498.330000000002;
    troe_a[1]    = 0.46999999999999997;
    troe_Tsss[1] = 100;
    troe_Ts[1]   = 2000;
    troe_Tss[1]  = 1000000000000000;
    troe_len[1]  = 4;
    prefactor_units[1]  = 1;
    activation_units[1] = 0.50321666580471969;
    phase_units[1]      = pow(10,-6.000000);
    is_PD[1] = 1;
    nTB[1] = 4;
    TB[1] = (double *) malloc(4 * sizeof(double));
    TBid[1] = (int *) malloc(4 * sizeof(int));
    TBid[1][0] = 10; TB[1][0] = 1.8999999999999999; // CO
    TBid[1][1] = 2; TB[1][1] = 2.5; // H2
    TBid[1][2] = 5; TB[1][2] = 12; // H2O
    TBid[1][3] = 9; TB[1][3] = 3.7999999999999998; // CO2

    // (18):  CH + O2 => HCO + O
    kiv[33] = {6,1};
    nuv[33] = {-1,1};
    kiv_qss[33] = {0,1};
    nuv_qss[33] = {-1,1};
    // (18):  CH + O2 => HCO + O
    fwd_A[33]     = 33000000000000;
    fwd_beta[33]  = 0;
    fwd_Ea[33]    = 0;
    prefactor_units[33]  = 1.0000000000000002e-06;
    activation_units[33] = 0.50321666580471969;
    phase_units[33]      = pow(10,-12.000000);
    is_PD[33] = 0;
    nTB[33] = 0;

    // (19):  CH2 + O2 => CO2 + 2.000000 H
    kiv[34] = {6,9,3};
    nuv[34] = {-1,1,2.0};
    kiv_qss[34] = {2};
    nuv_qss[34] = {-1};
    // (19):  CH2 + O2 => CO2 + 2.000000 H
    fwd_A[34]     = 3.29e+21;
    fwd_beta[34]  = -3.2999999999999998;
    fwd_Ea[34]    = 2868.0700000000002;
    prefactor_units[34]  = 1.0000000000000002e-06;
    activation_units[34] = 0.50321666580471969;
    phase_units[34]      = pow(10,-12.000000);
    is_PD[34] = 0;
    nTB[34] = 0;

    // (20):  CH2 + O2 => CO + H2O
    kiv[35] = {6,10,5};
    nuv[35] = {-1,1,1};
    kiv_qss[35] = {2};
    nuv_qss[35] = {-1};
    // (20):  CH2 + O2 => CO + H2O
    fwd_A[35]     = 7.28e+19;
    fwd_beta[35]  = -2.54;
    fwd_Ea[35]    = 1809.03;
    prefactor_units[35]  = 1.0000000000000002e-06;
    activation_units[35] = 0.50321666580471969;
    phase_units[35]      = pow(10,-12.000000);
    is_PD[35] = 0;
    nTB[35] = 0;

    // (21):  CH2 + O => CO + 2.000000 H
    kiv[36] = {1,10,3};
    nuv[36] = {-1,1,2.0};
    kiv_qss[36] = {2};
    nuv_qss[36] = {-1};
    // (21):  CH2 + O => CO + 2.000000 H
    fwd_A[36]     = 50000000000000;
    fwd_beta[36]  = 0;
    fwd_Ea[36]    = 0;
    prefactor_units[36]  = 1.0000000000000002e-06;
    activation_units[36] = 0.50321666580471969;
    phase_units[36]      = pow(10,-12.000000);
    is_PD[36] = 0;
    nTB[36] = 0;

    // (22):  CH2 + O2 => CH2O + O
    kiv[37] = {6,11,1};
    nuv[37] = {-1,1,1};
    kiv_qss[37] = {2};
    nuv_qss[37] = {-1};
    // (22):  CH2 + O2 => CH2O + O
    fwd_A[37]     = 3.29e+21;
    fwd_beta[37]  = -3.2999999999999998;
    fwd_Ea[37]    = 2868.0700000000002;
    prefactor_units[37]  = 1.0000000000000002e-06;
    activation_units[37] = 0.50321666580471969;
    phase_units[37]      = pow(10,-12.000000);
    is_PD[37] = 0;
    nTB[37] = 0;

    // (23):  CH2 + H => CH + H2
    kiv[38] = {3,2};
    nuv[38] = {-1,1};
    kiv_qss[38] = {2,0};
    nuv_qss[38] = {-1,1};
    // (23):  CH2 + H => CH + H2
    fwd_A[38]     = 1e+18;
    fwd_beta[38]  = -1.5600000000000001;
    fwd_Ea[38]    = 0;
    prefactor_units[38]  = 1.0000000000000002e-06;
    activation_units[38] = 0.50321666580471969;
    phase_units[38]      = pow(10,-12.000000);
    is_PD[38] = 0;
    nTB[38] = 0;

    // (24):  CH + H2 => CH2 + H
    kiv[39] = {2,3};
    nuv[39] = {-1,1};
    kiv_qss[39] = {0,2};
    nuv_qss[39] = {-1,1};
    // (24):  CH + H2 => CH2 + H
    fwd_A[39]     = 7.026e+17;
    fwd_beta[39]  = -1.5600000000000001;
    fwd_Ea[39]    = 2989.96;
    prefactor_units[39]  = 1.0000000000000002e-06;
    activation_units[39] = 0.50321666580471969;
    phase_units[39]      = pow(10,-12.000000);
    is_PD[39] = 0;
    nTB[39] = 0;

    // (25):  CH2 + OH => CH + H2O
    kiv[40] = {4,5};
    nuv[40] = {-1,1};
    kiv_qss[40] = {2,0};
    nuv_qss[40] = {-1,1};
    // (25):  CH2 + OH => CH + H2O
    fwd_A[40]     = 11300000;
    fwd_beta[40]  = 2;
    fwd_Ea[40]    = 3000;
    prefactor_units[40]  = 1.0000000000000002e-06;
    activation_units[40] = 0.50321666580471969;
    phase_units[40]      = pow(10,-12.000000);
    is_PD[40] = 0;
    nTB[40] = 0;

    // (26):  CH + H2O => CH2 + OH
    kiv[41] = {5,4};
    nuv[41] = {-1,1};
    kiv_qss[41] = {0,2};
    nuv_qss[41] = {-1,1};
    // (26):  CH + H2O => CH2 + OH
    fwd_A[41]     = 34370000;
    fwd_beta[41]  = 2;
    fwd_Ea[41]    = 21140.060000000001;
    prefactor_units[41]  = 1.0000000000000002e-06;
    activation_units[41] = 0.50321666580471969;
    phase_units[41]      = pow(10,-12.000000);
    is_PD[41] = 0;
    nTB[41] = 0;

    // (27):  CH2 + O2 => CO2 + H2
    kiv[42] = {6,9,2};
    nuv[42] = {-1,1,1};
    kiv_qss[42] = {2};
    nuv_qss[42] = {-1};
    // (27):  CH2 + O2 => CO2 + H2
    fwd_A[42]     = 1.01e+21;
    fwd_beta[42]  = -3.2999999999999998;
    fwd_Ea[42]    = 1507.8900000000001;
    prefactor_units[42]  = 1.0000000000000002e-06;
    activation_units[42] = 0.50321666580471969;
    phase_units[42]      = pow(10,-12.000000);
    is_PD[42] = 0;
    nTB[42] = 0;

    // (28):  CH2GSG + H => CH + H2
    kiv[43] = {3,2};
    nuv[43] = {-1,1};
    kiv_qss[43] = {3,0};
    nuv_qss[43] = {-1,1};
    // (28):  CH2GSG + H => CH + H2
    fwd_A[43]     = 30000000000000;
    fwd_beta[43]  = 0;
    fwd_Ea[43]    = 0;
    prefactor_units[43]  = 1.0000000000000002e-06;
    activation_units[43] = 0.50321666580471969;
    phase_units[43]      = pow(10,-12.000000);
    is_PD[43] = 0;
    nTB[43] = 0;

    // (29):  CH2GSG + M => CH2 + M
    kiv[12] = {};
    nuv[12] = {};
    kiv_qss[12] = {3,2};
    nuv_qss[12] = {-1,1};
    // (29):  CH2GSG + M => CH2 + M
    fwd_A[12]     = 10000000000000;
    fwd_beta[12]  = 0;
    fwd_Ea[12]    = 0;
    prefactor_units[12]  = 1.0000000000000002e-06;
    activation_units[12] = 0.50321666580471969;
    phase_units[12]      = pow(10,-6.000000);
    is_PD[12] = 0;
    nTB[12] = 0;

    // (30):  CH2 + M => CH2GSG + M
    kiv[13] = {};
    nuv[13] = {};
    kiv_qss[13] = {2,3};
    nuv_qss[13] = {-1,1};
    // (30):  CH2 + M => CH2GSG + M
    fwd_A[13]     = 7161000000000000;
    fwd_beta[13]  = -0.89000000000000001;
    fwd_Ea[13]    = 11429.969999999999;
    prefactor_units[13]  = 1.0000000000000002e-06;
    activation_units[13] = 0.50321666580471969;
    phase_units[13]      = pow(10,-6.000000);
    is_PD[13] = 0;
    nTB[13] = 0;

    // (31):  CH2GSG + H2 => CH3 + H
    kiv[44] = {2,12,3};
    nuv[44] = {-1,1,1};
    kiv_qss[44] = {3};
    nuv_qss[44] = {-1};
    // (31):  CH2GSG + H2 => CH3 + H
    fwd_A[44]     = 70000000000000;
    fwd_beta[44]  = 0;
    fwd_Ea[44]    = 0;
    prefactor_units[44]  = 1.0000000000000002e-06;
    activation_units[44] = 0.50321666580471969;
    phase_units[44]      = pow(10,-12.000000);
    is_PD[44] = 0;
    nTB[44] = 0;

    // (32):  CH3 + H => CH2GSG + H2
    kiv[45] = {12,3,2};
    nuv[45] = {-1,-1,1};
    kiv_qss[45] = {3};
    nuv_qss[45] = {1};
    // (32):  CH3 + H => CH2GSG + H2
    fwd_A[45]     = 2.482e+17;
    fwd_beta[45]  = -0.89000000000000001;
    fwd_Ea[45]    = 16130.02;
    prefactor_units[45]  = 1.0000000000000002e-06;
    activation_units[45] = 0.50321666580471969;
    phase_units[45]      = pow(10,-12.000000);
    is_PD[45] = 0;
    nTB[45] = 0;

    // (33):  CH2GSG + O2 => CO + OH + H
    kiv[46] = {6,10,4,3};
    nuv[46] = {-1,1,1,1};
    kiv_qss[46] = {3};
    nuv_qss[46] = {-1};
    // (33):  CH2GSG + O2 => CO + OH + H
    fwd_A[46]     = 70000000000000;
    fwd_beta[46]  = 0;
    fwd_Ea[46]    = 0;
    prefactor_units[46]  = 1.0000000000000002e-06;
    activation_units[46] = 0.50321666580471969;
    phase_units[46]      = pow(10,-12.000000);
    is_PD[46] = 0;
    nTB[46] = 0;

    // (34):  CH2GSG + OH => CH2O + H
    kiv[47] = {4,11,3};
    nuv[47] = {-1,1,1};
    kiv_qss[47] = {3};
    nuv_qss[47] = {-1};
    // (34):  CH2GSG + OH => CH2O + H
    fwd_A[47]     = 30000000000000;
    fwd_beta[47]  = 0;
    fwd_Ea[47]    = 0;
    prefactor_units[47]  = 1.0000000000000002e-06;
    activation_units[47] = 0.50321666580471969;
    phase_units[47]      = pow(10,-12.000000);
    is_PD[47] = 0;
    nTB[47] = 0;

    // (35):  CH3 + OH => CH2O + H2
    kiv[48] = {12,4,11,2};
    nuv[48] = {-1,-1,1,1};
    kiv_qss[48] = {};
    nuv_qss[48] = {};
    // (35):  CH3 + OH => CH2O + H2
    fwd_A[48]     = 22500000000000;
    fwd_beta[48]  = 0;
    fwd_Ea[48]    = 4299.9499999999998;
    prefactor_units[48]  = 1.0000000000000002e-06;
    activation_units[48] = 0.50321666580471969;
    phase_units[48]      = pow(10,-12.000000);
    is_PD[48] = 0;
    nTB[48] = 0;

    // (36):  CH3 + OH => CH2GSG + H2O
    kiv[49] = {12,4,5};
    nuv[49] = {-1,-1,1};
    kiv_qss[49] = {3};
    nuv_qss[49] = {1};
    // (36):  CH3 + OH => CH2GSG + H2O
    fwd_A[49]     = 26500000000000;
    fwd_beta[49]  = 0;
    fwd_Ea[49]    = 2185.9499999999998;
    prefactor_units[49]  = 1.0000000000000002e-06;
    activation_units[49] = 0.50321666580471969;
    phase_units[49]      = pow(10,-12.000000);
    is_PD[49] = 0;
    nTB[49] = 0;

    // (37):  CH2GSG + H2O => CH3 + OH
    kiv[50] = {5,12,4};
    nuv[50] = {-1,1,1};
    kiv_qss[50] = {3};
    nuv_qss[50] = {-1};
    // (37):  CH2GSG + H2O => CH3 + OH
    fwd_A[50]     = 32360000000;
    fwd_beta[50]  = 0.89000000000000001;
    fwd_Ea[50]    = 1211.04;
    prefactor_units[50]  = 1.0000000000000002e-06;
    activation_units[50] = 0.50321666580471969;
    phase_units[50]      = pow(10,-12.000000);
    is_PD[50] = 0;
    nTB[50] = 0;

    // (38):  CH3 + O => CH2O + H
    kiv[51] = {12,1,11,3};
    nuv[51] = {-1,-1,1,1};
    kiv_qss[51] = {};
    nuv_qss[51] = {};
    // (38):  CH3 + O => CH2O + H
    fwd_A[51]     = 80000000000000;
    fwd_beta[51]  = 0;
    fwd_Ea[51]    = 0;
    prefactor_units[51]  = 1.0000000000000002e-06;
    activation_units[51] = 0.50321666580471969;
    phase_units[51]      = pow(10,-12.000000);
    is_PD[51] = 0;
    nTB[51] = 0;

    // (39):  CH3 + HO2 => CH3O + OH
    kiv[52] = {12,7,4};
    nuv[52] = {-1,-1,1};
    kiv_qss[52] = {4};
    nuv_qss[52] = {1};
    // (39):  CH3 + HO2 => CH3O + OH
    fwd_A[52]     = 11000000000000;
    fwd_beta[52]  = 0;
    fwd_Ea[52]    = 0;
    prefactor_units[52]  = 1.0000000000000002e-06;
    activation_units[52] = 0.50321666580471969;
    phase_units[52]      = pow(10,-12.000000);
    is_PD[52] = 0;
    nTB[52] = 0;

    // (40):  CH3 + HO2 => CH4 + O2
    kiv[53] = {12,7,13,6};
    nuv[53] = {-1,-1,1,1};
    kiv_qss[53] = {};
    nuv_qss[53] = {};
    // (40):  CH3 + HO2 => CH4 + O2
    fwd_A[53]     = 3600000000000;
    fwd_beta[53]  = 0;
    fwd_Ea[53]    = 0;
    prefactor_units[53]  = 1.0000000000000002e-06;
    activation_units[53] = 0.50321666580471969;
    phase_units[53]      = pow(10,-12.000000);
    is_PD[53] = 0;
    nTB[53] = 0;

    // (41):  OH + CH3 (+M) => CH3OH (+M)
    kiv[2] = {4,12,14};
    nuv[2] = {-1,-1,1};
    kiv_qss[2] = {};
    nuv_qss[2] = {};
    // (41):  OH + CH3 (+M) => CH3OH (+M)
    fwd_A[2]     = 83430000000;
    fwd_beta[2]  = 0.79700000000000004;
    fwd_Ea[2]    = -1788.96;
    low_A[2]     = 1.295e+39;
    low_beta[2]  = -6.5529999999999999;
    low_Ea[2]    = 1941.2;
    troe_a[2]    = 0.41399999999999998;
    troe_Tsss[2] = 279;
    troe_Ts[2]   = 5459;
    troe_Tss[2]  = 1e+100;
    troe_len[2]  = 4;
    prefactor_units[2]  = 1.0000000000000002e-06;
    activation_units[2] = 0.50321666580471969;
    phase_units[2]      = pow(10,-12.000000);
    is_PD[2] = 1;
    nTB[2] = 6;
    TB[2] = (double *) malloc(6 * sizeof(double));
    TBid[2] = (int *) malloc(6 * sizeof(int));
    TBid[2][0] = 10; TB[2][0] = 1.5; // CO
    TBid[2][1] = 2; TB[2][1] = 2; // H2
    TBid[2][2] = 5; TB[2][2] = 6; // H2O
    TBid[2][3] = 9; TB[2][3] = 2; // CO2
    TBid[2][4] = 13; TB[2][4] = 2; // CH4
    TBid[2][5] = 15; TB[2][5] = 3; // C2H6

    // (42):  CH3 + O2 => CH2O + OH
    kiv[54] = {12,6,11,4};
    nuv[54] = {-1,-1,1,1};
    kiv_qss[54] = {};
    nuv_qss[54] = {};
    // (42):  CH3 + O2 => CH2O + OH
    fwd_A[54]     = 747000000000;
    fwd_beta[54]  = 0;
    fwd_Ea[54]    = 14250;
    prefactor_units[54]  = 1.0000000000000002e-06;
    activation_units[54] = 0.50321666580471969;
    phase_units[54]      = pow(10,-12.000000);
    is_PD[54] = 0;
    nTB[54] = 0;

    // (43):  CH3 + H (+M) => CH4 (+M)
    kiv[3] = {12,3,13};
    nuv[3] = {-1,-1,1};
    kiv_qss[3] = {};
    nuv_qss[3] = {};
    // (43):  CH3 + H (+M) => CH4 (+M)
    fwd_A[3]     = 2138000000000000;
    fwd_beta[3]  = -0.40000000000000002;
    fwd_Ea[3]    = 0;
    low_A[3]     = 3.31e+30;
    low_beta[3]  = -4;
    low_Ea[3]    = 2108.0300000000002;
    troe_a[3]    = 0;
    troe_Tsss[3] = 0;
    troe_Ts[3]   = 0;
    troe_Tss[3]  = 40;
    troe_len[3]  = 4;
    prefactor_units[3]  = 1.0000000000000002e-06;
    activation_units[3] = 0.50321666580471969;
    phase_units[3]      = pow(10,-12.000000);
    is_PD[3] = 1;
    nTB[3] = 4;
    TB[3] = (double *) malloc(4 * sizeof(double));
    TBid[3] = (int *) malloc(4 * sizeof(int));
    TBid[3][0] = 10; TB[3][0] = 2; // CO
    TBid[3][1] = 2; TB[3][1] = 2; // H2
    TBid[3][2] = 5; TB[3][2] = 5; // H2O
    TBid[3][3] = 9; TB[3][3] = 3; // CO2

    // (44):  CH3 + H => CH2 + H2
    kiv[55] = {12,3,2};
    nuv[55] = {-1,-1,1};
    kiv_qss[55] = {2};
    nuv_qss[55] = {1};
    // (44):  CH3 + H => CH2 + H2
    fwd_A[55]     = 90000000000000;
    fwd_beta[55]  = 0;
    fwd_Ea[55]    = 15099.9;
    prefactor_units[55]  = 1.0000000000000002e-06;
    activation_units[55] = 0.50321666580471969;
    phase_units[55]      = pow(10,-12.000000);
    is_PD[55] = 0;
    nTB[55] = 0;

    // (45):  CH2 + H2 => CH3 + H
    kiv[56] = {2,12,3};
    nuv[56] = {-1,1,1};
    kiv_qss[56] = {2};
    nuv_qss[56] = {-1};
    // (45):  CH2 + H2 => CH3 + H
    fwd_A[56]     = 18180000000000;
    fwd_beta[56]  = 0;
    fwd_Ea[56]    = 10400.1;
    prefactor_units[56]  = 1.0000000000000002e-06;
    activation_units[56] = 0.50321666580471969;
    phase_units[56]      = pow(10,-12.000000);
    is_PD[56] = 0;
    nTB[56] = 0;

    // (46):  2.000000 CH3 (+M) => C2H6 (+M)
    kiv[4] = {12,15};
    nuv[4] = {-2.0,1};
    kiv_qss[4] = {};
    nuv_qss[4] = {};
    // (46):  2.000000 CH3 (+M) => C2H6 (+M)
    fwd_A[4]     = 92140000000000000;
    fwd_beta[4]  = -1.1699999999999999;
    fwd_Ea[4]    = 635.75999999999999;
    low_A[4]     = 1.135e+36;
    low_beta[4]  = -5.2460000000000004;
    low_Ea[4]    = 1705.0699999999999;
    troe_a[4]    = 0.40500000000000003;
    troe_Tsss[4] = 1120;
    troe_Ts[4]   = 69.599999999999994;
    troe_Tss[4]  = 1000000000000000;
    troe_len[4]  = 4;
    prefactor_units[4]  = 1.0000000000000002e-06;
    activation_units[4] = 0.50321666580471969;
    phase_units[4]      = pow(10,-12.000000);
    is_PD[4] = 1;
    nTB[4] = 4;
    TB[4] = (double *) malloc(4 * sizeof(double));
    TBid[4] = (int *) malloc(4 * sizeof(int));
    TBid[4][0] = 10; TB[4][0] = 2; // CO
    TBid[4][1] = 2; TB[4][1] = 2; // H2
    TBid[4][2] = 5; TB[4][2] = 5; // H2O
    TBid[4][3] = 9; TB[4][3] = 3; // CO2

    // (47):  2.000000 CH3 <=> H + C2H5
    kiv[57] = {12,3};
    nuv[57] = {-2.0,1};
    kiv_qss[57] = {5};
    nuv_qss[57] = {1};
    // (47):  2.000000 CH3 <=> H + C2H5
    fwd_A[57]     = 6840000000000;
    fwd_beta[57]  = 0.10000000000000001;
    fwd_Ea[57]    = 10599.9;
    prefactor_units[57]  = 1.0000000000000002e-06;
    activation_units[57] = 0.50321666580471969;
    phase_units[57]      = pow(10,-12.000000);
    is_PD[57] = 0;
    nTB[57] = 0;

    // (48):  CH3 + OH => CH2 + H2O
    kiv[58] = {12,4,5};
    nuv[58] = {-1,-1,1};
    kiv_qss[58] = {2};
    nuv_qss[58] = {1};
    // (48):  CH3 + OH => CH2 + H2O
    fwd_A[58]     = 3000000;
    fwd_beta[58]  = 2;
    fwd_Ea[58]    = 2500;
    prefactor_units[58]  = 1.0000000000000002e-06;
    activation_units[58] = 0.50321666580471969;
    phase_units[58]      = pow(10,-12.000000);
    is_PD[58] = 0;
    nTB[58] = 0;

    // (49):  CH2 + H2O => CH3 + OH
    kiv[59] = {5,12,4};
    nuv[59] = {-1,1,1};
    kiv_qss[59] = {2};
    nuv_qss[59] = {-1};
    // (49):  CH2 + H2O => CH3 + OH
    fwd_A[59]     = 2623000;
    fwd_beta[59]  = 2;
    fwd_Ea[59]    = 12960.09;
    prefactor_units[59]  = 1.0000000000000002e-06;
    activation_units[59] = 0.50321666580471969;
    phase_units[59]      = pow(10,-12.000000);
    is_PD[59] = 0;
    nTB[59] = 0;

    // (50):  CH4 + O => CH3 + OH
    kiv[60] = {13,1,12,4};
    nuv[60] = {-1,-1,1,1};
    kiv_qss[60] = {};
    nuv_qss[60] = {};
    // (50):  CH4 + O => CH3 + OH
    fwd_A[60]     = 3150000000000;
    fwd_beta[60]  = 0.5;
    fwd_Ea[60]    = 10289.91;
    prefactor_units[60]  = 1.0000000000000002e-06;
    activation_units[60] = 0.50321666580471969;
    phase_units[60]      = pow(10,-12.000000);
    is_PD[60] = 0;
    nTB[60] = 0;

    // (51):  CH4 + H => CH3 + H2
    kiv[61] = {13,3,12,2};
    nuv[61] = {-1,-1,1,1};
    kiv_qss[61] = {};
    nuv_qss[61] = {};
    // (51):  CH4 + H => CH3 + H2
    fwd_A[61]     = 17270;
    fwd_beta[61]  = 3;
    fwd_Ea[61]    = 8223.9500000000007;
    prefactor_units[61]  = 1.0000000000000002e-06;
    activation_units[61] = 0.50321666580471969;
    phase_units[61]      = pow(10,-12.000000);
    is_PD[61] = 0;
    nTB[61] = 0;

    // (52):  CH3 + H2 => CH4 + H
    kiv[62] = {12,2,13,3};
    nuv[62] = {-1,-1,1,1};
    kiv_qss[62] = {};
    nuv_qss[62] = {};
    // (52):  CH3 + H2 => CH4 + H
    fwd_A[62]     = 661;
    fwd_beta[62]  = 3;
    fwd_Ea[62]    = 7744.0200000000004;
    prefactor_units[62]  = 1.0000000000000002e-06;
    activation_units[62] = 0.50321666580471969;
    phase_units[62]      = pow(10,-12.000000);
    is_PD[62] = 0;
    nTB[62] = 0;

    // (53):  CH4 + OH => CH3 + H2O
    kiv[63] = {13,4,12,5};
    nuv[63] = {-1,-1,1,1};
    kiv_qss[63] = {};
    nuv_qss[63] = {};
    // (53):  CH4 + OH => CH3 + H2O
    fwd_A[63]     = 193000;
    fwd_beta[63]  = 2.3999999999999999;
    fwd_Ea[63]    = 2106.1199999999999;
    prefactor_units[63]  = 1.0000000000000002e-06;
    activation_units[63] = 0.50321666580471969;
    phase_units[63]      = pow(10,-12.000000);
    is_PD[63] = 0;
    nTB[63] = 0;

    // (54):  CH3 + H2O => CH4 + OH
    kiv[64] = {12,5,13,4};
    nuv[64] = {-1,-1,1,1};
    kiv_qss[64] = {};
    nuv_qss[64] = {};
    // (54):  CH3 + H2O => CH4 + OH
    fwd_A[64]     = 31990;
    fwd_beta[64]  = 2.3999999999999999;
    fwd_Ea[64]    = 16780.110000000001;
    prefactor_units[64]  = 1.0000000000000002e-06;
    activation_units[64] = 0.50321666580471969;
    phase_units[64]      = pow(10,-12.000000);
    is_PD[64] = 0;
    nTB[64] = 0;

    // (55):  CO + CH2 (+M) => CH2CO (+M)
    kiv[5] = {10,16};
    nuv[5] = {-1,1};
    kiv_qss[5] = {2};
    nuv_qss[5] = {-1};
    // (55):  CO + CH2 (+M) => CH2CO (+M)
    fwd_A[5]     = 591900;
    fwd_beta[5]  = 1.7829999999999999;
    fwd_Ea[5]    = -8782.7399999999998;
    low_A[5]     = 7103000;
    low_beta[5]  = 1.7829999999999999;
    low_Ea[5]    = -20492.59;
    troe_a[5]    = 1;
    troe_Tsss[5] = 1;
    troe_Ts[5]   = 10000000;
    troe_Tss[5]  = 10000000;
    troe_len[5]  = 4;
    prefactor_units[5]  = 1.0000000000000002e-06;
    activation_units[5] = 0.50321666580471969;
    phase_units[5]      = pow(10,-12.000000);
    is_PD[5] = 1;
    nTB[5] = 0;

    // (56):  CO + O (+M) => CO2 (+M)
    kiv[6] = {10,1,9};
    nuv[6] = {-1,-1,1};
    kiv_qss[6] = {};
    nuv_qss[6] = {};
    // (56):  CO + O (+M) => CO2 (+M)
    fwd_A[6]     = 18000000000;
    fwd_beta[6]  = 0;
    fwd_Ea[6]    = 2384.0799999999999;
    low_A[6]     = 1.35e+24;
    low_beta[6]  = -2.7879999999999998;
    low_Ea[6]    = 4190.9700000000003;
    troe_a[6]    = 1;
    troe_Tsss[6] = 1;
    troe_Ts[6]   = 10000000;
    troe_Tss[6]  = 10000000;
    troe_len[6]  = 4;
    prefactor_units[6]  = 1.0000000000000002e-06;
    activation_units[6] = 0.50321666580471969;
    phase_units[6]      = pow(10,-12.000000);
    is_PD[6] = 1;
    nTB[6] = 4;
    TB[6] = (double *) malloc(4 * sizeof(double));
    TBid[6] = (int *) malloc(4 * sizeof(int));
    TBid[6][0] = 10; TB[6][0] = 1.8999999999999999; // CO
    TBid[6][1] = 2; TB[6][1] = 2.5; // H2
    TBid[6][2] = 5; TB[6][2] = 12; // H2O
    TBid[6][3] = 9; TB[6][3] = 3.7999999999999998; // CO2

    // (57):  CO + OH => CO2 + H
    kiv[65] = {10,4,9,3};
    nuv[65] = {-1,-1,1,1};
    kiv_qss[65] = {};
    nuv_qss[65] = {};
    // (57):  CO + OH => CO2 + H
    fwd_A[65]     = 140000;
    fwd_beta[65]  = 1.95;
    fwd_Ea[65]    = -1347.04;
    prefactor_units[65]  = 1.0000000000000002e-06;
    activation_units[65] = 0.50321666580471969;
    phase_units[65]      = pow(10,-12.000000);
    is_PD[65] = 0;
    nTB[65] = 0;

    // (58):  CO2 + H => CO + OH
    kiv[66] = {9,3,10,4};
    nuv[66] = {-1,-1,1,1};
    kiv_qss[66] = {};
    nuv_qss[66] = {};
    // (58):  CO2 + H => CO + OH
    fwd_A[66]     = 15680000;
    fwd_beta[66]  = 1.95;
    fwd_Ea[66]    = 20989.959999999999;
    prefactor_units[66]  = 1.0000000000000002e-06;
    activation_units[66] = 0.50321666580471969;
    phase_units[66]      = pow(10,-12.000000);
    is_PD[66] = 0;
    nTB[66] = 0;

    // (59):  HCO + O2 => CO + HO2
    kiv[67] = {6,10,7};
    nuv[67] = {-1,1,1};
    kiv_qss[67] = {1};
    nuv_qss[67] = {-1};
    // (59):  HCO + O2 => CO + HO2
    fwd_A[67]     = 7580000000000;
    fwd_beta[67]  = 0;
    fwd_Ea[67]    = 409.88999999999999;
    prefactor_units[67]  = 1.0000000000000002e-06;
    activation_units[67] = 0.50321666580471969;
    phase_units[67]      = pow(10,-12.000000);
    is_PD[67] = 0;
    nTB[67] = 0;

    // (60):  HCO + O => CO2 + H
    kiv[68] = {1,9,3};
    nuv[68] = {-1,1,1};
    kiv_qss[68] = {1};
    nuv_qss[68] = {-1};
    // (60):  HCO + O => CO2 + H
    fwd_A[68]     = 30000000000000;
    fwd_beta[68]  = 0;
    fwd_Ea[68]    = 0;
    prefactor_units[68]  = 1.0000000000000002e-06;
    activation_units[68] = 0.50321666580471969;
    phase_units[68]      = pow(10,-12.000000);
    is_PD[68] = 0;
    nTB[68] = 0;

    // (61):  HCO + OH => CO + H2O
    kiv[69] = {4,10,5};
    nuv[69] = {-1,1,1};
    kiv_qss[69] = {1};
    nuv_qss[69] = {-1};
    // (61):  HCO + OH => CO + H2O
    fwd_A[69]     = 102000000000000;
    fwd_beta[69]  = 0;
    fwd_Ea[69]    = 0;
    prefactor_units[69]  = 1.0000000000000002e-06;
    activation_units[69] = 0.50321666580471969;
    phase_units[69]      = pow(10,-12.000000);
    is_PD[69] = 0;
    nTB[69] = 0;

    // (62):  HCO + H => CO + H2
    kiv[70] = {3,10,2};
    nuv[70] = {-1,1,1};
    kiv_qss[70] = {1};
    nuv_qss[70] = {-1};
    // (62):  HCO + H => CO + H2
    fwd_A[70]     = 73400000000000;
    fwd_beta[70]  = 0;
    fwd_Ea[70]    = 0;
    prefactor_units[70]  = 1.0000000000000002e-06;
    activation_units[70] = 0.50321666580471969;
    phase_units[70]      = pow(10,-12.000000);
    is_PD[70] = 0;
    nTB[70] = 0;

    // (63):  HCO + O => CO + OH
    kiv[71] = {1,10,4};
    nuv[71] = {-1,1,1};
    kiv_qss[71] = {1};
    nuv_qss[71] = {-1};
    // (63):  HCO + O => CO + OH
    fwd_A[71]     = 30200000000000;
    fwd_beta[71]  = 0;
    fwd_Ea[71]    = 0;
    prefactor_units[71]  = 1.0000000000000002e-06;
    activation_units[71] = 0.50321666580471969;
    phase_units[71]      = pow(10,-12.000000);
    is_PD[71] = 0;
    nTB[71] = 0;

    // (64):  HCO + M => H + CO + M
    kiv[14] = {3,10};
    nuv[14] = {1,1};
    kiv_qss[14] = {1};
    nuv_qss[14] = {-1};
    // (64):  HCO + M => H + CO + M
    fwd_A[14]     = 1.86e+17;
    fwd_beta[14]  = -1;
    fwd_Ea[14]    = 17000;
    prefactor_units[14]  = 1.0000000000000002e-06;
    activation_units[14] = 0.50321666580471969;
    phase_units[14]      = pow(10,-6.000000);
    is_PD[14] = 0;
    nTB[14] = 4;
    TB[14] = (double *) malloc(4 * sizeof(double));
    TBid[14] = (int *) malloc(4 * sizeof(int));
    TBid[14][0] = 10; TB[14][0] = 1.8999999999999999; // CO
    TBid[14][1] = 2; TB[14][1] = 2.5; // H2
    TBid[14][2] = 5; TB[14][2] = 6; // H2O
    TBid[14][3] = 9; TB[14][3] = 3.7999999999999998; // CO2

    // (65):  HCO + CH3 => CH4 + CO
    kiv[72] = {12,13,10};
    nuv[72] = {-1,1,1};
    kiv_qss[72] = {1};
    nuv_qss[72] = {-1};
    // (65):  HCO + CH3 => CH4 + CO
    fwd_A[72]     = 121000000000000;
    fwd_beta[72]  = 0;
    fwd_Ea[72]    = 0;
    prefactor_units[72]  = 1.0000000000000002e-06;
    activation_units[72] = 0.50321666580471969;
    phase_units[72]      = pow(10,-12.000000);
    is_PD[72] = 0;
    nTB[72] = 0;

    // (66):  CH2O + OH => HCO + H2O
    kiv[73] = {11,4,5};
    nuv[73] = {-1,-1,1};
    kiv_qss[73] = {1};
    nuv_qss[73] = {1};
    // (66):  CH2O + OH => HCO + H2O
    fwd_A[73]     = 3430000000;
    fwd_beta[73]  = 1.1799999999999999;
    fwd_Ea[73]    = -446.94;
    prefactor_units[73]  = 1.0000000000000002e-06;
    activation_units[73] = 0.50321666580471969;
    phase_units[73]      = pow(10,-12.000000);
    is_PD[73] = 0;
    nTB[73] = 0;

    // (67):  CH2O + O => HCO + OH
    kiv[74] = {11,1,4};
    nuv[74] = {-1,-1,1};
    kiv_qss[74] = {1};
    nuv_qss[74] = {1};
    // (67):  CH2O + O => HCO + OH
    fwd_A[74]     = 416000000000;
    fwd_beta[74]  = 0.56999999999999995;
    fwd_Ea[74]    = 2761.9499999999998;
    prefactor_units[74]  = 1.0000000000000002e-06;
    activation_units[74] = 0.50321666580471969;
    phase_units[74]      = pow(10,-12.000000);
    is_PD[74] = 0;
    nTB[74] = 0;

    // (68):  CH2O + H => HCO + H2
    kiv[75] = {11,3,2};
    nuv[75] = {-1,-1,1};
    kiv_qss[75] = {1};
    nuv_qss[75] = {1};
    // (68):  CH2O + H => HCO + H2
    fwd_A[75]     = 933400000;
    fwd_beta[75]  = 1.5;
    fwd_Ea[75]    = 2976.0999999999999;
    prefactor_units[75]  = 1.0000000000000002e-06;
    activation_units[75] = 0.50321666580471969;
    phase_units[75]      = pow(10,-12.000000);
    is_PD[75] = 0;
    nTB[75] = 0;

    // (69):  CH2O + CH3 => HCO + CH4
    kiv[76] = {11,12,13};
    nuv[76] = {-1,-1,1};
    kiv_qss[76] = {1};
    nuv_qss[76] = {1};
    // (69):  CH2O + CH3 => HCO + CH4
    fwd_A[76]     = 3.636e-06;
    fwd_beta[76]  = 5.4199999999999999;
    fwd_Ea[76]    = 998.09000000000003;
    prefactor_units[76]  = 1.0000000000000002e-06;
    activation_units[76] = 0.50321666580471969;
    phase_units[76]      = pow(10,-12.000000);
    is_PD[76] = 0;
    nTB[76] = 0;

    // (70):  2.000000 CH3O => CH3OH + CH2O
    kiv[77] = {14,11};
    nuv[77] = {1,1};
    kiv_qss[77] = {4};
    nuv_qss[77] = {-2.0};
    // (70):  2.000000 CH3O => CH3OH + CH2O
    fwd_A[77]     = 60300000000000;
    fwd_beta[77]  = 0;
    fwd_Ea[77]    = 0;
    prefactor_units[77]  = 1.0000000000000002e-06;
    activation_units[77] = 0.50321666580471969;
    phase_units[77]      = pow(10,-12.000000);
    is_PD[77] = 0;
    nTB[77] = 0;

    // (71):  CH3O + O2 => CH2O + HO2
    kiv[78] = {6,11,7};
    nuv[78] = {-1,1,1};
    kiv_qss[78] = {4};
    nuv_qss[78] = {-1};
    // (71):  CH3O + O2 => CH2O + HO2
    fwd_A[78]     = 55000000000;
    fwd_beta[78]  = 0;
    fwd_Ea[78]    = 2424;
    prefactor_units[78]  = 1.0000000000000002e-06;
    activation_units[78] = 0.50321666580471969;
    phase_units[78]      = pow(10,-12.000000);
    is_PD[78] = 0;
    nTB[78] = 0;

    // (72):  CH3O (+M) => CH2O + H (+M)
    kiv[7] = {11,3};
    nuv[7] = {1,1};
    kiv_qss[7] = {4};
    nuv_qss[7] = {-1};
    // (72):  CH3O (+M) => CH2O + H (+M)
    fwd_A[7]     = 54500000000000;
    fwd_beta[7]  = 0;
    fwd_Ea[7]    = 13500;
    low_A[7]     = 2.344e+25;
    low_beta[7]  = -2.7000000000000002;
    low_Ea[7]    = 30599.900000000001;
    troe_a[7]    = 1;
    troe_Tsss[7] = 1;
    troe_Ts[7]   = 10000000;
    troe_Tss[7]  = 10000000;
    troe_len[7]  = 4;
    prefactor_units[7]  = 1;
    activation_units[7] = 0.50321666580471969;
    phase_units[7]      = pow(10,-6.000000);
    is_PD[7] = 1;
    nTB[7] = 0;

    // (73):  CH3O + H2 => CH3OH + H
    kiv[79] = {2,14,3};
    nuv[79] = {-1,1,1};
    kiv_qss[79] = {4};
    nuv_qss[79] = {-1};
    // (73):  CH3O + H2 => CH3OH + H
    fwd_A[79]     = 7467000000000;
    fwd_beta[79]  = -0.02;
    fwd_Ea[79]    = 7825.0500000000002;
    prefactor_units[79]  = 1.0000000000000002e-06;
    activation_units[79] = 0.50321666580471969;
    phase_units[79]      = pow(10,-12.000000);
    is_PD[79] = 0;
    nTB[79] = 0;

    // (74):  CH3OH + OH => CH3O + H2O
    kiv[80] = {14,4,5};
    nuv[80] = {-1,-1,1};
    kiv_qss[80] = {4};
    nuv_qss[80] = {1};
    // (74):  CH3OH + OH => CH3O + H2O
    fwd_A[80]     = 1000000;
    fwd_beta[80]  = 2.1000000000000001;
    fwd_Ea[80]    = 496.64999999999998;
    prefactor_units[80]  = 1.0000000000000002e-06;
    activation_units[80] = 0.50321666580471969;
    phase_units[80]      = pow(10,-12.000000);
    is_PD[80] = 0;
    nTB[80] = 0;

    // (75):  CH2GSG + CO2 => CH2O + CO
    kiv[81] = {9,11,10};
    nuv[81] = {-1,1,1};
    kiv_qss[81] = {3};
    nuv_qss[81] = {-1};
    // (75):  CH2GSG + CO2 => CH2O + CO
    fwd_A[81]     = 3000000000000;
    fwd_beta[81]  = 0;
    fwd_Ea[81]    = 0;
    prefactor_units[81]  = 1.0000000000000002e-06;
    activation_units[81] = 0.50321666580471969;
    phase_units[81]      = pow(10,-12.000000);
    is_PD[81] = 0;
    nTB[81] = 0;

    // (76):  HOCHO + H => H2 + CO + OH
    kiv[82] = {17,3,2,10,4};
    nuv[82] = {-1,-1,1,1,1};
    kiv_qss[82] = {};
    nuv_qss[82] = {};
    // (76):  HOCHO + H => H2 + CO + OH
    fwd_A[82]     = 60300000000000;
    fwd_beta[82]  = -0.34999999999999998;
    fwd_Ea[82]    = 2988.0500000000002;
    prefactor_units[82]  = 1.0000000000000002e-06;
    activation_units[82] = 0.50321666580471969;
    phase_units[82]      = pow(10,-12.000000);
    is_PD[82] = 0;
    nTB[82] = 0;

    // (77):  HOCHO + OH => H2O + CO + OH
    kiv[83] = {17,4,5,10,4};
    nuv[83] = {-1,-1,1,1,1};
    kiv_qss[83] = {};
    nuv_qss[83] = {};
    // (77):  HOCHO + OH => H2O + CO + OH
    fwd_A[83]     = 18500000;
    fwd_beta[83]  = 1.51;
    fwd_Ea[83]    = -962;
    prefactor_units[83]  = 1.0000000000000002e-06;
    activation_units[83] = 0.50321666580471969;
    phase_units[83]      = pow(10,-12.000000);
    is_PD[83] = 0;
    nTB[83] = 0;

    // (78):  HOCHO => HCO + OH
    kiv[84] = {17,4};
    nuv[84] = {-1,1};
    kiv_qss[84] = {1};
    nuv_qss[84] = {1};
    // (78):  HOCHO => HCO + OH
    fwd_A[84]     = 4.593e+18;
    fwd_beta[84]  = -0.46000000000000002;
    fwd_Ea[84]    = 108299.95;
    prefactor_units[84]  = 1;
    activation_units[84] = 0.50321666580471969;
    phase_units[84]      = pow(10,-6.000000);
    is_PD[84] = 0;
    nTB[84] = 0;

    // (79):  HCO + OH => HOCHO
    kiv[85] = {4,17};
    nuv[85] = {-1,1};
    kiv_qss[85] = {1};
    nuv_qss[85] = {-1};
    // (79):  HCO + OH => HOCHO
    fwd_A[85]     = 100000000000000;
    fwd_beta[85]  = 0;
    fwd_Ea[85]    = 0;
    prefactor_units[85]  = 1.0000000000000002e-06;
    activation_units[85] = 0.50321666580471969;
    phase_units[85]      = pow(10,-12.000000);
    is_PD[85] = 0;
    nTB[85] = 0;

    // (80):  HOCHO + H => H2 + CO2 + H
    kiv[86] = {17,3,2,9,3};
    nuv[86] = {-1,-1,1,1,1};
    kiv_qss[86] = {};
    nuv_qss[86] = {};
    // (80):  HOCHO + H => H2 + CO2 + H
    fwd_A[86]     = 4240000;
    fwd_beta[86]  = 2.1000000000000001;
    fwd_Ea[86]    = 4868.0699999999997;
    prefactor_units[86]  = 1.0000000000000002e-06;
    activation_units[86] = 0.50321666580471969;
    phase_units[86]      = pow(10,-12.000000);
    is_PD[86] = 0;
    nTB[86] = 0;

    // (81):  HOCHO + OH => H2O + CO2 + H
    kiv[87] = {17,4,5,9,3};
    nuv[87] = {-1,-1,1,1,1};
    kiv_qss[87] = {};
    nuv_qss[87] = {};
    // (81):  HOCHO + OH => H2O + CO2 + H
    fwd_A[87]     = 2620000;
    fwd_beta[87]  = 2.0600000000000001;
    fwd_Ea[87]    = 916.11000000000001;
    prefactor_units[87]  = 1.0000000000000002e-06;
    activation_units[87] = 0.50321666580471969;
    phase_units[87]      = pow(10,-12.000000);
    is_PD[87] = 0;
    nTB[87] = 0;

    // (82):  CH3O2 + CH3 => 2.000000 CH3O
    kiv[88] = {12};
    nuv[88] = {-1};
    kiv_qss[88] = {6,4};
    nuv_qss[88] = {-1,2.0};
    // (82):  CH3O2 + CH3 => 2.000000 CH3O
    fwd_A[88]     = 7000000000000;
    fwd_beta[88]  = 0;
    fwd_Ea[88]    = -1000;
    prefactor_units[88]  = 1.0000000000000002e-06;
    activation_units[88] = 0.50321666580471969;
    phase_units[88]      = pow(10,-12.000000);
    is_PD[88] = 0;
    nTB[88] = 0;

    // (83):  CH3O2 + HO2 => CH3O2H + O2
    kiv[89] = {7,6};
    nuv[89] = {-1,1};
    kiv_qss[89] = {6,7};
    nuv_qss[89] = {-1,1};
    // (83):  CH3O2 + HO2 => CH3O2H + O2
    fwd_A[89]     = 17500000000;
    fwd_beta[89]  = 0;
    fwd_Ea[89]    = -3275.0999999999999;
    prefactor_units[89]  = 1.0000000000000002e-06;
    activation_units[89] = 0.50321666580471969;
    phase_units[89]      = pow(10,-12.000000);
    is_PD[89] = 0;
    nTB[89] = 0;

    // (84):  CH3O2 + M => CH3 + O2 + M
    kiv[15] = {12,6};
    nuv[15] = {1,1};
    kiv_qss[15] = {6};
    nuv_qss[15] = {-1};
    // (84):  CH3O2 + M => CH3 + O2 + M
    fwd_A[15]     = 4.3430000000000002e+27;
    fwd_beta[15]  = -3.4199999999999999;
    fwd_Ea[15]    = 30469.889999999999;
    prefactor_units[15]  = 1.0000000000000002e-06;
    activation_units[15] = 0.50321666580471969;
    phase_units[15]      = pow(10,-6.000000);
    is_PD[15] = 0;
    nTB[15] = 0;

    // (85):  CH3 + O2 + M => CH3O2 + M
    kiv[16] = {12,6};
    nuv[16] = {-1,-1};
    kiv_qss[16] = {6};
    nuv_qss[16] = {1};
    // (85):  CH3 + O2 + M => CH3O2 + M
    fwd_A[16]     = 5.4400000000000001e+25;
    fwd_beta[16]  = -3.2999999999999998;
    fwd_Ea[16]    = 0;
    prefactor_units[16]  = 1.0000000000000002e-12;
    activation_units[16] = 0.50321666580471969;
    phase_units[16]      = pow(10,-12.000000);
    is_PD[16] = 0;
    nTB[16] = 0;

    // (86):  CH3O2H => CH3O + OH
    kiv[90] = {4};
    nuv[90] = {1};
    kiv_qss[90] = {7,4};
    nuv_qss[90] = {-1,1};
    // (86):  CH3O2H => CH3O + OH
    fwd_A[90]     = 631000000000000;
    fwd_beta[90]  = 0;
    fwd_Ea[90]    = 42299.949999999997;
    prefactor_units[90]  = 1;
    activation_units[90] = 0.50321666580471969;
    phase_units[90]      = pow(10,-6.000000);
    is_PD[90] = 0;
    nTB[90] = 0;

    // (87):  C2H2 + O => CH2 + CO
    kiv[91] = {18,1,10};
    nuv[91] = {-1,-1,1};
    kiv_qss[91] = {2};
    nuv_qss[91] = {1};
    // (87):  C2H2 + O => CH2 + CO
    fwd_A[91]     = 6120000;
    fwd_beta[91]  = 2;
    fwd_Ea[91]    = 1900.0999999999999;
    prefactor_units[91]  = 1.0000000000000002e-06;
    activation_units[91] = 0.50321666580471969;
    phase_units[91]      = pow(10,-12.000000);
    is_PD[91] = 0;
    nTB[91] = 0;

    // (88):  C2H2 + O => HCCO + H
    kiv[92] = {18,1,3};
    nuv[92] = {-1,-1,1};
    kiv_qss[92] = {8};
    nuv_qss[92] = {1};
    // (88):  C2H2 + O => HCCO + H
    fwd_A[92]     = 14300000;
    fwd_beta[92]  = 2;
    fwd_Ea[92]    = 1900.0999999999999;
    prefactor_units[92]  = 1.0000000000000002e-06;
    activation_units[92] = 0.50321666580471969;
    phase_units[92]      = pow(10,-12.000000);
    is_PD[92] = 0;
    nTB[92] = 0;

    // (89):  C2H3 + H => C2H2 + H2
    kiv[93] = {3,18,2};
    nuv[93] = {-1,1,1};
    kiv_qss[93] = {9};
    nuv_qss[93] = {-1};
    // (89):  C2H3 + H => C2H2 + H2
    fwd_A[93]     = 20000000000000;
    fwd_beta[93]  = 0;
    fwd_Ea[93]    = 2500;
    prefactor_units[93]  = 1.0000000000000002e-06;
    activation_units[93] = 0.50321666580471969;
    phase_units[93]      = pow(10,-12.000000);
    is_PD[93] = 0;
    nTB[93] = 0;

    // (90):  C2H3 + O2 => CH2CHO + O
    kiv[94] = {6,1};
    nuv[94] = {-1,1};
    kiv_qss[94] = {9,10};
    nuv_qss[94] = {-1,1};
    // (90):  C2H3 + O2 => CH2CHO + O
    fwd_A[94]     = 350000000000000;
    fwd_beta[94]  = -0.60999999999999999;
    fwd_Ea[94]    = 5260.04;
    prefactor_units[94]  = 1.0000000000000002e-06;
    activation_units[94] = 0.50321666580471969;
    phase_units[94]      = pow(10,-12.000000);
    is_PD[94] = 0;
    nTB[94] = 0;

    // (91):  C2H3 + CH3 => C3H6
    kiv[95] = {12,19};
    nuv[95] = {-1,1};
    kiv_qss[95] = {9};
    nuv_qss[95] = {-1};
    // (91):  C2H3 + CH3 => C3H6
    fwd_A[95]     = 4.7119999999999996e+59;
    fwd_beta[95]  = -13.19;
    fwd_Ea[95]    = 29539.91;
    prefactor_units[95]  = 1.0000000000000002e-06;
    activation_units[95] = 0.50321666580471969;
    phase_units[95]      = pow(10,-12.000000);
    is_PD[95] = 0;
    nTB[95] = 0;

    // (92):  C2H3 + O2 => C2H2 + HO2
    kiv[96] = {6,18,7};
    nuv[96] = {-1,1,1};
    kiv_qss[96] = {9};
    nuv_qss[96] = {-1};
    // (92):  C2H3 + O2 => C2H2 + HO2
    fwd_A[96]     = 2.12e-06;
    fwd_beta[96]  = 6;
    fwd_Ea[96]    = 9483.9899999999998;
    prefactor_units[96]  = 1.0000000000000002e-06;
    activation_units[96] = 0.50321666580471969;
    phase_units[96]      = pow(10,-12.000000);
    is_PD[96] = 0;
    nTB[96] = 0;

    // (93):  C2H3 + O2 => CH2O + HCO
    kiv[97] = {6,11};
    nuv[97] = {-1,1};
    kiv_qss[97] = {9,1};
    nuv_qss[97] = {-1,1};
    // (93):  C2H3 + O2 => CH2O + HCO
    fwd_A[97]     = 1.6999999999999999e+29;
    fwd_beta[97]  = -5.3099999999999996;
    fwd_Ea[97]    = 6500;
    prefactor_units[97]  = 1.0000000000000002e-06;
    activation_units[97] = 0.50321666580471969;
    phase_units[97]      = pow(10,-12.000000);
    is_PD[97] = 0;
    nTB[97] = 0;

    // (94):  C2H3 (+M) => H + C2H2 (+M)
    kiv[8] = {3,18};
    nuv[8] = {1,1};
    kiv_qss[8] = {9};
    nuv_qss[8] = {-1};
    // (94):  C2H3 (+M) => H + C2H2 (+M)
    fwd_A[8]     = 16060000000;
    fwd_beta[8]  = 1.028;
    fwd_Ea[8]    = 40503.589999999997;
    low_A[8]     = 1.164e+39;
    low_beta[8]  = -6.8209999999999997;
    low_Ea[8]    = 44491.629999999997;
    troe_a[8]    = 1;
    troe_Tsss[8] = 0;
    troe_Ts[8]   = 675;
    troe_Tss[8]  = 1000000000000000;
    troe_len[8]  = 4;
    prefactor_units[8]  = 1;
    activation_units[8] = 0.50321666580471969;
    phase_units[8]      = pow(10,-6.000000);
    is_PD[8] = 1;
    nTB[8] = 4;
    TB[8] = (double *) malloc(4 * sizeof(double));
    TBid[8] = (int *) malloc(4 * sizeof(int));
    TBid[8][0] = 10; TB[8][0] = 2; // CO
    TBid[8][1] = 2; TB[8][1] = 2; // H2
    TBid[8][2] = 5; TB[8][2] = 5; // H2O
    TBid[8][3] = 9; TB[8][3] = 3; // CO2

    // (95):  C2H4 + CH3 => C2H3 + CH4
    kiv[98] = {20,12,13};
    nuv[98] = {-1,-1,1};
    kiv_qss[98] = {9};
    nuv_qss[98] = {1};
    // (95):  C2H4 + CH3 => C2H3 + CH4
    fwd_A[98]     = 6.6200000000000001;
    fwd_beta[98]  = 3.7000000000000002;
    fwd_Ea[98]    = 9500;
    prefactor_units[98]  = 1.0000000000000002e-06;
    activation_units[98] = 0.50321666580471969;
    phase_units[98]      = pow(10,-12.000000);
    is_PD[98] = 0;
    nTB[98] = 0;

    // (96):  C2H4 + O => CH3 + HCO
    kiv[99] = {20,1,12};
    nuv[99] = {-1,-1,1};
    kiv_qss[99] = {1};
    nuv_qss[99] = {1};
    // (96):  C2H4 + O => CH3 + HCO
    fwd_A[99]     = 10200000;
    fwd_beta[99]  = 1.8799999999999999;
    fwd_Ea[99]    = 179.02000000000001;
    prefactor_units[99]  = 1.0000000000000002e-06;
    activation_units[99] = 0.50321666580471969;
    phase_units[99]      = pow(10,-12.000000);
    is_PD[99] = 0;
    nTB[99] = 0;

    // (97):  C2H4 + OH => C2H3 + H2O
    kiv[100] = {20,4,5};
    nuv[100] = {-1,-1,1};
    kiv_qss[100] = {9};
    nuv_qss[100] = {1};
    // (97):  C2H4 + OH => C2H3 + H2O
    fwd_A[100]     = 20500000000000;
    fwd_beta[100]  = 0;
    fwd_Ea[100]    = 5950.0500000000002;
    prefactor_units[100]  = 1.0000000000000002e-06;
    activation_units[100] = 0.50321666580471969;
    phase_units[100]      = pow(10,-12.000000);
    is_PD[100] = 0;
    nTB[100] = 0;

    // (98):  H + C2H4 (+M) <=> C2H5 (+M)
    kiv[9] = {3,20};
    nuv[9] = {-1,-1};
    kiv_qss[9] = {5};
    nuv_qss[9] = {1};
    // (98):  H + C2H4 (+M) <=> C2H5 (+M)
    fwd_A[9]     = 1081000000000;
    fwd_beta[9]  = 0.45000000000000001;
    fwd_Ea[9]    = 1821.9400000000001;
    low_A[9]     = 1.1120000000000001e+34;
    low_beta[9]  = -5;
    low_Ea[9]    = 4447.8999999999996;
    troe_a[9]    = 1;
    troe_Tsss[9] = 0;
    troe_Ts[9]   = 95;
    troe_Tss[9]  = 200;
    troe_len[9]  = 4;
    prefactor_units[9]  = 1.0000000000000002e-06;
    activation_units[9] = 0.50321666580471969;
    phase_units[9]      = pow(10,-12.000000);
    is_PD[9] = 1;
    nTB[9] = 0;

    // (99):  C2H4 + O => CH2CHO + H
    kiv[101] = {20,1,3};
    nuv[101] = {-1,-1,1};
    kiv_qss[101] = {10};
    nuv_qss[101] = {1};
    // (99):  C2H4 + O => CH2CHO + H
    fwd_A[101]     = 3390000;
    fwd_beta[101]  = 1.8799999999999999;
    fwd_Ea[101]    = 179.02000000000001;
    prefactor_units[101]  = 1.0000000000000002e-06;
    activation_units[101] = 0.50321666580471969;
    phase_units[101]      = pow(10,-12.000000);
    is_PD[101] = 0;
    nTB[101] = 0;

    // (100):  C2H4 + H => C2H3 + H2
    kiv[102] = {20,3,2};
    nuv[102] = {-1,-1,1};
    kiv_qss[102] = {9};
    nuv_qss[102] = {1};
    // (100):  C2H4 + H => C2H3 + H2
    fwd_A[102]     = 0.0084200000000000004;
    fwd_beta[102]  = 4.6200000000000001;
    fwd_Ea[102]    = 2582.9299999999998;
    prefactor_units[102]  = 1.0000000000000002e-06;
    activation_units[102] = 0.50321666580471969;
    phase_units[102]      = pow(10,-12.000000);
    is_PD[102] = 0;
    nTB[102] = 0;

    // (101):  C2H3 + H2 => C2H4 + H
    kiv[103] = {2,20,3};
    nuv[103] = {-1,1,1};
    kiv_qss[103] = {9};
    nuv_qss[103] = {-1};
    // (101):  C2H3 + H2 => C2H4 + H
    fwd_A[103]     = 0.57230000000000003;
    fwd_beta[103]  = 3.79;
    fwd_Ea[103]    = 3233.0300000000002;
    prefactor_units[103]  = 1.0000000000000002e-06;
    activation_units[103] = 0.50321666580471969;
    phase_units[103]      = pow(10,-12.000000);
    is_PD[103] = 0;
    nTB[103] = 0;

    // (102):  H + C2H5 => C2H6
    kiv[104] = {3,15};
    nuv[104] = {-1,1};
    kiv_qss[104] = {5};
    nuv_qss[104] = {-1};
    // (102):  H + C2H5 => C2H6
    fwd_A[104]     = 583100000000;
    fwd_beta[104]  = 0.59899999999999998;
    fwd_Ea[104]    = -2913;
    prefactor_units[104]  = 1.0000000000000002e-06;
    activation_units[104] = 0.50321666580471969;
    phase_units[104]      = pow(10,-12.000000);
    is_PD[104] = 0;
    nTB[104] = 0;

    // (103):  C2H5 + HO2 => C2H5O + OH
    kiv[105] = {7,4};
    nuv[105] = {-1,1};
    kiv_qss[105] = {5,11};
    nuv_qss[105] = {-1,1};
    // (103):  C2H5 + HO2 => C2H5O + OH
    fwd_A[105]     = 32000000000000;
    fwd_beta[105]  = 0;
    fwd_Ea[105]    = 0;
    prefactor_units[105]  = 1.0000000000000002e-06;
    activation_units[105] = 0.50321666580471969;
    phase_units[105]      = pow(10,-12.000000);
    is_PD[105] = 0;
    nTB[105] = 0;

    // (104):  C2H5 + O2 => C2H4 + HO2
    kiv[106] = {6,20,7};
    nuv[106] = {-1,1,1};
    kiv_qss[106] = {5};
    nuv_qss[106] = {-1};
    // (104):  C2H5 + O2 => C2H4 + HO2
    fwd_A[106]     = 1.22e+30;
    fwd_beta[106]  = -5.7599999999999998;
    fwd_Ea[106]    = 10099.9;
    prefactor_units[106]  = 1.0000000000000002e-06;
    activation_units[106] = 0.50321666580471969;
    phase_units[106]      = pow(10,-12.000000);
    is_PD[106] = 0;
    nTB[106] = 0;

    // (105):  C2H6 + O => C2H5 + OH
    kiv[107] = {15,1,4};
    nuv[107] = {-1,-1,1};
    kiv_qss[107] = {5};
    nuv_qss[107] = {1};
    // (105):  C2H6 + O => C2H5 + OH
    fwd_A[107]     = 13000000;
    fwd_beta[107]  = 2.1299999999999999;
    fwd_Ea[107]    = 5190.0100000000002;
    prefactor_units[107]  = 1.0000000000000002e-06;
    activation_units[107] = 0.50321666580471969;
    phase_units[107]      = pow(10,-12.000000);
    is_PD[107] = 0;
    nTB[107] = 0;

    // (106):  C2H6 + OH => C2H5 + H2O
    kiv[108] = {15,4,5};
    nuv[108] = {-1,-1,1};
    kiv_qss[108] = {5};
    nuv_qss[108] = {1};
    // (106):  C2H6 + OH => C2H5 + H2O
    fwd_A[108]     = 58000000;
    fwd_beta[108]  = 1.73;
    fwd_Ea[108]    = 1159.8900000000001;
    prefactor_units[108]  = 1.0000000000000002e-06;
    activation_units[108] = 0.50321666580471969;
    phase_units[108]      = pow(10,-12.000000);
    is_PD[108] = 0;
    nTB[108] = 0;

    // (107):  C2H6 + H => C2H5 + H2
    kiv[109] = {15,3,2};
    nuv[109] = {-1,-1,1};
    kiv_qss[109] = {5};
    nuv_qss[109] = {1};
    // (107):  C2H6 + H => C2H5 + H2
    fwd_A[109]     = 554;
    fwd_beta[109]  = 3.5;
    fwd_Ea[109]    = 5167.0699999999997;
    prefactor_units[109]  = 1.0000000000000002e-06;
    activation_units[109] = 0.50321666580471969;
    phase_units[109]      = pow(10,-12.000000);
    is_PD[109] = 0;
    nTB[109] = 0;

    // (108):  HCCO + O => H + 2.000000 CO
    kiv[110] = {1,3,10};
    nuv[110] = {-1,1,2.0};
    kiv_qss[110] = {8};
    nuv_qss[110] = {-1};
    // (108):  HCCO + O => H + 2.000000 CO
    fwd_A[110]     = 80000000000000;
    fwd_beta[110]  = 0;
    fwd_Ea[110]    = 0;
    prefactor_units[110]  = 1.0000000000000002e-06;
    activation_units[110] = 0.50321666580471969;
    phase_units[110]      = pow(10,-12.000000);
    is_PD[110] = 0;
    nTB[110] = 0;

    // (109):  HCCO + OH => 2.000000 HCO
    kiv[111] = {4};
    nuv[111] = {-1};
    kiv_qss[111] = {8,1};
    nuv_qss[111] = {-1,2.0};
    // (109):  HCCO + OH => 2.000000 HCO
    fwd_A[111]     = 10000000000000;
    fwd_beta[111]  = 0;
    fwd_Ea[111]    = 0;
    prefactor_units[111]  = 1.0000000000000002e-06;
    activation_units[111] = 0.50321666580471969;
    phase_units[111]      = pow(10,-12.000000);
    is_PD[111] = 0;
    nTB[111] = 0;

    // (110):  HCCO + O2 => CO2 + HCO
    kiv[112] = {6,9};
    nuv[112] = {-1,1};
    kiv_qss[112] = {8,1};
    nuv_qss[112] = {-1,1};
    // (110):  HCCO + O2 => CO2 + HCO
    fwd_A[112]     = 240000000000;
    fwd_beta[112]  = 0;
    fwd_Ea[112]    = -853.97000000000003;
    prefactor_units[112]  = 1.0000000000000002e-06;
    activation_units[112] = 0.50321666580471969;
    phase_units[112]      = pow(10,-12.000000);
    is_PD[112] = 0;
    nTB[112] = 0;

    // (111):  HCCO + H => CH2GSG + CO
    kiv[113] = {3,10};
    nuv[113] = {-1,1};
    kiv_qss[113] = {8,3};
    nuv_qss[113] = {-1,1};
    // (111):  HCCO + H => CH2GSG + CO
    fwd_A[113]     = 110000000000000;
    fwd_beta[113]  = 0;
    fwd_Ea[113]    = 0;
    prefactor_units[113]  = 1.0000000000000002e-06;
    activation_units[113] = 0.50321666580471969;
    phase_units[113]      = pow(10,-12.000000);
    is_PD[113] = 0;
    nTB[113] = 0;

    // (112):  CH2GSG + CO => HCCO + H
    kiv[114] = {10,3};
    nuv[114] = {-1,1};
    kiv_qss[114] = {3,8};
    nuv_qss[114] = {-1,1};
    // (112):  CH2GSG + CO => HCCO + H
    fwd_A[114]     = 2046000000000;
    fwd_beta[114]  = 0.89000000000000001;
    fwd_Ea[114]    = 27830.07;
    prefactor_units[114]  = 1.0000000000000002e-06;
    activation_units[114] = 0.50321666580471969;
    phase_units[114]      = pow(10,-12.000000);
    is_PD[114] = 0;
    nTB[114] = 0;

    // (113):  CH2CO + O => HCCO + OH
    kiv[115] = {16,1,4};
    nuv[115] = {-1,-1,1};
    kiv_qss[115] = {8};
    nuv_qss[115] = {1};
    // (113):  CH2CO + O => HCCO + OH
    fwd_A[115]     = 10000000000000;
    fwd_beta[115]  = 0;
    fwd_Ea[115]    = 8000;
    prefactor_units[115]  = 1.0000000000000002e-06;
    activation_units[115] = 0.50321666580471969;
    phase_units[115]      = pow(10,-12.000000);
    is_PD[115] = 0;
    nTB[115] = 0;

    // (114):  CH2CO + H => HCCO + H2
    kiv[116] = {16,3,2};
    nuv[116] = {-1,-1,1};
    kiv_qss[116] = {8};
    nuv_qss[116] = {1};
    // (114):  CH2CO + H => HCCO + H2
    fwd_A[116]     = 200000000000000;
    fwd_beta[116]  = 0;
    fwd_Ea[116]    = 8000;
    prefactor_units[116]  = 1.0000000000000002e-06;
    activation_units[116] = 0.50321666580471969;
    phase_units[116]      = pow(10,-12.000000);
    is_PD[116] = 0;
    nTB[116] = 0;

    // (115):  HCCO + H2 => CH2CO + H
    kiv[117] = {2,16,3};
    nuv[117] = {-1,1,1};
    kiv_qss[117] = {8};
    nuv_qss[117] = {-1};
    // (115):  HCCO + H2 => CH2CO + H
    fwd_A[117]     = 652200000000;
    fwd_beta[117]  = 0;
    fwd_Ea[117]    = 840.11000000000001;
    prefactor_units[117]  = 1.0000000000000002e-06;
    activation_units[117] = 0.50321666580471969;
    phase_units[117]      = pow(10,-12.000000);
    is_PD[117] = 0;
    nTB[117] = 0;

    // (116):  CH2CO + H => CH3 + CO
    kiv[118] = {16,3,12,10};
    nuv[118] = {-1,-1,1,1};
    kiv_qss[118] = {};
    nuv_qss[118] = {};
    // (116):  CH2CO + H => CH3 + CO
    fwd_A[118]     = 11000000000000;
    fwd_beta[118]  = 0;
    fwd_Ea[118]    = 3400.0999999999999;
    prefactor_units[118]  = 1.0000000000000002e-06;
    activation_units[118] = 0.50321666580471969;
    phase_units[118]      = pow(10,-12.000000);
    is_PD[118] = 0;
    nTB[118] = 0;

    // (117):  CH2CO + O => CH2 + CO2
    kiv[119] = {16,1,9};
    nuv[119] = {-1,-1,1};
    kiv_qss[119] = {2};
    nuv_qss[119] = {1};
    // (117):  CH2CO + O => CH2 + CO2
    fwd_A[119]     = 1750000000000;
    fwd_beta[119]  = 0;
    fwd_Ea[119]    = 1349.9000000000001;
    prefactor_units[119]  = 1.0000000000000002e-06;
    activation_units[119] = 0.50321666580471969;
    phase_units[119]      = pow(10,-12.000000);
    is_PD[119] = 0;
    nTB[119] = 0;

    // (118):  CH2CO + OH => HCCO + H2O
    kiv[120] = {16,4,5};
    nuv[120] = {-1,-1,1};
    kiv_qss[120] = {8};
    nuv_qss[120] = {1};
    // (118):  CH2CO + OH => HCCO + H2O
    fwd_A[120]     = 10000000000000;
    fwd_beta[120]  = 0;
    fwd_Ea[120]    = 2000;
    prefactor_units[120]  = 1.0000000000000002e-06;
    activation_units[120] = 0.50321666580471969;
    phase_units[120]      = pow(10,-12.000000);
    is_PD[120] = 0;
    nTB[120] = 0;

    // (119):  CH2CHO + O2 => CH2O + CO + OH
    kiv[121] = {6,11,10,4};
    nuv[121] = {-1,1,1,1};
    kiv_qss[121] = {10};
    nuv_qss[121] = {-1};
    // (119):  CH2CHO + O2 => CH2O + CO + OH
    fwd_A[121]     = 20000000000000;
    fwd_beta[121]  = 0;
    fwd_Ea[121]    = 4200.0500000000002;
    prefactor_units[121]  = 1.0000000000000002e-06;
    activation_units[121] = 0.50321666580471969;
    phase_units[121]      = pow(10,-12.000000);
    is_PD[121] = 0;
    nTB[121] = 0;

    // (120):  CH2CHO => CH2CO + H
    kiv[122] = {16,3};
    nuv[122] = {1,1};
    kiv_qss[122] = {10};
    nuv_qss[122] = {-1};
    // (120):  CH2CHO => CH2CO + H
    fwd_A[122]     = 3094000000000000;
    fwd_beta[122]  = -0.26000000000000001;
    fwd_Ea[122]    = 50820.029999999999;
    prefactor_units[122]  = 1;
    activation_units[122] = 0.50321666580471969;
    phase_units[122]      = pow(10,-6.000000);
    is_PD[122] = 0;
    nTB[122] = 0;

    // (121):  CH2CO + H => CH2CHO
    kiv[123] = {16,3};
    nuv[123] = {-1,-1};
    kiv_qss[123] = {10};
    nuv_qss[123] = {1};
    // (121):  CH2CO + H => CH2CHO
    fwd_A[123]     = 50000000000000;
    fwd_beta[123]  = 0;
    fwd_Ea[123]    = 12299.950000000001;
    prefactor_units[123]  = 1.0000000000000002e-06;
    activation_units[123] = 0.50321666580471969;
    phase_units[123]      = pow(10,-12.000000);
    is_PD[123] = 0;
    nTB[123] = 0;

    // (122):  CH3CO (+M) => CH3 + CO (+M)
    kiv[10] = {12,10};
    nuv[10] = {1,1};
    kiv_qss[10] = {12};
    nuv_qss[10] = {-1};
    // (122):  CH3CO (+M) => CH3 + CO (+M)
    fwd_A[10]     = 3000000000000;
    fwd_beta[10]  = 0;
    fwd_Ea[10]    = 16719.889999999999;
    low_A[10]     = 1200000000000000;
    low_beta[10]  = 0;
    low_Ea[10]    = 12520.08;
    troe_a[10]    = 1;
    troe_Tsss[10] = 1;
    troe_Ts[10]   = 10000000;
    troe_Tss[10]  = 10000000;
    troe_len[10]  = 4;
    prefactor_units[10]  = 1;
    activation_units[10] = 0.50321666580471969;
    phase_units[10]      = pow(10,-6.000000);
    is_PD[10] = 1;
    nTB[10] = 0;

    // (123):  C2H5O + M => CH3 + CH2O + M
    kiv[17] = {12,11};
    nuv[17] = {1,1};
    kiv_qss[17] = {11};
    nuv_qss[17] = {-1};
    // (123):  C2H5O + M => CH3 + CH2O + M
    fwd_A[17]     = 1.35e+38;
    fwd_beta[17]  = -6.96;
    fwd_Ea[17]    = 23799.950000000001;
    prefactor_units[17]  = 1.0000000000000002e-06;
    activation_units[17] = 0.50321666580471969;
    phase_units[17]      = pow(10,-6.000000);
    is_PD[17] = 0;
    nTB[17] = 0;

    // (124):  C2H5O2 => C2H5 + O2
    kiv[124] = {6};
    nuv[124] = {1};
    kiv_qss[124] = {13,5};
    nuv_qss[124] = {-1,1};
    // (124):  C2H5O2 => C2H5 + O2
    fwd_A[124]     = 4.9299999999999996e+50;
    fwd_beta[124]  = -11.5;
    fwd_Ea[124]    = 42250;
    prefactor_units[124]  = 1;
    activation_units[124] = 0.50321666580471969;
    phase_units[124]      = pow(10,-6.000000);
    is_PD[124] = 0;
    nTB[124] = 0;

    // (125):  C2H5 + O2 => C2H5O2
    kiv[125] = {6};
    nuv[125] = {-1};
    kiv_qss[125] = {5,13};
    nuv_qss[125] = {-1,1};
    // (125):  C2H5 + O2 => C2H5O2
    fwd_A[125]     = 1.0900000000000001e+48;
    fwd_beta[125]  = -11.539999999999999;
    fwd_Ea[125]    = 10219.889999999999;
    prefactor_units[125]  = 1.0000000000000002e-06;
    activation_units[125] = 0.50321666580471969;
    phase_units[125]      = pow(10,-12.000000);
    is_PD[125] = 0;
    nTB[125] = 0;

    // (126):  C2H5O2 => C2H4 + HO2
    kiv[126] = {20,7};
    nuv[126] = {1,1};
    kiv_qss[126] = {13};
    nuv_qss[126] = {-1};
    // (126):  C2H5O2 => C2H4 + HO2
    fwd_A[126]     = 3.3700000000000002e+55;
    fwd_beta[126]  = -13.42;
    fwd_Ea[126]    = 44669.93;
    prefactor_units[126]  = 1;
    activation_units[126] = 0.50321666580471969;
    phase_units[126]      = pow(10,-6.000000);
    is_PD[126] = 0;
    nTB[126] = 0;

    // (127):  C3H2 + O2 => HCCO + CO + H
    kiv[127] = {6,10,3};
    nuv[127] = {-1,1,1};
    kiv_qss[127] = {14,8};
    nuv_qss[127] = {-1,1};
    // (127):  C3H2 + O2 => HCCO + CO + H
    fwd_A[127]     = 50000000000000;
    fwd_beta[127]  = 0;
    fwd_Ea[127]    = 0;
    prefactor_units[127]  = 1.0000000000000002e-06;
    activation_units[127] = 0.50321666580471969;
    phase_units[127]      = pow(10,-12.000000);
    is_PD[127] = 0;
    nTB[127] = 0;

    // (128):  C3H2 + OH => C2H2 + HCO
    kiv[128] = {4,18};
    nuv[128] = {-1,1};
    kiv_qss[128] = {14,1};
    nuv_qss[128] = {-1,1};
    // (128):  C3H2 + OH => C2H2 + HCO
    fwd_A[128]     = 50000000000000;
    fwd_beta[128]  = 0;
    fwd_Ea[128]    = 0;
    prefactor_units[128]  = 1.0000000000000002e-06;
    activation_units[128] = 0.50321666580471969;
    phase_units[128]      = pow(10,-12.000000);
    is_PD[128] = 0;
    nTB[128] = 0;

    // (129):  C3H3 + O2 => CH2CO + HCO
    kiv[129] = {6,16};
    nuv[129] = {-1,1};
    kiv_qss[129] = {15,1};
    nuv_qss[129] = {-1,1};
    // (129):  C3H3 + O2 => CH2CO + HCO
    fwd_A[129]     = 30100000000;
    fwd_beta[129]  = 0;
    fwd_Ea[129]    = 2869.98;
    prefactor_units[129]  = 1.0000000000000002e-06;
    activation_units[129] = 0.50321666580471969;
    phase_units[129]      = pow(10,-12.000000);
    is_PD[129] = 0;
    nTB[129] = 0;

    // (130):  C3H3 + HO2 => C3H4XA + O2
    kiv[130] = {7,21,6};
    nuv[130] = {-1,1,1};
    kiv_qss[130] = {15};
    nuv_qss[130] = {-1};
    // (130):  C3H3 + HO2 => C3H4XA + O2
    fwd_A[130]     = 117500000000;
    fwd_beta[130]  = 0.29999999999999999;
    fwd_Ea[130]    = 38;
    prefactor_units[130]  = 1.0000000000000002e-06;
    activation_units[130] = 0.50321666580471969;
    phase_units[130]      = pow(10,-12.000000);
    is_PD[130] = 0;
    nTB[130] = 0;

    // (131):  C3H3 + H => C3H2 + H2
    kiv[131] = {3,2};
    nuv[131] = {-1,1};
    kiv_qss[131] = {15,14};
    nuv_qss[131] = {-1,1};
    // (131):  C3H3 + H => C3H2 + H2
    fwd_A[131]     = 50000000000000;
    fwd_beta[131]  = 0;
    fwd_Ea[131]    = 0;
    prefactor_units[131]  = 1.0000000000000002e-06;
    activation_units[131] = 0.50321666580471969;
    phase_units[131]      = pow(10,-12.000000);
    is_PD[131] = 0;
    nTB[131] = 0;

    // (132):  C3H3 + OH => C3H2 + H2O
    kiv[132] = {4,5};
    nuv[132] = {-1,1};
    kiv_qss[132] = {15,14};
    nuv_qss[132] = {-1,1};
    // (132):  C3H3 + OH => C3H2 + H2O
    fwd_A[132]     = 10000000000000;
    fwd_beta[132]  = 0;
    fwd_Ea[132]    = 0;
    prefactor_units[132]  = 1.0000000000000002e-06;
    activation_units[132] = 0.50321666580471969;
    phase_units[132]      = pow(10,-12.000000);
    is_PD[132] = 0;
    nTB[132] = 0;

    // (133):  C3H2 + H2O => C3H3 + OH
    kiv[133] = {5,4};
    nuv[133] = {-1,1};
    kiv_qss[133] = {14,15};
    nuv_qss[133] = {-1,1};
    // (133):  C3H2 + H2O => C3H3 + OH
    fwd_A[133]     = 1343000000000000;
    fwd_beta[133]  = 0;
    fwd_Ea[133]    = 15679.969999999999;
    prefactor_units[133]  = 1.0000000000000002e-06;
    activation_units[133] = 0.50321666580471969;
    phase_units[133]      = pow(10,-12.000000);
    is_PD[133] = 0;
    nTB[133] = 0;

    // (134):  C3H4XA + H => C3H3 + H2
    kiv[134] = {21,3,2};
    nuv[134] = {-1,-1,1};
    kiv_qss[134] = {15};
    nuv_qss[134] = {1};
    // (134):  C3H4XA + H => C3H3 + H2
    fwd_A[134]     = 20000000;
    fwd_beta[134]  = 2;
    fwd_Ea[134]    = 5000;
    prefactor_units[134]  = 1.0000000000000002e-06;
    activation_units[134] = 0.50321666580471969;
    phase_units[134]      = pow(10,-12.000000);
    is_PD[134] = 0;
    nTB[134] = 0;

    // (135):  C3H4XA + OH => C3H3 + H2O
    kiv[135] = {21,4,5};
    nuv[135] = {-1,-1,1};
    kiv_qss[135] = {15};
    nuv_qss[135] = {1};
    // (135):  C3H4XA + OH => C3H3 + H2O
    fwd_A[135]     = 10000000;
    fwd_beta[135]  = 2;
    fwd_Ea[135]    = 1000;
    prefactor_units[135]  = 1.0000000000000002e-06;
    activation_units[135] = 0.50321666580471969;
    phase_units[135]      = pow(10,-12.000000);
    is_PD[135] = 0;
    nTB[135] = 0;

    // (136):  C3H4XA + O => C2H4 + CO
    kiv[136] = {21,1,20,10};
    nuv[136] = {-1,-1,1,1};
    kiv_qss[136] = {};
    nuv_qss[136] = {};
    // (136):  C3H4XA + O => C2H4 + CO
    fwd_A[136]     = 7800000000000;
    fwd_beta[136]  = 0;
    fwd_Ea[136]    = 1599.9000000000001;
    prefactor_units[136]  = 1.0000000000000002e-06;
    activation_units[136] = 0.50321666580471969;
    phase_units[136]      = pow(10,-12.000000);
    is_PD[136] = 0;
    nTB[136] = 0;

    // (137):  C3H5XA + H => C3H4XA + H2
    kiv[137] = {3,21,2};
    nuv[137] = {-1,1,1};
    kiv_qss[137] = {16};
    nuv_qss[137] = {-1};
    // (137):  C3H5XA + H => C3H4XA + H2
    fwd_A[137]     = 18100000000000;
    fwd_beta[137]  = 0;
    fwd_Ea[137]    = 0;
    prefactor_units[137]  = 1.0000000000000002e-06;
    activation_units[137] = 0.50321666580471969;
    phase_units[137]      = pow(10,-12.000000);
    is_PD[137] = 0;
    nTB[137] = 0;

    // (138):  C3H5XA + HO2 => C3H6 + O2
    kiv[138] = {7,19,6};
    nuv[138] = {-1,1,1};
    kiv_qss[138] = {16};
    nuv_qss[138] = {-1};
    // (138):  C3H5XA + HO2 => C3H6 + O2
    fwd_A[138]     = 33320000000;
    fwd_beta[138]  = 0.34000000000000002;
    fwd_Ea[138]    = -555.92999999999995;
    prefactor_units[138]  = 1.0000000000000002e-06;
    activation_units[138] = 0.50321666580471969;
    phase_units[138]      = pow(10,-12.000000);
    is_PD[138] = 0;
    nTB[138] = 0;

    // (139):  C3H5XA + H => C3H6
    kiv[139] = {3,19};
    nuv[139] = {-1,1};
    kiv_qss[139] = {16};
    nuv_qss[139] = {-1};
    // (139):  C3H5XA + H => C3H6
    fwd_A[139]     = 4.8869999999999999e+56;
    fwd_beta[139]  = -12.25;
    fwd_Ea[139]    = 28080.07;
    prefactor_units[139]  = 1.0000000000000002e-06;
    activation_units[139] = 0.50321666580471969;
    phase_units[139]      = pow(10,-12.000000);
    is_PD[139] = 0;
    nTB[139] = 0;

    // (140):  C3H5XA => C2H2 + CH3
    kiv[140] = {18,12};
    nuv[140] = {1,1};
    kiv_qss[140] = {16};
    nuv_qss[140] = {-1};
    // (140):  C3H5XA => C2H2 + CH3
    fwd_A[140]     = 2.397e+48;
    fwd_beta[140]  = -9.9000000000000004;
    fwd_Ea[140]    = 82080.070000000007;
    prefactor_units[140]  = 1;
    activation_units[140] = 0.50321666580471969;
    phase_units[140]      = pow(10,-6.000000);
    is_PD[140] = 0;
    nTB[140] = 0;

    // (141):  C3H5XA => C3H4XA + H
    kiv[141] = {21,3};
    nuv[141] = {1,1};
    kiv_qss[141] = {16};
    nuv_qss[141] = {-1};
    // (141):  C3H5XA => C3H4XA + H
    fwd_A[141]     = 6663000000000000;
    fwd_beta[141]  = -0.42999999999999999;
    fwd_Ea[141]    = 63219.889999999999;
    prefactor_units[141]  = 1;
    activation_units[141] = 0.50321666580471969;
    phase_units[141]      = pow(10,-6.000000);
    is_PD[141] = 0;
    nTB[141] = 0;

    // (142):  C3H4XA + H => C3H5XA
    kiv[142] = {21,3};
    nuv[142] = {-1,-1};
    kiv_qss[142] = {16};
    nuv_qss[142] = {1};
    // (142):  C3H4XA + H => C3H5XA
    fwd_A[142]     = 240000000000;
    fwd_beta[142]  = 0.68999999999999995;
    fwd_Ea[142]    = 3006.9299999999998;
    prefactor_units[142]  = 1.0000000000000002e-06;
    activation_units[142] = 0.50321666580471969;
    phase_units[142]      = pow(10,-12.000000);
    is_PD[142] = 0;
    nTB[142] = 0;

    // (143):  C3H5XA + CH2O => C3H6 + HCO
    kiv[143] = {11,19};
    nuv[143] = {-1,1};
    kiv_qss[143] = {16,1};
    nuv_qss[143] = {-1,1};
    // (143):  C3H5XA + CH2O => C3H6 + HCO
    fwd_A[143]     = 630000000;
    fwd_beta[143]  = 1.8999999999999999;
    fwd_Ea[143]    = 18190.009999999998;
    prefactor_units[143]  = 1.0000000000000002e-06;
    activation_units[143] = 0.50321666580471969;
    phase_units[143]      = pow(10,-12.000000);
    is_PD[143] = 0;
    nTB[143] = 0;

    // (144):  C3H6 + H => C2H4 + CH3
    kiv[144] = {19,3,20,12};
    nuv[144] = {-1,-1,1,1};
    kiv_qss[144] = {};
    nuv_qss[144] = {};
    // (144):  C3H6 + H => C2H4 + CH3
    fwd_A[144]     = 4.8299999999999998e+33;
    fwd_beta[144]  = -5.8099999999999996;
    fwd_Ea[144]    = 18500;
    prefactor_units[144]  = 1.0000000000000002e-06;
    activation_units[144] = 0.50321666580471969;
    phase_units[144]      = pow(10,-12.000000);
    is_PD[144] = 0;
    nTB[144] = 0;

    // (145):  C3H6 + H => C3H5XA + H2
    kiv[145] = {19,3,2};
    nuv[145] = {-1,-1,1};
    kiv_qss[145] = {16};
    nuv_qss[145] = {1};
    // (145):  C3H6 + H => C3H5XA + H2
    fwd_A[145]     = 173000;
    fwd_beta[145]  = 2.5;
    fwd_Ea[145]    = 2492.1100000000001;
    prefactor_units[145]  = 1.0000000000000002e-06;
    activation_units[145] = 0.50321666580471969;
    phase_units[145]      = pow(10,-12.000000);
    is_PD[145] = 0;
    nTB[145] = 0;

    // (146):  C3H6 + O => C2H5 + HCO
    kiv[146] = {19,1};
    nuv[146] = {-1,-1};
    kiv_qss[146] = {5,1};
    nuv_qss[146] = {1,1};
    // (146):  C3H6 + O => C2H5 + HCO
    fwd_A[146]     = 15800000;
    fwd_beta[146]  = 1.76;
    fwd_Ea[146]    = -1216.0599999999999;
    prefactor_units[146]  = 1.0000000000000002e-06;
    activation_units[146] = 0.50321666580471969;
    phase_units[146]      = pow(10,-12.000000);
    is_PD[146] = 0;
    nTB[146] = 0;

    // (147):  C3H6 + O => C3H5XA + OH
    kiv[147] = {19,1,4};
    nuv[147] = {-1,-1,1};
    kiv_qss[147] = {16};
    nuv_qss[147] = {1};
    // (147):  C3H6 + O => C3H5XA + OH
    fwd_A[147]     = 524000000000;
    fwd_beta[147]  = 0.69999999999999996;
    fwd_Ea[147]    = 5884.0799999999999;
    prefactor_units[147]  = 1.0000000000000002e-06;
    activation_units[147] = 0.50321666580471969;
    phase_units[147]      = pow(10,-12.000000);
    is_PD[147] = 0;
    nTB[147] = 0;

    // (148):  C3H6 + O => CH2CO + CH3 + H
    kiv[148] = {19,1,16,12,3};
    nuv[148] = {-1,-1,1,1,1};
    kiv_qss[148] = {};
    nuv_qss[148] = {};
    // (148):  C3H6 + O => CH2CO + CH3 + H
    fwd_A[148]     = 25000000;
    fwd_beta[148]  = 1.76;
    fwd_Ea[148]    = 76;
    prefactor_units[148]  = 1.0000000000000002e-06;
    activation_units[148] = 0.50321666580471969;
    phase_units[148]      = pow(10,-12.000000);
    is_PD[148] = 0;
    nTB[148] = 0;

    // (149):  C3H6 + OH => C3H5XA + H2O
    kiv[149] = {19,4,5};
    nuv[149] = {-1,-1,1};
    kiv_qss[149] = {16};
    nuv_qss[149] = {1};
    // (149):  C3H6 + OH => C3H5XA + H2O
    fwd_A[149]     = 3120000;
    fwd_beta[149]  = 2;
    fwd_Ea[149]    = -298.04000000000002;
    prefactor_units[149]  = 1.0000000000000002e-06;
    activation_units[149] = 0.50321666580471969;
    phase_units[149]      = pow(10,-12.000000);
    is_PD[149] = 0;
    nTB[149] = 0;

    // (150):  NXC3H7 + O2 => C3H6 + HO2
    kiv[150] = {6,19,7};
    nuv[150] = {-1,1,1};
    kiv_qss[150] = {17};
    nuv_qss[150] = {-1};
    // (150):  NXC3H7 + O2 => C3H6 + HO2
    fwd_A[150]     = 300000000000;
    fwd_beta[150]  = 0;
    fwd_Ea[150]    = 3000;
    prefactor_units[150]  = 1.0000000000000002e-06;
    activation_units[150] = 0.50321666580471969;
    phase_units[150]      = pow(10,-12.000000);
    is_PD[150] = 0;
    nTB[150] = 0;

    // (151):  NXC3H7 => CH3 + C2H4
    kiv[151] = {12,20};
    nuv[151] = {1,1};
    kiv_qss[151] = {17};
    nuv_qss[151] = {-1};
    // (151):  NXC3H7 => CH3 + C2H4
    fwd_A[151]     = 228400000000000;
    fwd_beta[151]  = -0.55000000000000004;
    fwd_Ea[151]    = 28400.099999999999;
    prefactor_units[151]  = 1;
    activation_units[151] = 0.50321666580471969;
    phase_units[151]      = pow(10,-6.000000);
    is_PD[151] = 0;
    nTB[151] = 0;

    // (152):  CH3 + C2H4 => NXC3H7
    kiv[152] = {12,20};
    nuv[152] = {-1,-1};
    kiv_qss[152] = {17};
    nuv_qss[152] = {1};
    // (152):  CH3 + C2H4 => NXC3H7
    fwd_A[152]     = 410000000000;
    fwd_beta[152]  = 0;
    fwd_Ea[152]    = 7204.1099999999997;
    prefactor_units[152]  = 1.0000000000000002e-06;
    activation_units[152] = 0.50321666580471969;
    phase_units[152]      = pow(10,-12.000000);
    is_PD[152] = 0;
    nTB[152] = 0;

    // (153):  NXC3H7 => H + C3H6
    kiv[153] = {3,19};
    nuv[153] = {1,1};
    kiv_qss[153] = {17};
    nuv_qss[153] = {-1};
    // (153):  NXC3H7 => H + C3H6
    fwd_A[153]     = 2667000000000000;
    fwd_beta[153]  = -0.64000000000000001;
    fwd_Ea[153]    = 36820.029999999999;
    prefactor_units[153]  = 1;
    activation_units[153] = 0.50321666580471969;
    phase_units[153]      = pow(10,-6.000000);
    is_PD[153] = 0;
    nTB[153] = 0;

    // (154):  H + C3H6 => NXC3H7
    kiv[154] = {3,19};
    nuv[154] = {-1,-1};
    kiv_qss[154] = {17};
    nuv_qss[154] = {1};
    // (154):  H + C3H6 => NXC3H7
    fwd_A[154]     = 10000000000000;
    fwd_beta[154]  = 0;
    fwd_Ea[154]    = 2500;
    prefactor_units[154]  = 1.0000000000000002e-06;
    activation_units[154] = 0.50321666580471969;
    phase_units[154]      = pow(10,-12.000000);
    is_PD[154] = 0;
    nTB[154] = 0;

    // (155):  NXC3H7O2 => NXC3H7 + O2
    kiv[155] = {6};
    nuv[155] = {1};
    kiv_qss[155] = {18,17};
    nuv_qss[155] = {-1,1};
    // (155):  NXC3H7O2 => NXC3H7 + O2
    fwd_A[155]     = 3.364e+19;
    fwd_beta[155]  = -1.3200000000000001;
    fwd_Ea[155]    = 35760.040000000001;
    prefactor_units[155]  = 1;
    activation_units[155] = 0.50321666580471969;
    phase_units[155]      = pow(10,-6.000000);
    is_PD[155] = 0;
    nTB[155] = 0;

    // (156):  NXC3H7 + O2 => NXC3H7O2
    kiv[156] = {6};
    nuv[156] = {-1};
    kiv_qss[156] = {17,18};
    nuv_qss[156] = {-1,1};
    // (156):  NXC3H7 + O2 => NXC3H7O2
    fwd_A[156]     = 4520000000000;
    fwd_beta[156]  = 0;
    fwd_Ea[156]    = 0;
    prefactor_units[156]  = 1.0000000000000002e-06;
    activation_units[156] = 0.50321666580471969;
    phase_units[156]      = pow(10,-12.000000);
    is_PD[156] = 0;
    nTB[156] = 0;

    // (157):  C4H6 => 2.000000 C2H3
    kiv[157] = {22};
    nuv[157] = {-1};
    kiv_qss[157] = {9};
    nuv_qss[157] = {2.0};
    // (157):  C4H6 => 2.000000 C2H3
    fwd_A[157]     = 4.027e+19;
    fwd_beta[157]  = -1;
    fwd_Ea[157]    = 98150.100000000006;
    prefactor_units[157]  = 1;
    activation_units[157] = 0.50321666580471969;
    phase_units[157]      = pow(10,-6.000000);
    is_PD[157] = 0;
    nTB[157] = 0;

    // (158):  2.000000 C2H3 => C4H6
    kiv[158] = {22};
    nuv[158] = {1};
    kiv_qss[158] = {9};
    nuv_qss[158] = {-2.0};
    // (158):  2.000000 C2H3 => C4H6
    fwd_A[158]     = 12600000000000;
    fwd_beta[158]  = 0;
    fwd_Ea[158]    = 0;
    prefactor_units[158]  = 1.0000000000000002e-06;
    activation_units[158] = 0.50321666580471969;
    phase_units[158]      = pow(10,-12.000000);
    is_PD[158] = 0;
    nTB[158] = 0;

    // (159):  C4H6 + OH => CH2O + C3H5XA
    kiv[159] = {22,4,11};
    nuv[159] = {-1,-1,1};
    kiv_qss[159] = {16};
    nuv_qss[159] = {1};
    // (159):  C4H6 + OH => CH2O + C3H5XA
    fwd_A[159]     = 1000000000000;
    fwd_beta[159]  = 0;
    fwd_Ea[159]    = 0;
    prefactor_units[159]  = 1.0000000000000002e-06;
    activation_units[159] = 0.50321666580471969;
    phase_units[159]      = pow(10,-12.000000);
    is_PD[159] = 0;
    nTB[159] = 0;

    // (160):  C4H6 + OH => C2H5 + CH2CO
    kiv[160] = {22,4,16};
    nuv[160] = {-1,-1,1};
    kiv_qss[160] = {5};
    nuv_qss[160] = {1};
    // (160):  C4H6 + OH => C2H5 + CH2CO
    fwd_A[160]     = 1000000000000;
    fwd_beta[160]  = 0;
    fwd_Ea[160]    = 0;
    prefactor_units[160]  = 1.0000000000000002e-06;
    activation_units[160] = 0.50321666580471969;
    phase_units[160]      = pow(10,-12.000000);
    is_PD[160] = 0;
    nTB[160] = 0;

    // (161):  C4H6 + O => C2H4 + CH2CO
    kiv[161] = {22,1,20,16};
    nuv[161] = {-1,-1,1,1};
    kiv_qss[161] = {};
    nuv_qss[161] = {};
    // (161):  C4H6 + O => C2H4 + CH2CO
    fwd_A[161]     = 1000000000000;
    fwd_beta[161]  = 0;
    fwd_Ea[161]    = 0;
    prefactor_units[161]  = 1.0000000000000002e-06;
    activation_units[161] = 0.50321666580471969;
    phase_units[161]      = pow(10,-12.000000);
    is_PD[161] = 0;
    nTB[161] = 0;

    // (162):  C4H6 + H => C2H3 + C2H4
    kiv[162] = {22,3,20};
    nuv[162] = {-1,-1,1};
    kiv_qss[162] = {9};
    nuv_qss[162] = {1};
    // (162):  C4H6 + H => C2H3 + C2H4
    fwd_A[162]     = 10000000000000;
    fwd_beta[162]  = 0;
    fwd_Ea[162]    = 4700.0500000000002;
    prefactor_units[162]  = 1.0000000000000002e-06;
    activation_units[162] = 0.50321666580471969;
    phase_units[162]      = pow(10,-12.000000);
    is_PD[162] = 0;
    nTB[162] = 0;

    // (163):  C4H6 + O => CH2O + C3H4XA
    kiv[163] = {22,1,11,21};
    nuv[163] = {-1,-1,1,1};
    kiv_qss[163] = {};
    nuv_qss[163] = {};
    // (163):  C4H6 + O => CH2O + C3H4XA
    fwd_A[163]     = 1000000000000;
    fwd_beta[163]  = 0;
    fwd_Ea[163]    = 0;
    prefactor_units[163]  = 1.0000000000000002e-06;
    activation_units[163] = 0.50321666580471969;
    phase_units[163]      = pow(10,-12.000000);
    is_PD[163] = 0;
    nTB[163] = 0;

    // (164):  H + C4H7 => C4H8X1
    kiv[164] = {3,23};
    nuv[164] = {-1,1};
    kiv_qss[164] = {19};
    nuv_qss[164] = {-1};
    // (164):  H + C4H7 => C4H8X1
    fwd_A[164]     = 50000000000000;
    fwd_beta[164]  = 0;
    fwd_Ea[164]    = 0;
    prefactor_units[164]  = 1.0000000000000002e-06;
    activation_units[164] = 0.50321666580471969;
    phase_units[164]      = pow(10,-12.000000);
    is_PD[164] = 0;
    nTB[164] = 0;

    // (165):  C4H7 => C4H6 + H
    kiv[165] = {22,3};
    nuv[165] = {1,1};
    kiv_qss[165] = {19};
    nuv_qss[165] = {-1};
    // (165):  C4H7 => C4H6 + H
    fwd_A[165]     = 120000000000000;
    fwd_beta[165]  = 0;
    fwd_Ea[165]    = 49299.949999999997;
    prefactor_units[165]  = 1;
    activation_units[165] = 0.50321666580471969;
    phase_units[165]      = pow(10,-6.000000);
    is_PD[165] = 0;
    nTB[165] = 0;

    // (166):  C4H6 + H => C4H7
    kiv[166] = {22,3};
    nuv[166] = {-1,-1};
    kiv_qss[166] = {19};
    nuv_qss[166] = {1};
    // (166):  C4H6 + H => C4H7
    fwd_A[166]     = 40000000000000;
    fwd_beta[166]  = 0;
    fwd_Ea[166]    = 1299.95;
    prefactor_units[166]  = 1.0000000000000002e-06;
    activation_units[166] = 0.50321666580471969;
    phase_units[166]      = pow(10,-12.000000);
    is_PD[166] = 0;
    nTB[166] = 0;

    // (167):  C4H7 + CH3 => C4H6 + CH4
    kiv[167] = {12,22,13};
    nuv[167] = {-1,1,1};
    kiv_qss[167] = {19};
    nuv_qss[167] = {-1};
    // (167):  C4H7 + CH3 => C4H6 + CH4
    fwd_A[167]     = 8000000000000;
    fwd_beta[167]  = 0;
    fwd_Ea[167]    = 0;
    prefactor_units[167]  = 1.0000000000000002e-06;
    activation_units[167] = 0.50321666580471969;
    phase_units[167]      = pow(10,-12.000000);
    is_PD[167] = 0;
    nTB[167] = 0;

    // (168):  C4H7 + HO2 => C4H8X1 + O2
    kiv[168] = {7,23,6};
    nuv[168] = {-1,1,1};
    kiv_qss[168] = {19};
    nuv_qss[168] = {-1};
    // (168):  C4H7 + HO2 => C4H8X1 + O2
    fwd_A[168]     = 300000000000;
    fwd_beta[168]  = 0;
    fwd_Ea[168]    = 0;
    prefactor_units[168]  = 1.0000000000000002e-06;
    activation_units[168] = 0.50321666580471969;
    phase_units[168]      = pow(10,-12.000000);
    is_PD[168] = 0;
    nTB[168] = 0;

    // (169):  C4H7 + O2 => C4H6 + HO2
    kiv[169] = {6,22,7};
    nuv[169] = {-1,1,1};
    kiv_qss[169] = {19};
    nuv_qss[169] = {-1};
    // (169):  C4H7 + O2 => C4H6 + HO2
    fwd_A[169]     = 1000000000;
    fwd_beta[169]  = 0;
    fwd_Ea[169]    = 0;
    prefactor_units[169]  = 1.0000000000000002e-06;
    activation_units[169] = 0.50321666580471969;
    phase_units[169]      = pow(10,-12.000000);
    is_PD[169] = 0;
    nTB[169] = 0;

    // (170):  C4H7 => C2H4 + C2H3
    kiv[170] = {20};
    nuv[170] = {1};
    kiv_qss[170] = {19,9};
    nuv_qss[170] = {-1,1};
    // (170):  C4H7 => C2H4 + C2H3
    fwd_A[170]     = 100000000000;
    fwd_beta[170]  = 0;
    fwd_Ea[170]    = 37000;
    prefactor_units[170]  = 1;
    activation_units[170] = 0.50321666580471969;
    phase_units[170]      = pow(10,-6.000000);
    is_PD[170] = 0;
    nTB[170] = 0;

    // (171):  H + C4H7 => C4H6 + H2
    kiv[171] = {3,22,2};
    nuv[171] = {-1,1,1};
    kiv_qss[171] = {19};
    nuv_qss[171] = {-1};
    // (171):  H + C4H7 => C4H6 + H2
    fwd_A[171]     = 31600000000000;
    fwd_beta[171]  = 0;
    fwd_Ea[171]    = 0;
    prefactor_units[171]  = 1.0000000000000002e-06;
    activation_units[171] = 0.50321666580471969;
    phase_units[171]      = pow(10,-12.000000);
    is_PD[171] = 0;
    nTB[171] = 0;

    // (172):  C4H8X1 + H => C4H7 + H2
    kiv[172] = {23,3,2};
    nuv[172] = {-1,-1,1};
    kiv_qss[172] = {19};
    nuv_qss[172] = {1};
    // (172):  C4H8X1 + H => C4H7 + H2
    fwd_A[172]     = 50000000000000;
    fwd_beta[172]  = 0;
    fwd_Ea[172]    = 3900.0999999999999;
    prefactor_units[172]  = 1.0000000000000002e-06;
    activation_units[172] = 0.50321666580471969;
    phase_units[172]      = pow(10,-12.000000);
    is_PD[172] = 0;
    nTB[172] = 0;

    // (173):  C4H8X1 + OH => NXC3H7 + CH2O
    kiv[173] = {23,4,11};
    nuv[173] = {-1,-1,1};
    kiv_qss[173] = {17};
    nuv_qss[173] = {1};
    // (173):  C4H8X1 + OH => NXC3H7 + CH2O
    fwd_A[173]     = 1000000000000;
    fwd_beta[173]  = 0;
    fwd_Ea[173]    = 0;
    prefactor_units[173]  = 1.0000000000000002e-06;
    activation_units[173] = 0.50321666580471969;
    phase_units[173]      = pow(10,-12.000000);
    is_PD[173] = 0;
    nTB[173] = 0;

    // (174):  C4H8X1 + OH => CH3CO + C2H6
    kiv[174] = {23,4,15};
    nuv[174] = {-1,-1,1};
    kiv_qss[174] = {12};
    nuv_qss[174] = {1};
    // (174):  C4H8X1 + OH => CH3CO + C2H6
    fwd_A[174]     = 500000000000;
    fwd_beta[174]  = 0;
    fwd_Ea[174]    = 0;
    prefactor_units[174]  = 1.0000000000000002e-06;
    activation_units[174] = 0.50321666580471969;
    phase_units[174]      = pow(10,-12.000000);
    is_PD[174] = 0;
    nTB[174] = 0;

    // (175):  C4H8X1 + O => CH3CO + C2H5
    kiv[175] = {23,1};
    nuv[175] = {-1,-1};
    kiv_qss[175] = {12,5};
    nuv_qss[175] = {1,1};
    // (175):  C4H8X1 + O => CH3CO + C2H5
    fwd_A[175]     = 13000000000000;
    fwd_beta[175]  = 0;
    fwd_Ea[175]    = 849.89999999999998;
    prefactor_units[175]  = 1.0000000000000002e-06;
    activation_units[175] = 0.50321666580471969;
    phase_units[175]      = pow(10,-12.000000);
    is_PD[175] = 0;
    nTB[175] = 0;

    // (176):  C4H8X1 + O => C3H6 + CH2O
    kiv[176] = {23,1,19,11};
    nuv[176] = {-1,-1,1,1};
    kiv_qss[176] = {};
    nuv_qss[176] = {};
    // (176):  C4H8X1 + O => C3H6 + CH2O
    fwd_A[176]     = 723000;
    fwd_beta[176]  = 2.3399999999999999;
    fwd_Ea[176]    = -1049.95;
    prefactor_units[176]  = 1.0000000000000002e-06;
    activation_units[176] = 0.50321666580471969;
    phase_units[176]      = pow(10,-12.000000);
    is_PD[176] = 0;
    nTB[176] = 0;

    // (177):  C4H8X1 + OH => C4H7 + H2O
    kiv[177] = {23,4,5};
    nuv[177] = {-1,-1,1};
    kiv_qss[177] = {19};
    nuv_qss[177] = {1};
    // (177):  C4H8X1 + OH => C4H7 + H2O
    fwd_A[177]     = 22500000000000;
    fwd_beta[177]  = 0;
    fwd_Ea[177]    = 2217.02;
    prefactor_units[177]  = 1.0000000000000002e-06;
    activation_units[177] = 0.50321666580471969;
    phase_units[177]      = pow(10,-12.000000);
    is_PD[177] = 0;
    nTB[177] = 0;

    // (178):  C4H8X1 => C3H5XA + CH3
    kiv[178] = {23,12};
    nuv[178] = {-1,1};
    kiv_qss[178] = {16};
    nuv_qss[178] = {1};
    // (178):  C4H8X1 => C3H5XA + CH3
    fwd_A[178]     = 5000000000000000;
    fwd_beta[178]  = 0;
    fwd_Ea[178]    = 71000;
    prefactor_units[178]  = 1;
    activation_units[178] = 0.50321666580471969;
    phase_units[178]      = pow(10,-6.000000);
    is_PD[178] = 0;
    nTB[178] = 0;

    // (179):  C3H5XA + CH3 => C4H8X1
    kiv[179] = {12,23};
    nuv[179] = {-1,1};
    kiv_qss[179] = {16};
    nuv_qss[179] = {-1};
    // (179):  C3H5XA + CH3 => C4H8X1
    fwd_A[179]     = 5000000000000;
    fwd_beta[179]  = 0;
    fwd_Ea[179]    = 0;
    prefactor_units[179]  = 1.0000000000000002e-06;
    activation_units[179] = 0.50321666580471969;
    phase_units[179]      = pow(10,-12.000000);
    is_PD[179] = 0;
    nTB[179] = 0;

    // (180):  PXC4H9 => C4H8X1 + H
    kiv[180] = {23,3};
    nuv[180] = {1,1};
    kiv_qss[180] = {20};
    nuv_qss[180] = {-1};
    // (180):  PXC4H9 => C4H8X1 + H
    fwd_A[180]     = 1.159e+17;
    fwd_beta[180]  = -1.1699999999999999;
    fwd_Ea[180]    = 38159.889999999999;
    prefactor_units[180]  = 1;
    activation_units[180] = 0.50321666580471969;
    phase_units[180]      = pow(10,-6.000000);
    is_PD[180] = 0;
    nTB[180] = 0;

    // (181):  C4H8X1 + H => PXC4H9
    kiv[181] = {23,3};
    nuv[181] = {-1,-1};
    kiv_qss[181] = {20};
    nuv_qss[181] = {1};
    // (181):  C4H8X1 + H => PXC4H9
    fwd_A[181]     = 10000000000000;
    fwd_beta[181]  = 0;
    fwd_Ea[181]    = 2900.0999999999999;
    prefactor_units[181]  = 1.0000000000000002e-06;
    activation_units[181] = 0.50321666580471969;
    phase_units[181]      = pow(10,-12.000000);
    is_PD[181] = 0;
    nTB[181] = 0;

    // (182):  PXC4H9 => C2H5 + C2H4
    kiv[182] = {20};
    nuv[182] = {1};
    kiv_qss[182] = {20,5};
    nuv_qss[182] = {-1,1};
    // (182):  PXC4H9 => C2H5 + C2H4
    fwd_A[182]     = 7.497e+17;
    fwd_beta[182]  = -1.4099999999999999;
    fwd_Ea[182]    = 29580.07;
    prefactor_units[182]  = 1;
    activation_units[182] = 0.50321666580471969;
    phase_units[182]      = pow(10,-6.000000);
    is_PD[182] = 0;
    nTB[182] = 0;

    // (183):  PXC4H9O2 => PXC4H9 + O2
    kiv[183] = {6};
    nuv[183] = {1};
    kiv_qss[183] = {21,20};
    nuv_qss[183] = {-1,1};
    // (183):  PXC4H9O2 => PXC4H9 + O2
    fwd_A[183]     = 6.155e+19;
    fwd_beta[183]  = -1.3799999999999999;
    fwd_Ea[183]    = 35510.040000000001;
    prefactor_units[183]  = 1;
    activation_units[183] = 0.50321666580471969;
    phase_units[183]      = pow(10,-6.000000);
    is_PD[183] = 0;
    nTB[183] = 0;

    // (184):  PXC4H9 + O2 => PXC4H9O2
    kiv[184] = {6};
    nuv[184] = {-1};
    kiv_qss[184] = {20,21};
    nuv_qss[184] = {-1,1};
    // (184):  PXC4H9 + O2 => PXC4H9O2
    fwd_A[184]     = 4520000000000;
    fwd_beta[184]  = 0;
    fwd_Ea[184]    = 0;
    prefactor_units[184]  = 1.0000000000000002e-06;
    activation_units[184] = 0.50321666580471969;
    phase_units[184]      = pow(10,-12.000000);
    is_PD[184] = 0;
    nTB[184] = 0;

    // (185):  C5H9 => C4H6 + CH3
    kiv[185] = {22,12};
    nuv[185] = {1,1};
    kiv_qss[185] = {22};
    nuv_qss[185] = {-1};
    // (185):  C5H9 => C4H6 + CH3
    fwd_A[185]     = 1339000000000000;
    fwd_beta[185]  = -0.52000000000000002;
    fwd_Ea[185]    = 38320.029999999999;
    prefactor_units[185]  = 1;
    activation_units[185] = 0.50321666580471969;
    phase_units[185]      = pow(10,-6.000000);
    is_PD[185] = 0;
    nTB[185] = 0;

    // (186):  C5H9 => C3H5XA + C2H4
    kiv[186] = {20};
    nuv[186] = {1};
    kiv_qss[186] = {22,16};
    nuv_qss[186] = {-1,1};
    // (186):  C5H9 => C3H5XA + C2H4
    fwd_A[186]     = 25000000000000;
    fwd_beta[186]  = 0;
    fwd_Ea[186]    = 45000;
    prefactor_units[186]  = 1;
    activation_units[186] = 0.50321666580471969;
    phase_units[186]      = pow(10,-6.000000);
    is_PD[186] = 0;
    nTB[186] = 0;

    // (187):  C5H10X1 + OH => C5H9 + H2O
    kiv[187] = {4,5};
    nuv[187] = {-1,1};
    kiv_qss[187] = {23,22};
    nuv_qss[187] = {-1,1};
    // (187):  C5H10X1 + OH => C5H9 + H2O
    fwd_A[187]     = 5120000;
    fwd_beta[187]  = 2;
    fwd_Ea[187]    = -298.04000000000002;
    prefactor_units[187]  = 1.0000000000000002e-06;
    activation_units[187] = 0.50321666580471969;
    phase_units[187]      = pow(10,-12.000000);
    is_PD[187] = 0;
    nTB[187] = 0;

    // (188):  C5H10X1 + H => C5H9 + H2
    kiv[188] = {3,2};
    nuv[188] = {-1,1};
    kiv_qss[188] = {23,22};
    nuv_qss[188] = {-1,1};
    // (188):  C5H10X1 + H => C5H9 + H2
    fwd_A[188]     = 28000000000000;
    fwd_beta[188]  = 0;
    fwd_Ea[188]    = 4000;
    prefactor_units[188]  = 1.0000000000000002e-06;
    activation_units[188] = 0.50321666580471969;
    phase_units[188]      = pow(10,-12.000000);
    is_PD[188] = 0;
    nTB[188] = 0;

    // (189):  C5H10X1 => C2H5 + C3H5XA
    kiv[189] = {};
    nuv[189] = {};
    kiv_qss[189] = {23,5,16};
    nuv_qss[189] = {-1,1,1};
    // (189):  C5H10X1 => C2H5 + C3H5XA
    fwd_A[189]     = 9.173e+20;
    fwd_beta[189]  = -1.6299999999999999;
    fwd_Ea[189]    = 73989.960000000006;
    prefactor_units[189]  = 1;
    activation_units[189] = 0.50321666580471969;
    phase_units[189]      = pow(10,-6.000000);
    is_PD[189] = 0;
    nTB[189] = 0;

    // (190):  C5H10X1 + O => C5H9 + OH
    kiv[190] = {1,4};
    nuv[190] = {-1,1};
    kiv_qss[190] = {23,22};
    nuv_qss[190] = {-1,1};
    // (190):  C5H10X1 + O => C5H9 + OH
    fwd_A[190]     = 254000;
    fwd_beta[190]  = 2.5600000000000001;
    fwd_Ea[190]    = -1130.02;
    prefactor_units[190]  = 1.0000000000000002e-06;
    activation_units[190] = 0.50321666580471969;
    phase_units[190]      = pow(10,-12.000000);
    is_PD[190] = 0;
    nTB[190] = 0;

    // (191):  C5H11X1 => C3H6 + C2H5
    kiv[191] = {19};
    nuv[191] = {1};
    kiv_qss[191] = {24,5};
    nuv_qss[191] = {-1,1};
    // (191):  C5H11X1 => C3H6 + C2H5
    fwd_A[191]     = 5.948e+17;
    fwd_beta[191]  = -1.268;
    fwd_Ea[191]    = 32384.32;
    prefactor_units[191]  = 1;
    activation_units[191] = 0.50321666580471969;
    phase_units[191]      = pow(10,-6.000000);
    is_PD[191] = 0;
    nTB[191] = 0;

    // (192):  C5H11X1 => C2H4 + NXC3H7
    kiv[192] = {20};
    nuv[192] = {1};
    kiv_qss[192] = {24,17};
    nuv_qss[192] = {-1,1};
    // (192):  C5H11X1 => C2H4 + NXC3H7
    fwd_A[192]     = 7.305e+18;
    fwd_beta[192]  = -1.7669999999999999;
    fwd_Ea[192]    = 29919.459999999999;
    prefactor_units[192]  = 1;
    activation_units[192] = 0.50321666580471969;
    phase_units[192]      = pow(10,-6.000000);
    is_PD[192] = 0;
    nTB[192] = 0;

    // (193):  C5H11X1 <=> C5H10X1 + H
    kiv[193] = {3};
    nuv[193] = {1};
    kiv_qss[193] = {24,23};
    nuv_qss[193] = {-1,1};
    // (193):  C5H11X1 <=> C5H10X1 + H
    fwd_A[193]     = 1325000000000000;
    fwd_beta[193]  = -0.55400000000000005;
    fwd_Ea[193]    = 37516.730000000003;
    prefactor_units[193]  = 1;
    activation_units[193] = 0.50321666580471969;
    phase_units[193]      = pow(10,-6.000000);
    is_PD[193] = 0;
    nTB[193] = 0;

    // (194):  C6H12X1 => NXC3H7 + C3H5XA
    kiv[194] = {};
    nuv[194] = {};
    kiv_qss[194] = {25,17,16};
    nuv_qss[194] = {-1,1,1};
    // (194):  C6H12X1 => NXC3H7 + C3H5XA
    fwd_A[194]     = 10000000000000000;
    fwd_beta[194]  = 0;
    fwd_Ea[194]    = 71000;
    prefactor_units[194]  = 1;
    activation_units[194] = 0.50321666580471969;
    phase_units[194]      = pow(10,-6.000000);
    is_PD[194] = 0;
    nTB[194] = 0;

    // (195):  C6H12X1 + OH => C5H11X1 + CH2O
    kiv[195] = {4,11};
    nuv[195] = {-1,1};
    kiv_qss[195] = {25,24};
    nuv_qss[195] = {-1,1};
    // (195):  C6H12X1 + OH => C5H11X1 + CH2O
    fwd_A[195]     = 100000000000;
    fwd_beta[195]  = -0;
    fwd_Ea[195]    = -4000;
    prefactor_units[195]  = 1.0000000000000002e-06;
    activation_units[195] = 0.50321666580471969;
    phase_units[195]      = pow(10,-12.000000);
    is_PD[195] = 0;
    nTB[195] = 0;

    // (196):  C7H15X2 => C6H12X1 + CH3
    kiv[196] = {12};
    nuv[196] = {1};
    kiv_qss[196] = {26,25};
    nuv_qss[196] = {-1,1};
    // (196):  C7H15X2 => C6H12X1 + CH3
    fwd_A[196]     = 261700000000000;
    fwd_beta[196]  = -0.65400000000000003;
    fwd_Ea[196]    = 29745.459999999999;
    prefactor_units[196]  = 1;
    activation_units[196] = 0.50321666580471969;
    phase_units[196]      = pow(10,-6.000000);
    is_PD[196] = 0;
    nTB[196] = 0;

    // (197):  C7H15X2 => PXC4H9 + C3H6
    kiv[197] = {19};
    nuv[197] = {1};
    kiv_qss[197] = {26,20};
    nuv_qss[197] = {-1,1};
    // (197):  C7H15X2 => PXC4H9 + C3H6
    fwd_A[197]     = 5.313e+17;
    fwd_beta[197]  = -1.411;
    fwd_Ea[197]    = 31432.84;
    prefactor_units[197]  = 1;
    activation_units[197] = 0.50321666580471969;
    phase_units[197]      = pow(10,-6.000000);
    is_PD[197] = 0;
    nTB[197] = 0;

    // (198):  C7H15X2 => C4H8X1 + NXC3H7
    kiv[198] = {23};
    nuv[198] = {1};
    kiv_qss[198] = {26,17};
    nuv_qss[198] = {-1,1};
    // (198):  C7H15X2 => C4H8X1 + NXC3H7
    fwd_A[198]     = 2.454e+18;
    fwd_beta[198]  = -1.6539999999999999;
    fwd_Ea[198]    = 31635.52;
    prefactor_units[198]  = 1;
    activation_units[198] = 0.50321666580471969;
    phase_units[198]      = pow(10,-6.000000);
    is_PD[198] = 0;
    nTB[198] = 0;

    // (199):  C7H15X2 => C5H11X1 + C2H4
    kiv[199] = {20};
    nuv[199] = {1};
    kiv_qss[199] = {26,24};
    nuv_qss[199] = {-1,1};
    // (199):  C7H15X2 => C5H11X1 + C2H4
    fwd_A[199]     = 3734000000000000;
    fwd_beta[199]  = -0.92700000000000005;
    fwd_Ea[199]    = 29637.91;
    prefactor_units[199]  = 1;
    activation_units[199] = 0.50321666580471969;
    phase_units[199]      = pow(10,-6.000000);
    is_PD[199] = 0;
    nTB[199] = 0;

    // (200):  C7H15X2 => C2H5 + C5H10X1
    kiv[200] = {};
    nuv[200] = {};
    kiv_qss[200] = {26,5,23};
    nuv_qss[200] = {-1,1,1};
    // (200):  C7H15X2 => C2H5 + C5H10X1
    fwd_A[200]     = 1.368e+17;
    fwd_beta[200]  = -1.3939999999999999;
    fwd_Ea[200]    = 29858.990000000002;
    prefactor_units[200]  = 1;
    activation_units[200] = 0.50321666580471969;
    phase_units[200]      = pow(10,-6.000000);
    is_PD[200] = 0;
    nTB[200] = 0;

    // (201):  C7H15X2 + HO2 => NXC7H16 + O2
    kiv[201] = {7,24,6};
    nuv[201] = {-1,1,1};
    kiv_qss[201] = {26};
    nuv_qss[201] = {-1};
    // (201):  C7H15X2 + HO2 => NXC7H16 + O2
    fwd_A[201]     = 191700000;
    fwd_beta[201]  = 0.871;
    fwd_Ea[201]    = -1588.9100000000001;
    prefactor_units[201]  = 1.0000000000000002e-06;
    activation_units[201] = 0.50321666580471969;
    phase_units[201]      = pow(10,-12.000000);
    is_PD[201] = 0;
    nTB[201] = 0;

    // (202):  NXC7H16 + CH3O2 => C7H15X2 + CH3O2H
    kiv[202] = {24};
    nuv[202] = {-1};
    kiv_qss[202] = {6,26,7};
    nuv_qss[202] = {-1,1,1};
    // (202):  NXC7H16 + CH3O2 => C7H15X2 + CH3O2H
    fwd_A[202]     = 5646000000000;
    fwd_beta[202]  = 0.20100000000000001;
    fwd_Ea[202]    = 17650.330000000002;
    prefactor_units[202]  = 1.0000000000000002e-06;
    activation_units[202] = 0.50321666580471969;
    phase_units[202]      = pow(10,-12.000000);
    is_PD[202] = 0;
    nTB[202] = 0;

    // (203):  NXC7H16 + H => C7H15X2 + H2
    kiv[203] = {24,3,2};
    nuv[203] = {-1,-1,1};
    kiv_qss[203] = {26};
    nuv_qss[203] = {1};
    // (203):  NXC7H16 + H => C7H15X2 + H2
    fwd_A[203]     = 1749000;
    fwd_beta[203]  = 2.6000000000000001;
    fwd_Ea[203]    = 4361.8500000000004;
    prefactor_units[203]  = 1.0000000000000002e-06;
    activation_units[203] = 0.50321666580471969;
    phase_units[203]      = pow(10,-12.000000);
    is_PD[203] = 0;
    nTB[203] = 0;

    // (204):  NXC7H16 => PXC4H9 + NXC3H7
    kiv[204] = {24};
    nuv[204] = {-1};
    kiv_qss[204] = {20,17};
    nuv_qss[204] = {1,1};
    // (204):  NXC7H16 => PXC4H9 + NXC3H7
    fwd_A[204]     = 1.415e+78;
    fwd_beta[204]  = -17.710000000000001;
    fwd_Ea[204]    = 120700.05;
    prefactor_units[204]  = 1;
    activation_units[204] = 0.50321666580471969;
    phase_units[204]      = pow(10,-6.000000);
    is_PD[204] = 0;
    nTB[204] = 0;

    // (205):  NXC7H16 + HO2 => C7H15X2 + H2O2
    kiv[205] = {24,7,8};
    nuv[205] = {-1,-1,1};
    kiv_qss[205] = {26};
    nuv_qss[205] = {1};
    // (205):  NXC7H16 + HO2 => C7H15X2 + H2O2
    fwd_A[205]     = 7741000000000;
    fwd_beta[205]  = 0.20300000000000001;
    fwd_Ea[205]    = 17636.950000000001;
    prefactor_units[205]  = 1.0000000000000002e-06;
    activation_units[205] = 0.50321666580471969;
    phase_units[205]      = pow(10,-12.000000);
    is_PD[205] = 0;
    nTB[205] = 0;

    // (206):  NXC7H16 => C5H11X1 + C2H5
    kiv[206] = {24};
    nuv[206] = {-1};
    kiv_qss[206] = {24,5};
    nuv_qss[206] = {1,1};
    // (206):  NXC7H16 => C5H11X1 + C2H5
    fwd_A[206]     = 8.0999999999999995e+77;
    fwd_beta[206]  = -17.620000000000001;
    fwd_Ea[206]    = 120400.10000000001;
    prefactor_units[206]  = 1;
    activation_units[206] = 0.50321666580471969;
    phase_units[206]      = pow(10,-6.000000);
    is_PD[206] = 0;
    nTB[206] = 0;

    // (207):  NXC7H16 + CH3O => C7H15X2 + CH3OH
    kiv[207] = {24,14};
    nuv[207] = {-1,1};
    kiv_qss[207] = {4,26};
    nuv_qss[207] = {-1,1};
    // (207):  NXC7H16 + CH3O => C7H15X2 + CH3OH
    fwd_A[207]     = 268900000000;
    fwd_beta[207]  = 0.13600000000000001;
    fwd_Ea[207]    = 5069.5500000000002;
    prefactor_units[207]  = 1.0000000000000002e-06;
    activation_units[207] = 0.50321666580471969;
    phase_units[207]      = pow(10,-12.000000);
    is_PD[207] = 0;
    nTB[207] = 0;

    // (208):  NXC7H16 + O => C7H15X2 + OH
    kiv[208] = {24,1,4};
    nuv[208] = {-1,-1,1};
    kiv_qss[208] = {26};
    nuv_qss[208] = {1};
    // (208):  NXC7H16 + O => C7H15X2 + OH
    fwd_A[208]     = 176600;
    fwd_beta[208]  = 2.802;
    fwd_Ea[208]    = 2265.3000000000002;
    prefactor_units[208]  = 1.0000000000000002e-06;
    activation_units[208] = 0.50321666580471969;
    phase_units[208]      = pow(10,-12.000000);
    is_PD[208] = 0;
    nTB[208] = 0;

    // (209):  NXC7H16 + OH => C7H15X2 + H2O
    kiv[209] = {24,4,5};
    nuv[209] = {-1,-1,1};
    kiv_qss[209] = {26};
    nuv_qss[209] = {1};
    // (209):  NXC7H16 + OH => C7H15X2 + H2O
    fwd_A[209]     = 751800000;
    fwd_beta[209]  = 1.494;
    fwd_Ea[209]    = 260.51999999999998;
    prefactor_units[209]  = 1.0000000000000002e-06;
    activation_units[209] = 0.50321666580471969;
    phase_units[209]      = pow(10,-12.000000);
    is_PD[209] = 0;
    nTB[209] = 0;

    // (210):  NXC7H16 + CH3 => C7H15X2 + CH4
    kiv[210] = {24,12,13};
    nuv[210] = {-1,-1,1};
    kiv_qss[210] = {26};
    nuv_qss[210] = {1};
    // (210):  NXC7H16 + CH3 => C7H15X2 + CH4
    fwd_A[210]     = 14420;
    fwd_beta[210]  = 2.573;
    fwd_Ea[210]    = 6933.5600000000004;
    prefactor_units[210]  = 1.0000000000000002e-06;
    activation_units[210] = 0.50321666580471969;
    phase_units[210]      = pow(10,-12.000000);
    is_PD[210] = 0;
    nTB[210] = 0;

}


/* Finalizes parameter database */
void CKFINALIZE()
{
  for (int i=0; i<211; ++i) {
    free(TB[i]); TB[i] = 0; 
    free(TBid[i]); TBid[i] = 0;
    nTB[i] = 0;
  }
}

#else
/* TODO: Remove on GPU, right now needed by chemistry_module on FORTRAN */
AMREX_GPU_HOST_DEVICE void CKINIT()
{
}

AMREX_GPU_HOST_DEVICE void CKFINALIZE()
{
}

#endif


/*A few mechanism parameters */
void CKINDX(int * mm, int * kk, int * ii, int * nfit)
{
    *mm = 4;
    *kk = 25;
    *ii = 211;
    *nfit = -1; /*Why do you need this anyway ?  */
}


/* Returns the vector of strings of element names */
void CKSYME_STR(amrex::Vector<std::string>& ename)
{
    ename.resize(4);
    ename[0] = "N";
    ename[1] = "O";
    ename[2] = "H";
    ename[3] = "C";
}


/* Returns the char strings of element names */
void CKSYME(int * kname, int * plenkname )
{
    int i; /*Loop Counter */
    int lenkname = *plenkname;
    /*clear kname */
    for (i=0; i<lenkname*4; i++) {
        kname[i] = ' ';
    }

    /* N  */
    kname[ 0*lenkname + 0 ] = 'N';
    kname[ 0*lenkname + 1 ] = ' ';

    /* O  */
    kname[ 1*lenkname + 0 ] = 'O';
    kname[ 1*lenkname + 1 ] = ' ';

    /* H  */
    kname[ 2*lenkname + 0 ] = 'H';
    kname[ 2*lenkname + 1 ] = ' ';

    /* C  */
    kname[ 3*lenkname + 0 ] = 'C';
    kname[ 3*lenkname + 1 ] = ' ';

}


/* Returns the vector of strings of species names */
void CKSYMS_STR(amrex::Vector<std::string>& kname)
{
    kname.resize(25);
    kname[0] = "N2";
    kname[1] = "O";
    kname[2] = "H2";
    kname[3] = "H";
    kname[4] = "OH";
    kname[5] = "H2O";
    kname[6] = "O2";
    kname[7] = "HO2";
    kname[8] = "H2O2";
    kname[9] = "CO2";
    kname[10] = "CO";
    kname[11] = "CH2O";
    kname[12] = "CH3";
    kname[13] = "CH4";
    kname[14] = "CH3OH";
    kname[15] = "C2H6";
    kname[16] = "CH2CO";
    kname[17] = "HOCHO";
    kname[18] = "C2H2";
    kname[19] = "C3H6";
    kname[20] = "C2H4";
    kname[21] = "C3H4XA";
    kname[22] = "C4H6";
    kname[23] = "C4H8X1";
    kname[24] = "NXC7H16";
}


/* Returns the char strings of species names */
void CKSYMS(int * kname, int * plenkname )
{
    int i; /*Loop Counter */
    int lenkname = *plenkname;
    /*clear kname */
    for (i=0; i<lenkname*25; i++) {
        kname[i] = ' ';
    }

    /* N2  */
    kname[ 0*lenkname + 0 ] = 'N';
    kname[ 0*lenkname + 1 ] = '2';
    kname[ 0*lenkname + 2 ] = ' ';

    /* O  */
    kname[ 1*lenkname + 0 ] = 'O';
    kname[ 1*lenkname + 1 ] = ' ';

    /* H2  */
    kname[ 2*lenkname + 0 ] = 'H';
    kname[ 2*lenkname + 1 ] = '2';
    kname[ 2*lenkname + 2 ] = ' ';

    /* H  */
    kname[ 3*lenkname + 0 ] = 'H';
    kname[ 3*lenkname + 1 ] = ' ';

    /* OH  */
    kname[ 4*lenkname + 0 ] = 'O';
    kname[ 4*lenkname + 1 ] = 'H';
    kname[ 4*lenkname + 2 ] = ' ';

    /* H2O  */
    kname[ 5*lenkname + 0 ] = 'H';
    kname[ 5*lenkname + 1 ] = '2';
    kname[ 5*lenkname + 2 ] = 'O';
    kname[ 5*lenkname + 3 ] = ' ';

    /* O2  */
    kname[ 6*lenkname + 0 ] = 'O';
    kname[ 6*lenkname + 1 ] = '2';
    kname[ 6*lenkname + 2 ] = ' ';

    /* HO2  */
    kname[ 7*lenkname + 0 ] = 'H';
    kname[ 7*lenkname + 1 ] = 'O';
    kname[ 7*lenkname + 2 ] = '2';
    kname[ 7*lenkname + 3 ] = ' ';

    /* H2O2  */
    kname[ 8*lenkname + 0 ] = 'H';
    kname[ 8*lenkname + 1 ] = '2';
    kname[ 8*lenkname + 2 ] = 'O';
    kname[ 8*lenkname + 3 ] = '2';
    kname[ 8*lenkname + 4 ] = ' ';

    /* CO2  */
    kname[ 9*lenkname + 0 ] = 'C';
    kname[ 9*lenkname + 1 ] = 'O';
    kname[ 9*lenkname + 2 ] = '2';
    kname[ 9*lenkname + 3 ] = ' ';

    /* CO  */
    kname[ 10*lenkname + 0 ] = 'C';
    kname[ 10*lenkname + 1 ] = 'O';
    kname[ 10*lenkname + 2 ] = ' ';

    /* CH2O  */
    kname[ 11*lenkname + 0 ] = 'C';
    kname[ 11*lenkname + 1 ] = 'H';
    kname[ 11*lenkname + 2 ] = '2';
    kname[ 11*lenkname + 3 ] = 'O';
    kname[ 11*lenkname + 4 ] = ' ';

    /* CH3  */
    kname[ 12*lenkname + 0 ] = 'C';
    kname[ 12*lenkname + 1 ] = 'H';
    kname[ 12*lenkname + 2 ] = '3';
    kname[ 12*lenkname + 3 ] = ' ';

    /* CH4  */
    kname[ 13*lenkname + 0 ] = 'C';
    kname[ 13*lenkname + 1 ] = 'H';
    kname[ 13*lenkname + 2 ] = '4';
    kname[ 13*lenkname + 3 ] = ' ';

    /* CH3OH  */
    kname[ 14*lenkname + 0 ] = 'C';
    kname[ 14*lenkname + 1 ] = 'H';
    kname[ 14*lenkname + 2 ] = '3';
    kname[ 14*lenkname + 3 ] = 'O';
    kname[ 14*lenkname + 4 ] = 'H';
    kname[ 14*lenkname + 5 ] = ' ';

    /* C2H6  */
    kname[ 15*lenkname + 0 ] = 'C';
    kname[ 15*lenkname + 1 ] = '2';
    kname[ 15*lenkname + 2 ] = 'H';
    kname[ 15*lenkname + 3 ] = '6';
    kname[ 15*lenkname + 4 ] = ' ';

    /* CH2CO  */
    kname[ 16*lenkname + 0 ] = 'C';
    kname[ 16*lenkname + 1 ] = 'H';
    kname[ 16*lenkname + 2 ] = '2';
    kname[ 16*lenkname + 3 ] = 'C';
    kname[ 16*lenkname + 4 ] = 'O';
    kname[ 16*lenkname + 5 ] = ' ';

    /* HOCHO  */
    kname[ 17*lenkname + 0 ] = 'H';
    kname[ 17*lenkname + 1 ] = 'O';
    kname[ 17*lenkname + 2 ] = 'C';
    kname[ 17*lenkname + 3 ] = 'H';
    kname[ 17*lenkname + 4 ] = 'O';
    kname[ 17*lenkname + 5 ] = ' ';

    /* C2H2  */
    kname[ 18*lenkname + 0 ] = 'C';
    kname[ 18*lenkname + 1 ] = '2';
    kname[ 18*lenkname + 2 ] = 'H';
    kname[ 18*lenkname + 3 ] = '2';
    kname[ 18*lenkname + 4 ] = ' ';

    /* C3H6  */
    kname[ 19*lenkname + 0 ] = 'C';
    kname[ 19*lenkname + 1 ] = '3';
    kname[ 19*lenkname + 2 ] = 'H';
    kname[ 19*lenkname + 3 ] = '6';
    kname[ 19*lenkname + 4 ] = ' ';

    /* C2H4  */
    kname[ 20*lenkname + 0 ] = 'C';
    kname[ 20*lenkname + 1 ] = '2';
    kname[ 20*lenkname + 2 ] = 'H';
    kname[ 20*lenkname + 3 ] = '4';
    kname[ 20*lenkname + 4 ] = ' ';

    /* C3H4XA  */
    kname[ 21*lenkname + 0 ] = 'C';
    kname[ 21*lenkname + 1 ] = '3';
    kname[ 21*lenkname + 2 ] = 'H';
    kname[ 21*lenkname + 3 ] = '4';
    kname[ 21*lenkname + 4 ] = 'X';
    kname[ 21*lenkname + 5 ] = 'A';
    kname[ 21*lenkname + 6 ] = ' ';

    /* C4H6  */
    kname[ 22*lenkname + 0 ] = 'C';
    kname[ 22*lenkname + 1 ] = '4';
    kname[ 22*lenkname + 2 ] = 'H';
    kname[ 22*lenkname + 3 ] = '6';
    kname[ 22*lenkname + 4 ] = ' ';

    /* C4H8X1  */
    kname[ 23*lenkname + 0 ] = 'C';
    kname[ 23*lenkname + 1 ] = '4';
    kname[ 23*lenkname + 2 ] = 'H';
    kname[ 23*lenkname + 3 ] = '8';
    kname[ 23*lenkname + 4 ] = 'X';
    kname[ 23*lenkname + 5 ] = '1';
    kname[ 23*lenkname + 6 ] = ' ';

    /* NXC7H16  */
    kname[ 24*lenkname + 0 ] = 'N';
    kname[ 24*lenkname + 1 ] = 'X';
    kname[ 24*lenkname + 2 ] = 'C';
    kname[ 24*lenkname + 3 ] = '7';
    kname[ 24*lenkname + 4 ] = 'H';
    kname[ 24*lenkname + 5 ] = '1';
    kname[ 24*lenkname + 6 ] = '6';
    kname[ 24*lenkname + 7 ] = ' ';

}


/* Returns R, Rc, Patm */
void CKRP(double *  ru, double *  ruc, double *  pa)
{
     *ru  = 8.31446261815324e+07; 
     *ruc = 1.98721558317399615845; 
     *pa  = 1.01325e+06; 
}


/*Compute P = rhoRT/W(x) */
void CKPX(double *  rho, double *  T, double *  x, double *  P)
{
    double XW = 0;/* To hold mean molecular wt */
    XW += x[0]*molecular_weights[0]; /*N2 */
    XW += x[1]*molecular_weights[1]; /*O */
    XW += x[2]*molecular_weights[2]; /*H2 */
    XW += x[3]*molecular_weights[3]; /*H */
    XW += x[4]*molecular_weights[4]; /*OH */
    XW += x[5]*molecular_weights[5]; /*H2O */
    XW += x[6]*molecular_weights[6]; /*O2 */
    XW += x[7]*molecular_weights[7]; /*HO2 */
    XW += x[8]*molecular_weights[8]; /*H2O2 */
    XW += x[9]*molecular_weights[9]; /*CO2 */
    XW += x[10]*molecular_weights[10]; /*CO */
    XW += x[11]*molecular_weights[11]; /*CH2O */
    XW += x[12]*molecular_weights[12]; /*CH3 */
    XW += x[13]*molecular_weights[13]; /*CH4 */
    XW += x[14]*molecular_weights[14]; /*CH3OH */
    XW += x[15]*molecular_weights[15]; /*C2H6 */
    XW += x[16]*molecular_weights[16]; /*CH2CO */
    XW += x[17]*molecular_weights[17]; /*HOCHO */
    XW += x[18]*molecular_weights[18]; /*C2H2 */
    XW += x[19]*molecular_weights[19]; /*C3H6 */
    XW += x[20]*molecular_weights[20]; /*C2H4 */
    XW += x[21]*molecular_weights[21]; /*C3H4XA */
    XW += x[22]*molecular_weights[22]; /*C4H6 */
    XW += x[23]*molecular_weights[23]; /*C4H8X1 */
    XW += x[24]*molecular_weights[24]; /*NXC7H16 */
    *P = *rho * 8.31446e+07 * (*T) / XW; /*P = rho*R*T/W */

    return;
}


/*Compute P = rhoRT/W(y) */
AMREX_GPU_HOST_DEVICE void CKPY(double *  rho, double *  T, double *  y,  double *  P)
{
    double YOW = 0;/* for computing mean MW */
    YOW += y[0]*imw[0]; /*N2 */
    YOW += y[1]*imw[1]; /*O */
    YOW += y[2]*imw[2]; /*H2 */
    YOW += y[3]*imw[3]; /*H */
    YOW += y[4]*imw[4]; /*OH */
    YOW += y[5]*imw[5]; /*H2O */
    YOW += y[6]*imw[6]; /*O2 */
    YOW += y[7]*imw[7]; /*HO2 */
    YOW += y[8]*imw[8]; /*H2O2 */
    YOW += y[9]*imw[9]; /*CO2 */
    YOW += y[10]*imw[10]; /*CO */
    YOW += y[11]*imw[11]; /*CH2O */
    YOW += y[12]*imw[12]; /*CH3 */
    YOW += y[13]*imw[13]; /*CH4 */
    YOW += y[14]*imw[14]; /*CH3OH */
    YOW += y[15]*imw[15]; /*C2H6 */
    YOW += y[16]*imw[16]; /*CH2CO */
    YOW += y[17]*imw[17]; /*HOCHO */
    YOW += y[18]*imw[18]; /*C2H2 */
    YOW += y[19]*imw[19]; /*C3H6 */
    YOW += y[20]*imw[20]; /*C2H4 */
    YOW += y[21]*imw[21]; /*C3H4XA */
    YOW += y[22]*imw[22]; /*C4H6 */
    YOW += y[23]*imw[23]; /*C4H8X1 */
    YOW += y[24]*imw[24]; /*NXC7H16 */
    *P = *rho * 8.31446e+07 * (*T) * YOW; /*P = rho*R*T/W */

    return;
}


/*Compute P = rhoRT/W(c) */
void CKPC(double *  rho, double *  T, double *  c,  double *  P)
{
    int id; /*loop counter */
    /*See Eq 5 in CK Manual */
    double W = 0;
    double sumC = 0;
    W += c[0]*28.013400; /*N2 */
    W += c[1]*15.999400; /*O */
    W += c[2]*2.015940; /*H2 */
    W += c[3]*1.007970; /*H */
    W += c[4]*17.007370; /*OH */
    W += c[5]*18.015340; /*H2O */
    W += c[6]*31.998800; /*O2 */
    W += c[7]*33.006770; /*HO2 */
    W += c[8]*34.014740; /*H2O2 */
    W += c[9]*44.009950; /*CO2 */
    W += c[10]*28.010550; /*CO */
    W += c[11]*30.026490; /*CH2O */
    W += c[12]*15.035060; /*CH3 */
    W += c[13]*16.043030; /*CH4 */
    W += c[14]*32.042430; /*CH3OH */
    W += c[15]*30.070120; /*C2H6 */
    W += c[16]*42.037640; /*CH2CO */
    W += c[17]*46.025890; /*HOCHO */
    W += c[18]*26.038240; /*C2H2 */
    W += c[19]*42.081270; /*C3H6 */
    W += c[20]*28.054180; /*C2H4 */
    W += c[21]*40.065330; /*C3H4XA */
    W += c[22]*54.092420; /*C4H6 */
    W += c[23]*56.108360; /*C4H8X1 */
    W += c[24]*100.205570; /*NXC7H16 */

    for (id = 0; id < 25; ++id) {
        sumC += c[id];
    }
    *P = *rho * 8.31446e+07 * (*T) * sumC / W; /*P = rho*R*T/W */

    return;
}


/*Compute rho = PW(x)/RT */
void CKRHOX(double *  P, double *  T, double *  x,  double *  rho)
{
    double XW = 0;/* To hold mean molecular wt */
    XW += x[0]*28.013400; /*N2 */
    XW += x[1]*15.999400; /*O */
    XW += x[2]*2.015940; /*H2 */
    XW += x[3]*1.007970; /*H */
    XW += x[4]*17.007370; /*OH */
    XW += x[5]*18.015340; /*H2O */
    XW += x[6]*31.998800; /*O2 */
    XW += x[7]*33.006770; /*HO2 */
    XW += x[8]*34.014740; /*H2O2 */
    XW += x[9]*44.009950; /*CO2 */
    XW += x[10]*28.010550; /*CO */
    XW += x[11]*30.026490; /*CH2O */
    XW += x[12]*15.035060; /*CH3 */
    XW += x[13]*16.043030; /*CH4 */
    XW += x[14]*32.042430; /*CH3OH */
    XW += x[15]*30.070120; /*C2H6 */
    XW += x[16]*42.037640; /*CH2CO */
    XW += x[17]*46.025890; /*HOCHO */
    XW += x[18]*26.038240; /*C2H2 */
    XW += x[19]*42.081270; /*C3H6 */
    XW += x[20]*28.054180; /*C2H4 */
    XW += x[21]*40.065330; /*C3H4XA */
    XW += x[22]*54.092420; /*C4H6 */
    XW += x[23]*56.108360; /*C4H8X1 */
    XW += x[24]*100.205570; /*NXC7H16 */
    *rho = *P * XW / (8.31446e+07 * (*T)); /*rho = P*W/(R*T) */

    return;
}


/*Compute rho = P*W(y)/RT */
AMREX_GPU_HOST_DEVICE void CKRHOY(double *  P, double *  T, double *  y,  double *  rho)
{
    double YOW = 0;
    double tmp[25];

    for (int i = 0; i < 25; i++)
    {
        tmp[i] = y[i]*imw[i];
    }
    for (int i = 0; i < 25; i++)
    {
        YOW += tmp[i];
    }

    *rho = *P / (8.31446e+07 * (*T) * YOW);/*rho = P*W/(R*T) */
    return;
}


/*Compute rho = P*W(c)/(R*T) */
void CKRHOC(double *  P, double *  T, double *  c,  double *  rho)
{
    int id; /*loop counter */
    /*See Eq 5 in CK Manual */
    double W = 0;
    double sumC = 0;
    W += c[0]*28.013400; /*N2 */
    W += c[1]*15.999400; /*O */
    W += c[2]*2.015940; /*H2 */
    W += c[3]*1.007970; /*H */
    W += c[4]*17.007370; /*OH */
    W += c[5]*18.015340; /*H2O */
    W += c[6]*31.998800; /*O2 */
    W += c[7]*33.006770; /*HO2 */
    W += c[8]*34.014740; /*H2O2 */
    W += c[9]*44.009950; /*CO2 */
    W += c[10]*28.010550; /*CO */
    W += c[11]*30.026490; /*CH2O */
    W += c[12]*15.035060; /*CH3 */
    W += c[13]*16.043030; /*CH4 */
    W += c[14]*32.042430; /*CH3OH */
    W += c[15]*30.070120; /*C2H6 */
    W += c[16]*42.037640; /*CH2CO */
    W += c[17]*46.025890; /*HOCHO */
    W += c[18]*26.038240; /*C2H2 */
    W += c[19]*42.081270; /*C3H6 */
    W += c[20]*28.054180; /*C2H4 */
    W += c[21]*40.065330; /*C3H4XA */
    W += c[22]*54.092420; /*C4H6 */
    W += c[23]*56.108360; /*C4H8X1 */
    W += c[24]*100.205570; /*NXC7H16 */

    for (id = 0; id < 25; ++id) {
        sumC += c[id];
    }
    *rho = *P * W / (sumC * (*T) * 8.31446e+07); /*rho = PW/(R*T) */

    return;
}


/*get molecular weight for all species */
void CKWT( double *  wt)
{
    get_mw(wt);
}


/*get atomic weight for all elements */
void CKAWT( double *  awt)
{
    atomicWeight(awt);
}


/*given y[species]: mass fractions */
/*returns mean molecular weight (gm/mole) */
AMREX_GPU_HOST_DEVICE void CKMMWY(double *  y,  double *  wtm)
{
    double YOW = 0;
    double tmp[25];

    for (int i = 0; i < 25; i++)
    {
        tmp[i] = y[i]*imw[i];
    }
    for (int i = 0; i < 25; i++)
    {
        YOW += tmp[i];
    }

    *wtm = 1.0 / YOW;
    return;
}


/*given x[species]: mole fractions */
/*returns mean molecular weight (gm/mole) */
void CKMMWX(double *  x,  double *  wtm)
{
    double XW = 0;/* see Eq 4 in CK Manual */
    XW += x[0]*molecular_weights[0]; /*N2 */
    XW += x[1]*molecular_weights[1]; /*O */
    XW += x[2]*molecular_weights[2]; /*H2 */
    XW += x[3]*molecular_weights[3]; /*H */
    XW += x[4]*molecular_weights[4]; /*OH */
    XW += x[5]*molecular_weights[5]; /*H2O */
    XW += x[6]*molecular_weights[6]; /*O2 */
    XW += x[7]*molecular_weights[7]; /*HO2 */
    XW += x[8]*molecular_weights[8]; /*H2O2 */
    XW += x[9]*molecular_weights[9]; /*CO2 */
    XW += x[10]*molecular_weights[10]; /*CO */
    XW += x[11]*molecular_weights[11]; /*CH2O */
    XW += x[12]*molecular_weights[12]; /*CH3 */
    XW += x[13]*molecular_weights[13]; /*CH4 */
    XW += x[14]*molecular_weights[14]; /*CH3OH */
    XW += x[15]*molecular_weights[15]; /*C2H6 */
    XW += x[16]*molecular_weights[16]; /*CH2CO */
    XW += x[17]*molecular_weights[17]; /*HOCHO */
    XW += x[18]*molecular_weights[18]; /*C2H2 */
    XW += x[19]*molecular_weights[19]; /*C3H6 */
    XW += x[20]*molecular_weights[20]; /*C2H4 */
    XW += x[21]*molecular_weights[21]; /*C3H4XA */
    XW += x[22]*molecular_weights[22]; /*C4H6 */
    XW += x[23]*molecular_weights[23]; /*C4H8X1 */
    XW += x[24]*molecular_weights[24]; /*NXC7H16 */
    *wtm = XW;

    return;
}


/*given c[species]: molar concentration */
/*returns mean molecular weight (gm/mole) */
void CKMMWC(double *  c,  double *  wtm)
{
    int id; /*loop counter */
    /*See Eq 5 in CK Manual */
    double W = 0;
    double sumC = 0;
    W += c[0]*molecular_weights[0]; /*N2 */
    W += c[1]*molecular_weights[1]; /*O */
    W += c[2]*molecular_weights[2]; /*H2 */
    W += c[3]*molecular_weights[3]; /*H */
    W += c[4]*molecular_weights[4]; /*OH */
    W += c[5]*molecular_weights[5]; /*H2O */
    W += c[6]*molecular_weights[6]; /*O2 */
    W += c[7]*molecular_weights[7]; /*HO2 */
    W += c[8]*molecular_weights[8]; /*H2O2 */
    W += c[9]*molecular_weights[9]; /*CO2 */
    W += c[10]*molecular_weights[10]; /*CO */
    W += c[11]*molecular_weights[11]; /*CH2O */
    W += c[12]*molecular_weights[12]; /*CH3 */
    W += c[13]*molecular_weights[13]; /*CH4 */
    W += c[14]*molecular_weights[14]; /*CH3OH */
    W += c[15]*molecular_weights[15]; /*C2H6 */
    W += c[16]*molecular_weights[16]; /*CH2CO */
    W += c[17]*molecular_weights[17]; /*HOCHO */
    W += c[18]*molecular_weights[18]; /*C2H2 */
    W += c[19]*molecular_weights[19]; /*C3H6 */
    W += c[20]*molecular_weights[20]; /*C2H4 */
    W += c[21]*molecular_weights[21]; /*C3H4XA */
    W += c[22]*molecular_weights[22]; /*C4H6 */
    W += c[23]*molecular_weights[23]; /*C4H8X1 */
    W += c[24]*molecular_weights[24]; /*NXC7H16 */

    for (id = 0; id < 25; ++id) {
        sumC += c[id];
    }
    /* CK provides no guard against divison by zero */
    *wtm = W/sumC;

    return;
}


/*convert y[species] (mass fracs) to x[species] (mole fracs) */
AMREX_GPU_HOST_DEVICE void CKYTX(double *  y,  double *  x)
{
    double YOW = 0;
    double tmp[25];

    for (int i = 0; i < 25; i++)
    {
        tmp[i] = y[i]*imw[i];
    }
    for (int i = 0; i < 25; i++)
    {
        YOW += tmp[i];
    }

    double YOWINV = 1.0/YOW;

    for (int i = 0; i < 25; i++)
    {
        x[i] = y[i]*imw[i]*YOWINV;
    }
    return;
}


/*convert y[species] (mass fracs) to c[species] (molar conc) */
void CKYTCP(double *  P, double *  T, double *  y,  double *  c)
{
    double YOW = 0;
    double PWORT;

    /*Compute inverse of mean molecular wt first */
    for (int i = 0; i < 25; i++)
    {
        c[i] = y[i]*imw[i];
    }
    for (int i = 0; i < 25; i++)
    {
        YOW += c[i];
    }

    /*PW/RT (see Eq. 7) */
    PWORT = (*P)/(YOW * 8.31446e+07 * (*T)); 
    /*Now compute conversion */

    for (int i = 0; i < 25; i++)
    {
        c[i] = PWORT * y[i] * imw[i];
    }
    return;
}


/*convert y[species] (mass fracs) to c[species] (molar conc) */
AMREX_GPU_HOST_DEVICE void CKYTCR(double *  rho, double *  T, double *  y,  double *  c)
{
    for (int i = 0; i < 25; i++)
    {
        c[i] = (*rho)  * y[i] * imw[i];
    }
}


/*convert x[species] (mole fracs) to y[species] (mass fracs) */
AMREX_GPU_HOST_DEVICE void CKXTY(double *  x,  double *  y)
{
    double XW = 0; /*See Eq 4, 9 in CK Manual */
    /*Compute mean molecular wt first */
    XW += x[0]*28.013400; /*N2 */
    XW += x[1]*15.999400; /*O */
    XW += x[2]*2.015940; /*H2 */
    XW += x[3]*1.007970; /*H */
    XW += x[4]*17.007370; /*OH */
    XW += x[5]*18.015340; /*H2O */
    XW += x[6]*31.998800; /*O2 */
    XW += x[7]*33.006770; /*HO2 */
    XW += x[8]*34.014740; /*H2O2 */
    XW += x[9]*44.009950; /*CO2 */
    XW += x[10]*28.010550; /*CO */
    XW += x[11]*30.026490; /*CH2O */
    XW += x[12]*15.035060; /*CH3 */
    XW += x[13]*16.043030; /*CH4 */
    XW += x[14]*32.042430; /*CH3OH */
    XW += x[15]*30.070120; /*C2H6 */
    XW += x[16]*42.037640; /*CH2CO */
    XW += x[17]*46.025890; /*HOCHO */
    XW += x[18]*26.038240; /*C2H2 */
    XW += x[19]*42.081270; /*C3H6 */
    XW += x[20]*28.054180; /*C2H4 */
    XW += x[21]*40.065330; /*C3H4XA */
    XW += x[22]*54.092420; /*C4H6 */
    XW += x[23]*56.108360; /*C4H8X1 */
    XW += x[24]*100.205570; /*NXC7H16 */

    /*Now compute conversion */
    double XWinv = 1.0/XW;
    y[0] = x[0]*28.013400*XWinv; 
    y[1] = x[1]*15.999400*XWinv; 
    y[2] = x[2]*2.015940*XWinv; 
    y[3] = x[3]*1.007970*XWinv; 
    y[4] = x[4]*17.007370*XWinv; 
    y[5] = x[5]*18.015340*XWinv; 
    y[6] = x[6]*31.998800*XWinv; 
    y[7] = x[7]*33.006770*XWinv; 
    y[8] = x[8]*34.014740*XWinv; 
    y[9] = x[9]*44.009950*XWinv; 
    y[10] = x[10]*28.010550*XWinv; 
    y[11] = x[11]*30.026490*XWinv; 
    y[12] = x[12]*15.035060*XWinv; 
    y[13] = x[13]*16.043030*XWinv; 
    y[14] = x[14]*32.042430*XWinv; 
    y[15] = x[15]*30.070120*XWinv; 
    y[16] = x[16]*42.037640*XWinv; 
    y[17] = x[17]*46.025890*XWinv; 
    y[18] = x[18]*26.038240*XWinv; 
    y[19] = x[19]*42.081270*XWinv; 
    y[20] = x[20]*28.054180*XWinv; 
    y[21] = x[21]*40.065330*XWinv; 
    y[22] = x[22]*54.092420*XWinv; 
    y[23] = x[23]*56.108360*XWinv; 
    y[24] = x[24]*100.205570*XWinv; 

    return;
}


/*convert x[species] (mole fracs) to c[species] (molar conc) */
void CKXTCP(double *  P, double *  T, double *  x,  double *  c)
{
    int id; /*loop counter */
    double PORT = (*P)/(8.31446e+07 * (*T)); /*P/RT */

    /*Compute conversion, see Eq 10 */
    for (id = 0; id < 25; ++id) {
        c[id] = x[id]*PORT;
    }

    return;
}


/*convert x[species] (mole fracs) to c[species] (molar conc) */
void CKXTCR(double *  rho, double *  T, double *  x, double *  c)
{
    int id; /*loop counter */
    double XW = 0; /*See Eq 4, 11 in CK Manual */
    double ROW; 
    /*Compute mean molecular wt first */
    XW += x[0]*28.013400; /*N2 */
    XW += x[1]*15.999400; /*O */
    XW += x[2]*2.015940; /*H2 */
    XW += x[3]*1.007970; /*H */
    XW += x[4]*17.007370; /*OH */
    XW += x[5]*18.015340; /*H2O */
    XW += x[6]*31.998800; /*O2 */
    XW += x[7]*33.006770; /*HO2 */
    XW += x[8]*34.014740; /*H2O2 */
    XW += x[9]*44.009950; /*CO2 */
    XW += x[10]*28.010550; /*CO */
    XW += x[11]*30.026490; /*CH2O */
    XW += x[12]*15.035060; /*CH3 */
    XW += x[13]*16.043030; /*CH4 */
    XW += x[14]*32.042430; /*CH3OH */
    XW += x[15]*30.070120; /*C2H6 */
    XW += x[16]*42.037640; /*CH2CO */
    XW += x[17]*46.025890; /*HOCHO */
    XW += x[18]*26.038240; /*C2H2 */
    XW += x[19]*42.081270; /*C3H6 */
    XW += x[20]*28.054180; /*C2H4 */
    XW += x[21]*40.065330; /*C3H4XA */
    XW += x[22]*54.092420; /*C4H6 */
    XW += x[23]*56.108360; /*C4H8X1 */
    XW += x[24]*100.205570; /*NXC7H16 */
    ROW = (*rho) / XW;

    /*Compute conversion, see Eq 11 */
    for (id = 0; id < 25; ++id) {
        c[id] = x[id]*ROW;
    }

    return;
}


/*convert c[species] (molar conc) to x[species] (mole fracs) */
void CKCTX(double *  c, double *  x)
{
    int id; /*loop counter */
    double sumC = 0; 

    /*compute sum of c  */
    for (id = 0; id < 25; ++id) {
        sumC += c[id];
    }

    /* See Eq 13  */
    double sumCinv = 1.0/sumC;
    for (id = 0; id < 25; ++id) {
        x[id] = c[id]*sumCinv;
    }

    return;
}


/*convert c[species] (molar conc) to y[species] (mass fracs) */
void CKCTY(double *  c, double *  y)
{
    double CW = 0; /*See Eq 12 in CK Manual */
    /*compute denominator in eq 12 first */
    CW += c[0]*28.013400; /*N2 */
    CW += c[1]*15.999400; /*O */
    CW += c[2]*2.015940; /*H2 */
    CW += c[3]*1.007970; /*H */
    CW += c[4]*17.007370; /*OH */
    CW += c[5]*18.015340; /*H2O */
    CW += c[6]*31.998800; /*O2 */
    CW += c[7]*33.006770; /*HO2 */
    CW += c[8]*34.014740; /*H2O2 */
    CW += c[9]*44.009950; /*CO2 */
    CW += c[10]*28.010550; /*CO */
    CW += c[11]*30.026490; /*CH2O */
    CW += c[12]*15.035060; /*CH3 */
    CW += c[13]*16.043030; /*CH4 */
    CW += c[14]*32.042430; /*CH3OH */
    CW += c[15]*30.070120; /*C2H6 */
    CW += c[16]*42.037640; /*CH2CO */
    CW += c[17]*46.025890; /*HOCHO */
    CW += c[18]*26.038240; /*C2H2 */
    CW += c[19]*42.081270; /*C3H6 */
    CW += c[20]*28.054180; /*C2H4 */
    CW += c[21]*40.065330; /*C3H4XA */
    CW += c[22]*54.092420; /*C4H6 */
    CW += c[23]*56.108360; /*C4H8X1 */
    CW += c[24]*100.205570; /*NXC7H16 */
    /*Now compute conversion */
    double CWinv = 1.0/CW;
    y[0] = c[0]*28.013400*CWinv; 
    y[1] = c[1]*15.999400*CWinv; 
    y[2] = c[2]*2.015940*CWinv; 
    y[3] = c[3]*1.007970*CWinv; 
    y[4] = c[4]*17.007370*CWinv; 
    y[5] = c[5]*18.015340*CWinv; 
    y[6] = c[6]*31.998800*CWinv; 
    y[7] = c[7]*33.006770*CWinv; 
    y[8] = c[8]*34.014740*CWinv; 
    y[9] = c[9]*44.009950*CWinv; 
    y[10] = c[10]*28.010550*CWinv; 
    y[11] = c[11]*30.026490*CWinv; 
    y[12] = c[12]*15.035060*CWinv; 
    y[13] = c[13]*16.043030*CWinv; 
    y[14] = c[14]*32.042430*CWinv; 
    y[15] = c[15]*30.070120*CWinv; 
    y[16] = c[16]*42.037640*CWinv; 
    y[17] = c[17]*46.025890*CWinv; 
    y[18] = c[18]*26.038240*CWinv; 
    y[19] = c[19]*42.081270*CWinv; 
    y[20] = c[20]*28.054180*CWinv; 
    y[21] = c[21]*40.065330*CWinv; 
    y[22] = c[22]*54.092420*CWinv; 
    y[23] = c[23]*56.108360*CWinv; 
    y[24] = c[24]*100.205570*CWinv; 

    return;
}


/*get Cp/R as a function of T  */
/*for all species (Eq 19) */
void CKCPOR(double *  T, double *  cpor)
{
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    cp_R(cpor, tc);
}


/*get H/RT as a function of T  */
/*for all species (Eq 20) */
void CKHORT(double *  T, double *  hort)
{
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    speciesEnthalpy(hort, tc);
}


/*get S/R as a function of T  */
/*for all species (Eq 21) */
void CKSOR(double *  T, double *  sor)
{
    double tT = *T; /*temporary temperature */
    double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    speciesEntropy(sor, tc);
}


/*Returns the specific heats at constant volume */
/*in mass units (Eq. 29) */
AMREX_GPU_HOST_DEVICE void CKCVMS(double *  T,  double *  cvms)
{
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    cv_R(cvms, tc);
    /*multiply by R/molecularweight */
    cvms[0] *= 2.968030520448514e+06; /*N2 */
    cvms[1] *= 5.196734013871295e+06; /*O */
    cvms[2] *= 4.124360158612479e+07; /*H2 */
    cvms[3] *= 8.248720317224957e+07; /*H */
    cvms[4] *= 4.888740950630956e+06; /*OH */
    cvms[5] *= 4.615212712140454e+06; /*H2O */
    cvms[6] *= 2.598367006935648e+06; /*O2 */
    cvms[7] *= 2.519017346487778e+06; /*HO2 */
    cvms[8] *= 2.444370475315478e+06; /*H2O2 */
    cvms[9] *= 1.889223372931176e+06; /*CO2 */
    cvms[10] *= 2.968332509769797e+06; /*CO */
    cvms[11] *= 2.769042474879095e+06; /*CH2O */
    cvms[12] *= 5.530049509714786e+06; /*CH3 */
    cvms[13] *= 5.182601178301878e+06; /*CH4 */
    cvms[14] *= 2.594828987112788e+06; /*CH3OH */
    cvms[15] *= 2.765024754857393e+06; /*C2H6 */
    cvms[16] *= 1.977861416138784e+06; /*CH2CO */
    cvms[17] *= 1.806475142176118e+06; /*HOCHO */
    cvms[18] *= 3.193173815954242e+06; /*C2H2 */
    cvms[19] *= 1.975810762876985e+06; /*C3H6 */
    cvms[20] *= 2.963716144315478e+06; /*C2H4 */
    cvms[21] *= 2.075226291198210e+06; /*C3H4XA */
    cvms[22] *= 1.537084607816260e+06; /*C4H6 */
    cvms[23] *= 1.481858072157739e+06; /*C4H8X1 */
    cvms[24] *= 8.297405641376262e+05; /*NXC7H16 */
}


/*Returns the specific heats at constant pressure */
/*in mass units (Eq. 26) */
AMREX_GPU_HOST_DEVICE void CKCPMS(double *  T,  double *  cpms)
{
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    cp_R(cpms, tc);
    /*multiply by R/molecularweight */
    cpms[0] *= 2.968030520448514e+06; /*N2 */
    cpms[1] *= 5.196734013871295e+06; /*O */
    cpms[2] *= 4.124360158612479e+07; /*H2 */
    cpms[3] *= 8.248720317224957e+07; /*H */
    cpms[4] *= 4.888740950630956e+06; /*OH */
    cpms[5] *= 4.615212712140454e+06; /*H2O */
    cpms[6] *= 2.598367006935648e+06; /*O2 */
    cpms[7] *= 2.519017346487778e+06; /*HO2 */
    cpms[8] *= 2.444370475315478e+06; /*H2O2 */
    cpms[9] *= 1.889223372931176e+06; /*CO2 */
    cpms[10] *= 2.968332509769797e+06; /*CO */
    cpms[11] *= 2.769042474879095e+06; /*CH2O */
    cpms[12] *= 5.530049509714786e+06; /*CH3 */
    cpms[13] *= 5.182601178301878e+06; /*CH4 */
    cpms[14] *= 2.594828987112788e+06; /*CH3OH */
    cpms[15] *= 2.765024754857393e+06; /*C2H6 */
    cpms[16] *= 1.977861416138784e+06; /*CH2CO */
    cpms[17] *= 1.806475142176118e+06; /*HOCHO */
    cpms[18] *= 3.193173815954242e+06; /*C2H2 */
    cpms[19] *= 1.975810762876985e+06; /*C3H6 */
    cpms[20] *= 2.963716144315478e+06; /*C2H4 */
    cpms[21] *= 2.075226291198210e+06; /*C3H4XA */
    cpms[22] *= 1.537084607816260e+06; /*C4H6 */
    cpms[23] *= 1.481858072157739e+06; /*C4H8X1 */
    cpms[24] *= 8.297405641376262e+05; /*NXC7H16 */
}


/*Returns internal energy in mass units (Eq 30.) */
AMREX_GPU_HOST_DEVICE void CKUMS(double *  T,  double *  ums)
{
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double RT = 8.31446e+07*tT; /*R*T */
    speciesInternalEnergy(ums, tc);
    for (int i = 0; i < 25; i++)
    {
        ums[i] *= RT*imw[i];
    }
}


/*Returns enthalpy in mass units (Eq 27.) */
AMREX_GPU_HOST_DEVICE void CKHMS(double *  T,  double *  hms)
{
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double RT = 8.31446e+07*tT; /*R*T */
    speciesEnthalpy(hms, tc);
    for (int i = 0; i < 25; i++)
    {
        hms[i] *= RT*imw[i];
    }
}


/*Returns gibbs in mass units (Eq 31.) */
void CKGMS(double *  T,  double *  gms)
{
    double tT = *T; /*temporary temperature */
    double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double RT = 8.31446e+07*tT; /*R*T */
    gibbs(gms, tc);
    for (int i = 0; i < 25; i++)
    {
        gms[i] *= RT*imw[i];
    }
}


/*Returns helmholtz in mass units (Eq 32.) */
void CKAMS(double *  T,  double *  ams)
{
    double tT = *T; /*temporary temperature */
    double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double RT = 8.31446e+07*tT; /*R*T */
    helmholtz(ams, tc);
    for (int i = 0; i < 25; i++)
    {
        ams[i] *= RT*imw[i];
    }
}


/*Returns the entropies in mass units (Eq 28.) */
void CKSMS(double *  T,  double *  sms)
{
    double tT = *T; /*temporary temperature */
    double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    speciesEntropy(sms, tc);
    /*multiply by R/molecularweight */
    sms[0] *= 2.968030520448514e+06; /*N2 */
    sms[1] *= 5.196734013871295e+06; /*O */
    sms[2] *= 4.124360158612479e+07; /*H2 */
    sms[3] *= 8.248720317224957e+07; /*H */
    sms[4] *= 4.888740950630956e+06; /*OH */
    sms[5] *= 4.615212712140454e+06; /*H2O */
    sms[6] *= 2.598367006935648e+06; /*O2 */
    sms[7] *= 2.519017346487778e+06; /*HO2 */
    sms[8] *= 2.444370475315478e+06; /*H2O2 */
    sms[9] *= 1.889223372931176e+06; /*CO2 */
    sms[10] *= 2.968332509769797e+06; /*CO */
    sms[11] *= 2.769042474879095e+06; /*CH2O */
    sms[12] *= 5.530049509714786e+06; /*CH3 */
    sms[13] *= 5.182601178301878e+06; /*CH4 */
    sms[14] *= 2.594828987112788e+06; /*CH3OH */
    sms[15] *= 2.765024754857393e+06; /*C2H6 */
    sms[16] *= 1.977861416138784e+06; /*CH2CO */
    sms[17] *= 1.806475142176118e+06; /*HOCHO */
    sms[18] *= 3.193173815954242e+06; /*C2H2 */
    sms[19] *= 1.975810762876985e+06; /*C3H6 */
    sms[20] *= 2.963716144315478e+06; /*C2H4 */
    sms[21] *= 2.075226291198210e+06; /*C3H4XA */
    sms[22] *= 1.537084607816260e+06; /*C4H6 */
    sms[23] *= 1.481858072157739e+06; /*C4H8X1 */
    sms[24] *= 8.297405641376262e+05; /*NXC7H16 */
}


/*Returns the mean specific heat at CP (Eq. 33) */
void CKCPBL(double *  T, double *  x,  double *  cpbl)
{
    int id; /*loop counter */
    double result = 0; 
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double cpor[25]; /* temporary storage */
    cp_R(cpor, tc);

    /*perform dot product */
    for (id = 0; id < 25; ++id) {
        result += x[id]*cpor[id];
    }

    *cpbl = result * 8.31446e+07;
}


/*Returns the mean specific heat at CP (Eq. 34) */
AMREX_GPU_HOST_DEVICE void CKCPBS(double *  T, double *  y,  double *  cpbs)
{
    double result = 0; 
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double cpor[25], tresult[25]; /* temporary storage */
    cp_R(cpor, tc);
    for (int i = 0; i < 25; i++)
    {
        tresult[i] = cpor[i]*y[i]*imw[i];

    }
    for (int i = 0; i < 25; i++)
    {
        result += tresult[i];
    }

    *cpbs = result * 8.31446e+07;
}


/*Returns the mean specific heat at CV (Eq. 35) */
void CKCVBL(double *  T, double *  x,  double *  cvbl)
{
    int id; /*loop counter */
    double result = 0; 
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double cvor[25]; /* temporary storage */
    cv_R(cvor, tc);

    /*perform dot product */
    for (id = 0; id < 25; ++id) {
        result += x[id]*cvor[id];
    }

    *cvbl = result * 8.31446e+07;
}


/*Returns the mean specific heat at CV (Eq. 36) */
AMREX_GPU_HOST_DEVICE void CKCVBS(double *  T, double *  y,  double *  cvbs)
{
    double result = 0; 
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double cvor[25]; /* temporary storage */
    cv_R(cvor, tc);
    /*multiply by y/molecularweight */
    result += cvor[0]*y[0]*imw[0]; /*N2 */
    result += cvor[1]*y[1]*imw[1]; /*O */
    result += cvor[2]*y[2]*imw[2]; /*H2 */
    result += cvor[3]*y[3]*imw[3]; /*H */
    result += cvor[4]*y[4]*imw[4]; /*OH */
    result += cvor[5]*y[5]*imw[5]; /*H2O */
    result += cvor[6]*y[6]*imw[6]; /*O2 */
    result += cvor[7]*y[7]*imw[7]; /*HO2 */
    result += cvor[8]*y[8]*imw[8]; /*H2O2 */
    result += cvor[9]*y[9]*imw[9]; /*CO2 */
    result += cvor[10]*y[10]*imw[10]; /*CO */
    result += cvor[11]*y[11]*imw[11]; /*CH2O */
    result += cvor[12]*y[12]*imw[12]; /*CH3 */
    result += cvor[13]*y[13]*imw[13]; /*CH4 */
    result += cvor[14]*y[14]*imw[14]; /*CH3OH */
    result += cvor[15]*y[15]*imw[15]; /*C2H6 */
    result += cvor[16]*y[16]*imw[16]; /*CH2CO */
    result += cvor[17]*y[17]*imw[17]; /*HOCHO */
    result += cvor[18]*y[18]*imw[18]; /*C2H2 */
    result += cvor[19]*y[19]*imw[19]; /*C3H6 */
    result += cvor[20]*y[20]*imw[20]; /*C2H4 */
    result += cvor[21]*y[21]*imw[21]; /*C3H4XA */
    result += cvor[22]*y[22]*imw[22]; /*C4H6 */
    result += cvor[23]*y[23]*imw[23]; /*C4H8X1 */
    result += cvor[24]*y[24]*imw[24]; /*NXC7H16 */

    *cvbs = result * 8.31446e+07;
}


/*Returns the mean enthalpy of the mixture in molar units */
void CKHBML(double *  T, double *  x,  double *  hbml)
{
    int id; /*loop counter */
    double result = 0; 
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double hml[25]; /* temporary storage */
    double RT = 8.31446e+07*tT; /*R*T */
    speciesEnthalpy(hml, tc);

    /*perform dot product */
    for (id = 0; id < 25; ++id) {
        result += x[id]*hml[id];
    }

    *hbml = result * RT;
}


/*Returns mean enthalpy of mixture in mass units */
AMREX_GPU_HOST_DEVICE void CKHBMS(double *  T, double *  y,  double *  hbms)
{
    double result = 0;
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double hml[25], tmp[25]; /* temporary storage */
    double RT = 8.31446e+07*tT; /*R*T */
    speciesEnthalpy(hml, tc);
    int id;
    for (id = 0; id < 25; ++id) {
        tmp[id] = y[id]*hml[id]*imw[id];
    }
    for (id = 0; id < 25; ++id) {
        result += tmp[id];
    }

    *hbms = result * RT;
}


/*get mean internal energy in molar units */
void CKUBML(double *  T, double *  x,  double *  ubml)
{
    int id; /*loop counter */
    double result = 0; 
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double uml[25]; /* temporary energy array */
    double RT = 8.31446e+07*tT; /*R*T */
    speciesInternalEnergy(uml, tc);

    /*perform dot product */
    for (id = 0; id < 25; ++id) {
        result += x[id]*uml[id];
    }

    *ubml = result * RT;
}


/*get mean internal energy in mass units */
AMREX_GPU_HOST_DEVICE void CKUBMS(double *  T, double *  y,  double *  ubms)
{
    double result = 0;
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double ums[25]; /* temporary energy array */
    double RT = 8.31446e+07*tT; /*R*T */
    speciesInternalEnergy(ums, tc);
    /*perform dot product + scaling by wt */
    result += y[0]*ums[0]*imw[0]; /*N2 */
    result += y[1]*ums[1]*imw[1]; /*O */
    result += y[2]*ums[2]*imw[2]; /*H2 */
    result += y[3]*ums[3]*imw[3]; /*H */
    result += y[4]*ums[4]*imw[4]; /*OH */
    result += y[5]*ums[5]*imw[5]; /*H2O */
    result += y[6]*ums[6]*imw[6]; /*O2 */
    result += y[7]*ums[7]*imw[7]; /*HO2 */
    result += y[8]*ums[8]*imw[8]; /*H2O2 */
    result += y[9]*ums[9]*imw[9]; /*CO2 */
    result += y[10]*ums[10]*imw[10]; /*CO */
    result += y[11]*ums[11]*imw[11]; /*CH2O */
    result += y[12]*ums[12]*imw[12]; /*CH3 */
    result += y[13]*ums[13]*imw[13]; /*CH4 */
    result += y[14]*ums[14]*imw[14]; /*CH3OH */
    result += y[15]*ums[15]*imw[15]; /*C2H6 */
    result += y[16]*ums[16]*imw[16]; /*CH2CO */
    result += y[17]*ums[17]*imw[17]; /*HOCHO */
    result += y[18]*ums[18]*imw[18]; /*C2H2 */
    result += y[19]*ums[19]*imw[19]; /*C3H6 */
    result += y[20]*ums[20]*imw[20]; /*C2H4 */
    result += y[21]*ums[21]*imw[21]; /*C3H4XA */
    result += y[22]*ums[22]*imw[22]; /*C4H6 */
    result += y[23]*ums[23]*imw[23]; /*C4H8X1 */
    result += y[24]*ums[24]*imw[24]; /*NXC7H16 */

    *ubms = result * RT;
}


/*get mixture entropy in molar units */
void CKSBML(double *  P, double *  T, double *  x,  double *  sbml)
{
    int id; /*loop counter */
    double result = 0; 
    /*Log of normalized pressure in cgs units dynes/cm^2 by Patm */
    double logPratio = log ( *P / 1013250.0 ); 
    double tT = *T; /*temporary temperature */
    double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double sor[25]; /* temporary storage */
    speciesEntropy(sor, tc);

    /*Compute Eq 42 */
    for (id = 0; id < 25; ++id) {
        result += x[id]*(sor[id]-log((x[id]+1e-100))-logPratio);
    }

    *sbml = result * 8.31446e+07;
}


/*get mixture entropy in mass units */
void CKSBMS(double *  P, double *  T, double *  y,  double *  sbms)
{
    double result = 0; 
    /*Log of normalized pressure in cgs units dynes/cm^2 by Patm */
    double logPratio = log ( *P / 1013250.0 ); 
    double tT = *T; /*temporary temperature */
    double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double sor[25]; /* temporary storage */
    double x[25]; /* need a ytx conversion */
    double YOW = 0; /*See Eq 4, 6 in CK Manual */
    /*Compute inverse of mean molecular wt first */
    YOW += y[0]*imw[0]; /*N2 */
    YOW += y[1]*imw[1]; /*O */
    YOW += y[2]*imw[2]; /*H2 */
    YOW += y[3]*imw[3]; /*H */
    YOW += y[4]*imw[4]; /*OH */
    YOW += y[5]*imw[5]; /*H2O */
    YOW += y[6]*imw[6]; /*O2 */
    YOW += y[7]*imw[7]; /*HO2 */
    YOW += y[8]*imw[8]; /*H2O2 */
    YOW += y[9]*imw[9]; /*CO2 */
    YOW += y[10]*imw[10]; /*CO */
    YOW += y[11]*imw[11]; /*CH2O */
    YOW += y[12]*imw[12]; /*CH3 */
    YOW += y[13]*imw[13]; /*CH4 */
    YOW += y[14]*imw[14]; /*CH3OH */
    YOW += y[15]*imw[15]; /*C2H6 */
    YOW += y[16]*imw[16]; /*CH2CO */
    YOW += y[17]*imw[17]; /*HOCHO */
    YOW += y[18]*imw[18]; /*C2H2 */
    YOW += y[19]*imw[19]; /*C3H6 */
    YOW += y[20]*imw[20]; /*C2H4 */
    YOW += y[21]*imw[21]; /*C3H4XA */
    YOW += y[22]*imw[22]; /*C4H6 */
    YOW += y[23]*imw[23]; /*C4H8X1 */
    YOW += y[24]*imw[24]; /*NXC7H16 */
    /*Now compute y to x conversion */
    x[0] = y[0]/(28.013400*YOW); 
    x[1] = y[1]/(15.999400*YOW); 
    x[2] = y[2]/(2.015940*YOW); 
    x[3] = y[3]/(1.007970*YOW); 
    x[4] = y[4]/(17.007370*YOW); 
    x[5] = y[5]/(18.015340*YOW); 
    x[6] = y[6]/(31.998800*YOW); 
    x[7] = y[7]/(33.006770*YOW); 
    x[8] = y[8]/(34.014740*YOW); 
    x[9] = y[9]/(44.009950*YOW); 
    x[10] = y[10]/(28.010550*YOW); 
    x[11] = y[11]/(30.026490*YOW); 
    x[12] = y[12]/(15.035060*YOW); 
    x[13] = y[13]/(16.043030*YOW); 
    x[14] = y[14]/(32.042430*YOW); 
    x[15] = y[15]/(30.070120*YOW); 
    x[16] = y[16]/(42.037640*YOW); 
    x[17] = y[17]/(46.025890*YOW); 
    x[18] = y[18]/(26.038240*YOW); 
    x[19] = y[19]/(42.081270*YOW); 
    x[20] = y[20]/(28.054180*YOW); 
    x[21] = y[21]/(40.065330*YOW); 
    x[22] = y[22]/(54.092420*YOW); 
    x[23] = y[23]/(56.108360*YOW); 
    x[24] = y[24]/(100.205570*YOW); 
    speciesEntropy(sor, tc);
    /*Perform computation in Eq 42 and 43 */
    result += x[0]*(sor[0]-log((x[0]+1e-100))-logPratio);
    result += x[1]*(sor[1]-log((x[1]+1e-100))-logPratio);
    result += x[2]*(sor[2]-log((x[2]+1e-100))-logPratio);
    result += x[3]*(sor[3]-log((x[3]+1e-100))-logPratio);
    result += x[4]*(sor[4]-log((x[4]+1e-100))-logPratio);
    result += x[5]*(sor[5]-log((x[5]+1e-100))-logPratio);
    result += x[6]*(sor[6]-log((x[6]+1e-100))-logPratio);
    result += x[7]*(sor[7]-log((x[7]+1e-100))-logPratio);
    result += x[8]*(sor[8]-log((x[8]+1e-100))-logPratio);
    result += x[9]*(sor[9]-log((x[9]+1e-100))-logPratio);
    result += x[10]*(sor[10]-log((x[10]+1e-100))-logPratio);
    result += x[11]*(sor[11]-log((x[11]+1e-100))-logPratio);
    result += x[12]*(sor[12]-log((x[12]+1e-100))-logPratio);
    result += x[13]*(sor[13]-log((x[13]+1e-100))-logPratio);
    result += x[14]*(sor[14]-log((x[14]+1e-100))-logPratio);
    result += x[15]*(sor[15]-log((x[15]+1e-100))-logPratio);
    result += x[16]*(sor[16]-log((x[16]+1e-100))-logPratio);
    result += x[17]*(sor[17]-log((x[17]+1e-100))-logPratio);
    result += x[18]*(sor[18]-log((x[18]+1e-100))-logPratio);
    result += x[19]*(sor[19]-log((x[19]+1e-100))-logPratio);
    result += x[20]*(sor[20]-log((x[20]+1e-100))-logPratio);
    result += x[21]*(sor[21]-log((x[21]+1e-100))-logPratio);
    result += x[22]*(sor[22]-log((x[22]+1e-100))-logPratio);
    result += x[23]*(sor[23]-log((x[23]+1e-100))-logPratio);
    result += x[24]*(sor[24]-log((x[24]+1e-100))-logPratio);
    /*Scale by R/W */
    *sbms = result * 8.31446e+07 * YOW;
}


/*Returns mean gibbs free energy in molar units */
void CKGBML(double *  P, double *  T, double *  x,  double *  gbml)
{
    int id; /*loop counter */
    double result = 0; 
    /*Log of normalized pressure in cgs units dynes/cm^2 by Patm */
    double logPratio = log ( *P / 1013250.0 ); 
    double tT = *T; /*temporary temperature */
    double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double RT = 8.31446e+07*tT; /*R*T */
    double gort[25]; /* temporary storage */
    /*Compute g/RT */
    gibbs(gort, tc);

    /*Compute Eq 44 */
    for (id = 0; id < 25; ++id) {
        result += x[id]*(gort[id]+log((x[id]+1e-100))+logPratio);
    }

    *gbml = result * RT;
}


/*Returns mixture gibbs free energy in mass units */
void CKGBMS(double *  P, double *  T, double *  y,  double *  gbms)
{
    double result = 0; 
    /*Log of normalized pressure in cgs units dynes/cm^2 by Patm */
    double logPratio = log ( *P / 1013250.0 ); 
    double tT = *T; /*temporary temperature */
    double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double RT = 8.31446e+07*tT; /*R*T */
    double gort[25]; /* temporary storage */
    double x[25]; /* need a ytx conversion */
    double YOW = 0; /*To hold 1/molecularweight */
    /*Compute inverse of mean molecular wt first */
    YOW += y[0]*imw[0]; /*N2 */
    YOW += y[1]*imw[1]; /*O */
    YOW += y[2]*imw[2]; /*H2 */
    YOW += y[3]*imw[3]; /*H */
    YOW += y[4]*imw[4]; /*OH */
    YOW += y[5]*imw[5]; /*H2O */
    YOW += y[6]*imw[6]; /*O2 */
    YOW += y[7]*imw[7]; /*HO2 */
    YOW += y[8]*imw[8]; /*H2O2 */
    YOW += y[9]*imw[9]; /*CO2 */
    YOW += y[10]*imw[10]; /*CO */
    YOW += y[11]*imw[11]; /*CH2O */
    YOW += y[12]*imw[12]; /*CH3 */
    YOW += y[13]*imw[13]; /*CH4 */
    YOW += y[14]*imw[14]; /*CH3OH */
    YOW += y[15]*imw[15]; /*C2H6 */
    YOW += y[16]*imw[16]; /*CH2CO */
    YOW += y[17]*imw[17]; /*HOCHO */
    YOW += y[18]*imw[18]; /*C2H2 */
    YOW += y[19]*imw[19]; /*C3H6 */
    YOW += y[20]*imw[20]; /*C2H4 */
    YOW += y[21]*imw[21]; /*C3H4XA */
    YOW += y[22]*imw[22]; /*C4H6 */
    YOW += y[23]*imw[23]; /*C4H8X1 */
    YOW += y[24]*imw[24]; /*NXC7H16 */
    /*Now compute y to x conversion */
    x[0] = y[0]/(28.013400*YOW); 
    x[1] = y[1]/(15.999400*YOW); 
    x[2] = y[2]/(2.015940*YOW); 
    x[3] = y[3]/(1.007970*YOW); 
    x[4] = y[4]/(17.007370*YOW); 
    x[5] = y[5]/(18.015340*YOW); 
    x[6] = y[6]/(31.998800*YOW); 
    x[7] = y[7]/(33.006770*YOW); 
    x[8] = y[8]/(34.014740*YOW); 
    x[9] = y[9]/(44.009950*YOW); 
    x[10] = y[10]/(28.010550*YOW); 
    x[11] = y[11]/(30.026490*YOW); 
    x[12] = y[12]/(15.035060*YOW); 
    x[13] = y[13]/(16.043030*YOW); 
    x[14] = y[14]/(32.042430*YOW); 
    x[15] = y[15]/(30.070120*YOW); 
    x[16] = y[16]/(42.037640*YOW); 
    x[17] = y[17]/(46.025890*YOW); 
    x[18] = y[18]/(26.038240*YOW); 
    x[19] = y[19]/(42.081270*YOW); 
    x[20] = y[20]/(28.054180*YOW); 
    x[21] = y[21]/(40.065330*YOW); 
    x[22] = y[22]/(54.092420*YOW); 
    x[23] = y[23]/(56.108360*YOW); 
    x[24] = y[24]/(100.205570*YOW); 
    gibbs(gort, tc);
    /*Perform computation in Eq 44 */
    result += x[0]*(gort[0]+log((x[0]+1e-100))+logPratio);
    result += x[1]*(gort[1]+log((x[1]+1e-100))+logPratio);
    result += x[2]*(gort[2]+log((x[2]+1e-100))+logPratio);
    result += x[3]*(gort[3]+log((x[3]+1e-100))+logPratio);
    result += x[4]*(gort[4]+log((x[4]+1e-100))+logPratio);
    result += x[5]*(gort[5]+log((x[5]+1e-100))+logPratio);
    result += x[6]*(gort[6]+log((x[6]+1e-100))+logPratio);
    result += x[7]*(gort[7]+log((x[7]+1e-100))+logPratio);
    result += x[8]*(gort[8]+log((x[8]+1e-100))+logPratio);
    result += x[9]*(gort[9]+log((x[9]+1e-100))+logPratio);
    result += x[10]*(gort[10]+log((x[10]+1e-100))+logPratio);
    result += x[11]*(gort[11]+log((x[11]+1e-100))+logPratio);
    result += x[12]*(gort[12]+log((x[12]+1e-100))+logPratio);
    result += x[13]*(gort[13]+log((x[13]+1e-100))+logPratio);
    result += x[14]*(gort[14]+log((x[14]+1e-100))+logPratio);
    result += x[15]*(gort[15]+log((x[15]+1e-100))+logPratio);
    result += x[16]*(gort[16]+log((x[16]+1e-100))+logPratio);
    result += x[17]*(gort[17]+log((x[17]+1e-100))+logPratio);
    result += x[18]*(gort[18]+log((x[18]+1e-100))+logPratio);
    result += x[19]*(gort[19]+log((x[19]+1e-100))+logPratio);
    result += x[20]*(gort[20]+log((x[20]+1e-100))+logPratio);
    result += x[21]*(gort[21]+log((x[21]+1e-100))+logPratio);
    result += x[22]*(gort[22]+log((x[22]+1e-100))+logPratio);
    result += x[23]*(gort[23]+log((x[23]+1e-100))+logPratio);
    result += x[24]*(gort[24]+log((x[24]+1e-100))+logPratio);
    /*Scale by RT/W */
    *gbms = result * RT * YOW;
}


/*Returns mean helmholtz free energy in molar units */
void CKABML(double *  P, double *  T, double *  x,  double *  abml)
{
    int id; /*loop counter */
    double result = 0; 
    /*Log of normalized pressure in cgs units dynes/cm^2 by Patm */
    double logPratio = log ( *P / 1013250.0 ); 
    double tT = *T; /*temporary temperature */
    double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double RT = 8.31446e+07*tT; /*R*T */
    double aort[25]; /* temporary storage */
    /*Compute g/RT */
    helmholtz(aort, tc);

    /*Compute Eq 44 */
    for (id = 0; id < 25; ++id) {
        result += x[id]*(aort[id]+log((x[id]+1e-100))+logPratio);
    }

    *abml = result * RT;
}


/*Returns mixture helmholtz free energy in mass units */
void CKABMS(double *  P, double *  T, double *  y,  double *  abms)
{
    double result = 0; 
    /*Log of normalized pressure in cgs units dynes/cm^2 by Patm */
    double logPratio = log ( *P / 1013250.0 ); 
    double tT = *T; /*temporary temperature */
    double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double RT = 8.31446e+07*tT; /*R*T */
    double aort[25]; /* temporary storage */
    double x[25]; /* need a ytx conversion */
    double YOW = 0; /*To hold 1/molecularweight */
    /*Compute inverse of mean molecular wt first */
    YOW += y[0]*imw[0]; /*N2 */
    YOW += y[1]*imw[1]; /*O */
    YOW += y[2]*imw[2]; /*H2 */
    YOW += y[3]*imw[3]; /*H */
    YOW += y[4]*imw[4]; /*OH */
    YOW += y[5]*imw[5]; /*H2O */
    YOW += y[6]*imw[6]; /*O2 */
    YOW += y[7]*imw[7]; /*HO2 */
    YOW += y[8]*imw[8]; /*H2O2 */
    YOW += y[9]*imw[9]; /*CO2 */
    YOW += y[10]*imw[10]; /*CO */
    YOW += y[11]*imw[11]; /*CH2O */
    YOW += y[12]*imw[12]; /*CH3 */
    YOW += y[13]*imw[13]; /*CH4 */
    YOW += y[14]*imw[14]; /*CH3OH */
    YOW += y[15]*imw[15]; /*C2H6 */
    YOW += y[16]*imw[16]; /*CH2CO */
    YOW += y[17]*imw[17]; /*HOCHO */
    YOW += y[18]*imw[18]; /*C2H2 */
    YOW += y[19]*imw[19]; /*C3H6 */
    YOW += y[20]*imw[20]; /*C2H4 */
    YOW += y[21]*imw[21]; /*C3H4XA */
    YOW += y[22]*imw[22]; /*C4H6 */
    YOW += y[23]*imw[23]; /*C4H8X1 */
    YOW += y[24]*imw[24]; /*NXC7H16 */
    /*Now compute y to x conversion */
    x[0] = y[0]/(28.013400*YOW); 
    x[1] = y[1]/(15.999400*YOW); 
    x[2] = y[2]/(2.015940*YOW); 
    x[3] = y[3]/(1.007970*YOW); 
    x[4] = y[4]/(17.007370*YOW); 
    x[5] = y[5]/(18.015340*YOW); 
    x[6] = y[6]/(31.998800*YOW); 
    x[7] = y[7]/(33.006770*YOW); 
    x[8] = y[8]/(34.014740*YOW); 
    x[9] = y[9]/(44.009950*YOW); 
    x[10] = y[10]/(28.010550*YOW); 
    x[11] = y[11]/(30.026490*YOW); 
    x[12] = y[12]/(15.035060*YOW); 
    x[13] = y[13]/(16.043030*YOW); 
    x[14] = y[14]/(32.042430*YOW); 
    x[15] = y[15]/(30.070120*YOW); 
    x[16] = y[16]/(42.037640*YOW); 
    x[17] = y[17]/(46.025890*YOW); 
    x[18] = y[18]/(26.038240*YOW); 
    x[19] = y[19]/(42.081270*YOW); 
    x[20] = y[20]/(28.054180*YOW); 
    x[21] = y[21]/(40.065330*YOW); 
    x[22] = y[22]/(54.092420*YOW); 
    x[23] = y[23]/(56.108360*YOW); 
    x[24] = y[24]/(100.205570*YOW); 
    helmholtz(aort, tc);
    /*Perform computation in Eq 44 */
    result += x[0]*(aort[0]+log((x[0]+1e-100))+logPratio);
    result += x[1]*(aort[1]+log((x[1]+1e-100))+logPratio);
    result += x[2]*(aort[2]+log((x[2]+1e-100))+logPratio);
    result += x[3]*(aort[3]+log((x[3]+1e-100))+logPratio);
    result += x[4]*(aort[4]+log((x[4]+1e-100))+logPratio);
    result += x[5]*(aort[5]+log((x[5]+1e-100))+logPratio);
    result += x[6]*(aort[6]+log((x[6]+1e-100))+logPratio);
    result += x[7]*(aort[7]+log((x[7]+1e-100))+logPratio);
    result += x[8]*(aort[8]+log((x[8]+1e-100))+logPratio);
    result += x[9]*(aort[9]+log((x[9]+1e-100))+logPratio);
    result += x[10]*(aort[10]+log((x[10]+1e-100))+logPratio);
    result += x[11]*(aort[11]+log((x[11]+1e-100))+logPratio);
    result += x[12]*(aort[12]+log((x[12]+1e-100))+logPratio);
    result += x[13]*(aort[13]+log((x[13]+1e-100))+logPratio);
    result += x[14]*(aort[14]+log((x[14]+1e-100))+logPratio);
    result += x[15]*(aort[15]+log((x[15]+1e-100))+logPratio);
    result += x[16]*(aort[16]+log((x[16]+1e-100))+logPratio);
    result += x[17]*(aort[17]+log((x[17]+1e-100))+logPratio);
    result += x[18]*(aort[18]+log((x[18]+1e-100))+logPratio);
    result += x[19]*(aort[19]+log((x[19]+1e-100))+logPratio);
    result += x[20]*(aort[20]+log((x[20]+1e-100))+logPratio);
    result += x[21]*(aort[21]+log((x[21]+1e-100))+logPratio);
    result += x[22]*(aort[22]+log((x[22]+1e-100))+logPratio);
    result += x[23]*(aort[23]+log((x[23]+1e-100))+logPratio);
    result += x[24]*(aort[24]+log((x[24]+1e-100))+logPratio);
    /*Scale by RT/W */
    *abms = result * RT * YOW;
}


/*compute the production rate for each species */
AMREX_GPU_HOST_DEVICE void CKWC(double *  T, double *  C,  double *  wdot)
{
    int id; /*loop counter */

    /*convert to SI */
    for (id = 0; id < 25; ++id) {
        C[id] *= 1.0e6;
    }

    /*convert to chemkin units */
    productionRate(wdot, C, *T);

    /*convert to chemkin units */
    for (id = 0; id < 25; ++id) {
        C[id] *= 1.0e-6;
        wdot[id] *= 1.0e-6;
    }
}


/*Returns the molar production rate of species */
/*Given P, T, and mass fractions */
void CKWYP(double *  P, double *  T, double *  y,  double *  wdot)
{
    int id; /*loop counter */
    double c[25]; /*temporary storage */
    double YOW = 0; 
    double PWORT; 
    /*Compute inverse of mean molecular wt first */
    YOW += y[0]*imw[0]; /*N2 */
    YOW += y[1]*imw[1]; /*O */
    YOW += y[2]*imw[2]; /*H2 */
    YOW += y[3]*imw[3]; /*H */
    YOW += y[4]*imw[4]; /*OH */
    YOW += y[5]*imw[5]; /*H2O */
    YOW += y[6]*imw[6]; /*O2 */
    YOW += y[7]*imw[7]; /*HO2 */
    YOW += y[8]*imw[8]; /*H2O2 */
    YOW += y[9]*imw[9]; /*CO2 */
    YOW += y[10]*imw[10]; /*CO */
    YOW += y[11]*imw[11]; /*CH2O */
    YOW += y[12]*imw[12]; /*CH3 */
    YOW += y[13]*imw[13]; /*CH4 */
    YOW += y[14]*imw[14]; /*CH3OH */
    YOW += y[15]*imw[15]; /*C2H6 */
    YOW += y[16]*imw[16]; /*CH2CO */
    YOW += y[17]*imw[17]; /*HOCHO */
    YOW += y[18]*imw[18]; /*C2H2 */
    YOW += y[19]*imw[19]; /*C3H6 */
    YOW += y[20]*imw[20]; /*C2H4 */
    YOW += y[21]*imw[21]; /*C3H4XA */
    YOW += y[22]*imw[22]; /*C4H6 */
    YOW += y[23]*imw[23]; /*C4H8X1 */
    YOW += y[24]*imw[24]; /*NXC7H16 */
    /*PW/RT (see Eq. 7) */
    PWORT = (*P)/(YOW * 8.31446e+07 * (*T)); 
    /*multiply by 1e6 so c goes to SI */
    PWORT *= 1e6; 
    /*Now compute conversion (and go to SI) */
    c[0] = PWORT * y[0]*imw[0]; 
    c[1] = PWORT * y[1]*imw[1]; 
    c[2] = PWORT * y[2]*imw[2]; 
    c[3] = PWORT * y[3]*imw[3]; 
    c[4] = PWORT * y[4]*imw[4]; 
    c[5] = PWORT * y[5]*imw[5]; 
    c[6] = PWORT * y[6]*imw[6]; 
    c[7] = PWORT * y[7]*imw[7]; 
    c[8] = PWORT * y[8]*imw[8]; 
    c[9] = PWORT * y[9]*imw[9]; 
    c[10] = PWORT * y[10]*imw[10]; 
    c[11] = PWORT * y[11]*imw[11]; 
    c[12] = PWORT * y[12]*imw[12]; 
    c[13] = PWORT * y[13]*imw[13]; 
    c[14] = PWORT * y[14]*imw[14]; 
    c[15] = PWORT * y[15]*imw[15]; 
    c[16] = PWORT * y[16]*imw[16]; 
    c[17] = PWORT * y[17]*imw[17]; 
    c[18] = PWORT * y[18]*imw[18]; 
    c[19] = PWORT * y[19]*imw[19]; 
    c[20] = PWORT * y[20]*imw[20]; 
    c[21] = PWORT * y[21]*imw[21]; 
    c[22] = PWORT * y[22]*imw[22]; 
    c[23] = PWORT * y[23]*imw[23]; 
    c[24] = PWORT * y[24]*imw[24]; 

    /*convert to chemkin units */
    productionRate(wdot, c, *T);

    /*convert to chemkin units */
    for (id = 0; id < 25; ++id) {
        wdot[id] *= 1.0e-6;
    }
}


/*Returns the molar production rate of species */
/*Given P, T, and mole fractions */
void CKWXP(double *  P, double *  T, double *  x,  double *  wdot)
{
    int id; /*loop counter */
    double c[25]; /*temporary storage */
    double PORT = 1e6 * (*P)/(8.31446e+07 * (*T)); /*1e6 * P/RT so c goes to SI units */

    /*Compute conversion, see Eq 10 */
    for (id = 0; id < 25; ++id) {
        c[id] = x[id]*PORT;
    }

    /*convert to chemkin units */
    productionRate(wdot, c, *T);

    /*convert to chemkin units */
    for (id = 0; id < 25; ++id) {
        wdot[id] *= 1.0e-6;
    }
}


/*Returns the molar production rate of species */
/*Given rho, T, and mass fractions */
AMREX_GPU_HOST_DEVICE void CKWYR(double *  rho, double *  T, double *  y,  double *  wdot)
{
    int id; /*loop counter */
    double c[25]; /*temporary storage */
    /*See Eq 8 with an extra 1e6 so c goes to SI */
    c[0] = 1e6 * (*rho) * y[0]*imw[0]; 
    c[1] = 1e6 * (*rho) * y[1]*imw[1]; 
    c[2] = 1e6 * (*rho) * y[2]*imw[2]; 
    c[3] = 1e6 * (*rho) * y[3]*imw[3]; 
    c[4] = 1e6 * (*rho) * y[4]*imw[4]; 
    c[5] = 1e6 * (*rho) * y[5]*imw[5]; 
    c[6] = 1e6 * (*rho) * y[6]*imw[6]; 
    c[7] = 1e6 * (*rho) * y[7]*imw[7]; 
    c[8] = 1e6 * (*rho) * y[8]*imw[8]; 
    c[9] = 1e6 * (*rho) * y[9]*imw[9]; 
    c[10] = 1e6 * (*rho) * y[10]*imw[10]; 
    c[11] = 1e6 * (*rho) * y[11]*imw[11]; 
    c[12] = 1e6 * (*rho) * y[12]*imw[12]; 
    c[13] = 1e6 * (*rho) * y[13]*imw[13]; 
    c[14] = 1e6 * (*rho) * y[14]*imw[14]; 
    c[15] = 1e6 * (*rho) * y[15]*imw[15]; 
    c[16] = 1e6 * (*rho) * y[16]*imw[16]; 
    c[17] = 1e6 * (*rho) * y[17]*imw[17]; 
    c[18] = 1e6 * (*rho) * y[18]*imw[18]; 
    c[19] = 1e6 * (*rho) * y[19]*imw[19]; 
    c[20] = 1e6 * (*rho) * y[20]*imw[20]; 
    c[21] = 1e6 * (*rho) * y[21]*imw[21]; 
    c[22] = 1e6 * (*rho) * y[22]*imw[22]; 
    c[23] = 1e6 * (*rho) * y[23]*imw[23]; 
    c[24] = 1e6 * (*rho) * y[24]*imw[24]; 

    /*call productionRate */
    productionRate(wdot, c, *T);

    /*convert to chemkin units */
    for (id = 0; id < 25; ++id) {
        wdot[id] *= 1.0e-6;
    }
}


/*Returns the molar production rate of species */
/*Given rho, T, and mole fractions */
void CKWXR(double *  rho, double *  T, double *  x,  double *  wdot)
{
    int id; /*loop counter */
    double c[25]; /*temporary storage */
    double XW = 0; /*See Eq 4, 11 in CK Manual */
    double ROW; 
    /*Compute mean molecular wt first */
    XW += x[0]*28.013400; /*N2 */
    XW += x[1]*15.999400; /*O */
    XW += x[2]*2.015940; /*H2 */
    XW += x[3]*1.007970; /*H */
    XW += x[4]*17.007370; /*OH */
    XW += x[5]*18.015340; /*H2O */
    XW += x[6]*31.998800; /*O2 */
    XW += x[7]*33.006770; /*HO2 */
    XW += x[8]*34.014740; /*H2O2 */
    XW += x[9]*44.009950; /*CO2 */
    XW += x[10]*28.010550; /*CO */
    XW += x[11]*30.026490; /*CH2O */
    XW += x[12]*15.035060; /*CH3 */
    XW += x[13]*16.043030; /*CH4 */
    XW += x[14]*32.042430; /*CH3OH */
    XW += x[15]*30.070120; /*C2H6 */
    XW += x[16]*42.037640; /*CH2CO */
    XW += x[17]*46.025890; /*HOCHO */
    XW += x[18]*26.038240; /*C2H2 */
    XW += x[19]*42.081270; /*C3H6 */
    XW += x[20]*28.054180; /*C2H4 */
    XW += x[21]*40.065330; /*C3H4XA */
    XW += x[22]*54.092420; /*C4H6 */
    XW += x[23]*56.108360; /*C4H8X1 */
    XW += x[24]*100.205570; /*NXC7H16 */
    /*Extra 1e6 factor to take c to SI */
    ROW = 1e6*(*rho) / XW;

    /*Compute conversion, see Eq 11 */
    for (id = 0; id < 25; ++id) {
        c[id] = x[id]*ROW;
    }

    /*convert to chemkin units */
    productionRate(wdot, c, *T);

    /*convert to chemkin units */
    for (id = 0; id < 25; ++id) {
        wdot[id] *= 1.0e-6;
    }
}


/*Returns the elemental composition  */
/*of the species (mdim is num of elements) */
void CKNCF(int * ncf)
{
    int id; /*loop counter */
    int kd = 4; 
    /*Zero ncf */
    for (id = 0; id < kd * 25; ++ id) {
         ncf[id] = 0; 
    }

    /*N2 */
    ncf[ 0 * kd + 0 ] = 2; /*N */

    /*O */
    ncf[ 1 * kd + 1 ] = 1; /*O */

    /*H2 */
    ncf[ 2 * kd + 2 ] = 2; /*H */

    /*H */
    ncf[ 3 * kd + 2 ] = 1; /*H */

    /*OH */
    ncf[ 4 * kd + 2 ] = 1; /*H */
    ncf[ 4 * kd + 1 ] = 1; /*O */

    /*H2O */
    ncf[ 5 * kd + 2 ] = 2; /*H */
    ncf[ 5 * kd + 1 ] = 1; /*O */

    /*O2 */
    ncf[ 6 * kd + 1 ] = 2; /*O */

    /*HO2 */
    ncf[ 7 * kd + 2 ] = 1; /*H */
    ncf[ 7 * kd + 1 ] = 2; /*O */

    /*H2O2 */
    ncf[ 8 * kd + 2 ] = 2; /*H */
    ncf[ 8 * kd + 1 ] = 2; /*O */

    /*CO2 */
    ncf[ 9 * kd + 3 ] = 1; /*C */
    ncf[ 9 * kd + 1 ] = 2; /*O */

    /*CO */
    ncf[ 10 * kd + 3 ] = 1; /*C */
    ncf[ 10 * kd + 1 ] = 1; /*O */

    /*CH2O */
    ncf[ 11 * kd + 3 ] = 1; /*C */
    ncf[ 11 * kd + 2 ] = 2; /*H */
    ncf[ 11 * kd + 1 ] = 1; /*O */

    /*CH3 */
    ncf[ 12 * kd + 3 ] = 1; /*C */
    ncf[ 12 * kd + 2 ] = 3; /*H */

    /*CH4 */
    ncf[ 13 * kd + 3 ] = 1; /*C */
    ncf[ 13 * kd + 2 ] = 4; /*H */

    /*CH3OH */
    ncf[ 14 * kd + 3 ] = 1; /*C */
    ncf[ 14 * kd + 2 ] = 4; /*H */
    ncf[ 14 * kd + 1 ] = 1; /*O */

    /*C2H6 */
    ncf[ 15 * kd + 3 ] = 2; /*C */
    ncf[ 15 * kd + 2 ] = 6; /*H */

    /*CH2CO */
    ncf[ 16 * kd + 3 ] = 2; /*C */
    ncf[ 16 * kd + 2 ] = 2; /*H */
    ncf[ 16 * kd + 1 ] = 1; /*O */

    /*HOCHO */
    ncf[ 17 * kd + 3 ] = 1; /*C */
    ncf[ 17 * kd + 2 ] = 2; /*H */
    ncf[ 17 * kd + 1 ] = 2; /*O */

    /*C2H2 */
    ncf[ 18 * kd + 3 ] = 2; /*C */
    ncf[ 18 * kd + 2 ] = 2; /*H */

    /*C3H6 */
    ncf[ 19 * kd + 3 ] = 3; /*C */
    ncf[ 19 * kd + 2 ] = 6; /*H */

    /*C2H4 */
    ncf[ 20 * kd + 3 ] = 2; /*C */
    ncf[ 20 * kd + 2 ] = 4; /*H */

    /*C3H4XA */
    ncf[ 21 * kd + 2 ] = 4; /*H */
    ncf[ 21 * kd + 3 ] = 3; /*C */

    /*C4H6 */
    ncf[ 22 * kd + 3 ] = 4; /*C */
    ncf[ 22 * kd + 2 ] = 6; /*H */

    /*C4H8X1 */
    ncf[ 23 * kd + 3 ] = 4; /*C */
    ncf[ 23 * kd + 2 ] = 8; /*H */

    /*NXC7H16 */
    ncf[ 24 * kd + 3 ] = 7; /*C */
    ncf[ 24 * kd + 2 ] = 16; /*H */


}

#ifndef AMREX_USE_CUDA
static double T_save = -1;
#ifdef _OPENMP
#pragma omp threadprivate(T_save)
#endif

static double k_f_save[211];
#ifdef _OPENMP
#pragma omp threadprivate(k_f_save)
#endif

static double Kc_save[211];
#ifdef _OPENMP
#pragma omp threadprivate(Kc_save)
#endif

static double k_f_save_qss[167];
#ifdef _OPENMP
#pragma omp threadprivate(k_f_save)
#endif

static double Kc_save_qss[167];
#ifdef _OPENMP
#pragma omp threadprivate(Kc_save)
#endif


/*compute the production rate for each species pointwise on CPU */
void productionRate(double *  wdot, double *  sc, double T)
{
    double tc[] = { log(T), T, T*T, T*T*T, T*T*T*T }; /*temperature cache */
    double invT = 1.0 / tc[1];

    if (T != T_save)
    {
        T_save = T;
        comp_k_f(tc,invT,k_f_save);
        comp_Kc(tc,invT,Kc_save);

        comp_k_f_qss(tc,invT,k_f_save_qss);
        comp_Kc_qss(tc,invT,Kc_save_qss);
    }

    double qdot, q_f[211], q_r[211];
    double sc_qss[27];
    /* Fill sc_qss here*/
    comp_qss_sc(sc, sc_qss, tc, invT);
    comp_qfqr(q_f, q_r, sc, sc_qss, tc, invT);

    for (int i = 0; i < 25; ++i) {
        wdot[i] = 0.0;
    }

    qdot = q_f[0]-q_r[0];
    wdot[3] -= qdot;
    wdot[6] -= qdot;
    wdot[7] += qdot;

    qdot = q_f[1]-q_r[1];
    wdot[4] += 2.000000 * qdot;
    wdot[8] -= qdot;

    qdot = q_f[2]-q_r[2];
    wdot[4] -= qdot;
    wdot[12] -= qdot;
    wdot[14] += qdot;

    qdot = q_f[3]-q_r[3];
    wdot[3] -= qdot;
    wdot[12] -= qdot;
    wdot[13] += qdot;

    qdot = q_f[4]-q_r[4];
    wdot[12] -= 2.000000 * qdot;
    wdot[15] += qdot;

    qdot = q_f[5]-q_r[5];
    wdot[10] -= qdot;
    wdot[16] += qdot;

    qdot = q_f[6]-q_r[6];
    wdot[1] -= qdot;
    wdot[9] += qdot;
    wdot[10] -= qdot;

    qdot = q_f[7]-q_r[7];
    wdot[3] += qdot;
    wdot[11] += qdot;

    qdot = q_f[8]-q_r[8];
    wdot[3] += qdot;
    wdot[18] += qdot;

    qdot = q_f[9]-q_r[9];
    wdot[3] -= qdot;
    wdot[20] -= qdot;

    qdot = q_f[10]-q_r[10];
    wdot[10] += qdot;
    wdot[12] += qdot;

    qdot = q_f[11]-q_r[11];
    wdot[3] -= qdot;
    wdot[4] -= qdot;
    wdot[5] += qdot;

    qdot = q_f[12]-q_r[12];

    qdot = q_f[13]-q_r[13];

    qdot = q_f[14]-q_r[14];
    wdot[3] += qdot;
    wdot[10] += qdot;

    qdot = q_f[15]-q_r[15];
    wdot[6] += qdot;
    wdot[12] += qdot;

    qdot = q_f[16]-q_r[16];
    wdot[6] -= qdot;
    wdot[12] -= qdot;

    qdot = q_f[17]-q_r[17];
    wdot[11] += qdot;
    wdot[12] += qdot;

    qdot = q_f[18]-q_r[18];
    wdot[1] -= qdot;
    wdot[2] -= qdot;
    wdot[3] += qdot;
    wdot[4] += qdot;

    qdot = q_f[19]-q_r[19];
    wdot[1] += qdot;
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[4] -= qdot;

    qdot = q_f[20]-q_r[20];
    wdot[2] -= qdot;
    wdot[3] += qdot;
    wdot[4] -= qdot;
    wdot[5] += qdot;

    qdot = q_f[21]-q_r[21];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[4] += qdot;
    wdot[5] -= qdot;

    qdot = q_f[22]-q_r[22];
    wdot[1] -= qdot;
    wdot[4] += 2.000000 * qdot;
    wdot[5] -= qdot;

    qdot = q_f[23]-q_r[23];
    wdot[1] += qdot;
    wdot[4] -= 2.000000 * qdot;
    wdot[5] += qdot;

    qdot = q_f[24]-q_r[24];
    wdot[1] += qdot;
    wdot[3] -= qdot;
    wdot[4] += qdot;
    wdot[6] -= qdot;

    qdot = q_f[25]-q_r[25];
    wdot[1] -= qdot;
    wdot[3] += qdot;
    wdot[4] -= qdot;
    wdot[6] += qdot;

    qdot = q_f[26]-q_r[26];
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[6] += qdot;
    wdot[7] -= qdot;

    qdot = q_f[27]-q_r[27];
    wdot[1] -= qdot;
    wdot[4] += qdot;
    wdot[6] += qdot;
    wdot[7] -= qdot;

    qdot = q_f[28]-q_r[28];
    wdot[6] += qdot;
    wdot[7] -= 2.000000 * qdot;
    wdot[8] += qdot;

    qdot = q_f[29]-q_r[29];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[6] += qdot;
    wdot[7] -= qdot;

    qdot = q_f[30]-q_r[30];
    wdot[6] += qdot;
    wdot[7] -= 2.000000 * qdot;
    wdot[8] += qdot;

    qdot = q_f[31]-q_r[31];
    wdot[3] -= qdot;
    wdot[4] += 2.000000 * qdot;
    wdot[7] -= qdot;

    qdot = q_f[32]-q_r[32];
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[7] += qdot;
    wdot[8] -= qdot;

    qdot = q_f[33]-q_r[33];
    wdot[1] += qdot;
    wdot[6] -= qdot;

    qdot = q_f[34]-q_r[34];
    wdot[3] += 2.000000 * qdot;
    wdot[6] -= qdot;
    wdot[9] += qdot;

    qdot = q_f[35]-q_r[35];
    wdot[5] += qdot;
    wdot[6] -= qdot;
    wdot[10] += qdot;

    qdot = q_f[36]-q_r[36];
    wdot[1] -= qdot;
    wdot[3] += 2.000000 * qdot;
    wdot[10] += qdot;

    qdot = q_f[37]-q_r[37];
    wdot[1] += qdot;
    wdot[6] -= qdot;
    wdot[11] += qdot;

    qdot = q_f[38]-q_r[38];
    wdot[2] += qdot;
    wdot[3] -= qdot;

    qdot = q_f[39]-q_r[39];
    wdot[2] -= qdot;
    wdot[3] += qdot;

    qdot = q_f[40]-q_r[40];
    wdot[4] -= qdot;
    wdot[5] += qdot;

    qdot = q_f[41]-q_r[41];
    wdot[4] += qdot;
    wdot[5] -= qdot;

    qdot = q_f[42]-q_r[42];
    wdot[2] += qdot;
    wdot[6] -= qdot;
    wdot[9] += qdot;

    qdot = q_f[43]-q_r[43];
    wdot[2] += qdot;
    wdot[3] -= qdot;

    qdot = q_f[44]-q_r[44];
    wdot[2] -= qdot;
    wdot[3] += qdot;
    wdot[12] += qdot;

    qdot = q_f[45]-q_r[45];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[12] -= qdot;

    qdot = q_f[46]-q_r[46];
    wdot[3] += qdot;
    wdot[4] += qdot;
    wdot[6] -= qdot;
    wdot[10] += qdot;

    qdot = q_f[47]-q_r[47];
    wdot[3] += qdot;
    wdot[4] -= qdot;
    wdot[11] += qdot;

    qdot = q_f[48]-q_r[48];
    wdot[2] += qdot;
    wdot[4] -= qdot;
    wdot[11] += qdot;
    wdot[12] -= qdot;

    qdot = q_f[49]-q_r[49];
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[12] -= qdot;

    qdot = q_f[50]-q_r[50];
    wdot[4] += qdot;
    wdot[5] -= qdot;
    wdot[12] += qdot;

    qdot = q_f[51]-q_r[51];
    wdot[1] -= qdot;
    wdot[3] += qdot;
    wdot[11] += qdot;
    wdot[12] -= qdot;

    qdot = q_f[52]-q_r[52];
    wdot[4] += qdot;
    wdot[7] -= qdot;
    wdot[12] -= qdot;

    qdot = q_f[53]-q_r[53];
    wdot[6] += qdot;
    wdot[7] -= qdot;
    wdot[12] -= qdot;
    wdot[13] += qdot;

    qdot = q_f[54]-q_r[54];
    wdot[4] += qdot;
    wdot[6] -= qdot;
    wdot[11] += qdot;
    wdot[12] -= qdot;

    qdot = q_f[55]-q_r[55];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[12] -= qdot;

    qdot = q_f[56]-q_r[56];
    wdot[2] -= qdot;
    wdot[3] += qdot;
    wdot[12] += qdot;

    qdot = q_f[57]-q_r[57];
    wdot[3] += qdot;
    wdot[12] -= 2.000000 * qdot;

    qdot = q_f[58]-q_r[58];
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[12] -= qdot;

    qdot = q_f[59]-q_r[59];
    wdot[4] += qdot;
    wdot[5] -= qdot;
    wdot[12] += qdot;

    qdot = q_f[60]-q_r[60];
    wdot[1] -= qdot;
    wdot[4] += qdot;
    wdot[12] += qdot;
    wdot[13] -= qdot;

    qdot = q_f[61]-q_r[61];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[12] += qdot;
    wdot[13] -= qdot;

    qdot = q_f[62]-q_r[62];
    wdot[2] -= qdot;
    wdot[3] += qdot;
    wdot[12] -= qdot;
    wdot[13] += qdot;

    qdot = q_f[63]-q_r[63];
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[12] += qdot;
    wdot[13] -= qdot;

    qdot = q_f[64]-q_r[64];
    wdot[4] += qdot;
    wdot[5] -= qdot;
    wdot[12] -= qdot;
    wdot[13] += qdot;

    qdot = q_f[65]-q_r[65];
    wdot[3] += qdot;
    wdot[4] -= qdot;
    wdot[9] += qdot;
    wdot[10] -= qdot;

    qdot = q_f[66]-q_r[66];
    wdot[3] -= qdot;
    wdot[4] += qdot;
    wdot[9] -= qdot;
    wdot[10] += qdot;

    qdot = q_f[67]-q_r[67];
    wdot[6] -= qdot;
    wdot[7] += qdot;
    wdot[10] += qdot;

    qdot = q_f[68]-q_r[68];
    wdot[1] -= qdot;
    wdot[3] += qdot;
    wdot[9] += qdot;

    qdot = q_f[69]-q_r[69];
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[10] += qdot;

    qdot = q_f[70]-q_r[70];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[10] += qdot;

    qdot = q_f[71]-q_r[71];
    wdot[1] -= qdot;
    wdot[4] += qdot;
    wdot[10] += qdot;

    qdot = q_f[72]-q_r[72];
    wdot[10] += qdot;
    wdot[12] -= qdot;
    wdot[13] += qdot;

    qdot = q_f[73]-q_r[73];
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[11] -= qdot;

    qdot = q_f[74]-q_r[74];
    wdot[1] -= qdot;
    wdot[4] += qdot;
    wdot[11] -= qdot;

    qdot = q_f[75]-q_r[75];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[11] -= qdot;

    qdot = q_f[76]-q_r[76];
    wdot[11] -= qdot;
    wdot[12] -= qdot;
    wdot[13] += qdot;

    qdot = q_f[77]-q_r[77];
    wdot[11] += qdot;
    wdot[14] += qdot;

    qdot = q_f[78]-q_r[78];
    wdot[6] -= qdot;
    wdot[7] += qdot;
    wdot[11] += qdot;

    qdot = q_f[79]-q_r[79];
    wdot[2] -= qdot;
    wdot[3] += qdot;
    wdot[14] += qdot;

    qdot = q_f[80]-q_r[80];
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[14] -= qdot;

    qdot = q_f[81]-q_r[81];
    wdot[9] -= qdot;
    wdot[10] += qdot;
    wdot[11] += qdot;

    qdot = q_f[82]-q_r[82];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[4] += qdot;
    wdot[10] += qdot;
    wdot[17] -= qdot;

    qdot = q_f[83]-q_r[83];
    wdot[4] -= qdot;
    wdot[4] += qdot;
    wdot[5] += qdot;
    wdot[10] += qdot;
    wdot[17] -= qdot;

    qdot = q_f[84]-q_r[84];
    wdot[4] += qdot;
    wdot[17] -= qdot;

    qdot = q_f[85]-q_r[85];
    wdot[4] -= qdot;
    wdot[17] += qdot;

    qdot = q_f[86]-q_r[86];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[3] += qdot;
    wdot[9] += qdot;
    wdot[17] -= qdot;

    qdot = q_f[87]-q_r[87];
    wdot[3] += qdot;
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[9] += qdot;
    wdot[17] -= qdot;

    qdot = q_f[88]-q_r[88];
    wdot[12] -= qdot;

    qdot = q_f[89]-q_r[89];
    wdot[6] += qdot;
    wdot[7] -= qdot;

    qdot = q_f[90]-q_r[90];
    wdot[4] += qdot;

    qdot = q_f[91]-q_r[91];
    wdot[1] -= qdot;
    wdot[10] += qdot;
    wdot[18] -= qdot;

    qdot = q_f[92]-q_r[92];
    wdot[1] -= qdot;
    wdot[3] += qdot;
    wdot[18] -= qdot;

    qdot = q_f[93]-q_r[93];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[18] += qdot;

    qdot = q_f[94]-q_r[94];
    wdot[1] += qdot;
    wdot[6] -= qdot;

    qdot = q_f[95]-q_r[95];
    wdot[12] -= qdot;
    wdot[19] += qdot;

    qdot = q_f[96]-q_r[96];
    wdot[6] -= qdot;
    wdot[7] += qdot;
    wdot[18] += qdot;

    qdot = q_f[97]-q_r[97];
    wdot[6] -= qdot;
    wdot[11] += qdot;

    qdot = q_f[98]-q_r[98];
    wdot[12] -= qdot;
    wdot[13] += qdot;
    wdot[20] -= qdot;

    qdot = q_f[99]-q_r[99];
    wdot[1] -= qdot;
    wdot[12] += qdot;
    wdot[20] -= qdot;

    qdot = q_f[100]-q_r[100];
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[20] -= qdot;

    qdot = q_f[101]-q_r[101];
    wdot[1] -= qdot;
    wdot[3] += qdot;
    wdot[20] -= qdot;

    qdot = q_f[102]-q_r[102];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[20] -= qdot;

    qdot = q_f[103]-q_r[103];
    wdot[2] -= qdot;
    wdot[3] += qdot;
    wdot[20] += qdot;

    qdot = q_f[104]-q_r[104];
    wdot[3] -= qdot;
    wdot[15] += qdot;

    qdot = q_f[105]-q_r[105];
    wdot[4] += qdot;
    wdot[7] -= qdot;

    qdot = q_f[106]-q_r[106];
    wdot[6] -= qdot;
    wdot[7] += qdot;
    wdot[20] += qdot;

    qdot = q_f[107]-q_r[107];
    wdot[1] -= qdot;
    wdot[4] += qdot;
    wdot[15] -= qdot;

    qdot = q_f[108]-q_r[108];
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[15] -= qdot;

    qdot = q_f[109]-q_r[109];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[15] -= qdot;

    qdot = q_f[110]-q_r[110];
    wdot[1] -= qdot;
    wdot[3] += qdot;
    wdot[10] += 2.000000 * qdot;

    qdot = q_f[111]-q_r[111];
    wdot[4] -= qdot;

    qdot = q_f[112]-q_r[112];
    wdot[6] -= qdot;
    wdot[9] += qdot;

    qdot = q_f[113]-q_r[113];
    wdot[3] -= qdot;
    wdot[10] += qdot;

    qdot = q_f[114]-q_r[114];
    wdot[3] += qdot;
    wdot[10] -= qdot;

    qdot = q_f[115]-q_r[115];
    wdot[1] -= qdot;
    wdot[4] += qdot;
    wdot[16] -= qdot;

    qdot = q_f[116]-q_r[116];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[16] -= qdot;

    qdot = q_f[117]-q_r[117];
    wdot[2] -= qdot;
    wdot[3] += qdot;
    wdot[16] += qdot;

    qdot = q_f[118]-q_r[118];
    wdot[3] -= qdot;
    wdot[10] += qdot;
    wdot[12] += qdot;
    wdot[16] -= qdot;

    qdot = q_f[119]-q_r[119];
    wdot[1] -= qdot;
    wdot[9] += qdot;
    wdot[16] -= qdot;

    qdot = q_f[120]-q_r[120];
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[16] -= qdot;

    qdot = q_f[121]-q_r[121];
    wdot[4] += qdot;
    wdot[6] -= qdot;
    wdot[10] += qdot;
    wdot[11] += qdot;

    qdot = q_f[122]-q_r[122];
    wdot[3] += qdot;
    wdot[16] += qdot;

    qdot = q_f[123]-q_r[123];
    wdot[3] -= qdot;
    wdot[16] -= qdot;

    qdot = q_f[124]-q_r[124];
    wdot[6] += qdot;

    qdot = q_f[125]-q_r[125];
    wdot[6] -= qdot;

    qdot = q_f[126]-q_r[126];
    wdot[7] += qdot;
    wdot[20] += qdot;

    qdot = q_f[127]-q_r[127];
    wdot[3] += qdot;
    wdot[6] -= qdot;
    wdot[10] += qdot;

    qdot = q_f[128]-q_r[128];
    wdot[4] -= qdot;
    wdot[18] += qdot;

    qdot = q_f[129]-q_r[129];
    wdot[6] -= qdot;
    wdot[16] += qdot;

    qdot = q_f[130]-q_r[130];
    wdot[6] += qdot;
    wdot[7] -= qdot;
    wdot[21] += qdot;

    qdot = q_f[131]-q_r[131];
    wdot[2] += qdot;
    wdot[3] -= qdot;

    qdot = q_f[132]-q_r[132];
    wdot[4] -= qdot;
    wdot[5] += qdot;

    qdot = q_f[133]-q_r[133];
    wdot[4] += qdot;
    wdot[5] -= qdot;

    qdot = q_f[134]-q_r[134];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[21] -= qdot;

    qdot = q_f[135]-q_r[135];
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[21] -= qdot;

    qdot = q_f[136]-q_r[136];
    wdot[1] -= qdot;
    wdot[10] += qdot;
    wdot[20] += qdot;
    wdot[21] -= qdot;

    qdot = q_f[137]-q_r[137];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[21] += qdot;

    qdot = q_f[138]-q_r[138];
    wdot[6] += qdot;
    wdot[7] -= qdot;
    wdot[19] += qdot;

    qdot = q_f[139]-q_r[139];
    wdot[3] -= qdot;
    wdot[19] += qdot;

    qdot = q_f[140]-q_r[140];
    wdot[12] += qdot;
    wdot[18] += qdot;

    qdot = q_f[141]-q_r[141];
    wdot[3] += qdot;
    wdot[21] += qdot;

    qdot = q_f[142]-q_r[142];
    wdot[3] -= qdot;
    wdot[21] -= qdot;

    qdot = q_f[143]-q_r[143];
    wdot[11] -= qdot;
    wdot[19] += qdot;

    qdot = q_f[144]-q_r[144];
    wdot[3] -= qdot;
    wdot[12] += qdot;
    wdot[19] -= qdot;
    wdot[20] += qdot;

    qdot = q_f[145]-q_r[145];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[19] -= qdot;

    qdot = q_f[146]-q_r[146];
    wdot[1] -= qdot;
    wdot[19] -= qdot;

    qdot = q_f[147]-q_r[147];
    wdot[1] -= qdot;
    wdot[4] += qdot;
    wdot[19] -= qdot;

    qdot = q_f[148]-q_r[148];
    wdot[1] -= qdot;
    wdot[3] += qdot;
    wdot[12] += qdot;
    wdot[16] += qdot;
    wdot[19] -= qdot;

    qdot = q_f[149]-q_r[149];
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[19] -= qdot;

    qdot = q_f[150]-q_r[150];
    wdot[6] -= qdot;
    wdot[7] += qdot;
    wdot[19] += qdot;

    qdot = q_f[151]-q_r[151];
    wdot[12] += qdot;
    wdot[20] += qdot;

    qdot = q_f[152]-q_r[152];
    wdot[12] -= qdot;
    wdot[20] -= qdot;

    qdot = q_f[153]-q_r[153];
    wdot[3] += qdot;
    wdot[19] += qdot;

    qdot = q_f[154]-q_r[154];
    wdot[3] -= qdot;
    wdot[19] -= qdot;

    qdot = q_f[155]-q_r[155];
    wdot[6] += qdot;

    qdot = q_f[156]-q_r[156];
    wdot[6] -= qdot;

    qdot = q_f[157]-q_r[157];
    wdot[22] -= qdot;

    qdot = q_f[158]-q_r[158];
    wdot[22] += qdot;

    qdot = q_f[159]-q_r[159];
    wdot[4] -= qdot;
    wdot[11] += qdot;
    wdot[22] -= qdot;

    qdot = q_f[160]-q_r[160];
    wdot[4] -= qdot;
    wdot[16] += qdot;
    wdot[22] -= qdot;

    qdot = q_f[161]-q_r[161];
    wdot[1] -= qdot;
    wdot[16] += qdot;
    wdot[20] += qdot;
    wdot[22] -= qdot;

    qdot = q_f[162]-q_r[162];
    wdot[3] -= qdot;
    wdot[20] += qdot;
    wdot[22] -= qdot;

    qdot = q_f[163]-q_r[163];
    wdot[1] -= qdot;
    wdot[11] += qdot;
    wdot[21] += qdot;
    wdot[22] -= qdot;

    qdot = q_f[164]-q_r[164];
    wdot[3] -= qdot;
    wdot[23] += qdot;

    qdot = q_f[165]-q_r[165];
    wdot[3] += qdot;
    wdot[22] += qdot;

    qdot = q_f[166]-q_r[166];
    wdot[3] -= qdot;
    wdot[22] -= qdot;

    qdot = q_f[167]-q_r[167];
    wdot[12] -= qdot;
    wdot[13] += qdot;
    wdot[22] += qdot;

    qdot = q_f[168]-q_r[168];
    wdot[6] += qdot;
    wdot[7] -= qdot;
    wdot[23] += qdot;

    qdot = q_f[169]-q_r[169];
    wdot[6] -= qdot;
    wdot[7] += qdot;
    wdot[22] += qdot;

    qdot = q_f[170]-q_r[170];
    wdot[20] += qdot;

    qdot = q_f[171]-q_r[171];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[22] += qdot;

    qdot = q_f[172]-q_r[172];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[23] -= qdot;

    qdot = q_f[173]-q_r[173];
    wdot[4] -= qdot;
    wdot[11] += qdot;
    wdot[23] -= qdot;

    qdot = q_f[174]-q_r[174];
    wdot[4] -= qdot;
    wdot[15] += qdot;
    wdot[23] -= qdot;

    qdot = q_f[175]-q_r[175];
    wdot[1] -= qdot;
    wdot[23] -= qdot;

    qdot = q_f[176]-q_r[176];
    wdot[1] -= qdot;
    wdot[11] += qdot;
    wdot[19] += qdot;
    wdot[23] -= qdot;

    qdot = q_f[177]-q_r[177];
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[23] -= qdot;

    qdot = q_f[178]-q_r[178];
    wdot[12] += qdot;
    wdot[23] -= qdot;

    qdot = q_f[179]-q_r[179];
    wdot[12] -= qdot;
    wdot[23] += qdot;

    qdot = q_f[180]-q_r[180];
    wdot[3] += qdot;
    wdot[23] += qdot;

    qdot = q_f[181]-q_r[181];
    wdot[3] -= qdot;
    wdot[23] -= qdot;

    qdot = q_f[182]-q_r[182];
    wdot[20] += qdot;

    qdot = q_f[183]-q_r[183];
    wdot[6] += qdot;

    qdot = q_f[184]-q_r[184];
    wdot[6] -= qdot;

    qdot = q_f[185]-q_r[185];
    wdot[12] += qdot;
    wdot[22] += qdot;

    qdot = q_f[186]-q_r[186];
    wdot[20] += qdot;

    qdot = q_f[187]-q_r[187];
    wdot[4] -= qdot;
    wdot[5] += qdot;

    qdot = q_f[188]-q_r[188];
    wdot[2] += qdot;
    wdot[3] -= qdot;

    qdot = q_f[189]-q_r[189];

    qdot = q_f[190]-q_r[190];
    wdot[1] -= qdot;
    wdot[4] += qdot;

    qdot = q_f[191]-q_r[191];
    wdot[19] += qdot;

    qdot = q_f[192]-q_r[192];
    wdot[20] += qdot;

    qdot = q_f[193]-q_r[193];
    wdot[3] += qdot;

    qdot = q_f[194]-q_r[194];

    qdot = q_f[195]-q_r[195];
    wdot[4] -= qdot;
    wdot[11] += qdot;

    qdot = q_f[196]-q_r[196];
    wdot[12] += qdot;

    qdot = q_f[197]-q_r[197];
    wdot[19] += qdot;

    qdot = q_f[198]-q_r[198];
    wdot[23] += qdot;

    qdot = q_f[199]-q_r[199];
    wdot[20] += qdot;

    qdot = q_f[200]-q_r[200];

    qdot = q_f[201]-q_r[201];
    wdot[6] += qdot;
    wdot[7] -= qdot;
    wdot[24] += qdot;

    qdot = q_f[202]-q_r[202];
    wdot[24] -= qdot;

    qdot = q_f[203]-q_r[203];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[24] -= qdot;

    qdot = q_f[204]-q_r[204];
    wdot[24] -= qdot;

    qdot = q_f[205]-q_r[205];
    wdot[7] -= qdot;
    wdot[8] += qdot;
    wdot[24] -= qdot;

    qdot = q_f[206]-q_r[206];
    wdot[24] -= qdot;

    qdot = q_f[207]-q_r[207];
    wdot[14] += qdot;
    wdot[24] -= qdot;

    qdot = q_f[208]-q_r[208];
    wdot[1] -= qdot;
    wdot[4] += qdot;
    wdot[24] -= qdot;

    qdot = q_f[209]-q_r[209];
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[24] -= qdot;

    qdot = q_f[210]-q_r[210];
    wdot[12] -= qdot;
    wdot[13] += qdot;
    wdot[24] -= qdot;

    return;
}

void comp_k_f(double *  tc, double invT, double *  k_f)
{
#ifdef __INTEL_COMPILER
    #pragma simd
#endif
    for (int i=0; i<211; ++i) {
        k_f[i] = prefactor_units[i] * fwd_A[i]
                    * exp(fwd_beta[i] * tc[0] - activation_units[i] * fwd_Ea[i] * invT);
    };
    return;
}

void comp_Kc(double *  tc, double invT, double *  Kc)
{
    /*compute the Gibbs free energy */
    double g_RT[25], g_RT_qss[27];
    gibbs(g_RT, tc);
    gibbs_qss(g_RT_qss, tc);

    Kc[0] = g_RT[3] + g_RT[6] - g_RT[7];
    Kc[1] = -2.000000*g_RT[4] + g_RT[8];
    Kc[2] = g_RT[4] + g_RT[12] - g_RT[14];
    Kc[3] = g_RT[3] + g_RT[12] - g_RT[13];
    Kc[4] = 2.000000*g_RT[12] - g_RT[15];
    Kc[5] = g_RT[10] - g_RT[16] + g_RT_qss[2];
    Kc[6] = g_RT[1] - g_RT[9] + g_RT[10];
    Kc[7] = -g_RT[3] - g_RT[11] + g_RT_qss[4];
    Kc[8] = -g_RT[3] - g_RT[18] + g_RT_qss[9];
    Kc[9] = g_RT[3] + g_RT[20] - g_RT_qss[5];
    Kc[10] = -g_RT[10] - g_RT[12] + g_RT_qss[12];
    Kc[11] = g_RT[3] + g_RT[4] - g_RT[5];
    Kc[12] = -g_RT_qss[2] + g_RT_qss[3];
    Kc[13] = g_RT_qss[2] - g_RT_qss[3];
    Kc[14] = -g_RT[3] - g_RT[10] + g_RT_qss[1];
    Kc[15] = -g_RT[6] - g_RT[12] + g_RT_qss[6];
    Kc[16] = g_RT[6] + g_RT[12] - g_RT_qss[6];
    Kc[17] = -g_RT[11] - g_RT[12] + g_RT_qss[11];
    Kc[18] = g_RT[1] + g_RT[2] - g_RT[3] - g_RT[4];
    Kc[19] = -g_RT[1] - g_RT[2] + g_RT[3] + g_RT[4];
    Kc[20] = g_RT[2] - g_RT[3] + g_RT[4] - g_RT[5];
    Kc[21] = -g_RT[2] + g_RT[3] - g_RT[4] + g_RT[5];
    Kc[22] = g_RT[1] - 2.000000*g_RT[4] + g_RT[5];
    Kc[23] = -g_RT[1] + 2.000000*g_RT[4] - g_RT[5];
    Kc[24] = -g_RT[1] + g_RT[3] - g_RT[4] + g_RT[6];
    Kc[25] = g_RT[1] - g_RT[3] + g_RT[4] - g_RT[6];
    Kc[26] = g_RT[4] - g_RT[5] - g_RT[6] + g_RT[7];
    Kc[27] = g_RT[1] - g_RT[4] - g_RT[6] + g_RT[7];
    Kc[28] = -g_RT[6] + 2.000000*g_RT[7] - g_RT[8];
    Kc[29] = -g_RT[2] + g_RT[3] - g_RT[6] + g_RT[7];
    Kc[30] = -g_RT[6] + 2.000000*g_RT[7] - g_RT[8];
    Kc[31] = g_RT[3] - 2.000000*g_RT[4] + g_RT[7];
    Kc[32] = g_RT[4] - g_RT[5] - g_RT[7] + g_RT[8];
    Kc[33] = -g_RT[1] + g_RT[6] + g_RT_qss[0] - g_RT_qss[1];
    Kc[34] = -2.000000*g_RT[3] + g_RT[6] - g_RT[9] + g_RT_qss[2];
    Kc[35] = -g_RT[5] + g_RT[6] - g_RT[10] + g_RT_qss[2];
    Kc[36] = g_RT[1] - 2.000000*g_RT[3] - g_RT[10] + g_RT_qss[2];
    Kc[37] = -g_RT[1] + g_RT[6] - g_RT[11] + g_RT_qss[2];
    Kc[38] = -g_RT[2] + g_RT[3] - g_RT_qss[0] + g_RT_qss[2];
    Kc[39] = g_RT[2] - g_RT[3] + g_RT_qss[0] - g_RT_qss[2];
    Kc[40] = g_RT[4] - g_RT[5] - g_RT_qss[0] + g_RT_qss[2];
    Kc[41] = -g_RT[4] + g_RT[5] + g_RT_qss[0] - g_RT_qss[2];
    Kc[42] = -g_RT[2] + g_RT[6] - g_RT[9] + g_RT_qss[2];
    Kc[43] = -g_RT[2] + g_RT[3] - g_RT_qss[0] + g_RT_qss[3];
    Kc[44] = g_RT[2] - g_RT[3] - g_RT[12] + g_RT_qss[3];
    Kc[45] = -g_RT[2] + g_RT[3] + g_RT[12] - g_RT_qss[3];
    Kc[46] = -g_RT[3] - g_RT[4] + g_RT[6] - g_RT[10] + g_RT_qss[3];
    Kc[47] = -g_RT[3] + g_RT[4] - g_RT[11] + g_RT_qss[3];
    Kc[48] = -g_RT[2] + g_RT[4] - g_RT[11] + g_RT[12];
    Kc[49] = g_RT[4] - g_RT[5] + g_RT[12] - g_RT_qss[3];
    Kc[50] = -g_RT[4] + g_RT[5] - g_RT[12] + g_RT_qss[3];
    Kc[51] = g_RT[1] - g_RT[3] - g_RT[11] + g_RT[12];
    Kc[52] = -g_RT[4] + g_RT[7] + g_RT[12] - g_RT_qss[4];
    Kc[53] = -g_RT[6] + g_RT[7] + g_RT[12] - g_RT[13];
    Kc[54] = -g_RT[4] + g_RT[6] - g_RT[11] + g_RT[12];
    Kc[55] = -g_RT[2] + g_RT[3] + g_RT[12] - g_RT_qss[2];
    Kc[56] = g_RT[2] - g_RT[3] - g_RT[12] + g_RT_qss[2];
    Kc[57] = -g_RT[3] + 2.000000*g_RT[12] - g_RT_qss[5];
    Kc[58] = g_RT[4] - g_RT[5] + g_RT[12] - g_RT_qss[2];
    Kc[59] = -g_RT[4] + g_RT[5] - g_RT[12] + g_RT_qss[2];
    Kc[60] = g_RT[1] - g_RT[4] - g_RT[12] + g_RT[13];
    Kc[61] = -g_RT[2] + g_RT[3] - g_RT[12] + g_RT[13];
    Kc[62] = g_RT[2] - g_RT[3] + g_RT[12] - g_RT[13];
    Kc[63] = g_RT[4] - g_RT[5] - g_RT[12] + g_RT[13];
    Kc[64] = -g_RT[4] + g_RT[5] + g_RT[12] - g_RT[13];
    Kc[65] = -g_RT[3] + g_RT[4] - g_RT[9] + g_RT[10];
    Kc[66] = g_RT[3] - g_RT[4] + g_RT[9] - g_RT[10];
    Kc[67] = g_RT[6] - g_RT[7] - g_RT[10] + g_RT_qss[1];
    Kc[68] = g_RT[1] - g_RT[3] - g_RT[9] + g_RT_qss[1];
    Kc[69] = g_RT[4] - g_RT[5] - g_RT[10] + g_RT_qss[1];
    Kc[70] = -g_RT[2] + g_RT[3] - g_RT[10] + g_RT_qss[1];
    Kc[71] = g_RT[1] - g_RT[4] - g_RT[10] + g_RT_qss[1];
    Kc[72] = -g_RT[10] + g_RT[12] - g_RT[13] + g_RT_qss[1];
    Kc[73] = g_RT[4] - g_RT[5] + g_RT[11] - g_RT_qss[1];
    Kc[74] = g_RT[1] - g_RT[4] + g_RT[11] - g_RT_qss[1];
    Kc[75] = -g_RT[2] + g_RT[3] + g_RT[11] - g_RT_qss[1];
    Kc[76] = g_RT[11] + g_RT[12] - g_RT[13] - g_RT_qss[1];
    Kc[77] = -g_RT[11] - g_RT[14] + 2.000000*g_RT_qss[4];
    Kc[78] = g_RT[6] - g_RT[7] - g_RT[11] + g_RT_qss[4];
    Kc[79] = g_RT[2] - g_RT[3] - g_RT[14] + g_RT_qss[4];
    Kc[80] = g_RT[4] - g_RT[5] + g_RT[14] - g_RT_qss[4];
    Kc[81] = g_RT[9] - g_RT[10] - g_RT[11] + g_RT_qss[3];
    Kc[82] = -g_RT[2] + g_RT[3] - g_RT[4] - g_RT[10] + g_RT[17];
    Kc[83] = g_RT[4] - g_RT[4] - g_RT[5] - g_RT[10] + g_RT[17];
    Kc[84] = -g_RT[4] + g_RT[17] - g_RT_qss[1];
    Kc[85] = g_RT[4] - g_RT[17] + g_RT_qss[1];
    Kc[86] = -g_RT[2] + g_RT[3] - g_RT[3] - g_RT[9] + g_RT[17];
    Kc[87] = -g_RT[3] + g_RT[4] - g_RT[5] - g_RT[9] + g_RT[17];
    Kc[88] = g_RT[12] - 2.000000*g_RT_qss[4] + g_RT_qss[6];
    Kc[89] = -g_RT[6] + g_RT[7] + g_RT_qss[6] - g_RT_qss[7];
    Kc[90] = -g_RT[4] - g_RT_qss[4] + g_RT_qss[7];
    Kc[91] = g_RT[1] - g_RT[10] + g_RT[18] - g_RT_qss[2];
    Kc[92] = g_RT[1] - g_RT[3] + g_RT[18] - g_RT_qss[8];
    Kc[93] = -g_RT[2] + g_RT[3] - g_RT[18] + g_RT_qss[9];
    Kc[94] = -g_RT[1] + g_RT[6] + g_RT_qss[9] - g_RT_qss[10];
    Kc[95] = g_RT[12] - g_RT[19] + g_RT_qss[9];
    Kc[96] = g_RT[6] - g_RT[7] - g_RT[18] + g_RT_qss[9];
    Kc[97] = g_RT[6] - g_RT[11] - g_RT_qss[1] + g_RT_qss[9];
    Kc[98] = g_RT[12] - g_RT[13] + g_RT[20] - g_RT_qss[9];
    Kc[99] = g_RT[1] - g_RT[12] + g_RT[20] - g_RT_qss[1];
    Kc[100] = g_RT[4] - g_RT[5] + g_RT[20] - g_RT_qss[9];
    Kc[101] = g_RT[1] - g_RT[3] + g_RT[20] - g_RT_qss[10];
    Kc[102] = -g_RT[2] + g_RT[3] + g_RT[20] - g_RT_qss[9];
    Kc[103] = g_RT[2] - g_RT[3] - g_RT[20] + g_RT_qss[9];
    Kc[104] = g_RT[3] - g_RT[15] + g_RT_qss[5];
    Kc[105] = -g_RT[4] + g_RT[7] + g_RT_qss[5] - g_RT_qss[11];
    Kc[106] = g_RT[6] - g_RT[7] - g_RT[20] + g_RT_qss[5];
    Kc[107] = g_RT[1] - g_RT[4] + g_RT[15] - g_RT_qss[5];
    Kc[108] = g_RT[4] - g_RT[5] + g_RT[15] - g_RT_qss[5];
    Kc[109] = -g_RT[2] + g_RT[3] + g_RT[15] - g_RT_qss[5];
    Kc[110] = g_RT[1] - g_RT[3] - 2.000000*g_RT[10] + g_RT_qss[8];
    Kc[111] = g_RT[4] - 2.000000*g_RT_qss[1] + g_RT_qss[8];
    Kc[112] = g_RT[6] - g_RT[9] - g_RT_qss[1] + g_RT_qss[8];
    Kc[113] = g_RT[3] - g_RT[10] - g_RT_qss[3] + g_RT_qss[8];
    Kc[114] = -g_RT[3] + g_RT[10] + g_RT_qss[3] - g_RT_qss[8];
    Kc[115] = g_RT[1] - g_RT[4] + g_RT[16] - g_RT_qss[8];
    Kc[116] = -g_RT[2] + g_RT[3] + g_RT[16] - g_RT_qss[8];
    Kc[117] = g_RT[2] - g_RT[3] - g_RT[16] + g_RT_qss[8];
    Kc[118] = g_RT[3] - g_RT[10] - g_RT[12] + g_RT[16];
    Kc[119] = g_RT[1] - g_RT[9] + g_RT[16] - g_RT_qss[2];
    Kc[120] = g_RT[4] - g_RT[5] + g_RT[16] - g_RT_qss[8];
    Kc[121] = -g_RT[4] + g_RT[6] - g_RT[10] - g_RT[11] + g_RT_qss[10];
    Kc[122] = -g_RT[3] - g_RT[16] + g_RT_qss[10];
    Kc[123] = g_RT[3] + g_RT[16] - g_RT_qss[10];
    Kc[124] = -g_RT[6] - g_RT_qss[5] + g_RT_qss[13];
    Kc[125] = g_RT[6] + g_RT_qss[5] - g_RT_qss[13];
    Kc[126] = -g_RT[7] - g_RT[20] + g_RT_qss[13];
    Kc[127] = -g_RT[3] + g_RT[6] - g_RT[10] - g_RT_qss[8] + g_RT_qss[14];
    Kc[128] = g_RT[4] - g_RT[18] - g_RT_qss[1] + g_RT_qss[14];
    Kc[129] = g_RT[6] - g_RT[16] - g_RT_qss[1] + g_RT_qss[15];
    Kc[130] = -g_RT[6] + g_RT[7] - g_RT[21] + g_RT_qss[15];
    Kc[131] = -g_RT[2] + g_RT[3] - g_RT_qss[14] + g_RT_qss[15];
    Kc[132] = g_RT[4] - g_RT[5] - g_RT_qss[14] + g_RT_qss[15];
    Kc[133] = -g_RT[4] + g_RT[5] + g_RT_qss[14] - g_RT_qss[15];
    Kc[134] = -g_RT[2] + g_RT[3] + g_RT[21] - g_RT_qss[15];
    Kc[135] = g_RT[4] - g_RT[5] + g_RT[21] - g_RT_qss[15];
    Kc[136] = g_RT[1] - g_RT[10] - g_RT[20] + g_RT[21];
    Kc[137] = -g_RT[2] + g_RT[3] - g_RT[21] + g_RT_qss[16];
    Kc[138] = -g_RT[6] + g_RT[7] - g_RT[19] + g_RT_qss[16];
    Kc[139] = g_RT[3] - g_RT[19] + g_RT_qss[16];
    Kc[140] = -g_RT[12] - g_RT[18] + g_RT_qss[16];
    Kc[141] = -g_RT[3] - g_RT[21] + g_RT_qss[16];
    Kc[142] = g_RT[3] + g_RT[21] - g_RT_qss[16];
    Kc[143] = g_RT[11] - g_RT[19] - g_RT_qss[1] + g_RT_qss[16];
    Kc[144] = g_RT[3] - g_RT[12] + g_RT[19] - g_RT[20];
    Kc[145] = -g_RT[2] + g_RT[3] + g_RT[19] - g_RT_qss[16];
    Kc[146] = g_RT[1] + g_RT[19] - g_RT_qss[1] - g_RT_qss[5];
    Kc[147] = g_RT[1] - g_RT[4] + g_RT[19] - g_RT_qss[16];
    Kc[148] = g_RT[1] - g_RT[3] - g_RT[12] - g_RT[16] + g_RT[19];
    Kc[149] = g_RT[4] - g_RT[5] + g_RT[19] - g_RT_qss[16];
    Kc[150] = g_RT[6] - g_RT[7] - g_RT[19] + g_RT_qss[17];
    Kc[151] = -g_RT[12] - g_RT[20] + g_RT_qss[17];
    Kc[152] = g_RT[12] + g_RT[20] - g_RT_qss[17];
    Kc[153] = -g_RT[3] - g_RT[19] + g_RT_qss[17];
    Kc[154] = g_RT[3] + g_RT[19] - g_RT_qss[17];
    Kc[155] = -g_RT[6] - g_RT_qss[17] + g_RT_qss[18];
    Kc[156] = g_RT[6] + g_RT_qss[17] - g_RT_qss[18];
    Kc[157] = g_RT[22] - 2.000000*g_RT_qss[9];
    Kc[158] = -g_RT[22] + 2.000000*g_RT_qss[9];
    Kc[159] = g_RT[4] - g_RT[11] + g_RT[22] - g_RT_qss[16];
    Kc[160] = g_RT[4] - g_RT[16] + g_RT[22] - g_RT_qss[5];
    Kc[161] = g_RT[1] - g_RT[16] - g_RT[20] + g_RT[22];
    Kc[162] = g_RT[3] - g_RT[20] + g_RT[22] - g_RT_qss[9];
    Kc[163] = g_RT[1] - g_RT[11] - g_RT[21] + g_RT[22];
    Kc[164] = g_RT[3] - g_RT[23] + g_RT_qss[19];
    Kc[165] = -g_RT[3] - g_RT[22] + g_RT_qss[19];
    Kc[166] = g_RT[3] + g_RT[22] - g_RT_qss[19];
    Kc[167] = g_RT[12] - g_RT[13] - g_RT[22] + g_RT_qss[19];
    Kc[168] = -g_RT[6] + g_RT[7] - g_RT[23] + g_RT_qss[19];
    Kc[169] = g_RT[6] - g_RT[7] - g_RT[22] + g_RT_qss[19];
    Kc[170] = -g_RT[20] - g_RT_qss[9] + g_RT_qss[19];
    Kc[171] = -g_RT[2] + g_RT[3] - g_RT[22] + g_RT_qss[19];
    Kc[172] = -g_RT[2] + g_RT[3] + g_RT[23] - g_RT_qss[19];
    Kc[173] = g_RT[4] - g_RT[11] + g_RT[23] - g_RT_qss[17];
    Kc[174] = g_RT[4] - g_RT[15] + g_RT[23] - g_RT_qss[12];
    Kc[175] = g_RT[1] + g_RT[23] - g_RT_qss[5] - g_RT_qss[12];
    Kc[176] = g_RT[1] - g_RT[11] - g_RT[19] + g_RT[23];
    Kc[177] = g_RT[4] - g_RT[5] + g_RT[23] - g_RT_qss[19];
    Kc[178] = -g_RT[12] + g_RT[23] - g_RT_qss[16];
    Kc[179] = g_RT[12] - g_RT[23] + g_RT_qss[16];
    Kc[180] = -g_RT[3] - g_RT[23] + g_RT_qss[20];
    Kc[181] = g_RT[3] + g_RT[23] - g_RT_qss[20];
    Kc[182] = -g_RT[20] - g_RT_qss[5] + g_RT_qss[20];
    Kc[183] = -g_RT[6] - g_RT_qss[20] + g_RT_qss[21];
    Kc[184] = g_RT[6] + g_RT_qss[20] - g_RT_qss[21];
    Kc[185] = -g_RT[12] - g_RT[22] + g_RT_qss[22];
    Kc[186] = -g_RT[20] - g_RT_qss[16] + g_RT_qss[22];
    Kc[187] = g_RT[4] - g_RT[5] - g_RT_qss[22] + g_RT_qss[23];
    Kc[188] = -g_RT[2] + g_RT[3] - g_RT_qss[22] + g_RT_qss[23];
    Kc[189] = -g_RT_qss[5] - g_RT_qss[16] + g_RT_qss[23];
    Kc[190] = g_RT[1] - g_RT[4] - g_RT_qss[22] + g_RT_qss[23];
    Kc[191] = -g_RT[19] - g_RT_qss[5] + g_RT_qss[24];
    Kc[192] = -g_RT[20] - g_RT_qss[17] + g_RT_qss[24];
    Kc[193] = -g_RT[3] - g_RT_qss[23] + g_RT_qss[24];
    Kc[194] = -g_RT_qss[16] - g_RT_qss[17] + g_RT_qss[25];
    Kc[195] = g_RT[4] - g_RT[11] - g_RT_qss[24] + g_RT_qss[25];
    Kc[196] = -g_RT[12] - g_RT_qss[25] + g_RT_qss[26];
    Kc[197] = -g_RT[19] - g_RT_qss[20] + g_RT_qss[26];
    Kc[198] = -g_RT[23] - g_RT_qss[17] + g_RT_qss[26];
    Kc[199] = -g_RT[20] - g_RT_qss[24] + g_RT_qss[26];
    Kc[200] = -g_RT_qss[5] - g_RT_qss[23] + g_RT_qss[26];
    Kc[201] = -g_RT[6] + g_RT[7] - g_RT[24] + g_RT_qss[26];
    Kc[202] = g_RT[24] + g_RT_qss[6] - g_RT_qss[7] - g_RT_qss[26];
    Kc[203] = -g_RT[2] + g_RT[3] + g_RT[24] - g_RT_qss[26];
    Kc[204] = g_RT[24] - g_RT_qss[17] - g_RT_qss[20];
    Kc[205] = g_RT[7] - g_RT[8] + g_RT[24] - g_RT_qss[26];
    Kc[206] = g_RT[24] - g_RT_qss[5] - g_RT_qss[24];
    Kc[207] = -g_RT[14] + g_RT[24] + g_RT_qss[4] - g_RT_qss[26];
    Kc[208] = g_RT[1] - g_RT[4] + g_RT[24] - g_RT_qss[26];
    Kc[209] = g_RT[4] - g_RT[5] + g_RT[24] - g_RT_qss[26];
    Kc[210] = g_RT[12] - g_RT[13] + g_RT[24] - g_RT_qss[26];

#ifdef __INTEL_COMPILER
     #pragma simd
#endif
    for (int i=0; i<211; ++i) {
        Kc[i] = exp(Kc[i]);
    };

    /*reference concentration: P_atm / (RT) in inverse mol/m^3 */
    double refC = 101325 / 8.31446 * invT;
    double refCinv = 1 / refC;

    Kc[0] *= refCinv;
    Kc[1] *= refC;
    Kc[2] *= refCinv;
    Kc[3] *= refCinv;
    Kc[4] *= refCinv;
    Kc[5] *= refCinv;
    Kc[6] *= refCinv;
    Kc[7] *= refC;
    Kc[8] *= refC;
    Kc[9] *= refCinv;
    Kc[10] *= refC;
    Kc[11] *= refCinv;
    Kc[14] *= refC;
    Kc[15] *= refC;
    Kc[16] *= refCinv;
    Kc[17] *= refC;
    Kc[34] *= refC;
    Kc[36] *= refC;
    Kc[46] *= refC;
    Kc[82] *= refC;
    Kc[83] *= refC;
    Kc[84] *= refC;
    Kc[85] *= refCinv;
    Kc[86] *= refC;
    Kc[87] *= refC;
    Kc[90] *= refC;
    Kc[95] *= refCinv;
    Kc[104] *= refCinv;
    Kc[110] *= refC;
    Kc[121] *= refC;
    Kc[122] *= refC;
    Kc[123] *= refCinv;
    Kc[124] *= refC;
    Kc[125] *= refCinv;
    Kc[126] *= refC;
    Kc[127] *= refC;
    Kc[139] *= refCinv;
    Kc[140] *= refC;
    Kc[141] *= refC;
    Kc[142] *= refCinv;
    Kc[148] *= refC;
    Kc[151] *= refC;
    Kc[152] *= refCinv;
    Kc[153] *= refC;
    Kc[154] *= refCinv;
    Kc[155] *= refC;
    Kc[156] *= refCinv;
    Kc[157] *= refC;
    Kc[158] *= refCinv;
    Kc[164] *= refCinv;
    Kc[165] *= refC;
    Kc[166] *= refCinv;
    Kc[170] *= refC;
    Kc[178] *= refC;
    Kc[179] *= refCinv;
    Kc[180] *= refC;
    Kc[181] *= refCinv;
    Kc[182] *= refC;
    Kc[183] *= refC;
    Kc[184] *= refCinv;
    Kc[185] *= refC;
    Kc[186] *= refC;
    Kc[189] *= refC;
    Kc[191] *= refC;
    Kc[192] *= refC;
    Kc[193] *= refC;
    Kc[194] *= refC;
    Kc[196] *= refC;
    Kc[197] *= refC;
    Kc[198] *= refC;
    Kc[199] *= refC;
    Kc[200] *= refC;
    Kc[204] *= refC;
    Kc[206] *= refC;

    return;
}

void comp_qfqr(double *  qf, double *  qr, double *  sc, double * qss_sc, double *  tc, double invT)
{

    /*reaction 1: H + O2 (+M) => HO2 (+M) */
    qf[0] = sc[3]*sc[6];
    qr[0] = 0.0;

    /*reaction 2: H2O2 (+M) => 2.000000 OH (+M) */
    qf[1] = sc[8];
    qr[1] = 0.0;

    /*reaction 3: OH + CH3 (+M) => CH3OH (+M) */
    qf[2] = sc[4]*sc[12];
    qr[2] = 0.0;

    /*reaction 4: CH3 + H (+M) => CH4 (+M) */
    qf[3] = sc[3]*sc[12];
    qr[3] = 0.0;

    /*reaction 5: 2.000000 CH3 (+M) => C2H6 (+M) */
    qf[4] = pow(sc[12], 2.000000);
    qr[4] = 0.0;

    /*reaction 6: CO + CH2 (+M) => CH2CO (+M) */
    qf[5] = qss_sc[2]*sc[10];
    qr[5] = 0.0;

    /*reaction 7: CO + O (+M) => CO2 (+M) */
    qf[6] = sc[1]*sc[10];
    qr[6] = 0.0;

    /*reaction 8: CH3O (+M) => CH2O + H (+M) */
    qf[7] = qss_sc[4];
    qr[7] = 0.0;

    /*reaction 9: C2H3 (+M) => H + C2H2 (+M) */
    qf[8] = qss_sc[9];
    qr[8] = 0.0;

    /*reaction 10: H + C2H4 (+M) <=> C2H5 (+M) */
    qf[9] = sc[3]*sc[20];
    qr[9] = qss_sc[5];

    /*reaction 11: CH3CO (+M) => CH3 + CO (+M) */
    qf[10] = qss_sc[12];
    qr[10] = 0.0;

    /*reaction 12: H + OH + M => H2O + M */
    qf[11] = sc[3]*sc[4];
    qr[11] = 0.0;

    /*reaction 13: CH2GSG + M => CH2 + M */
    qf[12] = qss_sc[3];
    qr[12] = 0.0;

    /*reaction 14: CH2 + M => CH2GSG + M */
    qf[13] = qss_sc[2];
    qr[13] = 0.0;

    /*reaction 15: HCO + M => H + CO + M */
    qf[14] = qss_sc[1];
    qr[14] = 0.0;

    /*reaction 16: CH3O2 + M => CH3 + O2 + M */
    qf[15] = qss_sc[6];
    qr[15] = 0.0;

    /*reaction 17: CH3 + O2 + M => CH3O2 + M */
    qf[16] = sc[6]*sc[12];
    qr[16] = 0.0;

    /*reaction 18: C2H5O + M => CH3 + CH2O + M */
    qf[17] = qss_sc[11];
    qr[17] = 0.0;

    /*reaction 19: O + H2 => H + OH */
    qf[18] = sc[1]*sc[2];
    qr[18] = 0.0;

    /*reaction 20: H + OH => O + H2 */
    qf[19] = sc[3]*sc[4];
    qr[19] = 0.0;

    /*reaction 21: OH + H2 => H + H2O */
    qf[20] = sc[2]*sc[4];
    qr[20] = 0.0;

    /*reaction 22: H + H2O => OH + H2 */
    qf[21] = sc[3]*sc[5];
    qr[21] = 0.0;

    /*reaction 23: O + H2O => 2.000000 OH */
    qf[22] = sc[1]*sc[5];
    qr[22] = 0.0;

    /*reaction 24: 2.000000 OH => O + H2O */
    qf[23] = pow(sc[4], 2.000000);
    qr[23] = 0.0;

    /*reaction 25: H + O2 => O + OH */
    qf[24] = sc[3]*sc[6];
    qr[24] = 0.0;

    /*reaction 26: O + OH => H + O2 */
    qf[25] = sc[1]*sc[4];
    qr[25] = 0.0;

    /*reaction 27: HO2 + OH => H2O + O2 */
    qf[26] = sc[4]*sc[7];
    qr[26] = 0.0;

    /*reaction 28: HO2 + O => OH + O2 */
    qf[27] = sc[1]*sc[7];
    qr[27] = 0.0;

    /*reaction 29: 2.000000 HO2 => H2O2 + O2 */
    qf[28] = pow(sc[7], 2.000000);
    qr[28] = 0.0;

    /*reaction 30: HO2 + H => H2 + O2 */
    qf[29] = sc[3]*sc[7];
    qr[29] = 0.0;

    /*reaction 31: 2.000000 HO2 => H2O2 + O2 */
    qf[30] = pow(sc[7], 2.000000);
    qr[30] = 0.0;

    /*reaction 32: HO2 + H => 2.000000 OH */
    qf[31] = sc[3]*sc[7];
    qr[31] = 0.0;

    /*reaction 33: H2O2 + OH => H2O + HO2 */
    qf[32] = sc[4]*sc[8];
    qr[32] = 0.0;

    /*reaction 34: CH + O2 => HCO + O */
    qf[33] = sc[6]*qss_sc[0];
    qr[33] = 0.0;

    /*reaction 35: CH2 + O2 => CO2 + 2.000000 H */
    qf[34] = sc[6]*qss_sc[2];
    qr[34] = 0.0;

    /*reaction 36: CH2 + O2 => CO + H2O */
    qf[35] = sc[6]*qss_sc[2];
    qr[35] = 0.0;

    /*reaction 37: CH2 + O => CO + 2.000000 H */
    qf[36] = sc[1]*qss_sc[2];
    qr[36] = 0.0;

    /*reaction 38: CH2 + O2 => CH2O + O */
    qf[37] = sc[6]*qss_sc[2];
    qr[37] = 0.0;

    /*reaction 39: CH2 + H => CH + H2 */
    qf[38] = sc[3]*qss_sc[2];
    qr[38] = 0.0;

    /*reaction 40: CH + H2 => CH2 + H */
    qf[39] = sc[2]*qss_sc[0];
    qr[39] = 0.0;

    /*reaction 41: CH2 + OH => CH + H2O */
    qf[40] = sc[4]*qss_sc[2];
    qr[40] = 0.0;

    /*reaction 42: CH + H2O => CH2 + OH */
    qf[41] = sc[5]*qss_sc[0];
    qr[41] = 0.0;

    /*reaction 43: CH2 + O2 => CO2 + H2 */
    qf[42] = sc[6]*qss_sc[2];
    qr[42] = 0.0;

    /*reaction 44: CH2GSG + H => CH + H2 */
    qf[43] = sc[3]*qss_sc[3];
    qr[43] = 0.0;

    /*reaction 45: CH2GSG + H2 => CH3 + H */
    qf[44] = sc[2]*qss_sc[3];
    qr[44] = 0.0;

    /*reaction 46: CH3 + H => CH2GSG + H2 */
    qf[45] = sc[3]*sc[12];
    qr[45] = 0.0;

    /*reaction 47: CH2GSG + O2 => CO + OH + H */
    qf[46] = sc[6]*qss_sc[3];
    qr[46] = 0.0;

    /*reaction 48: CH2GSG + OH => CH2O + H */
    qf[47] = sc[4]*qss_sc[3];
    qr[47] = 0.0;

    /*reaction 49: CH3 + OH => CH2O + H2 */
    qf[48] = sc[4]*sc[12];
    qr[48] = 0.0;

    /*reaction 50: CH3 + OH => CH2GSG + H2O */
    qf[49] = sc[4]*sc[12];
    qr[49] = 0.0;

    /*reaction 51: CH2GSG + H2O => CH3 + OH */
    qf[50] = sc[5]*qss_sc[3];
    qr[50] = 0.0;

    /*reaction 52: CH3 + O => CH2O + H */
    qf[51] = sc[1]*sc[12];
    qr[51] = 0.0;

    /*reaction 53: CH3 + HO2 => CH3O + OH */
    qf[52] = sc[7]*sc[12];
    qr[52] = 0.0;

    /*reaction 54: CH3 + HO2 => CH4 + O2 */
    qf[53] = sc[7]*sc[12];
    qr[53] = 0.0;

    /*reaction 55: CH3 + O2 => CH2O + OH */
    qf[54] = sc[6]*sc[12];
    qr[54] = 0.0;

    /*reaction 56: CH3 + H => CH2 + H2 */
    qf[55] = sc[3]*sc[12];
    qr[55] = 0.0;

    /*reaction 57: CH2 + H2 => CH3 + H */
    qf[56] = sc[2]*qss_sc[2];
    qr[56] = 0.0;

    /*reaction 58: 2.000000 CH3 <=> H + C2H5 */
    qf[57] = pow(sc[12], 2.000000);
    qr[57] = sc[3]*qss_sc[5];

    /*reaction 59: CH3 + OH => CH2 + H2O */
    qf[58] = sc[4]*sc[12];
    qr[58] = 0.0;

    /*reaction 60: CH2 + H2O => CH3 + OH */
    qf[59] = sc[5]*qss_sc[2];
    qr[59] = 0.0;

    /*reaction 61: CH4 + O => CH3 + OH */
    qf[60] = sc[1]*sc[13];
    qr[60] = 0.0;

    /*reaction 62: CH4 + H => CH3 + H2 */
    qf[61] = sc[3]*sc[13];
    qr[61] = 0.0;

    /*reaction 63: CH3 + H2 => CH4 + H */
    qf[62] = sc[2]*sc[12];
    qr[62] = 0.0;

    /*reaction 64: CH4 + OH => CH3 + H2O */
    qf[63] = sc[4]*sc[13];
    qr[63] = 0.0;

    /*reaction 65: CH3 + H2O => CH4 + OH */
    qf[64] = sc[5]*sc[12];
    qr[64] = 0.0;

    /*reaction 66: CO + OH => CO2 + H */
    qf[65] = sc[4]*sc[10];
    qr[65] = 0.0;

    /*reaction 67: CO2 + H => CO + OH */
    qf[66] = sc[3]*sc[9];
    qr[66] = 0.0;

    /*reaction 68: HCO + O2 => CO + HO2 */
    qf[67] = sc[6]*qss_sc[1];
    qr[67] = 0.0;

    /*reaction 69: HCO + O => CO2 + H */
    qf[68] = sc[1]*qss_sc[1];
    qr[68] = 0.0;

    /*reaction 70: HCO + OH => CO + H2O */
    qf[69] = sc[4]*qss_sc[1];
    qr[69] = 0.0;

    /*reaction 71: HCO + H => CO + H2 */
    qf[70] = sc[3]*qss_sc[1];
    qr[70] = 0.0;

    /*reaction 72: HCO + O => CO + OH */
    qf[71] = sc[1]*qss_sc[1];
    qr[71] = 0.0;

    /*reaction 73: HCO + CH3 => CH4 + CO */
    qf[72] = qss_sc[1]*sc[12];
    qr[72] = 0.0;

    /*reaction 74: CH2O + OH => HCO + H2O */
    qf[73] = sc[4]*sc[11];
    qr[73] = 0.0;

    /*reaction 75: CH2O + O => HCO + OH */
    qf[74] = sc[1]*sc[11];
    qr[74] = 0.0;

    /*reaction 76: CH2O + H => HCO + H2 */
    qf[75] = sc[3]*sc[11];
    qr[75] = 0.0;

    /*reaction 77: CH2O + CH3 => HCO + CH4 */
    qf[76] = sc[11]*sc[12];
    qr[76] = 0.0;

    /*reaction 78: 2.000000 CH3O => CH3OH + CH2O */
    qf[77] = pow(qss_sc[4], 2.000000);
    qr[77] = 0.0;

    /*reaction 79: CH3O + O2 => CH2O + HO2 */
    qf[78] = sc[6]*qss_sc[4];
    qr[78] = 0.0;

    /*reaction 80: CH3O + H2 => CH3OH + H */
    qf[79] = sc[2]*qss_sc[4];
    qr[79] = 0.0;

    /*reaction 81: CH3OH + OH => CH3O + H2O */
    qf[80] = sc[4]*sc[14];
    qr[80] = 0.0;

    /*reaction 82: CH2GSG + CO2 => CH2O + CO */
    qf[81] = sc[9]*qss_sc[3];
    qr[81] = 0.0;

    /*reaction 83: HOCHO + H => H2 + CO + OH */
    qf[82] = sc[3]*sc[17];
    qr[82] = 0.0;

    /*reaction 84: HOCHO + OH => H2O + CO + OH */
    qf[83] = sc[4]*sc[17];
    qr[83] = 0.0;

    /*reaction 85: HOCHO => HCO + OH */
    qf[84] = sc[17];
    qr[84] = 0.0;

    /*reaction 86: HCO + OH => HOCHO */
    qf[85] = sc[4]*qss_sc[1];
    qr[85] = 0.0;

    /*reaction 87: HOCHO + H => H2 + CO2 + H */
    qf[86] = sc[3]*sc[17];
    qr[86] = 0.0;

    /*reaction 88: HOCHO + OH => H2O + CO2 + H */
    qf[87] = sc[4]*sc[17];
    qr[87] = 0.0;

    /*reaction 89: CH3O2 + CH3 => 2.000000 CH3O */
    qf[88] = sc[12]*qss_sc[6];
    qr[88] = 0.0;

    /*reaction 90: CH3O2 + HO2 => CH3O2H + O2 */
    qf[89] = sc[7]*qss_sc[6];
    qr[89] = 0.0;

    /*reaction 91: CH3O2H => CH3O + OH */
    qf[90] = qss_sc[7];
    qr[90] = 0.0;

    /*reaction 92: C2H2 + O => CH2 + CO */
    qf[91] = sc[1]*sc[18];
    qr[91] = 0.0;

    /*reaction 93: C2H2 + O => HCCO + H */
    qf[92] = sc[1]*sc[18];
    qr[92] = 0.0;

    /*reaction 94: C2H3 + H => C2H2 + H2 */
    qf[93] = sc[3]*qss_sc[9];
    qr[93] = 0.0;

    /*reaction 95: C2H3 + O2 => CH2CHO + O */
    qf[94] = sc[6]*qss_sc[9];
    qr[94] = 0.0;

    /*reaction 96: C2H3 + CH3 => C3H6 */
    qf[95] = sc[12]*qss_sc[9];
    qr[95] = 0.0;

    /*reaction 97: C2H3 + O2 => C2H2 + HO2 */
    qf[96] = sc[6]*qss_sc[9];
    qr[96] = 0.0;

    /*reaction 98: C2H3 + O2 => CH2O + HCO */
    qf[97] = sc[6]*qss_sc[9];
    qr[97] = 0.0;

    /*reaction 99: C2H4 + CH3 => C2H3 + CH4 */
    qf[98] = sc[12]*sc[20];
    qr[98] = 0.0;

    /*reaction 100: C2H4 + O => CH3 + HCO */
    qf[99] = sc[1]*sc[20];
    qr[99] = 0.0;

    /*reaction 101: C2H4 + OH => C2H3 + H2O */
    qf[100] = sc[4]*sc[20];
    qr[100] = 0.0;

    /*reaction 102: C2H4 + O => CH2CHO + H */
    qf[101] = sc[1]*sc[20];
    qr[101] = 0.0;

    /*reaction 103: C2H4 + H => C2H3 + H2 */
    qf[102] = sc[3]*sc[20];
    qr[102] = 0.0;

    /*reaction 104: C2H3 + H2 => C2H4 + H */
    qf[103] = sc[2]*qss_sc[9];
    qr[103] = 0.0;

    /*reaction 105: H + C2H5 => C2H6 */
    qf[104] = sc[3]*qss_sc[5];
    qr[104] = 0.0;

    /*reaction 106: C2H5 + HO2 => C2H5O + OH */
    qf[105] = sc[7]*qss_sc[5];
    qr[105] = 0.0;

    /*reaction 107: C2H5 + O2 => C2H4 + HO2 */
    qf[106] = sc[6]*qss_sc[5];
    qr[106] = 0.0;

    /*reaction 108: C2H6 + O => C2H5 + OH */
    qf[107] = sc[1]*sc[15];
    qr[107] = 0.0;

    /*reaction 109: C2H6 + OH => C2H5 + H2O */
    qf[108] = sc[4]*sc[15];
    qr[108] = 0.0;

    /*reaction 110: C2H6 + H => C2H5 + H2 */
    qf[109] = sc[3]*sc[15];
    qr[109] = 0.0;

    /*reaction 111: HCCO + O => H + 2.000000 CO */
    qf[110] = sc[1]*qss_sc[8];
    qr[110] = 0.0;

    /*reaction 112: HCCO + OH => 2.000000 HCO */
    qf[111] = sc[4]*qss_sc[8];
    qr[111] = 0.0;

    /*reaction 113: HCCO + O2 => CO2 + HCO */
    qf[112] = sc[6]*qss_sc[8];
    qr[112] = 0.0;

    /*reaction 114: HCCO + H => CH2GSG + CO */
    qf[113] = sc[3]*qss_sc[8];
    qr[113] = 0.0;

    /*reaction 115: CH2GSG + CO => HCCO + H */
    qf[114] = sc[10]*qss_sc[3];
    qr[114] = 0.0;

    /*reaction 116: CH2CO + O => HCCO + OH */
    qf[115] = sc[1]*sc[16];
    qr[115] = 0.0;

    /*reaction 117: CH2CO + H => HCCO + H2 */
    qf[116] = sc[3]*sc[16];
    qr[116] = 0.0;

    /*reaction 118: HCCO + H2 => CH2CO + H */
    qf[117] = sc[2]*qss_sc[8];
    qr[117] = 0.0;

    /*reaction 119: CH2CO + H => CH3 + CO */
    qf[118] = sc[3]*sc[16];
    qr[118] = 0.0;

    /*reaction 120: CH2CO + O => CH2 + CO2 */
    qf[119] = sc[1]*sc[16];
    qr[119] = 0.0;

    /*reaction 121: CH2CO + OH => HCCO + H2O */
    qf[120] = sc[4]*sc[16];
    qr[120] = 0.0;

    /*reaction 122: CH2CHO + O2 => CH2O + CO + OH */
    qf[121] = sc[6]*qss_sc[10];
    qr[121] = 0.0;

    /*reaction 123: CH2CHO => CH2CO + H */
    qf[122] = qss_sc[10];
    qr[122] = 0.0;

    /*reaction 124: CH2CO + H => CH2CHO */
    qf[123] = sc[3]*sc[16];
    qr[123] = 0.0;

    /*reaction 125: C2H5O2 => C2H5 + O2 */
    qf[124] = qss_sc[13];
    qr[124] = 0.0;

    /*reaction 126: C2H5 + O2 => C2H5O2 */
    qf[125] = sc[6]*qss_sc[5];
    qr[125] = 0.0;

    /*reaction 127: C2H5O2 => C2H4 + HO2 */
    qf[126] = qss_sc[13];
    qr[126] = 0.0;

    /*reaction 128: C3H2 + O2 => HCCO + CO + H */
    qf[127] = sc[6]*qss_sc[14];
    qr[127] = 0.0;

    /*reaction 129: C3H2 + OH => C2H2 + HCO */
    qf[128] = sc[4]*qss_sc[14];
    qr[128] = 0.0;

    /*reaction 130: C3H3 + O2 => CH2CO + HCO */
    qf[129] = sc[6]*qss_sc[15];
    qr[129] = 0.0;

    /*reaction 131: C3H3 + HO2 => C3H4XA + O2 */
    qf[130] = sc[7]*qss_sc[15];
    qr[130] = 0.0;

    /*reaction 132: C3H3 + H => C3H2 + H2 */
    qf[131] = sc[3]*qss_sc[15];
    qr[131] = 0.0;

    /*reaction 133: C3H3 + OH => C3H2 + H2O */
    qf[132] = sc[4]*qss_sc[15];
    qr[132] = 0.0;

    /*reaction 134: C3H2 + H2O => C3H3 + OH */
    qf[133] = sc[5]*qss_sc[14];
    qr[133] = 0.0;

    /*reaction 135: C3H4XA + H => C3H3 + H2 */
    qf[134] = sc[3]*sc[21];
    qr[134] = 0.0;

    /*reaction 136: C3H4XA + OH => C3H3 + H2O */
    qf[135] = sc[4]*sc[21];
    qr[135] = 0.0;

    /*reaction 137: C3H4XA + O => C2H4 + CO */
    qf[136] = sc[1]*sc[21];
    qr[136] = 0.0;

    /*reaction 138: C3H5XA + H => C3H4XA + H2 */
    qf[137] = sc[3]*qss_sc[16];
    qr[137] = 0.0;

    /*reaction 139: C3H5XA + HO2 => C3H6 + O2 */
    qf[138] = sc[7]*qss_sc[16];
    qr[138] = 0.0;

    /*reaction 140: C3H5XA + H => C3H6 */
    qf[139] = sc[3]*qss_sc[16];
    qr[139] = 0.0;

    /*reaction 141: C3H5XA => C2H2 + CH3 */
    qf[140] = qss_sc[16];
    qr[140] = 0.0;

    /*reaction 142: C3H5XA => C3H4XA + H */
    qf[141] = qss_sc[16];
    qr[141] = 0.0;

    /*reaction 143: C3H4XA + H => C3H5XA */
    qf[142] = sc[3]*sc[21];
    qr[142] = 0.0;

    /*reaction 144: C3H5XA + CH2O => C3H6 + HCO */
    qf[143] = sc[11]*qss_sc[16];
    qr[143] = 0.0;

    /*reaction 145: C3H6 + H => C2H4 + CH3 */
    qf[144] = sc[3]*sc[19];
    qr[144] = 0.0;

    /*reaction 146: C3H6 + H => C3H5XA + H2 */
    qf[145] = sc[3]*sc[19];
    qr[145] = 0.0;

    /*reaction 147: C3H6 + O => C2H5 + HCO */
    qf[146] = sc[1]*sc[19];
    qr[146] = 0.0;

    /*reaction 148: C3H6 + O => C3H5XA + OH */
    qf[147] = sc[1]*sc[19];
    qr[147] = 0.0;

    /*reaction 149: C3H6 + O => CH2CO + CH3 + H */
    qf[148] = sc[1]*sc[19];
    qr[148] = 0.0;

    /*reaction 150: C3H6 + OH => C3H5XA + H2O */
    qf[149] = sc[4]*sc[19];
    qr[149] = 0.0;

    /*reaction 151: NXC3H7 + O2 => C3H6 + HO2 */
    qf[150] = sc[6]*qss_sc[17];
    qr[150] = 0.0;

    /*reaction 152: NXC3H7 => CH3 + C2H4 */
    qf[151] = qss_sc[17];
    qr[151] = 0.0;

    /*reaction 153: CH3 + C2H4 => NXC3H7 */
    qf[152] = sc[12]*sc[20];
    qr[152] = 0.0;

    /*reaction 154: NXC3H7 => H + C3H6 */
    qf[153] = qss_sc[17];
    qr[153] = 0.0;

    /*reaction 155: H + C3H6 => NXC3H7 */
    qf[154] = sc[3]*sc[19];
    qr[154] = 0.0;

    /*reaction 156: NXC3H7O2 => NXC3H7 + O2 */
    qf[155] = qss_sc[18];
    qr[155] = 0.0;

    /*reaction 157: NXC3H7 + O2 => NXC3H7O2 */
    qf[156] = sc[6]*qss_sc[17];
    qr[156] = 0.0;

    /*reaction 158: C4H6 => 2.000000 C2H3 */
    qf[157] = sc[22];
    qr[157] = 0.0;

    /*reaction 159: 2.000000 C2H3 => C4H6 */
    qf[158] = pow(qss_sc[9], 2.000000);
    qr[158] = 0.0;

    /*reaction 160: C4H6 + OH => CH2O + C3H5XA */
    qf[159] = sc[4]*sc[22];
    qr[159] = 0.0;

    /*reaction 161: C4H6 + OH => C2H5 + CH2CO */
    qf[160] = sc[4]*sc[22];
    qr[160] = 0.0;

    /*reaction 162: C4H6 + O => C2H4 + CH2CO */
    qf[161] = sc[1]*sc[22];
    qr[161] = 0.0;

    /*reaction 163: C4H6 + H => C2H3 + C2H4 */
    qf[162] = sc[3]*sc[22];
    qr[162] = 0.0;

    /*reaction 164: C4H6 + O => CH2O + C3H4XA */
    qf[163] = sc[1]*sc[22];
    qr[163] = 0.0;

    /*reaction 165: H + C4H7 => C4H8X1 */
    qf[164] = sc[3]*qss_sc[19];
    qr[164] = 0.0;

    /*reaction 166: C4H7 => C4H6 + H */
    qf[165] = qss_sc[19];
    qr[165] = 0.0;

    /*reaction 167: C4H6 + H => C4H7 */
    qf[166] = sc[3]*sc[22];
    qr[166] = 0.0;

    /*reaction 168: C4H7 + CH3 => C4H6 + CH4 */
    qf[167] = sc[12]*qss_sc[19];
    qr[167] = 0.0;

    /*reaction 169: C4H7 + HO2 => C4H8X1 + O2 */
    qf[168] = sc[7]*qss_sc[19];
    qr[168] = 0.0;

    /*reaction 170: C4H7 + O2 => C4H6 + HO2 */
    qf[169] = sc[6]*qss_sc[19];
    qr[169] = 0.0;

    /*reaction 171: C4H7 => C2H4 + C2H3 */
    qf[170] = qss_sc[19];
    qr[170] = 0.0;

    /*reaction 172: H + C4H7 => C4H6 + H2 */
    qf[171] = sc[3]*qss_sc[19];
    qr[171] = 0.0;

    /*reaction 173: C4H8X1 + H => C4H7 + H2 */
    qf[172] = sc[3]*sc[23];
    qr[172] = 0.0;

    /*reaction 174: C4H8X1 + OH => NXC3H7 + CH2O */
    qf[173] = sc[4]*sc[23];
    qr[173] = 0.0;

    /*reaction 175: C4H8X1 + OH => CH3CO + C2H6 */
    qf[174] = sc[4]*sc[23];
    qr[174] = 0.0;

    /*reaction 176: C4H8X1 + O => CH3CO + C2H5 */
    qf[175] = sc[1]*sc[23];
    qr[175] = 0.0;

    /*reaction 177: C4H8X1 + O => C3H6 + CH2O */
    qf[176] = sc[1]*sc[23];
    qr[176] = 0.0;

    /*reaction 178: C4H8X1 + OH => C4H7 + H2O */
    qf[177] = sc[4]*sc[23];
    qr[177] = 0.0;

    /*reaction 179: C4H8X1 => C3H5XA + CH3 */
    qf[178] = sc[23];
    qr[178] = 0.0;

    /*reaction 180: C3H5XA + CH3 => C4H8X1 */
    qf[179] = sc[12]*qss_sc[16];
    qr[179] = 0.0;

    /*reaction 181: PXC4H9 => C4H8X1 + H */
    qf[180] = qss_sc[20];
    qr[180] = 0.0;

    /*reaction 182: C4H8X1 + H => PXC4H9 */
    qf[181] = sc[3]*sc[23];
    qr[181] = 0.0;

    /*reaction 183: PXC4H9 => C2H5 + C2H4 */
    qf[182] = qss_sc[20];
    qr[182] = 0.0;

    /*reaction 184: PXC4H9O2 => PXC4H9 + O2 */
    qf[183] = qss_sc[21];
    qr[183] = 0.0;

    /*reaction 185: PXC4H9 + O2 => PXC4H9O2 */
    qf[184] = sc[6]*qss_sc[20];
    qr[184] = 0.0;

    /*reaction 186: C5H9 => C4H6 + CH3 */
    qf[185] = qss_sc[22];
    qr[185] = 0.0;

    /*reaction 187: C5H9 => C3H5XA + C2H4 */
    qf[186] = qss_sc[22];
    qr[186] = 0.0;

    /*reaction 188: C5H10X1 + OH => C5H9 + H2O */
    qf[187] = sc[4]*qss_sc[23];
    qr[187] = 0.0;

    /*reaction 189: C5H10X1 + H => C5H9 + H2 */
    qf[188] = sc[3]*qss_sc[23];
    qr[188] = 0.0;

    /*reaction 190: C5H10X1 => C2H5 + C3H5XA */
    qf[189] = qss_sc[23];
    qr[189] = 0.0;

    /*reaction 191: C5H10X1 + O => C5H9 + OH */
    qf[190] = sc[1]*qss_sc[23];
    qr[190] = 0.0;

    /*reaction 192: C5H11X1 => C3H6 + C2H5 */
    qf[191] = qss_sc[24];
    qr[191] = 0.0;

    /*reaction 193: C5H11X1 => C2H4 + NXC3H7 */
    qf[192] = qss_sc[24];
    qr[192] = 0.0;

    /*reaction 194: C5H11X1 <=> C5H10X1 + H */
    qf[193] = qss_sc[24];
    qr[193] = sc[3]*qss_sc[23];

    /*reaction 195: C6H12X1 => NXC3H7 + C3H5XA */
    qf[194] = qss_sc[25];
    qr[194] = 0.0;

    /*reaction 196: C6H12X1 + OH => C5H11X1 + CH2O */
    qf[195] = sc[4]*qss_sc[25];
    qr[195] = 0.0;

    /*reaction 197: C7H15X2 => C6H12X1 + CH3 */
    qf[196] = qss_sc[26];
    qr[196] = 0.0;

    /*reaction 198: C7H15X2 => PXC4H9 + C3H6 */
    qf[197] = qss_sc[26];
    qr[197] = 0.0;

    /*reaction 199: C7H15X2 => C4H8X1 + NXC3H7 */
    qf[198] = qss_sc[26];
    qr[198] = 0.0;

    /*reaction 200: C7H15X2 => C5H11X1 + C2H4 */
    qf[199] = qss_sc[26];
    qr[199] = 0.0;

    /*reaction 201: C7H15X2 => C2H5 + C5H10X1 */
    qf[200] = qss_sc[26];
    qr[200] = 0.0;

    /*reaction 202: C7H15X2 + HO2 => NXC7H16 + O2 */
    qf[201] = sc[7]*qss_sc[26];
    qr[201] = 0.0;

    /*reaction 203: NXC7H16 + CH3O2 => C7H15X2 + CH3O2H */
    qf[202] = qss_sc[6]*sc[24];
    qr[202] = 0.0;

    /*reaction 204: NXC7H16 + H => C7H15X2 + H2 */
    qf[203] = sc[3]*sc[24];
    qr[203] = 0.0;

    /*reaction 205: NXC7H16 => PXC4H9 + NXC3H7 */
    qf[204] = sc[24];
    qr[204] = 0.0;

    /*reaction 206: NXC7H16 + HO2 => C7H15X2 + H2O2 */
    qf[205] = sc[7]*sc[24];
    qr[205] = 0.0;

    /*reaction 207: NXC7H16 => C5H11X1 + C2H5 */
    qf[206] = sc[24];
    qr[206] = 0.0;

    /*reaction 208: NXC7H16 + CH3O => C7H15X2 + CH3OH */
    qf[207] = qss_sc[4]*sc[24];
    qr[207] = 0.0;

    /*reaction 209: NXC7H16 + O => C7H15X2 + OH */
    qf[208] = sc[1]*sc[24];
    qr[208] = 0.0;

    /*reaction 210: NXC7H16 + OH => C7H15X2 + H2O */
    qf[209] = sc[4]*sc[24];
    qr[209] = 0.0;

    /*reaction 211: NXC7H16 + CH3 => C7H15X2 + CH4 */
    qf[210] = sc[12]*sc[24];
    qr[210] = 0.0;

    double T = tc[1];

    /*compute the mixture concentration */
    double mixture = 0.0;
    for (int i = 0; i < 25; ++i) {
        mixture += sc[i];
    }

    double Corr[211];
    for (int i = 0; i < 211; ++i) {
        Corr[i] = 1.0;
    }

    /* troe */
    {
        double alpha[11];
        alpha[0] = mixture + (TB[0][0] - 1)*sc[10] + (TB[0][1] - 1)*sc[2] + (TB[0][2] - 1)*sc[5] + (TB[0][3] - 1)*sc[9];
        alpha[1] = mixture + (TB[1][0] - 1)*sc[10] + (TB[1][1] - 1)*sc[2] + (TB[1][2] - 1)*sc[5] + (TB[1][3] - 1)*sc[9];
        alpha[2] = mixture + (TB[2][0] - 1)*sc[10] + (TB[2][1] - 1)*sc[2] + (TB[2][2] - 1)*sc[5] + (TB[2][3] - 1)*sc[9] + (TB[2][4] - 1)*sc[13] + (TB[2][5] - 1)*sc[15];
        alpha[3] = mixture + (TB[3][0] - 1)*sc[10] + (TB[3][1] - 1)*sc[2] + (TB[3][2] - 1)*sc[5] + (TB[3][3] - 1)*sc[9];
        alpha[4] = mixture + (TB[4][0] - 1)*sc[10] + (TB[4][1] - 1)*sc[2] + (TB[4][2] - 1)*sc[5] + (TB[4][3] - 1)*sc[9];
        alpha[5] = mixture;
        alpha[6] = mixture + (TB[6][0] - 1)*sc[10] + (TB[6][1] - 1)*sc[2] + (TB[6][2] - 1)*sc[5] + (TB[6][3] - 1)*sc[9];
        alpha[7] = alpha[5];
        alpha[8] = mixture + (TB[8][0] - 1)*sc[10] + (TB[8][1] - 1)*sc[2] + (TB[8][2] - 1)*sc[5] + (TB[8][3] - 1)*sc[9];
        alpha[9] = alpha[5];
        alpha[10] = alpha[5];
#ifdef __INTEL_COMPILER
         #pragma simd
#endif
        for (int i=0; i<11; i++)
        {
            double redP, F, logPred, logFcent, troe_c, troe_n, troe, F_troe;
            redP = alpha[i-0] / k_f_save[i] * phase_units[i] * low_A[i] * exp(low_beta[i] * tc[0] - activation_units[i] * low_Ea[i] *invT);
            F = redP / (1.0 + redP);
            logPred = log10(redP);
            logFcent = log10(
                (fabs(troe_Tsss[i]) > 1.e-100 ? (1.-troe_a[i])*exp(-T/troe_Tsss[i]) : 0.) 
                + (fabs(troe_Ts[i]) > 1.e-100 ? troe_a[i] * exp(-T/troe_Ts[i]) : 0.) 
                + (troe_len[i] == 4 ? exp(-troe_Tss[i] * invT) : 0.) );
            troe_c = -.4 - .67 * logFcent;
            troe_n = .75 - 1.27 * logFcent;
            troe = (troe_c + logPred) / (troe_n - .14*(troe_c + logPred));
            F_troe = pow(10., logFcent / (1.0 + troe*troe));
            Corr[i] = F * F_troe;
        }
    }

    /* simple three-body correction */
    {
        double alpha;
        alpha = mixture + (TB[11][0] - 1)*sc[10] + (TB[11][1] - 1)*sc[2] + (TB[11][2] - 1)*sc[5] + (TB[11][3] - 1)*sc[9];
        Corr[11] = alpha;
        alpha = mixture;
        Corr[12] = alpha;
        Corr[13] = alpha;
        alpha = mixture + (TB[14][0] - 1)*sc[10] + (TB[14][1] - 1)*sc[2] + (TB[14][2] - 1)*sc[5] + (TB[14][3] - 1)*sc[9];
        Corr[14] = alpha;
        alpha = mixture;
        Corr[15] = alpha;
        Corr[16] = alpha;
        Corr[17] = alpha;
    }

    for (int i=0; i<211; i++)
    {
        qf[i] *= Corr[i] * k_f_save[i];
        qr[i] *= Corr[i] * k_f_save[i] / Kc_save[i];
    }

    return;
}
#endif

void comp_k_f_qss(double *  tc, double invT, double *  k_f)
{
#ifdef __INTEL_COMPILER
    #pragma simd
#endif
    k_f[0] = prefactor_units[5] * fwd_A[5]
               * exp(fwd_beta[5] * tc[0] - activation_units[5] * fwd_Ea[5] * invT);
    k_f[1] = prefactor_units[7] * fwd_A[7]
               * exp(fwd_beta[7] * tc[0] - activation_units[7] * fwd_Ea[7] * invT);
    k_f[2] = prefactor_units[8] * fwd_A[8]
               * exp(fwd_beta[8] * tc[0] - activation_units[8] * fwd_Ea[8] * invT);
    k_f[3] = prefactor_units[9] * fwd_A[9]
               * exp(fwd_beta[9] * tc[0] - activation_units[9] * fwd_Ea[9] * invT);
    k_f[4] = prefactor_units[10] * fwd_A[10]
               * exp(fwd_beta[10] * tc[0] - activation_units[10] * fwd_Ea[10] * invT);
    k_f[5] = prefactor_units[12] * fwd_A[12]
               * exp(fwd_beta[12] * tc[0] - activation_units[12] * fwd_Ea[12] * invT);
    k_f[6] = prefactor_units[13] * fwd_A[13]
               * exp(fwd_beta[13] * tc[0] - activation_units[13] * fwd_Ea[13] * invT);
    k_f[7] = prefactor_units[14] * fwd_A[14]
               * exp(fwd_beta[14] * tc[0] - activation_units[14] * fwd_Ea[14] * invT);
    k_f[8] = prefactor_units[15] * fwd_A[15]
               * exp(fwd_beta[15] * tc[0] - activation_units[15] * fwd_Ea[15] * invT);
    k_f[9] = prefactor_units[16] * fwd_A[16]
               * exp(fwd_beta[16] * tc[0] - activation_units[16] * fwd_Ea[16] * invT);
    k_f[10] = prefactor_units[17] * fwd_A[17]
               * exp(fwd_beta[17] * tc[0] - activation_units[17] * fwd_Ea[17] * invT);
    k_f[11] = prefactor_units[33] * fwd_A[33]
               * exp(fwd_beta[33] * tc[0] - activation_units[33] * fwd_Ea[33] * invT);
    k_f[12] = prefactor_units[34] * fwd_A[34]
               * exp(fwd_beta[34] * tc[0] - activation_units[34] * fwd_Ea[34] * invT);
    k_f[13] = prefactor_units[35] * fwd_A[35]
               * exp(fwd_beta[35] * tc[0] - activation_units[35] * fwd_Ea[35] * invT);
    k_f[14] = prefactor_units[36] * fwd_A[36]
               * exp(fwd_beta[36] * tc[0] - activation_units[36] * fwd_Ea[36] * invT);
    k_f[15] = prefactor_units[37] * fwd_A[37]
               * exp(fwd_beta[37] * tc[0] - activation_units[37] * fwd_Ea[37] * invT);
    k_f[16] = prefactor_units[38] * fwd_A[38]
               * exp(fwd_beta[38] * tc[0] - activation_units[38] * fwd_Ea[38] * invT);
    k_f[17] = prefactor_units[39] * fwd_A[39]
               * exp(fwd_beta[39] * tc[0] - activation_units[39] * fwd_Ea[39] * invT);
    k_f[18] = prefactor_units[40] * fwd_A[40]
               * exp(fwd_beta[40] * tc[0] - activation_units[40] * fwd_Ea[40] * invT);
    k_f[19] = prefactor_units[41] * fwd_A[41]
               * exp(fwd_beta[41] * tc[0] - activation_units[41] * fwd_Ea[41] * invT);
    k_f[20] = prefactor_units[42] * fwd_A[42]
               * exp(fwd_beta[42] * tc[0] - activation_units[42] * fwd_Ea[42] * invT);
    k_f[21] = prefactor_units[43] * fwd_A[43]
               * exp(fwd_beta[43] * tc[0] - activation_units[43] * fwd_Ea[43] * invT);
    k_f[22] = prefactor_units[44] * fwd_A[44]
               * exp(fwd_beta[44] * tc[0] - activation_units[44] * fwd_Ea[44] * invT);
    k_f[23] = prefactor_units[45] * fwd_A[45]
               * exp(fwd_beta[45] * tc[0] - activation_units[45] * fwd_Ea[45] * invT);
    k_f[24] = prefactor_units[46] * fwd_A[46]
               * exp(fwd_beta[46] * tc[0] - activation_units[46] * fwd_Ea[46] * invT);
    k_f[25] = prefactor_units[47] * fwd_A[47]
               * exp(fwd_beta[47] * tc[0] - activation_units[47] * fwd_Ea[47] * invT);
    k_f[26] = prefactor_units[49] * fwd_A[49]
               * exp(fwd_beta[49] * tc[0] - activation_units[49] * fwd_Ea[49] * invT);
    k_f[27] = prefactor_units[50] * fwd_A[50]
               * exp(fwd_beta[50] * tc[0] - activation_units[50] * fwd_Ea[50] * invT);
    k_f[28] = prefactor_units[52] * fwd_A[52]
               * exp(fwd_beta[52] * tc[0] - activation_units[52] * fwd_Ea[52] * invT);
    k_f[29] = prefactor_units[55] * fwd_A[55]
               * exp(fwd_beta[55] * tc[0] - activation_units[55] * fwd_Ea[55] * invT);
    k_f[30] = prefactor_units[56] * fwd_A[56]
               * exp(fwd_beta[56] * tc[0] - activation_units[56] * fwd_Ea[56] * invT);
    k_f[31] = prefactor_units[57] * fwd_A[57]
               * exp(fwd_beta[57] * tc[0] - activation_units[57] * fwd_Ea[57] * invT);
    k_f[32] = prefactor_units[58] * fwd_A[58]
               * exp(fwd_beta[58] * tc[0] - activation_units[58] * fwd_Ea[58] * invT);
    k_f[33] = prefactor_units[59] * fwd_A[59]
               * exp(fwd_beta[59] * tc[0] - activation_units[59] * fwd_Ea[59] * invT);
    k_f[34] = prefactor_units[67] * fwd_A[67]
               * exp(fwd_beta[67] * tc[0] - activation_units[67] * fwd_Ea[67] * invT);
    k_f[35] = prefactor_units[68] * fwd_A[68]
               * exp(fwd_beta[68] * tc[0] - activation_units[68] * fwd_Ea[68] * invT);
    k_f[36] = prefactor_units[69] * fwd_A[69]
               * exp(fwd_beta[69] * tc[0] - activation_units[69] * fwd_Ea[69] * invT);
    k_f[37] = prefactor_units[70] * fwd_A[70]
               * exp(fwd_beta[70] * tc[0] - activation_units[70] * fwd_Ea[70] * invT);
    k_f[38] = prefactor_units[71] * fwd_A[71]
               * exp(fwd_beta[71] * tc[0] - activation_units[71] * fwd_Ea[71] * invT);
    k_f[39] = prefactor_units[72] * fwd_A[72]
               * exp(fwd_beta[72] * tc[0] - activation_units[72] * fwd_Ea[72] * invT);
    k_f[40] = prefactor_units[73] * fwd_A[73]
               * exp(fwd_beta[73] * tc[0] - activation_units[73] * fwd_Ea[73] * invT);
    k_f[41] = prefactor_units[74] * fwd_A[74]
               * exp(fwd_beta[74] * tc[0] - activation_units[74] * fwd_Ea[74] * invT);
    k_f[42] = prefactor_units[75] * fwd_A[75]
               * exp(fwd_beta[75] * tc[0] - activation_units[75] * fwd_Ea[75] * invT);
    k_f[43] = prefactor_units[76] * fwd_A[76]
               * exp(fwd_beta[76] * tc[0] - activation_units[76] * fwd_Ea[76] * invT);
    k_f[44] = prefactor_units[77] * fwd_A[77]
               * exp(fwd_beta[77] * tc[0] - activation_units[77] * fwd_Ea[77] * invT);
    k_f[45] = prefactor_units[78] * fwd_A[78]
               * exp(fwd_beta[78] * tc[0] - activation_units[78] * fwd_Ea[78] * invT);
    k_f[46] = prefactor_units[79] * fwd_A[79]
               * exp(fwd_beta[79] * tc[0] - activation_units[79] * fwd_Ea[79] * invT);
    k_f[47] = prefactor_units[80] * fwd_A[80]
               * exp(fwd_beta[80] * tc[0] - activation_units[80] * fwd_Ea[80] * invT);
    k_f[48] = prefactor_units[81] * fwd_A[81]
               * exp(fwd_beta[81] * tc[0] - activation_units[81] * fwd_Ea[81] * invT);
    k_f[49] = prefactor_units[84] * fwd_A[84]
               * exp(fwd_beta[84] * tc[0] - activation_units[84] * fwd_Ea[84] * invT);
    k_f[50] = prefactor_units[85] * fwd_A[85]
               * exp(fwd_beta[85] * tc[0] - activation_units[85] * fwd_Ea[85] * invT);
    k_f[51] = prefactor_units[88] * fwd_A[88]
               * exp(fwd_beta[88] * tc[0] - activation_units[88] * fwd_Ea[88] * invT);
    k_f[52] = prefactor_units[89] * fwd_A[89]
               * exp(fwd_beta[89] * tc[0] - activation_units[89] * fwd_Ea[89] * invT);
    k_f[53] = prefactor_units[90] * fwd_A[90]
               * exp(fwd_beta[90] * tc[0] - activation_units[90] * fwd_Ea[90] * invT);
    k_f[54] = prefactor_units[91] * fwd_A[91]
               * exp(fwd_beta[91] * tc[0] - activation_units[91] * fwd_Ea[91] * invT);
    k_f[55] = prefactor_units[92] * fwd_A[92]
               * exp(fwd_beta[92] * tc[0] - activation_units[92] * fwd_Ea[92] * invT);
    k_f[56] = prefactor_units[93] * fwd_A[93]
               * exp(fwd_beta[93] * tc[0] - activation_units[93] * fwd_Ea[93] * invT);
    k_f[57] = prefactor_units[94] * fwd_A[94]
               * exp(fwd_beta[94] * tc[0] - activation_units[94] * fwd_Ea[94] * invT);
    k_f[58] = prefactor_units[95] * fwd_A[95]
               * exp(fwd_beta[95] * tc[0] - activation_units[95] * fwd_Ea[95] * invT);
    k_f[59] = prefactor_units[96] * fwd_A[96]
               * exp(fwd_beta[96] * tc[0] - activation_units[96] * fwd_Ea[96] * invT);
    k_f[60] = prefactor_units[97] * fwd_A[97]
               * exp(fwd_beta[97] * tc[0] - activation_units[97] * fwd_Ea[97] * invT);
    k_f[61] = prefactor_units[98] * fwd_A[98]
               * exp(fwd_beta[98] * tc[0] - activation_units[98] * fwd_Ea[98] * invT);
    k_f[62] = prefactor_units[99] * fwd_A[99]
               * exp(fwd_beta[99] * tc[0] - activation_units[99] * fwd_Ea[99] * invT);
    k_f[63] = prefactor_units[100] * fwd_A[100]
               * exp(fwd_beta[100] * tc[0] - activation_units[100] * fwd_Ea[100] * invT);
    k_f[64] = prefactor_units[101] * fwd_A[101]
               * exp(fwd_beta[101] * tc[0] - activation_units[101] * fwd_Ea[101] * invT);
    k_f[65] = prefactor_units[102] * fwd_A[102]
               * exp(fwd_beta[102] * tc[0] - activation_units[102] * fwd_Ea[102] * invT);
    k_f[66] = prefactor_units[103] * fwd_A[103]
               * exp(fwd_beta[103] * tc[0] - activation_units[103] * fwd_Ea[103] * invT);
    k_f[67] = prefactor_units[104] * fwd_A[104]
               * exp(fwd_beta[104] * tc[0] - activation_units[104] * fwd_Ea[104] * invT);
    k_f[68] = prefactor_units[105] * fwd_A[105]
               * exp(fwd_beta[105] * tc[0] - activation_units[105] * fwd_Ea[105] * invT);
    k_f[69] = prefactor_units[106] * fwd_A[106]
               * exp(fwd_beta[106] * tc[0] - activation_units[106] * fwd_Ea[106] * invT);
    k_f[70] = prefactor_units[107] * fwd_A[107]
               * exp(fwd_beta[107] * tc[0] - activation_units[107] * fwd_Ea[107] * invT);
    k_f[71] = prefactor_units[108] * fwd_A[108]
               * exp(fwd_beta[108] * tc[0] - activation_units[108] * fwd_Ea[108] * invT);
    k_f[72] = prefactor_units[109] * fwd_A[109]
               * exp(fwd_beta[109] * tc[0] - activation_units[109] * fwd_Ea[109] * invT);
    k_f[73] = prefactor_units[110] * fwd_A[110]
               * exp(fwd_beta[110] * tc[0] - activation_units[110] * fwd_Ea[110] * invT);
    k_f[74] = prefactor_units[111] * fwd_A[111]
               * exp(fwd_beta[111] * tc[0] - activation_units[111] * fwd_Ea[111] * invT);
    k_f[75] = prefactor_units[112] * fwd_A[112]
               * exp(fwd_beta[112] * tc[0] - activation_units[112] * fwd_Ea[112] * invT);
    k_f[76] = prefactor_units[113] * fwd_A[113]
               * exp(fwd_beta[113] * tc[0] - activation_units[113] * fwd_Ea[113] * invT);
    k_f[77] = prefactor_units[114] * fwd_A[114]
               * exp(fwd_beta[114] * tc[0] - activation_units[114] * fwd_Ea[114] * invT);
    k_f[78] = prefactor_units[115] * fwd_A[115]
               * exp(fwd_beta[115] * tc[0] - activation_units[115] * fwd_Ea[115] * invT);
    k_f[79] = prefactor_units[116] * fwd_A[116]
               * exp(fwd_beta[116] * tc[0] - activation_units[116] * fwd_Ea[116] * invT);
    k_f[80] = prefactor_units[117] * fwd_A[117]
               * exp(fwd_beta[117] * tc[0] - activation_units[117] * fwd_Ea[117] * invT);
    k_f[81] = prefactor_units[119] * fwd_A[119]
               * exp(fwd_beta[119] * tc[0] - activation_units[119] * fwd_Ea[119] * invT);
    k_f[82] = prefactor_units[120] * fwd_A[120]
               * exp(fwd_beta[120] * tc[0] - activation_units[120] * fwd_Ea[120] * invT);
    k_f[83] = prefactor_units[121] * fwd_A[121]
               * exp(fwd_beta[121] * tc[0] - activation_units[121] * fwd_Ea[121] * invT);
    k_f[84] = prefactor_units[122] * fwd_A[122]
               * exp(fwd_beta[122] * tc[0] - activation_units[122] * fwd_Ea[122] * invT);
    k_f[85] = prefactor_units[123] * fwd_A[123]
               * exp(fwd_beta[123] * tc[0] - activation_units[123] * fwd_Ea[123] * invT);
    k_f[86] = prefactor_units[124] * fwd_A[124]
               * exp(fwd_beta[124] * tc[0] - activation_units[124] * fwd_Ea[124] * invT);
    k_f[87] = prefactor_units[125] * fwd_A[125]
               * exp(fwd_beta[125] * tc[0] - activation_units[125] * fwd_Ea[125] * invT);
    k_f[88] = prefactor_units[126] * fwd_A[126]
               * exp(fwd_beta[126] * tc[0] - activation_units[126] * fwd_Ea[126] * invT);
    k_f[89] = prefactor_units[127] * fwd_A[127]
               * exp(fwd_beta[127] * tc[0] - activation_units[127] * fwd_Ea[127] * invT);
    k_f[90] = prefactor_units[128] * fwd_A[128]
               * exp(fwd_beta[128] * tc[0] - activation_units[128] * fwd_Ea[128] * invT);
    k_f[91] = prefactor_units[129] * fwd_A[129]
               * exp(fwd_beta[129] * tc[0] - activation_units[129] * fwd_Ea[129] * invT);
    k_f[92] = prefactor_units[130] * fwd_A[130]
               * exp(fwd_beta[130] * tc[0] - activation_units[130] * fwd_Ea[130] * invT);
    k_f[93] = prefactor_units[131] * fwd_A[131]
               * exp(fwd_beta[131] * tc[0] - activation_units[131] * fwd_Ea[131] * invT);
    k_f[94] = prefactor_units[132] * fwd_A[132]
               * exp(fwd_beta[132] * tc[0] - activation_units[132] * fwd_Ea[132] * invT);
    k_f[95] = prefactor_units[133] * fwd_A[133]
               * exp(fwd_beta[133] * tc[0] - activation_units[133] * fwd_Ea[133] * invT);
    k_f[96] = prefactor_units[134] * fwd_A[134]
               * exp(fwd_beta[134] * tc[0] - activation_units[134] * fwd_Ea[134] * invT);
    k_f[97] = prefactor_units[135] * fwd_A[135]
               * exp(fwd_beta[135] * tc[0] - activation_units[135] * fwd_Ea[135] * invT);
    k_f[98] = prefactor_units[137] * fwd_A[137]
               * exp(fwd_beta[137] * tc[0] - activation_units[137] * fwd_Ea[137] * invT);
    k_f[99] = prefactor_units[138] * fwd_A[138]
               * exp(fwd_beta[138] * tc[0] - activation_units[138] * fwd_Ea[138] * invT);
    k_f[100] = prefactor_units[139] * fwd_A[139]
               * exp(fwd_beta[139] * tc[0] - activation_units[139] * fwd_Ea[139] * invT);
    k_f[101] = prefactor_units[140] * fwd_A[140]
               * exp(fwd_beta[140] * tc[0] - activation_units[140] * fwd_Ea[140] * invT);
    k_f[102] = prefactor_units[141] * fwd_A[141]
               * exp(fwd_beta[141] * tc[0] - activation_units[141] * fwd_Ea[141] * invT);
    k_f[103] = prefactor_units[142] * fwd_A[142]
               * exp(fwd_beta[142] * tc[0] - activation_units[142] * fwd_Ea[142] * invT);
    k_f[104] = prefactor_units[143] * fwd_A[143]
               * exp(fwd_beta[143] * tc[0] - activation_units[143] * fwd_Ea[143] * invT);
    k_f[105] = prefactor_units[145] * fwd_A[145]
               * exp(fwd_beta[145] * tc[0] - activation_units[145] * fwd_Ea[145] * invT);
    k_f[106] = prefactor_units[146] * fwd_A[146]
               * exp(fwd_beta[146] * tc[0] - activation_units[146] * fwd_Ea[146] * invT);
    k_f[107] = prefactor_units[147] * fwd_A[147]
               * exp(fwd_beta[147] * tc[0] - activation_units[147] * fwd_Ea[147] * invT);
    k_f[108] = prefactor_units[149] * fwd_A[149]
               * exp(fwd_beta[149] * tc[0] - activation_units[149] * fwd_Ea[149] * invT);
    k_f[109] = prefactor_units[150] * fwd_A[150]
               * exp(fwd_beta[150] * tc[0] - activation_units[150] * fwd_Ea[150] * invT);
    k_f[110] = prefactor_units[151] * fwd_A[151]
               * exp(fwd_beta[151] * tc[0] - activation_units[151] * fwd_Ea[151] * invT);
    k_f[111] = prefactor_units[152] * fwd_A[152]
               * exp(fwd_beta[152] * tc[0] - activation_units[152] * fwd_Ea[152] * invT);
    k_f[112] = prefactor_units[153] * fwd_A[153]
               * exp(fwd_beta[153] * tc[0] - activation_units[153] * fwd_Ea[153] * invT);
    k_f[113] = prefactor_units[154] * fwd_A[154]
               * exp(fwd_beta[154] * tc[0] - activation_units[154] * fwd_Ea[154] * invT);
    k_f[114] = prefactor_units[155] * fwd_A[155]
               * exp(fwd_beta[155] * tc[0] - activation_units[155] * fwd_Ea[155] * invT);
    k_f[115] = prefactor_units[156] * fwd_A[156]
               * exp(fwd_beta[156] * tc[0] - activation_units[156] * fwd_Ea[156] * invT);
    k_f[116] = prefactor_units[157] * fwd_A[157]
               * exp(fwd_beta[157] * tc[0] - activation_units[157] * fwd_Ea[157] * invT);
    k_f[117] = prefactor_units[158] * fwd_A[158]
               * exp(fwd_beta[158] * tc[0] - activation_units[158] * fwd_Ea[158] * invT);
    k_f[118] = prefactor_units[159] * fwd_A[159]
               * exp(fwd_beta[159] * tc[0] - activation_units[159] * fwd_Ea[159] * invT);
    k_f[119] = prefactor_units[160] * fwd_A[160]
               * exp(fwd_beta[160] * tc[0] - activation_units[160] * fwd_Ea[160] * invT);
    k_f[120] = prefactor_units[162] * fwd_A[162]
               * exp(fwd_beta[162] * tc[0] - activation_units[162] * fwd_Ea[162] * invT);
    k_f[121] = prefactor_units[164] * fwd_A[164]
               * exp(fwd_beta[164] * tc[0] - activation_units[164] * fwd_Ea[164] * invT);
    k_f[122] = prefactor_units[165] * fwd_A[165]
               * exp(fwd_beta[165] * tc[0] - activation_units[165] * fwd_Ea[165] * invT);
    k_f[123] = prefactor_units[166] * fwd_A[166]
               * exp(fwd_beta[166] * tc[0] - activation_units[166] * fwd_Ea[166] * invT);
    k_f[124] = prefactor_units[167] * fwd_A[167]
               * exp(fwd_beta[167] * tc[0] - activation_units[167] * fwd_Ea[167] * invT);
    k_f[125] = prefactor_units[168] * fwd_A[168]
               * exp(fwd_beta[168] * tc[0] - activation_units[168] * fwd_Ea[168] * invT);
    k_f[126] = prefactor_units[169] * fwd_A[169]
               * exp(fwd_beta[169] * tc[0] - activation_units[169] * fwd_Ea[169] * invT);
    k_f[127] = prefactor_units[170] * fwd_A[170]
               * exp(fwd_beta[170] * tc[0] - activation_units[170] * fwd_Ea[170] * invT);
    k_f[128] = prefactor_units[171] * fwd_A[171]
               * exp(fwd_beta[171] * tc[0] - activation_units[171] * fwd_Ea[171] * invT);
    k_f[129] = prefactor_units[172] * fwd_A[172]
               * exp(fwd_beta[172] * tc[0] - activation_units[172] * fwd_Ea[172] * invT);
    k_f[130] = prefactor_units[173] * fwd_A[173]
               * exp(fwd_beta[173] * tc[0] - activation_units[173] * fwd_Ea[173] * invT);
    k_f[131] = prefactor_units[174] * fwd_A[174]
               * exp(fwd_beta[174] * tc[0] - activation_units[174] * fwd_Ea[174] * invT);
    k_f[132] = prefactor_units[175] * fwd_A[175]
               * exp(fwd_beta[175] * tc[0] - activation_units[175] * fwd_Ea[175] * invT);
    k_f[133] = prefactor_units[177] * fwd_A[177]
               * exp(fwd_beta[177] * tc[0] - activation_units[177] * fwd_Ea[177] * invT);
    k_f[134] = prefactor_units[178] * fwd_A[178]
               * exp(fwd_beta[178] * tc[0] - activation_units[178] * fwd_Ea[178] * invT);
    k_f[135] = prefactor_units[179] * fwd_A[179]
               * exp(fwd_beta[179] * tc[0] - activation_units[179] * fwd_Ea[179] * invT);
    k_f[136] = prefactor_units[180] * fwd_A[180]
               * exp(fwd_beta[180] * tc[0] - activation_units[180] * fwd_Ea[180] * invT);
    k_f[137] = prefactor_units[181] * fwd_A[181]
               * exp(fwd_beta[181] * tc[0] - activation_units[181] * fwd_Ea[181] * invT);
    k_f[138] = prefactor_units[182] * fwd_A[182]
               * exp(fwd_beta[182] * tc[0] - activation_units[182] * fwd_Ea[182] * invT);
    k_f[139] = prefactor_units[183] * fwd_A[183]
               * exp(fwd_beta[183] * tc[0] - activation_units[183] * fwd_Ea[183] * invT);
    k_f[140] = prefactor_units[184] * fwd_A[184]
               * exp(fwd_beta[184] * tc[0] - activation_units[184] * fwd_Ea[184] * invT);
    k_f[141] = prefactor_units[185] * fwd_A[185]
               * exp(fwd_beta[185] * tc[0] - activation_units[185] * fwd_Ea[185] * invT);
    k_f[142] = prefactor_units[186] * fwd_A[186]
               * exp(fwd_beta[186] * tc[0] - activation_units[186] * fwd_Ea[186] * invT);
    k_f[143] = prefactor_units[187] * fwd_A[187]
               * exp(fwd_beta[187] * tc[0] - activation_units[187] * fwd_Ea[187] * invT);
    k_f[144] = prefactor_units[188] * fwd_A[188]
               * exp(fwd_beta[188] * tc[0] - activation_units[188] * fwd_Ea[188] * invT);
    k_f[145] = prefactor_units[189] * fwd_A[189]
               * exp(fwd_beta[189] * tc[0] - activation_units[189] * fwd_Ea[189] * invT);
    k_f[146] = prefactor_units[190] * fwd_A[190]
               * exp(fwd_beta[190] * tc[0] - activation_units[190] * fwd_Ea[190] * invT);
    k_f[147] = prefactor_units[191] * fwd_A[191]
               * exp(fwd_beta[191] * tc[0] - activation_units[191] * fwd_Ea[191] * invT);
    k_f[148] = prefactor_units[192] * fwd_A[192]
               * exp(fwd_beta[192] * tc[0] - activation_units[192] * fwd_Ea[192] * invT);
    k_f[149] = prefactor_units[193] * fwd_A[193]
               * exp(fwd_beta[193] * tc[0] - activation_units[193] * fwd_Ea[193] * invT);
    k_f[150] = prefactor_units[194] * fwd_A[194]
               * exp(fwd_beta[194] * tc[0] - activation_units[194] * fwd_Ea[194] * invT);
    k_f[151] = prefactor_units[195] * fwd_A[195]
               * exp(fwd_beta[195] * tc[0] - activation_units[195] * fwd_Ea[195] * invT);
    k_f[152] = prefactor_units[196] * fwd_A[196]
               * exp(fwd_beta[196] * tc[0] - activation_units[196] * fwd_Ea[196] * invT);
    k_f[153] = prefactor_units[197] * fwd_A[197]
               * exp(fwd_beta[197] * tc[0] - activation_units[197] * fwd_Ea[197] * invT);
    k_f[154] = prefactor_units[198] * fwd_A[198]
               * exp(fwd_beta[198] * tc[0] - activation_units[198] * fwd_Ea[198] * invT);
    k_f[155] = prefactor_units[199] * fwd_A[199]
               * exp(fwd_beta[199] * tc[0] - activation_units[199] * fwd_Ea[199] * invT);
    k_f[156] = prefactor_units[200] * fwd_A[200]
               * exp(fwd_beta[200] * tc[0] - activation_units[200] * fwd_Ea[200] * invT);
    k_f[157] = prefactor_units[201] * fwd_A[201]
               * exp(fwd_beta[201] * tc[0] - activation_units[201] * fwd_Ea[201] * invT);
    k_f[158] = prefactor_units[202] * fwd_A[202]
               * exp(fwd_beta[202] * tc[0] - activation_units[202] * fwd_Ea[202] * invT);
    k_f[159] = prefactor_units[203] * fwd_A[203]
               * exp(fwd_beta[203] * tc[0] - activation_units[203] * fwd_Ea[203] * invT);
    k_f[160] = prefactor_units[204] * fwd_A[204]
               * exp(fwd_beta[204] * tc[0] - activation_units[204] * fwd_Ea[204] * invT);
    k_f[161] = prefactor_units[205] * fwd_A[205]
               * exp(fwd_beta[205] * tc[0] - activation_units[205] * fwd_Ea[205] * invT);
    k_f[162] = prefactor_units[206] * fwd_A[206]
               * exp(fwd_beta[206] * tc[0] - activation_units[206] * fwd_Ea[206] * invT);
    k_f[163] = prefactor_units[207] * fwd_A[207]
               * exp(fwd_beta[207] * tc[0] - activation_units[207] * fwd_Ea[207] * invT);
    k_f[164] = prefactor_units[208] * fwd_A[208]
               * exp(fwd_beta[208] * tc[0] - activation_units[208] * fwd_Ea[208] * invT);
    k_f[165] = prefactor_units[209] * fwd_A[209]
               * exp(fwd_beta[209] * tc[0] - activation_units[209] * fwd_Ea[209] * invT);
    k_f[166] = prefactor_units[210] * fwd_A[210]
               * exp(fwd_beta[210] * tc[0] - activation_units[210] * fwd_Ea[210] * invT);

    return;
}

void comp_Kc_qss(double *  tc, double invT, double *  Kc)
{
    /*compute the Gibbs free energy */
    double g_RT[25], g_RT_qss[27];
    gibbs(g_RT, tc);
    gibbs_qss(g_RT_qss, tc);

    /*Reaction 6 */
    Kc[0] = g_RT[10] - g_RT[16] + g_RT_qss[2];
    /*Reaction 8 */
    Kc[1] = -g_RT[3] - g_RT[11] + g_RT_qss[4];
    /*Reaction 9 */
    Kc[2] = -g_RT[3] - g_RT[18] + g_RT_qss[9];
    /*Reaction 10 */
    Kc[3] = g_RT[3] + g_RT[20] - g_RT_qss[5];
    /*Reaction 11 */
    Kc[4] = -g_RT[10] - g_RT[12] + g_RT_qss[12];
    /*Reaction 13 */
    Kc[5] = -g_RT_qss[2] + g_RT_qss[3];
    /*Reaction 14 */
    Kc[6] = g_RT_qss[2] - g_RT_qss[3];
    /*Reaction 15 */
    Kc[7] = -g_RT[3] - g_RT[10] + g_RT_qss[1];
    /*Reaction 16 */
    Kc[8] = -g_RT[6] - g_RT[12] + g_RT_qss[6];
    /*Reaction 17 */
    Kc[9] = g_RT[6] + g_RT[12] - g_RT_qss[6];
    /*Reaction 18 */
    Kc[10] = -g_RT[11] - g_RT[12] + g_RT_qss[11];
    /*Reaction 34 */
    Kc[11] = -g_RT[1] + g_RT[6] + g_RT_qss[0] - g_RT_qss[1];
    /*Reaction 35 */
    Kc[12] = -2.000000*g_RT[3] + g_RT[6] - g_RT[9] + g_RT_qss[2];
    /*Reaction 36 */
    Kc[13] = -g_RT[5] + g_RT[6] - g_RT[10] + g_RT_qss[2];
    /*Reaction 37 */
    Kc[14] = g_RT[1] - 2.000000*g_RT[3] - g_RT[10] + g_RT_qss[2];
    /*Reaction 38 */
    Kc[15] = -g_RT[1] + g_RT[6] - g_RT[11] + g_RT_qss[2];
    /*Reaction 39 */
    Kc[16] = -g_RT[2] + g_RT[3] - g_RT_qss[0] + g_RT_qss[2];
    /*Reaction 40 */
    Kc[17] = g_RT[2] - g_RT[3] + g_RT_qss[0] - g_RT_qss[2];
    /*Reaction 41 */
    Kc[18] = g_RT[4] - g_RT[5] - g_RT_qss[0] + g_RT_qss[2];
    /*Reaction 42 */
    Kc[19] = -g_RT[4] + g_RT[5] + g_RT_qss[0] - g_RT_qss[2];
    /*Reaction 43 */
    Kc[20] = -g_RT[2] + g_RT[6] - g_RT[9] + g_RT_qss[2];
    /*Reaction 44 */
    Kc[21] = -g_RT[2] + g_RT[3] - g_RT_qss[0] + g_RT_qss[3];
    /*Reaction 45 */
    Kc[22] = g_RT[2] - g_RT[3] - g_RT[12] + g_RT_qss[3];
    /*Reaction 46 */
    Kc[23] = -g_RT[2] + g_RT[3] + g_RT[12] - g_RT_qss[3];
    /*Reaction 47 */
    Kc[24] = -g_RT[3] - g_RT[4] + g_RT[6] - g_RT[10] + g_RT_qss[3];
    /*Reaction 48 */
    Kc[25] = -g_RT[3] + g_RT[4] - g_RT[11] + g_RT_qss[3];
    /*Reaction 50 */
    Kc[26] = g_RT[4] - g_RT[5] + g_RT[12] - g_RT_qss[3];
    /*Reaction 51 */
    Kc[27] = -g_RT[4] + g_RT[5] - g_RT[12] + g_RT_qss[3];
    /*Reaction 53 */
    Kc[28] = -g_RT[4] + g_RT[7] + g_RT[12] - g_RT_qss[4];
    /*Reaction 56 */
    Kc[29] = -g_RT[2] + g_RT[3] + g_RT[12] - g_RT_qss[2];
    /*Reaction 57 */
    Kc[30] = g_RT[2] - g_RT[3] - g_RT[12] + g_RT_qss[2];
    /*Reaction 58 */
    Kc[31] = -g_RT[3] + 2.000000*g_RT[12] - g_RT_qss[5];
    /*Reaction 59 */
    Kc[32] = g_RT[4] - g_RT[5] + g_RT[12] - g_RT_qss[2];
    /*Reaction 60 */
    Kc[33] = -g_RT[4] + g_RT[5] - g_RT[12] + g_RT_qss[2];
    /*Reaction 68 */
    Kc[34] = g_RT[6] - g_RT[7] - g_RT[10] + g_RT_qss[1];
    /*Reaction 69 */
    Kc[35] = g_RT[1] - g_RT[3] - g_RT[9] + g_RT_qss[1];
    /*Reaction 70 */
    Kc[36] = g_RT[4] - g_RT[5] - g_RT[10] + g_RT_qss[1];
    /*Reaction 71 */
    Kc[37] = -g_RT[2] + g_RT[3] - g_RT[10] + g_RT_qss[1];
    /*Reaction 72 */
    Kc[38] = g_RT[1] - g_RT[4] - g_RT[10] + g_RT_qss[1];
    /*Reaction 73 */
    Kc[39] = -g_RT[10] + g_RT[12] - g_RT[13] + g_RT_qss[1];
    /*Reaction 74 */
    Kc[40] = g_RT[4] - g_RT[5] + g_RT[11] - g_RT_qss[1];
    /*Reaction 75 */
    Kc[41] = g_RT[1] - g_RT[4] + g_RT[11] - g_RT_qss[1];
    /*Reaction 76 */
    Kc[42] = -g_RT[2] + g_RT[3] + g_RT[11] - g_RT_qss[1];
    /*Reaction 77 */
    Kc[43] = g_RT[11] + g_RT[12] - g_RT[13] - g_RT_qss[1];
    /*Reaction 78 */
    Kc[44] = -g_RT[11] - g_RT[14] + 2.000000*g_RT_qss[4];
    /*Reaction 79 */
    Kc[45] = g_RT[6] - g_RT[7] - g_RT[11] + g_RT_qss[4];
    /*Reaction 80 */
    Kc[46] = g_RT[2] - g_RT[3] - g_RT[14] + g_RT_qss[4];
    /*Reaction 81 */
    Kc[47] = g_RT[4] - g_RT[5] + g_RT[14] - g_RT_qss[4];
    /*Reaction 82 */
    Kc[48] = g_RT[9] - g_RT[10] - g_RT[11] + g_RT_qss[3];
    /*Reaction 85 */
    Kc[49] = -g_RT[4] + g_RT[17] - g_RT_qss[1];
    /*Reaction 86 */
    Kc[50] = g_RT[4] - g_RT[17] + g_RT_qss[1];
    /*Reaction 89 */
    Kc[51] = g_RT[12] - 2.000000*g_RT_qss[4] + g_RT_qss[6];
    /*Reaction 90 */
    Kc[52] = -g_RT[6] + g_RT[7] + g_RT_qss[6] - g_RT_qss[7];
    /*Reaction 91 */
    Kc[53] = -g_RT[4] - g_RT_qss[4] + g_RT_qss[7];
    /*Reaction 92 */
    Kc[54] = g_RT[1] - g_RT[10] + g_RT[18] - g_RT_qss[2];
    /*Reaction 93 */
    Kc[55] = g_RT[1] - g_RT[3] + g_RT[18] - g_RT_qss[8];
    /*Reaction 94 */
    Kc[56] = -g_RT[2] + g_RT[3] - g_RT[18] + g_RT_qss[9];
    /*Reaction 95 */
    Kc[57] = -g_RT[1] + g_RT[6] + g_RT_qss[9] - g_RT_qss[10];
    /*Reaction 96 */
    Kc[58] = g_RT[12] - g_RT[19] + g_RT_qss[9];
    /*Reaction 97 */
    Kc[59] = g_RT[6] - g_RT[7] - g_RT[18] + g_RT_qss[9];
    /*Reaction 98 */
    Kc[60] = g_RT[6] - g_RT[11] - g_RT_qss[1] + g_RT_qss[9];
    /*Reaction 99 */
    Kc[61] = g_RT[12] - g_RT[13] + g_RT[20] - g_RT_qss[9];
    /*Reaction 100 */
    Kc[62] = g_RT[1] - g_RT[12] + g_RT[20] - g_RT_qss[1];
    /*Reaction 101 */
    Kc[63] = g_RT[4] - g_RT[5] + g_RT[20] - g_RT_qss[9];
    /*Reaction 102 */
    Kc[64] = g_RT[1] - g_RT[3] + g_RT[20] - g_RT_qss[10];
    /*Reaction 103 */
    Kc[65] = -g_RT[2] + g_RT[3] + g_RT[20] - g_RT_qss[9];
    /*Reaction 104 */
    Kc[66] = g_RT[2] - g_RT[3] - g_RT[20] + g_RT_qss[9];
    /*Reaction 105 */
    Kc[67] = g_RT[3] - g_RT[15] + g_RT_qss[5];
    /*Reaction 106 */
    Kc[68] = -g_RT[4] + g_RT[7] + g_RT_qss[5] - g_RT_qss[11];
    /*Reaction 107 */
    Kc[69] = g_RT[6] - g_RT[7] - g_RT[20] + g_RT_qss[5];
    /*Reaction 108 */
    Kc[70] = g_RT[1] - g_RT[4] + g_RT[15] - g_RT_qss[5];
    /*Reaction 109 */
    Kc[71] = g_RT[4] - g_RT[5] + g_RT[15] - g_RT_qss[5];
    /*Reaction 110 */
    Kc[72] = -g_RT[2] + g_RT[3] + g_RT[15] - g_RT_qss[5];
    /*Reaction 111 */
    Kc[73] = g_RT[1] - g_RT[3] - 2.000000*g_RT[10] + g_RT_qss[8];
    /*Reaction 112 */
    Kc[74] = g_RT[4] - 2.000000*g_RT_qss[1] + g_RT_qss[8];
    /*Reaction 113 */
    Kc[75] = g_RT[6] - g_RT[9] - g_RT_qss[1] + g_RT_qss[8];
    /*Reaction 114 */
    Kc[76] = g_RT[3] - g_RT[10] - g_RT_qss[3] + g_RT_qss[8];
    /*Reaction 115 */
    Kc[77] = -g_RT[3] + g_RT[10] + g_RT_qss[3] - g_RT_qss[8];
    /*Reaction 116 */
    Kc[78] = g_RT[1] - g_RT[4] + g_RT[16] - g_RT_qss[8];
    /*Reaction 117 */
    Kc[79] = -g_RT[2] + g_RT[3] + g_RT[16] - g_RT_qss[8];
    /*Reaction 118 */
    Kc[80] = g_RT[2] - g_RT[3] - g_RT[16] + g_RT_qss[8];
    /*Reaction 120 */
    Kc[81] = g_RT[1] - g_RT[9] + g_RT[16] - g_RT_qss[2];
    /*Reaction 121 */
    Kc[82] = g_RT[4] - g_RT[5] + g_RT[16] - g_RT_qss[8];
    /*Reaction 122 */
    Kc[83] = -g_RT[4] + g_RT[6] - g_RT[10] - g_RT[11] + g_RT_qss[10];
    /*Reaction 123 */
    Kc[84] = -g_RT[3] - g_RT[16] + g_RT_qss[10];
    /*Reaction 124 */
    Kc[85] = g_RT[3] + g_RT[16] - g_RT_qss[10];
    /*Reaction 125 */
    Kc[86] = -g_RT[6] - g_RT_qss[5] + g_RT_qss[13];
    /*Reaction 126 */
    Kc[87] = g_RT[6] + g_RT_qss[5] - g_RT_qss[13];
    /*Reaction 127 */
    Kc[88] = -g_RT[7] - g_RT[20] + g_RT_qss[13];
    /*Reaction 128 */
    Kc[89] = -g_RT[3] + g_RT[6] - g_RT[10] - g_RT_qss[8] + g_RT_qss[14];
    /*Reaction 129 */
    Kc[90] = g_RT[4] - g_RT[18] - g_RT_qss[1] + g_RT_qss[14];
    /*Reaction 130 */
    Kc[91] = g_RT[6] - g_RT[16] - g_RT_qss[1] + g_RT_qss[15];
    /*Reaction 131 */
    Kc[92] = -g_RT[6] + g_RT[7] - g_RT[21] + g_RT_qss[15];
    /*Reaction 132 */
    Kc[93] = -g_RT[2] + g_RT[3] - g_RT_qss[14] + g_RT_qss[15];
    /*Reaction 133 */
    Kc[94] = g_RT[4] - g_RT[5] - g_RT_qss[14] + g_RT_qss[15];
    /*Reaction 134 */
    Kc[95] = -g_RT[4] + g_RT[5] + g_RT_qss[14] - g_RT_qss[15];
    /*Reaction 135 */
    Kc[96] = -g_RT[2] + g_RT[3] + g_RT[21] - g_RT_qss[15];
    /*Reaction 136 */
    Kc[97] = g_RT[4] - g_RT[5] + g_RT[21] - g_RT_qss[15];
    /*Reaction 138 */
    Kc[98] = -g_RT[2] + g_RT[3] - g_RT[21] + g_RT_qss[16];
    /*Reaction 139 */
    Kc[99] = -g_RT[6] + g_RT[7] - g_RT[19] + g_RT_qss[16];
    /*Reaction 140 */
    Kc[100] = g_RT[3] - g_RT[19] + g_RT_qss[16];
    /*Reaction 141 */
    Kc[101] = -g_RT[12] - g_RT[18] + g_RT_qss[16];
    /*Reaction 142 */
    Kc[102] = -g_RT[3] - g_RT[21] + g_RT_qss[16];
    /*Reaction 143 */
    Kc[103] = g_RT[3] + g_RT[21] - g_RT_qss[16];
    /*Reaction 144 */
    Kc[104] = g_RT[11] - g_RT[19] - g_RT_qss[1] + g_RT_qss[16];
    /*Reaction 146 */
    Kc[105] = -g_RT[2] + g_RT[3] + g_RT[19] - g_RT_qss[16];
    /*Reaction 147 */
    Kc[106] = g_RT[1] + g_RT[19] - g_RT_qss[1] - g_RT_qss[5];
    /*Reaction 148 */
    Kc[107] = g_RT[1] - g_RT[4] + g_RT[19] - g_RT_qss[16];
    /*Reaction 150 */
    Kc[108] = g_RT[4] - g_RT[5] + g_RT[19] - g_RT_qss[16];
    /*Reaction 151 */
    Kc[109] = g_RT[6] - g_RT[7] - g_RT[19] + g_RT_qss[17];
    /*Reaction 152 */
    Kc[110] = -g_RT[12] - g_RT[20] + g_RT_qss[17];
    /*Reaction 153 */
    Kc[111] = g_RT[12] + g_RT[20] - g_RT_qss[17];
    /*Reaction 154 */
    Kc[112] = -g_RT[3] - g_RT[19] + g_RT_qss[17];
    /*Reaction 155 */
    Kc[113] = g_RT[3] + g_RT[19] - g_RT_qss[17];
    /*Reaction 156 */
    Kc[114] = -g_RT[6] - g_RT_qss[17] + g_RT_qss[18];
    /*Reaction 157 */
    Kc[115] = g_RT[6] + g_RT_qss[17] - g_RT_qss[18];
    /*Reaction 158 */
    Kc[116] = g_RT[22] - 2.000000*g_RT_qss[9];
    /*Reaction 159 */
    Kc[117] = -g_RT[22] + 2.000000*g_RT_qss[9];
    /*Reaction 160 */
    Kc[118] = g_RT[4] - g_RT[11] + g_RT[22] - g_RT_qss[16];
    /*Reaction 161 */
    Kc[119] = g_RT[4] - g_RT[16] + g_RT[22] - g_RT_qss[5];
    /*Reaction 163 */
    Kc[120] = g_RT[3] - g_RT[20] + g_RT[22] - g_RT_qss[9];
    /*Reaction 165 */
    Kc[121] = g_RT[3] - g_RT[23] + g_RT_qss[19];
    /*Reaction 166 */
    Kc[122] = -g_RT[3] - g_RT[22] + g_RT_qss[19];
    /*Reaction 167 */
    Kc[123] = g_RT[3] + g_RT[22] - g_RT_qss[19];
    /*Reaction 168 */
    Kc[124] = g_RT[12] - g_RT[13] - g_RT[22] + g_RT_qss[19];
    /*Reaction 169 */
    Kc[125] = -g_RT[6] + g_RT[7] - g_RT[23] + g_RT_qss[19];
    /*Reaction 170 */
    Kc[126] = g_RT[6] - g_RT[7] - g_RT[22] + g_RT_qss[19];
    /*Reaction 171 */
    Kc[127] = -g_RT[20] - g_RT_qss[9] + g_RT_qss[19];
    /*Reaction 172 */
    Kc[128] = -g_RT[2] + g_RT[3] - g_RT[22] + g_RT_qss[19];
    /*Reaction 173 */
    Kc[129] = -g_RT[2] + g_RT[3] + g_RT[23] - g_RT_qss[19];
    /*Reaction 174 */
    Kc[130] = g_RT[4] - g_RT[11] + g_RT[23] - g_RT_qss[17];
    /*Reaction 175 */
    Kc[131] = g_RT[4] - g_RT[15] + g_RT[23] - g_RT_qss[12];
    /*Reaction 176 */
    Kc[132] = g_RT[1] + g_RT[23] - g_RT_qss[5] - g_RT_qss[12];
    /*Reaction 178 */
    Kc[133] = g_RT[4] - g_RT[5] + g_RT[23] - g_RT_qss[19];
    /*Reaction 179 */
    Kc[134] = -g_RT[12] + g_RT[23] - g_RT_qss[16];
    /*Reaction 180 */
    Kc[135] = g_RT[12] - g_RT[23] + g_RT_qss[16];
    /*Reaction 181 */
    Kc[136] = -g_RT[3] - g_RT[23] + g_RT_qss[20];
    /*Reaction 182 */
    Kc[137] = g_RT[3] + g_RT[23] - g_RT_qss[20];
    /*Reaction 183 */
    Kc[138] = -g_RT[20] - g_RT_qss[5] + g_RT_qss[20];
    /*Reaction 184 */
    Kc[139] = -g_RT[6] - g_RT_qss[20] + g_RT_qss[21];
    /*Reaction 185 */
    Kc[140] = g_RT[6] + g_RT_qss[20] - g_RT_qss[21];
    /*Reaction 186 */
    Kc[141] = -g_RT[12] - g_RT[22] + g_RT_qss[22];
    /*Reaction 187 */
    Kc[142] = -g_RT[20] - g_RT_qss[16] + g_RT_qss[22];
    /*Reaction 188 */
    Kc[143] = g_RT[4] - g_RT[5] - g_RT_qss[22] + g_RT_qss[23];
    /*Reaction 189 */
    Kc[144] = -g_RT[2] + g_RT[3] - g_RT_qss[22] + g_RT_qss[23];
    /*Reaction 190 */
    Kc[145] = -g_RT_qss[5] - g_RT_qss[16] + g_RT_qss[23];
    /*Reaction 191 */
    Kc[146] = g_RT[1] - g_RT[4] - g_RT_qss[22] + g_RT_qss[23];
    /*Reaction 192 */
    Kc[147] = -g_RT[19] - g_RT_qss[5] + g_RT_qss[24];
    /*Reaction 193 */
    Kc[148] = -g_RT[20] - g_RT_qss[17] + g_RT_qss[24];
    /*Reaction 194 */
    Kc[149] = -g_RT[3] - g_RT_qss[23] + g_RT_qss[24];
    /*Reaction 195 */
    Kc[150] = -g_RT_qss[16] - g_RT_qss[17] + g_RT_qss[25];
    /*Reaction 196 */
    Kc[151] = g_RT[4] - g_RT[11] - g_RT_qss[24] + g_RT_qss[25];
    /*Reaction 197 */
    Kc[152] = -g_RT[12] - g_RT_qss[25] + g_RT_qss[26];
    /*Reaction 198 */
    Kc[153] = -g_RT[19] - g_RT_qss[20] + g_RT_qss[26];
    /*Reaction 199 */
    Kc[154] = -g_RT[23] - g_RT_qss[17] + g_RT_qss[26];
    /*Reaction 200 */
    Kc[155] = -g_RT[20] - g_RT_qss[24] + g_RT_qss[26];
    /*Reaction 201 */
    Kc[156] = -g_RT_qss[5] - g_RT_qss[23] + g_RT_qss[26];
    /*Reaction 202 */
    Kc[157] = -g_RT[6] + g_RT[7] - g_RT[24] + g_RT_qss[26];
    /*Reaction 203 */
    Kc[158] = g_RT[24] + g_RT_qss[6] - g_RT_qss[7] - g_RT_qss[26];
    /*Reaction 204 */
    Kc[159] = -g_RT[2] + g_RT[3] + g_RT[24] - g_RT_qss[26];
    /*Reaction 205 */
    Kc[160] = g_RT[24] - g_RT_qss[17] - g_RT_qss[20];
    /*Reaction 206 */
    Kc[161] = g_RT[7] - g_RT[8] + g_RT[24] - g_RT_qss[26];
    /*Reaction 207 */
    Kc[162] = g_RT[24] - g_RT_qss[5] - g_RT_qss[24];
    /*Reaction 208 */
    Kc[163] = -g_RT[14] + g_RT[24] + g_RT_qss[4] - g_RT_qss[26];
    /*Reaction 209 */
    Kc[164] = g_RT[1] - g_RT[4] + g_RT[24] - g_RT_qss[26];
    /*Reaction 210 */
    Kc[165] = g_RT[4] - g_RT[5] + g_RT[24] - g_RT_qss[26];
    /*Reaction 211 */
    Kc[166] = g_RT[12] - g_RT[13] + g_RT[24] - g_RT_qss[26];

#ifdef __INTEL_COMPILER
     #pragma simd
#endif
    for (int i=0; i<167; ++i) {
        Kc[i] = exp(Kc[i]);
    };

    /*reference concentration: P_atm / (RT) in inverse mol/m^3 */
    double refC = 101325 / 8.31446 * invT;
    double refCinv = 1 / refC;

    Kc[0] *= refCinv;
    Kc[1] *= refC;
    Kc[2] *= refC;
    Kc[3] *= refCinv;
    Kc[4] *= refC;
    Kc[7] *= refC;
    Kc[8] *= refC;
    Kc[9] *= refCinv;
    Kc[10] *= refC;
    Kc[12] *= refC;
    Kc[14] *= refC;
    Kc[24] *= refC;
    Kc[49] *= refC;
    Kc[50] *= refCinv;
    Kc[53] *= refC;
    Kc[58] *= refCinv;
    Kc[67] *= refCinv;
    Kc[73] *= refC;
    Kc[83] *= refC;
    Kc[84] *= refC;
    Kc[85] *= refCinv;
    Kc[86] *= refC;
    Kc[87] *= refCinv;
    Kc[88] *= refC;
    Kc[89] *= refC;
    Kc[100] *= refCinv;
    Kc[101] *= refC;
    Kc[102] *= refC;
    Kc[103] *= refCinv;
    Kc[110] *= refC;
    Kc[111] *= refCinv;
    Kc[112] *= refC;
    Kc[113] *= refCinv;
    Kc[114] *= refC;
    Kc[115] *= refCinv;
    Kc[116] *= refC;
    Kc[117] *= refCinv;
    Kc[121] *= refCinv;
    Kc[122] *= refC;
    Kc[123] *= refCinv;
    Kc[127] *= refC;
    Kc[134] *= refC;
    Kc[135] *= refCinv;
    Kc[136] *= refC;
    Kc[137] *= refCinv;
    Kc[138] *= refC;
    Kc[139] *= refC;
    Kc[140] *= refCinv;
    Kc[141] *= refC;
    Kc[142] *= refC;
    Kc[145] *= refC;
    Kc[147] *= refC;
    Kc[148] *= refC;
    Kc[149] *= refC;
    Kc[150] *= refC;
    Kc[152] *= refC;
    Kc[153] *= refC;
    Kc[154] *= refC;
    Kc[155] *= refC;
    Kc[156] *= refC;
    Kc[160] *= refC;
    Kc[162] *= refC;

    return;
}

void comp_qss_coeff(double *  qf_co, double *  qr_co, double *  sc, double *  tc, double invT)
{

    /*reaction 6: CO + CH2 (+M) => CH2CO (+M) */
    qf_co[0] = 1.0*sc[10];
    qr_co[0] = 0.0;

    /*reaction 8: CH3O (+M) => CH2O + H (+M) */
    qf_co[1] = 1.0;
    qr_co[1] = 0.0;

    /*reaction 9: C2H3 (+M) => H + C2H2 (+M) */
    qf_co[2] = 1.0;
    qr_co[2] = 0.0;

    /*reaction 10: H + C2H4 (+M) <=> C2H5 (+M) */
    qf_co[3] = sc[3]*sc[20];
    qr_co[3] = 1.0;

    /*reaction 11: CH3CO (+M) => CH3 + CO (+M) */
    qf_co[4] = 1.0;
    qr_co[4] = 0.0;

    /*reaction 13: CH2GSG + M => CH2 + M */
    qf_co[5] = 1.0;
    qr_co[5] = 0.0;

    /*reaction 14: CH2 + M => CH2GSG + M */
    qf_co[6] = 1.0;
    qr_co[6] = 0.0;

    /*reaction 15: HCO + M => H + CO + M */
    qf_co[7] = 1.0;
    qr_co[7] = 0.0;

    /*reaction 16: CH3O2 + M => CH3 + O2 + M */
    qf_co[8] = 1.0;
    qr_co[8] = 0.0;

    /*reaction 17: CH3 + O2 + M => CH3O2 + M */
    qf_co[9] = sc[6]*sc[12];
    qr_co[9] = 0.0;

    /*reaction 18: C2H5O + M => CH3 + CH2O + M */
    qf_co[10] = 1.0;
    qr_co[10] = 0.0;

    /*reaction 34: CH + O2 => HCO + O */
    qf_co[11] = sc[6];
    qr_co[11] = 0.0;

    /*reaction 35: CH2 + O2 => CO2 + 2.000000 H */
    qf_co[12] = sc[6];
    qr_co[12] = 0.0;

    /*reaction 36: CH2 + O2 => CO + H2O */
    qf_co[13] = sc[6];
    qr_co[13] = 0.0;

    /*reaction 37: CH2 + O => CO + 2.000000 H */
    qf_co[14] = sc[1];
    qr_co[14] = 0.0;

    /*reaction 38: CH2 + O2 => CH2O + O */
    qf_co[15] = sc[6];
    qr_co[15] = 0.0;

    /*reaction 39: CH2 + H => CH + H2 */
    qf_co[16] = sc[3];
    qr_co[16] = 0.0;

    /*reaction 40: CH + H2 => CH2 + H */
    qf_co[17] = sc[2];
    qr_co[17] = 0.0;

    /*reaction 41: CH2 + OH => CH + H2O */
    qf_co[18] = sc[4];
    qr_co[18] = 0.0;

    /*reaction 42: CH + H2O => CH2 + OH */
    qf_co[19] = sc[5];
    qr_co[19] = 0.0;

    /*reaction 43: CH2 + O2 => CO2 + H2 */
    qf_co[20] = sc[6];
    qr_co[20] = 0.0;

    /*reaction 44: CH2GSG + H => CH + H2 */
    qf_co[21] = sc[3];
    qr_co[21] = 0.0;

    /*reaction 45: CH2GSG + H2 => CH3 + H */
    qf_co[22] = sc[2];
    qr_co[22] = 0.0;

    /*reaction 46: CH3 + H => CH2GSG + H2 */
    qf_co[23] = sc[3]*sc[12];
    qr_co[23] = 0.0;

    /*reaction 47: CH2GSG + O2 => CO + OH + H */
    qf_co[24] = sc[6];
    qr_co[24] = 0.0;

    /*reaction 48: CH2GSG + OH => CH2O + H */
    qf_co[25] = sc[4];
    qr_co[25] = 0.0;

    /*reaction 50: CH3 + OH => CH2GSG + H2O */
    qf_co[26] = sc[4]*sc[12];
    qr_co[26] = 0.0;

    /*reaction 51: CH2GSG + H2O => CH3 + OH */
    qf_co[27] = sc[5];
    qr_co[27] = 0.0;

    /*reaction 53: CH3 + HO2 => CH3O + OH */
    qf_co[28] = sc[7]*sc[12];
    qr_co[28] = 0.0;

    /*reaction 56: CH3 + H => CH2 + H2 */
    qf_co[29] = sc[3]*sc[12];
    qr_co[29] = 0.0;

    /*reaction 57: CH2 + H2 => CH3 + H */
    qf_co[30] = sc[2];
    qr_co[30] = 0.0;

    /*reaction 58: 2.000000 CH3 <=> H + C2H5 */
    qf_co[31] = pow(sc[12], 2.000000);
    qr_co[31] = sc[3];

    /*reaction 59: CH3 + OH => CH2 + H2O */
    qf_co[32] = sc[4]*sc[12];
    qr_co[32] = 0.0;

    /*reaction 60: CH2 + H2O => CH3 + OH */
    qf_co[33] = sc[5];
    qr_co[33] = 0.0;

    /*reaction 68: HCO + O2 => CO + HO2 */
    qf_co[34] = sc[6];
    qr_co[34] = 0.0;

    /*reaction 69: HCO + O => CO2 + H */
    qf_co[35] = sc[1];
    qr_co[35] = 0.0;

    /*reaction 70: HCO + OH => CO + H2O */
    qf_co[36] = sc[4];
    qr_co[36] = 0.0;

    /*reaction 71: HCO + H => CO + H2 */
    qf_co[37] = sc[3];
    qr_co[37] = 0.0;

    /*reaction 72: HCO + O => CO + OH */
    qf_co[38] = sc[1];
    qr_co[38] = 0.0;

    /*reaction 73: HCO + CH3 => CH4 + CO */
    qf_co[39] = 1.0*sc[12];
    qr_co[39] = 0.0;

    /*reaction 74: CH2O + OH => HCO + H2O */
    qf_co[40] = sc[4]*sc[11];
    qr_co[40] = 0.0;

    /*reaction 75: CH2O + O => HCO + OH */
    qf_co[41] = sc[1]*sc[11];
    qr_co[41] = 0.0;

    /*reaction 76: CH2O + H => HCO + H2 */
    qf_co[42] = sc[3]*sc[11];
    qr_co[42] = 0.0;

    /*reaction 77: CH2O + CH3 => HCO + CH4 */
    qf_co[43] = sc[11]*sc[12];
    qr_co[43] = 0.0;

    /*reaction 78: 2.000000 CH3O => CH3OH + CH2O */
    qf_co[44] = 1.0;
    qr_co[44] = 0.0;

    /*reaction 79: CH3O + O2 => CH2O + HO2 */
    qf_co[45] = sc[6];
    qr_co[45] = 0.0;

    /*reaction 80: CH3O + H2 => CH3OH + H */
    qf_co[46] = sc[2];
    qr_co[46] = 0.0;

    /*reaction 81: CH3OH + OH => CH3O + H2O */
    qf_co[47] = sc[4]*sc[14];
    qr_co[47] = 0.0;

    /*reaction 82: CH2GSG + CO2 => CH2O + CO */
    qf_co[48] = sc[9];
    qr_co[48] = 0.0;

    /*reaction 85: HOCHO => HCO + OH */
    qf_co[49] = sc[17];
    qr_co[49] = 0.0;

    /*reaction 86: HCO + OH => HOCHO */
    qf_co[50] = sc[4];
    qr_co[50] = 0.0;

    /*reaction 89: CH3O2 + CH3 => 2.000000 CH3O */
    qf_co[51] = sc[12];
    qr_co[51] = 0.0;

    /*reaction 90: CH3O2 + HO2 => CH3O2H + O2 */
    qf_co[52] = sc[7];
    qr_co[52] = 0.0;

    /*reaction 91: CH3O2H => CH3O + OH */
    qf_co[53] = 1.0;
    qr_co[53] = 0.0;

    /*reaction 92: C2H2 + O => CH2 + CO */
    qf_co[54] = sc[1]*sc[18];
    qr_co[54] = 0.0;

    /*reaction 93: C2H2 + O => HCCO + H */
    qf_co[55] = sc[1]*sc[18];
    qr_co[55] = 0.0;

    /*reaction 94: C2H3 + H => C2H2 + H2 */
    qf_co[56] = sc[3];
    qr_co[56] = 0.0;

    /*reaction 95: C2H3 + O2 => CH2CHO + O */
    qf_co[57] = sc[6];
    qr_co[57] = 0.0;

    /*reaction 96: C2H3 + CH3 => C3H6 */
    qf_co[58] = sc[12];
    qr_co[58] = 0.0;

    /*reaction 97: C2H3 + O2 => C2H2 + HO2 */
    qf_co[59] = sc[6];
    qr_co[59] = 0.0;

    /*reaction 98: C2H3 + O2 => CH2O + HCO */
    qf_co[60] = sc[6];
    qr_co[60] = 0.0;

    /*reaction 99: C2H4 + CH3 => C2H3 + CH4 */
    qf_co[61] = sc[12]*sc[20];
    qr_co[61] = 0.0;

    /*reaction 100: C2H4 + O => CH3 + HCO */
    qf_co[62] = sc[1]*sc[20];
    qr_co[62] = 0.0;

    /*reaction 101: C2H4 + OH => C2H3 + H2O */
    qf_co[63] = sc[4]*sc[20];
    qr_co[63] = 0.0;

    /*reaction 102: C2H4 + O => CH2CHO + H */
    qf_co[64] = sc[1]*sc[20];
    qr_co[64] = 0.0;

    /*reaction 103: C2H4 + H => C2H3 + H2 */
    qf_co[65] = sc[3]*sc[20];
    qr_co[65] = 0.0;

    /*reaction 104: C2H3 + H2 => C2H4 + H */
    qf_co[66] = sc[2];
    qr_co[66] = 0.0;

    /*reaction 105: H + C2H5 => C2H6 */
    qf_co[67] = sc[3];
    qr_co[67] = 0.0;

    /*reaction 106: C2H5 + HO2 => C2H5O + OH */
    qf_co[68] = sc[7];
    qr_co[68] = 0.0;

    /*reaction 107: C2H5 + O2 => C2H4 + HO2 */
    qf_co[69] = sc[6];
    qr_co[69] = 0.0;

    /*reaction 108: C2H6 + O => C2H5 + OH */
    qf_co[70] = sc[1]*sc[15];
    qr_co[70] = 0.0;

    /*reaction 109: C2H6 + OH => C2H5 + H2O */
    qf_co[71] = sc[4]*sc[15];
    qr_co[71] = 0.0;

    /*reaction 110: C2H6 + H => C2H5 + H2 */
    qf_co[72] = sc[3]*sc[15];
    qr_co[72] = 0.0;

    /*reaction 111: HCCO + O => H + 2.000000 CO */
    qf_co[73] = sc[1];
    qr_co[73] = 0.0;

    /*reaction 112: HCCO + OH => 2.000000 HCO */
    qf_co[74] = sc[4];
    qr_co[74] = 0.0;

    /*reaction 113: HCCO + O2 => CO2 + HCO */
    qf_co[75] = sc[6];
    qr_co[75] = 0.0;

    /*reaction 114: HCCO + H => CH2GSG + CO */
    qf_co[76] = sc[3];
    qr_co[76] = 0.0;

    /*reaction 115: CH2GSG + CO => HCCO + H */
    qf_co[77] = sc[10];
    qr_co[77] = 0.0;

    /*reaction 116: CH2CO + O => HCCO + OH */
    qf_co[78] = sc[1]*sc[16];
    qr_co[78] = 0.0;

    /*reaction 117: CH2CO + H => HCCO + H2 */
    qf_co[79] = sc[3]*sc[16];
    qr_co[79] = 0.0;

    /*reaction 118: HCCO + H2 => CH2CO + H */
    qf_co[80] = sc[2];
    qr_co[80] = 0.0;

    /*reaction 120: CH2CO + O => CH2 + CO2 */
    qf_co[81] = sc[1]*sc[16];
    qr_co[81] = 0.0;

    /*reaction 121: CH2CO + OH => HCCO + H2O */
    qf_co[82] = sc[4]*sc[16];
    qr_co[82] = 0.0;

    /*reaction 122: CH2CHO + O2 => CH2O + CO + OH */
    qf_co[83] = sc[6];
    qr_co[83] = 0.0;

    /*reaction 123: CH2CHO => CH2CO + H */
    qf_co[84] = 1.0;
    qr_co[84] = 0.0;

    /*reaction 124: CH2CO + H => CH2CHO */
    qf_co[85] = sc[3]*sc[16];
    qr_co[85] = 0.0;

    /*reaction 125: C2H5O2 => C2H5 + O2 */
    qf_co[86] = 1.0;
    qr_co[86] = 0.0;

    /*reaction 126: C2H5 + O2 => C2H5O2 */
    qf_co[87] = sc[6];
    qr_co[87] = 0.0;

    /*reaction 127: C2H5O2 => C2H4 + HO2 */
    qf_co[88] = 1.0;
    qr_co[88] = 0.0;

    /*reaction 128: C3H2 + O2 => HCCO + CO + H */
    qf_co[89] = sc[6];
    qr_co[89] = 0.0;

    /*reaction 129: C3H2 + OH => C2H2 + HCO */
    qf_co[90] = sc[4];
    qr_co[90] = 0.0;

    /*reaction 130: C3H3 + O2 => CH2CO + HCO */
    qf_co[91] = sc[6];
    qr_co[91] = 0.0;

    /*reaction 131: C3H3 + HO2 => C3H4XA + O2 */
    qf_co[92] = sc[7];
    qr_co[92] = 0.0;

    /*reaction 132: C3H3 + H => C3H2 + H2 */
    qf_co[93] = sc[3];
    qr_co[93] = 0.0;

    /*reaction 133: C3H3 + OH => C3H2 + H2O */
    qf_co[94] = sc[4];
    qr_co[94] = 0.0;

    /*reaction 134: C3H2 + H2O => C3H3 + OH */
    qf_co[95] = sc[5];
    qr_co[95] = 0.0;

    /*reaction 135: C3H4XA + H => C3H3 + H2 */
    qf_co[96] = sc[3]*sc[21];
    qr_co[96] = 0.0;

    /*reaction 136: C3H4XA + OH => C3H3 + H2O */
    qf_co[97] = sc[4]*sc[21];
    qr_co[97] = 0.0;

    /*reaction 138: C3H5XA + H => C3H4XA + H2 */
    qf_co[98] = sc[3];
    qr_co[98] = 0.0;

    /*reaction 139: C3H5XA + HO2 => C3H6 + O2 */
    qf_co[99] = sc[7];
    qr_co[99] = 0.0;

    /*reaction 140: C3H5XA + H => C3H6 */
    qf_co[100] = sc[3];
    qr_co[100] = 0.0;

    /*reaction 141: C3H5XA => C2H2 + CH3 */
    qf_co[101] = 1.0;
    qr_co[101] = 0.0;

    /*reaction 142: C3H5XA => C3H4XA + H */
    qf_co[102] = 1.0;
    qr_co[102] = 0.0;

    /*reaction 143: C3H4XA + H => C3H5XA */
    qf_co[103] = sc[3]*sc[21];
    qr_co[103] = 0.0;

    /*reaction 144: C3H5XA + CH2O => C3H6 + HCO */
    qf_co[104] = sc[11];
    qr_co[104] = 0.0;

    /*reaction 146: C3H6 + H => C3H5XA + H2 */
    qf_co[105] = sc[3]*sc[19];
    qr_co[105] = 0.0;

    /*reaction 147: C3H6 + O => C2H5 + HCO */
    qf_co[106] = sc[1]*sc[19];
    qr_co[106] = 0.0;

    /*reaction 148: C3H6 + O => C3H5XA + OH */
    qf_co[107] = sc[1]*sc[19];
    qr_co[107] = 0.0;

    /*reaction 150: C3H6 + OH => C3H5XA + H2O */
    qf_co[108] = sc[4]*sc[19];
    qr_co[108] = 0.0;

    /*reaction 151: NXC3H7 + O2 => C3H6 + HO2 */
    qf_co[109] = sc[6];
    qr_co[109] = 0.0;

    /*reaction 152: NXC3H7 => CH3 + C2H4 */
    qf_co[110] = 1.0;
    qr_co[110] = 0.0;

    /*reaction 153: CH3 + C2H4 => NXC3H7 */
    qf_co[111] = sc[12]*sc[20];
    qr_co[111] = 0.0;

    /*reaction 154: NXC3H7 => H + C3H6 */
    qf_co[112] = 1.0;
    qr_co[112] = 0.0;

    /*reaction 155: H + C3H6 => NXC3H7 */
    qf_co[113] = sc[3]*sc[19];
    qr_co[113] = 0.0;

    /*reaction 156: NXC3H7O2 => NXC3H7 + O2 */
    qf_co[114] = 1.0;
    qr_co[114] = 0.0;

    /*reaction 157: NXC3H7 + O2 => NXC3H7O2 */
    qf_co[115] = sc[6];
    qr_co[115] = 0.0;

    /*reaction 158: C4H6 => 2.000000 C2H3 */
    qf_co[116] = sc[22];
    qr_co[116] = 0.0;

    /*reaction 159: 2.000000 C2H3 => C4H6 */
    qf_co[117] = 1.0;
    qr_co[117] = 0.0;

    /*reaction 160: C4H6 + OH => CH2O + C3H5XA */
    qf_co[118] = sc[4]*sc[22];
    qr_co[118] = 0.0;

    /*reaction 161: C4H6 + OH => C2H5 + CH2CO */
    qf_co[119] = sc[4]*sc[22];
    qr_co[119] = 0.0;

    /*reaction 163: C4H6 + H => C2H3 + C2H4 */
    qf_co[120] = sc[3]*sc[22];
    qr_co[120] = 0.0;

    /*reaction 165: H + C4H7 => C4H8X1 */
    qf_co[121] = sc[3];
    qr_co[121] = 0.0;

    /*reaction 166: C4H7 => C4H6 + H */
    qf_co[122] = 1.0;
    qr_co[122] = 0.0;

    /*reaction 167: C4H6 + H => C4H7 */
    qf_co[123] = sc[3]*sc[22];
    qr_co[123] = 0.0;

    /*reaction 168: C4H7 + CH3 => C4H6 + CH4 */
    qf_co[124] = sc[12];
    qr_co[124] = 0.0;

    /*reaction 169: C4H7 + HO2 => C4H8X1 + O2 */
    qf_co[125] = sc[7];
    qr_co[125] = 0.0;

    /*reaction 170: C4H7 + O2 => C4H6 + HO2 */
    qf_co[126] = sc[6];
    qr_co[126] = 0.0;

    /*reaction 171: C4H7 => C2H4 + C2H3 */
    qf_co[127] = 1.0;
    qr_co[127] = 0.0;

    /*reaction 172: H + C4H7 => C4H6 + H2 */
    qf_co[128] = sc[3];
    qr_co[128] = 0.0;

    /*reaction 173: C4H8X1 + H => C4H7 + H2 */
    qf_co[129] = sc[3]*sc[23];
    qr_co[129] = 0.0;

    /*reaction 174: C4H8X1 + OH => NXC3H7 + CH2O */
    qf_co[130] = sc[4]*sc[23];
    qr_co[130] = 0.0;

    /*reaction 175: C4H8X1 + OH => CH3CO + C2H6 */
    qf_co[131] = sc[4]*sc[23];
    qr_co[131] = 0.0;

    /*reaction 176: C4H8X1 + O => CH3CO + C2H5 */
    qf_co[132] = sc[1]*sc[23];
    qr_co[132] = 0.0;

    /*reaction 178: C4H8X1 + OH => C4H7 + H2O */
    qf_co[133] = sc[4]*sc[23];
    qr_co[133] = 0.0;

    /*reaction 179: C4H8X1 => C3H5XA + CH3 */
    qf_co[134] = sc[23];
    qr_co[134] = 0.0;

    /*reaction 180: C3H5XA + CH3 => C4H8X1 */
    qf_co[135] = sc[12];
    qr_co[135] = 0.0;

    /*reaction 181: PXC4H9 => C4H8X1 + H */
    qf_co[136] = 1.0;
    qr_co[136] = 0.0;

    /*reaction 182: C4H8X1 + H => PXC4H9 */
    qf_co[137] = sc[3]*sc[23];
    qr_co[137] = 0.0;

    /*reaction 183: PXC4H9 => C2H5 + C2H4 */
    qf_co[138] = 1.0;
    qr_co[138] = 0.0;

    /*reaction 184: PXC4H9O2 => PXC4H9 + O2 */
    qf_co[139] = 1.0;
    qr_co[139] = 0.0;

    /*reaction 185: PXC4H9 + O2 => PXC4H9O2 */
    qf_co[140] = sc[6];
    qr_co[140] = 0.0;

    /*reaction 186: C5H9 => C4H6 + CH3 */
    qf_co[141] = 1.0;
    qr_co[141] = 0.0;

    /*reaction 187: C5H9 => C3H5XA + C2H4 */
    qf_co[142] = 1.0;
    qr_co[142] = 0.0;

    /*reaction 188: C5H10X1 + OH => C5H9 + H2O */
    qf_co[143] = sc[4];
    qr_co[143] = 0.0;

    /*reaction 189: C5H10X1 + H => C5H9 + H2 */
    qf_co[144] = sc[3];
    qr_co[144] = 0.0;

    /*reaction 190: C5H10X1 => C2H5 + C3H5XA */
    qf_co[145] = 1.0;
    qr_co[145] = 0.0;

    /*reaction 191: C5H10X1 + O => C5H9 + OH */
    qf_co[146] = sc[1];
    qr_co[146] = 0.0;

    /*reaction 192: C5H11X1 => C3H6 + C2H5 */
    qf_co[147] = 1.0;
    qr_co[147] = 0.0;

    /*reaction 193: C5H11X1 => C2H4 + NXC3H7 */
    qf_co[148] = 1.0;
    qr_co[148] = 0.0;

    /*reaction 194: C5H11X1 <=> C5H10X1 + H */
    qf_co[149] = 1.0;
    qr_co[149] = sc[3];

    /*reaction 195: C6H12X1 => NXC3H7 + C3H5XA */
    qf_co[150] = 1.0;
    qr_co[150] = 0.0;

    /*reaction 196: C6H12X1 + OH => C5H11X1 + CH2O */
    qf_co[151] = sc[4];
    qr_co[151] = 0.0;

    /*reaction 197: C7H15X2 => C6H12X1 + CH3 */
    qf_co[152] = 1.0;
    qr_co[152] = 0.0;

    /*reaction 198: C7H15X2 => PXC4H9 + C3H6 */
    qf_co[153] = 1.0;
    qr_co[153] = 0.0;

    /*reaction 199: C7H15X2 => C4H8X1 + NXC3H7 */
    qf_co[154] = 1.0;
    qr_co[154] = 0.0;

    /*reaction 200: C7H15X2 => C5H11X1 + C2H4 */
    qf_co[155] = 1.0;
    qr_co[155] = 0.0;

    /*reaction 201: C7H15X2 => C2H5 + C5H10X1 */
    qf_co[156] = 1.0;
    qr_co[156] = 0.0;

    /*reaction 202: C7H15X2 + HO2 => NXC7H16 + O2 */
    qf_co[157] = sc[7];
    qr_co[157] = 0.0;

    /*reaction 203: NXC7H16 + CH3O2 => C7H15X2 + CH3O2H */
    qf_co[158] = 1.0*sc[24];
    qr_co[158] = 0.0;

    /*reaction 204: NXC7H16 + H => C7H15X2 + H2 */
    qf_co[159] = sc[3]*sc[24];
    qr_co[159] = 0.0;

    /*reaction 205: NXC7H16 => PXC4H9 + NXC3H7 */
    qf_co[160] = sc[24];
    qr_co[160] = 0.0;

    /*reaction 206: NXC7H16 + HO2 => C7H15X2 + H2O2 */
    qf_co[161] = sc[7]*sc[24];
    qr_co[161] = 0.0;

    /*reaction 207: NXC7H16 => C5H11X1 + C2H5 */
    qf_co[162] = sc[24];
    qr_co[162] = 0.0;

    /*reaction 208: NXC7H16 + CH3O => C7H15X2 + CH3OH */
    qf_co[163] = 1.0*sc[24];
    qr_co[163] = 0.0;

    /*reaction 209: NXC7H16 + O => C7H15X2 + OH */
    qf_co[164] = sc[1]*sc[24];
    qr_co[164] = 0.0;

    /*reaction 210: NXC7H16 + OH => C7H15X2 + H2O */
    qf_co[165] = sc[4]*sc[24];
    qr_co[165] = 0.0;

    /*reaction 211: NXC7H16 + CH3 => C7H15X2 + CH4 */
    qf_co[166] = sc[12]*sc[24];
    qr_co[166] = 0.0;

    double T = tc[1];

    /*compute the mixture concentration */
    double mixture = 0.0;
    for (int i = 0; i < 25; ++i) {
        mixture += sc[i];
    }

    double Corr[167];
    for (int i = 0; i < 167; ++i) {
        Corr[i] = 1.0;
    }

    /* troe */
    {
        double alpha[5];
        alpha[0] = mixture;
        alpha[1] = alpha[0];
        alpha[2] = mixture + (TB[8][0] - 1)*sc[10] + (TB[8][1] - 1)*sc[2] + (TB[8][2] - 1)*sc[5] + (TB[8][3] - 1)*sc[9];
        alpha[3] = alpha[0];
        alpha[4] = alpha[0];
#ifdef __INTEL_COMPILER
         #pragma simd
#endif
        double redP, F, logPred, logFcent, troe_c, troe_n, troe, F_troe;
        /*Index for alpha is 0 */
        /*Reaction index is 5 */
        /*QSS reaction list index (corresponds to index needed by k_f_save_qss, Corr, Kc_save_qss) is 0 */
        redP = alpha[0] / k_f_save_qss[0] * phase_units[5] * low_A[5] * exp(low_beta[5] * tc[0] - activation_units[5] * low_Ea[5] *invT);
        F = redP / (1.0 + redP);
        logPred = log10(redP);
        logFcent = log10(
            (fabs(troe_Tsss[5]) > 1.e-100 ? (1.-troe_a[5])*exp(-T/troe_Tsss[5]) : 0.) 
            + (fabs(troe_Ts[5]) > 1.e-100 ? troe_a[5] * exp(-T/troe_Ts[5]) : 0.) 
            + (troe_len[5] == 4 ? exp(-troe_Tss[5] * invT) : 0.) );
        troe_c = -.4 - .67 * logFcent;
        troe_n = .75 - 1.27 * logFcent;
        troe = (troe_c + logPred) / (troe_n - .14*(troe_c + logPred));
        F_troe = pow(10., logFcent / (1.0 + troe*troe));
        Corr[0] = F * F_troe;
        /*Index for alpha is 1 */
        /*Reaction index is 7 */
        /*QSS reaction list index (corresponds to index needed by k_f_save_qss, Corr, Kc_save_qss) is 1 */
        redP = alpha[1] / k_f_save_qss[1] * phase_units[7] * low_A[7] * exp(low_beta[7] * tc[0] - activation_units[7] * low_Ea[7] *invT);
        F = redP / (1.0 + redP);
        logPred = log10(redP);
        logFcent = log10(
            (fabs(troe_Tsss[7]) > 1.e-100 ? (1.-troe_a[7])*exp(-T/troe_Tsss[7]) : 0.) 
            + (fabs(troe_Ts[7]) > 1.e-100 ? troe_a[7] * exp(-T/troe_Ts[7]) : 0.) 
            + (troe_len[7] == 4 ? exp(-troe_Tss[7] * invT) : 0.) );
        troe_c = -.4 - .67 * logFcent;
        troe_n = .75 - 1.27 * logFcent;
        troe = (troe_c + logPred) / (troe_n - .14*(troe_c + logPred));
        F_troe = pow(10., logFcent / (1.0 + troe*troe));
        Corr[1] = F * F_troe;
        /*Index for alpha is 2 */
        /*Reaction index is 8 */
        /*QSS reaction list index (corresponds to index needed by k_f_save_qss, Corr, Kc_save_qss) is 2 */
        redP = alpha[2] / k_f_save_qss[2] * phase_units[8] * low_A[8] * exp(low_beta[8] * tc[0] - activation_units[8] * low_Ea[8] *invT);
        F = redP / (1.0 + redP);
        logPred = log10(redP);
        logFcent = log10(
            (fabs(troe_Tsss[8]) > 1.e-100 ? (1.-troe_a[8])*exp(-T/troe_Tsss[8]) : 0.) 
            + (fabs(troe_Ts[8]) > 1.e-100 ? troe_a[8] * exp(-T/troe_Ts[8]) : 0.) 
            + (troe_len[8] == 4 ? exp(-troe_Tss[8] * invT) : 0.) );
        troe_c = -.4 - .67 * logFcent;
        troe_n = .75 - 1.27 * logFcent;
        troe = (troe_c + logPred) / (troe_n - .14*(troe_c + logPred));
        F_troe = pow(10., logFcent / (1.0 + troe*troe));
        Corr[2] = F * F_troe;
        /*Index for alpha is 3 */
        /*Reaction index is 9 */
        /*QSS reaction list index (corresponds to index needed by k_f_save_qss, Corr, Kc_save_qss) is 3 */
        redP = alpha[3] / k_f_save_qss[3] * phase_units[9] * low_A[9] * exp(low_beta[9] * tc[0] - activation_units[9] * low_Ea[9] *invT);
        F = redP / (1.0 + redP);
        logPred = log10(redP);
        logFcent = log10(
            (fabs(troe_Tsss[9]) > 1.e-100 ? (1.-troe_a[9])*exp(-T/troe_Tsss[9]) : 0.) 
            + (fabs(troe_Ts[9]) > 1.e-100 ? troe_a[9] * exp(-T/troe_Ts[9]) : 0.) 
            + (troe_len[9] == 4 ? exp(-troe_Tss[9] * invT) : 0.) );
        troe_c = -.4 - .67 * logFcent;
        troe_n = .75 - 1.27 * logFcent;
        troe = (troe_c + logPred) / (troe_n - .14*(troe_c + logPred));
        F_troe = pow(10., logFcent / (1.0 + troe*troe));
        Corr[3] = F * F_troe;
        /*Index for alpha is 4 */
        /*Reaction index is 10 */
        /*QSS reaction list index (corresponds to index needed by k_f_save_qss, Corr, Kc_save_qss) is 4 */
        redP = alpha[4] / k_f_save_qss[4] * phase_units[10] * low_A[10] * exp(low_beta[10] * tc[0] - activation_units[10] * low_Ea[10] *invT);
        F = redP / (1.0 + redP);
        logPred = log10(redP);
        logFcent = log10(
            (fabs(troe_Tsss[10]) > 1.e-100 ? (1.-troe_a[10])*exp(-T/troe_Tsss[10]) : 0.) 
            + (fabs(troe_Ts[10]) > 1.e-100 ? troe_a[10] * exp(-T/troe_Ts[10]) : 0.) 
            + (troe_len[10] == 4 ? exp(-troe_Tss[10] * invT) : 0.) );
        troe_c = -.4 - .67 * logFcent;
        troe_n = .75 - 1.27 * logFcent;
        troe = (troe_c + logPred) / (troe_n - .14*(troe_c + logPred));
        F_troe = pow(10., logFcent / (1.0 + troe*troe));
        Corr[4] = F * F_troe;
    }

    /* simple three-body correction */
    {
        double alpha;
        alpha = mixture;
        Corr[5] = alpha;
        Corr[6] = alpha;
        alpha = mixture + (TB[14][0] - 1)*sc[10] + (TB[14][1] - 1)*sc[2] + (TB[14][2] - 1)*sc[5] + (TB[14][3] - 1)*sc[9];
        Corr[7] = alpha;
        alpha = mixture;
        Corr[8] = alpha;
        Corr[9] = alpha;
        Corr[10] = alpha;
    }

    for (int i=0; i<167; i++)
    {
        qf_co[i] *= Corr[i] * k_f_save_qss[i];
        qr_co[i] *= Corr[i] * k_f_save_qss[i] / Kc_save_qss[i];
    }

    return;
}

void comp_qss_sc(double * sc, double * sc_qss, double * tc, double invT)
{

    double H_0, H_1, H_2;
    double  qf_co[167], qr_co[167];
    double epsilon = 1e-12;

    comp_qss_coeff(qf_co, qr_co, sc, tc, invT);

    /*QSS species 6: CH3O2 */

    double CH3O2_num = epsilon -qf_co[9];
    double CH3O2_denom = epsilon -qf_co[8] -qf_co[51] -qf_co[52] -qf_co[158];

    sc_qss[6] = CH3O2_num/CH3O2_denom;

    /*QSS species 12: CH3CO */

    double CH3CO_num = epsilon -qf_co[131];
    double CH3CO_denom = epsilon -qf_co[4];

    sc_qss[12] = CH3CO_num/CH3CO_denom;

    /*QSS species 19: C4H7 */

    double C4H7_num = epsilon -qf_co[123] -qf_co[129] -qf_co[133];
    double C4H7_denom = epsilon -qf_co[121] -qf_co[122] -qf_co[124] -qf_co[125] -qf_co[126] -qf_co[127] -qf_co[128];

    sc_qss[19] = C4H7_num/C4H7_denom;

    /* QSS coupling between C3H2  C3H3*/
    /*QSS species 14: C3H2 */

    double C3H2_num = epsilon ;
    double C3H2_denom = epsilon -qf_co[89] -qf_co[90] -qf_co[95];
    double C3H2_rhs = C3H2_num/C3H2_denom;

    double C3H2_C3H3 = (epsilon +qf_co[93] +qf_co[94])/C3H2_denom;

    /*QSS species 15: C3H3 */

    double C3H3_num = epsilon -qf_co[96] -qf_co[97];
    double C3H3_denom = epsilon -qf_co[91] -qf_co[92] -qf_co[93] -qf_co[94];
    double C3H3_rhs = C3H3_num/C3H3_denom;

    double C3H3_C3H2 = (epsilon +qf_co[95])/C3H3_denom;

    H_0 = C3H3_C3H2;
    sc_qss[15] = (C3H3_rhs -C3H2_rhs*H_0)/(1 -C3H2_C3H3*H_0);

    sc_qss[14] = C3H2_rhs -(C3H2_C3H3*sc_qss[15]);

    /*QSS species 7: CH3O2H */

    double CH3O2H_num = epsilon ;
    double CH3O2H_denom = epsilon -qf_co[53];

    sc_qss[7] = CH3O2H_num/CH3O2H_denom;

    /*QSS species 9: C2H3 */

    double C2H3_num = epsilon -qf_co[61] -qf_co[63] -qf_co[65] -qf_co[116] -qf_co[120];
    double C2H3_denom = epsilon -qf_co[2] -qf_co[56] -qf_co[57] -qf_co[58] -qf_co[59] -qf_co[60] -qf_co[66] -qf_co[117];

    sc_qss[9] = C2H3_num/C2H3_denom;

    /* QSS coupling between CH  CH2  CH2GSG  HCCO*/
    /*QSS species 0: CH */

    double CH_num = epsilon ;
    double CH_denom = epsilon -qf_co[11] -qf_co[17] -qf_co[19];
    double CH_rhs = CH_num/CH_denom;

    double CH_CH2 = (epsilon +qf_co[16] +qf_co[18])/CH_denom;
    double CH_CH2GSG = (epsilon +qf_co[21])/CH_denom;

    /*QSS species 2: CH2 */

    double CH2_num = epsilon -qf_co[29] -qf_co[32] -qf_co[54] -qf_co[81];
    double CH2_denom = epsilon -qf_co[0] -qf_co[6] -qf_co[12] -qf_co[13] -qf_co[14] -qf_co[15] -qf_co[16]
                        -qf_co[18] -qf_co[20] -qf_co[30] -qf_co[33];
    double CH2_rhs = CH2_num/CH2_denom;

    double CH2_CH = (epsilon +qf_co[17] +qf_co[19])/CH2_denom;
    double CH2_CH2GSG = (epsilon +qf_co[5])/CH2_denom;

    /*QSS species 3: CH2GSG */

    double CH2GSG_num = epsilon -qf_co[23] -qf_co[26];
    double CH2GSG_denom = epsilon -qf_co[5] -qf_co[21] -qf_co[22] -qf_co[24] -qf_co[25] -qf_co[27] -qf_co[48]
                        -qf_co[77];
    double CH2GSG_rhs = CH2GSG_num/CH2GSG_denom;

    double CH2GSG_CH2 = (epsilon +qf_co[6])/CH2GSG_denom;
    double CH2GSG_HCCO = (epsilon +qf_co[76])/CH2GSG_denom;

    /*QSS species 8: HCCO */

    double HCCO_num = epsilon -qf_co[55] -qf_co[78] -qf_co[79] -qf_co[82];
    double HCCO_denom = epsilon -qf_co[73] -qf_co[74] -qf_co[75] -qf_co[76] -qf_co[80];
    double HCCO_rhs = HCCO_num/HCCO_denom;

    double HCCO_CH2GSG = (epsilon +qf_co[77])/HCCO_denom;

    H_0 = CH2_CH;
    H_1 = CH2GSG_CH2/(1 -CH_CH2*H_0);
    H_2 = HCCO_CH2GSG/(1 -(CH2_CH2GSG -CH_CH2GSG*H_0)*H_1);
    sc_qss[8] = (HCCO_rhs -(CH2GSG_rhs -(CH2_rhs -CH_rhs*H_0)*H_1)*H_2)/(1
                        -CH2GSG_HCCO*H_2);

    sc_qss[3] = ((CH2GSG_rhs -(CH2_rhs -CH_rhs*H_0)*H_1) -(CH2GSG_HCCO*sc_qss[8]))/(1
                        -(CH2_CH2GSG -CH_CH2GSG*H_0)*H_1);

    sc_qss[2] = ((CH2_rhs -CH_rhs*H_0) -( +(CH2_CH2GSG
                        -CH_CH2GSG*H_0)*sc_qss[3]))/(1 -CH_CH2*H_0);

    sc_qss[0] = CH_rhs -( +CH_CH2GSG*sc_qss[3] +CH_CH2*sc_qss[2]);

    /*QSS species 4: CH3O */

    double CH3O_num = epsilon -qf_co[28] -qf_co[47];
    double CH3O_denom = epsilon -qf_co[1] -qf_co[44] -qf_co[45] -qf_co[46] -qf_co[163];

    sc_qss[4] = CH3O_num/CH3O_denom;

    /*QSS species 10: CH2CHO */

    double CH2CHO_num = epsilon -qf_co[64] -qf_co[85];
    double CH2CHO_denom = epsilon -qf_co[83] -qf_co[84];

    sc_qss[10] = CH2CHO_num/CH2CHO_denom;

    /*QSS species 26: C7H15X2 */

    double C7H15X2_num = epsilon -qf_co[159] -qf_co[161] -qf_co[164] -qf_co[165] -qf_co[166];
    double C7H15X2_denom = epsilon -qf_co[152] -qf_co[153] -qf_co[154] -qf_co[155] -qf_co[156] -qf_co[157];

    sc_qss[26] = C7H15X2_num/C7H15X2_denom;

    /*QSS species 25: C6H12X1 */

    double C6H12X1_num = epsilon ;
    double C6H12X1_denom = epsilon -qf_co[150] -qf_co[151];

    sc_qss[25] = C6H12X1_num/C6H12X1_denom;

    /* QSS coupling between PXC4H9  PXC4H9O2*/
    /*QSS species 20: PXC4H9 */

    double PXC4H9_num = epsilon -qf_co[137];
    double PXC4H9_denom = epsilon -qf_co[136] -qf_co[138] -qf_co[140];
    double PXC4H9_rhs = PXC4H9_num/PXC4H9_denom;

    double PXC4H9_PXC4H9O2 = (epsilon +qf_co[139])/PXC4H9_denom;

    /*QSS species 21: PXC4H9O2 */

    double PXC4H9O2_num = epsilon ;
    double PXC4H9O2_denom = epsilon -qf_co[139];
    double PXC4H9O2_rhs = PXC4H9O2_num/PXC4H9O2_denom;

    double PXC4H9O2_PXC4H9 = (epsilon +qf_co[140])/PXC4H9O2_denom;

    H_0 = PXC4H9O2_PXC4H9;
    sc_qss[21] = (PXC4H9O2_rhs -PXC4H9_rhs*H_0)/(1 -PXC4H9_PXC4H9O2*H_0);

    sc_qss[20] = PXC4H9_rhs -(PXC4H9_PXC4H9O2*sc_qss[21]);

    /* QSS coupling between C5H10X1  C5H11X1*/
    /*QSS species 23: C5H10X1 */

    double C5H10X1_num = epsilon ;
    double C5H10X1_denom = epsilon -qf_co[143] -qf_co[144] -qf_co[145] -qf_co[146] -qr_co[149];
    double C5H10X1_rhs = C5H10X1_num/C5H10X1_denom;

    double C5H10X1_C5H11X1 = (epsilon +qf_co[149])/C5H10X1_denom;

    /*QSS species 24: C5H11X1 */

    double C5H11X1_num = epsilon ;
    double C5H11X1_denom = epsilon -qf_co[147] -qf_co[148] -qf_co[149];
    double C5H11X1_rhs = C5H11X1_num/C5H11X1_denom;

    double C5H11X1_C5H10X1 = (epsilon +qr_co[149])/C5H11X1_denom;

    H_0 = C5H11X1_C5H10X1;
    sc_qss[24] = (C5H11X1_rhs -C5H10X1_rhs*H_0)/(1 -C5H10X1_C5H11X1*H_0);

    sc_qss[23] = C5H10X1_rhs -(C5H10X1_C5H11X1*sc_qss[24]);

    /*QSS species 22: C5H9 */

    double C5H9_num = epsilon ;
    double C5H9_denom = epsilon -qf_co[141] -qf_co[142];

    sc_qss[22] = C5H9_num/C5H9_denom;

    /* QSS coupling between C2H5  C2H5O2*/
    /*QSS species 5: C2H5 */

    double C2H5_num = epsilon -qf_co[3] -qf_co[31] -qf_co[70] -qf_co[71] -qf_co[72] -qf_co[119];
    double C2H5_denom = epsilon -qr_co[3] -qr_co[31] -qf_co[67] -qf_co[68] -qf_co[69] -qf_co[87];
    double C2H5_rhs = C2H5_num/C2H5_denom;

    double C2H5_C2H5O2 = (epsilon +qf_co[86])/C2H5_denom;

    /*QSS species 13: C2H5O2 */

    double C2H5O2_num = epsilon ;
    double C2H5O2_denom = epsilon -qf_co[86] -qf_co[88];
    double C2H5O2_rhs = C2H5O2_num/C2H5O2_denom;

    double C2H5O2_C2H5 = (epsilon +qf_co[87])/C2H5O2_denom;

    H_0 = C2H5O2_C2H5;
    sc_qss[13] = (C2H5O2_rhs -C2H5_rhs*H_0)/(1 -C2H5_C2H5O2*H_0);

    sc_qss[5] = C2H5_rhs -(C2H5_C2H5O2*sc_qss[13]);

    /* QSS coupling between NXC3H7  NXC3H7O2*/
    /*QSS species 17: NXC3H7 */

    double NXC3H7_num = epsilon -qf_co[111] -qf_co[113] -qf_co[130];
    double NXC3H7_denom = epsilon -qf_co[109] -qf_co[110] -qf_co[112] -qf_co[115];
    double NXC3H7_rhs = NXC3H7_num/NXC3H7_denom;

    double NXC3H7_NXC3H7O2 = (epsilon +qf_co[114])/NXC3H7_denom;

    /*QSS species 18: NXC3H7O2 */

    double NXC3H7O2_num = epsilon ;
    double NXC3H7O2_denom = epsilon -qf_co[114];
    double NXC3H7O2_rhs = NXC3H7O2_num/NXC3H7O2_denom;

    double NXC3H7O2_NXC3H7 = (epsilon +qf_co[115])/NXC3H7O2_denom;

    H_0 = NXC3H7O2_NXC3H7;
    sc_qss[18] = (NXC3H7O2_rhs -NXC3H7_rhs*H_0)/(1 -NXC3H7_NXC3H7O2*H_0);

    sc_qss[17] = NXC3H7_rhs -(NXC3H7_NXC3H7O2*sc_qss[18]);

    /*QSS species 11: C2H5O */

    double C2H5O_num = epsilon ;
    double C2H5O_denom = epsilon -qf_co[10];

    sc_qss[11] = C2H5O_num/C2H5O_denom;

    /*QSS species 16: C3H5XA */

    double C3H5XA_num = epsilon -qf_co[103] -qf_co[105] -qf_co[107] -qf_co[108] -qf_co[118] -qf_co[134];
    double C3H5XA_denom = epsilon -qf_co[98] -qf_co[99] -qf_co[100] -qf_co[101] -qf_co[102] -qf_co[104] -qf_co[135];

    sc_qss[16] = C3H5XA_num/C3H5XA_denom;

    /*QSS species 1: HCO */

    double HCO_num = epsilon -qf_co[40] -qf_co[41] -qf_co[42] -qf_co[43] -qf_co[49] -qf_co[62];
    double HCO_denom = epsilon -qf_co[7] -qf_co[34] -qf_co[35] -qf_co[36] -qf_co[37] -qf_co[38] -qf_co[39] -qf_co[50];

    sc_qss[1] = HCO_num/HCO_denom;


    return;
}

/*compute an approx to the reaction Jacobian (for preconditioning) */
AMREX_GPU_HOST_DEVICE void DWDOT_SIMPLIFIED(double *  J, double *  sc, double *  Tp, int * HP)
{
    double c[25];

    for (int k=0; k<25; k++) {
        c[k] = 1.e6 * sc[k];
    }

    aJacobian_precond(J, c, *Tp, *HP);

    /* dwdot[k]/dT */
    /* dTdot/d[X] */
    for (int k=0; k<25; k++) {
        J[650+k] *= 1.e-6;
        J[k*26+25] *= 1.e6;
    }

    return;
}

/*compute the reaction Jacobian */
AMREX_GPU_HOST_DEVICE void DWDOT(double *  J, double *  sc, double *  Tp, int * consP)
{
    double c[25];

    for (int k=0; k<25; k++) {
        c[k] = 1.e6 * sc[k];
    }

    aJacobian(J, c, *Tp, *consP);

    /* dwdot[k]/dT */
    /* dTdot/d[X] */
    for (int k=0; k<25; k++) {
        J[650+k] *= 1.e-6;
        J[k*26+25] *= 1.e6;
    }

    return;
}

/*compute the sparsity pattern of the chemistry Jacobian */
AMREX_GPU_HOST_DEVICE void SPARSITY_INFO( int * nJdata, int * consP, int NCELLS)
{
    double c[25];
    double J[676];

    for (int k=0; k<25; k++) {
        c[k] = 1.0/ 25.000000 ;
    }

    aJacobian(J, c, 1500.0, *consP);

    int nJdata_tmp = 0;
    for (int k=0; k<26; k++) {
        for (int l=0; l<26; l++) {
            if(J[ 26 * k + l] != 0.0){
                nJdata_tmp = nJdata_tmp + 1;
            }
        }
    }

    *nJdata = NCELLS * nJdata_tmp;

    return;
}



/*compute the sparsity pattern of the system Jacobian */
AMREX_GPU_HOST_DEVICE void SPARSITY_INFO_SYST( int * nJdata, int * consP, int NCELLS)
{
    double c[25];
    double J[676];

    for (int k=0; k<25; k++) {
        c[k] = 1.0/ 25.000000 ;
    }

    aJacobian(J, c, 1500.0, *consP);

    int nJdata_tmp = 0;
    for (int k=0; k<26; k++) {
        for (int l=0; l<26; l++) {
            if(k == l){
                nJdata_tmp = nJdata_tmp + 1;
            } else {
                if(J[ 26 * k + l] != 0.0){
                    nJdata_tmp = nJdata_tmp + 1;
                }
            }
        }
    }

    *nJdata = NCELLS * nJdata_tmp;

    return;
}



/*compute the sparsity pattern of the simplified (for preconditioning) system Jacobian */
AMREX_GPU_HOST_DEVICE void SPARSITY_INFO_SYST_SIMPLIFIED( int * nJdata, int * consP)
{
    double c[25];
    double J[676];

    for (int k=0; k<25; k++) {
        c[k] = 1.0/ 25.000000 ;
    }

    aJacobian_precond(J, c, 1500.0, *consP);

    int nJdata_tmp = 0;
    for (int k=0; k<26; k++) {
        for (int l=0; l<26; l++) {
            if(k == l){
                nJdata_tmp = nJdata_tmp + 1;
            } else {
                if(J[ 26 * k + l] != 0.0){
                    nJdata_tmp = nJdata_tmp + 1;
                }
            }
        }
    }

    nJdata[0] = nJdata_tmp;

    return;
}


/*compute the sparsity pattern of the chemistry Jacobian in CSC format -- base 0 */
AMREX_GPU_HOST_DEVICE void SPARSITY_PREPROC_CSC(int *  rowVals, int *  colPtrs, int * consP, int NCELLS)
{
    double c[25];
    double J[676];
    int offset_row;
    int offset_col;

    for (int k=0; k<25; k++) {
        c[k] = 1.0/ 25.000000 ;
    }

    aJacobian(J, c, 1500.0, *consP);

    colPtrs[0] = 0;
    int nJdata_tmp = 0;
    for (int nc=0; nc<NCELLS; nc++) {
        offset_row = nc * 26;
        offset_col = nc * 26;
        for (int k=0; k<26; k++) {
            for (int l=0; l<26; l++) {
                if(J[26*k + l] != 0.0) {
                    rowVals[nJdata_tmp] = l + offset_row; 
                    nJdata_tmp = nJdata_tmp + 1; 
                }
            }
            colPtrs[offset_col + (k + 1)] = nJdata_tmp;
        }
    }

    return;
}

/*compute the sparsity pattern of the chemistry Jacobian in CSR format -- base 0 */
AMREX_GPU_HOST_DEVICE void SPARSITY_PREPROC_CSR(int * colVals, int * rowPtrs, int * consP, int NCELLS, int base)
{
    double c[25];
    double J[676];
    int offset;

    for (int k=0; k<25; k++) {
        c[k] = 1.0/ 25.000000 ;
    }

    aJacobian(J, c, 1500.0, *consP);

    if (base == 1) {
        rowPtrs[0] = 1;
        int nJdata_tmp = 1;
        for (int nc=0; nc<NCELLS; nc++) {
            offset = nc * 26;
            for (int l=0; l<26; l++) {
                for (int k=0; k<26; k++) {
                    if(J[26*k + l] != 0.0) {
                        colVals[nJdata_tmp-1] = k+1 + offset; 
                        nJdata_tmp = nJdata_tmp + 1; 
                    }
                }
                rowPtrs[offset + (l + 1)] = nJdata_tmp;
            }
        }
    } else {
        rowPtrs[0] = 0;
        int nJdata_tmp = 0;
        for (int nc=0; nc<NCELLS; nc++) {
            offset = nc * 26;
            for (int l=0; l<26; l++) {
                for (int k=0; k<26; k++) {
                    if(J[26*k + l] != 0.0) {
                        colVals[nJdata_tmp] = k + offset; 
                        nJdata_tmp = nJdata_tmp + 1; 
                    }
                }
                rowPtrs[offset + (l + 1)] = nJdata_tmp;
            }
        }
    }

    return;
}

/*compute the sparsity pattern of the system Jacobian */
/*CSR format BASE is user choice */
AMREX_GPU_HOST_DEVICE void SPARSITY_PREPROC_SYST_CSR(int * colVals, int * rowPtr, int * consP, int NCELLS, int base)
{
    double c[25];
    double J[676];
    int offset;

    for (int k=0; k<25; k++) {
        c[k] = 1.0/ 25.000000 ;
    }

    aJacobian(J, c, 1500.0, *consP);

    if (base == 1) {
        rowPtr[0] = 1;
        int nJdata_tmp = 1;
        for (int nc=0; nc<NCELLS; nc++) {
            offset = nc * 26;
            for (int l=0; l<26; l++) {
                for (int k=0; k<26; k++) {
                    if (k == l) {
                        colVals[nJdata_tmp-1] = l+1 + offset; 
                        nJdata_tmp = nJdata_tmp + 1; 
                    } else {
                        if(J[26*k + l] != 0.0) {
                            colVals[nJdata_tmp-1] = k+1 + offset; 
                            nJdata_tmp = nJdata_tmp + 1; 
                        }
                    }
                }
                rowPtr[offset + (l + 1)] = nJdata_tmp;
            }
        }
    } else {
        rowPtr[0] = 0;
        int nJdata_tmp = 0;
        for (int nc=0; nc<NCELLS; nc++) {
            offset = nc * 26;
            for (int l=0; l<26; l++) {
                for (int k=0; k<26; k++) {
                    if (k == l) {
                        colVals[nJdata_tmp] = l + offset; 
                        nJdata_tmp = nJdata_tmp + 1; 
                    } else {
                        if(J[26*k + l] != 0.0) {
                            colVals[nJdata_tmp] = k + offset; 
                            nJdata_tmp = nJdata_tmp + 1; 
                        }
                    }
                }
                rowPtr[offset + (l + 1)] = nJdata_tmp;
            }
        }
    }

    return;
}

/*compute the sparsity pattern of the simplified (for precond) system Jacobian on CPU */
/*BASE 0 */
AMREX_GPU_HOST_DEVICE void SPARSITY_PREPROC_SYST_SIMPLIFIED_CSC(int * rowVals, int * colPtrs, int * indx, int * consP)
{
    double c[25];
    double J[676];

    for (int k=0; k<25; k++) {
        c[k] = 1.0/ 25.000000 ;
    }

    aJacobian_precond(J, c, 1500.0, *consP);

    colPtrs[0] = 0;
    int nJdata_tmp = 0;
    for (int k=0; k<26; k++) {
        for (int l=0; l<26; l++) {
            if (k == l) {
                rowVals[nJdata_tmp] = l; 
                indx[nJdata_tmp] = 26*k + l;
                nJdata_tmp = nJdata_tmp + 1; 
            } else {
                if(J[26*k + l] != 0.0) {
                    rowVals[nJdata_tmp] = l; 
                    indx[nJdata_tmp] = 26*k + l;
                    nJdata_tmp = nJdata_tmp + 1; 
                }
            }
        }
        colPtrs[k+1] = nJdata_tmp;
    }

    return;
}

/*compute the sparsity pattern of the simplified (for precond) system Jacobian */
/*CSR format BASE is under choice */
AMREX_GPU_HOST_DEVICE void SPARSITY_PREPROC_SYST_SIMPLIFIED_CSR(int * colVals, int * rowPtr, int * consP, int base)
{
    double c[25];
    double J[676];

    for (int k=0; k<25; k++) {
        c[k] = 1.0/ 25.000000 ;
    }

    aJacobian_precond(J, c, 1500.0, *consP);

    if (base == 1) {
        rowPtr[0] = 1;
        int nJdata_tmp = 1;
        for (int l=0; l<26; l++) {
            for (int k=0; k<26; k++) {
                if (k == l) {
                    colVals[nJdata_tmp-1] = l+1; 
                    nJdata_tmp = nJdata_tmp + 1; 
                } else {
                    if(J[26*k + l] != 0.0) {
                        colVals[nJdata_tmp-1] = k+1; 
                        nJdata_tmp = nJdata_tmp + 1; 
                    }
                }
            }
            rowPtr[l+1] = nJdata_tmp;
        }
    } else {
        rowPtr[0] = 0;
        int nJdata_tmp = 0;
        for (int l=0; l<26; l++) {
            for (int k=0; k<26; k++) {
                if (k == l) {
                    colVals[nJdata_tmp] = l; 
                    nJdata_tmp = nJdata_tmp + 1; 
                } else {
                    if(J[26*k + l] != 0.0) {
                        colVals[nJdata_tmp] = k; 
                        nJdata_tmp = nJdata_tmp + 1; 
                    }
                }
            }
            rowPtr[l+1] = nJdata_tmp;
        }
    }

    return;
}


#ifndef AMREX_USE_CUDA
/*compute the reaction Jacobian on CPU */
void aJacobian(double *  J, double *  sc, double T, int consP)
{
    for (int i=0; i<676; i++) {
        J[i] = 0.0;
    }
}
#endif


/*compute an approx to the reaction Jacobian */
AMREX_GPU_HOST_DEVICE void aJacobian_precond(double *  J, double *  sc, double T, int HP)
{
    for (int i=0; i<676; i++) {
        J[i] = 0.0;
    }
}


/*compute d(Cp/R)/dT and d(Cv/R)/dT at the given temperature */
/*tc contains precomputed powers of T, tc[0] = log(T) */
AMREX_GPU_HOST_DEVICE void dcvpRdT(double * species, double *  tc)
{

    /*temperature */
    double T = tc[1];

    /*species with midpoint at T=1000 kelvin */
    if (T < 1000) {
        /*species 0: N2 */
        species[0] =
            +1.40824000e-03
            -7.92644400e-06 * tc[1]
            +1.69245450e-08 * tc[2]
            -9.77942000e-12 * tc[3];
        /*species 1: O */
        species[1] =
            -1.63816600e-03
            +4.84206400e-06 * tc[1]
            -4.80852900e-09 * tc[2]
            +1.55627840e-12 * tc[3];
        /*species 2: H2 */
        species[2] =
            +8.24944200e-04
            -1.62860300e-06 * tc[1]
            -2.84263020e-10 * tc[2]
            +1.65394880e-12 * tc[3];
        /*species 3: H */
        species[3] =
            +0.00000000e+00
            +0.00000000e+00 * tc[1]
            +0.00000000e+00 * tc[2]
            +0.00000000e+00 * tc[3];
        /*species 5: H2O */
        species[5] =
            +3.47498200e-03
            -1.27093920e-05 * tc[1]
            +2.09057430e-08 * tc[2]
            -1.00263520e-11 * tc[3];
        /*species 6: O2 */
        species[6] =
            +1.12748600e-03
            -1.15123000e-06 * tc[1]
            +3.94163100e-09 * tc[2]
            -3.50742160e-12 * tc[3];
        /*species 8: H2O2 */
        species[8] =
            +6.56922600e-03
            -2.97002600e-07 * tc[1]
            -1.38774180e-08 * tc[2]
            +9.88606000e-12 * tc[3];
        /*species 9: CO2 */
        species[9] =
            +9.92207200e-03
            -2.08182200e-05 * tc[1]
            +2.06000610e-08 * tc[2]
            -8.46912000e-12 * tc[3];
        /*species 10: CO */
        species[10] =
            +1.51194100e-03
            -7.76351000e-06 * tc[1]
            +1.67458320e-08 * tc[2]
            -9.89980400e-12 * tc[3];
        /*species 11: CH2O */
        species[11] =
            +1.26314400e-02
            -3.77633600e-05 * tc[1]
            +6.15009300e-08 * tc[2]
            -3.36529480e-11 * tc[3];
        /*species 12: CH3 */
        species[12] =
            +1.11241000e-02
            -3.36044000e-05 * tc[1]
            +4.86548700e-08 * tc[2]
            -2.34598120e-11 * tc[3];
        /*species 13: CH4 */
        species[13] =
            +1.74766800e-02
            -5.56681800e-05 * tc[1]
            +9.14912400e-08 * tc[2]
            -4.89572400e-11 * tc[3];
        /*species 14: CH3OH */
        species[14] =
            +7.34150800e-03
            +1.43401020e-05 * tc[1]
            -2.63795820e-08 * tc[2]
            +9.56228000e-12 * tc[3];
        /*species 16: CH2CO */
        species[16] =
            +1.21187100e-02
            -4.69009200e-06 * tc[1]
            -1.94000550e-08 * tc[2]
            +1.56225960e-11 * tc[3];
        /*species 18: C2H2 */
        species[18] =
            +1.51904500e-02
            -3.23263800e-05 * tc[1]
            +2.72369760e-08 * tc[2]
            -7.65098400e-12 * tc[3];
        /*species 20: C2H4 */
        species[20] =
            +2.79616300e-02
            -6.77735400e-05 * tc[1]
            +8.35545600e-08 * tc[2]
            -3.89515160e-11 * tc[3];
    } else {
        /*species 0: N2 */
        species[0] =
            +1.48797700e-03
            -1.13695220e-06 * tc[1]
            +3.02911200e-10 * tc[2]
            -2.70134040e-14 * tc[3];
        /*species 1: O */
        species[1] =
            -2.75506200e-05
            -6.20560600e-09 * tc[1]
            +1.36532010e-11 * tc[2]
            -1.74722080e-15 * tc[3];
        /*species 2: H2 */
        species[2] =
            +7.00064400e-04
            -1.12676580e-07 * tc[1]
            -2.76947340e-11 * tc[2]
            +6.33100800e-15 * tc[3];
        /*species 3: H */
        species[3] =
            +0.00000000e+00
            +0.00000000e+00 * tc[1]
            +0.00000000e+00 * tc[2]
            +0.00000000e+00 * tc[3];
        /*species 5: H2O */
        species[5] =
            +3.05629300e-03
            -1.74605200e-06 * tc[1]
            +3.60298800e-10 * tc[2]
            -2.55664720e-14 * tc[3];
        /*species 6: O2 */
        species[6] =
            +6.13519700e-04
            -2.51768400e-07 * tc[1]
            +5.32584300e-11 * tc[2]
            -4.54574000e-15 * tc[3];
        /*species 8: H2O2 */
        species[8] =
            +4.33613600e-03
            -2.94937800e-06 * tc[1]
            +7.04671200e-10 * tc[2]
            -5.72661600e-14 * tc[3];
        /*species 9: CO2 */
        species[9] =
            +3.14016900e-03
            -2.55682200e-06 * tc[1]
            +7.18199100e-10 * tc[2]
            -6.67613200e-14 * tc[3];
        /*species 10: CO */
        species[10] =
            +1.44268900e-03
            -1.12616560e-06 * tc[1]
            +3.05574300e-10 * tc[2]
            -2.76438080e-14 * tc[3];
        /*species 11: CH2O */
        species[11] =
            +6.68132100e-03
            -5.25791000e-06 * tc[1]
            +1.42114590e-09 * tc[2]
            -1.28500680e-13 * tc[3];
        /*species 12: CH3 */
        species[12] =
            +6.13797400e-03
            -4.46069000e-06 * tc[1]
            +1.13554830e-09 * tc[2]
            -9.80863600e-14 * tc[3];
        /*species 13: CH4 */
        species[13] =
            +1.02372400e-02
            -7.75025800e-06 * tc[1]
            +2.03567550e-09 * tc[2]
            -1.80136920e-13 * tc[3];
        /*species 14: CH3OH */
        species[14] =
            +9.37659300e-03
            -6.10050800e-06 * tc[1]
            +1.30763790e-09 * tc[2]
            -8.89889200e-14 * tc[3];
        /*species 16: CH2CO */
        species[16] =
            +5.80484000e-03
            -3.84190800e-06 * tc[1]
            +8.38345500e-10 * tc[2]
            -5.83547200e-14 * tc[3];
        /*species 18: C2H2 */
        species[18] =
            +5.37603900e-03
            -3.82563400e-06 * tc[1]
            +9.85913700e-10 * tc[2]
            -8.62684000e-14 * tc[3];
        /*species 20: C2H4 */
        species[20] =
            +1.14851800e-02
            -8.83677000e-06 * tc[1]
            +2.35338030e-09 * tc[2]
            -2.10673920e-13 * tc[3];
    }

    /*species with midpoint at T=1357 kelvin */
    if (T < 1357) {
        /*species 4: OH */
        species[4] =
            +2.02235804e-04
            -2.27092824e-07 * tc[1]
            +7.27335447e-10 * tc[2]
            -2.97460412e-13 * tc[3];
    } else {
        /*species 4: OH */
        species[4] =
            +1.31992406e-03
            -7.19449340e-07 * tc[1]
            +1.27689240e-10 * tc[2]
            -7.28192064e-15 * tc[3];
    }

    /*species with midpoint at T=1390 kelvin */
    if (T < 1390) {
        /*species 7: HO2 */
        species[7] =
            +3.66767950e-03
            -1.86477024e-06 * tc[1]
            -9.77558757e-10 * tc[2]
            +6.04559648e-13 * tc[3];
    } else {
        /*species 7: HO2 */
        species[7] =
            +2.38452835e-03
            -1.61269598e-06 * tc[1]
            +3.72575169e-10 * tc[2]
            -2.86560043e-14 * tc[3];
    }

    /*species with midpoint at T=1384 kelvin */
    if (T < 1384) {
        /*species 15: C2H6 */
        species[15] =
            +2.40764754e-02
            -2.23786944e-05 * tc[1]
            +6.25022703e-09 * tc[2]
            -2.11947446e-13 * tc[3];
    } else {
        /*species 15: C2H6 */
        species[15] =
            +1.29236361e-02
            -8.85054392e-06 * tc[1]
            +2.06217518e-09 * tc[2]
            -1.59560693e-13 * tc[3];
    }

    /*species with midpoint at T=1376 kelvin */
    if (T < 1376) {
        /*species 17: HOCHO */
        species[17] =
            +1.63363016e-02
            -2.12514842e-05 * tc[1]
            +9.96398931e-09 * tc[2]
            -1.60870441e-12 * tc[3];
    } else {
        /*species 17: HOCHO */
        species[17] =
            +5.14289368e-03
            -3.64477026e-06 * tc[1]
            +8.69157489e-10 * tc[2]
            -6.83568796e-14 * tc[3];
    }

    /*species with midpoint at T=1388 kelvin */
    if (T < 1388) {
        /*species 19: C3H6 */
        species[19] =
            +2.89107662e-02
            -3.09773616e-05 * tc[1]
            +1.16644263e-08 * tc[2]
            -1.35156141e-12 * tc[3];
    } else {
        /*species 19: C3H6 */
        species[19] =
            +1.37023634e-02
            -9.32499466e-06 * tc[1]
            +2.16376321e-09 * tc[2]
            -1.66948050e-13 * tc[3];
    }

    /*species with midpoint at T=1400 kelvin */
    if (T < 1400) {
        /*species 21: C3H4XA */
        species[21] =
            +1.63343700e-02
            -3.52990000e-06 * tc[1]
            -1.39420950e-08 * tc[2]
            +6.91652400e-12 * tc[3];
    } else {
        /*species 21: C3H4XA */
        species[21] =
            +5.30213800e-03
            -7.40223600e-07 * tc[1]
            -9.07915800e-10 * tc[2]
            +2.03583240e-13 * tc[3];
    }

    /*species with midpoint at T=1398 kelvin */
    if (T < 1398) {
        /*species 22: C4H6 */
        species[22] =
            +4.78706062e-02
            -8.30893600e-05 * tc[1]
            +5.74648656e-08 * tc[2]
            -1.42863403e-11 * tc[3];
    } else {
        /*species 22: C4H6 */
        species[22] =
            +1.37163965e-02
            -9.39431566e-06 * tc[1]
            +2.18908151e-09 * tc[2]
            -1.69394481e-13 * tc[3];
    }

    /*species with midpoint at T=1392 kelvin */
    if (T < 1392) {
        /*species 23: C4H8X1 */
        species[23] =
            +4.52580978e-02
            -5.87317118e-05 * tc[1]
            +3.00661308e-08 * tc[2]
            -5.72766720e-12 * tc[3];
    } else {
        /*species 23: C4H8X1 */
        species[23] =
            +1.80617877e-02
            -1.23218606e-05 * tc[1]
            +2.86395888e-09 * tc[2]
            -2.21235856e-13 * tc[3];
    }

    /*species with midpoint at T=1391 kelvin */
    if (T < 1391) {
        /*species 24: NXC7H16 */
        species[24] =
            +8.54355820e-02
            -1.05069357e-04 * tc[1]
            +4.88837163e-08 * tc[2]
            -8.09579700e-12 * tc[3];
    } else {
        /*species 24: NXC7H16 */
        species[24] =
            +3.47675750e-02
            -2.36814258e-05 * tc[1]
            +5.49895434e-09 * tc[2]
            -4.24521064e-13 * tc[3];
    }
    return;
}


/*compute the equilibrium constants for each reaction */
void equilibriumConstants(double *  kc, double *  g_RT, double T)
{
    /*reference concentration: P_atm / (RT) in inverse mol/m^3 */
    double refC = 101325 / 8.31446 / T;
    double tc[] = { log(T), T, T*T, T*T*T, T*T*T*T }; /*temperature cache */
    double g_RT_qss[27];
    gibbs_qss(g_RT_qss, tc);

    /*reaction 1: H + O2 (+M) => HO2 (+M) */
    kc[0] = 1.0 / (refC) * exp((g_RT[3] + g_RT[6]) - (g_RT[7]));

    /*reaction 2: H2O2 (+M) => 2.000000 OH (+M) */
    kc[1] = refC * exp((g_RT[8]) - (2.000000 * g_RT[4]));

    /*reaction 3: OH + CH3 (+M) => CH3OH (+M) */
    kc[2] = 1.0 / (refC) * exp((g_RT[4] + g_RT[12]) - (g_RT[14]));

    /*reaction 4: CH3 + H (+M) => CH4 (+M) */
    kc[3] = 1.0 / (refC) * exp((g_RT[12] + g_RT[3]) - (g_RT[13]));

    /*reaction 5: 2.000000 CH3 (+M) => C2H6 (+M) */
    kc[4] = 1.0 / (refC) * exp((2.000000 * g_RT[12]) - (g_RT[15]));

    /*reaction 6: CO + CH2 (+M) => CH2CO (+M) */
    kc[5] = 1.0 / (refC) * exp((g_RT[10] + g_RT_qss[2]) - (g_RT[16]));

    /*reaction 7: CO + O (+M) => CO2 (+M) */
    kc[6] = 1.0 / (refC) * exp((g_RT[10] + g_RT[1]) - (g_RT[9]));

    /*reaction 8: CH3O (+M) => CH2O + H (+M) */
    kc[7] = refC * exp((g_RT_qss[4]) - (g_RT[11] + g_RT[3]));

    /*reaction 9: C2H3 (+M) => H + C2H2 (+M) */
    kc[8] = refC * exp((g_RT_qss[9]) - (g_RT[3] + g_RT[18]));

    /*reaction 10: H + C2H4 (+M) <=> C2H5 (+M) */
    kc[9] = 1.0 / (refC) * exp((g_RT[3] + g_RT[20]) - (g_RT_qss[5]));

    /*reaction 11: CH3CO (+M) => CH3 + CO (+M) */
    kc[10] = refC * exp((g_RT_qss[12]) - (g_RT[12] + g_RT[10]));

    /*reaction 12: H + OH + M => H2O + M */
    kc[11] = 1.0 / (refC) * exp((g_RT[3] + g_RT[4]) - (g_RT[5]));

    /*reaction 13: CH2GSG + M => CH2 + M */
    kc[12] = exp((g_RT_qss[3]) - (g_RT_qss[2]));

    /*reaction 14: CH2 + M => CH2GSG + M */
    kc[13] = exp((g_RT_qss[2]) - (g_RT_qss[3]));

    /*reaction 15: HCO + M => H + CO + M */
    kc[14] = refC * exp((g_RT_qss[1]) - (g_RT[3] + g_RT[10]));

    /*reaction 16: CH3O2 + M => CH3 + O2 + M */
    kc[15] = refC * exp((g_RT_qss[6]) - (g_RT[12] + g_RT[6]));

    /*reaction 17: CH3 + O2 + M => CH3O2 + M */
    kc[16] = 1.0 / (refC) * exp((g_RT[12] + g_RT[6]) - (g_RT_qss[6]));

    /*reaction 18: C2H5O + M => CH3 + CH2O + M */
    kc[17] = refC * exp((g_RT_qss[11]) - (g_RT[12] + g_RT[11]));

    /*reaction 19: O + H2 => H + OH */
    kc[18] = exp((g_RT[1] + g_RT[2]) - (g_RT[3] + g_RT[4]));

    /*reaction 20: H + OH => O + H2 */
    kc[19] = exp((g_RT[3] + g_RT[4]) - (g_RT[1] + g_RT[2]));

    /*reaction 21: OH + H2 => H + H2O */
    kc[20] = exp((g_RT[4] + g_RT[2]) - (g_RT[3] + g_RT[5]));

    /*reaction 22: H + H2O => OH + H2 */
    kc[21] = exp((g_RT[3] + g_RT[5]) - (g_RT[4] + g_RT[2]));

    /*reaction 23: O + H2O => 2.000000 OH */
    kc[22] = exp((g_RT[1] + g_RT[5]) - (2.000000 * g_RT[4]));

    /*reaction 24: 2.000000 OH => O + H2O */
    kc[23] = exp((2.000000 * g_RT[4]) - (g_RT[1] + g_RT[5]));

    /*reaction 25: H + O2 => O + OH */
    kc[24] = exp((g_RT[3] + g_RT[6]) - (g_RT[1] + g_RT[4]));

    /*reaction 26: O + OH => H + O2 */
    kc[25] = exp((g_RT[1] + g_RT[4]) - (g_RT[3] + g_RT[6]));

    /*reaction 27: HO2 + OH => H2O + O2 */
    kc[26] = exp((g_RT[7] + g_RT[4]) - (g_RT[5] + g_RT[6]));

    /*reaction 28: HO2 + O => OH + O2 */
    kc[27] = exp((g_RT[7] + g_RT[1]) - (g_RT[4] + g_RT[6]));

    /*reaction 29: 2.000000 HO2 => H2O2 + O2 */
    kc[28] = exp((2.000000 * g_RT[7]) - (g_RT[8] + g_RT[6]));

    /*reaction 30: HO2 + H => H2 + O2 */
    kc[29] = exp((g_RT[7] + g_RT[3]) - (g_RT[2] + g_RT[6]));

    /*reaction 31: 2.000000 HO2 => H2O2 + O2 */
    kc[30] = exp((2.000000 * g_RT[7]) - (g_RT[8] + g_RT[6]));

    /*reaction 32: HO2 + H => 2.000000 OH */
    kc[31] = exp((g_RT[7] + g_RT[3]) - (2.000000 * g_RT[4]));

    /*reaction 33: H2O2 + OH => H2O + HO2 */
    kc[32] = exp((g_RT[8] + g_RT[4]) - (g_RT[5] + g_RT[7]));

    /*reaction 34: CH + O2 => HCO + O */
    kc[33] = exp((g_RT_qss[0] + g_RT[6]) - (g_RT_qss[1] + g_RT[1]));

    /*reaction 35: CH2 + O2 => CO2 + 2.000000 H */
    kc[34] = refC * exp((g_RT_qss[2] + g_RT[6]) - (g_RT[9] + 2.000000 * g_RT[3]));

    /*reaction 36: CH2 + O2 => CO + H2O */
    kc[35] = exp((g_RT_qss[2] + g_RT[6]) - (g_RT[10] + g_RT[5]));

    /*reaction 37: CH2 + O => CO + 2.000000 H */
    kc[36] = refC * exp((g_RT_qss[2] + g_RT[1]) - (g_RT[10] + 2.000000 * g_RT[3]));

    /*reaction 38: CH2 + O2 => CH2O + O */
    kc[37] = exp((g_RT_qss[2] + g_RT[6]) - (g_RT[11] + g_RT[1]));

    /*reaction 39: CH2 + H => CH + H2 */
    kc[38] = exp((g_RT_qss[2] + g_RT[3]) - (g_RT_qss[0] + g_RT[2]));

    /*reaction 40: CH + H2 => CH2 + H */
    kc[39] = exp((g_RT_qss[0] + g_RT[2]) - (g_RT_qss[2] + g_RT[3]));

    /*reaction 41: CH2 + OH => CH + H2O */
    kc[40] = exp((g_RT_qss[2] + g_RT[4]) - (g_RT_qss[0] + g_RT[5]));

    /*reaction 42: CH + H2O => CH2 + OH */
    kc[41] = exp((g_RT_qss[0] + g_RT[5]) - (g_RT_qss[2] + g_RT[4]));

    /*reaction 43: CH2 + O2 => CO2 + H2 */
    kc[42] = exp((g_RT_qss[2] + g_RT[6]) - (g_RT[9] + g_RT[2]));

    /*reaction 44: CH2GSG + H => CH + H2 */
    kc[43] = exp((g_RT_qss[3] + g_RT[3]) - (g_RT_qss[0] + g_RT[2]));

    /*reaction 45: CH2GSG + H2 => CH3 + H */
    kc[44] = exp((g_RT_qss[3] + g_RT[2]) - (g_RT[12] + g_RT[3]));

    /*reaction 46: CH3 + H => CH2GSG + H2 */
    kc[45] = exp((g_RT[12] + g_RT[3]) - (g_RT_qss[3] + g_RT[2]));

    /*reaction 47: CH2GSG + O2 => CO + OH + H */
    kc[46] = refC * exp((g_RT_qss[3] + g_RT[6]) - (g_RT[10] + g_RT[4] + g_RT[3]));

    /*reaction 48: CH2GSG + OH => CH2O + H */
    kc[47] = exp((g_RT_qss[3] + g_RT[4]) - (g_RT[11] + g_RT[3]));

    /*reaction 49: CH3 + OH => CH2O + H2 */
    kc[48] = exp((g_RT[12] + g_RT[4]) - (g_RT[11] + g_RT[2]));

    /*reaction 50: CH3 + OH => CH2GSG + H2O */
    kc[49] = exp((g_RT[12] + g_RT[4]) - (g_RT_qss[3] + g_RT[5]));

    /*reaction 51: CH2GSG + H2O => CH3 + OH */
    kc[50] = exp((g_RT_qss[3] + g_RT[5]) - (g_RT[12] + g_RT[4]));

    /*reaction 52: CH3 + O => CH2O + H */
    kc[51] = exp((g_RT[12] + g_RT[1]) - (g_RT[11] + g_RT[3]));

    /*reaction 53: CH3 + HO2 => CH3O + OH */
    kc[52] = exp((g_RT[12] + g_RT[7]) - (g_RT_qss[4] + g_RT[4]));

    /*reaction 54: CH3 + HO2 => CH4 + O2 */
    kc[53] = exp((g_RT[12] + g_RT[7]) - (g_RT[13] + g_RT[6]));

    /*reaction 55: CH3 + O2 => CH2O + OH */
    kc[54] = exp((g_RT[12] + g_RT[6]) - (g_RT[11] + g_RT[4]));

    /*reaction 56: CH3 + H => CH2 + H2 */
    kc[55] = exp((g_RT[12] + g_RT[3]) - (g_RT_qss[2] + g_RT[2]));

    /*reaction 57: CH2 + H2 => CH3 + H */
    kc[56] = exp((g_RT_qss[2] + g_RT[2]) - (g_RT[12] + g_RT[3]));

    /*reaction 58: 2.000000 CH3 <=> H + C2H5 */
    kc[57] = exp((2.000000 * g_RT[12]) - (g_RT[3] + g_RT_qss[5]));

    /*reaction 59: CH3 + OH => CH2 + H2O */
    kc[58] = exp((g_RT[12] + g_RT[4]) - (g_RT_qss[2] + g_RT[5]));

    /*reaction 60: CH2 + H2O => CH3 + OH */
    kc[59] = exp((g_RT_qss[2] + g_RT[5]) - (g_RT[12] + g_RT[4]));

    /*reaction 61: CH4 + O => CH3 + OH */
    kc[60] = exp((g_RT[13] + g_RT[1]) - (g_RT[12] + g_RT[4]));

    /*reaction 62: CH4 + H => CH3 + H2 */
    kc[61] = exp((g_RT[13] + g_RT[3]) - (g_RT[12] + g_RT[2]));

    /*reaction 63: CH3 + H2 => CH4 + H */
    kc[62] = exp((g_RT[12] + g_RT[2]) - (g_RT[13] + g_RT[3]));

    /*reaction 64: CH4 + OH => CH3 + H2O */
    kc[63] = exp((g_RT[13] + g_RT[4]) - (g_RT[12] + g_RT[5]));

    /*reaction 65: CH3 + H2O => CH4 + OH */
    kc[64] = exp((g_RT[12] + g_RT[5]) - (g_RT[13] + g_RT[4]));

    /*reaction 66: CO + OH => CO2 + H */
    kc[65] = exp((g_RT[10] + g_RT[4]) - (g_RT[9] + g_RT[3]));

    /*reaction 67: CO2 + H => CO + OH */
    kc[66] = exp((g_RT[9] + g_RT[3]) - (g_RT[10] + g_RT[4]));

    /*reaction 68: HCO + O2 => CO + HO2 */
    kc[67] = exp((g_RT_qss[1] + g_RT[6]) - (g_RT[10] + g_RT[7]));

    /*reaction 69: HCO + O => CO2 + H */
    kc[68] = exp((g_RT_qss[1] + g_RT[1]) - (g_RT[9] + g_RT[3]));

    /*reaction 70: HCO + OH => CO + H2O */
    kc[69] = exp((g_RT_qss[1] + g_RT[4]) - (g_RT[10] + g_RT[5]));

    /*reaction 71: HCO + H => CO + H2 */
    kc[70] = exp((g_RT_qss[1] + g_RT[3]) - (g_RT[10] + g_RT[2]));

    /*reaction 72: HCO + O => CO + OH */
    kc[71] = exp((g_RT_qss[1] + g_RT[1]) - (g_RT[10] + g_RT[4]));

    /*reaction 73: HCO + CH3 => CH4 + CO */
    kc[72] = exp((g_RT_qss[1] + g_RT[12]) - (g_RT[13] + g_RT[10]));

    /*reaction 74: CH2O + OH => HCO + H2O */
    kc[73] = exp((g_RT[11] + g_RT[4]) - (g_RT_qss[1] + g_RT[5]));

    /*reaction 75: CH2O + O => HCO + OH */
    kc[74] = exp((g_RT[11] + g_RT[1]) - (g_RT_qss[1] + g_RT[4]));

    /*reaction 76: CH2O + H => HCO + H2 */
    kc[75] = exp((g_RT[11] + g_RT[3]) - (g_RT_qss[1] + g_RT[2]));

    /*reaction 77: CH2O + CH3 => HCO + CH4 */
    kc[76] = exp((g_RT[11] + g_RT[12]) - (g_RT_qss[1] + g_RT[13]));

    /*reaction 78: 2.000000 CH3O => CH3OH + CH2O */
    kc[77] = exp((2.000000 * g_RT_qss[4]) - (g_RT[14] + g_RT[11]));

    /*reaction 79: CH3O + O2 => CH2O + HO2 */
    kc[78] = exp((g_RT_qss[4] + g_RT[6]) - (g_RT[11] + g_RT[7]));

    /*reaction 80: CH3O + H2 => CH3OH + H */
    kc[79] = exp((g_RT_qss[4] + g_RT[2]) - (g_RT[14] + g_RT[3]));

    /*reaction 81: CH3OH + OH => CH3O + H2O */
    kc[80] = exp((g_RT[14] + g_RT[4]) - (g_RT_qss[4] + g_RT[5]));

    /*reaction 82: CH2GSG + CO2 => CH2O + CO */
    kc[81] = exp((g_RT_qss[3] + g_RT[9]) - (g_RT[11] + g_RT[10]));

    /*reaction 83: HOCHO + H => H2 + CO + OH */
    kc[82] = refC * exp((g_RT[17] + g_RT[3]) - (g_RT[2] + g_RT[10] + g_RT[4]));

    /*reaction 84: HOCHO + OH => H2O + CO + OH */
    kc[83] = refC * exp((g_RT[17] + g_RT[4]) - (g_RT[5] + g_RT[10] + g_RT[4]));

    /*reaction 85: HOCHO => HCO + OH */
    kc[84] = refC * exp((g_RT[17]) - (g_RT_qss[1] + g_RT[4]));

    /*reaction 86: HCO + OH => HOCHO */
    kc[85] = 1.0 / (refC) * exp((g_RT_qss[1] + g_RT[4]) - (g_RT[17]));

    /*reaction 87: HOCHO + H => H2 + CO2 + H */
    kc[86] = refC * exp((g_RT[17] + g_RT[3]) - (g_RT[2] + g_RT[9] + g_RT[3]));

    /*reaction 88: HOCHO + OH => H2O + CO2 + H */
    kc[87] = refC * exp((g_RT[17] + g_RT[4]) - (g_RT[5] + g_RT[9] + g_RT[3]));

    /*reaction 89: CH3O2 + CH3 => 2.000000 CH3O */
    kc[88] = exp((g_RT_qss[6] + g_RT[12]) - (2.000000 * g_RT_qss[4]));

    /*reaction 90: CH3O2 + HO2 => CH3O2H + O2 */
    kc[89] = exp((g_RT_qss[6] + g_RT[7]) - (g_RT_qss[7] + g_RT[6]));

    /*reaction 91: CH3O2H => CH3O + OH */
    kc[90] = refC * exp((g_RT_qss[7]) - (g_RT_qss[4] + g_RT[4]));

    /*reaction 92: C2H2 + O => CH2 + CO */
    kc[91] = exp((g_RT[18] + g_RT[1]) - (g_RT_qss[2] + g_RT[10]));

    /*reaction 93: C2H2 + O => HCCO + H */
    kc[92] = exp((g_RT[18] + g_RT[1]) - (g_RT_qss[8] + g_RT[3]));

    /*reaction 94: C2H3 + H => C2H2 + H2 */
    kc[93] = exp((g_RT_qss[9] + g_RT[3]) - (g_RT[18] + g_RT[2]));

    /*reaction 95: C2H3 + O2 => CH2CHO + O */
    kc[94] = exp((g_RT_qss[9] + g_RT[6]) - (g_RT_qss[10] + g_RT[1]));

    /*reaction 96: C2H3 + CH3 => C3H6 */
    kc[95] = 1.0 / (refC) * exp((g_RT_qss[9] + g_RT[12]) - (g_RT[19]));

    /*reaction 97: C2H3 + O2 => C2H2 + HO2 */
    kc[96] = exp((g_RT_qss[9] + g_RT[6]) - (g_RT[18] + g_RT[7]));

    /*reaction 98: C2H3 + O2 => CH2O + HCO */
    kc[97] = exp((g_RT_qss[9] + g_RT[6]) - (g_RT[11] + g_RT_qss[1]));

    /*reaction 99: C2H4 + CH3 => C2H3 + CH4 */
    kc[98] = exp((g_RT[20] + g_RT[12]) - (g_RT_qss[9] + g_RT[13]));

    /*reaction 100: C2H4 + O => CH3 + HCO */
    kc[99] = exp((g_RT[20] + g_RT[1]) - (g_RT[12] + g_RT_qss[1]));

    /*reaction 101: C2H4 + OH => C2H3 + H2O */
    kc[100] = exp((g_RT[20] + g_RT[4]) - (g_RT_qss[9] + g_RT[5]));

    /*reaction 102: C2H4 + O => CH2CHO + H */
    kc[101] = exp((g_RT[20] + g_RT[1]) - (g_RT_qss[10] + g_RT[3]));

    /*reaction 103: C2H4 + H => C2H3 + H2 */
    kc[102] = exp((g_RT[20] + g_RT[3]) - (g_RT_qss[9] + g_RT[2]));

    /*reaction 104: C2H3 + H2 => C2H4 + H */
    kc[103] = exp((g_RT_qss[9] + g_RT[2]) - (g_RT[20] + g_RT[3]));

    /*reaction 105: H + C2H5 => C2H6 */
    kc[104] = 1.0 / (refC) * exp((g_RT[3] + g_RT_qss[5]) - (g_RT[15]));

    /*reaction 106: C2H5 + HO2 => C2H5O + OH */
    kc[105] = exp((g_RT_qss[5] + g_RT[7]) - (g_RT_qss[11] + g_RT[4]));

    /*reaction 107: C2H5 + O2 => C2H4 + HO2 */
    kc[106] = exp((g_RT_qss[5] + g_RT[6]) - (g_RT[20] + g_RT[7]));

    /*reaction 108: C2H6 + O => C2H5 + OH */
    kc[107] = exp((g_RT[15] + g_RT[1]) - (g_RT_qss[5] + g_RT[4]));

    /*reaction 109: C2H6 + OH => C2H5 + H2O */
    kc[108] = exp((g_RT[15] + g_RT[4]) - (g_RT_qss[5] + g_RT[5]));

    /*reaction 110: C2H6 + H => C2H5 + H2 */
    kc[109] = exp((g_RT[15] + g_RT[3]) - (g_RT_qss[5] + g_RT[2]));

    /*reaction 111: HCCO + O => H + 2.000000 CO */
    kc[110] = refC * exp((g_RT_qss[8] + g_RT[1]) - (g_RT[3] + 2.000000 * g_RT[10]));

    /*reaction 112: HCCO + OH => 2.000000 HCO */
    kc[111] = exp((g_RT_qss[8] + g_RT[4]) - (2.000000 * g_RT_qss[1]));

    /*reaction 113: HCCO + O2 => CO2 + HCO */
    kc[112] = exp((g_RT_qss[8] + g_RT[6]) - (g_RT[9] + g_RT_qss[1]));

    /*reaction 114: HCCO + H => CH2GSG + CO */
    kc[113] = exp((g_RT_qss[8] + g_RT[3]) - (g_RT_qss[3] + g_RT[10]));

    /*reaction 115: CH2GSG + CO => HCCO + H */
    kc[114] = exp((g_RT_qss[3] + g_RT[10]) - (g_RT_qss[8] + g_RT[3]));

    /*reaction 116: CH2CO + O => HCCO + OH */
    kc[115] = exp((g_RT[16] + g_RT[1]) - (g_RT_qss[8] + g_RT[4]));

    /*reaction 117: CH2CO + H => HCCO + H2 */
    kc[116] = exp((g_RT[16] + g_RT[3]) - (g_RT_qss[8] + g_RT[2]));

    /*reaction 118: HCCO + H2 => CH2CO + H */
    kc[117] = exp((g_RT_qss[8] + g_RT[2]) - (g_RT[16] + g_RT[3]));

    /*reaction 119: CH2CO + H => CH3 + CO */
    kc[118] = exp((g_RT[16] + g_RT[3]) - (g_RT[12] + g_RT[10]));

    /*reaction 120: CH2CO + O => CH2 + CO2 */
    kc[119] = exp((g_RT[16] + g_RT[1]) - (g_RT_qss[2] + g_RT[9]));

    /*reaction 121: CH2CO + OH => HCCO + H2O */
    kc[120] = exp((g_RT[16] + g_RT[4]) - (g_RT_qss[8] + g_RT[5]));

    /*reaction 122: CH2CHO + O2 => CH2O + CO + OH */
    kc[121] = refC * exp((g_RT_qss[10] + g_RT[6]) - (g_RT[11] + g_RT[10] + g_RT[4]));

    /*reaction 123: CH2CHO => CH2CO + H */
    kc[122] = refC * exp((g_RT_qss[10]) - (g_RT[16] + g_RT[3]));

    /*reaction 124: CH2CO + H => CH2CHO */
    kc[123] = 1.0 / (refC) * exp((g_RT[16] + g_RT[3]) - (g_RT_qss[10]));

    /*reaction 125: C2H5O2 => C2H5 + O2 */
    kc[124] = refC * exp((g_RT_qss[13]) - (g_RT_qss[5] + g_RT[6]));

    /*reaction 126: C2H5 + O2 => C2H5O2 */
    kc[125] = 1.0 / (refC) * exp((g_RT_qss[5] + g_RT[6]) - (g_RT_qss[13]));

    /*reaction 127: C2H5O2 => C2H4 + HO2 */
    kc[126] = refC * exp((g_RT_qss[13]) - (g_RT[20] + g_RT[7]));

    /*reaction 128: C3H2 + O2 => HCCO + CO + H */
    kc[127] = refC * exp((g_RT_qss[14] + g_RT[6]) - (g_RT_qss[8] + g_RT[10] + g_RT[3]));

    /*reaction 129: C3H2 + OH => C2H2 + HCO */
    kc[128] = exp((g_RT_qss[14] + g_RT[4]) - (g_RT[18] + g_RT_qss[1]));

    /*reaction 130: C3H3 + O2 => CH2CO + HCO */
    kc[129] = exp((g_RT_qss[15] + g_RT[6]) - (g_RT[16] + g_RT_qss[1]));

    /*reaction 131: C3H3 + HO2 => C3H4XA + O2 */
    kc[130] = exp((g_RT_qss[15] + g_RT[7]) - (g_RT[21] + g_RT[6]));

    /*reaction 132: C3H3 + H => C3H2 + H2 */
    kc[131] = exp((g_RT_qss[15] + g_RT[3]) - (g_RT_qss[14] + g_RT[2]));

    /*reaction 133: C3H3 + OH => C3H2 + H2O */
    kc[132] = exp((g_RT_qss[15] + g_RT[4]) - (g_RT_qss[14] + g_RT[5]));

    /*reaction 134: C3H2 + H2O => C3H3 + OH */
    kc[133] = exp((g_RT_qss[14] + g_RT[5]) - (g_RT_qss[15] + g_RT[4]));

    /*reaction 135: C3H4XA + H => C3H3 + H2 */
    kc[134] = exp((g_RT[21] + g_RT[3]) - (g_RT_qss[15] + g_RT[2]));

    /*reaction 136: C3H4XA + OH => C3H3 + H2O */
    kc[135] = exp((g_RT[21] + g_RT[4]) - (g_RT_qss[15] + g_RT[5]));

    /*reaction 137: C3H4XA + O => C2H4 + CO */
    kc[136] = exp((g_RT[21] + g_RT[1]) - (g_RT[20] + g_RT[10]));

    /*reaction 138: C3H5XA + H => C3H4XA + H2 */
    kc[137] = exp((g_RT_qss[16] + g_RT[3]) - (g_RT[21] + g_RT[2]));

    /*reaction 139: C3H5XA + HO2 => C3H6 + O2 */
    kc[138] = exp((g_RT_qss[16] + g_RT[7]) - (g_RT[19] + g_RT[6]));

    /*reaction 140: C3H5XA + H => C3H6 */
    kc[139] = 1.0 / (refC) * exp((g_RT_qss[16] + g_RT[3]) - (g_RT[19]));

    /*reaction 141: C3H5XA => C2H2 + CH3 */
    kc[140] = refC * exp((g_RT_qss[16]) - (g_RT[18] + g_RT[12]));

    /*reaction 142: C3H5XA => C3H4XA + H */
    kc[141] = refC * exp((g_RT_qss[16]) - (g_RT[21] + g_RT[3]));

    /*reaction 143: C3H4XA + H => C3H5XA */
    kc[142] = 1.0 / (refC) * exp((g_RT[21] + g_RT[3]) - (g_RT_qss[16]));

    /*reaction 144: C3H5XA + CH2O => C3H6 + HCO */
    kc[143] = exp((g_RT_qss[16] + g_RT[11]) - (g_RT[19] + g_RT_qss[1]));

    /*reaction 145: C3H6 + H => C2H4 + CH3 */
    kc[144] = exp((g_RT[19] + g_RT[3]) - (g_RT[20] + g_RT[12]));

    /*reaction 146: C3H6 + H => C3H5XA + H2 */
    kc[145] = exp((g_RT[19] + g_RT[3]) - (g_RT_qss[16] + g_RT[2]));

    /*reaction 147: C3H6 + O => C2H5 + HCO */
    kc[146] = exp((g_RT[19] + g_RT[1]) - (g_RT_qss[5] + g_RT_qss[1]));

    /*reaction 148: C3H6 + O => C3H5XA + OH */
    kc[147] = exp((g_RT[19] + g_RT[1]) - (g_RT_qss[16] + g_RT[4]));

    /*reaction 149: C3H6 + O => CH2CO + CH3 + H */
    kc[148] = refC * exp((g_RT[19] + g_RT[1]) - (g_RT[16] + g_RT[12] + g_RT[3]));

    /*reaction 150: C3H6 + OH => C3H5XA + H2O */
    kc[149] = exp((g_RT[19] + g_RT[4]) - (g_RT_qss[16] + g_RT[5]));

    /*reaction 151: NXC3H7 + O2 => C3H6 + HO2 */
    kc[150] = exp((g_RT_qss[17] + g_RT[6]) - (g_RT[19] + g_RT[7]));

    /*reaction 152: NXC3H7 => CH3 + C2H4 */
    kc[151] = refC * exp((g_RT_qss[17]) - (g_RT[12] + g_RT[20]));

    /*reaction 153: CH3 + C2H4 => NXC3H7 */
    kc[152] = 1.0 / (refC) * exp((g_RT[12] + g_RT[20]) - (g_RT_qss[17]));

    /*reaction 154: NXC3H7 => H + C3H6 */
    kc[153] = refC * exp((g_RT_qss[17]) - (g_RT[3] + g_RT[19]));

    /*reaction 155: H + C3H6 => NXC3H7 */
    kc[154] = 1.0 / (refC) * exp((g_RT[3] + g_RT[19]) - (g_RT_qss[17]));

    /*reaction 156: NXC3H7O2 => NXC3H7 + O2 */
    kc[155] = refC * exp((g_RT_qss[18]) - (g_RT_qss[17] + g_RT[6]));

    /*reaction 157: NXC3H7 + O2 => NXC3H7O2 */
    kc[156] = 1.0 / (refC) * exp((g_RT_qss[17] + g_RT[6]) - (g_RT_qss[18]));

    /*reaction 158: C4H6 => 2.000000 C2H3 */
    kc[157] = refC * exp((g_RT[22]) - (2.000000 * g_RT_qss[9]));

    /*reaction 159: 2.000000 C2H3 => C4H6 */
    kc[158] = 1.0 / (refC) * exp((2.000000 * g_RT_qss[9]) - (g_RT[22]));

    /*reaction 160: C4H6 + OH => CH2O + C3H5XA */
    kc[159] = exp((g_RT[22] + g_RT[4]) - (g_RT[11] + g_RT_qss[16]));

    /*reaction 161: C4H6 + OH => C2H5 + CH2CO */
    kc[160] = exp((g_RT[22] + g_RT[4]) - (g_RT_qss[5] + g_RT[16]));

    /*reaction 162: C4H6 + O => C2H4 + CH2CO */
    kc[161] = exp((g_RT[22] + g_RT[1]) - (g_RT[20] + g_RT[16]));

    /*reaction 163: C4H6 + H => C2H3 + C2H4 */
    kc[162] = exp((g_RT[22] + g_RT[3]) - (g_RT_qss[9] + g_RT[20]));

    /*reaction 164: C4H6 + O => CH2O + C3H4XA */
    kc[163] = exp((g_RT[22] + g_RT[1]) - (g_RT[11] + g_RT[21]));

    /*reaction 165: H + C4H7 => C4H8X1 */
    kc[164] = 1.0 / (refC) * exp((g_RT[3] + g_RT_qss[19]) - (g_RT[23]));

    /*reaction 166: C4H7 => C4H6 + H */
    kc[165] = refC * exp((g_RT_qss[19]) - (g_RT[22] + g_RT[3]));

    /*reaction 167: C4H6 + H => C4H7 */
    kc[166] = 1.0 / (refC) * exp((g_RT[22] + g_RT[3]) - (g_RT_qss[19]));

    /*reaction 168: C4H7 + CH3 => C4H6 + CH4 */
    kc[167] = exp((g_RT_qss[19] + g_RT[12]) - (g_RT[22] + g_RT[13]));

    /*reaction 169: C4H7 + HO2 => C4H8X1 + O2 */
    kc[168] = exp((g_RT_qss[19] + g_RT[7]) - (g_RT[23] + g_RT[6]));

    /*reaction 170: C4H7 + O2 => C4H6 + HO2 */
    kc[169] = exp((g_RT_qss[19] + g_RT[6]) - (g_RT[22] + g_RT[7]));

    /*reaction 171: C4H7 => C2H4 + C2H3 */
    kc[170] = refC * exp((g_RT_qss[19]) - (g_RT[20] + g_RT_qss[9]));

    /*reaction 172: H + C4H7 => C4H6 + H2 */
    kc[171] = exp((g_RT[3] + g_RT_qss[19]) - (g_RT[22] + g_RT[2]));

    /*reaction 173: C4H8X1 + H => C4H7 + H2 */
    kc[172] = exp((g_RT[23] + g_RT[3]) - (g_RT_qss[19] + g_RT[2]));

    /*reaction 174: C4H8X1 + OH => NXC3H7 + CH2O */
    kc[173] = exp((g_RT[23] + g_RT[4]) - (g_RT_qss[17] + g_RT[11]));

    /*reaction 175: C4H8X1 + OH => CH3CO + C2H6 */
    kc[174] = exp((g_RT[23] + g_RT[4]) - (g_RT_qss[12] + g_RT[15]));

    /*reaction 176: C4H8X1 + O => CH3CO + C2H5 */
    kc[175] = exp((g_RT[23] + g_RT[1]) - (g_RT_qss[12] + g_RT_qss[5]));

    /*reaction 177: C4H8X1 + O => C3H6 + CH2O */
    kc[176] = exp((g_RT[23] + g_RT[1]) - (g_RT[19] + g_RT[11]));

    /*reaction 178: C4H8X1 + OH => C4H7 + H2O */
    kc[177] = exp((g_RT[23] + g_RT[4]) - (g_RT_qss[19] + g_RT[5]));

    /*reaction 179: C4H8X1 => C3H5XA + CH3 */
    kc[178] = refC * exp((g_RT[23]) - (g_RT_qss[16] + g_RT[12]));

    /*reaction 180: C3H5XA + CH3 => C4H8X1 */
    kc[179] = 1.0 / (refC) * exp((g_RT_qss[16] + g_RT[12]) - (g_RT[23]));

    /*reaction 181: PXC4H9 => C4H8X1 + H */
    kc[180] = refC * exp((g_RT_qss[20]) - (g_RT[23] + g_RT[3]));

    /*reaction 182: C4H8X1 + H => PXC4H9 */
    kc[181] = 1.0 / (refC) * exp((g_RT[23] + g_RT[3]) - (g_RT_qss[20]));

    /*reaction 183: PXC4H9 => C2H5 + C2H4 */
    kc[182] = refC * exp((g_RT_qss[20]) - (g_RT_qss[5] + g_RT[20]));

    /*reaction 184: PXC4H9O2 => PXC4H9 + O2 */
    kc[183] = refC * exp((g_RT_qss[21]) - (g_RT_qss[20] + g_RT[6]));

    /*reaction 185: PXC4H9 + O2 => PXC4H9O2 */
    kc[184] = 1.0 / (refC) * exp((g_RT_qss[20] + g_RT[6]) - (g_RT_qss[21]));

    /*reaction 186: C5H9 => C4H6 + CH3 */
    kc[185] = refC * exp((g_RT_qss[22]) - (g_RT[22] + g_RT[12]));

    /*reaction 187: C5H9 => C3H5XA + C2H4 */
    kc[186] = refC * exp((g_RT_qss[22]) - (g_RT_qss[16] + g_RT[20]));

    /*reaction 188: C5H10X1 + OH => C5H9 + H2O */
    kc[187] = exp((g_RT_qss[23] + g_RT[4]) - (g_RT_qss[22] + g_RT[5]));

    /*reaction 189: C5H10X1 + H => C5H9 + H2 */
    kc[188] = exp((g_RT_qss[23] + g_RT[3]) - (g_RT_qss[22] + g_RT[2]));

    /*reaction 190: C5H10X1 => C2H5 + C3H5XA */
    kc[189] = refC * exp((g_RT_qss[23]) - (g_RT_qss[5] + g_RT_qss[16]));

    /*reaction 191: C5H10X1 + O => C5H9 + OH */
    kc[190] = exp((g_RT_qss[23] + g_RT[1]) - (g_RT_qss[22] + g_RT[4]));

    /*reaction 192: C5H11X1 => C3H6 + C2H5 */
    kc[191] = refC * exp((g_RT_qss[24]) - (g_RT[19] + g_RT_qss[5]));

    /*reaction 193: C5H11X1 => C2H4 + NXC3H7 */
    kc[192] = refC * exp((g_RT_qss[24]) - (g_RT[20] + g_RT_qss[17]));

    /*reaction 194: C5H11X1 <=> C5H10X1 + H */
    kc[193] = refC * exp((g_RT_qss[24]) - (g_RT_qss[23] + g_RT[3]));

    /*reaction 195: C6H12X1 => NXC3H7 + C3H5XA */
    kc[194] = refC * exp((g_RT_qss[25]) - (g_RT_qss[17] + g_RT_qss[16]));

    /*reaction 196: C6H12X1 + OH => C5H11X1 + CH2O */
    kc[195] = exp((g_RT_qss[25] + g_RT[4]) - (g_RT_qss[24] + g_RT[11]));

    /*reaction 197: C7H15X2 => C6H12X1 + CH3 */
    kc[196] = refC * exp((g_RT_qss[26]) - (g_RT_qss[25] + g_RT[12]));

    /*reaction 198: C7H15X2 => PXC4H9 + C3H6 */
    kc[197] = refC * exp((g_RT_qss[26]) - (g_RT_qss[20] + g_RT[19]));

    /*reaction 199: C7H15X2 => C4H8X1 + NXC3H7 */
    kc[198] = refC * exp((g_RT_qss[26]) - (g_RT[23] + g_RT_qss[17]));

    /*reaction 200: C7H15X2 => C5H11X1 + C2H4 */
    kc[199] = refC * exp((g_RT_qss[26]) - (g_RT_qss[24] + g_RT[20]));

    /*reaction 201: C7H15X2 => C2H5 + C5H10X1 */
    kc[200] = refC * exp((g_RT_qss[26]) - (g_RT_qss[5] + g_RT_qss[23]));

    /*reaction 202: C7H15X2 + HO2 => NXC7H16 + O2 */
    kc[201] = exp((g_RT_qss[26] + g_RT[7]) - (g_RT[24] + g_RT[6]));

    /*reaction 203: NXC7H16 + CH3O2 => C7H15X2 + CH3O2H */
    kc[202] = exp((g_RT[24] + g_RT_qss[6]) - (g_RT_qss[26] + g_RT_qss[7]));

    /*reaction 204: NXC7H16 + H => C7H15X2 + H2 */
    kc[203] = exp((g_RT[24] + g_RT[3]) - (g_RT_qss[26] + g_RT[2]));

    /*reaction 205: NXC7H16 => PXC4H9 + NXC3H7 */
    kc[204] = refC * exp((g_RT[24]) - (g_RT_qss[20] + g_RT_qss[17]));

    /*reaction 206: NXC7H16 + HO2 => C7H15X2 + H2O2 */
    kc[205] = exp((g_RT[24] + g_RT[7]) - (g_RT_qss[26] + g_RT[8]));

    /*reaction 207: NXC7H16 => C5H11X1 + C2H5 */
    kc[206] = refC * exp((g_RT[24]) - (g_RT_qss[24] + g_RT_qss[5]));

    /*reaction 208: NXC7H16 + CH3O => C7H15X2 + CH3OH */
    kc[207] = exp((g_RT[24] + g_RT_qss[4]) - (g_RT_qss[26] + g_RT[14]));

    /*reaction 209: NXC7H16 + O => C7H15X2 + OH */
    kc[208] = exp((g_RT[24] + g_RT[1]) - (g_RT_qss[26] + g_RT[4]));

    /*reaction 210: NXC7H16 + OH => C7H15X2 + H2O */
    kc[209] = exp((g_RT[24] + g_RT[4]) - (g_RT_qss[26] + g_RT[5]));

    /*reaction 211: NXC7H16 + CH3 => C7H15X2 + CH4 */
    kc[210] = exp((g_RT[24] + g_RT[12]) - (g_RT_qss[26] + g_RT[13]));

    return;
}


/*compute the g/(RT) at the given temperature */
/*tc contains precomputed powers of T, tc[0] = log(T) */
AMREX_GPU_HOST_DEVICE void gibbs(double * species, double *  tc)
{

    /*temperature */
    double T = tc[1];
    double invT = 1 / T;

    /*species with midpoint at T=1000 kelvin */
    if (T < 1000) {
        /*species 0: N2 */
        species[0] =
            -1.020900000000000e+03 * invT
            -6.516950000000001e-01
            -3.298677000000000e+00 * tc[0]
            -7.041200000000000e-04 * tc[1]
            +6.605369999999999e-07 * tc[2]
            -4.701262500000001e-10 * tc[3]
            +1.222427500000000e-13 * tc[4];
        /*species 1: O */
        species[1] =
            +2.914764000000000e+04 * invT
            -1.756599999999997e-02
            -2.946429000000000e+00 * tc[0]
            +8.190830000000000e-04 * tc[1]
            -4.035053333333333e-07 * tc[2]
            +1.335702500000000e-10 * tc[3]
            -1.945348000000000e-14 * tc[4];
        /*species 2: H2 */
        species[2] =
            -1.012521000000000e+03 * invT
            +6.592218000000000e+00
            -3.298124000000000e+00 * tc[0]
            -4.124721000000000e-04 * tc[1]
            +1.357169166666667e-07 * tc[2]
            +7.896194999999999e-12 * tc[3]
            -2.067436000000000e-14 * tc[4];
        /*species 3: H */
        species[3] =
            +2.547474660000000e+04 * invT
            +2.966385537000000e+00
            -2.501044220000000e+00 * tc[0]
            -0.000000000000000e+00 * tc[1]
            -0.000000000000000e+00 * tc[2]
            -0.000000000000000e+00 * tc[3]
            -0.000000000000000e+00 * tc[4];
        /*species 5: H2O */
        species[5] =
            -3.020811000000000e+04 * invT
            +7.966090000000001e-01
            -3.386842000000000e+00 * tc[0]
            -1.737491000000000e-03 * tc[1]
            +1.059116000000000e-06 * tc[2]
            -5.807150833333333e-10 * tc[3]
            +1.253294000000000e-13 * tc[4];
        /*species 6: O2 */
        species[6] =
            -1.005249000000000e+03 * invT
            -2.821802000000000e+00
            -3.212936000000000e+00 * tc[0]
            -5.637430000000000e-04 * tc[1]
            +9.593583333333333e-08 * tc[2]
            -1.094897500000000e-10 * tc[3]
            +4.384277000000000e-14 * tc[4];
        /*species 8: H2O2 */
        species[8] =
            -1.766315000000000e+04 * invT
            -3.396609000000000e+00
            -3.388754000000000e+00 * tc[0]
            -3.284613000000000e-03 * tc[1]
            +2.475021666666666e-08 * tc[2]
            +3.854838333333333e-10 * tc[3]
            -1.235757500000000e-13 * tc[4];
        /*species 9: CO2 */
        species[9] =
            -4.837314000000000e+04 * invT
            -7.912765000000000e+00
            -2.275725000000000e+00 * tc[0]
            -4.961036000000000e-03 * tc[1]
            +1.734851666666667e-06 * tc[2]
            -5.722239166666667e-10 * tc[3]
            +1.058640000000000e-13 * tc[4];
        /*species 10: CO */
        species[10] =
            -1.431054000000000e+04 * invT
            -1.586445000000000e+00
            -3.262452000000000e+00 * tc[0]
            -7.559705000000000e-04 * tc[1]
            +6.469591666666667e-07 * tc[2]
            -4.651620000000000e-10 * tc[3]
            +1.237475500000000e-13 * tc[4];
        /*species 11: CH2O */
        species[11] =
            -1.486540000000000e+04 * invT
            -1.213208900000000e+01
            -1.652731000000000e+00 * tc[0]
            -6.315720000000000e-03 * tc[1]
            +3.146946666666667e-06 * tc[2]
            -1.708359166666667e-09 * tc[3]
            +4.206618500000000e-13 * tc[4];
        /*species 12: CH3 */
        species[12] =
            +1.642378000000000e+04 * invT
            -4.359351000000000e+00
            -2.430443000000000e+00 * tc[0]
            -5.562050000000000e-03 * tc[1]
            +2.800366666666666e-06 * tc[2]
            -1.351524166666667e-09 * tc[3]
            +2.932476500000000e-13 * tc[4];
        /*species 13: CH4 */
        species[13] =
            -9.825228999999999e+03 * invT
            -1.294344850000000e+01
            -7.787415000000000e-01 * tc[0]
            -8.738340000000001e-03 * tc[1]
            +4.639015000000000e-06 * tc[2]
            -2.541423333333333e-09 * tc[3]
            +6.119655000000000e-13 * tc[4];
        /*species 14: CH3OH */
        species[14] =
            -2.535348000000000e+04 * invT
            -8.572515000000001e+00
            -2.660115000000000e+00 * tc[0]
            -3.670754000000000e-03 * tc[1]
            -1.195008500000000e-06 * tc[2]
            +7.327661666666667e-10 * tc[3]
            -1.195285000000000e-13 * tc[4];
        /*species 16: CH2CO */
        species[16] =
            -7.632637000000000e+03 * invT
            -5.698582000000000e+00
            -2.974971000000000e+00 * tc[0]
            -6.059355000000000e-03 * tc[1]
            +3.908410000000000e-07 * tc[2]
            +5.388904166666666e-10 * tc[3]
            -1.952824500000000e-13 * tc[4];
        /*species 18: C2H2 */
        species[18] =
            +2.612444000000000e+04 * invT
            -6.791815999999999e+00
            -2.013562000000000e+00 * tc[0]
            -7.595225000000000e-03 * tc[1]
            +2.693865000000000e-06 * tc[2]
            -7.565826666666667e-10 * tc[3]
            +9.563730000000000e-14 * tc[4];
        /*species 20: C2H4 */
        species[20] =
            +5.573046000000000e+03 * invT
            -2.507297800000000e+01
            +8.614880000000000e-01 * tc[0]
            -1.398081500000000e-02 * tc[1]
            +5.647795000000000e-06 * tc[2]
            -2.320960000000000e-09 * tc[3]
            +4.868939500000000e-13 * tc[4];
    } else {
        /*species 0: N2 */
        species[0] =
            -9.227977000000000e+02 * invT
            -3.053888000000000e+00
            -2.926640000000000e+00 * tc[0]
            -7.439885000000000e-04 * tc[1]
            +9.474601666666666e-08 * tc[2]
            -8.414199999999999e-12 * tc[3]
            +3.376675500000000e-16 * tc[4];
        /*species 1: O */
        species[1] =
            +2.923080000000000e+04 * invT
            -2.378248000000000e+00
            -2.542060000000000e+00 * tc[0]
            +1.377531000000000e-05 * tc[1]
            +5.171338333333333e-10 * tc[2]
            -3.792555833333334e-13 * tc[3]
            +2.184026000000000e-17 * tc[4];
        /*species 2: H2 */
        species[2] =
            -8.350340000000000e+02 * invT
            +4.346533000000000e+00
            -2.991423000000000e+00 * tc[0]
            -3.500322000000000e-04 * tc[1]
            +9.389715000000000e-09 * tc[2]
            +7.692981666666667e-13 * tc[3]
            -7.913760000000000e-17 * tc[4];
        /*species 3: H */
        species[3] =
            +2.547474660000000e+04 * invT
            +2.966385537000000e+00
            -2.501044220000000e+00 * tc[0]
            -0.000000000000000e+00 * tc[1]
            -0.000000000000000e+00 * tc[2]
            -0.000000000000000e+00 * tc[3]
            -0.000000000000000e+00 * tc[4];
        /*species 5: H2O */
        species[5] =
            -2.989921000000000e+04 * invT
            -4.190671000000000e+00
            -2.672146000000000e+00 * tc[0]
            -1.528146500000000e-03 * tc[1]
            +1.455043333333333e-07 * tc[2]
            -1.000830000000000e-11 * tc[3]
            +3.195809000000000e-16 * tc[4];
        /*species 6: O2 */
        species[6] =
            -1.233930000000000e+03 * invT
            +5.084119999999999e-01
            -3.697578000000000e+00 * tc[0]
            -3.067598500000000e-04 * tc[1]
            +2.098070000000000e-08 * tc[2]
            -1.479400833333333e-12 * tc[3]
            +5.682175000000001e-17 * tc[4];
        /*species 8: H2O2 */
        species[8] =
            -1.800696000000000e+04 * invT
            +4.072030000000000e+00
            -4.573167000000000e+00 * tc[0]
            -2.168068000000000e-03 * tc[1]
            +2.457815000000000e-07 * tc[2]
            -1.957420000000000e-11 * tc[3]
            +7.158270000000000e-16 * tc[4];
        /*species 9: CO2 */
        species[9] =
            -4.896696000000000e+04 * invT
            +5.409018900000000e+00
            -4.453623000000000e+00 * tc[0]
            -1.570084500000000e-03 * tc[1]
            +2.130685000000000e-07 * tc[2]
            -1.994997500000000e-11 * tc[3]
            +8.345165000000000e-16 * tc[4];
        /*species 10: CO */
        species[10] =
            -1.426835000000000e+04 * invT
            -3.083140000000000e+00
            -3.025078000000000e+00 * tc[0]
            -7.213445000000000e-04 * tc[1]
            +9.384713333333334e-08 * tc[2]
            -8.488174999999999e-12 * tc[3]
            +3.455476000000000e-16 * tc[4];
        /*species 11: CH2O */
        species[11] =
            -1.532037000000000e+04 * invT
            -3.916966000000000e+00
            -2.995606000000000e+00 * tc[0]
            -3.340660500000000e-03 * tc[1]
            +4.381591666666666e-07 * tc[2]
            -3.947627500000000e-11 * tc[3]
            +1.606258500000000e-15 * tc[4];
        /*species 12: CH3 */
        species[12] =
            +1.643781000000000e+04 * invT
            -2.608645000000000e+00
            -2.844052000000000e+00 * tc[0]
            -3.068987000000000e-03 * tc[1]
            +3.717241666666666e-07 * tc[2]
            -3.154300833333333e-11 * tc[3]
            +1.226079500000000e-15 * tc[4];
        /*species 13: CH4 */
        species[13] =
            -1.008079000000000e+04 * invT
            -7.939916000000000e+00
            -1.683479000000000e+00 * tc[0]
            -5.118620000000000e-03 * tc[1]
            +6.458548333333333e-07 * tc[2]
            -5.654654166666667e-11 * tc[3]
            +2.251711500000000e-15 * tc[4];
        /*species 14: CH3OH */
        species[14] =
            -2.615791000000000e+04 * invT
            +1.650865000000000e+00
            -4.029061000000000e+00 * tc[0]
            -4.688296500000000e-03 * tc[1]
            +5.083756666666666e-07 * tc[2]
            -3.632327500000000e-11 * tc[3]
            +1.112361500000000e-15 * tc[4];
        /*species 16: CH2CO */
        species[16] =
            -8.583402000000000e+03 * invT
            +1.369639800000000e+01
            -6.038817000000000e+00 * tc[0]
            -2.902420000000000e-03 * tc[1]
            +3.201590000000000e-07 * tc[2]
            -2.328737500000000e-11 * tc[3]
            +7.294340000000000e-16 * tc[4];
        /*species 18: C2H2 */
        species[18] =
            +2.566766000000000e+04 * invT
            +7.237108000000000e+00
            -4.436770000000000e+00 * tc[0]
            -2.688019500000000e-03 * tc[1]
            +3.188028333333333e-07 * tc[2]
            -2.738649166666667e-11 * tc[3]
            +1.078355000000000e-15 * tc[4];
        /*species 20: C2H4 */
        species[20] =
            +4.428289000000000e+03 * invT
            +1.298030000000000e+00
            -3.528419000000000e+00 * tc[0]
            -5.742590000000000e-03 * tc[1]
            +7.363975000000000e-07 * tc[2]
            -6.537167500000001e-11 * tc[3]
            +2.633424000000000e-15 * tc[4];
    }

    /*species with midpoint at T=1357 kelvin */
    if (T < 1357) {
        /*species 4: OH */
        species[4] =
            +3.743212520000000e+03 * invT
            +9.857209199999999e-01
            -3.435862190000000e+00 * tc[0]
            -1.011179020000000e-04 * tc[1]
            +1.892440200000000e-08 * tc[2]
            -2.020376241666666e-11 * tc[3]
            +3.718255155000000e-15 * tc[4];
    } else {
        /*species 4: OH */
        species[4] =
            +4.120853740000000e+03 * invT
            -4.480675530000000e+00
            -2.625997540000000e+00 * tc[0]
            -6.599620300000000e-04 * tc[1]
            +5.995411166666667e-08 * tc[2]
            -3.546923333333333e-12 * tc[3]
            +9.102400800000000e-17 * tc[4];
    }

    /*species with midpoint at T=1390 kelvin */
    if (T < 1390) {
        /*species 7: HO2 */
        species[7] =
            +8.091810130000000e+02 * invT
            -5.210604430000000e+00
            -3.183106560000000e+00 * tc[0]
            -1.833839750000000e-03 * tc[1]
            +1.553975203333333e-07 * tc[2]
            +2.715440991666667e-11 * tc[3]
            -7.556995600000000e-15 * tc[4];
    } else {
        /*species 7: HO2 */
        species[7] =
            +3.981276890000000e+02 * invT
            +9.803158699999996e-01
            -4.105474230000000e+00 * tc[0]
            -1.192264175000000e-03 * tc[1]
            +1.343913315000000e-07 * tc[2]
            -1.034931025000000e-11 * tc[3]
            +3.582000540000000e-16 * tc[4];
    }

    /*species with midpoint at T=1384 kelvin */
    if (T < 1384) {
        /*species 15: C2H6 */
        species[15] =
            -1.123455340000000e+04 * invT
            -2.119016043440000e+01
            +2.528543440000000e-02 * tc[0]
            -1.203823770000000e-02 * tc[1]
            +1.864891200000000e-06 * tc[2]
            -1.736174175000000e-10 * tc[3]
            +2.649343080000000e-15 * tc[4];
    } else {
        /*species 15: C2H6 */
        species[15] =
            -1.375000140000000e+04 * invT
            +1.911495885000000e+01
            -6.106833850000000e+00 * tc[0]
            -6.461818050000000e-03 * tc[1]
            +7.375453266666666e-07 * tc[2]
            -5.728264383333333e-11 * tc[3]
            +1.994508660000000e-15 * tc[4];
    }

    /*species with midpoint at T=1376 kelvin */
    if (T < 1376) {
        /*species 17: HOCHO */
        species[17] =
            -4.646165040000000e+04 * invT
            -1.585309795000000e+01
            -1.435481850000000e+00 * tc[0]
            -8.168150799999999e-03 * tc[1]
            +1.770957016666667e-06 * tc[2]
            -2.767774808333333e-10 * tc[3]
            +2.010880515000000e-14 * tc[4];
    } else {
        /*species 17: HOCHO */
        species[17] =
            -4.839954000000000e+04 * invT
            +1.799780993000000e+01
            -6.687330130000000e+00 * tc[0]
            -2.571446840000000e-03 * tc[1]
            +3.037308550000000e-07 * tc[2]
            -2.414326358333334e-11 * tc[3]
            +8.544609950000001e-16 * tc[4];
    }

    /*species with midpoint at T=1388 kelvin */
    if (T < 1388) {
        /*species 19: C3H6 */
        species[19] =
            +1.066881640000000e+03 * invT
            -2.150575815600000e+01
            -3.946154440000000e-01 * tc[0]
            -1.445538310000000e-02 * tc[1]
            +2.581446800000000e-06 * tc[2]
            -3.240118408333333e-10 * tc[3]
            +1.689451760000000e-14 * tc[4];
    } else {
        /*species 19: C3H6 */
        species[19] =
            -1.878212710000000e+03 * invT
            +2.803202638000000e+01
            -8.015959580000001e+00 * tc[0]
            -6.851181700000000e-03 * tc[1]
            +7.770828883333333e-07 * tc[2]
            -6.010453350000000e-11 * tc[3]
            +2.086850630000000e-15 * tc[4];
    }

    /*species with midpoint at T=1400 kelvin */
    if (T < 1400) {
        /*species 21: C3H4XA */
        species[21] =
            +2.251243000000000e+04 * invT
            -7.395871000000000e+00
            -2.539831000000000e+00 * tc[0]
            -8.167185000000000e-03 * tc[1]
            +2.941583333333333e-07 * tc[2]
            +3.872804166666666e-10 * tc[3]
            -8.645655000000001e-14 * tc[4];
    } else {
        /*species 21: C3H4XA */
        species[21] =
            +1.954972000000000e+04 * invT
            +4.054686600000000e+01
            -9.776256000000000e+00 * tc[0]
            -2.651069000000000e-03 * tc[1]
            +6.168530000000000e-08 * tc[2]
            +2.521988333333334e-11 * tc[3]
            -2.544790500000000e-15 * tc[4];
    }

    /*species with midpoint at T=1398 kelvin */
    if (T < 1398) {
        /*species 22: C4H6 */
        species[22] =
            +1.175513140000000e+04 * invT
            -3.051353451000000e+01
            +1.430951210000000e+00 * tc[0]
            -2.393530310000000e-02 * tc[1]
            +6.924113333333333e-06 * tc[2]
            -1.596246266666667e-09 * tc[3]
            +1.785792535000000e-13 * tc[4];
    } else {
        /*species 22: C4H6 */
        species[22] =
            +7.790397700000000e+03 * invT
            +4.814817380000000e+01
            -1.116337890000000e+01 * tc[0]
            -6.858198250000000e-03 * tc[1]
            +7.828596383333334e-07 * tc[2]
            -6.080781966666666e-11 * tc[3]
            +2.117431015000000e-15 * tc[4];
    }

    /*species with midpoint at T=1392 kelvin */
    if (T < 1392) {
        /*species 23: C4H8X1 */
        species[23] =
            -1.578750350000000e+03 * invT
            -3.033979568900000e+01
            +8.313720890000000e-01 * tc[0]
            -2.262904890000000e-02 * tc[1]
            +4.894309316666667e-06 * tc[2]
            -8.351702999999999e-10 * tc[3]
            +7.159584000000000e-14 * tc[4];
    } else {
        /*species 23: C4H8X1 */
        species[23] =
            -5.978710380000000e+03 * invT
            +4.778781060000000e+01
            -1.135086680000000e+01 * tc[0]
            -9.030893850000001e-03 * tc[1]
            +1.026821715000000e-06 * tc[2]
            -7.955441325000001e-11 * tc[3]
            +2.765448205000000e-15 * tc[4];
    }

    /*species with midpoint at T=1391 kelvin */
    if (T < 1391) {
        /*species 24: NXC7H16 */
        species[24] =
            -2.565865650000000e+04 * invT
            -3.664165307000000e+01
            +1.268361870000000e+00 * tc[0]
            -4.271779100000000e-02 * tc[1]
            +8.755779766666667e-06 * tc[2]
            -1.357881008333333e-09 * tc[3]
            +1.011974625000000e-13 * tc[4];
    } else {
        /*species 24: NXC7H16 */
        species[24] =
            -3.427600810000000e+04 * invT
            +1.145189165000000e+02
            -2.221489690000000e+01 * tc[0]
            -1.738378750000000e-02 * tc[1]
            +1.973452150000000e-06 * tc[2]
            -1.527487316666667e-10 * tc[3]
            +5.306513300000000e-15 * tc[4];
    }
    return;
}


/*compute the g/(RT) at the given temperature */
/*tc contains precomputed powers of T, tc[0] = log(T) */
AMREX_GPU_HOST_DEVICE void gibbs_qss(double * species, double *  tc)
{

    /*temperature */
    double T = tc[1];
    double invT = 1 / T;

    /*species with midpoint at T=1000 kelvin */
    if (T < 1000) {
        /*species 0: CH */
        species[0] =
            +7.045259000000000e+04 * invT
            -1.313860000000000e-01
            -3.200202000000000e+00 * tc[0]
            -1.036438000000000e-03 * tc[1]
            +8.557385000000000e-07 * tc[2]
            -4.778241666666666e-10 * tc[3]
            +9.777665000000000e-14 * tc[4];
        /*species 1: HCO */
        species[1] =
            +4.159922000000000e+03 * invT
            -6.085284000000000e+00
            -2.898330000000000e+00 * tc[0]
            -3.099573500000000e-03 * tc[1]
            +1.603847333333333e-06 * tc[2]
            -9.081875000000000e-10 * tc[3]
            +2.287442500000000e-13 * tc[4];
        /*species 2: CH2 */
        species[2] =
            +4.536791000000000e+04 * invT
            +2.049659000000000e+00
            -3.762237000000000e+00 * tc[0]
            -5.799095000000000e-04 * tc[1]
            -4.149308333333333e-08 * tc[2]
            -7.334030000000001e-11 * tc[3]
            +3.666217500000000e-14 * tc[4];
        /*species 3: CH2GSG */
        species[3] =
            +4.989368000000000e+04 * invT
            +3.913732930000000e+00
            -3.971265000000000e+00 * tc[0]
            +8.495445000000000e-05 * tc[1]
            -1.708948333333333e-07 * tc[2]
            -2.077125833333333e-10 * tc[3]
            +9.906330000000000e-14 * tc[4];
        /*species 4: CH3O */
        species[4] =
            +9.786011000000000e+02 * invT
            -1.104597600000000e+01
            -2.106204000000000e+00 * tc[0]
            -3.608297500000000e-03 * tc[1]
            -8.897453333333333e-07 * tc[2]
            +6.148030000000000e-10 * tc[3]
            -1.037805500000000e-13 * tc[4];
        /*species 5: C2H5 */
        species[5] =
            +1.287040000000000e+04 * invT
            -9.447498000000000e+00
            -2.690702000000000e+00 * tc[0]
            -4.359566500000000e-03 * tc[1]
            -7.366398333333332e-07 * tc[2]
            -7.782252500000001e-11 * tc[3]
            +1.963886500000000e-13 * tc[4];
        /*species 8: HCCO */
        species[8] =
            +1.965892000000000e+04 * invT
            +4.566121099999999e+00
            -5.047965000000000e+00 * tc[0]
            -2.226739000000000e-03 * tc[1]
            -3.780471666666667e-08 * tc[2]
            +1.235079166666667e-10 * tc[3]
            -1.125371000000000e-14 * tc[4];
        /*species 9: C2H3 */
        species[9] =
            +3.335225000000000e+04 * invT
            -9.096924000000001e+00
            -2.459276000000000e+00 * tc[0]
            -3.685738000000000e-03 * tc[1]
            -3.516455000000000e-07 * tc[2]
            +1.101368333333333e-10 * tc[3]
            +5.923920000000000e-14 * tc[4];
        /*species 10: CH2CHO */
        species[10] =
            +1.521477000000000e+03 * invT
            -6.149227999999999e+00
            -3.409062000000000e+00 * tc[0]
            -5.369285000000000e-03 * tc[1]
            -3.152486666666667e-07 * tc[2]
            +5.965485833333333e-10 * tc[3]
            -1.433692500000000e-13 * tc[4];
        /*species 12: CH3CO */
        species[12] =
            -4.108508000000000e+03 * invT
            -8.103572000000000e+00
            -3.125278000000000e+00 * tc[0]
            -4.889110000000000e-03 * tc[1]
            -7.535746666666667e-07 * tc[2]
            +7.507885000000000e-10 * tc[3]
            -1.596859000000000e-13 * tc[4];
        /*species 14: C3H2 */
        species[14] =
            +6.350421000000000e+04 * invT
            -5.702732000000000e+00
            -3.166714000000000e+00 * tc[0]
            -1.241286000000000e-02 * tc[1]
            +7.652728333333333e-06 * tc[2]
            -3.556682500000000e-09 * tc[3]
            +7.410759999999999e-13 * tc[4];
        /*species 15: C3H3 */
        species[15] =
            +3.988883000000000e+04 * invT
            +4.168745100000000e+00
            -4.754200000000000e+00 * tc[0]
            -5.540140000000000e-03 * tc[1]
            -4.655538333333333e-08 * tc[2]
            +4.566010000000000e-10 * tc[3]
            -9.748145000000000e-14 * tc[4];
        /*species 17: NXC3H7 */
        species[17] =
            +9.713281000000001e+03 * invT
            -1.207017300000000e+01
            -1.922537000000000e+00 * tc[0]
            -1.239463500000000e-02 * tc[1]
            -3.017081666666666e-07 * tc[2]
            +1.486055000000000e-09 * tc[3]
            -4.291498000000000e-13 * tc[4];
    } else {
        /*species 0: CH */
        species[0] =
            +7.086723000000000e+04 * invT
            -6.982150000000001e+00
            -2.196223000000000e+00 * tc[0]
            -1.170190500000000e-03 * tc[1]
            +1.176366833333333e-07 * tc[2]
            -7.506318333333334e-12 * tc[3]
            +1.927520000000000e-16 * tc[4];
        /*species 1: HCO */
        species[1] =
            +3.916324000000000e+03 * invT
            -1.995028000000000e+00
            -3.557271000000000e+00 * tc[0]
            -1.672786500000000e-03 * tc[1]
            +2.225010000000000e-07 * tc[2]
            -2.058810833333333e-11 * tc[3]
            +8.569255000000000e-16 * tc[4];
        /*species 2: CH2 */
        species[2] =
            +4.534134000000000e+04 * invT
            +1.479847000000000e+00
            -3.636408000000000e+00 * tc[0]
            -9.665285000000000e-04 * tc[1]
            +2.811693333333333e-08 * tc[2]
            +8.415825000000000e-12 * tc[3]
            -9.041279999999999e-16 * tc[4];
        /*species 3: CH2GSG */
        species[3] =
            +4.984975000000000e+04 * invT
            +1.866319000000000e+00
            -3.552889000000000e+00 * tc[0]
            -1.033394000000000e-03 * tc[1]
            +3.190193333333333e-08 * tc[2]
            +9.205608333333333e-12 * tc[3]
            -1.010675000000000e-15 * tc[4];
        /*species 4: CH3O */
        species[4] =
            +1.278325000000000e+02 * invT
            +8.412250000000001e-01
            -3.770800000000000e+00 * tc[0]
            -3.935748500000000e-03 * tc[1]
            +4.427306666666667e-07 * tc[2]
            -3.287025833333333e-11 * tc[3]
            +1.056308000000000e-15 * tc[4];
        /*species 5: C2H5 */
        species[5] =
            +1.067455000000000e+04 * invT
            +2.197137000000000e+01
            -7.190480000000000e+00 * tc[0]
            -3.242038500000000e-03 * tc[1]
            +1.071344166666667e-07 * tc[2]
            +1.956565833333333e-11 * tc[3]
            -1.940438500000000e-15 * tc[4];
        /*species 8: HCCO */
        species[8] =
            +1.901513000000000e+04 * invT
            +1.582933500000000e+01
            -6.758073000000000e+00 * tc[0]
            -1.000200000000000e-03 * tc[1]
            +3.379345000000000e-08 * tc[2]
            +8.676100000000000e-12 * tc[3]
            -9.825825000000000e-16 * tc[4];
        /*species 9: C2H3 */
        species[9] =
            +3.185435000000000e+04 * invT
            +1.446378100000000e+01
            -5.933468000000000e+00 * tc[0]
            -2.008873000000000e-03 * tc[1]
            +6.611233333333333e-08 * tc[2]
            +1.201055833333333e-11 * tc[3]
            -1.189322000000000e-15 * tc[4];
        /*species 10: CH2CHO */
        species[10] =
            +4.903218000000000e+02 * invT
            +1.102092100000000e+01
            -5.975670000000000e+00 * tc[0]
            -4.065295500000000e-03 * tc[1]
            +4.572706666666667e-07 * tc[2]
            -3.391920000000000e-11 * tc[3]
            +1.088008500000000e-15 * tc[4];
        /*species 12: CH3CO */
        species[12] =
            -5.187863000000000e+03 * invT
            +8.887228000000000e+00
            -5.612279000000000e+00 * tc[0]
            -4.224943000000000e-03 * tc[1]
            +4.756911666666667e-07 * tc[2]
            -3.531980000000000e-11 * tc[3]
            +1.134202000000000e-15 * tc[4];
        /*species 14: C3H2 */
        species[14] =
            +6.259722000000000e+04 * invT
            +2.003988100000000e+01
            -7.670981000000000e+00 * tc[0]
            -1.374374500000000e-03 * tc[1]
            +7.284905000000000e-08 * tc[2]
            +5.379665833333334e-12 * tc[3]
            -8.319435000000000e-16 * tc[4];
        /*species 15: C3H3 */
        species[15] =
            +3.847420000000000e+04 * invT
            +3.061023700000000e+01
            -8.831047000000000e+00 * tc[0]
            -2.178597500000000e-03 * tc[1]
            +6.848445000000000e-08 * tc[2]
            +1.973935833333333e-11 * tc[3]
            -2.188260000000000e-15 * tc[4];
        /*species 17: NXC3H7 */
        species[17] =
            +7.579402000000000e+03 * invT
            +2.733440100000000e+01
            -7.978291000000000e+00 * tc[0]
            -7.880565000000001e-03 * tc[1]
            +8.622071666666666e-07 * tc[2]
            -6.203243333333333e-11 * tc[3]
            +1.912489000000000e-15 * tc[4];
    }

    /*species with midpoint at T=1385 kelvin */
    if (T < 1385) {
        /*species 6: CH3O2 */
        species[6] =
            -6.843942590000000e+02 * invT
            -9.018341400000001e-01
            -4.261469060000000e+00 * tc[0]
            -5.043679950000000e-03 * tc[1]
            +5.358436400000000e-07 * tc[2]
            -1.745077225000000e-11 * tc[3]
            -2.091695515000000e-15 * tc[4];
        /*species 21: PXC4H9O2 */
        species[21] =
            -1.083581030000000e+04 * invT
            -1.940667840000000e+01
            -1.943636500000000e+00 * tc[0]
            -2.577565815000000e-02 * tc[1]
            +5.471406666666667e-06 * tc[2]
            -9.422071666666667e-10 * tc[3]
            +8.505930300000000e-14 * tc[4];
    } else {
        /*species 6: CH3O2 */
        species[6] =
            -1.535748380000000e+03 * invT
            +1.067751777000000e+01
            -5.957878910000000e+00 * tc[0]
            -3.953643130000000e-03 * tc[1]
            +4.470770566666667e-07 * tc[2]
            -3.449094475000000e-11 * tc[3]
            +1.195036650000000e-15 * tc[4];
        /*species 21: PXC4H9O2 */
        species[21] =
            -1.601460540000000e+04 * invT
            +6.982339730000000e+01
            -1.578454480000000e+01 * tc[0]
            -1.076054550000000e-02 * tc[1]
            +1.241515028333333e-06 * tc[2]
            -9.713172583333334e-11 * tc[3]
            +3.399428045000000e-15 * tc[4];
    }

    /*species with midpoint at T=1390 kelvin */
    if (T < 1390) {
        /*species 7: CH3O2H */
        species[7] =
            -1.771979260000000e+04 * invT
            -6.021811320000000e+00
            -3.234428170000000e+00 * tc[0]
            -9.506488350000000e-03 * tc[1]
            +1.889771450000000e-06 * tc[2]
            -2.835888775000000e-10 * tc[3]
            +2.059151110000000e-14 * tc[4];
    } else {
        /*species 7: CH3O2H */
        species[7] =
            -1.966787710000000e+04 * invT
            +2.754823381000000e+01
            -8.431170910000001e+00 * tc[0]
            -4.034089545000000e-03 * tc[1]
            +4.618248683333333e-07 * tc[2]
            -3.594435358333333e-11 * tc[3]
            +1.253460730000000e-15 * tc[4];
    }

    /*species with midpoint at T=1389 kelvin */
    if (T < 1389) {
        /*species 11: C2H5O */
        species[11] =
            -3.352529250000000e+03 * invT
            -2.231351709200000e+01
            -4.944207080000000e-01 * tc[0]
            -1.358872170000000e-02 * tc[1]
            +2.765150166666667e-06 * tc[2]
            -4.293368333333333e-10 * tc[3]
            +3.242484575000000e-14 * tc[4];
    } else {
        /*species 11: C2H5O */
        species[11] =
            -6.072749530000000e+03 * invT
            +2.521507672000000e+01
            -7.873397720000000e+00 * tc[0]
            -5.653645350000000e-03 * tc[1]
            +6.407023683333334e-07 * tc[2]
            -4.953450875000000e-11 * tc[3]
            +1.719472690000000e-15 * tc[4];
    }

    /*species with midpoint at T=1382 kelvin */
    if (T < 1382) {
        /*species 13: C2H5O2 */
        species[13] =
            -5.038807580000000e+03 * invT
            -1.420893532000000e+01
            -2.268461880000000e+00 * tc[0]
            -1.384712890000000e-02 * tc[1]
            +2.846735100000000e-06 * tc[2]
            -4.899598983333333e-10 * tc[3]
            +4.604495345000000e-14 * tc[4];
        /*species 26: C7H15X2 */
        species[26] =
            -2.356053030000000e+03 * invT
            -3.377006617670000e+01
            +3.791557670000000e-02 * tc[0]
            -3.783632850000000e-02 * tc[1]
            +6.791227233333333e-06 * tc[2]
            -7.772324525000000e-10 * tc[3]
            +2.461803725000000e-14 * tc[4];
    } else {
        /*species 13: C2H5O2 */
        species[13] =
            -7.824817950000000e+03 * invT
            +3.254826223000000e+01
            -9.486960229999999e+00 * tc[0]
            -6.223627250000000e-03 * tc[1]
            +7.202692933333333e-07 * tc[2]
            -5.646525275000000e-11 * tc[3]
            +1.978922840000000e-15 * tc[4];
        /*species 26: C7H15X2 */
        species[26] =
            -1.058736160000000e+04 * invT
            +1.068578495000000e+02
            -2.163688420000000e+01 * tc[0]
            -1.616624020000000e-02 * tc[1]
            +1.821230116666667e-06 * tc[2]
            -1.402975500000000e-10 * tc[3]
            +4.858870455000000e-15 * tc[4];
    }

    /*species with midpoint at T=1397 kelvin */
    if (T < 1397) {
        /*species 16: C3H5XA */
        species[16] =
            +1.938342260000000e+04 * invT
            -2.583584505800000e+01
            +5.291319580000000e-01 * tc[0]
            -1.672795500000000e-02 * tc[1]
            +4.223350450000001e-06 * tc[2]
            -8.572146166666666e-10 * tc[3]
            +8.662917000000000e-14 * tc[4];
        /*species 24: C5H11X1 */
        species[24] =
            +4.839953030000000e+03 * invT
            -3.346275221200000e+01
            +9.052559120000000e-01 * tc[0]
            -3.053164260000000e-02 * tc[1]
            +6.824863750000000e-06 * tc[2]
            -1.217445583333333e-09 * tc[3]
            +1.094298075000000e-13 * tc[4];
    } else {
        /*species 16: C3H5XA */
        species[16] =
            +1.635760920000000e+04 * invT
            +3.103978458000000e+01
            -8.458839579999999e+00 * tc[0]
            -5.634774150000000e-03 * tc[1]
            +6.396547733333333e-07 * tc[2]
            -4.950492658333333e-11 * tc[3]
            +1.719590150000000e-15 * tc[4];
        /*species 24: C5H11X1 */
        species[24] =
            -9.232416370000000e+02 * invT
            +7.027635990000000e+01
            -1.532347400000000e+01 * tc[0]
            -1.195206000000000e-02 * tc[1]
            +1.357952698333333e-06 * tc[2]
            -1.051468633333333e-10 * tc[3]
            +3.653386675000001e-15 * tc[4];
    }

    /*species with midpoint at T=1384 kelvin */
    if (T < 1384) {
        /*species 18: NXC3H7O2 */
        species[18] =
            -7.937455670000000e+03 * invT
            -1.680095988000000e+01
            -2.107314920000000e+00 * tc[0]
            -1.980824930000000e-02 * tc[1]
            +4.158193316666667e-06 * tc[2]
            -7.162085833333334e-10 * tc[3]
            +6.562016500000000e-14 * tc[4];
    } else {
        /*species 18: NXC3H7O2 */
        species[18] =
            -1.191946520000000e+04 * invT
            +5.116763560000000e+01
            -1.263270590000000e+01 * tc[0]
            -8.495536300000000e-03 * tc[1]
            +9.814447883333334e-07 * tc[2]
            -7.684961633333334e-11 * tc[3]
            +2.691153375000000e-15 * tc[4];
    }

    /*species with midpoint at T=1392 kelvin */
    if (T < 1392) {
        /*species 19: C4H7 */
        species[19] =
            +1.499335910000000e+04 * invT
            -2.708007795200000e+01
            +3.505083520000000e-01 * tc[0]
            -2.132556215000000e-02 * tc[1]
            +4.849656216666667e-06 * tc[2]
            -8.783659500000000e-10 * tc[3]
            +8.002992699999999e-14 * tc[4];
        /*species 22: C5H9 */
        species[22] =
            +1.255898240000000e+04 * invT
            -3.402426990000000e+01
            +1.380139500000000e+00 * tc[0]
            -2.788042435000000e-02 * tc[1]
            +6.169065466666666e-06 * tc[2]
            -1.057365841666667e-09 * tc[3]
            +8.926941750000000e-14 * tc[4];
        /*species 23: C5H10X1 */
        species[23] =
            -4.465466660000000e+03 * invT
            -3.333621381000000e+01
            +1.062234810000000e+00 * tc[0]
            -2.871091470000000e-02 * tc[1]
            +6.241448166666667e-06 * tc[2]
            -1.061374908333333e-09 * tc[3]
            +8.980489449999999e-14 * tc[4];
        /*species 25: C6H12X1 */
        species[25] =
            -7.343686170000000e+03 * invT
            -3.666482115000000e+01
            +1.352752050000000e+00 * tc[0]
            -3.493277130000000e-02 * tc[1]
            +7.656800366666667e-06 * tc[2]
            -1.308061191666667e-09 * tc[3]
            +1.106480875000000e-13 * tc[4];
    } else {
        /*species 19: C4H7 */
        species[19] =
            +1.090419370000000e+04 * invT
            +4.676965930000000e+01
            -1.121035780000000e+01 * tc[0]
            -8.024159800000000e-03 * tc[1]
            +9.108371533333334e-07 * tc[2]
            -7.049508775000000e-11 * tc[3]
            +2.448863695000000e-15 * tc[4];
        /*species 22: C5H9 */
        species[22] =
            +7.004961350000000e+03 * invT
            +6.563622270000000e+01
            -1.418604540000000e+01 * tc[0]
            -1.035644495000000e-02 * tc[1]
            +1.178267695000000e-06 * tc[2]
            -9.133927750000000e-11 * tc[3]
            +3.176611040000000e-15 * tc[4];
        /*species 23: C5H10X1 */
        species[23] =
            -1.008982050000000e+04 * invT
            +6.695354750000000e+01
            -1.458515390000000e+01 * tc[0]
            -1.120362355000000e-02 * tc[1]
            +1.272246708333333e-06 * tc[2]
            -9.849080500000001e-11 * tc[3]
            +3.421925695000000e-15 * tc[4];
        /*species 25: C6H12X1 */
        species[25] =
            -1.420628600000000e+04 * invT
            +8.621563800000001e+01
            -1.783375290000000e+01 * tc[0]
            -1.336888290000000e-02 * tc[1]
            +1.516727955000000e-06 * tc[2]
            -1.173498066666667e-10 * tc[3]
            +4.075621220000000e-15 * tc[4];
    }

    /*species with midpoint at T=1395 kelvin */
    if (T < 1395) {
        /*species 20: PXC4H9 */
        species[20] =
            +7.689452480000000e+03 * invT
            -2.912305292500000e+01
            +4.377797250000000e-01 * tc[0]
            -2.394861820000000e-02 * tc[1]
            +5.233719316666667e-06 * tc[2]
            -9.148872666666667e-10 * tc[3]
            +8.100533200000001e-14 * tc[4];
    } else {
        /*species 20: PXC4H9 */
        species[20] =
            +3.172319420000000e+03 * invT
            +5.149359040000000e+01
            -1.215100820000000e+01 * tc[0]
            -9.715535850000000e-03 * tc[1]
            +1.102629916666667e-06 * tc[2]
            -8.531261333333333e-11 * tc[3]
            +2.962648535000000e-15 * tc[4];
    }
    return;
}


/*compute the a/(RT) at the given temperature */
/*tc contains precomputed powers of T, tc[0] = log(T) */
AMREX_GPU_HOST_DEVICE void helmholtz(double * species, double *  tc)
{

    /*temperature */
    double T = tc[1];
    double invT = 1 / T;

    /*species with midpoint at T=1000 kelvin */
    if (T < 1000) {
        /*species 0: N2 */
        species[0] =
            -1.02090000e+03 * invT
            -1.65169500e+00
            -3.29867700e+00 * tc[0]
            -7.04120000e-04 * tc[1]
            +6.60537000e-07 * tc[2]
            -4.70126250e-10 * tc[3]
            +1.22242750e-13 * tc[4];
        /*species 1: O */
        species[1] =
            +2.91476400e+04 * invT
            -1.01756600e+00
            -2.94642900e+00 * tc[0]
            +8.19083000e-04 * tc[1]
            -4.03505333e-07 * tc[2]
            +1.33570250e-10 * tc[3]
            -1.94534800e-14 * tc[4];
        /*species 2: H2 */
        species[2] =
            -1.01252100e+03 * invT
            +5.59221800e+00
            -3.29812400e+00 * tc[0]
            -4.12472100e-04 * tc[1]
            +1.35716917e-07 * tc[2]
            +7.89619500e-12 * tc[3]
            -2.06743600e-14 * tc[4];
        /*species 3: H */
        species[3] =
            +2.54747466e+04 * invT
            +1.96638554e+00
            -2.50104422e+00 * tc[0]
            -0.00000000e+00 * tc[1]
            -0.00000000e+00 * tc[2]
            -0.00000000e+00 * tc[3]
            -0.00000000e+00 * tc[4];
        /*species 5: H2O */
        species[5] =
            -3.02081100e+04 * invT
            -2.03391000e-01
            -3.38684200e+00 * tc[0]
            -1.73749100e-03 * tc[1]
            +1.05911600e-06 * tc[2]
            -5.80715083e-10 * tc[3]
            +1.25329400e-13 * tc[4];
        /*species 6: O2 */
        species[6] =
            -1.00524900e+03 * invT
            -3.82180200e+00
            -3.21293600e+00 * tc[0]
            -5.63743000e-04 * tc[1]
            +9.59358333e-08 * tc[2]
            -1.09489750e-10 * tc[3]
            +4.38427700e-14 * tc[4];
        /*species 8: H2O2 */
        species[8] =
            -1.76631500e+04 * invT
            -4.39660900e+00
            -3.38875400e+00 * tc[0]
            -3.28461300e-03 * tc[1]
            +2.47502167e-08 * tc[2]
            +3.85483833e-10 * tc[3]
            -1.23575750e-13 * tc[4];
        /*species 9: CO2 */
        species[9] =
            -4.83731400e+04 * invT
            -8.91276500e+00
            -2.27572500e+00 * tc[0]
            -4.96103600e-03 * tc[1]
            +1.73485167e-06 * tc[2]
            -5.72223917e-10 * tc[3]
            +1.05864000e-13 * tc[4];
        /*species 10: CO */
        species[10] =
            -1.43105400e+04 * invT
            -2.58644500e+00
            -3.26245200e+00 * tc[0]
            -7.55970500e-04 * tc[1]
            +6.46959167e-07 * tc[2]
            -4.65162000e-10 * tc[3]
            +1.23747550e-13 * tc[4];
        /*species 11: CH2O */
        species[11] =
            -1.48654000e+04 * invT
            -1.31320890e+01
            -1.65273100e+00 * tc[0]
            -6.31572000e-03 * tc[1]
            +3.14694667e-06 * tc[2]
            -1.70835917e-09 * tc[3]
            +4.20661850e-13 * tc[4];
        /*species 12: CH3 */
        species[12] =
            +1.64237800e+04 * invT
            -5.35935100e+00
            -2.43044300e+00 * tc[0]
            -5.56205000e-03 * tc[1]
            +2.80036667e-06 * tc[2]
            -1.35152417e-09 * tc[3]
            +2.93247650e-13 * tc[4];
        /*species 13: CH4 */
        species[13] =
            -9.82522900e+03 * invT
            -1.39434485e+01
            -7.78741500e-01 * tc[0]
            -8.73834000e-03 * tc[1]
            +4.63901500e-06 * tc[2]
            -2.54142333e-09 * tc[3]
            +6.11965500e-13 * tc[4];
        /*species 14: CH3OH */
        species[14] =
            -2.53534800e+04 * invT
            -9.57251500e+00
            -2.66011500e+00 * tc[0]
            -3.67075400e-03 * tc[1]
            -1.19500850e-06 * tc[2]
            +7.32766167e-10 * tc[3]
            -1.19528500e-13 * tc[4];
        /*species 16: CH2CO */
        species[16] =
            -7.63263700e+03 * invT
            -6.69858200e+00
            -2.97497100e+00 * tc[0]
            -6.05935500e-03 * tc[1]
            +3.90841000e-07 * tc[2]
            +5.38890417e-10 * tc[3]
            -1.95282450e-13 * tc[4];
        /*species 18: C2H2 */
        species[18] =
            +2.61244400e+04 * invT
            -7.79181600e+00
            -2.01356200e+00 * tc[0]
            -7.59522500e-03 * tc[1]
            +2.69386500e-06 * tc[2]
            -7.56582667e-10 * tc[3]
            +9.56373000e-14 * tc[4];
        /*species 20: C2H4 */
        species[20] =
            +5.57304600e+03 * invT
            -2.60729780e+01
            +8.61488000e-01 * tc[0]
            -1.39808150e-02 * tc[1]
            +5.64779500e-06 * tc[2]
            -2.32096000e-09 * tc[3]
            +4.86893950e-13 * tc[4];
    } else {
        /*species 0: N2 */
        species[0] =
            -9.22797700e+02 * invT
            -4.05388800e+00
            -2.92664000e+00 * tc[0]
            -7.43988500e-04 * tc[1]
            +9.47460167e-08 * tc[2]
            -8.41420000e-12 * tc[3]
            +3.37667550e-16 * tc[4];
        /*species 1: O */
        species[1] =
            +2.92308000e+04 * invT
            -3.37824800e+00
            -2.54206000e+00 * tc[0]
            +1.37753100e-05 * tc[1]
            +5.17133833e-10 * tc[2]
            -3.79255583e-13 * tc[3]
            +2.18402600e-17 * tc[4];
        /*species 2: H2 */
        species[2] =
            -8.35034000e+02 * invT
            +3.34653300e+00
            -2.99142300e+00 * tc[0]
            -3.50032200e-04 * tc[1]
            +9.38971500e-09 * tc[2]
            +7.69298167e-13 * tc[3]
            -7.91376000e-17 * tc[4];
        /*species 3: H */
        species[3] =
            +2.54747466e+04 * invT
            +1.96638554e+00
            -2.50104422e+00 * tc[0]
            -0.00000000e+00 * tc[1]
            -0.00000000e+00 * tc[2]
            -0.00000000e+00 * tc[3]
            -0.00000000e+00 * tc[4];
        /*species 5: H2O */
        species[5] =
            -2.98992100e+04 * invT
            -5.19067100e+00
            -2.67214600e+00 * tc[0]
            -1.52814650e-03 * tc[1]
            +1.45504333e-07 * tc[2]
            -1.00083000e-11 * tc[3]
            +3.19580900e-16 * tc[4];
        /*species 6: O2 */
        species[6] =
            -1.23393000e+03 * invT
            -4.91588000e-01
            -3.69757800e+00 * tc[0]
            -3.06759850e-04 * tc[1]
            +2.09807000e-08 * tc[2]
            -1.47940083e-12 * tc[3]
            +5.68217500e-17 * tc[4];
        /*species 8: H2O2 */
        species[8] =
            -1.80069600e+04 * invT
            +3.07203000e+00
            -4.57316700e+00 * tc[0]
            -2.16806800e-03 * tc[1]
            +2.45781500e-07 * tc[2]
            -1.95742000e-11 * tc[3]
            +7.15827000e-16 * tc[4];
        /*species 9: CO2 */
        species[9] =
            -4.89669600e+04 * invT
            +4.40901890e+00
            -4.45362300e+00 * tc[0]
            -1.57008450e-03 * tc[1]
            +2.13068500e-07 * tc[2]
            -1.99499750e-11 * tc[3]
            +8.34516500e-16 * tc[4];
        /*species 10: CO */
        species[10] =
            -1.42683500e+04 * invT
            -4.08314000e+00
            -3.02507800e+00 * tc[0]
            -7.21344500e-04 * tc[1]
            +9.38471333e-08 * tc[2]
            -8.48817500e-12 * tc[3]
            +3.45547600e-16 * tc[4];
        /*species 11: CH2O */
        species[11] =
            -1.53203700e+04 * invT
            -4.91696600e+00
            -2.99560600e+00 * tc[0]
            -3.34066050e-03 * tc[1]
            +4.38159167e-07 * tc[2]
            -3.94762750e-11 * tc[3]
            +1.60625850e-15 * tc[4];
        /*species 12: CH3 */
        species[12] =
            +1.64378100e+04 * invT
            -3.60864500e+00
            -2.84405200e+00 * tc[0]
            -3.06898700e-03 * tc[1]
            +3.71724167e-07 * tc[2]
            -3.15430083e-11 * tc[3]
            +1.22607950e-15 * tc[4];
        /*species 13: CH4 */
        species[13] =
            -1.00807900e+04 * invT
            -8.93991600e+00
            -1.68347900e+00 * tc[0]
            -5.11862000e-03 * tc[1]
            +6.45854833e-07 * tc[2]
            -5.65465417e-11 * tc[3]
            +2.25171150e-15 * tc[4];
        /*species 14: CH3OH */
        species[14] =
            -2.61579100e+04 * invT
            +6.50865000e-01
            -4.02906100e+00 * tc[0]
            -4.68829650e-03 * tc[1]
            +5.08375667e-07 * tc[2]
            -3.63232750e-11 * tc[3]
            +1.11236150e-15 * tc[4];
        /*species 16: CH2CO */
        species[16] =
            -8.58340200e+03 * invT
            +1.26963980e+01
            -6.03881700e+00 * tc[0]
            -2.90242000e-03 * tc[1]
            +3.20159000e-07 * tc[2]
            -2.32873750e-11 * tc[3]
            +7.29434000e-16 * tc[4];
        /*species 18: C2H2 */
        species[18] =
            +2.56676600e+04 * invT
            +6.23710800e+00
            -4.43677000e+00 * tc[0]
            -2.68801950e-03 * tc[1]
            +3.18802833e-07 * tc[2]
            -2.73864917e-11 * tc[3]
            +1.07835500e-15 * tc[4];
        /*species 20: C2H4 */
        species[20] =
            +4.42828900e+03 * invT
            +2.98030000e-01
            -3.52841900e+00 * tc[0]
            -5.74259000e-03 * tc[1]
            +7.36397500e-07 * tc[2]
            -6.53716750e-11 * tc[3]
            +2.63342400e-15 * tc[4];
    }

    /*species with midpoint at T=1357 kelvin */
    if (T < 1357) {
        /*species 4: OH */
        species[4] =
            +3.74321252e+03 * invT
            -1.42790800e-02
            -3.43586219e+00 * tc[0]
            -1.01117902e-04 * tc[1]
            +1.89244020e-08 * tc[2]
            -2.02037624e-11 * tc[3]
            +3.71825515e-15 * tc[4];
    } else {
        /*species 4: OH */
        species[4] =
            +4.12085374e+03 * invT
            -5.48067553e+00
            -2.62599754e+00 * tc[0]
            -6.59962030e-04 * tc[1]
            +5.99541117e-08 * tc[2]
            -3.54692333e-12 * tc[3]
            +9.10240080e-17 * tc[4];
    }

    /*species with midpoint at T=1390 kelvin */
    if (T < 1390) {
        /*species 7: HO2 */
        species[7] =
            +8.09181013e+02 * invT
            -6.21060443e+00
            -3.18310656e+00 * tc[0]
            -1.83383975e-03 * tc[1]
            +1.55397520e-07 * tc[2]
            +2.71544099e-11 * tc[3]
            -7.55699560e-15 * tc[4];
    } else {
        /*species 7: HO2 */
        species[7] =
            +3.98127689e+02 * invT
            -1.96841300e-02
            -4.10547423e+00 * tc[0]
            -1.19226417e-03 * tc[1]
            +1.34391332e-07 * tc[2]
            -1.03493103e-11 * tc[3]
            +3.58200054e-16 * tc[4];
    }

    /*species with midpoint at T=1384 kelvin */
    if (T < 1384) {
        /*species 15: C2H6 */
        species[15] =
            -1.12345534e+04 * invT
            -2.21901604e+01
            +2.52854344e-02 * tc[0]
            -1.20382377e-02 * tc[1]
            +1.86489120e-06 * tc[2]
            -1.73617417e-10 * tc[3]
            +2.64934308e-15 * tc[4];
    } else {
        /*species 15: C2H6 */
        species[15] =
            -1.37500014e+04 * invT
            +1.81149589e+01
            -6.10683385e+00 * tc[0]
            -6.46181805e-03 * tc[1]
            +7.37545327e-07 * tc[2]
            -5.72826438e-11 * tc[3]
            +1.99450866e-15 * tc[4];
    }

    /*species with midpoint at T=1376 kelvin */
    if (T < 1376) {
        /*species 17: HOCHO */
        species[17] =
            -4.64616504e+04 * invT
            -1.68530979e+01
            -1.43548185e+00 * tc[0]
            -8.16815080e-03 * tc[1]
            +1.77095702e-06 * tc[2]
            -2.76777481e-10 * tc[3]
            +2.01088051e-14 * tc[4];
    } else {
        /*species 17: HOCHO */
        species[17] =
            -4.83995400e+04 * invT
            +1.69978099e+01
            -6.68733013e+00 * tc[0]
            -2.57144684e-03 * tc[1]
            +3.03730855e-07 * tc[2]
            -2.41432636e-11 * tc[3]
            +8.54460995e-16 * tc[4];
    }

    /*species with midpoint at T=1388 kelvin */
    if (T < 1388) {
        /*species 19: C3H6 */
        species[19] =
            +1.06688164e+03 * invT
            -2.25057582e+01
            -3.94615444e-01 * tc[0]
            -1.44553831e-02 * tc[1]
            +2.58144680e-06 * tc[2]
            -3.24011841e-10 * tc[3]
            +1.68945176e-14 * tc[4];
    } else {
        /*species 19: C3H6 */
        species[19] =
            -1.87821271e+03 * invT
            +2.70320264e+01
            -8.01595958e+00 * tc[0]
            -6.85118170e-03 * tc[1]
            +7.77082888e-07 * tc[2]
            -6.01045335e-11 * tc[3]
            +2.08685063e-15 * tc[4];
    }

    /*species with midpoint at T=1400 kelvin */
    if (T < 1400) {
        /*species 21: C3H4XA */
        species[21] =
            +2.25124300e+04 * invT
            -8.39587100e+00
            -2.53983100e+00 * tc[0]
            -8.16718500e-03 * tc[1]
            +2.94158333e-07 * tc[2]
            +3.87280417e-10 * tc[3]
            -8.64565500e-14 * tc[4];
    } else {
        /*species 21: C3H4XA */
        species[21] =
            +1.95497200e+04 * invT
            +3.95468660e+01
            -9.77625600e+00 * tc[0]
            -2.65106900e-03 * tc[1]
            +6.16853000e-08 * tc[2]
            +2.52198833e-11 * tc[3]
            -2.54479050e-15 * tc[4];
    }

    /*species with midpoint at T=1398 kelvin */
    if (T < 1398) {
        /*species 22: C4H6 */
        species[22] =
            +1.17551314e+04 * invT
            -3.15135345e+01
            +1.43095121e+00 * tc[0]
            -2.39353031e-02 * tc[1]
            +6.92411333e-06 * tc[2]
            -1.59624627e-09 * tc[3]
            +1.78579253e-13 * tc[4];
    } else {
        /*species 22: C4H6 */
        species[22] =
            +7.79039770e+03 * invT
            +4.71481738e+01
            -1.11633789e+01 * tc[0]
            -6.85819825e-03 * tc[1]
            +7.82859638e-07 * tc[2]
            -6.08078197e-11 * tc[3]
            +2.11743101e-15 * tc[4];
    }

    /*species with midpoint at T=1392 kelvin */
    if (T < 1392) {
        /*species 23: C4H8X1 */
        species[23] =
            -1.57875035e+03 * invT
            -3.13397957e+01
            +8.31372089e-01 * tc[0]
            -2.26290489e-02 * tc[1]
            +4.89430932e-06 * tc[2]
            -8.35170300e-10 * tc[3]
            +7.15958400e-14 * tc[4];
    } else {
        /*species 23: C4H8X1 */
        species[23] =
            -5.97871038e+03 * invT
            +4.67878106e+01
            -1.13508668e+01 * tc[0]
            -9.03089385e-03 * tc[1]
            +1.02682171e-06 * tc[2]
            -7.95544133e-11 * tc[3]
            +2.76544820e-15 * tc[4];
    }

    /*species with midpoint at T=1391 kelvin */
    if (T < 1391) {
        /*species 24: NXC7H16 */
        species[24] =
            -2.56586565e+04 * invT
            -3.76416531e+01
            +1.26836187e+00 * tc[0]
            -4.27177910e-02 * tc[1]
            +8.75577977e-06 * tc[2]
            -1.35788101e-09 * tc[3]
            +1.01197462e-13 * tc[4];
    } else {
        /*species 24: NXC7H16 */
        species[24] =
            -3.42760081e+04 * invT
            +1.13518917e+02
            -2.22148969e+01 * tc[0]
            -1.73837875e-02 * tc[1]
            +1.97345215e-06 * tc[2]
            -1.52748732e-10 * tc[3]
            +5.30651330e-15 * tc[4];
    }
    return;
}


/*compute Cv/R at the given temperature */
/*tc contains precomputed powers of T, tc[0] = log(T) */
AMREX_GPU_HOST_DEVICE void cv_R(double * species, double *  tc)
{

    /*temperature */
    double T = tc[1];

    /*species with midpoint at T=1000 kelvin */
    if (T < 1000) {
        /*species 0: N2 */
        species[0] =
            +2.29867700e+00
            +1.40824000e-03 * tc[1]
            -3.96322200e-06 * tc[2]
            +5.64151500e-09 * tc[3]
            -2.44485500e-12 * tc[4];
        /*species 1: O */
        species[1] =
            +1.94642900e+00
            -1.63816600e-03 * tc[1]
            +2.42103200e-06 * tc[2]
            -1.60284300e-09 * tc[3]
            +3.89069600e-13 * tc[4];
        /*species 2: H2 */
        species[2] =
            +2.29812400e+00
            +8.24944200e-04 * tc[1]
            -8.14301500e-07 * tc[2]
            -9.47543400e-11 * tc[3]
            +4.13487200e-13 * tc[4];
        /*species 3: H */
        species[3] =
            +1.50104422e+00
            +0.00000000e+00 * tc[1]
            +0.00000000e+00 * tc[2]
            +0.00000000e+00 * tc[3]
            +0.00000000e+00 * tc[4];
        /*species 5: H2O */
        species[5] =
            +2.38684200e+00
            +3.47498200e-03 * tc[1]
            -6.35469600e-06 * tc[2]
            +6.96858100e-09 * tc[3]
            -2.50658800e-12 * tc[4];
        /*species 6: O2 */
        species[6] =
            +2.21293600e+00
            +1.12748600e-03 * tc[1]
            -5.75615000e-07 * tc[2]
            +1.31387700e-09 * tc[3]
            -8.76855400e-13 * tc[4];
        /*species 8: H2O2 */
        species[8] =
            +2.38875400e+00
            +6.56922600e-03 * tc[1]
            -1.48501300e-07 * tc[2]
            -4.62580600e-09 * tc[3]
            +2.47151500e-12 * tc[4];
        /*species 9: CO2 */
        species[9] =
            +1.27572500e+00
            +9.92207200e-03 * tc[1]
            -1.04091100e-05 * tc[2]
            +6.86668700e-09 * tc[3]
            -2.11728000e-12 * tc[4];
        /*species 10: CO */
        species[10] =
            +2.26245200e+00
            +1.51194100e-03 * tc[1]
            -3.88175500e-06 * tc[2]
            +5.58194400e-09 * tc[3]
            -2.47495100e-12 * tc[4];
        /*species 11: CH2O */
        species[11] =
            +6.52731000e-01
            +1.26314400e-02 * tc[1]
            -1.88816800e-05 * tc[2]
            +2.05003100e-08 * tc[3]
            -8.41323700e-12 * tc[4];
        /*species 12: CH3 */
        species[12] =
            +1.43044300e+00
            +1.11241000e-02 * tc[1]
            -1.68022000e-05 * tc[2]
            +1.62182900e-08 * tc[3]
            -5.86495300e-12 * tc[4];
        /*species 13: CH4 */
        species[13] =
            -2.21258500e-01
            +1.74766800e-02 * tc[1]
            -2.78340900e-05 * tc[2]
            +3.04970800e-08 * tc[3]
            -1.22393100e-11 * tc[4];
        /*species 14: CH3OH */
        species[14] =
            +1.66011500e+00
            +7.34150800e-03 * tc[1]
            +7.17005100e-06 * tc[2]
            -8.79319400e-09 * tc[3]
            +2.39057000e-12 * tc[4];
        /*species 16: CH2CO */
        species[16] =
            +1.97497100e+00
            +1.21187100e-02 * tc[1]
            -2.34504600e-06 * tc[2]
            -6.46668500e-09 * tc[3]
            +3.90564900e-12 * tc[4];
        /*species 18: C2H2 */
        species[18] =
            +1.01356200e+00
            +1.51904500e-02 * tc[1]
            -1.61631900e-05 * tc[2]
            +9.07899200e-09 * tc[3]
            -1.91274600e-12 * tc[4];
        /*species 20: C2H4 */
        species[20] =
            -1.86148800e+00
            +2.79616300e-02 * tc[1]
            -3.38867700e-05 * tc[2]
            +2.78515200e-08 * tc[3]
            -9.73787900e-12 * tc[4];
    } else {
        /*species 0: N2 */
        species[0] =
            +1.92664000e+00
            +1.48797700e-03 * tc[1]
            -5.68476100e-07 * tc[2]
            +1.00970400e-10 * tc[3]
            -6.75335100e-15 * tc[4];
        /*species 1: O */
        species[1] =
            +1.54206000e+00
            -2.75506200e-05 * tc[1]
            -3.10280300e-09 * tc[2]
            +4.55106700e-12 * tc[3]
            -4.36805200e-16 * tc[4];
        /*species 2: H2 */
        species[2] =
            +1.99142300e+00
            +7.00064400e-04 * tc[1]
            -5.63382900e-08 * tc[2]
            -9.23157800e-12 * tc[3]
            +1.58275200e-15 * tc[4];
        /*species 3: H */
        species[3] =
            +1.50104422e+00
            +0.00000000e+00 * tc[1]
            +0.00000000e+00 * tc[2]
            +0.00000000e+00 * tc[3]
            +0.00000000e+00 * tc[4];
        /*species 5: H2O */
        species[5] =
            +1.67214600e+00
            +3.05629300e-03 * tc[1]
            -8.73026000e-07 * tc[2]
            +1.20099600e-10 * tc[3]
            -6.39161800e-15 * tc[4];
        /*species 6: O2 */
        species[6] =
            +2.69757800e+00
            +6.13519700e-04 * tc[1]
            -1.25884200e-07 * tc[2]
            +1.77528100e-11 * tc[3]
            -1.13643500e-15 * tc[4];
        /*species 8: H2O2 */
        species[8] =
            +3.57316700e+00
            +4.33613600e-03 * tc[1]
            -1.47468900e-06 * tc[2]
            +2.34890400e-10 * tc[3]
            -1.43165400e-14 * tc[4];
        /*species 9: CO2 */
        species[9] =
            +3.45362300e+00
            +3.14016900e-03 * tc[1]
            -1.27841100e-06 * tc[2]
            +2.39399700e-10 * tc[3]
            -1.66903300e-14 * tc[4];
        /*species 10: CO */
        species[10] =
            +2.02507800e+00
            +1.44268900e-03 * tc[1]
            -5.63082800e-07 * tc[2]
            +1.01858100e-10 * tc[3]
            -6.91095200e-15 * tc[4];
        /*species 11: CH2O */
        species[11] =
            +1.99560600e+00
            +6.68132100e-03 * tc[1]
            -2.62895500e-06 * tc[2]
            +4.73715300e-10 * tc[3]
            -3.21251700e-14 * tc[4];
        /*species 12: CH3 */
        species[12] =
            +1.84405200e+00
            +6.13797400e-03 * tc[1]
            -2.23034500e-06 * tc[2]
            +3.78516100e-10 * tc[3]
            -2.45215900e-14 * tc[4];
        /*species 13: CH4 */
        species[13] =
            +6.83479000e-01
            +1.02372400e-02 * tc[1]
            -3.87512900e-06 * tc[2]
            +6.78558500e-10 * tc[3]
            -4.50342300e-14 * tc[4];
        /*species 14: CH3OH */
        species[14] =
            +3.02906100e+00
            +9.37659300e-03 * tc[1]
            -3.05025400e-06 * tc[2]
            +4.35879300e-10 * tc[3]
            -2.22472300e-14 * tc[4];
        /*species 16: CH2CO */
        species[16] =
            +5.03881700e+00
            +5.80484000e-03 * tc[1]
            -1.92095400e-06 * tc[2]
            +2.79448500e-10 * tc[3]
            -1.45886800e-14 * tc[4];
        /*species 18: C2H2 */
        species[18] =
            +3.43677000e+00
            +5.37603900e-03 * tc[1]
            -1.91281700e-06 * tc[2]
            +3.28637900e-10 * tc[3]
            -2.15671000e-14 * tc[4];
        /*species 20: C2H4 */
        species[20] =
            +2.52841900e+00
            +1.14851800e-02 * tc[1]
            -4.41838500e-06 * tc[2]
            +7.84460100e-10 * tc[3]
            -5.26684800e-14 * tc[4];
    }

    /*species with midpoint at T=1357 kelvin */
    if (T < 1357) {
        /*species 4: OH */
        species[4] =
            +2.43586219e+00
            +2.02235804e-04 * tc[1]
            -1.13546412e-07 * tc[2]
            +2.42445149e-10 * tc[3]
            -7.43651031e-14 * tc[4];
    } else {
        /*species 4: OH */
        species[4] =
            +1.62599754e+00
            +1.31992406e-03 * tc[1]
            -3.59724670e-07 * tc[2]
            +4.25630800e-11 * tc[3]
            -1.82048016e-15 * tc[4];
    }

    /*species with midpoint at T=1390 kelvin */
    if (T < 1390) {
        /*species 7: HO2 */
        species[7] =
            +2.18310656e+00
            +3.66767950e-03 * tc[1]
            -9.32385122e-07 * tc[2]
            -3.25852919e-10 * tc[3]
            +1.51139912e-13 * tc[4];
    } else {
        /*species 7: HO2 */
        species[7] =
            +3.10547423e+00
            +2.38452835e-03 * tc[1]
            -8.06347989e-07 * tc[2]
            +1.24191723e-10 * tc[3]
            -7.16400108e-15 * tc[4];
    }

    /*species with midpoint at T=1384 kelvin */
    if (T < 1384) {
        /*species 15: C2H6 */
        species[15] =
            -1.02528543e+00
            +2.40764754e-02 * tc[1]
            -1.11893472e-05 * tc[2]
            +2.08340901e-09 * tc[3]
            -5.29868616e-14 * tc[4];
    } else {
        /*species 15: C2H6 */
        species[15] =
            +5.10683385e+00
            +1.29236361e-02 * tc[1]
            -4.42527196e-06 * tc[2]
            +6.87391726e-10 * tc[3]
            -3.98901732e-14 * tc[4];
    }

    /*species with midpoint at T=1376 kelvin */
    if (T < 1376) {
        /*species 17: HOCHO */
        species[17] =
            +4.35481850e-01
            +1.63363016e-02 * tc[1]
            -1.06257421e-05 * tc[2]
            +3.32132977e-09 * tc[3]
            -4.02176103e-13 * tc[4];
    } else {
        /*species 17: HOCHO */
        species[17] =
            +5.68733013e+00
            +5.14289368e-03 * tc[1]
            -1.82238513e-06 * tc[2]
            +2.89719163e-10 * tc[3]
            -1.70892199e-14 * tc[4];
    }

    /*species with midpoint at T=1388 kelvin */
    if (T < 1388) {
        /*species 19: C3H6 */
        species[19] =
            -6.05384556e-01
            +2.89107662e-02 * tc[1]
            -1.54886808e-05 * tc[2]
            +3.88814209e-09 * tc[3]
            -3.37890352e-13 * tc[4];
    } else {
        /*species 19: C3H6 */
        species[19] =
            +7.01595958e+00
            +1.37023634e-02 * tc[1]
            -4.66249733e-06 * tc[2]
            +7.21254402e-10 * tc[3]
            -4.17370126e-14 * tc[4];
    }

    /*species with midpoint at T=1400 kelvin */
    if (T < 1400) {
        /*species 21: C3H4XA */
        species[21] =
            +1.53983100e+00
            +1.63343700e-02 * tc[1]
            -1.76495000e-06 * tc[2]
            -4.64736500e-09 * tc[3]
            +1.72913100e-12 * tc[4];
    } else {
        /*species 21: C3H4XA */
        species[21] =
            +8.77625600e+00
            +5.30213800e-03 * tc[1]
            -3.70111800e-07 * tc[2]
            -3.02638600e-10 * tc[3]
            +5.08958100e-14 * tc[4];
    }

    /*species with midpoint at T=1398 kelvin */
    if (T < 1398) {
        /*species 22: C4H6 */
        species[22] =
            -2.43095121e+00
            +4.78706062e-02 * tc[1]
            -4.15446800e-05 * tc[2]
            +1.91549552e-08 * tc[3]
            -3.57158507e-12 * tc[4];
    } else {
        /*species 22: C4H6 */
        species[22] =
            +1.01633789e+01
            +1.37163965e-02 * tc[1]
            -4.69715783e-06 * tc[2]
            +7.29693836e-10 * tc[3]
            -4.23486203e-14 * tc[4];
    }

    /*species with midpoint at T=1392 kelvin */
    if (T < 1392) {
        /*species 23: C4H8X1 */
        species[23] =
            -1.83137209e+00
            +4.52580978e-02 * tc[1]
            -2.93658559e-05 * tc[2]
            +1.00220436e-08 * tc[3]
            -1.43191680e-12 * tc[4];
    } else {
        /*species 23: C4H8X1 */
        species[23] =
            +1.03508668e+01
            +1.80617877e-02 * tc[1]
            -6.16093029e-06 * tc[2]
            +9.54652959e-10 * tc[3]
            -5.53089641e-14 * tc[4];
    }

    /*species with midpoint at T=1391 kelvin */
    if (T < 1391) {
        /*species 24: NXC7H16 */
        species[24] =
            -2.26836187e+00
            +8.54355820e-02 * tc[1]
            -5.25346786e-05 * tc[2]
            +1.62945721e-08 * tc[3]
            -2.02394925e-12 * tc[4];
    } else {
        /*species 24: NXC7H16 */
        species[24] =
            +2.12148969e+01
            +3.47675750e-02 * tc[1]
            -1.18407129e-05 * tc[2]
            +1.83298478e-09 * tc[3]
            -1.06130266e-13 * tc[4];
    }
    return;
}


/*compute Cp/R at the given temperature */
/*tc contains precomputed powers of T, tc[0] = log(T) */
AMREX_GPU_HOST_DEVICE void cp_R(double * species, double *  tc)
{

    /*temperature */
    double T = tc[1];

    /*species with midpoint at T=1000 kelvin */
    if (T < 1000) {
        /*species 0: N2 */
        species[0] =
            +3.29867700e+00
            +1.40824000e-03 * tc[1]
            -3.96322200e-06 * tc[2]
            +5.64151500e-09 * tc[3]
            -2.44485500e-12 * tc[4];
        /*species 1: O */
        species[1] =
            +2.94642900e+00
            -1.63816600e-03 * tc[1]
            +2.42103200e-06 * tc[2]
            -1.60284300e-09 * tc[3]
            +3.89069600e-13 * tc[4];
        /*species 2: H2 */
        species[2] =
            +3.29812400e+00
            +8.24944200e-04 * tc[1]
            -8.14301500e-07 * tc[2]
            -9.47543400e-11 * tc[3]
            +4.13487200e-13 * tc[4];
        /*species 3: H */
        species[3] =
            +2.50104422e+00
            +0.00000000e+00 * tc[1]
            +0.00000000e+00 * tc[2]
            +0.00000000e+00 * tc[3]
            +0.00000000e+00 * tc[4];
        /*species 5: H2O */
        species[5] =
            +3.38684200e+00
            +3.47498200e-03 * tc[1]
            -6.35469600e-06 * tc[2]
            +6.96858100e-09 * tc[3]
            -2.50658800e-12 * tc[4];
        /*species 6: O2 */
        species[6] =
            +3.21293600e+00
            +1.12748600e-03 * tc[1]
            -5.75615000e-07 * tc[2]
            +1.31387700e-09 * tc[3]
            -8.76855400e-13 * tc[4];
        /*species 8: H2O2 */
        species[8] =
            +3.38875400e+00
            +6.56922600e-03 * tc[1]
            -1.48501300e-07 * tc[2]
            -4.62580600e-09 * tc[3]
            +2.47151500e-12 * tc[4];
        /*species 9: CO2 */
        species[9] =
            +2.27572500e+00
            +9.92207200e-03 * tc[1]
            -1.04091100e-05 * tc[2]
            +6.86668700e-09 * tc[3]
            -2.11728000e-12 * tc[4];
        /*species 10: CO */
        species[10] =
            +3.26245200e+00
            +1.51194100e-03 * tc[1]
            -3.88175500e-06 * tc[2]
            +5.58194400e-09 * tc[3]
            -2.47495100e-12 * tc[4];
        /*species 11: CH2O */
        species[11] =
            +1.65273100e+00
            +1.26314400e-02 * tc[1]
            -1.88816800e-05 * tc[2]
            +2.05003100e-08 * tc[3]
            -8.41323700e-12 * tc[4];
        /*species 12: CH3 */
        species[12] =
            +2.43044300e+00
            +1.11241000e-02 * tc[1]
            -1.68022000e-05 * tc[2]
            +1.62182900e-08 * tc[3]
            -5.86495300e-12 * tc[4];
        /*species 13: CH4 */
        species[13] =
            +7.78741500e-01
            +1.74766800e-02 * tc[1]
            -2.78340900e-05 * tc[2]
            +3.04970800e-08 * tc[3]
            -1.22393100e-11 * tc[4];
        /*species 14: CH3OH */
        species[14] =
            +2.66011500e+00
            +7.34150800e-03 * tc[1]
            +7.17005100e-06 * tc[2]
            -8.79319400e-09 * tc[3]
            +2.39057000e-12 * tc[4];
        /*species 16: CH2CO */
        species[16] =
            +2.97497100e+00
            +1.21187100e-02 * tc[1]
            -2.34504600e-06 * tc[2]
            -6.46668500e-09 * tc[3]
            +3.90564900e-12 * tc[4];
        /*species 18: C2H2 */
        species[18] =
            +2.01356200e+00
            +1.51904500e-02 * tc[1]
            -1.61631900e-05 * tc[2]
            +9.07899200e-09 * tc[3]
            -1.91274600e-12 * tc[4];
        /*species 20: C2H4 */
        species[20] =
            -8.61488000e-01
            +2.79616300e-02 * tc[1]
            -3.38867700e-05 * tc[2]
            +2.78515200e-08 * tc[3]
            -9.73787900e-12 * tc[4];
    } else {
        /*species 0: N2 */
        species[0] =
            +2.92664000e+00
            +1.48797700e-03 * tc[1]
            -5.68476100e-07 * tc[2]
            +1.00970400e-10 * tc[3]
            -6.75335100e-15 * tc[4];
        /*species 1: O */
        species[1] =
            +2.54206000e+00
            -2.75506200e-05 * tc[1]
            -3.10280300e-09 * tc[2]
            +4.55106700e-12 * tc[3]
            -4.36805200e-16 * tc[4];
        /*species 2: H2 */
        species[2] =
            +2.99142300e+00
            +7.00064400e-04 * tc[1]
            -5.63382900e-08 * tc[2]
            -9.23157800e-12 * tc[3]
            +1.58275200e-15 * tc[4];
        /*species 3: H */
        species[3] =
            +2.50104422e+00
            +0.00000000e+00 * tc[1]
            +0.00000000e+00 * tc[2]
            +0.00000000e+00 * tc[3]
            +0.00000000e+00 * tc[4];
        /*species 5: H2O */
        species[5] =
            +2.67214600e+00
            +3.05629300e-03 * tc[1]
            -8.73026000e-07 * tc[2]
            +1.20099600e-10 * tc[3]
            -6.39161800e-15 * tc[4];
        /*species 6: O2 */
        species[6] =
            +3.69757800e+00
            +6.13519700e-04 * tc[1]
            -1.25884200e-07 * tc[2]
            +1.77528100e-11 * tc[3]
            -1.13643500e-15 * tc[4];
        /*species 8: H2O2 */
        species[8] =
            +4.57316700e+00
            +4.33613600e-03 * tc[1]
            -1.47468900e-06 * tc[2]
            +2.34890400e-10 * tc[3]
            -1.43165400e-14 * tc[4];
        /*species 9: CO2 */
        species[9] =
            +4.45362300e+00
            +3.14016900e-03 * tc[1]
            -1.27841100e-06 * tc[2]
            +2.39399700e-10 * tc[3]
            -1.66903300e-14 * tc[4];
        /*species 10: CO */
        species[10] =
            +3.02507800e+00
            +1.44268900e-03 * tc[1]
            -5.63082800e-07 * tc[2]
            +1.01858100e-10 * tc[3]
            -6.91095200e-15 * tc[4];
        /*species 11: CH2O */
        species[11] =
            +2.99560600e+00
            +6.68132100e-03 * tc[1]
            -2.62895500e-06 * tc[2]
            +4.73715300e-10 * tc[3]
            -3.21251700e-14 * tc[4];
        /*species 12: CH3 */
        species[12] =
            +2.84405200e+00
            +6.13797400e-03 * tc[1]
            -2.23034500e-06 * tc[2]
            +3.78516100e-10 * tc[3]
            -2.45215900e-14 * tc[4];
        /*species 13: CH4 */
        species[13] =
            +1.68347900e+00
            +1.02372400e-02 * tc[1]
            -3.87512900e-06 * tc[2]
            +6.78558500e-10 * tc[3]
            -4.50342300e-14 * tc[4];
        /*species 14: CH3OH */
        species[14] =
            +4.02906100e+00
            +9.37659300e-03 * tc[1]
            -3.05025400e-06 * tc[2]
            +4.35879300e-10 * tc[3]
            -2.22472300e-14 * tc[4];
        /*species 16: CH2CO */
        species[16] =
            +6.03881700e+00
            +5.80484000e-03 * tc[1]
            -1.92095400e-06 * tc[2]
            +2.79448500e-10 * tc[3]
            -1.45886800e-14 * tc[4];
        /*species 18: C2H2 */
        species[18] =
            +4.43677000e+00
            +5.37603900e-03 * tc[1]
            -1.91281700e-06 * tc[2]
            +3.28637900e-10 * tc[3]
            -2.15671000e-14 * tc[4];
        /*species 20: C2H4 */
        species[20] =
            +3.52841900e+00
            +1.14851800e-02 * tc[1]
            -4.41838500e-06 * tc[2]
            +7.84460100e-10 * tc[3]
            -5.26684800e-14 * tc[4];
    }

    /*species with midpoint at T=1357 kelvin */
    if (T < 1357) {
        /*species 4: OH */
        species[4] =
            +3.43586219e+00
            +2.02235804e-04 * tc[1]
            -1.13546412e-07 * tc[2]
            +2.42445149e-10 * tc[3]
            -7.43651031e-14 * tc[4];
    } else {
        /*species 4: OH */
        species[4] =
            +2.62599754e+00
            +1.31992406e-03 * tc[1]
            -3.59724670e-07 * tc[2]
            +4.25630800e-11 * tc[3]
            -1.82048016e-15 * tc[4];
    }

    /*species with midpoint at T=1390 kelvin */
    if (T < 1390) {
        /*species 7: HO2 */
        species[7] =
            +3.18310656e+00
            +3.66767950e-03 * tc[1]
            -9.32385122e-07 * tc[2]
            -3.25852919e-10 * tc[3]
            +1.51139912e-13 * tc[4];
    } else {
        /*species 7: HO2 */
        species[7] =
            +4.10547423e+00
            +2.38452835e-03 * tc[1]
            -8.06347989e-07 * tc[2]
            +1.24191723e-10 * tc[3]
            -7.16400108e-15 * tc[4];
    }

    /*species with midpoint at T=1384 kelvin */
    if (T < 1384) {
        /*species 15: C2H6 */
        species[15] =
            -2.52854344e-02
            +2.40764754e-02 * tc[1]
            -1.11893472e-05 * tc[2]
            +2.08340901e-09 * tc[3]
            -5.29868616e-14 * tc[4];
    } else {
        /*species 15: C2H6 */
        species[15] =
            +6.10683385e+00
            +1.29236361e-02 * tc[1]
            -4.42527196e-06 * tc[2]
            +6.87391726e-10 * tc[3]
            -3.98901732e-14 * tc[4];
    }

    /*species with midpoint at T=1376 kelvin */
    if (T < 1376) {
        /*species 17: HOCHO */
        species[17] =
            +1.43548185e+00
            +1.63363016e-02 * tc[1]
            -1.06257421e-05 * tc[2]
            +3.32132977e-09 * tc[3]
            -4.02176103e-13 * tc[4];
    } else {
        /*species 17: HOCHO */
        species[17] =
            +6.68733013e+00
            +5.14289368e-03 * tc[1]
            -1.82238513e-06 * tc[2]
            +2.89719163e-10 * tc[3]
            -1.70892199e-14 * tc[4];
    }

    /*species with midpoint at T=1388 kelvin */
    if (T < 1388) {
        /*species 19: C3H6 */
        species[19] =
            +3.94615444e-01
            +2.89107662e-02 * tc[1]
            -1.54886808e-05 * tc[2]
            +3.88814209e-09 * tc[3]
            -3.37890352e-13 * tc[4];
    } else {
        /*species 19: C3H6 */
        species[19] =
            +8.01595958e+00
            +1.37023634e-02 * tc[1]
            -4.66249733e-06 * tc[2]
            +7.21254402e-10 * tc[3]
            -4.17370126e-14 * tc[4];
    }

    /*species with midpoint at T=1400 kelvin */
    if (T < 1400) {
        /*species 21: C3H4XA */
        species[21] =
            +2.53983100e+00
            +1.63343700e-02 * tc[1]
            -1.76495000e-06 * tc[2]
            -4.64736500e-09 * tc[3]
            +1.72913100e-12 * tc[4];
    } else {
        /*species 21: C3H4XA */
        species[21] =
            +9.77625600e+00
            +5.30213800e-03 * tc[1]
            -3.70111800e-07 * tc[2]
            -3.02638600e-10 * tc[3]
            +5.08958100e-14 * tc[4];
    }

    /*species with midpoint at T=1398 kelvin */
    if (T < 1398) {
        /*species 22: C4H6 */
        species[22] =
            -1.43095121e+00
            +4.78706062e-02 * tc[1]
            -4.15446800e-05 * tc[2]
            +1.91549552e-08 * tc[3]
            -3.57158507e-12 * tc[4];
    } else {
        /*species 22: C4H6 */
        species[22] =
            +1.11633789e+01
            +1.37163965e-02 * tc[1]
            -4.69715783e-06 * tc[2]
            +7.29693836e-10 * tc[3]
            -4.23486203e-14 * tc[4];
    }

    /*species with midpoint at T=1392 kelvin */
    if (T < 1392) {
        /*species 23: C4H8X1 */
        species[23] =
            -8.31372089e-01
            +4.52580978e-02 * tc[1]
            -2.93658559e-05 * tc[2]
            +1.00220436e-08 * tc[3]
            -1.43191680e-12 * tc[4];
    } else {
        /*species 23: C4H8X1 */
        species[23] =
            +1.13508668e+01
            +1.80617877e-02 * tc[1]
            -6.16093029e-06 * tc[2]
            +9.54652959e-10 * tc[3]
            -5.53089641e-14 * tc[4];
    }

    /*species with midpoint at T=1391 kelvin */
    if (T < 1391) {
        /*species 24: NXC7H16 */
        species[24] =
            -1.26836187e+00
            +8.54355820e-02 * tc[1]
            -5.25346786e-05 * tc[2]
            +1.62945721e-08 * tc[3]
            -2.02394925e-12 * tc[4];
    } else {
        /*species 24: NXC7H16 */
        species[24] =
            +2.22148969e+01
            +3.47675750e-02 * tc[1]
            -1.18407129e-05 * tc[2]
            +1.83298478e-09 * tc[3]
            -1.06130266e-13 * tc[4];
    }
    return;
}


/*compute the e/(RT) at the given temperature */
/*tc contains precomputed powers of T, tc[0] = log(T) */
AMREX_GPU_HOST_DEVICE void speciesInternalEnergy(double * species, double *  tc)
{

    /*temperature */
    double T = tc[1];
    double invT = 1 / T;

    /*species with midpoint at T=1000 kelvin */
    if (T < 1000) {
        /*species 0: N2 */
        species[0] =
            +2.29867700e+00
            +7.04120000e-04 * tc[1]
            -1.32107400e-06 * tc[2]
            +1.41037875e-09 * tc[3]
            -4.88971000e-13 * tc[4]
            -1.02090000e+03 * invT;
        /*species 1: O */
        species[1] =
            +1.94642900e+00
            -8.19083000e-04 * tc[1]
            +8.07010667e-07 * tc[2]
            -4.00710750e-10 * tc[3]
            +7.78139200e-14 * tc[4]
            +2.91476400e+04 * invT;
        /*species 2: H2 */
        species[2] =
            +2.29812400e+00
            +4.12472100e-04 * tc[1]
            -2.71433833e-07 * tc[2]
            -2.36885850e-11 * tc[3]
            +8.26974400e-14 * tc[4]
            -1.01252100e+03 * invT;
        /*species 3: H */
        species[3] =
            +1.50104422e+00
            +0.00000000e+00 * tc[1]
            +0.00000000e+00 * tc[2]
            +0.00000000e+00 * tc[3]
            +0.00000000e+00 * tc[4]
            +2.54747466e+04 * invT;
        /*species 5: H2O */
        species[5] =
            +2.38684200e+00
            +1.73749100e-03 * tc[1]
            -2.11823200e-06 * tc[2]
            +1.74214525e-09 * tc[3]
            -5.01317600e-13 * tc[4]
            -3.02081100e+04 * invT;
        /*species 6: O2 */
        species[6] =
            +2.21293600e+00
            +5.63743000e-04 * tc[1]
            -1.91871667e-07 * tc[2]
            +3.28469250e-10 * tc[3]
            -1.75371080e-13 * tc[4]
            -1.00524900e+03 * invT;
        /*species 8: H2O2 */
        species[8] =
            +2.38875400e+00
            +3.28461300e-03 * tc[1]
            -4.95004333e-08 * tc[2]
            -1.15645150e-09 * tc[3]
            +4.94303000e-13 * tc[4]
            -1.76631500e+04 * invT;
        /*species 9: CO2 */
        species[9] =
            +1.27572500e+00
            +4.96103600e-03 * tc[1]
            -3.46970333e-06 * tc[2]
            +1.71667175e-09 * tc[3]
            -4.23456000e-13 * tc[4]
            -4.83731400e+04 * invT;
        /*species 10: CO */
        species[10] =
            +2.26245200e+00
            +7.55970500e-04 * tc[1]
            -1.29391833e-06 * tc[2]
            +1.39548600e-09 * tc[3]
            -4.94990200e-13 * tc[4]
            -1.43105400e+04 * invT;
        /*species 11: CH2O */
        species[11] =
            +6.52731000e-01
            +6.31572000e-03 * tc[1]
            -6.29389333e-06 * tc[2]
            +5.12507750e-09 * tc[3]
            -1.68264740e-12 * tc[4]
            -1.48654000e+04 * invT;
        /*species 12: CH3 */
        species[12] =
            +1.43044300e+00
            +5.56205000e-03 * tc[1]
            -5.60073333e-06 * tc[2]
            +4.05457250e-09 * tc[3]
            -1.17299060e-12 * tc[4]
            +1.64237800e+04 * invT;
        /*species 13: CH4 */
        species[13] =
            -2.21258500e-01
            +8.73834000e-03 * tc[1]
            -9.27803000e-06 * tc[2]
            +7.62427000e-09 * tc[3]
            -2.44786200e-12 * tc[4]
            -9.82522900e+03 * invT;
        /*species 14: CH3OH */
        species[14] =
            +1.66011500e+00
            +3.67075400e-03 * tc[1]
            +2.39001700e-06 * tc[2]
            -2.19829850e-09 * tc[3]
            +4.78114000e-13 * tc[4]
            -2.53534800e+04 * invT;
        /*species 16: CH2CO */
        species[16] =
            +1.97497100e+00
            +6.05935500e-03 * tc[1]
            -7.81682000e-07 * tc[2]
            -1.61667125e-09 * tc[3]
            +7.81129800e-13 * tc[4]
            -7.63263700e+03 * invT;
        /*species 18: C2H2 */
        species[18] =
            +1.01356200e+00
            +7.59522500e-03 * tc[1]
            -5.38773000e-06 * tc[2]
            +2.26974800e-09 * tc[3]
            -3.82549200e-13 * tc[4]
            +2.61244400e+04 * invT;
        /*species 20: C2H4 */
        species[20] =
            -1.86148800e+00
            +1.39808150e-02 * tc[1]
            -1.12955900e-05 * tc[2]
            +6.96288000e-09 * tc[3]
            -1.94757580e-12 * tc[4]
            +5.57304600e+03 * invT;
    } else {
        /*species 0: N2 */
        species[0] =
            +1.92664000e+00
            +7.43988500e-04 * tc[1]
            -1.89492033e-07 * tc[2]
            +2.52426000e-11 * tc[3]
            -1.35067020e-15 * tc[4]
            -9.22797700e+02 * invT;
        /*species 1: O */
        species[1] =
            +1.54206000e+00
            -1.37753100e-05 * tc[1]
            -1.03426767e-09 * tc[2]
            +1.13776675e-12 * tc[3]
            -8.73610400e-17 * tc[4]
            +2.92308000e+04 * invT;
        /*species 2: H2 */
        species[2] =
            +1.99142300e+00
            +3.50032200e-04 * tc[1]
            -1.87794300e-08 * tc[2]
            -2.30789450e-12 * tc[3]
            +3.16550400e-16 * tc[4]
            -8.35034000e+02 * invT;
        /*species 3: H */
        species[3] =
            +1.50104422e+00
            +0.00000000e+00 * tc[1]
            +0.00000000e+00 * tc[2]
            +0.00000000e+00 * tc[3]
            +0.00000000e+00 * tc[4]
            +2.54747466e+04 * invT;
        /*species 5: H2O */
        species[5] =
            +1.67214600e+00
            +1.52814650e-03 * tc[1]
            -2.91008667e-07 * tc[2]
            +3.00249000e-11 * tc[3]
            -1.27832360e-15 * tc[4]
            -2.98992100e+04 * invT;
        /*species 6: O2 */
        species[6] =
            +2.69757800e+00
            +3.06759850e-04 * tc[1]
            -4.19614000e-08 * tc[2]
            +4.43820250e-12 * tc[3]
            -2.27287000e-16 * tc[4]
            -1.23393000e+03 * invT;
        /*species 8: H2O2 */
        species[8] =
            +3.57316700e+00
            +2.16806800e-03 * tc[1]
            -4.91563000e-07 * tc[2]
            +5.87226000e-11 * tc[3]
            -2.86330800e-15 * tc[4]
            -1.80069600e+04 * invT;
        /*species 9: CO2 */
        species[9] =
            +3.45362300e+00
            +1.57008450e-03 * tc[1]
            -4.26137000e-07 * tc[2]
            +5.98499250e-11 * tc[3]
            -3.33806600e-15 * tc[4]
            -4.89669600e+04 * invT;
        /*species 10: CO */
        species[10] =
            +2.02507800e+00
            +7.21344500e-04 * tc[1]
            -1.87694267e-07 * tc[2]
            +2.54645250e-11 * tc[3]
            -1.38219040e-15 * tc[4]
            -1.42683500e+04 * invT;
        /*species 11: CH2O */
        species[11] =
            +1.99560600e+00
            +3.34066050e-03 * tc[1]
            -8.76318333e-07 * tc[2]
            +1.18428825e-10 * tc[3]
            -6.42503400e-15 * tc[4]
            -1.53203700e+04 * invT;
        /*species 12: CH3 */
        species[12] =
            +1.84405200e+00
            +3.06898700e-03 * tc[1]
            -7.43448333e-07 * tc[2]
            +9.46290250e-11 * tc[3]
            -4.90431800e-15 * tc[4]
            +1.64378100e+04 * invT;
        /*species 13: CH4 */
        species[13] =
            +6.83479000e-01
            +5.11862000e-03 * tc[1]
            -1.29170967e-06 * tc[2]
            +1.69639625e-10 * tc[3]
            -9.00684600e-15 * tc[4]
            -1.00807900e+04 * invT;
        /*species 14: CH3OH */
        species[14] =
            +3.02906100e+00
            +4.68829650e-03 * tc[1]
            -1.01675133e-06 * tc[2]
            +1.08969825e-10 * tc[3]
            -4.44944600e-15 * tc[4]
            -2.61579100e+04 * invT;
        /*species 16: CH2CO */
        species[16] =
            +5.03881700e+00
            +2.90242000e-03 * tc[1]
            -6.40318000e-07 * tc[2]
            +6.98621250e-11 * tc[3]
            -2.91773600e-15 * tc[4]
            -8.58340200e+03 * invT;
        /*species 18: C2H2 */
        species[18] =
            +3.43677000e+00
            +2.68801950e-03 * tc[1]
            -6.37605667e-07 * tc[2]
            +8.21594750e-11 * tc[3]
            -4.31342000e-15 * tc[4]
            +2.56676600e+04 * invT;
        /*species 20: C2H4 */
        species[20] =
            +2.52841900e+00
            +5.74259000e-03 * tc[1]
            -1.47279500e-06 * tc[2]
            +1.96115025e-10 * tc[3]
            -1.05336960e-14 * tc[4]
            +4.42828900e+03 * invT;
    }

    /*species with midpoint at T=1357 kelvin */
    if (T < 1357) {
        /*species 4: OH */
        species[4] =
            +2.43586219e+00
            +1.01117902e-04 * tc[1]
            -3.78488040e-08 * tc[2]
            +6.06112872e-11 * tc[3]
            -1.48730206e-14 * tc[4]
            +3.74321252e+03 * invT;
    } else {
        /*species 4: OH */
        species[4] =
            +1.62599754e+00
            +6.59962030e-04 * tc[1]
            -1.19908223e-07 * tc[2]
            +1.06407700e-11 * tc[3]
            -3.64096032e-16 * tc[4]
            +4.12085374e+03 * invT;
    }

    /*species with midpoint at T=1390 kelvin */
    if (T < 1390) {
        /*species 7: HO2 */
        species[7] =
            +2.18310656e+00
            +1.83383975e-03 * tc[1]
            -3.10795041e-07 * tc[2]
            -8.14632298e-11 * tc[3]
            +3.02279824e-14 * tc[4]
            +8.09181013e+02 * invT;
    } else {
        /*species 7: HO2 */
        species[7] =
            +3.10547423e+00
            +1.19226417e-03 * tc[1]
            -2.68782663e-07 * tc[2]
            +3.10479308e-11 * tc[3]
            -1.43280022e-15 * tc[4]
            +3.98127689e+02 * invT;
    }

    /*species with midpoint at T=1384 kelvin */
    if (T < 1384) {
        /*species 15: C2H6 */
        species[15] =
            -1.02528543e+00
            +1.20382377e-02 * tc[1]
            -3.72978240e-06 * tc[2]
            +5.20852252e-10 * tc[3]
            -1.05973723e-14 * tc[4]
            -1.12345534e+04 * invT;
    } else {
        /*species 15: C2H6 */
        species[15] =
            +5.10683385e+00
            +6.46181805e-03 * tc[1]
            -1.47509065e-06 * tc[2]
            +1.71847932e-10 * tc[3]
            -7.97803464e-15 * tc[4]
            -1.37500014e+04 * invT;
    }

    /*species with midpoint at T=1376 kelvin */
    if (T < 1376) {
        /*species 17: HOCHO */
        species[17] =
            +4.35481850e-01
            +8.16815080e-03 * tc[1]
            -3.54191403e-06 * tc[2]
            +8.30332443e-10 * tc[3]
            -8.04352206e-14 * tc[4]
            -4.64616504e+04 * invT;
    } else {
        /*species 17: HOCHO */
        species[17] =
            +5.68733013e+00
            +2.57144684e-03 * tc[1]
            -6.07461710e-07 * tc[2]
            +7.24297908e-11 * tc[3]
            -3.41784398e-15 * tc[4]
            -4.83995400e+04 * invT;
    }

    /*species with midpoint at T=1388 kelvin */
    if (T < 1388) {
        /*species 19: C3H6 */
        species[19] =
            -6.05384556e-01
            +1.44553831e-02 * tc[1]
            -5.16289360e-06 * tc[2]
            +9.72035522e-10 * tc[3]
            -6.75780704e-14 * tc[4]
            +1.06688164e+03 * invT;
    } else {
        /*species 19: C3H6 */
        species[19] =
            +7.01595958e+00
            +6.85118170e-03 * tc[1]
            -1.55416578e-06 * tc[2]
            +1.80313601e-10 * tc[3]
            -8.34740252e-15 * tc[4]
            -1.87821271e+03 * invT;
    }

    /*species with midpoint at T=1400 kelvin */
    if (T < 1400) {
        /*species 21: C3H4XA */
        species[21] =
            +1.53983100e+00
            +8.16718500e-03 * tc[1]
            -5.88316667e-07 * tc[2]
            -1.16184125e-09 * tc[3]
            +3.45826200e-13 * tc[4]
            +2.25124300e+04 * invT;
    } else {
        /*species 21: C3H4XA */
        species[21] =
            +8.77625600e+00
            +2.65106900e-03 * tc[1]
            -1.23370600e-07 * tc[2]
            -7.56596500e-11 * tc[3]
            +1.01791620e-14 * tc[4]
            +1.95497200e+04 * invT;
    }

    /*species with midpoint at T=1398 kelvin */
    if (T < 1398) {
        /*species 22: C4H6 */
        species[22] =
            -2.43095121e+00
            +2.39353031e-02 * tc[1]
            -1.38482267e-05 * tc[2]
            +4.78873880e-09 * tc[3]
            -7.14317014e-13 * tc[4]
            +1.17551314e+04 * invT;
    } else {
        /*species 22: C4H6 */
        species[22] =
            +1.01633789e+01
            +6.85819825e-03 * tc[1]
            -1.56571928e-06 * tc[2]
            +1.82423459e-10 * tc[3]
            -8.46972406e-15 * tc[4]
            +7.79039770e+03 * invT;
    }

    /*species with midpoint at T=1392 kelvin */
    if (T < 1392) {
        /*species 23: C4H8X1 */
        species[23] =
            -1.83137209e+00
            +2.26290489e-02 * tc[1]
            -9.78861863e-06 * tc[2]
            +2.50551090e-09 * tc[3]
            -2.86383360e-13 * tc[4]
            -1.57875035e+03 * invT;
    } else {
        /*species 23: C4H8X1 */
        species[23] =
            +1.03508668e+01
            +9.03089385e-03 * tc[1]
            -2.05364343e-06 * tc[2]
            +2.38663240e-10 * tc[3]
            -1.10617928e-14 * tc[4]
            -5.97871038e+03 * invT;
    }

    /*species with midpoint at T=1391 kelvin */
    if (T < 1391) {
        /*species 24: NXC7H16 */
        species[24] =
            -2.26836187e+00
            +4.27177910e-02 * tc[1]
            -1.75115595e-05 * tc[2]
            +4.07364302e-09 * tc[3]
            -4.04789850e-13 * tc[4]
            -2.56586565e+04 * invT;
    } else {
        /*species 24: NXC7H16 */
        species[24] =
            +2.12148969e+01
            +1.73837875e-02 * tc[1]
            -3.94690430e-06 * tc[2]
            +4.58246195e-10 * tc[3]
            -2.12260532e-14 * tc[4]
            -3.42760081e+04 * invT;
    }
    return;
}


/*compute the h/(RT) at the given temperature (Eq 20) */
/*tc contains precomputed powers of T, tc[0] = log(T) */
AMREX_GPU_HOST_DEVICE void speciesEnthalpy(double * species, double *  tc)
{

    /*temperature */
    double T = tc[1];
    double invT = 1 / T;

    /*species with midpoint at T=1000 kelvin */
    if (T < 1000) {
        /*species 0: N2 */
        species[0] =
            +3.29867700e+00
            +7.04120000e-04 * tc[1]
            -1.32107400e-06 * tc[2]
            +1.41037875e-09 * tc[3]
            -4.88971000e-13 * tc[4]
            -1.02090000e+03 * invT;
        /*species 1: O */
        species[1] =
            +2.94642900e+00
            -8.19083000e-04 * tc[1]
            +8.07010667e-07 * tc[2]
            -4.00710750e-10 * tc[3]
            +7.78139200e-14 * tc[4]
            +2.91476400e+04 * invT;
        /*species 2: H2 */
        species[2] =
            +3.29812400e+00
            +4.12472100e-04 * tc[1]
            -2.71433833e-07 * tc[2]
            -2.36885850e-11 * tc[3]
            +8.26974400e-14 * tc[4]
            -1.01252100e+03 * invT;
        /*species 3: H */
        species[3] =
            +2.50104422e+00
            +0.00000000e+00 * tc[1]
            +0.00000000e+00 * tc[2]
            +0.00000000e+00 * tc[3]
            +0.00000000e+00 * tc[4]
            +2.54747466e+04 * invT;
        /*species 5: H2O */
        species[5] =
            +3.38684200e+00
            +1.73749100e-03 * tc[1]
            -2.11823200e-06 * tc[2]
            +1.74214525e-09 * tc[3]
            -5.01317600e-13 * tc[4]
            -3.02081100e+04 * invT;
        /*species 6: O2 */
        species[6] =
            +3.21293600e+00
            +5.63743000e-04 * tc[1]
            -1.91871667e-07 * tc[2]
            +3.28469250e-10 * tc[3]
            -1.75371080e-13 * tc[4]
            -1.00524900e+03 * invT;
        /*species 8: H2O2 */
        species[8] =
            +3.38875400e+00
            +3.28461300e-03 * tc[1]
            -4.95004333e-08 * tc[2]
            -1.15645150e-09 * tc[3]
            +4.94303000e-13 * tc[4]
            -1.76631500e+04 * invT;
        /*species 9: CO2 */
        species[9] =
            +2.27572500e+00
            +4.96103600e-03 * tc[1]
            -3.46970333e-06 * tc[2]
            +1.71667175e-09 * tc[3]
            -4.23456000e-13 * tc[4]
            -4.83731400e+04 * invT;
        /*species 10: CO */
        species[10] =
            +3.26245200e+00
            +7.55970500e-04 * tc[1]
            -1.29391833e-06 * tc[2]
            +1.39548600e-09 * tc[3]
            -4.94990200e-13 * tc[4]
            -1.43105400e+04 * invT;
        /*species 11: CH2O */
        species[11] =
            +1.65273100e+00
            +6.31572000e-03 * tc[1]
            -6.29389333e-06 * tc[2]
            +5.12507750e-09 * tc[3]
            -1.68264740e-12 * tc[4]
            -1.48654000e+04 * invT;
        /*species 12: CH3 */
        species[12] =
            +2.43044300e+00
            +5.56205000e-03 * tc[1]
            -5.60073333e-06 * tc[2]
            +4.05457250e-09 * tc[3]
            -1.17299060e-12 * tc[4]
            +1.64237800e+04 * invT;
        /*species 13: CH4 */
        species[13] =
            +7.78741500e-01
            +8.73834000e-03 * tc[1]
            -9.27803000e-06 * tc[2]
            +7.62427000e-09 * tc[3]
            -2.44786200e-12 * tc[4]
            -9.82522900e+03 * invT;
        /*species 14: CH3OH */
        species[14] =
            +2.66011500e+00
            +3.67075400e-03 * tc[1]
            +2.39001700e-06 * tc[2]
            -2.19829850e-09 * tc[3]
            +4.78114000e-13 * tc[4]
            -2.53534800e+04 * invT;
        /*species 16: CH2CO */
        species[16] =
            +2.97497100e+00
            +6.05935500e-03 * tc[1]
            -7.81682000e-07 * tc[2]
            -1.61667125e-09 * tc[3]
            +7.81129800e-13 * tc[4]
            -7.63263700e+03 * invT;
        /*species 18: C2H2 */
        species[18] =
            +2.01356200e+00
            +7.59522500e-03 * tc[1]
            -5.38773000e-06 * tc[2]
            +2.26974800e-09 * tc[3]
            -3.82549200e-13 * tc[4]
            +2.61244400e+04 * invT;
        /*species 20: C2H4 */
        species[20] =
            -8.61488000e-01
            +1.39808150e-02 * tc[1]
            -1.12955900e-05 * tc[2]
            +6.96288000e-09 * tc[3]
            -1.94757580e-12 * tc[4]
            +5.57304600e+03 * invT;
    } else {
        /*species 0: N2 */
        species[0] =
            +2.92664000e+00
            +7.43988500e-04 * tc[1]
            -1.89492033e-07 * tc[2]
            +2.52426000e-11 * tc[3]
            -1.35067020e-15 * tc[4]
            -9.22797700e+02 * invT;
        /*species 1: O */
        species[1] =
            +2.54206000e+00
            -1.37753100e-05 * tc[1]
            -1.03426767e-09 * tc[2]
            +1.13776675e-12 * tc[3]
            -8.73610400e-17 * tc[4]
            +2.92308000e+04 * invT;
        /*species 2: H2 */
        species[2] =
            +2.99142300e+00
            +3.50032200e-04 * tc[1]
            -1.87794300e-08 * tc[2]
            -2.30789450e-12 * tc[3]
            +3.16550400e-16 * tc[4]
            -8.35034000e+02 * invT;
        /*species 3: H */
        species[3] =
            +2.50104422e+00
            +0.00000000e+00 * tc[1]
            +0.00000000e+00 * tc[2]
            +0.00000000e+00 * tc[3]
            +0.00000000e+00 * tc[4]
            +2.54747466e+04 * invT;
        /*species 5: H2O */
        species[5] =
            +2.67214600e+00
            +1.52814650e-03 * tc[1]
            -2.91008667e-07 * tc[2]
            +3.00249000e-11 * tc[3]
            -1.27832360e-15 * tc[4]
            -2.98992100e+04 * invT;
        /*species 6: O2 */
        species[6] =
            +3.69757800e+00
            +3.06759850e-04 * tc[1]
            -4.19614000e-08 * tc[2]
            +4.43820250e-12 * tc[3]
            -2.27287000e-16 * tc[4]
            -1.23393000e+03 * invT;
        /*species 8: H2O2 */
        species[8] =
            +4.57316700e+00
            +2.16806800e-03 * tc[1]
            -4.91563000e-07 * tc[2]
            +5.87226000e-11 * tc[3]
            -2.86330800e-15 * tc[4]
            -1.80069600e+04 * invT;
        /*species 9: CO2 */
        species[9] =
            +4.45362300e+00
            +1.57008450e-03 * tc[1]
            -4.26137000e-07 * tc[2]
            +5.98499250e-11 * tc[3]
            -3.33806600e-15 * tc[4]
            -4.89669600e+04 * invT;
        /*species 10: CO */
        species[10] =
            +3.02507800e+00
            +7.21344500e-04 * tc[1]
            -1.87694267e-07 * tc[2]
            +2.54645250e-11 * tc[3]
            -1.38219040e-15 * tc[4]
            -1.42683500e+04 * invT;
        /*species 11: CH2O */
        species[11] =
            +2.99560600e+00
            +3.34066050e-03 * tc[1]
            -8.76318333e-07 * tc[2]
            +1.18428825e-10 * tc[3]
            -6.42503400e-15 * tc[4]
            -1.53203700e+04 * invT;
        /*species 12: CH3 */
        species[12] =
            +2.84405200e+00
            +3.06898700e-03 * tc[1]
            -7.43448333e-07 * tc[2]
            +9.46290250e-11 * tc[3]
            -4.90431800e-15 * tc[4]
            +1.64378100e+04 * invT;
        /*species 13: CH4 */
        species[13] =
            +1.68347900e+00
            +5.11862000e-03 * tc[1]
            -1.29170967e-06 * tc[2]
            +1.69639625e-10 * tc[3]
            -9.00684600e-15 * tc[4]
            -1.00807900e+04 * invT;
        /*species 14: CH3OH */
        species[14] =
            +4.02906100e+00
            +4.68829650e-03 * tc[1]
            -1.01675133e-06 * tc[2]
            +1.08969825e-10 * tc[3]
            -4.44944600e-15 * tc[4]
            -2.61579100e+04 * invT;
        /*species 16: CH2CO */
        species[16] =
            +6.03881700e+00
            +2.90242000e-03 * tc[1]
            -6.40318000e-07 * tc[2]
            +6.98621250e-11 * tc[3]
            -2.91773600e-15 * tc[4]
            -8.58340200e+03 * invT;
        /*species 18: C2H2 */
        species[18] =
            +4.43677000e+00
            +2.68801950e-03 * tc[1]
            -6.37605667e-07 * tc[2]
            +8.21594750e-11 * tc[3]
            -4.31342000e-15 * tc[4]
            +2.56676600e+04 * invT;
        /*species 20: C2H4 */
        species[20] =
            +3.52841900e+00
            +5.74259000e-03 * tc[1]
            -1.47279500e-06 * tc[2]
            +1.96115025e-10 * tc[3]
            -1.05336960e-14 * tc[4]
            +4.42828900e+03 * invT;
    }

    /*species with midpoint at T=1357 kelvin */
    if (T < 1357) {
        /*species 4: OH */
        species[4] =
            +3.43586219e+00
            +1.01117902e-04 * tc[1]
            -3.78488040e-08 * tc[2]
            +6.06112872e-11 * tc[3]
            -1.48730206e-14 * tc[4]
            +3.74321252e+03 * invT;
    } else {
        /*species 4: OH */
        species[4] =
            +2.62599754e+00
            +6.59962030e-04 * tc[1]
            -1.19908223e-07 * tc[2]
            +1.06407700e-11 * tc[3]
            -3.64096032e-16 * tc[4]
            +4.12085374e+03 * invT;
    }

    /*species with midpoint at T=1390 kelvin */
    if (T < 1390) {
        /*species 7: HO2 */
        species[7] =
            +3.18310656e+00
            +1.83383975e-03 * tc[1]
            -3.10795041e-07 * tc[2]
            -8.14632298e-11 * tc[3]
            +3.02279824e-14 * tc[4]
            +8.09181013e+02 * invT;
    } else {
        /*species 7: HO2 */
        species[7] =
            +4.10547423e+00
            +1.19226417e-03 * tc[1]
            -2.68782663e-07 * tc[2]
            +3.10479308e-11 * tc[3]
            -1.43280022e-15 * tc[4]
            +3.98127689e+02 * invT;
    }

    /*species with midpoint at T=1384 kelvin */
    if (T < 1384) {
        /*species 15: C2H6 */
        species[15] =
            -2.52854344e-02
            +1.20382377e-02 * tc[1]
            -3.72978240e-06 * tc[2]
            +5.20852252e-10 * tc[3]
            -1.05973723e-14 * tc[4]
            -1.12345534e+04 * invT;
    } else {
        /*species 15: C2H6 */
        species[15] =
            +6.10683385e+00
            +6.46181805e-03 * tc[1]
            -1.47509065e-06 * tc[2]
            +1.71847932e-10 * tc[3]
            -7.97803464e-15 * tc[4]
            -1.37500014e+04 * invT;
    }

    /*species with midpoint at T=1376 kelvin */
    if (T < 1376) {
        /*species 17: HOCHO */
        species[17] =
            +1.43548185e+00
            +8.16815080e-03 * tc[1]
            -3.54191403e-06 * tc[2]
            +8.30332443e-10 * tc[3]
            -8.04352206e-14 * tc[4]
            -4.64616504e+04 * invT;
    } else {
        /*species 17: HOCHO */
        species[17] =
            +6.68733013e+00
            +2.57144684e-03 * tc[1]
            -6.07461710e-07 * tc[2]
            +7.24297908e-11 * tc[3]
            -3.41784398e-15 * tc[4]
            -4.83995400e+04 * invT;
    }

    /*species with midpoint at T=1388 kelvin */
    if (T < 1388) {
        /*species 19: C3H6 */
        species[19] =
            +3.94615444e-01
            +1.44553831e-02 * tc[1]
            -5.16289360e-06 * tc[2]
            +9.72035522e-10 * tc[3]
            -6.75780704e-14 * tc[4]
            +1.06688164e+03 * invT;
    } else {
        /*species 19: C3H6 */
        species[19] =
            +8.01595958e+00
            +6.85118170e-03 * tc[1]
            -1.55416578e-06 * tc[2]
            +1.80313601e-10 * tc[3]
            -8.34740252e-15 * tc[4]
            -1.87821271e+03 * invT;
    }

    /*species with midpoint at T=1400 kelvin */
    if (T < 1400) {
        /*species 21: C3H4XA */
        species[21] =
            +2.53983100e+00
            +8.16718500e-03 * tc[1]
            -5.88316667e-07 * tc[2]
            -1.16184125e-09 * tc[3]
            +3.45826200e-13 * tc[4]
            +2.25124300e+04 * invT;
    } else {
        /*species 21: C3H4XA */
        species[21] =
            +9.77625600e+00
            +2.65106900e-03 * tc[1]
            -1.23370600e-07 * tc[2]
            -7.56596500e-11 * tc[3]
            +1.01791620e-14 * tc[4]
            +1.95497200e+04 * invT;
    }

    /*species with midpoint at T=1398 kelvin */
    if (T < 1398) {
        /*species 22: C4H6 */
        species[22] =
            -1.43095121e+00
            +2.39353031e-02 * tc[1]
            -1.38482267e-05 * tc[2]
            +4.78873880e-09 * tc[3]
            -7.14317014e-13 * tc[4]
            +1.17551314e+04 * invT;
    } else {
        /*species 22: C4H6 */
        species[22] =
            +1.11633789e+01
            +6.85819825e-03 * tc[1]
            -1.56571928e-06 * tc[2]
            +1.82423459e-10 * tc[3]
            -8.46972406e-15 * tc[4]
            +7.79039770e+03 * invT;
    }

    /*species with midpoint at T=1392 kelvin */
    if (T < 1392) {
        /*species 23: C4H8X1 */
        species[23] =
            -8.31372089e-01
            +2.26290489e-02 * tc[1]
            -9.78861863e-06 * tc[2]
            +2.50551090e-09 * tc[3]
            -2.86383360e-13 * tc[4]
            -1.57875035e+03 * invT;
    } else {
        /*species 23: C4H8X1 */
        species[23] =
            +1.13508668e+01
            +9.03089385e-03 * tc[1]
            -2.05364343e-06 * tc[2]
            +2.38663240e-10 * tc[3]
            -1.10617928e-14 * tc[4]
            -5.97871038e+03 * invT;
    }

    /*species with midpoint at T=1391 kelvin */
    if (T < 1391) {
        /*species 24: NXC7H16 */
        species[24] =
            -1.26836187e+00
            +4.27177910e-02 * tc[1]
            -1.75115595e-05 * tc[2]
            +4.07364302e-09 * tc[3]
            -4.04789850e-13 * tc[4]
            -2.56586565e+04 * invT;
    } else {
        /*species 24: NXC7H16 */
        species[24] =
            +2.22148969e+01
            +1.73837875e-02 * tc[1]
            -3.94690430e-06 * tc[2]
            +4.58246195e-10 * tc[3]
            -2.12260532e-14 * tc[4]
            -3.42760081e+04 * invT;
    }
    return;
}


/*compute the S/R at the given temperature (Eq 21) */
/*tc contains precomputed powers of T, tc[0] = log(T) */
AMREX_GPU_HOST_DEVICE void speciesEntropy(double * species, double *  tc)
{

    /*temperature */
    double T = tc[1];

    /*species with midpoint at T=1000 kelvin */
    if (T < 1000) {
        /*species 0: N2 */
        species[0] =
            +3.29867700e+00 * tc[0]
            +1.40824000e-03 * tc[1]
            -1.98161100e-06 * tc[2]
            +1.88050500e-09 * tc[3]
            -6.11213750e-13 * tc[4]
            +3.95037200e+00 ;
        /*species 1: O */
        species[1] =
            +2.94642900e+00 * tc[0]
            -1.63816600e-03 * tc[1]
            +1.21051600e-06 * tc[2]
            -5.34281000e-10 * tc[3]
            +9.72674000e-14 * tc[4]
            +2.96399500e+00 ;
        /*species 2: H2 */
        species[2] =
            +3.29812400e+00 * tc[0]
            +8.24944200e-04 * tc[1]
            -4.07150750e-07 * tc[2]
            -3.15847800e-11 * tc[3]
            +1.03371800e-13 * tc[4]
            -3.29409400e+00 ;
        /*species 3: H */
        species[3] =
            +2.50104422e+00 * tc[0]
            +0.00000000e+00 * tc[1]
            +0.00000000e+00 * tc[2]
            +0.00000000e+00 * tc[3]
            +0.00000000e+00 * tc[4]
            -4.65341317e-01 ;
        /*species 5: H2O */
        species[5] =
            +3.38684200e+00 * tc[0]
            +3.47498200e-03 * tc[1]
            -3.17734800e-06 * tc[2]
            +2.32286033e-09 * tc[3]
            -6.26647000e-13 * tc[4]
            +2.59023300e+00 ;
        /*species 6: O2 */
        species[6] =
            +3.21293600e+00 * tc[0]
            +1.12748600e-03 * tc[1]
            -2.87807500e-07 * tc[2]
            +4.37959000e-10 * tc[3]
            -2.19213850e-13 * tc[4]
            +6.03473800e+00 ;
        /*species 8: H2O2 */
        species[8] =
            +3.38875400e+00 * tc[0]
            +6.56922600e-03 * tc[1]
            -7.42506500e-08 * tc[2]
            -1.54193533e-09 * tc[3]
            +6.17878750e-13 * tc[4]
            +6.78536300e+00 ;
        /*species 9: CO2 */
        species[9] =
            +2.27572500e+00 * tc[0]
            +9.92207200e-03 * tc[1]
            -5.20455500e-06 * tc[2]
            +2.28889567e-09 * tc[3]
            -5.29320000e-13 * tc[4]
            +1.01884900e+01 ;
        /*species 10: CO */
        species[10] =
            +3.26245200e+00 * tc[0]
            +1.51194100e-03 * tc[1]
            -1.94087750e-06 * tc[2]
            +1.86064800e-09 * tc[3]
            -6.18737750e-13 * tc[4]
            +4.84889700e+00 ;
        /*species 11: CH2O */
        species[11] =
            +1.65273100e+00 * tc[0]
            +1.26314400e-02 * tc[1]
            -9.44084000e-06 * tc[2]
            +6.83343667e-09 * tc[3]
            -2.10330925e-12 * tc[4]
            +1.37848200e+01 ;
        /*species 12: CH3 */
        species[12] =
            +2.43044300e+00 * tc[0]
            +1.11241000e-02 * tc[1]
            -8.40110000e-06 * tc[2]
            +5.40609667e-09 * tc[3]
            -1.46623825e-12 * tc[4]
            +6.78979400e+00 ;
        /*species 13: CH4 */
        species[13] =
            +7.78741500e-01 * tc[0]
            +1.74766800e-02 * tc[1]
            -1.39170450e-05 * tc[2]
            +1.01656933e-08 * tc[3]
            -3.05982750e-12 * tc[4]
            +1.37221900e+01 ;
        /*species 14: CH3OH */
        species[14] =
            +2.66011500e+00 * tc[0]
            +7.34150800e-03 * tc[1]
            +3.58502550e-06 * tc[2]
            -2.93106467e-09 * tc[3]
            +5.97642500e-13 * tc[4]
            +1.12326300e+01 ;
        /*species 16: CH2CO */
        species[16] =
            +2.97497100e+00 * tc[0]
            +1.21187100e-02 * tc[1]
            -1.17252300e-06 * tc[2]
            -2.15556167e-09 * tc[3]
            +9.76412250e-13 * tc[4]
            +8.67355300e+00 ;
        /*species 18: C2H2 */
        species[18] =
            +2.01356200e+00 * tc[0]
            +1.51904500e-02 * tc[1]
            -8.08159500e-06 * tc[2]
            +3.02633067e-09 * tc[3]
            -4.78186500e-13 * tc[4]
            +8.80537800e+00 ;
        /*species 20: C2H4 */
        species[20] =
            -8.61488000e-01 * tc[0]
            +2.79616300e-02 * tc[1]
            -1.69433850e-05 * tc[2]
            +9.28384000e-09 * tc[3]
            -2.43446975e-12 * tc[4]
            +2.42114900e+01 ;
    } else {
        /*species 0: N2 */
        species[0] =
            +2.92664000e+00 * tc[0]
            +1.48797700e-03 * tc[1]
            -2.84238050e-07 * tc[2]
            +3.36568000e-11 * tc[3]
            -1.68833775e-15 * tc[4]
            +5.98052800e+00 ;
        /*species 1: O */
        species[1] =
            +2.54206000e+00 * tc[0]
            -2.75506200e-05 * tc[1]
            -1.55140150e-09 * tc[2]
            +1.51702233e-12 * tc[3]
            -1.09201300e-16 * tc[4]
            +4.92030800e+00 ;
        /*species 2: H2 */
        species[2] =
            +2.99142300e+00 * tc[0]
            +7.00064400e-04 * tc[1]
            -2.81691450e-08 * tc[2]
            -3.07719267e-12 * tc[3]
            +3.95688000e-16 * tc[4]
            -1.35511000e+00 ;
        /*species 3: H */
        species[3] =
            +2.50104422e+00 * tc[0]
            +0.00000000e+00 * tc[1]
            +0.00000000e+00 * tc[2]
            +0.00000000e+00 * tc[3]
            +0.00000000e+00 * tc[4]
            -4.65341317e-01 ;
        /*species 5: H2O */
        species[5] =
            +2.67214600e+00 * tc[0]
            +3.05629300e-03 * tc[1]
            -4.36513000e-07 * tc[2]
            +4.00332000e-11 * tc[3]
            -1.59790450e-15 * tc[4]
            +6.86281700e+00 ;
        /*species 6: O2 */
        species[6] =
            +3.69757800e+00 * tc[0]
            +6.13519700e-04 * tc[1]
            -6.29421000e-08 * tc[2]
            +5.91760333e-12 * tc[3]
            -2.84108750e-16 * tc[4]
            +3.18916600e+00 ;
        /*species 8: H2O2 */
        species[8] =
            +4.57316700e+00 * tc[0]
            +4.33613600e-03 * tc[1]
            -7.37344500e-07 * tc[2]
            +7.82968000e-11 * tc[3]
            -3.57913500e-15 * tc[4]
            +5.01137000e-01 ;
        /*species 9: CO2 */
        species[9] =
            +4.45362300e+00 * tc[0]
            +3.14016900e-03 * tc[1]
            -6.39205500e-07 * tc[2]
            +7.97999000e-11 * tc[3]
            -4.17258250e-15 * tc[4]
            -9.55395900e-01 ;
        /*species 10: CO */
        species[10] =
            +3.02507800e+00 * tc[0]
            +1.44268900e-03 * tc[1]
            -2.81541400e-07 * tc[2]
            +3.39527000e-11 * tc[3]
            -1.72773800e-15 * tc[4]
            +6.10821800e+00 ;
        /*species 11: CH2O */
        species[11] =
            +2.99560600e+00 * tc[0]
            +6.68132100e-03 * tc[1]
            -1.31447750e-06 * tc[2]
            +1.57905100e-10 * tc[3]
            -8.03129250e-15 * tc[4]
            +6.91257200e+00 ;
        /*species 12: CH3 */
        species[12] =
            +2.84405200e+00 * tc[0]
            +6.13797400e-03 * tc[1]
            -1.11517250e-06 * tc[2]
            +1.26172033e-10 * tc[3]
            -6.13039750e-15 * tc[4]
            +5.45269700e+00 ;
        /*species 13: CH4 */
        species[13] =
            +1.68347900e+00 * tc[0]
            +1.02372400e-02 * tc[1]
            -1.93756450e-06 * tc[2]
            +2.26186167e-10 * tc[3]
            -1.12585575e-14 * tc[4]
            +9.62339500e+00 ;
        /*species 14: CH3OH */
        species[14] =
            +4.02906100e+00 * tc[0]
            +9.37659300e-03 * tc[1]
            -1.52512700e-06 * tc[2]
            +1.45293100e-10 * tc[3]
            -5.56180750e-15 * tc[4]
            +2.37819600e+00 ;
        /*species 16: CH2CO */
        species[16] =
            +6.03881700e+00 * tc[0]
            +5.80484000e-03 * tc[1]
            -9.60477000e-07 * tc[2]
            +9.31495000e-11 * tc[3]
            -3.64717000e-15 * tc[4]
            -7.65758100e+00 ;
        /*species 18: C2H2 */
        species[18] =
            +4.43677000e+00 * tc[0]
            +5.37603900e-03 * tc[1]
            -9.56408500e-07 * tc[2]
            +1.09545967e-10 * tc[3]
            -5.39177500e-15 * tc[4]
            -2.80033800e+00 ;
        /*species 20: C2H4 */
        species[20] =
            +3.52841900e+00 * tc[0]
            +1.14851800e-02 * tc[1]
            -2.20919250e-06 * tc[2]
            +2.61486700e-10 * tc[3]
            -1.31671200e-14 * tc[4]
            +2.23038900e+00 ;
    }

    /*species with midpoint at T=1357 kelvin */
    if (T < 1357) {
        /*species 4: OH */
        species[4] =
            +3.43586219e+00 * tc[0]
            +2.02235804e-04 * tc[1]
            -5.67732060e-08 * tc[2]
            +8.08150497e-11 * tc[3]
            -1.85912758e-14 * tc[4]
            +2.45014127e+00 ;
    } else {
        /*species 4: OH */
        species[4] =
            +2.62599754e+00 * tc[0]
            +1.31992406e-03 * tc[1]
            -1.79862335e-07 * tc[2]
            +1.41876933e-11 * tc[3]
            -4.55120040e-16 * tc[4]
            +7.10667307e+00 ;
    }

    /*species with midpoint at T=1390 kelvin */
    if (T < 1390) {
        /*species 7: HO2 */
        species[7] =
            +3.18310656e+00 * tc[0]
            +3.66767950e-03 * tc[1]
            -4.66192561e-07 * tc[2]
            -1.08617640e-10 * tc[3]
            +3.77849780e-14 * tc[4]
            +8.39371099e+00 ;
    } else {
        /*species 7: HO2 */
        species[7] =
            +4.10547423e+00 * tc[0]
            +2.38452835e-03 * tc[1]
            -4.03173995e-07 * tc[2]
            +4.13972410e-11 * tc[3]
            -1.79100027e-15 * tc[4]
            +3.12515836e+00 ;
    }

    /*species with midpoint at T=1384 kelvin */
    if (T < 1384) {
        /*species 15: C2H6 */
        species[15] =
            -2.52854344e-02 * tc[0]
            +2.40764754e-02 * tc[1]
            -5.59467360e-06 * tc[2]
            +6.94469670e-10 * tc[3]
            -1.32467154e-14 * tc[4]
            +2.11648750e+01 ;
    } else {
        /*species 15: C2H6 */
        species[15] =
            +6.10683385e+00 * tc[0]
            +1.29236361e-02 * tc[1]
            -2.21263598e-06 * tc[2]
            +2.29130575e-10 * tc[3]
            -9.97254330e-15 * tc[4]
            -1.30081250e+01 ;
    }

    /*species with midpoint at T=1376 kelvin */
    if (T < 1376) {
        /*species 17: HOCHO */
        species[17] =
            +1.43548185e+00 * tc[0]
            +1.63363016e-02 * tc[1]
            -5.31287105e-06 * tc[2]
            +1.10710992e-09 * tc[3]
            -1.00544026e-13 * tc[4]
            +1.72885798e+01 ;
    } else {
        /*species 17: HOCHO */
        species[17] =
            +6.68733013e+00 * tc[0]
            +5.14289368e-03 * tc[1]
            -9.11192565e-07 * tc[2]
            +9.65730543e-11 * tc[3]
            -4.27230498e-15 * tc[4]
            -1.13104798e+01 ;
    }

    /*species with midpoint at T=1388 kelvin */
    if (T < 1388) {
        /*species 19: C3H6 */
        species[19] =
            +3.94615444e-01 * tc[0]
            +2.89107662e-02 * tc[1]
            -7.74434040e-06 * tc[2]
            +1.29604736e-09 * tc[3]
            -8.44725880e-14 * tc[4]
            +2.19003736e+01 ;
    } else {
        /*species 19: C3H6 */
        species[19] =
            +8.01595958e+00 * tc[0]
            +1.37023634e-02 * tc[1]
            -2.33124867e-06 * tc[2]
            +2.40418134e-10 * tc[3]
            -1.04342532e-14 * tc[4]
            -2.00160668e+01 ;
    }

    /*species with midpoint at T=1400 kelvin */
    if (T < 1400) {
        /*species 21: C3H4XA */
        species[21] =
            +2.53983100e+00 * tc[0]
            +1.63343700e-02 * tc[1]
            -8.82475000e-07 * tc[2]
            -1.54912167e-09 * tc[3]
            +4.32282750e-13 * tc[4]
            +9.93570200e+00 ;
    } else {
        /*species 21: C3H4XA */
        species[21] =
            +9.77625600e+00 * tc[0]
            +5.30213800e-03 * tc[1]
            -1.85055900e-07 * tc[2]
            -1.00879533e-10 * tc[3]
            +1.27239525e-14 * tc[4]
            -3.07706100e+01 ;
    }

    /*species with midpoint at T=1398 kelvin */
    if (T < 1398) {
        /*species 22: C4H6 */
        species[22] =
            -1.43095121e+00 * tc[0]
            +4.78706062e-02 * tc[1]
            -2.07723400e-05 * tc[2]
            +6.38498507e-09 * tc[3]
            -8.92896267e-13 * tc[4]
            +2.90825833e+01 ;
    } else {
        /*species 22: C4H6 */
        species[22] =
            +1.11633789e+01 * tc[0]
            +1.37163965e-02 * tc[1]
            -2.34857892e-06 * tc[2]
            +2.43231279e-10 * tc[3]
            -1.05871551e-14 * tc[4]
            -3.69847949e+01 ;
    }

    /*species with midpoint at T=1392 kelvin */
    if (T < 1392) {
        /*species 23: C4H8X1 */
        species[23] =
            -8.31372089e-01 * tc[0]
            +4.52580978e-02 * tc[1]
            -1.46829280e-05 * tc[2]
            +3.34068120e-09 * tc[3]
            -3.57979200e-13 * tc[4]
            +2.95084236e+01 ;
    } else {
        /*species 23: C4H8X1 */
        species[23] =
            +1.13508668e+01 * tc[0]
            +1.80617877e-02 * tc[1]
            -3.08046515e-06 * tc[2]
            +3.18217653e-10 * tc[3]
            -1.38272410e-14 * tc[4]
            -3.64369438e+01 ;
    }

    /*species with midpoint at T=1391 kelvin */
    if (T < 1391) {
        /*species 24: NXC7H16 */
        species[24] =
            -1.26836187e+00 * tc[0]
            +8.54355820e-02 * tc[1]
            -2.62673393e-05 * tc[2]
            +5.43152403e-09 * tc[3]
            -5.05987313e-13 * tc[4]
            +3.53732912e+01 ;
    } else {
        /*species 24: NXC7H16 */
        species[24] =
            +2.22148969e+01 * tc[0]
            +3.47675750e-02 * tc[1]
            -5.92035645e-06 * tc[2]
            +6.10994927e-10 * tc[3]
            -2.65325665e-14 * tc[4]
            -9.23040196e+01 ;
    }
    return;
}


/*save atomic weights into array */
void atomicWeight(double *  awt)
{
    awt[0] = 14.006700; /*N */
    awt[1] = 15.999400; /*O */
    awt[2] = 1.007970; /*H */
    awt[3] = 12.011150; /*C */

    return;
}


/* get temperature given internal energy in mass units and mass fracs */
AMREX_GPU_HOST_DEVICE void GET_T_GIVEN_EY(double *  e, double *  y, double *  t, int * ierr)
{
#ifdef CONVERGENCE
    const int maxiter = 5000;
    const double tol  = 1.e-12;
#else
    const int maxiter = 200;
    const double tol  = 1.e-6;
#endif
    double ein  = *e;
    double tmin = 90;/*max lower bound for thermo def */
    double tmax = 4000;/*min upper bound for thermo def */
    double e1,emin,emax,cv,t1,dt;
    int i;/* loop counter */
    CKUBMS(&tmin, y, &emin);
    CKUBMS(&tmax, y, &emax);
    if (ein < emin) {
        /*Linear Extrapolation below tmin */
        CKCVBS(&tmin, y, &cv);
        *t = tmin - (emin-ein)/cv;
        *ierr = 1;
        return;
    }
    if (ein > emax) {
        /*Linear Extrapolation above tmax */
        CKCVBS(&tmax, y, &cv);
        *t = tmax - (emax-ein)/cv;
        *ierr = 1;
        return;
    }
    t1 = *t;
    if (t1 < tmin || t1 > tmax) {
        t1 = tmin + (tmax-tmin)/(emax-emin)*(ein-emin);
    }
    for (i = 0; i < maxiter; ++i) {
        CKUBMS(&t1,y,&e1);
        CKCVBS(&t1,y,&cv);
        dt = (ein - e1) / cv;
        if (dt > 100.) { dt = 100.; }
        else if (dt < -100.) { dt = -100.; }
        else if (fabs(dt) < tol) break;
        else if (t1+dt == t1) break;
        t1 += dt;
    }
    *t = t1;
    *ierr = 0;
    return;
}

/* get temperature given enthalpy in mass units and mass fracs */
AMREX_GPU_HOST_DEVICE void GET_T_GIVEN_HY(double *  h, double *  y, double *  t, int * ierr)
{
#ifdef CONVERGENCE
    const int maxiter = 5000;
    const double tol  = 1.e-12;
#else
    const int maxiter = 200;
    const double tol  = 1.e-6;
#endif
    double hin  = *h;
    double tmin = 90;/*max lower bound for thermo def */
    double tmax = 4000;/*min upper bound for thermo def */
    double h1,hmin,hmax,cp,t1,dt;
    int i;/* loop counter */
    CKHBMS(&tmin, y, &hmin);
    CKHBMS(&tmax, y, &hmax);
    if (hin < hmin) {
        /*Linear Extrapolation below tmin */
        CKCPBS(&tmin, y, &cp);
        *t = tmin - (hmin-hin)/cp;
        *ierr = 1;
        return;
    }
    if (hin > hmax) {
        /*Linear Extrapolation above tmax */
        CKCPBS(&tmax, y, &cp);
        *t = tmax - (hmax-hin)/cp;
        *ierr = 1;
        return;
    }
    t1 = *t;
    if (t1 < tmin || t1 > tmax) {
        t1 = tmin + (tmax-tmin)/(hmax-hmin)*(hin-hmin);
    }
    for (i = 0; i < maxiter; ++i) {
        CKHBMS(&t1,y,&h1);
        CKCPBS(&t1,y,&cp);
        dt = (hin - h1) / cp;
        if (dt > 100.) { dt = 100.; }
        else if (dt < -100.) { dt = -100.; }
        else if (fabs(dt) < tol) break;
        else if (t1+dt == t1) break;
        t1 += dt;
    }
    *t = t1;
    *ierr = 0;
    return;
}


void egtransetLENIMC(int* LENIMC ) {
    *LENIMC = 102;}


void egtransetLENRMC(int* LENRMC ) {
    *LENRMC = 12750;}


void egtransetNO(int* NO ) {
    *NO = 4;}


void egtransetKK(int* KK ) {
    *KK = 25;}


void egtransetNLITE(int* NLITE ) {
    *NLITE = 2;}


/*Patm in ergs/cm3 */
void egtransetPATM(double* PATM) {
    *PATM =   0.1013250000000000E+07;}


/*the molecular weights in g/mol */
void egtransetWT(double* WT ) {
    WT[0] = 2.80134000E+01;
    WT[1] = 1.59994000E+01;
    WT[2] = 2.01594000E+00;
    WT[3] = 1.00797000E+00;
    WT[4] = 1.70073700E+01;
    WT[5] = 1.80153400E+01;
    WT[6] = 3.19988000E+01;
    WT[7] = 3.30067700E+01;
    WT[8] = 3.40147400E+01;
    WT[9] = 4.40099500E+01;
    WT[10] = 2.80105500E+01;
    WT[11] = 3.00264900E+01;
    WT[12] = 1.50350600E+01;
    WT[13] = 1.60430300E+01;
    WT[14] = 3.20424300E+01;
    WT[15] = 3.00701200E+01;
    WT[16] = 4.20376400E+01;
    WT[17] = 4.60258900E+01;
    WT[18] = 2.60382400E+01;
    WT[19] = 4.20812700E+01;
    WT[20] = 2.80541800E+01;
    WT[21] = 4.00653300E+01;
    WT[22] = 5.40924200E+01;
    WT[23] = 5.61083600E+01;
    WT[24] = 1.00205570E+02;
}


/*the lennard-jones potential well depth eps/kb in K */
void egtransetEPS(double* EPS ) {
    EPS[0] = 9.75300000E+01;
    EPS[1] = 8.00000000E+01;
    EPS[2] = 3.80000000E+01;
    EPS[3] = 1.45000000E+02;
    EPS[4] = 8.00000000E+01;
    EPS[5] = 5.72400000E+02;
    EPS[6] = 1.07400000E+02;
    EPS[7] = 1.07400000E+02;
    EPS[8] = 1.07400000E+02;
    EPS[9] = 2.44000000E+02;
    EPS[10] = 9.81000000E+01;
    EPS[11] = 4.98000000E+02;
    EPS[12] = 1.44000000E+02;
    EPS[13] = 1.41400000E+02;
    EPS[14] = 4.81800000E+02;
    EPS[15] = 2.47500000E+02;
    EPS[16] = 4.36000000E+02;
    EPS[17] = 4.36000000E+02;
    EPS[18] = 2.65300000E+02;
    EPS[19] = 3.07800000E+02;
    EPS[20] = 2.38400000E+02;
    EPS[21] = 3.24800000E+02;
    EPS[22] = 3.57000000E+02;
    EPS[23] = 3.55000000E+02;
    EPS[24] = 4.59600000E+02;
}


/*the lennard-jones collision diameter in Angstroms */
void egtransetSIG(double* SIG ) {
    SIG[0] = 3.62100000E+00;
    SIG[1] = 2.75000000E+00;
    SIG[2] = 2.92000000E+00;
    SIG[3] = 2.05000000E+00;
    SIG[4] = 2.75000000E+00;
    SIG[5] = 2.60500000E+00;
    SIG[6] = 3.45800000E+00;
    SIG[7] = 3.45800000E+00;
    SIG[8] = 3.45800000E+00;
    SIG[9] = 3.76300000E+00;
    SIG[10] = 3.65000000E+00;
    SIG[11] = 3.59000000E+00;
    SIG[12] = 3.80000000E+00;
    SIG[13] = 3.74600000E+00;
    SIG[14] = 3.62600000E+00;
    SIG[15] = 4.35000000E+00;
    SIG[16] = 3.97000000E+00;
    SIG[17] = 3.97000000E+00;
    SIG[18] = 3.72100000E+00;
    SIG[19] = 4.14000000E+00;
    SIG[20] = 3.49600000E+00;
    SIG[21] = 4.29000000E+00;
    SIG[22] = 4.72000000E+00;
    SIG[23] = 4.65000000E+00;
    SIG[24] = 6.25300000E+00;
}


/*the dipole moment in Debye */
void egtransetDIP(double* DIP ) {
    DIP[0] = 0.00000000E+00;
    DIP[1] = 0.00000000E+00;
    DIP[2] = 0.00000000E+00;
    DIP[3] = 0.00000000E+00;
    DIP[4] = 0.00000000E+00;
    DIP[5] = 1.84400000E+00;
    DIP[6] = 0.00000000E+00;
    DIP[7] = 0.00000000E+00;
    DIP[8] = 0.00000000E+00;
    DIP[9] = 0.00000000E+00;
    DIP[10] = 0.00000000E+00;
    DIP[11] = 0.00000000E+00;
    DIP[12] = 0.00000000E+00;
    DIP[13] = 0.00000000E+00;
    DIP[14] = 0.00000000E+00;
    DIP[15] = 0.00000000E+00;
    DIP[16] = 0.00000000E+00;
    DIP[17] = 0.00000000E+00;
    DIP[18] = 0.00000000E+00;
    DIP[19] = 0.00000000E+00;
    DIP[20] = 0.00000000E+00;
    DIP[21] = 0.00000000E+00;
    DIP[22] = 0.00000000E+00;
    DIP[23] = 0.00000000E+00;
    DIP[24] = 0.00000000E+00;
}


/*the polarizability in cubic Angstroms */
void egtransetPOL(double* POL ) {
    POL[0] = 1.76000000E+00;
    POL[1] = 0.00000000E+00;
    POL[2] = 7.90000000E-01;
    POL[3] = 0.00000000E+00;
    POL[4] = 0.00000000E+00;
    POL[5] = 0.00000000E+00;
    POL[6] = 1.60000000E+00;
    POL[7] = 0.00000000E+00;
    POL[8] = 0.00000000E+00;
    POL[9] = 2.65000000E+00;
    POL[10] = 1.95000000E+00;
    POL[11] = 0.00000000E+00;
    POL[12] = 0.00000000E+00;
    POL[13] = 2.60000000E+00;
    POL[14] = 0.00000000E+00;
    POL[15] = 0.00000000E+00;
    POL[16] = 0.00000000E+00;
    POL[17] = 0.00000000E+00;
    POL[18] = 0.00000000E+00;
    POL[19] = 0.00000000E+00;
    POL[20] = 0.00000000E+00;
    POL[21] = 0.00000000E+00;
    POL[22] = 0.00000000E+00;
    POL[23] = 0.00000000E+00;
    POL[24] = 0.00000000E+00;
}


/*the rotational relaxation collision number at 298 K */
void egtransetZROT(double* ZROT ) {
    ZROT[0] = 4.00000000E+00;
    ZROT[1] = 0.00000000E+00;
    ZROT[2] = 2.80000000E+02;
    ZROT[3] = 0.00000000E+00;
    ZROT[4] = 0.00000000E+00;
    ZROT[5] = 4.00000000E+00;
    ZROT[6] = 3.80000000E+00;
    ZROT[7] = 1.00000000E+00;
    ZROT[8] = 3.80000000E+00;
    ZROT[9] = 2.10000000E+00;
    ZROT[10] = 1.80000000E+00;
    ZROT[11] = 2.00000000E+00;
    ZROT[12] = 0.00000000E+00;
    ZROT[13] = 1.30000000E+01;
    ZROT[14] = 1.00000000E+00;
    ZROT[15] = 1.50000000E+00;
    ZROT[16] = 2.00000000E+00;
    ZROT[17] = 2.00000000E+00;
    ZROT[18] = 2.50000000E+00;
    ZROT[19] = 1.00000000E+00;
    ZROT[20] = 1.50000000E+00;
    ZROT[21] = 1.00000000E+00;
    ZROT[22] = 1.00000000E+00;
    ZROT[23] = 1.00000000E+00;
    ZROT[24] = 1.00000000E+00;
}


/*0: monoatomic, 1: linear, 2: nonlinear */
void egtransetNLIN(int* NLIN) {
    NLIN[0] = 1;
    NLIN[1] = 0;
    NLIN[2] = 1;
    NLIN[3] = 0;
    NLIN[4] = 1;
    NLIN[5] = 2;
    NLIN[6] = 1;
    NLIN[7] = 2;
    NLIN[8] = 2;
    NLIN[9] = 1;
    NLIN[10] = 1;
    NLIN[11] = 2;
    NLIN[12] = 1;
    NLIN[13] = 2;
    NLIN[14] = 2;
    NLIN[15] = 2;
    NLIN[16] = 2;
    NLIN[17] = 2;
    NLIN[18] = 1;
    NLIN[19] = 2;
    NLIN[20] = 2;
    NLIN[21] = 1;
    NLIN[22] = 2;
    NLIN[23] = 2;
    NLIN[24] = 2;
}


/*Poly fits for the viscosities, dim NO*KK */
void egtransetCOFETA(double* COFETA) {
    COFETA[0] = -1.65695594E+01;
    COFETA[1] = 2.39056562E+00;
    COFETA[2] = -2.34558144E-01;
    COFETA[3] = 1.05024037E-02;
    COFETA[4] = -1.50926240E+01;
    COFETA[5] = 1.92606504E+00;
    COFETA[6] = -1.73487476E-01;
    COFETA[7] = 7.82572931E-03;
    COFETA[8] = -1.38347699E+01;
    COFETA[9] = 1.00106621E+00;
    COFETA[10] = -4.98105694E-02;
    COFETA[11] = 2.31450475E-03;
    COFETA[12] = -2.04078397E+01;
    COFETA[13] = 3.65436395E+00;
    COFETA[14] = -3.98339635E-01;
    COFETA[15] = 1.75883009E-02;
    COFETA[16] = -1.50620763E+01;
    COFETA[17] = 1.92606504E+00;
    COFETA[18] = -1.73487476E-01;
    COFETA[19] = 7.82572931E-03;
    COFETA[20] = -1.05420863E+01;
    COFETA[21] = -1.37777096E+00;
    COFETA[22] = 4.20502308E-01;
    COFETA[23] = -2.40627230E-02;
    COFETA[24] = -1.71618309E+01;
    COFETA[25] = 2.68036374E+00;
    COFETA[26] = -2.72570227E-01;
    COFETA[27] = 1.21650964E-02;
    COFETA[28] = -1.71463238E+01;
    COFETA[29] = 2.68036374E+00;
    COFETA[30] = -2.72570227E-01;
    COFETA[31] = 1.21650964E-02;
    COFETA[32] = -1.71312832E+01;
    COFETA[33] = 2.68036374E+00;
    COFETA[34] = -2.72570227E-01;
    COFETA[35] = 1.21650964E-02;
    COFETA[36] = -2.40014975E+01;
    COFETA[37] = 5.14359547E+00;
    COFETA[38] = -5.74269731E-01;
    COFETA[39] = 2.44937679E-02;
    COFETA[40] = -1.66188336E+01;
    COFETA[41] = 2.40307799E+00;
    COFETA[42] = -2.36167638E-01;
    COFETA[43] = 1.05714061E-02;
    COFETA[44] = -1.98330577E+01;
    COFETA[45] = 2.69480162E+00;
    COFETA[46] = -1.65880845E-01;
    COFETA[47] = 3.14504769E-03;
    COFETA[48] = -2.02316497E+01;
    COFETA[49] = 3.63241793E+00;
    COFETA[50] = -3.95581049E-01;
    COFETA[51] = 1.74725495E-02;
    COFETA[52] = -2.00094664E+01;
    COFETA[53] = 3.57220167E+00;
    COFETA[54] = -3.87936446E-01;
    COFETA[55] = 1.71483254E-02;
    COFETA[56] = -2.05644525E+01;
    COFETA[57] = 3.03946431E+00;
    COFETA[58] = -2.16994867E-01;
    COFETA[59] = 5.61394012E-03;
    COFETA[60] = -2.45432160E+01;
    COFETA[61] = 5.15878990E+00;
    COFETA[62] = -5.75274341E-01;
    COFETA[63] = 2.44975136E-02;
    COFETA[64] = -2.23395647E+01;
    COFETA[65] = 3.86433912E+00;
    COFETA[66] = -3.41553983E-01;
    COFETA[67] = 1.17083447E-02;
    COFETA[68] = -2.22942453E+01;
    COFETA[69] = 3.86433912E+00;
    COFETA[70] = -3.41553983E-01;
    COFETA[71] = 1.17083447E-02;
    COFETA[72] = -2.47697856E+01;
    COFETA[73] = 5.30039568E+00;
    COFETA[74] = -5.89273639E-01;
    COFETA[75] = 2.49261407E-02;
    COFETA[76] = -2.49727893E+01;
    COFETA[77] = 5.27067543E+00;
    COFETA[78] = -5.71909526E-01;
    COFETA[79] = 2.36230940E-02;
    COFETA[80] = -2.39690472E+01;
    COFETA[81] = 5.11436059E+00;
    COFETA[82] = -5.71999954E-01;
    COFETA[83] = 2.44581334E-02;
    COFETA[84] = -2.50199983E+01;
    COFETA[85] = 5.20184077E+00;
    COFETA[86] = -5.57265947E-01;
    COFETA[87] = 2.27565676E-02;
    COFETA[88] = -2.46654710E+01;
    COFETA[89] = 4.94595777E+00;
    COFETA[90] = -5.12278955E-01;
    COFETA[91] = 2.03286378E-02;
    COFETA[92] = -2.46476176E+01;
    COFETA[93] = 4.96413364E+00;
    COFETA[94] = -5.15375011E-01;
    COFETA[95] = 2.04926972E-02;
    COFETA[96] = -2.19841167E+01;
    COFETA[97] = 3.46341268E+00;
    COFETA[98] = -2.80516687E-01;
    COFETA[99] = 8.70427548E-03;
}


/*Poly fits for the conductivities, dim NO*KK */
void egtransetCOFLAM(double* COFLAM) {
    COFLAM[0] = 1.29306625E+01;
    COFLAM[1] = -3.52819393E+00;
    COFLAM[2] = 6.45501923E-01;
    COFLAM[3] = -3.19376675E-02;
    COFLAM[4] = 1.69267361E+00;
    COFLAM[5] = 1.92606504E+00;
    COFLAM[6] = -1.73487476E-01;
    COFLAM[7] = 7.82572931E-03;
    COFLAM[8] = 9.13734407E+00;
    COFLAM[9] = -4.36833375E-01;
    COFLAM[10] = 1.12981765E-01;
    COFLAM[11] = -2.54610412E-03;
    COFLAM[12] = -8.57929284E-01;
    COFLAM[13] = 3.65436395E+00;
    COFLAM[14] = -3.98339635E-01;
    COFLAM[15] = 1.75883009E-02;
    COFLAM[16] = 1.00130995E+01;
    COFLAM[17] = -1.54006603E+00;
    COFLAM[18] = 3.02750813E-01;
    COFLAM[19] = -1.29859622E-02;
    COFLAM[20] = 2.35086340E+01;
    COFLAM[21] = -9.05997330E+00;
    COFLAM[22] = 1.54816025E+00;
    COFLAM[23] = -7.71639384E-02;
    COFLAM[24] = -1.93888086E+00;
    COFLAM[25] = 2.89244157E+00;
    COFLAM[26] = -2.71258557E-01;
    COFLAM[27] = 1.15340544E-02;
    COFLAM[28] = 4.76619374E+00;
    COFLAM[29] = -4.27755023E-01;
    COFLAM[30] = 2.68283274E-01;
    COFLAM[31] = -1.65411221E-02;
    COFLAM[32] = 8.99254403E-01;
    COFLAM[33] = 1.32506478E+00;
    COFLAM[34] = 1.81955930E-02;
    COFLAM[35] = -4.46691285E-03;
    COFLAM[36] = -1.15552013E+01;
    COFLAM[37] = 5.97444378E+00;
    COFLAM[38] = -5.83493959E-01;
    COFLAM[39] = 2.11390997E-02;
    COFLAM[40] = 1.15794127E+01;
    COFLAM[41] = -3.02088602E+00;
    COFLAM[42] = 5.82091962E-01;
    COFLAM[43] = -2.93406692E-02;
    COFLAM[44] = 4.84793117E+00;
    COFLAM[45] = -2.13603179E+00;
    COFLAM[46] = 6.99833906E-01;
    COFLAM[47] = -4.38222985E-02;
    COFLAM[48] = 1.36305859E+01;
    COFLAM[49] = -4.45086721E+00;
    COFLAM[50] = 8.75862692E-01;
    COFLAM[51] = -4.60275277E-02;
    COFLAM[52] = 1.29622480E+01;
    COFLAM[53] = -4.85747192E+00;
    COFLAM[54] = 1.02918185E+00;
    COFLAM[55] = -5.69931976E-02;
    COFLAM[56] = -3.48612719E+00;
    COFLAM[57] = 1.33821415E+00;
    COFLAM[58] = 2.29051402E-01;
    COFLAM[59] = -2.22522544E-02;
    COFLAM[60] = -1.44773293E+01;
    COFLAM[61] = 6.20799727E+00;
    COFLAM[62] = -4.66686188E-01;
    COFLAM[63] = 1.03037078E-02;
    COFLAM[64] = -9.55248736E+00;
    COFLAM[65] = 4.54181017E+00;
    COFLAM[66] = -3.09443018E-01;
    COFLAM[67] = 5.98150058E-03;
    COFLAM[68] = -1.42688121E+01;
    COFLAM[69] = 6.13139162E+00;
    COFLAM[70] = -4.81580164E-01;
    COFLAM[71] = 1.17158883E-02;
    COFLAM[72] = -9.20687365E+00;
    COFLAM[73] = 5.13028609E+00;
    COFLAM[74] = -4.67868863E-01;
    COFLAM[75] = 1.64674383E-02;
    COFLAM[76] = -1.54410770E+01;
    COFLAM[77] = 6.67114766E+00;
    COFLAM[78] = -5.37137624E-01;
    COFLAM[79] = 1.38051704E-02;
    COFLAM[80] = -1.34447168E+01;
    COFLAM[81] = 6.12380203E+00;
    COFLAM[82] = -4.86657425E-01;
    COFLAM[83] = 1.24614518E-02;
    COFLAM[84] = -1.32966554E+01;
    COFLAM[85] = 5.92585034E+00;
    COFLAM[86] = -4.64901365E-01;
    COFLAM[87] = 1.16662523E-02;
    COFLAM[88] = -2.26611414E+01;
    COFLAM[89] = 9.78565333E+00;
    COFLAM[90] = -9.94033497E-01;
    COFLAM[91] = 3.57950722E-02;
    COFLAM[92] = -1.96439129E+01;
    COFLAM[93] = 8.31169569E+00;
    COFLAM[94] = -7.56268608E-01;
    COFLAM[95] = 2.35727121E-02;
    COFLAM[96] = -1.79582416E+01;
    COFLAM[97] = 7.27686902E+00;
    COFLAM[98] = -5.88898453E-01;
    COFLAM[99] = 1.49980279E-02;
}


/*Poly fits for the diffusion coefficients, dim NO*KK*KK */
void egtransetCOFD(double* COFD) {
    COFD[0] = -1.49828430E+01;
    COFD[1] = 3.25781069E+00;
    COFD[2] = -2.12199367E-01;
    COFD[3] = 9.36657283E-03;
    COFD[4] = -1.40756935E+01;
    COFD[5] = 3.07549274E+00;
    COFD[6] = -1.88889344E-01;
    COFD[7] = 8.37152866E-03;
    COFD[8] = -1.16906297E+01;
    COFD[9] = 2.47469981E+00;
    COFD[10] = -1.10436257E-01;
    COFD[11] = 4.95273813E-03;
    COFD[12] = -1.42894441E+01;
    COFD[13] = 3.67490723E+00;
    COFD[14] = -2.65114792E-01;
    COFD[15] = 1.16092671E-02;
    COFD[16] = -1.40949196E+01;
    COFD[17] = 3.07549274E+00;
    COFD[18] = -1.88889344E-01;
    COFD[19] = 8.37152866E-03;
    COFD[20] = -2.10643259E+01;
    COFD[21] = 5.53614847E+00;
    COFD[22] = -4.86046736E-01;
    COFD[23] = 2.03659188E-02;
    COFD[24] = -1.52414485E+01;
    COFD[25] = 3.35922578E+00;
    COFD[26] = -2.25181399E-01;
    COFD[27] = 9.92132878E-03;
    COFD[28] = -1.52486273E+01;
    COFD[29] = 3.35922578E+00;
    COFD[30] = -2.25181399E-01;
    COFD[31] = 9.92132878E-03;
    COFD[32] = -1.52554761E+01;
    COFD[33] = 3.35922578E+00;
    COFD[34] = -2.25181399E-01;
    COFD[35] = 9.92132878E-03;
    COFD[36] = -1.81432461E+01;
    COFD[37] = 4.37565431E+00;
    COFD[38] = -3.53906025E-01;
    COFD[39] = 1.53760786E-02;
    COFD[40] = -1.50031687E+01;
    COFD[41] = 3.26223357E+00;
    COFD[42] = -2.12746642E-01;
    COFD[43] = 9.38912883E-03;
    COFD[44] = -2.04833713E+01;
    COFD[45] = 5.23112374E+00;
    COFD[46] = -4.54967682E-01;
    COFD[47] = 1.93570423E-02;
    COFD[48] = -1.59633387E+01;
    COFD[49] = 3.66853818E+00;
    COFD[50] = -2.64346221E-01;
    COFD[51] = 1.15784613E-02;
    COFD[52] = -1.59327297E+01;
    COFD[53] = 3.65620899E+00;
    COFD[54] = -2.62933804E-01;
    COFD[55] = 1.15253223E-02;
    COFD[56] = -2.03844252E+01;
    COFD[57] = 5.18856872E+00;
    COFD[58] = -4.50001829E-01;
    COFD[59] = 1.91636142E-02;
    COFD[60] = -1.82673770E+01;
    COFD[61] = 4.39538102E+00;
    COFD[62] = -3.56367230E-01;
    COFD[63] = 1.54788461E-02;
    COFD[64] = -2.02646611E+01;
    COFD[65] = 5.10426133E+00;
    COFD[66] = -4.41256919E-01;
    COFD[67] = 1.88737290E-02;
    COFD[68] = -2.02822946E+01;
    COFD[69] = 5.10426133E+00;
    COFD[70] = -4.41256919E-01;
    COFD[71] = 1.88737290E-02;
    COFD[72] = -1.83039618E+01;
    COFD[73] = 4.47952077E+00;
    COFD[74] = -3.66569471E-01;
    COFD[75] = 1.58916129E-02;
    COFD[76] = -1.90859283E+01;
    COFD[77] = 4.68079396E+00;
    COFD[78] = -3.91231550E-01;
    COFD[79] = 1.69021170E-02;
    COFD[80] = -1.78815889E+01;
    COFD[81] = 4.34347890E+00;
    COFD[82] = -3.49890003E-01;
    COFD[83] = 1.52083459E-02;
    COFD[84] = -1.92783884E+01;
    COFD[85] = 4.73660584E+00;
    COFD[86] = -3.97704978E-01;
    COFD[87] = 1.71514887E-02;
    COFD[88] = -1.97484166E+01;
    COFD[89] = 4.84231878E+00;
    COFD[90] = -4.10101001E-01;
    COFD[91] = 1.76356687E-02;
    COFD[92] = -1.97226856E+01;
    COFD[93] = 4.83750266E+00;
    COFD[94] = -4.09581452E-01;
    COFD[95] = 1.76174739E-02;
    COFD[96] = -2.10685573E+01;
    COFD[97] = 5.15027524E+00;
    COFD[98] = -4.46126111E-01;
    COFD[99] = 1.90401391E-02;
    COFD[100] = -1.40756935E+01;
    COFD[101] = 3.07549274E+00;
    COFD[102] = -1.88889344E-01;
    COFD[103] = 8.37152866E-03;
    COFD[104] = -1.32093628E+01;
    COFD[105] = 2.90778936E+00;
    COFD[106] = -1.67388544E-01;
    COFD[107] = 7.45220609E-03;
    COFD[108] = -1.09595712E+01;
    COFD[109] = 2.30836460E+00;
    COFD[110] = -8.76339315E-02;
    COFD[111] = 3.90878445E-03;
    COFD[112] = -1.34230272E+01;
    COFD[113] = 3.48624238E+00;
    COFD[114] = -2.41554467E-01;
    COFD[115] = 1.06263545E-02;
    COFD[116] = -1.32244035E+01;
    COFD[117] = 2.90778936E+00;
    COFD[118] = -1.67388544E-01;
    COFD[119] = 7.45220609E-03;
    COFD[120] = -1.94093572E+01;
    COFD[121] = 5.16013126E+00;
    COFD[122] = -4.46824543E-01;
    COFD[123] = 1.90464887E-02;
    COFD[124] = -1.43139231E+01;
    COFD[125] = 3.17651319E+00;
    COFD[126] = -2.02028974E-01;
    COFD[127] = 8.94232502E-03;
    COFD[128] = -1.43190389E+01;
    COFD[129] = 3.17651319E+00;
    COFD[130] = -2.02028974E-01;
    COFD[131] = 8.94232502E-03;
    COFD[132] = -1.43238998E+01;
    COFD[133] = 3.17651319E+00;
    COFD[134] = -2.02028974E-01;
    COFD[135] = 8.94232502E-03;
    COFD[136] = -1.70534856E+01;
    COFD[137] = 4.14240922E+00;
    COFD[138] = -3.25239774E-01;
    COFD[139] = 1.41980687E-02;
    COFD[140] = -1.40999008E+01;
    COFD[141] = 3.08120012E+00;
    COFD[142] = -1.89629903E-01;
    COFD[143] = 8.40361952E-03;
    COFD[144] = -1.94373127E+01;
    COFD[145] = 5.02567894E+00;
    COFD[146] = -4.32045169E-01;
    COFD[147] = 1.85132214E-02;
    COFD[148] = -1.50766130E+01;
    COFD[149] = 3.47945612E+00;
    COFD[150] = -2.40703722E-01;
    COFD[151] = 1.05907441E-02;
    COFD[152] = -1.50270339E+01;
    COFD[153] = 3.46140064E+00;
    COFD[154] = -2.38440092E-01;
    COFD[155] = 1.04960087E-02;
    COFD[156] = -1.93364585E+01;
    COFD[157] = 4.98286777E+00;
    COFD[158] = -4.26970814E-01;
    COFD[159] = 1.83122917E-02;
    COFD[160] = -1.72112971E+01;
    COFD[161] = 4.15807461E+00;
    COFD[162] = -3.27178539E-01;
    COFD[163] = 1.42784349E-02;
    COFD[164] = -1.90883268E+01;
    COFD[165] = 4.84384483E+00;
    COFD[166] = -4.10265575E-01;
    COFD[167] = 1.76414287E-02;
    COFD[168] = -1.91004157E+01;
    COFD[169] = 4.84384483E+00;
    COFD[170] = -4.10265575E-01;
    COFD[171] = 1.76414287E-02;
    COFD[172] = -1.72286007E+01;
    COFD[173] = 4.24084025E+00;
    COFD[174] = -3.37428619E-01;
    COFD[175] = 1.47032793E-02;
    COFD[176] = -1.79361160E+01;
    COFD[177] = 4.42139452E+00;
    COFD[178] = -3.59567329E-01;
    COFD[179] = 1.56103969E-02;
    COFD[180] = -1.68343393E+01;
    COFD[181] = 4.11954900E+00;
    COFD[182] = -3.22470391E-01;
    COFD[183] = 1.40859564E-02;
    COFD[184] = -1.81499793E+01;
    COFD[185] = 4.48398491E+00;
    COFD[186] = -3.67097129E-01;
    COFD[187] = 1.59123634E-02;
    COFD[188] = -1.86652603E+01;
    COFD[189] = 4.61260432E+00;
    COFD[190] = -3.82854484E-01;
    COFD[191] = 1.65575163E-02;
    COFD[192] = -1.86254955E+01;
    COFD[193] = 4.60336076E+00;
    COFD[194] = -3.81691643E-01;
    COFD[195] = 1.65085234E-02;
    COFD[196] = -1.99792167E+01;
    COFD[197] = 4.92184026E+00;
    COFD[198] = -4.19745472E-01;
    COFD[199] = 1.80268154E-02;
    COFD[200] = -1.16906297E+01;
    COFD[201] = 2.47469981E+00;
    COFD[202] = -1.10436257E-01;
    COFD[203] = 4.95273813E-03;
    COFD[204] = -1.09595712E+01;
    COFD[205] = 2.30836460E+00;
    COFD[206] = -8.76339315E-02;
    COFD[207] = 3.90878445E-03;
    COFD[208] = -1.03270606E+01;
    COFD[209] = 2.19285409E+00;
    COFD[210] = -7.54492786E-02;
    COFD[211] = 3.51398213E-03;
    COFD[212] = -1.14366381E+01;
    COFD[213] = 2.78323501E+00;
    COFD[214] = -1.51214064E-01;
    COFD[215] = 6.75150012E-03;
    COFD[216] = -1.09628982E+01;
    COFD[217] = 2.30836460E+00;
    COFD[218] = -8.76339315E-02;
    COFD[219] = 3.90878445E-03;
    COFD[220] = -1.71982995E+01;
    COFD[221] = 4.63881404E+00;
    COFD[222] = -3.86139633E-01;
    COFD[223] = 1.66955081E-02;
    COFD[224] = -1.18988955E+01;
    COFD[225] = 2.57507000E+00;
    COFD[226] = -1.24033737E-01;
    COFD[227] = 5.56694959E-03;
    COFD[228] = -1.18998012E+01;
    COFD[229] = 2.57507000E+00;
    COFD[230] = -1.24033737E-01;
    COFD[231] = 5.56694959E-03;
    COFD[232] = -1.19006548E+01;
    COFD[233] = 2.57507000E+00;
    COFD[234] = -1.24033737E-01;
    COFD[235] = 5.56694959E-03;
    COFD[236] = -1.37794315E+01;
    COFD[237] = 3.23973858E+00;
    COFD[238] = -2.09989036E-01;
    COFD[239] = 9.27667906E-03;
    COFD[240] = -1.17159737E+01;
    COFD[241] = 2.48123210E+00;
    COFD[242] = -1.11322604E-01;
    COFD[243] = 4.99282389E-03;
    COFD[244] = -1.60528285E+01;
    COFD[245] = 4.11188603E+00;
    COFD[246] = -3.21540884E-01;
    COFD[247] = 1.40482564E-02;
    COFD[248] = -1.25141260E+01;
    COFD[249] = 2.77873601E+00;
    COFD[250] = -1.50637360E-01;
    COFD[251] = 6.72684281E-03;
    COFD[252] = -1.24693568E+01;
    COFD[253] = 2.76686648E+00;
    COFD[254] = -1.49120141E-01;
    COFD[255] = 6.66220432E-03;
    COFD[256] = -1.59537247E+01;
    COFD[257] = 4.07051484E+00;
    COFD[258] = -3.16303109E-01;
    COFD[259] = 1.38259377E-02;
    COFD[260] = -1.39658996E+01;
    COFD[261] = 3.24966086E+00;
    COFD[262] = -2.11199992E-01;
    COFD[263] = 9.32580661E-03;
    COFD[264] = -1.57034851E+01;
    COFD[265] = 3.93614244E+00;
    COFD[266] = -2.99111497E-01;
    COFD[267] = 1.30888229E-02;
    COFD[268] = -1.57054717E+01;
    COFD[269] = 3.93614244E+00;
    COFD[270] = -2.99111497E-01;
    COFD[271] = 1.30888229E-02;
    COFD[272] = -1.39315266E+01;
    COFD[273] = 3.30394764E+00;
    COFD[274] = -2.17920112E-01;
    COFD[275] = 9.60284243E-03;
    COFD[276] = -1.45715797E+01;
    COFD[277] = 3.49477850E+00;
    COFD[278] = -2.42635772E-01;
    COFD[279] = 1.06721490E-02;
    COFD[280] = -1.36336373E+01;
    COFD[281] = 3.22088176E+00;
    COFD[282] = -2.07623790E-01;
    COFD[283] = 9.17771542E-03;
    COFD[284] = -1.47725694E+01;
    COFD[285] = 3.55444478E+00;
    COFD[286] = -2.50272707E-01;
    COFD[287] = 1.09990787E-02;
    COFD[288] = -1.51448279E+01;
    COFD[289] = 3.64565939E+00;
    COFD[290] = -2.61726871E-01;
    COFD[291] = 1.14799244E-02;
    COFD[292] = -1.51163041E+01;
    COFD[293] = 3.64206330E+00;
    COFD[294] = -2.61313444E-01;
    COFD[295] = 1.14642754E-02;
    COFD[296] = -1.64899530E+01;
    COFD[297] = 4.01175649E+00;
    COFD[298] = -3.08860971E-01;
    COFD[299] = 1.35100076E-02;
    COFD[300] = -1.42894441E+01;
    COFD[301] = 3.67490723E+00;
    COFD[302] = -2.65114792E-01;
    COFD[303] = 1.16092671E-02;
    COFD[304] = -1.34230272E+01;
    COFD[305] = 3.48624238E+00;
    COFD[306] = -2.41554467E-01;
    COFD[307] = 1.06263545E-02;
    COFD[308] = -1.14366381E+01;
    COFD[309] = 2.78323501E+00;
    COFD[310] = -1.51214064E-01;
    COFD[311] = 6.75150012E-03;
    COFD[312] = -1.47968712E+01;
    COFD[313] = 4.23027636E+00;
    COFD[314] = -3.36139991E-01;
    COFD[315] = 1.46507621E-02;
    COFD[316] = -1.34247866E+01;
    COFD[317] = 3.48624238E+00;
    COFD[318] = -2.41554467E-01;
    COFD[319] = 1.06263545E-02;
    COFD[320] = -1.95739570E+01;
    COFD[321] = 5.61113230E+00;
    COFD[322] = -4.90190187E-01;
    COFD[323] = 2.03260675E-02;
    COFD[324] = -1.46550083E+01;
    COFD[325] = 3.83606243E+00;
    COFD[326] = -2.86076532E-01;
    COFD[327] = 1.25205829E-02;
    COFD[328] = -1.46554748E+01;
    COFD[329] = 3.83606243E+00;
    COFD[330] = -2.86076532E-01;
    COFD[331] = 1.25205829E-02;
    COFD[332] = -1.46559141E+01;
    COFD[333] = 3.83606243E+00;
    COFD[334] = -2.86076532E-01;
    COFD[335] = 1.25205829E-02;
    COFD[336] = -1.76147026E+01;
    COFD[337] = 4.86049500E+00;
    COFD[338] = -4.12200578E-01;
    COFD[339] = 1.77160971E-02;
    COFD[340] = -1.43151174E+01;
    COFD[341] = 3.68038508E+00;
    COFD[342] = -2.65779346E-01;
    COFD[343] = 1.16360771E-02;
    COFD[344] = -1.97550088E+01;
    COFD[345] = 5.56931926E+00;
    COFD[346] = -4.89105511E-01;
    COFD[347] = 2.04493129E-02;
    COFD[348] = -1.57994893E+01;
    COFD[349] = 4.22225052E+00;
    COFD[350] = -3.35156428E-01;
    COFD[351] = 1.46104855E-02;
    COFD[352] = -1.57199037E+01;
    COFD[353] = 4.19936335E+00;
    COFD[354] = -3.32311009E-01;
    COFD[355] = 1.44921003E-02;
    COFD[356] = -1.96866103E+01;
    COFD[357] = 5.54637286E+00;
    COFD[358] = -4.87070324E-01;
    COFD[359] = 2.03983467E-02;
    COFD[360] = -1.78637178E+01;
    COFD[361] = 4.88268692E+00;
    COFD[362] = -4.14917638E-01;
    COFD[363] = 1.78274298E-02;
    COFD[364] = -1.94688688E+01;
    COFD[365] = 5.43830787E+00;
    COFD[366] = -4.75472880E-01;
    COFD[367] = 1.99909996E-02;
    COFD[368] = -1.94698843E+01;
    COFD[369] = 5.43830787E+00;
    COFD[370] = -4.75472880E-01;
    COFD[371] = 1.99909996E-02;
    COFD[372] = -1.79310765E+01;
    COFD[373] = 4.98037650E+00;
    COFD[374] = -4.26676911E-01;
    COFD[375] = 1.83007231E-02;
    COFD[376] = -1.85748546E+01;
    COFD[377] = 5.14789919E+00;
    COFD[378] = -4.45930850E-01;
    COFD[379] = 1.90363341E-02;
    COFD[380] = -1.74407963E+01;
    COFD[381] = 4.83580036E+00;
    COFD[382] = -4.09383573E-01;
    COFD[383] = 1.76098175E-02;
    COFD[384] = -1.87647862E+01;
    COFD[385] = 5.19146813E+00;
    COFD[386] = -4.50340408E-01;
    COFD[387] = 1.91768178E-02;
    COFD[388] = -1.92784178E+01;
    COFD[389] = 5.32291505E+00;
    COFD[390] = -4.65883522E-01;
    COFD[391] = 1.97916109E-02;
    COFD[392] = -1.92361841E+01;
    COFD[393] = 5.31542554E+00;
    COFD[394] = -4.65003780E-01;
    COFD[395] = 1.97570185E-02;
    COFD[396] = -2.03114210E+01;
    COFD[397] = 5.50136606E+00;
    COFD[398] = -4.82461887E-01;
    COFD[399] = 2.02471523E-02;
    COFD[400] = -1.40949196E+01;
    COFD[401] = 3.07549274E+00;
    COFD[402] = -1.88889344E-01;
    COFD[403] = 8.37152866E-03;
    COFD[404] = -1.32244035E+01;
    COFD[405] = 2.90778936E+00;
    COFD[406] = -1.67388544E-01;
    COFD[407] = 7.45220609E-03;
    COFD[408] = -1.09628982E+01;
    COFD[409] = 2.30836460E+00;
    COFD[410] = -8.76339315E-02;
    COFD[411] = 3.90878445E-03;
    COFD[412] = -1.34247866E+01;
    COFD[413] = 3.48624238E+00;
    COFD[414] = -2.41554467E-01;
    COFD[415] = 1.06263545E-02;
    COFD[416] = -1.32399106E+01;
    COFD[417] = 2.90778936E+00;
    COFD[418] = -1.67388544E-01;
    COFD[419] = 7.45220609E-03;
    COFD[420] = -1.94253036E+01;
    COFD[421] = 5.16013126E+00;
    COFD[422] = -4.46824543E-01;
    COFD[423] = 1.90464887E-02;
    COFD[424] = -1.43340796E+01;
    COFD[425] = 3.17651319E+00;
    COFD[426] = -2.02028974E-01;
    COFD[427] = 8.94232502E-03;
    COFD[428] = -1.43394069E+01;
    COFD[429] = 3.17651319E+00;
    COFD[430] = -2.02028974E-01;
    COFD[431] = 8.94232502E-03;
    COFD[432] = -1.43444709E+01;
    COFD[433] = 3.17651319E+00;
    COFD[434] = -2.02028974E-01;
    COFD[435] = 8.94232502E-03;
    COFD[436] = -1.70757047E+01;
    COFD[437] = 4.14240922E+00;
    COFD[438] = -3.25239774E-01;
    COFD[439] = 1.41980687E-02;
    COFD[440] = -1.41191261E+01;
    COFD[441] = 3.08120012E+00;
    COFD[442] = -1.89629903E-01;
    COFD[443] = 8.40361952E-03;
    COFD[444] = -1.94570287E+01;
    COFD[445] = 5.02567894E+00;
    COFD[446] = -4.32045169E-01;
    COFD[447] = 1.85132214E-02;
    COFD[448] = -1.50911794E+01;
    COFD[449] = 3.47945612E+00;
    COFD[450] = -2.40703722E-01;
    COFD[451] = 1.05907441E-02;
    COFD[452] = -1.50420953E+01;
    COFD[453] = 3.46140064E+00;
    COFD[454] = -2.38440092E-01;
    COFD[455] = 1.04960087E-02;
    COFD[456] = -1.93566243E+01;
    COFD[457] = 4.98286777E+00;
    COFD[458] = -4.26970814E-01;
    COFD[459] = 1.83122917E-02;
    COFD[460] = -1.72310232E+01;
    COFD[461] = 4.15807461E+00;
    COFD[462] = -3.27178539E-01;
    COFD[463] = 1.42784349E-02;
    COFD[464] = -1.91102652E+01;
    COFD[465] = 4.84384483E+00;
    COFD[466] = -4.10265575E-01;
    COFD[467] = 1.76414287E-02;
    COFD[468] = -1.91229033E+01;
    COFD[469] = 4.84384483E+00;
    COFD[470] = -4.10265575E-01;
    COFD[471] = 1.76414287E-02;
    COFD[472] = -1.72473011E+01;
    COFD[473] = 4.24084025E+00;
    COFD[474] = -3.37428619E-01;
    COFD[475] = 1.47032793E-02;
    COFD[476] = -1.79580609E+01;
    COFD[477] = 4.42139452E+00;
    COFD[478] = -3.59567329E-01;
    COFD[479] = 1.56103969E-02;
    COFD[480] = -1.68535757E+01;
    COFD[481] = 4.11954900E+00;
    COFD[482] = -3.22470391E-01;
    COFD[483] = 1.40859564E-02;
    COFD[484] = -1.81716176E+01;
    COFD[485] = 4.48398491E+00;
    COFD[486] = -3.67097129E-01;
    COFD[487] = 1.59123634E-02;
    COFD[488] = -1.86886689E+01;
    COFD[489] = 4.61260432E+00;
    COFD[490] = -3.82854484E-01;
    COFD[491] = 1.65575163E-02;
    COFD[492] = -1.86491023E+01;
    COFD[493] = 4.60336076E+00;
    COFD[494] = -3.81691643E-01;
    COFD[495] = 1.65085234E-02;
    COFD[496] = -2.00054461E+01;
    COFD[497] = 4.92184026E+00;
    COFD[498] = -4.19745472E-01;
    COFD[499] = 1.80268154E-02;
    COFD[500] = -2.10643259E+01;
    COFD[501] = 5.53614847E+00;
    COFD[502] = -4.86046736E-01;
    COFD[503] = 2.03659188E-02;
    COFD[504] = -1.94093572E+01;
    COFD[505] = 5.16013126E+00;
    COFD[506] = -4.46824543E-01;
    COFD[507] = 1.90464887E-02;
    COFD[508] = -1.71982995E+01;
    COFD[509] = 4.63881404E+00;
    COFD[510] = -3.86139633E-01;
    COFD[511] = 1.66955081E-02;
    COFD[512] = -1.95739570E+01;
    COFD[513] = 5.61113230E+00;
    COFD[514] = -4.90190187E-01;
    COFD[515] = 2.03260675E-02;
    COFD[516] = -1.94253036E+01;
    COFD[517] = 5.16013126E+00;
    COFD[518] = -4.46824543E-01;
    COFD[519] = 1.90464887E-02;
    COFD[520] = -1.19157919E+01;
    COFD[521] = 9.28955130E-01;
    COFD[522] = 2.42107090E-01;
    COFD[523] = -1.59823963E-02;
    COFD[524] = -2.12652533E+01;
    COFD[525] = 5.59961818E+00;
    COFD[526] = -4.91624858E-01;
    COFD[527] = 2.05035550E-02;
    COFD[528] = -2.06463744E+01;
    COFD[529] = 5.41688482E+00;
    COFD[530] = -4.73387188E-01;
    COFD[531] = 1.99280175E-02;
    COFD[532] = -2.06516336E+01;
    COFD[533] = 5.41688482E+00;
    COFD[534] = -4.73387188E-01;
    COFD[535] = 1.99280175E-02;
    COFD[536] = -2.07653719E+01;
    COFD[537] = 5.01092022E+00;
    COFD[538] = -3.77985635E-01;
    COFD[539] = 1.40968645E-02;
    COFD[540] = -2.11388331E+01;
    COFD[541] = 5.55529675E+00;
    COFD[542] = -4.87942518E-01;
    COFD[543] = 2.04249054E-02;
    COFD[544] = -1.77563250E+01;
    COFD[545] = 3.57475686E+00;
    COFD[546] = -1.56396297E-01;
    COFD[547] = 3.12157721E-03;
    COFD[548] = -2.12831323E+01;
    COFD[549] = 5.61184117E+00;
    COFD[550] = -4.90532156E-01;
    COFD[551] = 2.03507922E-02;
    COFD[552] = -2.14087397E+01;
    COFD[553] = 5.57282008E+00;
    COFD[554] = -4.76690890E-01;
    COFD[555] = 1.94000719E-02;
    COFD[556] = -1.80253664E+01;
    COFD[557] = 3.69199168E+00;
    COFD[558] = -1.74005516E-01;
    COFD[559] = 3.97694372E-03;
    COFD[560] = -2.13148887E+01;
    COFD[561] = 5.27210469E+00;
    COFD[562] = -4.21419216E-01;
    COFD[563] = 1.63567178E-02;
    COFD[564] = -1.87383952E+01;
    COFD[565] = 3.96926341E+00;
    COFD[566] = -2.16412264E-01;
    COFD[567] = 6.06012078E-03;
    COFD[568] = -1.87515645E+01;
    COFD[569] = 3.96926341E+00;
    COFD[570] = -2.16412264E-01;
    COFD[571] = 6.06012078E-03;
    COFD[572] = -2.09565916E+01;
    COFD[573] = 5.18380539E+00;
    COFD[574] = -4.06234719E-01;
    COFD[575] = 1.55515345E-02;
    COFD[576] = -2.06310304E+01;
    COFD[577] = 4.89289496E+00;
    COFD[578] = -3.59346263E-01;
    COFD[579] = 1.31570901E-02;
    COFD[580] = -2.11309197E+01;
    COFD[581] = 5.32644193E+00;
    COFD[582] = -4.30581064E-01;
    COFD[583] = 1.68379725E-02;
    COFD[584] = -2.04397451E+01;
    COFD[585] = 4.77398686E+00;
    COFD[586] = -3.40522956E-01;
    COFD[587] = 1.22072846E-02;
    COFD[588] = -2.02184916E+01;
    COFD[589] = 4.57152878E+00;
    COFD[590] = -3.08371263E-01;
    COFD[591] = 1.05838559E-02;
    COFD[592] = -2.02287739E+01;
    COFD[593] = 4.58441724E+00;
    COFD[594] = -3.10392854E-01;
    COFD[595] = 1.06849990E-02;
    COFD[596] = -1.91334529E+01;
    COFD[597] = 3.82263611E+00;
    COFD[598] = -1.93983472E-01;
    COFD[599] = 4.95789388E-03;
    COFD[600] = -1.52414485E+01;
    COFD[601] = 3.35922578E+00;
    COFD[602] = -2.25181399E-01;
    COFD[603] = 9.92132878E-03;
    COFD[604] = -1.43139231E+01;
    COFD[605] = 3.17651319E+00;
    COFD[606] = -2.02028974E-01;
    COFD[607] = 8.94232502E-03;
    COFD[608] = -1.18988955E+01;
    COFD[609] = 2.57507000E+00;
    COFD[610] = -1.24033737E-01;
    COFD[611] = 5.56694959E-03;
    COFD[612] = -1.46550083E+01;
    COFD[613] = 3.83606243E+00;
    COFD[614] = -2.86076532E-01;
    COFD[615] = 1.25205829E-02;
    COFD[616] = -1.43340796E+01;
    COFD[617] = 3.17651319E+00;
    COFD[618] = -2.02028974E-01;
    COFD[619] = 8.94232502E-03;
    COFD[620] = -2.12652533E+01;
    COFD[621] = 5.59961818E+00;
    COFD[622] = -4.91624858E-01;
    COFD[623] = 2.05035550E-02;
    COFD[624] = -1.55511344E+01;
    COFD[625] = 3.48070094E+00;
    COFD[626] = -2.40859499E-01;
    COFD[627] = 1.05972514E-02;
    COFD[628] = -1.55588279E+01;
    COFD[629] = 3.48070094E+00;
    COFD[630] = -2.40859499E-01;
    COFD[631] = 1.05972514E-02;
    COFD[632] = -1.55661750E+01;
    COFD[633] = 3.48070094E+00;
    COFD[634] = -2.40859499E-01;
    COFD[635] = 1.05972514E-02;
    COFD[636] = -1.84688406E+01;
    COFD[637] = 4.49330851E+00;
    COFD[638] = -3.68208715E-01;
    COFD[639] = 1.59565402E-02;
    COFD[640] = -1.52721107E+01;
    COFD[641] = 3.36790500E+00;
    COFD[642] = -2.26321740E-01;
    COFD[643] = 9.97135055E-03;
    COFD[644] = -2.08293255E+01;
    COFD[645] = 5.35267674E+00;
    COFD[646] = -4.69010505E-01;
    COFD[647] = 1.98979152E-02;
    COFD[648] = -1.63493345E+01;
    COFD[649] = 3.82388595E+00;
    COFD[650] = -2.84480724E-01;
    COFD[651] = 1.24506311E-02;
    COFD[652] = -1.62724462E+01;
    COFD[653] = 3.79163564E+00;
    COFD[654] = -2.80257365E-01;
    COFD[655] = 1.22656902E-02;
    COFD[656] = -2.07595845E+01;
    COFD[657] = 5.32244593E+00;
    COFD[658] = -4.65829403E-01;
    COFD[659] = 1.97895274E-02;
    COFD[660] = -1.85844688E+01;
    COFD[661] = 4.51052425E+00;
    COFD[662] = -3.70301627E-01;
    COFD[663] = 1.60416153E-02;
    COFD[664] = -2.05184870E+01;
    COFD[665] = 5.18417470E+00;
    COFD[666] = -4.49491573E-01;
    COFD[667] = 1.91438508E-02;
    COFD[668] = -2.05375724E+01;
    COFD[669] = 5.18417470E+00;
    COFD[670] = -4.49491573E-01;
    COFD[671] = 1.91438508E-02;
    COFD[672] = -1.86507213E+01;
    COFD[673] = 4.60874797E+00;
    COFD[674] = -3.82368716E-01;
    COFD[675] = 1.65370164E-02;
    COFD[676] = -1.93917298E+01;
    COFD[677] = 4.78708023E+00;
    COFD[678] = -4.03693144E-01;
    COFD[679] = 1.73884817E-02;
    COFD[680] = -1.82145353E+01;
    COFD[681] = 4.46848269E+00;
    COFD[682] = -3.65269718E-01;
    COFD[683] = 1.58407652E-02;
    COFD[684] = -1.95875976E+01;
    COFD[685] = 4.84393038E+00;
    COFD[686] = -4.10274737E-01;
    COFD[687] = 1.76417458E-02;
    COFD[688] = -2.01315602E+01;
    COFD[689] = 4.97613338E+00;
    COFD[690] = -4.26175206E-01;
    COFD[691] = 1.82809270E-02;
    COFD[692] = -2.00997774E+01;
    COFD[693] = 4.96870443E+00;
    COFD[694] = -4.25292447E-01;
    COFD[695] = 1.82459096E-02;
    COFD[696] = -2.13968281E+01;
    COFD[697] = 5.25183817E+00;
    COFD[698] = -4.57376333E-01;
    COFD[699] = 1.94504429E-02;
    COFD[700] = -1.52486273E+01;
    COFD[701] = 3.35922578E+00;
    COFD[702] = -2.25181399E-01;
    COFD[703] = 9.92132878E-03;
    COFD[704] = -1.43190389E+01;
    COFD[705] = 3.17651319E+00;
    COFD[706] = -2.02028974E-01;
    COFD[707] = 8.94232502E-03;
    COFD[708] = -1.18998012E+01;
    COFD[709] = 2.57507000E+00;
    COFD[710] = -1.24033737E-01;
    COFD[711] = 5.56694959E-03;
    COFD[712] = -1.46554748E+01;
    COFD[713] = 3.83606243E+00;
    COFD[714] = -2.86076532E-01;
    COFD[715] = 1.25205829E-02;
    COFD[716] = -1.43394069E+01;
    COFD[717] = 3.17651319E+00;
    COFD[718] = -2.02028974E-01;
    COFD[719] = 8.94232502E-03;
    COFD[720] = -2.06463744E+01;
    COFD[721] = 5.41688482E+00;
    COFD[722] = -4.73387188E-01;
    COFD[723] = 1.99280175E-02;
    COFD[724] = -1.55588279E+01;
    COFD[725] = 3.48070094E+00;
    COFD[726] = -2.40859499E-01;
    COFD[727] = 1.05972514E-02;
    COFD[728] = -1.55666415E+01;
    COFD[729] = 3.48070094E+00;
    COFD[730] = -2.40859499E-01;
    COFD[731] = 1.05972514E-02;
    COFD[732] = -1.55741053E+01;
    COFD[733] = 3.48070094E+00;
    COFD[734] = -2.40859499E-01;
    COFD[735] = 1.05972514E-02;
    COFD[736] = -1.84777607E+01;
    COFD[737] = 4.49330851E+00;
    COFD[738] = -3.68208715E-01;
    COFD[739] = 1.59565402E-02;
    COFD[740] = -1.52792891E+01;
    COFD[741] = 3.36790500E+00;
    COFD[742] = -2.26321740E-01;
    COFD[743] = 9.97135055E-03;
    COFD[744] = -2.08367725E+01;
    COFD[745] = 5.35267674E+00;
    COFD[746] = -4.69010505E-01;
    COFD[747] = 1.98979152E-02;
    COFD[748] = -1.63542394E+01;
    COFD[749] = 3.82388595E+00;
    COFD[750] = -2.84480724E-01;
    COFD[751] = 1.24506311E-02;
    COFD[752] = -1.62775714E+01;
    COFD[753] = 3.79163564E+00;
    COFD[754] = -2.80257365E-01;
    COFD[755] = 1.22656902E-02;
    COFD[756] = -2.07672833E+01;
    COFD[757] = 5.32244593E+00;
    COFD[758] = -4.65829403E-01;
    COFD[759] = 1.97895274E-02;
    COFD[760] = -1.85919214E+01;
    COFD[761] = 4.51052425E+00;
    COFD[762] = -3.70301627E-01;
    COFD[763] = 1.60416153E-02;
    COFD[764] = -2.05272328E+01;
    COFD[765] = 5.18417470E+00;
    COFD[766] = -4.49491573E-01;
    COFD[767] = 1.91438508E-02;
    COFD[768] = -2.05466616E+01;
    COFD[769] = 5.18417470E+00;
    COFD[770] = -4.49491573E-01;
    COFD[771] = 1.91438508E-02;
    COFD[772] = -1.86576191E+01;
    COFD[773] = 4.60874797E+00;
    COFD[774] = -3.82368716E-01;
    COFD[775] = 1.65370164E-02;
    COFD[776] = -1.94004795E+01;
    COFD[777] = 4.78708023E+00;
    COFD[778] = -4.03693144E-01;
    COFD[779] = 1.73884817E-02;
    COFD[780] = -1.82217198E+01;
    COFD[781] = 4.46848269E+00;
    COFD[782] = -3.65269718E-01;
    COFD[783] = 1.58407652E-02;
    COFD[784] = -1.95961596E+01;
    COFD[785] = 4.84393038E+00;
    COFD[786] = -4.10274737E-01;
    COFD[787] = 1.76417458E-02;
    COFD[788] = -2.01412473E+01;
    COFD[789] = 4.97613338E+00;
    COFD[790] = -4.26175206E-01;
    COFD[791] = 1.82809270E-02;
    COFD[792] = -2.01095969E+01;
    COFD[793] = 4.96870443E+00;
    COFD[794] = -4.25292447E-01;
    COFD[795] = 1.82459096E-02;
    COFD[796] = -2.14085375E+01;
    COFD[797] = 5.25183817E+00;
    COFD[798] = -4.57376333E-01;
    COFD[799] = 1.94504429E-02;
    COFD[800] = -1.52554761E+01;
    COFD[801] = 3.35922578E+00;
    COFD[802] = -2.25181399E-01;
    COFD[803] = 9.92132878E-03;
    COFD[804] = -1.43238998E+01;
    COFD[805] = 3.17651319E+00;
    COFD[806] = -2.02028974E-01;
    COFD[807] = 8.94232502E-03;
    COFD[808] = -1.19006548E+01;
    COFD[809] = 2.57507000E+00;
    COFD[810] = -1.24033737E-01;
    COFD[811] = 5.56694959E-03;
    COFD[812] = -1.46559141E+01;
    COFD[813] = 3.83606243E+00;
    COFD[814] = -2.86076532E-01;
    COFD[815] = 1.25205829E-02;
    COFD[816] = -1.43444709E+01;
    COFD[817] = 3.17651319E+00;
    COFD[818] = -2.02028974E-01;
    COFD[819] = 8.94232502E-03;
    COFD[820] = -2.06516336E+01;
    COFD[821] = 5.41688482E+00;
    COFD[822] = -4.73387188E-01;
    COFD[823] = 1.99280175E-02;
    COFD[824] = -1.55661750E+01;
    COFD[825] = 3.48070094E+00;
    COFD[826] = -2.40859499E-01;
    COFD[827] = 1.05972514E-02;
    COFD[828] = -1.55741053E+01;
    COFD[829] = 3.48070094E+00;
    COFD[830] = -2.40859499E-01;
    COFD[831] = 1.05972514E-02;
    COFD[832] = -1.55816822E+01;
    COFD[833] = 3.48070094E+00;
    COFD[834] = -2.40859499E-01;
    COFD[835] = 1.05972514E-02;
    COFD[836] = -1.84863000E+01;
    COFD[837] = 4.49330851E+00;
    COFD[838] = -3.68208715E-01;
    COFD[839] = 1.59565402E-02;
    COFD[840] = -1.52861376E+01;
    COFD[841] = 3.36790500E+00;
    COFD[842] = -2.26321740E-01;
    COFD[843] = 9.97135055E-03;
    COFD[844] = -2.08438809E+01;
    COFD[845] = 5.35267674E+00;
    COFD[846] = -4.69010505E-01;
    COFD[847] = 1.98979152E-02;
    COFD[848] = -1.63588981E+01;
    COFD[849] = 3.82388595E+00;
    COFD[850] = -2.84480724E-01;
    COFD[851] = 1.24506311E-02;
    COFD[852] = -1.62824412E+01;
    COFD[853] = 3.79163564E+00;
    COFD[854] = -2.80257365E-01;
    COFD[855] = 1.22656902E-02;
    COFD[856] = -2.07746356E+01;
    COFD[857] = 5.32244593E+00;
    COFD[858] = -4.65829403E-01;
    COFD[859] = 1.97895274E-02;
    COFD[860] = -1.85990352E+01;
    COFD[861] = 4.51052425E+00;
    COFD[862] = -3.70301627E-01;
    COFD[863] = 1.60416153E-02;
    COFD[864] = -2.05356023E+01;
    COFD[865] = 5.18417470E+00;
    COFD[866] = -4.49491573E-01;
    COFD[867] = 1.91438508E-02;
    COFD[868] = -2.05553656E+01;
    COFD[869] = 5.18417470E+00;
    COFD[870] = -4.49491573E-01;
    COFD[871] = 1.91438508E-02;
    COFD[872] = -1.86641962E+01;
    COFD[873] = 4.60874797E+00;
    COFD[874] = -3.82368716E-01;
    COFD[875] = 1.65370164E-02;
    COFD[876] = -1.94088529E+01;
    COFD[877] = 4.78708023E+00;
    COFD[878] = -4.03693144E-01;
    COFD[879] = 1.73884817E-02;
    COFD[880] = -1.82285740E+01;
    COFD[881] = 4.46848269E+00;
    COFD[882] = -3.65269718E-01;
    COFD[883] = 1.58407652E-02;
    COFD[884] = -1.96043503E+01;
    COFD[885] = 4.84393038E+00;
    COFD[886] = -4.10274737E-01;
    COFD[887] = 1.76417458E-02;
    COFD[888] = -2.01505348E+01;
    COFD[889] = 4.97613338E+00;
    COFD[890] = -4.26175206E-01;
    COFD[891] = 1.82809270E-02;
    COFD[892] = -2.01190139E+01;
    COFD[893] = 4.96870443E+00;
    COFD[894] = -4.25292447E-01;
    COFD[895] = 1.82459096E-02;
    COFD[896] = -2.14198091E+01;
    COFD[897] = 5.25183817E+00;
    COFD[898] = -4.57376333E-01;
    COFD[899] = 1.94504429E-02;
    COFD[900] = -1.81432461E+01;
    COFD[901] = 4.37565431E+00;
    COFD[902] = -3.53906025E-01;
    COFD[903] = 1.53760786E-02;
    COFD[904] = -1.70534856E+01;
    COFD[905] = 4.14240922E+00;
    COFD[906] = -3.25239774E-01;
    COFD[907] = 1.41980687E-02;
    COFD[908] = -1.37794315E+01;
    COFD[909] = 3.23973858E+00;
    COFD[910] = -2.09989036E-01;
    COFD[911] = 9.27667906E-03;
    COFD[912] = -1.76147026E+01;
    COFD[913] = 4.86049500E+00;
    COFD[914] = -4.12200578E-01;
    COFD[915] = 1.77160971E-02;
    COFD[916] = -1.70757047E+01;
    COFD[917] = 4.14240922E+00;
    COFD[918] = -3.25239774E-01;
    COFD[919] = 1.41980687E-02;
    COFD[920] = -2.07653719E+01;
    COFD[921] = 5.01092022E+00;
    COFD[922] = -3.77985635E-01;
    COFD[923] = 1.40968645E-02;
    COFD[924] = -1.84688406E+01;
    COFD[925] = 4.49330851E+00;
    COFD[926] = -3.68208715E-01;
    COFD[927] = 1.59565402E-02;
    COFD[928] = -1.84777607E+01;
    COFD[929] = 4.49330851E+00;
    COFD[930] = -3.68208715E-01;
    COFD[931] = 1.59565402E-02;
    COFD[932] = -1.84863000E+01;
    COFD[933] = 4.49330851E+00;
    COFD[934] = -3.68208715E-01;
    COFD[935] = 1.59565402E-02;
    COFD[936] = -2.13425698E+01;
    COFD[937] = 5.40460130E+00;
    COFD[938] = -4.72718910E-01;
    COFD[939] = 1.99362717E-02;
    COFD[940] = -1.81735763E+01;
    COFD[941] = 4.38391495E+00;
    COFD[942] = -3.54941287E-01;
    COFD[943] = 1.54195107E-02;
    COFD[944] = -2.19317743E+01;
    COFD[945] = 5.45216133E+00;
    COFD[946] = -4.52916925E-01;
    COFD[947] = 1.80456400E-02;
    COFD[948] = -1.93276434E+01;
    COFD[949] = 4.85015581E+00;
    COFD[950] = -4.10945109E-01;
    COFD[951] = 1.76651398E-02;
    COFD[952] = -1.92867554E+01;
    COFD[953] = 4.83375900E+00;
    COFD[954] = -4.09146560E-01;
    COFD[955] = 1.76006599E-02;
    COFD[956] = -2.20063594E+01;
    COFD[957] = 5.48540187E+00;
    COFD[958] = -4.58962148E-01;
    COFD[959] = 1.83770355E-02;
    COFD[960] = -2.14151520E+01;
    COFD[961] = 5.41122754E+00;
    COFD[962] = -4.73185889E-01;
    COFD[963] = 1.99407905E-02;
    COFD[964] = -2.22116706E+01;
    COFD[965] = 5.54251230E+00;
    COFD[966] = -4.70946314E-01;
    COFD[967] = 1.90785869E-02;
    COFD[968] = -2.22343363E+01;
    COFD[969] = 5.54251230E+00;
    COFD[970] = -4.70946314E-01;
    COFD[971] = 1.90785869E-02;
    COFD[972] = -2.13961414E+01;
    COFD[973] = 5.46685775E+00;
    COFD[974] = -4.78665416E-01;
    COFD[975] = 2.01093915E-02;
    COFD[976] = -2.20725883E+01;
    COFD[977] = 5.59642965E+00;
    COFD[978] = -4.91577716E-01;
    COFD[979] = 2.05159582E-02;
    COFD[980] = -2.11031143E+01;
    COFD[981] = 5.39439999E+00;
    COFD[982] = -4.72050184E-01;
    COFD[983] = 1.99336257E-02;
    COFD[984] = -2.21697404E+01;
    COFD[985] = 5.60807471E+00;
    COFD[986] = -4.91339309E-01;
    COFD[987] = 2.04365761E-02;
    COFD[988] = -2.23890317E+01;
    COFD[989] = 5.59178974E+00;
    COFD[990] = -4.85668031E-01;
    COFD[991] = 2.00491907E-02;
    COFD[992] = -2.23812726E+01;
    COFD[993] = 5.59425354E+00;
    COFD[994] = -4.86232980E-01;
    COFD[995] = 2.00835981E-02;
    COFD[996] = -2.28671232E+01;
    COFD[997] = 5.50522401E+00;
    COFD[998] = -4.63604304E-01;
    COFD[999] = 1.86600785E-02;
    COFD[1000] = -1.50031687E+01;
    COFD[1001] = 3.26223357E+00;
    COFD[1002] = -2.12746642E-01;
    COFD[1003] = 9.38912883E-03;
    COFD[1004] = -1.40999008E+01;
    COFD[1005] = 3.08120012E+00;
    COFD[1006] = -1.89629903E-01;
    COFD[1007] = 8.40361952E-03;
    COFD[1008] = -1.17159737E+01;
    COFD[1009] = 2.48123210E+00;
    COFD[1010] = -1.11322604E-01;
    COFD[1011] = 4.99282389E-03;
    COFD[1012] = -1.43151174E+01;
    COFD[1013] = 3.68038508E+00;
    COFD[1014] = -2.65779346E-01;
    COFD[1015] = 1.16360771E-02;
    COFD[1016] = -1.41191261E+01;
    COFD[1017] = 3.08120012E+00;
    COFD[1018] = -1.89629903E-01;
    COFD[1019] = 8.40361952E-03;
    COFD[1020] = -2.11388331E+01;
    COFD[1021] = 5.55529675E+00;
    COFD[1022] = -4.87942518E-01;
    COFD[1023] = 2.04249054E-02;
    COFD[1024] = -1.52721107E+01;
    COFD[1025] = 3.36790500E+00;
    COFD[1026] = -2.26321740E-01;
    COFD[1027] = 9.97135055E-03;
    COFD[1028] = -1.52792891E+01;
    COFD[1029] = 3.36790500E+00;
    COFD[1030] = -2.26321740E-01;
    COFD[1031] = 9.97135055E-03;
    COFD[1032] = -1.52861376E+01;
    COFD[1033] = 3.36790500E+00;
    COFD[1034] = -2.26321740E-01;
    COFD[1035] = 9.97135055E-03;
    COFD[1036] = -1.81735763E+01;
    COFD[1037] = 4.38391495E+00;
    COFD[1038] = -3.54941287E-01;
    COFD[1039] = 1.54195107E-02;
    COFD[1040] = -1.50233475E+01;
    COFD[1041] = 3.26660767E+00;
    COFD[1042] = -2.13287177E-01;
    COFD[1043] = 9.41137857E-03;
    COFD[1044] = -2.05128705E+01;
    COFD[1045] = 5.23843909E+00;
    COFD[1046] = -4.55815614E-01;
    COFD[1047] = 1.93898040E-02;
    COFD[1048] = -1.59863030E+01;
    COFD[1049] = 3.67388294E+00;
    COFD[1050] = -2.64990709E-01;
    COFD[1051] = 1.16042706E-02;
    COFD[1052] = -1.59525102E+01;
    COFD[1053] = 3.66023858E+00;
    COFD[1054] = -2.63401043E-01;
    COFD[1055] = 1.15432000E-02;
    COFD[1056] = -2.04144604E+01;
    COFD[1057] = 5.19614628E+00;
    COFD[1058] = -4.50889164E-01;
    COFD[1059] = 1.91983328E-02;
    COFD[1060] = -1.82955252E+01;
    COFD[1061] = 4.40289649E+00;
    COFD[1062] = -3.57289765E-01;
    COFD[1063] = 1.55166804E-02;
    COFD[1064] = -2.02922701E+01;
    COFD[1065] = 5.11106992E+00;
    COFD[1066] = -4.42047129E-01;
    COFD[1067] = 1.89042990E-02;
    COFD[1068] = -2.03099025E+01;
    COFD[1069] = 5.11106992E+00;
    COFD[1070] = -4.42047129E-01;
    COFD[1071] = 1.89042990E-02;
    COFD[1072] = -1.83296965E+01;
    COFD[1073] = 4.48570999E+00;
    COFD[1074] = -3.67301524E-01;
    COFD[1075] = 1.59204254E-02;
    COFD[1076] = -1.91118445E+01;
    COFD[1077] = 4.68715685E+00;
    COFD[1078] = -3.91979493E-01;
    COFD[1079] = 1.69314004E-02;
    COFD[1080] = -1.79116531E+01;
    COFD[1081] = 4.35148286E+00;
    COFD[1082] = -3.50886647E-01;
    COFD[1083] = 1.52498573E-02;
    COFD[1084] = -1.93064215E+01;
    COFD[1085] = 4.74387793E+00;
    COFD[1086] = -3.98574972E-01;
    COFD[1087] = 1.71862289E-02;
    COFD[1088] = -1.97709603E+01;
    COFD[1089] = 4.84731557E+00;
    COFD[1090] = -4.10638352E-01;
    COFD[1091] = 1.76543886E-02;
    COFD[1092] = -1.97452574E+01;
    COFD[1093] = 4.84249900E+00;
    COFD[1094] = -4.10120448E-01;
    COFD[1095] = 1.76363500E-02;
    COFD[1096] = -2.10855099E+01;
    COFD[1097] = 5.15315713E+00;
    COFD[1098] = -4.46344043E-01;
    COFD[1099] = 1.90431546E-02;
    COFD[1100] = -2.04833713E+01;
    COFD[1101] = 5.23112374E+00;
    COFD[1102] = -4.54967682E-01;
    COFD[1103] = 1.93570423E-02;
    COFD[1104] = -1.94373127E+01;
    COFD[1105] = 5.02567894E+00;
    COFD[1106] = -4.32045169E-01;
    COFD[1107] = 1.85132214E-02;
    COFD[1108] = -1.60528285E+01;
    COFD[1109] = 4.11188603E+00;
    COFD[1110] = -3.21540884E-01;
    COFD[1111] = 1.40482564E-02;
    COFD[1112] = -1.97550088E+01;
    COFD[1113] = 5.56931926E+00;
    COFD[1114] = -4.89105511E-01;
    COFD[1115] = 2.04493129E-02;
    COFD[1116] = -1.94570287E+01;
    COFD[1117] = 5.02567894E+00;
    COFD[1118] = -4.32045169E-01;
    COFD[1119] = 1.85132214E-02;
    COFD[1120] = -1.77563250E+01;
    COFD[1121] = 3.57475686E+00;
    COFD[1122] = -1.56396297E-01;
    COFD[1123] = 3.12157721E-03;
    COFD[1124] = -2.08293255E+01;
    COFD[1125] = 5.35267674E+00;
    COFD[1126] = -4.69010505E-01;
    COFD[1127] = 1.98979152E-02;
    COFD[1128] = -2.08367725E+01;
    COFD[1129] = 5.35267674E+00;
    COFD[1130] = -4.69010505E-01;
    COFD[1131] = 1.98979152E-02;
    COFD[1132] = -2.08438809E+01;
    COFD[1133] = 5.35267674E+00;
    COFD[1134] = -4.69010505E-01;
    COFD[1135] = 1.98979152E-02;
    COFD[1136] = -2.19317743E+01;
    COFD[1137] = 5.45216133E+00;
    COFD[1138] = -4.52916925E-01;
    COFD[1139] = 1.80456400E-02;
    COFD[1140] = -2.05128705E+01;
    COFD[1141] = 5.23843909E+00;
    COFD[1142] = -4.55815614E-01;
    COFD[1143] = 1.93898040E-02;
    COFD[1144] = -1.90499441E+01;
    COFD[1145] = 3.99221757E+00;
    COFD[1146] = -2.19854880E-01;
    COFD[1147] = 6.22736279E-03;
    COFD[1148] = -2.14449559E+01;
    COFD[1149] = 5.56531152E+00;
    COFD[1150] = -4.88789821E-01;
    COFD[1151] = 2.04437116E-02;
    COFD[1152] = -2.14082453E+01;
    COFD[1153] = 5.55346617E+00;
    COFD[1154] = -4.87783156E-01;
    COFD[1155] = 2.04210886E-02;
    COFD[1156] = -1.93214527E+01;
    COFD[1157] = 4.10954793E+00;
    COFD[1158] = -2.37523329E-01;
    COFD[1159] = 7.08858141E-03;
    COFD[1160] = -2.19786173E+01;
    COFD[1161] = 5.43750833E+00;
    COFD[1162] = -4.50273329E-01;
    COFD[1163] = 1.79013718E-02;
    COFD[1164] = -2.01015340E+01;
    COFD[1165] = 4.41511629E+00;
    COFD[1166] = -2.84086963E-01;
    COFD[1167] = 9.37586971E-03;
    COFD[1168] = -2.01199204E+01;
    COFD[1169] = 4.41511629E+00;
    COFD[1170] = -2.84086963E-01;
    COFD[1171] = 9.37586971E-03;
    COFD[1172] = -2.16798265E+01;
    COFD[1173] = 5.36811769E+00;
    COFD[1174] = -4.37727086E-01;
    COFD[1175] = 1.72167686E-02;
    COFD[1176] = -2.15802788E+01;
    COFD[1177] = 5.16868516E+00;
    COFD[1178] = -4.03721581E-01;
    COFD[1179] = 1.54206640E-02;
    COFD[1180] = -2.17855148E+01;
    COFD[1181] = 5.47519298E+00;
    COFD[1182] = -4.57113040E-01;
    COFD[1183] = 1.82758312E-02;
    COFD[1184] = -2.14453157E+01;
    COFD[1185] = 5.07680397E+00;
    COFD[1186] = -3.88612087E-01;
    COFD[1187] = 1.46395101E-02;
    COFD[1188] = -2.12219677E+01;
    COFD[1189] = 4.87252053E+00;
    COFD[1190] = -3.56127804E-01;
    COFD[1191] = 1.29948788E-02;
    COFD[1192] = -2.12362684E+01;
    COFD[1193] = 4.88535789E+00;
    COFD[1194] = -3.58153894E-01;
    COFD[1195] = 1.30969624E-02;
    COFD[1196] = -2.04632210E+01;
    COFD[1197] = 4.26473557E+00;
    COFD[1198] = -2.61033037E-01;
    COFD[1199] = 8.23906412E-03;
    COFD[1200] = -1.59633387E+01;
    COFD[1201] = 3.66853818E+00;
    COFD[1202] = -2.64346221E-01;
    COFD[1203] = 1.15784613E-02;
    COFD[1204] = -1.50766130E+01;
    COFD[1205] = 3.47945612E+00;
    COFD[1206] = -2.40703722E-01;
    COFD[1207] = 1.05907441E-02;
    COFD[1208] = -1.25141260E+01;
    COFD[1209] = 2.77873601E+00;
    COFD[1210] = -1.50637360E-01;
    COFD[1211] = 6.72684281E-03;
    COFD[1212] = -1.57994893E+01;
    COFD[1213] = 4.22225052E+00;
    COFD[1214] = -3.35156428E-01;
    COFD[1215] = 1.46104855E-02;
    COFD[1216] = -1.50911794E+01;
    COFD[1217] = 3.47945612E+00;
    COFD[1218] = -2.40703722E-01;
    COFD[1219] = 1.05907441E-02;
    COFD[1220] = -2.12831323E+01;
    COFD[1221] = 5.61184117E+00;
    COFD[1222] = -4.90532156E-01;
    COFD[1223] = 2.03507922E-02;
    COFD[1224] = -1.63493345E+01;
    COFD[1225] = 3.82388595E+00;
    COFD[1226] = -2.84480724E-01;
    COFD[1227] = 1.24506311E-02;
    COFD[1228] = -1.63542394E+01;
    COFD[1229] = 3.82388595E+00;
    COFD[1230] = -2.84480724E-01;
    COFD[1231] = 1.24506311E-02;
    COFD[1232] = -1.63588981E+01;
    COFD[1233] = 3.82388595E+00;
    COFD[1234] = -2.84480724E-01;
    COFD[1235] = 1.24506311E-02;
    COFD[1236] = -1.93276434E+01;
    COFD[1237] = 4.85015581E+00;
    COFD[1238] = -4.10945109E-01;
    COFD[1239] = 1.76651398E-02;
    COFD[1240] = -1.59863030E+01;
    COFD[1241] = 3.67388294E+00;
    COFD[1242] = -2.64990709E-01;
    COFD[1243] = 1.16042706E-02;
    COFD[1244] = -2.14449559E+01;
    COFD[1245] = 5.56531152E+00;
    COFD[1246] = -4.88789821E-01;
    COFD[1247] = 2.04437116E-02;
    COFD[1248] = -1.73374529E+01;
    COFD[1249] = 4.21416723E+00;
    COFD[1250] = -3.34163932E-01;
    COFD[1251] = 1.45697432E-02;
    COFD[1252] = -1.72738845E+01;
    COFD[1253] = 4.19029808E+00;
    COFD[1254] = -3.31177076E-01;
    COFD[1255] = 1.44446234E-02;
    COFD[1256] = -2.13777308E+01;
    COFD[1257] = 5.54007827E+00;
    COFD[1258] = -4.86434511E-01;
    COFD[1259] = 2.03779006E-02;
    COFD[1260] = -1.94819080E+01;
    COFD[1261] = 4.87180830E+00;
    COFD[1262] = -4.13582958E-01;
    COFD[1263] = 1.77726094E-02;
    COFD[1264] = -2.11606963E+01;
    COFD[1265] = 5.42846112E+00;
    COFD[1266] = -4.74321870E-01;
    COFD[1267] = 1.99459749E-02;
    COFD[1268] = -2.11722423E+01;
    COFD[1269] = 5.42846112E+00;
    COFD[1270] = -4.74321870E-01;
    COFD[1271] = 1.99459749E-02;
    COFD[1272] = -1.95770968E+01;
    COFD[1273] = 4.97133070E+00;
    COFD[1274] = -4.25604177E-01;
    COFD[1275] = 1.82582594E-02;
    COFD[1276] = -2.02692384E+01;
    COFD[1277] = 5.14418672E+00;
    COFD[1278] = -4.45631004E-01;
    COFD[1279] = 1.90308403E-02;
    COFD[1280] = -1.91225414E+01;
    COFD[1281] = 4.82869066E+00;
    COFD[1282] = -4.08564514E-01;
    COFD[1283] = 1.75784675E-02;
    COFD[1284] = -2.04309557E+01;
    COFD[1285] = 5.18271974E+00;
    COFD[1286] = -4.49323627E-01;
    COFD[1287] = 1.91373940E-02;
    COFD[1288] = -2.09490548E+01;
    COFD[1289] = 5.31360223E+00;
    COFD[1290] = -4.64787000E-01;
    COFD[1291] = 1.97483720E-02;
    COFD[1292] = -2.09127554E+01;
    COFD[1293] = 5.30526648E+00;
    COFD[1294] = -4.63785596E-01;
    COFD[1295] = 1.97079873E-02;
    COFD[1296] = -2.19557531E+01;
    COFD[1297] = 5.49350509E+00;
    COFD[1298] = -4.81613405E-01;
    COFD[1299] = 2.02171734E-02;
    COFD[1300] = -1.59327297E+01;
    COFD[1301] = 3.65620899E+00;
    COFD[1302] = -2.62933804E-01;
    COFD[1303] = 1.15253223E-02;
    COFD[1304] = -1.50270339E+01;
    COFD[1305] = 3.46140064E+00;
    COFD[1306] = -2.38440092E-01;
    COFD[1307] = 1.04960087E-02;
    COFD[1308] = -1.24693568E+01;
    COFD[1309] = 2.76686648E+00;
    COFD[1310] = -1.49120141E-01;
    COFD[1311] = 6.66220432E-03;
    COFD[1312] = -1.57199037E+01;
    COFD[1313] = 4.19936335E+00;
    COFD[1314] = -3.32311009E-01;
    COFD[1315] = 1.44921003E-02;
    COFD[1316] = -1.50420953E+01;
    COFD[1317] = 3.46140064E+00;
    COFD[1318] = -2.38440092E-01;
    COFD[1319] = 1.04960087E-02;
    COFD[1320] = -2.14087397E+01;
    COFD[1321] = 5.57282008E+00;
    COFD[1322] = -4.76690890E-01;
    COFD[1323] = 1.94000719E-02;
    COFD[1324] = -1.62724462E+01;
    COFD[1325] = 3.79163564E+00;
    COFD[1326] = -2.80257365E-01;
    COFD[1327] = 1.22656902E-02;
    COFD[1328] = -1.62775714E+01;
    COFD[1329] = 3.79163564E+00;
    COFD[1330] = -2.80257365E-01;
    COFD[1331] = 1.22656902E-02;
    COFD[1332] = -1.62824412E+01;
    COFD[1333] = 3.79163564E+00;
    COFD[1334] = -2.80257365E-01;
    COFD[1335] = 1.22656902E-02;
    COFD[1336] = -1.92867554E+01;
    COFD[1337] = 4.83375900E+00;
    COFD[1338] = -4.09146560E-01;
    COFD[1339] = 1.76006599E-02;
    COFD[1340] = -1.59525102E+01;
    COFD[1341] = 3.66023858E+00;
    COFD[1342] = -2.63401043E-01;
    COFD[1343] = 1.15432000E-02;
    COFD[1344] = -2.14082453E+01;
    COFD[1345] = 5.55346617E+00;
    COFD[1346] = -4.87783156E-01;
    COFD[1347] = 2.04210886E-02;
    COFD[1348] = -1.72738845E+01;
    COFD[1349] = 4.19029808E+00;
    COFD[1350] = -3.31177076E-01;
    COFD[1351] = 1.44446234E-02;
    COFD[1352] = -1.72167708E+01;
    COFD[1353] = 4.16886779E+00;
    COFD[1354] = -3.28518156E-01;
    COFD[1355] = 1.43341626E-02;
    COFD[1356] = -2.13319784E+01;
    COFD[1357] = 5.52422470E+00;
    COFD[1358] = -4.84872944E-01;
    COFD[1359] = 2.03298213E-02;
    COFD[1360] = -1.94186547E+01;
    COFD[1361] = 4.84669430E+00;
    COFD[1362] = -4.10571455E-01;
    COFD[1363] = 1.76520543E-02;
    COFD[1364] = -2.11309207E+01;
    COFD[1365] = 5.41773516E+00;
    COFD[1366] = -4.73414338E-01;
    COFD[1367] = 1.99258685E-02;
    COFD[1368] = -2.11430338E+01;
    COFD[1369] = 5.41773516E+00;
    COFD[1370] = -4.73414338E-01;
    COFD[1371] = 1.99258685E-02;
    COFD[1372] = -1.95154079E+01;
    COFD[1373] = 4.94787350E+00;
    COFD[1374] = -4.22829292E-01;
    COFD[1375] = 1.81487163E-02;
    COFD[1376] = -2.02318658E+01;
    COFD[1377] = 5.12963391E+00;
    COFD[1378] = -4.44146826E-01;
    COFD[1379] = 1.89829640E-02;
    COFD[1380] = -1.90692595E+01;
    COFD[1381] = 4.80830699E+00;
    COFD[1382] = -4.06171933E-01;
    COFD[1383] = 1.74848791E-02;
    COFD[1384] = -2.03775651E+01;
    COFD[1385] = 5.16159436E+00;
    COFD[1386] = -4.46935283E-01;
    COFD[1387] = 1.90480297E-02;
    COFD[1388] = -2.08822487E+01;
    COFD[1389] = 5.28557747E+00;
    COFD[1390] = -4.61402384E-01;
    COFD[1391] = 1.96111546E-02;
    COFD[1392] = -2.08447974E+01;
    COFD[1393] = 5.27674330E+00;
    COFD[1394] = -4.60336155E-01;
    COFD[1395] = 1.95680191E-02;
    COFD[1396] = -2.19060847E+01;
    COFD[1397] = 5.47162499E+00;
    COFD[1398] = -4.79195552E-01;
    COFD[1399] = 2.01289088E-02;
    COFD[1400] = -2.03844252E+01;
    COFD[1401] = 5.18856872E+00;
    COFD[1402] = -4.50001829E-01;
    COFD[1403] = 1.91636142E-02;
    COFD[1404] = -1.93364585E+01;
    COFD[1405] = 4.98286777E+00;
    COFD[1406] = -4.26970814E-01;
    COFD[1407] = 1.83122917E-02;
    COFD[1408] = -1.59537247E+01;
    COFD[1409] = 4.07051484E+00;
    COFD[1410] = -3.16303109E-01;
    COFD[1411] = 1.38259377E-02;
    COFD[1412] = -1.96866103E+01;
    COFD[1413] = 5.54637286E+00;
    COFD[1414] = -4.87070324E-01;
    COFD[1415] = 2.03983467E-02;
    COFD[1416] = -1.93566243E+01;
    COFD[1417] = 4.98286777E+00;
    COFD[1418] = -4.26970814E-01;
    COFD[1419] = 1.83122917E-02;
    COFD[1420] = -1.80253664E+01;
    COFD[1421] = 3.69199168E+00;
    COFD[1422] = -1.74005516E-01;
    COFD[1423] = 3.97694372E-03;
    COFD[1424] = -2.07595845E+01;
    COFD[1425] = 5.32244593E+00;
    COFD[1426] = -4.65829403E-01;
    COFD[1427] = 1.97895274E-02;
    COFD[1428] = -2.07672833E+01;
    COFD[1429] = 5.32244593E+00;
    COFD[1430] = -4.65829403E-01;
    COFD[1431] = 1.97895274E-02;
    COFD[1432] = -2.07746356E+01;
    COFD[1433] = 5.32244593E+00;
    COFD[1434] = -4.65829403E-01;
    COFD[1435] = 1.97895274E-02;
    COFD[1436] = -2.20063594E+01;
    COFD[1437] = 5.48540187E+00;
    COFD[1438] = -4.58962148E-01;
    COFD[1439] = 1.83770355E-02;
    COFD[1440] = -2.04144604E+01;
    COFD[1441] = 5.19614628E+00;
    COFD[1442] = -4.50889164E-01;
    COFD[1443] = 1.91983328E-02;
    COFD[1444] = -1.93214527E+01;
    COFD[1445] = 4.10954793E+00;
    COFD[1446] = -2.37523329E-01;
    COFD[1447] = 7.08858141E-03;
    COFD[1448] = -2.13777308E+01;
    COFD[1449] = 5.54007827E+00;
    COFD[1450] = -4.86434511E-01;
    COFD[1451] = 2.03779006E-02;
    COFD[1452] = -2.13319784E+01;
    COFD[1453] = 5.52422470E+00;
    COFD[1454] = -4.84872944E-01;
    COFD[1455] = 2.03298213E-02;
    COFD[1456] = -1.95785144E+01;
    COFD[1457] = 4.22062499E+00;
    COFD[1458] = -2.54326872E-01;
    COFD[1459] = 7.91017784E-03;
    COFD[1460] = -2.20495822E+01;
    COFD[1461] = 5.47072190E+00;
    COFD[1462] = -4.56301261E-01;
    COFD[1463] = 1.82313566E-02;
    COFD[1464] = -2.03036402E+01;
    COFD[1465] = 4.50250781E+00;
    COFD[1466] = -2.97622106E-01;
    COFD[1467] = 1.00481473E-02;
    COFD[1468] = -2.03227406E+01;
    COFD[1469] = 4.50250781E+00;
    COFD[1470] = -2.97622106E-01;
    COFD[1471] = 1.00481473E-02;
    COFD[1472] = -2.17547312E+01;
    COFD[1473] = 5.40298848E+00;
    COFD[1474] = -4.43954594E-01;
    COFD[1475] = 1.75542998E-02;
    COFD[1476] = -2.16936515E+01;
    COFD[1477] = 5.21869603E+00;
    COFD[1478] = -4.12084772E-01;
    COFD[1479] = 1.58573035E-02;
    COFD[1480] = -2.18356866E+01;
    COFD[1481] = 5.49906960E+00;
    COFD[1482] = -4.61793001E-01;
    COFD[1483] = 1.85415189E-02;
    COFD[1484] = -2.15816909E+01;
    COFD[1485] = 5.13708607E+00;
    COFD[1486] = -3.98445708E-01;
    COFD[1487] = 1.51455626E-02;
    COFD[1488] = -2.13985484E+01;
    COFD[1489] = 4.94878244E+00;
    COFD[1490] = -3.68158605E-01;
    COFD[1491] = 1.36008797E-02;
    COFD[1492] = -2.14144448E+01;
    COFD[1493] = 4.96219227E+00;
    COFD[1494] = -3.70270843E-01;
    COFD[1495] = 1.37072211E-02;
    COFD[1496] = -2.06870442E+01;
    COFD[1497] = 4.35920123E+00;
    COFD[1498] = -2.75491273E-01;
    COFD[1499] = 8.95100289E-03;
    COFD[1500] = -1.82673770E+01;
    COFD[1501] = 4.39538102E+00;
    COFD[1502] = -3.56367230E-01;
    COFD[1503] = 1.54788461E-02;
    COFD[1504] = -1.72112971E+01;
    COFD[1505] = 4.15807461E+00;
    COFD[1506] = -3.27178539E-01;
    COFD[1507] = 1.42784349E-02;
    COFD[1508] = -1.39658996E+01;
    COFD[1509] = 3.24966086E+00;
    COFD[1510] = -2.11199992E-01;
    COFD[1511] = 9.32580661E-03;
    COFD[1512] = -1.78637178E+01;
    COFD[1513] = 4.88268692E+00;
    COFD[1514] = -4.14917638E-01;
    COFD[1515] = 1.78274298E-02;
    COFD[1516] = -1.72310232E+01;
    COFD[1517] = 4.15807461E+00;
    COFD[1518] = -3.27178539E-01;
    COFD[1519] = 1.42784349E-02;
    COFD[1520] = -2.13148887E+01;
    COFD[1521] = 5.27210469E+00;
    COFD[1522] = -4.21419216E-01;
    COFD[1523] = 1.63567178E-02;
    COFD[1524] = -1.85844688E+01;
    COFD[1525] = 4.51052425E+00;
    COFD[1526] = -3.70301627E-01;
    COFD[1527] = 1.60416153E-02;
    COFD[1528] = -1.85919214E+01;
    COFD[1529] = 4.51052425E+00;
    COFD[1530] = -3.70301627E-01;
    COFD[1531] = 1.60416153E-02;
    COFD[1532] = -1.85990352E+01;
    COFD[1533] = 4.51052425E+00;
    COFD[1534] = -3.70301627E-01;
    COFD[1535] = 1.60416153E-02;
    COFD[1536] = -2.14151520E+01;
    COFD[1537] = 5.41122754E+00;
    COFD[1538] = -4.73185889E-01;
    COFD[1539] = 1.99407905E-02;
    COFD[1540] = -1.82955252E+01;
    COFD[1541] = 4.40289649E+00;
    COFD[1542] = -3.57289765E-01;
    COFD[1543] = 1.55166804E-02;
    COFD[1544] = -2.19786173E+01;
    COFD[1545] = 5.43750833E+00;
    COFD[1546] = -4.50273329E-01;
    COFD[1547] = 1.79013718E-02;
    COFD[1548] = -1.94819080E+01;
    COFD[1549] = 4.87180830E+00;
    COFD[1550] = -4.13582958E-01;
    COFD[1551] = 1.77726094E-02;
    COFD[1552] = -1.94186547E+01;
    COFD[1553] = 4.84669430E+00;
    COFD[1554] = -4.10571455E-01;
    COFD[1555] = 1.76520543E-02;
    COFD[1556] = -2.20495822E+01;
    COFD[1557] = 5.47072190E+00;
    COFD[1558] = -4.56301261E-01;
    COFD[1559] = 1.82313566E-02;
    COFD[1560] = -2.14907782E+01;
    COFD[1561] = 5.41585806E+00;
    COFD[1562] = -4.73359323E-01;
    COFD[1563] = 1.99310239E-02;
    COFD[1564] = -2.22429814E+01;
    COFD[1565] = 5.53139819E+00;
    COFD[1566] = -4.68828555E-01;
    COFD[1567] = 1.89597887E-02;
    COFD[1568] = -2.22613837E+01;
    COFD[1569] = 5.53139819E+00;
    COFD[1570] = -4.68828555E-01;
    COFD[1571] = 1.89597887E-02;
    COFD[1572] = -2.15206146E+01;
    COFD[1573] = 5.48426911E+00;
    COFD[1574] = -4.80606512E-01;
    COFD[1575] = 2.01811046E-02;
    COFD[1576] = -2.21343023E+01;
    COFD[1577] = 5.60010742E+00;
    COFD[1578] = -4.91597429E-01;
    COFD[1579] = 2.04987718E-02;
    COFD[1580] = -2.12014186E+01;
    COFD[1581] = 5.40060531E+00;
    COFD[1582] = -4.72449699E-01;
    COFD[1583] = 1.99345817E-02;
    COFD[1584] = -2.22317182E+01;
    COFD[1585] = 5.61211818E+00;
    COFD[1586] = -4.91432482E-01;
    COFD[1587] = 2.04238731E-02;
    COFD[1588] = -2.24120415E+01;
    COFD[1589] = 5.58744076E+00;
    COFD[1590] = -4.84489462E-01;
    COFD[1591] = 1.99733042E-02;
    COFD[1592] = -2.24025650E+01;
    COFD[1593] = 5.58952429E+00;
    COFD[1594] = -4.85012530E-01;
    COFD[1595] = 2.00062142E-02;
    COFD[1596] = -2.28458380E+01;
    COFD[1597] = 5.50134401E+00;
    COFD[1598] = -4.62488197E-01;
    COFD[1599] = 1.85873697E-02;
    COFD[1600] = -2.02646611E+01;
    COFD[1601] = 5.10426133E+00;
    COFD[1602] = -4.41256919E-01;
    COFD[1603] = 1.88737290E-02;
    COFD[1604] = -1.90883268E+01;
    COFD[1605] = 4.84384483E+00;
    COFD[1606] = -4.10265575E-01;
    COFD[1607] = 1.76414287E-02;
    COFD[1608] = -1.57034851E+01;
    COFD[1609] = 3.93614244E+00;
    COFD[1610] = -2.99111497E-01;
    COFD[1611] = 1.30888229E-02;
    COFD[1612] = -1.94688688E+01;
    COFD[1613] = 5.43830787E+00;
    COFD[1614] = -4.75472880E-01;
    COFD[1615] = 1.99909996E-02;
    COFD[1616] = -1.91102652E+01;
    COFD[1617] = 4.84384483E+00;
    COFD[1618] = -4.10265575E-01;
    COFD[1619] = 1.76414287E-02;
    COFD[1620] = -1.87383952E+01;
    COFD[1621] = 3.96926341E+00;
    COFD[1622] = -2.16412264E-01;
    COFD[1623] = 6.06012078E-03;
    COFD[1624] = -2.05184870E+01;
    COFD[1625] = 5.18417470E+00;
    COFD[1626] = -4.49491573E-01;
    COFD[1627] = 1.91438508E-02;
    COFD[1628] = -2.05272328E+01;
    COFD[1629] = 5.18417470E+00;
    COFD[1630] = -4.49491573E-01;
    COFD[1631] = 1.91438508E-02;
    COFD[1632] = -2.05356023E+01;
    COFD[1633] = 5.18417470E+00;
    COFD[1634] = -4.49491573E-01;
    COFD[1635] = 1.91438508E-02;
    COFD[1636] = -2.22116706E+01;
    COFD[1637] = 5.54251230E+00;
    COFD[1638] = -4.70946314E-01;
    COFD[1639] = 1.90785869E-02;
    COFD[1640] = -2.02922701E+01;
    COFD[1641] = 5.11106992E+00;
    COFD[1642] = -4.42047129E-01;
    COFD[1643] = 1.89042990E-02;
    COFD[1644] = -2.01015340E+01;
    COFD[1645] = 4.41511629E+00;
    COFD[1646] = -2.84086963E-01;
    COFD[1647] = 9.37586971E-03;
    COFD[1648] = -2.11606963E+01;
    COFD[1649] = 5.42846112E+00;
    COFD[1650] = -4.74321870E-01;
    COFD[1651] = 1.99459749E-02;
    COFD[1652] = -2.11309207E+01;
    COFD[1653] = 5.41773516E+00;
    COFD[1654] = -4.73414338E-01;
    COFD[1655] = 1.99258685E-02;
    COFD[1656] = -2.03036402E+01;
    COFD[1657] = 4.50250781E+00;
    COFD[1658] = -2.97622106E-01;
    COFD[1659] = 1.00481473E-02;
    COFD[1660] = -2.22429814E+01;
    COFD[1661] = 5.53139819E+00;
    COFD[1662] = -4.68828555E-01;
    COFD[1663] = 1.89597887E-02;
    COFD[1664] = -2.09002742E+01;
    COFD[1665] = 4.72895031E+00;
    COFD[1666] = -3.33332771E-01;
    COFD[1667] = 1.18431478E-02;
    COFD[1668] = -2.09224206E+01;
    COFD[1669] = 4.72895031E+00;
    COFD[1670] = -3.33332771E-01;
    COFD[1671] = 1.18431478E-02;
    COFD[1672] = -2.20262793E+01;
    COFD[1673] = 5.49663315E+00;
    COFD[1674] = -4.61182837E-01;
    COFD[1675] = 1.85035558E-02;
    COFD[1676] = -2.20597305E+01;
    COFD[1677] = 5.34774760E+00;
    COFD[1678] = -4.34239753E-01;
    COFD[1679] = 1.70320676E-02;
    COFD[1680] = -2.20398328E+01;
    COFD[1681] = 5.56049839E+00;
    COFD[1682] = -4.74367872E-01;
    COFD[1683] = 1.92702787E-02;
    COFD[1684] = -2.19592125E+01;
    COFD[1685] = 5.27258289E+00;
    COFD[1686] = -4.21502790E-01;
    COFD[1687] = 1.63611949E-02;
    COFD[1688] = -2.19253091E+01;
    COFD[1689] = 5.14570932E+00;
    COFD[1690] = -3.99877142E-01;
    COFD[1691] = 1.52199557E-02;
    COFD[1692] = -2.19322003E+01;
    COFD[1693] = 5.15446948E+00;
    COFD[1694] = -4.01332769E-01;
    COFD[1695] = 1.52956262E-02;
    COFD[1696] = -2.13539532E+01;
    COFD[1697] = 4.61201872E+00;
    COFD[1698] = -3.14803338E-01;
    COFD[1699] = 1.09082984E-02;
    COFD[1700] = -2.02822946E+01;
    COFD[1701] = 5.10426133E+00;
    COFD[1702] = -4.41256919E-01;
    COFD[1703] = 1.88737290E-02;
    COFD[1704] = -1.91004157E+01;
    COFD[1705] = 4.84384483E+00;
    COFD[1706] = -4.10265575E-01;
    COFD[1707] = 1.76414287E-02;
    COFD[1708] = -1.57054717E+01;
    COFD[1709] = 3.93614244E+00;
    COFD[1710] = -2.99111497E-01;
    COFD[1711] = 1.30888229E-02;
    COFD[1712] = -1.94698843E+01;
    COFD[1713] = 5.43830787E+00;
    COFD[1714] = -4.75472880E-01;
    COFD[1715] = 1.99909996E-02;
    COFD[1716] = -1.91229033E+01;
    COFD[1717] = 4.84384483E+00;
    COFD[1718] = -4.10265575E-01;
    COFD[1719] = 1.76414287E-02;
    COFD[1720] = -1.87515645E+01;
    COFD[1721] = 3.96926341E+00;
    COFD[1722] = -2.16412264E-01;
    COFD[1723] = 6.06012078E-03;
    COFD[1724] = -2.05375724E+01;
    COFD[1725] = 5.18417470E+00;
    COFD[1726] = -4.49491573E-01;
    COFD[1727] = 1.91438508E-02;
    COFD[1728] = -2.05466616E+01;
    COFD[1729] = 5.18417470E+00;
    COFD[1730] = -4.49491573E-01;
    COFD[1731] = 1.91438508E-02;
    COFD[1732] = -2.05553656E+01;
    COFD[1733] = 5.18417470E+00;
    COFD[1734] = -4.49491573E-01;
    COFD[1735] = 1.91438508E-02;
    COFD[1736] = -2.22343363E+01;
    COFD[1737] = 5.54251230E+00;
    COFD[1738] = -4.70946314E-01;
    COFD[1739] = 1.90785869E-02;
    COFD[1740] = -2.03099025E+01;
    COFD[1741] = 5.11106992E+00;
    COFD[1742] = -4.42047129E-01;
    COFD[1743] = 1.89042990E-02;
    COFD[1744] = -2.01199204E+01;
    COFD[1745] = 4.41511629E+00;
    COFD[1746] = -2.84086963E-01;
    COFD[1747] = 9.37586971E-03;
    COFD[1748] = -2.11722423E+01;
    COFD[1749] = 5.42846112E+00;
    COFD[1750] = -4.74321870E-01;
    COFD[1751] = 1.99459749E-02;
    COFD[1752] = -2.11430338E+01;
    COFD[1753] = 5.41773516E+00;
    COFD[1754] = -4.73414338E-01;
    COFD[1755] = 1.99258685E-02;
    COFD[1756] = -2.03227406E+01;
    COFD[1757] = 4.50250781E+00;
    COFD[1758] = -2.97622106E-01;
    COFD[1759] = 1.00481473E-02;
    COFD[1760] = -2.22613837E+01;
    COFD[1761] = 5.53139819E+00;
    COFD[1762] = -4.68828555E-01;
    COFD[1763] = 1.89597887E-02;
    COFD[1764] = -2.09224206E+01;
    COFD[1765] = 4.72895031E+00;
    COFD[1766] = -3.33332771E-01;
    COFD[1767] = 1.18431478E-02;
    COFD[1768] = -2.09455936E+01;
    COFD[1769] = 4.72895031E+00;
    COFD[1770] = -3.33332771E-01;
    COFD[1771] = 1.18431478E-02;
    COFD[1772] = -2.20431319E+01;
    COFD[1773] = 5.49663315E+00;
    COFD[1774] = -4.61182837E-01;
    COFD[1775] = 1.85035558E-02;
    COFD[1776] = -2.20818886E+01;
    COFD[1777] = 5.34774760E+00;
    COFD[1778] = -4.34239753E-01;
    COFD[1779] = 1.70320676E-02;
    COFD[1780] = -2.20574820E+01;
    COFD[1781] = 5.56049839E+00;
    COFD[1782] = -4.74367872E-01;
    COFD[1783] = 1.92702787E-02;
    COFD[1784] = -2.19808152E+01;
    COFD[1785] = 5.27258289E+00;
    COFD[1786] = -4.21502790E-01;
    COFD[1787] = 1.63611949E-02;
    COFD[1788] = -2.19503032E+01;
    COFD[1789] = 5.14570932E+00;
    COFD[1790] = -3.99877142E-01;
    COFD[1791] = 1.52199557E-02;
    COFD[1792] = -2.19576037E+01;
    COFD[1793] = 5.15446948E+00;
    COFD[1794] = -4.01332769E-01;
    COFD[1795] = 1.52956262E-02;
    COFD[1796] = -2.13854464E+01;
    COFD[1797] = 4.61201872E+00;
    COFD[1798] = -3.14803338E-01;
    COFD[1799] = 1.09082984E-02;
    COFD[1800] = -1.83039618E+01;
    COFD[1801] = 4.47952077E+00;
    COFD[1802] = -3.66569471E-01;
    COFD[1803] = 1.58916129E-02;
    COFD[1804] = -1.72286007E+01;
    COFD[1805] = 4.24084025E+00;
    COFD[1806] = -3.37428619E-01;
    COFD[1807] = 1.47032793E-02;
    COFD[1808] = -1.39315266E+01;
    COFD[1809] = 3.30394764E+00;
    COFD[1810] = -2.17920112E-01;
    COFD[1811] = 9.60284243E-03;
    COFD[1812] = -1.79310765E+01;
    COFD[1813] = 4.98037650E+00;
    COFD[1814] = -4.26676911E-01;
    COFD[1815] = 1.83007231E-02;
    COFD[1816] = -1.72473011E+01;
    COFD[1817] = 4.24084025E+00;
    COFD[1818] = -3.37428619E-01;
    COFD[1819] = 1.47032793E-02;
    COFD[1820] = -2.09565916E+01;
    COFD[1821] = 5.18380539E+00;
    COFD[1822] = -4.06234719E-01;
    COFD[1823] = 1.55515345E-02;
    COFD[1824] = -1.86507213E+01;
    COFD[1825] = 4.60874797E+00;
    COFD[1826] = -3.82368716E-01;
    COFD[1827] = 1.65370164E-02;
    COFD[1828] = -1.86576191E+01;
    COFD[1829] = 4.60874797E+00;
    COFD[1830] = -3.82368716E-01;
    COFD[1831] = 1.65370164E-02;
    COFD[1832] = -1.86641962E+01;
    COFD[1833] = 4.60874797E+00;
    COFD[1834] = -3.82368716E-01;
    COFD[1835] = 1.65370164E-02;
    COFD[1836] = -2.13961414E+01;
    COFD[1837] = 5.46685775E+00;
    COFD[1838] = -4.78665416E-01;
    COFD[1839] = 2.01093915E-02;
    COFD[1840] = -1.83296965E+01;
    COFD[1841] = 4.48570999E+00;
    COFD[1842] = -3.67301524E-01;
    COFD[1843] = 1.59204254E-02;
    COFD[1844] = -2.16798265E+01;
    COFD[1845] = 5.36811769E+00;
    COFD[1846] = -4.37727086E-01;
    COFD[1847] = 1.72167686E-02;
    COFD[1848] = -1.95770968E+01;
    COFD[1849] = 4.97133070E+00;
    COFD[1850] = -4.25604177E-01;
    COFD[1851] = 1.82582594E-02;
    COFD[1852] = -1.95154079E+01;
    COFD[1853] = 4.94787350E+00;
    COFD[1854] = -4.22829292E-01;
    COFD[1855] = 1.81487163E-02;
    COFD[1856] = -2.17547312E+01;
    COFD[1857] = 5.40298848E+00;
    COFD[1858] = -4.43954594E-01;
    COFD[1859] = 1.75542998E-02;
    COFD[1860] = -2.15206146E+01;
    COFD[1861] = 5.48426911E+00;
    COFD[1862] = -4.80606512E-01;
    COFD[1863] = 2.01811046E-02;
    COFD[1864] = -2.20262793E+01;
    COFD[1865] = 5.49663315E+00;
    COFD[1866] = -4.61182837E-01;
    COFD[1867] = 1.85035558E-02;
    COFD[1868] = -2.20431319E+01;
    COFD[1869] = 5.49663315E+00;
    COFD[1870] = -4.61182837E-01;
    COFD[1871] = 1.85035558E-02;
    COFD[1872] = -2.15453676E+01;
    COFD[1873] = 5.55313619E+00;
    COFD[1874] = -4.87753729E-01;
    COFD[1875] = 2.04203421E-02;
    COFD[1876] = -2.20228343E+01;
    COFD[1877] = 5.61211028E+00;
    COFD[1878] = -4.90893171E-01;
    COFD[1879] = 2.03793118E-02;
    COFD[1880] = -2.11427744E+01;
    COFD[1881] = 5.43893233E+00;
    COFD[1882] = -4.75546039E-01;
    COFD[1883] = 1.99938690E-02;
    COFD[1884] = -2.20606550E+01;
    COFD[1885] = 5.59649805E+00;
    COFD[1886] = -4.86750336E-01;
    COFD[1887] = 2.01151498E-02;
    COFD[1888] = -2.22801170E+01;
    COFD[1889] = 5.58507108E+00;
    COFD[1890] = -4.81395065E-01;
    COFD[1891] = 1.97276199E-02;
    COFD[1892] = -2.22638165E+01;
    COFD[1893] = 5.58490856E+00;
    COFD[1894] = -4.81588720E-01;
    COFD[1895] = 1.97445317E-02;
    COFD[1896] = -2.26099899E+01;
    COFD[1897] = 5.44867280E+00;
    COFD[1898] = -4.52284883E-01;
    COFD[1899] = 1.80110706E-02;
    COFD[1900] = -1.90859283E+01;
    COFD[1901] = 4.68079396E+00;
    COFD[1902] = -3.91231550E-01;
    COFD[1903] = 1.69021170E-02;
    COFD[1904] = -1.79361160E+01;
    COFD[1905] = 4.42139452E+00;
    COFD[1906] = -3.59567329E-01;
    COFD[1907] = 1.56103969E-02;
    COFD[1908] = -1.45715797E+01;
    COFD[1909] = 3.49477850E+00;
    COFD[1910] = -2.42635772E-01;
    COFD[1911] = 1.06721490E-02;
    COFD[1912] = -1.85748546E+01;
    COFD[1913] = 5.14789919E+00;
    COFD[1914] = -4.45930850E-01;
    COFD[1915] = 1.90363341E-02;
    COFD[1916] = -1.79580609E+01;
    COFD[1917] = 4.42139452E+00;
    COFD[1918] = -3.59567329E-01;
    COFD[1919] = 1.56103969E-02;
    COFD[1920] = -2.06310304E+01;
    COFD[1921] = 4.89289496E+00;
    COFD[1922] = -3.59346263E-01;
    COFD[1923] = 1.31570901E-02;
    COFD[1924] = -1.93917298E+01;
    COFD[1925] = 4.78708023E+00;
    COFD[1926] = -4.03693144E-01;
    COFD[1927] = 1.73884817E-02;
    COFD[1928] = -1.94004795E+01;
    COFD[1929] = 4.78708023E+00;
    COFD[1930] = -4.03693144E-01;
    COFD[1931] = 1.73884817E-02;
    COFD[1932] = -1.94088529E+01;
    COFD[1933] = 4.78708023E+00;
    COFD[1934] = -4.03693144E-01;
    COFD[1935] = 1.73884817E-02;
    COFD[1936] = -2.20725883E+01;
    COFD[1937] = 5.59642965E+00;
    COFD[1938] = -4.91577716E-01;
    COFD[1939] = 2.05159582E-02;
    COFD[1940] = -1.91118445E+01;
    COFD[1941] = 4.68715685E+00;
    COFD[1942] = -3.91979493E-01;
    COFD[1943] = 1.69314004E-02;
    COFD[1944] = -2.15802788E+01;
    COFD[1945] = 5.16868516E+00;
    COFD[1946] = -4.03721581E-01;
    COFD[1947] = 1.54206640E-02;
    COFD[1948] = -2.02692384E+01;
    COFD[1949] = 5.14418672E+00;
    COFD[1950] = -4.45631004E-01;
    COFD[1951] = 1.90308403E-02;
    COFD[1952] = -2.02318658E+01;
    COFD[1953] = 5.12963391E+00;
    COFD[1954] = -4.44146826E-01;
    COFD[1955] = 1.89829640E-02;
    COFD[1956] = -2.16936515E+01;
    COFD[1957] = 5.21869603E+00;
    COFD[1958] = -4.12084772E-01;
    COFD[1959] = 1.58573035E-02;
    COFD[1960] = -2.21343023E+01;
    COFD[1961] = 5.60010742E+00;
    COFD[1962] = -4.91597429E-01;
    COFD[1963] = 2.04987718E-02;
    COFD[1964] = -2.20597305E+01;
    COFD[1965] = 5.34774760E+00;
    COFD[1966] = -4.34239753E-01;
    COFD[1967] = 1.70320676E-02;
    COFD[1968] = -2.20818886E+01;
    COFD[1969] = 5.34774760E+00;
    COFD[1970] = -4.34239753E-01;
    COFD[1971] = 1.70320676E-02;
    COFD[1972] = -2.20228343E+01;
    COFD[1973] = 5.61211028E+00;
    COFD[1974] = -4.90893171E-01;
    COFD[1975] = 2.03793118E-02;
    COFD[1976] = -2.23318349E+01;
    COFD[1977] = 5.58508387E+00;
    COFD[1978] = -4.81385216E-01;
    COFD[1979] = 1.97267369E-02;
    COFD[1980] = -2.18222696E+01;
    COFD[1981] = 5.57940140E+00;
    COFD[1982] = -4.89964112E-01;
    COFD[1983] = 2.04689539E-02;
    COFD[1984] = -2.23996837E+01;
    COFD[1985] = 5.58325398E+00;
    COFD[1986] = -4.79084067E-01;
    COFD[1987] = 1.95452935E-02;
    COFD[1988] = -2.25041734E+01;
    COFD[1989] = 5.51797622E+00;
    COFD[1990] = -4.66229499E-01;
    COFD[1991] = 1.88128348E-02;
    COFD[1992] = -2.25004333E+01;
    COFD[1993] = 5.52198915E+00;
    COFD[1994] = -4.67014474E-01;
    COFD[1995] = 1.88574253E-02;
    COFD[1996] = -2.26044889E+01;
    COFD[1997] = 5.27383847E+00;
    COFD[1998] = -4.21722368E-01;
    COFD[1999] = 1.63729618E-02;
    COFD[2000] = -1.78815889E+01;
    COFD[2001] = 4.34347890E+00;
    COFD[2002] = -3.49890003E-01;
    COFD[2003] = 1.52083459E-02;
    COFD[2004] = -1.68343393E+01;
    COFD[2005] = 4.11954900E+00;
    COFD[2006] = -3.22470391E-01;
    COFD[2007] = 1.40859564E-02;
    COFD[2008] = -1.36336373E+01;
    COFD[2009] = 3.22088176E+00;
    COFD[2010] = -2.07623790E-01;
    COFD[2011] = 9.17771542E-03;
    COFD[2012] = -1.74407963E+01;
    COFD[2013] = 4.83580036E+00;
    COFD[2014] = -4.09383573E-01;
    COFD[2015] = 1.76098175E-02;
    COFD[2016] = -1.68535757E+01;
    COFD[2017] = 4.11954900E+00;
    COFD[2018] = -3.22470391E-01;
    COFD[2019] = 1.40859564E-02;
    COFD[2020] = -2.11309197E+01;
    COFD[2021] = 5.32644193E+00;
    COFD[2022] = -4.30581064E-01;
    COFD[2023] = 1.68379725E-02;
    COFD[2024] = -1.82145353E+01;
    COFD[2025] = 4.46848269E+00;
    COFD[2026] = -3.65269718E-01;
    COFD[2027] = 1.58407652E-02;
    COFD[2028] = -1.82217198E+01;
    COFD[2029] = 4.46848269E+00;
    COFD[2030] = -3.65269718E-01;
    COFD[2031] = 1.58407652E-02;
    COFD[2032] = -1.82285740E+01;
    COFD[2033] = 4.46848269E+00;
    COFD[2034] = -3.65269718E-01;
    COFD[2035] = 1.58407652E-02;
    COFD[2036] = -2.11031143E+01;
    COFD[2037] = 5.39439999E+00;
    COFD[2038] = -4.72050184E-01;
    COFD[2039] = 1.99336257E-02;
    COFD[2040] = -1.79116531E+01;
    COFD[2041] = 4.35148286E+00;
    COFD[2042] = -3.50886647E-01;
    COFD[2043] = 1.52498573E-02;
    COFD[2044] = -2.17855148E+01;
    COFD[2045] = 5.47519298E+00;
    COFD[2046] = -4.57113040E-01;
    COFD[2047] = 1.82758312E-02;
    COFD[2048] = -1.91225414E+01;
    COFD[2049] = 4.82869066E+00;
    COFD[2050] = -4.08564514E-01;
    COFD[2051] = 1.75784675E-02;
    COFD[2052] = -1.90692595E+01;
    COFD[2053] = 4.80830699E+00;
    COFD[2054] = -4.06171933E-01;
    COFD[2055] = 1.74848791E-02;
    COFD[2056] = -2.18356866E+01;
    COFD[2057] = 5.49906960E+00;
    COFD[2058] = -4.61793001E-01;
    COFD[2059] = 1.85415189E-02;
    COFD[2060] = -2.12014186E+01;
    COFD[2061] = 5.40060531E+00;
    COFD[2062] = -4.72449699E-01;
    COFD[2063] = 1.99345817E-02;
    COFD[2064] = -2.20398328E+01;
    COFD[2065] = 5.56049839E+00;
    COFD[2066] = -4.74367872E-01;
    COFD[2067] = 1.92702787E-02;
    COFD[2068] = -2.20574820E+01;
    COFD[2069] = 5.56049839E+00;
    COFD[2070] = -4.74367872E-01;
    COFD[2071] = 1.92702787E-02;
    COFD[2072] = -2.11427744E+01;
    COFD[2073] = 5.43893233E+00;
    COFD[2074] = -4.75546039E-01;
    COFD[2075] = 1.99938690E-02;
    COFD[2076] = -2.18222696E+01;
    COFD[2077] = 5.57940140E+00;
    COFD[2078] = -4.89964112E-01;
    COFD[2079] = 2.04689539E-02;
    COFD[2080] = -2.08820897E+01;
    COFD[2081] = 5.38250415E+00;
    COFD[2082] = -4.71144140E-01;
    COFD[2083] = 1.99199779E-02;
    COFD[2084] = -2.19501296E+01;
    COFD[2085] = 5.60255148E+00;
    COFD[2086] = -4.91366572E-01;
    COFD[2087] = 2.04670553E-02;
    COFD[2088] = -2.21913393E+01;
    COFD[2089] = 5.60175327E+00;
    COFD[2090] = -4.87953216E-01;
    COFD[2091] = 2.01882171E-02;
    COFD[2092] = -2.21822461E+01;
    COFD[2093] = 5.60465338E+00;
    COFD[2094] = -4.88572478E-01;
    COFD[2095] = 2.02248525E-02;
    COFD[2096] = -2.26591038E+01;
    COFD[2097] = 5.52001624E+00;
    COFD[2098] = -4.66629503E-01;
    COFD[2099] = 1.88355817E-02;
    COFD[2100] = -1.92783884E+01;
    COFD[2101] = 4.73660584E+00;
    COFD[2102] = -3.97704978E-01;
    COFD[2103] = 1.71514887E-02;
    COFD[2104] = -1.81499793E+01;
    COFD[2105] = 4.48398491E+00;
    COFD[2106] = -3.67097129E-01;
    COFD[2107] = 1.59123634E-02;
    COFD[2108] = -1.47725694E+01;
    COFD[2109] = 3.55444478E+00;
    COFD[2110] = -2.50272707E-01;
    COFD[2111] = 1.09990787E-02;
    COFD[2112] = -1.87647862E+01;
    COFD[2113] = 5.19146813E+00;
    COFD[2114] = -4.50340408E-01;
    COFD[2115] = 1.91768178E-02;
    COFD[2116] = -1.81716176E+01;
    COFD[2117] = 4.48398491E+00;
    COFD[2118] = -3.67097129E-01;
    COFD[2119] = 1.59123634E-02;
    COFD[2120] = -2.04397451E+01;
    COFD[2121] = 4.77398686E+00;
    COFD[2122] = -3.40522956E-01;
    COFD[2123] = 1.22072846E-02;
    COFD[2124] = -1.95875976E+01;
    COFD[2125] = 4.84393038E+00;
    COFD[2126] = -4.10274737E-01;
    COFD[2127] = 1.76417458E-02;
    COFD[2128] = -1.95961596E+01;
    COFD[2129] = 4.84393038E+00;
    COFD[2130] = -4.10274737E-01;
    COFD[2131] = 1.76417458E-02;
    COFD[2132] = -1.96043503E+01;
    COFD[2133] = 4.84393038E+00;
    COFD[2134] = -4.10274737E-01;
    COFD[2135] = 1.76417458E-02;
    COFD[2136] = -2.21697404E+01;
    COFD[2137] = 5.60807471E+00;
    COFD[2138] = -4.91339309E-01;
    COFD[2139] = 2.04365761E-02;
    COFD[2140] = -1.93064215E+01;
    COFD[2141] = 4.74387793E+00;
    COFD[2142] = -3.98574972E-01;
    COFD[2143] = 1.71862289E-02;
    COFD[2144] = -2.14453157E+01;
    COFD[2145] = 5.07680397E+00;
    COFD[2146] = -3.88612087E-01;
    COFD[2147] = 1.46395101E-02;
    COFD[2148] = -2.04309557E+01;
    COFD[2149] = 5.18271974E+00;
    COFD[2150] = -4.49323627E-01;
    COFD[2151] = 1.91373940E-02;
    COFD[2152] = -2.03775651E+01;
    COFD[2153] = 5.16159436E+00;
    COFD[2154] = -4.46935283E-01;
    COFD[2155] = 1.90480297E-02;
    COFD[2156] = -2.15816909E+01;
    COFD[2157] = 5.13708607E+00;
    COFD[2158] = -3.98445708E-01;
    COFD[2159] = 1.51455626E-02;
    COFD[2160] = -2.22317182E+01;
    COFD[2161] = 5.61211818E+00;
    COFD[2162] = -4.91432482E-01;
    COFD[2163] = 2.04238731E-02;
    COFD[2164] = -2.19592125E+01;
    COFD[2165] = 5.27258289E+00;
    COFD[2166] = -4.21502790E-01;
    COFD[2167] = 1.63611949E-02;
    COFD[2168] = -2.19808152E+01;
    COFD[2169] = 5.27258289E+00;
    COFD[2170] = -4.21502790E-01;
    COFD[2171] = 1.63611949E-02;
    COFD[2172] = -2.20606550E+01;
    COFD[2173] = 5.59649805E+00;
    COFD[2174] = -4.86750336E-01;
    COFD[2175] = 2.01151498E-02;
    COFD[2176] = -2.23996837E+01;
    COFD[2177] = 5.58325398E+00;
    COFD[2178] = -4.79084067E-01;
    COFD[2179] = 1.95452935E-02;
    COFD[2180] = -2.19501296E+01;
    COFD[2181] = 5.60255148E+00;
    COFD[2182] = -4.91366572E-01;
    COFD[2183] = 2.04670553E-02;
    COFD[2184] = -2.23915398E+01;
    COFD[2185] = 5.54890339E+00;
    COFD[2186] = -4.72166228E-01;
    COFD[2187] = 1.91470071E-02;
    COFD[2188] = -2.25219004E+01;
    COFD[2189] = 5.49554403E+00;
    COFD[2190] = -4.60936491E-01;
    COFD[2191] = 1.84887572E-02;
    COFD[2192] = -2.25140525E+01;
    COFD[2193] = 5.49776513E+00;
    COFD[2194] = -4.61463030E-01;
    COFD[2195] = 1.85209236E-02;
    COFD[2196] = -2.25175307E+01;
    COFD[2197] = 5.21003123E+00;
    COFD[2198] = -4.10612564E-01;
    COFD[2199] = 1.57798598E-02;
    COFD[2200] = -1.97484166E+01;
    COFD[2201] = 4.84231878E+00;
    COFD[2202] = -4.10101001E-01;
    COFD[2203] = 1.76356687E-02;
    COFD[2204] = -1.86652603E+01;
    COFD[2205] = 4.61260432E+00;
    COFD[2206] = -3.82854484E-01;
    COFD[2207] = 1.65575163E-02;
    COFD[2208] = -1.51448279E+01;
    COFD[2209] = 3.64565939E+00;
    COFD[2210] = -2.61726871E-01;
    COFD[2211] = 1.14799244E-02;
    COFD[2212] = -1.92784178E+01;
    COFD[2213] = 5.32291505E+00;
    COFD[2214] = -4.65883522E-01;
    COFD[2215] = 1.97916109E-02;
    COFD[2216] = -1.86886689E+01;
    COFD[2217] = 4.61260432E+00;
    COFD[2218] = -3.82854484E-01;
    COFD[2219] = 1.65575163E-02;
    COFD[2220] = -2.02184916E+01;
    COFD[2221] = 4.57152878E+00;
    COFD[2222] = -3.08371263E-01;
    COFD[2223] = 1.05838559E-02;
    COFD[2224] = -2.01315602E+01;
    COFD[2225] = 4.97613338E+00;
    COFD[2226] = -4.26175206E-01;
    COFD[2227] = 1.82809270E-02;
    COFD[2228] = -2.01412473E+01;
    COFD[2229] = 4.97613338E+00;
    COFD[2230] = -4.26175206E-01;
    COFD[2231] = 1.82809270E-02;
    COFD[2232] = -2.01505348E+01;
    COFD[2233] = 4.97613338E+00;
    COFD[2234] = -4.26175206E-01;
    COFD[2235] = 1.82809270E-02;
    COFD[2236] = -2.23890317E+01;
    COFD[2237] = 5.59178974E+00;
    COFD[2238] = -4.85668031E-01;
    COFD[2239] = 2.00491907E-02;
    COFD[2240] = -1.97709603E+01;
    COFD[2241] = 4.84731557E+00;
    COFD[2242] = -4.10638352E-01;
    COFD[2243] = 1.76543886E-02;
    COFD[2244] = -2.12219677E+01;
    COFD[2245] = 4.87252053E+00;
    COFD[2246] = -3.56127804E-01;
    COFD[2247] = 1.29948788E-02;
    COFD[2248] = -2.09490548E+01;
    COFD[2249] = 5.31360223E+00;
    COFD[2250] = -4.64787000E-01;
    COFD[2251] = 1.97483720E-02;
    COFD[2252] = -2.08822487E+01;
    COFD[2253] = 5.28557747E+00;
    COFD[2254] = -4.61402384E-01;
    COFD[2255] = 1.96111546E-02;
    COFD[2256] = -2.13985484E+01;
    COFD[2257] = 4.94878244E+00;
    COFD[2258] = -3.68158605E-01;
    COFD[2259] = 1.36008797E-02;
    COFD[2260] = -2.24120415E+01;
    COFD[2261] = 5.58744076E+00;
    COFD[2262] = -4.84489462E-01;
    COFD[2263] = 1.99733042E-02;
    COFD[2264] = -2.19253091E+01;
    COFD[2265] = 5.14570932E+00;
    COFD[2266] = -3.99877142E-01;
    COFD[2267] = 1.52199557E-02;
    COFD[2268] = -2.19503032E+01;
    COFD[2269] = 5.14570932E+00;
    COFD[2270] = -3.99877142E-01;
    COFD[2271] = 1.52199557E-02;
    COFD[2272] = -2.22801170E+01;
    COFD[2273] = 5.58507108E+00;
    COFD[2274] = -4.81395065E-01;
    COFD[2275] = 1.97276199E-02;
    COFD[2276] = -2.25041734E+01;
    COFD[2277] = 5.51797622E+00;
    COFD[2278] = -4.66229499E-01;
    COFD[2279] = 1.88128348E-02;
    COFD[2280] = -2.21913393E+01;
    COFD[2281] = 5.60175327E+00;
    COFD[2282] = -4.87953216E-01;
    COFD[2283] = 2.01882171E-02;
    COFD[2284] = -2.25219004E+01;
    COFD[2285] = 5.49554403E+00;
    COFD[2286] = -4.60936491E-01;
    COFD[2287] = 1.84887572E-02;
    COFD[2288] = -2.25758616E+01;
    COFD[2289] = 5.40563818E+00;
    COFD[2290] = -4.44444322E-01;
    COFD[2291] = 1.75813146E-02;
    COFD[2292] = -2.25760230E+01;
    COFD[2293] = 5.41049872E+00;
    COFD[2294] = -4.45356411E-01;
    COFD[2295] = 1.76320470E-02;
    COFD[2296] = -2.24179759E+01;
    COFD[2297] = 5.05061421E+00;
    COFD[2298] = -3.84359196E-01;
    COFD[2299] = 1.44214004E-02;
    COFD[2300] = -1.97226856E+01;
    COFD[2301] = 4.83750266E+00;
    COFD[2302] = -4.09581452E-01;
    COFD[2303] = 1.76174739E-02;
    COFD[2304] = -1.86254955E+01;
    COFD[2305] = 4.60336076E+00;
    COFD[2306] = -3.81691643E-01;
    COFD[2307] = 1.65085234E-02;
    COFD[2308] = -1.51163041E+01;
    COFD[2309] = 3.64206330E+00;
    COFD[2310] = -2.61313444E-01;
    COFD[2311] = 1.14642754E-02;
    COFD[2312] = -1.92361841E+01;
    COFD[2313] = 5.31542554E+00;
    COFD[2314] = -4.65003780E-01;
    COFD[2315] = 1.97570185E-02;
    COFD[2316] = -1.86491023E+01;
    COFD[2317] = 4.60336076E+00;
    COFD[2318] = -3.81691643E-01;
    COFD[2319] = 1.65085234E-02;
    COFD[2320] = -2.02287739E+01;
    COFD[2321] = 4.58441724E+00;
    COFD[2322] = -3.10392854E-01;
    COFD[2323] = 1.06849990E-02;
    COFD[2324] = -2.00997774E+01;
    COFD[2325] = 4.96870443E+00;
    COFD[2326] = -4.25292447E-01;
    COFD[2327] = 1.82459096E-02;
    COFD[2328] = -2.01095969E+01;
    COFD[2329] = 4.96870443E+00;
    COFD[2330] = -4.25292447E-01;
    COFD[2331] = 1.82459096E-02;
    COFD[2332] = -2.01190139E+01;
    COFD[2333] = 4.96870443E+00;
    COFD[2334] = -4.25292447E-01;
    COFD[2335] = 1.82459096E-02;
    COFD[2336] = -2.23812726E+01;
    COFD[2337] = 5.59425354E+00;
    COFD[2338] = -4.86232980E-01;
    COFD[2339] = 2.00835981E-02;
    COFD[2340] = -1.97452574E+01;
    COFD[2341] = 4.84249900E+00;
    COFD[2342] = -4.10120448E-01;
    COFD[2343] = 1.76363500E-02;
    COFD[2344] = -2.12362684E+01;
    COFD[2345] = 4.88535789E+00;
    COFD[2346] = -3.58153894E-01;
    COFD[2347] = 1.30969624E-02;
    COFD[2348] = -2.09127554E+01;
    COFD[2349] = 5.30526648E+00;
    COFD[2350] = -4.63785596E-01;
    COFD[2351] = 1.97079873E-02;
    COFD[2352] = -2.08447974E+01;
    COFD[2353] = 5.27674330E+00;
    COFD[2354] = -4.60336155E-01;
    COFD[2355] = 1.95680191E-02;
    COFD[2356] = -2.14144448E+01;
    COFD[2357] = 4.96219227E+00;
    COFD[2358] = -3.70270843E-01;
    COFD[2359] = 1.37072211E-02;
    COFD[2360] = -2.24025650E+01;
    COFD[2361] = 5.58952429E+00;
    COFD[2362] = -4.85012530E-01;
    COFD[2363] = 2.00062142E-02;
    COFD[2364] = -2.19322003E+01;
    COFD[2365] = 5.15446948E+00;
    COFD[2366] = -4.01332769E-01;
    COFD[2367] = 1.52956262E-02;
    COFD[2368] = -2.19576037E+01;
    COFD[2369] = 5.15446948E+00;
    COFD[2370] = -4.01332769E-01;
    COFD[2371] = 1.52956262E-02;
    COFD[2372] = -2.22638165E+01;
    COFD[2373] = 5.58490856E+00;
    COFD[2374] = -4.81588720E-01;
    COFD[2375] = 1.97445317E-02;
    COFD[2376] = -2.25004333E+01;
    COFD[2377] = 5.52198915E+00;
    COFD[2378] = -4.67014474E-01;
    COFD[2379] = 1.88574253E-02;
    COFD[2380] = -2.21822461E+01;
    COFD[2381] = 5.60465338E+00;
    COFD[2382] = -4.88572478E-01;
    COFD[2383] = 2.02248525E-02;
    COFD[2384] = -2.25140525E+01;
    COFD[2385] = 5.49776513E+00;
    COFD[2386] = -4.61463030E-01;
    COFD[2387] = 1.85209236E-02;
    COFD[2388] = -2.25760230E+01;
    COFD[2389] = 5.41049872E+00;
    COFD[2390] = -4.45356411E-01;
    COFD[2391] = 1.76320470E-02;
    COFD[2392] = -2.25762645E+01;
    COFD[2393] = 5.41536807E+00;
    COFD[2394] = -4.46269562E-01;
    COFD[2395] = 1.76828228E-02;
    COFD[2396] = -2.24360850E+01;
    COFD[2397] = 5.06106414E+00;
    COFD[2398] = -3.86053039E-01;
    COFD[2399] = 1.45081784E-02;
    COFD[2400] = -2.10685573E+01;
    COFD[2401] = 5.15027524E+00;
    COFD[2402] = -4.46126111E-01;
    COFD[2403] = 1.90401391E-02;
    COFD[2404] = -1.99792167E+01;
    COFD[2405] = 4.92184026E+00;
    COFD[2406] = -4.19745472E-01;
    COFD[2407] = 1.80268154E-02;
    COFD[2408] = -1.64899530E+01;
    COFD[2409] = 4.01175649E+00;
    COFD[2410] = -3.08860971E-01;
    COFD[2411] = 1.35100076E-02;
    COFD[2412] = -2.03114210E+01;
    COFD[2413] = 5.50136606E+00;
    COFD[2414] = -4.82461887E-01;
    COFD[2415] = 2.02471523E-02;
    COFD[2416] = -2.00054461E+01;
    COFD[2417] = 4.92184026E+00;
    COFD[2418] = -4.19745472E-01;
    COFD[2419] = 1.80268154E-02;
    COFD[2420] = -1.91334529E+01;
    COFD[2421] = 3.82263611E+00;
    COFD[2422] = -1.93983472E-01;
    COFD[2423] = 4.95789388E-03;
    COFD[2424] = -2.13968281E+01;
    COFD[2425] = 5.25183817E+00;
    COFD[2426] = -4.57376333E-01;
    COFD[2427] = 1.94504429E-02;
    COFD[2428] = -2.14085375E+01;
    COFD[2429] = 5.25183817E+00;
    COFD[2430] = -4.57376333E-01;
    COFD[2431] = 1.94504429E-02;
    COFD[2432] = -2.14198091E+01;
    COFD[2433] = 5.25183817E+00;
    COFD[2434] = -4.57376333E-01;
    COFD[2435] = 1.94504429E-02;
    COFD[2436] = -2.28671232E+01;
    COFD[2437] = 5.50522401E+00;
    COFD[2438] = -4.63604304E-01;
    COFD[2439] = 1.86600785E-02;
    COFD[2440] = -2.10855099E+01;
    COFD[2441] = 5.15315713E+00;
    COFD[2442] = -4.46344043E-01;
    COFD[2443] = 1.90431546E-02;
    COFD[2444] = -2.04632210E+01;
    COFD[2445] = 4.26473557E+00;
    COFD[2446] = -2.61033037E-01;
    COFD[2447] = 8.23906412E-03;
    COFD[2448] = -2.19557531E+01;
    COFD[2449] = 5.49350509E+00;
    COFD[2450] = -4.81613405E-01;
    COFD[2451] = 2.02171734E-02;
    COFD[2452] = -2.19060847E+01;
    COFD[2453] = 5.47162499E+00;
    COFD[2454] = -4.79195552E-01;
    COFD[2455] = 2.01289088E-02;
    COFD[2456] = -2.06870442E+01;
    COFD[2457] = 4.35920123E+00;
    COFD[2458] = -2.75491273E-01;
    COFD[2459] = 8.95100289E-03;
    COFD[2460] = -2.28458380E+01;
    COFD[2461] = 5.50134401E+00;
    COFD[2462] = -4.62488197E-01;
    COFD[2463] = 1.85873697E-02;
    COFD[2464] = -2.13539532E+01;
    COFD[2465] = 4.61201872E+00;
    COFD[2466] = -3.14803338E-01;
    COFD[2467] = 1.09082984E-02;
    COFD[2468] = -2.13854464E+01;
    COFD[2469] = 4.61201872E+00;
    COFD[2470] = -3.14803338E-01;
    COFD[2471] = 1.09082984E-02;
    COFD[2472] = -2.26099899E+01;
    COFD[2473] = 5.44867280E+00;
    COFD[2474] = -4.52284883E-01;
    COFD[2475] = 1.80110706E-02;
    COFD[2476] = -2.26044889E+01;
    COFD[2477] = 5.27383847E+00;
    COFD[2478] = -4.21722368E-01;
    COFD[2479] = 1.63729618E-02;
    COFD[2480] = -2.26591038E+01;
    COFD[2481] = 5.52001624E+00;
    COFD[2482] = -4.66629503E-01;
    COFD[2483] = 1.88355817E-02;
    COFD[2484] = -2.25175307E+01;
    COFD[2485] = 5.21003123E+00;
    COFD[2486] = -4.10612564E-01;
    COFD[2487] = 1.57798598E-02;
    COFD[2488] = -2.24179759E+01;
    COFD[2489] = 5.05061421E+00;
    COFD[2490] = -3.84359196E-01;
    COFD[2491] = 1.44214004E-02;
    COFD[2492] = -2.24360850E+01;
    COFD[2493] = 5.06106414E+00;
    COFD[2494] = -3.86053039E-01;
    COFD[2495] = 1.45081784E-02;
    COFD[2496] = -2.17797084E+01;
    COFD[2497] = 4.48837319E+00;
    COFD[2498] = -2.95423315E-01;
    COFD[2499] = 9.93861345E-03;
}


/*List of specs with small weight, dim NLITE */
void egtransetKTDIF(int* KTDIF) {
    KTDIF[0] = 3;
    KTDIF[1] = 4;
}


/*Poly fits for thermal diff ratios, dim NO*NLITE*KK */
void egtransetCOFTD(double* COFTD) {
    COFTD[0] = 4.31331269E-01;
    COFTD[1] = 9.20536800E-05;
    COFTD[2] = -5.94509616E-08;
    COFTD[3] = 1.21437993E-11;
    COFTD[4] = 4.06682492E-01;
    COFTD[5] = 3.84705248E-05;
    COFTD[6] = -2.54846868E-08;
    COFTD[7] = 5.86302354E-12;
    COFTD[8] = 0.00000000E+00;
    COFTD[9] = 0.00000000E+00;
    COFTD[10] = 0.00000000E+00;
    COFTD[11] = 0.00000000E+00;
    COFTD[12] = -1.44152190E-01;
    COFTD[13] = -7.99993584E-05;
    COFTD[14] = 4.89707442E-08;
    COFTD[15] = -9.14277269E-12;
    COFTD[16] = 4.12895615E-01;
    COFTD[17] = 3.90582612E-05;
    COFTD[18] = -2.58740310E-08;
    COFTD[19] = 5.95259633E-12;
    COFTD[20] = 2.27469146E-02;
    COFTD[21] = 6.73078907E-04;
    COFTD[22] = -3.40935843E-07;
    COFTD[23] = 5.48499211E-11;
    COFTD[24] = 4.26579943E-01;
    COFTD[25] = 1.20407274E-04;
    COFTD[26] = -7.67298757E-08;
    COFTD[27] = 1.52090336E-11;
    COFTD[28] = 4.28230888E-01;
    COFTD[29] = 1.20873273E-04;
    COFTD[30] = -7.70268349E-08;
    COFTD[31] = 1.52678954E-11;
    COFTD[32] = 4.29789463E-01;
    COFTD[33] = 1.21313199E-04;
    COFTD[34] = -7.73071792E-08;
    COFTD[35] = 1.53234639E-11;
    COFTD[36] = 2.93191523E-01;
    COFTD[37] = 4.01430006E-04;
    COFTD[38] = -2.30705763E-07;
    COFTD[39] = 4.05176586E-11;
    COFTD[40] = 4.30605547E-01;
    COFTD[41] = 9.35961902E-05;
    COFTD[42] = -6.03983623E-08;
    COFTD[43] = 1.23115170E-11;
    COFTD[44] = 1.22693382E-01;
    COFTD[45] = 6.21278143E-04;
    COFTD[46] = -3.29965208E-07;
    COFTD[47] = 5.47161548E-11;
    COFTD[48] = 3.31191185E-01;
    COFTD[49] = 1.81326714E-04;
    COFTD[50] = -1.11096391E-07;
    COFTD[51] = 2.07635959E-11;
    COFTD[52] = 3.39557243E-01;
    COFTD[53] = 1.79335036E-04;
    COFTD[54] = -1.10135705E-07;
    COFTD[55] = 2.06427239E-11;
    COFTD[56] = 1.31424053E-01;
    COFTD[57] = 6.16429134E-04;
    COFTD[58] = -3.28571348E-07;
    COFTD[59] = 5.46153434E-11;
    COFTD[60] = 2.78021896E-01;
    COFTD[61] = 3.89608886E-04;
    COFTD[62] = -2.23546590E-07;
    COFTD[63] = 3.92078724E-11;
    COFTD[64] = 1.59288984E-01;
    COFTD[65] = 6.02833801E-04;
    COFTD[66] = -3.24837576E-07;
    COFTD[67] = 5.43909010E-11;
    COFTD[68] = 1.60621157E-01;
    COFTD[69] = 6.07875449E-04;
    COFTD[70] = -3.27554273E-07;
    COFTD[71] = 5.48457855E-11;
    COFTD[72] = 2.58066832E-01;
    COFTD[73] = 4.05072593E-04;
    COFTD[74] = -2.30587443E-07;
    COFTD[75] = 4.01863841E-11;
    COFTD[76] = 2.40639006E-01;
    COFTD[77] = 4.82930111E-04;
    COFTD[78] = -2.70362190E-07;
    COFTD[79] = 4.65173265E-11;
    COFTD[80] = 2.82974392E-01;
    COFTD[81] = 3.73032949E-04;
    COFTD[82] = -2.14959161E-07;
    COFTD[83] = 3.78355155E-11;
    COFTD[84] = 2.27261590E-01;
    COFTD[85] = 4.99550076E-04;
    COFTD[86] = -2.78004320E-07;
    COFTD[87] = 4.76209155E-11;
    COFTD[88] = 2.10934836E-01;
    COFTD[89] = 5.46607649E-04;
    COFTD[90] = -3.01041232E-07;
    COFTD[91] = 5.11789725E-11;
    COFTD[92] = 2.12842514E-01;
    COFTD[93] = 5.46075564E-04;
    COFTD[94] = -3.00933730E-07;
    COFTD[95] = 5.11832891E-11;
    COFTD[96] = 1.55121130E-01;
    COFTD[97] = 6.55610653E-04;
    COFTD[98] = -3.51255177E-07;
    COFTD[99] = 5.85866700E-11;
    COFTD[100] = 2.01521643E-01;
    COFTD[101] = 5.62744089E-04;
    COFTD[102] = -3.08519239E-07;
    COFTD[103] = 5.22805986E-11;
    COFTD[104] = 2.35283119E-01;
    COFTD[105] = 4.65670599E-04;
    COFTD[106] = -2.60939824E-07;
    COFTD[107] = 4.49271822E-11;
    COFTD[108] = 1.44152190E-01;
    COFTD[109] = 7.99993584E-05;
    COFTD[110] = -4.89707442E-08;
    COFTD[111] = 9.14277269E-12;
    COFTD[112] = 0.00000000E+00;
    COFTD[113] = 0.00000000E+00;
    COFTD[114] = 0.00000000E+00;
    COFTD[115] = 0.00000000E+00;
    COFTD[116] = 2.37053352E-01;
    COFTD[117] = 4.69174231E-04;
    COFTD[118] = -2.62903094E-07;
    COFTD[119] = 4.52652072E-11;
    COFTD[120] = -1.74352698E-01;
    COFTD[121] = 8.62246873E-04;
    COFTD[122] = -3.79545489E-07;
    COFTD[123] = 5.60262093E-11;
    COFTD[124] = 1.79840299E-01;
    COFTD[125] = 6.01722902E-04;
    COFTD[126] = -3.26433894E-07;
    COFTD[127] = 5.49112302E-11;
    COFTD[128] = 1.80186965E-01;
    COFTD[129] = 6.02882805E-04;
    COFTD[130] = -3.27063140E-07;
    COFTD[131] = 5.50170790E-11;
    COFTD[132] = 1.80513677E-01;
    COFTD[133] = 6.03975942E-04;
    COFTD[134] = -3.27656165E-07;
    COFTD[135] = 5.51168351E-11;
    COFTD[136] = -2.00309448E-02;
    COFTD[137] = 8.50440115E-04;
    COFTD[138] = -4.21064468E-07;
    COFTD[139] = 6.67959710E-11;
    COFTD[140] = 2.00119897E-01;
    COFTD[141] = 5.64793704E-04;
    COFTD[142] = -3.09445484E-07;
    COFTD[143] = 5.24139335E-11;
    COFTD[144] = -1.61357564E-01;
    COFTD[145] = 9.05920260E-04;
    COFTD[146] = -4.07879153E-07;
    COFTD[147] = 6.10626290E-11;
    COFTD[148] = 1.00039110E-01;
    COFTD[149] = 6.50468660E-04;
    COFTD[150] = -3.41778999E-07;
    COFTD[151] = 5.62779132E-11;
    COFTD[152] = 1.05124122E-01;
    COFTD[153] = 6.50665957E-04;
    COFTD[154] = -3.42564538E-07;
    COFTD[155] = 5.64804120E-11;
    COFTD[156] = -1.56651581E-01;
    COFTD[157] = 9.09789751E-04;
    COFTD[158] = -4.11714242E-07;
    COFTD[159] = 6.18310893E-11;
    COFTD[160] = -2.28637575E-02;
    COFTD[161] = 8.35412914E-04;
    COFTD[162] = -4.12929260E-07;
    COFTD[163] = 6.54380945E-11;
    COFTD[164] = -1.41640506E-01;
    COFTD[165] = 9.21404324E-04;
    COFTD[166] = -4.23210110E-07;
    COFTD[167] = 6.41400322E-11;
    COFTD[168] = -1.42230624E-01;
    COFTD[169] = 9.25243177E-04;
    COFTD[170] = -4.24973333E-07;
    COFTD[171] = 6.44072593E-11;
    COFTD[172] = -3.81470765E-02;
    COFTD[173] = 8.39833490E-04;
    COFTD[174] = -4.11688915E-07;
    COFTD[175] = 6.49124952E-11;
    COFTD[176] = -7.23038994E-02;
    COFTD[177] = 8.89466098E-04;
    COFTD[178] = -4.28124818E-07;
    COFTD[179] = 6.67586244E-11;
    COFTD[180] = -1.42100396E-02;
    COFTD[181] = 8.23812102E-04;
    COFTD[182] = -4.08995515E-07;
    COFTD[183] = 6.49899310E-11;
    COFTD[184] = -8.35962674E-02;
    COFTD[185] = 8.94627716E-04;
    COFTD[186] = -4.27681210E-07;
    COFTD[187] = 6.64140378E-11;
    COFTD[188] = -1.04463705E-01;
    COFTD[189] = 9.17317898E-04;
    COFTD[190] = -4.33159478E-07;
    COFTD[191] = 6.67640055E-11;
    COFTD[192] = -1.03451906E-01;
    COFTD[193] = 9.17972300E-04;
    COFTD[194] = -4.33791320E-07;
    COFTD[195] = 6.68916093E-11;
    COFTD[196] = -1.55228210E-01;
    COFTD[197] = 9.48987307E-04;
    COFTD[198] = -4.32517553E-07;
    COFTD[199] = 6.52402330E-11;
}

/* Replace this routine with the one generated by the Gauss Jordan solver of DW */
AMREX_GPU_HOST_DEVICE void sgjsolve(double* A, double* x, double* b) {
    amrex::Abort("sgjsolve not implemented, choose a different solver ");
}

/* Replace this routine with the one generated by the Gauss Jordan solver of DW */
AMREX_GPU_HOST_DEVICE void sgjsolve_simplified(double* A, double* x, double* b) {
    amrex::Abort("sgjsolve_simplified not implemented, choose a different solver ");
}

