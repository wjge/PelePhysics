#include "chemistry_file.H"

#ifndef AMREX_USE_CUDA
namespace thermo
{
    double fwd_A[172], fwd_beta[172], fwd_Ea[172];
    double low_A[172], low_beta[172], low_Ea[172];
    double rev_A[172], rev_beta[172], rev_Ea[172];
    double troe_a[172],troe_Ts[172], troe_Tss[172], troe_Tsss[172];
    double sri_a[172], sri_b[172], sri_c[172], sri_d[172], sri_e[172];
    double activation_units[172], prefactor_units[172], phase_units[172];
    int is_PD[172], troe_len[172], sri_len[172], nTB[172], *TBid[172];
    double *TB[172];
    std::vector<std::vector<double>> kiv(172); 
    std::vector<std::vector<double>> nuv(172); 
    std::vector<std::vector<double>> kiv_qss(172); 
    std::vector<std::vector<double>> nuv_qss(172); 
};

using namespace thermo;
#endif

/* Inverse molecular weights */
static AMREX_GPU_DEVICE_MANAGED double imw[24] = {
    1.0 / 28.013400,  /*N2 */
    1.0 / 1.007970,  /*H */
    1.0 / 2.015940,  /*H2 */
    1.0 / 15.999400,  /*O */
    1.0 / 17.007370,  /*OH */
    1.0 / 31.998800,  /*O2 */
    1.0 / 34.014740,  /*H2O2 */
    1.0 / 18.015340,  /*H2O */
    1.0 / 33.006770,  /*HO2 */
    1.0 / 28.010550,  /*CO */
    1.0 / 15.035060,  /*CH3 */
    1.0 / 30.026490,  /*CH2O */
    1.0 / 44.009950,  /*CO2 */
    1.0 / 16.043030,  /*CH4 */
    1.0 / 26.038240,  /*C2H2 */
    1.0 / 28.054180,  /*C2H4 */
    1.0 / 42.037640,  /*CH2CO */
    1.0 / 30.070120,  /*C2H6 */
    1.0 / 27.046210,  /*C2H3 */
    1.0 / 29.062150,  /*C2H5 */
    1.0 / 41.029670,  /*HCCO */
    1.0 / 44.053580,  /*CH3CHO */
    1.0 / 43.045610,  /*CH2CHO */
    1.0 / 45.061550};  /*C2H5O */

/* Molecular weights */
static AMREX_GPU_DEVICE_MANAGED double molecular_weights[24] = {
    28.013400,  /*N2 */
    1.007970,  /*H */
    2.015940,  /*H2 */
    15.999400,  /*O */
    17.007370,  /*OH */
    31.998800,  /*O2 */
    34.014740,  /*H2O2 */
    18.015340,  /*H2O */
    33.006770,  /*HO2 */
    28.010550,  /*CO */
    15.035060,  /*CH3 */
    30.026490,  /*CH2O */
    44.009950,  /*CO2 */
    16.043030,  /*CH4 */
    26.038240,  /*C2H2 */
    28.054180,  /*C2H4 */
    42.037640,  /*CH2CO */
    30.070120,  /*C2H6 */
    27.046210,  /*C2H3 */
    29.062150,  /*C2H5 */
    41.029670,  /*HCCO */
    44.053580,  /*CH3CHO */
    43.045610,  /*CH2CHO */
    45.061550};  /*C2H5O */

AMREX_GPU_HOST_DEVICE
void get_imw(double imw_new[]){
    for(int i = 0; i<24; ++i) imw_new[i] = imw[i];
}

/* TODO: check necessity because redundant with CKWT */
AMREX_GPU_HOST_DEVICE
void get_mw(double mw_new[]){
    for(int i = 0; i<24; ++i) mw_new[i] = molecular_weights[i];
}


#ifndef AMREX_USE_CUDA
/* Initializes parameter database */
void CKINIT()
{

    // (0):  2.000000 H + M <=> H2 + M
    kiv[15] = {1,2};
    nuv[15] = {-2.0,1};
    kiv_qss[15] = {};
    nuv_qss[15] = {};
    // (0):  2.000000 H + M <=> H2 + M
    fwd_A[15]     = 1.78e+18;
    fwd_beta[15]  = -1;
    fwd_Ea[15]    = 0;
    prefactor_units[15]  = 1.0000000000000002e-12;
    activation_units[15] = 0.50321666580471969;
    phase_units[15]      = pow(10,-12.000000);
    is_PD[15] = 0;
    nTB[15] = 5;
    TB[15] = (double *) malloc(5 * sizeof(double));
    TBid[15] = (int *) malloc(5 * sizeof(int));
    TBid[15][0] = 2; TB[15][0] = 0; // H2
    TBid[15][1] = 7; TB[15][1] = 0; // H2O
    TBid[15][2] = 12; TB[15][2] = 0; // CO2
    TBid[15][3] = 13; TB[15][3] = 2; // CH4
    TBid[15][4] = 17; TB[15][4] = 3; // C2H6

    // (1):  2.000000 H + H2 <=> 2.000000 H2
    kiv[20] = {1,2,2};
    nuv[20] = {-2.0,-1,2.0};
    kiv_qss[20] = {};
    nuv_qss[20] = {};
    // (1):  2.000000 H + H2 <=> 2.000000 H2
    fwd_A[20]     = 90000000000000000;
    fwd_beta[20]  = -0.59999999999999998;
    fwd_Ea[20]    = 0;
    prefactor_units[20]  = 1.0000000000000002e-12;
    activation_units[20] = 0.50321666580471969;
    phase_units[20]      = pow(10,-18.000000);
    is_PD[20] = 0;
    nTB[20] = 0;

    // (2):  O + H2 <=> H + OH
    kiv[21] = {3,2,1,4};
    nuv[21] = {-1,-1,1,1};
    kiv_qss[21] = {};
    nuv_qss[21] = {};
    // (2):  O + H2 <=> H + OH
    fwd_A[21]     = 45900;
    fwd_beta[21]  = 2.7000000000000002;
    fwd_Ea[21]    = 6259.5600000000004;
    prefactor_units[21]  = 1.0000000000000002e-06;
    activation_units[21] = 0.50321666580471969;
    phase_units[21]      = pow(10,-12.000000);
    is_PD[21] = 0;
    nTB[21] = 0;

    // (3):  H + O + M <=> OH + M
    kiv[16] = {1,3,4};
    nuv[16] = {-1,-1,1};
    kiv_qss[16] = {};
    nuv_qss[16] = {};
    // (3):  H + O + M <=> OH + M
    fwd_A[16]     = 9.43e+18;
    fwd_beta[16]  = -1;
    fwd_Ea[16]    = 0;
    prefactor_units[16]  = 1.0000000000000002e-12;
    activation_units[16] = 0.50321666580471969;
    phase_units[16]      = pow(10,-12.000000);
    is_PD[16] = 0;
    nTB[16] = 6;
    TB[16] = (double *) malloc(6 * sizeof(double));
    TBid[16] = (int *) malloc(6 * sizeof(int));
    TBid[16][0] = 2; TB[16][0] = 2; // H2
    TBid[16][1] = 7; TB[16][1] = 12; // H2O
    TBid[16][2] = 9; TB[16][2] = 1.75; // CO
    TBid[16][3] = 12; TB[16][3] = 3.6000000000000001; // CO2
    TBid[16][4] = 13; TB[16][4] = 2; // CH4
    TBid[16][5] = 17; TB[16][5] = 3; // C2H6

    // (4):  2.000000 O + M <=> O2 + M
    kiv[17] = {3,5};
    nuv[17] = {-2.0,1};
    kiv_qss[17] = {};
    nuv_qss[17] = {};
    // (4):  2.000000 O + M <=> O2 + M
    fwd_A[17]     = 1.2e+17;
    fwd_beta[17]  = -1;
    fwd_Ea[17]    = 0;
    prefactor_units[17]  = 1.0000000000000002e-12;
    activation_units[17] = 0.50321666580471969;
    phase_units[17]      = pow(10,-12.000000);
    is_PD[17] = 0;
    nTB[17] = 6;
    TB[17] = (double *) malloc(6 * sizeof(double));
    TBid[17] = (int *) malloc(6 * sizeof(int));
    TBid[17][0] = 2; TB[17][0] = 2.3999999999999999; // H2
    TBid[17][1] = 7; TB[17][1] = 15.4; // H2O
    TBid[17][2] = 9; TB[17][2] = 1.75; // CO
    TBid[17][3] = 12; TB[17][3] = 3.6000000000000001; // CO2
    TBid[17][4] = 13; TB[17][4] = 2; // CH4
    TBid[17][5] = 17; TB[17][5] = 3; // C2H6

    // (5):  2.000000 OH (+M) <=> H2O2 (+M)
    kiv[0] = {4,6};
    nuv[0] = {-2.0,1};
    kiv_qss[0] = {};
    nuv_qss[0] = {};
    // (5):  2.000000 OH (+M) <=> H2O2 (+M)
    fwd_A[0]     = 111000000000000;
    fwd_beta[0]  = -0.37;
    fwd_Ea[0]    = 0;
    low_A[0]     = 2.01e+17;
    low_beta[0]  = -0.57999999999999996;
    low_Ea[0]    = -2292.0700000000002;
    troe_a[0]    = 0.73460000000000003;
    troe_Tsss[0] = 94;
    troe_Ts[0]   = 1756;
    troe_Tss[0]  = 5182;
    troe_len[0]  = 4;
    prefactor_units[0]  = 1.0000000000000002e-06;
    activation_units[0] = 0.50321666580471969;
    phase_units[0]      = pow(10,-12.000000);
    is_PD[0] = 1;
    nTB[0] = 6;
    TB[0] = (double *) malloc(6 * sizeof(double));
    TBid[0] = (int *) malloc(6 * sizeof(int));
    TBid[0][0] = 2; TB[0][0] = 2; // H2
    TBid[0][1] = 7; TB[0][1] = 12; // H2O
    TBid[0][2] = 9; TB[0][2] = 1.75; // CO
    TBid[0][3] = 12; TB[0][3] = 3.6000000000000001; // CO2
    TBid[0][4] = 13; TB[0][4] = 2; // CH4
    TBid[0][5] = 17; TB[0][5] = 3; // C2H6

    // (6):  H + OH + M <=> H2O + M
    kiv[18] = {1,4,7};
    nuv[18] = {-1,-1,1};
    kiv_qss[18] = {};
    nuv_qss[18] = {};
    // (6):  H + OH + M <=> H2O + M
    fwd_A[18]     = 4.4e+22;
    fwd_beta[18]  = -2;
    fwd_Ea[18]    = 0;
    prefactor_units[18]  = 1.0000000000000002e-12;
    activation_units[18] = 0.50321666580471969;
    phase_units[18]      = pow(10,-12.000000);
    is_PD[18] = 0;
    nTB[18] = 6;
    TB[18] = (double *) malloc(6 * sizeof(double));
    TBid[18] = (int *) malloc(6 * sizeof(int));
    TBid[18][0] = 2; TB[18][0] = 2; // H2
    TBid[18][1] = 7; TB[18][1] = 6.2999999999999998; // H2O
    TBid[18][2] = 9; TB[18][2] = 1.75; // CO
    TBid[18][3] = 12; TB[18][3] = 3.6000000000000001; // CO2
    TBid[18][4] = 13; TB[18][4] = 2; // CH4
    TBid[18][5] = 17; TB[18][5] = 3; // C2H6

    // (7):  2.000000 OH <=> O + H2O
    kiv[22] = {4,3,7};
    nuv[22] = {-2.0,1,1};
    kiv_qss[22] = {};
    nuv_qss[22] = {};
    // (7):  2.000000 OH <=> O + H2O
    fwd_A[22]     = 39700;
    fwd_beta[22]  = 2.3999999999999999;
    fwd_Ea[22]    = -2110.4200000000001;
    prefactor_units[22]  = 1.0000000000000002e-06;
    activation_units[22] = 0.50321666580471969;
    phase_units[22]      = pow(10,-12.000000);
    is_PD[22] = 0;
    nTB[22] = 0;

    // (8):  OH + H2 <=> H + H2O
    kiv[23] = {4,2,1,7};
    nuv[23] = {-1,-1,1,1};
    kiv_qss[23] = {};
    nuv_qss[23] = {};
    // (8):  OH + H2 <=> H + H2O
    fwd_A[23]     = 173000000;
    fwd_beta[23]  = 1.51;
    fwd_Ea[23]    = 3429.73;
    prefactor_units[23]  = 1.0000000000000002e-06;
    activation_units[23] = 0.50321666580471969;
    phase_units[23]      = pow(10,-12.000000);
    is_PD[23] = 0;
    nTB[23] = 0;

    // (9):  2.000000 H + H2O <=> H2 + H2O
    kiv[24] = {1,7,2,7};
    nuv[24] = {-2.0,-1,1,1};
    kiv_qss[24] = {};
    nuv_qss[24] = {};
    // (9):  2.000000 H + H2O <=> H2 + H2O
    fwd_A[24]     = 5.62e+19;
    fwd_beta[24]  = -1.25;
    fwd_Ea[24]    = 0;
    prefactor_units[24]  = 1.0000000000000002e-12;
    activation_units[24] = 0.50321666580471969;
    phase_units[24]      = pow(10,-18.000000);
    is_PD[24] = 0;
    nTB[24] = 0;

    // (10):  H + O2 (+M) <=> HO2 (+M)
    kiv[1] = {1,5,8};
    nuv[1] = {-1,-1,1};
    kiv_qss[1] = {};
    nuv_qss[1] = {};
    // (10):  H + O2 (+M) <=> HO2 (+M)
    fwd_A[1]     = 5120000000000;
    fwd_beta[1]  = 0.44;
    fwd_Ea[1]    = 0;
    low_A[1]     = 6.33e+19;
    low_beta[1]  = -1.3999999999999999;
    low_Ea[1]    = 0;
    troe_a[1]    = 0.5;
    troe_Tsss[1] = 0;
    troe_Ts[1]   = 10000000000;
    troe_len[1]  = 3;
    prefactor_units[1]  = 1.0000000000000002e-06;
    activation_units[1] = 0.50321666580471969;
    phase_units[1]      = pow(10,-12.000000);
    is_PD[1] = 1;
    nTB[1] = 5;
    TB[1] = (double *) malloc(5 * sizeof(double));
    TBid[1] = (int *) malloc(5 * sizeof(int));
    TBid[1][0] = 2; TB[1][0] = 0.75; // H2
    TBid[1][1] = 5; TB[1][1] = 0.84999999999999998; // O2
    TBid[1][2] = 7; TB[1][2] = 11.890000000000001; // H2O
    TBid[1][3] = 9; TB[1][3] = 1.0900000000000001; // CO
    TBid[1][4] = 12; TB[1][4] = 2.1800000000000002; // CO2

    // (11):  H + O2 <=> O + OH
    kiv[25] = {1,5,3,4};
    nuv[25] = {-1,-1,1,1};
    kiv_qss[25] = {};
    nuv_qss[25] = {};
    // (11):  H + O2 <=> O + OH
    fwd_A[25]     = 26400000000000000;
    fwd_beta[25]  = -0.67000000000000004;
    fwd_Ea[25]    = 17041.110000000001;
    prefactor_units[25]  = 1.0000000000000002e-06;
    activation_units[25] = 0.50321666580471969;
    phase_units[25]      = pow(10,-12.000000);
    is_PD[25] = 0;
    nTB[25] = 0;

    // (12):  H2 + O2 <=> HO2 + H
    kiv[26] = {2,5,8,1};
    nuv[26] = {-1,-1,1,1};
    kiv_qss[26] = {};
    nuv_qss[26] = {};
    // (12):  H2 + O2 <=> HO2 + H
    fwd_A[26]     = 592000;
    fwd_beta[26]  = 2.4300000000000002;
    fwd_Ea[26]    = 53501.43;
    prefactor_units[26]  = 1.0000000000000002e-06;
    activation_units[26] = 0.50321666580471969;
    phase_units[26]      = pow(10,-12.000000);
    is_PD[26] = 0;
    nTB[26] = 0;

    // (13):  HO2 + OH <=> H2O + O2
    kiv[27] = {8,4,7,5};
    nuv[27] = {-1,-1,1,1};
    kiv_qss[27] = {};
    nuv_qss[27] = {};
    // (13):  HO2 + OH <=> H2O + O2
    fwd_A[27]     = 23800000000000;
    fwd_beta[27]  = 0;
    fwd_Ea[27]    = -499.51999999999998;
    prefactor_units[27]  = 1.0000000000000002e-06;
    activation_units[27] = 0.50321666580471969;
    phase_units[27]      = pow(10,-12.000000);
    is_PD[27] = 0;
    nTB[27] = 0;

    // (14):  HO2 + H <=> 2.000000 OH
    kiv[28] = {8,1,4};
    nuv[28] = {-1,-1,2.0};
    kiv_qss[28] = {};
    nuv_qss[28] = {};
    // (14):  HO2 + H <=> 2.000000 OH
    fwd_A[28]     = 74900000000000;
    fwd_beta[28]  = 0;
    fwd_Ea[28]    = 635.75999999999999;
    prefactor_units[28]  = 1.0000000000000002e-06;
    activation_units[28] = 0.50321666580471969;
    phase_units[28]      = pow(10,-12.000000);
    is_PD[28] = 0;
    nTB[28] = 0;

    // (15):  HO2 + O <=> OH + O2
    kiv[29] = {8,3,4,5};
    nuv[29] = {-1,-1,1,1};
    kiv_qss[29] = {};
    nuv_qss[29] = {};
    // (15):  HO2 + O <=> OH + O2
    fwd_A[29]     = 40000000000000;
    fwd_beta[29]  = 0;
    fwd_Ea[29]    = 0;
    prefactor_units[29]  = 1.0000000000000002e-06;
    activation_units[29] = 0.50321666580471969;
    phase_units[29]      = pow(10,-12.000000);
    is_PD[29] = 0;
    nTB[29] = 0;

    // (16):  HO2 + H <=> O + H2O
    kiv[30] = {8,1,3,7};
    nuv[30] = {-1,-1,1,1};
    kiv_qss[30] = {};
    nuv_qss[30] = {};
    // (16):  HO2 + H <=> O + H2O
    fwd_A[30]     = 3970000000000;
    fwd_beta[30]  = 0;
    fwd_Ea[30]    = 671.61000000000001;
    prefactor_units[30]  = 1.0000000000000002e-06;
    activation_units[30] = 0.50321666580471969;
    phase_units[30]      = pow(10,-12.000000);
    is_PD[30] = 0;
    nTB[30] = 0;

    // (17):  HO2 + OH <=> H2O + O2
    kiv[31] = {8,4,7,5};
    nuv[31] = {-1,-1,1,1};
    kiv_qss[31] = {};
    nuv_qss[31] = {};
    // (17):  HO2 + OH <=> H2O + O2
    fwd_A[31]     = 10000000000000000;
    fwd_beta[31]  = 0;
    fwd_Ea[31]    = 17330.310000000001;
    prefactor_units[31]  = 1.0000000000000002e-06;
    activation_units[31] = 0.50321666580471969;
    phase_units[31]      = pow(10,-12.000000);
    is_PD[31] = 0;
    nTB[31] = 0;

    // (18):  H2O2 + O <=> HO2 + OH
    kiv[32] = {6,3,8,4};
    nuv[32] = {-1,-1,1,1};
    kiv_qss[32] = {};
    nuv_qss[32] = {};
    // (18):  H2O2 + O <=> HO2 + OH
    fwd_A[32]     = 9630000;
    fwd_beta[32]  = 2;
    fwd_Ea[32]    = 3969.8899999999999;
    prefactor_units[32]  = 1.0000000000000002e-06;
    activation_units[32] = 0.50321666580471969;
    phase_units[32]      = pow(10,-12.000000);
    is_PD[32] = 0;
    nTB[32] = 0;

    // (19):  H2O2 + H <=> H2O + OH
    kiv[33] = {6,1,7,4};
    nuv[33] = {-1,-1,1,1};
    kiv_qss[33] = {};
    nuv_qss[33] = {};
    // (19):  H2O2 + H <=> H2O + OH
    fwd_A[33]     = 24100000000000;
    fwd_beta[33]  = 0;
    fwd_Ea[33]    = 3969.8899999999999;
    prefactor_units[33]  = 1.0000000000000002e-06;
    activation_units[33] = 0.50321666580471969;
    phase_units[33]      = pow(10,-12.000000);
    is_PD[33] = 0;
    nTB[33] = 0;

    // (20):  H2O2 + H <=> HO2 + H2
    kiv[34] = {6,1,8,2};
    nuv[34] = {-1,-1,1,1};
    kiv_qss[34] = {};
    nuv_qss[34] = {};
    // (20):  H2O2 + H <=> HO2 + H2
    fwd_A[34]     = 6050000;
    fwd_beta[34]  = 2;
    fwd_Ea[34]    = 5200.7600000000002;
    prefactor_units[34]  = 1.0000000000000002e-06;
    activation_units[34] = 0.50321666580471969;
    phase_units[34]      = pow(10,-12.000000);
    is_PD[34] = 0;
    nTB[34] = 0;

    // (21):  H2O2 + OH <=> HO2 + H2O
    kiv[35] = {6,4,8,7};
    nuv[35] = {-1,-1,1,1};
    kiv_qss[35] = {};
    nuv_qss[35] = {};
    // (21):  H2O2 + OH <=> HO2 + H2O
    fwd_A[35]     = 2.6700000000000001e+41;
    fwd_beta[35]  = -7;
    fwd_Ea[35]    = 37600.379999999997;
    prefactor_units[35]  = 1.0000000000000002e-06;
    activation_units[35] = 0.50321666580471969;
    phase_units[35]      = pow(10,-12.000000);
    is_PD[35] = 0;
    nTB[35] = 0;

    // (22):  H2O2 + OH <=> HO2 + H2O
    kiv[36] = {6,4,8,7};
    nuv[36] = {-1,-1,1,1};
    kiv_qss[36] = {};
    nuv_qss[36] = {};
    // (22):  H2O2 + OH <=> HO2 + H2O
    fwd_A[36]     = 2000000000000;
    fwd_beta[36]  = 0;
    fwd_Ea[36]    = 427.81999999999999;
    prefactor_units[36]  = 1.0000000000000002e-06;
    activation_units[36] = 0.50321666580471969;
    phase_units[36]      = pow(10,-12.000000);
    is_PD[36] = 0;
    nTB[36] = 0;

    // (23):  C + O2 <=> CO + O
    kiv[37] = {5,9,3};
    nuv[37] = {-1,1,1};
    kiv_qss[37] = {0};
    nuv_qss[37] = {-1};
    // (23):  C + O2 <=> CO + O
    fwd_A[37]     = 58000000000000;
    fwd_beta[37]  = 0;
    fwd_Ea[37]    = 576;
    prefactor_units[37]  = 1.0000000000000002e-06;
    activation_units[37] = 0.50321666580471969;
    phase_units[37]      = pow(10,-12.000000);
    is_PD[37] = 0;
    nTB[37] = 0;

    // (24):  C + OH <=> CO + H
    kiv[38] = {4,9,1};
    nuv[38] = {-1,1,1};
    kiv_qss[38] = {0};
    nuv_qss[38] = {-1};
    // (24):  C + OH <=> CO + H
    fwd_A[38]     = 50000000000000;
    fwd_beta[38]  = 0;
    fwd_Ea[38]    = 0;
    prefactor_units[38]  = 1.0000000000000002e-06;
    activation_units[38] = 0.50321666580471969;
    phase_units[38]      = pow(10,-12.000000);
    is_PD[38] = 0;
    nTB[38] = 0;

    // (25):  CH + OH <=> HCO + H
    kiv[39] = {4,1};
    nuv[39] = {-1,1};
    kiv_qss[39] = {1,2};
    nuv_qss[39] = {-1,1};
    // (25):  CH + OH <=> HCO + H
    fwd_A[39]     = 30000000000000;
    fwd_beta[39]  = 0;
    fwd_Ea[39]    = 0;
    prefactor_units[39]  = 1.0000000000000002e-06;
    activation_units[39] = 0.50321666580471969;
    phase_units[39]      = pow(10,-12.000000);
    is_PD[39] = 0;
    nTB[39] = 0;

    // (26):  CH + H2 <=> TXCH2 + H
    kiv[40] = {2,1};
    nuv[40] = {-1,1};
    kiv_qss[40] = {1,3};
    nuv_qss[40] = {-1,1};
    // (26):  CH + H2 <=> TXCH2 + H
    fwd_A[40]     = 108000000000000;
    fwd_beta[40]  = 0;
    fwd_Ea[40]    = 3109.46;
    prefactor_units[40]  = 1.0000000000000002e-06;
    activation_units[40] = 0.50321666580471969;
    phase_units[40]      = pow(10,-12.000000);
    is_PD[40] = 0;
    nTB[40] = 0;

    // (27):  CH + O <=> CO + H
    kiv[41] = {3,9,1};
    nuv[41] = {-1,1,1};
    kiv_qss[41] = {1};
    nuv_qss[41] = {-1};
    // (27):  CH + O <=> CO + H
    fwd_A[41]     = 57000000000000;
    fwd_beta[41]  = 0;
    fwd_Ea[41]    = 0;
    prefactor_units[41]  = 1.0000000000000002e-06;
    activation_units[41] = 0.50321666580471969;
    phase_units[41]      = pow(10,-12.000000);
    is_PD[41] = 0;
    nTB[41] = 0;

    // (28):  CH + O2 <=> HCO + O
    kiv[42] = {5,3};
    nuv[42] = {-1,1};
    kiv_qss[42] = {1,2};
    nuv_qss[42] = {-1,1};
    // (28):  CH + O2 <=> HCO + O
    fwd_A[42]     = 67100000000000;
    fwd_beta[42]  = 0;
    fwd_Ea[42]    = 0;
    prefactor_units[42]  = 1.0000000000000002e-06;
    activation_units[42] = 0.50321666580471969;
    phase_units[42]      = pow(10,-12.000000);
    is_PD[42] = 0;
    nTB[42] = 0;

    // (29):  CH + H <=> C + H2
    kiv[43] = {1,2};
    nuv[43] = {-1,1};
    kiv_qss[43] = {1,0};
    nuv_qss[43] = {-1,1};
    // (29):  CH + H <=> C + H2
    fwd_A[43]     = 165000000000000;
    fwd_beta[43]  = 0;
    fwd_Ea[43]    = 0;
    prefactor_units[43]  = 1.0000000000000002e-06;
    activation_units[43] = 0.50321666580471969;
    phase_units[43]      = pow(10,-12.000000);
    is_PD[43] = 0;
    nTB[43] = 0;

    // (30):  CH + H2 (+M) <=> CH3 (+M)
    kiv[2] = {2,10};
    nuv[2] = {-1,1};
    kiv_qss[2] = {1};
    nuv_qss[2] = {-1};
    // (30):  CH + H2 (+M) <=> CH3 (+M)
    fwd_A[2]     = 1970000000000;
    fwd_beta[2]  = 0.42999999999999999;
    fwd_Ea[2]    = -370.45999999999998;
    low_A[2]     = 4.82e+25;
    low_beta[2]  = -2.7999999999999998;
    low_Ea[2]    = 590.34000000000003;
    troe_a[2]    = 0.57799999999999996;
    troe_Tsss[2] = 122;
    troe_Ts[2]   = 2535;
    troe_Tss[2]  = 9365;
    troe_len[2]  = 4;
    prefactor_units[2]  = 1.0000000000000002e-06;
    activation_units[2] = 0.50321666580471969;
    phase_units[2]      = pow(10,-12.000000);
    is_PD[2] = 1;
    nTB[2] = 6;
    TB[2] = (double *) malloc(6 * sizeof(double));
    TBid[2] = (int *) malloc(6 * sizeof(int));
    TBid[2][0] = 2; TB[2][0] = 2; // H2
    TBid[2][1] = 7; TB[2][1] = 12; // H2O
    TBid[2][2] = 9; TB[2][2] = 1.75; // CO
    TBid[2][3] = 12; TB[2][3] = 3.6000000000000001; // CO2
    TBid[2][4] = 13; TB[2][4] = 2; // CH4
    TBid[2][5] = 17; TB[2][5] = 3; // C2H6

    // (31):  CH + H2O <=> CH2O + H
    kiv[44] = {7,11,1};
    nuv[44] = {-1,1,1};
    kiv_qss[44] = {1};
    nuv_qss[44] = {-1};
    // (31):  CH + H2O <=> CH2O + H
    fwd_A[44]     = 5710000000000;
    fwd_beta[44]  = 0;
    fwd_Ea[44]    = -755.25999999999999;
    prefactor_units[44]  = 1.0000000000000002e-06;
    activation_units[44] = 0.50321666580471969;
    phase_units[44]      = pow(10,-12.000000);
    is_PD[44] = 0;
    nTB[44] = 0;

    // (32):  TXCH2 + H (+M) <=> CH3 (+M)
    kiv[3] = {1,10};
    nuv[3] = {-1,1};
    kiv_qss[3] = {3};
    nuv_qss[3] = {-1};
    // (32):  TXCH2 + H (+M) <=> CH3 (+M)
    fwd_A[3]     = 600000000000000;
    fwd_beta[3]  = 0;
    fwd_Ea[3]    = 0;
    low_A[3]     = 1.0399999999999999e+26;
    low_beta[3]  = -2.7599999999999998;
    low_Ea[3]    = 1598.95;
    troe_a[3]    = 0.56200000000000006;
    troe_Tsss[3] = 91;
    troe_Ts[3]   = 5836;
    troe_Tss[3]  = 8552;
    troe_len[3]  = 4;
    prefactor_units[3]  = 1.0000000000000002e-06;
    activation_units[3] = 0.50321666580471969;
    phase_units[3]      = pow(10,-12.000000);
    is_PD[3] = 1;
    nTB[3] = 6;
    TB[3] = (double *) malloc(6 * sizeof(double));
    TBid[3] = (int *) malloc(6 * sizeof(int));
    TBid[3][0] = 2; TB[3][0] = 2; // H2
    TBid[3][1] = 7; TB[3][1] = 12; // H2O
    TBid[3][2] = 9; TB[3][2] = 1.75; // CO
    TBid[3][3] = 12; TB[3][3] = 3.6000000000000001; // CO2
    TBid[3][4] = 13; TB[3][4] = 2; // CH4
    TBid[3][5] = 17; TB[3][5] = 3; // C2H6

    // (33):  TXCH2 + O2 => OH + H + CO
    kiv[45] = {5,4,1,9};
    nuv[45] = {-1,1,1,1};
    kiv_qss[45] = {3};
    nuv_qss[45] = {-1};
    // (33):  TXCH2 + O2 => OH + H + CO
    fwd_A[45]     = 5000000000000;
    fwd_beta[45]  = 0;
    fwd_Ea[45]    = 1500.96;
    prefactor_units[45]  = 1.0000000000000002e-06;
    activation_units[45] = 0.50321666580471969;
    phase_units[45]      = pow(10,-12.000000);
    is_PD[45] = 0;
    nTB[45] = 0;

    // (34):  TXCH2 + O2 <=> CH2O + O
    kiv[46] = {5,11,3};
    nuv[46] = {-1,1,1};
    kiv_qss[46] = {3};
    nuv_qss[46] = {-1};
    // (34):  TXCH2 + O2 <=> CH2O + O
    fwd_A[46]     = 2400000000000;
    fwd_beta[46]  = 0;
    fwd_Ea[46]    = 1500.96;
    prefactor_units[46]  = 1.0000000000000002e-06;
    activation_units[46] = 0.50321666580471969;
    phase_units[46]      = pow(10,-12.000000);
    is_PD[46] = 0;
    nTB[46] = 0;

    // (35):  TXCH2 + OH <=> CH2O + H
    kiv[47] = {4,11,1};
    nuv[47] = {-1,1,1};
    kiv_qss[47] = {3};
    nuv_qss[47] = {-1};
    // (35):  TXCH2 + OH <=> CH2O + H
    fwd_A[47]     = 20000000000000;
    fwd_beta[47]  = 0;
    fwd_Ea[47]    = 0;
    prefactor_units[47]  = 1.0000000000000002e-06;
    activation_units[47] = 0.50321666580471969;
    phase_units[47]      = pow(10,-12.000000);
    is_PD[47] = 0;
    nTB[47] = 0;

    // (36):  TXCH2 + HO2 <=> CH2O + OH
    kiv[48] = {8,11,4};
    nuv[48] = {-1,1,1};
    kiv_qss[48] = {3};
    nuv_qss[48] = {-1};
    // (36):  TXCH2 + HO2 <=> CH2O + OH
    fwd_A[48]     = 20000000000000;
    fwd_beta[48]  = 0;
    fwd_Ea[48]    = 0;
    prefactor_units[48]  = 1.0000000000000002e-06;
    activation_units[48] = 0.50321666580471969;
    phase_units[48]      = pow(10,-12.000000);
    is_PD[48] = 0;
    nTB[48] = 0;

    // (37):  TXCH2 + O2 => CO2 + 2.000000 H
    kiv[49] = {5,12,1};
    nuv[49] = {-1,1,2.0};
    kiv_qss[49] = {3};
    nuv_qss[49] = {-1};
    // (37):  TXCH2 + O2 => CO2 + 2.000000 H
    fwd_A[49]     = 5800000000000;
    fwd_beta[49]  = 0;
    fwd_Ea[49]    = 1500.96;
    prefactor_units[49]  = 1.0000000000000002e-06;
    activation_units[49] = 0.50321666580471969;
    phase_units[49]      = pow(10,-12.000000);
    is_PD[49] = 0;
    nTB[49] = 0;

    // (38):  TXCH2 + OH <=> CH + H2O
    kiv[50] = {4,7};
    nuv[50] = {-1,1};
    kiv_qss[50] = {3,1};
    nuv_qss[50] = {-1,1};
    // (38):  TXCH2 + OH <=> CH + H2O
    fwd_A[50]     = 11300000;
    fwd_beta[50]  = 2;
    fwd_Ea[50]    = 2999.52;
    prefactor_units[50]  = 1.0000000000000002e-06;
    activation_units[50] = 0.50321666580471969;
    phase_units[50]      = pow(10,-12.000000);
    is_PD[50] = 0;
    nTB[50] = 0;

    // (39):  TXCH2 + O <=> HCO + H
    kiv[51] = {3,1};
    nuv[51] = {-1,1};
    kiv_qss[51] = {3,2};
    nuv_qss[51] = {-1,1};
    // (39):  TXCH2 + O <=> HCO + H
    fwd_A[51]     = 80000000000000;
    fwd_beta[51]  = 0;
    fwd_Ea[51]    = 0;
    prefactor_units[51]  = 1.0000000000000002e-06;
    activation_units[51] = 0.50321666580471969;
    phase_units[51]      = pow(10,-12.000000);
    is_PD[51] = 0;
    nTB[51] = 0;

    // (40):  TXCH2 + H2 <=> H + CH3
    kiv[52] = {2,1,10};
    nuv[52] = {-1,1,1};
    kiv_qss[52] = {3};
    nuv_qss[52] = {-1};
    // (40):  TXCH2 + H2 <=> H + CH3
    fwd_A[52]     = 500000;
    fwd_beta[52]  = 2;
    fwd_Ea[52]    = 7229.9200000000001;
    prefactor_units[52]  = 1.0000000000000002e-06;
    activation_units[52] = 0.50321666580471969;
    phase_units[52]      = pow(10,-12.000000);
    is_PD[52] = 0;
    nTB[52] = 0;

    // (41):  SXCH2 + H2O <=> TXCH2 + H2O
    kiv[53] = {7,7};
    nuv[53] = {-1,1};
    kiv_qss[53] = {4,3};
    nuv_qss[53] = {-1,1};
    // (41):  SXCH2 + H2O <=> TXCH2 + H2O
    fwd_A[53]     = 30000000000000;
    fwd_beta[53]  = 0;
    fwd_Ea[53]    = 0;
    prefactor_units[53]  = 1.0000000000000002e-06;
    activation_units[53] = 0.50321666580471969;
    phase_units[53]      = pow(10,-12.000000);
    is_PD[53] = 0;
    nTB[53] = 0;

    // (42):  SXCH2 + H <=> CH + H2
    kiv[54] = {1,2};
    nuv[54] = {-1,1};
    kiv_qss[54] = {4,1};
    nuv_qss[54] = {-1,1};
    // (42):  SXCH2 + H <=> CH + H2
    fwd_A[54]     = 30000000000000;
    fwd_beta[54]  = 0;
    fwd_Ea[54]    = 0;
    prefactor_units[54]  = 1.0000000000000002e-06;
    activation_units[54] = 0.50321666580471969;
    phase_units[54]      = pow(10,-12.000000);
    is_PD[54] = 0;
    nTB[54] = 0;

    // (43):  SXCH2 + O2 <=> H + OH + CO
    kiv[55] = {5,1,4,9};
    nuv[55] = {-1,1,1,1};
    kiv_qss[55] = {4};
    nuv_qss[55] = {-1};
    // (43):  SXCH2 + O2 <=> H + OH + CO
    fwd_A[55]     = 28000000000000;
    fwd_beta[55]  = 0;
    fwd_Ea[55]    = 0;
    prefactor_units[55]  = 1.0000000000000002e-06;
    activation_units[55] = 0.50321666580471969;
    phase_units[55]      = pow(10,-12.000000);
    is_PD[55] = 0;
    nTB[55] = 0;

    // (44):  SXCH2 + O <=> CO + H2
    kiv[56] = {3,9,2};
    nuv[56] = {-1,1,1};
    kiv_qss[56] = {4};
    nuv_qss[56] = {-1};
    // (44):  SXCH2 + O <=> CO + H2
    fwd_A[56]     = 15000000000000;
    fwd_beta[56]  = 0;
    fwd_Ea[56]    = 0;
    prefactor_units[56]  = 1.0000000000000002e-06;
    activation_units[56] = 0.50321666580471969;
    phase_units[56]      = pow(10,-12.000000);
    is_PD[56] = 0;
    nTB[56] = 0;

    // (45):  SXCH2 + O2 <=> CO + H2O
    kiv[57] = {5,9,7};
    nuv[57] = {-1,1,1};
    kiv_qss[57] = {4};
    nuv_qss[57] = {-1};
    // (45):  SXCH2 + O2 <=> CO + H2O
    fwd_A[57]     = 12000000000000;
    fwd_beta[57]  = 0;
    fwd_Ea[57]    = 0;
    prefactor_units[57]  = 1.0000000000000002e-06;
    activation_units[57] = 0.50321666580471969;
    phase_units[57]      = pow(10,-12.000000);
    is_PD[57] = 0;
    nTB[57] = 0;

    // (46):  SXCH2 + H2 <=> CH3 + H
    kiv[58] = {2,10,1};
    nuv[58] = {-1,1,1};
    kiv_qss[58] = {4};
    nuv_qss[58] = {-1};
    // (46):  SXCH2 + H2 <=> CH3 + H
    fwd_A[58]     = 70000000000000;
    fwd_beta[58]  = 0;
    fwd_Ea[58]    = 0;
    prefactor_units[58]  = 1.0000000000000002e-06;
    activation_units[58] = 0.50321666580471969;
    phase_units[58]      = pow(10,-12.000000);
    is_PD[58] = 0;
    nTB[58] = 0;

    // (47):  SXCH2 + O <=> HCO + H
    kiv[59] = {3,1};
    nuv[59] = {-1,1};
    kiv_qss[59] = {4,2};
    nuv_qss[59] = {-1,1};
    // (47):  SXCH2 + O <=> HCO + H
    fwd_A[59]     = 15000000000000;
    fwd_beta[59]  = 0;
    fwd_Ea[59]    = 0;
    prefactor_units[59]  = 1.0000000000000002e-06;
    activation_units[59] = 0.50321666580471969;
    phase_units[59]      = pow(10,-12.000000);
    is_PD[59] = 0;
    nTB[59] = 0;

    // (48):  SXCH2 + H2O => H2 + CH2O
    kiv[60] = {7,2,11};
    nuv[60] = {-1,1,1};
    kiv_qss[60] = {4};
    nuv_qss[60] = {-1};
    // (48):  SXCH2 + H2O => H2 + CH2O
    fwd_A[60]     = 68200000000;
    fwd_beta[60]  = 0.25;
    fwd_Ea[60]    = -934.50999999999999;
    prefactor_units[60]  = 1.0000000000000002e-06;
    activation_units[60] = 0.50321666580471969;
    phase_units[60]      = pow(10,-12.000000);
    is_PD[60] = 0;
    nTB[60] = 0;

    // (49):  SXCH2 + OH <=> CH2O + H
    kiv[61] = {4,11,1};
    nuv[61] = {-1,1,1};
    kiv_qss[61] = {4};
    nuv_qss[61] = {-1};
    // (49):  SXCH2 + OH <=> CH2O + H
    fwd_A[61]     = 30000000000000;
    fwd_beta[61]  = 0;
    fwd_Ea[61]    = 0;
    prefactor_units[61]  = 1.0000000000000002e-06;
    activation_units[61] = 0.50321666580471969;
    phase_units[61]      = pow(10,-12.000000);
    is_PD[61] = 0;
    nTB[61] = 0;

    // (50):  CH3 + OH => H2 + CH2O
    kiv[62] = {10,4,2,11};
    nuv[62] = {-1,-1,1,1};
    kiv_qss[62] = {};
    nuv_qss[62] = {};
    // (50):  CH3 + OH => H2 + CH2O
    fwd_A[62]     = 8000000000;
    fwd_beta[62]  = 0;
    fwd_Ea[62]    = -1754.3;
    prefactor_units[62]  = 1.0000000000000002e-06;
    activation_units[62] = 0.50321666580471969;
    phase_units[62]      = pow(10,-12.000000);
    is_PD[62] = 0;
    nTB[62] = 0;

    // (51):  CH3 + H2O2 <=> CH4 + HO2
    kiv[63] = {10,6,13,8};
    nuv[63] = {-1,-1,1,1};
    kiv_qss[63] = {};
    nuv_qss[63] = {};
    // (51):  CH3 + H2O2 <=> CH4 + HO2
    fwd_A[63]     = 24500;
    fwd_beta[63]  = 2.4700000000000002;
    fwd_Ea[63]    = 5179.25;
    prefactor_units[63]  = 1.0000000000000002e-06;
    activation_units[63] = 0.50321666580471969;
    phase_units[63]      = pow(10,-12.000000);
    is_PD[63] = 0;
    nTB[63] = 0;

    // (52):  CH3 + O2 <=> CH2O + OH
    kiv[64] = {10,5,11,4};
    nuv[64] = {-1,-1,1,1};
    kiv_qss[64] = {};
    nuv_qss[64] = {};
    // (52):  CH3 + O2 <=> CH2O + OH
    fwd_A[64]     = 587000000000;
    fwd_beta[64]  = 0;
    fwd_Ea[64]    = 13840.82;
    prefactor_units[64]  = 1.0000000000000002e-06;
    activation_units[64] = 0.50321666580471969;
    phase_units[64]      = pow(10,-12.000000);
    is_PD[64] = 0;
    nTB[64] = 0;

    // (53):  CH3 + CH <=> C2H3 + H
    kiv[65] = {10,18,1};
    nuv[65] = {-1,1,1};
    kiv_qss[65] = {1};
    nuv_qss[65] = {-1};
    // (53):  CH3 + CH <=> C2H3 + H
    fwd_A[65]     = 30000000000000;
    fwd_beta[65]  = 0;
    fwd_Ea[65]    = 0;
    prefactor_units[65]  = 1.0000000000000002e-06;
    activation_units[65] = 0.50321666580471969;
    phase_units[65]      = pow(10,-12.000000);
    is_PD[65] = 0;
    nTB[65] = 0;

    // (54):  CH3 + O <=> CH2O + H
    kiv[66] = {10,3,11,1};
    nuv[66] = {-1,-1,1,1};
    kiv_qss[66] = {};
    nuv_qss[66] = {};
    // (54):  CH3 + O <=> CH2O + H
    fwd_A[66]     = 50600000000000;
    fwd_beta[66]  = 0;
    fwd_Ea[66]    = 0;
    prefactor_units[66]  = 1.0000000000000002e-06;
    activation_units[66] = 0.50321666580471969;
    phase_units[66]      = pow(10,-12.000000);
    is_PD[66] = 0;
    nTB[66] = 0;

    // (55):  CH3 + C <=> C2H2 + H
    kiv[67] = {10,14,1};
    nuv[67] = {-1,1,1};
    kiv_qss[67] = {0};
    nuv_qss[67] = {-1};
    // (55):  CH3 + C <=> C2H2 + H
    fwd_A[67]     = 50000000000000;
    fwd_beta[67]  = 0;
    fwd_Ea[67]    = 0;
    prefactor_units[67]  = 1.0000000000000002e-06;
    activation_units[67] = 0.50321666580471969;
    phase_units[67]      = pow(10,-12.000000);
    is_PD[67] = 0;
    nTB[67] = 0;

    // (56):  CH3 + H (+M) <=> CH4 (+M)
    kiv[4] = {10,1,13};
    nuv[4] = {-1,-1,1};
    kiv_qss[4] = {};
    nuv_qss[4] = {};
    // (56):  CH3 + H (+M) <=> CH4 (+M)
    fwd_A[4]     = 69200000000000;
    fwd_beta[4]  = 0.17999999999999999;
    fwd_Ea[4]    = 0;
    low_A[4]     = 3.4700000000000003e+38;
    low_beta[4]  = -6.2999999999999998;
    low_Ea[4]    = 5074.0900000000001;
    troe_a[4]    = 0.78300000000000003;
    troe_Tsss[4] = 74;
    troe_Ts[4]   = 2941;
    troe_Tss[4]  = 6964;
    troe_len[4]  = 4;
    prefactor_units[4]  = 1.0000000000000002e-06;
    activation_units[4] = 0.50321666580471969;
    phase_units[4]      = pow(10,-12.000000);
    is_PD[4] = 1;
    nTB[4] = 6;
    TB[4] = (double *) malloc(6 * sizeof(double));
    TBid[4] = (int *) malloc(6 * sizeof(int));
    TBid[4][0] = 2; TB[4][0] = 2; // H2
    TBid[4][1] = 7; TB[4][1] = 6; // H2O
    TBid[4][2] = 9; TB[4][2] = 1.5; // CO
    TBid[4][3] = 12; TB[4][3] = 2; // CO2
    TBid[4][4] = 13; TB[4][4] = 3; // CH4
    TBid[4][5] = 17; TB[4][5] = 3; // C2H6

    // (57):  CH3 + OH <=> TXCH2 + H2O
    kiv[68] = {10,4,7};
    nuv[68] = {-1,-1,1};
    kiv_qss[68] = {3};
    nuv_qss[68] = {1};
    // (57):  CH3 + OH <=> TXCH2 + H2O
    fwd_A[68]     = 56000000;
    fwd_beta[68]  = 1.6000000000000001;
    fwd_Ea[68]    = 5420.6499999999996;
    prefactor_units[68]  = 1.0000000000000002e-06;
    activation_units[68] = 0.50321666580471969;
    phase_units[68]      = pow(10,-12.000000);
    is_PD[68] = 0;
    nTB[68] = 0;

    // (58):  CH3 + SXCH2 <=> C2H4 + H
    kiv[69] = {10,15,1};
    nuv[69] = {-1,1,1};
    kiv_qss[69] = {4};
    nuv_qss[69] = {-1};
    // (58):  CH3 + SXCH2 <=> C2H4 + H
    fwd_A[69]     = 12000000000000;
    fwd_beta[69]  = 0;
    fwd_Ea[69]    = -571.22000000000003;
    prefactor_units[69]  = 1.0000000000000002e-06;
    activation_units[69] = 0.50321666580471969;
    phase_units[69]      = pow(10,-12.000000);
    is_PD[69] = 0;
    nTB[69] = 0;

    // (59):  CH3 + OH <=> SXCH2 + H2O
    kiv[70] = {10,4,7};
    nuv[70] = {-1,-1,1};
    kiv_qss[70] = {4};
    nuv_qss[70] = {1};
    // (59):  CH3 + OH <=> SXCH2 + H2O
    fwd_A[70]     = 6.44e+17;
    fwd_beta[70]  = -1.3400000000000001;
    fwd_Ea[70]    = 1417.3;
    prefactor_units[70]  = 1.0000000000000002e-06;
    activation_units[70] = 0.50321666580471969;
    phase_units[70]      = pow(10,-12.000000);
    is_PD[70] = 0;
    nTB[70] = 0;

    // (60):  2.000000 CH3 <=> C2H5 + H
    kiv[71] = {10,19,1};
    nuv[71] = {-2.0,1,1};
    kiv_qss[71] = {};
    nuv_qss[71] = {};
    // (60):  2.000000 CH3 <=> C2H5 + H
    fwd_A[71]     = 6840000000000;
    fwd_beta[71]  = 0.10000000000000001;
    fwd_Ea[71]    = 10599.9;
    prefactor_units[71]  = 1.0000000000000002e-06;
    activation_units[71] = 0.50321666580471969;
    phase_units[71]      = pow(10,-12.000000);
    is_PD[71] = 0;
    nTB[71] = 0;

    // (61):  CH3 + HO2 <=> CH4 + O2
    kiv[72] = {10,8,13,5};
    nuv[72] = {-1,-1,1,1};
    kiv_qss[72] = {};
    nuv_qss[72] = {};
    // (61):  CH3 + HO2 <=> CH4 + O2
    fwd_A[72]     = 3610000000000;
    fwd_beta[72]  = 0;
    fwd_Ea[72]    = 0;
    prefactor_units[72]  = 1.0000000000000002e-06;
    activation_units[72] = 0.50321666580471969;
    phase_units[72]      = pow(10,-12.000000);
    is_PD[72] = 0;
    nTB[72] = 0;

    // (62):  CH3 + TXCH2 <=> C2H4 + H
    kiv[73] = {10,15,1};
    nuv[73] = {-1,1,1};
    kiv_qss[73] = {3};
    nuv_qss[73] = {-1};
    // (62):  CH3 + TXCH2 <=> C2H4 + H
    fwd_A[73]     = 100000000000000;
    fwd_beta[73]  = 0;
    fwd_Ea[73]    = 0;
    prefactor_units[73]  = 1.0000000000000002e-06;
    activation_units[73] = 0.50321666580471969;
    phase_units[73]      = pow(10,-12.000000);
    is_PD[73] = 0;
    nTB[73] = 0;

    // (63):  CH3 + O => H + H2 + CO
    kiv[74] = {10,3,1,2,9};
    nuv[74] = {-1,-1,1,1,1};
    kiv_qss[74] = {};
    nuv_qss[74] = {};
    // (63):  CH3 + O => H + H2 + CO
    fwd_A[74]     = 33700000000000;
    fwd_beta[74]  = 0;
    fwd_Ea[74]    = 0;
    prefactor_units[74]  = 1.0000000000000002e-06;
    activation_units[74] = 0.50321666580471969;
    phase_units[74]      = pow(10,-12.000000);
    is_PD[74] = 0;
    nTB[74] = 0;

    // (64):  CH4 + CH <=> C2H4 + H
    kiv[75] = {13,15,1};
    nuv[75] = {-1,1,1};
    kiv_qss[75] = {1};
    nuv_qss[75] = {-1};
    // (64):  CH4 + CH <=> C2H4 + H
    fwd_A[75]     = 60000000000000;
    fwd_beta[75]  = 0;
    fwd_Ea[75]    = 0;
    prefactor_units[75]  = 1.0000000000000002e-06;
    activation_units[75] = 0.50321666580471969;
    phase_units[75]      = pow(10,-12.000000);
    is_PD[75] = 0;
    nTB[75] = 0;

    // (65):  CH4 + SXCH2 <=> 2.000000 CH3
    kiv[76] = {13,10};
    nuv[76] = {-1,2.0};
    kiv_qss[76] = {4};
    nuv_qss[76] = {-1};
    // (65):  CH4 + SXCH2 <=> 2.000000 CH3
    fwd_A[76]     = 16000000000000;
    fwd_beta[76]  = 0;
    fwd_Ea[76]    = -571.22000000000003;
    prefactor_units[76]  = 1.0000000000000002e-06;
    activation_units[76] = 0.50321666580471969;
    phase_units[76]      = pow(10,-12.000000);
    is_PD[76] = 0;
    nTB[76] = 0;

    // (66):  CH4 + O <=> CH3 + OH
    kiv[77] = {13,3,10,4};
    nuv[77] = {-1,-1,1,1};
    kiv_qss[77] = {};
    nuv_qss[77] = {};
    // (66):  CH4 + O <=> CH3 + OH
    fwd_A[77]     = 1020000000;
    fwd_beta[77]  = 1.5;
    fwd_Ea[77]    = 8599.4300000000003;
    prefactor_units[77]  = 1.0000000000000002e-06;
    activation_units[77] = 0.50321666580471969;
    phase_units[77]      = pow(10,-12.000000);
    is_PD[77] = 0;
    nTB[77] = 0;

    // (67):  CH4 + OH <=> CH3 + H2O
    kiv[78] = {13,4,10,7};
    nuv[78] = {-1,-1,1,1};
    kiv_qss[78] = {};
    nuv_qss[78] = {};
    // (67):  CH4 + OH <=> CH3 + H2O
    fwd_A[78]     = 100000000;
    fwd_beta[78]  = 1.6000000000000001;
    fwd_Ea[78]    = 3119.02;
    prefactor_units[78]  = 1.0000000000000002e-06;
    activation_units[78] = 0.50321666580471969;
    phase_units[78]      = pow(10,-12.000000);
    is_PD[78] = 0;
    nTB[78] = 0;

    // (68):  CH4 + TXCH2 <=> 2.000000 CH3
    kiv[79] = {13,10};
    nuv[79] = {-1,2.0};
    kiv_qss[79] = {3};
    nuv_qss[79] = {-1};
    // (68):  CH4 + TXCH2 <=> 2.000000 CH3
    fwd_A[79]     = 2460000;
    fwd_beta[79]  = 2;
    fwd_Ea[79]    = 8269.6000000000004;
    prefactor_units[79]  = 1.0000000000000002e-06;
    activation_units[79] = 0.50321666580471969;
    phase_units[79]      = pow(10,-12.000000);
    is_PD[79] = 0;
    nTB[79] = 0;

    // (69):  CH4 + H <=> CH3 + H2
    kiv[80] = {13,1,10,2};
    nuv[80] = {-1,-1,1,1};
    kiv_qss[80] = {};
    nuv_qss[80] = {};
    // (69):  CH4 + H <=> CH3 + H2
    fwd_A[80]     = 660000000;
    fwd_beta[80]  = 1.6200000000000001;
    fwd_Ea[80]    = 10841.299999999999;
    prefactor_units[80]  = 1.0000000000000002e-06;
    activation_units[80] = 0.50321666580471969;
    phase_units[80]      = pow(10,-12.000000);
    is_PD[80] = 0;
    nTB[80] = 0;

    // (70):  TXCH2 + CO (+M) <=> CH2CO (+M)
    kiv[5] = {9,16};
    nuv[5] = {-1,1};
    kiv_qss[5] = {3};
    nuv_qss[5] = {-1};
    // (70):  TXCH2 + CO (+M) <=> CH2CO (+M)
    fwd_A[5]     = 810000000000;
    fwd_beta[5]  = 0.5;
    fwd_Ea[5]    = 4510.04;
    low_A[5]     = 2.69e+33;
    low_beta[5]  = -5.1100000000000003;
    low_Ea[5]    = 7096.0799999999999;
    troe_a[5]    = 0.5907;
    troe_Tsss[5] = 275;
    troe_Ts[5]   = 1226;
    troe_Tss[5]  = 5185;
    troe_len[5]  = 4;
    prefactor_units[5]  = 1.0000000000000002e-06;
    activation_units[5] = 0.50321666580471969;
    phase_units[5]      = pow(10,-12.000000);
    is_PD[5] = 1;
    nTB[5] = 6;
    TB[5] = (double *) malloc(6 * sizeof(double));
    TBid[5] = (int *) malloc(6 * sizeof(int));
    TBid[5][0] = 2; TB[5][0] = 2; // H2
    TBid[5][1] = 7; TB[5][1] = 12; // H2O
    TBid[5][2] = 9; TB[5][2] = 1.75; // CO
    TBid[5][3] = 12; TB[5][3] = 3.6000000000000001; // CO2
    TBid[5][4] = 13; TB[5][4] = 2; // CH4
    TBid[5][5] = 17; TB[5][5] = 3; // C2H6

    // (71):  SXCH2 + CO <=> TXCH2 + CO
    kiv[81] = {9,9};
    nuv[81] = {-1,1};
    kiv_qss[81] = {4,3};
    nuv_qss[81] = {-1,1};
    // (71):  SXCH2 + CO <=> TXCH2 + CO
    fwd_A[81]     = 9000000000000;
    fwd_beta[81]  = 0;
    fwd_Ea[81]    = 0;
    prefactor_units[81]  = 1.0000000000000002e-06;
    activation_units[81] = 0.50321666580471969;
    phase_units[81]      = pow(10,-12.000000);
    is_PD[81] = 0;
    nTB[81] = 0;

    // (72):  CO + O2 <=> CO2 + O
    kiv[82] = {9,5,12,3};
    nuv[82] = {-1,-1,1,1};
    kiv_qss[82] = {};
    nuv_qss[82] = {};
    // (72):  CO + O2 <=> CO2 + O
    fwd_A[82]     = 1120000000000;
    fwd_beta[82]  = 0;
    fwd_Ea[82]    = 47700.760000000002;
    prefactor_units[82]  = 1.0000000000000002e-06;
    activation_units[82] = 0.50321666580471969;
    phase_units[82]      = pow(10,-12.000000);
    is_PD[82] = 0;
    nTB[82] = 0;

    // (73):  CO + OH <=> CO2 + H
    kiv[83] = {9,4,12,1};
    nuv[83] = {-1,-1,1,1};
    kiv_qss[83] = {};
    nuv_qss[83] = {};
    // (73):  CO + OH <=> CO2 + H
    fwd_A[83]     = 87800000000;
    fwd_beta[83]  = 0.029999999999999999;
    fwd_Ea[83]    = -16.73;
    prefactor_units[83]  = 1.0000000000000002e-06;
    activation_units[83] = 0.50321666580471969;
    phase_units[83]      = pow(10,-12.000000);
    is_PD[83] = 0;
    nTB[83] = 0;

    // (74):  CO + H2 (+M) <=> CH2O (+M)
    kiv[6] = {9,2,11};
    nuv[6] = {-1,-1,1};
    kiv_qss[6] = {};
    nuv_qss[6] = {};
    // (74):  CO + H2 (+M) <=> CH2O (+M)
    fwd_A[6]     = 43000000;
    fwd_beta[6]  = 1.5;
    fwd_Ea[6]    = 79600.860000000001;
    low_A[6]     = 5.0699999999999998e+27;
    low_beta[6]  = -3.4199999999999999;
    low_Ea[6]    = 84349.899999999994;
    troe_a[6]    = 0.93200000000000005;
    troe_Tsss[6] = 197;
    troe_Ts[6]   = 1540;
    troe_Tss[6]  = 10300;
    troe_len[6]  = 4;
    prefactor_units[6]  = 1.0000000000000002e-06;
    activation_units[6] = 0.50321666580471969;
    phase_units[6]      = pow(10,-12.000000);
    is_PD[6] = 1;
    nTB[6] = 6;
    TB[6] = (double *) malloc(6 * sizeof(double));
    TBid[6] = (int *) malloc(6 * sizeof(int));
    TBid[6][0] = 2; TB[6][0] = 2; // H2
    TBid[6][1] = 7; TB[6][1] = 12; // H2O
    TBid[6][2] = 9; TB[6][2] = 1.75; // CO
    TBid[6][3] = 12; TB[6][3] = 3.6000000000000001; // CO2
    TBid[6][4] = 13; TB[6][4] = 2; // CH4
    TBid[6][5] = 17; TB[6][5] = 3; // C2H6

    // (75):  CH + CO (+M) <=> HCCO (+M)
    kiv[7] = {9,20};
    nuv[7] = {-1,1};
    kiv_qss[7] = {1};
    nuv_qss[7] = {-1};
    // (75):  CH + CO (+M) <=> HCCO (+M)
    fwd_A[7]     = 50000000000000;
    fwd_beta[7]  = 0;
    fwd_Ea[7]    = 0;
    low_A[7]     = 2.6899999999999998e+28;
    low_beta[7]  = -3.7400000000000002;
    low_Ea[7]    = 1935.95;
    troe_a[7]    = 0.57569999999999999;
    troe_Tsss[7] = 237;
    troe_Ts[7]   = 1652;
    troe_Tss[7]  = 5069;
    troe_len[7]  = 4;
    prefactor_units[7]  = 1.0000000000000002e-06;
    activation_units[7] = 0.50321666580471969;
    phase_units[7]      = pow(10,-12.000000);
    is_PD[7] = 1;
    nTB[7] = 6;
    TB[7] = (double *) malloc(6 * sizeof(double));
    TBid[7] = (int *) malloc(6 * sizeof(int));
    TBid[7][0] = 2; TB[7][0] = 2; // H2
    TBid[7][1] = 7; TB[7][1] = 12; // H2O
    TBid[7][2] = 9; TB[7][2] = 1.75; // CO
    TBid[7][3] = 12; TB[7][3] = 3.6000000000000001; // CO2
    TBid[7][4] = 13; TB[7][4] = 2; // CH4
    TBid[7][5] = 17; TB[7][5] = 3; // C2H6

    // (76):  CO + OH <=> CO2 + H
    kiv[84] = {9,4,12,1};
    nuv[84] = {-1,-1,1,1};
    kiv_qss[84] = {};
    nuv_qss[84] = {};
    // (76):  CO + OH <=> CO2 + H
    fwd_A[84]     = 800000000000;
    fwd_beta[84]  = 0.14000000000000001;
    fwd_Ea[84]    = 7351.8199999999997;
    prefactor_units[84]  = 1.0000000000000002e-06;
    activation_units[84] = 0.50321666580471969;
    phase_units[84]      = pow(10,-12.000000);
    is_PD[84] = 0;
    nTB[84] = 0;

    // (77):  CO + O (+M) <=> CO2 (+M)
    kiv[8] = {9,3,12};
    nuv[8] = {-1,-1,1};
    kiv_qss[8] = {};
    nuv_qss[8] = {};
    // (77):  CO + O (+M) <=> CO2 (+M)
    fwd_A[8]     = 13600000000;
    fwd_beta[8]  = 0;
    fwd_Ea[8]    = 2385.2800000000002;
    low_A[8]     = 1.17e+24;
    low_beta[8]  = -2.79;
    low_Ea[8]    = 4192.1599999999999;
    troe_a[8]    = 1;
    troe_Tsss[8] = 1;
    troe_Ts[8]   = 10000000;
    troe_Tss[8]  = 10000000;
    troe_len[8]  = 4;
    prefactor_units[8]  = 1.0000000000000002e-06;
    activation_units[8] = 0.50321666580471969;
    phase_units[8]      = pow(10,-12.000000);
    is_PD[8] = 1;
    nTB[8] = 6;
    TB[8] = (double *) malloc(6 * sizeof(double));
    TBid[8] = (int *) malloc(6 * sizeof(int));
    TBid[8][0] = 2; TB[8][0] = 2; // H2
    TBid[8][1] = 7; TB[8][1] = 12; // H2O
    TBid[8][2] = 9; TB[8][2] = 1.75; // CO
    TBid[8][3] = 12; TB[8][3] = 3.6000000000000001; // CO2
    TBid[8][4] = 13; TB[8][4] = 2; // CH4
    TBid[8][5] = 17; TB[8][5] = 3; // C2H6

    // (78):  CO + HO2 <=> CO2 + OH
    kiv[85] = {9,8,12,4};
    nuv[85] = {-1,-1,1,1};
    kiv_qss[85] = {};
    nuv_qss[85] = {};
    // (78):  CO + HO2 <=> CO2 + OH
    fwd_A[85]     = 30100000000000;
    fwd_beta[85]  = 0;
    fwd_Ea[85]    = 22999.52;
    prefactor_units[85]  = 1.0000000000000002e-06;
    activation_units[85] = 0.50321666580471969;
    phase_units[85]      = pow(10,-12.000000);
    is_PD[85] = 0;
    nTB[85] = 0;

    // (79):  HCO + H <=> CO + H2
    kiv[86] = {1,9,2};
    nuv[86] = {-1,1,1};
    kiv_qss[86] = {2};
    nuv_qss[86] = {-1};
    // (79):  HCO + H <=> CO + H2
    fwd_A[86]     = 120000000000000;
    fwd_beta[86]  = 0;
    fwd_Ea[86]    = 0;
    prefactor_units[86]  = 1.0000000000000002e-06;
    activation_units[86] = 0.50321666580471969;
    phase_units[86]      = pow(10,-12.000000);
    is_PD[86] = 0;
    nTB[86] = 0;

    // (80):  HCO + H (+M) <=> CH2O (+M)
    kiv[9] = {1,11};
    nuv[9] = {-1,1};
    kiv_qss[9] = {2};
    nuv_qss[9] = {-1};
    // (80):  HCO + H (+M) <=> CH2O (+M)
    fwd_A[9]     = 1090000000000;
    fwd_beta[9]  = 0.47999999999999998;
    fwd_Ea[9]    = -260.51999999999998;
    low_A[9]     = 2.4700000000000001e+24;
    low_beta[9]  = -2.5699999999999998;
    low_Ea[9]    = 425.43000000000001;
    troe_a[9]    = 0.78239999999999998;
    troe_Tsss[9] = 271;
    troe_Ts[9]   = 2755;
    troe_Tss[9]  = 6570;
    troe_len[9]  = 4;
    prefactor_units[9]  = 1.0000000000000002e-06;
    activation_units[9] = 0.50321666580471969;
    phase_units[9]      = pow(10,-12.000000);
    is_PD[9] = 1;
    nTB[9] = 6;
    TB[9] = (double *) malloc(6 * sizeof(double));
    TBid[9] = (int *) malloc(6 * sizeof(int));
    TBid[9][0] = 2; TB[9][0] = 2; // H2
    TBid[9][1] = 7; TB[9][1] = 12; // H2O
    TBid[9][2] = 9; TB[9][2] = 1.75; // CO
    TBid[9][3] = 12; TB[9][3] = 3.6000000000000001; // CO2
    TBid[9][4] = 13; TB[9][4] = 2; // CH4
    TBid[9][5] = 17; TB[9][5] = 3; // C2H6

    // (81):  CH3 + HCO <=> CH3CHO
    kiv[87] = {10,21};
    nuv[87] = {-1,1};
    kiv_qss[87] = {2};
    nuv_qss[87] = {-1};
    // (81):  CH3 + HCO <=> CH3CHO
    fwd_A[87]     = 50000000000000;
    fwd_beta[87]  = 0;
    fwd_Ea[87]    = 0;
    prefactor_units[87]  = 1.0000000000000002e-06;
    activation_units[87] = 0.50321666580471969;
    phase_units[87]      = pow(10,-12.000000);
    is_PD[87] = 0;
    nTB[87] = 0;

    // (82):  HCO + M <=> CO + H + M
    kiv[19] = {9,1};
    nuv[19] = {1,1};
    kiv_qss[19] = {2};
    nuv_qss[19] = {-1};
    // (82):  HCO + M <=> CO + H + M
    fwd_A[19]     = 1.87e+17;
    fwd_beta[19]  = -1;
    fwd_Ea[19]    = 17000.48;
    prefactor_units[19]  = 1.0000000000000002e-06;
    activation_units[19] = 0.50321666580471969;
    phase_units[19]      = pow(10,-6.000000);
    is_PD[19] = 0;
    nTB[19] = 6;
    TB[19] = (double *) malloc(6 * sizeof(double));
    TBid[19] = (int *) malloc(6 * sizeof(int));
    TBid[19][0] = 2; TB[19][0] = 2; // H2
    TBid[19][1] = 7; TB[19][1] = 0; // H2O
    TBid[19][2] = 9; TB[19][2] = 1.75; // CO
    TBid[19][3] = 12; TB[19][3] = 3.6000000000000001; // CO2
    TBid[19][4] = 13; TB[19][4] = 2; // CH4
    TBid[19][5] = 17; TB[19][5] = 3; // C2H6

    // (83):  HCO + H2O <=> CO + H + H2O
    kiv[88] = {7,9,1,7};
    nuv[88] = {-1,1,1,1};
    kiv_qss[88] = {2};
    nuv_qss[88] = {-1};
    // (83):  HCO + H2O <=> CO + H + H2O
    fwd_A[88]     = 2.24e+18;
    fwd_beta[88]  = -1;
    fwd_Ea[88]    = 17000.48;
    prefactor_units[88]  = 1.0000000000000002e-06;
    activation_units[88] = 0.50321666580471969;
    phase_units[88]      = pow(10,-12.000000);
    is_PD[88] = 0;
    nTB[88] = 0;

    // (84):  HCO + O <=> CO + OH
    kiv[89] = {3,9,4};
    nuv[89] = {-1,1,1};
    kiv_qss[89] = {2};
    nuv_qss[89] = {-1};
    // (84):  HCO + O <=> CO + OH
    fwd_A[89]     = 30000000000000;
    fwd_beta[89]  = 0;
    fwd_Ea[89]    = 0;
    prefactor_units[89]  = 1.0000000000000002e-06;
    activation_units[89] = 0.50321666580471969;
    phase_units[89]      = pow(10,-12.000000);
    is_PD[89] = 0;
    nTB[89] = 0;

    // (85):  HCO + OH <=> CO + H2O
    kiv[90] = {4,9,7};
    nuv[90] = {-1,1,1};
    kiv_qss[90] = {2};
    nuv_qss[90] = {-1};
    // (85):  HCO + OH <=> CO + H2O
    fwd_A[90]     = 30200000000000;
    fwd_beta[90]  = 0;
    fwd_Ea[90]    = 0;
    prefactor_units[90]  = 1.0000000000000002e-06;
    activation_units[90] = 0.50321666580471969;
    phase_units[90]      = pow(10,-12.000000);
    is_PD[90] = 0;
    nTB[90] = 0;

    // (86):  CH3 + HCO <=> CH4 + CO
    kiv[91] = {10,13,9};
    nuv[91] = {-1,1,1};
    kiv_qss[91] = {2};
    nuv_qss[91] = {-1};
    // (86):  CH3 + HCO <=> CH4 + CO
    fwd_A[91]     = 26500000000000;
    fwd_beta[91]  = 0;
    fwd_Ea[91]    = 0;
    prefactor_units[91]  = 1.0000000000000002e-06;
    activation_units[91] = 0.50321666580471969;
    phase_units[91]      = pow(10,-12.000000);
    is_PD[91] = 0;
    nTB[91] = 0;

    // (87):  HCO + O <=> CO2 + H
    kiv[92] = {3,12,1};
    nuv[92] = {-1,1,1};
    kiv_qss[92] = {2};
    nuv_qss[92] = {-1};
    // (87):  HCO + O <=> CO2 + H
    fwd_A[92]     = 30000000000000;
    fwd_beta[92]  = 0;
    fwd_Ea[92]    = 0;
    prefactor_units[92]  = 1.0000000000000002e-06;
    activation_units[92] = 0.50321666580471969;
    phase_units[92]      = pow(10,-12.000000);
    is_PD[92] = 0;
    nTB[92] = 0;

    // (88):  HCO + O2 <=> CO + HO2
    kiv[93] = {5,9,8};
    nuv[93] = {-1,1,1};
    kiv_qss[93] = {2};
    nuv_qss[93] = {-1};
    // (88):  HCO + O2 <=> CO + HO2
    fwd_A[93]     = 12000000000;
    fwd_beta[93]  = 0.81000000000000005;
    fwd_Ea[93]    = -726.58000000000004;
    prefactor_units[93]  = 1.0000000000000002e-06;
    activation_units[93] = 0.50321666580471969;
    phase_units[93]      = pow(10,-12.000000);
    is_PD[93] = 0;
    nTB[93] = 0;

    // (89):  CH2O + H <=> HCO + H2
    kiv[94] = {11,1,2};
    nuv[94] = {-1,-1,1};
    kiv_qss[94] = {2};
    nuv_qss[94] = {1};
    // (89):  CH2O + H <=> HCO + H2
    fwd_A[94]     = 57400000;
    fwd_beta[94]  = 1.8999999999999999;
    fwd_Ea[94]    = 2741.4000000000001;
    prefactor_units[94]  = 1.0000000000000002e-06;
    activation_units[94] = 0.50321666580471969;
    phase_units[94]      = pow(10,-12.000000);
    is_PD[94] = 0;
    nTB[94] = 0;

    // (90):  CH2O + O <=> HCO + OH
    kiv[95] = {11,3,4};
    nuv[95] = {-1,-1,1};
    kiv_qss[95] = {2};
    nuv_qss[95] = {1};
    // (90):  CH2O + O <=> HCO + OH
    fwd_A[95]     = 39000000000000;
    fwd_beta[95]  = 0;
    fwd_Ea[95]    = 3539.6700000000001;
    prefactor_units[95]  = 1.0000000000000002e-06;
    activation_units[95] = 0.50321666580471969;
    phase_units[95]      = pow(10,-12.000000);
    is_PD[95] = 0;
    nTB[95] = 0;

    // (91):  CH3 + CH2O <=> CH4 + HCO
    kiv[96] = {10,11,13};
    nuv[96] = {-1,-1,1};
    kiv_qss[96] = {2};
    nuv_qss[96] = {1};
    // (91):  CH3 + CH2O <=> CH4 + HCO
    fwd_A[96]     = 3320;
    fwd_beta[96]  = 2.8100000000000001;
    fwd_Ea[96]    = 5860.4200000000001;
    prefactor_units[96]  = 1.0000000000000002e-06;
    activation_units[96] = 0.50321666580471969;
    phase_units[96]      = pow(10,-12.000000);
    is_PD[96] = 0;
    nTB[96] = 0;

    // (92):  CH2O + OH <=> HCO + H2O
    kiv[97] = {11,4,7};
    nuv[97] = {-1,-1,1};
    kiv_qss[97] = {2};
    nuv_qss[97] = {1};
    // (92):  CH2O + OH <=> HCO + H2O
    fwd_A[97]     = 3430000000;
    fwd_beta[97]  = 1.1799999999999999;
    fwd_Ea[97]    = -446.94;
    prefactor_units[97]  = 1.0000000000000002e-06;
    activation_units[97] = 0.50321666580471969;
    phase_units[97]      = pow(10,-12.000000);
    is_PD[97] = 0;
    nTB[97] = 0;

    // (93):  CH2O + CH <=> CH2CO + H
    kiv[98] = {11,16,1};
    nuv[98] = {-1,1,1};
    kiv_qss[98] = {1};
    nuv_qss[98] = {-1};
    // (93):  CH2O + CH <=> CH2CO + H
    fwd_A[98]     = 94600000000000;
    fwd_beta[98]  = 0;
    fwd_Ea[98]    = -516.25;
    prefactor_units[98]  = 1.0000000000000002e-06;
    activation_units[98] = 0.50321666580471969;
    phase_units[98]      = pow(10,-12.000000);
    is_PD[98] = 0;
    nTB[98] = 0;

    // (94):  CH2O + O2 <=> HCO + HO2
    kiv[99] = {11,5,8};
    nuv[99] = {-1,-1,1};
    kiv_qss[99] = {2};
    nuv_qss[99] = {1};
    // (94):  CH2O + O2 <=> HCO + HO2
    fwd_A[99]     = 100000000000000;
    fwd_beta[99]  = 0;
    fwd_Ea[99]    = 40000;
    prefactor_units[99]  = 1.0000000000000002e-06;
    activation_units[99] = 0.50321666580471969;
    phase_units[99]      = pow(10,-12.000000);
    is_PD[99] = 0;
    nTB[99] = 0;

    // (95):  CH2O + HO2 <=> HCO + H2O2
    kiv[100] = {11,8,6};
    nuv[100] = {-1,-1,1};
    kiv_qss[100] = {2};
    nuv_qss[100] = {1};
    // (95):  CH2O + HO2 <=> HCO + H2O2
    fwd_A[100]     = 5600000;
    fwd_beta[100]  = 2;
    fwd_Ea[100]    = 12000.48;
    prefactor_units[100]  = 1.0000000000000002e-06;
    activation_units[100] = 0.50321666580471969;
    phase_units[100]      = pow(10,-12.000000);
    is_PD[100] = 0;
    nTB[100] = 0;

    // (96):  2.000000 H + CO2 <=> H2 + CO2
    kiv[101] = {1,12,2,12};
    nuv[101] = {-2.0,-1,1,1};
    kiv_qss[101] = {};
    nuv_qss[101] = {};
    // (96):  2.000000 H + CO2 <=> H2 + CO2
    fwd_A[101]     = 5.5e+20;
    fwd_beta[101]  = -2;
    fwd_Ea[101]    = 0;
    prefactor_units[101]  = 1.0000000000000002e-12;
    activation_units[101] = 0.50321666580471969;
    phase_units[101]      = pow(10,-18.000000);
    is_PD[101] = 0;
    nTB[101] = 0;

    // (97):  SXCH2 + CO2 <=> TXCH2 + CO2
    kiv[102] = {12,12};
    nuv[102] = {-1,1};
    kiv_qss[102] = {4,3};
    nuv_qss[102] = {-1,1};
    // (97):  SXCH2 + CO2 <=> TXCH2 + CO2
    fwd_A[102]     = 7000000000000;
    fwd_beta[102]  = 0;
    fwd_Ea[102]    = 0;
    prefactor_units[102]  = 1.0000000000000002e-06;
    activation_units[102] = 0.50321666580471969;
    phase_units[102]      = pow(10,-12.000000);
    is_PD[102] = 0;
    nTB[102] = 0;

    // (98):  SXCH2 + CO2 <=> CH2O + CO
    kiv[103] = {12,11,9};
    nuv[103] = {-1,1,1};
    kiv_qss[103] = {4};
    nuv_qss[103] = {-1};
    // (98):  SXCH2 + CO2 <=> CH2O + CO
    fwd_A[103]     = 14000000000000;
    fwd_beta[103]  = 0;
    fwd_Ea[103]    = 0;
    prefactor_units[103]  = 1.0000000000000002e-06;
    activation_units[103] = 0.50321666580471969;
    phase_units[103]      = pow(10,-12.000000);
    is_PD[103] = 0;
    nTB[103] = 0;

    // (99):  CH + CO2 <=> HCO + CO
    kiv[104] = {12,9};
    nuv[104] = {-1,1};
    kiv_qss[104] = {1,2};
    nuv_qss[104] = {-1,1};
    // (99):  CH + CO2 <=> HCO + CO
    fwd_A[104]     = 190000000000000;
    fwd_beta[104]  = 0;
    fwd_Ea[104]    = 15791.110000000001;
    prefactor_units[104]  = 1.0000000000000002e-06;
    activation_units[104] = 0.50321666580471969;
    phase_units[104]      = pow(10,-12.000000);
    is_PD[104] = 0;
    nTB[104] = 0;

    // (100):  C2H2 + O <=> TXCH2 + CO
    kiv[105] = {14,3,9};
    nuv[105] = {-1,-1,1};
    kiv_qss[105] = {3};
    nuv_qss[105] = {1};
    // (100):  C2H2 + O <=> TXCH2 + CO
    fwd_A[105]     = 12500000;
    fwd_beta[105]  = 2;
    fwd_Ea[105]    = 1900.0999999999999;
    prefactor_units[105]  = 1.0000000000000002e-06;
    activation_units[105] = 0.50321666580471969;
    phase_units[105]      = pow(10,-12.000000);
    is_PD[105] = 0;
    nTB[105] = 0;

    // (101):  C2H2 + OH <=> CH3 + CO
    kiv[106] = {14,4,10,9};
    nuv[106] = {-1,-1,1,1};
    kiv_qss[106] = {};
    nuv_qss[106] = {};
    // (101):  C2H2 + OH <=> CH3 + CO
    fwd_A[106]     = 1280000000;
    fwd_beta[106]  = 0.72999999999999998;
    fwd_Ea[106]    = 2578.8699999999999;
    prefactor_units[106]  = 1.0000000000000002e-06;
    activation_units[106] = 0.50321666580471969;
    phase_units[106]      = pow(10,-12.000000);
    is_PD[106] = 0;
    nTB[106] = 0;

    // (102):  C2H2 + H (+M) <=> C2H3 (+M)
    kiv[10] = {14,1,18};
    nuv[10] = {-1,-1,1};
    kiv_qss[10] = {};
    nuv_qss[10] = {};
    // (102):  C2H2 + H (+M) <=> C2H3 (+M)
    fwd_A[10]     = 17100000000;
    fwd_beta[10]  = 1.27;
    fwd_Ea[10]    = 2707.9299999999998;
    low_A[10]     = 6.3399999999999996e+31;
    low_beta[10]  = -4.6600000000000001;
    low_Ea[10]    = 3781.0700000000002;
    troe_a[10]    = 0.2122;
    troe_Tsss[10] = 1;
    troe_Ts[10]   = -10210;
    troe_len[10]  = 3;
    prefactor_units[10]  = 1.0000000000000002e-06;
    activation_units[10] = 0.50321666580471969;
    phase_units[10]      = pow(10,-12.000000);
    is_PD[10] = 1;
    nTB[10] = 6;
    TB[10] = (double *) malloc(6 * sizeof(double));
    TBid[10] = (int *) malloc(6 * sizeof(int));
    TBid[10][0] = 2; TB[10][0] = 2; // H2
    TBid[10][1] = 7; TB[10][1] = 12; // H2O
    TBid[10][2] = 9; TB[10][2] = 1.75; // CO
    TBid[10][3] = 12; TB[10][3] = 3.6000000000000001; // CO2
    TBid[10][4] = 13; TB[10][4] = 2; // CH4
    TBid[10][5] = 17; TB[10][5] = 3; // C2H6

    // (103):  C2H2 + OH <=> CH2CO + H
    kiv[107] = {14,4,16,1};
    nuv[107] = {-1,-1,1,1};
    kiv_qss[107] = {};
    nuv_qss[107] = {};
    // (103):  C2H2 + OH <=> CH2CO + H
    fwd_A[107]     = 7530000;
    fwd_beta[107]  = 1.55;
    fwd_Ea[107]    = 2105.6399999999999;
    prefactor_units[107]  = 1.0000000000000002e-06;
    activation_units[107] = 0.50321666580471969;
    phase_units[107]      = pow(10,-12.000000);
    is_PD[107] = 0;
    nTB[107] = 0;

    // (104):  C2H2 + O <=> HCCO + H
    kiv[108] = {14,3,20,1};
    nuv[108] = {-1,-1,1,1};
    kiv_qss[108] = {};
    nuv_qss[108] = {};
    // (104):  C2H2 + O <=> HCCO + H
    fwd_A[108]     = 8100000;
    fwd_beta[108]  = 2;
    fwd_Ea[108]    = 1900.0999999999999;
    prefactor_units[108]  = 1.0000000000000002e-06;
    activation_units[108] = 0.50321666580471969;
    phase_units[108]      = pow(10,-12.000000);
    is_PD[108] = 0;
    nTB[108] = 0;

    // (105):  C2H3 + OH <=> C2H2 + H2O
    kiv[109] = {18,4,14,7};
    nuv[109] = {-1,-1,1,1};
    kiv_qss[109] = {};
    nuv_qss[109] = {};
    // (105):  C2H3 + OH <=> C2H2 + H2O
    fwd_A[109]     = 5000000000000;
    fwd_beta[109]  = 0;
    fwd_Ea[109]    = 0;
    prefactor_units[109]  = 1.0000000000000002e-06;
    activation_units[109] = 0.50321666580471969;
    phase_units[109]      = pow(10,-12.000000);
    is_PD[109] = 0;
    nTB[109] = 0;

    // (106):  C2H3 + O2 <=> CH2CHO + O
    kiv[110] = {18,5,22,3};
    nuv[110] = {-1,-1,1,1};
    kiv_qss[110] = {};
    nuv_qss[110] = {};
    // (106):  C2H3 + O2 <=> CH2CHO + O
    fwd_A[110]     = 303000000000;
    fwd_beta[110]  = 0.28999999999999998;
    fwd_Ea[110]    = 11.949999999999999;
    prefactor_units[110]  = 1.0000000000000002e-06;
    activation_units[110] = 0.50321666580471969;
    phase_units[110]      = pow(10,-12.000000);
    is_PD[110] = 0;
    nTB[110] = 0;

    // (107):  C2H3 + O <=> CH2CHO
    kiv[111] = {18,3,22};
    nuv[111] = {-1,-1,1};
    kiv_qss[111] = {};
    nuv_qss[111] = {};
    // (107):  C2H3 + O <=> CH2CHO
    fwd_A[111]     = 10300000000000;
    fwd_beta[111]  = 0.20999999999999999;
    fwd_Ea[111]    = -427.81999999999999;
    prefactor_units[111]  = 1.0000000000000002e-06;
    activation_units[111] = 0.50321666580471969;
    phase_units[111]      = pow(10,-12.000000);
    is_PD[111] = 0;
    nTB[111] = 0;

    // (108):  C2H3 + H <=> C2H2 + H2
    kiv[112] = {18,1,14,2};
    nuv[112] = {-1,-1,1,1};
    kiv_qss[112] = {};
    nuv_qss[112] = {};
    // (108):  C2H3 + H <=> C2H2 + H2
    fwd_A[112]     = 30000000000000;
    fwd_beta[112]  = 0;
    fwd_Ea[112]    = 0;
    prefactor_units[112]  = 1.0000000000000002e-06;
    activation_units[112] = 0.50321666580471969;
    phase_units[112]      = pow(10,-12.000000);
    is_PD[112] = 0;
    nTB[112] = 0;

    // (109):  C2H3 + CH3 <=> C2H2 + CH4
    kiv[113] = {18,10,14,13};
    nuv[113] = {-1,-1,1,1};
    kiv_qss[113] = {};
    nuv_qss[113] = {};
    // (109):  C2H3 + CH3 <=> C2H2 + CH4
    fwd_A[113]     = 9030000000000;
    fwd_beta[113]  = 0;
    fwd_Ea[113]    = -764.82000000000005;
    prefactor_units[113]  = 1.0000000000000002e-06;
    activation_units[113] = 0.50321666580471969;
    phase_units[113]      = pow(10,-12.000000);
    is_PD[113] = 0;
    nTB[113] = 0;

    // (110):  C2H3 + O2 <=> HCO + CH2O
    kiv[114] = {18,5,11};
    nuv[114] = {-1,-1,1};
    kiv_qss[114] = {2};
    nuv_qss[114] = {1};
    // (110):  C2H3 + O2 <=> HCO + CH2O
    fwd_A[114]     = 45800000000000000;
    fwd_beta[114]  = -1.3899999999999999;
    fwd_Ea[114]    = 1015.77;
    prefactor_units[114]  = 1.0000000000000002e-06;
    activation_units[114] = 0.50321666580471969;
    phase_units[114]      = pow(10,-12.000000);
    is_PD[114] = 0;
    nTB[114] = 0;

    // (111):  C2H3 + H (+M) <=> C2H4 (+M)
    kiv[11] = {18,1,15};
    nuv[11] = {-1,-1,1};
    kiv_qss[11] = {};
    nuv_qss[11] = {};
    // (111):  C2H3 + H (+M) <=> C2H4 (+M)
    fwd_A[11]     = 6080000000000;
    fwd_beta[11]  = 0.27000000000000002;
    fwd_Ea[11]    = 279.63999999999999;
    low_A[11]     = 1.3999999999999999e+30;
    low_beta[11]  = -3.8599999999999999;
    low_Ea[11]    = 3319.79;
    troe_a[11]    = 0.78200000000000003;
    troe_Tsss[11] = 207.5;
    troe_Ts[11]   = 2663;
    troe_Tss[11]  = 6095;
    troe_len[11]  = 4;
    prefactor_units[11]  = 1.0000000000000002e-06;
    activation_units[11] = 0.50321666580471969;
    phase_units[11]      = pow(10,-12.000000);
    is_PD[11] = 1;
    nTB[11] = 6;
    TB[11] = (double *) malloc(6 * sizeof(double));
    TBid[11] = (int *) malloc(6 * sizeof(int));
    TBid[11][0] = 2; TB[11][0] = 2; // H2
    TBid[11][1] = 7; TB[11][1] = 12; // H2O
    TBid[11][2] = 9; TB[11][2] = 1.75; // CO
    TBid[11][3] = 12; TB[11][3] = 3.6000000000000001; // CO2
    TBid[11][4] = 13; TB[11][4] = 2; // CH4
    TBid[11][5] = 17; TB[11][5] = 3; // C2H6

    // (112):  C2H3 + H2O2 <=> C2H4 + HO2
    kiv[115] = {18,6,15,8};
    nuv[115] = {-1,-1,1,1};
    kiv_qss[115] = {};
    nuv_qss[115] = {};
    // (112):  C2H3 + H2O2 <=> C2H4 + HO2
    fwd_A[115]     = 12100000000;
    fwd_beta[115]  = 0;
    fwd_Ea[115]    = -595.12;
    prefactor_units[115]  = 1.0000000000000002e-06;
    activation_units[115] = 0.50321666580471969;
    phase_units[115]      = pow(10,-12.000000);
    is_PD[115] = 0;
    nTB[115] = 0;

    // (113):  C2H3 + O2 <=> C2H2 + HO2
    kiv[116] = {18,5,14,8};
    nuv[116] = {-1,-1,1,1};
    kiv_qss[116] = {};
    nuv_qss[116] = {};
    // (113):  C2H3 + O2 <=> C2H2 + HO2
    fwd_A[116]     = 1340000;
    fwd_beta[116]  = 1.6100000000000001;
    fwd_Ea[116]    = -384.80000000000001;
    prefactor_units[116]  = 1.0000000000000002e-06;
    activation_units[116] = 0.50321666580471969;
    phase_units[116]      = pow(10,-12.000000);
    is_PD[116] = 0;
    nTB[116] = 0;

    // (114):  C2H4 + CH3 <=> C2H3 + CH4
    kiv[117] = {15,10,18,13};
    nuv[117] = {-1,-1,1,1};
    kiv_qss[117] = {};
    nuv_qss[117] = {};
    // (114):  C2H4 + CH3 <=> C2H3 + CH4
    fwd_A[117]     = 227000;
    fwd_beta[117]  = 2;
    fwd_Ea[117]    = 9199.3299999999999;
    prefactor_units[117]  = 1.0000000000000002e-06;
    activation_units[117] = 0.50321666580471969;
    phase_units[117]      = pow(10,-12.000000);
    is_PD[117] = 0;
    nTB[117] = 0;

    // (115):  C2H4 + H (+M) <=> C2H5 (+M)
    kiv[12] = {15,1,19};
    nuv[12] = {-1,-1,1};
    kiv_qss[12] = {};
    nuv_qss[12] = {};
    // (115):  C2H4 + H (+M) <=> C2H5 (+M)
    fwd_A[12]     = 1370000000;
    fwd_beta[12]  = 1.46;
    fwd_Ea[12]    = 1355.1600000000001;
    low_A[12]     = 2.0300000000000001e+39;
    low_beta[12]  = -6.6399999999999997;
    low_Ea[12]    = 5769.6000000000004;
    troe_a[12]    = -0.56899999999999995;
    troe_Tsss[12] = 299;
    troe_Ts[12]   = -9147;
    troe_Tss[12]  = 152.40000000000001;
    troe_len[12]  = 4;
    prefactor_units[12]  = 1.0000000000000002e-06;
    activation_units[12] = 0.50321666580471969;
    phase_units[12]      = pow(10,-12.000000);
    is_PD[12] = 1;
    nTB[12] = 6;
    TB[12] = (double *) malloc(6 * sizeof(double));
    TBid[12] = (int *) malloc(6 * sizeof(int));
    TBid[12][0] = 2; TB[12][0] = 2; // H2
    TBid[12][1] = 7; TB[12][1] = 12; // H2O
    TBid[12][2] = 9; TB[12][2] = 1.75; // CO
    TBid[12][3] = 12; TB[12][3] = 3.6000000000000001; // CO2
    TBid[12][4] = 13; TB[12][4] = 2; // CH4
    TBid[12][5] = 17; TB[12][5] = 3; // C2H6

    // (116):  C2H4 + O2 => CH3 + CO2 + H
    kiv[118] = {15,5,10,12,1};
    nuv[118] = {-1,-1,1,1,1};
    kiv_qss[118] = {};
    nuv_qss[118] = {};
    // (116):  C2H4 + O2 => CH3 + CO2 + H
    fwd_A[118]     = 4900000000000;
    fwd_beta[118]  = 0.41999999999999998;
    fwd_Ea[118]    = 75800.669999999998;
    prefactor_units[118]  = 1.0000000000000002e-06;
    activation_units[118] = 0.50321666580471969;
    phase_units[118]      = pow(10,-12.000000);
    is_PD[118] = 0;
    nTB[118] = 0;

    // (117):  C2H4 + OH <=> C2H3 + H2O
    kiv[119] = {15,4,18,7};
    nuv[119] = {-1,-1,1,1};
    kiv_qss[119] = {};
    nuv_qss[119] = {};
    // (117):  C2H4 + OH <=> C2H3 + H2O
    fwd_A[119]     = 0.13100000000000001;
    fwd_beta[119]  = 4.2000000000000002;
    fwd_Ea[119]    = -860.41999999999996;
    prefactor_units[119]  = 1.0000000000000002e-06;
    activation_units[119] = 0.50321666580471969;
    phase_units[119]      = pow(10,-12.000000);
    is_PD[119] = 0;
    nTB[119] = 0;

    // (118):  C2H4 + OH <=> C2H5O
    kiv[120] = {15,4,23};
    nuv[120] = {-1,-1,1};
    kiv_qss[120] = {};
    nuv_qss[120] = {};
    // (118):  C2H4 + OH <=> C2H5O
    fwd_A[120]     = 3.7500000000000003e+36;
    fwd_beta[120]  = -7.7999999999999998;
    fwd_Ea[120]    = 7060.2299999999996;
    prefactor_units[120]  = 1.0000000000000002e-06;
    activation_units[120] = 0.50321666580471969;
    phase_units[120]      = pow(10,-12.000000);
    is_PD[120] = 0;
    nTB[120] = 0;

    // (119):  C2H4 + O <=> CH2CHO + H
    kiv[121] = {15,3,22,1};
    nuv[121] = {-1,-1,1,1};
    kiv_qss[121] = {};
    nuv_qss[121] = {};
    // (119):  C2H4 + O <=> CH2CHO + H
    fwd_A[121]     = 7660000000;
    fwd_beta[121]  = 0.88;
    fwd_Ea[121]    = 1140.0599999999999;
    prefactor_units[121]  = 1.0000000000000002e-06;
    activation_units[121] = 0.50321666580471969;
    phase_units[121]      = pow(10,-12.000000);
    is_PD[121] = 0;
    nTB[121] = 0;

    // (120):  C2H4 + O <=> CH3 + HCO
    kiv[122] = {15,3,10};
    nuv[122] = {-1,-1,1};
    kiv_qss[122] = {2};
    nuv_qss[122] = {1};
    // (120):  C2H4 + O <=> CH3 + HCO
    fwd_A[122]     = 389000000;
    fwd_beta[122]  = 1.3600000000000001;
    fwd_Ea[122]    = 886.71000000000004;
    prefactor_units[122]  = 1.0000000000000002e-06;
    activation_units[122] = 0.50321666580471969;
    phase_units[122]      = pow(10,-12.000000);
    is_PD[122] = 0;
    nTB[122] = 0;

    // (121):  C2H4 + O2 <=> C2H3 + HO2
    kiv[123] = {15,5,18,8};
    nuv[123] = {-1,-1,1,1};
    kiv_qss[123] = {};
    nuv_qss[123] = {};
    // (121):  C2H4 + O2 <=> C2H3 + HO2
    fwd_A[123]     = 42200000000000;
    fwd_beta[123]  = 0;
    fwd_Ea[123]    = 62100.860000000001;
    prefactor_units[123]  = 1.0000000000000002e-06;
    activation_units[123] = 0.50321666580471969;
    phase_units[123]      = pow(10,-12.000000);
    is_PD[123] = 0;
    nTB[123] = 0;

    // (122):  C2H4 + H <=> C2H3 + H2
    kiv[124] = {15,1,18,2};
    nuv[124] = {-1,-1,1,1};
    kiv_qss[124] = {};
    nuv_qss[124] = {};
    // (122):  C2H4 + H <=> C2H3 + H2
    fwd_A[124]     = 127000;
    fwd_beta[124]  = 2.75;
    fwd_Ea[124]    = 11649.139999999999;
    prefactor_units[124]  = 1.0000000000000002e-06;
    activation_units[124] = 0.50321666580471969;
    phase_units[124]      = pow(10,-12.000000);
    is_PD[124] = 0;
    nTB[124] = 0;

    // (123):  C2H4 + O <=> TXCH2 + CH2O
    kiv[125] = {15,3,11};
    nuv[125] = {-1,-1,1};
    kiv_qss[125] = {3};
    nuv_qss[125] = {1};
    // (123):  C2H4 + O <=> TXCH2 + CH2O
    fwd_A[125]     = 71500;
    fwd_beta[125]  = 2.4700000000000002;
    fwd_Ea[125]    = 929.73000000000002;
    prefactor_units[125]  = 1.0000000000000002e-06;
    activation_units[125] = 0.50321666580471969;
    phase_units[125]      = pow(10,-12.000000);
    is_PD[125] = 0;
    nTB[125] = 0;

    // (124):  C2H5 + HO2 <=> C2H4 + H2O2
    kiv[126] = {19,8,15,6};
    nuv[126] = {-1,-1,1,1};
    kiv_qss[126] = {};
    nuv_qss[126] = {};
    // (124):  C2H5 + HO2 <=> C2H4 + H2O2
    fwd_A[126]     = 300000000000;
    fwd_beta[126]  = 0;
    fwd_Ea[126]    = 0;
    prefactor_units[126]  = 1.0000000000000002e-06;
    activation_units[126] = 0.50321666580471969;
    phase_units[126]      = pow(10,-12.000000);
    is_PD[126] = 0;
    nTB[126] = 0;

    // (125):  C2H5 + H (+M) <=> C2H6 (+M)
    kiv[13] = {19,1,17};
    nuv[13] = {-1,-1,1};
    kiv_qss[13] = {};
    nuv_qss[13] = {};
    // (125):  C2H5 + H (+M) <=> C2H6 (+M)
    fwd_A[13]     = 5.21e+17;
    fwd_beta[13]  = -0.98999999999999999;
    fwd_Ea[13]    = 1579.8299999999999;
    low_A[13]     = 1.9900000000000001e+41;
    low_beta[13]  = -7.0800000000000001;
    low_Ea[13]    = 6684.9899999999998;
    troe_a[13]    = 0.84219999999999995;
    troe_Tsss[13] = 125;
    troe_Ts[13]   = 2219;
    troe_Tss[13]  = 6882;
    troe_len[13]  = 4;
    prefactor_units[13]  = 1.0000000000000002e-06;
    activation_units[13] = 0.50321666580471969;
    phase_units[13]      = pow(10,-12.000000);
    is_PD[13] = 1;
    nTB[13] = 6;
    TB[13] = (double *) malloc(6 * sizeof(double));
    TBid[13] = (int *) malloc(6 * sizeof(int));
    TBid[13][0] = 2; TB[13][0] = 2; // H2
    TBid[13][1] = 7; TB[13][1] = 12; // H2O
    TBid[13][2] = 9; TB[13][2] = 1.75; // CO
    TBid[13][3] = 12; TB[13][3] = 3.6000000000000001; // CO2
    TBid[13][4] = 13; TB[13][4] = 2; // CH4
    TBid[13][5] = 17; TB[13][5] = 3; // C2H6

    // (126):  C2H5 + HO2 <=> C2H5O + OH
    kiv[127] = {19,8,23,4};
    nuv[127] = {-1,-1,1,1};
    kiv_qss[127] = {};
    nuv_qss[127] = {};
    // (126):  C2H5 + HO2 <=> C2H5O + OH
    fwd_A[127]     = 31000000000000;
    fwd_beta[127]  = 0;
    fwd_Ea[127]    = 0;
    prefactor_units[127]  = 1.0000000000000002e-06;
    activation_units[127] = 0.50321666580471969;
    phase_units[127]      = pow(10,-12.000000);
    is_PD[127] = 0;
    nTB[127] = 0;

    // (127):  C2H5 + O <=> C2H5O
    kiv[128] = {19,3,23};
    nuv[128] = {-1,-1,1};
    kiv_qss[128] = {};
    nuv_qss[128] = {};
    // (127):  C2H5 + O <=> C2H5O
    fwd_A[128]     = 31700000000000;
    fwd_beta[128]  = 0.029999999999999999;
    fwd_Ea[128]    = -394.36000000000001;
    prefactor_units[128]  = 1.0000000000000002e-06;
    activation_units[128] = 0.50321666580471969;
    phase_units[128]      = pow(10,-12.000000);
    is_PD[128] = 0;
    nTB[128] = 0;

    // (128):  C2H5 + H <=> C2H4 + H2
    kiv[129] = {19,1,15,2};
    nuv[129] = {-1,-1,1,1};
    kiv_qss[129] = {};
    nuv_qss[129] = {};
    // (128):  C2H5 + H <=> C2H4 + H2
    fwd_A[129]     = 2000000000000;
    fwd_beta[129]  = 0;
    fwd_Ea[129]    = 0;
    prefactor_units[129]  = 1.0000000000000002e-06;
    activation_units[129] = 0.50321666580471969;
    phase_units[129]      = pow(10,-12.000000);
    is_PD[129] = 0;
    nTB[129] = 0;

    // (129):  C2H5 + O2 <=> C2H4 + HO2
    kiv[130] = {19,5,15,8};
    nuv[130] = {-1,-1,1,1};
    kiv_qss[130] = {};
    nuv_qss[130] = {};
    // (129):  C2H5 + O2 <=> C2H4 + HO2
    fwd_A[130]     = 19200000;
    fwd_beta[130]  = 1.02;
    fwd_Ea[130]    = -2033.9400000000001;
    prefactor_units[130]  = 1.0000000000000002e-06;
    activation_units[130] = 0.50321666580471969;
    phase_units[130]      = pow(10,-12.000000);
    is_PD[130] = 0;
    nTB[130] = 0;

    // (130):  C2H5 + HO2 <=> C2H6 + O2
    kiv[131] = {19,8,17,5};
    nuv[131] = {-1,-1,1,1};
    kiv_qss[131] = {};
    nuv_qss[131] = {};
    // (130):  C2H5 + HO2 <=> C2H6 + O2
    fwd_A[131]     = 300000000000;
    fwd_beta[131]  = 0;
    fwd_Ea[131]    = 0;
    prefactor_units[131]  = 1.0000000000000002e-06;
    activation_units[131] = 0.50321666580471969;
    phase_units[131]      = pow(10,-12.000000);
    is_PD[131] = 0;
    nTB[131] = 0;

    // (131):  C2H5 + CH3 <=> C2H4 + CH4
    kiv[132] = {19,10,15,13};
    nuv[132] = {-1,-1,1,1};
    kiv_qss[132] = {};
    nuv_qss[132] = {};
    // (131):  C2H5 + CH3 <=> C2H4 + CH4
    fwd_A[132]     = 11800;
    fwd_beta[132]  = 2.4500000000000002;
    fwd_Ea[132]    = 2920.6500000000001;
    prefactor_units[132]  = 1.0000000000000002e-06;
    activation_units[132] = 0.50321666580471969;
    phase_units[132]      = pow(10,-12.000000);
    is_PD[132] = 0;
    nTB[132] = 0;

    // (132):  C2H6 + SXCH2 <=> C2H5 + CH3
    kiv[133] = {17,19,10};
    nuv[133] = {-1,1,1};
    kiv_qss[133] = {4};
    nuv_qss[133] = {-1};
    // (132):  C2H6 + SXCH2 <=> C2H5 + CH3
    fwd_A[133]     = 40000000000000;
    fwd_beta[133]  = 0;
    fwd_Ea[133]    = -549.71000000000004;
    prefactor_units[133]  = 1.0000000000000002e-06;
    activation_units[133] = 0.50321666580471969;
    phase_units[133]      = pow(10,-12.000000);
    is_PD[133] = 0;
    nTB[133] = 0;

    // (133):  C2H6 + CH3 <=> C2H5 + CH4
    kiv[134] = {17,10,19,13};
    nuv[134] = {-1,-1,1,1};
    kiv_qss[134] = {};
    nuv_qss[134] = {};
    // (133):  C2H6 + CH3 <=> C2H5 + CH4
    fwd_A[134]     = 843000000000000;
    fwd_beta[134]  = 0;
    fwd_Ea[134]    = 22256.209999999999;
    prefactor_units[134]  = 1.0000000000000002e-06;
    activation_units[134] = 0.50321666580471969;
    phase_units[134]      = pow(10,-12.000000);
    is_PD[134] = 0;
    nTB[134] = 0;

    // (134):  C2H6 + O <=> C2H5 + OH
    kiv[135] = {17,3,19,4};
    nuv[135] = {-1,-1,1,1};
    kiv_qss[135] = {};
    nuv_qss[135] = {};
    // (134):  C2H6 + O <=> C2H5 + OH
    fwd_A[135]     = 31.699999999999999;
    fwd_beta[135]  = 3.7999999999999998;
    fwd_Ea[135]    = 3130.98;
    prefactor_units[135]  = 1.0000000000000002e-06;
    activation_units[135] = 0.50321666580471969;
    phase_units[135]      = pow(10,-12.000000);
    is_PD[135] = 0;
    nTB[135] = 0;

    // (135):  C2H6 (+M) <=> 2.000000 CH3 (+M)
    kiv[14] = {17,10};
    nuv[14] = {-1,2.0};
    kiv_qss[14] = {};
    nuv_qss[14] = {};
    // (135):  C2H6 (+M) <=> 2.000000 CH3 (+M)
    fwd_A[14]     = 1.8799999999999999e+50;
    fwd_beta[14]  = -9.7200000000000006;
    fwd_Ea[14]    = 107342.25999999999;
    low_A[14]     = 3.7199999999999999e+65;
    low_beta[14]  = -13.140000000000001;
    low_Ea[14]    = 101579.83;
    troe_a[14]    = 0.39000000000000001;
    troe_Tsss[14] = 100;
    troe_Ts[14]   = 1900;
    troe_Tss[14]  = 6000;
    troe_len[14]  = 4;
    prefactor_units[14]  = 1;
    activation_units[14] = 0.50321666580471969;
    phase_units[14]      = pow(10,-6.000000);
    is_PD[14] = 1;
    nTB[14] = 6;
    TB[14] = (double *) malloc(6 * sizeof(double));
    TBid[14] = (int *) malloc(6 * sizeof(int));
    TBid[14][0] = 2; TB[14][0] = 2; // H2
    TBid[14][1] = 7; TB[14][1] = 12; // H2O
    TBid[14][2] = 9; TB[14][2] = 1.75; // CO
    TBid[14][3] = 12; TB[14][3] = 3.6000000000000001; // CO2
    TBid[14][4] = 13; TB[14][4] = 2; // CH4
    TBid[14][5] = 17; TB[14][5] = 3; // C2H6

    // (136):  C2H6 + HO2 <=> C2H5 + H2O2
    kiv[136] = {17,8,19,6};
    nuv[136] = {-1,-1,1,1};
    kiv_qss[136] = {};
    nuv_qss[136] = {};
    // (136):  C2H6 + HO2 <=> C2H5 + H2O2
    fwd_A[136]     = 261;
    fwd_beta[136]  = 3.3700000000000001;
    fwd_Ea[136]    = 15913;
    prefactor_units[136]  = 1.0000000000000002e-06;
    activation_units[136] = 0.50321666580471969;
    phase_units[136]      = pow(10,-12.000000);
    is_PD[136] = 0;
    nTB[136] = 0;

    // (137):  C2H6 + H <=> C2H5 + H2
    kiv[137] = {17,1,19,2};
    nuv[137] = {-1,-1,1,1};
    kiv_qss[137] = {};
    nuv_qss[137] = {};
    // (137):  C2H6 + H <=> C2H5 + H2
    fwd_A[137]     = 170000;
    fwd_beta[137]  = 2.7000000000000002;
    fwd_Ea[137]    = 5740.9200000000001;
    prefactor_units[137]  = 1.0000000000000002e-06;
    activation_units[137] = 0.50321666580471969;
    phase_units[137]      = pow(10,-12.000000);
    is_PD[137] = 0;
    nTB[137] = 0;

    // (138):  C2H6 + OH <=> C2H5 + H2O
    kiv[138] = {17,4,19,7};
    nuv[138] = {-1,-1,1,1};
    kiv_qss[138] = {};
    nuv_qss[138] = {};
    // (138):  C2H6 + OH <=> C2H5 + H2O
    fwd_A[138]     = 1610000;
    fwd_beta[138]  = 2.2200000000000002;
    fwd_Ea[138]    = 740.91999999999996;
    prefactor_units[138]  = 1.0000000000000002e-06;
    activation_units[138] = 0.50321666580471969;
    phase_units[138]      = pow(10,-12.000000);
    is_PD[138] = 0;
    nTB[138] = 0;

    // (139):  HCCO + O2 <=> OH + 2.000000 CO
    kiv[139] = {20,5,4,9};
    nuv[139] = {-1,-1,1,2.0};
    kiv_qss[139] = {};
    nuv_qss[139] = {};
    // (139):  HCCO + O2 <=> OH + 2.000000 CO
    fwd_A[139]     = 42000000000;
    fwd_beta[139]  = 0;
    fwd_Ea[139]    = 853.25;
    prefactor_units[139]  = 1.0000000000000002e-06;
    activation_units[139] = 0.50321666580471969;
    phase_units[139]      = pow(10,-12.000000);
    is_PD[139] = 0;
    nTB[139] = 0;

    // (140):  HCCO + O <=> H + 2.000000 CO
    kiv[140] = {20,3,1,9};
    nuv[140] = {-1,-1,1,2.0};
    kiv_qss[140] = {};
    nuv_qss[140] = {};
    // (140):  HCCO + O <=> H + 2.000000 CO
    fwd_A[140]     = 100000000000000;
    fwd_beta[140]  = 0;
    fwd_Ea[140]    = 0;
    prefactor_units[140]  = 1.0000000000000002e-06;
    activation_units[140] = 0.50321666580471969;
    phase_units[140]      = pow(10,-12.000000);
    is_PD[140] = 0;
    nTB[140] = 0;

    // (141):  HCCO + CH3 <=> C2H4 + CO
    kiv[141] = {20,10,15,9};
    nuv[141] = {-1,-1,1,1};
    kiv_qss[141] = {};
    nuv_qss[141] = {};
    // (141):  HCCO + CH3 <=> C2H4 + CO
    fwd_A[141]     = 50000000000000;
    fwd_beta[141]  = 0;
    fwd_Ea[141]    = 0;
    prefactor_units[141]  = 1.0000000000000002e-06;
    activation_units[141] = 0.50321666580471969;
    phase_units[141]      = pow(10,-12.000000);
    is_PD[141] = 0;
    nTB[141] = 0;

    // (142):  HCCO + H <=> SXCH2 + CO
    kiv[142] = {20,1,9};
    nuv[142] = {-1,-1,1};
    kiv_qss[142] = {4};
    nuv_qss[142] = {1};
    // (142):  HCCO + H <=> SXCH2 + CO
    fwd_A[142]     = 100000000000000;
    fwd_beta[142]  = 0;
    fwd_Ea[142]    = 0;
    prefactor_units[142]  = 1.0000000000000002e-06;
    activation_units[142] = 0.50321666580471969;
    phase_units[142]      = pow(10,-12.000000);
    is_PD[142] = 0;
    nTB[142] = 0;

    // (143):  CH2CO + H <=> CH3 + CO
    kiv[143] = {16,1,10,9};
    nuv[143] = {-1,-1,1,1};
    kiv_qss[143] = {};
    nuv_qss[143] = {};
    // (143):  CH2CO + H <=> CH3 + CO
    fwd_A[143]     = 1500000000;
    fwd_beta[143]  = 1.3799999999999999;
    fwd_Ea[143]    = 614.24000000000001;
    prefactor_units[143]  = 1.0000000000000002e-06;
    activation_units[143] = 0.50321666580471969;
    phase_units[143]      = pow(10,-12.000000);
    is_PD[143] = 0;
    nTB[143] = 0;

    // (144):  CH2CO + TXCH2 <=> C2H4 + CO
    kiv[144] = {16,15,9};
    nuv[144] = {-1,1,1};
    kiv_qss[144] = {3};
    nuv_qss[144] = {-1};
    // (144):  CH2CO + TXCH2 <=> C2H4 + CO
    fwd_A[144]     = 1000000000000;
    fwd_beta[144]  = 0;
    fwd_Ea[144]    = 0;
    prefactor_units[144]  = 1.0000000000000002e-06;
    activation_units[144] = 0.50321666580471969;
    phase_units[144]      = pow(10,-12.000000);
    is_PD[144] = 0;
    nTB[144] = 0;

    // (145):  CH2CO + O <=> HCCO + OH
    kiv[145] = {16,3,20,4};
    nuv[145] = {-1,-1,1,1};
    kiv_qss[145] = {};
    nuv_qss[145] = {};
    // (145):  CH2CO + O <=> HCCO + OH
    fwd_A[145]     = 10000000000000;
    fwd_beta[145]  = 0;
    fwd_Ea[145]    = 7999.5200000000004;
    prefactor_units[145]  = 1.0000000000000002e-06;
    activation_units[145] = 0.50321666580471969;
    phase_units[145]      = pow(10,-12.000000);
    is_PD[145] = 0;
    nTB[145] = 0;

    // (146):  CH2CO + CH3 <=> HCCO + CH4
    kiv[146] = {16,10,20,13};
    nuv[146] = {-1,-1,1,1};
    kiv_qss[146] = {};
    nuv_qss[146] = {};
    // (146):  CH2CO + CH3 <=> HCCO + CH4
    fwd_A[146]     = 7500000000000;
    fwd_beta[146]  = 0;
    fwd_Ea[146]    = 12999.52;
    prefactor_units[146]  = 1.0000000000000002e-06;
    activation_units[146] = 0.50321666580471969;
    phase_units[146]      = pow(10,-12.000000);
    is_PD[146] = 0;
    nTB[146] = 0;

    // (147):  CH2CO + O <=> TXCH2 + CO2
    kiv[147] = {16,3,12};
    nuv[147] = {-1,-1,1};
    kiv_qss[147] = {3};
    nuv_qss[147] = {1};
    // (147):  CH2CO + O <=> TXCH2 + CO2
    fwd_A[147]     = 1750000000000;
    fwd_beta[147]  = 0;
    fwd_Ea[147]    = 1350.3800000000001;
    prefactor_units[147]  = 1.0000000000000002e-06;
    activation_units[147] = 0.50321666580471969;
    phase_units[147]      = pow(10,-12.000000);
    is_PD[147] = 0;
    nTB[147] = 0;

    // (148):  CH2CO + CH3 <=> C2H5 + CO
    kiv[148] = {16,10,19,9};
    nuv[148] = {-1,-1,1,1};
    kiv_qss[148] = {};
    nuv_qss[148] = {};
    // (148):  CH2CO + CH3 <=> C2H5 + CO
    fwd_A[148]     = 90000000000;
    fwd_beta[148]  = 0;
    fwd_Ea[148]    = 0;
    prefactor_units[148]  = 1.0000000000000002e-06;
    activation_units[148] = 0.50321666580471969;
    phase_units[148]      = pow(10,-12.000000);
    is_PD[148] = 0;
    nTB[148] = 0;

    // (149):  CH2CO + OH <=> HCCO + H2O
    kiv[149] = {16,4,20,7};
    nuv[149] = {-1,-1,1,1};
    kiv_qss[149] = {};
    nuv_qss[149] = {};
    // (149):  CH2CO + OH <=> HCCO + H2O
    fwd_A[149]     = 7500000000000;
    fwd_beta[149]  = 0;
    fwd_Ea[149]    = 2000.48;
    prefactor_units[149]  = 1.0000000000000002e-06;
    activation_units[149] = 0.50321666580471969;
    phase_units[149]      = pow(10,-12.000000);
    is_PD[149] = 0;
    nTB[149] = 0;

    // (150):  CH2CO + H <=> HCCO + H2
    kiv[150] = {16,1,20,2};
    nuv[150] = {-1,-1,1,1};
    kiv_qss[150] = {};
    nuv_qss[150] = {};
    // (150):  CH2CO + H <=> HCCO + H2
    fwd_A[150]     = 50000000000000;
    fwd_beta[150]  = 0;
    fwd_Ea[150]    = 7999.5200000000004;
    prefactor_units[150]  = 1.0000000000000002e-06;
    activation_units[150] = 0.50321666580471969;
    phase_units[150]      = pow(10,-12.000000);
    is_PD[150] = 0;
    nTB[150] = 0;

    // (151):  CH2CO + TXCH2 <=> HCCO + CH3
    kiv[151] = {16,20,10};
    nuv[151] = {-1,1,1};
    kiv_qss[151] = {3};
    nuv_qss[151] = {-1};
    // (151):  CH2CO + TXCH2 <=> HCCO + CH3
    fwd_A[151]     = 36000000000000;
    fwd_beta[151]  = 0;
    fwd_Ea[151]    = 10999.040000000001;
    prefactor_units[151]  = 1.0000000000000002e-06;
    activation_units[151] = 0.50321666580471969;
    phase_units[151]      = pow(10,-12.000000);
    is_PD[151] = 0;
    nTB[151] = 0;

    // (152):  CH2CHO + O <=> CH2O + HCO
    kiv[152] = {22,3,11};
    nuv[152] = {-1,-1,1};
    kiv_qss[152] = {2};
    nuv_qss[152] = {1};
    // (152):  CH2CHO + O <=> CH2O + HCO
    fwd_A[152]     = 31700000000000;
    fwd_beta[152]  = 0.029999999999999999;
    fwd_Ea[152]    = -394.36000000000001;
    prefactor_units[152]  = 1.0000000000000002e-06;
    activation_units[152] = 0.50321666580471969;
    phase_units[152]      = pow(10,-12.000000);
    is_PD[152] = 0;
    nTB[152] = 0;

    // (153):  CH2CHO <=> CH2CO + H
    kiv[153] = {22,16,1};
    nuv[153] = {-1,1,1};
    kiv_qss[153] = {};
    nuv_qss[153] = {};
    // (153):  CH2CHO <=> CH2CO + H
    fwd_A[153]     = 1.3199999999999999e+34;
    fwd_beta[153]  = -6.5700000000000003;
    fwd_Ea[153]    = 49457.459999999999;
    prefactor_units[153]  = 1;
    activation_units[153] = 0.50321666580471969;
    phase_units[153]      = pow(10,-6.000000);
    is_PD[153] = 0;
    nTB[153] = 0;

    // (154):  CH2CHO + OH <=> H2O + CH2CO
    kiv[154] = {22,4,7,16};
    nuv[154] = {-1,-1,1,1};
    kiv_qss[154] = {};
    nuv_qss[154] = {};
    // (154):  CH2CHO + OH <=> H2O + CH2CO
    fwd_A[154]     = 12000000000000;
    fwd_beta[154]  = 0;
    fwd_Ea[154]    = 0;
    prefactor_units[154]  = 1.0000000000000002e-06;
    activation_units[154] = 0.50321666580471969;
    phase_units[154]      = pow(10,-12.000000);
    is_PD[154] = 0;
    nTB[154] = 0;

    // (155):  CH2CHO + H <=> CH2CO + H2
    kiv[155] = {22,1,16,2};
    nuv[155] = {-1,-1,1,1};
    kiv_qss[155] = {};
    nuv_qss[155] = {};
    // (155):  CH2CHO + H <=> CH2CO + H2
    fwd_A[155]     = 11000000000000;
    fwd_beta[155]  = 0;
    fwd_Ea[155]    = 0;
    prefactor_units[155]  = 1.0000000000000002e-06;
    activation_units[155] = 0.50321666580471969;
    phase_units[155]      = pow(10,-12.000000);
    is_PD[155] = 0;
    nTB[155] = 0;

    // (156):  CH2CHO + O2 => OH + CO + CH2O
    kiv[156] = {22,5,4,9,11};
    nuv[156] = {-1,-1,1,1,1};
    kiv_qss[156] = {};
    nuv_qss[156] = {};
    // (156):  CH2CHO + O2 => OH + CO + CH2O
    fwd_A[156]     = 18100000000;
    fwd_beta[156]  = 0;
    fwd_Ea[156]    = 0;
    prefactor_units[156]  = 1.0000000000000002e-06;
    activation_units[156] = 0.50321666580471969;
    phase_units[156]      = pow(10,-12.000000);
    is_PD[156] = 0;
    nTB[156] = 0;

    // (157):  CH2CHO <=> CH3 + CO
    kiv[157] = {22,10,9};
    nuv[157] = {-1,1,1};
    kiv_qss[157] = {};
    nuv_qss[157] = {};
    // (157):  CH2CHO <=> CH3 + CO
    fwd_A[157]     = 6.5100000000000001e+34;
    fwd_beta[157]  = -6.8700000000000001;
    fwd_Ea[157]    = 47194.07;
    prefactor_units[157]  = 1;
    activation_units[157] = 0.50321666580471969;
    phase_units[157]      = pow(10,-6.000000);
    is_PD[157] = 0;
    nTB[157] = 0;

    // (158):  CH2CHO + O2 => OH + 2.000000 HCO
    kiv[158] = {22,5,4};
    nuv[158] = {-1,-1,1};
    kiv_qss[158] = {2};
    nuv_qss[158] = {2.0};
    // (158):  CH2CHO + O2 => OH + 2.000000 HCO
    fwd_A[158]     = 23500000000;
    fwd_beta[158]  = 0;
    fwd_Ea[158]    = 0;
    prefactor_units[158]  = 1.0000000000000002e-06;
    activation_units[158] = 0.50321666580471969;
    phase_units[158]      = pow(10,-12.000000);
    is_PD[158] = 0;
    nTB[158] = 0;

    // (159):  CH2CHO + H <=> CH3 + HCO
    kiv[159] = {22,1,10};
    nuv[159] = {-1,-1,1};
    kiv_qss[159] = {2};
    nuv_qss[159] = {1};
    // (159):  CH2CHO + H <=> CH3 + HCO
    fwd_A[159]     = 22000000000000;
    fwd_beta[159]  = 0;
    fwd_Ea[159]    = 0;
    prefactor_units[159]  = 1.0000000000000002e-06;
    activation_units[159] = 0.50321666580471969;
    phase_units[159]      = pow(10,-12.000000);
    is_PD[159] = 0;
    nTB[159] = 0;

    // (160):  CH3CHO + O => CH3 + CO + OH
    kiv[160] = {21,3,10,9,4};
    nuv[160] = {-1,-1,1,1,1};
    kiv_qss[160] = {};
    nuv_qss[160] = {};
    // (160):  CH3CHO + O => CH3 + CO + OH
    fwd_A[160]     = 2920000000000;
    fwd_beta[160]  = 0;
    fwd_Ea[160]    = 1809.27;
    prefactor_units[160]  = 1.0000000000000002e-06;
    activation_units[160] = 0.50321666580471969;
    phase_units[160]      = pow(10,-12.000000);
    is_PD[160] = 0;
    nTB[160] = 0;

    // (161):  CH3CHO + O2 => CH3 + CO + HO2
    kiv[161] = {21,5,10,9,8};
    nuv[161] = {-1,-1,1,1,1};
    kiv_qss[161] = {};
    nuv_qss[161] = {};
    // (161):  CH3CHO + O2 => CH3 + CO + HO2
    fwd_A[161]     = 30100000000000;
    fwd_beta[161]  = 0;
    fwd_Ea[161]    = 39149.139999999999;
    prefactor_units[161]  = 1.0000000000000002e-06;
    activation_units[161] = 0.50321666580471969;
    phase_units[161]      = pow(10,-12.000000);
    is_PD[161] = 0;
    nTB[161] = 0;

    // (162):  CH3CHO + OH => CH3 + CO + H2O
    kiv[162] = {21,4,10,9,7};
    nuv[162] = {-1,-1,1,1,1};
    kiv_qss[162] = {};
    nuv_qss[162] = {};
    // (162):  CH3CHO + OH => CH3 + CO + H2O
    fwd_A[162]     = 23400000000;
    fwd_beta[162]  = 0.72999999999999998;
    fwd_Ea[162]    = -1113.77;
    prefactor_units[162]  = 1.0000000000000002e-06;
    activation_units[162] = 0.50321666580471969;
    phase_units[162]      = pow(10,-12.000000);
    is_PD[162] = 0;
    nTB[162] = 0;

    // (163):  CH3CHO + H <=> CH2CHO + H2
    kiv[163] = {21,1,22,2};
    nuv[163] = {-1,-1,1,1};
    kiv_qss[163] = {};
    nuv_qss[163] = {};
    // (163):  CH3CHO + H <=> CH2CHO + H2
    fwd_A[163]     = 2050000000;
    fwd_beta[163]  = 1.1599999999999999;
    fwd_Ea[163]    = 2404.4000000000001;
    prefactor_units[163]  = 1.0000000000000002e-06;
    activation_units[163] = 0.50321666580471969;
    phase_units[163]      = pow(10,-12.000000);
    is_PD[163] = 0;
    nTB[163] = 0;

    // (164):  CH3CHO + H => CH3 + CO + H2
    kiv[164] = {21,1,10,9,2};
    nuv[164] = {-1,-1,1,1,1};
    kiv_qss[164] = {};
    nuv_qss[164] = {};
    // (164):  CH3CHO + H => CH3 + CO + H2
    fwd_A[164]     = 2050000000;
    fwd_beta[164]  = 1.1599999999999999;
    fwd_Ea[164]    = 2404.4000000000001;
    prefactor_units[164]  = 1.0000000000000002e-06;
    activation_units[164] = 0.50321666580471969;
    phase_units[164]      = pow(10,-12.000000);
    is_PD[164] = 0;
    nTB[164] = 0;

    // (165):  CH3CHO + O <=> CH2CHO + OH
    kiv[165] = {21,3,22,4};
    nuv[165] = {-1,-1,1,1};
    kiv_qss[165] = {};
    nuv_qss[165] = {};
    // (165):  CH3CHO + O <=> CH2CHO + OH
    fwd_A[165]     = 2920000000000;
    fwd_beta[165]  = 0;
    fwd_Ea[165]    = 1809.27;
    prefactor_units[165]  = 1.0000000000000002e-06;
    activation_units[165] = 0.50321666580471969;
    phase_units[165]      = pow(10,-12.000000);
    is_PD[165] = 0;
    nTB[165] = 0;

    // (166):  CH3CHO + CH3 => CH3 + CO + CH4
    kiv[166] = {21,10,10,9,13};
    nuv[166] = {-1,-1,1,1,1};
    kiv_qss[166] = {};
    nuv_qss[166] = {};
    // (166):  CH3CHO + CH3 => CH3 + CO + CH4
    fwd_A[166]     = 2720000;
    fwd_beta[166]  = 1.77;
    fwd_Ea[166]    = 5920.1700000000001;
    prefactor_units[166]  = 1.0000000000000002e-06;
    activation_units[166] = 0.50321666580471969;
    phase_units[166]      = pow(10,-12.000000);
    is_PD[166] = 0;
    nTB[166] = 0;

    // (167):  CH3CHO + HO2 => CH3 + CO + H2O2
    kiv[167] = {21,8,10,9,6};
    nuv[167] = {-1,-1,1,1,1};
    kiv_qss[167] = {};
    nuv_qss[167] = {};
    // (167):  CH3CHO + HO2 => CH3 + CO + H2O2
    fwd_A[167]     = 3010000000000;
    fwd_beta[167]  = 0;
    fwd_Ea[167]    = 11924;
    prefactor_units[167]  = 1.0000000000000002e-06;
    activation_units[167] = 0.50321666580471969;
    phase_units[167]      = pow(10,-12.000000);
    is_PD[167] = 0;
    nTB[167] = 0;

    // (168):  C2H5O <=> CH3 + CH2O
    kiv[168] = {23,10,11};
    nuv[168] = {-1,1,1};
    kiv_qss[168] = {};
    nuv_qss[168] = {};
    // (168):  C2H5O <=> CH3 + CH2O
    fwd_A[168]     = 1.32e+20;
    fwd_beta[168]  = -2.02;
    fwd_Ea[168]    = 20750.48;
    prefactor_units[168]  = 1;
    activation_units[168] = 0.50321666580471969;
    phase_units[168]      = pow(10,-6.000000);
    is_PD[168] = 0;
    nTB[168] = 0;

    // (169):  C2H5O <=> CH3CHO + H
    kiv[169] = {23,21,1};
    nuv[169] = {-1,1,1};
    kiv_qss[169] = {};
    nuv_qss[169] = {};
    // (169):  C2H5O <=> CH3CHO + H
    fwd_A[169]     = 5450000000000000;
    fwd_beta[169]  = -0.68999999999999995;
    fwd_Ea[169]    = 22229.919999999998;
    prefactor_units[169]  = 1;
    activation_units[169] = 0.50321666580471969;
    phase_units[169]      = pow(10,-6.000000);
    is_PD[169] = 0;
    nTB[169] = 0;

    // (170):  C2H5O + O2 <=> CH3CHO + HO2
    kiv[170] = {23,5,21,8};
    nuv[170] = {-1,-1,1,1};
    kiv_qss[170] = {};
    nuv_qss[170] = {};
    // (170):  C2H5O + O2 <=> CH3CHO + HO2
    fwd_A[170]     = 22900000000;
    fwd_beta[170]  = 0;
    fwd_Ea[170]    = 874.75999999999999;
    prefactor_units[170]  = 1.0000000000000002e-06;
    activation_units[170] = 0.50321666580471969;
    phase_units[170]      = pow(10,-12.000000);
    is_PD[170] = 0;
    nTB[170] = 0;

    // (171):  SXCH2 + N2 <=> TXCH2 + N2
    kiv[171] = {0,0};
    nuv[171] = {-1,1};
    kiv_qss[171] = {4,3};
    nuv_qss[171] = {-1,1};
    // (171):  SXCH2 + N2 <=> TXCH2 + N2
    fwd_A[171]     = 15000000000000;
    fwd_beta[171]  = 0;
    fwd_Ea[171]    = 599.89999999999998;
    prefactor_units[171]  = 1.0000000000000002e-06;
    activation_units[171] = 0.50321666580471969;
    phase_units[171]      = pow(10,-12.000000);
    is_PD[171] = 0;
    nTB[171] = 0;

}


/* Finalizes parameter database */
void CKFINALIZE()
{
  for (int i=0; i<172; ++i) {
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
    *kk = 24;
    *ii = 172;
    *nfit = -1; /*Why do you need this anyway ?  */
}


/* Returns the vector of strings of element names */
void CKSYME_STR(amrex::Vector<std::string>& ename)
{
    ename.resize(4);
    ename[0] = "N";
    ename[1] = "H";
    ename[2] = "O";
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

    /* H  */
    kname[ 1*lenkname + 0 ] = 'H';
    kname[ 1*lenkname + 1 ] = ' ';

    /* O  */
    kname[ 2*lenkname + 0 ] = 'O';
    kname[ 2*lenkname + 1 ] = ' ';

    /* C  */
    kname[ 3*lenkname + 0 ] = 'C';
    kname[ 3*lenkname + 1 ] = ' ';

}


/* Returns the vector of strings of species names */
void CKSYMS_STR(amrex::Vector<std::string>& kname)
{
    kname.resize(24);
    kname[0] = "N2";
    kname[1] = "H";
    kname[2] = "H2";
    kname[3] = "O";
    kname[4] = "OH";
    kname[5] = "O2";
    kname[6] = "H2O2";
    kname[7] = "H2O";
    kname[8] = "HO2";
    kname[9] = "CO";
    kname[10] = "CH3";
    kname[11] = "CH2O";
    kname[12] = "CO2";
    kname[13] = "CH4";
    kname[14] = "C2H2";
    kname[15] = "C2H4";
    kname[16] = "CH2CO";
    kname[17] = "C2H6";
    kname[18] = "C2H3";
    kname[19] = "C2H5";
    kname[20] = "HCCO";
    kname[21] = "CH3CHO";
    kname[22] = "CH2CHO";
    kname[23] = "C2H5O";
}


/* Returns the char strings of species names */
void CKSYMS(int * kname, int * plenkname )
{
    int i; /*Loop Counter */
    int lenkname = *plenkname;
    /*clear kname */
    for (i=0; i<lenkname*24; i++) {
        kname[i] = ' ';
    }

    /* N2  */
    kname[ 0*lenkname + 0 ] = 'N';
    kname[ 0*lenkname + 1 ] = '2';
    kname[ 0*lenkname + 2 ] = ' ';

    /* H  */
    kname[ 1*lenkname + 0 ] = 'H';
    kname[ 1*lenkname + 1 ] = ' ';

    /* H2  */
    kname[ 2*lenkname + 0 ] = 'H';
    kname[ 2*lenkname + 1 ] = '2';
    kname[ 2*lenkname + 2 ] = ' ';

    /* O  */
    kname[ 3*lenkname + 0 ] = 'O';
    kname[ 3*lenkname + 1 ] = ' ';

    /* OH  */
    kname[ 4*lenkname + 0 ] = 'O';
    kname[ 4*lenkname + 1 ] = 'H';
    kname[ 4*lenkname + 2 ] = ' ';

    /* O2  */
    kname[ 5*lenkname + 0 ] = 'O';
    kname[ 5*lenkname + 1 ] = '2';
    kname[ 5*lenkname + 2 ] = ' ';

    /* H2O2  */
    kname[ 6*lenkname + 0 ] = 'H';
    kname[ 6*lenkname + 1 ] = '2';
    kname[ 6*lenkname + 2 ] = 'O';
    kname[ 6*lenkname + 3 ] = '2';
    kname[ 6*lenkname + 4 ] = ' ';

    /* H2O  */
    kname[ 7*lenkname + 0 ] = 'H';
    kname[ 7*lenkname + 1 ] = '2';
    kname[ 7*lenkname + 2 ] = 'O';
    kname[ 7*lenkname + 3 ] = ' ';

    /* HO2  */
    kname[ 8*lenkname + 0 ] = 'H';
    kname[ 8*lenkname + 1 ] = 'O';
    kname[ 8*lenkname + 2 ] = '2';
    kname[ 8*lenkname + 3 ] = ' ';

    /* CO  */
    kname[ 9*lenkname + 0 ] = 'C';
    kname[ 9*lenkname + 1 ] = 'O';
    kname[ 9*lenkname + 2 ] = ' ';

    /* CH3  */
    kname[ 10*lenkname + 0 ] = 'C';
    kname[ 10*lenkname + 1 ] = 'H';
    kname[ 10*lenkname + 2 ] = '3';
    kname[ 10*lenkname + 3 ] = ' ';

    /* CH2O  */
    kname[ 11*lenkname + 0 ] = 'C';
    kname[ 11*lenkname + 1 ] = 'H';
    kname[ 11*lenkname + 2 ] = '2';
    kname[ 11*lenkname + 3 ] = 'O';
    kname[ 11*lenkname + 4 ] = ' ';

    /* CO2  */
    kname[ 12*lenkname + 0 ] = 'C';
    kname[ 12*lenkname + 1 ] = 'O';
    kname[ 12*lenkname + 2 ] = '2';
    kname[ 12*lenkname + 3 ] = ' ';

    /* CH4  */
    kname[ 13*lenkname + 0 ] = 'C';
    kname[ 13*lenkname + 1 ] = 'H';
    kname[ 13*lenkname + 2 ] = '4';
    kname[ 13*lenkname + 3 ] = ' ';

    /* C2H2  */
    kname[ 14*lenkname + 0 ] = 'C';
    kname[ 14*lenkname + 1 ] = '2';
    kname[ 14*lenkname + 2 ] = 'H';
    kname[ 14*lenkname + 3 ] = '2';
    kname[ 14*lenkname + 4 ] = ' ';

    /* C2H4  */
    kname[ 15*lenkname + 0 ] = 'C';
    kname[ 15*lenkname + 1 ] = '2';
    kname[ 15*lenkname + 2 ] = 'H';
    kname[ 15*lenkname + 3 ] = '4';
    kname[ 15*lenkname + 4 ] = ' ';

    /* CH2CO  */
    kname[ 16*lenkname + 0 ] = 'C';
    kname[ 16*lenkname + 1 ] = 'H';
    kname[ 16*lenkname + 2 ] = '2';
    kname[ 16*lenkname + 3 ] = 'C';
    kname[ 16*lenkname + 4 ] = 'O';
    kname[ 16*lenkname + 5 ] = ' ';

    /* C2H6  */
    kname[ 17*lenkname + 0 ] = 'C';
    kname[ 17*lenkname + 1 ] = '2';
    kname[ 17*lenkname + 2 ] = 'H';
    kname[ 17*lenkname + 3 ] = '6';
    kname[ 17*lenkname + 4 ] = ' ';

    /* C2H3  */
    kname[ 18*lenkname + 0 ] = 'C';
    kname[ 18*lenkname + 1 ] = '2';
    kname[ 18*lenkname + 2 ] = 'H';
    kname[ 18*lenkname + 3 ] = '3';
    kname[ 18*lenkname + 4 ] = ' ';

    /* C2H5  */
    kname[ 19*lenkname + 0 ] = 'C';
    kname[ 19*lenkname + 1 ] = '2';
    kname[ 19*lenkname + 2 ] = 'H';
    kname[ 19*lenkname + 3 ] = '5';
    kname[ 19*lenkname + 4 ] = ' ';

    /* HCCO  */
    kname[ 20*lenkname + 0 ] = 'H';
    kname[ 20*lenkname + 1 ] = 'C';
    kname[ 20*lenkname + 2 ] = 'C';
    kname[ 20*lenkname + 3 ] = 'O';
    kname[ 20*lenkname + 4 ] = ' ';

    /* CH3CHO  */
    kname[ 21*lenkname + 0 ] = 'C';
    kname[ 21*lenkname + 1 ] = 'H';
    kname[ 21*lenkname + 2 ] = '3';
    kname[ 21*lenkname + 3 ] = 'C';
    kname[ 21*lenkname + 4 ] = 'H';
    kname[ 21*lenkname + 5 ] = 'O';
    kname[ 21*lenkname + 6 ] = ' ';

    /* CH2CHO  */
    kname[ 22*lenkname + 0 ] = 'C';
    kname[ 22*lenkname + 1 ] = 'H';
    kname[ 22*lenkname + 2 ] = '2';
    kname[ 22*lenkname + 3 ] = 'C';
    kname[ 22*lenkname + 4 ] = 'H';
    kname[ 22*lenkname + 5 ] = 'O';
    kname[ 22*lenkname + 6 ] = ' ';

    /* C2H5O  */
    kname[ 23*lenkname + 0 ] = 'C';
    kname[ 23*lenkname + 1 ] = '2';
    kname[ 23*lenkname + 2 ] = 'H';
    kname[ 23*lenkname + 3 ] = '5';
    kname[ 23*lenkname + 4 ] = 'O';
    kname[ 23*lenkname + 5 ] = ' ';

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
    XW += x[1]*molecular_weights[1]; /*H */
    XW += x[2]*molecular_weights[2]; /*H2 */
    XW += x[3]*molecular_weights[3]; /*O */
    XW += x[4]*molecular_weights[4]; /*OH */
    XW += x[5]*molecular_weights[5]; /*O2 */
    XW += x[6]*molecular_weights[6]; /*H2O2 */
    XW += x[7]*molecular_weights[7]; /*H2O */
    XW += x[8]*molecular_weights[8]; /*HO2 */
    XW += x[9]*molecular_weights[9]; /*CO */
    XW += x[10]*molecular_weights[10]; /*CH3 */
    XW += x[11]*molecular_weights[11]; /*CH2O */
    XW += x[12]*molecular_weights[12]; /*CO2 */
    XW += x[13]*molecular_weights[13]; /*CH4 */
    XW += x[14]*molecular_weights[14]; /*C2H2 */
    XW += x[15]*molecular_weights[15]; /*C2H4 */
    XW += x[16]*molecular_weights[16]; /*CH2CO */
    XW += x[17]*molecular_weights[17]; /*C2H6 */
    XW += x[18]*molecular_weights[18]; /*C2H3 */
    XW += x[19]*molecular_weights[19]; /*C2H5 */
    XW += x[20]*molecular_weights[20]; /*HCCO */
    XW += x[21]*molecular_weights[21]; /*CH3CHO */
    XW += x[22]*molecular_weights[22]; /*CH2CHO */
    XW += x[23]*molecular_weights[23]; /*C2H5O */
    *P = *rho * 8.31446e+07 * (*T) / XW; /*P = rho*R*T/W */

    return;
}


/*Compute P = rhoRT/W(y) */
AMREX_GPU_HOST_DEVICE void CKPY(double *  rho, double *  T, double *  y,  double *  P)
{
    double YOW = 0;/* for computing mean MW */
    YOW += y[0]*imw[0]; /*N2 */
    YOW += y[1]*imw[1]; /*H */
    YOW += y[2]*imw[2]; /*H2 */
    YOW += y[3]*imw[3]; /*O */
    YOW += y[4]*imw[4]; /*OH */
    YOW += y[5]*imw[5]; /*O2 */
    YOW += y[6]*imw[6]; /*H2O2 */
    YOW += y[7]*imw[7]; /*H2O */
    YOW += y[8]*imw[8]; /*HO2 */
    YOW += y[9]*imw[9]; /*CO */
    YOW += y[10]*imw[10]; /*CH3 */
    YOW += y[11]*imw[11]; /*CH2O */
    YOW += y[12]*imw[12]; /*CO2 */
    YOW += y[13]*imw[13]; /*CH4 */
    YOW += y[14]*imw[14]; /*C2H2 */
    YOW += y[15]*imw[15]; /*C2H4 */
    YOW += y[16]*imw[16]; /*CH2CO */
    YOW += y[17]*imw[17]; /*C2H6 */
    YOW += y[18]*imw[18]; /*C2H3 */
    YOW += y[19]*imw[19]; /*C2H5 */
    YOW += y[20]*imw[20]; /*HCCO */
    YOW += y[21]*imw[21]; /*CH3CHO */
    YOW += y[22]*imw[22]; /*CH2CHO */
    YOW += y[23]*imw[23]; /*C2H5O */
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
    W += c[1]*1.007970; /*H */
    W += c[2]*2.015940; /*H2 */
    W += c[3]*15.999400; /*O */
    W += c[4]*17.007370; /*OH */
    W += c[5]*31.998800; /*O2 */
    W += c[6]*34.014740; /*H2O2 */
    W += c[7]*18.015340; /*H2O */
    W += c[8]*33.006770; /*HO2 */
    W += c[9]*28.010550; /*CO */
    W += c[10]*15.035060; /*CH3 */
    W += c[11]*30.026490; /*CH2O */
    W += c[12]*44.009950; /*CO2 */
    W += c[13]*16.043030; /*CH4 */
    W += c[14]*26.038240; /*C2H2 */
    W += c[15]*28.054180; /*C2H4 */
    W += c[16]*42.037640; /*CH2CO */
    W += c[17]*30.070120; /*C2H6 */
    W += c[18]*27.046210; /*C2H3 */
    W += c[19]*29.062150; /*C2H5 */
    W += c[20]*41.029670; /*HCCO */
    W += c[21]*44.053580; /*CH3CHO */
    W += c[22]*43.045610; /*CH2CHO */
    W += c[23]*45.061550; /*C2H5O */

    for (id = 0; id < 24; ++id) {
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
    XW += x[1]*1.007970; /*H */
    XW += x[2]*2.015940; /*H2 */
    XW += x[3]*15.999400; /*O */
    XW += x[4]*17.007370; /*OH */
    XW += x[5]*31.998800; /*O2 */
    XW += x[6]*34.014740; /*H2O2 */
    XW += x[7]*18.015340; /*H2O */
    XW += x[8]*33.006770; /*HO2 */
    XW += x[9]*28.010550; /*CO */
    XW += x[10]*15.035060; /*CH3 */
    XW += x[11]*30.026490; /*CH2O */
    XW += x[12]*44.009950; /*CO2 */
    XW += x[13]*16.043030; /*CH4 */
    XW += x[14]*26.038240; /*C2H2 */
    XW += x[15]*28.054180; /*C2H4 */
    XW += x[16]*42.037640; /*CH2CO */
    XW += x[17]*30.070120; /*C2H6 */
    XW += x[18]*27.046210; /*C2H3 */
    XW += x[19]*29.062150; /*C2H5 */
    XW += x[20]*41.029670; /*HCCO */
    XW += x[21]*44.053580; /*CH3CHO */
    XW += x[22]*43.045610; /*CH2CHO */
    XW += x[23]*45.061550; /*C2H5O */
    *rho = *P * XW / (8.31446e+07 * (*T)); /*rho = P*W/(R*T) */

    return;
}


/*Compute rho = P*W(y)/RT */
AMREX_GPU_HOST_DEVICE void CKRHOY(double *  P, double *  T, double *  y,  double *  rho)
{
    double YOW = 0;
    double tmp[24];

    for (int i = 0; i < 24; i++)
    {
        tmp[i] = y[i]*imw[i];
    }
    for (int i = 0; i < 24; i++)
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
    W += c[1]*1.007970; /*H */
    W += c[2]*2.015940; /*H2 */
    W += c[3]*15.999400; /*O */
    W += c[4]*17.007370; /*OH */
    W += c[5]*31.998800; /*O2 */
    W += c[6]*34.014740; /*H2O2 */
    W += c[7]*18.015340; /*H2O */
    W += c[8]*33.006770; /*HO2 */
    W += c[9]*28.010550; /*CO */
    W += c[10]*15.035060; /*CH3 */
    W += c[11]*30.026490; /*CH2O */
    W += c[12]*44.009950; /*CO2 */
    W += c[13]*16.043030; /*CH4 */
    W += c[14]*26.038240; /*C2H2 */
    W += c[15]*28.054180; /*C2H4 */
    W += c[16]*42.037640; /*CH2CO */
    W += c[17]*30.070120; /*C2H6 */
    W += c[18]*27.046210; /*C2H3 */
    W += c[19]*29.062150; /*C2H5 */
    W += c[20]*41.029670; /*HCCO */
    W += c[21]*44.053580; /*CH3CHO */
    W += c[22]*43.045610; /*CH2CHO */
    W += c[23]*45.061550; /*C2H5O */

    for (id = 0; id < 24; ++id) {
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
    double tmp[24];

    for (int i = 0; i < 24; i++)
    {
        tmp[i] = y[i]*imw[i];
    }
    for (int i = 0; i < 24; i++)
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
    XW += x[1]*molecular_weights[1]; /*H */
    XW += x[2]*molecular_weights[2]; /*H2 */
    XW += x[3]*molecular_weights[3]; /*O */
    XW += x[4]*molecular_weights[4]; /*OH */
    XW += x[5]*molecular_weights[5]; /*O2 */
    XW += x[6]*molecular_weights[6]; /*H2O2 */
    XW += x[7]*molecular_weights[7]; /*H2O */
    XW += x[8]*molecular_weights[8]; /*HO2 */
    XW += x[9]*molecular_weights[9]; /*CO */
    XW += x[10]*molecular_weights[10]; /*CH3 */
    XW += x[11]*molecular_weights[11]; /*CH2O */
    XW += x[12]*molecular_weights[12]; /*CO2 */
    XW += x[13]*molecular_weights[13]; /*CH4 */
    XW += x[14]*molecular_weights[14]; /*C2H2 */
    XW += x[15]*molecular_weights[15]; /*C2H4 */
    XW += x[16]*molecular_weights[16]; /*CH2CO */
    XW += x[17]*molecular_weights[17]; /*C2H6 */
    XW += x[18]*molecular_weights[18]; /*C2H3 */
    XW += x[19]*molecular_weights[19]; /*C2H5 */
    XW += x[20]*molecular_weights[20]; /*HCCO */
    XW += x[21]*molecular_weights[21]; /*CH3CHO */
    XW += x[22]*molecular_weights[22]; /*CH2CHO */
    XW += x[23]*molecular_weights[23]; /*C2H5O */
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
    W += c[1]*molecular_weights[1]; /*H */
    W += c[2]*molecular_weights[2]; /*H2 */
    W += c[3]*molecular_weights[3]; /*O */
    W += c[4]*molecular_weights[4]; /*OH */
    W += c[5]*molecular_weights[5]; /*O2 */
    W += c[6]*molecular_weights[6]; /*H2O2 */
    W += c[7]*molecular_weights[7]; /*H2O */
    W += c[8]*molecular_weights[8]; /*HO2 */
    W += c[9]*molecular_weights[9]; /*CO */
    W += c[10]*molecular_weights[10]; /*CH3 */
    W += c[11]*molecular_weights[11]; /*CH2O */
    W += c[12]*molecular_weights[12]; /*CO2 */
    W += c[13]*molecular_weights[13]; /*CH4 */
    W += c[14]*molecular_weights[14]; /*C2H2 */
    W += c[15]*molecular_weights[15]; /*C2H4 */
    W += c[16]*molecular_weights[16]; /*CH2CO */
    W += c[17]*molecular_weights[17]; /*C2H6 */
    W += c[18]*molecular_weights[18]; /*C2H3 */
    W += c[19]*molecular_weights[19]; /*C2H5 */
    W += c[20]*molecular_weights[20]; /*HCCO */
    W += c[21]*molecular_weights[21]; /*CH3CHO */
    W += c[22]*molecular_weights[22]; /*CH2CHO */
    W += c[23]*molecular_weights[23]; /*C2H5O */

    for (id = 0; id < 24; ++id) {
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
    double tmp[24];

    for (int i = 0; i < 24; i++)
    {
        tmp[i] = y[i]*imw[i];
    }
    for (int i = 0; i < 24; i++)
    {
        YOW += tmp[i];
    }

    double YOWINV = 1.0/YOW;

    for (int i = 0; i < 24; i++)
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
    for (int i = 0; i < 24; i++)
    {
        c[i] = y[i]*imw[i];
    }
    for (int i = 0; i < 24; i++)
    {
        YOW += c[i];
    }

    /*PW/RT (see Eq. 7) */
    PWORT = (*P)/(YOW * 8.31446e+07 * (*T)); 
    /*Now compute conversion */

    for (int i = 0; i < 24; i++)
    {
        c[i] = PWORT * y[i] * imw[i];
    }
    return;
}


/*convert y[species] (mass fracs) to c[species] (molar conc) */
AMREX_GPU_HOST_DEVICE void CKYTCR(double *  rho, double *  T, double *  y,  double *  c)
{
    for (int i = 0; i < 24; i++)
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
    XW += x[1]*1.007970; /*H */
    XW += x[2]*2.015940; /*H2 */
    XW += x[3]*15.999400; /*O */
    XW += x[4]*17.007370; /*OH */
    XW += x[5]*31.998800; /*O2 */
    XW += x[6]*34.014740; /*H2O2 */
    XW += x[7]*18.015340; /*H2O */
    XW += x[8]*33.006770; /*HO2 */
    XW += x[9]*28.010550; /*CO */
    XW += x[10]*15.035060; /*CH3 */
    XW += x[11]*30.026490; /*CH2O */
    XW += x[12]*44.009950; /*CO2 */
    XW += x[13]*16.043030; /*CH4 */
    XW += x[14]*26.038240; /*C2H2 */
    XW += x[15]*28.054180; /*C2H4 */
    XW += x[16]*42.037640; /*CH2CO */
    XW += x[17]*30.070120; /*C2H6 */
    XW += x[18]*27.046210; /*C2H3 */
    XW += x[19]*29.062150; /*C2H5 */
    XW += x[20]*41.029670; /*HCCO */
    XW += x[21]*44.053580; /*CH3CHO */
    XW += x[22]*43.045610; /*CH2CHO */
    XW += x[23]*45.061550; /*C2H5O */

    /*Now compute conversion */
    double XWinv = 1.0/XW;
    y[0] = x[0]*28.013400*XWinv; 
    y[1] = x[1]*1.007970*XWinv; 
    y[2] = x[2]*2.015940*XWinv; 
    y[3] = x[3]*15.999400*XWinv; 
    y[4] = x[4]*17.007370*XWinv; 
    y[5] = x[5]*31.998800*XWinv; 
    y[6] = x[6]*34.014740*XWinv; 
    y[7] = x[7]*18.015340*XWinv; 
    y[8] = x[8]*33.006770*XWinv; 
    y[9] = x[9]*28.010550*XWinv; 
    y[10] = x[10]*15.035060*XWinv; 
    y[11] = x[11]*30.026490*XWinv; 
    y[12] = x[12]*44.009950*XWinv; 
    y[13] = x[13]*16.043030*XWinv; 
    y[14] = x[14]*26.038240*XWinv; 
    y[15] = x[15]*28.054180*XWinv; 
    y[16] = x[16]*42.037640*XWinv; 
    y[17] = x[17]*30.070120*XWinv; 
    y[18] = x[18]*27.046210*XWinv; 
    y[19] = x[19]*29.062150*XWinv; 
    y[20] = x[20]*41.029670*XWinv; 
    y[21] = x[21]*44.053580*XWinv; 
    y[22] = x[22]*43.045610*XWinv; 
    y[23] = x[23]*45.061550*XWinv; 

    return;
}


/*convert x[species] (mole fracs) to c[species] (molar conc) */
void CKXTCP(double *  P, double *  T, double *  x,  double *  c)
{
    int id; /*loop counter */
    double PORT = (*P)/(8.31446e+07 * (*T)); /*P/RT */

    /*Compute conversion, see Eq 10 */
    for (id = 0; id < 24; ++id) {
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
    XW += x[1]*1.007970; /*H */
    XW += x[2]*2.015940; /*H2 */
    XW += x[3]*15.999400; /*O */
    XW += x[4]*17.007370; /*OH */
    XW += x[5]*31.998800; /*O2 */
    XW += x[6]*34.014740; /*H2O2 */
    XW += x[7]*18.015340; /*H2O */
    XW += x[8]*33.006770; /*HO2 */
    XW += x[9]*28.010550; /*CO */
    XW += x[10]*15.035060; /*CH3 */
    XW += x[11]*30.026490; /*CH2O */
    XW += x[12]*44.009950; /*CO2 */
    XW += x[13]*16.043030; /*CH4 */
    XW += x[14]*26.038240; /*C2H2 */
    XW += x[15]*28.054180; /*C2H4 */
    XW += x[16]*42.037640; /*CH2CO */
    XW += x[17]*30.070120; /*C2H6 */
    XW += x[18]*27.046210; /*C2H3 */
    XW += x[19]*29.062150; /*C2H5 */
    XW += x[20]*41.029670; /*HCCO */
    XW += x[21]*44.053580; /*CH3CHO */
    XW += x[22]*43.045610; /*CH2CHO */
    XW += x[23]*45.061550; /*C2H5O */
    ROW = (*rho) / XW;

    /*Compute conversion, see Eq 11 */
    for (id = 0; id < 24; ++id) {
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
    for (id = 0; id < 24; ++id) {
        sumC += c[id];
    }

    /* See Eq 13  */
    double sumCinv = 1.0/sumC;
    for (id = 0; id < 24; ++id) {
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
    CW += c[1]*1.007970; /*H */
    CW += c[2]*2.015940; /*H2 */
    CW += c[3]*15.999400; /*O */
    CW += c[4]*17.007370; /*OH */
    CW += c[5]*31.998800; /*O2 */
    CW += c[6]*34.014740; /*H2O2 */
    CW += c[7]*18.015340; /*H2O */
    CW += c[8]*33.006770; /*HO2 */
    CW += c[9]*28.010550; /*CO */
    CW += c[10]*15.035060; /*CH3 */
    CW += c[11]*30.026490; /*CH2O */
    CW += c[12]*44.009950; /*CO2 */
    CW += c[13]*16.043030; /*CH4 */
    CW += c[14]*26.038240; /*C2H2 */
    CW += c[15]*28.054180; /*C2H4 */
    CW += c[16]*42.037640; /*CH2CO */
    CW += c[17]*30.070120; /*C2H6 */
    CW += c[18]*27.046210; /*C2H3 */
    CW += c[19]*29.062150; /*C2H5 */
    CW += c[20]*41.029670; /*HCCO */
    CW += c[21]*44.053580; /*CH3CHO */
    CW += c[22]*43.045610; /*CH2CHO */
    CW += c[23]*45.061550; /*C2H5O */
    /*Now compute conversion */
    double CWinv = 1.0/CW;
    y[0] = c[0]*28.013400*CWinv; 
    y[1] = c[1]*1.007970*CWinv; 
    y[2] = c[2]*2.015940*CWinv; 
    y[3] = c[3]*15.999400*CWinv; 
    y[4] = c[4]*17.007370*CWinv; 
    y[5] = c[5]*31.998800*CWinv; 
    y[6] = c[6]*34.014740*CWinv; 
    y[7] = c[7]*18.015340*CWinv; 
    y[8] = c[8]*33.006770*CWinv; 
    y[9] = c[9]*28.010550*CWinv; 
    y[10] = c[10]*15.035060*CWinv; 
    y[11] = c[11]*30.026490*CWinv; 
    y[12] = c[12]*44.009950*CWinv; 
    y[13] = c[13]*16.043030*CWinv; 
    y[14] = c[14]*26.038240*CWinv; 
    y[15] = c[15]*28.054180*CWinv; 
    y[16] = c[16]*42.037640*CWinv; 
    y[17] = c[17]*30.070120*CWinv; 
    y[18] = c[18]*27.046210*CWinv; 
    y[19] = c[19]*29.062150*CWinv; 
    y[20] = c[20]*41.029670*CWinv; 
    y[21] = c[21]*44.053580*CWinv; 
    y[22] = c[22]*43.045610*CWinv; 
    y[23] = c[23]*45.061550*CWinv; 

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
    cvms[1] *= 8.248720317224957e+07; /*H */
    cvms[2] *= 4.124360158612479e+07; /*H2 */
    cvms[3] *= 5.196734013871295e+06; /*O */
    cvms[4] *= 4.888740950630956e+06; /*OH */
    cvms[5] *= 2.598367006935648e+06; /*O2 */
    cvms[6] *= 2.444370475315478e+06; /*H2O2 */
    cvms[7] *= 4.615212712140454e+06; /*H2O */
    cvms[8] *= 2.519017346487778e+06; /*HO2 */
    cvms[9] *= 2.968332509769797e+06; /*CO */
    cvms[10] *= 5.530049509714786e+06; /*CH3 */
    cvms[11] *= 2.769042474879095e+06; /*CH2O */
    cvms[12] *= 1.889223372931176e+06; /*CO2 */
    cvms[13] *= 5.182601178301878e+06; /*CH4 */
    cvms[14] *= 3.193173815954242e+06; /*C2H2 */
    cvms[15] *= 2.963716144315478e+06; /*C2H4 */
    cvms[16] *= 1.977861416138784e+06; /*CH2CO */
    cvms[17] *= 2.765024754857393e+06; /*C2H6 */
    cvms[18] *= 3.074169215632519e+06; /*C2H3 */
    cvms[19] *= 2.860924817383862e+06; /*C2H5 */
    cvms[20] *= 2.026451252996488e+06; /*HCCO */
    cvms[21] *= 1.887352314648035e+06; /*CH3CHO */
    cvms[22] *= 1.931547170118681e+06; /*CH2CHO */
    cvms[23] *= 1.845134625451907e+06; /*C2H5O */
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
    cpms[1] *= 8.248720317224957e+07; /*H */
    cpms[2] *= 4.124360158612479e+07; /*H2 */
    cpms[3] *= 5.196734013871295e+06; /*O */
    cpms[4] *= 4.888740950630956e+06; /*OH */
    cpms[5] *= 2.598367006935648e+06; /*O2 */
    cpms[6] *= 2.444370475315478e+06; /*H2O2 */
    cpms[7] *= 4.615212712140454e+06; /*H2O */
    cpms[8] *= 2.519017346487778e+06; /*HO2 */
    cpms[9] *= 2.968332509769797e+06; /*CO */
    cpms[10] *= 5.530049509714786e+06; /*CH3 */
    cpms[11] *= 2.769042474879095e+06; /*CH2O */
    cpms[12] *= 1.889223372931176e+06; /*CO2 */
    cpms[13] *= 5.182601178301878e+06; /*CH4 */
    cpms[14] *= 3.193173815954242e+06; /*C2H2 */
    cpms[15] *= 2.963716144315478e+06; /*C2H4 */
    cpms[16] *= 1.977861416138784e+06; /*CH2CO */
    cpms[17] *= 2.765024754857393e+06; /*C2H6 */
    cpms[18] *= 3.074169215632519e+06; /*C2H3 */
    cpms[19] *= 2.860924817383862e+06; /*C2H5 */
    cpms[20] *= 2.026451252996488e+06; /*HCCO */
    cpms[21] *= 1.887352314648035e+06; /*CH3CHO */
    cpms[22] *= 1.931547170118681e+06; /*CH2CHO */
    cpms[23] *= 1.845134625451907e+06; /*C2H5O */
}


/*Returns internal energy in mass units (Eq 30.) */
AMREX_GPU_HOST_DEVICE void CKUMS(double *  T,  double *  ums)
{
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double RT = 8.31446e+07*tT; /*R*T */
    speciesInternalEnergy(ums, tc);
    for (int i = 0; i < 24; i++)
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
    for (int i = 0; i < 24; i++)
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
    for (int i = 0; i < 24; i++)
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
    for (int i = 0; i < 24; i++)
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
    sms[1] *= 8.248720317224957e+07; /*H */
    sms[2] *= 4.124360158612479e+07; /*H2 */
    sms[3] *= 5.196734013871295e+06; /*O */
    sms[4] *= 4.888740950630956e+06; /*OH */
    sms[5] *= 2.598367006935648e+06; /*O2 */
    sms[6] *= 2.444370475315478e+06; /*H2O2 */
    sms[7] *= 4.615212712140454e+06; /*H2O */
    sms[8] *= 2.519017346487778e+06; /*HO2 */
    sms[9] *= 2.968332509769797e+06; /*CO */
    sms[10] *= 5.530049509714786e+06; /*CH3 */
    sms[11] *= 2.769042474879095e+06; /*CH2O */
    sms[12] *= 1.889223372931176e+06; /*CO2 */
    sms[13] *= 5.182601178301878e+06; /*CH4 */
    sms[14] *= 3.193173815954242e+06; /*C2H2 */
    sms[15] *= 2.963716144315478e+06; /*C2H4 */
    sms[16] *= 1.977861416138784e+06; /*CH2CO */
    sms[17] *= 2.765024754857393e+06; /*C2H6 */
    sms[18] *= 3.074169215632519e+06; /*C2H3 */
    sms[19] *= 2.860924817383862e+06; /*C2H5 */
    sms[20] *= 2.026451252996488e+06; /*HCCO */
    sms[21] *= 1.887352314648035e+06; /*CH3CHO */
    sms[22] *= 1.931547170118681e+06; /*CH2CHO */
    sms[23] *= 1.845134625451907e+06; /*C2H5O */
}


/*Returns the mean specific heat at CP (Eq. 33) */
void CKCPBL(double *  T, double *  x,  double *  cpbl)
{
    int id; /*loop counter */
    double result = 0; 
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double cpor[24]; /* temporary storage */
    cp_R(cpor, tc);

    /*perform dot product */
    for (id = 0; id < 24; ++id) {
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
    double cpor[24], tresult[24]; /* temporary storage */
    cp_R(cpor, tc);
    for (int i = 0; i < 24; i++)
    {
        tresult[i] = cpor[i]*y[i]*imw[i];

    }
    for (int i = 0; i < 24; i++)
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
    double cvor[24]; /* temporary storage */
    cv_R(cvor, tc);

    /*perform dot product */
    for (id = 0; id < 24; ++id) {
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
    double cvor[24]; /* temporary storage */
    cv_R(cvor, tc);
    /*multiply by y/molecularweight */
    result += cvor[0]*y[0]*imw[0]; /*N2 */
    result += cvor[1]*y[1]*imw[1]; /*H */
    result += cvor[2]*y[2]*imw[2]; /*H2 */
    result += cvor[3]*y[3]*imw[3]; /*O */
    result += cvor[4]*y[4]*imw[4]; /*OH */
    result += cvor[5]*y[5]*imw[5]; /*O2 */
    result += cvor[6]*y[6]*imw[6]; /*H2O2 */
    result += cvor[7]*y[7]*imw[7]; /*H2O */
    result += cvor[8]*y[8]*imw[8]; /*HO2 */
    result += cvor[9]*y[9]*imw[9]; /*CO */
    result += cvor[10]*y[10]*imw[10]; /*CH3 */
    result += cvor[11]*y[11]*imw[11]; /*CH2O */
    result += cvor[12]*y[12]*imw[12]; /*CO2 */
    result += cvor[13]*y[13]*imw[13]; /*CH4 */
    result += cvor[14]*y[14]*imw[14]; /*C2H2 */
    result += cvor[15]*y[15]*imw[15]; /*C2H4 */
    result += cvor[16]*y[16]*imw[16]; /*CH2CO */
    result += cvor[17]*y[17]*imw[17]; /*C2H6 */
    result += cvor[18]*y[18]*imw[18]; /*C2H3 */
    result += cvor[19]*y[19]*imw[19]; /*C2H5 */
    result += cvor[20]*y[20]*imw[20]; /*HCCO */
    result += cvor[21]*y[21]*imw[21]; /*CH3CHO */
    result += cvor[22]*y[22]*imw[22]; /*CH2CHO */
    result += cvor[23]*y[23]*imw[23]; /*C2H5O */

    *cvbs = result * 8.31446e+07;
}


/*Returns the mean enthalpy of the mixture in molar units */
void CKHBML(double *  T, double *  x,  double *  hbml)
{
    int id; /*loop counter */
    double result = 0; 
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double hml[24]; /* temporary storage */
    double RT = 8.31446e+07*tT; /*R*T */
    speciesEnthalpy(hml, tc);

    /*perform dot product */
    for (id = 0; id < 24; ++id) {
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
    double hml[24], tmp[24]; /* temporary storage */
    double RT = 8.31446e+07*tT; /*R*T */
    speciesEnthalpy(hml, tc);
    int id;
    for (id = 0; id < 24; ++id) {
        tmp[id] = y[id]*hml[id]*imw[id];
    }
    for (id = 0; id < 24; ++id) {
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
    double uml[24]; /* temporary energy array */
    double RT = 8.31446e+07*tT; /*R*T */
    speciesInternalEnergy(uml, tc);

    /*perform dot product */
    for (id = 0; id < 24; ++id) {
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
    double ums[24]; /* temporary energy array */
    double RT = 8.31446e+07*tT; /*R*T */
    speciesInternalEnergy(ums, tc);
    /*perform dot product + scaling by wt */
    result += y[0]*ums[0]*imw[0]; /*N2 */
    result += y[1]*ums[1]*imw[1]; /*H */
    result += y[2]*ums[2]*imw[2]; /*H2 */
    result += y[3]*ums[3]*imw[3]; /*O */
    result += y[4]*ums[4]*imw[4]; /*OH */
    result += y[5]*ums[5]*imw[5]; /*O2 */
    result += y[6]*ums[6]*imw[6]; /*H2O2 */
    result += y[7]*ums[7]*imw[7]; /*H2O */
    result += y[8]*ums[8]*imw[8]; /*HO2 */
    result += y[9]*ums[9]*imw[9]; /*CO */
    result += y[10]*ums[10]*imw[10]; /*CH3 */
    result += y[11]*ums[11]*imw[11]; /*CH2O */
    result += y[12]*ums[12]*imw[12]; /*CO2 */
    result += y[13]*ums[13]*imw[13]; /*CH4 */
    result += y[14]*ums[14]*imw[14]; /*C2H2 */
    result += y[15]*ums[15]*imw[15]; /*C2H4 */
    result += y[16]*ums[16]*imw[16]; /*CH2CO */
    result += y[17]*ums[17]*imw[17]; /*C2H6 */
    result += y[18]*ums[18]*imw[18]; /*C2H3 */
    result += y[19]*ums[19]*imw[19]; /*C2H5 */
    result += y[20]*ums[20]*imw[20]; /*HCCO */
    result += y[21]*ums[21]*imw[21]; /*CH3CHO */
    result += y[22]*ums[22]*imw[22]; /*CH2CHO */
    result += y[23]*ums[23]*imw[23]; /*C2H5O */

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
    double sor[24]; /* temporary storage */
    speciesEntropy(sor, tc);

    /*Compute Eq 42 */
    for (id = 0; id < 24; ++id) {
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
    double sor[24]; /* temporary storage */
    double x[24]; /* need a ytx conversion */
    double YOW = 0; /*See Eq 4, 6 in CK Manual */
    /*Compute inverse of mean molecular wt first */
    YOW += y[0]*imw[0]; /*N2 */
    YOW += y[1]*imw[1]; /*H */
    YOW += y[2]*imw[2]; /*H2 */
    YOW += y[3]*imw[3]; /*O */
    YOW += y[4]*imw[4]; /*OH */
    YOW += y[5]*imw[5]; /*O2 */
    YOW += y[6]*imw[6]; /*H2O2 */
    YOW += y[7]*imw[7]; /*H2O */
    YOW += y[8]*imw[8]; /*HO2 */
    YOW += y[9]*imw[9]; /*CO */
    YOW += y[10]*imw[10]; /*CH3 */
    YOW += y[11]*imw[11]; /*CH2O */
    YOW += y[12]*imw[12]; /*CO2 */
    YOW += y[13]*imw[13]; /*CH4 */
    YOW += y[14]*imw[14]; /*C2H2 */
    YOW += y[15]*imw[15]; /*C2H4 */
    YOW += y[16]*imw[16]; /*CH2CO */
    YOW += y[17]*imw[17]; /*C2H6 */
    YOW += y[18]*imw[18]; /*C2H3 */
    YOW += y[19]*imw[19]; /*C2H5 */
    YOW += y[20]*imw[20]; /*HCCO */
    YOW += y[21]*imw[21]; /*CH3CHO */
    YOW += y[22]*imw[22]; /*CH2CHO */
    YOW += y[23]*imw[23]; /*C2H5O */
    /*Now compute y to x conversion */
    x[0] = y[0]/(28.013400*YOW); 
    x[1] = y[1]/(1.007970*YOW); 
    x[2] = y[2]/(2.015940*YOW); 
    x[3] = y[3]/(15.999400*YOW); 
    x[4] = y[4]/(17.007370*YOW); 
    x[5] = y[5]/(31.998800*YOW); 
    x[6] = y[6]/(34.014740*YOW); 
    x[7] = y[7]/(18.015340*YOW); 
    x[8] = y[8]/(33.006770*YOW); 
    x[9] = y[9]/(28.010550*YOW); 
    x[10] = y[10]/(15.035060*YOW); 
    x[11] = y[11]/(30.026490*YOW); 
    x[12] = y[12]/(44.009950*YOW); 
    x[13] = y[13]/(16.043030*YOW); 
    x[14] = y[14]/(26.038240*YOW); 
    x[15] = y[15]/(28.054180*YOW); 
    x[16] = y[16]/(42.037640*YOW); 
    x[17] = y[17]/(30.070120*YOW); 
    x[18] = y[18]/(27.046210*YOW); 
    x[19] = y[19]/(29.062150*YOW); 
    x[20] = y[20]/(41.029670*YOW); 
    x[21] = y[21]/(44.053580*YOW); 
    x[22] = y[22]/(43.045610*YOW); 
    x[23] = y[23]/(45.061550*YOW); 
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
    double gort[24]; /* temporary storage */
    /*Compute g/RT */
    gibbs(gort, tc);

    /*Compute Eq 44 */
    for (id = 0; id < 24; ++id) {
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
    double gort[24]; /* temporary storage */
    double x[24]; /* need a ytx conversion */
    double YOW = 0; /*To hold 1/molecularweight */
    /*Compute inverse of mean molecular wt first */
    YOW += y[0]*imw[0]; /*N2 */
    YOW += y[1]*imw[1]; /*H */
    YOW += y[2]*imw[2]; /*H2 */
    YOW += y[3]*imw[3]; /*O */
    YOW += y[4]*imw[4]; /*OH */
    YOW += y[5]*imw[5]; /*O2 */
    YOW += y[6]*imw[6]; /*H2O2 */
    YOW += y[7]*imw[7]; /*H2O */
    YOW += y[8]*imw[8]; /*HO2 */
    YOW += y[9]*imw[9]; /*CO */
    YOW += y[10]*imw[10]; /*CH3 */
    YOW += y[11]*imw[11]; /*CH2O */
    YOW += y[12]*imw[12]; /*CO2 */
    YOW += y[13]*imw[13]; /*CH4 */
    YOW += y[14]*imw[14]; /*C2H2 */
    YOW += y[15]*imw[15]; /*C2H4 */
    YOW += y[16]*imw[16]; /*CH2CO */
    YOW += y[17]*imw[17]; /*C2H6 */
    YOW += y[18]*imw[18]; /*C2H3 */
    YOW += y[19]*imw[19]; /*C2H5 */
    YOW += y[20]*imw[20]; /*HCCO */
    YOW += y[21]*imw[21]; /*CH3CHO */
    YOW += y[22]*imw[22]; /*CH2CHO */
    YOW += y[23]*imw[23]; /*C2H5O */
    /*Now compute y to x conversion */
    x[0] = y[0]/(28.013400*YOW); 
    x[1] = y[1]/(1.007970*YOW); 
    x[2] = y[2]/(2.015940*YOW); 
    x[3] = y[3]/(15.999400*YOW); 
    x[4] = y[4]/(17.007370*YOW); 
    x[5] = y[5]/(31.998800*YOW); 
    x[6] = y[6]/(34.014740*YOW); 
    x[7] = y[7]/(18.015340*YOW); 
    x[8] = y[8]/(33.006770*YOW); 
    x[9] = y[9]/(28.010550*YOW); 
    x[10] = y[10]/(15.035060*YOW); 
    x[11] = y[11]/(30.026490*YOW); 
    x[12] = y[12]/(44.009950*YOW); 
    x[13] = y[13]/(16.043030*YOW); 
    x[14] = y[14]/(26.038240*YOW); 
    x[15] = y[15]/(28.054180*YOW); 
    x[16] = y[16]/(42.037640*YOW); 
    x[17] = y[17]/(30.070120*YOW); 
    x[18] = y[18]/(27.046210*YOW); 
    x[19] = y[19]/(29.062150*YOW); 
    x[20] = y[20]/(41.029670*YOW); 
    x[21] = y[21]/(44.053580*YOW); 
    x[22] = y[22]/(43.045610*YOW); 
    x[23] = y[23]/(45.061550*YOW); 
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
    double aort[24]; /* temporary storage */
    /*Compute g/RT */
    helmholtz(aort, tc);

    /*Compute Eq 44 */
    for (id = 0; id < 24; ++id) {
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
    double aort[24]; /* temporary storage */
    double x[24]; /* need a ytx conversion */
    double YOW = 0; /*To hold 1/molecularweight */
    /*Compute inverse of mean molecular wt first */
    YOW += y[0]*imw[0]; /*N2 */
    YOW += y[1]*imw[1]; /*H */
    YOW += y[2]*imw[2]; /*H2 */
    YOW += y[3]*imw[3]; /*O */
    YOW += y[4]*imw[4]; /*OH */
    YOW += y[5]*imw[5]; /*O2 */
    YOW += y[6]*imw[6]; /*H2O2 */
    YOW += y[7]*imw[7]; /*H2O */
    YOW += y[8]*imw[8]; /*HO2 */
    YOW += y[9]*imw[9]; /*CO */
    YOW += y[10]*imw[10]; /*CH3 */
    YOW += y[11]*imw[11]; /*CH2O */
    YOW += y[12]*imw[12]; /*CO2 */
    YOW += y[13]*imw[13]; /*CH4 */
    YOW += y[14]*imw[14]; /*C2H2 */
    YOW += y[15]*imw[15]; /*C2H4 */
    YOW += y[16]*imw[16]; /*CH2CO */
    YOW += y[17]*imw[17]; /*C2H6 */
    YOW += y[18]*imw[18]; /*C2H3 */
    YOW += y[19]*imw[19]; /*C2H5 */
    YOW += y[20]*imw[20]; /*HCCO */
    YOW += y[21]*imw[21]; /*CH3CHO */
    YOW += y[22]*imw[22]; /*CH2CHO */
    YOW += y[23]*imw[23]; /*C2H5O */
    /*Now compute y to x conversion */
    x[0] = y[0]/(28.013400*YOW); 
    x[1] = y[1]/(1.007970*YOW); 
    x[2] = y[2]/(2.015940*YOW); 
    x[3] = y[3]/(15.999400*YOW); 
    x[4] = y[4]/(17.007370*YOW); 
    x[5] = y[5]/(31.998800*YOW); 
    x[6] = y[6]/(34.014740*YOW); 
    x[7] = y[7]/(18.015340*YOW); 
    x[8] = y[8]/(33.006770*YOW); 
    x[9] = y[9]/(28.010550*YOW); 
    x[10] = y[10]/(15.035060*YOW); 
    x[11] = y[11]/(30.026490*YOW); 
    x[12] = y[12]/(44.009950*YOW); 
    x[13] = y[13]/(16.043030*YOW); 
    x[14] = y[14]/(26.038240*YOW); 
    x[15] = y[15]/(28.054180*YOW); 
    x[16] = y[16]/(42.037640*YOW); 
    x[17] = y[17]/(30.070120*YOW); 
    x[18] = y[18]/(27.046210*YOW); 
    x[19] = y[19]/(29.062150*YOW); 
    x[20] = y[20]/(41.029670*YOW); 
    x[21] = y[21]/(44.053580*YOW); 
    x[22] = y[22]/(43.045610*YOW); 
    x[23] = y[23]/(45.061550*YOW); 
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
    /*Scale by RT/W */
    *abms = result * RT * YOW;
}


/*compute the production rate for each species */
AMREX_GPU_HOST_DEVICE void CKWC(double *  T, double *  C,  double *  wdot)
{
    int id; /*loop counter */

    /*convert to SI */
    for (id = 0; id < 24; ++id) {
        C[id] *= 1.0e6;
    }

    /*convert to chemkin units */
    productionRate(wdot, C, *T);

    /*convert to chemkin units */
    for (id = 0; id < 24; ++id) {
        C[id] *= 1.0e-6;
        wdot[id] *= 1.0e-6;
    }
}


/*Returns the molar production rate of species */
/*Given P, T, and mass fractions */
void CKWYP(double *  P, double *  T, double *  y,  double *  wdot)
{
    int id; /*loop counter */
    double c[24]; /*temporary storage */
    double YOW = 0; 
    double PWORT; 
    /*Compute inverse of mean molecular wt first */
    YOW += y[0]*imw[0]; /*N2 */
    YOW += y[1]*imw[1]; /*H */
    YOW += y[2]*imw[2]; /*H2 */
    YOW += y[3]*imw[3]; /*O */
    YOW += y[4]*imw[4]; /*OH */
    YOW += y[5]*imw[5]; /*O2 */
    YOW += y[6]*imw[6]; /*H2O2 */
    YOW += y[7]*imw[7]; /*H2O */
    YOW += y[8]*imw[8]; /*HO2 */
    YOW += y[9]*imw[9]; /*CO */
    YOW += y[10]*imw[10]; /*CH3 */
    YOW += y[11]*imw[11]; /*CH2O */
    YOW += y[12]*imw[12]; /*CO2 */
    YOW += y[13]*imw[13]; /*CH4 */
    YOW += y[14]*imw[14]; /*C2H2 */
    YOW += y[15]*imw[15]; /*C2H4 */
    YOW += y[16]*imw[16]; /*CH2CO */
    YOW += y[17]*imw[17]; /*C2H6 */
    YOW += y[18]*imw[18]; /*C2H3 */
    YOW += y[19]*imw[19]; /*C2H5 */
    YOW += y[20]*imw[20]; /*HCCO */
    YOW += y[21]*imw[21]; /*CH3CHO */
    YOW += y[22]*imw[22]; /*CH2CHO */
    YOW += y[23]*imw[23]; /*C2H5O */
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

    /*convert to chemkin units */
    productionRate(wdot, c, *T);

    /*convert to chemkin units */
    for (id = 0; id < 24; ++id) {
        wdot[id] *= 1.0e-6;
    }
}


/*Returns the molar production rate of species */
/*Given P, T, and mole fractions */
void CKWXP(double *  P, double *  T, double *  x,  double *  wdot)
{
    int id; /*loop counter */
    double c[24]; /*temporary storage */
    double PORT = 1e6 * (*P)/(8.31446e+07 * (*T)); /*1e6 * P/RT so c goes to SI units */

    /*Compute conversion, see Eq 10 */
    for (id = 0; id < 24; ++id) {
        c[id] = x[id]*PORT;
    }

    /*convert to chemkin units */
    productionRate(wdot, c, *T);

    /*convert to chemkin units */
    for (id = 0; id < 24; ++id) {
        wdot[id] *= 1.0e-6;
    }
}


/*Returns the molar production rate of species */
/*Given rho, T, and mass fractions */
AMREX_GPU_HOST_DEVICE void CKWYR(double *  rho, double *  T, double *  y,  double *  wdot)
{
    int id; /*loop counter */
    double c[24]; /*temporary storage */
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

    /*call productionRate */
    productionRate(wdot, c, *T);

    /*convert to chemkin units */
    for (id = 0; id < 24; ++id) {
        wdot[id] *= 1.0e-6;
    }
}


/*Returns the molar production rate of species */
/*Given rho, T, and mole fractions */
void CKWXR(double *  rho, double *  T, double *  x,  double *  wdot)
{
    int id; /*loop counter */
    double c[24]; /*temporary storage */
    double XW = 0; /*See Eq 4, 11 in CK Manual */
    double ROW; 
    /*Compute mean molecular wt first */
    XW += x[0]*28.013400; /*N2 */
    XW += x[1]*1.007970; /*H */
    XW += x[2]*2.015940; /*H2 */
    XW += x[3]*15.999400; /*O */
    XW += x[4]*17.007370; /*OH */
    XW += x[5]*31.998800; /*O2 */
    XW += x[6]*34.014740; /*H2O2 */
    XW += x[7]*18.015340; /*H2O */
    XW += x[8]*33.006770; /*HO2 */
    XW += x[9]*28.010550; /*CO */
    XW += x[10]*15.035060; /*CH3 */
    XW += x[11]*30.026490; /*CH2O */
    XW += x[12]*44.009950; /*CO2 */
    XW += x[13]*16.043030; /*CH4 */
    XW += x[14]*26.038240; /*C2H2 */
    XW += x[15]*28.054180; /*C2H4 */
    XW += x[16]*42.037640; /*CH2CO */
    XW += x[17]*30.070120; /*C2H6 */
    XW += x[18]*27.046210; /*C2H3 */
    XW += x[19]*29.062150; /*C2H5 */
    XW += x[20]*41.029670; /*HCCO */
    XW += x[21]*44.053580; /*CH3CHO */
    XW += x[22]*43.045610; /*CH2CHO */
    XW += x[23]*45.061550; /*C2H5O */
    /*Extra 1e6 factor to take c to SI */
    ROW = 1e6*(*rho) / XW;

    /*Compute conversion, see Eq 11 */
    for (id = 0; id < 24; ++id) {
        c[id] = x[id]*ROW;
    }

    /*convert to chemkin units */
    productionRate(wdot, c, *T);

    /*convert to chemkin units */
    for (id = 0; id < 24; ++id) {
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
    for (id = 0; id < kd * 24; ++ id) {
         ncf[id] = 0; 
    }

    /*N2 */
    ncf[ 0 * kd + 0 ] = 2; /*N */

    /*H */
    ncf[ 1 * kd + 1 ] = 1; /*H */

    /*H2 */
    ncf[ 2 * kd + 1 ] = 2; /*H */

    /*O */
    ncf[ 3 * kd + 2 ] = 1; /*O */

    /*OH */
    ncf[ 4 * kd + 2 ] = 1; /*O */
    ncf[ 4 * kd + 1 ] = 1; /*H */

    /*O2 */
    ncf[ 5 * kd + 2 ] = 2; /*O */

    /*H2O2 */
    ncf[ 6 * kd + 1 ] = 2; /*H */
    ncf[ 6 * kd + 2 ] = 2; /*O */

    /*H2O */
    ncf[ 7 * kd + 1 ] = 2; /*H */
    ncf[ 7 * kd + 2 ] = 1; /*O */

    /*HO2 */
    ncf[ 8 * kd + 1 ] = 1; /*H */
    ncf[ 8 * kd + 2 ] = 2; /*O */

    /*CO */
    ncf[ 9 * kd + 3 ] = 1; /*C */
    ncf[ 9 * kd + 2 ] = 1; /*O */

    /*CH3 */
    ncf[ 10 * kd + 3 ] = 1; /*C */
    ncf[ 10 * kd + 1 ] = 3; /*H */

    /*CH2O */
    ncf[ 11 * kd + 1 ] = 2; /*H */
    ncf[ 11 * kd + 3 ] = 1; /*C */
    ncf[ 11 * kd + 2 ] = 1; /*O */

    /*CO2 */
    ncf[ 12 * kd + 3 ] = 1; /*C */
    ncf[ 12 * kd + 2 ] = 2; /*O */

    /*CH4 */
    ncf[ 13 * kd + 3 ] = 1; /*C */
    ncf[ 13 * kd + 1 ] = 4; /*H */

    /*C2H2 */
    ncf[ 14 * kd + 3 ] = 2; /*C */
    ncf[ 14 * kd + 1 ] = 2; /*H */

    /*C2H4 */
    ncf[ 15 * kd + 3 ] = 2; /*C */
    ncf[ 15 * kd + 1 ] = 4; /*H */

    /*CH2CO */
    ncf[ 16 * kd + 3 ] = 2; /*C */
    ncf[ 16 * kd + 1 ] = 2; /*H */
    ncf[ 16 * kd + 2 ] = 1; /*O */

    /*C2H6 */
    ncf[ 17 * kd + 3 ] = 2; /*C */
    ncf[ 17 * kd + 1 ] = 6; /*H */

    /*C2H3 */
    ncf[ 18 * kd + 3 ] = 2; /*C */
    ncf[ 18 * kd + 1 ] = 3; /*H */

    /*C2H5 */
    ncf[ 19 * kd + 3 ] = 2; /*C */
    ncf[ 19 * kd + 1 ] = 5; /*H */

    /*HCCO */
    ncf[ 20 * kd + 1 ] = 1; /*H */
    ncf[ 20 * kd + 3 ] = 2; /*C */
    ncf[ 20 * kd + 2 ] = 1; /*O */

    /*CH3CHO */
    ncf[ 21 * kd + 1 ] = 4; /*H */
    ncf[ 21 * kd + 3 ] = 2; /*C */
    ncf[ 21 * kd + 2 ] = 1; /*O */

    /*CH2CHO */
    ncf[ 22 * kd + 1 ] = 3; /*H */
    ncf[ 22 * kd + 3 ] = 2; /*C */
    ncf[ 22 * kd + 2 ] = 1; /*O */

    /*C2H5O */
    ncf[ 23 * kd + 3 ] = 2; /*C */
    ncf[ 23 * kd + 1 ] = 5; /*H */
    ncf[ 23 * kd + 2 ] = 1; /*O */


}

#ifndef AMREX_USE_CUDA
static double T_save = -1;
#ifdef _OPENMP
#pragma omp threadprivate(T_save)
#endif

static double k_f_save[172];
#ifdef _OPENMP
#pragma omp threadprivate(k_f_save)
#endif

static double Kc_save[172];
#ifdef _OPENMP
#pragma omp threadprivate(Kc_save)
#endif

static double k_f_save_qss[72];
#ifdef _OPENMP
#pragma omp threadprivate(k_f_save)
#endif

static double Kc_save_qss[72];
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

    double qdot, q_f[172], q_r[172];
    double sc_qss[5];
    /* Fill sc_qss here*/
    comp_qss_sc(sc, sc_qss, tc, invT);
    comp_qfqr(q_f, q_r, sc, sc_qss, tc, invT);

    for (int i = 0; i < 24; ++i) {
        wdot[i] = 0.0;
    }

    qdot = q_f[0]-q_r[0];
    wdot[4] -= 2.000000 * qdot;
    wdot[6] += qdot;

    qdot = q_f[1]-q_r[1];
    wdot[1] -= qdot;
    wdot[5] -= qdot;
    wdot[8] += qdot;

    qdot = q_f[2]-q_r[2];
    wdot[2] -= qdot;
    wdot[10] += qdot;

    qdot = q_f[3]-q_r[3];
    wdot[1] -= qdot;
    wdot[10] += qdot;

    qdot = q_f[4]-q_r[4];
    wdot[1] -= qdot;
    wdot[10] -= qdot;
    wdot[13] += qdot;

    qdot = q_f[5]-q_r[5];
    wdot[9] -= qdot;
    wdot[16] += qdot;

    qdot = q_f[6]-q_r[6];
    wdot[2] -= qdot;
    wdot[9] -= qdot;
    wdot[11] += qdot;

    qdot = q_f[7]-q_r[7];
    wdot[9] -= qdot;
    wdot[20] += qdot;

    qdot = q_f[8]-q_r[8];
    wdot[3] -= qdot;
    wdot[9] -= qdot;
    wdot[12] += qdot;

    qdot = q_f[9]-q_r[9];
    wdot[1] -= qdot;
    wdot[11] += qdot;

    qdot = q_f[10]-q_r[10];
    wdot[1] -= qdot;
    wdot[14] -= qdot;
    wdot[18] += qdot;

    qdot = q_f[11]-q_r[11];
    wdot[1] -= qdot;
    wdot[15] += qdot;
    wdot[18] -= qdot;

    qdot = q_f[12]-q_r[12];
    wdot[1] -= qdot;
    wdot[15] -= qdot;
    wdot[19] += qdot;

    qdot = q_f[13]-q_r[13];
    wdot[1] -= qdot;
    wdot[17] += qdot;
    wdot[19] -= qdot;

    qdot = q_f[14]-q_r[14];
    wdot[10] += 2.000000 * qdot;
    wdot[17] -= qdot;

    qdot = q_f[15]-q_r[15];
    wdot[1] -= 2.000000 * qdot;
    wdot[2] += qdot;

    qdot = q_f[16]-q_r[16];
    wdot[1] -= qdot;
    wdot[3] -= qdot;
    wdot[4] += qdot;

    qdot = q_f[17]-q_r[17];
    wdot[3] -= 2.000000 * qdot;
    wdot[5] += qdot;

    qdot = q_f[18]-q_r[18];
    wdot[1] -= qdot;
    wdot[4] -= qdot;
    wdot[7] += qdot;

    qdot = q_f[19]-q_r[19];
    wdot[1] += qdot;
    wdot[9] += qdot;

    qdot = q_f[20]-q_r[20];
    wdot[1] -= 2.000000 * qdot;
    wdot[2] -= qdot;
    wdot[2] += 2.000000 * qdot;

    qdot = q_f[21]-q_r[21];
    wdot[1] += qdot;
    wdot[2] -= qdot;
    wdot[3] -= qdot;
    wdot[4] += qdot;

    qdot = q_f[22]-q_r[22];
    wdot[3] += qdot;
    wdot[4] -= 2.000000 * qdot;
    wdot[7] += qdot;

    qdot = q_f[23]-q_r[23];
    wdot[1] += qdot;
    wdot[2] -= qdot;
    wdot[4] -= qdot;
    wdot[7] += qdot;

    qdot = q_f[24]-q_r[24];
    wdot[1] -= 2.000000 * qdot;
    wdot[2] += qdot;
    wdot[7] -= qdot;
    wdot[7] += qdot;

    qdot = q_f[25]-q_r[25];
    wdot[1] -= qdot;
    wdot[3] += qdot;
    wdot[4] += qdot;
    wdot[5] -= qdot;

    qdot = q_f[26]-q_r[26];
    wdot[1] += qdot;
    wdot[2] -= qdot;
    wdot[5] -= qdot;
    wdot[8] += qdot;

    qdot = q_f[27]-q_r[27];
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[7] += qdot;
    wdot[8] -= qdot;

    qdot = q_f[28]-q_r[28];
    wdot[1] -= qdot;
    wdot[4] += 2.000000 * qdot;
    wdot[8] -= qdot;

    qdot = q_f[29]-q_r[29];
    wdot[3] -= qdot;
    wdot[4] += qdot;
    wdot[5] += qdot;
    wdot[8] -= qdot;

    qdot = q_f[30]-q_r[30];
    wdot[1] -= qdot;
    wdot[3] += qdot;
    wdot[7] += qdot;
    wdot[8] -= qdot;

    qdot = q_f[31]-q_r[31];
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[7] += qdot;
    wdot[8] -= qdot;

    qdot = q_f[32]-q_r[32];
    wdot[3] -= qdot;
    wdot[4] += qdot;
    wdot[6] -= qdot;
    wdot[8] += qdot;

    qdot = q_f[33]-q_r[33];
    wdot[1] -= qdot;
    wdot[4] += qdot;
    wdot[6] -= qdot;
    wdot[7] += qdot;

    qdot = q_f[34]-q_r[34];
    wdot[1] -= qdot;
    wdot[2] += qdot;
    wdot[6] -= qdot;
    wdot[8] += qdot;

    qdot = q_f[35]-q_r[35];
    wdot[4] -= qdot;
    wdot[6] -= qdot;
    wdot[7] += qdot;
    wdot[8] += qdot;

    qdot = q_f[36]-q_r[36];
    wdot[4] -= qdot;
    wdot[6] -= qdot;
    wdot[7] += qdot;
    wdot[8] += qdot;

    qdot = q_f[37]-q_r[37];
    wdot[3] += qdot;
    wdot[5] -= qdot;
    wdot[9] += qdot;

    qdot = q_f[38]-q_r[38];
    wdot[1] += qdot;
    wdot[4] -= qdot;
    wdot[9] += qdot;

    qdot = q_f[39]-q_r[39];
    wdot[1] += qdot;
    wdot[4] -= qdot;

    qdot = q_f[40]-q_r[40];
    wdot[1] += qdot;
    wdot[2] -= qdot;

    qdot = q_f[41]-q_r[41];
    wdot[1] += qdot;
    wdot[3] -= qdot;
    wdot[9] += qdot;

    qdot = q_f[42]-q_r[42];
    wdot[3] += qdot;
    wdot[5] -= qdot;

    qdot = q_f[43]-q_r[43];
    wdot[1] -= qdot;
    wdot[2] += qdot;

    qdot = q_f[44]-q_r[44];
    wdot[1] += qdot;
    wdot[7] -= qdot;
    wdot[11] += qdot;

    qdot = q_f[45]-q_r[45];
    wdot[1] += qdot;
    wdot[4] += qdot;
    wdot[5] -= qdot;
    wdot[9] += qdot;

    qdot = q_f[46]-q_r[46];
    wdot[3] += qdot;
    wdot[5] -= qdot;
    wdot[11] += qdot;

    qdot = q_f[47]-q_r[47];
    wdot[1] += qdot;
    wdot[4] -= qdot;
    wdot[11] += qdot;

    qdot = q_f[48]-q_r[48];
    wdot[4] += qdot;
    wdot[8] -= qdot;
    wdot[11] += qdot;

    qdot = q_f[49]-q_r[49];
    wdot[1] += 2.000000 * qdot;
    wdot[5] -= qdot;
    wdot[12] += qdot;

    qdot = q_f[50]-q_r[50];
    wdot[4] -= qdot;
    wdot[7] += qdot;

    qdot = q_f[51]-q_r[51];
    wdot[1] += qdot;
    wdot[3] -= qdot;

    qdot = q_f[52]-q_r[52];
    wdot[1] += qdot;
    wdot[2] -= qdot;
    wdot[10] += qdot;

    qdot = q_f[53]-q_r[53];
    wdot[7] -= qdot;
    wdot[7] += qdot;

    qdot = q_f[54]-q_r[54];
    wdot[1] -= qdot;
    wdot[2] += qdot;

    qdot = q_f[55]-q_r[55];
    wdot[1] += qdot;
    wdot[4] += qdot;
    wdot[5] -= qdot;
    wdot[9] += qdot;

    qdot = q_f[56]-q_r[56];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[9] += qdot;

    qdot = q_f[57]-q_r[57];
    wdot[5] -= qdot;
    wdot[7] += qdot;
    wdot[9] += qdot;

    qdot = q_f[58]-q_r[58];
    wdot[1] += qdot;
    wdot[2] -= qdot;
    wdot[10] += qdot;

    qdot = q_f[59]-q_r[59];
    wdot[1] += qdot;
    wdot[3] -= qdot;

    qdot = q_f[60]-q_r[60];
    wdot[2] += qdot;
    wdot[7] -= qdot;
    wdot[11] += qdot;

    qdot = q_f[61]-q_r[61];
    wdot[1] += qdot;
    wdot[4] -= qdot;
    wdot[11] += qdot;

    qdot = q_f[62]-q_r[62];
    wdot[2] += qdot;
    wdot[4] -= qdot;
    wdot[10] -= qdot;
    wdot[11] += qdot;

    qdot = q_f[63]-q_r[63];
    wdot[6] -= qdot;
    wdot[8] += qdot;
    wdot[10] -= qdot;
    wdot[13] += qdot;

    qdot = q_f[64]-q_r[64];
    wdot[4] += qdot;
    wdot[5] -= qdot;
    wdot[10] -= qdot;
    wdot[11] += qdot;

    qdot = q_f[65]-q_r[65];
    wdot[1] += qdot;
    wdot[10] -= qdot;
    wdot[18] += qdot;

    qdot = q_f[66]-q_r[66];
    wdot[1] += qdot;
    wdot[3] -= qdot;
    wdot[10] -= qdot;
    wdot[11] += qdot;

    qdot = q_f[67]-q_r[67];
    wdot[1] += qdot;
    wdot[10] -= qdot;
    wdot[14] += qdot;

    qdot = q_f[68]-q_r[68];
    wdot[4] -= qdot;
    wdot[7] += qdot;
    wdot[10] -= qdot;

    qdot = q_f[69]-q_r[69];
    wdot[1] += qdot;
    wdot[10] -= qdot;
    wdot[15] += qdot;

    qdot = q_f[70]-q_r[70];
    wdot[4] -= qdot;
    wdot[7] += qdot;
    wdot[10] -= qdot;

    qdot = q_f[71]-q_r[71];
    wdot[1] += qdot;
    wdot[10] -= 2.000000 * qdot;
    wdot[19] += qdot;

    qdot = q_f[72]-q_r[72];
    wdot[5] += qdot;
    wdot[8] -= qdot;
    wdot[10] -= qdot;
    wdot[13] += qdot;

    qdot = q_f[73]-q_r[73];
    wdot[1] += qdot;
    wdot[10] -= qdot;
    wdot[15] += qdot;

    qdot = q_f[74]-q_r[74];
    wdot[1] += qdot;
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[9] += qdot;
    wdot[10] -= qdot;

    qdot = q_f[75]-q_r[75];
    wdot[1] += qdot;
    wdot[13] -= qdot;
    wdot[15] += qdot;

    qdot = q_f[76]-q_r[76];
    wdot[10] += 2.000000 * qdot;
    wdot[13] -= qdot;

    qdot = q_f[77]-q_r[77];
    wdot[3] -= qdot;
    wdot[4] += qdot;
    wdot[10] += qdot;
    wdot[13] -= qdot;

    qdot = q_f[78]-q_r[78];
    wdot[4] -= qdot;
    wdot[7] += qdot;
    wdot[10] += qdot;
    wdot[13] -= qdot;

    qdot = q_f[79]-q_r[79];
    wdot[10] += 2.000000 * qdot;
    wdot[13] -= qdot;

    qdot = q_f[80]-q_r[80];
    wdot[1] -= qdot;
    wdot[2] += qdot;
    wdot[10] += qdot;
    wdot[13] -= qdot;

    qdot = q_f[81]-q_r[81];
    wdot[9] -= qdot;
    wdot[9] += qdot;

    qdot = q_f[82]-q_r[82];
    wdot[3] += qdot;
    wdot[5] -= qdot;
    wdot[9] -= qdot;
    wdot[12] += qdot;

    qdot = q_f[83]-q_r[83];
    wdot[1] += qdot;
    wdot[4] -= qdot;
    wdot[9] -= qdot;
    wdot[12] += qdot;

    qdot = q_f[84]-q_r[84];
    wdot[1] += qdot;
    wdot[4] -= qdot;
    wdot[9] -= qdot;
    wdot[12] += qdot;

    qdot = q_f[85]-q_r[85];
    wdot[4] += qdot;
    wdot[8] -= qdot;
    wdot[9] -= qdot;
    wdot[12] += qdot;

    qdot = q_f[86]-q_r[86];
    wdot[1] -= qdot;
    wdot[2] += qdot;
    wdot[9] += qdot;

    qdot = q_f[87]-q_r[87];
    wdot[10] -= qdot;
    wdot[21] += qdot;

    qdot = q_f[88]-q_r[88];
    wdot[1] += qdot;
    wdot[7] -= qdot;
    wdot[7] += qdot;
    wdot[9] += qdot;

    qdot = q_f[89]-q_r[89];
    wdot[3] -= qdot;
    wdot[4] += qdot;
    wdot[9] += qdot;

    qdot = q_f[90]-q_r[90];
    wdot[4] -= qdot;
    wdot[7] += qdot;
    wdot[9] += qdot;

    qdot = q_f[91]-q_r[91];
    wdot[9] += qdot;
    wdot[10] -= qdot;
    wdot[13] += qdot;

    qdot = q_f[92]-q_r[92];
    wdot[1] += qdot;
    wdot[3] -= qdot;
    wdot[12] += qdot;

    qdot = q_f[93]-q_r[93];
    wdot[5] -= qdot;
    wdot[8] += qdot;
    wdot[9] += qdot;

    qdot = q_f[94]-q_r[94];
    wdot[1] -= qdot;
    wdot[2] += qdot;
    wdot[11] -= qdot;

    qdot = q_f[95]-q_r[95];
    wdot[3] -= qdot;
    wdot[4] += qdot;
    wdot[11] -= qdot;

    qdot = q_f[96]-q_r[96];
    wdot[10] -= qdot;
    wdot[11] -= qdot;
    wdot[13] += qdot;

    qdot = q_f[97]-q_r[97];
    wdot[4] -= qdot;
    wdot[7] += qdot;
    wdot[11] -= qdot;

    qdot = q_f[98]-q_r[98];
    wdot[1] += qdot;
    wdot[11] -= qdot;
    wdot[16] += qdot;

    qdot = q_f[99]-q_r[99];
    wdot[5] -= qdot;
    wdot[8] += qdot;
    wdot[11] -= qdot;

    qdot = q_f[100]-q_r[100];
    wdot[6] += qdot;
    wdot[8] -= qdot;
    wdot[11] -= qdot;

    qdot = q_f[101]-q_r[101];
    wdot[1] -= 2.000000 * qdot;
    wdot[2] += qdot;
    wdot[12] -= qdot;
    wdot[12] += qdot;

    qdot = q_f[102]-q_r[102];
    wdot[12] -= qdot;
    wdot[12] += qdot;

    qdot = q_f[103]-q_r[103];
    wdot[9] += qdot;
    wdot[11] += qdot;
    wdot[12] -= qdot;

    qdot = q_f[104]-q_r[104];
    wdot[9] += qdot;
    wdot[12] -= qdot;

    qdot = q_f[105]-q_r[105];
    wdot[3] -= qdot;
    wdot[9] += qdot;
    wdot[14] -= qdot;

    qdot = q_f[106]-q_r[106];
    wdot[4] -= qdot;
    wdot[9] += qdot;
    wdot[10] += qdot;
    wdot[14] -= qdot;

    qdot = q_f[107]-q_r[107];
    wdot[1] += qdot;
    wdot[4] -= qdot;
    wdot[14] -= qdot;
    wdot[16] += qdot;

    qdot = q_f[108]-q_r[108];
    wdot[1] += qdot;
    wdot[3] -= qdot;
    wdot[14] -= qdot;
    wdot[20] += qdot;

    qdot = q_f[109]-q_r[109];
    wdot[4] -= qdot;
    wdot[7] += qdot;
    wdot[14] += qdot;
    wdot[18] -= qdot;

    qdot = q_f[110]-q_r[110];
    wdot[3] += qdot;
    wdot[5] -= qdot;
    wdot[18] -= qdot;
    wdot[22] += qdot;

    qdot = q_f[111]-q_r[111];
    wdot[3] -= qdot;
    wdot[18] -= qdot;
    wdot[22] += qdot;

    qdot = q_f[112]-q_r[112];
    wdot[1] -= qdot;
    wdot[2] += qdot;
    wdot[14] += qdot;
    wdot[18] -= qdot;

    qdot = q_f[113]-q_r[113];
    wdot[10] -= qdot;
    wdot[13] += qdot;
    wdot[14] += qdot;
    wdot[18] -= qdot;

    qdot = q_f[114]-q_r[114];
    wdot[5] -= qdot;
    wdot[11] += qdot;
    wdot[18] -= qdot;

    qdot = q_f[115]-q_r[115];
    wdot[6] -= qdot;
    wdot[8] += qdot;
    wdot[15] += qdot;
    wdot[18] -= qdot;

    qdot = q_f[116]-q_r[116];
    wdot[5] -= qdot;
    wdot[8] += qdot;
    wdot[14] += qdot;
    wdot[18] -= qdot;

    qdot = q_f[117]-q_r[117];
    wdot[10] -= qdot;
    wdot[13] += qdot;
    wdot[15] -= qdot;
    wdot[18] += qdot;

    qdot = q_f[118]-q_r[118];
    wdot[1] += qdot;
    wdot[5] -= qdot;
    wdot[10] += qdot;
    wdot[12] += qdot;
    wdot[15] -= qdot;

    qdot = q_f[119]-q_r[119];
    wdot[4] -= qdot;
    wdot[7] += qdot;
    wdot[15] -= qdot;
    wdot[18] += qdot;

    qdot = q_f[120]-q_r[120];
    wdot[4] -= qdot;
    wdot[15] -= qdot;
    wdot[23] += qdot;

    qdot = q_f[121]-q_r[121];
    wdot[1] += qdot;
    wdot[3] -= qdot;
    wdot[15] -= qdot;
    wdot[22] += qdot;

    qdot = q_f[122]-q_r[122];
    wdot[3] -= qdot;
    wdot[10] += qdot;
    wdot[15] -= qdot;

    qdot = q_f[123]-q_r[123];
    wdot[5] -= qdot;
    wdot[8] += qdot;
    wdot[15] -= qdot;
    wdot[18] += qdot;

    qdot = q_f[124]-q_r[124];
    wdot[1] -= qdot;
    wdot[2] += qdot;
    wdot[15] -= qdot;
    wdot[18] += qdot;

    qdot = q_f[125]-q_r[125];
    wdot[3] -= qdot;
    wdot[11] += qdot;
    wdot[15] -= qdot;

    qdot = q_f[126]-q_r[126];
    wdot[6] += qdot;
    wdot[8] -= qdot;
    wdot[15] += qdot;
    wdot[19] -= qdot;

    qdot = q_f[127]-q_r[127];
    wdot[4] += qdot;
    wdot[8] -= qdot;
    wdot[19] -= qdot;
    wdot[23] += qdot;

    qdot = q_f[128]-q_r[128];
    wdot[3] -= qdot;
    wdot[19] -= qdot;
    wdot[23] += qdot;

    qdot = q_f[129]-q_r[129];
    wdot[1] -= qdot;
    wdot[2] += qdot;
    wdot[15] += qdot;
    wdot[19] -= qdot;

    qdot = q_f[130]-q_r[130];
    wdot[5] -= qdot;
    wdot[8] += qdot;
    wdot[15] += qdot;
    wdot[19] -= qdot;

    qdot = q_f[131]-q_r[131];
    wdot[5] += qdot;
    wdot[8] -= qdot;
    wdot[17] += qdot;
    wdot[19] -= qdot;

    qdot = q_f[132]-q_r[132];
    wdot[10] -= qdot;
    wdot[13] += qdot;
    wdot[15] += qdot;
    wdot[19] -= qdot;

    qdot = q_f[133]-q_r[133];
    wdot[10] += qdot;
    wdot[17] -= qdot;
    wdot[19] += qdot;

    qdot = q_f[134]-q_r[134];
    wdot[10] -= qdot;
    wdot[13] += qdot;
    wdot[17] -= qdot;
    wdot[19] += qdot;

    qdot = q_f[135]-q_r[135];
    wdot[3] -= qdot;
    wdot[4] += qdot;
    wdot[17] -= qdot;
    wdot[19] += qdot;

    qdot = q_f[136]-q_r[136];
    wdot[6] += qdot;
    wdot[8] -= qdot;
    wdot[17] -= qdot;
    wdot[19] += qdot;

    qdot = q_f[137]-q_r[137];
    wdot[1] -= qdot;
    wdot[2] += qdot;
    wdot[17] -= qdot;
    wdot[19] += qdot;

    qdot = q_f[138]-q_r[138];
    wdot[4] -= qdot;
    wdot[7] += qdot;
    wdot[17] -= qdot;
    wdot[19] += qdot;

    qdot = q_f[139]-q_r[139];
    wdot[4] += qdot;
    wdot[5] -= qdot;
    wdot[9] += 2.000000 * qdot;
    wdot[20] -= qdot;

    qdot = q_f[140]-q_r[140];
    wdot[1] += qdot;
    wdot[3] -= qdot;
    wdot[9] += 2.000000 * qdot;
    wdot[20] -= qdot;

    qdot = q_f[141]-q_r[141];
    wdot[9] += qdot;
    wdot[10] -= qdot;
    wdot[15] += qdot;
    wdot[20] -= qdot;

    qdot = q_f[142]-q_r[142];
    wdot[1] -= qdot;
    wdot[9] += qdot;
    wdot[20] -= qdot;

    qdot = q_f[143]-q_r[143];
    wdot[1] -= qdot;
    wdot[9] += qdot;
    wdot[10] += qdot;
    wdot[16] -= qdot;

    qdot = q_f[144]-q_r[144];
    wdot[9] += qdot;
    wdot[15] += qdot;
    wdot[16] -= qdot;

    qdot = q_f[145]-q_r[145];
    wdot[3] -= qdot;
    wdot[4] += qdot;
    wdot[16] -= qdot;
    wdot[20] += qdot;

    qdot = q_f[146]-q_r[146];
    wdot[10] -= qdot;
    wdot[13] += qdot;
    wdot[16] -= qdot;
    wdot[20] += qdot;

    qdot = q_f[147]-q_r[147];
    wdot[3] -= qdot;
    wdot[12] += qdot;
    wdot[16] -= qdot;

    qdot = q_f[148]-q_r[148];
    wdot[9] += qdot;
    wdot[10] -= qdot;
    wdot[16] -= qdot;
    wdot[19] += qdot;

    qdot = q_f[149]-q_r[149];
    wdot[4] -= qdot;
    wdot[7] += qdot;
    wdot[16] -= qdot;
    wdot[20] += qdot;

    qdot = q_f[150]-q_r[150];
    wdot[1] -= qdot;
    wdot[2] += qdot;
    wdot[16] -= qdot;
    wdot[20] += qdot;

    qdot = q_f[151]-q_r[151];
    wdot[10] += qdot;
    wdot[16] -= qdot;
    wdot[20] += qdot;

    qdot = q_f[152]-q_r[152];
    wdot[3] -= qdot;
    wdot[11] += qdot;
    wdot[22] -= qdot;

    qdot = q_f[153]-q_r[153];
    wdot[1] += qdot;
    wdot[16] += qdot;
    wdot[22] -= qdot;

    qdot = q_f[154]-q_r[154];
    wdot[4] -= qdot;
    wdot[7] += qdot;
    wdot[16] += qdot;
    wdot[22] -= qdot;

    qdot = q_f[155]-q_r[155];
    wdot[1] -= qdot;
    wdot[2] += qdot;
    wdot[16] += qdot;
    wdot[22] -= qdot;

    qdot = q_f[156]-q_r[156];
    wdot[4] += qdot;
    wdot[5] -= qdot;
    wdot[9] += qdot;
    wdot[11] += qdot;
    wdot[22] -= qdot;

    qdot = q_f[157]-q_r[157];
    wdot[9] += qdot;
    wdot[10] += qdot;
    wdot[22] -= qdot;

    qdot = q_f[158]-q_r[158];
    wdot[4] += qdot;
    wdot[5] -= qdot;
    wdot[22] -= qdot;

    qdot = q_f[159]-q_r[159];
    wdot[1] -= qdot;
    wdot[10] += qdot;
    wdot[22] -= qdot;

    qdot = q_f[160]-q_r[160];
    wdot[3] -= qdot;
    wdot[4] += qdot;
    wdot[9] += qdot;
    wdot[10] += qdot;
    wdot[21] -= qdot;

    qdot = q_f[161]-q_r[161];
    wdot[5] -= qdot;
    wdot[8] += qdot;
    wdot[9] += qdot;
    wdot[10] += qdot;
    wdot[21] -= qdot;

    qdot = q_f[162]-q_r[162];
    wdot[4] -= qdot;
    wdot[7] += qdot;
    wdot[9] += qdot;
    wdot[10] += qdot;
    wdot[21] -= qdot;

    qdot = q_f[163]-q_r[163];
    wdot[1] -= qdot;
    wdot[2] += qdot;
    wdot[21] -= qdot;
    wdot[22] += qdot;

    qdot = q_f[164]-q_r[164];
    wdot[1] -= qdot;
    wdot[2] += qdot;
    wdot[9] += qdot;
    wdot[10] += qdot;
    wdot[21] -= qdot;

    qdot = q_f[165]-q_r[165];
    wdot[3] -= qdot;
    wdot[4] += qdot;
    wdot[21] -= qdot;
    wdot[22] += qdot;

    qdot = q_f[166]-q_r[166];
    wdot[9] += qdot;
    wdot[10] -= qdot;
    wdot[10] += qdot;
    wdot[13] += qdot;
    wdot[21] -= qdot;

    qdot = q_f[167]-q_r[167];
    wdot[6] += qdot;
    wdot[8] -= qdot;
    wdot[9] += qdot;
    wdot[10] += qdot;
    wdot[21] -= qdot;

    qdot = q_f[168]-q_r[168];
    wdot[10] += qdot;
    wdot[11] += qdot;
    wdot[23] -= qdot;

    qdot = q_f[169]-q_r[169];
    wdot[1] += qdot;
    wdot[21] += qdot;
    wdot[23] -= qdot;

    qdot = q_f[170]-q_r[170];
    wdot[5] -= qdot;
    wdot[8] += qdot;
    wdot[21] += qdot;
    wdot[23] -= qdot;

    qdot = q_f[171]-q_r[171];
    wdot[0] -= qdot;
    wdot[0] += qdot;

    return;
}

void comp_k_f(double *  tc, double invT, double *  k_f)
{
#ifdef __INTEL_COMPILER
    #pragma simd
#endif
    for (int i=0; i<172; ++i) {
        k_f[i] = prefactor_units[i] * fwd_A[i]
                    * exp(fwd_beta[i] * tc[0] - activation_units[i] * fwd_Ea[i] * invT);
    };
    return;
}

void comp_Kc(double *  tc, double invT, double *  Kc)
{
    /*compute the Gibbs free energy */
    double g_RT[24], g_RT_qss[5];
    gibbs(g_RT, tc);
    gibbs_qss(g_RT_qss, tc);

    Kc[0] = 2.000000*g_RT[4] - g_RT[6];
    Kc[1] = g_RT[1] + g_RT[5] - g_RT[8];
    Kc[2] = g_RT[2] - g_RT[10] + g_RT_qss[1];
    Kc[3] = g_RT[1] - g_RT[10] + g_RT_qss[3];
    Kc[4] = g_RT[1] + g_RT[10] - g_RT[13];
    Kc[5] = g_RT[9] - g_RT[16] + g_RT_qss[3];
    Kc[6] = g_RT[2] + g_RT[9] - g_RT[11];
    Kc[7] = g_RT[9] - g_RT[20] + g_RT_qss[1];
    Kc[8] = g_RT[3] + g_RT[9] - g_RT[12];
    Kc[9] = g_RT[1] - g_RT[11] + g_RT_qss[2];
    Kc[10] = g_RT[1] + g_RT[14] - g_RT[18];
    Kc[11] = g_RT[1] - g_RT[15] + g_RT[18];
    Kc[12] = g_RT[1] + g_RT[15] - g_RT[19];
    Kc[13] = g_RT[1] - g_RT[17] + g_RT[19];
    Kc[14] = -2.000000*g_RT[10] + g_RT[17];
    Kc[15] = 2.000000*g_RT[1] - g_RT[2];
    Kc[16] = g_RT[1] + g_RT[3] - g_RT[4];
    Kc[17] = 2.000000*g_RT[3] - g_RT[5];
    Kc[18] = g_RT[1] + g_RT[4] - g_RT[7];
    Kc[19] = -g_RT[1] - g_RT[9] + g_RT_qss[2];
    Kc[20] = 2.000000*g_RT[1] + g_RT[2] - 2.000000*g_RT[2];
    Kc[21] = -g_RT[1] + g_RT[2] + g_RT[3] - g_RT[4];
    Kc[22] = -g_RT[3] + 2.000000*g_RT[4] - g_RT[7];
    Kc[23] = -g_RT[1] + g_RT[2] + g_RT[4] - g_RT[7];
    Kc[24] = 2.000000*g_RT[1] - g_RT[2] + g_RT[7] - g_RT[7];
    Kc[25] = g_RT[1] - g_RT[3] - g_RT[4] + g_RT[5];
    Kc[26] = -g_RT[1] + g_RT[2] + g_RT[5] - g_RT[8];
    Kc[27] = g_RT[4] - g_RT[5] - g_RT[7] + g_RT[8];
    Kc[28] = g_RT[1] - 2.000000*g_RT[4] + g_RT[8];
    Kc[29] = g_RT[3] - g_RT[4] - g_RT[5] + g_RT[8];
    Kc[30] = g_RT[1] - g_RT[3] - g_RT[7] + g_RT[8];
    Kc[31] = g_RT[4] - g_RT[5] - g_RT[7] + g_RT[8];
    Kc[32] = g_RT[3] - g_RT[4] + g_RT[6] - g_RT[8];
    Kc[33] = g_RT[1] - g_RT[4] + g_RT[6] - g_RT[7];
    Kc[34] = g_RT[1] - g_RT[2] + g_RT[6] - g_RT[8];
    Kc[35] = g_RT[4] + g_RT[6] - g_RT[7] - g_RT[8];
    Kc[36] = g_RT[4] + g_RT[6] - g_RT[7] - g_RT[8];
    Kc[37] = -g_RT[3] + g_RT[5] - g_RT[9] + g_RT_qss[0];
    Kc[38] = -g_RT[1] + g_RT[4] - g_RT[9] + g_RT_qss[0];
    Kc[39] = -g_RT[1] + g_RT[4] + g_RT_qss[1] - g_RT_qss[2];
    Kc[40] = -g_RT[1] + g_RT[2] + g_RT_qss[1] - g_RT_qss[3];
    Kc[41] = -g_RT[1] + g_RT[3] - g_RT[9] + g_RT_qss[1];
    Kc[42] = -g_RT[3] + g_RT[5] + g_RT_qss[1] - g_RT_qss[2];
    Kc[43] = g_RT[1] - g_RT[2] - g_RT_qss[0] + g_RT_qss[1];
    Kc[44] = -g_RT[1] + g_RT[7] - g_RT[11] + g_RT_qss[1];
    Kc[45] = -g_RT[1] - g_RT[4] + g_RT[5] - g_RT[9] + g_RT_qss[3];
    Kc[46] = -g_RT[3] + g_RT[5] - g_RT[11] + g_RT_qss[3];
    Kc[47] = -g_RT[1] + g_RT[4] - g_RT[11] + g_RT_qss[3];
    Kc[48] = -g_RT[4] + g_RT[8] - g_RT[11] + g_RT_qss[3];
    Kc[49] = -2.000000*g_RT[1] + g_RT[5] - g_RT[12] + g_RT_qss[3];
    Kc[50] = g_RT[4] - g_RT[7] - g_RT_qss[1] + g_RT_qss[3];
    Kc[51] = -g_RT[1] + g_RT[3] - g_RT_qss[2] + g_RT_qss[3];
    Kc[52] = -g_RT[1] + g_RT[2] - g_RT[10] + g_RT_qss[3];
    Kc[53] = g_RT[7] - g_RT[7] - g_RT_qss[3] + g_RT_qss[4];
    Kc[54] = g_RT[1] - g_RT[2] - g_RT_qss[1] + g_RT_qss[4];
    Kc[55] = -g_RT[1] - g_RT[4] + g_RT[5] - g_RT[9] + g_RT_qss[4];
    Kc[56] = -g_RT[2] + g_RT[3] - g_RT[9] + g_RT_qss[4];
    Kc[57] = g_RT[5] - g_RT[7] - g_RT[9] + g_RT_qss[4];
    Kc[58] = -g_RT[1] + g_RT[2] - g_RT[10] + g_RT_qss[4];
    Kc[59] = -g_RT[1] + g_RT[3] - g_RT_qss[2] + g_RT_qss[4];
    Kc[60] = -g_RT[2] + g_RT[7] - g_RT[11] + g_RT_qss[4];
    Kc[61] = -g_RT[1] + g_RT[4] - g_RT[11] + g_RT_qss[4];
    Kc[62] = -g_RT[2] + g_RT[4] + g_RT[10] - g_RT[11];
    Kc[63] = g_RT[6] - g_RT[8] + g_RT[10] - g_RT[13];
    Kc[64] = -g_RT[4] + g_RT[5] + g_RT[10] - g_RT[11];
    Kc[65] = -g_RT[1] + g_RT[10] - g_RT[18] + g_RT_qss[1];
    Kc[66] = -g_RT[1] + g_RT[3] + g_RT[10] - g_RT[11];
    Kc[67] = -g_RT[1] + g_RT[10] - g_RT[14] + g_RT_qss[0];
    Kc[68] = g_RT[4] - g_RT[7] + g_RT[10] - g_RT_qss[3];
    Kc[69] = -g_RT[1] + g_RT[10] - g_RT[15] + g_RT_qss[4];
    Kc[70] = g_RT[4] - g_RT[7] + g_RT[10] - g_RT_qss[4];
    Kc[71] = -g_RT[1] + 2.000000*g_RT[10] - g_RT[19];
    Kc[72] = -g_RT[5] + g_RT[8] + g_RT[10] - g_RT[13];
    Kc[73] = -g_RT[1] + g_RT[10] - g_RT[15] + g_RT_qss[3];
    Kc[74] = -g_RT[1] - g_RT[2] + g_RT[3] - g_RT[9] + g_RT[10];
    Kc[75] = -g_RT[1] + g_RT[13] - g_RT[15] + g_RT_qss[1];
    Kc[76] = -2.000000*g_RT[10] + g_RT[13] + g_RT_qss[4];
    Kc[77] = g_RT[3] - g_RT[4] - g_RT[10] + g_RT[13];
    Kc[78] = g_RT[4] - g_RT[7] - g_RT[10] + g_RT[13];
    Kc[79] = -2.000000*g_RT[10] + g_RT[13] + g_RT_qss[3];
    Kc[80] = g_RT[1] - g_RT[2] - g_RT[10] + g_RT[13];
    Kc[81] = g_RT[9] - g_RT[9] - g_RT_qss[3] + g_RT_qss[4];
    Kc[82] = -g_RT[3] + g_RT[5] + g_RT[9] - g_RT[12];
    Kc[83] = -g_RT[1] + g_RT[4] + g_RT[9] - g_RT[12];
    Kc[84] = -g_RT[1] + g_RT[4] + g_RT[9] - g_RT[12];
    Kc[85] = -g_RT[4] + g_RT[8] + g_RT[9] - g_RT[12];
    Kc[86] = g_RT[1] - g_RT[2] - g_RT[9] + g_RT_qss[2];
    Kc[87] = g_RT[10] - g_RT[21] + g_RT_qss[2];
    Kc[88] = -g_RT[1] + g_RT[7] - g_RT[7] - g_RT[9] + g_RT_qss[2];
    Kc[89] = g_RT[3] - g_RT[4] - g_RT[9] + g_RT_qss[2];
    Kc[90] = g_RT[4] - g_RT[7] - g_RT[9] + g_RT_qss[2];
    Kc[91] = -g_RT[9] + g_RT[10] - g_RT[13] + g_RT_qss[2];
    Kc[92] = -g_RT[1] + g_RT[3] - g_RT[12] + g_RT_qss[2];
    Kc[93] = g_RT[5] - g_RT[8] - g_RT[9] + g_RT_qss[2];
    Kc[94] = g_RT[1] - g_RT[2] + g_RT[11] - g_RT_qss[2];
    Kc[95] = g_RT[3] - g_RT[4] + g_RT[11] - g_RT_qss[2];
    Kc[96] = g_RT[10] + g_RT[11] - g_RT[13] - g_RT_qss[2];
    Kc[97] = g_RT[4] - g_RT[7] + g_RT[11] - g_RT_qss[2];
    Kc[98] = -g_RT[1] + g_RT[11] - g_RT[16] + g_RT_qss[1];
    Kc[99] = g_RT[5] - g_RT[8] + g_RT[11] - g_RT_qss[2];
    Kc[100] = -g_RT[6] + g_RT[8] + g_RT[11] - g_RT_qss[2];
    Kc[101] = 2.000000*g_RT[1] - g_RT[2] + g_RT[12] - g_RT[12];
    Kc[102] = g_RT[12] - g_RT[12] - g_RT_qss[3] + g_RT_qss[4];
    Kc[103] = -g_RT[9] - g_RT[11] + g_RT[12] + g_RT_qss[4];
    Kc[104] = -g_RT[9] + g_RT[12] + g_RT_qss[1] - g_RT_qss[2];
    Kc[105] = g_RT[3] - g_RT[9] + g_RT[14] - g_RT_qss[3];
    Kc[106] = g_RT[4] - g_RT[9] - g_RT[10] + g_RT[14];
    Kc[107] = -g_RT[1] + g_RT[4] + g_RT[14] - g_RT[16];
    Kc[108] = -g_RT[1] + g_RT[3] + g_RT[14] - g_RT[20];
    Kc[109] = g_RT[4] - g_RT[7] - g_RT[14] + g_RT[18];
    Kc[110] = -g_RT[3] + g_RT[5] + g_RT[18] - g_RT[22];
    Kc[111] = g_RT[3] + g_RT[18] - g_RT[22];
    Kc[112] = g_RT[1] - g_RT[2] - g_RT[14] + g_RT[18];
    Kc[113] = g_RT[10] - g_RT[13] - g_RT[14] + g_RT[18];
    Kc[114] = g_RT[5] - g_RT[11] + g_RT[18] - g_RT_qss[2];
    Kc[115] = g_RT[6] - g_RT[8] - g_RT[15] + g_RT[18];
    Kc[116] = g_RT[5] - g_RT[8] - g_RT[14] + g_RT[18];
    Kc[117] = g_RT[10] - g_RT[13] + g_RT[15] - g_RT[18];
    Kc[118] = -g_RT[1] + g_RT[5] - g_RT[10] - g_RT[12] + g_RT[15];
    Kc[119] = g_RT[4] - g_RT[7] + g_RT[15] - g_RT[18];
    Kc[120] = g_RT[4] + g_RT[15] - g_RT[23];
    Kc[121] = -g_RT[1] + g_RT[3] + g_RT[15] - g_RT[22];
    Kc[122] = g_RT[3] - g_RT[10] + g_RT[15] - g_RT_qss[2];
    Kc[123] = g_RT[5] - g_RT[8] + g_RT[15] - g_RT[18];
    Kc[124] = g_RT[1] - g_RT[2] + g_RT[15] - g_RT[18];
    Kc[125] = g_RT[3] - g_RT[11] + g_RT[15] - g_RT_qss[3];
    Kc[126] = -g_RT[6] + g_RT[8] - g_RT[15] + g_RT[19];
    Kc[127] = -g_RT[4] + g_RT[8] + g_RT[19] - g_RT[23];
    Kc[128] = g_RT[3] + g_RT[19] - g_RT[23];
    Kc[129] = g_RT[1] - g_RT[2] - g_RT[15] + g_RT[19];
    Kc[130] = g_RT[5] - g_RT[8] - g_RT[15] + g_RT[19];
    Kc[131] = -g_RT[5] + g_RT[8] - g_RT[17] + g_RT[19];
    Kc[132] = g_RT[10] - g_RT[13] - g_RT[15] + g_RT[19];
    Kc[133] = -g_RT[10] + g_RT[17] - g_RT[19] + g_RT_qss[4];
    Kc[134] = g_RT[10] - g_RT[13] + g_RT[17] - g_RT[19];
    Kc[135] = g_RT[3] - g_RT[4] + g_RT[17] - g_RT[19];
    Kc[136] = -g_RT[6] + g_RT[8] + g_RT[17] - g_RT[19];
    Kc[137] = g_RT[1] - g_RT[2] + g_RT[17] - g_RT[19];
    Kc[138] = g_RT[4] - g_RT[7] + g_RT[17] - g_RT[19];
    Kc[139] = -g_RT[4] + g_RT[5] - 2.000000*g_RT[9] + g_RT[20];
    Kc[140] = -g_RT[1] + g_RT[3] - 2.000000*g_RT[9] + g_RT[20];
    Kc[141] = -g_RT[9] + g_RT[10] - g_RT[15] + g_RT[20];
    Kc[142] = g_RT[1] - g_RT[9] + g_RT[20] - g_RT_qss[4];
    Kc[143] = g_RT[1] - g_RT[9] - g_RT[10] + g_RT[16];
    Kc[144] = -g_RT[9] - g_RT[15] + g_RT[16] + g_RT_qss[3];
    Kc[145] = g_RT[3] - g_RT[4] + g_RT[16] - g_RT[20];
    Kc[146] = g_RT[10] - g_RT[13] + g_RT[16] - g_RT[20];
    Kc[147] = g_RT[3] - g_RT[12] + g_RT[16] - g_RT_qss[3];
    Kc[148] = -g_RT[9] + g_RT[10] + g_RT[16] - g_RT[19];
    Kc[149] = g_RT[4] - g_RT[7] + g_RT[16] - g_RT[20];
    Kc[150] = g_RT[1] - g_RT[2] + g_RT[16] - g_RT[20];
    Kc[151] = -g_RT[10] + g_RT[16] - g_RT[20] + g_RT_qss[3];
    Kc[152] = g_RT[3] - g_RT[11] + g_RT[22] - g_RT_qss[2];
    Kc[153] = -g_RT[1] - g_RT[16] + g_RT[22];
    Kc[154] = g_RT[4] - g_RT[7] - g_RT[16] + g_RT[22];
    Kc[155] = g_RT[1] - g_RT[2] - g_RT[16] + g_RT[22];
    Kc[156] = -g_RT[4] + g_RT[5] - g_RT[9] - g_RT[11] + g_RT[22];
    Kc[157] = -g_RT[9] - g_RT[10] + g_RT[22];
    Kc[158] = -g_RT[4] + g_RT[5] + g_RT[22] - 2.000000*g_RT_qss[2];
    Kc[159] = g_RT[1] - g_RT[10] + g_RT[22] - g_RT_qss[2];
    Kc[160] = g_RT[3] - g_RT[4] - g_RT[9] - g_RT[10] + g_RT[21];
    Kc[161] = g_RT[5] - g_RT[8] - g_RT[9] - g_RT[10] + g_RT[21];
    Kc[162] = g_RT[4] - g_RT[7] - g_RT[9] - g_RT[10] + g_RT[21];
    Kc[163] = g_RT[1] - g_RT[2] + g_RT[21] - g_RT[22];
    Kc[164] = g_RT[1] - g_RT[2] - g_RT[9] - g_RT[10] + g_RT[21];
    Kc[165] = g_RT[3] - g_RT[4] + g_RT[21] - g_RT[22];
    Kc[166] = -g_RT[9] + g_RT[10] - g_RT[10] - g_RT[13] + g_RT[21];
    Kc[167] = -g_RT[6] + g_RT[8] - g_RT[9] - g_RT[10] + g_RT[21];
    Kc[168] = -g_RT[10] - g_RT[11] + g_RT[23];
    Kc[169] = -g_RT[1] - g_RT[21] + g_RT[23];
    Kc[170] = g_RT[5] - g_RT[8] - g_RT[21] + g_RT[23];
    Kc[171] = g_RT[0] - g_RT[0] - g_RT_qss[3] + g_RT_qss[4];

#ifdef __INTEL_COMPILER
     #pragma simd
#endif
    for (int i=0; i<172; ++i) {
        Kc[i] = exp(Kc[i]);
    };

    /*reference concentration: P_atm / (RT) in inverse mol/m^3 */
    double refC = 101325 / 8.31446 * invT;
    double refCinv = 1 / refC;

    Kc[0] *= refCinv;
    Kc[1] *= refCinv;
    Kc[2] *= refCinv;
    Kc[3] *= refCinv;
    Kc[4] *= refCinv;
    Kc[5] *= refCinv;
    Kc[6] *= refCinv;
    Kc[7] *= refCinv;
    Kc[8] *= refCinv;
    Kc[9] *= refCinv;
    Kc[10] *= refCinv;
    Kc[11] *= refCinv;
    Kc[12] *= refCinv;
    Kc[13] *= refCinv;
    Kc[14] *= refC;
    Kc[15] *= refCinv;
    Kc[16] *= refCinv;
    Kc[17] *= refCinv;
    Kc[18] *= refCinv;
    Kc[19] *= refC;
    Kc[20] *= refCinv;
    Kc[24] *= refCinv;
    Kc[45] *= refC;
    Kc[49] *= refC;
    Kc[55] *= refC;
    Kc[74] *= refC;
    Kc[87] *= refCinv;
    Kc[88] *= refC;
    Kc[101] *= refCinv;
    Kc[111] *= refCinv;
    Kc[118] *= refC;
    Kc[120] *= refCinv;
    Kc[128] *= refCinv;
    Kc[139] *= refC;
    Kc[140] *= refC;
    Kc[153] *= refC;
    Kc[156] *= refC;
    Kc[157] *= refC;
    Kc[158] *= refC;
    Kc[160] *= refC;
    Kc[161] *= refC;
    Kc[162] *= refC;
    Kc[164] *= refC;
    Kc[166] *= refC;
    Kc[167] *= refC;
    Kc[168] *= refC;
    Kc[169] *= refC;

    return;
}

void comp_qfqr(double *  qf, double *  qr, double *  sc, double * qss_sc, double *  tc, double invT)
{

    /*reaction 1: 2.000000 OH (+M) <=> H2O2 (+M) */
    qf[0] = pow(sc[4], 2.000000);
    qr[0] = sc[6];

    /*reaction 2: H + O2 (+M) <=> HO2 (+M) */
    qf[1] = sc[1]*sc[5];
    qr[1] = sc[8];

    /*reaction 3: CH + H2 (+M) <=> CH3 (+M) */
    qf[2] = sc[2]*qss_sc[1];
    qr[2] = sc[10];

    /*reaction 4: TXCH2 + H (+M) <=> CH3 (+M) */
    qf[3] = sc[1]*qss_sc[3];
    qr[3] = sc[10];

    /*reaction 5: CH3 + H (+M) <=> CH4 (+M) */
    qf[4] = sc[1]*sc[10];
    qr[4] = sc[13];

    /*reaction 6: TXCH2 + CO (+M) <=> CH2CO (+M) */
    qf[5] = sc[9]*qss_sc[3];
    qr[5] = sc[16];

    /*reaction 7: CO + H2 (+M) <=> CH2O (+M) */
    qf[6] = sc[2]*sc[9];
    qr[6] = sc[11];

    /*reaction 8: CH + CO (+M) <=> HCCO (+M) */
    qf[7] = sc[9]*qss_sc[1];
    qr[7] = sc[20];

    /*reaction 9: CO + O (+M) <=> CO2 (+M) */
    qf[8] = sc[3]*sc[9];
    qr[8] = sc[12];

    /*reaction 10: HCO + H (+M) <=> CH2O (+M) */
    qf[9] = sc[1]*qss_sc[2];
    qr[9] = sc[11];

    /*reaction 11: C2H2 + H (+M) <=> C2H3 (+M) */
    qf[10] = sc[1]*sc[14];
    qr[10] = sc[18];

    /*reaction 12: C2H3 + H (+M) <=> C2H4 (+M) */
    qf[11] = sc[1]*sc[18];
    qr[11] = sc[15];

    /*reaction 13: C2H4 + H (+M) <=> C2H5 (+M) */
    qf[12] = sc[1]*sc[15];
    qr[12] = sc[19];

    /*reaction 14: C2H5 + H (+M) <=> C2H6 (+M) */
    qf[13] = sc[1]*sc[19];
    qr[13] = sc[17];

    /*reaction 15: C2H6 (+M) <=> 2.000000 CH3 (+M) */
    qf[14] = sc[17];
    qr[14] = pow(sc[10], 2.000000);

    /*reaction 16: 2.000000 H + M <=> H2 + M */
    qf[15] = pow(sc[1], 2.000000);
    qr[15] = sc[2];

    /*reaction 17: H + O + M <=> OH + M */
    qf[16] = sc[1]*sc[3];
    qr[16] = sc[4];

    /*reaction 18: 2.000000 O + M <=> O2 + M */
    qf[17] = pow(sc[3], 2.000000);
    qr[17] = sc[5];

    /*reaction 19: H + OH + M <=> H2O + M */
    qf[18] = sc[1]*sc[4];
    qr[18] = sc[7];

    /*reaction 20: HCO + M <=> CO + H + M */
    qf[19] = qss_sc[2];
    qr[19] = sc[1]*sc[9];

    /*reaction 21: 2.000000 H + H2 <=> 2.000000 H2 */
    qf[20] = pow(sc[1], 2.000000)*sc[2];
    qr[20] = pow(sc[2], 2.000000);

    /*reaction 22: O + H2 <=> H + OH */
    qf[21] = sc[2]*sc[3];
    qr[21] = sc[1]*sc[4];

    /*reaction 23: 2.000000 OH <=> O + H2O */
    qf[22] = pow(sc[4], 2.000000);
    qr[22] = sc[3]*sc[7];

    /*reaction 24: OH + H2 <=> H + H2O */
    qf[23] = sc[2]*sc[4];
    qr[23] = sc[1]*sc[7];

    /*reaction 25: 2.000000 H + H2O <=> H2 + H2O */
    qf[24] = pow(sc[1], 2.000000)*sc[7];
    qr[24] = sc[2]*sc[7];

    /*reaction 26: H + O2 <=> O + OH */
    qf[25] = sc[1]*sc[5];
    qr[25] = sc[3]*sc[4];

    /*reaction 27: H2 + O2 <=> HO2 + H */
    qf[26] = sc[2]*sc[5];
    qr[26] = sc[1]*sc[8];

    /*reaction 28: HO2 + OH <=> H2O + O2 */
    qf[27] = sc[4]*sc[8];
    qr[27] = sc[5]*sc[7];

    /*reaction 29: HO2 + H <=> 2.000000 OH */
    qf[28] = sc[1]*sc[8];
    qr[28] = pow(sc[4], 2.000000);

    /*reaction 30: HO2 + O <=> OH + O2 */
    qf[29] = sc[3]*sc[8];
    qr[29] = sc[4]*sc[5];

    /*reaction 31: HO2 + H <=> O + H2O */
    qf[30] = sc[1]*sc[8];
    qr[30] = sc[3]*sc[7];

    /*reaction 32: HO2 + OH <=> H2O + O2 */
    qf[31] = sc[4]*sc[8];
    qr[31] = sc[5]*sc[7];

    /*reaction 33: H2O2 + O <=> HO2 + OH */
    qf[32] = sc[3]*sc[6];
    qr[32] = sc[4]*sc[8];

    /*reaction 34: H2O2 + H <=> H2O + OH */
    qf[33] = sc[1]*sc[6];
    qr[33] = sc[4]*sc[7];

    /*reaction 35: H2O2 + H <=> HO2 + H2 */
    qf[34] = sc[1]*sc[6];
    qr[34] = sc[2]*sc[8];

    /*reaction 36: H2O2 + OH <=> HO2 + H2O */
    qf[35] = sc[4]*sc[6];
    qr[35] = sc[7]*sc[8];

    /*reaction 37: H2O2 + OH <=> HO2 + H2O */
    qf[36] = sc[4]*sc[6];
    qr[36] = sc[7]*sc[8];

    /*reaction 38: C + O2 <=> CO + O */
    qf[37] = sc[5]*qss_sc[0];
    qr[37] = sc[3]*sc[9];

    /*reaction 39: C + OH <=> CO + H */
    qf[38] = sc[4]*qss_sc[0];
    qr[38] = sc[1]*sc[9];

    /*reaction 40: CH + OH <=> HCO + H */
    qf[39] = sc[4]*qss_sc[1];
    qr[39] = sc[1]*qss_sc[2];

    /*reaction 41: CH + H2 <=> TXCH2 + H */
    qf[40] = sc[2]*qss_sc[1];
    qr[40] = sc[1]*qss_sc[3];

    /*reaction 42: CH + O <=> CO + H */
    qf[41] = sc[3]*qss_sc[1];
    qr[41] = sc[1]*sc[9];

    /*reaction 43: CH + O2 <=> HCO + O */
    qf[42] = sc[5]*qss_sc[1];
    qr[42] = sc[3]*qss_sc[2];

    /*reaction 44: CH + H <=> C + H2 */
    qf[43] = sc[1]*qss_sc[1];
    qr[43] = sc[2]*qss_sc[0];

    /*reaction 45: CH + H2O <=> CH2O + H */
    qf[44] = sc[7]*qss_sc[1];
    qr[44] = sc[1]*sc[11];

    /*reaction 46: TXCH2 + O2 => OH + H + CO */
    qf[45] = sc[5]*qss_sc[3];
    qr[45] = 0.0;

    /*reaction 47: TXCH2 + O2 <=> CH2O + O */
    qf[46] = sc[5]*qss_sc[3];
    qr[46] = sc[3]*sc[11];

    /*reaction 48: TXCH2 + OH <=> CH2O + H */
    qf[47] = sc[4]*qss_sc[3];
    qr[47] = sc[1]*sc[11];

    /*reaction 49: TXCH2 + HO2 <=> CH2O + OH */
    qf[48] = sc[8]*qss_sc[3];
    qr[48] = sc[4]*sc[11];

    /*reaction 50: TXCH2 + O2 => CO2 + 2.000000 H */
    qf[49] = sc[5]*qss_sc[3];
    qr[49] = 0.0;

    /*reaction 51: TXCH2 + OH <=> CH + H2O */
    qf[50] = sc[4]*qss_sc[3];
    qr[50] = sc[7]*qss_sc[1];

    /*reaction 52: TXCH2 + O <=> HCO + H */
    qf[51] = sc[3]*qss_sc[3];
    qr[51] = sc[1]*qss_sc[2];

    /*reaction 53: TXCH2 + H2 <=> H + CH3 */
    qf[52] = sc[2]*qss_sc[3];
    qr[52] = sc[1]*sc[10];

    /*reaction 54: SXCH2 + H2O <=> TXCH2 + H2O */
    qf[53] = sc[7]*qss_sc[4];
    qr[53] = sc[7]*qss_sc[3];

    /*reaction 55: SXCH2 + H <=> CH + H2 */
    qf[54] = sc[1]*qss_sc[4];
    qr[54] = sc[2]*qss_sc[1];

    /*reaction 56: SXCH2 + O2 <=> H + OH + CO */
    qf[55] = sc[5]*qss_sc[4];
    qr[55] = sc[1]*sc[4]*sc[9];

    /*reaction 57: SXCH2 + O <=> CO + H2 */
    qf[56] = sc[3]*qss_sc[4];
    qr[56] = sc[2]*sc[9];

    /*reaction 58: SXCH2 + O2 <=> CO + H2O */
    qf[57] = sc[5]*qss_sc[4];
    qr[57] = sc[7]*sc[9];

    /*reaction 59: SXCH2 + H2 <=> CH3 + H */
    qf[58] = sc[2]*qss_sc[4];
    qr[58] = sc[1]*sc[10];

    /*reaction 60: SXCH2 + O <=> HCO + H */
    qf[59] = sc[3]*qss_sc[4];
    qr[59] = sc[1]*qss_sc[2];

    /*reaction 61: SXCH2 + H2O => H2 + CH2O */
    qf[60] = sc[7]*qss_sc[4];
    qr[60] = 0.0;

    /*reaction 62: SXCH2 + OH <=> CH2O + H */
    qf[61] = sc[4]*qss_sc[4];
    qr[61] = sc[1]*sc[11];

    /*reaction 63: CH3 + OH => H2 + CH2O */
    qf[62] = sc[4]*sc[10];
    qr[62] = 0.0;

    /*reaction 64: CH3 + H2O2 <=> CH4 + HO2 */
    qf[63] = sc[6]*sc[10];
    qr[63] = sc[8]*sc[13];

    /*reaction 65: CH3 + O2 <=> CH2O + OH */
    qf[64] = sc[5]*sc[10];
    qr[64] = sc[4]*sc[11];

    /*reaction 66: CH3 + CH <=> C2H3 + H */
    qf[65] = sc[10]*qss_sc[1];
    qr[65] = sc[1]*sc[18];

    /*reaction 67: CH3 + O <=> CH2O + H */
    qf[66] = sc[3]*sc[10];
    qr[66] = sc[1]*sc[11];

    /*reaction 68: CH3 + C <=> C2H2 + H */
    qf[67] = sc[10]*qss_sc[0];
    qr[67] = sc[1]*sc[14];

    /*reaction 69: CH3 + OH <=> TXCH2 + H2O */
    qf[68] = sc[4]*sc[10];
    qr[68] = sc[7]*qss_sc[3];

    /*reaction 70: CH3 + SXCH2 <=> C2H4 + H */
    qf[69] = sc[10]*qss_sc[4];
    qr[69] = sc[1]*sc[15];

    /*reaction 71: CH3 + OH <=> SXCH2 + H2O */
    qf[70] = sc[4]*sc[10];
    qr[70] = sc[7]*qss_sc[4];

    /*reaction 72: 2.000000 CH3 <=> C2H5 + H */
    qf[71] = pow(sc[10], 2.000000);
    qr[71] = sc[1]*sc[19];

    /*reaction 73: CH3 + HO2 <=> CH4 + O2 */
    qf[72] = sc[8]*sc[10];
    qr[72] = sc[5]*sc[13];

    /*reaction 74: CH3 + TXCH2 <=> C2H4 + H */
    qf[73] = sc[10]*qss_sc[3];
    qr[73] = sc[1]*sc[15];

    /*reaction 75: CH3 + O => H + H2 + CO */
    qf[74] = sc[3]*sc[10];
    qr[74] = 0.0;

    /*reaction 76: CH4 + CH <=> C2H4 + H */
    qf[75] = sc[13]*qss_sc[1];
    qr[75] = sc[1]*sc[15];

    /*reaction 77: CH4 + SXCH2 <=> 2.000000 CH3 */
    qf[76] = sc[13]*qss_sc[4];
    qr[76] = pow(sc[10], 2.000000);

    /*reaction 78: CH4 + O <=> CH3 + OH */
    qf[77] = sc[3]*sc[13];
    qr[77] = sc[4]*sc[10];

    /*reaction 79: CH4 + OH <=> CH3 + H2O */
    qf[78] = sc[4]*sc[13];
    qr[78] = sc[7]*sc[10];

    /*reaction 80: CH4 + TXCH2 <=> 2.000000 CH3 */
    qf[79] = sc[13]*qss_sc[3];
    qr[79] = pow(sc[10], 2.000000);

    /*reaction 81: CH4 + H <=> CH3 + H2 */
    qf[80] = sc[1]*sc[13];
    qr[80] = sc[2]*sc[10];

    /*reaction 82: SXCH2 + CO <=> TXCH2 + CO */
    qf[81] = sc[9]*qss_sc[4];
    qr[81] = sc[9]*qss_sc[3];

    /*reaction 83: CO + O2 <=> CO2 + O */
    qf[82] = sc[5]*sc[9];
    qr[82] = sc[3]*sc[12];

    /*reaction 84: CO + OH <=> CO2 + H */
    qf[83] = sc[4]*sc[9];
    qr[83] = sc[1]*sc[12];

    /*reaction 85: CO + OH <=> CO2 + H */
    qf[84] = sc[4]*sc[9];
    qr[84] = sc[1]*sc[12];

    /*reaction 86: CO + HO2 <=> CO2 + OH */
    qf[85] = sc[8]*sc[9];
    qr[85] = sc[4]*sc[12];

    /*reaction 87: HCO + H <=> CO + H2 */
    qf[86] = sc[1]*qss_sc[2];
    qr[86] = sc[2]*sc[9];

    /*reaction 88: CH3 + HCO <=> CH3CHO */
    qf[87] = sc[10]*qss_sc[2];
    qr[87] = sc[21];

    /*reaction 89: HCO + H2O <=> CO + H + H2O */
    qf[88] = sc[7]*qss_sc[2];
    qr[88] = sc[1]*sc[7]*sc[9];

    /*reaction 90: HCO + O <=> CO + OH */
    qf[89] = sc[3]*qss_sc[2];
    qr[89] = sc[4]*sc[9];

    /*reaction 91: HCO + OH <=> CO + H2O */
    qf[90] = sc[4]*qss_sc[2];
    qr[90] = sc[7]*sc[9];

    /*reaction 92: CH3 + HCO <=> CH4 + CO */
    qf[91] = sc[10]*qss_sc[2];
    qr[91] = sc[9]*sc[13];

    /*reaction 93: HCO + O <=> CO2 + H */
    qf[92] = sc[3]*qss_sc[2];
    qr[92] = sc[1]*sc[12];

    /*reaction 94: HCO + O2 <=> CO + HO2 */
    qf[93] = sc[5]*qss_sc[2];
    qr[93] = sc[8]*sc[9];

    /*reaction 95: CH2O + H <=> HCO + H2 */
    qf[94] = sc[1]*sc[11];
    qr[94] = sc[2]*qss_sc[2];

    /*reaction 96: CH2O + O <=> HCO + OH */
    qf[95] = sc[3]*sc[11];
    qr[95] = sc[4]*qss_sc[2];

    /*reaction 97: CH3 + CH2O <=> CH4 + HCO */
    qf[96] = sc[10]*sc[11];
    qr[96] = sc[13]*qss_sc[2];

    /*reaction 98: CH2O + OH <=> HCO + H2O */
    qf[97] = sc[4]*sc[11];
    qr[97] = sc[7]*qss_sc[2];

    /*reaction 99: CH2O + CH <=> CH2CO + H */
    qf[98] = sc[11]*qss_sc[1];
    qr[98] = sc[1]*sc[16];

    /*reaction 100: CH2O + O2 <=> HCO + HO2 */
    qf[99] = sc[5]*sc[11];
    qr[99] = sc[8]*qss_sc[2];

    /*reaction 101: CH2O + HO2 <=> HCO + H2O2 */
    qf[100] = sc[8]*sc[11];
    qr[100] = sc[6]*qss_sc[2];

    /*reaction 102: 2.000000 H + CO2 <=> H2 + CO2 */
    qf[101] = pow(sc[1], 2.000000)*sc[12];
    qr[101] = sc[2]*sc[12];

    /*reaction 103: SXCH2 + CO2 <=> TXCH2 + CO2 */
    qf[102] = sc[12]*qss_sc[4];
    qr[102] = sc[12]*qss_sc[3];

    /*reaction 104: SXCH2 + CO2 <=> CH2O + CO */
    qf[103] = sc[12]*qss_sc[4];
    qr[103] = sc[9]*sc[11];

    /*reaction 105: CH + CO2 <=> HCO + CO */
    qf[104] = sc[12]*qss_sc[1];
    qr[104] = sc[9]*qss_sc[2];

    /*reaction 106: C2H2 + O <=> TXCH2 + CO */
    qf[105] = sc[3]*sc[14];
    qr[105] = sc[9]*qss_sc[3];

    /*reaction 107: C2H2 + OH <=> CH3 + CO */
    qf[106] = sc[4]*sc[14];
    qr[106] = sc[9]*sc[10];

    /*reaction 108: C2H2 + OH <=> CH2CO + H */
    qf[107] = sc[4]*sc[14];
    qr[107] = sc[1]*sc[16];

    /*reaction 109: C2H2 + O <=> HCCO + H */
    qf[108] = sc[3]*sc[14];
    qr[108] = sc[1]*sc[20];

    /*reaction 110: C2H3 + OH <=> C2H2 + H2O */
    qf[109] = sc[4]*sc[18];
    qr[109] = sc[7]*sc[14];

    /*reaction 111: C2H3 + O2 <=> CH2CHO + O */
    qf[110] = sc[5]*sc[18];
    qr[110] = sc[3]*sc[22];

    /*reaction 112: C2H3 + O <=> CH2CHO */
    qf[111] = sc[3]*sc[18];
    qr[111] = sc[22];

    /*reaction 113: C2H3 + H <=> C2H2 + H2 */
    qf[112] = sc[1]*sc[18];
    qr[112] = sc[2]*sc[14];

    /*reaction 114: C2H3 + CH3 <=> C2H2 + CH4 */
    qf[113] = sc[10]*sc[18];
    qr[113] = sc[13]*sc[14];

    /*reaction 115: C2H3 + O2 <=> HCO + CH2O */
    qf[114] = sc[5]*sc[18];
    qr[114] = sc[11]*qss_sc[2];

    /*reaction 116: C2H3 + H2O2 <=> C2H4 + HO2 */
    qf[115] = sc[6]*sc[18];
    qr[115] = sc[8]*sc[15];

    /*reaction 117: C2H3 + O2 <=> C2H2 + HO2 */
    qf[116] = sc[5]*sc[18];
    qr[116] = sc[8]*sc[14];

    /*reaction 118: C2H4 + CH3 <=> C2H3 + CH4 */
    qf[117] = sc[10]*sc[15];
    qr[117] = sc[13]*sc[18];

    /*reaction 119: C2H4 + O2 => CH3 + CO2 + H */
    qf[118] = sc[5]*sc[15];
    qr[118] = 0.0;

    /*reaction 120: C2H4 + OH <=> C2H3 + H2O */
    qf[119] = sc[4]*sc[15];
    qr[119] = sc[7]*sc[18];

    /*reaction 121: C2H4 + OH <=> C2H5O */
    qf[120] = sc[4]*sc[15];
    qr[120] = sc[23];

    /*reaction 122: C2H4 + O <=> CH2CHO + H */
    qf[121] = sc[3]*sc[15];
    qr[121] = sc[1]*sc[22];

    /*reaction 123: C2H4 + O <=> CH3 + HCO */
    qf[122] = sc[3]*sc[15];
    qr[122] = sc[10]*qss_sc[2];

    /*reaction 124: C2H4 + O2 <=> C2H3 + HO2 */
    qf[123] = sc[5]*sc[15];
    qr[123] = sc[8]*sc[18];

    /*reaction 125: C2H4 + H <=> C2H3 + H2 */
    qf[124] = sc[1]*sc[15];
    qr[124] = sc[2]*sc[18];

    /*reaction 126: C2H4 + O <=> TXCH2 + CH2O */
    qf[125] = sc[3]*sc[15];
    qr[125] = sc[11]*qss_sc[3];

    /*reaction 127: C2H5 + HO2 <=> C2H4 + H2O2 */
    qf[126] = sc[8]*sc[19];
    qr[126] = sc[6]*sc[15];

    /*reaction 128: C2H5 + HO2 <=> C2H5O + OH */
    qf[127] = sc[8]*sc[19];
    qr[127] = sc[4]*sc[23];

    /*reaction 129: C2H5 + O <=> C2H5O */
    qf[128] = sc[3]*sc[19];
    qr[128] = sc[23];

    /*reaction 130: C2H5 + H <=> C2H4 + H2 */
    qf[129] = sc[1]*sc[19];
    qr[129] = sc[2]*sc[15];

    /*reaction 131: C2H5 + O2 <=> C2H4 + HO2 */
    qf[130] = sc[5]*sc[19];
    qr[130] = sc[8]*sc[15];

    /*reaction 132: C2H5 + HO2 <=> C2H6 + O2 */
    qf[131] = sc[8]*sc[19];
    qr[131] = sc[5]*sc[17];

    /*reaction 133: C2H5 + CH3 <=> C2H4 + CH4 */
    qf[132] = sc[10]*sc[19];
    qr[132] = sc[13]*sc[15];

    /*reaction 134: C2H6 + SXCH2 <=> C2H5 + CH3 */
    qf[133] = sc[17]*qss_sc[4];
    qr[133] = sc[10]*sc[19];

    /*reaction 135: C2H6 + CH3 <=> C2H5 + CH4 */
    qf[134] = sc[10]*sc[17];
    qr[134] = sc[13]*sc[19];

    /*reaction 136: C2H6 + O <=> C2H5 + OH */
    qf[135] = sc[3]*sc[17];
    qr[135] = sc[4]*sc[19];

    /*reaction 137: C2H6 + HO2 <=> C2H5 + H2O2 */
    qf[136] = sc[8]*sc[17];
    qr[136] = sc[6]*sc[19];

    /*reaction 138: C2H6 + H <=> C2H5 + H2 */
    qf[137] = sc[1]*sc[17];
    qr[137] = sc[2]*sc[19];

    /*reaction 139: C2H6 + OH <=> C2H5 + H2O */
    qf[138] = sc[4]*sc[17];
    qr[138] = sc[7]*sc[19];

    /*reaction 140: HCCO + O2 <=> OH + 2.000000 CO */
    qf[139] = sc[5]*sc[20];
    qr[139] = sc[4]*pow(sc[9], 2.000000);

    /*reaction 141: HCCO + O <=> H + 2.000000 CO */
    qf[140] = sc[3]*sc[20];
    qr[140] = sc[1]*pow(sc[9], 2.000000);

    /*reaction 142: HCCO + CH3 <=> C2H4 + CO */
    qf[141] = sc[10]*sc[20];
    qr[141] = sc[9]*sc[15];

    /*reaction 143: HCCO + H <=> SXCH2 + CO */
    qf[142] = sc[1]*sc[20];
    qr[142] = sc[9]*qss_sc[4];

    /*reaction 144: CH2CO + H <=> CH3 + CO */
    qf[143] = sc[1]*sc[16];
    qr[143] = sc[9]*sc[10];

    /*reaction 145: CH2CO + TXCH2 <=> C2H4 + CO */
    qf[144] = sc[16]*qss_sc[3];
    qr[144] = sc[9]*sc[15];

    /*reaction 146: CH2CO + O <=> HCCO + OH */
    qf[145] = sc[3]*sc[16];
    qr[145] = sc[4]*sc[20];

    /*reaction 147: CH2CO + CH3 <=> HCCO + CH4 */
    qf[146] = sc[10]*sc[16];
    qr[146] = sc[13]*sc[20];

    /*reaction 148: CH2CO + O <=> TXCH2 + CO2 */
    qf[147] = sc[3]*sc[16];
    qr[147] = sc[12]*qss_sc[3];

    /*reaction 149: CH2CO + CH3 <=> C2H5 + CO */
    qf[148] = sc[10]*sc[16];
    qr[148] = sc[9]*sc[19];

    /*reaction 150: CH2CO + OH <=> HCCO + H2O */
    qf[149] = sc[4]*sc[16];
    qr[149] = sc[7]*sc[20];

    /*reaction 151: CH2CO + H <=> HCCO + H2 */
    qf[150] = sc[1]*sc[16];
    qr[150] = sc[2]*sc[20];

    /*reaction 152: CH2CO + TXCH2 <=> HCCO + CH3 */
    qf[151] = sc[16]*qss_sc[3];
    qr[151] = sc[10]*sc[20];

    /*reaction 153: CH2CHO + O <=> CH2O + HCO */
    qf[152] = sc[3]*sc[22];
    qr[152] = sc[11]*qss_sc[2];

    /*reaction 154: CH2CHO <=> CH2CO + H */
    qf[153] = sc[22];
    qr[153] = sc[1]*sc[16];

    /*reaction 155: CH2CHO + OH <=> H2O + CH2CO */
    qf[154] = sc[4]*sc[22];
    qr[154] = sc[7]*sc[16];

    /*reaction 156: CH2CHO + H <=> CH2CO + H2 */
    qf[155] = sc[1]*sc[22];
    qr[155] = sc[2]*sc[16];

    /*reaction 157: CH2CHO + O2 => OH + CO + CH2O */
    qf[156] = sc[5]*sc[22];
    qr[156] = 0.0;

    /*reaction 158: CH2CHO <=> CH3 + CO */
    qf[157] = sc[22];
    qr[157] = sc[9]*sc[10];

    /*reaction 159: CH2CHO + O2 => OH + 2.000000 HCO */
    qf[158] = sc[5]*sc[22];
    qr[158] = 0.0;

    /*reaction 160: CH2CHO + H <=> CH3 + HCO */
    qf[159] = sc[1]*sc[22];
    qr[159] = sc[10]*qss_sc[2];

    /*reaction 161: CH3CHO + O => CH3 + CO + OH */
    qf[160] = sc[3]*sc[21];
    qr[160] = 0.0;

    /*reaction 162: CH3CHO + O2 => CH3 + CO + HO2 */
    qf[161] = sc[5]*sc[21];
    qr[161] = 0.0;

    /*reaction 163: CH3CHO + OH => CH3 + CO + H2O */
    qf[162] = sc[4]*sc[21];
    qr[162] = 0.0;

    /*reaction 164: CH3CHO + H <=> CH2CHO + H2 */
    qf[163] = sc[1]*sc[21];
    qr[163] = sc[2]*sc[22];

    /*reaction 165: CH3CHO + H => CH3 + CO + H2 */
    qf[164] = sc[1]*sc[21];
    qr[164] = 0.0;

    /*reaction 166: CH3CHO + O <=> CH2CHO + OH */
    qf[165] = sc[3]*sc[21];
    qr[165] = sc[4]*sc[22];

    /*reaction 167: CH3CHO + CH3 => CH3 + CO + CH4 */
    qf[166] = sc[10]*sc[21];
    qr[166] = 0.0;

    /*reaction 168: CH3CHO + HO2 => CH3 + CO + H2O2 */
    qf[167] = sc[8]*sc[21];
    qr[167] = 0.0;

    /*reaction 169: C2H5O <=> CH3 + CH2O */
    qf[168] = sc[23];
    qr[168] = sc[10]*sc[11];

    /*reaction 170: C2H5O <=> CH3CHO + H */
    qf[169] = sc[23];
    qr[169] = sc[1]*sc[21];

    /*reaction 171: C2H5O + O2 <=> CH3CHO + HO2 */
    qf[170] = sc[5]*sc[23];
    qr[170] = sc[8]*sc[21];

    /*reaction 172: SXCH2 + N2 <=> TXCH2 + N2 */
    qf[171] = sc[0]*qss_sc[4];
    qr[171] = sc[0]*qss_sc[3];

    double T = tc[1];

    /*compute the mixture concentration */
    double mixture = 0.0;
    for (int i = 0; i < 24; ++i) {
        mixture += sc[i];
    }

    double Corr[172];
    for (int i = 0; i < 172; ++i) {
        Corr[i] = 1.0;
    }

    /* troe */
    {
        double alpha[15];
        alpha[0] = mixture + (TB[0][0] - 1)*sc[2] + (TB[0][1] - 1)*sc[7] + (TB[0][2] - 1)*sc[9] + (TB[0][3] - 1)*sc[12] + (TB[0][4] - 1)*sc[13] + (TB[0][5] - 1)*sc[17];
        alpha[1] = mixture + (TB[1][0] - 1)*sc[2] + (TB[1][1] - 1)*sc[5] + (TB[1][2] - 1)*sc[7] + (TB[1][3] - 1)*sc[9] + (TB[1][4] - 1)*sc[12];
        alpha[2] = mixture + (TB[2][0] - 1)*sc[2] + (TB[2][1] - 1)*sc[7] + (TB[2][2] - 1)*sc[9] + (TB[2][3] - 1)*sc[12] + (TB[2][4] - 1)*sc[13] + (TB[2][5] - 1)*sc[17];
        alpha[3] = mixture + (TB[3][0] - 1)*sc[2] + (TB[3][1] - 1)*sc[7] + (TB[3][2] - 1)*sc[9] + (TB[3][3] - 1)*sc[12] + (TB[3][4] - 1)*sc[13] + (TB[3][5] - 1)*sc[17];
        alpha[4] = mixture + (TB[4][0] - 1)*sc[2] + (TB[4][1] - 1)*sc[7] + (TB[4][2] - 1)*sc[9] + (TB[4][3] - 1)*sc[12] + (TB[4][4] - 1)*sc[13] + (TB[4][5] - 1)*sc[17];
        alpha[5] = mixture + (TB[5][0] - 1)*sc[2] + (TB[5][1] - 1)*sc[7] + (TB[5][2] - 1)*sc[9] + (TB[5][3] - 1)*sc[12] + (TB[5][4] - 1)*sc[13] + (TB[5][5] - 1)*sc[17];
        alpha[6] = mixture + (TB[6][0] - 1)*sc[2] + (TB[6][1] - 1)*sc[7] + (TB[6][2] - 1)*sc[9] + (TB[6][3] - 1)*sc[12] + (TB[6][4] - 1)*sc[13] + (TB[6][5] - 1)*sc[17];
        alpha[7] = mixture + (TB[7][0] - 1)*sc[2] + (TB[7][1] - 1)*sc[7] + (TB[7][2] - 1)*sc[9] + (TB[7][3] - 1)*sc[12] + (TB[7][4] - 1)*sc[13] + (TB[7][5] - 1)*sc[17];
        alpha[8] = mixture + (TB[8][0] - 1)*sc[2] + (TB[8][1] - 1)*sc[7] + (TB[8][2] - 1)*sc[9] + (TB[8][3] - 1)*sc[12] + (TB[8][4] - 1)*sc[13] + (TB[8][5] - 1)*sc[17];
        alpha[9] = mixture + (TB[9][0] - 1)*sc[2] + (TB[9][1] - 1)*sc[7] + (TB[9][2] - 1)*sc[9] + (TB[9][3] - 1)*sc[12] + (TB[9][4] - 1)*sc[13] + (TB[9][5] - 1)*sc[17];
        alpha[10] = mixture + (TB[10][0] - 1)*sc[2] + (TB[10][1] - 1)*sc[7] + (TB[10][2] - 1)*sc[9] + (TB[10][3] - 1)*sc[12] + (TB[10][4] - 1)*sc[13] + (TB[10][5] - 1)*sc[17];
        alpha[11] = mixture + (TB[11][0] - 1)*sc[2] + (TB[11][1] - 1)*sc[7] + (TB[11][2] - 1)*sc[9] + (TB[11][3] - 1)*sc[12] + (TB[11][4] - 1)*sc[13] + (TB[11][5] - 1)*sc[17];
        alpha[12] = mixture + (TB[12][0] - 1)*sc[2] + (TB[12][1] - 1)*sc[7] + (TB[12][2] - 1)*sc[9] + (TB[12][3] - 1)*sc[12] + (TB[12][4] - 1)*sc[13] + (TB[12][5] - 1)*sc[17];
        alpha[13] = mixture + (TB[13][0] - 1)*sc[2] + (TB[13][1] - 1)*sc[7] + (TB[13][2] - 1)*sc[9] + (TB[13][3] - 1)*sc[12] + (TB[13][4] - 1)*sc[13] + (TB[13][5] - 1)*sc[17];
        alpha[14] = mixture + (TB[14][0] - 1)*sc[2] + (TB[14][1] - 1)*sc[7] + (TB[14][2] - 1)*sc[9] + (TB[14][3] - 1)*sc[12] + (TB[14][4] - 1)*sc[13] + (TB[14][5] - 1)*sc[17];
#ifdef __INTEL_COMPILER
         #pragma simd
#endif
        for (int i=0; i<15; i++)
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
        alpha = mixture + (TB[15][0] - 1)*sc[2] + (TB[15][1] - 1)*sc[7] + (TB[15][2] - 1)*sc[12] + (TB[15][3] - 1)*sc[13] + (TB[15][4] - 1)*sc[17];
        Corr[15] = alpha;
        alpha = mixture + (TB[16][0] - 1)*sc[2] + (TB[16][1] - 1)*sc[7] + (TB[16][2] - 1)*sc[9] + (TB[16][3] - 1)*sc[12] + (TB[16][4] - 1)*sc[13] + (TB[16][5] - 1)*sc[17];
        Corr[16] = alpha;
        alpha = mixture + (TB[17][0] - 1)*sc[2] + (TB[17][1] - 1)*sc[7] + (TB[17][2] - 1)*sc[9] + (TB[17][3] - 1)*sc[12] + (TB[17][4] - 1)*sc[13] + (TB[17][5] - 1)*sc[17];
        Corr[17] = alpha;
        alpha = mixture + (TB[18][0] - 1)*sc[2] + (TB[18][1] - 1)*sc[7] + (TB[18][2] - 1)*sc[9] + (TB[18][3] - 1)*sc[12] + (TB[18][4] - 1)*sc[13] + (TB[18][5] - 1)*sc[17];
        Corr[18] = alpha;
        alpha = mixture + (TB[19][0] - 1)*sc[2] + (TB[19][1] - 1)*sc[7] + (TB[19][2] - 1)*sc[9] + (TB[19][3] - 1)*sc[12] + (TB[19][4] - 1)*sc[13] + (TB[19][5] - 1)*sc[17];
        Corr[19] = alpha;
    }

    for (int i=0; i<172; i++)
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
    k_f[0] = prefactor_units[2] * fwd_A[2]
               * exp(fwd_beta[2] * tc[0] - activation_units[2] * fwd_Ea[2] * invT);
    k_f[1] = prefactor_units[3] * fwd_A[3]
               * exp(fwd_beta[3] * tc[0] - activation_units[3] * fwd_Ea[3] * invT);
    k_f[2] = prefactor_units[5] * fwd_A[5]
               * exp(fwd_beta[5] * tc[0] - activation_units[5] * fwd_Ea[5] * invT);
    k_f[3] = prefactor_units[7] * fwd_A[7]
               * exp(fwd_beta[7] * tc[0] - activation_units[7] * fwd_Ea[7] * invT);
    k_f[4] = prefactor_units[9] * fwd_A[9]
               * exp(fwd_beta[9] * tc[0] - activation_units[9] * fwd_Ea[9] * invT);
    k_f[5] = prefactor_units[19] * fwd_A[19]
               * exp(fwd_beta[19] * tc[0] - activation_units[19] * fwd_Ea[19] * invT);
    k_f[6] = prefactor_units[37] * fwd_A[37]
               * exp(fwd_beta[37] * tc[0] - activation_units[37] * fwd_Ea[37] * invT);
    k_f[7] = prefactor_units[38] * fwd_A[38]
               * exp(fwd_beta[38] * tc[0] - activation_units[38] * fwd_Ea[38] * invT);
    k_f[8] = prefactor_units[39] * fwd_A[39]
               * exp(fwd_beta[39] * tc[0] - activation_units[39] * fwd_Ea[39] * invT);
    k_f[9] = prefactor_units[40] * fwd_A[40]
               * exp(fwd_beta[40] * tc[0] - activation_units[40] * fwd_Ea[40] * invT);
    k_f[10] = prefactor_units[41] * fwd_A[41]
               * exp(fwd_beta[41] * tc[0] - activation_units[41] * fwd_Ea[41] * invT);
    k_f[11] = prefactor_units[42] * fwd_A[42]
               * exp(fwd_beta[42] * tc[0] - activation_units[42] * fwd_Ea[42] * invT);
    k_f[12] = prefactor_units[43] * fwd_A[43]
               * exp(fwd_beta[43] * tc[0] - activation_units[43] * fwd_Ea[43] * invT);
    k_f[13] = prefactor_units[44] * fwd_A[44]
               * exp(fwd_beta[44] * tc[0] - activation_units[44] * fwd_Ea[44] * invT);
    k_f[14] = prefactor_units[45] * fwd_A[45]
               * exp(fwd_beta[45] * tc[0] - activation_units[45] * fwd_Ea[45] * invT);
    k_f[15] = prefactor_units[46] * fwd_A[46]
               * exp(fwd_beta[46] * tc[0] - activation_units[46] * fwd_Ea[46] * invT);
    k_f[16] = prefactor_units[47] * fwd_A[47]
               * exp(fwd_beta[47] * tc[0] - activation_units[47] * fwd_Ea[47] * invT);
    k_f[17] = prefactor_units[48] * fwd_A[48]
               * exp(fwd_beta[48] * tc[0] - activation_units[48] * fwd_Ea[48] * invT);
    k_f[18] = prefactor_units[49] * fwd_A[49]
               * exp(fwd_beta[49] * tc[0] - activation_units[49] * fwd_Ea[49] * invT);
    k_f[19] = prefactor_units[50] * fwd_A[50]
               * exp(fwd_beta[50] * tc[0] - activation_units[50] * fwd_Ea[50] * invT);
    k_f[20] = prefactor_units[51] * fwd_A[51]
               * exp(fwd_beta[51] * tc[0] - activation_units[51] * fwd_Ea[51] * invT);
    k_f[21] = prefactor_units[52] * fwd_A[52]
               * exp(fwd_beta[52] * tc[0] - activation_units[52] * fwd_Ea[52] * invT);
    k_f[22] = prefactor_units[53] * fwd_A[53]
               * exp(fwd_beta[53] * tc[0] - activation_units[53] * fwd_Ea[53] * invT);
    k_f[23] = prefactor_units[54] * fwd_A[54]
               * exp(fwd_beta[54] * tc[0] - activation_units[54] * fwd_Ea[54] * invT);
    k_f[24] = prefactor_units[55] * fwd_A[55]
               * exp(fwd_beta[55] * tc[0] - activation_units[55] * fwd_Ea[55] * invT);
    k_f[25] = prefactor_units[56] * fwd_A[56]
               * exp(fwd_beta[56] * tc[0] - activation_units[56] * fwd_Ea[56] * invT);
    k_f[26] = prefactor_units[57] * fwd_A[57]
               * exp(fwd_beta[57] * tc[0] - activation_units[57] * fwd_Ea[57] * invT);
    k_f[27] = prefactor_units[58] * fwd_A[58]
               * exp(fwd_beta[58] * tc[0] - activation_units[58] * fwd_Ea[58] * invT);
    k_f[28] = prefactor_units[59] * fwd_A[59]
               * exp(fwd_beta[59] * tc[0] - activation_units[59] * fwd_Ea[59] * invT);
    k_f[29] = prefactor_units[60] * fwd_A[60]
               * exp(fwd_beta[60] * tc[0] - activation_units[60] * fwd_Ea[60] * invT);
    k_f[30] = prefactor_units[61] * fwd_A[61]
               * exp(fwd_beta[61] * tc[0] - activation_units[61] * fwd_Ea[61] * invT);
    k_f[31] = prefactor_units[65] * fwd_A[65]
               * exp(fwd_beta[65] * tc[0] - activation_units[65] * fwd_Ea[65] * invT);
    k_f[32] = prefactor_units[67] * fwd_A[67]
               * exp(fwd_beta[67] * tc[0] - activation_units[67] * fwd_Ea[67] * invT);
    k_f[33] = prefactor_units[68] * fwd_A[68]
               * exp(fwd_beta[68] * tc[0] - activation_units[68] * fwd_Ea[68] * invT);
    k_f[34] = prefactor_units[69] * fwd_A[69]
               * exp(fwd_beta[69] * tc[0] - activation_units[69] * fwd_Ea[69] * invT);
    k_f[35] = prefactor_units[70] * fwd_A[70]
               * exp(fwd_beta[70] * tc[0] - activation_units[70] * fwd_Ea[70] * invT);
    k_f[36] = prefactor_units[73] * fwd_A[73]
               * exp(fwd_beta[73] * tc[0] - activation_units[73] * fwd_Ea[73] * invT);
    k_f[37] = prefactor_units[75] * fwd_A[75]
               * exp(fwd_beta[75] * tc[0] - activation_units[75] * fwd_Ea[75] * invT);
    k_f[38] = prefactor_units[76] * fwd_A[76]
               * exp(fwd_beta[76] * tc[0] - activation_units[76] * fwd_Ea[76] * invT);
    k_f[39] = prefactor_units[79] * fwd_A[79]
               * exp(fwd_beta[79] * tc[0] - activation_units[79] * fwd_Ea[79] * invT);
    k_f[40] = prefactor_units[81] * fwd_A[81]
               * exp(fwd_beta[81] * tc[0] - activation_units[81] * fwd_Ea[81] * invT);
    k_f[41] = prefactor_units[86] * fwd_A[86]
               * exp(fwd_beta[86] * tc[0] - activation_units[86] * fwd_Ea[86] * invT);
    k_f[42] = prefactor_units[87] * fwd_A[87]
               * exp(fwd_beta[87] * tc[0] - activation_units[87] * fwd_Ea[87] * invT);
    k_f[43] = prefactor_units[88] * fwd_A[88]
               * exp(fwd_beta[88] * tc[0] - activation_units[88] * fwd_Ea[88] * invT);
    k_f[44] = prefactor_units[89] * fwd_A[89]
               * exp(fwd_beta[89] * tc[0] - activation_units[89] * fwd_Ea[89] * invT);
    k_f[45] = prefactor_units[90] * fwd_A[90]
               * exp(fwd_beta[90] * tc[0] - activation_units[90] * fwd_Ea[90] * invT);
    k_f[46] = prefactor_units[91] * fwd_A[91]
               * exp(fwd_beta[91] * tc[0] - activation_units[91] * fwd_Ea[91] * invT);
    k_f[47] = prefactor_units[92] * fwd_A[92]
               * exp(fwd_beta[92] * tc[0] - activation_units[92] * fwd_Ea[92] * invT);
    k_f[48] = prefactor_units[93] * fwd_A[93]
               * exp(fwd_beta[93] * tc[0] - activation_units[93] * fwd_Ea[93] * invT);
    k_f[49] = prefactor_units[94] * fwd_A[94]
               * exp(fwd_beta[94] * tc[0] - activation_units[94] * fwd_Ea[94] * invT);
    k_f[50] = prefactor_units[95] * fwd_A[95]
               * exp(fwd_beta[95] * tc[0] - activation_units[95] * fwd_Ea[95] * invT);
    k_f[51] = prefactor_units[96] * fwd_A[96]
               * exp(fwd_beta[96] * tc[0] - activation_units[96] * fwd_Ea[96] * invT);
    k_f[52] = prefactor_units[97] * fwd_A[97]
               * exp(fwd_beta[97] * tc[0] - activation_units[97] * fwd_Ea[97] * invT);
    k_f[53] = prefactor_units[98] * fwd_A[98]
               * exp(fwd_beta[98] * tc[0] - activation_units[98] * fwd_Ea[98] * invT);
    k_f[54] = prefactor_units[99] * fwd_A[99]
               * exp(fwd_beta[99] * tc[0] - activation_units[99] * fwd_Ea[99] * invT);
    k_f[55] = prefactor_units[100] * fwd_A[100]
               * exp(fwd_beta[100] * tc[0] - activation_units[100] * fwd_Ea[100] * invT);
    k_f[56] = prefactor_units[102] * fwd_A[102]
               * exp(fwd_beta[102] * tc[0] - activation_units[102] * fwd_Ea[102] * invT);
    k_f[57] = prefactor_units[103] * fwd_A[103]
               * exp(fwd_beta[103] * tc[0] - activation_units[103] * fwd_Ea[103] * invT);
    k_f[58] = prefactor_units[104] * fwd_A[104]
               * exp(fwd_beta[104] * tc[0] - activation_units[104] * fwd_Ea[104] * invT);
    k_f[59] = prefactor_units[105] * fwd_A[105]
               * exp(fwd_beta[105] * tc[0] - activation_units[105] * fwd_Ea[105] * invT);
    k_f[60] = prefactor_units[114] * fwd_A[114]
               * exp(fwd_beta[114] * tc[0] - activation_units[114] * fwd_Ea[114] * invT);
    k_f[61] = prefactor_units[122] * fwd_A[122]
               * exp(fwd_beta[122] * tc[0] - activation_units[122] * fwd_Ea[122] * invT);
    k_f[62] = prefactor_units[125] * fwd_A[125]
               * exp(fwd_beta[125] * tc[0] - activation_units[125] * fwd_Ea[125] * invT);
    k_f[63] = prefactor_units[133] * fwd_A[133]
               * exp(fwd_beta[133] * tc[0] - activation_units[133] * fwd_Ea[133] * invT);
    k_f[64] = prefactor_units[142] * fwd_A[142]
               * exp(fwd_beta[142] * tc[0] - activation_units[142] * fwd_Ea[142] * invT);
    k_f[65] = prefactor_units[144] * fwd_A[144]
               * exp(fwd_beta[144] * tc[0] - activation_units[144] * fwd_Ea[144] * invT);
    k_f[66] = prefactor_units[147] * fwd_A[147]
               * exp(fwd_beta[147] * tc[0] - activation_units[147] * fwd_Ea[147] * invT);
    k_f[67] = prefactor_units[151] * fwd_A[151]
               * exp(fwd_beta[151] * tc[0] - activation_units[151] * fwd_Ea[151] * invT);
    k_f[68] = prefactor_units[152] * fwd_A[152]
               * exp(fwd_beta[152] * tc[0] - activation_units[152] * fwd_Ea[152] * invT);
    k_f[69] = prefactor_units[158] * fwd_A[158]
               * exp(fwd_beta[158] * tc[0] - activation_units[158] * fwd_Ea[158] * invT);
    k_f[70] = prefactor_units[159] * fwd_A[159]
               * exp(fwd_beta[159] * tc[0] - activation_units[159] * fwd_Ea[159] * invT);
    k_f[71] = prefactor_units[171] * fwd_A[171]
               * exp(fwd_beta[171] * tc[0] - activation_units[171] * fwd_Ea[171] * invT);

    return;
}

void comp_Kc_qss(double *  tc, double invT, double *  Kc)
{
    /*compute the Gibbs free energy */
    double g_RT[24], g_RT_qss[5];
    gibbs(g_RT, tc);
    gibbs_qss(g_RT_qss, tc);

    /*Reaction 3 */
    Kc[0] = g_RT[2] - g_RT[10] + g_RT_qss[1];
    /*Reaction 4 */
    Kc[1] = g_RT[1] - g_RT[10] + g_RT_qss[3];
    /*Reaction 6 */
    Kc[2] = g_RT[9] - g_RT[16] + g_RT_qss[3];
    /*Reaction 8 */
    Kc[3] = g_RT[9] - g_RT[20] + g_RT_qss[1];
    /*Reaction 10 */
    Kc[4] = g_RT[1] - g_RT[11] + g_RT_qss[2];
    /*Reaction 20 */
    Kc[5] = -g_RT[1] - g_RT[9] + g_RT_qss[2];
    /*Reaction 38 */
    Kc[6] = -g_RT[3] + g_RT[5] - g_RT[9] + g_RT_qss[0];
    /*Reaction 39 */
    Kc[7] = -g_RT[1] + g_RT[4] - g_RT[9] + g_RT_qss[0];
    /*Reaction 40 */
    Kc[8] = -g_RT[1] + g_RT[4] + g_RT_qss[1] - g_RT_qss[2];
    /*Reaction 41 */
    Kc[9] = -g_RT[1] + g_RT[2] + g_RT_qss[1] - g_RT_qss[3];
    /*Reaction 42 */
    Kc[10] = -g_RT[1] + g_RT[3] - g_RT[9] + g_RT_qss[1];
    /*Reaction 43 */
    Kc[11] = -g_RT[3] + g_RT[5] + g_RT_qss[1] - g_RT_qss[2];
    /*Reaction 44 */
    Kc[12] = g_RT[1] - g_RT[2] - g_RT_qss[0] + g_RT_qss[1];
    /*Reaction 45 */
    Kc[13] = -g_RT[1] + g_RT[7] - g_RT[11] + g_RT_qss[1];
    /*Reaction 46 */
    Kc[14] = -g_RT[1] - g_RT[4] + g_RT[5] - g_RT[9] + g_RT_qss[3];
    /*Reaction 47 */
    Kc[15] = -g_RT[3] + g_RT[5] - g_RT[11] + g_RT_qss[3];
    /*Reaction 48 */
    Kc[16] = -g_RT[1] + g_RT[4] - g_RT[11] + g_RT_qss[3];
    /*Reaction 49 */
    Kc[17] = -g_RT[4] + g_RT[8] - g_RT[11] + g_RT_qss[3];
    /*Reaction 50 */
    Kc[18] = -2.000000*g_RT[1] + g_RT[5] - g_RT[12] + g_RT_qss[3];
    /*Reaction 51 */
    Kc[19] = g_RT[4] - g_RT[7] - g_RT_qss[1] + g_RT_qss[3];
    /*Reaction 52 */
    Kc[20] = -g_RT[1] + g_RT[3] - g_RT_qss[2] + g_RT_qss[3];
    /*Reaction 53 */
    Kc[21] = -g_RT[1] + g_RT[2] - g_RT[10] + g_RT_qss[3];
    /*Reaction 54 */
    Kc[22] = g_RT[7] - g_RT[7] - g_RT_qss[3] + g_RT_qss[4];
    /*Reaction 55 */
    Kc[23] = g_RT[1] - g_RT[2] - g_RT_qss[1] + g_RT_qss[4];
    /*Reaction 56 */
    Kc[24] = -g_RT[1] - g_RT[4] + g_RT[5] - g_RT[9] + g_RT_qss[4];
    /*Reaction 57 */
    Kc[25] = -g_RT[2] + g_RT[3] - g_RT[9] + g_RT_qss[4];
    /*Reaction 58 */
    Kc[26] = g_RT[5] - g_RT[7] - g_RT[9] + g_RT_qss[4];
    /*Reaction 59 */
    Kc[27] = -g_RT[1] + g_RT[2] - g_RT[10] + g_RT_qss[4];
    /*Reaction 60 */
    Kc[28] = -g_RT[1] + g_RT[3] - g_RT_qss[2] + g_RT_qss[4];
    /*Reaction 61 */
    Kc[29] = -g_RT[2] + g_RT[7] - g_RT[11] + g_RT_qss[4];
    /*Reaction 62 */
    Kc[30] = -g_RT[1] + g_RT[4] - g_RT[11] + g_RT_qss[4];
    /*Reaction 66 */
    Kc[31] = -g_RT[1] + g_RT[10] - g_RT[18] + g_RT_qss[1];
    /*Reaction 68 */
    Kc[32] = -g_RT[1] + g_RT[10] - g_RT[14] + g_RT_qss[0];
    /*Reaction 69 */
    Kc[33] = g_RT[4] - g_RT[7] + g_RT[10] - g_RT_qss[3];
    /*Reaction 70 */
    Kc[34] = -g_RT[1] + g_RT[10] - g_RT[15] + g_RT_qss[4];
    /*Reaction 71 */
    Kc[35] = g_RT[4] - g_RT[7] + g_RT[10] - g_RT_qss[4];
    /*Reaction 74 */
    Kc[36] = -g_RT[1] + g_RT[10] - g_RT[15] + g_RT_qss[3];
    /*Reaction 76 */
    Kc[37] = -g_RT[1] + g_RT[13] - g_RT[15] + g_RT_qss[1];
    /*Reaction 77 */
    Kc[38] = -2.000000*g_RT[10] + g_RT[13] + g_RT_qss[4];
    /*Reaction 80 */
    Kc[39] = -2.000000*g_RT[10] + g_RT[13] + g_RT_qss[3];
    /*Reaction 82 */
    Kc[40] = g_RT[9] - g_RT[9] - g_RT_qss[3] + g_RT_qss[4];
    /*Reaction 87 */
    Kc[41] = g_RT[1] - g_RT[2] - g_RT[9] + g_RT_qss[2];
    /*Reaction 88 */
    Kc[42] = g_RT[10] - g_RT[21] + g_RT_qss[2];
    /*Reaction 89 */
    Kc[43] = -g_RT[1] + g_RT[7] - g_RT[7] - g_RT[9] + g_RT_qss[2];
    /*Reaction 90 */
    Kc[44] = g_RT[3] - g_RT[4] - g_RT[9] + g_RT_qss[2];
    /*Reaction 91 */
    Kc[45] = g_RT[4] - g_RT[7] - g_RT[9] + g_RT_qss[2];
    /*Reaction 92 */
    Kc[46] = -g_RT[9] + g_RT[10] - g_RT[13] + g_RT_qss[2];
    /*Reaction 93 */
    Kc[47] = -g_RT[1] + g_RT[3] - g_RT[12] + g_RT_qss[2];
    /*Reaction 94 */
    Kc[48] = g_RT[5] - g_RT[8] - g_RT[9] + g_RT_qss[2];
    /*Reaction 95 */
    Kc[49] = g_RT[1] - g_RT[2] + g_RT[11] - g_RT_qss[2];
    /*Reaction 96 */
    Kc[50] = g_RT[3] - g_RT[4] + g_RT[11] - g_RT_qss[2];
    /*Reaction 97 */
    Kc[51] = g_RT[10] + g_RT[11] - g_RT[13] - g_RT_qss[2];
    /*Reaction 98 */
    Kc[52] = g_RT[4] - g_RT[7] + g_RT[11] - g_RT_qss[2];
    /*Reaction 99 */
    Kc[53] = -g_RT[1] + g_RT[11] - g_RT[16] + g_RT_qss[1];
    /*Reaction 100 */
    Kc[54] = g_RT[5] - g_RT[8] + g_RT[11] - g_RT_qss[2];
    /*Reaction 101 */
    Kc[55] = -g_RT[6] + g_RT[8] + g_RT[11] - g_RT_qss[2];
    /*Reaction 103 */
    Kc[56] = g_RT[12] - g_RT[12] - g_RT_qss[3] + g_RT_qss[4];
    /*Reaction 104 */
    Kc[57] = -g_RT[9] - g_RT[11] + g_RT[12] + g_RT_qss[4];
    /*Reaction 105 */
    Kc[58] = -g_RT[9] + g_RT[12] + g_RT_qss[1] - g_RT_qss[2];
    /*Reaction 106 */
    Kc[59] = g_RT[3] - g_RT[9] + g_RT[14] - g_RT_qss[3];
    /*Reaction 115 */
    Kc[60] = g_RT[5] - g_RT[11] + g_RT[18] - g_RT_qss[2];
    /*Reaction 123 */
    Kc[61] = g_RT[3] - g_RT[10] + g_RT[15] - g_RT_qss[2];
    /*Reaction 126 */
    Kc[62] = g_RT[3] - g_RT[11] + g_RT[15] - g_RT_qss[3];
    /*Reaction 134 */
    Kc[63] = -g_RT[10] + g_RT[17] - g_RT[19] + g_RT_qss[4];
    /*Reaction 143 */
    Kc[64] = g_RT[1] - g_RT[9] + g_RT[20] - g_RT_qss[4];
    /*Reaction 145 */
    Kc[65] = -g_RT[9] - g_RT[15] + g_RT[16] + g_RT_qss[3];
    /*Reaction 148 */
    Kc[66] = g_RT[3] - g_RT[12] + g_RT[16] - g_RT_qss[3];
    /*Reaction 152 */
    Kc[67] = -g_RT[10] + g_RT[16] - g_RT[20] + g_RT_qss[3];
    /*Reaction 153 */
    Kc[68] = g_RT[3] - g_RT[11] + g_RT[22] - g_RT_qss[2];
    /*Reaction 159 */
    Kc[69] = -g_RT[4] + g_RT[5] + g_RT[22] - 2.000000*g_RT_qss[2];
    /*Reaction 160 */
    Kc[70] = g_RT[1] - g_RT[10] + g_RT[22] - g_RT_qss[2];
    /*Reaction 172 */
    Kc[71] = g_RT[0] - g_RT[0] - g_RT_qss[3] + g_RT_qss[4];

#ifdef __INTEL_COMPILER
     #pragma simd
#endif
    for (int i=0; i<72; ++i) {
        Kc[i] = exp(Kc[i]);
    };

    /*reference concentration: P_atm / (RT) in inverse mol/m^3 */
    double refC = 101325 / 8.31446 * invT;
    double refCinv = 1 / refC;

    Kc[0] *= refCinv;
    Kc[1] *= refCinv;
    Kc[2] *= refCinv;
    Kc[3] *= refCinv;
    Kc[4] *= refCinv;
    Kc[5] *= refC;
    Kc[14] *= refC;
    Kc[18] *= refC;
    Kc[24] *= refC;
    Kc[42] *= refCinv;
    Kc[43] *= refC;
    Kc[69] *= refC;

    return;
}

void comp_qss_coeff(double *  qf_co, double *  qr_co, double *  sc, double *  tc, double invT)
{

    /*reaction 3: CH + H2 (+M) <=> CH3 (+M) */
    qf_co[0] = sc[2];
    qr_co[0] = sc[10];

    /*reaction 4: TXCH2 + H (+M) <=> CH3 (+M) */
    qf_co[1] = sc[1];
    qr_co[1] = sc[10];

    /*reaction 6: TXCH2 + CO (+M) <=> CH2CO (+M) */
    qf_co[2] = sc[9];
    qr_co[2] = sc[16];

    /*reaction 8: CH + CO (+M) <=> HCCO (+M) */
    qf_co[3] = sc[9];
    qr_co[3] = sc[20];

    /*reaction 10: HCO + H (+M) <=> CH2O (+M) */
    qf_co[4] = sc[1];
    qr_co[4] = sc[11];

    /*reaction 20: HCO + M <=> CO + H + M */
    qf_co[5] = 1.0;
    qr_co[5] = sc[1]*sc[9];

    /*reaction 38: C + O2 <=> CO + O */
    qf_co[6] = sc[5];
    qr_co[6] = sc[3]*sc[9];

    /*reaction 39: C + OH <=> CO + H */
    qf_co[7] = sc[4];
    qr_co[7] = sc[1]*sc[9];

    /*reaction 40: CH + OH <=> HCO + H */
    qf_co[8] = sc[4];
    qr_co[8] = sc[1];

    /*reaction 41: CH + H2 <=> TXCH2 + H */
    qf_co[9] = sc[2];
    qr_co[9] = sc[1];

    /*reaction 42: CH + O <=> CO + H */
    qf_co[10] = sc[3];
    qr_co[10] = sc[1]*sc[9];

    /*reaction 43: CH + O2 <=> HCO + O */
    qf_co[11] = sc[5];
    qr_co[11] = sc[3];

    /*reaction 44: CH + H <=> C + H2 */
    qf_co[12] = sc[1];
    qr_co[12] = sc[2];

    /*reaction 45: CH + H2O <=> CH2O + H */
    qf_co[13] = sc[7];
    qr_co[13] = sc[1]*sc[11];

    /*reaction 46: TXCH2 + O2 => OH + H + CO */
    qf_co[14] = sc[5];
    qr_co[14] = 0.0;

    /*reaction 47: TXCH2 + O2 <=> CH2O + O */
    qf_co[15] = sc[5];
    qr_co[15] = sc[3]*sc[11];

    /*reaction 48: TXCH2 + OH <=> CH2O + H */
    qf_co[16] = sc[4];
    qr_co[16] = sc[1]*sc[11];

    /*reaction 49: TXCH2 + HO2 <=> CH2O + OH */
    qf_co[17] = sc[8];
    qr_co[17] = sc[4]*sc[11];

    /*reaction 50: TXCH2 + O2 => CO2 + 2.000000 H */
    qf_co[18] = sc[5];
    qr_co[18] = 0.0;

    /*reaction 51: TXCH2 + OH <=> CH + H2O */
    qf_co[19] = sc[4];
    qr_co[19] = sc[7];

    /*reaction 52: TXCH2 + O <=> HCO + H */
    qf_co[20] = sc[3];
    qr_co[20] = sc[1];

    /*reaction 53: TXCH2 + H2 <=> H + CH3 */
    qf_co[21] = sc[2];
    qr_co[21] = sc[1]*sc[10];

    /*reaction 54: SXCH2 + H2O <=> TXCH2 + H2O */
    qf_co[22] = sc[7];
    qr_co[22] = sc[7];

    /*reaction 55: SXCH2 + H <=> CH + H2 */
    qf_co[23] = sc[1];
    qr_co[23] = sc[2];

    /*reaction 56: SXCH2 + O2 <=> H + OH + CO */
    qf_co[24] = sc[5];
    qr_co[24] = sc[1]*sc[4]*sc[9];

    /*reaction 57: SXCH2 + O <=> CO + H2 */
    qf_co[25] = sc[3];
    qr_co[25] = sc[2]*sc[9];

    /*reaction 58: SXCH2 + O2 <=> CO + H2O */
    qf_co[26] = sc[5];
    qr_co[26] = sc[7]*sc[9];

    /*reaction 59: SXCH2 + H2 <=> CH3 + H */
    qf_co[27] = sc[2];
    qr_co[27] = sc[1]*sc[10];

    /*reaction 60: SXCH2 + O <=> HCO + H */
    qf_co[28] = sc[3];
    qr_co[28] = sc[1];

    /*reaction 61: SXCH2 + H2O => H2 + CH2O */
    qf_co[29] = sc[7];
    qr_co[29] = 0.0;

    /*reaction 62: SXCH2 + OH <=> CH2O + H */
    qf_co[30] = sc[4];
    qr_co[30] = sc[1]*sc[11];

    /*reaction 66: CH3 + CH <=> C2H3 + H */
    qf_co[31] = sc[10];
    qr_co[31] = sc[1]*sc[18];

    /*reaction 68: CH3 + C <=> C2H2 + H */
    qf_co[32] = sc[10];
    qr_co[32] = sc[1]*sc[14];

    /*reaction 69: CH3 + OH <=> TXCH2 + H2O */
    qf_co[33] = sc[4]*sc[10];
    qr_co[33] = sc[7];

    /*reaction 70: CH3 + SXCH2 <=> C2H4 + H */
    qf_co[34] = sc[10];
    qr_co[34] = sc[1]*sc[15];

    /*reaction 71: CH3 + OH <=> SXCH2 + H2O */
    qf_co[35] = sc[4]*sc[10];
    qr_co[35] = sc[7];

    /*reaction 74: CH3 + TXCH2 <=> C2H4 + H */
    qf_co[36] = sc[10];
    qr_co[36] = sc[1]*sc[15];

    /*reaction 76: CH4 + CH <=> C2H4 + H */
    qf_co[37] = sc[13];
    qr_co[37] = sc[1]*sc[15];

    /*reaction 77: CH4 + SXCH2 <=> 2.000000 CH3 */
    qf_co[38] = sc[13];
    qr_co[38] = pow(sc[10], 2.000000);

    /*reaction 80: CH4 + TXCH2 <=> 2.000000 CH3 */
    qf_co[39] = sc[13];
    qr_co[39] = pow(sc[10], 2.000000);

    /*reaction 82: SXCH2 + CO <=> TXCH2 + CO */
    qf_co[40] = sc[9];
    qr_co[40] = sc[9];

    /*reaction 87: HCO + H <=> CO + H2 */
    qf_co[41] = sc[1];
    qr_co[41] = sc[2]*sc[9];

    /*reaction 88: CH3 + HCO <=> CH3CHO */
    qf_co[42] = sc[10];
    qr_co[42] = sc[21];

    /*reaction 89: HCO + H2O <=> CO + H + H2O */
    qf_co[43] = sc[7];
    qr_co[43] = sc[1]*sc[7]*sc[9];

    /*reaction 90: HCO + O <=> CO + OH */
    qf_co[44] = sc[3];
    qr_co[44] = sc[4]*sc[9];

    /*reaction 91: HCO + OH <=> CO + H2O */
    qf_co[45] = sc[4];
    qr_co[45] = sc[7]*sc[9];

    /*reaction 92: CH3 + HCO <=> CH4 + CO */
    qf_co[46] = sc[10];
    qr_co[46] = sc[9]*sc[13];

    /*reaction 93: HCO + O <=> CO2 + H */
    qf_co[47] = sc[3];
    qr_co[47] = sc[1]*sc[12];

    /*reaction 94: HCO + O2 <=> CO + HO2 */
    qf_co[48] = sc[5];
    qr_co[48] = sc[8]*sc[9];

    /*reaction 95: CH2O + H <=> HCO + H2 */
    qf_co[49] = sc[1]*sc[11];
    qr_co[49] = sc[2];

    /*reaction 96: CH2O + O <=> HCO + OH */
    qf_co[50] = sc[3]*sc[11];
    qr_co[50] = sc[4];

    /*reaction 97: CH3 + CH2O <=> CH4 + HCO */
    qf_co[51] = sc[10]*sc[11];
    qr_co[51] = sc[13];

    /*reaction 98: CH2O + OH <=> HCO + H2O */
    qf_co[52] = sc[4]*sc[11];
    qr_co[52] = sc[7];

    /*reaction 99: CH2O + CH <=> CH2CO + H */
    qf_co[53] = sc[11];
    qr_co[53] = sc[1]*sc[16];

    /*reaction 100: CH2O + O2 <=> HCO + HO2 */
    qf_co[54] = sc[5]*sc[11];
    qr_co[54] = sc[8];

    /*reaction 101: CH2O + HO2 <=> HCO + H2O2 */
    qf_co[55] = sc[8]*sc[11];
    qr_co[55] = sc[6];

    /*reaction 103: SXCH2 + CO2 <=> TXCH2 + CO2 */
    qf_co[56] = sc[12];
    qr_co[56] = sc[12];

    /*reaction 104: SXCH2 + CO2 <=> CH2O + CO */
    qf_co[57] = sc[12];
    qr_co[57] = sc[9]*sc[11];

    /*reaction 105: CH + CO2 <=> HCO + CO */
    qf_co[58] = sc[12];
    qr_co[58] = sc[9];

    /*reaction 106: C2H2 + O <=> TXCH2 + CO */
    qf_co[59] = sc[3]*sc[14];
    qr_co[59] = sc[9];

    /*reaction 115: C2H3 + O2 <=> HCO + CH2O */
    qf_co[60] = sc[5]*sc[18];
    qr_co[60] = sc[11];

    /*reaction 123: C2H4 + O <=> CH3 + HCO */
    qf_co[61] = sc[3]*sc[15];
    qr_co[61] = sc[10];

    /*reaction 126: C2H4 + O <=> TXCH2 + CH2O */
    qf_co[62] = sc[3]*sc[15];
    qr_co[62] = sc[11];

    /*reaction 134: C2H6 + SXCH2 <=> C2H5 + CH3 */
    qf_co[63] = sc[17];
    qr_co[63] = sc[10]*sc[19];

    /*reaction 143: HCCO + H <=> SXCH2 + CO */
    qf_co[64] = sc[1]*sc[20];
    qr_co[64] = sc[9];

    /*reaction 145: CH2CO + TXCH2 <=> C2H4 + CO */
    qf_co[65] = sc[16];
    qr_co[65] = sc[9]*sc[15];

    /*reaction 148: CH2CO + O <=> TXCH2 + CO2 */
    qf_co[66] = sc[3]*sc[16];
    qr_co[66] = sc[12];

    /*reaction 152: CH2CO + TXCH2 <=> HCCO + CH3 */
    qf_co[67] = sc[16];
    qr_co[67] = sc[10]*sc[20];

    /*reaction 153: CH2CHO + O <=> CH2O + HCO */
    qf_co[68] = sc[3]*sc[22];
    qr_co[68] = sc[11];

    /*reaction 159: CH2CHO + O2 => OH + 2.000000 HCO */
    qf_co[69] = sc[5]*sc[22];
    qr_co[69] = 0.0;

    /*reaction 160: CH2CHO + H <=> CH3 + HCO */
    qf_co[70] = sc[1]*sc[22];
    qr_co[70] = sc[10];

    /*reaction 172: SXCH2 + N2 <=> TXCH2 + N2 */
    qf_co[71] = sc[0];
    qr_co[71] = sc[0];

    double T = tc[1];

    /*compute the mixture concentration */
    double mixture = 0.0;
    for (int i = 0; i < 24; ++i) {
        mixture += sc[i];
    }

    double Corr[72];
    for (int i = 0; i < 72; ++i) {
        Corr[i] = 1.0;
    }

    /* troe */
    {
        double alpha[5];
        alpha[0] = mixture + (TB[2][0] - 1)*sc[2] + (TB[2][1] - 1)*sc[7] + (TB[2][2] - 1)*sc[9] + (TB[2][3] - 1)*sc[12] + (TB[2][4] - 1)*sc[13] + (TB[2][5] - 1)*sc[17];
        alpha[1] = mixture + (TB[3][0] - 1)*sc[2] + (TB[3][1] - 1)*sc[7] + (TB[3][2] - 1)*sc[9] + (TB[3][3] - 1)*sc[12] + (TB[3][4] - 1)*sc[13] + (TB[3][5] - 1)*sc[17];
        alpha[2] = mixture + (TB[5][0] - 1)*sc[2] + (TB[5][1] - 1)*sc[7] + (TB[5][2] - 1)*sc[9] + (TB[5][3] - 1)*sc[12] + (TB[5][4] - 1)*sc[13] + (TB[5][5] - 1)*sc[17];
        alpha[3] = mixture + (TB[7][0] - 1)*sc[2] + (TB[7][1] - 1)*sc[7] + (TB[7][2] - 1)*sc[9] + (TB[7][3] - 1)*sc[12] + (TB[7][4] - 1)*sc[13] + (TB[7][5] - 1)*sc[17];
        alpha[4] = mixture + (TB[9][0] - 1)*sc[2] + (TB[9][1] - 1)*sc[7] + (TB[9][2] - 1)*sc[9] + (TB[9][3] - 1)*sc[12] + (TB[9][4] - 1)*sc[13] + (TB[9][5] - 1)*sc[17];
#ifdef __INTEL_COMPILER
         #pragma simd
#endif
        double redP, F, logPred, logFcent, troe_c, troe_n, troe, F_troe;
        /*Index for alpha is 0 */
        /*Reaction index is 2 */
        /*QSS reaction list index (corresponds to index needed by k_f_save_qss, Corr, Kc_save_qss) is 0 */
        redP = alpha[0] / k_f_save_qss[0] * phase_units[2] * low_A[2] * exp(low_beta[2] * tc[0] - activation_units[2] * low_Ea[2] *invT);
        F = redP / (1.0 + redP);
        logPred = log10(redP);
        logFcent = log10(
            (fabs(troe_Tsss[2]) > 1.e-100 ? (1.-troe_a[2])*exp(-T/troe_Tsss[2]) : 0.) 
            + (fabs(troe_Ts[2]) > 1.e-100 ? troe_a[2] * exp(-T/troe_Ts[2]) : 0.) 
            + (troe_len[2] == 4 ? exp(-troe_Tss[2] * invT) : 0.) );
        troe_c = -.4 - .67 * logFcent;
        troe_n = .75 - 1.27 * logFcent;
        troe = (troe_c + logPred) / (troe_n - .14*(troe_c + logPred));
        F_troe = pow(10., logFcent / (1.0 + troe*troe));
        Corr[0] = F * F_troe;
        /*Index for alpha is 1 */
        /*Reaction index is 3 */
        /*QSS reaction list index (corresponds to index needed by k_f_save_qss, Corr, Kc_save_qss) is 1 */
        redP = alpha[1] / k_f_save_qss[1] * phase_units[3] * low_A[3] * exp(low_beta[3] * tc[0] - activation_units[3] * low_Ea[3] *invT);
        F = redP / (1.0 + redP);
        logPred = log10(redP);
        logFcent = log10(
            (fabs(troe_Tsss[3]) > 1.e-100 ? (1.-troe_a[3])*exp(-T/troe_Tsss[3]) : 0.) 
            + (fabs(troe_Ts[3]) > 1.e-100 ? troe_a[3] * exp(-T/troe_Ts[3]) : 0.) 
            + (troe_len[3] == 4 ? exp(-troe_Tss[3] * invT) : 0.) );
        troe_c = -.4 - .67 * logFcent;
        troe_n = .75 - 1.27 * logFcent;
        troe = (troe_c + logPred) / (troe_n - .14*(troe_c + logPred));
        F_troe = pow(10., logFcent / (1.0 + troe*troe));
        Corr[1] = F * F_troe;
        /*Index for alpha is 2 */
        /*Reaction index is 5 */
        /*QSS reaction list index (corresponds to index needed by k_f_save_qss, Corr, Kc_save_qss) is 2 */
        redP = alpha[2] / k_f_save_qss[2] * phase_units[5] * low_A[5] * exp(low_beta[5] * tc[0] - activation_units[5] * low_Ea[5] *invT);
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
        Corr[2] = F * F_troe;
        /*Index for alpha is 3 */
        /*Reaction index is 7 */
        /*QSS reaction list index (corresponds to index needed by k_f_save_qss, Corr, Kc_save_qss) is 3 */
        redP = alpha[3] / k_f_save_qss[3] * phase_units[7] * low_A[7] * exp(low_beta[7] * tc[0] - activation_units[7] * low_Ea[7] *invT);
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
        Corr[3] = F * F_troe;
        /*Index for alpha is 4 */
        /*Reaction index is 9 */
        /*QSS reaction list index (corresponds to index needed by k_f_save_qss, Corr, Kc_save_qss) is 4 */
        redP = alpha[4] / k_f_save_qss[4] * phase_units[9] * low_A[9] * exp(low_beta[9] * tc[0] - activation_units[9] * low_Ea[9] *invT);
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
        Corr[4] = F * F_troe;
    }

    /* simple three-body correction */
    {
        double alpha;
        alpha = mixture + (TB[19][0] - 1)*sc[2] + (TB[19][1] - 1)*sc[7] + (TB[19][2] - 1)*sc[9] + (TB[19][3] - 1)*sc[12] + (TB[19][4] - 1)*sc[13] + (TB[19][5] - 1)*sc[17];
        Corr[5] = alpha;
    }

    for (int i=0; i<72; i++)
    {
        qf_co[i] *= Corr[i] * k_f_save_qss[i];
        qr_co[i] *= Corr[i] * k_f_save_qss[i] / Kc_save_qss[i];
    }

    return;
}

void comp_qss_sc(double * sc, double * sc_qss, double * tc, double invT)
{

    double  qf_co[72], qr_co[72];
    double epsilon = 1e-12;

    comp_qss_coeff(qf_co, qr_co, sc, tc, invT);

    /* QSS coupling between C  CH  HCO  TXCH2  SXCH2*/
    /*QSS species 0: C */

    double C_num = epsilon -qr_co[6] -qr_co[7] -qr_co[32];
    double C_denom = epsilon -qf_co[6] -qf_co[7] -qr_co[12] -qf_co[32];
    double C_rhs = C_num/C_denom;

    double C_CH = (epsilon + qf_co[12])/C_denom;

    /*QSS species 1: CH */

    double CH_num = epsilon -qr_co[0] -qr_co[3] -qr_co[10] -qr_co[13] -qr_co[31] -qr_co[37] -qr_co[53];
    double CH_denom = epsilon -qf_co[0] -qf_co[3] -qf_co[8] -qf_co[9] -qf_co[10] -qf_co[11] -qf_co[12]
                        -qf_co[13] -qr_co[19] -qr_co[23] -qf_co[31] -qf_co[37] -qf_co[53] -qf_co[58];
    double CH_rhs = CH_num/CH_denom;

    double CH_C = (epsilon + qr_co[12])/CH_denom;
    double CH_HCO = (epsilon + qr_co[8] + qr_co[11] + qr_co[58])/CH_denom;
    double CH_TXCH2 = (epsilon + qr_co[9] + qf_co[19])/CH_denom;
    double CH_SXCH2 = (epsilon + qf_co[23])/CH_denom;

    /*QSS species 2: HCO */

    double HCO_num = epsilon -qr_co[4] -qr_co[5] -qr_co[41] -qr_co[42] -qr_co[43] -qr_co[44] -qr_co[45]
                        -qr_co[46] -qr_co[47] -qr_co[48] -qf_co[49] -qf_co[50] -qf_co[51] -qf_co[52]
                        -qf_co[54] -qf_co[55] -qf_co[60] -qf_co[61] -qf_co[68] -qf_co[69] -qf_co[70];
    double HCO_denom = epsilon -qf_co[4] -qf_co[5] -qr_co[8] -qr_co[11] -qr_co[20] -qr_co[28] -qf_co[41]
                        -qf_co[42] -qf_co[43] -qf_co[44] -qf_co[45] -qf_co[46] -qf_co[47] -qf_co[48]
                        -qr_co[49] -qr_co[50] -qr_co[51] -qr_co[52] -qr_co[54] -qr_co[55] -qr_co[58]
                        -qr_co[60] -qr_co[61] -qr_co[68] -qr_co[70];
    double HCO_rhs = HCO_num/HCO_denom;

    double HCO_CH = (epsilon + qf_co[8] + qf_co[11] + qf_co[58])/HCO_denom;
    double HCO_TXCH2 = (epsilon + qf_co[20])/HCO_denom;
    double HCO_SXCH2 = (epsilon + qf_co[28])/HCO_denom;

    /*QSS species 3: TXCH2 */

    double TXCH2_num = epsilon -qr_co[1] -qr_co[2] -qr_co[15] -qr_co[16] -qr_co[17] -qr_co[21] -qf_co[33]
                        -qr_co[36] -qr_co[39] -qf_co[59] -qf_co[62] -qr_co[65] -qf_co[66] -qr_co[67];
    double TXCH2_denom = epsilon -qf_co[1] -qf_co[2] -qr_co[9] -qf_co[14] -qf_co[15] -qf_co[16] -qf_co[17]
                        -qf_co[18] -qf_co[19] -qf_co[20] -qf_co[21] -qr_co[22] -qr_co[33] -qf_co[36]
                        -qf_co[39] -qr_co[40] -qr_co[56] -qr_co[59] -qr_co[62] -qf_co[65] -qr_co[66]
                        -qf_co[67] -qr_co[71];
    double TXCH2_rhs = TXCH2_num/TXCH2_denom;

    double TXCH2_CH = (epsilon + qf_co[9] + qr_co[19])/TXCH2_denom;
    double TXCH2_HCO = (epsilon + qr_co[20])/TXCH2_denom;
    double TXCH2_SXCH2 = (epsilon + qf_co[22] + qf_co[40] + qf_co[56] + qf_co[71])/TXCH2_denom;

    /*QSS species 4: SXCH2 */

    double SXCH2_num = epsilon -qr_co[24] -qr_co[25] -qr_co[26] -qr_co[27] -qr_co[30] -qr_co[34] -qf_co[35]
                        -qr_co[38] -qr_co[57] -qr_co[63] -qf_co[64];
    double SXCH2_denom = epsilon -qf_co[22] -qf_co[23] -qf_co[24] -qf_co[25] -qf_co[26] -qf_co[27] -qf_co[28]
                        -qf_co[29] -qf_co[30] -qf_co[34] -qr_co[35] -qf_co[38] -qf_co[40] -qf_co[56]
                        -qf_co[57] -qf_co[63] -qr_co[64] -qf_co[71];
    double SXCH2_rhs = SXCH2_num/SXCH2_denom;

    double SXCH2_CH = (epsilon + qr_co[23])/SXCH2_denom;
    double SXCH2_HCO = (epsilon + qr_co[28])/SXCH2_denom;
    double SXCH2_TXCH2 = (epsilon + qr_co[22] + qr_co[40] + qr_co[56] + qr_co[71])/SXCH2_denom;

    sc_qss[4] = (((SXCH2_rhs - (CH_rhs - C_rhs * CH_C) * SXCH2_CH / (1 - C_CH * CH_C)) - (HCO_rhs - (CH_rhs - C_rhs * CH_C) * HCO_CH / (1 - C_CH * CH_C)) * (SXCH2_HCO - CH_HCO * SXCH2_CH/ (1 - C_CH * CH_C)) / (1 - CH_HCO * HCO_CH/ (1 - C_CH * CH_C))) - ((TXCH2_rhs - (CH_rhs - C_rhs * CH_C) * TXCH2_CH / (1 - C_CH * CH_C)) - (HCO_rhs - (CH_rhs - C_rhs * CH_C) * HCO_CH / (1 - C_CH * CH_C)) * (TXCH2_HCO - CH_HCO * TXCH2_CH/ (1 - C_CH * CH_C)) / (1 - CH_HCO * HCO_CH/ (1 - C_CH * CH_C))) * ((SXCH2_TXCH2 - CH_TXCH2 * SXCH2_CH/ (1 - C_CH * CH_C)) - (HCO_TXCH2 - CH_TXCH2 * HCO_CH/ (1 - C_CH * CH_C)) * (SXCH2_HCO - CH_HCO * SXCH2_CH/ (1 - C_CH * CH_C))/ (1 - CH_HCO * HCO_CH/ (1 - C_CH * CH_C))) / ((1 - CH_TXCH2 * TXCH2_CH/ (1 - C_CH * CH_C)) - (HCO_TXCH2 - CH_TXCH2 * HCO_CH/ (1 - C_CH * CH_C)) * (TXCH2_HCO - CH_HCO * TXCH2_CH/ (1 - C_CH * CH_C))/ (1 - CH_HCO * HCO_CH/ (1 - C_CH * CH_C)))) / (((1 - CH_SXCH2 * SXCH2_CH/ (1 - C_CH * CH_C)) - (HCO_SXCH2 - CH_SXCH2 * HCO_CH/ (1 - C_CH * CH_C)) * (SXCH2_HCO - CH_HCO * SXCH2_CH/ (1 - C_CH * CH_C))/ (1 - CH_HCO * HCO_CH/ (1 - C_CH * CH_C))) - ((TXCH2_SXCH2 - CH_SXCH2 * TXCH2_CH/ (1 - C_CH * CH_C)) - (HCO_SXCH2 - CH_SXCH2 * HCO_CH/ (1 - C_CH * CH_C)) * (TXCH2_HCO - CH_HCO * TXCH2_CH/ (1 - C_CH * CH_C))/ (1 - CH_HCO * HCO_CH/ (1 - C_CH * CH_C))) * ((SXCH2_TXCH2 - CH_TXCH2 * SXCH2_CH/ (1 - C_CH * CH_C)) - (HCO_TXCH2 - CH_TXCH2 * HCO_CH/ (1 - C_CH * CH_C)) * (SXCH2_HCO - CH_HCO * SXCH2_CH/ (1 - C_CH * CH_C))/ (1 - CH_HCO * HCO_CH/ (1 - C_CH * CH_C)))/ ((1 - CH_TXCH2 * TXCH2_CH/ (1 - C_CH * CH_C)) - (HCO_TXCH2 - CH_TXCH2 * HCO_CH/ (1 - C_CH * CH_C)) * (TXCH2_HCO - CH_HCO * TXCH2_CH/ (1 - C_CH * CH_C))/ (1 - CH_HCO * HCO_CH/ (1 - C_CH * CH_C))));

    sc_qss[3] = (((TXCH2_rhs - (CH_rhs - C_rhs * CH_C) * TXCH2_CH / (1 - C_CH * CH_C)) - (HCO_rhs - (CH_rhs - C_rhs * CH_C) * HCO_CH / (1 - C_CH * CH_C)) * (TXCH2_HCO - CH_HCO * TXCH2_CH/ (1 - C_CH * CH_C)) / (1 - CH_HCO * HCO_CH/ (1 - C_CH * CH_C))) - (((TXCH2_SXCH2 - CH_SXCH2 * TXCH2_CH/ (1 - C_CH * CH_C)) - (HCO_SXCH2 - CH_SXCH2 * HCO_CH/ (1 - C_CH * CH_C)) * (TXCH2_HCO - CH_HCO * TXCH2_CH/ (1 - C_CH * CH_C))/ (1 - CH_HCO * HCO_CH/ (1 - C_CH * CH_C))) * sc_qss[4])) / ((1 - CH_TXCH2 * TXCH2_CH/ (1 - C_CH * CH_C)) - (HCO_TXCH2 - CH_TXCH2 * HCO_CH/ (1 - C_CH * CH_C)) * (TXCH2_HCO - CH_HCO * TXCH2_CH/ (1 - C_CH * CH_C))/ (1 - CH_HCO * HCO_CH/ (1 - C_CH * CH_C)));

    sc_qss[2] = ((HCO_rhs - (CH_rhs - C_rhs * CH_C) * HCO_CH / (1 - C_CH * CH_C)) - ((HCO_SXCH2 - CH_SXCH2 * HCO_CH/ (1 - C_CH * CH_C)) * sc_qss[4] + (HCO_TXCH2 - CH_TXCH2 * HCO_CH/ (1 - C_CH * CH_C)) * sc_qss[3])) / (1 - CH_HCO * HCO_CH/ (1 - C_CH * CH_C));

    sc_qss[1] = ((CH_rhs - C_rhs * CH_C) - (CH_SXCH2 * sc_qss[4] + CH_TXCH2 * sc_qss[3] + CH_HCO * sc_qss[2])) / (1 - C_CH * CH_C);

    sc_qss[0] = C_rhs - ( + C_CH * sc_qss[1]);


    return;
}

/*compute an approx to the reaction Jacobian (for preconditioning) */
AMREX_GPU_HOST_DEVICE void DWDOT_SIMPLIFIED(double *  J, double *  sc, double *  Tp, int * HP)
{
    double c[24];

    for (int k=0; k<24; k++) {
        c[k] = 1.e6 * sc[k];
    }

    aJacobian_precond(J, c, *Tp, *HP);

    /* dwdot[k]/dT */
    /* dTdot/d[X] */
    for (int k=0; k<24; k++) {
        J[600+k] *= 1.e-6;
        J[k*25+24] *= 1.e6;
    }

    return;
}

/*compute the reaction Jacobian */
AMREX_GPU_HOST_DEVICE void DWDOT(double *  J, double *  sc, double *  Tp, int * consP)
{
    double c[24];

    for (int k=0; k<24; k++) {
        c[k] = 1.e6 * sc[k];
    }

    aJacobian(J, c, *Tp, *consP);

    /* dwdot[k]/dT */
    /* dTdot/d[X] */
    for (int k=0; k<24; k++) {
        J[600+k] *= 1.e-6;
        J[k*25+24] *= 1.e6;
    }

    return;
}

/*compute the sparsity pattern of the chemistry Jacobian */
AMREX_GPU_HOST_DEVICE void SPARSITY_INFO( int * nJdata, int * consP, int NCELLS)
{
    double c[24];
    double J[625];

    for (int k=0; k<24; k++) {
        c[k] = 1.0/ 24.000000 ;
    }

    aJacobian(J, c, 1500.0, *consP);

    int nJdata_tmp = 0;
    for (int k=0; k<25; k++) {
        for (int l=0; l<25; l++) {
            if(J[ 25 * k + l] != 0.0){
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
    double c[24];
    double J[625];

    for (int k=0; k<24; k++) {
        c[k] = 1.0/ 24.000000 ;
    }

    aJacobian(J, c, 1500.0, *consP);

    int nJdata_tmp = 0;
    for (int k=0; k<25; k++) {
        for (int l=0; l<25; l++) {
            if(k == l){
                nJdata_tmp = nJdata_tmp + 1;
            } else {
                if(J[ 25 * k + l] != 0.0){
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
    double c[24];
    double J[625];

    for (int k=0; k<24; k++) {
        c[k] = 1.0/ 24.000000 ;
    }

    aJacobian_precond(J, c, 1500.0, *consP);

    int nJdata_tmp = 0;
    for (int k=0; k<25; k++) {
        for (int l=0; l<25; l++) {
            if(k == l){
                nJdata_tmp = nJdata_tmp + 1;
            } else {
                if(J[ 25 * k + l] != 0.0){
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
    double c[24];
    double J[625];
    int offset_row;
    int offset_col;

    for (int k=0; k<24; k++) {
        c[k] = 1.0/ 24.000000 ;
    }

    aJacobian(J, c, 1500.0, *consP);

    colPtrs[0] = 0;
    int nJdata_tmp = 0;
    for (int nc=0; nc<NCELLS; nc++) {
        offset_row = nc * 25;
        offset_col = nc * 25;
        for (int k=0; k<25; k++) {
            for (int l=0; l<25; l++) {
                if(J[25*k + l] != 0.0) {
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
    double c[24];
    double J[625];
    int offset;

    for (int k=0; k<24; k++) {
        c[k] = 1.0/ 24.000000 ;
    }

    aJacobian(J, c, 1500.0, *consP);

    if (base == 1) {
        rowPtrs[0] = 1;
        int nJdata_tmp = 1;
        for (int nc=0; nc<NCELLS; nc++) {
            offset = nc * 25;
            for (int l=0; l<25; l++) {
                for (int k=0; k<25; k++) {
                    if(J[25*k + l] != 0.0) {
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
            offset = nc * 25;
            for (int l=0; l<25; l++) {
                for (int k=0; k<25; k++) {
                    if(J[25*k + l] != 0.0) {
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
    double c[24];
    double J[625];
    int offset;

    for (int k=0; k<24; k++) {
        c[k] = 1.0/ 24.000000 ;
    }

    aJacobian(J, c, 1500.0, *consP);

    if (base == 1) {
        rowPtr[0] = 1;
        int nJdata_tmp = 1;
        for (int nc=0; nc<NCELLS; nc++) {
            offset = nc * 25;
            for (int l=0; l<25; l++) {
                for (int k=0; k<25; k++) {
                    if (k == l) {
                        colVals[nJdata_tmp-1] = l+1 + offset; 
                        nJdata_tmp = nJdata_tmp + 1; 
                    } else {
                        if(J[25*k + l] != 0.0) {
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
            offset = nc * 25;
            for (int l=0; l<25; l++) {
                for (int k=0; k<25; k++) {
                    if (k == l) {
                        colVals[nJdata_tmp] = l + offset; 
                        nJdata_tmp = nJdata_tmp + 1; 
                    } else {
                        if(J[25*k + l] != 0.0) {
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
    double c[24];
    double J[625];

    for (int k=0; k<24; k++) {
        c[k] = 1.0/ 24.000000 ;
    }

    aJacobian_precond(J, c, 1500.0, *consP);

    colPtrs[0] = 0;
    int nJdata_tmp = 0;
    for (int k=0; k<25; k++) {
        for (int l=0; l<25; l++) {
            if (k == l) {
                rowVals[nJdata_tmp] = l; 
                indx[nJdata_tmp] = 25*k + l;
                nJdata_tmp = nJdata_tmp + 1; 
            } else {
                if(J[25*k + l] != 0.0) {
                    rowVals[nJdata_tmp] = l; 
                    indx[nJdata_tmp] = 25*k + l;
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
    double c[24];
    double J[625];

    for (int k=0; k<24; k++) {
        c[k] = 1.0/ 24.000000 ;
    }

    aJacobian_precond(J, c, 1500.0, *consP);

    if (base == 1) {
        rowPtr[0] = 1;
        int nJdata_tmp = 1;
        for (int l=0; l<25; l++) {
            for (int k=0; k<25; k++) {
                if (k == l) {
                    colVals[nJdata_tmp-1] = l+1; 
                    nJdata_tmp = nJdata_tmp + 1; 
                } else {
                    if(J[25*k + l] != 0.0) {
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
        for (int l=0; l<25; l++) {
            for (int k=0; k<25; k++) {
                if (k == l) {
                    colVals[nJdata_tmp] = l; 
                    nJdata_tmp = nJdata_tmp + 1; 
                } else {
                    if(J[25*k + l] != 0.0) {
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
    for (int i=0; i<625; i++) {
        J[i] = 0.0;
    }
}
#endif


/*compute an approx to the reaction Jacobian */
AMREX_GPU_HOST_DEVICE void aJacobian_precond(double *  J, double *  sc, double T, int HP)
{
    for (int i=0; i<625; i++) {
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
            +1.40824040e-03
            -7.92644400e-06 * tc[1]
            +1.69245450e-08 * tc[2]
            -9.77941600e-12 * tc[3];
        /*species 1: H */
        species[1] =
            +7.05332819e-13
            -3.99183928e-15 * tc[1]
            +6.90244896e-18 * tc[2]
            -3.71092933e-21 * tc[3];
        /*species 2: H2 */
        species[2] =
            +7.98052075e-03
            -3.89563020e-05 * tc[1]
            +6.04716282e-08 * tc[2]
            -2.95044704e-11 * tc[3];
        /*species 3: O */
        species[3] =
            -3.27931884e-03
            +1.32861279e-05 * tc[1]
            -1.83841987e-08 * tc[2]
            +8.45063884e-12 * tc[3];
        /*species 4: OH */
        species[4] =
            -3.22544939e-03
            +1.30552938e-05 * tc[1]
            -1.73956093e-08 * tc[2]
            +8.24949516e-12 * tc[3];
        /*species 5: O2 */
        species[5] =
            -2.99673416e-03
            +1.96946040e-05 * tc[1]
            -2.90438853e-08 * tc[2]
            +1.29749135e-11 * tc[3];
        /*species 6: H2O2 */
        species[6] =
            -5.42822417e-04
            +3.34671402e-05 * tc[1]
            -6.47312439e-08 * tc[2]
            +3.44981745e-11 * tc[3];
        /*species 7: H2O */
        species[7] =
            -2.03643410e-03
            +1.30408042e-05 * tc[1]
            -1.64639119e-08 * tc[2]
            +7.08791268e-12 * tc[3];
        /*species 8: HO2 */
        species[8] =
            -4.74912051e-03
            +4.23165782e-05 * tc[1]
            -7.28291682e-08 * tc[2]
            +3.71690050e-11 * tc[3];
        /*species 9: CO */
        species[9] =
            -6.10353680e-04
            +2.03362866e-06 * tc[1]
            +2.72101765e-09 * tc[2]
            -3.61769800e-12 * tc[3];
        /*species 10: CH3 */
        species[10] =
            +2.12659790e-03
            +1.09167766e-05 * tc[1]
            -1.98543009e-08 * tc[2]
            +9.86282960e-12 * tc[3];
        /*species 11: CH2O */
        species[11] =
            -9.90833369e-03
            +7.46440016e-05 * tc[1]
            -1.13785578e-07 * tc[2]
            +5.27090608e-11 * tc[3];
        /*species 12: CO2 */
        species[12] =
            +8.98459677e-03
            -1.42471254e-05 * tc[1]
            +7.37757066e-09 * tc[2]
            -5.74798192e-13 * tc[3];
        /*species 13: CH4 */
        species[13] =
            -1.36622009e-02
            +9.82907842e-05 * tc[1]
            -1.45274030e-07 * tc[2]
            +6.66413764e-11 * tc[3];
        /*species 14: C2H2 */
        species[14] =
            +2.33615629e-02
            -7.10343630e-05 * tc[1]
            +8.40457311e-08 * tc[2]
            -3.40029190e-11 * tc[3];
        /*species 15: C2H4 */
        species[15] =
            -7.57052247e-03
            +1.14198058e-04 * tc[1]
            -2.07476626e-07 * tc[2]
            +1.07953749e-10 * tc[3];
        /*species 16: CH2CO */
        species[16] =
            +1.81188721e-02
            -3.47894948e-05 * tc[1]
            +2.80319270e-08 * tc[2]
            -8.05830460e-12 * tc[3];
        /*species 17: C2H6 */
        species[17] =
            -5.50154270e-03
            +1.19887658e-04 * tc[1]
            -2.12539886e-07 * tc[2]
            +1.07474308e-10 * tc[3];
        /*species 18: C2H3 */
        species[18] =
            +1.51479162e-03
            +5.18418824e-05 * tc[1]
            -1.07297354e-07 * tc[2]
            +5.88603492e-11 * tc[3];
        /*species 19: C2H5 */
        species[19] =
            -4.18658892e-03
            +9.94285614e-05 * tc[1]
            -1.79737982e-07 * tc[2]
            +9.22036016e-11 * tc[3];
        /*species 20: HCCO */
        species[20] =
            +1.76550210e-02
            -4.74582020e-05 * tc[1]
            +5.18272770e-08 * tc[2]
            -2.02659244e-11 * tc[3];
        /*species 21: CH3CHO */
        species[21] =
            +2.16984438e-02
            -2.95146530e-05 * tc[1]
            +2.19130643e-08 * tc[2]
            -8.36477868e-12 * tc[3];
        /*species 22: CH2CHO */
        species[22] =
            +2.20228796e-02
            -2.89166888e-05 * tc[1]
            +9.02338734e-09 * tc[2]
            +2.43597151e-12 * tc[3];
        /*species 23: C2H5O */
        species[23] =
            +2.71774434e-02
            -3.31818020e-05 * tc[1]
            +1.54561260e-08 * tc[2]
            -2.59398766e-12 * tc[3];
    } else {
        /*species 0: N2 */
        species[0] =
            +1.48797680e-03
            -1.13695200e-06 * tc[1]
            +3.02911140e-10 * tc[2]
            -2.70134040e-14 * tc[3];
        /*species 1: H */
        species[1] =
            -2.30842973e-11
            +3.23123896e-14 * tc[1]
            -1.42054571e-17 * tc[2]
            +1.99278943e-21 * tc[3];
        /*species 2: H2 */
        species[2] =
            -4.94024731e-05
            +9.98913556e-07 * tc[1]
            -5.38699182e-10 * tc[2]
            +8.01021504e-14 * tc[3];
        /*species 3: O */
        species[3] =
            -8.59741137e-05
            +8.38969178e-08 * tc[1]
            -3.00533397e-11 * tc[2]
            +4.91334764e-15 * tc[3];
        /*species 4: OH */
        species[4] =
            +1.05650448e-03
            -5.18165516e-07 * tc[1]
            +9.15656022e-11 * tc[2]
            -5.32783504e-15 * tc[3];
        /*species 5: O2 */
        species[5] =
            +1.48308754e-03
            -1.51593334e-06 * tc[1]
            +6.28411665e-10 * tc[2]
            -8.66871176e-14 * tc[3];
        /*species 6: H2O2 */
        species[6] =
            +4.90831694e-03
            -3.80278450e-06 * tc[1]
            +1.11355796e-09 * tc[2]
            -1.15163322e-13 * tc[3];
        /*species 7: H2O */
        species[7] =
            +2.17691804e-03
            -3.28145036e-07 * tc[1]
            -2.91125961e-10 * tc[2]
            +6.72803968e-14 * tc[3];
        /*species 8: HO2 */
        species[8] =
            +2.23982013e-03
            -1.26731630e-06 * tc[1]
            +3.42739110e-10 * tc[2]
            -4.31634140e-14 * tc[3];
        /*species 9: CO */
        species[9] =
            +2.06252743e-03
            -1.99765154e-06 * tc[1]
            +6.90159024e-10 * tc[2]
            -8.14590864e-14 * tc[3];
        /*species 10: CH3 */
        species[10] =
            +5.79785200e-03
            -3.95116000e-06 * tc[1]
            +9.21893700e-10 * tc[2]
            -7.16696640e-14 * tc[3];
        /*species 11: CH2O */
        species[11] =
            +9.20000082e-03
            -8.84517626e-06 * tc[1]
            +3.01923636e-09 * tc[2]
            -3.53542256e-13 * tc[3];
        /*species 12: CO2 */
        species[12] =
            +4.41437026e-03
            -4.42962808e-06 * tc[1]
            +1.57047056e-09 * tc[2]
            -1.88833666e-13 * tc[3];
        /*species 13: CH4 */
        species[13] =
            +1.00263099e-02
            -6.63322476e-06 * tc[1]
            +1.60944941e-09 * tc[2]
            -1.25878703e-13 * tc[3];
        /*species 14: C2H2 */
        species[14] =
            +5.96166664e-03
            -4.74589704e-06 * tc[1]
            +1.40223651e-09 * tc[2]
            -1.44494085e-13 * tc[3];
        /*species 15: C2H4 */
        species[15] =
            +1.46454151e-02
            -1.34215583e-05 * tc[1]
            +4.41668769e-09 * tc[2]
            -5.02824244e-13 * tc[3];
        /*species 16: CH2CO */
        species[16] =
            +9.00359745e-03
            -8.33879270e-06 * tc[1]
            +2.77003765e-09 * tc[2]
            -3.17935280e-13 * tc[3];
        /*species 17: C2H6 */
        species[17] =
            +2.16852677e-02
            -2.00512134e-05 * tc[1]
            +6.64236003e-09 * tc[2]
            -7.60011560e-13 * tc[3];
        /*species 18: C2H3 */
        species[18] =
            +1.03302292e-02
            -9.36164698e-06 * tc[1]
            +3.05289864e-09 * tc[2]
            -3.45042816e-13 * tc[3];
        /*species 19: C2H5 */
        species[19] =
            +1.73972722e-02
            -1.59641334e-05 * tc[1]
            +5.25653067e-09 * tc[2]
            -5.98566304e-13 * tc[3];
        /*species 20: HCCO */
        species[20] =
            +4.08534010e-03
            -3.18690940e-06 * tc[1]
            +8.58781560e-10 * tc[2]
            -7.76313280e-14 * tc[3];
        /*species 21: CH3CHO */
        species[21] =
            +1.76802373e-02
            -1.73080548e-05 * tc[1]
            +6.11041767e-09 * tc[2]
            -7.50523740e-13 * tc[3];
        /*species 22: CH2CHO */
        species[22] =
            +1.72400021e-02
            -1.95426424e-05 * tc[1]
            +7.99667016e-09 * tc[2]
            -1.12848031e-12 * tc[3];
        /*species 23: C2H5O */
        species[23] =
            +2.09503959e-02
            -1.87858350e-05 * tc[1]
            +4.69321881e-09 * tc[2]
            +0.00000000e+00 * tc[3];
    }
    return;
}


/*compute the equilibrium constants for each reaction */
void equilibriumConstants(double *  kc, double *  g_RT, double T)
{
    /*reference concentration: P_atm / (RT) in inverse mol/m^3 */
    double refC = 101325 / 8.31446 / T;
    double tc[] = { log(T), T, T*T, T*T*T, T*T*T*T }; /*temperature cache */
    double g_RT_qss[5];
    gibbs_qss(g_RT_qss, tc);

    /*reaction 1: 2.000000 OH (+M) <=> H2O2 (+M) */
    kc[0] = 1.0 / (refC) * exp((2.000000 * g_RT[4]) - (g_RT[6]));

    /*reaction 2: H + O2 (+M) <=> HO2 (+M) */
    kc[1] = 1.0 / (refC) * exp((g_RT[1] + g_RT[5]) - (g_RT[8]));

    /*reaction 3: CH + H2 (+M) <=> CH3 (+M) */
    kc[2] = 1.0 / (refC) * exp((g_RT_qss[1] + g_RT[2]) - (g_RT[10]));

    /*reaction 4: TXCH2 + H (+M) <=> CH3 (+M) */
    kc[3] = 1.0 / (refC) * exp((g_RT_qss[3] + g_RT[1]) - (g_RT[10]));

    /*reaction 5: CH3 + H (+M) <=> CH4 (+M) */
    kc[4] = 1.0 / (refC) * exp((g_RT[10] + g_RT[1]) - (g_RT[13]));

    /*reaction 6: TXCH2 + CO (+M) <=> CH2CO (+M) */
    kc[5] = 1.0 / (refC) * exp((g_RT_qss[3] + g_RT[9]) - (g_RT[16]));

    /*reaction 7: CO + H2 (+M) <=> CH2O (+M) */
    kc[6] = 1.0 / (refC) * exp((g_RT[9] + g_RT[2]) - (g_RT[11]));

    /*reaction 8: CH + CO (+M) <=> HCCO (+M) */
    kc[7] = 1.0 / (refC) * exp((g_RT_qss[1] + g_RT[9]) - (g_RT[20]));

    /*reaction 9: CO + O (+M) <=> CO2 (+M) */
    kc[8] = 1.0 / (refC) * exp((g_RT[9] + g_RT[3]) - (g_RT[12]));

    /*reaction 10: HCO + H (+M) <=> CH2O (+M) */
    kc[9] = 1.0 / (refC) * exp((g_RT_qss[2] + g_RT[1]) - (g_RT[11]));

    /*reaction 11: C2H2 + H (+M) <=> C2H3 (+M) */
    kc[10] = 1.0 / (refC) * exp((g_RT[14] + g_RT[1]) - (g_RT[18]));

    /*reaction 12: C2H3 + H (+M) <=> C2H4 (+M) */
    kc[11] = 1.0 / (refC) * exp((g_RT[18] + g_RT[1]) - (g_RT[15]));

    /*reaction 13: C2H4 + H (+M) <=> C2H5 (+M) */
    kc[12] = 1.0 / (refC) * exp((g_RT[15] + g_RT[1]) - (g_RT[19]));

    /*reaction 14: C2H5 + H (+M) <=> C2H6 (+M) */
    kc[13] = 1.0 / (refC) * exp((g_RT[19] + g_RT[1]) - (g_RT[17]));

    /*reaction 15: C2H6 (+M) <=> 2.000000 CH3 (+M) */
    kc[14] = refC * exp((g_RT[17]) - (2.000000 * g_RT[10]));

    /*reaction 16: 2.000000 H + M <=> H2 + M */
    kc[15] = 1.0 / (refC) * exp((2.000000 * g_RT[1]) - (g_RT[2]));

    /*reaction 17: H + O + M <=> OH + M */
    kc[16] = 1.0 / (refC) * exp((g_RT[1] + g_RT[3]) - (g_RT[4]));

    /*reaction 18: 2.000000 O + M <=> O2 + M */
    kc[17] = 1.0 / (refC) * exp((2.000000 * g_RT[3]) - (g_RT[5]));

    /*reaction 19: H + OH + M <=> H2O + M */
    kc[18] = 1.0 / (refC) * exp((g_RT[1] + g_RT[4]) - (g_RT[7]));

    /*reaction 20: HCO + M <=> CO + H + M */
    kc[19] = refC * exp((g_RT_qss[2]) - (g_RT[9] + g_RT[1]));

    /*reaction 21: 2.000000 H + H2 <=> 2.000000 H2 */
    kc[20] = 1.0 / (refC) * exp((2.000000 * g_RT[1] + g_RT[2]) - (2.000000 * g_RT[2]));

    /*reaction 22: O + H2 <=> H + OH */
    kc[21] = exp((g_RT[3] + g_RT[2]) - (g_RT[1] + g_RT[4]));

    /*reaction 23: 2.000000 OH <=> O + H2O */
    kc[22] = exp((2.000000 * g_RT[4]) - (g_RT[3] + g_RT[7]));

    /*reaction 24: OH + H2 <=> H + H2O */
    kc[23] = exp((g_RT[4] + g_RT[2]) - (g_RT[1] + g_RT[7]));

    /*reaction 25: 2.000000 H + H2O <=> H2 + H2O */
    kc[24] = 1.0 / (refC) * exp((2.000000 * g_RT[1] + g_RT[7]) - (g_RT[2] + g_RT[7]));

    /*reaction 26: H + O2 <=> O + OH */
    kc[25] = exp((g_RT[1] + g_RT[5]) - (g_RT[3] + g_RT[4]));

    /*reaction 27: H2 + O2 <=> HO2 + H */
    kc[26] = exp((g_RT[2] + g_RT[5]) - (g_RT[8] + g_RT[1]));

    /*reaction 28: HO2 + OH <=> H2O + O2 */
    kc[27] = exp((g_RT[8] + g_RT[4]) - (g_RT[7] + g_RT[5]));

    /*reaction 29: HO2 + H <=> 2.000000 OH */
    kc[28] = exp((g_RT[8] + g_RT[1]) - (2.000000 * g_RT[4]));

    /*reaction 30: HO2 + O <=> OH + O2 */
    kc[29] = exp((g_RT[8] + g_RT[3]) - (g_RT[4] + g_RT[5]));

    /*reaction 31: HO2 + H <=> O + H2O */
    kc[30] = exp((g_RT[8] + g_RT[1]) - (g_RT[3] + g_RT[7]));

    /*reaction 32: HO2 + OH <=> H2O + O2 */
    kc[31] = exp((g_RT[8] + g_RT[4]) - (g_RT[7] + g_RT[5]));

    /*reaction 33: H2O2 + O <=> HO2 + OH */
    kc[32] = exp((g_RT[6] + g_RT[3]) - (g_RT[8] + g_RT[4]));

    /*reaction 34: H2O2 + H <=> H2O + OH */
    kc[33] = exp((g_RT[6] + g_RT[1]) - (g_RT[7] + g_RT[4]));

    /*reaction 35: H2O2 + H <=> HO2 + H2 */
    kc[34] = exp((g_RT[6] + g_RT[1]) - (g_RT[8] + g_RT[2]));

    /*reaction 36: H2O2 + OH <=> HO2 + H2O */
    kc[35] = exp((g_RT[6] + g_RT[4]) - (g_RT[8] + g_RT[7]));

    /*reaction 37: H2O2 + OH <=> HO2 + H2O */
    kc[36] = exp((g_RT[6] + g_RT[4]) - (g_RT[8] + g_RT[7]));

    /*reaction 38: C + O2 <=> CO + O */
    kc[37] = exp((g_RT_qss[0] + g_RT[5]) - (g_RT[9] + g_RT[3]));

    /*reaction 39: C + OH <=> CO + H */
    kc[38] = exp((g_RT_qss[0] + g_RT[4]) - (g_RT[9] + g_RT[1]));

    /*reaction 40: CH + OH <=> HCO + H */
    kc[39] = exp((g_RT_qss[1] + g_RT[4]) - (g_RT_qss[2] + g_RT[1]));

    /*reaction 41: CH + H2 <=> TXCH2 + H */
    kc[40] = exp((g_RT_qss[1] + g_RT[2]) - (g_RT_qss[3] + g_RT[1]));

    /*reaction 42: CH + O <=> CO + H */
    kc[41] = exp((g_RT_qss[1] + g_RT[3]) - (g_RT[9] + g_RT[1]));

    /*reaction 43: CH + O2 <=> HCO + O */
    kc[42] = exp((g_RT_qss[1] + g_RT[5]) - (g_RT_qss[2] + g_RT[3]));

    /*reaction 44: CH + H <=> C + H2 */
    kc[43] = exp((g_RT_qss[1] + g_RT[1]) - (g_RT_qss[0] + g_RT[2]));

    /*reaction 45: CH + H2O <=> CH2O + H */
    kc[44] = exp((g_RT_qss[1] + g_RT[7]) - (g_RT[11] + g_RT[1]));

    /*reaction 46: TXCH2 + O2 => OH + H + CO */
    kc[45] = refC * exp((g_RT_qss[3] + g_RT[5]) - (g_RT[4] + g_RT[1] + g_RT[9]));

    /*reaction 47: TXCH2 + O2 <=> CH2O + O */
    kc[46] = exp((g_RT_qss[3] + g_RT[5]) - (g_RT[11] + g_RT[3]));

    /*reaction 48: TXCH2 + OH <=> CH2O + H */
    kc[47] = exp((g_RT_qss[3] + g_RT[4]) - (g_RT[11] + g_RT[1]));

    /*reaction 49: TXCH2 + HO2 <=> CH2O + OH */
    kc[48] = exp((g_RT_qss[3] + g_RT[8]) - (g_RT[11] + g_RT[4]));

    /*reaction 50: TXCH2 + O2 => CO2 + 2.000000 H */
    kc[49] = refC * exp((g_RT_qss[3] + g_RT[5]) - (g_RT[12] + 2.000000 * g_RT[1]));

    /*reaction 51: TXCH2 + OH <=> CH + H2O */
    kc[50] = exp((g_RT_qss[3] + g_RT[4]) - (g_RT_qss[1] + g_RT[7]));

    /*reaction 52: TXCH2 + O <=> HCO + H */
    kc[51] = exp((g_RT_qss[3] + g_RT[3]) - (g_RT_qss[2] + g_RT[1]));

    /*reaction 53: TXCH2 + H2 <=> H + CH3 */
    kc[52] = exp((g_RT_qss[3] + g_RT[2]) - (g_RT[1] + g_RT[10]));

    /*reaction 54: SXCH2 + H2O <=> TXCH2 + H2O */
    kc[53] = exp((g_RT_qss[4] + g_RT[7]) - (g_RT_qss[3] + g_RT[7]));

    /*reaction 55: SXCH2 + H <=> CH + H2 */
    kc[54] = exp((g_RT_qss[4] + g_RT[1]) - (g_RT_qss[1] + g_RT[2]));

    /*reaction 56: SXCH2 + O2 <=> H + OH + CO */
    kc[55] = refC * exp((g_RT_qss[4] + g_RT[5]) - (g_RT[1] + g_RT[4] + g_RT[9]));

    /*reaction 57: SXCH2 + O <=> CO + H2 */
    kc[56] = exp((g_RT_qss[4] + g_RT[3]) - (g_RT[9] + g_RT[2]));

    /*reaction 58: SXCH2 + O2 <=> CO + H2O */
    kc[57] = exp((g_RT_qss[4] + g_RT[5]) - (g_RT[9] + g_RT[7]));

    /*reaction 59: SXCH2 + H2 <=> CH3 + H */
    kc[58] = exp((g_RT_qss[4] + g_RT[2]) - (g_RT[10] + g_RT[1]));

    /*reaction 60: SXCH2 + O <=> HCO + H */
    kc[59] = exp((g_RT_qss[4] + g_RT[3]) - (g_RT_qss[2] + g_RT[1]));

    /*reaction 61: SXCH2 + H2O => H2 + CH2O */
    kc[60] = exp((g_RT_qss[4] + g_RT[7]) - (g_RT[2] + g_RT[11]));

    /*reaction 62: SXCH2 + OH <=> CH2O + H */
    kc[61] = exp((g_RT_qss[4] + g_RT[4]) - (g_RT[11] + g_RT[1]));

    /*reaction 63: CH3 + OH => H2 + CH2O */
    kc[62] = exp((g_RT[10] + g_RT[4]) - (g_RT[2] + g_RT[11]));

    /*reaction 64: CH3 + H2O2 <=> CH4 + HO2 */
    kc[63] = exp((g_RT[10] + g_RT[6]) - (g_RT[13] + g_RT[8]));

    /*reaction 65: CH3 + O2 <=> CH2O + OH */
    kc[64] = exp((g_RT[10] + g_RT[5]) - (g_RT[11] + g_RT[4]));

    /*reaction 66: CH3 + CH <=> C2H3 + H */
    kc[65] = exp((g_RT[10] + g_RT_qss[1]) - (g_RT[18] + g_RT[1]));

    /*reaction 67: CH3 + O <=> CH2O + H */
    kc[66] = exp((g_RT[10] + g_RT[3]) - (g_RT[11] + g_RT[1]));

    /*reaction 68: CH3 + C <=> C2H2 + H */
    kc[67] = exp((g_RT[10] + g_RT_qss[0]) - (g_RT[14] + g_RT[1]));

    /*reaction 69: CH3 + OH <=> TXCH2 + H2O */
    kc[68] = exp((g_RT[10] + g_RT[4]) - (g_RT_qss[3] + g_RT[7]));

    /*reaction 70: CH3 + SXCH2 <=> C2H4 + H */
    kc[69] = exp((g_RT[10] + g_RT_qss[4]) - (g_RT[15] + g_RT[1]));

    /*reaction 71: CH3 + OH <=> SXCH2 + H2O */
    kc[70] = exp((g_RT[10] + g_RT[4]) - (g_RT_qss[4] + g_RT[7]));

    /*reaction 72: 2.000000 CH3 <=> C2H5 + H */
    kc[71] = exp((2.000000 * g_RT[10]) - (g_RT[19] + g_RT[1]));

    /*reaction 73: CH3 + HO2 <=> CH4 + O2 */
    kc[72] = exp((g_RT[10] + g_RT[8]) - (g_RT[13] + g_RT[5]));

    /*reaction 74: CH3 + TXCH2 <=> C2H4 + H */
    kc[73] = exp((g_RT[10] + g_RT_qss[3]) - (g_RT[15] + g_RT[1]));

    /*reaction 75: CH3 + O => H + H2 + CO */
    kc[74] = refC * exp((g_RT[10] + g_RT[3]) - (g_RT[1] + g_RT[2] + g_RT[9]));

    /*reaction 76: CH4 + CH <=> C2H4 + H */
    kc[75] = exp((g_RT[13] + g_RT_qss[1]) - (g_RT[15] + g_RT[1]));

    /*reaction 77: CH4 + SXCH2 <=> 2.000000 CH3 */
    kc[76] = exp((g_RT[13] + g_RT_qss[4]) - (2.000000 * g_RT[10]));

    /*reaction 78: CH4 + O <=> CH3 + OH */
    kc[77] = exp((g_RT[13] + g_RT[3]) - (g_RT[10] + g_RT[4]));

    /*reaction 79: CH4 + OH <=> CH3 + H2O */
    kc[78] = exp((g_RT[13] + g_RT[4]) - (g_RT[10] + g_RT[7]));

    /*reaction 80: CH4 + TXCH2 <=> 2.000000 CH3 */
    kc[79] = exp((g_RT[13] + g_RT_qss[3]) - (2.000000 * g_RT[10]));

    /*reaction 81: CH4 + H <=> CH3 + H2 */
    kc[80] = exp((g_RT[13] + g_RT[1]) - (g_RT[10] + g_RT[2]));

    /*reaction 82: SXCH2 + CO <=> TXCH2 + CO */
    kc[81] = exp((g_RT_qss[4] + g_RT[9]) - (g_RT_qss[3] + g_RT[9]));

    /*reaction 83: CO + O2 <=> CO2 + O */
    kc[82] = exp((g_RT[9] + g_RT[5]) - (g_RT[12] + g_RT[3]));

    /*reaction 84: CO + OH <=> CO2 + H */
    kc[83] = exp((g_RT[9] + g_RT[4]) - (g_RT[12] + g_RT[1]));

    /*reaction 85: CO + OH <=> CO2 + H */
    kc[84] = exp((g_RT[9] + g_RT[4]) - (g_RT[12] + g_RT[1]));

    /*reaction 86: CO + HO2 <=> CO2 + OH */
    kc[85] = exp((g_RT[9] + g_RT[8]) - (g_RT[12] + g_RT[4]));

    /*reaction 87: HCO + H <=> CO + H2 */
    kc[86] = exp((g_RT_qss[2] + g_RT[1]) - (g_RT[9] + g_RT[2]));

    /*reaction 88: CH3 + HCO <=> CH3CHO */
    kc[87] = 1.0 / (refC) * exp((g_RT[10] + g_RT_qss[2]) - (g_RT[21]));

    /*reaction 89: HCO + H2O <=> CO + H + H2O */
    kc[88] = refC * exp((g_RT_qss[2] + g_RT[7]) - (g_RT[9] + g_RT[1] + g_RT[7]));

    /*reaction 90: HCO + O <=> CO + OH */
    kc[89] = exp((g_RT_qss[2] + g_RT[3]) - (g_RT[9] + g_RT[4]));

    /*reaction 91: HCO + OH <=> CO + H2O */
    kc[90] = exp((g_RT_qss[2] + g_RT[4]) - (g_RT[9] + g_RT[7]));

    /*reaction 92: CH3 + HCO <=> CH4 + CO */
    kc[91] = exp((g_RT[10] + g_RT_qss[2]) - (g_RT[13] + g_RT[9]));

    /*reaction 93: HCO + O <=> CO2 + H */
    kc[92] = exp((g_RT_qss[2] + g_RT[3]) - (g_RT[12] + g_RT[1]));

    /*reaction 94: HCO + O2 <=> CO + HO2 */
    kc[93] = exp((g_RT_qss[2] + g_RT[5]) - (g_RT[9] + g_RT[8]));

    /*reaction 95: CH2O + H <=> HCO + H2 */
    kc[94] = exp((g_RT[11] + g_RT[1]) - (g_RT_qss[2] + g_RT[2]));

    /*reaction 96: CH2O + O <=> HCO + OH */
    kc[95] = exp((g_RT[11] + g_RT[3]) - (g_RT_qss[2] + g_RT[4]));

    /*reaction 97: CH3 + CH2O <=> CH4 + HCO */
    kc[96] = exp((g_RT[10] + g_RT[11]) - (g_RT[13] + g_RT_qss[2]));

    /*reaction 98: CH2O + OH <=> HCO + H2O */
    kc[97] = exp((g_RT[11] + g_RT[4]) - (g_RT_qss[2] + g_RT[7]));

    /*reaction 99: CH2O + CH <=> CH2CO + H */
    kc[98] = exp((g_RT[11] + g_RT_qss[1]) - (g_RT[16] + g_RT[1]));

    /*reaction 100: CH2O + O2 <=> HCO + HO2 */
    kc[99] = exp((g_RT[11] + g_RT[5]) - (g_RT_qss[2] + g_RT[8]));

    /*reaction 101: CH2O + HO2 <=> HCO + H2O2 */
    kc[100] = exp((g_RT[11] + g_RT[8]) - (g_RT_qss[2] + g_RT[6]));

    /*reaction 102: 2.000000 H + CO2 <=> H2 + CO2 */
    kc[101] = 1.0 / (refC) * exp((2.000000 * g_RT[1] + g_RT[12]) - (g_RT[2] + g_RT[12]));

    /*reaction 103: SXCH2 + CO2 <=> TXCH2 + CO2 */
    kc[102] = exp((g_RT_qss[4] + g_RT[12]) - (g_RT_qss[3] + g_RT[12]));

    /*reaction 104: SXCH2 + CO2 <=> CH2O + CO */
    kc[103] = exp((g_RT_qss[4] + g_RT[12]) - (g_RT[11] + g_RT[9]));

    /*reaction 105: CH + CO2 <=> HCO + CO */
    kc[104] = exp((g_RT_qss[1] + g_RT[12]) - (g_RT_qss[2] + g_RT[9]));

    /*reaction 106: C2H2 + O <=> TXCH2 + CO */
    kc[105] = exp((g_RT[14] + g_RT[3]) - (g_RT_qss[3] + g_RT[9]));

    /*reaction 107: C2H2 + OH <=> CH3 + CO */
    kc[106] = exp((g_RT[14] + g_RT[4]) - (g_RT[10] + g_RT[9]));

    /*reaction 108: C2H2 + OH <=> CH2CO + H */
    kc[107] = exp((g_RT[14] + g_RT[4]) - (g_RT[16] + g_RT[1]));

    /*reaction 109: C2H2 + O <=> HCCO + H */
    kc[108] = exp((g_RT[14] + g_RT[3]) - (g_RT[20] + g_RT[1]));

    /*reaction 110: C2H3 + OH <=> C2H2 + H2O */
    kc[109] = exp((g_RT[18] + g_RT[4]) - (g_RT[14] + g_RT[7]));

    /*reaction 111: C2H3 + O2 <=> CH2CHO + O */
    kc[110] = exp((g_RT[18] + g_RT[5]) - (g_RT[22] + g_RT[3]));

    /*reaction 112: C2H3 + O <=> CH2CHO */
    kc[111] = 1.0 / (refC) * exp((g_RT[18] + g_RT[3]) - (g_RT[22]));

    /*reaction 113: C2H3 + H <=> C2H2 + H2 */
    kc[112] = exp((g_RT[18] + g_RT[1]) - (g_RT[14] + g_RT[2]));

    /*reaction 114: C2H3 + CH3 <=> C2H2 + CH4 */
    kc[113] = exp((g_RT[18] + g_RT[10]) - (g_RT[14] + g_RT[13]));

    /*reaction 115: C2H3 + O2 <=> HCO + CH2O */
    kc[114] = exp((g_RT[18] + g_RT[5]) - (g_RT_qss[2] + g_RT[11]));

    /*reaction 116: C2H3 + H2O2 <=> C2H4 + HO2 */
    kc[115] = exp((g_RT[18] + g_RT[6]) - (g_RT[15] + g_RT[8]));

    /*reaction 117: C2H3 + O2 <=> C2H2 + HO2 */
    kc[116] = exp((g_RT[18] + g_RT[5]) - (g_RT[14] + g_RT[8]));

    /*reaction 118: C2H4 + CH3 <=> C2H3 + CH4 */
    kc[117] = exp((g_RT[15] + g_RT[10]) - (g_RT[18] + g_RT[13]));

    /*reaction 119: C2H4 + O2 => CH3 + CO2 + H */
    kc[118] = refC * exp((g_RT[15] + g_RT[5]) - (g_RT[10] + g_RT[12] + g_RT[1]));

    /*reaction 120: C2H4 + OH <=> C2H3 + H2O */
    kc[119] = exp((g_RT[15] + g_RT[4]) - (g_RT[18] + g_RT[7]));

    /*reaction 121: C2H4 + OH <=> C2H5O */
    kc[120] = 1.0 / (refC) * exp((g_RT[15] + g_RT[4]) - (g_RT[23]));

    /*reaction 122: C2H4 + O <=> CH2CHO + H */
    kc[121] = exp((g_RT[15] + g_RT[3]) - (g_RT[22] + g_RT[1]));

    /*reaction 123: C2H4 + O <=> CH3 + HCO */
    kc[122] = exp((g_RT[15] + g_RT[3]) - (g_RT[10] + g_RT_qss[2]));

    /*reaction 124: C2H4 + O2 <=> C2H3 + HO2 */
    kc[123] = exp((g_RT[15] + g_RT[5]) - (g_RT[18] + g_RT[8]));

    /*reaction 125: C2H4 + H <=> C2H3 + H2 */
    kc[124] = exp((g_RT[15] + g_RT[1]) - (g_RT[18] + g_RT[2]));

    /*reaction 126: C2H4 + O <=> TXCH2 + CH2O */
    kc[125] = exp((g_RT[15] + g_RT[3]) - (g_RT_qss[3] + g_RT[11]));

    /*reaction 127: C2H5 + HO2 <=> C2H4 + H2O2 */
    kc[126] = exp((g_RT[19] + g_RT[8]) - (g_RT[15] + g_RT[6]));

    /*reaction 128: C2H5 + HO2 <=> C2H5O + OH */
    kc[127] = exp((g_RT[19] + g_RT[8]) - (g_RT[23] + g_RT[4]));

    /*reaction 129: C2H5 + O <=> C2H5O */
    kc[128] = 1.0 / (refC) * exp((g_RT[19] + g_RT[3]) - (g_RT[23]));

    /*reaction 130: C2H5 + H <=> C2H4 + H2 */
    kc[129] = exp((g_RT[19] + g_RT[1]) - (g_RT[15] + g_RT[2]));

    /*reaction 131: C2H5 + O2 <=> C2H4 + HO2 */
    kc[130] = exp((g_RT[19] + g_RT[5]) - (g_RT[15] + g_RT[8]));

    /*reaction 132: C2H5 + HO2 <=> C2H6 + O2 */
    kc[131] = exp((g_RT[19] + g_RT[8]) - (g_RT[17] + g_RT[5]));

    /*reaction 133: C2H5 + CH3 <=> C2H4 + CH4 */
    kc[132] = exp((g_RT[19] + g_RT[10]) - (g_RT[15] + g_RT[13]));

    /*reaction 134: C2H6 + SXCH2 <=> C2H5 + CH3 */
    kc[133] = exp((g_RT[17] + g_RT_qss[4]) - (g_RT[19] + g_RT[10]));

    /*reaction 135: C2H6 + CH3 <=> C2H5 + CH4 */
    kc[134] = exp((g_RT[17] + g_RT[10]) - (g_RT[19] + g_RT[13]));

    /*reaction 136: C2H6 + O <=> C2H5 + OH */
    kc[135] = exp((g_RT[17] + g_RT[3]) - (g_RT[19] + g_RT[4]));

    /*reaction 137: C2H6 + HO2 <=> C2H5 + H2O2 */
    kc[136] = exp((g_RT[17] + g_RT[8]) - (g_RT[19] + g_RT[6]));

    /*reaction 138: C2H6 + H <=> C2H5 + H2 */
    kc[137] = exp((g_RT[17] + g_RT[1]) - (g_RT[19] + g_RT[2]));

    /*reaction 139: C2H6 + OH <=> C2H5 + H2O */
    kc[138] = exp((g_RT[17] + g_RT[4]) - (g_RT[19] + g_RT[7]));

    /*reaction 140: HCCO + O2 <=> OH + 2.000000 CO */
    kc[139] = refC * exp((g_RT[20] + g_RT[5]) - (g_RT[4] + 2.000000 * g_RT[9]));

    /*reaction 141: HCCO + O <=> H + 2.000000 CO */
    kc[140] = refC * exp((g_RT[20] + g_RT[3]) - (g_RT[1] + 2.000000 * g_RT[9]));

    /*reaction 142: HCCO + CH3 <=> C2H4 + CO */
    kc[141] = exp((g_RT[20] + g_RT[10]) - (g_RT[15] + g_RT[9]));

    /*reaction 143: HCCO + H <=> SXCH2 + CO */
    kc[142] = exp((g_RT[20] + g_RT[1]) - (g_RT_qss[4] + g_RT[9]));

    /*reaction 144: CH2CO + H <=> CH3 + CO */
    kc[143] = exp((g_RT[16] + g_RT[1]) - (g_RT[10] + g_RT[9]));

    /*reaction 145: CH2CO + TXCH2 <=> C2H4 + CO */
    kc[144] = exp((g_RT[16] + g_RT_qss[3]) - (g_RT[15] + g_RT[9]));

    /*reaction 146: CH2CO + O <=> HCCO + OH */
    kc[145] = exp((g_RT[16] + g_RT[3]) - (g_RT[20] + g_RT[4]));

    /*reaction 147: CH2CO + CH3 <=> HCCO + CH4 */
    kc[146] = exp((g_RT[16] + g_RT[10]) - (g_RT[20] + g_RT[13]));

    /*reaction 148: CH2CO + O <=> TXCH2 + CO2 */
    kc[147] = exp((g_RT[16] + g_RT[3]) - (g_RT_qss[3] + g_RT[12]));

    /*reaction 149: CH2CO + CH3 <=> C2H5 + CO */
    kc[148] = exp((g_RT[16] + g_RT[10]) - (g_RT[19] + g_RT[9]));

    /*reaction 150: CH2CO + OH <=> HCCO + H2O */
    kc[149] = exp((g_RT[16] + g_RT[4]) - (g_RT[20] + g_RT[7]));

    /*reaction 151: CH2CO + H <=> HCCO + H2 */
    kc[150] = exp((g_RT[16] + g_RT[1]) - (g_RT[20] + g_RT[2]));

    /*reaction 152: CH2CO + TXCH2 <=> HCCO + CH3 */
    kc[151] = exp((g_RT[16] + g_RT_qss[3]) - (g_RT[20] + g_RT[10]));

    /*reaction 153: CH2CHO + O <=> CH2O + HCO */
    kc[152] = exp((g_RT[22] + g_RT[3]) - (g_RT[11] + g_RT_qss[2]));

    /*reaction 154: CH2CHO <=> CH2CO + H */
    kc[153] = refC * exp((g_RT[22]) - (g_RT[16] + g_RT[1]));

    /*reaction 155: CH2CHO + OH <=> H2O + CH2CO */
    kc[154] = exp((g_RT[22] + g_RT[4]) - (g_RT[7] + g_RT[16]));

    /*reaction 156: CH2CHO + H <=> CH2CO + H2 */
    kc[155] = exp((g_RT[22] + g_RT[1]) - (g_RT[16] + g_RT[2]));

    /*reaction 157: CH2CHO + O2 => OH + CO + CH2O */
    kc[156] = refC * exp((g_RT[22] + g_RT[5]) - (g_RT[4] + g_RT[9] + g_RT[11]));

    /*reaction 158: CH2CHO <=> CH3 + CO */
    kc[157] = refC * exp((g_RT[22]) - (g_RT[10] + g_RT[9]));

    /*reaction 159: CH2CHO + O2 => OH + 2.000000 HCO */
    kc[158] = refC * exp((g_RT[22] + g_RT[5]) - (g_RT[4] + 2.000000 * g_RT_qss[2]));

    /*reaction 160: CH2CHO + H <=> CH3 + HCO */
    kc[159] = exp((g_RT[22] + g_RT[1]) - (g_RT[10] + g_RT_qss[2]));

    /*reaction 161: CH3CHO + O => CH3 + CO + OH */
    kc[160] = refC * exp((g_RT[21] + g_RT[3]) - (g_RT[10] + g_RT[9] + g_RT[4]));

    /*reaction 162: CH3CHO + O2 => CH3 + CO + HO2 */
    kc[161] = refC * exp((g_RT[21] + g_RT[5]) - (g_RT[10] + g_RT[9] + g_RT[8]));

    /*reaction 163: CH3CHO + OH => CH3 + CO + H2O */
    kc[162] = refC * exp((g_RT[21] + g_RT[4]) - (g_RT[10] + g_RT[9] + g_RT[7]));

    /*reaction 164: CH3CHO + H <=> CH2CHO + H2 */
    kc[163] = exp((g_RT[21] + g_RT[1]) - (g_RT[22] + g_RT[2]));

    /*reaction 165: CH3CHO + H => CH3 + CO + H2 */
    kc[164] = refC * exp((g_RT[21] + g_RT[1]) - (g_RT[10] + g_RT[9] + g_RT[2]));

    /*reaction 166: CH3CHO + O <=> CH2CHO + OH */
    kc[165] = exp((g_RT[21] + g_RT[3]) - (g_RT[22] + g_RT[4]));

    /*reaction 167: CH3CHO + CH3 => CH3 + CO + CH4 */
    kc[166] = refC * exp((g_RT[21] + g_RT[10]) - (g_RT[10] + g_RT[9] + g_RT[13]));

    /*reaction 168: CH3CHO + HO2 => CH3 + CO + H2O2 */
    kc[167] = refC * exp((g_RT[21] + g_RT[8]) - (g_RT[10] + g_RT[9] + g_RT[6]));

    /*reaction 169: C2H5O <=> CH3 + CH2O */
    kc[168] = refC * exp((g_RT[23]) - (g_RT[10] + g_RT[11]));

    /*reaction 170: C2H5O <=> CH3CHO + H */
    kc[169] = refC * exp((g_RT[23]) - (g_RT[21] + g_RT[1]));

    /*reaction 171: C2H5O + O2 <=> CH3CHO + HO2 */
    kc[170] = exp((g_RT[23] + g_RT[5]) - (g_RT[21] + g_RT[8]));

    /*reaction 172: SXCH2 + N2 <=> TXCH2 + N2 */
    kc[171] = exp((g_RT_qss[4] + g_RT[0]) - (g_RT_qss[3] + g_RT[0]));

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
            -1.020899900000000e+03 * invT
            -6.516950000000001e-01
            -3.298677000000000e+00 * tc[0]
            -7.041202000000000e-04 * tc[1]
            +6.605369999999999e-07 * tc[2]
            -4.701262500000001e-10 * tc[3]
            +1.222427000000000e-13 * tc[4];
        /*species 1: H */
        species[1] =
            +2.547365990000000e+04 * invT
            +2.946682853000000e+00
            -2.500000000000000e+00 * tc[0]
            -3.526664095000000e-13 * tc[1]
            +3.326532733333333e-16 * tc[2]
            -1.917346933333333e-19 * tc[3]
            +4.638661660000000e-23 * tc[4];
        /*species 2: H2 */
        species[2] =
            -9.179351730000000e+02 * invT
            +1.661320882000000e+00
            -2.344331120000000e+00 * tc[0]
            -3.990260375000000e-03 * tc[1]
            +3.246358500000000e-06 * tc[2]
            -1.679767450000000e-09 * tc[3]
            +3.688058805000000e-13 * tc[4];
        /*species 3: O */
        species[3] =
            +2.912225920000000e+04 * invT
            +1.116333640000000e+00
            -3.168267100000000e+00 * tc[0]
            +1.639659420000000e-03 * tc[1]
            -1.107177326666667e-06 * tc[2]
            +5.106721866666666e-10 * tc[3]
            -1.056329855000000e-13 * tc[4];
        /*species 4: OH */
        species[4] =
            +3.381538120000000e+03 * invT
            +4.815738570000000e+00
            -4.125305610000000e+00 * tc[0]
            +1.612724695000000e-03 * tc[1]
            -1.087941151666667e-06 * tc[2]
            +4.832113691666666e-10 * tc[3]
            -1.031186895000000e-13 * tc[4];
        /*species 5: O2 */
        species[5] =
            -1.063943560000000e+03 * invT
            +1.247806300000001e-01
            -3.782456360000000e+00 * tc[0]
            +1.498367080000000e-03 * tc[1]
            -1.641217001666667e-06 * tc[2]
            +8.067745908333334e-10 * tc[3]
            -1.621864185000000e-13 * tc[4];
        /*species 6: H2O2 */
        species[6] =
            -1.770258210000000e+04 * invT
            +8.410619499999998e-01
            -4.276112690000000e+00 * tc[0]
            +2.714112085000000e-04 * tc[1]
            -2.788928350000000e-06 * tc[2]
            +1.798090108333333e-09 * tc[3]
            -4.312271815000000e-13 * tc[4];
        /*species 7: H2O */
        species[7] =
            -3.029372670000000e+04 * invT
            +5.047672768000000e+00
            -4.198640560000000e+00 * tc[0]
            +1.018217050000000e-03 * tc[1]
            -1.086733685000000e-06 * tc[2]
            +4.573308850000000e-10 * tc[3]
            -8.859890850000000e-14 * tc[4];
        /*species 8: HO2 */
        species[8] =
            +2.948080400000000e+02 * invT
            +5.851355599999999e-01
            -4.301798010000000e+00 * tc[0]
            +2.374560255000000e-03 * tc[1]
            -3.526381516666666e-06 * tc[2]
            +2.023032450000000e-09 * tc[3]
            -4.646125620000001e-13 * tc[4];
        /*species 9: CO */
        species[9] =
            -1.434408600000000e+04 * invT
            +7.112418999999992e-02
            -3.579533470000000e+00 * tc[0]
            +3.051768400000000e-04 * tc[1]
            -1.694690550000000e-07 * tc[2]
            -7.558382366666667e-11 * tc[3]
            +4.522122495000000e-14 * tc[4];
        /*species 10: CH3 */
        species[10] =
            +1.642271600000000e+04 * invT
            +1.983644300000000e+00
            -3.657179700000000e+00 * tc[0]
            -1.063298950000000e-03 * tc[1]
            -9.097313833333333e-07 * tc[2]
            +5.515083583333334e-10 * tc[3]
            -1.232853700000000e-13 * tc[4];
        /*species 11: CH2O */
        species[11] =
            -1.430895670000000e+04 * invT
            +4.190910250000000e+00
            -4.793723150000000e+00 * tc[0]
            +4.954166845000000e-03 * tc[1]
            -6.220333466666666e-06 * tc[2]
            +3.160710508333333e-09 * tc[3]
            -6.588632600000000e-13 * tc[4];
        /*species 12: CO2 */
        species[12] =
            -4.837196970000000e+04 * invT
            -7.544278700000000e+00
            -2.356773520000000e+00 * tc[0]
            -4.492298385000000e-03 * tc[1]
            +1.187260448333333e-06 * tc[2]
            -2.049325183333333e-10 * tc[3]
            +7.184977399999999e-15 * tc[4];
        /*species 13: CH4 */
        species[13] =
            -1.024659830000000e+04 * invT
            +9.787603100000000e+00
            -5.149114680000000e+00 * tc[0]
            +6.831100450000000e-03 * tc[1]
            -8.190898683333333e-06 * tc[2]
            +4.035389725000000e-09 * tc[3]
            -8.330172050000000e-13 * tc[4];
        /*species 14: C2H2 */
        species[14] =
            +2.642898070000000e+04 * invT
            -1.313102400600000e+01
            -8.086810940000000e-01 * tc[0]
            -1.168078145000000e-02 * tc[1]
            +5.919530250000000e-06 * tc[2]
            -2.334603641666667e-09 * tc[3]
            +4.250364870000000e-13 * tc[4];
        /*species 15: C2H4 */
        species[15] =
            +5.089775930000000e+03 * invT
            -1.381294799999999e-01
            -3.959201480000000e+00 * tc[0]
            +3.785261235000000e-03 * tc[1]
            -9.516504866666667e-06 * tc[2]
            +5.763239608333333e-09 * tc[3]
            -1.349421865000000e-12 * tc[4];
        /*species 16: CH2CO */
        species[16] =
            -7.042918040000000e+03 * invT
            -1.007981170000000e+01
            -2.135836300000000e+00 * tc[0]
            -9.059436050000000e-03 * tc[1]
            +2.899124566666666e-06 * tc[2]
            -7.786646400000000e-10 * tc[3]
            +1.007288075000000e-13 * tc[4];
        /*species 17: C2H6 */
        species[17] =
            -1.152220550000000e+04 * invT
            +1.624601760000000e+00
            -4.291424920000000e+00 * tc[0]
            +2.750771350000000e-03 * tc[1]
            -9.990638133333334e-06 * tc[2]
            +5.903885708333334e-09 * tc[3]
            -1.343428855000000e-12 * tc[4];
        /*species 18: C2H3 */
        species[18] =
            +3.485984680000000e+04 * invT
            -5.298073800000000e+00
            -3.212466450000000e+00 * tc[0]
            -7.573958100000000e-04 * tc[1]
            -4.320156866666666e-06 * tc[2]
            +2.980482058333333e-09 * tc[3]
            -7.357543650000000e-13 * tc[4];
        /*species 19: C2H5 */
        species[19] =
            +1.284162650000000e+04 * invT
            -4.007435600000004e-01
            -4.306465680000000e+00 * tc[0]
            +2.093294460000000e-03 * tc[1]
            -8.285713450000000e-06 * tc[2]
            +4.992721716666666e-09 * tc[3]
            -1.152545020000000e-12 * tc[4];
        /*species 20: HCCO */
        species[20] =
            +2.005944900000000e+04 * invT
            -1.023869560000000e+01
            -2.251721400000000e+00 * tc[0]
            -8.827510500000000e-03 * tc[1]
            +3.954850166666666e-06 * tc[2]
            -1.439646583333334e-09 * tc[3]
            +2.533240550000000e-13 * tc[4];
        /*species 21: CH3CHO */
        species[21] =
            -2.179732230000000e+04 * invT
            -1.634478794000000e+01
            -1.406538560000000e+00 * tc[0]
            -1.084922190000000e-02 * tc[1]
            +2.459554416666667e-06 * tc[2]
            -6.086962316666666e-10 * tc[3]
            +1.045597335000000e-13 * tc[4];
        /*species 22: CH2CHO */
        species[22] =
            +1.069433220000000e+03 * invT
            -1.791262397000000e+01
            -1.096857330000000e+00 * tc[0]
            -1.101143980000000e-02 * tc[1]
            +2.409724066666667e-06 * tc[2]
            -2.506496483333333e-10 * tc[3]
            -3.044964385000000e-14 * tc[4];
        /*species 23: C2H5O */
        species[23] =
            -3.352529250000000e+03 * invT
            -2.231351709200000e+01
            -4.944207080000000e-01 * tc[0]
            -1.358872170000000e-02 * tc[1]
            +2.765150166666667e-06 * tc[2]
            -4.293368333333333e-10 * tc[3]
            +3.242484575000000e-14 * tc[4];
    } else {
        /*species 0: N2 */
        species[0] =
            -9.227977000000000e+02 * invT
            -3.053888000000000e+00
            -2.926640000000000e+00 * tc[0]
            -7.439884000000000e-04 * tc[1]
            +9.474600000000001e-08 * tc[2]
            -8.414198333333333e-12 * tc[3]
            +3.376675500000000e-16 * tc[4];
        /*species 1: H */
        species[1] =
            +2.547365990000000e+04 * invT
            +2.946682924000000e+00
            -2.500000010000000e+00 * tc[0]
            +1.154214865000000e-11 * tc[1]
            -2.692699133333334e-15 * tc[2]
            +3.945960291666667e-19 * tc[3]
            -2.490986785000000e-23 * tc[4];
        /*species 2: H2 */
        species[2] =
            -9.501589220000000e+02 * invT
            +6.542302510000000e+00
            -3.337279200000000e+00 * tc[0]
            +2.470123655000000e-05 * tc[1]
            -8.324279633333333e-08 * tc[2]
            +1.496386616666667e-11 * tc[3]
            -1.001276880000000e-15 * tc[4];
        /*species 3: O */
        species[3] =
            +2.921757910000000e+04 * invT
            -2.214917859999999e+00
            -2.569420780000000e+00 * tc[0]
            +4.298705685000000e-05 * tc[1]
            -6.991409816666667e-09 * tc[2]
            +8.348149916666666e-13 * tc[3]
            -6.141684549999999e-17 * tc[4];
        /*species 4: OH */
        species[4] =
            +3.718857740000000e+03 * invT
            -2.836911870000000e+00
            -2.864728860000000e+00 * tc[0]
            -5.282522400000000e-04 * tc[1]
            +4.318045966666667e-08 * tc[2]
            -2.543488950000000e-12 * tc[3]
            +6.659793800000000e-17 * tc[4];
        /*species 5: O2 */
        species[5] =
            -1.088457720000000e+03 * invT
            -2.170693450000000e+00
            -3.282537840000000e+00 * tc[0]
            -7.415437700000000e-04 * tc[1]
            +1.263277781666667e-07 * tc[2]
            -1.745587958333333e-11 * tc[3]
            +1.083588970000000e-15 * tc[4];
        /*species 6: H2O2 */
        species[6] =
            -1.786178770000000e+04 * invT
            +1.248846229999999e+00
            -4.165002850000000e+00 * tc[0]
            -2.454158470000000e-03 * tc[1]
            +3.168987083333333e-07 * tc[2]
            -3.093216550000000e-11 * tc[3]
            +1.439541525000000e-15 * tc[4];
        /*species 7: H2O */
        species[7] =
            -3.000429710000000e+04 * invT
            -1.932777610000000e+00
            -3.033992490000000e+00 * tc[0]
            -1.088459020000000e-03 * tc[1]
            +2.734541966666666e-08 * tc[2]
            +8.086832250000000e-12 * tc[3]
            -8.410049600000000e-16 * tc[4];
        /*species 8: HO2 */
        species[8] =
            +1.118567130000000e+02 * invT
            +2.321087500000001e-01
            -4.017210900000000e+00 * tc[0]
            -1.119910065000000e-03 * tc[1]
            +1.056096916666667e-07 * tc[2]
            -9.520530833333334e-12 * tc[3]
            +5.395426750000000e-16 * tc[4];
        /*species 9: CO */
        species[9] =
            -1.415187240000000e+04 * invT
            -5.103502110000000e+00
            -2.715185610000000e+00 * tc[0]
            -1.031263715000000e-03 * tc[1]
            +1.664709618333334e-07 * tc[2]
            -1.917108400000000e-11 * tc[3]
            +1.018238580000000e-15 * tc[4];
        /*species 10: CH3 */
        species[10] =
            +1.650951300000000e+04 * invT
            -1.744359300000000e+00
            -2.978120600000000e+00 * tc[0]
            -2.898926000000000e-03 * tc[1]
            +3.292633333333333e-07 * tc[2]
            -2.560815833333334e-11 * tc[3]
            +8.958708000000000e-16 * tc[4];
        /*species 11: CH2O */
        species[11] =
            -1.399583230000000e+04 * invT
            -1.189563292000000e+01
            -1.760690080000000e+00 * tc[0]
            -4.600000410000000e-03 * tc[1]
            +7.370980216666666e-07 * tc[2]
            -8.386767666666666e-11 * tc[3]
            +4.419278200000001e-15 * tc[4];
        /*species 12: CO2 */
        species[12] =
            -4.875916600000000e+04 * invT
            +1.585822230000000e+00
            -3.857460290000000e+00 * tc[0]
            -2.207185130000000e-03 * tc[1]
            +3.691356733333334e-07 * tc[2]
            -4.362418233333334e-11 * tc[3]
            +2.360420820000000e-15 * tc[4];
        /*species 13: CH4 */
        species[13] =
            -1.000959360000000e+04 * invT
            -8.251800570000000e+00
            -1.653262260000000e+00 * tc[0]
            -5.013154950000000e-03 * tc[1]
            +5.527687300000000e-07 * tc[2]
            -4.470692816666667e-11 * tc[3]
            +1.573483790000000e-15 * tc[4];
        /*species 14: C2H2 */
        species[14] =
            +2.593599920000000e+04 * invT
            +5.377850850000001e+00
            -4.147569640000000e+00 * tc[0]
            -2.980833320000000e-03 * tc[1]
            +3.954914200000000e-07 * tc[2]
            -3.895101425000000e-11 * tc[3]
            +1.806176065000000e-15 * tc[4];
        /*species 15: C2H4 */
        species[15] =
            +4.939886140000000e+03 * invT
            -8.269258140000002e+00
            -2.036111160000000e+00 * tc[0]
            -7.322707550000000e-03 * tc[1]
            +1.118463191666667e-06 * tc[2]
            -1.226857691666667e-10 * tc[3]
            +6.285303050000000e-15 * tc[4];
        /*species 16: CH2CO */
        species[16] =
            -7.551053110000000e+03 * invT
            +3.879050115000000e+00
            -4.511297320000000e+00 * tc[0]
            -4.501798725000000e-03 * tc[1]
            +6.948993916666666e-07 * tc[2]
            -7.694549016666667e-11 * tc[3]
            +3.974191005000000e-15 * tc[4];
        /*species 17: C2H6 */
        species[17] =
            -1.142639320000000e+04 * invT
            -1.404372920000000e+01
            -1.071881500000000e+00 * tc[0]
            -1.084263385000000e-02 * tc[1]
            +1.670934450000000e-06 * tc[2]
            -1.845100008333333e-10 * tc[3]
            +9.500144500000000e-15 * tc[4];
        /*species 18: C2H3 */
        species[18] =
            +3.461287390000000e+04 * invT
            -4.770599780000000e+00
            -3.016724000000000e+00 * tc[0]
            -5.165114600000000e-03 * tc[1]
            +7.801372483333333e-07 * tc[2]
            -8.480274000000000e-11 * tc[3]
            +4.313035205000000e-15 * tc[4];
        /*species 19: C2H5 */
        species[19] =
            +1.285752000000000e+04 * invT
            -1.150777788000000e+01
            -1.954656420000000e+00 * tc[0]
            -8.698636100000001e-03 * tc[1]
            +1.330344446666667e-06 * tc[2]
            -1.460147408333333e-10 * tc[3]
            +7.482078800000000e-15 * tc[4];
        /*species 20: HCCO */
        species[20] =
            +1.932721500000000e+04 * invT
            +9.558465300000000e+00
            -5.628205800000000e+00 * tc[0]
            -2.042670050000000e-03 * tc[1]
            +2.655757833333333e-07 * tc[2]
            -2.385504333333333e-11 * tc[3]
            +9.703915999999999e-16 * tc[4];
        /*species 21: CH3CHO */
        species[21] =
            -2.216537010000000e+04 * invT
            -8.478134180000000e+00
            -2.685431120000000e+00 * tc[0]
            -8.840118649999999e-03 * tc[1]
            +1.442337898333333e-06 * tc[2]
            -1.697338241666667e-10 * tc[3]
            +9.381546749999999e-15 * tc[4];
        /*species 22: CH2CHO */
        species[22] =
            +8.331069900000000e+02 * invT
            -1.017781013000000e+01
            -2.426063570000000e+00 * tc[0]
            -8.620001049999999e-03 * tc[1]
            +1.628553531666667e-06 * tc[2]
            -2.221297266666667e-10 * tc[3]
            +1.410600390000000e-14 * tc[4];
        /*species 23: C2H5O */
        species[23] =
            -3.839326580000000e+03 * invT
            -1.041126121000000e+01
            -2.462623490000000e+00 * tc[0]
            -1.047519795000000e-02 * tc[1]
            +1.565486250000000e-06 * tc[2]
            -1.303671891666667e-10 * tc[3]
            -0.000000000000000e+00 * tc[4];
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
        /*species 0: C */
        species[0] =
            +8.544388320000000e+04 * invT
            -1.977068930000000e+00
            -2.554239550000000e+00 * tc[0]
            +1.607688620000000e-04 * tc[1]
            -1.222987075000000e-07 * tc[2]
            +6.101957408333333e-11 * tc[3]
            -1.332607230000000e-14 * tc[4];
        /*species 1: CH */
        species[1] =
            +7.079729340000000e+04 * invT
            +1.405805570000000e+00
            -3.489816650000000e+00 * tc[0]
            -1.619177705000000e-04 * tc[1]
            +2.814984416666667e-07 * tc[2]
            -2.635144391666666e-10 * tc[3]
            +7.030453350000001e-14 * tc[4];
        /*species 2: HCO */
        species[2] =
            +3.839564960000000e+03 * invT
            +8.268134100000002e-01
            -4.221185840000000e+00 * tc[0]
            +1.621962660000000e-03 * tc[1]
            -2.296657433333333e-06 * tc[2]
            +1.109534108333333e-09 * tc[3]
            -2.168844325000000e-13 * tc[4];
        /*species 3: TXCH2 */
        species[3] =
            +4.600404010000000e+04 * invT
            +2.200146820000000e+00
            -3.762678670000000e+00 * tc[0]
            -4.844360715000000e-04 * tc[1]
            -4.658164016666667e-07 * tc[2]
            +3.209092941666667e-10 * tc[3]
            -8.437085950000000e-14 * tc[4];
        /*species 4: SXCH2 */
        species[4] =
            +5.049681630000000e+04 * invT
            +4.967723077000000e+00
            -4.198604110000000e+00 * tc[0]
            +1.183307095000000e-03 * tc[1]
            -1.372160366666667e-06 * tc[2]
            +5.573466508333334e-10 * tc[3]
            -9.715736850000000e-14 * tc[4];
    } else {
        /*species 0: C */
        species[0] =
            +8.545129530000000e+04 * invT
            -2.308834850000000e+00
            -2.492668880000000e+00 * tc[0]
            -2.399446420000000e-05 * tc[1]
            +1.207225033333333e-08 * tc[2]
            -3.119091908333333e-12 * tc[3]
            +2.436389465000000e-16 * tc[4];
        /*species 1: CH */
        species[1] =
            +7.101243640000001e+04 * invT
            -2.606515260000000e+00
            -2.878464730000000e+00 * tc[0]
            -4.854568405000000e-04 * tc[1]
            -2.407427583333333e-08 * tc[2]
            +1.089065408333333e-11 * tc[3]
            -8.803969149999999e-16 * tc[4];
        /*species 2: HCO */
        species[2] =
            +4.011918150000000e+03 * invT
            -7.026170540000000e+00
            -2.772174380000000e+00 * tc[0]
            -2.478477630000000e-03 * tc[1]
            +4.140760216666667e-07 * tc[2]
            -4.909681483333334e-11 * tc[3]
            +2.667543555000000e-15 * tc[4];
        /*species 3: TXCH2 */
        species[3] =
            +4.626360400000000e+04 * invT
            -3.297092110000000e+00
            -2.874101130000000e+00 * tc[0]
            -1.828196460000000e-03 * tc[1]
            +2.348243283333333e-07 * tc[2]
            -2.168162908333333e-11 * tc[3]
            +9.386378350000000e-16 * tc[4];
        /*species 4: SXCH2 */
        species[4] =
            +5.092599970000000e+04 * invT
            -6.334463270000000e+00
            -2.292038420000000e+00 * tc[0]
            -2.327943185000000e-03 * tc[1]
            +3.353199116666667e-07 * tc[2]
            -3.482550000000000e-11 * tc[3]
            +1.698581825000000e-15 * tc[4];
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
            -1.02089990e+03 * invT
            -1.65169500e+00
            -3.29867700e+00 * tc[0]
            -7.04120200e-04 * tc[1]
            +6.60537000e-07 * tc[2]
            -4.70126250e-10 * tc[3]
            +1.22242700e-13 * tc[4];
        /*species 1: H */
        species[1] =
            +2.54736599e+04 * invT
            +1.94668285e+00
            -2.50000000e+00 * tc[0]
            -3.52666409e-13 * tc[1]
            +3.32653273e-16 * tc[2]
            -1.91734693e-19 * tc[3]
            +4.63866166e-23 * tc[4];
        /*species 2: H2 */
        species[2] =
            -9.17935173e+02 * invT
            +6.61320882e-01
            -2.34433112e+00 * tc[0]
            -3.99026037e-03 * tc[1]
            +3.24635850e-06 * tc[2]
            -1.67976745e-09 * tc[3]
            +3.68805881e-13 * tc[4];
        /*species 3: O */
        species[3] =
            +2.91222592e+04 * invT
            +1.16333640e-01
            -3.16826710e+00 * tc[0]
            +1.63965942e-03 * tc[1]
            -1.10717733e-06 * tc[2]
            +5.10672187e-10 * tc[3]
            -1.05632985e-13 * tc[4];
        /*species 4: OH */
        species[4] =
            +3.38153812e+03 * invT
            +3.81573857e+00
            -4.12530561e+00 * tc[0]
            +1.61272470e-03 * tc[1]
            -1.08794115e-06 * tc[2]
            +4.83211369e-10 * tc[3]
            -1.03118689e-13 * tc[4];
        /*species 5: O2 */
        species[5] =
            -1.06394356e+03 * invT
            -8.75219370e-01
            -3.78245636e+00 * tc[0]
            +1.49836708e-03 * tc[1]
            -1.64121700e-06 * tc[2]
            +8.06774591e-10 * tc[3]
            -1.62186418e-13 * tc[4];
        /*species 6: H2O2 */
        species[6] =
            -1.77025821e+04 * invT
            -1.58938050e-01
            -4.27611269e+00 * tc[0]
            +2.71411208e-04 * tc[1]
            -2.78892835e-06 * tc[2]
            +1.79809011e-09 * tc[3]
            -4.31227182e-13 * tc[4];
        /*species 7: H2O */
        species[7] =
            -3.02937267e+04 * invT
            +4.04767277e+00
            -4.19864056e+00 * tc[0]
            +1.01821705e-03 * tc[1]
            -1.08673369e-06 * tc[2]
            +4.57330885e-10 * tc[3]
            -8.85989085e-14 * tc[4];
        /*species 8: HO2 */
        species[8] =
            +2.94808040e+02 * invT
            -4.14864440e-01
            -4.30179801e+00 * tc[0]
            +2.37456025e-03 * tc[1]
            -3.52638152e-06 * tc[2]
            +2.02303245e-09 * tc[3]
            -4.64612562e-13 * tc[4];
        /*species 9: CO */
        species[9] =
            -1.43440860e+04 * invT
            -9.28875810e-01
            -3.57953347e+00 * tc[0]
            +3.05176840e-04 * tc[1]
            -1.69469055e-07 * tc[2]
            -7.55838237e-11 * tc[3]
            +4.52212249e-14 * tc[4];
        /*species 10: CH3 */
        species[10] =
            +1.64227160e+04 * invT
            +9.83644300e-01
            -3.65717970e+00 * tc[0]
            -1.06329895e-03 * tc[1]
            -9.09731383e-07 * tc[2]
            +5.51508358e-10 * tc[3]
            -1.23285370e-13 * tc[4];
        /*species 11: CH2O */
        species[11] =
            -1.43089567e+04 * invT
            +3.19091025e+00
            -4.79372315e+00 * tc[0]
            +4.95416684e-03 * tc[1]
            -6.22033347e-06 * tc[2]
            +3.16071051e-09 * tc[3]
            -6.58863260e-13 * tc[4];
        /*species 12: CO2 */
        species[12] =
            -4.83719697e+04 * invT
            -8.54427870e+00
            -2.35677352e+00 * tc[0]
            -4.49229839e-03 * tc[1]
            +1.18726045e-06 * tc[2]
            -2.04932518e-10 * tc[3]
            +7.18497740e-15 * tc[4];
        /*species 13: CH4 */
        species[13] =
            -1.02465983e+04 * invT
            +8.78760310e+00
            -5.14911468e+00 * tc[0]
            +6.83110045e-03 * tc[1]
            -8.19089868e-06 * tc[2]
            +4.03538972e-09 * tc[3]
            -8.33017205e-13 * tc[4];
        /*species 14: C2H2 */
        species[14] =
            +2.64289807e+04 * invT
            -1.41310240e+01
            -8.08681094e-01 * tc[0]
            -1.16807815e-02 * tc[1]
            +5.91953025e-06 * tc[2]
            -2.33460364e-09 * tc[3]
            +4.25036487e-13 * tc[4];
        /*species 15: C2H4 */
        species[15] =
            +5.08977593e+03 * invT
            -1.13812948e+00
            -3.95920148e+00 * tc[0]
            +3.78526124e-03 * tc[1]
            -9.51650487e-06 * tc[2]
            +5.76323961e-09 * tc[3]
            -1.34942187e-12 * tc[4];
        /*species 16: CH2CO */
        species[16] =
            -7.04291804e+03 * invT
            -1.10798117e+01
            -2.13583630e+00 * tc[0]
            -9.05943605e-03 * tc[1]
            +2.89912457e-06 * tc[2]
            -7.78664640e-10 * tc[3]
            +1.00728807e-13 * tc[4];
        /*species 17: C2H6 */
        species[17] =
            -1.15222055e+04 * invT
            +6.24601760e-01
            -4.29142492e+00 * tc[0]
            +2.75077135e-03 * tc[1]
            -9.99063813e-06 * tc[2]
            +5.90388571e-09 * tc[3]
            -1.34342886e-12 * tc[4];
        /*species 18: C2H3 */
        species[18] =
            +3.48598468e+04 * invT
            -6.29807380e+00
            -3.21246645e+00 * tc[0]
            -7.57395810e-04 * tc[1]
            -4.32015687e-06 * tc[2]
            +2.98048206e-09 * tc[3]
            -7.35754365e-13 * tc[4];
        /*species 19: C2H5 */
        species[19] =
            +1.28416265e+04 * invT
            -1.40074356e+00
            -4.30646568e+00 * tc[0]
            +2.09329446e-03 * tc[1]
            -8.28571345e-06 * tc[2]
            +4.99272172e-09 * tc[3]
            -1.15254502e-12 * tc[4];
        /*species 20: HCCO */
        species[20] =
            +2.00594490e+04 * invT
            -1.12386956e+01
            -2.25172140e+00 * tc[0]
            -8.82751050e-03 * tc[1]
            +3.95485017e-06 * tc[2]
            -1.43964658e-09 * tc[3]
            +2.53324055e-13 * tc[4];
        /*species 21: CH3CHO */
        species[21] =
            -2.17973223e+04 * invT
            -1.73447879e+01
            -1.40653856e+00 * tc[0]
            -1.08492219e-02 * tc[1]
            +2.45955442e-06 * tc[2]
            -6.08696232e-10 * tc[3]
            +1.04559734e-13 * tc[4];
        /*species 22: CH2CHO */
        species[22] =
            +1.06943322e+03 * invT
            -1.89126240e+01
            -1.09685733e+00 * tc[0]
            -1.10114398e-02 * tc[1]
            +2.40972407e-06 * tc[2]
            -2.50649648e-10 * tc[3]
            -3.04496438e-14 * tc[4];
        /*species 23: C2H5O */
        species[23] =
            -3.35252925e+03 * invT
            -2.33135171e+01
            -4.94420708e-01 * tc[0]
            -1.35887217e-02 * tc[1]
            +2.76515017e-06 * tc[2]
            -4.29336833e-10 * tc[3]
            +3.24248457e-14 * tc[4];
    } else {
        /*species 0: N2 */
        species[0] =
            -9.22797700e+02 * invT
            -4.05388800e+00
            -2.92664000e+00 * tc[0]
            -7.43988400e-04 * tc[1]
            +9.47460000e-08 * tc[2]
            -8.41419833e-12 * tc[3]
            +3.37667550e-16 * tc[4];
        /*species 1: H */
        species[1] =
            +2.54736599e+04 * invT
            +1.94668292e+00
            -2.50000001e+00 * tc[0]
            +1.15421486e-11 * tc[1]
            -2.69269913e-15 * tc[2]
            +3.94596029e-19 * tc[3]
            -2.49098679e-23 * tc[4];
        /*species 2: H2 */
        species[2] =
            -9.50158922e+02 * invT
            +5.54230251e+00
            -3.33727920e+00 * tc[0]
            +2.47012365e-05 * tc[1]
            -8.32427963e-08 * tc[2]
            +1.49638662e-11 * tc[3]
            -1.00127688e-15 * tc[4];
        /*species 3: O */
        species[3] =
            +2.92175791e+04 * invT
            -3.21491786e+00
            -2.56942078e+00 * tc[0]
            +4.29870569e-05 * tc[1]
            -6.99140982e-09 * tc[2]
            +8.34814992e-13 * tc[3]
            -6.14168455e-17 * tc[4];
        /*species 4: OH */
        species[4] =
            +3.71885774e+03 * invT
            -3.83691187e+00
            -2.86472886e+00 * tc[0]
            -5.28252240e-04 * tc[1]
            +4.31804597e-08 * tc[2]
            -2.54348895e-12 * tc[3]
            +6.65979380e-17 * tc[4];
        /*species 5: O2 */
        species[5] =
            -1.08845772e+03 * invT
            -3.17069345e+00
            -3.28253784e+00 * tc[0]
            -7.41543770e-04 * tc[1]
            +1.26327778e-07 * tc[2]
            -1.74558796e-11 * tc[3]
            +1.08358897e-15 * tc[4];
        /*species 6: H2O2 */
        species[6] =
            -1.78617877e+04 * invT
            +2.48846230e-01
            -4.16500285e+00 * tc[0]
            -2.45415847e-03 * tc[1]
            +3.16898708e-07 * tc[2]
            -3.09321655e-11 * tc[3]
            +1.43954153e-15 * tc[4];
        /*species 7: H2O */
        species[7] =
            -3.00042971e+04 * invT
            -2.93277761e+00
            -3.03399249e+00 * tc[0]
            -1.08845902e-03 * tc[1]
            +2.73454197e-08 * tc[2]
            +8.08683225e-12 * tc[3]
            -8.41004960e-16 * tc[4];
        /*species 8: HO2 */
        species[8] =
            +1.11856713e+02 * invT
            -7.67891250e-01
            -4.01721090e+00 * tc[0]
            -1.11991006e-03 * tc[1]
            +1.05609692e-07 * tc[2]
            -9.52053083e-12 * tc[3]
            +5.39542675e-16 * tc[4];
        /*species 9: CO */
        species[9] =
            -1.41518724e+04 * invT
            -6.10350211e+00
            -2.71518561e+00 * tc[0]
            -1.03126372e-03 * tc[1]
            +1.66470962e-07 * tc[2]
            -1.91710840e-11 * tc[3]
            +1.01823858e-15 * tc[4];
        /*species 10: CH3 */
        species[10] =
            +1.65095130e+04 * invT
            -2.74435930e+00
            -2.97812060e+00 * tc[0]
            -2.89892600e-03 * tc[1]
            +3.29263333e-07 * tc[2]
            -2.56081583e-11 * tc[3]
            +8.95870800e-16 * tc[4];
        /*species 11: CH2O */
        species[11] =
            -1.39958323e+04 * invT
            -1.28956329e+01
            -1.76069008e+00 * tc[0]
            -4.60000041e-03 * tc[1]
            +7.37098022e-07 * tc[2]
            -8.38676767e-11 * tc[3]
            +4.41927820e-15 * tc[4];
        /*species 12: CO2 */
        species[12] =
            -4.87591660e+04 * invT
            +5.85822230e-01
            -3.85746029e+00 * tc[0]
            -2.20718513e-03 * tc[1]
            +3.69135673e-07 * tc[2]
            -4.36241823e-11 * tc[3]
            +2.36042082e-15 * tc[4];
        /*species 13: CH4 */
        species[13] =
            -1.00095936e+04 * invT
            -9.25180057e+00
            -1.65326226e+00 * tc[0]
            -5.01315495e-03 * tc[1]
            +5.52768730e-07 * tc[2]
            -4.47069282e-11 * tc[3]
            +1.57348379e-15 * tc[4];
        /*species 14: C2H2 */
        species[14] =
            +2.59359992e+04 * invT
            +4.37785085e+00
            -4.14756964e+00 * tc[0]
            -2.98083332e-03 * tc[1]
            +3.95491420e-07 * tc[2]
            -3.89510143e-11 * tc[3]
            +1.80617607e-15 * tc[4];
        /*species 15: C2H4 */
        species[15] =
            +4.93988614e+03 * invT
            -9.26925814e+00
            -2.03611116e+00 * tc[0]
            -7.32270755e-03 * tc[1]
            +1.11846319e-06 * tc[2]
            -1.22685769e-10 * tc[3]
            +6.28530305e-15 * tc[4];
        /*species 16: CH2CO */
        species[16] =
            -7.55105311e+03 * invT
            +2.87905011e+00
            -4.51129732e+00 * tc[0]
            -4.50179872e-03 * tc[1]
            +6.94899392e-07 * tc[2]
            -7.69454902e-11 * tc[3]
            +3.97419100e-15 * tc[4];
        /*species 17: C2H6 */
        species[17] =
            -1.14263932e+04 * invT
            -1.50437292e+01
            -1.07188150e+00 * tc[0]
            -1.08426339e-02 * tc[1]
            +1.67093445e-06 * tc[2]
            -1.84510001e-10 * tc[3]
            +9.50014450e-15 * tc[4];
        /*species 18: C2H3 */
        species[18] =
            +3.46128739e+04 * invT
            -5.77059978e+00
            -3.01672400e+00 * tc[0]
            -5.16511460e-03 * tc[1]
            +7.80137248e-07 * tc[2]
            -8.48027400e-11 * tc[3]
            +4.31303520e-15 * tc[4];
        /*species 19: C2H5 */
        species[19] =
            +1.28575200e+04 * invT
            -1.25077779e+01
            -1.95465642e+00 * tc[0]
            -8.69863610e-03 * tc[1]
            +1.33034445e-06 * tc[2]
            -1.46014741e-10 * tc[3]
            +7.48207880e-15 * tc[4];
        /*species 20: HCCO */
        species[20] =
            +1.93272150e+04 * invT
            +8.55846530e+00
            -5.62820580e+00 * tc[0]
            -2.04267005e-03 * tc[1]
            +2.65575783e-07 * tc[2]
            -2.38550433e-11 * tc[3]
            +9.70391600e-16 * tc[4];
        /*species 21: CH3CHO */
        species[21] =
            -2.21653701e+04 * invT
            -9.47813418e+00
            -2.68543112e+00 * tc[0]
            -8.84011865e-03 * tc[1]
            +1.44233790e-06 * tc[2]
            -1.69733824e-10 * tc[3]
            +9.38154675e-15 * tc[4];
        /*species 22: CH2CHO */
        species[22] =
            +8.33106990e+02 * invT
            -1.11778101e+01
            -2.42606357e+00 * tc[0]
            -8.62000105e-03 * tc[1]
            +1.62855353e-06 * tc[2]
            -2.22129727e-10 * tc[3]
            +1.41060039e-14 * tc[4];
        /*species 23: C2H5O */
        species[23] =
            -3.83932658e+03 * invT
            -1.14112612e+01
            -2.46262349e+00 * tc[0]
            -1.04751979e-02 * tc[1]
            +1.56548625e-06 * tc[2]
            -1.30367189e-10 * tc[3]
            -0.00000000e+00 * tc[4];
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
            +1.40824040e-03 * tc[1]
            -3.96322200e-06 * tc[2]
            +5.64151500e-09 * tc[3]
            -2.44485400e-12 * tc[4];
        /*species 1: H */
        species[1] =
            +1.50000000e+00
            +7.05332819e-13 * tc[1]
            -1.99591964e-15 * tc[2]
            +2.30081632e-18 * tc[3]
            -9.27732332e-22 * tc[4];
        /*species 2: H2 */
        species[2] =
            +1.34433112e+00
            +7.98052075e-03 * tc[1]
            -1.94781510e-05 * tc[2]
            +2.01572094e-08 * tc[3]
            -7.37611761e-12 * tc[4];
        /*species 3: O */
        species[3] =
            +2.16826710e+00
            -3.27931884e-03 * tc[1]
            +6.64306396e-06 * tc[2]
            -6.12806624e-09 * tc[3]
            +2.11265971e-12 * tc[4];
        /*species 4: OH */
        species[4] =
            +3.12530561e+00
            -3.22544939e-03 * tc[1]
            +6.52764691e-06 * tc[2]
            -5.79853643e-09 * tc[3]
            +2.06237379e-12 * tc[4];
        /*species 5: O2 */
        species[5] =
            +2.78245636e+00
            -2.99673416e-03 * tc[1]
            +9.84730201e-06 * tc[2]
            -9.68129509e-09 * tc[3]
            +3.24372837e-12 * tc[4];
        /*species 6: H2O2 */
        species[6] =
            +3.27611269e+00
            -5.42822417e-04 * tc[1]
            +1.67335701e-05 * tc[2]
            -2.15770813e-08 * tc[3]
            +8.62454363e-12 * tc[4];
        /*species 7: H2O */
        species[7] =
            +3.19864056e+00
            -2.03643410e-03 * tc[1]
            +6.52040211e-06 * tc[2]
            -5.48797062e-09 * tc[3]
            +1.77197817e-12 * tc[4];
        /*species 8: HO2 */
        species[8] =
            +3.30179801e+00
            -4.74912051e-03 * tc[1]
            +2.11582891e-05 * tc[2]
            -2.42763894e-08 * tc[3]
            +9.29225124e-12 * tc[4];
        /*species 9: CO */
        species[9] =
            +2.57953347e+00
            -6.10353680e-04 * tc[1]
            +1.01681433e-06 * tc[2]
            +9.07005884e-10 * tc[3]
            -9.04424499e-13 * tc[4];
        /*species 10: CH3 */
        species[10] =
            +2.65717970e+00
            +2.12659790e-03 * tc[1]
            +5.45838830e-06 * tc[2]
            -6.61810030e-09 * tc[3]
            +2.46570740e-12 * tc[4];
        /*species 11: CH2O */
        species[11] =
            +3.79372315e+00
            -9.90833369e-03 * tc[1]
            +3.73220008e-05 * tc[2]
            -3.79285261e-08 * tc[3]
            +1.31772652e-11 * tc[4];
        /*species 12: CO2 */
        species[12] =
            +1.35677352e+00
            +8.98459677e-03 * tc[1]
            -7.12356269e-06 * tc[2]
            +2.45919022e-09 * tc[3]
            -1.43699548e-13 * tc[4];
        /*species 13: CH4 */
        species[13] =
            +4.14911468e+00
            -1.36622009e-02 * tc[1]
            +4.91453921e-05 * tc[2]
            -4.84246767e-08 * tc[3]
            +1.66603441e-11 * tc[4];
        /*species 14: C2H2 */
        species[14] =
            -1.91318906e-01
            +2.33615629e-02 * tc[1]
            -3.55171815e-05 * tc[2]
            +2.80152437e-08 * tc[3]
            -8.50072974e-12 * tc[4];
        /*species 15: C2H4 */
        species[15] =
            +2.95920148e+00
            -7.57052247e-03 * tc[1]
            +5.70990292e-05 * tc[2]
            -6.91588753e-08 * tc[3]
            +2.69884373e-11 * tc[4];
        /*species 16: CH2CO */
        species[16] =
            +1.13583630e+00
            +1.81188721e-02 * tc[1]
            -1.73947474e-05 * tc[2]
            +9.34397568e-09 * tc[3]
            -2.01457615e-12 * tc[4];
        /*species 17: C2H6 */
        species[17] =
            +3.29142492e+00
            -5.50154270e-03 * tc[1]
            +5.99438288e-05 * tc[2]
            -7.08466285e-08 * tc[3]
            +2.68685771e-11 * tc[4];
        /*species 18: C2H3 */
        species[18] =
            +2.21246645e+00
            +1.51479162e-03 * tc[1]
            +2.59209412e-05 * tc[2]
            -3.57657847e-08 * tc[3]
            +1.47150873e-11 * tc[4];
        /*species 19: C2H5 */
        species[19] =
            +3.30646568e+00
            -4.18658892e-03 * tc[1]
            +4.97142807e-05 * tc[2]
            -5.99126606e-08 * tc[3]
            +2.30509004e-11 * tc[4];
        /*species 20: HCCO */
        species[20] =
            +1.25172140e+00
            +1.76550210e-02 * tc[1]
            -2.37291010e-05 * tc[2]
            +1.72757590e-08 * tc[3]
            -5.06648110e-12 * tc[4];
        /*species 21: CH3CHO */
        species[21] =
            +4.06538560e-01
            +2.16984438e-02 * tc[1]
            -1.47573265e-05 * tc[2]
            +7.30435478e-09 * tc[3]
            -2.09119467e-12 * tc[4];
        /*species 22: CH2CHO */
        species[22] =
            +9.68573300e-02
            +2.20228796e-02 * tc[1]
            -1.44583444e-05 * tc[2]
            +3.00779578e-09 * tc[3]
            +6.08992877e-13 * tc[4];
        /*species 23: C2H5O */
        species[23] =
            -5.05579292e-01
            +2.71774434e-02 * tc[1]
            -1.65909010e-05 * tc[2]
            +5.15204200e-09 * tc[3]
            -6.48496915e-13 * tc[4];
    } else {
        /*species 0: N2 */
        species[0] =
            +1.92664000e+00
            +1.48797680e-03 * tc[1]
            -5.68476000e-07 * tc[2]
            +1.00970380e-10 * tc[3]
            -6.75335100e-15 * tc[4];
        /*species 1: H */
        species[1] =
            +1.50000001e+00
            -2.30842973e-11 * tc[1]
            +1.61561948e-14 * tc[2]
            -4.73515235e-18 * tc[3]
            +4.98197357e-22 * tc[4];
        /*species 2: H2 */
        species[2] =
            +2.33727920e+00
            -4.94024731e-05 * tc[1]
            +4.99456778e-07 * tc[2]
            -1.79566394e-10 * tc[3]
            +2.00255376e-14 * tc[4];
        /*species 3: O */
        species[3] =
            +1.56942078e+00
            -8.59741137e-05 * tc[1]
            +4.19484589e-08 * tc[2]
            -1.00177799e-11 * tc[3]
            +1.22833691e-15 * tc[4];
        /*species 4: OH */
        species[4] =
            +1.86472886e+00
            +1.05650448e-03 * tc[1]
            -2.59082758e-07 * tc[2]
            +3.05218674e-11 * tc[3]
            -1.33195876e-15 * tc[4];
        /*species 5: O2 */
        species[5] =
            +2.28253784e+00
            +1.48308754e-03 * tc[1]
            -7.57966669e-07 * tc[2]
            +2.09470555e-10 * tc[3]
            -2.16717794e-14 * tc[4];
        /*species 6: H2O2 */
        species[6] =
            +3.16500285e+00
            +4.90831694e-03 * tc[1]
            -1.90139225e-06 * tc[2]
            +3.71185986e-10 * tc[3]
            -2.87908305e-14 * tc[4];
        /*species 7: H2O */
        species[7] =
            +2.03399249e+00
            +2.17691804e-03 * tc[1]
            -1.64072518e-07 * tc[2]
            -9.70419870e-11 * tc[3]
            +1.68200992e-14 * tc[4];
        /*species 8: HO2 */
        species[8] =
            +3.01721090e+00
            +2.23982013e-03 * tc[1]
            -6.33658150e-07 * tc[2]
            +1.14246370e-10 * tc[3]
            -1.07908535e-14 * tc[4];
        /*species 9: CO */
        species[9] =
            +1.71518561e+00
            +2.06252743e-03 * tc[1]
            -9.98825771e-07 * tc[2]
            +2.30053008e-10 * tc[3]
            -2.03647716e-14 * tc[4];
        /*species 10: CH3 */
        species[10] =
            +1.97812060e+00
            +5.79785200e-03 * tc[1]
            -1.97558000e-06 * tc[2]
            +3.07297900e-10 * tc[3]
            -1.79174160e-14 * tc[4];
        /*species 11: CH2O */
        species[11] =
            +7.60690080e-01
            +9.20000082e-03 * tc[1]
            -4.42258813e-06 * tc[2]
            +1.00641212e-09 * tc[3]
            -8.83855640e-14 * tc[4];
        /*species 12: CO2 */
        species[12] =
            +2.85746029e+00
            +4.41437026e-03 * tc[1]
            -2.21481404e-06 * tc[2]
            +5.23490188e-10 * tc[3]
            -4.72084164e-14 * tc[4];
        /*species 13: CH4 */
        species[13] =
            +6.53262260e-01
            +1.00263099e-02 * tc[1]
            -3.31661238e-06 * tc[2]
            +5.36483138e-10 * tc[3]
            -3.14696758e-14 * tc[4];
        /*species 14: C2H2 */
        species[14] =
            +3.14756964e+00
            +5.96166664e-03 * tc[1]
            -2.37294852e-06 * tc[2]
            +4.67412171e-10 * tc[3]
            -3.61235213e-14 * tc[4];
        /*species 15: C2H4 */
        species[15] =
            +1.03611116e+00
            +1.46454151e-02 * tc[1]
            -6.71077915e-06 * tc[2]
            +1.47222923e-09 * tc[3]
            -1.25706061e-13 * tc[4];
        /*species 16: CH2CO */
        species[16] =
            +3.51129732e+00
            +9.00359745e-03 * tc[1]
            -4.16939635e-06 * tc[2]
            +9.23345882e-10 * tc[3]
            -7.94838201e-14 * tc[4];
        /*species 17: C2H6 */
        species[17] =
            +7.18815000e-02
            +2.16852677e-02 * tc[1]
            -1.00256067e-05 * tc[2]
            +2.21412001e-09 * tc[3]
            -1.90002890e-13 * tc[4];
        /*species 18: C2H3 */
        species[18] =
            +2.01672400e+00
            +1.03302292e-02 * tc[1]
            -4.68082349e-06 * tc[2]
            +1.01763288e-09 * tc[3]
            -8.62607041e-14 * tc[4];
        /*species 19: C2H5 */
        species[19] =
            +9.54656420e-01
            +1.73972722e-02 * tc[1]
            -7.98206668e-06 * tc[2]
            +1.75217689e-09 * tc[3]
            -1.49641576e-13 * tc[4];
        /*species 20: HCCO */
        species[20] =
            +4.62820580e+00
            +4.08534010e-03 * tc[1]
            -1.59345470e-06 * tc[2]
            +2.86260520e-10 * tc[3]
            -1.94078320e-14 * tc[4];
        /*species 21: CH3CHO */
        species[21] =
            +1.68543112e+00
            +1.76802373e-02 * tc[1]
            -8.65402739e-06 * tc[2]
            +2.03680589e-09 * tc[3]
            -1.87630935e-13 * tc[4];
        /*species 22: CH2CHO */
        species[22] =
            +1.42606357e+00
            +1.72400021e-02 * tc[1]
            -9.77132119e-06 * tc[2]
            +2.66555672e-09 * tc[3]
            -2.82120078e-13 * tc[4];
        /*species 23: C2H5O */
        species[23] =
            +1.46262349e+00
            +2.09503959e-02 * tc[1]
            -9.39291750e-06 * tc[2]
            +1.56440627e-09 * tc[3]
            +0.00000000e+00 * tc[4];
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
            +1.40824040e-03 * tc[1]
            -3.96322200e-06 * tc[2]
            +5.64151500e-09 * tc[3]
            -2.44485400e-12 * tc[4];
        /*species 1: H */
        species[1] =
            +2.50000000e+00
            +7.05332819e-13 * tc[1]
            -1.99591964e-15 * tc[2]
            +2.30081632e-18 * tc[3]
            -9.27732332e-22 * tc[4];
        /*species 2: H2 */
        species[2] =
            +2.34433112e+00
            +7.98052075e-03 * tc[1]
            -1.94781510e-05 * tc[2]
            +2.01572094e-08 * tc[3]
            -7.37611761e-12 * tc[4];
        /*species 3: O */
        species[3] =
            +3.16826710e+00
            -3.27931884e-03 * tc[1]
            +6.64306396e-06 * tc[2]
            -6.12806624e-09 * tc[3]
            +2.11265971e-12 * tc[4];
        /*species 4: OH */
        species[4] =
            +4.12530561e+00
            -3.22544939e-03 * tc[1]
            +6.52764691e-06 * tc[2]
            -5.79853643e-09 * tc[3]
            +2.06237379e-12 * tc[4];
        /*species 5: O2 */
        species[5] =
            +3.78245636e+00
            -2.99673416e-03 * tc[1]
            +9.84730201e-06 * tc[2]
            -9.68129509e-09 * tc[3]
            +3.24372837e-12 * tc[4];
        /*species 6: H2O2 */
        species[6] =
            +4.27611269e+00
            -5.42822417e-04 * tc[1]
            +1.67335701e-05 * tc[2]
            -2.15770813e-08 * tc[3]
            +8.62454363e-12 * tc[4];
        /*species 7: H2O */
        species[7] =
            +4.19864056e+00
            -2.03643410e-03 * tc[1]
            +6.52040211e-06 * tc[2]
            -5.48797062e-09 * tc[3]
            +1.77197817e-12 * tc[4];
        /*species 8: HO2 */
        species[8] =
            +4.30179801e+00
            -4.74912051e-03 * tc[1]
            +2.11582891e-05 * tc[2]
            -2.42763894e-08 * tc[3]
            +9.29225124e-12 * tc[4];
        /*species 9: CO */
        species[9] =
            +3.57953347e+00
            -6.10353680e-04 * tc[1]
            +1.01681433e-06 * tc[2]
            +9.07005884e-10 * tc[3]
            -9.04424499e-13 * tc[4];
        /*species 10: CH3 */
        species[10] =
            +3.65717970e+00
            +2.12659790e-03 * tc[1]
            +5.45838830e-06 * tc[2]
            -6.61810030e-09 * tc[3]
            +2.46570740e-12 * tc[4];
        /*species 11: CH2O */
        species[11] =
            +4.79372315e+00
            -9.90833369e-03 * tc[1]
            +3.73220008e-05 * tc[2]
            -3.79285261e-08 * tc[3]
            +1.31772652e-11 * tc[4];
        /*species 12: CO2 */
        species[12] =
            +2.35677352e+00
            +8.98459677e-03 * tc[1]
            -7.12356269e-06 * tc[2]
            +2.45919022e-09 * tc[3]
            -1.43699548e-13 * tc[4];
        /*species 13: CH4 */
        species[13] =
            +5.14911468e+00
            -1.36622009e-02 * tc[1]
            +4.91453921e-05 * tc[2]
            -4.84246767e-08 * tc[3]
            +1.66603441e-11 * tc[4];
        /*species 14: C2H2 */
        species[14] =
            +8.08681094e-01
            +2.33615629e-02 * tc[1]
            -3.55171815e-05 * tc[2]
            +2.80152437e-08 * tc[3]
            -8.50072974e-12 * tc[4];
        /*species 15: C2H4 */
        species[15] =
            +3.95920148e+00
            -7.57052247e-03 * tc[1]
            +5.70990292e-05 * tc[2]
            -6.91588753e-08 * tc[3]
            +2.69884373e-11 * tc[4];
        /*species 16: CH2CO */
        species[16] =
            +2.13583630e+00
            +1.81188721e-02 * tc[1]
            -1.73947474e-05 * tc[2]
            +9.34397568e-09 * tc[3]
            -2.01457615e-12 * tc[4];
        /*species 17: C2H6 */
        species[17] =
            +4.29142492e+00
            -5.50154270e-03 * tc[1]
            +5.99438288e-05 * tc[2]
            -7.08466285e-08 * tc[3]
            +2.68685771e-11 * tc[4];
        /*species 18: C2H3 */
        species[18] =
            +3.21246645e+00
            +1.51479162e-03 * tc[1]
            +2.59209412e-05 * tc[2]
            -3.57657847e-08 * tc[3]
            +1.47150873e-11 * tc[4];
        /*species 19: C2H5 */
        species[19] =
            +4.30646568e+00
            -4.18658892e-03 * tc[1]
            +4.97142807e-05 * tc[2]
            -5.99126606e-08 * tc[3]
            +2.30509004e-11 * tc[4];
        /*species 20: HCCO */
        species[20] =
            +2.25172140e+00
            +1.76550210e-02 * tc[1]
            -2.37291010e-05 * tc[2]
            +1.72757590e-08 * tc[3]
            -5.06648110e-12 * tc[4];
        /*species 21: CH3CHO */
        species[21] =
            +1.40653856e+00
            +2.16984438e-02 * tc[1]
            -1.47573265e-05 * tc[2]
            +7.30435478e-09 * tc[3]
            -2.09119467e-12 * tc[4];
        /*species 22: CH2CHO */
        species[22] =
            +1.09685733e+00
            +2.20228796e-02 * tc[1]
            -1.44583444e-05 * tc[2]
            +3.00779578e-09 * tc[3]
            +6.08992877e-13 * tc[4];
        /*species 23: C2H5O */
        species[23] =
            +4.94420708e-01
            +2.71774434e-02 * tc[1]
            -1.65909010e-05 * tc[2]
            +5.15204200e-09 * tc[3]
            -6.48496915e-13 * tc[4];
    } else {
        /*species 0: N2 */
        species[0] =
            +2.92664000e+00
            +1.48797680e-03 * tc[1]
            -5.68476000e-07 * tc[2]
            +1.00970380e-10 * tc[3]
            -6.75335100e-15 * tc[4];
        /*species 1: H */
        species[1] =
            +2.50000001e+00
            -2.30842973e-11 * tc[1]
            +1.61561948e-14 * tc[2]
            -4.73515235e-18 * tc[3]
            +4.98197357e-22 * tc[4];
        /*species 2: H2 */
        species[2] =
            +3.33727920e+00
            -4.94024731e-05 * tc[1]
            +4.99456778e-07 * tc[2]
            -1.79566394e-10 * tc[3]
            +2.00255376e-14 * tc[4];
        /*species 3: O */
        species[3] =
            +2.56942078e+00
            -8.59741137e-05 * tc[1]
            +4.19484589e-08 * tc[2]
            -1.00177799e-11 * tc[3]
            +1.22833691e-15 * tc[4];
        /*species 4: OH */
        species[4] =
            +2.86472886e+00
            +1.05650448e-03 * tc[1]
            -2.59082758e-07 * tc[2]
            +3.05218674e-11 * tc[3]
            -1.33195876e-15 * tc[4];
        /*species 5: O2 */
        species[5] =
            +3.28253784e+00
            +1.48308754e-03 * tc[1]
            -7.57966669e-07 * tc[2]
            +2.09470555e-10 * tc[3]
            -2.16717794e-14 * tc[4];
        /*species 6: H2O2 */
        species[6] =
            +4.16500285e+00
            +4.90831694e-03 * tc[1]
            -1.90139225e-06 * tc[2]
            +3.71185986e-10 * tc[3]
            -2.87908305e-14 * tc[4];
        /*species 7: H2O */
        species[7] =
            +3.03399249e+00
            +2.17691804e-03 * tc[1]
            -1.64072518e-07 * tc[2]
            -9.70419870e-11 * tc[3]
            +1.68200992e-14 * tc[4];
        /*species 8: HO2 */
        species[8] =
            +4.01721090e+00
            +2.23982013e-03 * tc[1]
            -6.33658150e-07 * tc[2]
            +1.14246370e-10 * tc[3]
            -1.07908535e-14 * tc[4];
        /*species 9: CO */
        species[9] =
            +2.71518561e+00
            +2.06252743e-03 * tc[1]
            -9.98825771e-07 * tc[2]
            +2.30053008e-10 * tc[3]
            -2.03647716e-14 * tc[4];
        /*species 10: CH3 */
        species[10] =
            +2.97812060e+00
            +5.79785200e-03 * tc[1]
            -1.97558000e-06 * tc[2]
            +3.07297900e-10 * tc[3]
            -1.79174160e-14 * tc[4];
        /*species 11: CH2O */
        species[11] =
            +1.76069008e+00
            +9.20000082e-03 * tc[1]
            -4.42258813e-06 * tc[2]
            +1.00641212e-09 * tc[3]
            -8.83855640e-14 * tc[4];
        /*species 12: CO2 */
        species[12] =
            +3.85746029e+00
            +4.41437026e-03 * tc[1]
            -2.21481404e-06 * tc[2]
            +5.23490188e-10 * tc[3]
            -4.72084164e-14 * tc[4];
        /*species 13: CH4 */
        species[13] =
            +1.65326226e+00
            +1.00263099e-02 * tc[1]
            -3.31661238e-06 * tc[2]
            +5.36483138e-10 * tc[3]
            -3.14696758e-14 * tc[4];
        /*species 14: C2H2 */
        species[14] =
            +4.14756964e+00
            +5.96166664e-03 * tc[1]
            -2.37294852e-06 * tc[2]
            +4.67412171e-10 * tc[3]
            -3.61235213e-14 * tc[4];
        /*species 15: C2H4 */
        species[15] =
            +2.03611116e+00
            +1.46454151e-02 * tc[1]
            -6.71077915e-06 * tc[2]
            +1.47222923e-09 * tc[3]
            -1.25706061e-13 * tc[4];
        /*species 16: CH2CO */
        species[16] =
            +4.51129732e+00
            +9.00359745e-03 * tc[1]
            -4.16939635e-06 * tc[2]
            +9.23345882e-10 * tc[3]
            -7.94838201e-14 * tc[4];
        /*species 17: C2H6 */
        species[17] =
            +1.07188150e+00
            +2.16852677e-02 * tc[1]
            -1.00256067e-05 * tc[2]
            +2.21412001e-09 * tc[3]
            -1.90002890e-13 * tc[4];
        /*species 18: C2H3 */
        species[18] =
            +3.01672400e+00
            +1.03302292e-02 * tc[1]
            -4.68082349e-06 * tc[2]
            +1.01763288e-09 * tc[3]
            -8.62607041e-14 * tc[4];
        /*species 19: C2H5 */
        species[19] =
            +1.95465642e+00
            +1.73972722e-02 * tc[1]
            -7.98206668e-06 * tc[2]
            +1.75217689e-09 * tc[3]
            -1.49641576e-13 * tc[4];
        /*species 20: HCCO */
        species[20] =
            +5.62820580e+00
            +4.08534010e-03 * tc[1]
            -1.59345470e-06 * tc[2]
            +2.86260520e-10 * tc[3]
            -1.94078320e-14 * tc[4];
        /*species 21: CH3CHO */
        species[21] =
            +2.68543112e+00
            +1.76802373e-02 * tc[1]
            -8.65402739e-06 * tc[2]
            +2.03680589e-09 * tc[3]
            -1.87630935e-13 * tc[4];
        /*species 22: CH2CHO */
        species[22] =
            +2.42606357e+00
            +1.72400021e-02 * tc[1]
            -9.77132119e-06 * tc[2]
            +2.66555672e-09 * tc[3]
            -2.82120078e-13 * tc[4];
        /*species 23: C2H5O */
        species[23] =
            +2.46262349e+00
            +2.09503959e-02 * tc[1]
            -9.39291750e-06 * tc[2]
            +1.56440627e-09 * tc[3]
            +0.00000000e+00 * tc[4];
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
            +7.04120200e-04 * tc[1]
            -1.32107400e-06 * tc[2]
            +1.41037875e-09 * tc[3]
            -4.88970800e-13 * tc[4]
            -1.02089990e+03 * invT;
        /*species 1: H */
        species[1] =
            +1.50000000e+00
            +3.52666409e-13 * tc[1]
            -6.65306547e-16 * tc[2]
            +5.75204080e-19 * tc[3]
            -1.85546466e-22 * tc[4]
            +2.54736599e+04 * invT;
        /*species 2: H2 */
        species[2] =
            +1.34433112e+00
            +3.99026037e-03 * tc[1]
            -6.49271700e-06 * tc[2]
            +5.03930235e-09 * tc[3]
            -1.47522352e-12 * tc[4]
            -9.17935173e+02 * invT;
        /*species 3: O */
        species[3] =
            +2.16826710e+00
            -1.63965942e-03 * tc[1]
            +2.21435465e-06 * tc[2]
            -1.53201656e-09 * tc[3]
            +4.22531942e-13 * tc[4]
            +2.91222592e+04 * invT;
        /*species 4: OH */
        species[4] =
            +3.12530561e+00
            -1.61272470e-03 * tc[1]
            +2.17588230e-06 * tc[2]
            -1.44963411e-09 * tc[3]
            +4.12474758e-13 * tc[4]
            +3.38153812e+03 * invT;
        /*species 5: O2 */
        species[5] =
            +2.78245636e+00
            -1.49836708e-03 * tc[1]
            +3.28243400e-06 * tc[2]
            -2.42032377e-09 * tc[3]
            +6.48745674e-13 * tc[4]
            -1.06394356e+03 * invT;
        /*species 6: H2O2 */
        species[6] =
            +3.27611269e+00
            -2.71411208e-04 * tc[1]
            +5.57785670e-06 * tc[2]
            -5.39427032e-09 * tc[3]
            +1.72490873e-12 * tc[4]
            -1.77025821e+04 * invT;
        /*species 7: H2O */
        species[7] =
            +3.19864056e+00
            -1.01821705e-03 * tc[1]
            +2.17346737e-06 * tc[2]
            -1.37199266e-09 * tc[3]
            +3.54395634e-13 * tc[4]
            -3.02937267e+04 * invT;
        /*species 8: HO2 */
        species[8] =
            +3.30179801e+00
            -2.37456025e-03 * tc[1]
            +7.05276303e-06 * tc[2]
            -6.06909735e-09 * tc[3]
            +1.85845025e-12 * tc[4]
            +2.94808040e+02 * invT;
        /*species 9: CO */
        species[9] =
            +2.57953347e+00
            -3.05176840e-04 * tc[1]
            +3.38938110e-07 * tc[2]
            +2.26751471e-10 * tc[3]
            -1.80884900e-13 * tc[4]
            -1.43440860e+04 * invT;
        /*species 10: CH3 */
        species[10] =
            +2.65717970e+00
            +1.06329895e-03 * tc[1]
            +1.81946277e-06 * tc[2]
            -1.65452507e-09 * tc[3]
            +4.93141480e-13 * tc[4]
            +1.64227160e+04 * invT;
        /*species 11: CH2O */
        species[11] =
            +3.79372315e+00
            -4.95416684e-03 * tc[1]
            +1.24406669e-05 * tc[2]
            -9.48213152e-09 * tc[3]
            +2.63545304e-12 * tc[4]
            -1.43089567e+04 * invT;
        /*species 12: CO2 */
        species[12] =
            +1.35677352e+00
            +4.49229839e-03 * tc[1]
            -2.37452090e-06 * tc[2]
            +6.14797555e-10 * tc[3]
            -2.87399096e-14 * tc[4]
            -4.83719697e+04 * invT;
        /*species 13: CH4 */
        species[13] =
            +4.14911468e+00
            -6.83110045e-03 * tc[1]
            +1.63817974e-05 * tc[2]
            -1.21061692e-08 * tc[3]
            +3.33206882e-12 * tc[4]
            -1.02465983e+04 * invT;
        /*species 14: C2H2 */
        species[14] =
            -1.91318906e-01
            +1.16807815e-02 * tc[1]
            -1.18390605e-05 * tc[2]
            +7.00381092e-09 * tc[3]
            -1.70014595e-12 * tc[4]
            +2.64289807e+04 * invT;
        /*species 15: C2H4 */
        species[15] =
            +2.95920148e+00
            -3.78526124e-03 * tc[1]
            +1.90330097e-05 * tc[2]
            -1.72897188e-08 * tc[3]
            +5.39768746e-12 * tc[4]
            +5.08977593e+03 * invT;
        /*species 16: CH2CO */
        species[16] =
            +1.13583630e+00
            +9.05943605e-03 * tc[1]
            -5.79824913e-06 * tc[2]
            +2.33599392e-09 * tc[3]
            -4.02915230e-13 * tc[4]
            -7.04291804e+03 * invT;
        /*species 17: C2H6 */
        species[17] =
            +3.29142492e+00
            -2.75077135e-03 * tc[1]
            +1.99812763e-05 * tc[2]
            -1.77116571e-08 * tc[3]
            +5.37371542e-12 * tc[4]
            -1.15222055e+04 * invT;
        /*species 18: C2H3 */
        species[18] =
            +2.21246645e+00
            +7.57395810e-04 * tc[1]
            +8.64031373e-06 * tc[2]
            -8.94144617e-09 * tc[3]
            +2.94301746e-12 * tc[4]
            +3.48598468e+04 * invT;
        /*species 19: C2H5 */
        species[19] =
            +3.30646568e+00
            -2.09329446e-03 * tc[1]
            +1.65714269e-05 * tc[2]
            -1.49781651e-08 * tc[3]
            +4.61018008e-12 * tc[4]
            +1.28416265e+04 * invT;
        /*species 20: HCCO */
        species[20] =
            +1.25172140e+00
            +8.82751050e-03 * tc[1]
            -7.90970033e-06 * tc[2]
            +4.31893975e-09 * tc[3]
            -1.01329622e-12 * tc[4]
            +2.00594490e+04 * invT;
        /*species 21: CH3CHO */
        species[21] =
            +4.06538560e-01
            +1.08492219e-02 * tc[1]
            -4.91910883e-06 * tc[2]
            +1.82608869e-09 * tc[3]
            -4.18238934e-13 * tc[4]
            -2.17973223e+04 * invT;
        /*species 22: CH2CHO */
        species[22] =
            +9.68573300e-02
            +1.10114398e-02 * tc[1]
            -4.81944813e-06 * tc[2]
            +7.51948945e-10 * tc[3]
            +1.21798575e-13 * tc[4]
            +1.06943322e+03 * invT;
        /*species 23: C2H5O */
        species[23] =
            -5.05579292e-01
            +1.35887217e-02 * tc[1]
            -5.53030033e-06 * tc[2]
            +1.28801050e-09 * tc[3]
            -1.29699383e-13 * tc[4]
            -3.35252925e+03 * invT;
    } else {
        /*species 0: N2 */
        species[0] =
            +1.92664000e+00
            +7.43988400e-04 * tc[1]
            -1.89492000e-07 * tc[2]
            +2.52425950e-11 * tc[3]
            -1.35067020e-15 * tc[4]
            -9.22797700e+02 * invT;
        /*species 1: H */
        species[1] =
            +1.50000001e+00
            -1.15421486e-11 * tc[1]
            +5.38539827e-15 * tc[2]
            -1.18378809e-18 * tc[3]
            +9.96394714e-23 * tc[4]
            +2.54736599e+04 * invT;
        /*species 2: H2 */
        species[2] =
            +2.33727920e+00
            -2.47012365e-05 * tc[1]
            +1.66485593e-07 * tc[2]
            -4.48915985e-11 * tc[3]
            +4.00510752e-15 * tc[4]
            -9.50158922e+02 * invT;
        /*species 3: O */
        species[3] =
            +1.56942078e+00
            -4.29870569e-05 * tc[1]
            +1.39828196e-08 * tc[2]
            -2.50444497e-12 * tc[3]
            +2.45667382e-16 * tc[4]
            +2.92175791e+04 * invT;
        /*species 4: OH */
        species[4] =
            +1.86472886e+00
            +5.28252240e-04 * tc[1]
            -8.63609193e-08 * tc[2]
            +7.63046685e-12 * tc[3]
            -2.66391752e-16 * tc[4]
            +3.71885774e+03 * invT;
        /*species 5: O2 */
        species[5] =
            +2.28253784e+00
            +7.41543770e-04 * tc[1]
            -2.52655556e-07 * tc[2]
            +5.23676387e-11 * tc[3]
            -4.33435588e-15 * tc[4]
            -1.08845772e+03 * invT;
        /*species 6: H2O2 */
        species[6] =
            +3.16500285e+00
            +2.45415847e-03 * tc[1]
            -6.33797417e-07 * tc[2]
            +9.27964965e-11 * tc[3]
            -5.75816610e-15 * tc[4]
            -1.78617877e+04 * invT;
        /*species 7: H2O */
        species[7] =
            +2.03399249e+00
            +1.08845902e-03 * tc[1]
            -5.46908393e-08 * tc[2]
            -2.42604967e-11 * tc[3]
            +3.36401984e-15 * tc[4]
            -3.00042971e+04 * invT;
        /*species 8: HO2 */
        species[8] =
            +3.01721090e+00
            +1.11991006e-03 * tc[1]
            -2.11219383e-07 * tc[2]
            +2.85615925e-11 * tc[3]
            -2.15817070e-15 * tc[4]
            +1.11856713e+02 * invT;
        /*species 9: CO */
        species[9] =
            +1.71518561e+00
            +1.03126372e-03 * tc[1]
            -3.32941924e-07 * tc[2]
            +5.75132520e-11 * tc[3]
            -4.07295432e-15 * tc[4]
            -1.41518724e+04 * invT;
        /*species 10: CH3 */
        species[10] =
            +1.97812060e+00
            +2.89892600e-03 * tc[1]
            -6.58526667e-07 * tc[2]
            +7.68244750e-11 * tc[3]
            -3.58348320e-15 * tc[4]
            +1.65095130e+04 * invT;
        /*species 11: CH2O */
        species[11] =
            +7.60690080e-01
            +4.60000041e-03 * tc[1]
            -1.47419604e-06 * tc[2]
            +2.51603030e-10 * tc[3]
            -1.76771128e-14 * tc[4]
            -1.39958323e+04 * invT;
        /*species 12: CO2 */
        species[12] =
            +2.85746029e+00
            +2.20718513e-03 * tc[1]
            -7.38271347e-07 * tc[2]
            +1.30872547e-10 * tc[3]
            -9.44168328e-15 * tc[4]
            -4.87591660e+04 * invT;
        /*species 13: CH4 */
        species[13] =
            +6.53262260e-01
            +5.01315495e-03 * tc[1]
            -1.10553746e-06 * tc[2]
            +1.34120785e-10 * tc[3]
            -6.29393516e-15 * tc[4]
            -1.00095936e+04 * invT;
        /*species 14: C2H2 */
        species[14] =
            +3.14756964e+00
            +2.98083332e-03 * tc[1]
            -7.90982840e-07 * tc[2]
            +1.16853043e-10 * tc[3]
            -7.22470426e-15 * tc[4]
            +2.59359992e+04 * invT;
        /*species 15: C2H4 */
        species[15] =
            +1.03611116e+00
            +7.32270755e-03 * tc[1]
            -2.23692638e-06 * tc[2]
            +3.68057308e-10 * tc[3]
            -2.51412122e-14 * tc[4]
            +4.93988614e+03 * invT;
        /*species 16: CH2CO */
        species[16] =
            +3.51129732e+00
            +4.50179872e-03 * tc[1]
            -1.38979878e-06 * tc[2]
            +2.30836470e-10 * tc[3]
            -1.58967640e-14 * tc[4]
            -7.55105311e+03 * invT;
        /*species 17: C2H6 */
        species[17] =
            +7.18815000e-02
            +1.08426339e-02 * tc[1]
            -3.34186890e-06 * tc[2]
            +5.53530003e-10 * tc[3]
            -3.80005780e-14 * tc[4]
            -1.14263932e+04 * invT;
        /*species 18: C2H3 */
        species[18] =
            +2.01672400e+00
            +5.16511460e-03 * tc[1]
            -1.56027450e-06 * tc[2]
            +2.54408220e-10 * tc[3]
            -1.72521408e-14 * tc[4]
            +3.46128739e+04 * invT;
        /*species 19: C2H5 */
        species[19] =
            +9.54656420e-01
            +8.69863610e-03 * tc[1]
            -2.66068889e-06 * tc[2]
            +4.38044223e-10 * tc[3]
            -2.99283152e-14 * tc[4]
            +1.28575200e+04 * invT;
        /*species 20: HCCO */
        species[20] =
            +4.62820580e+00
            +2.04267005e-03 * tc[1]
            -5.31151567e-07 * tc[2]
            +7.15651300e-11 * tc[3]
            -3.88156640e-15 * tc[4]
            +1.93272150e+04 * invT;
        /*species 21: CH3CHO */
        species[21] =
            +1.68543112e+00
            +8.84011865e-03 * tc[1]
            -2.88467580e-06 * tc[2]
            +5.09201472e-10 * tc[3]
            -3.75261870e-14 * tc[4]
            -2.21653701e+04 * invT;
        /*species 22: CH2CHO */
        species[22] =
            +1.42606357e+00
            +8.62000105e-03 * tc[1]
            -3.25710706e-06 * tc[2]
            +6.66389180e-10 * tc[3]
            -5.64240156e-14 * tc[4]
            +8.33106990e+02 * invT;
        /*species 23: C2H5O */
        species[23] =
            +1.46262349e+00
            +1.04751979e-02 * tc[1]
            -3.13097250e-06 * tc[2]
            +3.91101567e-10 * tc[3]
            +0.00000000e+00 * tc[4]
            -3.83932658e+03 * invT;
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
            +7.04120200e-04 * tc[1]
            -1.32107400e-06 * tc[2]
            +1.41037875e-09 * tc[3]
            -4.88970800e-13 * tc[4]
            -1.02089990e+03 * invT;
        /*species 1: H */
        species[1] =
            +2.50000000e+00
            +3.52666409e-13 * tc[1]
            -6.65306547e-16 * tc[2]
            +5.75204080e-19 * tc[3]
            -1.85546466e-22 * tc[4]
            +2.54736599e+04 * invT;
        /*species 2: H2 */
        species[2] =
            +2.34433112e+00
            +3.99026037e-03 * tc[1]
            -6.49271700e-06 * tc[2]
            +5.03930235e-09 * tc[3]
            -1.47522352e-12 * tc[4]
            -9.17935173e+02 * invT;
        /*species 3: O */
        species[3] =
            +3.16826710e+00
            -1.63965942e-03 * tc[1]
            +2.21435465e-06 * tc[2]
            -1.53201656e-09 * tc[3]
            +4.22531942e-13 * tc[4]
            +2.91222592e+04 * invT;
        /*species 4: OH */
        species[4] =
            +4.12530561e+00
            -1.61272470e-03 * tc[1]
            +2.17588230e-06 * tc[2]
            -1.44963411e-09 * tc[3]
            +4.12474758e-13 * tc[4]
            +3.38153812e+03 * invT;
        /*species 5: O2 */
        species[5] =
            +3.78245636e+00
            -1.49836708e-03 * tc[1]
            +3.28243400e-06 * tc[2]
            -2.42032377e-09 * tc[3]
            +6.48745674e-13 * tc[4]
            -1.06394356e+03 * invT;
        /*species 6: H2O2 */
        species[6] =
            +4.27611269e+00
            -2.71411208e-04 * tc[1]
            +5.57785670e-06 * tc[2]
            -5.39427032e-09 * tc[3]
            +1.72490873e-12 * tc[4]
            -1.77025821e+04 * invT;
        /*species 7: H2O */
        species[7] =
            +4.19864056e+00
            -1.01821705e-03 * tc[1]
            +2.17346737e-06 * tc[2]
            -1.37199266e-09 * tc[3]
            +3.54395634e-13 * tc[4]
            -3.02937267e+04 * invT;
        /*species 8: HO2 */
        species[8] =
            +4.30179801e+00
            -2.37456025e-03 * tc[1]
            +7.05276303e-06 * tc[2]
            -6.06909735e-09 * tc[3]
            +1.85845025e-12 * tc[4]
            +2.94808040e+02 * invT;
        /*species 9: CO */
        species[9] =
            +3.57953347e+00
            -3.05176840e-04 * tc[1]
            +3.38938110e-07 * tc[2]
            +2.26751471e-10 * tc[3]
            -1.80884900e-13 * tc[4]
            -1.43440860e+04 * invT;
        /*species 10: CH3 */
        species[10] =
            +3.65717970e+00
            +1.06329895e-03 * tc[1]
            +1.81946277e-06 * tc[2]
            -1.65452507e-09 * tc[3]
            +4.93141480e-13 * tc[4]
            +1.64227160e+04 * invT;
        /*species 11: CH2O */
        species[11] =
            +4.79372315e+00
            -4.95416684e-03 * tc[1]
            +1.24406669e-05 * tc[2]
            -9.48213152e-09 * tc[3]
            +2.63545304e-12 * tc[4]
            -1.43089567e+04 * invT;
        /*species 12: CO2 */
        species[12] =
            +2.35677352e+00
            +4.49229839e-03 * tc[1]
            -2.37452090e-06 * tc[2]
            +6.14797555e-10 * tc[3]
            -2.87399096e-14 * tc[4]
            -4.83719697e+04 * invT;
        /*species 13: CH4 */
        species[13] =
            +5.14911468e+00
            -6.83110045e-03 * tc[1]
            +1.63817974e-05 * tc[2]
            -1.21061692e-08 * tc[3]
            +3.33206882e-12 * tc[4]
            -1.02465983e+04 * invT;
        /*species 14: C2H2 */
        species[14] =
            +8.08681094e-01
            +1.16807815e-02 * tc[1]
            -1.18390605e-05 * tc[2]
            +7.00381092e-09 * tc[3]
            -1.70014595e-12 * tc[4]
            +2.64289807e+04 * invT;
        /*species 15: C2H4 */
        species[15] =
            +3.95920148e+00
            -3.78526124e-03 * tc[1]
            +1.90330097e-05 * tc[2]
            -1.72897188e-08 * tc[3]
            +5.39768746e-12 * tc[4]
            +5.08977593e+03 * invT;
        /*species 16: CH2CO */
        species[16] =
            +2.13583630e+00
            +9.05943605e-03 * tc[1]
            -5.79824913e-06 * tc[2]
            +2.33599392e-09 * tc[3]
            -4.02915230e-13 * tc[4]
            -7.04291804e+03 * invT;
        /*species 17: C2H6 */
        species[17] =
            +4.29142492e+00
            -2.75077135e-03 * tc[1]
            +1.99812763e-05 * tc[2]
            -1.77116571e-08 * tc[3]
            +5.37371542e-12 * tc[4]
            -1.15222055e+04 * invT;
        /*species 18: C2H3 */
        species[18] =
            +3.21246645e+00
            +7.57395810e-04 * tc[1]
            +8.64031373e-06 * tc[2]
            -8.94144617e-09 * tc[3]
            +2.94301746e-12 * tc[4]
            +3.48598468e+04 * invT;
        /*species 19: C2H5 */
        species[19] =
            +4.30646568e+00
            -2.09329446e-03 * tc[1]
            +1.65714269e-05 * tc[2]
            -1.49781651e-08 * tc[3]
            +4.61018008e-12 * tc[4]
            +1.28416265e+04 * invT;
        /*species 20: HCCO */
        species[20] =
            +2.25172140e+00
            +8.82751050e-03 * tc[1]
            -7.90970033e-06 * tc[2]
            +4.31893975e-09 * tc[3]
            -1.01329622e-12 * tc[4]
            +2.00594490e+04 * invT;
        /*species 21: CH3CHO */
        species[21] =
            +1.40653856e+00
            +1.08492219e-02 * tc[1]
            -4.91910883e-06 * tc[2]
            +1.82608869e-09 * tc[3]
            -4.18238934e-13 * tc[4]
            -2.17973223e+04 * invT;
        /*species 22: CH2CHO */
        species[22] =
            +1.09685733e+00
            +1.10114398e-02 * tc[1]
            -4.81944813e-06 * tc[2]
            +7.51948945e-10 * tc[3]
            +1.21798575e-13 * tc[4]
            +1.06943322e+03 * invT;
        /*species 23: C2H5O */
        species[23] =
            +4.94420708e-01
            +1.35887217e-02 * tc[1]
            -5.53030033e-06 * tc[2]
            +1.28801050e-09 * tc[3]
            -1.29699383e-13 * tc[4]
            -3.35252925e+03 * invT;
    } else {
        /*species 0: N2 */
        species[0] =
            +2.92664000e+00
            +7.43988400e-04 * tc[1]
            -1.89492000e-07 * tc[2]
            +2.52425950e-11 * tc[3]
            -1.35067020e-15 * tc[4]
            -9.22797700e+02 * invT;
        /*species 1: H */
        species[1] =
            +2.50000001e+00
            -1.15421486e-11 * tc[1]
            +5.38539827e-15 * tc[2]
            -1.18378809e-18 * tc[3]
            +9.96394714e-23 * tc[4]
            +2.54736599e+04 * invT;
        /*species 2: H2 */
        species[2] =
            +3.33727920e+00
            -2.47012365e-05 * tc[1]
            +1.66485593e-07 * tc[2]
            -4.48915985e-11 * tc[3]
            +4.00510752e-15 * tc[4]
            -9.50158922e+02 * invT;
        /*species 3: O */
        species[3] =
            +2.56942078e+00
            -4.29870569e-05 * tc[1]
            +1.39828196e-08 * tc[2]
            -2.50444497e-12 * tc[3]
            +2.45667382e-16 * tc[4]
            +2.92175791e+04 * invT;
        /*species 4: OH */
        species[4] =
            +2.86472886e+00
            +5.28252240e-04 * tc[1]
            -8.63609193e-08 * tc[2]
            +7.63046685e-12 * tc[3]
            -2.66391752e-16 * tc[4]
            +3.71885774e+03 * invT;
        /*species 5: O2 */
        species[5] =
            +3.28253784e+00
            +7.41543770e-04 * tc[1]
            -2.52655556e-07 * tc[2]
            +5.23676387e-11 * tc[3]
            -4.33435588e-15 * tc[4]
            -1.08845772e+03 * invT;
        /*species 6: H2O2 */
        species[6] =
            +4.16500285e+00
            +2.45415847e-03 * tc[1]
            -6.33797417e-07 * tc[2]
            +9.27964965e-11 * tc[3]
            -5.75816610e-15 * tc[4]
            -1.78617877e+04 * invT;
        /*species 7: H2O */
        species[7] =
            +3.03399249e+00
            +1.08845902e-03 * tc[1]
            -5.46908393e-08 * tc[2]
            -2.42604967e-11 * tc[3]
            +3.36401984e-15 * tc[4]
            -3.00042971e+04 * invT;
        /*species 8: HO2 */
        species[8] =
            +4.01721090e+00
            +1.11991006e-03 * tc[1]
            -2.11219383e-07 * tc[2]
            +2.85615925e-11 * tc[3]
            -2.15817070e-15 * tc[4]
            +1.11856713e+02 * invT;
        /*species 9: CO */
        species[9] =
            +2.71518561e+00
            +1.03126372e-03 * tc[1]
            -3.32941924e-07 * tc[2]
            +5.75132520e-11 * tc[3]
            -4.07295432e-15 * tc[4]
            -1.41518724e+04 * invT;
        /*species 10: CH3 */
        species[10] =
            +2.97812060e+00
            +2.89892600e-03 * tc[1]
            -6.58526667e-07 * tc[2]
            +7.68244750e-11 * tc[3]
            -3.58348320e-15 * tc[4]
            +1.65095130e+04 * invT;
        /*species 11: CH2O */
        species[11] =
            +1.76069008e+00
            +4.60000041e-03 * tc[1]
            -1.47419604e-06 * tc[2]
            +2.51603030e-10 * tc[3]
            -1.76771128e-14 * tc[4]
            -1.39958323e+04 * invT;
        /*species 12: CO2 */
        species[12] =
            +3.85746029e+00
            +2.20718513e-03 * tc[1]
            -7.38271347e-07 * tc[2]
            +1.30872547e-10 * tc[3]
            -9.44168328e-15 * tc[4]
            -4.87591660e+04 * invT;
        /*species 13: CH4 */
        species[13] =
            +1.65326226e+00
            +5.01315495e-03 * tc[1]
            -1.10553746e-06 * tc[2]
            +1.34120785e-10 * tc[3]
            -6.29393516e-15 * tc[4]
            -1.00095936e+04 * invT;
        /*species 14: C2H2 */
        species[14] =
            +4.14756964e+00
            +2.98083332e-03 * tc[1]
            -7.90982840e-07 * tc[2]
            +1.16853043e-10 * tc[3]
            -7.22470426e-15 * tc[4]
            +2.59359992e+04 * invT;
        /*species 15: C2H4 */
        species[15] =
            +2.03611116e+00
            +7.32270755e-03 * tc[1]
            -2.23692638e-06 * tc[2]
            +3.68057308e-10 * tc[3]
            -2.51412122e-14 * tc[4]
            +4.93988614e+03 * invT;
        /*species 16: CH2CO */
        species[16] =
            +4.51129732e+00
            +4.50179872e-03 * tc[1]
            -1.38979878e-06 * tc[2]
            +2.30836470e-10 * tc[3]
            -1.58967640e-14 * tc[4]
            -7.55105311e+03 * invT;
        /*species 17: C2H6 */
        species[17] =
            +1.07188150e+00
            +1.08426339e-02 * tc[1]
            -3.34186890e-06 * tc[2]
            +5.53530003e-10 * tc[3]
            -3.80005780e-14 * tc[4]
            -1.14263932e+04 * invT;
        /*species 18: C2H3 */
        species[18] =
            +3.01672400e+00
            +5.16511460e-03 * tc[1]
            -1.56027450e-06 * tc[2]
            +2.54408220e-10 * tc[3]
            -1.72521408e-14 * tc[4]
            +3.46128739e+04 * invT;
        /*species 19: C2H5 */
        species[19] =
            +1.95465642e+00
            +8.69863610e-03 * tc[1]
            -2.66068889e-06 * tc[2]
            +4.38044223e-10 * tc[3]
            -2.99283152e-14 * tc[4]
            +1.28575200e+04 * invT;
        /*species 20: HCCO */
        species[20] =
            +5.62820580e+00
            +2.04267005e-03 * tc[1]
            -5.31151567e-07 * tc[2]
            +7.15651300e-11 * tc[3]
            -3.88156640e-15 * tc[4]
            +1.93272150e+04 * invT;
        /*species 21: CH3CHO */
        species[21] =
            +2.68543112e+00
            +8.84011865e-03 * tc[1]
            -2.88467580e-06 * tc[2]
            +5.09201472e-10 * tc[3]
            -3.75261870e-14 * tc[4]
            -2.21653701e+04 * invT;
        /*species 22: CH2CHO */
        species[22] =
            +2.42606357e+00
            +8.62000105e-03 * tc[1]
            -3.25710706e-06 * tc[2]
            +6.66389180e-10 * tc[3]
            -5.64240156e-14 * tc[4]
            +8.33106990e+02 * invT;
        /*species 23: C2H5O */
        species[23] =
            +2.46262349e+00
            +1.04751979e-02 * tc[1]
            -3.13097250e-06 * tc[2]
            +3.91101567e-10 * tc[3]
            +0.00000000e+00 * tc[4]
            -3.83932658e+03 * invT;
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
            +1.40824040e-03 * tc[1]
            -1.98161100e-06 * tc[2]
            +1.88050500e-09 * tc[3]
            -6.11213500e-13 * tc[4]
            +3.95037200e+00 ;
        /*species 1: H */
        species[1] =
            +2.50000000e+00 * tc[0]
            +7.05332819e-13 * tc[1]
            -9.97959820e-16 * tc[2]
            +7.66938773e-19 * tc[3]
            -2.31933083e-22 * tc[4]
            -4.46682853e-01 ;
        /*species 2: H2 */
        species[2] =
            +2.34433112e+00 * tc[0]
            +7.98052075e-03 * tc[1]
            -9.73907550e-06 * tc[2]
            +6.71906980e-09 * tc[3]
            -1.84402940e-12 * tc[4]
            +6.83010238e-01 ;
        /*species 3: O */
        species[3] =
            +3.16826710e+00 * tc[0]
            -3.27931884e-03 * tc[1]
            +3.32153198e-06 * tc[2]
            -2.04268875e-09 * tc[3]
            +5.28164927e-13 * tc[4]
            +2.05193346e+00 ;
        /*species 4: OH */
        species[4] =
            +4.12530561e+00 * tc[0]
            -3.22544939e-03 * tc[1]
            +3.26382346e-06 * tc[2]
            -1.93284548e-09 * tc[3]
            +5.15593447e-13 * tc[4]
            -6.90432960e-01 ;
        /*species 5: O2 */
        species[5] =
            +3.78245636e+00 * tc[0]
            -2.99673416e-03 * tc[1]
            +4.92365101e-06 * tc[2]
            -3.22709836e-09 * tc[3]
            +8.10932092e-13 * tc[4]
            +3.65767573e+00 ;
        /*species 6: H2O2 */
        species[6] =
            +4.27611269e+00 * tc[0]
            -5.42822417e-04 * tc[1]
            +8.36678505e-06 * tc[2]
            -7.19236043e-09 * tc[3]
            +2.15613591e-12 * tc[4]
            +3.43505074e+00 ;
        /*species 7: H2O */
        species[7] =
            +4.19864056e+00 * tc[0]
            -2.03643410e-03 * tc[1]
            +3.26020105e-06 * tc[2]
            -1.82932354e-09 * tc[3]
            +4.42994543e-13 * tc[4]
            -8.49032208e-01 ;
        /*species 8: HO2 */
        species[8] =
            +4.30179801e+00 * tc[0]
            -4.74912051e-03 * tc[1]
            +1.05791445e-05 * tc[2]
            -8.09212980e-09 * tc[3]
            +2.32306281e-12 * tc[4]
            +3.71666245e+00 ;
        /*species 9: CO */
        species[9] =
            +3.57953347e+00 * tc[0]
            -6.10353680e-04 * tc[1]
            +5.08407165e-07 * tc[2]
            +3.02335295e-10 * tc[3]
            -2.26106125e-13 * tc[4]
            +3.50840928e+00 ;
        /*species 10: CH3 */
        species[10] =
            +3.65717970e+00 * tc[0]
            +2.12659790e-03 * tc[1]
            +2.72919415e-06 * tc[2]
            -2.20603343e-09 * tc[3]
            +6.16426850e-13 * tc[4]
            +1.67353540e+00 ;
        /*species 11: CH2O */
        species[11] =
            +4.79372315e+00 * tc[0]
            -9.90833369e-03 * tc[1]
            +1.86610004e-05 * tc[2]
            -1.26428420e-08 * tc[3]
            +3.29431630e-12 * tc[4]
            +6.02812900e-01 ;
        /*species 12: CO2 */
        species[12] =
            +2.35677352e+00 * tc[0]
            +8.98459677e-03 * tc[1]
            -3.56178134e-06 * tc[2]
            +8.19730073e-10 * tc[3]
            -3.59248870e-14 * tc[4]
            +9.90105222e+00 ;
        /*species 13: CH4 */
        species[13] =
            +5.14911468e+00 * tc[0]
            -1.36622009e-02 * tc[1]
            +2.45726961e-05 * tc[2]
            -1.61415589e-08 * tc[3]
            +4.16508602e-12 * tc[4]
            -4.63848842e+00 ;
        /*species 14: C2H2 */
        species[14] =
            +8.08681094e-01 * tc[0]
            +2.33615629e-02 * tc[1]
            -1.77585907e-05 * tc[2]
            +9.33841457e-09 * tc[3]
            -2.12518243e-12 * tc[4]
            +1.39397051e+01 ;
        /*species 15: C2H4 */
        species[15] =
            +3.95920148e+00 * tc[0]
            -7.57052247e-03 * tc[1]
            +2.85495146e-05 * tc[2]
            -2.30529584e-08 * tc[3]
            +6.74710933e-12 * tc[4]
            +4.09733096e+00 ;
        /*species 16: CH2CO */
        species[16] =
            +2.13583630e+00 * tc[0]
            +1.81188721e-02 * tc[1]
            -8.69737370e-06 * tc[2]
            +3.11465856e-09 * tc[3]
            -5.03644037e-13 * tc[4]
            +1.22156480e+01 ;
        /*species 17: C2H6 */
        species[17] =
            +4.29142492e+00 * tc[0]
            -5.50154270e-03 * tc[1]
            +2.99719144e-05 * tc[2]
            -2.36155428e-08 * tc[3]
            +6.71714427e-12 * tc[4]
            +2.66682316e+00 ;
        /*species 18: C2H3 */
        species[18] =
            +3.21246645e+00 * tc[0]
            +1.51479162e-03 * tc[1]
            +1.29604706e-05 * tc[2]
            -1.19219282e-08 * tc[3]
            +3.67877182e-12 * tc[4]
            +8.51054025e+00 ;
        /*species 19: C2H5 */
        species[19] =
            +4.30646568e+00 * tc[0]
            -4.18658892e-03 * tc[1]
            +2.48571403e-05 * tc[2]
            -1.99708869e-08 * tc[3]
            +5.76272510e-12 * tc[4]
            +4.70720924e+00 ;
        /*species 20: HCCO */
        species[20] =
            +2.25172140e+00 * tc[0]
            +1.76550210e-02 * tc[1]
            -1.18645505e-05 * tc[2]
            +5.75858633e-09 * tc[3]
            -1.26662028e-12 * tc[4]
            +1.24904170e+01 ;
        /*species 21: CH3CHO */
        species[21] =
            +1.40653856e+00 * tc[0]
            +2.16984438e-02 * tc[1]
            -7.37866325e-06 * tc[2]
            +2.43478493e-09 * tc[3]
            -5.22798668e-13 * tc[4]
            +1.77513265e+01 ;
        /*species 22: CH2CHO */
        species[22] =
            +1.09685733e+00 * tc[0]
            +2.20228796e-02 * tc[1]
            -7.22917220e-06 * tc[2]
            +1.00259859e-09 * tc[3]
            +1.52248219e-13 * tc[4]
            +1.90094813e+01 ;
        /*species 23: C2H5O */
        species[23] =
            +4.94420708e-01 * tc[0]
            +2.71774434e-02 * tc[1]
            -8.29545050e-06 * tc[2]
            +1.71734733e-09 * tc[3]
            -1.62124229e-13 * tc[4]
            +2.28079378e+01 ;
    } else {
        /*species 0: N2 */
        species[0] =
            +2.92664000e+00 * tc[0]
            +1.48797680e-03 * tc[1]
            -2.84238000e-07 * tc[2]
            +3.36567933e-11 * tc[3]
            -1.68833775e-15 * tc[4]
            +5.98052800e+00 ;
        /*species 1: H */
        species[1] =
            +2.50000001e+00 * tc[0]
            -2.30842973e-11 * tc[1]
            +8.07809740e-15 * tc[2]
            -1.57838412e-18 * tc[3]
            +1.24549339e-22 * tc[4]
            -4.46682914e-01 ;
        /*species 2: H2 */
        species[2] =
            +3.33727920e+00 * tc[0]
            -4.94024731e-05 * tc[1]
            +2.49728389e-07 * tc[2]
            -5.98554647e-11 * tc[3]
            +5.00638440e-15 * tc[4]
            -3.20502331e+00 ;
        /*species 3: O */
        species[3] =
            +2.56942078e+00 * tc[0]
            -8.59741137e-05 * tc[1]
            +2.09742295e-08 * tc[2]
            -3.33925997e-12 * tc[3]
            +3.07084227e-16 * tc[4]
            +4.78433864e+00 ;
        /*species 4: OH */
        species[4] =
            +2.86472886e+00 * tc[0]
            +1.05650448e-03 * tc[1]
            -1.29541379e-07 * tc[2]
            +1.01739558e-11 * tc[3]
            -3.32989690e-16 * tc[4]
            +5.70164073e+00 ;
        /*species 5: O2 */
        species[5] =
            +3.28253784e+00 * tc[0]
            +1.48308754e-03 * tc[1]
            -3.78983334e-07 * tc[2]
            +6.98235183e-11 * tc[3]
            -5.41794485e-15 * tc[4]
            +5.45323129e+00 ;
        /*species 6: H2O2 */
        species[6] =
            +4.16500285e+00 * tc[0]
            +4.90831694e-03 * tc[1]
            -9.50696125e-07 * tc[2]
            +1.23728662e-10 * tc[3]
            -7.19770763e-15 * tc[4]
            +2.91615662e+00 ;
        /*species 7: H2O */
        species[7] =
            +3.03399249e+00 * tc[0]
            +2.17691804e-03 * tc[1]
            -8.20362590e-08 * tc[2]
            -3.23473290e-11 * tc[3]
            +4.20502480e-15 * tc[4]
            +4.96677010e+00 ;
        /*species 8: HO2 */
        species[8] =
            +4.01721090e+00 * tc[0]
            +2.23982013e-03 * tc[1]
            -3.16829075e-07 * tc[2]
            +3.80821233e-11 * tc[3]
            -2.69771337e-15 * tc[4]
            +3.78510215e+00 ;
        /*species 9: CO */
        species[9] =
            +2.71518561e+00 * tc[0]
            +2.06252743e-03 * tc[1]
            -4.99412886e-07 * tc[2]
            +7.66843360e-11 * tc[3]
            -5.09119290e-15 * tc[4]
            +7.81868772e+00 ;
        /*species 10: CH3 */
        species[10] =
            +2.97812060e+00 * tc[0]
            +5.79785200e-03 * tc[1]
            -9.87790000e-07 * tc[2]
            +1.02432633e-10 * tc[3]
            -4.47935400e-15 * tc[4]
            +4.72247990e+00 ;
        /*species 11: CH2O */
        species[11] =
            +1.76069008e+00 * tc[0]
            +9.20000082e-03 * tc[1]
            -2.21129406e-06 * tc[2]
            +3.35470707e-10 * tc[3]
            -2.20963910e-14 * tc[4]
            +1.36563230e+01 ;
        /*species 12: CO2 */
        species[12] =
            +3.85746029e+00 * tc[0]
            +4.41437026e-03 * tc[1]
            -1.10740702e-06 * tc[2]
            +1.74496729e-10 * tc[3]
            -1.18021041e-14 * tc[4]
            +2.27163806e+00 ;
        /*species 13: CH4 */
        species[13] =
            +1.65326226e+00 * tc[0]
            +1.00263099e-02 * tc[1]
            -1.65830619e-06 * tc[2]
            +1.78827713e-10 * tc[3]
            -7.86741895e-15 * tc[4]
            +9.90506283e+00 ;
        /*species 14: C2H2 */
        species[14] =
            +4.14756964e+00 * tc[0]
            +5.96166664e-03 * tc[1]
            -1.18647426e-06 * tc[2]
            +1.55804057e-10 * tc[3]
            -9.03088033e-15 * tc[4]
            -1.23028121e+00 ;
        /*species 15: C2H4 */
        species[15] =
            +2.03611116e+00 * tc[0]
            +1.46454151e-02 * tc[1]
            -3.35538958e-06 * tc[2]
            +4.90743077e-10 * tc[3]
            -3.14265152e-14 * tc[4]
            +1.03053693e+01 ;
        /*species 16: CH2CO */
        species[16] =
            +4.51129732e+00 * tc[0]
            +9.00359745e-03 * tc[1]
            -2.08469817e-06 * tc[2]
            +3.07781961e-10 * tc[3]
            -1.98709550e-14 * tc[4]
            +6.32247205e-01 ;
        /*species 17: C2H6 */
        species[17] =
            +1.07188150e+00 * tc[0]
            +2.16852677e-02 * tc[1]
            -5.01280335e-06 * tc[2]
            +7.38040003e-10 * tc[3]
            -4.75007225e-14 * tc[4]
            +1.51156107e+01 ;
        /*species 18: C2H3 */
        species[18] =
            +3.01672400e+00 * tc[0]
            +1.03302292e-02 * tc[1]
            -2.34041174e-06 * tc[2]
            +3.39210960e-10 * tc[3]
            -2.15651760e-14 * tc[4]
            +7.78732378e+00 ;
        /*species 19: C2H5 */
        species[19] =
            +1.95465642e+00 * tc[0]
            +1.73972722e-02 * tc[1]
            -3.99103334e-06 * tc[2]
            +5.84058963e-10 * tc[3]
            -3.74103940e-14 * tc[4]
            +1.34624343e+01 ;
        /*species 20: HCCO */
        species[20] =
            +5.62820580e+00 * tc[0]
            +4.08534010e-03 * tc[1]
            -7.96727350e-07 * tc[2]
            +9.54201733e-11 * tc[3]
            -4.85195800e-15 * tc[4]
            -3.93025950e+00 ;
        /*species 21: CH3CHO */
        species[21] =
            +2.68543112e+00 * tc[0]
            +1.76802373e-02 * tc[1]
            -4.32701369e-06 * tc[2]
            +6.78935297e-10 * tc[3]
            -4.69077337e-14 * tc[4]
            +1.11635653e+01 ;
        /*species 22: CH2CHO */
        species[22] =
            +2.42606357e+00 * tc[0]
            +1.72400021e-02 * tc[1]
            -4.88566059e-06 * tc[2]
            +8.88518907e-10 * tc[3]
            -7.05300195e-14 * tc[4]
            +1.26038737e+01 ;
        /*species 23: C2H5O */
        species[23] =
            +2.46262349e+00 * tc[0]
            +2.09503959e-02 * tc[1]
            -4.69645875e-06 * tc[2]
            +5.21468757e-10 * tc[3]
            +0.00000000e+00 * tc[4]
            +1.28738847e+01 ;
    }
    return;
}


/*save atomic weights into array */
void atomicWeight(double *  awt)
{
    awt[0] = 14.006700; /*N */
    awt[1] = 1.007970; /*H */
    awt[2] = 15.999400; /*O */
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
    *LENIMC = 98;}


void egtransetLENRMC(int* LENRMC ) {
    *LENRMC = 11784;}


void egtransetNO(int* NO ) {
    *NO = 4;}


void egtransetKK(int* KK ) {
    *KK = 24;}


void egtransetNLITE(int* NLITE ) {
    *NLITE = 2;}


/*Patm in ergs/cm3 */
void egtransetPATM(double* PATM) {
    *PATM =   0.1013250000000000E+07;}


/*the molecular weights in g/mol */
void egtransetWT(double* WT ) {
    WT[0] = 2.80134000E+01;
    WT[1] = 1.00797000E+00;
    WT[2] = 2.01594000E+00;
    WT[3] = 1.59994000E+01;
    WT[4] = 1.70073700E+01;
    WT[5] = 3.19988000E+01;
    WT[6] = 3.40147400E+01;
    WT[7] = 1.80153400E+01;
    WT[8] = 3.30067700E+01;
    WT[9] = 2.80105500E+01;
    WT[10] = 1.50350600E+01;
    WT[11] = 3.00264900E+01;
    WT[12] = 4.40099500E+01;
    WT[13] = 1.60430300E+01;
    WT[14] = 2.60382400E+01;
    WT[15] = 2.80541800E+01;
    WT[16] = 4.20376400E+01;
    WT[17] = 3.00701200E+01;
    WT[18] = 2.70462100E+01;
    WT[19] = 2.90621500E+01;
    WT[20] = 4.10296700E+01;
    WT[21] = 4.40535800E+01;
    WT[22] = 4.30456100E+01;
    WT[23] = 4.50615500E+01;
}


/*the lennard-jones potential well depth eps/kb in K */
void egtransetEPS(double* EPS ) {
    EPS[0] = 9.75300000E+01;
    EPS[1] = 1.45000000E+02;
    EPS[2] = 3.80000000E+01;
    EPS[3] = 8.00000000E+01;
    EPS[4] = 8.00000000E+01;
    EPS[5] = 1.07400000E+02;
    EPS[6] = 1.07400000E+02;
    EPS[7] = 5.72400000E+02;
    EPS[8] = 1.07400000E+02;
    EPS[9] = 9.81000000E+01;
    EPS[10] = 1.44000000E+02;
    EPS[11] = 4.98000000E+02;
    EPS[12] = 2.44000000E+02;
    EPS[13] = 1.41400000E+02;
    EPS[14] = 2.09000000E+02;
    EPS[15] = 2.80800000E+02;
    EPS[16] = 4.36000000E+02;
    EPS[17] = 2.52300000E+02;
    EPS[18] = 2.09000000E+02;
    EPS[19] = 2.52300000E+02;
    EPS[20] = 1.50000000E+02;
    EPS[21] = 4.36000000E+02;
    EPS[22] = 4.36000000E+02;
    EPS[23] = 4.70600000E+02;
}


/*the lennard-jones collision diameter in Angstroms */
void egtransetSIG(double* SIG ) {
    SIG[0] = 3.62100000E+00;
    SIG[1] = 2.05000000E+00;
    SIG[2] = 2.92000000E+00;
    SIG[3] = 2.75000000E+00;
    SIG[4] = 2.75000000E+00;
    SIG[5] = 3.45800000E+00;
    SIG[6] = 3.45800000E+00;
    SIG[7] = 2.60500000E+00;
    SIG[8] = 3.45800000E+00;
    SIG[9] = 3.65000000E+00;
    SIG[10] = 3.80000000E+00;
    SIG[11] = 3.59000000E+00;
    SIG[12] = 3.76300000E+00;
    SIG[13] = 3.74600000E+00;
    SIG[14] = 4.10000000E+00;
    SIG[15] = 3.97100000E+00;
    SIG[16] = 3.97000000E+00;
    SIG[17] = 4.30200000E+00;
    SIG[18] = 4.10000000E+00;
    SIG[19] = 4.30200000E+00;
    SIG[20] = 2.50000000E+00;
    SIG[21] = 3.97000000E+00;
    SIG[22] = 3.97000000E+00;
    SIG[23] = 4.41000000E+00;
}


/*the dipole moment in Debye */
void egtransetDIP(double* DIP ) {
    DIP[0] = 0.00000000E+00;
    DIP[1] = 0.00000000E+00;
    DIP[2] = 0.00000000E+00;
    DIP[3] = 0.00000000E+00;
    DIP[4] = 0.00000000E+00;
    DIP[5] = 0.00000000E+00;
    DIP[6] = 0.00000000E+00;
    DIP[7] = 1.84400000E+00;
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
}


/*the polarizability in cubic Angstroms */
void egtransetPOL(double* POL ) {
    POL[0] = 1.76000000E+00;
    POL[1] = 0.00000000E+00;
    POL[2] = 7.90000000E-01;
    POL[3] = 0.00000000E+00;
    POL[4] = 0.00000000E+00;
    POL[5] = 1.60000000E+00;
    POL[6] = 0.00000000E+00;
    POL[7] = 0.00000000E+00;
    POL[8] = 0.00000000E+00;
    POL[9] = 1.95000000E+00;
    POL[10] = 0.00000000E+00;
    POL[11] = 0.00000000E+00;
    POL[12] = 2.65000000E+00;
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
}


/*the rotational relaxation collision number at 298 K */
void egtransetZROT(double* ZROT ) {
    ZROT[0] = 4.00000000E+00;
    ZROT[1] = 0.00000000E+00;
    ZROT[2] = 2.80000000E+02;
    ZROT[3] = 0.00000000E+00;
    ZROT[4] = 0.00000000E+00;
    ZROT[5] = 3.80000000E+00;
    ZROT[6] = 3.80000000E+00;
    ZROT[7] = 4.00000000E+00;
    ZROT[8] = 1.00000000E+00;
    ZROT[9] = 1.80000000E+00;
    ZROT[10] = 0.00000000E+00;
    ZROT[11] = 2.00000000E+00;
    ZROT[12] = 2.10000000E+00;
    ZROT[13] = 1.30000000E+01;
    ZROT[14] = 2.50000000E+00;
    ZROT[15] = 1.50000000E+00;
    ZROT[16] = 2.00000000E+00;
    ZROT[17] = 1.50000000E+00;
    ZROT[18] = 1.00000000E+00;
    ZROT[19] = 1.50000000E+00;
    ZROT[20] = 1.00000000E+00;
    ZROT[21] = 2.00000000E+00;
    ZROT[22] = 2.00000000E+00;
    ZROT[23] = 1.50000000E+00;
}


/*0: monoatomic, 1: linear, 2: nonlinear */
void egtransetNLIN(int* NLIN) {
    NLIN[0] = 1;
    NLIN[1] = 0;
    NLIN[2] = 1;
    NLIN[3] = 0;
    NLIN[4] = 1;
    NLIN[5] = 1;
    NLIN[6] = 2;
    NLIN[7] = 2;
    NLIN[8] = 2;
    NLIN[9] = 1;
    NLIN[10] = 1;
    NLIN[11] = 2;
    NLIN[12] = 1;
    NLIN[13] = 2;
    NLIN[14] = 1;
    NLIN[15] = 2;
    NLIN[16] = 2;
    NLIN[17] = 2;
    NLIN[18] = 2;
    NLIN[19] = 2;
    NLIN[20] = 2;
    NLIN[21] = 2;
    NLIN[22] = 2;
    NLIN[23] = 2;
}


/*Poly fits for the viscosities, dim NO*KK */
void egtransetCOFETA(double* COFETA) {
    COFETA[0] = -1.73976942E+01;
    COFETA[1] = 2.73482764E+00;
    COFETA[2] = -2.81916034E-01;
    COFETA[3] = 1.26588405E-02;
    COFETA[4] = -2.08449616E+01;
    COFETA[5] = 3.82794146E+00;
    COFETA[6] = -4.20999003E-01;
    COFETA[7] = 1.85596636E-02;
    COFETA[8] = -1.40419527E+01;
    COFETA[9] = 1.08789225E+00;
    COFETA[10] = -6.18592115E-02;
    COFETA[11] = 2.86838304E-03;
    COFETA[12] = -1.57832266E+01;
    COFETA[13] = 2.21311180E+00;
    COFETA[14] = -2.12959301E-01;
    COFETA[15] = 9.62195191E-03;
    COFETA[16] = -1.57526788E+01;
    COFETA[17] = 2.21311180E+00;
    COFETA[18] = -2.12959301E-01;
    COFETA[19] = 9.62195191E-03;
    COFETA[20] = -1.78915826E+01;
    COFETA[21] = 2.98311502E+00;
    COFETA[22] = -3.14105508E-01;
    COFETA[23] = 1.40500162E-02;
    COFETA[24] = -1.78610348E+01;
    COFETA[25] = 2.98311502E+00;
    COFETA[26] = -3.14105508E-01;
    COFETA[27] = 1.40500162E-02;
    COFETA[28] = -1.14441613E+01;
    COFETA[29] = -9.67162014E-01;
    COFETA[30] = 3.58651197E-01;
    COFETA[31] = -2.09789135E-02;
    COFETA[32] = -1.78760754E+01;
    COFETA[33] = 2.98311502E+00;
    COFETA[34] = -3.14105508E-01;
    COFETA[35] = 1.40500162E-02;
    COFETA[36] = -1.74469975E+01;
    COFETA[37] = 2.74728386E+00;
    COFETA[38] = -2.83509015E-01;
    COFETA[39] = 1.27267083E-02;
    COFETA[40] = -2.06651974E+01;
    COFETA[41] = 3.80445652E+00;
    COFETA[42] = -4.18022689E-01;
    COFETA[43] = 1.84337719E-02;
    COFETA[44] = -1.57321558E+01;
    COFETA[45] = 9.81716009E-01;
    COFETA[46] = 7.09938181E-02;
    COFETA[47] = -7.70023966E-03;
    COFETA[48] = -2.28110345E+01;
    COFETA[49] = 4.62954710E+00;
    COFETA[50] = -5.00689001E-01;
    COFETA[51] = 2.10012969E-02;
    COFETA[52] = -2.04642214E+01;
    COFETA[53] = 3.75363357E+00;
    COFETA[54] = -4.11756834E-01;
    COFETA[55] = 1.81766117E-02;
    COFETA[56] = -2.28688119E+01;
    COFETA[57] = 4.58175030E+00;
    COFETA[58] = -5.05624274E-01;
    COFETA[59] = 2.16754140E-02;
    COFETA[60] = -2.29541095E+01;
    COFETA[61] = 4.43807276E+00;
    COFETA[62] = -4.62324516E-01;
    COFETA[63] = 1.87855972E-02;
    COFETA[64] = -1.83427672E+01;
    COFETA[65] = 2.18529340E+00;
    COFETA[66] = -1.07967566E-01;
    COFETA[67] = 9.43208566E-04;
    COFETA[68] = -2.32591637E+01;
    COFETA[69] = 4.59930831E+00;
    COFETA[70] = -4.93738512E-01;
    COFETA[71] = 2.05755211E-02;
    COFETA[72] = -2.28498216E+01;
    COFETA[73] = 4.58175030E+00;
    COFETA[74] = -5.05624274E-01;
    COFETA[75] = 2.16754140E-02;
    COFETA[76] = -2.32762114E+01;
    COFETA[77] = 4.59930831E+00;
    COFETA[78] = -4.93738512E-01;
    COFETA[79] = 2.05755211E-02;
    COFETA[80] = -1.96853484E+01;
    COFETA[81] = 3.93804789E+00;
    COFETA[82] = -4.34874374E-01;
    COFETA[83] = 1.91434397E-02;
    COFETA[84] = -1.83193466E+01;
    COFETA[85] = 2.18529340E+00;
    COFETA[86] = -1.07967566E-01;
    COFETA[87] = 9.43208566E-04;
    COFETA[88] = -1.83309198E+01;
    COFETA[89] = 2.18529340E+00;
    COFETA[90] = -1.07967566E-01;
    COFETA[91] = 9.43208566E-04;
    COFETA[92] = -1.70946961E+01;
    COFETA[93] = 1.51676428E+00;
    COFETA[94] = -8.14494324E-03;
    COFETA[95] = -3.89467245E-03;
}


/*Poly fits for the conductivities, dim NO*KK */
void egtransetCOFLAM(double* COFLAM) {
    COFLAM[0] = 7.77703429E+00;
    COFLAM[1] = -1.30957605E+00;
    COFLAM[2] = 3.28842203E-01;
    COFLAM[3] = -1.69485484E-02;
    COFLAM[4] = -1.29505111E+00;
    COFLAM[5] = 3.82794146E+00;
    COFLAM[6] = -4.20999003E-01;
    COFLAM[7] = 1.85596636E-02;
    COFLAM[8] = 4.34729192E+00;
    COFLAM[9] = 1.55347646E+00;
    COFLAM[10] = -1.60615552E-01;
    COFLAM[11] = 9.89934485E-03;
    COFLAM[12] = 1.00207105E+00;
    COFLAM[13] = 2.21311180E+00;
    COFLAM[14] = -2.12959301E-01;
    COFLAM[15] = 9.62195191E-03;
    COFLAM[16] = 9.94836323E+00;
    COFLAM[17] = -1.50754535E+00;
    COFLAM[18] = 2.97485797E-01;
    COFLAM[19] = -1.26939780E-02;
    COFLAM[20] = 5.31578403E-01;
    COFLAM[21] = 1.87067453E+00;
    COFLAM[22] = -1.31586198E-01;
    COFLAM[23] = 5.22416151E-03;
    COFLAM[24] = 2.91645348E+00;
    COFLAM[25] = 4.58703148E-01;
    COFLAM[26] = 1.38770970E-01;
    COFLAM[27] = -9.95034889E-03;
    COFLAM[28] = 1.81612053E+01;
    COFLAM[29] = -6.74137053E+00;
    COFLAM[30] = 1.21372119E+00;
    COFLAM[31] = -6.11027962E-02;
    COFLAM[32] = 3.32637961E+00;
    COFLAM[33] = 4.13455985E-01;
    COFLAM[34] = 1.13001798E-01;
    COFLAM[35] = -7.33441917E-03;
    COFLAM[36] = 8.17515440E+00;
    COFLAM[37] = -1.53836440E+00;
    COFLAM[38] = 3.68036945E-01;
    COFLAM[39] = -1.90917513E-02;
    COFLAM[40] = 1.07504941E+01;
    COFLAM[41] = -3.21671396E+00;
    COFLAM[42] = 6.99755311E-01;
    COFLAM[43] = -3.76626365E-02;
    COFLAM[44] = 1.43280346E+01;
    COFLAM[45] = -6.06564714E+00;
    COFLAM[46] = 1.23845525E+00;
    COFLAM[47] = -6.82424853E-02;
    COFLAM[48] = -8.74831432E+00;
    COFLAM[49] = 4.79275291E+00;
    COFLAM[50] = -4.18685061E-01;
    COFLAM[51] = 1.35210242E-02;
    COFLAM[52] = 1.62788502E+01;
    COFLAM[53] = -6.01787002E+00;
    COFLAM[54] = 1.15009703E+00;
    COFLAM[55] = -6.01972208E-02;
    COFLAM[56] = -1.08133123E+01;
    COFLAM[57] = 5.87459846E+00;
    COFLAM[58] = -5.86231060E-01;
    COFLAM[59] = 2.24413694E-02;
    COFLAM[60] = -4.03621314E+00;
    COFLAM[61] = 1.90761320E+00;
    COFLAM[62] = 1.17538667E-01;
    COFLAM[63] = -1.61049429E-02;
    COFLAM[64] = -5.12380709E+00;
    COFLAM[65] = 2.64646984E+00;
    COFLAM[66] = -4.03547100E-02;
    COFLAM[67] = -6.69366224E-03;
    COFLAM[68] = -2.11843381E+00;
    COFLAM[69] = 9.97190789E-01;
    COFLAM[70] = 2.61096626E-01;
    COFLAM[71] = -2.33522396E-02;
    COFLAM[72] = -4.10451426E+00;
    COFLAM[73] = 2.44778927E+00;
    COFLAM[74] = -2.59480466E-02;
    COFLAM[75] = -6.86823584E-03;
    COFLAM[76] = -1.28446241E+00;
    COFLAM[77] = 8.20566934E-01;
    COFLAM[78] = 2.59388957E-01;
    COFLAM[79] = -2.22604138E-02;
    COFLAM[80] = -6.40750956E+00;
    COFLAM[81] = 4.57147879E+00;
    COFLAM[82] = -4.21103535E-01;
    COFLAM[83] = 1.49974513E-02;
    COFLAM[84] = -4.24789577E+00;
    COFLAM[85] = 1.81945153E+00;
    COFLAM[86] = 1.42329377E-01;
    COFLAM[87] = -1.77039932E-02;
    COFLAM[88] = -8.22876105E+00;
    COFLAM[89] = 3.63517070E+00;
    COFLAM[90] = -1.29160800E-01;
    COFLAM[91] = -4.62103507E-03;
    COFLAM[92] = -3.10284616E+01;
    COFLAM[93] = 1.39182798E+01;
    COFLAM[94] = -1.68191130E+00;
    COFLAM[95] = 7.36684976E-02;
}


/*Poly fits for the diffusion coefficients, dim NO*KK*KK */
void egtransetCOFD(double* COFD) {
    COFD[0] = -1.56019563E+01;
    COFD[1] = 3.51542686E+00;
    COFD[2] = -2.47677471E-01;
    COFD[3] = 1.09841319E-02;
    COFD[4] = -1.49410568E+01;
    COFD[5] = 3.94457262E+00;
    COFD[6] = -3.02046024E-01;
    COFD[7] = 1.32834546E-02;
    COFD[8] = -1.20381391E+01;
    COFD[9] = 2.61421687E+00;
    COFD[10] = -1.28887086E-01;
    COFD[11] = 5.75609167E-03;
    COFD[12] = -1.47115631E+01;
    COFD[13] = 3.34121229E+00;
    COFD[14] = -2.25642864E-01;
    COFD[15] = 1.00547477E-02;
    COFD[16] = -1.47307892E+01;
    COFD[17] = 3.34121229E+00;
    COFD[18] = -2.25642864E-01;
    COFD[19] = 1.00547477E-02;
    COFD[20] = -1.58214936E+01;
    COFD[21] = 3.60000113E+00;
    COFD[22] = -2.58255120E-01;
    COFD[23] = 1.14251480E-02;
    COFD[24] = -1.58355212E+01;
    COFD[25] = 3.60000113E+00;
    COFD[26] = -2.58255120E-01;
    COFD[27] = 1.14251480E-02;
    COFD[28] = -1.99472346E+01;
    COFD[29] = 5.05636623E+00;
    COFD[30] = -4.17733686E-01;
    COFD[31] = 1.71403506E-02;
    COFD[32] = -1.58286724E+01;
    COFD[33] = 3.60000113E+00;
    COFD[34] = -2.58255120E-01;
    COFD[35] = 1.14251480E-02;
    COFD[36] = -1.56221627E+01;
    COFD[37] = 3.51977302E+00;
    COFD[38] = -2.48210923E-01;
    COFD[39] = 1.10059241E-02;
    COFD[40] = -1.66140265E+01;
    COFD[41] = 3.93776813E+00;
    COFD[42] = -3.01210554E-01;
    COFD[43] = 1.32492810E-02;
    COFD[44] = -1.99958660E+01;
    COFD[45] = 5.01699515E+00;
    COFD[46] = -4.23791128E-01;
    COFD[47] = 1.78520267E-02;
    COFD[48] = -1.85068873E+01;
    COFD[49] = 4.52122572E+00;
    COFD[50] = -3.73088946E-01;
    COFD[51] = 1.62076520E-02;
    COFD[52] = -1.65706884E+01;
    COFD[53] = 3.92005093E+00;
    COFD[54] = -2.99040611E-01;
    COFD[55] = 1.31607610E-02;
    COFD[56] = -1.79334289E+01;
    COFD[57] = 4.32370416E+00;
    COFD[58] = -3.48535839E-01;
    COFD[59] = 1.51861878E-02;
    COFD[60] = -1.88223266E+01;
    COFD[61] = 4.64181074E+00;
    COFD[62] = -3.86929312E-01;
    COFD[63] = 1.67349243E-02;
    COFD[64] = -1.99828895E+01;
    COFD[65] = 4.97669986E+00;
    COFD[66] = -4.22149407E-01;
    COFD[67] = 1.79263822E-02;
    COFD[68] = -1.86644793E+01;
    COFD[69] = 4.55615891E+00;
    COFD[70] = -3.77295769E-01;
    COFD[71] = 1.63770696E-02;
    COFD[72] = -1.79431810E+01;
    COFD[73] = 4.32370416E+00;
    COFD[74] = -3.48535839E-01;
    COFD[75] = 1.51861878E-02;
    COFD[76] = -1.86561847E+01;
    COFD[77] = 4.55615891E+00;
    COFD[78] = -3.77295769E-01;
    COFD[79] = 1.63770696E-02;
    COFD[80] = -1.66078875E+01;
    COFD[81] = 3.97848216E+00;
    COFD[82] = -3.06230410E-01;
    COFD[83] = 1.34556563E-02;
    COFD[84] = -1.99921241E+01;
    COFD[85] = 4.97669986E+00;
    COFD[86] = -4.22149407E-01;
    COFD[87] = 1.79263822E-02;
    COFD[88] = -1.99875936E+01;
    COFD[89] = 4.97669986E+00;
    COFD[90] = -4.22149407E-01;
    COFD[91] = 1.79263822E-02;
    COFD[92] = -2.02337108E+01;
    COFD[93] = 5.00680505E+00;
    COFD[94] = -4.24113291E-01;
    COFD[95] = 1.79327553E-02;
    COFD[96] = -1.49410568E+01;
    COFD[97] = 3.94457262E+00;
    COFD[98] = -3.02046024E-01;
    COFD[99] = 1.32834546E-02;
    COFD[100] = -1.51395399E+01;
    COFD[101] = 4.36621619E+00;
    COFD[102] = -3.53866950E-01;
    COFD[103] = 1.54097445E-02;
    COFD[104] = -1.19370371E+01;
    COFD[105] = 2.99054752E+00;
    COFD[106] = -1.79624448E-01;
    COFD[107] = 8.03970815E-03;
    COFD[108] = -1.39677477E+01;
    COFD[109] = 3.71279442E+00;
    COFD[110] = -2.72718508E-01;
    COFD[111] = 1.20446550E-02;
    COFD[112] = -1.39695071E+01;
    COFD[113] = 3.71279442E+00;
    COFD[114] = -2.72718508E-01;
    COFD[115] = 1.20446550E-02;
    COFD[116] = -1.51442279E+01;
    COFD[117] = 4.03719698E+00;
    COFD[118] = -3.13407940E-01;
    COFD[119] = 1.37483092E-02;
    COFD[120] = -1.51451337E+01;
    COFD[121] = 4.03719698E+00;
    COFD[122] = -3.13407940E-01;
    COFD[123] = 1.37483092E-02;
    COFD[124] = -1.80638064E+01;
    COFD[125] = 4.96477835E+00;
    COFD[126] = -3.98489680E-01;
    COFD[127] = 1.60121621E-02;
    COFD[128] = -1.51446944E+01;
    COFD[129] = 4.03719698E+00;
    COFD[130] = -3.13407940E-01;
    COFD[131] = 1.37483092E-02;
    COFD[132] = -1.49673213E+01;
    COFD[133] = 3.95033191E+00;
    COFD[134] = -3.02754076E-01;
    COFD[135] = 1.33124608E-02;
    COFD[136] = -1.61384688E+01;
    COFD[137] = 4.35660780E+00;
    COFD[138] = -3.52659055E-01;
    COFD[139] = 1.53589488E-02;
    COFD[140] = -1.85337299E+01;
    COFD[141] = 5.04618677E+00;
    COFD[142] = -4.14801122E-01;
    COFD[143] = 1.69487185E-02;
    COFD[144] = -1.76943558E+01;
    COFD[145] = 4.88672714E+00;
    COFD[146] = -4.14778255E-01;
    COFD[147] = 1.77823474E-02;
    COFD[148] = -1.60530416E+01;
    COFD[149] = 4.33128165E+00;
    COFD[150] = -3.49478110E-01;
    COFD[151] = 1.52253415E-02;
    COFD[152] = -1.74064914E+01;
    COFD[153] = 4.76097419E+00;
    COFD[154] = -4.01175756E-01;
    COFD[155] = 1.73035102E-02;
    COFD[156] = -1.80153619E+01;
    COFD[157] = 4.95555480E+00;
    COFD[158] = -4.20418912E-01;
    COFD[159] = 1.78921168E-02;
    COFD[160] = -1.86132026E+01;
    COFD[161] = 5.06806259E+00;
    COFD[162] = -4.22384317E-01;
    COFD[163] = 1.74675251E-02;
    COFD[164] = -1.79190223E+01;
    COFD[165] = 4.89943438E+00;
    COFD[166] = -4.15602392E-01;
    COFD[167] = 1.77841166E-02;
    COFD[168] = -1.74071864E+01;
    COFD[169] = 4.76097419E+00;
    COFD[170] = -4.01175756E-01;
    COFD[171] = 1.73035102E-02;
    COFD[172] = -1.79184601E+01;
    COFD[173] = 4.89943438E+00;
    COFD[174] = -4.15602392E-01;
    COFD[175] = 1.77841166E-02;
    COFD[176] = -1.58083316E+01;
    COFD[177] = 4.41251740E+00;
    COFD[178] = -3.59679513E-01;
    COFD[179] = 1.56539353E-02;
    COFD[180] = -1.86137387E+01;
    COFD[181] = 5.06806259E+00;
    COFD[182] = -4.22384317E-01;
    COFD[183] = 1.74675251E-02;
    COFD[184] = -1.86134769E+01;
    COFD[185] = 5.06806259E+00;
    COFD[186] = -4.22384317E-01;
    COFD[187] = 1.74675251E-02;
    COFD[188] = -1.87941740E+01;
    COFD[189] = 5.05845386E+00;
    COFD[190] = -4.18429681E-01;
    COFD[191] = 1.71874793E-02;
    COFD[192] = -1.20381391E+01;
    COFD[193] = 2.61421687E+00;
    COFD[194] = -1.28887086E-01;
    COFD[195] = 5.75609167E-03;
    COFD[196] = -1.19370371E+01;
    COFD[197] = 2.99054752E+00;
    COFD[198] = -1.79624448E-01;
    COFD[199] = 8.03970815E-03;
    COFD[200] = -1.04285080E+01;
    COFD[201] = 2.23477534E+00;
    COFD[202] = -8.11809423E-02;
    COFD[203] = 3.77342041E-03;
    COFD[204] = -1.12653981E+01;
    COFD[205] = 2.43094296E+00;
    COFD[206] = -1.03798673E-01;
    COFD[207] = 4.60962717E-03;
    COFD[208] = -1.12687251E+01;
    COFD[209] = 2.43094296E+00;
    COFD[210] = -1.03798673E-01;
    COFD[211] = 4.60962717E-03;
    COFD[212] = -1.22181183E+01;
    COFD[213] = 2.70415313E+00;
    COFD[214] = -1.41236971E-01;
    COFD[215] = 6.32236816E-03;
    COFD[216] = -1.22198776E+01;
    COFD[217] = 2.70415313E+00;
    COFD[218] = -1.41236971E-01;
    COFD[219] = 6.32236816E-03;
    COFD[220] = -1.73864044E+01;
    COFD[221] = 4.71143088E+00;
    COFD[222] = -3.95288626E-01;
    COFD[223] = 1.70702272E-02;
    COFD[224] = -1.22190241E+01;
    COFD[225] = 2.70415313E+00;
    COFD[226] = -1.41236971E-01;
    COFD[227] = 6.32236816E-03;
    COFD[228] = -1.20607690E+01;
    COFD[229] = 2.61969379E+00;
    COFD[230] = -1.29638429E-01;
    COFD[231] = 5.79050588E-03;
    COFD[232] = -1.30102341E+01;
    COFD[233] = 2.98419920E+00;
    COFD[234] = -1.78783571E-01;
    COFD[235] = 8.00253963E-03;
    COFD[236] = -1.64253252E+01;
    COFD[237] = 4.26219537E+00;
    COFD[238] = -3.41524254E-01;
    COFD[239] = 1.49232070E-02;
    COFD[240] = -1.43978662E+01;
    COFD[241] = 3.49721576E+00;
    COFD[242] = -2.45465191E-01;
    COFD[243] = 1.08948372E-02;
    COFD[244] = -1.29544834E+01;
    COFD[245] = 2.96758239E+00;
    COFD[246] = -1.76586224E-01;
    COFD[247] = 7.90559536E-03;
    COFD[248] = -1.41052578E+01;
    COFD[249] = 3.35888991E+00;
    COFD[250] = -2.27939148E-01;
    COFD[251] = 1.01542945E-02;
    COFD[252] = -1.47832812E+01;
    COFD[253] = 3.61918603E+00;
    COFD[254] = -2.60701094E-01;
    COFD[255] = 1.15292870E-02;
    COFD[256] = -1.61505677E+01;
    COFD[257] = 4.11963986E+00;
    COFD[258] = -3.23983555E-01;
    COFD[259] = 1.42022890E-02;
    COFD[260] = -1.46113269E+01;
    COFD[261] = 3.52131229E+00;
    COFD[262] = -2.48399935E-01;
    COFD[263] = 1.10136492E-02;
    COFD[264] = -1.41065986E+01;
    COFD[265] = 3.35888991E+00;
    COFD[266] = -2.27939148E-01;
    COFD[267] = 1.01542945E-02;
    COFD[268] = -1.46102386E+01;
    COFD[269] = 3.52131229E+00;
    COFD[270] = -2.48399935E-01;
    COFD[271] = 1.10136492E-02;
    COFD[272] = -1.27155593E+01;
    COFD[273] = 3.02109660E+00;
    COFD[274] = -1.83658057E-01;
    COFD[275] = 8.21743108E-03;
    COFD[276] = -1.61516158E+01;
    COFD[277] = 4.11963986E+00;
    COFD[278] = -3.23983555E-01;
    COFD[279] = 1.42022890E-02;
    COFD[280] = -1.61511037E+01;
    COFD[281] = 4.11963986E+00;
    COFD[282] = -3.23983555E-01;
    COFD[283] = 1.42022890E-02;
    COFD[284] = -1.65400407E+01;
    COFD[285] = 4.21819964E+00;
    COFD[286] = -3.36577566E-01;
    COFD[287] = 1.47408361E-02;
    COFD[288] = -1.47115631E+01;
    COFD[289] = 3.34121229E+00;
    COFD[290] = -2.25642864E-01;
    COFD[291] = 1.00547477E-02;
    COFD[292] = -1.39677477E+01;
    COFD[293] = 3.71279442E+00;
    COFD[294] = -2.72718508E-01;
    COFD[295] = 1.20446550E-02;
    COFD[296] = -1.12653981E+01;
    COFD[297] = 2.43094296E+00;
    COFD[298] = -1.03798673E-01;
    COFD[299] = 4.60962717E-03;
    COFD[300] = -1.37174845E+01;
    COFD[301] = 3.11889373E+00;
    COFD[302] = -1.96402933E-01;
    COFD[303] = 8.77180880E-03;
    COFD[304] = -1.37325251E+01;
    COFD[305] = 3.11889373E+00;
    COFD[306] = -1.96402933E-01;
    COFD[307] = 8.77180880E-03;
    COFD[308] = -1.49340210E+01;
    COFD[309] = 3.43509376E+00;
    COFD[310] = -2.37713783E-01;
    COFD[311] = 1.05726006E-02;
    COFD[312] = -1.49439976E+01;
    COFD[313] = 3.43509376E+00;
    COFD[314] = -2.37713783E-01;
    COFD[315] = 1.05726006E-02;
    COFD[316] = -1.90632945E+01;
    COFD[317] = 5.00590780E+00;
    COFD[318] = -4.24055652E-01;
    COFD[319] = 1.79326322E-02;
    COFD[320] = -1.49391368E+01;
    COFD[321] = 3.43509376E+00;
    COFD[322] = -2.37713783E-01;
    COFD[323] = 1.05726006E-02;
    COFD[324] = -1.47359425E+01;
    COFD[325] = 3.34699328E+00;
    COFD[326] = -2.26393424E-01;
    COFD[327] = 1.00872697E-02;
    COFD[328] = -1.56185416E+01;
    COFD[329] = 3.70484350E+00;
    COFD[330] = -2.71706480E-01;
    COFD[331] = 1.20016288E-02;
    COFD[332] = -1.92562118E+01;
    COFD[333] = 4.94108508E+00;
    COFD[334] = -4.19008197E-01;
    COFD[335] = 1.78499457E-02;
    COFD[336] = -1.74119025E+01;
    COFD[337] = 4.28601449E+00;
    COFD[338] = -3.44182880E-01;
    COFD[339] = 1.50201783E-02;
    COFD[340] = -1.55602858E+01;
    COFD[341] = 3.68289956E+00;
    COFD[342] = -2.68870371E-01;
    COFD[343] = 1.18791438E-02;
    COFD[344] = -1.69909812E+01;
    COFD[345] = 4.13179721E+00;
    COFD[346] = -3.25544993E-01;
    COFD[347] = 1.42694195E-02;
    COFD[348] = -1.78714250E+01;
    COFD[349] = 4.45397294E+00;
    COFD[350] = -3.64836010E-01;
    COFD[351] = 1.58686336E-02;
    COFD[352] = -1.91880072E+01;
    COFD[353] = 4.87847769E+00;
    COFD[354] = -4.14012329E-01;
    COFD[355] = 1.77616586E-02;
    COFD[356] = -1.75974956E+01;
    COFD[357] = 4.31462284E+00;
    COFD[358] = -3.47471819E-01;
    COFD[359] = 1.51448695E-02;
    COFD[360] = -1.69981241E+01;
    COFD[361] = 4.13179721E+00;
    COFD[362] = -3.25544993E-01;
    COFD[363] = 1.42694195E-02;
    COFD[364] = -1.75915091E+01;
    COFD[365] = 4.31462284E+00;
    COFD[366] = -3.47471819E-01;
    COFD[367] = 1.51448695E-02;
    COFD[368] = -1.55038857E+01;
    COFD[369] = 3.75336567E+00;
    COFD[370] = -2.77901896E-01;
    COFD[371] = 1.22658049E-02;
    COFD[372] = -1.91943549E+01;
    COFD[373] = 4.87847769E+00;
    COFD[374] = -4.14012329E-01;
    COFD[375] = 1.77616586E-02;
    COFD[376] = -1.91912453E+01;
    COFD[377] = 4.87847769E+00;
    COFD[378] = -4.14012329E-01;
    COFD[379] = 1.77616586E-02;
    COFD[380] = -1.94550962E+01;
    COFD[381] = 4.91220991E+00;
    COFD[382] = -4.16560501E-01;
    COFD[383] = 1.77969106E-02;
    COFD[384] = -1.47307892E+01;
    COFD[385] = 3.34121229E+00;
    COFD[386] = -2.25642864E-01;
    COFD[387] = 1.00547477E-02;
    COFD[388] = -1.39695071E+01;
    COFD[389] = 3.71279442E+00;
    COFD[390] = -2.72718508E-01;
    COFD[391] = 1.20446550E-02;
    COFD[392] = -1.12687251E+01;
    COFD[393] = 2.43094296E+00;
    COFD[394] = -1.03798673E-01;
    COFD[395] = 4.60962717E-03;
    COFD[396] = -1.37325251E+01;
    COFD[397] = 3.11889373E+00;
    COFD[398] = -1.96402933E-01;
    COFD[399] = 8.77180880E-03;
    COFD[400] = -1.37480322E+01;
    COFD[401] = 3.11889373E+00;
    COFD[402] = -1.96402933E-01;
    COFD[403] = 8.77180880E-03;
    COFD[404] = -1.49541774E+01;
    COFD[405] = 3.43509376E+00;
    COFD[406] = -2.37713783E-01;
    COFD[407] = 1.05726006E-02;
    COFD[408] = -1.49645688E+01;
    COFD[409] = 3.43509376E+00;
    COFD[410] = -2.37713783E-01;
    COFD[411] = 1.05726006E-02;
    COFD[412] = -1.90792409E+01;
    COFD[413] = 5.00590780E+00;
    COFD[414] = -4.24055652E-01;
    COFD[415] = 1.79326322E-02;
    COFD[416] = -1.49595048E+01;
    COFD[417] = 3.43509376E+00;
    COFD[418] = -2.37713783E-01;
    COFD[419] = 1.05726006E-02;
    COFD[420] = -1.47551678E+01;
    COFD[421] = 3.34699328E+00;
    COFD[422] = -2.26393424E-01;
    COFD[423] = 1.00872697E-02;
    COFD[424] = -1.56331080E+01;
    COFD[425] = 3.70484350E+00;
    COFD[426] = -2.71706480E-01;
    COFD[427] = 1.20016288E-02;
    COFD[428] = -1.92759277E+01;
    COFD[429] = 4.94108508E+00;
    COFD[430] = -4.19008197E-01;
    COFD[431] = 1.78499457E-02;
    COFD[432] = -1.74341216E+01;
    COFD[433] = 4.28601449E+00;
    COFD[434] = -3.44182880E-01;
    COFD[435] = 1.50201783E-02;
    COFD[436] = -1.55753472E+01;
    COFD[437] = 3.68289956E+00;
    COFD[438] = -2.68870371E-01;
    COFD[439] = 1.18791438E-02;
    COFD[440] = -1.70096816E+01;
    COFD[441] = 4.13179721E+00;
    COFD[442] = -3.25544993E-01;
    COFD[443] = 1.42694195E-02;
    COFD[444] = -1.78906614E+01;
    COFD[445] = 4.45397294E+00;
    COFD[446] = -3.64836010E-01;
    COFD[447] = 1.58686336E-02;
    COFD[448] = -1.92099456E+01;
    COFD[449] = 4.87847769E+00;
    COFD[450] = -4.14012329E-01;
    COFD[451] = 1.77616586E-02;
    COFD[452] = -1.76172217E+01;
    COFD[453] = 4.31462284E+00;
    COFD[454] = -3.47471819E-01;
    COFD[455] = 1.51448695E-02;
    COFD[456] = -1.70170987E+01;
    COFD[457] = 4.13179721E+00;
    COFD[458] = -3.25544993E-01;
    COFD[459] = 1.42694195E-02;
    COFD[460] = -1.76109957E+01;
    COFD[461] = 4.31462284E+00;
    COFD[462] = -3.47471819E-01;
    COFD[463] = 1.51448695E-02;
    COFD[464] = -1.55256733E+01;
    COFD[465] = 3.75336567E+00;
    COFD[466] = -2.77901896E-01;
    COFD[467] = 1.22658049E-02;
    COFD[468] = -1.92165800E+01;
    COFD[469] = 4.87847769E+00;
    COFD[470] = -4.14012329E-01;
    COFD[471] = 1.77616586E-02;
    COFD[472] = -1.92133295E+01;
    COFD[473] = 4.87847769E+00;
    COFD[474] = -4.14012329E-01;
    COFD[475] = 1.77616586E-02;
    COFD[476] = -1.94774575E+01;
    COFD[477] = 4.91220991E+00;
    COFD[478] = -4.16560501E-01;
    COFD[479] = 1.77969106E-02;
    COFD[480] = -1.58214936E+01;
    COFD[481] = 3.60000113E+00;
    COFD[482] = -2.58255120E-01;
    COFD[483] = 1.14251480E-02;
    COFD[484] = -1.51442279E+01;
    COFD[485] = 4.03719698E+00;
    COFD[486] = -3.13407940E-01;
    COFD[487] = 1.37483092E-02;
    COFD[488] = -1.22181183E+01;
    COFD[489] = 2.70415313E+00;
    COFD[490] = -1.41236971E-01;
    COFD[491] = 6.32236816E-03;
    COFD[492] = -1.49340210E+01;
    COFD[493] = 3.43509376E+00;
    COFD[494] = -2.37713783E-01;
    COFD[495] = 1.05726006E-02;
    COFD[496] = -1.49541774E+01;
    COFD[497] = 3.43509376E+00;
    COFD[498] = -2.37713783E-01;
    COFD[499] = 1.05726006E-02;
    COFD[500] = -1.60936570E+01;
    COFD[501] = 3.70633871E+00;
    COFD[502] = -2.71897253E-01;
    COFD[503] = 1.20097588E-02;
    COFD[504] = -1.61086977E+01;
    COFD[505] = 3.70633871E+00;
    COFD[506] = -2.71897253E-01;
    COFD[507] = 1.20097588E-02;
    COFD[508] = -1.99035583E+01;
    COFD[509] = 5.01694644E+00;
    COFD[510] = -4.08963011E-01;
    COFD[511] = 1.66143416E-02;
    COFD[512] = -1.61013505E+01;
    COFD[513] = 3.70633871E+00;
    COFD[514] = -2.71897253E-01;
    COFD[515] = 1.20097588E-02;
    COFD[516] = -1.58458281E+01;
    COFD[517] = 3.60600362E+00;
    COFD[518] = -2.59019961E-01;
    COFD[519] = 1.14576923E-02;
    COFD[520] = -1.68515683E+01;
    COFD[521] = 4.03054927E+00;
    COFD[522] = -3.12591852E-01;
    COFD[523] = 1.37148695E-02;
    COFD[524] = -2.01508812E+01;
    COFD[525] = 5.05721632E+00;
    COFD[526] = -4.26359426E-01;
    COFD[527] = 1.78564586E-02;
    COFD[528] = -1.87634092E+01;
    COFD[529] = 4.61060397E+00;
    COFD[530] = -3.83564503E-01;
    COFD[531] = 1.66168246E-02;
    COFD[532] = -1.68102713E+01;
    COFD[533] = 4.01337907E+00;
    COFD[534] = -3.10488902E-01;
    COFD[535] = 1.36288975E-02;
    COFD[536] = -1.82748931E+01;
    COFD[537] = 4.45298414E+00;
    COFD[538] = -3.64713014E-01;
    COFD[539] = 1.58635108E-02;
    COFD[540] = -1.91360445E+01;
    COFD[541] = 4.75557446E+00;
    COFD[542] = -4.00539622E-01;
    COFD[543] = 1.72785461E-02;
    COFD[544] = -2.01264169E+01;
    COFD[545] = 5.01062757E+00;
    COFD[546] = -4.24032393E-01;
    COFD[547] = 1.79058125E-02;
    COFD[548] = -1.88920637E+01;
    COFD[549] = 4.63397272E+00;
    COFD[550] = -3.86072736E-01;
    COFD[551] = 1.67042917E-02;
    COFD[552] = -1.82852741E+01;
    COFD[553] = 4.45298414E+00;
    COFD[554] = -3.64713014E-01;
    COFD[555] = 1.58635108E-02;
    COFD[556] = -1.88832025E+01;
    COFD[557] = 4.63397272E+00;
    COFD[558] = -3.86072736E-01;
    COFD[559] = 1.67042917E-02;
    COFD[560] = -1.68776819E+01;
    COFD[561] = 4.08227013E+00;
    COFD[562] = -3.19188042E-01;
    COFD[563] = 1.39963401E-02;
    COFD[564] = -2.01364051E+01;
    COFD[565] = 5.01062757E+00;
    COFD[566] = -4.24032393E-01;
    COFD[567] = 1.79058125E-02;
    COFD[568] = -2.01315030E+01;
    COFD[569] = 5.01062757E+00;
    COFD[570] = -4.24032393E-01;
    COFD[571] = 1.79058125E-02;
    COFD[572] = -2.03654908E+01;
    COFD[573] = 5.03298632E+00;
    COFD[574] = -4.24771669E-01;
    COFD[575] = 1.78502612E-02;
    COFD[576] = -1.58355212E+01;
    COFD[577] = 3.60000113E+00;
    COFD[578] = -2.58255120E-01;
    COFD[579] = 1.14251480E-02;
    COFD[580] = -1.51451337E+01;
    COFD[581] = 4.03719698E+00;
    COFD[582] = -3.13407940E-01;
    COFD[583] = 1.37483092E-02;
    COFD[584] = -1.22198776E+01;
    COFD[585] = 2.70415313E+00;
    COFD[586] = -1.41236971E-01;
    COFD[587] = 6.32236816E-03;
    COFD[588] = -1.49439976E+01;
    COFD[589] = 3.43509376E+00;
    COFD[590] = -2.37713783E-01;
    COFD[591] = 1.05726006E-02;
    COFD[592] = -1.49645688E+01;
    COFD[593] = 3.43509376E+00;
    COFD[594] = -2.37713783E-01;
    COFD[595] = 1.05726006E-02;
    COFD[596] = -1.61086977E+01;
    COFD[597] = 3.70633871E+00;
    COFD[598] = -2.71897253E-01;
    COFD[599] = 1.20097588E-02;
    COFD[600] = -1.61242048E+01;
    COFD[601] = 3.70633871E+00;
    COFD[602] = -2.71897253E-01;
    COFD[603] = 1.20097588E-02;
    COFD[604] = -1.98507823E+01;
    COFD[605] = 5.07004702E+00;
    COFD[606] = -4.23605778E-01;
    COFD[607] = 1.75592300E-02;
    COFD[608] = -1.61166279E+01;
    COFD[609] = 3.70633871E+00;
    COFD[610] = -2.71897253E-01;
    COFD[611] = 1.20097588E-02;
    COFD[612] = -1.58598550E+01;
    COFD[613] = 3.60600362E+00;
    COFD[614] = -2.59019961E-01;
    COFD[615] = 1.14576923E-02;
    COFD[616] = -1.68611320E+01;
    COFD[617] = 4.03054927E+00;
    COFD[618] = -3.12591852E-01;
    COFD[619] = 1.37148695E-02;
    COFD[620] = -2.01654365E+01;
    COFD[621] = 5.05721632E+00;
    COFD[622] = -4.26359426E-01;
    COFD[623] = 1.78564586E-02;
    COFD[624] = -1.87808686E+01;
    COFD[625] = 4.61060397E+00;
    COFD[626] = -3.83564503E-01;
    COFD[627] = 1.66168246E-02;
    COFD[628] = -1.68202663E+01;
    COFD[629] = 4.01337907E+00;
    COFD[630] = -3.10488902E-01;
    COFD[631] = 1.36288975E-02;
    COFD[632] = -1.82883680E+01;
    COFD[633] = 4.45298414E+00;
    COFD[634] = -3.64713014E-01;
    COFD[635] = 1.58635108E-02;
    COFD[636] = -1.91500832E+01;
    COFD[637] = 4.75557446E+00;
    COFD[638] = -4.00539622E-01;
    COFD[639] = 1.72785461E-02;
    COFD[640] = -2.01435322E+01;
    COFD[641] = 5.01062757E+00;
    COFD[642] = -4.24032393E-01;
    COFD[643] = 1.79058125E-02;
    COFD[644] = -1.89066301E+01;
    COFD[645] = 4.63397272E+00;
    COFD[646] = -3.86072736E-01;
    COFD[647] = 1.67042917E-02;
    COFD[648] = -1.82990356E+01;
    COFD[649] = 4.45298414E+00;
    COFD[650] = -3.64713014E-01;
    COFD[651] = 1.58635108E-02;
    COFD[652] = -1.88975093E+01;
    COFD[653] = 4.63397272E+00;
    COFD[654] = -3.86072736E-01;
    COFD[655] = 1.67042917E-02;
    COFD[656] = -1.68946144E+01;
    COFD[657] = 4.08227013E+00;
    COFD[658] = -3.19188042E-01;
    COFD[659] = 1.39963401E-02;
    COFD[660] = -2.01538718E+01;
    COFD[661] = 5.01062757E+00;
    COFD[662] = -4.24032393E-01;
    COFD[663] = 1.79058125E-02;
    COFD[664] = -2.01487964E+01;
    COFD[665] = 5.01062757E+00;
    COFD[666] = -4.24032393E-01;
    COFD[667] = 1.79058125E-02;
    COFD[668] = -2.03831264E+01;
    COFD[669] = 5.03298632E+00;
    COFD[670] = -4.24771669E-01;
    COFD[671] = 1.78502612E-02;
    COFD[672] = -1.99472346E+01;
    COFD[673] = 5.05636623E+00;
    COFD[674] = -4.17733686E-01;
    COFD[675] = 1.71403506E-02;
    COFD[676] = -1.80638064E+01;
    COFD[677] = 4.96477835E+00;
    COFD[678] = -3.98489680E-01;
    COFD[679] = 1.60121621E-02;
    COFD[680] = -1.73864044E+01;
    COFD[681] = 4.71143088E+00;
    COFD[682] = -3.95288626E-01;
    COFD[683] = 1.70702272E-02;
    COFD[684] = -1.90632945E+01;
    COFD[685] = 5.00590780E+00;
    COFD[686] = -4.24055652E-01;
    COFD[687] = 1.79326322E-02;
    COFD[688] = -1.90792409E+01;
    COFD[689] = 5.00590780E+00;
    COFD[690] = -4.24055652E-01;
    COFD[691] = 1.79326322E-02;
    COFD[692] = -1.99035583E+01;
    COFD[693] = 5.01694644E+00;
    COFD[694] = -4.08963011E-01;
    COFD[695] = 1.66143416E-02;
    COFD[696] = -1.98507823E+01;
    COFD[697] = 5.07004702E+00;
    COFD[698] = -4.23605778E-01;
    COFD[699] = 1.75592300E-02;
    COFD[700] = -1.16123849E+01;
    COFD[701] = 8.27754782E-01;
    COFD[702] = 2.52262233E-01;
    COFD[703] = -1.62567414E-02;
    COFD[704] = -1.98455231E+01;
    COFD[705] = 5.07004702E+00;
    COFD[706] = -4.23605778E-01;
    COFD[707] = 1.75592300E-02;
    COFD[708] = -1.99647405E+01;
    COFD[709] = 5.05179386E+00;
    COFD[710] = -4.16351103E-01;
    COFD[711] = 1.70488551E-02;
    COFD[712] = -1.97774955E+01;
    COFD[713] = 4.96750408E+00;
    COFD[714] = -3.99127523E-01;
    COFD[715] = 1.60511657E-02;
    COFD[716] = -1.48284816E+01;
    COFD[717] = 2.35400310E+00;
    COFD[718] = 1.20736855E-02;
    COFD[719] = -4.57625832E-03;
    COFD[720] = -1.82187624E+01;
    COFD[721] = 3.93854160E+00;
    COFD[722] = -2.28424632E-01;
    COFD[723] = 7.18603342E-03;
    COFD[724] = -1.95657633E+01;
    COFD[725] = 4.78813636E+00;
    COFD[726] = -3.65976055E-01;
    COFD[727] = 1.42215137E-02;
    COFD[728] = -1.94649953E+01;
    COFD[729] = 4.60063128E+00;
    COFD[730] = -3.33961692E-01;
    COFD[731] = 1.25315265E-02;
    COFD[732] = -1.83509011E+01;
    COFD[733] = 4.02222416E+00;
    COFD[734] = -2.41671549E-01;
    COFD[735] = 7.85284958E-03;
    COFD[736] = -1.59040213E+01;
    COFD[737] = 2.78426888E+00;
    COFD[738] = -5.24195109E-02;
    COFD[739] = -1.45484926E-03;
    COFD[740] = -1.89487614E+01;
    COFD[741] = 4.27158022E+00;
    COFD[742] = -2.80784144E-01;
    COFD[743] = 9.81431834E-03;
    COFD[744] = -1.94726743E+01;
    COFD[745] = 4.60063128E+00;
    COFD[746] = -3.33961692E-01;
    COFD[747] = 1.25315265E-02;
    COFD[748] = -1.89423062E+01;
    COFD[749] = 4.27158022E+00;
    COFD[750] = -2.80784144E-01;
    COFD[751] = 9.81431834E-03;
    COFD[752] = -1.95268261E+01;
    COFD[753] = 4.94867364E+00;
    COFD[754] = -3.94985793E-01;
    COFD[755] = 1.58038735E-02;
    COFD[756] = -1.59109328E+01;
    COFD[757] = 2.78426888E+00;
    COFD[758] = -5.24195109E-02;
    COFD[759] = -1.45484926E-03;
    COFD[760] = -1.59075460E+01;
    COFD[761] = 2.78426888E+00;
    COFD[762] = -5.24195109E-02;
    COFD[763] = -1.45484926E-03;
    COFD[764] = -1.55408232E+01;
    COFD[765] = 2.54259134E+00;
    COFD[766] = -1.60736376E-02;
    COFD[767] = -3.21856796E-03;
    COFD[768] = -1.58286724E+01;
    COFD[769] = 3.60000113E+00;
    COFD[770] = -2.58255120E-01;
    COFD[771] = 1.14251480E-02;
    COFD[772] = -1.51446944E+01;
    COFD[773] = 4.03719698E+00;
    COFD[774] = -3.13407940E-01;
    COFD[775] = 1.37483092E-02;
    COFD[776] = -1.22190241E+01;
    COFD[777] = 2.70415313E+00;
    COFD[778] = -1.41236971E-01;
    COFD[779] = 6.32236816E-03;
    COFD[780] = -1.49391368E+01;
    COFD[781] = 3.43509376E+00;
    COFD[782] = -2.37713783E-01;
    COFD[783] = 1.05726006E-02;
    COFD[784] = -1.49595048E+01;
    COFD[785] = 3.43509376E+00;
    COFD[786] = -2.37713783E-01;
    COFD[787] = 1.05726006E-02;
    COFD[788] = -1.61013505E+01;
    COFD[789] = 3.70633871E+00;
    COFD[790] = -2.71897253E-01;
    COFD[791] = 1.20097588E-02;
    COFD[792] = -1.61166279E+01;
    COFD[793] = 3.70633871E+00;
    COFD[794] = -2.71897253E-01;
    COFD[795] = 1.20097588E-02;
    COFD[796] = -1.98455231E+01;
    COFD[797] = 5.07004702E+00;
    COFD[798] = -4.23605778E-01;
    COFD[799] = 1.75592300E-02;
    COFD[800] = -1.61091642E+01;
    COFD[801] = 3.70633871E+00;
    COFD[802] = -2.71897253E-01;
    COFD[803] = 1.20097588E-02;
    COFD[804] = -1.58530066E+01;
    COFD[805] = 3.60600362E+00;
    COFD[806] = -2.59019961E-01;
    COFD[807] = 1.14576923E-02;
    COFD[808] = -1.68564733E+01;
    COFD[809] = 4.03054927E+00;
    COFD[810] = -3.12591852E-01;
    COFD[811] = 1.37148695E-02;
    COFD[812] = -2.01583281E+01;
    COFD[813] = 5.05721632E+00;
    COFD[814] = -4.26359426E-01;
    COFD[815] = 1.78564586E-02;
    COFD[816] = -1.87723293E+01;
    COFD[817] = 4.61060397E+00;
    COFD[818] = -3.83564503E-01;
    COFD[819] = 1.66168246E-02;
    COFD[820] = -1.68153964E+01;
    COFD[821] = 4.01337907E+00;
    COFD[822] = -3.10488902E-01;
    COFD[823] = 1.36288975E-02;
    COFD[824] = -1.82817909E+01;
    COFD[825] = 4.45298414E+00;
    COFD[826] = -3.64713014E-01;
    COFD[827] = 1.58635108E-02;
    COFD[828] = -1.91432290E+01;
    COFD[829] = 4.75557446E+00;
    COFD[830] = -4.00539622E-01;
    COFD[831] = 1.72785461E-02;
    COFD[832] = -2.01351627E+01;
    COFD[833] = 5.01062757E+00;
    COFD[834] = -4.24032393E-01;
    COFD[835] = 1.79058125E-02;
    COFD[836] = -1.88995163E+01;
    COFD[837] = 4.63397272E+00;
    COFD[838] = -3.86072736E-01;
    COFD[839] = 1.67042917E-02;
    COFD[840] = -1.82923177E+01;
    COFD[841] = 4.45298414E+00;
    COFD[842] = -3.64713014E-01;
    COFD[843] = 1.58635108E-02;
    COFD[844] = -1.88905232E+01;
    COFD[845] = 4.63397272E+00;
    COFD[846] = -3.86072736E-01;
    COFD[847] = 1.67042917E-02;
    COFD[848] = -1.68863351E+01;
    COFD[849] = 4.08227013E+00;
    COFD[850] = -3.19188042E-01;
    COFD[851] = 1.39963401E-02;
    COFD[852] = -2.01453289E+01;
    COFD[853] = 5.01062757E+00;
    COFD[854] = -4.24032393E-01;
    COFD[855] = 1.79058125E-02;
    COFD[856] = -2.01403390E+01;
    COFD[857] = 5.01062757E+00;
    COFD[858] = -4.24032393E-01;
    COFD[859] = 1.79058125E-02;
    COFD[860] = -2.03745002E+01;
    COFD[861] = 5.03298632E+00;
    COFD[862] = -4.24771669E-01;
    COFD[863] = 1.78502612E-02;
    COFD[864] = -1.56221627E+01;
    COFD[865] = 3.51977302E+00;
    COFD[866] = -2.48210923E-01;
    COFD[867] = 1.10059241E-02;
    COFD[868] = -1.49673213E+01;
    COFD[869] = 3.95033191E+00;
    COFD[870] = -3.02754076E-01;
    COFD[871] = 1.33124608E-02;
    COFD[872] = -1.20607690E+01;
    COFD[873] = 2.61969379E+00;
    COFD[874] = -1.29638429E-01;
    COFD[875] = 5.79050588E-03;
    COFD[876] = -1.47359425E+01;
    COFD[877] = 3.34699328E+00;
    COFD[878] = -2.26393424E-01;
    COFD[879] = 1.00872697E-02;
    COFD[880] = -1.47551678E+01;
    COFD[881] = 3.34699328E+00;
    COFD[882] = -2.26393424E-01;
    COFD[883] = 1.00872697E-02;
    COFD[884] = -1.58458281E+01;
    COFD[885] = 3.60600362E+00;
    COFD[886] = -2.59019961E-01;
    COFD[887] = 1.14576923E-02;
    COFD[888] = -1.58598550E+01;
    COFD[889] = 3.60600362E+00;
    COFD[890] = -2.59019961E-01;
    COFD[891] = 1.14576923E-02;
    COFD[892] = -1.99647405E+01;
    COFD[893] = 5.05179386E+00;
    COFD[894] = -4.16351103E-01;
    COFD[895] = 1.70488551E-02;
    COFD[896] = -1.58530066E+01;
    COFD[897] = 3.60600362E+00;
    COFD[898] = -2.59019961E-01;
    COFD[899] = 1.14576923E-02;
    COFD[900] = -1.56423580E+01;
    COFD[901] = 3.52412711E+00;
    COFD[902] = -2.48745351E-01;
    COFD[903] = 1.10277551E-02;
    COFD[904] = -1.66378056E+01;
    COFD[905] = 3.94349539E+00;
    COFD[906] = -3.01913683E-01;
    COFD[907] = 1.32780377E-02;
    COFD[908] = -2.00109539E+01;
    COFD[909] = 5.01818529E+00;
    COFD[910] = -4.23776772E-01;
    COFD[911] = 1.78445623E-02;
    COFD[912] = -1.85324360E+01;
    COFD[913] = 4.52748688E+00;
    COFD[914] = -3.73847542E-01;
    COFD[915] = 1.62384117E-02;
    COFD[916] = -1.65940140E+01;
    COFD[917] = 3.92553905E+00;
    COFD[918] = -2.99706984E-01;
    COFD[919] = 1.31876655E-02;
    COFD[920] = -1.79617008E+01;
    COFD[921] = 4.33127484E+00;
    COFD[922] = -3.49477255E-01;
    COFD[923] = 1.52253055E-02;
    COFD[924] = -1.88433165E+01;
    COFD[925] = 4.64610663E+00;
    COFD[926] = -3.87397753E-01;
    COFD[927] = 1.67516180E-02;
    COFD[928] = -2.00005226E+01;
    COFD[929] = 4.97925805E+00;
    COFD[930] = -4.22339984E-01;
    COFD[931] = 1.79289294E-02;
    COFD[932] = -1.86872650E+01;
    COFD[933] = 4.56143198E+00;
    COFD[934] = -3.77911104E-01;
    COFD[935] = 1.64009932E-02;
    COFD[936] = -1.79714524E+01;
    COFD[937] = 4.33127484E+00;
    COFD[938] = -3.49477255E-01;
    COFD[939] = 1.52253055E-02;
    COFD[940] = -1.86789709E+01;
    COFD[941] = 4.56143198E+00;
    COFD[942] = -3.77911104E-01;
    COFD[943] = 1.64009932E-02;
    COFD[944] = -1.66322681E+01;
    COFD[945] = 3.98376384E+00;
    COFD[946] = -3.06871085E-01;
    COFD[947] = 1.34815439E-02;
    COFD[948] = -2.00097568E+01;
    COFD[949] = 4.97925805E+00;
    COFD[950] = -4.22339984E-01;
    COFD[951] = 1.79289294E-02;
    COFD[952] = -2.00052265E+01;
    COFD[953] = 4.97925805E+00;
    COFD[954] = -4.22339984E-01;
    COFD[955] = 1.79289294E-02;
    COFD[956] = -2.02493305E+01;
    COFD[957] = 5.00861852E+00;
    COFD[958] = -4.24191458E-01;
    COFD[959] = 1.79297653E-02;
    COFD[960] = -1.66140265E+01;
    COFD[961] = 3.93776813E+00;
    COFD[962] = -3.01210554E-01;
    COFD[963] = 1.32492810E-02;
    COFD[964] = -1.61384688E+01;
    COFD[965] = 4.35660780E+00;
    COFD[966] = -3.52659055E-01;
    COFD[967] = 1.53589488E-02;
    COFD[968] = -1.30102341E+01;
    COFD[969] = 2.98419920E+00;
    COFD[970] = -1.78783571E-01;
    COFD[971] = 8.00253963E-03;
    COFD[972] = -1.56185416E+01;
    COFD[973] = 3.70484350E+00;
    COFD[974] = -2.71706480E-01;
    COFD[975] = 1.20016288E-02;
    COFD[976] = -1.56331080E+01;
    COFD[977] = 3.70484350E+00;
    COFD[978] = -2.71706480E-01;
    COFD[979] = 1.20016288E-02;
    COFD[980] = -1.68515683E+01;
    COFD[981] = 4.03054927E+00;
    COFD[982] = -3.12591852E-01;
    COFD[983] = 1.37148695E-02;
    COFD[984] = -1.68611320E+01;
    COFD[985] = 4.03054927E+00;
    COFD[986] = -3.12591852E-01;
    COFD[987] = 1.37148695E-02;
    COFD[988] = -1.97774955E+01;
    COFD[989] = 4.96750408E+00;
    COFD[990] = -3.99127523E-01;
    COFD[991] = 1.60511657E-02;
    COFD[992] = -1.68564733E+01;
    COFD[993] = 4.03054927E+00;
    COFD[994] = -3.12591852E-01;
    COFD[995] = 1.37148695E-02;
    COFD[996] = -1.66378056E+01;
    COFD[997] = 3.94349539E+00;
    COFD[998] = -3.01913683E-01;
    COFD[999] = 1.32780377E-02;
    COFD[1000] = -1.76728581E+01;
    COFD[1001] = 4.34699543E+00;
    COFD[1002] = -3.51450562E-01;
    COFD[1003] = 1.53081221E-02;
    COFD[1004] = -2.02381830E+01;
    COFD[1005] = 5.04826260E+00;
    COFD[1006] = -4.15332602E-01;
    COFD[1007] = 1.69822815E-02;
    COFD[1008] = -1.94233234E+01;
    COFD[1009] = 4.88315812E+00;
    COFD[1010] = -4.14471408E-01;
    COFD[1011] = 1.77754868E-02;
    COFD[1012] = -1.76080457E+01;
    COFD[1013] = 4.32268865E+00;
    COFD[1014] = -3.48416561E-01;
    COFD[1015] = 1.51815419E-02;
    COFD[1016] = -1.90274201E+01;
    COFD[1017] = 4.75331782E+00;
    COFD[1018] = -4.00274636E-01;
    COFD[1019] = 1.72681883E-02;
    COFD[1020] = -1.96663788E+01;
    COFD[1021] = 4.95183665E+00;
    COFD[1022] = -4.20101552E-01;
    COFD[1023] = 1.78850963E-02;
    COFD[1024] = -2.03270738E+01;
    COFD[1025] = 5.06757610E+00;
    COFD[1026] = -4.22549988E-01;
    COFD[1027] = 1.74838847E-02;
    COFD[1028] = -1.95590709E+01;
    COFD[1029] = 4.89649396E+00;
    COFD[1030] = -4.15388755E-01;
    COFD[1031] = 1.77817581E-02;
    COFD[1032] = -1.90342882E+01;
    COFD[1033] = 4.75331782E+00;
    COFD[1034] = -4.00274636E-01;
    COFD[1035] = 1.72681883E-02;
    COFD[1036] = -1.95533235E+01;
    COFD[1037] = 4.89649396E+00;
    COFD[1038] = -4.15388755E-01;
    COFD[1039] = 1.77817581E-02;
    COFD[1040] = -1.76407601E+01;
    COFD[1041] = 4.40309399E+00;
    COFD[1042] = -3.58496732E-01;
    COFD[1043] = 1.56042461E-02;
    COFD[1044] = -2.03331380E+01;
    COFD[1045] = 5.06757610E+00;
    COFD[1046] = -4.22549988E-01;
    COFD[1047] = 1.74838847E-02;
    COFD[1048] = -2.03301677E+01;
    COFD[1049] = 5.06757610E+00;
    COFD[1050] = -4.22549988E-01;
    COFD[1051] = 1.74838847E-02;
    COFD[1052] = -2.04891102E+01;
    COFD[1053] = 5.05959508E+00;
    COFD[1054] = -4.18821304E-01;
    COFD[1055] = 1.72141752E-02;
    COFD[1056] = -1.99958660E+01;
    COFD[1057] = 5.01699515E+00;
    COFD[1058] = -4.23791128E-01;
    COFD[1059] = 1.78520267E-02;
    COFD[1060] = -1.85337299E+01;
    COFD[1061] = 5.04618677E+00;
    COFD[1062] = -4.14801122E-01;
    COFD[1063] = 1.69487185E-02;
    COFD[1064] = -1.64253252E+01;
    COFD[1065] = 4.26219537E+00;
    COFD[1066] = -3.41524254E-01;
    COFD[1067] = 1.49232070E-02;
    COFD[1068] = -1.92562118E+01;
    COFD[1069] = 4.94108508E+00;
    COFD[1070] = -4.19008197E-01;
    COFD[1071] = 1.78499457E-02;
    COFD[1072] = -1.92759277E+01;
    COFD[1073] = 4.94108508E+00;
    COFD[1074] = -4.19008197E-01;
    COFD[1075] = 1.78499457E-02;
    COFD[1076] = -2.01508812E+01;
    COFD[1077] = 5.05721632E+00;
    COFD[1078] = -4.26359426E-01;
    COFD[1079] = 1.78564586E-02;
    COFD[1080] = -2.01654365E+01;
    COFD[1081] = 5.05721632E+00;
    COFD[1082] = -4.26359426E-01;
    COFD[1083] = 1.78564586E-02;
    COFD[1084] = -1.48284816E+01;
    COFD[1085] = 2.35400310E+00;
    COFD[1086] = 1.20736855E-02;
    COFD[1087] = -4.57625832E-03;
    COFD[1088] = -2.01583281E+01;
    COFD[1089] = 5.05721632E+00;
    COFD[1090] = -4.26359426E-01;
    COFD[1091] = 1.78564586E-02;
    COFD[1092] = -2.00109539E+01;
    COFD[1093] = 5.01818529E+00;
    COFD[1094] = -4.23776772E-01;
    COFD[1095] = 1.78445623E-02;
    COFD[1096] = -2.02381830E+01;
    COFD[1097] = 5.04826260E+00;
    COFD[1098] = -4.15332602E-01;
    COFD[1099] = 1.69822815E-02;
    COFD[1100] = -1.62108779E+01;
    COFD[1101] = 2.80507926E+00;
    COFD[1102] = -5.55394339E-02;
    COFD[1103] = -1.30364179E-03;
    COFD[1104] = -1.98663762E+01;
    COFD[1105] = 4.57685026E+00;
    COFD[1106] = -3.30016794E-01;
    COFD[1107] = 1.23264865E-02;
    COFD[1108] = -2.02400513E+01;
    COFD[1109] = 5.05240792E+00;
    COFD[1110] = -4.16528519E-01;
    COFD[1111] = 1.70604570E-02;
    COFD[1112] = -2.01538194E+01;
    COFD[1113] = 4.77237985E+00;
    COFD[1114] = -3.63201359E-01;
    COFD[1115] = 1.40726267E-02;
    COFD[1116] = -1.93849308E+01;
    COFD[1117] = 4.33884875E+00;
    COFD[1118] = -2.91425625E-01;
    COFD[1119] = 1.03508068E-02;
    COFD[1120] = -1.73079550E+01;
    COFD[1121] = 3.24454040E+00;
    COFD[1122] = -1.21683836E-01;
    COFD[1123] = 1.91319147E-03;
    COFD[1124] = -1.98270365E+01;
    COFD[1125] = 4.52357870E+00;
    COFD[1126] = -3.21272099E-01;
    COFD[1127] = 1.18751756E-02;
    COFD[1128] = -2.01639003E+01;
    COFD[1129] = 4.77237985E+00;
    COFD[1130] = -3.63201359E-01;
    COFD[1131] = 1.40726267E-02;
    COFD[1132] = -1.98184462E+01;
    COFD[1133] = 4.52357870E+00;
    COFD[1134] = -3.21272099E-01;
    COFD[1135] = 1.18751756E-02;
    COFD[1136] = -2.01085993E+01;
    COFD[1137] = 5.02691790E+00;
    COFD[1138] = -4.10940210E-01;
    COFD[1139] = 1.67272564E-02;
    COFD[1140] = -1.73175806E+01;
    COFD[1141] = 3.24454040E+00;
    COFD[1142] = -1.21683836E-01;
    COFD[1143] = 1.91319147E-03;
    COFD[1144] = -1.73128574E+01;
    COFD[1145] = 3.24454040E+00;
    COFD[1146] = -1.21683836E-01;
    COFD[1147] = 1.91319147E-03;
    COFD[1148] = -1.69183106E+01;
    COFD[1149] = 2.99595407E+00;
    COFD[1150] = -8.41851353E-02;
    COFD[1151] = 8.62518660E-05;
    COFD[1152] = -1.85068873E+01;
    COFD[1153] = 4.52122572E+00;
    COFD[1154] = -3.73088946E-01;
    COFD[1155] = 1.62076520E-02;
    COFD[1156] = -1.76943558E+01;
    COFD[1157] = 4.88672714E+00;
    COFD[1158] = -4.14778255E-01;
    COFD[1159] = 1.77823474E-02;
    COFD[1160] = -1.43978662E+01;
    COFD[1161] = 3.49721576E+00;
    COFD[1162] = -2.45465191E-01;
    COFD[1163] = 1.08948372E-02;
    COFD[1164] = -1.74119025E+01;
    COFD[1165] = 4.28601449E+00;
    COFD[1166] = -3.44182880E-01;
    COFD[1167] = 1.50201783E-02;
    COFD[1168] = -1.74341216E+01;
    COFD[1169] = 4.28601449E+00;
    COFD[1170] = -3.44182880E-01;
    COFD[1171] = 1.50201783E-02;
    COFD[1172] = -1.87634092E+01;
    COFD[1173] = 4.61060397E+00;
    COFD[1174] = -3.83564503E-01;
    COFD[1175] = 1.66168246E-02;
    COFD[1176] = -1.87808686E+01;
    COFD[1177] = 4.61060397E+00;
    COFD[1178] = -3.83564503E-01;
    COFD[1179] = 1.66168246E-02;
    COFD[1180] = -1.82187624E+01;
    COFD[1181] = 3.93854160E+00;
    COFD[1182] = -2.28424632E-01;
    COFD[1183] = 7.18603342E-03;
    COFD[1184] = -1.87723293E+01;
    COFD[1185] = 4.61060397E+00;
    COFD[1186] = -3.83564503E-01;
    COFD[1187] = 1.66168246E-02;
    COFD[1188] = -1.85324360E+01;
    COFD[1189] = 4.52748688E+00;
    COFD[1190] = -3.73847542E-01;
    COFD[1191] = 1.62384117E-02;
    COFD[1192] = -1.94233234E+01;
    COFD[1193] = 4.88315812E+00;
    COFD[1194] = -4.14471408E-01;
    COFD[1195] = 1.77754868E-02;
    COFD[1196] = -1.98663762E+01;
    COFD[1197] = 4.57685026E+00;
    COFD[1198] = -3.30016794E-01;
    COFD[1199] = 1.23264865E-02;
    COFD[1200] = -2.05810669E+01;
    COFD[1201] = 5.07469434E+00;
    COFD[1202] = -4.25340301E-01;
    COFD[1203] = 1.76800795E-02;
    COFD[1204] = -1.93937264E+01;
    COFD[1205] = 4.87146645E+00;
    COFD[1206] = -4.13323360E-01;
    COFD[1207] = 1.77408400E-02;
    COFD[1208] = -2.03135529E+01;
    COFD[1209] = 5.03708859E+00;
    COFD[1210] = -4.25057078E-01;
    COFD[1211] = 1.78526666E-02;
    COFD[1212] = -2.05833353E+01;
    COFD[1213] = 5.05774582E+00;
    COFD[1214] = -4.18194338E-01;
    COFD[1215] = 1.71715531E-02;
    COFD[1216] = -2.03451319E+01;
    COFD[1217] = 4.74915125E+00;
    COFD[1218] = -3.59194781E-01;
    COFD[1219] = 1.38602188E-02;
    COFD[1220] = -2.06302315E+01;
    COFD[1221] = 5.06978417E+00;
    COFD[1222] = -4.23521964E-01;
    COFD[1223] = 1.75535927E-02;
    COFD[1224] = -2.03253997E+01;
    COFD[1225] = 5.03708859E+00;
    COFD[1226] = -4.25057078E-01;
    COFD[1227] = 1.78526666E-02;
    COFD[1228] = -2.06200338E+01;
    COFD[1229] = 5.06978417E+00;
    COFD[1230] = -4.23521964E-01;
    COFD[1231] = 1.75535927E-02;
    COFD[1232] = -1.94319186E+01;
    COFD[1233] = 4.89964403E+00;
    COFD[1234] = -4.15618593E-01;
    COFD[1235] = 1.77843657E-02;
    COFD[1236] = -2.03569735E+01;
    COFD[1237] = 4.74915125E+00;
    COFD[1238] = -3.59194781E-01;
    COFD[1239] = 1.38602188E-02;
    COFD[1240] = -2.03511563E+01;
    COFD[1241] = 4.74915125E+00;
    COFD[1242] = -3.59194781E-01;
    COFD[1243] = 1.38602188E-02;
    COFD[1244] = -2.03360102E+01;
    COFD[1245] = 4.66227065E+00;
    COFD[1246] = -3.44237793E-01;
    COFD[1247] = 1.30673058E-02;
    COFD[1248] = -1.65706884E+01;
    COFD[1249] = 3.92005093E+00;
    COFD[1250] = -2.99040611E-01;
    COFD[1251] = 1.31607610E-02;
    COFD[1252] = -1.60530416E+01;
    COFD[1253] = 4.33128165E+00;
    COFD[1254] = -3.49478110E-01;
    COFD[1255] = 1.52253415E-02;
    COFD[1256] = -1.29544834E+01;
    COFD[1257] = 2.96758239E+00;
    COFD[1258] = -1.76586224E-01;
    COFD[1259] = 7.90559536E-03;
    COFD[1260] = -1.55602858E+01;
    COFD[1261] = 3.68289956E+00;
    COFD[1262] = -2.68870371E-01;
    COFD[1263] = 1.18791438E-02;
    COFD[1264] = -1.55753472E+01;
    COFD[1265] = 3.68289956E+00;
    COFD[1266] = -2.68870371E-01;
    COFD[1267] = 1.18791438E-02;
    COFD[1268] = -1.68102713E+01;
    COFD[1269] = 4.01337907E+00;
    COFD[1270] = -3.10488902E-01;
    COFD[1271] = 1.36288975E-02;
    COFD[1272] = -1.68202663E+01;
    COFD[1273] = 4.01337907E+00;
    COFD[1274] = -3.10488902E-01;
    COFD[1275] = 1.36288975E-02;
    COFD[1276] = -1.95657633E+01;
    COFD[1277] = 4.78813636E+00;
    COFD[1278] = -3.65976055E-01;
    COFD[1279] = 1.42215137E-02;
    COFD[1280] = -1.68153964E+01;
    COFD[1281] = 4.01337907E+00;
    COFD[1282] = -3.10488902E-01;
    COFD[1283] = 1.36288975E-02;
    COFD[1284] = -1.65940140E+01;
    COFD[1285] = 3.92553905E+00;
    COFD[1286] = -2.99706984E-01;
    COFD[1287] = 1.31876655E-02;
    COFD[1288] = -1.76080457E+01;
    COFD[1289] = 4.32268865E+00;
    COFD[1290] = -3.48416561E-01;
    COFD[1291] = 1.51815419E-02;
    COFD[1292] = -2.02400513E+01;
    COFD[1293] = 5.05240792E+00;
    COFD[1294] = -4.16528519E-01;
    COFD[1295] = 1.70604570E-02;
    COFD[1296] = -1.93937264E+01;
    COFD[1297] = 4.87146645E+00;
    COFD[1298] = -4.13323360E-01;
    COFD[1299] = 1.77408400E-02;
    COFD[1300] = -1.75618457E+01;
    COFD[1301] = 4.30617914E+00;
    COFD[1302] = -3.46490389E-01;
    COFD[1303] = 1.51071405E-02;
    COFD[1304] = -1.89744858E+01;
    COFD[1305] = 4.73272421E+00;
    COFD[1306] = -3.97843585E-01;
    COFD[1307] = 1.71725994E-02;
    COFD[1308] = -1.96315733E+01;
    COFD[1309] = 4.93927762E+00;
    COFD[1310] = -4.18850242E-01;
    COFD[1311] = 1.78462242E-02;
    COFD[1312] = -2.03268718E+01;
    COFD[1313] = 5.06951156E+00;
    COFD[1314] = -4.23434786E-01;
    COFD[1315] = 1.75477252E-02;
    COFD[1316] = -1.95390986E+01;
    COFD[1317] = 4.88982264E+00;
    COFD[1318] = -4.14974963E-01;
    COFD[1319] = 1.77824366E-02;
    COFD[1320] = -1.89816408E+01;
    COFD[1321] = 4.73272421E+00;
    COFD[1322] = -3.97843585E-01;
    COFD[1323] = 1.71725994E-02;
    COFD[1324] = -1.95331014E+01;
    COFD[1325] = 4.88982264E+00;
    COFD[1326] = -4.14974963E-01;
    COFD[1327] = 1.77824366E-02;
    COFD[1328] = -1.75794891E+01;
    COFD[1329] = 4.37827292E+00;
    COFD[1330] = -3.55380310E-01;
    COFD[1331] = 1.54732833E-02;
    COFD[1332] = -2.03332321E+01;
    COFD[1333] = 5.06951156E+00;
    COFD[1334] = -4.23434786E-01;
    COFD[1335] = 1.75477252E-02;
    COFD[1336] = -2.03301163E+01;
    COFD[1337] = 5.06951156E+00;
    COFD[1338] = -4.23434786E-01;
    COFD[1339] = 1.75477252E-02;
    COFD[1340] = -2.04918650E+01;
    COFD[1341] = 5.06231507E+00;
    COFD[1342] = -4.19815596E-01;
    COFD[1343] = 1.72829136E-02;
    COFD[1344] = -1.79334289E+01;
    COFD[1345] = 4.32370416E+00;
    COFD[1346] = -3.48535839E-01;
    COFD[1347] = 1.51861878E-02;
    COFD[1348] = -1.74064914E+01;
    COFD[1349] = 4.76097419E+00;
    COFD[1350] = -4.01175756E-01;
    COFD[1351] = 1.73035102E-02;
    COFD[1352] = -1.41052578E+01;
    COFD[1353] = 3.35888991E+00;
    COFD[1354] = -2.27939148E-01;
    COFD[1355] = 1.01542945E-02;
    COFD[1356] = -1.69909812E+01;
    COFD[1357] = 4.13179721E+00;
    COFD[1358] = -3.25544993E-01;
    COFD[1359] = 1.42694195E-02;
    COFD[1360] = -1.70096816E+01;
    COFD[1361] = 4.13179721E+00;
    COFD[1362] = -3.25544993E-01;
    COFD[1363] = 1.42694195E-02;
    COFD[1364] = -1.82748931E+01;
    COFD[1365] = 4.45298414E+00;
    COFD[1366] = -3.64713014E-01;
    COFD[1367] = 1.58635108E-02;
    COFD[1368] = -1.82883680E+01;
    COFD[1369] = 4.45298414E+00;
    COFD[1370] = -3.64713014E-01;
    COFD[1371] = 1.58635108E-02;
    COFD[1372] = -1.94649953E+01;
    COFD[1373] = 4.60063128E+00;
    COFD[1374] = -3.33961692E-01;
    COFD[1375] = 1.25315265E-02;
    COFD[1376] = -1.82817909E+01;
    COFD[1377] = 4.45298414E+00;
    COFD[1378] = -3.64713014E-01;
    COFD[1379] = 1.58635108E-02;
    COFD[1380] = -1.79617008E+01;
    COFD[1381] = 4.33127484E+00;
    COFD[1382] = -3.49477255E-01;
    COFD[1383] = 1.52253055E-02;
    COFD[1384] = -1.90274201E+01;
    COFD[1385] = 4.75331782E+00;
    COFD[1386] = -4.00274636E-01;
    COFD[1387] = 1.72681883E-02;
    COFD[1388] = -2.01538194E+01;
    COFD[1389] = 4.77237985E+00;
    COFD[1390] = -3.63201359E-01;
    COFD[1391] = 1.40726267E-02;
    COFD[1392] = -2.03135529E+01;
    COFD[1393] = 5.03708859E+00;
    COFD[1394] = -4.25057078E-01;
    COFD[1395] = 1.78526666E-02;
    COFD[1396] = -1.89744858E+01;
    COFD[1397] = 4.73272421E+00;
    COFD[1398] = -3.97843585E-01;
    COFD[1399] = 1.71725994E-02;
    COFD[1400] = -2.00558102E+01;
    COFD[1401] = 4.98853197E+00;
    COFD[1402] = -4.23032272E-01;
    COFD[1403] = 1.79382596E-02;
    COFD[1404] = -2.04691752E+01;
    COFD[1405] = 5.07632177E+00;
    COFD[1406] = -4.26053731E-01;
    COFD[1407] = 1.77312885E-02;
    COFD[1408] = -2.05037743E+01;
    COFD[1409] = 4.89869111E+00;
    COFD[1410] = -3.85699208E-01;
    COFD[1411] = 1.52891592E-02;
    COFD[1412] = -2.04259672E+01;
    COFD[1413] = 5.05147258E+00;
    COFD[1414] = -4.25999526E-01;
    COFD[1415] = 1.78562806E-02;
    COFD[1416] = -2.00652152E+01;
    COFD[1417] = 4.98853197E+00;
    COFD[1418] = -4.23032272E-01;
    COFD[1419] = 1.79382596E-02;
    COFD[1420] = -2.04179835E+01;
    COFD[1421] = 5.05147258E+00;
    COFD[1422] = -4.25999526E-01;
    COFD[1423] = 1.78562806E-02;
    COFD[1424] = -1.90489173E+01;
    COFD[1425] = 4.79559716E+00;
    COFD[1426] = -4.05182028E-01;
    COFD[1427] = 1.74574754E-02;
    COFD[1428] = -2.05126034E+01;
    COFD[1429] = 4.89869111E+00;
    COFD[1430] = -3.85699208E-01;
    COFD[1431] = 1.52891592E-02;
    COFD[1432] = -2.05082727E+01;
    COFD[1433] = 4.89869111E+00;
    COFD[1434] = -3.85699208E-01;
    COFD[1435] = 1.52891592E-02;
    COFD[1436] = -2.05198084E+01;
    COFD[1437] = 4.82637944E+00;
    COFD[1438] = -3.72748017E-01;
    COFD[1439] = 1.45861475E-02;
    COFD[1440] = -1.88223266E+01;
    COFD[1441] = 4.64181074E+00;
    COFD[1442] = -3.86929312E-01;
    COFD[1443] = 1.67349243E-02;
    COFD[1444] = -1.80153619E+01;
    COFD[1445] = 4.95555480E+00;
    COFD[1446] = -4.20418912E-01;
    COFD[1447] = 1.78921168E-02;
    COFD[1448] = -1.47832812E+01;
    COFD[1449] = 3.61918603E+00;
    COFD[1450] = -2.60701094E-01;
    COFD[1451] = 1.15292870E-02;
    COFD[1452] = -1.78714250E+01;
    COFD[1453] = 4.45397294E+00;
    COFD[1454] = -3.64836010E-01;
    COFD[1455] = 1.58686336E-02;
    COFD[1456] = -1.78906614E+01;
    COFD[1457] = 4.45397294E+00;
    COFD[1458] = -3.64836010E-01;
    COFD[1459] = 1.58686336E-02;
    COFD[1460] = -1.91360445E+01;
    COFD[1461] = 4.75557446E+00;
    COFD[1462] = -4.00539622E-01;
    COFD[1463] = 1.72785461E-02;
    COFD[1464] = -1.91500832E+01;
    COFD[1465] = 4.75557446E+00;
    COFD[1466] = -4.00539622E-01;
    COFD[1467] = 1.72785461E-02;
    COFD[1468] = -1.83509011E+01;
    COFD[1469] = 4.02222416E+00;
    COFD[1470] = -2.41671549E-01;
    COFD[1471] = 7.85284958E-03;
    COFD[1472] = -1.91432290E+01;
    COFD[1473] = 4.75557446E+00;
    COFD[1474] = -4.00539622E-01;
    COFD[1475] = 1.72785461E-02;
    COFD[1476] = -1.88433165E+01;
    COFD[1477] = 4.64610663E+00;
    COFD[1478] = -3.87397753E-01;
    COFD[1479] = 1.67516180E-02;
    COFD[1480] = -1.96663788E+01;
    COFD[1481] = 4.95183665E+00;
    COFD[1482] = -4.20101552E-01;
    COFD[1483] = 1.78850963E-02;
    COFD[1484] = -1.93849308E+01;
    COFD[1485] = 4.33884875E+00;
    COFD[1486] = -2.91425625E-01;
    COFD[1487] = 1.03508068E-02;
    COFD[1488] = -2.05833353E+01;
    COFD[1489] = 5.05774582E+00;
    COFD[1490] = -4.18194338E-01;
    COFD[1491] = 1.71715531E-02;
    COFD[1492] = -1.96315733E+01;
    COFD[1493] = 4.93927762E+00;
    COFD[1494] = -4.18850242E-01;
    COFD[1495] = 1.78462242E-02;
    COFD[1496] = -2.04691752E+01;
    COFD[1497] = 5.07632177E+00;
    COFD[1498] = -4.26053731E-01;
    COFD[1499] = 1.77312885E-02;
    COFD[1500] = -2.04987792E+01;
    COFD[1501] = 4.99303150E+00;
    COFD[1502] = -4.04298713E-01;
    COFD[1503] = 1.63496647E-02;
    COFD[1504] = -1.99701893E+01;
    COFD[1505] = 4.56517018E+00;
    COFD[1506] = -3.28088445E-01;
    COFD[1507] = 1.22265811E-02;
    COFD[1508] = -2.06459911E+01;
    COFD[1509] = 5.05102272E+00;
    COFD[1510] = -4.16130187E-01;
    COFD[1511] = 1.70344431E-02;
    COFD[1512] = -2.04789341E+01;
    COFD[1513] = 5.07632177E+00;
    COFD[1514] = -4.26053731E-01;
    COFD[1515] = 1.77312885E-02;
    COFD[1516] = -2.06376903E+01;
    COFD[1517] = 5.05102272E+00;
    COFD[1518] = -4.16130187E-01;
    COFD[1519] = 1.70344431E-02;
    COFD[1520] = -1.96421788E+01;
    COFD[1521] = 4.97225057E+00;
    COFD[1522] = -4.21797234E-01;
    COFD[1523] = 1.79201792E-02;
    COFD[1524] = -1.99794321E+01;
    COFD[1525] = 4.56517018E+00;
    COFD[1526] = -3.28088445E-01;
    COFD[1527] = 1.22265811E-02;
    COFD[1528] = -1.99748976E+01;
    COFD[1529] = 4.56517018E+00;
    COFD[1530] = -3.28088445E-01;
    COFD[1531] = 1.22265811E-02;
    COFD[1532] = -1.98664135E+01;
    COFD[1533] = 4.44007006E+00;
    COFD[1534] = -3.07688868E-01;
    COFD[1535] = 1.11784091E-02;
    COFD[1536] = -1.99828895E+01;
    COFD[1537] = 4.97669986E+00;
    COFD[1538] = -4.22149407E-01;
    COFD[1539] = 1.79263822E-02;
    COFD[1540] = -1.86132026E+01;
    COFD[1541] = 5.06806259E+00;
    COFD[1542] = -4.22384317E-01;
    COFD[1543] = 1.74675251E-02;
    COFD[1544] = -1.61505677E+01;
    COFD[1545] = 4.11963986E+00;
    COFD[1546] = -3.23983555E-01;
    COFD[1547] = 1.42022890E-02;
    COFD[1548] = -1.91880072E+01;
    COFD[1549] = 4.87847769E+00;
    COFD[1550] = -4.14012329E-01;
    COFD[1551] = 1.77616586E-02;
    COFD[1552] = -1.92099456E+01;
    COFD[1553] = 4.87847769E+00;
    COFD[1554] = -4.14012329E-01;
    COFD[1555] = 1.77616586E-02;
    COFD[1556] = -2.01264169E+01;
    COFD[1557] = 5.01062757E+00;
    COFD[1558] = -4.24032393E-01;
    COFD[1559] = 1.79058125E-02;
    COFD[1560] = -2.01435322E+01;
    COFD[1561] = 5.01062757E+00;
    COFD[1562] = -4.24032393E-01;
    COFD[1563] = 1.79058125E-02;
    COFD[1564] = -1.59040213E+01;
    COFD[1565] = 2.78426888E+00;
    COFD[1566] = -5.24195109E-02;
    COFD[1567] = -1.45484926E-03;
    COFD[1568] = -2.01351627E+01;
    COFD[1569] = 5.01062757E+00;
    COFD[1570] = -4.24032393E-01;
    COFD[1571] = 1.79058125E-02;
    COFD[1572] = -2.00005226E+01;
    COFD[1573] = 4.97925805E+00;
    COFD[1574] = -4.22339984E-01;
    COFD[1575] = 1.79289294E-02;
    COFD[1576] = -2.03270738E+01;
    COFD[1577] = 5.06757610E+00;
    COFD[1578] = -4.22549988E-01;
    COFD[1579] = 1.74838847E-02;
    COFD[1580] = -1.73079550E+01;
    COFD[1581] = 3.24454040E+00;
    COFD[1582] = -1.21683836E-01;
    COFD[1583] = 1.91319147E-03;
    COFD[1584] = -2.03451319E+01;
    COFD[1585] = 4.74915125E+00;
    COFD[1586] = -3.59194781E-01;
    COFD[1587] = 1.38602188E-02;
    COFD[1588] = -2.03268718E+01;
    COFD[1589] = 5.06951156E+00;
    COFD[1590] = -4.23434786E-01;
    COFD[1591] = 1.75477252E-02;
    COFD[1592] = -2.05037743E+01;
    COFD[1593] = 4.89869111E+00;
    COFD[1594] = -3.85699208E-01;
    COFD[1595] = 1.52891592E-02;
    COFD[1596] = -1.99701893E+01;
    COFD[1597] = 4.56517018E+00;
    COFD[1598] = -3.28088445E-01;
    COFD[1599] = 1.22265811E-02;
    COFD[1600] = -1.82961153E+01;
    COFD[1601] = 3.63551893E+00;
    COFD[1602] = -1.81306348E-01;
    COFD[1603] = 4.84147153E-03;
    COFD[1604] = -2.03217812E+01;
    COFD[1605] = 4.71342614E+00;
    COFD[1606] = -3.53042983E-01;
    COFD[1607] = 1.35344523E-02;
    COFD[1608] = -2.05154157E+01;
    COFD[1609] = 4.89869111E+00;
    COFD[1610] = -3.85699208E-01;
    COFD[1611] = 1.52891592E-02;
    COFD[1612] = -2.03117722E+01;
    COFD[1613] = 4.71342614E+00;
    COFD[1614] = -3.53042983E-01;
    COFD[1615] = 1.35344523E-02;
    COFD[1616] = -2.03011295E+01;
    COFD[1617] = 5.06448019E+00;
    COFD[1618] = -4.20705675E-01;
    COFD[1619] = 1.73458737E-02;
    COFD[1620] = -1.83076885E+01;
    COFD[1621] = 3.63551893E+00;
    COFD[1622] = -1.81306348E-01;
    COFD[1623] = 4.84147153E-03;
    COFD[1624] = -1.83020039E+01;
    COFD[1625] = 3.63551893E+00;
    COFD[1626] = -1.81306348E-01;
    COFD[1627] = 4.84147153E-03;
    COFD[1628] = -1.79761233E+01;
    COFD[1629] = 3.41675012E+00;
    COFD[1630] = -1.47848340E-01;
    COFD[1631] = 3.19467888E-03;
    COFD[1632] = -1.86644793E+01;
    COFD[1633] = 4.55615891E+00;
    COFD[1634] = -3.77295769E-01;
    COFD[1635] = 1.63770696E-02;
    COFD[1636] = -1.79190223E+01;
    COFD[1637] = 4.89943438E+00;
    COFD[1638] = -4.15602392E-01;
    COFD[1639] = 1.77841166E-02;
    COFD[1640] = -1.46113269E+01;
    COFD[1641] = 3.52131229E+00;
    COFD[1642] = -2.48399935E-01;
    COFD[1643] = 1.10136492E-02;
    COFD[1644] = -1.75974956E+01;
    COFD[1645] = 4.31462284E+00;
    COFD[1646] = -3.47471819E-01;
    COFD[1647] = 1.51448695E-02;
    COFD[1648] = -1.76172217E+01;
    COFD[1649] = 4.31462284E+00;
    COFD[1650] = -3.47471819E-01;
    COFD[1651] = 1.51448695E-02;
    COFD[1652] = -1.88920637E+01;
    COFD[1653] = 4.63397272E+00;
    COFD[1654] = -3.86072736E-01;
    COFD[1655] = 1.67042917E-02;
    COFD[1656] = -1.89066301E+01;
    COFD[1657] = 4.63397272E+00;
    COFD[1658] = -3.86072736E-01;
    COFD[1659] = 1.67042917E-02;
    COFD[1660] = -1.89487614E+01;
    COFD[1661] = 4.27158022E+00;
    COFD[1662] = -2.80784144E-01;
    COFD[1663] = 9.81431834E-03;
    COFD[1664] = -1.88995163E+01;
    COFD[1665] = 4.63397272E+00;
    COFD[1666] = -3.86072736E-01;
    COFD[1667] = 1.67042917E-02;
    COFD[1668] = -1.86872650E+01;
    COFD[1669] = 4.56143198E+00;
    COFD[1670] = -3.77911104E-01;
    COFD[1671] = 1.64009932E-02;
    COFD[1672] = -1.95590709E+01;
    COFD[1673] = 4.89649396E+00;
    COFD[1674] = -4.15388755E-01;
    COFD[1675] = 1.77817581E-02;
    COFD[1676] = -1.98270365E+01;
    COFD[1677] = 4.52357870E+00;
    COFD[1678] = -3.21272099E-01;
    COFD[1679] = 1.18751756E-02;
    COFD[1680] = -2.06302315E+01;
    COFD[1681] = 5.06978417E+00;
    COFD[1682] = -4.23521964E-01;
    COFD[1683] = 1.75535927E-02;
    COFD[1684] = -1.95390986E+01;
    COFD[1685] = 4.88982264E+00;
    COFD[1686] = -4.14974963E-01;
    COFD[1687] = 1.77824366E-02;
    COFD[1688] = -2.04259672E+01;
    COFD[1689] = 5.05147258E+00;
    COFD[1690] = -4.25999526E-01;
    COFD[1691] = 1.78562806E-02;
    COFD[1692] = -2.06459911E+01;
    COFD[1693] = 5.05102272E+00;
    COFD[1694] = -4.16130187E-01;
    COFD[1695] = 1.70344431E-02;
    COFD[1696] = -2.03217812E+01;
    COFD[1697] = 4.71342614E+00;
    COFD[1698] = -3.53042983E-01;
    COFD[1699] = 1.35344523E-02;
    COFD[1700] = -2.06949058E+01;
    COFD[1701] = 5.06751728E+00;
    COFD[1702] = -4.22067962E-01;
    COFD[1703] = 1.74439059E-02;
    COFD[1704] = -2.04360549E+01;
    COFD[1705] = 5.05147258E+00;
    COFD[1706] = -4.25999526E-01;
    COFD[1707] = 1.78562806E-02;
    COFD[1708] = -2.06863093E+01;
    COFD[1709] = 5.06751728E+00;
    COFD[1710] = -4.22067962E-01;
    COFD[1711] = 1.74439059E-02;
    COFD[1712] = -1.95542611E+01;
    COFD[1713] = 4.91462607E+00;
    COFD[1714] = -4.16749402E-01;
    COFD[1715] = 1.77999922E-02;
    COFD[1716] = -2.03314150E+01;
    COFD[1717] = 4.71342614E+00;
    COFD[1718] = -3.53042983E-01;
    COFD[1719] = 1.35344523E-02;
    COFD[1720] = -2.03266877E+01;
    COFD[1721] = 4.71342614E+00;
    COFD[1722] = -3.53042983E-01;
    COFD[1723] = 1.35344523E-02;
    COFD[1724] = -2.02725598E+01;
    COFD[1725] = 4.61233255E+00;
    COFD[1726] = -3.35903219E-01;
    COFD[1727] = 1.26325020E-02;
    COFD[1728] = -1.79431810E+01;
    COFD[1729] = 4.32370416E+00;
    COFD[1730] = -3.48535839E-01;
    COFD[1731] = 1.51861878E-02;
    COFD[1732] = -1.74071864E+01;
    COFD[1733] = 4.76097419E+00;
    COFD[1734] = -4.01175756E-01;
    COFD[1735] = 1.73035102E-02;
    COFD[1736] = -1.41065986E+01;
    COFD[1737] = 3.35888991E+00;
    COFD[1738] = -2.27939148E-01;
    COFD[1739] = 1.01542945E-02;
    COFD[1740] = -1.69981241E+01;
    COFD[1741] = 4.13179721E+00;
    COFD[1742] = -3.25544993E-01;
    COFD[1743] = 1.42694195E-02;
    COFD[1744] = -1.70170987E+01;
    COFD[1745] = 4.13179721E+00;
    COFD[1746] = -3.25544993E-01;
    COFD[1747] = 1.42694195E-02;
    COFD[1748] = -1.82852741E+01;
    COFD[1749] = 4.45298414E+00;
    COFD[1750] = -3.64713014E-01;
    COFD[1751] = 1.58635108E-02;
    COFD[1752] = -1.82990356E+01;
    COFD[1753] = 4.45298414E+00;
    COFD[1754] = -3.64713014E-01;
    COFD[1755] = 1.58635108E-02;
    COFD[1756] = -1.94726743E+01;
    COFD[1757] = 4.60063128E+00;
    COFD[1758] = -3.33961692E-01;
    COFD[1759] = 1.25315265E-02;
    COFD[1760] = -1.82923177E+01;
    COFD[1761] = 4.45298414E+00;
    COFD[1762] = -3.64713014E-01;
    COFD[1763] = 1.58635108E-02;
    COFD[1764] = -1.79714524E+01;
    COFD[1765] = 4.33127484E+00;
    COFD[1766] = -3.49477255E-01;
    COFD[1767] = 1.52253055E-02;
    COFD[1768] = -1.90342882E+01;
    COFD[1769] = 4.75331782E+00;
    COFD[1770] = -4.00274636E-01;
    COFD[1771] = 1.72681883E-02;
    COFD[1772] = -2.01639003E+01;
    COFD[1773] = 4.77237985E+00;
    COFD[1774] = -3.63201359E-01;
    COFD[1775] = 1.40726267E-02;
    COFD[1776] = -2.03253997E+01;
    COFD[1777] = 5.03708859E+00;
    COFD[1778] = -4.25057078E-01;
    COFD[1779] = 1.78526666E-02;
    COFD[1780] = -1.89816408E+01;
    COFD[1781] = 4.73272421E+00;
    COFD[1782] = -3.97843585E-01;
    COFD[1783] = 1.71725994E-02;
    COFD[1784] = -2.00652152E+01;
    COFD[1785] = 4.98853197E+00;
    COFD[1786] = -4.23032272E-01;
    COFD[1787] = 1.79382596E-02;
    COFD[1788] = -2.04789341E+01;
    COFD[1789] = 5.07632177E+00;
    COFD[1790] = -4.26053731E-01;
    COFD[1791] = 1.77312885E-02;
    COFD[1792] = -2.05154157E+01;
    COFD[1793] = 4.89869111E+00;
    COFD[1794] = -3.85699208E-01;
    COFD[1795] = 1.52891592E-02;
    COFD[1796] = -2.04360549E+01;
    COFD[1797] = 5.05147258E+00;
    COFD[1798] = -4.25999526E-01;
    COFD[1799] = 1.78562806E-02;
    COFD[1800] = -2.00748005E+01;
    COFD[1801] = 4.98853197E+00;
    COFD[1802] = -4.23032272E-01;
    COFD[1803] = 1.79382596E-02;
    COFD[1804] = -2.04279098E+01;
    COFD[1805] = 5.05147258E+00;
    COFD[1806] = -4.25999526E-01;
    COFD[1807] = 1.78562806E-02;
    COFD[1808] = -1.90604490E+01;
    COFD[1809] = 4.79559716E+00;
    COFD[1810] = -4.05182028E-01;
    COFD[1811] = 1.74574754E-02;
    COFD[1812] = -2.05244546E+01;
    COFD[1813] = 4.89869111E+00;
    COFD[1814] = -3.85699208E-01;
    COFD[1815] = 1.52891592E-02;
    COFD[1816] = -2.05200205E+01;
    COFD[1817] = 4.89869111E+00;
    COFD[1818] = -3.85699208E-01;
    COFD[1819] = 1.52891592E-02;
    COFD[1820] = -2.05317601E+01;
    COFD[1821] = 4.82637944E+00;
    COFD[1822] = -3.72748017E-01;
    COFD[1823] = 1.45861475E-02;
    COFD[1824] = -1.86561847E+01;
    COFD[1825] = 4.55615891E+00;
    COFD[1826] = -3.77295769E-01;
    COFD[1827] = 1.63770696E-02;
    COFD[1828] = -1.79184601E+01;
    COFD[1829] = 4.89943438E+00;
    COFD[1830] = -4.15602392E-01;
    COFD[1831] = 1.77841166E-02;
    COFD[1832] = -1.46102386E+01;
    COFD[1833] = 3.52131229E+00;
    COFD[1834] = -2.48399935E-01;
    COFD[1835] = 1.10136492E-02;
    COFD[1836] = -1.75915091E+01;
    COFD[1837] = 4.31462284E+00;
    COFD[1838] = -3.47471819E-01;
    COFD[1839] = 1.51448695E-02;
    COFD[1840] = -1.76109957E+01;
    COFD[1841] = 4.31462284E+00;
    COFD[1842] = -3.47471819E-01;
    COFD[1843] = 1.51448695E-02;
    COFD[1844] = -1.88832025E+01;
    COFD[1845] = 4.63397272E+00;
    COFD[1846] = -3.86072736E-01;
    COFD[1847] = 1.67042917E-02;
    COFD[1848] = -1.88975093E+01;
    COFD[1849] = 4.63397272E+00;
    COFD[1850] = -3.86072736E-01;
    COFD[1851] = 1.67042917E-02;
    COFD[1852] = -1.89423062E+01;
    COFD[1853] = 4.27158022E+00;
    COFD[1854] = -2.80784144E-01;
    COFD[1855] = 9.81431834E-03;
    COFD[1856] = -1.88905232E+01;
    COFD[1857] = 4.63397272E+00;
    COFD[1858] = -3.86072736E-01;
    COFD[1859] = 1.67042917E-02;
    COFD[1860] = -1.86789709E+01;
    COFD[1861] = 4.56143198E+00;
    COFD[1862] = -3.77911104E-01;
    COFD[1863] = 1.64009932E-02;
    COFD[1864] = -1.95533235E+01;
    COFD[1865] = 4.89649396E+00;
    COFD[1866] = -4.15388755E-01;
    COFD[1867] = 1.77817581E-02;
    COFD[1868] = -1.98184462E+01;
    COFD[1869] = 4.52357870E+00;
    COFD[1870] = -3.21272099E-01;
    COFD[1871] = 1.18751756E-02;
    COFD[1872] = -2.06200338E+01;
    COFD[1873] = 5.06978417E+00;
    COFD[1874] = -4.23521964E-01;
    COFD[1875] = 1.75535927E-02;
    COFD[1876] = -1.95331014E+01;
    COFD[1877] = 4.88982264E+00;
    COFD[1878] = -4.14974963E-01;
    COFD[1879] = 1.77824366E-02;
    COFD[1880] = -2.04179835E+01;
    COFD[1881] = 5.05147258E+00;
    COFD[1882] = -4.25999526E-01;
    COFD[1883] = 1.78562806E-02;
    COFD[1884] = -2.06376903E+01;
    COFD[1885] = 5.05102272E+00;
    COFD[1886] = -4.16130187E-01;
    COFD[1887] = 1.70344431E-02;
    COFD[1888] = -2.03117722E+01;
    COFD[1889] = 4.71342614E+00;
    COFD[1890] = -3.53042983E-01;
    COFD[1891] = 1.35344523E-02;
    COFD[1892] = -2.06863093E+01;
    COFD[1893] = 5.06751728E+00;
    COFD[1894] = -4.22067962E-01;
    COFD[1895] = 1.74439059E-02;
    COFD[1896] = -2.04279098E+01;
    COFD[1897] = 5.05147258E+00;
    COFD[1898] = -4.25999526E-01;
    COFD[1899] = 1.78562806E-02;
    COFD[1900] = -2.06778581E+01;
    COFD[1901] = 5.06751728E+00;
    COFD[1902] = -4.22067962E-01;
    COFD[1903] = 1.74439059E-02;
    COFD[1904] = -1.95443526E+01;
    COFD[1905] = 4.91462607E+00;
    COFD[1906] = -4.16749402E-01;
    COFD[1907] = 1.77999922E-02;
    COFD[1908] = -2.03212133E+01;
    COFD[1909] = 4.71342614E+00;
    COFD[1910] = -3.53042983E-01;
    COFD[1911] = 1.35344523E-02;
    COFD[1912] = -2.03165810E+01;
    COFD[1913] = 4.71342614E+00;
    COFD[1914] = -3.53042983E-01;
    COFD[1915] = 1.35344523E-02;
    COFD[1916] = -2.02622655E+01;
    COFD[1917] = 4.61233255E+00;
    COFD[1918] = -3.35903219E-01;
    COFD[1919] = 1.26325020E-02;
    COFD[1920] = -1.66078875E+01;
    COFD[1921] = 3.97848216E+00;
    COFD[1922] = -3.06230410E-01;
    COFD[1923] = 1.34556563E-02;
    COFD[1924] = -1.58083316E+01;
    COFD[1925] = 4.41251740E+00;
    COFD[1926] = -3.59679513E-01;
    COFD[1927] = 1.56539353E-02;
    COFD[1928] = -1.27155593E+01;
    COFD[1929] = 3.02109660E+00;
    COFD[1930] = -1.83658057E-01;
    COFD[1931] = 8.21743108E-03;
    COFD[1932] = -1.55038857E+01;
    COFD[1933] = 3.75336567E+00;
    COFD[1934] = -2.77901896E-01;
    COFD[1935] = 1.22658049E-02;
    COFD[1936] = -1.55256733E+01;
    COFD[1937] = 3.75336567E+00;
    COFD[1938] = -2.77901896E-01;
    COFD[1939] = 1.22658049E-02;
    COFD[1940] = -1.68776819E+01;
    COFD[1941] = 4.08227013E+00;
    COFD[1942] = -3.19188042E-01;
    COFD[1943] = 1.39963401E-02;
    COFD[1944] = -1.68946144E+01;
    COFD[1945] = 4.08227013E+00;
    COFD[1946] = -3.19188042E-01;
    COFD[1947] = 1.39963401E-02;
    COFD[1948] = -1.95268261E+01;
    COFD[1949] = 4.94867364E+00;
    COFD[1950] = -3.94985793E-01;
    COFD[1951] = 1.58038735E-02;
    COFD[1952] = -1.68863351E+01;
    COFD[1953] = 4.08227013E+00;
    COFD[1954] = -3.19188042E-01;
    COFD[1955] = 1.39963401E-02;
    COFD[1956] = -1.66322681E+01;
    COFD[1957] = 3.98376384E+00;
    COFD[1958] = -3.06871085E-01;
    COFD[1959] = 1.34815439E-02;
    COFD[1960] = -1.76407601E+01;
    COFD[1961] = 4.40309399E+00;
    COFD[1962] = -3.58496732E-01;
    COFD[1963] = 1.56042461E-02;
    COFD[1964] = -2.01085993E+01;
    COFD[1965] = 5.02691790E+00;
    COFD[1966] = -4.10940210E-01;
    COFD[1967] = 1.67272564E-02;
    COFD[1968] = -1.94319186E+01;
    COFD[1969] = 4.89964403E+00;
    COFD[1970] = -4.15618593E-01;
    COFD[1971] = 1.77843657E-02;
    COFD[1972] = -1.75794891E+01;
    COFD[1973] = 4.37827292E+00;
    COFD[1974] = -3.55380310E-01;
    COFD[1975] = 1.54732833E-02;
    COFD[1976] = -1.90489173E+01;
    COFD[1977] = 4.79559716E+00;
    COFD[1978] = -4.05182028E-01;
    COFD[1979] = 1.74574754E-02;
    COFD[1980] = -1.96421788E+01;
    COFD[1981] = 4.97225057E+00;
    COFD[1982] = -4.21797234E-01;
    COFD[1983] = 1.79201792E-02;
    COFD[1984] = -2.03011295E+01;
    COFD[1985] = 5.06448019E+00;
    COFD[1986] = -4.20705675E-01;
    COFD[1987] = 1.73458737E-02;
    COFD[1988] = -1.95542611E+01;
    COFD[1989] = 4.91462607E+00;
    COFD[1990] = -4.16749402E-01;
    COFD[1991] = 1.77999922E-02;
    COFD[1992] = -1.90604490E+01;
    COFD[1993] = 4.79559716E+00;
    COFD[1994] = -4.05182028E-01;
    COFD[1995] = 1.74574754E-02;
    COFD[1996] = -1.95443526E+01;
    COFD[1997] = 4.91462607E+00;
    COFD[1998] = -4.16749402E-01;
    COFD[1999] = 1.77999922E-02;
    COFD[2000] = -1.76353757E+01;
    COFD[2001] = 4.45601256E+00;
    COFD[2002] = -3.65089653E-01;
    COFD[2003] = 1.58791949E-02;
    COFD[2004] = -2.03125607E+01;
    COFD[2005] = 5.06448019E+00;
    COFD[2006] = -4.20705675E-01;
    COFD[2007] = 1.73458737E-02;
    COFD[2008] = -2.03069463E+01;
    COFD[2009] = 5.06448019E+00;
    COFD[2010] = -4.20705675E-01;
    COFD[2011] = 1.73458737E-02;
    COFD[2012] = -2.04809161E+01;
    COFD[2013] = 5.05185392E+00;
    COFD[2014] = -4.16368395E-01;
    COFD[2015] = 1.70499848E-02;
    COFD[2016] = -1.99921241E+01;
    COFD[2017] = 4.97669986E+00;
    COFD[2018] = -4.22149407E-01;
    COFD[2019] = 1.79263822E-02;
    COFD[2020] = -1.86137387E+01;
    COFD[2021] = 5.06806259E+00;
    COFD[2022] = -4.22384317E-01;
    COFD[2023] = 1.74675251E-02;
    COFD[2024] = -1.61516158E+01;
    COFD[2025] = 4.11963986E+00;
    COFD[2026] = -3.23983555E-01;
    COFD[2027] = 1.42022890E-02;
    COFD[2028] = -1.91943549E+01;
    COFD[2029] = 4.87847769E+00;
    COFD[2030] = -4.14012329E-01;
    COFD[2031] = 1.77616586E-02;
    COFD[2032] = -1.92165800E+01;
    COFD[2033] = 4.87847769E+00;
    COFD[2034] = -4.14012329E-01;
    COFD[2035] = 1.77616586E-02;
    COFD[2036] = -2.01364051E+01;
    COFD[2037] = 5.01062757E+00;
    COFD[2038] = -4.24032393E-01;
    COFD[2039] = 1.79058125E-02;
    COFD[2040] = -2.01538718E+01;
    COFD[2041] = 5.01062757E+00;
    COFD[2042] = -4.24032393E-01;
    COFD[2043] = 1.79058125E-02;
    COFD[2044] = -1.59109328E+01;
    COFD[2045] = 2.78426888E+00;
    COFD[2046] = -5.24195109E-02;
    COFD[2047] = -1.45484926E-03;
    COFD[2048] = -2.01453289E+01;
    COFD[2049] = 5.01062757E+00;
    COFD[2050] = -4.24032393E-01;
    COFD[2051] = 1.79058125E-02;
    COFD[2052] = -2.00097568E+01;
    COFD[2053] = 4.97925805E+00;
    COFD[2054] = -4.22339984E-01;
    COFD[2055] = 1.79289294E-02;
    COFD[2056] = -2.03331380E+01;
    COFD[2057] = 5.06757610E+00;
    COFD[2058] = -4.22549988E-01;
    COFD[2059] = 1.74838847E-02;
    COFD[2060] = -1.73175806E+01;
    COFD[2061] = 3.24454040E+00;
    COFD[2062] = -1.21683836E-01;
    COFD[2063] = 1.91319147E-03;
    COFD[2064] = -2.03569735E+01;
    COFD[2065] = 4.74915125E+00;
    COFD[2066] = -3.59194781E-01;
    COFD[2067] = 1.38602188E-02;
    COFD[2068] = -2.03332321E+01;
    COFD[2069] = 5.06951156E+00;
    COFD[2070] = -4.23434786E-01;
    COFD[2071] = 1.75477252E-02;
    COFD[2072] = -2.05126034E+01;
    COFD[2073] = 4.89869111E+00;
    COFD[2074] = -3.85699208E-01;
    COFD[2075] = 1.52891592E-02;
    COFD[2076] = -1.99794321E+01;
    COFD[2077] = 4.56517018E+00;
    COFD[2078] = -3.28088445E-01;
    COFD[2079] = 1.22265811E-02;
    COFD[2080] = -1.83076885E+01;
    COFD[2081] = 3.63551893E+00;
    COFD[2082] = -1.81306348E-01;
    COFD[2083] = 4.84147153E-03;
    COFD[2084] = -2.03314150E+01;
    COFD[2085] = 4.71342614E+00;
    COFD[2086] = -3.53042983E-01;
    COFD[2087] = 1.35344523E-02;
    COFD[2088] = -2.05244546E+01;
    COFD[2089] = 4.89869111E+00;
    COFD[2090] = -3.85699208E-01;
    COFD[2091] = 1.52891592E-02;
    COFD[2092] = -2.03212133E+01;
    COFD[2093] = 4.71342614E+00;
    COFD[2094] = -3.53042983E-01;
    COFD[2095] = 1.35344523E-02;
    COFD[2096] = -2.03125607E+01;
    COFD[2097] = 5.06448019E+00;
    COFD[2098] = -4.20705675E-01;
    COFD[2099] = 1.73458737E-02;
    COFD[2100] = -1.83195359E+01;
    COFD[2101] = 3.63551893E+00;
    COFD[2102] = -1.81306348E-01;
    COFD[2103] = 4.84147153E-03;
    COFD[2104] = -1.83137158E+01;
    COFD[2105] = 3.63551893E+00;
    COFD[2106] = -1.81306348E-01;
    COFD[2107] = 4.84147153E-03;
    COFD[2108] = -1.79881031E+01;
    COFD[2109] = 3.41675012E+00;
    COFD[2110] = -1.47848340E-01;
    COFD[2111] = 3.19467888E-03;
    COFD[2112] = -1.99875936E+01;
    COFD[2113] = 4.97669986E+00;
    COFD[2114] = -4.22149407E-01;
    COFD[2115] = 1.79263822E-02;
    COFD[2116] = -1.86134769E+01;
    COFD[2117] = 5.06806259E+00;
    COFD[2118] = -4.22384317E-01;
    COFD[2119] = 1.74675251E-02;
    COFD[2120] = -1.61511037E+01;
    COFD[2121] = 4.11963986E+00;
    COFD[2122] = -3.23983555E-01;
    COFD[2123] = 1.42022890E-02;
    COFD[2124] = -1.91912453E+01;
    COFD[2125] = 4.87847769E+00;
    COFD[2126] = -4.14012329E-01;
    COFD[2127] = 1.77616586E-02;
    COFD[2128] = -1.92133295E+01;
    COFD[2129] = 4.87847769E+00;
    COFD[2130] = -4.14012329E-01;
    COFD[2131] = 1.77616586E-02;
    COFD[2132] = -2.01315030E+01;
    COFD[2133] = 5.01062757E+00;
    COFD[2134] = -4.24032393E-01;
    COFD[2135] = 1.79058125E-02;
    COFD[2136] = -2.01487964E+01;
    COFD[2137] = 5.01062757E+00;
    COFD[2138] = -4.24032393E-01;
    COFD[2139] = 1.79058125E-02;
    COFD[2140] = -1.59075460E+01;
    COFD[2141] = 2.78426888E+00;
    COFD[2142] = -5.24195109E-02;
    COFD[2143] = -1.45484926E-03;
    COFD[2144] = -2.01403390E+01;
    COFD[2145] = 5.01062757E+00;
    COFD[2146] = -4.24032393E-01;
    COFD[2147] = 1.79058125E-02;
    COFD[2148] = -2.00052265E+01;
    COFD[2149] = 4.97925805E+00;
    COFD[2150] = -4.22339984E-01;
    COFD[2151] = 1.79289294E-02;
    COFD[2152] = -2.03301677E+01;
    COFD[2153] = 5.06757610E+00;
    COFD[2154] = -4.22549988E-01;
    COFD[2155] = 1.74838847E-02;
    COFD[2156] = -1.73128574E+01;
    COFD[2157] = 3.24454040E+00;
    COFD[2158] = -1.21683836E-01;
    COFD[2159] = 1.91319147E-03;
    COFD[2160] = -2.03511563E+01;
    COFD[2161] = 4.74915125E+00;
    COFD[2162] = -3.59194781E-01;
    COFD[2163] = 1.38602188E-02;
    COFD[2164] = -2.03301163E+01;
    COFD[2165] = 5.06951156E+00;
    COFD[2166] = -4.23434786E-01;
    COFD[2167] = 1.75477252E-02;
    COFD[2168] = -2.05082727E+01;
    COFD[2169] = 4.89869111E+00;
    COFD[2170] = -3.85699208E-01;
    COFD[2171] = 1.52891592E-02;
    COFD[2172] = -1.99748976E+01;
    COFD[2173] = 4.56517018E+00;
    COFD[2174] = -3.28088445E-01;
    COFD[2175] = 1.22265811E-02;
    COFD[2176] = -1.83020039E+01;
    COFD[2177] = 3.63551893E+00;
    COFD[2178] = -1.81306348E-01;
    COFD[2179] = 4.84147153E-03;
    COFD[2180] = -2.03266877E+01;
    COFD[2181] = 4.71342614E+00;
    COFD[2182] = -3.53042983E-01;
    COFD[2183] = 1.35344523E-02;
    COFD[2184] = -2.05200205E+01;
    COFD[2185] = 4.89869111E+00;
    COFD[2186] = -3.85699208E-01;
    COFD[2187] = 1.52891592E-02;
    COFD[2188] = -2.03165810E+01;
    COFD[2189] = 4.71342614E+00;
    COFD[2190] = -3.53042983E-01;
    COFD[2191] = 1.35344523E-02;
    COFD[2192] = -2.03069463E+01;
    COFD[2193] = 5.06448019E+00;
    COFD[2194] = -4.20705675E-01;
    COFD[2195] = 1.73458737E-02;
    COFD[2196] = -1.83137158E+01;
    COFD[2197] = 3.63551893E+00;
    COFD[2198] = -1.81306348E-01;
    COFD[2199] = 4.84147153E-03;
    COFD[2200] = -1.83079627E+01;
    COFD[2201] = 3.63551893E+00;
    COFD[2202] = -1.81306348E-01;
    COFD[2203] = 4.84147153E-03;
    COFD[2204] = -1.79822176E+01;
    COFD[2205] = 3.41675012E+00;
    COFD[2206] = -1.47848340E-01;
    COFD[2207] = 3.19467888E-03;
    COFD[2208] = -2.02337108E+01;
    COFD[2209] = 5.00680505E+00;
    COFD[2210] = -4.24113291E-01;
    COFD[2211] = 1.79327553E-02;
    COFD[2212] = -1.87941740E+01;
    COFD[2213] = 5.05845386E+00;
    COFD[2214] = -4.18429681E-01;
    COFD[2215] = 1.71874793E-02;
    COFD[2216] = -1.65400407E+01;
    COFD[2217] = 4.21819964E+00;
    COFD[2218] = -3.36577566E-01;
    COFD[2219] = 1.47408361E-02;
    COFD[2220] = -1.94550962E+01;
    COFD[2221] = 4.91220991E+00;
    COFD[2222] = -4.16560501E-01;
    COFD[2223] = 1.77969106E-02;
    COFD[2224] = -1.94774575E+01;
    COFD[2225] = 4.91220991E+00;
    COFD[2226] = -4.16560501E-01;
    COFD[2227] = 1.77969106E-02;
    COFD[2228] = -2.03654908E+01;
    COFD[2229] = 5.03298632E+00;
    COFD[2230] = -4.24771669E-01;
    COFD[2231] = 1.78502612E-02;
    COFD[2232] = -2.03831264E+01;
    COFD[2233] = 5.03298632E+00;
    COFD[2234] = -4.24771669E-01;
    COFD[2235] = 1.78502612E-02;
    COFD[2236] = -1.55408232E+01;
    COFD[2237] = 2.54259134E+00;
    COFD[2238] = -1.60736376E-02;
    COFD[2239] = -3.21856796E-03;
    COFD[2240] = -2.03745002E+01;
    COFD[2241] = 5.03298632E+00;
    COFD[2242] = -4.24771669E-01;
    COFD[2243] = 1.78502612E-02;
    COFD[2244] = -2.02493305E+01;
    COFD[2245] = 5.00861852E+00;
    COFD[2246] = -4.24191458E-01;
    COFD[2247] = 1.79297653E-02;
    COFD[2248] = -2.04891102E+01;
    COFD[2249] = 5.05959508E+00;
    COFD[2250] = -4.18821304E-01;
    COFD[2251] = 1.72141752E-02;
    COFD[2252] = -1.69183106E+01;
    COFD[2253] = 2.99595407E+00;
    COFD[2254] = -8.41851353E-02;
    COFD[2255] = 8.62518660E-05;
    COFD[2256] = -2.03360102E+01;
    COFD[2257] = 4.66227065E+00;
    COFD[2258] = -3.44237793E-01;
    COFD[2259] = 1.30673058E-02;
    COFD[2260] = -2.04918650E+01;
    COFD[2261] = 5.06231507E+00;
    COFD[2262] = -4.19815596E-01;
    COFD[2263] = 1.72829136E-02;
    COFD[2264] = -2.05198084E+01;
    COFD[2265] = 4.82637944E+00;
    COFD[2266] = -3.72748017E-01;
    COFD[2267] = 1.45861475E-02;
    COFD[2268] = -1.98664135E+01;
    COFD[2269] = 4.44007006E+00;
    COFD[2270] = -3.07688868E-01;
    COFD[2271] = 1.11784091E-02;
    COFD[2272] = -1.79761233E+01;
    COFD[2273] = 3.41675012E+00;
    COFD[2274] = -1.47848340E-01;
    COFD[2275] = 3.19467888E-03;
    COFD[2276] = -2.02725598E+01;
    COFD[2277] = 4.61233255E+00;
    COFD[2278] = -3.35903219E-01;
    COFD[2279] = 1.26325020E-02;
    COFD[2280] = -2.05317601E+01;
    COFD[2281] = 4.82637944E+00;
    COFD[2282] = -3.72748017E-01;
    COFD[2283] = 1.45861475E-02;
    COFD[2284] = -2.02622655E+01;
    COFD[2285] = 4.61233255E+00;
    COFD[2286] = -3.35903219E-01;
    COFD[2287] = 1.26325020E-02;
    COFD[2288] = -2.04809161E+01;
    COFD[2289] = 5.05185392E+00;
    COFD[2290] = -4.16368395E-01;
    COFD[2291] = 1.70499848E-02;
    COFD[2292] = -1.79881031E+01;
    COFD[2293] = 3.41675012E+00;
    COFD[2294] = -1.47848340E-01;
    COFD[2295] = 3.19467888E-03;
    COFD[2296] = -1.79822176E+01;
    COFD[2297] = 3.41675012E+00;
    COFD[2298] = -1.47848340E-01;
    COFD[2299] = 3.19467888E-03;
    COFD[2300] = -1.76128345E+01;
    COFD[2301] = 3.18206700E+00;
    COFD[2302] = -1.12232277E-01;
    COFD[2303] = 1.45170418E-03;
}


/*List of specs with small weight, dim NLITE */
void egtransetKTDIF(int* KTDIF) {
    KTDIF[0] = 2;
    KTDIF[1] = 3;
}


/*Poly fits for thermal diff ratios, dim NO*NLITE*KK */
void egtransetCOFTD(double* COFTD) {
    COFTD[0] = 1.60027184E-01;
    COFTD[1] = 5.94963466E-04;
    COFTD[2] = -2.96929756E-07;
    COFTD[3] = 4.51232494E-11;
    COFTD[4] = 0.00000000E+00;
    COFTD[5] = 0.00000000E+00;
    COFTD[6] = 0.00000000E+00;
    COFTD[7] = 0.00000000E+00;
    COFTD[8] = 1.26715692E-01;
    COFTD[9] = 1.02530485E-04;
    COFTD[10] = -5.45604892E-08;
    COFTD[11] = 8.85181063E-12;
    COFTD[12] = 1.93107545E-01;
    COFTD[13] = 5.04123759E-04;
    COFTD[14] = -2.55796068E-07;
    COFTD[15] = 3.93758307E-11;
    COFTD[16] = 1.94560455E-01;
    COFTD[17] = 5.07916706E-04;
    COFTD[18] = -2.57720638E-07;
    COFTD[19] = 3.96720883E-11;
    COFTD[20] = 1.39745676E-01;
    COFTD[21] = 6.29810814E-04;
    COFTD[22] = -3.11694011E-07;
    COFTD[23] = 4.70755830E-11;
    COFTD[24] = 1.40268928E-01;
    COFTD[25] = 6.32169024E-04;
    COFTD[26] = -3.12861091E-07;
    COFTD[27] = 4.72518487E-11;
    COFTD[28] = -1.60928523E-01;
    COFTD[29] = 8.01685562E-04;
    COFTD[30] = -3.24976618E-07;
    COFTD[31] = 4.31958164E-11;
    COFTD[32] = 1.40015055E-01;
    COFTD[33] = 6.31024860E-04;
    COFTD[34] = -3.12294843E-07;
    COFTD[35] = 4.71663275E-11;
    COFTD[36] = 1.58726794E-01;
    COFTD[37] = 5.96753380E-04;
    COFTD[38] = -2.97673771E-07;
    COFTD[39] = 4.52193864E-11;
    COFTD[40] = 6.87555609E-02;
    COFTD[41] = 6.63104891E-04;
    COFTD[42] = -3.19484931E-07;
    COFTD[43] = 4.73610096E-11;
    COFTD[44] = -1.52955380E-01;
    COFTD[45] = 8.50147266E-04;
    COFTD[46] = -3.52941514E-07;
    COFTD[47] = 4.76345274E-11;
    COFTD[48] = -3.84966170E-02;
    COFTD[49] = 8.34794926E-04;
    COFTD[50] = -3.81031231E-07;
    COFTD[51] = 5.45531902E-11;
    COFTD[52] = 7.31516344E-02;
    COFTD[53] = 6.64295541E-04;
    COFTD[54] = -3.20611260E-07;
    COFTD[55] = 4.75831553E-11;
    COFTD[56] = -6.48003554E-03;
    COFTD[57] = 7.83606549E-04;
    COFTD[58] = -3.63734726E-07;
    COFTD[59] = 5.26307223E-11;
    COFTD[60] = -6.41009872E-02;
    COFTD[61] = 8.31131214E-04;
    COFTD[62] = -3.73287728E-07;
    COFTD[63] = 5.29105483E-11;
    COFTD[64] = -1.38459273E-01;
    COFTD[65] = 8.72086221E-04;
    COFTD[66] = -3.69606901E-07;
    COFTD[67] = 5.05366623E-11;
    COFTD[68] = -4.41931237E-02;
    COFTD[69] = 8.21967460E-04;
    COFTD[70] = -3.73768016E-07;
    COFTD[71] = 5.33878066E-11;
    COFTD[72] = -6.49878719E-03;
    COFTD[73] = 7.85874116E-04;
    COFTD[74] = -3.64787286E-07;
    COFTD[75] = 5.27830228E-11;
    COFTD[76] = -4.40903654E-02;
    COFTD[77] = 8.20056213E-04;
    COFTD[78] = -3.72898927E-07;
    COFTD[79] = 5.32636687E-11;
    COFTD[80] = 6.56752532E-02;
    COFTD[81] = 7.32573145E-04;
    COFTD[82] = -3.51582658E-07;
    COFTD[83] = 5.19837010E-11;
    COFTD[84] = -1.38763623E-01;
    COFTD[85] = 8.74003169E-04;
    COFTD[86] = -3.70419341E-07;
    COFTD[87] = 5.06477479E-11;
    COFTD[88] = -1.38614930E-01;
    COFTD[89] = 8.73066625E-04;
    COFTD[90] = -3.70022415E-07;
    COFTD[91] = 5.05934759E-11;
    COFTD[92] = -1.49257151E-01;
    COFTD[93] = 8.72452872E-04;
    COFTD[94] = -3.65483657E-07;
    COFTD[95] = 4.96091723E-11;
    COFTD[96] = 3.88181125E-01;
    COFTD[97] = 1.55380218E-04;
    COFTD[98] = -8.20880914E-08;
    COFTD[99] = 1.37104636E-11;
    COFTD[100] = -1.26715692E-01;
    COFTD[101] = -1.02530485E-04;
    COFTD[102] = 5.45604892E-08;
    COFTD[103] = -8.85181063E-12;
    COFTD[104] = 0.00000000E+00;
    COFTD[105] = 0.00000000E+00;
    COFTD[106] = 0.00000000E+00;
    COFTD[107] = 0.00000000E+00;
    COFTD[108] = 3.69825311E-01;
    COFTD[109] = 9.58927840E-05;
    COFTD[110] = -4.86954016E-08;
    COFTD[111] = 8.20765740E-12;
    COFTD[112] = 3.75475346E-01;
    COFTD[113] = 9.73577933E-05;
    COFTD[114] = -4.94393493E-08;
    COFTD[115] = 8.33305051E-12;
    COFTD[116] = 3.81864172E-01;
    COFTD[117] = 1.84117353E-04;
    COFTD[118] = -9.79617476E-08;
    COFTD[119] = 1.62542227E-11;
    COFTD[120] = 3.84737257E-01;
    COFTD[121] = 1.85502623E-04;
    COFTD[122] = -9.86987963E-08;
    COFTD[123] = 1.63765169E-11;
    COFTD[124] = 1.95123509E-03;
    COFTD[125] = 6.69470998E-04;
    COFTD[126] = -3.12148757E-07;
    COFTD[127] = 4.52949938E-11;
    COFTD[128] = 3.83342059E-01;
    COFTD[129] = 1.84829923E-04;
    COFTD[130] = -9.83408783E-08;
    COFTD[131] = 1.63171297E-11;
    COFTD[132] = 3.87405318E-01;
    COFTD[133] = 1.56883797E-04;
    COFTD[134] = -8.29309791E-08;
    COFTD[135] = 1.38460299E-11;
    COFTD[136] = 2.91262628E-01;
    COFTD[137] = 2.33045049E-04;
    COFTD[138] = -1.24040097E-07;
    COFTD[139] = 2.01345857E-11;
    COFTD[140] = 8.92182336E-02;
    COFTD[141] = 6.38597299E-04;
    COFTD[142] = -3.10526695E-07;
    COFTD[143] = 4.63218238E-11;
    COFTD[144] = 2.47129011E-01;
    COFTD[145] = 4.49395677E-04;
    COFTD[146] = -2.32030740E-07;
    COFTD[147] = 3.62578797E-11;
    COFTD[148] = 2.98973958E-01;
    COFTD[149] = 2.32230992E-04;
    COFTD[150] = -1.23675023E-07;
    COFTD[151] = 2.01029713E-11;
    COFTD[152] = 2.61383000E-01;
    COFTD[153] = 3.74100112E-04;
    COFTD[154] = -1.95275625E-07;
    COFTD[155] = 3.08444615E-11;
    COFTD[156] = 2.06621091E-01;
    COFTD[157] = 4.69839097E-04;
    COFTD[158] = -2.39996550E-07;
    COFTD[159] = 3.71486636E-11;
    COFTD[160] = 1.21690755E-01;
    COFTD[161] = 6.27167274E-04;
    COFTD[162] = -3.08682525E-07;
    COFTD[163] = 4.64372279E-11;
    COFTD[164] = 2.30178126E-01;
    COFTD[165] = 4.41123938E-04;
    COFTD[166] = -2.27192767E-07;
    COFTD[167] = 3.54210043E-11;
    COFTD[168] = 2.62904564E-01;
    COFTD[169] = 3.76277825E-04;
    COFTD[170] = -1.96412365E-07;
    COFTD[171] = 3.10240134E-11;
    COFTD[172] = 2.29105206E-01;
    COFTD[173] = 4.39067745E-04;
    COFTD[174] = -2.26133762E-07;
    COFTD[175] = 3.52558978E-11;
    COFTD[176] = 3.38642057E-01;
    COFTD[177] = 2.89373567E-04;
    COFTD[178] = -1.53802895E-07;
    COFTD[179] = 2.48891465E-11;
    COFTD[180] = 1.22227211E-01;
    COFTD[181] = 6.29932046E-04;
    COFTD[182] = -3.10043305E-07;
    COFTD[183] = 4.66419394E-11;
    COFTD[184] = 1.21964983E-01;
    COFTD[185] = 6.28580582E-04;
    COFTD[186] = -3.09378135E-07;
    COFTD[187] = 4.65418733E-11;
    COFTD[188] = 1.05711474E-01;
    COFTD[189] = 6.52619511E-04;
    COFTD[190] = -3.19000041E-07;
    COFTD[191] = 4.77570895E-11;
}

/* Replace this routine with the one generated by the Gauss Jordan solver of DW */
AMREX_GPU_HOST_DEVICE void sgjsolve(double* A, double* x, double* b) {
    amrex::Abort("sgjsolve not implemented, choose a different solver ");
}

/* Replace this routine with the one generated by the Gauss Jordan solver of DW */
AMREX_GPU_HOST_DEVICE void sgjsolve_simplified(double* A, double* x, double* b) {
    amrex::Abort("sgjsolve_simplified not implemented, choose a different solver ");
}

