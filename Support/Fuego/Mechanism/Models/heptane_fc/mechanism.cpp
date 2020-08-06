#include "chemistry_file.H"

#ifndef AMREX_USE_CUDA
namespace thermo
{
    double fwd_A[218], fwd_beta[218], fwd_Ea[218];
    double low_A[218], low_beta[218], low_Ea[218];
    double rev_A[218], rev_beta[218], rev_Ea[218];
    double troe_a[218],troe_Ts[218], troe_Tss[218], troe_Tsss[218];
    double sri_a[218], sri_b[218], sri_c[218], sri_d[218], sri_e[218];
    double activation_units[218], prefactor_units[218], phase_units[218];
    int is_PD[218], troe_len[218], sri_len[218], nTB[218], *TBid[218];
    double *TB[218];
    std::vector<std::vector<double>> kiv(218); 
    std::vector<std::vector<double>> nuv(218); 
    std::vector<std::vector<double>> kiv_qss(218); 
    std::vector<std::vector<double>> nuv_qss(218); 
};

using namespace thermo;
#endif

/* Inverse molecular weights */
static AMREX_GPU_DEVICE_MANAGED double imw[52] = {
    1.0 / 28.013400,  /*N2 */
    1.0 / 15.999400,  /*O */
    1.0 / 2.015940,  /*H2 */
    1.0 / 1.007970,  /*H */
    1.0 / 17.007370,  /*OH */
    1.0 / 18.015340,  /*H2O */
    1.0 / 31.998800,  /*O2 */
    1.0 / 33.006770,  /*HO2 */
    1.0 / 34.014740,  /*H2O2 */
    1.0 / 13.019120,  /*CH */
    1.0 / 29.018520,  /*HCO */
    1.0 / 14.027090,  /*CH2 */
    1.0 / 44.009950,  /*CO2 */
    1.0 / 28.010550,  /*CO */
    1.0 / 30.026490,  /*CH2O */
    1.0 / 14.027090,  /*CH2GSG */
    1.0 / 15.035060,  /*CH3 */
    1.0 / 31.034460,  /*CH3O */
    1.0 / 16.043030,  /*CH4 */
    1.0 / 32.042430,  /*CH3OH */
    1.0 / 30.070120,  /*C2H6 */
    1.0 / 29.062150,  /*C2H5 */
    1.0 / 42.037640,  /*CH2CO */
    1.0 / 46.025890,  /*HOCHO */
    1.0 / 47.033860,  /*CH3O2 */
    1.0 / 48.041830,  /*CH3O2H */
    1.0 / 26.038240,  /*C2H2 */
    1.0 / 41.029670,  /*HCCO */
    1.0 / 27.046210,  /*C2H3 */
    1.0 / 43.045610,  /*CH2CHO */
    1.0 / 42.081270,  /*C3H6 */
    1.0 / 28.054180,  /*C2H4 */
    1.0 / 45.061550,  /*C2H5O */
    1.0 / 43.045610,  /*CH3CO */
    1.0 / 61.060950,  /*C2H5O2 */
    1.0 / 38.049390,  /*C3H2 */
    1.0 / 39.057360,  /*C3H3 */
    1.0 / 40.065330,  /*C3H4XA */
    1.0 / 41.073300,  /*C3H5XA */
    1.0 / 43.089240,  /*NXC3H7 */
    1.0 / 75.088040,  /*NXC3H7O2 */
    1.0 / 54.092420,  /*C4H6 */
    1.0 / 55.100390,  /*C4H7 */
    1.0 / 56.108360,  /*C4H8X1 */
    1.0 / 57.116330,  /*PXC4H9 */
    1.0 / 89.115130,  /*PXC4H9O2 */
    1.0 / 69.127480,  /*C5H9 */
    1.0 / 70.135450,  /*C5H10X1 */
    1.0 / 71.143420,  /*C5H11X1 */
    1.0 / 84.162540,  /*C6H12X1 */
    1.0 / 99.197600,  /*C7H15X2 */
    1.0 / 100.205570};  /*NXC7H16 */

/* Molecular weights */
static AMREX_GPU_DEVICE_MANAGED double molecular_weights[52] = {
    28.013400,  /*N2 */
    15.999400,  /*O */
    2.015940,  /*H2 */
    1.007970,  /*H */
    17.007370,  /*OH */
    18.015340,  /*H2O */
    31.998800,  /*O2 */
    33.006770,  /*HO2 */
    34.014740,  /*H2O2 */
    13.019120,  /*CH */
    29.018520,  /*HCO */
    14.027090,  /*CH2 */
    44.009950,  /*CO2 */
    28.010550,  /*CO */
    30.026490,  /*CH2O */
    14.027090,  /*CH2GSG */
    15.035060,  /*CH3 */
    31.034460,  /*CH3O */
    16.043030,  /*CH4 */
    32.042430,  /*CH3OH */
    30.070120,  /*C2H6 */
    29.062150,  /*C2H5 */
    42.037640,  /*CH2CO */
    46.025890,  /*HOCHO */
    47.033860,  /*CH3O2 */
    48.041830,  /*CH3O2H */
    26.038240,  /*C2H2 */
    41.029670,  /*HCCO */
    27.046210,  /*C2H3 */
    43.045610,  /*CH2CHO */
    42.081270,  /*C3H6 */
    28.054180,  /*C2H4 */
    45.061550,  /*C2H5O */
    43.045610,  /*CH3CO */
    61.060950,  /*C2H5O2 */
    38.049390,  /*C3H2 */
    39.057360,  /*C3H3 */
    40.065330,  /*C3H4XA */
    41.073300,  /*C3H5XA */
    43.089240,  /*NXC3H7 */
    75.088040,  /*NXC3H7O2 */
    54.092420,  /*C4H6 */
    55.100390,  /*C4H7 */
    56.108360,  /*C4H8X1 */
    57.116330,  /*PXC4H9 */
    89.115130,  /*PXC4H9O2 */
    69.127480,  /*C5H9 */
    70.135450,  /*C5H10X1 */
    71.143420,  /*C5H11X1 */
    84.162540,  /*C6H12X1 */
    99.197600,  /*C7H15X2 */
    100.205570};  /*NXC7H16 */

AMREX_GPU_HOST_DEVICE
void get_imw(double imw_new[]){
    for(int i = 0; i<52; ++i) imw_new[i] = imw[i];
}

/* TODO: check necessity because redundant with CKWT */
AMREX_GPU_HOST_DEVICE
void get_mw(double mw_new[]){
    for(int i = 0; i<52; ++i) mw_new[i] = molecular_weights[i];
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
    nTB[11] = 15;
    TB[11] = (double *) malloc(15 * sizeof(double));
    TBid[11] = (int *) malloc(15 * sizeof(int));
    TBid[11][0] = 29; TB[11][0] = 0; // CH2CHO
    TBid[11][1] = 9; TB[11][1] = 0; // CH
    TBid[11][2] = 13; TB[11][2] = 1.8999999999999999; // CO
    TBid[11][3] = 2; TB[11][3] = 2.5; // H2
    TBid[11][4] = 44; TB[11][4] = 0; // PXC4H9
    TBid[11][5] = 15; TB[11][5] = 0; // CH2GSG
    TBid[11][6] = 42; TB[11][6] = 0; // C4H7
    TBid[11][7] = 27; TB[11][7] = 0; // HCCO
    TBid[11][8] = 48; TB[11][8] = 0; // C5H11X1
    TBid[11][9] = 5; TB[11][9] = 12; // H2O
    TBid[11][10] = 12; TB[11][10] = 3.7999999999999998; // CO2
    TBid[11][11] = 32; TB[11][11] = 0; // C2H5O
    TBid[11][12] = 35; TB[11][12] = 0; // C3H2
    TBid[11][13] = 33; TB[11][13] = 0; // CH3CO
    TBid[11][14] = 50; TB[11][14] = 0; // C7H15X2

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
    nTB[0] = 15;
    TB[0] = (double *) malloc(15 * sizeof(double));
    TBid[0] = (int *) malloc(15 * sizeof(int));
    TBid[0][0] = 29; TB[0][0] = 0; // CH2CHO
    TBid[0][1] = 9; TB[0][1] = 0; // CH
    TBid[0][2] = 13; TB[0][2] = 1.8999999999999999; // CO
    TBid[0][3] = 2; TB[0][3] = 2.5; // H2
    TBid[0][4] = 44; TB[0][4] = 0; // PXC4H9
    TBid[0][5] = 15; TB[0][5] = 0; // CH2GSG
    TBid[0][6] = 42; TB[0][6] = 0; // C4H7
    TBid[0][7] = 27; TB[0][7] = 0; // HCCO
    TBid[0][8] = 48; TB[0][8] = 0; // C5H11X1
    TBid[0][9] = 5; TB[0][9] = 12; // H2O
    TBid[0][10] = 12; TB[0][10] = 3.7999999999999998; // CO2
    TBid[0][11] = 32; TB[0][11] = 0; // C2H5O
    TBid[0][12] = 35; TB[0][12] = 0; // C3H2
    TBid[0][13] = 33; TB[0][13] = 0; // CH3CO
    TBid[0][14] = 50; TB[0][14] = 0; // C7H15X2

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
    nTB[1] = 15;
    TB[1] = (double *) malloc(15 * sizeof(double));
    TBid[1] = (int *) malloc(15 * sizeof(int));
    TBid[1][0] = 29; TB[1][0] = 0; // CH2CHO
    TBid[1][1] = 9; TB[1][1] = 0; // CH
    TBid[1][2] = 13; TB[1][2] = 1.8999999999999999; // CO
    TBid[1][3] = 2; TB[1][3] = 2.5; // H2
    TBid[1][4] = 44; TB[1][4] = 0; // PXC4H9
    TBid[1][5] = 15; TB[1][5] = 0; // CH2GSG
    TBid[1][6] = 42; TB[1][6] = 0; // C4H7
    TBid[1][7] = 27; TB[1][7] = 0; // HCCO
    TBid[1][8] = 48; TB[1][8] = 0; // C5H11X1
    TBid[1][9] = 5; TB[1][9] = 12; // H2O
    TBid[1][10] = 12; TB[1][10] = 3.7999999999999998; // CO2
    TBid[1][11] = 32; TB[1][11] = 0; // C2H5O
    TBid[1][12] = 35; TB[1][12] = 0; // C3H2
    TBid[1][13] = 33; TB[1][13] = 0; // CH3CO
    TBid[1][14] = 50; TB[1][14] = 0; // C7H15X2

    // (18):  CH + O2 => HCO + O
    kiv[33] = {9,6,10,1};
    nuv[33] = {-1,-1,1,1};
    kiv_qss[33] = {};
    nuv_qss[33] = {};
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
    kiv[34] = {11,6,12,3};
    nuv[34] = {-1,-1,1,2.0};
    kiv_qss[34] = {};
    nuv_qss[34] = {};
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
    kiv[35] = {11,6,13,5};
    nuv[35] = {-1,-1,1,1};
    kiv_qss[35] = {};
    nuv_qss[35] = {};
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
    kiv[36] = {11,1,13,3};
    nuv[36] = {-1,-1,1,2.0};
    kiv_qss[36] = {};
    nuv_qss[36] = {};
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
    kiv[37] = {11,6,14,1};
    nuv[37] = {-1,-1,1,1};
    kiv_qss[37] = {};
    nuv_qss[37] = {};
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
    kiv[38] = {11,3,9,2};
    nuv[38] = {-1,-1,1,1};
    kiv_qss[38] = {};
    nuv_qss[38] = {};
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
    kiv[39] = {9,2,11,3};
    nuv[39] = {-1,-1,1,1};
    kiv_qss[39] = {};
    nuv_qss[39] = {};
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
    kiv[40] = {11,4,9,5};
    nuv[40] = {-1,-1,1,1};
    kiv_qss[40] = {};
    nuv_qss[40] = {};
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
    kiv[41] = {9,5,11,4};
    nuv[41] = {-1,-1,1,1};
    kiv_qss[41] = {};
    nuv_qss[41] = {};
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
    kiv[42] = {11,6,12,2};
    nuv[42] = {-1,-1,1,1};
    kiv_qss[42] = {};
    nuv_qss[42] = {};
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
    kiv[43] = {15,3,9,2};
    nuv[43] = {-1,-1,1,1};
    kiv_qss[43] = {};
    nuv_qss[43] = {};
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
    kiv[12] = {15,11};
    nuv[12] = {-1,1};
    kiv_qss[12] = {};
    nuv_qss[12] = {};
    // (29):  CH2GSG + M => CH2 + M
    fwd_A[12]     = 10000000000000;
    fwd_beta[12]  = 0;
    fwd_Ea[12]    = 0;
    prefactor_units[12]  = 1.0000000000000002e-06;
    activation_units[12] = 0.50321666580471969;
    phase_units[12]      = pow(10,-6.000000);
    is_PD[12] = 0;
    nTB[12] = 11;
    TB[12] = (double *) malloc(11 * sizeof(double));
    TBid[12] = (int *) malloc(11 * sizeof(int));
    TBid[12][0] = 29; TB[12][0] = 0; // CH2CHO
    TBid[12][1] = 9; TB[12][1] = 0; // CH
    TBid[12][2] = 35; TB[12][2] = 0; // C3H2
    TBid[12][3] = 44; TB[12][3] = 0; // PXC4H9
    TBid[12][4] = 15; TB[12][4] = 0; // CH2GSG
    TBid[12][5] = 42; TB[12][5] = 0; // C4H7
    TBid[12][6] = 27; TB[12][6] = 0; // HCCO
    TBid[12][7] = 48; TB[12][7] = 0; // C5H11X1
    TBid[12][8] = 33; TB[12][8] = 0; // CH3CO
    TBid[12][9] = 32; TB[12][9] = 0; // C2H5O
    TBid[12][10] = 50; TB[12][10] = 0; // C7H15X2

    // (30):  CH2 + M => CH2GSG + M
    kiv[13] = {11,15};
    nuv[13] = {-1,1};
    kiv_qss[13] = {};
    nuv_qss[13] = {};
    // (30):  CH2 + M => CH2GSG + M
    fwd_A[13]     = 7161000000000000;
    fwd_beta[13]  = -0.89000000000000001;
    fwd_Ea[13]    = 11429.969999999999;
    prefactor_units[13]  = 1.0000000000000002e-06;
    activation_units[13] = 0.50321666580471969;
    phase_units[13]      = pow(10,-6.000000);
    is_PD[13] = 0;
    nTB[13] = 11;
    TB[13] = (double *) malloc(11 * sizeof(double));
    TBid[13] = (int *) malloc(11 * sizeof(int));
    TBid[13][0] = 29; TB[13][0] = 0; // CH2CHO
    TBid[13][1] = 9; TB[13][1] = 0; // CH
    TBid[13][2] = 35; TB[13][2] = 0; // C3H2
    TBid[13][3] = 44; TB[13][3] = 0; // PXC4H9
    TBid[13][4] = 15; TB[13][4] = 0; // CH2GSG
    TBid[13][5] = 42; TB[13][5] = 0; // C4H7
    TBid[13][6] = 27; TB[13][6] = 0; // HCCO
    TBid[13][7] = 48; TB[13][7] = 0; // C5H11X1
    TBid[13][8] = 33; TB[13][8] = 0; // CH3CO
    TBid[13][9] = 32; TB[13][9] = 0; // C2H5O
    TBid[13][10] = 50; TB[13][10] = 0; // C7H15X2

    // (31):  CH2GSG + H2 => CH3 + H
    kiv[44] = {15,2,16,3};
    nuv[44] = {-1,-1,1,1};
    kiv_qss[44] = {};
    nuv_qss[44] = {};
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
    kiv[45] = {16,3,15,2};
    nuv[45] = {-1,-1,1,1};
    kiv_qss[45] = {};
    nuv_qss[45] = {};
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
    kiv[46] = {15,6,13,4,3};
    nuv[46] = {-1,-1,1,1,1};
    kiv_qss[46] = {};
    nuv_qss[46] = {};
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
    kiv[47] = {15,4,14,3};
    nuv[47] = {-1,-1,1,1};
    kiv_qss[47] = {};
    nuv_qss[47] = {};
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
    kiv[48] = {16,4,14,2};
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
    kiv[49] = {16,4,15,5};
    nuv[49] = {-1,-1,1,1};
    kiv_qss[49] = {};
    nuv_qss[49] = {};
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
    kiv[50] = {15,5,16,4};
    nuv[50] = {-1,-1,1,1};
    kiv_qss[50] = {};
    nuv_qss[50] = {};
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
    kiv[51] = {16,1,14,3};
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
    kiv[52] = {16,7,17,4};
    nuv[52] = {-1,-1,1,1};
    kiv_qss[52] = {};
    nuv_qss[52] = {};
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
    kiv[53] = {16,7,18,6};
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
    kiv[2] = {4,16,19};
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
    nTB[2] = 17;
    TB[2] = (double *) malloc(17 * sizeof(double));
    TBid[2] = (int *) malloc(17 * sizeof(int));
    TBid[2][0] = 29; TB[2][0] = 0; // CH2CHO
    TBid[2][1] = 9; TB[2][1] = 0; // CH
    TBid[2][2] = 13; TB[2][2] = 1.5; // CO
    TBid[2][3] = 2; TB[2][3] = 2; // H2
    TBid[2][4] = 44; TB[2][4] = 0; // PXC4H9
    TBid[2][5] = 15; TB[2][5] = 0; // CH2GSG
    TBid[2][6] = 42; TB[2][6] = 0; // C4H7
    TBid[2][7] = 27; TB[2][7] = 0; // HCCO
    TBid[2][8] = 48; TB[2][8] = 0; // C5H11X1
    TBid[2][9] = 5; TB[2][9] = 6; // H2O
    TBid[2][10] = 12; TB[2][10] = 2; // CO2
    TBid[2][11] = 18; TB[2][11] = 2; // CH4
    TBid[2][12] = 20; TB[2][12] = 3; // C2H6
    TBid[2][13] = 32; TB[2][13] = 0; // C2H5O
    TBid[2][14] = 35; TB[2][14] = 0; // C3H2
    TBid[2][15] = 33; TB[2][15] = 0; // CH3CO
    TBid[2][16] = 50; TB[2][16] = 0; // C7H15X2

    // (42):  CH3 + O2 => CH2O + OH
    kiv[54] = {16,6,14,4};
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
    kiv[3] = {16,3,18};
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
    nTB[3] = 15;
    TB[3] = (double *) malloc(15 * sizeof(double));
    TBid[3] = (int *) malloc(15 * sizeof(int));
    TBid[3][0] = 29; TB[3][0] = 0; // CH2CHO
    TBid[3][1] = 9; TB[3][1] = 0; // CH
    TBid[3][2] = 13; TB[3][2] = 2; // CO
    TBid[3][3] = 2; TB[3][3] = 2; // H2
    TBid[3][4] = 44; TB[3][4] = 0; // PXC4H9
    TBid[3][5] = 15; TB[3][5] = 0; // CH2GSG
    TBid[3][6] = 42; TB[3][6] = 0; // C4H7
    TBid[3][7] = 27; TB[3][7] = 0; // HCCO
    TBid[3][8] = 48; TB[3][8] = 0; // C5H11X1
    TBid[3][9] = 5; TB[3][9] = 5; // H2O
    TBid[3][10] = 12; TB[3][10] = 3; // CO2
    TBid[3][11] = 32; TB[3][11] = 0; // C2H5O
    TBid[3][12] = 35; TB[3][12] = 0; // C3H2
    TBid[3][13] = 33; TB[3][13] = 0; // CH3CO
    TBid[3][14] = 50; TB[3][14] = 0; // C7H15X2

    // (44):  CH3 + H => CH2 + H2
    kiv[55] = {16,3,11,2};
    nuv[55] = {-1,-1,1,1};
    kiv_qss[55] = {};
    nuv_qss[55] = {};
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
    kiv[56] = {11,2,16,3};
    nuv[56] = {-1,-1,1,1};
    kiv_qss[56] = {};
    nuv_qss[56] = {};
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
    kiv[4] = {16,20};
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
    nTB[4] = 15;
    TB[4] = (double *) malloc(15 * sizeof(double));
    TBid[4] = (int *) malloc(15 * sizeof(int));
    TBid[4][0] = 29; TB[4][0] = 0; // CH2CHO
    TBid[4][1] = 9; TB[4][1] = 0; // CH
    TBid[4][2] = 13; TB[4][2] = 2; // CO
    TBid[4][3] = 2; TB[4][3] = 2; // H2
    TBid[4][4] = 44; TB[4][4] = 0; // PXC4H9
    TBid[4][5] = 15; TB[4][5] = 0; // CH2GSG
    TBid[4][6] = 42; TB[4][6] = 0; // C4H7
    TBid[4][7] = 27; TB[4][7] = 0; // HCCO
    TBid[4][8] = 48; TB[4][8] = 0; // C5H11X1
    TBid[4][9] = 5; TB[4][9] = 5; // H2O
    TBid[4][10] = 12; TB[4][10] = 3; // CO2
    TBid[4][11] = 32; TB[4][11] = 0; // C2H5O
    TBid[4][12] = 35; TB[4][12] = 0; // C3H2
    TBid[4][13] = 33; TB[4][13] = 0; // CH3CO
    TBid[4][14] = 50; TB[4][14] = 0; // C7H15X2

    // (47):  2.000000 CH3 <=> H + C2H5
    kiv[57] = {16,3,21};
    nuv[57] = {-2.0,1,1};
    kiv_qss[57] = {};
    nuv_qss[57] = {};
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
    kiv[58] = {16,4,11,5};
    nuv[58] = {-1,-1,1,1};
    kiv_qss[58] = {};
    nuv_qss[58] = {};
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
    kiv[59] = {11,5,16,4};
    nuv[59] = {-1,-1,1,1};
    kiv_qss[59] = {};
    nuv_qss[59] = {};
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
    kiv[60] = {18,1,16,4};
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
    kiv[61] = {18,3,16,2};
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
    kiv[62] = {16,2,18,3};
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
    kiv[63] = {18,4,16,5};
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
    kiv[64] = {16,5,18,4};
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
    kiv[5] = {13,11,22};
    nuv[5] = {-1,-1,1};
    kiv_qss[5] = {};
    nuv_qss[5] = {};
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
    nTB[5] = 11;
    TB[5] = (double *) malloc(11 * sizeof(double));
    TBid[5] = (int *) malloc(11 * sizeof(int));
    TBid[5][0] = 29; TB[5][0] = 0; // CH2CHO
    TBid[5][1] = 9; TB[5][1] = 0; // CH
    TBid[5][2] = 35; TB[5][2] = 0; // C3H2
    TBid[5][3] = 44; TB[5][3] = 0; // PXC4H9
    TBid[5][4] = 15; TB[5][4] = 0; // CH2GSG
    TBid[5][5] = 42; TB[5][5] = 0; // C4H7
    TBid[5][6] = 27; TB[5][6] = 0; // HCCO
    TBid[5][7] = 48; TB[5][7] = 0; // C5H11X1
    TBid[5][8] = 33; TB[5][8] = 0; // CH3CO
    TBid[5][9] = 32; TB[5][9] = 0; // C2H5O
    TBid[5][10] = 50; TB[5][10] = 0; // C7H15X2

    // (56):  CO + O (+M) => CO2 (+M)
    kiv[6] = {13,1,12};
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
    nTB[6] = 15;
    TB[6] = (double *) malloc(15 * sizeof(double));
    TBid[6] = (int *) malloc(15 * sizeof(int));
    TBid[6][0] = 29; TB[6][0] = 0; // CH2CHO
    TBid[6][1] = 9; TB[6][1] = 0; // CH
    TBid[6][2] = 13; TB[6][2] = 1.8999999999999999; // CO
    TBid[6][3] = 2; TB[6][3] = 2.5; // H2
    TBid[6][4] = 44; TB[6][4] = 0; // PXC4H9
    TBid[6][5] = 15; TB[6][5] = 0; // CH2GSG
    TBid[6][6] = 42; TB[6][6] = 0; // C4H7
    TBid[6][7] = 27; TB[6][7] = 0; // HCCO
    TBid[6][8] = 48; TB[6][8] = 0; // C5H11X1
    TBid[6][9] = 5; TB[6][9] = 12; // H2O
    TBid[6][10] = 12; TB[6][10] = 3.7999999999999998; // CO2
    TBid[6][11] = 32; TB[6][11] = 0; // C2H5O
    TBid[6][12] = 35; TB[6][12] = 0; // C3H2
    TBid[6][13] = 33; TB[6][13] = 0; // CH3CO
    TBid[6][14] = 50; TB[6][14] = 0; // C7H15X2

    // (57):  CO + OH => CO2 + H
    kiv[65] = {13,4,12,3};
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
    kiv[66] = {12,3,13,4};
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
    kiv[67] = {10,6,13,7};
    nuv[67] = {-1,-1,1,1};
    kiv_qss[67] = {};
    nuv_qss[67] = {};
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
    kiv[68] = {10,1,12,3};
    nuv[68] = {-1,-1,1,1};
    kiv_qss[68] = {};
    nuv_qss[68] = {};
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
    kiv[69] = {10,4,13,5};
    nuv[69] = {-1,-1,1,1};
    kiv_qss[69] = {};
    nuv_qss[69] = {};
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
    kiv[70] = {10,3,13,2};
    nuv[70] = {-1,-1,1,1};
    kiv_qss[70] = {};
    nuv_qss[70] = {};
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
    kiv[71] = {10,1,13,4};
    nuv[71] = {-1,-1,1,1};
    kiv_qss[71] = {};
    nuv_qss[71] = {};
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
    kiv[14] = {10,3,13};
    nuv[14] = {-1,1,1};
    kiv_qss[14] = {};
    nuv_qss[14] = {};
    // (64):  HCO + M => H + CO + M
    fwd_A[14]     = 1.86e+17;
    fwd_beta[14]  = -1;
    fwd_Ea[14]    = 17000;
    prefactor_units[14]  = 1.0000000000000002e-06;
    activation_units[14] = 0.50321666580471969;
    phase_units[14]      = pow(10,-6.000000);
    is_PD[14] = 0;
    nTB[14] = 15;
    TB[14] = (double *) malloc(15 * sizeof(double));
    TBid[14] = (int *) malloc(15 * sizeof(int));
    TBid[14][0] = 29; TB[14][0] = 0; // CH2CHO
    TBid[14][1] = 9; TB[14][1] = 0; // CH
    TBid[14][2] = 13; TB[14][2] = 1.8999999999999999; // CO
    TBid[14][3] = 2; TB[14][3] = 2.5; // H2
    TBid[14][4] = 44; TB[14][4] = 0; // PXC4H9
    TBid[14][5] = 15; TB[14][5] = 0; // CH2GSG
    TBid[14][6] = 42; TB[14][6] = 0; // C4H7
    TBid[14][7] = 27; TB[14][7] = 0; // HCCO
    TBid[14][8] = 48; TB[14][8] = 0; // C5H11X1
    TBid[14][9] = 5; TB[14][9] = 6; // H2O
    TBid[14][10] = 12; TB[14][10] = 3.7999999999999998; // CO2
    TBid[14][11] = 32; TB[14][11] = 0; // C2H5O
    TBid[14][12] = 35; TB[14][12] = 0; // C3H2
    TBid[14][13] = 33; TB[14][13] = 0; // CH3CO
    TBid[14][14] = 50; TB[14][14] = 0; // C7H15X2

    // (65):  HCO + CH3 => CH4 + CO
    kiv[72] = {10,16,18,13};
    nuv[72] = {-1,-1,1,1};
    kiv_qss[72] = {};
    nuv_qss[72] = {};
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
    kiv[73] = {14,4,10,5};
    nuv[73] = {-1,-1,1,1};
    kiv_qss[73] = {};
    nuv_qss[73] = {};
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
    kiv[74] = {14,1,10,4};
    nuv[74] = {-1,-1,1,1};
    kiv_qss[74] = {};
    nuv_qss[74] = {};
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
    kiv[75] = {14,3,10,2};
    nuv[75] = {-1,-1,1,1};
    kiv_qss[75] = {};
    nuv_qss[75] = {};
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
    kiv[76] = {14,16,10,18};
    nuv[76] = {-1,-1,1,1};
    kiv_qss[76] = {};
    nuv_qss[76] = {};
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
    kiv[77] = {17,19,14};
    nuv[77] = {-2.0,1,1};
    kiv_qss[77] = {};
    nuv_qss[77] = {};
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
    kiv[78] = {17,6,14,7};
    nuv[78] = {-1,-1,1,1};
    kiv_qss[78] = {};
    nuv_qss[78] = {};
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
    kiv[7] = {17,14,3};
    nuv[7] = {-1,1,1};
    kiv_qss[7] = {};
    nuv_qss[7] = {};
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
    nTB[7] = 11;
    TB[7] = (double *) malloc(11 * sizeof(double));
    TBid[7] = (int *) malloc(11 * sizeof(int));
    TBid[7][0] = 29; TB[7][0] = 0; // CH2CHO
    TBid[7][1] = 9; TB[7][1] = 0; // CH
    TBid[7][2] = 35; TB[7][2] = 0; // C3H2
    TBid[7][3] = 44; TB[7][3] = 0; // PXC4H9
    TBid[7][4] = 15; TB[7][4] = 0; // CH2GSG
    TBid[7][5] = 42; TB[7][5] = 0; // C4H7
    TBid[7][6] = 27; TB[7][6] = 0; // HCCO
    TBid[7][7] = 48; TB[7][7] = 0; // C5H11X1
    TBid[7][8] = 33; TB[7][8] = 0; // CH3CO
    TBid[7][9] = 32; TB[7][9] = 0; // C2H5O
    TBid[7][10] = 50; TB[7][10] = 0; // C7H15X2

    // (73):  CH3O + H2 => CH3OH + H
    kiv[79] = {17,2,19,3};
    nuv[79] = {-1,-1,1,1};
    kiv_qss[79] = {};
    nuv_qss[79] = {};
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
    kiv[80] = {19,4,17,5};
    nuv[80] = {-1,-1,1,1};
    kiv_qss[80] = {};
    nuv_qss[80] = {};
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
    kiv[81] = {15,12,14,13};
    nuv[81] = {-1,-1,1,1};
    kiv_qss[81] = {};
    nuv_qss[81] = {};
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
    kiv[82] = {23,3,2,13,4};
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
    kiv[83] = {23,4,5,13,4};
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
    kiv[84] = {23,10,4};
    nuv[84] = {-1,1,1};
    kiv_qss[84] = {};
    nuv_qss[84] = {};
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
    kiv[85] = {10,4,23};
    nuv[85] = {-1,-1,1};
    kiv_qss[85] = {};
    nuv_qss[85] = {};
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
    kiv[86] = {23,3,2,12,3};
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
    kiv[87] = {23,4,5,12,3};
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

    // (82):  2.000000 CH3O2 => O2 + 2.000000 CH3O
    kiv[88] = {24,6,17};
    nuv[88] = {-2.0,1,2.0};
    kiv_qss[88] = {};
    nuv_qss[88] = {};
    // (82):  2.000000 CH3O2 => O2 + 2.000000 CH3O
    fwd_A[88]     = 14000000000000000;
    fwd_beta[88]  = -1.6100000000000001;
    fwd_Ea[88]    = 1859.9400000000001;
    prefactor_units[88]  = 1.0000000000000002e-06;
    activation_units[88] = 0.50321666580471969;
    phase_units[88]      = pow(10,-12.000000);
    is_PD[88] = 0;
    nTB[88] = 0;

    // (83):  CH3O2 + CH3 => 2.000000 CH3O
    kiv[89] = {24,16,17};
    nuv[89] = {-1,-1,2.0};
    kiv_qss[89] = {};
    nuv_qss[89] = {};
    // (83):  CH3O2 + CH3 => 2.000000 CH3O
    fwd_A[89]     = 7000000000000;
    fwd_beta[89]  = 0;
    fwd_Ea[89]    = -1000;
    prefactor_units[89]  = 1.0000000000000002e-06;
    activation_units[89] = 0.50321666580471969;
    phase_units[89]      = pow(10,-12.000000);
    is_PD[89] = 0;
    nTB[89] = 0;

    // (84):  2.000000 CH3O2 => CH2O + CH3OH + O2
    kiv[90] = {24,14,19,6};
    nuv[90] = {-2.0,1,1,1};
    kiv_qss[90] = {};
    nuv_qss[90] = {};
    // (84):  2.000000 CH3O2 => CH2O + CH3OH + O2
    fwd_A[90]     = 311000000000000;
    fwd_beta[90]  = -1.6100000000000001;
    fwd_Ea[90]    = -1050.9100000000001;
    prefactor_units[90]  = 1.0000000000000002e-06;
    activation_units[90] = 0.50321666580471969;
    phase_units[90]      = pow(10,-12.000000);
    is_PD[90] = 0;
    nTB[90] = 0;

    // (85):  CH3O2 + HO2 => CH3O2H + O2
    kiv[91] = {24,7,25,6};
    nuv[91] = {-1,-1,1,1};
    kiv_qss[91] = {};
    nuv_qss[91] = {};
    // (85):  CH3O2 + HO2 => CH3O2H + O2
    fwd_A[91]     = 17500000000;
    fwd_beta[91]  = 0;
    fwd_Ea[91]    = -3275.0999999999999;
    prefactor_units[91]  = 1.0000000000000002e-06;
    activation_units[91] = 0.50321666580471969;
    phase_units[91]      = pow(10,-12.000000);
    is_PD[91] = 0;
    nTB[91] = 0;

    // (86):  CH3O2 + M => CH3 + O2 + M
    kiv[15] = {24,16,6};
    nuv[15] = {-1,1,1};
    kiv_qss[15] = {};
    nuv_qss[15] = {};
    // (86):  CH3O2 + M => CH3 + O2 + M
    fwd_A[15]     = 4.3430000000000002e+27;
    fwd_beta[15]  = -3.4199999999999999;
    fwd_Ea[15]    = 30469.889999999999;
    prefactor_units[15]  = 1.0000000000000002e-06;
    activation_units[15] = 0.50321666580471969;
    phase_units[15]      = pow(10,-6.000000);
    is_PD[15] = 0;
    nTB[15] = 11;
    TB[15] = (double *) malloc(11 * sizeof(double));
    TBid[15] = (int *) malloc(11 * sizeof(int));
    TBid[15][0] = 29; TB[15][0] = 0; // CH2CHO
    TBid[15][1] = 9; TB[15][1] = 0; // CH
    TBid[15][2] = 35; TB[15][2] = 0; // C3H2
    TBid[15][3] = 44; TB[15][3] = 0; // PXC4H9
    TBid[15][4] = 15; TB[15][4] = 0; // CH2GSG
    TBid[15][5] = 42; TB[15][5] = 0; // C4H7
    TBid[15][6] = 27; TB[15][6] = 0; // HCCO
    TBid[15][7] = 48; TB[15][7] = 0; // C5H11X1
    TBid[15][8] = 33; TB[15][8] = 0; // CH3CO
    TBid[15][9] = 32; TB[15][9] = 0; // C2H5O
    TBid[15][10] = 50; TB[15][10] = 0; // C7H15X2

    // (87):  CH3 + O2 + M => CH3O2 + M
    kiv[16] = {16,6,24};
    nuv[16] = {-1,-1,1};
    kiv_qss[16] = {};
    nuv_qss[16] = {};
    // (87):  CH3 + O2 + M => CH3O2 + M
    fwd_A[16]     = 5.4400000000000001e+25;
    fwd_beta[16]  = -3.2999999999999998;
    fwd_Ea[16]    = 0;
    prefactor_units[16]  = 1.0000000000000002e-12;
    activation_units[16] = 0.50321666580471969;
    phase_units[16]      = pow(10,-12.000000);
    is_PD[16] = 0;
    nTB[16] = 11;
    TB[16] = (double *) malloc(11 * sizeof(double));
    TBid[16] = (int *) malloc(11 * sizeof(int));
    TBid[16][0] = 29; TB[16][0] = 0; // CH2CHO
    TBid[16][1] = 9; TB[16][1] = 0; // CH
    TBid[16][2] = 35; TB[16][2] = 0; // C3H2
    TBid[16][3] = 44; TB[16][3] = 0; // PXC4H9
    TBid[16][4] = 15; TB[16][4] = 0; // CH2GSG
    TBid[16][5] = 42; TB[16][5] = 0; // C4H7
    TBid[16][6] = 27; TB[16][6] = 0; // HCCO
    TBid[16][7] = 48; TB[16][7] = 0; // C5H11X1
    TBid[16][8] = 33; TB[16][8] = 0; // CH3CO
    TBid[16][9] = 32; TB[16][9] = 0; // C2H5O
    TBid[16][10] = 50; TB[16][10] = 0; // C7H15X2

    // (88):  CH3O2H => CH3O + OH
    kiv[92] = {25,17,4};
    nuv[92] = {-1,1,1};
    kiv_qss[92] = {};
    nuv_qss[92] = {};
    // (88):  CH3O2H => CH3O + OH
    fwd_A[92]     = 631000000000000;
    fwd_beta[92]  = 0;
    fwd_Ea[92]    = 42299.949999999997;
    prefactor_units[92]  = 1;
    activation_units[92] = 0.50321666580471969;
    phase_units[92]      = pow(10,-6.000000);
    is_PD[92] = 0;
    nTB[92] = 0;

    // (89):  C2H2 + O => CH2 + CO
    kiv[93] = {26,1,11,13};
    nuv[93] = {-1,-1,1,1};
    kiv_qss[93] = {};
    nuv_qss[93] = {};
    // (89):  C2H2 + O => CH2 + CO
    fwd_A[93]     = 6120000;
    fwd_beta[93]  = 2;
    fwd_Ea[93]    = 1900.0999999999999;
    prefactor_units[93]  = 1.0000000000000002e-06;
    activation_units[93] = 0.50321666580471969;
    phase_units[93]      = pow(10,-12.000000);
    is_PD[93] = 0;
    nTB[93] = 0;

    // (90):  C2H2 + O => HCCO + H
    kiv[94] = {26,1,27,3};
    nuv[94] = {-1,-1,1,1};
    kiv_qss[94] = {};
    nuv_qss[94] = {};
    // (90):  C2H2 + O => HCCO + H
    fwd_A[94]     = 14300000;
    fwd_beta[94]  = 2;
    fwd_Ea[94]    = 1900.0999999999999;
    prefactor_units[94]  = 1.0000000000000002e-06;
    activation_units[94] = 0.50321666580471969;
    phase_units[94]      = pow(10,-12.000000);
    is_PD[94] = 0;
    nTB[94] = 0;

    // (91):  C2H3 + H => C2H2 + H2
    kiv[95] = {28,3,26,2};
    nuv[95] = {-1,-1,1,1};
    kiv_qss[95] = {};
    nuv_qss[95] = {};
    // (91):  C2H3 + H => C2H2 + H2
    fwd_A[95]     = 20000000000000;
    fwd_beta[95]  = 0;
    fwd_Ea[95]    = 2500;
    prefactor_units[95]  = 1.0000000000000002e-06;
    activation_units[95] = 0.50321666580471969;
    phase_units[95]      = pow(10,-12.000000);
    is_PD[95] = 0;
    nTB[95] = 0;

    // (92):  C2H3 + O2 => CH2CHO + O
    kiv[96] = {28,6,29,1};
    nuv[96] = {-1,-1,1,1};
    kiv_qss[96] = {};
    nuv_qss[96] = {};
    // (92):  C2H3 + O2 => CH2CHO + O
    fwd_A[96]     = 350000000000000;
    fwd_beta[96]  = -0.60999999999999999;
    fwd_Ea[96]    = 5260.04;
    prefactor_units[96]  = 1.0000000000000002e-06;
    activation_units[96] = 0.50321666580471969;
    phase_units[96]      = pow(10,-12.000000);
    is_PD[96] = 0;
    nTB[96] = 0;

    // (93):  C2H3 + CH3 => C3H6
    kiv[97] = {28,16,30};
    nuv[97] = {-1,-1,1};
    kiv_qss[97] = {};
    nuv_qss[97] = {};
    // (93):  C2H3 + CH3 => C3H6
    fwd_A[97]     = 4.7119999999999996e+59;
    fwd_beta[97]  = -13.19;
    fwd_Ea[97]    = 29539.91;
    prefactor_units[97]  = 1.0000000000000002e-06;
    activation_units[97] = 0.50321666580471969;
    phase_units[97]      = pow(10,-12.000000);
    is_PD[97] = 0;
    nTB[97] = 0;

    // (94):  C2H3 + O2 => C2H2 + HO2
    kiv[98] = {28,6,26,7};
    nuv[98] = {-1,-1,1,1};
    kiv_qss[98] = {};
    nuv_qss[98] = {};
    // (94):  C2H3 + O2 => C2H2 + HO2
    fwd_A[98]     = 2.12e-06;
    fwd_beta[98]  = 6;
    fwd_Ea[98]    = 9483.9899999999998;
    prefactor_units[98]  = 1.0000000000000002e-06;
    activation_units[98] = 0.50321666580471969;
    phase_units[98]      = pow(10,-12.000000);
    is_PD[98] = 0;
    nTB[98] = 0;

    // (95):  C2H3 + O2 => CH2O + HCO
    kiv[99] = {28,6,14,10};
    nuv[99] = {-1,-1,1,1};
    kiv_qss[99] = {};
    nuv_qss[99] = {};
    // (95):  C2H3 + O2 => CH2O + HCO
    fwd_A[99]     = 1.6999999999999999e+29;
    fwd_beta[99]  = -5.3099999999999996;
    fwd_Ea[99]    = 6500;
    prefactor_units[99]  = 1.0000000000000002e-06;
    activation_units[99] = 0.50321666580471969;
    phase_units[99]      = pow(10,-12.000000);
    is_PD[99] = 0;
    nTB[99] = 0;

    // (96):  C2H3 (+M) => H + C2H2 (+M)
    kiv[8] = {28,3,26};
    nuv[8] = {-1,1,1};
    kiv_qss[8] = {};
    nuv_qss[8] = {};
    // (96):  C2H3 (+M) => H + C2H2 (+M)
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
    nTB[8] = 15;
    TB[8] = (double *) malloc(15 * sizeof(double));
    TBid[8] = (int *) malloc(15 * sizeof(int));
    TBid[8][0] = 29; TB[8][0] = 0; // CH2CHO
    TBid[8][1] = 9; TB[8][1] = 0; // CH
    TBid[8][2] = 13; TB[8][2] = 2; // CO
    TBid[8][3] = 2; TB[8][3] = 2; // H2
    TBid[8][4] = 44; TB[8][4] = 0; // PXC4H9
    TBid[8][5] = 15; TB[8][5] = 0; // CH2GSG
    TBid[8][6] = 42; TB[8][6] = 0; // C4H7
    TBid[8][7] = 27; TB[8][7] = 0; // HCCO
    TBid[8][8] = 48; TB[8][8] = 0; // C5H11X1
    TBid[8][9] = 5; TB[8][9] = 5; // H2O
    TBid[8][10] = 12; TB[8][10] = 3; // CO2
    TBid[8][11] = 32; TB[8][11] = 0; // C2H5O
    TBid[8][12] = 35; TB[8][12] = 0; // C3H2
    TBid[8][13] = 33; TB[8][13] = 0; // CH3CO
    TBid[8][14] = 50; TB[8][14] = 0; // C7H15X2

    // (97):  C2H4 + CH3 => C2H3 + CH4
    kiv[100] = {31,16,28,18};
    nuv[100] = {-1,-1,1,1};
    kiv_qss[100] = {};
    nuv_qss[100] = {};
    // (97):  C2H4 + CH3 => C2H3 + CH4
    fwd_A[100]     = 6.6200000000000001;
    fwd_beta[100]  = 3.7000000000000002;
    fwd_Ea[100]    = 9500;
    prefactor_units[100]  = 1.0000000000000002e-06;
    activation_units[100] = 0.50321666580471969;
    phase_units[100]      = pow(10,-12.000000);
    is_PD[100] = 0;
    nTB[100] = 0;

    // (98):  C2H4 + O => CH3 + HCO
    kiv[101] = {31,1,16,10};
    nuv[101] = {-1,-1,1,1};
    kiv_qss[101] = {};
    nuv_qss[101] = {};
    // (98):  C2H4 + O => CH3 + HCO
    fwd_A[101]     = 10200000;
    fwd_beta[101]  = 1.8799999999999999;
    fwd_Ea[101]    = 179.02000000000001;
    prefactor_units[101]  = 1.0000000000000002e-06;
    activation_units[101] = 0.50321666580471969;
    phase_units[101]      = pow(10,-12.000000);
    is_PD[101] = 0;
    nTB[101] = 0;

    // (99):  C2H4 + OH => C2H3 + H2O
    kiv[102] = {31,4,28,5};
    nuv[102] = {-1,-1,1,1};
    kiv_qss[102] = {};
    nuv_qss[102] = {};
    // (99):  C2H4 + OH => C2H3 + H2O
    fwd_A[102]     = 20500000000000;
    fwd_beta[102]  = 0;
    fwd_Ea[102]    = 5950.0500000000002;
    prefactor_units[102]  = 1.0000000000000002e-06;
    activation_units[102] = 0.50321666580471969;
    phase_units[102]      = pow(10,-12.000000);
    is_PD[102] = 0;
    nTB[102] = 0;

    // (100):  H + C2H4 (+M) <=> C2H5 (+M)
    kiv[9] = {3,31,21};
    nuv[9] = {-1,-1,1};
    kiv_qss[9] = {};
    nuv_qss[9] = {};
    // (100):  H + C2H4 (+M) <=> C2H5 (+M)
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
    nTB[9] = 11;
    TB[9] = (double *) malloc(11 * sizeof(double));
    TBid[9] = (int *) malloc(11 * sizeof(int));
    TBid[9][0] = 29; TB[9][0] = 0; // CH2CHO
    TBid[9][1] = 9; TB[9][1] = 0; // CH
    TBid[9][2] = 35; TB[9][2] = 0; // C3H2
    TBid[9][3] = 44; TB[9][3] = 0; // PXC4H9
    TBid[9][4] = 15; TB[9][4] = 0; // CH2GSG
    TBid[9][5] = 42; TB[9][5] = 0; // C4H7
    TBid[9][6] = 27; TB[9][6] = 0; // HCCO
    TBid[9][7] = 48; TB[9][7] = 0; // C5H11X1
    TBid[9][8] = 33; TB[9][8] = 0; // CH3CO
    TBid[9][9] = 32; TB[9][9] = 0; // C2H5O
    TBid[9][10] = 50; TB[9][10] = 0; // C7H15X2

    // (101):  C2H4 + O => CH2CHO + H
    kiv[103] = {31,1,29,3};
    nuv[103] = {-1,-1,1,1};
    kiv_qss[103] = {};
    nuv_qss[103] = {};
    // (101):  C2H4 + O => CH2CHO + H
    fwd_A[103]     = 3390000;
    fwd_beta[103]  = 1.8799999999999999;
    fwd_Ea[103]    = 179.02000000000001;
    prefactor_units[103]  = 1.0000000000000002e-06;
    activation_units[103] = 0.50321666580471969;
    phase_units[103]      = pow(10,-12.000000);
    is_PD[103] = 0;
    nTB[103] = 0;

    // (102):  C2H4 + H => C2H3 + H2
    kiv[104] = {31,3,28,2};
    nuv[104] = {-1,-1,1,1};
    kiv_qss[104] = {};
    nuv_qss[104] = {};
    // (102):  C2H4 + H => C2H3 + H2
    fwd_A[104]     = 0.0084200000000000004;
    fwd_beta[104]  = 4.6200000000000001;
    fwd_Ea[104]    = 2582.9299999999998;
    prefactor_units[104]  = 1.0000000000000002e-06;
    activation_units[104] = 0.50321666580471969;
    phase_units[104]      = pow(10,-12.000000);
    is_PD[104] = 0;
    nTB[104] = 0;

    // (103):  C2H3 + H2 => C2H4 + H
    kiv[105] = {28,2,31,3};
    nuv[105] = {-1,-1,1,1};
    kiv_qss[105] = {};
    nuv_qss[105] = {};
    // (103):  C2H3 + H2 => C2H4 + H
    fwd_A[105]     = 0.57230000000000003;
    fwd_beta[105]  = 3.79;
    fwd_Ea[105]    = 3233.0300000000002;
    prefactor_units[105]  = 1.0000000000000002e-06;
    activation_units[105] = 0.50321666580471969;
    phase_units[105]      = pow(10,-12.000000);
    is_PD[105] = 0;
    nTB[105] = 0;

    // (104):  H + C2H5 => C2H6
    kiv[106] = {3,21,20};
    nuv[106] = {-1,-1,1};
    kiv_qss[106] = {};
    nuv_qss[106] = {};
    // (104):  H + C2H5 => C2H6
    fwd_A[106]     = 583100000000;
    fwd_beta[106]  = 0.59899999999999998;
    fwd_Ea[106]    = -2913;
    prefactor_units[106]  = 1.0000000000000002e-06;
    activation_units[106] = 0.50321666580471969;
    phase_units[106]      = pow(10,-12.000000);
    is_PD[106] = 0;
    nTB[106] = 0;

    // (105):  CH3O2 + C2H5 => CH3O + C2H5O
    kiv[107] = {24,21,17,32};
    nuv[107] = {-1,-1,1,1};
    kiv_qss[107] = {};
    nuv_qss[107] = {};
    // (105):  CH3O2 + C2H5 => CH3O + C2H5O
    fwd_A[107]     = 7000000000000;
    fwd_beta[107]  = 0;
    fwd_Ea[107]    = -1000;
    prefactor_units[107]  = 1.0000000000000002e-06;
    activation_units[107] = 0.50321666580471969;
    phase_units[107]      = pow(10,-12.000000);
    is_PD[107] = 0;
    nTB[107] = 0;

    // (106):  C2H5 + HO2 => C2H5O + OH
    kiv[108] = {21,7,32,4};
    nuv[108] = {-1,-1,1,1};
    kiv_qss[108] = {};
    nuv_qss[108] = {};
    // (106):  C2H5 + HO2 => C2H5O + OH
    fwd_A[108]     = 32000000000000;
    fwd_beta[108]  = 0;
    fwd_Ea[108]    = 0;
    prefactor_units[108]  = 1.0000000000000002e-06;
    activation_units[108] = 0.50321666580471969;
    phase_units[108]      = pow(10,-12.000000);
    is_PD[108] = 0;
    nTB[108] = 0;

    // (107):  C2H5 + O2 => C2H4 + HO2
    kiv[109] = {21,6,31,7};
    nuv[109] = {-1,-1,1,1};
    kiv_qss[109] = {};
    nuv_qss[109] = {};
    // (107):  C2H5 + O2 => C2H4 + HO2
    fwd_A[109]     = 1.22e+30;
    fwd_beta[109]  = -5.7599999999999998;
    fwd_Ea[109]    = 10099.9;
    prefactor_units[109]  = 1.0000000000000002e-06;
    activation_units[109] = 0.50321666580471969;
    phase_units[109]      = pow(10,-12.000000);
    is_PD[109] = 0;
    nTB[109] = 0;

    // (108):  C2H6 + O => C2H5 + OH
    kiv[110] = {20,1,21,4};
    nuv[110] = {-1,-1,1,1};
    kiv_qss[110] = {};
    nuv_qss[110] = {};
    // (108):  C2H6 + O => C2H5 + OH
    fwd_A[110]     = 13000000;
    fwd_beta[110]  = 2.1299999999999999;
    fwd_Ea[110]    = 5190.0100000000002;
    prefactor_units[110]  = 1.0000000000000002e-06;
    activation_units[110] = 0.50321666580471969;
    phase_units[110]      = pow(10,-12.000000);
    is_PD[110] = 0;
    nTB[110] = 0;

    // (109):  C2H6 + OH => C2H5 + H2O
    kiv[111] = {20,4,21,5};
    nuv[111] = {-1,-1,1,1};
    kiv_qss[111] = {};
    nuv_qss[111] = {};
    // (109):  C2H6 + OH => C2H5 + H2O
    fwd_A[111]     = 58000000;
    fwd_beta[111]  = 1.73;
    fwd_Ea[111]    = 1159.8900000000001;
    prefactor_units[111]  = 1.0000000000000002e-06;
    activation_units[111] = 0.50321666580471969;
    phase_units[111]      = pow(10,-12.000000);
    is_PD[111] = 0;
    nTB[111] = 0;

    // (110):  C2H6 + H => C2H5 + H2
    kiv[112] = {20,3,21,2};
    nuv[112] = {-1,-1,1,1};
    kiv_qss[112] = {};
    nuv_qss[112] = {};
    // (110):  C2H6 + H => C2H5 + H2
    fwd_A[112]     = 554;
    fwd_beta[112]  = 3.5;
    fwd_Ea[112]    = 5167.0699999999997;
    prefactor_units[112]  = 1.0000000000000002e-06;
    activation_units[112] = 0.50321666580471969;
    phase_units[112]      = pow(10,-12.000000);
    is_PD[112] = 0;
    nTB[112] = 0;

    // (111):  HCCO + O => H + 2.000000 CO
    kiv[113] = {27,1,3,13};
    nuv[113] = {-1,-1,1,2.0};
    kiv_qss[113] = {};
    nuv_qss[113] = {};
    // (111):  HCCO + O => H + 2.000000 CO
    fwd_A[113]     = 80000000000000;
    fwd_beta[113]  = 0;
    fwd_Ea[113]    = 0;
    prefactor_units[113]  = 1.0000000000000002e-06;
    activation_units[113] = 0.50321666580471969;
    phase_units[113]      = pow(10,-12.000000);
    is_PD[113] = 0;
    nTB[113] = 0;

    // (112):  HCCO + OH => 2.000000 HCO
    kiv[114] = {27,4,10};
    nuv[114] = {-1,-1,2.0};
    kiv_qss[114] = {};
    nuv_qss[114] = {};
    // (112):  HCCO + OH => 2.000000 HCO
    fwd_A[114]     = 10000000000000;
    fwd_beta[114]  = 0;
    fwd_Ea[114]    = 0;
    prefactor_units[114]  = 1.0000000000000002e-06;
    activation_units[114] = 0.50321666580471969;
    phase_units[114]      = pow(10,-12.000000);
    is_PD[114] = 0;
    nTB[114] = 0;

    // (113):  HCCO + O2 => CO2 + HCO
    kiv[115] = {27,6,12,10};
    nuv[115] = {-1,-1,1,1};
    kiv_qss[115] = {};
    nuv_qss[115] = {};
    // (113):  HCCO + O2 => CO2 + HCO
    fwd_A[115]     = 240000000000;
    fwd_beta[115]  = 0;
    fwd_Ea[115]    = -853.97000000000003;
    prefactor_units[115]  = 1.0000000000000002e-06;
    activation_units[115] = 0.50321666580471969;
    phase_units[115]      = pow(10,-12.000000);
    is_PD[115] = 0;
    nTB[115] = 0;

    // (114):  HCCO + H => CH2GSG + CO
    kiv[116] = {27,3,15,13};
    nuv[116] = {-1,-1,1,1};
    kiv_qss[116] = {};
    nuv_qss[116] = {};
    // (114):  HCCO + H => CH2GSG + CO
    fwd_A[116]     = 110000000000000;
    fwd_beta[116]  = 0;
    fwd_Ea[116]    = 0;
    prefactor_units[116]  = 1.0000000000000002e-06;
    activation_units[116] = 0.50321666580471969;
    phase_units[116]      = pow(10,-12.000000);
    is_PD[116] = 0;
    nTB[116] = 0;

    // (115):  CH2GSG + CO => HCCO + H
    kiv[117] = {15,13,27,3};
    nuv[117] = {-1,-1,1,1};
    kiv_qss[117] = {};
    nuv_qss[117] = {};
    // (115):  CH2GSG + CO => HCCO + H
    fwd_A[117]     = 2046000000000;
    fwd_beta[117]  = 0.89000000000000001;
    fwd_Ea[117]    = 27830.07;
    prefactor_units[117]  = 1.0000000000000002e-06;
    activation_units[117] = 0.50321666580471969;
    phase_units[117]      = pow(10,-12.000000);
    is_PD[117] = 0;
    nTB[117] = 0;

    // (116):  CH2CO + O => HCCO + OH
    kiv[118] = {22,1,27,4};
    nuv[118] = {-1,-1,1,1};
    kiv_qss[118] = {};
    nuv_qss[118] = {};
    // (116):  CH2CO + O => HCCO + OH
    fwd_A[118]     = 10000000000000;
    fwd_beta[118]  = 0;
    fwd_Ea[118]    = 8000;
    prefactor_units[118]  = 1.0000000000000002e-06;
    activation_units[118] = 0.50321666580471969;
    phase_units[118]      = pow(10,-12.000000);
    is_PD[118] = 0;
    nTB[118] = 0;

    // (117):  CH2CO + H => HCCO + H2
    kiv[119] = {22,3,27,2};
    nuv[119] = {-1,-1,1,1};
    kiv_qss[119] = {};
    nuv_qss[119] = {};
    // (117):  CH2CO + H => HCCO + H2
    fwd_A[119]     = 200000000000000;
    fwd_beta[119]  = 0;
    fwd_Ea[119]    = 8000;
    prefactor_units[119]  = 1.0000000000000002e-06;
    activation_units[119] = 0.50321666580471969;
    phase_units[119]      = pow(10,-12.000000);
    is_PD[119] = 0;
    nTB[119] = 0;

    // (118):  HCCO + H2 => CH2CO + H
    kiv[120] = {27,2,22,3};
    nuv[120] = {-1,-1,1,1};
    kiv_qss[120] = {};
    nuv_qss[120] = {};
    // (118):  HCCO + H2 => CH2CO + H
    fwd_A[120]     = 652200000000;
    fwd_beta[120]  = 0;
    fwd_Ea[120]    = 840.11000000000001;
    prefactor_units[120]  = 1.0000000000000002e-06;
    activation_units[120] = 0.50321666580471969;
    phase_units[120]      = pow(10,-12.000000);
    is_PD[120] = 0;
    nTB[120] = 0;

    // (119):  CH2CO + H => CH3 + CO
    kiv[121] = {22,3,16,13};
    nuv[121] = {-1,-1,1,1};
    kiv_qss[121] = {};
    nuv_qss[121] = {};
    // (119):  CH2CO + H => CH3 + CO
    fwd_A[121]     = 11000000000000;
    fwd_beta[121]  = 0;
    fwd_Ea[121]    = 3400.0999999999999;
    prefactor_units[121]  = 1.0000000000000002e-06;
    activation_units[121] = 0.50321666580471969;
    phase_units[121]      = pow(10,-12.000000);
    is_PD[121] = 0;
    nTB[121] = 0;

    // (120):  CH2CO + O => CH2 + CO2
    kiv[122] = {22,1,11,12};
    nuv[122] = {-1,-1,1,1};
    kiv_qss[122] = {};
    nuv_qss[122] = {};
    // (120):  CH2CO + O => CH2 + CO2
    fwd_A[122]     = 1750000000000;
    fwd_beta[122]  = 0;
    fwd_Ea[122]    = 1349.9000000000001;
    prefactor_units[122]  = 1.0000000000000002e-06;
    activation_units[122] = 0.50321666580471969;
    phase_units[122]      = pow(10,-12.000000);
    is_PD[122] = 0;
    nTB[122] = 0;

    // (121):  CH2CO + OH => HCCO + H2O
    kiv[123] = {22,4,27,5};
    nuv[123] = {-1,-1,1,1};
    kiv_qss[123] = {};
    nuv_qss[123] = {};
    // (121):  CH2CO + OH => HCCO + H2O
    fwd_A[123]     = 10000000000000;
    fwd_beta[123]  = 0;
    fwd_Ea[123]    = 2000;
    prefactor_units[123]  = 1.0000000000000002e-06;
    activation_units[123] = 0.50321666580471969;
    phase_units[123]      = pow(10,-12.000000);
    is_PD[123] = 0;
    nTB[123] = 0;

    // (122):  CH2CHO + O2 => CH2O + CO + OH
    kiv[124] = {29,6,14,13,4};
    nuv[124] = {-1,-1,1,1,1};
    kiv_qss[124] = {};
    nuv_qss[124] = {};
    // (122):  CH2CHO + O2 => CH2O + CO + OH
    fwd_A[124]     = 20000000000000;
    fwd_beta[124]  = 0;
    fwd_Ea[124]    = 4200.0500000000002;
    prefactor_units[124]  = 1.0000000000000002e-06;
    activation_units[124] = 0.50321666580471969;
    phase_units[124]      = pow(10,-12.000000);
    is_PD[124] = 0;
    nTB[124] = 0;

    // (123):  CH2CHO => CH2CO + H
    kiv[125] = {29,22,3};
    nuv[125] = {-1,1,1};
    kiv_qss[125] = {};
    nuv_qss[125] = {};
    // (123):  CH2CHO => CH2CO + H
    fwd_A[125]     = 3094000000000000;
    fwd_beta[125]  = -0.26000000000000001;
    fwd_Ea[125]    = 50820.029999999999;
    prefactor_units[125]  = 1;
    activation_units[125] = 0.50321666580471969;
    phase_units[125]      = pow(10,-6.000000);
    is_PD[125] = 0;
    nTB[125] = 0;

    // (124):  CH2CO + H => CH2CHO
    kiv[126] = {22,3,29};
    nuv[126] = {-1,-1,1};
    kiv_qss[126] = {};
    nuv_qss[126] = {};
    // (124):  CH2CO + H => CH2CHO
    fwd_A[126]     = 50000000000000;
    fwd_beta[126]  = 0;
    fwd_Ea[126]    = 12299.950000000001;
    prefactor_units[126]  = 1.0000000000000002e-06;
    activation_units[126] = 0.50321666580471969;
    phase_units[126]      = pow(10,-12.000000);
    is_PD[126] = 0;
    nTB[126] = 0;

    // (125):  CH3CO (+M) => CH3 + CO (+M)
    kiv[10] = {33,16,13};
    nuv[10] = {-1,1,1};
    kiv_qss[10] = {};
    nuv_qss[10] = {};
    // (125):  CH3CO (+M) => CH3 + CO (+M)
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
    nTB[10] = 11;
    TB[10] = (double *) malloc(11 * sizeof(double));
    TBid[10] = (int *) malloc(11 * sizeof(int));
    TBid[10][0] = 29; TB[10][0] = 0; // CH2CHO
    TBid[10][1] = 9; TB[10][1] = 0; // CH
    TBid[10][2] = 35; TB[10][2] = 0; // C3H2
    TBid[10][3] = 44; TB[10][3] = 0; // PXC4H9
    TBid[10][4] = 15; TB[10][4] = 0; // CH2GSG
    TBid[10][5] = 42; TB[10][5] = 0; // C4H7
    TBid[10][6] = 27; TB[10][6] = 0; // HCCO
    TBid[10][7] = 48; TB[10][7] = 0; // C5H11X1
    TBid[10][8] = 33; TB[10][8] = 0; // CH3CO
    TBid[10][9] = 32; TB[10][9] = 0; // C2H5O
    TBid[10][10] = 50; TB[10][10] = 0; // C7H15X2

    // (126):  C2H5O + M => CH3 + CH2O + M
    kiv[17] = {32,16,14};
    nuv[17] = {-1,1,1};
    kiv_qss[17] = {};
    nuv_qss[17] = {};
    // (126):  C2H5O + M => CH3 + CH2O + M
    fwd_A[17]     = 1.35e+38;
    fwd_beta[17]  = -6.96;
    fwd_Ea[17]    = 23799.950000000001;
    prefactor_units[17]  = 1.0000000000000002e-06;
    activation_units[17] = 0.50321666580471969;
    phase_units[17]      = pow(10,-6.000000);
    is_PD[17] = 0;
    nTB[17] = 11;
    TB[17] = (double *) malloc(11 * sizeof(double));
    TBid[17] = (int *) malloc(11 * sizeof(int));
    TBid[17][0] = 29; TB[17][0] = 0; // CH2CHO
    TBid[17][1] = 9; TB[17][1] = 0; // CH
    TBid[17][2] = 35; TB[17][2] = 0; // C3H2
    TBid[17][3] = 44; TB[17][3] = 0; // PXC4H9
    TBid[17][4] = 15; TB[17][4] = 0; // CH2GSG
    TBid[17][5] = 42; TB[17][5] = 0; // C4H7
    TBid[17][6] = 27; TB[17][6] = 0; // HCCO
    TBid[17][7] = 48; TB[17][7] = 0; // C5H11X1
    TBid[17][8] = 33; TB[17][8] = 0; // CH3CO
    TBid[17][9] = 32; TB[17][9] = 0; // C2H5O
    TBid[17][10] = 50; TB[17][10] = 0; // C7H15X2

    // (127):  C2H5O2 => C2H5 + O2
    kiv[127] = {34,21,6};
    nuv[127] = {-1,1,1};
    kiv_qss[127] = {};
    nuv_qss[127] = {};
    // (127):  C2H5O2 => C2H5 + O2
    fwd_A[127]     = 4.9299999999999996e+50;
    fwd_beta[127]  = -11.5;
    fwd_Ea[127]    = 42250;
    prefactor_units[127]  = 1;
    activation_units[127] = 0.50321666580471969;
    phase_units[127]      = pow(10,-6.000000);
    is_PD[127] = 0;
    nTB[127] = 0;

    // (128):  C2H5 + O2 => C2H5O2
    kiv[128] = {21,6,34};
    nuv[128] = {-1,-1,1};
    kiv_qss[128] = {};
    nuv_qss[128] = {};
    // (128):  C2H5 + O2 => C2H5O2
    fwd_A[128]     = 1.0900000000000001e+48;
    fwd_beta[128]  = -11.539999999999999;
    fwd_Ea[128]    = 10219.889999999999;
    prefactor_units[128]  = 1.0000000000000002e-06;
    activation_units[128] = 0.50321666580471969;
    phase_units[128]      = pow(10,-12.000000);
    is_PD[128] = 0;
    nTB[128] = 0;

    // (129):  C2H5O2 => C2H4 + HO2
    kiv[129] = {34,31,7};
    nuv[129] = {-1,1,1};
    kiv_qss[129] = {};
    nuv_qss[129] = {};
    // (129):  C2H5O2 => C2H4 + HO2
    fwd_A[129]     = 3.3700000000000002e+55;
    fwd_beta[129]  = -13.42;
    fwd_Ea[129]    = 44669.93;
    prefactor_units[129]  = 1;
    activation_units[129] = 0.50321666580471969;
    phase_units[129]      = pow(10,-6.000000);
    is_PD[129] = 0;
    nTB[129] = 0;

    // (130):  C3H2 + O2 => HCCO + CO + H
    kiv[130] = {35,6,27,13,3};
    nuv[130] = {-1,-1,1,1,1};
    kiv_qss[130] = {};
    nuv_qss[130] = {};
    // (130):  C3H2 + O2 => HCCO + CO + H
    fwd_A[130]     = 50000000000000;
    fwd_beta[130]  = 0;
    fwd_Ea[130]    = 0;
    prefactor_units[130]  = 1.0000000000000002e-06;
    activation_units[130] = 0.50321666580471969;
    phase_units[130]      = pow(10,-12.000000);
    is_PD[130] = 0;
    nTB[130] = 0;

    // (131):  C3H2 + OH => C2H2 + HCO
    kiv[131] = {35,4,26,10};
    nuv[131] = {-1,-1,1,1};
    kiv_qss[131] = {};
    nuv_qss[131] = {};
    // (131):  C3H2 + OH => C2H2 + HCO
    fwd_A[131]     = 50000000000000;
    fwd_beta[131]  = 0;
    fwd_Ea[131]    = 0;
    prefactor_units[131]  = 1.0000000000000002e-06;
    activation_units[131] = 0.50321666580471969;
    phase_units[131]      = pow(10,-12.000000);
    is_PD[131] = 0;
    nTB[131] = 0;

    // (132):  C3H3 + O2 => CH2CO + HCO
    kiv[132] = {36,6,22,10};
    nuv[132] = {-1,-1,1,1};
    kiv_qss[132] = {};
    nuv_qss[132] = {};
    // (132):  C3H3 + O2 => CH2CO + HCO
    fwd_A[132]     = 30100000000;
    fwd_beta[132]  = 0;
    fwd_Ea[132]    = 2869.98;
    prefactor_units[132]  = 1.0000000000000002e-06;
    activation_units[132] = 0.50321666580471969;
    phase_units[132]      = pow(10,-12.000000);
    is_PD[132] = 0;
    nTB[132] = 0;

    // (133):  C3H3 + HO2 => C3H4XA + O2
    kiv[133] = {36,7,37,6};
    nuv[133] = {-1,-1,1,1};
    kiv_qss[133] = {};
    nuv_qss[133] = {};
    // (133):  C3H3 + HO2 => C3H4XA + O2
    fwd_A[133]     = 117500000000;
    fwd_beta[133]  = 0.29999999999999999;
    fwd_Ea[133]    = 38;
    prefactor_units[133]  = 1.0000000000000002e-06;
    activation_units[133] = 0.50321666580471969;
    phase_units[133]      = pow(10,-12.000000);
    is_PD[133] = 0;
    nTB[133] = 0;

    // (134):  C3H3 + H => C3H2 + H2
    kiv[134] = {36,3,35,2};
    nuv[134] = {-1,-1,1,1};
    kiv_qss[134] = {};
    nuv_qss[134] = {};
    // (134):  C3H3 + H => C3H2 + H2
    fwd_A[134]     = 50000000000000;
    fwd_beta[134]  = 0;
    fwd_Ea[134]    = 0;
    prefactor_units[134]  = 1.0000000000000002e-06;
    activation_units[134] = 0.50321666580471969;
    phase_units[134]      = pow(10,-12.000000);
    is_PD[134] = 0;
    nTB[134] = 0;

    // (135):  C3H3 + OH => C3H2 + H2O
    kiv[135] = {36,4,35,5};
    nuv[135] = {-1,-1,1,1};
    kiv_qss[135] = {};
    nuv_qss[135] = {};
    // (135):  C3H3 + OH => C3H2 + H2O
    fwd_A[135]     = 10000000000000;
    fwd_beta[135]  = 0;
    fwd_Ea[135]    = 0;
    prefactor_units[135]  = 1.0000000000000002e-06;
    activation_units[135] = 0.50321666580471969;
    phase_units[135]      = pow(10,-12.000000);
    is_PD[135] = 0;
    nTB[135] = 0;

    // (136):  C3H2 + H2O => C3H3 + OH
    kiv[136] = {35,5,36,4};
    nuv[136] = {-1,-1,1,1};
    kiv_qss[136] = {};
    nuv_qss[136] = {};
    // (136):  C3H2 + H2O => C3H3 + OH
    fwd_A[136]     = 1343000000000000;
    fwd_beta[136]  = 0;
    fwd_Ea[136]    = 15679.969999999999;
    prefactor_units[136]  = 1.0000000000000002e-06;
    activation_units[136] = 0.50321666580471969;
    phase_units[136]      = pow(10,-12.000000);
    is_PD[136] = 0;
    nTB[136] = 0;

    // (137):  C3H4XA + H => C3H3 + H2
    kiv[137] = {37,3,36,2};
    nuv[137] = {-1,-1,1,1};
    kiv_qss[137] = {};
    nuv_qss[137] = {};
    // (137):  C3H4XA + H => C3H3 + H2
    fwd_A[137]     = 20000000;
    fwd_beta[137]  = 2;
    fwd_Ea[137]    = 5000;
    prefactor_units[137]  = 1.0000000000000002e-06;
    activation_units[137] = 0.50321666580471969;
    phase_units[137]      = pow(10,-12.000000);
    is_PD[137] = 0;
    nTB[137] = 0;

    // (138):  C3H4XA + OH => C3H3 + H2O
    kiv[138] = {37,4,36,5};
    nuv[138] = {-1,-1,1,1};
    kiv_qss[138] = {};
    nuv_qss[138] = {};
    // (138):  C3H4XA + OH => C3H3 + H2O
    fwd_A[138]     = 10000000;
    fwd_beta[138]  = 2;
    fwd_Ea[138]    = 1000;
    prefactor_units[138]  = 1.0000000000000002e-06;
    activation_units[138] = 0.50321666580471969;
    phase_units[138]      = pow(10,-12.000000);
    is_PD[138] = 0;
    nTB[138] = 0;

    // (139):  C3H4XA + O => C2H4 + CO
    kiv[139] = {37,1,31,13};
    nuv[139] = {-1,-1,1,1};
    kiv_qss[139] = {};
    nuv_qss[139] = {};
    // (139):  C3H4XA + O => C2H4 + CO
    fwd_A[139]     = 7800000000000;
    fwd_beta[139]  = 0;
    fwd_Ea[139]    = 1599.9000000000001;
    prefactor_units[139]  = 1.0000000000000002e-06;
    activation_units[139] = 0.50321666580471969;
    phase_units[139]      = pow(10,-12.000000);
    is_PD[139] = 0;
    nTB[139] = 0;

    // (140):  C3H5XA + H => C3H4XA + H2
    kiv[140] = {38,3,37,2};
    nuv[140] = {-1,-1,1,1};
    kiv_qss[140] = {};
    nuv_qss[140] = {};
    // (140):  C3H5XA + H => C3H4XA + H2
    fwd_A[140]     = 18100000000000;
    fwd_beta[140]  = 0;
    fwd_Ea[140]    = 0;
    prefactor_units[140]  = 1.0000000000000002e-06;
    activation_units[140] = 0.50321666580471969;
    phase_units[140]      = pow(10,-12.000000);
    is_PD[140] = 0;
    nTB[140] = 0;

    // (141):  C3H5XA + HO2 => C3H6 + O2
    kiv[141] = {38,7,30,6};
    nuv[141] = {-1,-1,1,1};
    kiv_qss[141] = {};
    nuv_qss[141] = {};
    // (141):  C3H5XA + HO2 => C3H6 + O2
    fwd_A[141]     = 33320000000;
    fwd_beta[141]  = 0.34000000000000002;
    fwd_Ea[141]    = -555.92999999999995;
    prefactor_units[141]  = 1.0000000000000002e-06;
    activation_units[141] = 0.50321666580471969;
    phase_units[141]      = pow(10,-12.000000);
    is_PD[141] = 0;
    nTB[141] = 0;

    // (142):  C3H5XA + H => C3H6
    kiv[142] = {38,3,30};
    nuv[142] = {-1,-1,1};
    kiv_qss[142] = {};
    nuv_qss[142] = {};
    // (142):  C3H5XA + H => C3H6
    fwd_A[142]     = 4.8869999999999999e+56;
    fwd_beta[142]  = -12.25;
    fwd_Ea[142]    = 28080.07;
    prefactor_units[142]  = 1.0000000000000002e-06;
    activation_units[142] = 0.50321666580471969;
    phase_units[142]      = pow(10,-12.000000);
    is_PD[142] = 0;
    nTB[142] = 0;

    // (143):  C3H5XA => C2H2 + CH3
    kiv[143] = {38,26,16};
    nuv[143] = {-1,1,1};
    kiv_qss[143] = {};
    nuv_qss[143] = {};
    // (143):  C3H5XA => C2H2 + CH3
    fwd_A[143]     = 2.397e+48;
    fwd_beta[143]  = -9.9000000000000004;
    fwd_Ea[143]    = 82080.070000000007;
    prefactor_units[143]  = 1;
    activation_units[143] = 0.50321666580471969;
    phase_units[143]      = pow(10,-6.000000);
    is_PD[143] = 0;
    nTB[143] = 0;

    // (144):  C3H5XA => C3H4XA + H
    kiv[144] = {38,37,3};
    nuv[144] = {-1,1,1};
    kiv_qss[144] = {};
    nuv_qss[144] = {};
    // (144):  C3H5XA => C3H4XA + H
    fwd_A[144]     = 6663000000000000;
    fwd_beta[144]  = -0.42999999999999999;
    fwd_Ea[144]    = 63219.889999999999;
    prefactor_units[144]  = 1;
    activation_units[144] = 0.50321666580471969;
    phase_units[144]      = pow(10,-6.000000);
    is_PD[144] = 0;
    nTB[144] = 0;

    // (145):  C3H4XA + H => C3H5XA
    kiv[145] = {37,3,38};
    nuv[145] = {-1,-1,1};
    kiv_qss[145] = {};
    nuv_qss[145] = {};
    // (145):  C3H4XA + H => C3H5XA
    fwd_A[145]     = 240000000000;
    fwd_beta[145]  = 0.68999999999999995;
    fwd_Ea[145]    = 3006.9299999999998;
    prefactor_units[145]  = 1.0000000000000002e-06;
    activation_units[145] = 0.50321666580471969;
    phase_units[145]      = pow(10,-12.000000);
    is_PD[145] = 0;
    nTB[145] = 0;

    // (146):  C3H5XA + CH2O => C3H6 + HCO
    kiv[146] = {38,14,30,10};
    nuv[146] = {-1,-1,1,1};
    kiv_qss[146] = {};
    nuv_qss[146] = {};
    // (146):  C3H5XA + CH2O => C3H6 + HCO
    fwd_A[146]     = 630000000;
    fwd_beta[146]  = 1.8999999999999999;
    fwd_Ea[146]    = 18190.009999999998;
    prefactor_units[146]  = 1.0000000000000002e-06;
    activation_units[146] = 0.50321666580471969;
    phase_units[146]      = pow(10,-12.000000);
    is_PD[146] = 0;
    nTB[146] = 0;

    // (147):  2.000000 C3H5XA => C3H4XA + C3H6
    kiv[147] = {38,37,30};
    nuv[147] = {-2.0,1,1};
    kiv_qss[147] = {};
    nuv_qss[147] = {};
    // (147):  2.000000 C3H5XA => C3H4XA + C3H6
    fwd_A[147]     = 1000000000000;
    fwd_beta[147]  = 0;
    fwd_Ea[147]    = 0;
    prefactor_units[147]  = 1.0000000000000002e-06;
    activation_units[147] = 0.50321666580471969;
    phase_units[147]      = pow(10,-12.000000);
    is_PD[147] = 0;
    nTB[147] = 0;

    // (148):  C3H6 + H => C2H4 + CH3
    kiv[148] = {30,3,31,16};
    nuv[148] = {-1,-1,1,1};
    kiv_qss[148] = {};
    nuv_qss[148] = {};
    // (148):  C3H6 + H => C2H4 + CH3
    fwd_A[148]     = 4.8299999999999998e+33;
    fwd_beta[148]  = -5.8099999999999996;
    fwd_Ea[148]    = 18500;
    prefactor_units[148]  = 1.0000000000000002e-06;
    activation_units[148] = 0.50321666580471969;
    phase_units[148]      = pow(10,-12.000000);
    is_PD[148] = 0;
    nTB[148] = 0;

    // (149):  C3H6 + H => C3H5XA + H2
    kiv[149] = {30,3,38,2};
    nuv[149] = {-1,-1,1,1};
    kiv_qss[149] = {};
    nuv_qss[149] = {};
    // (149):  C3H6 + H => C3H5XA + H2
    fwd_A[149]     = 173000;
    fwd_beta[149]  = 2.5;
    fwd_Ea[149]    = 2492.1100000000001;
    prefactor_units[149]  = 1.0000000000000002e-06;
    activation_units[149] = 0.50321666580471969;
    phase_units[149]      = pow(10,-12.000000);
    is_PD[149] = 0;
    nTB[149] = 0;

    // (150):  C3H6 + O => C2H5 + HCO
    kiv[150] = {30,1,21,10};
    nuv[150] = {-1,-1,1,1};
    kiv_qss[150] = {};
    nuv_qss[150] = {};
    // (150):  C3H6 + O => C2H5 + HCO
    fwd_A[150]     = 15800000;
    fwd_beta[150]  = 1.76;
    fwd_Ea[150]    = -1216.0599999999999;
    prefactor_units[150]  = 1.0000000000000002e-06;
    activation_units[150] = 0.50321666580471969;
    phase_units[150]      = pow(10,-12.000000);
    is_PD[150] = 0;
    nTB[150] = 0;

    // (151):  C3H6 + O => C3H5XA + OH
    kiv[151] = {30,1,38,4};
    nuv[151] = {-1,-1,1,1};
    kiv_qss[151] = {};
    nuv_qss[151] = {};
    // (151):  C3H6 + O => C3H5XA + OH
    fwd_A[151]     = 524000000000;
    fwd_beta[151]  = 0.69999999999999996;
    fwd_Ea[151]    = 5884.0799999999999;
    prefactor_units[151]  = 1.0000000000000002e-06;
    activation_units[151] = 0.50321666580471969;
    phase_units[151]      = pow(10,-12.000000);
    is_PD[151] = 0;
    nTB[151] = 0;

    // (152):  C3H6 + O => CH2CO + CH3 + H
    kiv[152] = {30,1,22,16,3};
    nuv[152] = {-1,-1,1,1,1};
    kiv_qss[152] = {};
    nuv_qss[152] = {};
    // (152):  C3H6 + O => CH2CO + CH3 + H
    fwd_A[152]     = 25000000;
    fwd_beta[152]  = 1.76;
    fwd_Ea[152]    = 76;
    prefactor_units[152]  = 1.0000000000000002e-06;
    activation_units[152] = 0.50321666580471969;
    phase_units[152]      = pow(10,-12.000000);
    is_PD[152] = 0;
    nTB[152] = 0;

    // (153):  C3H6 + OH => C3H5XA + H2O
    kiv[153] = {30,4,38,5};
    nuv[153] = {-1,-1,1,1};
    kiv_qss[153] = {};
    nuv_qss[153] = {};
    // (153):  C3H6 + OH => C3H5XA + H2O
    fwd_A[153]     = 3120000;
    fwd_beta[153]  = 2;
    fwd_Ea[153]    = -298.04000000000002;
    prefactor_units[153]  = 1.0000000000000002e-06;
    activation_units[153] = 0.50321666580471969;
    phase_units[153]      = pow(10,-12.000000);
    is_PD[153] = 0;
    nTB[153] = 0;

    // (154):  NXC3H7 + O2 => C3H6 + HO2
    kiv[154] = {39,6,30,7};
    nuv[154] = {-1,-1,1,1};
    kiv_qss[154] = {};
    nuv_qss[154] = {};
    // (154):  NXC3H7 + O2 => C3H6 + HO2
    fwd_A[154]     = 300000000000;
    fwd_beta[154]  = 0;
    fwd_Ea[154]    = 3000;
    prefactor_units[154]  = 1.0000000000000002e-06;
    activation_units[154] = 0.50321666580471969;
    phase_units[154]      = pow(10,-12.000000);
    is_PD[154] = 0;
    nTB[154] = 0;

    // (155):  NXC3H7 => CH3 + C2H4
    kiv[155] = {39,16,31};
    nuv[155] = {-1,1,1};
    kiv_qss[155] = {};
    nuv_qss[155] = {};
    // (155):  NXC3H7 => CH3 + C2H4
    fwd_A[155]     = 228400000000000;
    fwd_beta[155]  = -0.55000000000000004;
    fwd_Ea[155]    = 28400.099999999999;
    prefactor_units[155]  = 1;
    activation_units[155] = 0.50321666580471969;
    phase_units[155]      = pow(10,-6.000000);
    is_PD[155] = 0;
    nTB[155] = 0;

    // (156):  CH3 + C2H4 => NXC3H7
    kiv[156] = {16,31,39};
    nuv[156] = {-1,-1,1};
    kiv_qss[156] = {};
    nuv_qss[156] = {};
    // (156):  CH3 + C2H4 => NXC3H7
    fwd_A[156]     = 410000000000;
    fwd_beta[156]  = 0;
    fwd_Ea[156]    = 7204.1099999999997;
    prefactor_units[156]  = 1.0000000000000002e-06;
    activation_units[156] = 0.50321666580471969;
    phase_units[156]      = pow(10,-12.000000);
    is_PD[156] = 0;
    nTB[156] = 0;

    // (157):  NXC3H7 => H + C3H6
    kiv[157] = {39,3,30};
    nuv[157] = {-1,1,1};
    kiv_qss[157] = {};
    nuv_qss[157] = {};
    // (157):  NXC3H7 => H + C3H6
    fwd_A[157]     = 2667000000000000;
    fwd_beta[157]  = -0.64000000000000001;
    fwd_Ea[157]    = 36820.029999999999;
    prefactor_units[157]  = 1;
    activation_units[157] = 0.50321666580471969;
    phase_units[157]      = pow(10,-6.000000);
    is_PD[157] = 0;
    nTB[157] = 0;

    // (158):  H + C3H6 => NXC3H7
    kiv[158] = {3,30,39};
    nuv[158] = {-1,-1,1};
    kiv_qss[158] = {};
    nuv_qss[158] = {};
    // (158):  H + C3H6 => NXC3H7
    fwd_A[158]     = 10000000000000;
    fwd_beta[158]  = 0;
    fwd_Ea[158]    = 2500;
    prefactor_units[158]  = 1.0000000000000002e-06;
    activation_units[158] = 0.50321666580471969;
    phase_units[158]      = pow(10,-12.000000);
    is_PD[158] = 0;
    nTB[158] = 0;

    // (159):  NXC3H7O2 => NXC3H7 + O2
    kiv[159] = {40,39,6};
    nuv[159] = {-1,1,1};
    kiv_qss[159] = {};
    nuv_qss[159] = {};
    // (159):  NXC3H7O2 => NXC3H7 + O2
    fwd_A[159]     = 3.364e+19;
    fwd_beta[159]  = -1.3200000000000001;
    fwd_Ea[159]    = 35760.040000000001;
    prefactor_units[159]  = 1;
    activation_units[159] = 0.50321666580471969;
    phase_units[159]      = pow(10,-6.000000);
    is_PD[159] = 0;
    nTB[159] = 0;

    // (160):  NXC3H7 + O2 => NXC3H7O2
    kiv[160] = {39,6,40};
    nuv[160] = {-1,-1,1};
    kiv_qss[160] = {};
    nuv_qss[160] = {};
    // (160):  NXC3H7 + O2 => NXC3H7O2
    fwd_A[160]     = 4520000000000;
    fwd_beta[160]  = 0;
    fwd_Ea[160]    = 0;
    prefactor_units[160]  = 1.0000000000000002e-06;
    activation_units[160] = 0.50321666580471969;
    phase_units[160]      = pow(10,-12.000000);
    is_PD[160] = 0;
    nTB[160] = 0;

    // (161):  C4H6 => 2.000000 C2H3
    kiv[161] = {41,28};
    nuv[161] = {-1,2.0};
    kiv_qss[161] = {};
    nuv_qss[161] = {};
    // (161):  C4H6 => 2.000000 C2H3
    fwd_A[161]     = 4.027e+19;
    fwd_beta[161]  = -1;
    fwd_Ea[161]    = 98150.100000000006;
    prefactor_units[161]  = 1;
    activation_units[161] = 0.50321666580471969;
    phase_units[161]      = pow(10,-6.000000);
    is_PD[161] = 0;
    nTB[161] = 0;

    // (162):  2.000000 C2H3 => C4H6
    kiv[162] = {28,41};
    nuv[162] = {-2.0,1};
    kiv_qss[162] = {};
    nuv_qss[162] = {};
    // (162):  2.000000 C2H3 => C4H6
    fwd_A[162]     = 12600000000000;
    fwd_beta[162]  = 0;
    fwd_Ea[162]    = 0;
    prefactor_units[162]  = 1.0000000000000002e-06;
    activation_units[162] = 0.50321666580471969;
    phase_units[162]      = pow(10,-12.000000);
    is_PD[162] = 0;
    nTB[162] = 0;

    // (163):  C4H6 + OH => CH2O + C3H5XA
    kiv[163] = {41,4,14,38};
    nuv[163] = {-1,-1,1,1};
    kiv_qss[163] = {};
    nuv_qss[163] = {};
    // (163):  C4H6 + OH => CH2O + C3H5XA
    fwd_A[163]     = 1000000000000;
    fwd_beta[163]  = 0;
    fwd_Ea[163]    = 0;
    prefactor_units[163]  = 1.0000000000000002e-06;
    activation_units[163] = 0.50321666580471969;
    phase_units[163]      = pow(10,-12.000000);
    is_PD[163] = 0;
    nTB[163] = 0;

    // (164):  C4H6 + OH => C2H5 + CH2CO
    kiv[164] = {41,4,21,22};
    nuv[164] = {-1,-1,1,1};
    kiv_qss[164] = {};
    nuv_qss[164] = {};
    // (164):  C4H6 + OH => C2H5 + CH2CO
    fwd_A[164]     = 1000000000000;
    fwd_beta[164]  = 0;
    fwd_Ea[164]    = 0;
    prefactor_units[164]  = 1.0000000000000002e-06;
    activation_units[164] = 0.50321666580471969;
    phase_units[164]      = pow(10,-12.000000);
    is_PD[164] = 0;
    nTB[164] = 0;

    // (165):  C4H6 + O => C2H4 + CH2CO
    kiv[165] = {41,1,31,22};
    nuv[165] = {-1,-1,1,1};
    kiv_qss[165] = {};
    nuv_qss[165] = {};
    // (165):  C4H6 + O => C2H4 + CH2CO
    fwd_A[165]     = 1000000000000;
    fwd_beta[165]  = 0;
    fwd_Ea[165]    = 0;
    prefactor_units[165]  = 1.0000000000000002e-06;
    activation_units[165] = 0.50321666580471969;
    phase_units[165]      = pow(10,-12.000000);
    is_PD[165] = 0;
    nTB[165] = 0;

    // (166):  C4H6 + H => C2H3 + C2H4
    kiv[166] = {41,3,28,31};
    nuv[166] = {-1,-1,1,1};
    kiv_qss[166] = {};
    nuv_qss[166] = {};
    // (166):  C4H6 + H => C2H3 + C2H4
    fwd_A[166]     = 10000000000000;
    fwd_beta[166]  = 0;
    fwd_Ea[166]    = 4700.0500000000002;
    prefactor_units[166]  = 1.0000000000000002e-06;
    activation_units[166] = 0.50321666580471969;
    phase_units[166]      = pow(10,-12.000000);
    is_PD[166] = 0;
    nTB[166] = 0;

    // (167):  C4H6 + O => CH2O + C3H4XA
    kiv[167] = {41,1,14,37};
    nuv[167] = {-1,-1,1,1};
    kiv_qss[167] = {};
    nuv_qss[167] = {};
    // (167):  C4H6 + O => CH2O + C3H4XA
    fwd_A[167]     = 1000000000000;
    fwd_beta[167]  = 0;
    fwd_Ea[167]    = 0;
    prefactor_units[167]  = 1.0000000000000002e-06;
    activation_units[167] = 0.50321666580471969;
    phase_units[167]      = pow(10,-12.000000);
    is_PD[167] = 0;
    nTB[167] = 0;

    // (168):  H + C4H7 => C4H8X1
    kiv[168] = {3,42,43};
    nuv[168] = {-1,-1,1};
    kiv_qss[168] = {};
    nuv_qss[168] = {};
    // (168):  H + C4H7 => C4H8X1
    fwd_A[168]     = 50000000000000;
    fwd_beta[168]  = 0;
    fwd_Ea[168]    = 0;
    prefactor_units[168]  = 1.0000000000000002e-06;
    activation_units[168] = 0.50321666580471969;
    phase_units[168]      = pow(10,-12.000000);
    is_PD[168] = 0;
    nTB[168] = 0;

    // (169):  C3H5XA + C4H7 => C3H6 + C4H6
    kiv[169] = {38,42,30,41};
    nuv[169] = {-1,-1,1,1};
    kiv_qss[169] = {};
    nuv_qss[169] = {};
    // (169):  C3H5XA + C4H7 => C3H6 + C4H6
    fwd_A[169]     = 6310000000000;
    fwd_beta[169]  = 0;
    fwd_Ea[169]    = 0;
    prefactor_units[169]  = 1.0000000000000002e-06;
    activation_units[169] = 0.50321666580471969;
    phase_units[169]      = pow(10,-12.000000);
    is_PD[169] = 0;
    nTB[169] = 0;

    // (170):  C2H5 + C4H7 => C4H6 + C2H6
    kiv[170] = {21,42,41,20};
    nuv[170] = {-1,-1,1,1};
    kiv_qss[170] = {};
    nuv_qss[170] = {};
    // (170):  C2H5 + C4H7 => C4H6 + C2H6
    fwd_A[170]     = 3980000000000;
    fwd_beta[170]  = 0;
    fwd_Ea[170]    = 0;
    prefactor_units[170]  = 1.0000000000000002e-06;
    activation_units[170] = 0.50321666580471969;
    phase_units[170]      = pow(10,-12.000000);
    is_PD[170] = 0;
    nTB[170] = 0;

    // (171):  C4H7 => C4H6 + H
    kiv[171] = {42,41,3};
    nuv[171] = {-1,1,1};
    kiv_qss[171] = {};
    nuv_qss[171] = {};
    // (171):  C4H7 => C4H6 + H
    fwd_A[171]     = 120000000000000;
    fwd_beta[171]  = 0;
    fwd_Ea[171]    = 49299.949999999997;
    prefactor_units[171]  = 1;
    activation_units[171] = 0.50321666580471969;
    phase_units[171]      = pow(10,-6.000000);
    is_PD[171] = 0;
    nTB[171] = 0;

    // (172):  C4H6 + H => C4H7
    kiv[172] = {41,3,42};
    nuv[172] = {-1,-1,1};
    kiv_qss[172] = {};
    nuv_qss[172] = {};
    // (172):  C4H6 + H => C4H7
    fwd_A[172]     = 40000000000000;
    fwd_beta[172]  = 0;
    fwd_Ea[172]    = 1299.95;
    prefactor_units[172]  = 1.0000000000000002e-06;
    activation_units[172] = 0.50321666580471969;
    phase_units[172]      = pow(10,-12.000000);
    is_PD[172] = 0;
    nTB[172] = 0;

    // (173):  C4H7 + CH3 => C4H6 + CH4
    kiv[173] = {42,16,41,18};
    nuv[173] = {-1,-1,1,1};
    kiv_qss[173] = {};
    nuv_qss[173] = {};
    // (173):  C4H7 + CH3 => C4H6 + CH4
    fwd_A[173]     = 8000000000000;
    fwd_beta[173]  = 0;
    fwd_Ea[173]    = 0;
    prefactor_units[173]  = 1.0000000000000002e-06;
    activation_units[173] = 0.50321666580471969;
    phase_units[173]      = pow(10,-12.000000);
    is_PD[173] = 0;
    nTB[173] = 0;

    // (174):  C4H7 + HO2 => C4H8X1 + O2
    kiv[174] = {42,7,43,6};
    nuv[174] = {-1,-1,1,1};
    kiv_qss[174] = {};
    nuv_qss[174] = {};
    // (174):  C4H7 + HO2 => C4H8X1 + O2
    fwd_A[174]     = 300000000000;
    fwd_beta[174]  = 0;
    fwd_Ea[174]    = 0;
    prefactor_units[174]  = 1.0000000000000002e-06;
    activation_units[174] = 0.50321666580471969;
    phase_units[174]      = pow(10,-12.000000);
    is_PD[174] = 0;
    nTB[174] = 0;

    // (175):  C4H7 + O2 => C4H6 + HO2
    kiv[175] = {42,6,41,7};
    nuv[175] = {-1,-1,1,1};
    kiv_qss[175] = {};
    nuv_qss[175] = {};
    // (175):  C4H7 + O2 => C4H6 + HO2
    fwd_A[175]     = 1000000000;
    fwd_beta[175]  = 0;
    fwd_Ea[175]    = 0;
    prefactor_units[175]  = 1.0000000000000002e-06;
    activation_units[175] = 0.50321666580471969;
    phase_units[175]      = pow(10,-12.000000);
    is_PD[175] = 0;
    nTB[175] = 0;

    // (176):  C4H7 => C2H4 + C2H3
    kiv[176] = {42,31,28};
    nuv[176] = {-1,1,1};
    kiv_qss[176] = {};
    nuv_qss[176] = {};
    // (176):  C4H7 => C2H4 + C2H3
    fwd_A[176]     = 100000000000;
    fwd_beta[176]  = 0;
    fwd_Ea[176]    = 37000;
    prefactor_units[176]  = 1;
    activation_units[176] = 0.50321666580471969;
    phase_units[176]      = pow(10,-6.000000);
    is_PD[176] = 0;
    nTB[176] = 0;

    // (177):  H + C4H7 => C4H6 + H2
    kiv[177] = {3,42,41,2};
    nuv[177] = {-1,-1,1,1};
    kiv_qss[177] = {};
    nuv_qss[177] = {};
    // (177):  H + C4H7 => C4H6 + H2
    fwd_A[177]     = 31600000000000;
    fwd_beta[177]  = 0;
    fwd_Ea[177]    = 0;
    prefactor_units[177]  = 1.0000000000000002e-06;
    activation_units[177] = 0.50321666580471969;
    phase_units[177]      = pow(10,-12.000000);
    is_PD[177] = 0;
    nTB[177] = 0;

    // (178):  C4H8X1 + H => C4H7 + H2
    kiv[178] = {43,3,42,2};
    nuv[178] = {-1,-1,1,1};
    kiv_qss[178] = {};
    nuv_qss[178] = {};
    // (178):  C4H8X1 + H => C4H7 + H2
    fwd_A[178]     = 50000000000000;
    fwd_beta[178]  = 0;
    fwd_Ea[178]    = 3900.0999999999999;
    prefactor_units[178]  = 1.0000000000000002e-06;
    activation_units[178] = 0.50321666580471969;
    phase_units[178]      = pow(10,-12.000000);
    is_PD[178] = 0;
    nTB[178] = 0;

    // (179):  C4H8X1 + OH => NXC3H7 + CH2O
    kiv[179] = {43,4,39,14};
    nuv[179] = {-1,-1,1,1};
    kiv_qss[179] = {};
    nuv_qss[179] = {};
    // (179):  C4H8X1 + OH => NXC3H7 + CH2O
    fwd_A[179]     = 1000000000000;
    fwd_beta[179]  = 0;
    fwd_Ea[179]    = 0;
    prefactor_units[179]  = 1.0000000000000002e-06;
    activation_units[179] = 0.50321666580471969;
    phase_units[179]      = pow(10,-12.000000);
    is_PD[179] = 0;
    nTB[179] = 0;

    // (180):  C4H8X1 + OH => CH3CO + C2H6
    kiv[180] = {43,4,33,20};
    nuv[180] = {-1,-1,1,1};
    kiv_qss[180] = {};
    nuv_qss[180] = {};
    // (180):  C4H8X1 + OH => CH3CO + C2H6
    fwd_A[180]     = 500000000000;
    fwd_beta[180]  = 0;
    fwd_Ea[180]    = 0;
    prefactor_units[180]  = 1.0000000000000002e-06;
    activation_units[180] = 0.50321666580471969;
    phase_units[180]      = pow(10,-12.000000);
    is_PD[180] = 0;
    nTB[180] = 0;

    // (181):  C4H8X1 + O => CH3CO + C2H5
    kiv[181] = {43,1,33,21};
    nuv[181] = {-1,-1,1,1};
    kiv_qss[181] = {};
    nuv_qss[181] = {};
    // (181):  C4H8X1 + O => CH3CO + C2H5
    fwd_A[181]     = 13000000000000;
    fwd_beta[181]  = 0;
    fwd_Ea[181]    = 849.89999999999998;
    prefactor_units[181]  = 1.0000000000000002e-06;
    activation_units[181] = 0.50321666580471969;
    phase_units[181]      = pow(10,-12.000000);
    is_PD[181] = 0;
    nTB[181] = 0;

    // (182):  C4H8X1 + O => C3H6 + CH2O
    kiv[182] = {43,1,30,14};
    nuv[182] = {-1,-1,1,1};
    kiv_qss[182] = {};
    nuv_qss[182] = {};
    // (182):  C4H8X1 + O => C3H6 + CH2O
    fwd_A[182]     = 723000;
    fwd_beta[182]  = 2.3399999999999999;
    fwd_Ea[182]    = -1049.95;
    prefactor_units[182]  = 1.0000000000000002e-06;
    activation_units[182] = 0.50321666580471969;
    phase_units[182]      = pow(10,-12.000000);
    is_PD[182] = 0;
    nTB[182] = 0;

    // (183):  C4H8X1 + OH => C4H7 + H2O
    kiv[183] = {43,4,42,5};
    nuv[183] = {-1,-1,1,1};
    kiv_qss[183] = {};
    nuv_qss[183] = {};
    // (183):  C4H8X1 + OH => C4H7 + H2O
    fwd_A[183]     = 22500000000000;
    fwd_beta[183]  = 0;
    fwd_Ea[183]    = 2217.02;
    prefactor_units[183]  = 1.0000000000000002e-06;
    activation_units[183] = 0.50321666580471969;
    phase_units[183]      = pow(10,-12.000000);
    is_PD[183] = 0;
    nTB[183] = 0;

    // (184):  C4H8X1 => C3H5XA + CH3
    kiv[184] = {43,38,16};
    nuv[184] = {-1,1,1};
    kiv_qss[184] = {};
    nuv_qss[184] = {};
    // (184):  C4H8X1 => C3H5XA + CH3
    fwd_A[184]     = 5000000000000000;
    fwd_beta[184]  = 0;
    fwd_Ea[184]    = 71000;
    prefactor_units[184]  = 1;
    activation_units[184] = 0.50321666580471969;
    phase_units[184]      = pow(10,-6.000000);
    is_PD[184] = 0;
    nTB[184] = 0;

    // (185):  C3H5XA + CH3 => C4H8X1
    kiv[185] = {38,16,43};
    nuv[185] = {-1,-1,1};
    kiv_qss[185] = {};
    nuv_qss[185] = {};
    // (185):  C3H5XA + CH3 => C4H8X1
    fwd_A[185]     = 5000000000000;
    fwd_beta[185]  = 0;
    fwd_Ea[185]    = 0;
    prefactor_units[185]  = 1.0000000000000002e-06;
    activation_units[185] = 0.50321666580471969;
    phase_units[185]      = pow(10,-12.000000);
    is_PD[185] = 0;
    nTB[185] = 0;

    // (186):  PXC4H9 => C4H8X1 + H
    kiv[186] = {44,43,3};
    nuv[186] = {-1,1,1};
    kiv_qss[186] = {};
    nuv_qss[186] = {};
    // (186):  PXC4H9 => C4H8X1 + H
    fwd_A[186]     = 1.159e+17;
    fwd_beta[186]  = -1.1699999999999999;
    fwd_Ea[186]    = 38159.889999999999;
    prefactor_units[186]  = 1;
    activation_units[186] = 0.50321666580471969;
    phase_units[186]      = pow(10,-6.000000);
    is_PD[186] = 0;
    nTB[186] = 0;

    // (187):  C4H8X1 + H => PXC4H9
    kiv[187] = {43,3,44};
    nuv[187] = {-1,-1,1};
    kiv_qss[187] = {};
    nuv_qss[187] = {};
    // (187):  C4H8X1 + H => PXC4H9
    fwd_A[187]     = 10000000000000;
    fwd_beta[187]  = 0;
    fwd_Ea[187]    = 2900.0999999999999;
    prefactor_units[187]  = 1.0000000000000002e-06;
    activation_units[187] = 0.50321666580471969;
    phase_units[187]      = pow(10,-12.000000);
    is_PD[187] = 0;
    nTB[187] = 0;

    // (188):  PXC4H9 => C2H5 + C2H4
    kiv[188] = {44,21,31};
    nuv[188] = {-1,1,1};
    kiv_qss[188] = {};
    nuv_qss[188] = {};
    // (188):  PXC4H9 => C2H5 + C2H4
    fwd_A[188]     = 7.497e+17;
    fwd_beta[188]  = -1.4099999999999999;
    fwd_Ea[188]    = 29580.07;
    prefactor_units[188]  = 1;
    activation_units[188] = 0.50321666580471969;
    phase_units[188]      = pow(10,-6.000000);
    is_PD[188] = 0;
    nTB[188] = 0;

    // (189):  PXC4H9O2 => PXC4H9 + O2
    kiv[189] = {45,44,6};
    nuv[189] = {-1,1,1};
    kiv_qss[189] = {};
    nuv_qss[189] = {};
    // (189):  PXC4H9O2 => PXC4H9 + O2
    fwd_A[189]     = 6.155e+19;
    fwd_beta[189]  = -1.3799999999999999;
    fwd_Ea[189]    = 35510.040000000001;
    prefactor_units[189]  = 1;
    activation_units[189] = 0.50321666580471969;
    phase_units[189]      = pow(10,-6.000000);
    is_PD[189] = 0;
    nTB[189] = 0;

    // (190):  PXC4H9 + O2 => PXC4H9O2
    kiv[190] = {44,6,45};
    nuv[190] = {-1,-1,1};
    kiv_qss[190] = {};
    nuv_qss[190] = {};
    // (190):  PXC4H9 + O2 => PXC4H9O2
    fwd_A[190]     = 4520000000000;
    fwd_beta[190]  = 0;
    fwd_Ea[190]    = 0;
    prefactor_units[190]  = 1.0000000000000002e-06;
    activation_units[190] = 0.50321666580471969;
    phase_units[190]      = pow(10,-12.000000);
    is_PD[190] = 0;
    nTB[190] = 0;

    // (191):  C5H9 => C4H6 + CH3
    kiv[191] = {46,41,16};
    nuv[191] = {-1,1,1};
    kiv_qss[191] = {};
    nuv_qss[191] = {};
    // (191):  C5H9 => C4H6 + CH3
    fwd_A[191]     = 1339000000000000;
    fwd_beta[191]  = -0.52000000000000002;
    fwd_Ea[191]    = 38320.029999999999;
    prefactor_units[191]  = 1;
    activation_units[191] = 0.50321666580471969;
    phase_units[191]      = pow(10,-6.000000);
    is_PD[191] = 0;
    nTB[191] = 0;

    // (192):  C5H9 => C3H5XA + C2H4
    kiv[192] = {46,38,31};
    nuv[192] = {-1,1,1};
    kiv_qss[192] = {};
    nuv_qss[192] = {};
    // (192):  C5H9 => C3H5XA + C2H4
    fwd_A[192]     = 25000000000000;
    fwd_beta[192]  = 0;
    fwd_Ea[192]    = 45000;
    prefactor_units[192]  = 1;
    activation_units[192] = 0.50321666580471969;
    phase_units[192]      = pow(10,-6.000000);
    is_PD[192] = 0;
    nTB[192] = 0;

    // (193):  C5H10X1 + OH => C5H9 + H2O
    kiv[193] = {47,4,46,5};
    nuv[193] = {-1,-1,1,1};
    kiv_qss[193] = {};
    nuv_qss[193] = {};
    // (193):  C5H10X1 + OH => C5H9 + H2O
    fwd_A[193]     = 5120000;
    fwd_beta[193]  = 2;
    fwd_Ea[193]    = -298.04000000000002;
    prefactor_units[193]  = 1.0000000000000002e-06;
    activation_units[193] = 0.50321666580471969;
    phase_units[193]      = pow(10,-12.000000);
    is_PD[193] = 0;
    nTB[193] = 0;

    // (194):  C5H10X1 + H => C5H9 + H2
    kiv[194] = {47,3,46,2};
    nuv[194] = {-1,-1,1,1};
    kiv_qss[194] = {};
    nuv_qss[194] = {};
    // (194):  C5H10X1 + H => C5H9 + H2
    fwd_A[194]     = 28000000000000;
    fwd_beta[194]  = 0;
    fwd_Ea[194]    = 4000;
    prefactor_units[194]  = 1.0000000000000002e-06;
    activation_units[194] = 0.50321666580471969;
    phase_units[194]      = pow(10,-12.000000);
    is_PD[194] = 0;
    nTB[194] = 0;

    // (195):  C5H10X1 => C2H5 + C3H5XA
    kiv[195] = {47,21,38};
    nuv[195] = {-1,1,1};
    kiv_qss[195] = {};
    nuv_qss[195] = {};
    // (195):  C5H10X1 => C2H5 + C3H5XA
    fwd_A[195]     = 9.173e+20;
    fwd_beta[195]  = -1.6299999999999999;
    fwd_Ea[195]    = 73989.960000000006;
    prefactor_units[195]  = 1;
    activation_units[195] = 0.50321666580471969;
    phase_units[195]      = pow(10,-6.000000);
    is_PD[195] = 0;
    nTB[195] = 0;

    // (196):  C2H5 + C3H5XA => C5H10X1
    kiv[196] = {21,38,47};
    nuv[196] = {-1,-1,1};
    kiv_qss[196] = {};
    nuv_qss[196] = {};
    // (196):  C2H5 + C3H5XA => C5H10X1
    fwd_A[196]     = 4000000000000;
    fwd_beta[196]  = 0;
    fwd_Ea[196]    = -596.08000000000004;
    prefactor_units[196]  = 1.0000000000000002e-06;
    activation_units[196] = 0.50321666580471969;
    phase_units[196]      = pow(10,-12.000000);
    is_PD[196] = 0;
    nTB[196] = 0;

    // (197):  C5H10X1 + O => C5H9 + OH
    kiv[197] = {47,1,46,4};
    nuv[197] = {-1,-1,1,1};
    kiv_qss[197] = {};
    nuv_qss[197] = {};
    // (197):  C5H10X1 + O => C5H9 + OH
    fwd_A[197]     = 254000;
    fwd_beta[197]  = 2.5600000000000001;
    fwd_Ea[197]    = -1130.02;
    prefactor_units[197]  = 1.0000000000000002e-06;
    activation_units[197] = 0.50321666580471969;
    phase_units[197]      = pow(10,-12.000000);
    is_PD[197] = 0;
    nTB[197] = 0;

    // (198):  C5H11X1 => C3H6 + C2H5
    kiv[198] = {48,30,21};
    nuv[198] = {-1,1,1};
    kiv_qss[198] = {};
    nuv_qss[198] = {};
    // (198):  C5H11X1 => C3H6 + C2H5
    fwd_A[198]     = 5.948e+17;
    fwd_beta[198]  = -1.268;
    fwd_Ea[198]    = 32384.32;
    prefactor_units[198]  = 1;
    activation_units[198] = 0.50321666580471969;
    phase_units[198]      = pow(10,-6.000000);
    is_PD[198] = 0;
    nTB[198] = 0;

    // (199):  C5H11X1 => C2H4 + NXC3H7
    kiv[199] = {48,31,39};
    nuv[199] = {-1,1,1};
    kiv_qss[199] = {};
    nuv_qss[199] = {};
    // (199):  C5H11X1 => C2H4 + NXC3H7
    fwd_A[199]     = 7.305e+18;
    fwd_beta[199]  = -1.7669999999999999;
    fwd_Ea[199]    = 29919.459999999999;
    prefactor_units[199]  = 1;
    activation_units[199] = 0.50321666580471969;
    phase_units[199]      = pow(10,-6.000000);
    is_PD[199] = 0;
    nTB[199] = 0;

    // (200):  C5H11X1 <=> C5H10X1 + H
    kiv[200] = {48,47,3};
    nuv[200] = {-1,1,1};
    kiv_qss[200] = {};
    nuv_qss[200] = {};
    // (200):  C5H11X1 <=> C5H10X1 + H
    fwd_A[200]     = 1325000000000000;
    fwd_beta[200]  = -0.55400000000000005;
    fwd_Ea[200]    = 37516.730000000003;
    prefactor_units[200]  = 1;
    activation_units[200] = 0.50321666580471969;
    phase_units[200]      = pow(10,-6.000000);
    is_PD[200] = 0;
    nTB[200] = 0;

    // (201):  C6H12X1 => NXC3H7 + C3H5XA
    kiv[201] = {49,39,38};
    nuv[201] = {-1,1,1};
    kiv_qss[201] = {};
    nuv_qss[201] = {};
    // (201):  C6H12X1 => NXC3H7 + C3H5XA
    fwd_A[201]     = 10000000000000000;
    fwd_beta[201]  = 0;
    fwd_Ea[201]    = 71000;
    prefactor_units[201]  = 1;
    activation_units[201] = 0.50321666580471969;
    phase_units[201]      = pow(10,-6.000000);
    is_PD[201] = 0;
    nTB[201] = 0;

    // (202):  C6H12X1 + OH => C5H11X1 + CH2O
    kiv[202] = {49,4,48,14};
    nuv[202] = {-1,-1,1,1};
    kiv_qss[202] = {};
    nuv_qss[202] = {};
    // (202):  C6H12X1 + OH => C5H11X1 + CH2O
    fwd_A[202]     = 100000000000;
    fwd_beta[202]  = -0;
    fwd_Ea[202]    = -4000;
    prefactor_units[202]  = 1.0000000000000002e-06;
    activation_units[202] = 0.50321666580471969;
    phase_units[202]      = pow(10,-12.000000);
    is_PD[202] = 0;
    nTB[202] = 0;

    // (203):  C7H15X2 => C6H12X1 + CH3
    kiv[203] = {50,49,16};
    nuv[203] = {-1,1,1};
    kiv_qss[203] = {};
    nuv_qss[203] = {};
    // (203):  C7H15X2 => C6H12X1 + CH3
    fwd_A[203]     = 261700000000000;
    fwd_beta[203]  = -0.65400000000000003;
    fwd_Ea[203]    = 29745.459999999999;
    prefactor_units[203]  = 1;
    activation_units[203] = 0.50321666580471969;
    phase_units[203]      = pow(10,-6.000000);
    is_PD[203] = 0;
    nTB[203] = 0;

    // (204):  C7H15X2 => PXC4H9 + C3H6
    kiv[204] = {50,44,30};
    nuv[204] = {-1,1,1};
    kiv_qss[204] = {};
    nuv_qss[204] = {};
    // (204):  C7H15X2 => PXC4H9 + C3H6
    fwd_A[204]     = 5.313e+17;
    fwd_beta[204]  = -1.411;
    fwd_Ea[204]    = 31432.84;
    prefactor_units[204]  = 1;
    activation_units[204] = 0.50321666580471969;
    phase_units[204]      = pow(10,-6.000000);
    is_PD[204] = 0;
    nTB[204] = 0;

    // (205):  C7H15X2 => C4H8X1 + NXC3H7
    kiv[205] = {50,43,39};
    nuv[205] = {-1,1,1};
    kiv_qss[205] = {};
    nuv_qss[205] = {};
    // (205):  C7H15X2 => C4H8X1 + NXC3H7
    fwd_A[205]     = 2.454e+18;
    fwd_beta[205]  = -1.6539999999999999;
    fwd_Ea[205]    = 31635.52;
    prefactor_units[205]  = 1;
    activation_units[205] = 0.50321666580471969;
    phase_units[205]      = pow(10,-6.000000);
    is_PD[205] = 0;
    nTB[205] = 0;

    // (206):  C7H15X2 => C5H11X1 + C2H4
    kiv[206] = {50,48,31};
    nuv[206] = {-1,1,1};
    kiv_qss[206] = {};
    nuv_qss[206] = {};
    // (206):  C7H15X2 => C5H11X1 + C2H4
    fwd_A[206]     = 3734000000000000;
    fwd_beta[206]  = -0.92700000000000005;
    fwd_Ea[206]    = 29637.91;
    prefactor_units[206]  = 1;
    activation_units[206] = 0.50321666580471969;
    phase_units[206]      = pow(10,-6.000000);
    is_PD[206] = 0;
    nTB[206] = 0;

    // (207):  C7H15X2 => C2H5 + C5H10X1
    kiv[207] = {50,21,47};
    nuv[207] = {-1,1,1};
    kiv_qss[207] = {};
    nuv_qss[207] = {};
    // (207):  C7H15X2 => C2H5 + C5H10X1
    fwd_A[207]     = 1.368e+17;
    fwd_beta[207]  = -1.3939999999999999;
    fwd_Ea[207]    = 29858.990000000002;
    prefactor_units[207]  = 1;
    activation_units[207] = 0.50321666580471969;
    phase_units[207]      = pow(10,-6.000000);
    is_PD[207] = 0;
    nTB[207] = 0;

    // (208):  C7H15X2 + HO2 => NXC7H16 + O2
    kiv[208] = {50,7,51,6};
    nuv[208] = {-1,-1,1,1};
    kiv_qss[208] = {};
    nuv_qss[208] = {};
    // (208):  C7H15X2 + HO2 => NXC7H16 + O2
    fwd_A[208]     = 191700000;
    fwd_beta[208]  = 0.871;
    fwd_Ea[208]    = -1588.9100000000001;
    prefactor_units[208]  = 1.0000000000000002e-06;
    activation_units[208] = 0.50321666580471969;
    phase_units[208]      = pow(10,-12.000000);
    is_PD[208] = 0;
    nTB[208] = 0;

    // (209):  NXC7H16 + CH3O2 => C7H15X2 + CH3O2H
    kiv[209] = {51,24,50,25};
    nuv[209] = {-1,-1,1,1};
    kiv_qss[209] = {};
    nuv_qss[209] = {};
    // (209):  NXC7H16 + CH3O2 => C7H15X2 + CH3O2H
    fwd_A[209]     = 5646000000000;
    fwd_beta[209]  = 0.20100000000000001;
    fwd_Ea[209]    = 17650.330000000002;
    prefactor_units[209]  = 1.0000000000000002e-06;
    activation_units[209] = 0.50321666580471969;
    phase_units[209]      = pow(10,-12.000000);
    is_PD[209] = 0;
    nTB[209] = 0;

    // (210):  NXC7H16 + H => C7H15X2 + H2
    kiv[210] = {51,3,50,2};
    nuv[210] = {-1,-1,1,1};
    kiv_qss[210] = {};
    nuv_qss[210] = {};
    // (210):  NXC7H16 + H => C7H15X2 + H2
    fwd_A[210]     = 1749000;
    fwd_beta[210]  = 2.6000000000000001;
    fwd_Ea[210]    = 4361.8500000000004;
    prefactor_units[210]  = 1.0000000000000002e-06;
    activation_units[210] = 0.50321666580471969;
    phase_units[210]      = pow(10,-12.000000);
    is_PD[210] = 0;
    nTB[210] = 0;

    // (211):  NXC7H16 => PXC4H9 + NXC3H7
    kiv[211] = {51,44,39};
    nuv[211] = {-1,1,1};
    kiv_qss[211] = {};
    nuv_qss[211] = {};
    // (211):  NXC7H16 => PXC4H9 + NXC3H7
    fwd_A[211]     = 1.415e+78;
    fwd_beta[211]  = -17.710000000000001;
    fwd_Ea[211]    = 120700.05;
    prefactor_units[211]  = 1;
    activation_units[211] = 0.50321666580471969;
    phase_units[211]      = pow(10,-6.000000);
    is_PD[211] = 0;
    nTB[211] = 0;

    // (212):  NXC7H16 + HO2 => C7H15X2 + H2O2
    kiv[212] = {51,7,50,8};
    nuv[212] = {-1,-1,1,1};
    kiv_qss[212] = {};
    nuv_qss[212] = {};
    // (212):  NXC7H16 + HO2 => C7H15X2 + H2O2
    fwd_A[212]     = 7741000000000;
    fwd_beta[212]  = 0.20300000000000001;
    fwd_Ea[212]    = 17636.950000000001;
    prefactor_units[212]  = 1.0000000000000002e-06;
    activation_units[212] = 0.50321666580471969;
    phase_units[212]      = pow(10,-12.000000);
    is_PD[212] = 0;
    nTB[212] = 0;

    // (213):  NXC7H16 => C5H11X1 + C2H5
    kiv[213] = {51,48,21};
    nuv[213] = {-1,1,1};
    kiv_qss[213] = {};
    nuv_qss[213] = {};
    // (213):  NXC7H16 => C5H11X1 + C2H5
    fwd_A[213]     = 8.0999999999999995e+77;
    fwd_beta[213]  = -17.620000000000001;
    fwd_Ea[213]    = 120400.10000000001;
    prefactor_units[213]  = 1;
    activation_units[213] = 0.50321666580471969;
    phase_units[213]      = pow(10,-6.000000);
    is_PD[213] = 0;
    nTB[213] = 0;

    // (214):  NXC7H16 + CH3O => C7H15X2 + CH3OH
    kiv[214] = {51,17,50,19};
    nuv[214] = {-1,-1,1,1};
    kiv_qss[214] = {};
    nuv_qss[214] = {};
    // (214):  NXC7H16 + CH3O => C7H15X2 + CH3OH
    fwd_A[214]     = 268900000000;
    fwd_beta[214]  = 0.13600000000000001;
    fwd_Ea[214]    = 5069.5500000000002;
    prefactor_units[214]  = 1.0000000000000002e-06;
    activation_units[214] = 0.50321666580471969;
    phase_units[214]      = pow(10,-12.000000);
    is_PD[214] = 0;
    nTB[214] = 0;

    // (215):  NXC7H16 + O => C7H15X2 + OH
    kiv[215] = {51,1,50,4};
    nuv[215] = {-1,-1,1,1};
    kiv_qss[215] = {};
    nuv_qss[215] = {};
    // (215):  NXC7H16 + O => C7H15X2 + OH
    fwd_A[215]     = 176600;
    fwd_beta[215]  = 2.802;
    fwd_Ea[215]    = 2265.3000000000002;
    prefactor_units[215]  = 1.0000000000000002e-06;
    activation_units[215] = 0.50321666580471969;
    phase_units[215]      = pow(10,-12.000000);
    is_PD[215] = 0;
    nTB[215] = 0;

    // (216):  NXC7H16 + OH => C7H15X2 + H2O
    kiv[216] = {51,4,50,5};
    nuv[216] = {-1,-1,1,1};
    kiv_qss[216] = {};
    nuv_qss[216] = {};
    // (216):  NXC7H16 + OH => C7H15X2 + H2O
    fwd_A[216]     = 751800000;
    fwd_beta[216]  = 1.494;
    fwd_Ea[216]    = 260.51999999999998;
    prefactor_units[216]  = 1.0000000000000002e-06;
    activation_units[216] = 0.50321666580471969;
    phase_units[216]      = pow(10,-12.000000);
    is_PD[216] = 0;
    nTB[216] = 0;

    // (217):  NXC7H16 + CH3 => C7H15X2 + CH4
    kiv[217] = {51,16,50,18};
    nuv[217] = {-1,-1,1,1};
    kiv_qss[217] = {};
    nuv_qss[217] = {};
    // (217):  NXC7H16 + CH3 => C7H15X2 + CH4
    fwd_A[217]     = 14420;
    fwd_beta[217]  = 2.573;
    fwd_Ea[217]    = 6933.5600000000004;
    prefactor_units[217]  = 1.0000000000000002e-06;
    activation_units[217] = 0.50321666580471969;
    phase_units[217]      = pow(10,-12.000000);
    is_PD[217] = 0;
    nTB[217] = 0;

}


/* Finalizes parameter database */
void CKFINALIZE()
{
  for (int i=0; i<218; ++i) {
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
    *kk = 52;
    *ii = 218;
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
    kname.resize(52);
    kname[0] = "N2";
    kname[1] = "O";
    kname[2] = "H2";
    kname[3] = "H";
    kname[4] = "OH";
    kname[5] = "H2O";
    kname[6] = "O2";
    kname[7] = "HO2";
    kname[8] = "H2O2";
    kname[9] = "CH";
    kname[10] = "HCO";
    kname[11] = "CH2";
    kname[12] = "CO2";
    kname[13] = "CO";
    kname[14] = "CH2O";
    kname[15] = "CH2GSG";
    kname[16] = "CH3";
    kname[17] = "CH3O";
    kname[18] = "CH4";
    kname[19] = "CH3OH";
    kname[20] = "C2H6";
    kname[21] = "C2H5";
    kname[22] = "CH2CO";
    kname[23] = "HOCHO";
    kname[24] = "CH3O2";
    kname[25] = "CH3O2H";
    kname[26] = "C2H2";
    kname[27] = "HCCO";
    kname[28] = "C2H3";
    kname[29] = "CH2CHO";
    kname[30] = "C3H6";
    kname[31] = "C2H4";
    kname[32] = "C2H5O";
    kname[33] = "CH3CO";
    kname[34] = "C2H5O2";
    kname[35] = "C3H2";
    kname[36] = "C3H3";
    kname[37] = "C3H4XA";
    kname[38] = "C3H5XA";
    kname[39] = "NXC3H7";
    kname[40] = "NXC3H7O2";
    kname[41] = "C4H6";
    kname[42] = "C4H7";
    kname[43] = "C4H8X1";
    kname[44] = "PXC4H9";
    kname[45] = "PXC4H9O2";
    kname[46] = "C5H9";
    kname[47] = "C5H10X1";
    kname[48] = "C5H11X1";
    kname[49] = "C6H12X1";
    kname[50] = "C7H15X2";
    kname[51] = "NXC7H16";
}


/* Returns the char strings of species names */
void CKSYMS(int * kname, int * plenkname )
{
    int i; /*Loop Counter */
    int lenkname = *plenkname;
    /*clear kname */
    for (i=0; i<lenkname*52; i++) {
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

    /* CH  */
    kname[ 9*lenkname + 0 ] = 'C';
    kname[ 9*lenkname + 1 ] = 'H';
    kname[ 9*lenkname + 2 ] = ' ';

    /* HCO  */
    kname[ 10*lenkname + 0 ] = 'H';
    kname[ 10*lenkname + 1 ] = 'C';
    kname[ 10*lenkname + 2 ] = 'O';
    kname[ 10*lenkname + 3 ] = ' ';

    /* CH2  */
    kname[ 11*lenkname + 0 ] = 'C';
    kname[ 11*lenkname + 1 ] = 'H';
    kname[ 11*lenkname + 2 ] = '2';
    kname[ 11*lenkname + 3 ] = ' ';

    /* CO2  */
    kname[ 12*lenkname + 0 ] = 'C';
    kname[ 12*lenkname + 1 ] = 'O';
    kname[ 12*lenkname + 2 ] = '2';
    kname[ 12*lenkname + 3 ] = ' ';

    /* CO  */
    kname[ 13*lenkname + 0 ] = 'C';
    kname[ 13*lenkname + 1 ] = 'O';
    kname[ 13*lenkname + 2 ] = ' ';

    /* CH2O  */
    kname[ 14*lenkname + 0 ] = 'C';
    kname[ 14*lenkname + 1 ] = 'H';
    kname[ 14*lenkname + 2 ] = '2';
    kname[ 14*lenkname + 3 ] = 'O';
    kname[ 14*lenkname + 4 ] = ' ';

    /* CH2GSG  */
    kname[ 15*lenkname + 0 ] = 'C';
    kname[ 15*lenkname + 1 ] = 'H';
    kname[ 15*lenkname + 2 ] = '2';
    kname[ 15*lenkname + 3 ] = 'G';
    kname[ 15*lenkname + 4 ] = 'S';
    kname[ 15*lenkname + 5 ] = 'G';
    kname[ 15*lenkname + 6 ] = ' ';

    /* CH3  */
    kname[ 16*lenkname + 0 ] = 'C';
    kname[ 16*lenkname + 1 ] = 'H';
    kname[ 16*lenkname + 2 ] = '3';
    kname[ 16*lenkname + 3 ] = ' ';

    /* CH3O  */
    kname[ 17*lenkname + 0 ] = 'C';
    kname[ 17*lenkname + 1 ] = 'H';
    kname[ 17*lenkname + 2 ] = '3';
    kname[ 17*lenkname + 3 ] = 'O';
    kname[ 17*lenkname + 4 ] = ' ';

    /* CH4  */
    kname[ 18*lenkname + 0 ] = 'C';
    kname[ 18*lenkname + 1 ] = 'H';
    kname[ 18*lenkname + 2 ] = '4';
    kname[ 18*lenkname + 3 ] = ' ';

    /* CH3OH  */
    kname[ 19*lenkname + 0 ] = 'C';
    kname[ 19*lenkname + 1 ] = 'H';
    kname[ 19*lenkname + 2 ] = '3';
    kname[ 19*lenkname + 3 ] = 'O';
    kname[ 19*lenkname + 4 ] = 'H';
    kname[ 19*lenkname + 5 ] = ' ';

    /* C2H6  */
    kname[ 20*lenkname + 0 ] = 'C';
    kname[ 20*lenkname + 1 ] = '2';
    kname[ 20*lenkname + 2 ] = 'H';
    kname[ 20*lenkname + 3 ] = '6';
    kname[ 20*lenkname + 4 ] = ' ';

    /* C2H5  */
    kname[ 21*lenkname + 0 ] = 'C';
    kname[ 21*lenkname + 1 ] = '2';
    kname[ 21*lenkname + 2 ] = 'H';
    kname[ 21*lenkname + 3 ] = '5';
    kname[ 21*lenkname + 4 ] = ' ';

    /* CH2CO  */
    kname[ 22*lenkname + 0 ] = 'C';
    kname[ 22*lenkname + 1 ] = 'H';
    kname[ 22*lenkname + 2 ] = '2';
    kname[ 22*lenkname + 3 ] = 'C';
    kname[ 22*lenkname + 4 ] = 'O';
    kname[ 22*lenkname + 5 ] = ' ';

    /* HOCHO  */
    kname[ 23*lenkname + 0 ] = 'H';
    kname[ 23*lenkname + 1 ] = 'O';
    kname[ 23*lenkname + 2 ] = 'C';
    kname[ 23*lenkname + 3 ] = 'H';
    kname[ 23*lenkname + 4 ] = 'O';
    kname[ 23*lenkname + 5 ] = ' ';

    /* CH3O2  */
    kname[ 24*lenkname + 0 ] = 'C';
    kname[ 24*lenkname + 1 ] = 'H';
    kname[ 24*lenkname + 2 ] = '3';
    kname[ 24*lenkname + 3 ] = 'O';
    kname[ 24*lenkname + 4 ] = '2';
    kname[ 24*lenkname + 5 ] = ' ';

    /* CH3O2H  */
    kname[ 25*lenkname + 0 ] = 'C';
    kname[ 25*lenkname + 1 ] = 'H';
    kname[ 25*lenkname + 2 ] = '3';
    kname[ 25*lenkname + 3 ] = 'O';
    kname[ 25*lenkname + 4 ] = '2';
    kname[ 25*lenkname + 5 ] = 'H';
    kname[ 25*lenkname + 6 ] = ' ';

    /* C2H2  */
    kname[ 26*lenkname + 0 ] = 'C';
    kname[ 26*lenkname + 1 ] = '2';
    kname[ 26*lenkname + 2 ] = 'H';
    kname[ 26*lenkname + 3 ] = '2';
    kname[ 26*lenkname + 4 ] = ' ';

    /* HCCO  */
    kname[ 27*lenkname + 0 ] = 'H';
    kname[ 27*lenkname + 1 ] = 'C';
    kname[ 27*lenkname + 2 ] = 'C';
    kname[ 27*lenkname + 3 ] = 'O';
    kname[ 27*lenkname + 4 ] = ' ';

    /* C2H3  */
    kname[ 28*lenkname + 0 ] = 'C';
    kname[ 28*lenkname + 1 ] = '2';
    kname[ 28*lenkname + 2 ] = 'H';
    kname[ 28*lenkname + 3 ] = '3';
    kname[ 28*lenkname + 4 ] = ' ';

    /* CH2CHO  */
    kname[ 29*lenkname + 0 ] = 'C';
    kname[ 29*lenkname + 1 ] = 'H';
    kname[ 29*lenkname + 2 ] = '2';
    kname[ 29*lenkname + 3 ] = 'C';
    kname[ 29*lenkname + 4 ] = 'H';
    kname[ 29*lenkname + 5 ] = 'O';
    kname[ 29*lenkname + 6 ] = ' ';

    /* C3H6  */
    kname[ 30*lenkname + 0 ] = 'C';
    kname[ 30*lenkname + 1 ] = '3';
    kname[ 30*lenkname + 2 ] = 'H';
    kname[ 30*lenkname + 3 ] = '6';
    kname[ 30*lenkname + 4 ] = ' ';

    /* C2H4  */
    kname[ 31*lenkname + 0 ] = 'C';
    kname[ 31*lenkname + 1 ] = '2';
    kname[ 31*lenkname + 2 ] = 'H';
    kname[ 31*lenkname + 3 ] = '4';
    kname[ 31*lenkname + 4 ] = ' ';

    /* C2H5O  */
    kname[ 32*lenkname + 0 ] = 'C';
    kname[ 32*lenkname + 1 ] = '2';
    kname[ 32*lenkname + 2 ] = 'H';
    kname[ 32*lenkname + 3 ] = '5';
    kname[ 32*lenkname + 4 ] = 'O';
    kname[ 32*lenkname + 5 ] = ' ';

    /* CH3CO  */
    kname[ 33*lenkname + 0 ] = 'C';
    kname[ 33*lenkname + 1 ] = 'H';
    kname[ 33*lenkname + 2 ] = '3';
    kname[ 33*lenkname + 3 ] = 'C';
    kname[ 33*lenkname + 4 ] = 'O';
    kname[ 33*lenkname + 5 ] = ' ';

    /* C2H5O2  */
    kname[ 34*lenkname + 0 ] = 'C';
    kname[ 34*lenkname + 1 ] = '2';
    kname[ 34*lenkname + 2 ] = 'H';
    kname[ 34*lenkname + 3 ] = '5';
    kname[ 34*lenkname + 4 ] = 'O';
    kname[ 34*lenkname + 5 ] = '2';
    kname[ 34*lenkname + 6 ] = ' ';

    /* C3H2  */
    kname[ 35*lenkname + 0 ] = 'C';
    kname[ 35*lenkname + 1 ] = '3';
    kname[ 35*lenkname + 2 ] = 'H';
    kname[ 35*lenkname + 3 ] = '2';
    kname[ 35*lenkname + 4 ] = ' ';

    /* C3H3  */
    kname[ 36*lenkname + 0 ] = 'C';
    kname[ 36*lenkname + 1 ] = '3';
    kname[ 36*lenkname + 2 ] = 'H';
    kname[ 36*lenkname + 3 ] = '3';
    kname[ 36*lenkname + 4 ] = ' ';

    /* C3H4XA  */
    kname[ 37*lenkname + 0 ] = 'C';
    kname[ 37*lenkname + 1 ] = '3';
    kname[ 37*lenkname + 2 ] = 'H';
    kname[ 37*lenkname + 3 ] = '4';
    kname[ 37*lenkname + 4 ] = 'X';
    kname[ 37*lenkname + 5 ] = 'A';
    kname[ 37*lenkname + 6 ] = ' ';

    /* C3H5XA  */
    kname[ 38*lenkname + 0 ] = 'C';
    kname[ 38*lenkname + 1 ] = '3';
    kname[ 38*lenkname + 2 ] = 'H';
    kname[ 38*lenkname + 3 ] = '5';
    kname[ 38*lenkname + 4 ] = 'X';
    kname[ 38*lenkname + 5 ] = 'A';
    kname[ 38*lenkname + 6 ] = ' ';

    /* NXC3H7  */
    kname[ 39*lenkname + 0 ] = 'N';
    kname[ 39*lenkname + 1 ] = 'X';
    kname[ 39*lenkname + 2 ] = 'C';
    kname[ 39*lenkname + 3 ] = '3';
    kname[ 39*lenkname + 4 ] = 'H';
    kname[ 39*lenkname + 5 ] = '7';
    kname[ 39*lenkname + 6 ] = ' ';

    /* NXC3H7O2  */
    kname[ 40*lenkname + 0 ] = 'N';
    kname[ 40*lenkname + 1 ] = 'X';
    kname[ 40*lenkname + 2 ] = 'C';
    kname[ 40*lenkname + 3 ] = '3';
    kname[ 40*lenkname + 4 ] = 'H';
    kname[ 40*lenkname + 5 ] = '7';
    kname[ 40*lenkname + 6 ] = 'O';
    kname[ 40*lenkname + 7 ] = '2';
    kname[ 40*lenkname + 8 ] = ' ';

    /* C4H6  */
    kname[ 41*lenkname + 0 ] = 'C';
    kname[ 41*lenkname + 1 ] = '4';
    kname[ 41*lenkname + 2 ] = 'H';
    kname[ 41*lenkname + 3 ] = '6';
    kname[ 41*lenkname + 4 ] = ' ';

    /* C4H7  */
    kname[ 42*lenkname + 0 ] = 'C';
    kname[ 42*lenkname + 1 ] = '4';
    kname[ 42*lenkname + 2 ] = 'H';
    kname[ 42*lenkname + 3 ] = '7';
    kname[ 42*lenkname + 4 ] = ' ';

    /* C4H8X1  */
    kname[ 43*lenkname + 0 ] = 'C';
    kname[ 43*lenkname + 1 ] = '4';
    kname[ 43*lenkname + 2 ] = 'H';
    kname[ 43*lenkname + 3 ] = '8';
    kname[ 43*lenkname + 4 ] = 'X';
    kname[ 43*lenkname + 5 ] = '1';
    kname[ 43*lenkname + 6 ] = ' ';

    /* PXC4H9  */
    kname[ 44*lenkname + 0 ] = 'P';
    kname[ 44*lenkname + 1 ] = 'X';
    kname[ 44*lenkname + 2 ] = 'C';
    kname[ 44*lenkname + 3 ] = '4';
    kname[ 44*lenkname + 4 ] = 'H';
    kname[ 44*lenkname + 5 ] = '9';
    kname[ 44*lenkname + 6 ] = ' ';

    /* PXC4H9O2  */
    kname[ 45*lenkname + 0 ] = 'P';
    kname[ 45*lenkname + 1 ] = 'X';
    kname[ 45*lenkname + 2 ] = 'C';
    kname[ 45*lenkname + 3 ] = '4';
    kname[ 45*lenkname + 4 ] = 'H';
    kname[ 45*lenkname + 5 ] = '9';
    kname[ 45*lenkname + 6 ] = 'O';
    kname[ 45*lenkname + 7 ] = '2';
    kname[ 45*lenkname + 8 ] = ' ';

    /* C5H9  */
    kname[ 46*lenkname + 0 ] = 'C';
    kname[ 46*lenkname + 1 ] = '5';
    kname[ 46*lenkname + 2 ] = 'H';
    kname[ 46*lenkname + 3 ] = '9';
    kname[ 46*lenkname + 4 ] = ' ';

    /* C5H10X1  */
    kname[ 47*lenkname + 0 ] = 'C';
    kname[ 47*lenkname + 1 ] = '5';
    kname[ 47*lenkname + 2 ] = 'H';
    kname[ 47*lenkname + 3 ] = '1';
    kname[ 47*lenkname + 4 ] = '0';
    kname[ 47*lenkname + 5 ] = 'X';
    kname[ 47*lenkname + 6 ] = '1';
    kname[ 47*lenkname + 7 ] = ' ';

    /* C5H11X1  */
    kname[ 48*lenkname + 0 ] = 'C';
    kname[ 48*lenkname + 1 ] = '5';
    kname[ 48*lenkname + 2 ] = 'H';
    kname[ 48*lenkname + 3 ] = '1';
    kname[ 48*lenkname + 4 ] = '1';
    kname[ 48*lenkname + 5 ] = 'X';
    kname[ 48*lenkname + 6 ] = '1';
    kname[ 48*lenkname + 7 ] = ' ';

    /* C6H12X1  */
    kname[ 49*lenkname + 0 ] = 'C';
    kname[ 49*lenkname + 1 ] = '6';
    kname[ 49*lenkname + 2 ] = 'H';
    kname[ 49*lenkname + 3 ] = '1';
    kname[ 49*lenkname + 4 ] = '2';
    kname[ 49*lenkname + 5 ] = 'X';
    kname[ 49*lenkname + 6 ] = '1';
    kname[ 49*lenkname + 7 ] = ' ';

    /* C7H15X2  */
    kname[ 50*lenkname + 0 ] = 'C';
    kname[ 50*lenkname + 1 ] = '7';
    kname[ 50*lenkname + 2 ] = 'H';
    kname[ 50*lenkname + 3 ] = '1';
    kname[ 50*lenkname + 4 ] = '5';
    kname[ 50*lenkname + 5 ] = 'X';
    kname[ 50*lenkname + 6 ] = '2';
    kname[ 50*lenkname + 7 ] = ' ';

    /* NXC7H16  */
    kname[ 51*lenkname + 0 ] = 'N';
    kname[ 51*lenkname + 1 ] = 'X';
    kname[ 51*lenkname + 2 ] = 'C';
    kname[ 51*lenkname + 3 ] = '7';
    kname[ 51*lenkname + 4 ] = 'H';
    kname[ 51*lenkname + 5 ] = '1';
    kname[ 51*lenkname + 6 ] = '6';
    kname[ 51*lenkname + 7 ] = ' ';

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
    XW += x[9]*molecular_weights[9]; /*CH */
    XW += x[10]*molecular_weights[10]; /*HCO */
    XW += x[11]*molecular_weights[11]; /*CH2 */
    XW += x[12]*molecular_weights[12]; /*CO2 */
    XW += x[13]*molecular_weights[13]; /*CO */
    XW += x[14]*molecular_weights[14]; /*CH2O */
    XW += x[15]*molecular_weights[15]; /*CH2GSG */
    XW += x[16]*molecular_weights[16]; /*CH3 */
    XW += x[17]*molecular_weights[17]; /*CH3O */
    XW += x[18]*molecular_weights[18]; /*CH4 */
    XW += x[19]*molecular_weights[19]; /*CH3OH */
    XW += x[20]*molecular_weights[20]; /*C2H6 */
    XW += x[21]*molecular_weights[21]; /*C2H5 */
    XW += x[22]*molecular_weights[22]; /*CH2CO */
    XW += x[23]*molecular_weights[23]; /*HOCHO */
    XW += x[24]*molecular_weights[24]; /*CH3O2 */
    XW += x[25]*molecular_weights[25]; /*CH3O2H */
    XW += x[26]*molecular_weights[26]; /*C2H2 */
    XW += x[27]*molecular_weights[27]; /*HCCO */
    XW += x[28]*molecular_weights[28]; /*C2H3 */
    XW += x[29]*molecular_weights[29]; /*CH2CHO */
    XW += x[30]*molecular_weights[30]; /*C3H6 */
    XW += x[31]*molecular_weights[31]; /*C2H4 */
    XW += x[32]*molecular_weights[32]; /*C2H5O */
    XW += x[33]*molecular_weights[33]; /*CH3CO */
    XW += x[34]*molecular_weights[34]; /*C2H5O2 */
    XW += x[35]*molecular_weights[35]; /*C3H2 */
    XW += x[36]*molecular_weights[36]; /*C3H3 */
    XW += x[37]*molecular_weights[37]; /*C3H4XA */
    XW += x[38]*molecular_weights[38]; /*C3H5XA */
    XW += x[39]*molecular_weights[39]; /*NXC3H7 */
    XW += x[40]*molecular_weights[40]; /*NXC3H7O2 */
    XW += x[41]*molecular_weights[41]; /*C4H6 */
    XW += x[42]*molecular_weights[42]; /*C4H7 */
    XW += x[43]*molecular_weights[43]; /*C4H8X1 */
    XW += x[44]*molecular_weights[44]; /*PXC4H9 */
    XW += x[45]*molecular_weights[45]; /*PXC4H9O2 */
    XW += x[46]*molecular_weights[46]; /*C5H9 */
    XW += x[47]*molecular_weights[47]; /*C5H10X1 */
    XW += x[48]*molecular_weights[48]; /*C5H11X1 */
    XW += x[49]*molecular_weights[49]; /*C6H12X1 */
    XW += x[50]*molecular_weights[50]; /*C7H15X2 */
    XW += x[51]*molecular_weights[51]; /*NXC7H16 */
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
    YOW += y[9]*imw[9]; /*CH */
    YOW += y[10]*imw[10]; /*HCO */
    YOW += y[11]*imw[11]; /*CH2 */
    YOW += y[12]*imw[12]; /*CO2 */
    YOW += y[13]*imw[13]; /*CO */
    YOW += y[14]*imw[14]; /*CH2O */
    YOW += y[15]*imw[15]; /*CH2GSG */
    YOW += y[16]*imw[16]; /*CH3 */
    YOW += y[17]*imw[17]; /*CH3O */
    YOW += y[18]*imw[18]; /*CH4 */
    YOW += y[19]*imw[19]; /*CH3OH */
    YOW += y[20]*imw[20]; /*C2H6 */
    YOW += y[21]*imw[21]; /*C2H5 */
    YOW += y[22]*imw[22]; /*CH2CO */
    YOW += y[23]*imw[23]; /*HOCHO */
    YOW += y[24]*imw[24]; /*CH3O2 */
    YOW += y[25]*imw[25]; /*CH3O2H */
    YOW += y[26]*imw[26]; /*C2H2 */
    YOW += y[27]*imw[27]; /*HCCO */
    YOW += y[28]*imw[28]; /*C2H3 */
    YOW += y[29]*imw[29]; /*CH2CHO */
    YOW += y[30]*imw[30]; /*C3H6 */
    YOW += y[31]*imw[31]; /*C2H4 */
    YOW += y[32]*imw[32]; /*C2H5O */
    YOW += y[33]*imw[33]; /*CH3CO */
    YOW += y[34]*imw[34]; /*C2H5O2 */
    YOW += y[35]*imw[35]; /*C3H2 */
    YOW += y[36]*imw[36]; /*C3H3 */
    YOW += y[37]*imw[37]; /*C3H4XA */
    YOW += y[38]*imw[38]; /*C3H5XA */
    YOW += y[39]*imw[39]; /*NXC3H7 */
    YOW += y[40]*imw[40]; /*NXC3H7O2 */
    YOW += y[41]*imw[41]; /*C4H6 */
    YOW += y[42]*imw[42]; /*C4H7 */
    YOW += y[43]*imw[43]; /*C4H8X1 */
    YOW += y[44]*imw[44]; /*PXC4H9 */
    YOW += y[45]*imw[45]; /*PXC4H9O2 */
    YOW += y[46]*imw[46]; /*C5H9 */
    YOW += y[47]*imw[47]; /*C5H10X1 */
    YOW += y[48]*imw[48]; /*C5H11X1 */
    YOW += y[49]*imw[49]; /*C6H12X1 */
    YOW += y[50]*imw[50]; /*C7H15X2 */
    YOW += y[51]*imw[51]; /*NXC7H16 */
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
    W += c[9]*13.019120; /*CH */
    W += c[10]*29.018520; /*HCO */
    W += c[11]*14.027090; /*CH2 */
    W += c[12]*44.009950; /*CO2 */
    W += c[13]*28.010550; /*CO */
    W += c[14]*30.026490; /*CH2O */
    W += c[15]*14.027090; /*CH2GSG */
    W += c[16]*15.035060; /*CH3 */
    W += c[17]*31.034460; /*CH3O */
    W += c[18]*16.043030; /*CH4 */
    W += c[19]*32.042430; /*CH3OH */
    W += c[20]*30.070120; /*C2H6 */
    W += c[21]*29.062150; /*C2H5 */
    W += c[22]*42.037640; /*CH2CO */
    W += c[23]*46.025890; /*HOCHO */
    W += c[24]*47.033860; /*CH3O2 */
    W += c[25]*48.041830; /*CH3O2H */
    W += c[26]*26.038240; /*C2H2 */
    W += c[27]*41.029670; /*HCCO */
    W += c[28]*27.046210; /*C2H3 */
    W += c[29]*43.045610; /*CH2CHO */
    W += c[30]*42.081270; /*C3H6 */
    W += c[31]*28.054180; /*C2H4 */
    W += c[32]*45.061550; /*C2H5O */
    W += c[33]*43.045610; /*CH3CO */
    W += c[34]*61.060950; /*C2H5O2 */
    W += c[35]*38.049390; /*C3H2 */
    W += c[36]*39.057360; /*C3H3 */
    W += c[37]*40.065330; /*C3H4XA */
    W += c[38]*41.073300; /*C3H5XA */
    W += c[39]*43.089240; /*NXC3H7 */
    W += c[40]*75.088040; /*NXC3H7O2 */
    W += c[41]*54.092420; /*C4H6 */
    W += c[42]*55.100390; /*C4H7 */
    W += c[43]*56.108360; /*C4H8X1 */
    W += c[44]*57.116330; /*PXC4H9 */
    W += c[45]*89.115130; /*PXC4H9O2 */
    W += c[46]*69.127480; /*C5H9 */
    W += c[47]*70.135450; /*C5H10X1 */
    W += c[48]*71.143420; /*C5H11X1 */
    W += c[49]*84.162540; /*C6H12X1 */
    W += c[50]*99.197600; /*C7H15X2 */
    W += c[51]*100.205570; /*NXC7H16 */

    for (id = 0; id < 52; ++id) {
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
    XW += x[9]*13.019120; /*CH */
    XW += x[10]*29.018520; /*HCO */
    XW += x[11]*14.027090; /*CH2 */
    XW += x[12]*44.009950; /*CO2 */
    XW += x[13]*28.010550; /*CO */
    XW += x[14]*30.026490; /*CH2O */
    XW += x[15]*14.027090; /*CH2GSG */
    XW += x[16]*15.035060; /*CH3 */
    XW += x[17]*31.034460; /*CH3O */
    XW += x[18]*16.043030; /*CH4 */
    XW += x[19]*32.042430; /*CH3OH */
    XW += x[20]*30.070120; /*C2H6 */
    XW += x[21]*29.062150; /*C2H5 */
    XW += x[22]*42.037640; /*CH2CO */
    XW += x[23]*46.025890; /*HOCHO */
    XW += x[24]*47.033860; /*CH3O2 */
    XW += x[25]*48.041830; /*CH3O2H */
    XW += x[26]*26.038240; /*C2H2 */
    XW += x[27]*41.029670; /*HCCO */
    XW += x[28]*27.046210; /*C2H3 */
    XW += x[29]*43.045610; /*CH2CHO */
    XW += x[30]*42.081270; /*C3H6 */
    XW += x[31]*28.054180; /*C2H4 */
    XW += x[32]*45.061550; /*C2H5O */
    XW += x[33]*43.045610; /*CH3CO */
    XW += x[34]*61.060950; /*C2H5O2 */
    XW += x[35]*38.049390; /*C3H2 */
    XW += x[36]*39.057360; /*C3H3 */
    XW += x[37]*40.065330; /*C3H4XA */
    XW += x[38]*41.073300; /*C3H5XA */
    XW += x[39]*43.089240; /*NXC3H7 */
    XW += x[40]*75.088040; /*NXC3H7O2 */
    XW += x[41]*54.092420; /*C4H6 */
    XW += x[42]*55.100390; /*C4H7 */
    XW += x[43]*56.108360; /*C4H8X1 */
    XW += x[44]*57.116330; /*PXC4H9 */
    XW += x[45]*89.115130; /*PXC4H9O2 */
    XW += x[46]*69.127480; /*C5H9 */
    XW += x[47]*70.135450; /*C5H10X1 */
    XW += x[48]*71.143420; /*C5H11X1 */
    XW += x[49]*84.162540; /*C6H12X1 */
    XW += x[50]*99.197600; /*C7H15X2 */
    XW += x[51]*100.205570; /*NXC7H16 */
    *rho = *P * XW / (8.31446e+07 * (*T)); /*rho = P*W/(R*T) */

    return;
}


/*Compute rho = P*W(y)/RT */
AMREX_GPU_HOST_DEVICE void CKRHOY(double *  P, double *  T, double *  y,  double *  rho)
{
    double YOW = 0;
    double tmp[52];

    for (int i = 0; i < 52; i++)
    {
        tmp[i] = y[i]*imw[i];
    }
    for (int i = 0; i < 52; i++)
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
    W += c[9]*13.019120; /*CH */
    W += c[10]*29.018520; /*HCO */
    W += c[11]*14.027090; /*CH2 */
    W += c[12]*44.009950; /*CO2 */
    W += c[13]*28.010550; /*CO */
    W += c[14]*30.026490; /*CH2O */
    W += c[15]*14.027090; /*CH2GSG */
    W += c[16]*15.035060; /*CH3 */
    W += c[17]*31.034460; /*CH3O */
    W += c[18]*16.043030; /*CH4 */
    W += c[19]*32.042430; /*CH3OH */
    W += c[20]*30.070120; /*C2H6 */
    W += c[21]*29.062150; /*C2H5 */
    W += c[22]*42.037640; /*CH2CO */
    W += c[23]*46.025890; /*HOCHO */
    W += c[24]*47.033860; /*CH3O2 */
    W += c[25]*48.041830; /*CH3O2H */
    W += c[26]*26.038240; /*C2H2 */
    W += c[27]*41.029670; /*HCCO */
    W += c[28]*27.046210; /*C2H3 */
    W += c[29]*43.045610; /*CH2CHO */
    W += c[30]*42.081270; /*C3H6 */
    W += c[31]*28.054180; /*C2H4 */
    W += c[32]*45.061550; /*C2H5O */
    W += c[33]*43.045610; /*CH3CO */
    W += c[34]*61.060950; /*C2H5O2 */
    W += c[35]*38.049390; /*C3H2 */
    W += c[36]*39.057360; /*C3H3 */
    W += c[37]*40.065330; /*C3H4XA */
    W += c[38]*41.073300; /*C3H5XA */
    W += c[39]*43.089240; /*NXC3H7 */
    W += c[40]*75.088040; /*NXC3H7O2 */
    W += c[41]*54.092420; /*C4H6 */
    W += c[42]*55.100390; /*C4H7 */
    W += c[43]*56.108360; /*C4H8X1 */
    W += c[44]*57.116330; /*PXC4H9 */
    W += c[45]*89.115130; /*PXC4H9O2 */
    W += c[46]*69.127480; /*C5H9 */
    W += c[47]*70.135450; /*C5H10X1 */
    W += c[48]*71.143420; /*C5H11X1 */
    W += c[49]*84.162540; /*C6H12X1 */
    W += c[50]*99.197600; /*C7H15X2 */
    W += c[51]*100.205570; /*NXC7H16 */

    for (id = 0; id < 52; ++id) {
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
    double tmp[52];

    for (int i = 0; i < 52; i++)
    {
        tmp[i] = y[i]*imw[i];
    }
    for (int i = 0; i < 52; i++)
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
    XW += x[9]*molecular_weights[9]; /*CH */
    XW += x[10]*molecular_weights[10]; /*HCO */
    XW += x[11]*molecular_weights[11]; /*CH2 */
    XW += x[12]*molecular_weights[12]; /*CO2 */
    XW += x[13]*molecular_weights[13]; /*CO */
    XW += x[14]*molecular_weights[14]; /*CH2O */
    XW += x[15]*molecular_weights[15]; /*CH2GSG */
    XW += x[16]*molecular_weights[16]; /*CH3 */
    XW += x[17]*molecular_weights[17]; /*CH3O */
    XW += x[18]*molecular_weights[18]; /*CH4 */
    XW += x[19]*molecular_weights[19]; /*CH3OH */
    XW += x[20]*molecular_weights[20]; /*C2H6 */
    XW += x[21]*molecular_weights[21]; /*C2H5 */
    XW += x[22]*molecular_weights[22]; /*CH2CO */
    XW += x[23]*molecular_weights[23]; /*HOCHO */
    XW += x[24]*molecular_weights[24]; /*CH3O2 */
    XW += x[25]*molecular_weights[25]; /*CH3O2H */
    XW += x[26]*molecular_weights[26]; /*C2H2 */
    XW += x[27]*molecular_weights[27]; /*HCCO */
    XW += x[28]*molecular_weights[28]; /*C2H3 */
    XW += x[29]*molecular_weights[29]; /*CH2CHO */
    XW += x[30]*molecular_weights[30]; /*C3H6 */
    XW += x[31]*molecular_weights[31]; /*C2H4 */
    XW += x[32]*molecular_weights[32]; /*C2H5O */
    XW += x[33]*molecular_weights[33]; /*CH3CO */
    XW += x[34]*molecular_weights[34]; /*C2H5O2 */
    XW += x[35]*molecular_weights[35]; /*C3H2 */
    XW += x[36]*molecular_weights[36]; /*C3H3 */
    XW += x[37]*molecular_weights[37]; /*C3H4XA */
    XW += x[38]*molecular_weights[38]; /*C3H5XA */
    XW += x[39]*molecular_weights[39]; /*NXC3H7 */
    XW += x[40]*molecular_weights[40]; /*NXC3H7O2 */
    XW += x[41]*molecular_weights[41]; /*C4H6 */
    XW += x[42]*molecular_weights[42]; /*C4H7 */
    XW += x[43]*molecular_weights[43]; /*C4H8X1 */
    XW += x[44]*molecular_weights[44]; /*PXC4H9 */
    XW += x[45]*molecular_weights[45]; /*PXC4H9O2 */
    XW += x[46]*molecular_weights[46]; /*C5H9 */
    XW += x[47]*molecular_weights[47]; /*C5H10X1 */
    XW += x[48]*molecular_weights[48]; /*C5H11X1 */
    XW += x[49]*molecular_weights[49]; /*C6H12X1 */
    XW += x[50]*molecular_weights[50]; /*C7H15X2 */
    XW += x[51]*molecular_weights[51]; /*NXC7H16 */
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
    W += c[9]*molecular_weights[9]; /*CH */
    W += c[10]*molecular_weights[10]; /*HCO */
    W += c[11]*molecular_weights[11]; /*CH2 */
    W += c[12]*molecular_weights[12]; /*CO2 */
    W += c[13]*molecular_weights[13]; /*CO */
    W += c[14]*molecular_weights[14]; /*CH2O */
    W += c[15]*molecular_weights[15]; /*CH2GSG */
    W += c[16]*molecular_weights[16]; /*CH3 */
    W += c[17]*molecular_weights[17]; /*CH3O */
    W += c[18]*molecular_weights[18]; /*CH4 */
    W += c[19]*molecular_weights[19]; /*CH3OH */
    W += c[20]*molecular_weights[20]; /*C2H6 */
    W += c[21]*molecular_weights[21]; /*C2H5 */
    W += c[22]*molecular_weights[22]; /*CH2CO */
    W += c[23]*molecular_weights[23]; /*HOCHO */
    W += c[24]*molecular_weights[24]; /*CH3O2 */
    W += c[25]*molecular_weights[25]; /*CH3O2H */
    W += c[26]*molecular_weights[26]; /*C2H2 */
    W += c[27]*molecular_weights[27]; /*HCCO */
    W += c[28]*molecular_weights[28]; /*C2H3 */
    W += c[29]*molecular_weights[29]; /*CH2CHO */
    W += c[30]*molecular_weights[30]; /*C3H6 */
    W += c[31]*molecular_weights[31]; /*C2H4 */
    W += c[32]*molecular_weights[32]; /*C2H5O */
    W += c[33]*molecular_weights[33]; /*CH3CO */
    W += c[34]*molecular_weights[34]; /*C2H5O2 */
    W += c[35]*molecular_weights[35]; /*C3H2 */
    W += c[36]*molecular_weights[36]; /*C3H3 */
    W += c[37]*molecular_weights[37]; /*C3H4XA */
    W += c[38]*molecular_weights[38]; /*C3H5XA */
    W += c[39]*molecular_weights[39]; /*NXC3H7 */
    W += c[40]*molecular_weights[40]; /*NXC3H7O2 */
    W += c[41]*molecular_weights[41]; /*C4H6 */
    W += c[42]*molecular_weights[42]; /*C4H7 */
    W += c[43]*molecular_weights[43]; /*C4H8X1 */
    W += c[44]*molecular_weights[44]; /*PXC4H9 */
    W += c[45]*molecular_weights[45]; /*PXC4H9O2 */
    W += c[46]*molecular_weights[46]; /*C5H9 */
    W += c[47]*molecular_weights[47]; /*C5H10X1 */
    W += c[48]*molecular_weights[48]; /*C5H11X1 */
    W += c[49]*molecular_weights[49]; /*C6H12X1 */
    W += c[50]*molecular_weights[50]; /*C7H15X2 */
    W += c[51]*molecular_weights[51]; /*NXC7H16 */

    for (id = 0; id < 52; ++id) {
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
    double tmp[52];

    for (int i = 0; i < 52; i++)
    {
        tmp[i] = y[i]*imw[i];
    }
    for (int i = 0; i < 52; i++)
    {
        YOW += tmp[i];
    }

    double YOWINV = 1.0/YOW;

    for (int i = 0; i < 52; i++)
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
    for (int i = 0; i < 52; i++)
    {
        c[i] = y[i]*imw[i];
    }
    for (int i = 0; i < 52; i++)
    {
        YOW += c[i];
    }

    /*PW/RT (see Eq. 7) */
    PWORT = (*P)/(YOW * 8.31446e+07 * (*T)); 
    /*Now compute conversion */

    for (int i = 0; i < 52; i++)
    {
        c[i] = PWORT * y[i] * imw[i];
    }
    return;
}


/*convert y[species] (mass fracs) to c[species] (molar conc) */
AMREX_GPU_HOST_DEVICE void CKYTCR(double *  rho, double *  T, double *  y,  double *  c)
{
    for (int i = 0; i < 52; i++)
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
    XW += x[9]*13.019120; /*CH */
    XW += x[10]*29.018520; /*HCO */
    XW += x[11]*14.027090; /*CH2 */
    XW += x[12]*44.009950; /*CO2 */
    XW += x[13]*28.010550; /*CO */
    XW += x[14]*30.026490; /*CH2O */
    XW += x[15]*14.027090; /*CH2GSG */
    XW += x[16]*15.035060; /*CH3 */
    XW += x[17]*31.034460; /*CH3O */
    XW += x[18]*16.043030; /*CH4 */
    XW += x[19]*32.042430; /*CH3OH */
    XW += x[20]*30.070120; /*C2H6 */
    XW += x[21]*29.062150; /*C2H5 */
    XW += x[22]*42.037640; /*CH2CO */
    XW += x[23]*46.025890; /*HOCHO */
    XW += x[24]*47.033860; /*CH3O2 */
    XW += x[25]*48.041830; /*CH3O2H */
    XW += x[26]*26.038240; /*C2H2 */
    XW += x[27]*41.029670; /*HCCO */
    XW += x[28]*27.046210; /*C2H3 */
    XW += x[29]*43.045610; /*CH2CHO */
    XW += x[30]*42.081270; /*C3H6 */
    XW += x[31]*28.054180; /*C2H4 */
    XW += x[32]*45.061550; /*C2H5O */
    XW += x[33]*43.045610; /*CH3CO */
    XW += x[34]*61.060950; /*C2H5O2 */
    XW += x[35]*38.049390; /*C3H2 */
    XW += x[36]*39.057360; /*C3H3 */
    XW += x[37]*40.065330; /*C3H4XA */
    XW += x[38]*41.073300; /*C3H5XA */
    XW += x[39]*43.089240; /*NXC3H7 */
    XW += x[40]*75.088040; /*NXC3H7O2 */
    XW += x[41]*54.092420; /*C4H6 */
    XW += x[42]*55.100390; /*C4H7 */
    XW += x[43]*56.108360; /*C4H8X1 */
    XW += x[44]*57.116330; /*PXC4H9 */
    XW += x[45]*89.115130; /*PXC4H9O2 */
    XW += x[46]*69.127480; /*C5H9 */
    XW += x[47]*70.135450; /*C5H10X1 */
    XW += x[48]*71.143420; /*C5H11X1 */
    XW += x[49]*84.162540; /*C6H12X1 */
    XW += x[50]*99.197600; /*C7H15X2 */
    XW += x[51]*100.205570; /*NXC7H16 */

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
    y[9] = x[9]*13.019120*XWinv; 
    y[10] = x[10]*29.018520*XWinv; 
    y[11] = x[11]*14.027090*XWinv; 
    y[12] = x[12]*44.009950*XWinv; 
    y[13] = x[13]*28.010550*XWinv; 
    y[14] = x[14]*30.026490*XWinv; 
    y[15] = x[15]*14.027090*XWinv; 
    y[16] = x[16]*15.035060*XWinv; 
    y[17] = x[17]*31.034460*XWinv; 
    y[18] = x[18]*16.043030*XWinv; 
    y[19] = x[19]*32.042430*XWinv; 
    y[20] = x[20]*30.070120*XWinv; 
    y[21] = x[21]*29.062150*XWinv; 
    y[22] = x[22]*42.037640*XWinv; 
    y[23] = x[23]*46.025890*XWinv; 
    y[24] = x[24]*47.033860*XWinv; 
    y[25] = x[25]*48.041830*XWinv; 
    y[26] = x[26]*26.038240*XWinv; 
    y[27] = x[27]*41.029670*XWinv; 
    y[28] = x[28]*27.046210*XWinv; 
    y[29] = x[29]*43.045610*XWinv; 
    y[30] = x[30]*42.081270*XWinv; 
    y[31] = x[31]*28.054180*XWinv; 
    y[32] = x[32]*45.061550*XWinv; 
    y[33] = x[33]*43.045610*XWinv; 
    y[34] = x[34]*61.060950*XWinv; 
    y[35] = x[35]*38.049390*XWinv; 
    y[36] = x[36]*39.057360*XWinv; 
    y[37] = x[37]*40.065330*XWinv; 
    y[38] = x[38]*41.073300*XWinv; 
    y[39] = x[39]*43.089240*XWinv; 
    y[40] = x[40]*75.088040*XWinv; 
    y[41] = x[41]*54.092420*XWinv; 
    y[42] = x[42]*55.100390*XWinv; 
    y[43] = x[43]*56.108360*XWinv; 
    y[44] = x[44]*57.116330*XWinv; 
    y[45] = x[45]*89.115130*XWinv; 
    y[46] = x[46]*69.127480*XWinv; 
    y[47] = x[47]*70.135450*XWinv; 
    y[48] = x[48]*71.143420*XWinv; 
    y[49] = x[49]*84.162540*XWinv; 
    y[50] = x[50]*99.197600*XWinv; 
    y[51] = x[51]*100.205570*XWinv; 

    return;
}


/*convert x[species] (mole fracs) to c[species] (molar conc) */
void CKXTCP(double *  P, double *  T, double *  x,  double *  c)
{
    int id; /*loop counter */
    double PORT = (*P)/(8.31446e+07 * (*T)); /*P/RT */

    /*Compute conversion, see Eq 10 */
    for (id = 0; id < 52; ++id) {
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
    XW += x[9]*13.019120; /*CH */
    XW += x[10]*29.018520; /*HCO */
    XW += x[11]*14.027090; /*CH2 */
    XW += x[12]*44.009950; /*CO2 */
    XW += x[13]*28.010550; /*CO */
    XW += x[14]*30.026490; /*CH2O */
    XW += x[15]*14.027090; /*CH2GSG */
    XW += x[16]*15.035060; /*CH3 */
    XW += x[17]*31.034460; /*CH3O */
    XW += x[18]*16.043030; /*CH4 */
    XW += x[19]*32.042430; /*CH3OH */
    XW += x[20]*30.070120; /*C2H6 */
    XW += x[21]*29.062150; /*C2H5 */
    XW += x[22]*42.037640; /*CH2CO */
    XW += x[23]*46.025890; /*HOCHO */
    XW += x[24]*47.033860; /*CH3O2 */
    XW += x[25]*48.041830; /*CH3O2H */
    XW += x[26]*26.038240; /*C2H2 */
    XW += x[27]*41.029670; /*HCCO */
    XW += x[28]*27.046210; /*C2H3 */
    XW += x[29]*43.045610; /*CH2CHO */
    XW += x[30]*42.081270; /*C3H6 */
    XW += x[31]*28.054180; /*C2H4 */
    XW += x[32]*45.061550; /*C2H5O */
    XW += x[33]*43.045610; /*CH3CO */
    XW += x[34]*61.060950; /*C2H5O2 */
    XW += x[35]*38.049390; /*C3H2 */
    XW += x[36]*39.057360; /*C3H3 */
    XW += x[37]*40.065330; /*C3H4XA */
    XW += x[38]*41.073300; /*C3H5XA */
    XW += x[39]*43.089240; /*NXC3H7 */
    XW += x[40]*75.088040; /*NXC3H7O2 */
    XW += x[41]*54.092420; /*C4H6 */
    XW += x[42]*55.100390; /*C4H7 */
    XW += x[43]*56.108360; /*C4H8X1 */
    XW += x[44]*57.116330; /*PXC4H9 */
    XW += x[45]*89.115130; /*PXC4H9O2 */
    XW += x[46]*69.127480; /*C5H9 */
    XW += x[47]*70.135450; /*C5H10X1 */
    XW += x[48]*71.143420; /*C5H11X1 */
    XW += x[49]*84.162540; /*C6H12X1 */
    XW += x[50]*99.197600; /*C7H15X2 */
    XW += x[51]*100.205570; /*NXC7H16 */
    ROW = (*rho) / XW;

    /*Compute conversion, see Eq 11 */
    for (id = 0; id < 52; ++id) {
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
    for (id = 0; id < 52; ++id) {
        sumC += c[id];
    }

    /* See Eq 13  */
    double sumCinv = 1.0/sumC;
    for (id = 0; id < 52; ++id) {
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
    CW += c[9]*13.019120; /*CH */
    CW += c[10]*29.018520; /*HCO */
    CW += c[11]*14.027090; /*CH2 */
    CW += c[12]*44.009950; /*CO2 */
    CW += c[13]*28.010550; /*CO */
    CW += c[14]*30.026490; /*CH2O */
    CW += c[15]*14.027090; /*CH2GSG */
    CW += c[16]*15.035060; /*CH3 */
    CW += c[17]*31.034460; /*CH3O */
    CW += c[18]*16.043030; /*CH4 */
    CW += c[19]*32.042430; /*CH3OH */
    CW += c[20]*30.070120; /*C2H6 */
    CW += c[21]*29.062150; /*C2H5 */
    CW += c[22]*42.037640; /*CH2CO */
    CW += c[23]*46.025890; /*HOCHO */
    CW += c[24]*47.033860; /*CH3O2 */
    CW += c[25]*48.041830; /*CH3O2H */
    CW += c[26]*26.038240; /*C2H2 */
    CW += c[27]*41.029670; /*HCCO */
    CW += c[28]*27.046210; /*C2H3 */
    CW += c[29]*43.045610; /*CH2CHO */
    CW += c[30]*42.081270; /*C3H6 */
    CW += c[31]*28.054180; /*C2H4 */
    CW += c[32]*45.061550; /*C2H5O */
    CW += c[33]*43.045610; /*CH3CO */
    CW += c[34]*61.060950; /*C2H5O2 */
    CW += c[35]*38.049390; /*C3H2 */
    CW += c[36]*39.057360; /*C3H3 */
    CW += c[37]*40.065330; /*C3H4XA */
    CW += c[38]*41.073300; /*C3H5XA */
    CW += c[39]*43.089240; /*NXC3H7 */
    CW += c[40]*75.088040; /*NXC3H7O2 */
    CW += c[41]*54.092420; /*C4H6 */
    CW += c[42]*55.100390; /*C4H7 */
    CW += c[43]*56.108360; /*C4H8X1 */
    CW += c[44]*57.116330; /*PXC4H9 */
    CW += c[45]*89.115130; /*PXC4H9O2 */
    CW += c[46]*69.127480; /*C5H9 */
    CW += c[47]*70.135450; /*C5H10X1 */
    CW += c[48]*71.143420; /*C5H11X1 */
    CW += c[49]*84.162540; /*C6H12X1 */
    CW += c[50]*99.197600; /*C7H15X2 */
    CW += c[51]*100.205570; /*NXC7H16 */
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
    y[9] = c[9]*13.019120*CWinv; 
    y[10] = c[10]*29.018520*CWinv; 
    y[11] = c[11]*14.027090*CWinv; 
    y[12] = c[12]*44.009950*CWinv; 
    y[13] = c[13]*28.010550*CWinv; 
    y[14] = c[14]*30.026490*CWinv; 
    y[15] = c[15]*14.027090*CWinv; 
    y[16] = c[16]*15.035060*CWinv; 
    y[17] = c[17]*31.034460*CWinv; 
    y[18] = c[18]*16.043030*CWinv; 
    y[19] = c[19]*32.042430*CWinv; 
    y[20] = c[20]*30.070120*CWinv; 
    y[21] = c[21]*29.062150*CWinv; 
    y[22] = c[22]*42.037640*CWinv; 
    y[23] = c[23]*46.025890*CWinv; 
    y[24] = c[24]*47.033860*CWinv; 
    y[25] = c[25]*48.041830*CWinv; 
    y[26] = c[26]*26.038240*CWinv; 
    y[27] = c[27]*41.029670*CWinv; 
    y[28] = c[28]*27.046210*CWinv; 
    y[29] = c[29]*43.045610*CWinv; 
    y[30] = c[30]*42.081270*CWinv; 
    y[31] = c[31]*28.054180*CWinv; 
    y[32] = c[32]*45.061550*CWinv; 
    y[33] = c[33]*43.045610*CWinv; 
    y[34] = c[34]*61.060950*CWinv; 
    y[35] = c[35]*38.049390*CWinv; 
    y[36] = c[36]*39.057360*CWinv; 
    y[37] = c[37]*40.065330*CWinv; 
    y[38] = c[38]*41.073300*CWinv; 
    y[39] = c[39]*43.089240*CWinv; 
    y[40] = c[40]*75.088040*CWinv; 
    y[41] = c[41]*54.092420*CWinv; 
    y[42] = c[42]*55.100390*CWinv; 
    y[43] = c[43]*56.108360*CWinv; 
    y[44] = c[44]*57.116330*CWinv; 
    y[45] = c[45]*89.115130*CWinv; 
    y[46] = c[46]*69.127480*CWinv; 
    y[47] = c[47]*70.135450*CWinv; 
    y[48] = c[48]*71.143420*CWinv; 
    y[49] = c[49]*84.162540*CWinv; 
    y[50] = c[50]*99.197600*CWinv; 
    y[51] = c[51]*100.205570*CWinv; 

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
    cvms[9] *= 6.386347631908485e+06; /*CH */
    cvms[10] *= 2.865226282440744e+06; /*HCO */
    cvms[11] *= 5.927432288630956e+06; /*CH2 */
    cvms[12] *= 1.889223372931176e+06; /*CO2 */
    cvms[13] *= 2.968332509769797e+06; /*CO */
    cvms[14] *= 2.769042474879095e+06; /*CH2O */
    cvms[15] *= 5.927432288630956e+06; /*CH2GSG */
    cvms[16] *= 5.530049509714786e+06; /*CH3 */
    cvms[17] *= 2.679106586083096e+06; /*CH3O */
    cvms[18] *= 5.182601178301878e+06; /*CH4 */
    cvms[19] *= 2.594828987112788e+06; /*CH3OH */
    cvms[20] *= 2.765024754857393e+06; /*C2H6 */
    cvms[21] *= 2.860924817383862e+06; /*C2H5 */
    cvms[22] *= 1.977861416138784e+06; /*CH2CO */
    cvms[23] *= 1.806475142176118e+06; /*HOCHO */
    cvms[24] *= 1.767761059405552e+06; /*CH3O2 */
    cvms[25] *= 1.730671504010825e+06; /*CH3O2H */
    cvms[26] *= 3.193173815954242e+06; /*C2H2 */
    cvms[27] *= 2.026451252996488e+06; /*HCCO */
    cvms[28] *= 3.074169215632519e+06; /*C2H3 */
    cvms[29] *= 1.931547170118681e+06; /*CH2CHO */
    cvms[30] *= 1.975810762876985e+06; /*C3H6 */
    cvms[31] *= 2.963716144315478e+06; /*C2H4 */
    cvms[32] *= 1.845134625451907e+06; /*C2H5O */
    cvms[33] *= 1.931547170118681e+06; /*CH3CO */
    cvms[34] *= 1.361666108724682e+06; /*C2H5O2 */
    cvms[35] *= 2.185176324286208e+06; /*C3H2 */
    cvms[36] *= 2.128782543969495e+06; /*C3H3 */
    cvms[37] *= 2.075226291198210e+06; /*C3H4XA */
    cvms[38] *= 2.024298660724422e+06; /*C3H5XA */
    cvms[39] *= 1.929591382478141e+06; /*NXC3H7 */
    cvms[40] *= 1.107295198829699e+06; /*NXC3H7O2 */
    cvms[41] *= 1.537084607816260e+06; /*C4H6 */
    cvms[42] *= 1.508966201174482e+06; /*C4H7 */
    cvms[43] *= 1.481858072157739e+06; /*C4H8X1 */
    cvms[44] *= 1.455706733635239e+06; /*PXC4H9 */
    cvms[45] *= 9.330023552850385e+05; /*PXC4H9O2 */
    cvms[46] *= 1.202772416722444e+06; /*C5H9 */
    cvms[47] *= 1.185486457726191e+06; /*C5H10X1 */
    cvms[48] *= 1.168690318535887e+06; /*C5H11X1 */
    cvms[49] *= 9.879053814384926e+05; /*C6H12X1 */
    cvms[50] *= 8.381717519529947e+05; /*C7H15X2 */
    cvms[51] *= 8.297405641376262e+05; /*NXC7H16 */
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
    cpms[9] *= 6.386347631908485e+06; /*CH */
    cpms[10] *= 2.865226282440744e+06; /*HCO */
    cpms[11] *= 5.927432288630956e+06; /*CH2 */
    cpms[12] *= 1.889223372931176e+06; /*CO2 */
    cpms[13] *= 2.968332509769797e+06; /*CO */
    cpms[14] *= 2.769042474879095e+06; /*CH2O */
    cpms[15] *= 5.927432288630956e+06; /*CH2GSG */
    cpms[16] *= 5.530049509714786e+06; /*CH3 */
    cpms[17] *= 2.679106586083096e+06; /*CH3O */
    cpms[18] *= 5.182601178301878e+06; /*CH4 */
    cpms[19] *= 2.594828987112788e+06; /*CH3OH */
    cpms[20] *= 2.765024754857393e+06; /*C2H6 */
    cpms[21] *= 2.860924817383862e+06; /*C2H5 */
    cpms[22] *= 1.977861416138784e+06; /*CH2CO */
    cpms[23] *= 1.806475142176118e+06; /*HOCHO */
    cpms[24] *= 1.767761059405552e+06; /*CH3O2 */
    cpms[25] *= 1.730671504010825e+06; /*CH3O2H */
    cpms[26] *= 3.193173815954242e+06; /*C2H2 */
    cpms[27] *= 2.026451252996488e+06; /*HCCO */
    cpms[28] *= 3.074169215632519e+06; /*C2H3 */
    cpms[29] *= 1.931547170118681e+06; /*CH2CHO */
    cpms[30] *= 1.975810762876985e+06; /*C3H6 */
    cpms[31] *= 2.963716144315478e+06; /*C2H4 */
    cpms[32] *= 1.845134625451907e+06; /*C2H5O */
    cpms[33] *= 1.931547170118681e+06; /*CH3CO */
    cpms[34] *= 1.361666108724682e+06; /*C2H5O2 */
    cpms[35] *= 2.185176324286208e+06; /*C3H2 */
    cpms[36] *= 2.128782543969495e+06; /*C3H3 */
    cpms[37] *= 2.075226291198210e+06; /*C3H4XA */
    cpms[38] *= 2.024298660724422e+06; /*C3H5XA */
    cpms[39] *= 1.929591382478141e+06; /*NXC3H7 */
    cpms[40] *= 1.107295198829699e+06; /*NXC3H7O2 */
    cpms[41] *= 1.537084607816260e+06; /*C4H6 */
    cpms[42] *= 1.508966201174482e+06; /*C4H7 */
    cpms[43] *= 1.481858072157739e+06; /*C4H8X1 */
    cpms[44] *= 1.455706733635239e+06; /*PXC4H9 */
    cpms[45] *= 9.330023552850385e+05; /*PXC4H9O2 */
    cpms[46] *= 1.202772416722444e+06; /*C5H9 */
    cpms[47] *= 1.185486457726191e+06; /*C5H10X1 */
    cpms[48] *= 1.168690318535887e+06; /*C5H11X1 */
    cpms[49] *= 9.879053814384926e+05; /*C6H12X1 */
    cpms[50] *= 8.381717519529947e+05; /*C7H15X2 */
    cpms[51] *= 8.297405641376262e+05; /*NXC7H16 */
}


/*Returns internal energy in mass units (Eq 30.) */
AMREX_GPU_HOST_DEVICE void CKUMS(double *  T,  double *  ums)
{
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double RT = 8.31446e+07*tT; /*R*T */
    speciesInternalEnergy(ums, tc);
    for (int i = 0; i < 52; i++)
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
    for (int i = 0; i < 52; i++)
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
    for (int i = 0; i < 52; i++)
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
    for (int i = 0; i < 52; i++)
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
    sms[9] *= 6.386347631908485e+06; /*CH */
    sms[10] *= 2.865226282440744e+06; /*HCO */
    sms[11] *= 5.927432288630956e+06; /*CH2 */
    sms[12] *= 1.889223372931176e+06; /*CO2 */
    sms[13] *= 2.968332509769797e+06; /*CO */
    sms[14] *= 2.769042474879095e+06; /*CH2O */
    sms[15] *= 5.927432288630956e+06; /*CH2GSG */
    sms[16] *= 5.530049509714786e+06; /*CH3 */
    sms[17] *= 2.679106586083096e+06; /*CH3O */
    sms[18] *= 5.182601178301878e+06; /*CH4 */
    sms[19] *= 2.594828987112788e+06; /*CH3OH */
    sms[20] *= 2.765024754857393e+06; /*C2H6 */
    sms[21] *= 2.860924817383862e+06; /*C2H5 */
    sms[22] *= 1.977861416138784e+06; /*CH2CO */
    sms[23] *= 1.806475142176118e+06; /*HOCHO */
    sms[24] *= 1.767761059405552e+06; /*CH3O2 */
    sms[25] *= 1.730671504010825e+06; /*CH3O2H */
    sms[26] *= 3.193173815954242e+06; /*C2H2 */
    sms[27] *= 2.026451252996488e+06; /*HCCO */
    sms[28] *= 3.074169215632519e+06; /*C2H3 */
    sms[29] *= 1.931547170118681e+06; /*CH2CHO */
    sms[30] *= 1.975810762876985e+06; /*C3H6 */
    sms[31] *= 2.963716144315478e+06; /*C2H4 */
    sms[32] *= 1.845134625451907e+06; /*C2H5O */
    sms[33] *= 1.931547170118681e+06; /*CH3CO */
    sms[34] *= 1.361666108724682e+06; /*C2H5O2 */
    sms[35] *= 2.185176324286208e+06; /*C3H2 */
    sms[36] *= 2.128782543969495e+06; /*C3H3 */
    sms[37] *= 2.075226291198210e+06; /*C3H4XA */
    sms[38] *= 2.024298660724422e+06; /*C3H5XA */
    sms[39] *= 1.929591382478141e+06; /*NXC3H7 */
    sms[40] *= 1.107295198829699e+06; /*NXC3H7O2 */
    sms[41] *= 1.537084607816260e+06; /*C4H6 */
    sms[42] *= 1.508966201174482e+06; /*C4H7 */
    sms[43] *= 1.481858072157739e+06; /*C4H8X1 */
    sms[44] *= 1.455706733635239e+06; /*PXC4H9 */
    sms[45] *= 9.330023552850385e+05; /*PXC4H9O2 */
    sms[46] *= 1.202772416722444e+06; /*C5H9 */
    sms[47] *= 1.185486457726191e+06; /*C5H10X1 */
    sms[48] *= 1.168690318535887e+06; /*C5H11X1 */
    sms[49] *= 9.879053814384926e+05; /*C6H12X1 */
    sms[50] *= 8.381717519529947e+05; /*C7H15X2 */
    sms[51] *= 8.297405641376262e+05; /*NXC7H16 */
}


/*Returns the mean specific heat at CP (Eq. 33) */
void CKCPBL(double *  T, double *  x,  double *  cpbl)
{
    int id; /*loop counter */
    double result = 0; 
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double cpor[52]; /* temporary storage */
    cp_R(cpor, tc);

    /*perform dot product */
    for (id = 0; id < 52; ++id) {
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
    double cpor[52], tresult[52]; /* temporary storage */
    cp_R(cpor, tc);
    for (int i = 0; i < 52; i++)
    {
        tresult[i] = cpor[i]*y[i]*imw[i];

    }
    for (int i = 0; i < 52; i++)
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
    double cvor[52]; /* temporary storage */
    cv_R(cvor, tc);

    /*perform dot product */
    for (id = 0; id < 52; ++id) {
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
    double cvor[52]; /* temporary storage */
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
    result += cvor[9]*y[9]*imw[9]; /*CH */
    result += cvor[10]*y[10]*imw[10]; /*HCO */
    result += cvor[11]*y[11]*imw[11]; /*CH2 */
    result += cvor[12]*y[12]*imw[12]; /*CO2 */
    result += cvor[13]*y[13]*imw[13]; /*CO */
    result += cvor[14]*y[14]*imw[14]; /*CH2O */
    result += cvor[15]*y[15]*imw[15]; /*CH2GSG */
    result += cvor[16]*y[16]*imw[16]; /*CH3 */
    result += cvor[17]*y[17]*imw[17]; /*CH3O */
    result += cvor[18]*y[18]*imw[18]; /*CH4 */
    result += cvor[19]*y[19]*imw[19]; /*CH3OH */
    result += cvor[20]*y[20]*imw[20]; /*C2H6 */
    result += cvor[21]*y[21]*imw[21]; /*C2H5 */
    result += cvor[22]*y[22]*imw[22]; /*CH2CO */
    result += cvor[23]*y[23]*imw[23]; /*HOCHO */
    result += cvor[24]*y[24]*imw[24]; /*CH3O2 */
    result += cvor[25]*y[25]*imw[25]; /*CH3O2H */
    result += cvor[26]*y[26]*imw[26]; /*C2H2 */
    result += cvor[27]*y[27]*imw[27]; /*HCCO */
    result += cvor[28]*y[28]*imw[28]; /*C2H3 */
    result += cvor[29]*y[29]*imw[29]; /*CH2CHO */
    result += cvor[30]*y[30]*imw[30]; /*C3H6 */
    result += cvor[31]*y[31]*imw[31]; /*C2H4 */
    result += cvor[32]*y[32]*imw[32]; /*C2H5O */
    result += cvor[33]*y[33]*imw[33]; /*CH3CO */
    result += cvor[34]*y[34]*imw[34]; /*C2H5O2 */
    result += cvor[35]*y[35]*imw[35]; /*C3H2 */
    result += cvor[36]*y[36]*imw[36]; /*C3H3 */
    result += cvor[37]*y[37]*imw[37]; /*C3H4XA */
    result += cvor[38]*y[38]*imw[38]; /*C3H5XA */
    result += cvor[39]*y[39]*imw[39]; /*NXC3H7 */
    result += cvor[40]*y[40]*imw[40]; /*NXC3H7O2 */
    result += cvor[41]*y[41]*imw[41]; /*C4H6 */
    result += cvor[42]*y[42]*imw[42]; /*C4H7 */
    result += cvor[43]*y[43]*imw[43]; /*C4H8X1 */
    result += cvor[44]*y[44]*imw[44]; /*PXC4H9 */
    result += cvor[45]*y[45]*imw[45]; /*PXC4H9O2 */
    result += cvor[46]*y[46]*imw[46]; /*C5H9 */
    result += cvor[47]*y[47]*imw[47]; /*C5H10X1 */
    result += cvor[48]*y[48]*imw[48]; /*C5H11X1 */
    result += cvor[49]*y[49]*imw[49]; /*C6H12X1 */
    result += cvor[50]*y[50]*imw[50]; /*C7H15X2 */
    result += cvor[51]*y[51]*imw[51]; /*NXC7H16 */

    *cvbs = result * 8.31446e+07;
}


/*Returns the mean enthalpy of the mixture in molar units */
void CKHBML(double *  T, double *  x,  double *  hbml)
{
    int id; /*loop counter */
    double result = 0; 
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double hml[52]; /* temporary storage */
    double RT = 8.31446e+07*tT; /*R*T */
    speciesEnthalpy(hml, tc);

    /*perform dot product */
    for (id = 0; id < 52; ++id) {
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
    double hml[52], tmp[52]; /* temporary storage */
    double RT = 8.31446e+07*tT; /*R*T */
    speciesEnthalpy(hml, tc);
    int id;
    for (id = 0; id < 52; ++id) {
        tmp[id] = y[id]*hml[id]*imw[id];
    }
    for (id = 0; id < 52; ++id) {
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
    double uml[52]; /* temporary energy array */
    double RT = 8.31446e+07*tT; /*R*T */
    speciesInternalEnergy(uml, tc);

    /*perform dot product */
    for (id = 0; id < 52; ++id) {
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
    double ums[52]; /* temporary energy array */
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
    result += y[9]*ums[9]*imw[9]; /*CH */
    result += y[10]*ums[10]*imw[10]; /*HCO */
    result += y[11]*ums[11]*imw[11]; /*CH2 */
    result += y[12]*ums[12]*imw[12]; /*CO2 */
    result += y[13]*ums[13]*imw[13]; /*CO */
    result += y[14]*ums[14]*imw[14]; /*CH2O */
    result += y[15]*ums[15]*imw[15]; /*CH2GSG */
    result += y[16]*ums[16]*imw[16]; /*CH3 */
    result += y[17]*ums[17]*imw[17]; /*CH3O */
    result += y[18]*ums[18]*imw[18]; /*CH4 */
    result += y[19]*ums[19]*imw[19]; /*CH3OH */
    result += y[20]*ums[20]*imw[20]; /*C2H6 */
    result += y[21]*ums[21]*imw[21]; /*C2H5 */
    result += y[22]*ums[22]*imw[22]; /*CH2CO */
    result += y[23]*ums[23]*imw[23]; /*HOCHO */
    result += y[24]*ums[24]*imw[24]; /*CH3O2 */
    result += y[25]*ums[25]*imw[25]; /*CH3O2H */
    result += y[26]*ums[26]*imw[26]; /*C2H2 */
    result += y[27]*ums[27]*imw[27]; /*HCCO */
    result += y[28]*ums[28]*imw[28]; /*C2H3 */
    result += y[29]*ums[29]*imw[29]; /*CH2CHO */
    result += y[30]*ums[30]*imw[30]; /*C3H6 */
    result += y[31]*ums[31]*imw[31]; /*C2H4 */
    result += y[32]*ums[32]*imw[32]; /*C2H5O */
    result += y[33]*ums[33]*imw[33]; /*CH3CO */
    result += y[34]*ums[34]*imw[34]; /*C2H5O2 */
    result += y[35]*ums[35]*imw[35]; /*C3H2 */
    result += y[36]*ums[36]*imw[36]; /*C3H3 */
    result += y[37]*ums[37]*imw[37]; /*C3H4XA */
    result += y[38]*ums[38]*imw[38]; /*C3H5XA */
    result += y[39]*ums[39]*imw[39]; /*NXC3H7 */
    result += y[40]*ums[40]*imw[40]; /*NXC3H7O2 */
    result += y[41]*ums[41]*imw[41]; /*C4H6 */
    result += y[42]*ums[42]*imw[42]; /*C4H7 */
    result += y[43]*ums[43]*imw[43]; /*C4H8X1 */
    result += y[44]*ums[44]*imw[44]; /*PXC4H9 */
    result += y[45]*ums[45]*imw[45]; /*PXC4H9O2 */
    result += y[46]*ums[46]*imw[46]; /*C5H9 */
    result += y[47]*ums[47]*imw[47]; /*C5H10X1 */
    result += y[48]*ums[48]*imw[48]; /*C5H11X1 */
    result += y[49]*ums[49]*imw[49]; /*C6H12X1 */
    result += y[50]*ums[50]*imw[50]; /*C7H15X2 */
    result += y[51]*ums[51]*imw[51]; /*NXC7H16 */

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
    double sor[52]; /* temporary storage */
    speciesEntropy(sor, tc);

    /*Compute Eq 42 */
    for (id = 0; id < 52; ++id) {
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
    double sor[52]; /* temporary storage */
    double x[52]; /* need a ytx conversion */
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
    YOW += y[9]*imw[9]; /*CH */
    YOW += y[10]*imw[10]; /*HCO */
    YOW += y[11]*imw[11]; /*CH2 */
    YOW += y[12]*imw[12]; /*CO2 */
    YOW += y[13]*imw[13]; /*CO */
    YOW += y[14]*imw[14]; /*CH2O */
    YOW += y[15]*imw[15]; /*CH2GSG */
    YOW += y[16]*imw[16]; /*CH3 */
    YOW += y[17]*imw[17]; /*CH3O */
    YOW += y[18]*imw[18]; /*CH4 */
    YOW += y[19]*imw[19]; /*CH3OH */
    YOW += y[20]*imw[20]; /*C2H6 */
    YOW += y[21]*imw[21]; /*C2H5 */
    YOW += y[22]*imw[22]; /*CH2CO */
    YOW += y[23]*imw[23]; /*HOCHO */
    YOW += y[24]*imw[24]; /*CH3O2 */
    YOW += y[25]*imw[25]; /*CH3O2H */
    YOW += y[26]*imw[26]; /*C2H2 */
    YOW += y[27]*imw[27]; /*HCCO */
    YOW += y[28]*imw[28]; /*C2H3 */
    YOW += y[29]*imw[29]; /*CH2CHO */
    YOW += y[30]*imw[30]; /*C3H6 */
    YOW += y[31]*imw[31]; /*C2H4 */
    YOW += y[32]*imw[32]; /*C2H5O */
    YOW += y[33]*imw[33]; /*CH3CO */
    YOW += y[34]*imw[34]; /*C2H5O2 */
    YOW += y[35]*imw[35]; /*C3H2 */
    YOW += y[36]*imw[36]; /*C3H3 */
    YOW += y[37]*imw[37]; /*C3H4XA */
    YOW += y[38]*imw[38]; /*C3H5XA */
    YOW += y[39]*imw[39]; /*NXC3H7 */
    YOW += y[40]*imw[40]; /*NXC3H7O2 */
    YOW += y[41]*imw[41]; /*C4H6 */
    YOW += y[42]*imw[42]; /*C4H7 */
    YOW += y[43]*imw[43]; /*C4H8X1 */
    YOW += y[44]*imw[44]; /*PXC4H9 */
    YOW += y[45]*imw[45]; /*PXC4H9O2 */
    YOW += y[46]*imw[46]; /*C5H9 */
    YOW += y[47]*imw[47]; /*C5H10X1 */
    YOW += y[48]*imw[48]; /*C5H11X1 */
    YOW += y[49]*imw[49]; /*C6H12X1 */
    YOW += y[50]*imw[50]; /*C7H15X2 */
    YOW += y[51]*imw[51]; /*NXC7H16 */
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
    x[9] = y[9]/(13.019120*YOW); 
    x[10] = y[10]/(29.018520*YOW); 
    x[11] = y[11]/(14.027090*YOW); 
    x[12] = y[12]/(44.009950*YOW); 
    x[13] = y[13]/(28.010550*YOW); 
    x[14] = y[14]/(30.026490*YOW); 
    x[15] = y[15]/(14.027090*YOW); 
    x[16] = y[16]/(15.035060*YOW); 
    x[17] = y[17]/(31.034460*YOW); 
    x[18] = y[18]/(16.043030*YOW); 
    x[19] = y[19]/(32.042430*YOW); 
    x[20] = y[20]/(30.070120*YOW); 
    x[21] = y[21]/(29.062150*YOW); 
    x[22] = y[22]/(42.037640*YOW); 
    x[23] = y[23]/(46.025890*YOW); 
    x[24] = y[24]/(47.033860*YOW); 
    x[25] = y[25]/(48.041830*YOW); 
    x[26] = y[26]/(26.038240*YOW); 
    x[27] = y[27]/(41.029670*YOW); 
    x[28] = y[28]/(27.046210*YOW); 
    x[29] = y[29]/(43.045610*YOW); 
    x[30] = y[30]/(42.081270*YOW); 
    x[31] = y[31]/(28.054180*YOW); 
    x[32] = y[32]/(45.061550*YOW); 
    x[33] = y[33]/(43.045610*YOW); 
    x[34] = y[34]/(61.060950*YOW); 
    x[35] = y[35]/(38.049390*YOW); 
    x[36] = y[36]/(39.057360*YOW); 
    x[37] = y[37]/(40.065330*YOW); 
    x[38] = y[38]/(41.073300*YOW); 
    x[39] = y[39]/(43.089240*YOW); 
    x[40] = y[40]/(75.088040*YOW); 
    x[41] = y[41]/(54.092420*YOW); 
    x[42] = y[42]/(55.100390*YOW); 
    x[43] = y[43]/(56.108360*YOW); 
    x[44] = y[44]/(57.116330*YOW); 
    x[45] = y[45]/(89.115130*YOW); 
    x[46] = y[46]/(69.127480*YOW); 
    x[47] = y[47]/(70.135450*YOW); 
    x[48] = y[48]/(71.143420*YOW); 
    x[49] = y[49]/(84.162540*YOW); 
    x[50] = y[50]/(99.197600*YOW); 
    x[51] = y[51]/(100.205570*YOW); 
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
    result += x[25]*(sor[25]-log((x[25]+1e-100))-logPratio);
    result += x[26]*(sor[26]-log((x[26]+1e-100))-logPratio);
    result += x[27]*(sor[27]-log((x[27]+1e-100))-logPratio);
    result += x[28]*(sor[28]-log((x[28]+1e-100))-logPratio);
    result += x[29]*(sor[29]-log((x[29]+1e-100))-logPratio);
    result += x[30]*(sor[30]-log((x[30]+1e-100))-logPratio);
    result += x[31]*(sor[31]-log((x[31]+1e-100))-logPratio);
    result += x[32]*(sor[32]-log((x[32]+1e-100))-logPratio);
    result += x[33]*(sor[33]-log((x[33]+1e-100))-logPratio);
    result += x[34]*(sor[34]-log((x[34]+1e-100))-logPratio);
    result += x[35]*(sor[35]-log((x[35]+1e-100))-logPratio);
    result += x[36]*(sor[36]-log((x[36]+1e-100))-logPratio);
    result += x[37]*(sor[37]-log((x[37]+1e-100))-logPratio);
    result += x[38]*(sor[38]-log((x[38]+1e-100))-logPratio);
    result += x[39]*(sor[39]-log((x[39]+1e-100))-logPratio);
    result += x[40]*(sor[40]-log((x[40]+1e-100))-logPratio);
    result += x[41]*(sor[41]-log((x[41]+1e-100))-logPratio);
    result += x[42]*(sor[42]-log((x[42]+1e-100))-logPratio);
    result += x[43]*(sor[43]-log((x[43]+1e-100))-logPratio);
    result += x[44]*(sor[44]-log((x[44]+1e-100))-logPratio);
    result += x[45]*(sor[45]-log((x[45]+1e-100))-logPratio);
    result += x[46]*(sor[46]-log((x[46]+1e-100))-logPratio);
    result += x[47]*(sor[47]-log((x[47]+1e-100))-logPratio);
    result += x[48]*(sor[48]-log((x[48]+1e-100))-logPratio);
    result += x[49]*(sor[49]-log((x[49]+1e-100))-logPratio);
    result += x[50]*(sor[50]-log((x[50]+1e-100))-logPratio);
    result += x[51]*(sor[51]-log((x[51]+1e-100))-logPratio);
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
    double gort[52]; /* temporary storage */
    /*Compute g/RT */
    gibbs(gort, tc);

    /*Compute Eq 44 */
    for (id = 0; id < 52; ++id) {
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
    double gort[52]; /* temporary storage */
    double x[52]; /* need a ytx conversion */
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
    YOW += y[9]*imw[9]; /*CH */
    YOW += y[10]*imw[10]; /*HCO */
    YOW += y[11]*imw[11]; /*CH2 */
    YOW += y[12]*imw[12]; /*CO2 */
    YOW += y[13]*imw[13]; /*CO */
    YOW += y[14]*imw[14]; /*CH2O */
    YOW += y[15]*imw[15]; /*CH2GSG */
    YOW += y[16]*imw[16]; /*CH3 */
    YOW += y[17]*imw[17]; /*CH3O */
    YOW += y[18]*imw[18]; /*CH4 */
    YOW += y[19]*imw[19]; /*CH3OH */
    YOW += y[20]*imw[20]; /*C2H6 */
    YOW += y[21]*imw[21]; /*C2H5 */
    YOW += y[22]*imw[22]; /*CH2CO */
    YOW += y[23]*imw[23]; /*HOCHO */
    YOW += y[24]*imw[24]; /*CH3O2 */
    YOW += y[25]*imw[25]; /*CH3O2H */
    YOW += y[26]*imw[26]; /*C2H2 */
    YOW += y[27]*imw[27]; /*HCCO */
    YOW += y[28]*imw[28]; /*C2H3 */
    YOW += y[29]*imw[29]; /*CH2CHO */
    YOW += y[30]*imw[30]; /*C3H6 */
    YOW += y[31]*imw[31]; /*C2H4 */
    YOW += y[32]*imw[32]; /*C2H5O */
    YOW += y[33]*imw[33]; /*CH3CO */
    YOW += y[34]*imw[34]; /*C2H5O2 */
    YOW += y[35]*imw[35]; /*C3H2 */
    YOW += y[36]*imw[36]; /*C3H3 */
    YOW += y[37]*imw[37]; /*C3H4XA */
    YOW += y[38]*imw[38]; /*C3H5XA */
    YOW += y[39]*imw[39]; /*NXC3H7 */
    YOW += y[40]*imw[40]; /*NXC3H7O2 */
    YOW += y[41]*imw[41]; /*C4H6 */
    YOW += y[42]*imw[42]; /*C4H7 */
    YOW += y[43]*imw[43]; /*C4H8X1 */
    YOW += y[44]*imw[44]; /*PXC4H9 */
    YOW += y[45]*imw[45]; /*PXC4H9O2 */
    YOW += y[46]*imw[46]; /*C5H9 */
    YOW += y[47]*imw[47]; /*C5H10X1 */
    YOW += y[48]*imw[48]; /*C5H11X1 */
    YOW += y[49]*imw[49]; /*C6H12X1 */
    YOW += y[50]*imw[50]; /*C7H15X2 */
    YOW += y[51]*imw[51]; /*NXC7H16 */
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
    x[9] = y[9]/(13.019120*YOW); 
    x[10] = y[10]/(29.018520*YOW); 
    x[11] = y[11]/(14.027090*YOW); 
    x[12] = y[12]/(44.009950*YOW); 
    x[13] = y[13]/(28.010550*YOW); 
    x[14] = y[14]/(30.026490*YOW); 
    x[15] = y[15]/(14.027090*YOW); 
    x[16] = y[16]/(15.035060*YOW); 
    x[17] = y[17]/(31.034460*YOW); 
    x[18] = y[18]/(16.043030*YOW); 
    x[19] = y[19]/(32.042430*YOW); 
    x[20] = y[20]/(30.070120*YOW); 
    x[21] = y[21]/(29.062150*YOW); 
    x[22] = y[22]/(42.037640*YOW); 
    x[23] = y[23]/(46.025890*YOW); 
    x[24] = y[24]/(47.033860*YOW); 
    x[25] = y[25]/(48.041830*YOW); 
    x[26] = y[26]/(26.038240*YOW); 
    x[27] = y[27]/(41.029670*YOW); 
    x[28] = y[28]/(27.046210*YOW); 
    x[29] = y[29]/(43.045610*YOW); 
    x[30] = y[30]/(42.081270*YOW); 
    x[31] = y[31]/(28.054180*YOW); 
    x[32] = y[32]/(45.061550*YOW); 
    x[33] = y[33]/(43.045610*YOW); 
    x[34] = y[34]/(61.060950*YOW); 
    x[35] = y[35]/(38.049390*YOW); 
    x[36] = y[36]/(39.057360*YOW); 
    x[37] = y[37]/(40.065330*YOW); 
    x[38] = y[38]/(41.073300*YOW); 
    x[39] = y[39]/(43.089240*YOW); 
    x[40] = y[40]/(75.088040*YOW); 
    x[41] = y[41]/(54.092420*YOW); 
    x[42] = y[42]/(55.100390*YOW); 
    x[43] = y[43]/(56.108360*YOW); 
    x[44] = y[44]/(57.116330*YOW); 
    x[45] = y[45]/(89.115130*YOW); 
    x[46] = y[46]/(69.127480*YOW); 
    x[47] = y[47]/(70.135450*YOW); 
    x[48] = y[48]/(71.143420*YOW); 
    x[49] = y[49]/(84.162540*YOW); 
    x[50] = y[50]/(99.197600*YOW); 
    x[51] = y[51]/(100.205570*YOW); 
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
    result += x[25]*(gort[25]+log((x[25]+1e-100))+logPratio);
    result += x[26]*(gort[26]+log((x[26]+1e-100))+logPratio);
    result += x[27]*(gort[27]+log((x[27]+1e-100))+logPratio);
    result += x[28]*(gort[28]+log((x[28]+1e-100))+logPratio);
    result += x[29]*(gort[29]+log((x[29]+1e-100))+logPratio);
    result += x[30]*(gort[30]+log((x[30]+1e-100))+logPratio);
    result += x[31]*(gort[31]+log((x[31]+1e-100))+logPratio);
    result += x[32]*(gort[32]+log((x[32]+1e-100))+logPratio);
    result += x[33]*(gort[33]+log((x[33]+1e-100))+logPratio);
    result += x[34]*(gort[34]+log((x[34]+1e-100))+logPratio);
    result += x[35]*(gort[35]+log((x[35]+1e-100))+logPratio);
    result += x[36]*(gort[36]+log((x[36]+1e-100))+logPratio);
    result += x[37]*(gort[37]+log((x[37]+1e-100))+logPratio);
    result += x[38]*(gort[38]+log((x[38]+1e-100))+logPratio);
    result += x[39]*(gort[39]+log((x[39]+1e-100))+logPratio);
    result += x[40]*(gort[40]+log((x[40]+1e-100))+logPratio);
    result += x[41]*(gort[41]+log((x[41]+1e-100))+logPratio);
    result += x[42]*(gort[42]+log((x[42]+1e-100))+logPratio);
    result += x[43]*(gort[43]+log((x[43]+1e-100))+logPratio);
    result += x[44]*(gort[44]+log((x[44]+1e-100))+logPratio);
    result += x[45]*(gort[45]+log((x[45]+1e-100))+logPratio);
    result += x[46]*(gort[46]+log((x[46]+1e-100))+logPratio);
    result += x[47]*(gort[47]+log((x[47]+1e-100))+logPratio);
    result += x[48]*(gort[48]+log((x[48]+1e-100))+logPratio);
    result += x[49]*(gort[49]+log((x[49]+1e-100))+logPratio);
    result += x[50]*(gort[50]+log((x[50]+1e-100))+logPratio);
    result += x[51]*(gort[51]+log((x[51]+1e-100))+logPratio);
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
    double aort[52]; /* temporary storage */
    /*Compute g/RT */
    helmholtz(aort, tc);

    /*Compute Eq 44 */
    for (id = 0; id < 52; ++id) {
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
    double aort[52]; /* temporary storage */
    double x[52]; /* need a ytx conversion */
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
    YOW += y[9]*imw[9]; /*CH */
    YOW += y[10]*imw[10]; /*HCO */
    YOW += y[11]*imw[11]; /*CH2 */
    YOW += y[12]*imw[12]; /*CO2 */
    YOW += y[13]*imw[13]; /*CO */
    YOW += y[14]*imw[14]; /*CH2O */
    YOW += y[15]*imw[15]; /*CH2GSG */
    YOW += y[16]*imw[16]; /*CH3 */
    YOW += y[17]*imw[17]; /*CH3O */
    YOW += y[18]*imw[18]; /*CH4 */
    YOW += y[19]*imw[19]; /*CH3OH */
    YOW += y[20]*imw[20]; /*C2H6 */
    YOW += y[21]*imw[21]; /*C2H5 */
    YOW += y[22]*imw[22]; /*CH2CO */
    YOW += y[23]*imw[23]; /*HOCHO */
    YOW += y[24]*imw[24]; /*CH3O2 */
    YOW += y[25]*imw[25]; /*CH3O2H */
    YOW += y[26]*imw[26]; /*C2H2 */
    YOW += y[27]*imw[27]; /*HCCO */
    YOW += y[28]*imw[28]; /*C2H3 */
    YOW += y[29]*imw[29]; /*CH2CHO */
    YOW += y[30]*imw[30]; /*C3H6 */
    YOW += y[31]*imw[31]; /*C2H4 */
    YOW += y[32]*imw[32]; /*C2H5O */
    YOW += y[33]*imw[33]; /*CH3CO */
    YOW += y[34]*imw[34]; /*C2H5O2 */
    YOW += y[35]*imw[35]; /*C3H2 */
    YOW += y[36]*imw[36]; /*C3H3 */
    YOW += y[37]*imw[37]; /*C3H4XA */
    YOW += y[38]*imw[38]; /*C3H5XA */
    YOW += y[39]*imw[39]; /*NXC3H7 */
    YOW += y[40]*imw[40]; /*NXC3H7O2 */
    YOW += y[41]*imw[41]; /*C4H6 */
    YOW += y[42]*imw[42]; /*C4H7 */
    YOW += y[43]*imw[43]; /*C4H8X1 */
    YOW += y[44]*imw[44]; /*PXC4H9 */
    YOW += y[45]*imw[45]; /*PXC4H9O2 */
    YOW += y[46]*imw[46]; /*C5H9 */
    YOW += y[47]*imw[47]; /*C5H10X1 */
    YOW += y[48]*imw[48]; /*C5H11X1 */
    YOW += y[49]*imw[49]; /*C6H12X1 */
    YOW += y[50]*imw[50]; /*C7H15X2 */
    YOW += y[51]*imw[51]; /*NXC7H16 */
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
    x[9] = y[9]/(13.019120*YOW); 
    x[10] = y[10]/(29.018520*YOW); 
    x[11] = y[11]/(14.027090*YOW); 
    x[12] = y[12]/(44.009950*YOW); 
    x[13] = y[13]/(28.010550*YOW); 
    x[14] = y[14]/(30.026490*YOW); 
    x[15] = y[15]/(14.027090*YOW); 
    x[16] = y[16]/(15.035060*YOW); 
    x[17] = y[17]/(31.034460*YOW); 
    x[18] = y[18]/(16.043030*YOW); 
    x[19] = y[19]/(32.042430*YOW); 
    x[20] = y[20]/(30.070120*YOW); 
    x[21] = y[21]/(29.062150*YOW); 
    x[22] = y[22]/(42.037640*YOW); 
    x[23] = y[23]/(46.025890*YOW); 
    x[24] = y[24]/(47.033860*YOW); 
    x[25] = y[25]/(48.041830*YOW); 
    x[26] = y[26]/(26.038240*YOW); 
    x[27] = y[27]/(41.029670*YOW); 
    x[28] = y[28]/(27.046210*YOW); 
    x[29] = y[29]/(43.045610*YOW); 
    x[30] = y[30]/(42.081270*YOW); 
    x[31] = y[31]/(28.054180*YOW); 
    x[32] = y[32]/(45.061550*YOW); 
    x[33] = y[33]/(43.045610*YOW); 
    x[34] = y[34]/(61.060950*YOW); 
    x[35] = y[35]/(38.049390*YOW); 
    x[36] = y[36]/(39.057360*YOW); 
    x[37] = y[37]/(40.065330*YOW); 
    x[38] = y[38]/(41.073300*YOW); 
    x[39] = y[39]/(43.089240*YOW); 
    x[40] = y[40]/(75.088040*YOW); 
    x[41] = y[41]/(54.092420*YOW); 
    x[42] = y[42]/(55.100390*YOW); 
    x[43] = y[43]/(56.108360*YOW); 
    x[44] = y[44]/(57.116330*YOW); 
    x[45] = y[45]/(89.115130*YOW); 
    x[46] = y[46]/(69.127480*YOW); 
    x[47] = y[47]/(70.135450*YOW); 
    x[48] = y[48]/(71.143420*YOW); 
    x[49] = y[49]/(84.162540*YOW); 
    x[50] = y[50]/(99.197600*YOW); 
    x[51] = y[51]/(100.205570*YOW); 
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
    result += x[25]*(aort[25]+log((x[25]+1e-100))+logPratio);
    result += x[26]*(aort[26]+log((x[26]+1e-100))+logPratio);
    result += x[27]*(aort[27]+log((x[27]+1e-100))+logPratio);
    result += x[28]*(aort[28]+log((x[28]+1e-100))+logPratio);
    result += x[29]*(aort[29]+log((x[29]+1e-100))+logPratio);
    result += x[30]*(aort[30]+log((x[30]+1e-100))+logPratio);
    result += x[31]*(aort[31]+log((x[31]+1e-100))+logPratio);
    result += x[32]*(aort[32]+log((x[32]+1e-100))+logPratio);
    result += x[33]*(aort[33]+log((x[33]+1e-100))+logPratio);
    result += x[34]*(aort[34]+log((x[34]+1e-100))+logPratio);
    result += x[35]*(aort[35]+log((x[35]+1e-100))+logPratio);
    result += x[36]*(aort[36]+log((x[36]+1e-100))+logPratio);
    result += x[37]*(aort[37]+log((x[37]+1e-100))+logPratio);
    result += x[38]*(aort[38]+log((x[38]+1e-100))+logPratio);
    result += x[39]*(aort[39]+log((x[39]+1e-100))+logPratio);
    result += x[40]*(aort[40]+log((x[40]+1e-100))+logPratio);
    result += x[41]*(aort[41]+log((x[41]+1e-100))+logPratio);
    result += x[42]*(aort[42]+log((x[42]+1e-100))+logPratio);
    result += x[43]*(aort[43]+log((x[43]+1e-100))+logPratio);
    result += x[44]*(aort[44]+log((x[44]+1e-100))+logPratio);
    result += x[45]*(aort[45]+log((x[45]+1e-100))+logPratio);
    result += x[46]*(aort[46]+log((x[46]+1e-100))+logPratio);
    result += x[47]*(aort[47]+log((x[47]+1e-100))+logPratio);
    result += x[48]*(aort[48]+log((x[48]+1e-100))+logPratio);
    result += x[49]*(aort[49]+log((x[49]+1e-100))+logPratio);
    result += x[50]*(aort[50]+log((x[50]+1e-100))+logPratio);
    result += x[51]*(aort[51]+log((x[51]+1e-100))+logPratio);
    /*Scale by RT/W */
    *abms = result * RT * YOW;
}


/*compute the production rate for each species */
AMREX_GPU_HOST_DEVICE void CKWC(double *  T, double *  C,  double *  wdot)
{
    int id; /*loop counter */

    /*convert to SI */
    for (id = 0; id < 52; ++id) {
        C[id] *= 1.0e6;
    }

    /*convert to chemkin units */
    productionRate(wdot, C, *T);

    /*convert to chemkin units */
    for (id = 0; id < 52; ++id) {
        C[id] *= 1.0e-6;
        wdot[id] *= 1.0e-6;
    }
}


/*Returns the molar production rate of species */
/*Given P, T, and mass fractions */
void CKWYP(double *  P, double *  T, double *  y,  double *  wdot)
{
    int id; /*loop counter */
    double c[52]; /*temporary storage */
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
    YOW += y[9]*imw[9]; /*CH */
    YOW += y[10]*imw[10]; /*HCO */
    YOW += y[11]*imw[11]; /*CH2 */
    YOW += y[12]*imw[12]; /*CO2 */
    YOW += y[13]*imw[13]; /*CO */
    YOW += y[14]*imw[14]; /*CH2O */
    YOW += y[15]*imw[15]; /*CH2GSG */
    YOW += y[16]*imw[16]; /*CH3 */
    YOW += y[17]*imw[17]; /*CH3O */
    YOW += y[18]*imw[18]; /*CH4 */
    YOW += y[19]*imw[19]; /*CH3OH */
    YOW += y[20]*imw[20]; /*C2H6 */
    YOW += y[21]*imw[21]; /*C2H5 */
    YOW += y[22]*imw[22]; /*CH2CO */
    YOW += y[23]*imw[23]; /*HOCHO */
    YOW += y[24]*imw[24]; /*CH3O2 */
    YOW += y[25]*imw[25]; /*CH3O2H */
    YOW += y[26]*imw[26]; /*C2H2 */
    YOW += y[27]*imw[27]; /*HCCO */
    YOW += y[28]*imw[28]; /*C2H3 */
    YOW += y[29]*imw[29]; /*CH2CHO */
    YOW += y[30]*imw[30]; /*C3H6 */
    YOW += y[31]*imw[31]; /*C2H4 */
    YOW += y[32]*imw[32]; /*C2H5O */
    YOW += y[33]*imw[33]; /*CH3CO */
    YOW += y[34]*imw[34]; /*C2H5O2 */
    YOW += y[35]*imw[35]; /*C3H2 */
    YOW += y[36]*imw[36]; /*C3H3 */
    YOW += y[37]*imw[37]; /*C3H4XA */
    YOW += y[38]*imw[38]; /*C3H5XA */
    YOW += y[39]*imw[39]; /*NXC3H7 */
    YOW += y[40]*imw[40]; /*NXC3H7O2 */
    YOW += y[41]*imw[41]; /*C4H6 */
    YOW += y[42]*imw[42]; /*C4H7 */
    YOW += y[43]*imw[43]; /*C4H8X1 */
    YOW += y[44]*imw[44]; /*PXC4H9 */
    YOW += y[45]*imw[45]; /*PXC4H9O2 */
    YOW += y[46]*imw[46]; /*C5H9 */
    YOW += y[47]*imw[47]; /*C5H10X1 */
    YOW += y[48]*imw[48]; /*C5H11X1 */
    YOW += y[49]*imw[49]; /*C6H12X1 */
    YOW += y[50]*imw[50]; /*C7H15X2 */
    YOW += y[51]*imw[51]; /*NXC7H16 */
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
    c[25] = PWORT * y[25]*imw[25]; 
    c[26] = PWORT * y[26]*imw[26]; 
    c[27] = PWORT * y[27]*imw[27]; 
    c[28] = PWORT * y[28]*imw[28]; 
    c[29] = PWORT * y[29]*imw[29]; 
    c[30] = PWORT * y[30]*imw[30]; 
    c[31] = PWORT * y[31]*imw[31]; 
    c[32] = PWORT * y[32]*imw[32]; 
    c[33] = PWORT * y[33]*imw[33]; 
    c[34] = PWORT * y[34]*imw[34]; 
    c[35] = PWORT * y[35]*imw[35]; 
    c[36] = PWORT * y[36]*imw[36]; 
    c[37] = PWORT * y[37]*imw[37]; 
    c[38] = PWORT * y[38]*imw[38]; 
    c[39] = PWORT * y[39]*imw[39]; 
    c[40] = PWORT * y[40]*imw[40]; 
    c[41] = PWORT * y[41]*imw[41]; 
    c[42] = PWORT * y[42]*imw[42]; 
    c[43] = PWORT * y[43]*imw[43]; 
    c[44] = PWORT * y[44]*imw[44]; 
    c[45] = PWORT * y[45]*imw[45]; 
    c[46] = PWORT * y[46]*imw[46]; 
    c[47] = PWORT * y[47]*imw[47]; 
    c[48] = PWORT * y[48]*imw[48]; 
    c[49] = PWORT * y[49]*imw[49]; 
    c[50] = PWORT * y[50]*imw[50]; 
    c[51] = PWORT * y[51]*imw[51]; 

    /*convert to chemkin units */
    productionRate(wdot, c, *T);

    /*convert to chemkin units */
    for (id = 0; id < 52; ++id) {
        wdot[id] *= 1.0e-6;
    }
}


/*Returns the molar production rate of species */
/*Given P, T, and mole fractions */
void CKWXP(double *  P, double *  T, double *  x,  double *  wdot)
{
    int id; /*loop counter */
    double c[52]; /*temporary storage */
    double PORT = 1e6 * (*P)/(8.31446e+07 * (*T)); /*1e6 * P/RT so c goes to SI units */

    /*Compute conversion, see Eq 10 */
    for (id = 0; id < 52; ++id) {
        c[id] = x[id]*PORT;
    }

    /*convert to chemkin units */
    productionRate(wdot, c, *T);

    /*convert to chemkin units */
    for (id = 0; id < 52; ++id) {
        wdot[id] *= 1.0e-6;
    }
}


/*Returns the molar production rate of species */
/*Given rho, T, and mass fractions */
AMREX_GPU_HOST_DEVICE void CKWYR(double *  rho, double *  T, double *  y,  double *  wdot)
{
    int id; /*loop counter */
    double c[52]; /*temporary storage */
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
    c[25] = 1e6 * (*rho) * y[25]*imw[25]; 
    c[26] = 1e6 * (*rho) * y[26]*imw[26]; 
    c[27] = 1e6 * (*rho) * y[27]*imw[27]; 
    c[28] = 1e6 * (*rho) * y[28]*imw[28]; 
    c[29] = 1e6 * (*rho) * y[29]*imw[29]; 
    c[30] = 1e6 * (*rho) * y[30]*imw[30]; 
    c[31] = 1e6 * (*rho) * y[31]*imw[31]; 
    c[32] = 1e6 * (*rho) * y[32]*imw[32]; 
    c[33] = 1e6 * (*rho) * y[33]*imw[33]; 
    c[34] = 1e6 * (*rho) * y[34]*imw[34]; 
    c[35] = 1e6 * (*rho) * y[35]*imw[35]; 
    c[36] = 1e6 * (*rho) * y[36]*imw[36]; 
    c[37] = 1e6 * (*rho) * y[37]*imw[37]; 
    c[38] = 1e6 * (*rho) * y[38]*imw[38]; 
    c[39] = 1e6 * (*rho) * y[39]*imw[39]; 
    c[40] = 1e6 * (*rho) * y[40]*imw[40]; 
    c[41] = 1e6 * (*rho) * y[41]*imw[41]; 
    c[42] = 1e6 * (*rho) * y[42]*imw[42]; 
    c[43] = 1e6 * (*rho) * y[43]*imw[43]; 
    c[44] = 1e6 * (*rho) * y[44]*imw[44]; 
    c[45] = 1e6 * (*rho) * y[45]*imw[45]; 
    c[46] = 1e6 * (*rho) * y[46]*imw[46]; 
    c[47] = 1e6 * (*rho) * y[47]*imw[47]; 
    c[48] = 1e6 * (*rho) * y[48]*imw[48]; 
    c[49] = 1e6 * (*rho) * y[49]*imw[49]; 
    c[50] = 1e6 * (*rho) * y[50]*imw[50]; 
    c[51] = 1e6 * (*rho) * y[51]*imw[51]; 

    /*call productionRate */
    productionRate(wdot, c, *T);

    /*convert to chemkin units */
    for (id = 0; id < 52; ++id) {
        wdot[id] *= 1.0e-6;
    }
}


/*Returns the molar production rate of species */
/*Given rho, T, and mole fractions */
void CKWXR(double *  rho, double *  T, double *  x,  double *  wdot)
{
    int id; /*loop counter */
    double c[52]; /*temporary storage */
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
    XW += x[9]*13.019120; /*CH */
    XW += x[10]*29.018520; /*HCO */
    XW += x[11]*14.027090; /*CH2 */
    XW += x[12]*44.009950; /*CO2 */
    XW += x[13]*28.010550; /*CO */
    XW += x[14]*30.026490; /*CH2O */
    XW += x[15]*14.027090; /*CH2GSG */
    XW += x[16]*15.035060; /*CH3 */
    XW += x[17]*31.034460; /*CH3O */
    XW += x[18]*16.043030; /*CH4 */
    XW += x[19]*32.042430; /*CH3OH */
    XW += x[20]*30.070120; /*C2H6 */
    XW += x[21]*29.062150; /*C2H5 */
    XW += x[22]*42.037640; /*CH2CO */
    XW += x[23]*46.025890; /*HOCHO */
    XW += x[24]*47.033860; /*CH3O2 */
    XW += x[25]*48.041830; /*CH3O2H */
    XW += x[26]*26.038240; /*C2H2 */
    XW += x[27]*41.029670; /*HCCO */
    XW += x[28]*27.046210; /*C2H3 */
    XW += x[29]*43.045610; /*CH2CHO */
    XW += x[30]*42.081270; /*C3H6 */
    XW += x[31]*28.054180; /*C2H4 */
    XW += x[32]*45.061550; /*C2H5O */
    XW += x[33]*43.045610; /*CH3CO */
    XW += x[34]*61.060950; /*C2H5O2 */
    XW += x[35]*38.049390; /*C3H2 */
    XW += x[36]*39.057360; /*C3H3 */
    XW += x[37]*40.065330; /*C3H4XA */
    XW += x[38]*41.073300; /*C3H5XA */
    XW += x[39]*43.089240; /*NXC3H7 */
    XW += x[40]*75.088040; /*NXC3H7O2 */
    XW += x[41]*54.092420; /*C4H6 */
    XW += x[42]*55.100390; /*C4H7 */
    XW += x[43]*56.108360; /*C4H8X1 */
    XW += x[44]*57.116330; /*PXC4H9 */
    XW += x[45]*89.115130; /*PXC4H9O2 */
    XW += x[46]*69.127480; /*C5H9 */
    XW += x[47]*70.135450; /*C5H10X1 */
    XW += x[48]*71.143420; /*C5H11X1 */
    XW += x[49]*84.162540; /*C6H12X1 */
    XW += x[50]*99.197600; /*C7H15X2 */
    XW += x[51]*100.205570; /*NXC7H16 */
    /*Extra 1e6 factor to take c to SI */
    ROW = 1e6*(*rho) / XW;

    /*Compute conversion, see Eq 11 */
    for (id = 0; id < 52; ++id) {
        c[id] = x[id]*ROW;
    }

    /*convert to chemkin units */
    productionRate(wdot, c, *T);

    /*convert to chemkin units */
    for (id = 0; id < 52; ++id) {
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
    for (id = 0; id < kd * 52; ++ id) {
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

    /*CH */
    ncf[ 9 * kd + 3 ] = 1; /*C */
    ncf[ 9 * kd + 2 ] = 1; /*H */

    /*HCO */
    ncf[ 10 * kd + 2 ] = 1; /*H */
    ncf[ 10 * kd + 3 ] = 1; /*C */
    ncf[ 10 * kd + 1 ] = 1; /*O */

    /*CH2 */
    ncf[ 11 * kd + 3 ] = 1; /*C */
    ncf[ 11 * kd + 2 ] = 2; /*H */

    /*CO2 */
    ncf[ 12 * kd + 3 ] = 1; /*C */
    ncf[ 12 * kd + 1 ] = 2; /*O */

    /*CO */
    ncf[ 13 * kd + 3 ] = 1; /*C */
    ncf[ 13 * kd + 1 ] = 1; /*O */

    /*CH2O */
    ncf[ 14 * kd + 3 ] = 1; /*C */
    ncf[ 14 * kd + 2 ] = 2; /*H */
    ncf[ 14 * kd + 1 ] = 1; /*O */

    /*CH2GSG */
    ncf[ 15 * kd + 3 ] = 1; /*C */
    ncf[ 15 * kd + 2 ] = 2; /*H */

    /*CH3 */
    ncf[ 16 * kd + 3 ] = 1; /*C */
    ncf[ 16 * kd + 2 ] = 3; /*H */

    /*CH3O */
    ncf[ 17 * kd + 3 ] = 1; /*C */
    ncf[ 17 * kd + 2 ] = 3; /*H */
    ncf[ 17 * kd + 1 ] = 1; /*O */

    /*CH4 */
    ncf[ 18 * kd + 3 ] = 1; /*C */
    ncf[ 18 * kd + 2 ] = 4; /*H */

    /*CH3OH */
    ncf[ 19 * kd + 3 ] = 1; /*C */
    ncf[ 19 * kd + 2 ] = 4; /*H */
    ncf[ 19 * kd + 1 ] = 1; /*O */

    /*C2H6 */
    ncf[ 20 * kd + 3 ] = 2; /*C */
    ncf[ 20 * kd + 2 ] = 6; /*H */

    /*C2H5 */
    ncf[ 21 * kd + 3 ] = 2; /*C */
    ncf[ 21 * kd + 2 ] = 5; /*H */

    /*CH2CO */
    ncf[ 22 * kd + 3 ] = 2; /*C */
    ncf[ 22 * kd + 2 ] = 2; /*H */
    ncf[ 22 * kd + 1 ] = 1; /*O */

    /*HOCHO */
    ncf[ 23 * kd + 3 ] = 1; /*C */
    ncf[ 23 * kd + 2 ] = 2; /*H */
    ncf[ 23 * kd + 1 ] = 2; /*O */

    /*CH3O2 */
    ncf[ 24 * kd + 3 ] = 1; /*C */
    ncf[ 24 * kd + 2 ] = 3; /*H */
    ncf[ 24 * kd + 1 ] = 2; /*O */

    /*CH3O2H */
    ncf[ 25 * kd + 3 ] = 1; /*C */
    ncf[ 25 * kd + 2 ] = 4; /*H */
    ncf[ 25 * kd + 1 ] = 2; /*O */

    /*C2H2 */
    ncf[ 26 * kd + 3 ] = 2; /*C */
    ncf[ 26 * kd + 2 ] = 2; /*H */

    /*HCCO */
    ncf[ 27 * kd + 2 ] = 1; /*H */
    ncf[ 27 * kd + 3 ] = 2; /*C */
    ncf[ 27 * kd + 1 ] = 1; /*O */

    /*C2H3 */
    ncf[ 28 * kd + 3 ] = 2; /*C */
    ncf[ 28 * kd + 2 ] = 3; /*H */

    /*CH2CHO */
    ncf[ 29 * kd + 1 ] = 1; /*O */
    ncf[ 29 * kd + 2 ] = 3; /*H */
    ncf[ 29 * kd + 3 ] = 2; /*C */

    /*C3H6 */
    ncf[ 30 * kd + 3 ] = 3; /*C */
    ncf[ 30 * kd + 2 ] = 6; /*H */

    /*C2H4 */
    ncf[ 31 * kd + 3 ] = 2; /*C */
    ncf[ 31 * kd + 2 ] = 4; /*H */

    /*C2H5O */
    ncf[ 32 * kd + 3 ] = 2; /*C */
    ncf[ 32 * kd + 2 ] = 5; /*H */
    ncf[ 32 * kd + 1 ] = 1; /*O */

    /*CH3CO */
    ncf[ 33 * kd + 3 ] = 2; /*C */
    ncf[ 33 * kd + 2 ] = 3; /*H */
    ncf[ 33 * kd + 1 ] = 1; /*O */

    /*C2H5O2 */
    ncf[ 34 * kd + 3 ] = 2; /*C */
    ncf[ 34 * kd + 2 ] = 5; /*H */
    ncf[ 34 * kd + 1 ] = 2; /*O */

    /*C3H2 */
    ncf[ 35 * kd + 2 ] = 2; /*H */
    ncf[ 35 * kd + 3 ] = 3; /*C */

    /*C3H3 */
    ncf[ 36 * kd + 3 ] = 3; /*C */
    ncf[ 36 * kd + 2 ] = 3; /*H */

    /*C3H4XA */
    ncf[ 37 * kd + 2 ] = 4; /*H */
    ncf[ 37 * kd + 3 ] = 3; /*C */

    /*C3H5XA */
    ncf[ 38 * kd + 3 ] = 3; /*C */
    ncf[ 38 * kd + 2 ] = 5; /*H */

    /*NXC3H7 */
    ncf[ 39 * kd + 3 ] = 3; /*C */
    ncf[ 39 * kd + 2 ] = 7; /*H */

    /*NXC3H7O2 */
    ncf[ 40 * kd + 3 ] = 3; /*C */
    ncf[ 40 * kd + 2 ] = 7; /*H */
    ncf[ 40 * kd + 1 ] = 2; /*O */

    /*C4H6 */
    ncf[ 41 * kd + 3 ] = 4; /*C */
    ncf[ 41 * kd + 2 ] = 6; /*H */

    /*C4H7 */
    ncf[ 42 * kd + 3 ] = 4; /*C */
    ncf[ 42 * kd + 2 ] = 7; /*H */

    /*C4H8X1 */
    ncf[ 43 * kd + 3 ] = 4; /*C */
    ncf[ 43 * kd + 2 ] = 8; /*H */

    /*PXC4H9 */
    ncf[ 44 * kd + 3 ] = 4; /*C */
    ncf[ 44 * kd + 2 ] = 9; /*H */

    /*PXC4H9O2 */
    ncf[ 45 * kd + 3 ] = 4; /*C */
    ncf[ 45 * kd + 2 ] = 9; /*H */
    ncf[ 45 * kd + 1 ] = 2; /*O */

    /*C5H9 */
    ncf[ 46 * kd + 3 ] = 5; /*C */
    ncf[ 46 * kd + 2 ] = 9; /*H */

    /*C5H10X1 */
    ncf[ 47 * kd + 3 ] = 5; /*C */
    ncf[ 47 * kd + 2 ] = 10; /*H */

    /*C5H11X1 */
    ncf[ 48 * kd + 3 ] = 5; /*C */
    ncf[ 48 * kd + 2 ] = 11; /*H */

    /*C6H12X1 */
    ncf[ 49 * kd + 3 ] = 6; /*C */
    ncf[ 49 * kd + 2 ] = 12; /*H */

    /*C7H15X2 */
    ncf[ 50 * kd + 3 ] = 7; /*C */
    ncf[ 50 * kd + 2 ] = 15; /*H */

    /*NXC7H16 */
    ncf[ 51 * kd + 3 ] = 7; /*C */
    ncf[ 51 * kd + 2 ] = 16; /*H */


}

#ifndef AMREX_USE_CUDA
static double T_save = -1;
#ifdef _OPENMP
#pragma omp threadprivate(T_save)
#endif

static double k_f_save[218];
#ifdef _OPENMP
#pragma omp threadprivate(k_f_save)
#endif

static double Kc_save[218];
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
    }

    double qdot, q_f[218], q_r[218];
    double sc_qss[0];
    comp_qfqr(q_f, q_r, sc, sc_qss, tc, invT);

    for (int i = 0; i < 52; ++i) {
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
    wdot[16] -= qdot;
    wdot[19] += qdot;

    qdot = q_f[3]-q_r[3];
    wdot[3] -= qdot;
    wdot[16] -= qdot;
    wdot[18] += qdot;

    qdot = q_f[4]-q_r[4];
    wdot[16] -= 2.000000 * qdot;
    wdot[20] += qdot;

    qdot = q_f[5]-q_r[5];
    wdot[11] -= qdot;
    wdot[13] -= qdot;
    wdot[22] += qdot;

    qdot = q_f[6]-q_r[6];
    wdot[1] -= qdot;
    wdot[12] += qdot;
    wdot[13] -= qdot;

    qdot = q_f[7]-q_r[7];
    wdot[3] += qdot;
    wdot[14] += qdot;
    wdot[17] -= qdot;

    qdot = q_f[8]-q_r[8];
    wdot[3] += qdot;
    wdot[26] += qdot;
    wdot[28] -= qdot;

    qdot = q_f[9]-q_r[9];
    wdot[3] -= qdot;
    wdot[21] += qdot;
    wdot[31] -= qdot;

    qdot = q_f[10]-q_r[10];
    wdot[13] += qdot;
    wdot[16] += qdot;
    wdot[33] -= qdot;

    qdot = q_f[11]-q_r[11];
    wdot[3] -= qdot;
    wdot[4] -= qdot;
    wdot[5] += qdot;

    qdot = q_f[12]-q_r[12];
    wdot[11] += qdot;
    wdot[15] -= qdot;

    qdot = q_f[13]-q_r[13];
    wdot[11] -= qdot;
    wdot[15] += qdot;

    qdot = q_f[14]-q_r[14];
    wdot[3] += qdot;
    wdot[10] -= qdot;
    wdot[13] += qdot;

    qdot = q_f[15]-q_r[15];
    wdot[6] += qdot;
    wdot[16] += qdot;
    wdot[24] -= qdot;

    qdot = q_f[16]-q_r[16];
    wdot[6] -= qdot;
    wdot[16] -= qdot;
    wdot[24] += qdot;

    qdot = q_f[17]-q_r[17];
    wdot[14] += qdot;
    wdot[16] += qdot;
    wdot[32] -= qdot;

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
    wdot[9] -= qdot;
    wdot[10] += qdot;

    qdot = q_f[34]-q_r[34];
    wdot[3] += 2.000000 * qdot;
    wdot[6] -= qdot;
    wdot[11] -= qdot;
    wdot[12] += qdot;

    qdot = q_f[35]-q_r[35];
    wdot[5] += qdot;
    wdot[6] -= qdot;
    wdot[11] -= qdot;
    wdot[13] += qdot;

    qdot = q_f[36]-q_r[36];
    wdot[1] -= qdot;
    wdot[3] += 2.000000 * qdot;
    wdot[11] -= qdot;
    wdot[13] += qdot;

    qdot = q_f[37]-q_r[37];
    wdot[1] += qdot;
    wdot[6] -= qdot;
    wdot[11] -= qdot;
    wdot[14] += qdot;

    qdot = q_f[38]-q_r[38];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[9] += qdot;
    wdot[11] -= qdot;

    qdot = q_f[39]-q_r[39];
    wdot[2] -= qdot;
    wdot[3] += qdot;
    wdot[9] -= qdot;
    wdot[11] += qdot;

    qdot = q_f[40]-q_r[40];
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[9] += qdot;
    wdot[11] -= qdot;

    qdot = q_f[41]-q_r[41];
    wdot[4] += qdot;
    wdot[5] -= qdot;
    wdot[9] -= qdot;
    wdot[11] += qdot;

    qdot = q_f[42]-q_r[42];
    wdot[2] += qdot;
    wdot[6] -= qdot;
    wdot[11] -= qdot;
    wdot[12] += qdot;

    qdot = q_f[43]-q_r[43];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[9] += qdot;
    wdot[15] -= qdot;

    qdot = q_f[44]-q_r[44];
    wdot[2] -= qdot;
    wdot[3] += qdot;
    wdot[15] -= qdot;
    wdot[16] += qdot;

    qdot = q_f[45]-q_r[45];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[15] += qdot;
    wdot[16] -= qdot;

    qdot = q_f[46]-q_r[46];
    wdot[3] += qdot;
    wdot[4] += qdot;
    wdot[6] -= qdot;
    wdot[13] += qdot;
    wdot[15] -= qdot;

    qdot = q_f[47]-q_r[47];
    wdot[3] += qdot;
    wdot[4] -= qdot;
    wdot[14] += qdot;
    wdot[15] -= qdot;

    qdot = q_f[48]-q_r[48];
    wdot[2] += qdot;
    wdot[4] -= qdot;
    wdot[14] += qdot;
    wdot[16] -= qdot;

    qdot = q_f[49]-q_r[49];
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[15] += qdot;
    wdot[16] -= qdot;

    qdot = q_f[50]-q_r[50];
    wdot[4] += qdot;
    wdot[5] -= qdot;
    wdot[15] -= qdot;
    wdot[16] += qdot;

    qdot = q_f[51]-q_r[51];
    wdot[1] -= qdot;
    wdot[3] += qdot;
    wdot[14] += qdot;
    wdot[16] -= qdot;

    qdot = q_f[52]-q_r[52];
    wdot[4] += qdot;
    wdot[7] -= qdot;
    wdot[16] -= qdot;
    wdot[17] += qdot;

    qdot = q_f[53]-q_r[53];
    wdot[6] += qdot;
    wdot[7] -= qdot;
    wdot[16] -= qdot;
    wdot[18] += qdot;

    qdot = q_f[54]-q_r[54];
    wdot[4] += qdot;
    wdot[6] -= qdot;
    wdot[14] += qdot;
    wdot[16] -= qdot;

    qdot = q_f[55]-q_r[55];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[11] += qdot;
    wdot[16] -= qdot;

    qdot = q_f[56]-q_r[56];
    wdot[2] -= qdot;
    wdot[3] += qdot;
    wdot[11] -= qdot;
    wdot[16] += qdot;

    qdot = q_f[57]-q_r[57];
    wdot[3] += qdot;
    wdot[16] -= 2.000000 * qdot;
    wdot[21] += qdot;

    qdot = q_f[58]-q_r[58];
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[11] += qdot;
    wdot[16] -= qdot;

    qdot = q_f[59]-q_r[59];
    wdot[4] += qdot;
    wdot[5] -= qdot;
    wdot[11] -= qdot;
    wdot[16] += qdot;

    qdot = q_f[60]-q_r[60];
    wdot[1] -= qdot;
    wdot[4] += qdot;
    wdot[16] += qdot;
    wdot[18] -= qdot;

    qdot = q_f[61]-q_r[61];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[16] += qdot;
    wdot[18] -= qdot;

    qdot = q_f[62]-q_r[62];
    wdot[2] -= qdot;
    wdot[3] += qdot;
    wdot[16] -= qdot;
    wdot[18] += qdot;

    qdot = q_f[63]-q_r[63];
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[16] += qdot;
    wdot[18] -= qdot;

    qdot = q_f[64]-q_r[64];
    wdot[4] += qdot;
    wdot[5] -= qdot;
    wdot[16] -= qdot;
    wdot[18] += qdot;

    qdot = q_f[65]-q_r[65];
    wdot[3] += qdot;
    wdot[4] -= qdot;
    wdot[12] += qdot;
    wdot[13] -= qdot;

    qdot = q_f[66]-q_r[66];
    wdot[3] -= qdot;
    wdot[4] += qdot;
    wdot[12] -= qdot;
    wdot[13] += qdot;

    qdot = q_f[67]-q_r[67];
    wdot[6] -= qdot;
    wdot[7] += qdot;
    wdot[10] -= qdot;
    wdot[13] += qdot;

    qdot = q_f[68]-q_r[68];
    wdot[1] -= qdot;
    wdot[3] += qdot;
    wdot[10] -= qdot;
    wdot[12] += qdot;

    qdot = q_f[69]-q_r[69];
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[10] -= qdot;
    wdot[13] += qdot;

    qdot = q_f[70]-q_r[70];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[10] -= qdot;
    wdot[13] += qdot;

    qdot = q_f[71]-q_r[71];
    wdot[1] -= qdot;
    wdot[4] += qdot;
    wdot[10] -= qdot;
    wdot[13] += qdot;

    qdot = q_f[72]-q_r[72];
    wdot[10] -= qdot;
    wdot[13] += qdot;
    wdot[16] -= qdot;
    wdot[18] += qdot;

    qdot = q_f[73]-q_r[73];
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[10] += qdot;
    wdot[14] -= qdot;

    qdot = q_f[74]-q_r[74];
    wdot[1] -= qdot;
    wdot[4] += qdot;
    wdot[10] += qdot;
    wdot[14] -= qdot;

    qdot = q_f[75]-q_r[75];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[10] += qdot;
    wdot[14] -= qdot;

    qdot = q_f[76]-q_r[76];
    wdot[10] += qdot;
    wdot[14] -= qdot;
    wdot[16] -= qdot;
    wdot[18] += qdot;

    qdot = q_f[77]-q_r[77];
    wdot[14] += qdot;
    wdot[17] -= 2.000000 * qdot;
    wdot[19] += qdot;

    qdot = q_f[78]-q_r[78];
    wdot[6] -= qdot;
    wdot[7] += qdot;
    wdot[14] += qdot;
    wdot[17] -= qdot;

    qdot = q_f[79]-q_r[79];
    wdot[2] -= qdot;
    wdot[3] += qdot;
    wdot[17] -= qdot;
    wdot[19] += qdot;

    qdot = q_f[80]-q_r[80];
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[17] += qdot;
    wdot[19] -= qdot;

    qdot = q_f[81]-q_r[81];
    wdot[12] -= qdot;
    wdot[13] += qdot;
    wdot[14] += qdot;
    wdot[15] -= qdot;

    qdot = q_f[82]-q_r[82];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[4] += qdot;
    wdot[13] += qdot;
    wdot[23] -= qdot;

    qdot = q_f[83]-q_r[83];
    wdot[4] -= qdot;
    wdot[4] += qdot;
    wdot[5] += qdot;
    wdot[13] += qdot;
    wdot[23] -= qdot;

    qdot = q_f[84]-q_r[84];
    wdot[4] += qdot;
    wdot[10] += qdot;
    wdot[23] -= qdot;

    qdot = q_f[85]-q_r[85];
    wdot[4] -= qdot;
    wdot[10] -= qdot;
    wdot[23] += qdot;

    qdot = q_f[86]-q_r[86];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[3] += qdot;
    wdot[12] += qdot;
    wdot[23] -= qdot;

    qdot = q_f[87]-q_r[87];
    wdot[3] += qdot;
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[12] += qdot;
    wdot[23] -= qdot;

    qdot = q_f[88]-q_r[88];
    wdot[6] += qdot;
    wdot[17] += 2.000000 * qdot;
    wdot[24] -= 2.000000 * qdot;

    qdot = q_f[89]-q_r[89];
    wdot[16] -= qdot;
    wdot[17] += 2.000000 * qdot;
    wdot[24] -= qdot;

    qdot = q_f[90]-q_r[90];
    wdot[6] += qdot;
    wdot[14] += qdot;
    wdot[19] += qdot;
    wdot[24] -= 2.000000 * qdot;

    qdot = q_f[91]-q_r[91];
    wdot[6] += qdot;
    wdot[7] -= qdot;
    wdot[24] -= qdot;
    wdot[25] += qdot;

    qdot = q_f[92]-q_r[92];
    wdot[4] += qdot;
    wdot[17] += qdot;
    wdot[25] -= qdot;

    qdot = q_f[93]-q_r[93];
    wdot[1] -= qdot;
    wdot[11] += qdot;
    wdot[13] += qdot;
    wdot[26] -= qdot;

    qdot = q_f[94]-q_r[94];
    wdot[1] -= qdot;
    wdot[3] += qdot;
    wdot[26] -= qdot;
    wdot[27] += qdot;

    qdot = q_f[95]-q_r[95];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[26] += qdot;
    wdot[28] -= qdot;

    qdot = q_f[96]-q_r[96];
    wdot[1] += qdot;
    wdot[6] -= qdot;
    wdot[28] -= qdot;
    wdot[29] += qdot;

    qdot = q_f[97]-q_r[97];
    wdot[16] -= qdot;
    wdot[28] -= qdot;
    wdot[30] += qdot;

    qdot = q_f[98]-q_r[98];
    wdot[6] -= qdot;
    wdot[7] += qdot;
    wdot[26] += qdot;
    wdot[28] -= qdot;

    qdot = q_f[99]-q_r[99];
    wdot[6] -= qdot;
    wdot[10] += qdot;
    wdot[14] += qdot;
    wdot[28] -= qdot;

    qdot = q_f[100]-q_r[100];
    wdot[16] -= qdot;
    wdot[18] += qdot;
    wdot[28] += qdot;
    wdot[31] -= qdot;

    qdot = q_f[101]-q_r[101];
    wdot[1] -= qdot;
    wdot[10] += qdot;
    wdot[16] += qdot;
    wdot[31] -= qdot;

    qdot = q_f[102]-q_r[102];
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[28] += qdot;
    wdot[31] -= qdot;

    qdot = q_f[103]-q_r[103];
    wdot[1] -= qdot;
    wdot[3] += qdot;
    wdot[29] += qdot;
    wdot[31] -= qdot;

    qdot = q_f[104]-q_r[104];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[28] += qdot;
    wdot[31] -= qdot;

    qdot = q_f[105]-q_r[105];
    wdot[2] -= qdot;
    wdot[3] += qdot;
    wdot[28] -= qdot;
    wdot[31] += qdot;

    qdot = q_f[106]-q_r[106];
    wdot[3] -= qdot;
    wdot[20] += qdot;
    wdot[21] -= qdot;

    qdot = q_f[107]-q_r[107];
    wdot[17] += qdot;
    wdot[21] -= qdot;
    wdot[24] -= qdot;
    wdot[32] += qdot;

    qdot = q_f[108]-q_r[108];
    wdot[4] += qdot;
    wdot[7] -= qdot;
    wdot[21] -= qdot;
    wdot[32] += qdot;

    qdot = q_f[109]-q_r[109];
    wdot[6] -= qdot;
    wdot[7] += qdot;
    wdot[21] -= qdot;
    wdot[31] += qdot;

    qdot = q_f[110]-q_r[110];
    wdot[1] -= qdot;
    wdot[4] += qdot;
    wdot[20] -= qdot;
    wdot[21] += qdot;

    qdot = q_f[111]-q_r[111];
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[20] -= qdot;
    wdot[21] += qdot;

    qdot = q_f[112]-q_r[112];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[20] -= qdot;
    wdot[21] += qdot;

    qdot = q_f[113]-q_r[113];
    wdot[1] -= qdot;
    wdot[3] += qdot;
    wdot[13] += 2.000000 * qdot;
    wdot[27] -= qdot;

    qdot = q_f[114]-q_r[114];
    wdot[4] -= qdot;
    wdot[10] += 2.000000 * qdot;
    wdot[27] -= qdot;

    qdot = q_f[115]-q_r[115];
    wdot[6] -= qdot;
    wdot[10] += qdot;
    wdot[12] += qdot;
    wdot[27] -= qdot;

    qdot = q_f[116]-q_r[116];
    wdot[3] -= qdot;
    wdot[13] += qdot;
    wdot[15] += qdot;
    wdot[27] -= qdot;

    qdot = q_f[117]-q_r[117];
    wdot[3] += qdot;
    wdot[13] -= qdot;
    wdot[15] -= qdot;
    wdot[27] += qdot;

    qdot = q_f[118]-q_r[118];
    wdot[1] -= qdot;
    wdot[4] += qdot;
    wdot[22] -= qdot;
    wdot[27] += qdot;

    qdot = q_f[119]-q_r[119];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[22] -= qdot;
    wdot[27] += qdot;

    qdot = q_f[120]-q_r[120];
    wdot[2] -= qdot;
    wdot[3] += qdot;
    wdot[22] += qdot;
    wdot[27] -= qdot;

    qdot = q_f[121]-q_r[121];
    wdot[3] -= qdot;
    wdot[13] += qdot;
    wdot[16] += qdot;
    wdot[22] -= qdot;

    qdot = q_f[122]-q_r[122];
    wdot[1] -= qdot;
    wdot[11] += qdot;
    wdot[12] += qdot;
    wdot[22] -= qdot;

    qdot = q_f[123]-q_r[123];
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[22] -= qdot;
    wdot[27] += qdot;

    qdot = q_f[124]-q_r[124];
    wdot[4] += qdot;
    wdot[6] -= qdot;
    wdot[13] += qdot;
    wdot[14] += qdot;
    wdot[29] -= qdot;

    qdot = q_f[125]-q_r[125];
    wdot[3] += qdot;
    wdot[22] += qdot;
    wdot[29] -= qdot;

    qdot = q_f[126]-q_r[126];
    wdot[3] -= qdot;
    wdot[22] -= qdot;
    wdot[29] += qdot;

    qdot = q_f[127]-q_r[127];
    wdot[6] += qdot;
    wdot[21] += qdot;
    wdot[34] -= qdot;

    qdot = q_f[128]-q_r[128];
    wdot[6] -= qdot;
    wdot[21] -= qdot;
    wdot[34] += qdot;

    qdot = q_f[129]-q_r[129];
    wdot[7] += qdot;
    wdot[31] += qdot;
    wdot[34] -= qdot;

    qdot = q_f[130]-q_r[130];
    wdot[3] += qdot;
    wdot[6] -= qdot;
    wdot[13] += qdot;
    wdot[27] += qdot;
    wdot[35] -= qdot;

    qdot = q_f[131]-q_r[131];
    wdot[4] -= qdot;
    wdot[10] += qdot;
    wdot[26] += qdot;
    wdot[35] -= qdot;

    qdot = q_f[132]-q_r[132];
    wdot[6] -= qdot;
    wdot[10] += qdot;
    wdot[22] += qdot;
    wdot[36] -= qdot;

    qdot = q_f[133]-q_r[133];
    wdot[6] += qdot;
    wdot[7] -= qdot;
    wdot[36] -= qdot;
    wdot[37] += qdot;

    qdot = q_f[134]-q_r[134];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[35] += qdot;
    wdot[36] -= qdot;

    qdot = q_f[135]-q_r[135];
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[35] += qdot;
    wdot[36] -= qdot;

    qdot = q_f[136]-q_r[136];
    wdot[4] += qdot;
    wdot[5] -= qdot;
    wdot[35] -= qdot;
    wdot[36] += qdot;

    qdot = q_f[137]-q_r[137];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[36] += qdot;
    wdot[37] -= qdot;

    qdot = q_f[138]-q_r[138];
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[36] += qdot;
    wdot[37] -= qdot;

    qdot = q_f[139]-q_r[139];
    wdot[1] -= qdot;
    wdot[13] += qdot;
    wdot[31] += qdot;
    wdot[37] -= qdot;

    qdot = q_f[140]-q_r[140];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[37] += qdot;
    wdot[38] -= qdot;

    qdot = q_f[141]-q_r[141];
    wdot[6] += qdot;
    wdot[7] -= qdot;
    wdot[30] += qdot;
    wdot[38] -= qdot;

    qdot = q_f[142]-q_r[142];
    wdot[3] -= qdot;
    wdot[30] += qdot;
    wdot[38] -= qdot;

    qdot = q_f[143]-q_r[143];
    wdot[16] += qdot;
    wdot[26] += qdot;
    wdot[38] -= qdot;

    qdot = q_f[144]-q_r[144];
    wdot[3] += qdot;
    wdot[37] += qdot;
    wdot[38] -= qdot;

    qdot = q_f[145]-q_r[145];
    wdot[3] -= qdot;
    wdot[37] -= qdot;
    wdot[38] += qdot;

    qdot = q_f[146]-q_r[146];
    wdot[10] += qdot;
    wdot[14] -= qdot;
    wdot[30] += qdot;
    wdot[38] -= qdot;

    qdot = q_f[147]-q_r[147];
    wdot[30] += qdot;
    wdot[37] += qdot;
    wdot[38] -= 2.000000 * qdot;

    qdot = q_f[148]-q_r[148];
    wdot[3] -= qdot;
    wdot[16] += qdot;
    wdot[30] -= qdot;
    wdot[31] += qdot;

    qdot = q_f[149]-q_r[149];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[30] -= qdot;
    wdot[38] += qdot;

    qdot = q_f[150]-q_r[150];
    wdot[1] -= qdot;
    wdot[10] += qdot;
    wdot[21] += qdot;
    wdot[30] -= qdot;

    qdot = q_f[151]-q_r[151];
    wdot[1] -= qdot;
    wdot[4] += qdot;
    wdot[30] -= qdot;
    wdot[38] += qdot;

    qdot = q_f[152]-q_r[152];
    wdot[1] -= qdot;
    wdot[3] += qdot;
    wdot[16] += qdot;
    wdot[22] += qdot;
    wdot[30] -= qdot;

    qdot = q_f[153]-q_r[153];
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[30] -= qdot;
    wdot[38] += qdot;

    qdot = q_f[154]-q_r[154];
    wdot[6] -= qdot;
    wdot[7] += qdot;
    wdot[30] += qdot;
    wdot[39] -= qdot;

    qdot = q_f[155]-q_r[155];
    wdot[16] += qdot;
    wdot[31] += qdot;
    wdot[39] -= qdot;

    qdot = q_f[156]-q_r[156];
    wdot[16] -= qdot;
    wdot[31] -= qdot;
    wdot[39] += qdot;

    qdot = q_f[157]-q_r[157];
    wdot[3] += qdot;
    wdot[30] += qdot;
    wdot[39] -= qdot;

    qdot = q_f[158]-q_r[158];
    wdot[3] -= qdot;
    wdot[30] -= qdot;
    wdot[39] += qdot;

    qdot = q_f[159]-q_r[159];
    wdot[6] += qdot;
    wdot[39] += qdot;
    wdot[40] -= qdot;

    qdot = q_f[160]-q_r[160];
    wdot[6] -= qdot;
    wdot[39] -= qdot;
    wdot[40] += qdot;

    qdot = q_f[161]-q_r[161];
    wdot[28] += 2.000000 * qdot;
    wdot[41] -= qdot;

    qdot = q_f[162]-q_r[162];
    wdot[28] -= 2.000000 * qdot;
    wdot[41] += qdot;

    qdot = q_f[163]-q_r[163];
    wdot[4] -= qdot;
    wdot[14] += qdot;
    wdot[38] += qdot;
    wdot[41] -= qdot;

    qdot = q_f[164]-q_r[164];
    wdot[4] -= qdot;
    wdot[21] += qdot;
    wdot[22] += qdot;
    wdot[41] -= qdot;

    qdot = q_f[165]-q_r[165];
    wdot[1] -= qdot;
    wdot[22] += qdot;
    wdot[31] += qdot;
    wdot[41] -= qdot;

    qdot = q_f[166]-q_r[166];
    wdot[3] -= qdot;
    wdot[28] += qdot;
    wdot[31] += qdot;
    wdot[41] -= qdot;

    qdot = q_f[167]-q_r[167];
    wdot[1] -= qdot;
    wdot[14] += qdot;
    wdot[37] += qdot;
    wdot[41] -= qdot;

    qdot = q_f[168]-q_r[168];
    wdot[3] -= qdot;
    wdot[42] -= qdot;
    wdot[43] += qdot;

    qdot = q_f[169]-q_r[169];
    wdot[30] += qdot;
    wdot[38] -= qdot;
    wdot[41] += qdot;
    wdot[42] -= qdot;

    qdot = q_f[170]-q_r[170];
    wdot[20] += qdot;
    wdot[21] -= qdot;
    wdot[41] += qdot;
    wdot[42] -= qdot;

    qdot = q_f[171]-q_r[171];
    wdot[3] += qdot;
    wdot[41] += qdot;
    wdot[42] -= qdot;

    qdot = q_f[172]-q_r[172];
    wdot[3] -= qdot;
    wdot[41] -= qdot;
    wdot[42] += qdot;

    qdot = q_f[173]-q_r[173];
    wdot[16] -= qdot;
    wdot[18] += qdot;
    wdot[41] += qdot;
    wdot[42] -= qdot;

    qdot = q_f[174]-q_r[174];
    wdot[6] += qdot;
    wdot[7] -= qdot;
    wdot[42] -= qdot;
    wdot[43] += qdot;

    qdot = q_f[175]-q_r[175];
    wdot[6] -= qdot;
    wdot[7] += qdot;
    wdot[41] += qdot;
    wdot[42] -= qdot;

    qdot = q_f[176]-q_r[176];
    wdot[28] += qdot;
    wdot[31] += qdot;
    wdot[42] -= qdot;

    qdot = q_f[177]-q_r[177];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[41] += qdot;
    wdot[42] -= qdot;

    qdot = q_f[178]-q_r[178];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[42] += qdot;
    wdot[43] -= qdot;

    qdot = q_f[179]-q_r[179];
    wdot[4] -= qdot;
    wdot[14] += qdot;
    wdot[39] += qdot;
    wdot[43] -= qdot;

    qdot = q_f[180]-q_r[180];
    wdot[4] -= qdot;
    wdot[20] += qdot;
    wdot[33] += qdot;
    wdot[43] -= qdot;

    qdot = q_f[181]-q_r[181];
    wdot[1] -= qdot;
    wdot[21] += qdot;
    wdot[33] += qdot;
    wdot[43] -= qdot;

    qdot = q_f[182]-q_r[182];
    wdot[1] -= qdot;
    wdot[14] += qdot;
    wdot[30] += qdot;
    wdot[43] -= qdot;

    qdot = q_f[183]-q_r[183];
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[42] += qdot;
    wdot[43] -= qdot;

    qdot = q_f[184]-q_r[184];
    wdot[16] += qdot;
    wdot[38] += qdot;
    wdot[43] -= qdot;

    qdot = q_f[185]-q_r[185];
    wdot[16] -= qdot;
    wdot[38] -= qdot;
    wdot[43] += qdot;

    qdot = q_f[186]-q_r[186];
    wdot[3] += qdot;
    wdot[43] += qdot;
    wdot[44] -= qdot;

    qdot = q_f[187]-q_r[187];
    wdot[3] -= qdot;
    wdot[43] -= qdot;
    wdot[44] += qdot;

    qdot = q_f[188]-q_r[188];
    wdot[21] += qdot;
    wdot[31] += qdot;
    wdot[44] -= qdot;

    qdot = q_f[189]-q_r[189];
    wdot[6] += qdot;
    wdot[44] += qdot;
    wdot[45] -= qdot;

    qdot = q_f[190]-q_r[190];
    wdot[6] -= qdot;
    wdot[44] -= qdot;
    wdot[45] += qdot;

    qdot = q_f[191]-q_r[191];
    wdot[16] += qdot;
    wdot[41] += qdot;
    wdot[46] -= qdot;

    qdot = q_f[192]-q_r[192];
    wdot[31] += qdot;
    wdot[38] += qdot;
    wdot[46] -= qdot;

    qdot = q_f[193]-q_r[193];
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[46] += qdot;
    wdot[47] -= qdot;

    qdot = q_f[194]-q_r[194];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[46] += qdot;
    wdot[47] -= qdot;

    qdot = q_f[195]-q_r[195];
    wdot[21] += qdot;
    wdot[38] += qdot;
    wdot[47] -= qdot;

    qdot = q_f[196]-q_r[196];
    wdot[21] -= qdot;
    wdot[38] -= qdot;
    wdot[47] += qdot;

    qdot = q_f[197]-q_r[197];
    wdot[1] -= qdot;
    wdot[4] += qdot;
    wdot[46] += qdot;
    wdot[47] -= qdot;

    qdot = q_f[198]-q_r[198];
    wdot[21] += qdot;
    wdot[30] += qdot;
    wdot[48] -= qdot;

    qdot = q_f[199]-q_r[199];
    wdot[31] += qdot;
    wdot[39] += qdot;
    wdot[48] -= qdot;

    qdot = q_f[200]-q_r[200];
    wdot[3] += qdot;
    wdot[47] += qdot;
    wdot[48] -= qdot;

    qdot = q_f[201]-q_r[201];
    wdot[38] += qdot;
    wdot[39] += qdot;
    wdot[49] -= qdot;

    qdot = q_f[202]-q_r[202];
    wdot[4] -= qdot;
    wdot[14] += qdot;
    wdot[48] += qdot;
    wdot[49] -= qdot;

    qdot = q_f[203]-q_r[203];
    wdot[16] += qdot;
    wdot[49] += qdot;
    wdot[50] -= qdot;

    qdot = q_f[204]-q_r[204];
    wdot[30] += qdot;
    wdot[44] += qdot;
    wdot[50] -= qdot;

    qdot = q_f[205]-q_r[205];
    wdot[39] += qdot;
    wdot[43] += qdot;
    wdot[50] -= qdot;

    qdot = q_f[206]-q_r[206];
    wdot[31] += qdot;
    wdot[48] += qdot;
    wdot[50] -= qdot;

    qdot = q_f[207]-q_r[207];
    wdot[21] += qdot;
    wdot[47] += qdot;
    wdot[50] -= qdot;

    qdot = q_f[208]-q_r[208];
    wdot[6] += qdot;
    wdot[7] -= qdot;
    wdot[50] -= qdot;
    wdot[51] += qdot;

    qdot = q_f[209]-q_r[209];
    wdot[24] -= qdot;
    wdot[25] += qdot;
    wdot[50] += qdot;
    wdot[51] -= qdot;

    qdot = q_f[210]-q_r[210];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[50] += qdot;
    wdot[51] -= qdot;

    qdot = q_f[211]-q_r[211];
    wdot[39] += qdot;
    wdot[44] += qdot;
    wdot[51] -= qdot;

    qdot = q_f[212]-q_r[212];
    wdot[7] -= qdot;
    wdot[8] += qdot;
    wdot[50] += qdot;
    wdot[51] -= qdot;

    qdot = q_f[213]-q_r[213];
    wdot[21] += qdot;
    wdot[48] += qdot;
    wdot[51] -= qdot;

    qdot = q_f[214]-q_r[214];
    wdot[17] -= qdot;
    wdot[19] += qdot;
    wdot[50] += qdot;
    wdot[51] -= qdot;

    qdot = q_f[215]-q_r[215];
    wdot[1] -= qdot;
    wdot[4] += qdot;
    wdot[50] += qdot;
    wdot[51] -= qdot;

    qdot = q_f[216]-q_r[216];
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[50] += qdot;
    wdot[51] -= qdot;

    qdot = q_f[217]-q_r[217];
    wdot[16] -= qdot;
    wdot[18] += qdot;
    wdot[50] += qdot;
    wdot[51] -= qdot;

    return;
}

void comp_k_f(double *  tc, double invT, double *  k_f)
{
#ifdef __INTEL_COMPILER
    #pragma simd
#endif
    for (int i=0; i<218; ++i) {
        k_f[i] = prefactor_units[i] * fwd_A[i]
                    * exp(fwd_beta[i] * tc[0] - activation_units[i] * fwd_Ea[i] * invT);
    };
    return;
}

void comp_Kc(double *  tc, double invT, double *  Kc)
{
    /*compute the Gibbs free energy */
    double g_RT[52];
    gibbs(g_RT, tc);

    Kc[0] = g_RT[3] + g_RT[6] - g_RT[7];
    Kc[1] = -2.000000*g_RT[4] + g_RT[8];
    Kc[2] = g_RT[4] + g_RT[16] - g_RT[19];
    Kc[3] = g_RT[3] + g_RT[16] - g_RT[18];
    Kc[4] = 2.000000*g_RT[16] - g_RT[20];
    Kc[5] = g_RT[11] + g_RT[13] - g_RT[22];
    Kc[6] = g_RT[1] - g_RT[12] + g_RT[13];
    Kc[7] = -g_RT[3] - g_RT[14] + g_RT[17];
    Kc[8] = -g_RT[3] - g_RT[26] + g_RT[28];
    Kc[9] = g_RT[3] - g_RT[21] + g_RT[31];
    Kc[10] = -g_RT[13] - g_RT[16] + g_RT[33];
    Kc[11] = g_RT[3] + g_RT[4] - g_RT[5];
    Kc[12] = -g_RT[11] + g_RT[15];
    Kc[13] = g_RT[11] - g_RT[15];
    Kc[14] = -g_RT[3] + g_RT[10] - g_RT[13];
    Kc[15] = -g_RT[6] - g_RT[16] + g_RT[24];
    Kc[16] = g_RT[6] + g_RT[16] - g_RT[24];
    Kc[17] = -g_RT[14] - g_RT[16] + g_RT[32];
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
    Kc[33] = -g_RT[1] + g_RT[6] + g_RT[9] - g_RT[10];
    Kc[34] = -2.000000*g_RT[3] + g_RT[6] + g_RT[11] - g_RT[12];
    Kc[35] = -g_RT[5] + g_RT[6] + g_RT[11] - g_RT[13];
    Kc[36] = g_RT[1] - 2.000000*g_RT[3] + g_RT[11] - g_RT[13];
    Kc[37] = -g_RT[1] + g_RT[6] + g_RT[11] - g_RT[14];
    Kc[38] = -g_RT[2] + g_RT[3] - g_RT[9] + g_RT[11];
    Kc[39] = g_RT[2] - g_RT[3] + g_RT[9] - g_RT[11];
    Kc[40] = g_RT[4] - g_RT[5] - g_RT[9] + g_RT[11];
    Kc[41] = -g_RT[4] + g_RT[5] + g_RT[9] - g_RT[11];
    Kc[42] = -g_RT[2] + g_RT[6] + g_RT[11] - g_RT[12];
    Kc[43] = -g_RT[2] + g_RT[3] - g_RT[9] + g_RT[15];
    Kc[44] = g_RT[2] - g_RT[3] + g_RT[15] - g_RT[16];
    Kc[45] = -g_RT[2] + g_RT[3] - g_RT[15] + g_RT[16];
    Kc[46] = -g_RT[3] - g_RT[4] + g_RT[6] - g_RT[13] + g_RT[15];
    Kc[47] = -g_RT[3] + g_RT[4] - g_RT[14] + g_RT[15];
    Kc[48] = -g_RT[2] + g_RT[4] - g_RT[14] + g_RT[16];
    Kc[49] = g_RT[4] - g_RT[5] - g_RT[15] + g_RT[16];
    Kc[50] = -g_RT[4] + g_RT[5] + g_RT[15] - g_RT[16];
    Kc[51] = g_RT[1] - g_RT[3] - g_RT[14] + g_RT[16];
    Kc[52] = -g_RT[4] + g_RT[7] + g_RT[16] - g_RT[17];
    Kc[53] = -g_RT[6] + g_RT[7] + g_RT[16] - g_RT[18];
    Kc[54] = -g_RT[4] + g_RT[6] - g_RT[14] + g_RT[16];
    Kc[55] = -g_RT[2] + g_RT[3] - g_RT[11] + g_RT[16];
    Kc[56] = g_RT[2] - g_RT[3] + g_RT[11] - g_RT[16];
    Kc[57] = -g_RT[3] + 2.000000*g_RT[16] - g_RT[21];
    Kc[58] = g_RT[4] - g_RT[5] - g_RT[11] + g_RT[16];
    Kc[59] = -g_RT[4] + g_RT[5] + g_RT[11] - g_RT[16];
    Kc[60] = g_RT[1] - g_RT[4] - g_RT[16] + g_RT[18];
    Kc[61] = -g_RT[2] + g_RT[3] - g_RT[16] + g_RT[18];
    Kc[62] = g_RT[2] - g_RT[3] + g_RT[16] - g_RT[18];
    Kc[63] = g_RT[4] - g_RT[5] - g_RT[16] + g_RT[18];
    Kc[64] = -g_RT[4] + g_RT[5] + g_RT[16] - g_RT[18];
    Kc[65] = -g_RT[3] + g_RT[4] - g_RT[12] + g_RT[13];
    Kc[66] = g_RT[3] - g_RT[4] + g_RT[12] - g_RT[13];
    Kc[67] = g_RT[6] - g_RT[7] + g_RT[10] - g_RT[13];
    Kc[68] = g_RT[1] - g_RT[3] + g_RT[10] - g_RT[12];
    Kc[69] = g_RT[4] - g_RT[5] + g_RT[10] - g_RT[13];
    Kc[70] = -g_RT[2] + g_RT[3] + g_RT[10] - g_RT[13];
    Kc[71] = g_RT[1] - g_RT[4] + g_RT[10] - g_RT[13];
    Kc[72] = g_RT[10] - g_RT[13] + g_RT[16] - g_RT[18];
    Kc[73] = g_RT[4] - g_RT[5] - g_RT[10] + g_RT[14];
    Kc[74] = g_RT[1] - g_RT[4] - g_RT[10] + g_RT[14];
    Kc[75] = -g_RT[2] + g_RT[3] - g_RT[10] + g_RT[14];
    Kc[76] = -g_RT[10] + g_RT[14] + g_RT[16] - g_RT[18];
    Kc[77] = -g_RT[14] + 2.000000*g_RT[17] - g_RT[19];
    Kc[78] = g_RT[6] - g_RT[7] - g_RT[14] + g_RT[17];
    Kc[79] = g_RT[2] - g_RT[3] + g_RT[17] - g_RT[19];
    Kc[80] = g_RT[4] - g_RT[5] - g_RT[17] + g_RT[19];
    Kc[81] = g_RT[12] - g_RT[13] - g_RT[14] + g_RT[15];
    Kc[82] = -g_RT[2] + g_RT[3] - g_RT[4] - g_RT[13] + g_RT[23];
    Kc[83] = g_RT[4] - g_RT[4] - g_RT[5] - g_RT[13] + g_RT[23];
    Kc[84] = -g_RT[4] - g_RT[10] + g_RT[23];
    Kc[85] = g_RT[4] + g_RT[10] - g_RT[23];
    Kc[86] = -g_RT[2] + g_RT[3] - g_RT[3] - g_RT[12] + g_RT[23];
    Kc[87] = -g_RT[3] + g_RT[4] - g_RT[5] - g_RT[12] + g_RT[23];
    Kc[88] = -g_RT[6] - 2.000000*g_RT[17] + 2.000000*g_RT[24];
    Kc[89] = g_RT[16] - 2.000000*g_RT[17] + g_RT[24];
    Kc[90] = -g_RT[6] - g_RT[14] - g_RT[19] + 2.000000*g_RT[24];
    Kc[91] = -g_RT[6] + g_RT[7] + g_RT[24] - g_RT[25];
    Kc[92] = -g_RT[4] - g_RT[17] + g_RT[25];
    Kc[93] = g_RT[1] - g_RT[11] - g_RT[13] + g_RT[26];
    Kc[94] = g_RT[1] - g_RT[3] + g_RT[26] - g_RT[27];
    Kc[95] = -g_RT[2] + g_RT[3] - g_RT[26] + g_RT[28];
    Kc[96] = -g_RT[1] + g_RT[6] + g_RT[28] - g_RT[29];
    Kc[97] = g_RT[16] + g_RT[28] - g_RT[30];
    Kc[98] = g_RT[6] - g_RT[7] - g_RT[26] + g_RT[28];
    Kc[99] = g_RT[6] - g_RT[10] - g_RT[14] + g_RT[28];
    Kc[100] = g_RT[16] - g_RT[18] - g_RT[28] + g_RT[31];
    Kc[101] = g_RT[1] - g_RT[10] - g_RT[16] + g_RT[31];
    Kc[102] = g_RT[4] - g_RT[5] - g_RT[28] + g_RT[31];
    Kc[103] = g_RT[1] - g_RT[3] - g_RT[29] + g_RT[31];
    Kc[104] = -g_RT[2] + g_RT[3] - g_RT[28] + g_RT[31];
    Kc[105] = g_RT[2] - g_RT[3] + g_RT[28] - g_RT[31];
    Kc[106] = g_RT[3] - g_RT[20] + g_RT[21];
    Kc[107] = -g_RT[17] + g_RT[21] + g_RT[24] - g_RT[32];
    Kc[108] = -g_RT[4] + g_RT[7] + g_RT[21] - g_RT[32];
    Kc[109] = g_RT[6] - g_RT[7] + g_RT[21] - g_RT[31];
    Kc[110] = g_RT[1] - g_RT[4] + g_RT[20] - g_RT[21];
    Kc[111] = g_RT[4] - g_RT[5] + g_RT[20] - g_RT[21];
    Kc[112] = -g_RT[2] + g_RT[3] + g_RT[20] - g_RT[21];
    Kc[113] = g_RT[1] - g_RT[3] - 2.000000*g_RT[13] + g_RT[27];
    Kc[114] = g_RT[4] - 2.000000*g_RT[10] + g_RT[27];
    Kc[115] = g_RT[6] - g_RT[10] - g_RT[12] + g_RT[27];
    Kc[116] = g_RT[3] - g_RT[13] - g_RT[15] + g_RT[27];
    Kc[117] = -g_RT[3] + g_RT[13] + g_RT[15] - g_RT[27];
    Kc[118] = g_RT[1] - g_RT[4] + g_RT[22] - g_RT[27];
    Kc[119] = -g_RT[2] + g_RT[3] + g_RT[22] - g_RT[27];
    Kc[120] = g_RT[2] - g_RT[3] - g_RT[22] + g_RT[27];
    Kc[121] = g_RT[3] - g_RT[13] - g_RT[16] + g_RT[22];
    Kc[122] = g_RT[1] - g_RT[11] - g_RT[12] + g_RT[22];
    Kc[123] = g_RT[4] - g_RT[5] + g_RT[22] - g_RT[27];
    Kc[124] = -g_RT[4] + g_RT[6] - g_RT[13] - g_RT[14] + g_RT[29];
    Kc[125] = -g_RT[3] - g_RT[22] + g_RT[29];
    Kc[126] = g_RT[3] + g_RT[22] - g_RT[29];
    Kc[127] = -g_RT[6] - g_RT[21] + g_RT[34];
    Kc[128] = g_RT[6] + g_RT[21] - g_RT[34];
    Kc[129] = -g_RT[7] - g_RT[31] + g_RT[34];
    Kc[130] = -g_RT[3] + g_RT[6] - g_RT[13] - g_RT[27] + g_RT[35];
    Kc[131] = g_RT[4] - g_RT[10] - g_RT[26] + g_RT[35];
    Kc[132] = g_RT[6] - g_RT[10] - g_RT[22] + g_RT[36];
    Kc[133] = -g_RT[6] + g_RT[7] + g_RT[36] - g_RT[37];
    Kc[134] = -g_RT[2] + g_RT[3] - g_RT[35] + g_RT[36];
    Kc[135] = g_RT[4] - g_RT[5] - g_RT[35] + g_RT[36];
    Kc[136] = -g_RT[4] + g_RT[5] + g_RT[35] - g_RT[36];
    Kc[137] = -g_RT[2] + g_RT[3] - g_RT[36] + g_RT[37];
    Kc[138] = g_RT[4] - g_RT[5] - g_RT[36] + g_RT[37];
    Kc[139] = g_RT[1] - g_RT[13] - g_RT[31] + g_RT[37];
    Kc[140] = -g_RT[2] + g_RT[3] - g_RT[37] + g_RT[38];
    Kc[141] = -g_RT[6] + g_RT[7] - g_RT[30] + g_RT[38];
    Kc[142] = g_RT[3] - g_RT[30] + g_RT[38];
    Kc[143] = -g_RT[16] - g_RT[26] + g_RT[38];
    Kc[144] = -g_RT[3] - g_RT[37] + g_RT[38];
    Kc[145] = g_RT[3] + g_RT[37] - g_RT[38];
    Kc[146] = -g_RT[10] + g_RT[14] - g_RT[30] + g_RT[38];
    Kc[147] = -g_RT[30] - g_RT[37] + 2.000000*g_RT[38];
    Kc[148] = g_RT[3] - g_RT[16] + g_RT[30] - g_RT[31];
    Kc[149] = -g_RT[2] + g_RT[3] + g_RT[30] - g_RT[38];
    Kc[150] = g_RT[1] - g_RT[10] - g_RT[21] + g_RT[30];
    Kc[151] = g_RT[1] - g_RT[4] + g_RT[30] - g_RT[38];
    Kc[152] = g_RT[1] - g_RT[3] - g_RT[16] - g_RT[22] + g_RT[30];
    Kc[153] = g_RT[4] - g_RT[5] + g_RT[30] - g_RT[38];
    Kc[154] = g_RT[6] - g_RT[7] - g_RT[30] + g_RT[39];
    Kc[155] = -g_RT[16] - g_RT[31] + g_RT[39];
    Kc[156] = g_RT[16] + g_RT[31] - g_RT[39];
    Kc[157] = -g_RT[3] - g_RT[30] + g_RT[39];
    Kc[158] = g_RT[3] + g_RT[30] - g_RT[39];
    Kc[159] = -g_RT[6] - g_RT[39] + g_RT[40];
    Kc[160] = g_RT[6] + g_RT[39] - g_RT[40];
    Kc[161] = -2.000000*g_RT[28] + g_RT[41];
    Kc[162] = 2.000000*g_RT[28] - g_RT[41];
    Kc[163] = g_RT[4] - g_RT[14] - g_RT[38] + g_RT[41];
    Kc[164] = g_RT[4] - g_RT[21] - g_RT[22] + g_RT[41];
    Kc[165] = g_RT[1] - g_RT[22] - g_RT[31] + g_RT[41];
    Kc[166] = g_RT[3] - g_RT[28] - g_RT[31] + g_RT[41];
    Kc[167] = g_RT[1] - g_RT[14] - g_RT[37] + g_RT[41];
    Kc[168] = g_RT[3] + g_RT[42] - g_RT[43];
    Kc[169] = -g_RT[30] + g_RT[38] - g_RT[41] + g_RT[42];
    Kc[170] = -g_RT[20] + g_RT[21] - g_RT[41] + g_RT[42];
    Kc[171] = -g_RT[3] - g_RT[41] + g_RT[42];
    Kc[172] = g_RT[3] + g_RT[41] - g_RT[42];
    Kc[173] = g_RT[16] - g_RT[18] - g_RT[41] + g_RT[42];
    Kc[174] = -g_RT[6] + g_RT[7] + g_RT[42] - g_RT[43];
    Kc[175] = g_RT[6] - g_RT[7] - g_RT[41] + g_RT[42];
    Kc[176] = -g_RT[28] - g_RT[31] + g_RT[42];
    Kc[177] = -g_RT[2] + g_RT[3] - g_RT[41] + g_RT[42];
    Kc[178] = -g_RT[2] + g_RT[3] - g_RT[42] + g_RT[43];
    Kc[179] = g_RT[4] - g_RT[14] - g_RT[39] + g_RT[43];
    Kc[180] = g_RT[4] - g_RT[20] - g_RT[33] + g_RT[43];
    Kc[181] = g_RT[1] - g_RT[21] - g_RT[33] + g_RT[43];
    Kc[182] = g_RT[1] - g_RT[14] - g_RT[30] + g_RT[43];
    Kc[183] = g_RT[4] - g_RT[5] - g_RT[42] + g_RT[43];
    Kc[184] = -g_RT[16] - g_RT[38] + g_RT[43];
    Kc[185] = g_RT[16] + g_RT[38] - g_RT[43];
    Kc[186] = -g_RT[3] - g_RT[43] + g_RT[44];
    Kc[187] = g_RT[3] + g_RT[43] - g_RT[44];
    Kc[188] = -g_RT[21] - g_RT[31] + g_RT[44];
    Kc[189] = -g_RT[6] - g_RT[44] + g_RT[45];
    Kc[190] = g_RT[6] + g_RT[44] - g_RT[45];
    Kc[191] = -g_RT[16] - g_RT[41] + g_RT[46];
    Kc[192] = -g_RT[31] - g_RT[38] + g_RT[46];
    Kc[193] = g_RT[4] - g_RT[5] - g_RT[46] + g_RT[47];
    Kc[194] = -g_RT[2] + g_RT[3] - g_RT[46] + g_RT[47];
    Kc[195] = -g_RT[21] - g_RT[38] + g_RT[47];
    Kc[196] = g_RT[21] + g_RT[38] - g_RT[47];
    Kc[197] = g_RT[1] - g_RT[4] - g_RT[46] + g_RT[47];
    Kc[198] = -g_RT[21] - g_RT[30] + g_RT[48];
    Kc[199] = -g_RT[31] - g_RT[39] + g_RT[48];
    Kc[200] = -g_RT[3] - g_RT[47] + g_RT[48];
    Kc[201] = -g_RT[38] - g_RT[39] + g_RT[49];
    Kc[202] = g_RT[4] - g_RT[14] - g_RT[48] + g_RT[49];
    Kc[203] = -g_RT[16] - g_RT[49] + g_RT[50];
    Kc[204] = -g_RT[30] - g_RT[44] + g_RT[50];
    Kc[205] = -g_RT[39] - g_RT[43] + g_RT[50];
    Kc[206] = -g_RT[31] - g_RT[48] + g_RT[50];
    Kc[207] = -g_RT[21] - g_RT[47] + g_RT[50];
    Kc[208] = -g_RT[6] + g_RT[7] + g_RT[50] - g_RT[51];
    Kc[209] = g_RT[24] - g_RT[25] - g_RT[50] + g_RT[51];
    Kc[210] = -g_RT[2] + g_RT[3] - g_RT[50] + g_RT[51];
    Kc[211] = -g_RT[39] - g_RT[44] + g_RT[51];
    Kc[212] = g_RT[7] - g_RT[8] - g_RT[50] + g_RT[51];
    Kc[213] = -g_RT[21] - g_RT[48] + g_RT[51];
    Kc[214] = g_RT[17] - g_RT[19] - g_RT[50] + g_RT[51];
    Kc[215] = g_RT[1] - g_RT[4] - g_RT[50] + g_RT[51];
    Kc[216] = g_RT[4] - g_RT[5] - g_RT[50] + g_RT[51];
    Kc[217] = g_RT[16] - g_RT[18] - g_RT[50] + g_RT[51];

#ifdef __INTEL_COMPILER
     #pragma simd
#endif
    for (int i=0; i<218; ++i) {
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
    Kc[88] *= refC;
    Kc[90] *= refC;
    Kc[92] *= refC;
    Kc[97] *= refCinv;
    Kc[106] *= refCinv;
    Kc[113] *= refC;
    Kc[124] *= refC;
    Kc[125] *= refC;
    Kc[126] *= refCinv;
    Kc[127] *= refC;
    Kc[128] *= refCinv;
    Kc[129] *= refC;
    Kc[130] *= refC;
    Kc[142] *= refCinv;
    Kc[143] *= refC;
    Kc[144] *= refC;
    Kc[145] *= refCinv;
    Kc[152] *= refC;
    Kc[155] *= refC;
    Kc[156] *= refCinv;
    Kc[157] *= refC;
    Kc[158] *= refCinv;
    Kc[159] *= refC;
    Kc[160] *= refCinv;
    Kc[161] *= refC;
    Kc[162] *= refCinv;
    Kc[168] *= refCinv;
    Kc[171] *= refC;
    Kc[172] *= refCinv;
    Kc[176] *= refC;
    Kc[184] *= refC;
    Kc[185] *= refCinv;
    Kc[186] *= refC;
    Kc[187] *= refCinv;
    Kc[188] *= refC;
    Kc[189] *= refC;
    Kc[190] *= refCinv;
    Kc[191] *= refC;
    Kc[192] *= refC;
    Kc[195] *= refC;
    Kc[196] *= refCinv;
    Kc[198] *= refC;
    Kc[199] *= refC;
    Kc[200] *= refC;
    Kc[201] *= refC;
    Kc[203] *= refC;
    Kc[204] *= refC;
    Kc[205] *= refC;
    Kc[206] *= refC;
    Kc[207] *= refC;
    Kc[211] *= refC;
    Kc[213] *= refC;

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
    qf[2] = sc[4]*sc[16];
    qr[2] = 0.0;

    /*reaction 4: CH3 + H (+M) => CH4 (+M) */
    qf[3] = sc[3]*sc[16];
    qr[3] = 0.0;

    /*reaction 5: 2.000000 CH3 (+M) => C2H6 (+M) */
    qf[4] = pow(sc[16], 2.000000);
    qr[4] = 0.0;

    /*reaction 6: CO + CH2 (+M) => CH2CO (+M) */
    qf[5] = sc[11]*sc[13];
    qr[5] = 0.0;

    /*reaction 7: CO + O (+M) => CO2 (+M) */
    qf[6] = sc[1]*sc[13];
    qr[6] = 0.0;

    /*reaction 8: CH3O (+M) => CH2O + H (+M) */
    qf[7] = sc[17];
    qr[7] = 0.0;

    /*reaction 9: C2H3 (+M) => H + C2H2 (+M) */
    qf[8] = sc[28];
    qr[8] = 0.0;

    /*reaction 10: H + C2H4 (+M) <=> C2H5 (+M) */
    qf[9] = sc[3]*sc[31];
    qr[9] = sc[21];

    /*reaction 11: CH3CO (+M) => CH3 + CO (+M) */
    qf[10] = sc[33];
    qr[10] = 0.0;

    /*reaction 12: H + OH + M => H2O + M */
    qf[11] = sc[3]*sc[4];
    qr[11] = 0.0;

    /*reaction 13: CH2GSG + M => CH2 + M */
    qf[12] = sc[15];
    qr[12] = 0.0;

    /*reaction 14: CH2 + M => CH2GSG + M */
    qf[13] = sc[11];
    qr[13] = 0.0;

    /*reaction 15: HCO + M => H + CO + M */
    qf[14] = sc[10];
    qr[14] = 0.0;

    /*reaction 16: CH3O2 + M => CH3 + O2 + M */
    qf[15] = sc[24];
    qr[15] = 0.0;

    /*reaction 17: CH3 + O2 + M => CH3O2 + M */
    qf[16] = sc[6]*sc[16];
    qr[16] = 0.0;

    /*reaction 18: C2H5O + M => CH3 + CH2O + M */
    qf[17] = sc[32];
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
    qf[33] = sc[6]*sc[9];
    qr[33] = 0.0;

    /*reaction 35: CH2 + O2 => CO2 + 2.000000 H */
    qf[34] = sc[6]*sc[11];
    qr[34] = 0.0;

    /*reaction 36: CH2 + O2 => CO + H2O */
    qf[35] = sc[6]*sc[11];
    qr[35] = 0.0;

    /*reaction 37: CH2 + O => CO + 2.000000 H */
    qf[36] = sc[1]*sc[11];
    qr[36] = 0.0;

    /*reaction 38: CH2 + O2 => CH2O + O */
    qf[37] = sc[6]*sc[11];
    qr[37] = 0.0;

    /*reaction 39: CH2 + H => CH + H2 */
    qf[38] = sc[3]*sc[11];
    qr[38] = 0.0;

    /*reaction 40: CH + H2 => CH2 + H */
    qf[39] = sc[2]*sc[9];
    qr[39] = 0.0;

    /*reaction 41: CH2 + OH => CH + H2O */
    qf[40] = sc[4]*sc[11];
    qr[40] = 0.0;

    /*reaction 42: CH + H2O => CH2 + OH */
    qf[41] = sc[5]*sc[9];
    qr[41] = 0.0;

    /*reaction 43: CH2 + O2 => CO2 + H2 */
    qf[42] = sc[6]*sc[11];
    qr[42] = 0.0;

    /*reaction 44: CH2GSG + H => CH + H2 */
    qf[43] = sc[3]*sc[15];
    qr[43] = 0.0;

    /*reaction 45: CH2GSG + H2 => CH3 + H */
    qf[44] = sc[2]*sc[15];
    qr[44] = 0.0;

    /*reaction 46: CH3 + H => CH2GSG + H2 */
    qf[45] = sc[3]*sc[16];
    qr[45] = 0.0;

    /*reaction 47: CH2GSG + O2 => CO + OH + H */
    qf[46] = sc[6]*sc[15];
    qr[46] = 0.0;

    /*reaction 48: CH2GSG + OH => CH2O + H */
    qf[47] = sc[4]*sc[15];
    qr[47] = 0.0;

    /*reaction 49: CH3 + OH => CH2O + H2 */
    qf[48] = sc[4]*sc[16];
    qr[48] = 0.0;

    /*reaction 50: CH3 + OH => CH2GSG + H2O */
    qf[49] = sc[4]*sc[16];
    qr[49] = 0.0;

    /*reaction 51: CH2GSG + H2O => CH3 + OH */
    qf[50] = sc[5]*sc[15];
    qr[50] = 0.0;

    /*reaction 52: CH3 + O => CH2O + H */
    qf[51] = sc[1]*sc[16];
    qr[51] = 0.0;

    /*reaction 53: CH3 + HO2 => CH3O + OH */
    qf[52] = sc[7]*sc[16];
    qr[52] = 0.0;

    /*reaction 54: CH3 + HO2 => CH4 + O2 */
    qf[53] = sc[7]*sc[16];
    qr[53] = 0.0;

    /*reaction 55: CH3 + O2 => CH2O + OH */
    qf[54] = sc[6]*sc[16];
    qr[54] = 0.0;

    /*reaction 56: CH3 + H => CH2 + H2 */
    qf[55] = sc[3]*sc[16];
    qr[55] = 0.0;

    /*reaction 57: CH2 + H2 => CH3 + H */
    qf[56] = sc[2]*sc[11];
    qr[56] = 0.0;

    /*reaction 58: 2.000000 CH3 <=> H + C2H5 */
    qf[57] = pow(sc[16], 2.000000);
    qr[57] = sc[3]*sc[21];

    /*reaction 59: CH3 + OH => CH2 + H2O */
    qf[58] = sc[4]*sc[16];
    qr[58] = 0.0;

    /*reaction 60: CH2 + H2O => CH3 + OH */
    qf[59] = sc[5]*sc[11];
    qr[59] = 0.0;

    /*reaction 61: CH4 + O => CH3 + OH */
    qf[60] = sc[1]*sc[18];
    qr[60] = 0.0;

    /*reaction 62: CH4 + H => CH3 + H2 */
    qf[61] = sc[3]*sc[18];
    qr[61] = 0.0;

    /*reaction 63: CH3 + H2 => CH4 + H */
    qf[62] = sc[2]*sc[16];
    qr[62] = 0.0;

    /*reaction 64: CH4 + OH => CH3 + H2O */
    qf[63] = sc[4]*sc[18];
    qr[63] = 0.0;

    /*reaction 65: CH3 + H2O => CH4 + OH */
    qf[64] = sc[5]*sc[16];
    qr[64] = 0.0;

    /*reaction 66: CO + OH => CO2 + H */
    qf[65] = sc[4]*sc[13];
    qr[65] = 0.0;

    /*reaction 67: CO2 + H => CO + OH */
    qf[66] = sc[3]*sc[12];
    qr[66] = 0.0;

    /*reaction 68: HCO + O2 => CO + HO2 */
    qf[67] = sc[6]*sc[10];
    qr[67] = 0.0;

    /*reaction 69: HCO + O => CO2 + H */
    qf[68] = sc[1]*sc[10];
    qr[68] = 0.0;

    /*reaction 70: HCO + OH => CO + H2O */
    qf[69] = sc[4]*sc[10];
    qr[69] = 0.0;

    /*reaction 71: HCO + H => CO + H2 */
    qf[70] = sc[3]*sc[10];
    qr[70] = 0.0;

    /*reaction 72: HCO + O => CO + OH */
    qf[71] = sc[1]*sc[10];
    qr[71] = 0.0;

    /*reaction 73: HCO + CH3 => CH4 + CO */
    qf[72] = sc[10]*sc[16];
    qr[72] = 0.0;

    /*reaction 74: CH2O + OH => HCO + H2O */
    qf[73] = sc[4]*sc[14];
    qr[73] = 0.0;

    /*reaction 75: CH2O + O => HCO + OH */
    qf[74] = sc[1]*sc[14];
    qr[74] = 0.0;

    /*reaction 76: CH2O + H => HCO + H2 */
    qf[75] = sc[3]*sc[14];
    qr[75] = 0.0;

    /*reaction 77: CH2O + CH3 => HCO + CH4 */
    qf[76] = sc[14]*sc[16];
    qr[76] = 0.0;

    /*reaction 78: 2.000000 CH3O => CH3OH + CH2O */
    qf[77] = pow(sc[17], 2.000000);
    qr[77] = 0.0;

    /*reaction 79: CH3O + O2 => CH2O + HO2 */
    qf[78] = sc[6]*sc[17];
    qr[78] = 0.0;

    /*reaction 80: CH3O + H2 => CH3OH + H */
    qf[79] = sc[2]*sc[17];
    qr[79] = 0.0;

    /*reaction 81: CH3OH + OH => CH3O + H2O */
    qf[80] = sc[4]*sc[19];
    qr[80] = 0.0;

    /*reaction 82: CH2GSG + CO2 => CH2O + CO */
    qf[81] = sc[12]*sc[15];
    qr[81] = 0.0;

    /*reaction 83: HOCHO + H => H2 + CO + OH */
    qf[82] = sc[3]*sc[23];
    qr[82] = 0.0;

    /*reaction 84: HOCHO + OH => H2O + CO + OH */
    qf[83] = sc[4]*sc[23];
    qr[83] = 0.0;

    /*reaction 85: HOCHO => HCO + OH */
    qf[84] = sc[23];
    qr[84] = 0.0;

    /*reaction 86: HCO + OH => HOCHO */
    qf[85] = sc[4]*sc[10];
    qr[85] = 0.0;

    /*reaction 87: HOCHO + H => H2 + CO2 + H */
    qf[86] = sc[3]*sc[23];
    qr[86] = 0.0;

    /*reaction 88: HOCHO + OH => H2O + CO2 + H */
    qf[87] = sc[4]*sc[23];
    qr[87] = 0.0;

    /*reaction 89: 2.000000 CH3O2 => O2 + 2.000000 CH3O */
    qf[88] = pow(sc[24], 2.000000);
    qr[88] = 0.0;

    /*reaction 90: CH3O2 + CH3 => 2.000000 CH3O */
    qf[89] = sc[16]*sc[24];
    qr[89] = 0.0;

    /*reaction 91: 2.000000 CH3O2 => CH2O + CH3OH + O2 */
    qf[90] = pow(sc[24], 2.000000);
    qr[90] = 0.0;

    /*reaction 92: CH3O2 + HO2 => CH3O2H + O2 */
    qf[91] = sc[7]*sc[24];
    qr[91] = 0.0;

    /*reaction 93: CH3O2H => CH3O + OH */
    qf[92] = sc[25];
    qr[92] = 0.0;

    /*reaction 94: C2H2 + O => CH2 + CO */
    qf[93] = sc[1]*sc[26];
    qr[93] = 0.0;

    /*reaction 95: C2H2 + O => HCCO + H */
    qf[94] = sc[1]*sc[26];
    qr[94] = 0.0;

    /*reaction 96: C2H3 + H => C2H2 + H2 */
    qf[95] = sc[3]*sc[28];
    qr[95] = 0.0;

    /*reaction 97: C2H3 + O2 => CH2CHO + O */
    qf[96] = sc[6]*sc[28];
    qr[96] = 0.0;

    /*reaction 98: C2H3 + CH3 => C3H6 */
    qf[97] = sc[16]*sc[28];
    qr[97] = 0.0;

    /*reaction 99: C2H3 + O2 => C2H2 + HO2 */
    qf[98] = sc[6]*sc[28];
    qr[98] = 0.0;

    /*reaction 100: C2H3 + O2 => CH2O + HCO */
    qf[99] = sc[6]*sc[28];
    qr[99] = 0.0;

    /*reaction 101: C2H4 + CH3 => C2H3 + CH4 */
    qf[100] = sc[16]*sc[31];
    qr[100] = 0.0;

    /*reaction 102: C2H4 + O => CH3 + HCO */
    qf[101] = sc[1]*sc[31];
    qr[101] = 0.0;

    /*reaction 103: C2H4 + OH => C2H3 + H2O */
    qf[102] = sc[4]*sc[31];
    qr[102] = 0.0;

    /*reaction 104: C2H4 + O => CH2CHO + H */
    qf[103] = sc[1]*sc[31];
    qr[103] = 0.0;

    /*reaction 105: C2H4 + H => C2H3 + H2 */
    qf[104] = sc[3]*sc[31];
    qr[104] = 0.0;

    /*reaction 106: C2H3 + H2 => C2H4 + H */
    qf[105] = sc[2]*sc[28];
    qr[105] = 0.0;

    /*reaction 107: H + C2H5 => C2H6 */
    qf[106] = sc[3]*sc[21];
    qr[106] = 0.0;

    /*reaction 108: CH3O2 + C2H5 => CH3O + C2H5O */
    qf[107] = sc[21]*sc[24];
    qr[107] = 0.0;

    /*reaction 109: C2H5 + HO2 => C2H5O + OH */
    qf[108] = sc[7]*sc[21];
    qr[108] = 0.0;

    /*reaction 110: C2H5 + O2 => C2H4 + HO2 */
    qf[109] = sc[6]*sc[21];
    qr[109] = 0.0;

    /*reaction 111: C2H6 + O => C2H5 + OH */
    qf[110] = sc[1]*sc[20];
    qr[110] = 0.0;

    /*reaction 112: C2H6 + OH => C2H5 + H2O */
    qf[111] = sc[4]*sc[20];
    qr[111] = 0.0;

    /*reaction 113: C2H6 + H => C2H5 + H2 */
    qf[112] = sc[3]*sc[20];
    qr[112] = 0.0;

    /*reaction 114: HCCO + O => H + 2.000000 CO */
    qf[113] = sc[1]*sc[27];
    qr[113] = 0.0;

    /*reaction 115: HCCO + OH => 2.000000 HCO */
    qf[114] = sc[4]*sc[27];
    qr[114] = 0.0;

    /*reaction 116: HCCO + O2 => CO2 + HCO */
    qf[115] = sc[6]*sc[27];
    qr[115] = 0.0;

    /*reaction 117: HCCO + H => CH2GSG + CO */
    qf[116] = sc[3]*sc[27];
    qr[116] = 0.0;

    /*reaction 118: CH2GSG + CO => HCCO + H */
    qf[117] = sc[13]*sc[15];
    qr[117] = 0.0;

    /*reaction 119: CH2CO + O => HCCO + OH */
    qf[118] = sc[1]*sc[22];
    qr[118] = 0.0;

    /*reaction 120: CH2CO + H => HCCO + H2 */
    qf[119] = sc[3]*sc[22];
    qr[119] = 0.0;

    /*reaction 121: HCCO + H2 => CH2CO + H */
    qf[120] = sc[2]*sc[27];
    qr[120] = 0.0;

    /*reaction 122: CH2CO + H => CH3 + CO */
    qf[121] = sc[3]*sc[22];
    qr[121] = 0.0;

    /*reaction 123: CH2CO + O => CH2 + CO2 */
    qf[122] = sc[1]*sc[22];
    qr[122] = 0.0;

    /*reaction 124: CH2CO + OH => HCCO + H2O */
    qf[123] = sc[4]*sc[22];
    qr[123] = 0.0;

    /*reaction 125: CH2CHO + O2 => CH2O + CO + OH */
    qf[124] = sc[6]*sc[29];
    qr[124] = 0.0;

    /*reaction 126: CH2CHO => CH2CO + H */
    qf[125] = sc[29];
    qr[125] = 0.0;

    /*reaction 127: CH2CO + H => CH2CHO */
    qf[126] = sc[3]*sc[22];
    qr[126] = 0.0;

    /*reaction 128: C2H5O2 => C2H5 + O2 */
    qf[127] = sc[34];
    qr[127] = 0.0;

    /*reaction 129: C2H5 + O2 => C2H5O2 */
    qf[128] = sc[6]*sc[21];
    qr[128] = 0.0;

    /*reaction 130: C2H5O2 => C2H4 + HO2 */
    qf[129] = sc[34];
    qr[129] = 0.0;

    /*reaction 131: C3H2 + O2 => HCCO + CO + H */
    qf[130] = sc[6]*sc[35];
    qr[130] = 0.0;

    /*reaction 132: C3H2 + OH => C2H2 + HCO */
    qf[131] = sc[4]*sc[35];
    qr[131] = 0.0;

    /*reaction 133: C3H3 + O2 => CH2CO + HCO */
    qf[132] = sc[6]*sc[36];
    qr[132] = 0.0;

    /*reaction 134: C3H3 + HO2 => C3H4XA + O2 */
    qf[133] = sc[7]*sc[36];
    qr[133] = 0.0;

    /*reaction 135: C3H3 + H => C3H2 + H2 */
    qf[134] = sc[3]*sc[36];
    qr[134] = 0.0;

    /*reaction 136: C3H3 + OH => C3H2 + H2O */
    qf[135] = sc[4]*sc[36];
    qr[135] = 0.0;

    /*reaction 137: C3H2 + H2O => C3H3 + OH */
    qf[136] = sc[5]*sc[35];
    qr[136] = 0.0;

    /*reaction 138: C3H4XA + H => C3H3 + H2 */
    qf[137] = sc[3]*sc[37];
    qr[137] = 0.0;

    /*reaction 139: C3H4XA + OH => C3H3 + H2O */
    qf[138] = sc[4]*sc[37];
    qr[138] = 0.0;

    /*reaction 140: C3H4XA + O => C2H4 + CO */
    qf[139] = sc[1]*sc[37];
    qr[139] = 0.0;

    /*reaction 141: C3H5XA + H => C3H4XA + H2 */
    qf[140] = sc[3]*sc[38];
    qr[140] = 0.0;

    /*reaction 142: C3H5XA + HO2 => C3H6 + O2 */
    qf[141] = sc[7]*sc[38];
    qr[141] = 0.0;

    /*reaction 143: C3H5XA + H => C3H6 */
    qf[142] = sc[3]*sc[38];
    qr[142] = 0.0;

    /*reaction 144: C3H5XA => C2H2 + CH3 */
    qf[143] = sc[38];
    qr[143] = 0.0;

    /*reaction 145: C3H5XA => C3H4XA + H */
    qf[144] = sc[38];
    qr[144] = 0.0;

    /*reaction 146: C3H4XA + H => C3H5XA */
    qf[145] = sc[3]*sc[37];
    qr[145] = 0.0;

    /*reaction 147: C3H5XA + CH2O => C3H6 + HCO */
    qf[146] = sc[14]*sc[38];
    qr[146] = 0.0;

    /*reaction 148: 2.000000 C3H5XA => C3H4XA + C3H6 */
    qf[147] = pow(sc[38], 2.000000);
    qr[147] = 0.0;

    /*reaction 149: C3H6 + H => C2H4 + CH3 */
    qf[148] = sc[3]*sc[30];
    qr[148] = 0.0;

    /*reaction 150: C3H6 + H => C3H5XA + H2 */
    qf[149] = sc[3]*sc[30];
    qr[149] = 0.0;

    /*reaction 151: C3H6 + O => C2H5 + HCO */
    qf[150] = sc[1]*sc[30];
    qr[150] = 0.0;

    /*reaction 152: C3H6 + O => C3H5XA + OH */
    qf[151] = sc[1]*sc[30];
    qr[151] = 0.0;

    /*reaction 153: C3H6 + O => CH2CO + CH3 + H */
    qf[152] = sc[1]*sc[30];
    qr[152] = 0.0;

    /*reaction 154: C3H6 + OH => C3H5XA + H2O */
    qf[153] = sc[4]*sc[30];
    qr[153] = 0.0;

    /*reaction 155: NXC3H7 + O2 => C3H6 + HO2 */
    qf[154] = sc[6]*sc[39];
    qr[154] = 0.0;

    /*reaction 156: NXC3H7 => CH3 + C2H4 */
    qf[155] = sc[39];
    qr[155] = 0.0;

    /*reaction 157: CH3 + C2H4 => NXC3H7 */
    qf[156] = sc[16]*sc[31];
    qr[156] = 0.0;

    /*reaction 158: NXC3H7 => H + C3H6 */
    qf[157] = sc[39];
    qr[157] = 0.0;

    /*reaction 159: H + C3H6 => NXC3H7 */
    qf[158] = sc[3]*sc[30];
    qr[158] = 0.0;

    /*reaction 160: NXC3H7O2 => NXC3H7 + O2 */
    qf[159] = sc[40];
    qr[159] = 0.0;

    /*reaction 161: NXC3H7 + O2 => NXC3H7O2 */
    qf[160] = sc[6]*sc[39];
    qr[160] = 0.0;

    /*reaction 162: C4H6 => 2.000000 C2H3 */
    qf[161] = sc[41];
    qr[161] = 0.0;

    /*reaction 163: 2.000000 C2H3 => C4H6 */
    qf[162] = pow(sc[28], 2.000000);
    qr[162] = 0.0;

    /*reaction 164: C4H6 + OH => CH2O + C3H5XA */
    qf[163] = sc[4]*sc[41];
    qr[163] = 0.0;

    /*reaction 165: C4H6 + OH => C2H5 + CH2CO */
    qf[164] = sc[4]*sc[41];
    qr[164] = 0.0;

    /*reaction 166: C4H6 + O => C2H4 + CH2CO */
    qf[165] = sc[1]*sc[41];
    qr[165] = 0.0;

    /*reaction 167: C4H6 + H => C2H3 + C2H4 */
    qf[166] = sc[3]*sc[41];
    qr[166] = 0.0;

    /*reaction 168: C4H6 + O => CH2O + C3H4XA */
    qf[167] = sc[1]*sc[41];
    qr[167] = 0.0;

    /*reaction 169: H + C4H7 => C4H8X1 */
    qf[168] = sc[3]*sc[42];
    qr[168] = 0.0;

    /*reaction 170: C3H5XA + C4H7 => C3H6 + C4H6 */
    qf[169] = sc[38]*sc[42];
    qr[169] = 0.0;

    /*reaction 171: C2H5 + C4H7 => C4H6 + C2H6 */
    qf[170] = sc[21]*sc[42];
    qr[170] = 0.0;

    /*reaction 172: C4H7 => C4H6 + H */
    qf[171] = sc[42];
    qr[171] = 0.0;

    /*reaction 173: C4H6 + H => C4H7 */
    qf[172] = sc[3]*sc[41];
    qr[172] = 0.0;

    /*reaction 174: C4H7 + CH3 => C4H6 + CH4 */
    qf[173] = sc[16]*sc[42];
    qr[173] = 0.0;

    /*reaction 175: C4H7 + HO2 => C4H8X1 + O2 */
    qf[174] = sc[7]*sc[42];
    qr[174] = 0.0;

    /*reaction 176: C4H7 + O2 => C4H6 + HO2 */
    qf[175] = sc[6]*sc[42];
    qr[175] = 0.0;

    /*reaction 177: C4H7 => C2H4 + C2H3 */
    qf[176] = sc[42];
    qr[176] = 0.0;

    /*reaction 178: H + C4H7 => C4H6 + H2 */
    qf[177] = sc[3]*sc[42];
    qr[177] = 0.0;

    /*reaction 179: C4H8X1 + H => C4H7 + H2 */
    qf[178] = sc[3]*sc[43];
    qr[178] = 0.0;

    /*reaction 180: C4H8X1 + OH => NXC3H7 + CH2O */
    qf[179] = sc[4]*sc[43];
    qr[179] = 0.0;

    /*reaction 181: C4H8X1 + OH => CH3CO + C2H6 */
    qf[180] = sc[4]*sc[43];
    qr[180] = 0.0;

    /*reaction 182: C4H8X1 + O => CH3CO + C2H5 */
    qf[181] = sc[1]*sc[43];
    qr[181] = 0.0;

    /*reaction 183: C4H8X1 + O => C3H6 + CH2O */
    qf[182] = sc[1]*sc[43];
    qr[182] = 0.0;

    /*reaction 184: C4H8X1 + OH => C4H7 + H2O */
    qf[183] = sc[4]*sc[43];
    qr[183] = 0.0;

    /*reaction 185: C4H8X1 => C3H5XA + CH3 */
    qf[184] = sc[43];
    qr[184] = 0.0;

    /*reaction 186: C3H5XA + CH3 => C4H8X1 */
    qf[185] = sc[16]*sc[38];
    qr[185] = 0.0;

    /*reaction 187: PXC4H9 => C4H8X1 + H */
    qf[186] = sc[44];
    qr[186] = 0.0;

    /*reaction 188: C4H8X1 + H => PXC4H9 */
    qf[187] = sc[3]*sc[43];
    qr[187] = 0.0;

    /*reaction 189: PXC4H9 => C2H5 + C2H4 */
    qf[188] = sc[44];
    qr[188] = 0.0;

    /*reaction 190: PXC4H9O2 => PXC4H9 + O2 */
    qf[189] = sc[45];
    qr[189] = 0.0;

    /*reaction 191: PXC4H9 + O2 => PXC4H9O2 */
    qf[190] = sc[6]*sc[44];
    qr[190] = 0.0;

    /*reaction 192: C5H9 => C4H6 + CH3 */
    qf[191] = sc[46];
    qr[191] = 0.0;

    /*reaction 193: C5H9 => C3H5XA + C2H4 */
    qf[192] = sc[46];
    qr[192] = 0.0;

    /*reaction 194: C5H10X1 + OH => C5H9 + H2O */
    qf[193] = sc[4]*sc[47];
    qr[193] = 0.0;

    /*reaction 195: C5H10X1 + H => C5H9 + H2 */
    qf[194] = sc[3]*sc[47];
    qr[194] = 0.0;

    /*reaction 196: C5H10X1 => C2H5 + C3H5XA */
    qf[195] = sc[47];
    qr[195] = 0.0;

    /*reaction 197: C2H5 + C3H5XA => C5H10X1 */
    qf[196] = sc[21]*sc[38];
    qr[196] = 0.0;

    /*reaction 198: C5H10X1 + O => C5H9 + OH */
    qf[197] = sc[1]*sc[47];
    qr[197] = 0.0;

    /*reaction 199: C5H11X1 => C3H6 + C2H5 */
    qf[198] = sc[48];
    qr[198] = 0.0;

    /*reaction 200: C5H11X1 => C2H4 + NXC3H7 */
    qf[199] = sc[48];
    qr[199] = 0.0;

    /*reaction 201: C5H11X1 <=> C5H10X1 + H */
    qf[200] = sc[48];
    qr[200] = sc[3]*sc[47];

    /*reaction 202: C6H12X1 => NXC3H7 + C3H5XA */
    qf[201] = sc[49];
    qr[201] = 0.0;

    /*reaction 203: C6H12X1 + OH => C5H11X1 + CH2O */
    qf[202] = sc[4]*sc[49];
    qr[202] = 0.0;

    /*reaction 204: C7H15X2 => C6H12X1 + CH3 */
    qf[203] = sc[50];
    qr[203] = 0.0;

    /*reaction 205: C7H15X2 => PXC4H9 + C3H6 */
    qf[204] = sc[50];
    qr[204] = 0.0;

    /*reaction 206: C7H15X2 => C4H8X1 + NXC3H7 */
    qf[205] = sc[50];
    qr[205] = 0.0;

    /*reaction 207: C7H15X2 => C5H11X1 + C2H4 */
    qf[206] = sc[50];
    qr[206] = 0.0;

    /*reaction 208: C7H15X2 => C2H5 + C5H10X1 */
    qf[207] = sc[50];
    qr[207] = 0.0;

    /*reaction 209: C7H15X2 + HO2 => NXC7H16 + O2 */
    qf[208] = sc[7]*sc[50];
    qr[208] = 0.0;

    /*reaction 210: NXC7H16 + CH3O2 => C7H15X2 + CH3O2H */
    qf[209] = sc[24]*sc[51];
    qr[209] = 0.0;

    /*reaction 211: NXC7H16 + H => C7H15X2 + H2 */
    qf[210] = sc[3]*sc[51];
    qr[210] = 0.0;

    /*reaction 212: NXC7H16 => PXC4H9 + NXC3H7 */
    qf[211] = sc[51];
    qr[211] = 0.0;

    /*reaction 213: NXC7H16 + HO2 => C7H15X2 + H2O2 */
    qf[212] = sc[7]*sc[51];
    qr[212] = 0.0;

    /*reaction 214: NXC7H16 => C5H11X1 + C2H5 */
    qf[213] = sc[51];
    qr[213] = 0.0;

    /*reaction 215: NXC7H16 + CH3O => C7H15X2 + CH3OH */
    qf[214] = sc[17]*sc[51];
    qr[214] = 0.0;

    /*reaction 216: NXC7H16 + O => C7H15X2 + OH */
    qf[215] = sc[1]*sc[51];
    qr[215] = 0.0;

    /*reaction 217: NXC7H16 + OH => C7H15X2 + H2O */
    qf[216] = sc[4]*sc[51];
    qr[216] = 0.0;

    /*reaction 218: NXC7H16 + CH3 => C7H15X2 + CH4 */
    qf[217] = sc[16]*sc[51];
    qr[217] = 0.0;

    double T = tc[1];

    /*compute the mixture concentration */
    double mixture = 0.0;
    for (int i = 0; i < 52; ++i) {
        mixture += sc[i];
    }

    double Corr[218];
    for (int i = 0; i < 218; ++i) {
        Corr[i] = 1.0;
    }

    /* troe */
    {
        double alpha[11];
        alpha[0] = mixture + (TB[0][0] - 1)*sc[29] + (TB[0][1] - 1)*sc[9] + (TB[0][2] - 1)*sc[13] + (TB[0][3] - 1)*sc[2] + (TB[0][4] - 1)*sc[44] + (TB[0][5] - 1)*sc[15] + (TB[0][6] - 1)*sc[42] + (TB[0][7] - 1)*sc[27] + (TB[0][8] - 1)*sc[48] + (TB[0][9] - 1)*sc[5] + (TB[0][10] - 1)*sc[12] + (TB[0][11] - 1)*sc[32] + (TB[0][12] - 1)*sc[35] + (TB[0][13] - 1)*sc[33] + (TB[0][14] - 1)*sc[50];
        alpha[1] = mixture + (TB[1][0] - 1)*sc[29] + (TB[1][1] - 1)*sc[9] + (TB[1][2] - 1)*sc[13] + (TB[1][3] - 1)*sc[2] + (TB[1][4] - 1)*sc[44] + (TB[1][5] - 1)*sc[15] + (TB[1][6] - 1)*sc[42] + (TB[1][7] - 1)*sc[27] + (TB[1][8] - 1)*sc[48] + (TB[1][9] - 1)*sc[5] + (TB[1][10] - 1)*sc[12] + (TB[1][11] - 1)*sc[32] + (TB[1][12] - 1)*sc[35] + (TB[1][13] - 1)*sc[33] + (TB[1][14] - 1)*sc[50];
        alpha[2] = mixture + (TB[2][0] - 1)*sc[29] + (TB[2][1] - 1)*sc[9] + (TB[2][2] - 1)*sc[13] + (TB[2][3] - 1)*sc[2] + (TB[2][4] - 1)*sc[44] + (TB[2][5] - 1)*sc[15] + (TB[2][6] - 1)*sc[42] + (TB[2][7] - 1)*sc[27] + (TB[2][8] - 1)*sc[48] + (TB[2][9] - 1)*sc[5] + (TB[2][10] - 1)*sc[12] + (TB[2][11] - 1)*sc[18] + (TB[2][12] - 1)*sc[20] + (TB[2][13] - 1)*sc[32] + (TB[2][14] - 1)*sc[35] + (TB[2][15] - 1)*sc[33] + (TB[2][16] - 1)*sc[50];
        alpha[3] = mixture + (TB[3][0] - 1)*sc[29] + (TB[3][1] - 1)*sc[9] + (TB[3][2] - 1)*sc[13] + (TB[3][3] - 1)*sc[2] + (TB[3][4] - 1)*sc[44] + (TB[3][5] - 1)*sc[15] + (TB[3][6] - 1)*sc[42] + (TB[3][7] - 1)*sc[27] + (TB[3][8] - 1)*sc[48] + (TB[3][9] - 1)*sc[5] + (TB[3][10] - 1)*sc[12] + (TB[3][11] - 1)*sc[32] + (TB[3][12] - 1)*sc[35] + (TB[3][13] - 1)*sc[33] + (TB[3][14] - 1)*sc[50];
        alpha[4] = mixture + (TB[4][0] - 1)*sc[29] + (TB[4][1] - 1)*sc[9] + (TB[4][2] - 1)*sc[13] + (TB[4][3] - 1)*sc[2] + (TB[4][4] - 1)*sc[44] + (TB[4][5] - 1)*sc[15] + (TB[4][6] - 1)*sc[42] + (TB[4][7] - 1)*sc[27] + (TB[4][8] - 1)*sc[48] + (TB[4][9] - 1)*sc[5] + (TB[4][10] - 1)*sc[12] + (TB[4][11] - 1)*sc[32] + (TB[4][12] - 1)*sc[35] + (TB[4][13] - 1)*sc[33] + (TB[4][14] - 1)*sc[50];
        alpha[5] = mixture + (TB[5][0] - 1)*sc[29] + (TB[5][1] - 1)*sc[9] + (TB[5][2] - 1)*sc[35] + (TB[5][3] - 1)*sc[44] + (TB[5][4] - 1)*sc[15] + (TB[5][5] - 1)*sc[42] + (TB[5][6] - 1)*sc[27] + (TB[5][7] - 1)*sc[48] + (TB[5][8] - 1)*sc[33] + (TB[5][9] - 1)*sc[32] + (TB[5][10] - 1)*sc[50];
        alpha[6] = mixture + (TB[6][0] - 1)*sc[29] + (TB[6][1] - 1)*sc[9] + (TB[6][2] - 1)*sc[13] + (TB[6][3] - 1)*sc[2] + (TB[6][4] - 1)*sc[44] + (TB[6][5] - 1)*sc[15] + (TB[6][6] - 1)*sc[42] + (TB[6][7] - 1)*sc[27] + (TB[6][8] - 1)*sc[48] + (TB[6][9] - 1)*sc[5] + (TB[6][10] - 1)*sc[12] + (TB[6][11] - 1)*sc[32] + (TB[6][12] - 1)*sc[35] + (TB[6][13] - 1)*sc[33] + (TB[6][14] - 1)*sc[50];
        alpha[7] = mixture + (TB[7][0] - 1)*sc[29] + (TB[7][1] - 1)*sc[9] + (TB[7][2] - 1)*sc[35] + (TB[7][3] - 1)*sc[44] + (TB[7][4] - 1)*sc[15] + (TB[7][5] - 1)*sc[42] + (TB[7][6] - 1)*sc[27] + (TB[7][7] - 1)*sc[48] + (TB[7][8] - 1)*sc[33] + (TB[7][9] - 1)*sc[32] + (TB[7][10] - 1)*sc[50];
        alpha[8] = mixture + (TB[8][0] - 1)*sc[29] + (TB[8][1] - 1)*sc[9] + (TB[8][2] - 1)*sc[13] + (TB[8][3] - 1)*sc[2] + (TB[8][4] - 1)*sc[44] + (TB[8][5] - 1)*sc[15] + (TB[8][6] - 1)*sc[42] + (TB[8][7] - 1)*sc[27] + (TB[8][8] - 1)*sc[48] + (TB[8][9] - 1)*sc[5] + (TB[8][10] - 1)*sc[12] + (TB[8][11] - 1)*sc[32] + (TB[8][12] - 1)*sc[35] + (TB[8][13] - 1)*sc[33] + (TB[8][14] - 1)*sc[50];
        alpha[9] = mixture + (TB[9][0] - 1)*sc[29] + (TB[9][1] - 1)*sc[9] + (TB[9][2] - 1)*sc[35] + (TB[9][3] - 1)*sc[44] + (TB[9][4] - 1)*sc[15] + (TB[9][5] - 1)*sc[42] + (TB[9][6] - 1)*sc[27] + (TB[9][7] - 1)*sc[48] + (TB[9][8] - 1)*sc[33] + (TB[9][9] - 1)*sc[32] + (TB[9][10] - 1)*sc[50];
        alpha[10] = mixture + (TB[10][0] - 1)*sc[29] + (TB[10][1] - 1)*sc[9] + (TB[10][2] - 1)*sc[35] + (TB[10][3] - 1)*sc[44] + (TB[10][4] - 1)*sc[15] + (TB[10][5] - 1)*sc[42] + (TB[10][6] - 1)*sc[27] + (TB[10][7] - 1)*sc[48] + (TB[10][8] - 1)*sc[33] + (TB[10][9] - 1)*sc[32] + (TB[10][10] - 1)*sc[50];
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
        alpha = mixture + (TB[11][0] - 1)*sc[29] + (TB[11][1] - 1)*sc[9] + (TB[11][2] - 1)*sc[13] + (TB[11][3] - 1)*sc[2] + (TB[11][4] - 1)*sc[44] + (TB[11][5] - 1)*sc[15] + (TB[11][6] - 1)*sc[42] + (TB[11][7] - 1)*sc[27] + (TB[11][8] - 1)*sc[48] + (TB[11][9] - 1)*sc[5] + (TB[11][10] - 1)*sc[12] + (TB[11][11] - 1)*sc[32] + (TB[11][12] - 1)*sc[35] + (TB[11][13] - 1)*sc[33] + (TB[11][14] - 1)*sc[50];
        Corr[11] = alpha;
        alpha = mixture + (TB[12][0] - 1)*sc[29] + (TB[12][1] - 1)*sc[9] + (TB[12][2] - 1)*sc[35] + (TB[12][3] - 1)*sc[44] + (TB[12][4] - 1)*sc[15] + (TB[12][5] - 1)*sc[42] + (TB[12][6] - 1)*sc[27] + (TB[12][7] - 1)*sc[48] + (TB[12][8] - 1)*sc[33] + (TB[12][9] - 1)*sc[32] + (TB[12][10] - 1)*sc[50];
        Corr[12] = alpha;
        alpha = mixture + (TB[13][0] - 1)*sc[29] + (TB[13][1] - 1)*sc[9] + (TB[13][2] - 1)*sc[35] + (TB[13][3] - 1)*sc[44] + (TB[13][4] - 1)*sc[15] + (TB[13][5] - 1)*sc[42] + (TB[13][6] - 1)*sc[27] + (TB[13][7] - 1)*sc[48] + (TB[13][8] - 1)*sc[33] + (TB[13][9] - 1)*sc[32] + (TB[13][10] - 1)*sc[50];
        Corr[13] = alpha;
        alpha = mixture + (TB[14][0] - 1)*sc[29] + (TB[14][1] - 1)*sc[9] + (TB[14][2] - 1)*sc[13] + (TB[14][3] - 1)*sc[2] + (TB[14][4] - 1)*sc[44] + (TB[14][5] - 1)*sc[15] + (TB[14][6] - 1)*sc[42] + (TB[14][7] - 1)*sc[27] + (TB[14][8] - 1)*sc[48] + (TB[14][9] - 1)*sc[5] + (TB[14][10] - 1)*sc[12] + (TB[14][11] - 1)*sc[32] + (TB[14][12] - 1)*sc[35] + (TB[14][13] - 1)*sc[33] + (TB[14][14] - 1)*sc[50];
        Corr[14] = alpha;
        alpha = mixture + (TB[15][0] - 1)*sc[29] + (TB[15][1] - 1)*sc[9] + (TB[15][2] - 1)*sc[35] + (TB[15][3] - 1)*sc[44] + (TB[15][4] - 1)*sc[15] + (TB[15][5] - 1)*sc[42] + (TB[15][6] - 1)*sc[27] + (TB[15][7] - 1)*sc[48] + (TB[15][8] - 1)*sc[33] + (TB[15][9] - 1)*sc[32] + (TB[15][10] - 1)*sc[50];
        Corr[15] = alpha;
        alpha = mixture + (TB[16][0] - 1)*sc[29] + (TB[16][1] - 1)*sc[9] + (TB[16][2] - 1)*sc[35] + (TB[16][3] - 1)*sc[44] + (TB[16][4] - 1)*sc[15] + (TB[16][5] - 1)*sc[42] + (TB[16][6] - 1)*sc[27] + (TB[16][7] - 1)*sc[48] + (TB[16][8] - 1)*sc[33] + (TB[16][9] - 1)*sc[32] + (TB[16][10] - 1)*sc[50];
        Corr[16] = alpha;
        alpha = mixture + (TB[17][0] - 1)*sc[29] + (TB[17][1] - 1)*sc[9] + (TB[17][2] - 1)*sc[35] + (TB[17][3] - 1)*sc[44] + (TB[17][4] - 1)*sc[15] + (TB[17][5] - 1)*sc[42] + (TB[17][6] - 1)*sc[27] + (TB[17][7] - 1)*sc[48] + (TB[17][8] - 1)*sc[33] + (TB[17][9] - 1)*sc[32] + (TB[17][10] - 1)*sc[50];
        Corr[17] = alpha;
    }

    for (int i=0; i<218; i++)
    {
        qf[i] *= Corr[i] * k_f_save[i];
        qr[i] *= Corr[i] * k_f_save[i] / Kc_save[i];
    }

    return;
}
#endif

/*compute an approx to the reaction Jacobian (for preconditioning) */
AMREX_GPU_HOST_DEVICE void DWDOT_SIMPLIFIED(double *  J, double *  sc, double *  Tp, int * HP)
{
    double c[52];

    for (int k=0; k<52; k++) {
        c[k] = 1.e6 * sc[k];
    }

    aJacobian_precond(J, c, *Tp, *HP);

    /* dwdot[k]/dT */
    /* dTdot/d[X] */
    for (int k=0; k<52; k++) {
        J[2756+k] *= 1.e-6;
        J[k*53+52] *= 1.e6;
    }

    return;
}

/*compute the reaction Jacobian */
AMREX_GPU_HOST_DEVICE void DWDOT(double *  J, double *  sc, double *  Tp, int * consP)
{
    double c[52];

    for (int k=0; k<52; k++) {
        c[k] = 1.e6 * sc[k];
    }

    aJacobian(J, c, *Tp, *consP);

    /* dwdot[k]/dT */
    /* dTdot/d[X] */
    for (int k=0; k<52; k++) {
        J[2756+k] *= 1.e-6;
        J[k*53+52] *= 1.e6;
    }

    return;
}

/*compute the sparsity pattern of the chemistry Jacobian */
AMREX_GPU_HOST_DEVICE void SPARSITY_INFO( int * nJdata, int * consP, int NCELLS)
{
    double c[52];
    double J[2809];

    for (int k=0; k<52; k++) {
        c[k] = 1.0/ 52.000000 ;
    }

    aJacobian(J, c, 1500.0, *consP);

    int nJdata_tmp = 0;
    for (int k=0; k<53; k++) {
        for (int l=0; l<53; l++) {
            if(J[ 53 * k + l] != 0.0){
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
    double c[52];
    double J[2809];

    for (int k=0; k<52; k++) {
        c[k] = 1.0/ 52.000000 ;
    }

    aJacobian(J, c, 1500.0, *consP);

    int nJdata_tmp = 0;
    for (int k=0; k<53; k++) {
        for (int l=0; l<53; l++) {
            if(k == l){
                nJdata_tmp = nJdata_tmp + 1;
            } else {
                if(J[ 53 * k + l] != 0.0){
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
    double c[52];
    double J[2809];

    for (int k=0; k<52; k++) {
        c[k] = 1.0/ 52.000000 ;
    }

    aJacobian_precond(J, c, 1500.0, *consP);

    int nJdata_tmp = 0;
    for (int k=0; k<53; k++) {
        for (int l=0; l<53; l++) {
            if(k == l){
                nJdata_tmp = nJdata_tmp + 1;
            } else {
                if(J[ 53 * k + l] != 0.0){
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
    double c[52];
    double J[2809];
    int offset_row;
    int offset_col;

    for (int k=0; k<52; k++) {
        c[k] = 1.0/ 52.000000 ;
    }

    aJacobian(J, c, 1500.0, *consP);

    colPtrs[0] = 0;
    int nJdata_tmp = 0;
    for (int nc=0; nc<NCELLS; nc++) {
        offset_row = nc * 53;
        offset_col = nc * 53;
        for (int k=0; k<53; k++) {
            for (int l=0; l<53; l++) {
                if(J[53*k + l] != 0.0) {
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
    double c[52];
    double J[2809];
    int offset;

    for (int k=0; k<52; k++) {
        c[k] = 1.0/ 52.000000 ;
    }

    aJacobian(J, c, 1500.0, *consP);

    if (base == 1) {
        rowPtrs[0] = 1;
        int nJdata_tmp = 1;
        for (int nc=0; nc<NCELLS; nc++) {
            offset = nc * 53;
            for (int l=0; l<53; l++) {
                for (int k=0; k<53; k++) {
                    if(J[53*k + l] != 0.0) {
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
            offset = nc * 53;
            for (int l=0; l<53; l++) {
                for (int k=0; k<53; k++) {
                    if(J[53*k + l] != 0.0) {
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
    double c[52];
    double J[2809];
    int offset;

    for (int k=0; k<52; k++) {
        c[k] = 1.0/ 52.000000 ;
    }

    aJacobian(J, c, 1500.0, *consP);

    if (base == 1) {
        rowPtr[0] = 1;
        int nJdata_tmp = 1;
        for (int nc=0; nc<NCELLS; nc++) {
            offset = nc * 53;
            for (int l=0; l<53; l++) {
                for (int k=0; k<53; k++) {
                    if (k == l) {
                        colVals[nJdata_tmp-1] = l+1 + offset; 
                        nJdata_tmp = nJdata_tmp + 1; 
                    } else {
                        if(J[53*k + l] != 0.0) {
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
            offset = nc * 53;
            for (int l=0; l<53; l++) {
                for (int k=0; k<53; k++) {
                    if (k == l) {
                        colVals[nJdata_tmp] = l + offset; 
                        nJdata_tmp = nJdata_tmp + 1; 
                    } else {
                        if(J[53*k + l] != 0.0) {
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
    double c[52];
    double J[2809];

    for (int k=0; k<52; k++) {
        c[k] = 1.0/ 52.000000 ;
    }

    aJacobian_precond(J, c, 1500.0, *consP);

    colPtrs[0] = 0;
    int nJdata_tmp = 0;
    for (int k=0; k<53; k++) {
        for (int l=0; l<53; l++) {
            if (k == l) {
                rowVals[nJdata_tmp] = l; 
                indx[nJdata_tmp] = 53*k + l;
                nJdata_tmp = nJdata_tmp + 1; 
            } else {
                if(J[53*k + l] != 0.0) {
                    rowVals[nJdata_tmp] = l; 
                    indx[nJdata_tmp] = 53*k + l;
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
    double c[52];
    double J[2809];

    for (int k=0; k<52; k++) {
        c[k] = 1.0/ 52.000000 ;
    }

    aJacobian_precond(J, c, 1500.0, *consP);

    if (base == 1) {
        rowPtr[0] = 1;
        int nJdata_tmp = 1;
        for (int l=0; l<53; l++) {
            for (int k=0; k<53; k++) {
                if (k == l) {
                    colVals[nJdata_tmp-1] = l+1; 
                    nJdata_tmp = nJdata_tmp + 1; 
                } else {
                    if(J[53*k + l] != 0.0) {
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
        for (int l=0; l<53; l++) {
            for (int k=0; k<53; k++) {
                if (k == l) {
                    colVals[nJdata_tmp] = l; 
                    nJdata_tmp = nJdata_tmp + 1; 
                } else {
                    if(J[53*k + l] != 0.0) {
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
    for (int i=0; i<2809; i++) {
        J[i] = 0.0;
    }
}
#endif


/*compute an approx to the reaction Jacobian */
AMREX_GPU_HOST_DEVICE void aJacobian_precond(double *  J, double *  sc, double T, int HP)
{
    for (int i=0; i<2809; i++) {
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
        /*species 9: CH */
        species[9] =
            +2.07287600e-03
            -1.02688620e-05 * tc[1]
            +1.72016700e-08 * tc[2]
            -7.82213200e-12 * tc[3];
        /*species 10: HCO */
        species[10] =
            +6.19914700e-03
            -1.92461680e-05 * tc[1]
            +3.26947500e-08 * tc[2]
            -1.82995400e-11 * tc[3];
        /*species 11: CH2 */
        species[11] =
            +1.15981900e-03
            +4.97917000e-07 * tc[1]
            +2.64025080e-09 * tc[2]
            -2.93297400e-12 * tc[3];
        /*species 12: CO2 */
        species[12] =
            +9.92207200e-03
            -2.08182200e-05 * tc[1]
            +2.06000610e-08 * tc[2]
            -8.46912000e-12 * tc[3];
        /*species 13: CO */
        species[13] =
            +1.51194100e-03
            -7.76351000e-06 * tc[1]
            +1.67458320e-08 * tc[2]
            -9.89980400e-12 * tc[3];
        /*species 14: CH2O */
        species[14] =
            +1.26314400e-02
            -3.77633600e-05 * tc[1]
            +6.15009300e-08 * tc[2]
            -3.36529480e-11 * tc[3];
        /*species 15: CH2GSG */
        species[15] =
            -1.69908900e-04
            +2.05073800e-06 * tc[1]
            +7.47765300e-09 * tc[2]
            -7.92506400e-12 * tc[3];
        /*species 16: CH3 */
        species[16] =
            +1.11241000e-02
            -3.36044000e-05 * tc[1]
            +4.86548700e-08 * tc[2]
            -2.34598120e-11 * tc[3];
        /*species 17: CH3O */
        species[17] =
            +7.21659500e-03
            +1.06769440e-05 * tc[1]
            -2.21329080e-08 * tc[2]
            +8.30244400e-12 * tc[3];
        /*species 18: CH4 */
        species[18] =
            +1.74766800e-02
            -5.56681800e-05 * tc[1]
            +9.14912400e-08 * tc[2]
            -4.89572400e-11 * tc[3];
        /*species 19: CH3OH */
        species[19] =
            +7.34150800e-03
            +1.43401020e-05 * tc[1]
            -2.63795820e-08 * tc[2]
            +9.56228000e-12 * tc[3];
        /*species 21: C2H5 */
        species[21] =
            +8.71913300e-03
            +8.83967800e-06 * tc[1]
            +2.80161090e-09 * tc[2]
            -1.57110920e-11 * tc[3];
        /*species 22: CH2CO */
        species[22] =
            +1.21187100e-02
            -4.69009200e-06 * tc[1]
            -1.94000550e-08 * tc[2]
            +1.56225960e-11 * tc[3];
        /*species 26: C2H2 */
        species[26] =
            +1.51904500e-02
            -3.23263800e-05 * tc[1]
            +2.72369760e-08 * tc[2]
            -7.65098400e-12 * tc[3];
        /*species 27: HCCO */
        species[27] =
            +4.45347800e-03
            +4.53656600e-07 * tc[1]
            -4.44628500e-09 * tc[2]
            +9.00296800e-13 * tc[3];
        /*species 28: C2H3 */
        species[28] =
            +7.37147600e-03
            +4.21974600e-06 * tc[1]
            -3.96492600e-09 * tc[2]
            -4.73913600e-12 * tc[3];
        /*species 29: CH2CHO */
        species[29] =
            +1.07385700e-02
            +3.78298400e-06 * tc[1]
            -2.14757490e-08 * tc[2]
            +1.14695400e-11 * tc[3];
        /*species 31: C2H4 */
        species[31] =
            +2.79616300e-02
            -6.77735400e-05 * tc[1]
            +8.35545600e-08 * tc[2]
            -3.89515160e-11 * tc[3];
        /*species 33: CH3CO */
        species[33] =
            +9.77822000e-03
            +9.04289600e-06 * tc[1]
            -2.70283860e-08 * tc[2]
            +1.27748720e-11 * tc[3];
        /*species 35: C3H2 */
        species[35] =
            +2.48257200e-02
            -9.18327400e-05 * tc[1]
            +1.28040570e-07 * tc[2]
            -5.92860800e-11 * tc[3];
        /*species 36: C3H3 */
        species[36] =
            +1.10802800e-02
            +5.58664600e-07 * tc[1]
            -1.64376360e-08 * tc[2]
            +7.79851600e-12 * tc[3];
        /*species 39: NXC3H7 */
        species[39] =
            +2.47892700e-02
            +3.62049800e-06 * tc[1]
            -5.34979800e-08 * tc[2]
            +3.43319840e-11 * tc[3];
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
        /*species 9: CH */
        species[9] =
            +2.34038100e-03
            -1.41164020e-06 * tc[1]
            +2.70227460e-10 * tc[2]
            -1.54201600e-14 * tc[3];
        /*species 10: HCO */
        species[10] =
            +3.34557300e-03
            -2.67001200e-06 * tc[1]
            +7.41171900e-10 * tc[2]
            -6.85540400e-14 * tc[3];
        /*species 11: CH2 */
        species[11] =
            +1.93305700e-03
            -3.37403200e-07 * tc[1]
            -3.02969700e-10 * tc[2]
            +7.23302400e-14 * tc[3];
        /*species 12: CO2 */
        species[12] =
            +3.14016900e-03
            -2.55682200e-06 * tc[1]
            +7.18199100e-10 * tc[2]
            -6.67613200e-14 * tc[3];
        /*species 13: CO */
        species[13] =
            +1.44268900e-03
            -1.12616560e-06 * tc[1]
            +3.05574300e-10 * tc[2]
            -2.76438080e-14 * tc[3];
        /*species 14: CH2O */
        species[14] =
            +6.68132100e-03
            -5.25791000e-06 * tc[1]
            +1.42114590e-09 * tc[2]
            -1.28500680e-13 * tc[3];
        /*species 15: CH2GSG */
        species[15] =
            +2.06678800e-03
            -3.82823200e-07 * tc[1]
            -3.31401900e-10 * tc[2]
            +8.08540000e-14 * tc[3];
        /*species 16: CH3 */
        species[16] =
            +6.13797400e-03
            -4.46069000e-06 * tc[1]
            +1.13554830e-09 * tc[2]
            -9.80863600e-14 * tc[3];
        /*species 17: CH3O */
        species[17] =
            +7.87149700e-03
            -5.31276800e-06 * tc[1]
            +1.18332930e-09 * tc[2]
            -8.45046400e-14 * tc[3];
        /*species 18: CH4 */
        species[18] =
            +1.02372400e-02
            -7.75025800e-06 * tc[1]
            +2.03567550e-09 * tc[2]
            -1.80136920e-13 * tc[3];
        /*species 19: CH3OH */
        species[19] =
            +9.37659300e-03
            -6.10050800e-06 * tc[1]
            +1.30763790e-09 * tc[2]
            -8.89889200e-14 * tc[3];
        /*species 21: C2H5 */
        species[21] =
            +6.48407700e-03
            -1.28561300e-06 * tc[1]
            -7.04363700e-10 * tc[2]
            +1.55235080e-13 * tc[3];
        /*species 22: CH2CO */
        species[22] =
            +5.80484000e-03
            -3.84190800e-06 * tc[1]
            +8.38345500e-10 * tc[2]
            -5.83547200e-14 * tc[3];
        /*species 26: C2H2 */
        species[26] =
            +5.37603900e-03
            -3.82563400e-06 * tc[1]
            +9.85913700e-10 * tc[2]
            -8.62684000e-14 * tc[3];
        /*species 27: HCCO */
        species[27] =
            +2.00040000e-03
            -4.05521400e-07 * tc[1]
            -3.12339600e-10 * tc[2]
            +7.86066000e-14 * tc[3];
        /*species 28: C2H3 */
        species[28] =
            +4.01774600e-03
            -7.93348000e-07 * tc[1]
            -4.32380100e-10 * tc[2]
            +9.51457600e-14 * tc[3];
        /*species 29: CH2CHO */
        species[29] =
            +8.13059100e-03
            -5.48724800e-06 * tc[1]
            +1.22109120e-09 * tc[2]
            -8.70406800e-14 * tc[3];
        /*species 31: C2H4 */
        species[31] =
            +1.14851800e-02
            -8.83677000e-06 * tc[1]
            +2.35338030e-09 * tc[2]
            -2.10673920e-13 * tc[3];
        /*species 33: CH3CO */
        species[33] =
            +8.44988600e-03
            -5.70829400e-06 * tc[1]
            +1.27151280e-09 * tc[2]
            -9.07361600e-14 * tc[3];
        /*species 35: C3H2 */
        species[35] =
            +2.74874900e-03
            -8.74188600e-07 * tc[1]
            -1.93667970e-10 * tc[2]
            +6.65554800e-14 * tc[3];
        /*species 36: C3H3 */
        species[36] =
            +4.35719500e-03
            -8.21813400e-07 * tc[1]
            -7.10616900e-10 * tc[2]
            +1.75060800e-13 * tc[3];
        /*species 39: NXC3H7 */
        species[39] =
            +1.57611300e-02
            -1.03464860e-05 * tc[1]
            +2.23316760e-09 * tc[2]
            -1.52999120e-13 * tc[3];
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
        /*species 25: CH3O2H */
        species[25] =
            +1.90129767e-02
            -2.26772574e-05 * tc[1]
            +1.02091996e-08 * tc[2]
            -1.64732089e-12 * tc[3];
    } else {
        /*species 7: HO2 */
        species[7] =
            +2.38452835e-03
            -1.61269598e-06 * tc[1]
            +3.72575169e-10 * tc[2]
            -2.86560043e-14 * tc[3];
        /*species 25: CH3O2H */
        species[25] =
            +8.06817909e-03
            -5.54189842e-06 * tc[1]
            +1.29399673e-09 * tc[2]
            -1.00276858e-13 * tc[3];
    }

    /*species with midpoint at T=1384 kelvin */
    if (T < 1384) {
        /*species 20: C2H6 */
        species[20] =
            +2.40764754e-02
            -2.23786944e-05 * tc[1]
            +6.25022703e-09 * tc[2]
            -2.11947446e-13 * tc[3];
        /*species 40: NXC3H7O2 */
        species[40] =
            +3.96164986e-02
            -4.98983198e-05 * tc[1]
            +2.57835090e-08 * tc[2]
            -5.24961320e-12 * tc[3];
    } else {
        /*species 20: C2H6 */
        species[20] =
            +1.29236361e-02
            -8.85054392e-06 * tc[1]
            +2.06217518e-09 * tc[2]
            -1.59560693e-13 * tc[3];
        /*species 40: NXC3H7O2 */
        species[40] =
            +1.69910726e-02
            -1.17773375e-05 * tc[1]
            +2.76658619e-09 * tc[2]
            -2.15292270e-13 * tc[3];
    }

    /*species with midpoint at T=1376 kelvin */
    if (T < 1376) {
        /*species 23: HOCHO */
        species[23] =
            +1.63363016e-02
            -2.12514842e-05 * tc[1]
            +9.96398931e-09 * tc[2]
            -1.60870441e-12 * tc[3];
    } else {
        /*species 23: HOCHO */
        species[23] =
            +5.14289368e-03
            -3.64477026e-06 * tc[1]
            +8.69157489e-10 * tc[2]
            -6.83568796e-14 * tc[3];
    }

    /*species with midpoint at T=1385 kelvin */
    if (T < 1385) {
        /*species 24: CH3O2 */
        species[24] =
            +1.00873599e-02
            -6.43012368e-06 * tc[1]
            +6.28227801e-10 * tc[2]
            +1.67335641e-13 * tc[3];
        /*species 45: PXC4H9O2 */
        species[45] =
            +5.15513163e-02
            -6.56568800e-05 * tc[1]
            +3.39194580e-08 * tc[2]
            -6.80474424e-12 * tc[3];
    } else {
        /*species 24: CH3O2 */
        species[24] =
            +7.90728626e-03
            -5.36492468e-06 * tc[1]
            +1.24167401e-09 * tc[2]
            -9.56029320e-14 * tc[3];
        /*species 45: PXC4H9O2 */
        species[45] =
            +2.15210910e-02
            -1.48981803e-05 * tc[1]
            +3.49674213e-09 * tc[2]
            -2.71954244e-13 * tc[3];
    }

    /*species with midpoint at T=1388 kelvin */
    if (T < 1388) {
        /*species 30: C3H6 */
        species[30] =
            +2.89107662e-02
            -3.09773616e-05 * tc[1]
            +1.16644263e-08 * tc[2]
            -1.35156141e-12 * tc[3];
    } else {
        /*species 30: C3H6 */
        species[30] =
            +1.37023634e-02
            -9.32499466e-06 * tc[1]
            +2.16376321e-09 * tc[2]
            -1.66948050e-13 * tc[3];
    }

    /*species with midpoint at T=1389 kelvin */
    if (T < 1389) {
        /*species 32: C2H5O */
        species[32] =
            +2.71774434e-02
            -3.31818020e-05 * tc[1]
            +1.54561260e-08 * tc[2]
            -2.59398766e-12 * tc[3];
    } else {
        /*species 32: C2H5O */
        species[32] =
            +1.13072907e-02
            -7.68842842e-06 * tc[1]
            +1.78324232e-09 * tc[2]
            -1.37557815e-13 * tc[3];
    }

    /*species with midpoint at T=1382 kelvin */
    if (T < 1382) {
        /*species 34: C2H5O2 */
        species[34] =
            +2.76942578e-02
            -3.41608212e-05 * tc[1]
            +1.76385563e-08 * tc[2]
            -3.68359628e-12 * tc[3];
        /*species 50: C7H15X2 */
        species[50] =
            +7.56726570e-02
            -8.14947268e-05 * tc[1]
            +2.79803683e-08 * tc[2]
            -1.96944298e-12 * tc[3];
    } else {
        /*species 34: C2H5O2 */
        species[34] =
            +1.24472545e-02
            -8.64323152e-06 * tc[1]
            +2.03274910e-09 * tc[2]
            -1.58313827e-13 * tc[3];
        /*species 50: C7H15X2 */
        species[50] =
            +3.23324804e-02
            -2.18547614e-05 * tc[1]
            +5.05071180e-09 * tc[2]
            -3.88709636e-13 * tc[3];
    }

    /*species with midpoint at T=1400 kelvin */
    if (T < 1400) {
        /*species 37: C3H4XA */
        species[37] =
            +1.63343700e-02
            -3.52990000e-06 * tc[1]
            -1.39420950e-08 * tc[2]
            +6.91652400e-12 * tc[3];
    } else {
        /*species 37: C3H4XA */
        species[37] =
            +5.30213800e-03
            -7.40223600e-07 * tc[1]
            -9.07915800e-10 * tc[2]
            +2.03583240e-13 * tc[3];
    }

    /*species with midpoint at T=1397 kelvin */
    if (T < 1397) {
        /*species 38: C3H5XA */
        species[38] =
            +3.34559100e-02
            -5.06802054e-05 * tc[1]
            +3.08597262e-08 * tc[2]
            -6.93033360e-12 * tc[3];
        /*species 48: C5H11X1 */
        species[48] =
            +6.10632852e-02
            -8.18983650e-05 * tc[1]
            +4.38280410e-08 * tc[2]
            -8.75438460e-12 * tc[3];
    } else {
        /*species 38: C3H5XA */
        species[38] =
            +1.12695483e-02
            -7.67585728e-06 * tc[1]
            +1.78217736e-09 * tc[2]
            -1.37567212e-13 * tc[3];
        /*species 48: C5H11X1 */
        species[48] =
            +2.39041200e-02
            -1.62954324e-05 * tc[1]
            +3.78528708e-09 * tc[2]
            -2.92270934e-13 * tc[3];
    }

    /*species with midpoint at T=1398 kelvin */
    if (T < 1398) {
        /*species 41: C4H6 */
        species[41] =
            +4.78706062e-02
            -8.30893600e-05 * tc[1]
            +5.74648656e-08 * tc[2]
            -1.42863403e-11 * tc[3];
    } else {
        /*species 41: C4H6 */
        species[41] =
            +1.37163965e-02
            -9.39431566e-06 * tc[1]
            +2.18908151e-09 * tc[2]
            -1.69394481e-13 * tc[3];
    }

    /*species with midpoint at T=1392 kelvin */
    if (T < 1392) {
        /*species 42: C4H7 */
        species[42] =
            +4.26511243e-02
            -5.81958746e-05 * tc[1]
            +3.16211742e-08 * tc[2]
            -6.40239416e-12 * tc[3];
        /*species 43: C4H8X1 */
        species[43] =
            +4.52580978e-02
            -5.87317118e-05 * tc[1]
            +3.00661308e-08 * tc[2]
            -5.72766720e-12 * tc[3];
        /*species 46: C5H9 */
        species[46] =
            +5.57608487e-02
            -7.40287856e-05 * tc[1]
            +3.80651703e-08 * tc[2]
            -7.14155340e-12 * tc[3];
        /*species 47: C5H10X1 */
        species[47] =
            +5.74218294e-02
            -7.48973780e-05 * tc[1]
            +3.82094967e-08 * tc[2]
            -7.18439156e-12 * tc[3];
        /*species 49: C6H12X1 */
        species[49] =
            +6.98655426e-02
            -9.18816044e-05 * tc[1]
            +4.70902029e-08 * tc[2]
            -8.85184700e-12 * tc[3];
    } else {
        /*species 42: C4H7 */
        species[42] =
            +1.60483196e-02
            -1.09300458e-05 * tc[1]
            +2.53782316e-09 * tc[2]
            -1.95909096e-13 * tc[3];
        /*species 43: C4H8X1 */
        species[43] =
            +1.80617877e-02
            -1.23218606e-05 * tc[1]
            +2.86395888e-09 * tc[2]
            -2.21235856e-13 * tc[3];
        /*species 46: C5H9 */
        species[46] =
            +2.07128899e-02
            -1.41392123e-05 * tc[1]
            +3.28821399e-09 * tc[2]
            -2.54128883e-13 * tc[3];
        /*species 47: C5H10X1 */
        species[47] =
            +2.24072471e-02
            -1.52669605e-05 * tc[1]
            +3.54566898e-09 * tc[2]
            -2.73754056e-13 * tc[3];
        /*species 49: C6H12X1 */
        species[49] =
            +2.67377658e-02
            -1.82007355e-05 * tc[1]
            +4.22459304e-09 * tc[2]
            -3.26049698e-13 * tc[3];
    }

    /*species with midpoint at T=1395 kelvin */
    if (T < 1395) {
        /*species 44: PXC4H9 */
        species[44] =
            +4.78972364e-02
            -6.28046318e-05 * tc[1]
            +3.29359416e-08 * tc[2]
            -6.48042656e-12 * tc[3];
    } else {
        /*species 44: PXC4H9 */
        species[44] =
            +1.94310717e-02
            -1.32315590e-05 * tc[1]
            +3.07125408e-09 * tc[2]
            -2.37011883e-13 * tc[3];
    }

    /*species with midpoint at T=1391 kelvin */
    if (T < 1391) {
        /*species 51: NXC7H16 */
        species[51] =
            +8.54355820e-02
            -1.05069357e-04 * tc[1]
            +4.88837163e-08 * tc[2]
            -8.09579700e-12 * tc[3];
    } else {
        /*species 51: NXC7H16 */
        species[51] =
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

    /*reaction 1: H + O2 (+M) => HO2 (+M) */
    kc[0] = 1.0 / (refC) * exp((g_RT[3] + g_RT[6]) - (g_RT[7]));

    /*reaction 2: H2O2 (+M) => 2.000000 OH (+M) */
    kc[1] = refC * exp((g_RT[8]) - (2.000000 * g_RT[4]));

    /*reaction 3: OH + CH3 (+M) => CH3OH (+M) */
    kc[2] = 1.0 / (refC) * exp((g_RT[4] + g_RT[16]) - (g_RT[19]));

    /*reaction 4: CH3 + H (+M) => CH4 (+M) */
    kc[3] = 1.0 / (refC) * exp((g_RT[16] + g_RT[3]) - (g_RT[18]));

    /*reaction 5: 2.000000 CH3 (+M) => C2H6 (+M) */
    kc[4] = 1.0 / (refC) * exp((2.000000 * g_RT[16]) - (g_RT[20]));

    /*reaction 6: CO + CH2 (+M) => CH2CO (+M) */
    kc[5] = 1.0 / (refC) * exp((g_RT[13] + g_RT[11]) - (g_RT[22]));

    /*reaction 7: CO + O (+M) => CO2 (+M) */
    kc[6] = 1.0 / (refC) * exp((g_RT[13] + g_RT[1]) - (g_RT[12]));

    /*reaction 8: CH3O (+M) => CH2O + H (+M) */
    kc[7] = refC * exp((g_RT[17]) - (g_RT[14] + g_RT[3]));

    /*reaction 9: C2H3 (+M) => H + C2H2 (+M) */
    kc[8] = refC * exp((g_RT[28]) - (g_RT[3] + g_RT[26]));

    /*reaction 10: H + C2H4 (+M) <=> C2H5 (+M) */
    kc[9] = 1.0 / (refC) * exp((g_RT[3] + g_RT[31]) - (g_RT[21]));

    /*reaction 11: CH3CO (+M) => CH3 + CO (+M) */
    kc[10] = refC * exp((g_RT[33]) - (g_RT[16] + g_RT[13]));

    /*reaction 12: H + OH + M => H2O + M */
    kc[11] = 1.0 / (refC) * exp((g_RT[3] + g_RT[4]) - (g_RT[5]));

    /*reaction 13: CH2GSG + M => CH2 + M */
    kc[12] = exp((g_RT[15]) - (g_RT[11]));

    /*reaction 14: CH2 + M => CH2GSG + M */
    kc[13] = exp((g_RT[11]) - (g_RT[15]));

    /*reaction 15: HCO + M => H + CO + M */
    kc[14] = refC * exp((g_RT[10]) - (g_RT[3] + g_RT[13]));

    /*reaction 16: CH3O2 + M => CH3 + O2 + M */
    kc[15] = refC * exp((g_RT[24]) - (g_RT[16] + g_RT[6]));

    /*reaction 17: CH3 + O2 + M => CH3O2 + M */
    kc[16] = 1.0 / (refC) * exp((g_RT[16] + g_RT[6]) - (g_RT[24]));

    /*reaction 18: C2H5O + M => CH3 + CH2O + M */
    kc[17] = refC * exp((g_RT[32]) - (g_RT[16] + g_RT[14]));

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
    kc[33] = exp((g_RT[9] + g_RT[6]) - (g_RT[10] + g_RT[1]));

    /*reaction 35: CH2 + O2 => CO2 + 2.000000 H */
    kc[34] = refC * exp((g_RT[11] + g_RT[6]) - (g_RT[12] + 2.000000 * g_RT[3]));

    /*reaction 36: CH2 + O2 => CO + H2O */
    kc[35] = exp((g_RT[11] + g_RT[6]) - (g_RT[13] + g_RT[5]));

    /*reaction 37: CH2 + O => CO + 2.000000 H */
    kc[36] = refC * exp((g_RT[11] + g_RT[1]) - (g_RT[13] + 2.000000 * g_RT[3]));

    /*reaction 38: CH2 + O2 => CH2O + O */
    kc[37] = exp((g_RT[11] + g_RT[6]) - (g_RT[14] + g_RT[1]));

    /*reaction 39: CH2 + H => CH + H2 */
    kc[38] = exp((g_RT[11] + g_RT[3]) - (g_RT[9] + g_RT[2]));

    /*reaction 40: CH + H2 => CH2 + H */
    kc[39] = exp((g_RT[9] + g_RT[2]) - (g_RT[11] + g_RT[3]));

    /*reaction 41: CH2 + OH => CH + H2O */
    kc[40] = exp((g_RT[11] + g_RT[4]) - (g_RT[9] + g_RT[5]));

    /*reaction 42: CH + H2O => CH2 + OH */
    kc[41] = exp((g_RT[9] + g_RT[5]) - (g_RT[11] + g_RT[4]));

    /*reaction 43: CH2 + O2 => CO2 + H2 */
    kc[42] = exp((g_RT[11] + g_RT[6]) - (g_RT[12] + g_RT[2]));

    /*reaction 44: CH2GSG + H => CH + H2 */
    kc[43] = exp((g_RT[15] + g_RT[3]) - (g_RT[9] + g_RT[2]));

    /*reaction 45: CH2GSG + H2 => CH3 + H */
    kc[44] = exp((g_RT[15] + g_RT[2]) - (g_RT[16] + g_RT[3]));

    /*reaction 46: CH3 + H => CH2GSG + H2 */
    kc[45] = exp((g_RT[16] + g_RT[3]) - (g_RT[15] + g_RT[2]));

    /*reaction 47: CH2GSG + O2 => CO + OH + H */
    kc[46] = refC * exp((g_RT[15] + g_RT[6]) - (g_RT[13] + g_RT[4] + g_RT[3]));

    /*reaction 48: CH2GSG + OH => CH2O + H */
    kc[47] = exp((g_RT[15] + g_RT[4]) - (g_RT[14] + g_RT[3]));

    /*reaction 49: CH3 + OH => CH2O + H2 */
    kc[48] = exp((g_RT[16] + g_RT[4]) - (g_RT[14] + g_RT[2]));

    /*reaction 50: CH3 + OH => CH2GSG + H2O */
    kc[49] = exp((g_RT[16] + g_RT[4]) - (g_RT[15] + g_RT[5]));

    /*reaction 51: CH2GSG + H2O => CH3 + OH */
    kc[50] = exp((g_RT[15] + g_RT[5]) - (g_RT[16] + g_RT[4]));

    /*reaction 52: CH3 + O => CH2O + H */
    kc[51] = exp((g_RT[16] + g_RT[1]) - (g_RT[14] + g_RT[3]));

    /*reaction 53: CH3 + HO2 => CH3O + OH */
    kc[52] = exp((g_RT[16] + g_RT[7]) - (g_RT[17] + g_RT[4]));

    /*reaction 54: CH3 + HO2 => CH4 + O2 */
    kc[53] = exp((g_RT[16] + g_RT[7]) - (g_RT[18] + g_RT[6]));

    /*reaction 55: CH3 + O2 => CH2O + OH */
    kc[54] = exp((g_RT[16] + g_RT[6]) - (g_RT[14] + g_RT[4]));

    /*reaction 56: CH3 + H => CH2 + H2 */
    kc[55] = exp((g_RT[16] + g_RT[3]) - (g_RT[11] + g_RT[2]));

    /*reaction 57: CH2 + H2 => CH3 + H */
    kc[56] = exp((g_RT[11] + g_RT[2]) - (g_RT[16] + g_RT[3]));

    /*reaction 58: 2.000000 CH3 <=> H + C2H5 */
    kc[57] = exp((2.000000 * g_RT[16]) - (g_RT[3] + g_RT[21]));

    /*reaction 59: CH3 + OH => CH2 + H2O */
    kc[58] = exp((g_RT[16] + g_RT[4]) - (g_RT[11] + g_RT[5]));

    /*reaction 60: CH2 + H2O => CH3 + OH */
    kc[59] = exp((g_RT[11] + g_RT[5]) - (g_RT[16] + g_RT[4]));

    /*reaction 61: CH4 + O => CH3 + OH */
    kc[60] = exp((g_RT[18] + g_RT[1]) - (g_RT[16] + g_RT[4]));

    /*reaction 62: CH4 + H => CH3 + H2 */
    kc[61] = exp((g_RT[18] + g_RT[3]) - (g_RT[16] + g_RT[2]));

    /*reaction 63: CH3 + H2 => CH4 + H */
    kc[62] = exp((g_RT[16] + g_RT[2]) - (g_RT[18] + g_RT[3]));

    /*reaction 64: CH4 + OH => CH3 + H2O */
    kc[63] = exp((g_RT[18] + g_RT[4]) - (g_RT[16] + g_RT[5]));

    /*reaction 65: CH3 + H2O => CH4 + OH */
    kc[64] = exp((g_RT[16] + g_RT[5]) - (g_RT[18] + g_RT[4]));

    /*reaction 66: CO + OH => CO2 + H */
    kc[65] = exp((g_RT[13] + g_RT[4]) - (g_RT[12] + g_RT[3]));

    /*reaction 67: CO2 + H => CO + OH */
    kc[66] = exp((g_RT[12] + g_RT[3]) - (g_RT[13] + g_RT[4]));

    /*reaction 68: HCO + O2 => CO + HO2 */
    kc[67] = exp((g_RT[10] + g_RT[6]) - (g_RT[13] + g_RT[7]));

    /*reaction 69: HCO + O => CO2 + H */
    kc[68] = exp((g_RT[10] + g_RT[1]) - (g_RT[12] + g_RT[3]));

    /*reaction 70: HCO + OH => CO + H2O */
    kc[69] = exp((g_RT[10] + g_RT[4]) - (g_RT[13] + g_RT[5]));

    /*reaction 71: HCO + H => CO + H2 */
    kc[70] = exp((g_RT[10] + g_RT[3]) - (g_RT[13] + g_RT[2]));

    /*reaction 72: HCO + O => CO + OH */
    kc[71] = exp((g_RT[10] + g_RT[1]) - (g_RT[13] + g_RT[4]));

    /*reaction 73: HCO + CH3 => CH4 + CO */
    kc[72] = exp((g_RT[10] + g_RT[16]) - (g_RT[18] + g_RT[13]));

    /*reaction 74: CH2O + OH => HCO + H2O */
    kc[73] = exp((g_RT[14] + g_RT[4]) - (g_RT[10] + g_RT[5]));

    /*reaction 75: CH2O + O => HCO + OH */
    kc[74] = exp((g_RT[14] + g_RT[1]) - (g_RT[10] + g_RT[4]));

    /*reaction 76: CH2O + H => HCO + H2 */
    kc[75] = exp((g_RT[14] + g_RT[3]) - (g_RT[10] + g_RT[2]));

    /*reaction 77: CH2O + CH3 => HCO + CH4 */
    kc[76] = exp((g_RT[14] + g_RT[16]) - (g_RT[10] + g_RT[18]));

    /*reaction 78: 2.000000 CH3O => CH3OH + CH2O */
    kc[77] = exp((2.000000 * g_RT[17]) - (g_RT[19] + g_RT[14]));

    /*reaction 79: CH3O + O2 => CH2O + HO2 */
    kc[78] = exp((g_RT[17] + g_RT[6]) - (g_RT[14] + g_RT[7]));

    /*reaction 80: CH3O + H2 => CH3OH + H */
    kc[79] = exp((g_RT[17] + g_RT[2]) - (g_RT[19] + g_RT[3]));

    /*reaction 81: CH3OH + OH => CH3O + H2O */
    kc[80] = exp((g_RT[19] + g_RT[4]) - (g_RT[17] + g_RT[5]));

    /*reaction 82: CH2GSG + CO2 => CH2O + CO */
    kc[81] = exp((g_RT[15] + g_RT[12]) - (g_RT[14] + g_RT[13]));

    /*reaction 83: HOCHO + H => H2 + CO + OH */
    kc[82] = refC * exp((g_RT[23] + g_RT[3]) - (g_RT[2] + g_RT[13] + g_RT[4]));

    /*reaction 84: HOCHO + OH => H2O + CO + OH */
    kc[83] = refC * exp((g_RT[23] + g_RT[4]) - (g_RT[5] + g_RT[13] + g_RT[4]));

    /*reaction 85: HOCHO => HCO + OH */
    kc[84] = refC * exp((g_RT[23]) - (g_RT[10] + g_RT[4]));

    /*reaction 86: HCO + OH => HOCHO */
    kc[85] = 1.0 / (refC) * exp((g_RT[10] + g_RT[4]) - (g_RT[23]));

    /*reaction 87: HOCHO + H => H2 + CO2 + H */
    kc[86] = refC * exp((g_RT[23] + g_RT[3]) - (g_RT[2] + g_RT[12] + g_RT[3]));

    /*reaction 88: HOCHO + OH => H2O + CO2 + H */
    kc[87] = refC * exp((g_RT[23] + g_RT[4]) - (g_RT[5] + g_RT[12] + g_RT[3]));

    /*reaction 89: 2.000000 CH3O2 => O2 + 2.000000 CH3O */
    kc[88] = refC * exp((2.000000 * g_RT[24]) - (g_RT[6] + 2.000000 * g_RT[17]));

    /*reaction 90: CH3O2 + CH3 => 2.000000 CH3O */
    kc[89] = exp((g_RT[24] + g_RT[16]) - (2.000000 * g_RT[17]));

    /*reaction 91: 2.000000 CH3O2 => CH2O + CH3OH + O2 */
    kc[90] = refC * exp((2.000000 * g_RT[24]) - (g_RT[14] + g_RT[19] + g_RT[6]));

    /*reaction 92: CH3O2 + HO2 => CH3O2H + O2 */
    kc[91] = exp((g_RT[24] + g_RT[7]) - (g_RT[25] + g_RT[6]));

    /*reaction 93: CH3O2H => CH3O + OH */
    kc[92] = refC * exp((g_RT[25]) - (g_RT[17] + g_RT[4]));

    /*reaction 94: C2H2 + O => CH2 + CO */
    kc[93] = exp((g_RT[26] + g_RT[1]) - (g_RT[11] + g_RT[13]));

    /*reaction 95: C2H2 + O => HCCO + H */
    kc[94] = exp((g_RT[26] + g_RT[1]) - (g_RT[27] + g_RT[3]));

    /*reaction 96: C2H3 + H => C2H2 + H2 */
    kc[95] = exp((g_RT[28] + g_RT[3]) - (g_RT[26] + g_RT[2]));

    /*reaction 97: C2H3 + O2 => CH2CHO + O */
    kc[96] = exp((g_RT[28] + g_RT[6]) - (g_RT[29] + g_RT[1]));

    /*reaction 98: C2H3 + CH3 => C3H6 */
    kc[97] = 1.0 / (refC) * exp((g_RT[28] + g_RT[16]) - (g_RT[30]));

    /*reaction 99: C2H3 + O2 => C2H2 + HO2 */
    kc[98] = exp((g_RT[28] + g_RT[6]) - (g_RT[26] + g_RT[7]));

    /*reaction 100: C2H3 + O2 => CH2O + HCO */
    kc[99] = exp((g_RT[28] + g_RT[6]) - (g_RT[14] + g_RT[10]));

    /*reaction 101: C2H4 + CH3 => C2H3 + CH4 */
    kc[100] = exp((g_RT[31] + g_RT[16]) - (g_RT[28] + g_RT[18]));

    /*reaction 102: C2H4 + O => CH3 + HCO */
    kc[101] = exp((g_RT[31] + g_RT[1]) - (g_RT[16] + g_RT[10]));

    /*reaction 103: C2H4 + OH => C2H3 + H2O */
    kc[102] = exp((g_RT[31] + g_RT[4]) - (g_RT[28] + g_RT[5]));

    /*reaction 104: C2H4 + O => CH2CHO + H */
    kc[103] = exp((g_RT[31] + g_RT[1]) - (g_RT[29] + g_RT[3]));

    /*reaction 105: C2H4 + H => C2H3 + H2 */
    kc[104] = exp((g_RT[31] + g_RT[3]) - (g_RT[28] + g_RT[2]));

    /*reaction 106: C2H3 + H2 => C2H4 + H */
    kc[105] = exp((g_RT[28] + g_RT[2]) - (g_RT[31] + g_RT[3]));

    /*reaction 107: H + C2H5 => C2H6 */
    kc[106] = 1.0 / (refC) * exp((g_RT[3] + g_RT[21]) - (g_RT[20]));

    /*reaction 108: CH3O2 + C2H5 => CH3O + C2H5O */
    kc[107] = exp((g_RT[24] + g_RT[21]) - (g_RT[17] + g_RT[32]));

    /*reaction 109: C2H5 + HO2 => C2H5O + OH */
    kc[108] = exp((g_RT[21] + g_RT[7]) - (g_RT[32] + g_RT[4]));

    /*reaction 110: C2H5 + O2 => C2H4 + HO2 */
    kc[109] = exp((g_RT[21] + g_RT[6]) - (g_RT[31] + g_RT[7]));

    /*reaction 111: C2H6 + O => C2H5 + OH */
    kc[110] = exp((g_RT[20] + g_RT[1]) - (g_RT[21] + g_RT[4]));

    /*reaction 112: C2H6 + OH => C2H5 + H2O */
    kc[111] = exp((g_RT[20] + g_RT[4]) - (g_RT[21] + g_RT[5]));

    /*reaction 113: C2H6 + H => C2H5 + H2 */
    kc[112] = exp((g_RT[20] + g_RT[3]) - (g_RT[21] + g_RT[2]));

    /*reaction 114: HCCO + O => H + 2.000000 CO */
    kc[113] = refC * exp((g_RT[27] + g_RT[1]) - (g_RT[3] + 2.000000 * g_RT[13]));

    /*reaction 115: HCCO + OH => 2.000000 HCO */
    kc[114] = exp((g_RT[27] + g_RT[4]) - (2.000000 * g_RT[10]));

    /*reaction 116: HCCO + O2 => CO2 + HCO */
    kc[115] = exp((g_RT[27] + g_RT[6]) - (g_RT[12] + g_RT[10]));

    /*reaction 117: HCCO + H => CH2GSG + CO */
    kc[116] = exp((g_RT[27] + g_RT[3]) - (g_RT[15] + g_RT[13]));

    /*reaction 118: CH2GSG + CO => HCCO + H */
    kc[117] = exp((g_RT[15] + g_RT[13]) - (g_RT[27] + g_RT[3]));

    /*reaction 119: CH2CO + O => HCCO + OH */
    kc[118] = exp((g_RT[22] + g_RT[1]) - (g_RT[27] + g_RT[4]));

    /*reaction 120: CH2CO + H => HCCO + H2 */
    kc[119] = exp((g_RT[22] + g_RT[3]) - (g_RT[27] + g_RT[2]));

    /*reaction 121: HCCO + H2 => CH2CO + H */
    kc[120] = exp((g_RT[27] + g_RT[2]) - (g_RT[22] + g_RT[3]));

    /*reaction 122: CH2CO + H => CH3 + CO */
    kc[121] = exp((g_RT[22] + g_RT[3]) - (g_RT[16] + g_RT[13]));

    /*reaction 123: CH2CO + O => CH2 + CO2 */
    kc[122] = exp((g_RT[22] + g_RT[1]) - (g_RT[11] + g_RT[12]));

    /*reaction 124: CH2CO + OH => HCCO + H2O */
    kc[123] = exp((g_RT[22] + g_RT[4]) - (g_RT[27] + g_RT[5]));

    /*reaction 125: CH2CHO + O2 => CH2O + CO + OH */
    kc[124] = refC * exp((g_RT[29] + g_RT[6]) - (g_RT[14] + g_RT[13] + g_RT[4]));

    /*reaction 126: CH2CHO => CH2CO + H */
    kc[125] = refC * exp((g_RT[29]) - (g_RT[22] + g_RT[3]));

    /*reaction 127: CH2CO + H => CH2CHO */
    kc[126] = 1.0 / (refC) * exp((g_RT[22] + g_RT[3]) - (g_RT[29]));

    /*reaction 128: C2H5O2 => C2H5 + O2 */
    kc[127] = refC * exp((g_RT[34]) - (g_RT[21] + g_RT[6]));

    /*reaction 129: C2H5 + O2 => C2H5O2 */
    kc[128] = 1.0 / (refC) * exp((g_RT[21] + g_RT[6]) - (g_RT[34]));

    /*reaction 130: C2H5O2 => C2H4 + HO2 */
    kc[129] = refC * exp((g_RT[34]) - (g_RT[31] + g_RT[7]));

    /*reaction 131: C3H2 + O2 => HCCO + CO + H */
    kc[130] = refC * exp((g_RT[35] + g_RT[6]) - (g_RT[27] + g_RT[13] + g_RT[3]));

    /*reaction 132: C3H2 + OH => C2H2 + HCO */
    kc[131] = exp((g_RT[35] + g_RT[4]) - (g_RT[26] + g_RT[10]));

    /*reaction 133: C3H3 + O2 => CH2CO + HCO */
    kc[132] = exp((g_RT[36] + g_RT[6]) - (g_RT[22] + g_RT[10]));

    /*reaction 134: C3H3 + HO2 => C3H4XA + O2 */
    kc[133] = exp((g_RT[36] + g_RT[7]) - (g_RT[37] + g_RT[6]));

    /*reaction 135: C3H3 + H => C3H2 + H2 */
    kc[134] = exp((g_RT[36] + g_RT[3]) - (g_RT[35] + g_RT[2]));

    /*reaction 136: C3H3 + OH => C3H2 + H2O */
    kc[135] = exp((g_RT[36] + g_RT[4]) - (g_RT[35] + g_RT[5]));

    /*reaction 137: C3H2 + H2O => C3H3 + OH */
    kc[136] = exp((g_RT[35] + g_RT[5]) - (g_RT[36] + g_RT[4]));

    /*reaction 138: C3H4XA + H => C3H3 + H2 */
    kc[137] = exp((g_RT[37] + g_RT[3]) - (g_RT[36] + g_RT[2]));

    /*reaction 139: C3H4XA + OH => C3H3 + H2O */
    kc[138] = exp((g_RT[37] + g_RT[4]) - (g_RT[36] + g_RT[5]));

    /*reaction 140: C3H4XA + O => C2H4 + CO */
    kc[139] = exp((g_RT[37] + g_RT[1]) - (g_RT[31] + g_RT[13]));

    /*reaction 141: C3H5XA + H => C3H4XA + H2 */
    kc[140] = exp((g_RT[38] + g_RT[3]) - (g_RT[37] + g_RT[2]));

    /*reaction 142: C3H5XA + HO2 => C3H6 + O2 */
    kc[141] = exp((g_RT[38] + g_RT[7]) - (g_RT[30] + g_RT[6]));

    /*reaction 143: C3H5XA + H => C3H6 */
    kc[142] = 1.0 / (refC) * exp((g_RT[38] + g_RT[3]) - (g_RT[30]));

    /*reaction 144: C3H5XA => C2H2 + CH3 */
    kc[143] = refC * exp((g_RT[38]) - (g_RT[26] + g_RT[16]));

    /*reaction 145: C3H5XA => C3H4XA + H */
    kc[144] = refC * exp((g_RT[38]) - (g_RT[37] + g_RT[3]));

    /*reaction 146: C3H4XA + H => C3H5XA */
    kc[145] = 1.0 / (refC) * exp((g_RT[37] + g_RT[3]) - (g_RT[38]));

    /*reaction 147: C3H5XA + CH2O => C3H6 + HCO */
    kc[146] = exp((g_RT[38] + g_RT[14]) - (g_RT[30] + g_RT[10]));

    /*reaction 148: 2.000000 C3H5XA => C3H4XA + C3H6 */
    kc[147] = exp((2.000000 * g_RT[38]) - (g_RT[37] + g_RT[30]));

    /*reaction 149: C3H6 + H => C2H4 + CH3 */
    kc[148] = exp((g_RT[30] + g_RT[3]) - (g_RT[31] + g_RT[16]));

    /*reaction 150: C3H6 + H => C3H5XA + H2 */
    kc[149] = exp((g_RT[30] + g_RT[3]) - (g_RT[38] + g_RT[2]));

    /*reaction 151: C3H6 + O => C2H5 + HCO */
    kc[150] = exp((g_RT[30] + g_RT[1]) - (g_RT[21] + g_RT[10]));

    /*reaction 152: C3H6 + O => C3H5XA + OH */
    kc[151] = exp((g_RT[30] + g_RT[1]) - (g_RT[38] + g_RT[4]));

    /*reaction 153: C3H6 + O => CH2CO + CH3 + H */
    kc[152] = refC * exp((g_RT[30] + g_RT[1]) - (g_RT[22] + g_RT[16] + g_RT[3]));

    /*reaction 154: C3H6 + OH => C3H5XA + H2O */
    kc[153] = exp((g_RT[30] + g_RT[4]) - (g_RT[38] + g_RT[5]));

    /*reaction 155: NXC3H7 + O2 => C3H6 + HO2 */
    kc[154] = exp((g_RT[39] + g_RT[6]) - (g_RT[30] + g_RT[7]));

    /*reaction 156: NXC3H7 => CH3 + C2H4 */
    kc[155] = refC * exp((g_RT[39]) - (g_RT[16] + g_RT[31]));

    /*reaction 157: CH3 + C2H4 => NXC3H7 */
    kc[156] = 1.0 / (refC) * exp((g_RT[16] + g_RT[31]) - (g_RT[39]));

    /*reaction 158: NXC3H7 => H + C3H6 */
    kc[157] = refC * exp((g_RT[39]) - (g_RT[3] + g_RT[30]));

    /*reaction 159: H + C3H6 => NXC3H7 */
    kc[158] = 1.0 / (refC) * exp((g_RT[3] + g_RT[30]) - (g_RT[39]));

    /*reaction 160: NXC3H7O2 => NXC3H7 + O2 */
    kc[159] = refC * exp((g_RT[40]) - (g_RT[39] + g_RT[6]));

    /*reaction 161: NXC3H7 + O2 => NXC3H7O2 */
    kc[160] = 1.0 / (refC) * exp((g_RT[39] + g_RT[6]) - (g_RT[40]));

    /*reaction 162: C4H6 => 2.000000 C2H3 */
    kc[161] = refC * exp((g_RT[41]) - (2.000000 * g_RT[28]));

    /*reaction 163: 2.000000 C2H3 => C4H6 */
    kc[162] = 1.0 / (refC) * exp((2.000000 * g_RT[28]) - (g_RT[41]));

    /*reaction 164: C4H6 + OH => CH2O + C3H5XA */
    kc[163] = exp((g_RT[41] + g_RT[4]) - (g_RT[14] + g_RT[38]));

    /*reaction 165: C4H6 + OH => C2H5 + CH2CO */
    kc[164] = exp((g_RT[41] + g_RT[4]) - (g_RT[21] + g_RT[22]));

    /*reaction 166: C4H6 + O => C2H4 + CH2CO */
    kc[165] = exp((g_RT[41] + g_RT[1]) - (g_RT[31] + g_RT[22]));

    /*reaction 167: C4H6 + H => C2H3 + C2H4 */
    kc[166] = exp((g_RT[41] + g_RT[3]) - (g_RT[28] + g_RT[31]));

    /*reaction 168: C4H6 + O => CH2O + C3H4XA */
    kc[167] = exp((g_RT[41] + g_RT[1]) - (g_RT[14] + g_RT[37]));

    /*reaction 169: H + C4H7 => C4H8X1 */
    kc[168] = 1.0 / (refC) * exp((g_RT[3] + g_RT[42]) - (g_RT[43]));

    /*reaction 170: C3H5XA + C4H7 => C3H6 + C4H6 */
    kc[169] = exp((g_RT[38] + g_RT[42]) - (g_RT[30] + g_RT[41]));

    /*reaction 171: C2H5 + C4H7 => C4H6 + C2H6 */
    kc[170] = exp((g_RT[21] + g_RT[42]) - (g_RT[41] + g_RT[20]));

    /*reaction 172: C4H7 => C4H6 + H */
    kc[171] = refC * exp((g_RT[42]) - (g_RT[41] + g_RT[3]));

    /*reaction 173: C4H6 + H => C4H7 */
    kc[172] = 1.0 / (refC) * exp((g_RT[41] + g_RT[3]) - (g_RT[42]));

    /*reaction 174: C4H7 + CH3 => C4H6 + CH4 */
    kc[173] = exp((g_RT[42] + g_RT[16]) - (g_RT[41] + g_RT[18]));

    /*reaction 175: C4H7 + HO2 => C4H8X1 + O2 */
    kc[174] = exp((g_RT[42] + g_RT[7]) - (g_RT[43] + g_RT[6]));

    /*reaction 176: C4H7 + O2 => C4H6 + HO2 */
    kc[175] = exp((g_RT[42] + g_RT[6]) - (g_RT[41] + g_RT[7]));

    /*reaction 177: C4H7 => C2H4 + C2H3 */
    kc[176] = refC * exp((g_RT[42]) - (g_RT[31] + g_RT[28]));

    /*reaction 178: H + C4H7 => C4H6 + H2 */
    kc[177] = exp((g_RT[3] + g_RT[42]) - (g_RT[41] + g_RT[2]));

    /*reaction 179: C4H8X1 + H => C4H7 + H2 */
    kc[178] = exp((g_RT[43] + g_RT[3]) - (g_RT[42] + g_RT[2]));

    /*reaction 180: C4H8X1 + OH => NXC3H7 + CH2O */
    kc[179] = exp((g_RT[43] + g_RT[4]) - (g_RT[39] + g_RT[14]));

    /*reaction 181: C4H8X1 + OH => CH3CO + C2H6 */
    kc[180] = exp((g_RT[43] + g_RT[4]) - (g_RT[33] + g_RT[20]));

    /*reaction 182: C4H8X1 + O => CH3CO + C2H5 */
    kc[181] = exp((g_RT[43] + g_RT[1]) - (g_RT[33] + g_RT[21]));

    /*reaction 183: C4H8X1 + O => C3H6 + CH2O */
    kc[182] = exp((g_RT[43] + g_RT[1]) - (g_RT[30] + g_RT[14]));

    /*reaction 184: C4H8X1 + OH => C4H7 + H2O */
    kc[183] = exp((g_RT[43] + g_RT[4]) - (g_RT[42] + g_RT[5]));

    /*reaction 185: C4H8X1 => C3H5XA + CH3 */
    kc[184] = refC * exp((g_RT[43]) - (g_RT[38] + g_RT[16]));

    /*reaction 186: C3H5XA + CH3 => C4H8X1 */
    kc[185] = 1.0 / (refC) * exp((g_RT[38] + g_RT[16]) - (g_RT[43]));

    /*reaction 187: PXC4H9 => C4H8X1 + H */
    kc[186] = refC * exp((g_RT[44]) - (g_RT[43] + g_RT[3]));

    /*reaction 188: C4H8X1 + H => PXC4H9 */
    kc[187] = 1.0 / (refC) * exp((g_RT[43] + g_RT[3]) - (g_RT[44]));

    /*reaction 189: PXC4H9 => C2H5 + C2H4 */
    kc[188] = refC * exp((g_RT[44]) - (g_RT[21] + g_RT[31]));

    /*reaction 190: PXC4H9O2 => PXC4H9 + O2 */
    kc[189] = refC * exp((g_RT[45]) - (g_RT[44] + g_RT[6]));

    /*reaction 191: PXC4H9 + O2 => PXC4H9O2 */
    kc[190] = 1.0 / (refC) * exp((g_RT[44] + g_RT[6]) - (g_RT[45]));

    /*reaction 192: C5H9 => C4H6 + CH3 */
    kc[191] = refC * exp((g_RT[46]) - (g_RT[41] + g_RT[16]));

    /*reaction 193: C5H9 => C3H5XA + C2H4 */
    kc[192] = refC * exp((g_RT[46]) - (g_RT[38] + g_RT[31]));

    /*reaction 194: C5H10X1 + OH => C5H9 + H2O */
    kc[193] = exp((g_RT[47] + g_RT[4]) - (g_RT[46] + g_RT[5]));

    /*reaction 195: C5H10X1 + H => C5H9 + H2 */
    kc[194] = exp((g_RT[47] + g_RT[3]) - (g_RT[46] + g_RT[2]));

    /*reaction 196: C5H10X1 => C2H5 + C3H5XA */
    kc[195] = refC * exp((g_RT[47]) - (g_RT[21] + g_RT[38]));

    /*reaction 197: C2H5 + C3H5XA => C5H10X1 */
    kc[196] = 1.0 / (refC) * exp((g_RT[21] + g_RT[38]) - (g_RT[47]));

    /*reaction 198: C5H10X1 + O => C5H9 + OH */
    kc[197] = exp((g_RT[47] + g_RT[1]) - (g_RT[46] + g_RT[4]));

    /*reaction 199: C5H11X1 => C3H6 + C2H5 */
    kc[198] = refC * exp((g_RT[48]) - (g_RT[30] + g_RT[21]));

    /*reaction 200: C5H11X1 => C2H4 + NXC3H7 */
    kc[199] = refC * exp((g_RT[48]) - (g_RT[31] + g_RT[39]));

    /*reaction 201: C5H11X1 <=> C5H10X1 + H */
    kc[200] = refC * exp((g_RT[48]) - (g_RT[47] + g_RT[3]));

    /*reaction 202: C6H12X1 => NXC3H7 + C3H5XA */
    kc[201] = refC * exp((g_RT[49]) - (g_RT[39] + g_RT[38]));

    /*reaction 203: C6H12X1 + OH => C5H11X1 + CH2O */
    kc[202] = exp((g_RT[49] + g_RT[4]) - (g_RT[48] + g_RT[14]));

    /*reaction 204: C7H15X2 => C6H12X1 + CH3 */
    kc[203] = refC * exp((g_RT[50]) - (g_RT[49] + g_RT[16]));

    /*reaction 205: C7H15X2 => PXC4H9 + C3H6 */
    kc[204] = refC * exp((g_RT[50]) - (g_RT[44] + g_RT[30]));

    /*reaction 206: C7H15X2 => C4H8X1 + NXC3H7 */
    kc[205] = refC * exp((g_RT[50]) - (g_RT[43] + g_RT[39]));

    /*reaction 207: C7H15X2 => C5H11X1 + C2H4 */
    kc[206] = refC * exp((g_RT[50]) - (g_RT[48] + g_RT[31]));

    /*reaction 208: C7H15X2 => C2H5 + C5H10X1 */
    kc[207] = refC * exp((g_RT[50]) - (g_RT[21] + g_RT[47]));

    /*reaction 209: C7H15X2 + HO2 => NXC7H16 + O2 */
    kc[208] = exp((g_RT[50] + g_RT[7]) - (g_RT[51] + g_RT[6]));

    /*reaction 210: NXC7H16 + CH3O2 => C7H15X2 + CH3O2H */
    kc[209] = exp((g_RT[51] + g_RT[24]) - (g_RT[50] + g_RT[25]));

    /*reaction 211: NXC7H16 + H => C7H15X2 + H2 */
    kc[210] = exp((g_RT[51] + g_RT[3]) - (g_RT[50] + g_RT[2]));

    /*reaction 212: NXC7H16 => PXC4H9 + NXC3H7 */
    kc[211] = refC * exp((g_RT[51]) - (g_RT[44] + g_RT[39]));

    /*reaction 213: NXC7H16 + HO2 => C7H15X2 + H2O2 */
    kc[212] = exp((g_RT[51] + g_RT[7]) - (g_RT[50] + g_RT[8]));

    /*reaction 214: NXC7H16 => C5H11X1 + C2H5 */
    kc[213] = refC * exp((g_RT[51]) - (g_RT[48] + g_RT[21]));

    /*reaction 215: NXC7H16 + CH3O => C7H15X2 + CH3OH */
    kc[214] = exp((g_RT[51] + g_RT[17]) - (g_RT[50] + g_RT[19]));

    /*reaction 216: NXC7H16 + O => C7H15X2 + OH */
    kc[215] = exp((g_RT[51] + g_RT[1]) - (g_RT[50] + g_RT[4]));

    /*reaction 217: NXC7H16 + OH => C7H15X2 + H2O */
    kc[216] = exp((g_RT[51] + g_RT[4]) - (g_RT[50] + g_RT[5]));

    /*reaction 218: NXC7H16 + CH3 => C7H15X2 + CH4 */
    kc[217] = exp((g_RT[51] + g_RT[16]) - (g_RT[50] + g_RT[18]));

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
        /*species 9: CH */
        species[9] =
            +7.045259000000000e+04 * invT
            -1.313860000000000e-01
            -3.200202000000000e+00 * tc[0]
            -1.036438000000000e-03 * tc[1]
            +8.557385000000000e-07 * tc[2]
            -4.778241666666666e-10 * tc[3]
            +9.777665000000000e-14 * tc[4];
        /*species 10: HCO */
        species[10] =
            +4.159922000000000e+03 * invT
            -6.085284000000000e+00
            -2.898330000000000e+00 * tc[0]
            -3.099573500000000e-03 * tc[1]
            +1.603847333333333e-06 * tc[2]
            -9.081875000000000e-10 * tc[3]
            +2.287442500000000e-13 * tc[4];
        /*species 11: CH2 */
        species[11] =
            +4.536791000000000e+04 * invT
            +2.049659000000000e+00
            -3.762237000000000e+00 * tc[0]
            -5.799095000000000e-04 * tc[1]
            -4.149308333333333e-08 * tc[2]
            -7.334030000000001e-11 * tc[3]
            +3.666217500000000e-14 * tc[4];
        /*species 12: CO2 */
        species[12] =
            -4.837314000000000e+04 * invT
            -7.912765000000000e+00
            -2.275725000000000e+00 * tc[0]
            -4.961036000000000e-03 * tc[1]
            +1.734851666666667e-06 * tc[2]
            -5.722239166666667e-10 * tc[3]
            +1.058640000000000e-13 * tc[4];
        /*species 13: CO */
        species[13] =
            -1.431054000000000e+04 * invT
            -1.586445000000000e+00
            -3.262452000000000e+00 * tc[0]
            -7.559705000000000e-04 * tc[1]
            +6.469591666666667e-07 * tc[2]
            -4.651620000000000e-10 * tc[3]
            +1.237475500000000e-13 * tc[4];
        /*species 14: CH2O */
        species[14] =
            -1.486540000000000e+04 * invT
            -1.213208900000000e+01
            -1.652731000000000e+00 * tc[0]
            -6.315720000000000e-03 * tc[1]
            +3.146946666666667e-06 * tc[2]
            -1.708359166666667e-09 * tc[3]
            +4.206618500000000e-13 * tc[4];
        /*species 15: CH2GSG */
        species[15] =
            +4.989368000000000e+04 * invT
            +3.913732930000000e+00
            -3.971265000000000e+00 * tc[0]
            +8.495445000000000e-05 * tc[1]
            -1.708948333333333e-07 * tc[2]
            -2.077125833333333e-10 * tc[3]
            +9.906330000000000e-14 * tc[4];
        /*species 16: CH3 */
        species[16] =
            +1.642378000000000e+04 * invT
            -4.359351000000000e+00
            -2.430443000000000e+00 * tc[0]
            -5.562050000000000e-03 * tc[1]
            +2.800366666666666e-06 * tc[2]
            -1.351524166666667e-09 * tc[3]
            +2.932476500000000e-13 * tc[4];
        /*species 17: CH3O */
        species[17] =
            +9.786011000000000e+02 * invT
            -1.104597600000000e+01
            -2.106204000000000e+00 * tc[0]
            -3.608297500000000e-03 * tc[1]
            -8.897453333333333e-07 * tc[2]
            +6.148030000000000e-10 * tc[3]
            -1.037805500000000e-13 * tc[4];
        /*species 18: CH4 */
        species[18] =
            -9.825228999999999e+03 * invT
            -1.294344850000000e+01
            -7.787415000000000e-01 * tc[0]
            -8.738340000000001e-03 * tc[1]
            +4.639015000000000e-06 * tc[2]
            -2.541423333333333e-09 * tc[3]
            +6.119655000000000e-13 * tc[4];
        /*species 19: CH3OH */
        species[19] =
            -2.535348000000000e+04 * invT
            -8.572515000000001e+00
            -2.660115000000000e+00 * tc[0]
            -3.670754000000000e-03 * tc[1]
            -1.195008500000000e-06 * tc[2]
            +7.327661666666667e-10 * tc[3]
            -1.195285000000000e-13 * tc[4];
        /*species 21: C2H5 */
        species[21] =
            +1.287040000000000e+04 * invT
            -9.447498000000000e+00
            -2.690702000000000e+00 * tc[0]
            -4.359566500000000e-03 * tc[1]
            -7.366398333333332e-07 * tc[2]
            -7.782252500000001e-11 * tc[3]
            +1.963886500000000e-13 * tc[4];
        /*species 22: CH2CO */
        species[22] =
            -7.632637000000000e+03 * invT
            -5.698582000000000e+00
            -2.974971000000000e+00 * tc[0]
            -6.059355000000000e-03 * tc[1]
            +3.908410000000000e-07 * tc[2]
            +5.388904166666666e-10 * tc[3]
            -1.952824500000000e-13 * tc[4];
        /*species 26: C2H2 */
        species[26] =
            +2.612444000000000e+04 * invT
            -6.791815999999999e+00
            -2.013562000000000e+00 * tc[0]
            -7.595225000000000e-03 * tc[1]
            +2.693865000000000e-06 * tc[2]
            -7.565826666666667e-10 * tc[3]
            +9.563730000000000e-14 * tc[4];
        /*species 27: HCCO */
        species[27] =
            +1.965892000000000e+04 * invT
            +4.566121099999999e+00
            -5.047965000000000e+00 * tc[0]
            -2.226739000000000e-03 * tc[1]
            -3.780471666666667e-08 * tc[2]
            +1.235079166666667e-10 * tc[3]
            -1.125371000000000e-14 * tc[4];
        /*species 28: C2H3 */
        species[28] =
            +3.335225000000000e+04 * invT
            -9.096924000000001e+00
            -2.459276000000000e+00 * tc[0]
            -3.685738000000000e-03 * tc[1]
            -3.516455000000000e-07 * tc[2]
            +1.101368333333333e-10 * tc[3]
            +5.923920000000000e-14 * tc[4];
        /*species 29: CH2CHO */
        species[29] =
            +1.521477000000000e+03 * invT
            -6.149227999999999e+00
            -3.409062000000000e+00 * tc[0]
            -5.369285000000000e-03 * tc[1]
            -3.152486666666667e-07 * tc[2]
            +5.965485833333333e-10 * tc[3]
            -1.433692500000000e-13 * tc[4];
        /*species 31: C2H4 */
        species[31] =
            +5.573046000000000e+03 * invT
            -2.507297800000000e+01
            +8.614880000000000e-01 * tc[0]
            -1.398081500000000e-02 * tc[1]
            +5.647795000000000e-06 * tc[2]
            -2.320960000000000e-09 * tc[3]
            +4.868939500000000e-13 * tc[4];
        /*species 33: CH3CO */
        species[33] =
            -4.108508000000000e+03 * invT
            -8.103572000000000e+00
            -3.125278000000000e+00 * tc[0]
            -4.889110000000000e-03 * tc[1]
            -7.535746666666667e-07 * tc[2]
            +7.507885000000000e-10 * tc[3]
            -1.596859000000000e-13 * tc[4];
        /*species 35: C3H2 */
        species[35] =
            +6.350421000000000e+04 * invT
            -5.702732000000000e+00
            -3.166714000000000e+00 * tc[0]
            -1.241286000000000e-02 * tc[1]
            +7.652728333333333e-06 * tc[2]
            -3.556682500000000e-09 * tc[3]
            +7.410759999999999e-13 * tc[4];
        /*species 36: C3H3 */
        species[36] =
            +3.988883000000000e+04 * invT
            +4.168745100000000e+00
            -4.754200000000000e+00 * tc[0]
            -5.540140000000000e-03 * tc[1]
            -4.655538333333333e-08 * tc[2]
            +4.566010000000000e-10 * tc[3]
            -9.748145000000000e-14 * tc[4];
        /*species 39: NXC3H7 */
        species[39] =
            +9.713281000000001e+03 * invT
            -1.207017300000000e+01
            -1.922537000000000e+00 * tc[0]
            -1.239463500000000e-02 * tc[1]
            -3.017081666666666e-07 * tc[2]
            +1.486055000000000e-09 * tc[3]
            -4.291498000000000e-13 * tc[4];
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
        /*species 9: CH */
        species[9] =
            +7.086723000000000e+04 * invT
            -6.982150000000001e+00
            -2.196223000000000e+00 * tc[0]
            -1.170190500000000e-03 * tc[1]
            +1.176366833333333e-07 * tc[2]
            -7.506318333333334e-12 * tc[3]
            +1.927520000000000e-16 * tc[4];
        /*species 10: HCO */
        species[10] =
            +3.916324000000000e+03 * invT
            -1.995028000000000e+00
            -3.557271000000000e+00 * tc[0]
            -1.672786500000000e-03 * tc[1]
            +2.225010000000000e-07 * tc[2]
            -2.058810833333333e-11 * tc[3]
            +8.569255000000000e-16 * tc[4];
        /*species 11: CH2 */
        species[11] =
            +4.534134000000000e+04 * invT
            +1.479847000000000e+00
            -3.636408000000000e+00 * tc[0]
            -9.665285000000000e-04 * tc[1]
            +2.811693333333333e-08 * tc[2]
            +8.415825000000000e-12 * tc[3]
            -9.041279999999999e-16 * tc[4];
        /*species 12: CO2 */
        species[12] =
            -4.896696000000000e+04 * invT
            +5.409018900000000e+00
            -4.453623000000000e+00 * tc[0]
            -1.570084500000000e-03 * tc[1]
            +2.130685000000000e-07 * tc[2]
            -1.994997500000000e-11 * tc[3]
            +8.345165000000000e-16 * tc[4];
        /*species 13: CO */
        species[13] =
            -1.426835000000000e+04 * invT
            -3.083140000000000e+00
            -3.025078000000000e+00 * tc[0]
            -7.213445000000000e-04 * tc[1]
            +9.384713333333334e-08 * tc[2]
            -8.488174999999999e-12 * tc[3]
            +3.455476000000000e-16 * tc[4];
        /*species 14: CH2O */
        species[14] =
            -1.532037000000000e+04 * invT
            -3.916966000000000e+00
            -2.995606000000000e+00 * tc[0]
            -3.340660500000000e-03 * tc[1]
            +4.381591666666666e-07 * tc[2]
            -3.947627500000000e-11 * tc[3]
            +1.606258500000000e-15 * tc[4];
        /*species 15: CH2GSG */
        species[15] =
            +4.984975000000000e+04 * invT
            +1.866319000000000e+00
            -3.552889000000000e+00 * tc[0]
            -1.033394000000000e-03 * tc[1]
            +3.190193333333333e-08 * tc[2]
            +9.205608333333333e-12 * tc[3]
            -1.010675000000000e-15 * tc[4];
        /*species 16: CH3 */
        species[16] =
            +1.643781000000000e+04 * invT
            -2.608645000000000e+00
            -2.844052000000000e+00 * tc[0]
            -3.068987000000000e-03 * tc[1]
            +3.717241666666666e-07 * tc[2]
            -3.154300833333333e-11 * tc[3]
            +1.226079500000000e-15 * tc[4];
        /*species 17: CH3O */
        species[17] =
            +1.278325000000000e+02 * invT
            +8.412250000000001e-01
            -3.770800000000000e+00 * tc[0]
            -3.935748500000000e-03 * tc[1]
            +4.427306666666667e-07 * tc[2]
            -3.287025833333333e-11 * tc[3]
            +1.056308000000000e-15 * tc[4];
        /*species 18: CH4 */
        species[18] =
            -1.008079000000000e+04 * invT
            -7.939916000000000e+00
            -1.683479000000000e+00 * tc[0]
            -5.118620000000000e-03 * tc[1]
            +6.458548333333333e-07 * tc[2]
            -5.654654166666667e-11 * tc[3]
            +2.251711500000000e-15 * tc[4];
        /*species 19: CH3OH */
        species[19] =
            -2.615791000000000e+04 * invT
            +1.650865000000000e+00
            -4.029061000000000e+00 * tc[0]
            -4.688296500000000e-03 * tc[1]
            +5.083756666666666e-07 * tc[2]
            -3.632327500000000e-11 * tc[3]
            +1.112361500000000e-15 * tc[4];
        /*species 21: C2H5 */
        species[21] =
            +1.067455000000000e+04 * invT
            +2.197137000000000e+01
            -7.190480000000000e+00 * tc[0]
            -3.242038500000000e-03 * tc[1]
            +1.071344166666667e-07 * tc[2]
            +1.956565833333333e-11 * tc[3]
            -1.940438500000000e-15 * tc[4];
        /*species 22: CH2CO */
        species[22] =
            -8.583402000000000e+03 * invT
            +1.369639800000000e+01
            -6.038817000000000e+00 * tc[0]
            -2.902420000000000e-03 * tc[1]
            +3.201590000000000e-07 * tc[2]
            -2.328737500000000e-11 * tc[3]
            +7.294340000000000e-16 * tc[4];
        /*species 26: C2H2 */
        species[26] =
            +2.566766000000000e+04 * invT
            +7.237108000000000e+00
            -4.436770000000000e+00 * tc[0]
            -2.688019500000000e-03 * tc[1]
            +3.188028333333333e-07 * tc[2]
            -2.738649166666667e-11 * tc[3]
            +1.078355000000000e-15 * tc[4];
        /*species 27: HCCO */
        species[27] =
            +1.901513000000000e+04 * invT
            +1.582933500000000e+01
            -6.758073000000000e+00 * tc[0]
            -1.000200000000000e-03 * tc[1]
            +3.379345000000000e-08 * tc[2]
            +8.676100000000000e-12 * tc[3]
            -9.825825000000000e-16 * tc[4];
        /*species 28: C2H3 */
        species[28] =
            +3.185435000000000e+04 * invT
            +1.446378100000000e+01
            -5.933468000000000e+00 * tc[0]
            -2.008873000000000e-03 * tc[1]
            +6.611233333333333e-08 * tc[2]
            +1.201055833333333e-11 * tc[3]
            -1.189322000000000e-15 * tc[4];
        /*species 29: CH2CHO */
        species[29] =
            +4.903218000000000e+02 * invT
            +1.102092100000000e+01
            -5.975670000000000e+00 * tc[0]
            -4.065295500000000e-03 * tc[1]
            +4.572706666666667e-07 * tc[2]
            -3.391920000000000e-11 * tc[3]
            +1.088008500000000e-15 * tc[4];
        /*species 31: C2H4 */
        species[31] =
            +4.428289000000000e+03 * invT
            +1.298030000000000e+00
            -3.528419000000000e+00 * tc[0]
            -5.742590000000000e-03 * tc[1]
            +7.363975000000000e-07 * tc[2]
            -6.537167500000001e-11 * tc[3]
            +2.633424000000000e-15 * tc[4];
        /*species 33: CH3CO */
        species[33] =
            -5.187863000000000e+03 * invT
            +8.887228000000000e+00
            -5.612279000000000e+00 * tc[0]
            -4.224943000000000e-03 * tc[1]
            +4.756911666666667e-07 * tc[2]
            -3.531980000000000e-11 * tc[3]
            +1.134202000000000e-15 * tc[4];
        /*species 35: C3H2 */
        species[35] =
            +6.259722000000000e+04 * invT
            +2.003988100000000e+01
            -7.670981000000000e+00 * tc[0]
            -1.374374500000000e-03 * tc[1]
            +7.284905000000000e-08 * tc[2]
            +5.379665833333334e-12 * tc[3]
            -8.319435000000000e-16 * tc[4];
        /*species 36: C3H3 */
        species[36] =
            +3.847420000000000e+04 * invT
            +3.061023700000000e+01
            -8.831047000000000e+00 * tc[0]
            -2.178597500000000e-03 * tc[1]
            +6.848445000000000e-08 * tc[2]
            +1.973935833333333e-11 * tc[3]
            -2.188260000000000e-15 * tc[4];
        /*species 39: NXC3H7 */
        species[39] =
            +7.579402000000000e+03 * invT
            +2.733440100000000e+01
            -7.978291000000000e+00 * tc[0]
            -7.880565000000001e-03 * tc[1]
            +8.622071666666666e-07 * tc[2]
            -6.203243333333333e-11 * tc[3]
            +1.912489000000000e-15 * tc[4];
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
        /*species 25: CH3O2H */
        species[25] =
            -1.771979260000000e+04 * invT
            -6.021811320000000e+00
            -3.234428170000000e+00 * tc[0]
            -9.506488350000000e-03 * tc[1]
            +1.889771450000000e-06 * tc[2]
            -2.835888775000000e-10 * tc[3]
            +2.059151110000000e-14 * tc[4];
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
        /*species 25: CH3O2H */
        species[25] =
            -1.966787710000000e+04 * invT
            +2.754823381000000e+01
            -8.431170910000001e+00 * tc[0]
            -4.034089545000000e-03 * tc[1]
            +4.618248683333333e-07 * tc[2]
            -3.594435358333333e-11 * tc[3]
            +1.253460730000000e-15 * tc[4];
    }

    /*species with midpoint at T=1384 kelvin */
    if (T < 1384) {
        /*species 20: C2H6 */
        species[20] =
            -1.123455340000000e+04 * invT
            -2.119016043440000e+01
            +2.528543440000000e-02 * tc[0]
            -1.203823770000000e-02 * tc[1]
            +1.864891200000000e-06 * tc[2]
            -1.736174175000000e-10 * tc[3]
            +2.649343080000000e-15 * tc[4];
        /*species 40: NXC3H7O2 */
        species[40] =
            -7.937455670000000e+03 * invT
            -1.680095988000000e+01
            -2.107314920000000e+00 * tc[0]
            -1.980824930000000e-02 * tc[1]
            +4.158193316666667e-06 * tc[2]
            -7.162085833333334e-10 * tc[3]
            +6.562016500000000e-14 * tc[4];
    } else {
        /*species 20: C2H6 */
        species[20] =
            -1.375000140000000e+04 * invT
            +1.911495885000000e+01
            -6.106833850000000e+00 * tc[0]
            -6.461818050000000e-03 * tc[1]
            +7.375453266666666e-07 * tc[2]
            -5.728264383333333e-11 * tc[3]
            +1.994508660000000e-15 * tc[4];
        /*species 40: NXC3H7O2 */
        species[40] =
            -1.191946520000000e+04 * invT
            +5.116763560000000e+01
            -1.263270590000000e+01 * tc[0]
            -8.495536300000000e-03 * tc[1]
            +9.814447883333334e-07 * tc[2]
            -7.684961633333334e-11 * tc[3]
            +2.691153375000000e-15 * tc[4];
    }

    /*species with midpoint at T=1376 kelvin */
    if (T < 1376) {
        /*species 23: HOCHO */
        species[23] =
            -4.646165040000000e+04 * invT
            -1.585309795000000e+01
            -1.435481850000000e+00 * tc[0]
            -8.168150799999999e-03 * tc[1]
            +1.770957016666667e-06 * tc[2]
            -2.767774808333333e-10 * tc[3]
            +2.010880515000000e-14 * tc[4];
    } else {
        /*species 23: HOCHO */
        species[23] =
            -4.839954000000000e+04 * invT
            +1.799780993000000e+01
            -6.687330130000000e+00 * tc[0]
            -2.571446840000000e-03 * tc[1]
            +3.037308550000000e-07 * tc[2]
            -2.414326358333334e-11 * tc[3]
            +8.544609950000001e-16 * tc[4];
    }

    /*species with midpoint at T=1385 kelvin */
    if (T < 1385) {
        /*species 24: CH3O2 */
        species[24] =
            -6.843942590000000e+02 * invT
            -9.018341400000001e-01
            -4.261469060000000e+00 * tc[0]
            -5.043679950000000e-03 * tc[1]
            +5.358436400000000e-07 * tc[2]
            -1.745077225000000e-11 * tc[3]
            -2.091695515000000e-15 * tc[4];
        /*species 45: PXC4H9O2 */
        species[45] =
            -1.083581030000000e+04 * invT
            -1.940667840000000e+01
            -1.943636500000000e+00 * tc[0]
            -2.577565815000000e-02 * tc[1]
            +5.471406666666667e-06 * tc[2]
            -9.422071666666667e-10 * tc[3]
            +8.505930300000000e-14 * tc[4];
    } else {
        /*species 24: CH3O2 */
        species[24] =
            -1.535748380000000e+03 * invT
            +1.067751777000000e+01
            -5.957878910000000e+00 * tc[0]
            -3.953643130000000e-03 * tc[1]
            +4.470770566666667e-07 * tc[2]
            -3.449094475000000e-11 * tc[3]
            +1.195036650000000e-15 * tc[4];
        /*species 45: PXC4H9O2 */
        species[45] =
            -1.601460540000000e+04 * invT
            +6.982339730000000e+01
            -1.578454480000000e+01 * tc[0]
            -1.076054550000000e-02 * tc[1]
            +1.241515028333333e-06 * tc[2]
            -9.713172583333334e-11 * tc[3]
            +3.399428045000000e-15 * tc[4];
    }

    /*species with midpoint at T=1388 kelvin */
    if (T < 1388) {
        /*species 30: C3H6 */
        species[30] =
            +1.066881640000000e+03 * invT
            -2.150575815600000e+01
            -3.946154440000000e-01 * tc[0]
            -1.445538310000000e-02 * tc[1]
            +2.581446800000000e-06 * tc[2]
            -3.240118408333333e-10 * tc[3]
            +1.689451760000000e-14 * tc[4];
    } else {
        /*species 30: C3H6 */
        species[30] =
            -1.878212710000000e+03 * invT
            +2.803202638000000e+01
            -8.015959580000001e+00 * tc[0]
            -6.851181700000000e-03 * tc[1]
            +7.770828883333333e-07 * tc[2]
            -6.010453350000000e-11 * tc[3]
            +2.086850630000000e-15 * tc[4];
    }

    /*species with midpoint at T=1389 kelvin */
    if (T < 1389) {
        /*species 32: C2H5O */
        species[32] =
            -3.352529250000000e+03 * invT
            -2.231351709200000e+01
            -4.944207080000000e-01 * tc[0]
            -1.358872170000000e-02 * tc[1]
            +2.765150166666667e-06 * tc[2]
            -4.293368333333333e-10 * tc[3]
            +3.242484575000000e-14 * tc[4];
    } else {
        /*species 32: C2H5O */
        species[32] =
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
        /*species 34: C2H5O2 */
        species[34] =
            -5.038807580000000e+03 * invT
            -1.420893532000000e+01
            -2.268461880000000e+00 * tc[0]
            -1.384712890000000e-02 * tc[1]
            +2.846735100000000e-06 * tc[2]
            -4.899598983333333e-10 * tc[3]
            +4.604495345000000e-14 * tc[4];
        /*species 50: C7H15X2 */
        species[50] =
            -2.356053030000000e+03 * invT
            -3.377006617670000e+01
            +3.791557670000000e-02 * tc[0]
            -3.783632850000000e-02 * tc[1]
            +6.791227233333333e-06 * tc[2]
            -7.772324525000000e-10 * tc[3]
            +2.461803725000000e-14 * tc[4];
    } else {
        /*species 34: C2H5O2 */
        species[34] =
            -7.824817950000000e+03 * invT
            +3.254826223000000e+01
            -9.486960229999999e+00 * tc[0]
            -6.223627250000000e-03 * tc[1]
            +7.202692933333333e-07 * tc[2]
            -5.646525275000000e-11 * tc[3]
            +1.978922840000000e-15 * tc[4];
        /*species 50: C7H15X2 */
        species[50] =
            -1.058736160000000e+04 * invT
            +1.068578495000000e+02
            -2.163688420000000e+01 * tc[0]
            -1.616624020000000e-02 * tc[1]
            +1.821230116666667e-06 * tc[2]
            -1.402975500000000e-10 * tc[3]
            +4.858870455000000e-15 * tc[4];
    }

    /*species with midpoint at T=1400 kelvin */
    if (T < 1400) {
        /*species 37: C3H4XA */
        species[37] =
            +2.251243000000000e+04 * invT
            -7.395871000000000e+00
            -2.539831000000000e+00 * tc[0]
            -8.167185000000000e-03 * tc[1]
            +2.941583333333333e-07 * tc[2]
            +3.872804166666666e-10 * tc[3]
            -8.645655000000001e-14 * tc[4];
    } else {
        /*species 37: C3H4XA */
        species[37] =
            +1.954972000000000e+04 * invT
            +4.054686600000000e+01
            -9.776256000000000e+00 * tc[0]
            -2.651069000000000e-03 * tc[1]
            +6.168530000000000e-08 * tc[2]
            +2.521988333333334e-11 * tc[3]
            -2.544790500000000e-15 * tc[4];
    }

    /*species with midpoint at T=1397 kelvin */
    if (T < 1397) {
        /*species 38: C3H5XA */
        species[38] =
            +1.938342260000000e+04 * invT
            -2.583584505800000e+01
            +5.291319580000000e-01 * tc[0]
            -1.672795500000000e-02 * tc[1]
            +4.223350450000001e-06 * tc[2]
            -8.572146166666666e-10 * tc[3]
            +8.662917000000000e-14 * tc[4];
        /*species 48: C5H11X1 */
        species[48] =
            +4.839953030000000e+03 * invT
            -3.346275221200000e+01
            +9.052559120000000e-01 * tc[0]
            -3.053164260000000e-02 * tc[1]
            +6.824863750000000e-06 * tc[2]
            -1.217445583333333e-09 * tc[3]
            +1.094298075000000e-13 * tc[4];
    } else {
        /*species 38: C3H5XA */
        species[38] =
            +1.635760920000000e+04 * invT
            +3.103978458000000e+01
            -8.458839579999999e+00 * tc[0]
            -5.634774150000000e-03 * tc[1]
            +6.396547733333333e-07 * tc[2]
            -4.950492658333333e-11 * tc[3]
            +1.719590150000000e-15 * tc[4];
        /*species 48: C5H11X1 */
        species[48] =
            -9.232416370000000e+02 * invT
            +7.027635990000000e+01
            -1.532347400000000e+01 * tc[0]
            -1.195206000000000e-02 * tc[1]
            +1.357952698333333e-06 * tc[2]
            -1.051468633333333e-10 * tc[3]
            +3.653386675000001e-15 * tc[4];
    }

    /*species with midpoint at T=1398 kelvin */
    if (T < 1398) {
        /*species 41: C4H6 */
        species[41] =
            +1.175513140000000e+04 * invT
            -3.051353451000000e+01
            +1.430951210000000e+00 * tc[0]
            -2.393530310000000e-02 * tc[1]
            +6.924113333333333e-06 * tc[2]
            -1.596246266666667e-09 * tc[3]
            +1.785792535000000e-13 * tc[4];
    } else {
        /*species 41: C4H6 */
        species[41] =
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
        /*species 42: C4H7 */
        species[42] =
            +1.499335910000000e+04 * invT
            -2.708007795200000e+01
            +3.505083520000000e-01 * tc[0]
            -2.132556215000000e-02 * tc[1]
            +4.849656216666667e-06 * tc[2]
            -8.783659500000000e-10 * tc[3]
            +8.002992699999999e-14 * tc[4];
        /*species 43: C4H8X1 */
        species[43] =
            -1.578750350000000e+03 * invT
            -3.033979568900000e+01
            +8.313720890000000e-01 * tc[0]
            -2.262904890000000e-02 * tc[1]
            +4.894309316666667e-06 * tc[2]
            -8.351702999999999e-10 * tc[3]
            +7.159584000000000e-14 * tc[4];
        /*species 46: C5H9 */
        species[46] =
            +1.255898240000000e+04 * invT
            -3.402426990000000e+01
            +1.380139500000000e+00 * tc[0]
            -2.788042435000000e-02 * tc[1]
            +6.169065466666666e-06 * tc[2]
            -1.057365841666667e-09 * tc[3]
            +8.926941750000000e-14 * tc[4];
        /*species 47: C5H10X1 */
        species[47] =
            -4.465466660000000e+03 * invT
            -3.333621381000000e+01
            +1.062234810000000e+00 * tc[0]
            -2.871091470000000e-02 * tc[1]
            +6.241448166666667e-06 * tc[2]
            -1.061374908333333e-09 * tc[3]
            +8.980489449999999e-14 * tc[4];
        /*species 49: C6H12X1 */
        species[49] =
            -7.343686170000000e+03 * invT
            -3.666482115000000e+01
            +1.352752050000000e+00 * tc[0]
            -3.493277130000000e-02 * tc[1]
            +7.656800366666667e-06 * tc[2]
            -1.308061191666667e-09 * tc[3]
            +1.106480875000000e-13 * tc[4];
    } else {
        /*species 42: C4H7 */
        species[42] =
            +1.090419370000000e+04 * invT
            +4.676965930000000e+01
            -1.121035780000000e+01 * tc[0]
            -8.024159800000000e-03 * tc[1]
            +9.108371533333334e-07 * tc[2]
            -7.049508775000000e-11 * tc[3]
            +2.448863695000000e-15 * tc[4];
        /*species 43: C4H8X1 */
        species[43] =
            -5.978710380000000e+03 * invT
            +4.778781060000000e+01
            -1.135086680000000e+01 * tc[0]
            -9.030893850000001e-03 * tc[1]
            +1.026821715000000e-06 * tc[2]
            -7.955441325000001e-11 * tc[3]
            +2.765448205000000e-15 * tc[4];
        /*species 46: C5H9 */
        species[46] =
            +7.004961350000000e+03 * invT
            +6.563622270000000e+01
            -1.418604540000000e+01 * tc[0]
            -1.035644495000000e-02 * tc[1]
            +1.178267695000000e-06 * tc[2]
            -9.133927750000000e-11 * tc[3]
            +3.176611040000000e-15 * tc[4];
        /*species 47: C5H10X1 */
        species[47] =
            -1.008982050000000e+04 * invT
            +6.695354750000000e+01
            -1.458515390000000e+01 * tc[0]
            -1.120362355000000e-02 * tc[1]
            +1.272246708333333e-06 * tc[2]
            -9.849080500000001e-11 * tc[3]
            +3.421925695000000e-15 * tc[4];
        /*species 49: C6H12X1 */
        species[49] =
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
        /*species 44: PXC4H9 */
        species[44] =
            +7.689452480000000e+03 * invT
            -2.912305292500000e+01
            +4.377797250000000e-01 * tc[0]
            -2.394861820000000e-02 * tc[1]
            +5.233719316666667e-06 * tc[2]
            -9.148872666666667e-10 * tc[3]
            +8.100533200000001e-14 * tc[4];
    } else {
        /*species 44: PXC4H9 */
        species[44] =
            +3.172319420000000e+03 * invT
            +5.149359040000000e+01
            -1.215100820000000e+01 * tc[0]
            -9.715535850000000e-03 * tc[1]
            +1.102629916666667e-06 * tc[2]
            -8.531261333333333e-11 * tc[3]
            +2.962648535000000e-15 * tc[4];
    }

    /*species with midpoint at T=1391 kelvin */
    if (T < 1391) {
        /*species 51: NXC7H16 */
        species[51] =
            -2.565865650000000e+04 * invT
            -3.664165307000000e+01
            +1.268361870000000e+00 * tc[0]
            -4.271779100000000e-02 * tc[1]
            +8.755779766666667e-06 * tc[2]
            -1.357881008333333e-09 * tc[3]
            +1.011974625000000e-13 * tc[4];
    } else {
        /*species 51: NXC7H16 */
        species[51] =
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
        /*species 9: CH */
        species[9] =
            +7.04525900e+04 * invT
            -1.13138600e+00
            -3.20020200e+00 * tc[0]
            -1.03643800e-03 * tc[1]
            +8.55738500e-07 * tc[2]
            -4.77824167e-10 * tc[3]
            +9.77766500e-14 * tc[4];
        /*species 10: HCO */
        species[10] =
            +4.15992200e+03 * invT
            -7.08528400e+00
            -2.89833000e+00 * tc[0]
            -3.09957350e-03 * tc[1]
            +1.60384733e-06 * tc[2]
            -9.08187500e-10 * tc[3]
            +2.28744250e-13 * tc[4];
        /*species 11: CH2 */
        species[11] =
            +4.53679100e+04 * invT
            +1.04965900e+00
            -3.76223700e+00 * tc[0]
            -5.79909500e-04 * tc[1]
            -4.14930833e-08 * tc[2]
            -7.33403000e-11 * tc[3]
            +3.66621750e-14 * tc[4];
        /*species 12: CO2 */
        species[12] =
            -4.83731400e+04 * invT
            -8.91276500e+00
            -2.27572500e+00 * tc[0]
            -4.96103600e-03 * tc[1]
            +1.73485167e-06 * tc[2]
            -5.72223917e-10 * tc[3]
            +1.05864000e-13 * tc[4];
        /*species 13: CO */
        species[13] =
            -1.43105400e+04 * invT
            -2.58644500e+00
            -3.26245200e+00 * tc[0]
            -7.55970500e-04 * tc[1]
            +6.46959167e-07 * tc[2]
            -4.65162000e-10 * tc[3]
            +1.23747550e-13 * tc[4];
        /*species 14: CH2O */
        species[14] =
            -1.48654000e+04 * invT
            -1.31320890e+01
            -1.65273100e+00 * tc[0]
            -6.31572000e-03 * tc[1]
            +3.14694667e-06 * tc[2]
            -1.70835917e-09 * tc[3]
            +4.20661850e-13 * tc[4];
        /*species 15: CH2GSG */
        species[15] =
            +4.98936800e+04 * invT
            +2.91373293e+00
            -3.97126500e+00 * tc[0]
            +8.49544500e-05 * tc[1]
            -1.70894833e-07 * tc[2]
            -2.07712583e-10 * tc[3]
            +9.90633000e-14 * tc[4];
        /*species 16: CH3 */
        species[16] =
            +1.64237800e+04 * invT
            -5.35935100e+00
            -2.43044300e+00 * tc[0]
            -5.56205000e-03 * tc[1]
            +2.80036667e-06 * tc[2]
            -1.35152417e-09 * tc[3]
            +2.93247650e-13 * tc[4];
        /*species 17: CH3O */
        species[17] =
            +9.78601100e+02 * invT
            -1.20459760e+01
            -2.10620400e+00 * tc[0]
            -3.60829750e-03 * tc[1]
            -8.89745333e-07 * tc[2]
            +6.14803000e-10 * tc[3]
            -1.03780550e-13 * tc[4];
        /*species 18: CH4 */
        species[18] =
            -9.82522900e+03 * invT
            -1.39434485e+01
            -7.78741500e-01 * tc[0]
            -8.73834000e-03 * tc[1]
            +4.63901500e-06 * tc[2]
            -2.54142333e-09 * tc[3]
            +6.11965500e-13 * tc[4];
        /*species 19: CH3OH */
        species[19] =
            -2.53534800e+04 * invT
            -9.57251500e+00
            -2.66011500e+00 * tc[0]
            -3.67075400e-03 * tc[1]
            -1.19500850e-06 * tc[2]
            +7.32766167e-10 * tc[3]
            -1.19528500e-13 * tc[4];
        /*species 21: C2H5 */
        species[21] =
            +1.28704000e+04 * invT
            -1.04474980e+01
            -2.69070200e+00 * tc[0]
            -4.35956650e-03 * tc[1]
            -7.36639833e-07 * tc[2]
            -7.78225250e-11 * tc[3]
            +1.96388650e-13 * tc[4];
        /*species 22: CH2CO */
        species[22] =
            -7.63263700e+03 * invT
            -6.69858200e+00
            -2.97497100e+00 * tc[0]
            -6.05935500e-03 * tc[1]
            +3.90841000e-07 * tc[2]
            +5.38890417e-10 * tc[3]
            -1.95282450e-13 * tc[4];
        /*species 26: C2H2 */
        species[26] =
            +2.61244400e+04 * invT
            -7.79181600e+00
            -2.01356200e+00 * tc[0]
            -7.59522500e-03 * tc[1]
            +2.69386500e-06 * tc[2]
            -7.56582667e-10 * tc[3]
            +9.56373000e-14 * tc[4];
        /*species 27: HCCO */
        species[27] =
            +1.96589200e+04 * invT
            +3.56612110e+00
            -5.04796500e+00 * tc[0]
            -2.22673900e-03 * tc[1]
            -3.78047167e-08 * tc[2]
            +1.23507917e-10 * tc[3]
            -1.12537100e-14 * tc[4];
        /*species 28: C2H3 */
        species[28] =
            +3.33522500e+04 * invT
            -1.00969240e+01
            -2.45927600e+00 * tc[0]
            -3.68573800e-03 * tc[1]
            -3.51645500e-07 * tc[2]
            +1.10136833e-10 * tc[3]
            +5.92392000e-14 * tc[4];
        /*species 29: CH2CHO */
        species[29] =
            +1.52147700e+03 * invT
            -7.14922800e+00
            -3.40906200e+00 * tc[0]
            -5.36928500e-03 * tc[1]
            -3.15248667e-07 * tc[2]
            +5.96548583e-10 * tc[3]
            -1.43369250e-13 * tc[4];
        /*species 31: C2H4 */
        species[31] =
            +5.57304600e+03 * invT
            -2.60729780e+01
            +8.61488000e-01 * tc[0]
            -1.39808150e-02 * tc[1]
            +5.64779500e-06 * tc[2]
            -2.32096000e-09 * tc[3]
            +4.86893950e-13 * tc[4];
        /*species 33: CH3CO */
        species[33] =
            -4.10850800e+03 * invT
            -9.10357200e+00
            -3.12527800e+00 * tc[0]
            -4.88911000e-03 * tc[1]
            -7.53574667e-07 * tc[2]
            +7.50788500e-10 * tc[3]
            -1.59685900e-13 * tc[4];
        /*species 35: C3H2 */
        species[35] =
            +6.35042100e+04 * invT
            -6.70273200e+00
            -3.16671400e+00 * tc[0]
            -1.24128600e-02 * tc[1]
            +7.65272833e-06 * tc[2]
            -3.55668250e-09 * tc[3]
            +7.41076000e-13 * tc[4];
        /*species 36: C3H3 */
        species[36] =
            +3.98888300e+04 * invT
            +3.16874510e+00
            -4.75420000e+00 * tc[0]
            -5.54014000e-03 * tc[1]
            -4.65553833e-08 * tc[2]
            +4.56601000e-10 * tc[3]
            -9.74814500e-14 * tc[4];
        /*species 39: NXC3H7 */
        species[39] =
            +9.71328100e+03 * invT
            -1.30701730e+01
            -1.92253700e+00 * tc[0]
            -1.23946350e-02 * tc[1]
            -3.01708167e-07 * tc[2]
            +1.48605500e-09 * tc[3]
            -4.29149800e-13 * tc[4];
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
        /*species 9: CH */
        species[9] =
            +7.08672300e+04 * invT
            -7.98215000e+00
            -2.19622300e+00 * tc[0]
            -1.17019050e-03 * tc[1]
            +1.17636683e-07 * tc[2]
            -7.50631833e-12 * tc[3]
            +1.92752000e-16 * tc[4];
        /*species 10: HCO */
        species[10] =
            +3.91632400e+03 * invT
            -2.99502800e+00
            -3.55727100e+00 * tc[0]
            -1.67278650e-03 * tc[1]
            +2.22501000e-07 * tc[2]
            -2.05881083e-11 * tc[3]
            +8.56925500e-16 * tc[4];
        /*species 11: CH2 */
        species[11] =
            +4.53413400e+04 * invT
            +4.79847000e-01
            -3.63640800e+00 * tc[0]
            -9.66528500e-04 * tc[1]
            +2.81169333e-08 * tc[2]
            +8.41582500e-12 * tc[3]
            -9.04128000e-16 * tc[4];
        /*species 12: CO2 */
        species[12] =
            -4.89669600e+04 * invT
            +4.40901890e+00
            -4.45362300e+00 * tc[0]
            -1.57008450e-03 * tc[1]
            +2.13068500e-07 * tc[2]
            -1.99499750e-11 * tc[3]
            +8.34516500e-16 * tc[4];
        /*species 13: CO */
        species[13] =
            -1.42683500e+04 * invT
            -4.08314000e+00
            -3.02507800e+00 * tc[0]
            -7.21344500e-04 * tc[1]
            +9.38471333e-08 * tc[2]
            -8.48817500e-12 * tc[3]
            +3.45547600e-16 * tc[4];
        /*species 14: CH2O */
        species[14] =
            -1.53203700e+04 * invT
            -4.91696600e+00
            -2.99560600e+00 * tc[0]
            -3.34066050e-03 * tc[1]
            +4.38159167e-07 * tc[2]
            -3.94762750e-11 * tc[3]
            +1.60625850e-15 * tc[4];
        /*species 15: CH2GSG */
        species[15] =
            +4.98497500e+04 * invT
            +8.66319000e-01
            -3.55288900e+00 * tc[0]
            -1.03339400e-03 * tc[1]
            +3.19019333e-08 * tc[2]
            +9.20560833e-12 * tc[3]
            -1.01067500e-15 * tc[4];
        /*species 16: CH3 */
        species[16] =
            +1.64378100e+04 * invT
            -3.60864500e+00
            -2.84405200e+00 * tc[0]
            -3.06898700e-03 * tc[1]
            +3.71724167e-07 * tc[2]
            -3.15430083e-11 * tc[3]
            +1.22607950e-15 * tc[4];
        /*species 17: CH3O */
        species[17] =
            +1.27832500e+02 * invT
            -1.58775000e-01
            -3.77080000e+00 * tc[0]
            -3.93574850e-03 * tc[1]
            +4.42730667e-07 * tc[2]
            -3.28702583e-11 * tc[3]
            +1.05630800e-15 * tc[4];
        /*species 18: CH4 */
        species[18] =
            -1.00807900e+04 * invT
            -8.93991600e+00
            -1.68347900e+00 * tc[0]
            -5.11862000e-03 * tc[1]
            +6.45854833e-07 * tc[2]
            -5.65465417e-11 * tc[3]
            +2.25171150e-15 * tc[4];
        /*species 19: CH3OH */
        species[19] =
            -2.61579100e+04 * invT
            +6.50865000e-01
            -4.02906100e+00 * tc[0]
            -4.68829650e-03 * tc[1]
            +5.08375667e-07 * tc[2]
            -3.63232750e-11 * tc[3]
            +1.11236150e-15 * tc[4];
        /*species 21: C2H5 */
        species[21] =
            +1.06745500e+04 * invT
            +2.09713700e+01
            -7.19048000e+00 * tc[0]
            -3.24203850e-03 * tc[1]
            +1.07134417e-07 * tc[2]
            +1.95656583e-11 * tc[3]
            -1.94043850e-15 * tc[4];
        /*species 22: CH2CO */
        species[22] =
            -8.58340200e+03 * invT
            +1.26963980e+01
            -6.03881700e+00 * tc[0]
            -2.90242000e-03 * tc[1]
            +3.20159000e-07 * tc[2]
            -2.32873750e-11 * tc[3]
            +7.29434000e-16 * tc[4];
        /*species 26: C2H2 */
        species[26] =
            +2.56676600e+04 * invT
            +6.23710800e+00
            -4.43677000e+00 * tc[0]
            -2.68801950e-03 * tc[1]
            +3.18802833e-07 * tc[2]
            -2.73864917e-11 * tc[3]
            +1.07835500e-15 * tc[4];
        /*species 27: HCCO */
        species[27] =
            +1.90151300e+04 * invT
            +1.48293350e+01
            -6.75807300e+00 * tc[0]
            -1.00020000e-03 * tc[1]
            +3.37934500e-08 * tc[2]
            +8.67610000e-12 * tc[3]
            -9.82582500e-16 * tc[4];
        /*species 28: C2H3 */
        species[28] =
            +3.18543500e+04 * invT
            +1.34637810e+01
            -5.93346800e+00 * tc[0]
            -2.00887300e-03 * tc[1]
            +6.61123333e-08 * tc[2]
            +1.20105583e-11 * tc[3]
            -1.18932200e-15 * tc[4];
        /*species 29: CH2CHO */
        species[29] =
            +4.90321800e+02 * invT
            +1.00209210e+01
            -5.97567000e+00 * tc[0]
            -4.06529550e-03 * tc[1]
            +4.57270667e-07 * tc[2]
            -3.39192000e-11 * tc[3]
            +1.08800850e-15 * tc[4];
        /*species 31: C2H4 */
        species[31] =
            +4.42828900e+03 * invT
            +2.98030000e-01
            -3.52841900e+00 * tc[0]
            -5.74259000e-03 * tc[1]
            +7.36397500e-07 * tc[2]
            -6.53716750e-11 * tc[3]
            +2.63342400e-15 * tc[4];
        /*species 33: CH3CO */
        species[33] =
            -5.18786300e+03 * invT
            +7.88722800e+00
            -5.61227900e+00 * tc[0]
            -4.22494300e-03 * tc[1]
            +4.75691167e-07 * tc[2]
            -3.53198000e-11 * tc[3]
            +1.13420200e-15 * tc[4];
        /*species 35: C3H2 */
        species[35] =
            +6.25972200e+04 * invT
            +1.90398810e+01
            -7.67098100e+00 * tc[0]
            -1.37437450e-03 * tc[1]
            +7.28490500e-08 * tc[2]
            +5.37966583e-12 * tc[3]
            -8.31943500e-16 * tc[4];
        /*species 36: C3H3 */
        species[36] =
            +3.84742000e+04 * invT
            +2.96102370e+01
            -8.83104700e+00 * tc[0]
            -2.17859750e-03 * tc[1]
            +6.84844500e-08 * tc[2]
            +1.97393583e-11 * tc[3]
            -2.18826000e-15 * tc[4];
        /*species 39: NXC3H7 */
        species[39] =
            +7.57940200e+03 * invT
            +2.63344010e+01
            -7.97829100e+00 * tc[0]
            -7.88056500e-03 * tc[1]
            +8.62207167e-07 * tc[2]
            -6.20324333e-11 * tc[3]
            +1.91248900e-15 * tc[4];
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
        /*species 25: CH3O2H */
        species[25] =
            -1.77197926e+04 * invT
            -7.02181132e+00
            -3.23442817e+00 * tc[0]
            -9.50648835e-03 * tc[1]
            +1.88977145e-06 * tc[2]
            -2.83588878e-10 * tc[3]
            +2.05915111e-14 * tc[4];
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
        /*species 25: CH3O2H */
        species[25] =
            -1.96678771e+04 * invT
            +2.65482338e+01
            -8.43117091e+00 * tc[0]
            -4.03408955e-03 * tc[1]
            +4.61824868e-07 * tc[2]
            -3.59443536e-11 * tc[3]
            +1.25346073e-15 * tc[4];
    }

    /*species with midpoint at T=1384 kelvin */
    if (T < 1384) {
        /*species 20: C2H6 */
        species[20] =
            -1.12345534e+04 * invT
            -2.21901604e+01
            +2.52854344e-02 * tc[0]
            -1.20382377e-02 * tc[1]
            +1.86489120e-06 * tc[2]
            -1.73617417e-10 * tc[3]
            +2.64934308e-15 * tc[4];
        /*species 40: NXC3H7O2 */
        species[40] =
            -7.93745567e+03 * invT
            -1.78009599e+01
            -2.10731492e+00 * tc[0]
            -1.98082493e-02 * tc[1]
            +4.15819332e-06 * tc[2]
            -7.16208583e-10 * tc[3]
            +6.56201650e-14 * tc[4];
    } else {
        /*species 20: C2H6 */
        species[20] =
            -1.37500014e+04 * invT
            +1.81149589e+01
            -6.10683385e+00 * tc[0]
            -6.46181805e-03 * tc[1]
            +7.37545327e-07 * tc[2]
            -5.72826438e-11 * tc[3]
            +1.99450866e-15 * tc[4];
        /*species 40: NXC3H7O2 */
        species[40] =
            -1.19194652e+04 * invT
            +5.01676356e+01
            -1.26327059e+01 * tc[0]
            -8.49553630e-03 * tc[1]
            +9.81444788e-07 * tc[2]
            -7.68496163e-11 * tc[3]
            +2.69115337e-15 * tc[4];
    }

    /*species with midpoint at T=1376 kelvin */
    if (T < 1376) {
        /*species 23: HOCHO */
        species[23] =
            -4.64616504e+04 * invT
            -1.68530979e+01
            -1.43548185e+00 * tc[0]
            -8.16815080e-03 * tc[1]
            +1.77095702e-06 * tc[2]
            -2.76777481e-10 * tc[3]
            +2.01088051e-14 * tc[4];
    } else {
        /*species 23: HOCHO */
        species[23] =
            -4.83995400e+04 * invT
            +1.69978099e+01
            -6.68733013e+00 * tc[0]
            -2.57144684e-03 * tc[1]
            +3.03730855e-07 * tc[2]
            -2.41432636e-11 * tc[3]
            +8.54460995e-16 * tc[4];
    }

    /*species with midpoint at T=1385 kelvin */
    if (T < 1385) {
        /*species 24: CH3O2 */
        species[24] =
            -6.84394259e+02 * invT
            -1.90183414e+00
            -4.26146906e+00 * tc[0]
            -5.04367995e-03 * tc[1]
            +5.35843640e-07 * tc[2]
            -1.74507723e-11 * tc[3]
            -2.09169552e-15 * tc[4];
        /*species 45: PXC4H9O2 */
        species[45] =
            -1.08358103e+04 * invT
            -2.04066784e+01
            -1.94363650e+00 * tc[0]
            -2.57756581e-02 * tc[1]
            +5.47140667e-06 * tc[2]
            -9.42207167e-10 * tc[3]
            +8.50593030e-14 * tc[4];
    } else {
        /*species 24: CH3O2 */
        species[24] =
            -1.53574838e+03 * invT
            +9.67751777e+00
            -5.95787891e+00 * tc[0]
            -3.95364313e-03 * tc[1]
            +4.47077057e-07 * tc[2]
            -3.44909447e-11 * tc[3]
            +1.19503665e-15 * tc[4];
        /*species 45: PXC4H9O2 */
        species[45] =
            -1.60146054e+04 * invT
            +6.88233973e+01
            -1.57845448e+01 * tc[0]
            -1.07605455e-02 * tc[1]
            +1.24151503e-06 * tc[2]
            -9.71317258e-11 * tc[3]
            +3.39942805e-15 * tc[4];
    }

    /*species with midpoint at T=1388 kelvin */
    if (T < 1388) {
        /*species 30: C3H6 */
        species[30] =
            +1.06688164e+03 * invT
            -2.25057582e+01
            -3.94615444e-01 * tc[0]
            -1.44553831e-02 * tc[1]
            +2.58144680e-06 * tc[2]
            -3.24011841e-10 * tc[3]
            +1.68945176e-14 * tc[4];
    } else {
        /*species 30: C3H6 */
        species[30] =
            -1.87821271e+03 * invT
            +2.70320264e+01
            -8.01595958e+00 * tc[0]
            -6.85118170e-03 * tc[1]
            +7.77082888e-07 * tc[2]
            -6.01045335e-11 * tc[3]
            +2.08685063e-15 * tc[4];
    }

    /*species with midpoint at T=1389 kelvin */
    if (T < 1389) {
        /*species 32: C2H5O */
        species[32] =
            -3.35252925e+03 * invT
            -2.33135171e+01
            -4.94420708e-01 * tc[0]
            -1.35887217e-02 * tc[1]
            +2.76515017e-06 * tc[2]
            -4.29336833e-10 * tc[3]
            +3.24248457e-14 * tc[4];
    } else {
        /*species 32: C2H5O */
        species[32] =
            -6.07274953e+03 * invT
            +2.42150767e+01
            -7.87339772e+00 * tc[0]
            -5.65364535e-03 * tc[1]
            +6.40702368e-07 * tc[2]
            -4.95345088e-11 * tc[3]
            +1.71947269e-15 * tc[4];
    }

    /*species with midpoint at T=1382 kelvin */
    if (T < 1382) {
        /*species 34: C2H5O2 */
        species[34] =
            -5.03880758e+03 * invT
            -1.52089353e+01
            -2.26846188e+00 * tc[0]
            -1.38471289e-02 * tc[1]
            +2.84673510e-06 * tc[2]
            -4.89959898e-10 * tc[3]
            +4.60449534e-14 * tc[4];
        /*species 50: C7H15X2 */
        species[50] =
            -2.35605303e+03 * invT
            -3.47700662e+01
            +3.79155767e-02 * tc[0]
            -3.78363285e-02 * tc[1]
            +6.79122723e-06 * tc[2]
            -7.77232453e-10 * tc[3]
            +2.46180373e-14 * tc[4];
    } else {
        /*species 34: C2H5O2 */
        species[34] =
            -7.82481795e+03 * invT
            +3.15482622e+01
            -9.48696023e+00 * tc[0]
            -6.22362725e-03 * tc[1]
            +7.20269293e-07 * tc[2]
            -5.64652528e-11 * tc[3]
            +1.97892284e-15 * tc[4];
        /*species 50: C7H15X2 */
        species[50] =
            -1.05873616e+04 * invT
            +1.05857850e+02
            -2.16368842e+01 * tc[0]
            -1.61662402e-02 * tc[1]
            +1.82123012e-06 * tc[2]
            -1.40297550e-10 * tc[3]
            +4.85887045e-15 * tc[4];
    }

    /*species with midpoint at T=1400 kelvin */
    if (T < 1400) {
        /*species 37: C3H4XA */
        species[37] =
            +2.25124300e+04 * invT
            -8.39587100e+00
            -2.53983100e+00 * tc[0]
            -8.16718500e-03 * tc[1]
            +2.94158333e-07 * tc[2]
            +3.87280417e-10 * tc[3]
            -8.64565500e-14 * tc[4];
    } else {
        /*species 37: C3H4XA */
        species[37] =
            +1.95497200e+04 * invT
            +3.95468660e+01
            -9.77625600e+00 * tc[0]
            -2.65106900e-03 * tc[1]
            +6.16853000e-08 * tc[2]
            +2.52198833e-11 * tc[3]
            -2.54479050e-15 * tc[4];
    }

    /*species with midpoint at T=1397 kelvin */
    if (T < 1397) {
        /*species 38: C3H5XA */
        species[38] =
            +1.93834226e+04 * invT
            -2.68358451e+01
            +5.29131958e-01 * tc[0]
            -1.67279550e-02 * tc[1]
            +4.22335045e-06 * tc[2]
            -8.57214617e-10 * tc[3]
            +8.66291700e-14 * tc[4];
        /*species 48: C5H11X1 */
        species[48] =
            +4.83995303e+03 * invT
            -3.44627522e+01
            +9.05255912e-01 * tc[0]
            -3.05316426e-02 * tc[1]
            +6.82486375e-06 * tc[2]
            -1.21744558e-09 * tc[3]
            +1.09429807e-13 * tc[4];
    } else {
        /*species 38: C3H5XA */
        species[38] =
            +1.63576092e+04 * invT
            +3.00397846e+01
            -8.45883958e+00 * tc[0]
            -5.63477415e-03 * tc[1]
            +6.39654773e-07 * tc[2]
            -4.95049266e-11 * tc[3]
            +1.71959015e-15 * tc[4];
        /*species 48: C5H11X1 */
        species[48] =
            -9.23241637e+02 * invT
            +6.92763599e+01
            -1.53234740e+01 * tc[0]
            -1.19520600e-02 * tc[1]
            +1.35795270e-06 * tc[2]
            -1.05146863e-10 * tc[3]
            +3.65338668e-15 * tc[4];
    }

    /*species with midpoint at T=1398 kelvin */
    if (T < 1398) {
        /*species 41: C4H6 */
        species[41] =
            +1.17551314e+04 * invT
            -3.15135345e+01
            +1.43095121e+00 * tc[0]
            -2.39353031e-02 * tc[1]
            +6.92411333e-06 * tc[2]
            -1.59624627e-09 * tc[3]
            +1.78579253e-13 * tc[4];
    } else {
        /*species 41: C4H6 */
        species[41] =
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
        /*species 42: C4H7 */
        species[42] =
            +1.49933591e+04 * invT
            -2.80800780e+01
            +3.50508352e-01 * tc[0]
            -2.13255622e-02 * tc[1]
            +4.84965622e-06 * tc[2]
            -8.78365950e-10 * tc[3]
            +8.00299270e-14 * tc[4];
        /*species 43: C4H8X1 */
        species[43] =
            -1.57875035e+03 * invT
            -3.13397957e+01
            +8.31372089e-01 * tc[0]
            -2.26290489e-02 * tc[1]
            +4.89430932e-06 * tc[2]
            -8.35170300e-10 * tc[3]
            +7.15958400e-14 * tc[4];
        /*species 46: C5H9 */
        species[46] =
            +1.25589824e+04 * invT
            -3.50242699e+01
            +1.38013950e+00 * tc[0]
            -2.78804243e-02 * tc[1]
            +6.16906547e-06 * tc[2]
            -1.05736584e-09 * tc[3]
            +8.92694175e-14 * tc[4];
        /*species 47: C5H10X1 */
        species[47] =
            -4.46546666e+03 * invT
            -3.43362138e+01
            +1.06223481e+00 * tc[0]
            -2.87109147e-02 * tc[1]
            +6.24144817e-06 * tc[2]
            -1.06137491e-09 * tc[3]
            +8.98048945e-14 * tc[4];
        /*species 49: C6H12X1 */
        species[49] =
            -7.34368617e+03 * invT
            -3.76648212e+01
            +1.35275205e+00 * tc[0]
            -3.49327713e-02 * tc[1]
            +7.65680037e-06 * tc[2]
            -1.30806119e-09 * tc[3]
            +1.10648088e-13 * tc[4];
    } else {
        /*species 42: C4H7 */
        species[42] =
            +1.09041937e+04 * invT
            +4.57696593e+01
            -1.12103578e+01 * tc[0]
            -8.02415980e-03 * tc[1]
            +9.10837153e-07 * tc[2]
            -7.04950877e-11 * tc[3]
            +2.44886369e-15 * tc[4];
        /*species 43: C4H8X1 */
        species[43] =
            -5.97871038e+03 * invT
            +4.67878106e+01
            -1.13508668e+01 * tc[0]
            -9.03089385e-03 * tc[1]
            +1.02682171e-06 * tc[2]
            -7.95544133e-11 * tc[3]
            +2.76544820e-15 * tc[4];
        /*species 46: C5H9 */
        species[46] =
            +7.00496135e+03 * invT
            +6.46362227e+01
            -1.41860454e+01 * tc[0]
            -1.03564449e-02 * tc[1]
            +1.17826770e-06 * tc[2]
            -9.13392775e-11 * tc[3]
            +3.17661104e-15 * tc[4];
        /*species 47: C5H10X1 */
        species[47] =
            -1.00898205e+04 * invT
            +6.59535475e+01
            -1.45851539e+01 * tc[0]
            -1.12036235e-02 * tc[1]
            +1.27224671e-06 * tc[2]
            -9.84908050e-11 * tc[3]
            +3.42192569e-15 * tc[4];
        /*species 49: C6H12X1 */
        species[49] =
            -1.42062860e+04 * invT
            +8.52156380e+01
            -1.78337529e+01 * tc[0]
            -1.33688829e-02 * tc[1]
            +1.51672796e-06 * tc[2]
            -1.17349807e-10 * tc[3]
            +4.07562122e-15 * tc[4];
    }

    /*species with midpoint at T=1395 kelvin */
    if (T < 1395) {
        /*species 44: PXC4H9 */
        species[44] =
            +7.68945248e+03 * invT
            -3.01230529e+01
            +4.37779725e-01 * tc[0]
            -2.39486182e-02 * tc[1]
            +5.23371932e-06 * tc[2]
            -9.14887267e-10 * tc[3]
            +8.10053320e-14 * tc[4];
    } else {
        /*species 44: PXC4H9 */
        species[44] =
            +3.17231942e+03 * invT
            +5.04935904e+01
            -1.21510082e+01 * tc[0]
            -9.71553585e-03 * tc[1]
            +1.10262992e-06 * tc[2]
            -8.53126133e-11 * tc[3]
            +2.96264853e-15 * tc[4];
    }

    /*species with midpoint at T=1391 kelvin */
    if (T < 1391) {
        /*species 51: NXC7H16 */
        species[51] =
            -2.56586565e+04 * invT
            -3.76416531e+01
            +1.26836187e+00 * tc[0]
            -4.27177910e-02 * tc[1]
            +8.75577977e-06 * tc[2]
            -1.35788101e-09 * tc[3]
            +1.01197462e-13 * tc[4];
    } else {
        /*species 51: NXC7H16 */
        species[51] =
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
        /*species 9: CH */
        species[9] =
            +2.20020200e+00
            +2.07287600e-03 * tc[1]
            -5.13443100e-06 * tc[2]
            +5.73389000e-09 * tc[3]
            -1.95553300e-12 * tc[4];
        /*species 10: HCO */
        species[10] =
            +1.89833000e+00
            +6.19914700e-03 * tc[1]
            -9.62308400e-06 * tc[2]
            +1.08982500e-08 * tc[3]
            -4.57488500e-12 * tc[4];
        /*species 11: CH2 */
        species[11] =
            +2.76223700e+00
            +1.15981900e-03 * tc[1]
            +2.48958500e-07 * tc[2]
            +8.80083600e-10 * tc[3]
            -7.33243500e-13 * tc[4];
        /*species 12: CO2 */
        species[12] =
            +1.27572500e+00
            +9.92207200e-03 * tc[1]
            -1.04091100e-05 * tc[2]
            +6.86668700e-09 * tc[3]
            -2.11728000e-12 * tc[4];
        /*species 13: CO */
        species[13] =
            +2.26245200e+00
            +1.51194100e-03 * tc[1]
            -3.88175500e-06 * tc[2]
            +5.58194400e-09 * tc[3]
            -2.47495100e-12 * tc[4];
        /*species 14: CH2O */
        species[14] =
            +6.52731000e-01
            +1.26314400e-02 * tc[1]
            -1.88816800e-05 * tc[2]
            +2.05003100e-08 * tc[3]
            -8.41323700e-12 * tc[4];
        /*species 15: CH2GSG */
        species[15] =
            +2.97126500e+00
            -1.69908900e-04 * tc[1]
            +1.02536900e-06 * tc[2]
            +2.49255100e-09 * tc[3]
            -1.98126600e-12 * tc[4];
        /*species 16: CH3 */
        species[16] =
            +1.43044300e+00
            +1.11241000e-02 * tc[1]
            -1.68022000e-05 * tc[2]
            +1.62182900e-08 * tc[3]
            -5.86495300e-12 * tc[4];
        /*species 17: CH3O */
        species[17] =
            +1.10620400e+00
            +7.21659500e-03 * tc[1]
            +5.33847200e-06 * tc[2]
            -7.37763600e-09 * tc[3]
            +2.07561100e-12 * tc[4];
        /*species 18: CH4 */
        species[18] =
            -2.21258500e-01
            +1.74766800e-02 * tc[1]
            -2.78340900e-05 * tc[2]
            +3.04970800e-08 * tc[3]
            -1.22393100e-11 * tc[4];
        /*species 19: CH3OH */
        species[19] =
            +1.66011500e+00
            +7.34150800e-03 * tc[1]
            +7.17005100e-06 * tc[2]
            -8.79319400e-09 * tc[3]
            +2.39057000e-12 * tc[4];
        /*species 21: C2H5 */
        species[21] =
            +1.69070200e+00
            +8.71913300e-03 * tc[1]
            +4.41983900e-06 * tc[2]
            +9.33870300e-10 * tc[3]
            -3.92777300e-12 * tc[4];
        /*species 22: CH2CO */
        species[22] =
            +1.97497100e+00
            +1.21187100e-02 * tc[1]
            -2.34504600e-06 * tc[2]
            -6.46668500e-09 * tc[3]
            +3.90564900e-12 * tc[4];
        /*species 26: C2H2 */
        species[26] =
            +1.01356200e+00
            +1.51904500e-02 * tc[1]
            -1.61631900e-05 * tc[2]
            +9.07899200e-09 * tc[3]
            -1.91274600e-12 * tc[4];
        /*species 27: HCCO */
        species[27] =
            +4.04796500e+00
            +4.45347800e-03 * tc[1]
            +2.26828300e-07 * tc[2]
            -1.48209500e-09 * tc[3]
            +2.25074200e-13 * tc[4];
        /*species 28: C2H3 */
        species[28] =
            +1.45927600e+00
            +7.37147600e-03 * tc[1]
            +2.10987300e-06 * tc[2]
            -1.32164200e-09 * tc[3]
            -1.18478400e-12 * tc[4];
        /*species 29: CH2CHO */
        species[29] =
            +2.40906200e+00
            +1.07385700e-02 * tc[1]
            +1.89149200e-06 * tc[2]
            -7.15858300e-09 * tc[3]
            +2.86738500e-12 * tc[4];
        /*species 31: C2H4 */
        species[31] =
            -1.86148800e+00
            +2.79616300e-02 * tc[1]
            -3.38867700e-05 * tc[2]
            +2.78515200e-08 * tc[3]
            -9.73787900e-12 * tc[4];
        /*species 33: CH3CO */
        species[33] =
            +2.12527800e+00
            +9.77822000e-03 * tc[1]
            +4.52144800e-06 * tc[2]
            -9.00946200e-09 * tc[3]
            +3.19371800e-12 * tc[4];
        /*species 35: C3H2 */
        species[35] =
            +2.16671400e+00
            +2.48257200e-02 * tc[1]
            -4.59163700e-05 * tc[2]
            +4.26801900e-08 * tc[3]
            -1.48215200e-11 * tc[4];
        /*species 36: C3H3 */
        species[36] =
            +3.75420000e+00
            +1.10802800e-02 * tc[1]
            +2.79332300e-07 * tc[2]
            -5.47921200e-09 * tc[3]
            +1.94962900e-12 * tc[4];
        /*species 39: NXC3H7 */
        species[39] =
            +9.22537000e-01
            +2.47892700e-02 * tc[1]
            +1.81024900e-06 * tc[2]
            -1.78326600e-08 * tc[3]
            +8.58299600e-12 * tc[4];
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
        /*species 9: CH */
        species[9] =
            +1.19622300e+00
            +2.34038100e-03 * tc[1]
            -7.05820100e-07 * tc[2]
            +9.00758200e-11 * tc[3]
            -3.85504000e-15 * tc[4];
        /*species 10: HCO */
        species[10] =
            +2.55727100e+00
            +3.34557300e-03 * tc[1]
            -1.33500600e-06 * tc[2]
            +2.47057300e-10 * tc[3]
            -1.71385100e-14 * tc[4];
        /*species 11: CH2 */
        species[11] =
            +2.63640800e+00
            +1.93305700e-03 * tc[1]
            -1.68701600e-07 * tc[2]
            -1.00989900e-10 * tc[3]
            +1.80825600e-14 * tc[4];
        /*species 12: CO2 */
        species[12] =
            +3.45362300e+00
            +3.14016900e-03 * tc[1]
            -1.27841100e-06 * tc[2]
            +2.39399700e-10 * tc[3]
            -1.66903300e-14 * tc[4];
        /*species 13: CO */
        species[13] =
            +2.02507800e+00
            +1.44268900e-03 * tc[1]
            -5.63082800e-07 * tc[2]
            +1.01858100e-10 * tc[3]
            -6.91095200e-15 * tc[4];
        /*species 14: CH2O */
        species[14] =
            +1.99560600e+00
            +6.68132100e-03 * tc[1]
            -2.62895500e-06 * tc[2]
            +4.73715300e-10 * tc[3]
            -3.21251700e-14 * tc[4];
        /*species 15: CH2GSG */
        species[15] =
            +2.55288900e+00
            +2.06678800e-03 * tc[1]
            -1.91411600e-07 * tc[2]
            -1.10467300e-10 * tc[3]
            +2.02135000e-14 * tc[4];
        /*species 16: CH3 */
        species[16] =
            +1.84405200e+00
            +6.13797400e-03 * tc[1]
            -2.23034500e-06 * tc[2]
            +3.78516100e-10 * tc[3]
            -2.45215900e-14 * tc[4];
        /*species 17: CH3O */
        species[17] =
            +2.77080000e+00
            +7.87149700e-03 * tc[1]
            -2.65638400e-06 * tc[2]
            +3.94443100e-10 * tc[3]
            -2.11261600e-14 * tc[4];
        /*species 18: CH4 */
        species[18] =
            +6.83479000e-01
            +1.02372400e-02 * tc[1]
            -3.87512900e-06 * tc[2]
            +6.78558500e-10 * tc[3]
            -4.50342300e-14 * tc[4];
        /*species 19: CH3OH */
        species[19] =
            +3.02906100e+00
            +9.37659300e-03 * tc[1]
            -3.05025400e-06 * tc[2]
            +4.35879300e-10 * tc[3]
            -2.22472300e-14 * tc[4];
        /*species 21: C2H5 */
        species[21] =
            +6.19048000e+00
            +6.48407700e-03 * tc[1]
            -6.42806500e-07 * tc[2]
            -2.34787900e-10 * tc[3]
            +3.88087700e-14 * tc[4];
        /*species 22: CH2CO */
        species[22] =
            +5.03881700e+00
            +5.80484000e-03 * tc[1]
            -1.92095400e-06 * tc[2]
            +2.79448500e-10 * tc[3]
            -1.45886800e-14 * tc[4];
        /*species 26: C2H2 */
        species[26] =
            +3.43677000e+00
            +5.37603900e-03 * tc[1]
            -1.91281700e-06 * tc[2]
            +3.28637900e-10 * tc[3]
            -2.15671000e-14 * tc[4];
        /*species 27: HCCO */
        species[27] =
            +5.75807300e+00
            +2.00040000e-03 * tc[1]
            -2.02760700e-07 * tc[2]
            -1.04113200e-10 * tc[3]
            +1.96516500e-14 * tc[4];
        /*species 28: C2H3 */
        species[28] =
            +4.93346800e+00
            +4.01774600e-03 * tc[1]
            -3.96674000e-07 * tc[2]
            -1.44126700e-10 * tc[3]
            +2.37864400e-14 * tc[4];
        /*species 29: CH2CHO */
        species[29] =
            +4.97567000e+00
            +8.13059100e-03 * tc[1]
            -2.74362400e-06 * tc[2]
            +4.07030400e-10 * tc[3]
            -2.17601700e-14 * tc[4];
        /*species 31: C2H4 */
        species[31] =
            +2.52841900e+00
            +1.14851800e-02 * tc[1]
            -4.41838500e-06 * tc[2]
            +7.84460100e-10 * tc[3]
            -5.26684800e-14 * tc[4];
        /*species 33: CH3CO */
        species[33] =
            +4.61227900e+00
            +8.44988600e-03 * tc[1]
            -2.85414700e-06 * tc[2]
            +4.23837600e-10 * tc[3]
            -2.26840400e-14 * tc[4];
        /*species 35: C3H2 */
        species[35] =
            +6.67098100e+00
            +2.74874900e-03 * tc[1]
            -4.37094300e-07 * tc[2]
            -6.45559900e-11 * tc[3]
            +1.66388700e-14 * tc[4];
        /*species 36: C3H3 */
        species[36] =
            +7.83104700e+00
            +4.35719500e-03 * tc[1]
            -4.10906700e-07 * tc[2]
            -2.36872300e-10 * tc[3]
            +4.37652000e-14 * tc[4];
        /*species 39: NXC3H7 */
        species[39] =
            +6.97829100e+00
            +1.57611300e-02 * tc[1]
            -5.17324300e-06 * tc[2]
            +7.44389200e-10 * tc[3]
            -3.82497800e-14 * tc[4];
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
        /*species 25: CH3O2H */
        species[25] =
            +2.23442817e+00
            +1.90129767e-02 * tc[1]
            -1.13386287e-05 * tc[2]
            +3.40306653e-09 * tc[3]
            -4.11830222e-13 * tc[4];
    } else {
        /*species 7: HO2 */
        species[7] =
            +3.10547423e+00
            +2.38452835e-03 * tc[1]
            -8.06347989e-07 * tc[2]
            +1.24191723e-10 * tc[3]
            -7.16400108e-15 * tc[4];
        /*species 25: CH3O2H */
        species[25] =
            +7.43117091e+00
            +8.06817909e-03 * tc[1]
            -2.77094921e-06 * tc[2]
            +4.31332243e-10 * tc[3]
            -2.50692146e-14 * tc[4];
    }

    /*species with midpoint at T=1384 kelvin */
    if (T < 1384) {
        /*species 20: C2H6 */
        species[20] =
            -1.02528543e+00
            +2.40764754e-02 * tc[1]
            -1.11893472e-05 * tc[2]
            +2.08340901e-09 * tc[3]
            -5.29868616e-14 * tc[4];
        /*species 40: NXC3H7O2 */
        species[40] =
            +1.10731492e+00
            +3.96164986e-02 * tc[1]
            -2.49491599e-05 * tc[2]
            +8.59450300e-09 * tc[3]
            -1.31240330e-12 * tc[4];
    } else {
        /*species 20: C2H6 */
        species[20] =
            +5.10683385e+00
            +1.29236361e-02 * tc[1]
            -4.42527196e-06 * tc[2]
            +6.87391726e-10 * tc[3]
            -3.98901732e-14 * tc[4];
        /*species 40: NXC3H7O2 */
        species[40] =
            +1.16327059e+01
            +1.69910726e-02 * tc[1]
            -5.88866873e-06 * tc[2]
            +9.22195396e-10 * tc[3]
            -5.38230675e-14 * tc[4];
    }

    /*species with midpoint at T=1376 kelvin */
    if (T < 1376) {
        /*species 23: HOCHO */
        species[23] =
            +4.35481850e-01
            +1.63363016e-02 * tc[1]
            -1.06257421e-05 * tc[2]
            +3.32132977e-09 * tc[3]
            -4.02176103e-13 * tc[4];
    } else {
        /*species 23: HOCHO */
        species[23] =
            +5.68733013e+00
            +5.14289368e-03 * tc[1]
            -1.82238513e-06 * tc[2]
            +2.89719163e-10 * tc[3]
            -1.70892199e-14 * tc[4];
    }

    /*species with midpoint at T=1385 kelvin */
    if (T < 1385) {
        /*species 24: CH3O2 */
        species[24] =
            +3.26146906e+00
            +1.00873599e-02 * tc[1]
            -3.21506184e-06 * tc[2]
            +2.09409267e-10 * tc[3]
            +4.18339103e-14 * tc[4];
        /*species 45: PXC4H9O2 */
        species[45] =
            +9.43636500e-01
            +5.15513163e-02 * tc[1]
            -3.28284400e-05 * tc[2]
            +1.13064860e-08 * tc[3]
            -1.70118606e-12 * tc[4];
    } else {
        /*species 24: CH3O2 */
        species[24] =
            +4.95787891e+00
            +7.90728626e-03 * tc[1]
            -2.68246234e-06 * tc[2]
            +4.13891337e-10 * tc[3]
            -2.39007330e-14 * tc[4];
        /*species 45: PXC4H9O2 */
        species[45] =
            +1.47845448e+01
            +2.15210910e-02 * tc[1]
            -7.44909017e-06 * tc[2]
            +1.16558071e-09 * tc[3]
            -6.79885609e-14 * tc[4];
    }

    /*species with midpoint at T=1388 kelvin */
    if (T < 1388) {
        /*species 30: C3H6 */
        species[30] =
            -6.05384556e-01
            +2.89107662e-02 * tc[1]
            -1.54886808e-05 * tc[2]
            +3.88814209e-09 * tc[3]
            -3.37890352e-13 * tc[4];
    } else {
        /*species 30: C3H6 */
        species[30] =
            +7.01595958e+00
            +1.37023634e-02 * tc[1]
            -4.66249733e-06 * tc[2]
            +7.21254402e-10 * tc[3]
            -4.17370126e-14 * tc[4];
    }

    /*species with midpoint at T=1389 kelvin */
    if (T < 1389) {
        /*species 32: C2H5O */
        species[32] =
            -5.05579292e-01
            +2.71774434e-02 * tc[1]
            -1.65909010e-05 * tc[2]
            +5.15204200e-09 * tc[3]
            -6.48496915e-13 * tc[4];
    } else {
        /*species 32: C2H5O */
        species[32] =
            +6.87339772e+00
            +1.13072907e-02 * tc[1]
            -3.84421421e-06 * tc[2]
            +5.94414105e-10 * tc[3]
            -3.43894538e-14 * tc[4];
    }

    /*species with midpoint at T=1382 kelvin */
    if (T < 1382) {
        /*species 34: C2H5O2 */
        species[34] =
            +1.26846188e+00
            +2.76942578e-02 * tc[1]
            -1.70804106e-05 * tc[2]
            +5.87951878e-09 * tc[3]
            -9.20899069e-13 * tc[4];
        /*species 50: C7H15X2 */
        species[50] =
            -1.03791558e+00
            +7.56726570e-02 * tc[1]
            -4.07473634e-05 * tc[2]
            +9.32678943e-09 * tc[3]
            -4.92360745e-13 * tc[4];
    } else {
        /*species 34: C2H5O2 */
        species[34] =
            +8.48696023e+00
            +1.24472545e-02 * tc[1]
            -4.32161576e-06 * tc[2]
            +6.77583033e-10 * tc[3]
            -3.95784568e-14 * tc[4];
        /*species 50: C7H15X2 */
        species[50] =
            +2.06368842e+01
            +3.23324804e-02 * tc[1]
            -1.09273807e-05 * tc[2]
            +1.68357060e-09 * tc[3]
            -9.71774091e-14 * tc[4];
    }

    /*species with midpoint at T=1400 kelvin */
    if (T < 1400) {
        /*species 37: C3H4XA */
        species[37] =
            +1.53983100e+00
            +1.63343700e-02 * tc[1]
            -1.76495000e-06 * tc[2]
            -4.64736500e-09 * tc[3]
            +1.72913100e-12 * tc[4];
    } else {
        /*species 37: C3H4XA */
        species[37] =
            +8.77625600e+00
            +5.30213800e-03 * tc[1]
            -3.70111800e-07 * tc[2]
            -3.02638600e-10 * tc[3]
            +5.08958100e-14 * tc[4];
    }

    /*species with midpoint at T=1397 kelvin */
    if (T < 1397) {
        /*species 38: C3H5XA */
        species[38] =
            -1.52913196e+00
            +3.34559100e-02 * tc[1]
            -2.53401027e-05 * tc[2]
            +1.02865754e-08 * tc[3]
            -1.73258340e-12 * tc[4];
        /*species 48: C5H11X1 */
        species[48] =
            -1.90525591e+00
            +6.10632852e-02 * tc[1]
            -4.09491825e-05 * tc[2]
            +1.46093470e-08 * tc[3]
            -2.18859615e-12 * tc[4];
    } else {
        /*species 38: C3H5XA */
        species[38] =
            +7.45883958e+00
            +1.12695483e-02 * tc[1]
            -3.83792864e-06 * tc[2]
            +5.94059119e-10 * tc[3]
            -3.43918030e-14 * tc[4];
        /*species 48: C5H11X1 */
        species[48] =
            +1.43234740e+01
            +2.39041200e-02 * tc[1]
            -8.14771619e-06 * tc[2]
            +1.26176236e-09 * tc[3]
            -7.30677335e-14 * tc[4];
    }

    /*species with midpoint at T=1398 kelvin */
    if (T < 1398) {
        /*species 41: C4H6 */
        species[41] =
            -2.43095121e+00
            +4.78706062e-02 * tc[1]
            -4.15446800e-05 * tc[2]
            +1.91549552e-08 * tc[3]
            -3.57158507e-12 * tc[4];
    } else {
        /*species 41: C4H6 */
        species[41] =
            +1.01633789e+01
            +1.37163965e-02 * tc[1]
            -4.69715783e-06 * tc[2]
            +7.29693836e-10 * tc[3]
            -4.23486203e-14 * tc[4];
    }

    /*species with midpoint at T=1392 kelvin */
    if (T < 1392) {
        /*species 42: C4H7 */
        species[42] =
            -1.35050835e+00
            +4.26511243e-02 * tc[1]
            -2.90979373e-05 * tc[2]
            +1.05403914e-08 * tc[3]
            -1.60059854e-12 * tc[4];
        /*species 43: C4H8X1 */
        species[43] =
            -1.83137209e+00
            +4.52580978e-02 * tc[1]
            -2.93658559e-05 * tc[2]
            +1.00220436e-08 * tc[3]
            -1.43191680e-12 * tc[4];
        /*species 46: C5H9 */
        species[46] =
            -2.38013950e+00
            +5.57608487e-02 * tc[1]
            -3.70143928e-05 * tc[2]
            +1.26883901e-08 * tc[3]
            -1.78538835e-12 * tc[4];
        /*species 47: C5H10X1 */
        species[47] =
            -2.06223481e+00
            +5.74218294e-02 * tc[1]
            -3.74486890e-05 * tc[2]
            +1.27364989e-08 * tc[3]
            -1.79609789e-12 * tc[4];
        /*species 49: C6H12X1 */
        species[49] =
            -2.35275205e+00
            +6.98655426e-02 * tc[1]
            -4.59408022e-05 * tc[2]
            +1.56967343e-08 * tc[3]
            -2.21296175e-12 * tc[4];
    } else {
        /*species 42: C4H7 */
        species[42] =
            +1.02103578e+01
            +1.60483196e-02 * tc[1]
            -5.46502292e-06 * tc[2]
            +8.45941053e-10 * tc[3]
            -4.89772739e-14 * tc[4];
        /*species 43: C4H8X1 */
        species[43] =
            +1.03508668e+01
            +1.80617877e-02 * tc[1]
            -6.16093029e-06 * tc[2]
            +9.54652959e-10 * tc[3]
            -5.53089641e-14 * tc[4];
        /*species 46: C5H9 */
        species[46] =
            +1.31860454e+01
            +2.07128899e-02 * tc[1]
            -7.06960617e-06 * tc[2]
            +1.09607133e-09 * tc[3]
            -6.35322208e-14 * tc[4];
        /*species 47: C5H10X1 */
        species[47] =
            +1.35851539e+01
            +2.24072471e-02 * tc[1]
            -7.63348025e-06 * tc[2]
            +1.18188966e-09 * tc[3]
            -6.84385139e-14 * tc[4];
        /*species 49: C6H12X1 */
        species[49] =
            +1.68337529e+01
            +2.67377658e-02 * tc[1]
            -9.10036773e-06 * tc[2]
            +1.40819768e-09 * tc[3]
            -8.15124244e-14 * tc[4];
    }

    /*species with midpoint at T=1395 kelvin */
    if (T < 1395) {
        /*species 44: PXC4H9 */
        species[44] =
            -1.43777972e+00
            +4.78972364e-02 * tc[1]
            -3.14023159e-05 * tc[2]
            +1.09786472e-08 * tc[3]
            -1.62010664e-12 * tc[4];
    } else {
        /*species 44: PXC4H9 */
        species[44] =
            +1.11510082e+01
            +1.94310717e-02 * tc[1]
            -6.61577950e-06 * tc[2]
            +1.02375136e-09 * tc[3]
            -5.92529707e-14 * tc[4];
    }

    /*species with midpoint at T=1391 kelvin */
    if (T < 1391) {
        /*species 51: NXC7H16 */
        species[51] =
            -2.26836187e+00
            +8.54355820e-02 * tc[1]
            -5.25346786e-05 * tc[2]
            +1.62945721e-08 * tc[3]
            -2.02394925e-12 * tc[4];
    } else {
        /*species 51: NXC7H16 */
        species[51] =
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
        /*species 9: CH */
        species[9] =
            +3.20020200e+00
            +2.07287600e-03 * tc[1]
            -5.13443100e-06 * tc[2]
            +5.73389000e-09 * tc[3]
            -1.95553300e-12 * tc[4];
        /*species 10: HCO */
        species[10] =
            +2.89833000e+00
            +6.19914700e-03 * tc[1]
            -9.62308400e-06 * tc[2]
            +1.08982500e-08 * tc[3]
            -4.57488500e-12 * tc[4];
        /*species 11: CH2 */
        species[11] =
            +3.76223700e+00
            +1.15981900e-03 * tc[1]
            +2.48958500e-07 * tc[2]
            +8.80083600e-10 * tc[3]
            -7.33243500e-13 * tc[4];
        /*species 12: CO2 */
        species[12] =
            +2.27572500e+00
            +9.92207200e-03 * tc[1]
            -1.04091100e-05 * tc[2]
            +6.86668700e-09 * tc[3]
            -2.11728000e-12 * tc[4];
        /*species 13: CO */
        species[13] =
            +3.26245200e+00
            +1.51194100e-03 * tc[1]
            -3.88175500e-06 * tc[2]
            +5.58194400e-09 * tc[3]
            -2.47495100e-12 * tc[4];
        /*species 14: CH2O */
        species[14] =
            +1.65273100e+00
            +1.26314400e-02 * tc[1]
            -1.88816800e-05 * tc[2]
            +2.05003100e-08 * tc[3]
            -8.41323700e-12 * tc[4];
        /*species 15: CH2GSG */
        species[15] =
            +3.97126500e+00
            -1.69908900e-04 * tc[1]
            +1.02536900e-06 * tc[2]
            +2.49255100e-09 * tc[3]
            -1.98126600e-12 * tc[4];
        /*species 16: CH3 */
        species[16] =
            +2.43044300e+00
            +1.11241000e-02 * tc[1]
            -1.68022000e-05 * tc[2]
            +1.62182900e-08 * tc[3]
            -5.86495300e-12 * tc[4];
        /*species 17: CH3O */
        species[17] =
            +2.10620400e+00
            +7.21659500e-03 * tc[1]
            +5.33847200e-06 * tc[2]
            -7.37763600e-09 * tc[3]
            +2.07561100e-12 * tc[4];
        /*species 18: CH4 */
        species[18] =
            +7.78741500e-01
            +1.74766800e-02 * tc[1]
            -2.78340900e-05 * tc[2]
            +3.04970800e-08 * tc[3]
            -1.22393100e-11 * tc[4];
        /*species 19: CH3OH */
        species[19] =
            +2.66011500e+00
            +7.34150800e-03 * tc[1]
            +7.17005100e-06 * tc[2]
            -8.79319400e-09 * tc[3]
            +2.39057000e-12 * tc[4];
        /*species 21: C2H5 */
        species[21] =
            +2.69070200e+00
            +8.71913300e-03 * tc[1]
            +4.41983900e-06 * tc[2]
            +9.33870300e-10 * tc[3]
            -3.92777300e-12 * tc[4];
        /*species 22: CH2CO */
        species[22] =
            +2.97497100e+00
            +1.21187100e-02 * tc[1]
            -2.34504600e-06 * tc[2]
            -6.46668500e-09 * tc[3]
            +3.90564900e-12 * tc[4];
        /*species 26: C2H2 */
        species[26] =
            +2.01356200e+00
            +1.51904500e-02 * tc[1]
            -1.61631900e-05 * tc[2]
            +9.07899200e-09 * tc[3]
            -1.91274600e-12 * tc[4];
        /*species 27: HCCO */
        species[27] =
            +5.04796500e+00
            +4.45347800e-03 * tc[1]
            +2.26828300e-07 * tc[2]
            -1.48209500e-09 * tc[3]
            +2.25074200e-13 * tc[4];
        /*species 28: C2H3 */
        species[28] =
            +2.45927600e+00
            +7.37147600e-03 * tc[1]
            +2.10987300e-06 * tc[2]
            -1.32164200e-09 * tc[3]
            -1.18478400e-12 * tc[4];
        /*species 29: CH2CHO */
        species[29] =
            +3.40906200e+00
            +1.07385700e-02 * tc[1]
            +1.89149200e-06 * tc[2]
            -7.15858300e-09 * tc[3]
            +2.86738500e-12 * tc[4];
        /*species 31: C2H4 */
        species[31] =
            -8.61488000e-01
            +2.79616300e-02 * tc[1]
            -3.38867700e-05 * tc[2]
            +2.78515200e-08 * tc[3]
            -9.73787900e-12 * tc[4];
        /*species 33: CH3CO */
        species[33] =
            +3.12527800e+00
            +9.77822000e-03 * tc[1]
            +4.52144800e-06 * tc[2]
            -9.00946200e-09 * tc[3]
            +3.19371800e-12 * tc[4];
        /*species 35: C3H2 */
        species[35] =
            +3.16671400e+00
            +2.48257200e-02 * tc[1]
            -4.59163700e-05 * tc[2]
            +4.26801900e-08 * tc[3]
            -1.48215200e-11 * tc[4];
        /*species 36: C3H3 */
        species[36] =
            +4.75420000e+00
            +1.10802800e-02 * tc[1]
            +2.79332300e-07 * tc[2]
            -5.47921200e-09 * tc[3]
            +1.94962900e-12 * tc[4];
        /*species 39: NXC3H7 */
        species[39] =
            +1.92253700e+00
            +2.47892700e-02 * tc[1]
            +1.81024900e-06 * tc[2]
            -1.78326600e-08 * tc[3]
            +8.58299600e-12 * tc[4];
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
        /*species 9: CH */
        species[9] =
            +2.19622300e+00
            +2.34038100e-03 * tc[1]
            -7.05820100e-07 * tc[2]
            +9.00758200e-11 * tc[3]
            -3.85504000e-15 * tc[4];
        /*species 10: HCO */
        species[10] =
            +3.55727100e+00
            +3.34557300e-03 * tc[1]
            -1.33500600e-06 * tc[2]
            +2.47057300e-10 * tc[3]
            -1.71385100e-14 * tc[4];
        /*species 11: CH2 */
        species[11] =
            +3.63640800e+00
            +1.93305700e-03 * tc[1]
            -1.68701600e-07 * tc[2]
            -1.00989900e-10 * tc[3]
            +1.80825600e-14 * tc[4];
        /*species 12: CO2 */
        species[12] =
            +4.45362300e+00
            +3.14016900e-03 * tc[1]
            -1.27841100e-06 * tc[2]
            +2.39399700e-10 * tc[3]
            -1.66903300e-14 * tc[4];
        /*species 13: CO */
        species[13] =
            +3.02507800e+00
            +1.44268900e-03 * tc[1]
            -5.63082800e-07 * tc[2]
            +1.01858100e-10 * tc[3]
            -6.91095200e-15 * tc[4];
        /*species 14: CH2O */
        species[14] =
            +2.99560600e+00
            +6.68132100e-03 * tc[1]
            -2.62895500e-06 * tc[2]
            +4.73715300e-10 * tc[3]
            -3.21251700e-14 * tc[4];
        /*species 15: CH2GSG */
        species[15] =
            +3.55288900e+00
            +2.06678800e-03 * tc[1]
            -1.91411600e-07 * tc[2]
            -1.10467300e-10 * tc[3]
            +2.02135000e-14 * tc[4];
        /*species 16: CH3 */
        species[16] =
            +2.84405200e+00
            +6.13797400e-03 * tc[1]
            -2.23034500e-06 * tc[2]
            +3.78516100e-10 * tc[3]
            -2.45215900e-14 * tc[4];
        /*species 17: CH3O */
        species[17] =
            +3.77080000e+00
            +7.87149700e-03 * tc[1]
            -2.65638400e-06 * tc[2]
            +3.94443100e-10 * tc[3]
            -2.11261600e-14 * tc[4];
        /*species 18: CH4 */
        species[18] =
            +1.68347900e+00
            +1.02372400e-02 * tc[1]
            -3.87512900e-06 * tc[2]
            +6.78558500e-10 * tc[3]
            -4.50342300e-14 * tc[4];
        /*species 19: CH3OH */
        species[19] =
            +4.02906100e+00
            +9.37659300e-03 * tc[1]
            -3.05025400e-06 * tc[2]
            +4.35879300e-10 * tc[3]
            -2.22472300e-14 * tc[4];
        /*species 21: C2H5 */
        species[21] =
            +7.19048000e+00
            +6.48407700e-03 * tc[1]
            -6.42806500e-07 * tc[2]
            -2.34787900e-10 * tc[3]
            +3.88087700e-14 * tc[4];
        /*species 22: CH2CO */
        species[22] =
            +6.03881700e+00
            +5.80484000e-03 * tc[1]
            -1.92095400e-06 * tc[2]
            +2.79448500e-10 * tc[3]
            -1.45886800e-14 * tc[4];
        /*species 26: C2H2 */
        species[26] =
            +4.43677000e+00
            +5.37603900e-03 * tc[1]
            -1.91281700e-06 * tc[2]
            +3.28637900e-10 * tc[3]
            -2.15671000e-14 * tc[4];
        /*species 27: HCCO */
        species[27] =
            +6.75807300e+00
            +2.00040000e-03 * tc[1]
            -2.02760700e-07 * tc[2]
            -1.04113200e-10 * tc[3]
            +1.96516500e-14 * tc[4];
        /*species 28: C2H3 */
        species[28] =
            +5.93346800e+00
            +4.01774600e-03 * tc[1]
            -3.96674000e-07 * tc[2]
            -1.44126700e-10 * tc[3]
            +2.37864400e-14 * tc[4];
        /*species 29: CH2CHO */
        species[29] =
            +5.97567000e+00
            +8.13059100e-03 * tc[1]
            -2.74362400e-06 * tc[2]
            +4.07030400e-10 * tc[3]
            -2.17601700e-14 * tc[4];
        /*species 31: C2H4 */
        species[31] =
            +3.52841900e+00
            +1.14851800e-02 * tc[1]
            -4.41838500e-06 * tc[2]
            +7.84460100e-10 * tc[3]
            -5.26684800e-14 * tc[4];
        /*species 33: CH3CO */
        species[33] =
            +5.61227900e+00
            +8.44988600e-03 * tc[1]
            -2.85414700e-06 * tc[2]
            +4.23837600e-10 * tc[3]
            -2.26840400e-14 * tc[4];
        /*species 35: C3H2 */
        species[35] =
            +7.67098100e+00
            +2.74874900e-03 * tc[1]
            -4.37094300e-07 * tc[2]
            -6.45559900e-11 * tc[3]
            +1.66388700e-14 * tc[4];
        /*species 36: C3H3 */
        species[36] =
            +8.83104700e+00
            +4.35719500e-03 * tc[1]
            -4.10906700e-07 * tc[2]
            -2.36872300e-10 * tc[3]
            +4.37652000e-14 * tc[4];
        /*species 39: NXC3H7 */
        species[39] =
            +7.97829100e+00
            +1.57611300e-02 * tc[1]
            -5.17324300e-06 * tc[2]
            +7.44389200e-10 * tc[3]
            -3.82497800e-14 * tc[4];
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
        /*species 25: CH3O2H */
        species[25] =
            +3.23442817e+00
            +1.90129767e-02 * tc[1]
            -1.13386287e-05 * tc[2]
            +3.40306653e-09 * tc[3]
            -4.11830222e-13 * tc[4];
    } else {
        /*species 7: HO2 */
        species[7] =
            +4.10547423e+00
            +2.38452835e-03 * tc[1]
            -8.06347989e-07 * tc[2]
            +1.24191723e-10 * tc[3]
            -7.16400108e-15 * tc[4];
        /*species 25: CH3O2H */
        species[25] =
            +8.43117091e+00
            +8.06817909e-03 * tc[1]
            -2.77094921e-06 * tc[2]
            +4.31332243e-10 * tc[3]
            -2.50692146e-14 * tc[4];
    }

    /*species with midpoint at T=1384 kelvin */
    if (T < 1384) {
        /*species 20: C2H6 */
        species[20] =
            -2.52854344e-02
            +2.40764754e-02 * tc[1]
            -1.11893472e-05 * tc[2]
            +2.08340901e-09 * tc[3]
            -5.29868616e-14 * tc[4];
        /*species 40: NXC3H7O2 */
        species[40] =
            +2.10731492e+00
            +3.96164986e-02 * tc[1]
            -2.49491599e-05 * tc[2]
            +8.59450300e-09 * tc[3]
            -1.31240330e-12 * tc[4];
    } else {
        /*species 20: C2H6 */
        species[20] =
            +6.10683385e+00
            +1.29236361e-02 * tc[1]
            -4.42527196e-06 * tc[2]
            +6.87391726e-10 * tc[3]
            -3.98901732e-14 * tc[4];
        /*species 40: NXC3H7O2 */
        species[40] =
            +1.26327059e+01
            +1.69910726e-02 * tc[1]
            -5.88866873e-06 * tc[2]
            +9.22195396e-10 * tc[3]
            -5.38230675e-14 * tc[4];
    }

    /*species with midpoint at T=1376 kelvin */
    if (T < 1376) {
        /*species 23: HOCHO */
        species[23] =
            +1.43548185e+00
            +1.63363016e-02 * tc[1]
            -1.06257421e-05 * tc[2]
            +3.32132977e-09 * tc[3]
            -4.02176103e-13 * tc[4];
    } else {
        /*species 23: HOCHO */
        species[23] =
            +6.68733013e+00
            +5.14289368e-03 * tc[1]
            -1.82238513e-06 * tc[2]
            +2.89719163e-10 * tc[3]
            -1.70892199e-14 * tc[4];
    }

    /*species with midpoint at T=1385 kelvin */
    if (T < 1385) {
        /*species 24: CH3O2 */
        species[24] =
            +4.26146906e+00
            +1.00873599e-02 * tc[1]
            -3.21506184e-06 * tc[2]
            +2.09409267e-10 * tc[3]
            +4.18339103e-14 * tc[4];
        /*species 45: PXC4H9O2 */
        species[45] =
            +1.94363650e+00
            +5.15513163e-02 * tc[1]
            -3.28284400e-05 * tc[2]
            +1.13064860e-08 * tc[3]
            -1.70118606e-12 * tc[4];
    } else {
        /*species 24: CH3O2 */
        species[24] =
            +5.95787891e+00
            +7.90728626e-03 * tc[1]
            -2.68246234e-06 * tc[2]
            +4.13891337e-10 * tc[3]
            -2.39007330e-14 * tc[4];
        /*species 45: PXC4H9O2 */
        species[45] =
            +1.57845448e+01
            +2.15210910e-02 * tc[1]
            -7.44909017e-06 * tc[2]
            +1.16558071e-09 * tc[3]
            -6.79885609e-14 * tc[4];
    }

    /*species with midpoint at T=1388 kelvin */
    if (T < 1388) {
        /*species 30: C3H6 */
        species[30] =
            +3.94615444e-01
            +2.89107662e-02 * tc[1]
            -1.54886808e-05 * tc[2]
            +3.88814209e-09 * tc[3]
            -3.37890352e-13 * tc[4];
    } else {
        /*species 30: C3H6 */
        species[30] =
            +8.01595958e+00
            +1.37023634e-02 * tc[1]
            -4.66249733e-06 * tc[2]
            +7.21254402e-10 * tc[3]
            -4.17370126e-14 * tc[4];
    }

    /*species with midpoint at T=1389 kelvin */
    if (T < 1389) {
        /*species 32: C2H5O */
        species[32] =
            +4.94420708e-01
            +2.71774434e-02 * tc[1]
            -1.65909010e-05 * tc[2]
            +5.15204200e-09 * tc[3]
            -6.48496915e-13 * tc[4];
    } else {
        /*species 32: C2H5O */
        species[32] =
            +7.87339772e+00
            +1.13072907e-02 * tc[1]
            -3.84421421e-06 * tc[2]
            +5.94414105e-10 * tc[3]
            -3.43894538e-14 * tc[4];
    }

    /*species with midpoint at T=1382 kelvin */
    if (T < 1382) {
        /*species 34: C2H5O2 */
        species[34] =
            +2.26846188e+00
            +2.76942578e-02 * tc[1]
            -1.70804106e-05 * tc[2]
            +5.87951878e-09 * tc[3]
            -9.20899069e-13 * tc[4];
        /*species 50: C7H15X2 */
        species[50] =
            -3.79155767e-02
            +7.56726570e-02 * tc[1]
            -4.07473634e-05 * tc[2]
            +9.32678943e-09 * tc[3]
            -4.92360745e-13 * tc[4];
    } else {
        /*species 34: C2H5O2 */
        species[34] =
            +9.48696023e+00
            +1.24472545e-02 * tc[1]
            -4.32161576e-06 * tc[2]
            +6.77583033e-10 * tc[3]
            -3.95784568e-14 * tc[4];
        /*species 50: C7H15X2 */
        species[50] =
            +2.16368842e+01
            +3.23324804e-02 * tc[1]
            -1.09273807e-05 * tc[2]
            +1.68357060e-09 * tc[3]
            -9.71774091e-14 * tc[4];
    }

    /*species with midpoint at T=1400 kelvin */
    if (T < 1400) {
        /*species 37: C3H4XA */
        species[37] =
            +2.53983100e+00
            +1.63343700e-02 * tc[1]
            -1.76495000e-06 * tc[2]
            -4.64736500e-09 * tc[3]
            +1.72913100e-12 * tc[4];
    } else {
        /*species 37: C3H4XA */
        species[37] =
            +9.77625600e+00
            +5.30213800e-03 * tc[1]
            -3.70111800e-07 * tc[2]
            -3.02638600e-10 * tc[3]
            +5.08958100e-14 * tc[4];
    }

    /*species with midpoint at T=1397 kelvin */
    if (T < 1397) {
        /*species 38: C3H5XA */
        species[38] =
            -5.29131958e-01
            +3.34559100e-02 * tc[1]
            -2.53401027e-05 * tc[2]
            +1.02865754e-08 * tc[3]
            -1.73258340e-12 * tc[4];
        /*species 48: C5H11X1 */
        species[48] =
            -9.05255912e-01
            +6.10632852e-02 * tc[1]
            -4.09491825e-05 * tc[2]
            +1.46093470e-08 * tc[3]
            -2.18859615e-12 * tc[4];
    } else {
        /*species 38: C3H5XA */
        species[38] =
            +8.45883958e+00
            +1.12695483e-02 * tc[1]
            -3.83792864e-06 * tc[2]
            +5.94059119e-10 * tc[3]
            -3.43918030e-14 * tc[4];
        /*species 48: C5H11X1 */
        species[48] =
            +1.53234740e+01
            +2.39041200e-02 * tc[1]
            -8.14771619e-06 * tc[2]
            +1.26176236e-09 * tc[3]
            -7.30677335e-14 * tc[4];
    }

    /*species with midpoint at T=1398 kelvin */
    if (T < 1398) {
        /*species 41: C4H6 */
        species[41] =
            -1.43095121e+00
            +4.78706062e-02 * tc[1]
            -4.15446800e-05 * tc[2]
            +1.91549552e-08 * tc[3]
            -3.57158507e-12 * tc[4];
    } else {
        /*species 41: C4H6 */
        species[41] =
            +1.11633789e+01
            +1.37163965e-02 * tc[1]
            -4.69715783e-06 * tc[2]
            +7.29693836e-10 * tc[3]
            -4.23486203e-14 * tc[4];
    }

    /*species with midpoint at T=1392 kelvin */
    if (T < 1392) {
        /*species 42: C4H7 */
        species[42] =
            -3.50508352e-01
            +4.26511243e-02 * tc[1]
            -2.90979373e-05 * tc[2]
            +1.05403914e-08 * tc[3]
            -1.60059854e-12 * tc[4];
        /*species 43: C4H8X1 */
        species[43] =
            -8.31372089e-01
            +4.52580978e-02 * tc[1]
            -2.93658559e-05 * tc[2]
            +1.00220436e-08 * tc[3]
            -1.43191680e-12 * tc[4];
        /*species 46: C5H9 */
        species[46] =
            -1.38013950e+00
            +5.57608487e-02 * tc[1]
            -3.70143928e-05 * tc[2]
            +1.26883901e-08 * tc[3]
            -1.78538835e-12 * tc[4];
        /*species 47: C5H10X1 */
        species[47] =
            -1.06223481e+00
            +5.74218294e-02 * tc[1]
            -3.74486890e-05 * tc[2]
            +1.27364989e-08 * tc[3]
            -1.79609789e-12 * tc[4];
        /*species 49: C6H12X1 */
        species[49] =
            -1.35275205e+00
            +6.98655426e-02 * tc[1]
            -4.59408022e-05 * tc[2]
            +1.56967343e-08 * tc[3]
            -2.21296175e-12 * tc[4];
    } else {
        /*species 42: C4H7 */
        species[42] =
            +1.12103578e+01
            +1.60483196e-02 * tc[1]
            -5.46502292e-06 * tc[2]
            +8.45941053e-10 * tc[3]
            -4.89772739e-14 * tc[4];
        /*species 43: C4H8X1 */
        species[43] =
            +1.13508668e+01
            +1.80617877e-02 * tc[1]
            -6.16093029e-06 * tc[2]
            +9.54652959e-10 * tc[3]
            -5.53089641e-14 * tc[4];
        /*species 46: C5H9 */
        species[46] =
            +1.41860454e+01
            +2.07128899e-02 * tc[1]
            -7.06960617e-06 * tc[2]
            +1.09607133e-09 * tc[3]
            -6.35322208e-14 * tc[4];
        /*species 47: C5H10X1 */
        species[47] =
            +1.45851539e+01
            +2.24072471e-02 * tc[1]
            -7.63348025e-06 * tc[2]
            +1.18188966e-09 * tc[3]
            -6.84385139e-14 * tc[4];
        /*species 49: C6H12X1 */
        species[49] =
            +1.78337529e+01
            +2.67377658e-02 * tc[1]
            -9.10036773e-06 * tc[2]
            +1.40819768e-09 * tc[3]
            -8.15124244e-14 * tc[4];
    }

    /*species with midpoint at T=1395 kelvin */
    if (T < 1395) {
        /*species 44: PXC4H9 */
        species[44] =
            -4.37779725e-01
            +4.78972364e-02 * tc[1]
            -3.14023159e-05 * tc[2]
            +1.09786472e-08 * tc[3]
            -1.62010664e-12 * tc[4];
    } else {
        /*species 44: PXC4H9 */
        species[44] =
            +1.21510082e+01
            +1.94310717e-02 * tc[1]
            -6.61577950e-06 * tc[2]
            +1.02375136e-09 * tc[3]
            -5.92529707e-14 * tc[4];
    }

    /*species with midpoint at T=1391 kelvin */
    if (T < 1391) {
        /*species 51: NXC7H16 */
        species[51] =
            -1.26836187e+00
            +8.54355820e-02 * tc[1]
            -5.25346786e-05 * tc[2]
            +1.62945721e-08 * tc[3]
            -2.02394925e-12 * tc[4];
    } else {
        /*species 51: NXC7H16 */
        species[51] =
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
        /*species 9: CH */
        species[9] =
            +2.20020200e+00
            +1.03643800e-03 * tc[1]
            -1.71147700e-06 * tc[2]
            +1.43347250e-09 * tc[3]
            -3.91106600e-13 * tc[4]
            +7.04525900e+04 * invT;
        /*species 10: HCO */
        species[10] =
            +1.89833000e+00
            +3.09957350e-03 * tc[1]
            -3.20769467e-06 * tc[2]
            +2.72456250e-09 * tc[3]
            -9.14977000e-13 * tc[4]
            +4.15992200e+03 * invT;
        /*species 11: CH2 */
        species[11] =
            +2.76223700e+00
            +5.79909500e-04 * tc[1]
            +8.29861667e-08 * tc[2]
            +2.20020900e-10 * tc[3]
            -1.46648700e-13 * tc[4]
            +4.53679100e+04 * invT;
        /*species 12: CO2 */
        species[12] =
            +1.27572500e+00
            +4.96103600e-03 * tc[1]
            -3.46970333e-06 * tc[2]
            +1.71667175e-09 * tc[3]
            -4.23456000e-13 * tc[4]
            -4.83731400e+04 * invT;
        /*species 13: CO */
        species[13] =
            +2.26245200e+00
            +7.55970500e-04 * tc[1]
            -1.29391833e-06 * tc[2]
            +1.39548600e-09 * tc[3]
            -4.94990200e-13 * tc[4]
            -1.43105400e+04 * invT;
        /*species 14: CH2O */
        species[14] =
            +6.52731000e-01
            +6.31572000e-03 * tc[1]
            -6.29389333e-06 * tc[2]
            +5.12507750e-09 * tc[3]
            -1.68264740e-12 * tc[4]
            -1.48654000e+04 * invT;
        /*species 15: CH2GSG */
        species[15] =
            +2.97126500e+00
            -8.49544500e-05 * tc[1]
            +3.41789667e-07 * tc[2]
            +6.23137750e-10 * tc[3]
            -3.96253200e-13 * tc[4]
            +4.98936800e+04 * invT;
        /*species 16: CH3 */
        species[16] =
            +1.43044300e+00
            +5.56205000e-03 * tc[1]
            -5.60073333e-06 * tc[2]
            +4.05457250e-09 * tc[3]
            -1.17299060e-12 * tc[4]
            +1.64237800e+04 * invT;
        /*species 17: CH3O */
        species[17] =
            +1.10620400e+00
            +3.60829750e-03 * tc[1]
            +1.77949067e-06 * tc[2]
            -1.84440900e-09 * tc[3]
            +4.15122200e-13 * tc[4]
            +9.78601100e+02 * invT;
        /*species 18: CH4 */
        species[18] =
            -2.21258500e-01
            +8.73834000e-03 * tc[1]
            -9.27803000e-06 * tc[2]
            +7.62427000e-09 * tc[3]
            -2.44786200e-12 * tc[4]
            -9.82522900e+03 * invT;
        /*species 19: CH3OH */
        species[19] =
            +1.66011500e+00
            +3.67075400e-03 * tc[1]
            +2.39001700e-06 * tc[2]
            -2.19829850e-09 * tc[3]
            +4.78114000e-13 * tc[4]
            -2.53534800e+04 * invT;
        /*species 21: C2H5 */
        species[21] =
            +1.69070200e+00
            +4.35956650e-03 * tc[1]
            +1.47327967e-06 * tc[2]
            +2.33467575e-10 * tc[3]
            -7.85554600e-13 * tc[4]
            +1.28704000e+04 * invT;
        /*species 22: CH2CO */
        species[22] =
            +1.97497100e+00
            +6.05935500e-03 * tc[1]
            -7.81682000e-07 * tc[2]
            -1.61667125e-09 * tc[3]
            +7.81129800e-13 * tc[4]
            -7.63263700e+03 * invT;
        /*species 26: C2H2 */
        species[26] =
            +1.01356200e+00
            +7.59522500e-03 * tc[1]
            -5.38773000e-06 * tc[2]
            +2.26974800e-09 * tc[3]
            -3.82549200e-13 * tc[4]
            +2.61244400e+04 * invT;
        /*species 27: HCCO */
        species[27] =
            +4.04796500e+00
            +2.22673900e-03 * tc[1]
            +7.56094333e-08 * tc[2]
            -3.70523750e-10 * tc[3]
            +4.50148400e-14 * tc[4]
            +1.96589200e+04 * invT;
        /*species 28: C2H3 */
        species[28] =
            +1.45927600e+00
            +3.68573800e-03 * tc[1]
            +7.03291000e-07 * tc[2]
            -3.30410500e-10 * tc[3]
            -2.36956800e-13 * tc[4]
            +3.33522500e+04 * invT;
        /*species 29: CH2CHO */
        species[29] =
            +2.40906200e+00
            +5.36928500e-03 * tc[1]
            +6.30497333e-07 * tc[2]
            -1.78964575e-09 * tc[3]
            +5.73477000e-13 * tc[4]
            +1.52147700e+03 * invT;
        /*species 31: C2H4 */
        species[31] =
            -1.86148800e+00
            +1.39808150e-02 * tc[1]
            -1.12955900e-05 * tc[2]
            +6.96288000e-09 * tc[3]
            -1.94757580e-12 * tc[4]
            +5.57304600e+03 * invT;
        /*species 33: CH3CO */
        species[33] =
            +2.12527800e+00
            +4.88911000e-03 * tc[1]
            +1.50714933e-06 * tc[2]
            -2.25236550e-09 * tc[3]
            +6.38743600e-13 * tc[4]
            -4.10850800e+03 * invT;
        /*species 35: C3H2 */
        species[35] =
            +2.16671400e+00
            +1.24128600e-02 * tc[1]
            -1.53054567e-05 * tc[2]
            +1.06700475e-08 * tc[3]
            -2.96430400e-12 * tc[4]
            +6.35042100e+04 * invT;
        /*species 36: C3H3 */
        species[36] =
            +3.75420000e+00
            +5.54014000e-03 * tc[1]
            +9.31107667e-08 * tc[2]
            -1.36980300e-09 * tc[3]
            +3.89925800e-13 * tc[4]
            +3.98888300e+04 * invT;
        /*species 39: NXC3H7 */
        species[39] =
            +9.22537000e-01
            +1.23946350e-02 * tc[1]
            +6.03416333e-07 * tc[2]
            -4.45816500e-09 * tc[3]
            +1.71659920e-12 * tc[4]
            +9.71328100e+03 * invT;
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
        /*species 9: CH */
        species[9] =
            +1.19622300e+00
            +1.17019050e-03 * tc[1]
            -2.35273367e-07 * tc[2]
            +2.25189550e-11 * tc[3]
            -7.71008000e-16 * tc[4]
            +7.08672300e+04 * invT;
        /*species 10: HCO */
        species[10] =
            +2.55727100e+00
            +1.67278650e-03 * tc[1]
            -4.45002000e-07 * tc[2]
            +6.17643250e-11 * tc[3]
            -3.42770200e-15 * tc[4]
            +3.91632400e+03 * invT;
        /*species 11: CH2 */
        species[11] =
            +2.63640800e+00
            +9.66528500e-04 * tc[1]
            -5.62338667e-08 * tc[2]
            -2.52474750e-11 * tc[3]
            +3.61651200e-15 * tc[4]
            +4.53413400e+04 * invT;
        /*species 12: CO2 */
        species[12] =
            +3.45362300e+00
            +1.57008450e-03 * tc[1]
            -4.26137000e-07 * tc[2]
            +5.98499250e-11 * tc[3]
            -3.33806600e-15 * tc[4]
            -4.89669600e+04 * invT;
        /*species 13: CO */
        species[13] =
            +2.02507800e+00
            +7.21344500e-04 * tc[1]
            -1.87694267e-07 * tc[2]
            +2.54645250e-11 * tc[3]
            -1.38219040e-15 * tc[4]
            -1.42683500e+04 * invT;
        /*species 14: CH2O */
        species[14] =
            +1.99560600e+00
            +3.34066050e-03 * tc[1]
            -8.76318333e-07 * tc[2]
            +1.18428825e-10 * tc[3]
            -6.42503400e-15 * tc[4]
            -1.53203700e+04 * invT;
        /*species 15: CH2GSG */
        species[15] =
            +2.55288900e+00
            +1.03339400e-03 * tc[1]
            -6.38038667e-08 * tc[2]
            -2.76168250e-11 * tc[3]
            +4.04270000e-15 * tc[4]
            +4.98497500e+04 * invT;
        /*species 16: CH3 */
        species[16] =
            +1.84405200e+00
            +3.06898700e-03 * tc[1]
            -7.43448333e-07 * tc[2]
            +9.46290250e-11 * tc[3]
            -4.90431800e-15 * tc[4]
            +1.64378100e+04 * invT;
        /*species 17: CH3O */
        species[17] =
            +2.77080000e+00
            +3.93574850e-03 * tc[1]
            -8.85461333e-07 * tc[2]
            +9.86107750e-11 * tc[3]
            -4.22523200e-15 * tc[4]
            +1.27832500e+02 * invT;
        /*species 18: CH4 */
        species[18] =
            +6.83479000e-01
            +5.11862000e-03 * tc[1]
            -1.29170967e-06 * tc[2]
            +1.69639625e-10 * tc[3]
            -9.00684600e-15 * tc[4]
            -1.00807900e+04 * invT;
        /*species 19: CH3OH */
        species[19] =
            +3.02906100e+00
            +4.68829650e-03 * tc[1]
            -1.01675133e-06 * tc[2]
            +1.08969825e-10 * tc[3]
            -4.44944600e-15 * tc[4]
            -2.61579100e+04 * invT;
        /*species 21: C2H5 */
        species[21] =
            +6.19048000e+00
            +3.24203850e-03 * tc[1]
            -2.14268833e-07 * tc[2]
            -5.86969750e-11 * tc[3]
            +7.76175400e-15 * tc[4]
            +1.06745500e+04 * invT;
        /*species 22: CH2CO */
        species[22] =
            +5.03881700e+00
            +2.90242000e-03 * tc[1]
            -6.40318000e-07 * tc[2]
            +6.98621250e-11 * tc[3]
            -2.91773600e-15 * tc[4]
            -8.58340200e+03 * invT;
        /*species 26: C2H2 */
        species[26] =
            +3.43677000e+00
            +2.68801950e-03 * tc[1]
            -6.37605667e-07 * tc[2]
            +8.21594750e-11 * tc[3]
            -4.31342000e-15 * tc[4]
            +2.56676600e+04 * invT;
        /*species 27: HCCO */
        species[27] =
            +5.75807300e+00
            +1.00020000e-03 * tc[1]
            -6.75869000e-08 * tc[2]
            -2.60283000e-11 * tc[3]
            +3.93033000e-15 * tc[4]
            +1.90151300e+04 * invT;
        /*species 28: C2H3 */
        species[28] =
            +4.93346800e+00
            +2.00887300e-03 * tc[1]
            -1.32224667e-07 * tc[2]
            -3.60316750e-11 * tc[3]
            +4.75728800e-15 * tc[4]
            +3.18543500e+04 * invT;
        /*species 29: CH2CHO */
        species[29] =
            +4.97567000e+00
            +4.06529550e-03 * tc[1]
            -9.14541333e-07 * tc[2]
            +1.01757600e-10 * tc[3]
            -4.35203400e-15 * tc[4]
            +4.90321800e+02 * invT;
        /*species 31: C2H4 */
        species[31] =
            +2.52841900e+00
            +5.74259000e-03 * tc[1]
            -1.47279500e-06 * tc[2]
            +1.96115025e-10 * tc[3]
            -1.05336960e-14 * tc[4]
            +4.42828900e+03 * invT;
        /*species 33: CH3CO */
        species[33] =
            +4.61227900e+00
            +4.22494300e-03 * tc[1]
            -9.51382333e-07 * tc[2]
            +1.05959400e-10 * tc[3]
            -4.53680800e-15 * tc[4]
            -5.18786300e+03 * invT;
        /*species 35: C3H2 */
        species[35] =
            +6.67098100e+00
            +1.37437450e-03 * tc[1]
            -1.45698100e-07 * tc[2]
            -1.61389975e-11 * tc[3]
            +3.32777400e-15 * tc[4]
            +6.25972200e+04 * invT;
        /*species 36: C3H3 */
        species[36] =
            +7.83104700e+00
            +2.17859750e-03 * tc[1]
            -1.36968900e-07 * tc[2]
            -5.92180750e-11 * tc[3]
            +8.75304000e-15 * tc[4]
            +3.84742000e+04 * invT;
        /*species 39: NXC3H7 */
        species[39] =
            +6.97829100e+00
            +7.88056500e-03 * tc[1]
            -1.72441433e-06 * tc[2]
            +1.86097300e-10 * tc[3]
            -7.64995600e-15 * tc[4]
            +7.57940200e+03 * invT;
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
        /*species 25: CH3O2H */
        species[25] =
            +2.23442817e+00
            +9.50648835e-03 * tc[1]
            -3.77954290e-06 * tc[2]
            +8.50766632e-10 * tc[3]
            -8.23660444e-14 * tc[4]
            -1.77197926e+04 * invT;
    } else {
        /*species 7: HO2 */
        species[7] =
            +3.10547423e+00
            +1.19226417e-03 * tc[1]
            -2.68782663e-07 * tc[2]
            +3.10479308e-11 * tc[3]
            -1.43280022e-15 * tc[4]
            +3.98127689e+02 * invT;
        /*species 25: CH3O2H */
        species[25] =
            +7.43117091e+00
            +4.03408955e-03 * tc[1]
            -9.23649737e-07 * tc[2]
            +1.07833061e-10 * tc[3]
            -5.01384292e-15 * tc[4]
            -1.96678771e+04 * invT;
    }

    /*species with midpoint at T=1384 kelvin */
    if (T < 1384) {
        /*species 20: C2H6 */
        species[20] =
            -1.02528543e+00
            +1.20382377e-02 * tc[1]
            -3.72978240e-06 * tc[2]
            +5.20852252e-10 * tc[3]
            -1.05973723e-14 * tc[4]
            -1.12345534e+04 * invT;
        /*species 40: NXC3H7O2 */
        species[40] =
            +1.10731492e+00
            +1.98082493e-02 * tc[1]
            -8.31638663e-06 * tc[2]
            +2.14862575e-09 * tc[3]
            -2.62480660e-13 * tc[4]
            -7.93745567e+03 * invT;
    } else {
        /*species 20: C2H6 */
        species[20] =
            +5.10683385e+00
            +6.46181805e-03 * tc[1]
            -1.47509065e-06 * tc[2]
            +1.71847932e-10 * tc[3]
            -7.97803464e-15 * tc[4]
            -1.37500014e+04 * invT;
        /*species 40: NXC3H7O2 */
        species[40] =
            +1.16327059e+01
            +8.49553630e-03 * tc[1]
            -1.96288958e-06 * tc[2]
            +2.30548849e-10 * tc[3]
            -1.07646135e-14 * tc[4]
            -1.19194652e+04 * invT;
    }

    /*species with midpoint at T=1376 kelvin */
    if (T < 1376) {
        /*species 23: HOCHO */
        species[23] =
            +4.35481850e-01
            +8.16815080e-03 * tc[1]
            -3.54191403e-06 * tc[2]
            +8.30332443e-10 * tc[3]
            -8.04352206e-14 * tc[4]
            -4.64616504e+04 * invT;
    } else {
        /*species 23: HOCHO */
        species[23] =
            +5.68733013e+00
            +2.57144684e-03 * tc[1]
            -6.07461710e-07 * tc[2]
            +7.24297908e-11 * tc[3]
            -3.41784398e-15 * tc[4]
            -4.83995400e+04 * invT;
    }

    /*species with midpoint at T=1385 kelvin */
    if (T < 1385) {
        /*species 24: CH3O2 */
        species[24] =
            +3.26146906e+00
            +5.04367995e-03 * tc[1]
            -1.07168728e-06 * tc[2]
            +5.23523168e-11 * tc[3]
            +8.36678206e-15 * tc[4]
            -6.84394259e+02 * invT;
        /*species 45: PXC4H9O2 */
        species[45] =
            +9.43636500e-01
            +2.57756581e-02 * tc[1]
            -1.09428133e-05 * tc[2]
            +2.82662150e-09 * tc[3]
            -3.40237212e-13 * tc[4]
            -1.08358103e+04 * invT;
    } else {
        /*species 24: CH3O2 */
        species[24] =
            +4.95787891e+00
            +3.95364313e-03 * tc[1]
            -8.94154113e-07 * tc[2]
            +1.03472834e-10 * tc[3]
            -4.78014660e-15 * tc[4]
            -1.53574838e+03 * invT;
        /*species 45: PXC4H9O2 */
        species[45] =
            +1.47845448e+01
            +1.07605455e-02 * tc[1]
            -2.48303006e-06 * tc[2]
            +2.91395178e-10 * tc[3]
            -1.35977122e-14 * tc[4]
            -1.60146054e+04 * invT;
    }

    /*species with midpoint at T=1388 kelvin */
    if (T < 1388) {
        /*species 30: C3H6 */
        species[30] =
            -6.05384556e-01
            +1.44553831e-02 * tc[1]
            -5.16289360e-06 * tc[2]
            +9.72035522e-10 * tc[3]
            -6.75780704e-14 * tc[4]
            +1.06688164e+03 * invT;
    } else {
        /*species 30: C3H6 */
        species[30] =
            +7.01595958e+00
            +6.85118170e-03 * tc[1]
            -1.55416578e-06 * tc[2]
            +1.80313601e-10 * tc[3]
            -8.34740252e-15 * tc[4]
            -1.87821271e+03 * invT;
    }

    /*species with midpoint at T=1389 kelvin */
    if (T < 1389) {
        /*species 32: C2H5O */
        species[32] =
            -5.05579292e-01
            +1.35887217e-02 * tc[1]
            -5.53030033e-06 * tc[2]
            +1.28801050e-09 * tc[3]
            -1.29699383e-13 * tc[4]
            -3.35252925e+03 * invT;
    } else {
        /*species 32: C2H5O */
        species[32] =
            +6.87339772e+00
            +5.65364535e-03 * tc[1]
            -1.28140474e-06 * tc[2]
            +1.48603526e-10 * tc[3]
            -6.87789076e-15 * tc[4]
            -6.07274953e+03 * invT;
    }

    /*species with midpoint at T=1382 kelvin */
    if (T < 1382) {
        /*species 34: C2H5O2 */
        species[34] =
            +1.26846188e+00
            +1.38471289e-02 * tc[1]
            -5.69347020e-06 * tc[2]
            +1.46987970e-09 * tc[3]
            -1.84179814e-13 * tc[4]
            -5.03880758e+03 * invT;
        /*species 50: C7H15X2 */
        species[50] =
            -1.03791558e+00
            +3.78363285e-02 * tc[1]
            -1.35824545e-05 * tc[2]
            +2.33169736e-09 * tc[3]
            -9.84721490e-14 * tc[4]
            -2.35605303e+03 * invT;
    } else {
        /*species 34: C2H5O2 */
        species[34] =
            +8.48696023e+00
            +6.22362725e-03 * tc[1]
            -1.44053859e-06 * tc[2]
            +1.69395758e-10 * tc[3]
            -7.91569136e-15 * tc[4]
            -7.82481795e+03 * invT;
        /*species 50: C7H15X2 */
        species[50] =
            +2.06368842e+01
            +1.61662402e-02 * tc[1]
            -3.64246023e-06 * tc[2]
            +4.20892650e-10 * tc[3]
            -1.94354818e-14 * tc[4]
            -1.05873616e+04 * invT;
    }

    /*species with midpoint at T=1400 kelvin */
    if (T < 1400) {
        /*species 37: C3H4XA */
        species[37] =
            +1.53983100e+00
            +8.16718500e-03 * tc[1]
            -5.88316667e-07 * tc[2]
            -1.16184125e-09 * tc[3]
            +3.45826200e-13 * tc[4]
            +2.25124300e+04 * invT;
    } else {
        /*species 37: C3H4XA */
        species[37] =
            +8.77625600e+00
            +2.65106900e-03 * tc[1]
            -1.23370600e-07 * tc[2]
            -7.56596500e-11 * tc[3]
            +1.01791620e-14 * tc[4]
            +1.95497200e+04 * invT;
    }

    /*species with midpoint at T=1397 kelvin */
    if (T < 1397) {
        /*species 38: C3H5XA */
        species[38] =
            -1.52913196e+00
            +1.67279550e-02 * tc[1]
            -8.44670090e-06 * tc[2]
            +2.57164385e-09 * tc[3]
            -3.46516680e-13 * tc[4]
            +1.93834226e+04 * invT;
        /*species 48: C5H11X1 */
        species[48] =
            -1.90525591e+00
            +3.05316426e-02 * tc[1]
            -1.36497275e-05 * tc[2]
            +3.65233675e-09 * tc[3]
            -4.37719230e-13 * tc[4]
            +4.83995303e+03 * invT;
    } else {
        /*species 38: C3H5XA */
        species[38] =
            +7.45883958e+00
            +5.63477415e-03 * tc[1]
            -1.27930955e-06 * tc[2]
            +1.48514780e-10 * tc[3]
            -6.87836060e-15 * tc[4]
            +1.63576092e+04 * invT;
        /*species 48: C5H11X1 */
        species[48] =
            +1.43234740e+01
            +1.19520600e-02 * tc[1]
            -2.71590540e-06 * tc[2]
            +3.15440590e-10 * tc[3]
            -1.46135467e-14 * tc[4]
            -9.23241637e+02 * invT;
    }

    /*species with midpoint at T=1398 kelvin */
    if (T < 1398) {
        /*species 41: C4H6 */
        species[41] =
            -2.43095121e+00
            +2.39353031e-02 * tc[1]
            -1.38482267e-05 * tc[2]
            +4.78873880e-09 * tc[3]
            -7.14317014e-13 * tc[4]
            +1.17551314e+04 * invT;
    } else {
        /*species 41: C4H6 */
        species[41] =
            +1.01633789e+01
            +6.85819825e-03 * tc[1]
            -1.56571928e-06 * tc[2]
            +1.82423459e-10 * tc[3]
            -8.46972406e-15 * tc[4]
            +7.79039770e+03 * invT;
    }

    /*species with midpoint at T=1392 kelvin */
    if (T < 1392) {
        /*species 42: C4H7 */
        species[42] =
            -1.35050835e+00
            +2.13255622e-02 * tc[1]
            -9.69931243e-06 * tc[2]
            +2.63509785e-09 * tc[3]
            -3.20119708e-13 * tc[4]
            +1.49933591e+04 * invT;
        /*species 43: C4H8X1 */
        species[43] =
            -1.83137209e+00
            +2.26290489e-02 * tc[1]
            -9.78861863e-06 * tc[2]
            +2.50551090e-09 * tc[3]
            -2.86383360e-13 * tc[4]
            -1.57875035e+03 * invT;
        /*species 46: C5H9 */
        species[46] =
            -2.38013950e+00
            +2.78804243e-02 * tc[1]
            -1.23381309e-05 * tc[2]
            +3.17209752e-09 * tc[3]
            -3.57077670e-13 * tc[4]
            +1.25589824e+04 * invT;
        /*species 47: C5H10X1 */
        species[47] =
            -2.06223481e+00
            +2.87109147e-02 * tc[1]
            -1.24828963e-05 * tc[2]
            +3.18412472e-09 * tc[3]
            -3.59219578e-13 * tc[4]
            -4.46546666e+03 * invT;
        /*species 49: C6H12X1 */
        species[49] =
            -2.35275205e+00
            +3.49327713e-02 * tc[1]
            -1.53136007e-05 * tc[2]
            +3.92418358e-09 * tc[3]
            -4.42592350e-13 * tc[4]
            -7.34368617e+03 * invT;
    } else {
        /*species 42: C4H7 */
        species[42] =
            +1.02103578e+01
            +8.02415980e-03 * tc[1]
            -1.82167431e-06 * tc[2]
            +2.11485263e-10 * tc[3]
            -9.79545478e-15 * tc[4]
            +1.09041937e+04 * invT;
        /*species 43: C4H8X1 */
        species[43] =
            +1.03508668e+01
            +9.03089385e-03 * tc[1]
            -2.05364343e-06 * tc[2]
            +2.38663240e-10 * tc[3]
            -1.10617928e-14 * tc[4]
            -5.97871038e+03 * invT;
        /*species 46: C5H9 */
        species[46] =
            +1.31860454e+01
            +1.03564449e-02 * tc[1]
            -2.35653539e-06 * tc[2]
            +2.74017833e-10 * tc[3]
            -1.27064442e-14 * tc[4]
            +7.00496135e+03 * invT;
        /*species 47: C5H10X1 */
        species[47] =
            +1.35851539e+01
            +1.12036235e-02 * tc[1]
            -2.54449342e-06 * tc[2]
            +2.95472415e-10 * tc[3]
            -1.36877028e-14 * tc[4]
            -1.00898205e+04 * invT;
        /*species 49: C6H12X1 */
        species[49] =
            +1.68337529e+01
            +1.33688829e-02 * tc[1]
            -3.03345591e-06 * tc[2]
            +3.52049420e-10 * tc[3]
            -1.63024849e-14 * tc[4]
            -1.42062860e+04 * invT;
    }

    /*species with midpoint at T=1395 kelvin */
    if (T < 1395) {
        /*species 44: PXC4H9 */
        species[44] =
            -1.43777972e+00
            +2.39486182e-02 * tc[1]
            -1.04674386e-05 * tc[2]
            +2.74466180e-09 * tc[3]
            -3.24021328e-13 * tc[4]
            +7.68945248e+03 * invT;
    } else {
        /*species 44: PXC4H9 */
        species[44] =
            +1.11510082e+01
            +9.71553585e-03 * tc[1]
            -2.20525983e-06 * tc[2]
            +2.55937840e-10 * tc[3]
            -1.18505941e-14 * tc[4]
            +3.17231942e+03 * invT;
    }

    /*species with midpoint at T=1391 kelvin */
    if (T < 1391) {
        /*species 51: NXC7H16 */
        species[51] =
            -2.26836187e+00
            +4.27177910e-02 * tc[1]
            -1.75115595e-05 * tc[2]
            +4.07364302e-09 * tc[3]
            -4.04789850e-13 * tc[4]
            -2.56586565e+04 * invT;
    } else {
        /*species 51: NXC7H16 */
        species[51] =
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
        /*species 9: CH */
        species[9] =
            +3.20020200e+00
            +1.03643800e-03 * tc[1]
            -1.71147700e-06 * tc[2]
            +1.43347250e-09 * tc[3]
            -3.91106600e-13 * tc[4]
            +7.04525900e+04 * invT;
        /*species 10: HCO */
        species[10] =
            +2.89833000e+00
            +3.09957350e-03 * tc[1]
            -3.20769467e-06 * tc[2]
            +2.72456250e-09 * tc[3]
            -9.14977000e-13 * tc[4]
            +4.15992200e+03 * invT;
        /*species 11: CH2 */
        species[11] =
            +3.76223700e+00
            +5.79909500e-04 * tc[1]
            +8.29861667e-08 * tc[2]
            +2.20020900e-10 * tc[3]
            -1.46648700e-13 * tc[4]
            +4.53679100e+04 * invT;
        /*species 12: CO2 */
        species[12] =
            +2.27572500e+00
            +4.96103600e-03 * tc[1]
            -3.46970333e-06 * tc[2]
            +1.71667175e-09 * tc[3]
            -4.23456000e-13 * tc[4]
            -4.83731400e+04 * invT;
        /*species 13: CO */
        species[13] =
            +3.26245200e+00
            +7.55970500e-04 * tc[1]
            -1.29391833e-06 * tc[2]
            +1.39548600e-09 * tc[3]
            -4.94990200e-13 * tc[4]
            -1.43105400e+04 * invT;
        /*species 14: CH2O */
        species[14] =
            +1.65273100e+00
            +6.31572000e-03 * tc[1]
            -6.29389333e-06 * tc[2]
            +5.12507750e-09 * tc[3]
            -1.68264740e-12 * tc[4]
            -1.48654000e+04 * invT;
        /*species 15: CH2GSG */
        species[15] =
            +3.97126500e+00
            -8.49544500e-05 * tc[1]
            +3.41789667e-07 * tc[2]
            +6.23137750e-10 * tc[3]
            -3.96253200e-13 * tc[4]
            +4.98936800e+04 * invT;
        /*species 16: CH3 */
        species[16] =
            +2.43044300e+00
            +5.56205000e-03 * tc[1]
            -5.60073333e-06 * tc[2]
            +4.05457250e-09 * tc[3]
            -1.17299060e-12 * tc[4]
            +1.64237800e+04 * invT;
        /*species 17: CH3O */
        species[17] =
            +2.10620400e+00
            +3.60829750e-03 * tc[1]
            +1.77949067e-06 * tc[2]
            -1.84440900e-09 * tc[3]
            +4.15122200e-13 * tc[4]
            +9.78601100e+02 * invT;
        /*species 18: CH4 */
        species[18] =
            +7.78741500e-01
            +8.73834000e-03 * tc[1]
            -9.27803000e-06 * tc[2]
            +7.62427000e-09 * tc[3]
            -2.44786200e-12 * tc[4]
            -9.82522900e+03 * invT;
        /*species 19: CH3OH */
        species[19] =
            +2.66011500e+00
            +3.67075400e-03 * tc[1]
            +2.39001700e-06 * tc[2]
            -2.19829850e-09 * tc[3]
            +4.78114000e-13 * tc[4]
            -2.53534800e+04 * invT;
        /*species 21: C2H5 */
        species[21] =
            +2.69070200e+00
            +4.35956650e-03 * tc[1]
            +1.47327967e-06 * tc[2]
            +2.33467575e-10 * tc[3]
            -7.85554600e-13 * tc[4]
            +1.28704000e+04 * invT;
        /*species 22: CH2CO */
        species[22] =
            +2.97497100e+00
            +6.05935500e-03 * tc[1]
            -7.81682000e-07 * tc[2]
            -1.61667125e-09 * tc[3]
            +7.81129800e-13 * tc[4]
            -7.63263700e+03 * invT;
        /*species 26: C2H2 */
        species[26] =
            +2.01356200e+00
            +7.59522500e-03 * tc[1]
            -5.38773000e-06 * tc[2]
            +2.26974800e-09 * tc[3]
            -3.82549200e-13 * tc[4]
            +2.61244400e+04 * invT;
        /*species 27: HCCO */
        species[27] =
            +5.04796500e+00
            +2.22673900e-03 * tc[1]
            +7.56094333e-08 * tc[2]
            -3.70523750e-10 * tc[3]
            +4.50148400e-14 * tc[4]
            +1.96589200e+04 * invT;
        /*species 28: C2H3 */
        species[28] =
            +2.45927600e+00
            +3.68573800e-03 * tc[1]
            +7.03291000e-07 * tc[2]
            -3.30410500e-10 * tc[3]
            -2.36956800e-13 * tc[4]
            +3.33522500e+04 * invT;
        /*species 29: CH2CHO */
        species[29] =
            +3.40906200e+00
            +5.36928500e-03 * tc[1]
            +6.30497333e-07 * tc[2]
            -1.78964575e-09 * tc[3]
            +5.73477000e-13 * tc[4]
            +1.52147700e+03 * invT;
        /*species 31: C2H4 */
        species[31] =
            -8.61488000e-01
            +1.39808150e-02 * tc[1]
            -1.12955900e-05 * tc[2]
            +6.96288000e-09 * tc[3]
            -1.94757580e-12 * tc[4]
            +5.57304600e+03 * invT;
        /*species 33: CH3CO */
        species[33] =
            +3.12527800e+00
            +4.88911000e-03 * tc[1]
            +1.50714933e-06 * tc[2]
            -2.25236550e-09 * tc[3]
            +6.38743600e-13 * tc[4]
            -4.10850800e+03 * invT;
        /*species 35: C3H2 */
        species[35] =
            +3.16671400e+00
            +1.24128600e-02 * tc[1]
            -1.53054567e-05 * tc[2]
            +1.06700475e-08 * tc[3]
            -2.96430400e-12 * tc[4]
            +6.35042100e+04 * invT;
        /*species 36: C3H3 */
        species[36] =
            +4.75420000e+00
            +5.54014000e-03 * tc[1]
            +9.31107667e-08 * tc[2]
            -1.36980300e-09 * tc[3]
            +3.89925800e-13 * tc[4]
            +3.98888300e+04 * invT;
        /*species 39: NXC3H7 */
        species[39] =
            +1.92253700e+00
            +1.23946350e-02 * tc[1]
            +6.03416333e-07 * tc[2]
            -4.45816500e-09 * tc[3]
            +1.71659920e-12 * tc[4]
            +9.71328100e+03 * invT;
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
        /*species 9: CH */
        species[9] =
            +2.19622300e+00
            +1.17019050e-03 * tc[1]
            -2.35273367e-07 * tc[2]
            +2.25189550e-11 * tc[3]
            -7.71008000e-16 * tc[4]
            +7.08672300e+04 * invT;
        /*species 10: HCO */
        species[10] =
            +3.55727100e+00
            +1.67278650e-03 * tc[1]
            -4.45002000e-07 * tc[2]
            +6.17643250e-11 * tc[3]
            -3.42770200e-15 * tc[4]
            +3.91632400e+03 * invT;
        /*species 11: CH2 */
        species[11] =
            +3.63640800e+00
            +9.66528500e-04 * tc[1]
            -5.62338667e-08 * tc[2]
            -2.52474750e-11 * tc[3]
            +3.61651200e-15 * tc[4]
            +4.53413400e+04 * invT;
        /*species 12: CO2 */
        species[12] =
            +4.45362300e+00
            +1.57008450e-03 * tc[1]
            -4.26137000e-07 * tc[2]
            +5.98499250e-11 * tc[3]
            -3.33806600e-15 * tc[4]
            -4.89669600e+04 * invT;
        /*species 13: CO */
        species[13] =
            +3.02507800e+00
            +7.21344500e-04 * tc[1]
            -1.87694267e-07 * tc[2]
            +2.54645250e-11 * tc[3]
            -1.38219040e-15 * tc[4]
            -1.42683500e+04 * invT;
        /*species 14: CH2O */
        species[14] =
            +2.99560600e+00
            +3.34066050e-03 * tc[1]
            -8.76318333e-07 * tc[2]
            +1.18428825e-10 * tc[3]
            -6.42503400e-15 * tc[4]
            -1.53203700e+04 * invT;
        /*species 15: CH2GSG */
        species[15] =
            +3.55288900e+00
            +1.03339400e-03 * tc[1]
            -6.38038667e-08 * tc[2]
            -2.76168250e-11 * tc[3]
            +4.04270000e-15 * tc[4]
            +4.98497500e+04 * invT;
        /*species 16: CH3 */
        species[16] =
            +2.84405200e+00
            +3.06898700e-03 * tc[1]
            -7.43448333e-07 * tc[2]
            +9.46290250e-11 * tc[3]
            -4.90431800e-15 * tc[4]
            +1.64378100e+04 * invT;
        /*species 17: CH3O */
        species[17] =
            +3.77080000e+00
            +3.93574850e-03 * tc[1]
            -8.85461333e-07 * tc[2]
            +9.86107750e-11 * tc[3]
            -4.22523200e-15 * tc[4]
            +1.27832500e+02 * invT;
        /*species 18: CH4 */
        species[18] =
            +1.68347900e+00
            +5.11862000e-03 * tc[1]
            -1.29170967e-06 * tc[2]
            +1.69639625e-10 * tc[3]
            -9.00684600e-15 * tc[4]
            -1.00807900e+04 * invT;
        /*species 19: CH3OH */
        species[19] =
            +4.02906100e+00
            +4.68829650e-03 * tc[1]
            -1.01675133e-06 * tc[2]
            +1.08969825e-10 * tc[3]
            -4.44944600e-15 * tc[4]
            -2.61579100e+04 * invT;
        /*species 21: C2H5 */
        species[21] =
            +7.19048000e+00
            +3.24203850e-03 * tc[1]
            -2.14268833e-07 * tc[2]
            -5.86969750e-11 * tc[3]
            +7.76175400e-15 * tc[4]
            +1.06745500e+04 * invT;
        /*species 22: CH2CO */
        species[22] =
            +6.03881700e+00
            +2.90242000e-03 * tc[1]
            -6.40318000e-07 * tc[2]
            +6.98621250e-11 * tc[3]
            -2.91773600e-15 * tc[4]
            -8.58340200e+03 * invT;
        /*species 26: C2H2 */
        species[26] =
            +4.43677000e+00
            +2.68801950e-03 * tc[1]
            -6.37605667e-07 * tc[2]
            +8.21594750e-11 * tc[3]
            -4.31342000e-15 * tc[4]
            +2.56676600e+04 * invT;
        /*species 27: HCCO */
        species[27] =
            +6.75807300e+00
            +1.00020000e-03 * tc[1]
            -6.75869000e-08 * tc[2]
            -2.60283000e-11 * tc[3]
            +3.93033000e-15 * tc[4]
            +1.90151300e+04 * invT;
        /*species 28: C2H3 */
        species[28] =
            +5.93346800e+00
            +2.00887300e-03 * tc[1]
            -1.32224667e-07 * tc[2]
            -3.60316750e-11 * tc[3]
            +4.75728800e-15 * tc[4]
            +3.18543500e+04 * invT;
        /*species 29: CH2CHO */
        species[29] =
            +5.97567000e+00
            +4.06529550e-03 * tc[1]
            -9.14541333e-07 * tc[2]
            +1.01757600e-10 * tc[3]
            -4.35203400e-15 * tc[4]
            +4.90321800e+02 * invT;
        /*species 31: C2H4 */
        species[31] =
            +3.52841900e+00
            +5.74259000e-03 * tc[1]
            -1.47279500e-06 * tc[2]
            +1.96115025e-10 * tc[3]
            -1.05336960e-14 * tc[4]
            +4.42828900e+03 * invT;
        /*species 33: CH3CO */
        species[33] =
            +5.61227900e+00
            +4.22494300e-03 * tc[1]
            -9.51382333e-07 * tc[2]
            +1.05959400e-10 * tc[3]
            -4.53680800e-15 * tc[4]
            -5.18786300e+03 * invT;
        /*species 35: C3H2 */
        species[35] =
            +7.67098100e+00
            +1.37437450e-03 * tc[1]
            -1.45698100e-07 * tc[2]
            -1.61389975e-11 * tc[3]
            +3.32777400e-15 * tc[4]
            +6.25972200e+04 * invT;
        /*species 36: C3H3 */
        species[36] =
            +8.83104700e+00
            +2.17859750e-03 * tc[1]
            -1.36968900e-07 * tc[2]
            -5.92180750e-11 * tc[3]
            +8.75304000e-15 * tc[4]
            +3.84742000e+04 * invT;
        /*species 39: NXC3H7 */
        species[39] =
            +7.97829100e+00
            +7.88056500e-03 * tc[1]
            -1.72441433e-06 * tc[2]
            +1.86097300e-10 * tc[3]
            -7.64995600e-15 * tc[4]
            +7.57940200e+03 * invT;
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
        /*species 25: CH3O2H */
        species[25] =
            +3.23442817e+00
            +9.50648835e-03 * tc[1]
            -3.77954290e-06 * tc[2]
            +8.50766632e-10 * tc[3]
            -8.23660444e-14 * tc[4]
            -1.77197926e+04 * invT;
    } else {
        /*species 7: HO2 */
        species[7] =
            +4.10547423e+00
            +1.19226417e-03 * tc[1]
            -2.68782663e-07 * tc[2]
            +3.10479308e-11 * tc[3]
            -1.43280022e-15 * tc[4]
            +3.98127689e+02 * invT;
        /*species 25: CH3O2H */
        species[25] =
            +8.43117091e+00
            +4.03408955e-03 * tc[1]
            -9.23649737e-07 * tc[2]
            +1.07833061e-10 * tc[3]
            -5.01384292e-15 * tc[4]
            -1.96678771e+04 * invT;
    }

    /*species with midpoint at T=1384 kelvin */
    if (T < 1384) {
        /*species 20: C2H6 */
        species[20] =
            -2.52854344e-02
            +1.20382377e-02 * tc[1]
            -3.72978240e-06 * tc[2]
            +5.20852252e-10 * tc[3]
            -1.05973723e-14 * tc[4]
            -1.12345534e+04 * invT;
        /*species 40: NXC3H7O2 */
        species[40] =
            +2.10731492e+00
            +1.98082493e-02 * tc[1]
            -8.31638663e-06 * tc[2]
            +2.14862575e-09 * tc[3]
            -2.62480660e-13 * tc[4]
            -7.93745567e+03 * invT;
    } else {
        /*species 20: C2H6 */
        species[20] =
            +6.10683385e+00
            +6.46181805e-03 * tc[1]
            -1.47509065e-06 * tc[2]
            +1.71847932e-10 * tc[3]
            -7.97803464e-15 * tc[4]
            -1.37500014e+04 * invT;
        /*species 40: NXC3H7O2 */
        species[40] =
            +1.26327059e+01
            +8.49553630e-03 * tc[1]
            -1.96288958e-06 * tc[2]
            +2.30548849e-10 * tc[3]
            -1.07646135e-14 * tc[4]
            -1.19194652e+04 * invT;
    }

    /*species with midpoint at T=1376 kelvin */
    if (T < 1376) {
        /*species 23: HOCHO */
        species[23] =
            +1.43548185e+00
            +8.16815080e-03 * tc[1]
            -3.54191403e-06 * tc[2]
            +8.30332443e-10 * tc[3]
            -8.04352206e-14 * tc[4]
            -4.64616504e+04 * invT;
    } else {
        /*species 23: HOCHO */
        species[23] =
            +6.68733013e+00
            +2.57144684e-03 * tc[1]
            -6.07461710e-07 * tc[2]
            +7.24297908e-11 * tc[3]
            -3.41784398e-15 * tc[4]
            -4.83995400e+04 * invT;
    }

    /*species with midpoint at T=1385 kelvin */
    if (T < 1385) {
        /*species 24: CH3O2 */
        species[24] =
            +4.26146906e+00
            +5.04367995e-03 * tc[1]
            -1.07168728e-06 * tc[2]
            +5.23523168e-11 * tc[3]
            +8.36678206e-15 * tc[4]
            -6.84394259e+02 * invT;
        /*species 45: PXC4H9O2 */
        species[45] =
            +1.94363650e+00
            +2.57756581e-02 * tc[1]
            -1.09428133e-05 * tc[2]
            +2.82662150e-09 * tc[3]
            -3.40237212e-13 * tc[4]
            -1.08358103e+04 * invT;
    } else {
        /*species 24: CH3O2 */
        species[24] =
            +5.95787891e+00
            +3.95364313e-03 * tc[1]
            -8.94154113e-07 * tc[2]
            +1.03472834e-10 * tc[3]
            -4.78014660e-15 * tc[4]
            -1.53574838e+03 * invT;
        /*species 45: PXC4H9O2 */
        species[45] =
            +1.57845448e+01
            +1.07605455e-02 * tc[1]
            -2.48303006e-06 * tc[2]
            +2.91395178e-10 * tc[3]
            -1.35977122e-14 * tc[4]
            -1.60146054e+04 * invT;
    }

    /*species with midpoint at T=1388 kelvin */
    if (T < 1388) {
        /*species 30: C3H6 */
        species[30] =
            +3.94615444e-01
            +1.44553831e-02 * tc[1]
            -5.16289360e-06 * tc[2]
            +9.72035522e-10 * tc[3]
            -6.75780704e-14 * tc[4]
            +1.06688164e+03 * invT;
    } else {
        /*species 30: C3H6 */
        species[30] =
            +8.01595958e+00
            +6.85118170e-03 * tc[1]
            -1.55416578e-06 * tc[2]
            +1.80313601e-10 * tc[3]
            -8.34740252e-15 * tc[4]
            -1.87821271e+03 * invT;
    }

    /*species with midpoint at T=1389 kelvin */
    if (T < 1389) {
        /*species 32: C2H5O */
        species[32] =
            +4.94420708e-01
            +1.35887217e-02 * tc[1]
            -5.53030033e-06 * tc[2]
            +1.28801050e-09 * tc[3]
            -1.29699383e-13 * tc[4]
            -3.35252925e+03 * invT;
    } else {
        /*species 32: C2H5O */
        species[32] =
            +7.87339772e+00
            +5.65364535e-03 * tc[1]
            -1.28140474e-06 * tc[2]
            +1.48603526e-10 * tc[3]
            -6.87789076e-15 * tc[4]
            -6.07274953e+03 * invT;
    }

    /*species with midpoint at T=1382 kelvin */
    if (T < 1382) {
        /*species 34: C2H5O2 */
        species[34] =
            +2.26846188e+00
            +1.38471289e-02 * tc[1]
            -5.69347020e-06 * tc[2]
            +1.46987970e-09 * tc[3]
            -1.84179814e-13 * tc[4]
            -5.03880758e+03 * invT;
        /*species 50: C7H15X2 */
        species[50] =
            -3.79155767e-02
            +3.78363285e-02 * tc[1]
            -1.35824545e-05 * tc[2]
            +2.33169736e-09 * tc[3]
            -9.84721490e-14 * tc[4]
            -2.35605303e+03 * invT;
    } else {
        /*species 34: C2H5O2 */
        species[34] =
            +9.48696023e+00
            +6.22362725e-03 * tc[1]
            -1.44053859e-06 * tc[2]
            +1.69395758e-10 * tc[3]
            -7.91569136e-15 * tc[4]
            -7.82481795e+03 * invT;
        /*species 50: C7H15X2 */
        species[50] =
            +2.16368842e+01
            +1.61662402e-02 * tc[1]
            -3.64246023e-06 * tc[2]
            +4.20892650e-10 * tc[3]
            -1.94354818e-14 * tc[4]
            -1.05873616e+04 * invT;
    }

    /*species with midpoint at T=1400 kelvin */
    if (T < 1400) {
        /*species 37: C3H4XA */
        species[37] =
            +2.53983100e+00
            +8.16718500e-03 * tc[1]
            -5.88316667e-07 * tc[2]
            -1.16184125e-09 * tc[3]
            +3.45826200e-13 * tc[4]
            +2.25124300e+04 * invT;
    } else {
        /*species 37: C3H4XA */
        species[37] =
            +9.77625600e+00
            +2.65106900e-03 * tc[1]
            -1.23370600e-07 * tc[2]
            -7.56596500e-11 * tc[3]
            +1.01791620e-14 * tc[4]
            +1.95497200e+04 * invT;
    }

    /*species with midpoint at T=1397 kelvin */
    if (T < 1397) {
        /*species 38: C3H5XA */
        species[38] =
            -5.29131958e-01
            +1.67279550e-02 * tc[1]
            -8.44670090e-06 * tc[2]
            +2.57164385e-09 * tc[3]
            -3.46516680e-13 * tc[4]
            +1.93834226e+04 * invT;
        /*species 48: C5H11X1 */
        species[48] =
            -9.05255912e-01
            +3.05316426e-02 * tc[1]
            -1.36497275e-05 * tc[2]
            +3.65233675e-09 * tc[3]
            -4.37719230e-13 * tc[4]
            +4.83995303e+03 * invT;
    } else {
        /*species 38: C3H5XA */
        species[38] =
            +8.45883958e+00
            +5.63477415e-03 * tc[1]
            -1.27930955e-06 * tc[2]
            +1.48514780e-10 * tc[3]
            -6.87836060e-15 * tc[4]
            +1.63576092e+04 * invT;
        /*species 48: C5H11X1 */
        species[48] =
            +1.53234740e+01
            +1.19520600e-02 * tc[1]
            -2.71590540e-06 * tc[2]
            +3.15440590e-10 * tc[3]
            -1.46135467e-14 * tc[4]
            -9.23241637e+02 * invT;
    }

    /*species with midpoint at T=1398 kelvin */
    if (T < 1398) {
        /*species 41: C4H6 */
        species[41] =
            -1.43095121e+00
            +2.39353031e-02 * tc[1]
            -1.38482267e-05 * tc[2]
            +4.78873880e-09 * tc[3]
            -7.14317014e-13 * tc[4]
            +1.17551314e+04 * invT;
    } else {
        /*species 41: C4H6 */
        species[41] =
            +1.11633789e+01
            +6.85819825e-03 * tc[1]
            -1.56571928e-06 * tc[2]
            +1.82423459e-10 * tc[3]
            -8.46972406e-15 * tc[4]
            +7.79039770e+03 * invT;
    }

    /*species with midpoint at T=1392 kelvin */
    if (T < 1392) {
        /*species 42: C4H7 */
        species[42] =
            -3.50508352e-01
            +2.13255622e-02 * tc[1]
            -9.69931243e-06 * tc[2]
            +2.63509785e-09 * tc[3]
            -3.20119708e-13 * tc[4]
            +1.49933591e+04 * invT;
        /*species 43: C4H8X1 */
        species[43] =
            -8.31372089e-01
            +2.26290489e-02 * tc[1]
            -9.78861863e-06 * tc[2]
            +2.50551090e-09 * tc[3]
            -2.86383360e-13 * tc[4]
            -1.57875035e+03 * invT;
        /*species 46: C5H9 */
        species[46] =
            -1.38013950e+00
            +2.78804243e-02 * tc[1]
            -1.23381309e-05 * tc[2]
            +3.17209752e-09 * tc[3]
            -3.57077670e-13 * tc[4]
            +1.25589824e+04 * invT;
        /*species 47: C5H10X1 */
        species[47] =
            -1.06223481e+00
            +2.87109147e-02 * tc[1]
            -1.24828963e-05 * tc[2]
            +3.18412472e-09 * tc[3]
            -3.59219578e-13 * tc[4]
            -4.46546666e+03 * invT;
        /*species 49: C6H12X1 */
        species[49] =
            -1.35275205e+00
            +3.49327713e-02 * tc[1]
            -1.53136007e-05 * tc[2]
            +3.92418358e-09 * tc[3]
            -4.42592350e-13 * tc[4]
            -7.34368617e+03 * invT;
    } else {
        /*species 42: C4H7 */
        species[42] =
            +1.12103578e+01
            +8.02415980e-03 * tc[1]
            -1.82167431e-06 * tc[2]
            +2.11485263e-10 * tc[3]
            -9.79545478e-15 * tc[4]
            +1.09041937e+04 * invT;
        /*species 43: C4H8X1 */
        species[43] =
            +1.13508668e+01
            +9.03089385e-03 * tc[1]
            -2.05364343e-06 * tc[2]
            +2.38663240e-10 * tc[3]
            -1.10617928e-14 * tc[4]
            -5.97871038e+03 * invT;
        /*species 46: C5H9 */
        species[46] =
            +1.41860454e+01
            +1.03564449e-02 * tc[1]
            -2.35653539e-06 * tc[2]
            +2.74017833e-10 * tc[3]
            -1.27064442e-14 * tc[4]
            +7.00496135e+03 * invT;
        /*species 47: C5H10X1 */
        species[47] =
            +1.45851539e+01
            +1.12036235e-02 * tc[1]
            -2.54449342e-06 * tc[2]
            +2.95472415e-10 * tc[3]
            -1.36877028e-14 * tc[4]
            -1.00898205e+04 * invT;
        /*species 49: C6H12X1 */
        species[49] =
            +1.78337529e+01
            +1.33688829e-02 * tc[1]
            -3.03345591e-06 * tc[2]
            +3.52049420e-10 * tc[3]
            -1.63024849e-14 * tc[4]
            -1.42062860e+04 * invT;
    }

    /*species with midpoint at T=1395 kelvin */
    if (T < 1395) {
        /*species 44: PXC4H9 */
        species[44] =
            -4.37779725e-01
            +2.39486182e-02 * tc[1]
            -1.04674386e-05 * tc[2]
            +2.74466180e-09 * tc[3]
            -3.24021328e-13 * tc[4]
            +7.68945248e+03 * invT;
    } else {
        /*species 44: PXC4H9 */
        species[44] =
            +1.21510082e+01
            +9.71553585e-03 * tc[1]
            -2.20525983e-06 * tc[2]
            +2.55937840e-10 * tc[3]
            -1.18505941e-14 * tc[4]
            +3.17231942e+03 * invT;
    }

    /*species with midpoint at T=1391 kelvin */
    if (T < 1391) {
        /*species 51: NXC7H16 */
        species[51] =
            -1.26836187e+00
            +4.27177910e-02 * tc[1]
            -1.75115595e-05 * tc[2]
            +4.07364302e-09 * tc[3]
            -4.04789850e-13 * tc[4]
            -2.56586565e+04 * invT;
    } else {
        /*species 51: NXC7H16 */
        species[51] =
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
        /*species 9: CH */
        species[9] =
            +3.20020200e+00 * tc[0]
            +2.07287600e-03 * tc[1]
            -2.56721550e-06 * tc[2]
            +1.91129667e-09 * tc[3]
            -4.88883250e-13 * tc[4]
            +3.33158800e+00 ;
        /*species 10: HCO */
        species[10] =
            +2.89833000e+00 * tc[0]
            +6.19914700e-03 * tc[1]
            -4.81154200e-06 * tc[2]
            +3.63275000e-09 * tc[3]
            -1.14372125e-12 * tc[4]
            +8.98361400e+00 ;
        /*species 11: CH2 */
        species[11] =
            +3.76223700e+00 * tc[0]
            +1.15981900e-03 * tc[1]
            +1.24479250e-07 * tc[2]
            +2.93361200e-10 * tc[3]
            -1.83310875e-13 * tc[4]
            +1.71257800e+00 ;
        /*species 12: CO2 */
        species[12] =
            +2.27572500e+00 * tc[0]
            +9.92207200e-03 * tc[1]
            -5.20455500e-06 * tc[2]
            +2.28889567e-09 * tc[3]
            -5.29320000e-13 * tc[4]
            +1.01884900e+01 ;
        /*species 13: CO */
        species[13] =
            +3.26245200e+00 * tc[0]
            +1.51194100e-03 * tc[1]
            -1.94087750e-06 * tc[2]
            +1.86064800e-09 * tc[3]
            -6.18737750e-13 * tc[4]
            +4.84889700e+00 ;
        /*species 14: CH2O */
        species[14] =
            +1.65273100e+00 * tc[0]
            +1.26314400e-02 * tc[1]
            -9.44084000e-06 * tc[2]
            +6.83343667e-09 * tc[3]
            -2.10330925e-12 * tc[4]
            +1.37848200e+01 ;
        /*species 15: CH2GSG */
        species[15] =
            +3.97126500e+00 * tc[0]
            -1.69908900e-04 * tc[1]
            +5.12684500e-07 * tc[2]
            +8.30850333e-10 * tc[3]
            -4.95316500e-13 * tc[4]
            +5.75320700e-02 ;
        /*species 16: CH3 */
        species[16] =
            +2.43044300e+00 * tc[0]
            +1.11241000e-02 * tc[1]
            -8.40110000e-06 * tc[2]
            +5.40609667e-09 * tc[3]
            -1.46623825e-12 * tc[4]
            +6.78979400e+00 ;
        /*species 17: CH3O */
        species[17] =
            +2.10620400e+00 * tc[0]
            +7.21659500e-03 * tc[1]
            +2.66923600e-06 * tc[2]
            -2.45921200e-09 * tc[3]
            +5.18902750e-13 * tc[4]
            +1.31521800e+01 ;
        /*species 18: CH4 */
        species[18] =
            +7.78741500e-01 * tc[0]
            +1.74766800e-02 * tc[1]
            -1.39170450e-05 * tc[2]
            +1.01656933e-08 * tc[3]
            -3.05982750e-12 * tc[4]
            +1.37221900e+01 ;
        /*species 19: CH3OH */
        species[19] =
            +2.66011500e+00 * tc[0]
            +7.34150800e-03 * tc[1]
            +3.58502550e-06 * tc[2]
            -2.93106467e-09 * tc[3]
            +5.97642500e-13 * tc[4]
            +1.12326300e+01 ;
        /*species 21: C2H5 */
        species[21] =
            +2.69070200e+00 * tc[0]
            +8.71913300e-03 * tc[1]
            +2.20991950e-06 * tc[2]
            +3.11290100e-10 * tc[3]
            -9.81943250e-13 * tc[4]
            +1.21382000e+01 ;
        /*species 22: CH2CO */
        species[22] =
            +2.97497100e+00 * tc[0]
            +1.21187100e-02 * tc[1]
            -1.17252300e-06 * tc[2]
            -2.15556167e-09 * tc[3]
            +9.76412250e-13 * tc[4]
            +8.67355300e+00 ;
        /*species 26: C2H2 */
        species[26] =
            +2.01356200e+00 * tc[0]
            +1.51904500e-02 * tc[1]
            -8.08159500e-06 * tc[2]
            +3.02633067e-09 * tc[3]
            -4.78186500e-13 * tc[4]
            +8.80537800e+00 ;
        /*species 27: HCCO */
        species[27] =
            +5.04796500e+00 * tc[0]
            +4.45347800e-03 * tc[1]
            +1.13414150e-07 * tc[2]
            -4.94031667e-10 * tc[3]
            +5.62685500e-14 * tc[4]
            +4.81843900e-01 ;
        /*species 28: C2H3 */
        species[28] =
            +2.45927600e+00 * tc[0]
            +7.37147600e-03 * tc[1]
            +1.05493650e-06 * tc[2]
            -4.40547333e-10 * tc[3]
            -2.96196000e-13 * tc[4]
            +1.15562000e+01 ;
        /*species 29: CH2CHO */
        species[29] =
            +3.40906200e+00 * tc[0]
            +1.07385700e-02 * tc[1]
            +9.45746000e-07 * tc[2]
            -2.38619433e-09 * tc[3]
            +7.16846250e-13 * tc[4]
            +9.55829000e+00 ;
        /*species 31: C2H4 */
        species[31] =
            -8.61488000e-01 * tc[0]
            +2.79616300e-02 * tc[1]
            -1.69433850e-05 * tc[2]
            +9.28384000e-09 * tc[3]
            -2.43446975e-12 * tc[4]
            +2.42114900e+01 ;
        /*species 33: CH3CO */
        species[33] =
            +3.12527800e+00 * tc[0]
            +9.77822000e-03 * tc[1]
            +2.26072400e-06 * tc[2]
            -3.00315400e-09 * tc[3]
            +7.98429500e-13 * tc[4]
            +1.12288500e+01 ;
        /*species 35: C3H2 */
        species[35] =
            +3.16671400e+00 * tc[0]
            +2.48257200e-02 * tc[1]
            -2.29581850e-05 * tc[2]
            +1.42267300e-08 * tc[3]
            -3.70538000e-12 * tc[4]
            +8.86944600e+00 ;
        /*species 36: C3H3 */
        species[36] =
            +4.75420000e+00 * tc[0]
            +1.10802800e-02 * tc[1]
            +1.39666150e-07 * tc[2]
            -1.82640400e-09 * tc[3]
            +4.87407250e-13 * tc[4]
            +5.85454900e-01 ;
        /*species 39: NXC3H7 */
        species[39] =
            +1.92253700e+00 * tc[0]
            +2.47892700e-02 * tc[1]
            +9.05124500e-07 * tc[2]
            -5.94422000e-09 * tc[3]
            +2.14574900e-12 * tc[4]
            +1.39927100e+01 ;
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
        /*species 9: CH */
        species[9] =
            +2.19622300e+00 * tc[0]
            +2.34038100e-03 * tc[1]
            -3.52910050e-07 * tc[2]
            +3.00252733e-11 * tc[3]
            -9.63760000e-16 * tc[4]
            +9.17837300e+00 ;
        /*species 10: HCO */
        species[10] =
            +3.55727100e+00 * tc[0]
            +3.34557300e-03 * tc[1]
            -6.67503000e-07 * tc[2]
            +8.23524333e-11 * tc[3]
            -4.28462750e-15 * tc[4]
            +5.55229900e+00 ;
        /*species 11: CH2 */
        species[11] =
            +3.63640800e+00 * tc[0]
            +1.93305700e-03 * tc[1]
            -8.43508000e-08 * tc[2]
            -3.36633000e-11 * tc[3]
            +4.52064000e-15 * tc[4]
            +2.15656100e+00 ;
        /*species 12: CO2 */
        species[12] =
            +4.45362300e+00 * tc[0]
            +3.14016900e-03 * tc[1]
            -6.39205500e-07 * tc[2]
            +7.97999000e-11 * tc[3]
            -4.17258250e-15 * tc[4]
            -9.55395900e-01 ;
        /*species 13: CO */
        species[13] =
            +3.02507800e+00 * tc[0]
            +1.44268900e-03 * tc[1]
            -2.81541400e-07 * tc[2]
            +3.39527000e-11 * tc[3]
            -1.72773800e-15 * tc[4]
            +6.10821800e+00 ;
        /*species 14: CH2O */
        species[14] =
            +2.99560600e+00 * tc[0]
            +6.68132100e-03 * tc[1]
            -1.31447750e-06 * tc[2]
            +1.57905100e-10 * tc[3]
            -8.03129250e-15 * tc[4]
            +6.91257200e+00 ;
        /*species 15: CH2GSG */
        species[15] =
            +3.55288900e+00 * tc[0]
            +2.06678800e-03 * tc[1]
            -9.57058000e-08 * tc[2]
            -3.68224333e-11 * tc[3]
            +5.05337500e-15 * tc[4]
            +1.68657000e+00 ;
        /*species 16: CH3 */
        species[16] =
            +2.84405200e+00 * tc[0]
            +6.13797400e-03 * tc[1]
            -1.11517250e-06 * tc[2]
            +1.26172033e-10 * tc[3]
            -6.13039750e-15 * tc[4]
            +5.45269700e+00 ;
        /*species 17: CH3O */
        species[17] =
            +3.77080000e+00 * tc[0]
            +7.87149700e-03 * tc[1]
            -1.32819200e-06 * tc[2]
            +1.31481033e-10 * tc[3]
            -5.28154000e-15 * tc[4]
            +2.92957500e+00 ;
        /*species 18: CH4 */
        species[18] =
            +1.68347900e+00 * tc[0]
            +1.02372400e-02 * tc[1]
            -1.93756450e-06 * tc[2]
            +2.26186167e-10 * tc[3]
            -1.12585575e-14 * tc[4]
            +9.62339500e+00 ;
        /*species 19: CH3OH */
        species[19] =
            +4.02906100e+00 * tc[0]
            +9.37659300e-03 * tc[1]
            -1.52512700e-06 * tc[2]
            +1.45293100e-10 * tc[3]
            -5.56180750e-15 * tc[4]
            +2.37819600e+00 ;
        /*species 21: C2H5 */
        species[21] =
            +7.19048000e+00 * tc[0]
            +6.48407700e-03 * tc[1]
            -3.21403250e-07 * tc[2]
            -7.82626333e-11 * tc[3]
            +9.70219250e-15 * tc[4]
            -1.47808900e+01 ;
        /*species 22: CH2CO */
        species[22] =
            +6.03881700e+00 * tc[0]
            +5.80484000e-03 * tc[1]
            -9.60477000e-07 * tc[2]
            +9.31495000e-11 * tc[3]
            -3.64717000e-15 * tc[4]
            -7.65758100e+00 ;
        /*species 26: C2H2 */
        species[26] =
            +4.43677000e+00 * tc[0]
            +5.37603900e-03 * tc[1]
            -9.56408500e-07 * tc[2]
            +1.09545967e-10 * tc[3]
            -5.39177500e-15 * tc[4]
            -2.80033800e+00 ;
        /*species 27: HCCO */
        species[27] =
            +6.75807300e+00 * tc[0]
            +2.00040000e-03 * tc[1]
            -1.01380350e-07 * tc[2]
            -3.47044000e-11 * tc[3]
            +4.91291250e-15 * tc[4]
            -9.07126200e+00 ;
        /*species 28: C2H3 */
        species[28] =
            +5.93346800e+00 * tc[0]
            +4.01774600e-03 * tc[1]
            -1.98337000e-07 * tc[2]
            -4.80422333e-11 * tc[3]
            +5.94661000e-15 * tc[4]
            -8.53031300e+00 ;
        /*species 29: CH2CHO */
        species[29] =
            +5.97567000e+00 * tc[0]
            +8.13059100e-03 * tc[1]
            -1.37181200e-06 * tc[2]
            +1.35676800e-10 * tc[3]
            -5.44004250e-15 * tc[4]
            -5.04525100e+00 ;
        /*species 31: C2H4 */
        species[31] =
            +3.52841900e+00 * tc[0]
            +1.14851800e-02 * tc[1]
            -2.20919250e-06 * tc[2]
            +2.61486700e-10 * tc[3]
            -1.31671200e-14 * tc[4]
            +2.23038900e+00 ;
        /*species 33: CH3CO */
        species[33] =
            +5.61227900e+00 * tc[0]
            +8.44988600e-03 * tc[1]
            -1.42707350e-06 * tc[2]
            +1.41279200e-10 * tc[3]
            -5.67101000e-15 * tc[4]
            -3.27494900e+00 ;
        /*species 35: C3H2 */
        species[35] =
            +7.67098100e+00 * tc[0]
            +2.74874900e-03 * tc[1]
            -2.18547150e-07 * tc[2]
            -2.15186633e-11 * tc[3]
            +4.15971750e-15 * tc[4]
            -1.23689000e+01 ;
        /*species 36: C3H3 */
        species[36] =
            +8.83104700e+00 * tc[0]
            +4.35719500e-03 * tc[1]
            -2.05453350e-07 * tc[2]
            -7.89574333e-11 * tc[3]
            +1.09413000e-14 * tc[4]
            -2.17791900e+01 ;
        /*species 39: NXC3H7 */
        species[39] =
            +7.97829100e+00 * tc[0]
            +1.57611300e-02 * tc[1]
            -2.58662150e-06 * tc[2]
            +2.48129733e-10 * tc[3]
            -9.56244500e-15 * tc[4]
            -1.93561100e+01 ;
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
        /*species 25: CH3O2H */
        species[25] =
            +3.23442817e+00 * tc[0]
            +1.90129767e-02 * tc[1]
            -5.66931435e-06 * tc[2]
            +1.13435551e-09 * tc[3]
            -1.02957555e-13 * tc[4]
            +9.25623949e+00 ;
    } else {
        /*species 7: HO2 */
        species[7] =
            +4.10547423e+00 * tc[0]
            +2.38452835e-03 * tc[1]
            -4.03173995e-07 * tc[2]
            +4.13972410e-11 * tc[3]
            -1.79100027e-15 * tc[4]
            +3.12515836e+00 ;
        /*species 25: CH3O2H */
        species[25] =
            +8.43117091e+00 * tc[0]
            +8.06817909e-03 * tc[1]
            -1.38547461e-06 * tc[2]
            +1.43777414e-10 * tc[3]
            -6.26730365e-15 * tc[4]
            -1.91170629e+01 ;
    }

    /*species with midpoint at T=1384 kelvin */
    if (T < 1384) {
        /*species 20: C2H6 */
        species[20] =
            -2.52854344e-02 * tc[0]
            +2.40764754e-02 * tc[1]
            -5.59467360e-06 * tc[2]
            +6.94469670e-10 * tc[3]
            -1.32467154e-14 * tc[4]
            +2.11648750e+01 ;
        /*species 40: NXC3H7O2 */
        species[40] =
            +2.10731492e+00 * tc[0]
            +3.96164986e-02 * tc[1]
            -1.24745800e-05 * tc[2]
            +2.86483433e-09 * tc[3]
            -3.28100825e-13 * tc[4]
            +1.89082748e+01 ;
    } else {
        /*species 20: C2H6 */
        species[20] =
            +6.10683385e+00 * tc[0]
            +1.29236361e-02 * tc[1]
            -2.21263598e-06 * tc[2]
            +2.29130575e-10 * tc[3]
            -9.97254330e-15 * tc[4]
            -1.30081250e+01 ;
        /*species 40: NXC3H7O2 */
        species[40] =
            +1.26327059e+01 * tc[0]
            +1.69910726e-02 * tc[1]
            -2.94433436e-06 * tc[2]
            +3.07398465e-10 * tc[3]
            -1.34557669e-14 * tc[4]
            -3.85349297e+01 ;
    }

    /*species with midpoint at T=1376 kelvin */
    if (T < 1376) {
        /*species 23: HOCHO */
        species[23] =
            +1.43548185e+00 * tc[0]
            +1.63363016e-02 * tc[1]
            -5.31287105e-06 * tc[2]
            +1.10710992e-09 * tc[3]
            -1.00544026e-13 * tc[4]
            +1.72885798e+01 ;
    } else {
        /*species 23: HOCHO */
        species[23] =
            +6.68733013e+00 * tc[0]
            +5.14289368e-03 * tc[1]
            -9.11192565e-07 * tc[2]
            +9.65730543e-11 * tc[3]
            -4.27230498e-15 * tc[4]
            -1.13104798e+01 ;
    }

    /*species with midpoint at T=1385 kelvin */
    if (T < 1385) {
        /*species 24: CH3O2 */
        species[24] =
            +4.26146906e+00 * tc[0]
            +1.00873599e-02 * tc[1]
            -1.60753092e-06 * tc[2]
            +6.98030890e-11 * tc[3]
            +1.04584776e-14 * tc[4]
            +5.16330320e+00 ;
        /*species 45: PXC4H9O2 */
        species[45] =
            +1.94363650e+00 * tc[0]
            +5.15513163e-02 * tc[1]
            -1.64142200e-05 * tc[2]
            +3.76882867e-09 * tc[3]
            -4.25296515e-13 * tc[4]
            +2.13503149e+01 ;
    } else {
        /*species 24: CH3O2 */
        species[24] =
            +5.95787891e+00 * tc[0]
            +7.90728626e-03 * tc[1]
            -1.34123117e-06 * tc[2]
            +1.37963779e-10 * tc[3]
            -5.97518325e-15 * tc[4]
            -4.71963886e+00 ;
        /*species 45: PXC4H9O2 */
        species[45] =
            +1.57845448e+01 * tc[0]
            +2.15210910e-02 * tc[1]
            -3.72454509e-06 * tc[2]
            +3.88526903e-10 * tc[3]
            -1.69971402e-14 * tc[4]
            -5.40388525e+01 ;
    }

    /*species with midpoint at T=1388 kelvin */
    if (T < 1388) {
        /*species 30: C3H6 */
        species[30] =
            +3.94615444e-01 * tc[0]
            +2.89107662e-02 * tc[1]
            -7.74434040e-06 * tc[2]
            +1.29604736e-09 * tc[3]
            -8.44725880e-14 * tc[4]
            +2.19003736e+01 ;
    } else {
        /*species 30: C3H6 */
        species[30] =
            +8.01595958e+00 * tc[0]
            +1.37023634e-02 * tc[1]
            -2.33124867e-06 * tc[2]
            +2.40418134e-10 * tc[3]
            -1.04342532e-14 * tc[4]
            -2.00160668e+01 ;
    }

    /*species with midpoint at T=1389 kelvin */
    if (T < 1389) {
        /*species 32: C2H5O */
        species[32] =
            +4.94420708e-01 * tc[0]
            +2.71774434e-02 * tc[1]
            -8.29545050e-06 * tc[2]
            +1.71734733e-09 * tc[3]
            -1.62124229e-13 * tc[4]
            +2.28079378e+01 ;
    } else {
        /*species 32: C2H5O */
        species[32] =
            +7.87339772e+00 * tc[0]
            +1.13072907e-02 * tc[1]
            -1.92210711e-06 * tc[2]
            +1.98138035e-10 * tc[3]
            -8.59736345e-15 * tc[4]
            -1.73416790e+01 ;
    }

    /*species with midpoint at T=1382 kelvin */
    if (T < 1382) {
        /*species 34: C2H5O2 */
        species[34] =
            +2.26846188e+00 * tc[0]
            +2.76942578e-02 * tc[1]
            -8.54020530e-06 * tc[2]
            +1.95983959e-09 * tc[3]
            -2.30224767e-13 * tc[4]
            +1.64773972e+01 ;
        /*species 50: C7H15X2 */
        species[50] =
            -3.79155767e-02 * tc[0]
            +7.56726570e-02 * tc[1]
            -2.03736817e-05 * tc[2]
            +3.10892981e-09 * tc[3]
            -1.23090186e-13 * tc[4]
            +3.37321506e+01 ;
    } else {
        /*species 34: C2H5O2 */
        species[34] =
            +9.48696023e+00 * tc[0]
            +1.24472545e-02 * tc[1]
            -2.16080788e-06 * tc[2]
            +2.25861011e-10 * tc[3]
            -9.89461420e-15 * tc[4]
            -2.30613020e+01 ;
        /*species 50: C7H15X2 */
        species[50] =
            +2.16368842e+01 * tc[0]
            +3.23324804e-02 * tc[1]
            -5.46369035e-06 * tc[2]
            +5.61190200e-10 * tc[3]
            -2.42943523e-14 * tc[4]
            -8.52209653e+01 ;
    }

    /*species with midpoint at T=1400 kelvin */
    if (T < 1400) {
        /*species 37: C3H4XA */
        species[37] =
            +2.53983100e+00 * tc[0]
            +1.63343700e-02 * tc[1]
            -8.82475000e-07 * tc[2]
            -1.54912167e-09 * tc[3]
            +4.32282750e-13 * tc[4]
            +9.93570200e+00 ;
    } else {
        /*species 37: C3H4XA */
        species[37] =
            +9.77625600e+00 * tc[0]
            +5.30213800e-03 * tc[1]
            -1.85055900e-07 * tc[2]
            -1.00879533e-10 * tc[3]
            +1.27239525e-14 * tc[4]
            -3.07706100e+01 ;
    }

    /*species with midpoint at T=1397 kelvin */
    if (T < 1397) {
        /*species 38: C3H5XA */
        species[38] =
            -5.29131958e-01 * tc[0]
            +3.34559100e-02 * tc[1]
            -1.26700514e-05 * tc[2]
            +3.42885847e-09 * tc[3]
            -4.33145850e-13 * tc[4]
            +2.53067131e+01 ;
        /*species 48: C5H11X1 */
        species[48] =
            -9.05255912e-01 * tc[0]
            +6.10632852e-02 * tc[1]
            -2.04745912e-05 * tc[2]
            +4.86978233e-09 * tc[3]
            -5.47149037e-13 * tc[4]
            +3.25574963e+01 ;
    } else {
        /*species 38: C3H5XA */
        species[38] =
            +8.45883958e+00 * tc[0]
            +1.12695483e-02 * tc[1]
            -1.91896432e-06 * tc[2]
            +1.98019706e-10 * tc[3]
            -8.59795075e-15 * tc[4]
            -2.25809450e+01 ;
        /*species 48: C5H11X1 */
        species[48] =
            +1.53234740e+01 * tc[0]
            +2.39041200e-02 * tc[1]
            -4.07385810e-06 * tc[2]
            +4.20587453e-10 * tc[3]
            -1.82669334e-14 * tc[4]
            -5.49528859e+01 ;
    }

    /*species with midpoint at T=1398 kelvin */
    if (T < 1398) {
        /*species 41: C4H6 */
        species[41] =
            -1.43095121e+00 * tc[0]
            +4.78706062e-02 * tc[1]
            -2.07723400e-05 * tc[2]
            +6.38498507e-09 * tc[3]
            -8.92896267e-13 * tc[4]
            +2.90825833e+01 ;
    } else {
        /*species 41: C4H6 */
        species[41] =
            +1.11633789e+01 * tc[0]
            +1.37163965e-02 * tc[1]
            -2.34857892e-06 * tc[2]
            +2.43231279e-10 * tc[3]
            -1.05871551e-14 * tc[4]
            -3.69847949e+01 ;
    }

    /*species with midpoint at T=1392 kelvin */
    if (T < 1392) {
        /*species 42: C4H7 */
        species[42] =
            -3.50508352e-01 * tc[0]
            +4.26511243e-02 * tc[1]
            -1.45489687e-05 * tc[2]
            +3.51346380e-09 * tc[3]
            -4.00149635e-13 * tc[4]
            +2.67295696e+01 ;
        /*species 43: C4H8X1 */
        species[43] =
            -8.31372089e-01 * tc[0]
            +4.52580978e-02 * tc[1]
            -1.46829280e-05 * tc[2]
            +3.34068120e-09 * tc[3]
            -3.57979200e-13 * tc[4]
            +2.95084236e+01 ;
        /*species 46: C5H9 */
        species[46] =
            -1.38013950e+00 * tc[0]
            +5.57608487e-02 * tc[1]
            -1.85071964e-05 * tc[2]
            +4.22946337e-09 * tc[3]
            -4.46347087e-13 * tc[4]
            +3.26441304e+01 ;
        /*species 47: C5H10X1 */
        species[47] =
            -1.06223481e+00 * tc[0]
            +5.74218294e-02 * tc[1]
            -1.87243445e-05 * tc[2]
            +4.24549963e-09 * tc[3]
            -4.49024472e-13 * tc[4]
            +3.22739790e+01 ;
        /*species 49: C6H12X1 */
        species[49] =
            -1.35275205e+00 * tc[0]
            +6.98655426e-02 * tc[1]
            -2.29704011e-05 * tc[2]
            +5.23224477e-09 * tc[3]
            -5.53240438e-13 * tc[4]
            +3.53120691e+01 ;
    } else {
        /*species 42: C4H7 */
        species[42] =
            +1.12103578e+01 * tc[0]
            +1.60483196e-02 * tc[1]
            -2.73251146e-06 * tc[2]
            +2.81980351e-10 * tc[3]
            -1.22443185e-14 * tc[4]
            -3.55593015e+01 ;
        /*species 43: C4H8X1 */
        species[43] =
            +1.13508668e+01 * tc[0]
            +1.80617877e-02 * tc[1]
            -3.08046515e-06 * tc[2]
            +3.18217653e-10 * tc[3]
            -1.38272410e-14 * tc[4]
            -3.64369438e+01 ;
        /*species 46: C5H9 */
        species[46] =
            +1.41860454e+01 * tc[0]
            +2.07128899e-02 * tc[1]
            -3.53480309e-06 * tc[2]
            +3.65357110e-10 * tc[3]
            -1.58830552e-14 * tc[4]
            -5.14501773e+01 ;
        /*species 47: C5H10X1 */
        species[47] =
            +1.45851539e+01 * tc[0]
            +2.24072471e-02 * tc[1]
            -3.81674012e-06 * tc[2]
            +3.93963220e-10 * tc[3]
            -1.71096285e-14 * tc[4]
            -5.23683936e+01 ;
        /*species 49: C6H12X1 */
        species[49] =
            +1.78337529e+01 * tc[0]
            +2.67377658e-02 * tc[1]
            -4.55018387e-06 * tc[2]
            +4.69399227e-10 * tc[3]
            -2.03781061e-14 * tc[4]
            -6.83818851e+01 ;
    }

    /*species with midpoint at T=1395 kelvin */
    if (T < 1395) {
        /*species 44: PXC4H9 */
        species[44] =
            -4.37779725e-01 * tc[0]
            +4.78972364e-02 * tc[1]
            -1.57011580e-05 * tc[2]
            +3.65954907e-09 * tc[3]
            -4.05026660e-13 * tc[4]
            +2.86852732e+01 ;
    } else {
        /*species 44: PXC4H9 */
        species[44] =
            +1.21510082e+01 * tc[0]
            +1.94310717e-02 * tc[1]
            -3.30788975e-06 * tc[2]
            +3.41250453e-10 * tc[3]
            -1.48132427e-14 * tc[4]
            -3.93425822e+01 ;
    }

    /*species with midpoint at T=1391 kelvin */
    if (T < 1391) {
        /*species 51: NXC7H16 */
        species[51] =
            -1.26836187e+00 * tc[0]
            +8.54355820e-02 * tc[1]
            -2.62673393e-05 * tc[2]
            +5.43152403e-09 * tc[3]
            -5.05987313e-13 * tc[4]
            +3.53732912e+01 ;
    } else {
        /*species 51: NXC7H16 */
        species[51] =
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
    *LENIMC = 210;}


void egtransetLENRMC(int* LENRMC ) {
    *LENRMC = 53196;}


void egtransetNO(int* NO ) {
    *NO = 4;}


void egtransetKK(int* KK ) {
    *KK = 52;}


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
    WT[9] = 1.30191200E+01;
    WT[10] = 2.90185200E+01;
    WT[11] = 1.40270900E+01;
    WT[12] = 4.40099500E+01;
    WT[13] = 2.80105500E+01;
    WT[14] = 3.00264900E+01;
    WT[15] = 1.40270900E+01;
    WT[16] = 1.50350600E+01;
    WT[17] = 3.10344600E+01;
    WT[18] = 1.60430300E+01;
    WT[19] = 3.20424300E+01;
    WT[20] = 3.00701200E+01;
    WT[21] = 2.90621500E+01;
    WT[22] = 4.20376400E+01;
    WT[23] = 4.60258900E+01;
    WT[24] = 4.70338600E+01;
    WT[25] = 4.80418300E+01;
    WT[26] = 2.60382400E+01;
    WT[27] = 4.10296700E+01;
    WT[28] = 2.70462100E+01;
    WT[29] = 4.30456100E+01;
    WT[30] = 4.20812700E+01;
    WT[31] = 2.80541800E+01;
    WT[32] = 4.50615500E+01;
    WT[33] = 4.30456100E+01;
    WT[34] = 6.10609500E+01;
    WT[35] = 3.80493900E+01;
    WT[36] = 3.90573600E+01;
    WT[37] = 4.00653300E+01;
    WT[38] = 4.10733000E+01;
    WT[39] = 4.30892400E+01;
    WT[40] = 7.50880400E+01;
    WT[41] = 5.40924200E+01;
    WT[42] = 5.51003900E+01;
    WT[43] = 5.61083600E+01;
    WT[44] = 5.71163300E+01;
    WT[45] = 8.91151300E+01;
    WT[46] = 6.91274800E+01;
    WT[47] = 7.01354500E+01;
    WT[48] = 7.11434200E+01;
    WT[49] = 8.41625400E+01;
    WT[50] = 9.91976000E+01;
    WT[51] = 1.00205570E+02;
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
    EPS[9] = 8.00000000E+01;
    EPS[10] = 4.98000000E+02;
    EPS[11] = 1.44000000E+02;
    EPS[12] = 2.44000000E+02;
    EPS[13] = 9.81000000E+01;
    EPS[14] = 4.98000000E+02;
    EPS[15] = 1.44000000E+02;
    EPS[16] = 1.44000000E+02;
    EPS[17] = 4.17000000E+02;
    EPS[18] = 1.41400000E+02;
    EPS[19] = 4.81800000E+02;
    EPS[20] = 2.47500000E+02;
    EPS[21] = 2.47500000E+02;
    EPS[22] = 4.36000000E+02;
    EPS[23] = 4.36000000E+02;
    EPS[24] = 4.81800000E+02;
    EPS[25] = 4.81800000E+02;
    EPS[26] = 2.65300000E+02;
    EPS[27] = 1.50000000E+02;
    EPS[28] = 2.65300000E+02;
    EPS[29] = 4.36000000E+02;
    EPS[30] = 3.07800000E+02;
    EPS[31] = 2.38400000E+02;
    EPS[32] = 4.70600000E+02;
    EPS[33] = 4.36000000E+02;
    EPS[34] = 4.70600000E+02;
    EPS[35] = 2.09000000E+02;
    EPS[36] = 3.24800000E+02;
    EPS[37] = 3.24800000E+02;
    EPS[38] = 3.16000000E+02;
    EPS[39] = 3.03400000E+02;
    EPS[40] = 4.81500000E+02;
    EPS[41] = 3.57000000E+02;
    EPS[42] = 3.55000000E+02;
    EPS[43] = 3.55000000E+02;
    EPS[44] = 3.52000000E+02;
    EPS[45] = 4.96000000E+02;
    EPS[46] = 3.96800000E+02;
    EPS[47] = 3.86200000E+02;
    EPS[48] = 4.40735000E+02;
    EPS[49] = 4.85857000E+02;
    EPS[50] = 4.59600000E+02;
    EPS[51] = 4.59600000E+02;
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
    SIG[9] = 2.75000000E+00;
    SIG[10] = 3.59000000E+00;
    SIG[11] = 3.80000000E+00;
    SIG[12] = 3.76300000E+00;
    SIG[13] = 3.65000000E+00;
    SIG[14] = 3.59000000E+00;
    SIG[15] = 3.80000000E+00;
    SIG[16] = 3.80000000E+00;
    SIG[17] = 3.69000000E+00;
    SIG[18] = 3.74600000E+00;
    SIG[19] = 3.62600000E+00;
    SIG[20] = 4.35000000E+00;
    SIG[21] = 4.35000000E+00;
    SIG[22] = 3.97000000E+00;
    SIG[23] = 3.97000000E+00;
    SIG[24] = 3.62600000E+00;
    SIG[25] = 3.62600000E+00;
    SIG[26] = 3.72100000E+00;
    SIG[27] = 2.50000000E+00;
    SIG[28] = 3.72100000E+00;
    SIG[29] = 3.97000000E+00;
    SIG[30] = 4.14000000E+00;
    SIG[31] = 3.49600000E+00;
    SIG[32] = 4.41000000E+00;
    SIG[33] = 3.97000000E+00;
    SIG[34] = 4.41000000E+00;
    SIG[35] = 4.10000000E+00;
    SIG[36] = 4.29000000E+00;
    SIG[37] = 4.29000000E+00;
    SIG[38] = 4.22000000E+00;
    SIG[39] = 4.81000000E+00;
    SIG[40] = 4.99700000E+00;
    SIG[41] = 4.72000000E+00;
    SIG[42] = 4.65000000E+00;
    SIG[43] = 4.65000000E+00;
    SIG[44] = 5.24000000E+00;
    SIG[45] = 5.20000000E+00;
    SIG[46] = 5.45800000E+00;
    SIG[47] = 5.48900000E+00;
    SIG[48] = 5.04100000E+00;
    SIG[49] = 5.32800000E+00;
    SIG[50] = 6.25300000E+00;
    SIG[51] = 6.25300000E+00;
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
    DIP[17] = 1.70000000E+00;
    DIP[18] = 0.00000000E+00;
    DIP[19] = 0.00000000E+00;
    DIP[20] = 0.00000000E+00;
    DIP[21] = 0.00000000E+00;
    DIP[22] = 0.00000000E+00;
    DIP[23] = 0.00000000E+00;
    DIP[24] = 0.00000000E+00;
    DIP[25] = 0.00000000E+00;
    DIP[26] = 0.00000000E+00;
    DIP[27] = 0.00000000E+00;
    DIP[28] = 0.00000000E+00;
    DIP[29] = 0.00000000E+00;
    DIP[30] = 0.00000000E+00;
    DIP[31] = 0.00000000E+00;
    DIP[32] = 0.00000000E+00;
    DIP[33] = 0.00000000E+00;
    DIP[34] = 0.00000000E+00;
    DIP[35] = 0.00000000E+00;
    DIP[36] = 0.00000000E+00;
    DIP[37] = 0.00000000E+00;
    DIP[38] = 0.00000000E+00;
    DIP[39] = 0.00000000E+00;
    DIP[40] = 1.70000000E+00;
    DIP[41] = 0.00000000E+00;
    DIP[42] = 0.00000000E+00;
    DIP[43] = 0.00000000E+00;
    DIP[44] = 0.00000000E+00;
    DIP[45] = 0.00000000E+00;
    DIP[46] = 0.00000000E+00;
    DIP[47] = 4.00000000E-01;
    DIP[48] = 0.00000000E+00;
    DIP[49] = 0.00000000E+00;
    DIP[50] = 0.00000000E+00;
    DIP[51] = 0.00000000E+00;
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
    POL[9] = 0.00000000E+00;
    POL[10] = 0.00000000E+00;
    POL[11] = 0.00000000E+00;
    POL[12] = 2.65000000E+00;
    POL[13] = 1.95000000E+00;
    POL[14] = 0.00000000E+00;
    POL[15] = 0.00000000E+00;
    POL[16] = 0.00000000E+00;
    POL[17] = 0.00000000E+00;
    POL[18] = 2.60000000E+00;
    POL[19] = 0.00000000E+00;
    POL[20] = 0.00000000E+00;
    POL[21] = 0.00000000E+00;
    POL[22] = 0.00000000E+00;
    POL[23] = 0.00000000E+00;
    POL[24] = 0.00000000E+00;
    POL[25] = 0.00000000E+00;
    POL[26] = 0.00000000E+00;
    POL[27] = 0.00000000E+00;
    POL[28] = 0.00000000E+00;
    POL[29] = 0.00000000E+00;
    POL[30] = 0.00000000E+00;
    POL[31] = 0.00000000E+00;
    POL[32] = 0.00000000E+00;
    POL[33] = 0.00000000E+00;
    POL[34] = 0.00000000E+00;
    POL[35] = 0.00000000E+00;
    POL[36] = 0.00000000E+00;
    POL[37] = 0.00000000E+00;
    POL[38] = 0.00000000E+00;
    POL[39] = 0.00000000E+00;
    POL[40] = 0.00000000E+00;
    POL[41] = 0.00000000E+00;
    POL[42] = 0.00000000E+00;
    POL[43] = 0.00000000E+00;
    POL[44] = 0.00000000E+00;
    POL[45] = 0.00000000E+00;
    POL[46] = 0.00000000E+00;
    POL[47] = 0.00000000E+00;
    POL[48] = 0.00000000E+00;
    POL[49] = 0.00000000E+00;
    POL[50] = 0.00000000E+00;
    POL[51] = 0.00000000E+00;
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
    ZROT[9] = 0.00000000E+00;
    ZROT[10] = 0.00000000E+00;
    ZROT[11] = 0.00000000E+00;
    ZROT[12] = 2.10000000E+00;
    ZROT[13] = 1.80000000E+00;
    ZROT[14] = 2.00000000E+00;
    ZROT[15] = 0.00000000E+00;
    ZROT[16] = 0.00000000E+00;
    ZROT[17] = 2.00000000E+00;
    ZROT[18] = 1.30000000E+01;
    ZROT[19] = 1.00000000E+00;
    ZROT[20] = 1.50000000E+00;
    ZROT[21] = 1.50000000E+00;
    ZROT[22] = 2.00000000E+00;
    ZROT[23] = 2.00000000E+00;
    ZROT[24] = 1.00000000E+00;
    ZROT[25] = 1.00000000E+00;
    ZROT[26] = 2.50000000E+00;
    ZROT[27] = 1.00000000E+00;
    ZROT[28] = 1.00000000E+00;
    ZROT[29] = 2.00000000E+00;
    ZROT[30] = 1.00000000E+00;
    ZROT[31] = 1.50000000E+00;
    ZROT[32] = 1.50000000E+00;
    ZROT[33] = 2.00000000E+00;
    ZROT[34] = 1.50000000E+00;
    ZROT[35] = 1.00000000E+00;
    ZROT[36] = 1.00000000E+00;
    ZROT[37] = 1.00000000E+00;
    ZROT[38] = 1.00000000E+00;
    ZROT[39] = 1.00000000E+00;
    ZROT[40] = 1.00000000E+00;
    ZROT[41] = 1.00000000E+00;
    ZROT[42] = 1.00000000E+00;
    ZROT[43] = 1.00000000E+00;
    ZROT[44] = 1.00000000E+00;
    ZROT[45] = 1.00000000E+00;
    ZROT[46] = 1.00000000E+00;
    ZROT[47] = 1.00000000E+00;
    ZROT[48] = 0.00000000E+00;
    ZROT[49] = 0.00000000E+00;
    ZROT[50] = 1.00000000E+00;
    ZROT[51] = 1.00000000E+00;
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
    NLIN[10] = 2;
    NLIN[11] = 1;
    NLIN[12] = 1;
    NLIN[13] = 1;
    NLIN[14] = 2;
    NLIN[15] = 1;
    NLIN[16] = 1;
    NLIN[17] = 2;
    NLIN[18] = 2;
    NLIN[19] = 2;
    NLIN[20] = 2;
    NLIN[21] = 2;
    NLIN[22] = 2;
    NLIN[23] = 2;
    NLIN[24] = 2;
    NLIN[25] = 2;
    NLIN[26] = 1;
    NLIN[27] = 2;
    NLIN[28] = 2;
    NLIN[29] = 2;
    NLIN[30] = 2;
    NLIN[31] = 2;
    NLIN[32] = 2;
    NLIN[33] = 2;
    NLIN[34] = 2;
    NLIN[35] = 2;
    NLIN[36] = 1;
    NLIN[37] = 1;
    NLIN[38] = 2;
    NLIN[39] = 2;
    NLIN[40] = 2;
    NLIN[41] = 2;
    NLIN[42] = 2;
    NLIN[43] = 2;
    NLIN[44] = 2;
    NLIN[45] = 2;
    NLIN[46] = 2;
    NLIN[47] = 2;
    NLIN[48] = 2;
    NLIN[49] = 2;
    NLIN[50] = 2;
    NLIN[51] = 2;
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
    COFETA[36] = -1.51956901E+01;
    COFETA[37] = 1.92606504E+00;
    COFETA[38] = -1.73487476E-01;
    COFETA[39] = 7.82572931E-03;
    COFETA[40] = -1.98501306E+01;
    COFETA[41] = 2.69480162E+00;
    COFETA[42] = -1.65880845E-01;
    COFETA[43] = 3.14504769E-03;
    COFETA[44] = -2.02663469E+01;
    COFETA[45] = 3.63241793E+00;
    COFETA[46] = -3.95581049E-01;
    COFETA[47] = 1.74725495E-02;
    COFETA[48] = -2.40014975E+01;
    COFETA[49] = 5.14359547E+00;
    COFETA[50] = -5.74269731E-01;
    COFETA[51] = 2.44937679E-02;
    COFETA[52] = -1.66188336E+01;
    COFETA[53] = 2.40307799E+00;
    COFETA[54] = -2.36167638E-01;
    COFETA[55] = 1.05714061E-02;
    COFETA[56] = -1.98330577E+01;
    COFETA[57] = 2.69480162E+00;
    COFETA[58] = -1.65880845E-01;
    COFETA[59] = 3.14504769E-03;
    COFETA[60] = -2.02663469E+01;
    COFETA[61] = 3.63241793E+00;
    COFETA[62] = -3.95581049E-01;
    COFETA[63] = 1.74725495E-02;
    COFETA[64] = -2.02316497E+01;
    COFETA[65] = 3.63241793E+00;
    COFETA[66] = -3.95581049E-01;
    COFETA[67] = 1.74725495E-02;
    COFETA[68] = -1.99945919E+01;
    COFETA[69] = 2.86923313E+00;
    COFETA[70] = -2.03325661E-01;
    COFETA[71] = 5.39056989E-03;
    COFETA[72] = -2.00094664E+01;
    COFETA[73] = 3.57220167E+00;
    COFETA[74] = -3.87936446E-01;
    COFETA[75] = 1.71483254E-02;
    COFETA[76] = -2.05644525E+01;
    COFETA[77] = 3.03946431E+00;
    COFETA[78] = -2.16994867E-01;
    COFETA[79] = 5.61394012E-03;
    COFETA[80] = -2.45432160E+01;
    COFETA[81] = 5.15878990E+00;
    COFETA[82] = -5.75274341E-01;
    COFETA[83] = 2.44975136E-02;
    COFETA[84] = -2.45602637E+01;
    COFETA[85] = 5.15878990E+00;
    COFETA[86] = -5.75274341E-01;
    COFETA[87] = 2.44975136E-02;
    COFETA[88] = -2.23395647E+01;
    COFETA[89] = 3.86433912E+00;
    COFETA[90] = -3.41553983E-01;
    COFETA[91] = 1.17083447E-02;
    COFETA[92] = -2.22942453E+01;
    COFETA[93] = 3.86433912E+00;
    COFETA[94] = -3.41553983E-01;
    COFETA[95] = 1.17083447E-02;
    COFETA[96] = -2.03725491E+01;
    COFETA[97] = 3.03946431E+00;
    COFETA[98] = -2.16994867E-01;
    COFETA[99] = 5.61394012E-03;
    COFETA[100] = -2.03619469E+01;
    COFETA[101] = 3.03946431E+00;
    COFETA[102] = -2.16994867E-01;
    COFETA[103] = 5.61394012E-03;
    COFETA[104] = -2.47697856E+01;
    COFETA[105] = 5.30039568E+00;
    COFETA[106] = -5.89273639E-01;
    COFETA[107] = 2.49261407E-02;
    COFETA[108] = -1.92183831E+01;
    COFETA[109] = 3.75164499E+00;
    COFETA[110] = -4.10390993E-01;
    COFETA[111] = 1.80861665E-02;
    COFETA[112] = -2.47507953E+01;
    COFETA[113] = 5.30039568E+00;
    COFETA[114] = -5.89273639E-01;
    COFETA[115] = 2.49261407E-02;
    COFETA[116] = -2.23277173E+01;
    COFETA[117] = 3.86433912E+00;
    COFETA[118] = -3.41553983E-01;
    COFETA[119] = 1.17083447E-02;
    COFETA[120] = -2.49727893E+01;
    COFETA[121] = 5.27067543E+00;
    COFETA[122] = -5.71909526E-01;
    COFETA[123] = 2.36230940E-02;
    COFETA[124] = -2.39690472E+01;
    COFETA[125] = 5.11436059E+00;
    COFETA[126] = -5.71999954E-01;
    COFETA[127] = 2.44581334E-02;
    COFETA[128] = -2.12500723E+01;
    COFETA[129] = 3.25726898E+00;
    COFETA[130] = -2.49519605E-01;
    COFETA[131] = 7.19215196E-03;
    COFETA[132] = -2.23277173E+01;
    COFETA[133] = 3.86433912E+00;
    COFETA[134] = -3.41553983E-01;
    COFETA[135] = 1.17083447E-02;
    COFETA[136] = -2.10981507E+01;
    COFETA[137] = 3.25726898E+00;
    COFETA[138] = -2.49519605E-01;
    COFETA[139] = 7.19215196E-03;
    COFETA[140] = -2.31769852E+01;
    COFETA[141] = 4.80350223E+00;
    COFETA[142] = -5.38341336E-01;
    COFETA[143] = 2.32747213E-02;
    COFETA[144] = -2.50327383E+01;
    COFETA[145] = 5.20184077E+00;
    COFETA[146] = -5.57265947E-01;
    COFETA[147] = 2.27565676E-02;
    COFETA[148] = -2.50199983E+01;
    COFETA[149] = 5.20184077E+00;
    COFETA[150] = -5.57265947E-01;
    COFETA[151] = 2.27565676E-02;
    COFETA[152] = -2.50402232E+01;
    COFETA[153] = 5.25451220E+00;
    COFETA[154] = -5.67228955E-01;
    COFETA[155] = 2.33156489E-02;
    COFETA[156] = -2.52462994E+01;
    COFETA[157] = 5.27749097E+00;
    COFETA[158] = -5.74219215E-01;
    COFETA[159] = 2.37811608E-02;
    COFETA[160] = -2.00330714E+01;
    COFETA[161] = 2.73318358E+00;
    COFETA[162] = -1.75653565E-01;
    COFETA[163] = 3.77126610E-03;
    COFETA[164] = -2.46654710E+01;
    COFETA[165] = 4.94595777E+00;
    COFETA[166] = -5.12278955E-01;
    COFETA[167] = 2.03286378E-02;
    COFETA[168] = -2.46566816E+01;
    COFETA[169] = 4.96413364E+00;
    COFETA[170] = -5.15375011E-01;
    COFETA[171] = 2.04926972E-02;
    COFETA[172] = -2.46476176E+01;
    COFETA[173] = 4.96413364E+00;
    COFETA[174] = -5.15375011E-01;
    COFETA[175] = 2.04926972E-02;
    COFETA[176] = -2.49295265E+01;
    COFETA[177] = 4.99407558E+00;
    COFETA[178] = -5.20389056E-01;
    COFETA[179] = 2.07557283E-02;
    COFETA[180] = -2.01225145E+01;
    COFETA[181] = 2.73742901E+00;
    COFETA[182] = -1.72182607E-01;
    COFETA[183] = 3.44858406E-03;
    COFETA[184] = -2.39731890E+01;
    COFETA[185] = 4.48124637E+00;
    COFETA[186] = -4.36934973E-01;
    COFETA[187] = 1.64568564E-02;
    COFETA[188] = -2.42040008E+01;
    COFETA[189] = 4.60377392E+00;
    COFETA[190] = -4.56799058E-01;
    COFETA[191] = 1.74771833E-02;
    COFETA[192] = -2.23900786E+01;
    COFETA[193] = 3.78460980E+00;
    COFETA[194] = -3.29370574E-01;
    COFETA[195] = 1.11069352E-02;
    COFETA[196] = -2.06651397E+01;
    COFETA[197] = 2.95298213E+00;
    COFETA[198] = -2.04141647E-01;
    COFETA[199] = 4.99194219E-03;
    COFETA[200] = -2.19891717E+01;
    COFETA[201] = 3.46341268E+00;
    COFETA[202] = -2.80516687E-01;
    COFETA[203] = 8.70427548E-03;
    COFETA[204] = -2.19841167E+01;
    COFETA[205] = 3.46341268E+00;
    COFETA[206] = -2.80516687E-01;
    COFETA[207] = 8.70427548E-03;
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
    COFLAM[36] = 2.76121550E+01;
    COFLAM[37] = -9.31549593E+00;
    COFLAM[38] = 1.44163486E+00;
    COFLAM[39] = -6.77060940E-02;
    COFLAM[40] = 5.62029187E+00;
    COFLAM[41] = -1.91574800E+00;
    COFLAM[42] = 5.90225152E-01;
    COFLAM[43] = -3.57616080E-02;
    COFLAM[44] = 1.32940078E+01;
    COFLAM[45] = -3.90945048E+00;
    COFLAM[46] = 7.39705471E-01;
    COFLAM[47] = -3.74149296E-02;
    COFLAM[48] = -1.15552013E+01;
    COFLAM[49] = 5.97444378E+00;
    COFLAM[50] = -5.83493959E-01;
    COFLAM[51] = 2.11390997E-02;
    COFLAM[52] = 1.15794127E+01;
    COFLAM[53] = -3.02088602E+00;
    COFLAM[54] = 5.82091962E-01;
    COFLAM[55] = -2.93406692E-02;
    COFLAM[56] = 4.84793117E+00;
    COFLAM[57] = -2.13603179E+00;
    COFLAM[58] = 6.99833906E-01;
    COFLAM[59] = -4.38222985E-02;
    COFLAM[60] = 1.83932434E+01;
    COFLAM[61] = -6.25754971E+00;
    COFLAM[62] = 1.09378595E+00;
    COFLAM[63] = -5.49435895E-02;
    COFLAM[64] = 1.36305859E+01;
    COFLAM[65] = -4.45086721E+00;
    COFLAM[66] = 8.75862692E-01;
    COFLAM[67] = -4.60275277E-02;
    COFLAM[68] = -6.14587484E+00;
    COFLAM[69] = 2.47428533E+00;
    COFLAM[70] = 6.44004931E-02;
    COFLAM[71] = -1.45368612E-02;
    COFLAM[72] = 1.29622480E+01;
    COFLAM[73] = -4.85747192E+00;
    COFLAM[74] = 1.02918185E+00;
    COFLAM[75] = -5.69931976E-02;
    COFLAM[76] = -3.48612719E+00;
    COFLAM[77] = 1.33821415E+00;
    COFLAM[78] = 2.29051402E-01;
    COFLAM[79] = -2.22522544E-02;
    COFLAM[80] = -1.44773293E+01;
    COFLAM[81] = 6.20799727E+00;
    COFLAM[82] = -4.66686188E-01;
    COFLAM[83] = 1.03037078E-02;
    COFLAM[84] = -1.32867451E+01;
    COFLAM[85] = 5.82553048E+00;
    COFLAM[86] = -4.38733487E-01;
    COFLAM[87] = 1.02209771E-02;
    COFLAM[88] = -9.55248736E+00;
    COFLAM[89] = 4.54181017E+00;
    COFLAM[90] = -3.09443018E-01;
    COFLAM[91] = 5.98150058E-03;
    COFLAM[92] = -1.42688121E+01;
    COFLAM[93] = 6.13139162E+00;
    COFLAM[94] = -4.81580164E-01;
    COFLAM[95] = 1.17158883E-02;
    COFLAM[96] = 5.28056570E+00;
    COFLAM[97] = -1.92758693E+00;
    COFLAM[98] = 6.29141169E-01;
    COFLAM[99] = -3.87203309E-02;
    COFLAM[100] = -5.85803496E+00;
    COFLAM[101] = 2.80606054E+00;
    COFLAM[102] = -2.49415114E-02;
    COFLAM[103] = -8.88253132E-03;
    COFLAM[104] = -9.20687365E+00;
    COFLAM[105] = 5.13028609E+00;
    COFLAM[106] = -4.67868863E-01;
    COFLAM[107] = 1.64674383E-02;
    COFLAM[108] = -5.31228079E-01;
    COFLAM[109] = 2.21998671E+00;
    COFLAM[110] = -1.07502048E-01;
    COFLAM[111] = 1.06754974E-03;
    COFLAM[112] = -1.36861349E+01;
    COFLAM[113] = 6.35261898E+00;
    COFLAM[114] = -5.53497228E-01;
    COFLAM[115] = 1.70958139E-02;
    COFLAM[116] = -6.27418855E+00;
    COFLAM[117] = 2.90468350E+00;
    COFLAM[118] = -4.35101878E-02;
    COFLAM[119] = -7.77942889E-03;
    COFLAM[120] = -1.54410770E+01;
    COFLAM[121] = 6.67114766E+00;
    COFLAM[122] = -5.37137624E-01;
    COFLAM[123] = 1.38051704E-02;
    COFLAM[124] = -1.34447168E+01;
    COFLAM[125] = 6.12380203E+00;
    COFLAM[126] = -4.86657425E-01;
    COFLAM[127] = 1.24614518E-02;
    COFLAM[128] = -1.43280999E+01;
    COFLAM[129] = 5.86486794E+00;
    COFLAM[130] = -4.04615405E-01;
    COFLAM[131] = 6.95849698E-03;
    COFLAM[132] = -7.51756469E+00;
    COFLAM[133] = 3.30311461E+00;
    COFLAM[134] = -8.47654747E-02;
    COFLAM[135] = -6.42328466E-03;
    COFLAM[136] = -7.35375659E+00;
    COFLAM[137] = 3.05592385E+00;
    COFLAM[138] = -2.73121212E-02;
    COFLAM[139] = -1.00073545E-02;
    COFLAM[140] = -2.41661928E+00;
    COFLAM[141] = 2.57272912E+00;
    COFLAM[142] = -1.49379556E-01;
    COFLAM[143] = 2.85758744E-03;
    COFLAM[144] = -9.95261562E+00;
    COFLAM[145] = 5.06627945E+00;
    COFLAM[146] = -4.15761512E-01;
    COFLAM[147] = 1.21197123E-02;
    COFLAM[148] = -1.32966554E+01;
    COFLAM[149] = 5.92585034E+00;
    COFLAM[150] = -4.64901365E-01;
    COFLAM[151] = 1.16662523E-02;
    COFLAM[152] = -2.08328673E+01;
    COFLAM[153] = 9.07593204E+00;
    COFLAM[154] = -8.93990863E-01;
    COFLAM[155] = 3.11142957E-02;
    COFLAM[156] = -1.59315280E+01;
    COFLAM[157] = 7.00975811E+00;
    COFLAM[158] = -6.12923034E-01;
    COFLAM[159] = 1.85439169E-02;
    COFLAM[160] = -9.04441654E+00;
    COFLAM[161] = 3.66855955E+00;
    COFLAM[162] = -1.04994889E-01;
    COFLAM[163] = -6.68408235E-03;
    COFLAM[164] = -2.26611414E+01;
    COFLAM[165] = 9.78565333E+00;
    COFLAM[166] = -9.94033497E-01;
    COFLAM[167] = 3.57950722E-02;
    COFLAM[168] = -1.90915971E+01;
    COFLAM[169] = 8.15678005E+00;
    COFLAM[170] = -7.46604137E-01;
    COFLAM[171] = 2.35743405E-02;
    COFLAM[172] = -1.96439129E+01;
    COFLAM[173] = 8.31169569E+00;
    COFLAM[174] = -7.56268608E-01;
    COFLAM[175] = 2.35727121E-02;
    COFLAM[176] = -1.83524587E+01;
    COFLAM[177] = 7.74571154E+00;
    COFLAM[178] = -6.83646893E-01;
    COFLAM[179] = 2.04759926E-02;
    COFLAM[180] = -1.04744531E+01;
    COFLAM[181] = 4.24091183E+00;
    COFLAM[182] = -1.78599110E-01;
    COFLAM[183] = -3.51607728E-03;
    COFLAM[184] = -2.13451516E+01;
    COFLAM[185] = 8.84739285E+00;
    COFLAM[186] = -8.23421444E-01;
    COFLAM[187] = 2.63782925E-02;
    COFLAM[188] = -1.98965346E+01;
    COFLAM[189] = 8.27532362E+00;
    COFLAM[190] = -7.46184920E-01;
    COFLAM[191] = 2.29240275E-02;
    COFLAM[192] = -1.77209721E+01;
    COFLAM[193] = 7.36216062E+00;
    COFLAM[194] = -6.11089543E-01;
    COFLAM[195] = 1.64273679E-02;
    COFLAM[196] = -1.80139132E+01;
    COFLAM[197] = 7.35713318E+00;
    COFLAM[198] = -6.00090479E-01;
    COFLAM[199] = 1.55359162E-02;
    COFLAM[200] = -1.72838492E+01;
    COFLAM[201] = 6.97737723E+00;
    COFLAM[202] = -5.47365626E-01;
    COFLAM[203] = 1.30795303E-02;
    COFLAM[204] = -1.79582416E+01;
    COFLAM[205] = 7.27686902E+00;
    COFLAM[206] = -5.88898453E-01;
    COFLAM[207] = 1.49980279E-02;
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
    COFD[36] = -1.40076852E+01;
    COFD[37] = 3.07549274E+00;
    COFD[38] = -1.88889344E-01;
    COFD[39] = 8.37152866E-03;
    COFD[40] = -2.04750581E+01;
    COFD[41] = 5.23112374E+00;
    COFD[42] = -4.54967682E-01;
    COFD[43] = 1.93570423E-02;
    COFD[44] = -1.59404882E+01;
    COFD[45] = 3.66853818E+00;
    COFD[46] = -2.64346221E-01;
    COFD[47] = 1.15784613E-02;
    COFD[48] = -1.81432461E+01;
    COFD[49] = 4.37565431E+00;
    COFD[50] = -3.53906025E-01;
    COFD[51] = 1.53760786E-02;
    COFD[52] = -1.50031687E+01;
    COFD[53] = 3.26223357E+00;
    COFD[54] = -2.12746642E-01;
    COFD[55] = 9.38912883E-03;
    COFD[56] = -2.04833713E+01;
    COFD[57] = 5.23112374E+00;
    COFD[58] = -4.54967682E-01;
    COFD[59] = 1.93570423E-02;
    COFD[60] = -1.59404882E+01;
    COFD[61] = 3.66853818E+00;
    COFD[62] = -2.64346221E-01;
    COFD[63] = 1.15784613E-02;
    COFD[64] = -1.59633387E+01;
    COFD[65] = 3.66853818E+00;
    COFD[66] = -2.64346221E-01;
    COFD[67] = 1.15784613E-02;
    COFD[68] = -2.02268902E+01;
    COFD[69] = 5.13632093E+00;
    COFD[70] = -4.44839124E-01;
    COFD[71] = 1.90058354E-02;
    COFD[72] = -1.59327297E+01;
    COFD[73] = 3.65620899E+00;
    COFD[74] = -2.62933804E-01;
    COFD[75] = 1.15253223E-02;
    COFD[76] = -2.03844252E+01;
    COFD[77] = 5.18856872E+00;
    COFD[78] = -4.50001829E-01;
    COFD[79] = 1.91636142E-02;
    COFD[80] = -1.82673770E+01;
    COFD[81] = 4.39538102E+00;
    COFD[82] = -3.56367230E-01;
    COFD[83] = 1.54788461E-02;
    COFD[84] = -1.82590824E+01;
    COFD[85] = 4.39538102E+00;
    COFD[86] = -3.56367230E-01;
    COFD[87] = 1.54788461E-02;
    COFD[88] = -2.02646611E+01;
    COFD[89] = 5.10426133E+00;
    COFD[90] = -4.41256919E-01;
    COFD[91] = 1.88737290E-02;
    COFD[92] = -2.02822946E+01;
    COFD[93] = 5.10426133E+00;
    COFD[94] = -4.41256919E-01;
    COFD[95] = 1.88737290E-02;
    COFD[96] = -2.04649069E+01;
    COFD[97] = 5.18856872E+00;
    COFD[98] = -4.50001829E-01;
    COFD[99] = 1.91636142E-02;
    COFD[100] = -2.04688382E+01;
    COFD[101] = 5.18856872E+00;
    COFD[102] = -4.50001829E-01;
    COFD[103] = 1.91636142E-02;
    COFD[104] = -1.83039618E+01;
    COFD[105] = 4.47952077E+00;
    COFD[106] = -3.66569471E-01;
    COFD[107] = 1.58916129E-02;
    COFD[108] = -1.59884305E+01;
    COFD[109] = 3.72220402E+00;
    COFD[110] = -2.71150591E-01;
    COFD[111] = 1.18665265E-02;
    COFD[112] = -1.83137139E+01;
    COFD[113] = 4.47952077E+00;
    COFD[114] = -3.66569471E-01;
    COFD[115] = 1.58916129E-02;
    COFD[116] = -2.02693653E+01;
    COFD[117] = 5.10426133E+00;
    COFD[118] = -4.41256919E-01;
    COFD[119] = 1.88737290E-02;
    COFD[120] = -1.90859283E+01;
    COFD[121] = 4.68079396E+00;
    COFD[122] = -3.91231550E-01;
    COFD[123] = 1.69021170E-02;
    COFD[124] = -1.78815889E+01;
    COFD[125] = 4.34347890E+00;
    COFD[126] = -3.49890003E-01;
    COFD[127] = 1.52083459E-02;
    COFD[128] = -2.05802296E+01;
    COFD[129] = 5.16117916E+00;
    COFD[130] = -4.46897404E-01;
    COFD[131] = 1.90470443E-02;
    COFD[132] = -2.02693653E+01;
    COFD[133] = 5.10426133E+00;
    COFD[134] = -4.41256919E-01;
    COFD[135] = 1.88737290E-02;
    COFD[136] = -2.06331583E+01;
    COFD[137] = 5.16117916E+00;
    COFD[138] = -4.46897404E-01;
    COFD[139] = 1.90470443E-02;
    COFD[140] = -1.76895296E+01;
    COFD[141] = 4.19171952E+00;
    COFD[142] = -3.31354810E-01;
    COFD[143] = 1.44520623E-02;
    COFD[144] = -1.92731067E+01;
    COFD[145] = 4.73660584E+00;
    COFD[146] = -3.97704978E-01;
    COFD[147] = 1.71514887E-02;
    COFD[148] = -1.92783884E+01;
    COFD[149] = 4.73660584E+00;
    COFD[150] = -3.97704978E-01;
    COFD[151] = 1.71514887E-02;
    COFD[152] = -1.91796663E+01;
    COFD[153] = 4.70714822E+00;
    COFD[154] = -3.94261134E-01;
    COFD[155] = 1.70175169E-02;
    COFD[156] = -1.92062897E+01;
    COFD[157] = 4.66318669E+00;
    COFD[158] = -3.89108667E-01;
    COFD[159] = 1.68165377E-02;
    COFD[160] = -2.09943481E+01;
    COFD[161] = 5.22468467E+00;
    COFD[162] = -4.54220128E-01;
    COFD[163] = 1.93281042E-02;
    COFD[164] = -1.97484166E+01;
    COFD[165] = 4.84231878E+00;
    COFD[166] = -4.10101001E-01;
    COFD[167] = 1.76356687E-02;
    COFD[168] = -1.97196489E+01;
    COFD[169] = 4.83750266E+00;
    COFD[170] = -4.09581452E-01;
    COFD[171] = 1.76174739E-02;
    COFD[172] = -1.97226856E+01;
    COFD[173] = 4.83750266E+00;
    COFD[174] = -4.09581452E-01;
    COFD[175] = 1.76174739E-02;
    COFD[176] = -1.98374654E+01;
    COFD[177] = 4.82871870E+00;
    COFD[178] = -4.08567726E-01;
    COFD[179] = 1.75785896E-02;
    COFD[180] = -2.10643735E+01;
    COFD[181] = 5.22604478E+00;
    COFD[182] = -4.54378127E-01;
    COFD[183] = 1.93342248E-02;
    COFD[184] = -2.03706752E+01;
    COFD[185] = 4.98803076E+00;
    COFD[186] = -4.27580621E-01;
    COFD[187] = 1.83363274E-02;
    COFD[188] = -2.02840235E+01;
    COFD[189] = 4.95484018E+00;
    COFD[190] = -4.23654881E-01;
    COFD[191] = 1.81813866E-02;
    COFD[192] = -2.06548278E+01;
    COFD[193] = 5.11678107E+00;
    COFD[194] = -4.42706538E-01;
    COFD[195] = 1.89296424E-02;
    COFD[196] = -2.10086887E+01;
    COFD[197] = 5.19953529E+00;
    COFD[198] = -4.51287802E-01;
    COFD[199] = 1.92140123E-02;
    COFD[200] = -2.10674485E+01;
    COFD[201] = 5.15027524E+00;
    COFD[202] = -4.46126111E-01;
    COFD[203] = 1.90401391E-02;
    COFD[204] = -2.10685573E+01;
    COFD[205] = 5.15027524E+00;
    COFD[206] = -4.46126111E-01;
    COFD[207] = 1.90401391E-02;
    COFD[208] = -1.40756935E+01;
    COFD[209] = 3.07549274E+00;
    COFD[210] = -1.88889344E-01;
    COFD[211] = 8.37152866E-03;
    COFD[212] = -1.32093628E+01;
    COFD[213] = 2.90778936E+00;
    COFD[214] = -1.67388544E-01;
    COFD[215] = 7.45220609E-03;
    COFD[216] = -1.09595712E+01;
    COFD[217] = 2.30836460E+00;
    COFD[218] = -8.76339315E-02;
    COFD[219] = 3.90878445E-03;
    COFD[220] = -1.34230272E+01;
    COFD[221] = 3.48624238E+00;
    COFD[222] = -2.41554467E-01;
    COFD[223] = 1.06263545E-02;
    COFD[224] = -1.32244035E+01;
    COFD[225] = 2.90778936E+00;
    COFD[226] = -1.67388544E-01;
    COFD[227] = 7.45220609E-03;
    COFD[228] = -1.94093572E+01;
    COFD[229] = 5.16013126E+00;
    COFD[230] = -4.46824543E-01;
    COFD[231] = 1.90464887E-02;
    COFD[232] = -1.43139231E+01;
    COFD[233] = 3.17651319E+00;
    COFD[234] = -2.02028974E-01;
    COFD[235] = 8.94232502E-03;
    COFD[236] = -1.43190389E+01;
    COFD[237] = 3.17651319E+00;
    COFD[238] = -2.02028974E-01;
    COFD[239] = 8.94232502E-03;
    COFD[240] = -1.43238998E+01;
    COFD[241] = 3.17651319E+00;
    COFD[242] = -2.02028974E-01;
    COFD[243] = 8.94232502E-03;
    COFD[244] = -1.31551788E+01;
    COFD[245] = 2.90778936E+00;
    COFD[246] = -1.67388544E-01;
    COFD[247] = 7.45220609E-03;
    COFD[248] = -1.94313116E+01;
    COFD[249] = 5.02567894E+00;
    COFD[250] = -4.32045169E-01;
    COFD[251] = 1.85132214E-02;
    COFD[252] = -1.50584249E+01;
    COFD[253] = 3.47945612E+00;
    COFD[254] = -2.40703722E-01;
    COFD[255] = 1.05907441E-02;
    COFD[256] = -1.70534856E+01;
    COFD[257] = 4.14240922E+00;
    COFD[258] = -3.25239774E-01;
    COFD[259] = 1.41980687E-02;
    COFD[260] = -1.40999008E+01;
    COFD[261] = 3.08120012E+00;
    COFD[262] = -1.89629903E-01;
    COFD[263] = 8.40361952E-03;
    COFD[264] = -1.94373127E+01;
    COFD[265] = 5.02567894E+00;
    COFD[266] = -4.32045169E-01;
    COFD[267] = 1.85132214E-02;
    COFD[268] = -1.50584249E+01;
    COFD[269] = 3.47945612E+00;
    COFD[270] = -2.40703722E-01;
    COFD[271] = 1.05907441E-02;
    COFD[272] = -1.50766130E+01;
    COFD[273] = 3.47945612E+00;
    COFD[274] = -2.40703722E-01;
    COFD[275] = 1.05907441E-02;
    COFD[276] = -1.88179418E+01;
    COFD[277] = 4.79683898E+00;
    COFD[278] = -4.04829719E-01;
    COFD[279] = 1.74325475E-02;
    COFD[280] = -1.50270339E+01;
    COFD[281] = 3.46140064E+00;
    COFD[282] = -2.38440092E-01;
    COFD[283] = 1.04960087E-02;
    COFD[284] = -1.93364585E+01;
    COFD[285] = 4.98286777E+00;
    COFD[286] = -4.26970814E-01;
    COFD[287] = 1.83122917E-02;
    COFD[288] = -1.72112971E+01;
    COFD[289] = 4.15807461E+00;
    COFD[290] = -3.27178539E-01;
    COFD[291] = 1.42784349E-02;
    COFD[292] = -1.72053106E+01;
    COFD[293] = 4.15807461E+00;
    COFD[294] = -3.27178539E-01;
    COFD[295] = 1.42784349E-02;
    COFD[296] = -1.90883268E+01;
    COFD[297] = 4.84384483E+00;
    COFD[298] = -4.10265575E-01;
    COFD[299] = 1.76414287E-02;
    COFD[300] = -1.91004157E+01;
    COFD[301] = 4.84384483E+00;
    COFD[302] = -4.10265575E-01;
    COFD[303] = 1.76414287E-02;
    COFD[304] = -1.93925667E+01;
    COFD[305] = 4.98286777E+00;
    COFD[306] = -4.26970814E-01;
    COFD[307] = 1.83122917E-02;
    COFD[308] = -1.93952366E+01;
    COFD[309] = 4.98286777E+00;
    COFD[310] = -4.26970814E-01;
    COFD[311] = 1.83122917E-02;
    COFD[312] = -1.72286007E+01;
    COFD[313] = 4.24084025E+00;
    COFD[314] = -3.37428619E-01;
    COFD[315] = 1.47032793E-02;
    COFD[316] = -1.49500357E+01;
    COFD[317] = 3.52327209E+00;
    COFD[318] = -2.46286208E-01;
    COFD[319] = 1.08285963E-02;
    COFD[320] = -1.72357436E+01;
    COFD[321] = 4.24084025E+00;
    COFD[322] = -3.37428619E-01;
    COFD[323] = 1.47032793E-02;
    COFD[324] = -1.90915649E+01;
    COFD[325] = 4.84384483E+00;
    COFD[326] = -4.10265575E-01;
    COFD[327] = 1.76414287E-02;
    COFD[328] = -1.79361160E+01;
    COFD[329] = 4.42139452E+00;
    COFD[330] = -3.59567329E-01;
    COFD[331] = 1.56103969E-02;
    COFD[332] = -1.68343393E+01;
    COFD[333] = 4.11954900E+00;
    COFD[334] = -3.22470391E-01;
    COFD[335] = 1.40859564E-02;
    COFD[336] = -1.95314689E+01;
    COFD[337] = 4.95249173E+00;
    COFD[338] = -4.23376552E-01;
    COFD[339] = 1.81703714E-02;
    COFD[340] = -1.90915649E+01;
    COFD[341] = 4.84384483E+00;
    COFD[342] = -4.10265575E-01;
    COFD[343] = 1.76414287E-02;
    COFD[344] = -1.95670324E+01;
    COFD[345] = 4.95249173E+00;
    COFD[346] = -4.23376552E-01;
    COFD[347] = 1.81703714E-02;
    COFD[348] = -1.66128343E+01;
    COFD[349] = 3.95035840E+00;
    COFD[350] = -3.00959418E-01;
    COFD[351] = 1.31692593E-02;
    COFD[352] = -1.81463104E+01;
    COFD[353] = 4.48398491E+00;
    COFD[354] = -3.67097129E-01;
    COFD[355] = 1.59123634E-02;
    COFD[356] = -1.81499793E+01;
    COFD[357] = 4.48398491E+00;
    COFD[358] = -3.67097129E-01;
    COFD[359] = 1.59123634E-02;
    COFD[360] = -1.80480958E+01;
    COFD[361] = 4.45434023E+00;
    COFD[362] = -3.63584633E-01;
    COFD[363] = 1.57739270E-02;
    COFD[364] = -1.80724788E+01;
    COFD[365] = 4.40247898E+00;
    COFD[366] = -3.57238362E-01;
    COFD[367] = 1.55145651E-02;
    COFD[368] = -1.98296243E+01;
    COFD[369] = 4.98207523E+00;
    COFD[370] = -4.26877291E-01;
    COFD[371] = 1.83086094E-02;
    COFD[372] = -1.86652603E+01;
    COFD[373] = 4.61260432E+00;
    COFD[374] = -3.82854484E-01;
    COFD[375] = 1.65575163E-02;
    COFD[376] = -1.86234701E+01;
    COFD[377] = 4.60336076E+00;
    COFD[378] = -3.81691643E-01;
    COFD[379] = 1.65085234E-02;
    COFD[380] = -1.86254955E+01;
    COFD[381] = 4.60336076E+00;
    COFD[382] = -3.81691643E-01;
    COFD[383] = 1.65085234E-02;
    COFD[384] = -1.87433618E+01;
    COFD[385] = 4.58956960E+00;
    COFD[386] = -3.79964215E-01;
    COFD[387] = 1.64361138E-02;
    COFD[388] = -2.00070284E+01;
    COFD[389] = 5.02095434E+00;
    COFD[390] = -4.31496874E-01;
    COFD[391] = 1.84920392E-02;
    COFD[392] = -1.92404583E+01;
    COFD[393] = 4.73921581E+00;
    COFD[394] = -3.98017274E-01;
    COFD[395] = 1.71639614E-02;
    COFD[396] = -1.91633071E+01;
    COFD[397] = 4.70966098E+00;
    COFD[398] = -3.94551217E-01;
    COFD[399] = 1.70286289E-02;
    COFD[400] = -1.94778445E+01;
    COFD[401] = 4.85518471E+00;
    COFD[402] = -4.11551624E-01;
    COFD[403] = 1.76895651E-02;
    COFD[404] = -1.99562868E+01;
    COFD[405] = 4.99367362E+00;
    COFD[406] = -4.28249956E-01;
    COFD[407] = 1.83628509E-02;
    COFD[408] = -1.99785176E+01;
    COFD[409] = 4.92184026E+00;
    COFD[410] = -4.19745472E-01;
    COFD[411] = 1.80268154E-02;
    COFD[412] = -1.99792167E+01;
    COFD[413] = 4.92184026E+00;
    COFD[414] = -4.19745472E-01;
    COFD[415] = 1.80268154E-02;
    COFD[416] = -1.16906297E+01;
    COFD[417] = 2.47469981E+00;
    COFD[418] = -1.10436257E-01;
    COFD[419] = 4.95273813E-03;
    COFD[420] = -1.09595712E+01;
    COFD[421] = 2.30836460E+00;
    COFD[422] = -8.76339315E-02;
    COFD[423] = 3.90878445E-03;
    COFD[424] = -1.03270606E+01;
    COFD[425] = 2.19285409E+00;
    COFD[426] = -7.54492786E-02;
    COFD[427] = 3.51398213E-03;
    COFD[428] = -1.14366381E+01;
    COFD[429] = 2.78323501E+00;
    COFD[430] = -1.51214064E-01;
    COFD[431] = 6.75150012E-03;
    COFD[432] = -1.09628982E+01;
    COFD[433] = 2.30836460E+00;
    COFD[434] = -8.76339315E-02;
    COFD[435] = 3.90878445E-03;
    COFD[436] = -1.71982995E+01;
    COFD[437] = 4.63881404E+00;
    COFD[438] = -3.86139633E-01;
    COFD[439] = 1.66955081E-02;
    COFD[440] = -1.18988955E+01;
    COFD[441] = 2.57507000E+00;
    COFD[442] = -1.24033737E-01;
    COFD[443] = 5.56694959E-03;
    COFD[444] = -1.18998012E+01;
    COFD[445] = 2.57507000E+00;
    COFD[446] = -1.24033737E-01;
    COFD[447] = 5.56694959E-03;
    COFD[448] = -1.19006548E+01;
    COFD[449] = 2.57507000E+00;
    COFD[450] = -1.24033737E-01;
    COFD[451] = 5.56694959E-03;
    COFD[452] = -1.09469245E+01;
    COFD[453] = 2.30836460E+00;
    COFD[454] = -8.76339315E-02;
    COFD[455] = 3.90878445E-03;
    COFD[456] = -1.60517370E+01;
    COFD[457] = 4.11188603E+00;
    COFD[458] = -3.21540884E-01;
    COFD[459] = 1.40482564E-02;
    COFD[460] = -1.25098960E+01;
    COFD[461] = 2.77873601E+00;
    COFD[462] = -1.50637360E-01;
    COFD[463] = 6.72684281E-03;
    COFD[464] = -1.37794315E+01;
    COFD[465] = 3.23973858E+00;
    COFD[466] = -2.09989036E-01;
    COFD[467] = 9.27667906E-03;
    COFD[468] = -1.17159737E+01;
    COFD[469] = 2.48123210E+00;
    COFD[470] = -1.11322604E-01;
    COFD[471] = 4.99282389E-03;
    COFD[472] = -1.60528285E+01;
    COFD[473] = 4.11188603E+00;
    COFD[474] = -3.21540884E-01;
    COFD[475] = 1.40482564E-02;
    COFD[476] = -1.25098960E+01;
    COFD[477] = 2.77873601E+00;
    COFD[478] = -1.50637360E-01;
    COFD[479] = 6.72684281E-03;
    COFD[480] = -1.25141260E+01;
    COFD[481] = 2.77873601E+00;
    COFD[482] = -1.50637360E-01;
    COFD[483] = 6.72684281E-03;
    COFD[484] = -1.58456300E+01;
    COFD[485] = 4.02074783E+00;
    COFD[486] = -3.10018522E-01;
    COFD[487] = 1.35599552E-02;
    COFD[488] = -1.24693568E+01;
    COFD[489] = 2.76686648E+00;
    COFD[490] = -1.49120141E-01;
    COFD[491] = 6.66220432E-03;
    COFD[492] = -1.59537247E+01;
    COFD[493] = 4.07051484E+00;
    COFD[494] = -3.16303109E-01;
    COFD[495] = 1.38259377E-02;
    COFD[496] = -1.39658996E+01;
    COFD[497] = 3.24966086E+00;
    COFD[498] = -2.11199992E-01;
    COFD[499] = 9.32580661E-03;
    COFD[500] = -1.39648112E+01;
    COFD[501] = 3.24966086E+00;
    COFD[502] = -2.11199992E-01;
    COFD[503] = 9.32580661E-03;
    COFD[504] = -1.57034851E+01;
    COFD[505] = 3.93614244E+00;
    COFD[506] = -2.99111497E-01;
    COFD[507] = 1.30888229E-02;
    COFD[508] = -1.57054717E+01;
    COFD[509] = 3.93614244E+00;
    COFD[510] = -2.99111497E-01;
    COFD[511] = 1.30888229E-02;
    COFD[512] = -1.59632479E+01;
    COFD[513] = 4.07051484E+00;
    COFD[514] = -3.16303109E-01;
    COFD[515] = 1.38259377E-02;
    COFD[516] = -1.59636793E+01;
    COFD[517] = 4.07051484E+00;
    COFD[518] = -3.16303109E-01;
    COFD[519] = 1.38259377E-02;
    COFD[520] = -1.39315266E+01;
    COFD[521] = 3.30394764E+00;
    COFD[522] = -2.17920112E-01;
    COFD[523] = 9.60284243E-03;
    COFD[524] = -1.22004324E+01;
    COFD[525] = 2.80725489E+00;
    COFD[526] = -1.54291406E-01;
    COFD[527] = 6.88290911E-03;
    COFD[528] = -1.39328674E+01;
    COFD[529] = 3.30394764E+00;
    COFD[530] = -2.17920112E-01;
    COFD[531] = 9.60284243E-03;
    COFD[532] = -1.57040212E+01;
    COFD[533] = 3.93614244E+00;
    COFD[534] = -2.99111497E-01;
    COFD[535] = 1.30888229E-02;
    COFD[536] = -1.45715797E+01;
    COFD[537] = 3.49477850E+00;
    COFD[538] = -2.42635772E-01;
    COFD[539] = 1.06721490E-02;
    COFD[540] = -1.36336373E+01;
    COFD[541] = 3.22088176E+00;
    COFD[542] = -2.07623790E-01;
    COFD[543] = 9.17771542E-03;
    COFD[544] = -1.61116686E+01;
    COFD[545] = 4.04227735E+00;
    COFD[546] = -3.12745253E-01;
    COFD[547] = 1.36756977E-02;
    COFD[548] = -1.57040212E+01;
    COFD[549] = 3.93614244E+00;
    COFD[550] = -2.99111497E-01;
    COFD[551] = 1.30888229E-02;
    COFD[552] = -1.61173105E+01;
    COFD[553] = 4.04227735E+00;
    COFD[554] = -3.12745253E-01;
    COFD[555] = 1.36756977E-02;
    COFD[556] = -1.34824532E+01;
    COFD[557] = 3.09379603E+00;
    COFD[558] = -1.91268635E-01;
    COFD[559] = 8.47480224E-03;
    COFD[560] = -1.47719516E+01;
    COFD[561] = 3.55444478E+00;
    COFD[562] = -2.50272707E-01;
    COFD[563] = 1.09990787E-02;
    COFD[564] = -1.47725694E+01;
    COFD[565] = 3.55444478E+00;
    COFD[566] = -2.50272707E-01;
    COFD[567] = 1.09990787E-02;
    COFD[568] = -1.46719197E+01;
    COFD[569] = 3.52400594E+00;
    COFD[570] = -2.46379985E-01;
    COFD[571] = 1.08326032E-02;
    COFD[572] = -1.47137939E+01;
    COFD[573] = 3.48023191E+00;
    COFD[574] = -2.40800798E-01;
    COFD[575] = 1.05947990E-02;
    COFD[576] = -1.64819183E+01;
    COFD[577] = 4.11726215E+00;
    COFD[578] = -3.22193015E-01;
    COFD[579] = 1.40747074E-02;
    COFD[580] = -1.51448279E+01;
    COFD[581] = 3.64565939E+00;
    COFD[582] = -2.61726871E-01;
    COFD[583] = 1.14799244E-02;
    COFD[584] = -1.51159870E+01;
    COFD[585] = 3.64206330E+00;
    COFD[586] = -2.61313444E-01;
    COFD[587] = 1.14642754E-02;
    COFD[588] = -1.51163041E+01;
    COFD[589] = 3.64206330E+00;
    COFD[590] = -2.61313444E-01;
    COFD[591] = 1.14642754E-02;
    COFD[592] = -1.52503668E+01;
    COFD[593] = 3.63657318E+00;
    COFD[594] = -2.60678457E-01;
    COFD[595] = 1.14400550E-02;
    COFD[596] = -1.65048875E+01;
    COFD[597] = 4.10792536E+00;
    COFD[598] = -3.21060656E-01;
    COFD[599] = 1.40287900E-02;
    COFD[600] = -1.56919143E+01;
    COFD[601] = 3.77842689E+00;
    COFD[602] = -2.78523399E-01;
    COFD[603] = 1.21896111E-02;
    COFD[604] = -1.55781966E+01;
    COFD[605] = 3.73153794E+00;
    COFD[606] = -2.72372598E-01;
    COFD[607] = 1.19199668E-02;
    COFD[608] = -1.60461372E+01;
    COFD[609] = 3.95298868E+00;
    COFD[610] = -3.01302078E-01;
    COFD[611] = 1.31842095E-02;
    COFD[612] = -1.64639359E+01;
    COFD[613] = 4.08142484E+00;
    COFD[614] = -3.17696496E-01;
    COFD[615] = 1.38856294E-02;
    COFD[616] = -1.64898528E+01;
    COFD[617] = 4.01175649E+00;
    COFD[618] = -3.08860971E-01;
    COFD[619] = 1.35100076E-02;
    COFD[620] = -1.64899530E+01;
    COFD[621] = 4.01175649E+00;
    COFD[622] = -3.08860971E-01;
    COFD[623] = 1.35100076E-02;
    COFD[624] = -1.42894441E+01;
    COFD[625] = 3.67490723E+00;
    COFD[626] = -2.65114792E-01;
    COFD[627] = 1.16092671E-02;
    COFD[628] = -1.34230272E+01;
    COFD[629] = 3.48624238E+00;
    COFD[630] = -2.41554467E-01;
    COFD[631] = 1.06263545E-02;
    COFD[632] = -1.14366381E+01;
    COFD[633] = 2.78323501E+00;
    COFD[634] = -1.51214064E-01;
    COFD[635] = 6.75150012E-03;
    COFD[636] = -1.47968712E+01;
    COFD[637] = 4.23027636E+00;
    COFD[638] = -3.36139991E-01;
    COFD[639] = 1.46507621E-02;
    COFD[640] = -1.34247866E+01;
    COFD[641] = 3.48624238E+00;
    COFD[642] = -2.41554467E-01;
    COFD[643] = 1.06263545E-02;
    COFD[644] = -1.95739570E+01;
    COFD[645] = 5.61113230E+00;
    COFD[646] = -4.90190187E-01;
    COFD[647] = 2.03260675E-02;
    COFD[648] = -1.46550083E+01;
    COFD[649] = 3.83606243E+00;
    COFD[650] = -2.86076532E-01;
    COFD[651] = 1.25205829E-02;
    COFD[652] = -1.46554748E+01;
    COFD[653] = 3.83606243E+00;
    COFD[654] = -2.86076532E-01;
    COFD[655] = 1.25205829E-02;
    COFD[656] = -1.46559141E+01;
    COFD[657] = 3.83606243E+00;
    COFD[658] = -2.86076532E-01;
    COFD[659] = 1.25205829E-02;
    COFD[660] = -1.34162893E+01;
    COFD[661] = 3.48624238E+00;
    COFD[662] = -2.41554467E-01;
    COFD[663] = 1.06263545E-02;
    COFD[664] = -1.97544450E+01;
    COFD[665] = 5.56931926E+00;
    COFD[666] = -4.89105511E-01;
    COFD[667] = 2.04493129E-02;
    COFD[668] = -1.57972369E+01;
    COFD[669] = 4.22225052E+00;
    COFD[670] = -3.35156428E-01;
    COFD[671] = 1.46104855E-02;
    COFD[672] = -1.76147026E+01;
    COFD[673] = 4.86049500E+00;
    COFD[674] = -4.12200578E-01;
    COFD[675] = 1.77160971E-02;
    COFD[676] = -1.43151174E+01;
    COFD[677] = 3.68038508E+00;
    COFD[678] = -2.65779346E-01;
    COFD[679] = 1.16360771E-02;
    COFD[680] = -1.97550088E+01;
    COFD[681] = 5.56931926E+00;
    COFD[682] = -4.89105511E-01;
    COFD[683] = 2.04493129E-02;
    COFD[684] = -1.57972369E+01;
    COFD[685] = 4.22225052E+00;
    COFD[686] = -3.35156428E-01;
    COFD[687] = 1.46104855E-02;
    COFD[688] = -1.57994893E+01;
    COFD[689] = 4.22225052E+00;
    COFD[690] = -3.35156428E-01;
    COFD[691] = 1.46104855E-02;
    COFD[692] = -1.92718582E+01;
    COFD[693] = 5.41172124E+00;
    COFD[694] = -4.73213887E-01;
    COFD[695] = 1.99405473E-02;
    COFD[696] = -1.57199037E+01;
    COFD[697] = 4.19936335E+00;
    COFD[698] = -3.32311009E-01;
    COFD[699] = 1.44921003E-02;
    COFD[700] = -1.96866103E+01;
    COFD[701] = 5.54637286E+00;
    COFD[702] = -4.87070324E-01;
    COFD[703] = 2.03983467E-02;
    COFD[704] = -1.78637178E+01;
    COFD[705] = 4.88268692E+00;
    COFD[706] = -4.14917638E-01;
    COFD[707] = 1.78274298E-02;
    COFD[708] = -1.78631557E+01;
    COFD[709] = 4.88268692E+00;
    COFD[710] = -4.14917638E-01;
    COFD[711] = 1.78274298E-02;
    COFD[712] = -1.94688688E+01;
    COFD[713] = 5.43830787E+00;
    COFD[714] = -4.75472880E-01;
    COFD[715] = 1.99909996E-02;
    COFD[716] = -1.94698843E+01;
    COFD[717] = 5.43830787E+00;
    COFD[718] = -4.75472880E-01;
    COFD[719] = 1.99909996E-02;
    COFD[720] = -1.96914944E+01;
    COFD[721] = 5.54637286E+00;
    COFD[722] = -4.87070324E-01;
    COFD[723] = 2.03983467E-02;
    COFD[724] = -1.96917146E+01;
    COFD[725] = 5.54637286E+00;
    COFD[726] = -4.87070324E-01;
    COFD[727] = 2.03983467E-02;
    COFD[728] = -1.79310765E+01;
    COFD[729] = 4.98037650E+00;
    COFD[730] = -4.26676911E-01;
    COFD[731] = 1.83007231E-02;
    COFD[732] = -1.54460820E+01;
    COFD[733] = 4.26819983E+00;
    COFD[734] = -3.40766379E-01;
    COFD[735] = 1.48393361E-02;
    COFD[736] = -1.79317714E+01;
    COFD[737] = 4.98037650E+00;
    COFD[738] = -4.26676911E-01;
    COFD[739] = 1.83007231E-02;
    COFD[740] = -1.94691430E+01;
    COFD[741] = 5.43830787E+00;
    COFD[742] = -4.75472880E-01;
    COFD[743] = 1.99909996E-02;
    COFD[744] = -1.85748546E+01;
    COFD[745] = 5.14789919E+00;
    COFD[746] = -4.45930850E-01;
    COFD[747] = 1.90363341E-02;
    COFD[748] = -1.74407963E+01;
    COFD[749] = 4.83580036E+00;
    COFD[750] = -4.09383573E-01;
    COFD[751] = 1.76098175E-02;
    COFD[752] = -1.98806372E+01;
    COFD[753] = 5.52555673E+00;
    COFD[754] = -4.84999851E-01;
    COFD[755] = 2.03334931E-02;
    COFD[756] = -1.94691430E+01;
    COFD[757] = 5.43830787E+00;
    COFD[758] = -4.75472880E-01;
    COFD[759] = 1.99909996E-02;
    COFD[760] = -1.98835119E+01;
    COFD[761] = 5.52555673E+00;
    COFD[762] = -4.84999851E-01;
    COFD[763] = 2.03334931E-02;
    COFD[764] = -1.72291395E+01;
    COFD[765] = 4.69060745E+00;
    COFD[766] = -3.92369888E-01;
    COFD[767] = 1.69459661E-02;
    COFD[768] = -1.87644697E+01;
    COFD[769] = 5.19146813E+00;
    COFD[770] = -4.50340408E-01;
    COFD[771] = 1.91768178E-02;
    COFD[772] = -1.87647862E+01;
    COFD[773] = 5.19146813E+00;
    COFD[774] = -4.50340408E-01;
    COFD[775] = 1.91768178E-02;
    COFD[776] = -1.86493112E+01;
    COFD[777] = 5.16040659E+00;
    COFD[778] = -4.46843492E-01;
    COFD[779] = 1.90466181E-02;
    COFD[780] = -1.87481780E+01;
    COFD[781] = 5.13858656E+00;
    COFD[782] = -4.45075387E-01;
    COFD[783] = 1.90137309E-02;
    COFD[784] = -2.01262921E+01;
    COFD[785] = 5.54581286E+00;
    COFD[786] = -4.87014004E-01;
    COFD[787] = 2.03965482E-02;
    COFD[788] = -1.92784178E+01;
    COFD[789] = 5.32291505E+00;
    COFD[790] = -4.65883522E-01;
    COFD[791] = 1.97916109E-02;
    COFD[792] = -1.92360228E+01;
    COFD[793] = 5.31542554E+00;
    COFD[794] = -4.65003780E-01;
    COFD[795] = 1.97570185E-02;
    COFD[796] = -1.92361841E+01;
    COFD[797] = 5.31542554E+00;
    COFD[798] = -4.65003780E-01;
    COFD[799] = 1.97570185E-02;
    COFD[800] = -1.93693740E+01;
    COFD[801] = 5.30286598E+00;
    COFD[802] = -4.63495567E-01;
    COFD[803] = 1.96962203E-02;
    COFD[804] = -2.02592914E+01;
    COFD[805] = 5.56701235E+00;
    COFD[806] = -4.88925090E-01;
    COFD[807] = 2.04461948E-02;
    COFD[808] = -1.97252269E+01;
    COFD[809] = 5.38884098E+00;
    COFD[810] = -4.71627912E-01;
    COFD[811] = 1.99273178E-02;
    COFD[812] = -1.96804200E+01;
    COFD[813] = 5.37526595E+00;
    COFD[814] = -4.70621144E-01;
    COFD[815] = 1.99141073E-02;
    COFD[816] = -1.98424714E+01;
    COFD[817] = 5.45215174E+00;
    COFD[818] = -4.77051991E-01;
    COFD[819] = 2.00510347E-02;
    COFD[820] = -2.02451923E+01;
    COFD[821] = 5.55377454E+00;
    COFD[822] = -4.87810074E-01;
    COFD[823] = 2.04217376E-02;
    COFD[824] = -2.03113704E+01;
    COFD[825] = 5.50136606E+00;
    COFD[826] = -4.82461887E-01;
    COFD[827] = 2.02471523E-02;
    COFD[828] = -2.03114210E+01;
    COFD[829] = 5.50136606E+00;
    COFD[830] = -4.82461887E-01;
    COFD[831] = 2.02471523E-02;
    COFD[832] = -1.40949196E+01;
    COFD[833] = 3.07549274E+00;
    COFD[834] = -1.88889344E-01;
    COFD[835] = 8.37152866E-03;
    COFD[836] = -1.32244035E+01;
    COFD[837] = 2.90778936E+00;
    COFD[838] = -1.67388544E-01;
    COFD[839] = 7.45220609E-03;
    COFD[840] = -1.09628982E+01;
    COFD[841] = 2.30836460E+00;
    COFD[842] = -8.76339315E-02;
    COFD[843] = 3.90878445E-03;
    COFD[844] = -1.34247866E+01;
    COFD[845] = 3.48624238E+00;
    COFD[846] = -2.41554467E-01;
    COFD[847] = 1.06263545E-02;
    COFD[848] = -1.32399106E+01;
    COFD[849] = 2.90778936E+00;
    COFD[850] = -1.67388544E-01;
    COFD[851] = 7.45220609E-03;
    COFD[852] = -1.94253036E+01;
    COFD[853] = 5.16013126E+00;
    COFD[854] = -4.46824543E-01;
    COFD[855] = 1.90464887E-02;
    COFD[856] = -1.43340796E+01;
    COFD[857] = 3.17651319E+00;
    COFD[858] = -2.02028974E-01;
    COFD[859] = 8.94232502E-03;
    COFD[860] = -1.43394069E+01;
    COFD[861] = 3.17651319E+00;
    COFD[862] = -2.02028974E-01;
    COFD[863] = 8.94232502E-03;
    COFD[864] = -1.43444709E+01;
    COFD[865] = 3.17651319E+00;
    COFD[866] = -2.02028974E-01;
    COFD[867] = 8.94232502E-03;
    COFD[868] = -1.31686537E+01;
    COFD[869] = 2.90778936E+00;
    COFD[870] = -1.67388544E-01;
    COFD[871] = 7.45220609E-03;
    COFD[872] = -1.94507876E+01;
    COFD[873] = 5.02567894E+00;
    COFD[874] = -4.32045169E-01;
    COFD[875] = 1.85132214E-02;
    COFD[876] = -1.50724636E+01;
    COFD[877] = 3.47945612E+00;
    COFD[878] = -2.40703722E-01;
    COFD[879] = 1.05907441E-02;
    COFD[880] = -1.70757047E+01;
    COFD[881] = 4.14240922E+00;
    COFD[882] = -3.25239774E-01;
    COFD[883] = 1.41980687E-02;
    COFD[884] = -1.41191261E+01;
    COFD[885] = 3.08120012E+00;
    COFD[886] = -1.89629903E-01;
    COFD[887] = 8.40361952E-03;
    COFD[888] = -1.94570287E+01;
    COFD[889] = 5.02567894E+00;
    COFD[890] = -4.32045169E-01;
    COFD[891] = 1.85132214E-02;
    COFD[892] = -1.50724636E+01;
    COFD[893] = 3.47945612E+00;
    COFD[894] = -2.40703722E-01;
    COFD[895] = 1.05907441E-02;
    COFD[896] = -1.50911794E+01;
    COFD[897] = 3.47945612E+00;
    COFD[898] = -2.40703722E-01;
    COFD[899] = 1.05907441E-02;
    COFD[900] = -1.88378874E+01;
    COFD[901] = 4.79683898E+00;
    COFD[902] = -4.04829719E-01;
    COFD[903] = 1.74325475E-02;
    COFD[904] = -1.50420953E+01;
    COFD[905] = 3.46140064E+00;
    COFD[906] = -2.38440092E-01;
    COFD[907] = 1.04960087E-02;
    COFD[908] = -1.93566243E+01;
    COFD[909] = 4.98286777E+00;
    COFD[910] = -4.26970814E-01;
    COFD[911] = 1.83122917E-02;
    COFD[912] = -1.72310232E+01;
    COFD[913] = 4.15807461E+00;
    COFD[914] = -3.27178539E-01;
    COFD[915] = 1.42784349E-02;
    COFD[916] = -1.72247972E+01;
    COFD[917] = 4.15807461E+00;
    COFD[918] = -3.27178539E-01;
    COFD[919] = 1.42784349E-02;
    COFD[920] = -1.91102652E+01;
    COFD[921] = 4.84384483E+00;
    COFD[922] = -4.10265575E-01;
    COFD[923] = 1.76414287E-02;
    COFD[924] = -1.91229033E+01;
    COFD[925] = 4.84384483E+00;
    COFD[926] = -4.10265575E-01;
    COFD[927] = 1.76414287E-02;
    COFD[928] = -1.94151822E+01;
    COFD[929] = 4.98286777E+00;
    COFD[930] = -4.26970814E-01;
    COFD[931] = 1.83122917E-02;
    COFD[932] = -1.94179760E+01;
    COFD[933] = 4.98286777E+00;
    COFD[934] = -4.26970814E-01;
    COFD[935] = 1.83122917E-02;
    COFD[936] = -1.72473011E+01;
    COFD[937] = 4.24084025E+00;
    COFD[938] = -3.37428619E-01;
    COFD[939] = 1.47032793E-02;
    COFD[940] = -1.49718233E+01;
    COFD[941] = 3.52327209E+00;
    COFD[942] = -2.46286208E-01;
    COFD[943] = 1.08285963E-02;
    COFD[944] = -1.72547182E+01;
    COFD[945] = 4.24084025E+00;
    COFD[946] = -3.37428619E-01;
    COFD[947] = 1.47032793E-02;
    COFD[948] = -1.91136491E+01;
    COFD[949] = 4.84384483E+00;
    COFD[950] = -4.10265575E-01;
    COFD[951] = 1.76414287E-02;
    COFD[952] = -1.79580609E+01;
    COFD[953] = 4.42139452E+00;
    COFD[954] = -3.59567329E-01;
    COFD[955] = 1.56103969E-02;
    COFD[956] = -1.68535757E+01;
    COFD[957] = 4.11954900E+00;
    COFD[958] = -3.22470391E-01;
    COFD[959] = 1.40859564E-02;
    COFD[960] = -1.95538303E+01;
    COFD[961] = 4.95249173E+00;
    COFD[962] = -4.23376552E-01;
    COFD[963] = 1.81703714E-02;
    COFD[964] = -1.91136491E+01;
    COFD[965] = 4.84384483E+00;
    COFD[966] = -4.10265575E-01;
    COFD[967] = 1.76414287E-02;
    COFD[968] = -1.95910824E+01;
    COFD[969] = 4.95249173E+00;
    COFD[970] = -4.23376552E-01;
    COFD[971] = 1.81703714E-02;
    COFD[972] = -1.66341434E+01;
    COFD[973] = 3.95035840E+00;
    COFD[974] = -3.00959418E-01;
    COFD[975] = 1.31692593E-02;
    COFD[976] = -1.81677871E+01;
    COFD[977] = 4.48398491E+00;
    COFD[978] = -3.67097129E-01;
    COFD[979] = 1.59123634E-02;
    COFD[980] = -1.81716176E+01;
    COFD[981] = 4.48398491E+00;
    COFD[982] = -3.67097129E-01;
    COFD[983] = 1.59123634E-02;
    COFD[984] = -1.80698901E+01;
    COFD[985] = 4.45434023E+00;
    COFD[986] = -3.63584633E-01;
    COFD[987] = 1.57739270E-02;
    COFD[988] = -1.80945693E+01;
    COFD[989] = 4.40247898E+00;
    COFD[990] = -3.57238362E-01;
    COFD[991] = 1.55145651E-02;
    COFD[992] = -1.98546695E+01;
    COFD[993] = 4.98207523E+00;
    COFD[994] = -4.26877291E-01;
    COFD[995] = 1.83086094E-02;
    COFD[996] = -1.86886689E+01;
    COFD[997] = 4.61260432E+00;
    COFD[998] = -3.82854484E-01;
    COFD[999] = 1.65575163E-02;
    COFD[1000] = -1.86469792E+01;
    COFD[1001] = 4.60336076E+00;
    COFD[1002] = -3.81691643E-01;
    COFD[1003] = 1.65085234E-02;
    COFD[1004] = -1.86491023E+01;
    COFD[1005] = 4.60336076E+00;
    COFD[1006] = -3.81691643E-01;
    COFD[1007] = 1.65085234E-02;
    COFD[1008] = -1.87670637E+01;
    COFD[1009] = 4.58956960E+00;
    COFD[1010] = -3.79964215E-01;
    COFD[1011] = 1.64361138E-02;
    COFD[1012] = -2.00328044E+01;
    COFD[1013] = 5.02095434E+00;
    COFD[1014] = -4.31496874E-01;
    COFD[1015] = 1.84920392E-02;
    COFD[1016] = -1.92651204E+01;
    COFD[1017] = 4.73921581E+00;
    COFD[1018] = -3.98017274E-01;
    COFD[1019] = 1.71639614E-02;
    COFD[1020] = -1.91880377E+01;
    COFD[1021] = 4.70966098E+00;
    COFD[1022] = -3.94551217E-01;
    COFD[1023] = 1.70286289E-02;
    COFD[1024] = -1.95026421E+01;
    COFD[1025] = 4.85518471E+00;
    COFD[1026] = -4.11551624E-01;
    COFD[1027] = 1.76895651E-02;
    COFD[1028] = -1.99818280E+01;
    COFD[1029] = 4.99367362E+00;
    COFD[1030] = -4.28249956E-01;
    COFD[1031] = 1.83628509E-02;
    COFD[1032] = -2.00047095E+01;
    COFD[1033] = 4.92184026E+00;
    COFD[1034] = -4.19745472E-01;
    COFD[1035] = 1.80268154E-02;
    COFD[1036] = -2.00054461E+01;
    COFD[1037] = 4.92184026E+00;
    COFD[1038] = -4.19745472E-01;
    COFD[1039] = 1.80268154E-02;
    COFD[1040] = -2.10643259E+01;
    COFD[1041] = 5.53614847E+00;
    COFD[1042] = -4.86046736E-01;
    COFD[1043] = 2.03659188E-02;
    COFD[1044] = -1.94093572E+01;
    COFD[1045] = 5.16013126E+00;
    COFD[1046] = -4.46824543E-01;
    COFD[1047] = 1.90464887E-02;
    COFD[1048] = -1.71982995E+01;
    COFD[1049] = 4.63881404E+00;
    COFD[1050] = -3.86139633E-01;
    COFD[1051] = 1.66955081E-02;
    COFD[1052] = -1.95739570E+01;
    COFD[1053] = 5.61113230E+00;
    COFD[1054] = -4.90190187E-01;
    COFD[1055] = 2.03260675E-02;
    COFD[1056] = -1.94253036E+01;
    COFD[1057] = 5.16013126E+00;
    COFD[1058] = -4.46824543E-01;
    COFD[1059] = 1.90464887E-02;
    COFD[1060] = -1.19157919E+01;
    COFD[1061] = 9.28955130E-01;
    COFD[1062] = 2.42107090E-01;
    COFD[1063] = -1.59823963E-02;
    COFD[1064] = -2.12652533E+01;
    COFD[1065] = 5.59961818E+00;
    COFD[1066] = -4.91624858E-01;
    COFD[1067] = 2.05035550E-02;
    COFD[1068] = -2.06463744E+01;
    COFD[1069] = 5.41688482E+00;
    COFD[1070] = -4.73387188E-01;
    COFD[1071] = 1.99280175E-02;
    COFD[1072] = -2.06516336E+01;
    COFD[1073] = 5.41688482E+00;
    COFD[1074] = -4.73387188E-01;
    COFD[1075] = 1.99280175E-02;
    COFD[1076] = -1.93521390E+01;
    COFD[1077] = 5.16013126E+00;
    COFD[1078] = -4.46824543E-01;
    COFD[1079] = 1.90464887E-02;
    COFD[1080] = -1.77498543E+01;
    COFD[1081] = 3.57475686E+00;
    COFD[1082] = -1.56396297E-01;
    COFD[1083] = 3.12157721E-03;
    COFD[1084] = -2.12639214E+01;
    COFD[1085] = 5.61184117E+00;
    COFD[1086] = -4.90532156E-01;
    COFD[1087] = 2.03507922E-02;
    COFD[1088] = -2.07653719E+01;
    COFD[1089] = 5.01092022E+00;
    COFD[1090] = -3.77985635E-01;
    COFD[1091] = 1.40968645E-02;
    COFD[1092] = -2.11388331E+01;
    COFD[1093] = 5.55529675E+00;
    COFD[1094] = -4.87942518E-01;
    COFD[1095] = 2.04249054E-02;
    COFD[1096] = -1.77563250E+01;
    COFD[1097] = 3.57475686E+00;
    COFD[1098] = -1.56396297E-01;
    COFD[1099] = 3.12157721E-03;
    COFD[1100] = -2.12639214E+01;
    COFD[1101] = 5.61184117E+00;
    COFD[1102] = -4.90532156E-01;
    COFD[1103] = 2.03507922E-02;
    COFD[1104] = -2.12831323E+01;
    COFD[1105] = 5.61184117E+00;
    COFD[1106] = -4.90532156E-01;
    COFD[1107] = 2.03507922E-02;
    COFD[1108] = -1.65295288E+01;
    COFD[1109] = 2.97569206E+00;
    COFD[1110] = -6.75652842E-02;
    COFD[1111] = -1.08648422E-03;
    COFD[1112] = -2.14087397E+01;
    COFD[1113] = 5.57282008E+00;
    COFD[1114] = -4.76690890E-01;
    COFD[1115] = 1.94000719E-02;
    COFD[1116] = -1.80253664E+01;
    COFD[1117] = 3.69199168E+00;
    COFD[1118] = -1.74005516E-01;
    COFD[1119] = 3.97694372E-03;
    COFD[1120] = -2.13148887E+01;
    COFD[1121] = 5.27210469E+00;
    COFD[1122] = -4.21419216E-01;
    COFD[1123] = 1.63567178E-02;
    COFD[1124] = -2.13084334E+01;
    COFD[1125] = 5.27210469E+00;
    COFD[1126] = -4.21419216E-01;
    COFD[1127] = 1.63567178E-02;
    COFD[1128] = -1.87383952E+01;
    COFD[1129] = 3.96926341E+00;
    COFD[1130] = -2.16412264E-01;
    COFD[1131] = 6.06012078E-03;
    COFD[1132] = -1.87515645E+01;
    COFD[1133] = 3.96926341E+00;
    COFD[1134] = -2.16412264E-01;
    COFD[1135] = 6.06012078E-03;
    COFD[1136] = -1.80862867E+01;
    COFD[1137] = 3.69199168E+00;
    COFD[1138] = -1.74005516E-01;
    COFD[1139] = 3.97694372E-03;
    COFD[1140] = -1.80892005E+01;
    COFD[1141] = 3.69199168E+00;
    COFD[1142] = -1.74005516E-01;
    COFD[1143] = 3.97694372E-03;
    COFD[1144] = -2.09565916E+01;
    COFD[1145] = 5.18380539E+00;
    COFD[1146] = -4.06234719E-01;
    COFD[1147] = 1.55515345E-02;
    COFD[1148] = -2.10440675E+01;
    COFD[1149] = 5.59806282E+00;
    COFD[1150] = -4.87109535E-01;
    COFD[1151] = 2.01370226E-02;
    COFD[1152] = -2.09642705E+01;
    COFD[1153] = 5.18380539E+00;
    COFD[1154] = -4.06234719E-01;
    COFD[1155] = 1.55515345E-02;
    COFD[1156] = -1.87419199E+01;
    COFD[1157] = 3.96926341E+00;
    COFD[1158] = -2.16412264E-01;
    COFD[1159] = 6.06012078E-03;
    COFD[1160] = -2.06310304E+01;
    COFD[1161] = 4.89289496E+00;
    COFD[1162] = -3.59346263E-01;
    COFD[1163] = 1.31570901E-02;
    COFD[1164] = -2.11309197E+01;
    COFD[1165] = 5.32644193E+00;
    COFD[1166] = -4.30581064E-01;
    COFD[1167] = 1.68379725E-02;
    COFD[1168] = -1.84538368E+01;
    COFD[1169] = 3.75912079E+00;
    COFD[1170] = -1.84235105E-01;
    COFD[1171] = 4.47800951E-03;
    COFD[1172] = -1.87419199E+01;
    COFD[1173] = 3.96926341E+00;
    COFD[1174] = -2.16412264E-01;
    COFD[1175] = 6.06012078E-03;
    COFD[1176] = -1.84927291E+01;
    COFD[1177] = 3.75912079E+00;
    COFD[1178] = -1.84235105E-01;
    COFD[1179] = 4.47800951E-03;
    COFD[1180] = -2.15787073E+01;
    COFD[1181] = 5.46737673E+00;
    COFD[1182] = -4.55696085E-01;
    COFD[1183] = 1.81982625E-02;
    COFD[1184] = -2.04357586E+01;
    COFD[1185] = 4.77398686E+00;
    COFD[1186] = -3.40522956E-01;
    COFD[1187] = 1.22072846E-02;
    COFD[1188] = -2.04397451E+01;
    COFD[1189] = 4.77398686E+00;
    COFD[1190] = -3.40522956E-01;
    COFD[1191] = 1.22072846E-02;
    COFD[1192] = -2.05372411E+01;
    COFD[1193] = 4.83379373E+00;
    COFD[1194] = -3.50008083E-01;
    COFD[1195] = 1.26863426E-02;
    COFD[1196] = -2.08879167E+01;
    COFD[1197] = 4.92602269E+00;
    COFD[1198] = -3.64572914E-01;
    COFD[1199] = 1.34203681E-02;
    COFD[1200] = -1.73636900E+01;
    COFD[1201] = 3.17377130E+00;
    COFD[1202] = -1.00394383E-01;
    COFD[1203] = 5.69083899E-04;
    COFD[1204] = -2.02184916E+01;
    COFD[1205] = 4.57152878E+00;
    COFD[1206] = -3.08371263E-01;
    COFD[1207] = 1.05838559E-02;
    COFD[1208] = -2.02265558E+01;
    COFD[1209] = 4.58441724E+00;
    COFD[1210] = -3.10392854E-01;
    COFD[1211] = 1.06849990E-02;
    COFD[1212] = -2.02287739E+01;
    COFD[1213] = 4.58441724E+00;
    COFD[1214] = -3.10392854E-01;
    COFD[1215] = 1.06849990E-02;
    COFD[1216] = -2.04186424E+01;
    COFD[1217] = 4.60117690E+00;
    COFD[1218] = -3.13067257E-01;
    COFD[1219] = 1.08202310E-02;
    COFD[1220] = -1.83939699E+01;
    COFD[1221] = 3.59019527E+00;
    COFD[1222] = -1.58702132E-01;
    COFD[1223] = 3.23316765E-03;
    COFD[1224] = -1.98682752E+01;
    COFD[1225] = 4.28648872E+00;
    COFD[1226] = -2.64358750E-01;
    COFD[1227] = 8.40263071E-03;
    COFD[1228] = -1.98762802E+01;
    COFD[1229] = 4.29984430E+00;
    COFD[1230] = -2.67672378E-01;
    COFD[1231] = 8.61066133E-03;
    COFD[1232] = -1.90375666E+01;
    COFD[1233] = 3.93604965E+00;
    COFD[1234] = -2.11360409E-01;
    COFD[1235] = 5.81247394E-03;
    COFD[1236] = -1.85767826E+01;
    COFD[1237] = 3.66420353E+00;
    COFD[1238] = -1.69810177E-01;
    COFD[1239] = 3.77247849E-03;
    COFD[1240] = -1.91326792E+01;
    COFD[1241] = 3.82263611E+00;
    COFD[1242] = -1.93983472E-01;
    COFD[1243] = 4.95789388E-03;
    COFD[1244] = -1.91334529E+01;
    COFD[1245] = 3.82263611E+00;
    COFD[1246] = -1.93983472E-01;
    COFD[1247] = 4.95789388E-03;
    COFD[1248] = -1.52414485E+01;
    COFD[1249] = 3.35922578E+00;
    COFD[1250] = -2.25181399E-01;
    COFD[1251] = 9.92132878E-03;
    COFD[1252] = -1.43139231E+01;
    COFD[1253] = 3.17651319E+00;
    COFD[1254] = -2.02028974E-01;
    COFD[1255] = 8.94232502E-03;
    COFD[1256] = -1.18988955E+01;
    COFD[1257] = 2.57507000E+00;
    COFD[1258] = -1.24033737E-01;
    COFD[1259] = 5.56694959E-03;
    COFD[1260] = -1.46550083E+01;
    COFD[1261] = 3.83606243E+00;
    COFD[1262] = -2.86076532E-01;
    COFD[1263] = 1.25205829E-02;
    COFD[1264] = -1.43340796E+01;
    COFD[1265] = 3.17651319E+00;
    COFD[1266] = -2.02028974E-01;
    COFD[1267] = 8.94232502E-03;
    COFD[1268] = -2.12652533E+01;
    COFD[1269] = 5.59961818E+00;
    COFD[1270] = -4.91624858E-01;
    COFD[1271] = 2.05035550E-02;
    COFD[1272] = -1.55511344E+01;
    COFD[1273] = 3.48070094E+00;
    COFD[1274] = -2.40859499E-01;
    COFD[1275] = 1.05972514E-02;
    COFD[1276] = -1.55588279E+01;
    COFD[1277] = 3.48070094E+00;
    COFD[1278] = -2.40859499E-01;
    COFD[1279] = 1.05972514E-02;
    COFD[1280] = -1.55661750E+01;
    COFD[1281] = 3.48070094E+00;
    COFD[1282] = -2.40859499E-01;
    COFD[1283] = 1.05972514E-02;
    COFD[1284] = -1.42429085E+01;
    COFD[1285] = 3.17651319E+00;
    COFD[1286] = -2.02028974E-01;
    COFD[1287] = 8.94232502E-03;
    COFD[1288] = -2.08204449E+01;
    COFD[1289] = 5.35267674E+00;
    COFD[1290] = -4.69010505E-01;
    COFD[1291] = 1.98979152E-02;
    COFD[1292] = -1.63254691E+01;
    COFD[1293] = 3.82388595E+00;
    COFD[1294] = -2.84480724E-01;
    COFD[1295] = 1.24506311E-02;
    COFD[1296] = -1.84688406E+01;
    COFD[1297] = 4.49330851E+00;
    COFD[1298] = -3.68208715E-01;
    COFD[1299] = 1.59565402E-02;
    COFD[1300] = -1.52721107E+01;
    COFD[1301] = 3.36790500E+00;
    COFD[1302] = -2.26321740E-01;
    COFD[1303] = 9.97135055E-03;
    COFD[1304] = -2.08293255E+01;
    COFD[1305] = 5.35267674E+00;
    COFD[1306] = -4.69010505E-01;
    COFD[1307] = 1.98979152E-02;
    COFD[1308] = -1.63254691E+01;
    COFD[1309] = 3.82388595E+00;
    COFD[1310] = -2.84480724E-01;
    COFD[1311] = 1.24506311E-02;
    COFD[1312] = -1.63493345E+01;
    COFD[1313] = 3.82388595E+00;
    COFD[1314] = -2.84480724E-01;
    COFD[1315] = 1.24506311E-02;
    COFD[1316] = -2.04928958E+01;
    COFD[1317] = 5.22397933E+00;
    COFD[1318] = -4.54138171E-01;
    COFD[1319] = 1.93249285E-02;
    COFD[1320] = -1.62724462E+01;
    COFD[1321] = 3.79163564E+00;
    COFD[1322] = -2.80257365E-01;
    COFD[1323] = 1.22656902E-02;
    COFD[1324] = -2.07595845E+01;
    COFD[1325] = 5.32244593E+00;
    COFD[1326] = -4.65829403E-01;
    COFD[1327] = 1.97895274E-02;
    COFD[1328] = -1.85844688E+01;
    COFD[1329] = 4.51052425E+00;
    COFD[1330] = -3.70301627E-01;
    COFD[1331] = 1.60416153E-02;
    COFD[1332] = -1.85756076E+01;
    COFD[1333] = 4.51052425E+00;
    COFD[1334] = -3.70301627E-01;
    COFD[1335] = 1.60416153E-02;
    COFD[1336] = -2.05184870E+01;
    COFD[1337] = 5.18417470E+00;
    COFD[1338] = -4.49491573E-01;
    COFD[1339] = 1.91438508E-02;
    COFD[1340] = -2.05375724E+01;
    COFD[1341] = 5.18417470E+00;
    COFD[1342] = -4.49491573E-01;
    COFD[1343] = 1.91438508E-02;
    COFD[1344] = -2.08463209E+01;
    COFD[1345] = 5.32244593E+00;
    COFD[1346] = -4.65829403E-01;
    COFD[1347] = 1.97895274E-02;
    COFD[1348] = -2.08505864E+01;
    COFD[1349] = 5.32244593E+00;
    COFD[1350] = -4.65829403E-01;
    COFD[1351] = 1.97895274E-02;
    COFD[1352] = -1.86507213E+01;
    COFD[1353] = 4.60874797E+00;
    COFD[1354] = -3.82368716E-01;
    COFD[1355] = 1.65370164E-02;
    COFD[1356] = -1.64169433E+01;
    COFD[1357] = 3.89309916E+00;
    COFD[1358] = -2.93528188E-01;
    COFD[1359] = 1.28463177E-02;
    COFD[1360] = -1.86611023E+01;
    COFD[1361] = 4.60874797E+00;
    COFD[1362] = -3.82368716E-01;
    COFD[1363] = 1.65370164E-02;
    COFD[1364] = -2.05235731E+01;
    COFD[1365] = 5.18417470E+00;
    COFD[1366] = -4.49491573E-01;
    COFD[1367] = 1.91438508E-02;
    COFD[1368] = -1.93917298E+01;
    COFD[1369] = 4.78708023E+00;
    COFD[1370] = -4.03693144E-01;
    COFD[1371] = 1.73884817E-02;
    COFD[1372] = -1.82145353E+01;
    COFD[1373] = 4.46848269E+00;
    COFD[1374] = -3.65269718E-01;
    COFD[1375] = 1.58407652E-02;
    COFD[1376] = -2.09481051E+01;
    COFD[1377] = 5.28755355E+00;
    COFD[1378] = -4.61641920E-01;
    COFD[1379] = 1.96208961E-02;
    COFD[1380] = -2.05235731E+01;
    COFD[1381] = 5.18417470E+00;
    COFD[1382] = -4.49491573E-01;
    COFD[1383] = 1.91438508E-02;
    COFD[1384] = -2.10057003E+01;
    COFD[1385] = 5.28755355E+00;
    COFD[1386] = -4.61641920E-01;
    COFD[1387] = 1.96208961E-02;
    COFD[1388] = -1.79791019E+01;
    COFD[1389] = 4.29613154E+00;
    COFD[1390] = -3.44012526E-01;
    COFD[1391] = 1.49643715E-02;
    COFD[1392] = -1.95819005E+01;
    COFD[1393] = 4.84393038E+00;
    COFD[1394] = -4.10274737E-01;
    COFD[1395] = 1.76417458E-02;
    COFD[1396] = -1.95875976E+01;
    COFD[1397] = 4.84393038E+00;
    COFD[1398] = -4.10274737E-01;
    COFD[1399] = 1.76417458E-02;
    COFD[1400] = -1.94912151E+01;
    COFD[1401] = 4.81575071E+00;
    COFD[1402] = -4.07042139E-01;
    COFD[1403] = 1.75187504E-02;
    COFD[1404] = -1.95201830E+01;
    COFD[1405] = 4.77151544E+00;
    COFD[1406] = -4.01882811E-01;
    COFD[1407] = 1.73184814E-02;
    COFD[1408] = -2.13698722E+01;
    COFD[1409] = 5.34971865E+00;
    COFD[1410] = -4.68771123E-01;
    COFD[1411] = 1.98933811E-02;
    COFD[1412] = -2.01315602E+01;
    COFD[1413] = 4.97613338E+00;
    COFD[1414] = -4.26175206E-01;
    COFD[1415] = 1.82809270E-02;
    COFD[1416] = -2.00964665E+01;
    COFD[1417] = 4.96870443E+00;
    COFD[1418] = -4.25292447E-01;
    COFD[1419] = 1.82459096E-02;
    COFD[1420] = -2.00997774E+01;
    COFD[1421] = 4.96870443E+00;
    COFD[1422] = -4.25292447E-01;
    COFD[1423] = 1.82459096E-02;
    COFD[1424] = -2.02121663E+01;
    COFD[1425] = 4.95786261E+00;
    COFD[1426] = -4.24013131E-01;
    COFD[1427] = 1.81955669E-02;
    COFD[1428] = -2.14416336E+01;
    COFD[1429] = 5.35040988E+00;
    COFD[1430] = -4.68827063E-01;
    COFD[1431] = 1.98944407E-02;
    COFD[1432] = -2.07257272E+01;
    COFD[1433] = 5.10688723E+00;
    COFD[1434] = -4.41563971E-01;
    COFD[1435] = 1.88857198E-02;
    COFD[1436] = -2.06463142E+01;
    COFD[1437] = 5.07657482E+00;
    COFD[1438] = -4.38028804E-01;
    COFD[1439] = 1.87481371E-02;
    COFD[1440] = -2.09258526E+01;
    COFD[1441] = 5.19811866E+00;
    COFD[1442] = -4.51121211E-01;
    COFD[1443] = 1.92074617E-02;
    COFD[1444] = -2.14057339E+01;
    COFD[1445] = 5.33269880E+00;
    COFD[1446] = -4.67008439E-01;
    COFD[1447] = 1.98347416E-02;
    COFD[1448] = -2.13955999E+01;
    COFD[1449] = 5.25183817E+00;
    COFD[1450] = -4.57376333E-01;
    COFD[1451] = 1.94504429E-02;
    COFD[1452] = -2.13968281E+01;
    COFD[1453] = 5.25183817E+00;
    COFD[1454] = -4.57376333E-01;
    COFD[1455] = 1.94504429E-02;
    COFD[1456] = -1.52486273E+01;
    COFD[1457] = 3.35922578E+00;
    COFD[1458] = -2.25181399E-01;
    COFD[1459] = 9.92132878E-03;
    COFD[1460] = -1.43190389E+01;
    COFD[1461] = 3.17651319E+00;
    COFD[1462] = -2.02028974E-01;
    COFD[1463] = 8.94232502E-03;
    COFD[1464] = -1.18998012E+01;
    COFD[1465] = 2.57507000E+00;
    COFD[1466] = -1.24033737E-01;
    COFD[1467] = 5.56694959E-03;
    COFD[1468] = -1.46554748E+01;
    COFD[1469] = 3.83606243E+00;
    COFD[1470] = -2.86076532E-01;
    COFD[1471] = 1.25205829E-02;
    COFD[1472] = -1.43394069E+01;
    COFD[1473] = 3.17651319E+00;
    COFD[1474] = -2.02028974E-01;
    COFD[1475] = 8.94232502E-03;
    COFD[1476] = -2.06463744E+01;
    COFD[1477] = 5.41688482E+00;
    COFD[1478] = -4.73387188E-01;
    COFD[1479] = 1.99280175E-02;
    COFD[1480] = -1.55588279E+01;
    COFD[1481] = 3.48070094E+00;
    COFD[1482] = -2.40859499E-01;
    COFD[1483] = 1.05972514E-02;
    COFD[1484] = -1.55666415E+01;
    COFD[1485] = 3.48070094E+00;
    COFD[1486] = -2.40859499E-01;
    COFD[1487] = 1.05972514E-02;
    COFD[1488] = -1.55741053E+01;
    COFD[1489] = 3.48070094E+00;
    COFD[1490] = -2.40859499E-01;
    COFD[1491] = 1.05972514E-02;
    COFD[1492] = -1.42473439E+01;
    COFD[1493] = 3.17651319E+00;
    COFD[1494] = -2.02028974E-01;
    COFD[1495] = 8.94232502E-03;
    COFD[1496] = -2.08277598E+01;
    COFD[1497] = 5.35267674E+00;
    COFD[1498] = -4.69010505E-01;
    COFD[1499] = 1.98979152E-02;
    COFD[1500] = -1.63301444E+01;
    COFD[1501] = 3.82388595E+00;
    COFD[1502] = -2.84480724E-01;
    COFD[1503] = 1.24506311E-02;
    COFD[1504] = -1.84777607E+01;
    COFD[1505] = 4.49330851E+00;
    COFD[1506] = -3.68208715E-01;
    COFD[1507] = 1.59565402E-02;
    COFD[1508] = -1.52792891E+01;
    COFD[1509] = 3.36790500E+00;
    COFD[1510] = -2.26321740E-01;
    COFD[1511] = 9.97135055E-03;
    COFD[1512] = -2.08367725E+01;
    COFD[1513] = 5.35267674E+00;
    COFD[1514] = -4.69010505E-01;
    COFD[1515] = 1.98979152E-02;
    COFD[1516] = -1.63301444E+01;
    COFD[1517] = 3.82388595E+00;
    COFD[1518] = -2.84480724E-01;
    COFD[1519] = 1.24506311E-02;
    COFD[1520] = -1.63542394E+01;
    COFD[1521] = 3.82388595E+00;
    COFD[1522] = -2.84480724E-01;
    COFD[1523] = 1.24506311E-02;
    COFD[1524] = -2.02637994E+01;
    COFD[1525] = 5.14984081E+00;
    COFD[1526] = -4.46093018E-01;
    COFD[1527] = 1.90396647E-02;
    COFD[1528] = -1.62775714E+01;
    COFD[1529] = 3.79163564E+00;
    COFD[1530] = -2.80257365E-01;
    COFD[1531] = 1.22656902E-02;
    COFD[1532] = -2.07672833E+01;
    COFD[1533] = 5.32244593E+00;
    COFD[1534] = -4.65829403E-01;
    COFD[1535] = 1.97895274E-02;
    COFD[1536] = -1.85919214E+01;
    COFD[1537] = 4.51052425E+00;
    COFD[1538] = -3.70301627E-01;
    COFD[1539] = 1.60416153E-02;
    COFD[1540] = -1.85829283E+01;
    COFD[1541] = 4.51052425E+00;
    COFD[1542] = -3.70301627E-01;
    COFD[1543] = 1.60416153E-02;
    COFD[1544] = -2.05272328E+01;
    COFD[1545] = 5.18417470E+00;
    COFD[1546] = -4.49491573E-01;
    COFD[1547] = 1.91438508E-02;
    COFD[1548] = -2.05466616E+01;
    COFD[1549] = 5.18417470E+00;
    COFD[1550] = -4.49491573E-01;
    COFD[1551] = 1.91438508E-02;
    COFD[1552] = -2.08554914E+01;
    COFD[1553] = 5.32244593E+00;
    COFD[1554] = -4.65829403E-01;
    COFD[1555] = 1.97895274E-02;
    COFD[1556] = -2.08598363E+01;
    COFD[1557] = 5.32244593E+00;
    COFD[1558] = -4.65829403E-01;
    COFD[1559] = 1.97895274E-02;
    COFD[1560] = -1.86576191E+01;
    COFD[1561] = 4.60874797E+00;
    COFD[1562] = -3.82368716E-01;
    COFD[1563] = 1.65370164E-02;
    COFD[1564] = -1.64255964E+01;
    COFD[1565] = 3.89309916E+00;
    COFD[1566] = -2.93528188E-01;
    COFD[1567] = 1.28463177E-02;
    COFD[1568] = -1.86681459E+01;
    COFD[1569] = 4.60874797E+00;
    COFD[1570] = -3.82368716E-01;
    COFD[1571] = 1.65370164E-02;
    COFD[1572] = -2.05324091E+01;
    COFD[1573] = 5.18417470E+00;
    COFD[1574] = -4.49491573E-01;
    COFD[1575] = 1.91438508E-02;
    COFD[1576] = -1.94004795E+01;
    COFD[1577] = 4.78708023E+00;
    COFD[1578] = -4.03693144E-01;
    COFD[1579] = 1.73884817E-02;
    COFD[1580] = -1.82217198E+01;
    COFD[1581] = 4.46848269E+00;
    COFD[1582] = -3.65269718E-01;
    COFD[1583] = 1.58407652E-02;
    COFD[1584] = -2.09571146E+01;
    COFD[1585] = 5.28755355E+00;
    COFD[1586] = -4.61641920E-01;
    COFD[1587] = 1.96208961E-02;
    COFD[1588] = -2.05324091E+01;
    COFD[1589] = 5.18417470E+00;
    COFD[1590] = -4.49491573E-01;
    COFD[1591] = 1.91438508E-02;
    COFD[1592] = -2.10158209E+01;
    COFD[1593] = 5.28755355E+00;
    COFD[1594] = -4.61641920E-01;
    COFD[1595] = 1.96208961E-02;
    COFD[1596] = -1.79874655E+01;
    COFD[1597] = 4.29613154E+00;
    COFD[1598] = -3.44012526E-01;
    COFD[1599] = 1.49643715E-02;
    COFD[1600] = -1.95903647E+01;
    COFD[1601] = 4.84393038E+00;
    COFD[1602] = -4.10274737E-01;
    COFD[1603] = 1.76417458E-02;
    COFD[1604] = -1.95961596E+01;
    COFD[1605] = 4.84393038E+00;
    COFD[1606] = -4.10274737E-01;
    COFD[1607] = 1.76417458E-02;
    COFD[1608] = -1.94998722E+01;
    COFD[1609] = 4.81575071E+00;
    COFD[1610] = -4.07042139E-01;
    COFD[1611] = 1.75187504E-02;
    COFD[1612] = -1.95290229E+01;
    COFD[1613] = 4.77151544E+00;
    COFD[1614] = -4.01882811E-01;
    COFD[1615] = 1.73184814E-02;
    COFD[1616] = -2.12907159E+01;
    COFD[1617] = 5.32167660E+00;
    COFD[1618] = -4.65740624E-01;
    COFD[1619] = 1.97861081E-02;
    COFD[1620] = -2.01412473E+01;
    COFD[1621] = 4.97613338E+00;
    COFD[1622] = -4.26175206E-01;
    COFD[1623] = 1.82809270E-02;
    COFD[1624] = -2.01062206E+01;
    COFD[1625] = 4.96870443E+00;
    COFD[1626] = -4.25292447E-01;
    COFD[1627] = 1.82459096E-02;
    COFD[1628] = -2.01095969E+01;
    COFD[1629] = 4.96870443E+00;
    COFD[1630] = -4.25292447E-01;
    COFD[1631] = 1.82459096E-02;
    COFD[1632] = -2.02220498E+01;
    COFD[1633] = 4.95786261E+00;
    COFD[1634] = -4.24013131E-01;
    COFD[1635] = 1.81955669E-02;
    COFD[1636] = -2.14529967E+01;
    COFD[1637] = 5.35040988E+00;
    COFD[1638] = -4.68827063E-01;
    COFD[1639] = 1.98944407E-02;
    COFD[1640] = -2.07362753E+01;
    COFD[1641] = 5.10688723E+00;
    COFD[1642] = -4.41563971E-01;
    COFD[1643] = 1.88857198E-02;
    COFD[1644] = -2.06522508E+01;
    COFD[1645] = 5.07501764E+00;
    COFD[1646] = -4.37846596E-01;
    COFD[1647] = 1.87410133E-02;
    COFD[1648] = -2.09364971E+01;
    COFD[1649] = 5.19811866E+00;
    COFD[1650] = -4.51121211E-01;
    COFD[1651] = 1.92074617E-02;
    COFD[1652] = -2.14169211E+01;
    COFD[1653] = 5.33269880E+00;
    COFD[1654] = -4.67008439E-01;
    COFD[1655] = 1.98347416E-02;
    COFD[1656] = -2.14072803E+01;
    COFD[1657] = 5.25183817E+00;
    COFD[1658] = -4.57376333E-01;
    COFD[1659] = 1.94504429E-02;
    COFD[1660] = -2.14085375E+01;
    COFD[1661] = 5.25183817E+00;
    COFD[1662] = -4.57376333E-01;
    COFD[1663] = 1.94504429E-02;
    COFD[1664] = -1.52554761E+01;
    COFD[1665] = 3.35922578E+00;
    COFD[1666] = -2.25181399E-01;
    COFD[1667] = 9.92132878E-03;
    COFD[1668] = -1.43238998E+01;
    COFD[1669] = 3.17651319E+00;
    COFD[1670] = -2.02028974E-01;
    COFD[1671] = 8.94232502E-03;
    COFD[1672] = -1.19006548E+01;
    COFD[1673] = 2.57507000E+00;
    COFD[1674] = -1.24033737E-01;
    COFD[1675] = 5.56694959E-03;
    COFD[1676] = -1.46559141E+01;
    COFD[1677] = 3.83606243E+00;
    COFD[1678] = -2.86076532E-01;
    COFD[1679] = 1.25205829E-02;
    COFD[1680] = -1.43444709E+01;
    COFD[1681] = 3.17651319E+00;
    COFD[1682] = -2.02028974E-01;
    COFD[1683] = 8.94232502E-03;
    COFD[1684] = -2.06516336E+01;
    COFD[1685] = 5.41688482E+00;
    COFD[1686] = -4.73387188E-01;
    COFD[1687] = 1.99280175E-02;
    COFD[1688] = -1.55661750E+01;
    COFD[1689] = 3.48070094E+00;
    COFD[1690] = -2.40859499E-01;
    COFD[1691] = 1.05972514E-02;
    COFD[1692] = -1.55741053E+01;
    COFD[1693] = 3.48070094E+00;
    COFD[1694] = -2.40859499E-01;
    COFD[1695] = 1.05972514E-02;
    COFD[1696] = -1.55816822E+01;
    COFD[1697] = 3.48070094E+00;
    COFD[1698] = -2.40859499E-01;
    COFD[1699] = 1.05972514E-02;
    COFD[1700] = -1.42515527E+01;
    COFD[1701] = 3.17651319E+00;
    COFD[1702] = -2.02028974E-01;
    COFD[1703] = 8.94232502E-03;
    COFD[1704] = -2.08347403E+01;
    COFD[1705] = 5.35267674E+00;
    COFD[1706] = -4.69010505E-01;
    COFD[1707] = 1.98979152E-02;
    COFD[1708] = -1.63345829E+01;
    COFD[1709] = 3.82388595E+00;
    COFD[1710] = -2.84480724E-01;
    COFD[1711] = 1.24506311E-02;
    COFD[1712] = -1.84863000E+01;
    COFD[1713] = 4.49330851E+00;
    COFD[1714] = -3.68208715E-01;
    COFD[1715] = 1.59565402E-02;
    COFD[1716] = -1.52861376E+01;
    COFD[1717] = 3.36790500E+00;
    COFD[1718] = -2.26321740E-01;
    COFD[1719] = 9.97135055E-03;
    COFD[1720] = -2.08438809E+01;
    COFD[1721] = 5.35267674E+00;
    COFD[1722] = -4.69010505E-01;
    COFD[1723] = 1.98979152E-02;
    COFD[1724] = -1.63345829E+01;
    COFD[1725] = 3.82388595E+00;
    COFD[1726] = -2.84480724E-01;
    COFD[1727] = 1.24506311E-02;
    COFD[1728] = -1.63588981E+01;
    COFD[1729] = 3.82388595E+00;
    COFD[1730] = -2.84480724E-01;
    COFD[1731] = 1.24506311E-02;
    COFD[1732] = -2.02710316E+01;
    COFD[1733] = 5.14984081E+00;
    COFD[1734] = -4.46093018E-01;
    COFD[1735] = 1.90396647E-02;
    COFD[1736] = -1.62824412E+01;
    COFD[1737] = 3.79163564E+00;
    COFD[1738] = -2.80257365E-01;
    COFD[1739] = 1.22656902E-02;
    COFD[1740] = -2.07746356E+01;
    COFD[1741] = 5.32244593E+00;
    COFD[1742] = -4.65829403E-01;
    COFD[1743] = 1.97895274E-02;
    COFD[1744] = -1.85990352E+01;
    COFD[1745] = 4.51052425E+00;
    COFD[1746] = -3.70301627E-01;
    COFD[1747] = 1.60416153E-02;
    COFD[1748] = -1.85899144E+01;
    COFD[1749] = 4.51052425E+00;
    COFD[1750] = -3.70301627E-01;
    COFD[1751] = 1.60416153E-02;
    COFD[1752] = -2.05356023E+01;
    COFD[1753] = 5.18417470E+00;
    COFD[1754] = -4.49491573E-01;
    COFD[1755] = 1.91438508E-02;
    COFD[1756] = -2.05553656E+01;
    COFD[1757] = 5.18417470E+00;
    COFD[1758] = -4.49491573E-01;
    COFD[1759] = 1.91438508E-02;
    COFD[1760] = -2.08642748E+01;
    COFD[1761] = 5.32244593E+00;
    COFD[1762] = -4.65829403E-01;
    COFD[1763] = 1.97895274E-02;
    COFD[1764] = -2.08686970E+01;
    COFD[1765] = 5.32244593E+00;
    COFD[1766] = -4.65829403E-01;
    COFD[1767] = 1.97895274E-02;
    COFD[1768] = -1.86641962E+01;
    COFD[1769] = 4.60874797E+00;
    COFD[1770] = -3.82368716E-01;
    COFD[1771] = 1.65370164E-02;
    COFD[1772] = -1.64338757E+01;
    COFD[1773] = 3.89309916E+00;
    COFD[1774] = -2.93528188E-01;
    COFD[1775] = 1.28463177E-02;
    COFD[1776] = -1.86748638E+01;
    COFD[1777] = 4.60874797E+00;
    COFD[1778] = -3.82368716E-01;
    COFD[1779] = 1.65370164E-02;
    COFD[1780] = -2.05408665E+01;
    COFD[1781] = 5.18417470E+00;
    COFD[1782] = -4.49491573E-01;
    COFD[1783] = 1.91438508E-02;
    COFD[1784] = -1.94088529E+01;
    COFD[1785] = 4.78708023E+00;
    COFD[1786] = -4.03693144E-01;
    COFD[1787] = 1.73884817E-02;
    COFD[1788] = -1.82285740E+01;
    COFD[1789] = 4.46848269E+00;
    COFD[1790] = -3.65269718E-01;
    COFD[1791] = 1.58407652E-02;
    COFD[1792] = -2.09657408E+01;
    COFD[1793] = 5.28755355E+00;
    COFD[1794] = -4.61641920E-01;
    COFD[1795] = 1.96208961E-02;
    COFD[1796] = -2.05408665E+01;
    COFD[1797] = 5.18417470E+00;
    COFD[1798] = -4.49491573E-01;
    COFD[1799] = 1.91438508E-02;
    COFD[1800] = -2.10255323E+01;
    COFD[1801] = 5.28755355E+00;
    COFD[1802] = -4.61641920E-01;
    COFD[1803] = 1.96208961E-02;
    COFD[1804] = -1.79954632E+01;
    COFD[1805] = 4.29613154E+00;
    COFD[1806] = -3.44012526E-01;
    COFD[1807] = 1.49643715E-02;
    COFD[1808] = -1.95984602E+01;
    COFD[1809] = 4.84393038E+00;
    COFD[1810] = -4.10274737E-01;
    COFD[1811] = 1.76417458E-02;
    COFD[1812] = -1.96043503E+01;
    COFD[1813] = 4.84393038E+00;
    COFD[1814] = -4.10274737E-01;
    COFD[1815] = 1.76417458E-02;
    COFD[1816] = -1.95081555E+01;
    COFD[1817] = 4.81575071E+00;
    COFD[1818] = -4.07042139E-01;
    COFD[1819] = 1.75187504E-02;
    COFD[1820] = -1.95374840E+01;
    COFD[1821] = 4.77151544E+00;
    COFD[1822] = -4.01882811E-01;
    COFD[1823] = 1.73184814E-02;
    COFD[1824] = -2.13011157E+01;
    COFD[1825] = 5.32167660E+00;
    COFD[1826] = -4.65740624E-01;
    COFD[1827] = 1.97861081E-02;
    COFD[1828] = -2.01505348E+01;
    COFD[1829] = 4.97613338E+00;
    COFD[1830] = -4.26175206E-01;
    COFD[1831] = 1.82809270E-02;
    COFD[1832] = -2.01155735E+01;
    COFD[1833] = 4.96870443E+00;
    COFD[1834] = -4.25292447E-01;
    COFD[1835] = 1.82459096E-02;
    COFD[1836] = -2.01190139E+01;
    COFD[1837] = 4.96870443E+00;
    COFD[1838] = -4.25292447E-01;
    COFD[1839] = 1.82459096E-02;
    COFD[1840] = -2.02315293E+01;
    COFD[1841] = 4.95786261E+00;
    COFD[1842] = -4.24013131E-01;
    COFD[1843] = 1.81955669E-02;
    COFD[1844] = -2.14639274E+01;
    COFD[1845] = 5.35040988E+00;
    COFD[1846] = -4.68827063E-01;
    COFD[1847] = 1.98944407E-02;
    COFD[1848] = -2.07464056E+01;
    COFD[1849] = 5.10688723E+00;
    COFD[1850] = -4.41563971E-01;
    COFD[1851] = 1.88857198E-02;
    COFD[1852] = -2.06624288E+01;
    COFD[1853] = 5.07501764E+00;
    COFD[1854] = -4.37846596E-01;
    COFD[1855] = 1.87410133E-02;
    COFD[1856] = -2.09467220E+01;
    COFD[1857] = 5.19811866E+00;
    COFD[1858] = -4.51121211E-01;
    COFD[1859] = 1.92074617E-02;
    COFD[1860] = -2.14276788E+01;
    COFD[1861] = 5.33269880E+00;
    COFD[1862] = -4.67008439E-01;
    COFD[1863] = 1.98347416E-02;
    COFD[1864] = -2.14185232E+01;
    COFD[1865] = 5.25183817E+00;
    COFD[1866] = -4.57376333E-01;
    COFD[1867] = 1.94504429E-02;
    COFD[1868] = -2.14198091E+01;
    COFD[1869] = 5.25183817E+00;
    COFD[1870] = -4.57376333E-01;
    COFD[1871] = 1.94504429E-02;
    COFD[1872] = -1.40076852E+01;
    COFD[1873] = 3.07549274E+00;
    COFD[1874] = -1.88889344E-01;
    COFD[1875] = 8.37152866E-03;
    COFD[1876] = -1.31551788E+01;
    COFD[1877] = 2.90778936E+00;
    COFD[1878] = -1.67388544E-01;
    COFD[1879] = 7.45220609E-03;
    COFD[1880] = -1.09469245E+01;
    COFD[1881] = 2.30836460E+00;
    COFD[1882] = -8.76339315E-02;
    COFD[1883] = 3.90878445E-03;
    COFD[1884] = -1.34162893E+01;
    COFD[1885] = 3.48624238E+00;
    COFD[1886] = -2.41554467E-01;
    COFD[1887] = 1.06263545E-02;
    COFD[1888] = -1.31686537E+01;
    COFD[1889] = 2.90778936E+00;
    COFD[1890] = -1.67388544E-01;
    COFD[1891] = 7.45220609E-03;
    COFD[1892] = -1.93521390E+01;
    COFD[1893] = 5.16013126E+00;
    COFD[1894] = -4.46824543E-01;
    COFD[1895] = 1.90464887E-02;
    COFD[1896] = -1.42429085E+01;
    COFD[1897] = 3.17651319E+00;
    COFD[1898] = -2.02028974E-01;
    COFD[1899] = 8.94232502E-03;
    COFD[1900] = -1.42473439E+01;
    COFD[1901] = 3.17651319E+00;
    COFD[1902] = -2.02028974E-01;
    COFD[1903] = 8.94232502E-03;
    COFD[1904] = -1.42515527E+01;
    COFD[1905] = 3.17651319E+00;
    COFD[1906] = -2.02028974E-01;
    COFD[1907] = 8.94232502E-03;
    COFD[1908] = -1.31062967E+01;
    COFD[1909] = 2.90778936E+00;
    COFD[1910] = -1.67388544E-01;
    COFD[1911] = 7.45220609E-03;
    COFD[1912] = -1.93624931E+01;
    COFD[1913] = 5.02567894E+00;
    COFD[1914] = -4.32045169E-01;
    COFD[1915] = 1.85132214E-02;
    COFD[1916] = -1.50076254E+01;
    COFD[1917] = 3.47945612E+00;
    COFD[1918] = -2.40703722E-01;
    COFD[1919] = 1.05907441E-02;
    COFD[1920] = -1.69758891E+01;
    COFD[1921] = 4.14240922E+00;
    COFD[1922] = -3.25239774E-01;
    COFD[1923] = 1.41980687E-02;
    COFD[1924] = -1.40318948E+01;
    COFD[1925] = 3.08120012E+00;
    COFD[1926] = -1.89629903E-01;
    COFD[1927] = 8.40361952E-03;
    COFD[1928] = -1.93677186E+01;
    COFD[1929] = 5.02567894E+00;
    COFD[1930] = -4.32045169E-01;
    COFD[1931] = 1.85132214E-02;
    COFD[1932] = -1.50076254E+01;
    COFD[1933] = 3.47945612E+00;
    COFD[1934] = -2.40703722E-01;
    COFD[1935] = 1.05907441E-02;
    COFD[1936] = -1.50240272E+01;
    COFD[1937] = 3.47945612E+00;
    COFD[1938] = -2.40703722E-01;
    COFD[1939] = 1.05907441E-02;
    COFD[1940] = -1.87476063E+01;
    COFD[1941] = 4.79683898E+00;
    COFD[1942] = -4.04829719E-01;
    COFD[1943] = 1.74325475E-02;
    COFD[1944] = -1.49727799E+01;
    COFD[1945] = 3.46140064E+00;
    COFD[1946] = -2.38440092E-01;
    COFD[1947] = 1.04960087E-02;
    COFD[1948] = -1.92654138E+01;
    COFD[1949] = 4.98286777E+00;
    COFD[1950] = -4.26970814E-01;
    COFD[1951] = 1.83122917E-02;
    COFD[1952] = -1.71416702E+01;
    COFD[1953] = 4.15807461E+00;
    COFD[1954] = -3.27178539E-01;
    COFD[1955] = 1.42784349E-02;
    COFD[1956] = -1.71364578E+01;
    COFD[1957] = 4.15807461E+00;
    COFD[1958] = -3.27178539E-01;
    COFD[1959] = 1.42784349E-02;
    COFD[1960] = -1.90116191E+01;
    COFD[1961] = 4.84384483E+00;
    COFD[1962] = -4.10265575E-01;
    COFD[1963] = 1.76414287E-02;
    COFD[1964] = -1.90219707E+01;
    COFD[1965] = 4.84384483E+00;
    COFD[1966] = -4.10265575E-01;
    COFD[1967] = 1.76414287E-02;
    COFD[1968] = -1.93137183E+01;
    COFD[1969] = 4.98286777E+00;
    COFD[1970] = -4.26970814E-01;
    COFD[1971] = 1.83122917E-02;
    COFD[1972] = -1.93159978E+01;
    COFD[1973] = 4.98286777E+00;
    COFD[1974] = -4.26970814E-01;
    COFD[1975] = 1.83122917E-02;
    COFD[1976] = -1.71623017E+01;
    COFD[1977] = 4.24084025E+00;
    COFD[1978] = -3.37428619E-01;
    COFD[1979] = 1.47032793E-02;
    COFD[1980] = -1.48738066E+01;
    COFD[1981] = 3.52327209E+00;
    COFD[1982] = -2.46286208E-01;
    COFD[1983] = 1.08285963E-02;
    COFD[1984] = -1.71685520E+01;
    COFD[1985] = 4.24084025E+00;
    COFD[1986] = -3.37428619E-01;
    COFD[1987] = 1.47032793E-02;
    COFD[1988] = -1.90143953E+01;
    COFD[1989] = 4.84384483E+00;
    COFD[1990] = -4.10265575E-01;
    COFD[1991] = 1.76414287E-02;
    COFD[1992] = -1.78593879E+01;
    COFD[1993] = 4.42139452E+00;
    COFD[1994] = -3.59567329E-01;
    COFD[1995] = 1.56103969E-02;
    COFD[1996] = -1.67662974E+01;
    COFD[1997] = 4.11954900E+00;
    COFD[1998] = -3.22470391E-01;
    COFD[1999] = 1.40859564E-02;
    COFD[2000] = -1.94534227E+01;
    COFD[2001] = 4.95249173E+00;
    COFD[2002] = -4.23376552E-01;
    COFD[2003] = 1.81703714E-02;
    COFD[2004] = -1.90143953E+01;
    COFD[2005] = 4.84384483E+00;
    COFD[2006] = -4.10265575E-01;
    COFD[2007] = 1.76414287E-02;
    COFD[2008] = -1.94836874E+01;
    COFD[2009] = 4.95249173E+00;
    COFD[2010] = -4.23376552E-01;
    COFD[2011] = 1.81703714E-02;
    COFD[2012] = -1.65381278E+01;
    COFD[2013] = 3.95035840E+00;
    COFD[2014] = -3.00959418E-01;
    COFD[2015] = 1.31692593E-02;
    COFD[2016] = -1.80710700E+01;
    COFD[2017] = 4.48398491E+00;
    COFD[2018] = -3.67097129E-01;
    COFD[2019] = 1.59123634E-02;
    COFD[2020] = -1.80742247E+01;
    COFD[2021] = 4.48398491E+00;
    COFD[2022] = -3.67097129E-01;
    COFD[2023] = 1.59123634E-02;
    COFD[2024] = -1.79718457E+01;
    COFD[2025] = 4.45434023E+00;
    COFD[2026] = -3.63584633E-01;
    COFD[2027] = 1.57739270E-02;
    COFD[2028] = -1.79952897E+01;
    COFD[2029] = 4.40247898E+00;
    COFD[2030] = -3.57238362E-01;
    COFD[2031] = 1.55145651E-02;
    COFD[2032] = -1.97431913E+01;
    COFD[2033] = 4.98207523E+00;
    COFD[2034] = -4.26877291E-01;
    COFD[2035] = 1.83086094E-02;
    COFD[2036] = -1.85839192E+01;
    COFD[2037] = 4.61260432E+00;
    COFD[2038] = -3.82854484E-01;
    COFD[2039] = 1.65575163E-02;
    COFD[2040] = -1.85418144E+01;
    COFD[2041] = 4.60336076E+00;
    COFD[2042] = -3.81691643E-01;
    COFD[2043] = 1.65085234E-02;
    COFD[2044] = -1.85435341E+01;
    COFD[2045] = 4.60336076E+00;
    COFD[2046] = -3.81691643E-01;
    COFD[2047] = 1.65085234E-02;
    COFD[2048] = -1.86611033E+01;
    COFD[2049] = 4.58956960E+00;
    COFD[2050] = -3.79964215E-01;
    COFD[2051] = 1.64361138E-02;
    COFD[2052] = -1.99183435E+01;
    COFD[2053] = 5.02095434E+00;
    COFD[2054] = -4.31496874E-01;
    COFD[2055] = 1.84920392E-02;
    COFD[2056] = -1.91552109E+01;
    COFD[2057] = 4.73921581E+00;
    COFD[2058] = -3.98017274E-01;
    COFD[2059] = 1.71639614E-02;
    COFD[2060] = -1.90778475E+01;
    COFD[2061] = 4.70966098E+00;
    COFD[2062] = -3.94551217E-01;
    COFD[2063] = 1.70286289E-02;
    COFD[2064] = -1.93921777E+01;
    COFD[2065] = 4.85518471E+00;
    COFD[2066] = -4.11551624E-01;
    COFD[2067] = 1.76895651E-02;
    COFD[2068] = -1.98683238E+01;
    COFD[2069] = 4.99367362E+00;
    COFD[2070] = -4.28249956E-01;
    COFD[2071] = 1.83628509E-02;
    COFD[2072] = -1.98885574E+01;
    COFD[2073] = 4.92184026E+00;
    COFD[2074] = -4.19745472E-01;
    COFD[2075] = 1.80268154E-02;
    COFD[2076] = -1.98891413E+01;
    COFD[2077] = 4.92184026E+00;
    COFD[2078] = -4.19745472E-01;
    COFD[2079] = 1.80268154E-02;
    COFD[2080] = -2.04750581E+01;
    COFD[2081] = 5.23112374E+00;
    COFD[2082] = -4.54967682E-01;
    COFD[2083] = 1.93570423E-02;
    COFD[2084] = -1.94313116E+01;
    COFD[2085] = 5.02567894E+00;
    COFD[2086] = -4.32045169E-01;
    COFD[2087] = 1.85132214E-02;
    COFD[2088] = -1.60517370E+01;
    COFD[2089] = 4.11188603E+00;
    COFD[2090] = -3.21540884E-01;
    COFD[2091] = 1.40482564E-02;
    COFD[2092] = -1.97544450E+01;
    COFD[2093] = 5.56931926E+00;
    COFD[2094] = -4.89105511E-01;
    COFD[2095] = 2.04493129E-02;
    COFD[2096] = -1.94507876E+01;
    COFD[2097] = 5.02567894E+00;
    COFD[2098] = -4.32045169E-01;
    COFD[2099] = 1.85132214E-02;
    COFD[2100] = -1.77498543E+01;
    COFD[2101] = 3.57475686E+00;
    COFD[2102] = -1.56396297E-01;
    COFD[2103] = 3.12157721E-03;
    COFD[2104] = -2.08204449E+01;
    COFD[2105] = 5.35267674E+00;
    COFD[2106] = -4.69010505E-01;
    COFD[2107] = 1.98979152E-02;
    COFD[2108] = -2.08277598E+01;
    COFD[2109] = 5.35267674E+00;
    COFD[2110] = -4.69010505E-01;
    COFD[2111] = 1.98979152E-02;
    COFD[2112] = -2.08347403E+01;
    COFD[2113] = 5.35267674E+00;
    COFD[2114] = -4.69010505E-01;
    COFD[2115] = 1.98979152E-02;
    COFD[2116] = -1.93624931E+01;
    COFD[2117] = 5.02567894E+00;
    COFD[2118] = -4.32045169E-01;
    COFD[2119] = 1.85132214E-02;
    COFD[2120] = -1.90328712E+01;
    COFD[2121] = 3.99221757E+00;
    COFD[2122] = -2.19854880E-01;
    COFD[2123] = 6.22736279E-03;
    COFD[2124] = -2.14160703E+01;
    COFD[2125] = 5.56531152E+00;
    COFD[2126] = -4.88789821E-01;
    COFD[2127] = 2.04437116E-02;
    COFD[2128] = -2.19215555E+01;
    COFD[2129] = 5.45216133E+00;
    COFD[2130] = -4.52916925E-01;
    COFD[2131] = 1.80456400E-02;
    COFD[2132] = -2.05045578E+01;
    COFD[2133] = 5.23843909E+00;
    COFD[2134] = -4.55815614E-01;
    COFD[2135] = 1.93898040E-02;
    COFD[2136] = -1.90413348E+01;
    COFD[2137] = 3.99221757E+00;
    COFD[2138] = -2.19854880E-01;
    COFD[2139] = 6.22736279E-03;
    COFD[2140] = -2.14160703E+01;
    COFD[2141] = 5.56531152E+00;
    COFD[2142] = -4.88789821E-01;
    COFD[2143] = 2.04437116E-02;
    COFD[2144] = -2.14391943E+01;
    COFD[2145] = 5.56531152E+00;
    COFD[2146] = -4.88789821E-01;
    COFD[2147] = 2.04437116E-02;
    COFD[2148] = -2.01801667E+01;
    COFD[2149] = 4.53183330E+00;
    COFD[2150] = -3.02186760E-01;
    COFD[2151] = 1.02756490E-02;
    COFD[2152] = -2.14022336E+01;
    COFD[2153] = 5.55346617E+00;
    COFD[2154] = -4.87783156E-01;
    COFD[2155] = 2.04210886E-02;
    COFD[2156] = -1.93125662E+01;
    COFD[2157] = 4.10954793E+00;
    COFD[2158] = -2.37523329E-01;
    COFD[2159] = 7.08858141E-03;
    COFD[2160] = -2.19700018E+01;
    COFD[2161] = 5.43750833E+00;
    COFD[2162] = -4.50273329E-01;
    COFD[2163] = 1.79013718E-02;
    COFD[2164] = -2.19615570E+01;
    COFD[2165] = 5.43750833E+00;
    COFD[2166] = -4.50273329E-01;
    COFD[2167] = 1.79013718E-02;
    COFD[2168] = -2.00915040E+01;
    COFD[2169] = 4.41511629E+00;
    COFD[2170] = -2.84086963E-01;
    COFD[2171] = 9.37586971E-03;
    COFD[2172] = -2.01095186E+01;
    COFD[2173] = 4.41511629E+00;
    COFD[2174] = -2.84086963E-01;
    COFD[2175] = 9.37586971E-03;
    COFD[2176] = -1.93946947E+01;
    COFD[2177] = 4.10954793E+00;
    COFD[2178] = -2.37523329E-01;
    COFD[2179] = 7.08858141E-03;
    COFD[2180] = -1.93987136E+01;
    COFD[2181] = 4.10954793E+00;
    COFD[2182] = -2.37523329E-01;
    COFD[2183] = 7.08858141E-03;
    COFD[2184] = -2.16718247E+01;
    COFD[2185] = 5.36811769E+00;
    COFD[2186] = -4.37727086E-01;
    COFD[2187] = 1.72167686E-02;
    COFD[2188] = -2.14204185E+01;
    COFD[2189] = 5.59268435E+00;
    COFD[2190] = -4.91232974E-01;
    COFD[2191] = 2.05064746E-02;
    COFD[2192] = -2.16817439E+01;
    COFD[2193] = 5.36811769E+00;
    COFD[2194] = -4.37727086E-01;
    COFD[2195] = 1.72167686E-02;
    COFD[2196] = -2.00963085E+01;
    COFD[2197] = 4.41511629E+00;
    COFD[2198] = -2.84086963E-01;
    COFD[2199] = 9.37586971E-03;
    COFD[2200] = -2.15702446E+01;
    COFD[2201] = 5.16868516E+00;
    COFD[2202] = -4.03721581E-01;
    COFD[2203] = 1.54206640E-02;
    COFD[2204] = -2.17771954E+01;
    COFD[2205] = 5.47519298E+00;
    COFD[2206] = -4.57113040E-01;
    COFD[2207] = 1.82758312E-02;
    COFD[2208] = -1.97545910E+01;
    COFD[2209] = 4.18758010E+00;
    COFD[2210] = -2.49327776E-01;
    COFD[2211] = 7.66559103E-03;
    COFD[2212] = -2.00963085E+01;
    COFD[2213] = 4.41511629E+00;
    COFD[2214] = -2.84086963E-01;
    COFD[2215] = 9.37586971E-03;
    COFD[2216] = -1.98087397E+01;
    COFD[2217] = 4.18758010E+00;
    COFD[2218] = -2.49327776E-01;
    COFD[2219] = 7.66559103E-03;
    COFD[2220] = -2.20866241E+01;
    COFD[2221] = 5.55935694E+00;
    COFD[2222] = -4.74154740E-01;
    COFD[2223] = 1.92584304E-02;
    COFD[2224] = -2.14300943E+01;
    COFD[2225] = 5.07680397E+00;
    COFD[2226] = -3.88612087E-01;
    COFD[2227] = 1.46395101E-02;
    COFD[2228] = -2.14354853E+01;
    COFD[2229] = 5.07680397E+00;
    COFD[2230] = -3.88612087E-01;
    COFD[2231] = 1.46395101E-02;
    COFD[2232] = -2.15159231E+01;
    COFD[2233] = 5.12799307E+00;
    COFD[2234] = -3.96938732E-01;
    COFD[2235] = 1.50673195E-02;
    COFD[2236] = -2.17825544E+01;
    COFD[2237] = 5.19232842E+00;
    COFD[2238] = -4.07643284E-01;
    COFD[2239] = 1.56246434E-02;
    COFD[2240] = -1.98237209E+01;
    COFD[2241] = 4.11158627E+00;
    COFD[2242] = -2.37831519E-01;
    COFD[2243] = 7.10363413E-03;
    COFD[2244] = -2.12109223E+01;
    COFD[2245] = 4.87252053E+00;
    COFD[2246] = -3.56127804E-01;
    COFD[2247] = 1.29948788E-02;
    COFD[2248] = -2.12219728E+01;
    COFD[2249] = 4.88535789E+00;
    COFD[2250] = -3.58153894E-01;
    COFD[2251] = 1.30969624E-02;
    COFD[2252] = -2.12250811E+01;
    COFD[2253] = 4.88535789E+00;
    COFD[2254] = -3.58153894E-01;
    COFD[2255] = 1.30969624E-02;
    COFD[2256] = -2.14030307E+01;
    COFD[2257] = 4.90439970E+00;
    COFD[2258] = -3.61162615E-01;
    COFD[2259] = 1.32486109E-02;
    COFD[2260] = -1.96731865E+01;
    COFD[2261] = 4.00653795E+00;
    COFD[2262] = -2.22005804E-01;
    COFD[2263] = 6.33194910E-03;
    COFD[2264] = -2.09802383E+01;
    COFD[2265] = 4.64167142E+00;
    COFD[2266] = -3.19532110E-01;
    COFD[2267] = 1.11478359E-02;
    COFD[2268] = -2.11071988E+01;
    COFD[2269] = 4.70311989E+00;
    COFD[2270] = -3.29240106E-01;
    COFD[2271] = 1.16366808E-02;
    COFD[2272] = -2.03849874E+01;
    COFD[2273] = 4.38396848E+00;
    COFD[2274] = -2.79298901E-01;
    COFD[2275] = 9.13915001E-03;
    COFD[2276] = -1.98477259E+01;
    COFD[2277] = 4.07958166E+00;
    COFD[2278] = -2.33006871E-01;
    COFD[2279] = 6.86822015E-03;
    COFD[2280] = -2.04488935E+01;
    COFD[2281] = 4.26473557E+00;
    COFD[2282] = -2.61033037E-01;
    COFD[2283] = 8.23906412E-03;
    COFD[2284] = -2.04500331E+01;
    COFD[2285] = 4.26473557E+00;
    COFD[2286] = -2.61033037E-01;
    COFD[2287] = 8.23906412E-03;
    COFD[2288] = -1.59404882E+01;
    COFD[2289] = 3.66853818E+00;
    COFD[2290] = -2.64346221E-01;
    COFD[2291] = 1.15784613E-02;
    COFD[2292] = -1.50584249E+01;
    COFD[2293] = 3.47945612E+00;
    COFD[2294] = -2.40703722E-01;
    COFD[2295] = 1.05907441E-02;
    COFD[2296] = -1.25098960E+01;
    COFD[2297] = 2.77873601E+00;
    COFD[2298] = -1.50637360E-01;
    COFD[2299] = 6.72684281E-03;
    COFD[2300] = -1.57972369E+01;
    COFD[2301] = 4.22225052E+00;
    COFD[2302] = -3.35156428E-01;
    COFD[2303] = 1.46104855E-02;
    COFD[2304] = -1.50724636E+01;
    COFD[2305] = 3.47945612E+00;
    COFD[2306] = -2.40703722E-01;
    COFD[2307] = 1.05907441E-02;
    COFD[2308] = -2.12639214E+01;
    COFD[2309] = 5.61184117E+00;
    COFD[2310] = -4.90532156E-01;
    COFD[2311] = 2.03507922E-02;
    COFD[2312] = -1.63254691E+01;
    COFD[2313] = 3.82388595E+00;
    COFD[2314] = -2.84480724E-01;
    COFD[2315] = 1.24506311E-02;
    COFD[2316] = -1.63301444E+01;
    COFD[2317] = 3.82388595E+00;
    COFD[2318] = -2.84480724E-01;
    COFD[2319] = 1.24506311E-02;
    COFD[2320] = -1.63345829E+01;
    COFD[2321] = 3.82388595E+00;
    COFD[2322] = -2.84480724E-01;
    COFD[2323] = 1.24506311E-02;
    COFD[2324] = -1.50076254E+01;
    COFD[2325] = 3.47945612E+00;
    COFD[2326] = -2.40703722E-01;
    COFD[2327] = 1.05907441E-02;
    COFD[2328] = -2.14160703E+01;
    COFD[2329] = 5.56531152E+00;
    COFD[2330] = -4.88789821E-01;
    COFD[2331] = 2.04437116E-02;
    COFD[2332] = -1.73027557E+01;
    COFD[2333] = 4.21416723E+00;
    COFD[2334] = -3.34163932E-01;
    COFD[2335] = 1.45697432E-02;
    COFD[2336] = -1.93015555E+01;
    COFD[2337] = 4.85015581E+00;
    COFD[2338] = -4.10945109E-01;
    COFD[2339] = 1.76651398E-02;
    COFD[2340] = -1.59634533E+01;
    COFD[2341] = 3.67388294E+00;
    COFD[2342] = -2.64990709E-01;
    COFD[2343] = 1.16042706E-02;
    COFD[2344] = -2.14215700E+01;
    COFD[2345] = 5.56531152E+00;
    COFD[2346] = -4.88789821E-01;
    COFD[2347] = 2.04437116E-02;
    COFD[2348] = -1.73027557E+01;
    COFD[2349] = 4.21416723E+00;
    COFD[2350] = -3.34163932E-01;
    COFD[2351] = 1.45697432E-02;
    COFD[2352] = -1.73198034E+01;
    COFD[2353] = 4.21416723E+00;
    COFD[2354] = -3.34163932E-01;
    COFD[2355] = 1.45697432E-02;
    COFD[2356] = -2.09376196E+01;
    COFD[2357] = 5.40870099E+00;
    COFD[2358] = -4.73017610E-01;
    COFD[2359] = 1.99399066E-02;
    COFD[2360] = -1.72556729E+01;
    COFD[2361] = 4.19029808E+00;
    COFD[2362] = -3.31177076E-01;
    COFD[2363] = 1.44446234E-02;
    COFD[2364] = -2.13538553E+01;
    COFD[2365] = 5.54007827E+00;
    COFD[2366] = -4.86434511E-01;
    COFD[2367] = 2.03779006E-02;
    COFD[2368] = -1.94585111E+01;
    COFD[2369] = 4.87180830E+00;
    COFD[2370] = -4.13582958E-01;
    COFD[2371] = 1.77726094E-02;
    COFD[2372] = -1.94530250E+01;
    COFD[2373] = 4.87180830E+00;
    COFD[2374] = -4.13582958E-01;
    COFD[2375] = 1.77726094E-02;
    COFD[2376] = -2.11349086E+01;
    COFD[2377] = 5.42846112E+00;
    COFD[2378] = -4.74321870E-01;
    COFD[2379] = 1.99459749E-02;
    COFD[2380] = -2.11458678E+01;
    COFD[2381] = 5.42846112E+00;
    COFD[2382] = -4.74321870E-01;
    COFD[2383] = 1.99459749E-02;
    COFD[2384] = -2.14048982E+01;
    COFD[2385] = 5.54007827E+00;
    COFD[2386] = -4.86434511E-01;
    COFD[2387] = 2.03779006E-02;
    COFD[2388] = -2.14073140E+01;
    COFD[2389] = 5.54007827E+00;
    COFD[2390] = -4.86434511E-01;
    COFD[2391] = 2.03779006E-02;
    COFD[2392] = -1.95548230E+01;
    COFD[2393] = 4.97133070E+00;
    COFD[2394] = -4.25604177E-01;
    COFD[2395] = 1.82582594E-02;
    COFD[2396] = -1.72572042E+01;
    COFD[2397] = 4.26063341E+00;
    COFD[2398] = -3.39848064E-01;
    COFD[2399] = 1.48021313E-02;
    COFD[2400] = -1.95613899E+01;
    COFD[2401] = 4.97133070E+00;
    COFD[2402] = -4.25604177E-01;
    COFD[2403] = 1.82582594E-02;
    COFD[2404] = -2.11378465E+01;
    COFD[2405] = 5.42846112E+00;
    COFD[2406] = -4.74321870E-01;
    COFD[2407] = 1.99459749E-02;
    COFD[2408] = -2.02434438E+01;
    COFD[2409] = 5.14418672E+00;
    COFD[2410] = -4.45631004E-01;
    COFD[2411] = 1.90308403E-02;
    COFD[2412] = -1.90996795E+01;
    COFD[2413] = 4.82869066E+00;
    COFD[2414] = -4.08564514E-01;
    COFD[2415] = 1.75784675E-02;
    COFD[2416] = -2.15326361E+01;
    COFD[2417] = 5.51982454E+00;
    COFD[2418] = -4.84452039E-01;
    COFD[2419] = 2.03175522E-02;
    COFD[2420] = -2.11378465E+01;
    COFD[2421] = 5.42846112E+00;
    COFD[2422] = -4.74321870E-01;
    COFD[2423] = 1.99459749E-02;
    COFD[2424] = -2.15647464E+01;
    COFD[2425] = 5.51982454E+00;
    COFD[2426] = -4.84452039E-01;
    COFD[2427] = 2.03175522E-02;
    COFD[2428] = -1.88826663E+01;
    COFD[2429] = 4.68393046E+00;
    COFD[2430] = -3.91610863E-01;
    COFD[2431] = 1.69174645E-02;
    COFD[2432] = -2.04021549E+01;
    COFD[2433] = 5.18271974E+00;
    COFD[2434] = -4.49323627E-01;
    COFD[2435] = 1.91373940E-02;
    COFD[2436] = -2.04054899E+01;
    COFD[2437] = 5.18271974E+00;
    COFD[2438] = -4.49323627E-01;
    COFD[2439] = 1.91373940E-02;
    COFD[2440] = -2.03111230E+01;
    COFD[2441] = 5.15740122E+00;
    COFD[2442] = -4.46644818E-01;
    COFD[2443] = 1.90459001E-02;
    COFD[2444] = -2.03711787E+01;
    COFD[2445] = 5.13279789E+00;
    COFD[2446] = -4.44474174E-01;
    COFD[2447] = 1.89937678E-02;
    COFD[2448] = -2.17867314E+01;
    COFD[2449] = 5.53950393E+00;
    COFD[2450] = -4.86376204E-01;
    COFD[2451] = 2.03760106E-02;
    COFD[2452] = -2.09217020E+01;
    COFD[2453] = 5.31360223E+00;
    COFD[2454] = -4.64787000E-01;
    COFD[2455] = 1.97483720E-02;
    COFD[2456] = -2.08833669E+01;
    COFD[2457] = 5.30526648E+00;
    COFD[2458] = -4.63785596E-01;
    COFD[2459] = 1.97079873E-02;
    COFD[2460] = -2.08851929E+01;
    COFD[2461] = 5.30526648E+00;
    COFD[2462] = -4.63785596E-01;
    COFD[2463] = 1.97079873E-02;
    COFD[2464] = -2.09847776E+01;
    COFD[2465] = 5.29210705E+00;
    COFD[2466] = -4.62193217E-01;
    COFD[2467] = 1.96432872E-02;
    COFD[2468] = -2.19250377E+01;
    COFD[2469] = 5.56282156E+00;
    COFD[2470] = -4.88585679E-01;
    COFD[2471] = 2.04395879E-02;
    COFD[2472] = -2.13616804E+01;
    COFD[2473] = 5.38519776E+00;
    COFD[2474] = -4.71344997E-01;
    COFD[2475] = 1.99226932E-02;
    COFD[2476] = -2.13171682E+01;
    COFD[2477] = 5.37197338E+00;
    COFD[2478] = -4.70392872E-01;
    COFD[2479] = 1.99122802E-02;
    COFD[2480] = -2.14920449E+01;
    COFD[2481] = 5.44385051E+00;
    COFD[2482] = -4.76121506E-01;
    COFD[2483] = 2.00164081E-02;
    COFD[2484] = -2.18960800E+01;
    COFD[2485] = 5.54768472E+00;
    COFD[2486] = -4.87202065E-01;
    COFD[2487] = 2.04025437E-02;
    COFD[2488] = -2.19248250E+01;
    COFD[2489] = 5.49350509E+00;
    COFD[2490] = -4.81613405E-01;
    COFD[2491] = 2.02171734E-02;
    COFD[2492] = -2.19254485E+01;
    COFD[2493] = 5.49350509E+00;
    COFD[2494] = -4.81613405E-01;
    COFD[2495] = 2.02171734E-02;
    COFD[2496] = -1.81432461E+01;
    COFD[2497] = 4.37565431E+00;
    COFD[2498] = -3.53906025E-01;
    COFD[2499] = 1.53760786E-02;
    COFD[2500] = -1.70534856E+01;
    COFD[2501] = 4.14240922E+00;
    COFD[2502] = -3.25239774E-01;
    COFD[2503] = 1.41980687E-02;
    COFD[2504] = -1.37794315E+01;
    COFD[2505] = 3.23973858E+00;
    COFD[2506] = -2.09989036E-01;
    COFD[2507] = 9.27667906E-03;
    COFD[2508] = -1.76147026E+01;
    COFD[2509] = 4.86049500E+00;
    COFD[2510] = -4.12200578E-01;
    COFD[2511] = 1.77160971E-02;
    COFD[2512] = -1.70757047E+01;
    COFD[2513] = 4.14240922E+00;
    COFD[2514] = -3.25239774E-01;
    COFD[2515] = 1.41980687E-02;
    COFD[2516] = -2.07653719E+01;
    COFD[2517] = 5.01092022E+00;
    COFD[2518] = -3.77985635E-01;
    COFD[2519] = 1.40968645E-02;
    COFD[2520] = -1.84688406E+01;
    COFD[2521] = 4.49330851E+00;
    COFD[2522] = -3.68208715E-01;
    COFD[2523] = 1.59565402E-02;
    COFD[2524] = -1.84777607E+01;
    COFD[2525] = 4.49330851E+00;
    COFD[2526] = -3.68208715E-01;
    COFD[2527] = 1.59565402E-02;
    COFD[2528] = -1.84863000E+01;
    COFD[2529] = 4.49330851E+00;
    COFD[2530] = -3.68208715E-01;
    COFD[2531] = 1.59565402E-02;
    COFD[2532] = -1.69758891E+01;
    COFD[2533] = 4.14240922E+00;
    COFD[2534] = -3.25239774E-01;
    COFD[2535] = 1.41980687E-02;
    COFD[2536] = -2.19215555E+01;
    COFD[2537] = 5.45216133E+00;
    COFD[2538] = -4.52916925E-01;
    COFD[2539] = 1.80456400E-02;
    COFD[2540] = -1.93015555E+01;
    COFD[2541] = 4.85015581E+00;
    COFD[2542] = -4.10945109E-01;
    COFD[2543] = 1.76651398E-02;
    COFD[2544] = -2.13425698E+01;
    COFD[2545] = 5.40460130E+00;
    COFD[2546] = -4.72718910E-01;
    COFD[2547] = 1.99362717E-02;
    COFD[2548] = -1.81735763E+01;
    COFD[2549] = 4.38391495E+00;
    COFD[2550] = -3.54941287E-01;
    COFD[2551] = 1.54195107E-02;
    COFD[2552] = -2.19317743E+01;
    COFD[2553] = 5.45216133E+00;
    COFD[2554] = -4.52916925E-01;
    COFD[2555] = 1.80456400E-02;
    COFD[2556] = -1.93015555E+01;
    COFD[2557] = 4.85015581E+00;
    COFD[2558] = -4.10945109E-01;
    COFD[2559] = 1.76651398E-02;
    COFD[2560] = -1.93276434E+01;
    COFD[2561] = 4.85015581E+00;
    COFD[2562] = -4.10945109E-01;
    COFD[2563] = 1.76651398E-02;
    COFD[2564] = -2.20421041E+01;
    COFD[2565] = 5.52708332E+00;
    COFD[2566] = -4.68000808E-01;
    COFD[2567] = 1.89131908E-02;
    COFD[2568] = -1.92867554E+01;
    COFD[2569] = 4.83375900E+00;
    COFD[2570] = -4.09146560E-01;
    COFD[2571] = 1.76006599E-02;
    COFD[2572] = -2.20063594E+01;
    COFD[2573] = 5.48540187E+00;
    COFD[2574] = -4.58962148E-01;
    COFD[2575] = 1.83770355E-02;
    COFD[2576] = -2.14151520E+01;
    COFD[2577] = 5.41122754E+00;
    COFD[2578] = -4.73185889E-01;
    COFD[2579] = 1.99407905E-02;
    COFD[2580] = -2.14049543E+01;
    COFD[2581] = 5.41122754E+00;
    COFD[2582] = -4.73185889E-01;
    COFD[2583] = 1.99407905E-02;
    COFD[2584] = -2.22116706E+01;
    COFD[2585] = 5.54251230E+00;
    COFD[2586] = -4.70946314E-01;
    COFD[2587] = 1.90785869E-02;
    COFD[2588] = -2.22343363E+01;
    COFD[2589] = 5.54251230E+00;
    COFD[2590] = -4.70946314E-01;
    COFD[2591] = 1.90785869E-02;
    COFD[2592] = -2.21083035E+01;
    COFD[2593] = 5.48540187E+00;
    COFD[2594] = -4.58962148E-01;
    COFD[2595] = 1.83770355E-02;
    COFD[2596] = -2.21134005E+01;
    COFD[2597] = 5.48540187E+00;
    COFD[2598] = -4.58962148E-01;
    COFD[2599] = 1.83770355E-02;
    COFD[2600] = -2.13961414E+01;
    COFD[2601] = 5.46685775E+00;
    COFD[2602] = -4.78665416E-01;
    COFD[2603] = 2.01093915E-02;
    COFD[2604] = -1.94485982E+01;
    COFD[2605] = 4.91446566E+00;
    COFD[2606] = -4.18837152E-01;
    COFD[2607] = 1.79893537E-02;
    COFD[2608] = -2.14079882E+01;
    COFD[2609] = 5.46685775E+00;
    COFD[2610] = -4.78665416E-01;
    COFD[2611] = 2.01093915E-02;
    COFD[2612] = -2.22176950E+01;
    COFD[2613] = 5.54251230E+00;
    COFD[2614] = -4.70946314E-01;
    COFD[2615] = 1.90785869E-02;
    COFD[2616] = -2.20725883E+01;
    COFD[2617] = 5.59642965E+00;
    COFD[2618] = -4.91577716E-01;
    COFD[2619] = 2.05159582E-02;
    COFD[2620] = -2.11031143E+01;
    COFD[2621] = 5.39439999E+00;
    COFD[2622] = -4.72050184E-01;
    COFD[2623] = 1.99336257E-02;
    COFD[2624] = -2.23098172E+01;
    COFD[2625] = 5.49916900E+00;
    COFD[2626] = -4.61818485E-01;
    COFD[2627] = 1.85431163E-02;
    COFD[2628] = -2.22176950E+01;
    COFD[2629] = 5.54251230E+00;
    COFD[2630] = -4.70946314E-01;
    COFD[2631] = 1.90785869E-02;
    COFD[2632] = -2.23791409E+01;
    COFD[2633] = 5.49916900E+00;
    COFD[2634] = -4.61818485E-01;
    COFD[2635] = 1.85431163E-02;
    COFD[2636] = -2.10296583E+01;
    COFD[2637] = 5.30153901E+00;
    COFD[2638] = -4.63335119E-01;
    COFD[2639] = 1.96897053E-02;
    COFD[2640] = -2.21630311E+01;
    COFD[2641] = 5.60807471E+00;
    COFD[2642] = -4.91339309E-01;
    COFD[2643] = 2.04365761E-02;
    COFD[2644] = -2.21697404E+01;
    COFD[2645] = 5.60807471E+00;
    COFD[2646] = -4.91339309E-01;
    COFD[2647] = 2.04365761E-02;
    COFD[2648] = -2.21216828E+01;
    COFD[2649] = 5.60203389E+00;
    COFD[2650] = -4.91444416E-01;
    COFD[2651] = 2.04761886E-02;
    COFD[2652] = -2.22052004E+01;
    COFD[2653] = 5.58604166E+00;
    COFD[2654] = -4.90602184E-01;
    COFD[2655] = 2.04880352E-02;
    COFD[2656] = -2.25168081E+01;
    COFD[2657] = 5.46125558E+00;
    COFD[2658] = -4.54580949E-01;
    COFD[2659] = 1.81370928E-02;
    COFD[2660] = -2.23890317E+01;
    COFD[2661] = 5.59178974E+00;
    COFD[2662] = -4.85668031E-01;
    COFD[2663] = 2.00491907E-02;
    COFD[2664] = -2.23772680E+01;
    COFD[2665] = 5.59425354E+00;
    COFD[2666] = -4.86232980E-01;
    COFD[2667] = 2.00835981E-02;
    COFD[2668] = -2.23812726E+01;
    COFD[2669] = 5.59425354E+00;
    COFD[2670] = -4.86232980E-01;
    COFD[2671] = 2.00835981E-02;
    COFD[2672] = -2.25216613E+01;
    COFD[2673] = 5.59792043E+00;
    COFD[2674] = -4.87076900E-01;
    COFD[2675] = 2.01350364E-02;
    COFD[2676] = -2.25838099E+01;
    COFD[2677] = 5.45615714E+00;
    COFD[2678] = -4.53643844E-01;
    COFD[2679] = 1.80854821E-02;
    COFD[2680] = -2.26897188E+01;
    COFD[2681] = 5.58518389E+00;
    COFD[2682] = -4.80570209E-01;
    COFD[2683] = 1.96586179E-02;
    COFD[2684] = -2.26749993E+01;
    COFD[2685] = 5.58486459E+00;
    COFD[2686] = -4.81517134E-01;
    COFD[2687] = 1.97388064E-02;
    COFD[2688] = -2.25786655E+01;
    COFD[2689] = 5.53409384E+00;
    COFD[2690] = -4.69342499E-01;
    COFD[2691] = 1.89886374E-02;
    COFD[2692] = -2.26305728E+01;
    COFD[2693] = 5.47666967E+00;
    COFD[2694] = -4.57381900E-01;
    COFD[2695] = 1.82905822E-02;
    COFD[2696] = -2.28655752E+01;
    COFD[2697] = 5.50522401E+00;
    COFD[2698] = -4.63604304E-01;
    COFD[2699] = 1.86600785E-02;
    COFD[2700] = -2.28671232E+01;
    COFD[2701] = 5.50522401E+00;
    COFD[2702] = -4.63604304E-01;
    COFD[2703] = 1.86600785E-02;
    COFD[2704] = -1.50031687E+01;
    COFD[2705] = 3.26223357E+00;
    COFD[2706] = -2.12746642E-01;
    COFD[2707] = 9.38912883E-03;
    COFD[2708] = -1.40999008E+01;
    COFD[2709] = 3.08120012E+00;
    COFD[2710] = -1.89629903E-01;
    COFD[2711] = 8.40361952E-03;
    COFD[2712] = -1.17159737E+01;
    COFD[2713] = 2.48123210E+00;
    COFD[2714] = -1.11322604E-01;
    COFD[2715] = 4.99282389E-03;
    COFD[2716] = -1.43151174E+01;
    COFD[2717] = 3.68038508E+00;
    COFD[2718] = -2.65779346E-01;
    COFD[2719] = 1.16360771E-02;
    COFD[2720] = -1.41191261E+01;
    COFD[2721] = 3.08120012E+00;
    COFD[2722] = -1.89629903E-01;
    COFD[2723] = 8.40361952E-03;
    COFD[2724] = -2.11388331E+01;
    COFD[2725] = 5.55529675E+00;
    COFD[2726] = -4.87942518E-01;
    COFD[2727] = 2.04249054E-02;
    COFD[2728] = -1.52721107E+01;
    COFD[2729] = 3.36790500E+00;
    COFD[2730] = -2.26321740E-01;
    COFD[2731] = 9.97135055E-03;
    COFD[2732] = -1.52792891E+01;
    COFD[2733] = 3.36790500E+00;
    COFD[2734] = -2.26321740E-01;
    COFD[2735] = 9.97135055E-03;
    COFD[2736] = -1.52861376E+01;
    COFD[2737] = 3.36790500E+00;
    COFD[2738] = -2.26321740E-01;
    COFD[2739] = 9.97135055E-03;
    COFD[2740] = -1.40318948E+01;
    COFD[2741] = 3.08120012E+00;
    COFD[2742] = -1.89629903E-01;
    COFD[2743] = 8.40361952E-03;
    COFD[2744] = -2.05045578E+01;
    COFD[2745] = 5.23843909E+00;
    COFD[2746] = -4.55815614E-01;
    COFD[2747] = 1.93898040E-02;
    COFD[2748] = -1.59634533E+01;
    COFD[2749] = 3.67388294E+00;
    COFD[2750] = -2.64990709E-01;
    COFD[2751] = 1.16042706E-02;
    COFD[2752] = -1.81735763E+01;
    COFD[2753] = 4.38391495E+00;
    COFD[2754] = -3.54941287E-01;
    COFD[2755] = 1.54195107E-02;
    COFD[2756] = -1.50233475E+01;
    COFD[2757] = 3.26660767E+00;
    COFD[2758] = -2.13287177E-01;
    COFD[2759] = 9.41137857E-03;
    COFD[2760] = -2.05128705E+01;
    COFD[2761] = 5.23843909E+00;
    COFD[2762] = -4.55815614E-01;
    COFD[2763] = 1.93898040E-02;
    COFD[2764] = -1.59634533E+01;
    COFD[2765] = 3.67388294E+00;
    COFD[2766] = -2.64990709E-01;
    COFD[2767] = 1.16042706E-02;
    COFD[2768] = -1.59863030E+01;
    COFD[2769] = 3.67388294E+00;
    COFD[2770] = -2.64990709E-01;
    COFD[2771] = 1.16042706E-02;
    COFD[2772] = -2.02642227E+01;
    COFD[2773] = 5.14499740E+00;
    COFD[2774] = -4.45694430E-01;
    COFD[2775] = 1.90318646E-02;
    COFD[2776] = -1.59525102E+01;
    COFD[2777] = 3.66023858E+00;
    COFD[2778] = -2.63401043E-01;
    COFD[2779] = 1.15432000E-02;
    COFD[2780] = -2.04144604E+01;
    COFD[2781] = 5.19614628E+00;
    COFD[2782] = -4.50889164E-01;
    COFD[2783] = 1.91983328E-02;
    COFD[2784] = -1.82955252E+01;
    COFD[2785] = 4.40289649E+00;
    COFD[2786] = -3.57289765E-01;
    COFD[2787] = 1.55166804E-02;
    COFD[2788] = -1.82872310E+01;
    COFD[2789] = 4.40289649E+00;
    COFD[2790] = -3.57289765E-01;
    COFD[2791] = 1.55166804E-02;
    COFD[2792] = -2.02922701E+01;
    COFD[2793] = 5.11106992E+00;
    COFD[2794] = -4.42047129E-01;
    COFD[2795] = 1.89042990E-02;
    COFD[2796] = -2.03099025E+01;
    COFD[2797] = 5.11106992E+00;
    COFD[2798] = -4.42047129E-01;
    COFD[2799] = 1.89042990E-02;
    COFD[2800] = -2.04949373E+01;
    COFD[2801] = 5.19614628E+00;
    COFD[2802] = -4.50889164E-01;
    COFD[2803] = 1.91983328E-02;
    COFD[2804] = -2.04988684E+01;
    COFD[2805] = 5.19614628E+00;
    COFD[2806] = -4.50889164E-01;
    COFD[2807] = 1.91983328E-02;
    COFD[2808] = -1.83296965E+01;
    COFD[2809] = 4.48570999E+00;
    COFD[2810] = -3.67301524E-01;
    COFD[2811] = 1.59204254E-02;
    COFD[2812] = -1.60261675E+01;
    COFD[2813] = 3.73312045E+00;
    COFD[2814] = -2.72579779E-01;
    COFD[2815] = 1.19290272E-02;
    COFD[2816] = -1.83394481E+01;
    COFD[2817] = 4.48570999E+00;
    COFD[2818] = -3.67301524E-01;
    COFD[2819] = 1.59204254E-02;
    COFD[2820] = -2.02969740E+01;
    COFD[2821] = 5.11106992E+00;
    COFD[2822] = -4.42047129E-01;
    COFD[2823] = 1.89042990E-02;
    COFD[2824] = -1.91118445E+01;
    COFD[2825] = 4.68715685E+00;
    COFD[2826] = -3.91979493E-01;
    COFD[2827] = 1.69314004E-02;
    COFD[2828] = -1.79116531E+01;
    COFD[2829] = 4.35148286E+00;
    COFD[2830] = -3.50886647E-01;
    COFD[2831] = 1.52498573E-02;
    COFD[2832] = -2.06066440E+01;
    COFD[2833] = 5.16748146E+00;
    COFD[2834] = -4.47594939E-01;
    COFD[2835] = 1.90724110E-02;
    COFD[2836] = -2.02969740E+01;
    COFD[2837] = 5.11106992E+00;
    COFD[2838] = -4.42047129E-01;
    COFD[2839] = 1.89042990E-02;
    COFD[2840] = -2.06595692E+01;
    COFD[2841] = 5.16748146E+00;
    COFD[2842] = -4.47594939E-01;
    COFD[2843] = 1.90724110E-02;
    COFD[2844] = -1.77178857E+01;
    COFD[2845] = 4.19935698E+00;
    COFD[2846] = -3.32310212E-01;
    COFD[2847] = 1.44920670E-02;
    COFD[2848] = -1.93011401E+01;
    COFD[2849] = 4.74387793E+00;
    COFD[2850] = -3.98574972E-01;
    COFD[2851] = 1.71862289E-02;
    COFD[2852] = -1.93064215E+01;
    COFD[2853] = 4.74387793E+00;
    COFD[2854] = -3.98574972E-01;
    COFD[2855] = 1.71862289E-02;
    COFD[2856] = -1.92044492E+01;
    COFD[2857] = 4.71304783E+00;
    COFD[2858] = -3.94942083E-01;
    COFD[2859] = 1.70435959E-02;
    COFD[2860] = -1.92334028E+01;
    COFD[2861] = 4.67033934E+00;
    COFD[2862] = -3.89971551E-01;
    COFD[2863] = 1.68513441E-02;
    COFD[2864] = -2.10310742E+01;
    COFD[2865] = 5.23485505E+00;
    COFD[2866] = -4.55400362E-01;
    COFD[2867] = 1.93737680E-02;
    COFD[2868] = -1.97709603E+01;
    COFD[2869] = 4.84731557E+00;
    COFD[2870] = -4.10638352E-01;
    COFD[2871] = 1.76543886E-02;
    COFD[2872] = -1.97422209E+01;
    COFD[2873] = 4.84249900E+00;
    COFD[2874] = -4.10120448E-01;
    COFD[2875] = 1.76363500E-02;
    COFD[2876] = -1.97452574E+01;
    COFD[2877] = 4.84249900E+00;
    COFD[2878] = -4.10120448E-01;
    COFD[2879] = 1.76363500E-02;
    COFD[2880] = -1.98616115E+01;
    COFD[2881] = 4.83466791E+00;
    COFD[2882] = -4.09252052E-01;
    COFD[2883] = 1.76047341E-02;
    COFD[2884] = -2.10924694E+01;
    COFD[2885] = 5.23339224E+00;
    COFD[2886] = -4.55230780E-01;
    COFD[2887] = 1.93672146E-02;
    COFD[2888] = -2.03988322E+01;
    COFD[2889] = 4.99562188E+00;
    COFD[2890] = -4.28482025E-01;
    COFD[2891] = 1.83720948E-02;
    COFD[2892] = -2.03122895E+01;
    COFD[2893] = 4.96244824E+00;
    COFD[2894] = -4.24554494E-01;
    COFD[2895] = 1.82168885E-02;
    COFD[2896] = -2.06812067E+01;
    COFD[2897] = 5.12346096E+00;
    COFD[2898] = -4.43477411E-01;
    COFD[2899] = 1.89592529E-02;
    COFD[2900] = -2.10372026E+01;
    COFD[2901] = 5.20711052E+00;
    COFD[2902] = -4.52173945E-01;
    COFD[2903] = 1.92486273E-02;
    COFD[2904] = -2.10844012E+01;
    COFD[2905] = 5.15315713E+00;
    COFD[2906] = -4.46344043E-01;
    COFD[2907] = 1.90431546E-02;
    COFD[2908] = -2.10855099E+01;
    COFD[2909] = 5.15315713E+00;
    COFD[2910] = -4.46344043E-01;
    COFD[2911] = 1.90431546E-02;
    COFD[2912] = -2.04833713E+01;
    COFD[2913] = 5.23112374E+00;
    COFD[2914] = -4.54967682E-01;
    COFD[2915] = 1.93570423E-02;
    COFD[2916] = -1.94373127E+01;
    COFD[2917] = 5.02567894E+00;
    COFD[2918] = -4.32045169E-01;
    COFD[2919] = 1.85132214E-02;
    COFD[2920] = -1.60528285E+01;
    COFD[2921] = 4.11188603E+00;
    COFD[2922] = -3.21540884E-01;
    COFD[2923] = 1.40482564E-02;
    COFD[2924] = -1.97550088E+01;
    COFD[2925] = 5.56931926E+00;
    COFD[2926] = -4.89105511E-01;
    COFD[2927] = 2.04493129E-02;
    COFD[2928] = -1.94570287E+01;
    COFD[2929] = 5.02567894E+00;
    COFD[2930] = -4.32045169E-01;
    COFD[2931] = 1.85132214E-02;
    COFD[2932] = -1.77563250E+01;
    COFD[2933] = 3.57475686E+00;
    COFD[2934] = -1.56396297E-01;
    COFD[2935] = 3.12157721E-03;
    COFD[2936] = -2.08293255E+01;
    COFD[2937] = 5.35267674E+00;
    COFD[2938] = -4.69010505E-01;
    COFD[2939] = 1.98979152E-02;
    COFD[2940] = -2.08367725E+01;
    COFD[2941] = 5.35267674E+00;
    COFD[2942] = -4.69010505E-01;
    COFD[2943] = 1.98979152E-02;
    COFD[2944] = -2.08438809E+01;
    COFD[2945] = 5.35267674E+00;
    COFD[2946] = -4.69010505E-01;
    COFD[2947] = 1.98979152E-02;
    COFD[2948] = -1.93677186E+01;
    COFD[2949] = 5.02567894E+00;
    COFD[2950] = -4.32045169E-01;
    COFD[2951] = 1.85132214E-02;
    COFD[2952] = -1.90413348E+01;
    COFD[2953] = 3.99221757E+00;
    COFD[2954] = -2.19854880E-01;
    COFD[2955] = 6.22736279E-03;
    COFD[2956] = -2.14215700E+01;
    COFD[2957] = 5.56531152E+00;
    COFD[2958] = -4.88789821E-01;
    COFD[2959] = 2.04437116E-02;
    COFD[2960] = -2.19317743E+01;
    COFD[2961] = 5.45216133E+00;
    COFD[2962] = -4.52916925E-01;
    COFD[2963] = 1.80456400E-02;
    COFD[2964] = -2.05128705E+01;
    COFD[2965] = 5.23843909E+00;
    COFD[2966] = -4.55815614E-01;
    COFD[2967] = 1.93898040E-02;
    COFD[2968] = -1.90499441E+01;
    COFD[2969] = 3.99221757E+00;
    COFD[2970] = -2.19854880E-01;
    COFD[2971] = 6.22736279E-03;
    COFD[2972] = -2.14215700E+01;
    COFD[2973] = 5.56531152E+00;
    COFD[2974] = -4.88789821E-01;
    COFD[2975] = 2.04437116E-02;
    COFD[2976] = -2.14449559E+01;
    COFD[2977] = 5.56531152E+00;
    COFD[2978] = -4.88789821E-01;
    COFD[2979] = 2.04437116E-02;
    COFD[2980] = -2.01889168E+01;
    COFD[2981] = 4.53183330E+00;
    COFD[2982] = -3.02186760E-01;
    COFD[2983] = 1.02756490E-02;
    COFD[2984] = -2.14082453E+01;
    COFD[2985] = 5.55346617E+00;
    COFD[2986] = -4.87783156E-01;
    COFD[2987] = 2.04210886E-02;
    COFD[2988] = -1.93214527E+01;
    COFD[2989] = 4.10954793E+00;
    COFD[2990] = -2.37523329E-01;
    COFD[2991] = 7.08858141E-03;
    COFD[2992] = -2.19786173E+01;
    COFD[2993] = 5.43750833E+00;
    COFD[2994] = -4.50273329E-01;
    COFD[2995] = 1.79013718E-02;
    COFD[2996] = -2.19700270E+01;
    COFD[2997] = 5.43750833E+00;
    COFD[2998] = -4.50273329E-01;
    COFD[2999] = 1.79013718E-02;
    COFD[3000] = -2.01015340E+01;
    COFD[3001] = 4.41511629E+00;
    COFD[3002] = -2.84086963E-01;
    COFD[3003] = 9.37586971E-03;
    COFD[3004] = -2.01199204E+01;
    COFD[3005] = 4.41511629E+00;
    COFD[3006] = -2.84086963E-01;
    COFD[3007] = 9.37586971E-03;
    COFD[3008] = -1.94051843E+01;
    COFD[3009] = 4.10954793E+00;
    COFD[3010] = -2.37523329E-01;
    COFD[3011] = 7.08858141E-03;
    COFD[3012] = -1.94092888E+01;
    COFD[3013] = 4.10954793E+00;
    COFD[3014] = -2.37523329E-01;
    COFD[3015] = 7.08858141E-03;
    COFD[3016] = -2.16798265E+01;
    COFD[3017] = 5.36811769E+00;
    COFD[3018] = -4.37727086E-01;
    COFD[3019] = 1.72167686E-02;
    COFD[3020] = -2.14303479E+01;
    COFD[3021] = 5.59268435E+00;
    COFD[3022] = -4.91232974E-01;
    COFD[3023] = 2.05064746E-02;
    COFD[3024] = -2.16899073E+01;
    COFD[3025] = 5.36811769E+00;
    COFD[3026] = -4.37727086E-01;
    COFD[3027] = 1.72167686E-02;
    COFD[3028] = -2.01064363E+01;
    COFD[3029] = 4.41511629E+00;
    COFD[3030] = -2.84086963E-01;
    COFD[3031] = 9.37586971E-03;
    COFD[3032] = -2.15802788E+01;
    COFD[3033] = 5.16868516E+00;
    COFD[3034] = -4.03721581E-01;
    COFD[3035] = 1.54206640E-02;
    COFD[3036] = -2.17855148E+01;
    COFD[3037] = 5.47519298E+00;
    COFD[3038] = -4.57113040E-01;
    COFD[3039] = 1.82758312E-02;
    COFD[3040] = -1.97649065E+01;
    COFD[3041] = 4.18758010E+00;
    COFD[3042] = -2.49327776E-01;
    COFD[3043] = 7.66559103E-03;
    COFD[3044] = -2.01064363E+01;
    COFD[3045] = 4.41511629E+00;
    COFD[3046] = -2.84086963E-01;
    COFD[3047] = 9.37586971E-03;
    COFD[3048] = -1.98202487E+01;
    COFD[3049] = 4.18758010E+00;
    COFD[3050] = -2.49327776E-01;
    COFD[3051] = 7.66559103E-03;
    COFD[3052] = -2.20962383E+01;
    COFD[3053] = 5.55935694E+00;
    COFD[3054] = -4.74154740E-01;
    COFD[3055] = 1.92584304E-02;
    COFD[3056] = -2.14398182E+01;
    COFD[3057] = 5.07680397E+00;
    COFD[3058] = -3.88612087E-01;
    COFD[3059] = 1.46395101E-02;
    COFD[3060] = -2.14453157E+01;
    COFD[3061] = 5.07680397E+00;
    COFD[3062] = -3.88612087E-01;
    COFD[3063] = 1.46395101E-02;
    COFD[3064] = -2.15258568E+01;
    COFD[3065] = 5.12799307E+00;
    COFD[3066] = -3.96938732E-01;
    COFD[3067] = 1.50673195E-02;
    COFD[3068] = -2.17926864E+01;
    COFD[3069] = 5.19232842E+00;
    COFD[3070] = -4.07643284E-01;
    COFD[3071] = 1.56246434E-02;
    COFD[3072] = -1.98359760E+01;
    COFD[3073] = 4.11158627E+00;
    COFD[3074] = -2.37831519E-01;
    COFD[3075] = 7.10363413E-03;
    COFD[3076] = -2.12219677E+01;
    COFD[3077] = 4.87252053E+00;
    COFD[3078] = -3.56127804E-01;
    COFD[3079] = 1.29948788E-02;
    COFD[3080] = -2.12330900E+01;
    COFD[3081] = 4.88535789E+00;
    COFD[3082] = -3.58153894E-01;
    COFD[3083] = 1.30969624E-02;
    COFD[3084] = -2.12362684E+01;
    COFD[3085] = 4.88535789E+00;
    COFD[3086] = -3.58153894E-01;
    COFD[3087] = 1.30969624E-02;
    COFD[3088] = -2.14142864E+01;
    COFD[3089] = 4.90439970E+00;
    COFD[3090] = -3.61162615E-01;
    COFD[3091] = 1.32486109E-02;
    COFD[3092] = -1.96860113E+01;
    COFD[3093] = 4.00653795E+00;
    COFD[3094] = -2.22005804E-01;
    COFD[3095] = 6.33194910E-03;
    COFD[3096] = -2.09922023E+01;
    COFD[3097] = 4.64167142E+00;
    COFD[3098] = -3.19532110E-01;
    COFD[3099] = 1.11478359E-02;
    COFD[3100] = -2.11192145E+01;
    COFD[3101] = 4.70311989E+00;
    COFD[3102] = -3.29240106E-01;
    COFD[3103] = 1.16366808E-02;
    COFD[3104] = -2.03970537E+01;
    COFD[3105] = 4.38396848E+00;
    COFD[3106] = -2.79298901E-01;
    COFD[3107] = 9.13915001E-03;
    COFD[3108] = -1.98603655E+01;
    COFD[3109] = 4.07958166E+00;
    COFD[3110] = -2.33006871E-01;
    COFD[3111] = 6.86822015E-03;
    COFD[3112] = -2.04620510E+01;
    COFD[3113] = 4.26473557E+00;
    COFD[3114] = -2.61033037E-01;
    COFD[3115] = 8.23906412E-03;
    COFD[3116] = -2.04632210E+01;
    COFD[3117] = 4.26473557E+00;
    COFD[3118] = -2.61033037E-01;
    COFD[3119] = 8.23906412E-03;
    COFD[3120] = -1.59404882E+01;
    COFD[3121] = 3.66853818E+00;
    COFD[3122] = -2.64346221E-01;
    COFD[3123] = 1.15784613E-02;
    COFD[3124] = -1.50584249E+01;
    COFD[3125] = 3.47945612E+00;
    COFD[3126] = -2.40703722E-01;
    COFD[3127] = 1.05907441E-02;
    COFD[3128] = -1.25098960E+01;
    COFD[3129] = 2.77873601E+00;
    COFD[3130] = -1.50637360E-01;
    COFD[3131] = 6.72684281E-03;
    COFD[3132] = -1.57972369E+01;
    COFD[3133] = 4.22225052E+00;
    COFD[3134] = -3.35156428E-01;
    COFD[3135] = 1.46104855E-02;
    COFD[3136] = -1.50724636E+01;
    COFD[3137] = 3.47945612E+00;
    COFD[3138] = -2.40703722E-01;
    COFD[3139] = 1.05907441E-02;
    COFD[3140] = -2.12639214E+01;
    COFD[3141] = 5.61184117E+00;
    COFD[3142] = -4.90532156E-01;
    COFD[3143] = 2.03507922E-02;
    COFD[3144] = -1.63254691E+01;
    COFD[3145] = 3.82388595E+00;
    COFD[3146] = -2.84480724E-01;
    COFD[3147] = 1.24506311E-02;
    COFD[3148] = -1.63301444E+01;
    COFD[3149] = 3.82388595E+00;
    COFD[3150] = -2.84480724E-01;
    COFD[3151] = 1.24506311E-02;
    COFD[3152] = -1.63345829E+01;
    COFD[3153] = 3.82388595E+00;
    COFD[3154] = -2.84480724E-01;
    COFD[3155] = 1.24506311E-02;
    COFD[3156] = -1.50076254E+01;
    COFD[3157] = 3.47945612E+00;
    COFD[3158] = -2.40703722E-01;
    COFD[3159] = 1.05907441E-02;
    COFD[3160] = -2.14160703E+01;
    COFD[3161] = 5.56531152E+00;
    COFD[3162] = -4.88789821E-01;
    COFD[3163] = 2.04437116E-02;
    COFD[3164] = -1.73027557E+01;
    COFD[3165] = 4.21416723E+00;
    COFD[3166] = -3.34163932E-01;
    COFD[3167] = 1.45697432E-02;
    COFD[3168] = -1.93015555E+01;
    COFD[3169] = 4.85015581E+00;
    COFD[3170] = -4.10945109E-01;
    COFD[3171] = 1.76651398E-02;
    COFD[3172] = -1.59634533E+01;
    COFD[3173] = 3.67388294E+00;
    COFD[3174] = -2.64990709E-01;
    COFD[3175] = 1.16042706E-02;
    COFD[3176] = -2.14215700E+01;
    COFD[3177] = 5.56531152E+00;
    COFD[3178] = -4.88789821E-01;
    COFD[3179] = 2.04437116E-02;
    COFD[3180] = -1.73027557E+01;
    COFD[3181] = 4.21416723E+00;
    COFD[3182] = -3.34163932E-01;
    COFD[3183] = 1.45697432E-02;
    COFD[3184] = -1.73198034E+01;
    COFD[3185] = 4.21416723E+00;
    COFD[3186] = -3.34163932E-01;
    COFD[3187] = 1.45697432E-02;
    COFD[3188] = -2.09376196E+01;
    COFD[3189] = 5.40870099E+00;
    COFD[3190] = -4.73017610E-01;
    COFD[3191] = 1.99399066E-02;
    COFD[3192] = -1.72556729E+01;
    COFD[3193] = 4.19029808E+00;
    COFD[3194] = -3.31177076E-01;
    COFD[3195] = 1.44446234E-02;
    COFD[3196] = -2.13538553E+01;
    COFD[3197] = 5.54007827E+00;
    COFD[3198] = -4.86434511E-01;
    COFD[3199] = 2.03779006E-02;
    COFD[3200] = -1.94585111E+01;
    COFD[3201] = 4.87180830E+00;
    COFD[3202] = -4.13582958E-01;
    COFD[3203] = 1.77726094E-02;
    COFD[3204] = -1.94530250E+01;
    COFD[3205] = 4.87180830E+00;
    COFD[3206] = -4.13582958E-01;
    COFD[3207] = 1.77726094E-02;
    COFD[3208] = -2.11349086E+01;
    COFD[3209] = 5.42846112E+00;
    COFD[3210] = -4.74321870E-01;
    COFD[3211] = 1.99459749E-02;
    COFD[3212] = -2.11458678E+01;
    COFD[3213] = 5.42846112E+00;
    COFD[3214] = -4.74321870E-01;
    COFD[3215] = 1.99459749E-02;
    COFD[3216] = -2.14048982E+01;
    COFD[3217] = 5.54007827E+00;
    COFD[3218] = -4.86434511E-01;
    COFD[3219] = 2.03779006E-02;
    COFD[3220] = -2.14073140E+01;
    COFD[3221] = 5.54007827E+00;
    COFD[3222] = -4.86434511E-01;
    COFD[3223] = 2.03779006E-02;
    COFD[3224] = -1.95548230E+01;
    COFD[3225] = 4.97133070E+00;
    COFD[3226] = -4.25604177E-01;
    COFD[3227] = 1.82582594E-02;
    COFD[3228] = -1.72572042E+01;
    COFD[3229] = 4.26063341E+00;
    COFD[3230] = -3.39848064E-01;
    COFD[3231] = 1.48021313E-02;
    COFD[3232] = -1.95613899E+01;
    COFD[3233] = 4.97133070E+00;
    COFD[3234] = -4.25604177E-01;
    COFD[3235] = 1.82582594E-02;
    COFD[3236] = -2.11378465E+01;
    COFD[3237] = 5.42846112E+00;
    COFD[3238] = -4.74321870E-01;
    COFD[3239] = 1.99459749E-02;
    COFD[3240] = -2.02434438E+01;
    COFD[3241] = 5.14418672E+00;
    COFD[3242] = -4.45631004E-01;
    COFD[3243] = 1.90308403E-02;
    COFD[3244] = -1.90996795E+01;
    COFD[3245] = 4.82869066E+00;
    COFD[3246] = -4.08564514E-01;
    COFD[3247] = 1.75784675E-02;
    COFD[3248] = -2.15326361E+01;
    COFD[3249] = 5.51982454E+00;
    COFD[3250] = -4.84452039E-01;
    COFD[3251] = 2.03175522E-02;
    COFD[3252] = -2.11378465E+01;
    COFD[3253] = 5.42846112E+00;
    COFD[3254] = -4.74321870E-01;
    COFD[3255] = 1.99459749E-02;
    COFD[3256] = -2.15647464E+01;
    COFD[3257] = 5.51982454E+00;
    COFD[3258] = -4.84452039E-01;
    COFD[3259] = 2.03175522E-02;
    COFD[3260] = -1.88826663E+01;
    COFD[3261] = 4.68393046E+00;
    COFD[3262] = -3.91610863E-01;
    COFD[3263] = 1.69174645E-02;
    COFD[3264] = -2.04021549E+01;
    COFD[3265] = 5.18271974E+00;
    COFD[3266] = -4.49323627E-01;
    COFD[3267] = 1.91373940E-02;
    COFD[3268] = -2.04054899E+01;
    COFD[3269] = 5.18271974E+00;
    COFD[3270] = -4.49323627E-01;
    COFD[3271] = 1.91373940E-02;
    COFD[3272] = -2.03111230E+01;
    COFD[3273] = 5.15740122E+00;
    COFD[3274] = -4.46644818E-01;
    COFD[3275] = 1.90459001E-02;
    COFD[3276] = -2.03711787E+01;
    COFD[3277] = 5.13279789E+00;
    COFD[3278] = -4.44474174E-01;
    COFD[3279] = 1.89937678E-02;
    COFD[3280] = -2.17867314E+01;
    COFD[3281] = 5.53950393E+00;
    COFD[3282] = -4.86376204E-01;
    COFD[3283] = 2.03760106E-02;
    COFD[3284] = -2.09217020E+01;
    COFD[3285] = 5.31360223E+00;
    COFD[3286] = -4.64787000E-01;
    COFD[3287] = 1.97483720E-02;
    COFD[3288] = -2.08833669E+01;
    COFD[3289] = 5.30526648E+00;
    COFD[3290] = -4.63785596E-01;
    COFD[3291] = 1.97079873E-02;
    COFD[3292] = -2.08851929E+01;
    COFD[3293] = 5.30526648E+00;
    COFD[3294] = -4.63785596E-01;
    COFD[3295] = 1.97079873E-02;
    COFD[3296] = -2.09847776E+01;
    COFD[3297] = 5.29210705E+00;
    COFD[3298] = -4.62193217E-01;
    COFD[3299] = 1.96432872E-02;
    COFD[3300] = -2.19250377E+01;
    COFD[3301] = 5.56282156E+00;
    COFD[3302] = -4.88585679E-01;
    COFD[3303] = 2.04395879E-02;
    COFD[3304] = -2.13616804E+01;
    COFD[3305] = 5.38519776E+00;
    COFD[3306] = -4.71344997E-01;
    COFD[3307] = 1.99226932E-02;
    COFD[3308] = -2.13171682E+01;
    COFD[3309] = 5.37197338E+00;
    COFD[3310] = -4.70392872E-01;
    COFD[3311] = 1.99122802E-02;
    COFD[3312] = -2.14920449E+01;
    COFD[3313] = 5.44385051E+00;
    COFD[3314] = -4.76121506E-01;
    COFD[3315] = 2.00164081E-02;
    COFD[3316] = -2.18960800E+01;
    COFD[3317] = 5.54768472E+00;
    COFD[3318] = -4.87202065E-01;
    COFD[3319] = 2.04025437E-02;
    COFD[3320] = -2.19248250E+01;
    COFD[3321] = 5.49350509E+00;
    COFD[3322] = -4.81613405E-01;
    COFD[3323] = 2.02171734E-02;
    COFD[3324] = -2.19254485E+01;
    COFD[3325] = 5.49350509E+00;
    COFD[3326] = -4.81613405E-01;
    COFD[3327] = 2.02171734E-02;
    COFD[3328] = -1.59633387E+01;
    COFD[3329] = 3.66853818E+00;
    COFD[3330] = -2.64346221E-01;
    COFD[3331] = 1.15784613E-02;
    COFD[3332] = -1.50766130E+01;
    COFD[3333] = 3.47945612E+00;
    COFD[3334] = -2.40703722E-01;
    COFD[3335] = 1.05907441E-02;
    COFD[3336] = -1.25141260E+01;
    COFD[3337] = 2.77873601E+00;
    COFD[3338] = -1.50637360E-01;
    COFD[3339] = 6.72684281E-03;
    COFD[3340] = -1.57994893E+01;
    COFD[3341] = 4.22225052E+00;
    COFD[3342] = -3.35156428E-01;
    COFD[3343] = 1.46104855E-02;
    COFD[3344] = -1.50911794E+01;
    COFD[3345] = 3.47945612E+00;
    COFD[3346] = -2.40703722E-01;
    COFD[3347] = 1.05907441E-02;
    COFD[3348] = -2.12831323E+01;
    COFD[3349] = 5.61184117E+00;
    COFD[3350] = -4.90532156E-01;
    COFD[3351] = 2.03507922E-02;
    COFD[3352] = -1.63493345E+01;
    COFD[3353] = 3.82388595E+00;
    COFD[3354] = -2.84480724E-01;
    COFD[3355] = 1.24506311E-02;
    COFD[3356] = -1.63542394E+01;
    COFD[3357] = 3.82388595E+00;
    COFD[3358] = -2.84480724E-01;
    COFD[3359] = 1.24506311E-02;
    COFD[3360] = -1.63588981E+01;
    COFD[3361] = 3.82388595E+00;
    COFD[3362] = -2.84480724E-01;
    COFD[3363] = 1.24506311E-02;
    COFD[3364] = -1.50240272E+01;
    COFD[3365] = 3.47945612E+00;
    COFD[3366] = -2.40703722E-01;
    COFD[3367] = 1.05907441E-02;
    COFD[3368] = -2.14391943E+01;
    COFD[3369] = 5.56531152E+00;
    COFD[3370] = -4.88789821E-01;
    COFD[3371] = 2.04437116E-02;
    COFD[3372] = -1.73198034E+01;
    COFD[3373] = 4.21416723E+00;
    COFD[3374] = -3.34163932E-01;
    COFD[3375] = 1.45697432E-02;
    COFD[3376] = -1.93276434E+01;
    COFD[3377] = 4.85015581E+00;
    COFD[3378] = -4.10945109E-01;
    COFD[3379] = 1.76651398E-02;
    COFD[3380] = -1.59863030E+01;
    COFD[3381] = 3.67388294E+00;
    COFD[3382] = -2.64990709E-01;
    COFD[3383] = 1.16042706E-02;
    COFD[3384] = -2.14449559E+01;
    COFD[3385] = 5.56531152E+00;
    COFD[3386] = -4.88789821E-01;
    COFD[3387] = 2.04437116E-02;
    COFD[3388] = -1.73198034E+01;
    COFD[3389] = 4.21416723E+00;
    COFD[3390] = -3.34163932E-01;
    COFD[3391] = 1.45697432E-02;
    COFD[3392] = -1.73374529E+01;
    COFD[3393] = 4.21416723E+00;
    COFD[3394] = -3.34163932E-01;
    COFD[3395] = 1.45697432E-02;
    COFD[3396] = -2.09612557E+01;
    COFD[3397] = 5.40870099E+00;
    COFD[3398] = -4.73017610E-01;
    COFD[3399] = 1.99399066E-02;
    COFD[3400] = -1.72738845E+01;
    COFD[3401] = 4.19029808E+00;
    COFD[3402] = -3.31177076E-01;
    COFD[3403] = 1.44446234E-02;
    COFD[3404] = -2.13777308E+01;
    COFD[3405] = 5.54007827E+00;
    COFD[3406] = -4.86434511E-01;
    COFD[3407] = 2.03779006E-02;
    COFD[3408] = -1.94819080E+01;
    COFD[3409] = 4.87180830E+00;
    COFD[3410] = -4.13582958E-01;
    COFD[3411] = 1.77726094E-02;
    COFD[3412] = -1.94761606E+01;
    COFD[3413] = 4.87180830E+00;
    COFD[3414] = -4.13582958E-01;
    COFD[3415] = 1.77726094E-02;
    COFD[3416] = -2.11606963E+01;
    COFD[3417] = 5.42846112E+00;
    COFD[3418] = -4.74321870E-01;
    COFD[3419] = 1.99459749E-02;
    COFD[3420] = -2.11722423E+01;
    COFD[3421] = 5.42846112E+00;
    COFD[3422] = -4.74321870E-01;
    COFD[3423] = 1.99459749E-02;
    COFD[3424] = -2.14314090E+01;
    COFD[3425] = 5.54007827E+00;
    COFD[3426] = -4.86434511E-01;
    COFD[3427] = 2.03779006E-02;
    COFD[3428] = -2.14339566E+01;
    COFD[3429] = 5.54007827E+00;
    COFD[3430] = -4.86434511E-01;
    COFD[3431] = 2.03779006E-02;
    COFD[3432] = -1.95770968E+01;
    COFD[3433] = 4.97133070E+00;
    COFD[3434] = -4.25604177E-01;
    COFD[3435] = 1.82582594E-02;
    COFD[3436] = -1.72828302E+01;
    COFD[3437] = 4.26063341E+00;
    COFD[3438] = -3.39848064E-01;
    COFD[3439] = 1.48021313E-02;
    COFD[3440] = -1.95839648E+01;
    COFD[3441] = 4.97133070E+00;
    COFD[3442] = -4.25604177E-01;
    COFD[3443] = 1.82582594E-02;
    COFD[3444] = -2.11637902E+01;
    COFD[3445] = 5.42846112E+00;
    COFD[3446] = -4.74321870E-01;
    COFD[3447] = 1.99459749E-02;
    COFD[3448] = -2.02692384E+01;
    COFD[3449] = 5.14418672E+00;
    COFD[3450] = -4.45631004E-01;
    COFD[3451] = 1.90308403E-02;
    COFD[3452] = -1.91225414E+01;
    COFD[3453] = 4.82869066E+00;
    COFD[3454] = -4.08564514E-01;
    COFD[3455] = 1.75784675E-02;
    COFD[3456] = -2.15588759E+01;
    COFD[3457] = 5.51982454E+00;
    COFD[3458] = -4.84452039E-01;
    COFD[3459] = 2.03175522E-02;
    COFD[3460] = -2.11637902E+01;
    COFD[3461] = 5.42846112E+00;
    COFD[3462] = -4.74321870E-01;
    COFD[3463] = 1.99459749E-02;
    COFD[3464] = -2.15927763E+01;
    COFD[3465] = 5.51982454E+00;
    COFD[3466] = -4.84452039E-01;
    COFD[3467] = 2.03175522E-02;
    COFD[3468] = -1.89077781E+01;
    COFD[3469] = 4.68393046E+00;
    COFD[3470] = -3.91610863E-01;
    COFD[3471] = 1.69174645E-02;
    COFD[3472] = -2.04274471E+01;
    COFD[3473] = 5.18271974E+00;
    COFD[3474] = -4.49323627E-01;
    COFD[3475] = 1.91373940E-02;
    COFD[3476] = -2.04309557E+01;
    COFD[3477] = 5.18271974E+00;
    COFD[3478] = -4.49323627E-01;
    COFD[3479] = 1.91373940E-02;
    COFD[3480] = -2.03367561E+01;
    COFD[3481] = 5.15740122E+00;
    COFD[3482] = -4.46644818E-01;
    COFD[3483] = 1.90459001E-02;
    COFD[3484] = -2.03971290E+01;
    COFD[3485] = 5.13279789E+00;
    COFD[3486] = -4.44474174E-01;
    COFD[3487] = 1.89937678E-02;
    COFD[3488] = -2.18158049E+01;
    COFD[3489] = 5.53950393E+00;
    COFD[3490] = -4.86376204E-01;
    COFD[3491] = 2.03760106E-02;
    COFD[3492] = -2.09490548E+01;
    COFD[3493] = 5.31360223E+00;
    COFD[3494] = -4.64787000E-01;
    COFD[3495] = 1.97483720E-02;
    COFD[3496] = -2.09108261E+01;
    COFD[3497] = 5.30526648E+00;
    COFD[3498] = -4.63785596E-01;
    COFD[3499] = 1.97079873E-02;
    COFD[3500] = -2.09127554E+01;
    COFD[3501] = 5.30526648E+00;
    COFD[3502] = -4.63785596E-01;
    COFD[3503] = 1.97079873E-02;
    COFD[3504] = -2.10124405E+01;
    COFD[3505] = 5.29210705E+00;
    COFD[3506] = -4.62193217E-01;
    COFD[3507] = 1.96432872E-02;
    COFD[3508] = -2.19548723E+01;
    COFD[3509] = 5.56282156E+00;
    COFD[3510] = -4.88585679E-01;
    COFD[3511] = 2.04395879E-02;
    COFD[3512] = -2.13903532E+01;
    COFD[3513] = 5.38519776E+00;
    COFD[3514] = -4.71344997E-01;
    COFD[3515] = 1.99226932E-02;
    COFD[3516] = -2.13459128E+01;
    COFD[3517] = 5.37197338E+00;
    COFD[3518] = -4.70392872E-01;
    COFD[3519] = 1.99122802E-02;
    COFD[3520] = -2.15208595E+01;
    COFD[3521] = 5.44385051E+00;
    COFD[3522] = -4.76121506E-01;
    COFD[3523] = 2.00164081E-02;
    COFD[3524] = -2.19256706E+01;
    COFD[3525] = 5.54768472E+00;
    COFD[3526] = -4.87202065E-01;
    COFD[3527] = 2.04025437E-02;
    COFD[3528] = -2.19550907E+01;
    COFD[3529] = 5.49350509E+00;
    COFD[3530] = -4.81613405E-01;
    COFD[3531] = 2.02171734E-02;
    COFD[3532] = -2.19557531E+01;
    COFD[3533] = 5.49350509E+00;
    COFD[3534] = -4.81613405E-01;
    COFD[3535] = 2.02171734E-02;
    COFD[3536] = -2.02268902E+01;
    COFD[3537] = 5.13632093E+00;
    COFD[3538] = -4.44839124E-01;
    COFD[3539] = 1.90058354E-02;
    COFD[3540] = -1.88179418E+01;
    COFD[3541] = 4.79683898E+00;
    COFD[3542] = -4.04829719E-01;
    COFD[3543] = 1.74325475E-02;
    COFD[3544] = -1.58456300E+01;
    COFD[3545] = 4.02074783E+00;
    COFD[3546] = -3.10018522E-01;
    COFD[3547] = 1.35599552E-02;
    COFD[3548] = -1.92718582E+01;
    COFD[3549] = 5.41172124E+00;
    COFD[3550] = -4.73213887E-01;
    COFD[3551] = 1.99405473E-02;
    COFD[3552] = -1.88378874E+01;
    COFD[3553] = 4.79683898E+00;
    COFD[3554] = -4.04829719E-01;
    COFD[3555] = 1.74325475E-02;
    COFD[3556] = -1.65295288E+01;
    COFD[3557] = 2.97569206E+00;
    COFD[3558] = -6.75652842E-02;
    COFD[3559] = -1.08648422E-03;
    COFD[3560] = -2.04928958E+01;
    COFD[3561] = 5.22397933E+00;
    COFD[3562] = -4.54138171E-01;
    COFD[3563] = 1.93249285E-02;
    COFD[3564] = -2.02637994E+01;
    COFD[3565] = 5.14984081E+00;
    COFD[3566] = -4.46093018E-01;
    COFD[3567] = 1.90396647E-02;
    COFD[3568] = -2.02710316E+01;
    COFD[3569] = 5.14984081E+00;
    COFD[3570] = -4.46093018E-01;
    COFD[3571] = 1.90396647E-02;
    COFD[3572] = -1.87476063E+01;
    COFD[3573] = 4.79683898E+00;
    COFD[3574] = -4.04829719E-01;
    COFD[3575] = 1.74325475E-02;
    COFD[3576] = -2.01801667E+01;
    COFD[3577] = 4.53183330E+00;
    COFD[3578] = -3.02186760E-01;
    COFD[3579] = 1.02756490E-02;
    COFD[3580] = -2.09376196E+01;
    COFD[3581] = 5.40870099E+00;
    COFD[3582] = -4.73017610E-01;
    COFD[3583] = 1.99399066E-02;
    COFD[3584] = -2.20421041E+01;
    COFD[3585] = 5.52708332E+00;
    COFD[3586] = -4.68000808E-01;
    COFD[3587] = 1.89131908E-02;
    COFD[3588] = -2.02642227E+01;
    COFD[3589] = 5.14499740E+00;
    COFD[3590] = -4.45694430E-01;
    COFD[3591] = 1.90318646E-02;
    COFD[3592] = -2.01889168E+01;
    COFD[3593] = 4.53183330E+00;
    COFD[3594] = -3.02186760E-01;
    COFD[3595] = 1.02756490E-02;
    COFD[3596] = -2.09376196E+01;
    COFD[3597] = 5.40870099E+00;
    COFD[3598] = -4.73017610E-01;
    COFD[3599] = 1.99399066E-02;
    COFD[3600] = -2.09612557E+01;
    COFD[3601] = 5.40870099E+00;
    COFD[3602] = -4.73017610E-01;
    COFD[3603] = 1.99399066E-02;
    COFD[3604] = -1.95877017E+01;
    COFD[3605] = 4.27643051E+00;
    COFD[3606] = -2.68040901E-01;
    COFD[3607] = 8.77650113E-03;
    COFD[3608] = -2.11381508E+01;
    COFD[3609] = 5.45574440E+00;
    COFD[3610] = -4.77436155E-01;
    COFD[3611] = 2.00644596E-02;
    COFD[3612] = -2.03599050E+01;
    COFD[3613] = 4.60682543E+00;
    COFD[3614] = -3.13971634E-01;
    COFD[3615] = 1.08661011E-02;
    COFD[3616] = -2.21472114E+01;
    COFD[3617] = 5.56656297E+00;
    COFD[3618] = -4.75500048E-01;
    COFD[3619] = 1.93332291E-02;
    COFD[3620] = -2.21384805E+01;
    COFD[3621] = 5.56656297E+00;
    COFD[3622] = -4.75500048E-01;
    COFD[3623] = 1.93332291E-02;
    COFD[3624] = -2.09222454E+01;
    COFD[3625] = 4.82184721E+00;
    COFD[3626] = -3.48128875E-01;
    COFD[3627] = 1.25918978E-02;
    COFD[3628] = -2.09409936E+01;
    COFD[3629] = 4.82184721E+00;
    COFD[3630] = -3.48128875E-01;
    COFD[3631] = 1.25918978E-02;
    COFD[3632] = -2.04451935E+01;
    COFD[3633] = 4.60682543E+00;
    COFD[3634] = -3.13971634E-01;
    COFD[3635] = 1.08661011E-02;
    COFD[3636] = -2.04493813E+01;
    COFD[3637] = 4.60682543E+00;
    COFD[3638] = -3.13971634E-01;
    COFD[3639] = 1.08661011E-02;
    COFD[3640] = -2.18848136E+01;
    COFD[3641] = 5.51302074E+00;
    COFD[3642] = -4.65263979E-01;
    COFD[3643] = 1.87580679E-02;
    COFD[3644] = -2.09241647E+01;
    COFD[3645] = 5.42316225E+00;
    COFD[3646] = -4.73702801E-01;
    COFD[3647] = 1.99217718E-02;
    COFD[3648] = -2.18950505E+01;
    COFD[3649] = 5.51302074E+00;
    COFD[3650] = -4.65263979E-01;
    COFD[3651] = 1.87580679E-02;
    COFD[3652] = -2.09272429E+01;
    COFD[3653] = 4.82184721E+00;
    COFD[3654] = -3.48128875E-01;
    COFD[3655] = 1.25918978E-02;
    COFD[3656] = -2.19873532E+01;
    COFD[3657] = 5.39977369E+00;
    COFD[3658] = -4.43340854E-01;
    COFD[3659] = 1.75199613E-02;
    COFD[3660] = -2.19136842E+01;
    COFD[3661] = 5.58503445E+00;
    COFD[3662] = -4.79552117E-01;
    COFD[3663] = 1.95750393E-02;
    COFD[3664] = -2.07356106E+01;
    COFD[3665] = 4.65728078E+00;
    COFD[3666] = -3.22002062E-01;
    COFD[3667] = 1.12723316E-02;
    COFD[3668] = -2.09272429E+01;
    COFD[3669] = 4.82184721E+00;
    COFD[3670] = -3.48128875E-01;
    COFD[3671] = 1.25918978E-02;
    COFD[3672] = -2.07921175E+01;
    COFD[3673] = 4.65728078E+00;
    COFD[3674] = -3.22002062E-01;
    COFD[3675] = 1.12723316E-02;
    COFD[3676] = -2.20433329E+01;
    COFD[3677] = 5.59157589E+00;
    COFD[3678] = -4.85617912E-01;
    COFD[3679] = 2.00461138E-02;
    COFD[3680] = -2.19104953E+01;
    COFD[3681] = 5.33587903E+00;
    COFD[3682] = -4.32204887E-01;
    COFD[3683] = 1.69242106E-02;
    COFD[3684] = -2.19160962E+01;
    COFD[3685] = 5.33587903E+00;
    COFD[3686] = -4.32204887E-01;
    COFD[3687] = 1.69242106E-02;
    COFD[3688] = -2.19617977E+01;
    COFD[3689] = 5.37170913E+00;
    COFD[3690] = -4.38338667E-01;
    COFD[3691] = 1.72490835E-02;
    COFD[3692] = -2.21713935E+01;
    COFD[3693] = 5.41196486E+00;
    COFD[3694] = -4.45632422E-01;
    COFD[3695] = 1.76474237E-02;
    COFD[3696] = -2.01613414E+01;
    COFD[3697] = 4.29679630E+00;
    COFD[3698] = -2.69916064E-01;
    COFD[3699] = 8.81737046E-03;
    COFD[3700] = -2.18851200E+01;
    COFD[3701] = 5.21365421E+00;
    COFD[3702] = -4.11227771E-01;
    COFD[3703] = 1.58122118E-02;
    COFD[3704] = -2.18837863E+01;
    COFD[3705] = 5.22103227E+00;
    COFD[3706] = -4.12481899E-01;
    COFD[3707] = 1.58782021E-02;
    COFD[3708] = -2.18870332E+01;
    COFD[3709] = 5.22103227E+00;
    COFD[3710] = -4.12481899E-01;
    COFD[3711] = 1.58782021E-02;
    COFD[3712] = -2.20426031E+01;
    COFD[3713] = 5.23117744E+00;
    COFD[3714] = -4.14243780E-01;
    COFD[3715] = 1.59721173E-02;
    COFD[3716] = -2.08143755E+01;
    COFD[3717] = 4.54213239E+00;
    COFD[3718] = -3.03786739E-01;
    COFD[3719] = 1.03552672E-02;
    COFD[3720] = -2.17771707E+01;
    COFD[3721] = 5.03453866E+00;
    COFD[3722] = -3.81762947E-01;
    COFD[3723] = 1.42886762E-02;
    COFD[3724] = -2.17567719E+01;
    COFD[3725] = 5.03450665E+00;
    COFD[3726] = -3.83012475E-01;
    COFD[3727] = 1.43925680E-02;
    COFD[3728] = -2.12302151E+01;
    COFD[3729] = 4.79651003E+00;
    COFD[3730] = -3.44144386E-01;
    COFD[3731] = 1.23916372E-02;
    COFD[3732] = -2.09298486E+01;
    COFD[3733] = 4.59063108E+00;
    COFD[3734] = -3.11377715E-01;
    COFD[3735] = 1.07346023E-02;
    COFD[3736] = -2.13745703E+01;
    COFD[3737] = 4.71094320E+00;
    COFD[3738] = -3.30478653E-01;
    COFD[3739] = 1.16991305E-02;
    COFD[3740] = -2.13757703E+01;
    COFD[3741] = 4.71094320E+00;
    COFD[3742] = -3.30478653E-01;
    COFD[3743] = 1.16991305E-02;
    COFD[3744] = -1.59327297E+01;
    COFD[3745] = 3.65620899E+00;
    COFD[3746] = -2.62933804E-01;
    COFD[3747] = 1.15253223E-02;
    COFD[3748] = -1.50270339E+01;
    COFD[3749] = 3.46140064E+00;
    COFD[3750] = -2.38440092E-01;
    COFD[3751] = 1.04960087E-02;
    COFD[3752] = -1.24693568E+01;
    COFD[3753] = 2.76686648E+00;
    COFD[3754] = -1.49120141E-01;
    COFD[3755] = 6.66220432E-03;
    COFD[3756] = -1.57199037E+01;
    COFD[3757] = 4.19936335E+00;
    COFD[3758] = -3.32311009E-01;
    COFD[3759] = 1.44921003E-02;
    COFD[3760] = -1.50420953E+01;
    COFD[3761] = 3.46140064E+00;
    COFD[3762] = -2.38440092E-01;
    COFD[3763] = 1.04960087E-02;
    COFD[3764] = -2.14087397E+01;
    COFD[3765] = 5.57282008E+00;
    COFD[3766] = -4.76690890E-01;
    COFD[3767] = 1.94000719E-02;
    COFD[3768] = -1.62724462E+01;
    COFD[3769] = 3.79163564E+00;
    COFD[3770] = -2.80257365E-01;
    COFD[3771] = 1.22656902E-02;
    COFD[3772] = -1.62775714E+01;
    COFD[3773] = 3.79163564E+00;
    COFD[3774] = -2.80257365E-01;
    COFD[3775] = 1.22656902E-02;
    COFD[3776] = -1.62824412E+01;
    COFD[3777] = 3.79163564E+00;
    COFD[3778] = -2.80257365E-01;
    COFD[3779] = 1.22656902E-02;
    COFD[3780] = -1.49727799E+01;
    COFD[3781] = 3.46140064E+00;
    COFD[3782] = -2.38440092E-01;
    COFD[3783] = 1.04960087E-02;
    COFD[3784] = -2.14022336E+01;
    COFD[3785] = 5.55346617E+00;
    COFD[3786] = -4.87783156E-01;
    COFD[3787] = 2.04210886E-02;
    COFD[3788] = -1.72556729E+01;
    COFD[3789] = 4.19029808E+00;
    COFD[3790] = -3.31177076E-01;
    COFD[3791] = 1.44446234E-02;
    COFD[3792] = -1.92867554E+01;
    COFD[3793] = 4.83375900E+00;
    COFD[3794] = -4.09146560E-01;
    COFD[3795] = 1.76006599E-02;
    COFD[3796] = -1.59525102E+01;
    COFD[3797] = 3.66023858E+00;
    COFD[3798] = -2.63401043E-01;
    COFD[3799] = 1.15432000E-02;
    COFD[3800] = -2.14082453E+01;
    COFD[3801] = 5.55346617E+00;
    COFD[3802] = -4.87783156E-01;
    COFD[3803] = 2.04210886E-02;
    COFD[3804] = -1.72556729E+01;
    COFD[3805] = 4.19029808E+00;
    COFD[3806] = -3.31177076E-01;
    COFD[3807] = 1.44446234E-02;
    COFD[3808] = -1.72738845E+01;
    COFD[3809] = 4.19029808E+00;
    COFD[3810] = -3.31177076E-01;
    COFD[3811] = 1.44446234E-02;
    COFD[3812] = -2.11381508E+01;
    COFD[3813] = 5.45574440E+00;
    COFD[3814] = -4.77436155E-01;
    COFD[3815] = 2.00644596E-02;
    COFD[3816] = -1.72167708E+01;
    COFD[3817] = 4.16886779E+00;
    COFD[3818] = -3.28518156E-01;
    COFD[3819] = 1.43341626E-02;
    COFD[3820] = -2.13319784E+01;
    COFD[3821] = 5.52422470E+00;
    COFD[3822] = -4.84872944E-01;
    COFD[3823] = 2.03298213E-02;
    COFD[3824] = -1.94186547E+01;
    COFD[3825] = 4.84669430E+00;
    COFD[3826] = -4.10571455E-01;
    COFD[3827] = 1.76520543E-02;
    COFD[3828] = -1.94126575E+01;
    COFD[3829] = 4.84669430E+00;
    COFD[3830] = -4.10571455E-01;
    COFD[3831] = 1.76520543E-02;
    COFD[3832] = -2.11309207E+01;
    COFD[3833] = 5.41773516E+00;
    COFD[3834] = -4.73414338E-01;
    COFD[3835] = 1.99258685E-02;
    COFD[3836] = -2.11430338E+01;
    COFD[3837] = 5.41773516E+00;
    COFD[3838] = -4.73414338E-01;
    COFD[3839] = 1.99258685E-02;
    COFD[3840] = -2.13881945E+01;
    COFD[3841] = 5.52422470E+00;
    COFD[3842] = -4.84872944E-01;
    COFD[3843] = 2.03298213E-02;
    COFD[3844] = -2.13908698E+01;
    COFD[3845] = 5.52422470E+00;
    COFD[3846] = -4.84872944E-01;
    COFD[3847] = 2.03298213E-02;
    COFD[3848] = -1.95154079E+01;
    COFD[3849] = 4.94787350E+00;
    COFD[3850] = -4.22829292E-01;
    COFD[3851] = 1.81487163E-02;
    COFD[3852] = -1.72316148E+01;
    COFD[3853] = 4.24011069E+00;
    COFD[3854] = -3.37339810E-01;
    COFD[3855] = 1.46996679E-02;
    COFD[3856] = -1.95225629E+01;
    COFD[3857] = 4.94787350E+00;
    COFD[3858] = -4.22829292E-01;
    COFD[3859] = 1.81487163E-02;
    COFD[3860] = -2.11341653E+01;
    COFD[3861] = 5.41773516E+00;
    COFD[3862] = -4.73414338E-01;
    COFD[3863] = 1.99258685E-02;
    COFD[3864] = -2.02318658E+01;
    COFD[3865] = 5.12963391E+00;
    COFD[3866] = -4.44146826E-01;
    COFD[3867] = 1.89829640E-02;
    COFD[3868] = -1.90692595E+01;
    COFD[3869] = 4.80830699E+00;
    COFD[3870] = -4.06171933E-01;
    COFD[3871] = 1.74848791E-02;
    COFD[3872] = -2.15067581E+01;
    COFD[3873] = 5.49964831E+00;
    COFD[3874] = -4.82275380E-01;
    COFD[3875] = 2.02405072E-02;
    COFD[3876] = -2.11341653E+01;
    COFD[3877] = 5.41773516E+00;
    COFD[3878] = -4.73414338E-01;
    COFD[3879] = 1.99258685E-02;
    COFD[3880] = -2.15423956E+01;
    COFD[3881] = 5.49964831E+00;
    COFD[3882] = -4.82275380E-01;
    COFD[3883] = 2.02405072E-02;
    COFD[3884] = -1.88538435E+01;
    COFD[3885] = 4.66162351E+00;
    COFD[3886] = -3.88920477E-01;
    COFD[3887] = 1.68089648E-02;
    COFD[3888] = -2.03738891E+01;
    COFD[3889] = 5.16159436E+00;
    COFD[3890] = -4.46935283E-01;
    COFD[3891] = 1.90480297E-02;
    COFD[3892] = -2.03775651E+01;
    COFD[3893] = 5.16159436E+00;
    COFD[3894] = -4.46935283E-01;
    COFD[3895] = 1.90480297E-02;
    COFD[3896] = -2.03123540E+01;
    COFD[3897] = 5.14854169E+00;
    COFD[3898] = -4.45984343E-01;
    COFD[3899] = 1.90374217E-02;
    COFD[3900] = -2.03526104E+01;
    COFD[3901] = 5.11453301E+00;
    COFD[3902] = -4.42447016E-01;
    COFD[3903] = 1.89196698E-02;
    COFD[3904] = -2.18731920E+01;
    COFD[3905] = 5.55171660E+00;
    COFD[3906] = -4.87609504E-01;
    COFD[3907] = 2.04156590E-02;
    COFD[3908] = -2.08822487E+01;
    COFD[3909] = 5.28557747E+00;
    COFD[3910] = -4.61402384E-01;
    COFD[3911] = 1.96111546E-02;
    COFD[3912] = -2.08427678E+01;
    COFD[3913] = 5.27674330E+00;
    COFD[3914] = -4.60336155E-01;
    COFD[3915] = 1.95680191E-02;
    COFD[3916] = -2.08447974E+01;
    COFD[3917] = 5.27674330E+00;
    COFD[3918] = -4.60336155E-01;
    COFD[3919] = 1.95680191E-02;
    COFD[3920] = -2.09461018E+01;
    COFD[3921] = 5.26396793E+00;
    COFD[3922] = -4.58812213E-01;
    COFD[3923] = 1.95072180E-02;
    COFD[3924] = -2.19244555E+01;
    COFD[3925] = 5.54986547E+00;
    COFD[3926] = -4.87420926E-01;
    COFD[3927] = 2.04095097E-02;
    COFD[3928] = -2.13695648E+01;
    COFD[3929] = 5.37614538E+00;
    COFD[3930] = -4.70679659E-01;
    COFD[3931] = 1.99143937E-02;
    COFD[3932] = -2.13282915E+01;
    COFD[3933] = 5.36375915E+00;
    COFD[3934] = -4.69808195E-01;
    COFD[3935] = 1.99064589E-02;
    COFD[3936] = -2.14671205E+01;
    COFD[3937] = 5.42109069E+00;
    COFD[3938] = -4.73533096E-01;
    COFD[3939] = 1.99183547E-02;
    COFD[3940] = -2.18876256E+01;
    COFD[3941] = 5.53154746E+00;
    COFD[3942] = -4.85594344E-01;
    COFD[3943] = 2.03520324E-02;
    COFD[3944] = -2.19053841E+01;
    COFD[3945] = 5.47162499E+00;
    COFD[3946] = -4.79195552E-01;
    COFD[3947] = 2.01289088E-02;
    COFD[3948] = -2.19060847E+01;
    COFD[3949] = 5.47162499E+00;
    COFD[3950] = -4.79195552E-01;
    COFD[3951] = 2.01289088E-02;
    COFD[3952] = -2.03844252E+01;
    COFD[3953] = 5.18856872E+00;
    COFD[3954] = -4.50001829E-01;
    COFD[3955] = 1.91636142E-02;
    COFD[3956] = -1.93364585E+01;
    COFD[3957] = 4.98286777E+00;
    COFD[3958] = -4.26970814E-01;
    COFD[3959] = 1.83122917E-02;
    COFD[3960] = -1.59537247E+01;
    COFD[3961] = 4.07051484E+00;
    COFD[3962] = -3.16303109E-01;
    COFD[3963] = 1.38259377E-02;
    COFD[3964] = -1.96866103E+01;
    COFD[3965] = 5.54637286E+00;
    COFD[3966] = -4.87070324E-01;
    COFD[3967] = 2.03983467E-02;
    COFD[3968] = -1.93566243E+01;
    COFD[3969] = 4.98286777E+00;
    COFD[3970] = -4.26970814E-01;
    COFD[3971] = 1.83122917E-02;
    COFD[3972] = -1.80253664E+01;
    COFD[3973] = 3.69199168E+00;
    COFD[3974] = -1.74005516E-01;
    COFD[3975] = 3.97694372E-03;
    COFD[3976] = -2.07595845E+01;
    COFD[3977] = 5.32244593E+00;
    COFD[3978] = -4.65829403E-01;
    COFD[3979] = 1.97895274E-02;
    COFD[3980] = -2.07672833E+01;
    COFD[3981] = 5.32244593E+00;
    COFD[3982] = -4.65829403E-01;
    COFD[3983] = 1.97895274E-02;
    COFD[3984] = -2.07746356E+01;
    COFD[3985] = 5.32244593E+00;
    COFD[3986] = -4.65829403E-01;
    COFD[3987] = 1.97895274E-02;
    COFD[3988] = -1.92654138E+01;
    COFD[3989] = 4.98286777E+00;
    COFD[3990] = -4.26970814E-01;
    COFD[3991] = 1.83122917E-02;
    COFD[3992] = -1.93125662E+01;
    COFD[3993] = 4.10954793E+00;
    COFD[3994] = -2.37523329E-01;
    COFD[3995] = 7.08858141E-03;
    COFD[3996] = -2.13538553E+01;
    COFD[3997] = 5.54007827E+00;
    COFD[3998] = -4.86434511E-01;
    COFD[3999] = 2.03779006E-02;
    COFD[4000] = -2.20063594E+01;
    COFD[4001] = 5.48540187E+00;
    COFD[4002] = -4.58962148E-01;
    COFD[4003] = 1.83770355E-02;
    COFD[4004] = -2.04144604E+01;
    COFD[4005] = 5.19614628E+00;
    COFD[4006] = -4.50889164E-01;
    COFD[4007] = 1.91983328E-02;
    COFD[4008] = -1.93214527E+01;
    COFD[4009] = 4.10954793E+00;
    COFD[4010] = -2.37523329E-01;
    COFD[4011] = 7.08858141E-03;
    COFD[4012] = -2.13538553E+01;
    COFD[4013] = 5.54007827E+00;
    COFD[4014] = -4.86434511E-01;
    COFD[4015] = 2.03779006E-02;
    COFD[4016] = -2.13777308E+01;
    COFD[4017] = 5.54007827E+00;
    COFD[4018] = -4.86434511E-01;
    COFD[4019] = 2.03779006E-02;
    COFD[4020] = -2.03599050E+01;
    COFD[4021] = 4.60682543E+00;
    COFD[4022] = -3.13971634E-01;
    COFD[4023] = 1.08661011E-02;
    COFD[4024] = -2.13319784E+01;
    COFD[4025] = 5.52422470E+00;
    COFD[4026] = -4.84872944E-01;
    COFD[4027] = 2.03298213E-02;
    COFD[4028] = -1.95785144E+01;
    COFD[4029] = 4.22062499E+00;
    COFD[4030] = -2.54326872E-01;
    COFD[4031] = 7.91017784E-03;
    COFD[4032] = -2.20495822E+01;
    COFD[4033] = 5.47072190E+00;
    COFD[4034] = -4.56301261E-01;
    COFD[4035] = 1.82313566E-02;
    COFD[4036] = -2.20407152E+01;
    COFD[4037] = 5.47072190E+00;
    COFD[4038] = -4.56301261E-01;
    COFD[4039] = 1.82313566E-02;
    COFD[4040] = -2.03036402E+01;
    COFD[4041] = 4.50250781E+00;
    COFD[4042] = -2.97622106E-01;
    COFD[4043] = 1.00481473E-02;
    COFD[4044] = -2.03227406E+01;
    COFD[4045] = 4.50250781E+00;
    COFD[4046] = -2.97622106E-01;
    COFD[4047] = 1.00481473E-02;
    COFD[4048] = -1.96653154E+01;
    COFD[4049] = 4.22062499E+00;
    COFD[4050] = -2.54326872E-01;
    COFD[4051] = 7.91017784E-03;
    COFD[4052] = -1.96695844E+01;
    COFD[4053] = 4.22062499E+00;
    COFD[4054] = -2.54326872E-01;
    COFD[4055] = 7.91017784E-03;
    COFD[4056] = -2.17547312E+01;
    COFD[4057] = 5.40298848E+00;
    COFD[4058] = -4.43954594E-01;
    COFD[4059] = 1.75542998E-02;
    COFD[4060] = -2.13796303E+01;
    COFD[4061] = 5.56978987E+00;
    COFD[4062] = -4.89141980E-01;
    COFD[4063] = 2.04499210E-02;
    COFD[4064] = -2.17651187E+01;
    COFD[4065] = 5.40298848E+00;
    COFD[4066] = -4.43954594E-01;
    COFD[4067] = 1.75542998E-02;
    COFD[4068] = -2.03087302E+01;
    COFD[4069] = 4.50250781E+00;
    COFD[4070] = -2.97622106E-01;
    COFD[4071] = 1.00481473E-02;
    COFD[4072] = -2.16936515E+01;
    COFD[4073] = 5.21869603E+00;
    COFD[4074] = -4.12084772E-01;
    COFD[4075] = 1.58573035E-02;
    COFD[4076] = -2.18356866E+01;
    COFD[4077] = 5.49906960E+00;
    COFD[4078] = -4.61793001E-01;
    COFD[4079] = 1.85415189E-02;
    COFD[4080] = -2.00066696E+01;
    COFD[4081] = 4.29138907E+00;
    COFD[4082] = -2.65108149E-01;
    COFD[4083] = 8.43949637E-03;
    COFD[4084] = -2.03087302E+01;
    COFD[4085] = 4.50250781E+00;
    COFD[4086] = -2.97622106E-01;
    COFD[4087] = 1.00481473E-02;
    COFD[4088] = -2.00643134E+01;
    COFD[4089] = 4.29138907E+00;
    COFD[4090] = -2.65108149E-01;
    COFD[4091] = 8.43949637E-03;
    COFD[4092] = -2.21445051E+01;
    COFD[4093] = 5.58129885E+00;
    COFD[4094] = -4.78532921E-01;
    COFD[4095] = 1.95095699E-02;
    COFD[4096] = -2.15759895E+01;
    COFD[4097] = 5.13708607E+00;
    COFD[4098] = -3.98445708E-01;
    COFD[4099] = 1.51455626E-02;
    COFD[4100] = -2.15816909E+01;
    COFD[4101] = 5.13708607E+00;
    COFD[4102] = -3.98445708E-01;
    COFD[4103] = 1.51455626E-02;
    COFD[4104] = -2.16420936E+01;
    COFD[4105] = 5.17945041E+00;
    COFD[4106] = -4.05514689E-01;
    COFD[4107] = 1.55141412E-02;
    COFD[4108] = -2.18910102E+01;
    COFD[4109] = 5.23595129E+00;
    COFD[4110] = -4.15079064E-01;
    COFD[4111] = 1.60168286E-02;
    COFD[4112] = -2.00981944E+01;
    COFD[4113] = 4.22278378E+00;
    COFD[4114] = -2.54653500E-01;
    COFD[4115] = 7.92616085E-03;
    COFD[4116] = -2.13985484E+01;
    COFD[4117] = 4.94878244E+00;
    COFD[4118] = -3.68158605E-01;
    COFD[4119] = 1.36008797E-02;
    COFD[4120] = -2.14111310E+01;
    COFD[4121] = 4.96219227E+00;
    COFD[4122] = -3.70270843E-01;
    COFD[4123] = 1.37072211E-02;
    COFD[4124] = -2.14144448E+01;
    COFD[4125] = 4.96219227E+00;
    COFD[4126] = -3.70270843E-01;
    COFD[4127] = 1.37072211E-02;
    COFD[4128] = -2.15952753E+01;
    COFD[4129] = 4.98271982E+00;
    COFD[4130] = -3.73502341E-01;
    COFD[4131] = 1.38698700E-02;
    COFD[4132] = -1.99604682E+01;
    COFD[4133] = 4.12245214E+00;
    COFD[4134] = -2.39476227E-01;
    COFD[4135] = 7.18400558E-03;
    COFD[4136] = -2.11660262E+01;
    COFD[4137] = 4.71644372E+00;
    COFD[4138] = -3.31349990E-01;
    COFD[4139] = 1.17430818E-02;
    COFD[4140] = -2.12804720E+01;
    COFD[4141] = 4.77238689E+00;
    COFD[4142] = -3.40265855E-01;
    COFD[4143] = 1.21942137E-02;
    COFD[4144] = -2.06103015E+01;
    COFD[4145] = 4.47491202E+00;
    COFD[4146] = -2.93331059E-01;
    COFD[4147] = 9.83445305E-03;
    COFD[4148] = -2.01250987E+01;
    COFD[4149] = 4.19160608E+00;
    COFD[4150] = -2.49936771E-01;
    COFD[4151] = 7.69538319E-03;
    COFD[4152] = -2.06858147E+01;
    COFD[4153] = 4.35920123E+00;
    COFD[4154] = -2.75491273E-01;
    COFD[4155] = 8.95100289E-03;
    COFD[4156] = -2.06870442E+01;
    COFD[4157] = 4.35920123E+00;
    COFD[4158] = -2.75491273E-01;
    COFD[4159] = 8.95100289E-03;
    COFD[4160] = -1.82673770E+01;
    COFD[4161] = 4.39538102E+00;
    COFD[4162] = -3.56367230E-01;
    COFD[4163] = 1.54788461E-02;
    COFD[4164] = -1.72112971E+01;
    COFD[4165] = 4.15807461E+00;
    COFD[4166] = -3.27178539E-01;
    COFD[4167] = 1.42784349E-02;
    COFD[4168] = -1.39658996E+01;
    COFD[4169] = 3.24966086E+00;
    COFD[4170] = -2.11199992E-01;
    COFD[4171] = 9.32580661E-03;
    COFD[4172] = -1.78637178E+01;
    COFD[4173] = 4.88268692E+00;
    COFD[4174] = -4.14917638E-01;
    COFD[4175] = 1.78274298E-02;
    COFD[4176] = -1.72310232E+01;
    COFD[4177] = 4.15807461E+00;
    COFD[4178] = -3.27178539E-01;
    COFD[4179] = 1.42784349E-02;
    COFD[4180] = -2.13148887E+01;
    COFD[4181] = 5.27210469E+00;
    COFD[4182] = -4.21419216E-01;
    COFD[4183] = 1.63567178E-02;
    COFD[4184] = -1.85844688E+01;
    COFD[4185] = 4.51052425E+00;
    COFD[4186] = -3.70301627E-01;
    COFD[4187] = 1.60416153E-02;
    COFD[4188] = -1.85919214E+01;
    COFD[4189] = 4.51052425E+00;
    COFD[4190] = -3.70301627E-01;
    COFD[4191] = 1.60416153E-02;
    COFD[4192] = -1.85990352E+01;
    COFD[4193] = 4.51052425E+00;
    COFD[4194] = -3.70301627E-01;
    COFD[4195] = 1.60416153E-02;
    COFD[4196] = -1.71416702E+01;
    COFD[4197] = 4.15807461E+00;
    COFD[4198] = -3.27178539E-01;
    COFD[4199] = 1.42784349E-02;
    COFD[4200] = -2.19700018E+01;
    COFD[4201] = 5.43750833E+00;
    COFD[4202] = -4.50273329E-01;
    COFD[4203] = 1.79013718E-02;
    COFD[4204] = -1.94585111E+01;
    COFD[4205] = 4.87180830E+00;
    COFD[4206] = -4.13582958E-01;
    COFD[4207] = 1.77726094E-02;
    COFD[4208] = -2.14151520E+01;
    COFD[4209] = 5.41122754E+00;
    COFD[4210] = -4.73185889E-01;
    COFD[4211] = 1.99407905E-02;
    COFD[4212] = -1.82955252E+01;
    COFD[4213] = 4.40289649E+00;
    COFD[4214] = -3.57289765E-01;
    COFD[4215] = 1.55166804E-02;
    COFD[4216] = -2.19786173E+01;
    COFD[4217] = 5.43750833E+00;
    COFD[4218] = -4.50273329E-01;
    COFD[4219] = 1.79013718E-02;
    COFD[4220] = -1.94585111E+01;
    COFD[4221] = 4.87180830E+00;
    COFD[4222] = -4.13582958E-01;
    COFD[4223] = 1.77726094E-02;
    COFD[4224] = -1.94819080E+01;
    COFD[4225] = 4.87180830E+00;
    COFD[4226] = -4.13582958E-01;
    COFD[4227] = 1.77726094E-02;
    COFD[4228] = -2.21472114E+01;
    COFD[4229] = 5.56656297E+00;
    COFD[4230] = -4.75500048E-01;
    COFD[4231] = 1.93332291E-02;
    COFD[4232] = -1.94186547E+01;
    COFD[4233] = 4.84669430E+00;
    COFD[4234] = -4.10571455E-01;
    COFD[4235] = 1.76520543E-02;
    COFD[4236] = -2.20495822E+01;
    COFD[4237] = 5.47072190E+00;
    COFD[4238] = -4.56301261E-01;
    COFD[4239] = 1.82313566E-02;
    COFD[4240] = -2.14907782E+01;
    COFD[4241] = 5.41585806E+00;
    COFD[4242] = -4.73359323E-01;
    COFD[4243] = 1.99310239E-02;
    COFD[4244] = -2.14821817E+01;
    COFD[4245] = 5.41585806E+00;
    COFD[4246] = -4.73359323E-01;
    COFD[4247] = 1.99310239E-02;
    COFD[4248] = -2.22429814E+01;
    COFD[4249] = 5.53139819E+00;
    COFD[4250] = -4.68828555E-01;
    COFD[4251] = 1.89597887E-02;
    COFD[4252] = -2.22613837E+01;
    COFD[4253] = 5.53139819E+00;
    COFD[4254] = -4.68828555E-01;
    COFD[4255] = 1.89597887E-02;
    COFD[4256] = -2.21333822E+01;
    COFD[4257] = 5.47072190E+00;
    COFD[4258] = -4.56301261E-01;
    COFD[4259] = 1.82313566E-02;
    COFD[4260] = -2.21374903E+01;
    COFD[4261] = 5.47072190E+00;
    COFD[4262] = -4.56301261E-01;
    COFD[4263] = 1.82313566E-02;
    COFD[4264] = -2.15206146E+01;
    COFD[4265] = 5.48426911E+00;
    COFD[4266] = -4.80606512E-01;
    COFD[4267] = 2.01811046E-02;
    COFD[4268] = -1.95836394E+01;
    COFD[4269] = 4.93449043E+00;
    COFD[4270] = -4.21243802E-01;
    COFD[4271] = 1.80859966E-02;
    COFD[4272] = -2.15307023E+01;
    COFD[4273] = 5.48426911E+00;
    COFD[4274] = -4.80606512E-01;
    COFD[4275] = 2.01811046E-02;
    COFD[4276] = -2.22478879E+01;
    COFD[4277] = 5.53139819E+00;
    COFD[4278] = -4.68828555E-01;
    COFD[4279] = 1.89597887E-02;
    COFD[4280] = -2.21343023E+01;
    COFD[4281] = 5.60010742E+00;
    COFD[4282] = -4.91597429E-01;
    COFD[4283] = 2.04987718E-02;
    COFD[4284] = -2.12014186E+01;
    COFD[4285] = 5.40060531E+00;
    COFD[4286] = -4.72449699E-01;
    COFD[4287] = 1.99345817E-02;
    COFD[4288] = -2.23404275E+01;
    COFD[4289] = 5.49239750E+00;
    COFD[4290] = -4.60320987E-01;
    COFD[4291] = 1.84538922E-02;
    COFD[4292] = -2.22478879E+01;
    COFD[4293] = 5.53139819E+00;
    COFD[4294] = -4.68828555E-01;
    COFD[4295] = 1.89597887E-02;
    COFD[4296] = -2.23958208E+01;
    COFD[4297] = 5.49239750E+00;
    COFD[4298] = -4.60320987E-01;
    COFD[4299] = 1.84538922E-02;
    COFD[4300] = -2.11349436E+01;
    COFD[4301] = 5.32202066E+00;
    COFD[4302] = -4.65780334E-01;
    COFD[4303] = 1.97876377E-02;
    COFD[4304] = -2.22262162E+01;
    COFD[4305] = 5.61211818E+00;
    COFD[4306] = -4.91432482E-01;
    COFD[4307] = 2.04238731E-02;
    COFD[4308] = -2.22317182E+01;
    COFD[4309] = 5.61211818E+00;
    COFD[4310] = -4.91432482E-01;
    COFD[4311] = 2.04238731E-02;
    COFD[4312] = -2.21793326E+01;
    COFD[4313] = 5.60403905E+00;
    COFD[4314] = -4.91221691E-01;
    COFD[4315] = 2.04473483E-02;
    COFD[4316] = -2.22701953E+01;
    COFD[4317] = 5.59632316E+00;
    COFD[4318] = -4.91568011E-01;
    COFD[4319] = 2.05156966E-02;
    COFD[4320] = -2.25302512E+01;
    COFD[4321] = 5.47136127E+00;
    COFD[4322] = -4.56417141E-01;
    COFD[4323] = 1.82376994E-02;
    COFD[4324] = -2.24120415E+01;
    COFD[4325] = 5.58744076E+00;
    COFD[4326] = -4.84489462E-01;
    COFD[4327] = 1.99733042E-02;
    COFD[4328] = -2.23993836E+01;
    COFD[4329] = 5.58952429E+00;
    COFD[4330] = -4.85012530E-01;
    COFD[4331] = 2.00062142E-02;
    COFD[4332] = -2.24025650E+01;
    COFD[4333] = 5.58952429E+00;
    COFD[4334] = -4.85012530E-01;
    COFD[4335] = 2.00062142E-02;
    COFD[4336] = -2.25300734E+01;
    COFD[4337] = 5.59173268E+00;
    COFD[4338] = -4.85654660E-01;
    COFD[4339] = 2.00483698E-02;
    COFD[4340] = -2.25553202E+01;
    COFD[4341] = 5.44166443E+00;
    COFD[4342] = -4.51021243E-01;
    COFD[4343] = 1.79421190E-02;
    COFD[4344] = -2.27001899E+01;
    COFD[4345] = 5.58468914E+00;
    COFD[4346] = -4.79958407E-01;
    COFD[4347] = 1.96104043E-02;
    COFD[4348] = -2.26853912E+01;
    COFD[4349] = 5.58521030E+00;
    COFD[4350] = -4.81061650E-01;
    COFD[4351] = 1.96992215E-02;
    COFD[4352] = -2.25695574E+01;
    COFD[4353] = 5.52323975E+00;
    COFD[4354] = -4.67257607E-01;
    COFD[4355] = 1.88711975E-02;
    COFD[4356] = -2.26027431E+01;
    COFD[4357] = 5.46217527E+00;
    COFD[4358] = -4.54751471E-01;
    COFD[4359] = 1.81465218E-02;
    COFD[4360] = -2.28446667E+01;
    COFD[4361] = 5.50134401E+00;
    COFD[4362] = -4.62488197E-01;
    COFD[4363] = 1.85873697E-02;
    COFD[4364] = -2.28458380E+01;
    COFD[4365] = 5.50134401E+00;
    COFD[4366] = -4.62488197E-01;
    COFD[4367] = 1.85873697E-02;
    COFD[4368] = -1.82590824E+01;
    COFD[4369] = 4.39538102E+00;
    COFD[4370] = -3.56367230E-01;
    COFD[4371] = 1.54788461E-02;
    COFD[4372] = -1.72053106E+01;
    COFD[4373] = 4.15807461E+00;
    COFD[4374] = -3.27178539E-01;
    COFD[4375] = 1.42784349E-02;
    COFD[4376] = -1.39648112E+01;
    COFD[4377] = 3.24966086E+00;
    COFD[4378] = -2.11199992E-01;
    COFD[4379] = 9.32580661E-03;
    COFD[4380] = -1.78631557E+01;
    COFD[4381] = 4.88268692E+00;
    COFD[4382] = -4.14917638E-01;
    COFD[4383] = 1.78274298E-02;
    COFD[4384] = -1.72247972E+01;
    COFD[4385] = 4.15807461E+00;
    COFD[4386] = -3.27178539E-01;
    COFD[4387] = 1.42784349E-02;
    COFD[4388] = -2.13084334E+01;
    COFD[4389] = 5.27210469E+00;
    COFD[4390] = -4.21419216E-01;
    COFD[4391] = 1.63567178E-02;
    COFD[4392] = -1.85756076E+01;
    COFD[4393] = 4.51052425E+00;
    COFD[4394] = -3.70301627E-01;
    COFD[4395] = 1.60416153E-02;
    COFD[4396] = -1.85829283E+01;
    COFD[4397] = 4.51052425E+00;
    COFD[4398] = -3.70301627E-01;
    COFD[4399] = 1.60416153E-02;
    COFD[4400] = -1.85899144E+01;
    COFD[4401] = 4.51052425E+00;
    COFD[4402] = -3.70301627E-01;
    COFD[4403] = 1.60416153E-02;
    COFD[4404] = -1.71364578E+01;
    COFD[4405] = 4.15807461E+00;
    COFD[4406] = -3.27178539E-01;
    COFD[4407] = 1.42784349E-02;
    COFD[4408] = -2.19615570E+01;
    COFD[4409] = 5.43750833E+00;
    COFD[4410] = -4.50273329E-01;
    COFD[4411] = 1.79013718E-02;
    COFD[4412] = -1.94530250E+01;
    COFD[4413] = 4.87180830E+00;
    COFD[4414] = -4.13582958E-01;
    COFD[4415] = 1.77726094E-02;
    COFD[4416] = -2.14049543E+01;
    COFD[4417] = 5.41122754E+00;
    COFD[4418] = -4.73185889E-01;
    COFD[4419] = 1.99407905E-02;
    COFD[4420] = -1.82872310E+01;
    COFD[4421] = 4.40289649E+00;
    COFD[4422] = -3.57289765E-01;
    COFD[4423] = 1.55166804E-02;
    COFD[4424] = -2.19700270E+01;
    COFD[4425] = 5.43750833E+00;
    COFD[4426] = -4.50273329E-01;
    COFD[4427] = 1.79013718E-02;
    COFD[4428] = -1.94530250E+01;
    COFD[4429] = 4.87180830E+00;
    COFD[4430] = -4.13582958E-01;
    COFD[4431] = 1.77726094E-02;
    COFD[4432] = -1.94761606E+01;
    COFD[4433] = 4.87180830E+00;
    COFD[4434] = -4.13582958E-01;
    COFD[4435] = 1.77726094E-02;
    COFD[4436] = -2.21384805E+01;
    COFD[4437] = 5.56656297E+00;
    COFD[4438] = -4.75500048E-01;
    COFD[4439] = 1.93332291E-02;
    COFD[4440] = -1.94126575E+01;
    COFD[4441] = 4.84669430E+00;
    COFD[4442] = -4.10571455E-01;
    COFD[4443] = 1.76520543E-02;
    COFD[4444] = -2.20407152E+01;
    COFD[4445] = 5.47072190E+00;
    COFD[4446] = -4.56301261E-01;
    COFD[4447] = 1.82313566E-02;
    COFD[4448] = -2.14821817E+01;
    COFD[4449] = 5.41585806E+00;
    COFD[4450] = -4.73359323E-01;
    COFD[4451] = 1.99310239E-02;
    COFD[4452] = -2.14737305E+01;
    COFD[4453] = 5.41585806E+00;
    COFD[4454] = -4.73359323E-01;
    COFD[4455] = 1.99310239E-02;
    COFD[4456] = -2.22329724E+01;
    COFD[4457] = 5.53139819E+00;
    COFD[4458] = -4.68828555E-01;
    COFD[4459] = 1.89597887E-02;
    COFD[4460] = -2.22510033E+01;
    COFD[4461] = 5.53139819E+00;
    COFD[4462] = -4.68828555E-01;
    COFD[4463] = 1.89597887E-02;
    COFD[4464] = -2.21229141E+01;
    COFD[4465] = 5.47072190E+00;
    COFD[4466] = -4.56301261E-01;
    COFD[4467] = 1.82313566E-02;
    COFD[4468] = -2.21269367E+01;
    COFD[4469] = 5.47072190E+00;
    COFD[4470] = -4.56301261E-01;
    COFD[4471] = 1.82313566E-02;
    COFD[4472] = -2.15126310E+01;
    COFD[4473] = 5.48426911E+00;
    COFD[4474] = -4.80606512E-01;
    COFD[4475] = 2.01811046E-02;
    COFD[4476] = -1.95737308E+01;
    COFD[4477] = 4.93449043E+00;
    COFD[4478] = -4.21243802E-01;
    COFD[4479] = 1.80859966E-02;
    COFD[4480] = -2.15225573E+01;
    COFD[4481] = 5.48426911E+00;
    COFD[4482] = -4.80606512E-01;
    COFD[4483] = 2.01811046E-02;
    COFD[4484] = -2.22377812E+01;
    COFD[4485] = 5.53139819E+00;
    COFD[4486] = -4.68828555E-01;
    COFD[4487] = 1.89597887E-02;
    COFD[4488] = -2.21242889E+01;
    COFD[4489] = 5.60010742E+00;
    COFD[4490] = -4.91597429E-01;
    COFD[4491] = 2.04987718E-02;
    COFD[4492] = -2.11931178E+01;
    COFD[4493] = 5.40060531E+00;
    COFD[4494] = -4.72449699E-01;
    COFD[4495] = 1.99345817E-02;
    COFD[4496] = -2.23301333E+01;
    COFD[4497] = 5.49239750E+00;
    COFD[4498] = -4.60320987E-01;
    COFD[4499] = 1.84538922E-02;
    COFD[4500] = -2.22377812E+01;
    COFD[4501] = 5.53139819E+00;
    COFD[4502] = -4.68828555E-01;
    COFD[4503] = 1.89597887E-02;
    COFD[4504] = -2.23843343E+01;
    COFD[4505] = 5.49239750E+00;
    COFD[4506] = -4.60320987E-01;
    COFD[4507] = 1.84538922E-02;
    COFD[4508] = -2.11253498E+01;
    COFD[4509] = 5.32202066E+00;
    COFD[4510] = -4.65780334E-01;
    COFD[4511] = 1.97876377E-02;
    COFD[4512] = -2.22165128E+01;
    COFD[4513] = 5.61211818E+00;
    COFD[4514] = -4.91432482E-01;
    COFD[4515] = 2.04238731E-02;
    COFD[4516] = -2.22219085E+01;
    COFD[4517] = 5.61211818E+00;
    COFD[4518] = -4.91432482E-01;
    COFD[4519] = 2.04238731E-02;
    COFD[4520] = -2.21694197E+01;
    COFD[4521] = 5.60403905E+00;
    COFD[4522] = -4.91221691E-01;
    COFD[4523] = 2.04473483E-02;
    COFD[4524] = -2.22600844E+01;
    COFD[4525] = 5.59632316E+00;
    COFD[4526] = -4.91568011E-01;
    COFD[4527] = 2.05156966E-02;
    COFD[4528] = -2.25180193E+01;
    COFD[4529] = 5.47136127E+00;
    COFD[4530] = -4.56417141E-01;
    COFD[4531] = 1.82376994E-02;
    COFD[4532] = -2.24010182E+01;
    COFD[4533] = 5.58744076E+00;
    COFD[4534] = -4.84489462E-01;
    COFD[4535] = 1.99733042E-02;
    COFD[4536] = -2.23882886E+01;
    COFD[4537] = 5.58952429E+00;
    COFD[4538] = -4.85012530E-01;
    COFD[4539] = 2.00062142E-02;
    COFD[4540] = -2.23913999E+01;
    COFD[4541] = 5.58952429E+00;
    COFD[4542] = -4.85012530E-01;
    COFD[4543] = 2.00062142E-02;
    COFD[4544] = -2.25188399E+01;
    COFD[4545] = 5.59173268E+00;
    COFD[4546] = -4.85654660E-01;
    COFD[4547] = 2.00483698E-02;
    COFD[4548] = -2.25425191E+01;
    COFD[4549] = 5.44166443E+00;
    COFD[4550] = -4.51021243E-01;
    COFD[4551] = 1.79421190E-02;
    COFD[4552] = -2.26882488E+01;
    COFD[4553] = 5.58468914E+00;
    COFD[4554] = -4.79958407E-01;
    COFD[4555] = 1.96104043E-02;
    COFD[4556] = -2.26733985E+01;
    COFD[4557] = 5.58521030E+00;
    COFD[4558] = -4.81061650E-01;
    COFD[4559] = 1.96992215E-02;
    COFD[4560] = -2.25575141E+01;
    COFD[4561] = 5.52323975E+00;
    COFD[4562] = -4.67257607E-01;
    COFD[4563] = 1.88711975E-02;
    COFD[4564] = -2.25901270E+01;
    COFD[4565] = 5.46217527E+00;
    COFD[4566] = -4.54751471E-01;
    COFD[4567] = 1.81465218E-02;
    COFD[4568] = -2.28315330E+01;
    COFD[4569] = 5.50134401E+00;
    COFD[4570] = -4.62488197E-01;
    COFD[4571] = 1.85873697E-02;
    COFD[4572] = -2.28326740E+01;
    COFD[4573] = 5.50134401E+00;
    COFD[4574] = -4.62488197E-01;
    COFD[4575] = 1.85873697E-02;
    COFD[4576] = -2.02646611E+01;
    COFD[4577] = 5.10426133E+00;
    COFD[4578] = -4.41256919E-01;
    COFD[4579] = 1.88737290E-02;
    COFD[4580] = -1.90883268E+01;
    COFD[4581] = 4.84384483E+00;
    COFD[4582] = -4.10265575E-01;
    COFD[4583] = 1.76414287E-02;
    COFD[4584] = -1.57034851E+01;
    COFD[4585] = 3.93614244E+00;
    COFD[4586] = -2.99111497E-01;
    COFD[4587] = 1.30888229E-02;
    COFD[4588] = -1.94688688E+01;
    COFD[4589] = 5.43830787E+00;
    COFD[4590] = -4.75472880E-01;
    COFD[4591] = 1.99909996E-02;
    COFD[4592] = -1.91102652E+01;
    COFD[4593] = 4.84384483E+00;
    COFD[4594] = -4.10265575E-01;
    COFD[4595] = 1.76414287E-02;
    COFD[4596] = -1.87383952E+01;
    COFD[4597] = 3.96926341E+00;
    COFD[4598] = -2.16412264E-01;
    COFD[4599] = 6.06012078E-03;
    COFD[4600] = -2.05184870E+01;
    COFD[4601] = 5.18417470E+00;
    COFD[4602] = -4.49491573E-01;
    COFD[4603] = 1.91438508E-02;
    COFD[4604] = -2.05272328E+01;
    COFD[4605] = 5.18417470E+00;
    COFD[4606] = -4.49491573E-01;
    COFD[4607] = 1.91438508E-02;
    COFD[4608] = -2.05356023E+01;
    COFD[4609] = 5.18417470E+00;
    COFD[4610] = -4.49491573E-01;
    COFD[4611] = 1.91438508E-02;
    COFD[4612] = -1.90116191E+01;
    COFD[4613] = 4.84384483E+00;
    COFD[4614] = -4.10265575E-01;
    COFD[4615] = 1.76414287E-02;
    COFD[4616] = -2.00915040E+01;
    COFD[4617] = 4.41511629E+00;
    COFD[4618] = -2.84086963E-01;
    COFD[4619] = 9.37586971E-03;
    COFD[4620] = -2.11349086E+01;
    COFD[4621] = 5.42846112E+00;
    COFD[4622] = -4.74321870E-01;
    COFD[4623] = 1.99459749E-02;
    COFD[4624] = -2.22116706E+01;
    COFD[4625] = 5.54251230E+00;
    COFD[4626] = -4.70946314E-01;
    COFD[4627] = 1.90785869E-02;
    COFD[4628] = -2.02922701E+01;
    COFD[4629] = 5.11106992E+00;
    COFD[4630] = -4.42047129E-01;
    COFD[4631] = 1.89042990E-02;
    COFD[4632] = -2.01015340E+01;
    COFD[4633] = 4.41511629E+00;
    COFD[4634] = -2.84086963E-01;
    COFD[4635] = 9.37586971E-03;
    COFD[4636] = -2.11349086E+01;
    COFD[4637] = 5.42846112E+00;
    COFD[4638] = -4.74321870E-01;
    COFD[4639] = 1.99459749E-02;
    COFD[4640] = -2.11606963E+01;
    COFD[4641] = 5.42846112E+00;
    COFD[4642] = -4.74321870E-01;
    COFD[4643] = 1.99459749E-02;
    COFD[4644] = -2.09222454E+01;
    COFD[4645] = 4.82184721E+00;
    COFD[4646] = -3.48128875E-01;
    COFD[4647] = 1.25918978E-02;
    COFD[4648] = -2.11309207E+01;
    COFD[4649] = 5.41773516E+00;
    COFD[4650] = -4.73414338E-01;
    COFD[4651] = 1.99258685E-02;
    COFD[4652] = -2.03036402E+01;
    COFD[4653] = 4.50250781E+00;
    COFD[4654] = -2.97622106E-01;
    COFD[4655] = 1.00481473E-02;
    COFD[4656] = -2.22429814E+01;
    COFD[4657] = 5.53139819E+00;
    COFD[4658] = -4.68828555E-01;
    COFD[4659] = 1.89597887E-02;
    COFD[4660] = -2.22329724E+01;
    COFD[4661] = 5.53139819E+00;
    COFD[4662] = -4.68828555E-01;
    COFD[4663] = 1.89597887E-02;
    COFD[4664] = -2.09002742E+01;
    COFD[4665] = 4.72895031E+00;
    COFD[4666] = -3.33332771E-01;
    COFD[4667] = 1.18431478E-02;
    COFD[4668] = -2.09224206E+01;
    COFD[4669] = 4.72895031E+00;
    COFD[4670] = -3.33332771E-01;
    COFD[4671] = 1.18431478E-02;
    COFD[4672] = -2.04033972E+01;
    COFD[4673] = 4.50250781E+00;
    COFD[4674] = -2.97622106E-01;
    COFD[4675] = 1.00481473E-02;
    COFD[4676] = -2.04083729E+01;
    COFD[4677] = 4.50250781E+00;
    COFD[4678] = -2.97622106E-01;
    COFD[4679] = 1.00481473E-02;
    COFD[4680] = -2.20262793E+01;
    COFD[4681] = 5.49663315E+00;
    COFD[4682] = -4.61182837E-01;
    COFD[4683] = 1.85035558E-02;
    COFD[4684] = -2.12621914E+01;
    COFD[4685] = 5.47935225E+00;
    COFD[4686] = -4.80056796E-01;
    COFD[4687] = 2.01607180E-02;
    COFD[4688] = -2.20379206E+01;
    COFD[4689] = 5.49663315E+00;
    COFD[4690] = -4.61182837E-01;
    COFD[4691] = 1.85035558E-02;
    COFD[4692] = -2.09061629E+01;
    COFD[4693] = 4.72895031E+00;
    COFD[4694] = -3.33332771E-01;
    COFD[4695] = 1.18431478E-02;
    COFD[4696] = -2.20597305E+01;
    COFD[4697] = 5.34774760E+00;
    COFD[4698] = -4.34239753E-01;
    COFD[4699] = 1.70320676E-02;
    COFD[4700] = -2.20398328E+01;
    COFD[4701] = 5.56049839E+00;
    COFD[4702] = -4.74367872E-01;
    COFD[4703] = 1.92702787E-02;
    COFD[4704] = -2.07072145E+01;
    COFD[4705] = 4.56211059E+00;
    COFD[4706] = -3.06895158E-01;
    COFD[4707] = 1.05100393E-02;
    COFD[4708] = -2.09061629E+01;
    COFD[4709] = 4.72895031E+00;
    COFD[4710] = -3.33332771E-01;
    COFD[4711] = 1.18431478E-02;
    COFD[4712] = -2.07748171E+01;
    COFD[4713] = 4.56211059E+00;
    COFD[4714] = -3.06895158E-01;
    COFD[4715] = 1.05100393E-02;
    COFD[4716] = -2.22149446E+01;
    COFD[4717] = 5.58360799E+00;
    COFD[4718] = -4.82701436E-01;
    COFD[4719] = 1.98437922E-02;
    COFD[4720] = -2.19526490E+01;
    COFD[4721] = 5.27258289E+00;
    COFD[4722] = -4.21502790E-01;
    COFD[4723] = 1.63611949E-02;
    COFD[4724] = -2.19592125E+01;
    COFD[4725] = 5.27258289E+00;
    COFD[4726] = -4.21502790E-01;
    COFD[4727] = 1.63611949E-02;
    COFD[4728] = -2.20192352E+01;
    COFD[4729] = 5.31412694E+00;
    COFD[4730] = -4.28473898E-01;
    COFD[4731] = 1.67264841E-02;
    COFD[4732] = -2.22545356E+01;
    COFD[4733] = 5.36643605E+00;
    COFD[4734] = -4.37440735E-01;
    COFD[4735] = 1.72016388E-02;
    COFD[4736] = -2.08353693E+01;
    COFD[4737] = 4.50409026E+00;
    COFD[4738] = -2.97868419E-01;
    COFD[4739] = 1.00604224E-02;
    COFD[4740] = -2.19253091E+01;
    COFD[4741] = 5.14570932E+00;
    COFD[4742] = -3.99877142E-01;
    COFD[4743] = 1.52199557E-02;
    COFD[4744] = -2.19282979E+01;
    COFD[4745] = 5.15446948E+00;
    COFD[4746] = -4.01332769E-01;
    COFD[4747] = 1.52956262E-02;
    COFD[4748] = -2.19322003E+01;
    COFD[4749] = 5.15446948E+00;
    COFD[4750] = -4.01332769E-01;
    COFD[4751] = 1.52956262E-02;
    COFD[4752] = -2.20891322E+01;
    COFD[4753] = 5.16679492E+00;
    COFD[4754] = -4.03405751E-01;
    COFD[4755] = 1.54041741E-02;
    COFD[4756] = -2.07557953E+01;
    COFD[4757] = 4.42680848E+00;
    COFD[4758] = -2.85885288E-01;
    COFD[4759] = 9.46483934E-03;
    COFD[4760] = -2.17463767E+01;
    COFD[4761] = 4.93496210E+00;
    COFD[4762] = -3.65981745E-01;
    COFD[4763] = 1.34912948E-02;
    COFD[4764] = -2.18797352E+01;
    COFD[4765] = 4.99907484E+00;
    COFD[4766] = -3.76094627E-01;
    COFD[4767] = 1.40009262E-02;
    COFD[4768] = -2.12221678E+01;
    COFD[4769] = 4.70506024E+00;
    COFD[4770] = -3.29547212E-01;
    COFD[4771] = 1.16521630E-02;
    COFD[4772] = -2.08821587E+01;
    COFD[4773] = 4.48108132E+00;
    COFD[4774] = -2.94289899E-01;
    COFD[4775] = 9.88218297E-03;
    COFD[4776] = -2.13524540E+01;
    COFD[4777] = 4.61201872E+00;
    COFD[4778] = -3.14803338E-01;
    COFD[4779] = 1.09082984E-02;
    COFD[4780] = -2.13539532E+01;
    COFD[4781] = 4.61201872E+00;
    COFD[4782] = -3.14803338E-01;
    COFD[4783] = 1.09082984E-02;
    COFD[4784] = -2.02822946E+01;
    COFD[4785] = 5.10426133E+00;
    COFD[4786] = -4.41256919E-01;
    COFD[4787] = 1.88737290E-02;
    COFD[4788] = -1.91004157E+01;
    COFD[4789] = 4.84384483E+00;
    COFD[4790] = -4.10265575E-01;
    COFD[4791] = 1.76414287E-02;
    COFD[4792] = -1.57054717E+01;
    COFD[4793] = 3.93614244E+00;
    COFD[4794] = -2.99111497E-01;
    COFD[4795] = 1.30888229E-02;
    COFD[4796] = -1.94698843E+01;
    COFD[4797] = 5.43830787E+00;
    COFD[4798] = -4.75472880E-01;
    COFD[4799] = 1.99909996E-02;
    COFD[4800] = -1.91229033E+01;
    COFD[4801] = 4.84384483E+00;
    COFD[4802] = -4.10265575E-01;
    COFD[4803] = 1.76414287E-02;
    COFD[4804] = -1.87515645E+01;
    COFD[4805] = 3.96926341E+00;
    COFD[4806] = -2.16412264E-01;
    COFD[4807] = 6.06012078E-03;
    COFD[4808] = -2.05375724E+01;
    COFD[4809] = 5.18417470E+00;
    COFD[4810] = -4.49491573E-01;
    COFD[4811] = 1.91438508E-02;
    COFD[4812] = -2.05466616E+01;
    COFD[4813] = 5.18417470E+00;
    COFD[4814] = -4.49491573E-01;
    COFD[4815] = 1.91438508E-02;
    COFD[4816] = -2.05553656E+01;
    COFD[4817] = 5.18417470E+00;
    COFD[4818] = -4.49491573E-01;
    COFD[4819] = 1.91438508E-02;
    COFD[4820] = -1.90219707E+01;
    COFD[4821] = 4.84384483E+00;
    COFD[4822] = -4.10265575E-01;
    COFD[4823] = 1.76414287E-02;
    COFD[4824] = -2.01095186E+01;
    COFD[4825] = 4.41511629E+00;
    COFD[4826] = -2.84086963E-01;
    COFD[4827] = 9.37586971E-03;
    COFD[4828] = -2.11458678E+01;
    COFD[4829] = 5.42846112E+00;
    COFD[4830] = -4.74321870E-01;
    COFD[4831] = 1.99459749E-02;
    COFD[4832] = -2.22343363E+01;
    COFD[4833] = 5.54251230E+00;
    COFD[4834] = -4.70946314E-01;
    COFD[4835] = 1.90785869E-02;
    COFD[4836] = -2.03099025E+01;
    COFD[4837] = 5.11106992E+00;
    COFD[4838] = -4.42047129E-01;
    COFD[4839] = 1.89042990E-02;
    COFD[4840] = -2.01199204E+01;
    COFD[4841] = 4.41511629E+00;
    COFD[4842] = -2.84086963E-01;
    COFD[4843] = 9.37586971E-03;
    COFD[4844] = -2.11458678E+01;
    COFD[4845] = 5.42846112E+00;
    COFD[4846] = -4.74321870E-01;
    COFD[4847] = 1.99459749E-02;
    COFD[4848] = -2.11722423E+01;
    COFD[4849] = 5.42846112E+00;
    COFD[4850] = -4.74321870E-01;
    COFD[4851] = 1.99459749E-02;
    COFD[4852] = -2.09409936E+01;
    COFD[4853] = 4.82184721E+00;
    COFD[4854] = -3.48128875E-01;
    COFD[4855] = 1.25918978E-02;
    COFD[4856] = -2.11430338E+01;
    COFD[4857] = 5.41773516E+00;
    COFD[4858] = -4.73414338E-01;
    COFD[4859] = 1.99258685E-02;
    COFD[4860] = -2.03227406E+01;
    COFD[4861] = 4.50250781E+00;
    COFD[4862] = -2.97622106E-01;
    COFD[4863] = 1.00481473E-02;
    COFD[4864] = -2.22613837E+01;
    COFD[4865] = 5.53139819E+00;
    COFD[4866] = -4.68828555E-01;
    COFD[4867] = 1.89597887E-02;
    COFD[4868] = -2.22510033E+01;
    COFD[4869] = 5.53139819E+00;
    COFD[4870] = -4.68828555E-01;
    COFD[4871] = 1.89597887E-02;
    COFD[4872] = -2.09224206E+01;
    COFD[4873] = 4.72895031E+00;
    COFD[4874] = -3.33332771E-01;
    COFD[4875] = 1.18431478E-02;
    COFD[4876] = -2.09455936E+01;
    COFD[4877] = 4.72895031E+00;
    COFD[4878] = -3.33332771E-01;
    COFD[4879] = 1.18431478E-02;
    COFD[4880] = -2.04268153E+01;
    COFD[4881] = 4.50250781E+00;
    COFD[4882] = -2.97622106E-01;
    COFD[4883] = 1.00481473E-02;
    COFD[4884] = -2.04320309E+01;
    COFD[4885] = 4.50250781E+00;
    COFD[4886] = -2.97622106E-01;
    COFD[4887] = 1.00481473E-02;
    COFD[4888] = -2.20431319E+01;
    COFD[4889] = 5.49663315E+00;
    COFD[4890] = -4.61182837E-01;
    COFD[4891] = 1.85035558E-02;
    COFD[4892] = -2.12840631E+01;
    COFD[4893] = 5.47935225E+00;
    COFD[4894] = -4.80056796E-01;
    COFD[4895] = 2.01607180E-02;
    COFD[4896] = -2.20551771E+01;
    COFD[4897] = 5.49663315E+00;
    COFD[4898] = -4.61182837E-01;
    COFD[4899] = 1.85035558E-02;
    COFD[4900] = -2.09285776E+01;
    COFD[4901] = 4.72895031E+00;
    COFD[4902] = -3.33332771E-01;
    COFD[4903] = 1.18431478E-02;
    COFD[4904] = -2.20818886E+01;
    COFD[4905] = 5.34774760E+00;
    COFD[4906] = -4.34239753E-01;
    COFD[4907] = 1.70320676E-02;
    COFD[4908] = -2.20574820E+01;
    COFD[4909] = 5.56049839E+00;
    COFD[4910] = -4.74367872E-01;
    COFD[4911] = 1.92702787E-02;
    COFD[4912] = -2.07301477E+01;
    COFD[4913] = 4.56211059E+00;
    COFD[4914] = -3.06895158E-01;
    COFD[4915] = 1.05100393E-02;
    COFD[4916] = -2.09285776E+01;
    COFD[4917] = 4.72895031E+00;
    COFD[4918] = -3.33332771E-01;
    COFD[4919] = 1.18431478E-02;
    COFD[4920] = -2.08011592E+01;
    COFD[4921] = 4.56211059E+00;
    COFD[4922] = -3.06895158E-01;
    COFD[4923] = 1.05100393E-02;
    COFD[4924] = -2.22359646E+01;
    COFD[4925] = 5.58360799E+00;
    COFD[4926] = -4.82701436E-01;
    COFD[4927] = 1.98437922E-02;
    COFD[4928] = -2.19739638E+01;
    COFD[4929] = 5.27258289E+00;
    COFD[4930] = -4.21502790E-01;
    COFD[4931] = 1.63611949E-02;
    COFD[4932] = -2.19808152E+01;
    COFD[4933] = 5.27258289E+00;
    COFD[4934] = -4.21502790E-01;
    COFD[4935] = 1.63611949E-02;
    COFD[4936] = -2.20411190E+01;
    COFD[4937] = 5.31412694E+00;
    COFD[4938] = -4.28473898E-01;
    COFD[4939] = 1.67264841E-02;
    COFD[4940] = -2.22769618E+01;
    COFD[4941] = 5.36643605E+00;
    COFD[4942] = -4.37440735E-01;
    COFD[4943] = 1.72016388E-02;
    COFD[4944] = -2.08639466E+01;
    COFD[4945] = 4.50409026E+00;
    COFD[4946] = -2.97868419E-01;
    COFD[4947] = 1.00604224E-02;
    COFD[4948] = -2.19503032E+01;
    COFD[4949] = 5.14570932E+00;
    COFD[4950] = -3.99877142E-01;
    COFD[4951] = 1.52199557E-02;
    COFD[4952] = -2.19534987E+01;
    COFD[4953] = 5.15446948E+00;
    COFD[4954] = -4.01332769E-01;
    COFD[4955] = 1.52956262E-02;
    COFD[4956] = -2.19576037E+01;
    COFD[4957] = 5.15446948E+00;
    COFD[4958] = -4.01332769E-01;
    COFD[4959] = 1.52956262E-02;
    COFD[4960] = -2.21147341E+01;
    COFD[4961] = 5.16679492E+00;
    COFD[4962] = -4.03405751E-01;
    COFD[4963] = 1.54041741E-02;
    COFD[4964] = -2.07861367E+01;
    COFD[4965] = 4.42680848E+00;
    COFD[4966] = -2.85885288E-01;
    COFD[4967] = 9.46483934E-03;
    COFD[4968] = -2.17740719E+01;
    COFD[4969] = 4.93496210E+00;
    COFD[4970] = -3.65981745E-01;
    COFD[4971] = 1.34912948E-02;
    COFD[4972] = -2.19075860E+01;
    COFD[4973] = 4.99907484E+00;
    COFD[4974] = -3.76094627E-01;
    COFD[4975] = 1.40009262E-02;
    COFD[4976] = -2.12501716E+01;
    COFD[4977] = 4.70506024E+00;
    COFD[4978] = -3.29547212E-01;
    COFD[4979] = 1.16521630E-02;
    COFD[4980] = -2.09119213E+01;
    COFD[4981] = 4.48108132E+00;
    COFD[4982] = -2.94289899E-01;
    COFD[4983] = 9.88218297E-03;
    COFD[4984] = -2.13838498E+01;
    COFD[4985] = 4.61201872E+00;
    COFD[4986] = -3.14803338E-01;
    COFD[4987] = 1.09082984E-02;
    COFD[4988] = -2.13854464E+01;
    COFD[4989] = 4.61201872E+00;
    COFD[4990] = -3.14803338E-01;
    COFD[4991] = 1.09082984E-02;
    COFD[4992] = -2.04649069E+01;
    COFD[4993] = 5.18856872E+00;
    COFD[4994] = -4.50001829E-01;
    COFD[4995] = 1.91636142E-02;
    COFD[4996] = -1.93925667E+01;
    COFD[4997] = 4.98286777E+00;
    COFD[4998] = -4.26970814E-01;
    COFD[4999] = 1.83122917E-02;
    COFD[5000] = -1.59632479E+01;
    COFD[5001] = 4.07051484E+00;
    COFD[5002] = -3.16303109E-01;
    COFD[5003] = 1.38259377E-02;
    COFD[5004] = -1.96914944E+01;
    COFD[5005] = 5.54637286E+00;
    COFD[5006] = -4.87070324E-01;
    COFD[5007] = 2.03983467E-02;
    COFD[5008] = -1.94151822E+01;
    COFD[5009] = 4.98286777E+00;
    COFD[5010] = -4.26970814E-01;
    COFD[5011] = 1.83122917E-02;
    COFD[5012] = -1.80862867E+01;
    COFD[5013] = 3.69199168E+00;
    COFD[5014] = -1.74005516E-01;
    COFD[5015] = 3.97694372E-03;
    COFD[5016] = -2.08463209E+01;
    COFD[5017] = 5.32244593E+00;
    COFD[5018] = -4.65829403E-01;
    COFD[5019] = 1.97895274E-02;
    COFD[5020] = -2.08554914E+01;
    COFD[5021] = 5.32244593E+00;
    COFD[5022] = -4.65829403E-01;
    COFD[5023] = 1.97895274E-02;
    COFD[5024] = -2.08642748E+01;
    COFD[5025] = 5.32244593E+00;
    COFD[5026] = -4.65829403E-01;
    COFD[5027] = 1.97895274E-02;
    COFD[5028] = -1.93137183E+01;
    COFD[5029] = 4.98286777E+00;
    COFD[5030] = -4.26970814E-01;
    COFD[5031] = 1.83122917E-02;
    COFD[5032] = -1.93946947E+01;
    COFD[5033] = 4.10954793E+00;
    COFD[5034] = -2.37523329E-01;
    COFD[5035] = 7.08858141E-03;
    COFD[5036] = -2.14048982E+01;
    COFD[5037] = 5.54007827E+00;
    COFD[5038] = -4.86434511E-01;
    COFD[5039] = 2.03779006E-02;
    COFD[5040] = -2.21083035E+01;
    COFD[5041] = 5.48540187E+00;
    COFD[5042] = -4.58962148E-01;
    COFD[5043] = 1.83770355E-02;
    COFD[5044] = -2.04949373E+01;
    COFD[5045] = 5.19614628E+00;
    COFD[5046] = -4.50889164E-01;
    COFD[5047] = 1.91983328E-02;
    COFD[5048] = -1.94051843E+01;
    COFD[5049] = 4.10954793E+00;
    COFD[5050] = -2.37523329E-01;
    COFD[5051] = 7.08858141E-03;
    COFD[5052] = -2.14048982E+01;
    COFD[5053] = 5.54007827E+00;
    COFD[5054] = -4.86434511E-01;
    COFD[5055] = 2.03779006E-02;
    COFD[5056] = -2.14314090E+01;
    COFD[5057] = 5.54007827E+00;
    COFD[5058] = -4.86434511E-01;
    COFD[5059] = 2.03779006E-02;
    COFD[5060] = -2.04451935E+01;
    COFD[5061] = 4.60682543E+00;
    COFD[5062] = -3.13971634E-01;
    COFD[5063] = 1.08661011E-02;
    COFD[5064] = -2.13881945E+01;
    COFD[5065] = 5.52422470E+00;
    COFD[5066] = -4.84872944E-01;
    COFD[5067] = 2.03298213E-02;
    COFD[5068] = -1.96653154E+01;
    COFD[5069] = 4.22062499E+00;
    COFD[5070] = -2.54326872E-01;
    COFD[5071] = 7.91017784E-03;
    COFD[5072] = -2.21333822E+01;
    COFD[5073] = 5.47072190E+00;
    COFD[5074] = -4.56301261E-01;
    COFD[5075] = 1.82313566E-02;
    COFD[5076] = -2.21229141E+01;
    COFD[5077] = 5.47072190E+00;
    COFD[5078] = -4.56301261E-01;
    COFD[5079] = 1.82313566E-02;
    COFD[5080] = -2.04033972E+01;
    COFD[5081] = 4.50250781E+00;
    COFD[5082] = -2.97622106E-01;
    COFD[5083] = 1.00481473E-02;
    COFD[5084] = -2.04268153E+01;
    COFD[5085] = 4.50250781E+00;
    COFD[5086] = -2.97622106E-01;
    COFD[5087] = 1.00481473E-02;
    COFD[5088] = -1.97704178E+01;
    COFD[5089] = 4.22062499E+00;
    COFD[5090] = -2.54326872E-01;
    COFD[5091] = 7.91017784E-03;
    COFD[5092] = -1.97756908E+01;
    COFD[5093] = 4.22062499E+00;
    COFD[5094] = -2.54326872E-01;
    COFD[5095] = 7.91017784E-03;
    COFD[5096] = -2.18318278E+01;
    COFD[5097] = 5.40298848E+00;
    COFD[5098] = -4.43954594E-01;
    COFD[5099] = 1.75542998E-02;
    COFD[5100] = -2.14782277E+01;
    COFD[5101] = 5.56978987E+00;
    COFD[5102] = -4.89141980E-01;
    COFD[5103] = 2.04499210E-02;
    COFD[5104] = -2.18439681E+01;
    COFD[5105] = 5.40298848E+00;
    COFD[5106] = -4.43954594E-01;
    COFD[5107] = 1.75542998E-02;
    COFD[5108] = -2.04096182E+01;
    COFD[5109] = 4.50250781E+00;
    COFD[5110] = -2.97622106E-01;
    COFD[5111] = 1.00481473E-02;
    COFD[5112] = -2.17934580E+01;
    COFD[5113] = 5.21869603E+00;
    COFD[5114] = -4.12084772E-01;
    COFD[5115] = 1.58573035E-02;
    COFD[5116] = -2.19162360E+01;
    COFD[5117] = 5.49906960E+00;
    COFD[5118] = -4.61793001E-01;
    COFD[5119] = 1.85415189E-02;
    COFD[5120] = -2.01097379E+01;
    COFD[5121] = 4.29138907E+00;
    COFD[5122] = -2.65108149E-01;
    COFD[5123] = 8.43949637E-03;
    COFD[5124] = -2.04096182E+01;
    COFD[5125] = 4.50250781E+00;
    COFD[5126] = -2.97622106E-01;
    COFD[5127] = 1.00481473E-02;
    COFD[5128] = -2.01815677E+01;
    COFD[5129] = 4.29138907E+00;
    COFD[5130] = -2.65108149E-01;
    COFD[5131] = 8.43949637E-03;
    COFD[5132] = -2.22394964E+01;
    COFD[5133] = 5.58129885E+00;
    COFD[5134] = -4.78532921E-01;
    COFD[5135] = 1.95095699E-02;
    COFD[5136] = -2.16722314E+01;
    COFD[5137] = 5.13708607E+00;
    COFD[5138] = -3.98445708E-01;
    COFD[5139] = 1.51455626E-02;
    COFD[5140] = -2.16791513E+01;
    COFD[5141] = 5.13708607E+00;
    COFD[5142] = -3.98445708E-01;
    COFD[5143] = 1.51455626E-02;
    COFD[5144] = -2.17407419E+01;
    COFD[5145] = 5.17945041E+00;
    COFD[5146] = -4.05514689E-01;
    COFD[5147] = 1.55141412E-02;
    COFD[5148] = -2.19919464E+01;
    COFD[5149] = 5.23595129E+00;
    COFD[5150] = -4.15079064E-01;
    COFD[5151] = 1.60168286E-02;
    COFD[5152] = -2.02246117E+01;
    COFD[5153] = 4.22278378E+00;
    COFD[5154] = -2.54653500E-01;
    COFD[5155] = 7.92616085E-03;
    COFD[5156] = -2.15102238E+01;
    COFD[5157] = 4.94878244E+00;
    COFD[5158] = -3.68158605E-01;
    COFD[5159] = 1.36008797E-02;
    COFD[5160] = -2.15236645E+01;
    COFD[5161] = 4.96219227E+00;
    COFD[5162] = -3.70270843E-01;
    COFD[5163] = 1.37072211E-02;
    COFD[5164] = -2.15278182E+01;
    COFD[5165] = 4.96219227E+00;
    COFD[5166] = -3.70270843E-01;
    COFD[5167] = 1.37072211E-02;
    COFD[5168] = -2.17094710E+01;
    COFD[5169] = 4.98271982E+00;
    COFD[5170] = -3.73502341E-01;
    COFD[5171] = 1.38698700E-02;
    COFD[5172] = -2.00940426E+01;
    COFD[5173] = 4.12245214E+00;
    COFD[5174] = -2.39476227E-01;
    COFD[5175] = 7.18400558E-03;
    COFD[5176] = -2.12888403E+01;
    COFD[5177] = 4.71644372E+00;
    COFD[5178] = -3.31349990E-01;
    COFD[5179] = 1.17430818E-02;
    COFD[5180] = -2.14039230E+01;
    COFD[5181] = 4.77238689E+00;
    COFD[5182] = -3.40265855E-01;
    COFD[5183] = 1.21942137E-02;
    COFD[5184] = -2.07343778E+01;
    COFD[5185] = 4.47491202E+00;
    COFD[5186] = -2.93331059E-01;
    COFD[5187] = 9.83445305E-03;
    COFD[5188] = -2.02563322E+01;
    COFD[5189] = 4.19160608E+00;
    COFD[5190] = -2.49936771E-01;
    COFD[5191] = 7.69538319E-03;
    COFD[5192] = -2.08236367E+01;
    COFD[5193] = 4.35920123E+00;
    COFD[5194] = -2.75491273E-01;
    COFD[5195] = 8.95100289E-03;
    COFD[5196] = -2.08252570E+01;
    COFD[5197] = 4.35920123E+00;
    COFD[5198] = -2.75491273E-01;
    COFD[5199] = 8.95100289E-03;
    COFD[5200] = -2.04688382E+01;
    COFD[5201] = 5.18856872E+00;
    COFD[5202] = -4.50001829E-01;
    COFD[5203] = 1.91636142E-02;
    COFD[5204] = -1.93952366E+01;
    COFD[5205] = 4.98286777E+00;
    COFD[5206] = -4.26970814E-01;
    COFD[5207] = 1.83122917E-02;
    COFD[5208] = -1.59636793E+01;
    COFD[5209] = 4.07051484E+00;
    COFD[5210] = -3.16303109E-01;
    COFD[5211] = 1.38259377E-02;
    COFD[5212] = -1.96917146E+01;
    COFD[5213] = 5.54637286E+00;
    COFD[5214] = -4.87070324E-01;
    COFD[5215] = 2.03983467E-02;
    COFD[5216] = -1.94179760E+01;
    COFD[5217] = 4.98286777E+00;
    COFD[5218] = -4.26970814E-01;
    COFD[5219] = 1.83122917E-02;
    COFD[5220] = -1.80892005E+01;
    COFD[5221] = 3.69199168E+00;
    COFD[5222] = -1.74005516E-01;
    COFD[5223] = 3.97694372E-03;
    COFD[5224] = -2.08505864E+01;
    COFD[5225] = 5.32244593E+00;
    COFD[5226] = -4.65829403E-01;
    COFD[5227] = 1.97895274E-02;
    COFD[5228] = -2.08598363E+01;
    COFD[5229] = 5.32244593E+00;
    COFD[5230] = -4.65829403E-01;
    COFD[5231] = 1.97895274E-02;
    COFD[5232] = -2.08686970E+01;
    COFD[5233] = 5.32244593E+00;
    COFD[5234] = -4.65829403E-01;
    COFD[5235] = 1.97895274E-02;
    COFD[5236] = -1.93159978E+01;
    COFD[5237] = 4.98286777E+00;
    COFD[5238] = -4.26970814E-01;
    COFD[5239] = 1.83122917E-02;
    COFD[5240] = -1.93987136E+01;
    COFD[5241] = 4.10954793E+00;
    COFD[5242] = -2.37523329E-01;
    COFD[5243] = 7.08858141E-03;
    COFD[5244] = -2.14073140E+01;
    COFD[5245] = 5.54007827E+00;
    COFD[5246] = -4.86434511E-01;
    COFD[5247] = 2.03779006E-02;
    COFD[5248] = -2.21134005E+01;
    COFD[5249] = 5.48540187E+00;
    COFD[5250] = -4.58962148E-01;
    COFD[5251] = 1.83770355E-02;
    COFD[5252] = -2.04988684E+01;
    COFD[5253] = 5.19614628E+00;
    COFD[5254] = -4.50889164E-01;
    COFD[5255] = 1.91983328E-02;
    COFD[5256] = -1.94092888E+01;
    COFD[5257] = 4.10954793E+00;
    COFD[5258] = -2.37523329E-01;
    COFD[5259] = 7.08858141E-03;
    COFD[5260] = -2.14073140E+01;
    COFD[5261] = 5.54007827E+00;
    COFD[5262] = -4.86434511E-01;
    COFD[5263] = 2.03779006E-02;
    COFD[5264] = -2.14339566E+01;
    COFD[5265] = 5.54007827E+00;
    COFD[5266] = -4.86434511E-01;
    COFD[5267] = 2.03779006E-02;
    COFD[5268] = -2.04493813E+01;
    COFD[5269] = 4.60682543E+00;
    COFD[5270] = -3.13971634E-01;
    COFD[5271] = 1.08661011E-02;
    COFD[5272] = -2.13908698E+01;
    COFD[5273] = 5.52422470E+00;
    COFD[5274] = -4.84872944E-01;
    COFD[5275] = 2.03298213E-02;
    COFD[5276] = -1.96695844E+01;
    COFD[5277] = 4.22062499E+00;
    COFD[5278] = -2.54326872E-01;
    COFD[5279] = 7.91017784E-03;
    COFD[5280] = -2.21374903E+01;
    COFD[5281] = 5.47072190E+00;
    COFD[5282] = -4.56301261E-01;
    COFD[5283] = 1.82313566E-02;
    COFD[5284] = -2.21269367E+01;
    COFD[5285] = 5.47072190E+00;
    COFD[5286] = -4.56301261E-01;
    COFD[5287] = 1.82313566E-02;
    COFD[5288] = -2.04083729E+01;
    COFD[5289] = 4.50250781E+00;
    COFD[5290] = -2.97622106E-01;
    COFD[5291] = 1.00481473E-02;
    COFD[5292] = -2.04320309E+01;
    COFD[5293] = 4.50250781E+00;
    COFD[5294] = -2.97622106E-01;
    COFD[5295] = 1.00481473E-02;
    COFD[5296] = -1.97756908E+01;
    COFD[5297] = 4.22062499E+00;
    COFD[5298] = -2.54326872E-01;
    COFD[5299] = 7.91017784E-03;
    COFD[5300] = -1.97810200E+01;
    COFD[5301] = 4.22062499E+00;
    COFD[5302] = -2.54326872E-01;
    COFD[5303] = 7.91017784E-03;
    COFD[5304] = -2.18355800E+01;
    COFD[5305] = 5.40298848E+00;
    COFD[5306] = -4.43954594E-01;
    COFD[5307] = 1.75542998E-02;
    COFD[5308] = -2.14831394E+01;
    COFD[5309] = 5.56978987E+00;
    COFD[5310] = -4.89141980E-01;
    COFD[5311] = 2.04499210E-02;
    COFD[5312] = -2.18478129E+01;
    COFD[5313] = 5.40298848E+00;
    COFD[5314] = -4.43954594E-01;
    COFD[5315] = 1.75542998E-02;
    COFD[5316] = -2.04146565E+01;
    COFD[5317] = 4.50250781E+00;
    COFD[5318] = -2.97622106E-01;
    COFD[5319] = 1.00481473E-02;
    COFD[5320] = -2.17984365E+01;
    COFD[5321] = 5.21869603E+00;
    COFD[5322] = -4.12084772E-01;
    COFD[5323] = 1.58573035E-02;
    COFD[5324] = -2.19201709E+01;
    COFD[5325] = 5.49906960E+00;
    COFD[5326] = -4.61793001E-01;
    COFD[5327] = 1.85415189E-02;
    COFD[5328] = -2.01148973E+01;
    COFD[5329] = 4.29138907E+00;
    COFD[5330] = -2.65108149E-01;
    COFD[5331] = 8.43949637E-03;
    COFD[5332] = -2.04146565E+01;
    COFD[5333] = 4.50250781E+00;
    COFD[5334] = -2.97622106E-01;
    COFD[5335] = 1.00481473E-02;
    COFD[5336] = -2.01875290E+01;
    COFD[5337] = 4.29138907E+00;
    COFD[5338] = -2.65108149E-01;
    COFD[5339] = 8.43949637E-03;
    COFD[5340] = -2.22442100E+01;
    COFD[5341] = 5.58129885E+00;
    COFD[5342] = -4.78532921E-01;
    COFD[5343] = 1.95095699E-02;
    COFD[5344] = -2.16770134E+01;
    COFD[5345] = 5.13708607E+00;
    COFD[5346] = -3.98445708E-01;
    COFD[5347] = 1.51455626E-02;
    COFD[5348] = -2.16840004E+01;
    COFD[5349] = 5.13708607E+00;
    COFD[5350] = -3.98445708E-01;
    COFD[5351] = 1.51455626E-02;
    COFD[5352] = -2.17456564E+01;
    COFD[5353] = 5.17945041E+00;
    COFD[5354] = -4.05514689E-01;
    COFD[5355] = 1.55141412E-02;
    COFD[5356] = -2.19969874E+01;
    COFD[5357] = 5.23595129E+00;
    COFD[5358] = -4.15079064E-01;
    COFD[5359] = 1.60168286E-02;
    COFD[5360] = -2.02311039E+01;
    COFD[5361] = 4.22278378E+00;
    COFD[5362] = -2.54653500E-01;
    COFD[5363] = 7.92616085E-03;
    COFD[5364] = -2.15158669E+01;
    COFD[5365] = 4.94878244E+00;
    COFD[5366] = -3.68158605E-01;
    COFD[5367] = 1.36008797E-02;
    COFD[5368] = -2.15293564E+01;
    COFD[5369] = 4.96219227E+00;
    COFD[5370] = -3.70270843E-01;
    COFD[5371] = 1.37072211E-02;
    COFD[5372] = -2.15335578E+01;
    COFD[5373] = 4.96219227E+00;
    COFD[5374] = -3.70270843E-01;
    COFD[5375] = 1.37072211E-02;
    COFD[5376] = -2.17152574E+01;
    COFD[5377] = 4.98271982E+00;
    COFD[5378] = -3.73502341E-01;
    COFD[5379] = 1.38698700E-02;
    COFD[5380] = -2.01009567E+01;
    COFD[5381] = 4.12245214E+00;
    COFD[5382] = -2.39476227E-01;
    COFD[5383] = 7.18400558E-03;
    COFD[5384] = -2.12951225E+01;
    COFD[5385] = 4.71644372E+00;
    COFD[5386] = -3.31349990E-01;
    COFD[5387] = 1.17430818E-02;
    COFD[5388] = -2.14102422E+01;
    COFD[5389] = 4.77238689E+00;
    COFD[5390] = -3.40265855E-01;
    COFD[5391] = 1.21942137E-02;
    COFD[5392] = -2.07407334E+01;
    COFD[5393] = 4.47491202E+00;
    COFD[5394] = -2.93331059E-01;
    COFD[5395] = 9.83445305E-03;
    COFD[5396] = -2.02631076E+01;
    COFD[5397] = 4.19160608E+00;
    COFD[5398] = -2.49936771E-01;
    COFD[5399] = 7.69538319E-03;
    COFD[5400] = -2.08308042E+01;
    COFD[5401] = 4.35920123E+00;
    COFD[5402] = -2.75491273E-01;
    COFD[5403] = 8.95100289E-03;
    COFD[5404] = -2.08324480E+01;
    COFD[5405] = 4.35920123E+00;
    COFD[5406] = -2.75491273E-01;
    COFD[5407] = 8.95100289E-03;
    COFD[5408] = -1.83039618E+01;
    COFD[5409] = 4.47952077E+00;
    COFD[5410] = -3.66569471E-01;
    COFD[5411] = 1.58916129E-02;
    COFD[5412] = -1.72286007E+01;
    COFD[5413] = 4.24084025E+00;
    COFD[5414] = -3.37428619E-01;
    COFD[5415] = 1.47032793E-02;
    COFD[5416] = -1.39315266E+01;
    COFD[5417] = 3.30394764E+00;
    COFD[5418] = -2.17920112E-01;
    COFD[5419] = 9.60284243E-03;
    COFD[5420] = -1.79310765E+01;
    COFD[5421] = 4.98037650E+00;
    COFD[5422] = -4.26676911E-01;
    COFD[5423] = 1.83007231E-02;
    COFD[5424] = -1.72473011E+01;
    COFD[5425] = 4.24084025E+00;
    COFD[5426] = -3.37428619E-01;
    COFD[5427] = 1.47032793E-02;
    COFD[5428] = -2.09565916E+01;
    COFD[5429] = 5.18380539E+00;
    COFD[5430] = -4.06234719E-01;
    COFD[5431] = 1.55515345E-02;
    COFD[5432] = -1.86507213E+01;
    COFD[5433] = 4.60874797E+00;
    COFD[5434] = -3.82368716E-01;
    COFD[5435] = 1.65370164E-02;
    COFD[5436] = -1.86576191E+01;
    COFD[5437] = 4.60874797E+00;
    COFD[5438] = -3.82368716E-01;
    COFD[5439] = 1.65370164E-02;
    COFD[5440] = -1.86641962E+01;
    COFD[5441] = 4.60874797E+00;
    COFD[5442] = -3.82368716E-01;
    COFD[5443] = 1.65370164E-02;
    COFD[5444] = -1.71623017E+01;
    COFD[5445] = 4.24084025E+00;
    COFD[5446] = -3.37428619E-01;
    COFD[5447] = 1.47032793E-02;
    COFD[5448] = -2.16718247E+01;
    COFD[5449] = 5.36811769E+00;
    COFD[5450] = -4.37727086E-01;
    COFD[5451] = 1.72167686E-02;
    COFD[5452] = -1.95548230E+01;
    COFD[5453] = 4.97133070E+00;
    COFD[5454] = -4.25604177E-01;
    COFD[5455] = 1.82582594E-02;
    COFD[5456] = -2.13961414E+01;
    COFD[5457] = 5.46685775E+00;
    COFD[5458] = -4.78665416E-01;
    COFD[5459] = 2.01093915E-02;
    COFD[5460] = -1.83296965E+01;
    COFD[5461] = 4.48570999E+00;
    COFD[5462] = -3.67301524E-01;
    COFD[5463] = 1.59204254E-02;
    COFD[5464] = -2.16798265E+01;
    COFD[5465] = 5.36811769E+00;
    COFD[5466] = -4.37727086E-01;
    COFD[5467] = 1.72167686E-02;
    COFD[5468] = -1.95548230E+01;
    COFD[5469] = 4.97133070E+00;
    COFD[5470] = -4.25604177E-01;
    COFD[5471] = 1.82582594E-02;
    COFD[5472] = -1.95770968E+01;
    COFD[5473] = 4.97133070E+00;
    COFD[5474] = -4.25604177E-01;
    COFD[5475] = 1.82582594E-02;
    COFD[5476] = -2.18848136E+01;
    COFD[5477] = 5.51302074E+00;
    COFD[5478] = -4.65263979E-01;
    COFD[5479] = 1.87580679E-02;
    COFD[5480] = -1.95154079E+01;
    COFD[5481] = 4.94787350E+00;
    COFD[5482] = -4.22829292E-01;
    COFD[5483] = 1.81487163E-02;
    COFD[5484] = -2.17547312E+01;
    COFD[5485] = 5.40298848E+00;
    COFD[5486] = -4.43954594E-01;
    COFD[5487] = 1.75542998E-02;
    COFD[5488] = -2.15206146E+01;
    COFD[5489] = 5.48426911E+00;
    COFD[5490] = -4.80606512E-01;
    COFD[5491] = 2.01811046E-02;
    COFD[5492] = -2.15126310E+01;
    COFD[5493] = 5.48426911E+00;
    COFD[5494] = -4.80606512E-01;
    COFD[5495] = 2.01811046E-02;
    COFD[5496] = -2.20262793E+01;
    COFD[5497] = 5.49663315E+00;
    COFD[5498] = -4.61182837E-01;
    COFD[5499] = 1.85035558E-02;
    COFD[5500] = -2.20431319E+01;
    COFD[5501] = 5.49663315E+00;
    COFD[5502] = -4.61182837E-01;
    COFD[5503] = 1.85035558E-02;
    COFD[5504] = -2.18318278E+01;
    COFD[5505] = 5.40298848E+00;
    COFD[5506] = -4.43954594E-01;
    COFD[5507] = 1.75542998E-02;
    COFD[5508] = -2.18355800E+01;
    COFD[5509] = 5.40298848E+00;
    COFD[5510] = -4.43954594E-01;
    COFD[5511] = 1.75542998E-02;
    COFD[5512] = -2.15453676E+01;
    COFD[5513] = 5.55313619E+00;
    COFD[5514] = -4.87753729E-01;
    COFD[5515] = 2.04203421E-02;
    COFD[5516] = -1.96068586E+01;
    COFD[5517] = 5.02434088E+00;
    COFD[5518] = -4.31889635E-01;
    COFD[5519] = 1.85072024E-02;
    COFD[5520] = -2.15547727E+01;
    COFD[5521] = 5.55313619E+00;
    COFD[5522] = -4.87753729E-01;
    COFD[5523] = 2.04203421E-02;
    COFD[5524] = -2.20307777E+01;
    COFD[5525] = 5.49663315E+00;
    COFD[5526] = -4.61182837E-01;
    COFD[5527] = 1.85035558E-02;
    COFD[5528] = -2.20228343E+01;
    COFD[5529] = 5.61211028E+00;
    COFD[5530] = -4.90893171E-01;
    COFD[5531] = 2.03793118E-02;
    COFD[5532] = -2.11427744E+01;
    COFD[5533] = 5.43893233E+00;
    COFD[5534] = -4.75546039E-01;
    COFD[5535] = 1.99938690E-02;
    COFD[5536] = -2.20547393E+01;
    COFD[5537] = 5.42445100E+00;
    COFD[5538] = -4.47918761E-01;
    COFD[5539] = 1.77729995E-02;
    COFD[5540] = -2.20307777E+01;
    COFD[5541] = 5.49663315E+00;
    COFD[5542] = -4.61182837E-01;
    COFD[5543] = 1.85035558E-02;
    COFD[5544] = -2.21051793E+01;
    COFD[5545] = 5.42445100E+00;
    COFD[5546] = -4.47918761E-01;
    COFD[5547] = 1.77729995E-02;
    COFD[5548] = -2.11070024E+01;
    COFD[5549] = 5.37047121E+00;
    COFD[5550] = -4.70282612E-01;
    COFD[5551] = 1.99109322E-02;
    COFD[5552] = -2.20555979E+01;
    COFD[5553] = 5.59649805E+00;
    COFD[5554] = -4.86750336E-01;
    COFD[5555] = 2.01151498E-02;
    COFD[5556] = -2.20606550E+01;
    COFD[5557] = 5.59649805E+00;
    COFD[5558] = -4.86750336E-01;
    COFD[5559] = 2.01151498E-02;
    COFD[5560] = -2.20511271E+01;
    COFD[5561] = 5.60809037E+00;
    COFD[5562] = -4.89400803E-01;
    COFD[5563] = 2.02760802E-02;
    COFD[5564] = -2.21795362E+01;
    COFD[5565] = 5.61233637E+00;
    COFD[5566] = -4.91419253E-01;
    COFD[5567] = 2.04216738E-02;
    COFD[5568] = -2.22462130E+01;
    COFD[5569] = 5.40356304E+00;
    COFD[5570] = -4.44060256E-01;
    COFD[5571] = 1.75601121E-02;
    COFD[5572] = -2.22801170E+01;
    COFD[5573] = 5.58507108E+00;
    COFD[5574] = -4.81395065E-01;
    COFD[5575] = 1.97276199E-02;
    COFD[5576] = -2.22609256E+01;
    COFD[5577] = 5.58490856E+00;
    COFD[5578] = -4.81588720E-01;
    COFD[5579] = 1.97445317E-02;
    COFD[5580] = -2.22638165E+01;
    COFD[5581] = 5.58490856E+00;
    COFD[5582] = -4.81588720E-01;
    COFD[5583] = 1.97445317E-02;
    COFD[5584] = -2.23950513E+01;
    COFD[5585] = 5.58492366E+00;
    COFD[5586] = -4.81921868E-01;
    COFD[5587] = 1.97721534E-02;
    COFD[5588] = -2.22709427E+01;
    COFD[5589] = 5.37360713E+00;
    COFD[5590] = -4.38661889E-01;
    COFD[5591] = 1.72661628E-02;
    COFD[5592] = -2.24990717E+01;
    COFD[5593] = 5.55026833E+00;
    COFD[5594] = -4.72437808E-01;
    COFD[5595] = 1.91625195E-02;
    COFD[5596] = -2.25347527E+01;
    COFD[5597] = 5.57238332E+00;
    COFD[5598] = -4.76605097E-01;
    COFD[5599] = 1.93951822E-02;
    COFD[5600] = -2.23655523E+01;
    COFD[5601] = 5.48956505E+00;
    COFD[5602] = -4.59770566E-01;
    COFD[5603] = 1.84227929E-02;
    COFD[5604] = -2.23265991E+01;
    COFD[5605] = 5.39645154E+00;
    COFD[5606] = -4.42708323E-01;
    COFD[5607] = 1.74846134E-02;
    COFD[5608] = -2.26089431E+01;
    COFD[5609] = 5.44867280E+00;
    COFD[5610] = -4.52284883E-01;
    COFD[5611] = 1.80110706E-02;
    COFD[5612] = -2.26099899E+01;
    COFD[5613] = 5.44867280E+00;
    COFD[5614] = -4.52284883E-01;
    COFD[5615] = 1.80110706E-02;
    COFD[5616] = -1.59884305E+01;
    COFD[5617] = 3.72220402E+00;
    COFD[5618] = -2.71150591E-01;
    COFD[5619] = 1.18665265E-02;
    COFD[5620] = -1.49500357E+01;
    COFD[5621] = 3.52327209E+00;
    COFD[5622] = -2.46286208E-01;
    COFD[5623] = 1.08285963E-02;
    COFD[5624] = -1.22004324E+01;
    COFD[5625] = 2.80725489E+00;
    COFD[5626] = -1.54291406E-01;
    COFD[5627] = 6.88290911E-03;
    COFD[5628] = -1.54460820E+01;
    COFD[5629] = 4.26819983E+00;
    COFD[5630] = -3.40766379E-01;
    COFD[5631] = 1.48393361E-02;
    COFD[5632] = -1.49718233E+01;
    COFD[5633] = 3.52327209E+00;
    COFD[5634] = -2.46286208E-01;
    COFD[5635] = 1.08285963E-02;
    COFD[5636] = -2.10440675E+01;
    COFD[5637] = 5.59806282E+00;
    COFD[5638] = -4.87109535E-01;
    COFD[5639] = 2.01370226E-02;
    COFD[5640] = -1.64169433E+01;
    COFD[5641] = 3.89309916E+00;
    COFD[5642] = -2.93528188E-01;
    COFD[5643] = 1.28463177E-02;
    COFD[5644] = -1.64255964E+01;
    COFD[5645] = 3.89309916E+00;
    COFD[5646] = -2.93528188E-01;
    COFD[5647] = 1.28463177E-02;
    COFD[5648] = -1.64338757E+01;
    COFD[5649] = 3.89309916E+00;
    COFD[5650] = -2.93528188E-01;
    COFD[5651] = 1.28463177E-02;
    COFD[5652] = -1.48738066E+01;
    COFD[5653] = 3.52327209E+00;
    COFD[5654] = -2.46286208E-01;
    COFD[5655] = 1.08285963E-02;
    COFD[5656] = -2.14204185E+01;
    COFD[5657] = 5.59268435E+00;
    COFD[5658] = -4.91232974E-01;
    COFD[5659] = 2.05064746E-02;
    COFD[5660] = -1.72572042E+01;
    COFD[5661] = 4.26063341E+00;
    COFD[5662] = -3.39848064E-01;
    COFD[5663] = 1.48021313E-02;
    COFD[5664] = -1.94485982E+01;
    COFD[5665] = 4.91446566E+00;
    COFD[5666] = -4.18837152E-01;
    COFD[5667] = 1.79893537E-02;
    COFD[5668] = -1.60261675E+01;
    COFD[5669] = 3.73312045E+00;
    COFD[5670] = -2.72579779E-01;
    COFD[5671] = 1.19290272E-02;
    COFD[5672] = -2.14303479E+01;
    COFD[5673] = 5.59268435E+00;
    COFD[5674] = -4.91232974E-01;
    COFD[5675] = 2.05064746E-02;
    COFD[5676] = -1.72572042E+01;
    COFD[5677] = 4.26063341E+00;
    COFD[5678] = -3.39848064E-01;
    COFD[5679] = 1.48021313E-02;
    COFD[5680] = -1.72828302E+01;
    COFD[5681] = 4.26063341E+00;
    COFD[5682] = -3.39848064E-01;
    COFD[5683] = 1.48021313E-02;
    COFD[5684] = -2.09241647E+01;
    COFD[5685] = 5.42316225E+00;
    COFD[5686] = -4.73702801E-01;
    COFD[5687] = 1.99217718E-02;
    COFD[5688] = -1.72316148E+01;
    COFD[5689] = 4.24011069E+00;
    COFD[5690] = -3.37339810E-01;
    COFD[5691] = 1.46996679E-02;
    COFD[5692] = -2.13796303E+01;
    COFD[5693] = 5.56978987E+00;
    COFD[5694] = -4.89141980E-01;
    COFD[5695] = 2.04499210E-02;
    COFD[5696] = -1.95836394E+01;
    COFD[5697] = 4.93449043E+00;
    COFD[5698] = -4.21243802E-01;
    COFD[5699] = 1.80859966E-02;
    COFD[5700] = -1.95737308E+01;
    COFD[5701] = 4.93449043E+00;
    COFD[5702] = -4.21243802E-01;
    COFD[5703] = 1.80859966E-02;
    COFD[5704] = -2.12621914E+01;
    COFD[5705] = 5.47935225E+00;
    COFD[5706] = -4.80056796E-01;
    COFD[5707] = 2.01607180E-02;
    COFD[5708] = -2.12840631E+01;
    COFD[5709] = 5.47935225E+00;
    COFD[5710] = -4.80056796E-01;
    COFD[5711] = 2.01607180E-02;
    COFD[5712] = -2.14782277E+01;
    COFD[5713] = 5.56978987E+00;
    COFD[5714] = -4.89141980E-01;
    COFD[5715] = 2.04499210E-02;
    COFD[5716] = -2.14831394E+01;
    COFD[5717] = 5.56978987E+00;
    COFD[5718] = -4.89141980E-01;
    COFD[5719] = 2.04499210E-02;
    COFD[5720] = -1.96068586E+01;
    COFD[5721] = 5.02434088E+00;
    COFD[5722] = -4.31889635E-01;
    COFD[5723] = 1.85072024E-02;
    COFD[5724] = -1.72414862E+01;
    COFD[5725] = 4.29808578E+00;
    COFD[5726] = -3.44235570E-01;
    COFD[5727] = 1.49727727E-02;
    COFD[5728] = -1.96183903E+01;
    COFD[5729] = 5.02434088E+00;
    COFD[5730] = -4.31889635E-01;
    COFD[5731] = 1.85072024E-02;
    COFD[5732] = -2.12680082E+01;
    COFD[5733] = 5.47935225E+00;
    COFD[5734] = -4.80056796E-01;
    COFD[5735] = 2.01607180E-02;
    COFD[5736] = -2.03116266E+01;
    COFD[5737] = 5.16758304E+00;
    COFD[5738] = -4.47606321E-01;
    COFD[5739] = 1.90728318E-02;
    COFD[5740] = -1.91367023E+01;
    COFD[5741] = 4.87703209E+00;
    COFD[5742] = -4.14222202E-01;
    COFD[5743] = 1.77987878E-02;
    COFD[5744] = -2.16544368E+01;
    COFD[5745] = 5.55511977E+00;
    COFD[5746] = -4.87927156E-01;
    COFD[5747] = 2.04245402E-02;
    COFD[5748] = -2.12680082E+01;
    COFD[5749] = 5.47935225E+00;
    COFD[5750] = -4.80056796E-01;
    COFD[5751] = 2.01607180E-02;
    COFD[5752] = -2.17211317E+01;
    COFD[5753] = 5.55511977E+00;
    COFD[5754] = -4.87927156E-01;
    COFD[5755] = 2.04245402E-02;
    COFD[5756] = -1.89718952E+01;
    COFD[5757] = 4.72476764E+00;
    COFD[5758] = -3.96306836E-01;
    COFD[5759] = 1.70964541E-02;
    COFD[5760] = -2.05357412E+01;
    COFD[5761] = 5.23500188E+00;
    COFD[5762] = -4.55417380E-01;
    COFD[5763] = 1.93744255E-02;
    COFD[5764] = -2.05422276E+01;
    COFD[5765] = 5.23500188E+00;
    COFD[5766] = -4.55417380E-01;
    COFD[5767] = 1.93744255E-02;
    COFD[5768] = -2.04251023E+01;
    COFD[5769] = 5.19993608E+00;
    COFD[5770] = -4.51334924E-01;
    COFD[5771] = 1.92158646E-02;
    COFD[5772] = -2.04750337E+01;
    COFD[5773] = 5.15745622E+00;
    COFD[5774] = -4.46648283E-01;
    COFD[5775] = 1.90458987E-02;
    COFD[5776] = -2.19764159E+01;
    COFD[5777] = 5.56943713E+00;
    COFD[5778] = -4.89114655E-01;
    COFD[5779] = 2.04494661E-02;
    COFD[5780] = -2.10849000E+01;
    COFD[5781] = 5.35335833E+00;
    COFD[5782] = -4.69065665E-01;
    COFD[5783] = 1.98989604E-02;
    COFD[5784] = -2.10575083E+01;
    COFD[5785] = 5.35019396E+00;
    COFD[5786] = -4.68809590E-01;
    COFD[5787] = 1.98941097E-02;
    COFD[5788] = -2.10613569E+01;
    COFD[5789] = 5.35019396E+00;
    COFD[5790] = -4.68809590E-01;
    COFD[5791] = 1.98941097E-02;
    COFD[5792] = -2.12027460E+01;
    COFD[5793] = 5.34410059E+00;
    COFD[5794] = -4.68233157E-01;
    COFD[5795] = 1.98777314E-02;
    COFD[5796] = -2.21307579E+01;
    COFD[5797] = 5.58979675E+00;
    COFD[5798] = -4.90962731E-01;
    COFD[5799] = 2.04987927E-02;
    COFD[5800] = -2.15322533E+01;
    COFD[5801] = 5.40448560E+00;
    COFD[5802] = -4.72711417E-01;
    COFD[5803] = 1.99362480E-02;
    COFD[5804] = -2.14935984E+01;
    COFD[5805] = 5.39257286E+00;
    COFD[5806] = -4.71929831E-01;
    COFD[5807] = 1.99331101E-02;
    COFD[5808] = -2.17210124E+01;
    COFD[5809] = 5.49225467E+00;
    COFD[5810] = -4.81478120E-01;
    COFD[5811] = 2.02123784E-02;
    COFD[5812] = -2.21024680E+01;
    COFD[5813] = 5.57482264E+00;
    COFD[5814] = -4.89554775E-01;
    COFD[5815] = 2.04583790E-02;
    COFD[5816] = -2.22093870E+01;
    COFD[5817] = 5.53457356E+00;
    COFD[5818] = -4.85892223E-01;
    COFD[5819] = 2.03611937E-02;
    COFD[5820] = -2.22108608E+01;
    COFD[5821] = 5.53457356E+00;
    COFD[5822] = -4.85892223E-01;
    COFD[5823] = 2.03611937E-02;
    COFD[5824] = -1.83137139E+01;
    COFD[5825] = 4.47952077E+00;
    COFD[5826] = -3.66569471E-01;
    COFD[5827] = 1.58916129E-02;
    COFD[5828] = -1.72357436E+01;
    COFD[5829] = 4.24084025E+00;
    COFD[5830] = -3.37428619E-01;
    COFD[5831] = 1.47032793E-02;
    COFD[5832] = -1.39328674E+01;
    COFD[5833] = 3.30394764E+00;
    COFD[5834] = -2.17920112E-01;
    COFD[5835] = 9.60284243E-03;
    COFD[5836] = -1.79317714E+01;
    COFD[5837] = 4.98037650E+00;
    COFD[5838] = -4.26676911E-01;
    COFD[5839] = 1.83007231E-02;
    COFD[5840] = -1.72547182E+01;
    COFD[5841] = 4.24084025E+00;
    COFD[5842] = -3.37428619E-01;
    COFD[5843] = 1.47032793E-02;
    COFD[5844] = -2.09642705E+01;
    COFD[5845] = 5.18380539E+00;
    COFD[5846] = -4.06234719E-01;
    COFD[5847] = 1.55515345E-02;
    COFD[5848] = -1.86611023E+01;
    COFD[5849] = 4.60874797E+00;
    COFD[5850] = -3.82368716E-01;
    COFD[5851] = 1.65370164E-02;
    COFD[5852] = -1.86681459E+01;
    COFD[5853] = 4.60874797E+00;
    COFD[5854] = -3.82368716E-01;
    COFD[5855] = 1.65370164E-02;
    COFD[5856] = -1.86748638E+01;
    COFD[5857] = 4.60874797E+00;
    COFD[5858] = -3.82368716E-01;
    COFD[5859] = 1.65370164E-02;
    COFD[5860] = -1.71685520E+01;
    COFD[5861] = 4.24084025E+00;
    COFD[5862] = -3.37428619E-01;
    COFD[5863] = 1.47032793E-02;
    COFD[5864] = -2.16817439E+01;
    COFD[5865] = 5.36811769E+00;
    COFD[5866] = -4.37727086E-01;
    COFD[5867] = 1.72167686E-02;
    COFD[5868] = -1.95613899E+01;
    COFD[5869] = 4.97133070E+00;
    COFD[5870] = -4.25604177E-01;
    COFD[5871] = 1.82582594E-02;
    COFD[5872] = -2.14079882E+01;
    COFD[5873] = 5.46685775E+00;
    COFD[5874] = -4.78665416E-01;
    COFD[5875] = 2.01093915E-02;
    COFD[5876] = -1.83394481E+01;
    COFD[5877] = 4.48570999E+00;
    COFD[5878] = -3.67301524E-01;
    COFD[5879] = 1.59204254E-02;
    COFD[5880] = -2.16899073E+01;
    COFD[5881] = 5.36811769E+00;
    COFD[5882] = -4.37727086E-01;
    COFD[5883] = 1.72167686E-02;
    COFD[5884] = -1.95613899E+01;
    COFD[5885] = 4.97133070E+00;
    COFD[5886] = -4.25604177E-01;
    COFD[5887] = 1.82582594E-02;
    COFD[5888] = -1.95839648E+01;
    COFD[5889] = 4.97133070E+00;
    COFD[5890] = -4.25604177E-01;
    COFD[5891] = 1.82582594E-02;
    COFD[5892] = -2.18950505E+01;
    COFD[5893] = 5.51302074E+00;
    COFD[5894] = -4.65263979E-01;
    COFD[5895] = 1.87580679E-02;
    COFD[5896] = -1.95225629E+01;
    COFD[5897] = 4.94787350E+00;
    COFD[5898] = -4.22829292E-01;
    COFD[5899] = 1.81487163E-02;
    COFD[5900] = -2.17651187E+01;
    COFD[5901] = 5.40298848E+00;
    COFD[5902] = -4.43954594E-01;
    COFD[5903] = 1.75542998E-02;
    COFD[5904] = -2.15307023E+01;
    COFD[5905] = 5.48426911E+00;
    COFD[5906] = -4.80606512E-01;
    COFD[5907] = 2.01811046E-02;
    COFD[5908] = -2.15225573E+01;
    COFD[5909] = 5.48426911E+00;
    COFD[5910] = -4.80606512E-01;
    COFD[5911] = 2.01811046E-02;
    COFD[5912] = -2.20379206E+01;
    COFD[5913] = 5.49663315E+00;
    COFD[5914] = -4.61182837E-01;
    COFD[5915] = 1.85035558E-02;
    COFD[5916] = -2.20551771E+01;
    COFD[5917] = 5.49663315E+00;
    COFD[5918] = -4.61182837E-01;
    COFD[5919] = 1.85035558E-02;
    COFD[5920] = -2.18439681E+01;
    COFD[5921] = 5.40298848E+00;
    COFD[5922] = -4.43954594E-01;
    COFD[5923] = 1.75542998E-02;
    COFD[5924] = -2.18478129E+01;
    COFD[5925] = 5.40298848E+00;
    COFD[5926] = -4.43954594E-01;
    COFD[5927] = 1.75542998E-02;
    COFD[5928] = -2.15547727E+01;
    COFD[5929] = 5.55313619E+00;
    COFD[5930] = -4.87753729E-01;
    COFD[5931] = 2.04203421E-02;
    COFD[5932] = -1.96183903E+01;
    COFD[5933] = 5.02434088E+00;
    COFD[5934] = -4.31889635E-01;
    COFD[5935] = 1.85072024E-02;
    COFD[5936] = -2.15643580E+01;
    COFD[5937] = 5.55313619E+00;
    COFD[5938] = -4.87753729E-01;
    COFD[5939] = 2.04203421E-02;
    COFD[5940] = -2.20425255E+01;
    COFD[5941] = 5.49663315E+00;
    COFD[5942] = -4.61182837E-01;
    COFD[5943] = 1.85035558E-02;
    COFD[5944] = -2.20344803E+01;
    COFD[5945] = 5.61211028E+00;
    COFD[5946] = -4.90893171E-01;
    COFD[5947] = 2.03793118E-02;
    COFD[5948] = -2.11525334E+01;
    COFD[5949] = 5.43893233E+00;
    COFD[5950] = -4.75546039E-01;
    COFD[5951] = 1.99938690E-02;
    COFD[5952] = -2.20666909E+01;
    COFD[5953] = 5.42445100E+00;
    COFD[5954] = -4.47918761E-01;
    COFD[5955] = 1.77729995E-02;
    COFD[5956] = -2.20425255E+01;
    COFD[5957] = 5.49663315E+00;
    COFD[5958] = -4.61182837E-01;
    COFD[5959] = 1.85035558E-02;
    COFD[5960] = -2.21184165E+01;
    COFD[5961] = 5.42445100E+00;
    COFD[5962] = -4.47918761E-01;
    COFD[5963] = 1.77729995E-02;
    COFD[5964] = -2.11181899E+01;
    COFD[5965] = 5.37047121E+00;
    COFD[5966] = -4.70282612E-01;
    COFD[5967] = 1.99109322E-02;
    COFD[5968] = -2.20669053E+01;
    COFD[5969] = 5.59649805E+00;
    COFD[5970] = -4.86750336E-01;
    COFD[5971] = 2.01151498E-02;
    COFD[5972] = -2.20720787E+01;
    COFD[5973] = 5.59649805E+00;
    COFD[5974] = -4.86750336E-01;
    COFD[5975] = 2.01151498E-02;
    COFD[5976] = -2.20626636E+01;
    COFD[5977] = 5.60809037E+00;
    COFD[5978] = -4.89400803E-01;
    COFD[5979] = 2.02760802E-02;
    COFD[5980] = -2.21912885E+01;
    COFD[5981] = 5.61233637E+00;
    COFD[5982] = -4.91419253E-01;
    COFD[5983] = 2.04216738E-02;
    COFD[5984] = -2.22602443E+01;
    COFD[5985] = 5.40356304E+00;
    COFD[5986] = -4.44060256E-01;
    COFD[5987] = 1.75601121E-02;
    COFD[5988] = -2.22928570E+01;
    COFD[5989] = 5.58507108E+00;
    COFD[5990] = -4.81395065E-01;
    COFD[5991] = 1.97276199E-02;
    COFD[5992] = -2.22737428E+01;
    COFD[5993] = 5.58490856E+00;
    COFD[5994] = -4.81588720E-01;
    COFD[5995] = 1.97445317E-02;
    COFD[5996] = -2.22767090E+01;
    COFD[5997] = 5.58490856E+00;
    COFD[5998] = -4.81588720E-01;
    COFD[5999] = 1.97445317E-02;
    COFD[6000] = -2.24080172E+01;
    COFD[6001] = 5.58492366E+00;
    COFD[6002] = -4.81921868E-01;
    COFD[6003] = 1.97721534E-02;
    COFD[6004] = -2.22855755E+01;
    COFD[6005] = 5.37360713E+00;
    COFD[6006] = -4.38661889E-01;
    COFD[6007] = 1.72661628E-02;
    COFD[6008] = -2.25127940E+01;
    COFD[6009] = 5.55026833E+00;
    COFD[6010] = -4.72437808E-01;
    COFD[6011] = 1.91625195E-02;
    COFD[6012] = -2.25485299E+01;
    COFD[6013] = 5.57238332E+00;
    COFD[6014] = -4.76605097E-01;
    COFD[6015] = 1.93951822E-02;
    COFD[6016] = -2.23793834E+01;
    COFD[6017] = 5.48956505E+00;
    COFD[6018] = -4.59770566E-01;
    COFD[6019] = 1.84227929E-02;
    COFD[6020] = -2.23410369E+01;
    COFD[6021] = 5.39645154E+00;
    COFD[6022] = -4.42708323E-01;
    COFD[6023] = 1.74846134E-02;
    COFD[6024] = -2.26239253E+01;
    COFD[6025] = 5.44867280E+00;
    COFD[6026] = -4.52284883E-01;
    COFD[6027] = 1.80110706E-02;
    COFD[6028] = -2.26250040E+01;
    COFD[6029] = 5.44867280E+00;
    COFD[6030] = -4.52284883E-01;
    COFD[6031] = 1.80110706E-02;
    COFD[6032] = -2.02693653E+01;
    COFD[6033] = 5.10426133E+00;
    COFD[6034] = -4.41256919E-01;
    COFD[6035] = 1.88737290E-02;
    COFD[6036] = -1.90915649E+01;
    COFD[6037] = 4.84384483E+00;
    COFD[6038] = -4.10265575E-01;
    COFD[6039] = 1.76414287E-02;
    COFD[6040] = -1.57040212E+01;
    COFD[6041] = 3.93614244E+00;
    COFD[6042] = -2.99111497E-01;
    COFD[6043] = 1.30888229E-02;
    COFD[6044] = -1.94691430E+01;
    COFD[6045] = 5.43830787E+00;
    COFD[6046] = -4.75472880E-01;
    COFD[6047] = 1.99909996E-02;
    COFD[6048] = -1.91136491E+01;
    COFD[6049] = 4.84384483E+00;
    COFD[6050] = -4.10265575E-01;
    COFD[6051] = 1.76414287E-02;
    COFD[6052] = -1.87419199E+01;
    COFD[6053] = 3.96926341E+00;
    COFD[6054] = -2.16412264E-01;
    COFD[6055] = 6.06012078E-03;
    COFD[6056] = -2.05235731E+01;
    COFD[6057] = 5.18417470E+00;
    COFD[6058] = -4.49491573E-01;
    COFD[6059] = 1.91438508E-02;
    COFD[6060] = -2.05324091E+01;
    COFD[6061] = 5.18417470E+00;
    COFD[6062] = -4.49491573E-01;
    COFD[6063] = 1.91438508E-02;
    COFD[6064] = -2.05408665E+01;
    COFD[6065] = 5.18417470E+00;
    COFD[6066] = -4.49491573E-01;
    COFD[6067] = 1.91438508E-02;
    COFD[6068] = -1.90143953E+01;
    COFD[6069] = 4.84384483E+00;
    COFD[6070] = -4.10265575E-01;
    COFD[6071] = 1.76414287E-02;
    COFD[6072] = -2.00963085E+01;
    COFD[6073] = 4.41511629E+00;
    COFD[6074] = -2.84086963E-01;
    COFD[6075] = 9.37586971E-03;
    COFD[6076] = -2.11378465E+01;
    COFD[6077] = 5.42846112E+00;
    COFD[6078] = -4.74321870E-01;
    COFD[6079] = 1.99459749E-02;
    COFD[6080] = -2.22176950E+01;
    COFD[6081] = 5.54251230E+00;
    COFD[6082] = -4.70946314E-01;
    COFD[6083] = 1.90785869E-02;
    COFD[6084] = -2.02969740E+01;
    COFD[6085] = 5.11106992E+00;
    COFD[6086] = -4.42047129E-01;
    COFD[6087] = 1.89042990E-02;
    COFD[6088] = -2.01064363E+01;
    COFD[6089] = 4.41511629E+00;
    COFD[6090] = -2.84086963E-01;
    COFD[6091] = 9.37586971E-03;
    COFD[6092] = -2.11378465E+01;
    COFD[6093] = 5.42846112E+00;
    COFD[6094] = -4.74321870E-01;
    COFD[6095] = 1.99459749E-02;
    COFD[6096] = -2.11637902E+01;
    COFD[6097] = 5.42846112E+00;
    COFD[6098] = -4.74321870E-01;
    COFD[6099] = 1.99459749E-02;
    COFD[6100] = -2.09272429E+01;
    COFD[6101] = 4.82184721E+00;
    COFD[6102] = -3.48128875E-01;
    COFD[6103] = 1.25918978E-02;
    COFD[6104] = -2.11341653E+01;
    COFD[6105] = 5.41773516E+00;
    COFD[6106] = -4.73414338E-01;
    COFD[6107] = 1.99258685E-02;
    COFD[6108] = -2.03087302E+01;
    COFD[6109] = 4.50250781E+00;
    COFD[6110] = -2.97622106E-01;
    COFD[6111] = 1.00481473E-02;
    COFD[6112] = -2.22478879E+01;
    COFD[6113] = 5.53139819E+00;
    COFD[6114] = -4.68828555E-01;
    COFD[6115] = 1.89597887E-02;
    COFD[6116] = -2.22377812E+01;
    COFD[6117] = 5.53139819E+00;
    COFD[6118] = -4.68828555E-01;
    COFD[6119] = 1.89597887E-02;
    COFD[6120] = -2.09061629E+01;
    COFD[6121] = 4.72895031E+00;
    COFD[6122] = -3.33332771E-01;
    COFD[6123] = 1.18431478E-02;
    COFD[6124] = -2.09285776E+01;
    COFD[6125] = 4.72895031E+00;
    COFD[6126] = -3.33332771E-01;
    COFD[6127] = 1.18431478E-02;
    COFD[6128] = -2.04096182E+01;
    COFD[6129] = 4.50250781E+00;
    COFD[6130] = -2.97622106E-01;
    COFD[6131] = 1.00481473E-02;
    COFD[6132] = -2.04146565E+01;
    COFD[6133] = 4.50250781E+00;
    COFD[6134] = -2.97622106E-01;
    COFD[6135] = 1.00481473E-02;
    COFD[6136] = -2.20307777E+01;
    COFD[6137] = 5.49663315E+00;
    COFD[6138] = -4.61182837E-01;
    COFD[6139] = 1.85035558E-02;
    COFD[6140] = -2.12680082E+01;
    COFD[6141] = 5.47935225E+00;
    COFD[6142] = -4.80056796E-01;
    COFD[6143] = 2.01607180E-02;
    COFD[6144] = -2.20425255E+01;
    COFD[6145] = 5.49663315E+00;
    COFD[6146] = -4.61182837E-01;
    COFD[6147] = 1.85035558E-02;
    COFD[6148] = -2.09121217E+01;
    COFD[6149] = 4.72895031E+00;
    COFD[6150] = -3.33332771E-01;
    COFD[6151] = 1.18431478E-02;
    COFD[6152] = -2.20656222E+01;
    COFD[6153] = 5.34774760E+00;
    COFD[6154] = -4.34239753E-01;
    COFD[6155] = 1.70320676E-02;
    COFD[6156] = -2.20445411E+01;
    COFD[6157] = 5.56049839E+00;
    COFD[6158] = -4.74367872E-01;
    COFD[6159] = 1.92702787E-02;
    COFD[6160] = -2.07133089E+01;
    COFD[6161] = 4.56211059E+00;
    COFD[6162] = -3.06895158E-01;
    COFD[6163] = 1.05100393E-02;
    COFD[6164] = -2.09121217E+01;
    COFD[6165] = 4.72895031E+00;
    COFD[6166] = -3.33332771E-01;
    COFD[6167] = 1.18431478E-02;
    COFD[6168] = -2.07817999E+01;
    COFD[6169] = 4.56211059E+00;
    COFD[6170] = -3.06895158E-01;
    COFD[6171] = 1.05100393E-02;
    COFD[6172] = -2.22205383E+01;
    COFD[6173] = 5.58360799E+00;
    COFD[6174] = -4.82701436E-01;
    COFD[6175] = 1.98437922E-02;
    COFD[6176] = -2.19583199E+01;
    COFD[6177] = 5.27258289E+00;
    COFD[6178] = -4.21502790E-01;
    COFD[6179] = 1.63611949E-02;
    COFD[6180] = -2.19649589E+01;
    COFD[6181] = 5.27258289E+00;
    COFD[6182] = -4.21502790E-01;
    COFD[6183] = 1.63611949E-02;
    COFD[6184] = -2.20250551E+01;
    COFD[6185] = 5.31412694E+00;
    COFD[6186] = -4.28473898E-01;
    COFD[6187] = 1.67264841E-02;
    COFD[6188] = -2.22604974E+01;
    COFD[6189] = 5.36643605E+00;
    COFD[6190] = -4.37440735E-01;
    COFD[6191] = 1.72016388E-02;
    COFD[6192] = -2.08429322E+01;
    COFD[6193] = 4.50409026E+00;
    COFD[6194] = -2.97868419E-01;
    COFD[6195] = 1.00604224E-02;
    COFD[6196] = -2.19319411E+01;
    COFD[6197] = 5.14570932E+00;
    COFD[6198] = -3.99877142E-01;
    COFD[6199] = 1.52199557E-02;
    COFD[6200] = -2.19349837E+01;
    COFD[6201] = 5.15446948E+00;
    COFD[6202] = -4.01332769E-01;
    COFD[6203] = 1.52956262E-02;
    COFD[6204] = -2.19389389E+01;
    COFD[6205] = 5.15446948E+00;
    COFD[6206] = -4.01332769E-01;
    COFD[6207] = 1.52956262E-02;
    COFD[6208] = -2.20959225E+01;
    COFD[6209] = 5.16679492E+00;
    COFD[6210] = -4.03405751E-01;
    COFD[6211] = 1.54041741E-02;
    COFD[6212] = -2.07638147E+01;
    COFD[6213] = 4.42680848E+00;
    COFD[6214] = -2.85885288E-01;
    COFD[6215] = 9.46483934E-03;
    COFD[6216] = -2.17537109E+01;
    COFD[6217] = 4.93496210E+00;
    COFD[6218] = -3.65981745E-01;
    COFD[6219] = 1.34912948E-02;
    COFD[6220] = -2.18871097E+01;
    COFD[6221] = 4.99907484E+00;
    COFD[6222] = -3.76094627E-01;
    COFD[6223] = 1.40009262E-02;
    COFD[6224] = -2.12295821E+01;
    COFD[6225] = 4.70506024E+00;
    COFD[6226] = -3.29547212E-01;
    COFD[6227] = 1.16521630E-02;
    COFD[6228] = -2.08900285E+01;
    COFD[6229] = 4.48108132E+00;
    COFD[6230] = -2.94289899E-01;
    COFD[6231] = 9.88218297E-03;
    COFD[6232] = -2.13607457E+01;
    COFD[6233] = 4.61201872E+00;
    COFD[6234] = -3.14803338E-01;
    COFD[6235] = 1.09082984E-02;
    COFD[6236] = -2.13622700E+01;
    COFD[6237] = 4.61201872E+00;
    COFD[6238] = -3.14803338E-01;
    COFD[6239] = 1.09082984E-02;
    COFD[6240] = -1.90859283E+01;
    COFD[6241] = 4.68079396E+00;
    COFD[6242] = -3.91231550E-01;
    COFD[6243] = 1.69021170E-02;
    COFD[6244] = -1.79361160E+01;
    COFD[6245] = 4.42139452E+00;
    COFD[6246] = -3.59567329E-01;
    COFD[6247] = 1.56103969E-02;
    COFD[6248] = -1.45715797E+01;
    COFD[6249] = 3.49477850E+00;
    COFD[6250] = -2.42635772E-01;
    COFD[6251] = 1.06721490E-02;
    COFD[6252] = -1.85748546E+01;
    COFD[6253] = 5.14789919E+00;
    COFD[6254] = -4.45930850E-01;
    COFD[6255] = 1.90363341E-02;
    COFD[6256] = -1.79580609E+01;
    COFD[6257] = 4.42139452E+00;
    COFD[6258] = -3.59567329E-01;
    COFD[6259] = 1.56103969E-02;
    COFD[6260] = -2.06310304E+01;
    COFD[6261] = 4.89289496E+00;
    COFD[6262] = -3.59346263E-01;
    COFD[6263] = 1.31570901E-02;
    COFD[6264] = -1.93917298E+01;
    COFD[6265] = 4.78708023E+00;
    COFD[6266] = -4.03693144E-01;
    COFD[6267] = 1.73884817E-02;
    COFD[6268] = -1.94004795E+01;
    COFD[6269] = 4.78708023E+00;
    COFD[6270] = -4.03693144E-01;
    COFD[6271] = 1.73884817E-02;
    COFD[6272] = -1.94088529E+01;
    COFD[6273] = 4.78708023E+00;
    COFD[6274] = -4.03693144E-01;
    COFD[6275] = 1.73884817E-02;
    COFD[6276] = -1.78593879E+01;
    COFD[6277] = 4.42139452E+00;
    COFD[6278] = -3.59567329E-01;
    COFD[6279] = 1.56103969E-02;
    COFD[6280] = -2.15702446E+01;
    COFD[6281] = 5.16868516E+00;
    COFD[6282] = -4.03721581E-01;
    COFD[6283] = 1.54206640E-02;
    COFD[6284] = -2.02434438E+01;
    COFD[6285] = 5.14418672E+00;
    COFD[6286] = -4.45631004E-01;
    COFD[6287] = 1.90308403E-02;
    COFD[6288] = -2.20725883E+01;
    COFD[6289] = 5.59642965E+00;
    COFD[6290] = -4.91577716E-01;
    COFD[6291] = 2.05159582E-02;
    COFD[6292] = -1.91118445E+01;
    COFD[6293] = 4.68715685E+00;
    COFD[6294] = -3.91979493E-01;
    COFD[6295] = 1.69314004E-02;
    COFD[6296] = -2.15802788E+01;
    COFD[6297] = 5.16868516E+00;
    COFD[6298] = -4.03721581E-01;
    COFD[6299] = 1.54206640E-02;
    COFD[6300] = -2.02434438E+01;
    COFD[6301] = 5.14418672E+00;
    COFD[6302] = -4.45631004E-01;
    COFD[6303] = 1.90308403E-02;
    COFD[6304] = -2.02692384E+01;
    COFD[6305] = 5.14418672E+00;
    COFD[6306] = -4.45631004E-01;
    COFD[6307] = 1.90308403E-02;
    COFD[6308] = -2.19873532E+01;
    COFD[6309] = 5.39977369E+00;
    COFD[6310] = -4.43340854E-01;
    COFD[6311] = 1.75199613E-02;
    COFD[6312] = -2.02318658E+01;
    COFD[6313] = 5.12963391E+00;
    COFD[6314] = -4.44146826E-01;
    COFD[6315] = 1.89829640E-02;
    COFD[6316] = -2.16936515E+01;
    COFD[6317] = 5.21869603E+00;
    COFD[6318] = -4.12084772E-01;
    COFD[6319] = 1.58573035E-02;
    COFD[6320] = -2.21343023E+01;
    COFD[6321] = 5.60010742E+00;
    COFD[6322] = -4.91597429E-01;
    COFD[6323] = 2.04987718E-02;
    COFD[6324] = -2.21242889E+01;
    COFD[6325] = 5.60010742E+00;
    COFD[6326] = -4.91597429E-01;
    COFD[6327] = 2.04987718E-02;
    COFD[6328] = -2.20597305E+01;
    COFD[6329] = 5.34774760E+00;
    COFD[6330] = -4.34239753E-01;
    COFD[6331] = 1.70320676E-02;
    COFD[6332] = -2.20818886E+01;
    COFD[6333] = 5.34774760E+00;
    COFD[6334] = -4.34239753E-01;
    COFD[6335] = 1.70320676E-02;
    COFD[6336] = -2.17934580E+01;
    COFD[6337] = 5.21869603E+00;
    COFD[6338] = -4.12084772E-01;
    COFD[6339] = 1.58573035E-02;
    COFD[6340] = -2.17984365E+01;
    COFD[6341] = 5.21869603E+00;
    COFD[6342] = -4.12084772E-01;
    COFD[6343] = 1.58573035E-02;
    COFD[6344] = -2.20228343E+01;
    COFD[6345] = 5.61211028E+00;
    COFD[6346] = -4.90893171E-01;
    COFD[6347] = 2.03793118E-02;
    COFD[6348] = -2.03116266E+01;
    COFD[6349] = 5.16758304E+00;
    COFD[6350] = -4.47606321E-01;
    COFD[6351] = 1.90728318E-02;
    COFD[6352] = -2.20344803E+01;
    COFD[6353] = 5.61211028E+00;
    COFD[6354] = -4.90893171E-01;
    COFD[6355] = 2.03793118E-02;
    COFD[6356] = -2.20656222E+01;
    COFD[6357] = 5.34774760E+00;
    COFD[6358] = -4.34239753E-01;
    COFD[6359] = 1.70320676E-02;
    COFD[6360] = -2.23318349E+01;
    COFD[6361] = 5.58508387E+00;
    COFD[6362] = -4.81385216E-01;
    COFD[6363] = 1.97267369E-02;
    COFD[6364] = -2.18222696E+01;
    COFD[6365] = 5.57940140E+00;
    COFD[6366] = -4.89964112E-01;
    COFD[6367] = 2.04689539E-02;
    COFD[6368] = -2.20174614E+01;
    COFD[6369] = 5.24609974E+00;
    COFD[6370] = -4.16866354E-01;
    COFD[6371] = 1.61128051E-02;
    COFD[6372] = -2.20656222E+01;
    COFD[6373] = 5.34774760E+00;
    COFD[6374] = -4.34239753E-01;
    COFD[6375] = 1.70320676E-02;
    COFD[6376] = -2.20851028E+01;
    COFD[6377] = 5.24609974E+00;
    COFD[6378] = -4.16866354E-01;
    COFD[6379] = 1.61128051E-02;
    COFD[6380] = -2.16647422E+01;
    COFD[6381] = 5.45895254E+00;
    COFD[6382] = -4.77778067E-01;
    COFD[6383] = 2.00763518E-02;
    COFD[6384] = -2.23931168E+01;
    COFD[6385] = 5.58325398E+00;
    COFD[6386] = -4.79084067E-01;
    COFD[6387] = 1.95452935E-02;
    COFD[6388] = -2.23996837E+01;
    COFD[6389] = 5.58325398E+00;
    COFD[6390] = -4.79084067E-01;
    COFD[6391] = 1.95452935E-02;
    COFD[6392] = -2.23689627E+01;
    COFD[6393] = 5.58513878E+00;
    COFD[6394] = -4.80389524E-01;
    COFD[6395] = 1.96438689E-02;
    COFD[6396] = -2.24797372E+01;
    COFD[6397] = 5.58492389E+00;
    COFD[6398] = -4.81921515E-01;
    COFD[6399] = 1.97721229E-02;
    COFD[6400] = -2.22169882E+01;
    COFD[6401] = 5.21950983E+00;
    COFD[6402] = -4.12223195E-01;
    COFD[6403] = 1.58645894E-02;
    COFD[6404] = -2.25041734E+01;
    COFD[6405] = 5.51797622E+00;
    COFD[6406] = -4.66229499E-01;
    COFD[6407] = 1.88128348E-02;
    COFD[6408] = -2.24965286E+01;
    COFD[6409] = 5.52198915E+00;
    COFD[6410] = -4.67014474E-01;
    COFD[6411] = 1.88574253E-02;
    COFD[6412] = -2.25004333E+01;
    COFD[6413] = 5.52198915E+00;
    COFD[6414] = -4.67014474E-01;
    COFD[6415] = 1.88574253E-02;
    COFD[6416] = -2.26411013E+01;
    COFD[6417] = 5.52830072E+00;
    COFD[6418] = -4.68235018E-01;
    COFD[6419] = 1.89263933E-02;
    COFD[6420] = -2.22139496E+01;
    COFD[6421] = 5.17488844E+00;
    COFD[6422] = -4.04758505E-01;
    COFD[6423] = 1.54748177E-02;
    COFD[6424] = -2.26485311E+01;
    COFD[6425] = 5.44696782E+00;
    COFD[6426] = -4.51976837E-01;
    COFD[6427] = 1.79942461E-02;
    COFD[6428] = -2.26946865E+01;
    COFD[6429] = 5.47392239E+00;
    COFD[6430] = -4.56882004E-01;
    COFD[6431] = 1.82631638E-02;
    COFD[6432] = -2.23996701E+01;
    COFD[6433] = 5.33372666E+00;
    COFD[6434] = -4.31837946E-01;
    COFD[6435] = 1.69048117E-02;
    COFD[6436] = -2.22885235E+01;
    COFD[6437] = 5.20764658E+00;
    COFD[6438] = -4.10207913E-01;
    COFD[6439] = 1.57585882E-02;
    COFD[6440] = -2.26029886E+01;
    COFD[6441] = 5.27383847E+00;
    COFD[6442] = -4.21722368E-01;
    COFD[6443] = 1.63729618E-02;
    COFD[6444] = -2.26044889E+01;
    COFD[6445] = 5.27383847E+00;
    COFD[6446] = -4.21722368E-01;
    COFD[6447] = 1.63729618E-02;
    COFD[6448] = -1.78815889E+01;
    COFD[6449] = 4.34347890E+00;
    COFD[6450] = -3.49890003E-01;
    COFD[6451] = 1.52083459E-02;
    COFD[6452] = -1.68343393E+01;
    COFD[6453] = 4.11954900E+00;
    COFD[6454] = -3.22470391E-01;
    COFD[6455] = 1.40859564E-02;
    COFD[6456] = -1.36336373E+01;
    COFD[6457] = 3.22088176E+00;
    COFD[6458] = -2.07623790E-01;
    COFD[6459] = 9.17771542E-03;
    COFD[6460] = -1.74407963E+01;
    COFD[6461] = 4.83580036E+00;
    COFD[6462] = -4.09383573E-01;
    COFD[6463] = 1.76098175E-02;
    COFD[6464] = -1.68535757E+01;
    COFD[6465] = 4.11954900E+00;
    COFD[6466] = -3.22470391E-01;
    COFD[6467] = 1.40859564E-02;
    COFD[6468] = -2.11309197E+01;
    COFD[6469] = 5.32644193E+00;
    COFD[6470] = -4.30581064E-01;
    COFD[6471] = 1.68379725E-02;
    COFD[6472] = -1.82145353E+01;
    COFD[6473] = 4.46848269E+00;
    COFD[6474] = -3.65269718E-01;
    COFD[6475] = 1.58407652E-02;
    COFD[6476] = -1.82217198E+01;
    COFD[6477] = 4.46848269E+00;
    COFD[6478] = -3.65269718E-01;
    COFD[6479] = 1.58407652E-02;
    COFD[6480] = -1.82285740E+01;
    COFD[6481] = 4.46848269E+00;
    COFD[6482] = -3.65269718E-01;
    COFD[6483] = 1.58407652E-02;
    COFD[6484] = -1.67662974E+01;
    COFD[6485] = 4.11954900E+00;
    COFD[6486] = -3.22470391E-01;
    COFD[6487] = 1.40859564E-02;
    COFD[6488] = -2.17771954E+01;
    COFD[6489] = 5.47519298E+00;
    COFD[6490] = -4.57113040E-01;
    COFD[6491] = 1.82758312E-02;
    COFD[6492] = -1.90996795E+01;
    COFD[6493] = 4.82869066E+00;
    COFD[6494] = -4.08564514E-01;
    COFD[6495] = 1.75784675E-02;
    COFD[6496] = -2.11031143E+01;
    COFD[6497] = 5.39439999E+00;
    COFD[6498] = -4.72050184E-01;
    COFD[6499] = 1.99336257E-02;
    COFD[6500] = -1.79116531E+01;
    COFD[6501] = 4.35148286E+00;
    COFD[6502] = -3.50886647E-01;
    COFD[6503] = 1.52498573E-02;
    COFD[6504] = -2.17855148E+01;
    COFD[6505] = 5.47519298E+00;
    COFD[6506] = -4.57113040E-01;
    COFD[6507] = 1.82758312E-02;
    COFD[6508] = -1.90996795E+01;
    COFD[6509] = 4.82869066E+00;
    COFD[6510] = -4.08564514E-01;
    COFD[6511] = 1.75784675E-02;
    COFD[6512] = -1.91225414E+01;
    COFD[6513] = 4.82869066E+00;
    COFD[6514] = -4.08564514E-01;
    COFD[6515] = 1.75784675E-02;
    COFD[6516] = -2.19136842E+01;
    COFD[6517] = 5.58503445E+00;
    COFD[6518] = -4.79552117E-01;
    COFD[6519] = 1.95750393E-02;
    COFD[6520] = -1.90692595E+01;
    COFD[6521] = 4.80830699E+00;
    COFD[6522] = -4.06171933E-01;
    COFD[6523] = 1.74848791E-02;
    COFD[6524] = -2.18356866E+01;
    COFD[6525] = 5.49906960E+00;
    COFD[6526] = -4.61793001E-01;
    COFD[6527] = 1.85415189E-02;
    COFD[6528] = -2.12014186E+01;
    COFD[6529] = 5.40060531E+00;
    COFD[6530] = -4.72449699E-01;
    COFD[6531] = 1.99345817E-02;
    COFD[6532] = -2.11931178E+01;
    COFD[6533] = 5.40060531E+00;
    COFD[6534] = -4.72449699E-01;
    COFD[6535] = 1.99345817E-02;
    COFD[6536] = -2.20398328E+01;
    COFD[6537] = 5.56049839E+00;
    COFD[6538] = -4.74367872E-01;
    COFD[6539] = 1.92702787E-02;
    COFD[6540] = -2.20574820E+01;
    COFD[6541] = 5.56049839E+00;
    COFD[6542] = -4.74367872E-01;
    COFD[6543] = 1.92702787E-02;
    COFD[6544] = -2.19162360E+01;
    COFD[6545] = 5.49906960E+00;
    COFD[6546] = -4.61793001E-01;
    COFD[6547] = 1.85415189E-02;
    COFD[6548] = -2.19201709E+01;
    COFD[6549] = 5.49906960E+00;
    COFD[6550] = -4.61793001E-01;
    COFD[6551] = 1.85415189E-02;
    COFD[6552] = -2.11427744E+01;
    COFD[6553] = 5.43893233E+00;
    COFD[6554] = -4.75546039E-01;
    COFD[6555] = 1.99938690E-02;
    COFD[6556] = -1.91367023E+01;
    COFD[6557] = 4.87703209E+00;
    COFD[6558] = -4.14222202E-01;
    COFD[6559] = 1.77987878E-02;
    COFD[6560] = -2.11525334E+01;
    COFD[6561] = 5.43893233E+00;
    COFD[6562] = -4.75546039E-01;
    COFD[6563] = 1.99938690E-02;
    COFD[6564] = -2.20445411E+01;
    COFD[6565] = 5.56049839E+00;
    COFD[6566] = -4.74367872E-01;
    COFD[6567] = 1.92702787E-02;
    COFD[6568] = -2.18222696E+01;
    COFD[6569] = 5.57940140E+00;
    COFD[6570] = -4.89964112E-01;
    COFD[6571] = 2.04689539E-02;
    COFD[6572] = -2.08820897E+01;
    COFD[6573] = 5.38250415E+00;
    COFD[6574] = -4.71144140E-01;
    COFD[6575] = 1.99199779E-02;
    COFD[6576] = -2.21089333E+01;
    COFD[6577] = 5.50506115E+00;
    COFD[6578] = -4.63563533E-01;
    COFD[6579] = 1.86575247E-02;
    COFD[6580] = -2.20445411E+01;
    COFD[6581] = 5.56049839E+00;
    COFD[6582] = -4.74367872E-01;
    COFD[6583] = 1.92702787E-02;
    COFD[6584] = -2.21619121E+01;
    COFD[6585] = 5.50506115E+00;
    COFD[6586] = -4.63563533E-01;
    COFD[6587] = 1.86575247E-02;
    COFD[6588] = -2.07415218E+01;
    COFD[6589] = 5.26552592E+00;
    COFD[6590] = -4.58996898E-01;
    COFD[6591] = 1.95145314E-02;
    COFD[6592] = -2.19448434E+01;
    COFD[6593] = 5.60255148E+00;
    COFD[6594] = -4.91366572E-01;
    COFD[6595] = 2.04670553E-02;
    COFD[6596] = -2.19501296E+01;
    COFD[6597] = 5.60255148E+00;
    COFD[6598] = -4.91366572E-01;
    COFD[6599] = 2.04670553E-02;
    COFD[6600] = -2.19032561E+01;
    COFD[6601] = 5.59794138E+00;
    COFD[6602] = -4.91684532E-01;
    COFD[6603] = 2.05170953E-02;
    COFD[6604] = -2.19617258E+01;
    COFD[6605] = 5.57026255E+00;
    COFD[6606] = -4.89178491E-01;
    COFD[6607] = 2.04505218E-02;
    COFD[6608] = -2.23434237E+01;
    COFD[6609] = 5.49927389E+00;
    COFD[6610] = -4.61845436E-01;
    COFD[6611] = 1.85448066E-02;
    COFD[6612] = -2.21913393E+01;
    COFD[6613] = 5.60175327E+00;
    COFD[6614] = -4.87953216E-01;
    COFD[6615] = 2.01882171E-02;
    COFD[6616] = -2.21792065E+01;
    COFD[6617] = 5.60465338E+00;
    COFD[6618] = -4.88572478E-01;
    COFD[6619] = 2.02248525E-02;
    COFD[6620] = -2.21822461E+01;
    COFD[6621] = 5.60465338E+00;
    COFD[6622] = -4.88572478E-01;
    COFD[6623] = 2.02248525E-02;
    COFD[6624] = -2.23250359E+01;
    COFD[6625] = 5.60776666E+00;
    COFD[6626] = -4.89319792E-01;
    COFD[6627] = 2.02710069E-02;
    COFD[6628] = -2.23935500E+01;
    COFD[6629] = 5.47922490E+00;
    COFD[6630] = -4.57847893E-01;
    COFD[6631] = 1.83161707E-02;
    COFD[6632] = -2.24603310E+01;
    COFD[6633] = 5.58501539E+00;
    COFD[6634] = -4.81433860E-01;
    COFD[6635] = 1.97311245E-02;
    COFD[6636] = -2.24423544E+01;
    COFD[6637] = 5.58416166E+00;
    COFD[6638] = -4.82369720E-01;
    COFD[6639] = 1.98133127E-02;
    COFD[6640] = -2.23867276E+01;
    COFD[6641] = 5.55175851E+00;
    COFD[6642] = -4.72720598E-01;
    COFD[6643] = 1.91783487E-02;
    COFD[6644] = -2.24356779E+01;
    COFD[6645] = 5.49613266E+00;
    COFD[6646] = -4.61060586E-01;
    COFD[6647] = 1.84960110E-02;
    COFD[6648] = -2.26579938E+01;
    COFD[6649] = 5.52001624E+00;
    COFD[6650] = -4.66629503E-01;
    COFD[6651] = 1.88355817E-02;
    COFD[6652] = -2.26591038E+01;
    COFD[6653] = 5.52001624E+00;
    COFD[6654] = -4.66629503E-01;
    COFD[6655] = 1.88355817E-02;
    COFD[6656] = -2.05802296E+01;
    COFD[6657] = 5.16117916E+00;
    COFD[6658] = -4.46897404E-01;
    COFD[6659] = 1.90470443E-02;
    COFD[6660] = -1.95314689E+01;
    COFD[6661] = 4.95249173E+00;
    COFD[6662] = -4.23376552E-01;
    COFD[6663] = 1.81703714E-02;
    COFD[6664] = -1.61116686E+01;
    COFD[6665] = 4.04227735E+00;
    COFD[6666] = -3.12745253E-01;
    COFD[6667] = 1.36756977E-02;
    COFD[6668] = -1.98806372E+01;
    COFD[6669] = 5.52555673E+00;
    COFD[6670] = -4.84999851E-01;
    COFD[6671] = 2.03334931E-02;
    COFD[6672] = -1.95538303E+01;
    COFD[6673] = 4.95249173E+00;
    COFD[6674] = -4.23376552E-01;
    COFD[6675] = 1.81703714E-02;
    COFD[6676] = -1.84538368E+01;
    COFD[6677] = 3.75912079E+00;
    COFD[6678] = -1.84235105E-01;
    COFD[6679] = 4.47800951E-03;
    COFD[6680] = -2.09481051E+01;
    COFD[6681] = 5.28755355E+00;
    COFD[6682] = -4.61641920E-01;
    COFD[6683] = 1.96208961E-02;
    COFD[6684] = -2.09571146E+01;
    COFD[6685] = 5.28755355E+00;
    COFD[6686] = -4.61641920E-01;
    COFD[6687] = 1.96208961E-02;
    COFD[6688] = -2.09657408E+01;
    COFD[6689] = 5.28755355E+00;
    COFD[6690] = -4.61641920E-01;
    COFD[6691] = 1.96208961E-02;
    COFD[6692] = -1.94534227E+01;
    COFD[6693] = 4.95249173E+00;
    COFD[6694] = -4.23376552E-01;
    COFD[6695] = 1.81703714E-02;
    COFD[6696] = -1.97545910E+01;
    COFD[6697] = 4.18758010E+00;
    COFD[6698] = -2.49327776E-01;
    COFD[6699] = 7.66559103E-03;
    COFD[6700] = -2.15326361E+01;
    COFD[6701] = 5.51982454E+00;
    COFD[6702] = -4.84452039E-01;
    COFD[6703] = 2.03175522E-02;
    COFD[6704] = -2.23098172E+01;
    COFD[6705] = 5.49916900E+00;
    COFD[6706] = -4.61818485E-01;
    COFD[6707] = 1.85431163E-02;
    COFD[6708] = -2.06066440E+01;
    COFD[6709] = 5.16748146E+00;
    COFD[6710] = -4.47594939E-01;
    COFD[6711] = 1.90724110E-02;
    COFD[6712] = -1.97649065E+01;
    COFD[6713] = 4.18758010E+00;
    COFD[6714] = -2.49327776E-01;
    COFD[6715] = 7.66559103E-03;
    COFD[6716] = -2.15326361E+01;
    COFD[6717] = 5.51982454E+00;
    COFD[6718] = -4.84452039E-01;
    COFD[6719] = 2.03175522E-02;
    COFD[6720] = -2.15588759E+01;
    COFD[6721] = 5.51982454E+00;
    COFD[6722] = -4.84452039E-01;
    COFD[6723] = 2.03175522E-02;
    COFD[6724] = -2.07356106E+01;
    COFD[6725] = 4.65728078E+00;
    COFD[6726] = -3.22002062E-01;
    COFD[6727] = 1.12723316E-02;
    COFD[6728] = -2.15067581E+01;
    COFD[6729] = 5.49964831E+00;
    COFD[6730] = -4.82275380E-01;
    COFD[6731] = 2.02405072E-02;
    COFD[6732] = -2.00066696E+01;
    COFD[6733] = 4.29138907E+00;
    COFD[6734] = -2.65108149E-01;
    COFD[6735] = 8.43949637E-03;
    COFD[6736] = -2.23404275E+01;
    COFD[6737] = 5.49239750E+00;
    COFD[6738] = -4.60320987E-01;
    COFD[6739] = 1.84538922E-02;
    COFD[6740] = -2.23301333E+01;
    COFD[6741] = 5.49239750E+00;
    COFD[6742] = -4.60320987E-01;
    COFD[6743] = 1.84538922E-02;
    COFD[6744] = -2.07072145E+01;
    COFD[6745] = 4.56211059E+00;
    COFD[6746] = -3.06895158E-01;
    COFD[6747] = 1.05100393E-02;
    COFD[6748] = -2.07301477E+01;
    COFD[6749] = 4.56211059E+00;
    COFD[6750] = -3.06895158E-01;
    COFD[6751] = 1.05100393E-02;
    COFD[6752] = -2.01097379E+01;
    COFD[6753] = 4.29138907E+00;
    COFD[6754] = -2.65108149E-01;
    COFD[6755] = 8.43949637E-03;
    COFD[6756] = -2.01148973E+01;
    COFD[6757] = 4.29138907E+00;
    COFD[6758] = -2.65108149E-01;
    COFD[6759] = 8.43949637E-03;
    COFD[6760] = -2.20547393E+01;
    COFD[6761] = 5.42445100E+00;
    COFD[6762] = -4.47918761E-01;
    COFD[6763] = 1.77729995E-02;
    COFD[6764] = -2.16544368E+01;
    COFD[6765] = 5.55511977E+00;
    COFD[6766] = -4.87927156E-01;
    COFD[6767] = 2.04245402E-02;
    COFD[6768] = -2.20666909E+01;
    COFD[6769] = 5.42445100E+00;
    COFD[6770] = -4.47918761E-01;
    COFD[6771] = 1.77729995E-02;
    COFD[6772] = -2.07133089E+01;
    COFD[6773] = 4.56211059E+00;
    COFD[6774] = -3.06895158E-01;
    COFD[6775] = 1.05100393E-02;
    COFD[6776] = -2.20174614E+01;
    COFD[6777] = 5.24609974E+00;
    COFD[6778] = -4.16866354E-01;
    COFD[6779] = 1.61128051E-02;
    COFD[6780] = -2.21089333E+01;
    COFD[6781] = 5.50506115E+00;
    COFD[6782] = -4.63563533E-01;
    COFD[6783] = 1.86575247E-02;
    COFD[6784] = -2.04221575E+01;
    COFD[6785] = 4.35883159E+00;
    COFD[6786] = -2.75434484E-01;
    COFD[6787] = 8.94819804E-03;
    COFD[6788] = -2.07133089E+01;
    COFD[6789] = 4.56211059E+00;
    COFD[6790] = -3.06895158E-01;
    COFD[6791] = 1.05100393E-02;
    COFD[6792] = -2.04923703E+01;
    COFD[6793] = 4.35883159E+00;
    COFD[6794] = -2.75434484E-01;
    COFD[6795] = 8.94819804E-03;
    COFD[6796] = -2.24098606E+01;
    COFD[6797] = 5.58471203E+00;
    COFD[6798] = -4.79905311E-01;
    COFD[6799] = 1.96058913E-02;
    COFD[6800] = -2.19128405E+01;
    COFD[6801] = 5.17305355E+00;
    COFD[6802] = -4.04451717E-01;
    COFD[6803] = 1.54587933E-02;
    COFD[6804] = -2.19196248E+01;
    COFD[6805] = 5.17305355E+00;
    COFD[6806] = -4.04451717E-01;
    COFD[6807] = 1.54587933E-02;
    COFD[6808] = -2.19819796E+01;
    COFD[6809] = 5.21506351E+00;
    COFD[6810] = -4.11467220E-01;
    COFD[6811] = 1.58248077E-02;
    COFD[6812] = -2.21991216E+01;
    COFD[6813] = 5.26251942E+00;
    COFD[6814] = -4.19749995E-01;
    COFD[6815] = 1.62674716E-02;
    COFD[6816] = -2.05292988E+01;
    COFD[6817] = 4.29315014E+00;
    COFD[6818] = -2.65377485E-01;
    COFD[6819] = 8.45274673E-03;
    COFD[6820] = -2.17842783E+01;
    COFD[6821] = 5.00373919E+00;
    COFD[6822] = -3.76839143E-01;
    COFD[6823] = 1.40386989E-02;
    COFD[6824] = -2.17946700E+01;
    COFD[6825] = 5.01521891E+00;
    COFD[6826] = -3.78672535E-01;
    COFD[6827] = 1.41317315E-02;
    COFD[6828] = -2.17987275E+01;
    COFD[6829] = 5.01521891E+00;
    COFD[6830] = -3.78672535E-01;
    COFD[6831] = 1.41317315E-02;
    COFD[6832] = -2.19609064E+01;
    COFD[6833] = 5.03230486E+00;
    COFD[6834] = -3.81405277E-01;
    COFD[6835] = 1.42705027E-02;
    COFD[6836] = -2.04142326E+01;
    COFD[6837] = 4.20149142E+00;
    COFD[6838] = -2.51432163E-01;
    COFD[6839] = 7.76854246E-03;
    COFD[6840] = -2.15342960E+01;
    COFD[6841] = 4.76557679E+00;
    COFD[6842] = -3.39171992E-01;
    COFD[6843] = 1.21386188E-02;
    COFD[6844] = -2.16515578E+01;
    COFD[6845] = 4.82270577E+00;
    COFD[6846] = -3.48263719E-01;
    COFD[6847] = 1.25986681E-02;
    COFD[6848] = -2.10131896E+01;
    COFD[6849] = 4.53499682E+00;
    COFD[6850] = -3.02678130E-01;
    COFD[6851] = 1.03000978E-02;
    COFD[6852] = -2.05673423E+01;
    COFD[6853] = 4.26766320E+00;
    COFD[6854] = -2.61480535E-01;
    COFD[6855] = 8.26106960E-03;
    COFD[6856] = -2.10973496E+01;
    COFD[6857] = 4.42639566E+00;
    COFD[6858] = -2.85821723E-01;
    COFD[6859] = 9.46169352E-03;
    COFD[6860] = -2.10989231E+01;
    COFD[6861] = 4.42639566E+00;
    COFD[6862] = -2.85821723E-01;
    COFD[6863] = 9.46169352E-03;
    COFD[6864] = -2.02693653E+01;
    COFD[6865] = 5.10426133E+00;
    COFD[6866] = -4.41256919E-01;
    COFD[6867] = 1.88737290E-02;
    COFD[6868] = -1.90915649E+01;
    COFD[6869] = 4.84384483E+00;
    COFD[6870] = -4.10265575E-01;
    COFD[6871] = 1.76414287E-02;
    COFD[6872] = -1.57040212E+01;
    COFD[6873] = 3.93614244E+00;
    COFD[6874] = -2.99111497E-01;
    COFD[6875] = 1.30888229E-02;
    COFD[6876] = -1.94691430E+01;
    COFD[6877] = 5.43830787E+00;
    COFD[6878] = -4.75472880E-01;
    COFD[6879] = 1.99909996E-02;
    COFD[6880] = -1.91136491E+01;
    COFD[6881] = 4.84384483E+00;
    COFD[6882] = -4.10265575E-01;
    COFD[6883] = 1.76414287E-02;
    COFD[6884] = -1.87419199E+01;
    COFD[6885] = 3.96926341E+00;
    COFD[6886] = -2.16412264E-01;
    COFD[6887] = 6.06012078E-03;
    COFD[6888] = -2.05235731E+01;
    COFD[6889] = 5.18417470E+00;
    COFD[6890] = -4.49491573E-01;
    COFD[6891] = 1.91438508E-02;
    COFD[6892] = -2.05324091E+01;
    COFD[6893] = 5.18417470E+00;
    COFD[6894] = -4.49491573E-01;
    COFD[6895] = 1.91438508E-02;
    COFD[6896] = -2.05408665E+01;
    COFD[6897] = 5.18417470E+00;
    COFD[6898] = -4.49491573E-01;
    COFD[6899] = 1.91438508E-02;
    COFD[6900] = -1.90143953E+01;
    COFD[6901] = 4.84384483E+00;
    COFD[6902] = -4.10265575E-01;
    COFD[6903] = 1.76414287E-02;
    COFD[6904] = -2.00963085E+01;
    COFD[6905] = 4.41511629E+00;
    COFD[6906] = -2.84086963E-01;
    COFD[6907] = 9.37586971E-03;
    COFD[6908] = -2.11378465E+01;
    COFD[6909] = 5.42846112E+00;
    COFD[6910] = -4.74321870E-01;
    COFD[6911] = 1.99459749E-02;
    COFD[6912] = -2.22176950E+01;
    COFD[6913] = 5.54251230E+00;
    COFD[6914] = -4.70946314E-01;
    COFD[6915] = 1.90785869E-02;
    COFD[6916] = -2.02969740E+01;
    COFD[6917] = 5.11106992E+00;
    COFD[6918] = -4.42047129E-01;
    COFD[6919] = 1.89042990E-02;
    COFD[6920] = -2.01064363E+01;
    COFD[6921] = 4.41511629E+00;
    COFD[6922] = -2.84086963E-01;
    COFD[6923] = 9.37586971E-03;
    COFD[6924] = -2.11378465E+01;
    COFD[6925] = 5.42846112E+00;
    COFD[6926] = -4.74321870E-01;
    COFD[6927] = 1.99459749E-02;
    COFD[6928] = -2.11637902E+01;
    COFD[6929] = 5.42846112E+00;
    COFD[6930] = -4.74321870E-01;
    COFD[6931] = 1.99459749E-02;
    COFD[6932] = -2.09272429E+01;
    COFD[6933] = 4.82184721E+00;
    COFD[6934] = -3.48128875E-01;
    COFD[6935] = 1.25918978E-02;
    COFD[6936] = -2.11341653E+01;
    COFD[6937] = 5.41773516E+00;
    COFD[6938] = -4.73414338E-01;
    COFD[6939] = 1.99258685E-02;
    COFD[6940] = -2.03087302E+01;
    COFD[6941] = 4.50250781E+00;
    COFD[6942] = -2.97622106E-01;
    COFD[6943] = 1.00481473E-02;
    COFD[6944] = -2.22478879E+01;
    COFD[6945] = 5.53139819E+00;
    COFD[6946] = -4.68828555E-01;
    COFD[6947] = 1.89597887E-02;
    COFD[6948] = -2.22377812E+01;
    COFD[6949] = 5.53139819E+00;
    COFD[6950] = -4.68828555E-01;
    COFD[6951] = 1.89597887E-02;
    COFD[6952] = -2.09061629E+01;
    COFD[6953] = 4.72895031E+00;
    COFD[6954] = -3.33332771E-01;
    COFD[6955] = 1.18431478E-02;
    COFD[6956] = -2.09285776E+01;
    COFD[6957] = 4.72895031E+00;
    COFD[6958] = -3.33332771E-01;
    COFD[6959] = 1.18431478E-02;
    COFD[6960] = -2.04096182E+01;
    COFD[6961] = 4.50250781E+00;
    COFD[6962] = -2.97622106E-01;
    COFD[6963] = 1.00481473E-02;
    COFD[6964] = -2.04146565E+01;
    COFD[6965] = 4.50250781E+00;
    COFD[6966] = -2.97622106E-01;
    COFD[6967] = 1.00481473E-02;
    COFD[6968] = -2.20307777E+01;
    COFD[6969] = 5.49663315E+00;
    COFD[6970] = -4.61182837E-01;
    COFD[6971] = 1.85035558E-02;
    COFD[6972] = -2.12680082E+01;
    COFD[6973] = 5.47935225E+00;
    COFD[6974] = -4.80056796E-01;
    COFD[6975] = 2.01607180E-02;
    COFD[6976] = -2.20425255E+01;
    COFD[6977] = 5.49663315E+00;
    COFD[6978] = -4.61182837E-01;
    COFD[6979] = 1.85035558E-02;
    COFD[6980] = -2.09121217E+01;
    COFD[6981] = 4.72895031E+00;
    COFD[6982] = -3.33332771E-01;
    COFD[6983] = 1.18431478E-02;
    COFD[6984] = -2.20656222E+01;
    COFD[6985] = 5.34774760E+00;
    COFD[6986] = -4.34239753E-01;
    COFD[6987] = 1.70320676E-02;
    COFD[6988] = -2.20445411E+01;
    COFD[6989] = 5.56049839E+00;
    COFD[6990] = -4.74367872E-01;
    COFD[6991] = 1.92702787E-02;
    COFD[6992] = -2.07133089E+01;
    COFD[6993] = 4.56211059E+00;
    COFD[6994] = -3.06895158E-01;
    COFD[6995] = 1.05100393E-02;
    COFD[6996] = -2.09121217E+01;
    COFD[6997] = 4.72895031E+00;
    COFD[6998] = -3.33332771E-01;
    COFD[6999] = 1.18431478E-02;
    COFD[7000] = -2.07817999E+01;
    COFD[7001] = 4.56211059E+00;
    COFD[7002] = -3.06895158E-01;
    COFD[7003] = 1.05100393E-02;
    COFD[7004] = -2.22205383E+01;
    COFD[7005] = 5.58360799E+00;
    COFD[7006] = -4.82701436E-01;
    COFD[7007] = 1.98437922E-02;
    COFD[7008] = -2.19583199E+01;
    COFD[7009] = 5.27258289E+00;
    COFD[7010] = -4.21502790E-01;
    COFD[7011] = 1.63611949E-02;
    COFD[7012] = -2.19649589E+01;
    COFD[7013] = 5.27258289E+00;
    COFD[7014] = -4.21502790E-01;
    COFD[7015] = 1.63611949E-02;
    COFD[7016] = -2.20250551E+01;
    COFD[7017] = 5.31412694E+00;
    COFD[7018] = -4.28473898E-01;
    COFD[7019] = 1.67264841E-02;
    COFD[7020] = -2.22604974E+01;
    COFD[7021] = 5.36643605E+00;
    COFD[7022] = -4.37440735E-01;
    COFD[7023] = 1.72016388E-02;
    COFD[7024] = -2.08429322E+01;
    COFD[7025] = 4.50409026E+00;
    COFD[7026] = -2.97868419E-01;
    COFD[7027] = 1.00604224E-02;
    COFD[7028] = -2.19319411E+01;
    COFD[7029] = 5.14570932E+00;
    COFD[7030] = -3.99877142E-01;
    COFD[7031] = 1.52199557E-02;
    COFD[7032] = -2.19349837E+01;
    COFD[7033] = 5.15446948E+00;
    COFD[7034] = -4.01332769E-01;
    COFD[7035] = 1.52956262E-02;
    COFD[7036] = -2.19389389E+01;
    COFD[7037] = 5.15446948E+00;
    COFD[7038] = -4.01332769E-01;
    COFD[7039] = 1.52956262E-02;
    COFD[7040] = -2.20959225E+01;
    COFD[7041] = 5.16679492E+00;
    COFD[7042] = -4.03405751E-01;
    COFD[7043] = 1.54041741E-02;
    COFD[7044] = -2.07638147E+01;
    COFD[7045] = 4.42680848E+00;
    COFD[7046] = -2.85885288E-01;
    COFD[7047] = 9.46483934E-03;
    COFD[7048] = -2.17537109E+01;
    COFD[7049] = 4.93496210E+00;
    COFD[7050] = -3.65981745E-01;
    COFD[7051] = 1.34912948E-02;
    COFD[7052] = -2.18871097E+01;
    COFD[7053] = 4.99907484E+00;
    COFD[7054] = -3.76094627E-01;
    COFD[7055] = 1.40009262E-02;
    COFD[7056] = -2.12295821E+01;
    COFD[7057] = 4.70506024E+00;
    COFD[7058] = -3.29547212E-01;
    COFD[7059] = 1.16521630E-02;
    COFD[7060] = -2.08900285E+01;
    COFD[7061] = 4.48108132E+00;
    COFD[7062] = -2.94289899E-01;
    COFD[7063] = 9.88218297E-03;
    COFD[7064] = -2.13607457E+01;
    COFD[7065] = 4.61201872E+00;
    COFD[7066] = -3.14803338E-01;
    COFD[7067] = 1.09082984E-02;
    COFD[7068] = -2.13622700E+01;
    COFD[7069] = 4.61201872E+00;
    COFD[7070] = -3.14803338E-01;
    COFD[7071] = 1.09082984E-02;
    COFD[7072] = -2.06331583E+01;
    COFD[7073] = 5.16117916E+00;
    COFD[7074] = -4.46897404E-01;
    COFD[7075] = 1.90470443E-02;
    COFD[7076] = -1.95670324E+01;
    COFD[7077] = 4.95249173E+00;
    COFD[7078] = -4.23376552E-01;
    COFD[7079] = 1.81703714E-02;
    COFD[7080] = -1.61173105E+01;
    COFD[7081] = 4.04227735E+00;
    COFD[7082] = -3.12745253E-01;
    COFD[7083] = 1.36756977E-02;
    COFD[7084] = -1.98835119E+01;
    COFD[7085] = 5.52555673E+00;
    COFD[7086] = -4.84999851E-01;
    COFD[7087] = 2.03334931E-02;
    COFD[7088] = -1.95910824E+01;
    COFD[7089] = 4.95249173E+00;
    COFD[7090] = -4.23376552E-01;
    COFD[7091] = 1.81703714E-02;
    COFD[7092] = -1.84927291E+01;
    COFD[7093] = 3.75912079E+00;
    COFD[7094] = -1.84235105E-01;
    COFD[7095] = 4.47800951E-03;
    COFD[7096] = -2.10057003E+01;
    COFD[7097] = 5.28755355E+00;
    COFD[7098] = -4.61641920E-01;
    COFD[7099] = 1.96208961E-02;
    COFD[7100] = -2.10158209E+01;
    COFD[7101] = 5.28755355E+00;
    COFD[7102] = -4.61641920E-01;
    COFD[7103] = 1.96208961E-02;
    COFD[7104] = -2.10255323E+01;
    COFD[7105] = 5.28755355E+00;
    COFD[7106] = -4.61641920E-01;
    COFD[7107] = 1.96208961E-02;
    COFD[7108] = -1.94836874E+01;
    COFD[7109] = 4.95249173E+00;
    COFD[7110] = -4.23376552E-01;
    COFD[7111] = 1.81703714E-02;
    COFD[7112] = -1.98087397E+01;
    COFD[7113] = 4.18758010E+00;
    COFD[7114] = -2.49327776E-01;
    COFD[7115] = 7.66559103E-03;
    COFD[7116] = -2.15647464E+01;
    COFD[7117] = 5.51982454E+00;
    COFD[7118] = -4.84452039E-01;
    COFD[7119] = 2.03175522E-02;
    COFD[7120] = -2.23791409E+01;
    COFD[7121] = 5.49916900E+00;
    COFD[7122] = -4.61818485E-01;
    COFD[7123] = 1.85431163E-02;
    COFD[7124] = -2.06595692E+01;
    COFD[7125] = 5.16748146E+00;
    COFD[7126] = -4.47594939E-01;
    COFD[7127] = 1.90724110E-02;
    COFD[7128] = -1.98202487E+01;
    COFD[7129] = 4.18758010E+00;
    COFD[7130] = -2.49327776E-01;
    COFD[7131] = 7.66559103E-03;
    COFD[7132] = -2.15647464E+01;
    COFD[7133] = 5.51982454E+00;
    COFD[7134] = -4.84452039E-01;
    COFD[7135] = 2.03175522E-02;
    COFD[7136] = -2.15927763E+01;
    COFD[7137] = 5.51982454E+00;
    COFD[7138] = -4.84452039E-01;
    COFD[7139] = 2.03175522E-02;
    COFD[7140] = -2.07921175E+01;
    COFD[7141] = 4.65728078E+00;
    COFD[7142] = -3.22002062E-01;
    COFD[7143] = 1.12723316E-02;
    COFD[7144] = -2.15423956E+01;
    COFD[7145] = 5.49964831E+00;
    COFD[7146] = -4.82275380E-01;
    COFD[7147] = 2.02405072E-02;
    COFD[7148] = -2.00643134E+01;
    COFD[7149] = 4.29138907E+00;
    COFD[7150] = -2.65108149E-01;
    COFD[7151] = 8.43949637E-03;
    COFD[7152] = -2.23958208E+01;
    COFD[7153] = 5.49239750E+00;
    COFD[7154] = -4.60320987E-01;
    COFD[7155] = 1.84538922E-02;
    COFD[7156] = -2.23843343E+01;
    COFD[7157] = 5.49239750E+00;
    COFD[7158] = -4.60320987E-01;
    COFD[7159] = 1.84538922E-02;
    COFD[7160] = -2.07748171E+01;
    COFD[7161] = 4.56211059E+00;
    COFD[7162] = -3.06895158E-01;
    COFD[7163] = 1.05100393E-02;
    COFD[7164] = -2.08011592E+01;
    COFD[7165] = 4.56211059E+00;
    COFD[7166] = -3.06895158E-01;
    COFD[7167] = 1.05100393E-02;
    COFD[7168] = -2.01815677E+01;
    COFD[7169] = 4.29138907E+00;
    COFD[7170] = -2.65108149E-01;
    COFD[7171] = 8.43949637E-03;
    COFD[7172] = -2.01875290E+01;
    COFD[7173] = 4.29138907E+00;
    COFD[7174] = -2.65108149E-01;
    COFD[7175] = 8.43949637E-03;
    COFD[7176] = -2.21051793E+01;
    COFD[7177] = 5.42445100E+00;
    COFD[7178] = -4.47918761E-01;
    COFD[7179] = 1.77729995E-02;
    COFD[7180] = -2.17211317E+01;
    COFD[7181] = 5.55511977E+00;
    COFD[7182] = -4.87927156E-01;
    COFD[7183] = 2.04245402E-02;
    COFD[7184] = -2.21184165E+01;
    COFD[7185] = 5.42445100E+00;
    COFD[7186] = -4.47918761E-01;
    COFD[7187] = 1.77729995E-02;
    COFD[7188] = -2.07817999E+01;
    COFD[7189] = 4.56211059E+00;
    COFD[7190] = -3.06895158E-01;
    COFD[7191] = 1.05100393E-02;
    COFD[7192] = -2.20851028E+01;
    COFD[7193] = 5.24609974E+00;
    COFD[7194] = -4.16866354E-01;
    COFD[7195] = 1.61128051E-02;
    COFD[7196] = -2.21619121E+01;
    COFD[7197] = 5.50506115E+00;
    COFD[7198] = -4.63563533E-01;
    COFD[7199] = 1.86575247E-02;
    COFD[7200] = -2.04923703E+01;
    COFD[7201] = 4.35883159E+00;
    COFD[7202] = -2.75434484E-01;
    COFD[7203] = 8.94819804E-03;
    COFD[7204] = -2.07817999E+01;
    COFD[7205] = 4.56211059E+00;
    COFD[7206] = -3.06895158E-01;
    COFD[7207] = 1.05100393E-02;
    COFD[7208] = -2.05740791E+01;
    COFD[7209] = 4.35883159E+00;
    COFD[7210] = -2.75434484E-01;
    COFD[7211] = 8.94819804E-03;
    COFD[7212] = -2.24737535E+01;
    COFD[7213] = 5.58471203E+00;
    COFD[7214] = -4.79905311E-01;
    COFD[7215] = 1.96058913E-02;
    COFD[7216] = -2.19777015E+01;
    COFD[7217] = 5.17305355E+00;
    COFD[7218] = -4.04451717E-01;
    COFD[7219] = 1.54587933E-02;
    COFD[7220] = -2.19854328E+01;
    COFD[7221] = 5.17305355E+00;
    COFD[7222] = -4.04451717E-01;
    COFD[7223] = 1.54587933E-02;
    COFD[7224] = -2.20487142E+01;
    COFD[7225] = 5.21506351E+00;
    COFD[7226] = -4.11467220E-01;
    COFD[7227] = 1.58248077E-02;
    COFD[7228] = -2.22676506E+01;
    COFD[7229] = 5.26251942E+00;
    COFD[7230] = -4.19749995E-01;
    COFD[7231] = 1.62674716E-02;
    COFD[7232] = -2.06187142E+01;
    COFD[7233] = 4.29315014E+00;
    COFD[7234] = -2.65377485E-01;
    COFD[7235] = 8.45274673E-03;
    COFD[7236] = -2.18614044E+01;
    COFD[7237] = 5.00373919E+00;
    COFD[7238] = -3.76839143E-01;
    COFD[7239] = 1.40386989E-02;
    COFD[7240] = -2.18724957E+01;
    COFD[7241] = 5.01521891E+00;
    COFD[7242] = -3.78672535E-01;
    COFD[7243] = 1.41317315E-02;
    COFD[7244] = -2.18772398E+01;
    COFD[7245] = 5.01521891E+00;
    COFD[7246] = -3.78672535E-01;
    COFD[7247] = 1.41317315E-02;
    COFD[7248] = -2.20400927E+01;
    COFD[7249] = 5.03230486E+00;
    COFD[7250] = -3.81405277E-01;
    COFD[7251] = 1.42705027E-02;
    COFD[7252] = -2.05098287E+01;
    COFD[7253] = 4.20149142E+00;
    COFD[7254] = -2.51432163E-01;
    COFD[7255] = 7.76854246E-03;
    COFD[7256] = -2.16206538E+01;
    COFD[7257] = 4.76557679E+00;
    COFD[7258] = -3.39171992E-01;
    COFD[7259] = 1.21386188E-02;
    COFD[7260] = -2.17384535E+01;
    COFD[7261] = 4.82270577E+00;
    COFD[7262] = -3.48263719E-01;
    COFD[7263] = 1.25986681E-02;
    COFD[7264] = -2.11006145E+01;
    COFD[7265] = 4.53499682E+00;
    COFD[7266] = -3.02678130E-01;
    COFD[7267] = 1.03000978E-02;
    COFD[7268] = -2.06609009E+01;
    COFD[7269] = 4.26766320E+00;
    COFD[7270] = -2.61480535E-01;
    COFD[7271] = 8.26106960E-03;
    COFD[7272] = -2.11966826E+01;
    COFD[7273] = 4.42639566E+00;
    COFD[7274] = -2.85821723E-01;
    COFD[7275] = 9.46169352E-03;
    COFD[7276] = -2.11986026E+01;
    COFD[7277] = 4.42639566E+00;
    COFD[7278] = -2.85821723E-01;
    COFD[7279] = 9.46169352E-03;
    COFD[7280] = -1.76895296E+01;
    COFD[7281] = 4.19171952E+00;
    COFD[7282] = -3.31354810E-01;
    COFD[7283] = 1.44520623E-02;
    COFD[7284] = -1.66128343E+01;
    COFD[7285] = 3.95035840E+00;
    COFD[7286] = -3.00959418E-01;
    COFD[7287] = 1.31692593E-02;
    COFD[7288] = -1.34824532E+01;
    COFD[7289] = 3.09379603E+00;
    COFD[7290] = -1.91268635E-01;
    COFD[7291] = 8.47480224E-03;
    COFD[7292] = -1.72291395E+01;
    COFD[7293] = 4.69060745E+00;
    COFD[7294] = -3.92369888E-01;
    COFD[7295] = 1.69459661E-02;
    COFD[7296] = -1.66341434E+01;
    COFD[7297] = 3.95035840E+00;
    COFD[7298] = -3.00959418E-01;
    COFD[7299] = 1.31692593E-02;
    COFD[7300] = -2.15787073E+01;
    COFD[7301] = 5.46737673E+00;
    COFD[7302] = -4.55696085E-01;
    COFD[7303] = 1.81982625E-02;
    COFD[7304] = -1.79791019E+01;
    COFD[7305] = 4.29613154E+00;
    COFD[7306] = -3.44012526E-01;
    COFD[7307] = 1.49643715E-02;
    COFD[7308] = -1.79874655E+01;
    COFD[7309] = 4.29613154E+00;
    COFD[7310] = -3.44012526E-01;
    COFD[7311] = 1.49643715E-02;
    COFD[7312] = -1.79954632E+01;
    COFD[7313] = 4.29613154E+00;
    COFD[7314] = -3.44012526E-01;
    COFD[7315] = 1.49643715E-02;
    COFD[7316] = -1.65381278E+01;
    COFD[7317] = 3.95035840E+00;
    COFD[7318] = -3.00959418E-01;
    COFD[7319] = 1.31692593E-02;
    COFD[7320] = -2.20866241E+01;
    COFD[7321] = 5.55935694E+00;
    COFD[7322] = -4.74154740E-01;
    COFD[7323] = 1.92584304E-02;
    COFD[7324] = -1.88826663E+01;
    COFD[7325] = 4.68393046E+00;
    COFD[7326] = -3.91610863E-01;
    COFD[7327] = 1.69174645E-02;
    COFD[7328] = -2.10296583E+01;
    COFD[7329] = 5.30153901E+00;
    COFD[7330] = -4.63335119E-01;
    COFD[7331] = 1.96897053E-02;
    COFD[7332] = -1.77178857E+01;
    COFD[7333] = 4.19935698E+00;
    COFD[7334] = -3.32310212E-01;
    COFD[7335] = 1.44920670E-02;
    COFD[7336] = -2.20962383E+01;
    COFD[7337] = 5.55935694E+00;
    COFD[7338] = -4.74154740E-01;
    COFD[7339] = 1.92584304E-02;
    COFD[7340] = -1.88826663E+01;
    COFD[7341] = 4.68393046E+00;
    COFD[7342] = -3.91610863E-01;
    COFD[7343] = 1.69174645E-02;
    COFD[7344] = -1.89077781E+01;
    COFD[7345] = 4.68393046E+00;
    COFD[7346] = -3.91610863E-01;
    COFD[7347] = 1.69174645E-02;
    COFD[7348] = -2.20433329E+01;
    COFD[7349] = 5.59157589E+00;
    COFD[7350] = -4.85617912E-01;
    COFD[7351] = 2.00461138E-02;
    COFD[7352] = -1.88538435E+01;
    COFD[7353] = 4.66162351E+00;
    COFD[7354] = -3.88920477E-01;
    COFD[7355] = 1.68089648E-02;
    COFD[7356] = -2.21445051E+01;
    COFD[7357] = 5.58129885E+00;
    COFD[7358] = -4.78532921E-01;
    COFD[7359] = 1.95095699E-02;
    COFD[7360] = -2.11349436E+01;
    COFD[7361] = 5.32202066E+00;
    COFD[7362] = -4.65780334E-01;
    COFD[7363] = 1.97876377E-02;
    COFD[7364] = -2.11253498E+01;
    COFD[7365] = 5.32202066E+00;
    COFD[7366] = -4.65780334E-01;
    COFD[7367] = 1.97876377E-02;
    COFD[7368] = -2.22149446E+01;
    COFD[7369] = 5.58360799E+00;
    COFD[7370] = -4.82701436E-01;
    COFD[7371] = 1.98437922E-02;
    COFD[7372] = -2.22359646E+01;
    COFD[7373] = 5.58360799E+00;
    COFD[7374] = -4.82701436E-01;
    COFD[7375] = 1.98437922E-02;
    COFD[7376] = -2.22394964E+01;
    COFD[7377] = 5.58129885E+00;
    COFD[7378] = -4.78532921E-01;
    COFD[7379] = 1.95095699E-02;
    COFD[7380] = -2.22442100E+01;
    COFD[7381] = 5.58129885E+00;
    COFD[7382] = -4.78532921E-01;
    COFD[7383] = 1.95095699E-02;
    COFD[7384] = -2.11070024E+01;
    COFD[7385] = 5.37047121E+00;
    COFD[7386] = -4.70282612E-01;
    COFD[7387] = 1.99109322E-02;
    COFD[7388] = -1.89718952E+01;
    COFD[7389] = 4.72476764E+00;
    COFD[7390] = -3.96306836E-01;
    COFD[7391] = 1.70964541E-02;
    COFD[7392] = -2.11181899E+01;
    COFD[7393] = 5.37047121E+00;
    COFD[7394] = -4.70282612E-01;
    COFD[7395] = 1.99109322E-02;
    COFD[7396] = -2.22205383E+01;
    COFD[7397] = 5.58360799E+00;
    COFD[7398] = -4.82701436E-01;
    COFD[7399] = 1.98437922E-02;
    COFD[7400] = -2.16647422E+01;
    COFD[7401] = 5.45895254E+00;
    COFD[7402] = -4.77778067E-01;
    COFD[7403] = 2.00763518E-02;
    COFD[7404] = -2.07415218E+01;
    COFD[7405] = 5.26552592E+00;
    COFD[7406] = -4.58996898E-01;
    COFD[7407] = 1.95145314E-02;
    COFD[7408] = -2.24098606E+01;
    COFD[7409] = 5.58471203E+00;
    COFD[7410] = -4.79905311E-01;
    COFD[7411] = 1.96058913E-02;
    COFD[7412] = -2.22205383E+01;
    COFD[7413] = 5.58360799E+00;
    COFD[7414] = -4.82701436E-01;
    COFD[7415] = 1.98437922E-02;
    COFD[7416] = -2.24737535E+01;
    COFD[7417] = 5.58471203E+00;
    COFD[7418] = -4.79905311E-01;
    COFD[7419] = 1.96058913E-02;
    COFD[7420] = -2.05663544E+01;
    COFD[7421] = 5.13263469E+00;
    COFD[7422] = -4.44457285E-01;
    COFD[7423] = 1.89932102E-02;
    COFD[7424] = -2.18752697E+01;
    COFD[7425] = 5.52126150E+00;
    COFD[7426] = -4.84589774E-01;
    COFD[7427] = 2.03215822E-02;
    COFD[7428] = -2.18815159E+01;
    COFD[7429] = 5.52126150E+00;
    COFD[7430] = -4.84589774E-01;
    COFD[7431] = 2.03215822E-02;
    COFD[7432] = -2.17761843E+01;
    COFD[7433] = 5.49109933E+00;
    COFD[7434] = -4.81352923E-01;
    COFD[7435] = 2.02079312E-02;
    COFD[7436] = -2.17764634E+01;
    COFD[7437] = 5.44261208E+00;
    COFD[7438] = -4.75976721E-01;
    COFD[7439] = 2.00107428E-02;
    COFD[7440] = -2.26574530E+01;
    COFD[7441] = 5.58147197E+00;
    COFD[7442] = -4.78581151E-01;
    COFD[7443] = 1.95126856E-02;
    COFD[7444] = -2.22948697E+01;
    COFD[7445] = 5.59185582E+00;
    COFD[7446] = -4.91155812E-01;
    COFD[7447] = 2.05043018E-02;
    COFD[7448] = -2.22685220E+01;
    COFD[7449] = 5.58776204E+00;
    COFD[7450] = -4.90769522E-01;
    COFD[7451] = 2.04931464E-02;
    COFD[7452] = -2.22722046E+01;
    COFD[7453] = 5.58776204E+00;
    COFD[7454] = -4.90769522E-01;
    COFD[7455] = 2.04931464E-02;
    COFD[7456] = -2.23840677E+01;
    COFD[7457] = 5.58123742E+00;
    COFD[7458] = -4.90135630E-01;
    COFD[7459] = 2.04738180E-02;
    COFD[7460] = -2.27122564E+01;
    COFD[7461] = 5.56278679E+00;
    COFD[7462] = -4.74794778E-01;
    COFD[7463] = 1.92940001E-02;
    COFD[7464] = -2.26361390E+01;
    COFD[7465] = 5.61137362E+00;
    COFD[7466] = -4.90253090E-01;
    COFD[7467] = 2.03300559E-02;
    COFD[7468] = -2.26249652E+01;
    COFD[7469] = 5.61234946E+00;
    COFD[7470] = -4.91326412E-01;
    COFD[7471] = 2.04139363E-02;
    COFD[7472] = -2.25837110E+01;
    COFD[7473] = 5.58420073E+00;
    COFD[7474] = -4.82356716E-01;
    COFD[7475] = 1.98120306E-02;
    COFD[7476] = -2.27479974E+01;
    COFD[7477] = 5.57822325E+00;
    COFD[7478] = -4.77777262E-01;
    COFD[7479] = 1.94626011E-02;
    COFD[7480] = -2.29254220E+01;
    COFD[7481] = 5.58520405E+00;
    COFD[7482] = -4.80873447E-01;
    COFD[7483] = 1.96836519E-02;
    COFD[7484] = -2.29268183E+01;
    COFD[7485] = 5.58520405E+00;
    COFD[7486] = -4.80873447E-01;
    COFD[7487] = 1.96836519E-02;
    COFD[7488] = -1.92731067E+01;
    COFD[7489] = 4.73660584E+00;
    COFD[7490] = -3.97704978E-01;
    COFD[7491] = 1.71514887E-02;
    COFD[7492] = -1.81463104E+01;
    COFD[7493] = 4.48398491E+00;
    COFD[7494] = -3.67097129E-01;
    COFD[7495] = 1.59123634E-02;
    COFD[7496] = -1.47719516E+01;
    COFD[7497] = 3.55444478E+00;
    COFD[7498] = -2.50272707E-01;
    COFD[7499] = 1.09990787E-02;
    COFD[7500] = -1.87644697E+01;
    COFD[7501] = 5.19146813E+00;
    COFD[7502] = -4.50340408E-01;
    COFD[7503] = 1.91768178E-02;
    COFD[7504] = -1.81677871E+01;
    COFD[7505] = 4.48398491E+00;
    COFD[7506] = -3.67097129E-01;
    COFD[7507] = 1.59123634E-02;
    COFD[7508] = -2.04357586E+01;
    COFD[7509] = 4.77398686E+00;
    COFD[7510] = -3.40522956E-01;
    COFD[7511] = 1.22072846E-02;
    COFD[7512] = -1.95819005E+01;
    COFD[7513] = 4.84393038E+00;
    COFD[7514] = -4.10274737E-01;
    COFD[7515] = 1.76417458E-02;
    COFD[7516] = -1.95903647E+01;
    COFD[7517] = 4.84393038E+00;
    COFD[7518] = -4.10274737E-01;
    COFD[7519] = 1.76417458E-02;
    COFD[7520] = -1.95984602E+01;
    COFD[7521] = 4.84393038E+00;
    COFD[7522] = -4.10274737E-01;
    COFD[7523] = 1.76417458E-02;
    COFD[7524] = -1.80710700E+01;
    COFD[7525] = 4.48398491E+00;
    COFD[7526] = -3.67097129E-01;
    COFD[7527] = 1.59123634E-02;
    COFD[7528] = -2.14300943E+01;
    COFD[7529] = 5.07680397E+00;
    COFD[7530] = -3.88612087E-01;
    COFD[7531] = 1.46395101E-02;
    COFD[7532] = -2.04021549E+01;
    COFD[7533] = 5.18271974E+00;
    COFD[7534] = -4.49323627E-01;
    COFD[7535] = 1.91373940E-02;
    COFD[7536] = -2.21630311E+01;
    COFD[7537] = 5.60807471E+00;
    COFD[7538] = -4.91339309E-01;
    COFD[7539] = 2.04365761E-02;
    COFD[7540] = -1.93011401E+01;
    COFD[7541] = 4.74387793E+00;
    COFD[7542] = -3.98574972E-01;
    COFD[7543] = 1.71862289E-02;
    COFD[7544] = -2.14398182E+01;
    COFD[7545] = 5.07680397E+00;
    COFD[7546] = -3.88612087E-01;
    COFD[7547] = 1.46395101E-02;
    COFD[7548] = -2.04021549E+01;
    COFD[7549] = 5.18271974E+00;
    COFD[7550] = -4.49323627E-01;
    COFD[7551] = 1.91373940E-02;
    COFD[7552] = -2.04274471E+01;
    COFD[7553] = 5.18271974E+00;
    COFD[7554] = -4.49323627E-01;
    COFD[7555] = 1.91373940E-02;
    COFD[7556] = -2.19104953E+01;
    COFD[7557] = 5.33587903E+00;
    COFD[7558] = -4.32204887E-01;
    COFD[7559] = 1.69242106E-02;
    COFD[7560] = -2.03738891E+01;
    COFD[7561] = 5.16159436E+00;
    COFD[7562] = -4.46935283E-01;
    COFD[7563] = 1.90480297E-02;
    COFD[7564] = -2.15759895E+01;
    COFD[7565] = 5.13708607E+00;
    COFD[7566] = -3.98445708E-01;
    COFD[7567] = 1.51455626E-02;
    COFD[7568] = -2.22262162E+01;
    COFD[7569] = 5.61211818E+00;
    COFD[7570] = -4.91432482E-01;
    COFD[7571] = 2.04238731E-02;
    COFD[7572] = -2.22165128E+01;
    COFD[7573] = 5.61211818E+00;
    COFD[7574] = -4.91432482E-01;
    COFD[7575] = 2.04238731E-02;
    COFD[7576] = -2.19526490E+01;
    COFD[7577] = 5.27258289E+00;
    COFD[7578] = -4.21502790E-01;
    COFD[7579] = 1.63611949E-02;
    COFD[7580] = -2.19739638E+01;
    COFD[7581] = 5.27258289E+00;
    COFD[7582] = -4.21502790E-01;
    COFD[7583] = 1.63611949E-02;
    COFD[7584] = -2.16722314E+01;
    COFD[7585] = 5.13708607E+00;
    COFD[7586] = -3.98445708E-01;
    COFD[7587] = 1.51455626E-02;
    COFD[7588] = -2.16770134E+01;
    COFD[7589] = 5.13708607E+00;
    COFD[7590] = -3.98445708E-01;
    COFD[7591] = 1.51455626E-02;
    COFD[7592] = -2.20555979E+01;
    COFD[7593] = 5.59649805E+00;
    COFD[7594] = -4.86750336E-01;
    COFD[7595] = 2.01151498E-02;
    COFD[7596] = -2.05357412E+01;
    COFD[7597] = 5.23500188E+00;
    COFD[7598] = -4.55417380E-01;
    COFD[7599] = 1.93744255E-02;
    COFD[7600] = -2.20669053E+01;
    COFD[7601] = 5.59649805E+00;
    COFD[7602] = -4.86750336E-01;
    COFD[7603] = 2.01151498E-02;
    COFD[7604] = -2.19583199E+01;
    COFD[7605] = 5.27258289E+00;
    COFD[7606] = -4.21502790E-01;
    COFD[7607] = 1.63611949E-02;
    COFD[7608] = -2.23931168E+01;
    COFD[7609] = 5.58325398E+00;
    COFD[7610] = -4.79084067E-01;
    COFD[7611] = 1.95452935E-02;
    COFD[7612] = -2.19448434E+01;
    COFD[7613] = 5.60255148E+00;
    COFD[7614] = -4.91366572E-01;
    COFD[7615] = 2.04670553E-02;
    COFD[7616] = -2.19128405E+01;
    COFD[7617] = 5.17305355E+00;
    COFD[7618] = -4.04451717E-01;
    COFD[7619] = 1.54587933E-02;
    COFD[7620] = -2.19583199E+01;
    COFD[7621] = 5.27258289E+00;
    COFD[7622] = -4.21502790E-01;
    COFD[7623] = 1.63611949E-02;
    COFD[7624] = -2.19777015E+01;
    COFD[7625] = 5.17305355E+00;
    COFD[7626] = -4.04451717E-01;
    COFD[7627] = 1.54587933E-02;
    COFD[7628] = -2.18752697E+01;
    COFD[7629] = 5.52126150E+00;
    COFD[7630] = -4.84589774E-01;
    COFD[7631] = 2.03215822E-02;
    COFD[7632] = -2.23787998E+01;
    COFD[7633] = 5.54890339E+00;
    COFD[7634] = -4.72166228E-01;
    COFD[7635] = 1.91470071E-02;
    COFD[7636] = -2.23851292E+01;
    COFD[7637] = 5.54890339E+00;
    COFD[7638] = -4.72166228E-01;
    COFD[7639] = 1.91470071E-02;
    COFD[7640] = -2.24018266E+01;
    COFD[7641] = 5.57115285E+00;
    COFD[7642] = -4.76363416E-01;
    COFD[7643] = 1.93814080E-02;
    COFD[7644] = -2.25424404E+01;
    COFD[7645] = 5.58482894E+00;
    COFD[7646] = -4.79850522E-01;
    COFD[7647] = 1.96007690E-02;
    COFD[7648] = -2.20862991E+01;
    COFD[7649] = 5.13809011E+00;
    COFD[7650] = -3.98612308E-01;
    COFD[7651] = 1.51542189E-02;
    COFD[7652] = -2.25145418E+01;
    COFD[7653] = 5.49554403E+00;
    COFD[7654] = -4.60936491E-01;
    COFD[7655] = 1.84887572E-02;
    COFD[7656] = -2.25028406E+01;
    COFD[7657] = 5.49776513E+00;
    COFD[7658] = -4.61463030E-01;
    COFD[7659] = 1.85209236E-02;
    COFD[7660] = -2.25065805E+01;
    COFD[7661] = 5.49776513E+00;
    COFD[7662] = -4.61463030E-01;
    COFD[7663] = 1.85209236E-02;
    COFD[7664] = -2.26360108E+01;
    COFD[7665] = 5.50023958E+00;
    COFD[7666] = -4.62136179E-01;
    COFD[7667] = 1.85639061E-02;
    COFD[7668] = -2.20599362E+01;
    COFD[7669] = 5.08417640E+00;
    COFD[7670] = -3.89810534E-01;
    COFD[7671] = 1.47010214E-02;
    COFD[7672] = -2.25891024E+01;
    COFD[7673] = 5.39655717E+00;
    COFD[7674] = -4.42728390E-01;
    COFD[7675] = 1.74857336E-02;
    COFD[7676] = -2.26273108E+01;
    COFD[7677] = 5.42002683E+00;
    COFD[7678] = -4.47111163E-01;
    COFD[7679] = 1.77287360E-02;
    COFD[7680] = -2.22858832E+01;
    COFD[7681] = 5.25941804E+00;
    COFD[7682] = -4.19208672E-01;
    COFD[7683] = 1.62385114E-02;
    COFD[7684] = -2.21494624E+01;
    COFD[7685] = 5.12338366E+00;
    COFD[7686] = -3.96176894E-01;
    COFD[7687] = 1.50278196E-02;
    COFD[7688] = -2.25069737E+01;
    COFD[7689] = 5.21003123E+00;
    COFD[7690] = -4.10612564E-01;
    COFD[7691] = 1.57798598E-02;
    COFD[7692] = -2.25083966E+01;
    COFD[7693] = 5.21003123E+00;
    COFD[7694] = -4.10612564E-01;
    COFD[7695] = 1.57798598E-02;
    COFD[7696] = -1.92783884E+01;
    COFD[7697] = 4.73660584E+00;
    COFD[7698] = -3.97704978E-01;
    COFD[7699] = 1.71514887E-02;
    COFD[7700] = -1.81499793E+01;
    COFD[7701] = 4.48398491E+00;
    COFD[7702] = -3.67097129E-01;
    COFD[7703] = 1.59123634E-02;
    COFD[7704] = -1.47725694E+01;
    COFD[7705] = 3.55444478E+00;
    COFD[7706] = -2.50272707E-01;
    COFD[7707] = 1.09990787E-02;
    COFD[7708] = -1.87647862E+01;
    COFD[7709] = 5.19146813E+00;
    COFD[7710] = -4.50340408E-01;
    COFD[7711] = 1.91768178E-02;
    COFD[7712] = -1.81716176E+01;
    COFD[7713] = 4.48398491E+00;
    COFD[7714] = -3.67097129E-01;
    COFD[7715] = 1.59123634E-02;
    COFD[7716] = -2.04397451E+01;
    COFD[7717] = 4.77398686E+00;
    COFD[7718] = -3.40522956E-01;
    COFD[7719] = 1.22072846E-02;
    COFD[7720] = -1.95875976E+01;
    COFD[7721] = 4.84393038E+00;
    COFD[7722] = -4.10274737E-01;
    COFD[7723] = 1.76417458E-02;
    COFD[7724] = -1.95961596E+01;
    COFD[7725] = 4.84393038E+00;
    COFD[7726] = -4.10274737E-01;
    COFD[7727] = 1.76417458E-02;
    COFD[7728] = -1.96043503E+01;
    COFD[7729] = 4.84393038E+00;
    COFD[7730] = -4.10274737E-01;
    COFD[7731] = 1.76417458E-02;
    COFD[7732] = -1.80742247E+01;
    COFD[7733] = 4.48398491E+00;
    COFD[7734] = -3.67097129E-01;
    COFD[7735] = 1.59123634E-02;
    COFD[7736] = -2.14354853E+01;
    COFD[7737] = 5.07680397E+00;
    COFD[7738] = -3.88612087E-01;
    COFD[7739] = 1.46395101E-02;
    COFD[7740] = -2.04054899E+01;
    COFD[7741] = 5.18271974E+00;
    COFD[7742] = -4.49323627E-01;
    COFD[7743] = 1.91373940E-02;
    COFD[7744] = -2.21697404E+01;
    COFD[7745] = 5.60807471E+00;
    COFD[7746] = -4.91339309E-01;
    COFD[7747] = 2.04365761E-02;
    COFD[7748] = -1.93064215E+01;
    COFD[7749] = 4.74387793E+00;
    COFD[7750] = -3.98574972E-01;
    COFD[7751] = 1.71862289E-02;
    COFD[7752] = -2.14453157E+01;
    COFD[7753] = 5.07680397E+00;
    COFD[7754] = -3.88612087E-01;
    COFD[7755] = 1.46395101E-02;
    COFD[7756] = -2.04054899E+01;
    COFD[7757] = 5.18271974E+00;
    COFD[7758] = -4.49323627E-01;
    COFD[7759] = 1.91373940E-02;
    COFD[7760] = -2.04309557E+01;
    COFD[7761] = 5.18271974E+00;
    COFD[7762] = -4.49323627E-01;
    COFD[7763] = 1.91373940E-02;
    COFD[7764] = -2.19160962E+01;
    COFD[7765] = 5.33587903E+00;
    COFD[7766] = -4.32204887E-01;
    COFD[7767] = 1.69242106E-02;
    COFD[7768] = -2.03775651E+01;
    COFD[7769] = 5.16159436E+00;
    COFD[7770] = -4.46935283E-01;
    COFD[7771] = 1.90480297E-02;
    COFD[7772] = -2.15816909E+01;
    COFD[7773] = 5.13708607E+00;
    COFD[7774] = -3.98445708E-01;
    COFD[7775] = 1.51455626E-02;
    COFD[7776] = -2.22317182E+01;
    COFD[7777] = 5.61211818E+00;
    COFD[7778] = -4.91432482E-01;
    COFD[7779] = 2.04238731E-02;
    COFD[7780] = -2.22219085E+01;
    COFD[7781] = 5.61211818E+00;
    COFD[7782] = -4.91432482E-01;
    COFD[7783] = 2.04238731E-02;
    COFD[7784] = -2.19592125E+01;
    COFD[7785] = 5.27258289E+00;
    COFD[7786] = -4.21502790E-01;
    COFD[7787] = 1.63611949E-02;
    COFD[7788] = -2.19808152E+01;
    COFD[7789] = 5.27258289E+00;
    COFD[7790] = -4.21502790E-01;
    COFD[7791] = 1.63611949E-02;
    COFD[7792] = -2.16791513E+01;
    COFD[7793] = 5.13708607E+00;
    COFD[7794] = -3.98445708E-01;
    COFD[7795] = 1.51455626E-02;
    COFD[7796] = -2.16840004E+01;
    COFD[7797] = 5.13708607E+00;
    COFD[7798] = -3.98445708E-01;
    COFD[7799] = 1.51455626E-02;
    COFD[7800] = -2.20606550E+01;
    COFD[7801] = 5.59649805E+00;
    COFD[7802] = -4.86750336E-01;
    COFD[7803] = 2.01151498E-02;
    COFD[7804] = -2.05422276E+01;
    COFD[7805] = 5.23500188E+00;
    COFD[7806] = -4.55417380E-01;
    COFD[7807] = 1.93744255E-02;
    COFD[7808] = -2.20720787E+01;
    COFD[7809] = 5.59649805E+00;
    COFD[7810] = -4.86750336E-01;
    COFD[7811] = 2.01151498E-02;
    COFD[7812] = -2.19649589E+01;
    COFD[7813] = 5.27258289E+00;
    COFD[7814] = -4.21502790E-01;
    COFD[7815] = 1.63611949E-02;
    COFD[7816] = -2.23996837E+01;
    COFD[7817] = 5.58325398E+00;
    COFD[7818] = -4.79084067E-01;
    COFD[7819] = 1.95452935E-02;
    COFD[7820] = -2.19501296E+01;
    COFD[7821] = 5.60255148E+00;
    COFD[7822] = -4.91366572E-01;
    COFD[7823] = 2.04670553E-02;
    COFD[7824] = -2.19196248E+01;
    COFD[7825] = 5.17305355E+00;
    COFD[7826] = -4.04451717E-01;
    COFD[7827] = 1.54587933E-02;
    COFD[7828] = -2.19649589E+01;
    COFD[7829] = 5.27258289E+00;
    COFD[7830] = -4.21502790E-01;
    COFD[7831] = 1.63611949E-02;
    COFD[7832] = -2.19854328E+01;
    COFD[7833] = 5.17305355E+00;
    COFD[7834] = -4.04451717E-01;
    COFD[7835] = 1.54587933E-02;
    COFD[7836] = -2.18815159E+01;
    COFD[7837] = 5.52126150E+00;
    COFD[7838] = -4.84589774E-01;
    COFD[7839] = 2.03215822E-02;
    COFD[7840] = -2.23851292E+01;
    COFD[7841] = 5.54890339E+00;
    COFD[7842] = -4.72166228E-01;
    COFD[7843] = 1.91470071E-02;
    COFD[7844] = -2.23915398E+01;
    COFD[7845] = 5.54890339E+00;
    COFD[7846] = -4.72166228E-01;
    COFD[7847] = 1.91470071E-02;
    COFD[7848] = -2.24083163E+01;
    COFD[7849] = 5.57115285E+00;
    COFD[7850] = -4.76363416E-01;
    COFD[7851] = 1.93814080E-02;
    COFD[7852] = -2.25490826E+01;
    COFD[7853] = 5.58482894E+00;
    COFD[7854] = -4.79850522E-01;
    COFD[7855] = 1.96007690E-02;
    COFD[7856] = -2.20946432E+01;
    COFD[7857] = 5.13809011E+00;
    COFD[7858] = -3.98612308E-01;
    COFD[7859] = 1.51542189E-02;
    COFD[7860] = -2.25219004E+01;
    COFD[7861] = 5.49554403E+00;
    COFD[7862] = -4.60936491E-01;
    COFD[7863] = 1.84887572E-02;
    COFD[7864] = -2.25102565E+01;
    COFD[7865] = 5.49776513E+00;
    COFD[7866] = -4.61463030E-01;
    COFD[7867] = 1.85209236E-02;
    COFD[7868] = -2.25140525E+01;
    COFD[7869] = 5.49776513E+00;
    COFD[7870] = -4.61463030E-01;
    COFD[7871] = 1.85209236E-02;
    COFD[7872] = -2.26435378E+01;
    COFD[7873] = 5.50023958E+00;
    COFD[7874] = -4.62136179E-01;
    COFD[7875] = 1.85639061E-02;
    COFD[7876] = -2.20687596E+01;
    COFD[7877] = 5.08417640E+00;
    COFD[7878] = -3.89810534E-01;
    COFD[7879] = 1.47010214E-02;
    COFD[7880] = -2.25972054E+01;
    COFD[7881] = 5.39655717E+00;
    COFD[7882] = -4.42728390E-01;
    COFD[7883] = 1.74857336E-02;
    COFD[7884] = -2.26354564E+01;
    COFD[7885] = 5.42002683E+00;
    COFD[7886] = -4.47111163E-01;
    COFD[7887] = 1.77287360E-02;
    COFD[7888] = -2.22940707E+01;
    COFD[7889] = 5.25941804E+00;
    COFD[7890] = -4.19208672E-01;
    COFD[7891] = 1.62385114E-02;
    COFD[7892] = -2.21581289E+01;
    COFD[7893] = 5.12338366E+00;
    COFD[7894] = -3.96176894E-01;
    COFD[7895] = 1.50278196E-02;
    COFD[7896] = -2.25160816E+01;
    COFD[7897] = 5.21003123E+00;
    COFD[7898] = -4.10612564E-01;
    COFD[7899] = 1.57798598E-02;
    COFD[7900] = -2.25175307E+01;
    COFD[7901] = 5.21003123E+00;
    COFD[7902] = -4.10612564E-01;
    COFD[7903] = 1.57798598E-02;
    COFD[7904] = -1.91796663E+01;
    COFD[7905] = 4.70714822E+00;
    COFD[7906] = -3.94261134E-01;
    COFD[7907] = 1.70175169E-02;
    COFD[7908] = -1.80480958E+01;
    COFD[7909] = 4.45434023E+00;
    COFD[7910] = -3.63584633E-01;
    COFD[7911] = 1.57739270E-02;
    COFD[7912] = -1.46719197E+01;
    COFD[7913] = 3.52400594E+00;
    COFD[7914] = -2.46379985E-01;
    COFD[7915] = 1.08326032E-02;
    COFD[7916] = -1.86493112E+01;
    COFD[7917] = 5.16040659E+00;
    COFD[7918] = -4.46843492E-01;
    COFD[7919] = 1.90466181E-02;
    COFD[7920] = -1.80698901E+01;
    COFD[7921] = 4.45434023E+00;
    COFD[7922] = -3.63584633E-01;
    COFD[7923] = 1.57739270E-02;
    COFD[7924] = -2.05372411E+01;
    COFD[7925] = 4.83379373E+00;
    COFD[7926] = -3.50008083E-01;
    COFD[7927] = 1.26863426E-02;
    COFD[7928] = -1.94912151E+01;
    COFD[7929] = 4.81575071E+00;
    COFD[7930] = -4.07042139E-01;
    COFD[7931] = 1.75187504E-02;
    COFD[7932] = -1.94998722E+01;
    COFD[7933] = 4.81575071E+00;
    COFD[7934] = -4.07042139E-01;
    COFD[7935] = 1.75187504E-02;
    COFD[7936] = -1.95081555E+01;
    COFD[7937] = 4.81575071E+00;
    COFD[7938] = -4.07042139E-01;
    COFD[7939] = 1.75187504E-02;
    COFD[7940] = -1.79718457E+01;
    COFD[7941] = 4.45434023E+00;
    COFD[7942] = -3.63584633E-01;
    COFD[7943] = 1.57739270E-02;
    COFD[7944] = -2.15159231E+01;
    COFD[7945] = 5.12799307E+00;
    COFD[7946] = -3.96938732E-01;
    COFD[7947] = 1.50673195E-02;
    COFD[7948] = -2.03111230E+01;
    COFD[7949] = 5.15740122E+00;
    COFD[7950] = -4.46644818E-01;
    COFD[7951] = 1.90459001E-02;
    COFD[7952] = -2.21216828E+01;
    COFD[7953] = 5.60203389E+00;
    COFD[7954] = -4.91444416E-01;
    COFD[7955] = 2.04761886E-02;
    COFD[7956] = -1.92044492E+01;
    COFD[7957] = 4.71304783E+00;
    COFD[7958] = -3.94942083E-01;
    COFD[7959] = 1.70435959E-02;
    COFD[7960] = -2.15258568E+01;
    COFD[7961] = 5.12799307E+00;
    COFD[7962] = -3.96938732E-01;
    COFD[7963] = 1.50673195E-02;
    COFD[7964] = -2.03111230E+01;
    COFD[7965] = 5.15740122E+00;
    COFD[7966] = -4.46644818E-01;
    COFD[7967] = 1.90459001E-02;
    COFD[7968] = -2.03367561E+01;
    COFD[7969] = 5.15740122E+00;
    COFD[7970] = -4.46644818E-01;
    COFD[7971] = 1.90459001E-02;
    COFD[7972] = -2.19617977E+01;
    COFD[7973] = 5.37170913E+00;
    COFD[7974] = -4.38338667E-01;
    COFD[7975] = 1.72490835E-02;
    COFD[7976] = -2.03123540E+01;
    COFD[7977] = 5.14854169E+00;
    COFD[7978] = -4.45984343E-01;
    COFD[7979] = 1.90374217E-02;
    COFD[7980] = -2.16420936E+01;
    COFD[7981] = 5.17945041E+00;
    COFD[7982] = -4.05514689E-01;
    COFD[7983] = 1.55141412E-02;
    COFD[7984] = -2.21793326E+01;
    COFD[7985] = 5.60403905E+00;
    COFD[7986] = -4.91221691E-01;
    COFD[7987] = 2.04473483E-02;
    COFD[7988] = -2.21694197E+01;
    COFD[7989] = 5.60403905E+00;
    COFD[7990] = -4.91221691E-01;
    COFD[7991] = 2.04473483E-02;
    COFD[7992] = -2.20192352E+01;
    COFD[7993] = 5.31412694E+00;
    COFD[7994] = -4.28473898E-01;
    COFD[7995] = 1.67264841E-02;
    COFD[7996] = -2.20411190E+01;
    COFD[7997] = 5.31412694E+00;
    COFD[7998] = -4.28473898E-01;
    COFD[7999] = 1.67264841E-02;
    COFD[8000] = -2.17407419E+01;
    COFD[8001] = 5.17945041E+00;
    COFD[8002] = -4.05514689E-01;
    COFD[8003] = 1.55141412E-02;
    COFD[8004] = -2.17456564E+01;
    COFD[8005] = 5.17945041E+00;
    COFD[8006] = -4.05514689E-01;
    COFD[8007] = 1.55141412E-02;
    COFD[8008] = -2.20511271E+01;
    COFD[8009] = 5.60809037E+00;
    COFD[8010] = -4.89400803E-01;
    COFD[8011] = 2.02760802E-02;
    COFD[8012] = -2.04251023E+01;
    COFD[8013] = 5.19993608E+00;
    COFD[8014] = -4.51334924E-01;
    COFD[8015] = 1.92158646E-02;
    COFD[8016] = -2.20626636E+01;
    COFD[8017] = 5.60809037E+00;
    COFD[8018] = -4.89400803E-01;
    COFD[8019] = 2.02760802E-02;
    COFD[8020] = -2.20250551E+01;
    COFD[8021] = 5.31412694E+00;
    COFD[8022] = -4.28473898E-01;
    COFD[8023] = 1.67264841E-02;
    COFD[8024] = -2.23689627E+01;
    COFD[8025] = 5.58513878E+00;
    COFD[8026] = -4.80389524E-01;
    COFD[8027] = 1.96438689E-02;
    COFD[8028] = -2.19032561E+01;
    COFD[8029] = 5.59794138E+00;
    COFD[8030] = -4.91684532E-01;
    COFD[8031] = 2.05170953E-02;
    COFD[8032] = -2.19819796E+01;
    COFD[8033] = 5.21506351E+00;
    COFD[8034] = -4.11467220E-01;
    COFD[8035] = 1.58248077E-02;
    COFD[8036] = -2.20250551E+01;
    COFD[8037] = 5.31412694E+00;
    COFD[8038] = -4.28473898E-01;
    COFD[8039] = 1.67264841E-02;
    COFD[8040] = -2.20487142E+01;
    COFD[8041] = 5.21506351E+00;
    COFD[8042] = -4.11467220E-01;
    COFD[8043] = 1.58248077E-02;
    COFD[8044] = -2.17761843E+01;
    COFD[8045] = 5.49109933E+00;
    COFD[8046] = -4.81352923E-01;
    COFD[8047] = 2.02079312E-02;
    COFD[8048] = -2.24018266E+01;
    COFD[8049] = 5.57115285E+00;
    COFD[8050] = -4.76363416E-01;
    COFD[8051] = 1.93814080E-02;
    COFD[8052] = -2.24083163E+01;
    COFD[8053] = 5.57115285E+00;
    COFD[8054] = -4.76363416E-01;
    COFD[8055] = 1.93814080E-02;
    COFD[8056] = -2.24021886E+01;
    COFD[8057] = 5.58364149E+00;
    COFD[8058] = -4.79184111E-01;
    COFD[8059] = 1.95516164E-02;
    COFD[8060] = -2.25161211E+01;
    COFD[8061] = 5.58521783E+00;
    COFD[8062] = -4.80947522E-01;
    COFD[8063] = 1.96897222E-02;
    COFD[8064] = -2.21603646E+01;
    COFD[8065] = 5.18050127E+00;
    COFD[8066] = -4.05688517E-01;
    COFD[8067] = 1.55231713E-02;
    COFD[8068] = -2.25060112E+01;
    COFD[8069] = 5.50327119E+00;
    COFD[8070] = -4.63087223E-01;
    COFD[8071] = 1.86271401E-02;
    COFD[8072] = -2.24931486E+01;
    COFD[8073] = 5.50509817E+00;
    COFD[8074] = -4.63572794E-01;
    COFD[8075] = 1.86581046E-02;
    COFD[8076] = -2.24969995E+01;
    COFD[8077] = 5.50509817E+00;
    COFD[8078] = -4.63572794E-01;
    COFD[8079] = 1.86581046E-02;
    COFD[8080] = -2.26299936E+01;
    COFD[8081] = 5.50881574E+00;
    COFD[8082] = -4.64448886E-01;
    COFD[8083] = 1.87118881E-02;
    COFD[8084] = -2.21535971E+01;
    COFD[8085] = 5.13453409E+00;
    COFD[8086] = -3.98022439E-01;
    COFD[8087] = 1.51235760E-02;
    COFD[8088] = -2.26203761E+01;
    COFD[8089] = 5.42039607E+00;
    COFD[8090] = -4.47178505E-01;
    COFD[8091] = 1.77324253E-02;
    COFD[8092] = -2.26677753E+01;
    COFD[8093] = 5.44777353E+00;
    COFD[8094] = -4.52122340E-01;
    COFD[8095] = 1.80021910E-02;
    COFD[8096] = -2.23480908E+01;
    COFD[8097] = 5.29695321E+00;
    COFD[8098] = -4.25620113E-01;
    COFD[8099] = 1.65778213E-02;
    COFD[8100] = -2.22256643E+01;
    COFD[8101] = 5.16620234E+00;
    COFD[8102] = -4.03306755E-01;
    COFD[8103] = 1.53990058E-02;
    COFD[8104] = -2.25635595E+01;
    COFD[8105] = 5.24330646E+00;
    COFD[8106] = -4.16370120E-01;
    COFD[8107] = 1.60860486E-02;
    COFD[8108] = -2.25650343E+01;
    COFD[8109] = 5.24330646E+00;
    COFD[8110] = -4.16370120E-01;
    COFD[8111] = 1.60860486E-02;
    COFD[8112] = -1.92062897E+01;
    COFD[8113] = 4.66318669E+00;
    COFD[8114] = -3.89108667E-01;
    COFD[8115] = 1.68165377E-02;
    COFD[8116] = -1.80724788E+01;
    COFD[8117] = 4.40247898E+00;
    COFD[8118] = -3.57238362E-01;
    COFD[8119] = 1.55145651E-02;
    COFD[8120] = -1.47137939E+01;
    COFD[8121] = 3.48023191E+00;
    COFD[8122] = -2.40800798E-01;
    COFD[8123] = 1.05947990E-02;
    COFD[8124] = -1.87481780E+01;
    COFD[8125] = 5.13858656E+00;
    COFD[8126] = -4.45075387E-01;
    COFD[8127] = 1.90137309E-02;
    COFD[8128] = -1.80945693E+01;
    COFD[8129] = 4.40247898E+00;
    COFD[8130] = -3.57238362E-01;
    COFD[8131] = 1.55145651E-02;
    COFD[8132] = -2.08879167E+01;
    COFD[8133] = 4.92602269E+00;
    COFD[8134] = -3.64572914E-01;
    COFD[8135] = 1.34203681E-02;
    COFD[8136] = -1.95201830E+01;
    COFD[8137] = 4.77151544E+00;
    COFD[8138] = -4.01882811E-01;
    COFD[8139] = 1.73184814E-02;
    COFD[8140] = -1.95290229E+01;
    COFD[8141] = 4.77151544E+00;
    COFD[8142] = -4.01882811E-01;
    COFD[8143] = 1.73184814E-02;
    COFD[8144] = -1.95374840E+01;
    COFD[8145] = 4.77151544E+00;
    COFD[8146] = -4.01882811E-01;
    COFD[8147] = 1.73184814E-02;
    COFD[8148] = -1.79952897E+01;
    COFD[8149] = 4.40247898E+00;
    COFD[8150] = -3.57238362E-01;
    COFD[8151] = 1.55145651E-02;
    COFD[8152] = -2.17825544E+01;
    COFD[8153] = 5.19232842E+00;
    COFD[8154] = -4.07643284E-01;
    COFD[8155] = 1.56246434E-02;
    COFD[8156] = -2.03711787E+01;
    COFD[8157] = 5.13279789E+00;
    COFD[8158] = -4.44474174E-01;
    COFD[8159] = 1.89937678E-02;
    COFD[8160] = -2.22052004E+01;
    COFD[8161] = 5.58604166E+00;
    COFD[8162] = -4.90602184E-01;
    COFD[8163] = 2.04880352E-02;
    COFD[8164] = -1.92334028E+01;
    COFD[8165] = 4.67033934E+00;
    COFD[8166] = -3.89971551E-01;
    COFD[8167] = 1.68513441E-02;
    COFD[8168] = -2.17926864E+01;
    COFD[8169] = 5.19232842E+00;
    COFD[8170] = -4.07643284E-01;
    COFD[8171] = 1.56246434E-02;
    COFD[8172] = -2.03711787E+01;
    COFD[8173] = 5.13279789E+00;
    COFD[8174] = -4.44474174E-01;
    COFD[8175] = 1.89937678E-02;
    COFD[8176] = -2.03971290E+01;
    COFD[8177] = 5.13279789E+00;
    COFD[8178] = -4.44474174E-01;
    COFD[8179] = 1.89937678E-02;
    COFD[8180] = -2.21713935E+01;
    COFD[8181] = 5.41196486E+00;
    COFD[8182] = -4.45632422E-01;
    COFD[8183] = 1.76474237E-02;
    COFD[8184] = -2.03526104E+01;
    COFD[8185] = 5.11453301E+00;
    COFD[8186] = -4.42447016E-01;
    COFD[8187] = 1.89196698E-02;
    COFD[8188] = -2.18910102E+01;
    COFD[8189] = 5.23595129E+00;
    COFD[8190] = -4.15079064E-01;
    COFD[8191] = 1.60168286E-02;
    COFD[8192] = -2.22701953E+01;
    COFD[8193] = 5.59632316E+00;
    COFD[8194] = -4.91568011E-01;
    COFD[8195] = 2.05156966E-02;
    COFD[8196] = -2.22600844E+01;
    COFD[8197] = 5.59632316E+00;
    COFD[8198] = -4.91568011E-01;
    COFD[8199] = 2.05156966E-02;
    COFD[8200] = -2.22545356E+01;
    COFD[8201] = 5.36643605E+00;
    COFD[8202] = -4.37440735E-01;
    COFD[8203] = 1.72016388E-02;
    COFD[8204] = -2.22769618E+01;
    COFD[8205] = 5.36643605E+00;
    COFD[8206] = -4.37440735E-01;
    COFD[8207] = 1.72016388E-02;
    COFD[8208] = -2.19919464E+01;
    COFD[8209] = 5.23595129E+00;
    COFD[8210] = -4.15079064E-01;
    COFD[8211] = 1.60168286E-02;
    COFD[8212] = -2.19969874E+01;
    COFD[8213] = 5.23595129E+00;
    COFD[8214] = -4.15079064E-01;
    COFD[8215] = 1.60168286E-02;
    COFD[8216] = -2.21795362E+01;
    COFD[8217] = 5.61233637E+00;
    COFD[8218] = -4.91419253E-01;
    COFD[8219] = 2.04216738E-02;
    COFD[8220] = -2.04750337E+01;
    COFD[8221] = 5.15745622E+00;
    COFD[8222] = -4.46648283E-01;
    COFD[8223] = 1.90458987E-02;
    COFD[8224] = -2.21912885E+01;
    COFD[8225] = 5.61233637E+00;
    COFD[8226] = -4.91419253E-01;
    COFD[8227] = 2.04216738E-02;
    COFD[8228] = -2.22604974E+01;
    COFD[8229] = 5.36643605E+00;
    COFD[8230] = -4.37440735E-01;
    COFD[8231] = 1.72016388E-02;
    COFD[8232] = -2.24797372E+01;
    COFD[8233] = 5.58492389E+00;
    COFD[8234] = -4.81921515E-01;
    COFD[8235] = 1.97721229E-02;
    COFD[8236] = -2.19617258E+01;
    COFD[8237] = 5.57026255E+00;
    COFD[8238] = -4.89178491E-01;
    COFD[8239] = 2.04505218E-02;
    COFD[8240] = -2.21991216E+01;
    COFD[8241] = 5.26251942E+00;
    COFD[8242] = -4.19749995E-01;
    COFD[8243] = 1.62674716E-02;
    COFD[8244] = -2.22604974E+01;
    COFD[8245] = 5.36643605E+00;
    COFD[8246] = -4.37440735E-01;
    COFD[8247] = 1.72016388E-02;
    COFD[8248] = -2.22676506E+01;
    COFD[8249] = 5.26251942E+00;
    COFD[8250] = -4.19749995E-01;
    COFD[8251] = 1.62674716E-02;
    COFD[8252] = -2.17764634E+01;
    COFD[8253] = 5.44261208E+00;
    COFD[8254] = -4.75976721E-01;
    COFD[8255] = 2.00107428E-02;
    COFD[8256] = -2.25424404E+01;
    COFD[8257] = 5.58482894E+00;
    COFD[8258] = -4.79850522E-01;
    COFD[8259] = 1.96007690E-02;
    COFD[8260] = -2.25490826E+01;
    COFD[8261] = 5.58482894E+00;
    COFD[8262] = -4.79850522E-01;
    COFD[8263] = 1.96007690E-02;
    COFD[8264] = -2.25161211E+01;
    COFD[8265] = 5.58521783E+00;
    COFD[8266] = -4.80947522E-01;
    COFD[8267] = 1.96897222E-02;
    COFD[8268] = -2.26149345E+01;
    COFD[8269] = 5.58414475E+00;
    COFD[8270] = -4.82375215E-01;
    COFD[8271] = 1.98138565E-02;
    COFD[8272] = -2.23925982E+01;
    COFD[8273] = 5.23666690E+00;
    COFD[8274] = -4.15204403E-01;
    COFD[8275] = 1.60235416E-02;
    COFD[8276] = -2.26679870E+01;
    COFD[8277] = 5.52852425E+00;
    COFD[8278] = -4.68277964E-01;
    COFD[8279] = 1.89288127E-02;
    COFD[8280] = -2.26623034E+01;
    COFD[8281] = 5.53286772E+00;
    COFD[8282] = -4.69109018E-01;
    COFD[8283] = 1.89755392E-02;
    COFD[8284] = -2.26662608E+01;
    COFD[8285] = 5.53286772E+00;
    COFD[8286] = -4.69109018E-01;
    COFD[8287] = 1.89755392E-02;
    COFD[8288] = -2.27989630E+01;
    COFD[8289] = 5.53955653E+00;
    COFD[8290] = -4.70381353E-01;
    COFD[8291] = 1.90468698E-02;
    COFD[8292] = -2.24028537E+01;
    COFD[8293] = 5.19900179E+00;
    COFD[8294] = -4.08748226E-01;
    COFD[8295] = 1.56820407E-02;
    COFD[8296] = -2.28101231E+01;
    COFD[8297] = 5.46112592E+00;
    COFD[8298] = -4.54556926E-01;
    COFD[8299] = 1.81357650E-02;
    COFD[8300] = -2.28554026E+01;
    COFD[8301] = 5.48796011E+00;
    COFD[8302] = -4.59457942E-01;
    COFD[8303] = 1.84050728E-02;
    COFD[8304] = -2.25780442E+01;
    COFD[8305] = 5.35238497E+00;
    COFD[8306] = -4.35034945E-01;
    COFD[8307] = 1.70742216E-02;
    COFD[8308] = -2.24631694E+01;
    COFD[8309] = 5.22623384E+00;
    COFD[8310] = -4.13380324E-01;
    COFD[8311] = 1.59259437E-02;
    COFD[8312] = -2.27715883E+01;
    COFD[8313] = 5.29493402E+00;
    COFD[8314] = -4.25285978E-01;
    COFD[8315] = 1.65604533E-02;
    COFD[8316] = -2.27731137E+01;
    COFD[8317] = 5.29493402E+00;
    COFD[8318] = -4.25285978E-01;
    COFD[8319] = 1.65604533E-02;
    COFD[8320] = -2.09943481E+01;
    COFD[8321] = 5.22468467E+00;
    COFD[8322] = -4.54220128E-01;
    COFD[8323] = 1.93281042E-02;
    COFD[8324] = -1.98296243E+01;
    COFD[8325] = 4.98207523E+00;
    COFD[8326] = -4.26877291E-01;
    COFD[8327] = 1.83086094E-02;
    COFD[8328] = -1.64819183E+01;
    COFD[8329] = 4.11726215E+00;
    COFD[8330] = -3.22193015E-01;
    COFD[8331] = 1.40747074E-02;
    COFD[8332] = -2.01262921E+01;
    COFD[8333] = 5.54581286E+00;
    COFD[8334] = -4.87014004E-01;
    COFD[8335] = 2.03965482E-02;
    COFD[8336] = -1.98546695E+01;
    COFD[8337] = 4.98207523E+00;
    COFD[8338] = -4.26877291E-01;
    COFD[8339] = 1.83086094E-02;
    COFD[8340] = -1.73636900E+01;
    COFD[8341] = 3.17377130E+00;
    COFD[8342] = -1.00394383E-01;
    COFD[8343] = 5.69083899E-04;
    COFD[8344] = -2.13698722E+01;
    COFD[8345] = 5.34971865E+00;
    COFD[8346] = -4.68771123E-01;
    COFD[8347] = 1.98933811E-02;
    COFD[8348] = -2.12907159E+01;
    COFD[8349] = 5.32167660E+00;
    COFD[8350] = -4.65740624E-01;
    COFD[8351] = 1.97861081E-02;
    COFD[8352] = -2.13011157E+01;
    COFD[8353] = 5.32167660E+00;
    COFD[8354] = -4.65740624E-01;
    COFD[8355] = 1.97861081E-02;
    COFD[8356] = -1.97431913E+01;
    COFD[8357] = 4.98207523E+00;
    COFD[8358] = -4.26877291E-01;
    COFD[8359] = 1.83086094E-02;
    COFD[8360] = -1.98237209E+01;
    COFD[8361] = 4.11158627E+00;
    COFD[8362] = -2.37831519E-01;
    COFD[8363] = 7.10363413E-03;
    COFD[8364] = -2.17867314E+01;
    COFD[8365] = 5.53950393E+00;
    COFD[8366] = -4.86376204E-01;
    COFD[8367] = 2.03760106E-02;
    COFD[8368] = -2.25168081E+01;
    COFD[8369] = 5.46125558E+00;
    COFD[8370] = -4.54580949E-01;
    COFD[8371] = 1.81370928E-02;
    COFD[8372] = -2.10310742E+01;
    COFD[8373] = 5.23485505E+00;
    COFD[8374] = -4.55400362E-01;
    COFD[8375] = 1.93737680E-02;
    COFD[8376] = -1.98359760E+01;
    COFD[8377] = 4.11158627E+00;
    COFD[8378] = -2.37831519E-01;
    COFD[8379] = 7.10363413E-03;
    COFD[8380] = -2.17867314E+01;
    COFD[8381] = 5.53950393E+00;
    COFD[8382] = -4.86376204E-01;
    COFD[8383] = 2.03760106E-02;
    COFD[8384] = -2.18158049E+01;
    COFD[8385] = 5.53950393E+00;
    COFD[8386] = -4.86376204E-01;
    COFD[8387] = 2.03760106E-02;
    COFD[8388] = -2.01613414E+01;
    COFD[8389] = 4.29679630E+00;
    COFD[8390] = -2.69916064E-01;
    COFD[8391] = 8.81737046E-03;
    COFD[8392] = -2.18731920E+01;
    COFD[8393] = 5.55171660E+00;
    COFD[8394] = -4.87609504E-01;
    COFD[8395] = 2.04156590E-02;
    COFD[8396] = -2.00981944E+01;
    COFD[8397] = 4.22278378E+00;
    COFD[8398] = -2.54653500E-01;
    COFD[8399] = 7.92616085E-03;
    COFD[8400] = -2.25302512E+01;
    COFD[8401] = 5.47136127E+00;
    COFD[8402] = -4.56417141E-01;
    COFD[8403] = 1.82376994E-02;
    COFD[8404] = -2.25180193E+01;
    COFD[8405] = 5.47136127E+00;
    COFD[8406] = -4.56417141E-01;
    COFD[8407] = 1.82376994E-02;
    COFD[8408] = -2.08353693E+01;
    COFD[8409] = 4.50409026E+00;
    COFD[8410] = -2.97868419E-01;
    COFD[8411] = 1.00604224E-02;
    COFD[8412] = -2.08639466E+01;
    COFD[8413] = 4.50409026E+00;
    COFD[8414] = -2.97868419E-01;
    COFD[8415] = 1.00604224E-02;
    COFD[8416] = -2.02246117E+01;
    COFD[8417] = 4.22278378E+00;
    COFD[8418] = -2.54653500E-01;
    COFD[8419] = 7.92616085E-03;
    COFD[8420] = -2.02311039E+01;
    COFD[8421] = 4.22278378E+00;
    COFD[8422] = -2.54653500E-01;
    COFD[8423] = 7.92616085E-03;
    COFD[8424] = -2.22462130E+01;
    COFD[8425] = 5.40356304E+00;
    COFD[8426] = -4.44060256E-01;
    COFD[8427] = 1.75601121E-02;
    COFD[8428] = -2.19764159E+01;
    COFD[8429] = 5.56943713E+00;
    COFD[8430] = -4.89114655E-01;
    COFD[8431] = 2.04494661E-02;
    COFD[8432] = -2.22602443E+01;
    COFD[8433] = 5.40356304E+00;
    COFD[8434] = -4.44060256E-01;
    COFD[8435] = 1.75601121E-02;
    COFD[8436] = -2.08429322E+01;
    COFD[8437] = 4.50409026E+00;
    COFD[8438] = -2.97868419E-01;
    COFD[8439] = 1.00604224E-02;
    COFD[8440] = -2.22169882E+01;
    COFD[8441] = 5.21950983E+00;
    COFD[8442] = -4.12223195E-01;
    COFD[8443] = 1.58645894E-02;
    COFD[8444] = -2.23434237E+01;
    COFD[8445] = 5.49927389E+00;
    COFD[8446] = -4.61845436E-01;
    COFD[8447] = 1.85448066E-02;
    COFD[8448] = -2.05292988E+01;
    COFD[8449] = 4.29315014E+00;
    COFD[8450] = -2.65377485E-01;
    COFD[8451] = 8.45274673E-03;
    COFD[8452] = -2.08429322E+01;
    COFD[8453] = 4.50409026E+00;
    COFD[8454] = -2.97868419E-01;
    COFD[8455] = 1.00604224E-02;
    COFD[8456] = -2.06187142E+01;
    COFD[8457] = 4.29315014E+00;
    COFD[8458] = -2.65377485E-01;
    COFD[8459] = 8.45274673E-03;
    COFD[8460] = -2.26574530E+01;
    COFD[8461] = 5.58147197E+00;
    COFD[8462] = -4.78581151E-01;
    COFD[8463] = 1.95126856E-02;
    COFD[8464] = -2.20862991E+01;
    COFD[8465] = 5.13809011E+00;
    COFD[8466] = -3.98612308E-01;
    COFD[8467] = 1.51542189E-02;
    COFD[8468] = -2.20946432E+01;
    COFD[8469] = 5.13809011E+00;
    COFD[8470] = -3.98612308E-01;
    COFD[8471] = 1.51542189E-02;
    COFD[8472] = -2.21603646E+01;
    COFD[8473] = 5.18050127E+00;
    COFD[8474] = -4.05688517E-01;
    COFD[8475] = 1.55231713E-02;
    COFD[8476] = -2.23925982E+01;
    COFD[8477] = 5.23666690E+00;
    COFD[8478] = -4.15204403E-01;
    COFD[8479] = 1.60235416E-02;
    COFD[8480] = -2.02828056E+01;
    COFD[8481] = 4.06866060E+00;
    COFD[8482] = -2.33527101E-01;
    COFD[8483] = 6.97454219E-03;
    COFD[8484] = -2.19287691E+01;
    COFD[8485] = 4.95026695E+00;
    COFD[8486] = -3.68392434E-01;
    COFD[8487] = 1.36126514E-02;
    COFD[8488] = -2.19456782E+01;
    COFD[8489] = 4.96368178E+00;
    COFD[8490] = -3.70505465E-01;
    COFD[8491] = 1.37190339E-02;
    COFD[8492] = -2.19508859E+01;
    COFD[8493] = 4.96368178E+00;
    COFD[8494] = -3.70505465E-01;
    COFD[8495] = 1.37190339E-02;
    COFD[8496] = -2.21147103E+01;
    COFD[8497] = 4.98427447E+00;
    COFD[8498] = -3.73746896E-01;
    COFD[8499] = 1.38821805E-02;
    COFD[8500] = -2.05271615E+01;
    COFD[8501] = 4.12444157E+00;
    COFD[8502] = -2.39777376E-01;
    COFD[8503] = 7.19872269E-03;
    COFD[8504] = -2.16983332E+01;
    COFD[8505] = 4.71782117E+00;
    COFD[8506] = -3.31568259E-01;
    COFD[8507] = 1.17540937E-02;
    COFD[8508] = -2.17385144E+01;
    COFD[8509] = 4.74350080E+00;
    COFD[8510] = -3.36426340E-01;
    COFD[8511] = 1.20245796E-02;
    COFD[8512] = -2.11585495E+01;
    COFD[8513] = 4.47646812E+00;
    COFD[8514] = -2.93573165E-01;
    COFD[8515] = 9.84650920E-03;
    COFD[8516] = -2.06827490E+01;
    COFD[8517] = 4.19375892E+00;
    COFD[8518] = -2.50262428E-01;
    COFD[8519] = 7.71131487E-03;
    COFD[8520] = -2.12332312E+01;
    COFD[8521] = 4.36095377E+00;
    COFD[8522] = -2.75760539E-01;
    COFD[8523] = 8.96430249E-03;
    COFD[8524] = -2.12354028E+01;
    COFD[8525] = 4.36095377E+00;
    COFD[8526] = -2.75760539E-01;
    COFD[8527] = 8.96430249E-03;
    COFD[8528] = -1.97484166E+01;
    COFD[8529] = 4.84231878E+00;
    COFD[8530] = -4.10101001E-01;
    COFD[8531] = 1.76356687E-02;
    COFD[8532] = -1.86652603E+01;
    COFD[8533] = 4.61260432E+00;
    COFD[8534] = -3.82854484E-01;
    COFD[8535] = 1.65575163E-02;
    COFD[8536] = -1.51448279E+01;
    COFD[8537] = 3.64565939E+00;
    COFD[8538] = -2.61726871E-01;
    COFD[8539] = 1.14799244E-02;
    COFD[8540] = -1.92784178E+01;
    COFD[8541] = 5.32291505E+00;
    COFD[8542] = -4.65883522E-01;
    COFD[8543] = 1.97916109E-02;
    COFD[8544] = -1.86886689E+01;
    COFD[8545] = 4.61260432E+00;
    COFD[8546] = -3.82854484E-01;
    COFD[8547] = 1.65575163E-02;
    COFD[8548] = -2.02184916E+01;
    COFD[8549] = 4.57152878E+00;
    COFD[8550] = -3.08371263E-01;
    COFD[8551] = 1.05838559E-02;
    COFD[8552] = -2.01315602E+01;
    COFD[8553] = 4.97613338E+00;
    COFD[8554] = -4.26175206E-01;
    COFD[8555] = 1.82809270E-02;
    COFD[8556] = -2.01412473E+01;
    COFD[8557] = 4.97613338E+00;
    COFD[8558] = -4.26175206E-01;
    COFD[8559] = 1.82809270E-02;
    COFD[8560] = -2.01505348E+01;
    COFD[8561] = 4.97613338E+00;
    COFD[8562] = -4.26175206E-01;
    COFD[8563] = 1.82809270E-02;
    COFD[8564] = -1.85839192E+01;
    COFD[8565] = 4.61260432E+00;
    COFD[8566] = -3.82854484E-01;
    COFD[8567] = 1.65575163E-02;
    COFD[8568] = -2.12109223E+01;
    COFD[8569] = 4.87252053E+00;
    COFD[8570] = -3.56127804E-01;
    COFD[8571] = 1.29948788E-02;
    COFD[8572] = -2.09217020E+01;
    COFD[8573] = 5.31360223E+00;
    COFD[8574] = -4.64787000E-01;
    COFD[8575] = 1.97483720E-02;
    COFD[8576] = -2.23890317E+01;
    COFD[8577] = 5.59178974E+00;
    COFD[8578] = -4.85668031E-01;
    COFD[8579] = 2.00491907E-02;
    COFD[8580] = -1.97709603E+01;
    COFD[8581] = 4.84731557E+00;
    COFD[8582] = -4.10638352E-01;
    COFD[8583] = 1.76543886E-02;
    COFD[8584] = -2.12219677E+01;
    COFD[8585] = 4.87252053E+00;
    COFD[8586] = -3.56127804E-01;
    COFD[8587] = 1.29948788E-02;
    COFD[8588] = -2.09217020E+01;
    COFD[8589] = 5.31360223E+00;
    COFD[8590] = -4.64787000E-01;
    COFD[8591] = 1.97483720E-02;
    COFD[8592] = -2.09490548E+01;
    COFD[8593] = 5.31360223E+00;
    COFD[8594] = -4.64787000E-01;
    COFD[8595] = 1.97483720E-02;
    COFD[8596] = -2.18851200E+01;
    COFD[8597] = 5.21365421E+00;
    COFD[8598] = -4.11227771E-01;
    COFD[8599] = 1.58122118E-02;
    COFD[8600] = -2.08822487E+01;
    COFD[8601] = 5.28557747E+00;
    COFD[8602] = -4.61402384E-01;
    COFD[8603] = 1.96111546E-02;
    COFD[8604] = -2.13985484E+01;
    COFD[8605] = 4.94878244E+00;
    COFD[8606] = -3.68158605E-01;
    COFD[8607] = 1.36008797E-02;
    COFD[8608] = -2.24120415E+01;
    COFD[8609] = 5.58744076E+00;
    COFD[8610] = -4.84489462E-01;
    COFD[8611] = 1.99733042E-02;
    COFD[8612] = -2.24010182E+01;
    COFD[8613] = 5.58744076E+00;
    COFD[8614] = -4.84489462E-01;
    COFD[8615] = 1.99733042E-02;
    COFD[8616] = -2.19253091E+01;
    COFD[8617] = 5.14570932E+00;
    COFD[8618] = -3.99877142E-01;
    COFD[8619] = 1.52199557E-02;
    COFD[8620] = -2.19503032E+01;
    COFD[8621] = 5.14570932E+00;
    COFD[8622] = -3.99877142E-01;
    COFD[8623] = 1.52199557E-02;
    COFD[8624] = -2.15102238E+01;
    COFD[8625] = 4.94878244E+00;
    COFD[8626] = -3.68158605E-01;
    COFD[8627] = 1.36008797E-02;
    COFD[8628] = -2.15158669E+01;
    COFD[8629] = 4.94878244E+00;
    COFD[8630] = -3.68158605E-01;
    COFD[8631] = 1.36008797E-02;
    COFD[8632] = -2.22801170E+01;
    COFD[8633] = 5.58507108E+00;
    COFD[8634] = -4.81395065E-01;
    COFD[8635] = 1.97276199E-02;
    COFD[8636] = -2.10849000E+01;
    COFD[8637] = 5.35335833E+00;
    COFD[8638] = -4.69065665E-01;
    COFD[8639] = 1.98989604E-02;
    COFD[8640] = -2.22928570E+01;
    COFD[8641] = 5.58507108E+00;
    COFD[8642] = -4.81395065E-01;
    COFD[8643] = 1.97276199E-02;
    COFD[8644] = -2.19319411E+01;
    COFD[8645] = 5.14570932E+00;
    COFD[8646] = -3.99877142E-01;
    COFD[8647] = 1.52199557E-02;
    COFD[8648] = -2.25041734E+01;
    COFD[8649] = 5.51797622E+00;
    COFD[8650] = -4.66229499E-01;
    COFD[8651] = 1.88128348E-02;
    COFD[8652] = -2.21913393E+01;
    COFD[8653] = 5.60175327E+00;
    COFD[8654] = -4.87953216E-01;
    COFD[8655] = 2.01882171E-02;
    COFD[8656] = -2.17842783E+01;
    COFD[8657] = 5.00373919E+00;
    COFD[8658] = -3.76839143E-01;
    COFD[8659] = 1.40386989E-02;
    COFD[8660] = -2.19319411E+01;
    COFD[8661] = 5.14570932E+00;
    COFD[8662] = -3.99877142E-01;
    COFD[8663] = 1.52199557E-02;
    COFD[8664] = -2.18614044E+01;
    COFD[8665] = 5.00373919E+00;
    COFD[8666] = -3.76839143E-01;
    COFD[8667] = 1.40386989E-02;
    COFD[8668] = -2.22948697E+01;
    COFD[8669] = 5.59185582E+00;
    COFD[8670] = -4.91155812E-01;
    COFD[8671] = 2.05043018E-02;
    COFD[8672] = -2.25145418E+01;
    COFD[8673] = 5.49554403E+00;
    COFD[8674] = -4.60936491E-01;
    COFD[8675] = 1.84887572E-02;
    COFD[8676] = -2.25219004E+01;
    COFD[8677] = 5.49554403E+00;
    COFD[8678] = -4.60936491E-01;
    COFD[8679] = 1.84887572E-02;
    COFD[8680] = -2.25060112E+01;
    COFD[8681] = 5.50327119E+00;
    COFD[8682] = -4.63087223E-01;
    COFD[8683] = 1.86271401E-02;
    COFD[8684] = -2.26679870E+01;
    COFD[8685] = 5.52852425E+00;
    COFD[8686] = -4.68277964E-01;
    COFD[8687] = 1.89288127E-02;
    COFD[8688] = -2.19287691E+01;
    COFD[8689] = 4.95026695E+00;
    COFD[8690] = -3.68392434E-01;
    COFD[8691] = 1.36126514E-02;
    COFD[8692] = -2.25758616E+01;
    COFD[8693] = 5.40563818E+00;
    COFD[8694] = -4.44444322E-01;
    COFD[8695] = 1.75813146E-02;
    COFD[8696] = -2.25715533E+01;
    COFD[8697] = 5.41049872E+00;
    COFD[8698] = -4.45356411E-01;
    COFD[8699] = 1.76320470E-02;
    COFD[8700] = -2.25760230E+01;
    COFD[8701] = 5.41049872E+00;
    COFD[8702] = -4.45356411E-01;
    COFD[8703] = 1.76320470E-02;
    COFD[8704] = -2.27125829E+01;
    COFD[8705] = 5.41826700E+00;
    COFD[8706] = -4.46792049E-01;
    COFD[8707] = 1.77112976E-02;
    COFD[8708] = -2.18719802E+01;
    COFD[8709] = 4.88180276E+00;
    COFD[8710] = -3.57591995E-01;
    COFD[8711] = 1.30686372E-02;
    COFD[8712] = -2.25720229E+01;
    COFD[8713] = 5.27220175E+00;
    COFD[8714] = -4.21436175E-01;
    COFD[8715] = 1.63576263E-02;
    COFD[8716] = -2.26508835E+01;
    COFD[8717] = 5.31312101E+00;
    COFD[8718] = -4.28304541E-01;
    COFD[8719] = 1.67176023E-02;
    COFD[8720] = -2.22582201E+01;
    COFD[8721] = 5.12825866E+00;
    COFD[8722] = -3.96982702E-01;
    COFD[8723] = 1.50696010E-02;
    COFD[8724] = -2.19774160E+01;
    COFD[8725] = 4.92889157E+00;
    COFD[8726] = -3.65025286E-01;
    COFD[8727] = 1.34431452E-02;
    COFD[8728] = -2.24161979E+01;
    COFD[8729] = 5.05061421E+00;
    COFD[8730] = -3.84359196E-01;
    COFD[8731] = 1.44214004E-02;
    COFD[8732] = -2.24179759E+01;
    COFD[8733] = 5.05061421E+00;
    COFD[8734] = -3.84359196E-01;
    COFD[8735] = 1.44214004E-02;
    COFD[8736] = -1.97196489E+01;
    COFD[8737] = 4.83750266E+00;
    COFD[8738] = -4.09581452E-01;
    COFD[8739] = 1.76174739E-02;
    COFD[8740] = -1.86234701E+01;
    COFD[8741] = 4.60336076E+00;
    COFD[8742] = -3.81691643E-01;
    COFD[8743] = 1.65085234E-02;
    COFD[8744] = -1.51159870E+01;
    COFD[8745] = 3.64206330E+00;
    COFD[8746] = -2.61313444E-01;
    COFD[8747] = 1.14642754E-02;
    COFD[8748] = -1.92360228E+01;
    COFD[8749] = 5.31542554E+00;
    COFD[8750] = -4.65003780E-01;
    COFD[8751] = 1.97570185E-02;
    COFD[8752] = -1.86469792E+01;
    COFD[8753] = 4.60336076E+00;
    COFD[8754] = -3.81691643E-01;
    COFD[8755] = 1.65085234E-02;
    COFD[8756] = -2.02265558E+01;
    COFD[8757] = 4.58441724E+00;
    COFD[8758] = -3.10392854E-01;
    COFD[8759] = 1.06849990E-02;
    COFD[8760] = -2.00964665E+01;
    COFD[8761] = 4.96870443E+00;
    COFD[8762] = -4.25292447E-01;
    COFD[8763] = 1.82459096E-02;
    COFD[8764] = -2.01062206E+01;
    COFD[8765] = 4.96870443E+00;
    COFD[8766] = -4.25292447E-01;
    COFD[8767] = 1.82459096E-02;
    COFD[8768] = -2.01155735E+01;
    COFD[8769] = 4.96870443E+00;
    COFD[8770] = -4.25292447E-01;
    COFD[8771] = 1.82459096E-02;
    COFD[8772] = -1.85418144E+01;
    COFD[8773] = 4.60336076E+00;
    COFD[8774] = -3.81691643E-01;
    COFD[8775] = 1.65085234E-02;
    COFD[8776] = -2.12219728E+01;
    COFD[8777] = 4.88535789E+00;
    COFD[8778] = -3.58153894E-01;
    COFD[8779] = 1.30969624E-02;
    COFD[8780] = -2.08833669E+01;
    COFD[8781] = 5.30526648E+00;
    COFD[8782] = -4.63785596E-01;
    COFD[8783] = 1.97079873E-02;
    COFD[8784] = -2.23772680E+01;
    COFD[8785] = 5.59425354E+00;
    COFD[8786] = -4.86232980E-01;
    COFD[8787] = 2.00835981E-02;
    COFD[8788] = -1.97422209E+01;
    COFD[8789] = 4.84249900E+00;
    COFD[8790] = -4.10120448E-01;
    COFD[8791] = 1.76363500E-02;
    COFD[8792] = -2.12330900E+01;
    COFD[8793] = 4.88535789E+00;
    COFD[8794] = -3.58153894E-01;
    COFD[8795] = 1.30969624E-02;
    COFD[8796] = -2.08833669E+01;
    COFD[8797] = 5.30526648E+00;
    COFD[8798] = -4.63785596E-01;
    COFD[8799] = 1.97079873E-02;
    COFD[8800] = -2.09108261E+01;
    COFD[8801] = 5.30526648E+00;
    COFD[8802] = -4.63785596E-01;
    COFD[8803] = 1.97079873E-02;
    COFD[8804] = -2.18837863E+01;
    COFD[8805] = 5.22103227E+00;
    COFD[8806] = -4.12481899E-01;
    COFD[8807] = 1.58782021E-02;
    COFD[8808] = -2.08427678E+01;
    COFD[8809] = 5.27674330E+00;
    COFD[8810] = -4.60336155E-01;
    COFD[8811] = 1.95680191E-02;
    COFD[8812] = -2.14111310E+01;
    COFD[8813] = 4.96219227E+00;
    COFD[8814] = -3.70270843E-01;
    COFD[8815] = 1.37072211E-02;
    COFD[8816] = -2.23993836E+01;
    COFD[8817] = 5.58952429E+00;
    COFD[8818] = -4.85012530E-01;
    COFD[8819] = 2.00062142E-02;
    COFD[8820] = -2.23882886E+01;
    COFD[8821] = 5.58952429E+00;
    COFD[8822] = -4.85012530E-01;
    COFD[8823] = 2.00062142E-02;
    COFD[8824] = -2.19282979E+01;
    COFD[8825] = 5.15446948E+00;
    COFD[8826] = -4.01332769E-01;
    COFD[8827] = 1.52956262E-02;
    COFD[8828] = -2.19534987E+01;
    COFD[8829] = 5.15446948E+00;
    COFD[8830] = -4.01332769E-01;
    COFD[8831] = 1.52956262E-02;
    COFD[8832] = -2.15236645E+01;
    COFD[8833] = 4.96219227E+00;
    COFD[8834] = -3.70270843E-01;
    COFD[8835] = 1.37072211E-02;
    COFD[8836] = -2.15293564E+01;
    COFD[8837] = 4.96219227E+00;
    COFD[8838] = -3.70270843E-01;
    COFD[8839] = 1.37072211E-02;
    COFD[8840] = -2.22609256E+01;
    COFD[8841] = 5.58490856E+00;
    COFD[8842] = -4.81588720E-01;
    COFD[8843] = 1.97445317E-02;
    COFD[8844] = -2.10575083E+01;
    COFD[8845] = 5.35019396E+00;
    COFD[8846] = -4.68809590E-01;
    COFD[8847] = 1.98941097E-02;
    COFD[8848] = -2.22737428E+01;
    COFD[8849] = 5.58490856E+00;
    COFD[8850] = -4.81588720E-01;
    COFD[8851] = 1.97445317E-02;
    COFD[8852] = -2.19349837E+01;
    COFD[8853] = 5.15446948E+00;
    COFD[8854] = -4.01332769E-01;
    COFD[8855] = 1.52956262E-02;
    COFD[8856] = -2.24965286E+01;
    COFD[8857] = 5.52198915E+00;
    COFD[8858] = -4.67014474E-01;
    COFD[8859] = 1.88574253E-02;
    COFD[8860] = -2.21792065E+01;
    COFD[8861] = 5.60465338E+00;
    COFD[8862] = -4.88572478E-01;
    COFD[8863] = 2.02248525E-02;
    COFD[8864] = -2.17946700E+01;
    COFD[8865] = 5.01521891E+00;
    COFD[8866] = -3.78672535E-01;
    COFD[8867] = 1.41317315E-02;
    COFD[8868] = -2.19349837E+01;
    COFD[8869] = 5.15446948E+00;
    COFD[8870] = -4.01332769E-01;
    COFD[8871] = 1.52956262E-02;
    COFD[8872] = -2.18724957E+01;
    COFD[8873] = 5.01521891E+00;
    COFD[8874] = -3.78672535E-01;
    COFD[8875] = 1.41317315E-02;
    COFD[8876] = -2.22685220E+01;
    COFD[8877] = 5.58776204E+00;
    COFD[8878] = -4.90769522E-01;
    COFD[8879] = 2.04931464E-02;
    COFD[8880] = -2.25028406E+01;
    COFD[8881] = 5.49776513E+00;
    COFD[8882] = -4.61463030E-01;
    COFD[8883] = 1.85209236E-02;
    COFD[8884] = -2.25102565E+01;
    COFD[8885] = 5.49776513E+00;
    COFD[8886] = -4.61463030E-01;
    COFD[8887] = 1.85209236E-02;
    COFD[8888] = -2.24931486E+01;
    COFD[8889] = 5.50509817E+00;
    COFD[8890] = -4.63572794E-01;
    COFD[8891] = 1.86581046E-02;
    COFD[8892] = -2.26623034E+01;
    COFD[8893] = 5.53286772E+00;
    COFD[8894] = -4.69109018E-01;
    COFD[8895] = 1.89755392E-02;
    COFD[8896] = -2.19456782E+01;
    COFD[8897] = 4.96368178E+00;
    COFD[8898] = -3.70505465E-01;
    COFD[8899] = 1.37190339E-02;
    COFD[8900] = -2.25715533E+01;
    COFD[8901] = 5.41049872E+00;
    COFD[8902] = -4.45356411E-01;
    COFD[8903] = 1.76320470E-02;
    COFD[8904] = -2.25672005E+01;
    COFD[8905] = 5.41536807E+00;
    COFD[8906] = -4.46269562E-01;
    COFD[8907] = 1.76828228E-02;
    COFD[8908] = -2.25717119E+01;
    COFD[8909] = 5.41536807E+00;
    COFD[8910] = -4.46269562E-01;
    COFD[8911] = 1.76828228E-02;
    COFD[8912] = -2.27104617E+01;
    COFD[8913] = 5.42362435E+00;
    COFD[8914] = -4.47767764E-01;
    COFD[8915] = 1.77647205E-02;
    COFD[8916] = -2.18873269E+01;
    COFD[8917] = 4.89420156E+00;
    COFD[8918] = -3.59552981E-01;
    COFD[8919] = 1.31675149E-02;
    COFD[8920] = -2.25735380E+01;
    COFD[8921] = 5.27888680E+00;
    COFD[8922] = -4.22608469E-01;
    COFD[8923] = 1.64205533E-02;
    COFD[8924] = -2.26553469E+01;
    COFD[8925] = 5.32093606E+00;
    COFD[8926] = -4.29624801E-01;
    COFD[8927] = 1.67869730E-02;
    COFD[8928] = -2.22649250E+01;
    COFD[8929] = 5.13737172E+00;
    COFD[8930] = -3.98493102E-01;
    COFD[8931] = 1.51480250E-02;
    COFD[8932] = -2.19948202E+01;
    COFD[8933] = 4.94219368E+00;
    COFD[8934] = -3.67120797E-01;
    COFD[8935] = 1.35486343E-02;
    COFD[8936] = -2.24284563E+01;
    COFD[8937] = 5.06106414E+00;
    COFD[8938] = -3.86053039E-01;
    COFD[8939] = 1.45081784E-02;
    COFD[8940] = -2.24302556E+01;
    COFD[8941] = 5.06106414E+00;
    COFD[8942] = -3.86053039E-01;
    COFD[8943] = 1.45081784E-02;
    COFD[8944] = -1.97226856E+01;
    COFD[8945] = 4.83750266E+00;
    COFD[8946] = -4.09581452E-01;
    COFD[8947] = 1.76174739E-02;
    COFD[8948] = -1.86254955E+01;
    COFD[8949] = 4.60336076E+00;
    COFD[8950] = -3.81691643E-01;
    COFD[8951] = 1.65085234E-02;
    COFD[8952] = -1.51163041E+01;
    COFD[8953] = 3.64206330E+00;
    COFD[8954] = -2.61313444E-01;
    COFD[8955] = 1.14642754E-02;
    COFD[8956] = -1.92361841E+01;
    COFD[8957] = 5.31542554E+00;
    COFD[8958] = -4.65003780E-01;
    COFD[8959] = 1.97570185E-02;
    COFD[8960] = -1.86491023E+01;
    COFD[8961] = 4.60336076E+00;
    COFD[8962] = -3.81691643E-01;
    COFD[8963] = 1.65085234E-02;
    COFD[8964] = -2.02287739E+01;
    COFD[8965] = 4.58441724E+00;
    COFD[8966] = -3.10392854E-01;
    COFD[8967] = 1.06849990E-02;
    COFD[8968] = -2.00997774E+01;
    COFD[8969] = 4.96870443E+00;
    COFD[8970] = -4.25292447E-01;
    COFD[8971] = 1.82459096E-02;
    COFD[8972] = -2.01095969E+01;
    COFD[8973] = 4.96870443E+00;
    COFD[8974] = -4.25292447E-01;
    COFD[8975] = 1.82459096E-02;
    COFD[8976] = -2.01190139E+01;
    COFD[8977] = 4.96870443E+00;
    COFD[8978] = -4.25292447E-01;
    COFD[8979] = 1.82459096E-02;
    COFD[8980] = -1.85435341E+01;
    COFD[8981] = 4.60336076E+00;
    COFD[8982] = -3.81691643E-01;
    COFD[8983] = 1.65085234E-02;
    COFD[8984] = -2.12250811E+01;
    COFD[8985] = 4.88535789E+00;
    COFD[8986] = -3.58153894E-01;
    COFD[8987] = 1.30969624E-02;
    COFD[8988] = -2.08851929E+01;
    COFD[8989] = 5.30526648E+00;
    COFD[8990] = -4.63785596E-01;
    COFD[8991] = 1.97079873E-02;
    COFD[8992] = -2.23812726E+01;
    COFD[8993] = 5.59425354E+00;
    COFD[8994] = -4.86232980E-01;
    COFD[8995] = 2.00835981E-02;
    COFD[8996] = -1.97452574E+01;
    COFD[8997] = 4.84249900E+00;
    COFD[8998] = -4.10120448E-01;
    COFD[8999] = 1.76363500E-02;
    COFD[9000] = -2.12362684E+01;
    COFD[9001] = 4.88535789E+00;
    COFD[9002] = -3.58153894E-01;
    COFD[9003] = 1.30969624E-02;
    COFD[9004] = -2.08851929E+01;
    COFD[9005] = 5.30526648E+00;
    COFD[9006] = -4.63785596E-01;
    COFD[9007] = 1.97079873E-02;
    COFD[9008] = -2.09127554E+01;
    COFD[9009] = 5.30526648E+00;
    COFD[9010] = -4.63785596E-01;
    COFD[9011] = 1.97079873E-02;
    COFD[9012] = -2.18870332E+01;
    COFD[9013] = 5.22103227E+00;
    COFD[9014] = -4.12481899E-01;
    COFD[9015] = 1.58782021E-02;
    COFD[9016] = -2.08447974E+01;
    COFD[9017] = 5.27674330E+00;
    COFD[9018] = -4.60336155E-01;
    COFD[9019] = 1.95680191E-02;
    COFD[9020] = -2.14144448E+01;
    COFD[9021] = 4.96219227E+00;
    COFD[9022] = -3.70270843E-01;
    COFD[9023] = 1.37072211E-02;
    COFD[9024] = -2.24025650E+01;
    COFD[9025] = 5.58952429E+00;
    COFD[9026] = -4.85012530E-01;
    COFD[9027] = 2.00062142E-02;
    COFD[9028] = -2.23913999E+01;
    COFD[9029] = 5.58952429E+00;
    COFD[9030] = -4.85012530E-01;
    COFD[9031] = 2.00062142E-02;
    COFD[9032] = -2.19322003E+01;
    COFD[9033] = 5.15446948E+00;
    COFD[9034] = -4.01332769E-01;
    COFD[9035] = 1.52956262E-02;
    COFD[9036] = -2.19576037E+01;
    COFD[9037] = 5.15446948E+00;
    COFD[9038] = -4.01332769E-01;
    COFD[9039] = 1.52956262E-02;
    COFD[9040] = -2.15278182E+01;
    COFD[9041] = 4.96219227E+00;
    COFD[9042] = -3.70270843E-01;
    COFD[9043] = 1.37072211E-02;
    COFD[9044] = -2.15335578E+01;
    COFD[9045] = 4.96219227E+00;
    COFD[9046] = -3.70270843E-01;
    COFD[9047] = 1.37072211E-02;
    COFD[9048] = -2.22638165E+01;
    COFD[9049] = 5.58490856E+00;
    COFD[9050] = -4.81588720E-01;
    COFD[9051] = 1.97445317E-02;
    COFD[9052] = -2.10613569E+01;
    COFD[9053] = 5.35019396E+00;
    COFD[9054] = -4.68809590E-01;
    COFD[9055] = 1.98941097E-02;
    COFD[9056] = -2.22767090E+01;
    COFD[9057] = 5.58490856E+00;
    COFD[9058] = -4.81588720E-01;
    COFD[9059] = 1.97445317E-02;
    COFD[9060] = -2.19389389E+01;
    COFD[9061] = 5.15446948E+00;
    COFD[9062] = -4.01332769E-01;
    COFD[9063] = 1.52956262E-02;
    COFD[9064] = -2.25004333E+01;
    COFD[9065] = 5.52198915E+00;
    COFD[9066] = -4.67014474E-01;
    COFD[9067] = 1.88574253E-02;
    COFD[9068] = -2.21822461E+01;
    COFD[9069] = 5.60465338E+00;
    COFD[9070] = -4.88572478E-01;
    COFD[9071] = 2.02248525E-02;
    COFD[9072] = -2.17987275E+01;
    COFD[9073] = 5.01521891E+00;
    COFD[9074] = -3.78672535E-01;
    COFD[9075] = 1.41317315E-02;
    COFD[9076] = -2.19389389E+01;
    COFD[9077] = 5.15446948E+00;
    COFD[9078] = -4.01332769E-01;
    COFD[9079] = 1.52956262E-02;
    COFD[9080] = -2.18772398E+01;
    COFD[9081] = 5.01521891E+00;
    COFD[9082] = -3.78672535E-01;
    COFD[9083] = 1.41317315E-02;
    COFD[9084] = -2.22722046E+01;
    COFD[9085] = 5.58776204E+00;
    COFD[9086] = -4.90769522E-01;
    COFD[9087] = 2.04931464E-02;
    COFD[9088] = -2.25065805E+01;
    COFD[9089] = 5.49776513E+00;
    COFD[9090] = -4.61463030E-01;
    COFD[9091] = 1.85209236E-02;
    COFD[9092] = -2.25140525E+01;
    COFD[9093] = 5.49776513E+00;
    COFD[9094] = -4.61463030E-01;
    COFD[9095] = 1.85209236E-02;
    COFD[9096] = -2.24969995E+01;
    COFD[9097] = 5.50509817E+00;
    COFD[9098] = -4.63572794E-01;
    COFD[9099] = 1.86581046E-02;
    COFD[9100] = -2.26662608E+01;
    COFD[9101] = 5.53286772E+00;
    COFD[9102] = -4.69109018E-01;
    COFD[9103] = 1.89755392E-02;
    COFD[9104] = -2.19508859E+01;
    COFD[9105] = 4.96368178E+00;
    COFD[9106] = -3.70505465E-01;
    COFD[9107] = 1.37190339E-02;
    COFD[9108] = -2.25760230E+01;
    COFD[9109] = 5.41049872E+00;
    COFD[9110] = -4.45356411E-01;
    COFD[9111] = 1.76320470E-02;
    COFD[9112] = -2.25717119E+01;
    COFD[9113] = 5.41536807E+00;
    COFD[9114] = -4.46269562E-01;
    COFD[9115] = 1.76828228E-02;
    COFD[9116] = -2.25762645E+01;
    COFD[9117] = 5.41536807E+00;
    COFD[9118] = -4.46269562E-01;
    COFD[9119] = 1.76828228E-02;
    COFD[9120] = -2.27150546E+01;
    COFD[9121] = 5.42362435E+00;
    COFD[9122] = -4.47767764E-01;
    COFD[9123] = 1.77647205E-02;
    COFD[9124] = -2.18929084E+01;
    COFD[9125] = 4.89420156E+00;
    COFD[9126] = -3.59552981E-01;
    COFD[9127] = 1.31675149E-02;
    COFD[9128] = -2.25785614E+01;
    COFD[9129] = 5.27888680E+00;
    COFD[9130] = -4.22608469E-01;
    COFD[9131] = 1.64205533E-02;
    COFD[9132] = -2.26604027E+01;
    COFD[9133] = 5.32093606E+00;
    COFD[9134] = -4.29624801E-01;
    COFD[9135] = 1.67869730E-02;
    COFD[9136] = -2.22700127E+01;
    COFD[9137] = 5.13737172E+00;
    COFD[9138] = -3.98493102E-01;
    COFD[9139] = 1.51480250E-02;
    COFD[9140] = -2.20002783E+01;
    COFD[9141] = 4.94219368E+00;
    COFD[9142] = -3.67120797E-01;
    COFD[9143] = 1.35486343E-02;
    COFD[9144] = -2.24342646E+01;
    COFD[9145] = 5.06106414E+00;
    COFD[9146] = -3.86053039E-01;
    COFD[9147] = 1.45081784E-02;
    COFD[9148] = -2.24360850E+01;
    COFD[9149] = 5.06106414E+00;
    COFD[9150] = -3.86053039E-01;
    COFD[9151] = 1.45081784E-02;
    COFD[9152] = -1.98374654E+01;
    COFD[9153] = 4.82871870E+00;
    COFD[9154] = -4.08567726E-01;
    COFD[9155] = 1.75785896E-02;
    COFD[9156] = -1.87433618E+01;
    COFD[9157] = 4.58956960E+00;
    COFD[9158] = -3.79964215E-01;
    COFD[9159] = 1.64361138E-02;
    COFD[9160] = -1.52503668E+01;
    COFD[9161] = 3.63657318E+00;
    COFD[9162] = -2.60678457E-01;
    COFD[9163] = 1.14400550E-02;
    COFD[9164] = -1.93693740E+01;
    COFD[9165] = 5.30286598E+00;
    COFD[9166] = -4.63495567E-01;
    COFD[9167] = 1.96962203E-02;
    COFD[9168] = -1.87670637E+01;
    COFD[9169] = 4.58956960E+00;
    COFD[9170] = -3.79964215E-01;
    COFD[9171] = 1.64361138E-02;
    COFD[9172] = -2.04186424E+01;
    COFD[9173] = 4.60117690E+00;
    COFD[9174] = -3.13067257E-01;
    COFD[9175] = 1.08202310E-02;
    COFD[9176] = -2.02121663E+01;
    COFD[9177] = 4.95786261E+00;
    COFD[9178] = -4.24013131E-01;
    COFD[9179] = 1.81955669E-02;
    COFD[9180] = -2.02220498E+01;
    COFD[9181] = 4.95786261E+00;
    COFD[9182] = -4.24013131E-01;
    COFD[9183] = 1.81955669E-02;
    COFD[9184] = -2.02315293E+01;
    COFD[9185] = 4.95786261E+00;
    COFD[9186] = -4.24013131E-01;
    COFD[9187] = 1.81955669E-02;
    COFD[9188] = -1.86611033E+01;
    COFD[9189] = 4.58956960E+00;
    COFD[9190] = -3.79964215E-01;
    COFD[9191] = 1.64361138E-02;
    COFD[9192] = -2.14030307E+01;
    COFD[9193] = 4.90439970E+00;
    COFD[9194] = -3.61162615E-01;
    COFD[9195] = 1.32486109E-02;
    COFD[9196] = -2.09847776E+01;
    COFD[9197] = 5.29210705E+00;
    COFD[9198] = -4.62193217E-01;
    COFD[9199] = 1.96432872E-02;
    COFD[9200] = -2.25216613E+01;
    COFD[9201] = 5.59792043E+00;
    COFD[9202] = -4.87076900E-01;
    COFD[9203] = 2.01350364E-02;
    COFD[9204] = -1.98616115E+01;
    COFD[9205] = 4.83466791E+00;
    COFD[9206] = -4.09252052E-01;
    COFD[9207] = 1.76047341E-02;
    COFD[9208] = -2.14142864E+01;
    COFD[9209] = 4.90439970E+00;
    COFD[9210] = -3.61162615E-01;
    COFD[9211] = 1.32486109E-02;
    COFD[9212] = -2.09847776E+01;
    COFD[9213] = 5.29210705E+00;
    COFD[9214] = -4.62193217E-01;
    COFD[9215] = 1.96432872E-02;
    COFD[9216] = -2.10124405E+01;
    COFD[9217] = 5.29210705E+00;
    COFD[9218] = -4.62193217E-01;
    COFD[9219] = 1.96432872E-02;
    COFD[9220] = -2.20426031E+01;
    COFD[9221] = 5.23117744E+00;
    COFD[9222] = -4.14243780E-01;
    COFD[9223] = 1.59721173E-02;
    COFD[9224] = -2.09461018E+01;
    COFD[9225] = 5.26396793E+00;
    COFD[9226] = -4.58812213E-01;
    COFD[9227] = 1.95072180E-02;
    COFD[9228] = -2.15952753E+01;
    COFD[9229] = 4.98271982E+00;
    COFD[9230] = -3.73502341E-01;
    COFD[9231] = 1.38698700E-02;
    COFD[9232] = -2.25300734E+01;
    COFD[9233] = 5.59173268E+00;
    COFD[9234] = -4.85654660E-01;
    COFD[9235] = 2.00483698E-02;
    COFD[9236] = -2.25188399E+01;
    COFD[9237] = 5.59173268E+00;
    COFD[9238] = -4.85654660E-01;
    COFD[9239] = 2.00483698E-02;
    COFD[9240] = -2.20891322E+01;
    COFD[9241] = 5.16679492E+00;
    COFD[9242] = -4.03405751E-01;
    COFD[9243] = 1.54041741E-02;
    COFD[9244] = -2.21147341E+01;
    COFD[9245] = 5.16679492E+00;
    COFD[9246] = -4.03405751E-01;
    COFD[9247] = 1.54041741E-02;
    COFD[9248] = -2.17094710E+01;
    COFD[9249] = 4.98271982E+00;
    COFD[9250] = -3.73502341E-01;
    COFD[9251] = 1.38698700E-02;
    COFD[9252] = -2.17152574E+01;
    COFD[9253] = 4.98271982E+00;
    COFD[9254] = -3.73502341E-01;
    COFD[9255] = 1.38698700E-02;
    COFD[9256] = -2.23950513E+01;
    COFD[9257] = 5.58492366E+00;
    COFD[9258] = -4.81921868E-01;
    COFD[9259] = 1.97721534E-02;
    COFD[9260] = -2.12027460E+01;
    COFD[9261] = 5.34410059E+00;
    COFD[9262] = -4.68233157E-01;
    COFD[9263] = 1.98777314E-02;
    COFD[9264] = -2.24080172E+01;
    COFD[9265] = 5.58492366E+00;
    COFD[9266] = -4.81921868E-01;
    COFD[9267] = 1.97721534E-02;
    COFD[9268] = -2.20959225E+01;
    COFD[9269] = 5.16679492E+00;
    COFD[9270] = -4.03405751E-01;
    COFD[9271] = 1.54041741E-02;
    COFD[9272] = -2.26411013E+01;
    COFD[9273] = 5.52830072E+00;
    COFD[9274] = -4.68235018E-01;
    COFD[9275] = 1.89263933E-02;
    COFD[9276] = -2.23250359E+01;
    COFD[9277] = 5.60776666E+00;
    COFD[9278] = -4.89319792E-01;
    COFD[9279] = 2.02710069E-02;
    COFD[9280] = -2.19609064E+01;
    COFD[9281] = 5.03230486E+00;
    COFD[9282] = -3.81405277E-01;
    COFD[9283] = 1.42705027E-02;
    COFD[9284] = -2.20959225E+01;
    COFD[9285] = 5.16679492E+00;
    COFD[9286] = -4.03405751E-01;
    COFD[9287] = 1.54041741E-02;
    COFD[9288] = -2.20400927E+01;
    COFD[9289] = 5.03230486E+00;
    COFD[9290] = -3.81405277E-01;
    COFD[9291] = 1.42705027E-02;
    COFD[9292] = -2.23840677E+01;
    COFD[9293] = 5.58123742E+00;
    COFD[9294] = -4.90135630E-01;
    COFD[9295] = 2.04738180E-02;
    COFD[9296] = -2.26360108E+01;
    COFD[9297] = 5.50023958E+00;
    COFD[9298] = -4.62136179E-01;
    COFD[9299] = 1.85639061E-02;
    COFD[9300] = -2.26435378E+01;
    COFD[9301] = 5.50023958E+00;
    COFD[9302] = -4.62136179E-01;
    COFD[9303] = 1.85639061E-02;
    COFD[9304] = -2.26299936E+01;
    COFD[9305] = 5.50881574E+00;
    COFD[9306] = -4.64448886E-01;
    COFD[9307] = 1.87118881E-02;
    COFD[9308] = -2.27989630E+01;
    COFD[9309] = 5.53955653E+00;
    COFD[9310] = -4.70381353E-01;
    COFD[9311] = 1.90468698E-02;
    COFD[9312] = -2.21147103E+01;
    COFD[9313] = 4.98427447E+00;
    COFD[9314] = -3.73746896E-01;
    COFD[9315] = 1.38821805E-02;
    COFD[9316] = -2.27125829E+01;
    COFD[9317] = 5.41826700E+00;
    COFD[9318] = -4.46792049E-01;
    COFD[9319] = 1.77112976E-02;
    COFD[9320] = -2.27104617E+01;
    COFD[9321] = 5.42362435E+00;
    COFD[9322] = -4.47767764E-01;
    COFD[9323] = 1.77647205E-02;
    COFD[9324] = -2.27150546E+01;
    COFD[9325] = 5.42362435E+00;
    COFD[9326] = -4.47767764E-01;
    COFD[9327] = 1.77647205E-02;
    COFD[9328] = -2.28473186E+01;
    COFD[9329] = 5.43214736E+00;
    COFD[9330] = -4.49307302E-01;
    COFD[9331] = 1.78487187E-02;
    COFD[9332] = -2.20523730E+01;
    COFD[9333] = 4.91371677E+00;
    COFD[9334] = -3.62632206E-01;
    COFD[9335] = 1.33226347E-02;
    COFD[9336] = -2.27197587E+01;
    COFD[9337] = 5.29216864E+00;
    COFD[9338] = -4.24828662E-01;
    COFD[9339] = 1.65366835E-02;
    COFD[9340] = -2.27940520E+01;
    COFD[9341] = 5.33097599E+00;
    COFD[9342] = -4.31367349E-01;
    COFD[9343] = 1.68798869E-02;
    COFD[9344] = -2.24171125E+01;
    COFD[9345] = 5.15109664E+00;
    COFD[9346] = -4.00765892E-01;
    COFD[9347] = 1.52659560E-02;
    COFD[9348] = -2.21597603E+01;
    COFD[9349] = 4.96243463E+00;
    COFD[9350] = -3.70309018E-01;
    COFD[9351] = 1.37091432E-02;
    COFD[9352] = -2.25733338E+01;
    COFD[9353] = 5.07648425E+00;
    COFD[9354] = -3.88560019E-01;
    COFD[9355] = 1.46368353E-02;
    COFD[9356] = -2.25751750E+01;
    COFD[9357] = 5.07648425E+00;
    COFD[9358] = -3.88560019E-01;
    COFD[9359] = 1.46368353E-02;
    COFD[9360] = -2.10643735E+01;
    COFD[9361] = 5.22604478E+00;
    COFD[9362] = -4.54378127E-01;
    COFD[9363] = 1.93342248E-02;
    COFD[9364] = -2.00070284E+01;
    COFD[9365] = 5.02095434E+00;
    COFD[9366] = -4.31496874E-01;
    COFD[9367] = 1.84920392E-02;
    COFD[9368] = -1.65048875E+01;
    COFD[9369] = 4.10792536E+00;
    COFD[9370] = -3.21060656E-01;
    COFD[9371] = 1.40287900E-02;
    COFD[9372] = -2.02592914E+01;
    COFD[9373] = 5.56701235E+00;
    COFD[9374] = -4.88925090E-01;
    COFD[9375] = 2.04461948E-02;
    COFD[9376] = -2.00328044E+01;
    COFD[9377] = 5.02095434E+00;
    COFD[9378] = -4.31496874E-01;
    COFD[9379] = 1.84920392E-02;
    COFD[9380] = -1.83939699E+01;
    COFD[9381] = 3.59019527E+00;
    COFD[9382] = -1.58702132E-01;
    COFD[9383] = 3.23316765E-03;
    COFD[9384] = -2.14416336E+01;
    COFD[9385] = 5.35040988E+00;
    COFD[9386] = -4.68827063E-01;
    COFD[9387] = 1.98944407E-02;
    COFD[9388] = -2.14529967E+01;
    COFD[9389] = 5.35040988E+00;
    COFD[9390] = -4.68827063E-01;
    COFD[9391] = 1.98944407E-02;
    COFD[9392] = -2.14639274E+01;
    COFD[9393] = 5.35040988E+00;
    COFD[9394] = -4.68827063E-01;
    COFD[9395] = 1.98944407E-02;
    COFD[9396] = -1.99183435E+01;
    COFD[9397] = 5.02095434E+00;
    COFD[9398] = -4.31496874E-01;
    COFD[9399] = 1.84920392E-02;
    COFD[9400] = -1.96731865E+01;
    COFD[9401] = 4.00653795E+00;
    COFD[9402] = -2.22005804E-01;
    COFD[9403] = 6.33194910E-03;
    COFD[9404] = -2.19250377E+01;
    COFD[9405] = 5.56282156E+00;
    COFD[9406] = -4.88585679E-01;
    COFD[9407] = 2.04395879E-02;
    COFD[9408] = -2.25838099E+01;
    COFD[9409] = 5.45615714E+00;
    COFD[9410] = -4.53643844E-01;
    COFD[9411] = 1.80854821E-02;
    COFD[9412] = -2.10924694E+01;
    COFD[9413] = 5.23339224E+00;
    COFD[9414] = -4.55230780E-01;
    COFD[9415] = 1.93672146E-02;
    COFD[9416] = -1.96860113E+01;
    COFD[9417] = 4.00653795E+00;
    COFD[9418] = -2.22005804E-01;
    COFD[9419] = 6.33194910E-03;
    COFD[9420] = -2.19250377E+01;
    COFD[9421] = 5.56282156E+00;
    COFD[9422] = -4.88585679E-01;
    COFD[9423] = 2.04395879E-02;
    COFD[9424] = -2.19548723E+01;
    COFD[9425] = 5.56282156E+00;
    COFD[9426] = -4.88585679E-01;
    COFD[9427] = 2.04395879E-02;
    COFD[9428] = -2.08143755E+01;
    COFD[9429] = 4.54213239E+00;
    COFD[9430] = -3.03786739E-01;
    COFD[9431] = 1.03552672E-02;
    COFD[9432] = -2.19244555E+01;
    COFD[9433] = 5.54986547E+00;
    COFD[9434] = -4.87420926E-01;
    COFD[9435] = 2.04095097E-02;
    COFD[9436] = -1.99604682E+01;
    COFD[9437] = 4.12245214E+00;
    COFD[9438] = -2.39476227E-01;
    COFD[9439] = 7.18400558E-03;
    COFD[9440] = -2.25553202E+01;
    COFD[9441] = 5.44166443E+00;
    COFD[9442] = -4.51021243E-01;
    COFD[9443] = 1.79421190E-02;
    COFD[9444] = -2.25425191E+01;
    COFD[9445] = 5.44166443E+00;
    COFD[9446] = -4.51021243E-01;
    COFD[9447] = 1.79421190E-02;
    COFD[9448] = -2.07557953E+01;
    COFD[9449] = 4.42680848E+00;
    COFD[9450] = -2.85885288E-01;
    COFD[9451] = 9.46483934E-03;
    COFD[9452] = -2.07861367E+01;
    COFD[9453] = 4.42680848E+00;
    COFD[9454] = -2.85885288E-01;
    COFD[9455] = 9.46483934E-03;
    COFD[9456] = -2.00940426E+01;
    COFD[9457] = 4.12245214E+00;
    COFD[9458] = -2.39476227E-01;
    COFD[9459] = 7.18400558E-03;
    COFD[9460] = -2.01009567E+01;
    COFD[9461] = 4.12245214E+00;
    COFD[9462] = -2.39476227E-01;
    COFD[9463] = 7.18400558E-03;
    COFD[9464] = -2.22709427E+01;
    COFD[9465] = 5.37360713E+00;
    COFD[9466] = -4.38661889E-01;
    COFD[9467] = 1.72661628E-02;
    COFD[9468] = -2.21307579E+01;
    COFD[9469] = 5.58979675E+00;
    COFD[9470] = -4.90962731E-01;
    COFD[9471] = 2.04987927E-02;
    COFD[9472] = -2.22855755E+01;
    COFD[9473] = 5.37360713E+00;
    COFD[9474] = -4.38661889E-01;
    COFD[9475] = 1.72661628E-02;
    COFD[9476] = -2.07638147E+01;
    COFD[9477] = 4.42680848E+00;
    COFD[9478] = -2.85885288E-01;
    COFD[9479] = 9.46483934E-03;
    COFD[9480] = -2.22139496E+01;
    COFD[9481] = 5.17488844E+00;
    COFD[9482] = -4.04758505E-01;
    COFD[9483] = 1.54748177E-02;
    COFD[9484] = -2.23935500E+01;
    COFD[9485] = 5.47922490E+00;
    COFD[9486] = -4.57847893E-01;
    COFD[9487] = 1.83161707E-02;
    COFD[9488] = -2.04142326E+01;
    COFD[9489] = 4.20149142E+00;
    COFD[9490] = -2.51432163E-01;
    COFD[9491] = 7.76854246E-03;
    COFD[9492] = -2.07638147E+01;
    COFD[9493] = 4.42680848E+00;
    COFD[9494] = -2.85885288E-01;
    COFD[9495] = 9.46483934E-03;
    COFD[9496] = -2.05098287E+01;
    COFD[9497] = 4.20149142E+00;
    COFD[9498] = -2.51432163E-01;
    COFD[9499] = 7.76854246E-03;
    COFD[9500] = -2.27122564E+01;
    COFD[9501] = 5.56278679E+00;
    COFD[9502] = -4.74794778E-01;
    COFD[9503] = 1.92940001E-02;
    COFD[9504] = -2.20599362E+01;
    COFD[9505] = 5.08417640E+00;
    COFD[9506] = -3.89810534E-01;
    COFD[9507] = 1.47010214E-02;
    COFD[9508] = -2.20687596E+01;
    COFD[9509] = 5.08417640E+00;
    COFD[9510] = -3.89810534E-01;
    COFD[9511] = 1.47010214E-02;
    COFD[9512] = -2.21535971E+01;
    COFD[9513] = 5.13453409E+00;
    COFD[9514] = -3.98022439E-01;
    COFD[9515] = 1.51235760E-02;
    COFD[9516] = -2.24028537E+01;
    COFD[9517] = 5.19900179E+00;
    COFD[9518] = -4.08748226E-01;
    COFD[9519] = 1.56820407E-02;
    COFD[9520] = -2.05271615E+01;
    COFD[9521] = 4.12444157E+00;
    COFD[9522] = -2.39777376E-01;
    COFD[9523] = 7.19872269E-03;
    COFD[9524] = -2.18719802E+01;
    COFD[9525] = 4.88180276E+00;
    COFD[9526] = -3.57591995E-01;
    COFD[9527] = 1.30686372E-02;
    COFD[9528] = -2.18873269E+01;
    COFD[9529] = 4.89420156E+00;
    COFD[9530] = -3.59552981E-01;
    COFD[9531] = 1.31675149E-02;
    COFD[9532] = -2.18929084E+01;
    COFD[9533] = 4.89420156E+00;
    COFD[9534] = -3.59552981E-01;
    COFD[9535] = 1.31675149E-02;
    COFD[9536] = -2.20523730E+01;
    COFD[9537] = 4.91371677E+00;
    COFD[9538] = -3.62632206E-01;
    COFD[9539] = 1.33226347E-02;
    COFD[9540] = -2.03937742E+01;
    COFD[9541] = 4.02033531E+00;
    COFD[9542] = -2.24082500E-01;
    COFD[9543] = 6.43305206E-03;
    COFD[9544] = -2.16466727E+01;
    COFD[9545] = 4.65048212E+00;
    COFD[9546] = -3.20931552E-01;
    COFD[9547] = 1.12185393E-02;
    COFD[9548] = -2.17749249E+01;
    COFD[9549] = 4.71207875E+00;
    COFD[9550] = -3.30658500E-01;
    COFD[9551] = 1.17082011E-02;
    COFD[9552] = -2.10757112E+01;
    COFD[9553] = 4.39521460E+00;
    COFD[9554] = -2.81028854E-01;
    COFD[9555] = 9.22466916E-03;
    COFD[9556] = -2.05583124E+01;
    COFD[9557] = 4.09420232E+00;
    COFD[9558] = -2.35210019E-01;
    COFD[9559] = 6.97573395E-03;
    COFD[9560] = -2.11438235E+01;
    COFD[9561] = 4.27612828E+00;
    COFD[9562] = -2.62774610E-01;
    COFD[9563] = 8.32471127E-03;
    COFD[9564] = -2.11462093E+01;
    COFD[9565] = 4.27612828E+00;
    COFD[9566] = -2.62774610E-01;
    COFD[9567] = 8.32471127E-03;
    COFD[9568] = -2.03706752E+01;
    COFD[9569] = 4.98803076E+00;
    COFD[9570] = -4.27580621E-01;
    COFD[9571] = 1.83363274E-02;
    COFD[9572] = -1.92404583E+01;
    COFD[9573] = 4.73921581E+00;
    COFD[9574] = -3.98017274E-01;
    COFD[9575] = 1.71639614E-02;
    COFD[9576] = -1.56919143E+01;
    COFD[9577] = 3.77842689E+00;
    COFD[9578] = -2.78523399E-01;
    COFD[9579] = 1.21896111E-02;
    COFD[9580] = -1.97252269E+01;
    COFD[9581] = 5.38884098E+00;
    COFD[9582] = -4.71627912E-01;
    COFD[9583] = 1.99273178E-02;
    COFD[9584] = -1.92651204E+01;
    COFD[9585] = 4.73921581E+00;
    COFD[9586] = -3.98017274E-01;
    COFD[9587] = 1.71639614E-02;
    COFD[9588] = -1.98682752E+01;
    COFD[9589] = 4.28648872E+00;
    COFD[9590] = -2.64358750E-01;
    COFD[9591] = 8.40263071E-03;
    COFD[9592] = -2.07257272E+01;
    COFD[9593] = 5.10688723E+00;
    COFD[9594] = -4.41563971E-01;
    COFD[9595] = 1.88857198E-02;
    COFD[9596] = -2.07362753E+01;
    COFD[9597] = 5.10688723E+00;
    COFD[9598] = -4.41563971E-01;
    COFD[9599] = 1.88857198E-02;
    COFD[9600] = -2.07464056E+01;
    COFD[9601] = 5.10688723E+00;
    COFD[9602] = -4.41563971E-01;
    COFD[9603] = 1.88857198E-02;
    COFD[9604] = -1.91552109E+01;
    COFD[9605] = 4.73921581E+00;
    COFD[9606] = -3.98017274E-01;
    COFD[9607] = 1.71639614E-02;
    COFD[9608] = -2.09802383E+01;
    COFD[9609] = 4.64167142E+00;
    COFD[9610] = -3.19532110E-01;
    COFD[9611] = 1.11478359E-02;
    COFD[9612] = -2.13616804E+01;
    COFD[9613] = 5.38519776E+00;
    COFD[9614] = -4.71344997E-01;
    COFD[9615] = 1.99226932E-02;
    COFD[9616] = -2.26897188E+01;
    COFD[9617] = 5.58518389E+00;
    COFD[9618] = -4.80570209E-01;
    COFD[9619] = 1.96586179E-02;
    COFD[9620] = -2.03988322E+01;
    COFD[9621] = 4.99562188E+00;
    COFD[9622] = -4.28482025E-01;
    COFD[9623] = 1.83720948E-02;
    COFD[9624] = -2.09922023E+01;
    COFD[9625] = 4.64167142E+00;
    COFD[9626] = -3.19532110E-01;
    COFD[9627] = 1.11478359E-02;
    COFD[9628] = -2.13616804E+01;
    COFD[9629] = 5.38519776E+00;
    COFD[9630] = -4.71344997E-01;
    COFD[9631] = 1.99226932E-02;
    COFD[9632] = -2.13903532E+01;
    COFD[9633] = 5.38519776E+00;
    COFD[9634] = -4.71344997E-01;
    COFD[9635] = 1.99226932E-02;
    COFD[9636] = -2.17771707E+01;
    COFD[9637] = 5.03453866E+00;
    COFD[9638] = -3.81762947E-01;
    COFD[9639] = 1.42886762E-02;
    COFD[9640] = -2.13695648E+01;
    COFD[9641] = 5.37614538E+00;
    COFD[9642] = -4.70679659E-01;
    COFD[9643] = 1.99143937E-02;
    COFD[9644] = -2.11660262E+01;
    COFD[9645] = 4.71644372E+00;
    COFD[9646] = -3.31349990E-01;
    COFD[9647] = 1.17430818E-02;
    COFD[9648] = -2.27001899E+01;
    COFD[9649] = 5.58468914E+00;
    COFD[9650] = -4.79958407E-01;
    COFD[9651] = 1.96104043E-02;
    COFD[9652] = -2.26882488E+01;
    COFD[9653] = 5.58468914E+00;
    COFD[9654] = -4.79958407E-01;
    COFD[9655] = 1.96104043E-02;
    COFD[9656] = -2.17463767E+01;
    COFD[9657] = 4.93496210E+00;
    COFD[9658] = -3.65981745E-01;
    COFD[9659] = 1.34912948E-02;
    COFD[9660] = -2.17740719E+01;
    COFD[9661] = 4.93496210E+00;
    COFD[9662] = -3.65981745E-01;
    COFD[9663] = 1.34912948E-02;
    COFD[9664] = -2.12888403E+01;
    COFD[9665] = 4.71644372E+00;
    COFD[9666] = -3.31349990E-01;
    COFD[9667] = 1.17430818E-02;
    COFD[9668] = -2.12951225E+01;
    COFD[9669] = 4.71644372E+00;
    COFD[9670] = -3.31349990E-01;
    COFD[9671] = 1.17430818E-02;
    COFD[9672] = -2.24990717E+01;
    COFD[9673] = 5.55026833E+00;
    COFD[9674] = -4.72437808E-01;
    COFD[9675] = 1.91625195E-02;
    COFD[9676] = -2.15322533E+01;
    COFD[9677] = 5.40448560E+00;
    COFD[9678] = -4.72711417E-01;
    COFD[9679] = 1.99362480E-02;
    COFD[9680] = -2.25127940E+01;
    COFD[9681] = 5.55026833E+00;
    COFD[9682] = -4.72437808E-01;
    COFD[9683] = 1.91625195E-02;
    COFD[9684] = -2.17537109E+01;
    COFD[9685] = 4.93496210E+00;
    COFD[9686] = -3.65981745E-01;
    COFD[9687] = 1.34912948E-02;
    COFD[9688] = -2.26485311E+01;
    COFD[9689] = 5.44696782E+00;
    COFD[9690] = -4.51976837E-01;
    COFD[9691] = 1.79942461E-02;
    COFD[9692] = -2.24603310E+01;
    COFD[9693] = 5.58501539E+00;
    COFD[9694] = -4.81433860E-01;
    COFD[9695] = 1.97311245E-02;
    COFD[9696] = -2.15342960E+01;
    COFD[9697] = 4.76557679E+00;
    COFD[9698] = -3.39171992E-01;
    COFD[9699] = 1.21386188E-02;
    COFD[9700] = -2.17537109E+01;
    COFD[9701] = 4.93496210E+00;
    COFD[9702] = -3.65981745E-01;
    COFD[9703] = 1.34912948E-02;
    COFD[9704] = -2.16206538E+01;
    COFD[9705] = 4.76557679E+00;
    COFD[9706] = -3.39171992E-01;
    COFD[9707] = 1.21386188E-02;
    COFD[9708] = -2.26361390E+01;
    COFD[9709] = 5.61137362E+00;
    COFD[9710] = -4.90253090E-01;
    COFD[9711] = 2.03300559E-02;
    COFD[9712] = -2.25891024E+01;
    COFD[9713] = 5.39655717E+00;
    COFD[9714] = -4.42728390E-01;
    COFD[9715] = 1.74857336E-02;
    COFD[9716] = -2.25972054E+01;
    COFD[9717] = 5.39655717E+00;
    COFD[9718] = -4.42728390E-01;
    COFD[9719] = 1.74857336E-02;
    COFD[9720] = -2.26203761E+01;
    COFD[9721] = 5.42039607E+00;
    COFD[9722] = -4.47178505E-01;
    COFD[9723] = 1.77324253E-02;
    COFD[9724] = -2.28101231E+01;
    COFD[9725] = 5.46112592E+00;
    COFD[9726] = -4.54556926E-01;
    COFD[9727] = 1.81357650E-02;
    COFD[9728] = -2.16983332E+01;
    COFD[9729] = 4.71782117E+00;
    COFD[9730] = -3.31568259E-01;
    COFD[9731] = 1.17540937E-02;
    COFD[9732] = -2.25720229E+01;
    COFD[9733] = 5.27220175E+00;
    COFD[9734] = -4.21436175E-01;
    COFD[9735] = 1.63576263E-02;
    COFD[9736] = -2.25735380E+01;
    COFD[9737] = 5.27888680E+00;
    COFD[9738] = -4.22608469E-01;
    COFD[9739] = 1.64205533E-02;
    COFD[9740] = -2.25785614E+01;
    COFD[9741] = 5.27888680E+00;
    COFD[9742] = -4.22608469E-01;
    COFD[9743] = 1.64205533E-02;
    COFD[9744] = -2.27197587E+01;
    COFD[9745] = 5.29216864E+00;
    COFD[9746] = -4.24828662E-01;
    COFD[9747] = 1.65366835E-02;
    COFD[9748] = -2.16466727E+01;
    COFD[9749] = 4.65048212E+00;
    COFD[9750] = -3.20931552E-01;
    COFD[9751] = 1.12185393E-02;
    COFD[9752] = -2.25387409E+01;
    COFD[9753] = 5.12713969E+00;
    COFD[9754] = -3.96797470E-01;
    COFD[9755] = 1.50599903E-02;
    COFD[9756] = -2.26201345E+01;
    COFD[9757] = 5.16909062E+00;
    COFD[9758] = -4.03789335E-01;
    COFD[9759] = 1.54242019E-02;
    COFD[9760] = -2.20597984E+01;
    COFD[9761] = 4.90985174E+00;
    COFD[9762] = -3.62022555E-01;
    COFD[9763] = 1.32919255E-02;
    COFD[9764] = -2.17488664E+01;
    COFD[9765] = 4.69776924E+00;
    COFD[9766] = -3.28393518E-01;
    COFD[9767] = 1.15940104E-02;
    COFD[9768] = -2.21696464E+01;
    COFD[9769] = 4.81459861E+00;
    COFD[9770] = -3.46990321E-01;
    COFD[9771] = 1.25347154E-02;
    COFD[9772] = -2.21717162E+01;
    COFD[9773] = 4.81459861E+00;
    COFD[9774] = -3.46990321E-01;
    COFD[9775] = 1.25347154E-02;
    COFD[9776] = -2.02840235E+01;
    COFD[9777] = 4.95484018E+00;
    COFD[9778] = -4.23654881E-01;
    COFD[9779] = 1.81813866E-02;
    COFD[9780] = -1.91633071E+01;
    COFD[9781] = 4.70966098E+00;
    COFD[9782] = -3.94551217E-01;
    COFD[9783] = 1.70286289E-02;
    COFD[9784] = -1.55781966E+01;
    COFD[9785] = 3.73153794E+00;
    COFD[9786] = -2.72372598E-01;
    COFD[9787] = 1.19199668E-02;
    COFD[9788] = -1.96804200E+01;
    COFD[9789] = 5.37526595E+00;
    COFD[9790] = -4.70621144E-01;
    COFD[9791] = 1.99141073E-02;
    COFD[9792] = -1.91880377E+01;
    COFD[9793] = 4.70966098E+00;
    COFD[9794] = -3.94551217E-01;
    COFD[9795] = 1.70286289E-02;
    COFD[9796] = -1.98762802E+01;
    COFD[9797] = 4.29984430E+00;
    COFD[9798] = -2.67672378E-01;
    COFD[9799] = 8.61066133E-03;
    COFD[9800] = -2.06463142E+01;
    COFD[9801] = 5.07657482E+00;
    COFD[9802] = -4.38028804E-01;
    COFD[9803] = 1.87481371E-02;
    COFD[9804] = -2.06522508E+01;
    COFD[9805] = 5.07501764E+00;
    COFD[9806] = -4.37846596E-01;
    COFD[9807] = 1.87410133E-02;
    COFD[9808] = -2.06624288E+01;
    COFD[9809] = 5.07501764E+00;
    COFD[9810] = -4.37846596E-01;
    COFD[9811] = 1.87410133E-02;
    COFD[9812] = -1.90778475E+01;
    COFD[9813] = 4.70966098E+00;
    COFD[9814] = -3.94551217E-01;
    COFD[9815] = 1.70286289E-02;
    COFD[9816] = -2.11071988E+01;
    COFD[9817] = 4.70311989E+00;
    COFD[9818] = -3.29240106E-01;
    COFD[9819] = 1.16366808E-02;
    COFD[9820] = -2.13171682E+01;
    COFD[9821] = 5.37197338E+00;
    COFD[9822] = -4.70392872E-01;
    COFD[9823] = 1.99122802E-02;
    COFD[9824] = -2.26749993E+01;
    COFD[9825] = 5.58486459E+00;
    COFD[9826] = -4.81517134E-01;
    COFD[9827] = 1.97388064E-02;
    COFD[9828] = -2.03122895E+01;
    COFD[9829] = 4.96244824E+00;
    COFD[9830] = -4.24554494E-01;
    COFD[9831] = 1.82168885E-02;
    COFD[9832] = -2.11192145E+01;
    COFD[9833] = 4.70311989E+00;
    COFD[9834] = -3.29240106E-01;
    COFD[9835] = 1.16366808E-02;
    COFD[9836] = -2.13171682E+01;
    COFD[9837] = 5.37197338E+00;
    COFD[9838] = -4.70392872E-01;
    COFD[9839] = 1.99122802E-02;
    COFD[9840] = -2.13459128E+01;
    COFD[9841] = 5.37197338E+00;
    COFD[9842] = -4.70392872E-01;
    COFD[9843] = 1.99122802E-02;
    COFD[9844] = -2.17567719E+01;
    COFD[9845] = 5.03450665E+00;
    COFD[9846] = -3.83012475E-01;
    COFD[9847] = 1.43925680E-02;
    COFD[9848] = -2.13282915E+01;
    COFD[9849] = 5.36375915E+00;
    COFD[9850] = -4.69808195E-01;
    COFD[9851] = 1.99064589E-02;
    COFD[9852] = -2.12804720E+01;
    COFD[9853] = 4.77238689E+00;
    COFD[9854] = -3.40265855E-01;
    COFD[9855] = 1.21942137E-02;
    COFD[9856] = -2.26853912E+01;
    COFD[9857] = 5.58521030E+00;
    COFD[9858] = -4.81061650E-01;
    COFD[9859] = 1.96992215E-02;
    COFD[9860] = -2.26733985E+01;
    COFD[9861] = 5.58521030E+00;
    COFD[9862] = -4.81061650E-01;
    COFD[9863] = 1.96992215E-02;
    COFD[9864] = -2.18797352E+01;
    COFD[9865] = 4.99907484E+00;
    COFD[9866] = -3.76094627E-01;
    COFD[9867] = 1.40009262E-02;
    COFD[9868] = -2.19075860E+01;
    COFD[9869] = 4.99907484E+00;
    COFD[9870] = -3.76094627E-01;
    COFD[9871] = 1.40009262E-02;
    COFD[9872] = -2.14039230E+01;
    COFD[9873] = 4.77238689E+00;
    COFD[9874] = -3.40265855E-01;
    COFD[9875] = 1.21942137E-02;
    COFD[9876] = -2.14102422E+01;
    COFD[9877] = 4.77238689E+00;
    COFD[9878] = -3.40265855E-01;
    COFD[9879] = 1.21942137E-02;
    COFD[9880] = -2.25347527E+01;
    COFD[9881] = 5.57238332E+00;
    COFD[9882] = -4.76605097E-01;
    COFD[9883] = 1.93951822E-02;
    COFD[9884] = -2.14935984E+01;
    COFD[9885] = 5.39257286E+00;
    COFD[9886] = -4.71929831E-01;
    COFD[9887] = 1.99331101E-02;
    COFD[9888] = -2.25485299E+01;
    COFD[9889] = 5.57238332E+00;
    COFD[9890] = -4.76605097E-01;
    COFD[9891] = 1.93951822E-02;
    COFD[9892] = -2.18871097E+01;
    COFD[9893] = 4.99907484E+00;
    COFD[9894] = -3.76094627E-01;
    COFD[9895] = 1.40009262E-02;
    COFD[9896] = -2.26946865E+01;
    COFD[9897] = 5.47392239E+00;
    COFD[9898] = -4.56882004E-01;
    COFD[9899] = 1.82631638E-02;
    COFD[9900] = -2.24423544E+01;
    COFD[9901] = 5.58416166E+00;
    COFD[9902] = -4.82369720E-01;
    COFD[9903] = 1.98133127E-02;
    COFD[9904] = -2.16515578E+01;
    COFD[9905] = 4.82270577E+00;
    COFD[9906] = -3.48263719E-01;
    COFD[9907] = 1.25986681E-02;
    COFD[9908] = -2.18871097E+01;
    COFD[9909] = 4.99907484E+00;
    COFD[9910] = -3.76094627E-01;
    COFD[9911] = 1.40009262E-02;
    COFD[9912] = -2.17384535E+01;
    COFD[9913] = 4.82270577E+00;
    COFD[9914] = -3.48263719E-01;
    COFD[9915] = 1.25986681E-02;
    COFD[9916] = -2.26249652E+01;
    COFD[9917] = 5.61234946E+00;
    COFD[9918] = -4.91326412E-01;
    COFD[9919] = 2.04139363E-02;
    COFD[9920] = -2.26273108E+01;
    COFD[9921] = 5.42002683E+00;
    COFD[9922] = -4.47111163E-01;
    COFD[9923] = 1.77287360E-02;
    COFD[9924] = -2.26354564E+01;
    COFD[9925] = 5.42002683E+00;
    COFD[9926] = -4.47111163E-01;
    COFD[9927] = 1.77287360E-02;
    COFD[9928] = -2.26677753E+01;
    COFD[9929] = 5.44777353E+00;
    COFD[9930] = -4.52122340E-01;
    COFD[9931] = 1.80021910E-02;
    COFD[9932] = -2.28554026E+01;
    COFD[9933] = 5.48796011E+00;
    COFD[9934] = -4.59457942E-01;
    COFD[9935] = 1.84050728E-02;
    COFD[9936] = -2.17385144E+01;
    COFD[9937] = 4.74350080E+00;
    COFD[9938] = -3.36426340E-01;
    COFD[9939] = 1.20245796E-02;
    COFD[9940] = -2.26508835E+01;
    COFD[9941] = 5.31312101E+00;
    COFD[9942] = -4.28304541E-01;
    COFD[9943] = 1.67176023E-02;
    COFD[9944] = -2.26553469E+01;
    COFD[9945] = 5.32093606E+00;
    COFD[9946] = -4.29624801E-01;
    COFD[9947] = 1.67869730E-02;
    COFD[9948] = -2.26604027E+01;
    COFD[9949] = 5.32093606E+00;
    COFD[9950] = -4.29624801E-01;
    COFD[9951] = 1.67869730E-02;
    COFD[9952] = -2.27940520E+01;
    COFD[9953] = 5.33097599E+00;
    COFD[9954] = -4.31367349E-01;
    COFD[9955] = 1.68798869E-02;
    COFD[9956] = -2.17749249E+01;
    COFD[9957] = 4.71207875E+00;
    COFD[9958] = -3.30658500E-01;
    COFD[9959] = 1.17082011E-02;
    COFD[9960] = -2.26201345E+01;
    COFD[9961] = 5.16909062E+00;
    COFD[9962] = -4.03789335E-01;
    COFD[9963] = 1.54242019E-02;
    COFD[9964] = -2.26855826E+01;
    COFD[9965] = 5.20461425E+00;
    COFD[9966] = -4.09923838E-01;
    COFD[9967] = 1.57504726E-02;
    COFD[9968] = -2.21931069E+01;
    COFD[9969] = 4.97373439E+00;
    COFD[9970] = -3.72089281E-01;
    COFD[9971] = 1.37987774E-02;
    COFD[9972] = -2.18703454E+01;
    COFD[9973] = 4.75653375E+00;
    COFD[9974] = -3.37718114E-01;
    COFD[9975] = 1.20647238E-02;
    COFD[9976] = -2.22981928E+01;
    COFD[9977] = 4.87626494E+00;
    COFD[9978] = -3.56718447E-01;
    COFD[9979] = 1.30246317E-02;
    COFD[9980] = -2.23002803E+01;
    COFD[9981] = 4.87626494E+00;
    COFD[9982] = -3.56718447E-01;
    COFD[9983] = 1.30246317E-02;
    COFD[9984] = -2.06548278E+01;
    COFD[9985] = 5.11678107E+00;
    COFD[9986] = -4.42706538E-01;
    COFD[9987] = 1.89296424E-02;
    COFD[9988] = -1.94778445E+01;
    COFD[9989] = 4.85518471E+00;
    COFD[9990] = -4.11551624E-01;
    COFD[9991] = 1.76895651E-02;
    COFD[9992] = -1.60461372E+01;
    COFD[9993] = 3.95298868E+00;
    COFD[9994] = -3.01302078E-01;
    COFD[9995] = 1.31842095E-02;
    COFD[9996] = -1.98424714E+01;
    COFD[9997] = 5.45215174E+00;
    COFD[9998] = -4.77051991E-01;
    COFD[9999] = 2.00510347E-02;
    COFD[10000] = -1.95026421E+01;
    COFD[10001] = 4.85518471E+00;
    COFD[10002] = -4.11551624E-01;
    COFD[10003] = 1.76895651E-02;
    COFD[10004] = -1.90375666E+01;
    COFD[10005] = 3.93604965E+00;
    COFD[10006] = -2.11360409E-01;
    COFD[10007] = 5.81247394E-03;
    COFD[10008] = -2.09258526E+01;
    COFD[10009] = 5.19811866E+00;
    COFD[10010] = -4.51121211E-01;
    COFD[10011] = 1.92074617E-02;
    COFD[10012] = -2.09364971E+01;
    COFD[10013] = 5.19811866E+00;
    COFD[10014] = -4.51121211E-01;
    COFD[10015] = 1.92074617E-02;
    COFD[10016] = -2.09467220E+01;
    COFD[10017] = 5.19811866E+00;
    COFD[10018] = -4.51121211E-01;
    COFD[10019] = 1.92074617E-02;
    COFD[10020] = -1.93921777E+01;
    COFD[10021] = 4.85518471E+00;
    COFD[10022] = -4.11551624E-01;
    COFD[10023] = 1.76895651E-02;
    COFD[10024] = -2.03849874E+01;
    COFD[10025] = 4.38396848E+00;
    COFD[10026] = -2.79298901E-01;
    COFD[10027] = 9.13915001E-03;
    COFD[10028] = -2.14920449E+01;
    COFD[10029] = 5.44385051E+00;
    COFD[10030] = -4.76121506E-01;
    COFD[10031] = 2.00164081E-02;
    COFD[10032] = -2.25786655E+01;
    COFD[10033] = 5.53409384E+00;
    COFD[10034] = -4.69342499E-01;
    COFD[10035] = 1.89886374E-02;
    COFD[10036] = -2.06812067E+01;
    COFD[10037] = 5.12346096E+00;
    COFD[10038] = -4.43477411E-01;
    COFD[10039] = 1.89592529E-02;
    COFD[10040] = -2.03970537E+01;
    COFD[10041] = 4.38396848E+00;
    COFD[10042] = -2.79298901E-01;
    COFD[10043] = 9.13915001E-03;
    COFD[10044] = -2.14920449E+01;
    COFD[10045] = 5.44385051E+00;
    COFD[10046] = -4.76121506E-01;
    COFD[10047] = 2.00164081E-02;
    COFD[10048] = -2.15208595E+01;
    COFD[10049] = 5.44385051E+00;
    COFD[10050] = -4.76121506E-01;
    COFD[10051] = 2.00164081E-02;
    COFD[10052] = -2.12302151E+01;
    COFD[10053] = 4.79651003E+00;
    COFD[10054] = -3.44144386E-01;
    COFD[10055] = 1.23916372E-02;
    COFD[10056] = -2.14671205E+01;
    COFD[10057] = 5.42109069E+00;
    COFD[10058] = -4.73533096E-01;
    COFD[10059] = 1.99183547E-02;
    COFD[10060] = -2.06103015E+01;
    COFD[10061] = 4.47491202E+00;
    COFD[10062] = -2.93331059E-01;
    COFD[10063] = 9.83445305E-03;
    COFD[10064] = -2.25695574E+01;
    COFD[10065] = 5.52323975E+00;
    COFD[10066] = -4.67257607E-01;
    COFD[10067] = 1.88711975E-02;
    COFD[10068] = -2.25575141E+01;
    COFD[10069] = 5.52323975E+00;
    COFD[10070] = -4.67257607E-01;
    COFD[10071] = 1.88711975E-02;
    COFD[10072] = -2.12221678E+01;
    COFD[10073] = 4.70506024E+00;
    COFD[10074] = -3.29547212E-01;
    COFD[10075] = 1.16521630E-02;
    COFD[10076] = -2.12501716E+01;
    COFD[10077] = 4.70506024E+00;
    COFD[10078] = -3.29547212E-01;
    COFD[10079] = 1.16521630E-02;
    COFD[10080] = -2.07343778E+01;
    COFD[10081] = 4.47491202E+00;
    COFD[10082] = -2.93331059E-01;
    COFD[10083] = 9.83445305E-03;
    COFD[10084] = -2.07407334E+01;
    COFD[10085] = 4.47491202E+00;
    COFD[10086] = -2.93331059E-01;
    COFD[10087] = 9.83445305E-03;
    COFD[10088] = -2.23655523E+01;
    COFD[10089] = 5.48956505E+00;
    COFD[10090] = -4.59770566E-01;
    COFD[10091] = 1.84227929E-02;
    COFD[10092] = -2.17210124E+01;
    COFD[10093] = 5.49225467E+00;
    COFD[10094] = -4.81478120E-01;
    COFD[10095] = 2.02123784E-02;
    COFD[10096] = -2.23793834E+01;
    COFD[10097] = 5.48956505E+00;
    COFD[10098] = -4.59770566E-01;
    COFD[10099] = 1.84227929E-02;
    COFD[10100] = -2.12295821E+01;
    COFD[10101] = 4.70506024E+00;
    COFD[10102] = -3.29547212E-01;
    COFD[10103] = 1.16521630E-02;
    COFD[10104] = -2.23996701E+01;
    COFD[10105] = 5.33372666E+00;
    COFD[10106] = -4.31837946E-01;
    COFD[10107] = 1.69048117E-02;
    COFD[10108] = -2.23867276E+01;
    COFD[10109] = 5.55175851E+00;
    COFD[10110] = -4.72720598E-01;
    COFD[10111] = 1.91783487E-02;
    COFD[10112] = -2.10131896E+01;
    COFD[10113] = 4.53499682E+00;
    COFD[10114] = -3.02678130E-01;
    COFD[10115] = 1.03000978E-02;
    COFD[10116] = -2.12295821E+01;
    COFD[10117] = 4.70506024E+00;
    COFD[10118] = -3.29547212E-01;
    COFD[10119] = 1.16521630E-02;
    COFD[10120] = -2.11006145E+01;
    COFD[10121] = 4.53499682E+00;
    COFD[10122] = -3.02678130E-01;
    COFD[10123] = 1.03000978E-02;
    COFD[10124] = -2.25837110E+01;
    COFD[10125] = 5.58420073E+00;
    COFD[10126] = -4.82356716E-01;
    COFD[10127] = 1.98120306E-02;
    COFD[10128] = -2.22858832E+01;
    COFD[10129] = 5.25941804E+00;
    COFD[10130] = -4.19208672E-01;
    COFD[10131] = 1.62385114E-02;
    COFD[10132] = -2.22940707E+01;
    COFD[10133] = 5.25941804E+00;
    COFD[10134] = -4.19208672E-01;
    COFD[10135] = 1.62385114E-02;
    COFD[10136] = -2.23480908E+01;
    COFD[10137] = 5.29695321E+00;
    COFD[10138] = -4.25620113E-01;
    COFD[10139] = 1.65778213E-02;
    COFD[10140] = -2.25780442E+01;
    COFD[10141] = 5.35238497E+00;
    COFD[10142] = -4.35034945E-01;
    COFD[10143] = 1.70742216E-02;
    COFD[10144] = -2.11585495E+01;
    COFD[10145] = 4.47646812E+00;
    COFD[10146] = -2.93573165E-01;
    COFD[10147] = 9.84650920E-03;
    COFD[10148] = -2.22582201E+01;
    COFD[10149] = 5.12825866E+00;
    COFD[10150] = -3.96982702E-01;
    COFD[10151] = 1.50696010E-02;
    COFD[10152] = -2.22649250E+01;
    COFD[10153] = 5.13737172E+00;
    COFD[10154] = -3.98493102E-01;
    COFD[10155] = 1.51480250E-02;
    COFD[10156] = -2.22700127E+01;
    COFD[10157] = 5.13737172E+00;
    COFD[10158] = -3.98493102E-01;
    COFD[10159] = 1.51480250E-02;
    COFD[10160] = -2.24171125E+01;
    COFD[10161] = 5.15109664E+00;
    COFD[10162] = -4.00765892E-01;
    COFD[10163] = 1.52659560E-02;
    COFD[10164] = -2.10757112E+01;
    COFD[10165] = 4.39521460E+00;
    COFD[10166] = -2.81028854E-01;
    COFD[10167] = 9.22466916E-03;
    COFD[10168] = -2.20597984E+01;
    COFD[10169] = 4.90985174E+00;
    COFD[10170] = -3.62022555E-01;
    COFD[10171] = 1.32919255E-02;
    COFD[10172] = -2.21931069E+01;
    COFD[10173] = 4.97373439E+00;
    COFD[10174] = -3.72089281E-01;
    COFD[10175] = 1.37987774E-02;
    COFD[10176] = -2.15488348E+01;
    COFD[10177] = 4.68075977E+00;
    COFD[10178] = -3.25704931E-01;
    COFD[10179] = 1.14585845E-02;
    COFD[10180] = -2.12063903E+01;
    COFD[10181] = 4.45414225E+00;
    COFD[10182] = -2.90099937E-01;
    COFD[10183] = 9.67360320E-03;
    COFD[10184] = -2.16820603E+01;
    COFD[10185] = 4.59101412E+00;
    COFD[10186] = -3.11439033E-01;
    COFD[10187] = 1.07377082E-02;
    COFD[10188] = -2.16841653E+01;
    COFD[10189] = 4.59101412E+00;
    COFD[10190] = -3.11439033E-01;
    COFD[10191] = 1.07377082E-02;
    COFD[10192] = -2.10086887E+01;
    COFD[10193] = 5.19953529E+00;
    COFD[10194] = -4.51287802E-01;
    COFD[10195] = 1.92140123E-02;
    COFD[10196] = -1.99562868E+01;
    COFD[10197] = 4.99367362E+00;
    COFD[10198] = -4.28249956E-01;
    COFD[10199] = 1.83628509E-02;
    COFD[10200] = -1.64639359E+01;
    COFD[10201] = 4.08142484E+00;
    COFD[10202] = -3.17696496E-01;
    COFD[10203] = 1.38856294E-02;
    COFD[10204] = -2.02451923E+01;
    COFD[10205] = 5.55377454E+00;
    COFD[10206] = -4.87810074E-01;
    COFD[10207] = 2.04217376E-02;
    COFD[10208] = -1.99818280E+01;
    COFD[10209] = 4.99367362E+00;
    COFD[10210] = -4.28249956E-01;
    COFD[10211] = 1.83628509E-02;
    COFD[10212] = -1.85767826E+01;
    COFD[10213] = 3.66420353E+00;
    COFD[10214] = -1.69810177E-01;
    COFD[10215] = 3.77247849E-03;
    COFD[10216] = -2.14057339E+01;
    COFD[10217] = 5.33269880E+00;
    COFD[10218] = -4.67008439E-01;
    COFD[10219] = 1.98347416E-02;
    COFD[10220] = -2.14169211E+01;
    COFD[10221] = 5.33269880E+00;
    COFD[10222] = -4.67008439E-01;
    COFD[10223] = 1.98347416E-02;
    COFD[10224] = -2.14276788E+01;
    COFD[10225] = 5.33269880E+00;
    COFD[10226] = -4.67008439E-01;
    COFD[10227] = 1.98347416E-02;
    COFD[10228] = -1.98683238E+01;
    COFD[10229] = 4.99367362E+00;
    COFD[10230] = -4.28249956E-01;
    COFD[10231] = 1.83628509E-02;
    COFD[10232] = -1.98477259E+01;
    COFD[10233] = 4.07958166E+00;
    COFD[10234] = -2.33006871E-01;
    COFD[10235] = 6.86822015E-03;
    COFD[10236] = -2.18960800E+01;
    COFD[10237] = 5.54768472E+00;
    COFD[10238] = -4.87202065E-01;
    COFD[10239] = 2.04025437E-02;
    COFD[10240] = -2.26305728E+01;
    COFD[10241] = 5.47666967E+00;
    COFD[10242] = -4.57381900E-01;
    COFD[10243] = 1.82905822E-02;
    COFD[10244] = -2.10372026E+01;
    COFD[10245] = 5.20711052E+00;
    COFD[10246] = -4.52173945E-01;
    COFD[10247] = 1.92486273E-02;
    COFD[10248] = -1.98603655E+01;
    COFD[10249] = 4.07958166E+00;
    COFD[10250] = -2.33006871E-01;
    COFD[10251] = 6.86822015E-03;
    COFD[10252] = -2.18960800E+01;
    COFD[10253] = 5.54768472E+00;
    COFD[10254] = -4.87202065E-01;
    COFD[10255] = 2.04025437E-02;
    COFD[10256] = -2.19256706E+01;
    COFD[10257] = 5.54768472E+00;
    COFD[10258] = -4.87202065E-01;
    COFD[10259] = 2.04025437E-02;
    COFD[10260] = -2.09298486E+01;
    COFD[10261] = 4.59063108E+00;
    COFD[10262] = -3.11377715E-01;
    COFD[10263] = 1.07346023E-02;
    COFD[10264] = -2.18876256E+01;
    COFD[10265] = 5.53154746E+00;
    COFD[10266] = -4.85594344E-01;
    COFD[10267] = 2.03520324E-02;
    COFD[10268] = -2.01250987E+01;
    COFD[10269] = 4.19160608E+00;
    COFD[10270] = -2.49936771E-01;
    COFD[10271] = 7.69538319E-03;
    COFD[10272] = -2.26027431E+01;
    COFD[10273] = 5.46217527E+00;
    COFD[10274] = -4.54751471E-01;
    COFD[10275] = 1.81465218E-02;
    COFD[10276] = -2.25901270E+01;
    COFD[10277] = 5.46217527E+00;
    COFD[10278] = -4.54751471E-01;
    COFD[10279] = 1.81465218E-02;
    COFD[10280] = -2.08821587E+01;
    COFD[10281] = 4.48108132E+00;
    COFD[10282] = -2.94289899E-01;
    COFD[10283] = 9.88218297E-03;
    COFD[10284] = -2.09119213E+01;
    COFD[10285] = 4.48108132E+00;
    COFD[10286] = -2.94289899E-01;
    COFD[10287] = 9.88218297E-03;
    COFD[10288] = -2.02563322E+01;
    COFD[10289] = 4.19160608E+00;
    COFD[10290] = -2.49936771E-01;
    COFD[10291] = 7.69538319E-03;
    COFD[10292] = -2.02631076E+01;
    COFD[10293] = 4.19160608E+00;
    COFD[10294] = -2.49936771E-01;
    COFD[10295] = 7.69538319E-03;
    COFD[10296] = -2.23265991E+01;
    COFD[10297] = 5.39645154E+00;
    COFD[10298] = -4.42708323E-01;
    COFD[10299] = 1.74846134E-02;
    COFD[10300] = -2.21024680E+01;
    COFD[10301] = 5.57482264E+00;
    COFD[10302] = -4.89554775E-01;
    COFD[10303] = 2.04583790E-02;
    COFD[10304] = -2.23410369E+01;
    COFD[10305] = 5.39645154E+00;
    COFD[10306] = -4.42708323E-01;
    COFD[10307] = 1.74846134E-02;
    COFD[10308] = -2.08900285E+01;
    COFD[10309] = 4.48108132E+00;
    COFD[10310] = -2.94289899E-01;
    COFD[10311] = 9.88218297E-03;
    COFD[10312] = -2.22885235E+01;
    COFD[10313] = 5.20764658E+00;
    COFD[10314] = -4.10207913E-01;
    COFD[10315] = 1.57585882E-02;
    COFD[10316] = -2.24356779E+01;
    COFD[10317] = 5.49613266E+00;
    COFD[10318] = -4.61060586E-01;
    COFD[10319] = 1.84960110E-02;
    COFD[10320] = -2.05673423E+01;
    COFD[10321] = 4.26766320E+00;
    COFD[10322] = -2.61480535E-01;
    COFD[10323] = 8.26106960E-03;
    COFD[10324] = -2.08900285E+01;
    COFD[10325] = 4.48108132E+00;
    COFD[10326] = -2.94289899E-01;
    COFD[10327] = 9.88218297E-03;
    COFD[10328] = -2.06609009E+01;
    COFD[10329] = 4.26766320E+00;
    COFD[10330] = -2.61480535E-01;
    COFD[10331] = 8.26106960E-03;
    COFD[10332] = -2.27479974E+01;
    COFD[10333] = 5.57822325E+00;
    COFD[10334] = -4.77777262E-01;
    COFD[10335] = 1.94626011E-02;
    COFD[10336] = -2.21494624E+01;
    COFD[10337] = 5.12338366E+00;
    COFD[10338] = -3.96176894E-01;
    COFD[10339] = 1.50278196E-02;
    COFD[10340] = -2.21581289E+01;
    COFD[10341] = 5.12338366E+00;
    COFD[10342] = -3.96176894E-01;
    COFD[10343] = 1.50278196E-02;
    COFD[10344] = -2.22256643E+01;
    COFD[10345] = 5.16620234E+00;
    COFD[10346] = -4.03306755E-01;
    COFD[10347] = 1.53990058E-02;
    COFD[10348] = -2.24631694E+01;
    COFD[10349] = 5.22623384E+00;
    COFD[10350] = -4.13380324E-01;
    COFD[10351] = 1.59259437E-02;
    COFD[10352] = -2.06827490E+01;
    COFD[10353] = 4.19375892E+00;
    COFD[10354] = -2.50262428E-01;
    COFD[10355] = 7.71131487E-03;
    COFD[10356] = -2.19774160E+01;
    COFD[10357] = 4.92889157E+00;
    COFD[10358] = -3.65025286E-01;
    COFD[10359] = 1.34431452E-02;
    COFD[10360] = -2.19948202E+01;
    COFD[10361] = 4.94219368E+00;
    COFD[10362] = -3.67120797E-01;
    COFD[10363] = 1.35486343E-02;
    COFD[10364] = -2.20002783E+01;
    COFD[10365] = 4.94219368E+00;
    COFD[10366] = -3.67120797E-01;
    COFD[10367] = 1.35486343E-02;
    COFD[10368] = -2.21597603E+01;
    COFD[10369] = 4.96243463E+00;
    COFD[10370] = -3.70309018E-01;
    COFD[10371] = 1.37091432E-02;
    COFD[10372] = -2.05583124E+01;
    COFD[10373] = 4.09420232E+00;
    COFD[10374] = -2.35210019E-01;
    COFD[10375] = 6.97573395E-03;
    COFD[10376] = -2.17488664E+01;
    COFD[10377] = 4.69776924E+00;
    COFD[10378] = -3.28393518E-01;
    COFD[10379] = 1.15940104E-02;
    COFD[10380] = -2.18703454E+01;
    COFD[10381] = 4.75653375E+00;
    COFD[10382] = -3.37718114E-01;
    COFD[10383] = 1.20647238E-02;
    COFD[10384] = -2.12063903E+01;
    COFD[10385] = 4.45414225E+00;
    COFD[10386] = -2.90099937E-01;
    COFD[10387] = 9.67360320E-03;
    COFD[10388] = -2.07107178E+01;
    COFD[10389] = 4.16274636E+00;
    COFD[10390] = -2.45571617E-01;
    COFD[10391] = 7.48187191E-03;
    COFD[10392] = -2.12713524E+01;
    COFD[10393] = 4.33544466E+00;
    COFD[10394] = -2.71843874E-01;
    COFD[10395] = 8.77093391E-03;
    COFD[10396] = -2.12736663E+01;
    COFD[10397] = 4.33544466E+00;
    COFD[10398] = -2.71843874E-01;
    COFD[10399] = 8.77093391E-03;
    COFD[10400] = -2.10674485E+01;
    COFD[10401] = 5.15027524E+00;
    COFD[10402] = -4.46126111E-01;
    COFD[10403] = 1.90401391E-02;
    COFD[10404] = -1.99785176E+01;
    COFD[10405] = 4.92184026E+00;
    COFD[10406] = -4.19745472E-01;
    COFD[10407] = 1.80268154E-02;
    COFD[10408] = -1.64898528E+01;
    COFD[10409] = 4.01175649E+00;
    COFD[10410] = -3.08860971E-01;
    COFD[10411] = 1.35100076E-02;
    COFD[10412] = -2.03113704E+01;
    COFD[10413] = 5.50136606E+00;
    COFD[10414] = -4.82461887E-01;
    COFD[10415] = 2.02471523E-02;
    COFD[10416] = -2.00047095E+01;
    COFD[10417] = 4.92184026E+00;
    COFD[10418] = -4.19745472E-01;
    COFD[10419] = 1.80268154E-02;
    COFD[10420] = -1.91326792E+01;
    COFD[10421] = 3.82263611E+00;
    COFD[10422] = -1.93983472E-01;
    COFD[10423] = 4.95789388E-03;
    COFD[10424] = -2.13955999E+01;
    COFD[10425] = 5.25183817E+00;
    COFD[10426] = -4.57376333E-01;
    COFD[10427] = 1.94504429E-02;
    COFD[10428] = -2.14072803E+01;
    COFD[10429] = 5.25183817E+00;
    COFD[10430] = -4.57376333E-01;
    COFD[10431] = 1.94504429E-02;
    COFD[10432] = -2.14185232E+01;
    COFD[10433] = 5.25183817E+00;
    COFD[10434] = -4.57376333E-01;
    COFD[10435] = 1.94504429E-02;
    COFD[10436] = -1.98885574E+01;
    COFD[10437] = 4.92184026E+00;
    COFD[10438] = -4.19745472E-01;
    COFD[10439] = 1.80268154E-02;
    COFD[10440] = -2.04488935E+01;
    COFD[10441] = 4.26473557E+00;
    COFD[10442] = -2.61033037E-01;
    COFD[10443] = 8.23906412E-03;
    COFD[10444] = -2.19248250E+01;
    COFD[10445] = 5.49350509E+00;
    COFD[10446] = -4.81613405E-01;
    COFD[10447] = 2.02171734E-02;
    COFD[10448] = -2.28655752E+01;
    COFD[10449] = 5.50522401E+00;
    COFD[10450] = -4.63604304E-01;
    COFD[10451] = 1.86600785E-02;
    COFD[10452] = -2.10844012E+01;
    COFD[10453] = 5.15315713E+00;
    COFD[10454] = -4.46344043E-01;
    COFD[10455] = 1.90431546E-02;
    COFD[10456] = -2.04620510E+01;
    COFD[10457] = 4.26473557E+00;
    COFD[10458] = -2.61033037E-01;
    COFD[10459] = 8.23906412E-03;
    COFD[10460] = -2.19248250E+01;
    COFD[10461] = 5.49350509E+00;
    COFD[10462] = -4.81613405E-01;
    COFD[10463] = 2.02171734E-02;
    COFD[10464] = -2.19550907E+01;
    COFD[10465] = 5.49350509E+00;
    COFD[10466] = -4.81613405E-01;
    COFD[10467] = 2.02171734E-02;
    COFD[10468] = -2.13745703E+01;
    COFD[10469] = 4.71094320E+00;
    COFD[10470] = -3.30478653E-01;
    COFD[10471] = 1.16991305E-02;
    COFD[10472] = -2.19053841E+01;
    COFD[10473] = 5.47162499E+00;
    COFD[10474] = -4.79195552E-01;
    COFD[10475] = 2.01289088E-02;
    COFD[10476] = -2.06858147E+01;
    COFD[10477] = 4.35920123E+00;
    COFD[10478] = -2.75491273E-01;
    COFD[10479] = 8.95100289E-03;
    COFD[10480] = -2.28446667E+01;
    COFD[10481] = 5.50134401E+00;
    COFD[10482] = -4.62488197E-01;
    COFD[10483] = 1.85873697E-02;
    COFD[10484] = -2.28315330E+01;
    COFD[10485] = 5.50134401E+00;
    COFD[10486] = -4.62488197E-01;
    COFD[10487] = 1.85873697E-02;
    COFD[10488] = -2.13524540E+01;
    COFD[10489] = 4.61201872E+00;
    COFD[10490] = -3.14803338E-01;
    COFD[10491] = 1.09082984E-02;
    COFD[10492] = -2.13838498E+01;
    COFD[10493] = 4.61201872E+00;
    COFD[10494] = -3.14803338E-01;
    COFD[10495] = 1.09082984E-02;
    COFD[10496] = -2.08236367E+01;
    COFD[10497] = 4.35920123E+00;
    COFD[10498] = -2.75491273E-01;
    COFD[10499] = 8.95100289E-03;
    COFD[10500] = -2.08308042E+01;
    COFD[10501] = 4.35920123E+00;
    COFD[10502] = -2.75491273E-01;
    COFD[10503] = 8.95100289E-03;
    COFD[10504] = -2.26089431E+01;
    COFD[10505] = 5.44867280E+00;
    COFD[10506] = -4.52284883E-01;
    COFD[10507] = 1.80110706E-02;
    COFD[10508] = -2.22093870E+01;
    COFD[10509] = 5.53457356E+00;
    COFD[10510] = -4.85892223E-01;
    COFD[10511] = 2.03611937E-02;
    COFD[10512] = -2.26239253E+01;
    COFD[10513] = 5.44867280E+00;
    COFD[10514] = -4.52284883E-01;
    COFD[10515] = 1.80110706E-02;
    COFD[10516] = -2.13607457E+01;
    COFD[10517] = 4.61201872E+00;
    COFD[10518] = -3.14803338E-01;
    COFD[10519] = 1.09082984E-02;
    COFD[10520] = -2.26029886E+01;
    COFD[10521] = 5.27383847E+00;
    COFD[10522] = -4.21722368E-01;
    COFD[10523] = 1.63729618E-02;
    COFD[10524] = -2.26579938E+01;
    COFD[10525] = 5.52001624E+00;
    COFD[10526] = -4.66629503E-01;
    COFD[10527] = 1.88355817E-02;
    COFD[10528] = -2.10973496E+01;
    COFD[10529] = 4.42639566E+00;
    COFD[10530] = -2.85821723E-01;
    COFD[10531] = 9.46169352E-03;
    COFD[10532] = -2.13607457E+01;
    COFD[10533] = 4.61201872E+00;
    COFD[10534] = -3.14803338E-01;
    COFD[10535] = 1.09082984E-02;
    COFD[10536] = -2.11966826E+01;
    COFD[10537] = 4.42639566E+00;
    COFD[10538] = -2.85821723E-01;
    COFD[10539] = 9.46169352E-03;
    COFD[10540] = -2.29254220E+01;
    COFD[10541] = 5.58520405E+00;
    COFD[10542] = -4.80873447E-01;
    COFD[10543] = 1.96836519E-02;
    COFD[10544] = -2.25069737E+01;
    COFD[10545] = 5.21003123E+00;
    COFD[10546] = -4.10612564E-01;
    COFD[10547] = 1.57798598E-02;
    COFD[10548] = -2.25160816E+01;
    COFD[10549] = 5.21003123E+00;
    COFD[10550] = -4.10612564E-01;
    COFD[10551] = 1.57798598E-02;
    COFD[10552] = -2.25635595E+01;
    COFD[10553] = 5.24330646E+00;
    COFD[10554] = -4.16370120E-01;
    COFD[10555] = 1.60860486E-02;
    COFD[10556] = -2.27715883E+01;
    COFD[10557] = 5.29493402E+00;
    COFD[10558] = -4.25285978E-01;
    COFD[10559] = 1.65604533E-02;
    COFD[10560] = -2.12332312E+01;
    COFD[10561] = 4.36095377E+00;
    COFD[10562] = -2.75760539E-01;
    COFD[10563] = 8.96430249E-03;
    COFD[10564] = -2.24161979E+01;
    COFD[10565] = 5.05061421E+00;
    COFD[10566] = -3.84359196E-01;
    COFD[10567] = 1.44214004E-02;
    COFD[10568] = -2.24284563E+01;
    COFD[10569] = 5.06106414E+00;
    COFD[10570] = -3.86053039E-01;
    COFD[10571] = 1.45081784E-02;
    COFD[10572] = -2.24342646E+01;
    COFD[10573] = 5.06106414E+00;
    COFD[10574] = -3.86053039E-01;
    COFD[10575] = 1.45081784E-02;
    COFD[10576] = -2.25733338E+01;
    COFD[10577] = 5.07648425E+00;
    COFD[10578] = -3.88560019E-01;
    COFD[10579] = 1.46368353E-02;
    COFD[10580] = -2.11438235E+01;
    COFD[10581] = 4.27612828E+00;
    COFD[10582] = -2.62774610E-01;
    COFD[10583] = 8.32471127E-03;
    COFD[10584] = -2.21696464E+01;
    COFD[10585] = 4.81459861E+00;
    COFD[10586] = -3.46990321E-01;
    COFD[10587] = 1.25347154E-02;
    COFD[10588] = -2.22981928E+01;
    COFD[10589] = 4.87626494E+00;
    COFD[10590] = -3.56718447E-01;
    COFD[10591] = 1.30246317E-02;
    COFD[10592] = -2.16820603E+01;
    COFD[10593] = 4.59101412E+00;
    COFD[10594] = -3.11439033E-01;
    COFD[10595] = 1.07377082E-02;
    COFD[10596] = -2.12713524E+01;
    COFD[10597] = 4.33544466E+00;
    COFD[10598] = -2.71843874E-01;
    COFD[10599] = 8.77093391E-03;
    COFD[10600] = -2.17746534E+01;
    COFD[10601] = 4.48837319E+00;
    COFD[10602] = -2.95423315E-01;
    COFD[10603] = 9.93861345E-03;
    COFD[10604] = -2.17771745E+01;
    COFD[10605] = 4.48837319E+00;
    COFD[10606] = -2.95423315E-01;
    COFD[10607] = 9.93861345E-03;
    COFD[10608] = -2.10685573E+01;
    COFD[10609] = 5.15027524E+00;
    COFD[10610] = -4.46126111E-01;
    COFD[10611] = 1.90401391E-02;
    COFD[10612] = -1.99792167E+01;
    COFD[10613] = 4.92184026E+00;
    COFD[10614] = -4.19745472E-01;
    COFD[10615] = 1.80268154E-02;
    COFD[10616] = -1.64899530E+01;
    COFD[10617] = 4.01175649E+00;
    COFD[10618] = -3.08860971E-01;
    COFD[10619] = 1.35100076E-02;
    COFD[10620] = -2.03114210E+01;
    COFD[10621] = 5.50136606E+00;
    COFD[10622] = -4.82461887E-01;
    COFD[10623] = 2.02471523E-02;
    COFD[10624] = -2.00054461E+01;
    COFD[10625] = 4.92184026E+00;
    COFD[10626] = -4.19745472E-01;
    COFD[10627] = 1.80268154E-02;
    COFD[10628] = -1.91334529E+01;
    COFD[10629] = 3.82263611E+00;
    COFD[10630] = -1.93983472E-01;
    COFD[10631] = 4.95789388E-03;
    COFD[10632] = -2.13968281E+01;
    COFD[10633] = 5.25183817E+00;
    COFD[10634] = -4.57376333E-01;
    COFD[10635] = 1.94504429E-02;
    COFD[10636] = -2.14085375E+01;
    COFD[10637] = 5.25183817E+00;
    COFD[10638] = -4.57376333E-01;
    COFD[10639] = 1.94504429E-02;
    COFD[10640] = -2.14198091E+01;
    COFD[10641] = 5.25183817E+00;
    COFD[10642] = -4.57376333E-01;
    COFD[10643] = 1.94504429E-02;
    COFD[10644] = -1.98891413E+01;
    COFD[10645] = 4.92184026E+00;
    COFD[10646] = -4.19745472E-01;
    COFD[10647] = 1.80268154E-02;
    COFD[10648] = -2.04500331E+01;
    COFD[10649] = 4.26473557E+00;
    COFD[10650] = -2.61033037E-01;
    COFD[10651] = 8.23906412E-03;
    COFD[10652] = -2.19254485E+01;
    COFD[10653] = 5.49350509E+00;
    COFD[10654] = -4.81613405E-01;
    COFD[10655] = 2.02171734E-02;
    COFD[10656] = -2.28671232E+01;
    COFD[10657] = 5.50522401E+00;
    COFD[10658] = -4.63604304E-01;
    COFD[10659] = 1.86600785E-02;
    COFD[10660] = -2.10855099E+01;
    COFD[10661] = 5.15315713E+00;
    COFD[10662] = -4.46344043E-01;
    COFD[10663] = 1.90431546E-02;
    COFD[10664] = -2.04632210E+01;
    COFD[10665] = 4.26473557E+00;
    COFD[10666] = -2.61033037E-01;
    COFD[10667] = 8.23906412E-03;
    COFD[10668] = -2.19254485E+01;
    COFD[10669] = 5.49350509E+00;
    COFD[10670] = -4.81613405E-01;
    COFD[10671] = 2.02171734E-02;
    COFD[10672] = -2.19557531E+01;
    COFD[10673] = 5.49350509E+00;
    COFD[10674] = -4.81613405E-01;
    COFD[10675] = 2.02171734E-02;
    COFD[10676] = -2.13757703E+01;
    COFD[10677] = 4.71094320E+00;
    COFD[10678] = -3.30478653E-01;
    COFD[10679] = 1.16991305E-02;
    COFD[10680] = -2.19060847E+01;
    COFD[10681] = 5.47162499E+00;
    COFD[10682] = -4.79195552E-01;
    COFD[10683] = 2.01289088E-02;
    COFD[10684] = -2.06870442E+01;
    COFD[10685] = 4.35920123E+00;
    COFD[10686] = -2.75491273E-01;
    COFD[10687] = 8.95100289E-03;
    COFD[10688] = -2.28458380E+01;
    COFD[10689] = 5.50134401E+00;
    COFD[10690] = -4.62488197E-01;
    COFD[10691] = 1.85873697E-02;
    COFD[10692] = -2.28326740E+01;
    COFD[10693] = 5.50134401E+00;
    COFD[10694] = -4.62488197E-01;
    COFD[10695] = 1.85873697E-02;
    COFD[10696] = -2.13539532E+01;
    COFD[10697] = 4.61201872E+00;
    COFD[10698] = -3.14803338E-01;
    COFD[10699] = 1.09082984E-02;
    COFD[10700] = -2.13854464E+01;
    COFD[10701] = 4.61201872E+00;
    COFD[10702] = -3.14803338E-01;
    COFD[10703] = 1.09082984E-02;
    COFD[10704] = -2.08252570E+01;
    COFD[10705] = 4.35920123E+00;
    COFD[10706] = -2.75491273E-01;
    COFD[10707] = 8.95100289E-03;
    COFD[10708] = -2.08324480E+01;
    COFD[10709] = 4.35920123E+00;
    COFD[10710] = -2.75491273E-01;
    COFD[10711] = 8.95100289E-03;
    COFD[10712] = -2.26099899E+01;
    COFD[10713] = 5.44867280E+00;
    COFD[10714] = -4.52284883E-01;
    COFD[10715] = 1.80110706E-02;
    COFD[10716] = -2.22108608E+01;
    COFD[10717] = 5.53457356E+00;
    COFD[10718] = -4.85892223E-01;
    COFD[10719] = 2.03611937E-02;
    COFD[10720] = -2.26250040E+01;
    COFD[10721] = 5.44867280E+00;
    COFD[10722] = -4.52284883E-01;
    COFD[10723] = 1.80110706E-02;
    COFD[10724] = -2.13622700E+01;
    COFD[10725] = 4.61201872E+00;
    COFD[10726] = -3.14803338E-01;
    COFD[10727] = 1.09082984E-02;
    COFD[10728] = -2.26044889E+01;
    COFD[10729] = 5.27383847E+00;
    COFD[10730] = -4.21722368E-01;
    COFD[10731] = 1.63729618E-02;
    COFD[10732] = -2.26591038E+01;
    COFD[10733] = 5.52001624E+00;
    COFD[10734] = -4.66629503E-01;
    COFD[10735] = 1.88355817E-02;
    COFD[10736] = -2.10989231E+01;
    COFD[10737] = 4.42639566E+00;
    COFD[10738] = -2.85821723E-01;
    COFD[10739] = 9.46169352E-03;
    COFD[10740] = -2.13622700E+01;
    COFD[10741] = 4.61201872E+00;
    COFD[10742] = -3.14803338E-01;
    COFD[10743] = 1.09082984E-02;
    COFD[10744] = -2.11986026E+01;
    COFD[10745] = 4.42639566E+00;
    COFD[10746] = -2.85821723E-01;
    COFD[10747] = 9.46169352E-03;
    COFD[10748] = -2.29268183E+01;
    COFD[10749] = 5.58520405E+00;
    COFD[10750] = -4.80873447E-01;
    COFD[10751] = 1.96836519E-02;
    COFD[10752] = -2.25083966E+01;
    COFD[10753] = 5.21003123E+00;
    COFD[10754] = -4.10612564E-01;
    COFD[10755] = 1.57798598E-02;
    COFD[10756] = -2.25175307E+01;
    COFD[10757] = 5.21003123E+00;
    COFD[10758] = -4.10612564E-01;
    COFD[10759] = 1.57798598E-02;
    COFD[10760] = -2.25650343E+01;
    COFD[10761] = 5.24330646E+00;
    COFD[10762] = -4.16370120E-01;
    COFD[10763] = 1.60860486E-02;
    COFD[10764] = -2.27731137E+01;
    COFD[10765] = 5.29493402E+00;
    COFD[10766] = -4.25285978E-01;
    COFD[10767] = 1.65604533E-02;
    COFD[10768] = -2.12354028E+01;
    COFD[10769] = 4.36095377E+00;
    COFD[10770] = -2.75760539E-01;
    COFD[10771] = 8.96430249E-03;
    COFD[10772] = -2.24179759E+01;
    COFD[10773] = 5.05061421E+00;
    COFD[10774] = -3.84359196E-01;
    COFD[10775] = 1.44214004E-02;
    COFD[10776] = -2.24302556E+01;
    COFD[10777] = 5.06106414E+00;
    COFD[10778] = -3.86053039E-01;
    COFD[10779] = 1.45081784E-02;
    COFD[10780] = -2.24360850E+01;
    COFD[10781] = 5.06106414E+00;
    COFD[10782] = -3.86053039E-01;
    COFD[10783] = 1.45081784E-02;
    COFD[10784] = -2.25751750E+01;
    COFD[10785] = 5.07648425E+00;
    COFD[10786] = -3.88560019E-01;
    COFD[10787] = 1.46368353E-02;
    COFD[10788] = -2.11462093E+01;
    COFD[10789] = 4.27612828E+00;
    COFD[10790] = -2.62774610E-01;
    COFD[10791] = 8.32471127E-03;
    COFD[10792] = -2.21717162E+01;
    COFD[10793] = 4.81459861E+00;
    COFD[10794] = -3.46990321E-01;
    COFD[10795] = 1.25347154E-02;
    COFD[10796] = -2.23002803E+01;
    COFD[10797] = 4.87626494E+00;
    COFD[10798] = -3.56718447E-01;
    COFD[10799] = 1.30246317E-02;
    COFD[10800] = -2.16841653E+01;
    COFD[10801] = 4.59101412E+00;
    COFD[10802] = -3.11439033E-01;
    COFD[10803] = 1.07377082E-02;
    COFD[10804] = -2.12736663E+01;
    COFD[10805] = 4.33544466E+00;
    COFD[10806] = -2.71843874E-01;
    COFD[10807] = 8.77093391E-03;
    COFD[10808] = -2.17771745E+01;
    COFD[10809] = 4.48837319E+00;
    COFD[10810] = -2.95423315E-01;
    COFD[10811] = 9.93861345E-03;
    COFD[10812] = -2.17797084E+01;
    COFD[10813] = 4.48837319E+00;
    COFD[10814] = -2.95423315E-01;
    COFD[10815] = 9.93861345E-03;
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
    COFTD[36] = 3.83439056E-01;
    COFTD[37] = 3.62717894E-05;
    COFTD[38] = -2.40281409E-08;
    COFTD[39] = 5.52792966E-12;
    COFTD[40] = 1.22119780E-01;
    COFTD[41] = 6.18373616E-04;
    COFTD[42] = -3.28422593E-07;
    COFTD[43] = 5.44603522E-11;
    COFTD[44] = 3.24747031E-01;
    COFTD[45] = 1.77798548E-04;
    COFTD[46] = -1.08934732E-07;
    COFTD[47] = 2.03595881E-11;
    COFTD[48] = 2.93191523E-01;
    COFTD[49] = 4.01430006E-04;
    COFTD[50] = -2.30705763E-07;
    COFTD[51] = 4.05176586E-11;
    COFTD[52] = 4.30605547E-01;
    COFTD[53] = 9.35961902E-05;
    COFTD[54] = -6.03983623E-08;
    COFTD[55] = 1.23115170E-11;
    COFTD[56] = 1.22693382E-01;
    COFTD[57] = 6.21278143E-04;
    COFTD[58] = -3.29965208E-07;
    COFTD[59] = 5.47161548E-11;
    COFTD[60] = 3.24747031E-01;
    COFTD[61] = 1.77798548E-04;
    COFTD[62] = -1.08934732E-07;
    COFTD[63] = 2.03595881E-11;
    COFTD[64] = 3.31191185E-01;
    COFTD[65] = 1.81326714E-04;
    COFTD[66] = -1.11096391E-07;
    COFTD[67] = 2.07635959E-11;
    COFTD[68] = 1.40314191E-01;
    COFTD[69] = 6.01266129E-04;
    COFTD[70] = -3.21915137E-07;
    COFTD[71] = 5.36679068E-11;
    COFTD[72] = 3.39557243E-01;
    COFTD[73] = 1.79335036E-04;
    COFTD[74] = -1.10135705E-07;
    COFTD[75] = 2.06427239E-11;
    COFTD[76] = 1.31424053E-01;
    COFTD[77] = 6.16429134E-04;
    COFTD[78] = -3.28571348E-07;
    COFTD[79] = 5.46153434E-11;
    COFTD[80] = 2.78021896E-01;
    COFTD[81] = 3.89608886E-04;
    COFTD[82] = -2.23546590E-07;
    COFTD[83] = 3.92078724E-11;
    COFTD[84] = 2.76725963E-01;
    COFTD[85] = 3.87792818E-04;
    COFTD[86] = -2.22504581E-07;
    COFTD[87] = 3.90251143E-11;
    COFTD[88] = 1.59288984E-01;
    COFTD[89] = 6.02833801E-04;
    COFTD[90] = -3.24837576E-07;
    COFTD[91] = 5.43909010E-11;
    COFTD[92] = 1.60621157E-01;
    COFTD[93] = 6.07875449E-04;
    COFTD[94] = -3.27554273E-07;
    COFTD[95] = 5.48457855E-11;
    COFTD[96] = 1.36817715E-01;
    COFTD[97] = 6.41727473E-04;
    COFTD[98] = -3.42055963E-07;
    COFTD[99] = 5.68567648E-11;
    COFTD[100] = 1.37064455E-01;
    COFTD[101] = 6.42884781E-04;
    COFTD[102] = -3.42672835E-07;
    COFTD[103] = 5.69593018E-11;
    COFTD[104] = 2.58066832E-01;
    COFTD[105] = 4.05072593E-04;
    COFTD[106] = -2.30587443E-07;
    COFTD[107] = 4.01863841E-11;
    COFTD[108] = 3.86107464E-01;
    COFTD[109] = 2.28760446E-04;
    COFTD[110] = -1.39425040E-07;
    COFTD[111] = 2.58989754E-11;
    COFTD[112] = 2.59569092E-01;
    COFTD[113] = 4.07430603E-04;
    COFTD[114] = -2.31929740E-07;
    COFTD[115] = 4.04203173E-11;
    COFTD[116] = 1.59647939E-01;
    COFTD[117] = 6.04192274E-04;
    COFTD[118] = -3.25569591E-07;
    COFTD[119] = 5.45134698E-11;
    COFTD[120] = 2.40639006E-01;
    COFTD[121] = 4.82930111E-04;
    COFTD[122] = -2.70362190E-07;
    COFTD[123] = 4.65173265E-11;
    COFTD[124] = 2.82974392E-01;
    COFTD[125] = 3.73032949E-04;
    COFTD[126] = -2.14959161E-07;
    COFTD[127] = 3.78355155E-11;
    COFTD[128] = 1.41968940E-01;
    COFTD[129] = 6.31753578E-04;
    COFTD[130] = -3.37603052E-07;
    COFTD[131] = 5.62125242E-11;
    COFTD[132] = 1.59647939E-01;
    COFTD[133] = 6.04192274E-04;
    COFTD[134] = -3.25569591E-07;
    COFTD[135] = 5.45134698E-11;
    COFTD[136] = 1.45341856E-01;
    COFTD[137] = 6.46762858E-04;
    COFTD[138] = -3.45623868E-07;
    COFTD[139] = 5.75480284E-11;
    COFTD[140] = 3.20990558E-01;
    COFTD[141] = 3.40833828E-04;
    COFTD[142] = -1.99444117E-07;
    COFTD[143] = 3.55703764E-11;
    COFTD[144] = 2.26670609E-01;
    COFTD[145] = 4.98251023E-04;
    COFTD[146] = -2.77281385E-07;
    COFTD[147] = 4.74970799E-11;
    COFTD[148] = 2.27261590E-01;
    COFTD[149] = 4.99550076E-04;
    COFTD[150] = -2.78004320E-07;
    COFTD[151] = 4.76209155E-11;
    COFTD[152] = 2.34098762E-01;
    COFTD[153] = 4.91099181E-04;
    COFTD[154] = -2.74133967E-07;
    COFTD[155] = 4.70636702E-11;
    COFTD[156] = 2.44452926E-01;
    COFTD[157] = 4.78884724E-04;
    COFTD[158] = -2.68527379E-07;
    COFTD[159] = 4.62572763E-11;
    COFTD[160] = 1.31648645E-01;
    COFTD[161] = 6.75329826E-04;
    COFTD[162] = -3.58458833E-07;
    COFTD[163] = 5.94176903E-11;
    COFTD[164] = 2.10934836E-01;
    COFTD[165] = 5.46607649E-04;
    COFTD[166] = -3.01041232E-07;
    COFTD[167] = 5.11789725E-11;
    COFTD[168] = 2.12562541E-01;
    COFTD[169] = 5.45357255E-04;
    COFTD[170] = -3.00537881E-07;
    COFTD[171] = 5.11159625E-11;
    COFTD[172] = 2.12842514E-01;
    COFTD[173] = 5.46075564E-04;
    COFTD[174] = -3.00933730E-07;
    COFTD[175] = 5.11832891E-11;
    COFTD[176] = 2.15139505E-01;
    COFTD[177] = 5.43740408E-04;
    COFTD[178] = -2.99926299E-07;
    COFTD[179] = 5.10460631E-11;
    COFTD[180] = 1.35158514E-01;
    COFTD[181] = 6.77932393E-04;
    COFTD[182] = -3.60212591E-07;
    COFTD[183] = 5.97492207E-11;
    COFTD[184] = 1.88615713E-01;
    COFTD[185] = 5.93233141E-04;
    COFTD[186] = -3.22963156E-07;
    COFTD[187] = 5.44599859E-11;
    COFTD[188] = 1.94963988E-01;
    COFTD[189] = 5.84809248E-04;
    COFTD[190] = -3.19255828E-07;
    COFTD[191] = 5.39384246E-11;
    COFTD[192] = 1.62995868E-01;
    COFTD[193] = 6.30693413E-04;
    COFTD[194] = -3.39450362E-07;
    COFTD[195] = 5.67923159E-11;
    COFTD[196] = 1.39990702E-01;
    COFTD[197] = 6.69283762E-04;
    COFTD[198] = -3.56418508E-07;
    COFTD[199] = 5.92081617E-11;
    COFTD[200] = 1.55057696E-01;
    COFTD[201] = 6.55342553E-04;
    COFTD[202] = -3.51111538E-07;
    COFTD[203] = 5.85627121E-11;
    COFTD[204] = 1.55121130E-01;
    COFTD[205] = 6.55610653E-04;
    COFTD[206] = -3.51255177E-07;
    COFTD[207] = 5.85866700E-11;
    COFTD[208] = 2.01521643E-01;
    COFTD[209] = 5.62744089E-04;
    COFTD[210] = -3.08519239E-07;
    COFTD[211] = 5.22805986E-11;
    COFTD[212] = 2.35283119E-01;
    COFTD[213] = 4.65670599E-04;
    COFTD[214] = -2.60939824E-07;
    COFTD[215] = 4.49271822E-11;
    COFTD[216] = 1.44152190E-01;
    COFTD[217] = 7.99993584E-05;
    COFTD[218] = -4.89707442E-08;
    COFTD[219] = 9.14277269E-12;
    COFTD[220] = 0.00000000E+00;
    COFTD[221] = 0.00000000E+00;
    COFTD[222] = 0.00000000E+00;
    COFTD[223] = 0.00000000E+00;
    COFTD[224] = 2.37053352E-01;
    COFTD[225] = 4.69174231E-04;
    COFTD[226] = -2.62903094E-07;
    COFTD[227] = 4.52652072E-11;
    COFTD[228] = -1.74352698E-01;
    COFTD[229] = 8.62246873E-04;
    COFTD[230] = -3.79545489E-07;
    COFTD[231] = 5.60262093E-11;
    COFTD[232] = 1.79840299E-01;
    COFTD[233] = 6.01722902E-04;
    COFTD[234] = -3.26433894E-07;
    COFTD[235] = 5.49112302E-11;
    COFTD[236] = 1.80186965E-01;
    COFTD[237] = 6.02882805E-04;
    COFTD[238] = -3.27063140E-07;
    COFTD[239] = 5.50170790E-11;
    COFTD[240] = 1.80513677E-01;
    COFTD[241] = 6.03975942E-04;
    COFTD[242] = -3.27656165E-07;
    COFTD[243] = 5.51168351E-11;
    COFTD[244] = 2.28560867E-01;
    COFTD[245] = 4.52365967E-04;
    COFTD[246] = -2.53484536E-07;
    COFTD[247] = 4.36435719E-11;
    COFTD[248] = -1.60981264E-01;
    COFTD[249] = 9.03807572E-04;
    COFTD[250] = -4.06927941E-07;
    COFTD[251] = 6.09202254E-11;
    COFTD[252] = 9.90752318E-02;
    COFTD[253] = 6.44201384E-04;
    COFTD[254] = -3.38485953E-07;
    COFTD[255] = 5.57356746E-11;
    COFTD[256] = -2.00309448E-02;
    COFTD[257] = 8.50440115E-04;
    COFTD[258] = -4.21064468E-07;
    COFTD[259] = 6.67959710E-11;
    COFTD[260] = 2.00119897E-01;
    COFTD[261] = 5.64793704E-04;
    COFTD[262] = -3.09445484E-07;
    COFTD[263] = 5.24139335E-11;
    COFTD[264] = -1.61357564E-01;
    COFTD[265] = 9.05920260E-04;
    COFTD[266] = -4.07879153E-07;
    COFTD[267] = 6.10626290E-11;
    COFTD[268] = 9.90752318E-02;
    COFTD[269] = 6.44201384E-04;
    COFTD[270] = -3.38485953E-07;
    COFTD[271] = 5.57356746E-11;
    COFTD[272] = 1.00039110E-01;
    COFTD[273] = 6.50468660E-04;
    COFTD[274] = -3.41778999E-07;
    COFTD[275] = 5.62779132E-11;
    COFTD[276] = -1.31244519E-01;
    COFTD[277] = 9.03901384E-04;
    COFTD[278] = -4.17831507E-07;
    COFTD[279] = 6.35725667E-11;
    COFTD[280] = 1.05124122E-01;
    COFTD[281] = 6.50665957E-04;
    COFTD[282] = -3.42564538E-07;
    COFTD[283] = 5.64804120E-11;
    COFTD[284] = -1.56651581E-01;
    COFTD[285] = 9.09789751E-04;
    COFTD[286] = -4.11714242E-07;
    COFTD[287] = 6.18310893E-11;
    COFTD[288] = -2.28637575E-02;
    COFTD[289] = 8.35412914E-04;
    COFTD[290] = -4.12929260E-07;
    COFTD[291] = 6.54380945E-11;
    COFTD[292] = -2.28105944E-02;
    COFTD[293] = 8.33470403E-04;
    COFTD[294] = -4.11969112E-07;
    COFTD[295] = 6.52859371E-11;
    COFTD[296] = -1.41640506E-01;
    COFTD[297] = 9.21404324E-04;
    COFTD[298] = -4.23210110E-07;
    COFTD[299] = 6.41400322E-11;
    COFTD[300] = -1.42230624E-01;
    COFTD[301] = 9.25243177E-04;
    COFTD[302] = -4.24973333E-07;
    COFTD[303] = 6.44072593E-11;
    COFTD[304] = -1.59826932E-01;
    COFTD[305] = 9.28231324E-04;
    COFTD[306] = -4.20059750E-07;
    COFTD[307] = 6.30844146E-11;
    COFTD[308] = -1.59970790E-01;
    COFTD[309] = 9.29066816E-04;
    COFTD[310] = -4.20437842E-07;
    COFTD[311] = 6.31411962E-11;
    COFTD[312] = -3.81470765E-02;
    COFTD[313] = 8.39833490E-04;
    COFTD[314] = -4.11688915E-07;
    COFTD[315] = 6.49124952E-11;
    COFTD[316] = 9.86934401E-02;
    COFTD[317] = 7.20974863E-04;
    COFTD[318] = -3.77135221E-07;
    COFTD[319] = 6.19202579E-11;
    COFTD[320] = -3.82574649E-02;
    COFTD[321] = 8.42263764E-04;
    COFTD[322] = -4.12880242E-07;
    COFTD[323] = 6.51003362E-11;
    COFTD[324] = -1.41799739E-01;
    COFTD[325] = 9.22440172E-04;
    COFTD[326] = -4.23685885E-07;
    COFTD[327] = 6.42121388E-11;
    COFTD[328] = -7.23038994E-02;
    COFTD[329] = 8.89466098E-04;
    COFTD[330] = -4.28124818E-07;
    COFTD[331] = 6.67586244E-11;
    COFTD[332] = -1.42100396E-02;
    COFTD[333] = 8.23812102E-04;
    COFTD[334] = -4.08995515E-07;
    COFTD[335] = 6.49899310E-11;
    COFTD[336] = -1.55536623E-01;
    COFTD[337] = 9.26290092E-04;
    COFTD[338] = -4.20679731E-07;
    COFTD[339] = 6.33165565E-11;
    COFTD[340] = -1.41799739E-01;
    COFTD[341] = 9.22440172E-04;
    COFTD[342] = -4.23685885E-07;
    COFTD[343] = 6.42121388E-11;
    COFTD[344] = -1.57371296E-01;
    COFTD[345] = 9.37216387E-04;
    COFTD[346] = -4.25641968E-07;
    COFTD[347] = 6.40634233E-11;
    COFTD[348] = 1.67288378E-02;
    COFTD[349] = 8.09703539E-04;
    COFTD[350] = -4.08157430E-07;
    COFTD[351] = 6.54674608E-11;
    COFTD[352] = -8.34877147E-02;
    COFTD[353] = 8.93466011E-04;
    COFTD[354] = -4.27125851E-07;
    COFTD[355] = 6.63277969E-11;
    COFTD[356] = -8.35962674E-02;
    COFTD[357] = 8.94627716E-04;
    COFTD[358] = -4.27681210E-07;
    COFTD[359] = 6.64140378E-11;
    COFTD[360] = -7.78657454E-02;
    COFTD[361] = 8.92101015E-04;
    COFTD[362] = -4.27969255E-07;
    COFTD[363] = 6.66000503E-11;
    COFTD[364] = -6.92602151E-02;
    COFTD[365] = 8.88360172E-04;
    COFTD[366] = -4.28365765E-07;
    COFTD[367] = 6.68694606E-11;
    COFTD[368] = -1.62301128E-01;
    COFTD[369] = 9.43217155E-04;
    COFTD[370] = -4.26881994E-07;
    COFTD[371] = 6.41127358E-11;
    COFTD[372] = -1.04463705E-01;
    COFTD[373] = 9.17317898E-04;
    COFTD[374] = -4.33159478E-07;
    COFTD[375] = 6.67640055E-11;
    COFTD[376] = -1.03383911E-01;
    COFTD[377] = 9.17368946E-04;
    COFTD[378] = -4.33506203E-07;
    COFTD[379] = 6.68476435E-11;
    COFTD[380] = -1.03451906E-01;
    COFTD[381] = 9.17972300E-04;
    COFTD[382] = -4.33791320E-07;
    COFTD[383] = 6.68916093E-11;
    COFTD[384] = -1.01770402E-01;
    COFTD[385] = 9.17667014E-04;
    COFTD[386] = -4.34133683E-07;
    COFTD[387] = 6.69899783E-11;
    COFTD[388] = -1.68032187E-01;
    COFTD[389] = 9.47207191E-04;
    COFTD[390] = -4.26734789E-07;
    COFTD[391] = 6.39101707E-11;
    COFTD[392] = -1.26505967E-01;
    COFTD[393] = 9.33839500E-04;
    COFTD[394] = -4.34686874E-07;
    COFTD[395] = 6.64184154E-11;
    COFTD[396] = -1.21244149E-01;
    COFTD[397] = 9.32247360E-04;
    COFTD[398] = -4.35568436E-07;
    COFTD[399] = 6.67044478E-11;
    COFTD[400] = -1.46424793E-01;
    COFTD[401] = 9.40060436E-04;
    COFTD[402] = -4.31101818E-07;
    COFTD[403] = 6.52733173E-11;
    COFTD[404] = -1.64313478E-01;
    COFTD[405] = 9.45982311E-04;
    COFTD[406] = -4.27543725E-07;
    COFTD[407] = 6.41575424E-11;
    COFTD[408] = -1.55196478E-01;
    COFTD[409] = 9.48793311E-04;
    COFTD[410] = -4.32429136E-07;
    COFTD[411] = 6.52268964E-11;
    COFTD[412] = -1.55228210E-01;
    COFTD[413] = 9.48987307E-04;
    COFTD[414] = -4.32517553E-07;
    COFTD[415] = 6.52402330E-11;
}

/* Replace this routine with the one generated by the Gauss Jordan solver of DW */
AMREX_GPU_HOST_DEVICE void sgjsolve(double* A, double* x, double* b) {
    amrex::Abort("sgjsolve not implemented, choose a different solver ");
}

/* Replace this routine with the one generated by the Gauss Jordan solver of DW */
AMREX_GPU_HOST_DEVICE void sgjsolve_simplified(double* A, double* x, double* b) {
    amrex::Abort("sgjsolve_simplified not implemented, choose a different solver ");
}

