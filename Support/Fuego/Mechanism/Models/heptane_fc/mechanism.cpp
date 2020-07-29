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
static AMREX_GPU_DEVICE_MANAGED double imw[38] = {
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
    1.0 / 47.033860,  /*CH3O2 */
    1.0 / 26.038240,  /*C2H2 */
    1.0 / 42.081270,  /*C3H6 */
    1.0 / 28.054180,  /*C2H4 */
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
static AMREX_GPU_DEVICE_MANAGED double molecular_weights[38] = {
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
    47.033860,  /*CH3O2 */
    26.038240,  /*C2H2 */
    42.081270,  /*C3H6 */
    28.054180,  /*C2H4 */
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
    for(int i = 0; i<38; ++i) imw_new[i] = imw[i];
}

/* TODO: check necessity because redundant with CKWT */
AMREX_GPU_HOST_DEVICE
void get_mw(double mw_new[]){
    for(int i = 0; i<38; ++i) mw_new[i] = molecular_weights[i];
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
    TBid[11][0] = 47; TB[11][0] = 0; // CH2CHO
    TBid[11][1] = 38; TB[11][1] = 0; // CH
    TBid[11][2] = 10; TB[11][2] = 1.8999999999999999; // CO
    TBid[11][3] = 2; TB[11][3] = 2.5; // H2
    TBid[11][4] = 30; TB[11][4] = 0; // PXC4H9
    TBid[11][5] = 41; TB[11][5] = 0; // CH2GSG
    TBid[11][6] = 28; TB[11][6] = 0; // C4H7
    TBid[11][7] = 45; TB[11][7] = 0; // HCCO
    TBid[11][8] = 34; TB[11][8] = 0; // C5H11X1
    TBid[11][9] = 5; TB[11][9] = 12; // H2O
    TBid[11][10] = 9; TB[11][10] = 3.7999999999999998; // CO2
    TBid[11][11] = 48; TB[11][11] = 0; // C2H5O
    TBid[11][12] = 51; TB[11][12] = 0; // C3H2
    TBid[11][13] = 49; TB[11][13] = 0; // CH3CO
    TBid[11][14] = 36; TB[11][14] = 0; // C7H15X2

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
    TBid[0][0] = 47; TB[0][0] = 0; // CH2CHO
    TBid[0][1] = 38; TB[0][1] = 0; // CH
    TBid[0][2] = 10; TB[0][2] = 1.8999999999999999; // CO
    TBid[0][3] = 2; TB[0][3] = 2.5; // H2
    TBid[0][4] = 30; TB[0][4] = 0; // PXC4H9
    TBid[0][5] = 41; TB[0][5] = 0; // CH2GSG
    TBid[0][6] = 28; TB[0][6] = 0; // C4H7
    TBid[0][7] = 45; TB[0][7] = 0; // HCCO
    TBid[0][8] = 34; TB[0][8] = 0; // C5H11X1
    TBid[0][9] = 5; TB[0][9] = 12; // H2O
    TBid[0][10] = 9; TB[0][10] = 3.7999999999999998; // CO2
    TBid[0][11] = 48; TB[0][11] = 0; // C2H5O
    TBid[0][12] = 51; TB[0][12] = 0; // C3H2
    TBid[0][13] = 49; TB[0][13] = 0; // CH3CO
    TBid[0][14] = 36; TB[0][14] = 0; // C7H15X2

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
    TBid[1][0] = 47; TB[1][0] = 0; // CH2CHO
    TBid[1][1] = 38; TB[1][1] = 0; // CH
    TBid[1][2] = 10; TB[1][2] = 1.8999999999999999; // CO
    TBid[1][3] = 2; TB[1][3] = 2.5; // H2
    TBid[1][4] = 30; TB[1][4] = 0; // PXC4H9
    TBid[1][5] = 41; TB[1][5] = 0; // CH2GSG
    TBid[1][6] = 28; TB[1][6] = 0; // C4H7
    TBid[1][7] = 45; TB[1][7] = 0; // HCCO
    TBid[1][8] = 34; TB[1][8] = 0; // C5H11X1
    TBid[1][9] = 5; TB[1][9] = 12; // H2O
    TBid[1][10] = 9; TB[1][10] = 3.7999999999999998; // CO2
    TBid[1][11] = 48; TB[1][11] = 0; // C2H5O
    TBid[1][12] = 51; TB[1][12] = 0; // C3H2
    TBid[1][13] = 49; TB[1][13] = 0; // CH3CO
    TBid[1][14] = 36; TB[1][14] = 0; // C7H15X2

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
    nTB[12] = 11;
    TB[12] = (double *) malloc(11 * sizeof(double));
    TBid[12] = (int *) malloc(11 * sizeof(int));
    TBid[12][0] = 47; TB[12][0] = 0; // CH2CHO
    TBid[12][1] = 38; TB[12][1] = 0; // CH
    TBid[12][2] = 51; TB[12][2] = 0; // C3H2
    TBid[12][3] = 30; TB[12][3] = 0; // PXC4H9
    TBid[12][4] = 41; TB[12][4] = 0; // CH2GSG
    TBid[12][5] = 28; TB[12][5] = 0; // C4H7
    TBid[12][6] = 45; TB[12][6] = 0; // HCCO
    TBid[12][7] = 34; TB[12][7] = 0; // C5H11X1
    TBid[12][8] = 49; TB[12][8] = 0; // CH3CO
    TBid[12][9] = 48; TB[12][9] = 0; // C2H5O
    TBid[12][10] = 36; TB[12][10] = 0; // C7H15X2

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
    nTB[13] = 11;
    TB[13] = (double *) malloc(11 * sizeof(double));
    TBid[13] = (int *) malloc(11 * sizeof(int));
    TBid[13][0] = 47; TB[13][0] = 0; // CH2CHO
    TBid[13][1] = 38; TB[13][1] = 0; // CH
    TBid[13][2] = 51; TB[13][2] = 0; // C3H2
    TBid[13][3] = 30; TB[13][3] = 0; // PXC4H9
    TBid[13][4] = 41; TB[13][4] = 0; // CH2GSG
    TBid[13][5] = 28; TB[13][5] = 0; // C4H7
    TBid[13][6] = 45; TB[13][6] = 0; // HCCO
    TBid[13][7] = 34; TB[13][7] = 0; // C5H11X1
    TBid[13][8] = 49; TB[13][8] = 0; // CH3CO
    TBid[13][9] = 48; TB[13][9] = 0; // C2H5O
    TBid[13][10] = 36; TB[13][10] = 0; // C7H15X2

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
    nTB[2] = 17;
    TB[2] = (double *) malloc(17 * sizeof(double));
    TBid[2] = (int *) malloc(17 * sizeof(int));
    TBid[2][0] = 47; TB[2][0] = 0; // CH2CHO
    TBid[2][1] = 38; TB[2][1] = 0; // CH
    TBid[2][2] = 10; TB[2][2] = 1.5; // CO
    TBid[2][3] = 2; TB[2][3] = 2; // H2
    TBid[2][4] = 30; TB[2][4] = 0; // PXC4H9
    TBid[2][5] = 41; TB[2][5] = 0; // CH2GSG
    TBid[2][6] = 28; TB[2][6] = 0; // C4H7
    TBid[2][7] = 45; TB[2][7] = 0; // HCCO
    TBid[2][8] = 34; TB[2][8] = 0; // C5H11X1
    TBid[2][9] = 5; TB[2][9] = 6; // H2O
    TBid[2][10] = 9; TB[2][10] = 2; // CO2
    TBid[2][11] = 13; TB[2][11] = 2; // CH4
    TBid[2][12] = 15; TB[2][12] = 3; // C2H6
    TBid[2][13] = 48; TB[2][13] = 0; // C2H5O
    TBid[2][14] = 51; TB[2][14] = 0; // C3H2
    TBid[2][15] = 49; TB[2][15] = 0; // CH3CO
    TBid[2][16] = 36; TB[2][16] = 0; // C7H15X2

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
    nTB[3] = 15;
    TB[3] = (double *) malloc(15 * sizeof(double));
    TBid[3] = (int *) malloc(15 * sizeof(int));
    TBid[3][0] = 47; TB[3][0] = 0; // CH2CHO
    TBid[3][1] = 38; TB[3][1] = 0; // CH
    TBid[3][2] = 10; TB[3][2] = 2; // CO
    TBid[3][3] = 2; TB[3][3] = 2; // H2
    TBid[3][4] = 30; TB[3][4] = 0; // PXC4H9
    TBid[3][5] = 41; TB[3][5] = 0; // CH2GSG
    TBid[3][6] = 28; TB[3][6] = 0; // C4H7
    TBid[3][7] = 45; TB[3][7] = 0; // HCCO
    TBid[3][8] = 34; TB[3][8] = 0; // C5H11X1
    TBid[3][9] = 5; TB[3][9] = 5; // H2O
    TBid[3][10] = 9; TB[3][10] = 3; // CO2
    TBid[3][11] = 48; TB[3][11] = 0; // C2H5O
    TBid[3][12] = 51; TB[3][12] = 0; // C3H2
    TBid[3][13] = 49; TB[3][13] = 0; // CH3CO
    TBid[3][14] = 36; TB[3][14] = 0; // C7H15X2

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
    nTB[4] = 15;
    TB[4] = (double *) malloc(15 * sizeof(double));
    TBid[4] = (int *) malloc(15 * sizeof(int));
    TBid[4][0] = 47; TB[4][0] = 0; // CH2CHO
    TBid[4][1] = 38; TB[4][1] = 0; // CH
    TBid[4][2] = 10; TB[4][2] = 2; // CO
    TBid[4][3] = 2; TB[4][3] = 2; // H2
    TBid[4][4] = 30; TB[4][4] = 0; // PXC4H9
    TBid[4][5] = 41; TB[4][5] = 0; // CH2GSG
    TBid[4][6] = 28; TB[4][6] = 0; // C4H7
    TBid[4][7] = 45; TB[4][7] = 0; // HCCO
    TBid[4][8] = 34; TB[4][8] = 0; // C5H11X1
    TBid[4][9] = 5; TB[4][9] = 5; // H2O
    TBid[4][10] = 9; TB[4][10] = 3; // CO2
    TBid[4][11] = 48; TB[4][11] = 0; // C2H5O
    TBid[4][12] = 51; TB[4][12] = 0; // C3H2
    TBid[4][13] = 49; TB[4][13] = 0; // CH3CO
    TBid[4][14] = 36; TB[4][14] = 0; // C7H15X2

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
    nTB[5] = 11;
    TB[5] = (double *) malloc(11 * sizeof(double));
    TBid[5] = (int *) malloc(11 * sizeof(int));
    TBid[5][0] = 47; TB[5][0] = 0; // CH2CHO
    TBid[5][1] = 38; TB[5][1] = 0; // CH
    TBid[5][2] = 51; TB[5][2] = 0; // C3H2
    TBid[5][3] = 30; TB[5][3] = 0; // PXC4H9
    TBid[5][4] = 41; TB[5][4] = 0; // CH2GSG
    TBid[5][5] = 28; TB[5][5] = 0; // C4H7
    TBid[5][6] = 45; TB[5][6] = 0; // HCCO
    TBid[5][7] = 34; TB[5][7] = 0; // C5H11X1
    TBid[5][8] = 49; TB[5][8] = 0; // CH3CO
    TBid[5][9] = 48; TB[5][9] = 0; // C2H5O
    TBid[5][10] = 36; TB[5][10] = 0; // C7H15X2

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
    nTB[6] = 15;
    TB[6] = (double *) malloc(15 * sizeof(double));
    TBid[6] = (int *) malloc(15 * sizeof(int));
    TBid[6][0] = 47; TB[6][0] = 0; // CH2CHO
    TBid[6][1] = 38; TB[6][1] = 0; // CH
    TBid[6][2] = 10; TB[6][2] = 1.8999999999999999; // CO
    TBid[6][3] = 2; TB[6][3] = 2.5; // H2
    TBid[6][4] = 30; TB[6][4] = 0; // PXC4H9
    TBid[6][5] = 41; TB[6][5] = 0; // CH2GSG
    TBid[6][6] = 28; TB[6][6] = 0; // C4H7
    TBid[6][7] = 45; TB[6][7] = 0; // HCCO
    TBid[6][8] = 34; TB[6][8] = 0; // C5H11X1
    TBid[6][9] = 5; TB[6][9] = 12; // H2O
    TBid[6][10] = 9; TB[6][10] = 3.7999999999999998; // CO2
    TBid[6][11] = 48; TB[6][11] = 0; // C2H5O
    TBid[6][12] = 51; TB[6][12] = 0; // C3H2
    TBid[6][13] = 49; TB[6][13] = 0; // CH3CO
    TBid[6][14] = 36; TB[6][14] = 0; // C7H15X2

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
    nTB[14] = 15;
    TB[14] = (double *) malloc(15 * sizeof(double));
    TBid[14] = (int *) malloc(15 * sizeof(int));
    TBid[14][0] = 47; TB[14][0] = 0; // CH2CHO
    TBid[14][1] = 38; TB[14][1] = 0; // CH
    TBid[14][2] = 10; TB[14][2] = 1.8999999999999999; // CO
    TBid[14][3] = 2; TB[14][3] = 2.5; // H2
    TBid[14][4] = 30; TB[14][4] = 0; // PXC4H9
    TBid[14][5] = 41; TB[14][5] = 0; // CH2GSG
    TBid[14][6] = 28; TB[14][6] = 0; // C4H7
    TBid[14][7] = 45; TB[14][7] = 0; // HCCO
    TBid[14][8] = 34; TB[14][8] = 0; // C5H11X1
    TBid[14][9] = 5; TB[14][9] = 6; // H2O
    TBid[14][10] = 9; TB[14][10] = 3.7999999999999998; // CO2
    TBid[14][11] = 48; TB[14][11] = 0; // C2H5O
    TBid[14][12] = 51; TB[14][12] = 0; // C3H2
    TBid[14][13] = 49; TB[14][13] = 0; // CH3CO
    TBid[14][14] = 36; TB[14][14] = 0; // C7H15X2

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
    nTB[7] = 11;
    TB[7] = (double *) malloc(11 * sizeof(double));
    TBid[7] = (int *) malloc(11 * sizeof(int));
    TBid[7][0] = 47; TB[7][0] = 0; // CH2CHO
    TBid[7][1] = 38; TB[7][1] = 0; // CH
    TBid[7][2] = 51; TB[7][2] = 0; // C3H2
    TBid[7][3] = 30; TB[7][3] = 0; // PXC4H9
    TBid[7][4] = 41; TB[7][4] = 0; // CH2GSG
    TBid[7][5] = 28; TB[7][5] = 0; // C4H7
    TBid[7][6] = 45; TB[7][6] = 0; // HCCO
    TBid[7][7] = 34; TB[7][7] = 0; // C5H11X1
    TBid[7][8] = 49; TB[7][8] = 0; // CH3CO
    TBid[7][9] = 48; TB[7][9] = 0; // C2H5O
    TBid[7][10] = 36; TB[7][10] = 0; // C7H15X2

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

    // (82):  2.000000 CH3O2 => O2 + 2.000000 CH3O
    kiv[88] = {18,6};
    nuv[88] = {-2.0,1};
    kiv_qss[88] = {4};
    nuv_qss[88] = {2.0};
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
    kiv[89] = {18,12};
    nuv[89] = {-1,-1};
    kiv_qss[89] = {4};
    nuv_qss[89] = {2.0};
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
    kiv[90] = {18,11,14,6};
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
    kiv[91] = {18,7,6};
    nuv[91] = {-1,-1,1};
    kiv_qss[91] = {6};
    nuv_qss[91] = {1};
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
    kiv[15] = {18,12,6};
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
    TBid[15][0] = 47; TB[15][0] = 0; // CH2CHO
    TBid[15][1] = 38; TB[15][1] = 0; // CH
    TBid[15][2] = 51; TB[15][2] = 0; // C3H2
    TBid[15][3] = 30; TB[15][3] = 0; // PXC4H9
    TBid[15][4] = 41; TB[15][4] = 0; // CH2GSG
    TBid[15][5] = 28; TB[15][5] = 0; // C4H7
    TBid[15][6] = 45; TB[15][6] = 0; // HCCO
    TBid[15][7] = 34; TB[15][7] = 0; // C5H11X1
    TBid[15][8] = 49; TB[15][8] = 0; // CH3CO
    TBid[15][9] = 48; TB[15][9] = 0; // C2H5O
    TBid[15][10] = 36; TB[15][10] = 0; // C7H15X2

    // (87):  CH3 + O2 + M => CH3O2 + M
    kiv[16] = {12,6,18};
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
    TBid[16][0] = 47; TB[16][0] = 0; // CH2CHO
    TBid[16][1] = 38; TB[16][1] = 0; // CH
    TBid[16][2] = 51; TB[16][2] = 0; // C3H2
    TBid[16][3] = 30; TB[16][3] = 0; // PXC4H9
    TBid[16][4] = 41; TB[16][4] = 0; // CH2GSG
    TBid[16][5] = 28; TB[16][5] = 0; // C4H7
    TBid[16][6] = 45; TB[16][6] = 0; // HCCO
    TBid[16][7] = 34; TB[16][7] = 0; // C5H11X1
    TBid[16][8] = 49; TB[16][8] = 0; // CH3CO
    TBid[16][9] = 48; TB[16][9] = 0; // C2H5O
    TBid[16][10] = 36; TB[16][10] = 0; // C7H15X2

    // (88):  CH3O2H => CH3O + OH
    kiv[92] = {4};
    nuv[92] = {1};
    kiv_qss[92] = {6,4};
    nuv_qss[92] = {-1,1};
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
    kiv[93] = {19,1,10};
    nuv[93] = {-1,-1,1};
    kiv_qss[93] = {2};
    nuv_qss[93] = {1};
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
    kiv[94] = {19,1,3};
    nuv[94] = {-1,-1,1};
    kiv_qss[94] = {7};
    nuv_qss[94] = {1};
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
    kiv[95] = {3,19,2};
    nuv[95] = {-1,1,1};
    kiv_qss[95] = {8};
    nuv_qss[95] = {-1};
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
    kiv[96] = {6,1};
    nuv[96] = {-1,1};
    kiv_qss[96] = {8,9};
    nuv_qss[96] = {-1,1};
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
    kiv[97] = {12,20};
    nuv[97] = {-1,1};
    kiv_qss[97] = {8};
    nuv_qss[97] = {-1};
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
    kiv[98] = {6,19,7};
    nuv[98] = {-1,1,1};
    kiv_qss[98] = {8};
    nuv_qss[98] = {-1};
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
    kiv[99] = {6,11};
    nuv[99] = {-1,1};
    kiv_qss[99] = {8,1};
    nuv_qss[99] = {-1,1};
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
    kiv[8] = {3,19};
    nuv[8] = {1,1};
    kiv_qss[8] = {8};
    nuv_qss[8] = {-1};
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
    TBid[8][0] = 47; TB[8][0] = 0; // CH2CHO
    TBid[8][1] = 38; TB[8][1] = 0; // CH
    TBid[8][2] = 10; TB[8][2] = 2; // CO
    TBid[8][3] = 2; TB[8][3] = 2; // H2
    TBid[8][4] = 30; TB[8][4] = 0; // PXC4H9
    TBid[8][5] = 41; TB[8][5] = 0; // CH2GSG
    TBid[8][6] = 28; TB[8][6] = 0; // C4H7
    TBid[8][7] = 45; TB[8][7] = 0; // HCCO
    TBid[8][8] = 34; TB[8][8] = 0; // C5H11X1
    TBid[8][9] = 5; TB[8][9] = 5; // H2O
    TBid[8][10] = 9; TB[8][10] = 3; // CO2
    TBid[8][11] = 48; TB[8][11] = 0; // C2H5O
    TBid[8][12] = 51; TB[8][12] = 0; // C3H2
    TBid[8][13] = 49; TB[8][13] = 0; // CH3CO
    TBid[8][14] = 36; TB[8][14] = 0; // C7H15X2

    // (97):  C2H4 + CH3 => C2H3 + CH4
    kiv[100] = {21,12,13};
    nuv[100] = {-1,-1,1};
    kiv_qss[100] = {8};
    nuv_qss[100] = {1};
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
    kiv[101] = {21,1,12};
    nuv[101] = {-1,-1,1};
    kiv_qss[101] = {1};
    nuv_qss[101] = {1};
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
    kiv[102] = {21,4,5};
    nuv[102] = {-1,-1,1};
    kiv_qss[102] = {8};
    nuv_qss[102] = {1};
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
    kiv[9] = {3,21};
    nuv[9] = {-1,-1};
    kiv_qss[9] = {5};
    nuv_qss[9] = {1};
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
    TBid[9][0] = 47; TB[9][0] = 0; // CH2CHO
    TBid[9][1] = 38; TB[9][1] = 0; // CH
    TBid[9][2] = 51; TB[9][2] = 0; // C3H2
    TBid[9][3] = 30; TB[9][3] = 0; // PXC4H9
    TBid[9][4] = 41; TB[9][4] = 0; // CH2GSG
    TBid[9][5] = 28; TB[9][5] = 0; // C4H7
    TBid[9][6] = 45; TB[9][6] = 0; // HCCO
    TBid[9][7] = 34; TB[9][7] = 0; // C5H11X1
    TBid[9][8] = 49; TB[9][8] = 0; // CH3CO
    TBid[9][9] = 48; TB[9][9] = 0; // C2H5O
    TBid[9][10] = 36; TB[9][10] = 0; // C7H15X2

    // (101):  C2H4 + O => CH2CHO + H
    kiv[103] = {21,1,3};
    nuv[103] = {-1,-1,1};
    kiv_qss[103] = {9};
    nuv_qss[103] = {1};
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
    kiv[104] = {21,3,2};
    nuv[104] = {-1,-1,1};
    kiv_qss[104] = {8};
    nuv_qss[104] = {1};
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
    kiv[105] = {2,21,3};
    nuv[105] = {-1,1,1};
    kiv_qss[105] = {8};
    nuv_qss[105] = {-1};
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
    kiv[106] = {3,15};
    nuv[106] = {-1,1};
    kiv_qss[106] = {5};
    nuv_qss[106] = {-1};
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
    kiv[107] = {18};
    nuv[107] = {-1};
    kiv_qss[107] = {5,4,10};
    nuv_qss[107] = {-1,1,1};
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
    kiv[108] = {7,4};
    nuv[108] = {-1,1};
    kiv_qss[108] = {5,10};
    nuv_qss[108] = {-1,1};
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
    kiv[109] = {6,21,7};
    nuv[109] = {-1,1,1};
    kiv_qss[109] = {5};
    nuv_qss[109] = {-1};
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
    kiv[110] = {15,1,4};
    nuv[110] = {-1,-1,1};
    kiv_qss[110] = {5};
    nuv_qss[110] = {1};
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
    kiv[111] = {15,4,5};
    nuv[111] = {-1,-1,1};
    kiv_qss[111] = {5};
    nuv_qss[111] = {1};
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
    kiv[112] = {15,3,2};
    nuv[112] = {-1,-1,1};
    kiv_qss[112] = {5};
    nuv_qss[112] = {1};
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
    kiv[113] = {1,3,10};
    nuv[113] = {-1,1,2.0};
    kiv_qss[113] = {7};
    nuv_qss[113] = {-1};
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
    kiv[114] = {4};
    nuv[114] = {-1};
    kiv_qss[114] = {7,1};
    nuv_qss[114] = {-1,2.0};
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
    kiv[115] = {6,9};
    nuv[115] = {-1,1};
    kiv_qss[115] = {7,1};
    nuv_qss[115] = {-1,1};
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
    kiv[116] = {3,10};
    nuv[116] = {-1,1};
    kiv_qss[116] = {7,3};
    nuv_qss[116] = {-1,1};
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
    kiv[117] = {10,3};
    nuv[117] = {-1,1};
    kiv_qss[117] = {3,7};
    nuv_qss[117] = {-1,1};
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
    kiv[118] = {16,1,4};
    nuv[118] = {-1,-1,1};
    kiv_qss[118] = {7};
    nuv_qss[118] = {1};
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
    kiv[119] = {16,3,2};
    nuv[119] = {-1,-1,1};
    kiv_qss[119] = {7};
    nuv_qss[119] = {1};
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
    kiv[120] = {2,16,3};
    nuv[120] = {-1,1,1};
    kiv_qss[120] = {7};
    nuv_qss[120] = {-1};
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
    kiv[121] = {16,3,12,10};
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
    kiv[122] = {16,1,9};
    nuv[122] = {-1,-1,1};
    kiv_qss[122] = {2};
    nuv_qss[122] = {1};
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
    kiv[123] = {16,4,5};
    nuv[123] = {-1,-1,1};
    kiv_qss[123] = {7};
    nuv_qss[123] = {1};
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
    kiv[124] = {6,11,10,4};
    nuv[124] = {-1,1,1,1};
    kiv_qss[124] = {9};
    nuv_qss[124] = {-1};
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
    kiv[125] = {16,3};
    nuv[125] = {1,1};
    kiv_qss[125] = {9};
    nuv_qss[125] = {-1};
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
    kiv[126] = {16,3};
    nuv[126] = {-1,-1};
    kiv_qss[126] = {9};
    nuv_qss[126] = {1};
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
    kiv[10] = {12,10};
    nuv[10] = {1,1};
    kiv_qss[10] = {11};
    nuv_qss[10] = {-1};
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
    TBid[10][0] = 47; TB[10][0] = 0; // CH2CHO
    TBid[10][1] = 38; TB[10][1] = 0; // CH
    TBid[10][2] = 51; TB[10][2] = 0; // C3H2
    TBid[10][3] = 30; TB[10][3] = 0; // PXC4H9
    TBid[10][4] = 41; TB[10][4] = 0; // CH2GSG
    TBid[10][5] = 28; TB[10][5] = 0; // C4H7
    TBid[10][6] = 45; TB[10][6] = 0; // HCCO
    TBid[10][7] = 34; TB[10][7] = 0; // C5H11X1
    TBid[10][8] = 49; TB[10][8] = 0; // CH3CO
    TBid[10][9] = 48; TB[10][9] = 0; // C2H5O
    TBid[10][10] = 36; TB[10][10] = 0; // C7H15X2

    // (126):  C2H5O + M => CH3 + CH2O + M
    kiv[17] = {12,11};
    nuv[17] = {1,1};
    kiv_qss[17] = {10};
    nuv_qss[17] = {-1};
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
    TBid[17][0] = 47; TB[17][0] = 0; // CH2CHO
    TBid[17][1] = 38; TB[17][1] = 0; // CH
    TBid[17][2] = 51; TB[17][2] = 0; // C3H2
    TBid[17][3] = 30; TB[17][3] = 0; // PXC4H9
    TBid[17][4] = 41; TB[17][4] = 0; // CH2GSG
    TBid[17][5] = 28; TB[17][5] = 0; // C4H7
    TBid[17][6] = 45; TB[17][6] = 0; // HCCO
    TBid[17][7] = 34; TB[17][7] = 0; // C5H11X1
    TBid[17][8] = 49; TB[17][8] = 0; // CH3CO
    TBid[17][9] = 48; TB[17][9] = 0; // C2H5O
    TBid[17][10] = 36; TB[17][10] = 0; // C7H15X2

    // (127):  C2H5O2 => C2H5 + O2
    kiv[127] = {6};
    nuv[127] = {1};
    kiv_qss[127] = {12,5};
    nuv_qss[127] = {-1,1};
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
    kiv[128] = {6};
    nuv[128] = {-1};
    kiv_qss[128] = {5,12};
    nuv_qss[128] = {-1,1};
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
    kiv[129] = {21,7};
    nuv[129] = {1,1};
    kiv_qss[129] = {12};
    nuv_qss[129] = {-1};
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
    kiv[130] = {6,10,3};
    nuv[130] = {-1,1,1};
    kiv_qss[130] = {13,7};
    nuv_qss[130] = {-1,1};
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
    kiv[131] = {4,19};
    nuv[131] = {-1,1};
    kiv_qss[131] = {13,1};
    nuv_qss[131] = {-1,1};
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
    kiv[132] = {22,6,16};
    nuv[132] = {-1,-1,1};
    kiv_qss[132] = {1};
    nuv_qss[132] = {1};
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
    kiv[133] = {22,7,23,6};
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
    kiv[134] = {22,3,2};
    nuv[134] = {-1,-1,1};
    kiv_qss[134] = {13};
    nuv_qss[134] = {1};
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
    kiv[135] = {22,4,5};
    nuv[135] = {-1,-1,1};
    kiv_qss[135] = {13};
    nuv_qss[135] = {1};
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
    kiv[136] = {5,22,4};
    nuv[136] = {-1,1,1};
    kiv_qss[136] = {13};
    nuv_qss[136] = {-1};
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
    kiv[137] = {23,3,22,2};
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
    kiv[138] = {23,4,22,5};
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
    kiv[139] = {23,1,21,10};
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
    kiv[140] = {24,3,23,2};
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
    kiv[141] = {24,7,20,6};
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
    kiv[142] = {24,3,20};
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
    kiv[143] = {24,19,12};
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
    kiv[144] = {24,23,3};
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
    kiv[145] = {23,3,24};
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
    kiv[146] = {24,11,20};
    nuv[146] = {-1,-1,1};
    kiv_qss[146] = {1};
    nuv_qss[146] = {1};
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
    kiv[147] = {24,23,20};
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
    kiv[148] = {20,3,21,12};
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
    kiv[149] = {20,3,24,2};
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
    kiv[150] = {20,1};
    nuv[150] = {-1,-1};
    kiv_qss[150] = {5,1};
    nuv_qss[150] = {1,1};
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
    kiv[151] = {20,1,24,4};
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
    kiv[152] = {20,1,16,12,3};
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
    kiv[153] = {20,4,24,5};
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
    kiv[154] = {25,6,20,7};
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
    kiv[155] = {25,12,21};
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
    kiv[156] = {12,21,25};
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
    kiv[157] = {25,3,20};
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
    kiv[158] = {3,20,25};
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
    kiv[159] = {26,25,6};
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
    kiv[160] = {25,6,26};
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
    kiv[161] = {27};
    nuv[161] = {-1};
    kiv_qss[161] = {8};
    nuv_qss[161] = {2.0};
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
    kiv[162] = {27};
    nuv[162] = {1};
    kiv_qss[162] = {8};
    nuv_qss[162] = {-2.0};
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
    kiv[163] = {27,4,11,24};
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
    kiv[164] = {27,4,16};
    nuv[164] = {-1,-1,1};
    kiv_qss[164] = {5};
    nuv_qss[164] = {1};
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
    kiv[165] = {27,1,21,16};
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
    kiv[166] = {27,3,21};
    nuv[166] = {-1,-1,1};
    kiv_qss[166] = {8};
    nuv_qss[166] = {1};
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
    kiv[167] = {27,1,11,23};
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
    kiv[168] = {3,28,29};
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
    kiv[169] = {24,28,20,27};
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
    kiv[170] = {28,27,15};
    nuv[170] = {-1,1,1};
    kiv_qss[170] = {5};
    nuv_qss[170] = {-1};
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
    kiv[171] = {28,27,3};
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
    kiv[172] = {27,3,28};
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
    kiv[173] = {28,12,27,13};
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
    kiv[174] = {28,7,29,6};
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
    kiv[175] = {28,6,27,7};
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
    kiv[176] = {28,21};
    nuv[176] = {-1,1};
    kiv_qss[176] = {8};
    nuv_qss[176] = {1};
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
    kiv[177] = {3,28,27,2};
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
    kiv[178] = {29,3,28,2};
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
    kiv[179] = {29,4,25,11};
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
    kiv[180] = {29,4,15};
    nuv[180] = {-1,-1,1};
    kiv_qss[180] = {11};
    nuv_qss[180] = {1};
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
    kiv[181] = {29,1};
    nuv[181] = {-1,-1};
    kiv_qss[181] = {11,5};
    nuv_qss[181] = {1,1};
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
    kiv[182] = {29,1,20,11};
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
    kiv[183] = {29,4,28,5};
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
    kiv[184] = {29,24,12};
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
    kiv[185] = {24,12,29};
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
    kiv[186] = {30,29,3};
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
    kiv[187] = {29,3,30};
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
    kiv[188] = {30,21};
    nuv[188] = {-1,1};
    kiv_qss[188] = {5};
    nuv_qss[188] = {1};
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
    kiv[189] = {31,30,6};
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
    kiv[190] = {30,6,31};
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
    kiv[191] = {32,27,12};
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
    kiv[192] = {32,24,21};
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
    kiv[193] = {33,4,32,5};
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
    kiv[194] = {33,3,32,2};
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
    kiv[195] = {33,24};
    nuv[195] = {-1,1};
    kiv_qss[195] = {5};
    nuv_qss[195] = {1};
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
    kiv[196] = {24,33};
    nuv[196] = {-1,1};
    kiv_qss[196] = {5};
    nuv_qss[196] = {-1};
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
    kiv[197] = {33,1,32,4};
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
    kiv[198] = {34,20};
    nuv[198] = {-1,1};
    kiv_qss[198] = {5};
    nuv_qss[198] = {1};
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
    kiv[199] = {34,21,25};
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
    kiv[200] = {34,33,3};
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
    kiv[201] = {35,25,24};
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
    kiv[202] = {35,4,34,11};
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
    kiv[203] = {36,35,12};
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
    kiv[204] = {36,30,20};
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
    kiv[205] = {36,29,25};
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
    kiv[206] = {36,34,21};
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
    kiv[207] = {36,33};
    nuv[207] = {-1,1};
    kiv_qss[207] = {5};
    nuv_qss[207] = {1};
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
    kiv[208] = {36,7,37,6};
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
    kiv[209] = {37,18,36};
    nuv[209] = {-1,-1,1};
    kiv_qss[209] = {6};
    nuv_qss[209] = {1};
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
    kiv[210] = {37,3,36,2};
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
    kiv[211] = {37,30,25};
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
    kiv[212] = {37,7,36,8};
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
    kiv[213] = {37,34};
    nuv[213] = {-1,1};
    kiv_qss[213] = {5};
    nuv_qss[213] = {1};
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
    kiv[214] = {37,36,14};
    nuv[214] = {-1,1,1};
    kiv_qss[214] = {4};
    nuv_qss[214] = {-1};
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
    kiv[215] = {37,1,36,4};
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
    kiv[216] = {37,4,36,5};
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
    kiv[217] = {37,12,36,13};
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
    *kk = 38;
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
    kname.resize(38);
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
    kname[18] = "CH3O2";
    kname[19] = "C2H2";
    kname[20] = "C3H6";
    kname[21] = "C2H4";
    kname[22] = "C3H3";
    kname[23] = "C3H4XA";
    kname[24] = "C3H5XA";
    kname[25] = "NXC3H7";
    kname[26] = "NXC3H7O2";
    kname[27] = "C4H6";
    kname[28] = "C4H7";
    kname[29] = "C4H8X1";
    kname[30] = "PXC4H9";
    kname[31] = "PXC4H9O2";
    kname[32] = "C5H9";
    kname[33] = "C5H10X1";
    kname[34] = "C5H11X1";
    kname[35] = "C6H12X1";
    kname[36] = "C7H15X2";
    kname[37] = "NXC7H16";
}


/* Returns the char strings of species names */
void CKSYMS(int * kname, int * plenkname )
{
    int i; /*Loop Counter */
    int lenkname = *plenkname;
    /*clear kname */
    for (i=0; i<lenkname*38; i++) {
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

    /* CH3O2  */
    kname[ 18*lenkname + 0 ] = 'C';
    kname[ 18*lenkname + 1 ] = 'H';
    kname[ 18*lenkname + 2 ] = '3';
    kname[ 18*lenkname + 3 ] = 'O';
    kname[ 18*lenkname + 4 ] = '2';
    kname[ 18*lenkname + 5 ] = ' ';

    /* C2H2  */
    kname[ 19*lenkname + 0 ] = 'C';
    kname[ 19*lenkname + 1 ] = '2';
    kname[ 19*lenkname + 2 ] = 'H';
    kname[ 19*lenkname + 3 ] = '2';
    kname[ 19*lenkname + 4 ] = ' ';

    /* C3H6  */
    kname[ 20*lenkname + 0 ] = 'C';
    kname[ 20*lenkname + 1 ] = '3';
    kname[ 20*lenkname + 2 ] = 'H';
    kname[ 20*lenkname + 3 ] = '6';
    kname[ 20*lenkname + 4 ] = ' ';

    /* C2H4  */
    kname[ 21*lenkname + 0 ] = 'C';
    kname[ 21*lenkname + 1 ] = '2';
    kname[ 21*lenkname + 2 ] = 'H';
    kname[ 21*lenkname + 3 ] = '4';
    kname[ 21*lenkname + 4 ] = ' ';

    /* C3H3  */
    kname[ 22*lenkname + 0 ] = 'C';
    kname[ 22*lenkname + 1 ] = '3';
    kname[ 22*lenkname + 2 ] = 'H';
    kname[ 22*lenkname + 3 ] = '3';
    kname[ 22*lenkname + 4 ] = ' ';

    /* C3H4XA  */
    kname[ 23*lenkname + 0 ] = 'C';
    kname[ 23*lenkname + 1 ] = '3';
    kname[ 23*lenkname + 2 ] = 'H';
    kname[ 23*lenkname + 3 ] = '4';
    kname[ 23*lenkname + 4 ] = 'X';
    kname[ 23*lenkname + 5 ] = 'A';
    kname[ 23*lenkname + 6 ] = ' ';

    /* C3H5XA  */
    kname[ 24*lenkname + 0 ] = 'C';
    kname[ 24*lenkname + 1 ] = '3';
    kname[ 24*lenkname + 2 ] = 'H';
    kname[ 24*lenkname + 3 ] = '5';
    kname[ 24*lenkname + 4 ] = 'X';
    kname[ 24*lenkname + 5 ] = 'A';
    kname[ 24*lenkname + 6 ] = ' ';

    /* NXC3H7  */
    kname[ 25*lenkname + 0 ] = 'N';
    kname[ 25*lenkname + 1 ] = 'X';
    kname[ 25*lenkname + 2 ] = 'C';
    kname[ 25*lenkname + 3 ] = '3';
    kname[ 25*lenkname + 4 ] = 'H';
    kname[ 25*lenkname + 5 ] = '7';
    kname[ 25*lenkname + 6 ] = ' ';

    /* NXC3H7O2  */
    kname[ 26*lenkname + 0 ] = 'N';
    kname[ 26*lenkname + 1 ] = 'X';
    kname[ 26*lenkname + 2 ] = 'C';
    kname[ 26*lenkname + 3 ] = '3';
    kname[ 26*lenkname + 4 ] = 'H';
    kname[ 26*lenkname + 5 ] = '7';
    kname[ 26*lenkname + 6 ] = 'O';
    kname[ 26*lenkname + 7 ] = '2';
    kname[ 26*lenkname + 8 ] = ' ';

    /* C4H6  */
    kname[ 27*lenkname + 0 ] = 'C';
    kname[ 27*lenkname + 1 ] = '4';
    kname[ 27*lenkname + 2 ] = 'H';
    kname[ 27*lenkname + 3 ] = '6';
    kname[ 27*lenkname + 4 ] = ' ';

    /* C4H7  */
    kname[ 28*lenkname + 0 ] = 'C';
    kname[ 28*lenkname + 1 ] = '4';
    kname[ 28*lenkname + 2 ] = 'H';
    kname[ 28*lenkname + 3 ] = '7';
    kname[ 28*lenkname + 4 ] = ' ';

    /* C4H8X1  */
    kname[ 29*lenkname + 0 ] = 'C';
    kname[ 29*lenkname + 1 ] = '4';
    kname[ 29*lenkname + 2 ] = 'H';
    kname[ 29*lenkname + 3 ] = '8';
    kname[ 29*lenkname + 4 ] = 'X';
    kname[ 29*lenkname + 5 ] = '1';
    kname[ 29*lenkname + 6 ] = ' ';

    /* PXC4H9  */
    kname[ 30*lenkname + 0 ] = 'P';
    kname[ 30*lenkname + 1 ] = 'X';
    kname[ 30*lenkname + 2 ] = 'C';
    kname[ 30*lenkname + 3 ] = '4';
    kname[ 30*lenkname + 4 ] = 'H';
    kname[ 30*lenkname + 5 ] = '9';
    kname[ 30*lenkname + 6 ] = ' ';

    /* PXC4H9O2  */
    kname[ 31*lenkname + 0 ] = 'P';
    kname[ 31*lenkname + 1 ] = 'X';
    kname[ 31*lenkname + 2 ] = 'C';
    kname[ 31*lenkname + 3 ] = '4';
    kname[ 31*lenkname + 4 ] = 'H';
    kname[ 31*lenkname + 5 ] = '9';
    kname[ 31*lenkname + 6 ] = 'O';
    kname[ 31*lenkname + 7 ] = '2';
    kname[ 31*lenkname + 8 ] = ' ';

    /* C5H9  */
    kname[ 32*lenkname + 0 ] = 'C';
    kname[ 32*lenkname + 1 ] = '5';
    kname[ 32*lenkname + 2 ] = 'H';
    kname[ 32*lenkname + 3 ] = '9';
    kname[ 32*lenkname + 4 ] = ' ';

    /* C5H10X1  */
    kname[ 33*lenkname + 0 ] = 'C';
    kname[ 33*lenkname + 1 ] = '5';
    kname[ 33*lenkname + 2 ] = 'H';
    kname[ 33*lenkname + 3 ] = '1';
    kname[ 33*lenkname + 4 ] = '0';
    kname[ 33*lenkname + 5 ] = 'X';
    kname[ 33*lenkname + 6 ] = '1';
    kname[ 33*lenkname + 7 ] = ' ';

    /* C5H11X1  */
    kname[ 34*lenkname + 0 ] = 'C';
    kname[ 34*lenkname + 1 ] = '5';
    kname[ 34*lenkname + 2 ] = 'H';
    kname[ 34*lenkname + 3 ] = '1';
    kname[ 34*lenkname + 4 ] = '1';
    kname[ 34*lenkname + 5 ] = 'X';
    kname[ 34*lenkname + 6 ] = '1';
    kname[ 34*lenkname + 7 ] = ' ';

    /* C6H12X1  */
    kname[ 35*lenkname + 0 ] = 'C';
    kname[ 35*lenkname + 1 ] = '6';
    kname[ 35*lenkname + 2 ] = 'H';
    kname[ 35*lenkname + 3 ] = '1';
    kname[ 35*lenkname + 4 ] = '2';
    kname[ 35*lenkname + 5 ] = 'X';
    kname[ 35*lenkname + 6 ] = '1';
    kname[ 35*lenkname + 7 ] = ' ';

    /* C7H15X2  */
    kname[ 36*lenkname + 0 ] = 'C';
    kname[ 36*lenkname + 1 ] = '7';
    kname[ 36*lenkname + 2 ] = 'H';
    kname[ 36*lenkname + 3 ] = '1';
    kname[ 36*lenkname + 4 ] = '5';
    kname[ 36*lenkname + 5 ] = 'X';
    kname[ 36*lenkname + 6 ] = '2';
    kname[ 36*lenkname + 7 ] = ' ';

    /* NXC7H16  */
    kname[ 37*lenkname + 0 ] = 'N';
    kname[ 37*lenkname + 1 ] = 'X';
    kname[ 37*lenkname + 2 ] = 'C';
    kname[ 37*lenkname + 3 ] = '7';
    kname[ 37*lenkname + 4 ] = 'H';
    kname[ 37*lenkname + 5 ] = '1';
    kname[ 37*lenkname + 6 ] = '6';
    kname[ 37*lenkname + 7 ] = ' ';

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
    XW += x[18]*molecular_weights[18]; /*CH3O2 */
    XW += x[19]*molecular_weights[19]; /*C2H2 */
    XW += x[20]*molecular_weights[20]; /*C3H6 */
    XW += x[21]*molecular_weights[21]; /*C2H4 */
    XW += x[22]*molecular_weights[22]; /*C3H3 */
    XW += x[23]*molecular_weights[23]; /*C3H4XA */
    XW += x[24]*molecular_weights[24]; /*C3H5XA */
    XW += x[25]*molecular_weights[25]; /*NXC3H7 */
    XW += x[26]*molecular_weights[26]; /*NXC3H7O2 */
    XW += x[27]*molecular_weights[27]; /*C4H6 */
    XW += x[28]*molecular_weights[28]; /*C4H7 */
    XW += x[29]*molecular_weights[29]; /*C4H8X1 */
    XW += x[30]*molecular_weights[30]; /*PXC4H9 */
    XW += x[31]*molecular_weights[31]; /*PXC4H9O2 */
    XW += x[32]*molecular_weights[32]; /*C5H9 */
    XW += x[33]*molecular_weights[33]; /*C5H10X1 */
    XW += x[34]*molecular_weights[34]; /*C5H11X1 */
    XW += x[35]*molecular_weights[35]; /*C6H12X1 */
    XW += x[36]*molecular_weights[36]; /*C7H15X2 */
    XW += x[37]*molecular_weights[37]; /*NXC7H16 */
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
    YOW += y[18]*imw[18]; /*CH3O2 */
    YOW += y[19]*imw[19]; /*C2H2 */
    YOW += y[20]*imw[20]; /*C3H6 */
    YOW += y[21]*imw[21]; /*C2H4 */
    YOW += y[22]*imw[22]; /*C3H3 */
    YOW += y[23]*imw[23]; /*C3H4XA */
    YOW += y[24]*imw[24]; /*C3H5XA */
    YOW += y[25]*imw[25]; /*NXC3H7 */
    YOW += y[26]*imw[26]; /*NXC3H7O2 */
    YOW += y[27]*imw[27]; /*C4H6 */
    YOW += y[28]*imw[28]; /*C4H7 */
    YOW += y[29]*imw[29]; /*C4H8X1 */
    YOW += y[30]*imw[30]; /*PXC4H9 */
    YOW += y[31]*imw[31]; /*PXC4H9O2 */
    YOW += y[32]*imw[32]; /*C5H9 */
    YOW += y[33]*imw[33]; /*C5H10X1 */
    YOW += y[34]*imw[34]; /*C5H11X1 */
    YOW += y[35]*imw[35]; /*C6H12X1 */
    YOW += y[36]*imw[36]; /*C7H15X2 */
    YOW += y[37]*imw[37]; /*NXC7H16 */
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
    W += c[18]*47.033860; /*CH3O2 */
    W += c[19]*26.038240; /*C2H2 */
    W += c[20]*42.081270; /*C3H6 */
    W += c[21]*28.054180; /*C2H4 */
    W += c[22]*39.057360; /*C3H3 */
    W += c[23]*40.065330; /*C3H4XA */
    W += c[24]*41.073300; /*C3H5XA */
    W += c[25]*43.089240; /*NXC3H7 */
    W += c[26]*75.088040; /*NXC3H7O2 */
    W += c[27]*54.092420; /*C4H6 */
    W += c[28]*55.100390; /*C4H7 */
    W += c[29]*56.108360; /*C4H8X1 */
    W += c[30]*57.116330; /*PXC4H9 */
    W += c[31]*89.115130; /*PXC4H9O2 */
    W += c[32]*69.127480; /*C5H9 */
    W += c[33]*70.135450; /*C5H10X1 */
    W += c[34]*71.143420; /*C5H11X1 */
    W += c[35]*84.162540; /*C6H12X1 */
    W += c[36]*99.197600; /*C7H15X2 */
    W += c[37]*100.205570; /*NXC7H16 */

    for (id = 0; id < 38; ++id) {
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
    XW += x[18]*47.033860; /*CH3O2 */
    XW += x[19]*26.038240; /*C2H2 */
    XW += x[20]*42.081270; /*C3H6 */
    XW += x[21]*28.054180; /*C2H4 */
    XW += x[22]*39.057360; /*C3H3 */
    XW += x[23]*40.065330; /*C3H4XA */
    XW += x[24]*41.073300; /*C3H5XA */
    XW += x[25]*43.089240; /*NXC3H7 */
    XW += x[26]*75.088040; /*NXC3H7O2 */
    XW += x[27]*54.092420; /*C4H6 */
    XW += x[28]*55.100390; /*C4H7 */
    XW += x[29]*56.108360; /*C4H8X1 */
    XW += x[30]*57.116330; /*PXC4H9 */
    XW += x[31]*89.115130; /*PXC4H9O2 */
    XW += x[32]*69.127480; /*C5H9 */
    XW += x[33]*70.135450; /*C5H10X1 */
    XW += x[34]*71.143420; /*C5H11X1 */
    XW += x[35]*84.162540; /*C6H12X1 */
    XW += x[36]*99.197600; /*C7H15X2 */
    XW += x[37]*100.205570; /*NXC7H16 */
    *rho = *P * XW / (8.31446e+07 * (*T)); /*rho = P*W/(R*T) */

    return;
}


/*Compute rho = P*W(y)/RT */
AMREX_GPU_HOST_DEVICE void CKRHOY(double *  P, double *  T, double *  y,  double *  rho)
{
    double YOW = 0;
    double tmp[38];

    for (int i = 0; i < 38; i++)
    {
        tmp[i] = y[i]*imw[i];
    }
    for (int i = 0; i < 38; i++)
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
    W += c[18]*47.033860; /*CH3O2 */
    W += c[19]*26.038240; /*C2H2 */
    W += c[20]*42.081270; /*C3H6 */
    W += c[21]*28.054180; /*C2H4 */
    W += c[22]*39.057360; /*C3H3 */
    W += c[23]*40.065330; /*C3H4XA */
    W += c[24]*41.073300; /*C3H5XA */
    W += c[25]*43.089240; /*NXC3H7 */
    W += c[26]*75.088040; /*NXC3H7O2 */
    W += c[27]*54.092420; /*C4H6 */
    W += c[28]*55.100390; /*C4H7 */
    W += c[29]*56.108360; /*C4H8X1 */
    W += c[30]*57.116330; /*PXC4H9 */
    W += c[31]*89.115130; /*PXC4H9O2 */
    W += c[32]*69.127480; /*C5H9 */
    W += c[33]*70.135450; /*C5H10X1 */
    W += c[34]*71.143420; /*C5H11X1 */
    W += c[35]*84.162540; /*C6H12X1 */
    W += c[36]*99.197600; /*C7H15X2 */
    W += c[37]*100.205570; /*NXC7H16 */

    for (id = 0; id < 38; ++id) {
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
    double tmp[38];

    for (int i = 0; i < 38; i++)
    {
        tmp[i] = y[i]*imw[i];
    }
    for (int i = 0; i < 38; i++)
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
    XW += x[18]*molecular_weights[18]; /*CH3O2 */
    XW += x[19]*molecular_weights[19]; /*C2H2 */
    XW += x[20]*molecular_weights[20]; /*C3H6 */
    XW += x[21]*molecular_weights[21]; /*C2H4 */
    XW += x[22]*molecular_weights[22]; /*C3H3 */
    XW += x[23]*molecular_weights[23]; /*C3H4XA */
    XW += x[24]*molecular_weights[24]; /*C3H5XA */
    XW += x[25]*molecular_weights[25]; /*NXC3H7 */
    XW += x[26]*molecular_weights[26]; /*NXC3H7O2 */
    XW += x[27]*molecular_weights[27]; /*C4H6 */
    XW += x[28]*molecular_weights[28]; /*C4H7 */
    XW += x[29]*molecular_weights[29]; /*C4H8X1 */
    XW += x[30]*molecular_weights[30]; /*PXC4H9 */
    XW += x[31]*molecular_weights[31]; /*PXC4H9O2 */
    XW += x[32]*molecular_weights[32]; /*C5H9 */
    XW += x[33]*molecular_weights[33]; /*C5H10X1 */
    XW += x[34]*molecular_weights[34]; /*C5H11X1 */
    XW += x[35]*molecular_weights[35]; /*C6H12X1 */
    XW += x[36]*molecular_weights[36]; /*C7H15X2 */
    XW += x[37]*molecular_weights[37]; /*NXC7H16 */
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
    W += c[18]*molecular_weights[18]; /*CH3O2 */
    W += c[19]*molecular_weights[19]; /*C2H2 */
    W += c[20]*molecular_weights[20]; /*C3H6 */
    W += c[21]*molecular_weights[21]; /*C2H4 */
    W += c[22]*molecular_weights[22]; /*C3H3 */
    W += c[23]*molecular_weights[23]; /*C3H4XA */
    W += c[24]*molecular_weights[24]; /*C3H5XA */
    W += c[25]*molecular_weights[25]; /*NXC3H7 */
    W += c[26]*molecular_weights[26]; /*NXC3H7O2 */
    W += c[27]*molecular_weights[27]; /*C4H6 */
    W += c[28]*molecular_weights[28]; /*C4H7 */
    W += c[29]*molecular_weights[29]; /*C4H8X1 */
    W += c[30]*molecular_weights[30]; /*PXC4H9 */
    W += c[31]*molecular_weights[31]; /*PXC4H9O2 */
    W += c[32]*molecular_weights[32]; /*C5H9 */
    W += c[33]*molecular_weights[33]; /*C5H10X1 */
    W += c[34]*molecular_weights[34]; /*C5H11X1 */
    W += c[35]*molecular_weights[35]; /*C6H12X1 */
    W += c[36]*molecular_weights[36]; /*C7H15X2 */
    W += c[37]*molecular_weights[37]; /*NXC7H16 */

    for (id = 0; id < 38; ++id) {
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
    double tmp[38];

    for (int i = 0; i < 38; i++)
    {
        tmp[i] = y[i]*imw[i];
    }
    for (int i = 0; i < 38; i++)
    {
        YOW += tmp[i];
    }

    double YOWINV = 1.0/YOW;

    for (int i = 0; i < 38; i++)
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
    for (int i = 0; i < 38; i++)
    {
        c[i] = y[i]*imw[i];
    }
    for (int i = 0; i < 38; i++)
    {
        YOW += c[i];
    }

    /*PW/RT (see Eq. 7) */
    PWORT = (*P)/(YOW * 8.31446e+07 * (*T)); 
    /*Now compute conversion */

    for (int i = 0; i < 38; i++)
    {
        c[i] = PWORT * y[i] * imw[i];
    }
    return;
}


/*convert y[species] (mass fracs) to c[species] (molar conc) */
AMREX_GPU_HOST_DEVICE void CKYTCR(double *  rho, double *  T, double *  y,  double *  c)
{
    for (int i = 0; i < 38; i++)
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
    XW += x[18]*47.033860; /*CH3O2 */
    XW += x[19]*26.038240; /*C2H2 */
    XW += x[20]*42.081270; /*C3H6 */
    XW += x[21]*28.054180; /*C2H4 */
    XW += x[22]*39.057360; /*C3H3 */
    XW += x[23]*40.065330; /*C3H4XA */
    XW += x[24]*41.073300; /*C3H5XA */
    XW += x[25]*43.089240; /*NXC3H7 */
    XW += x[26]*75.088040; /*NXC3H7O2 */
    XW += x[27]*54.092420; /*C4H6 */
    XW += x[28]*55.100390; /*C4H7 */
    XW += x[29]*56.108360; /*C4H8X1 */
    XW += x[30]*57.116330; /*PXC4H9 */
    XW += x[31]*89.115130; /*PXC4H9O2 */
    XW += x[32]*69.127480; /*C5H9 */
    XW += x[33]*70.135450; /*C5H10X1 */
    XW += x[34]*71.143420; /*C5H11X1 */
    XW += x[35]*84.162540; /*C6H12X1 */
    XW += x[36]*99.197600; /*C7H15X2 */
    XW += x[37]*100.205570; /*NXC7H16 */

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
    y[18] = x[18]*47.033860*XWinv; 
    y[19] = x[19]*26.038240*XWinv; 
    y[20] = x[20]*42.081270*XWinv; 
    y[21] = x[21]*28.054180*XWinv; 
    y[22] = x[22]*39.057360*XWinv; 
    y[23] = x[23]*40.065330*XWinv; 
    y[24] = x[24]*41.073300*XWinv; 
    y[25] = x[25]*43.089240*XWinv; 
    y[26] = x[26]*75.088040*XWinv; 
    y[27] = x[27]*54.092420*XWinv; 
    y[28] = x[28]*55.100390*XWinv; 
    y[29] = x[29]*56.108360*XWinv; 
    y[30] = x[30]*57.116330*XWinv; 
    y[31] = x[31]*89.115130*XWinv; 
    y[32] = x[32]*69.127480*XWinv; 
    y[33] = x[33]*70.135450*XWinv; 
    y[34] = x[34]*71.143420*XWinv; 
    y[35] = x[35]*84.162540*XWinv; 
    y[36] = x[36]*99.197600*XWinv; 
    y[37] = x[37]*100.205570*XWinv; 

    return;
}


/*convert x[species] (mole fracs) to c[species] (molar conc) */
void CKXTCP(double *  P, double *  T, double *  x,  double *  c)
{
    int id; /*loop counter */
    double PORT = (*P)/(8.31446e+07 * (*T)); /*P/RT */

    /*Compute conversion, see Eq 10 */
    for (id = 0; id < 38; ++id) {
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
    XW += x[18]*47.033860; /*CH3O2 */
    XW += x[19]*26.038240; /*C2H2 */
    XW += x[20]*42.081270; /*C3H6 */
    XW += x[21]*28.054180; /*C2H4 */
    XW += x[22]*39.057360; /*C3H3 */
    XW += x[23]*40.065330; /*C3H4XA */
    XW += x[24]*41.073300; /*C3H5XA */
    XW += x[25]*43.089240; /*NXC3H7 */
    XW += x[26]*75.088040; /*NXC3H7O2 */
    XW += x[27]*54.092420; /*C4H6 */
    XW += x[28]*55.100390; /*C4H7 */
    XW += x[29]*56.108360; /*C4H8X1 */
    XW += x[30]*57.116330; /*PXC4H9 */
    XW += x[31]*89.115130; /*PXC4H9O2 */
    XW += x[32]*69.127480; /*C5H9 */
    XW += x[33]*70.135450; /*C5H10X1 */
    XW += x[34]*71.143420; /*C5H11X1 */
    XW += x[35]*84.162540; /*C6H12X1 */
    XW += x[36]*99.197600; /*C7H15X2 */
    XW += x[37]*100.205570; /*NXC7H16 */
    ROW = (*rho) / XW;

    /*Compute conversion, see Eq 11 */
    for (id = 0; id < 38; ++id) {
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
    for (id = 0; id < 38; ++id) {
        sumC += c[id];
    }

    /* See Eq 13  */
    double sumCinv = 1.0/sumC;
    for (id = 0; id < 38; ++id) {
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
    CW += c[18]*47.033860; /*CH3O2 */
    CW += c[19]*26.038240; /*C2H2 */
    CW += c[20]*42.081270; /*C3H6 */
    CW += c[21]*28.054180; /*C2H4 */
    CW += c[22]*39.057360; /*C3H3 */
    CW += c[23]*40.065330; /*C3H4XA */
    CW += c[24]*41.073300; /*C3H5XA */
    CW += c[25]*43.089240; /*NXC3H7 */
    CW += c[26]*75.088040; /*NXC3H7O2 */
    CW += c[27]*54.092420; /*C4H6 */
    CW += c[28]*55.100390; /*C4H7 */
    CW += c[29]*56.108360; /*C4H8X1 */
    CW += c[30]*57.116330; /*PXC4H9 */
    CW += c[31]*89.115130; /*PXC4H9O2 */
    CW += c[32]*69.127480; /*C5H9 */
    CW += c[33]*70.135450; /*C5H10X1 */
    CW += c[34]*71.143420; /*C5H11X1 */
    CW += c[35]*84.162540; /*C6H12X1 */
    CW += c[36]*99.197600; /*C7H15X2 */
    CW += c[37]*100.205570; /*NXC7H16 */
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
    y[18] = c[18]*47.033860*CWinv; 
    y[19] = c[19]*26.038240*CWinv; 
    y[20] = c[20]*42.081270*CWinv; 
    y[21] = c[21]*28.054180*CWinv; 
    y[22] = c[22]*39.057360*CWinv; 
    y[23] = c[23]*40.065330*CWinv; 
    y[24] = c[24]*41.073300*CWinv; 
    y[25] = c[25]*43.089240*CWinv; 
    y[26] = c[26]*75.088040*CWinv; 
    y[27] = c[27]*54.092420*CWinv; 
    y[28] = c[28]*55.100390*CWinv; 
    y[29] = c[29]*56.108360*CWinv; 
    y[30] = c[30]*57.116330*CWinv; 
    y[31] = c[31]*89.115130*CWinv; 
    y[32] = c[32]*69.127480*CWinv; 
    y[33] = c[33]*70.135450*CWinv; 
    y[34] = c[34]*71.143420*CWinv; 
    y[35] = c[35]*84.162540*CWinv; 
    y[36] = c[36]*99.197600*CWinv; 
    y[37] = c[37]*100.205570*CWinv; 

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
    cvms[18] *= 1.767761059405552e+06; /*CH3O2 */
    cvms[19] *= 3.193173815954242e+06; /*C2H2 */
    cvms[20] *= 1.975810762876985e+06; /*C3H6 */
    cvms[21] *= 2.963716144315478e+06; /*C2H4 */
    cvms[22] *= 2.128782543969495e+06; /*C3H3 */
    cvms[23] *= 2.075226291198210e+06; /*C3H4XA */
    cvms[24] *= 2.024298660724422e+06; /*C3H5XA */
    cvms[25] *= 1.929591382478141e+06; /*NXC3H7 */
    cvms[26] *= 1.107295198829699e+06; /*NXC3H7O2 */
    cvms[27] *= 1.537084607816260e+06; /*C4H6 */
    cvms[28] *= 1.508966201174482e+06; /*C4H7 */
    cvms[29] *= 1.481858072157739e+06; /*C4H8X1 */
    cvms[30] *= 1.455706733635239e+06; /*PXC4H9 */
    cvms[31] *= 9.330023552850385e+05; /*PXC4H9O2 */
    cvms[32] *= 1.202772416722444e+06; /*C5H9 */
    cvms[33] *= 1.185486457726191e+06; /*C5H10X1 */
    cvms[34] *= 1.168690318535887e+06; /*C5H11X1 */
    cvms[35] *= 9.879053814384926e+05; /*C6H12X1 */
    cvms[36] *= 8.381717519529947e+05; /*C7H15X2 */
    cvms[37] *= 8.297405641376262e+05; /*NXC7H16 */
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
    cpms[18] *= 1.767761059405552e+06; /*CH3O2 */
    cpms[19] *= 3.193173815954242e+06; /*C2H2 */
    cpms[20] *= 1.975810762876985e+06; /*C3H6 */
    cpms[21] *= 2.963716144315478e+06; /*C2H4 */
    cpms[22] *= 2.128782543969495e+06; /*C3H3 */
    cpms[23] *= 2.075226291198210e+06; /*C3H4XA */
    cpms[24] *= 2.024298660724422e+06; /*C3H5XA */
    cpms[25] *= 1.929591382478141e+06; /*NXC3H7 */
    cpms[26] *= 1.107295198829699e+06; /*NXC3H7O2 */
    cpms[27] *= 1.537084607816260e+06; /*C4H6 */
    cpms[28] *= 1.508966201174482e+06; /*C4H7 */
    cpms[29] *= 1.481858072157739e+06; /*C4H8X1 */
    cpms[30] *= 1.455706733635239e+06; /*PXC4H9 */
    cpms[31] *= 9.330023552850385e+05; /*PXC4H9O2 */
    cpms[32] *= 1.202772416722444e+06; /*C5H9 */
    cpms[33] *= 1.185486457726191e+06; /*C5H10X1 */
    cpms[34] *= 1.168690318535887e+06; /*C5H11X1 */
    cpms[35] *= 9.879053814384926e+05; /*C6H12X1 */
    cpms[36] *= 8.381717519529947e+05; /*C7H15X2 */
    cpms[37] *= 8.297405641376262e+05; /*NXC7H16 */
}


/*Returns internal energy in mass units (Eq 30.) */
AMREX_GPU_HOST_DEVICE void CKUMS(double *  T,  double *  ums)
{
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double RT = 8.31446e+07*tT; /*R*T */
    speciesInternalEnergy(ums, tc);
    for (int i = 0; i < 38; i++)
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
    for (int i = 0; i < 38; i++)
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
    for (int i = 0; i < 38; i++)
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
    for (int i = 0; i < 38; i++)
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
    sms[18] *= 1.767761059405552e+06; /*CH3O2 */
    sms[19] *= 3.193173815954242e+06; /*C2H2 */
    sms[20] *= 1.975810762876985e+06; /*C3H6 */
    sms[21] *= 2.963716144315478e+06; /*C2H4 */
    sms[22] *= 2.128782543969495e+06; /*C3H3 */
    sms[23] *= 2.075226291198210e+06; /*C3H4XA */
    sms[24] *= 2.024298660724422e+06; /*C3H5XA */
    sms[25] *= 1.929591382478141e+06; /*NXC3H7 */
    sms[26] *= 1.107295198829699e+06; /*NXC3H7O2 */
    sms[27] *= 1.537084607816260e+06; /*C4H6 */
    sms[28] *= 1.508966201174482e+06; /*C4H7 */
    sms[29] *= 1.481858072157739e+06; /*C4H8X1 */
    sms[30] *= 1.455706733635239e+06; /*PXC4H9 */
    sms[31] *= 9.330023552850385e+05; /*PXC4H9O2 */
    sms[32] *= 1.202772416722444e+06; /*C5H9 */
    sms[33] *= 1.185486457726191e+06; /*C5H10X1 */
    sms[34] *= 1.168690318535887e+06; /*C5H11X1 */
    sms[35] *= 9.879053814384926e+05; /*C6H12X1 */
    sms[36] *= 8.381717519529947e+05; /*C7H15X2 */
    sms[37] *= 8.297405641376262e+05; /*NXC7H16 */
}


/*Returns the mean specific heat at CP (Eq. 33) */
void CKCPBL(double *  T, double *  x,  double *  cpbl)
{
    int id; /*loop counter */
    double result = 0; 
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double cpor[38]; /* temporary storage */
    cp_R(cpor, tc);

    /*perform dot product */
    for (id = 0; id < 38; ++id) {
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
    double cpor[38], tresult[38]; /* temporary storage */
    cp_R(cpor, tc);
    for (int i = 0; i < 38; i++)
    {
        tresult[i] = cpor[i]*y[i]*imw[i];

    }
    for (int i = 0; i < 38; i++)
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
    double cvor[38]; /* temporary storage */
    cv_R(cvor, tc);

    /*perform dot product */
    for (id = 0; id < 38; ++id) {
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
    double cvor[38]; /* temporary storage */
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
    result += cvor[18]*y[18]*imw[18]; /*CH3O2 */
    result += cvor[19]*y[19]*imw[19]; /*C2H2 */
    result += cvor[20]*y[20]*imw[20]; /*C3H6 */
    result += cvor[21]*y[21]*imw[21]; /*C2H4 */
    result += cvor[22]*y[22]*imw[22]; /*C3H3 */
    result += cvor[23]*y[23]*imw[23]; /*C3H4XA */
    result += cvor[24]*y[24]*imw[24]; /*C3H5XA */
    result += cvor[25]*y[25]*imw[25]; /*NXC3H7 */
    result += cvor[26]*y[26]*imw[26]; /*NXC3H7O2 */
    result += cvor[27]*y[27]*imw[27]; /*C4H6 */
    result += cvor[28]*y[28]*imw[28]; /*C4H7 */
    result += cvor[29]*y[29]*imw[29]; /*C4H8X1 */
    result += cvor[30]*y[30]*imw[30]; /*PXC4H9 */
    result += cvor[31]*y[31]*imw[31]; /*PXC4H9O2 */
    result += cvor[32]*y[32]*imw[32]; /*C5H9 */
    result += cvor[33]*y[33]*imw[33]; /*C5H10X1 */
    result += cvor[34]*y[34]*imw[34]; /*C5H11X1 */
    result += cvor[35]*y[35]*imw[35]; /*C6H12X1 */
    result += cvor[36]*y[36]*imw[36]; /*C7H15X2 */
    result += cvor[37]*y[37]*imw[37]; /*NXC7H16 */

    *cvbs = result * 8.31446e+07;
}


/*Returns the mean enthalpy of the mixture in molar units */
void CKHBML(double *  T, double *  x,  double *  hbml)
{
    int id; /*loop counter */
    double result = 0; 
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double hml[38]; /* temporary storage */
    double RT = 8.31446e+07*tT; /*R*T */
    speciesEnthalpy(hml, tc);

    /*perform dot product */
    for (id = 0; id < 38; ++id) {
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
    double hml[38], tmp[38]; /* temporary storage */
    double RT = 8.31446e+07*tT; /*R*T */
    speciesEnthalpy(hml, tc);
    int id;
    for (id = 0; id < 38; ++id) {
        tmp[id] = y[id]*hml[id]*imw[id];
    }
    for (id = 0; id < 38; ++id) {
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
    double uml[38]; /* temporary energy array */
    double RT = 8.31446e+07*tT; /*R*T */
    speciesInternalEnergy(uml, tc);

    /*perform dot product */
    for (id = 0; id < 38; ++id) {
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
    double ums[38]; /* temporary energy array */
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
    result += y[18]*ums[18]*imw[18]; /*CH3O2 */
    result += y[19]*ums[19]*imw[19]; /*C2H2 */
    result += y[20]*ums[20]*imw[20]; /*C3H6 */
    result += y[21]*ums[21]*imw[21]; /*C2H4 */
    result += y[22]*ums[22]*imw[22]; /*C3H3 */
    result += y[23]*ums[23]*imw[23]; /*C3H4XA */
    result += y[24]*ums[24]*imw[24]; /*C3H5XA */
    result += y[25]*ums[25]*imw[25]; /*NXC3H7 */
    result += y[26]*ums[26]*imw[26]; /*NXC3H7O2 */
    result += y[27]*ums[27]*imw[27]; /*C4H6 */
    result += y[28]*ums[28]*imw[28]; /*C4H7 */
    result += y[29]*ums[29]*imw[29]; /*C4H8X1 */
    result += y[30]*ums[30]*imw[30]; /*PXC4H9 */
    result += y[31]*ums[31]*imw[31]; /*PXC4H9O2 */
    result += y[32]*ums[32]*imw[32]; /*C5H9 */
    result += y[33]*ums[33]*imw[33]; /*C5H10X1 */
    result += y[34]*ums[34]*imw[34]; /*C5H11X1 */
    result += y[35]*ums[35]*imw[35]; /*C6H12X1 */
    result += y[36]*ums[36]*imw[36]; /*C7H15X2 */
    result += y[37]*ums[37]*imw[37]; /*NXC7H16 */

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
    double sor[38]; /* temporary storage */
    speciesEntropy(sor, tc);

    /*Compute Eq 42 */
    for (id = 0; id < 38; ++id) {
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
    double sor[38]; /* temporary storage */
    double x[38]; /* need a ytx conversion */
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
    YOW += y[18]*imw[18]; /*CH3O2 */
    YOW += y[19]*imw[19]; /*C2H2 */
    YOW += y[20]*imw[20]; /*C3H6 */
    YOW += y[21]*imw[21]; /*C2H4 */
    YOW += y[22]*imw[22]; /*C3H3 */
    YOW += y[23]*imw[23]; /*C3H4XA */
    YOW += y[24]*imw[24]; /*C3H5XA */
    YOW += y[25]*imw[25]; /*NXC3H7 */
    YOW += y[26]*imw[26]; /*NXC3H7O2 */
    YOW += y[27]*imw[27]; /*C4H6 */
    YOW += y[28]*imw[28]; /*C4H7 */
    YOW += y[29]*imw[29]; /*C4H8X1 */
    YOW += y[30]*imw[30]; /*PXC4H9 */
    YOW += y[31]*imw[31]; /*PXC4H9O2 */
    YOW += y[32]*imw[32]; /*C5H9 */
    YOW += y[33]*imw[33]; /*C5H10X1 */
    YOW += y[34]*imw[34]; /*C5H11X1 */
    YOW += y[35]*imw[35]; /*C6H12X1 */
    YOW += y[36]*imw[36]; /*C7H15X2 */
    YOW += y[37]*imw[37]; /*NXC7H16 */
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
    x[18] = y[18]/(47.033860*YOW); 
    x[19] = y[19]/(26.038240*YOW); 
    x[20] = y[20]/(42.081270*YOW); 
    x[21] = y[21]/(28.054180*YOW); 
    x[22] = y[22]/(39.057360*YOW); 
    x[23] = y[23]/(40.065330*YOW); 
    x[24] = y[24]/(41.073300*YOW); 
    x[25] = y[25]/(43.089240*YOW); 
    x[26] = y[26]/(75.088040*YOW); 
    x[27] = y[27]/(54.092420*YOW); 
    x[28] = y[28]/(55.100390*YOW); 
    x[29] = y[29]/(56.108360*YOW); 
    x[30] = y[30]/(57.116330*YOW); 
    x[31] = y[31]/(89.115130*YOW); 
    x[32] = y[32]/(69.127480*YOW); 
    x[33] = y[33]/(70.135450*YOW); 
    x[34] = y[34]/(71.143420*YOW); 
    x[35] = y[35]/(84.162540*YOW); 
    x[36] = y[36]/(99.197600*YOW); 
    x[37] = y[37]/(100.205570*YOW); 
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
    double gort[38]; /* temporary storage */
    /*Compute g/RT */
    gibbs(gort, tc);

    /*Compute Eq 44 */
    for (id = 0; id < 38; ++id) {
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
    double gort[38]; /* temporary storage */
    double x[38]; /* need a ytx conversion */
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
    YOW += y[18]*imw[18]; /*CH3O2 */
    YOW += y[19]*imw[19]; /*C2H2 */
    YOW += y[20]*imw[20]; /*C3H6 */
    YOW += y[21]*imw[21]; /*C2H4 */
    YOW += y[22]*imw[22]; /*C3H3 */
    YOW += y[23]*imw[23]; /*C3H4XA */
    YOW += y[24]*imw[24]; /*C3H5XA */
    YOW += y[25]*imw[25]; /*NXC3H7 */
    YOW += y[26]*imw[26]; /*NXC3H7O2 */
    YOW += y[27]*imw[27]; /*C4H6 */
    YOW += y[28]*imw[28]; /*C4H7 */
    YOW += y[29]*imw[29]; /*C4H8X1 */
    YOW += y[30]*imw[30]; /*PXC4H9 */
    YOW += y[31]*imw[31]; /*PXC4H9O2 */
    YOW += y[32]*imw[32]; /*C5H9 */
    YOW += y[33]*imw[33]; /*C5H10X1 */
    YOW += y[34]*imw[34]; /*C5H11X1 */
    YOW += y[35]*imw[35]; /*C6H12X1 */
    YOW += y[36]*imw[36]; /*C7H15X2 */
    YOW += y[37]*imw[37]; /*NXC7H16 */
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
    x[18] = y[18]/(47.033860*YOW); 
    x[19] = y[19]/(26.038240*YOW); 
    x[20] = y[20]/(42.081270*YOW); 
    x[21] = y[21]/(28.054180*YOW); 
    x[22] = y[22]/(39.057360*YOW); 
    x[23] = y[23]/(40.065330*YOW); 
    x[24] = y[24]/(41.073300*YOW); 
    x[25] = y[25]/(43.089240*YOW); 
    x[26] = y[26]/(75.088040*YOW); 
    x[27] = y[27]/(54.092420*YOW); 
    x[28] = y[28]/(55.100390*YOW); 
    x[29] = y[29]/(56.108360*YOW); 
    x[30] = y[30]/(57.116330*YOW); 
    x[31] = y[31]/(89.115130*YOW); 
    x[32] = y[32]/(69.127480*YOW); 
    x[33] = y[33]/(70.135450*YOW); 
    x[34] = y[34]/(71.143420*YOW); 
    x[35] = y[35]/(84.162540*YOW); 
    x[36] = y[36]/(99.197600*YOW); 
    x[37] = y[37]/(100.205570*YOW); 
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
    double aort[38]; /* temporary storage */
    /*Compute g/RT */
    helmholtz(aort, tc);

    /*Compute Eq 44 */
    for (id = 0; id < 38; ++id) {
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
    double aort[38]; /* temporary storage */
    double x[38]; /* need a ytx conversion */
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
    YOW += y[18]*imw[18]; /*CH3O2 */
    YOW += y[19]*imw[19]; /*C2H2 */
    YOW += y[20]*imw[20]; /*C3H6 */
    YOW += y[21]*imw[21]; /*C2H4 */
    YOW += y[22]*imw[22]; /*C3H3 */
    YOW += y[23]*imw[23]; /*C3H4XA */
    YOW += y[24]*imw[24]; /*C3H5XA */
    YOW += y[25]*imw[25]; /*NXC3H7 */
    YOW += y[26]*imw[26]; /*NXC3H7O2 */
    YOW += y[27]*imw[27]; /*C4H6 */
    YOW += y[28]*imw[28]; /*C4H7 */
    YOW += y[29]*imw[29]; /*C4H8X1 */
    YOW += y[30]*imw[30]; /*PXC4H9 */
    YOW += y[31]*imw[31]; /*PXC4H9O2 */
    YOW += y[32]*imw[32]; /*C5H9 */
    YOW += y[33]*imw[33]; /*C5H10X1 */
    YOW += y[34]*imw[34]; /*C5H11X1 */
    YOW += y[35]*imw[35]; /*C6H12X1 */
    YOW += y[36]*imw[36]; /*C7H15X2 */
    YOW += y[37]*imw[37]; /*NXC7H16 */
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
    x[18] = y[18]/(47.033860*YOW); 
    x[19] = y[19]/(26.038240*YOW); 
    x[20] = y[20]/(42.081270*YOW); 
    x[21] = y[21]/(28.054180*YOW); 
    x[22] = y[22]/(39.057360*YOW); 
    x[23] = y[23]/(40.065330*YOW); 
    x[24] = y[24]/(41.073300*YOW); 
    x[25] = y[25]/(43.089240*YOW); 
    x[26] = y[26]/(75.088040*YOW); 
    x[27] = y[27]/(54.092420*YOW); 
    x[28] = y[28]/(55.100390*YOW); 
    x[29] = y[29]/(56.108360*YOW); 
    x[30] = y[30]/(57.116330*YOW); 
    x[31] = y[31]/(89.115130*YOW); 
    x[32] = y[32]/(69.127480*YOW); 
    x[33] = y[33]/(70.135450*YOW); 
    x[34] = y[34]/(71.143420*YOW); 
    x[35] = y[35]/(84.162540*YOW); 
    x[36] = y[36]/(99.197600*YOW); 
    x[37] = y[37]/(100.205570*YOW); 
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
    /*Scale by RT/W */
    *abms = result * RT * YOW;
}


/*compute the production rate for each species */
AMREX_GPU_HOST_DEVICE void CKWC(double *  T, double *  C,  double *  wdot)
{
    int id; /*loop counter */

    /*convert to SI */
    for (id = 0; id < 38; ++id) {
        C[id] *= 1.0e6;
    }

    /*convert to chemkin units */
    productionRate(wdot, C, *T);

    /*convert to chemkin units */
    for (id = 0; id < 38; ++id) {
        C[id] *= 1.0e-6;
        wdot[id] *= 1.0e-6;
    }
}


/*Returns the molar production rate of species */
/*Given P, T, and mass fractions */
void CKWYP(double *  P, double *  T, double *  y,  double *  wdot)
{
    int id; /*loop counter */
    double c[38]; /*temporary storage */
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
    YOW += y[18]*imw[18]; /*CH3O2 */
    YOW += y[19]*imw[19]; /*C2H2 */
    YOW += y[20]*imw[20]; /*C3H6 */
    YOW += y[21]*imw[21]; /*C2H4 */
    YOW += y[22]*imw[22]; /*C3H3 */
    YOW += y[23]*imw[23]; /*C3H4XA */
    YOW += y[24]*imw[24]; /*C3H5XA */
    YOW += y[25]*imw[25]; /*NXC3H7 */
    YOW += y[26]*imw[26]; /*NXC3H7O2 */
    YOW += y[27]*imw[27]; /*C4H6 */
    YOW += y[28]*imw[28]; /*C4H7 */
    YOW += y[29]*imw[29]; /*C4H8X1 */
    YOW += y[30]*imw[30]; /*PXC4H9 */
    YOW += y[31]*imw[31]; /*PXC4H9O2 */
    YOW += y[32]*imw[32]; /*C5H9 */
    YOW += y[33]*imw[33]; /*C5H10X1 */
    YOW += y[34]*imw[34]; /*C5H11X1 */
    YOW += y[35]*imw[35]; /*C6H12X1 */
    YOW += y[36]*imw[36]; /*C7H15X2 */
    YOW += y[37]*imw[37]; /*NXC7H16 */
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

    /*convert to chemkin units */
    productionRate(wdot, c, *T);

    /*convert to chemkin units */
    for (id = 0; id < 38; ++id) {
        wdot[id] *= 1.0e-6;
    }
}


/*Returns the molar production rate of species */
/*Given P, T, and mole fractions */
void CKWXP(double *  P, double *  T, double *  x,  double *  wdot)
{
    int id; /*loop counter */
    double c[38]; /*temporary storage */
    double PORT = 1e6 * (*P)/(8.31446e+07 * (*T)); /*1e6 * P/RT so c goes to SI units */

    /*Compute conversion, see Eq 10 */
    for (id = 0; id < 38; ++id) {
        c[id] = x[id]*PORT;
    }

    /*convert to chemkin units */
    productionRate(wdot, c, *T);

    /*convert to chemkin units */
    for (id = 0; id < 38; ++id) {
        wdot[id] *= 1.0e-6;
    }
}


/*Returns the molar production rate of species */
/*Given rho, T, and mass fractions */
AMREX_GPU_HOST_DEVICE void CKWYR(double *  rho, double *  T, double *  y,  double *  wdot)
{
    int id; /*loop counter */
    double c[38]; /*temporary storage */
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

    /*call productionRate */
    productionRate(wdot, c, *T);

    /*convert to chemkin units */
    for (id = 0; id < 38; ++id) {
        wdot[id] *= 1.0e-6;
    }
}


/*Returns the molar production rate of species */
/*Given rho, T, and mole fractions */
void CKWXR(double *  rho, double *  T, double *  x,  double *  wdot)
{
    int id; /*loop counter */
    double c[38]; /*temporary storage */
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
    XW += x[18]*47.033860; /*CH3O2 */
    XW += x[19]*26.038240; /*C2H2 */
    XW += x[20]*42.081270; /*C3H6 */
    XW += x[21]*28.054180; /*C2H4 */
    XW += x[22]*39.057360; /*C3H3 */
    XW += x[23]*40.065330; /*C3H4XA */
    XW += x[24]*41.073300; /*C3H5XA */
    XW += x[25]*43.089240; /*NXC3H7 */
    XW += x[26]*75.088040; /*NXC3H7O2 */
    XW += x[27]*54.092420; /*C4H6 */
    XW += x[28]*55.100390; /*C4H7 */
    XW += x[29]*56.108360; /*C4H8X1 */
    XW += x[30]*57.116330; /*PXC4H9 */
    XW += x[31]*89.115130; /*PXC4H9O2 */
    XW += x[32]*69.127480; /*C5H9 */
    XW += x[33]*70.135450; /*C5H10X1 */
    XW += x[34]*71.143420; /*C5H11X1 */
    XW += x[35]*84.162540; /*C6H12X1 */
    XW += x[36]*99.197600; /*C7H15X2 */
    XW += x[37]*100.205570; /*NXC7H16 */
    /*Extra 1e6 factor to take c to SI */
    ROW = 1e6*(*rho) / XW;

    /*Compute conversion, see Eq 11 */
    for (id = 0; id < 38; ++id) {
        c[id] = x[id]*ROW;
    }

    /*convert to chemkin units */
    productionRate(wdot, c, *T);

    /*convert to chemkin units */
    for (id = 0; id < 38; ++id) {
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
    for (id = 0; id < kd * 38; ++ id) {
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

    /*CH3O2 */
    ncf[ 18 * kd + 3 ] = 1; /*C */
    ncf[ 18 * kd + 2 ] = 3; /*H */
    ncf[ 18 * kd + 1 ] = 2; /*O */

    /*C2H2 */
    ncf[ 19 * kd + 3 ] = 2; /*C */
    ncf[ 19 * kd + 2 ] = 2; /*H */

    /*C3H6 */
    ncf[ 20 * kd + 3 ] = 3; /*C */
    ncf[ 20 * kd + 2 ] = 6; /*H */

    /*C2H4 */
    ncf[ 21 * kd + 3 ] = 2; /*C */
    ncf[ 21 * kd + 2 ] = 4; /*H */

    /*C3H3 */
    ncf[ 22 * kd + 3 ] = 3; /*C */
    ncf[ 22 * kd + 2 ] = 3; /*H */

    /*C3H4XA */
    ncf[ 23 * kd + 2 ] = 4; /*H */
    ncf[ 23 * kd + 3 ] = 3; /*C */

    /*C3H5XA */
    ncf[ 24 * kd + 3 ] = 3; /*C */
    ncf[ 24 * kd + 2 ] = 5; /*H */

    /*NXC3H7 */
    ncf[ 25 * kd + 3 ] = 3; /*C */
    ncf[ 25 * kd + 2 ] = 7; /*H */

    /*NXC3H7O2 */
    ncf[ 26 * kd + 3 ] = 3; /*C */
    ncf[ 26 * kd + 2 ] = 7; /*H */
    ncf[ 26 * kd + 1 ] = 2; /*O */

    /*C4H6 */
    ncf[ 27 * kd + 3 ] = 4; /*C */
    ncf[ 27 * kd + 2 ] = 6; /*H */

    /*C4H7 */
    ncf[ 28 * kd + 3 ] = 4; /*C */
    ncf[ 28 * kd + 2 ] = 7; /*H */

    /*C4H8X1 */
    ncf[ 29 * kd + 3 ] = 4; /*C */
    ncf[ 29 * kd + 2 ] = 8; /*H */

    /*PXC4H9 */
    ncf[ 30 * kd + 3 ] = 4; /*C */
    ncf[ 30 * kd + 2 ] = 9; /*H */

    /*PXC4H9O2 */
    ncf[ 31 * kd + 3 ] = 4; /*C */
    ncf[ 31 * kd + 2 ] = 9; /*H */
    ncf[ 31 * kd + 1 ] = 2; /*O */

    /*C5H9 */
    ncf[ 32 * kd + 3 ] = 5; /*C */
    ncf[ 32 * kd + 2 ] = 9; /*H */

    /*C5H10X1 */
    ncf[ 33 * kd + 3 ] = 5; /*C */
    ncf[ 33 * kd + 2 ] = 10; /*H */

    /*C5H11X1 */
    ncf[ 34 * kd + 3 ] = 5; /*C */
    ncf[ 34 * kd + 2 ] = 11; /*H */

    /*C6H12X1 */
    ncf[ 35 * kd + 3 ] = 6; /*C */
    ncf[ 35 * kd + 2 ] = 12; /*H */

    /*C7H15X2 */
    ncf[ 36 * kd + 3 ] = 7; /*C */
    ncf[ 36 * kd + 2 ] = 15; /*H */

    /*NXC7H16 */
    ncf[ 37 * kd + 3 ] = 7; /*C */
    ncf[ 37 * kd + 2 ] = 16; /*H */


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
    double sc_qss[14];
    /* Fill sc_qss here*/
    comp_qss_sc(sc, sc_qss, tc, invT);
    comp_qfqr(q_f, q_r, sc, sc_qss, tc, invT);

    for (int i = 0; i < 38; ++i) {
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
    wdot[19] += qdot;

    qdot = q_f[9]-q_r[9];
    wdot[3] -= qdot;
    wdot[21] -= qdot;

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
    wdot[18] -= qdot;

    qdot = q_f[16]-q_r[16];
    wdot[6] -= qdot;
    wdot[12] -= qdot;
    wdot[18] += qdot;

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
    wdot[6] += qdot;
    wdot[18] -= 2.000000 * qdot;

    qdot = q_f[89]-q_r[89];
    wdot[12] -= qdot;
    wdot[18] -= qdot;

    qdot = q_f[90]-q_r[90];
    wdot[6] += qdot;
    wdot[11] += qdot;
    wdot[14] += qdot;
    wdot[18] -= 2.000000 * qdot;

    qdot = q_f[91]-q_r[91];
    wdot[6] += qdot;
    wdot[7] -= qdot;
    wdot[18] -= qdot;

    qdot = q_f[92]-q_r[92];
    wdot[4] += qdot;

    qdot = q_f[93]-q_r[93];
    wdot[1] -= qdot;
    wdot[10] += qdot;
    wdot[19] -= qdot;

    qdot = q_f[94]-q_r[94];
    wdot[1] -= qdot;
    wdot[3] += qdot;
    wdot[19] -= qdot;

    qdot = q_f[95]-q_r[95];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[19] += qdot;

    qdot = q_f[96]-q_r[96];
    wdot[1] += qdot;
    wdot[6] -= qdot;

    qdot = q_f[97]-q_r[97];
    wdot[12] -= qdot;
    wdot[20] += qdot;

    qdot = q_f[98]-q_r[98];
    wdot[6] -= qdot;
    wdot[7] += qdot;
    wdot[19] += qdot;

    qdot = q_f[99]-q_r[99];
    wdot[6] -= qdot;
    wdot[11] += qdot;

    qdot = q_f[100]-q_r[100];
    wdot[12] -= qdot;
    wdot[13] += qdot;
    wdot[21] -= qdot;

    qdot = q_f[101]-q_r[101];
    wdot[1] -= qdot;
    wdot[12] += qdot;
    wdot[21] -= qdot;

    qdot = q_f[102]-q_r[102];
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[21] -= qdot;

    qdot = q_f[103]-q_r[103];
    wdot[1] -= qdot;
    wdot[3] += qdot;
    wdot[21] -= qdot;

    qdot = q_f[104]-q_r[104];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[21] -= qdot;

    qdot = q_f[105]-q_r[105];
    wdot[2] -= qdot;
    wdot[3] += qdot;
    wdot[21] += qdot;

    qdot = q_f[106]-q_r[106];
    wdot[3] -= qdot;
    wdot[15] += qdot;

    qdot = q_f[107]-q_r[107];
    wdot[18] -= qdot;

    qdot = q_f[108]-q_r[108];
    wdot[4] += qdot;
    wdot[7] -= qdot;

    qdot = q_f[109]-q_r[109];
    wdot[6] -= qdot;
    wdot[7] += qdot;
    wdot[21] += qdot;

    qdot = q_f[110]-q_r[110];
    wdot[1] -= qdot;
    wdot[4] += qdot;
    wdot[15] -= qdot;

    qdot = q_f[111]-q_r[111];
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[15] -= qdot;

    qdot = q_f[112]-q_r[112];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[15] -= qdot;

    qdot = q_f[113]-q_r[113];
    wdot[1] -= qdot;
    wdot[3] += qdot;
    wdot[10] += 2.000000 * qdot;

    qdot = q_f[114]-q_r[114];
    wdot[4] -= qdot;

    qdot = q_f[115]-q_r[115];
    wdot[6] -= qdot;
    wdot[9] += qdot;

    qdot = q_f[116]-q_r[116];
    wdot[3] -= qdot;
    wdot[10] += qdot;

    qdot = q_f[117]-q_r[117];
    wdot[3] += qdot;
    wdot[10] -= qdot;

    qdot = q_f[118]-q_r[118];
    wdot[1] -= qdot;
    wdot[4] += qdot;
    wdot[16] -= qdot;

    qdot = q_f[119]-q_r[119];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[16] -= qdot;

    qdot = q_f[120]-q_r[120];
    wdot[2] -= qdot;
    wdot[3] += qdot;
    wdot[16] += qdot;

    qdot = q_f[121]-q_r[121];
    wdot[3] -= qdot;
    wdot[10] += qdot;
    wdot[12] += qdot;
    wdot[16] -= qdot;

    qdot = q_f[122]-q_r[122];
    wdot[1] -= qdot;
    wdot[9] += qdot;
    wdot[16] -= qdot;

    qdot = q_f[123]-q_r[123];
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[16] -= qdot;

    qdot = q_f[124]-q_r[124];
    wdot[4] += qdot;
    wdot[6] -= qdot;
    wdot[10] += qdot;
    wdot[11] += qdot;

    qdot = q_f[125]-q_r[125];
    wdot[3] += qdot;
    wdot[16] += qdot;

    qdot = q_f[126]-q_r[126];
    wdot[3] -= qdot;
    wdot[16] -= qdot;

    qdot = q_f[127]-q_r[127];
    wdot[6] += qdot;

    qdot = q_f[128]-q_r[128];
    wdot[6] -= qdot;

    qdot = q_f[129]-q_r[129];
    wdot[7] += qdot;
    wdot[21] += qdot;

    qdot = q_f[130]-q_r[130];
    wdot[3] += qdot;
    wdot[6] -= qdot;
    wdot[10] += qdot;

    qdot = q_f[131]-q_r[131];
    wdot[4] -= qdot;
    wdot[19] += qdot;

    qdot = q_f[132]-q_r[132];
    wdot[6] -= qdot;
    wdot[16] += qdot;
    wdot[22] -= qdot;

    qdot = q_f[133]-q_r[133];
    wdot[6] += qdot;
    wdot[7] -= qdot;
    wdot[22] -= qdot;
    wdot[23] += qdot;

    qdot = q_f[134]-q_r[134];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[22] -= qdot;

    qdot = q_f[135]-q_r[135];
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[22] -= qdot;

    qdot = q_f[136]-q_r[136];
    wdot[4] += qdot;
    wdot[5] -= qdot;
    wdot[22] += qdot;

    qdot = q_f[137]-q_r[137];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[22] += qdot;
    wdot[23] -= qdot;

    qdot = q_f[138]-q_r[138];
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[22] += qdot;
    wdot[23] -= qdot;

    qdot = q_f[139]-q_r[139];
    wdot[1] -= qdot;
    wdot[10] += qdot;
    wdot[21] += qdot;
    wdot[23] -= qdot;

    qdot = q_f[140]-q_r[140];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[23] += qdot;
    wdot[24] -= qdot;

    qdot = q_f[141]-q_r[141];
    wdot[6] += qdot;
    wdot[7] -= qdot;
    wdot[20] += qdot;
    wdot[24] -= qdot;

    qdot = q_f[142]-q_r[142];
    wdot[3] -= qdot;
    wdot[20] += qdot;
    wdot[24] -= qdot;

    qdot = q_f[143]-q_r[143];
    wdot[12] += qdot;
    wdot[19] += qdot;
    wdot[24] -= qdot;

    qdot = q_f[144]-q_r[144];
    wdot[3] += qdot;
    wdot[23] += qdot;
    wdot[24] -= qdot;

    qdot = q_f[145]-q_r[145];
    wdot[3] -= qdot;
    wdot[23] -= qdot;
    wdot[24] += qdot;

    qdot = q_f[146]-q_r[146];
    wdot[11] -= qdot;
    wdot[20] += qdot;
    wdot[24] -= qdot;

    qdot = q_f[147]-q_r[147];
    wdot[20] += qdot;
    wdot[23] += qdot;
    wdot[24] -= 2.000000 * qdot;

    qdot = q_f[148]-q_r[148];
    wdot[3] -= qdot;
    wdot[12] += qdot;
    wdot[20] -= qdot;
    wdot[21] += qdot;

    qdot = q_f[149]-q_r[149];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[20] -= qdot;
    wdot[24] += qdot;

    qdot = q_f[150]-q_r[150];
    wdot[1] -= qdot;
    wdot[20] -= qdot;

    qdot = q_f[151]-q_r[151];
    wdot[1] -= qdot;
    wdot[4] += qdot;
    wdot[20] -= qdot;
    wdot[24] += qdot;

    qdot = q_f[152]-q_r[152];
    wdot[1] -= qdot;
    wdot[3] += qdot;
    wdot[12] += qdot;
    wdot[16] += qdot;
    wdot[20] -= qdot;

    qdot = q_f[153]-q_r[153];
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[20] -= qdot;
    wdot[24] += qdot;

    qdot = q_f[154]-q_r[154];
    wdot[6] -= qdot;
    wdot[7] += qdot;
    wdot[20] += qdot;
    wdot[25] -= qdot;

    qdot = q_f[155]-q_r[155];
    wdot[12] += qdot;
    wdot[21] += qdot;
    wdot[25] -= qdot;

    qdot = q_f[156]-q_r[156];
    wdot[12] -= qdot;
    wdot[21] -= qdot;
    wdot[25] += qdot;

    qdot = q_f[157]-q_r[157];
    wdot[3] += qdot;
    wdot[20] += qdot;
    wdot[25] -= qdot;

    qdot = q_f[158]-q_r[158];
    wdot[3] -= qdot;
    wdot[20] -= qdot;
    wdot[25] += qdot;

    qdot = q_f[159]-q_r[159];
    wdot[6] += qdot;
    wdot[25] += qdot;
    wdot[26] -= qdot;

    qdot = q_f[160]-q_r[160];
    wdot[6] -= qdot;
    wdot[25] -= qdot;
    wdot[26] += qdot;

    qdot = q_f[161]-q_r[161];
    wdot[27] -= qdot;

    qdot = q_f[162]-q_r[162];
    wdot[27] += qdot;

    qdot = q_f[163]-q_r[163];
    wdot[4] -= qdot;
    wdot[11] += qdot;
    wdot[24] += qdot;
    wdot[27] -= qdot;

    qdot = q_f[164]-q_r[164];
    wdot[4] -= qdot;
    wdot[16] += qdot;
    wdot[27] -= qdot;

    qdot = q_f[165]-q_r[165];
    wdot[1] -= qdot;
    wdot[16] += qdot;
    wdot[21] += qdot;
    wdot[27] -= qdot;

    qdot = q_f[166]-q_r[166];
    wdot[3] -= qdot;
    wdot[21] += qdot;
    wdot[27] -= qdot;

    qdot = q_f[167]-q_r[167];
    wdot[1] -= qdot;
    wdot[11] += qdot;
    wdot[23] += qdot;
    wdot[27] -= qdot;

    qdot = q_f[168]-q_r[168];
    wdot[3] -= qdot;
    wdot[28] -= qdot;
    wdot[29] += qdot;

    qdot = q_f[169]-q_r[169];
    wdot[20] += qdot;
    wdot[24] -= qdot;
    wdot[27] += qdot;
    wdot[28] -= qdot;

    qdot = q_f[170]-q_r[170];
    wdot[15] += qdot;
    wdot[27] += qdot;
    wdot[28] -= qdot;

    qdot = q_f[171]-q_r[171];
    wdot[3] += qdot;
    wdot[27] += qdot;
    wdot[28] -= qdot;

    qdot = q_f[172]-q_r[172];
    wdot[3] -= qdot;
    wdot[27] -= qdot;
    wdot[28] += qdot;

    qdot = q_f[173]-q_r[173];
    wdot[12] -= qdot;
    wdot[13] += qdot;
    wdot[27] += qdot;
    wdot[28] -= qdot;

    qdot = q_f[174]-q_r[174];
    wdot[6] += qdot;
    wdot[7] -= qdot;
    wdot[28] -= qdot;
    wdot[29] += qdot;

    qdot = q_f[175]-q_r[175];
    wdot[6] -= qdot;
    wdot[7] += qdot;
    wdot[27] += qdot;
    wdot[28] -= qdot;

    qdot = q_f[176]-q_r[176];
    wdot[21] += qdot;
    wdot[28] -= qdot;

    qdot = q_f[177]-q_r[177];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[27] += qdot;
    wdot[28] -= qdot;

    qdot = q_f[178]-q_r[178];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[28] += qdot;
    wdot[29] -= qdot;

    qdot = q_f[179]-q_r[179];
    wdot[4] -= qdot;
    wdot[11] += qdot;
    wdot[25] += qdot;
    wdot[29] -= qdot;

    qdot = q_f[180]-q_r[180];
    wdot[4] -= qdot;
    wdot[15] += qdot;
    wdot[29] -= qdot;

    qdot = q_f[181]-q_r[181];
    wdot[1] -= qdot;
    wdot[29] -= qdot;

    qdot = q_f[182]-q_r[182];
    wdot[1] -= qdot;
    wdot[11] += qdot;
    wdot[20] += qdot;
    wdot[29] -= qdot;

    qdot = q_f[183]-q_r[183];
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[28] += qdot;
    wdot[29] -= qdot;

    qdot = q_f[184]-q_r[184];
    wdot[12] += qdot;
    wdot[24] += qdot;
    wdot[29] -= qdot;

    qdot = q_f[185]-q_r[185];
    wdot[12] -= qdot;
    wdot[24] -= qdot;
    wdot[29] += qdot;

    qdot = q_f[186]-q_r[186];
    wdot[3] += qdot;
    wdot[29] += qdot;
    wdot[30] -= qdot;

    qdot = q_f[187]-q_r[187];
    wdot[3] -= qdot;
    wdot[29] -= qdot;
    wdot[30] += qdot;

    qdot = q_f[188]-q_r[188];
    wdot[21] += qdot;
    wdot[30] -= qdot;

    qdot = q_f[189]-q_r[189];
    wdot[6] += qdot;
    wdot[30] += qdot;
    wdot[31] -= qdot;

    qdot = q_f[190]-q_r[190];
    wdot[6] -= qdot;
    wdot[30] -= qdot;
    wdot[31] += qdot;

    qdot = q_f[191]-q_r[191];
    wdot[12] += qdot;
    wdot[27] += qdot;
    wdot[32] -= qdot;

    qdot = q_f[192]-q_r[192];
    wdot[21] += qdot;
    wdot[24] += qdot;
    wdot[32] -= qdot;

    qdot = q_f[193]-q_r[193];
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[32] += qdot;
    wdot[33] -= qdot;

    qdot = q_f[194]-q_r[194];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[32] += qdot;
    wdot[33] -= qdot;

    qdot = q_f[195]-q_r[195];
    wdot[24] += qdot;
    wdot[33] -= qdot;

    qdot = q_f[196]-q_r[196];
    wdot[24] -= qdot;
    wdot[33] += qdot;

    qdot = q_f[197]-q_r[197];
    wdot[1] -= qdot;
    wdot[4] += qdot;
    wdot[32] += qdot;
    wdot[33] -= qdot;

    qdot = q_f[198]-q_r[198];
    wdot[20] += qdot;
    wdot[34] -= qdot;

    qdot = q_f[199]-q_r[199];
    wdot[21] += qdot;
    wdot[25] += qdot;
    wdot[34] -= qdot;

    qdot = q_f[200]-q_r[200];
    wdot[3] += qdot;
    wdot[33] += qdot;
    wdot[34] -= qdot;

    qdot = q_f[201]-q_r[201];
    wdot[24] += qdot;
    wdot[25] += qdot;
    wdot[35] -= qdot;

    qdot = q_f[202]-q_r[202];
    wdot[4] -= qdot;
    wdot[11] += qdot;
    wdot[34] += qdot;
    wdot[35] -= qdot;

    qdot = q_f[203]-q_r[203];
    wdot[12] += qdot;
    wdot[35] += qdot;
    wdot[36] -= qdot;

    qdot = q_f[204]-q_r[204];
    wdot[20] += qdot;
    wdot[30] += qdot;
    wdot[36] -= qdot;

    qdot = q_f[205]-q_r[205];
    wdot[25] += qdot;
    wdot[29] += qdot;
    wdot[36] -= qdot;

    qdot = q_f[206]-q_r[206];
    wdot[21] += qdot;
    wdot[34] += qdot;
    wdot[36] -= qdot;

    qdot = q_f[207]-q_r[207];
    wdot[33] += qdot;
    wdot[36] -= qdot;

    qdot = q_f[208]-q_r[208];
    wdot[6] += qdot;
    wdot[7] -= qdot;
    wdot[36] -= qdot;
    wdot[37] += qdot;

    qdot = q_f[209]-q_r[209];
    wdot[18] -= qdot;
    wdot[36] += qdot;
    wdot[37] -= qdot;

    qdot = q_f[210]-q_r[210];
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[36] += qdot;
    wdot[37] -= qdot;

    qdot = q_f[211]-q_r[211];
    wdot[25] += qdot;
    wdot[30] += qdot;
    wdot[37] -= qdot;

    qdot = q_f[212]-q_r[212];
    wdot[7] -= qdot;
    wdot[8] += qdot;
    wdot[36] += qdot;
    wdot[37] -= qdot;

    qdot = q_f[213]-q_r[213];
    wdot[34] += qdot;
    wdot[37] -= qdot;

    qdot = q_f[214]-q_r[214];
    wdot[14] += qdot;
    wdot[36] += qdot;
    wdot[37] -= qdot;

    qdot = q_f[215]-q_r[215];
    wdot[1] -= qdot;
    wdot[4] += qdot;
    wdot[36] += qdot;
    wdot[37] -= qdot;

    qdot = q_f[216]-q_r[216];
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[36] += qdot;
    wdot[37] -= qdot;

    qdot = q_f[217]-q_r[217];
    wdot[12] -= qdot;
    wdot[13] += qdot;
    wdot[36] += qdot;
    wdot[37] -= qdot;

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
    double g_RT[38], g_RT_qss[14];
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
    Kc[8] = -g_RT[3] - g_RT[19] + g_RT_qss[8];
    Kc[9] = g_RT[3] + g_RT[21] - g_RT_qss[5];
    Kc[10] = -g_RT[10] - g_RT[12] + g_RT_qss[11];
    Kc[11] = g_RT[3] + g_RT[4] - g_RT[5];
    Kc[12] = -g_RT_qss[2] + g_RT_qss[3];
    Kc[13] = g_RT_qss[2] - g_RT_qss[3];
    Kc[14] = -g_RT[3] - g_RT[10] + g_RT_qss[1];
    Kc[15] = -g_RT[6] - g_RT[12] + g_RT[18];
    Kc[16] = g_RT[6] + g_RT[12] - g_RT[18];
    Kc[17] = -g_RT[11] - g_RT[12] + g_RT_qss[10];
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
    Kc[88] = -g_RT[6] + 2.000000*g_RT[18] - 2.000000*g_RT_qss[4];
    Kc[89] = g_RT[12] + g_RT[18] - 2.000000*g_RT_qss[4];
    Kc[90] = -g_RT[6] - g_RT[11] - g_RT[14] + 2.000000*g_RT[18];
    Kc[91] = -g_RT[6] + g_RT[7] + g_RT[18] - g_RT_qss[6];
    Kc[92] = -g_RT[4] - g_RT_qss[4] + g_RT_qss[6];
    Kc[93] = g_RT[1] - g_RT[10] + g_RT[19] - g_RT_qss[2];
    Kc[94] = g_RT[1] - g_RT[3] + g_RT[19] - g_RT_qss[7];
    Kc[95] = -g_RT[2] + g_RT[3] - g_RT[19] + g_RT_qss[8];
    Kc[96] = -g_RT[1] + g_RT[6] + g_RT_qss[8] - g_RT_qss[9];
    Kc[97] = g_RT[12] - g_RT[20] + g_RT_qss[8];
    Kc[98] = g_RT[6] - g_RT[7] - g_RT[19] + g_RT_qss[8];
    Kc[99] = g_RT[6] - g_RT[11] - g_RT_qss[1] + g_RT_qss[8];
    Kc[100] = g_RT[12] - g_RT[13] + g_RT[21] - g_RT_qss[8];
    Kc[101] = g_RT[1] - g_RT[12] + g_RT[21] - g_RT_qss[1];
    Kc[102] = g_RT[4] - g_RT[5] + g_RT[21] - g_RT_qss[8];
    Kc[103] = g_RT[1] - g_RT[3] + g_RT[21] - g_RT_qss[9];
    Kc[104] = -g_RT[2] + g_RT[3] + g_RT[21] - g_RT_qss[8];
    Kc[105] = g_RT[2] - g_RT[3] - g_RT[21] + g_RT_qss[8];
    Kc[106] = g_RT[3] - g_RT[15] + g_RT_qss[5];
    Kc[107] = g_RT[18] - g_RT_qss[4] + g_RT_qss[5] - g_RT_qss[10];
    Kc[108] = -g_RT[4] + g_RT[7] + g_RT_qss[5] - g_RT_qss[10];
    Kc[109] = g_RT[6] - g_RT[7] - g_RT[21] + g_RT_qss[5];
    Kc[110] = g_RT[1] - g_RT[4] + g_RT[15] - g_RT_qss[5];
    Kc[111] = g_RT[4] - g_RT[5] + g_RT[15] - g_RT_qss[5];
    Kc[112] = -g_RT[2] + g_RT[3] + g_RT[15] - g_RT_qss[5];
    Kc[113] = g_RT[1] - g_RT[3] - 2.000000*g_RT[10] + g_RT_qss[7];
    Kc[114] = g_RT[4] - 2.000000*g_RT_qss[1] + g_RT_qss[7];
    Kc[115] = g_RT[6] - g_RT[9] - g_RT_qss[1] + g_RT_qss[7];
    Kc[116] = g_RT[3] - g_RT[10] - g_RT_qss[3] + g_RT_qss[7];
    Kc[117] = -g_RT[3] + g_RT[10] + g_RT_qss[3] - g_RT_qss[7];
    Kc[118] = g_RT[1] - g_RT[4] + g_RT[16] - g_RT_qss[7];
    Kc[119] = -g_RT[2] + g_RT[3] + g_RT[16] - g_RT_qss[7];
    Kc[120] = g_RT[2] - g_RT[3] - g_RT[16] + g_RT_qss[7];
    Kc[121] = g_RT[3] - g_RT[10] - g_RT[12] + g_RT[16];
    Kc[122] = g_RT[1] - g_RT[9] + g_RT[16] - g_RT_qss[2];
    Kc[123] = g_RT[4] - g_RT[5] + g_RT[16] - g_RT_qss[7];
    Kc[124] = -g_RT[4] + g_RT[6] - g_RT[10] - g_RT[11] + g_RT_qss[9];
    Kc[125] = -g_RT[3] - g_RT[16] + g_RT_qss[9];
    Kc[126] = g_RT[3] + g_RT[16] - g_RT_qss[9];
    Kc[127] = -g_RT[6] - g_RT_qss[5] + g_RT_qss[12];
    Kc[128] = g_RT[6] + g_RT_qss[5] - g_RT_qss[12];
    Kc[129] = -g_RT[7] - g_RT[21] + g_RT_qss[12];
    Kc[130] = -g_RT[3] + g_RT[6] - g_RT[10] - g_RT_qss[7] + g_RT_qss[13];
    Kc[131] = g_RT[4] - g_RT[19] - g_RT_qss[1] + g_RT_qss[13];
    Kc[132] = g_RT[6] - g_RT[16] + g_RT[22] - g_RT_qss[1];
    Kc[133] = -g_RT[6] + g_RT[7] + g_RT[22] - g_RT[23];
    Kc[134] = -g_RT[2] + g_RT[3] + g_RT[22] - g_RT_qss[13];
    Kc[135] = g_RT[4] - g_RT[5] + g_RT[22] - g_RT_qss[13];
    Kc[136] = -g_RT[4] + g_RT[5] - g_RT[22] + g_RT_qss[13];
    Kc[137] = -g_RT[2] + g_RT[3] - g_RT[22] + g_RT[23];
    Kc[138] = g_RT[4] - g_RT[5] - g_RT[22] + g_RT[23];
    Kc[139] = g_RT[1] - g_RT[10] - g_RT[21] + g_RT[23];
    Kc[140] = -g_RT[2] + g_RT[3] - g_RT[23] + g_RT[24];
    Kc[141] = -g_RT[6] + g_RT[7] - g_RT[20] + g_RT[24];
    Kc[142] = g_RT[3] - g_RT[20] + g_RT[24];
    Kc[143] = -g_RT[12] - g_RT[19] + g_RT[24];
    Kc[144] = -g_RT[3] - g_RT[23] + g_RT[24];
    Kc[145] = g_RT[3] + g_RT[23] - g_RT[24];
    Kc[146] = g_RT[11] - g_RT[20] + g_RT[24] - g_RT_qss[1];
    Kc[147] = -g_RT[20] - g_RT[23] + 2.000000*g_RT[24];
    Kc[148] = g_RT[3] - g_RT[12] + g_RT[20] - g_RT[21];
    Kc[149] = -g_RT[2] + g_RT[3] + g_RT[20] - g_RT[24];
    Kc[150] = g_RT[1] + g_RT[20] - g_RT_qss[1] - g_RT_qss[5];
    Kc[151] = g_RT[1] - g_RT[4] + g_RT[20] - g_RT[24];
    Kc[152] = g_RT[1] - g_RT[3] - g_RT[12] - g_RT[16] + g_RT[20];
    Kc[153] = g_RT[4] - g_RT[5] + g_RT[20] - g_RT[24];
    Kc[154] = g_RT[6] - g_RT[7] - g_RT[20] + g_RT[25];
    Kc[155] = -g_RT[12] - g_RT[21] + g_RT[25];
    Kc[156] = g_RT[12] + g_RT[21] - g_RT[25];
    Kc[157] = -g_RT[3] - g_RT[20] + g_RT[25];
    Kc[158] = g_RT[3] + g_RT[20] - g_RT[25];
    Kc[159] = -g_RT[6] - g_RT[25] + g_RT[26];
    Kc[160] = g_RT[6] + g_RT[25] - g_RT[26];
    Kc[161] = g_RT[27] - 2.000000*g_RT_qss[8];
    Kc[162] = -g_RT[27] + 2.000000*g_RT_qss[8];
    Kc[163] = g_RT[4] - g_RT[11] - g_RT[24] + g_RT[27];
    Kc[164] = g_RT[4] - g_RT[16] + g_RT[27] - g_RT_qss[5];
    Kc[165] = g_RT[1] - g_RT[16] - g_RT[21] + g_RT[27];
    Kc[166] = g_RT[3] - g_RT[21] + g_RT[27] - g_RT_qss[8];
    Kc[167] = g_RT[1] - g_RT[11] - g_RT[23] + g_RT[27];
    Kc[168] = g_RT[3] + g_RT[28] - g_RT[29];
    Kc[169] = -g_RT[20] + g_RT[24] - g_RT[27] + g_RT[28];
    Kc[170] = -g_RT[15] - g_RT[27] + g_RT[28] + g_RT_qss[5];
    Kc[171] = -g_RT[3] - g_RT[27] + g_RT[28];
    Kc[172] = g_RT[3] + g_RT[27] - g_RT[28];
    Kc[173] = g_RT[12] - g_RT[13] - g_RT[27] + g_RT[28];
    Kc[174] = -g_RT[6] + g_RT[7] + g_RT[28] - g_RT[29];
    Kc[175] = g_RT[6] - g_RT[7] - g_RT[27] + g_RT[28];
    Kc[176] = -g_RT[21] + g_RT[28] - g_RT_qss[8];
    Kc[177] = -g_RT[2] + g_RT[3] - g_RT[27] + g_RT[28];
    Kc[178] = -g_RT[2] + g_RT[3] - g_RT[28] + g_RT[29];
    Kc[179] = g_RT[4] - g_RT[11] - g_RT[25] + g_RT[29];
    Kc[180] = g_RT[4] - g_RT[15] + g_RT[29] - g_RT_qss[11];
    Kc[181] = g_RT[1] + g_RT[29] - g_RT_qss[5] - g_RT_qss[11];
    Kc[182] = g_RT[1] - g_RT[11] - g_RT[20] + g_RT[29];
    Kc[183] = g_RT[4] - g_RT[5] - g_RT[28] + g_RT[29];
    Kc[184] = -g_RT[12] - g_RT[24] + g_RT[29];
    Kc[185] = g_RT[12] + g_RT[24] - g_RT[29];
    Kc[186] = -g_RT[3] - g_RT[29] + g_RT[30];
    Kc[187] = g_RT[3] + g_RT[29] - g_RT[30];
    Kc[188] = -g_RT[21] + g_RT[30] - g_RT_qss[5];
    Kc[189] = -g_RT[6] - g_RT[30] + g_RT[31];
    Kc[190] = g_RT[6] + g_RT[30] - g_RT[31];
    Kc[191] = -g_RT[12] - g_RT[27] + g_RT[32];
    Kc[192] = -g_RT[21] - g_RT[24] + g_RT[32];
    Kc[193] = g_RT[4] - g_RT[5] - g_RT[32] + g_RT[33];
    Kc[194] = -g_RT[2] + g_RT[3] - g_RT[32] + g_RT[33];
    Kc[195] = -g_RT[24] + g_RT[33] - g_RT_qss[5];
    Kc[196] = g_RT[24] - g_RT[33] + g_RT_qss[5];
    Kc[197] = g_RT[1] - g_RT[4] - g_RT[32] + g_RT[33];
    Kc[198] = -g_RT[20] + g_RT[34] - g_RT_qss[5];
    Kc[199] = -g_RT[21] - g_RT[25] + g_RT[34];
    Kc[200] = -g_RT[3] - g_RT[33] + g_RT[34];
    Kc[201] = -g_RT[24] - g_RT[25] + g_RT[35];
    Kc[202] = g_RT[4] - g_RT[11] - g_RT[34] + g_RT[35];
    Kc[203] = -g_RT[12] - g_RT[35] + g_RT[36];
    Kc[204] = -g_RT[20] - g_RT[30] + g_RT[36];
    Kc[205] = -g_RT[25] - g_RT[29] + g_RT[36];
    Kc[206] = -g_RT[21] - g_RT[34] + g_RT[36];
    Kc[207] = -g_RT[33] + g_RT[36] - g_RT_qss[5];
    Kc[208] = -g_RT[6] + g_RT[7] + g_RT[36] - g_RT[37];
    Kc[209] = g_RT[18] - g_RT[36] + g_RT[37] - g_RT_qss[6];
    Kc[210] = -g_RT[2] + g_RT[3] - g_RT[36] + g_RT[37];
    Kc[211] = -g_RT[25] - g_RT[30] + g_RT[37];
    Kc[212] = g_RT[7] - g_RT[8] - g_RT[36] + g_RT[37];
    Kc[213] = -g_RT[34] + g_RT[37] - g_RT_qss[5];
    Kc[214] = -g_RT[14] - g_RT[36] + g_RT[37] + g_RT_qss[4];
    Kc[215] = g_RT[1] - g_RT[4] - g_RT[36] + g_RT[37];
    Kc[216] = g_RT[4] - g_RT[5] - g_RT[36] + g_RT[37];
    Kc[217] = g_RT[12] - g_RT[13] - g_RT[36] + g_RT[37];

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
    qf[8] = qss_sc[8];
    qr[8] = 0.0;

    /*reaction 10: H + C2H4 (+M) <=> C2H5 (+M) */
    qf[9] = sc[3]*sc[21];
    qr[9] = qss_sc[5];

    /*reaction 11: CH3CO (+M) => CH3 + CO (+M) */
    qf[10] = qss_sc[11];
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
    qf[15] = sc[18];
    qr[15] = 0.0;

    /*reaction 17: CH3 + O2 + M => CH3O2 + M */
    qf[16] = sc[6]*sc[12];
    qr[16] = 0.0;

    /*reaction 18: C2H5O + M => CH3 + CH2O + M */
    qf[17] = qss_sc[10];
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

    /*reaction 89: 2.000000 CH3O2 => O2 + 2.000000 CH3O */
    qf[88] = pow(sc[18], 2.000000);
    qr[88] = 0.0;

    /*reaction 90: CH3O2 + CH3 => 2.000000 CH3O */
    qf[89] = sc[12]*sc[18];
    qr[89] = 0.0;

    /*reaction 91: 2.000000 CH3O2 => CH2O + CH3OH + O2 */
    qf[90] = pow(sc[18], 2.000000);
    qr[90] = 0.0;

    /*reaction 92: CH3O2 + HO2 => CH3O2H + O2 */
    qf[91] = sc[7]*sc[18];
    qr[91] = 0.0;

    /*reaction 93: CH3O2H => CH3O + OH */
    qf[92] = qss_sc[6];
    qr[92] = 0.0;

    /*reaction 94: C2H2 + O => CH2 + CO */
    qf[93] = sc[1]*sc[19];
    qr[93] = 0.0;

    /*reaction 95: C2H2 + O => HCCO + H */
    qf[94] = sc[1]*sc[19];
    qr[94] = 0.0;

    /*reaction 96: C2H3 + H => C2H2 + H2 */
    qf[95] = sc[3]*qss_sc[8];
    qr[95] = 0.0;

    /*reaction 97: C2H3 + O2 => CH2CHO + O */
    qf[96] = sc[6]*qss_sc[8];
    qr[96] = 0.0;

    /*reaction 98: C2H3 + CH3 => C3H6 */
    qf[97] = sc[12]*qss_sc[8];
    qr[97] = 0.0;

    /*reaction 99: C2H3 + O2 => C2H2 + HO2 */
    qf[98] = sc[6]*qss_sc[8];
    qr[98] = 0.0;

    /*reaction 100: C2H3 + O2 => CH2O + HCO */
    qf[99] = sc[6]*qss_sc[8];
    qr[99] = 0.0;

    /*reaction 101: C2H4 + CH3 => C2H3 + CH4 */
    qf[100] = sc[12]*sc[21];
    qr[100] = 0.0;

    /*reaction 102: C2H4 + O => CH3 + HCO */
    qf[101] = sc[1]*sc[21];
    qr[101] = 0.0;

    /*reaction 103: C2H4 + OH => C2H3 + H2O */
    qf[102] = sc[4]*sc[21];
    qr[102] = 0.0;

    /*reaction 104: C2H4 + O => CH2CHO + H */
    qf[103] = sc[1]*sc[21];
    qr[103] = 0.0;

    /*reaction 105: C2H4 + H => C2H3 + H2 */
    qf[104] = sc[3]*sc[21];
    qr[104] = 0.0;

    /*reaction 106: C2H3 + H2 => C2H4 + H */
    qf[105] = sc[2]*qss_sc[8];
    qr[105] = 0.0;

    /*reaction 107: H + C2H5 => C2H6 */
    qf[106] = sc[3]*qss_sc[5];
    qr[106] = 0.0;

    /*reaction 108: CH3O2 + C2H5 => CH3O + C2H5O */
    qf[107] = qss_sc[5]*sc[18];
    qr[107] = 0.0;

    /*reaction 109: C2H5 + HO2 => C2H5O + OH */
    qf[108] = sc[7]*qss_sc[5];
    qr[108] = 0.0;

    /*reaction 110: C2H5 + O2 => C2H4 + HO2 */
    qf[109] = sc[6]*qss_sc[5];
    qr[109] = 0.0;

    /*reaction 111: C2H6 + O => C2H5 + OH */
    qf[110] = sc[1]*sc[15];
    qr[110] = 0.0;

    /*reaction 112: C2H6 + OH => C2H5 + H2O */
    qf[111] = sc[4]*sc[15];
    qr[111] = 0.0;

    /*reaction 113: C2H6 + H => C2H5 + H2 */
    qf[112] = sc[3]*sc[15];
    qr[112] = 0.0;

    /*reaction 114: HCCO + O => H + 2.000000 CO */
    qf[113] = sc[1]*qss_sc[7];
    qr[113] = 0.0;

    /*reaction 115: HCCO + OH => 2.000000 HCO */
    qf[114] = sc[4]*qss_sc[7];
    qr[114] = 0.0;

    /*reaction 116: HCCO + O2 => CO2 + HCO */
    qf[115] = sc[6]*qss_sc[7];
    qr[115] = 0.0;

    /*reaction 117: HCCO + H => CH2GSG + CO */
    qf[116] = sc[3]*qss_sc[7];
    qr[116] = 0.0;

    /*reaction 118: CH2GSG + CO => HCCO + H */
    qf[117] = sc[10]*qss_sc[3];
    qr[117] = 0.0;

    /*reaction 119: CH2CO + O => HCCO + OH */
    qf[118] = sc[1]*sc[16];
    qr[118] = 0.0;

    /*reaction 120: CH2CO + H => HCCO + H2 */
    qf[119] = sc[3]*sc[16];
    qr[119] = 0.0;

    /*reaction 121: HCCO + H2 => CH2CO + H */
    qf[120] = sc[2]*qss_sc[7];
    qr[120] = 0.0;

    /*reaction 122: CH2CO + H => CH3 + CO */
    qf[121] = sc[3]*sc[16];
    qr[121] = 0.0;

    /*reaction 123: CH2CO + O => CH2 + CO2 */
    qf[122] = sc[1]*sc[16];
    qr[122] = 0.0;

    /*reaction 124: CH2CO + OH => HCCO + H2O */
    qf[123] = sc[4]*sc[16];
    qr[123] = 0.0;

    /*reaction 125: CH2CHO + O2 => CH2O + CO + OH */
    qf[124] = sc[6]*qss_sc[9];
    qr[124] = 0.0;

    /*reaction 126: CH2CHO => CH2CO + H */
    qf[125] = qss_sc[9];
    qr[125] = 0.0;

    /*reaction 127: CH2CO + H => CH2CHO */
    qf[126] = sc[3]*sc[16];
    qr[126] = 0.0;

    /*reaction 128: C2H5O2 => C2H5 + O2 */
    qf[127] = qss_sc[12];
    qr[127] = 0.0;

    /*reaction 129: C2H5 + O2 => C2H5O2 */
    qf[128] = sc[6]*qss_sc[5];
    qr[128] = 0.0;

    /*reaction 130: C2H5O2 => C2H4 + HO2 */
    qf[129] = qss_sc[12];
    qr[129] = 0.0;

    /*reaction 131: C3H2 + O2 => HCCO + CO + H */
    qf[130] = sc[6]*qss_sc[13];
    qr[130] = 0.0;

    /*reaction 132: C3H2 + OH => C2H2 + HCO */
    qf[131] = sc[4]*qss_sc[13];
    qr[131] = 0.0;

    /*reaction 133: C3H3 + O2 => CH2CO + HCO */
    qf[132] = sc[6]*sc[22];
    qr[132] = 0.0;

    /*reaction 134: C3H3 + HO2 => C3H4XA + O2 */
    qf[133] = sc[7]*sc[22];
    qr[133] = 0.0;

    /*reaction 135: C3H3 + H => C3H2 + H2 */
    qf[134] = sc[3]*sc[22];
    qr[134] = 0.0;

    /*reaction 136: C3H3 + OH => C3H2 + H2O */
    qf[135] = sc[4]*sc[22];
    qr[135] = 0.0;

    /*reaction 137: C3H2 + H2O => C3H3 + OH */
    qf[136] = sc[5]*qss_sc[13];
    qr[136] = 0.0;

    /*reaction 138: C3H4XA + H => C3H3 + H2 */
    qf[137] = sc[3]*sc[23];
    qr[137] = 0.0;

    /*reaction 139: C3H4XA + OH => C3H3 + H2O */
    qf[138] = sc[4]*sc[23];
    qr[138] = 0.0;

    /*reaction 140: C3H4XA + O => C2H4 + CO */
    qf[139] = sc[1]*sc[23];
    qr[139] = 0.0;

    /*reaction 141: C3H5XA + H => C3H4XA + H2 */
    qf[140] = sc[3]*sc[24];
    qr[140] = 0.0;

    /*reaction 142: C3H5XA + HO2 => C3H6 + O2 */
    qf[141] = sc[7]*sc[24];
    qr[141] = 0.0;

    /*reaction 143: C3H5XA + H => C3H6 */
    qf[142] = sc[3]*sc[24];
    qr[142] = 0.0;

    /*reaction 144: C3H5XA => C2H2 + CH3 */
    qf[143] = sc[24];
    qr[143] = 0.0;

    /*reaction 145: C3H5XA => C3H4XA + H */
    qf[144] = sc[24];
    qr[144] = 0.0;

    /*reaction 146: C3H4XA + H => C3H5XA */
    qf[145] = sc[3]*sc[23];
    qr[145] = 0.0;

    /*reaction 147: C3H5XA + CH2O => C3H6 + HCO */
    qf[146] = sc[11]*sc[24];
    qr[146] = 0.0;

    /*reaction 148: 2.000000 C3H5XA => C3H4XA + C3H6 */
    qf[147] = pow(sc[24], 2.000000);
    qr[147] = 0.0;

    /*reaction 149: C3H6 + H => C2H4 + CH3 */
    qf[148] = sc[3]*sc[20];
    qr[148] = 0.0;

    /*reaction 150: C3H6 + H => C3H5XA + H2 */
    qf[149] = sc[3]*sc[20];
    qr[149] = 0.0;

    /*reaction 151: C3H6 + O => C2H5 + HCO */
    qf[150] = sc[1]*sc[20];
    qr[150] = 0.0;

    /*reaction 152: C3H6 + O => C3H5XA + OH */
    qf[151] = sc[1]*sc[20];
    qr[151] = 0.0;

    /*reaction 153: C3H6 + O => CH2CO + CH3 + H */
    qf[152] = sc[1]*sc[20];
    qr[152] = 0.0;

    /*reaction 154: C3H6 + OH => C3H5XA + H2O */
    qf[153] = sc[4]*sc[20];
    qr[153] = 0.0;

    /*reaction 155: NXC3H7 + O2 => C3H6 + HO2 */
    qf[154] = sc[6]*sc[25];
    qr[154] = 0.0;

    /*reaction 156: NXC3H7 => CH3 + C2H4 */
    qf[155] = sc[25];
    qr[155] = 0.0;

    /*reaction 157: CH3 + C2H4 => NXC3H7 */
    qf[156] = sc[12]*sc[21];
    qr[156] = 0.0;

    /*reaction 158: NXC3H7 => H + C3H6 */
    qf[157] = sc[25];
    qr[157] = 0.0;

    /*reaction 159: H + C3H6 => NXC3H7 */
    qf[158] = sc[3]*sc[20];
    qr[158] = 0.0;

    /*reaction 160: NXC3H7O2 => NXC3H7 + O2 */
    qf[159] = sc[26];
    qr[159] = 0.0;

    /*reaction 161: NXC3H7 + O2 => NXC3H7O2 */
    qf[160] = sc[6]*sc[25];
    qr[160] = 0.0;

    /*reaction 162: C4H6 => 2.000000 C2H3 */
    qf[161] = sc[27];
    qr[161] = 0.0;

    /*reaction 163: 2.000000 C2H3 => C4H6 */
    qf[162] = pow(qss_sc[8], 2.000000);
    qr[162] = 0.0;

    /*reaction 164: C4H6 + OH => CH2O + C3H5XA */
    qf[163] = sc[4]*sc[27];
    qr[163] = 0.0;

    /*reaction 165: C4H6 + OH => C2H5 + CH2CO */
    qf[164] = sc[4]*sc[27];
    qr[164] = 0.0;

    /*reaction 166: C4H6 + O => C2H4 + CH2CO */
    qf[165] = sc[1]*sc[27];
    qr[165] = 0.0;

    /*reaction 167: C4H6 + H => C2H3 + C2H4 */
    qf[166] = sc[3]*sc[27];
    qr[166] = 0.0;

    /*reaction 168: C4H6 + O => CH2O + C3H4XA */
    qf[167] = sc[1]*sc[27];
    qr[167] = 0.0;

    /*reaction 169: H + C4H7 => C4H8X1 */
    qf[168] = sc[3]*sc[28];
    qr[168] = 0.0;

    /*reaction 170: C3H5XA + C4H7 => C3H6 + C4H6 */
    qf[169] = sc[24]*sc[28];
    qr[169] = 0.0;

    /*reaction 171: C2H5 + C4H7 => C4H6 + C2H6 */
    qf[170] = qss_sc[5]*sc[28];
    qr[170] = 0.0;

    /*reaction 172: C4H7 => C4H6 + H */
    qf[171] = sc[28];
    qr[171] = 0.0;

    /*reaction 173: C4H6 + H => C4H7 */
    qf[172] = sc[3]*sc[27];
    qr[172] = 0.0;

    /*reaction 174: C4H7 + CH3 => C4H6 + CH4 */
    qf[173] = sc[12]*sc[28];
    qr[173] = 0.0;

    /*reaction 175: C4H7 + HO2 => C4H8X1 + O2 */
    qf[174] = sc[7]*sc[28];
    qr[174] = 0.0;

    /*reaction 176: C4H7 + O2 => C4H6 + HO2 */
    qf[175] = sc[6]*sc[28];
    qr[175] = 0.0;

    /*reaction 177: C4H7 => C2H4 + C2H3 */
    qf[176] = sc[28];
    qr[176] = 0.0;

    /*reaction 178: H + C4H7 => C4H6 + H2 */
    qf[177] = sc[3]*sc[28];
    qr[177] = 0.0;

    /*reaction 179: C4H8X1 + H => C4H7 + H2 */
    qf[178] = sc[3]*sc[29];
    qr[178] = 0.0;

    /*reaction 180: C4H8X1 + OH => NXC3H7 + CH2O */
    qf[179] = sc[4]*sc[29];
    qr[179] = 0.0;

    /*reaction 181: C4H8X1 + OH => CH3CO + C2H6 */
    qf[180] = sc[4]*sc[29];
    qr[180] = 0.0;

    /*reaction 182: C4H8X1 + O => CH3CO + C2H5 */
    qf[181] = sc[1]*sc[29];
    qr[181] = 0.0;

    /*reaction 183: C4H8X1 + O => C3H6 + CH2O */
    qf[182] = sc[1]*sc[29];
    qr[182] = 0.0;

    /*reaction 184: C4H8X1 + OH => C4H7 + H2O */
    qf[183] = sc[4]*sc[29];
    qr[183] = 0.0;

    /*reaction 185: C4H8X1 => C3H5XA + CH3 */
    qf[184] = sc[29];
    qr[184] = 0.0;

    /*reaction 186: C3H5XA + CH3 => C4H8X1 */
    qf[185] = sc[12]*sc[24];
    qr[185] = 0.0;

    /*reaction 187: PXC4H9 => C4H8X1 + H */
    qf[186] = sc[30];
    qr[186] = 0.0;

    /*reaction 188: C4H8X1 + H => PXC4H9 */
    qf[187] = sc[3]*sc[29];
    qr[187] = 0.0;

    /*reaction 189: PXC4H9 => C2H5 + C2H4 */
    qf[188] = sc[30];
    qr[188] = 0.0;

    /*reaction 190: PXC4H9O2 => PXC4H9 + O2 */
    qf[189] = sc[31];
    qr[189] = 0.0;

    /*reaction 191: PXC4H9 + O2 => PXC4H9O2 */
    qf[190] = sc[6]*sc[30];
    qr[190] = 0.0;

    /*reaction 192: C5H9 => C4H6 + CH3 */
    qf[191] = sc[32];
    qr[191] = 0.0;

    /*reaction 193: C5H9 => C3H5XA + C2H4 */
    qf[192] = sc[32];
    qr[192] = 0.0;

    /*reaction 194: C5H10X1 + OH => C5H9 + H2O */
    qf[193] = sc[4]*sc[33];
    qr[193] = 0.0;

    /*reaction 195: C5H10X1 + H => C5H9 + H2 */
    qf[194] = sc[3]*sc[33];
    qr[194] = 0.0;

    /*reaction 196: C5H10X1 => C2H5 + C3H5XA */
    qf[195] = sc[33];
    qr[195] = 0.0;

    /*reaction 197: C2H5 + C3H5XA => C5H10X1 */
    qf[196] = qss_sc[5]*sc[24];
    qr[196] = 0.0;

    /*reaction 198: C5H10X1 + O => C5H9 + OH */
    qf[197] = sc[1]*sc[33];
    qr[197] = 0.0;

    /*reaction 199: C5H11X1 => C3H6 + C2H5 */
    qf[198] = sc[34];
    qr[198] = 0.0;

    /*reaction 200: C5H11X1 => C2H4 + NXC3H7 */
    qf[199] = sc[34];
    qr[199] = 0.0;

    /*reaction 201: C5H11X1 <=> C5H10X1 + H */
    qf[200] = sc[34];
    qr[200] = sc[3]*sc[33];

    /*reaction 202: C6H12X1 => NXC3H7 + C3H5XA */
    qf[201] = sc[35];
    qr[201] = 0.0;

    /*reaction 203: C6H12X1 + OH => C5H11X1 + CH2O */
    qf[202] = sc[4]*sc[35];
    qr[202] = 0.0;

    /*reaction 204: C7H15X2 => C6H12X1 + CH3 */
    qf[203] = sc[36];
    qr[203] = 0.0;

    /*reaction 205: C7H15X2 => PXC4H9 + C3H6 */
    qf[204] = sc[36];
    qr[204] = 0.0;

    /*reaction 206: C7H15X2 => C4H8X1 + NXC3H7 */
    qf[205] = sc[36];
    qr[205] = 0.0;

    /*reaction 207: C7H15X2 => C5H11X1 + C2H4 */
    qf[206] = sc[36];
    qr[206] = 0.0;

    /*reaction 208: C7H15X2 => C2H5 + C5H10X1 */
    qf[207] = sc[36];
    qr[207] = 0.0;

    /*reaction 209: C7H15X2 + HO2 => NXC7H16 + O2 */
    qf[208] = sc[7]*sc[36];
    qr[208] = 0.0;

    /*reaction 210: NXC7H16 + CH3O2 => C7H15X2 + CH3O2H */
    qf[209] = sc[18]*sc[37];
    qr[209] = 0.0;

    /*reaction 211: NXC7H16 + H => C7H15X2 + H2 */
    qf[210] = sc[3]*sc[37];
    qr[210] = 0.0;

    /*reaction 212: NXC7H16 => PXC4H9 + NXC3H7 */
    qf[211] = sc[37];
    qr[211] = 0.0;

    /*reaction 213: NXC7H16 + HO2 => C7H15X2 + H2O2 */
    qf[212] = sc[7]*sc[37];
    qr[212] = 0.0;

    /*reaction 214: NXC7H16 => C5H11X1 + C2H5 */
    qf[213] = sc[37];
    qr[213] = 0.0;

    /*reaction 215: NXC7H16 + CH3O => C7H15X2 + CH3OH */
    qf[214] = qss_sc[4]*sc[37];
    qr[214] = 0.0;

    /*reaction 216: NXC7H16 + O => C7H15X2 + OH */
    qf[215] = sc[1]*sc[37];
    qr[215] = 0.0;

    /*reaction 217: NXC7H16 + OH => C7H15X2 + H2O */
    qf[216] = sc[4]*sc[37];
    qr[216] = 0.0;

    /*reaction 218: NXC7H16 + CH3 => C7H15X2 + CH4 */
    qf[217] = sc[12]*sc[37];
    qr[217] = 0.0;

    double T = tc[1];

    /*compute the mixture concentration */
    double mixture = 0.0;
    for (int i = 0; i < 38; ++i) {
        mixture += sc[i];
    }

    double Corr[218];
    for (int i = 0; i < 218; ++i) {
        Corr[i] = 1.0;
    }

    /* troe */
    {
        double alpha[11];
        alpha[0] = mixture + (TB[0][2] - 1)*sc[10] + (TB[0][3] - 1)*sc[2] + (TB[0][4] - 1)*sc[30] + (TB[0][6] - 1)*sc[28] + (TB[0][8] - 1)*sc[34] + (TB[0][9] - 1)*sc[5] + (TB[0][10] - 1)*sc[9] + (TB[0][14] - 1)*sc[36];
        alpha[1] = mixture + (TB[1][2] - 1)*sc[10] + (TB[1][3] - 1)*sc[2] + (TB[1][4] - 1)*sc[30] + (TB[1][6] - 1)*sc[28] + (TB[1][8] - 1)*sc[34] + (TB[1][9] - 1)*sc[5] + (TB[1][10] - 1)*sc[9] + (TB[1][14] - 1)*sc[36];
        alpha[2] = mixture + (TB[2][2] - 1)*sc[10] + (TB[2][3] - 1)*sc[2] + (TB[2][4] - 1)*sc[30] + (TB[2][6] - 1)*sc[28] + (TB[2][8] - 1)*sc[34] + (TB[2][9] - 1)*sc[5] + (TB[2][10] - 1)*sc[9] + (TB[2][11] - 1)*sc[13] + (TB[2][12] - 1)*sc[15] + (TB[2][16] - 1)*sc[36];
        alpha[3] = mixture + (TB[3][2] - 1)*sc[10] + (TB[3][3] - 1)*sc[2] + (TB[3][4] - 1)*sc[30] + (TB[3][6] - 1)*sc[28] + (TB[3][8] - 1)*sc[34] + (TB[3][9] - 1)*sc[5] + (TB[3][10] - 1)*sc[9] + (TB[3][14] - 1)*sc[36];
        alpha[4] = mixture + (TB[4][2] - 1)*sc[10] + (TB[4][3] - 1)*sc[2] + (TB[4][4] - 1)*sc[30] + (TB[4][6] - 1)*sc[28] + (TB[4][8] - 1)*sc[34] + (TB[4][9] - 1)*sc[5] + (TB[4][10] - 1)*sc[9] + (TB[4][14] - 1)*sc[36];
        alpha[5] = mixture + (TB[5][3] - 1)*sc[30] + (TB[5][5] - 1)*sc[28] + (TB[5][7] - 1)*sc[34] + (TB[5][10] - 1)*sc[36];
        alpha[6] = mixture + (TB[6][2] - 1)*sc[10] + (TB[6][3] - 1)*sc[2] + (TB[6][4] - 1)*sc[30] + (TB[6][6] - 1)*sc[28] + (TB[6][8] - 1)*sc[34] + (TB[6][9] - 1)*sc[5] + (TB[6][10] - 1)*sc[9] + (TB[6][14] - 1)*sc[36];
        alpha[7] = mixture + (TB[7][3] - 1)*sc[30] + (TB[7][5] - 1)*sc[28] + (TB[7][7] - 1)*sc[34] + (TB[7][10] - 1)*sc[36];
        alpha[8] = mixture + (TB[8][2] - 1)*sc[10] + (TB[8][3] - 1)*sc[2] + (TB[8][4] - 1)*sc[30] + (TB[8][6] - 1)*sc[28] + (TB[8][8] - 1)*sc[34] + (TB[8][9] - 1)*sc[5] + (TB[8][10] - 1)*sc[9] + (TB[8][14] - 1)*sc[36];
        alpha[9] = mixture + (TB[9][3] - 1)*sc[30] + (TB[9][5] - 1)*sc[28] + (TB[9][7] - 1)*sc[34] + (TB[9][10] - 1)*sc[36];
        alpha[10] = mixture + (TB[10][3] - 1)*sc[30] + (TB[10][5] - 1)*sc[28] + (TB[10][7] - 1)*sc[34] + (TB[10][10] - 1)*sc[36];
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
        alpha = mixture + (TB[11][2] - 1)*sc[10] + (TB[11][3] - 1)*sc[2] + (TB[11][4] - 1)*sc[30] + (TB[11][6] - 1)*sc[28] + (TB[11][8] - 1)*sc[34] + (TB[11][9] - 1)*sc[5] + (TB[11][10] - 1)*sc[9] + (TB[11][14] - 1)*sc[36];
        Corr[11] = alpha;
        alpha = mixture + (TB[12][3] - 1)*sc[30] + (TB[12][5] - 1)*sc[28] + (TB[12][7] - 1)*sc[34] + (TB[12][10] - 1)*sc[36];
        Corr[12] = alpha;
        alpha = mixture + (TB[13][3] - 1)*sc[30] + (TB[13][5] - 1)*sc[28] + (TB[13][7] - 1)*sc[34] + (TB[13][10] - 1)*sc[36];
        Corr[13] = alpha;
        alpha = mixture + (TB[14][2] - 1)*sc[10] + (TB[14][3] - 1)*sc[2] + (TB[14][4] - 1)*sc[30] + (TB[14][6] - 1)*sc[28] + (TB[14][8] - 1)*sc[34] + (TB[14][9] - 1)*sc[5] + (TB[14][10] - 1)*sc[9] + (TB[14][14] - 1)*sc[36];
        Corr[14] = alpha;
        alpha = mixture + (TB[15][3] - 1)*sc[30] + (TB[15][5] - 1)*sc[28] + (TB[15][7] - 1)*sc[34] + (TB[15][10] - 1)*sc[36];
        Corr[15] = alpha;
        alpha = mixture + (TB[16][3] - 1)*sc[30] + (TB[16][5] - 1)*sc[28] + (TB[16][7] - 1)*sc[34] + (TB[16][10] - 1)*sc[36];
        Corr[16] = alpha;
        alpha = mixture + (TB[17][3] - 1)*sc[30] + (TB[17][5] - 1)*sc[28] + (TB[17][7] - 1)*sc[34] + (TB[17][10] - 1)*sc[36];
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

void comp_qss_coeff(double *  qf_co, double *  qr_co, double *  sc, double *  tc, double invT)
{

    /*reaction 6: CO + CH2 (+M) => CH2CO (+M) */
    qf_co[0] = 0.0*sc[10];
    qr_co[0] = 0.0;

    /*reaction 8: CH3O (+M) => CH2O + H (+M) */
    qf_co[1] = 0.0;
    qr_co[1] = 0.0;

    /*reaction 9: C2H3 (+M) => H + C2H2 (+M) */
    qf_co[2] = 0.0;
    qr_co[2] = 0.0;

    /*reaction 10: H + C2H4 (+M) <=> C2H5 (+M) */
    qf_co[3] = sc[3]*sc[21];
    qr_co[3] = 0.0;

    /*reaction 11: CH3CO (+M) => CH3 + CO (+M) */
    qf_co[4] = 0.0;
    qr_co[4] = 0.0;

    /*reaction 13: CH2GSG + M => CH2 + M */
    qf_co[5] = 0.0;
    qr_co[5] = 0.0;

    /*reaction 14: CH2 + M => CH2GSG + M */
    qf_co[6] = 0.0;
    qr_co[6] = 0.0;

    /*reaction 15: HCO + M => H + CO + M */
    qf_co[7] = 0.0;
    qr_co[7] = 0.0;

    /*reaction 18: C2H5O + M => CH3 + CH2O + M */
    qf_co[8] = 0.0;
    qr_co[8] = 0.0;

    /*reaction 34: CH + O2 => HCO + O */
    qf_co[9] = sc[6];
    qr_co[9] = 0.0;

    /*reaction 35: CH2 + O2 => CO2 + 2.000000 H */
    qf_co[10] = sc[6];
    qr_co[10] = 0.0;

    /*reaction 36: CH2 + O2 => CO + H2O */
    qf_co[11] = sc[6];
    qr_co[11] = 0.0;

    /*reaction 37: CH2 + O => CO + 2.000000 H */
    qf_co[12] = sc[1];
    qr_co[12] = 0.0;

    /*reaction 38: CH2 + O2 => CH2O + O */
    qf_co[13] = sc[6];
    qr_co[13] = 0.0;

    /*reaction 39: CH2 + H => CH + H2 */
    qf_co[14] = sc[3];
    qr_co[14] = 0.0;

    /*reaction 40: CH + H2 => CH2 + H */
    qf_co[15] = sc[2];
    qr_co[15] = 0.0;

    /*reaction 41: CH2 + OH => CH + H2O */
    qf_co[16] = sc[4];
    qr_co[16] = 0.0;

    /*reaction 42: CH + H2O => CH2 + OH */
    qf_co[17] = sc[5];
    qr_co[17] = 0.0;

    /*reaction 43: CH2 + O2 => CO2 + H2 */
    qf_co[18] = sc[6];
    qr_co[18] = 0.0;

    /*reaction 44: CH2GSG + H => CH + H2 */
    qf_co[19] = sc[3];
    qr_co[19] = 0.0;

    /*reaction 45: CH2GSG + H2 => CH3 + H */
    qf_co[20] = sc[2];
    qr_co[20] = 0.0;

    /*reaction 46: CH3 + H => CH2GSG + H2 */
    qf_co[21] = sc[3]*sc[12];
    qr_co[21] = 0.0;

    /*reaction 47: CH2GSG + O2 => CO + OH + H */
    qf_co[22] = sc[6];
    qr_co[22] = 0.0;

    /*reaction 48: CH2GSG + OH => CH2O + H */
    qf_co[23] = sc[4];
    qr_co[23] = 0.0;

    /*reaction 50: CH3 + OH => CH2GSG + H2O */
    qf_co[24] = sc[4]*sc[12];
    qr_co[24] = 0.0;

    /*reaction 51: CH2GSG + H2O => CH3 + OH */
    qf_co[25] = sc[5];
    qr_co[25] = 0.0;

    /*reaction 53: CH3 + HO2 => CH3O + OH */
    qf_co[26] = sc[7]*sc[12];
    qr_co[26] = 0.0;

    /*reaction 56: CH3 + H => CH2 + H2 */
    qf_co[27] = sc[3]*sc[12];
    qr_co[27] = 0.0;

    /*reaction 57: CH2 + H2 => CH3 + H */
    qf_co[28] = sc[2];
    qr_co[28] = 0.0;

    /*reaction 58: 2.000000 CH3 <=> H + C2H5 */
    qf_co[29] = pow(sc[12], 2.000000);
    qr_co[29] = sc[3];

    /*reaction 59: CH3 + OH => CH2 + H2O */
    qf_co[30] = sc[4]*sc[12];
    qr_co[30] = 0.0;

    /*reaction 60: CH2 + H2O => CH3 + OH */
    qf_co[31] = sc[5];
    qr_co[31] = 0.0;

    /*reaction 68: HCO + O2 => CO + HO2 */
    qf_co[32] = sc[6];
    qr_co[32] = 0.0;

    /*reaction 69: HCO + O => CO2 + H */
    qf_co[33] = sc[1];
    qr_co[33] = 0.0;

    /*reaction 70: HCO + OH => CO + H2O */
    qf_co[34] = sc[4];
    qr_co[34] = 0.0;

    /*reaction 71: HCO + H => CO + H2 */
    qf_co[35] = sc[3];
    qr_co[35] = 0.0;

    /*reaction 72: HCO + O => CO + OH */
    qf_co[36] = sc[1];
    qr_co[36] = 0.0;

    /*reaction 73: HCO + CH3 => CH4 + CO */
    qf_co[37] = 0.0*sc[12];
    qr_co[37] = 0.0;

    /*reaction 74: CH2O + OH => HCO + H2O */
    qf_co[38] = sc[4]*sc[11];
    qr_co[38] = 0.0;

    /*reaction 75: CH2O + O => HCO + OH */
    qf_co[39] = sc[1]*sc[11];
    qr_co[39] = 0.0;

    /*reaction 76: CH2O + H => HCO + H2 */
    qf_co[40] = sc[3]*sc[11];
    qr_co[40] = 0.0;

    /*reaction 77: CH2O + CH3 => HCO + CH4 */
    qf_co[41] = sc[11]*sc[12];
    qr_co[41] = 0.0;

    /*reaction 78: 2.000000 CH3O => CH3OH + CH2O */
    qf_co[42] = 0.0;
    qr_co[42] = 0.0;

    /*reaction 79: CH3O + O2 => CH2O + HO2 */
    qf_co[43] = sc[6];
    qr_co[43] = 0.0;

    /*reaction 80: CH3O + H2 => CH3OH + H */
    qf_co[44] = sc[2];
    qr_co[44] = 0.0;

    /*reaction 81: CH3OH + OH => CH3O + H2O */
    qf_co[45] = sc[4]*sc[14];
    qr_co[45] = 0.0;

    /*reaction 82: CH2GSG + CO2 => CH2O + CO */
    qf_co[46] = sc[9];
    qr_co[46] = 0.0;

    /*reaction 85: HOCHO => HCO + OH */
    qf_co[47] = sc[17];
    qr_co[47] = 0.0;

    /*reaction 86: HCO + OH => HOCHO */
    qf_co[48] = sc[4];
    qr_co[48] = 0.0;

    /*reaction 89: 2.000000 CH3O2 => O2 + 2.000000 CH3O */
    qf_co[49] = pow(sc[18], 2.000000);
    qr_co[49] = 0.0;

    /*reaction 90: CH3O2 + CH3 => 2.000000 CH3O */
    qf_co[50] = sc[12]*sc[18];
    qr_co[50] = 0.0;

    /*reaction 92: CH3O2 + HO2 => CH3O2H + O2 */
    qf_co[51] = sc[7]*sc[18];
    qr_co[51] = 0.0;

    /*reaction 93: CH3O2H => CH3O + OH */
    qf_co[52] = 0.0;
    qr_co[52] = 0.0;

    /*reaction 94: C2H2 + O => CH2 + CO */
    qf_co[53] = sc[1]*sc[19];
    qr_co[53] = 0.0;

    /*reaction 95: C2H2 + O => HCCO + H */
    qf_co[54] = sc[1]*sc[19];
    qr_co[54] = 0.0;

    /*reaction 96: C2H3 + H => C2H2 + H2 */
    qf_co[55] = sc[3];
    qr_co[55] = 0.0;

    /*reaction 97: C2H3 + O2 => CH2CHO + O */
    qf_co[56] = sc[6];
    qr_co[56] = 0.0;

    /*reaction 98: C2H3 + CH3 => C3H6 */
    qf_co[57] = sc[12];
    qr_co[57] = 0.0;

    /*reaction 99: C2H3 + O2 => C2H2 + HO2 */
    qf_co[58] = sc[6];
    qr_co[58] = 0.0;

    /*reaction 100: C2H3 + O2 => CH2O + HCO */
    qf_co[59] = sc[6];
    qr_co[59] = 0.0;

    /*reaction 101: C2H4 + CH3 => C2H3 + CH4 */
    qf_co[60] = sc[12]*sc[21];
    qr_co[60] = 0.0;

    /*reaction 102: C2H4 + O => CH3 + HCO */
    qf_co[61] = sc[1]*sc[21];
    qr_co[61] = 0.0;

    /*reaction 103: C2H4 + OH => C2H3 + H2O */
    qf_co[62] = sc[4]*sc[21];
    qr_co[62] = 0.0;

    /*reaction 104: C2H4 + O => CH2CHO + H */
    qf_co[63] = sc[1]*sc[21];
    qr_co[63] = 0.0;

    /*reaction 105: C2H4 + H => C2H3 + H2 */
    qf_co[64] = sc[3]*sc[21];
    qr_co[64] = 0.0;

    /*reaction 106: C2H3 + H2 => C2H4 + H */
    qf_co[65] = sc[2];
    qr_co[65] = 0.0;

    /*reaction 107: H + C2H5 => C2H6 */
    qf_co[66] = sc[3];
    qr_co[66] = 0.0;

    /*reaction 108: CH3O2 + C2H5 => CH3O + C2H5O */
    qf_co[67] = 0.0*sc[18];
    qr_co[67] = 0.0;

    /*reaction 109: C2H5 + HO2 => C2H5O + OH */
    qf_co[68] = sc[7];
    qr_co[68] = 0.0;

    /*reaction 110: C2H5 + O2 => C2H4 + HO2 */
    qf_co[69] = sc[6];
    qr_co[69] = 0.0;

    /*reaction 111: C2H6 + O => C2H5 + OH */
    qf_co[70] = sc[1]*sc[15];
    qr_co[70] = 0.0;

    /*reaction 112: C2H6 + OH => C2H5 + H2O */
    qf_co[71] = sc[4]*sc[15];
    qr_co[71] = 0.0;

    /*reaction 113: C2H6 + H => C2H5 + H2 */
    qf_co[72] = sc[3]*sc[15];
    qr_co[72] = 0.0;

    /*reaction 114: HCCO + O => H + 2.000000 CO */
    qf_co[73] = sc[1];
    qr_co[73] = 0.0;

    /*reaction 115: HCCO + OH => 2.000000 HCO */
    qf_co[74] = sc[4];
    qr_co[74] = 0.0;

    /*reaction 116: HCCO + O2 => CO2 + HCO */
    qf_co[75] = sc[6];
    qr_co[75] = 0.0;

    /*reaction 117: HCCO + H => CH2GSG + CO */
    qf_co[76] = sc[3];
    qr_co[76] = 0.0;

    /*reaction 118: CH2GSG + CO => HCCO + H */
    qf_co[77] = sc[10];
    qr_co[77] = 0.0;

    /*reaction 119: CH2CO + O => HCCO + OH */
    qf_co[78] = sc[1]*sc[16];
    qr_co[78] = 0.0;

    /*reaction 120: CH2CO + H => HCCO + H2 */
    qf_co[79] = sc[3]*sc[16];
    qr_co[79] = 0.0;

    /*reaction 121: HCCO + H2 => CH2CO + H */
    qf_co[80] = sc[2];
    qr_co[80] = 0.0;

    /*reaction 123: CH2CO + O => CH2 + CO2 */
    qf_co[81] = sc[1]*sc[16];
    qr_co[81] = 0.0;

    /*reaction 124: CH2CO + OH => HCCO + H2O */
    qf_co[82] = sc[4]*sc[16];
    qr_co[82] = 0.0;

    /*reaction 125: CH2CHO + O2 => CH2O + CO + OH */
    qf_co[83] = sc[6];
    qr_co[83] = 0.0;

    /*reaction 126: CH2CHO => CH2CO + H */
    qf_co[84] = 0.0;
    qr_co[84] = 0.0;

    /*reaction 127: CH2CO + H => CH2CHO */
    qf_co[85] = sc[3]*sc[16];
    qr_co[85] = 0.0;

    /*reaction 128: C2H5O2 => C2H5 + O2 */
    qf_co[86] = 0.0;
    qr_co[86] = 0.0;

    /*reaction 129: C2H5 + O2 => C2H5O2 */
    qf_co[87] = sc[6];
    qr_co[87] = 0.0;

    /*reaction 130: C2H5O2 => C2H4 + HO2 */
    qf_co[88] = 0.0;
    qr_co[88] = 0.0;

    /*reaction 131: C3H2 + O2 => HCCO + CO + H */
    qf_co[89] = sc[6];
    qr_co[89] = 0.0;

    /*reaction 132: C3H2 + OH => C2H2 + HCO */
    qf_co[90] = sc[4];
    qr_co[90] = 0.0;

    /*reaction 133: C3H3 + O2 => CH2CO + HCO */
    qf_co[91] = sc[6]*sc[22];
    qr_co[91] = 0.0;

    /*reaction 135: C3H3 + H => C3H2 + H2 */
    qf_co[92] = sc[3]*sc[22];
    qr_co[92] = 0.0;

    /*reaction 136: C3H3 + OH => C3H2 + H2O */
    qf_co[93] = sc[4]*sc[22];
    qr_co[93] = 0.0;

    /*reaction 137: C3H2 + H2O => C3H3 + OH */
    qf_co[94] = sc[5];
    qr_co[94] = 0.0;

    /*reaction 147: C3H5XA + CH2O => C3H6 + HCO */
    qf_co[95] = sc[11]*sc[24];
    qr_co[95] = 0.0;

    /*reaction 151: C3H6 + O => C2H5 + HCO */
    qf_co[96] = sc[1]*sc[20];
    qr_co[96] = 0.0;

    /*reaction 162: C4H6 => 2.000000 C2H3 */
    qf_co[97] = sc[27];
    qr_co[97] = 0.0;

    /*reaction 163: 2.000000 C2H3 => C4H6 */
    qf_co[98] = 0.0;
    qr_co[98] = 0.0;

    /*reaction 165: C4H6 + OH => C2H5 + CH2CO */
    qf_co[99] = sc[4]*sc[27];
    qr_co[99] = 0.0;

    /*reaction 167: C4H6 + H => C2H3 + C2H4 */
    qf_co[100] = sc[3]*sc[27];
    qr_co[100] = 0.0;

    /*reaction 171: C2H5 + C4H7 => C4H6 + C2H6 */
    qf_co[101] = 0.0*sc[28];
    qr_co[101] = 0.0;

    /*reaction 177: C4H7 => C2H4 + C2H3 */
    qf_co[102] = sc[28];
    qr_co[102] = 0.0;

    /*reaction 181: C4H8X1 + OH => CH3CO + C2H6 */
    qf_co[103] = sc[4]*sc[29];
    qr_co[103] = 0.0;

    /*reaction 182: C4H8X1 + O => CH3CO + C2H5 */
    qf_co[104] = sc[1]*sc[29];
    qr_co[104] = 0.0;

    /*reaction 189: PXC4H9 => C2H5 + C2H4 */
    qf_co[105] = sc[30];
    qr_co[105] = 0.0;

    /*reaction 196: C5H10X1 => C2H5 + C3H5XA */
    qf_co[106] = sc[33];
    qr_co[106] = 0.0;

    /*reaction 197: C2H5 + C3H5XA => C5H10X1 */
    qf_co[107] = 0.0*sc[24];
    qr_co[107] = 0.0;

    /*reaction 199: C5H11X1 => C3H6 + C2H5 */
    qf_co[108] = sc[34];
    qr_co[108] = 0.0;

    /*reaction 208: C7H15X2 => C2H5 + C5H10X1 */
    qf_co[109] = sc[36];
    qr_co[109] = 0.0;

    /*reaction 210: NXC7H16 + CH3O2 => C7H15X2 + CH3O2H */
    qf_co[110] = sc[18]*sc[37];
    qr_co[110] = 0.0;

    /*reaction 214: NXC7H16 => C5H11X1 + C2H5 */
    qf_co[111] = sc[37];
    qr_co[111] = 0.0;

    /*reaction 215: NXC7H16 + CH3O => C7H15X2 + CH3OH */
    qf_co[112] = 0.0*sc[37];
    qr_co[112] = 0.0;

    double T = tc[1];

    /*compute the mixture concentration */
    double mixture = 0.0;
    for (int i = 0; i < 38; ++i) {
        mixture += sc[i];
    }

    double Corr[113];
    for (int i = 0; i < 113; ++i) {
        Corr[i] = 1.0;
    }

    /* troe */
    {
        double alpha[5];
        alpha[0] = mixture + (TB[5][3] - 1)*sc[30] + (TB[5][5] - 1)*sc[28] + (TB[5][7] - 1)*sc[34] + (TB[5][10] - 1)*sc[36];
        alpha[1] = mixture + (TB[7][3] - 1)*sc[30] + (TB[7][5] - 1)*sc[28] + (TB[7][7] - 1)*sc[34] + (TB[7][10] - 1)*sc[36];
        alpha[2] = mixture + (TB[8][2] - 1)*sc[10] + (TB[8][3] - 1)*sc[2] + (TB[8][4] - 1)*sc[30] + (TB[8][6] - 1)*sc[28] + (TB[8][8] - 1)*sc[34] + (TB[8][9] - 1)*sc[5] + (TB[8][10] - 1)*sc[9] + (TB[8][14] - 1)*sc[36];
        alpha[3] = mixture + (TB[9][3] - 1)*sc[30] + (TB[9][5] - 1)*sc[28] + (TB[9][7] - 1)*sc[34] + (TB[9][10] - 1)*sc[36];
        alpha[4] = mixture + (TB[10][3] - 1)*sc[30] + (TB[10][5] - 1)*sc[28] + (TB[10][7] - 1)*sc[34] + (TB[10][10] - 1)*sc[36];
#ifdef __INTEL_COMPILER
         #pragma simd
#endif
        for (int i=0; i<5; i++)
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
        alpha = mixture + (TB[12][3] - 1)*sc[30] + (TB[12][5] - 1)*sc[28] + (TB[12][7] - 1)*sc[34] + (TB[12][10] - 1)*sc[36];
        Corr[5] = alpha;
        alpha = mixture + (TB[13][3] - 1)*sc[30] + (TB[13][5] - 1)*sc[28] + (TB[13][7] - 1)*sc[34] + (TB[13][10] - 1)*sc[36];
        Corr[6] = alpha;
        alpha = mixture + (TB[14][2] - 1)*sc[10] + (TB[14][3] - 1)*sc[2] + (TB[14][4] - 1)*sc[30] + (TB[14][6] - 1)*sc[28] + (TB[14][8] - 1)*sc[34] + (TB[14][9] - 1)*sc[5] + (TB[14][10] - 1)*sc[9] + (TB[14][14] - 1)*sc[36];
        Corr[7] = alpha;
        alpha = mixture + (TB[17][3] - 1)*sc[30] + (TB[17][5] - 1)*sc[28] + (TB[17][7] - 1)*sc[34] + (TB[17][10] - 1)*sc[36];
        Corr[8] = alpha;
    }

    for (int i=0; i<113; i++)
    {
        qf_co[i] *= Corr[i] * k_f_save[i];
        qr_co[i] *= Corr[i] * k_f_save[i] / Kc_save[i];
    }

    return;
}

void comp_qss_sc(double * sc, double * sc_qss, double * tc, double invT)
{

    double  qf_co[113], qr_co[113];
    double epsilon = 1e-16;

    comp_qss_coeff(qf_co, qr_co, sc, tc, invT);

    /*QSS species 6: CH3O2H */

    double CH3O2H_num = epsilon - qf_co[91] - qf_co[209];
    double CH3O2H_denom = epsilon - qf_co[92];

    sc_qss[6] = CH3O2H_num/CH3O2H_denom;



    /*QSS species 8: C2H3 */

    double C2H3_num = epsilon - qf_co[100] - qf_co[102] - qf_co[104] - qf_co[161] - qf_co[166] - qf_co[176];
    double C2H3_denom = epsilon - qf_co[8] - qf_co[95] - qf_co[96] - qf_co[97] - qf_co[98] - qf_co[99] - qf_co[105] - qf_co[162];

    sc_qss[8] = C2H3_num/C2H3_denom;



    /*QSS species 11: CH3CO */

    double CH3CO_num = epsilon - qf_co[180] - qf_co[181]*sc_qss[5];
    double CH3CO_denom = epsilon - qf_co[10];

    sc_qss[11] = CH3CO_num/CH3CO_denom;



    /*QSS species 13: C3H2 */

    double C3H2_num = epsilon - qf_co[134] - qf_co[135];
    double C3H2_denom = epsilon - qf_co[130] - qf_co[131] - qf_co[136];

    sc_qss[13] = C3H2_num/C3H2_denom;



    /*QSS species 5: C2H5 */

    double C2H5_num = epsilon - qf_co[9] - qf_co[57] - qf_co[110] - qf_co[111] - qf_co[112] - qf_co[150]*sc_qss[1] - qf_co[164] - qf_co[181]*sc_qss[11] - qf_co[188] - qf_co[195] - qf_co[198] - qf_co[207] - qf_co[213];
    double C2H5_denom = epsilon - qr_co[9] - qr_co[57] - qf_co[106] - qf_co[107] - qf_co[108] - qf_co[109] - qf_co[128] - qf_co[170] - qf_co[196];
    double C2H5_rhs = C2H5_num/C2H5_denom;

    double C2H5_C2H5O2 = (epsilon + qf_co[127])/C2H5_denom;

    /*QSS species 12: C2H5O2 */

    double C2H5O2_num = epsilon ;
    double C2H5O2_denom = epsilon - qf_co[127] - qf_co[129];
    double C2H5O2_rhs = C2H5O2_num/C2H5O2_denom;

    double C2H5O2_C2H5 = (epsilon + qf_co[128])/C2H5O2_denom;

    sc_qss[5] = C2H5_rhs - (C2H5_C2H5O2 * ((C2H5O2_rhs - C2H5_rhs * C2H5O2_C2H5) / (1 - C2H5_C2H5O2 * C2H5O2_C2H5)));
    sc_qss[12] = (C2H5O2_rhs - C2H5_rhs * C2H5O2_C2H5) / (1 - C2H5_C2H5O2 * C2H5O2_C2H5);



    /*QSS species 4: CH3O */

    double CH3O_num = epsilon - qf_co[52] - qf_co[80] - qf_co[88] - qf_co[89] - qf_co[92]*sc_qss[6] - qf_co[107]*sc_qss[5];
    double CH3O_denom = epsilon - qf_co[7] - qf_co[77] - qf_co[78] - qf_co[79] - qf_co[214];

    sc_qss[4] = CH3O_num/CH3O_denom;



    /*QSS species 9: CH2CHO */

    double CH2CHO_num = epsilon - qf_co[96]*sc_qss[8] - qf_co[103] - qf_co[126];
    double CH2CHO_denom = epsilon - qf_co[124] - qf_co[125];

    sc_qss[9] = CH2CHO_num/CH2CHO_denom;



    /*QSS species 10: C2H5O */

    double C2H5O_num = epsilon - qf_co[107]*sc_qss[5] - qf_co[108]*sc_qss[5];
    double C2H5O_denom = epsilon - qf_co[17];

    sc_qss[10] = C2H5O_num/C2H5O_denom;



    /*QSS species 0: CH */

    double CH_num = epsilon ;
    double CH_denom = epsilon - qf_co[33] - qf_co[39] - qf_co[41];
    double CH_rhs = CH_num/CH_denom;

    double CH_CH2 = (epsilon + qf_co[38] + qf_co[40] + qf_co[43])/CH_denom;

    double CH_CH2GSG = (epsilon + qf_co[38] + qf_co[40] + qf_co[43])/CH_denom;

    double CH_HCCO = (epsilon + qf_co[38] + qf_co[40] + qf_co[43])/CH_denom;

    /*QSS species 2: CH2 */

    double CH2_num = epsilon - qf_co[55] - qf_co[58] - qf_co[93] - qf_co[122];
    double CH2_denom = epsilon - qf_co[5] - qf_co[13] - qf_co[34] - qf_co[35] - qf_co[36] - qf_co[37] - qf_co[38] - qf_co[40] - qf_co[42] - qf_co[56] - qf_co[59];
    double CH2_rhs = CH2_num/CH2_denom;

    double CH2_CH = (epsilon + qf_co[12] + qf_co[39] + qf_co[41])/CH2_denom;

    double CH2_CH2GSG = (epsilon + qf_co[12] + qf_co[39] + qf_co[41])/CH2_denom;

    double CH2_HCCO = (epsilon + qf_co[12] + qf_co[39] + qf_co[41])/CH2_denom;

    /*QSS species 3: CH2GSG */

    double CH2GSG_num = epsilon - qf_co[45] - qf_co[49];
    double CH2GSG_denom = epsilon - qf_co[12] - qf_co[43] - qf_co[44] - qf_co[46] - qf_co[47] - qf_co[50] - qf_co[81] - qf_co[117];
    double CH2GSG_rhs = CH2GSG_num/CH2GSG_denom;

    double CH2GSG_CH = (epsilon + qf_co[13] + qf_co[116])/CH2GSG_denom;

    double CH2GSG_CH2 = (epsilon + qf_co[13] + qf_co[116])/CH2GSG_denom;

    double CH2GSG_HCCO = (epsilon + qf_co[13] + qf_co[116])/CH2GSG_denom;

    /*QSS species 7: HCCO */

    double HCCO_num = epsilon - qf_co[94] - qf_co[118] - qf_co[119] - qf_co[123] - qf_co[130]*sc_qss[13];
    double HCCO_denom = epsilon - qf_co[113] - qf_co[114] - qf_co[115] - qf_co[116] - qf_co[120];
    double HCCO_rhs = HCCO_num/HCCO_denom;

    double HCCO_CH = (epsilon + qf_co[117])/HCCO_denom;

    double HCCO_CH2 = (epsilon + qf_co[117])/HCCO_denom;

    double HCCO_CH2GSG = (epsilon + qf_co[117])/HCCO_denom;

    sc_qss[0] = CH_rhs - (CH_HCCO * ((((HCCO_rhs - CH_rhs * HCCO_CH) - (CH2_rhs - CH_rhs * CH2_CH) * (HCCO_CH2 - CH_CH2 * HCCO_CH) / (1 - CH_CH2 * CH2_CH)) - ((CH2GSG_rhs - CH_rhs * CH2GSG_CH) - (CH2_rhs - CH_rhs * CH2_CH) * (CH2GSG_CH2 - CH_CH2 * CH2GSG_CH) / (1 - CH_CH2 * CH2_CH)) * ((HCCO_CH2GSG - CH_CH2GSG * HCCO_CH) - (CH2_CH2GSG - CH_CH2GSG * CH2_CH) * (HCCO_CH2 - CH_CH2 * HCCO_CH)/ (1 - CH_CH2 * CH2_CH)) / ((1 - CH_CH2GSG * CH2GSG_CH) - (CH2_CH2GSG - CH_CH2GSG * CH2_CH) * (CH2GSG_CH2 - CH_CH2 * CH2GSG_CH)/ (1 - CH_CH2 * CH2_CH))) / (((1 - CH_HCCO * HCCO_CH) - (CH2_HCCO - CH_HCCO * CH2_CH) * (HCCO_CH2 - CH_CH2 * HCCO_CH)/ (1 - CH_CH2 * CH2_CH)) - ((CH2GSG_HCCO - CH_HCCO * CH2GSG_CH) - (CH2_HCCO - CH_HCCO * CH2_CH) * (CH2GSG_CH2 - CH_CH2 * CH2GSG_CH)/ (1 - CH_CH2 * CH2_CH)) * ((HCCO_CH2GSG - CH_CH2GSG * HCCO_CH) - (CH2_CH2GSG - CH_CH2GSG * CH2_CH) * (HCCO_CH2 - CH_CH2 * HCCO_CH)/ (1 - CH_CH2 * CH2_CH))/ ((1 - CH_CH2GSG * CH2GSG_CH) - (CH2_CH2GSG - CH_CH2GSG * CH2_CH) * (CH2GSG_CH2 - CH_CH2 * CH2GSG_CH)/ (1 - CH_CH2 * CH2_CH)))) - CH_CH2GSG * ((((CH2GSG_rhs - CH_rhs * CH2GSG_CH) - (CH2_rhs - CH_rhs * CH2_CH) * (CH2GSG_CH2 - CH_CH2 * CH2GSG_CH) / (1 - CH_CH2 * CH2_CH)) - (((CH2GSG_HCCO - CH_HCCO * CH2GSG_CH) - (CH2_HCCO - CH_HCCO * CH2_CH) * (CH2GSG_CH2 - CH_CH2 * CH2GSG_CH)/ (1 - CH_CH2 * CH2_CH)) * ((((HCCO_rhs - CH_rhs * HCCO_CH) - (CH2_rhs - CH_rhs * CH2_CH) * (HCCO_CH2 - CH_CH2 * HCCO_CH) / (1 - CH_CH2 * CH2_CH)) - ((CH2GSG_rhs - CH_rhs * CH2GSG_CH) - (CH2_rhs - CH_rhs * CH2_CH) * (CH2GSG_CH2 - CH_CH2 * CH2GSG_CH) / (1 - CH_CH2 * CH2_CH)) * ((HCCO_CH2GSG - CH_CH2GSG * HCCO_CH) - (CH2_CH2GSG - CH_CH2GSG * CH2_CH) * (HCCO_CH2 - CH_CH2 * HCCO_CH)/ (1 - CH_CH2 * CH2_CH)) / ((1 - CH_CH2GSG * CH2GSG_CH) - (CH2_CH2GSG - CH_CH2GSG * CH2_CH) * (CH2GSG_CH2 - CH_CH2 * CH2GSG_CH)/ (1 - CH_CH2 * CH2_CH))) / (((1 - CH_HCCO * HCCO_CH) - (CH2_HCCO - CH_HCCO * CH2_CH) * (HCCO_CH2 - CH_CH2 * HCCO_CH)/ (1 - CH_CH2 * CH2_CH)) - ((CH2GSG_HCCO - CH_HCCO * CH2GSG_CH) - (CH2_HCCO - CH_HCCO * CH2_CH) * (CH2GSG_CH2 - CH_CH2 * CH2GSG_CH)/ (1 - CH_CH2 * CH2_CH)) * ((HCCO_CH2GSG - CH_CH2GSG * HCCO_CH) - (CH2_CH2GSG - CH_CH2GSG * CH2_CH) * (HCCO_CH2 - CH_CH2 * HCCO_CH)/ (1 - CH_CH2 * CH2_CH))/ ((1 - CH_CH2GSG * CH2GSG_CH) - (CH2_CH2GSG - CH_CH2GSG * CH2_CH) * (CH2GSG_CH2 - CH_CH2 * CH2GSG_CH)/ (1 - CH_CH2 * CH2_CH)))))) / ((1 - CH_CH2GSG * CH2GSG_CH) - (CH2_CH2GSG - CH_CH2GSG * CH2_CH) * (CH2GSG_CH2 - CH_CH2 * CH2GSG_CH)/ (1 - CH_CH2 * CH2_CH))) - CH_CH2 * (((CH2_rhs - CH_rhs * CH2_CH) - ((CH2_HCCO - CH_HCCO * CH2_CH) * ((((HCCO_rhs - CH_rhs * HCCO_CH) - (CH2_rhs - CH_rhs * CH2_CH) * (HCCO_CH2 - CH_CH2 * HCCO_CH) / (1 - CH_CH2 * CH2_CH)) - ((CH2GSG_rhs - CH_rhs * CH2GSG_CH) - (CH2_rhs - CH_rhs * CH2_CH) * (CH2GSG_CH2 - CH_CH2 * CH2GSG_CH) / (1 - CH_CH2 * CH2_CH)) * ((HCCO_CH2GSG - CH_CH2GSG * HCCO_CH) - (CH2_CH2GSG - CH_CH2GSG * CH2_CH) * (HCCO_CH2 - CH_CH2 * HCCO_CH)/ (1 - CH_CH2 * CH2_CH)) / ((1 - CH_CH2GSG * CH2GSG_CH) - (CH2_CH2GSG - CH_CH2GSG * CH2_CH) * (CH2GSG_CH2 - CH_CH2 * CH2GSG_CH)/ (1 - CH_CH2 * CH2_CH))) / (((1 - CH_HCCO * HCCO_CH) - (CH2_HCCO - CH_HCCO * CH2_CH) * (HCCO_CH2 - CH_CH2 * HCCO_CH)/ (1 - CH_CH2 * CH2_CH)) - ((CH2GSG_HCCO - CH_HCCO * CH2GSG_CH) - (CH2_HCCO - CH_HCCO * CH2_CH) * (CH2GSG_CH2 - CH_CH2 * CH2GSG_CH)/ (1 - CH_CH2 * CH2_CH)) * ((HCCO_CH2GSG - CH_CH2GSG * HCCO_CH) - (CH2_CH2GSG - CH_CH2GSG * CH2_CH) * (HCCO_CH2 - CH_CH2 * HCCO_CH)/ (1 - CH_CH2 * CH2_CH))/ ((1 - CH_CH2GSG * CH2GSG_CH) - (CH2_CH2GSG - CH_CH2GSG * CH2_CH) * (CH2GSG_CH2 - CH_CH2 * CH2GSG_CH)/ (1 - CH_CH2 * CH2_CH)))) - (CH2_CH2GSG - CH_CH2GSG * CH2_CH) * ((((CH2GSG_rhs - CH_rhs * CH2GSG_CH) - (CH2_rhs - CH_rhs * CH2_CH) * (CH2GSG_CH2 - CH_CH2 * CH2GSG_CH) / (1 - CH_CH2 * CH2_CH)) - (((CH2GSG_HCCO - CH_HCCO * CH2GSG_CH) - (CH2_HCCO - CH_HCCO * CH2_CH) * (CH2GSG_CH2 - CH_CH2 * CH2GSG_CH)/ (1 - CH_CH2 * CH2_CH)) * ((((HCCO_rhs - CH_rhs * HCCO_CH) - (CH2_rhs - CH_rhs * CH2_CH) * (HCCO_CH2 - CH_CH2 * HCCO_CH) / (1 - CH_CH2 * CH2_CH)) - ((CH2GSG_rhs - CH_rhs * CH2GSG_CH) - (CH2_rhs - CH_rhs * CH2_CH) * (CH2GSG_CH2 - CH_CH2 * CH2GSG_CH) / (1 - CH_CH2 * CH2_CH)) * ((HCCO_CH2GSG - CH_CH2GSG * HCCO_CH) - (CH2_CH2GSG - CH_CH2GSG * CH2_CH) * (HCCO_CH2 - CH_CH2 * HCCO_CH)/ (1 - CH_CH2 * CH2_CH)) / ((1 - CH_CH2GSG * CH2GSG_CH) - (CH2_CH2GSG - CH_CH2GSG * CH2_CH) * (CH2GSG_CH2 - CH_CH2 * CH2GSG_CH)/ (1 - CH_CH2 * CH2_CH))) / (((1 - CH_HCCO * HCCO_CH) - (CH2_HCCO - CH_HCCO * CH2_CH) * (HCCO_CH2 - CH_CH2 * HCCO_CH)/ (1 - CH_CH2 * CH2_CH)) - ((CH2GSG_HCCO - CH_HCCO * CH2GSG_CH) - (CH2_HCCO - CH_HCCO * CH2_CH) * (CH2GSG_CH2 - CH_CH2 * CH2GSG_CH)/ (1 - CH_CH2 * CH2_CH)) * ((HCCO_CH2GSG - CH_CH2GSG * HCCO_CH) - (CH2_CH2GSG - CH_CH2GSG * CH2_CH) * (HCCO_CH2 - CH_CH2 * HCCO_CH)/ (1 - CH_CH2 * CH2_CH))/ ((1 - CH_CH2GSG * CH2GSG_CH) - (CH2_CH2GSG - CH_CH2GSG * CH2_CH) * (CH2GSG_CH2 - CH_CH2 * CH2GSG_CH)/ (1 - CH_CH2 * CH2_CH)))))) / ((1 - CH_CH2GSG * CH2GSG_CH) - (CH2_CH2GSG - CH_CH2GSG * CH2_CH) * (CH2GSG_CH2 - CH_CH2 * CH2GSG_CH)/ (1 - CH_CH2 * CH2_CH))))) / (1 - CH_CH2 * CH2_CH)));
    sc_qss[2] = ((CH2_rhs - CH_rhs * CH2_CH) - ((CH2_HCCO - CH_HCCO * CH2_CH) * ((((HCCO_rhs - CH_rhs * HCCO_CH) - (CH2_rhs - CH_rhs * CH2_CH) * (HCCO_CH2 - CH_CH2 * HCCO_CH) / (1 - CH_CH2 * CH2_CH)) - ((CH2GSG_rhs - CH_rhs * CH2GSG_CH) - (CH2_rhs - CH_rhs * CH2_CH) * (CH2GSG_CH2 - CH_CH2 * CH2GSG_CH) / (1 - CH_CH2 * CH2_CH)) * ((HCCO_CH2GSG - CH_CH2GSG * HCCO_CH) - (CH2_CH2GSG - CH_CH2GSG * CH2_CH) * (HCCO_CH2 - CH_CH2 * HCCO_CH)/ (1 - CH_CH2 * CH2_CH)) / ((1 - CH_CH2GSG * CH2GSG_CH) - (CH2_CH2GSG - CH_CH2GSG * CH2_CH) * (CH2GSG_CH2 - CH_CH2 * CH2GSG_CH)/ (1 - CH_CH2 * CH2_CH))) / (((1 - CH_HCCO * HCCO_CH) - (CH2_HCCO - CH_HCCO * CH2_CH) * (HCCO_CH2 - CH_CH2 * HCCO_CH)/ (1 - CH_CH2 * CH2_CH)) - ((CH2GSG_HCCO - CH_HCCO * CH2GSG_CH) - (CH2_HCCO - CH_HCCO * CH2_CH) * (CH2GSG_CH2 - CH_CH2 * CH2GSG_CH)/ (1 - CH_CH2 * CH2_CH)) * ((HCCO_CH2GSG - CH_CH2GSG * HCCO_CH) - (CH2_CH2GSG - CH_CH2GSG * CH2_CH) * (HCCO_CH2 - CH_CH2 * HCCO_CH)/ (1 - CH_CH2 * CH2_CH))/ ((1 - CH_CH2GSG * CH2GSG_CH) - (CH2_CH2GSG - CH_CH2GSG * CH2_CH) * (CH2GSG_CH2 - CH_CH2 * CH2GSG_CH)/ (1 - CH_CH2 * CH2_CH)))) - (CH2_CH2GSG - CH_CH2GSG * CH2_CH) * ((((CH2GSG_rhs - CH_rhs * CH2GSG_CH) - (CH2_rhs - CH_rhs * CH2_CH) * (CH2GSG_CH2 - CH_CH2 * CH2GSG_CH) / (1 - CH_CH2 * CH2_CH)) - (((CH2GSG_HCCO - CH_HCCO * CH2GSG_CH) - (CH2_HCCO - CH_HCCO * CH2_CH) * (CH2GSG_CH2 - CH_CH2 * CH2GSG_CH)/ (1 - CH_CH2 * CH2_CH)) * ((((HCCO_rhs - CH_rhs * HCCO_CH) - (CH2_rhs - CH_rhs * CH2_CH) * (HCCO_CH2 - CH_CH2 * HCCO_CH) / (1 - CH_CH2 * CH2_CH)) - ((CH2GSG_rhs - CH_rhs * CH2GSG_CH) - (CH2_rhs - CH_rhs * CH2_CH) * (CH2GSG_CH2 - CH_CH2 * CH2GSG_CH) / (1 - CH_CH2 * CH2_CH)) * ((HCCO_CH2GSG - CH_CH2GSG * HCCO_CH) - (CH2_CH2GSG - CH_CH2GSG * CH2_CH) * (HCCO_CH2 - CH_CH2 * HCCO_CH)/ (1 - CH_CH2 * CH2_CH)) / ((1 - CH_CH2GSG * CH2GSG_CH) - (CH2_CH2GSG - CH_CH2GSG * CH2_CH) * (CH2GSG_CH2 - CH_CH2 * CH2GSG_CH)/ (1 - CH_CH2 * CH2_CH))) / (((1 - CH_HCCO * HCCO_CH) - (CH2_HCCO - CH_HCCO * CH2_CH) * (HCCO_CH2 - CH_CH2 * HCCO_CH)/ (1 - CH_CH2 * CH2_CH)) - ((CH2GSG_HCCO - CH_HCCO * CH2GSG_CH) - (CH2_HCCO - CH_HCCO * CH2_CH) * (CH2GSG_CH2 - CH_CH2 * CH2GSG_CH)/ (1 - CH_CH2 * CH2_CH)) * ((HCCO_CH2GSG - CH_CH2GSG * HCCO_CH) - (CH2_CH2GSG - CH_CH2GSG * CH2_CH) * (HCCO_CH2 - CH_CH2 * HCCO_CH)/ (1 - CH_CH2 * CH2_CH))/ ((1 - CH_CH2GSG * CH2GSG_CH) - (CH2_CH2GSG - CH_CH2GSG * CH2_CH) * (CH2GSG_CH2 - CH_CH2 * CH2GSG_CH)/ (1 - CH_CH2 * CH2_CH)))))) / ((1 - CH_CH2GSG * CH2GSG_CH) - (CH2_CH2GSG - CH_CH2GSG * CH2_CH) * (CH2GSG_CH2 - CH_CH2 * CH2GSG_CH)/ (1 - CH_CH2 * CH2_CH))))) / (1 - CH_CH2 * CH2_CH);
    sc_qss[3] = (((CH2GSG_rhs - CH_rhs * CH2GSG_CH) - (CH2_rhs - CH_rhs * CH2_CH) * (CH2GSG_CH2 - CH_CH2 * CH2GSG_CH) / (1 - CH_CH2 * CH2_CH)) - (((CH2GSG_HCCO - CH_HCCO * CH2GSG_CH) - (CH2_HCCO - CH_HCCO * CH2_CH) * (CH2GSG_CH2 - CH_CH2 * CH2GSG_CH)/ (1 - CH_CH2 * CH2_CH)) * ((((HCCO_rhs - CH_rhs * HCCO_CH) - (CH2_rhs - CH_rhs * CH2_CH) * (HCCO_CH2 - CH_CH2 * HCCO_CH) / (1 - CH_CH2 * CH2_CH)) - ((CH2GSG_rhs - CH_rhs * CH2GSG_CH) - (CH2_rhs - CH_rhs * CH2_CH) * (CH2GSG_CH2 - CH_CH2 * CH2GSG_CH) / (1 - CH_CH2 * CH2_CH)) * ((HCCO_CH2GSG - CH_CH2GSG * HCCO_CH) - (CH2_CH2GSG - CH_CH2GSG * CH2_CH) * (HCCO_CH2 - CH_CH2 * HCCO_CH)/ (1 - CH_CH2 * CH2_CH)) / ((1 - CH_CH2GSG * CH2GSG_CH) - (CH2_CH2GSG - CH_CH2GSG * CH2_CH) * (CH2GSG_CH2 - CH_CH2 * CH2GSG_CH)/ (1 - CH_CH2 * CH2_CH))) / (((1 - CH_HCCO * HCCO_CH) - (CH2_HCCO - CH_HCCO * CH2_CH) * (HCCO_CH2 - CH_CH2 * HCCO_CH)/ (1 - CH_CH2 * CH2_CH)) - ((CH2GSG_HCCO - CH_HCCO * CH2GSG_CH) - (CH2_HCCO - CH_HCCO * CH2_CH) * (CH2GSG_CH2 - CH_CH2 * CH2GSG_CH)/ (1 - CH_CH2 * CH2_CH)) * ((HCCO_CH2GSG - CH_CH2GSG * HCCO_CH) - (CH2_CH2GSG - CH_CH2GSG * CH2_CH) * (HCCO_CH2 - CH_CH2 * HCCO_CH)/ (1 - CH_CH2 * CH2_CH))/ ((1 - CH_CH2GSG * CH2GSG_CH) - (CH2_CH2GSG - CH_CH2GSG * CH2_CH) * (CH2GSG_CH2 - CH_CH2 * CH2GSG_CH)/ (1 - CH_CH2 * CH2_CH)))))) / ((1 - CH_CH2GSG * CH2GSG_CH) - (CH2_CH2GSG - CH_CH2GSG * CH2_CH) * (CH2GSG_CH2 - CH_CH2 * CH2GSG_CH)/ (1 - CH_CH2 * CH2_CH));
    sc_qss[7] = (((HCCO_rhs - CH_rhs * HCCO_CH) - (CH2_rhs - CH_rhs * CH2_CH) * (HCCO_CH2 - CH_CH2 * HCCO_CH) / (1 - CH_CH2 * CH2_CH)) - ((CH2GSG_rhs - CH_rhs * CH2GSG_CH) - (CH2_rhs - CH_rhs * CH2_CH) * (CH2GSG_CH2 - CH_CH2 * CH2GSG_CH) / (1 - CH_CH2 * CH2_CH)) * ((HCCO_CH2GSG - CH_CH2GSG * HCCO_CH) - (CH2_CH2GSG - CH_CH2GSG * CH2_CH) * (HCCO_CH2 - CH_CH2 * HCCO_CH)/ (1 - CH_CH2 * CH2_CH)) / ((1 - CH_CH2GSG * CH2GSG_CH) - (CH2_CH2GSG - CH_CH2GSG * CH2_CH) * (CH2GSG_CH2 - CH_CH2 * CH2GSG_CH)/ (1 - CH_CH2 * CH2_CH))) / (((1 - CH_HCCO * HCCO_CH) - (CH2_HCCO - CH_HCCO * CH2_CH) * (HCCO_CH2 - CH_CH2 * HCCO_CH)/ (1 - CH_CH2 * CH2_CH)) - ((CH2GSG_HCCO - CH_HCCO * CH2GSG_CH) - (CH2_HCCO - CH_HCCO * CH2_CH) * (CH2GSG_CH2 - CH_CH2 * CH2GSG_CH)/ (1 - CH_CH2 * CH2_CH)) * ((HCCO_CH2GSG - CH_CH2GSG * HCCO_CH) - (CH2_CH2GSG - CH_CH2GSG * CH2_CH) * (HCCO_CH2 - CH_CH2 * HCCO_CH)/ (1 - CH_CH2 * CH2_CH))/ ((1 - CH_CH2GSG * CH2GSG_CH) - (CH2_CH2GSG - CH_CH2GSG * CH2_CH) * (CH2GSG_CH2 - CH_CH2 * CH2GSG_CH)/ (1 - CH_CH2 * CH2_CH)));



    /*QSS species 1: HCO */

    double HCO_num = epsilon - qf_co[33]*sc_qss[0] - qf_co[73] - qf_co[74] - qf_co[75] - qf_co[76] - qf_co[84] - qf_co[99]*sc_qss[8] - qf_co[101] - qf_co[114]*sc_qss[7] - qf_co[115]*sc_qss[7] - qf_co[131]*sc_qss[13] - qf_co[132] - qf_co[146] - qf_co[150]*sc_qss[5];
    double HCO_denom = epsilon - qf_co[14] - qf_co[67] - qf_co[68] - qf_co[69] - qf_co[70] - qf_co[71] - qf_co[72] - qf_co[85];

    sc_qss[1] = HCO_num/HCO_denom;



    return;
}

/*compute an approx to the reaction Jacobian (for preconditioning) */
AMREX_GPU_HOST_DEVICE void DWDOT_SIMPLIFIED(double *  J, double *  sc, double *  Tp, int * HP)
{
    double c[38];

    for (int k=0; k<38; k++) {
        c[k] = 1.e6 * sc[k];
    }

    aJacobian_precond(J, c, *Tp, *HP);

    /* dwdot[k]/dT */
    /* dTdot/d[X] */
    for (int k=0; k<38; k++) {
        J[1482+k] *= 1.e-6;
        J[k*39+38] *= 1.e6;
    }

    return;
}

/*compute the reaction Jacobian */
AMREX_GPU_HOST_DEVICE void DWDOT(double *  J, double *  sc, double *  Tp, int * consP)
{
    double c[38];

    for (int k=0; k<38; k++) {
        c[k] = 1.e6 * sc[k];
    }

    aJacobian(J, c, *Tp, *consP);

    /* dwdot[k]/dT */
    /* dTdot/d[X] */
    for (int k=0; k<38; k++) {
        J[1482+k] *= 1.e-6;
        J[k*39+38] *= 1.e6;
    }

    return;
}

/*compute the sparsity pattern of the chemistry Jacobian */
AMREX_GPU_HOST_DEVICE void SPARSITY_INFO( int * nJdata, int * consP, int NCELLS)
{
    double c[38];
    double J[1521];

    for (int k=0; k<38; k++) {
        c[k] = 1.0/ 38.000000 ;
    }

    aJacobian(J, c, 1500.0, *consP);

    int nJdata_tmp = 0;
    for (int k=0; k<39; k++) {
        for (int l=0; l<39; l++) {
            if(J[ 39 * k + l] != 0.0){
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
    double c[38];
    double J[1521];

    for (int k=0; k<38; k++) {
        c[k] = 1.0/ 38.000000 ;
    }

    aJacobian(J, c, 1500.0, *consP);

    int nJdata_tmp = 0;
    for (int k=0; k<39; k++) {
        for (int l=0; l<39; l++) {
            if(k == l){
                nJdata_tmp = nJdata_tmp + 1;
            } else {
                if(J[ 39 * k + l] != 0.0){
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
    double c[38];
    double J[1521];

    for (int k=0; k<38; k++) {
        c[k] = 1.0/ 38.000000 ;
    }

    aJacobian_precond(J, c, 1500.0, *consP);

    int nJdata_tmp = 0;
    for (int k=0; k<39; k++) {
        for (int l=0; l<39; l++) {
            if(k == l){
                nJdata_tmp = nJdata_tmp + 1;
            } else {
                if(J[ 39 * k + l] != 0.0){
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
    double c[38];
    double J[1521];
    int offset_row;
    int offset_col;

    for (int k=0; k<38; k++) {
        c[k] = 1.0/ 38.000000 ;
    }

    aJacobian(J, c, 1500.0, *consP);

    colPtrs[0] = 0;
    int nJdata_tmp = 0;
    for (int nc=0; nc<NCELLS; nc++) {
        offset_row = nc * 39;
        offset_col = nc * 39;
        for (int k=0; k<39; k++) {
            for (int l=0; l<39; l++) {
                if(J[39*k + l] != 0.0) {
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
    double c[38];
    double J[1521];
    int offset;

    for (int k=0; k<38; k++) {
        c[k] = 1.0/ 38.000000 ;
    }

    aJacobian(J, c, 1500.0, *consP);

    if (base == 1) {
        rowPtrs[0] = 1;
        int nJdata_tmp = 1;
        for (int nc=0; nc<NCELLS; nc++) {
            offset = nc * 39;
            for (int l=0; l<39; l++) {
                for (int k=0; k<39; k++) {
                    if(J[39*k + l] != 0.0) {
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
            offset = nc * 39;
            for (int l=0; l<39; l++) {
                for (int k=0; k<39; k++) {
                    if(J[39*k + l] != 0.0) {
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
    double c[38];
    double J[1521];
    int offset;

    for (int k=0; k<38; k++) {
        c[k] = 1.0/ 38.000000 ;
    }

    aJacobian(J, c, 1500.0, *consP);

    if (base == 1) {
        rowPtr[0] = 1;
        int nJdata_tmp = 1;
        for (int nc=0; nc<NCELLS; nc++) {
            offset = nc * 39;
            for (int l=0; l<39; l++) {
                for (int k=0; k<39; k++) {
                    if (k == l) {
                        colVals[nJdata_tmp-1] = l+1 + offset; 
                        nJdata_tmp = nJdata_tmp + 1; 
                    } else {
                        if(J[39*k + l] != 0.0) {
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
            offset = nc * 39;
            for (int l=0; l<39; l++) {
                for (int k=0; k<39; k++) {
                    if (k == l) {
                        colVals[nJdata_tmp] = l + offset; 
                        nJdata_tmp = nJdata_tmp + 1; 
                    } else {
                        if(J[39*k + l] != 0.0) {
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
    double c[38];
    double J[1521];

    for (int k=0; k<38; k++) {
        c[k] = 1.0/ 38.000000 ;
    }

    aJacobian_precond(J, c, 1500.0, *consP);

    colPtrs[0] = 0;
    int nJdata_tmp = 0;
    for (int k=0; k<39; k++) {
        for (int l=0; l<39; l++) {
            if (k == l) {
                rowVals[nJdata_tmp] = l; 
                indx[nJdata_tmp] = 39*k + l;
                nJdata_tmp = nJdata_tmp + 1; 
            } else {
                if(J[39*k + l] != 0.0) {
                    rowVals[nJdata_tmp] = l; 
                    indx[nJdata_tmp] = 39*k + l;
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
    double c[38];
    double J[1521];

    for (int k=0; k<38; k++) {
        c[k] = 1.0/ 38.000000 ;
    }

    aJacobian_precond(J, c, 1500.0, *consP);

    if (base == 1) {
        rowPtr[0] = 1;
        int nJdata_tmp = 1;
        for (int l=0; l<39; l++) {
            for (int k=0; k<39; k++) {
                if (k == l) {
                    colVals[nJdata_tmp-1] = l+1; 
                    nJdata_tmp = nJdata_tmp + 1; 
                } else {
                    if(J[39*k + l] != 0.0) {
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
        for (int l=0; l<39; l++) {
            for (int k=0; k<39; k++) {
                if (k == l) {
                    colVals[nJdata_tmp] = l; 
                    nJdata_tmp = nJdata_tmp + 1; 
                } else {
                    if(J[39*k + l] != 0.0) {
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
    for (int i=0; i<1521; i++) {
        J[i] = 0.0;
    }
}
#endif


/*compute an approx to the reaction Jacobian */
AMREX_GPU_HOST_DEVICE void aJacobian_precond(double *  J, double *  sc, double T, int HP)
{
    for (int i=0; i<1521; i++) {
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
        /*species 19: C2H2 */
        species[19] =
            +1.51904500e-02
            -3.23263800e-05 * tc[1]
            +2.72369760e-08 * tc[2]
            -7.65098400e-12 * tc[3];
        /*species 21: C2H4 */
        species[21] =
            +2.79616300e-02
            -6.77735400e-05 * tc[1]
            +8.35545600e-08 * tc[2]
            -3.89515160e-11 * tc[3];
        /*species 22: C3H3 */
        species[22] =
            +1.10802800e-02
            +5.58664600e-07 * tc[1]
            -1.64376360e-08 * tc[2]
            +7.79851600e-12 * tc[3];
        /*species 25: NXC3H7 */
        species[25] =
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
        /*species 19: C2H2 */
        species[19] =
            +5.37603900e-03
            -3.82563400e-06 * tc[1]
            +9.85913700e-10 * tc[2]
            -8.62684000e-14 * tc[3];
        /*species 21: C2H4 */
        species[21] =
            +1.14851800e-02
            -8.83677000e-06 * tc[1]
            +2.35338030e-09 * tc[2]
            -2.10673920e-13 * tc[3];
        /*species 22: C3H3 */
        species[22] =
            +4.35719500e-03
            -8.21813400e-07 * tc[1]
            -7.10616900e-10 * tc[2]
            +1.75060800e-13 * tc[3];
        /*species 25: NXC3H7 */
        species[25] =
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
        /*species 26: NXC3H7O2 */
        species[26] =
            +3.96164986e-02
            -4.98983198e-05 * tc[1]
            +2.57835090e-08 * tc[2]
            -5.24961320e-12 * tc[3];
    } else {
        /*species 15: C2H6 */
        species[15] =
            +1.29236361e-02
            -8.85054392e-06 * tc[1]
            +2.06217518e-09 * tc[2]
            -1.59560693e-13 * tc[3];
        /*species 26: NXC3H7O2 */
        species[26] =
            +1.69910726e-02
            -1.17773375e-05 * tc[1]
            +2.76658619e-09 * tc[2]
            -2.15292270e-13 * tc[3];
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

    /*species with midpoint at T=1385 kelvin */
    if (T < 1385) {
        /*species 18: CH3O2 */
        species[18] =
            +1.00873599e-02
            -6.43012368e-06 * tc[1]
            +6.28227801e-10 * tc[2]
            +1.67335641e-13 * tc[3];
        /*species 31: PXC4H9O2 */
        species[31] =
            +5.15513163e-02
            -6.56568800e-05 * tc[1]
            +3.39194580e-08 * tc[2]
            -6.80474424e-12 * tc[3];
    } else {
        /*species 18: CH3O2 */
        species[18] =
            +7.90728626e-03
            -5.36492468e-06 * tc[1]
            +1.24167401e-09 * tc[2]
            -9.56029320e-14 * tc[3];
        /*species 31: PXC4H9O2 */
        species[31] =
            +2.15210910e-02
            -1.48981803e-05 * tc[1]
            +3.49674213e-09 * tc[2]
            -2.71954244e-13 * tc[3];
    }

    /*species with midpoint at T=1388 kelvin */
    if (T < 1388) {
        /*species 20: C3H6 */
        species[20] =
            +2.89107662e-02
            -3.09773616e-05 * tc[1]
            +1.16644263e-08 * tc[2]
            -1.35156141e-12 * tc[3];
    } else {
        /*species 20: C3H6 */
        species[20] =
            +1.37023634e-02
            -9.32499466e-06 * tc[1]
            +2.16376321e-09 * tc[2]
            -1.66948050e-13 * tc[3];
    }

    /*species with midpoint at T=1400 kelvin */
    if (T < 1400) {
        /*species 23: C3H4XA */
        species[23] =
            +1.63343700e-02
            -3.52990000e-06 * tc[1]
            -1.39420950e-08 * tc[2]
            +6.91652400e-12 * tc[3];
    } else {
        /*species 23: C3H4XA */
        species[23] =
            +5.30213800e-03
            -7.40223600e-07 * tc[1]
            -9.07915800e-10 * tc[2]
            +2.03583240e-13 * tc[3];
    }

    /*species with midpoint at T=1397 kelvin */
    if (T < 1397) {
        /*species 24: C3H5XA */
        species[24] =
            +3.34559100e-02
            -5.06802054e-05 * tc[1]
            +3.08597262e-08 * tc[2]
            -6.93033360e-12 * tc[3];
        /*species 34: C5H11X1 */
        species[34] =
            +6.10632852e-02
            -8.18983650e-05 * tc[1]
            +4.38280410e-08 * tc[2]
            -8.75438460e-12 * tc[3];
    } else {
        /*species 24: C3H5XA */
        species[24] =
            +1.12695483e-02
            -7.67585728e-06 * tc[1]
            +1.78217736e-09 * tc[2]
            -1.37567212e-13 * tc[3];
        /*species 34: C5H11X1 */
        species[34] =
            +2.39041200e-02
            -1.62954324e-05 * tc[1]
            +3.78528708e-09 * tc[2]
            -2.92270934e-13 * tc[3];
    }

    /*species with midpoint at T=1398 kelvin */
    if (T < 1398) {
        /*species 27: C4H6 */
        species[27] =
            +4.78706062e-02
            -8.30893600e-05 * tc[1]
            +5.74648656e-08 * tc[2]
            -1.42863403e-11 * tc[3];
    } else {
        /*species 27: C4H6 */
        species[27] =
            +1.37163965e-02
            -9.39431566e-06 * tc[1]
            +2.18908151e-09 * tc[2]
            -1.69394481e-13 * tc[3];
    }

    /*species with midpoint at T=1392 kelvin */
    if (T < 1392) {
        /*species 28: C4H7 */
        species[28] =
            +4.26511243e-02
            -5.81958746e-05 * tc[1]
            +3.16211742e-08 * tc[2]
            -6.40239416e-12 * tc[3];
        /*species 29: C4H8X1 */
        species[29] =
            +4.52580978e-02
            -5.87317118e-05 * tc[1]
            +3.00661308e-08 * tc[2]
            -5.72766720e-12 * tc[3];
        /*species 32: C5H9 */
        species[32] =
            +5.57608487e-02
            -7.40287856e-05 * tc[1]
            +3.80651703e-08 * tc[2]
            -7.14155340e-12 * tc[3];
        /*species 33: C5H10X1 */
        species[33] =
            +5.74218294e-02
            -7.48973780e-05 * tc[1]
            +3.82094967e-08 * tc[2]
            -7.18439156e-12 * tc[3];
        /*species 35: C6H12X1 */
        species[35] =
            +6.98655426e-02
            -9.18816044e-05 * tc[1]
            +4.70902029e-08 * tc[2]
            -8.85184700e-12 * tc[3];
    } else {
        /*species 28: C4H7 */
        species[28] =
            +1.60483196e-02
            -1.09300458e-05 * tc[1]
            +2.53782316e-09 * tc[2]
            -1.95909096e-13 * tc[3];
        /*species 29: C4H8X1 */
        species[29] =
            +1.80617877e-02
            -1.23218606e-05 * tc[1]
            +2.86395888e-09 * tc[2]
            -2.21235856e-13 * tc[3];
        /*species 32: C5H9 */
        species[32] =
            +2.07128899e-02
            -1.41392123e-05 * tc[1]
            +3.28821399e-09 * tc[2]
            -2.54128883e-13 * tc[3];
        /*species 33: C5H10X1 */
        species[33] =
            +2.24072471e-02
            -1.52669605e-05 * tc[1]
            +3.54566898e-09 * tc[2]
            -2.73754056e-13 * tc[3];
        /*species 35: C6H12X1 */
        species[35] =
            +2.67377658e-02
            -1.82007355e-05 * tc[1]
            +4.22459304e-09 * tc[2]
            -3.26049698e-13 * tc[3];
    }

    /*species with midpoint at T=1395 kelvin */
    if (T < 1395) {
        /*species 30: PXC4H9 */
        species[30] =
            +4.78972364e-02
            -6.28046318e-05 * tc[1]
            +3.29359416e-08 * tc[2]
            -6.48042656e-12 * tc[3];
    } else {
        /*species 30: PXC4H9 */
        species[30] =
            +1.94310717e-02
            -1.32315590e-05 * tc[1]
            +3.07125408e-09 * tc[2]
            -2.37011883e-13 * tc[3];
    }

    /*species with midpoint at T=1382 kelvin */
    if (T < 1382) {
        /*species 36: C7H15X2 */
        species[36] =
            +7.56726570e-02
            -8.14947268e-05 * tc[1]
            +2.79803683e-08 * tc[2]
            -1.96944298e-12 * tc[3];
    } else {
        /*species 36: C7H15X2 */
        species[36] =
            +3.23324804e-02
            -2.18547614e-05 * tc[1]
            +5.05071180e-09 * tc[2]
            -3.88709636e-13 * tc[3];
    }

    /*species with midpoint at T=1391 kelvin */
    if (T < 1391) {
        /*species 37: NXC7H16 */
        species[37] =
            +8.54355820e-02
            -1.05069357e-04 * tc[1]
            +4.88837163e-08 * tc[2]
            -8.09579700e-12 * tc[3];
    } else {
        /*species 37: NXC7H16 */
        species[37] =
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
    double g_RT_qss[14];
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
    kc[8] = refC * exp((g_RT_qss[8]) - (g_RT[3] + g_RT[19]));

    /*reaction 10: H + C2H4 (+M) <=> C2H5 (+M) */
    kc[9] = 1.0 / (refC) * exp((g_RT[3] + g_RT[21]) - (g_RT_qss[5]));

    /*reaction 11: CH3CO (+M) => CH3 + CO (+M) */
    kc[10] = refC * exp((g_RT_qss[11]) - (g_RT[12] + g_RT[10]));

    /*reaction 12: H + OH + M => H2O + M */
    kc[11] = 1.0 / (refC) * exp((g_RT[3] + g_RT[4]) - (g_RT[5]));

    /*reaction 13: CH2GSG + M => CH2 + M */
    kc[12] = exp((g_RT_qss[3]) - (g_RT_qss[2]));

    /*reaction 14: CH2 + M => CH2GSG + M */
    kc[13] = exp((g_RT_qss[2]) - (g_RT_qss[3]));

    /*reaction 15: HCO + M => H + CO + M */
    kc[14] = refC * exp((g_RT_qss[1]) - (g_RT[3] + g_RT[10]));

    /*reaction 16: CH3O2 + M => CH3 + O2 + M */
    kc[15] = refC * exp((g_RT[18]) - (g_RT[12] + g_RT[6]));

    /*reaction 17: CH3 + O2 + M => CH3O2 + M */
    kc[16] = 1.0 / (refC) * exp((g_RT[12] + g_RT[6]) - (g_RT[18]));

    /*reaction 18: C2H5O + M => CH3 + CH2O + M */
    kc[17] = refC * exp((g_RT_qss[10]) - (g_RT[12] + g_RT[11]));

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

    /*reaction 89: 2.000000 CH3O2 => O2 + 2.000000 CH3O */
    kc[88] = refC * exp((2.000000 * g_RT[18]) - (g_RT[6] + 2.000000 * g_RT_qss[4]));

    /*reaction 90: CH3O2 + CH3 => 2.000000 CH3O */
    kc[89] = exp((g_RT[18] + g_RT[12]) - (2.000000 * g_RT_qss[4]));

    /*reaction 91: 2.000000 CH3O2 => CH2O + CH3OH + O2 */
    kc[90] = refC * exp((2.000000 * g_RT[18]) - (g_RT[11] + g_RT[14] + g_RT[6]));

    /*reaction 92: CH3O2 + HO2 => CH3O2H + O2 */
    kc[91] = exp((g_RT[18] + g_RT[7]) - (g_RT_qss[6] + g_RT[6]));

    /*reaction 93: CH3O2H => CH3O + OH */
    kc[92] = refC * exp((g_RT_qss[6]) - (g_RT_qss[4] + g_RT[4]));

    /*reaction 94: C2H2 + O => CH2 + CO */
    kc[93] = exp((g_RT[19] + g_RT[1]) - (g_RT_qss[2] + g_RT[10]));

    /*reaction 95: C2H2 + O => HCCO + H */
    kc[94] = exp((g_RT[19] + g_RT[1]) - (g_RT_qss[7] + g_RT[3]));

    /*reaction 96: C2H3 + H => C2H2 + H2 */
    kc[95] = exp((g_RT_qss[8] + g_RT[3]) - (g_RT[19] + g_RT[2]));

    /*reaction 97: C2H3 + O2 => CH2CHO + O */
    kc[96] = exp((g_RT_qss[8] + g_RT[6]) - (g_RT_qss[9] + g_RT[1]));

    /*reaction 98: C2H3 + CH3 => C3H6 */
    kc[97] = 1.0 / (refC) * exp((g_RT_qss[8] + g_RT[12]) - (g_RT[20]));

    /*reaction 99: C2H3 + O2 => C2H2 + HO2 */
    kc[98] = exp((g_RT_qss[8] + g_RT[6]) - (g_RT[19] + g_RT[7]));

    /*reaction 100: C2H3 + O2 => CH2O + HCO */
    kc[99] = exp((g_RT_qss[8] + g_RT[6]) - (g_RT[11] + g_RT_qss[1]));

    /*reaction 101: C2H4 + CH3 => C2H3 + CH4 */
    kc[100] = exp((g_RT[21] + g_RT[12]) - (g_RT_qss[8] + g_RT[13]));

    /*reaction 102: C2H4 + O => CH3 + HCO */
    kc[101] = exp((g_RT[21] + g_RT[1]) - (g_RT[12] + g_RT_qss[1]));

    /*reaction 103: C2H4 + OH => C2H3 + H2O */
    kc[102] = exp((g_RT[21] + g_RT[4]) - (g_RT_qss[8] + g_RT[5]));

    /*reaction 104: C2H4 + O => CH2CHO + H */
    kc[103] = exp((g_RT[21] + g_RT[1]) - (g_RT_qss[9] + g_RT[3]));

    /*reaction 105: C2H4 + H => C2H3 + H2 */
    kc[104] = exp((g_RT[21] + g_RT[3]) - (g_RT_qss[8] + g_RT[2]));

    /*reaction 106: C2H3 + H2 => C2H4 + H */
    kc[105] = exp((g_RT_qss[8] + g_RT[2]) - (g_RT[21] + g_RT[3]));

    /*reaction 107: H + C2H5 => C2H6 */
    kc[106] = 1.0 / (refC) * exp((g_RT[3] + g_RT_qss[5]) - (g_RT[15]));

    /*reaction 108: CH3O2 + C2H5 => CH3O + C2H5O */
    kc[107] = exp((g_RT[18] + g_RT_qss[5]) - (g_RT_qss[4] + g_RT_qss[10]));

    /*reaction 109: C2H5 + HO2 => C2H5O + OH */
    kc[108] = exp((g_RT_qss[5] + g_RT[7]) - (g_RT_qss[10] + g_RT[4]));

    /*reaction 110: C2H5 + O2 => C2H4 + HO2 */
    kc[109] = exp((g_RT_qss[5] + g_RT[6]) - (g_RT[21] + g_RT[7]));

    /*reaction 111: C2H6 + O => C2H5 + OH */
    kc[110] = exp((g_RT[15] + g_RT[1]) - (g_RT_qss[5] + g_RT[4]));

    /*reaction 112: C2H6 + OH => C2H5 + H2O */
    kc[111] = exp((g_RT[15] + g_RT[4]) - (g_RT_qss[5] + g_RT[5]));

    /*reaction 113: C2H6 + H => C2H5 + H2 */
    kc[112] = exp((g_RT[15] + g_RT[3]) - (g_RT_qss[5] + g_RT[2]));

    /*reaction 114: HCCO + O => H + 2.000000 CO */
    kc[113] = refC * exp((g_RT_qss[7] + g_RT[1]) - (g_RT[3] + 2.000000 * g_RT[10]));

    /*reaction 115: HCCO + OH => 2.000000 HCO */
    kc[114] = exp((g_RT_qss[7] + g_RT[4]) - (2.000000 * g_RT_qss[1]));

    /*reaction 116: HCCO + O2 => CO2 + HCO */
    kc[115] = exp((g_RT_qss[7] + g_RT[6]) - (g_RT[9] + g_RT_qss[1]));

    /*reaction 117: HCCO + H => CH2GSG + CO */
    kc[116] = exp((g_RT_qss[7] + g_RT[3]) - (g_RT_qss[3] + g_RT[10]));

    /*reaction 118: CH2GSG + CO => HCCO + H */
    kc[117] = exp((g_RT_qss[3] + g_RT[10]) - (g_RT_qss[7] + g_RT[3]));

    /*reaction 119: CH2CO + O => HCCO + OH */
    kc[118] = exp((g_RT[16] + g_RT[1]) - (g_RT_qss[7] + g_RT[4]));

    /*reaction 120: CH2CO + H => HCCO + H2 */
    kc[119] = exp((g_RT[16] + g_RT[3]) - (g_RT_qss[7] + g_RT[2]));

    /*reaction 121: HCCO + H2 => CH2CO + H */
    kc[120] = exp((g_RT_qss[7] + g_RT[2]) - (g_RT[16] + g_RT[3]));

    /*reaction 122: CH2CO + H => CH3 + CO */
    kc[121] = exp((g_RT[16] + g_RT[3]) - (g_RT[12] + g_RT[10]));

    /*reaction 123: CH2CO + O => CH2 + CO2 */
    kc[122] = exp((g_RT[16] + g_RT[1]) - (g_RT_qss[2] + g_RT[9]));

    /*reaction 124: CH2CO + OH => HCCO + H2O */
    kc[123] = exp((g_RT[16] + g_RT[4]) - (g_RT_qss[7] + g_RT[5]));

    /*reaction 125: CH2CHO + O2 => CH2O + CO + OH */
    kc[124] = refC * exp((g_RT_qss[9] + g_RT[6]) - (g_RT[11] + g_RT[10] + g_RT[4]));

    /*reaction 126: CH2CHO => CH2CO + H */
    kc[125] = refC * exp((g_RT_qss[9]) - (g_RT[16] + g_RT[3]));

    /*reaction 127: CH2CO + H => CH2CHO */
    kc[126] = 1.0 / (refC) * exp((g_RT[16] + g_RT[3]) - (g_RT_qss[9]));

    /*reaction 128: C2H5O2 => C2H5 + O2 */
    kc[127] = refC * exp((g_RT_qss[12]) - (g_RT_qss[5] + g_RT[6]));

    /*reaction 129: C2H5 + O2 => C2H5O2 */
    kc[128] = 1.0 / (refC) * exp((g_RT_qss[5] + g_RT[6]) - (g_RT_qss[12]));

    /*reaction 130: C2H5O2 => C2H4 + HO2 */
    kc[129] = refC * exp((g_RT_qss[12]) - (g_RT[21] + g_RT[7]));

    /*reaction 131: C3H2 + O2 => HCCO + CO + H */
    kc[130] = refC * exp((g_RT_qss[13] + g_RT[6]) - (g_RT_qss[7] + g_RT[10] + g_RT[3]));

    /*reaction 132: C3H2 + OH => C2H2 + HCO */
    kc[131] = exp((g_RT_qss[13] + g_RT[4]) - (g_RT[19] + g_RT_qss[1]));

    /*reaction 133: C3H3 + O2 => CH2CO + HCO */
    kc[132] = exp((g_RT[22] + g_RT[6]) - (g_RT[16] + g_RT_qss[1]));

    /*reaction 134: C3H3 + HO2 => C3H4XA + O2 */
    kc[133] = exp((g_RT[22] + g_RT[7]) - (g_RT[23] + g_RT[6]));

    /*reaction 135: C3H3 + H => C3H2 + H2 */
    kc[134] = exp((g_RT[22] + g_RT[3]) - (g_RT_qss[13] + g_RT[2]));

    /*reaction 136: C3H3 + OH => C3H2 + H2O */
    kc[135] = exp((g_RT[22] + g_RT[4]) - (g_RT_qss[13] + g_RT[5]));

    /*reaction 137: C3H2 + H2O => C3H3 + OH */
    kc[136] = exp((g_RT_qss[13] + g_RT[5]) - (g_RT[22] + g_RT[4]));

    /*reaction 138: C3H4XA + H => C3H3 + H2 */
    kc[137] = exp((g_RT[23] + g_RT[3]) - (g_RT[22] + g_RT[2]));

    /*reaction 139: C3H4XA + OH => C3H3 + H2O */
    kc[138] = exp((g_RT[23] + g_RT[4]) - (g_RT[22] + g_RT[5]));

    /*reaction 140: C3H4XA + O => C2H4 + CO */
    kc[139] = exp((g_RT[23] + g_RT[1]) - (g_RT[21] + g_RT[10]));

    /*reaction 141: C3H5XA + H => C3H4XA + H2 */
    kc[140] = exp((g_RT[24] + g_RT[3]) - (g_RT[23] + g_RT[2]));

    /*reaction 142: C3H5XA + HO2 => C3H6 + O2 */
    kc[141] = exp((g_RT[24] + g_RT[7]) - (g_RT[20] + g_RT[6]));

    /*reaction 143: C3H5XA + H => C3H6 */
    kc[142] = 1.0 / (refC) * exp((g_RT[24] + g_RT[3]) - (g_RT[20]));

    /*reaction 144: C3H5XA => C2H2 + CH3 */
    kc[143] = refC * exp((g_RT[24]) - (g_RT[19] + g_RT[12]));

    /*reaction 145: C3H5XA => C3H4XA + H */
    kc[144] = refC * exp((g_RT[24]) - (g_RT[23] + g_RT[3]));

    /*reaction 146: C3H4XA + H => C3H5XA */
    kc[145] = 1.0 / (refC) * exp((g_RT[23] + g_RT[3]) - (g_RT[24]));

    /*reaction 147: C3H5XA + CH2O => C3H6 + HCO */
    kc[146] = exp((g_RT[24] + g_RT[11]) - (g_RT[20] + g_RT_qss[1]));

    /*reaction 148: 2.000000 C3H5XA => C3H4XA + C3H6 */
    kc[147] = exp((2.000000 * g_RT[24]) - (g_RT[23] + g_RT[20]));

    /*reaction 149: C3H6 + H => C2H4 + CH3 */
    kc[148] = exp((g_RT[20] + g_RT[3]) - (g_RT[21] + g_RT[12]));

    /*reaction 150: C3H6 + H => C3H5XA + H2 */
    kc[149] = exp((g_RT[20] + g_RT[3]) - (g_RT[24] + g_RT[2]));

    /*reaction 151: C3H6 + O => C2H5 + HCO */
    kc[150] = exp((g_RT[20] + g_RT[1]) - (g_RT_qss[5] + g_RT_qss[1]));

    /*reaction 152: C3H6 + O => C3H5XA + OH */
    kc[151] = exp((g_RT[20] + g_RT[1]) - (g_RT[24] + g_RT[4]));

    /*reaction 153: C3H6 + O => CH2CO + CH3 + H */
    kc[152] = refC * exp((g_RT[20] + g_RT[1]) - (g_RT[16] + g_RT[12] + g_RT[3]));

    /*reaction 154: C3H6 + OH => C3H5XA + H2O */
    kc[153] = exp((g_RT[20] + g_RT[4]) - (g_RT[24] + g_RT[5]));

    /*reaction 155: NXC3H7 + O2 => C3H6 + HO2 */
    kc[154] = exp((g_RT[25] + g_RT[6]) - (g_RT[20] + g_RT[7]));

    /*reaction 156: NXC3H7 => CH3 + C2H4 */
    kc[155] = refC * exp((g_RT[25]) - (g_RT[12] + g_RT[21]));

    /*reaction 157: CH3 + C2H4 => NXC3H7 */
    kc[156] = 1.0 / (refC) * exp((g_RT[12] + g_RT[21]) - (g_RT[25]));

    /*reaction 158: NXC3H7 => H + C3H6 */
    kc[157] = refC * exp((g_RT[25]) - (g_RT[3] + g_RT[20]));

    /*reaction 159: H + C3H6 => NXC3H7 */
    kc[158] = 1.0 / (refC) * exp((g_RT[3] + g_RT[20]) - (g_RT[25]));

    /*reaction 160: NXC3H7O2 => NXC3H7 + O2 */
    kc[159] = refC * exp((g_RT[26]) - (g_RT[25] + g_RT[6]));

    /*reaction 161: NXC3H7 + O2 => NXC3H7O2 */
    kc[160] = 1.0 / (refC) * exp((g_RT[25] + g_RT[6]) - (g_RT[26]));

    /*reaction 162: C4H6 => 2.000000 C2H3 */
    kc[161] = refC * exp((g_RT[27]) - (2.000000 * g_RT_qss[8]));

    /*reaction 163: 2.000000 C2H3 => C4H6 */
    kc[162] = 1.0 / (refC) * exp((2.000000 * g_RT_qss[8]) - (g_RT[27]));

    /*reaction 164: C4H6 + OH => CH2O + C3H5XA */
    kc[163] = exp((g_RT[27] + g_RT[4]) - (g_RT[11] + g_RT[24]));

    /*reaction 165: C4H6 + OH => C2H5 + CH2CO */
    kc[164] = exp((g_RT[27] + g_RT[4]) - (g_RT_qss[5] + g_RT[16]));

    /*reaction 166: C4H6 + O => C2H4 + CH2CO */
    kc[165] = exp((g_RT[27] + g_RT[1]) - (g_RT[21] + g_RT[16]));

    /*reaction 167: C4H6 + H => C2H3 + C2H4 */
    kc[166] = exp((g_RT[27] + g_RT[3]) - (g_RT_qss[8] + g_RT[21]));

    /*reaction 168: C4H6 + O => CH2O + C3H4XA */
    kc[167] = exp((g_RT[27] + g_RT[1]) - (g_RT[11] + g_RT[23]));

    /*reaction 169: H + C4H7 => C4H8X1 */
    kc[168] = 1.0 / (refC) * exp((g_RT[3] + g_RT[28]) - (g_RT[29]));

    /*reaction 170: C3H5XA + C4H7 => C3H6 + C4H6 */
    kc[169] = exp((g_RT[24] + g_RT[28]) - (g_RT[20] + g_RT[27]));

    /*reaction 171: C2H5 + C4H7 => C4H6 + C2H6 */
    kc[170] = exp((g_RT_qss[5] + g_RT[28]) - (g_RT[27] + g_RT[15]));

    /*reaction 172: C4H7 => C4H6 + H */
    kc[171] = refC * exp((g_RT[28]) - (g_RT[27] + g_RT[3]));

    /*reaction 173: C4H6 + H => C4H7 */
    kc[172] = 1.0 / (refC) * exp((g_RT[27] + g_RT[3]) - (g_RT[28]));

    /*reaction 174: C4H7 + CH3 => C4H6 + CH4 */
    kc[173] = exp((g_RT[28] + g_RT[12]) - (g_RT[27] + g_RT[13]));

    /*reaction 175: C4H7 + HO2 => C4H8X1 + O2 */
    kc[174] = exp((g_RT[28] + g_RT[7]) - (g_RT[29] + g_RT[6]));

    /*reaction 176: C4H7 + O2 => C4H6 + HO2 */
    kc[175] = exp((g_RT[28] + g_RT[6]) - (g_RT[27] + g_RT[7]));

    /*reaction 177: C4H7 => C2H4 + C2H3 */
    kc[176] = refC * exp((g_RT[28]) - (g_RT[21] + g_RT_qss[8]));

    /*reaction 178: H + C4H7 => C4H6 + H2 */
    kc[177] = exp((g_RT[3] + g_RT[28]) - (g_RT[27] + g_RT[2]));

    /*reaction 179: C4H8X1 + H => C4H7 + H2 */
    kc[178] = exp((g_RT[29] + g_RT[3]) - (g_RT[28] + g_RT[2]));

    /*reaction 180: C4H8X1 + OH => NXC3H7 + CH2O */
    kc[179] = exp((g_RT[29] + g_RT[4]) - (g_RT[25] + g_RT[11]));

    /*reaction 181: C4H8X1 + OH => CH3CO + C2H6 */
    kc[180] = exp((g_RT[29] + g_RT[4]) - (g_RT_qss[11] + g_RT[15]));

    /*reaction 182: C4H8X1 + O => CH3CO + C2H5 */
    kc[181] = exp((g_RT[29] + g_RT[1]) - (g_RT_qss[11] + g_RT_qss[5]));

    /*reaction 183: C4H8X1 + O => C3H6 + CH2O */
    kc[182] = exp((g_RT[29] + g_RT[1]) - (g_RT[20] + g_RT[11]));

    /*reaction 184: C4H8X1 + OH => C4H7 + H2O */
    kc[183] = exp((g_RT[29] + g_RT[4]) - (g_RT[28] + g_RT[5]));

    /*reaction 185: C4H8X1 => C3H5XA + CH3 */
    kc[184] = refC * exp((g_RT[29]) - (g_RT[24] + g_RT[12]));

    /*reaction 186: C3H5XA + CH3 => C4H8X1 */
    kc[185] = 1.0 / (refC) * exp((g_RT[24] + g_RT[12]) - (g_RT[29]));

    /*reaction 187: PXC4H9 => C4H8X1 + H */
    kc[186] = refC * exp((g_RT[30]) - (g_RT[29] + g_RT[3]));

    /*reaction 188: C4H8X1 + H => PXC4H9 */
    kc[187] = 1.0 / (refC) * exp((g_RT[29] + g_RT[3]) - (g_RT[30]));

    /*reaction 189: PXC4H9 => C2H5 + C2H4 */
    kc[188] = refC * exp((g_RT[30]) - (g_RT_qss[5] + g_RT[21]));

    /*reaction 190: PXC4H9O2 => PXC4H9 + O2 */
    kc[189] = refC * exp((g_RT[31]) - (g_RT[30] + g_RT[6]));

    /*reaction 191: PXC4H9 + O2 => PXC4H9O2 */
    kc[190] = 1.0 / (refC) * exp((g_RT[30] + g_RT[6]) - (g_RT[31]));

    /*reaction 192: C5H9 => C4H6 + CH3 */
    kc[191] = refC * exp((g_RT[32]) - (g_RT[27] + g_RT[12]));

    /*reaction 193: C5H9 => C3H5XA + C2H4 */
    kc[192] = refC * exp((g_RT[32]) - (g_RT[24] + g_RT[21]));

    /*reaction 194: C5H10X1 + OH => C5H9 + H2O */
    kc[193] = exp((g_RT[33] + g_RT[4]) - (g_RT[32] + g_RT[5]));

    /*reaction 195: C5H10X1 + H => C5H9 + H2 */
    kc[194] = exp((g_RT[33] + g_RT[3]) - (g_RT[32] + g_RT[2]));

    /*reaction 196: C5H10X1 => C2H5 + C3H5XA */
    kc[195] = refC * exp((g_RT[33]) - (g_RT_qss[5] + g_RT[24]));

    /*reaction 197: C2H5 + C3H5XA => C5H10X1 */
    kc[196] = 1.0 / (refC) * exp((g_RT_qss[5] + g_RT[24]) - (g_RT[33]));

    /*reaction 198: C5H10X1 + O => C5H9 + OH */
    kc[197] = exp((g_RT[33] + g_RT[1]) - (g_RT[32] + g_RT[4]));

    /*reaction 199: C5H11X1 => C3H6 + C2H5 */
    kc[198] = refC * exp((g_RT[34]) - (g_RT[20] + g_RT_qss[5]));

    /*reaction 200: C5H11X1 => C2H4 + NXC3H7 */
    kc[199] = refC * exp((g_RT[34]) - (g_RT[21] + g_RT[25]));

    /*reaction 201: C5H11X1 <=> C5H10X1 + H */
    kc[200] = refC * exp((g_RT[34]) - (g_RT[33] + g_RT[3]));

    /*reaction 202: C6H12X1 => NXC3H7 + C3H5XA */
    kc[201] = refC * exp((g_RT[35]) - (g_RT[25] + g_RT[24]));

    /*reaction 203: C6H12X1 + OH => C5H11X1 + CH2O */
    kc[202] = exp((g_RT[35] + g_RT[4]) - (g_RT[34] + g_RT[11]));

    /*reaction 204: C7H15X2 => C6H12X1 + CH3 */
    kc[203] = refC * exp((g_RT[36]) - (g_RT[35] + g_RT[12]));

    /*reaction 205: C7H15X2 => PXC4H9 + C3H6 */
    kc[204] = refC * exp((g_RT[36]) - (g_RT[30] + g_RT[20]));

    /*reaction 206: C7H15X2 => C4H8X1 + NXC3H7 */
    kc[205] = refC * exp((g_RT[36]) - (g_RT[29] + g_RT[25]));

    /*reaction 207: C7H15X2 => C5H11X1 + C2H4 */
    kc[206] = refC * exp((g_RT[36]) - (g_RT[34] + g_RT[21]));

    /*reaction 208: C7H15X2 => C2H5 + C5H10X1 */
    kc[207] = refC * exp((g_RT[36]) - (g_RT_qss[5] + g_RT[33]));

    /*reaction 209: C7H15X2 + HO2 => NXC7H16 + O2 */
    kc[208] = exp((g_RT[36] + g_RT[7]) - (g_RT[37] + g_RT[6]));

    /*reaction 210: NXC7H16 + CH3O2 => C7H15X2 + CH3O2H */
    kc[209] = exp((g_RT[37] + g_RT[18]) - (g_RT[36] + g_RT_qss[6]));

    /*reaction 211: NXC7H16 + H => C7H15X2 + H2 */
    kc[210] = exp((g_RT[37] + g_RT[3]) - (g_RT[36] + g_RT[2]));

    /*reaction 212: NXC7H16 => PXC4H9 + NXC3H7 */
    kc[211] = refC * exp((g_RT[37]) - (g_RT[30] + g_RT[25]));

    /*reaction 213: NXC7H16 + HO2 => C7H15X2 + H2O2 */
    kc[212] = exp((g_RT[37] + g_RT[7]) - (g_RT[36] + g_RT[8]));

    /*reaction 214: NXC7H16 => C5H11X1 + C2H5 */
    kc[213] = refC * exp((g_RT[37]) - (g_RT[34] + g_RT_qss[5]));

    /*reaction 215: NXC7H16 + CH3O => C7H15X2 + CH3OH */
    kc[214] = exp((g_RT[37] + g_RT_qss[4]) - (g_RT[36] + g_RT[14]));

    /*reaction 216: NXC7H16 + O => C7H15X2 + OH */
    kc[215] = exp((g_RT[37] + g_RT[1]) - (g_RT[36] + g_RT[4]));

    /*reaction 217: NXC7H16 + OH => C7H15X2 + H2O */
    kc[216] = exp((g_RT[37] + g_RT[4]) - (g_RT[36] + g_RT[5]));

    /*reaction 218: NXC7H16 + CH3 => C7H15X2 + CH4 */
    kc[217] = exp((g_RT[37] + g_RT[12]) - (g_RT[36] + g_RT[13]));

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
        /*species 19: C2H2 */
        species[19] =
            +2.612444000000000e+04 * invT
            -6.791815999999999e+00
            -2.013562000000000e+00 * tc[0]
            -7.595225000000000e-03 * tc[1]
            +2.693865000000000e-06 * tc[2]
            -7.565826666666667e-10 * tc[3]
            +9.563730000000000e-14 * tc[4];
        /*species 21: C2H4 */
        species[21] =
            +5.573046000000000e+03 * invT
            -2.507297800000000e+01
            +8.614880000000000e-01 * tc[0]
            -1.398081500000000e-02 * tc[1]
            +5.647795000000000e-06 * tc[2]
            -2.320960000000000e-09 * tc[3]
            +4.868939500000000e-13 * tc[4];
        /*species 22: C3H3 */
        species[22] =
            +3.988883000000000e+04 * invT
            +4.168745100000000e+00
            -4.754200000000000e+00 * tc[0]
            -5.540140000000000e-03 * tc[1]
            -4.655538333333333e-08 * tc[2]
            +4.566010000000000e-10 * tc[3]
            -9.748145000000000e-14 * tc[4];
        /*species 25: NXC3H7 */
        species[25] =
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
        /*species 19: C2H2 */
        species[19] =
            +2.566766000000000e+04 * invT
            +7.237108000000000e+00
            -4.436770000000000e+00 * tc[0]
            -2.688019500000000e-03 * tc[1]
            +3.188028333333333e-07 * tc[2]
            -2.738649166666667e-11 * tc[3]
            +1.078355000000000e-15 * tc[4];
        /*species 21: C2H4 */
        species[21] =
            +4.428289000000000e+03 * invT
            +1.298030000000000e+00
            -3.528419000000000e+00 * tc[0]
            -5.742590000000000e-03 * tc[1]
            +7.363975000000000e-07 * tc[2]
            -6.537167500000001e-11 * tc[3]
            +2.633424000000000e-15 * tc[4];
        /*species 22: C3H3 */
        species[22] =
            +3.847420000000000e+04 * invT
            +3.061023700000000e+01
            -8.831047000000000e+00 * tc[0]
            -2.178597500000000e-03 * tc[1]
            +6.848445000000000e-08 * tc[2]
            +1.973935833333333e-11 * tc[3]
            -2.188260000000000e-15 * tc[4];
        /*species 25: NXC3H7 */
        species[25] =
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
        /*species 26: NXC3H7O2 */
        species[26] =
            -7.937455670000000e+03 * invT
            -1.680095988000000e+01
            -2.107314920000000e+00 * tc[0]
            -1.980824930000000e-02 * tc[1]
            +4.158193316666667e-06 * tc[2]
            -7.162085833333334e-10 * tc[3]
            +6.562016500000000e-14 * tc[4];
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
        /*species 26: NXC3H7O2 */
        species[26] =
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

    /*species with midpoint at T=1385 kelvin */
    if (T < 1385) {
        /*species 18: CH3O2 */
        species[18] =
            -6.843942590000000e+02 * invT
            -9.018341400000001e-01
            -4.261469060000000e+00 * tc[0]
            -5.043679950000000e-03 * tc[1]
            +5.358436400000000e-07 * tc[2]
            -1.745077225000000e-11 * tc[3]
            -2.091695515000000e-15 * tc[4];
        /*species 31: PXC4H9O2 */
        species[31] =
            -1.083581030000000e+04 * invT
            -1.940667840000000e+01
            -1.943636500000000e+00 * tc[0]
            -2.577565815000000e-02 * tc[1]
            +5.471406666666667e-06 * tc[2]
            -9.422071666666667e-10 * tc[3]
            +8.505930300000000e-14 * tc[4];
    } else {
        /*species 18: CH3O2 */
        species[18] =
            -1.535748380000000e+03 * invT
            +1.067751777000000e+01
            -5.957878910000000e+00 * tc[0]
            -3.953643130000000e-03 * tc[1]
            +4.470770566666667e-07 * tc[2]
            -3.449094475000000e-11 * tc[3]
            +1.195036650000000e-15 * tc[4];
        /*species 31: PXC4H9O2 */
        species[31] =
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
        /*species 20: C3H6 */
        species[20] =
            +1.066881640000000e+03 * invT
            -2.150575815600000e+01
            -3.946154440000000e-01 * tc[0]
            -1.445538310000000e-02 * tc[1]
            +2.581446800000000e-06 * tc[2]
            -3.240118408333333e-10 * tc[3]
            +1.689451760000000e-14 * tc[4];
    } else {
        /*species 20: C3H6 */
        species[20] =
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
        /*species 23: C3H4XA */
        species[23] =
            +2.251243000000000e+04 * invT
            -7.395871000000000e+00
            -2.539831000000000e+00 * tc[0]
            -8.167185000000000e-03 * tc[1]
            +2.941583333333333e-07 * tc[2]
            +3.872804166666666e-10 * tc[3]
            -8.645655000000001e-14 * tc[4];
    } else {
        /*species 23: C3H4XA */
        species[23] =
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
        /*species 24: C3H5XA */
        species[24] =
            +1.938342260000000e+04 * invT
            -2.583584505800000e+01
            +5.291319580000000e-01 * tc[0]
            -1.672795500000000e-02 * tc[1]
            +4.223350450000001e-06 * tc[2]
            -8.572146166666666e-10 * tc[3]
            +8.662917000000000e-14 * tc[4];
        /*species 34: C5H11X1 */
        species[34] =
            +4.839953030000000e+03 * invT
            -3.346275221200000e+01
            +9.052559120000000e-01 * tc[0]
            -3.053164260000000e-02 * tc[1]
            +6.824863750000000e-06 * tc[2]
            -1.217445583333333e-09 * tc[3]
            +1.094298075000000e-13 * tc[4];
    } else {
        /*species 24: C3H5XA */
        species[24] =
            +1.635760920000000e+04 * invT
            +3.103978458000000e+01
            -8.458839579999999e+00 * tc[0]
            -5.634774150000000e-03 * tc[1]
            +6.396547733333333e-07 * tc[2]
            -4.950492658333333e-11 * tc[3]
            +1.719590150000000e-15 * tc[4];
        /*species 34: C5H11X1 */
        species[34] =
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
        /*species 27: C4H6 */
        species[27] =
            +1.175513140000000e+04 * invT
            -3.051353451000000e+01
            +1.430951210000000e+00 * tc[0]
            -2.393530310000000e-02 * tc[1]
            +6.924113333333333e-06 * tc[2]
            -1.596246266666667e-09 * tc[3]
            +1.785792535000000e-13 * tc[4];
    } else {
        /*species 27: C4H6 */
        species[27] =
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
        /*species 28: C4H7 */
        species[28] =
            +1.499335910000000e+04 * invT
            -2.708007795200000e+01
            +3.505083520000000e-01 * tc[0]
            -2.132556215000000e-02 * tc[1]
            +4.849656216666667e-06 * tc[2]
            -8.783659500000000e-10 * tc[3]
            +8.002992699999999e-14 * tc[4];
        /*species 29: C4H8X1 */
        species[29] =
            -1.578750350000000e+03 * invT
            -3.033979568900000e+01
            +8.313720890000000e-01 * tc[0]
            -2.262904890000000e-02 * tc[1]
            +4.894309316666667e-06 * tc[2]
            -8.351702999999999e-10 * tc[3]
            +7.159584000000000e-14 * tc[4];
        /*species 32: C5H9 */
        species[32] =
            +1.255898240000000e+04 * invT
            -3.402426990000000e+01
            +1.380139500000000e+00 * tc[0]
            -2.788042435000000e-02 * tc[1]
            +6.169065466666666e-06 * tc[2]
            -1.057365841666667e-09 * tc[3]
            +8.926941750000000e-14 * tc[4];
        /*species 33: C5H10X1 */
        species[33] =
            -4.465466660000000e+03 * invT
            -3.333621381000000e+01
            +1.062234810000000e+00 * tc[0]
            -2.871091470000000e-02 * tc[1]
            +6.241448166666667e-06 * tc[2]
            -1.061374908333333e-09 * tc[3]
            +8.980489449999999e-14 * tc[4];
        /*species 35: C6H12X1 */
        species[35] =
            -7.343686170000000e+03 * invT
            -3.666482115000000e+01
            +1.352752050000000e+00 * tc[0]
            -3.493277130000000e-02 * tc[1]
            +7.656800366666667e-06 * tc[2]
            -1.308061191666667e-09 * tc[3]
            +1.106480875000000e-13 * tc[4];
    } else {
        /*species 28: C4H7 */
        species[28] =
            +1.090419370000000e+04 * invT
            +4.676965930000000e+01
            -1.121035780000000e+01 * tc[0]
            -8.024159800000000e-03 * tc[1]
            +9.108371533333334e-07 * tc[2]
            -7.049508775000000e-11 * tc[3]
            +2.448863695000000e-15 * tc[4];
        /*species 29: C4H8X1 */
        species[29] =
            -5.978710380000000e+03 * invT
            +4.778781060000000e+01
            -1.135086680000000e+01 * tc[0]
            -9.030893850000001e-03 * tc[1]
            +1.026821715000000e-06 * tc[2]
            -7.955441325000001e-11 * tc[3]
            +2.765448205000000e-15 * tc[4];
        /*species 32: C5H9 */
        species[32] =
            +7.004961350000000e+03 * invT
            +6.563622270000000e+01
            -1.418604540000000e+01 * tc[0]
            -1.035644495000000e-02 * tc[1]
            +1.178267695000000e-06 * tc[2]
            -9.133927750000000e-11 * tc[3]
            +3.176611040000000e-15 * tc[4];
        /*species 33: C5H10X1 */
        species[33] =
            -1.008982050000000e+04 * invT
            +6.695354750000000e+01
            -1.458515390000000e+01 * tc[0]
            -1.120362355000000e-02 * tc[1]
            +1.272246708333333e-06 * tc[2]
            -9.849080500000001e-11 * tc[3]
            +3.421925695000000e-15 * tc[4];
        /*species 35: C6H12X1 */
        species[35] =
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
        /*species 30: PXC4H9 */
        species[30] =
            +7.689452480000000e+03 * invT
            -2.912305292500000e+01
            +4.377797250000000e-01 * tc[0]
            -2.394861820000000e-02 * tc[1]
            +5.233719316666667e-06 * tc[2]
            -9.148872666666667e-10 * tc[3]
            +8.100533200000001e-14 * tc[4];
    } else {
        /*species 30: PXC4H9 */
        species[30] =
            +3.172319420000000e+03 * invT
            +5.149359040000000e+01
            -1.215100820000000e+01 * tc[0]
            -9.715535850000000e-03 * tc[1]
            +1.102629916666667e-06 * tc[2]
            -8.531261333333333e-11 * tc[3]
            +2.962648535000000e-15 * tc[4];
    }

    /*species with midpoint at T=1382 kelvin */
    if (T < 1382) {
        /*species 36: C7H15X2 */
        species[36] =
            -2.356053030000000e+03 * invT
            -3.377006617670000e+01
            +3.791557670000000e-02 * tc[0]
            -3.783632850000000e-02 * tc[1]
            +6.791227233333333e-06 * tc[2]
            -7.772324525000000e-10 * tc[3]
            +2.461803725000000e-14 * tc[4];
    } else {
        /*species 36: C7H15X2 */
        species[36] =
            -1.058736160000000e+04 * invT
            +1.068578495000000e+02
            -2.163688420000000e+01 * tc[0]
            -1.616624020000000e-02 * tc[1]
            +1.821230116666667e-06 * tc[2]
            -1.402975500000000e-10 * tc[3]
            +4.858870455000000e-15 * tc[4];
    }

    /*species with midpoint at T=1391 kelvin */
    if (T < 1391) {
        /*species 37: NXC7H16 */
        species[37] =
            -2.565865650000000e+04 * invT
            -3.664165307000000e+01
            +1.268361870000000e+00 * tc[0]
            -4.271779100000000e-02 * tc[1]
            +8.755779766666667e-06 * tc[2]
            -1.357881008333333e-09 * tc[3]
            +1.011974625000000e-13 * tc[4];
    } else {
        /*species 37: NXC7H16 */
        species[37] =
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
        /*species 7: HCCO */
        species[7] =
            +1.965892000000000e+04 * invT
            +4.566121099999999e+00
            -5.047965000000000e+00 * tc[0]
            -2.226739000000000e-03 * tc[1]
            -3.780471666666667e-08 * tc[2]
            +1.235079166666667e-10 * tc[3]
            -1.125371000000000e-14 * tc[4];
        /*species 8: C2H3 */
        species[8] =
            +3.335225000000000e+04 * invT
            -9.096924000000001e+00
            -2.459276000000000e+00 * tc[0]
            -3.685738000000000e-03 * tc[1]
            -3.516455000000000e-07 * tc[2]
            +1.101368333333333e-10 * tc[3]
            +5.923920000000000e-14 * tc[4];
        /*species 9: CH2CHO */
        species[9] =
            +1.521477000000000e+03 * invT
            -6.149227999999999e+00
            -3.409062000000000e+00 * tc[0]
            -5.369285000000000e-03 * tc[1]
            -3.152486666666667e-07 * tc[2]
            +5.965485833333333e-10 * tc[3]
            -1.433692500000000e-13 * tc[4];
        /*species 11: CH3CO */
        species[11] =
            -4.108508000000000e+03 * invT
            -8.103572000000000e+00
            -3.125278000000000e+00 * tc[0]
            -4.889110000000000e-03 * tc[1]
            -7.535746666666667e-07 * tc[2]
            +7.507885000000000e-10 * tc[3]
            -1.596859000000000e-13 * tc[4];
        /*species 13: C3H2 */
        species[13] =
            +6.350421000000000e+04 * invT
            -5.702732000000000e+00
            -3.166714000000000e+00 * tc[0]
            -1.241286000000000e-02 * tc[1]
            +7.652728333333333e-06 * tc[2]
            -3.556682500000000e-09 * tc[3]
            +7.410759999999999e-13 * tc[4];
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
        /*species 7: HCCO */
        species[7] =
            +1.901513000000000e+04 * invT
            +1.582933500000000e+01
            -6.758073000000000e+00 * tc[0]
            -1.000200000000000e-03 * tc[1]
            +3.379345000000000e-08 * tc[2]
            +8.676100000000000e-12 * tc[3]
            -9.825825000000000e-16 * tc[4];
        /*species 8: C2H3 */
        species[8] =
            +3.185435000000000e+04 * invT
            +1.446378100000000e+01
            -5.933468000000000e+00 * tc[0]
            -2.008873000000000e-03 * tc[1]
            +6.611233333333333e-08 * tc[2]
            +1.201055833333333e-11 * tc[3]
            -1.189322000000000e-15 * tc[4];
        /*species 9: CH2CHO */
        species[9] =
            +4.903218000000000e+02 * invT
            +1.102092100000000e+01
            -5.975670000000000e+00 * tc[0]
            -4.065295500000000e-03 * tc[1]
            +4.572706666666667e-07 * tc[2]
            -3.391920000000000e-11 * tc[3]
            +1.088008500000000e-15 * tc[4];
        /*species 11: CH3CO */
        species[11] =
            -5.187863000000000e+03 * invT
            +8.887228000000000e+00
            -5.612279000000000e+00 * tc[0]
            -4.224943000000000e-03 * tc[1]
            +4.756911666666667e-07 * tc[2]
            -3.531980000000000e-11 * tc[3]
            +1.134202000000000e-15 * tc[4];
        /*species 13: C3H2 */
        species[13] =
            +6.259722000000000e+04 * invT
            +2.003988100000000e+01
            -7.670981000000000e+00 * tc[0]
            -1.374374500000000e-03 * tc[1]
            +7.284905000000000e-08 * tc[2]
            +5.379665833333334e-12 * tc[3]
            -8.319435000000000e-16 * tc[4];
    }

    /*species with midpoint at T=1390 kelvin */
    if (T < 1390) {
        /*species 6: CH3O2H */
        species[6] =
            -1.771979260000000e+04 * invT
            -6.021811320000000e+00
            -3.234428170000000e+00 * tc[0]
            -9.506488350000000e-03 * tc[1]
            +1.889771450000000e-06 * tc[2]
            -2.835888775000000e-10 * tc[3]
            +2.059151110000000e-14 * tc[4];
    } else {
        /*species 6: CH3O2H */
        species[6] =
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
        /*species 10: C2H5O */
        species[10] =
            -3.352529250000000e+03 * invT
            -2.231351709200000e+01
            -4.944207080000000e-01 * tc[0]
            -1.358872170000000e-02 * tc[1]
            +2.765150166666667e-06 * tc[2]
            -4.293368333333333e-10 * tc[3]
            +3.242484575000000e-14 * tc[4];
    } else {
        /*species 10: C2H5O */
        species[10] =
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
        /*species 12: C2H5O2 */
        species[12] =
            -5.038807580000000e+03 * invT
            -1.420893532000000e+01
            -2.268461880000000e+00 * tc[0]
            -1.384712890000000e-02 * tc[1]
            +2.846735100000000e-06 * tc[2]
            -4.899598983333333e-10 * tc[3]
            +4.604495345000000e-14 * tc[4];
    } else {
        /*species 12: C2H5O2 */
        species[12] =
            -7.824817950000000e+03 * invT
            +3.254826223000000e+01
            -9.486960229999999e+00 * tc[0]
            -6.223627250000000e-03 * tc[1]
            +7.202692933333333e-07 * tc[2]
            -5.646525275000000e-11 * tc[3]
            +1.978922840000000e-15 * tc[4];
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
        /*species 19: C2H2 */
        species[19] =
            +2.61244400e+04 * invT
            -7.79181600e+00
            -2.01356200e+00 * tc[0]
            -7.59522500e-03 * tc[1]
            +2.69386500e-06 * tc[2]
            -7.56582667e-10 * tc[3]
            +9.56373000e-14 * tc[4];
        /*species 21: C2H4 */
        species[21] =
            +5.57304600e+03 * invT
            -2.60729780e+01
            +8.61488000e-01 * tc[0]
            -1.39808150e-02 * tc[1]
            +5.64779500e-06 * tc[2]
            -2.32096000e-09 * tc[3]
            +4.86893950e-13 * tc[4];
        /*species 22: C3H3 */
        species[22] =
            +3.98888300e+04 * invT
            +3.16874510e+00
            -4.75420000e+00 * tc[0]
            -5.54014000e-03 * tc[1]
            -4.65553833e-08 * tc[2]
            +4.56601000e-10 * tc[3]
            -9.74814500e-14 * tc[4];
        /*species 25: NXC3H7 */
        species[25] =
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
        /*species 19: C2H2 */
        species[19] =
            +2.56676600e+04 * invT
            +6.23710800e+00
            -4.43677000e+00 * tc[0]
            -2.68801950e-03 * tc[1]
            +3.18802833e-07 * tc[2]
            -2.73864917e-11 * tc[3]
            +1.07835500e-15 * tc[4];
        /*species 21: C2H4 */
        species[21] =
            +4.42828900e+03 * invT
            +2.98030000e-01
            -3.52841900e+00 * tc[0]
            -5.74259000e-03 * tc[1]
            +7.36397500e-07 * tc[2]
            -6.53716750e-11 * tc[3]
            +2.63342400e-15 * tc[4];
        /*species 22: C3H3 */
        species[22] =
            +3.84742000e+04 * invT
            +2.96102370e+01
            -8.83104700e+00 * tc[0]
            -2.17859750e-03 * tc[1]
            +6.84844500e-08 * tc[2]
            +1.97393583e-11 * tc[3]
            -2.18826000e-15 * tc[4];
        /*species 25: NXC3H7 */
        species[25] =
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
        /*species 26: NXC3H7O2 */
        species[26] =
            -7.93745567e+03 * invT
            -1.78009599e+01
            -2.10731492e+00 * tc[0]
            -1.98082493e-02 * tc[1]
            +4.15819332e-06 * tc[2]
            -7.16208583e-10 * tc[3]
            +6.56201650e-14 * tc[4];
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
        /*species 26: NXC3H7O2 */
        species[26] =
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

    /*species with midpoint at T=1385 kelvin */
    if (T < 1385) {
        /*species 18: CH3O2 */
        species[18] =
            -6.84394259e+02 * invT
            -1.90183414e+00
            -4.26146906e+00 * tc[0]
            -5.04367995e-03 * tc[1]
            +5.35843640e-07 * tc[2]
            -1.74507723e-11 * tc[3]
            -2.09169552e-15 * tc[4];
        /*species 31: PXC4H9O2 */
        species[31] =
            -1.08358103e+04 * invT
            -2.04066784e+01
            -1.94363650e+00 * tc[0]
            -2.57756581e-02 * tc[1]
            +5.47140667e-06 * tc[2]
            -9.42207167e-10 * tc[3]
            +8.50593030e-14 * tc[4];
    } else {
        /*species 18: CH3O2 */
        species[18] =
            -1.53574838e+03 * invT
            +9.67751777e+00
            -5.95787891e+00 * tc[0]
            -3.95364313e-03 * tc[1]
            +4.47077057e-07 * tc[2]
            -3.44909447e-11 * tc[3]
            +1.19503665e-15 * tc[4];
        /*species 31: PXC4H9O2 */
        species[31] =
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
        /*species 20: C3H6 */
        species[20] =
            +1.06688164e+03 * invT
            -2.25057582e+01
            -3.94615444e-01 * tc[0]
            -1.44553831e-02 * tc[1]
            +2.58144680e-06 * tc[2]
            -3.24011841e-10 * tc[3]
            +1.68945176e-14 * tc[4];
    } else {
        /*species 20: C3H6 */
        species[20] =
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
        /*species 23: C3H4XA */
        species[23] =
            +2.25124300e+04 * invT
            -8.39587100e+00
            -2.53983100e+00 * tc[0]
            -8.16718500e-03 * tc[1]
            +2.94158333e-07 * tc[2]
            +3.87280417e-10 * tc[3]
            -8.64565500e-14 * tc[4];
    } else {
        /*species 23: C3H4XA */
        species[23] =
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
        /*species 24: C3H5XA */
        species[24] =
            +1.93834226e+04 * invT
            -2.68358451e+01
            +5.29131958e-01 * tc[0]
            -1.67279550e-02 * tc[1]
            +4.22335045e-06 * tc[2]
            -8.57214617e-10 * tc[3]
            +8.66291700e-14 * tc[4];
        /*species 34: C5H11X1 */
        species[34] =
            +4.83995303e+03 * invT
            -3.44627522e+01
            +9.05255912e-01 * tc[0]
            -3.05316426e-02 * tc[1]
            +6.82486375e-06 * tc[2]
            -1.21744558e-09 * tc[3]
            +1.09429807e-13 * tc[4];
    } else {
        /*species 24: C3H5XA */
        species[24] =
            +1.63576092e+04 * invT
            +3.00397846e+01
            -8.45883958e+00 * tc[0]
            -5.63477415e-03 * tc[1]
            +6.39654773e-07 * tc[2]
            -4.95049266e-11 * tc[3]
            +1.71959015e-15 * tc[4];
        /*species 34: C5H11X1 */
        species[34] =
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
        /*species 27: C4H6 */
        species[27] =
            +1.17551314e+04 * invT
            -3.15135345e+01
            +1.43095121e+00 * tc[0]
            -2.39353031e-02 * tc[1]
            +6.92411333e-06 * tc[2]
            -1.59624627e-09 * tc[3]
            +1.78579253e-13 * tc[4];
    } else {
        /*species 27: C4H6 */
        species[27] =
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
        /*species 28: C4H7 */
        species[28] =
            +1.49933591e+04 * invT
            -2.80800780e+01
            +3.50508352e-01 * tc[0]
            -2.13255622e-02 * tc[1]
            +4.84965622e-06 * tc[2]
            -8.78365950e-10 * tc[3]
            +8.00299270e-14 * tc[4];
        /*species 29: C4H8X1 */
        species[29] =
            -1.57875035e+03 * invT
            -3.13397957e+01
            +8.31372089e-01 * tc[0]
            -2.26290489e-02 * tc[1]
            +4.89430932e-06 * tc[2]
            -8.35170300e-10 * tc[3]
            +7.15958400e-14 * tc[4];
        /*species 32: C5H9 */
        species[32] =
            +1.25589824e+04 * invT
            -3.50242699e+01
            +1.38013950e+00 * tc[0]
            -2.78804243e-02 * tc[1]
            +6.16906547e-06 * tc[2]
            -1.05736584e-09 * tc[3]
            +8.92694175e-14 * tc[4];
        /*species 33: C5H10X1 */
        species[33] =
            -4.46546666e+03 * invT
            -3.43362138e+01
            +1.06223481e+00 * tc[0]
            -2.87109147e-02 * tc[1]
            +6.24144817e-06 * tc[2]
            -1.06137491e-09 * tc[3]
            +8.98048945e-14 * tc[4];
        /*species 35: C6H12X1 */
        species[35] =
            -7.34368617e+03 * invT
            -3.76648212e+01
            +1.35275205e+00 * tc[0]
            -3.49327713e-02 * tc[1]
            +7.65680037e-06 * tc[2]
            -1.30806119e-09 * tc[3]
            +1.10648088e-13 * tc[4];
    } else {
        /*species 28: C4H7 */
        species[28] =
            +1.09041937e+04 * invT
            +4.57696593e+01
            -1.12103578e+01 * tc[0]
            -8.02415980e-03 * tc[1]
            +9.10837153e-07 * tc[2]
            -7.04950877e-11 * tc[3]
            +2.44886369e-15 * tc[4];
        /*species 29: C4H8X1 */
        species[29] =
            -5.97871038e+03 * invT
            +4.67878106e+01
            -1.13508668e+01 * tc[0]
            -9.03089385e-03 * tc[1]
            +1.02682171e-06 * tc[2]
            -7.95544133e-11 * tc[3]
            +2.76544820e-15 * tc[4];
        /*species 32: C5H9 */
        species[32] =
            +7.00496135e+03 * invT
            +6.46362227e+01
            -1.41860454e+01 * tc[0]
            -1.03564449e-02 * tc[1]
            +1.17826770e-06 * tc[2]
            -9.13392775e-11 * tc[3]
            +3.17661104e-15 * tc[4];
        /*species 33: C5H10X1 */
        species[33] =
            -1.00898205e+04 * invT
            +6.59535475e+01
            -1.45851539e+01 * tc[0]
            -1.12036235e-02 * tc[1]
            +1.27224671e-06 * tc[2]
            -9.84908050e-11 * tc[3]
            +3.42192569e-15 * tc[4];
        /*species 35: C6H12X1 */
        species[35] =
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
        /*species 30: PXC4H9 */
        species[30] =
            +7.68945248e+03 * invT
            -3.01230529e+01
            +4.37779725e-01 * tc[0]
            -2.39486182e-02 * tc[1]
            +5.23371932e-06 * tc[2]
            -9.14887267e-10 * tc[3]
            +8.10053320e-14 * tc[4];
    } else {
        /*species 30: PXC4H9 */
        species[30] =
            +3.17231942e+03 * invT
            +5.04935904e+01
            -1.21510082e+01 * tc[0]
            -9.71553585e-03 * tc[1]
            +1.10262992e-06 * tc[2]
            -8.53126133e-11 * tc[3]
            +2.96264853e-15 * tc[4];
    }

    /*species with midpoint at T=1382 kelvin */
    if (T < 1382) {
        /*species 36: C7H15X2 */
        species[36] =
            -2.35605303e+03 * invT
            -3.47700662e+01
            +3.79155767e-02 * tc[0]
            -3.78363285e-02 * tc[1]
            +6.79122723e-06 * tc[2]
            -7.77232453e-10 * tc[3]
            +2.46180373e-14 * tc[4];
    } else {
        /*species 36: C7H15X2 */
        species[36] =
            -1.05873616e+04 * invT
            +1.05857850e+02
            -2.16368842e+01 * tc[0]
            -1.61662402e-02 * tc[1]
            +1.82123012e-06 * tc[2]
            -1.40297550e-10 * tc[3]
            +4.85887045e-15 * tc[4];
    }

    /*species with midpoint at T=1391 kelvin */
    if (T < 1391) {
        /*species 37: NXC7H16 */
        species[37] =
            -2.56586565e+04 * invT
            -3.76416531e+01
            +1.26836187e+00 * tc[0]
            -4.27177910e-02 * tc[1]
            +8.75577977e-06 * tc[2]
            -1.35788101e-09 * tc[3]
            +1.01197462e-13 * tc[4];
    } else {
        /*species 37: NXC7H16 */
        species[37] =
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
        /*species 19: C2H2 */
        species[19] =
            +1.01356200e+00
            +1.51904500e-02 * tc[1]
            -1.61631900e-05 * tc[2]
            +9.07899200e-09 * tc[3]
            -1.91274600e-12 * tc[4];
        /*species 21: C2H4 */
        species[21] =
            -1.86148800e+00
            +2.79616300e-02 * tc[1]
            -3.38867700e-05 * tc[2]
            +2.78515200e-08 * tc[3]
            -9.73787900e-12 * tc[4];
        /*species 22: C3H3 */
        species[22] =
            +3.75420000e+00
            +1.10802800e-02 * tc[1]
            +2.79332300e-07 * tc[2]
            -5.47921200e-09 * tc[3]
            +1.94962900e-12 * tc[4];
        /*species 25: NXC3H7 */
        species[25] =
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
        /*species 19: C2H2 */
        species[19] =
            +3.43677000e+00
            +5.37603900e-03 * tc[1]
            -1.91281700e-06 * tc[2]
            +3.28637900e-10 * tc[3]
            -2.15671000e-14 * tc[4];
        /*species 21: C2H4 */
        species[21] =
            +2.52841900e+00
            +1.14851800e-02 * tc[1]
            -4.41838500e-06 * tc[2]
            +7.84460100e-10 * tc[3]
            -5.26684800e-14 * tc[4];
        /*species 22: C3H3 */
        species[22] =
            +7.83104700e+00
            +4.35719500e-03 * tc[1]
            -4.10906700e-07 * tc[2]
            -2.36872300e-10 * tc[3]
            +4.37652000e-14 * tc[4];
        /*species 25: NXC3H7 */
        species[25] =
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
        /*species 26: NXC3H7O2 */
        species[26] =
            +1.10731492e+00
            +3.96164986e-02 * tc[1]
            -2.49491599e-05 * tc[2]
            +8.59450300e-09 * tc[3]
            -1.31240330e-12 * tc[4];
    } else {
        /*species 15: C2H6 */
        species[15] =
            +5.10683385e+00
            +1.29236361e-02 * tc[1]
            -4.42527196e-06 * tc[2]
            +6.87391726e-10 * tc[3]
            -3.98901732e-14 * tc[4];
        /*species 26: NXC3H7O2 */
        species[26] =
            +1.16327059e+01
            +1.69910726e-02 * tc[1]
            -5.88866873e-06 * tc[2]
            +9.22195396e-10 * tc[3]
            -5.38230675e-14 * tc[4];
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

    /*species with midpoint at T=1385 kelvin */
    if (T < 1385) {
        /*species 18: CH3O2 */
        species[18] =
            +3.26146906e+00
            +1.00873599e-02 * tc[1]
            -3.21506184e-06 * tc[2]
            +2.09409267e-10 * tc[3]
            +4.18339103e-14 * tc[4];
        /*species 31: PXC4H9O2 */
        species[31] =
            +9.43636500e-01
            +5.15513163e-02 * tc[1]
            -3.28284400e-05 * tc[2]
            +1.13064860e-08 * tc[3]
            -1.70118606e-12 * tc[4];
    } else {
        /*species 18: CH3O2 */
        species[18] =
            +4.95787891e+00
            +7.90728626e-03 * tc[1]
            -2.68246234e-06 * tc[2]
            +4.13891337e-10 * tc[3]
            -2.39007330e-14 * tc[4];
        /*species 31: PXC4H9O2 */
        species[31] =
            +1.47845448e+01
            +2.15210910e-02 * tc[1]
            -7.44909017e-06 * tc[2]
            +1.16558071e-09 * tc[3]
            -6.79885609e-14 * tc[4];
    }

    /*species with midpoint at T=1388 kelvin */
    if (T < 1388) {
        /*species 20: C3H6 */
        species[20] =
            -6.05384556e-01
            +2.89107662e-02 * tc[1]
            -1.54886808e-05 * tc[2]
            +3.88814209e-09 * tc[3]
            -3.37890352e-13 * tc[4];
    } else {
        /*species 20: C3H6 */
        species[20] =
            +7.01595958e+00
            +1.37023634e-02 * tc[1]
            -4.66249733e-06 * tc[2]
            +7.21254402e-10 * tc[3]
            -4.17370126e-14 * tc[4];
    }

    /*species with midpoint at T=1400 kelvin */
    if (T < 1400) {
        /*species 23: C3H4XA */
        species[23] =
            +1.53983100e+00
            +1.63343700e-02 * tc[1]
            -1.76495000e-06 * tc[2]
            -4.64736500e-09 * tc[3]
            +1.72913100e-12 * tc[4];
    } else {
        /*species 23: C3H4XA */
        species[23] =
            +8.77625600e+00
            +5.30213800e-03 * tc[1]
            -3.70111800e-07 * tc[2]
            -3.02638600e-10 * tc[3]
            +5.08958100e-14 * tc[4];
    }

    /*species with midpoint at T=1397 kelvin */
    if (T < 1397) {
        /*species 24: C3H5XA */
        species[24] =
            -1.52913196e+00
            +3.34559100e-02 * tc[1]
            -2.53401027e-05 * tc[2]
            +1.02865754e-08 * tc[3]
            -1.73258340e-12 * tc[4];
        /*species 34: C5H11X1 */
        species[34] =
            -1.90525591e+00
            +6.10632852e-02 * tc[1]
            -4.09491825e-05 * tc[2]
            +1.46093470e-08 * tc[3]
            -2.18859615e-12 * tc[4];
    } else {
        /*species 24: C3H5XA */
        species[24] =
            +7.45883958e+00
            +1.12695483e-02 * tc[1]
            -3.83792864e-06 * tc[2]
            +5.94059119e-10 * tc[3]
            -3.43918030e-14 * tc[4];
        /*species 34: C5H11X1 */
        species[34] =
            +1.43234740e+01
            +2.39041200e-02 * tc[1]
            -8.14771619e-06 * tc[2]
            +1.26176236e-09 * tc[3]
            -7.30677335e-14 * tc[4];
    }

    /*species with midpoint at T=1398 kelvin */
    if (T < 1398) {
        /*species 27: C4H6 */
        species[27] =
            -2.43095121e+00
            +4.78706062e-02 * tc[1]
            -4.15446800e-05 * tc[2]
            +1.91549552e-08 * tc[3]
            -3.57158507e-12 * tc[4];
    } else {
        /*species 27: C4H6 */
        species[27] =
            +1.01633789e+01
            +1.37163965e-02 * tc[1]
            -4.69715783e-06 * tc[2]
            +7.29693836e-10 * tc[3]
            -4.23486203e-14 * tc[4];
    }

    /*species with midpoint at T=1392 kelvin */
    if (T < 1392) {
        /*species 28: C4H7 */
        species[28] =
            -1.35050835e+00
            +4.26511243e-02 * tc[1]
            -2.90979373e-05 * tc[2]
            +1.05403914e-08 * tc[3]
            -1.60059854e-12 * tc[4];
        /*species 29: C4H8X1 */
        species[29] =
            -1.83137209e+00
            +4.52580978e-02 * tc[1]
            -2.93658559e-05 * tc[2]
            +1.00220436e-08 * tc[3]
            -1.43191680e-12 * tc[4];
        /*species 32: C5H9 */
        species[32] =
            -2.38013950e+00
            +5.57608487e-02 * tc[1]
            -3.70143928e-05 * tc[2]
            +1.26883901e-08 * tc[3]
            -1.78538835e-12 * tc[4];
        /*species 33: C5H10X1 */
        species[33] =
            -2.06223481e+00
            +5.74218294e-02 * tc[1]
            -3.74486890e-05 * tc[2]
            +1.27364989e-08 * tc[3]
            -1.79609789e-12 * tc[4];
        /*species 35: C6H12X1 */
        species[35] =
            -2.35275205e+00
            +6.98655426e-02 * tc[1]
            -4.59408022e-05 * tc[2]
            +1.56967343e-08 * tc[3]
            -2.21296175e-12 * tc[4];
    } else {
        /*species 28: C4H7 */
        species[28] =
            +1.02103578e+01
            +1.60483196e-02 * tc[1]
            -5.46502292e-06 * tc[2]
            +8.45941053e-10 * tc[3]
            -4.89772739e-14 * tc[4];
        /*species 29: C4H8X1 */
        species[29] =
            +1.03508668e+01
            +1.80617877e-02 * tc[1]
            -6.16093029e-06 * tc[2]
            +9.54652959e-10 * tc[3]
            -5.53089641e-14 * tc[4];
        /*species 32: C5H9 */
        species[32] =
            +1.31860454e+01
            +2.07128899e-02 * tc[1]
            -7.06960617e-06 * tc[2]
            +1.09607133e-09 * tc[3]
            -6.35322208e-14 * tc[4];
        /*species 33: C5H10X1 */
        species[33] =
            +1.35851539e+01
            +2.24072471e-02 * tc[1]
            -7.63348025e-06 * tc[2]
            +1.18188966e-09 * tc[3]
            -6.84385139e-14 * tc[4];
        /*species 35: C6H12X1 */
        species[35] =
            +1.68337529e+01
            +2.67377658e-02 * tc[1]
            -9.10036773e-06 * tc[2]
            +1.40819768e-09 * tc[3]
            -8.15124244e-14 * tc[4];
    }

    /*species with midpoint at T=1395 kelvin */
    if (T < 1395) {
        /*species 30: PXC4H9 */
        species[30] =
            -1.43777972e+00
            +4.78972364e-02 * tc[1]
            -3.14023159e-05 * tc[2]
            +1.09786472e-08 * tc[3]
            -1.62010664e-12 * tc[4];
    } else {
        /*species 30: PXC4H9 */
        species[30] =
            +1.11510082e+01
            +1.94310717e-02 * tc[1]
            -6.61577950e-06 * tc[2]
            +1.02375136e-09 * tc[3]
            -5.92529707e-14 * tc[4];
    }

    /*species with midpoint at T=1382 kelvin */
    if (T < 1382) {
        /*species 36: C7H15X2 */
        species[36] =
            -1.03791558e+00
            +7.56726570e-02 * tc[1]
            -4.07473634e-05 * tc[2]
            +9.32678943e-09 * tc[3]
            -4.92360745e-13 * tc[4];
    } else {
        /*species 36: C7H15X2 */
        species[36] =
            +2.06368842e+01
            +3.23324804e-02 * tc[1]
            -1.09273807e-05 * tc[2]
            +1.68357060e-09 * tc[3]
            -9.71774091e-14 * tc[4];
    }

    /*species with midpoint at T=1391 kelvin */
    if (T < 1391) {
        /*species 37: NXC7H16 */
        species[37] =
            -2.26836187e+00
            +8.54355820e-02 * tc[1]
            -5.25346786e-05 * tc[2]
            +1.62945721e-08 * tc[3]
            -2.02394925e-12 * tc[4];
    } else {
        /*species 37: NXC7H16 */
        species[37] =
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
        /*species 19: C2H2 */
        species[19] =
            +2.01356200e+00
            +1.51904500e-02 * tc[1]
            -1.61631900e-05 * tc[2]
            +9.07899200e-09 * tc[3]
            -1.91274600e-12 * tc[4];
        /*species 21: C2H4 */
        species[21] =
            -8.61488000e-01
            +2.79616300e-02 * tc[1]
            -3.38867700e-05 * tc[2]
            +2.78515200e-08 * tc[3]
            -9.73787900e-12 * tc[4];
        /*species 22: C3H3 */
        species[22] =
            +4.75420000e+00
            +1.10802800e-02 * tc[1]
            +2.79332300e-07 * tc[2]
            -5.47921200e-09 * tc[3]
            +1.94962900e-12 * tc[4];
        /*species 25: NXC3H7 */
        species[25] =
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
        /*species 19: C2H2 */
        species[19] =
            +4.43677000e+00
            +5.37603900e-03 * tc[1]
            -1.91281700e-06 * tc[2]
            +3.28637900e-10 * tc[3]
            -2.15671000e-14 * tc[4];
        /*species 21: C2H4 */
        species[21] =
            +3.52841900e+00
            +1.14851800e-02 * tc[1]
            -4.41838500e-06 * tc[2]
            +7.84460100e-10 * tc[3]
            -5.26684800e-14 * tc[4];
        /*species 22: C3H3 */
        species[22] =
            +8.83104700e+00
            +4.35719500e-03 * tc[1]
            -4.10906700e-07 * tc[2]
            -2.36872300e-10 * tc[3]
            +4.37652000e-14 * tc[4];
        /*species 25: NXC3H7 */
        species[25] =
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
        /*species 26: NXC3H7O2 */
        species[26] =
            +2.10731492e+00
            +3.96164986e-02 * tc[1]
            -2.49491599e-05 * tc[2]
            +8.59450300e-09 * tc[3]
            -1.31240330e-12 * tc[4];
    } else {
        /*species 15: C2H6 */
        species[15] =
            +6.10683385e+00
            +1.29236361e-02 * tc[1]
            -4.42527196e-06 * tc[2]
            +6.87391726e-10 * tc[3]
            -3.98901732e-14 * tc[4];
        /*species 26: NXC3H7O2 */
        species[26] =
            +1.26327059e+01
            +1.69910726e-02 * tc[1]
            -5.88866873e-06 * tc[2]
            +9.22195396e-10 * tc[3]
            -5.38230675e-14 * tc[4];
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

    /*species with midpoint at T=1385 kelvin */
    if (T < 1385) {
        /*species 18: CH3O2 */
        species[18] =
            +4.26146906e+00
            +1.00873599e-02 * tc[1]
            -3.21506184e-06 * tc[2]
            +2.09409267e-10 * tc[3]
            +4.18339103e-14 * tc[4];
        /*species 31: PXC4H9O2 */
        species[31] =
            +1.94363650e+00
            +5.15513163e-02 * tc[1]
            -3.28284400e-05 * tc[2]
            +1.13064860e-08 * tc[3]
            -1.70118606e-12 * tc[4];
    } else {
        /*species 18: CH3O2 */
        species[18] =
            +5.95787891e+00
            +7.90728626e-03 * tc[1]
            -2.68246234e-06 * tc[2]
            +4.13891337e-10 * tc[3]
            -2.39007330e-14 * tc[4];
        /*species 31: PXC4H9O2 */
        species[31] =
            +1.57845448e+01
            +2.15210910e-02 * tc[1]
            -7.44909017e-06 * tc[2]
            +1.16558071e-09 * tc[3]
            -6.79885609e-14 * tc[4];
    }

    /*species with midpoint at T=1388 kelvin */
    if (T < 1388) {
        /*species 20: C3H6 */
        species[20] =
            +3.94615444e-01
            +2.89107662e-02 * tc[1]
            -1.54886808e-05 * tc[2]
            +3.88814209e-09 * tc[3]
            -3.37890352e-13 * tc[4];
    } else {
        /*species 20: C3H6 */
        species[20] =
            +8.01595958e+00
            +1.37023634e-02 * tc[1]
            -4.66249733e-06 * tc[2]
            +7.21254402e-10 * tc[3]
            -4.17370126e-14 * tc[4];
    }

    /*species with midpoint at T=1400 kelvin */
    if (T < 1400) {
        /*species 23: C3H4XA */
        species[23] =
            +2.53983100e+00
            +1.63343700e-02 * tc[1]
            -1.76495000e-06 * tc[2]
            -4.64736500e-09 * tc[3]
            +1.72913100e-12 * tc[4];
    } else {
        /*species 23: C3H4XA */
        species[23] =
            +9.77625600e+00
            +5.30213800e-03 * tc[1]
            -3.70111800e-07 * tc[2]
            -3.02638600e-10 * tc[3]
            +5.08958100e-14 * tc[4];
    }

    /*species with midpoint at T=1397 kelvin */
    if (T < 1397) {
        /*species 24: C3H5XA */
        species[24] =
            -5.29131958e-01
            +3.34559100e-02 * tc[1]
            -2.53401027e-05 * tc[2]
            +1.02865754e-08 * tc[3]
            -1.73258340e-12 * tc[4];
        /*species 34: C5H11X1 */
        species[34] =
            -9.05255912e-01
            +6.10632852e-02 * tc[1]
            -4.09491825e-05 * tc[2]
            +1.46093470e-08 * tc[3]
            -2.18859615e-12 * tc[4];
    } else {
        /*species 24: C3H5XA */
        species[24] =
            +8.45883958e+00
            +1.12695483e-02 * tc[1]
            -3.83792864e-06 * tc[2]
            +5.94059119e-10 * tc[3]
            -3.43918030e-14 * tc[4];
        /*species 34: C5H11X1 */
        species[34] =
            +1.53234740e+01
            +2.39041200e-02 * tc[1]
            -8.14771619e-06 * tc[2]
            +1.26176236e-09 * tc[3]
            -7.30677335e-14 * tc[4];
    }

    /*species with midpoint at T=1398 kelvin */
    if (T < 1398) {
        /*species 27: C4H6 */
        species[27] =
            -1.43095121e+00
            +4.78706062e-02 * tc[1]
            -4.15446800e-05 * tc[2]
            +1.91549552e-08 * tc[3]
            -3.57158507e-12 * tc[4];
    } else {
        /*species 27: C4H6 */
        species[27] =
            +1.11633789e+01
            +1.37163965e-02 * tc[1]
            -4.69715783e-06 * tc[2]
            +7.29693836e-10 * tc[3]
            -4.23486203e-14 * tc[4];
    }

    /*species with midpoint at T=1392 kelvin */
    if (T < 1392) {
        /*species 28: C4H7 */
        species[28] =
            -3.50508352e-01
            +4.26511243e-02 * tc[1]
            -2.90979373e-05 * tc[2]
            +1.05403914e-08 * tc[3]
            -1.60059854e-12 * tc[4];
        /*species 29: C4H8X1 */
        species[29] =
            -8.31372089e-01
            +4.52580978e-02 * tc[1]
            -2.93658559e-05 * tc[2]
            +1.00220436e-08 * tc[3]
            -1.43191680e-12 * tc[4];
        /*species 32: C5H9 */
        species[32] =
            -1.38013950e+00
            +5.57608487e-02 * tc[1]
            -3.70143928e-05 * tc[2]
            +1.26883901e-08 * tc[3]
            -1.78538835e-12 * tc[4];
        /*species 33: C5H10X1 */
        species[33] =
            -1.06223481e+00
            +5.74218294e-02 * tc[1]
            -3.74486890e-05 * tc[2]
            +1.27364989e-08 * tc[3]
            -1.79609789e-12 * tc[4];
        /*species 35: C6H12X1 */
        species[35] =
            -1.35275205e+00
            +6.98655426e-02 * tc[1]
            -4.59408022e-05 * tc[2]
            +1.56967343e-08 * tc[3]
            -2.21296175e-12 * tc[4];
    } else {
        /*species 28: C4H7 */
        species[28] =
            +1.12103578e+01
            +1.60483196e-02 * tc[1]
            -5.46502292e-06 * tc[2]
            +8.45941053e-10 * tc[3]
            -4.89772739e-14 * tc[4];
        /*species 29: C4H8X1 */
        species[29] =
            +1.13508668e+01
            +1.80617877e-02 * tc[1]
            -6.16093029e-06 * tc[2]
            +9.54652959e-10 * tc[3]
            -5.53089641e-14 * tc[4];
        /*species 32: C5H9 */
        species[32] =
            +1.41860454e+01
            +2.07128899e-02 * tc[1]
            -7.06960617e-06 * tc[2]
            +1.09607133e-09 * tc[3]
            -6.35322208e-14 * tc[4];
        /*species 33: C5H10X1 */
        species[33] =
            +1.45851539e+01
            +2.24072471e-02 * tc[1]
            -7.63348025e-06 * tc[2]
            +1.18188966e-09 * tc[3]
            -6.84385139e-14 * tc[4];
        /*species 35: C6H12X1 */
        species[35] =
            +1.78337529e+01
            +2.67377658e-02 * tc[1]
            -9.10036773e-06 * tc[2]
            +1.40819768e-09 * tc[3]
            -8.15124244e-14 * tc[4];
    }

    /*species with midpoint at T=1395 kelvin */
    if (T < 1395) {
        /*species 30: PXC4H9 */
        species[30] =
            -4.37779725e-01
            +4.78972364e-02 * tc[1]
            -3.14023159e-05 * tc[2]
            +1.09786472e-08 * tc[3]
            -1.62010664e-12 * tc[4];
    } else {
        /*species 30: PXC4H9 */
        species[30] =
            +1.21510082e+01
            +1.94310717e-02 * tc[1]
            -6.61577950e-06 * tc[2]
            +1.02375136e-09 * tc[3]
            -5.92529707e-14 * tc[4];
    }

    /*species with midpoint at T=1382 kelvin */
    if (T < 1382) {
        /*species 36: C7H15X2 */
        species[36] =
            -3.79155767e-02
            +7.56726570e-02 * tc[1]
            -4.07473634e-05 * tc[2]
            +9.32678943e-09 * tc[3]
            -4.92360745e-13 * tc[4];
    } else {
        /*species 36: C7H15X2 */
        species[36] =
            +2.16368842e+01
            +3.23324804e-02 * tc[1]
            -1.09273807e-05 * tc[2]
            +1.68357060e-09 * tc[3]
            -9.71774091e-14 * tc[4];
    }

    /*species with midpoint at T=1391 kelvin */
    if (T < 1391) {
        /*species 37: NXC7H16 */
        species[37] =
            -1.26836187e+00
            +8.54355820e-02 * tc[1]
            -5.25346786e-05 * tc[2]
            +1.62945721e-08 * tc[3]
            -2.02394925e-12 * tc[4];
    } else {
        /*species 37: NXC7H16 */
        species[37] =
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
        /*species 19: C2H2 */
        species[19] =
            +1.01356200e+00
            +7.59522500e-03 * tc[1]
            -5.38773000e-06 * tc[2]
            +2.26974800e-09 * tc[3]
            -3.82549200e-13 * tc[4]
            +2.61244400e+04 * invT;
        /*species 21: C2H4 */
        species[21] =
            -1.86148800e+00
            +1.39808150e-02 * tc[1]
            -1.12955900e-05 * tc[2]
            +6.96288000e-09 * tc[3]
            -1.94757580e-12 * tc[4]
            +5.57304600e+03 * invT;
        /*species 22: C3H3 */
        species[22] =
            +3.75420000e+00
            +5.54014000e-03 * tc[1]
            +9.31107667e-08 * tc[2]
            -1.36980300e-09 * tc[3]
            +3.89925800e-13 * tc[4]
            +3.98888300e+04 * invT;
        /*species 25: NXC3H7 */
        species[25] =
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
        /*species 19: C2H2 */
        species[19] =
            +3.43677000e+00
            +2.68801950e-03 * tc[1]
            -6.37605667e-07 * tc[2]
            +8.21594750e-11 * tc[3]
            -4.31342000e-15 * tc[4]
            +2.56676600e+04 * invT;
        /*species 21: C2H4 */
        species[21] =
            +2.52841900e+00
            +5.74259000e-03 * tc[1]
            -1.47279500e-06 * tc[2]
            +1.96115025e-10 * tc[3]
            -1.05336960e-14 * tc[4]
            +4.42828900e+03 * invT;
        /*species 22: C3H3 */
        species[22] =
            +7.83104700e+00
            +2.17859750e-03 * tc[1]
            -1.36968900e-07 * tc[2]
            -5.92180750e-11 * tc[3]
            +8.75304000e-15 * tc[4]
            +3.84742000e+04 * invT;
        /*species 25: NXC3H7 */
        species[25] =
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
        /*species 26: NXC3H7O2 */
        species[26] =
            +1.10731492e+00
            +1.98082493e-02 * tc[1]
            -8.31638663e-06 * tc[2]
            +2.14862575e-09 * tc[3]
            -2.62480660e-13 * tc[4]
            -7.93745567e+03 * invT;
    } else {
        /*species 15: C2H6 */
        species[15] =
            +5.10683385e+00
            +6.46181805e-03 * tc[1]
            -1.47509065e-06 * tc[2]
            +1.71847932e-10 * tc[3]
            -7.97803464e-15 * tc[4]
            -1.37500014e+04 * invT;
        /*species 26: NXC3H7O2 */
        species[26] =
            +1.16327059e+01
            +8.49553630e-03 * tc[1]
            -1.96288958e-06 * tc[2]
            +2.30548849e-10 * tc[3]
            -1.07646135e-14 * tc[4]
            -1.19194652e+04 * invT;
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

    /*species with midpoint at T=1385 kelvin */
    if (T < 1385) {
        /*species 18: CH3O2 */
        species[18] =
            +3.26146906e+00
            +5.04367995e-03 * tc[1]
            -1.07168728e-06 * tc[2]
            +5.23523168e-11 * tc[3]
            +8.36678206e-15 * tc[4]
            -6.84394259e+02 * invT;
        /*species 31: PXC4H9O2 */
        species[31] =
            +9.43636500e-01
            +2.57756581e-02 * tc[1]
            -1.09428133e-05 * tc[2]
            +2.82662150e-09 * tc[3]
            -3.40237212e-13 * tc[4]
            -1.08358103e+04 * invT;
    } else {
        /*species 18: CH3O2 */
        species[18] =
            +4.95787891e+00
            +3.95364313e-03 * tc[1]
            -8.94154113e-07 * tc[2]
            +1.03472834e-10 * tc[3]
            -4.78014660e-15 * tc[4]
            -1.53574838e+03 * invT;
        /*species 31: PXC4H9O2 */
        species[31] =
            +1.47845448e+01
            +1.07605455e-02 * tc[1]
            -2.48303006e-06 * tc[2]
            +2.91395178e-10 * tc[3]
            -1.35977122e-14 * tc[4]
            -1.60146054e+04 * invT;
    }

    /*species with midpoint at T=1388 kelvin */
    if (T < 1388) {
        /*species 20: C3H6 */
        species[20] =
            -6.05384556e-01
            +1.44553831e-02 * tc[1]
            -5.16289360e-06 * tc[2]
            +9.72035522e-10 * tc[3]
            -6.75780704e-14 * tc[4]
            +1.06688164e+03 * invT;
    } else {
        /*species 20: C3H6 */
        species[20] =
            +7.01595958e+00
            +6.85118170e-03 * tc[1]
            -1.55416578e-06 * tc[2]
            +1.80313601e-10 * tc[3]
            -8.34740252e-15 * tc[4]
            -1.87821271e+03 * invT;
    }

    /*species with midpoint at T=1400 kelvin */
    if (T < 1400) {
        /*species 23: C3H4XA */
        species[23] =
            +1.53983100e+00
            +8.16718500e-03 * tc[1]
            -5.88316667e-07 * tc[2]
            -1.16184125e-09 * tc[3]
            +3.45826200e-13 * tc[4]
            +2.25124300e+04 * invT;
    } else {
        /*species 23: C3H4XA */
        species[23] =
            +8.77625600e+00
            +2.65106900e-03 * tc[1]
            -1.23370600e-07 * tc[2]
            -7.56596500e-11 * tc[3]
            +1.01791620e-14 * tc[4]
            +1.95497200e+04 * invT;
    }

    /*species with midpoint at T=1397 kelvin */
    if (T < 1397) {
        /*species 24: C3H5XA */
        species[24] =
            -1.52913196e+00
            +1.67279550e-02 * tc[1]
            -8.44670090e-06 * tc[2]
            +2.57164385e-09 * tc[3]
            -3.46516680e-13 * tc[4]
            +1.93834226e+04 * invT;
        /*species 34: C5H11X1 */
        species[34] =
            -1.90525591e+00
            +3.05316426e-02 * tc[1]
            -1.36497275e-05 * tc[2]
            +3.65233675e-09 * tc[3]
            -4.37719230e-13 * tc[4]
            +4.83995303e+03 * invT;
    } else {
        /*species 24: C3H5XA */
        species[24] =
            +7.45883958e+00
            +5.63477415e-03 * tc[1]
            -1.27930955e-06 * tc[2]
            +1.48514780e-10 * tc[3]
            -6.87836060e-15 * tc[4]
            +1.63576092e+04 * invT;
        /*species 34: C5H11X1 */
        species[34] =
            +1.43234740e+01
            +1.19520600e-02 * tc[1]
            -2.71590540e-06 * tc[2]
            +3.15440590e-10 * tc[3]
            -1.46135467e-14 * tc[4]
            -9.23241637e+02 * invT;
    }

    /*species with midpoint at T=1398 kelvin */
    if (T < 1398) {
        /*species 27: C4H6 */
        species[27] =
            -2.43095121e+00
            +2.39353031e-02 * tc[1]
            -1.38482267e-05 * tc[2]
            +4.78873880e-09 * tc[3]
            -7.14317014e-13 * tc[4]
            +1.17551314e+04 * invT;
    } else {
        /*species 27: C4H6 */
        species[27] =
            +1.01633789e+01
            +6.85819825e-03 * tc[1]
            -1.56571928e-06 * tc[2]
            +1.82423459e-10 * tc[3]
            -8.46972406e-15 * tc[4]
            +7.79039770e+03 * invT;
    }

    /*species with midpoint at T=1392 kelvin */
    if (T < 1392) {
        /*species 28: C4H7 */
        species[28] =
            -1.35050835e+00
            +2.13255622e-02 * tc[1]
            -9.69931243e-06 * tc[2]
            +2.63509785e-09 * tc[3]
            -3.20119708e-13 * tc[4]
            +1.49933591e+04 * invT;
        /*species 29: C4H8X1 */
        species[29] =
            -1.83137209e+00
            +2.26290489e-02 * tc[1]
            -9.78861863e-06 * tc[2]
            +2.50551090e-09 * tc[3]
            -2.86383360e-13 * tc[4]
            -1.57875035e+03 * invT;
        /*species 32: C5H9 */
        species[32] =
            -2.38013950e+00
            +2.78804243e-02 * tc[1]
            -1.23381309e-05 * tc[2]
            +3.17209752e-09 * tc[3]
            -3.57077670e-13 * tc[4]
            +1.25589824e+04 * invT;
        /*species 33: C5H10X1 */
        species[33] =
            -2.06223481e+00
            +2.87109147e-02 * tc[1]
            -1.24828963e-05 * tc[2]
            +3.18412472e-09 * tc[3]
            -3.59219578e-13 * tc[4]
            -4.46546666e+03 * invT;
        /*species 35: C6H12X1 */
        species[35] =
            -2.35275205e+00
            +3.49327713e-02 * tc[1]
            -1.53136007e-05 * tc[2]
            +3.92418358e-09 * tc[3]
            -4.42592350e-13 * tc[4]
            -7.34368617e+03 * invT;
    } else {
        /*species 28: C4H7 */
        species[28] =
            +1.02103578e+01
            +8.02415980e-03 * tc[1]
            -1.82167431e-06 * tc[2]
            +2.11485263e-10 * tc[3]
            -9.79545478e-15 * tc[4]
            +1.09041937e+04 * invT;
        /*species 29: C4H8X1 */
        species[29] =
            +1.03508668e+01
            +9.03089385e-03 * tc[1]
            -2.05364343e-06 * tc[2]
            +2.38663240e-10 * tc[3]
            -1.10617928e-14 * tc[4]
            -5.97871038e+03 * invT;
        /*species 32: C5H9 */
        species[32] =
            +1.31860454e+01
            +1.03564449e-02 * tc[1]
            -2.35653539e-06 * tc[2]
            +2.74017833e-10 * tc[3]
            -1.27064442e-14 * tc[4]
            +7.00496135e+03 * invT;
        /*species 33: C5H10X1 */
        species[33] =
            +1.35851539e+01
            +1.12036235e-02 * tc[1]
            -2.54449342e-06 * tc[2]
            +2.95472415e-10 * tc[3]
            -1.36877028e-14 * tc[4]
            -1.00898205e+04 * invT;
        /*species 35: C6H12X1 */
        species[35] =
            +1.68337529e+01
            +1.33688829e-02 * tc[1]
            -3.03345591e-06 * tc[2]
            +3.52049420e-10 * tc[3]
            -1.63024849e-14 * tc[4]
            -1.42062860e+04 * invT;
    }

    /*species with midpoint at T=1395 kelvin */
    if (T < 1395) {
        /*species 30: PXC4H9 */
        species[30] =
            -1.43777972e+00
            +2.39486182e-02 * tc[1]
            -1.04674386e-05 * tc[2]
            +2.74466180e-09 * tc[3]
            -3.24021328e-13 * tc[4]
            +7.68945248e+03 * invT;
    } else {
        /*species 30: PXC4H9 */
        species[30] =
            +1.11510082e+01
            +9.71553585e-03 * tc[1]
            -2.20525983e-06 * tc[2]
            +2.55937840e-10 * tc[3]
            -1.18505941e-14 * tc[4]
            +3.17231942e+03 * invT;
    }

    /*species with midpoint at T=1382 kelvin */
    if (T < 1382) {
        /*species 36: C7H15X2 */
        species[36] =
            -1.03791558e+00
            +3.78363285e-02 * tc[1]
            -1.35824545e-05 * tc[2]
            +2.33169736e-09 * tc[3]
            -9.84721490e-14 * tc[4]
            -2.35605303e+03 * invT;
    } else {
        /*species 36: C7H15X2 */
        species[36] =
            +2.06368842e+01
            +1.61662402e-02 * tc[1]
            -3.64246023e-06 * tc[2]
            +4.20892650e-10 * tc[3]
            -1.94354818e-14 * tc[4]
            -1.05873616e+04 * invT;
    }

    /*species with midpoint at T=1391 kelvin */
    if (T < 1391) {
        /*species 37: NXC7H16 */
        species[37] =
            -2.26836187e+00
            +4.27177910e-02 * tc[1]
            -1.75115595e-05 * tc[2]
            +4.07364302e-09 * tc[3]
            -4.04789850e-13 * tc[4]
            -2.56586565e+04 * invT;
    } else {
        /*species 37: NXC7H16 */
        species[37] =
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
        /*species 19: C2H2 */
        species[19] =
            +2.01356200e+00
            +7.59522500e-03 * tc[1]
            -5.38773000e-06 * tc[2]
            +2.26974800e-09 * tc[3]
            -3.82549200e-13 * tc[4]
            +2.61244400e+04 * invT;
        /*species 21: C2H4 */
        species[21] =
            -8.61488000e-01
            +1.39808150e-02 * tc[1]
            -1.12955900e-05 * tc[2]
            +6.96288000e-09 * tc[3]
            -1.94757580e-12 * tc[4]
            +5.57304600e+03 * invT;
        /*species 22: C3H3 */
        species[22] =
            +4.75420000e+00
            +5.54014000e-03 * tc[1]
            +9.31107667e-08 * tc[2]
            -1.36980300e-09 * tc[3]
            +3.89925800e-13 * tc[4]
            +3.98888300e+04 * invT;
        /*species 25: NXC3H7 */
        species[25] =
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
        /*species 19: C2H2 */
        species[19] =
            +4.43677000e+00
            +2.68801950e-03 * tc[1]
            -6.37605667e-07 * tc[2]
            +8.21594750e-11 * tc[3]
            -4.31342000e-15 * tc[4]
            +2.56676600e+04 * invT;
        /*species 21: C2H4 */
        species[21] =
            +3.52841900e+00
            +5.74259000e-03 * tc[1]
            -1.47279500e-06 * tc[2]
            +1.96115025e-10 * tc[3]
            -1.05336960e-14 * tc[4]
            +4.42828900e+03 * invT;
        /*species 22: C3H3 */
        species[22] =
            +8.83104700e+00
            +2.17859750e-03 * tc[1]
            -1.36968900e-07 * tc[2]
            -5.92180750e-11 * tc[3]
            +8.75304000e-15 * tc[4]
            +3.84742000e+04 * invT;
        /*species 25: NXC3H7 */
        species[25] =
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
        /*species 26: NXC3H7O2 */
        species[26] =
            +2.10731492e+00
            +1.98082493e-02 * tc[1]
            -8.31638663e-06 * tc[2]
            +2.14862575e-09 * tc[3]
            -2.62480660e-13 * tc[4]
            -7.93745567e+03 * invT;
    } else {
        /*species 15: C2H6 */
        species[15] =
            +6.10683385e+00
            +6.46181805e-03 * tc[1]
            -1.47509065e-06 * tc[2]
            +1.71847932e-10 * tc[3]
            -7.97803464e-15 * tc[4]
            -1.37500014e+04 * invT;
        /*species 26: NXC3H7O2 */
        species[26] =
            +1.26327059e+01
            +8.49553630e-03 * tc[1]
            -1.96288958e-06 * tc[2]
            +2.30548849e-10 * tc[3]
            -1.07646135e-14 * tc[4]
            -1.19194652e+04 * invT;
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

    /*species with midpoint at T=1385 kelvin */
    if (T < 1385) {
        /*species 18: CH3O2 */
        species[18] =
            +4.26146906e+00
            +5.04367995e-03 * tc[1]
            -1.07168728e-06 * tc[2]
            +5.23523168e-11 * tc[3]
            +8.36678206e-15 * tc[4]
            -6.84394259e+02 * invT;
        /*species 31: PXC4H9O2 */
        species[31] =
            +1.94363650e+00
            +2.57756581e-02 * tc[1]
            -1.09428133e-05 * tc[2]
            +2.82662150e-09 * tc[3]
            -3.40237212e-13 * tc[4]
            -1.08358103e+04 * invT;
    } else {
        /*species 18: CH3O2 */
        species[18] =
            +5.95787891e+00
            +3.95364313e-03 * tc[1]
            -8.94154113e-07 * tc[2]
            +1.03472834e-10 * tc[3]
            -4.78014660e-15 * tc[4]
            -1.53574838e+03 * invT;
        /*species 31: PXC4H9O2 */
        species[31] =
            +1.57845448e+01
            +1.07605455e-02 * tc[1]
            -2.48303006e-06 * tc[2]
            +2.91395178e-10 * tc[3]
            -1.35977122e-14 * tc[4]
            -1.60146054e+04 * invT;
    }

    /*species with midpoint at T=1388 kelvin */
    if (T < 1388) {
        /*species 20: C3H6 */
        species[20] =
            +3.94615444e-01
            +1.44553831e-02 * tc[1]
            -5.16289360e-06 * tc[2]
            +9.72035522e-10 * tc[3]
            -6.75780704e-14 * tc[4]
            +1.06688164e+03 * invT;
    } else {
        /*species 20: C3H6 */
        species[20] =
            +8.01595958e+00
            +6.85118170e-03 * tc[1]
            -1.55416578e-06 * tc[2]
            +1.80313601e-10 * tc[3]
            -8.34740252e-15 * tc[4]
            -1.87821271e+03 * invT;
    }

    /*species with midpoint at T=1400 kelvin */
    if (T < 1400) {
        /*species 23: C3H4XA */
        species[23] =
            +2.53983100e+00
            +8.16718500e-03 * tc[1]
            -5.88316667e-07 * tc[2]
            -1.16184125e-09 * tc[3]
            +3.45826200e-13 * tc[4]
            +2.25124300e+04 * invT;
    } else {
        /*species 23: C3H4XA */
        species[23] =
            +9.77625600e+00
            +2.65106900e-03 * tc[1]
            -1.23370600e-07 * tc[2]
            -7.56596500e-11 * tc[3]
            +1.01791620e-14 * tc[4]
            +1.95497200e+04 * invT;
    }

    /*species with midpoint at T=1397 kelvin */
    if (T < 1397) {
        /*species 24: C3H5XA */
        species[24] =
            -5.29131958e-01
            +1.67279550e-02 * tc[1]
            -8.44670090e-06 * tc[2]
            +2.57164385e-09 * tc[3]
            -3.46516680e-13 * tc[4]
            +1.93834226e+04 * invT;
        /*species 34: C5H11X1 */
        species[34] =
            -9.05255912e-01
            +3.05316426e-02 * tc[1]
            -1.36497275e-05 * tc[2]
            +3.65233675e-09 * tc[3]
            -4.37719230e-13 * tc[4]
            +4.83995303e+03 * invT;
    } else {
        /*species 24: C3H5XA */
        species[24] =
            +8.45883958e+00
            +5.63477415e-03 * tc[1]
            -1.27930955e-06 * tc[2]
            +1.48514780e-10 * tc[3]
            -6.87836060e-15 * tc[4]
            +1.63576092e+04 * invT;
        /*species 34: C5H11X1 */
        species[34] =
            +1.53234740e+01
            +1.19520600e-02 * tc[1]
            -2.71590540e-06 * tc[2]
            +3.15440590e-10 * tc[3]
            -1.46135467e-14 * tc[4]
            -9.23241637e+02 * invT;
    }

    /*species with midpoint at T=1398 kelvin */
    if (T < 1398) {
        /*species 27: C4H6 */
        species[27] =
            -1.43095121e+00
            +2.39353031e-02 * tc[1]
            -1.38482267e-05 * tc[2]
            +4.78873880e-09 * tc[3]
            -7.14317014e-13 * tc[4]
            +1.17551314e+04 * invT;
    } else {
        /*species 27: C4H6 */
        species[27] =
            +1.11633789e+01
            +6.85819825e-03 * tc[1]
            -1.56571928e-06 * tc[2]
            +1.82423459e-10 * tc[3]
            -8.46972406e-15 * tc[4]
            +7.79039770e+03 * invT;
    }

    /*species with midpoint at T=1392 kelvin */
    if (T < 1392) {
        /*species 28: C4H7 */
        species[28] =
            -3.50508352e-01
            +2.13255622e-02 * tc[1]
            -9.69931243e-06 * tc[2]
            +2.63509785e-09 * tc[3]
            -3.20119708e-13 * tc[4]
            +1.49933591e+04 * invT;
        /*species 29: C4H8X1 */
        species[29] =
            -8.31372089e-01
            +2.26290489e-02 * tc[1]
            -9.78861863e-06 * tc[2]
            +2.50551090e-09 * tc[3]
            -2.86383360e-13 * tc[4]
            -1.57875035e+03 * invT;
        /*species 32: C5H9 */
        species[32] =
            -1.38013950e+00
            +2.78804243e-02 * tc[1]
            -1.23381309e-05 * tc[2]
            +3.17209752e-09 * tc[3]
            -3.57077670e-13 * tc[4]
            +1.25589824e+04 * invT;
        /*species 33: C5H10X1 */
        species[33] =
            -1.06223481e+00
            +2.87109147e-02 * tc[1]
            -1.24828963e-05 * tc[2]
            +3.18412472e-09 * tc[3]
            -3.59219578e-13 * tc[4]
            -4.46546666e+03 * invT;
        /*species 35: C6H12X1 */
        species[35] =
            -1.35275205e+00
            +3.49327713e-02 * tc[1]
            -1.53136007e-05 * tc[2]
            +3.92418358e-09 * tc[3]
            -4.42592350e-13 * tc[4]
            -7.34368617e+03 * invT;
    } else {
        /*species 28: C4H7 */
        species[28] =
            +1.12103578e+01
            +8.02415980e-03 * tc[1]
            -1.82167431e-06 * tc[2]
            +2.11485263e-10 * tc[3]
            -9.79545478e-15 * tc[4]
            +1.09041937e+04 * invT;
        /*species 29: C4H8X1 */
        species[29] =
            +1.13508668e+01
            +9.03089385e-03 * tc[1]
            -2.05364343e-06 * tc[2]
            +2.38663240e-10 * tc[3]
            -1.10617928e-14 * tc[4]
            -5.97871038e+03 * invT;
        /*species 32: C5H9 */
        species[32] =
            +1.41860454e+01
            +1.03564449e-02 * tc[1]
            -2.35653539e-06 * tc[2]
            +2.74017833e-10 * tc[3]
            -1.27064442e-14 * tc[4]
            +7.00496135e+03 * invT;
        /*species 33: C5H10X1 */
        species[33] =
            +1.45851539e+01
            +1.12036235e-02 * tc[1]
            -2.54449342e-06 * tc[2]
            +2.95472415e-10 * tc[3]
            -1.36877028e-14 * tc[4]
            -1.00898205e+04 * invT;
        /*species 35: C6H12X1 */
        species[35] =
            +1.78337529e+01
            +1.33688829e-02 * tc[1]
            -3.03345591e-06 * tc[2]
            +3.52049420e-10 * tc[3]
            -1.63024849e-14 * tc[4]
            -1.42062860e+04 * invT;
    }

    /*species with midpoint at T=1395 kelvin */
    if (T < 1395) {
        /*species 30: PXC4H9 */
        species[30] =
            -4.37779725e-01
            +2.39486182e-02 * tc[1]
            -1.04674386e-05 * tc[2]
            +2.74466180e-09 * tc[3]
            -3.24021328e-13 * tc[4]
            +7.68945248e+03 * invT;
    } else {
        /*species 30: PXC4H9 */
        species[30] =
            +1.21510082e+01
            +9.71553585e-03 * tc[1]
            -2.20525983e-06 * tc[2]
            +2.55937840e-10 * tc[3]
            -1.18505941e-14 * tc[4]
            +3.17231942e+03 * invT;
    }

    /*species with midpoint at T=1382 kelvin */
    if (T < 1382) {
        /*species 36: C7H15X2 */
        species[36] =
            -3.79155767e-02
            +3.78363285e-02 * tc[1]
            -1.35824545e-05 * tc[2]
            +2.33169736e-09 * tc[3]
            -9.84721490e-14 * tc[4]
            -2.35605303e+03 * invT;
    } else {
        /*species 36: C7H15X2 */
        species[36] =
            +2.16368842e+01
            +1.61662402e-02 * tc[1]
            -3.64246023e-06 * tc[2]
            +4.20892650e-10 * tc[3]
            -1.94354818e-14 * tc[4]
            -1.05873616e+04 * invT;
    }

    /*species with midpoint at T=1391 kelvin */
    if (T < 1391) {
        /*species 37: NXC7H16 */
        species[37] =
            -1.26836187e+00
            +4.27177910e-02 * tc[1]
            -1.75115595e-05 * tc[2]
            +4.07364302e-09 * tc[3]
            -4.04789850e-13 * tc[4]
            -2.56586565e+04 * invT;
    } else {
        /*species 37: NXC7H16 */
        species[37] =
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
        /*species 19: C2H2 */
        species[19] =
            +2.01356200e+00 * tc[0]
            +1.51904500e-02 * tc[1]
            -8.08159500e-06 * tc[2]
            +3.02633067e-09 * tc[3]
            -4.78186500e-13 * tc[4]
            +8.80537800e+00 ;
        /*species 21: C2H4 */
        species[21] =
            -8.61488000e-01 * tc[0]
            +2.79616300e-02 * tc[1]
            -1.69433850e-05 * tc[2]
            +9.28384000e-09 * tc[3]
            -2.43446975e-12 * tc[4]
            +2.42114900e+01 ;
        /*species 22: C3H3 */
        species[22] =
            +4.75420000e+00 * tc[0]
            +1.10802800e-02 * tc[1]
            +1.39666150e-07 * tc[2]
            -1.82640400e-09 * tc[3]
            +4.87407250e-13 * tc[4]
            +5.85454900e-01 ;
        /*species 25: NXC3H7 */
        species[25] =
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
        /*species 19: C2H2 */
        species[19] =
            +4.43677000e+00 * tc[0]
            +5.37603900e-03 * tc[1]
            -9.56408500e-07 * tc[2]
            +1.09545967e-10 * tc[3]
            -5.39177500e-15 * tc[4]
            -2.80033800e+00 ;
        /*species 21: C2H4 */
        species[21] =
            +3.52841900e+00 * tc[0]
            +1.14851800e-02 * tc[1]
            -2.20919250e-06 * tc[2]
            +2.61486700e-10 * tc[3]
            -1.31671200e-14 * tc[4]
            +2.23038900e+00 ;
        /*species 22: C3H3 */
        species[22] =
            +8.83104700e+00 * tc[0]
            +4.35719500e-03 * tc[1]
            -2.05453350e-07 * tc[2]
            -7.89574333e-11 * tc[3]
            +1.09413000e-14 * tc[4]
            -2.17791900e+01 ;
        /*species 25: NXC3H7 */
        species[25] =
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
        /*species 26: NXC3H7O2 */
        species[26] =
            +2.10731492e+00 * tc[0]
            +3.96164986e-02 * tc[1]
            -1.24745800e-05 * tc[2]
            +2.86483433e-09 * tc[3]
            -3.28100825e-13 * tc[4]
            +1.89082748e+01 ;
    } else {
        /*species 15: C2H6 */
        species[15] =
            +6.10683385e+00 * tc[0]
            +1.29236361e-02 * tc[1]
            -2.21263598e-06 * tc[2]
            +2.29130575e-10 * tc[3]
            -9.97254330e-15 * tc[4]
            -1.30081250e+01 ;
        /*species 26: NXC3H7O2 */
        species[26] =
            +1.26327059e+01 * tc[0]
            +1.69910726e-02 * tc[1]
            -2.94433436e-06 * tc[2]
            +3.07398465e-10 * tc[3]
            -1.34557669e-14 * tc[4]
            -3.85349297e+01 ;
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

    /*species with midpoint at T=1385 kelvin */
    if (T < 1385) {
        /*species 18: CH3O2 */
        species[18] =
            +4.26146906e+00 * tc[0]
            +1.00873599e-02 * tc[1]
            -1.60753092e-06 * tc[2]
            +6.98030890e-11 * tc[3]
            +1.04584776e-14 * tc[4]
            +5.16330320e+00 ;
        /*species 31: PXC4H9O2 */
        species[31] =
            +1.94363650e+00 * tc[0]
            +5.15513163e-02 * tc[1]
            -1.64142200e-05 * tc[2]
            +3.76882867e-09 * tc[3]
            -4.25296515e-13 * tc[4]
            +2.13503149e+01 ;
    } else {
        /*species 18: CH3O2 */
        species[18] =
            +5.95787891e+00 * tc[0]
            +7.90728626e-03 * tc[1]
            -1.34123117e-06 * tc[2]
            +1.37963779e-10 * tc[3]
            -5.97518325e-15 * tc[4]
            -4.71963886e+00 ;
        /*species 31: PXC4H9O2 */
        species[31] =
            +1.57845448e+01 * tc[0]
            +2.15210910e-02 * tc[1]
            -3.72454509e-06 * tc[2]
            +3.88526903e-10 * tc[3]
            -1.69971402e-14 * tc[4]
            -5.40388525e+01 ;
    }

    /*species with midpoint at T=1388 kelvin */
    if (T < 1388) {
        /*species 20: C3H6 */
        species[20] =
            +3.94615444e-01 * tc[0]
            +2.89107662e-02 * tc[1]
            -7.74434040e-06 * tc[2]
            +1.29604736e-09 * tc[3]
            -8.44725880e-14 * tc[4]
            +2.19003736e+01 ;
    } else {
        /*species 20: C3H6 */
        species[20] =
            +8.01595958e+00 * tc[0]
            +1.37023634e-02 * tc[1]
            -2.33124867e-06 * tc[2]
            +2.40418134e-10 * tc[3]
            -1.04342532e-14 * tc[4]
            -2.00160668e+01 ;
    }

    /*species with midpoint at T=1400 kelvin */
    if (T < 1400) {
        /*species 23: C3H4XA */
        species[23] =
            +2.53983100e+00 * tc[0]
            +1.63343700e-02 * tc[1]
            -8.82475000e-07 * tc[2]
            -1.54912167e-09 * tc[3]
            +4.32282750e-13 * tc[4]
            +9.93570200e+00 ;
    } else {
        /*species 23: C3H4XA */
        species[23] =
            +9.77625600e+00 * tc[0]
            +5.30213800e-03 * tc[1]
            -1.85055900e-07 * tc[2]
            -1.00879533e-10 * tc[3]
            +1.27239525e-14 * tc[4]
            -3.07706100e+01 ;
    }

    /*species with midpoint at T=1397 kelvin */
    if (T < 1397) {
        /*species 24: C3H5XA */
        species[24] =
            -5.29131958e-01 * tc[0]
            +3.34559100e-02 * tc[1]
            -1.26700514e-05 * tc[2]
            +3.42885847e-09 * tc[3]
            -4.33145850e-13 * tc[4]
            +2.53067131e+01 ;
        /*species 34: C5H11X1 */
        species[34] =
            -9.05255912e-01 * tc[0]
            +6.10632852e-02 * tc[1]
            -2.04745912e-05 * tc[2]
            +4.86978233e-09 * tc[3]
            -5.47149037e-13 * tc[4]
            +3.25574963e+01 ;
    } else {
        /*species 24: C3H5XA */
        species[24] =
            +8.45883958e+00 * tc[0]
            +1.12695483e-02 * tc[1]
            -1.91896432e-06 * tc[2]
            +1.98019706e-10 * tc[3]
            -8.59795075e-15 * tc[4]
            -2.25809450e+01 ;
        /*species 34: C5H11X1 */
        species[34] =
            +1.53234740e+01 * tc[0]
            +2.39041200e-02 * tc[1]
            -4.07385810e-06 * tc[2]
            +4.20587453e-10 * tc[3]
            -1.82669334e-14 * tc[4]
            -5.49528859e+01 ;
    }

    /*species with midpoint at T=1398 kelvin */
    if (T < 1398) {
        /*species 27: C4H6 */
        species[27] =
            -1.43095121e+00 * tc[0]
            +4.78706062e-02 * tc[1]
            -2.07723400e-05 * tc[2]
            +6.38498507e-09 * tc[3]
            -8.92896267e-13 * tc[4]
            +2.90825833e+01 ;
    } else {
        /*species 27: C4H6 */
        species[27] =
            +1.11633789e+01 * tc[0]
            +1.37163965e-02 * tc[1]
            -2.34857892e-06 * tc[2]
            +2.43231279e-10 * tc[3]
            -1.05871551e-14 * tc[4]
            -3.69847949e+01 ;
    }

    /*species with midpoint at T=1392 kelvin */
    if (T < 1392) {
        /*species 28: C4H7 */
        species[28] =
            -3.50508352e-01 * tc[0]
            +4.26511243e-02 * tc[1]
            -1.45489687e-05 * tc[2]
            +3.51346380e-09 * tc[3]
            -4.00149635e-13 * tc[4]
            +2.67295696e+01 ;
        /*species 29: C4H8X1 */
        species[29] =
            -8.31372089e-01 * tc[0]
            +4.52580978e-02 * tc[1]
            -1.46829280e-05 * tc[2]
            +3.34068120e-09 * tc[3]
            -3.57979200e-13 * tc[4]
            +2.95084236e+01 ;
        /*species 32: C5H9 */
        species[32] =
            -1.38013950e+00 * tc[0]
            +5.57608487e-02 * tc[1]
            -1.85071964e-05 * tc[2]
            +4.22946337e-09 * tc[3]
            -4.46347087e-13 * tc[4]
            +3.26441304e+01 ;
        /*species 33: C5H10X1 */
        species[33] =
            -1.06223481e+00 * tc[0]
            +5.74218294e-02 * tc[1]
            -1.87243445e-05 * tc[2]
            +4.24549963e-09 * tc[3]
            -4.49024472e-13 * tc[4]
            +3.22739790e+01 ;
        /*species 35: C6H12X1 */
        species[35] =
            -1.35275205e+00 * tc[0]
            +6.98655426e-02 * tc[1]
            -2.29704011e-05 * tc[2]
            +5.23224477e-09 * tc[3]
            -5.53240438e-13 * tc[4]
            +3.53120691e+01 ;
    } else {
        /*species 28: C4H7 */
        species[28] =
            +1.12103578e+01 * tc[0]
            +1.60483196e-02 * tc[1]
            -2.73251146e-06 * tc[2]
            +2.81980351e-10 * tc[3]
            -1.22443185e-14 * tc[4]
            -3.55593015e+01 ;
        /*species 29: C4H8X1 */
        species[29] =
            +1.13508668e+01 * tc[0]
            +1.80617877e-02 * tc[1]
            -3.08046515e-06 * tc[2]
            +3.18217653e-10 * tc[3]
            -1.38272410e-14 * tc[4]
            -3.64369438e+01 ;
        /*species 32: C5H9 */
        species[32] =
            +1.41860454e+01 * tc[0]
            +2.07128899e-02 * tc[1]
            -3.53480309e-06 * tc[2]
            +3.65357110e-10 * tc[3]
            -1.58830552e-14 * tc[4]
            -5.14501773e+01 ;
        /*species 33: C5H10X1 */
        species[33] =
            +1.45851539e+01 * tc[0]
            +2.24072471e-02 * tc[1]
            -3.81674012e-06 * tc[2]
            +3.93963220e-10 * tc[3]
            -1.71096285e-14 * tc[4]
            -5.23683936e+01 ;
        /*species 35: C6H12X1 */
        species[35] =
            +1.78337529e+01 * tc[0]
            +2.67377658e-02 * tc[1]
            -4.55018387e-06 * tc[2]
            +4.69399227e-10 * tc[3]
            -2.03781061e-14 * tc[4]
            -6.83818851e+01 ;
    }

    /*species with midpoint at T=1395 kelvin */
    if (T < 1395) {
        /*species 30: PXC4H9 */
        species[30] =
            -4.37779725e-01 * tc[0]
            +4.78972364e-02 * tc[1]
            -1.57011580e-05 * tc[2]
            +3.65954907e-09 * tc[3]
            -4.05026660e-13 * tc[4]
            +2.86852732e+01 ;
    } else {
        /*species 30: PXC4H9 */
        species[30] =
            +1.21510082e+01 * tc[0]
            +1.94310717e-02 * tc[1]
            -3.30788975e-06 * tc[2]
            +3.41250453e-10 * tc[3]
            -1.48132427e-14 * tc[4]
            -3.93425822e+01 ;
    }

    /*species with midpoint at T=1382 kelvin */
    if (T < 1382) {
        /*species 36: C7H15X2 */
        species[36] =
            -3.79155767e-02 * tc[0]
            +7.56726570e-02 * tc[1]
            -2.03736817e-05 * tc[2]
            +3.10892981e-09 * tc[3]
            -1.23090186e-13 * tc[4]
            +3.37321506e+01 ;
    } else {
        /*species 36: C7H15X2 */
        species[36] =
            +2.16368842e+01 * tc[0]
            +3.23324804e-02 * tc[1]
            -5.46369035e-06 * tc[2]
            +5.61190200e-10 * tc[3]
            -2.42943523e-14 * tc[4]
            -8.52209653e+01 ;
    }

    /*species with midpoint at T=1391 kelvin */
    if (T < 1391) {
        /*species 37: NXC7H16 */
        species[37] =
            -1.26836187e+00 * tc[0]
            +8.54355820e-02 * tc[1]
            -2.62673393e-05 * tc[2]
            +5.43152403e-09 * tc[3]
            -5.05987313e-13 * tc[4]
            +3.53732912e+01 ;
    } else {
        /*species 37: NXC7H16 */
        species[37] =
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
    *LENIMC = 154;}


void egtransetLENRMC(int* LENRMC ) {
    *LENRMC = 28766;}


void egtransetNO(int* NO ) {
    *NO = 4;}


void egtransetKK(int* KK ) {
    *KK = 38;}


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
    WT[18] = 4.70338600E+01;
    WT[19] = 2.60382400E+01;
    WT[20] = 4.20812700E+01;
    WT[21] = 2.80541800E+01;
    WT[22] = 3.90573600E+01;
    WT[23] = 4.00653300E+01;
    WT[24] = 4.10733000E+01;
    WT[25] = 4.30892400E+01;
    WT[26] = 7.50880400E+01;
    WT[27] = 5.40924200E+01;
    WT[28] = 5.51003900E+01;
    WT[29] = 5.61083600E+01;
    WT[30] = 5.71163300E+01;
    WT[31] = 8.91151300E+01;
    WT[32] = 6.91274800E+01;
    WT[33] = 7.01354500E+01;
    WT[34] = 7.11434200E+01;
    WT[35] = 8.41625400E+01;
    WT[36] = 9.91976000E+01;
    WT[37] = 1.00205570E+02;
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
    EPS[18] = 4.81800000E+02;
    EPS[19] = 2.65300000E+02;
    EPS[20] = 3.07800000E+02;
    EPS[21] = 2.38400000E+02;
    EPS[22] = 3.24800000E+02;
    EPS[23] = 3.24800000E+02;
    EPS[24] = 3.16000000E+02;
    EPS[25] = 3.03400000E+02;
    EPS[26] = 4.81500000E+02;
    EPS[27] = 3.57000000E+02;
    EPS[28] = 3.55000000E+02;
    EPS[29] = 3.55000000E+02;
    EPS[30] = 3.52000000E+02;
    EPS[31] = 4.96000000E+02;
    EPS[32] = 3.96800000E+02;
    EPS[33] = 3.86200000E+02;
    EPS[34] = 4.40735000E+02;
    EPS[35] = 4.85857000E+02;
    EPS[36] = 4.59600000E+02;
    EPS[37] = 4.59600000E+02;
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
    SIG[18] = 3.62600000E+00;
    SIG[19] = 3.72100000E+00;
    SIG[20] = 4.14000000E+00;
    SIG[21] = 3.49600000E+00;
    SIG[22] = 4.29000000E+00;
    SIG[23] = 4.29000000E+00;
    SIG[24] = 4.22000000E+00;
    SIG[25] = 4.81000000E+00;
    SIG[26] = 4.99700000E+00;
    SIG[27] = 4.72000000E+00;
    SIG[28] = 4.65000000E+00;
    SIG[29] = 4.65000000E+00;
    SIG[30] = 5.24000000E+00;
    SIG[31] = 5.20000000E+00;
    SIG[32] = 5.45800000E+00;
    SIG[33] = 5.48900000E+00;
    SIG[34] = 5.04100000E+00;
    SIG[35] = 5.32800000E+00;
    SIG[36] = 6.25300000E+00;
    SIG[37] = 6.25300000E+00;
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
    DIP[25] = 0.00000000E+00;
    DIP[26] = 1.70000000E+00;
    DIP[27] = 0.00000000E+00;
    DIP[28] = 0.00000000E+00;
    DIP[29] = 0.00000000E+00;
    DIP[30] = 0.00000000E+00;
    DIP[31] = 0.00000000E+00;
    DIP[32] = 0.00000000E+00;
    DIP[33] = 4.00000000E-01;
    DIP[34] = 0.00000000E+00;
    DIP[35] = 0.00000000E+00;
    DIP[36] = 0.00000000E+00;
    DIP[37] = 0.00000000E+00;
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
    ZROT[18] = 1.00000000E+00;
    ZROT[19] = 2.50000000E+00;
    ZROT[20] = 1.00000000E+00;
    ZROT[21] = 1.50000000E+00;
    ZROT[22] = 1.00000000E+00;
    ZROT[23] = 1.00000000E+00;
    ZROT[24] = 1.00000000E+00;
    ZROT[25] = 1.00000000E+00;
    ZROT[26] = 1.00000000E+00;
    ZROT[27] = 1.00000000E+00;
    ZROT[28] = 1.00000000E+00;
    ZROT[29] = 1.00000000E+00;
    ZROT[30] = 1.00000000E+00;
    ZROT[31] = 1.00000000E+00;
    ZROT[32] = 1.00000000E+00;
    ZROT[33] = 1.00000000E+00;
    ZROT[34] = 0.00000000E+00;
    ZROT[35] = 0.00000000E+00;
    ZROT[36] = 1.00000000E+00;
    ZROT[37] = 1.00000000E+00;
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
    NLIN[18] = 2;
    NLIN[19] = 1;
    NLIN[20] = 2;
    NLIN[21] = 2;
    NLIN[22] = 1;
    NLIN[23] = 1;
    NLIN[24] = 2;
    NLIN[25] = 2;
    NLIN[26] = 2;
    NLIN[27] = 2;
    NLIN[28] = 2;
    NLIN[29] = 2;
    NLIN[30] = 2;
    NLIN[31] = 2;
    NLIN[32] = 2;
    NLIN[33] = 2;
    NLIN[34] = 2;
    NLIN[35] = 2;
    NLIN[36] = 2;
    NLIN[37] = 2;
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
    COFETA[72] = -2.03725491E+01;
    COFETA[73] = 3.03946431E+00;
    COFETA[74] = -2.16994867E-01;
    COFETA[75] = 5.61394012E-03;
    COFETA[76] = -2.47697856E+01;
    COFETA[77] = 5.30039568E+00;
    COFETA[78] = -5.89273639E-01;
    COFETA[79] = 2.49261407E-02;
    COFETA[80] = -2.49727893E+01;
    COFETA[81] = 5.27067543E+00;
    COFETA[82] = -5.71909526E-01;
    COFETA[83] = 2.36230940E-02;
    COFETA[84] = -2.39690472E+01;
    COFETA[85] = 5.11436059E+00;
    COFETA[86] = -5.71999954E-01;
    COFETA[87] = 2.44581334E-02;
    COFETA[88] = -2.50327383E+01;
    COFETA[89] = 5.20184077E+00;
    COFETA[90] = -5.57265947E-01;
    COFETA[91] = 2.27565676E-02;
    COFETA[92] = -2.50199983E+01;
    COFETA[93] = 5.20184077E+00;
    COFETA[94] = -5.57265947E-01;
    COFETA[95] = 2.27565676E-02;
    COFETA[96] = -2.50402232E+01;
    COFETA[97] = 5.25451220E+00;
    COFETA[98] = -5.67228955E-01;
    COFETA[99] = 2.33156489E-02;
    COFETA[100] = -2.52462994E+01;
    COFETA[101] = 5.27749097E+00;
    COFETA[102] = -5.74219215E-01;
    COFETA[103] = 2.37811608E-02;
    COFETA[104] = -2.00330714E+01;
    COFETA[105] = 2.73318358E+00;
    COFETA[106] = -1.75653565E-01;
    COFETA[107] = 3.77126610E-03;
    COFETA[108] = -2.46654710E+01;
    COFETA[109] = 4.94595777E+00;
    COFETA[110] = -5.12278955E-01;
    COFETA[111] = 2.03286378E-02;
    COFETA[112] = -2.46566816E+01;
    COFETA[113] = 4.96413364E+00;
    COFETA[114] = -5.15375011E-01;
    COFETA[115] = 2.04926972E-02;
    COFETA[116] = -2.46476176E+01;
    COFETA[117] = 4.96413364E+00;
    COFETA[118] = -5.15375011E-01;
    COFETA[119] = 2.04926972E-02;
    COFETA[120] = -2.49295265E+01;
    COFETA[121] = 4.99407558E+00;
    COFETA[122] = -5.20389056E-01;
    COFETA[123] = 2.07557283E-02;
    COFETA[124] = -2.01225145E+01;
    COFETA[125] = 2.73742901E+00;
    COFETA[126] = -1.72182607E-01;
    COFETA[127] = 3.44858406E-03;
    COFETA[128] = -2.39731890E+01;
    COFETA[129] = 4.48124637E+00;
    COFETA[130] = -4.36934973E-01;
    COFETA[131] = 1.64568564E-02;
    COFETA[132] = -2.42040008E+01;
    COFETA[133] = 4.60377392E+00;
    COFETA[134] = -4.56799058E-01;
    COFETA[135] = 1.74771833E-02;
    COFETA[136] = -2.23900786E+01;
    COFETA[137] = 3.78460980E+00;
    COFETA[138] = -3.29370574E-01;
    COFETA[139] = 1.11069352E-02;
    COFETA[140] = -2.06651397E+01;
    COFETA[141] = 2.95298213E+00;
    COFETA[142] = -2.04141647E-01;
    COFETA[143] = 4.99194219E-03;
    COFETA[144] = -2.19891717E+01;
    COFETA[145] = 3.46341268E+00;
    COFETA[146] = -2.80516687E-01;
    COFETA[147] = 8.70427548E-03;
    COFETA[148] = -2.19841167E+01;
    COFETA[149] = 3.46341268E+00;
    COFETA[150] = -2.80516687E-01;
    COFETA[151] = 8.70427548E-03;
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
    COFLAM[72] = 5.28056570E+00;
    COFLAM[73] = -1.92758693E+00;
    COFLAM[74] = 6.29141169E-01;
    COFLAM[75] = -3.87203309E-02;
    COFLAM[76] = -9.20687365E+00;
    COFLAM[77] = 5.13028609E+00;
    COFLAM[78] = -4.67868863E-01;
    COFLAM[79] = 1.64674383E-02;
    COFLAM[80] = -1.54410770E+01;
    COFLAM[81] = 6.67114766E+00;
    COFLAM[82] = -5.37137624E-01;
    COFLAM[83] = 1.38051704E-02;
    COFLAM[84] = -1.34447168E+01;
    COFLAM[85] = 6.12380203E+00;
    COFLAM[86] = -4.86657425E-01;
    COFLAM[87] = 1.24614518E-02;
    COFLAM[88] = -9.95261562E+00;
    COFLAM[89] = 5.06627945E+00;
    COFLAM[90] = -4.15761512E-01;
    COFLAM[91] = 1.21197123E-02;
    COFLAM[92] = -1.32966554E+01;
    COFLAM[93] = 5.92585034E+00;
    COFLAM[94] = -4.64901365E-01;
    COFLAM[95] = 1.16662523E-02;
    COFLAM[96] = -2.08328673E+01;
    COFLAM[97] = 9.07593204E+00;
    COFLAM[98] = -8.93990863E-01;
    COFLAM[99] = 3.11142957E-02;
    COFLAM[100] = -1.59315280E+01;
    COFLAM[101] = 7.00975811E+00;
    COFLAM[102] = -6.12923034E-01;
    COFLAM[103] = 1.85439169E-02;
    COFLAM[104] = -9.04441654E+00;
    COFLAM[105] = 3.66855955E+00;
    COFLAM[106] = -1.04994889E-01;
    COFLAM[107] = -6.68408235E-03;
    COFLAM[108] = -2.26611414E+01;
    COFLAM[109] = 9.78565333E+00;
    COFLAM[110] = -9.94033497E-01;
    COFLAM[111] = 3.57950722E-02;
    COFLAM[112] = -1.90915971E+01;
    COFLAM[113] = 8.15678005E+00;
    COFLAM[114] = -7.46604137E-01;
    COFLAM[115] = 2.35743405E-02;
    COFLAM[116] = -1.96439129E+01;
    COFLAM[117] = 8.31169569E+00;
    COFLAM[118] = -7.56268608E-01;
    COFLAM[119] = 2.35727121E-02;
    COFLAM[120] = -1.83524587E+01;
    COFLAM[121] = 7.74571154E+00;
    COFLAM[122] = -6.83646893E-01;
    COFLAM[123] = 2.04759926E-02;
    COFLAM[124] = -1.04744531E+01;
    COFLAM[125] = 4.24091183E+00;
    COFLAM[126] = -1.78599110E-01;
    COFLAM[127] = -3.51607728E-03;
    COFLAM[128] = -2.13451516E+01;
    COFLAM[129] = 8.84739285E+00;
    COFLAM[130] = -8.23421444E-01;
    COFLAM[131] = 2.63782925E-02;
    COFLAM[132] = -1.98965346E+01;
    COFLAM[133] = 8.27532362E+00;
    COFLAM[134] = -7.46184920E-01;
    COFLAM[135] = 2.29240275E-02;
    COFLAM[136] = -1.77209721E+01;
    COFLAM[137] = 7.36216062E+00;
    COFLAM[138] = -6.11089543E-01;
    COFLAM[139] = 1.64273679E-02;
    COFLAM[140] = -1.80139132E+01;
    COFLAM[141] = 7.35713318E+00;
    COFLAM[142] = -6.00090479E-01;
    COFLAM[143] = 1.55359162E-02;
    COFLAM[144] = -1.72838492E+01;
    COFLAM[145] = 6.97737723E+00;
    COFLAM[146] = -5.47365626E-01;
    COFLAM[147] = 1.30795303E-02;
    COFLAM[148] = -1.79582416E+01;
    COFLAM[149] = 7.27686902E+00;
    COFLAM[150] = -5.88898453E-01;
    COFLAM[151] = 1.49980279E-02;
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
    COFD[72] = -2.04649069E+01;
    COFD[73] = 5.18856872E+00;
    COFD[74] = -4.50001829E-01;
    COFD[75] = 1.91636142E-02;
    COFD[76] = -1.83039618E+01;
    COFD[77] = 4.47952077E+00;
    COFD[78] = -3.66569471E-01;
    COFD[79] = 1.58916129E-02;
    COFD[80] = -1.90859283E+01;
    COFD[81] = 4.68079396E+00;
    COFD[82] = -3.91231550E-01;
    COFD[83] = 1.69021170E-02;
    COFD[84] = -1.78815889E+01;
    COFD[85] = 4.34347890E+00;
    COFD[86] = -3.49890003E-01;
    COFD[87] = 1.52083459E-02;
    COFD[88] = -1.92731067E+01;
    COFD[89] = 4.73660584E+00;
    COFD[90] = -3.97704978E-01;
    COFD[91] = 1.71514887E-02;
    COFD[92] = -1.92783884E+01;
    COFD[93] = 4.73660584E+00;
    COFD[94] = -3.97704978E-01;
    COFD[95] = 1.71514887E-02;
    COFD[96] = -1.91796663E+01;
    COFD[97] = 4.70714822E+00;
    COFD[98] = -3.94261134E-01;
    COFD[99] = 1.70175169E-02;
    COFD[100] = -1.92062897E+01;
    COFD[101] = 4.66318669E+00;
    COFD[102] = -3.89108667E-01;
    COFD[103] = 1.68165377E-02;
    COFD[104] = -2.09943481E+01;
    COFD[105] = 5.22468467E+00;
    COFD[106] = -4.54220128E-01;
    COFD[107] = 1.93281042E-02;
    COFD[108] = -1.97484166E+01;
    COFD[109] = 4.84231878E+00;
    COFD[110] = -4.10101001E-01;
    COFD[111] = 1.76356687E-02;
    COFD[112] = -1.97196489E+01;
    COFD[113] = 4.83750266E+00;
    COFD[114] = -4.09581452E-01;
    COFD[115] = 1.76174739E-02;
    COFD[116] = -1.97226856E+01;
    COFD[117] = 4.83750266E+00;
    COFD[118] = -4.09581452E-01;
    COFD[119] = 1.76174739E-02;
    COFD[120] = -1.98374654E+01;
    COFD[121] = 4.82871870E+00;
    COFD[122] = -4.08567726E-01;
    COFD[123] = 1.75785896E-02;
    COFD[124] = -2.10643735E+01;
    COFD[125] = 5.22604478E+00;
    COFD[126] = -4.54378127E-01;
    COFD[127] = 1.93342248E-02;
    COFD[128] = -2.03706752E+01;
    COFD[129] = 4.98803076E+00;
    COFD[130] = -4.27580621E-01;
    COFD[131] = 1.83363274E-02;
    COFD[132] = -2.02840235E+01;
    COFD[133] = 4.95484018E+00;
    COFD[134] = -4.23654881E-01;
    COFD[135] = 1.81813866E-02;
    COFD[136] = -2.06548278E+01;
    COFD[137] = 5.11678107E+00;
    COFD[138] = -4.42706538E-01;
    COFD[139] = 1.89296424E-02;
    COFD[140] = -2.10086887E+01;
    COFD[141] = 5.19953529E+00;
    COFD[142] = -4.51287802E-01;
    COFD[143] = 1.92140123E-02;
    COFD[144] = -2.10674485E+01;
    COFD[145] = 5.15027524E+00;
    COFD[146] = -4.46126111E-01;
    COFD[147] = 1.90401391E-02;
    COFD[148] = -2.10685573E+01;
    COFD[149] = 5.15027524E+00;
    COFD[150] = -4.46126111E-01;
    COFD[151] = 1.90401391E-02;
    COFD[152] = -1.40756935E+01;
    COFD[153] = 3.07549274E+00;
    COFD[154] = -1.88889344E-01;
    COFD[155] = 8.37152866E-03;
    COFD[156] = -1.32093628E+01;
    COFD[157] = 2.90778936E+00;
    COFD[158] = -1.67388544E-01;
    COFD[159] = 7.45220609E-03;
    COFD[160] = -1.09595712E+01;
    COFD[161] = 2.30836460E+00;
    COFD[162] = -8.76339315E-02;
    COFD[163] = 3.90878445E-03;
    COFD[164] = -1.34230272E+01;
    COFD[165] = 3.48624238E+00;
    COFD[166] = -2.41554467E-01;
    COFD[167] = 1.06263545E-02;
    COFD[168] = -1.32244035E+01;
    COFD[169] = 2.90778936E+00;
    COFD[170] = -1.67388544E-01;
    COFD[171] = 7.45220609E-03;
    COFD[172] = -1.94093572E+01;
    COFD[173] = 5.16013126E+00;
    COFD[174] = -4.46824543E-01;
    COFD[175] = 1.90464887E-02;
    COFD[176] = -1.43139231E+01;
    COFD[177] = 3.17651319E+00;
    COFD[178] = -2.02028974E-01;
    COFD[179] = 8.94232502E-03;
    COFD[180] = -1.43190389E+01;
    COFD[181] = 3.17651319E+00;
    COFD[182] = -2.02028974E-01;
    COFD[183] = 8.94232502E-03;
    COFD[184] = -1.43238998E+01;
    COFD[185] = 3.17651319E+00;
    COFD[186] = -2.02028974E-01;
    COFD[187] = 8.94232502E-03;
    COFD[188] = -1.70534856E+01;
    COFD[189] = 4.14240922E+00;
    COFD[190] = -3.25239774E-01;
    COFD[191] = 1.41980687E-02;
    COFD[192] = -1.40999008E+01;
    COFD[193] = 3.08120012E+00;
    COFD[194] = -1.89629903E-01;
    COFD[195] = 8.40361952E-03;
    COFD[196] = -1.94373127E+01;
    COFD[197] = 5.02567894E+00;
    COFD[198] = -4.32045169E-01;
    COFD[199] = 1.85132214E-02;
    COFD[200] = -1.50766130E+01;
    COFD[201] = 3.47945612E+00;
    COFD[202] = -2.40703722E-01;
    COFD[203] = 1.05907441E-02;
    COFD[204] = -1.50270339E+01;
    COFD[205] = 3.46140064E+00;
    COFD[206] = -2.38440092E-01;
    COFD[207] = 1.04960087E-02;
    COFD[208] = -1.93364585E+01;
    COFD[209] = 4.98286777E+00;
    COFD[210] = -4.26970814E-01;
    COFD[211] = 1.83122917E-02;
    COFD[212] = -1.72112971E+01;
    COFD[213] = 4.15807461E+00;
    COFD[214] = -3.27178539E-01;
    COFD[215] = 1.42784349E-02;
    COFD[216] = -1.90883268E+01;
    COFD[217] = 4.84384483E+00;
    COFD[218] = -4.10265575E-01;
    COFD[219] = 1.76414287E-02;
    COFD[220] = -1.91004157E+01;
    COFD[221] = 4.84384483E+00;
    COFD[222] = -4.10265575E-01;
    COFD[223] = 1.76414287E-02;
    COFD[224] = -1.93925667E+01;
    COFD[225] = 4.98286777E+00;
    COFD[226] = -4.26970814E-01;
    COFD[227] = 1.83122917E-02;
    COFD[228] = -1.72286007E+01;
    COFD[229] = 4.24084025E+00;
    COFD[230] = -3.37428619E-01;
    COFD[231] = 1.47032793E-02;
    COFD[232] = -1.79361160E+01;
    COFD[233] = 4.42139452E+00;
    COFD[234] = -3.59567329E-01;
    COFD[235] = 1.56103969E-02;
    COFD[236] = -1.68343393E+01;
    COFD[237] = 4.11954900E+00;
    COFD[238] = -3.22470391E-01;
    COFD[239] = 1.40859564E-02;
    COFD[240] = -1.81463104E+01;
    COFD[241] = 4.48398491E+00;
    COFD[242] = -3.67097129E-01;
    COFD[243] = 1.59123634E-02;
    COFD[244] = -1.81499793E+01;
    COFD[245] = 4.48398491E+00;
    COFD[246] = -3.67097129E-01;
    COFD[247] = 1.59123634E-02;
    COFD[248] = -1.80480958E+01;
    COFD[249] = 4.45434023E+00;
    COFD[250] = -3.63584633E-01;
    COFD[251] = 1.57739270E-02;
    COFD[252] = -1.80724788E+01;
    COFD[253] = 4.40247898E+00;
    COFD[254] = -3.57238362E-01;
    COFD[255] = 1.55145651E-02;
    COFD[256] = -1.98296243E+01;
    COFD[257] = 4.98207523E+00;
    COFD[258] = -4.26877291E-01;
    COFD[259] = 1.83086094E-02;
    COFD[260] = -1.86652603E+01;
    COFD[261] = 4.61260432E+00;
    COFD[262] = -3.82854484E-01;
    COFD[263] = 1.65575163E-02;
    COFD[264] = -1.86234701E+01;
    COFD[265] = 4.60336076E+00;
    COFD[266] = -3.81691643E-01;
    COFD[267] = 1.65085234E-02;
    COFD[268] = -1.86254955E+01;
    COFD[269] = 4.60336076E+00;
    COFD[270] = -3.81691643E-01;
    COFD[271] = 1.65085234E-02;
    COFD[272] = -1.87433618E+01;
    COFD[273] = 4.58956960E+00;
    COFD[274] = -3.79964215E-01;
    COFD[275] = 1.64361138E-02;
    COFD[276] = -2.00070284E+01;
    COFD[277] = 5.02095434E+00;
    COFD[278] = -4.31496874E-01;
    COFD[279] = 1.84920392E-02;
    COFD[280] = -1.92404583E+01;
    COFD[281] = 4.73921581E+00;
    COFD[282] = -3.98017274E-01;
    COFD[283] = 1.71639614E-02;
    COFD[284] = -1.91633071E+01;
    COFD[285] = 4.70966098E+00;
    COFD[286] = -3.94551217E-01;
    COFD[287] = 1.70286289E-02;
    COFD[288] = -1.94778445E+01;
    COFD[289] = 4.85518471E+00;
    COFD[290] = -4.11551624E-01;
    COFD[291] = 1.76895651E-02;
    COFD[292] = -1.99562868E+01;
    COFD[293] = 4.99367362E+00;
    COFD[294] = -4.28249956E-01;
    COFD[295] = 1.83628509E-02;
    COFD[296] = -1.99785176E+01;
    COFD[297] = 4.92184026E+00;
    COFD[298] = -4.19745472E-01;
    COFD[299] = 1.80268154E-02;
    COFD[300] = -1.99792167E+01;
    COFD[301] = 4.92184026E+00;
    COFD[302] = -4.19745472E-01;
    COFD[303] = 1.80268154E-02;
    COFD[304] = -1.16906297E+01;
    COFD[305] = 2.47469981E+00;
    COFD[306] = -1.10436257E-01;
    COFD[307] = 4.95273813E-03;
    COFD[308] = -1.09595712E+01;
    COFD[309] = 2.30836460E+00;
    COFD[310] = -8.76339315E-02;
    COFD[311] = 3.90878445E-03;
    COFD[312] = -1.03270606E+01;
    COFD[313] = 2.19285409E+00;
    COFD[314] = -7.54492786E-02;
    COFD[315] = 3.51398213E-03;
    COFD[316] = -1.14366381E+01;
    COFD[317] = 2.78323501E+00;
    COFD[318] = -1.51214064E-01;
    COFD[319] = 6.75150012E-03;
    COFD[320] = -1.09628982E+01;
    COFD[321] = 2.30836460E+00;
    COFD[322] = -8.76339315E-02;
    COFD[323] = 3.90878445E-03;
    COFD[324] = -1.71982995E+01;
    COFD[325] = 4.63881404E+00;
    COFD[326] = -3.86139633E-01;
    COFD[327] = 1.66955081E-02;
    COFD[328] = -1.18988955E+01;
    COFD[329] = 2.57507000E+00;
    COFD[330] = -1.24033737E-01;
    COFD[331] = 5.56694959E-03;
    COFD[332] = -1.18998012E+01;
    COFD[333] = 2.57507000E+00;
    COFD[334] = -1.24033737E-01;
    COFD[335] = 5.56694959E-03;
    COFD[336] = -1.19006548E+01;
    COFD[337] = 2.57507000E+00;
    COFD[338] = -1.24033737E-01;
    COFD[339] = 5.56694959E-03;
    COFD[340] = -1.37794315E+01;
    COFD[341] = 3.23973858E+00;
    COFD[342] = -2.09989036E-01;
    COFD[343] = 9.27667906E-03;
    COFD[344] = -1.17159737E+01;
    COFD[345] = 2.48123210E+00;
    COFD[346] = -1.11322604E-01;
    COFD[347] = 4.99282389E-03;
    COFD[348] = -1.60528285E+01;
    COFD[349] = 4.11188603E+00;
    COFD[350] = -3.21540884E-01;
    COFD[351] = 1.40482564E-02;
    COFD[352] = -1.25141260E+01;
    COFD[353] = 2.77873601E+00;
    COFD[354] = -1.50637360E-01;
    COFD[355] = 6.72684281E-03;
    COFD[356] = -1.24693568E+01;
    COFD[357] = 2.76686648E+00;
    COFD[358] = -1.49120141E-01;
    COFD[359] = 6.66220432E-03;
    COFD[360] = -1.59537247E+01;
    COFD[361] = 4.07051484E+00;
    COFD[362] = -3.16303109E-01;
    COFD[363] = 1.38259377E-02;
    COFD[364] = -1.39658996E+01;
    COFD[365] = 3.24966086E+00;
    COFD[366] = -2.11199992E-01;
    COFD[367] = 9.32580661E-03;
    COFD[368] = -1.57034851E+01;
    COFD[369] = 3.93614244E+00;
    COFD[370] = -2.99111497E-01;
    COFD[371] = 1.30888229E-02;
    COFD[372] = -1.57054717E+01;
    COFD[373] = 3.93614244E+00;
    COFD[374] = -2.99111497E-01;
    COFD[375] = 1.30888229E-02;
    COFD[376] = -1.59632479E+01;
    COFD[377] = 4.07051484E+00;
    COFD[378] = -3.16303109E-01;
    COFD[379] = 1.38259377E-02;
    COFD[380] = -1.39315266E+01;
    COFD[381] = 3.30394764E+00;
    COFD[382] = -2.17920112E-01;
    COFD[383] = 9.60284243E-03;
    COFD[384] = -1.45715797E+01;
    COFD[385] = 3.49477850E+00;
    COFD[386] = -2.42635772E-01;
    COFD[387] = 1.06721490E-02;
    COFD[388] = -1.36336373E+01;
    COFD[389] = 3.22088176E+00;
    COFD[390] = -2.07623790E-01;
    COFD[391] = 9.17771542E-03;
    COFD[392] = -1.47719516E+01;
    COFD[393] = 3.55444478E+00;
    COFD[394] = -2.50272707E-01;
    COFD[395] = 1.09990787E-02;
    COFD[396] = -1.47725694E+01;
    COFD[397] = 3.55444478E+00;
    COFD[398] = -2.50272707E-01;
    COFD[399] = 1.09990787E-02;
    COFD[400] = -1.46719197E+01;
    COFD[401] = 3.52400594E+00;
    COFD[402] = -2.46379985E-01;
    COFD[403] = 1.08326032E-02;
    COFD[404] = -1.47137939E+01;
    COFD[405] = 3.48023191E+00;
    COFD[406] = -2.40800798E-01;
    COFD[407] = 1.05947990E-02;
    COFD[408] = -1.64819183E+01;
    COFD[409] = 4.11726215E+00;
    COFD[410] = -3.22193015E-01;
    COFD[411] = 1.40747074E-02;
    COFD[412] = -1.51448279E+01;
    COFD[413] = 3.64565939E+00;
    COFD[414] = -2.61726871E-01;
    COFD[415] = 1.14799244E-02;
    COFD[416] = -1.51159870E+01;
    COFD[417] = 3.64206330E+00;
    COFD[418] = -2.61313444E-01;
    COFD[419] = 1.14642754E-02;
    COFD[420] = -1.51163041E+01;
    COFD[421] = 3.64206330E+00;
    COFD[422] = -2.61313444E-01;
    COFD[423] = 1.14642754E-02;
    COFD[424] = -1.52503668E+01;
    COFD[425] = 3.63657318E+00;
    COFD[426] = -2.60678457E-01;
    COFD[427] = 1.14400550E-02;
    COFD[428] = -1.65048875E+01;
    COFD[429] = 4.10792536E+00;
    COFD[430] = -3.21060656E-01;
    COFD[431] = 1.40287900E-02;
    COFD[432] = -1.56919143E+01;
    COFD[433] = 3.77842689E+00;
    COFD[434] = -2.78523399E-01;
    COFD[435] = 1.21896111E-02;
    COFD[436] = -1.55781966E+01;
    COFD[437] = 3.73153794E+00;
    COFD[438] = -2.72372598E-01;
    COFD[439] = 1.19199668E-02;
    COFD[440] = -1.60461372E+01;
    COFD[441] = 3.95298868E+00;
    COFD[442] = -3.01302078E-01;
    COFD[443] = 1.31842095E-02;
    COFD[444] = -1.64639359E+01;
    COFD[445] = 4.08142484E+00;
    COFD[446] = -3.17696496E-01;
    COFD[447] = 1.38856294E-02;
    COFD[448] = -1.64898528E+01;
    COFD[449] = 4.01175649E+00;
    COFD[450] = -3.08860971E-01;
    COFD[451] = 1.35100076E-02;
    COFD[452] = -1.64899530E+01;
    COFD[453] = 4.01175649E+00;
    COFD[454] = -3.08860971E-01;
    COFD[455] = 1.35100076E-02;
    COFD[456] = -1.42894441E+01;
    COFD[457] = 3.67490723E+00;
    COFD[458] = -2.65114792E-01;
    COFD[459] = 1.16092671E-02;
    COFD[460] = -1.34230272E+01;
    COFD[461] = 3.48624238E+00;
    COFD[462] = -2.41554467E-01;
    COFD[463] = 1.06263545E-02;
    COFD[464] = -1.14366381E+01;
    COFD[465] = 2.78323501E+00;
    COFD[466] = -1.51214064E-01;
    COFD[467] = 6.75150012E-03;
    COFD[468] = -1.47968712E+01;
    COFD[469] = 4.23027636E+00;
    COFD[470] = -3.36139991E-01;
    COFD[471] = 1.46507621E-02;
    COFD[472] = -1.34247866E+01;
    COFD[473] = 3.48624238E+00;
    COFD[474] = -2.41554467E-01;
    COFD[475] = 1.06263545E-02;
    COFD[476] = -1.95739570E+01;
    COFD[477] = 5.61113230E+00;
    COFD[478] = -4.90190187E-01;
    COFD[479] = 2.03260675E-02;
    COFD[480] = -1.46550083E+01;
    COFD[481] = 3.83606243E+00;
    COFD[482] = -2.86076532E-01;
    COFD[483] = 1.25205829E-02;
    COFD[484] = -1.46554748E+01;
    COFD[485] = 3.83606243E+00;
    COFD[486] = -2.86076532E-01;
    COFD[487] = 1.25205829E-02;
    COFD[488] = -1.46559141E+01;
    COFD[489] = 3.83606243E+00;
    COFD[490] = -2.86076532E-01;
    COFD[491] = 1.25205829E-02;
    COFD[492] = -1.76147026E+01;
    COFD[493] = 4.86049500E+00;
    COFD[494] = -4.12200578E-01;
    COFD[495] = 1.77160971E-02;
    COFD[496] = -1.43151174E+01;
    COFD[497] = 3.68038508E+00;
    COFD[498] = -2.65779346E-01;
    COFD[499] = 1.16360771E-02;
    COFD[500] = -1.97550088E+01;
    COFD[501] = 5.56931926E+00;
    COFD[502] = -4.89105511E-01;
    COFD[503] = 2.04493129E-02;
    COFD[504] = -1.57994893E+01;
    COFD[505] = 4.22225052E+00;
    COFD[506] = -3.35156428E-01;
    COFD[507] = 1.46104855E-02;
    COFD[508] = -1.57199037E+01;
    COFD[509] = 4.19936335E+00;
    COFD[510] = -3.32311009E-01;
    COFD[511] = 1.44921003E-02;
    COFD[512] = -1.96866103E+01;
    COFD[513] = 5.54637286E+00;
    COFD[514] = -4.87070324E-01;
    COFD[515] = 2.03983467E-02;
    COFD[516] = -1.78637178E+01;
    COFD[517] = 4.88268692E+00;
    COFD[518] = -4.14917638E-01;
    COFD[519] = 1.78274298E-02;
    COFD[520] = -1.94688688E+01;
    COFD[521] = 5.43830787E+00;
    COFD[522] = -4.75472880E-01;
    COFD[523] = 1.99909996E-02;
    COFD[524] = -1.94698843E+01;
    COFD[525] = 5.43830787E+00;
    COFD[526] = -4.75472880E-01;
    COFD[527] = 1.99909996E-02;
    COFD[528] = -1.96914944E+01;
    COFD[529] = 5.54637286E+00;
    COFD[530] = -4.87070324E-01;
    COFD[531] = 2.03983467E-02;
    COFD[532] = -1.79310765E+01;
    COFD[533] = 4.98037650E+00;
    COFD[534] = -4.26676911E-01;
    COFD[535] = 1.83007231E-02;
    COFD[536] = -1.85748546E+01;
    COFD[537] = 5.14789919E+00;
    COFD[538] = -4.45930850E-01;
    COFD[539] = 1.90363341E-02;
    COFD[540] = -1.74407963E+01;
    COFD[541] = 4.83580036E+00;
    COFD[542] = -4.09383573E-01;
    COFD[543] = 1.76098175E-02;
    COFD[544] = -1.87644697E+01;
    COFD[545] = 5.19146813E+00;
    COFD[546] = -4.50340408E-01;
    COFD[547] = 1.91768178E-02;
    COFD[548] = -1.87647862E+01;
    COFD[549] = 5.19146813E+00;
    COFD[550] = -4.50340408E-01;
    COFD[551] = 1.91768178E-02;
    COFD[552] = -1.86493112E+01;
    COFD[553] = 5.16040659E+00;
    COFD[554] = -4.46843492E-01;
    COFD[555] = 1.90466181E-02;
    COFD[556] = -1.87481780E+01;
    COFD[557] = 5.13858656E+00;
    COFD[558] = -4.45075387E-01;
    COFD[559] = 1.90137309E-02;
    COFD[560] = -2.01262921E+01;
    COFD[561] = 5.54581286E+00;
    COFD[562] = -4.87014004E-01;
    COFD[563] = 2.03965482E-02;
    COFD[564] = -1.92784178E+01;
    COFD[565] = 5.32291505E+00;
    COFD[566] = -4.65883522E-01;
    COFD[567] = 1.97916109E-02;
    COFD[568] = -1.92360228E+01;
    COFD[569] = 5.31542554E+00;
    COFD[570] = -4.65003780E-01;
    COFD[571] = 1.97570185E-02;
    COFD[572] = -1.92361841E+01;
    COFD[573] = 5.31542554E+00;
    COFD[574] = -4.65003780E-01;
    COFD[575] = 1.97570185E-02;
    COFD[576] = -1.93693740E+01;
    COFD[577] = 5.30286598E+00;
    COFD[578] = -4.63495567E-01;
    COFD[579] = 1.96962203E-02;
    COFD[580] = -2.02592914E+01;
    COFD[581] = 5.56701235E+00;
    COFD[582] = -4.88925090E-01;
    COFD[583] = 2.04461948E-02;
    COFD[584] = -1.97252269E+01;
    COFD[585] = 5.38884098E+00;
    COFD[586] = -4.71627912E-01;
    COFD[587] = 1.99273178E-02;
    COFD[588] = -1.96804200E+01;
    COFD[589] = 5.37526595E+00;
    COFD[590] = -4.70621144E-01;
    COFD[591] = 1.99141073E-02;
    COFD[592] = -1.98424714E+01;
    COFD[593] = 5.45215174E+00;
    COFD[594] = -4.77051991E-01;
    COFD[595] = 2.00510347E-02;
    COFD[596] = -2.02451923E+01;
    COFD[597] = 5.55377454E+00;
    COFD[598] = -4.87810074E-01;
    COFD[599] = 2.04217376E-02;
    COFD[600] = -2.03113704E+01;
    COFD[601] = 5.50136606E+00;
    COFD[602] = -4.82461887E-01;
    COFD[603] = 2.02471523E-02;
    COFD[604] = -2.03114210E+01;
    COFD[605] = 5.50136606E+00;
    COFD[606] = -4.82461887E-01;
    COFD[607] = 2.02471523E-02;
    COFD[608] = -1.40949196E+01;
    COFD[609] = 3.07549274E+00;
    COFD[610] = -1.88889344E-01;
    COFD[611] = 8.37152866E-03;
    COFD[612] = -1.32244035E+01;
    COFD[613] = 2.90778936E+00;
    COFD[614] = -1.67388544E-01;
    COFD[615] = 7.45220609E-03;
    COFD[616] = -1.09628982E+01;
    COFD[617] = 2.30836460E+00;
    COFD[618] = -8.76339315E-02;
    COFD[619] = 3.90878445E-03;
    COFD[620] = -1.34247866E+01;
    COFD[621] = 3.48624238E+00;
    COFD[622] = -2.41554467E-01;
    COFD[623] = 1.06263545E-02;
    COFD[624] = -1.32399106E+01;
    COFD[625] = 2.90778936E+00;
    COFD[626] = -1.67388544E-01;
    COFD[627] = 7.45220609E-03;
    COFD[628] = -1.94253036E+01;
    COFD[629] = 5.16013126E+00;
    COFD[630] = -4.46824543E-01;
    COFD[631] = 1.90464887E-02;
    COFD[632] = -1.43340796E+01;
    COFD[633] = 3.17651319E+00;
    COFD[634] = -2.02028974E-01;
    COFD[635] = 8.94232502E-03;
    COFD[636] = -1.43394069E+01;
    COFD[637] = 3.17651319E+00;
    COFD[638] = -2.02028974E-01;
    COFD[639] = 8.94232502E-03;
    COFD[640] = -1.43444709E+01;
    COFD[641] = 3.17651319E+00;
    COFD[642] = -2.02028974E-01;
    COFD[643] = 8.94232502E-03;
    COFD[644] = -1.70757047E+01;
    COFD[645] = 4.14240922E+00;
    COFD[646] = -3.25239774E-01;
    COFD[647] = 1.41980687E-02;
    COFD[648] = -1.41191261E+01;
    COFD[649] = 3.08120012E+00;
    COFD[650] = -1.89629903E-01;
    COFD[651] = 8.40361952E-03;
    COFD[652] = -1.94570287E+01;
    COFD[653] = 5.02567894E+00;
    COFD[654] = -4.32045169E-01;
    COFD[655] = 1.85132214E-02;
    COFD[656] = -1.50911794E+01;
    COFD[657] = 3.47945612E+00;
    COFD[658] = -2.40703722E-01;
    COFD[659] = 1.05907441E-02;
    COFD[660] = -1.50420953E+01;
    COFD[661] = 3.46140064E+00;
    COFD[662] = -2.38440092E-01;
    COFD[663] = 1.04960087E-02;
    COFD[664] = -1.93566243E+01;
    COFD[665] = 4.98286777E+00;
    COFD[666] = -4.26970814E-01;
    COFD[667] = 1.83122917E-02;
    COFD[668] = -1.72310232E+01;
    COFD[669] = 4.15807461E+00;
    COFD[670] = -3.27178539E-01;
    COFD[671] = 1.42784349E-02;
    COFD[672] = -1.91102652E+01;
    COFD[673] = 4.84384483E+00;
    COFD[674] = -4.10265575E-01;
    COFD[675] = 1.76414287E-02;
    COFD[676] = -1.91229033E+01;
    COFD[677] = 4.84384483E+00;
    COFD[678] = -4.10265575E-01;
    COFD[679] = 1.76414287E-02;
    COFD[680] = -1.94151822E+01;
    COFD[681] = 4.98286777E+00;
    COFD[682] = -4.26970814E-01;
    COFD[683] = 1.83122917E-02;
    COFD[684] = -1.72473011E+01;
    COFD[685] = 4.24084025E+00;
    COFD[686] = -3.37428619E-01;
    COFD[687] = 1.47032793E-02;
    COFD[688] = -1.79580609E+01;
    COFD[689] = 4.42139452E+00;
    COFD[690] = -3.59567329E-01;
    COFD[691] = 1.56103969E-02;
    COFD[692] = -1.68535757E+01;
    COFD[693] = 4.11954900E+00;
    COFD[694] = -3.22470391E-01;
    COFD[695] = 1.40859564E-02;
    COFD[696] = -1.81677871E+01;
    COFD[697] = 4.48398491E+00;
    COFD[698] = -3.67097129E-01;
    COFD[699] = 1.59123634E-02;
    COFD[700] = -1.81716176E+01;
    COFD[701] = 4.48398491E+00;
    COFD[702] = -3.67097129E-01;
    COFD[703] = 1.59123634E-02;
    COFD[704] = -1.80698901E+01;
    COFD[705] = 4.45434023E+00;
    COFD[706] = -3.63584633E-01;
    COFD[707] = 1.57739270E-02;
    COFD[708] = -1.80945693E+01;
    COFD[709] = 4.40247898E+00;
    COFD[710] = -3.57238362E-01;
    COFD[711] = 1.55145651E-02;
    COFD[712] = -1.98546695E+01;
    COFD[713] = 4.98207523E+00;
    COFD[714] = -4.26877291E-01;
    COFD[715] = 1.83086094E-02;
    COFD[716] = -1.86886689E+01;
    COFD[717] = 4.61260432E+00;
    COFD[718] = -3.82854484E-01;
    COFD[719] = 1.65575163E-02;
    COFD[720] = -1.86469792E+01;
    COFD[721] = 4.60336076E+00;
    COFD[722] = -3.81691643E-01;
    COFD[723] = 1.65085234E-02;
    COFD[724] = -1.86491023E+01;
    COFD[725] = 4.60336076E+00;
    COFD[726] = -3.81691643E-01;
    COFD[727] = 1.65085234E-02;
    COFD[728] = -1.87670637E+01;
    COFD[729] = 4.58956960E+00;
    COFD[730] = -3.79964215E-01;
    COFD[731] = 1.64361138E-02;
    COFD[732] = -2.00328044E+01;
    COFD[733] = 5.02095434E+00;
    COFD[734] = -4.31496874E-01;
    COFD[735] = 1.84920392E-02;
    COFD[736] = -1.92651204E+01;
    COFD[737] = 4.73921581E+00;
    COFD[738] = -3.98017274E-01;
    COFD[739] = 1.71639614E-02;
    COFD[740] = -1.91880377E+01;
    COFD[741] = 4.70966098E+00;
    COFD[742] = -3.94551217E-01;
    COFD[743] = 1.70286289E-02;
    COFD[744] = -1.95026421E+01;
    COFD[745] = 4.85518471E+00;
    COFD[746] = -4.11551624E-01;
    COFD[747] = 1.76895651E-02;
    COFD[748] = -1.99818280E+01;
    COFD[749] = 4.99367362E+00;
    COFD[750] = -4.28249956E-01;
    COFD[751] = 1.83628509E-02;
    COFD[752] = -2.00047095E+01;
    COFD[753] = 4.92184026E+00;
    COFD[754] = -4.19745472E-01;
    COFD[755] = 1.80268154E-02;
    COFD[756] = -2.00054461E+01;
    COFD[757] = 4.92184026E+00;
    COFD[758] = -4.19745472E-01;
    COFD[759] = 1.80268154E-02;
    COFD[760] = -2.10643259E+01;
    COFD[761] = 5.53614847E+00;
    COFD[762] = -4.86046736E-01;
    COFD[763] = 2.03659188E-02;
    COFD[764] = -1.94093572E+01;
    COFD[765] = 5.16013126E+00;
    COFD[766] = -4.46824543E-01;
    COFD[767] = 1.90464887E-02;
    COFD[768] = -1.71982995E+01;
    COFD[769] = 4.63881404E+00;
    COFD[770] = -3.86139633E-01;
    COFD[771] = 1.66955081E-02;
    COFD[772] = -1.95739570E+01;
    COFD[773] = 5.61113230E+00;
    COFD[774] = -4.90190187E-01;
    COFD[775] = 2.03260675E-02;
    COFD[776] = -1.94253036E+01;
    COFD[777] = 5.16013126E+00;
    COFD[778] = -4.46824543E-01;
    COFD[779] = 1.90464887E-02;
    COFD[780] = -1.19157919E+01;
    COFD[781] = 9.28955130E-01;
    COFD[782] = 2.42107090E-01;
    COFD[783] = -1.59823963E-02;
    COFD[784] = -2.12652533E+01;
    COFD[785] = 5.59961818E+00;
    COFD[786] = -4.91624858E-01;
    COFD[787] = 2.05035550E-02;
    COFD[788] = -2.06463744E+01;
    COFD[789] = 5.41688482E+00;
    COFD[790] = -4.73387188E-01;
    COFD[791] = 1.99280175E-02;
    COFD[792] = -2.06516336E+01;
    COFD[793] = 5.41688482E+00;
    COFD[794] = -4.73387188E-01;
    COFD[795] = 1.99280175E-02;
    COFD[796] = -2.07653719E+01;
    COFD[797] = 5.01092022E+00;
    COFD[798] = -3.77985635E-01;
    COFD[799] = 1.40968645E-02;
    COFD[800] = -2.11388331E+01;
    COFD[801] = 5.55529675E+00;
    COFD[802] = -4.87942518E-01;
    COFD[803] = 2.04249054E-02;
    COFD[804] = -1.77563250E+01;
    COFD[805] = 3.57475686E+00;
    COFD[806] = -1.56396297E-01;
    COFD[807] = 3.12157721E-03;
    COFD[808] = -2.12831323E+01;
    COFD[809] = 5.61184117E+00;
    COFD[810] = -4.90532156E-01;
    COFD[811] = 2.03507922E-02;
    COFD[812] = -2.14087397E+01;
    COFD[813] = 5.57282008E+00;
    COFD[814] = -4.76690890E-01;
    COFD[815] = 1.94000719E-02;
    COFD[816] = -1.80253664E+01;
    COFD[817] = 3.69199168E+00;
    COFD[818] = -1.74005516E-01;
    COFD[819] = 3.97694372E-03;
    COFD[820] = -2.13148887E+01;
    COFD[821] = 5.27210469E+00;
    COFD[822] = -4.21419216E-01;
    COFD[823] = 1.63567178E-02;
    COFD[824] = -1.87383952E+01;
    COFD[825] = 3.96926341E+00;
    COFD[826] = -2.16412264E-01;
    COFD[827] = 6.06012078E-03;
    COFD[828] = -1.87515645E+01;
    COFD[829] = 3.96926341E+00;
    COFD[830] = -2.16412264E-01;
    COFD[831] = 6.06012078E-03;
    COFD[832] = -1.80862867E+01;
    COFD[833] = 3.69199168E+00;
    COFD[834] = -1.74005516E-01;
    COFD[835] = 3.97694372E-03;
    COFD[836] = -2.09565916E+01;
    COFD[837] = 5.18380539E+00;
    COFD[838] = -4.06234719E-01;
    COFD[839] = 1.55515345E-02;
    COFD[840] = -2.06310304E+01;
    COFD[841] = 4.89289496E+00;
    COFD[842] = -3.59346263E-01;
    COFD[843] = 1.31570901E-02;
    COFD[844] = -2.11309197E+01;
    COFD[845] = 5.32644193E+00;
    COFD[846] = -4.30581064E-01;
    COFD[847] = 1.68379725E-02;
    COFD[848] = -2.04357586E+01;
    COFD[849] = 4.77398686E+00;
    COFD[850] = -3.40522956E-01;
    COFD[851] = 1.22072846E-02;
    COFD[852] = -2.04397451E+01;
    COFD[853] = 4.77398686E+00;
    COFD[854] = -3.40522956E-01;
    COFD[855] = 1.22072846E-02;
    COFD[856] = -2.05372411E+01;
    COFD[857] = 4.83379373E+00;
    COFD[858] = -3.50008083E-01;
    COFD[859] = 1.26863426E-02;
    COFD[860] = -2.08879167E+01;
    COFD[861] = 4.92602269E+00;
    COFD[862] = -3.64572914E-01;
    COFD[863] = 1.34203681E-02;
    COFD[864] = -1.73636900E+01;
    COFD[865] = 3.17377130E+00;
    COFD[866] = -1.00394383E-01;
    COFD[867] = 5.69083899E-04;
    COFD[868] = -2.02184916E+01;
    COFD[869] = 4.57152878E+00;
    COFD[870] = -3.08371263E-01;
    COFD[871] = 1.05838559E-02;
    COFD[872] = -2.02265558E+01;
    COFD[873] = 4.58441724E+00;
    COFD[874] = -3.10392854E-01;
    COFD[875] = 1.06849990E-02;
    COFD[876] = -2.02287739E+01;
    COFD[877] = 4.58441724E+00;
    COFD[878] = -3.10392854E-01;
    COFD[879] = 1.06849990E-02;
    COFD[880] = -2.04186424E+01;
    COFD[881] = 4.60117690E+00;
    COFD[882] = -3.13067257E-01;
    COFD[883] = 1.08202310E-02;
    COFD[884] = -1.83939699E+01;
    COFD[885] = 3.59019527E+00;
    COFD[886] = -1.58702132E-01;
    COFD[887] = 3.23316765E-03;
    COFD[888] = -1.98682752E+01;
    COFD[889] = 4.28648872E+00;
    COFD[890] = -2.64358750E-01;
    COFD[891] = 8.40263071E-03;
    COFD[892] = -1.98762802E+01;
    COFD[893] = 4.29984430E+00;
    COFD[894] = -2.67672378E-01;
    COFD[895] = 8.61066133E-03;
    COFD[896] = -1.90375666E+01;
    COFD[897] = 3.93604965E+00;
    COFD[898] = -2.11360409E-01;
    COFD[899] = 5.81247394E-03;
    COFD[900] = -1.85767826E+01;
    COFD[901] = 3.66420353E+00;
    COFD[902] = -1.69810177E-01;
    COFD[903] = 3.77247849E-03;
    COFD[904] = -1.91326792E+01;
    COFD[905] = 3.82263611E+00;
    COFD[906] = -1.93983472E-01;
    COFD[907] = 4.95789388E-03;
    COFD[908] = -1.91334529E+01;
    COFD[909] = 3.82263611E+00;
    COFD[910] = -1.93983472E-01;
    COFD[911] = 4.95789388E-03;
    COFD[912] = -1.52414485E+01;
    COFD[913] = 3.35922578E+00;
    COFD[914] = -2.25181399E-01;
    COFD[915] = 9.92132878E-03;
    COFD[916] = -1.43139231E+01;
    COFD[917] = 3.17651319E+00;
    COFD[918] = -2.02028974E-01;
    COFD[919] = 8.94232502E-03;
    COFD[920] = -1.18988955E+01;
    COFD[921] = 2.57507000E+00;
    COFD[922] = -1.24033737E-01;
    COFD[923] = 5.56694959E-03;
    COFD[924] = -1.46550083E+01;
    COFD[925] = 3.83606243E+00;
    COFD[926] = -2.86076532E-01;
    COFD[927] = 1.25205829E-02;
    COFD[928] = -1.43340796E+01;
    COFD[929] = 3.17651319E+00;
    COFD[930] = -2.02028974E-01;
    COFD[931] = 8.94232502E-03;
    COFD[932] = -2.12652533E+01;
    COFD[933] = 5.59961818E+00;
    COFD[934] = -4.91624858E-01;
    COFD[935] = 2.05035550E-02;
    COFD[936] = -1.55511344E+01;
    COFD[937] = 3.48070094E+00;
    COFD[938] = -2.40859499E-01;
    COFD[939] = 1.05972514E-02;
    COFD[940] = -1.55588279E+01;
    COFD[941] = 3.48070094E+00;
    COFD[942] = -2.40859499E-01;
    COFD[943] = 1.05972514E-02;
    COFD[944] = -1.55661750E+01;
    COFD[945] = 3.48070094E+00;
    COFD[946] = -2.40859499E-01;
    COFD[947] = 1.05972514E-02;
    COFD[948] = -1.84688406E+01;
    COFD[949] = 4.49330851E+00;
    COFD[950] = -3.68208715E-01;
    COFD[951] = 1.59565402E-02;
    COFD[952] = -1.52721107E+01;
    COFD[953] = 3.36790500E+00;
    COFD[954] = -2.26321740E-01;
    COFD[955] = 9.97135055E-03;
    COFD[956] = -2.08293255E+01;
    COFD[957] = 5.35267674E+00;
    COFD[958] = -4.69010505E-01;
    COFD[959] = 1.98979152E-02;
    COFD[960] = -1.63493345E+01;
    COFD[961] = 3.82388595E+00;
    COFD[962] = -2.84480724E-01;
    COFD[963] = 1.24506311E-02;
    COFD[964] = -1.62724462E+01;
    COFD[965] = 3.79163564E+00;
    COFD[966] = -2.80257365E-01;
    COFD[967] = 1.22656902E-02;
    COFD[968] = -2.07595845E+01;
    COFD[969] = 5.32244593E+00;
    COFD[970] = -4.65829403E-01;
    COFD[971] = 1.97895274E-02;
    COFD[972] = -1.85844688E+01;
    COFD[973] = 4.51052425E+00;
    COFD[974] = -3.70301627E-01;
    COFD[975] = 1.60416153E-02;
    COFD[976] = -2.05184870E+01;
    COFD[977] = 5.18417470E+00;
    COFD[978] = -4.49491573E-01;
    COFD[979] = 1.91438508E-02;
    COFD[980] = -2.05375724E+01;
    COFD[981] = 5.18417470E+00;
    COFD[982] = -4.49491573E-01;
    COFD[983] = 1.91438508E-02;
    COFD[984] = -2.08463209E+01;
    COFD[985] = 5.32244593E+00;
    COFD[986] = -4.65829403E-01;
    COFD[987] = 1.97895274E-02;
    COFD[988] = -1.86507213E+01;
    COFD[989] = 4.60874797E+00;
    COFD[990] = -3.82368716E-01;
    COFD[991] = 1.65370164E-02;
    COFD[992] = -1.93917298E+01;
    COFD[993] = 4.78708023E+00;
    COFD[994] = -4.03693144E-01;
    COFD[995] = 1.73884817E-02;
    COFD[996] = -1.82145353E+01;
    COFD[997] = 4.46848269E+00;
    COFD[998] = -3.65269718E-01;
    COFD[999] = 1.58407652E-02;
    COFD[1000] = -1.95819005E+01;
    COFD[1001] = 4.84393038E+00;
    COFD[1002] = -4.10274737E-01;
    COFD[1003] = 1.76417458E-02;
    COFD[1004] = -1.95875976E+01;
    COFD[1005] = 4.84393038E+00;
    COFD[1006] = -4.10274737E-01;
    COFD[1007] = 1.76417458E-02;
    COFD[1008] = -1.94912151E+01;
    COFD[1009] = 4.81575071E+00;
    COFD[1010] = -4.07042139E-01;
    COFD[1011] = 1.75187504E-02;
    COFD[1012] = -1.95201830E+01;
    COFD[1013] = 4.77151544E+00;
    COFD[1014] = -4.01882811E-01;
    COFD[1015] = 1.73184814E-02;
    COFD[1016] = -2.13698722E+01;
    COFD[1017] = 5.34971865E+00;
    COFD[1018] = -4.68771123E-01;
    COFD[1019] = 1.98933811E-02;
    COFD[1020] = -2.01315602E+01;
    COFD[1021] = 4.97613338E+00;
    COFD[1022] = -4.26175206E-01;
    COFD[1023] = 1.82809270E-02;
    COFD[1024] = -2.00964665E+01;
    COFD[1025] = 4.96870443E+00;
    COFD[1026] = -4.25292447E-01;
    COFD[1027] = 1.82459096E-02;
    COFD[1028] = -2.00997774E+01;
    COFD[1029] = 4.96870443E+00;
    COFD[1030] = -4.25292447E-01;
    COFD[1031] = 1.82459096E-02;
    COFD[1032] = -2.02121663E+01;
    COFD[1033] = 4.95786261E+00;
    COFD[1034] = -4.24013131E-01;
    COFD[1035] = 1.81955669E-02;
    COFD[1036] = -2.14416336E+01;
    COFD[1037] = 5.35040988E+00;
    COFD[1038] = -4.68827063E-01;
    COFD[1039] = 1.98944407E-02;
    COFD[1040] = -2.07257272E+01;
    COFD[1041] = 5.10688723E+00;
    COFD[1042] = -4.41563971E-01;
    COFD[1043] = 1.88857198E-02;
    COFD[1044] = -2.06463142E+01;
    COFD[1045] = 5.07657482E+00;
    COFD[1046] = -4.38028804E-01;
    COFD[1047] = 1.87481371E-02;
    COFD[1048] = -2.09258526E+01;
    COFD[1049] = 5.19811866E+00;
    COFD[1050] = -4.51121211E-01;
    COFD[1051] = 1.92074617E-02;
    COFD[1052] = -2.14057339E+01;
    COFD[1053] = 5.33269880E+00;
    COFD[1054] = -4.67008439E-01;
    COFD[1055] = 1.98347416E-02;
    COFD[1056] = -2.13955999E+01;
    COFD[1057] = 5.25183817E+00;
    COFD[1058] = -4.57376333E-01;
    COFD[1059] = 1.94504429E-02;
    COFD[1060] = -2.13968281E+01;
    COFD[1061] = 5.25183817E+00;
    COFD[1062] = -4.57376333E-01;
    COFD[1063] = 1.94504429E-02;
    COFD[1064] = -1.52486273E+01;
    COFD[1065] = 3.35922578E+00;
    COFD[1066] = -2.25181399E-01;
    COFD[1067] = 9.92132878E-03;
    COFD[1068] = -1.43190389E+01;
    COFD[1069] = 3.17651319E+00;
    COFD[1070] = -2.02028974E-01;
    COFD[1071] = 8.94232502E-03;
    COFD[1072] = -1.18998012E+01;
    COFD[1073] = 2.57507000E+00;
    COFD[1074] = -1.24033737E-01;
    COFD[1075] = 5.56694959E-03;
    COFD[1076] = -1.46554748E+01;
    COFD[1077] = 3.83606243E+00;
    COFD[1078] = -2.86076532E-01;
    COFD[1079] = 1.25205829E-02;
    COFD[1080] = -1.43394069E+01;
    COFD[1081] = 3.17651319E+00;
    COFD[1082] = -2.02028974E-01;
    COFD[1083] = 8.94232502E-03;
    COFD[1084] = -2.06463744E+01;
    COFD[1085] = 5.41688482E+00;
    COFD[1086] = -4.73387188E-01;
    COFD[1087] = 1.99280175E-02;
    COFD[1088] = -1.55588279E+01;
    COFD[1089] = 3.48070094E+00;
    COFD[1090] = -2.40859499E-01;
    COFD[1091] = 1.05972514E-02;
    COFD[1092] = -1.55666415E+01;
    COFD[1093] = 3.48070094E+00;
    COFD[1094] = -2.40859499E-01;
    COFD[1095] = 1.05972514E-02;
    COFD[1096] = -1.55741053E+01;
    COFD[1097] = 3.48070094E+00;
    COFD[1098] = -2.40859499E-01;
    COFD[1099] = 1.05972514E-02;
    COFD[1100] = -1.84777607E+01;
    COFD[1101] = 4.49330851E+00;
    COFD[1102] = -3.68208715E-01;
    COFD[1103] = 1.59565402E-02;
    COFD[1104] = -1.52792891E+01;
    COFD[1105] = 3.36790500E+00;
    COFD[1106] = -2.26321740E-01;
    COFD[1107] = 9.97135055E-03;
    COFD[1108] = -2.08367725E+01;
    COFD[1109] = 5.35267674E+00;
    COFD[1110] = -4.69010505E-01;
    COFD[1111] = 1.98979152E-02;
    COFD[1112] = -1.63542394E+01;
    COFD[1113] = 3.82388595E+00;
    COFD[1114] = -2.84480724E-01;
    COFD[1115] = 1.24506311E-02;
    COFD[1116] = -1.62775714E+01;
    COFD[1117] = 3.79163564E+00;
    COFD[1118] = -2.80257365E-01;
    COFD[1119] = 1.22656902E-02;
    COFD[1120] = -2.07672833E+01;
    COFD[1121] = 5.32244593E+00;
    COFD[1122] = -4.65829403E-01;
    COFD[1123] = 1.97895274E-02;
    COFD[1124] = -1.85919214E+01;
    COFD[1125] = 4.51052425E+00;
    COFD[1126] = -3.70301627E-01;
    COFD[1127] = 1.60416153E-02;
    COFD[1128] = -2.05272328E+01;
    COFD[1129] = 5.18417470E+00;
    COFD[1130] = -4.49491573E-01;
    COFD[1131] = 1.91438508E-02;
    COFD[1132] = -2.05466616E+01;
    COFD[1133] = 5.18417470E+00;
    COFD[1134] = -4.49491573E-01;
    COFD[1135] = 1.91438508E-02;
    COFD[1136] = -2.08554914E+01;
    COFD[1137] = 5.32244593E+00;
    COFD[1138] = -4.65829403E-01;
    COFD[1139] = 1.97895274E-02;
    COFD[1140] = -1.86576191E+01;
    COFD[1141] = 4.60874797E+00;
    COFD[1142] = -3.82368716E-01;
    COFD[1143] = 1.65370164E-02;
    COFD[1144] = -1.94004795E+01;
    COFD[1145] = 4.78708023E+00;
    COFD[1146] = -4.03693144E-01;
    COFD[1147] = 1.73884817E-02;
    COFD[1148] = -1.82217198E+01;
    COFD[1149] = 4.46848269E+00;
    COFD[1150] = -3.65269718E-01;
    COFD[1151] = 1.58407652E-02;
    COFD[1152] = -1.95903647E+01;
    COFD[1153] = 4.84393038E+00;
    COFD[1154] = -4.10274737E-01;
    COFD[1155] = 1.76417458E-02;
    COFD[1156] = -1.95961596E+01;
    COFD[1157] = 4.84393038E+00;
    COFD[1158] = -4.10274737E-01;
    COFD[1159] = 1.76417458E-02;
    COFD[1160] = -1.94998722E+01;
    COFD[1161] = 4.81575071E+00;
    COFD[1162] = -4.07042139E-01;
    COFD[1163] = 1.75187504E-02;
    COFD[1164] = -1.95290229E+01;
    COFD[1165] = 4.77151544E+00;
    COFD[1166] = -4.01882811E-01;
    COFD[1167] = 1.73184814E-02;
    COFD[1168] = -2.12907159E+01;
    COFD[1169] = 5.32167660E+00;
    COFD[1170] = -4.65740624E-01;
    COFD[1171] = 1.97861081E-02;
    COFD[1172] = -2.01412473E+01;
    COFD[1173] = 4.97613338E+00;
    COFD[1174] = -4.26175206E-01;
    COFD[1175] = 1.82809270E-02;
    COFD[1176] = -2.01062206E+01;
    COFD[1177] = 4.96870443E+00;
    COFD[1178] = -4.25292447E-01;
    COFD[1179] = 1.82459096E-02;
    COFD[1180] = -2.01095969E+01;
    COFD[1181] = 4.96870443E+00;
    COFD[1182] = -4.25292447E-01;
    COFD[1183] = 1.82459096E-02;
    COFD[1184] = -2.02220498E+01;
    COFD[1185] = 4.95786261E+00;
    COFD[1186] = -4.24013131E-01;
    COFD[1187] = 1.81955669E-02;
    COFD[1188] = -2.14529967E+01;
    COFD[1189] = 5.35040988E+00;
    COFD[1190] = -4.68827063E-01;
    COFD[1191] = 1.98944407E-02;
    COFD[1192] = -2.07362753E+01;
    COFD[1193] = 5.10688723E+00;
    COFD[1194] = -4.41563971E-01;
    COFD[1195] = 1.88857198E-02;
    COFD[1196] = -2.06522508E+01;
    COFD[1197] = 5.07501764E+00;
    COFD[1198] = -4.37846596E-01;
    COFD[1199] = 1.87410133E-02;
    COFD[1200] = -2.09364971E+01;
    COFD[1201] = 5.19811866E+00;
    COFD[1202] = -4.51121211E-01;
    COFD[1203] = 1.92074617E-02;
    COFD[1204] = -2.14169211E+01;
    COFD[1205] = 5.33269880E+00;
    COFD[1206] = -4.67008439E-01;
    COFD[1207] = 1.98347416E-02;
    COFD[1208] = -2.14072803E+01;
    COFD[1209] = 5.25183817E+00;
    COFD[1210] = -4.57376333E-01;
    COFD[1211] = 1.94504429E-02;
    COFD[1212] = -2.14085375E+01;
    COFD[1213] = 5.25183817E+00;
    COFD[1214] = -4.57376333E-01;
    COFD[1215] = 1.94504429E-02;
    COFD[1216] = -1.52554761E+01;
    COFD[1217] = 3.35922578E+00;
    COFD[1218] = -2.25181399E-01;
    COFD[1219] = 9.92132878E-03;
    COFD[1220] = -1.43238998E+01;
    COFD[1221] = 3.17651319E+00;
    COFD[1222] = -2.02028974E-01;
    COFD[1223] = 8.94232502E-03;
    COFD[1224] = -1.19006548E+01;
    COFD[1225] = 2.57507000E+00;
    COFD[1226] = -1.24033737E-01;
    COFD[1227] = 5.56694959E-03;
    COFD[1228] = -1.46559141E+01;
    COFD[1229] = 3.83606243E+00;
    COFD[1230] = -2.86076532E-01;
    COFD[1231] = 1.25205829E-02;
    COFD[1232] = -1.43444709E+01;
    COFD[1233] = 3.17651319E+00;
    COFD[1234] = -2.02028974E-01;
    COFD[1235] = 8.94232502E-03;
    COFD[1236] = -2.06516336E+01;
    COFD[1237] = 5.41688482E+00;
    COFD[1238] = -4.73387188E-01;
    COFD[1239] = 1.99280175E-02;
    COFD[1240] = -1.55661750E+01;
    COFD[1241] = 3.48070094E+00;
    COFD[1242] = -2.40859499E-01;
    COFD[1243] = 1.05972514E-02;
    COFD[1244] = -1.55741053E+01;
    COFD[1245] = 3.48070094E+00;
    COFD[1246] = -2.40859499E-01;
    COFD[1247] = 1.05972514E-02;
    COFD[1248] = -1.55816822E+01;
    COFD[1249] = 3.48070094E+00;
    COFD[1250] = -2.40859499E-01;
    COFD[1251] = 1.05972514E-02;
    COFD[1252] = -1.84863000E+01;
    COFD[1253] = 4.49330851E+00;
    COFD[1254] = -3.68208715E-01;
    COFD[1255] = 1.59565402E-02;
    COFD[1256] = -1.52861376E+01;
    COFD[1257] = 3.36790500E+00;
    COFD[1258] = -2.26321740E-01;
    COFD[1259] = 9.97135055E-03;
    COFD[1260] = -2.08438809E+01;
    COFD[1261] = 5.35267674E+00;
    COFD[1262] = -4.69010505E-01;
    COFD[1263] = 1.98979152E-02;
    COFD[1264] = -1.63588981E+01;
    COFD[1265] = 3.82388595E+00;
    COFD[1266] = -2.84480724E-01;
    COFD[1267] = 1.24506311E-02;
    COFD[1268] = -1.62824412E+01;
    COFD[1269] = 3.79163564E+00;
    COFD[1270] = -2.80257365E-01;
    COFD[1271] = 1.22656902E-02;
    COFD[1272] = -2.07746356E+01;
    COFD[1273] = 5.32244593E+00;
    COFD[1274] = -4.65829403E-01;
    COFD[1275] = 1.97895274E-02;
    COFD[1276] = -1.85990352E+01;
    COFD[1277] = 4.51052425E+00;
    COFD[1278] = -3.70301627E-01;
    COFD[1279] = 1.60416153E-02;
    COFD[1280] = -2.05356023E+01;
    COFD[1281] = 5.18417470E+00;
    COFD[1282] = -4.49491573E-01;
    COFD[1283] = 1.91438508E-02;
    COFD[1284] = -2.05553656E+01;
    COFD[1285] = 5.18417470E+00;
    COFD[1286] = -4.49491573E-01;
    COFD[1287] = 1.91438508E-02;
    COFD[1288] = -2.08642748E+01;
    COFD[1289] = 5.32244593E+00;
    COFD[1290] = -4.65829403E-01;
    COFD[1291] = 1.97895274E-02;
    COFD[1292] = -1.86641962E+01;
    COFD[1293] = 4.60874797E+00;
    COFD[1294] = -3.82368716E-01;
    COFD[1295] = 1.65370164E-02;
    COFD[1296] = -1.94088529E+01;
    COFD[1297] = 4.78708023E+00;
    COFD[1298] = -4.03693144E-01;
    COFD[1299] = 1.73884817E-02;
    COFD[1300] = -1.82285740E+01;
    COFD[1301] = 4.46848269E+00;
    COFD[1302] = -3.65269718E-01;
    COFD[1303] = 1.58407652E-02;
    COFD[1304] = -1.95984602E+01;
    COFD[1305] = 4.84393038E+00;
    COFD[1306] = -4.10274737E-01;
    COFD[1307] = 1.76417458E-02;
    COFD[1308] = -1.96043503E+01;
    COFD[1309] = 4.84393038E+00;
    COFD[1310] = -4.10274737E-01;
    COFD[1311] = 1.76417458E-02;
    COFD[1312] = -1.95081555E+01;
    COFD[1313] = 4.81575071E+00;
    COFD[1314] = -4.07042139E-01;
    COFD[1315] = 1.75187504E-02;
    COFD[1316] = -1.95374840E+01;
    COFD[1317] = 4.77151544E+00;
    COFD[1318] = -4.01882811E-01;
    COFD[1319] = 1.73184814E-02;
    COFD[1320] = -2.13011157E+01;
    COFD[1321] = 5.32167660E+00;
    COFD[1322] = -4.65740624E-01;
    COFD[1323] = 1.97861081E-02;
    COFD[1324] = -2.01505348E+01;
    COFD[1325] = 4.97613338E+00;
    COFD[1326] = -4.26175206E-01;
    COFD[1327] = 1.82809270E-02;
    COFD[1328] = -2.01155735E+01;
    COFD[1329] = 4.96870443E+00;
    COFD[1330] = -4.25292447E-01;
    COFD[1331] = 1.82459096E-02;
    COFD[1332] = -2.01190139E+01;
    COFD[1333] = 4.96870443E+00;
    COFD[1334] = -4.25292447E-01;
    COFD[1335] = 1.82459096E-02;
    COFD[1336] = -2.02315293E+01;
    COFD[1337] = 4.95786261E+00;
    COFD[1338] = -4.24013131E-01;
    COFD[1339] = 1.81955669E-02;
    COFD[1340] = -2.14639274E+01;
    COFD[1341] = 5.35040988E+00;
    COFD[1342] = -4.68827063E-01;
    COFD[1343] = 1.98944407E-02;
    COFD[1344] = -2.07464056E+01;
    COFD[1345] = 5.10688723E+00;
    COFD[1346] = -4.41563971E-01;
    COFD[1347] = 1.88857198E-02;
    COFD[1348] = -2.06624288E+01;
    COFD[1349] = 5.07501764E+00;
    COFD[1350] = -4.37846596E-01;
    COFD[1351] = 1.87410133E-02;
    COFD[1352] = -2.09467220E+01;
    COFD[1353] = 5.19811866E+00;
    COFD[1354] = -4.51121211E-01;
    COFD[1355] = 1.92074617E-02;
    COFD[1356] = -2.14276788E+01;
    COFD[1357] = 5.33269880E+00;
    COFD[1358] = -4.67008439E-01;
    COFD[1359] = 1.98347416E-02;
    COFD[1360] = -2.14185232E+01;
    COFD[1361] = 5.25183817E+00;
    COFD[1362] = -4.57376333E-01;
    COFD[1363] = 1.94504429E-02;
    COFD[1364] = -2.14198091E+01;
    COFD[1365] = 5.25183817E+00;
    COFD[1366] = -4.57376333E-01;
    COFD[1367] = 1.94504429E-02;
    COFD[1368] = -1.81432461E+01;
    COFD[1369] = 4.37565431E+00;
    COFD[1370] = -3.53906025E-01;
    COFD[1371] = 1.53760786E-02;
    COFD[1372] = -1.70534856E+01;
    COFD[1373] = 4.14240922E+00;
    COFD[1374] = -3.25239774E-01;
    COFD[1375] = 1.41980687E-02;
    COFD[1376] = -1.37794315E+01;
    COFD[1377] = 3.23973858E+00;
    COFD[1378] = -2.09989036E-01;
    COFD[1379] = 9.27667906E-03;
    COFD[1380] = -1.76147026E+01;
    COFD[1381] = 4.86049500E+00;
    COFD[1382] = -4.12200578E-01;
    COFD[1383] = 1.77160971E-02;
    COFD[1384] = -1.70757047E+01;
    COFD[1385] = 4.14240922E+00;
    COFD[1386] = -3.25239774E-01;
    COFD[1387] = 1.41980687E-02;
    COFD[1388] = -2.07653719E+01;
    COFD[1389] = 5.01092022E+00;
    COFD[1390] = -3.77985635E-01;
    COFD[1391] = 1.40968645E-02;
    COFD[1392] = -1.84688406E+01;
    COFD[1393] = 4.49330851E+00;
    COFD[1394] = -3.68208715E-01;
    COFD[1395] = 1.59565402E-02;
    COFD[1396] = -1.84777607E+01;
    COFD[1397] = 4.49330851E+00;
    COFD[1398] = -3.68208715E-01;
    COFD[1399] = 1.59565402E-02;
    COFD[1400] = -1.84863000E+01;
    COFD[1401] = 4.49330851E+00;
    COFD[1402] = -3.68208715E-01;
    COFD[1403] = 1.59565402E-02;
    COFD[1404] = -2.13425698E+01;
    COFD[1405] = 5.40460130E+00;
    COFD[1406] = -4.72718910E-01;
    COFD[1407] = 1.99362717E-02;
    COFD[1408] = -1.81735763E+01;
    COFD[1409] = 4.38391495E+00;
    COFD[1410] = -3.54941287E-01;
    COFD[1411] = 1.54195107E-02;
    COFD[1412] = -2.19317743E+01;
    COFD[1413] = 5.45216133E+00;
    COFD[1414] = -4.52916925E-01;
    COFD[1415] = 1.80456400E-02;
    COFD[1416] = -1.93276434E+01;
    COFD[1417] = 4.85015581E+00;
    COFD[1418] = -4.10945109E-01;
    COFD[1419] = 1.76651398E-02;
    COFD[1420] = -1.92867554E+01;
    COFD[1421] = 4.83375900E+00;
    COFD[1422] = -4.09146560E-01;
    COFD[1423] = 1.76006599E-02;
    COFD[1424] = -2.20063594E+01;
    COFD[1425] = 5.48540187E+00;
    COFD[1426] = -4.58962148E-01;
    COFD[1427] = 1.83770355E-02;
    COFD[1428] = -2.14151520E+01;
    COFD[1429] = 5.41122754E+00;
    COFD[1430] = -4.73185889E-01;
    COFD[1431] = 1.99407905E-02;
    COFD[1432] = -2.22116706E+01;
    COFD[1433] = 5.54251230E+00;
    COFD[1434] = -4.70946314E-01;
    COFD[1435] = 1.90785869E-02;
    COFD[1436] = -2.22343363E+01;
    COFD[1437] = 5.54251230E+00;
    COFD[1438] = -4.70946314E-01;
    COFD[1439] = 1.90785869E-02;
    COFD[1440] = -2.21083035E+01;
    COFD[1441] = 5.48540187E+00;
    COFD[1442] = -4.58962148E-01;
    COFD[1443] = 1.83770355E-02;
    COFD[1444] = -2.13961414E+01;
    COFD[1445] = 5.46685775E+00;
    COFD[1446] = -4.78665416E-01;
    COFD[1447] = 2.01093915E-02;
    COFD[1448] = -2.20725883E+01;
    COFD[1449] = 5.59642965E+00;
    COFD[1450] = -4.91577716E-01;
    COFD[1451] = 2.05159582E-02;
    COFD[1452] = -2.11031143E+01;
    COFD[1453] = 5.39439999E+00;
    COFD[1454] = -4.72050184E-01;
    COFD[1455] = 1.99336257E-02;
    COFD[1456] = -2.21630311E+01;
    COFD[1457] = 5.60807471E+00;
    COFD[1458] = -4.91339309E-01;
    COFD[1459] = 2.04365761E-02;
    COFD[1460] = -2.21697404E+01;
    COFD[1461] = 5.60807471E+00;
    COFD[1462] = -4.91339309E-01;
    COFD[1463] = 2.04365761E-02;
    COFD[1464] = -2.21216828E+01;
    COFD[1465] = 5.60203389E+00;
    COFD[1466] = -4.91444416E-01;
    COFD[1467] = 2.04761886E-02;
    COFD[1468] = -2.22052004E+01;
    COFD[1469] = 5.58604166E+00;
    COFD[1470] = -4.90602184E-01;
    COFD[1471] = 2.04880352E-02;
    COFD[1472] = -2.25168081E+01;
    COFD[1473] = 5.46125558E+00;
    COFD[1474] = -4.54580949E-01;
    COFD[1475] = 1.81370928E-02;
    COFD[1476] = -2.23890317E+01;
    COFD[1477] = 5.59178974E+00;
    COFD[1478] = -4.85668031E-01;
    COFD[1479] = 2.00491907E-02;
    COFD[1480] = -2.23772680E+01;
    COFD[1481] = 5.59425354E+00;
    COFD[1482] = -4.86232980E-01;
    COFD[1483] = 2.00835981E-02;
    COFD[1484] = -2.23812726E+01;
    COFD[1485] = 5.59425354E+00;
    COFD[1486] = -4.86232980E-01;
    COFD[1487] = 2.00835981E-02;
    COFD[1488] = -2.25216613E+01;
    COFD[1489] = 5.59792043E+00;
    COFD[1490] = -4.87076900E-01;
    COFD[1491] = 2.01350364E-02;
    COFD[1492] = -2.25838099E+01;
    COFD[1493] = 5.45615714E+00;
    COFD[1494] = -4.53643844E-01;
    COFD[1495] = 1.80854821E-02;
    COFD[1496] = -2.26897188E+01;
    COFD[1497] = 5.58518389E+00;
    COFD[1498] = -4.80570209E-01;
    COFD[1499] = 1.96586179E-02;
    COFD[1500] = -2.26749993E+01;
    COFD[1501] = 5.58486459E+00;
    COFD[1502] = -4.81517134E-01;
    COFD[1503] = 1.97388064E-02;
    COFD[1504] = -2.25786655E+01;
    COFD[1505] = 5.53409384E+00;
    COFD[1506] = -4.69342499E-01;
    COFD[1507] = 1.89886374E-02;
    COFD[1508] = -2.26305728E+01;
    COFD[1509] = 5.47666967E+00;
    COFD[1510] = -4.57381900E-01;
    COFD[1511] = 1.82905822E-02;
    COFD[1512] = -2.28655752E+01;
    COFD[1513] = 5.50522401E+00;
    COFD[1514] = -4.63604304E-01;
    COFD[1515] = 1.86600785E-02;
    COFD[1516] = -2.28671232E+01;
    COFD[1517] = 5.50522401E+00;
    COFD[1518] = -4.63604304E-01;
    COFD[1519] = 1.86600785E-02;
    COFD[1520] = -1.50031687E+01;
    COFD[1521] = 3.26223357E+00;
    COFD[1522] = -2.12746642E-01;
    COFD[1523] = 9.38912883E-03;
    COFD[1524] = -1.40999008E+01;
    COFD[1525] = 3.08120012E+00;
    COFD[1526] = -1.89629903E-01;
    COFD[1527] = 8.40361952E-03;
    COFD[1528] = -1.17159737E+01;
    COFD[1529] = 2.48123210E+00;
    COFD[1530] = -1.11322604E-01;
    COFD[1531] = 4.99282389E-03;
    COFD[1532] = -1.43151174E+01;
    COFD[1533] = 3.68038508E+00;
    COFD[1534] = -2.65779346E-01;
    COFD[1535] = 1.16360771E-02;
    COFD[1536] = -1.41191261E+01;
    COFD[1537] = 3.08120012E+00;
    COFD[1538] = -1.89629903E-01;
    COFD[1539] = 8.40361952E-03;
    COFD[1540] = -2.11388331E+01;
    COFD[1541] = 5.55529675E+00;
    COFD[1542] = -4.87942518E-01;
    COFD[1543] = 2.04249054E-02;
    COFD[1544] = -1.52721107E+01;
    COFD[1545] = 3.36790500E+00;
    COFD[1546] = -2.26321740E-01;
    COFD[1547] = 9.97135055E-03;
    COFD[1548] = -1.52792891E+01;
    COFD[1549] = 3.36790500E+00;
    COFD[1550] = -2.26321740E-01;
    COFD[1551] = 9.97135055E-03;
    COFD[1552] = -1.52861376E+01;
    COFD[1553] = 3.36790500E+00;
    COFD[1554] = -2.26321740E-01;
    COFD[1555] = 9.97135055E-03;
    COFD[1556] = -1.81735763E+01;
    COFD[1557] = 4.38391495E+00;
    COFD[1558] = -3.54941287E-01;
    COFD[1559] = 1.54195107E-02;
    COFD[1560] = -1.50233475E+01;
    COFD[1561] = 3.26660767E+00;
    COFD[1562] = -2.13287177E-01;
    COFD[1563] = 9.41137857E-03;
    COFD[1564] = -2.05128705E+01;
    COFD[1565] = 5.23843909E+00;
    COFD[1566] = -4.55815614E-01;
    COFD[1567] = 1.93898040E-02;
    COFD[1568] = -1.59863030E+01;
    COFD[1569] = 3.67388294E+00;
    COFD[1570] = -2.64990709E-01;
    COFD[1571] = 1.16042706E-02;
    COFD[1572] = -1.59525102E+01;
    COFD[1573] = 3.66023858E+00;
    COFD[1574] = -2.63401043E-01;
    COFD[1575] = 1.15432000E-02;
    COFD[1576] = -2.04144604E+01;
    COFD[1577] = 5.19614628E+00;
    COFD[1578] = -4.50889164E-01;
    COFD[1579] = 1.91983328E-02;
    COFD[1580] = -1.82955252E+01;
    COFD[1581] = 4.40289649E+00;
    COFD[1582] = -3.57289765E-01;
    COFD[1583] = 1.55166804E-02;
    COFD[1584] = -2.02922701E+01;
    COFD[1585] = 5.11106992E+00;
    COFD[1586] = -4.42047129E-01;
    COFD[1587] = 1.89042990E-02;
    COFD[1588] = -2.03099025E+01;
    COFD[1589] = 5.11106992E+00;
    COFD[1590] = -4.42047129E-01;
    COFD[1591] = 1.89042990E-02;
    COFD[1592] = -2.04949373E+01;
    COFD[1593] = 5.19614628E+00;
    COFD[1594] = -4.50889164E-01;
    COFD[1595] = 1.91983328E-02;
    COFD[1596] = -1.83296965E+01;
    COFD[1597] = 4.48570999E+00;
    COFD[1598] = -3.67301524E-01;
    COFD[1599] = 1.59204254E-02;
    COFD[1600] = -1.91118445E+01;
    COFD[1601] = 4.68715685E+00;
    COFD[1602] = -3.91979493E-01;
    COFD[1603] = 1.69314004E-02;
    COFD[1604] = -1.79116531E+01;
    COFD[1605] = 4.35148286E+00;
    COFD[1606] = -3.50886647E-01;
    COFD[1607] = 1.52498573E-02;
    COFD[1608] = -1.93011401E+01;
    COFD[1609] = 4.74387793E+00;
    COFD[1610] = -3.98574972E-01;
    COFD[1611] = 1.71862289E-02;
    COFD[1612] = -1.93064215E+01;
    COFD[1613] = 4.74387793E+00;
    COFD[1614] = -3.98574972E-01;
    COFD[1615] = 1.71862289E-02;
    COFD[1616] = -1.92044492E+01;
    COFD[1617] = 4.71304783E+00;
    COFD[1618] = -3.94942083E-01;
    COFD[1619] = 1.70435959E-02;
    COFD[1620] = -1.92334028E+01;
    COFD[1621] = 4.67033934E+00;
    COFD[1622] = -3.89971551E-01;
    COFD[1623] = 1.68513441E-02;
    COFD[1624] = -2.10310742E+01;
    COFD[1625] = 5.23485505E+00;
    COFD[1626] = -4.55400362E-01;
    COFD[1627] = 1.93737680E-02;
    COFD[1628] = -1.97709603E+01;
    COFD[1629] = 4.84731557E+00;
    COFD[1630] = -4.10638352E-01;
    COFD[1631] = 1.76543886E-02;
    COFD[1632] = -1.97422209E+01;
    COFD[1633] = 4.84249900E+00;
    COFD[1634] = -4.10120448E-01;
    COFD[1635] = 1.76363500E-02;
    COFD[1636] = -1.97452574E+01;
    COFD[1637] = 4.84249900E+00;
    COFD[1638] = -4.10120448E-01;
    COFD[1639] = 1.76363500E-02;
    COFD[1640] = -1.98616115E+01;
    COFD[1641] = 4.83466791E+00;
    COFD[1642] = -4.09252052E-01;
    COFD[1643] = 1.76047341E-02;
    COFD[1644] = -2.10924694E+01;
    COFD[1645] = 5.23339224E+00;
    COFD[1646] = -4.55230780E-01;
    COFD[1647] = 1.93672146E-02;
    COFD[1648] = -2.03988322E+01;
    COFD[1649] = 4.99562188E+00;
    COFD[1650] = -4.28482025E-01;
    COFD[1651] = 1.83720948E-02;
    COFD[1652] = -2.03122895E+01;
    COFD[1653] = 4.96244824E+00;
    COFD[1654] = -4.24554494E-01;
    COFD[1655] = 1.82168885E-02;
    COFD[1656] = -2.06812067E+01;
    COFD[1657] = 5.12346096E+00;
    COFD[1658] = -4.43477411E-01;
    COFD[1659] = 1.89592529E-02;
    COFD[1660] = -2.10372026E+01;
    COFD[1661] = 5.20711052E+00;
    COFD[1662] = -4.52173945E-01;
    COFD[1663] = 1.92486273E-02;
    COFD[1664] = -2.10844012E+01;
    COFD[1665] = 5.15315713E+00;
    COFD[1666] = -4.46344043E-01;
    COFD[1667] = 1.90431546E-02;
    COFD[1668] = -2.10855099E+01;
    COFD[1669] = 5.15315713E+00;
    COFD[1670] = -4.46344043E-01;
    COFD[1671] = 1.90431546E-02;
    COFD[1672] = -2.04833713E+01;
    COFD[1673] = 5.23112374E+00;
    COFD[1674] = -4.54967682E-01;
    COFD[1675] = 1.93570423E-02;
    COFD[1676] = -1.94373127E+01;
    COFD[1677] = 5.02567894E+00;
    COFD[1678] = -4.32045169E-01;
    COFD[1679] = 1.85132214E-02;
    COFD[1680] = -1.60528285E+01;
    COFD[1681] = 4.11188603E+00;
    COFD[1682] = -3.21540884E-01;
    COFD[1683] = 1.40482564E-02;
    COFD[1684] = -1.97550088E+01;
    COFD[1685] = 5.56931926E+00;
    COFD[1686] = -4.89105511E-01;
    COFD[1687] = 2.04493129E-02;
    COFD[1688] = -1.94570287E+01;
    COFD[1689] = 5.02567894E+00;
    COFD[1690] = -4.32045169E-01;
    COFD[1691] = 1.85132214E-02;
    COFD[1692] = -1.77563250E+01;
    COFD[1693] = 3.57475686E+00;
    COFD[1694] = -1.56396297E-01;
    COFD[1695] = 3.12157721E-03;
    COFD[1696] = -2.08293255E+01;
    COFD[1697] = 5.35267674E+00;
    COFD[1698] = -4.69010505E-01;
    COFD[1699] = 1.98979152E-02;
    COFD[1700] = -2.08367725E+01;
    COFD[1701] = 5.35267674E+00;
    COFD[1702] = -4.69010505E-01;
    COFD[1703] = 1.98979152E-02;
    COFD[1704] = -2.08438809E+01;
    COFD[1705] = 5.35267674E+00;
    COFD[1706] = -4.69010505E-01;
    COFD[1707] = 1.98979152E-02;
    COFD[1708] = -2.19317743E+01;
    COFD[1709] = 5.45216133E+00;
    COFD[1710] = -4.52916925E-01;
    COFD[1711] = 1.80456400E-02;
    COFD[1712] = -2.05128705E+01;
    COFD[1713] = 5.23843909E+00;
    COFD[1714] = -4.55815614E-01;
    COFD[1715] = 1.93898040E-02;
    COFD[1716] = -1.90499441E+01;
    COFD[1717] = 3.99221757E+00;
    COFD[1718] = -2.19854880E-01;
    COFD[1719] = 6.22736279E-03;
    COFD[1720] = -2.14449559E+01;
    COFD[1721] = 5.56531152E+00;
    COFD[1722] = -4.88789821E-01;
    COFD[1723] = 2.04437116E-02;
    COFD[1724] = -2.14082453E+01;
    COFD[1725] = 5.55346617E+00;
    COFD[1726] = -4.87783156E-01;
    COFD[1727] = 2.04210886E-02;
    COFD[1728] = -1.93214527E+01;
    COFD[1729] = 4.10954793E+00;
    COFD[1730] = -2.37523329E-01;
    COFD[1731] = 7.08858141E-03;
    COFD[1732] = -2.19786173E+01;
    COFD[1733] = 5.43750833E+00;
    COFD[1734] = -4.50273329E-01;
    COFD[1735] = 1.79013718E-02;
    COFD[1736] = -2.01015340E+01;
    COFD[1737] = 4.41511629E+00;
    COFD[1738] = -2.84086963E-01;
    COFD[1739] = 9.37586971E-03;
    COFD[1740] = -2.01199204E+01;
    COFD[1741] = 4.41511629E+00;
    COFD[1742] = -2.84086963E-01;
    COFD[1743] = 9.37586971E-03;
    COFD[1744] = -1.94051843E+01;
    COFD[1745] = 4.10954793E+00;
    COFD[1746] = -2.37523329E-01;
    COFD[1747] = 7.08858141E-03;
    COFD[1748] = -2.16798265E+01;
    COFD[1749] = 5.36811769E+00;
    COFD[1750] = -4.37727086E-01;
    COFD[1751] = 1.72167686E-02;
    COFD[1752] = -2.15802788E+01;
    COFD[1753] = 5.16868516E+00;
    COFD[1754] = -4.03721581E-01;
    COFD[1755] = 1.54206640E-02;
    COFD[1756] = -2.17855148E+01;
    COFD[1757] = 5.47519298E+00;
    COFD[1758] = -4.57113040E-01;
    COFD[1759] = 1.82758312E-02;
    COFD[1760] = -2.14398182E+01;
    COFD[1761] = 5.07680397E+00;
    COFD[1762] = -3.88612087E-01;
    COFD[1763] = 1.46395101E-02;
    COFD[1764] = -2.14453157E+01;
    COFD[1765] = 5.07680397E+00;
    COFD[1766] = -3.88612087E-01;
    COFD[1767] = 1.46395101E-02;
    COFD[1768] = -2.15258568E+01;
    COFD[1769] = 5.12799307E+00;
    COFD[1770] = -3.96938732E-01;
    COFD[1771] = 1.50673195E-02;
    COFD[1772] = -2.17926864E+01;
    COFD[1773] = 5.19232842E+00;
    COFD[1774] = -4.07643284E-01;
    COFD[1775] = 1.56246434E-02;
    COFD[1776] = -1.98359760E+01;
    COFD[1777] = 4.11158627E+00;
    COFD[1778] = -2.37831519E-01;
    COFD[1779] = 7.10363413E-03;
    COFD[1780] = -2.12219677E+01;
    COFD[1781] = 4.87252053E+00;
    COFD[1782] = -3.56127804E-01;
    COFD[1783] = 1.29948788E-02;
    COFD[1784] = -2.12330900E+01;
    COFD[1785] = 4.88535789E+00;
    COFD[1786] = -3.58153894E-01;
    COFD[1787] = 1.30969624E-02;
    COFD[1788] = -2.12362684E+01;
    COFD[1789] = 4.88535789E+00;
    COFD[1790] = -3.58153894E-01;
    COFD[1791] = 1.30969624E-02;
    COFD[1792] = -2.14142864E+01;
    COFD[1793] = 4.90439970E+00;
    COFD[1794] = -3.61162615E-01;
    COFD[1795] = 1.32486109E-02;
    COFD[1796] = -1.96860113E+01;
    COFD[1797] = 4.00653795E+00;
    COFD[1798] = -2.22005804E-01;
    COFD[1799] = 6.33194910E-03;
    COFD[1800] = -2.09922023E+01;
    COFD[1801] = 4.64167142E+00;
    COFD[1802] = -3.19532110E-01;
    COFD[1803] = 1.11478359E-02;
    COFD[1804] = -2.11192145E+01;
    COFD[1805] = 4.70311989E+00;
    COFD[1806] = -3.29240106E-01;
    COFD[1807] = 1.16366808E-02;
    COFD[1808] = -2.03970537E+01;
    COFD[1809] = 4.38396848E+00;
    COFD[1810] = -2.79298901E-01;
    COFD[1811] = 9.13915001E-03;
    COFD[1812] = -1.98603655E+01;
    COFD[1813] = 4.07958166E+00;
    COFD[1814] = -2.33006871E-01;
    COFD[1815] = 6.86822015E-03;
    COFD[1816] = -2.04620510E+01;
    COFD[1817] = 4.26473557E+00;
    COFD[1818] = -2.61033037E-01;
    COFD[1819] = 8.23906412E-03;
    COFD[1820] = -2.04632210E+01;
    COFD[1821] = 4.26473557E+00;
    COFD[1822] = -2.61033037E-01;
    COFD[1823] = 8.23906412E-03;
    COFD[1824] = -1.59633387E+01;
    COFD[1825] = 3.66853818E+00;
    COFD[1826] = -2.64346221E-01;
    COFD[1827] = 1.15784613E-02;
    COFD[1828] = -1.50766130E+01;
    COFD[1829] = 3.47945612E+00;
    COFD[1830] = -2.40703722E-01;
    COFD[1831] = 1.05907441E-02;
    COFD[1832] = -1.25141260E+01;
    COFD[1833] = 2.77873601E+00;
    COFD[1834] = -1.50637360E-01;
    COFD[1835] = 6.72684281E-03;
    COFD[1836] = -1.57994893E+01;
    COFD[1837] = 4.22225052E+00;
    COFD[1838] = -3.35156428E-01;
    COFD[1839] = 1.46104855E-02;
    COFD[1840] = -1.50911794E+01;
    COFD[1841] = 3.47945612E+00;
    COFD[1842] = -2.40703722E-01;
    COFD[1843] = 1.05907441E-02;
    COFD[1844] = -2.12831323E+01;
    COFD[1845] = 5.61184117E+00;
    COFD[1846] = -4.90532156E-01;
    COFD[1847] = 2.03507922E-02;
    COFD[1848] = -1.63493345E+01;
    COFD[1849] = 3.82388595E+00;
    COFD[1850] = -2.84480724E-01;
    COFD[1851] = 1.24506311E-02;
    COFD[1852] = -1.63542394E+01;
    COFD[1853] = 3.82388595E+00;
    COFD[1854] = -2.84480724E-01;
    COFD[1855] = 1.24506311E-02;
    COFD[1856] = -1.63588981E+01;
    COFD[1857] = 3.82388595E+00;
    COFD[1858] = -2.84480724E-01;
    COFD[1859] = 1.24506311E-02;
    COFD[1860] = -1.93276434E+01;
    COFD[1861] = 4.85015581E+00;
    COFD[1862] = -4.10945109E-01;
    COFD[1863] = 1.76651398E-02;
    COFD[1864] = -1.59863030E+01;
    COFD[1865] = 3.67388294E+00;
    COFD[1866] = -2.64990709E-01;
    COFD[1867] = 1.16042706E-02;
    COFD[1868] = -2.14449559E+01;
    COFD[1869] = 5.56531152E+00;
    COFD[1870] = -4.88789821E-01;
    COFD[1871] = 2.04437116E-02;
    COFD[1872] = -1.73374529E+01;
    COFD[1873] = 4.21416723E+00;
    COFD[1874] = -3.34163932E-01;
    COFD[1875] = 1.45697432E-02;
    COFD[1876] = -1.72738845E+01;
    COFD[1877] = 4.19029808E+00;
    COFD[1878] = -3.31177076E-01;
    COFD[1879] = 1.44446234E-02;
    COFD[1880] = -2.13777308E+01;
    COFD[1881] = 5.54007827E+00;
    COFD[1882] = -4.86434511E-01;
    COFD[1883] = 2.03779006E-02;
    COFD[1884] = -1.94819080E+01;
    COFD[1885] = 4.87180830E+00;
    COFD[1886] = -4.13582958E-01;
    COFD[1887] = 1.77726094E-02;
    COFD[1888] = -2.11606963E+01;
    COFD[1889] = 5.42846112E+00;
    COFD[1890] = -4.74321870E-01;
    COFD[1891] = 1.99459749E-02;
    COFD[1892] = -2.11722423E+01;
    COFD[1893] = 5.42846112E+00;
    COFD[1894] = -4.74321870E-01;
    COFD[1895] = 1.99459749E-02;
    COFD[1896] = -2.14314090E+01;
    COFD[1897] = 5.54007827E+00;
    COFD[1898] = -4.86434511E-01;
    COFD[1899] = 2.03779006E-02;
    COFD[1900] = -1.95770968E+01;
    COFD[1901] = 4.97133070E+00;
    COFD[1902] = -4.25604177E-01;
    COFD[1903] = 1.82582594E-02;
    COFD[1904] = -2.02692384E+01;
    COFD[1905] = 5.14418672E+00;
    COFD[1906] = -4.45631004E-01;
    COFD[1907] = 1.90308403E-02;
    COFD[1908] = -1.91225414E+01;
    COFD[1909] = 4.82869066E+00;
    COFD[1910] = -4.08564514E-01;
    COFD[1911] = 1.75784675E-02;
    COFD[1912] = -2.04274471E+01;
    COFD[1913] = 5.18271974E+00;
    COFD[1914] = -4.49323627E-01;
    COFD[1915] = 1.91373940E-02;
    COFD[1916] = -2.04309557E+01;
    COFD[1917] = 5.18271974E+00;
    COFD[1918] = -4.49323627E-01;
    COFD[1919] = 1.91373940E-02;
    COFD[1920] = -2.03367561E+01;
    COFD[1921] = 5.15740122E+00;
    COFD[1922] = -4.46644818E-01;
    COFD[1923] = 1.90459001E-02;
    COFD[1924] = -2.03971290E+01;
    COFD[1925] = 5.13279789E+00;
    COFD[1926] = -4.44474174E-01;
    COFD[1927] = 1.89937678E-02;
    COFD[1928] = -2.18158049E+01;
    COFD[1929] = 5.53950393E+00;
    COFD[1930] = -4.86376204E-01;
    COFD[1931] = 2.03760106E-02;
    COFD[1932] = -2.09490548E+01;
    COFD[1933] = 5.31360223E+00;
    COFD[1934] = -4.64787000E-01;
    COFD[1935] = 1.97483720E-02;
    COFD[1936] = -2.09108261E+01;
    COFD[1937] = 5.30526648E+00;
    COFD[1938] = -4.63785596E-01;
    COFD[1939] = 1.97079873E-02;
    COFD[1940] = -2.09127554E+01;
    COFD[1941] = 5.30526648E+00;
    COFD[1942] = -4.63785596E-01;
    COFD[1943] = 1.97079873E-02;
    COFD[1944] = -2.10124405E+01;
    COFD[1945] = 5.29210705E+00;
    COFD[1946] = -4.62193217E-01;
    COFD[1947] = 1.96432872E-02;
    COFD[1948] = -2.19548723E+01;
    COFD[1949] = 5.56282156E+00;
    COFD[1950] = -4.88585679E-01;
    COFD[1951] = 2.04395879E-02;
    COFD[1952] = -2.13903532E+01;
    COFD[1953] = 5.38519776E+00;
    COFD[1954] = -4.71344997E-01;
    COFD[1955] = 1.99226932E-02;
    COFD[1956] = -2.13459128E+01;
    COFD[1957] = 5.37197338E+00;
    COFD[1958] = -4.70392872E-01;
    COFD[1959] = 1.99122802E-02;
    COFD[1960] = -2.15208595E+01;
    COFD[1961] = 5.44385051E+00;
    COFD[1962] = -4.76121506E-01;
    COFD[1963] = 2.00164081E-02;
    COFD[1964] = -2.19256706E+01;
    COFD[1965] = 5.54768472E+00;
    COFD[1966] = -4.87202065E-01;
    COFD[1967] = 2.04025437E-02;
    COFD[1968] = -2.19550907E+01;
    COFD[1969] = 5.49350509E+00;
    COFD[1970] = -4.81613405E-01;
    COFD[1971] = 2.02171734E-02;
    COFD[1972] = -2.19557531E+01;
    COFD[1973] = 5.49350509E+00;
    COFD[1974] = -4.81613405E-01;
    COFD[1975] = 2.02171734E-02;
    COFD[1976] = -1.59327297E+01;
    COFD[1977] = 3.65620899E+00;
    COFD[1978] = -2.62933804E-01;
    COFD[1979] = 1.15253223E-02;
    COFD[1980] = -1.50270339E+01;
    COFD[1981] = 3.46140064E+00;
    COFD[1982] = -2.38440092E-01;
    COFD[1983] = 1.04960087E-02;
    COFD[1984] = -1.24693568E+01;
    COFD[1985] = 2.76686648E+00;
    COFD[1986] = -1.49120141E-01;
    COFD[1987] = 6.66220432E-03;
    COFD[1988] = -1.57199037E+01;
    COFD[1989] = 4.19936335E+00;
    COFD[1990] = -3.32311009E-01;
    COFD[1991] = 1.44921003E-02;
    COFD[1992] = -1.50420953E+01;
    COFD[1993] = 3.46140064E+00;
    COFD[1994] = -2.38440092E-01;
    COFD[1995] = 1.04960087E-02;
    COFD[1996] = -2.14087397E+01;
    COFD[1997] = 5.57282008E+00;
    COFD[1998] = -4.76690890E-01;
    COFD[1999] = 1.94000719E-02;
    COFD[2000] = -1.62724462E+01;
    COFD[2001] = 3.79163564E+00;
    COFD[2002] = -2.80257365E-01;
    COFD[2003] = 1.22656902E-02;
    COFD[2004] = -1.62775714E+01;
    COFD[2005] = 3.79163564E+00;
    COFD[2006] = -2.80257365E-01;
    COFD[2007] = 1.22656902E-02;
    COFD[2008] = -1.62824412E+01;
    COFD[2009] = 3.79163564E+00;
    COFD[2010] = -2.80257365E-01;
    COFD[2011] = 1.22656902E-02;
    COFD[2012] = -1.92867554E+01;
    COFD[2013] = 4.83375900E+00;
    COFD[2014] = -4.09146560E-01;
    COFD[2015] = 1.76006599E-02;
    COFD[2016] = -1.59525102E+01;
    COFD[2017] = 3.66023858E+00;
    COFD[2018] = -2.63401043E-01;
    COFD[2019] = 1.15432000E-02;
    COFD[2020] = -2.14082453E+01;
    COFD[2021] = 5.55346617E+00;
    COFD[2022] = -4.87783156E-01;
    COFD[2023] = 2.04210886E-02;
    COFD[2024] = -1.72738845E+01;
    COFD[2025] = 4.19029808E+00;
    COFD[2026] = -3.31177076E-01;
    COFD[2027] = 1.44446234E-02;
    COFD[2028] = -1.72167708E+01;
    COFD[2029] = 4.16886779E+00;
    COFD[2030] = -3.28518156E-01;
    COFD[2031] = 1.43341626E-02;
    COFD[2032] = -2.13319784E+01;
    COFD[2033] = 5.52422470E+00;
    COFD[2034] = -4.84872944E-01;
    COFD[2035] = 2.03298213E-02;
    COFD[2036] = -1.94186547E+01;
    COFD[2037] = 4.84669430E+00;
    COFD[2038] = -4.10571455E-01;
    COFD[2039] = 1.76520543E-02;
    COFD[2040] = -2.11309207E+01;
    COFD[2041] = 5.41773516E+00;
    COFD[2042] = -4.73414338E-01;
    COFD[2043] = 1.99258685E-02;
    COFD[2044] = -2.11430338E+01;
    COFD[2045] = 5.41773516E+00;
    COFD[2046] = -4.73414338E-01;
    COFD[2047] = 1.99258685E-02;
    COFD[2048] = -2.13881945E+01;
    COFD[2049] = 5.52422470E+00;
    COFD[2050] = -4.84872944E-01;
    COFD[2051] = 2.03298213E-02;
    COFD[2052] = -1.95154079E+01;
    COFD[2053] = 4.94787350E+00;
    COFD[2054] = -4.22829292E-01;
    COFD[2055] = 1.81487163E-02;
    COFD[2056] = -2.02318658E+01;
    COFD[2057] = 5.12963391E+00;
    COFD[2058] = -4.44146826E-01;
    COFD[2059] = 1.89829640E-02;
    COFD[2060] = -1.90692595E+01;
    COFD[2061] = 4.80830699E+00;
    COFD[2062] = -4.06171933E-01;
    COFD[2063] = 1.74848791E-02;
    COFD[2064] = -2.03738891E+01;
    COFD[2065] = 5.16159436E+00;
    COFD[2066] = -4.46935283E-01;
    COFD[2067] = 1.90480297E-02;
    COFD[2068] = -2.03775651E+01;
    COFD[2069] = 5.16159436E+00;
    COFD[2070] = -4.46935283E-01;
    COFD[2071] = 1.90480297E-02;
    COFD[2072] = -2.03123540E+01;
    COFD[2073] = 5.14854169E+00;
    COFD[2074] = -4.45984343E-01;
    COFD[2075] = 1.90374217E-02;
    COFD[2076] = -2.03526104E+01;
    COFD[2077] = 5.11453301E+00;
    COFD[2078] = -4.42447016E-01;
    COFD[2079] = 1.89196698E-02;
    COFD[2080] = -2.18731920E+01;
    COFD[2081] = 5.55171660E+00;
    COFD[2082] = -4.87609504E-01;
    COFD[2083] = 2.04156590E-02;
    COFD[2084] = -2.08822487E+01;
    COFD[2085] = 5.28557747E+00;
    COFD[2086] = -4.61402384E-01;
    COFD[2087] = 1.96111546E-02;
    COFD[2088] = -2.08427678E+01;
    COFD[2089] = 5.27674330E+00;
    COFD[2090] = -4.60336155E-01;
    COFD[2091] = 1.95680191E-02;
    COFD[2092] = -2.08447974E+01;
    COFD[2093] = 5.27674330E+00;
    COFD[2094] = -4.60336155E-01;
    COFD[2095] = 1.95680191E-02;
    COFD[2096] = -2.09461018E+01;
    COFD[2097] = 5.26396793E+00;
    COFD[2098] = -4.58812213E-01;
    COFD[2099] = 1.95072180E-02;
    COFD[2100] = -2.19244555E+01;
    COFD[2101] = 5.54986547E+00;
    COFD[2102] = -4.87420926E-01;
    COFD[2103] = 2.04095097E-02;
    COFD[2104] = -2.13695648E+01;
    COFD[2105] = 5.37614538E+00;
    COFD[2106] = -4.70679659E-01;
    COFD[2107] = 1.99143937E-02;
    COFD[2108] = -2.13282915E+01;
    COFD[2109] = 5.36375915E+00;
    COFD[2110] = -4.69808195E-01;
    COFD[2111] = 1.99064589E-02;
    COFD[2112] = -2.14671205E+01;
    COFD[2113] = 5.42109069E+00;
    COFD[2114] = -4.73533096E-01;
    COFD[2115] = 1.99183547E-02;
    COFD[2116] = -2.18876256E+01;
    COFD[2117] = 5.53154746E+00;
    COFD[2118] = -4.85594344E-01;
    COFD[2119] = 2.03520324E-02;
    COFD[2120] = -2.19053841E+01;
    COFD[2121] = 5.47162499E+00;
    COFD[2122] = -4.79195552E-01;
    COFD[2123] = 2.01289088E-02;
    COFD[2124] = -2.19060847E+01;
    COFD[2125] = 5.47162499E+00;
    COFD[2126] = -4.79195552E-01;
    COFD[2127] = 2.01289088E-02;
    COFD[2128] = -2.03844252E+01;
    COFD[2129] = 5.18856872E+00;
    COFD[2130] = -4.50001829E-01;
    COFD[2131] = 1.91636142E-02;
    COFD[2132] = -1.93364585E+01;
    COFD[2133] = 4.98286777E+00;
    COFD[2134] = -4.26970814E-01;
    COFD[2135] = 1.83122917E-02;
    COFD[2136] = -1.59537247E+01;
    COFD[2137] = 4.07051484E+00;
    COFD[2138] = -3.16303109E-01;
    COFD[2139] = 1.38259377E-02;
    COFD[2140] = -1.96866103E+01;
    COFD[2141] = 5.54637286E+00;
    COFD[2142] = -4.87070324E-01;
    COFD[2143] = 2.03983467E-02;
    COFD[2144] = -1.93566243E+01;
    COFD[2145] = 4.98286777E+00;
    COFD[2146] = -4.26970814E-01;
    COFD[2147] = 1.83122917E-02;
    COFD[2148] = -1.80253664E+01;
    COFD[2149] = 3.69199168E+00;
    COFD[2150] = -1.74005516E-01;
    COFD[2151] = 3.97694372E-03;
    COFD[2152] = -2.07595845E+01;
    COFD[2153] = 5.32244593E+00;
    COFD[2154] = -4.65829403E-01;
    COFD[2155] = 1.97895274E-02;
    COFD[2156] = -2.07672833E+01;
    COFD[2157] = 5.32244593E+00;
    COFD[2158] = -4.65829403E-01;
    COFD[2159] = 1.97895274E-02;
    COFD[2160] = -2.07746356E+01;
    COFD[2161] = 5.32244593E+00;
    COFD[2162] = -4.65829403E-01;
    COFD[2163] = 1.97895274E-02;
    COFD[2164] = -2.20063594E+01;
    COFD[2165] = 5.48540187E+00;
    COFD[2166] = -4.58962148E-01;
    COFD[2167] = 1.83770355E-02;
    COFD[2168] = -2.04144604E+01;
    COFD[2169] = 5.19614628E+00;
    COFD[2170] = -4.50889164E-01;
    COFD[2171] = 1.91983328E-02;
    COFD[2172] = -1.93214527E+01;
    COFD[2173] = 4.10954793E+00;
    COFD[2174] = -2.37523329E-01;
    COFD[2175] = 7.08858141E-03;
    COFD[2176] = -2.13777308E+01;
    COFD[2177] = 5.54007827E+00;
    COFD[2178] = -4.86434511E-01;
    COFD[2179] = 2.03779006E-02;
    COFD[2180] = -2.13319784E+01;
    COFD[2181] = 5.52422470E+00;
    COFD[2182] = -4.84872944E-01;
    COFD[2183] = 2.03298213E-02;
    COFD[2184] = -1.95785144E+01;
    COFD[2185] = 4.22062499E+00;
    COFD[2186] = -2.54326872E-01;
    COFD[2187] = 7.91017784E-03;
    COFD[2188] = -2.20495822E+01;
    COFD[2189] = 5.47072190E+00;
    COFD[2190] = -4.56301261E-01;
    COFD[2191] = 1.82313566E-02;
    COFD[2192] = -2.03036402E+01;
    COFD[2193] = 4.50250781E+00;
    COFD[2194] = -2.97622106E-01;
    COFD[2195] = 1.00481473E-02;
    COFD[2196] = -2.03227406E+01;
    COFD[2197] = 4.50250781E+00;
    COFD[2198] = -2.97622106E-01;
    COFD[2199] = 1.00481473E-02;
    COFD[2200] = -1.96653154E+01;
    COFD[2201] = 4.22062499E+00;
    COFD[2202] = -2.54326872E-01;
    COFD[2203] = 7.91017784E-03;
    COFD[2204] = -2.17547312E+01;
    COFD[2205] = 5.40298848E+00;
    COFD[2206] = -4.43954594E-01;
    COFD[2207] = 1.75542998E-02;
    COFD[2208] = -2.16936515E+01;
    COFD[2209] = 5.21869603E+00;
    COFD[2210] = -4.12084772E-01;
    COFD[2211] = 1.58573035E-02;
    COFD[2212] = -2.18356866E+01;
    COFD[2213] = 5.49906960E+00;
    COFD[2214] = -4.61793001E-01;
    COFD[2215] = 1.85415189E-02;
    COFD[2216] = -2.15759895E+01;
    COFD[2217] = 5.13708607E+00;
    COFD[2218] = -3.98445708E-01;
    COFD[2219] = 1.51455626E-02;
    COFD[2220] = -2.15816909E+01;
    COFD[2221] = 5.13708607E+00;
    COFD[2222] = -3.98445708E-01;
    COFD[2223] = 1.51455626E-02;
    COFD[2224] = -2.16420936E+01;
    COFD[2225] = 5.17945041E+00;
    COFD[2226] = -4.05514689E-01;
    COFD[2227] = 1.55141412E-02;
    COFD[2228] = -2.18910102E+01;
    COFD[2229] = 5.23595129E+00;
    COFD[2230] = -4.15079064E-01;
    COFD[2231] = 1.60168286E-02;
    COFD[2232] = -2.00981944E+01;
    COFD[2233] = 4.22278378E+00;
    COFD[2234] = -2.54653500E-01;
    COFD[2235] = 7.92616085E-03;
    COFD[2236] = -2.13985484E+01;
    COFD[2237] = 4.94878244E+00;
    COFD[2238] = -3.68158605E-01;
    COFD[2239] = 1.36008797E-02;
    COFD[2240] = -2.14111310E+01;
    COFD[2241] = 4.96219227E+00;
    COFD[2242] = -3.70270843E-01;
    COFD[2243] = 1.37072211E-02;
    COFD[2244] = -2.14144448E+01;
    COFD[2245] = 4.96219227E+00;
    COFD[2246] = -3.70270843E-01;
    COFD[2247] = 1.37072211E-02;
    COFD[2248] = -2.15952753E+01;
    COFD[2249] = 4.98271982E+00;
    COFD[2250] = -3.73502341E-01;
    COFD[2251] = 1.38698700E-02;
    COFD[2252] = -1.99604682E+01;
    COFD[2253] = 4.12245214E+00;
    COFD[2254] = -2.39476227E-01;
    COFD[2255] = 7.18400558E-03;
    COFD[2256] = -2.11660262E+01;
    COFD[2257] = 4.71644372E+00;
    COFD[2258] = -3.31349990E-01;
    COFD[2259] = 1.17430818E-02;
    COFD[2260] = -2.12804720E+01;
    COFD[2261] = 4.77238689E+00;
    COFD[2262] = -3.40265855E-01;
    COFD[2263] = 1.21942137E-02;
    COFD[2264] = -2.06103015E+01;
    COFD[2265] = 4.47491202E+00;
    COFD[2266] = -2.93331059E-01;
    COFD[2267] = 9.83445305E-03;
    COFD[2268] = -2.01250987E+01;
    COFD[2269] = 4.19160608E+00;
    COFD[2270] = -2.49936771E-01;
    COFD[2271] = 7.69538319E-03;
    COFD[2272] = -2.06858147E+01;
    COFD[2273] = 4.35920123E+00;
    COFD[2274] = -2.75491273E-01;
    COFD[2275] = 8.95100289E-03;
    COFD[2276] = -2.06870442E+01;
    COFD[2277] = 4.35920123E+00;
    COFD[2278] = -2.75491273E-01;
    COFD[2279] = 8.95100289E-03;
    COFD[2280] = -1.82673770E+01;
    COFD[2281] = 4.39538102E+00;
    COFD[2282] = -3.56367230E-01;
    COFD[2283] = 1.54788461E-02;
    COFD[2284] = -1.72112971E+01;
    COFD[2285] = 4.15807461E+00;
    COFD[2286] = -3.27178539E-01;
    COFD[2287] = 1.42784349E-02;
    COFD[2288] = -1.39658996E+01;
    COFD[2289] = 3.24966086E+00;
    COFD[2290] = -2.11199992E-01;
    COFD[2291] = 9.32580661E-03;
    COFD[2292] = -1.78637178E+01;
    COFD[2293] = 4.88268692E+00;
    COFD[2294] = -4.14917638E-01;
    COFD[2295] = 1.78274298E-02;
    COFD[2296] = -1.72310232E+01;
    COFD[2297] = 4.15807461E+00;
    COFD[2298] = -3.27178539E-01;
    COFD[2299] = 1.42784349E-02;
    COFD[2300] = -2.13148887E+01;
    COFD[2301] = 5.27210469E+00;
    COFD[2302] = -4.21419216E-01;
    COFD[2303] = 1.63567178E-02;
    COFD[2304] = -1.85844688E+01;
    COFD[2305] = 4.51052425E+00;
    COFD[2306] = -3.70301627E-01;
    COFD[2307] = 1.60416153E-02;
    COFD[2308] = -1.85919214E+01;
    COFD[2309] = 4.51052425E+00;
    COFD[2310] = -3.70301627E-01;
    COFD[2311] = 1.60416153E-02;
    COFD[2312] = -1.85990352E+01;
    COFD[2313] = 4.51052425E+00;
    COFD[2314] = -3.70301627E-01;
    COFD[2315] = 1.60416153E-02;
    COFD[2316] = -2.14151520E+01;
    COFD[2317] = 5.41122754E+00;
    COFD[2318] = -4.73185889E-01;
    COFD[2319] = 1.99407905E-02;
    COFD[2320] = -1.82955252E+01;
    COFD[2321] = 4.40289649E+00;
    COFD[2322] = -3.57289765E-01;
    COFD[2323] = 1.55166804E-02;
    COFD[2324] = -2.19786173E+01;
    COFD[2325] = 5.43750833E+00;
    COFD[2326] = -4.50273329E-01;
    COFD[2327] = 1.79013718E-02;
    COFD[2328] = -1.94819080E+01;
    COFD[2329] = 4.87180830E+00;
    COFD[2330] = -4.13582958E-01;
    COFD[2331] = 1.77726094E-02;
    COFD[2332] = -1.94186547E+01;
    COFD[2333] = 4.84669430E+00;
    COFD[2334] = -4.10571455E-01;
    COFD[2335] = 1.76520543E-02;
    COFD[2336] = -2.20495822E+01;
    COFD[2337] = 5.47072190E+00;
    COFD[2338] = -4.56301261E-01;
    COFD[2339] = 1.82313566E-02;
    COFD[2340] = -2.14907782E+01;
    COFD[2341] = 5.41585806E+00;
    COFD[2342] = -4.73359323E-01;
    COFD[2343] = 1.99310239E-02;
    COFD[2344] = -2.22429814E+01;
    COFD[2345] = 5.53139819E+00;
    COFD[2346] = -4.68828555E-01;
    COFD[2347] = 1.89597887E-02;
    COFD[2348] = -2.22613837E+01;
    COFD[2349] = 5.53139819E+00;
    COFD[2350] = -4.68828555E-01;
    COFD[2351] = 1.89597887E-02;
    COFD[2352] = -2.21333822E+01;
    COFD[2353] = 5.47072190E+00;
    COFD[2354] = -4.56301261E-01;
    COFD[2355] = 1.82313566E-02;
    COFD[2356] = -2.15206146E+01;
    COFD[2357] = 5.48426911E+00;
    COFD[2358] = -4.80606512E-01;
    COFD[2359] = 2.01811046E-02;
    COFD[2360] = -2.21343023E+01;
    COFD[2361] = 5.60010742E+00;
    COFD[2362] = -4.91597429E-01;
    COFD[2363] = 2.04987718E-02;
    COFD[2364] = -2.12014186E+01;
    COFD[2365] = 5.40060531E+00;
    COFD[2366] = -4.72449699E-01;
    COFD[2367] = 1.99345817E-02;
    COFD[2368] = -2.22262162E+01;
    COFD[2369] = 5.61211818E+00;
    COFD[2370] = -4.91432482E-01;
    COFD[2371] = 2.04238731E-02;
    COFD[2372] = -2.22317182E+01;
    COFD[2373] = 5.61211818E+00;
    COFD[2374] = -4.91432482E-01;
    COFD[2375] = 2.04238731E-02;
    COFD[2376] = -2.21793326E+01;
    COFD[2377] = 5.60403905E+00;
    COFD[2378] = -4.91221691E-01;
    COFD[2379] = 2.04473483E-02;
    COFD[2380] = -2.22701953E+01;
    COFD[2381] = 5.59632316E+00;
    COFD[2382] = -4.91568011E-01;
    COFD[2383] = 2.05156966E-02;
    COFD[2384] = -2.25302512E+01;
    COFD[2385] = 5.47136127E+00;
    COFD[2386] = -4.56417141E-01;
    COFD[2387] = 1.82376994E-02;
    COFD[2388] = -2.24120415E+01;
    COFD[2389] = 5.58744076E+00;
    COFD[2390] = -4.84489462E-01;
    COFD[2391] = 1.99733042E-02;
    COFD[2392] = -2.23993836E+01;
    COFD[2393] = 5.58952429E+00;
    COFD[2394] = -4.85012530E-01;
    COFD[2395] = 2.00062142E-02;
    COFD[2396] = -2.24025650E+01;
    COFD[2397] = 5.58952429E+00;
    COFD[2398] = -4.85012530E-01;
    COFD[2399] = 2.00062142E-02;
    COFD[2400] = -2.25300734E+01;
    COFD[2401] = 5.59173268E+00;
    COFD[2402] = -4.85654660E-01;
    COFD[2403] = 2.00483698E-02;
    COFD[2404] = -2.25553202E+01;
    COFD[2405] = 5.44166443E+00;
    COFD[2406] = -4.51021243E-01;
    COFD[2407] = 1.79421190E-02;
    COFD[2408] = -2.27001899E+01;
    COFD[2409] = 5.58468914E+00;
    COFD[2410] = -4.79958407E-01;
    COFD[2411] = 1.96104043E-02;
    COFD[2412] = -2.26853912E+01;
    COFD[2413] = 5.58521030E+00;
    COFD[2414] = -4.81061650E-01;
    COFD[2415] = 1.96992215E-02;
    COFD[2416] = -2.25695574E+01;
    COFD[2417] = 5.52323975E+00;
    COFD[2418] = -4.67257607E-01;
    COFD[2419] = 1.88711975E-02;
    COFD[2420] = -2.26027431E+01;
    COFD[2421] = 5.46217527E+00;
    COFD[2422] = -4.54751471E-01;
    COFD[2423] = 1.81465218E-02;
    COFD[2424] = -2.28446667E+01;
    COFD[2425] = 5.50134401E+00;
    COFD[2426] = -4.62488197E-01;
    COFD[2427] = 1.85873697E-02;
    COFD[2428] = -2.28458380E+01;
    COFD[2429] = 5.50134401E+00;
    COFD[2430] = -4.62488197E-01;
    COFD[2431] = 1.85873697E-02;
    COFD[2432] = -2.02646611E+01;
    COFD[2433] = 5.10426133E+00;
    COFD[2434] = -4.41256919E-01;
    COFD[2435] = 1.88737290E-02;
    COFD[2436] = -1.90883268E+01;
    COFD[2437] = 4.84384483E+00;
    COFD[2438] = -4.10265575E-01;
    COFD[2439] = 1.76414287E-02;
    COFD[2440] = -1.57034851E+01;
    COFD[2441] = 3.93614244E+00;
    COFD[2442] = -2.99111497E-01;
    COFD[2443] = 1.30888229E-02;
    COFD[2444] = -1.94688688E+01;
    COFD[2445] = 5.43830787E+00;
    COFD[2446] = -4.75472880E-01;
    COFD[2447] = 1.99909996E-02;
    COFD[2448] = -1.91102652E+01;
    COFD[2449] = 4.84384483E+00;
    COFD[2450] = -4.10265575E-01;
    COFD[2451] = 1.76414287E-02;
    COFD[2452] = -1.87383952E+01;
    COFD[2453] = 3.96926341E+00;
    COFD[2454] = -2.16412264E-01;
    COFD[2455] = 6.06012078E-03;
    COFD[2456] = -2.05184870E+01;
    COFD[2457] = 5.18417470E+00;
    COFD[2458] = -4.49491573E-01;
    COFD[2459] = 1.91438508E-02;
    COFD[2460] = -2.05272328E+01;
    COFD[2461] = 5.18417470E+00;
    COFD[2462] = -4.49491573E-01;
    COFD[2463] = 1.91438508E-02;
    COFD[2464] = -2.05356023E+01;
    COFD[2465] = 5.18417470E+00;
    COFD[2466] = -4.49491573E-01;
    COFD[2467] = 1.91438508E-02;
    COFD[2468] = -2.22116706E+01;
    COFD[2469] = 5.54251230E+00;
    COFD[2470] = -4.70946314E-01;
    COFD[2471] = 1.90785869E-02;
    COFD[2472] = -2.02922701E+01;
    COFD[2473] = 5.11106992E+00;
    COFD[2474] = -4.42047129E-01;
    COFD[2475] = 1.89042990E-02;
    COFD[2476] = -2.01015340E+01;
    COFD[2477] = 4.41511629E+00;
    COFD[2478] = -2.84086963E-01;
    COFD[2479] = 9.37586971E-03;
    COFD[2480] = -2.11606963E+01;
    COFD[2481] = 5.42846112E+00;
    COFD[2482] = -4.74321870E-01;
    COFD[2483] = 1.99459749E-02;
    COFD[2484] = -2.11309207E+01;
    COFD[2485] = 5.41773516E+00;
    COFD[2486] = -4.73414338E-01;
    COFD[2487] = 1.99258685E-02;
    COFD[2488] = -2.03036402E+01;
    COFD[2489] = 4.50250781E+00;
    COFD[2490] = -2.97622106E-01;
    COFD[2491] = 1.00481473E-02;
    COFD[2492] = -2.22429814E+01;
    COFD[2493] = 5.53139819E+00;
    COFD[2494] = -4.68828555E-01;
    COFD[2495] = 1.89597887E-02;
    COFD[2496] = -2.09002742E+01;
    COFD[2497] = 4.72895031E+00;
    COFD[2498] = -3.33332771E-01;
    COFD[2499] = 1.18431478E-02;
    COFD[2500] = -2.09224206E+01;
    COFD[2501] = 4.72895031E+00;
    COFD[2502] = -3.33332771E-01;
    COFD[2503] = 1.18431478E-02;
    COFD[2504] = -2.04033972E+01;
    COFD[2505] = 4.50250781E+00;
    COFD[2506] = -2.97622106E-01;
    COFD[2507] = 1.00481473E-02;
    COFD[2508] = -2.20262793E+01;
    COFD[2509] = 5.49663315E+00;
    COFD[2510] = -4.61182837E-01;
    COFD[2511] = 1.85035558E-02;
    COFD[2512] = -2.20597305E+01;
    COFD[2513] = 5.34774760E+00;
    COFD[2514] = -4.34239753E-01;
    COFD[2515] = 1.70320676E-02;
    COFD[2516] = -2.20398328E+01;
    COFD[2517] = 5.56049839E+00;
    COFD[2518] = -4.74367872E-01;
    COFD[2519] = 1.92702787E-02;
    COFD[2520] = -2.19526490E+01;
    COFD[2521] = 5.27258289E+00;
    COFD[2522] = -4.21502790E-01;
    COFD[2523] = 1.63611949E-02;
    COFD[2524] = -2.19592125E+01;
    COFD[2525] = 5.27258289E+00;
    COFD[2526] = -4.21502790E-01;
    COFD[2527] = 1.63611949E-02;
    COFD[2528] = -2.20192352E+01;
    COFD[2529] = 5.31412694E+00;
    COFD[2530] = -4.28473898E-01;
    COFD[2531] = 1.67264841E-02;
    COFD[2532] = -2.22545356E+01;
    COFD[2533] = 5.36643605E+00;
    COFD[2534] = -4.37440735E-01;
    COFD[2535] = 1.72016388E-02;
    COFD[2536] = -2.08353693E+01;
    COFD[2537] = 4.50409026E+00;
    COFD[2538] = -2.97868419E-01;
    COFD[2539] = 1.00604224E-02;
    COFD[2540] = -2.19253091E+01;
    COFD[2541] = 5.14570932E+00;
    COFD[2542] = -3.99877142E-01;
    COFD[2543] = 1.52199557E-02;
    COFD[2544] = -2.19282979E+01;
    COFD[2545] = 5.15446948E+00;
    COFD[2546] = -4.01332769E-01;
    COFD[2547] = 1.52956262E-02;
    COFD[2548] = -2.19322003E+01;
    COFD[2549] = 5.15446948E+00;
    COFD[2550] = -4.01332769E-01;
    COFD[2551] = 1.52956262E-02;
    COFD[2552] = -2.20891322E+01;
    COFD[2553] = 5.16679492E+00;
    COFD[2554] = -4.03405751E-01;
    COFD[2555] = 1.54041741E-02;
    COFD[2556] = -2.07557953E+01;
    COFD[2557] = 4.42680848E+00;
    COFD[2558] = -2.85885288E-01;
    COFD[2559] = 9.46483934E-03;
    COFD[2560] = -2.17463767E+01;
    COFD[2561] = 4.93496210E+00;
    COFD[2562] = -3.65981745E-01;
    COFD[2563] = 1.34912948E-02;
    COFD[2564] = -2.18797352E+01;
    COFD[2565] = 4.99907484E+00;
    COFD[2566] = -3.76094627E-01;
    COFD[2567] = 1.40009262E-02;
    COFD[2568] = -2.12221678E+01;
    COFD[2569] = 4.70506024E+00;
    COFD[2570] = -3.29547212E-01;
    COFD[2571] = 1.16521630E-02;
    COFD[2572] = -2.08821587E+01;
    COFD[2573] = 4.48108132E+00;
    COFD[2574] = -2.94289899E-01;
    COFD[2575] = 9.88218297E-03;
    COFD[2576] = -2.13524540E+01;
    COFD[2577] = 4.61201872E+00;
    COFD[2578] = -3.14803338E-01;
    COFD[2579] = 1.09082984E-02;
    COFD[2580] = -2.13539532E+01;
    COFD[2581] = 4.61201872E+00;
    COFD[2582] = -3.14803338E-01;
    COFD[2583] = 1.09082984E-02;
    COFD[2584] = -2.02822946E+01;
    COFD[2585] = 5.10426133E+00;
    COFD[2586] = -4.41256919E-01;
    COFD[2587] = 1.88737290E-02;
    COFD[2588] = -1.91004157E+01;
    COFD[2589] = 4.84384483E+00;
    COFD[2590] = -4.10265575E-01;
    COFD[2591] = 1.76414287E-02;
    COFD[2592] = -1.57054717E+01;
    COFD[2593] = 3.93614244E+00;
    COFD[2594] = -2.99111497E-01;
    COFD[2595] = 1.30888229E-02;
    COFD[2596] = -1.94698843E+01;
    COFD[2597] = 5.43830787E+00;
    COFD[2598] = -4.75472880E-01;
    COFD[2599] = 1.99909996E-02;
    COFD[2600] = -1.91229033E+01;
    COFD[2601] = 4.84384483E+00;
    COFD[2602] = -4.10265575E-01;
    COFD[2603] = 1.76414287E-02;
    COFD[2604] = -1.87515645E+01;
    COFD[2605] = 3.96926341E+00;
    COFD[2606] = -2.16412264E-01;
    COFD[2607] = 6.06012078E-03;
    COFD[2608] = -2.05375724E+01;
    COFD[2609] = 5.18417470E+00;
    COFD[2610] = -4.49491573E-01;
    COFD[2611] = 1.91438508E-02;
    COFD[2612] = -2.05466616E+01;
    COFD[2613] = 5.18417470E+00;
    COFD[2614] = -4.49491573E-01;
    COFD[2615] = 1.91438508E-02;
    COFD[2616] = -2.05553656E+01;
    COFD[2617] = 5.18417470E+00;
    COFD[2618] = -4.49491573E-01;
    COFD[2619] = 1.91438508E-02;
    COFD[2620] = -2.22343363E+01;
    COFD[2621] = 5.54251230E+00;
    COFD[2622] = -4.70946314E-01;
    COFD[2623] = 1.90785869E-02;
    COFD[2624] = -2.03099025E+01;
    COFD[2625] = 5.11106992E+00;
    COFD[2626] = -4.42047129E-01;
    COFD[2627] = 1.89042990E-02;
    COFD[2628] = -2.01199204E+01;
    COFD[2629] = 4.41511629E+00;
    COFD[2630] = -2.84086963E-01;
    COFD[2631] = 9.37586971E-03;
    COFD[2632] = -2.11722423E+01;
    COFD[2633] = 5.42846112E+00;
    COFD[2634] = -4.74321870E-01;
    COFD[2635] = 1.99459749E-02;
    COFD[2636] = -2.11430338E+01;
    COFD[2637] = 5.41773516E+00;
    COFD[2638] = -4.73414338E-01;
    COFD[2639] = 1.99258685E-02;
    COFD[2640] = -2.03227406E+01;
    COFD[2641] = 4.50250781E+00;
    COFD[2642] = -2.97622106E-01;
    COFD[2643] = 1.00481473E-02;
    COFD[2644] = -2.22613837E+01;
    COFD[2645] = 5.53139819E+00;
    COFD[2646] = -4.68828555E-01;
    COFD[2647] = 1.89597887E-02;
    COFD[2648] = -2.09224206E+01;
    COFD[2649] = 4.72895031E+00;
    COFD[2650] = -3.33332771E-01;
    COFD[2651] = 1.18431478E-02;
    COFD[2652] = -2.09455936E+01;
    COFD[2653] = 4.72895031E+00;
    COFD[2654] = -3.33332771E-01;
    COFD[2655] = 1.18431478E-02;
    COFD[2656] = -2.04268153E+01;
    COFD[2657] = 4.50250781E+00;
    COFD[2658] = -2.97622106E-01;
    COFD[2659] = 1.00481473E-02;
    COFD[2660] = -2.20431319E+01;
    COFD[2661] = 5.49663315E+00;
    COFD[2662] = -4.61182837E-01;
    COFD[2663] = 1.85035558E-02;
    COFD[2664] = -2.20818886E+01;
    COFD[2665] = 5.34774760E+00;
    COFD[2666] = -4.34239753E-01;
    COFD[2667] = 1.70320676E-02;
    COFD[2668] = -2.20574820E+01;
    COFD[2669] = 5.56049839E+00;
    COFD[2670] = -4.74367872E-01;
    COFD[2671] = 1.92702787E-02;
    COFD[2672] = -2.19739638E+01;
    COFD[2673] = 5.27258289E+00;
    COFD[2674] = -4.21502790E-01;
    COFD[2675] = 1.63611949E-02;
    COFD[2676] = -2.19808152E+01;
    COFD[2677] = 5.27258289E+00;
    COFD[2678] = -4.21502790E-01;
    COFD[2679] = 1.63611949E-02;
    COFD[2680] = -2.20411190E+01;
    COFD[2681] = 5.31412694E+00;
    COFD[2682] = -4.28473898E-01;
    COFD[2683] = 1.67264841E-02;
    COFD[2684] = -2.22769618E+01;
    COFD[2685] = 5.36643605E+00;
    COFD[2686] = -4.37440735E-01;
    COFD[2687] = 1.72016388E-02;
    COFD[2688] = -2.08639466E+01;
    COFD[2689] = 4.50409026E+00;
    COFD[2690] = -2.97868419E-01;
    COFD[2691] = 1.00604224E-02;
    COFD[2692] = -2.19503032E+01;
    COFD[2693] = 5.14570932E+00;
    COFD[2694] = -3.99877142E-01;
    COFD[2695] = 1.52199557E-02;
    COFD[2696] = -2.19534987E+01;
    COFD[2697] = 5.15446948E+00;
    COFD[2698] = -4.01332769E-01;
    COFD[2699] = 1.52956262E-02;
    COFD[2700] = -2.19576037E+01;
    COFD[2701] = 5.15446948E+00;
    COFD[2702] = -4.01332769E-01;
    COFD[2703] = 1.52956262E-02;
    COFD[2704] = -2.21147341E+01;
    COFD[2705] = 5.16679492E+00;
    COFD[2706] = -4.03405751E-01;
    COFD[2707] = 1.54041741E-02;
    COFD[2708] = -2.07861367E+01;
    COFD[2709] = 4.42680848E+00;
    COFD[2710] = -2.85885288E-01;
    COFD[2711] = 9.46483934E-03;
    COFD[2712] = -2.17740719E+01;
    COFD[2713] = 4.93496210E+00;
    COFD[2714] = -3.65981745E-01;
    COFD[2715] = 1.34912948E-02;
    COFD[2716] = -2.19075860E+01;
    COFD[2717] = 4.99907484E+00;
    COFD[2718] = -3.76094627E-01;
    COFD[2719] = 1.40009262E-02;
    COFD[2720] = -2.12501716E+01;
    COFD[2721] = 4.70506024E+00;
    COFD[2722] = -3.29547212E-01;
    COFD[2723] = 1.16521630E-02;
    COFD[2724] = -2.09119213E+01;
    COFD[2725] = 4.48108132E+00;
    COFD[2726] = -2.94289899E-01;
    COFD[2727] = 9.88218297E-03;
    COFD[2728] = -2.13838498E+01;
    COFD[2729] = 4.61201872E+00;
    COFD[2730] = -3.14803338E-01;
    COFD[2731] = 1.09082984E-02;
    COFD[2732] = -2.13854464E+01;
    COFD[2733] = 4.61201872E+00;
    COFD[2734] = -3.14803338E-01;
    COFD[2735] = 1.09082984E-02;
    COFD[2736] = -2.04649069E+01;
    COFD[2737] = 5.18856872E+00;
    COFD[2738] = -4.50001829E-01;
    COFD[2739] = 1.91636142E-02;
    COFD[2740] = -1.93925667E+01;
    COFD[2741] = 4.98286777E+00;
    COFD[2742] = -4.26970814E-01;
    COFD[2743] = 1.83122917E-02;
    COFD[2744] = -1.59632479E+01;
    COFD[2745] = 4.07051484E+00;
    COFD[2746] = -3.16303109E-01;
    COFD[2747] = 1.38259377E-02;
    COFD[2748] = -1.96914944E+01;
    COFD[2749] = 5.54637286E+00;
    COFD[2750] = -4.87070324E-01;
    COFD[2751] = 2.03983467E-02;
    COFD[2752] = -1.94151822E+01;
    COFD[2753] = 4.98286777E+00;
    COFD[2754] = -4.26970814E-01;
    COFD[2755] = 1.83122917E-02;
    COFD[2756] = -1.80862867E+01;
    COFD[2757] = 3.69199168E+00;
    COFD[2758] = -1.74005516E-01;
    COFD[2759] = 3.97694372E-03;
    COFD[2760] = -2.08463209E+01;
    COFD[2761] = 5.32244593E+00;
    COFD[2762] = -4.65829403E-01;
    COFD[2763] = 1.97895274E-02;
    COFD[2764] = -2.08554914E+01;
    COFD[2765] = 5.32244593E+00;
    COFD[2766] = -4.65829403E-01;
    COFD[2767] = 1.97895274E-02;
    COFD[2768] = -2.08642748E+01;
    COFD[2769] = 5.32244593E+00;
    COFD[2770] = -4.65829403E-01;
    COFD[2771] = 1.97895274E-02;
    COFD[2772] = -2.21083035E+01;
    COFD[2773] = 5.48540187E+00;
    COFD[2774] = -4.58962148E-01;
    COFD[2775] = 1.83770355E-02;
    COFD[2776] = -2.04949373E+01;
    COFD[2777] = 5.19614628E+00;
    COFD[2778] = -4.50889164E-01;
    COFD[2779] = 1.91983328E-02;
    COFD[2780] = -1.94051843E+01;
    COFD[2781] = 4.10954793E+00;
    COFD[2782] = -2.37523329E-01;
    COFD[2783] = 7.08858141E-03;
    COFD[2784] = -2.14314090E+01;
    COFD[2785] = 5.54007827E+00;
    COFD[2786] = -4.86434511E-01;
    COFD[2787] = 2.03779006E-02;
    COFD[2788] = -2.13881945E+01;
    COFD[2789] = 5.52422470E+00;
    COFD[2790] = -4.84872944E-01;
    COFD[2791] = 2.03298213E-02;
    COFD[2792] = -1.96653154E+01;
    COFD[2793] = 4.22062499E+00;
    COFD[2794] = -2.54326872E-01;
    COFD[2795] = 7.91017784E-03;
    COFD[2796] = -2.21333822E+01;
    COFD[2797] = 5.47072190E+00;
    COFD[2798] = -4.56301261E-01;
    COFD[2799] = 1.82313566E-02;
    COFD[2800] = -2.04033972E+01;
    COFD[2801] = 4.50250781E+00;
    COFD[2802] = -2.97622106E-01;
    COFD[2803] = 1.00481473E-02;
    COFD[2804] = -2.04268153E+01;
    COFD[2805] = 4.50250781E+00;
    COFD[2806] = -2.97622106E-01;
    COFD[2807] = 1.00481473E-02;
    COFD[2808] = -1.97704178E+01;
    COFD[2809] = 4.22062499E+00;
    COFD[2810] = -2.54326872E-01;
    COFD[2811] = 7.91017784E-03;
    COFD[2812] = -2.18318278E+01;
    COFD[2813] = 5.40298848E+00;
    COFD[2814] = -4.43954594E-01;
    COFD[2815] = 1.75542998E-02;
    COFD[2816] = -2.17934580E+01;
    COFD[2817] = 5.21869603E+00;
    COFD[2818] = -4.12084772E-01;
    COFD[2819] = 1.58573035E-02;
    COFD[2820] = -2.19162360E+01;
    COFD[2821] = 5.49906960E+00;
    COFD[2822] = -4.61793001E-01;
    COFD[2823] = 1.85415189E-02;
    COFD[2824] = -2.16722314E+01;
    COFD[2825] = 5.13708607E+00;
    COFD[2826] = -3.98445708E-01;
    COFD[2827] = 1.51455626E-02;
    COFD[2828] = -2.16791513E+01;
    COFD[2829] = 5.13708607E+00;
    COFD[2830] = -3.98445708E-01;
    COFD[2831] = 1.51455626E-02;
    COFD[2832] = -2.17407419E+01;
    COFD[2833] = 5.17945041E+00;
    COFD[2834] = -4.05514689E-01;
    COFD[2835] = 1.55141412E-02;
    COFD[2836] = -2.19919464E+01;
    COFD[2837] = 5.23595129E+00;
    COFD[2838] = -4.15079064E-01;
    COFD[2839] = 1.60168286E-02;
    COFD[2840] = -2.02246117E+01;
    COFD[2841] = 4.22278378E+00;
    COFD[2842] = -2.54653500E-01;
    COFD[2843] = 7.92616085E-03;
    COFD[2844] = -2.15102238E+01;
    COFD[2845] = 4.94878244E+00;
    COFD[2846] = -3.68158605E-01;
    COFD[2847] = 1.36008797E-02;
    COFD[2848] = -2.15236645E+01;
    COFD[2849] = 4.96219227E+00;
    COFD[2850] = -3.70270843E-01;
    COFD[2851] = 1.37072211E-02;
    COFD[2852] = -2.15278182E+01;
    COFD[2853] = 4.96219227E+00;
    COFD[2854] = -3.70270843E-01;
    COFD[2855] = 1.37072211E-02;
    COFD[2856] = -2.17094710E+01;
    COFD[2857] = 4.98271982E+00;
    COFD[2858] = -3.73502341E-01;
    COFD[2859] = 1.38698700E-02;
    COFD[2860] = -2.00940426E+01;
    COFD[2861] = 4.12245214E+00;
    COFD[2862] = -2.39476227E-01;
    COFD[2863] = 7.18400558E-03;
    COFD[2864] = -2.12888403E+01;
    COFD[2865] = 4.71644372E+00;
    COFD[2866] = -3.31349990E-01;
    COFD[2867] = 1.17430818E-02;
    COFD[2868] = -2.14039230E+01;
    COFD[2869] = 4.77238689E+00;
    COFD[2870] = -3.40265855E-01;
    COFD[2871] = 1.21942137E-02;
    COFD[2872] = -2.07343778E+01;
    COFD[2873] = 4.47491202E+00;
    COFD[2874] = -2.93331059E-01;
    COFD[2875] = 9.83445305E-03;
    COFD[2876] = -2.02563322E+01;
    COFD[2877] = 4.19160608E+00;
    COFD[2878] = -2.49936771E-01;
    COFD[2879] = 7.69538319E-03;
    COFD[2880] = -2.08236367E+01;
    COFD[2881] = 4.35920123E+00;
    COFD[2882] = -2.75491273E-01;
    COFD[2883] = 8.95100289E-03;
    COFD[2884] = -2.08252570E+01;
    COFD[2885] = 4.35920123E+00;
    COFD[2886] = -2.75491273E-01;
    COFD[2887] = 8.95100289E-03;
    COFD[2888] = -1.83039618E+01;
    COFD[2889] = 4.47952077E+00;
    COFD[2890] = -3.66569471E-01;
    COFD[2891] = 1.58916129E-02;
    COFD[2892] = -1.72286007E+01;
    COFD[2893] = 4.24084025E+00;
    COFD[2894] = -3.37428619E-01;
    COFD[2895] = 1.47032793E-02;
    COFD[2896] = -1.39315266E+01;
    COFD[2897] = 3.30394764E+00;
    COFD[2898] = -2.17920112E-01;
    COFD[2899] = 9.60284243E-03;
    COFD[2900] = -1.79310765E+01;
    COFD[2901] = 4.98037650E+00;
    COFD[2902] = -4.26676911E-01;
    COFD[2903] = 1.83007231E-02;
    COFD[2904] = -1.72473011E+01;
    COFD[2905] = 4.24084025E+00;
    COFD[2906] = -3.37428619E-01;
    COFD[2907] = 1.47032793E-02;
    COFD[2908] = -2.09565916E+01;
    COFD[2909] = 5.18380539E+00;
    COFD[2910] = -4.06234719E-01;
    COFD[2911] = 1.55515345E-02;
    COFD[2912] = -1.86507213E+01;
    COFD[2913] = 4.60874797E+00;
    COFD[2914] = -3.82368716E-01;
    COFD[2915] = 1.65370164E-02;
    COFD[2916] = -1.86576191E+01;
    COFD[2917] = 4.60874797E+00;
    COFD[2918] = -3.82368716E-01;
    COFD[2919] = 1.65370164E-02;
    COFD[2920] = -1.86641962E+01;
    COFD[2921] = 4.60874797E+00;
    COFD[2922] = -3.82368716E-01;
    COFD[2923] = 1.65370164E-02;
    COFD[2924] = -2.13961414E+01;
    COFD[2925] = 5.46685775E+00;
    COFD[2926] = -4.78665416E-01;
    COFD[2927] = 2.01093915E-02;
    COFD[2928] = -1.83296965E+01;
    COFD[2929] = 4.48570999E+00;
    COFD[2930] = -3.67301524E-01;
    COFD[2931] = 1.59204254E-02;
    COFD[2932] = -2.16798265E+01;
    COFD[2933] = 5.36811769E+00;
    COFD[2934] = -4.37727086E-01;
    COFD[2935] = 1.72167686E-02;
    COFD[2936] = -1.95770968E+01;
    COFD[2937] = 4.97133070E+00;
    COFD[2938] = -4.25604177E-01;
    COFD[2939] = 1.82582594E-02;
    COFD[2940] = -1.95154079E+01;
    COFD[2941] = 4.94787350E+00;
    COFD[2942] = -4.22829292E-01;
    COFD[2943] = 1.81487163E-02;
    COFD[2944] = -2.17547312E+01;
    COFD[2945] = 5.40298848E+00;
    COFD[2946] = -4.43954594E-01;
    COFD[2947] = 1.75542998E-02;
    COFD[2948] = -2.15206146E+01;
    COFD[2949] = 5.48426911E+00;
    COFD[2950] = -4.80606512E-01;
    COFD[2951] = 2.01811046E-02;
    COFD[2952] = -2.20262793E+01;
    COFD[2953] = 5.49663315E+00;
    COFD[2954] = -4.61182837E-01;
    COFD[2955] = 1.85035558E-02;
    COFD[2956] = -2.20431319E+01;
    COFD[2957] = 5.49663315E+00;
    COFD[2958] = -4.61182837E-01;
    COFD[2959] = 1.85035558E-02;
    COFD[2960] = -2.18318278E+01;
    COFD[2961] = 5.40298848E+00;
    COFD[2962] = -4.43954594E-01;
    COFD[2963] = 1.75542998E-02;
    COFD[2964] = -2.15453676E+01;
    COFD[2965] = 5.55313619E+00;
    COFD[2966] = -4.87753729E-01;
    COFD[2967] = 2.04203421E-02;
    COFD[2968] = -2.20228343E+01;
    COFD[2969] = 5.61211028E+00;
    COFD[2970] = -4.90893171E-01;
    COFD[2971] = 2.03793118E-02;
    COFD[2972] = -2.11427744E+01;
    COFD[2973] = 5.43893233E+00;
    COFD[2974] = -4.75546039E-01;
    COFD[2975] = 1.99938690E-02;
    COFD[2976] = -2.20555979E+01;
    COFD[2977] = 5.59649805E+00;
    COFD[2978] = -4.86750336E-01;
    COFD[2979] = 2.01151498E-02;
    COFD[2980] = -2.20606550E+01;
    COFD[2981] = 5.59649805E+00;
    COFD[2982] = -4.86750336E-01;
    COFD[2983] = 2.01151498E-02;
    COFD[2984] = -2.20511271E+01;
    COFD[2985] = 5.60809037E+00;
    COFD[2986] = -4.89400803E-01;
    COFD[2987] = 2.02760802E-02;
    COFD[2988] = -2.21795362E+01;
    COFD[2989] = 5.61233637E+00;
    COFD[2990] = -4.91419253E-01;
    COFD[2991] = 2.04216738E-02;
    COFD[2992] = -2.22462130E+01;
    COFD[2993] = 5.40356304E+00;
    COFD[2994] = -4.44060256E-01;
    COFD[2995] = 1.75601121E-02;
    COFD[2996] = -2.22801170E+01;
    COFD[2997] = 5.58507108E+00;
    COFD[2998] = -4.81395065E-01;
    COFD[2999] = 1.97276199E-02;
    COFD[3000] = -2.22609256E+01;
    COFD[3001] = 5.58490856E+00;
    COFD[3002] = -4.81588720E-01;
    COFD[3003] = 1.97445317E-02;
    COFD[3004] = -2.22638165E+01;
    COFD[3005] = 5.58490856E+00;
    COFD[3006] = -4.81588720E-01;
    COFD[3007] = 1.97445317E-02;
    COFD[3008] = -2.23950513E+01;
    COFD[3009] = 5.58492366E+00;
    COFD[3010] = -4.81921868E-01;
    COFD[3011] = 1.97721534E-02;
    COFD[3012] = -2.22709427E+01;
    COFD[3013] = 5.37360713E+00;
    COFD[3014] = -4.38661889E-01;
    COFD[3015] = 1.72661628E-02;
    COFD[3016] = -2.24990717E+01;
    COFD[3017] = 5.55026833E+00;
    COFD[3018] = -4.72437808E-01;
    COFD[3019] = 1.91625195E-02;
    COFD[3020] = -2.25347527E+01;
    COFD[3021] = 5.57238332E+00;
    COFD[3022] = -4.76605097E-01;
    COFD[3023] = 1.93951822E-02;
    COFD[3024] = -2.23655523E+01;
    COFD[3025] = 5.48956505E+00;
    COFD[3026] = -4.59770566E-01;
    COFD[3027] = 1.84227929E-02;
    COFD[3028] = -2.23265991E+01;
    COFD[3029] = 5.39645154E+00;
    COFD[3030] = -4.42708323E-01;
    COFD[3031] = 1.74846134E-02;
    COFD[3032] = -2.26089431E+01;
    COFD[3033] = 5.44867280E+00;
    COFD[3034] = -4.52284883E-01;
    COFD[3035] = 1.80110706E-02;
    COFD[3036] = -2.26099899E+01;
    COFD[3037] = 5.44867280E+00;
    COFD[3038] = -4.52284883E-01;
    COFD[3039] = 1.80110706E-02;
    COFD[3040] = -1.90859283E+01;
    COFD[3041] = 4.68079396E+00;
    COFD[3042] = -3.91231550E-01;
    COFD[3043] = 1.69021170E-02;
    COFD[3044] = -1.79361160E+01;
    COFD[3045] = 4.42139452E+00;
    COFD[3046] = -3.59567329E-01;
    COFD[3047] = 1.56103969E-02;
    COFD[3048] = -1.45715797E+01;
    COFD[3049] = 3.49477850E+00;
    COFD[3050] = -2.42635772E-01;
    COFD[3051] = 1.06721490E-02;
    COFD[3052] = -1.85748546E+01;
    COFD[3053] = 5.14789919E+00;
    COFD[3054] = -4.45930850E-01;
    COFD[3055] = 1.90363341E-02;
    COFD[3056] = -1.79580609E+01;
    COFD[3057] = 4.42139452E+00;
    COFD[3058] = -3.59567329E-01;
    COFD[3059] = 1.56103969E-02;
    COFD[3060] = -2.06310304E+01;
    COFD[3061] = 4.89289496E+00;
    COFD[3062] = -3.59346263E-01;
    COFD[3063] = 1.31570901E-02;
    COFD[3064] = -1.93917298E+01;
    COFD[3065] = 4.78708023E+00;
    COFD[3066] = -4.03693144E-01;
    COFD[3067] = 1.73884817E-02;
    COFD[3068] = -1.94004795E+01;
    COFD[3069] = 4.78708023E+00;
    COFD[3070] = -4.03693144E-01;
    COFD[3071] = 1.73884817E-02;
    COFD[3072] = -1.94088529E+01;
    COFD[3073] = 4.78708023E+00;
    COFD[3074] = -4.03693144E-01;
    COFD[3075] = 1.73884817E-02;
    COFD[3076] = -2.20725883E+01;
    COFD[3077] = 5.59642965E+00;
    COFD[3078] = -4.91577716E-01;
    COFD[3079] = 2.05159582E-02;
    COFD[3080] = -1.91118445E+01;
    COFD[3081] = 4.68715685E+00;
    COFD[3082] = -3.91979493E-01;
    COFD[3083] = 1.69314004E-02;
    COFD[3084] = -2.15802788E+01;
    COFD[3085] = 5.16868516E+00;
    COFD[3086] = -4.03721581E-01;
    COFD[3087] = 1.54206640E-02;
    COFD[3088] = -2.02692384E+01;
    COFD[3089] = 5.14418672E+00;
    COFD[3090] = -4.45631004E-01;
    COFD[3091] = 1.90308403E-02;
    COFD[3092] = -2.02318658E+01;
    COFD[3093] = 5.12963391E+00;
    COFD[3094] = -4.44146826E-01;
    COFD[3095] = 1.89829640E-02;
    COFD[3096] = -2.16936515E+01;
    COFD[3097] = 5.21869603E+00;
    COFD[3098] = -4.12084772E-01;
    COFD[3099] = 1.58573035E-02;
    COFD[3100] = -2.21343023E+01;
    COFD[3101] = 5.60010742E+00;
    COFD[3102] = -4.91597429E-01;
    COFD[3103] = 2.04987718E-02;
    COFD[3104] = -2.20597305E+01;
    COFD[3105] = 5.34774760E+00;
    COFD[3106] = -4.34239753E-01;
    COFD[3107] = 1.70320676E-02;
    COFD[3108] = -2.20818886E+01;
    COFD[3109] = 5.34774760E+00;
    COFD[3110] = -4.34239753E-01;
    COFD[3111] = 1.70320676E-02;
    COFD[3112] = -2.17934580E+01;
    COFD[3113] = 5.21869603E+00;
    COFD[3114] = -4.12084772E-01;
    COFD[3115] = 1.58573035E-02;
    COFD[3116] = -2.20228343E+01;
    COFD[3117] = 5.61211028E+00;
    COFD[3118] = -4.90893171E-01;
    COFD[3119] = 2.03793118E-02;
    COFD[3120] = -2.23318349E+01;
    COFD[3121] = 5.58508387E+00;
    COFD[3122] = -4.81385216E-01;
    COFD[3123] = 1.97267369E-02;
    COFD[3124] = -2.18222696E+01;
    COFD[3125] = 5.57940140E+00;
    COFD[3126] = -4.89964112E-01;
    COFD[3127] = 2.04689539E-02;
    COFD[3128] = -2.23931168E+01;
    COFD[3129] = 5.58325398E+00;
    COFD[3130] = -4.79084067E-01;
    COFD[3131] = 1.95452935E-02;
    COFD[3132] = -2.23996837E+01;
    COFD[3133] = 5.58325398E+00;
    COFD[3134] = -4.79084067E-01;
    COFD[3135] = 1.95452935E-02;
    COFD[3136] = -2.23689627E+01;
    COFD[3137] = 5.58513878E+00;
    COFD[3138] = -4.80389524E-01;
    COFD[3139] = 1.96438689E-02;
    COFD[3140] = -2.24797372E+01;
    COFD[3141] = 5.58492389E+00;
    COFD[3142] = -4.81921515E-01;
    COFD[3143] = 1.97721229E-02;
    COFD[3144] = -2.22169882E+01;
    COFD[3145] = 5.21950983E+00;
    COFD[3146] = -4.12223195E-01;
    COFD[3147] = 1.58645894E-02;
    COFD[3148] = -2.25041734E+01;
    COFD[3149] = 5.51797622E+00;
    COFD[3150] = -4.66229499E-01;
    COFD[3151] = 1.88128348E-02;
    COFD[3152] = -2.24965286E+01;
    COFD[3153] = 5.52198915E+00;
    COFD[3154] = -4.67014474E-01;
    COFD[3155] = 1.88574253E-02;
    COFD[3156] = -2.25004333E+01;
    COFD[3157] = 5.52198915E+00;
    COFD[3158] = -4.67014474E-01;
    COFD[3159] = 1.88574253E-02;
    COFD[3160] = -2.26411013E+01;
    COFD[3161] = 5.52830072E+00;
    COFD[3162] = -4.68235018E-01;
    COFD[3163] = 1.89263933E-02;
    COFD[3164] = -2.22139496E+01;
    COFD[3165] = 5.17488844E+00;
    COFD[3166] = -4.04758505E-01;
    COFD[3167] = 1.54748177E-02;
    COFD[3168] = -2.26485311E+01;
    COFD[3169] = 5.44696782E+00;
    COFD[3170] = -4.51976837E-01;
    COFD[3171] = 1.79942461E-02;
    COFD[3172] = -2.26946865E+01;
    COFD[3173] = 5.47392239E+00;
    COFD[3174] = -4.56882004E-01;
    COFD[3175] = 1.82631638E-02;
    COFD[3176] = -2.23996701E+01;
    COFD[3177] = 5.33372666E+00;
    COFD[3178] = -4.31837946E-01;
    COFD[3179] = 1.69048117E-02;
    COFD[3180] = -2.22885235E+01;
    COFD[3181] = 5.20764658E+00;
    COFD[3182] = -4.10207913E-01;
    COFD[3183] = 1.57585882E-02;
    COFD[3184] = -2.26029886E+01;
    COFD[3185] = 5.27383847E+00;
    COFD[3186] = -4.21722368E-01;
    COFD[3187] = 1.63729618E-02;
    COFD[3188] = -2.26044889E+01;
    COFD[3189] = 5.27383847E+00;
    COFD[3190] = -4.21722368E-01;
    COFD[3191] = 1.63729618E-02;
    COFD[3192] = -1.78815889E+01;
    COFD[3193] = 4.34347890E+00;
    COFD[3194] = -3.49890003E-01;
    COFD[3195] = 1.52083459E-02;
    COFD[3196] = -1.68343393E+01;
    COFD[3197] = 4.11954900E+00;
    COFD[3198] = -3.22470391E-01;
    COFD[3199] = 1.40859564E-02;
    COFD[3200] = -1.36336373E+01;
    COFD[3201] = 3.22088176E+00;
    COFD[3202] = -2.07623790E-01;
    COFD[3203] = 9.17771542E-03;
    COFD[3204] = -1.74407963E+01;
    COFD[3205] = 4.83580036E+00;
    COFD[3206] = -4.09383573E-01;
    COFD[3207] = 1.76098175E-02;
    COFD[3208] = -1.68535757E+01;
    COFD[3209] = 4.11954900E+00;
    COFD[3210] = -3.22470391E-01;
    COFD[3211] = 1.40859564E-02;
    COFD[3212] = -2.11309197E+01;
    COFD[3213] = 5.32644193E+00;
    COFD[3214] = -4.30581064E-01;
    COFD[3215] = 1.68379725E-02;
    COFD[3216] = -1.82145353E+01;
    COFD[3217] = 4.46848269E+00;
    COFD[3218] = -3.65269718E-01;
    COFD[3219] = 1.58407652E-02;
    COFD[3220] = -1.82217198E+01;
    COFD[3221] = 4.46848269E+00;
    COFD[3222] = -3.65269718E-01;
    COFD[3223] = 1.58407652E-02;
    COFD[3224] = -1.82285740E+01;
    COFD[3225] = 4.46848269E+00;
    COFD[3226] = -3.65269718E-01;
    COFD[3227] = 1.58407652E-02;
    COFD[3228] = -2.11031143E+01;
    COFD[3229] = 5.39439999E+00;
    COFD[3230] = -4.72050184E-01;
    COFD[3231] = 1.99336257E-02;
    COFD[3232] = -1.79116531E+01;
    COFD[3233] = 4.35148286E+00;
    COFD[3234] = -3.50886647E-01;
    COFD[3235] = 1.52498573E-02;
    COFD[3236] = -2.17855148E+01;
    COFD[3237] = 5.47519298E+00;
    COFD[3238] = -4.57113040E-01;
    COFD[3239] = 1.82758312E-02;
    COFD[3240] = -1.91225414E+01;
    COFD[3241] = 4.82869066E+00;
    COFD[3242] = -4.08564514E-01;
    COFD[3243] = 1.75784675E-02;
    COFD[3244] = -1.90692595E+01;
    COFD[3245] = 4.80830699E+00;
    COFD[3246] = -4.06171933E-01;
    COFD[3247] = 1.74848791E-02;
    COFD[3248] = -2.18356866E+01;
    COFD[3249] = 5.49906960E+00;
    COFD[3250] = -4.61793001E-01;
    COFD[3251] = 1.85415189E-02;
    COFD[3252] = -2.12014186E+01;
    COFD[3253] = 5.40060531E+00;
    COFD[3254] = -4.72449699E-01;
    COFD[3255] = 1.99345817E-02;
    COFD[3256] = -2.20398328E+01;
    COFD[3257] = 5.56049839E+00;
    COFD[3258] = -4.74367872E-01;
    COFD[3259] = 1.92702787E-02;
    COFD[3260] = -2.20574820E+01;
    COFD[3261] = 5.56049839E+00;
    COFD[3262] = -4.74367872E-01;
    COFD[3263] = 1.92702787E-02;
    COFD[3264] = -2.19162360E+01;
    COFD[3265] = 5.49906960E+00;
    COFD[3266] = -4.61793001E-01;
    COFD[3267] = 1.85415189E-02;
    COFD[3268] = -2.11427744E+01;
    COFD[3269] = 5.43893233E+00;
    COFD[3270] = -4.75546039E-01;
    COFD[3271] = 1.99938690E-02;
    COFD[3272] = -2.18222696E+01;
    COFD[3273] = 5.57940140E+00;
    COFD[3274] = -4.89964112E-01;
    COFD[3275] = 2.04689539E-02;
    COFD[3276] = -2.08820897E+01;
    COFD[3277] = 5.38250415E+00;
    COFD[3278] = -4.71144140E-01;
    COFD[3279] = 1.99199779E-02;
    COFD[3280] = -2.19448434E+01;
    COFD[3281] = 5.60255148E+00;
    COFD[3282] = -4.91366572E-01;
    COFD[3283] = 2.04670553E-02;
    COFD[3284] = -2.19501296E+01;
    COFD[3285] = 5.60255148E+00;
    COFD[3286] = -4.91366572E-01;
    COFD[3287] = 2.04670553E-02;
    COFD[3288] = -2.19032561E+01;
    COFD[3289] = 5.59794138E+00;
    COFD[3290] = -4.91684532E-01;
    COFD[3291] = 2.05170953E-02;
    COFD[3292] = -2.19617258E+01;
    COFD[3293] = 5.57026255E+00;
    COFD[3294] = -4.89178491E-01;
    COFD[3295] = 2.04505218E-02;
    COFD[3296] = -2.23434237E+01;
    COFD[3297] = 5.49927389E+00;
    COFD[3298] = -4.61845436E-01;
    COFD[3299] = 1.85448066E-02;
    COFD[3300] = -2.21913393E+01;
    COFD[3301] = 5.60175327E+00;
    COFD[3302] = -4.87953216E-01;
    COFD[3303] = 2.01882171E-02;
    COFD[3304] = -2.21792065E+01;
    COFD[3305] = 5.60465338E+00;
    COFD[3306] = -4.88572478E-01;
    COFD[3307] = 2.02248525E-02;
    COFD[3308] = -2.21822461E+01;
    COFD[3309] = 5.60465338E+00;
    COFD[3310] = -4.88572478E-01;
    COFD[3311] = 2.02248525E-02;
    COFD[3312] = -2.23250359E+01;
    COFD[3313] = 5.60776666E+00;
    COFD[3314] = -4.89319792E-01;
    COFD[3315] = 2.02710069E-02;
    COFD[3316] = -2.23935500E+01;
    COFD[3317] = 5.47922490E+00;
    COFD[3318] = -4.57847893E-01;
    COFD[3319] = 1.83161707E-02;
    COFD[3320] = -2.24603310E+01;
    COFD[3321] = 5.58501539E+00;
    COFD[3322] = -4.81433860E-01;
    COFD[3323] = 1.97311245E-02;
    COFD[3324] = -2.24423544E+01;
    COFD[3325] = 5.58416166E+00;
    COFD[3326] = -4.82369720E-01;
    COFD[3327] = 1.98133127E-02;
    COFD[3328] = -2.23867276E+01;
    COFD[3329] = 5.55175851E+00;
    COFD[3330] = -4.72720598E-01;
    COFD[3331] = 1.91783487E-02;
    COFD[3332] = -2.24356779E+01;
    COFD[3333] = 5.49613266E+00;
    COFD[3334] = -4.61060586E-01;
    COFD[3335] = 1.84960110E-02;
    COFD[3336] = -2.26579938E+01;
    COFD[3337] = 5.52001624E+00;
    COFD[3338] = -4.66629503E-01;
    COFD[3339] = 1.88355817E-02;
    COFD[3340] = -2.26591038E+01;
    COFD[3341] = 5.52001624E+00;
    COFD[3342] = -4.66629503E-01;
    COFD[3343] = 1.88355817E-02;
    COFD[3344] = -1.92731067E+01;
    COFD[3345] = 4.73660584E+00;
    COFD[3346] = -3.97704978E-01;
    COFD[3347] = 1.71514887E-02;
    COFD[3348] = -1.81463104E+01;
    COFD[3349] = 4.48398491E+00;
    COFD[3350] = -3.67097129E-01;
    COFD[3351] = 1.59123634E-02;
    COFD[3352] = -1.47719516E+01;
    COFD[3353] = 3.55444478E+00;
    COFD[3354] = -2.50272707E-01;
    COFD[3355] = 1.09990787E-02;
    COFD[3356] = -1.87644697E+01;
    COFD[3357] = 5.19146813E+00;
    COFD[3358] = -4.50340408E-01;
    COFD[3359] = 1.91768178E-02;
    COFD[3360] = -1.81677871E+01;
    COFD[3361] = 4.48398491E+00;
    COFD[3362] = -3.67097129E-01;
    COFD[3363] = 1.59123634E-02;
    COFD[3364] = -2.04357586E+01;
    COFD[3365] = 4.77398686E+00;
    COFD[3366] = -3.40522956E-01;
    COFD[3367] = 1.22072846E-02;
    COFD[3368] = -1.95819005E+01;
    COFD[3369] = 4.84393038E+00;
    COFD[3370] = -4.10274737E-01;
    COFD[3371] = 1.76417458E-02;
    COFD[3372] = -1.95903647E+01;
    COFD[3373] = 4.84393038E+00;
    COFD[3374] = -4.10274737E-01;
    COFD[3375] = 1.76417458E-02;
    COFD[3376] = -1.95984602E+01;
    COFD[3377] = 4.84393038E+00;
    COFD[3378] = -4.10274737E-01;
    COFD[3379] = 1.76417458E-02;
    COFD[3380] = -2.21630311E+01;
    COFD[3381] = 5.60807471E+00;
    COFD[3382] = -4.91339309E-01;
    COFD[3383] = 2.04365761E-02;
    COFD[3384] = -1.93011401E+01;
    COFD[3385] = 4.74387793E+00;
    COFD[3386] = -3.98574972E-01;
    COFD[3387] = 1.71862289E-02;
    COFD[3388] = -2.14398182E+01;
    COFD[3389] = 5.07680397E+00;
    COFD[3390] = -3.88612087E-01;
    COFD[3391] = 1.46395101E-02;
    COFD[3392] = -2.04274471E+01;
    COFD[3393] = 5.18271974E+00;
    COFD[3394] = -4.49323627E-01;
    COFD[3395] = 1.91373940E-02;
    COFD[3396] = -2.03738891E+01;
    COFD[3397] = 5.16159436E+00;
    COFD[3398] = -4.46935283E-01;
    COFD[3399] = 1.90480297E-02;
    COFD[3400] = -2.15759895E+01;
    COFD[3401] = 5.13708607E+00;
    COFD[3402] = -3.98445708E-01;
    COFD[3403] = 1.51455626E-02;
    COFD[3404] = -2.22262162E+01;
    COFD[3405] = 5.61211818E+00;
    COFD[3406] = -4.91432482E-01;
    COFD[3407] = 2.04238731E-02;
    COFD[3408] = -2.19526490E+01;
    COFD[3409] = 5.27258289E+00;
    COFD[3410] = -4.21502790E-01;
    COFD[3411] = 1.63611949E-02;
    COFD[3412] = -2.19739638E+01;
    COFD[3413] = 5.27258289E+00;
    COFD[3414] = -4.21502790E-01;
    COFD[3415] = 1.63611949E-02;
    COFD[3416] = -2.16722314E+01;
    COFD[3417] = 5.13708607E+00;
    COFD[3418] = -3.98445708E-01;
    COFD[3419] = 1.51455626E-02;
    COFD[3420] = -2.20555979E+01;
    COFD[3421] = 5.59649805E+00;
    COFD[3422] = -4.86750336E-01;
    COFD[3423] = 2.01151498E-02;
    COFD[3424] = -2.23931168E+01;
    COFD[3425] = 5.58325398E+00;
    COFD[3426] = -4.79084067E-01;
    COFD[3427] = 1.95452935E-02;
    COFD[3428] = -2.19448434E+01;
    COFD[3429] = 5.60255148E+00;
    COFD[3430] = -4.91366572E-01;
    COFD[3431] = 2.04670553E-02;
    COFD[3432] = -2.23787998E+01;
    COFD[3433] = 5.54890339E+00;
    COFD[3434] = -4.72166228E-01;
    COFD[3435] = 1.91470071E-02;
    COFD[3436] = -2.23851292E+01;
    COFD[3437] = 5.54890339E+00;
    COFD[3438] = -4.72166228E-01;
    COFD[3439] = 1.91470071E-02;
    COFD[3440] = -2.24018266E+01;
    COFD[3441] = 5.57115285E+00;
    COFD[3442] = -4.76363416E-01;
    COFD[3443] = 1.93814080E-02;
    COFD[3444] = -2.25424404E+01;
    COFD[3445] = 5.58482894E+00;
    COFD[3446] = -4.79850522E-01;
    COFD[3447] = 1.96007690E-02;
    COFD[3448] = -2.20862991E+01;
    COFD[3449] = 5.13809011E+00;
    COFD[3450] = -3.98612308E-01;
    COFD[3451] = 1.51542189E-02;
    COFD[3452] = -2.25145418E+01;
    COFD[3453] = 5.49554403E+00;
    COFD[3454] = -4.60936491E-01;
    COFD[3455] = 1.84887572E-02;
    COFD[3456] = -2.25028406E+01;
    COFD[3457] = 5.49776513E+00;
    COFD[3458] = -4.61463030E-01;
    COFD[3459] = 1.85209236E-02;
    COFD[3460] = -2.25065805E+01;
    COFD[3461] = 5.49776513E+00;
    COFD[3462] = -4.61463030E-01;
    COFD[3463] = 1.85209236E-02;
    COFD[3464] = -2.26360108E+01;
    COFD[3465] = 5.50023958E+00;
    COFD[3466] = -4.62136179E-01;
    COFD[3467] = 1.85639061E-02;
    COFD[3468] = -2.20599362E+01;
    COFD[3469] = 5.08417640E+00;
    COFD[3470] = -3.89810534E-01;
    COFD[3471] = 1.47010214E-02;
    COFD[3472] = -2.25891024E+01;
    COFD[3473] = 5.39655717E+00;
    COFD[3474] = -4.42728390E-01;
    COFD[3475] = 1.74857336E-02;
    COFD[3476] = -2.26273108E+01;
    COFD[3477] = 5.42002683E+00;
    COFD[3478] = -4.47111163E-01;
    COFD[3479] = 1.77287360E-02;
    COFD[3480] = -2.22858832E+01;
    COFD[3481] = 5.25941804E+00;
    COFD[3482] = -4.19208672E-01;
    COFD[3483] = 1.62385114E-02;
    COFD[3484] = -2.21494624E+01;
    COFD[3485] = 5.12338366E+00;
    COFD[3486] = -3.96176894E-01;
    COFD[3487] = 1.50278196E-02;
    COFD[3488] = -2.25069737E+01;
    COFD[3489] = 5.21003123E+00;
    COFD[3490] = -4.10612564E-01;
    COFD[3491] = 1.57798598E-02;
    COFD[3492] = -2.25083966E+01;
    COFD[3493] = 5.21003123E+00;
    COFD[3494] = -4.10612564E-01;
    COFD[3495] = 1.57798598E-02;
    COFD[3496] = -1.92783884E+01;
    COFD[3497] = 4.73660584E+00;
    COFD[3498] = -3.97704978E-01;
    COFD[3499] = 1.71514887E-02;
    COFD[3500] = -1.81499793E+01;
    COFD[3501] = 4.48398491E+00;
    COFD[3502] = -3.67097129E-01;
    COFD[3503] = 1.59123634E-02;
    COFD[3504] = -1.47725694E+01;
    COFD[3505] = 3.55444478E+00;
    COFD[3506] = -2.50272707E-01;
    COFD[3507] = 1.09990787E-02;
    COFD[3508] = -1.87647862E+01;
    COFD[3509] = 5.19146813E+00;
    COFD[3510] = -4.50340408E-01;
    COFD[3511] = 1.91768178E-02;
    COFD[3512] = -1.81716176E+01;
    COFD[3513] = 4.48398491E+00;
    COFD[3514] = -3.67097129E-01;
    COFD[3515] = 1.59123634E-02;
    COFD[3516] = -2.04397451E+01;
    COFD[3517] = 4.77398686E+00;
    COFD[3518] = -3.40522956E-01;
    COFD[3519] = 1.22072846E-02;
    COFD[3520] = -1.95875976E+01;
    COFD[3521] = 4.84393038E+00;
    COFD[3522] = -4.10274737E-01;
    COFD[3523] = 1.76417458E-02;
    COFD[3524] = -1.95961596E+01;
    COFD[3525] = 4.84393038E+00;
    COFD[3526] = -4.10274737E-01;
    COFD[3527] = 1.76417458E-02;
    COFD[3528] = -1.96043503E+01;
    COFD[3529] = 4.84393038E+00;
    COFD[3530] = -4.10274737E-01;
    COFD[3531] = 1.76417458E-02;
    COFD[3532] = -2.21697404E+01;
    COFD[3533] = 5.60807471E+00;
    COFD[3534] = -4.91339309E-01;
    COFD[3535] = 2.04365761E-02;
    COFD[3536] = -1.93064215E+01;
    COFD[3537] = 4.74387793E+00;
    COFD[3538] = -3.98574972E-01;
    COFD[3539] = 1.71862289E-02;
    COFD[3540] = -2.14453157E+01;
    COFD[3541] = 5.07680397E+00;
    COFD[3542] = -3.88612087E-01;
    COFD[3543] = 1.46395101E-02;
    COFD[3544] = -2.04309557E+01;
    COFD[3545] = 5.18271974E+00;
    COFD[3546] = -4.49323627E-01;
    COFD[3547] = 1.91373940E-02;
    COFD[3548] = -2.03775651E+01;
    COFD[3549] = 5.16159436E+00;
    COFD[3550] = -4.46935283E-01;
    COFD[3551] = 1.90480297E-02;
    COFD[3552] = -2.15816909E+01;
    COFD[3553] = 5.13708607E+00;
    COFD[3554] = -3.98445708E-01;
    COFD[3555] = 1.51455626E-02;
    COFD[3556] = -2.22317182E+01;
    COFD[3557] = 5.61211818E+00;
    COFD[3558] = -4.91432482E-01;
    COFD[3559] = 2.04238731E-02;
    COFD[3560] = -2.19592125E+01;
    COFD[3561] = 5.27258289E+00;
    COFD[3562] = -4.21502790E-01;
    COFD[3563] = 1.63611949E-02;
    COFD[3564] = -2.19808152E+01;
    COFD[3565] = 5.27258289E+00;
    COFD[3566] = -4.21502790E-01;
    COFD[3567] = 1.63611949E-02;
    COFD[3568] = -2.16791513E+01;
    COFD[3569] = 5.13708607E+00;
    COFD[3570] = -3.98445708E-01;
    COFD[3571] = 1.51455626E-02;
    COFD[3572] = -2.20606550E+01;
    COFD[3573] = 5.59649805E+00;
    COFD[3574] = -4.86750336E-01;
    COFD[3575] = 2.01151498E-02;
    COFD[3576] = -2.23996837E+01;
    COFD[3577] = 5.58325398E+00;
    COFD[3578] = -4.79084067E-01;
    COFD[3579] = 1.95452935E-02;
    COFD[3580] = -2.19501296E+01;
    COFD[3581] = 5.60255148E+00;
    COFD[3582] = -4.91366572E-01;
    COFD[3583] = 2.04670553E-02;
    COFD[3584] = -2.23851292E+01;
    COFD[3585] = 5.54890339E+00;
    COFD[3586] = -4.72166228E-01;
    COFD[3587] = 1.91470071E-02;
    COFD[3588] = -2.23915398E+01;
    COFD[3589] = 5.54890339E+00;
    COFD[3590] = -4.72166228E-01;
    COFD[3591] = 1.91470071E-02;
    COFD[3592] = -2.24083163E+01;
    COFD[3593] = 5.57115285E+00;
    COFD[3594] = -4.76363416E-01;
    COFD[3595] = 1.93814080E-02;
    COFD[3596] = -2.25490826E+01;
    COFD[3597] = 5.58482894E+00;
    COFD[3598] = -4.79850522E-01;
    COFD[3599] = 1.96007690E-02;
    COFD[3600] = -2.20946432E+01;
    COFD[3601] = 5.13809011E+00;
    COFD[3602] = -3.98612308E-01;
    COFD[3603] = 1.51542189E-02;
    COFD[3604] = -2.25219004E+01;
    COFD[3605] = 5.49554403E+00;
    COFD[3606] = -4.60936491E-01;
    COFD[3607] = 1.84887572E-02;
    COFD[3608] = -2.25102565E+01;
    COFD[3609] = 5.49776513E+00;
    COFD[3610] = -4.61463030E-01;
    COFD[3611] = 1.85209236E-02;
    COFD[3612] = -2.25140525E+01;
    COFD[3613] = 5.49776513E+00;
    COFD[3614] = -4.61463030E-01;
    COFD[3615] = 1.85209236E-02;
    COFD[3616] = -2.26435378E+01;
    COFD[3617] = 5.50023958E+00;
    COFD[3618] = -4.62136179E-01;
    COFD[3619] = 1.85639061E-02;
    COFD[3620] = -2.20687596E+01;
    COFD[3621] = 5.08417640E+00;
    COFD[3622] = -3.89810534E-01;
    COFD[3623] = 1.47010214E-02;
    COFD[3624] = -2.25972054E+01;
    COFD[3625] = 5.39655717E+00;
    COFD[3626] = -4.42728390E-01;
    COFD[3627] = 1.74857336E-02;
    COFD[3628] = -2.26354564E+01;
    COFD[3629] = 5.42002683E+00;
    COFD[3630] = -4.47111163E-01;
    COFD[3631] = 1.77287360E-02;
    COFD[3632] = -2.22940707E+01;
    COFD[3633] = 5.25941804E+00;
    COFD[3634] = -4.19208672E-01;
    COFD[3635] = 1.62385114E-02;
    COFD[3636] = -2.21581289E+01;
    COFD[3637] = 5.12338366E+00;
    COFD[3638] = -3.96176894E-01;
    COFD[3639] = 1.50278196E-02;
    COFD[3640] = -2.25160816E+01;
    COFD[3641] = 5.21003123E+00;
    COFD[3642] = -4.10612564E-01;
    COFD[3643] = 1.57798598E-02;
    COFD[3644] = -2.25175307E+01;
    COFD[3645] = 5.21003123E+00;
    COFD[3646] = -4.10612564E-01;
    COFD[3647] = 1.57798598E-02;
    COFD[3648] = -1.91796663E+01;
    COFD[3649] = 4.70714822E+00;
    COFD[3650] = -3.94261134E-01;
    COFD[3651] = 1.70175169E-02;
    COFD[3652] = -1.80480958E+01;
    COFD[3653] = 4.45434023E+00;
    COFD[3654] = -3.63584633E-01;
    COFD[3655] = 1.57739270E-02;
    COFD[3656] = -1.46719197E+01;
    COFD[3657] = 3.52400594E+00;
    COFD[3658] = -2.46379985E-01;
    COFD[3659] = 1.08326032E-02;
    COFD[3660] = -1.86493112E+01;
    COFD[3661] = 5.16040659E+00;
    COFD[3662] = -4.46843492E-01;
    COFD[3663] = 1.90466181E-02;
    COFD[3664] = -1.80698901E+01;
    COFD[3665] = 4.45434023E+00;
    COFD[3666] = -3.63584633E-01;
    COFD[3667] = 1.57739270E-02;
    COFD[3668] = -2.05372411E+01;
    COFD[3669] = 4.83379373E+00;
    COFD[3670] = -3.50008083E-01;
    COFD[3671] = 1.26863426E-02;
    COFD[3672] = -1.94912151E+01;
    COFD[3673] = 4.81575071E+00;
    COFD[3674] = -4.07042139E-01;
    COFD[3675] = 1.75187504E-02;
    COFD[3676] = -1.94998722E+01;
    COFD[3677] = 4.81575071E+00;
    COFD[3678] = -4.07042139E-01;
    COFD[3679] = 1.75187504E-02;
    COFD[3680] = -1.95081555E+01;
    COFD[3681] = 4.81575071E+00;
    COFD[3682] = -4.07042139E-01;
    COFD[3683] = 1.75187504E-02;
    COFD[3684] = -2.21216828E+01;
    COFD[3685] = 5.60203389E+00;
    COFD[3686] = -4.91444416E-01;
    COFD[3687] = 2.04761886E-02;
    COFD[3688] = -1.92044492E+01;
    COFD[3689] = 4.71304783E+00;
    COFD[3690] = -3.94942083E-01;
    COFD[3691] = 1.70435959E-02;
    COFD[3692] = -2.15258568E+01;
    COFD[3693] = 5.12799307E+00;
    COFD[3694] = -3.96938732E-01;
    COFD[3695] = 1.50673195E-02;
    COFD[3696] = -2.03367561E+01;
    COFD[3697] = 5.15740122E+00;
    COFD[3698] = -4.46644818E-01;
    COFD[3699] = 1.90459001E-02;
    COFD[3700] = -2.03123540E+01;
    COFD[3701] = 5.14854169E+00;
    COFD[3702] = -4.45984343E-01;
    COFD[3703] = 1.90374217E-02;
    COFD[3704] = -2.16420936E+01;
    COFD[3705] = 5.17945041E+00;
    COFD[3706] = -4.05514689E-01;
    COFD[3707] = 1.55141412E-02;
    COFD[3708] = -2.21793326E+01;
    COFD[3709] = 5.60403905E+00;
    COFD[3710] = -4.91221691E-01;
    COFD[3711] = 2.04473483E-02;
    COFD[3712] = -2.20192352E+01;
    COFD[3713] = 5.31412694E+00;
    COFD[3714] = -4.28473898E-01;
    COFD[3715] = 1.67264841E-02;
    COFD[3716] = -2.20411190E+01;
    COFD[3717] = 5.31412694E+00;
    COFD[3718] = -4.28473898E-01;
    COFD[3719] = 1.67264841E-02;
    COFD[3720] = -2.17407419E+01;
    COFD[3721] = 5.17945041E+00;
    COFD[3722] = -4.05514689E-01;
    COFD[3723] = 1.55141412E-02;
    COFD[3724] = -2.20511271E+01;
    COFD[3725] = 5.60809037E+00;
    COFD[3726] = -4.89400803E-01;
    COFD[3727] = 2.02760802E-02;
    COFD[3728] = -2.23689627E+01;
    COFD[3729] = 5.58513878E+00;
    COFD[3730] = -4.80389524E-01;
    COFD[3731] = 1.96438689E-02;
    COFD[3732] = -2.19032561E+01;
    COFD[3733] = 5.59794138E+00;
    COFD[3734] = -4.91684532E-01;
    COFD[3735] = 2.05170953E-02;
    COFD[3736] = -2.24018266E+01;
    COFD[3737] = 5.57115285E+00;
    COFD[3738] = -4.76363416E-01;
    COFD[3739] = 1.93814080E-02;
    COFD[3740] = -2.24083163E+01;
    COFD[3741] = 5.57115285E+00;
    COFD[3742] = -4.76363416E-01;
    COFD[3743] = 1.93814080E-02;
    COFD[3744] = -2.24021886E+01;
    COFD[3745] = 5.58364149E+00;
    COFD[3746] = -4.79184111E-01;
    COFD[3747] = 1.95516164E-02;
    COFD[3748] = -2.25161211E+01;
    COFD[3749] = 5.58521783E+00;
    COFD[3750] = -4.80947522E-01;
    COFD[3751] = 1.96897222E-02;
    COFD[3752] = -2.21603646E+01;
    COFD[3753] = 5.18050127E+00;
    COFD[3754] = -4.05688517E-01;
    COFD[3755] = 1.55231713E-02;
    COFD[3756] = -2.25060112E+01;
    COFD[3757] = 5.50327119E+00;
    COFD[3758] = -4.63087223E-01;
    COFD[3759] = 1.86271401E-02;
    COFD[3760] = -2.24931486E+01;
    COFD[3761] = 5.50509817E+00;
    COFD[3762] = -4.63572794E-01;
    COFD[3763] = 1.86581046E-02;
    COFD[3764] = -2.24969995E+01;
    COFD[3765] = 5.50509817E+00;
    COFD[3766] = -4.63572794E-01;
    COFD[3767] = 1.86581046E-02;
    COFD[3768] = -2.26299936E+01;
    COFD[3769] = 5.50881574E+00;
    COFD[3770] = -4.64448886E-01;
    COFD[3771] = 1.87118881E-02;
    COFD[3772] = -2.21535971E+01;
    COFD[3773] = 5.13453409E+00;
    COFD[3774] = -3.98022439E-01;
    COFD[3775] = 1.51235760E-02;
    COFD[3776] = -2.26203761E+01;
    COFD[3777] = 5.42039607E+00;
    COFD[3778] = -4.47178505E-01;
    COFD[3779] = 1.77324253E-02;
    COFD[3780] = -2.26677753E+01;
    COFD[3781] = 5.44777353E+00;
    COFD[3782] = -4.52122340E-01;
    COFD[3783] = 1.80021910E-02;
    COFD[3784] = -2.23480908E+01;
    COFD[3785] = 5.29695321E+00;
    COFD[3786] = -4.25620113E-01;
    COFD[3787] = 1.65778213E-02;
    COFD[3788] = -2.22256643E+01;
    COFD[3789] = 5.16620234E+00;
    COFD[3790] = -4.03306755E-01;
    COFD[3791] = 1.53990058E-02;
    COFD[3792] = -2.25635595E+01;
    COFD[3793] = 5.24330646E+00;
    COFD[3794] = -4.16370120E-01;
    COFD[3795] = 1.60860486E-02;
    COFD[3796] = -2.25650343E+01;
    COFD[3797] = 5.24330646E+00;
    COFD[3798] = -4.16370120E-01;
    COFD[3799] = 1.60860486E-02;
    COFD[3800] = -1.92062897E+01;
    COFD[3801] = 4.66318669E+00;
    COFD[3802] = -3.89108667E-01;
    COFD[3803] = 1.68165377E-02;
    COFD[3804] = -1.80724788E+01;
    COFD[3805] = 4.40247898E+00;
    COFD[3806] = -3.57238362E-01;
    COFD[3807] = 1.55145651E-02;
    COFD[3808] = -1.47137939E+01;
    COFD[3809] = 3.48023191E+00;
    COFD[3810] = -2.40800798E-01;
    COFD[3811] = 1.05947990E-02;
    COFD[3812] = -1.87481780E+01;
    COFD[3813] = 5.13858656E+00;
    COFD[3814] = -4.45075387E-01;
    COFD[3815] = 1.90137309E-02;
    COFD[3816] = -1.80945693E+01;
    COFD[3817] = 4.40247898E+00;
    COFD[3818] = -3.57238362E-01;
    COFD[3819] = 1.55145651E-02;
    COFD[3820] = -2.08879167E+01;
    COFD[3821] = 4.92602269E+00;
    COFD[3822] = -3.64572914E-01;
    COFD[3823] = 1.34203681E-02;
    COFD[3824] = -1.95201830E+01;
    COFD[3825] = 4.77151544E+00;
    COFD[3826] = -4.01882811E-01;
    COFD[3827] = 1.73184814E-02;
    COFD[3828] = -1.95290229E+01;
    COFD[3829] = 4.77151544E+00;
    COFD[3830] = -4.01882811E-01;
    COFD[3831] = 1.73184814E-02;
    COFD[3832] = -1.95374840E+01;
    COFD[3833] = 4.77151544E+00;
    COFD[3834] = -4.01882811E-01;
    COFD[3835] = 1.73184814E-02;
    COFD[3836] = -2.22052004E+01;
    COFD[3837] = 5.58604166E+00;
    COFD[3838] = -4.90602184E-01;
    COFD[3839] = 2.04880352E-02;
    COFD[3840] = -1.92334028E+01;
    COFD[3841] = 4.67033934E+00;
    COFD[3842] = -3.89971551E-01;
    COFD[3843] = 1.68513441E-02;
    COFD[3844] = -2.17926864E+01;
    COFD[3845] = 5.19232842E+00;
    COFD[3846] = -4.07643284E-01;
    COFD[3847] = 1.56246434E-02;
    COFD[3848] = -2.03971290E+01;
    COFD[3849] = 5.13279789E+00;
    COFD[3850] = -4.44474174E-01;
    COFD[3851] = 1.89937678E-02;
    COFD[3852] = -2.03526104E+01;
    COFD[3853] = 5.11453301E+00;
    COFD[3854] = -4.42447016E-01;
    COFD[3855] = 1.89196698E-02;
    COFD[3856] = -2.18910102E+01;
    COFD[3857] = 5.23595129E+00;
    COFD[3858] = -4.15079064E-01;
    COFD[3859] = 1.60168286E-02;
    COFD[3860] = -2.22701953E+01;
    COFD[3861] = 5.59632316E+00;
    COFD[3862] = -4.91568011E-01;
    COFD[3863] = 2.05156966E-02;
    COFD[3864] = -2.22545356E+01;
    COFD[3865] = 5.36643605E+00;
    COFD[3866] = -4.37440735E-01;
    COFD[3867] = 1.72016388E-02;
    COFD[3868] = -2.22769618E+01;
    COFD[3869] = 5.36643605E+00;
    COFD[3870] = -4.37440735E-01;
    COFD[3871] = 1.72016388E-02;
    COFD[3872] = -2.19919464E+01;
    COFD[3873] = 5.23595129E+00;
    COFD[3874] = -4.15079064E-01;
    COFD[3875] = 1.60168286E-02;
    COFD[3876] = -2.21795362E+01;
    COFD[3877] = 5.61233637E+00;
    COFD[3878] = -4.91419253E-01;
    COFD[3879] = 2.04216738E-02;
    COFD[3880] = -2.24797372E+01;
    COFD[3881] = 5.58492389E+00;
    COFD[3882] = -4.81921515E-01;
    COFD[3883] = 1.97721229E-02;
    COFD[3884] = -2.19617258E+01;
    COFD[3885] = 5.57026255E+00;
    COFD[3886] = -4.89178491E-01;
    COFD[3887] = 2.04505218E-02;
    COFD[3888] = -2.25424404E+01;
    COFD[3889] = 5.58482894E+00;
    COFD[3890] = -4.79850522E-01;
    COFD[3891] = 1.96007690E-02;
    COFD[3892] = -2.25490826E+01;
    COFD[3893] = 5.58482894E+00;
    COFD[3894] = -4.79850522E-01;
    COFD[3895] = 1.96007690E-02;
    COFD[3896] = -2.25161211E+01;
    COFD[3897] = 5.58521783E+00;
    COFD[3898] = -4.80947522E-01;
    COFD[3899] = 1.96897222E-02;
    COFD[3900] = -2.26149345E+01;
    COFD[3901] = 5.58414475E+00;
    COFD[3902] = -4.82375215E-01;
    COFD[3903] = 1.98138565E-02;
    COFD[3904] = -2.23925982E+01;
    COFD[3905] = 5.23666690E+00;
    COFD[3906] = -4.15204403E-01;
    COFD[3907] = 1.60235416E-02;
    COFD[3908] = -2.26679870E+01;
    COFD[3909] = 5.52852425E+00;
    COFD[3910] = -4.68277964E-01;
    COFD[3911] = 1.89288127E-02;
    COFD[3912] = -2.26623034E+01;
    COFD[3913] = 5.53286772E+00;
    COFD[3914] = -4.69109018E-01;
    COFD[3915] = 1.89755392E-02;
    COFD[3916] = -2.26662608E+01;
    COFD[3917] = 5.53286772E+00;
    COFD[3918] = -4.69109018E-01;
    COFD[3919] = 1.89755392E-02;
    COFD[3920] = -2.27989630E+01;
    COFD[3921] = 5.53955653E+00;
    COFD[3922] = -4.70381353E-01;
    COFD[3923] = 1.90468698E-02;
    COFD[3924] = -2.24028537E+01;
    COFD[3925] = 5.19900179E+00;
    COFD[3926] = -4.08748226E-01;
    COFD[3927] = 1.56820407E-02;
    COFD[3928] = -2.28101231E+01;
    COFD[3929] = 5.46112592E+00;
    COFD[3930] = -4.54556926E-01;
    COFD[3931] = 1.81357650E-02;
    COFD[3932] = -2.28554026E+01;
    COFD[3933] = 5.48796011E+00;
    COFD[3934] = -4.59457942E-01;
    COFD[3935] = 1.84050728E-02;
    COFD[3936] = -2.25780442E+01;
    COFD[3937] = 5.35238497E+00;
    COFD[3938] = -4.35034945E-01;
    COFD[3939] = 1.70742216E-02;
    COFD[3940] = -2.24631694E+01;
    COFD[3941] = 5.22623384E+00;
    COFD[3942] = -4.13380324E-01;
    COFD[3943] = 1.59259437E-02;
    COFD[3944] = -2.27715883E+01;
    COFD[3945] = 5.29493402E+00;
    COFD[3946] = -4.25285978E-01;
    COFD[3947] = 1.65604533E-02;
    COFD[3948] = -2.27731137E+01;
    COFD[3949] = 5.29493402E+00;
    COFD[3950] = -4.25285978E-01;
    COFD[3951] = 1.65604533E-02;
    COFD[3952] = -2.09943481E+01;
    COFD[3953] = 5.22468467E+00;
    COFD[3954] = -4.54220128E-01;
    COFD[3955] = 1.93281042E-02;
    COFD[3956] = -1.98296243E+01;
    COFD[3957] = 4.98207523E+00;
    COFD[3958] = -4.26877291E-01;
    COFD[3959] = 1.83086094E-02;
    COFD[3960] = -1.64819183E+01;
    COFD[3961] = 4.11726215E+00;
    COFD[3962] = -3.22193015E-01;
    COFD[3963] = 1.40747074E-02;
    COFD[3964] = -2.01262921E+01;
    COFD[3965] = 5.54581286E+00;
    COFD[3966] = -4.87014004E-01;
    COFD[3967] = 2.03965482E-02;
    COFD[3968] = -1.98546695E+01;
    COFD[3969] = 4.98207523E+00;
    COFD[3970] = -4.26877291E-01;
    COFD[3971] = 1.83086094E-02;
    COFD[3972] = -1.73636900E+01;
    COFD[3973] = 3.17377130E+00;
    COFD[3974] = -1.00394383E-01;
    COFD[3975] = 5.69083899E-04;
    COFD[3976] = -2.13698722E+01;
    COFD[3977] = 5.34971865E+00;
    COFD[3978] = -4.68771123E-01;
    COFD[3979] = 1.98933811E-02;
    COFD[3980] = -2.12907159E+01;
    COFD[3981] = 5.32167660E+00;
    COFD[3982] = -4.65740624E-01;
    COFD[3983] = 1.97861081E-02;
    COFD[3984] = -2.13011157E+01;
    COFD[3985] = 5.32167660E+00;
    COFD[3986] = -4.65740624E-01;
    COFD[3987] = 1.97861081E-02;
    COFD[3988] = -2.25168081E+01;
    COFD[3989] = 5.46125558E+00;
    COFD[3990] = -4.54580949E-01;
    COFD[3991] = 1.81370928E-02;
    COFD[3992] = -2.10310742E+01;
    COFD[3993] = 5.23485505E+00;
    COFD[3994] = -4.55400362E-01;
    COFD[3995] = 1.93737680E-02;
    COFD[3996] = -1.98359760E+01;
    COFD[3997] = 4.11158627E+00;
    COFD[3998] = -2.37831519E-01;
    COFD[3999] = 7.10363413E-03;
    COFD[4000] = -2.18158049E+01;
    COFD[4001] = 5.53950393E+00;
    COFD[4002] = -4.86376204E-01;
    COFD[4003] = 2.03760106E-02;
    COFD[4004] = -2.18731920E+01;
    COFD[4005] = 5.55171660E+00;
    COFD[4006] = -4.87609504E-01;
    COFD[4007] = 2.04156590E-02;
    COFD[4008] = -2.00981944E+01;
    COFD[4009] = 4.22278378E+00;
    COFD[4010] = -2.54653500E-01;
    COFD[4011] = 7.92616085E-03;
    COFD[4012] = -2.25302512E+01;
    COFD[4013] = 5.47136127E+00;
    COFD[4014] = -4.56417141E-01;
    COFD[4015] = 1.82376994E-02;
    COFD[4016] = -2.08353693E+01;
    COFD[4017] = 4.50409026E+00;
    COFD[4018] = -2.97868419E-01;
    COFD[4019] = 1.00604224E-02;
    COFD[4020] = -2.08639466E+01;
    COFD[4021] = 4.50409026E+00;
    COFD[4022] = -2.97868419E-01;
    COFD[4023] = 1.00604224E-02;
    COFD[4024] = -2.02246117E+01;
    COFD[4025] = 4.22278378E+00;
    COFD[4026] = -2.54653500E-01;
    COFD[4027] = 7.92616085E-03;
    COFD[4028] = -2.22462130E+01;
    COFD[4029] = 5.40356304E+00;
    COFD[4030] = -4.44060256E-01;
    COFD[4031] = 1.75601121E-02;
    COFD[4032] = -2.22169882E+01;
    COFD[4033] = 5.21950983E+00;
    COFD[4034] = -4.12223195E-01;
    COFD[4035] = 1.58645894E-02;
    COFD[4036] = -2.23434237E+01;
    COFD[4037] = 5.49927389E+00;
    COFD[4038] = -4.61845436E-01;
    COFD[4039] = 1.85448066E-02;
    COFD[4040] = -2.20862991E+01;
    COFD[4041] = 5.13809011E+00;
    COFD[4042] = -3.98612308E-01;
    COFD[4043] = 1.51542189E-02;
    COFD[4044] = -2.20946432E+01;
    COFD[4045] = 5.13809011E+00;
    COFD[4046] = -3.98612308E-01;
    COFD[4047] = 1.51542189E-02;
    COFD[4048] = -2.21603646E+01;
    COFD[4049] = 5.18050127E+00;
    COFD[4050] = -4.05688517E-01;
    COFD[4051] = 1.55231713E-02;
    COFD[4052] = -2.23925982E+01;
    COFD[4053] = 5.23666690E+00;
    COFD[4054] = -4.15204403E-01;
    COFD[4055] = 1.60235416E-02;
    COFD[4056] = -2.02828056E+01;
    COFD[4057] = 4.06866060E+00;
    COFD[4058] = -2.33527101E-01;
    COFD[4059] = 6.97454219E-03;
    COFD[4060] = -2.19287691E+01;
    COFD[4061] = 4.95026695E+00;
    COFD[4062] = -3.68392434E-01;
    COFD[4063] = 1.36126514E-02;
    COFD[4064] = -2.19456782E+01;
    COFD[4065] = 4.96368178E+00;
    COFD[4066] = -3.70505465E-01;
    COFD[4067] = 1.37190339E-02;
    COFD[4068] = -2.19508859E+01;
    COFD[4069] = 4.96368178E+00;
    COFD[4070] = -3.70505465E-01;
    COFD[4071] = 1.37190339E-02;
    COFD[4072] = -2.21147103E+01;
    COFD[4073] = 4.98427447E+00;
    COFD[4074] = -3.73746896E-01;
    COFD[4075] = 1.38821805E-02;
    COFD[4076] = -2.05271615E+01;
    COFD[4077] = 4.12444157E+00;
    COFD[4078] = -2.39777376E-01;
    COFD[4079] = 7.19872269E-03;
    COFD[4080] = -2.16983332E+01;
    COFD[4081] = 4.71782117E+00;
    COFD[4082] = -3.31568259E-01;
    COFD[4083] = 1.17540937E-02;
    COFD[4084] = -2.17385144E+01;
    COFD[4085] = 4.74350080E+00;
    COFD[4086] = -3.36426340E-01;
    COFD[4087] = 1.20245796E-02;
    COFD[4088] = -2.11585495E+01;
    COFD[4089] = 4.47646812E+00;
    COFD[4090] = -2.93573165E-01;
    COFD[4091] = 9.84650920E-03;
    COFD[4092] = -2.06827490E+01;
    COFD[4093] = 4.19375892E+00;
    COFD[4094] = -2.50262428E-01;
    COFD[4095] = 7.71131487E-03;
    COFD[4096] = -2.12332312E+01;
    COFD[4097] = 4.36095377E+00;
    COFD[4098] = -2.75760539E-01;
    COFD[4099] = 8.96430249E-03;
    COFD[4100] = -2.12354028E+01;
    COFD[4101] = 4.36095377E+00;
    COFD[4102] = -2.75760539E-01;
    COFD[4103] = 8.96430249E-03;
    COFD[4104] = -1.97484166E+01;
    COFD[4105] = 4.84231878E+00;
    COFD[4106] = -4.10101001E-01;
    COFD[4107] = 1.76356687E-02;
    COFD[4108] = -1.86652603E+01;
    COFD[4109] = 4.61260432E+00;
    COFD[4110] = -3.82854484E-01;
    COFD[4111] = 1.65575163E-02;
    COFD[4112] = -1.51448279E+01;
    COFD[4113] = 3.64565939E+00;
    COFD[4114] = -2.61726871E-01;
    COFD[4115] = 1.14799244E-02;
    COFD[4116] = -1.92784178E+01;
    COFD[4117] = 5.32291505E+00;
    COFD[4118] = -4.65883522E-01;
    COFD[4119] = 1.97916109E-02;
    COFD[4120] = -1.86886689E+01;
    COFD[4121] = 4.61260432E+00;
    COFD[4122] = -3.82854484E-01;
    COFD[4123] = 1.65575163E-02;
    COFD[4124] = -2.02184916E+01;
    COFD[4125] = 4.57152878E+00;
    COFD[4126] = -3.08371263E-01;
    COFD[4127] = 1.05838559E-02;
    COFD[4128] = -2.01315602E+01;
    COFD[4129] = 4.97613338E+00;
    COFD[4130] = -4.26175206E-01;
    COFD[4131] = 1.82809270E-02;
    COFD[4132] = -2.01412473E+01;
    COFD[4133] = 4.97613338E+00;
    COFD[4134] = -4.26175206E-01;
    COFD[4135] = 1.82809270E-02;
    COFD[4136] = -2.01505348E+01;
    COFD[4137] = 4.97613338E+00;
    COFD[4138] = -4.26175206E-01;
    COFD[4139] = 1.82809270E-02;
    COFD[4140] = -2.23890317E+01;
    COFD[4141] = 5.59178974E+00;
    COFD[4142] = -4.85668031E-01;
    COFD[4143] = 2.00491907E-02;
    COFD[4144] = -1.97709603E+01;
    COFD[4145] = 4.84731557E+00;
    COFD[4146] = -4.10638352E-01;
    COFD[4147] = 1.76543886E-02;
    COFD[4148] = -2.12219677E+01;
    COFD[4149] = 4.87252053E+00;
    COFD[4150] = -3.56127804E-01;
    COFD[4151] = 1.29948788E-02;
    COFD[4152] = -2.09490548E+01;
    COFD[4153] = 5.31360223E+00;
    COFD[4154] = -4.64787000E-01;
    COFD[4155] = 1.97483720E-02;
    COFD[4156] = -2.08822487E+01;
    COFD[4157] = 5.28557747E+00;
    COFD[4158] = -4.61402384E-01;
    COFD[4159] = 1.96111546E-02;
    COFD[4160] = -2.13985484E+01;
    COFD[4161] = 4.94878244E+00;
    COFD[4162] = -3.68158605E-01;
    COFD[4163] = 1.36008797E-02;
    COFD[4164] = -2.24120415E+01;
    COFD[4165] = 5.58744076E+00;
    COFD[4166] = -4.84489462E-01;
    COFD[4167] = 1.99733042E-02;
    COFD[4168] = -2.19253091E+01;
    COFD[4169] = 5.14570932E+00;
    COFD[4170] = -3.99877142E-01;
    COFD[4171] = 1.52199557E-02;
    COFD[4172] = -2.19503032E+01;
    COFD[4173] = 5.14570932E+00;
    COFD[4174] = -3.99877142E-01;
    COFD[4175] = 1.52199557E-02;
    COFD[4176] = -2.15102238E+01;
    COFD[4177] = 4.94878244E+00;
    COFD[4178] = -3.68158605E-01;
    COFD[4179] = 1.36008797E-02;
    COFD[4180] = -2.22801170E+01;
    COFD[4181] = 5.58507108E+00;
    COFD[4182] = -4.81395065E-01;
    COFD[4183] = 1.97276199E-02;
    COFD[4184] = -2.25041734E+01;
    COFD[4185] = 5.51797622E+00;
    COFD[4186] = -4.66229499E-01;
    COFD[4187] = 1.88128348E-02;
    COFD[4188] = -2.21913393E+01;
    COFD[4189] = 5.60175327E+00;
    COFD[4190] = -4.87953216E-01;
    COFD[4191] = 2.01882171E-02;
    COFD[4192] = -2.25145418E+01;
    COFD[4193] = 5.49554403E+00;
    COFD[4194] = -4.60936491E-01;
    COFD[4195] = 1.84887572E-02;
    COFD[4196] = -2.25219004E+01;
    COFD[4197] = 5.49554403E+00;
    COFD[4198] = -4.60936491E-01;
    COFD[4199] = 1.84887572E-02;
    COFD[4200] = -2.25060112E+01;
    COFD[4201] = 5.50327119E+00;
    COFD[4202] = -4.63087223E-01;
    COFD[4203] = 1.86271401E-02;
    COFD[4204] = -2.26679870E+01;
    COFD[4205] = 5.52852425E+00;
    COFD[4206] = -4.68277964E-01;
    COFD[4207] = 1.89288127E-02;
    COFD[4208] = -2.19287691E+01;
    COFD[4209] = 4.95026695E+00;
    COFD[4210] = -3.68392434E-01;
    COFD[4211] = 1.36126514E-02;
    COFD[4212] = -2.25758616E+01;
    COFD[4213] = 5.40563818E+00;
    COFD[4214] = -4.44444322E-01;
    COFD[4215] = 1.75813146E-02;
    COFD[4216] = -2.25715533E+01;
    COFD[4217] = 5.41049872E+00;
    COFD[4218] = -4.45356411E-01;
    COFD[4219] = 1.76320470E-02;
    COFD[4220] = -2.25760230E+01;
    COFD[4221] = 5.41049872E+00;
    COFD[4222] = -4.45356411E-01;
    COFD[4223] = 1.76320470E-02;
    COFD[4224] = -2.27125829E+01;
    COFD[4225] = 5.41826700E+00;
    COFD[4226] = -4.46792049E-01;
    COFD[4227] = 1.77112976E-02;
    COFD[4228] = -2.18719802E+01;
    COFD[4229] = 4.88180276E+00;
    COFD[4230] = -3.57591995E-01;
    COFD[4231] = 1.30686372E-02;
    COFD[4232] = -2.25720229E+01;
    COFD[4233] = 5.27220175E+00;
    COFD[4234] = -4.21436175E-01;
    COFD[4235] = 1.63576263E-02;
    COFD[4236] = -2.26508835E+01;
    COFD[4237] = 5.31312101E+00;
    COFD[4238] = -4.28304541E-01;
    COFD[4239] = 1.67176023E-02;
    COFD[4240] = -2.22582201E+01;
    COFD[4241] = 5.12825866E+00;
    COFD[4242] = -3.96982702E-01;
    COFD[4243] = 1.50696010E-02;
    COFD[4244] = -2.19774160E+01;
    COFD[4245] = 4.92889157E+00;
    COFD[4246] = -3.65025286E-01;
    COFD[4247] = 1.34431452E-02;
    COFD[4248] = -2.24161979E+01;
    COFD[4249] = 5.05061421E+00;
    COFD[4250] = -3.84359196E-01;
    COFD[4251] = 1.44214004E-02;
    COFD[4252] = -2.24179759E+01;
    COFD[4253] = 5.05061421E+00;
    COFD[4254] = -3.84359196E-01;
    COFD[4255] = 1.44214004E-02;
    COFD[4256] = -1.97196489E+01;
    COFD[4257] = 4.83750266E+00;
    COFD[4258] = -4.09581452E-01;
    COFD[4259] = 1.76174739E-02;
    COFD[4260] = -1.86234701E+01;
    COFD[4261] = 4.60336076E+00;
    COFD[4262] = -3.81691643E-01;
    COFD[4263] = 1.65085234E-02;
    COFD[4264] = -1.51159870E+01;
    COFD[4265] = 3.64206330E+00;
    COFD[4266] = -2.61313444E-01;
    COFD[4267] = 1.14642754E-02;
    COFD[4268] = -1.92360228E+01;
    COFD[4269] = 5.31542554E+00;
    COFD[4270] = -4.65003780E-01;
    COFD[4271] = 1.97570185E-02;
    COFD[4272] = -1.86469792E+01;
    COFD[4273] = 4.60336076E+00;
    COFD[4274] = -3.81691643E-01;
    COFD[4275] = 1.65085234E-02;
    COFD[4276] = -2.02265558E+01;
    COFD[4277] = 4.58441724E+00;
    COFD[4278] = -3.10392854E-01;
    COFD[4279] = 1.06849990E-02;
    COFD[4280] = -2.00964665E+01;
    COFD[4281] = 4.96870443E+00;
    COFD[4282] = -4.25292447E-01;
    COFD[4283] = 1.82459096E-02;
    COFD[4284] = -2.01062206E+01;
    COFD[4285] = 4.96870443E+00;
    COFD[4286] = -4.25292447E-01;
    COFD[4287] = 1.82459096E-02;
    COFD[4288] = -2.01155735E+01;
    COFD[4289] = 4.96870443E+00;
    COFD[4290] = -4.25292447E-01;
    COFD[4291] = 1.82459096E-02;
    COFD[4292] = -2.23772680E+01;
    COFD[4293] = 5.59425354E+00;
    COFD[4294] = -4.86232980E-01;
    COFD[4295] = 2.00835981E-02;
    COFD[4296] = -1.97422209E+01;
    COFD[4297] = 4.84249900E+00;
    COFD[4298] = -4.10120448E-01;
    COFD[4299] = 1.76363500E-02;
    COFD[4300] = -2.12330900E+01;
    COFD[4301] = 4.88535789E+00;
    COFD[4302] = -3.58153894E-01;
    COFD[4303] = 1.30969624E-02;
    COFD[4304] = -2.09108261E+01;
    COFD[4305] = 5.30526648E+00;
    COFD[4306] = -4.63785596E-01;
    COFD[4307] = 1.97079873E-02;
    COFD[4308] = -2.08427678E+01;
    COFD[4309] = 5.27674330E+00;
    COFD[4310] = -4.60336155E-01;
    COFD[4311] = 1.95680191E-02;
    COFD[4312] = -2.14111310E+01;
    COFD[4313] = 4.96219227E+00;
    COFD[4314] = -3.70270843E-01;
    COFD[4315] = 1.37072211E-02;
    COFD[4316] = -2.23993836E+01;
    COFD[4317] = 5.58952429E+00;
    COFD[4318] = -4.85012530E-01;
    COFD[4319] = 2.00062142E-02;
    COFD[4320] = -2.19282979E+01;
    COFD[4321] = 5.15446948E+00;
    COFD[4322] = -4.01332769E-01;
    COFD[4323] = 1.52956262E-02;
    COFD[4324] = -2.19534987E+01;
    COFD[4325] = 5.15446948E+00;
    COFD[4326] = -4.01332769E-01;
    COFD[4327] = 1.52956262E-02;
    COFD[4328] = -2.15236645E+01;
    COFD[4329] = 4.96219227E+00;
    COFD[4330] = -3.70270843E-01;
    COFD[4331] = 1.37072211E-02;
    COFD[4332] = -2.22609256E+01;
    COFD[4333] = 5.58490856E+00;
    COFD[4334] = -4.81588720E-01;
    COFD[4335] = 1.97445317E-02;
    COFD[4336] = -2.24965286E+01;
    COFD[4337] = 5.52198915E+00;
    COFD[4338] = -4.67014474E-01;
    COFD[4339] = 1.88574253E-02;
    COFD[4340] = -2.21792065E+01;
    COFD[4341] = 5.60465338E+00;
    COFD[4342] = -4.88572478E-01;
    COFD[4343] = 2.02248525E-02;
    COFD[4344] = -2.25028406E+01;
    COFD[4345] = 5.49776513E+00;
    COFD[4346] = -4.61463030E-01;
    COFD[4347] = 1.85209236E-02;
    COFD[4348] = -2.25102565E+01;
    COFD[4349] = 5.49776513E+00;
    COFD[4350] = -4.61463030E-01;
    COFD[4351] = 1.85209236E-02;
    COFD[4352] = -2.24931486E+01;
    COFD[4353] = 5.50509817E+00;
    COFD[4354] = -4.63572794E-01;
    COFD[4355] = 1.86581046E-02;
    COFD[4356] = -2.26623034E+01;
    COFD[4357] = 5.53286772E+00;
    COFD[4358] = -4.69109018E-01;
    COFD[4359] = 1.89755392E-02;
    COFD[4360] = -2.19456782E+01;
    COFD[4361] = 4.96368178E+00;
    COFD[4362] = -3.70505465E-01;
    COFD[4363] = 1.37190339E-02;
    COFD[4364] = -2.25715533E+01;
    COFD[4365] = 5.41049872E+00;
    COFD[4366] = -4.45356411E-01;
    COFD[4367] = 1.76320470E-02;
    COFD[4368] = -2.25672005E+01;
    COFD[4369] = 5.41536807E+00;
    COFD[4370] = -4.46269562E-01;
    COFD[4371] = 1.76828228E-02;
    COFD[4372] = -2.25717119E+01;
    COFD[4373] = 5.41536807E+00;
    COFD[4374] = -4.46269562E-01;
    COFD[4375] = 1.76828228E-02;
    COFD[4376] = -2.27104617E+01;
    COFD[4377] = 5.42362435E+00;
    COFD[4378] = -4.47767764E-01;
    COFD[4379] = 1.77647205E-02;
    COFD[4380] = -2.18873269E+01;
    COFD[4381] = 4.89420156E+00;
    COFD[4382] = -3.59552981E-01;
    COFD[4383] = 1.31675149E-02;
    COFD[4384] = -2.25735380E+01;
    COFD[4385] = 5.27888680E+00;
    COFD[4386] = -4.22608469E-01;
    COFD[4387] = 1.64205533E-02;
    COFD[4388] = -2.26553469E+01;
    COFD[4389] = 5.32093606E+00;
    COFD[4390] = -4.29624801E-01;
    COFD[4391] = 1.67869730E-02;
    COFD[4392] = -2.22649250E+01;
    COFD[4393] = 5.13737172E+00;
    COFD[4394] = -3.98493102E-01;
    COFD[4395] = 1.51480250E-02;
    COFD[4396] = -2.19948202E+01;
    COFD[4397] = 4.94219368E+00;
    COFD[4398] = -3.67120797E-01;
    COFD[4399] = 1.35486343E-02;
    COFD[4400] = -2.24284563E+01;
    COFD[4401] = 5.06106414E+00;
    COFD[4402] = -3.86053039E-01;
    COFD[4403] = 1.45081784E-02;
    COFD[4404] = -2.24302556E+01;
    COFD[4405] = 5.06106414E+00;
    COFD[4406] = -3.86053039E-01;
    COFD[4407] = 1.45081784E-02;
    COFD[4408] = -1.97226856E+01;
    COFD[4409] = 4.83750266E+00;
    COFD[4410] = -4.09581452E-01;
    COFD[4411] = 1.76174739E-02;
    COFD[4412] = -1.86254955E+01;
    COFD[4413] = 4.60336076E+00;
    COFD[4414] = -3.81691643E-01;
    COFD[4415] = 1.65085234E-02;
    COFD[4416] = -1.51163041E+01;
    COFD[4417] = 3.64206330E+00;
    COFD[4418] = -2.61313444E-01;
    COFD[4419] = 1.14642754E-02;
    COFD[4420] = -1.92361841E+01;
    COFD[4421] = 5.31542554E+00;
    COFD[4422] = -4.65003780E-01;
    COFD[4423] = 1.97570185E-02;
    COFD[4424] = -1.86491023E+01;
    COFD[4425] = 4.60336076E+00;
    COFD[4426] = -3.81691643E-01;
    COFD[4427] = 1.65085234E-02;
    COFD[4428] = -2.02287739E+01;
    COFD[4429] = 4.58441724E+00;
    COFD[4430] = -3.10392854E-01;
    COFD[4431] = 1.06849990E-02;
    COFD[4432] = -2.00997774E+01;
    COFD[4433] = 4.96870443E+00;
    COFD[4434] = -4.25292447E-01;
    COFD[4435] = 1.82459096E-02;
    COFD[4436] = -2.01095969E+01;
    COFD[4437] = 4.96870443E+00;
    COFD[4438] = -4.25292447E-01;
    COFD[4439] = 1.82459096E-02;
    COFD[4440] = -2.01190139E+01;
    COFD[4441] = 4.96870443E+00;
    COFD[4442] = -4.25292447E-01;
    COFD[4443] = 1.82459096E-02;
    COFD[4444] = -2.23812726E+01;
    COFD[4445] = 5.59425354E+00;
    COFD[4446] = -4.86232980E-01;
    COFD[4447] = 2.00835981E-02;
    COFD[4448] = -1.97452574E+01;
    COFD[4449] = 4.84249900E+00;
    COFD[4450] = -4.10120448E-01;
    COFD[4451] = 1.76363500E-02;
    COFD[4452] = -2.12362684E+01;
    COFD[4453] = 4.88535789E+00;
    COFD[4454] = -3.58153894E-01;
    COFD[4455] = 1.30969624E-02;
    COFD[4456] = -2.09127554E+01;
    COFD[4457] = 5.30526648E+00;
    COFD[4458] = -4.63785596E-01;
    COFD[4459] = 1.97079873E-02;
    COFD[4460] = -2.08447974E+01;
    COFD[4461] = 5.27674330E+00;
    COFD[4462] = -4.60336155E-01;
    COFD[4463] = 1.95680191E-02;
    COFD[4464] = -2.14144448E+01;
    COFD[4465] = 4.96219227E+00;
    COFD[4466] = -3.70270843E-01;
    COFD[4467] = 1.37072211E-02;
    COFD[4468] = -2.24025650E+01;
    COFD[4469] = 5.58952429E+00;
    COFD[4470] = -4.85012530E-01;
    COFD[4471] = 2.00062142E-02;
    COFD[4472] = -2.19322003E+01;
    COFD[4473] = 5.15446948E+00;
    COFD[4474] = -4.01332769E-01;
    COFD[4475] = 1.52956262E-02;
    COFD[4476] = -2.19576037E+01;
    COFD[4477] = 5.15446948E+00;
    COFD[4478] = -4.01332769E-01;
    COFD[4479] = 1.52956262E-02;
    COFD[4480] = -2.15278182E+01;
    COFD[4481] = 4.96219227E+00;
    COFD[4482] = -3.70270843E-01;
    COFD[4483] = 1.37072211E-02;
    COFD[4484] = -2.22638165E+01;
    COFD[4485] = 5.58490856E+00;
    COFD[4486] = -4.81588720E-01;
    COFD[4487] = 1.97445317E-02;
    COFD[4488] = -2.25004333E+01;
    COFD[4489] = 5.52198915E+00;
    COFD[4490] = -4.67014474E-01;
    COFD[4491] = 1.88574253E-02;
    COFD[4492] = -2.21822461E+01;
    COFD[4493] = 5.60465338E+00;
    COFD[4494] = -4.88572478E-01;
    COFD[4495] = 2.02248525E-02;
    COFD[4496] = -2.25065805E+01;
    COFD[4497] = 5.49776513E+00;
    COFD[4498] = -4.61463030E-01;
    COFD[4499] = 1.85209236E-02;
    COFD[4500] = -2.25140525E+01;
    COFD[4501] = 5.49776513E+00;
    COFD[4502] = -4.61463030E-01;
    COFD[4503] = 1.85209236E-02;
    COFD[4504] = -2.24969995E+01;
    COFD[4505] = 5.50509817E+00;
    COFD[4506] = -4.63572794E-01;
    COFD[4507] = 1.86581046E-02;
    COFD[4508] = -2.26662608E+01;
    COFD[4509] = 5.53286772E+00;
    COFD[4510] = -4.69109018E-01;
    COFD[4511] = 1.89755392E-02;
    COFD[4512] = -2.19508859E+01;
    COFD[4513] = 4.96368178E+00;
    COFD[4514] = -3.70505465E-01;
    COFD[4515] = 1.37190339E-02;
    COFD[4516] = -2.25760230E+01;
    COFD[4517] = 5.41049872E+00;
    COFD[4518] = -4.45356411E-01;
    COFD[4519] = 1.76320470E-02;
    COFD[4520] = -2.25717119E+01;
    COFD[4521] = 5.41536807E+00;
    COFD[4522] = -4.46269562E-01;
    COFD[4523] = 1.76828228E-02;
    COFD[4524] = -2.25762645E+01;
    COFD[4525] = 5.41536807E+00;
    COFD[4526] = -4.46269562E-01;
    COFD[4527] = 1.76828228E-02;
    COFD[4528] = -2.27150546E+01;
    COFD[4529] = 5.42362435E+00;
    COFD[4530] = -4.47767764E-01;
    COFD[4531] = 1.77647205E-02;
    COFD[4532] = -2.18929084E+01;
    COFD[4533] = 4.89420156E+00;
    COFD[4534] = -3.59552981E-01;
    COFD[4535] = 1.31675149E-02;
    COFD[4536] = -2.25785614E+01;
    COFD[4537] = 5.27888680E+00;
    COFD[4538] = -4.22608469E-01;
    COFD[4539] = 1.64205533E-02;
    COFD[4540] = -2.26604027E+01;
    COFD[4541] = 5.32093606E+00;
    COFD[4542] = -4.29624801E-01;
    COFD[4543] = 1.67869730E-02;
    COFD[4544] = -2.22700127E+01;
    COFD[4545] = 5.13737172E+00;
    COFD[4546] = -3.98493102E-01;
    COFD[4547] = 1.51480250E-02;
    COFD[4548] = -2.20002783E+01;
    COFD[4549] = 4.94219368E+00;
    COFD[4550] = -3.67120797E-01;
    COFD[4551] = 1.35486343E-02;
    COFD[4552] = -2.24342646E+01;
    COFD[4553] = 5.06106414E+00;
    COFD[4554] = -3.86053039E-01;
    COFD[4555] = 1.45081784E-02;
    COFD[4556] = -2.24360850E+01;
    COFD[4557] = 5.06106414E+00;
    COFD[4558] = -3.86053039E-01;
    COFD[4559] = 1.45081784E-02;
    COFD[4560] = -1.98374654E+01;
    COFD[4561] = 4.82871870E+00;
    COFD[4562] = -4.08567726E-01;
    COFD[4563] = 1.75785896E-02;
    COFD[4564] = -1.87433618E+01;
    COFD[4565] = 4.58956960E+00;
    COFD[4566] = -3.79964215E-01;
    COFD[4567] = 1.64361138E-02;
    COFD[4568] = -1.52503668E+01;
    COFD[4569] = 3.63657318E+00;
    COFD[4570] = -2.60678457E-01;
    COFD[4571] = 1.14400550E-02;
    COFD[4572] = -1.93693740E+01;
    COFD[4573] = 5.30286598E+00;
    COFD[4574] = -4.63495567E-01;
    COFD[4575] = 1.96962203E-02;
    COFD[4576] = -1.87670637E+01;
    COFD[4577] = 4.58956960E+00;
    COFD[4578] = -3.79964215E-01;
    COFD[4579] = 1.64361138E-02;
    COFD[4580] = -2.04186424E+01;
    COFD[4581] = 4.60117690E+00;
    COFD[4582] = -3.13067257E-01;
    COFD[4583] = 1.08202310E-02;
    COFD[4584] = -2.02121663E+01;
    COFD[4585] = 4.95786261E+00;
    COFD[4586] = -4.24013131E-01;
    COFD[4587] = 1.81955669E-02;
    COFD[4588] = -2.02220498E+01;
    COFD[4589] = 4.95786261E+00;
    COFD[4590] = -4.24013131E-01;
    COFD[4591] = 1.81955669E-02;
    COFD[4592] = -2.02315293E+01;
    COFD[4593] = 4.95786261E+00;
    COFD[4594] = -4.24013131E-01;
    COFD[4595] = 1.81955669E-02;
    COFD[4596] = -2.25216613E+01;
    COFD[4597] = 5.59792043E+00;
    COFD[4598] = -4.87076900E-01;
    COFD[4599] = 2.01350364E-02;
    COFD[4600] = -1.98616115E+01;
    COFD[4601] = 4.83466791E+00;
    COFD[4602] = -4.09252052E-01;
    COFD[4603] = 1.76047341E-02;
    COFD[4604] = -2.14142864E+01;
    COFD[4605] = 4.90439970E+00;
    COFD[4606] = -3.61162615E-01;
    COFD[4607] = 1.32486109E-02;
    COFD[4608] = -2.10124405E+01;
    COFD[4609] = 5.29210705E+00;
    COFD[4610] = -4.62193217E-01;
    COFD[4611] = 1.96432872E-02;
    COFD[4612] = -2.09461018E+01;
    COFD[4613] = 5.26396793E+00;
    COFD[4614] = -4.58812213E-01;
    COFD[4615] = 1.95072180E-02;
    COFD[4616] = -2.15952753E+01;
    COFD[4617] = 4.98271982E+00;
    COFD[4618] = -3.73502341E-01;
    COFD[4619] = 1.38698700E-02;
    COFD[4620] = -2.25300734E+01;
    COFD[4621] = 5.59173268E+00;
    COFD[4622] = -4.85654660E-01;
    COFD[4623] = 2.00483698E-02;
    COFD[4624] = -2.20891322E+01;
    COFD[4625] = 5.16679492E+00;
    COFD[4626] = -4.03405751E-01;
    COFD[4627] = 1.54041741E-02;
    COFD[4628] = -2.21147341E+01;
    COFD[4629] = 5.16679492E+00;
    COFD[4630] = -4.03405751E-01;
    COFD[4631] = 1.54041741E-02;
    COFD[4632] = -2.17094710E+01;
    COFD[4633] = 4.98271982E+00;
    COFD[4634] = -3.73502341E-01;
    COFD[4635] = 1.38698700E-02;
    COFD[4636] = -2.23950513E+01;
    COFD[4637] = 5.58492366E+00;
    COFD[4638] = -4.81921868E-01;
    COFD[4639] = 1.97721534E-02;
    COFD[4640] = -2.26411013E+01;
    COFD[4641] = 5.52830072E+00;
    COFD[4642] = -4.68235018E-01;
    COFD[4643] = 1.89263933E-02;
    COFD[4644] = -2.23250359E+01;
    COFD[4645] = 5.60776666E+00;
    COFD[4646] = -4.89319792E-01;
    COFD[4647] = 2.02710069E-02;
    COFD[4648] = -2.26360108E+01;
    COFD[4649] = 5.50023958E+00;
    COFD[4650] = -4.62136179E-01;
    COFD[4651] = 1.85639061E-02;
    COFD[4652] = -2.26435378E+01;
    COFD[4653] = 5.50023958E+00;
    COFD[4654] = -4.62136179E-01;
    COFD[4655] = 1.85639061E-02;
    COFD[4656] = -2.26299936E+01;
    COFD[4657] = 5.50881574E+00;
    COFD[4658] = -4.64448886E-01;
    COFD[4659] = 1.87118881E-02;
    COFD[4660] = -2.27989630E+01;
    COFD[4661] = 5.53955653E+00;
    COFD[4662] = -4.70381353E-01;
    COFD[4663] = 1.90468698E-02;
    COFD[4664] = -2.21147103E+01;
    COFD[4665] = 4.98427447E+00;
    COFD[4666] = -3.73746896E-01;
    COFD[4667] = 1.38821805E-02;
    COFD[4668] = -2.27125829E+01;
    COFD[4669] = 5.41826700E+00;
    COFD[4670] = -4.46792049E-01;
    COFD[4671] = 1.77112976E-02;
    COFD[4672] = -2.27104617E+01;
    COFD[4673] = 5.42362435E+00;
    COFD[4674] = -4.47767764E-01;
    COFD[4675] = 1.77647205E-02;
    COFD[4676] = -2.27150546E+01;
    COFD[4677] = 5.42362435E+00;
    COFD[4678] = -4.47767764E-01;
    COFD[4679] = 1.77647205E-02;
    COFD[4680] = -2.28473186E+01;
    COFD[4681] = 5.43214736E+00;
    COFD[4682] = -4.49307302E-01;
    COFD[4683] = 1.78487187E-02;
    COFD[4684] = -2.20523730E+01;
    COFD[4685] = 4.91371677E+00;
    COFD[4686] = -3.62632206E-01;
    COFD[4687] = 1.33226347E-02;
    COFD[4688] = -2.27197587E+01;
    COFD[4689] = 5.29216864E+00;
    COFD[4690] = -4.24828662E-01;
    COFD[4691] = 1.65366835E-02;
    COFD[4692] = -2.27940520E+01;
    COFD[4693] = 5.33097599E+00;
    COFD[4694] = -4.31367349E-01;
    COFD[4695] = 1.68798869E-02;
    COFD[4696] = -2.24171125E+01;
    COFD[4697] = 5.15109664E+00;
    COFD[4698] = -4.00765892E-01;
    COFD[4699] = 1.52659560E-02;
    COFD[4700] = -2.21597603E+01;
    COFD[4701] = 4.96243463E+00;
    COFD[4702] = -3.70309018E-01;
    COFD[4703] = 1.37091432E-02;
    COFD[4704] = -2.25733338E+01;
    COFD[4705] = 5.07648425E+00;
    COFD[4706] = -3.88560019E-01;
    COFD[4707] = 1.46368353E-02;
    COFD[4708] = -2.25751750E+01;
    COFD[4709] = 5.07648425E+00;
    COFD[4710] = -3.88560019E-01;
    COFD[4711] = 1.46368353E-02;
    COFD[4712] = -2.10643735E+01;
    COFD[4713] = 5.22604478E+00;
    COFD[4714] = -4.54378127E-01;
    COFD[4715] = 1.93342248E-02;
    COFD[4716] = -2.00070284E+01;
    COFD[4717] = 5.02095434E+00;
    COFD[4718] = -4.31496874E-01;
    COFD[4719] = 1.84920392E-02;
    COFD[4720] = -1.65048875E+01;
    COFD[4721] = 4.10792536E+00;
    COFD[4722] = -3.21060656E-01;
    COFD[4723] = 1.40287900E-02;
    COFD[4724] = -2.02592914E+01;
    COFD[4725] = 5.56701235E+00;
    COFD[4726] = -4.88925090E-01;
    COFD[4727] = 2.04461948E-02;
    COFD[4728] = -2.00328044E+01;
    COFD[4729] = 5.02095434E+00;
    COFD[4730] = -4.31496874E-01;
    COFD[4731] = 1.84920392E-02;
    COFD[4732] = -1.83939699E+01;
    COFD[4733] = 3.59019527E+00;
    COFD[4734] = -1.58702132E-01;
    COFD[4735] = 3.23316765E-03;
    COFD[4736] = -2.14416336E+01;
    COFD[4737] = 5.35040988E+00;
    COFD[4738] = -4.68827063E-01;
    COFD[4739] = 1.98944407E-02;
    COFD[4740] = -2.14529967E+01;
    COFD[4741] = 5.35040988E+00;
    COFD[4742] = -4.68827063E-01;
    COFD[4743] = 1.98944407E-02;
    COFD[4744] = -2.14639274E+01;
    COFD[4745] = 5.35040988E+00;
    COFD[4746] = -4.68827063E-01;
    COFD[4747] = 1.98944407E-02;
    COFD[4748] = -2.25838099E+01;
    COFD[4749] = 5.45615714E+00;
    COFD[4750] = -4.53643844E-01;
    COFD[4751] = 1.80854821E-02;
    COFD[4752] = -2.10924694E+01;
    COFD[4753] = 5.23339224E+00;
    COFD[4754] = -4.55230780E-01;
    COFD[4755] = 1.93672146E-02;
    COFD[4756] = -1.96860113E+01;
    COFD[4757] = 4.00653795E+00;
    COFD[4758] = -2.22005804E-01;
    COFD[4759] = 6.33194910E-03;
    COFD[4760] = -2.19548723E+01;
    COFD[4761] = 5.56282156E+00;
    COFD[4762] = -4.88585679E-01;
    COFD[4763] = 2.04395879E-02;
    COFD[4764] = -2.19244555E+01;
    COFD[4765] = 5.54986547E+00;
    COFD[4766] = -4.87420926E-01;
    COFD[4767] = 2.04095097E-02;
    COFD[4768] = -1.99604682E+01;
    COFD[4769] = 4.12245214E+00;
    COFD[4770] = -2.39476227E-01;
    COFD[4771] = 7.18400558E-03;
    COFD[4772] = -2.25553202E+01;
    COFD[4773] = 5.44166443E+00;
    COFD[4774] = -4.51021243E-01;
    COFD[4775] = 1.79421190E-02;
    COFD[4776] = -2.07557953E+01;
    COFD[4777] = 4.42680848E+00;
    COFD[4778] = -2.85885288E-01;
    COFD[4779] = 9.46483934E-03;
    COFD[4780] = -2.07861367E+01;
    COFD[4781] = 4.42680848E+00;
    COFD[4782] = -2.85885288E-01;
    COFD[4783] = 9.46483934E-03;
    COFD[4784] = -2.00940426E+01;
    COFD[4785] = 4.12245214E+00;
    COFD[4786] = -2.39476227E-01;
    COFD[4787] = 7.18400558E-03;
    COFD[4788] = -2.22709427E+01;
    COFD[4789] = 5.37360713E+00;
    COFD[4790] = -4.38661889E-01;
    COFD[4791] = 1.72661628E-02;
    COFD[4792] = -2.22139496E+01;
    COFD[4793] = 5.17488844E+00;
    COFD[4794] = -4.04758505E-01;
    COFD[4795] = 1.54748177E-02;
    COFD[4796] = -2.23935500E+01;
    COFD[4797] = 5.47922490E+00;
    COFD[4798] = -4.57847893E-01;
    COFD[4799] = 1.83161707E-02;
    COFD[4800] = -2.20599362E+01;
    COFD[4801] = 5.08417640E+00;
    COFD[4802] = -3.89810534E-01;
    COFD[4803] = 1.47010214E-02;
    COFD[4804] = -2.20687596E+01;
    COFD[4805] = 5.08417640E+00;
    COFD[4806] = -3.89810534E-01;
    COFD[4807] = 1.47010214E-02;
    COFD[4808] = -2.21535971E+01;
    COFD[4809] = 5.13453409E+00;
    COFD[4810] = -3.98022439E-01;
    COFD[4811] = 1.51235760E-02;
    COFD[4812] = -2.24028537E+01;
    COFD[4813] = 5.19900179E+00;
    COFD[4814] = -4.08748226E-01;
    COFD[4815] = 1.56820407E-02;
    COFD[4816] = -2.05271615E+01;
    COFD[4817] = 4.12444157E+00;
    COFD[4818] = -2.39777376E-01;
    COFD[4819] = 7.19872269E-03;
    COFD[4820] = -2.18719802E+01;
    COFD[4821] = 4.88180276E+00;
    COFD[4822] = -3.57591995E-01;
    COFD[4823] = 1.30686372E-02;
    COFD[4824] = -2.18873269E+01;
    COFD[4825] = 4.89420156E+00;
    COFD[4826] = -3.59552981E-01;
    COFD[4827] = 1.31675149E-02;
    COFD[4828] = -2.18929084E+01;
    COFD[4829] = 4.89420156E+00;
    COFD[4830] = -3.59552981E-01;
    COFD[4831] = 1.31675149E-02;
    COFD[4832] = -2.20523730E+01;
    COFD[4833] = 4.91371677E+00;
    COFD[4834] = -3.62632206E-01;
    COFD[4835] = 1.33226347E-02;
    COFD[4836] = -2.03937742E+01;
    COFD[4837] = 4.02033531E+00;
    COFD[4838] = -2.24082500E-01;
    COFD[4839] = 6.43305206E-03;
    COFD[4840] = -2.16466727E+01;
    COFD[4841] = 4.65048212E+00;
    COFD[4842] = -3.20931552E-01;
    COFD[4843] = 1.12185393E-02;
    COFD[4844] = -2.17749249E+01;
    COFD[4845] = 4.71207875E+00;
    COFD[4846] = -3.30658500E-01;
    COFD[4847] = 1.17082011E-02;
    COFD[4848] = -2.10757112E+01;
    COFD[4849] = 4.39521460E+00;
    COFD[4850] = -2.81028854E-01;
    COFD[4851] = 9.22466916E-03;
    COFD[4852] = -2.05583124E+01;
    COFD[4853] = 4.09420232E+00;
    COFD[4854] = -2.35210019E-01;
    COFD[4855] = 6.97573395E-03;
    COFD[4856] = -2.11438235E+01;
    COFD[4857] = 4.27612828E+00;
    COFD[4858] = -2.62774610E-01;
    COFD[4859] = 8.32471127E-03;
    COFD[4860] = -2.11462093E+01;
    COFD[4861] = 4.27612828E+00;
    COFD[4862] = -2.62774610E-01;
    COFD[4863] = 8.32471127E-03;
    COFD[4864] = -2.03706752E+01;
    COFD[4865] = 4.98803076E+00;
    COFD[4866] = -4.27580621E-01;
    COFD[4867] = 1.83363274E-02;
    COFD[4868] = -1.92404583E+01;
    COFD[4869] = 4.73921581E+00;
    COFD[4870] = -3.98017274E-01;
    COFD[4871] = 1.71639614E-02;
    COFD[4872] = -1.56919143E+01;
    COFD[4873] = 3.77842689E+00;
    COFD[4874] = -2.78523399E-01;
    COFD[4875] = 1.21896111E-02;
    COFD[4876] = -1.97252269E+01;
    COFD[4877] = 5.38884098E+00;
    COFD[4878] = -4.71627912E-01;
    COFD[4879] = 1.99273178E-02;
    COFD[4880] = -1.92651204E+01;
    COFD[4881] = 4.73921581E+00;
    COFD[4882] = -3.98017274E-01;
    COFD[4883] = 1.71639614E-02;
    COFD[4884] = -1.98682752E+01;
    COFD[4885] = 4.28648872E+00;
    COFD[4886] = -2.64358750E-01;
    COFD[4887] = 8.40263071E-03;
    COFD[4888] = -2.07257272E+01;
    COFD[4889] = 5.10688723E+00;
    COFD[4890] = -4.41563971E-01;
    COFD[4891] = 1.88857198E-02;
    COFD[4892] = -2.07362753E+01;
    COFD[4893] = 5.10688723E+00;
    COFD[4894] = -4.41563971E-01;
    COFD[4895] = 1.88857198E-02;
    COFD[4896] = -2.07464056E+01;
    COFD[4897] = 5.10688723E+00;
    COFD[4898] = -4.41563971E-01;
    COFD[4899] = 1.88857198E-02;
    COFD[4900] = -2.26897188E+01;
    COFD[4901] = 5.58518389E+00;
    COFD[4902] = -4.80570209E-01;
    COFD[4903] = 1.96586179E-02;
    COFD[4904] = -2.03988322E+01;
    COFD[4905] = 4.99562188E+00;
    COFD[4906] = -4.28482025E-01;
    COFD[4907] = 1.83720948E-02;
    COFD[4908] = -2.09922023E+01;
    COFD[4909] = 4.64167142E+00;
    COFD[4910] = -3.19532110E-01;
    COFD[4911] = 1.11478359E-02;
    COFD[4912] = -2.13903532E+01;
    COFD[4913] = 5.38519776E+00;
    COFD[4914] = -4.71344997E-01;
    COFD[4915] = 1.99226932E-02;
    COFD[4916] = -2.13695648E+01;
    COFD[4917] = 5.37614538E+00;
    COFD[4918] = -4.70679659E-01;
    COFD[4919] = 1.99143937E-02;
    COFD[4920] = -2.11660262E+01;
    COFD[4921] = 4.71644372E+00;
    COFD[4922] = -3.31349990E-01;
    COFD[4923] = 1.17430818E-02;
    COFD[4924] = -2.27001899E+01;
    COFD[4925] = 5.58468914E+00;
    COFD[4926] = -4.79958407E-01;
    COFD[4927] = 1.96104043E-02;
    COFD[4928] = -2.17463767E+01;
    COFD[4929] = 4.93496210E+00;
    COFD[4930] = -3.65981745E-01;
    COFD[4931] = 1.34912948E-02;
    COFD[4932] = -2.17740719E+01;
    COFD[4933] = 4.93496210E+00;
    COFD[4934] = -3.65981745E-01;
    COFD[4935] = 1.34912948E-02;
    COFD[4936] = -2.12888403E+01;
    COFD[4937] = 4.71644372E+00;
    COFD[4938] = -3.31349990E-01;
    COFD[4939] = 1.17430818E-02;
    COFD[4940] = -2.24990717E+01;
    COFD[4941] = 5.55026833E+00;
    COFD[4942] = -4.72437808E-01;
    COFD[4943] = 1.91625195E-02;
    COFD[4944] = -2.26485311E+01;
    COFD[4945] = 5.44696782E+00;
    COFD[4946] = -4.51976837E-01;
    COFD[4947] = 1.79942461E-02;
    COFD[4948] = -2.24603310E+01;
    COFD[4949] = 5.58501539E+00;
    COFD[4950] = -4.81433860E-01;
    COFD[4951] = 1.97311245E-02;
    COFD[4952] = -2.25891024E+01;
    COFD[4953] = 5.39655717E+00;
    COFD[4954] = -4.42728390E-01;
    COFD[4955] = 1.74857336E-02;
    COFD[4956] = -2.25972054E+01;
    COFD[4957] = 5.39655717E+00;
    COFD[4958] = -4.42728390E-01;
    COFD[4959] = 1.74857336E-02;
    COFD[4960] = -2.26203761E+01;
    COFD[4961] = 5.42039607E+00;
    COFD[4962] = -4.47178505E-01;
    COFD[4963] = 1.77324253E-02;
    COFD[4964] = -2.28101231E+01;
    COFD[4965] = 5.46112592E+00;
    COFD[4966] = -4.54556926E-01;
    COFD[4967] = 1.81357650E-02;
    COFD[4968] = -2.16983332E+01;
    COFD[4969] = 4.71782117E+00;
    COFD[4970] = -3.31568259E-01;
    COFD[4971] = 1.17540937E-02;
    COFD[4972] = -2.25720229E+01;
    COFD[4973] = 5.27220175E+00;
    COFD[4974] = -4.21436175E-01;
    COFD[4975] = 1.63576263E-02;
    COFD[4976] = -2.25735380E+01;
    COFD[4977] = 5.27888680E+00;
    COFD[4978] = -4.22608469E-01;
    COFD[4979] = 1.64205533E-02;
    COFD[4980] = -2.25785614E+01;
    COFD[4981] = 5.27888680E+00;
    COFD[4982] = -4.22608469E-01;
    COFD[4983] = 1.64205533E-02;
    COFD[4984] = -2.27197587E+01;
    COFD[4985] = 5.29216864E+00;
    COFD[4986] = -4.24828662E-01;
    COFD[4987] = 1.65366835E-02;
    COFD[4988] = -2.16466727E+01;
    COFD[4989] = 4.65048212E+00;
    COFD[4990] = -3.20931552E-01;
    COFD[4991] = 1.12185393E-02;
    COFD[4992] = -2.25387409E+01;
    COFD[4993] = 5.12713969E+00;
    COFD[4994] = -3.96797470E-01;
    COFD[4995] = 1.50599903E-02;
    COFD[4996] = -2.26201345E+01;
    COFD[4997] = 5.16909062E+00;
    COFD[4998] = -4.03789335E-01;
    COFD[4999] = 1.54242019E-02;
    COFD[5000] = -2.20597984E+01;
    COFD[5001] = 4.90985174E+00;
    COFD[5002] = -3.62022555E-01;
    COFD[5003] = 1.32919255E-02;
    COFD[5004] = -2.17488664E+01;
    COFD[5005] = 4.69776924E+00;
    COFD[5006] = -3.28393518E-01;
    COFD[5007] = 1.15940104E-02;
    COFD[5008] = -2.21696464E+01;
    COFD[5009] = 4.81459861E+00;
    COFD[5010] = -3.46990321E-01;
    COFD[5011] = 1.25347154E-02;
    COFD[5012] = -2.21717162E+01;
    COFD[5013] = 4.81459861E+00;
    COFD[5014] = -3.46990321E-01;
    COFD[5015] = 1.25347154E-02;
    COFD[5016] = -2.02840235E+01;
    COFD[5017] = 4.95484018E+00;
    COFD[5018] = -4.23654881E-01;
    COFD[5019] = 1.81813866E-02;
    COFD[5020] = -1.91633071E+01;
    COFD[5021] = 4.70966098E+00;
    COFD[5022] = -3.94551217E-01;
    COFD[5023] = 1.70286289E-02;
    COFD[5024] = -1.55781966E+01;
    COFD[5025] = 3.73153794E+00;
    COFD[5026] = -2.72372598E-01;
    COFD[5027] = 1.19199668E-02;
    COFD[5028] = -1.96804200E+01;
    COFD[5029] = 5.37526595E+00;
    COFD[5030] = -4.70621144E-01;
    COFD[5031] = 1.99141073E-02;
    COFD[5032] = -1.91880377E+01;
    COFD[5033] = 4.70966098E+00;
    COFD[5034] = -3.94551217E-01;
    COFD[5035] = 1.70286289E-02;
    COFD[5036] = -1.98762802E+01;
    COFD[5037] = 4.29984430E+00;
    COFD[5038] = -2.67672378E-01;
    COFD[5039] = 8.61066133E-03;
    COFD[5040] = -2.06463142E+01;
    COFD[5041] = 5.07657482E+00;
    COFD[5042] = -4.38028804E-01;
    COFD[5043] = 1.87481371E-02;
    COFD[5044] = -2.06522508E+01;
    COFD[5045] = 5.07501764E+00;
    COFD[5046] = -4.37846596E-01;
    COFD[5047] = 1.87410133E-02;
    COFD[5048] = -2.06624288E+01;
    COFD[5049] = 5.07501764E+00;
    COFD[5050] = -4.37846596E-01;
    COFD[5051] = 1.87410133E-02;
    COFD[5052] = -2.26749993E+01;
    COFD[5053] = 5.58486459E+00;
    COFD[5054] = -4.81517134E-01;
    COFD[5055] = 1.97388064E-02;
    COFD[5056] = -2.03122895E+01;
    COFD[5057] = 4.96244824E+00;
    COFD[5058] = -4.24554494E-01;
    COFD[5059] = 1.82168885E-02;
    COFD[5060] = -2.11192145E+01;
    COFD[5061] = 4.70311989E+00;
    COFD[5062] = -3.29240106E-01;
    COFD[5063] = 1.16366808E-02;
    COFD[5064] = -2.13459128E+01;
    COFD[5065] = 5.37197338E+00;
    COFD[5066] = -4.70392872E-01;
    COFD[5067] = 1.99122802E-02;
    COFD[5068] = -2.13282915E+01;
    COFD[5069] = 5.36375915E+00;
    COFD[5070] = -4.69808195E-01;
    COFD[5071] = 1.99064589E-02;
    COFD[5072] = -2.12804720E+01;
    COFD[5073] = 4.77238689E+00;
    COFD[5074] = -3.40265855E-01;
    COFD[5075] = 1.21942137E-02;
    COFD[5076] = -2.26853912E+01;
    COFD[5077] = 5.58521030E+00;
    COFD[5078] = -4.81061650E-01;
    COFD[5079] = 1.96992215E-02;
    COFD[5080] = -2.18797352E+01;
    COFD[5081] = 4.99907484E+00;
    COFD[5082] = -3.76094627E-01;
    COFD[5083] = 1.40009262E-02;
    COFD[5084] = -2.19075860E+01;
    COFD[5085] = 4.99907484E+00;
    COFD[5086] = -3.76094627E-01;
    COFD[5087] = 1.40009262E-02;
    COFD[5088] = -2.14039230E+01;
    COFD[5089] = 4.77238689E+00;
    COFD[5090] = -3.40265855E-01;
    COFD[5091] = 1.21942137E-02;
    COFD[5092] = -2.25347527E+01;
    COFD[5093] = 5.57238332E+00;
    COFD[5094] = -4.76605097E-01;
    COFD[5095] = 1.93951822E-02;
    COFD[5096] = -2.26946865E+01;
    COFD[5097] = 5.47392239E+00;
    COFD[5098] = -4.56882004E-01;
    COFD[5099] = 1.82631638E-02;
    COFD[5100] = -2.24423544E+01;
    COFD[5101] = 5.58416166E+00;
    COFD[5102] = -4.82369720E-01;
    COFD[5103] = 1.98133127E-02;
    COFD[5104] = -2.26273108E+01;
    COFD[5105] = 5.42002683E+00;
    COFD[5106] = -4.47111163E-01;
    COFD[5107] = 1.77287360E-02;
    COFD[5108] = -2.26354564E+01;
    COFD[5109] = 5.42002683E+00;
    COFD[5110] = -4.47111163E-01;
    COFD[5111] = 1.77287360E-02;
    COFD[5112] = -2.26677753E+01;
    COFD[5113] = 5.44777353E+00;
    COFD[5114] = -4.52122340E-01;
    COFD[5115] = 1.80021910E-02;
    COFD[5116] = -2.28554026E+01;
    COFD[5117] = 5.48796011E+00;
    COFD[5118] = -4.59457942E-01;
    COFD[5119] = 1.84050728E-02;
    COFD[5120] = -2.17385144E+01;
    COFD[5121] = 4.74350080E+00;
    COFD[5122] = -3.36426340E-01;
    COFD[5123] = 1.20245796E-02;
    COFD[5124] = -2.26508835E+01;
    COFD[5125] = 5.31312101E+00;
    COFD[5126] = -4.28304541E-01;
    COFD[5127] = 1.67176023E-02;
    COFD[5128] = -2.26553469E+01;
    COFD[5129] = 5.32093606E+00;
    COFD[5130] = -4.29624801E-01;
    COFD[5131] = 1.67869730E-02;
    COFD[5132] = -2.26604027E+01;
    COFD[5133] = 5.32093606E+00;
    COFD[5134] = -4.29624801E-01;
    COFD[5135] = 1.67869730E-02;
    COFD[5136] = -2.27940520E+01;
    COFD[5137] = 5.33097599E+00;
    COFD[5138] = -4.31367349E-01;
    COFD[5139] = 1.68798869E-02;
    COFD[5140] = -2.17749249E+01;
    COFD[5141] = 4.71207875E+00;
    COFD[5142] = -3.30658500E-01;
    COFD[5143] = 1.17082011E-02;
    COFD[5144] = -2.26201345E+01;
    COFD[5145] = 5.16909062E+00;
    COFD[5146] = -4.03789335E-01;
    COFD[5147] = 1.54242019E-02;
    COFD[5148] = -2.26855826E+01;
    COFD[5149] = 5.20461425E+00;
    COFD[5150] = -4.09923838E-01;
    COFD[5151] = 1.57504726E-02;
    COFD[5152] = -2.21931069E+01;
    COFD[5153] = 4.97373439E+00;
    COFD[5154] = -3.72089281E-01;
    COFD[5155] = 1.37987774E-02;
    COFD[5156] = -2.18703454E+01;
    COFD[5157] = 4.75653375E+00;
    COFD[5158] = -3.37718114E-01;
    COFD[5159] = 1.20647238E-02;
    COFD[5160] = -2.22981928E+01;
    COFD[5161] = 4.87626494E+00;
    COFD[5162] = -3.56718447E-01;
    COFD[5163] = 1.30246317E-02;
    COFD[5164] = -2.23002803E+01;
    COFD[5165] = 4.87626494E+00;
    COFD[5166] = -3.56718447E-01;
    COFD[5167] = 1.30246317E-02;
    COFD[5168] = -2.06548278E+01;
    COFD[5169] = 5.11678107E+00;
    COFD[5170] = -4.42706538E-01;
    COFD[5171] = 1.89296424E-02;
    COFD[5172] = -1.94778445E+01;
    COFD[5173] = 4.85518471E+00;
    COFD[5174] = -4.11551624E-01;
    COFD[5175] = 1.76895651E-02;
    COFD[5176] = -1.60461372E+01;
    COFD[5177] = 3.95298868E+00;
    COFD[5178] = -3.01302078E-01;
    COFD[5179] = 1.31842095E-02;
    COFD[5180] = -1.98424714E+01;
    COFD[5181] = 5.45215174E+00;
    COFD[5182] = -4.77051991E-01;
    COFD[5183] = 2.00510347E-02;
    COFD[5184] = -1.95026421E+01;
    COFD[5185] = 4.85518471E+00;
    COFD[5186] = -4.11551624E-01;
    COFD[5187] = 1.76895651E-02;
    COFD[5188] = -1.90375666E+01;
    COFD[5189] = 3.93604965E+00;
    COFD[5190] = -2.11360409E-01;
    COFD[5191] = 5.81247394E-03;
    COFD[5192] = -2.09258526E+01;
    COFD[5193] = 5.19811866E+00;
    COFD[5194] = -4.51121211E-01;
    COFD[5195] = 1.92074617E-02;
    COFD[5196] = -2.09364971E+01;
    COFD[5197] = 5.19811866E+00;
    COFD[5198] = -4.51121211E-01;
    COFD[5199] = 1.92074617E-02;
    COFD[5200] = -2.09467220E+01;
    COFD[5201] = 5.19811866E+00;
    COFD[5202] = -4.51121211E-01;
    COFD[5203] = 1.92074617E-02;
    COFD[5204] = -2.25786655E+01;
    COFD[5205] = 5.53409384E+00;
    COFD[5206] = -4.69342499E-01;
    COFD[5207] = 1.89886374E-02;
    COFD[5208] = -2.06812067E+01;
    COFD[5209] = 5.12346096E+00;
    COFD[5210] = -4.43477411E-01;
    COFD[5211] = 1.89592529E-02;
    COFD[5212] = -2.03970537E+01;
    COFD[5213] = 4.38396848E+00;
    COFD[5214] = -2.79298901E-01;
    COFD[5215] = 9.13915001E-03;
    COFD[5216] = -2.15208595E+01;
    COFD[5217] = 5.44385051E+00;
    COFD[5218] = -4.76121506E-01;
    COFD[5219] = 2.00164081E-02;
    COFD[5220] = -2.14671205E+01;
    COFD[5221] = 5.42109069E+00;
    COFD[5222] = -4.73533096E-01;
    COFD[5223] = 1.99183547E-02;
    COFD[5224] = -2.06103015E+01;
    COFD[5225] = 4.47491202E+00;
    COFD[5226] = -2.93331059E-01;
    COFD[5227] = 9.83445305E-03;
    COFD[5228] = -2.25695574E+01;
    COFD[5229] = 5.52323975E+00;
    COFD[5230] = -4.67257607E-01;
    COFD[5231] = 1.88711975E-02;
    COFD[5232] = -2.12221678E+01;
    COFD[5233] = 4.70506024E+00;
    COFD[5234] = -3.29547212E-01;
    COFD[5235] = 1.16521630E-02;
    COFD[5236] = -2.12501716E+01;
    COFD[5237] = 4.70506024E+00;
    COFD[5238] = -3.29547212E-01;
    COFD[5239] = 1.16521630E-02;
    COFD[5240] = -2.07343778E+01;
    COFD[5241] = 4.47491202E+00;
    COFD[5242] = -2.93331059E-01;
    COFD[5243] = 9.83445305E-03;
    COFD[5244] = -2.23655523E+01;
    COFD[5245] = 5.48956505E+00;
    COFD[5246] = -4.59770566E-01;
    COFD[5247] = 1.84227929E-02;
    COFD[5248] = -2.23996701E+01;
    COFD[5249] = 5.33372666E+00;
    COFD[5250] = -4.31837946E-01;
    COFD[5251] = 1.69048117E-02;
    COFD[5252] = -2.23867276E+01;
    COFD[5253] = 5.55175851E+00;
    COFD[5254] = -4.72720598E-01;
    COFD[5255] = 1.91783487E-02;
    COFD[5256] = -2.22858832E+01;
    COFD[5257] = 5.25941804E+00;
    COFD[5258] = -4.19208672E-01;
    COFD[5259] = 1.62385114E-02;
    COFD[5260] = -2.22940707E+01;
    COFD[5261] = 5.25941804E+00;
    COFD[5262] = -4.19208672E-01;
    COFD[5263] = 1.62385114E-02;
    COFD[5264] = -2.23480908E+01;
    COFD[5265] = 5.29695321E+00;
    COFD[5266] = -4.25620113E-01;
    COFD[5267] = 1.65778213E-02;
    COFD[5268] = -2.25780442E+01;
    COFD[5269] = 5.35238497E+00;
    COFD[5270] = -4.35034945E-01;
    COFD[5271] = 1.70742216E-02;
    COFD[5272] = -2.11585495E+01;
    COFD[5273] = 4.47646812E+00;
    COFD[5274] = -2.93573165E-01;
    COFD[5275] = 9.84650920E-03;
    COFD[5276] = -2.22582201E+01;
    COFD[5277] = 5.12825866E+00;
    COFD[5278] = -3.96982702E-01;
    COFD[5279] = 1.50696010E-02;
    COFD[5280] = -2.22649250E+01;
    COFD[5281] = 5.13737172E+00;
    COFD[5282] = -3.98493102E-01;
    COFD[5283] = 1.51480250E-02;
    COFD[5284] = -2.22700127E+01;
    COFD[5285] = 5.13737172E+00;
    COFD[5286] = -3.98493102E-01;
    COFD[5287] = 1.51480250E-02;
    COFD[5288] = -2.24171125E+01;
    COFD[5289] = 5.15109664E+00;
    COFD[5290] = -4.00765892E-01;
    COFD[5291] = 1.52659560E-02;
    COFD[5292] = -2.10757112E+01;
    COFD[5293] = 4.39521460E+00;
    COFD[5294] = -2.81028854E-01;
    COFD[5295] = 9.22466916E-03;
    COFD[5296] = -2.20597984E+01;
    COFD[5297] = 4.90985174E+00;
    COFD[5298] = -3.62022555E-01;
    COFD[5299] = 1.32919255E-02;
    COFD[5300] = -2.21931069E+01;
    COFD[5301] = 4.97373439E+00;
    COFD[5302] = -3.72089281E-01;
    COFD[5303] = 1.37987774E-02;
    COFD[5304] = -2.15488348E+01;
    COFD[5305] = 4.68075977E+00;
    COFD[5306] = -3.25704931E-01;
    COFD[5307] = 1.14585845E-02;
    COFD[5308] = -2.12063903E+01;
    COFD[5309] = 4.45414225E+00;
    COFD[5310] = -2.90099937E-01;
    COFD[5311] = 9.67360320E-03;
    COFD[5312] = -2.16820603E+01;
    COFD[5313] = 4.59101412E+00;
    COFD[5314] = -3.11439033E-01;
    COFD[5315] = 1.07377082E-02;
    COFD[5316] = -2.16841653E+01;
    COFD[5317] = 4.59101412E+00;
    COFD[5318] = -3.11439033E-01;
    COFD[5319] = 1.07377082E-02;
    COFD[5320] = -2.10086887E+01;
    COFD[5321] = 5.19953529E+00;
    COFD[5322] = -4.51287802E-01;
    COFD[5323] = 1.92140123E-02;
    COFD[5324] = -1.99562868E+01;
    COFD[5325] = 4.99367362E+00;
    COFD[5326] = -4.28249956E-01;
    COFD[5327] = 1.83628509E-02;
    COFD[5328] = -1.64639359E+01;
    COFD[5329] = 4.08142484E+00;
    COFD[5330] = -3.17696496E-01;
    COFD[5331] = 1.38856294E-02;
    COFD[5332] = -2.02451923E+01;
    COFD[5333] = 5.55377454E+00;
    COFD[5334] = -4.87810074E-01;
    COFD[5335] = 2.04217376E-02;
    COFD[5336] = -1.99818280E+01;
    COFD[5337] = 4.99367362E+00;
    COFD[5338] = -4.28249956E-01;
    COFD[5339] = 1.83628509E-02;
    COFD[5340] = -1.85767826E+01;
    COFD[5341] = 3.66420353E+00;
    COFD[5342] = -1.69810177E-01;
    COFD[5343] = 3.77247849E-03;
    COFD[5344] = -2.14057339E+01;
    COFD[5345] = 5.33269880E+00;
    COFD[5346] = -4.67008439E-01;
    COFD[5347] = 1.98347416E-02;
    COFD[5348] = -2.14169211E+01;
    COFD[5349] = 5.33269880E+00;
    COFD[5350] = -4.67008439E-01;
    COFD[5351] = 1.98347416E-02;
    COFD[5352] = -2.14276788E+01;
    COFD[5353] = 5.33269880E+00;
    COFD[5354] = -4.67008439E-01;
    COFD[5355] = 1.98347416E-02;
    COFD[5356] = -2.26305728E+01;
    COFD[5357] = 5.47666967E+00;
    COFD[5358] = -4.57381900E-01;
    COFD[5359] = 1.82905822E-02;
    COFD[5360] = -2.10372026E+01;
    COFD[5361] = 5.20711052E+00;
    COFD[5362] = -4.52173945E-01;
    COFD[5363] = 1.92486273E-02;
    COFD[5364] = -1.98603655E+01;
    COFD[5365] = 4.07958166E+00;
    COFD[5366] = -2.33006871E-01;
    COFD[5367] = 6.86822015E-03;
    COFD[5368] = -2.19256706E+01;
    COFD[5369] = 5.54768472E+00;
    COFD[5370] = -4.87202065E-01;
    COFD[5371] = 2.04025437E-02;
    COFD[5372] = -2.18876256E+01;
    COFD[5373] = 5.53154746E+00;
    COFD[5374] = -4.85594344E-01;
    COFD[5375] = 2.03520324E-02;
    COFD[5376] = -2.01250987E+01;
    COFD[5377] = 4.19160608E+00;
    COFD[5378] = -2.49936771E-01;
    COFD[5379] = 7.69538319E-03;
    COFD[5380] = -2.26027431E+01;
    COFD[5381] = 5.46217527E+00;
    COFD[5382] = -4.54751471E-01;
    COFD[5383] = 1.81465218E-02;
    COFD[5384] = -2.08821587E+01;
    COFD[5385] = 4.48108132E+00;
    COFD[5386] = -2.94289899E-01;
    COFD[5387] = 9.88218297E-03;
    COFD[5388] = -2.09119213E+01;
    COFD[5389] = 4.48108132E+00;
    COFD[5390] = -2.94289899E-01;
    COFD[5391] = 9.88218297E-03;
    COFD[5392] = -2.02563322E+01;
    COFD[5393] = 4.19160608E+00;
    COFD[5394] = -2.49936771E-01;
    COFD[5395] = 7.69538319E-03;
    COFD[5396] = -2.23265991E+01;
    COFD[5397] = 5.39645154E+00;
    COFD[5398] = -4.42708323E-01;
    COFD[5399] = 1.74846134E-02;
    COFD[5400] = -2.22885235E+01;
    COFD[5401] = 5.20764658E+00;
    COFD[5402] = -4.10207913E-01;
    COFD[5403] = 1.57585882E-02;
    COFD[5404] = -2.24356779E+01;
    COFD[5405] = 5.49613266E+00;
    COFD[5406] = -4.61060586E-01;
    COFD[5407] = 1.84960110E-02;
    COFD[5408] = -2.21494624E+01;
    COFD[5409] = 5.12338366E+00;
    COFD[5410] = -3.96176894E-01;
    COFD[5411] = 1.50278196E-02;
    COFD[5412] = -2.21581289E+01;
    COFD[5413] = 5.12338366E+00;
    COFD[5414] = -3.96176894E-01;
    COFD[5415] = 1.50278196E-02;
    COFD[5416] = -2.22256643E+01;
    COFD[5417] = 5.16620234E+00;
    COFD[5418] = -4.03306755E-01;
    COFD[5419] = 1.53990058E-02;
    COFD[5420] = -2.24631694E+01;
    COFD[5421] = 5.22623384E+00;
    COFD[5422] = -4.13380324E-01;
    COFD[5423] = 1.59259437E-02;
    COFD[5424] = -2.06827490E+01;
    COFD[5425] = 4.19375892E+00;
    COFD[5426] = -2.50262428E-01;
    COFD[5427] = 7.71131487E-03;
    COFD[5428] = -2.19774160E+01;
    COFD[5429] = 4.92889157E+00;
    COFD[5430] = -3.65025286E-01;
    COFD[5431] = 1.34431452E-02;
    COFD[5432] = -2.19948202E+01;
    COFD[5433] = 4.94219368E+00;
    COFD[5434] = -3.67120797E-01;
    COFD[5435] = 1.35486343E-02;
    COFD[5436] = -2.20002783E+01;
    COFD[5437] = 4.94219368E+00;
    COFD[5438] = -3.67120797E-01;
    COFD[5439] = 1.35486343E-02;
    COFD[5440] = -2.21597603E+01;
    COFD[5441] = 4.96243463E+00;
    COFD[5442] = -3.70309018E-01;
    COFD[5443] = 1.37091432E-02;
    COFD[5444] = -2.05583124E+01;
    COFD[5445] = 4.09420232E+00;
    COFD[5446] = -2.35210019E-01;
    COFD[5447] = 6.97573395E-03;
    COFD[5448] = -2.17488664E+01;
    COFD[5449] = 4.69776924E+00;
    COFD[5450] = -3.28393518E-01;
    COFD[5451] = 1.15940104E-02;
    COFD[5452] = -2.18703454E+01;
    COFD[5453] = 4.75653375E+00;
    COFD[5454] = -3.37718114E-01;
    COFD[5455] = 1.20647238E-02;
    COFD[5456] = -2.12063903E+01;
    COFD[5457] = 4.45414225E+00;
    COFD[5458] = -2.90099937E-01;
    COFD[5459] = 9.67360320E-03;
    COFD[5460] = -2.07107178E+01;
    COFD[5461] = 4.16274636E+00;
    COFD[5462] = -2.45571617E-01;
    COFD[5463] = 7.48187191E-03;
    COFD[5464] = -2.12713524E+01;
    COFD[5465] = 4.33544466E+00;
    COFD[5466] = -2.71843874E-01;
    COFD[5467] = 8.77093391E-03;
    COFD[5468] = -2.12736663E+01;
    COFD[5469] = 4.33544466E+00;
    COFD[5470] = -2.71843874E-01;
    COFD[5471] = 8.77093391E-03;
    COFD[5472] = -2.10674485E+01;
    COFD[5473] = 5.15027524E+00;
    COFD[5474] = -4.46126111E-01;
    COFD[5475] = 1.90401391E-02;
    COFD[5476] = -1.99785176E+01;
    COFD[5477] = 4.92184026E+00;
    COFD[5478] = -4.19745472E-01;
    COFD[5479] = 1.80268154E-02;
    COFD[5480] = -1.64898528E+01;
    COFD[5481] = 4.01175649E+00;
    COFD[5482] = -3.08860971E-01;
    COFD[5483] = 1.35100076E-02;
    COFD[5484] = -2.03113704E+01;
    COFD[5485] = 5.50136606E+00;
    COFD[5486] = -4.82461887E-01;
    COFD[5487] = 2.02471523E-02;
    COFD[5488] = -2.00047095E+01;
    COFD[5489] = 4.92184026E+00;
    COFD[5490] = -4.19745472E-01;
    COFD[5491] = 1.80268154E-02;
    COFD[5492] = -1.91326792E+01;
    COFD[5493] = 3.82263611E+00;
    COFD[5494] = -1.93983472E-01;
    COFD[5495] = 4.95789388E-03;
    COFD[5496] = -2.13955999E+01;
    COFD[5497] = 5.25183817E+00;
    COFD[5498] = -4.57376333E-01;
    COFD[5499] = 1.94504429E-02;
    COFD[5500] = -2.14072803E+01;
    COFD[5501] = 5.25183817E+00;
    COFD[5502] = -4.57376333E-01;
    COFD[5503] = 1.94504429E-02;
    COFD[5504] = -2.14185232E+01;
    COFD[5505] = 5.25183817E+00;
    COFD[5506] = -4.57376333E-01;
    COFD[5507] = 1.94504429E-02;
    COFD[5508] = -2.28655752E+01;
    COFD[5509] = 5.50522401E+00;
    COFD[5510] = -4.63604304E-01;
    COFD[5511] = 1.86600785E-02;
    COFD[5512] = -2.10844012E+01;
    COFD[5513] = 5.15315713E+00;
    COFD[5514] = -4.46344043E-01;
    COFD[5515] = 1.90431546E-02;
    COFD[5516] = -2.04620510E+01;
    COFD[5517] = 4.26473557E+00;
    COFD[5518] = -2.61033037E-01;
    COFD[5519] = 8.23906412E-03;
    COFD[5520] = -2.19550907E+01;
    COFD[5521] = 5.49350509E+00;
    COFD[5522] = -4.81613405E-01;
    COFD[5523] = 2.02171734E-02;
    COFD[5524] = -2.19053841E+01;
    COFD[5525] = 5.47162499E+00;
    COFD[5526] = -4.79195552E-01;
    COFD[5527] = 2.01289088E-02;
    COFD[5528] = -2.06858147E+01;
    COFD[5529] = 4.35920123E+00;
    COFD[5530] = -2.75491273E-01;
    COFD[5531] = 8.95100289E-03;
    COFD[5532] = -2.28446667E+01;
    COFD[5533] = 5.50134401E+00;
    COFD[5534] = -4.62488197E-01;
    COFD[5535] = 1.85873697E-02;
    COFD[5536] = -2.13524540E+01;
    COFD[5537] = 4.61201872E+00;
    COFD[5538] = -3.14803338E-01;
    COFD[5539] = 1.09082984E-02;
    COFD[5540] = -2.13838498E+01;
    COFD[5541] = 4.61201872E+00;
    COFD[5542] = -3.14803338E-01;
    COFD[5543] = 1.09082984E-02;
    COFD[5544] = -2.08236367E+01;
    COFD[5545] = 4.35920123E+00;
    COFD[5546] = -2.75491273E-01;
    COFD[5547] = 8.95100289E-03;
    COFD[5548] = -2.26089431E+01;
    COFD[5549] = 5.44867280E+00;
    COFD[5550] = -4.52284883E-01;
    COFD[5551] = 1.80110706E-02;
    COFD[5552] = -2.26029886E+01;
    COFD[5553] = 5.27383847E+00;
    COFD[5554] = -4.21722368E-01;
    COFD[5555] = 1.63729618E-02;
    COFD[5556] = -2.26579938E+01;
    COFD[5557] = 5.52001624E+00;
    COFD[5558] = -4.66629503E-01;
    COFD[5559] = 1.88355817E-02;
    COFD[5560] = -2.25069737E+01;
    COFD[5561] = 5.21003123E+00;
    COFD[5562] = -4.10612564E-01;
    COFD[5563] = 1.57798598E-02;
    COFD[5564] = -2.25160816E+01;
    COFD[5565] = 5.21003123E+00;
    COFD[5566] = -4.10612564E-01;
    COFD[5567] = 1.57798598E-02;
    COFD[5568] = -2.25635595E+01;
    COFD[5569] = 5.24330646E+00;
    COFD[5570] = -4.16370120E-01;
    COFD[5571] = 1.60860486E-02;
    COFD[5572] = -2.27715883E+01;
    COFD[5573] = 5.29493402E+00;
    COFD[5574] = -4.25285978E-01;
    COFD[5575] = 1.65604533E-02;
    COFD[5576] = -2.12332312E+01;
    COFD[5577] = 4.36095377E+00;
    COFD[5578] = -2.75760539E-01;
    COFD[5579] = 8.96430249E-03;
    COFD[5580] = -2.24161979E+01;
    COFD[5581] = 5.05061421E+00;
    COFD[5582] = -3.84359196E-01;
    COFD[5583] = 1.44214004E-02;
    COFD[5584] = -2.24284563E+01;
    COFD[5585] = 5.06106414E+00;
    COFD[5586] = -3.86053039E-01;
    COFD[5587] = 1.45081784E-02;
    COFD[5588] = -2.24342646E+01;
    COFD[5589] = 5.06106414E+00;
    COFD[5590] = -3.86053039E-01;
    COFD[5591] = 1.45081784E-02;
    COFD[5592] = -2.25733338E+01;
    COFD[5593] = 5.07648425E+00;
    COFD[5594] = -3.88560019E-01;
    COFD[5595] = 1.46368353E-02;
    COFD[5596] = -2.11438235E+01;
    COFD[5597] = 4.27612828E+00;
    COFD[5598] = -2.62774610E-01;
    COFD[5599] = 8.32471127E-03;
    COFD[5600] = -2.21696464E+01;
    COFD[5601] = 4.81459861E+00;
    COFD[5602] = -3.46990321E-01;
    COFD[5603] = 1.25347154E-02;
    COFD[5604] = -2.22981928E+01;
    COFD[5605] = 4.87626494E+00;
    COFD[5606] = -3.56718447E-01;
    COFD[5607] = 1.30246317E-02;
    COFD[5608] = -2.16820603E+01;
    COFD[5609] = 4.59101412E+00;
    COFD[5610] = -3.11439033E-01;
    COFD[5611] = 1.07377082E-02;
    COFD[5612] = -2.12713524E+01;
    COFD[5613] = 4.33544466E+00;
    COFD[5614] = -2.71843874E-01;
    COFD[5615] = 8.77093391E-03;
    COFD[5616] = -2.17746534E+01;
    COFD[5617] = 4.48837319E+00;
    COFD[5618] = -2.95423315E-01;
    COFD[5619] = 9.93861345E-03;
    COFD[5620] = -2.17771745E+01;
    COFD[5621] = 4.48837319E+00;
    COFD[5622] = -2.95423315E-01;
    COFD[5623] = 9.93861345E-03;
    COFD[5624] = -2.10685573E+01;
    COFD[5625] = 5.15027524E+00;
    COFD[5626] = -4.46126111E-01;
    COFD[5627] = 1.90401391E-02;
    COFD[5628] = -1.99792167E+01;
    COFD[5629] = 4.92184026E+00;
    COFD[5630] = -4.19745472E-01;
    COFD[5631] = 1.80268154E-02;
    COFD[5632] = -1.64899530E+01;
    COFD[5633] = 4.01175649E+00;
    COFD[5634] = -3.08860971E-01;
    COFD[5635] = 1.35100076E-02;
    COFD[5636] = -2.03114210E+01;
    COFD[5637] = 5.50136606E+00;
    COFD[5638] = -4.82461887E-01;
    COFD[5639] = 2.02471523E-02;
    COFD[5640] = -2.00054461E+01;
    COFD[5641] = 4.92184026E+00;
    COFD[5642] = -4.19745472E-01;
    COFD[5643] = 1.80268154E-02;
    COFD[5644] = -1.91334529E+01;
    COFD[5645] = 3.82263611E+00;
    COFD[5646] = -1.93983472E-01;
    COFD[5647] = 4.95789388E-03;
    COFD[5648] = -2.13968281E+01;
    COFD[5649] = 5.25183817E+00;
    COFD[5650] = -4.57376333E-01;
    COFD[5651] = 1.94504429E-02;
    COFD[5652] = -2.14085375E+01;
    COFD[5653] = 5.25183817E+00;
    COFD[5654] = -4.57376333E-01;
    COFD[5655] = 1.94504429E-02;
    COFD[5656] = -2.14198091E+01;
    COFD[5657] = 5.25183817E+00;
    COFD[5658] = -4.57376333E-01;
    COFD[5659] = 1.94504429E-02;
    COFD[5660] = -2.28671232E+01;
    COFD[5661] = 5.50522401E+00;
    COFD[5662] = -4.63604304E-01;
    COFD[5663] = 1.86600785E-02;
    COFD[5664] = -2.10855099E+01;
    COFD[5665] = 5.15315713E+00;
    COFD[5666] = -4.46344043E-01;
    COFD[5667] = 1.90431546E-02;
    COFD[5668] = -2.04632210E+01;
    COFD[5669] = 4.26473557E+00;
    COFD[5670] = -2.61033037E-01;
    COFD[5671] = 8.23906412E-03;
    COFD[5672] = -2.19557531E+01;
    COFD[5673] = 5.49350509E+00;
    COFD[5674] = -4.81613405E-01;
    COFD[5675] = 2.02171734E-02;
    COFD[5676] = -2.19060847E+01;
    COFD[5677] = 5.47162499E+00;
    COFD[5678] = -4.79195552E-01;
    COFD[5679] = 2.01289088E-02;
    COFD[5680] = -2.06870442E+01;
    COFD[5681] = 4.35920123E+00;
    COFD[5682] = -2.75491273E-01;
    COFD[5683] = 8.95100289E-03;
    COFD[5684] = -2.28458380E+01;
    COFD[5685] = 5.50134401E+00;
    COFD[5686] = -4.62488197E-01;
    COFD[5687] = 1.85873697E-02;
    COFD[5688] = -2.13539532E+01;
    COFD[5689] = 4.61201872E+00;
    COFD[5690] = -3.14803338E-01;
    COFD[5691] = 1.09082984E-02;
    COFD[5692] = -2.13854464E+01;
    COFD[5693] = 4.61201872E+00;
    COFD[5694] = -3.14803338E-01;
    COFD[5695] = 1.09082984E-02;
    COFD[5696] = -2.08252570E+01;
    COFD[5697] = 4.35920123E+00;
    COFD[5698] = -2.75491273E-01;
    COFD[5699] = 8.95100289E-03;
    COFD[5700] = -2.26099899E+01;
    COFD[5701] = 5.44867280E+00;
    COFD[5702] = -4.52284883E-01;
    COFD[5703] = 1.80110706E-02;
    COFD[5704] = -2.26044889E+01;
    COFD[5705] = 5.27383847E+00;
    COFD[5706] = -4.21722368E-01;
    COFD[5707] = 1.63729618E-02;
    COFD[5708] = -2.26591038E+01;
    COFD[5709] = 5.52001624E+00;
    COFD[5710] = -4.66629503E-01;
    COFD[5711] = 1.88355817E-02;
    COFD[5712] = -2.25083966E+01;
    COFD[5713] = 5.21003123E+00;
    COFD[5714] = -4.10612564E-01;
    COFD[5715] = 1.57798598E-02;
    COFD[5716] = -2.25175307E+01;
    COFD[5717] = 5.21003123E+00;
    COFD[5718] = -4.10612564E-01;
    COFD[5719] = 1.57798598E-02;
    COFD[5720] = -2.25650343E+01;
    COFD[5721] = 5.24330646E+00;
    COFD[5722] = -4.16370120E-01;
    COFD[5723] = 1.60860486E-02;
    COFD[5724] = -2.27731137E+01;
    COFD[5725] = 5.29493402E+00;
    COFD[5726] = -4.25285978E-01;
    COFD[5727] = 1.65604533E-02;
    COFD[5728] = -2.12354028E+01;
    COFD[5729] = 4.36095377E+00;
    COFD[5730] = -2.75760539E-01;
    COFD[5731] = 8.96430249E-03;
    COFD[5732] = -2.24179759E+01;
    COFD[5733] = 5.05061421E+00;
    COFD[5734] = -3.84359196E-01;
    COFD[5735] = 1.44214004E-02;
    COFD[5736] = -2.24302556E+01;
    COFD[5737] = 5.06106414E+00;
    COFD[5738] = -3.86053039E-01;
    COFD[5739] = 1.45081784E-02;
    COFD[5740] = -2.24360850E+01;
    COFD[5741] = 5.06106414E+00;
    COFD[5742] = -3.86053039E-01;
    COFD[5743] = 1.45081784E-02;
    COFD[5744] = -2.25751750E+01;
    COFD[5745] = 5.07648425E+00;
    COFD[5746] = -3.88560019E-01;
    COFD[5747] = 1.46368353E-02;
    COFD[5748] = -2.11462093E+01;
    COFD[5749] = 4.27612828E+00;
    COFD[5750] = -2.62774610E-01;
    COFD[5751] = 8.32471127E-03;
    COFD[5752] = -2.21717162E+01;
    COFD[5753] = 4.81459861E+00;
    COFD[5754] = -3.46990321E-01;
    COFD[5755] = 1.25347154E-02;
    COFD[5756] = -2.23002803E+01;
    COFD[5757] = 4.87626494E+00;
    COFD[5758] = -3.56718447E-01;
    COFD[5759] = 1.30246317E-02;
    COFD[5760] = -2.16841653E+01;
    COFD[5761] = 4.59101412E+00;
    COFD[5762] = -3.11439033E-01;
    COFD[5763] = 1.07377082E-02;
    COFD[5764] = -2.12736663E+01;
    COFD[5765] = 4.33544466E+00;
    COFD[5766] = -2.71843874E-01;
    COFD[5767] = 8.77093391E-03;
    COFD[5768] = -2.17771745E+01;
    COFD[5769] = 4.48837319E+00;
    COFD[5770] = -2.95423315E-01;
    COFD[5771] = 9.93861345E-03;
    COFD[5772] = -2.17797084E+01;
    COFD[5773] = 4.48837319E+00;
    COFD[5774] = -2.95423315E-01;
    COFD[5775] = 9.93861345E-03;
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
    COFTD[72] = 1.36817715E-01;
    COFTD[73] = 6.41727473E-04;
    COFTD[74] = -3.42055963E-07;
    COFTD[75] = 5.68567648E-11;
    COFTD[76] = 2.58066832E-01;
    COFTD[77] = 4.05072593E-04;
    COFTD[78] = -2.30587443E-07;
    COFTD[79] = 4.01863841E-11;
    COFTD[80] = 2.40639006E-01;
    COFTD[81] = 4.82930111E-04;
    COFTD[82] = -2.70362190E-07;
    COFTD[83] = 4.65173265E-11;
    COFTD[84] = 2.82974392E-01;
    COFTD[85] = 3.73032949E-04;
    COFTD[86] = -2.14959161E-07;
    COFTD[87] = 3.78355155E-11;
    COFTD[88] = 2.26670609E-01;
    COFTD[89] = 4.98251023E-04;
    COFTD[90] = -2.77281385E-07;
    COFTD[91] = 4.74970799E-11;
    COFTD[92] = 2.27261590E-01;
    COFTD[93] = 4.99550076E-04;
    COFTD[94] = -2.78004320E-07;
    COFTD[95] = 4.76209155E-11;
    COFTD[96] = 2.34098762E-01;
    COFTD[97] = 4.91099181E-04;
    COFTD[98] = -2.74133967E-07;
    COFTD[99] = 4.70636702E-11;
    COFTD[100] = 2.44452926E-01;
    COFTD[101] = 4.78884724E-04;
    COFTD[102] = -2.68527379E-07;
    COFTD[103] = 4.62572763E-11;
    COFTD[104] = 1.31648645E-01;
    COFTD[105] = 6.75329826E-04;
    COFTD[106] = -3.58458833E-07;
    COFTD[107] = 5.94176903E-11;
    COFTD[108] = 2.10934836E-01;
    COFTD[109] = 5.46607649E-04;
    COFTD[110] = -3.01041232E-07;
    COFTD[111] = 5.11789725E-11;
    COFTD[112] = 2.12562541E-01;
    COFTD[113] = 5.45357255E-04;
    COFTD[114] = -3.00537881E-07;
    COFTD[115] = 5.11159625E-11;
    COFTD[116] = 2.12842514E-01;
    COFTD[117] = 5.46075564E-04;
    COFTD[118] = -3.00933730E-07;
    COFTD[119] = 5.11832891E-11;
    COFTD[120] = 2.15139505E-01;
    COFTD[121] = 5.43740408E-04;
    COFTD[122] = -2.99926299E-07;
    COFTD[123] = 5.10460631E-11;
    COFTD[124] = 1.35158514E-01;
    COFTD[125] = 6.77932393E-04;
    COFTD[126] = -3.60212591E-07;
    COFTD[127] = 5.97492207E-11;
    COFTD[128] = 1.88615713E-01;
    COFTD[129] = 5.93233141E-04;
    COFTD[130] = -3.22963156E-07;
    COFTD[131] = 5.44599859E-11;
    COFTD[132] = 1.94963988E-01;
    COFTD[133] = 5.84809248E-04;
    COFTD[134] = -3.19255828E-07;
    COFTD[135] = 5.39384246E-11;
    COFTD[136] = 1.62995868E-01;
    COFTD[137] = 6.30693413E-04;
    COFTD[138] = -3.39450362E-07;
    COFTD[139] = 5.67923159E-11;
    COFTD[140] = 1.39990702E-01;
    COFTD[141] = 6.69283762E-04;
    COFTD[142] = -3.56418508E-07;
    COFTD[143] = 5.92081617E-11;
    COFTD[144] = 1.55057696E-01;
    COFTD[145] = 6.55342553E-04;
    COFTD[146] = -3.51111538E-07;
    COFTD[147] = 5.85627121E-11;
    COFTD[148] = 1.55121130E-01;
    COFTD[149] = 6.55610653E-04;
    COFTD[150] = -3.51255177E-07;
    COFTD[151] = 5.85866700E-11;
    COFTD[152] = 2.01521643E-01;
    COFTD[153] = 5.62744089E-04;
    COFTD[154] = -3.08519239E-07;
    COFTD[155] = 5.22805986E-11;
    COFTD[156] = 2.35283119E-01;
    COFTD[157] = 4.65670599E-04;
    COFTD[158] = -2.60939824E-07;
    COFTD[159] = 4.49271822E-11;
    COFTD[160] = 1.44152190E-01;
    COFTD[161] = 7.99993584E-05;
    COFTD[162] = -4.89707442E-08;
    COFTD[163] = 9.14277269E-12;
    COFTD[164] = 0.00000000E+00;
    COFTD[165] = 0.00000000E+00;
    COFTD[166] = 0.00000000E+00;
    COFTD[167] = 0.00000000E+00;
    COFTD[168] = 2.37053352E-01;
    COFTD[169] = 4.69174231E-04;
    COFTD[170] = -2.62903094E-07;
    COFTD[171] = 4.52652072E-11;
    COFTD[172] = -1.74352698E-01;
    COFTD[173] = 8.62246873E-04;
    COFTD[174] = -3.79545489E-07;
    COFTD[175] = 5.60262093E-11;
    COFTD[176] = 1.79840299E-01;
    COFTD[177] = 6.01722902E-04;
    COFTD[178] = -3.26433894E-07;
    COFTD[179] = 5.49112302E-11;
    COFTD[180] = 1.80186965E-01;
    COFTD[181] = 6.02882805E-04;
    COFTD[182] = -3.27063140E-07;
    COFTD[183] = 5.50170790E-11;
    COFTD[184] = 1.80513677E-01;
    COFTD[185] = 6.03975942E-04;
    COFTD[186] = -3.27656165E-07;
    COFTD[187] = 5.51168351E-11;
    COFTD[188] = -2.00309448E-02;
    COFTD[189] = 8.50440115E-04;
    COFTD[190] = -4.21064468E-07;
    COFTD[191] = 6.67959710E-11;
    COFTD[192] = 2.00119897E-01;
    COFTD[193] = 5.64793704E-04;
    COFTD[194] = -3.09445484E-07;
    COFTD[195] = 5.24139335E-11;
    COFTD[196] = -1.61357564E-01;
    COFTD[197] = 9.05920260E-04;
    COFTD[198] = -4.07879153E-07;
    COFTD[199] = 6.10626290E-11;
    COFTD[200] = 1.00039110E-01;
    COFTD[201] = 6.50468660E-04;
    COFTD[202] = -3.41778999E-07;
    COFTD[203] = 5.62779132E-11;
    COFTD[204] = 1.05124122E-01;
    COFTD[205] = 6.50665957E-04;
    COFTD[206] = -3.42564538E-07;
    COFTD[207] = 5.64804120E-11;
    COFTD[208] = -1.56651581E-01;
    COFTD[209] = 9.09789751E-04;
    COFTD[210] = -4.11714242E-07;
    COFTD[211] = 6.18310893E-11;
    COFTD[212] = -2.28637575E-02;
    COFTD[213] = 8.35412914E-04;
    COFTD[214] = -4.12929260E-07;
    COFTD[215] = 6.54380945E-11;
    COFTD[216] = -1.41640506E-01;
    COFTD[217] = 9.21404324E-04;
    COFTD[218] = -4.23210110E-07;
    COFTD[219] = 6.41400322E-11;
    COFTD[220] = -1.42230624E-01;
    COFTD[221] = 9.25243177E-04;
    COFTD[222] = -4.24973333E-07;
    COFTD[223] = 6.44072593E-11;
    COFTD[224] = -1.59826932E-01;
    COFTD[225] = 9.28231324E-04;
    COFTD[226] = -4.20059750E-07;
    COFTD[227] = 6.30844146E-11;
    COFTD[228] = -3.81470765E-02;
    COFTD[229] = 8.39833490E-04;
    COFTD[230] = -4.11688915E-07;
    COFTD[231] = 6.49124952E-11;
    COFTD[232] = -7.23038994E-02;
    COFTD[233] = 8.89466098E-04;
    COFTD[234] = -4.28124818E-07;
    COFTD[235] = 6.67586244E-11;
    COFTD[236] = -1.42100396E-02;
    COFTD[237] = 8.23812102E-04;
    COFTD[238] = -4.08995515E-07;
    COFTD[239] = 6.49899310E-11;
    COFTD[240] = -8.34877147E-02;
    COFTD[241] = 8.93466011E-04;
    COFTD[242] = -4.27125851E-07;
    COFTD[243] = 6.63277969E-11;
    COFTD[244] = -8.35962674E-02;
    COFTD[245] = 8.94627716E-04;
    COFTD[246] = -4.27681210E-07;
    COFTD[247] = 6.64140378E-11;
    COFTD[248] = -7.78657454E-02;
    COFTD[249] = 8.92101015E-04;
    COFTD[250] = -4.27969255E-07;
    COFTD[251] = 6.66000503E-11;
    COFTD[252] = -6.92602151E-02;
    COFTD[253] = 8.88360172E-04;
    COFTD[254] = -4.28365765E-07;
    COFTD[255] = 6.68694606E-11;
    COFTD[256] = -1.62301128E-01;
    COFTD[257] = 9.43217155E-04;
    COFTD[258] = -4.26881994E-07;
    COFTD[259] = 6.41127358E-11;
    COFTD[260] = -1.04463705E-01;
    COFTD[261] = 9.17317898E-04;
    COFTD[262] = -4.33159478E-07;
    COFTD[263] = 6.67640055E-11;
    COFTD[264] = -1.03383911E-01;
    COFTD[265] = 9.17368946E-04;
    COFTD[266] = -4.33506203E-07;
    COFTD[267] = 6.68476435E-11;
    COFTD[268] = -1.03451906E-01;
    COFTD[269] = 9.17972300E-04;
    COFTD[270] = -4.33791320E-07;
    COFTD[271] = 6.68916093E-11;
    COFTD[272] = -1.01770402E-01;
    COFTD[273] = 9.17667014E-04;
    COFTD[274] = -4.34133683E-07;
    COFTD[275] = 6.69899783E-11;
    COFTD[276] = -1.68032187E-01;
    COFTD[277] = 9.47207191E-04;
    COFTD[278] = -4.26734789E-07;
    COFTD[279] = 6.39101707E-11;
    COFTD[280] = -1.26505967E-01;
    COFTD[281] = 9.33839500E-04;
    COFTD[282] = -4.34686874E-07;
    COFTD[283] = 6.64184154E-11;
    COFTD[284] = -1.21244149E-01;
    COFTD[285] = 9.32247360E-04;
    COFTD[286] = -4.35568436E-07;
    COFTD[287] = 6.67044478E-11;
    COFTD[288] = -1.46424793E-01;
    COFTD[289] = 9.40060436E-04;
    COFTD[290] = -4.31101818E-07;
    COFTD[291] = 6.52733173E-11;
    COFTD[292] = -1.64313478E-01;
    COFTD[293] = 9.45982311E-04;
    COFTD[294] = -4.27543725E-07;
    COFTD[295] = 6.41575424E-11;
    COFTD[296] = -1.55196478E-01;
    COFTD[297] = 9.48793311E-04;
    COFTD[298] = -4.32429136E-07;
    COFTD[299] = 6.52268964E-11;
    COFTD[300] = -1.55228210E-01;
    COFTD[301] = 9.48987307E-04;
    COFTD[302] = -4.32517553E-07;
    COFTD[303] = 6.52402330E-11;
}

/* Replace this routine with the one generated by the Gauss Jordan solver of DW */
AMREX_GPU_HOST_DEVICE void sgjsolve(double* A, double* x, double* b) {
    amrex::Abort("sgjsolve not implemented, choose a different solver ");
}

/* Replace this routine with the one generated by the Gauss Jordan solver of DW */
AMREX_GPU_HOST_DEVICE void sgjsolve_simplified(double* A, double* x, double* b) {
    amrex::Abort("sgjsolve_simplified not implemented, choose a different solver ");
}

